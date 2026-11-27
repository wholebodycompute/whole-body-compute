import AppKit
import ApplicationServices
import CoreGraphics
import Foundation

/// Click-through, non-activating AppKit highlight halo for `body ears` live dictation.
/// Automatically frames the focused text input box (`AXTextField`, `AXTextArea`, `AXSearchField`,
/// `AXComboBox`, or editable web/Electron container) and falls back cleanly to the focused window
/// (or screen perimeter) when no distinct input box is exposed.
public final class EarsHighlightOverlay: @unchecked Sendable {
    public static let shared = EarsHighlightOverlay()

    public enum TargetKind: String, Sendable {
        case textBox = "text_box"
        case window = "window"
        case screen = "screen"
        case blocked = "blocked"
        case none = "none"
    }

    public struct StateSnapshot: Sendable {
        public var visible: Bool
        public var mode: String
        public var requestedTarget: String
        public var targetKind: TargetKind
        public var appName: String
        public var elementRole: String
        public var quartzBBox: (Double, Double, Double, Double)
    }

    private let stateLock = NSLock()
    private var state = StateSnapshot(
        visible: false,
        mode: "ptt",
        requestedTarget: "auto",
        targetKind: .none,
        appName: "",
        elementRole: "",
        quartzBBox: (0, 0, 0, 0)
    )
    private var sessionSeq: UInt64 = 0
    private var lastActivityTime: CFAbsoluteTime = 0

    // Main-thread UI state
    private var panel: NSPanel?
    private var highlightView: EarsHighlightBorderView?
    private var animTimer: Timer?
    private var trackTimer: Timer?
    private var watchdogTimer: Timer?
    private var killSwitchObserverID: UUID?

    private init() {}

    public func installKillSwitchObserver(killSwitch: KillSwitch = .shared) {
        guard killSwitchObserverID == nil else { return }
        killSwitchObserverID = killSwitch.addObserver { [weak self] snap in
            if snap.isPaused || snap.isEarsPaused {
                self?.hide()
            }
        }
    }

    public func snapshot() -> StateSnapshot {
        stateLock.lock()
        defer { stateLock.unlock() }
        return state
    }

    /// Handle `{"op": "ears_highlight", "action": "show"|"update"|"hide"|"status", ...}` from `SocketServer`.
    public func handleSocketRequest(
        _ req: [String: Any],
        killSwitch: KillSwitch = .shared,
        headlessTestMode: Bool = false
    ) -> [String: Any] {
        let action = ((req["action"] as? String) ?? "show").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let mode = ((req["mode"] as? String) ?? "ptt").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let requestedTarget = ((req["target"] as? String) ?? "auto").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let maxDuration: Double = {
            if let d = req["max_duration_s"] as? Double { return max(0.5, min(d, 300.0)) }
            if let n = req["max_duration_s"] as? NSNumber { return max(0.5, min(n.doubleValue, 300.0)) }
            return 95.0
        }()

        let isHeadless = headlessTestMode || ProcessInfo.processInfo.environment["WBC_EYES_HEADLESS"] == "1"

        if action == "status" {
            return formatResponse(snapshot())
        }

        if action == "hide" {
            hide(isHeadless: isHeadless)
            return formatResponse(snapshot())
        }

        if killSwitch.isPaused || killSwitch.isEarsPaused {
            hide(isHeadless: isHeadless)
            return [
                "status": "error",
                "paused": killSwitch.isPaused,
                "ears_paused": killSwitch.isEarsPaused,
                "kill_switch": true,
                "visible": false,
                "error": "Whole Body Compute Kill Switch is active (Paused)."
            ]
        }

        if isHeadless {
            let mockKind: TargetKind = (requestedTarget == "window") ? .window : .textBox
            let mockRole = (mockKind == .window) ? "AXWindow" : "AXTextArea"
            let mockBox = (mockKind == .window) ? (100.0, 100.0, 900.0, 640.0) : (220.0, 520.0, 560.0, 56.0)
            stateLock.lock()
            sessionSeq &+= 1
            lastActivityTime = CFAbsoluteTimeGetCurrent()
            state = StateSnapshot(
                visible: true,
                mode: mode,
                requestedTarget: requestedTarget,
                targetKind: mockKind,
                appName: "HeadlessMockApp",
                elementRole: mockRole,
                quartzBBox: mockBox
            )
            let snap = state
            stateLock.unlock()
            return formatResponse(snap)
        }

        let resolved = Self.resolveTargetGeometry(requestedTarget: requestedTarget, killSwitch: killSwitch)
        if resolved.kind == .blocked {
            hide(isHeadless: false)
            return [
                "status": "error",
                "security_blocked": true,
                "visible": false,
                "target_kind": TargetKind.blocked.rawValue,
                "error": "Highlight suppressed on protected security surface."
            ]
        }

        stateLock.lock()
        sessionSeq &+= 1
        let seq = sessionSeq
        lastActivityTime = CFAbsoluteTimeGetCurrent()
        state = StateSnapshot(
            visible: true,
            mode: mode,
            requestedTarget: requestedTarget,
            targetKind: resolved.kind,
            appName: resolved.appName,
            elementRole: resolved.role,
            quartzBBox: (
                Double(resolved.quartzRect.origin.x),
                Double(resolved.quartzRect.origin.y),
                Double(resolved.quartzRect.size.width),
                Double(resolved.quartzRect.size.height)
            )
        )
        let snap = state
        stateLock.unlock()

        DispatchQueue.main.async { [weak self] in
            self?.presentOnMain(
                seq: seq,
                mode: mode,
                requestedTarget: requestedTarget,
                resolved: resolved,
                maxDuration: maxDuration,
                killSwitch: killSwitch
            )
        }

        return formatResponse(snap)
    }

    /// Pulse the border aura when live partial transcription characters are injected.
    public func notifyTypingActivity(headlessTestMode: Bool = false) {
        let isHeadless = headlessTestMode || ProcessInfo.processInfo.environment["WBC_EYES_HEADLESS"] == "1"
        stateLock.lock()
        let isVis = state.visible
        if isVis {
            lastActivityTime = CFAbsoluteTimeGetCurrent()
        }
        stateLock.unlock()
        guard isVis, !isHeadless else { return }

        DispatchQueue.main.async { [weak self] in
            guard let self, let view = self.highlightView else { return }
            view.triggerSpeechBoost()
            self.scheduleWatchdogOnMain(maxDuration: 95.0)
        }
    }

    public func hide(isHeadless: Bool = false) {
        stateLock.lock()
        sessionSeq &+= 1
        state.visible = false
        stateLock.unlock()

        guard !isHeadless && ProcessInfo.processInfo.environment["WBC_EYES_HEADLESS"] != "1" else { return }
        DispatchQueue.main.async { [weak self] in
            self?.teardownOnMain()
        }
    }

    private func formatResponse(_ snap: StateSnapshot) -> [String: Any] {
        let (x, y, w, h) = snap.quartzBBox
        return [
            "status": "ok",
            "visible": snap.visible,
            "mode": snap.mode,
            "target": snap.requestedTarget,
            "target_kind": snap.targetKind.rawValue,
            "app_name": snap.appName,
            "element_role": snap.elementRole,
            "bbox": [x, y, w, h]
        ]
    }

    // MARK: - Main-Thread Presentation & Tracking

    private func presentOnMain(
        seq: UInt64,
        mode: String,
        requestedTarget: String,
        resolved: ResolvedGeometry,
        maxDuration: Double,
        killSwitch: KillSwitch
    ) {
        stateLock.lock()
        let stillActive = state.visible && sessionSeq == seq
        stateLock.unlock()
        guard stillActive else { return }

        ensurePanelOnMain()
        guard let panel, let highlightView else { return }

        applyGeometryOnMain(panel: panel, view: highlightView, mode: mode, resolved: resolved)
        panel.orderFrontRegardless()
        panel.alphaValue = 1.0

        startTimersOnMain(
            seq: seq,
            requestedTarget: requestedTarget,
            maxDuration: maxDuration,
            killSwitch: killSwitch
        )
    }

    private func ensurePanelOnMain() {
        if panel != nil && highlightView != nil { return }
        let initialRect = NSRect(x: 0, y: 0, width: 320, height: 80)
        let p = NSPanel(
            contentRect: initialRect,
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        p.isOpaque = false
        p.backgroundColor = .clear
        p.ignoresMouseEvents = true
        p.hasShadow = false
        p.hidesOnDeactivate = false
        p.isFloatingPanel = true
        p.level = NSWindow.Level(rawValue: NSWindow.Level.statusBar.rawValue + 2)
        p.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary, .ignoresCycle]

        let v = EarsHighlightBorderView(frame: NSRect(origin: .zero, size: initialRect.size))
        p.contentView = v
        p.alphaValue = 0.0
        p.orderOut(nil)

        self.panel = p
        self.highlightView = v
    }

    private func applyGeometryOnMain(
        panel: NSPanel,
        view: EarsHighlightBorderView,
        mode: String,
        resolved: ResolvedGeometry
    ) {
        let cocoaTarget = Self.quartzToCocoa(resolved.quartzRect)
        let outset: CGFloat = (resolved.kind == .textBox) ? 4.0 : (resolved.kind == .window ? 3.0 : -3.0)
        let framedBox = cocoaTarget.insetBy(dx: -outset, dy: -outset)
        let margin: CGFloat = 14.0

        let panelFrame = NSRect(
            x: framedBox.minX - margin,
            y: framedBox.minY - margin,
            width: max(48.0, framedBox.width + margin * 2.0),
            height: max(36.0, framedBox.height + margin * 2.0)
        )

        let localBox = NSRect(
            x: margin,
            y: margin,
            width: max(20.0, framedBox.width),
            height: max(16.0, framedBox.height)
        )

        if panel.frame != panelFrame {
            panel.setFrame(panelFrame, display: true)
            view.frame = NSRect(origin: .zero, size: panelFrame.size)
        }
        view.configure(
            boxRect: localBox,
            mode: mode,
            targetKind: resolved.kind
        )
    }

    private func startTimersOnMain(
        seq: UInt64,
        requestedTarget: String,
        maxDuration: Double,
        killSwitch: KillSwitch
    ) {
        scheduleWatchdogOnMain(maxDuration: maxDuration)

        if animTimer == nil {
            animTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 24.0, repeats: true) { [weak self] _ in
                guard let self else { return }
                self.stateLock.lock()
                let vis = self.state.visible
                self.stateLock.unlock()
                guard vis else {
                    self.teardownOnMain()
                    return
                }
                self.highlightView?.stepAnimation()
            }
        }

        trackTimer?.invalidate()
        trackTimer = Timer.scheduledTimer(withTimeInterval: 0.12, repeats: true) { [weak self] _ in
            guard let self else { return }
            self.stateLock.lock()
            let vis = self.state.visible && self.sessionSeq == seq
            let currentMode = self.state.mode
            self.stateLock.unlock()
            guard vis else { return }

            if killSwitch.isPaused || killSwitch.isEarsPaused {
                self.hide()
                return
            }

            let updated = Self.resolveTargetGeometry(requestedTarget: requestedTarget, killSwitch: killSwitch)
            if updated.kind == .blocked {
                self.hide()
                return
            }

            self.stateLock.lock()
            if self.state.visible && self.sessionSeq == seq {
                self.state.targetKind = updated.kind
                self.state.appName = updated.appName
                self.state.elementRole = updated.role
                self.state.quartzBBox = (
                    Double(updated.quartzRect.origin.x),
                    Double(updated.quartzRect.origin.y),
                    Double(updated.quartzRect.size.width),
                    Double(updated.quartzRect.size.height)
                )
            }
            self.stateLock.unlock()

            if let panel = self.panel, let view = self.highlightView {
                self.applyGeometryOnMain(panel: panel, view: view, mode: currentMode, resolved: updated)
            }
        }
    }

    private func scheduleWatchdogOnMain(maxDuration: Double) {
        watchdogTimer?.invalidate()
        watchdogTimer = Timer.scheduledTimer(withTimeInterval: maxDuration, repeats: false) { [weak self] _ in
            self?.hide()
        }
    }

    private func teardownOnMain() {
        animTimer?.invalidate()
        animTimer = nil
        trackTimer?.invalidate()
        trackTimer = nil
        watchdogTimer?.invalidate()
        watchdogTimer = nil
        if let panel {
            panel.alphaValue = 0.0
            panel.orderOut(nil)
        }
    }

    // MARK: - AX Target Resolution (Text Box -> Focused Window -> Screen)

    private struct ResolvedGeometry {
        let kind: TargetKind
        let quartzRect: CGRect
        let appName: String
        let role: String
    }

    private static let textBoxRoles: Set<String> = [
        "AXTextField",
        "AXTextArea",
        "AXSearchField",
        "AXComboBox"
    ]

    private static let unlockedPidsLock = NSLock()
    private static var unlockedPids: Set<pid_t> = []

    private static func unlockManualAXOnce(pid: pid_t, axApp: AXUIElement) {
        unlockedPidsLock.lock()
        let alreadyUnlocked = unlockedPids.contains(pid)
        if !alreadyUnlocked {
            unlockedPids.insert(pid)
        }
        unlockedPidsLock.unlock()
        if !alreadyUnlocked {
            // Set AXManualAccessibility once per PID (never AXEnhancedUserInterface, which triggers
            // Chromium screen-reader selection behavior if polled repeatedly while typing).
            _ = AXUIElementSetAttributeValue(axApp, "AXManualAccessibility" as CFString, kCFBooleanTrue)
        }
    }

    private static func resolveTargetGeometry(
        requestedTarget: String,
        killSwitch: KillSwitch
    ) -> ResolvedGeometry {
        let fallbackScreen = primaryQuartzScreenRect()
        guard let front = NSWorkspace.shared.frontmostApplication else {
            return ResolvedGeometry(kind: .screen, quartzRect: fallbackScreen, appName: "", role: "AXScreen")
        }

        let appName = front.localizedName ?? ""
        let bundleID = front.bundleIdentifier ?? ""
        if PrivacyBlindfold.checkApplication(appName: appName, bundleID: bundleID).blocked {
            return ResolvedGeometry(kind: .blocked, quartzRect: .zero, appName: appName, role: "BLOCKED")
        }

        guard AXIsProcessTrusted() else {
            return ResolvedGeometry(kind: .screen, quartzRect: fallbackScreen, appName: appName, role: "AXScreen")
        }

        let axApp = AXUIElementCreateApplication(front.processIdentifier)
        unlockManualAXOnce(pid: front.processIdentifier, axApp: axApp)

        let focusedWin = copyAXElement(axApp, "AXFocusedWindow")
            ?? copyAXElement(axApp, "AXMainWindow")
            ?? copyFirstAXWindow(axApp)

        var winRect: CGRect?
        if let w = focusedWin {
            let winTitle = copyAXString(w, "AXTitle")
            let docURL = copyAXString(w, "AXDocument")
            if PrivacyBlindfold.checkWindowOrigin(windowTitle: winTitle, documentURL: docURL, webAreaURL: "").blocked {
                return ResolvedGeometry(kind: .blocked, quartzRect: .zero, appName: appName, role: "BLOCKED")
            }
            if let r = copyAXRect(w), r.width >= 60, r.height >= 45 {
                winRect = r
            }
        }

        if requestedTarget != "window" {
            let focusedElem = copyAXElement(axApp, "AXFocusedUIElement")
                ?? focusedWin.flatMap { copyAXElement($0, "AXFocusedUIElement") }

            if let focusedElem {
                if let candidate = resolveFocusedInputBox(from: focusedElem, windowRect: winRect) {
                    switch candidate {
                    case .secureBlocked:
                        return ResolvedGeometry(kind: .blocked, quartzRect: .zero, appName: appName, role: "AXSecureTextField")
                    case let .box(rect, role):
                        return ResolvedGeometry(kind: .textBox, quartzRect: rect, appName: appName, role: role)
                    }
                }
            }
        }

        if let winRect {
            return ResolvedGeometry(kind: .window, quartzRect: winRect, appName: appName, role: "AXWindow")
        }

        return ResolvedGeometry(kind: .screen, quartzRect: fallbackScreen, appName: appName, role: "AXScreen")
    }

    private enum FocusedBoxCandidate {
        case secureBlocked
        case box(CGRect, String)
    }

    private static func resolveFocusedInputBox(
        from focusedElem: AXUIElement,
        windowRect: CGRect?
    ) -> FocusedBoxCandidate? {
        // Walk self + up to 6 ancestors to catch inner line/paragraph nodes inside contenteditable/textarea
        // and to clip scrolled AXTextAreas to their visible AXScrollArea viewport.
        struct AncestorNode {
            let role: String
            let subrole: String?
            let editable: Bool
            let rect: CGRect?
        }

        var chain: [AncestorNode] = []
        var curr: AXUIElement? = focusedElem
        for _ in 0..<7 {
            guard let el = curr else { break }
            let role = copyAXString(el, "AXRole")
            let subRaw = copyAXString(el, "AXSubrole")
            let subrole = subRaw.isEmpty ? nil : subRaw
            let title = copyAXString(el, "AXTitle")
            if PrivacyBlindfold.isSecureElement(role: role, subrole: subrole, title: title) {
                return .secureBlocked
            }
            if role == "AXWindow" || role == "AXApplication" {
                break
            }
            let editable = copyAXBool(el, "AXEditable") ?? false
            let rect = copyAXRect(el)
            chain.append(AncestorNode(role: role, subrole: subrole, editable: editable, rect: rect))
            curr = copyAXElement(el, "AXParent")
        }

        let winArea = (windowRect != nil) ? (windowRect!.width * windowRect!.height) : 0.0

        // 1. Prefer the outermost explicit text-input container (e.g. composer AXTextArea wrapping an AXGroup line)
        var chosenIdx: Int?
        for (idx, node) in chain.enumerated() {
            guard textBoxRoles.contains(node.role),
                  let r = node.rect,
                  r.width >= 36.0,
                  r.height >= 16.0
            else { continue }
            if winArea > 0, (r.width * r.height) >= winArea * 0.84 {
                // Element covers almost the entire window (e.g. Terminal), let caller highlight the window instead
                continue
            }
            chosenIdx = idx
        }

        // 2. Fallback: web/Electron editable group (`contenteditable="true"` without explicit AXTextArea role)
        if chosenIdx == nil {
            for (idx, node) in chain.enumerated() {
                guard (node.editable || node.subrole == "AXSearchField" || node.subrole == "AXTextInput"),
                      let r = node.rect,
                      r.width >= 36.0,
                      r.height >= 16.0
                else { continue }
                if winArea > 0, (r.width * r.height) >= winArea * 0.82 {
                    continue
                }
                chosenIdx = idx
                break
            }
        }

        guard let idx = chosenIdx, var targetRect = chain[idx].rect else {
            return nil
        }
        let chosenRole = chain[idx].role

        // 3. Clip tall scrolled AXTextAreas to their enclosing AXScrollArea so border stays on the visible box
        if idx + 1 < chain.count {
            for anc in chain[(idx + 1)...] where anc.role == "AXScrollArea" {
                if let sr = anc.rect, sr.width >= 36.0, sr.height >= 18.0 {
                    let clipped = targetRect.intersection(sr)
                    if !clipped.isNull && clipped.width >= 32.0 && clipped.height >= 16.0 {
                        // Use AXScrollArea width if close so border hugs the visible input bezel cleanly
                        if abs(sr.width - targetRect.width) <= 36.0 && sr.height <= targetRect.height + 16.0 {
                            targetRect = sr
                        } else {
                            targetRect = clipped
                        }
                    }
                    break
                }
            }
        }

        // 4. Clip to focused window bounds if partially scrolled near window edge
        if let windowRect {
            let clipped = targetRect.intersection(windowRect)
            if !clipped.isNull, clipped.width >= 32.0, clipped.height >= 16.0 {
                targetRect = clipped
            } else {
                return nil
            }
            if (targetRect.width * targetRect.height) >= winArea * 0.84 {
                return nil
            }
        }

        return .box(targetRect, chosenRole)
    }

    // MARK: - Coordinate & AX Primitives

    private static func primaryQuartzScreenRect() -> CGRect {
        if let first = NSScreen.screens.first {
            return CGRect(origin: .zero, size: first.frame.size)
        }
        return CGRect(x: 0, y: 0, width: 1440, height: 900)
    }

    private static func quartzToCocoa(_ qRect: CGRect) -> NSRect {
        let primaryH = NSScreen.screens.first?.frame.height
            ?? NSScreen.main?.frame.height
            ?? 900.0
        let cocoaY = primaryH - (qRect.origin.y + qRect.size.height)
        return NSRect(
            x: qRect.origin.x,
            y: cocoaY,
            width: qRect.size.width,
            height: qRect.size.height
        )
    }

    private static func copyAXRaw(_ elem: AXUIElement, _ attr: String) -> CFTypeRef? {
        var val: CFTypeRef?
        let err = AXUIElementCopyAttributeValue(elem, attr as CFString, &val)
        return err == .success ? val : nil
    }

    private static func copyAXElement(_ elem: AXUIElement, _ attr: String) -> AXUIElement? {
        guard let raw = copyAXRaw(elem, attr), CFGetTypeID(raw) == AXUIElementGetTypeID() else {
            return nil
        }
        return unsafeBitCast(raw, to: AXUIElement.self)
    }

    private static func copyFirstAXWindow(_ axApp: AXUIElement) -> AXUIElement? {
        guard let raw = copyAXRaw(axApp, "AXWindows"), let arr = raw as? [AnyObject] else {
            return nil
        }
        for item in arr where CFGetTypeID(item) == AXUIElementGetTypeID() {
            return unsafeBitCast(item, to: AXUIElement.self)
        }
        return nil
    }

    private static func copyAXString(_ elem: AXUIElement, _ attr: String) -> String {
        guard let raw = copyAXRaw(elem, attr) else { return "" }
        if let s = raw as? String { return s.trimmingCharacters(in: .whitespacesAndNewlines) }
        if let u = raw as? URL { return u.absoluteString }
        return ""
    }

    private static func copyAXBool(_ elem: AXUIElement, _ attr: String) -> Bool? {
        guard let raw = copyAXRaw(elem, attr) else { return nil }
        if let b = raw as? Bool { return b }
        if let n = raw as? NSNumber { return n.boolValue }
        return nil
    }

    private static func copyAXRect(_ elem: AXUIElement) -> CGRect? {
        var pt = CGPoint.zero
        var sz = CGSize.zero
        guard let posRaw = copyAXRaw(elem, "AXPosition"),
              CFGetTypeID(posRaw) == AXValueGetTypeID(),
              let szRaw = copyAXRaw(elem, "AXSize"),
              CFGetTypeID(szRaw) == AXValueGetTypeID()
        else {
            return nil
        }
        let posVal = unsafeBitCast(posRaw, to: AXValue.self)
        let szVal = unsafeBitCast(szRaw, to: AXValue.self)
        guard AXValueGetValue(posVal, .cgPoint, &pt),
              AXValueGetValue(szVal, .cgSize, &sz),
              sz.width > 1.0,
              sz.height > 1.0,
              pt.x.isFinite,
              pt.y.isFinite,
              sz.width.isFinite,
              sz.height.isFinite
        else {
            return nil
        }
        return CGRect(origin: pt, size: sz)
    }
}

// MARK: - Custom Hardware-Accelerated Border View

private final class EarsHighlightBorderView: NSView {
    private var boxRect: NSRect = .zero
    private var mode: String = "ptt"
    private var targetKind: EarsHighlightOverlay.TargetKind = .textBox
    private var phase: Double = 0.0
    private var speechBoost: CGFloat = 0.0

    func configure(
        boxRect: NSRect,
        mode: String,
        targetKind: EarsHighlightOverlay.TargetKind
    ) {
        self.boxRect = boxRect
        self.mode = mode
        self.targetKind = targetKind
        needsDisplay = true
    }

    func stepAnimation() {
        phase += 0.18
        if speechBoost > 0.01 {
            speechBoost *= 0.82
        } else {
            speechBoost = 0.0
        }
        needsDisplay = true
    }

    func triggerSpeechBoost() {
        speechBoost = 1.0
        needsDisplay = true
    }

    override func draw(_ dirtyRect: NSRect) {
        guard boxRect.width > 8, boxRect.height > 8 else { return }

        // Canonical Whole Body Compute Electric Red (#FF2A2A) border outline only (zero badges or interior wash)
        let accentColor: NSColor
        switch mode {
        case "hands_free", "opt+space":
            accentColor = NSColor(calibratedRed: 1.0, green: 0.24, blue: 0.14, alpha: 1.0)
        default:
            accentColor = NSColor(calibratedRed: 1.0, green: 0.165, blue: 0.165, alpha: 1.0) // #FF2A2A
        }

        let pulse = CGFloat((sin(phase) + 1.0) * 0.5) // 0...1
        let cornerRadius: CGFloat = (targetKind == .textBox)
            ? min(11.0, max(7.0, boxRect.height * 0.22))
            : 13.0

        // 1. Soft breathing outer glow
        let outerGlowWidth: CGFloat = (targetKind == .textBox ? 8.5 : 10.5) + speechBoost * 3.0
        let outerAlpha: CGFloat = min(0.52, (0.15 + pulse * 0.16) + speechBoost * 0.18)
        let outerPath = NSBezierPath(roundedRect: boxRect, xRadius: cornerRadius, yRadius: cornerRadius)
        outerPath.lineWidth = outerGlowWidth
        accentColor.withAlphaComponent(outerAlpha).setStroke()
        outerPath.stroke()

        // 2. Crisp inner glow halo
        let midPath = NSBezierPath(roundedRect: boxRect, xRadius: cornerRadius, yRadius: cornerRadius)
        midPath.lineWidth = (targetKind == .textBox ? 4.8 : 6.0) + speechBoost * 1.2
        accentColor.withAlphaComponent(min(0.68, 0.34 + pulse * 0.14 + speechBoost * 0.16)).setStroke()
        midPath.stroke()

        // 3. Sharp core border stroke (zero interior fill so text box stays 100% legible)
        let corePath = NSBezierPath(roundedRect: boxRect, xRadius: cornerRadius, yRadius: cornerRadius)
        corePath.lineWidth = (targetKind == .textBox) ? 2.5 : 3.25
        accentColor.withAlphaComponent(0.97).setStroke()
        corePath.stroke()
    }
}
