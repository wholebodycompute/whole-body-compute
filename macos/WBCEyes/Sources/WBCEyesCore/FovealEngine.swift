import AppKit
import ApplicationServices
import CoreGraphics
import Foundation

@_silgen_name("CGEventSourceSecondsSinceLastEventType")
private func wbc_CGEventSourceSecondsSinceLastEventType(_ stateID: Int32, _ eventType: UInt32) -> Double

public struct FovealElementRecord {
    public var id: String
    public var role: String
    public var title: String
    public var label: String
    public var description: String
    public var value: String?
    public var bbox: (Double, Double, Double, Double)
    public var center: (Double, Double)
    public var focused: Bool
    public var enabled: Bool
    public var subrole: String?
    public var identifier: String?
    public var depth: Int
    public var isSecure: Bool
    public var rawElement: AXUIElement?

    public func toDictionary() -> [String: Any] {
        let cleanLbl = PrivacyBlindfold.sanitizeDisplayString(title.isEmpty ? label : title, maxLength: 80)
        let cleanDesc = PrivacyBlindfold.sanitizeDisplayString(description, maxLength: 80)
        let cleanVal: Any = value != nil ? PrivacyBlindfold.sanitizeDisplayString(value, maxLength: 80) : NSNull()
        let cleanIdent: Any = identifier != nil ? PrivacyBlindfold.sanitizeDisplayString(identifier, maxLength: 80) : NSNull()
        let (x, y, w, h) = (
            Self.round1(bbox.0),
            Self.round1(bbox.1),
            Self.round1(bbox.2),
            Self.round1(bbox.3)
        )
        let (cx, cy) = (Self.round1(center.0), Self.round1(center.1))
        return [
            "id": id,
            "role": role,
            "title": cleanLbl,
            "label": cleanLbl,
            "description": cleanDesc,
            "value": cleanVal,
            "position": [x, y],
            "size": [w, h],
            "center": [cx, cy],
            "bounds": ["x": x, "y": y, "w": w, "h": h],
            "bbox": [x, y, w, h],
            "focused": focused,
            "enabled": enabled,
            "subrole": subrole ?? NSNull(),
            "identifier": cleanIdent,
            "depth": depth,
            "is_secure": isSecure
        ]
    }

    private static func round1(_ v: Double) -> Double {
        (v * 10.0).rounded() / 10.0
    }
}

public struct FovealSnapshotRecord {
    public var appName: String
    public var bundleID: String
    public var pid: Int
    public var windowTitle: String
    public var windowBBox: (Double, Double, Double, Double)
    public var focusedElement: FovealElementRecord?
    public var interactiveElements: [FovealElementRecord]
    public var timestamp: Double
    public var latencyMs: Double
    public var headlessRoute: String
    public var filterQuery: String?
    public var actuation: [String: Any]?
    public var tactileDelta: [String: Any]?
    public var lastMessage: String?
    public var lastMessages: [String]?
    public var replySettled: Bool?
    public var waitReplyMs: Double?

    public func toDictionary() -> [String: Any] {
        let elemsData = interactiveElements.map { $0.toDictionary() }
        let estTokens = max(25, interactiveElements.count * 12 + 35)
        let baseline = 15000
        let savingsPct = max(0.0, ((1.0 - (Double(estTokens) / Double(baseline))) * 1000.0).rounded() / 10.0)
        return [
            "app_name": appName,
            "bundle_id": bundleID,
            "pid": pid,
            "window_title": windowTitle,
            "window_bbox": [windowBBox.0, windowBBox.1, windowBBox.2, windowBBox.3],
            "latency_ms": (latencyMs * 1000.0).rounded() / 1000.0,
            "element_count": interactiveElements.count,
            "estimated_tokens": estTokens,
            "screenshot_token_baseline": baseline,
            "token_savings_pct": savingsPct,
            "elements": elemsData,
            "interactive_elements": elemsData,
            "focused_element": focusedElement?.toDictionary() ?? NSNull(),
            "timestamp": (timestamp * 1000.0).rounded() / 1000.0,
            "headless_route": headlessRoute,
            "frame_path": NSNull(),
            "latest_frame_path": NSNull(),
            "actuation": actuation ?? NSNull(),
            "tactile_delta": tactileDelta ?? NSNull(),
            "last_message": lastMessage ?? NSNull(),
            "last_messages": lastMessages ?? NSNull(),
            "reply_settled": replySettled ?? NSNull(),
            "wait_reply_ms": waitReplyMs ?? NSNull(),
            "helper_provider": "Whole Body Compute.app"
        ]
    }
}

/// Native Swift Foveal Vision (`AXUIElement`) & Tactile Motor (`CGEvent`) Engine.
public enum FovealEngine {

    private static let interactiveAXRoles: Set<String> = [
        "AXButton", "AXTextField", "AXTextArea", "AXCheckBox", "AXRadioButton",
        "AXPopUpButton", "AXMenuButton", "AXLink", "AXTabGroup", "AXTab",
        "AXComboBox", "AXSearchField", "AXSlider", "AXMenuItem", "AXMenuBarItem",
        "AXStaticText", "AXHeading", "AXCell", "AXWebArea"
    ]

    private static let ignoreTextSubroles: Set<String> = [
        "AXCloseButton", "AXMinimizeButton", "AXFullScreenButton", "AXZoomButton"
    ]

    private static let quartzKeycodes: [String: CGKeyCode] = [
        "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9,
        "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17, "1": 18, "2": 19,
        "3": 20, "4": 21, "6": 22, "5": 23, "equal": 24, "=": 24, "9": 25, "7": 26,
        "minus": 27, "-": 27, "8": 28, "0": 29, "rightbracket": 30, "]": 30, "o": 31,
        "u": 32, "leftbracket": 33, "[": 33, "i": 34, "p": 35, "return": 36, "enter": 36,
        "l": 37, "j": 38, "quote": 39, "'": 39, "k": 40, "semicolon": 41, ";": 41,
        "backslash": 42, "\\": 42, "comma": 43, ",": 43, "slash": 44, "/": 44, "n": 45,
        "m": 46, "period": 47, ".": 47, "grave": 50, "`": 50, "tab": 48, "space": 49,
        "delete": 51, "backspace": 51, "escape": 53, "esc": 53, "left": 123, "right": 124,
        "down": 125, "up": 126
    ]

    private static let quartzModifiers: [String: CGEventFlags] = [
        "cmd": .maskCommand,
        "command": .maskCommand,
        "shift": .maskShift,
        "alt": .maskAlternate,
        "option": .maskAlternate,
        "opt": .maskAlternate,
        "ctrl": .maskControl,
        "control": .maskControl
    ]

    // MARK: - Top-level Request Entrypoint

    public static func handleRequest(
        _ req: [String: Any],
        killSwitch: KillSwitch = .shared,
        headlessTestMode: Bool = false
    ) -> [String: Any] {
        if killSwitch.isPaused {
            return killSwitchErrorPayload()
        }

        let t0 = CFAbsoluteTimeGetCurrent()
        let pidArg = (req["pid"] as? Int) ?? (req["pid"] as? NSNumber)?.intValue
        let appNameArg = req["app_name"] as? String
        let titleQuery = req["window_title_query"] as? String
        let filterQuery = req["filter_query"] as? String
        let includeFloating = Self.boolValue(req["include_floating"])
        let includeStatusBar = Self.boolValue(req["include_status_bar"])
        let readLast = Self.boolValue(req["read_last"])
        let submit = Self.boolValue(req["submit"])
        let allowFocusSteal = Self.boolValue(req["allow_focus_steal"])
        let noCursor = Self.boolValue(req["no_cursor"])
            || ["1", "true", "yes"].contains(
                (ProcessInfo.processInfo.environment["WBC_HANDS_NO_CURSOR"] ?? "").lowercased()
            )

        let click = Self.nonEmptyString(req["click"])
        let clickXY = Self.nonEmptyString(req["click_xy"])
        let hover = Self.nonEmptyString(req["hover"])
        let rightClick = Self.nonEmptyString(req["right_click"])
        let attach = Self.nonEmptyString(req["attach"])
        let chainRaw = req["chain"]
        let hasChain = (chainRaw as? String)?.isEmpty == false || (chainRaw as? [Any])?.isEmpty == false
        let typeText = req["type_text"] as? String
        let pasteText = req["paste_text"] as? String
        let scroll = Self.nonEmptyString(req["scroll"])
        let hotkey = Self.nonEmptyString(req["hotkey"])
        let menu = Self.nonEmptyString(req["menu"])
        let waitReply: Double? = {
            if let d = req["wait_reply"] as? Double { return d }
            if let n = req["wait_reply"] as? NSNumber, !(req["wait_reply"] is Bool) { return n.doubleValue }
            return nil
        }()

        let hasActuation = (
            click != nil || clickXY != nil || hover != nil || rightClick != nil
            || attach != nil || hasChain || typeText != nil || pasteText != nil
            || scroll != nil || hotkey != nil || menu != nil || submit
        )

        var maxElements = (req["max_elements"] as? Int) ?? (req["max_elements"] as? NSNumber)?.intValue
        if let current = maxElements {
            if (readLast || waitReply != nil || pasteText != nil || typeText != nil || hasChain || attach != nil), current < 250 {
                maxElements = 250
            }
        } else {
            if readLast || waitReply != nil || pasteText != nil || typeText != nil || hasChain || attach != nil {
                maxElements = 250
            } else if click != nil || clickXY != nil || hover != nil || rightClick != nil {
                maxElements = 150
            } else if appNameArg != nil || pidArg != nil || titleQuery != nil || filterQuery != nil || includeFloating || includeStatusBar {
                maxElements = 40
            } else {
                maxElements = 10
            }
        }
        let resolvedMax = max(1, maxElements ?? 40)

        // Headless deterministic fixture mode for unit tests (never active in production launches)
        if headlessTestMode,
           ProcessInfo.processInfo.environment["WBC_EYES_MOCK_DOM"] == "1"
        {
            return handleMockDOMRequest(
                req: req,
                appNameArg: appNameArg,
                titleQuery: titleQuery,
                filterQuery: filterQuery,
                maxElements: resolvedMax,
                hasActuation: hasActuation,
                click: click,
                typeText: typeText,
                pasteText: pasteText,
                readLast: readLast,
                waitReply: waitReply,
                killSwitch: killSwitch,
                t0: t0
            )
        }

        var snap = captureSnapshot(
            pid: pidArg,
            appName: appNameArg,
            windowTitleQuery: titleQuery,
            maxElements: resolvedMax,
            filterQuery: filterQuery,
            includeFloating: includeFloating,
            includeStatusBar: includeStatusBar,
            killSwitch: killSwitch
        )

        if killSwitch.isPaused {
            return killSwitchErrorPayload()
        }

        if hasActuation {
            let actRes = actuate(
                click: click,
                clickXY: clickXY,
                hover: hover,
                rightClick: rightClick,
                attach: attach,
                chainRaw: chainRaw,
                typeText: typeText,
                pasteText: pasteText,
                submit: submit,
                scroll: scroll,
                hotkey: hotkey,
                menu: menu,
                snapshot: snap,
                appName: snap.appName.isEmpty ? appNameArg : snap.appName,
                pid: snap.pid > 0 ? snap.pid : pidArg,
                windowTitle: titleQuery ?? snap.windowTitle,
                allowFocusSteal: allowFocusSteal,
                noCursor: noCursor,
                killSwitch: killSwitch
            )
            if (actRes["kill_switch"] as? Bool) == true {
                return killSwitchErrorPayload()
            }
            snap.actuation = actRes
            snap.tactileDelta = actRes["tactile_delta"] as? [String: Any]
            snap.latencyMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
        }

        if let waitSec = waitReply {
            let replyRes = waitForReply(
                appName: snap.appName,
                pid: snap.pid > 0 ? snap.pid : pidArg,
                windowTitle: snap.windowTitle,
                timeoutSec: waitSec,
                preSnapshot: snap.actuation != nil ? snap : nil,
                killSwitch: killSwitch
            )
            if (replyRes["kill_switch"] as? Bool) == true {
                return killSwitchErrorPayload()
            }
            snap.replySettled = replyRes["settled"] as? Bool
            snap.waitReplyMs = replyRes["elapsed_ms"] as? Double
            snap.lastMessages = replyRes["new_messages"] as? [String]
            snap.lastMessage = replyRes["last_message"] as? String
        } else if readLast {
            let msgs = extractTextBlocks(from: snap)
            snap.lastMessages = msgs
            snap.lastMessage = msgs.last ?? ""
        }

        return snap.toDictionary()
    }

    public static func killSwitchErrorPayload() -> [String: Any] {
        [
            "status": "error",
            "paused": true,
            "kill_switch": true,
            "error": "Whole Body Compute Kill Switch is active (Paused from macOS menu bar). Resume Eyes & Hands in the menu bar to allow screen inspection or actuation."
        ]
    }

    // MARK: - AX Snapshot Capture

    public static func captureSnapshot(
        pid: Int? = nil,
        appName: String? = nil,
        windowTitleQuery: String? = nil,
        maxElements: Int = 40,
        maxDepth: Int = 40,
        filterQuery: String? = nil,
        includeFloating: Bool = false,
        includeStatusBar: Bool = false,
        killSwitch: KillSwitch = .shared
    ) -> FovealSnapshotRecord {
        let t0 = CFAbsoluteTimeGetCurrent()
        let nowTs = Date().timeIntervalSince1970

        if killSwitch.isPaused {
            return FovealSnapshotRecord(
                appName: appName ?? "Paused",
                bundleID: "",
                pid: pid ?? 0,
                windowTitle: "[PAUSED_KILL_SWITCH]",
                windowBBox: (0, 0, 0, 0),
                focusedElement: nil,
                interactiveElements: [],
                timestamp: nowTs,
                latencyMs: 0.1,
                headlessRoute: "PAUSED: WBC Eyes Kill Switch is active.",
                filterQuery: filterQuery
            )
        }

        let (resolvedPid, resolvedApp, bundleID) = resolveTargetApp(pid: pid, appName: appName)
        guard let targetPid = resolvedPid, targetPid > 0 else {
            return FovealSnapshotRecord(
                appName: "Unknown",
                bundleID: "",
                pid: 0,
                windowTitle: "",
                windowBBox: (0, 0, 0, 0),
                focusedElement: nil,
                interactiveElements: [],
                timestamp: nowTs,
                latencyMs: (CFAbsoluteTimeGetCurrent() - t0) * 1000.0,
                headlessRoute: "",
                filterQuery: filterQuery
            )
        }

        // 1. Native Swift App-Level Blindfold (Password Vaults & System Settings / SecurityAgent)
        let appBlindfold = PrivacyBlindfold.checkApplication(appName: resolvedApp, bundleID: bundleID)
        if appBlindfold.blocked {
            return FovealSnapshotRecord(
                appName: resolvedApp,
                bundleID: bundleID,
                pid: Int(targetPid),
                windowTitle: appBlindfold.placeholderTitle,
                windowBBox: (0, 0, 0, 0),
                focusedElement: nil,
                interactiveElements: [],
                timestamp: nowTs,
                latencyMs: (CFAbsoluteTimeGetCurrent() - t0) * 1000.0,
                headlessRoute: appBlindfold.reason,
                filterQuery: filterQuery
            )
        }

        guard AXIsProcessTrusted() else {
            return FovealSnapshotRecord(
                appName: resolvedApp,
                bundleID: bundleID,
                pid: Int(targetPid),
                windowTitle: "",
                windowBBox: (0, 0, 0, 0),
                focusedElement: nil,
                interactiveElements: [],
                timestamp: nowTs,
                latencyMs: (CFAbsoluteTimeGetCurrent() - t0) * 1000.0,
                headlessRoute: "TCC_UNTRUSTED: Grant Accessibility permission solely to Whole Body Compute.app in System Settings > Privacy & Security > Accessibility.",
                filterQuery: filterQuery
            )
        }

        let axApp = AXUIElementCreateApplication(targetPid)
        unlockAccessibilityTree(axApp, killSwitch: killSwitch)

        let rawFocused = getAXElement(axApp, attribute: "AXFocusedUIElement", killSwitch: killSwitch)
        let focusedElem = rawFocused.flatMap { parseUIElement($0, depth: 0, isFocusedOverride: true, killSwitch: killSwitch) }

        var targetWins = findTargetWindows(
            axApp: axApp,
            titleQuery: windowTitleQuery,
            includeFloating: includeFloating,
            includeStatusBar: includeStatusBar,
            killSwitch: killSwitch
        )
        for w in targetWins {
            unlockAccessibilityTree(w, killSwitch: killSwitch)
        }

        var windowTitle = ""
        var docURL = ""
        var webAreaURL = ""
        var windowBBox = (0.0, 0.0, 0.0, 0.0)
        if let primaryWin = targetWins.first {
            windowTitle = getAXString(primaryWin, attribute: "AXTitle", killSwitch: killSwitch)
            docURL = getAXString(primaryWin, attribute: "AXDocument", killSwitch: killSwitch)
            webAreaURL = findPrimaryWebAreaURL(in: primaryWin, killSwitch: killSwitch)
            windowBBox = extractBBox(primaryWin, killSwitch: killSwitch)
        }

        // 2. Native Swift Sensitive Web-Origin Blindfold (AXTitle + AXDocument + AXWebArea.AXURL)
        let originBlindfold = PrivacyBlindfold.checkWindowOrigin(
            windowTitle: windowTitle,
            documentURL: docURL,
            webAreaURL: webAreaURL,
            extraOrigins: loadCustomBlindfoldOrigins()
        )
        if originBlindfold.blocked {
            return FovealSnapshotRecord(
                appName: resolvedApp,
                bundleID: bundleID,
                pid: Int(targetPid),
                windowTitle: originBlindfold.placeholderTitle,
                windowBBox: (0, 0, 0, 0),
                focusedElement: nil,
                interactiveElements: [],
                timestamp: nowTs,
                latencyMs: (CFAbsoluteTimeGetCurrent() - t0) * 1000.0,
                headlessRoute: originBlindfold.reason,
                filterQuery: filterQuery
            )
        }

        var elems = traverseWindows(
            targetWins,
            maxElements: maxElements,
            maxDepth: maxDepth,
            filterQuery: filterQuery,
            killSwitch: killSwitch
        )

        // Bounded 120ms Electron/Chromium lazy accessibility hydration retry
        if isEmptyOrOnlyWindowControls(elems), (CFAbsoluteTimeGetCurrent() - t0) < 0.45, !killSwitch.isPaused {
            Thread.sleep(forTimeInterval: 0.12)
            targetWins = findTargetWindows(
                axApp: axApp,
                titleQuery: windowTitleQuery,
                includeFloating: includeFloating,
                includeStatusBar: includeStatusBar,
                killSwitch: killSwitch
            )
            for w in targetWins {
                unlockAccessibilityTree(w, killSwitch: killSwitch)
            }
            let retryElems = traverseWindows(
                targetWins,
                maxElements: maxElements,
                maxDepth: maxDepth,
                filterQuery: filterQuery,
                killSwitch: killSwitch
            )
            if retryElems.count > elems.count || !isEmptyOrOnlyWindowControls(retryElems) {
                elems = retryElems
            }
        }

        if elems.isEmpty, !killSwitch.isPaused {
            var fallbackWins: [AXUIElement] = []
            if let extras = getAXElement(axApp, attribute: "AXExtrasMenuBar", killSwitch: killSwitch) {
                fallbackWins.append(extras)
            }
            if includeStatusBar, let mb = getAXElement(axApp, attribute: "AXMenuBar", killSwitch: killSwitch) {
                fallbackWins.append(mb)
            }
            for ch in getAXElementArray(axApp, attribute: "AXChildren", killSwitch: killSwitch) {
                fallbackWins.append(ch)
            }
            if !fallbackWins.isEmpty {
                elems = traverseWindows(
                    fallbackWins,
                    maxElements: maxElements,
                    maxDepth: maxDepth,
                    filterQuery: filterQuery,
                    killSwitch: killSwitch
                )
            }
        }

        for idx in elems.indices {
            elems[idx].id = "e\(idx + 1)"
        }

        return FovealSnapshotRecord(
            appName: resolvedApp,
            bundleID: bundleID,
            pid: Int(targetPid),
            windowTitle: windowTitle,
            windowBBox: windowBBox,
            focusedElement: focusedElem,
            interactiveElements: elems,
            timestamp: nowTs,
            latencyMs: (CFAbsoluteTimeGetCurrent() - t0) * 1000.0,
            headlessRoute: "",
            filterQuery: filterQuery
        )
    }

    // MARK: - Tactile Actuation (`body hands`)

    public static func actuate(
        click: String? = nil,
        clickXY: String? = nil,
        hover: String? = nil,
        rightClick: String? = nil,
        attach: String? = nil,
        chainRaw: Any? = nil,
        typeText: String? = nil,
        pasteText: String? = nil,
        submit: Bool = false,
        scroll: String? = nil,
        hotkey: String? = nil,
        menu: String? = nil,
        snapshot: FovealSnapshotRecord,
        appName: String?,
        pid: Int?,
        windowTitle: String?,
        allowFocusSteal: Bool = false,
        noCursor: Bool = false,
        killSwitch: KillSwitch = .shared
    ) -> [String: Any] {
        if killSwitch.isPaused {
            return killSwitchErrorPayload()
        }

        let effApp = appName ?? snapshot.appName
        let effWin = windowTitle ?? snapshot.windowTitle
        let effBundle = snapshot.bundleID
        let targetPid: pid_t? = (pid ?? snapshot.pid) > 0 ? pid_t(pid ?? snapshot.pid) : nil

        if let secErr = checkActuationSecurityGate(
            actionName: "actuate",
            appName: effApp,
            bundleID: effBundle,
            windowTitle: effWin,
            snapshot: snapshot,
            element: nil,
            forbidSecureField: (typeText != nil || pasteText != nil) && click == nil
        ) {
            return [
                "status": "error",
                "security_blocked": true,
                "actions": [secErr],
                "error": secErr["error"] ?? "Security blocked",
                "summary": secErr["summary"] ?? "Blocked",
                "tactile_delta": ["verified": false, "security_blocked": true],
                "post_snapshot": NSNull()
            ]
        }

        var actionsTaken: [[String: Any]] = []

        if let hover {
            actionsTaken.append(performHover(
                target: hover,
                snapshot: snapshot,
                pid: targetPid,
                allowFocusSteal: allowFocusSteal,
                noCursor: noCursor,
                killSwitch: killSwitch
            ))
        }

        if let clickXY {
            let parts = clickXY.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
            if parts.count == 2, let x = Double(parts[0]), let y = Double(parts[1]) {
                actionsTaken.append(performClickXY(
                    x: x,
                    y: y,
                    pid: targetPid,
                    snapshot: snapshot,
                    allowFocusSteal: allowFocusSteal,
                    noCursor: noCursor,
                    killSwitch: killSwitch
                ))
            } else {
                actionsTaken.append(["status": "error", "action": "click_xy", "error": "Invalid coordinates '\(clickXY)'"])
            }
        }

        if let click {
            actionsTaken.append(performClickElement(
                target: click,
                snapshot: snapshot,
                allowFocusSteal: allowFocusSteal,
                noCursor: noCursor,
                killSwitch: killSwitch
            ))
        }

        if let rightClick {
            actionsTaken.append(performRightClick(
                target: rightClick,
                snapshot: snapshot,
                pid: targetPid,
                allowFocusSteal: allowFocusSteal,
                noCursor: noCursor,
                killSwitch: killSwitch
            ))
        }

        if let attach {
            if click != nil || clickXY != nil {
                Thread.sleep(forTimeInterval: 0.12)
            }
            actionsTaken.append(performAttachFile(
                filePath: attach,
                snapshot: snapshot,
                pid: targetPid,
                appName: effApp,
                allowFocusSteal: allowFocusSteal,
                killSwitch: killSwitch
            ))
        }

        if let pasteText {
            actionsTaken.append(performPaste(
                text: pasteText,
                target: clickXY == nil ? click : nil,
                snapshot: snapshot,
                pid: targetPid,
                allowFocusSteal: allowFocusSteal,
                noCursor: noCursor,
                killSwitch: killSwitch
            ))
        }

        if let typeText {
            actionsTaken.append(performTypeText(
                text: typeText,
                target: clickXY == nil ? click : nil,
                snapshot: snapshot,
                pid: targetPid,
                allowFocusSteal: allowFocusSteal,
                noCursor: noCursor,
                killSwitch: killSwitch
            ))
        }

        if let hotkey {
            actionsTaken.append(dispatchHotkey(
                chord: hotkey,
                pid: targetPid,
                appName: effApp,
                bundleID: effBundle,
                allowFocusSteal: allowFocusSteal,
                killSwitch: killSwitch
            ))
        }

        if submit {
            Thread.sleep(forTimeInterval: 0.04)
            actionsTaken.append(sendSubmit(
                snapshot: snapshot,
                pid: targetPid,
                allowFocusSteal: allowFocusSteal,
                noCursor: noCursor,
                killSwitch: killSwitch
            ))
        }

        if let menu {
            actionsTaken.append(triggerMenuItem(
                menuPath: menu,
                pid: targetPid,
                appName: effApp,
                bundleID: effBundle,
                killSwitch: killSwitch
            ))
        }

        if let scroll {
            actionsTaken.append(performScroll(
                direction: scroll,
                snapshot: snapshot,
                pid: targetPid,
                allowFocusSteal: allowFocusSteal,
                noCursor: noCursor,
                killSwitch: killSwitch
            ))
        }

        if let chainRaw {
            let chainSteps = parseChainSteps(chainRaw)
            if !chainSteps.isEmpty {
                actionsTaken.append(executeChain(
                    steps: chainSteps,
                    snapshot: snapshot,
                    appName: effApp,
                    pid: targetPid,
                    windowTitle: effWin,
                    allowFocusSteal: allowFocusSteal,
                    noCursor: noCursor,
                    killSwitch: killSwitch
                ))
            }
        }

        if killSwitch.isPaused || actionsTaken.contains(where: { ($0["kill_switch"] as? Bool) == true }) {
            return killSwitchErrorPayload()
        }

        Thread.sleep(forTimeInterval: 0.06)
        let postSnap = captureSnapshot(
            pid: targetPid.map { Int($0) },
            appName: effApp,
            windowTitleQuery: effWin.isEmpty ? nil : effWin,
            maxElements: max(60, snapshot.interactiveElements.count + 30),
            filterQuery: snapshot.filterQuery,
            killSwitch: killSwitch
        )
        let delta = computeTactileDelta(pre: snapshot, post: postSnap)
        let allOK = !actionsTaken.isEmpty && actionsTaken.allSatisfy { ($0["status"] as? String) == "success" }
        let summaryStr = actionsTaken.isEmpty
            ? "no action"
            : actionsTaken.map { ($0["summary"] as? String) ?? ($0["status"] as? String) ?? "done" }.joined(separator: " -> ")

        return [
            "status": allOK ? "success" : "partial",
            "actions": actionsTaken,
            "summary": summaryStr,
            "tactile_delta": delta,
            "post_snapshot": postSnap.toDictionary()
        ]
    }

    // MARK: - Individual Actuation Primitives

    private static func checkActuationSecurityGate(
        actionName: String,
        appName: String,
        bundleID: String = "",
        windowTitle: String = "",
        snapshot: FovealSnapshotRecord? = nil,
        element: FovealElementRecord? = nil,
        forbidSecureField: Bool = false
    ) -> [String: Any]? {
        let appDec = PrivacyBlindfold.checkApplication(appName: appName, bundleID: bundleID)
        if appDec.blocked {
            return [
                "status": "error",
                "action": actionName,
                "method": "blocked_security_gate",
                "security_blocked": true,
                "cursor_moved": false,
                "error": "Security violation: \(appDec.reason)",
                "summary": "Blocked \(actionName): \(appDec.reason)"
            ]
        }

        let origDec = PrivacyBlindfold.checkWindowOrigin(
            windowTitle: windowTitle,
            extraOrigins: loadCustomBlindfoldOrigins()
        )
        if origDec.blocked {
            return [
                "status": "error",
                "action": actionName,
                "method": "blocked_security_gate",
                "security_blocked": true,
                "cursor_moved": false,
                "error": "Security violation: \(origDec.reason)",
                "summary": "Blocked \(actionName): \(origDec.reason)"
            ]
        }

        if forbidSecureField, let cand = element ?? snapshot?.focusedElement {
            if cand.isSecure || PrivacyBlindfold.isSecureElement(role: cand.role, subrole: cand.subrole, title: cand.title) {
                let msg = "Forbidden \(actionName) into secure password field (\(cand.role))"
                return [
                    "status": "error",
                    "action": actionName,
                    "method": "blocked_secure_field",
                    "security_blocked": true,
                    "cursor_moved": false,
                    "error": "Security violation: \(msg)",
                    "summary": "Blocked \(actionName): secure password field"
                ]
            }
        }
        return nil
    }

    private static func checkHumanHIDGuard(targetPid: pid_t?, allowFocusSteal: Bool) -> [String: Any] {
        let rawIdle = wbc_CGEventSourceSecondsSinceLastEventType(0, UInt32.max)
        let idleSec = (rawIdle.isFinite && rawIdle >= 0) ? rawIdle : 999.0
        let frontPid = NSWorkspace.shared.frontmostApplication?.processIdentifier
        let humanActive = idleSec < 1.5
        let isFront = (targetPid != nil && frontPid != nil && targetPid == frontPid)
        return [
            "human_idle_seconds": (idleSec * 1000.0).rounded() / 1000.0,
            "human_active_recently": humanActive,
            "frontmost_pid": frontPid.map { Int($0) } ?? NSNull(),
            "target_pid": targetPid.map { Int($0) } ?? NSNull(),
            "is_target_frontmost": isFront,
            "allow_focus_steal": allowFocusSteal,
            "can_focus_steal": allowFocusSteal && !humanActive
        ]
    }

    private static func canUseGlobalCursor(
        snapshot: FovealSnapshotRecord,
        pid: pid_t?,
        allowFocusSteal: Bool,
        noCursor: Bool
    ) -> (Bool, [String: Any]) {
        var guardInfo = checkHumanHIDGuard(targetPid: pid, allowFocusSteal: allowFocusSteal)
        guardInfo["no_cursor"] = noCursor
        if noCursor {
            return (false, guardInfo)
        }
        let hasLiveAX = snapshot.interactiveElements.contains(where: { $0.rawElement != nil }) || snapshot.focusedElement?.rawElement != nil
        if hasLiveAX && !allowFocusSteal {
            return (false, guardInfo)
        }
        return (true, guardInfo)
    }

    private static func postKeyEvent(
        _ event: CGEvent,
        pid: pid_t?,
        allowFocusSteal: Bool,
        killSwitch: KillSwitch
    ) -> String {
        if killSwitch.isPaused {
            return "blocked_kill_switch"
        }
        if let pid, pid > 0 {
            event.postToPid(pid)
            return "CGEventPostToPid"
        }
        let guardInfo = checkHumanHIDGuard(targetPid: pid, allowFocusSteal: allowFocusSteal)
        if (guardInfo["human_active_recently"] as? Bool) == true && !allowFocusSteal {
            return "blocked_human_hid_guard"
        }
        event.post(tap: .cghidEventTap)
        return "kCGHIDEventTap"
    }

    private static func findMatchingElement(target: String, in snapshot: FovealSnapshotRecord) -> FovealElementRecord? {
        let clean = target.trimmingCharacters(in: .whitespacesAndNewlines)
        let low = clean.lowercased()
        if let exact = snapshot.interactiveElements.first(where: { $0.id.lowercased() == low }) {
            return exact
        }
        return snapshot.interactiveElements.first { el in
            let cand = "\(el.label) \(el.value ?? "") \(el.description) \(el.role)".lowercased()
            return cand.contains(low)
        }
    }

    private static func findElementAtPoint(x: Double, y: Double, in snapshot: FovealSnapshotRecord, tolerance: Double = 14.0) -> FovealElementRecord? {
        var best: FovealElementRecord?
        var bestArea = Double.infinity
        for el in snapshot.interactiveElements {
            let (bx, by, bw, bh) = el.bbox
            if bw > 0, bh > 0,
               (bx - tolerance)...(bx + bw + tolerance) ~= x,
               (by - tolerance)...(by + bh + tolerance) ~= y
            {
                let area = bw * bh
                if area < bestArea {
                    bestArea = area
                    best = el
                }
            }
        }
        return best
    }

    private static func performClickElement(
        target: String,
        snapshot: FovealSnapshotRecord,
        allowFocusSteal: Bool,
        noCursor: Bool,
        killSwitch: KillSwitch
    ) -> [String: Any] {
        if killSwitch.isPaused { return killSwitchErrorPayload() }
        guard let elem = findMatchingElement(target: target, in: snapshot) else {
            return [
                "status": "error",
                "method": "none",
                "target": target,
                "error": "Element '\(target)' not found in snapshot",
                "summary": "Element '\(target)' not found"
            ]
        }

        if let secErr = checkActuationSecurityGate(
            actionName: "click",
            appName: snapshot.appName,
            bundleID: snapshot.bundleID,
            windowTitle: snapshot.windowTitle,
            snapshot: snapshot,
            element: elem
        ) {
            return secErr
        }

        var axErrCode: AXError = .failure
        if let raw = elem.rawElement, !killSwitch.isPaused {
            _ = AXUIElementSetAttributeValue(raw, "AXFocused" as CFString, kCFBooleanTrue)
            axErrCode = AXUIElementPerformAction(raw, kAXPressAction as CFString)
            if axErrCode == .success {
                return [
                    "status": "success",
                    "method": "AXPress",
                    "element_id": elem.id,
                    "role": elem.role,
                    "label": elem.label,
                    "center": [elem.center.0, elem.center.1],
                    "cursor_moved": false,
                    "summary": "Pressed [\(elem.id)] '\(elem.label)' via native AXPress"
                ]
            }
        }

        let targetPid: pid_t? = snapshot.pid > 0 ? pid_t(snapshot.pid) : nil
        let (canCursor, guardInfo) = canUseGlobalCursor(
            snapshot: snapshot,
            pid: targetPid,
            allowFocusSteal: allowFocusSteal,
            noCursor: noCursor
        )
        guard canCursor else {
            return [
                "status": "error",
                "method": "AXPress_blocked_focus_steal",
                "element_id": elem.id,
                "role": elem.role,
                "label": elem.label,
                "center": [elem.center.0, elem.center.1],
                "cursor_moved": false,
                "hid_guard": guardInfo,
                "error": "AXPress failed (\(axErrCode.rawValue)) and global coordinate click blocked by focus-steal guard",
                "summary": "Blocked focus-stealing click on background [\(elem.id)] '\(elem.label)'"
            ]
        }

        var res = performClickXY(
            x: elem.center.0,
            y: elem.center.1,
            pid: targetPid,
            snapshot: snapshot,
            allowFocusSteal: true,
            noCursor: false,
            killSwitch: killSwitch
        )
        res["element_id"] = elem.id
        res["role"] = elem.role
        res["label"] = elem.label
        res["method"] = "CGEvent_fallback"
        res["summary"] = "Clicked [\(elem.id)] '\(elem.label)' at (\(elem.center.0), \(elem.center.1))"
        return res
    }

    private static func performClickXY(
        x: Double,
        y: Double,
        pid: pid_t?,
        snapshot: FovealSnapshotRecord,
        allowFocusSteal: Bool,
        noCursor: Bool,
        killSwitch: KillSwitch
    ) -> [String: Any] {
        if killSwitch.isPaused { return killSwitchErrorPayload() }
        if let secErr = checkActuationSecurityGate(
            actionName: "click_xy",
            appName: snapshot.appName,
            bundleID: snapshot.bundleID,
            windowTitle: snapshot.windowTitle,
            snapshot: snapshot
        ) {
            return secErr
        }

        let (canCursor, guardInfo) = canUseGlobalCursor(
            snapshot: snapshot,
            pid: pid,
            allowFocusSteal: allowFocusSteal,
            noCursor: noCursor
        )
        if !canCursor {
            if let hit = findElementAtPoint(x: x, y: y, in: snapshot), let raw = hit.rawElement, !killSwitch.isPaused {
                if AXUIElementPerformAction(raw, kAXPressAction as CFString) == .success {
                    return [
                        "status": "success",
                        "method": "AXPress",
                        "element_id": hit.id,
                        "role": hit.role,
                        "label": hit.label,
                        "coordinates": [x, y],
                        "rerouted_from_xy": true,
                        "cursor_moved": false,
                        "summary": "Pressed [\(hit.id)] '\(hit.label)' via zero-cursor AXPress"
                    ]
                }
            }
            return [
                "status": "error",
                "method": "CGEvent_xy_blocked_focus_steal",
                "coordinates": [x, y],
                "cursor_moved": false,
                "hid_guard": guardInfo,
                "error": String(format: "Blocked cursor-hijacking coordinate click at (%.1f, %.1f) while backgrounded", x, y),
                "summary": String(format: "Blocked focus-stealing coordinate click at (%.1f, %.1f)", x, y)
            ]
        }

        let pt = CGPoint(x: x, y: y)
        guard let down = CGEvent(mouseEventSource: nil, mouseType: .leftMouseDown, mouseCursorPosition: pt, mouseButton: .left),
              let up = CGEvent(mouseEventSource: nil, mouseType: .leftMouseUp, mouseCursorPosition: pt, mouseButton: .left),
              !killSwitch.isPaused
        else {
            return killSwitch.isPaused ? killSwitchErrorPayload() : [
                "status": "error",
                "method": "CGEvent_xy",
                "coordinates": [x, y],
                "error": "CGEventCreateMouseEvent failed"
            ]
        }
        down.post(tap: .cghidEventTap)
        Thread.sleep(forTimeInterval: 0.02)
        up.post(tap: .cghidEventTap)
        return [
            "status": "success",
            "method": "CGEvent_xy",
            "coordinates": [x, y],
            "cursor_moved": true,
            "summary": String(format: "Clicked at (%.1f, %.1f)", x, y)
        ]
    }

    private static func performHover(
        target: String,
        snapshot: FovealSnapshotRecord,
        pid: pid_t?,
        allowFocusSteal: Bool,
        noCursor: Bool,
        killSwitch: KillSwitch
    ) -> [String: Any] {
        if killSwitch.isPaused { return killSwitchErrorPayload() }
        let (cx, cy, elem) = resolveTargetCoordinates(target: target, snapshot: snapshot)
        guard let cx, let cy else {
            return ["status": "error", "action": "hover", "target": target, "error": "Target '\(target)' not found"]
        }
        let (canCursor, guardInfo) = canUseGlobalCursor(snapshot: snapshot, pid: pid, allowFocusSteal: allowFocusSteal, noCursor: noCursor)
        if !canCursor {
            if let raw = elem?.rawElement, !killSwitch.isPaused,
               AXUIElementSetAttributeValue(raw, "AXFocused" as CFString, kCFBooleanTrue) == .success
            {
                return [
                    "status": "success",
                    "action": "hover",
                    "method": "AXFocused_hover",
                    "cursor_moved": false,
                    "target": target,
                    "coordinates": [cx, cy],
                    "summary": "Focused '\(elem?.label ?? target)' via zero-cursor AXFocused"
                ]
            }
            return [
                "status": "error",
                "action": "hover",
                "method": "hover_blocked_focus_steal",
                "cursor_moved": false,
                "hid_guard": guardInfo,
                "error": "Blocked cursor-hijacking hover while backgrounded"
            ]
        }
        let pt = CGPoint(x: cx, y: cy)
        guard let move = CGEvent(mouseEventSource: nil, mouseType: .mouseMoved, mouseCursorPosition: pt, mouseButton: .left),
              !killSwitch.isPaused
        else {
            return killSwitchErrorPayload()
        }
        move.post(tap: .cghidEventTap)
        Thread.sleep(forTimeInterval: 0.08)
        return [
            "status": "success",
            "action": "hover",
            "method": "CGEvent_mouse_moved",
            "cursor_moved": true,
            "target": target,
            "coordinates": [cx, cy],
            "summary": String(format: "Hovered at (%.1f, %.1f)", cx, cy)
        ]
    }

    private static func performRightClick(
        target: String,
        snapshot: FovealSnapshotRecord,
        pid: pid_t?,
        allowFocusSteal: Bool,
        noCursor: Bool,
        killSwitch: KillSwitch
    ) -> [String: Any] {
        if killSwitch.isPaused { return killSwitchErrorPayload() }
        let (cx, cy, elem) = resolveTargetCoordinates(target: target, snapshot: snapshot)
        guard let cx, let cy else {
            return ["status": "error", "action": "right_click", "target": target, "error": "Target '\(target)' not found"]
        }
        let (canCursor, guardInfo) = canUseGlobalCursor(snapshot: snapshot, pid: pid, allowFocusSteal: allowFocusSteal, noCursor: noCursor)
        if !canCursor {
            if let raw = elem?.rawElement, !killSwitch.isPaused,
               AXUIElementPerformAction(raw, kAXShowMenuAction as CFString) == .success
            {
                return [
                    "status": "success",
                    "action": "right_click",
                    "method": "AXShowMenu",
                    "cursor_moved": false,
                    "target": target,
                    "coordinates": [cx, cy],
                    "summary": "Opened context menu on '\(elem?.label ?? target)' via AXShowMenu"
                ]
            }
            return [
                "status": "error",
                "action": "right_click",
                "method": "right_click_blocked_focus_steal",
                "cursor_moved": false,
                "hid_guard": guardInfo,
                "error": "Blocked cursor-hijacking right-click while backgrounded"
            ]
        }
        let pt = CGPoint(x: cx, y: cy)
        guard let down = CGEvent(mouseEventSource: nil, mouseType: .rightMouseDown, mouseCursorPosition: pt, mouseButton: .right),
              let up = CGEvent(mouseEventSource: nil, mouseType: .rightMouseUp, mouseCursorPosition: pt, mouseButton: .right),
              !killSwitch.isPaused
        else {
            return killSwitchErrorPayload()
        }
        down.post(tap: .cghidEventTap)
        Thread.sleep(forTimeInterval: 0.02)
        up.post(tap: .cghidEventTap)
        return [
            "status": "success",
            "action": "right_click",
            "method": "CGEvent_right_click",
            "cursor_moved": true,
            "target": target,
            "coordinates": [cx, cy],
            "summary": String(format: "Right-clicked at (%.1f, %.1f)", cx, cy)
        ]
    }

    private static func resolveInputTarget(target: String?, snapshot: FovealSnapshotRecord) -> FovealElementRecord? {
        if let target {
            return findMatchingElement(target: target, in: snapshot)
        }
        if let foc = snapshot.focusedElement, ["AXTextField", "AXTextArea", "AXSearchField"].contains(foc.role) {
            return foc
        }
        for el in snapshot.interactiveElements {
            let low = "\(el.label) \(el.description) \(el.title)".lowercased()
            if ["prompt", "message", "composer", "chat"].contains(where: { low.contains($0) }) {
                return el
            }
        }
        for el in snapshot.interactiveElements where ["AXTextArea", "AXTextField", "AXSearchField", "AXComboBox"].contains(el.role) {
            return el
        }
        return snapshot.focusedElement ?? snapshot.interactiveElements.first(where: { $0.role == "AXWebArea" })
    }

    private static func atomicPasteText(
        _ text: String,
        targetElem: FovealElementRecord?,
        pid: pid_t?,
        allowFocusSteal: Bool,
        killSwitch: KillSwitch
    ) -> Bool {
        if killSwitch.isPaused { return false }

        // 1. Direct background AXValue write first (0 clipboard mutation, 0 keyboard events)
        var rawTarget = targetElem?.rawElement
        if rawTarget == nil, let pid, pid > 0 {
            let axApp = AXUIElementCreateApplication(pid)
            rawTarget = getAXElement(axApp, attribute: "AXFocusedUIElement", killSwitch: killSwitch)
        }
        if let raw = rawTarget, !killSwitch.isPaused {
            _ = AXUIElementSetAttributeValue(raw, "AXFocused" as CFString, kCFBooleanTrue)
            if AXUIElementSetAttributeValue(raw, kAXValueAttribute as CFString, text as CFTypeRef) == .success {
                return true
            }
        }

        // 2. Clipboard Cmd+V fallback via CGEventPostToPid (refuse to clobber concealed vault clipboard)
        let pb = NSPasteboard.general
        let existingTypes = (pb.types ?? []).map { $0.rawValue }
        if PrivacyBlindfold.isPasteboardConcealed(typeIdentifiers: existingTypes) {
            return false
        }
        let priorString = pb.string(forType: .string)
        pb.clearContents()
        pb.setString(text, forType: .string)

        guard let down = CGEvent(keyboardEventSource: nil, virtualKey: 9, keyDown: true),
              let up = CGEvent(keyboardEventSource: nil, virtualKey: 9, keyDown: false),
              !killSwitch.isPaused
        else {
            return false
        }
        down.flags = .maskCommand
        up.flags = .maskCommand
        _ = postKeyEvent(down, pid: pid, allowFocusSteal: allowFocusSteal, killSwitch: killSwitch)
        Thread.sleep(forTimeInterval: 0.02)
        _ = postKeyEvent(up, pid: pid, allowFocusSteal: allowFocusSteal, killSwitch: killSwitch)
        Thread.sleep(forTimeInterval: 0.05)

        pb.clearContents()
        if let priorString {
            pb.setString(priorString, forType: .string)
        }
        return true
    }

    private static func performPaste(
        text: String,
        target: String?,
        snapshot: FovealSnapshotRecord,
        pid: pid_t?,
        allowFocusSteal: Bool,
        noCursor: Bool,
        killSwitch: KillSwitch
    ) -> [String: Any] {
        if killSwitch.isPaused { return killSwitchErrorPayload() }
        let targetElem = resolveInputTarget(target: target, snapshot: snapshot)
        if let secErr = checkActuationSecurityGate(
            actionName: "paste",
            appName: snapshot.appName,
            bundleID: snapshot.bundleID,
            windowTitle: snapshot.windowTitle,
            snapshot: snapshot,
            element: targetElem,
            forbidSecureField: true
        ) {
            return secErr
        }
        if let target {
            _ = performClickElement(target: target, snapshot: snapshot, allowFocusSteal: allowFocusSteal, noCursor: noCursor, killSwitch: killSwitch)
        } else if let targetElem, let raw = targetElem.rawElement, !killSwitch.isPaused {
            _ = AXUIElementPerformAction(raw, kAXPressAction as CFString)
            Thread.sleep(forTimeInterval: 0.04)
        }

        let ok = atomicPasteText(text, targetElem: targetElem, pid: pid, allowFocusSteal: allowFocusSteal, killSwitch: killSwitch)
        return [
            "status": ok ? "success" : "error",
            "action": "paste",
            "method": "atomic_paste_cmd_v",
            "target_element": targetElem?.toDictionary() ?? NSNull(),
            "summary": "Pasted \(text.count) chars via background-safe atomic paste"
        ]
    }

    private static func performTypeText(
        text: String,
        target: String?,
        snapshot: FovealSnapshotRecord,
        pid: pid_t?,
        allowFocusSteal: Bool,
        noCursor: Bool,
        killSwitch: KillSwitch
    ) -> [String: Any] {
        if killSwitch.isPaused { return killSwitchErrorPayload() }
        let targetElem = resolveInputTarget(target: target, snapshot: snapshot)
        if let secErr = checkActuationSecurityGate(
            actionName: "type_text",
            appName: snapshot.appName,
            bundleID: snapshot.bundleID,
            windowTitle: snapshot.windowTitle,
            snapshot: snapshot,
            element: targetElem,
            forbidSecureField: true
        ) {
            return secErr
        }

        if let target {
            _ = performClickElement(target: target, snapshot: snapshot, allowFocusSteal: allowFocusSteal, noCursor: noCursor, killSwitch: killSwitch)
        }

        var axSet = false
        if let raw = targetElem?.rawElement, !killSwitch.isPaused {
            _ = AXUIElementSetAttributeValue(raw, "AXFocused" as CFString, kCFBooleanTrue)
            if AXUIElementSetAttributeValue(raw, kAXValueAttribute as CFString, text as CFTypeRef) == .success {
                axSet = true
            }
        }

        if !axSet && text.count > 40 {
            if atomicPasteText(text, targetElem: targetElem, pid: pid, allowFocusSteal: allowFocusSteal, killSwitch: killSwitch) {
                return [
                    "status": "success",
                    "method": "atomic_paste",
                    "target_element": targetElem?.toDictionary() ?? NSNull(),
                    "summary": "Injected \(text.count) chars via background-safe atomic paste"
                ]
            }
        }

        var quartzOK = false
        if let down = CGEvent(keyboardEventSource: nil, virtualKey: 0, keyDown: true),
           let up = CGEvent(keyboardEventSource: nil, virtualKey: 0, keyDown: false),
           !killSwitch.isPaused
        {
            var utf16 = Array(text.utf16)
            down.keyboardSetUnicodeString(stringLength: utf16.count, unicodeString: &utf16)
            up.keyboardSetUnicodeString(stringLength: utf16.count, unicodeString: &utf16)
            let mode = postKeyEvent(down, pid: pid, allowFocusSteal: allowFocusSteal, killSwitch: killSwitch)
            Thread.sleep(forTimeInterval: 0.01)
            _ = postKeyEvent(up, pid: pid, allowFocusSteal: allowFocusSteal, killSwitch: killSwitch)
            quartzOK = !mode.hasPrefix("blocked_")
        }

        let method = (axSet && quartzOK) ? "AXValue+CGEvent" : (axSet ? "AXValue" : (quartzOK ? "CGEvent" : "failed"))
        return [
            "status": (axSet || quartzOK) ? "success" : "error",
            "method": method,
            "target_element": targetElem?.toDictionary() ?? NSNull(),
            "summary": "Typed \(text.count) chars via \(method)"
        ]
    }

    public static func dispatchHotkey(
        chord: String,
        pid: pid_t?,
        appName: String?,
        bundleID: String = "",
        allowFocusSteal: Bool = false,
        killSwitch: KillSwitch = .shared
    ) -> [String: Any] {
        if killSwitch.isPaused { return killSwitchErrorPayload() }
        let parts = chord
            .replacingOccurrences(of: "-", with: "+")
            .split(separator: "+")
            .map { $0.trimmingCharacters(in: .whitespaces).lowercased() }
            .filter { !$0.isEmpty }
        guard let keyName = parts.last else {
            return ["status": "error", "action": "hotkey", "error": "Empty hotkey chord"]
        }

        var flags: CGEventFlags = []
        for mod in parts.dropLast() {
            if let f = quartzModifiers[mod] {
                flags.insert(f)
            }
        }
        guard let keycode = quartzKeycodes[keyName] else {
            return ["status": "error", "action": "hotkey", "error": "Unknown key in hotkey chord: '\(keyName)'"]
        }

        if let secErr = checkActuationSecurityGate(actionName: "hotkey", appName: appName ?? "", bundleID: bundleID) {
            return secErr
        }

        guard let down = CGEvent(keyboardEventSource: nil, virtualKey: keycode, keyDown: true),
              let up = CGEvent(keyboardEventSource: nil, virtualKey: keycode, keyDown: false),
              !killSwitch.isPaused
        else {
            return killSwitchErrorPayload()
        }
        if !flags.isEmpty {
            down.flags = flags
            up.flags = flags
        }
        let mode = postKeyEvent(down, pid: pid, allowFocusSteal: allowFocusSteal, killSwitch: killSwitch)
        Thread.sleep(forTimeInterval: 0.02)
        _ = postKeyEvent(up, pid: pid, allowFocusSteal: allowFocusSteal, killSwitch: killSwitch)
        if mode.hasPrefix("blocked_") {
            return ["status": "error", "action": "hotkey", "chord": chord, "dispatch_mode": mode, "error": "Blocked by \(mode)"]
        }
        return [
            "status": "success",
            "action": "hotkey",
            "chord": chord,
            "keycode": Int(keycode),
            "pid": pid.map { Int($0) } ?? NSNull(),
            "dispatch_mode": mode,
            "summary": "Dispatched hotkey '\(chord)' (keycode \(keycode)) via \(mode)"
        ]
    }

    private static func sendSubmit(
        snapshot: FovealSnapshotRecord,
        pid: pid_t?,
        allowFocusSteal: Bool,
        noCursor: Bool,
        killSwitch: KillSwitch
    ) -> [String: Any] {
        if killSwitch.isPaused { return killSwitchErrorPayload() }
        guard let down = CGEvent(keyboardEventSource: nil, virtualKey: 36, keyDown: true),
              let up = CGEvent(keyboardEventSource: nil, virtualKey: 36, keyDown: false)
        else {
            return ["status": "error", "action": "submit", "error": "CGEventCreateKeyboardEvent failed"]
        }
        let mode = postKeyEvent(down, pid: pid, allowFocusSteal: allowFocusSteal, killSwitch: killSwitch)
        Thread.sleep(forTimeInterval: 0.03)
        _ = postKeyEvent(up, pid: pid, allowFocusSteal: allowFocusSteal, killSwitch: killSwitch)

        var buttonClicked: String?
        for el in snapshot.interactiveElements where ["AXButton", "AXPopUpButton"].contains(el.role) {
            let lbl = (el.label.isEmpty ? el.title : el.label).trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if ["send", "submit", "send message", "run", "run query", "execute"].contains(lbl) {
                _ = performClickElement(target: el.id, snapshot: snapshot, allowFocusSteal: allowFocusSteal, noCursor: noCursor, killSwitch: killSwitch)
                buttonClicked = el.label.isEmpty ? el.id : el.label
                break
            }
        }
        return [
            "status": "success",
            "action": "submit",
            "keycode": 36,
            "dispatch_mode": mode,
            "button_fallback": buttonClicked ?? NSNull(),
            "summary": "Sent Return/Enter (keycode 36 via \(mode))"
        ]
    }

    private static func triggerMenuItem(
        menuPath: String,
        pid: pid_t?,
        appName: String?,
        bundleID: String = "",
        killSwitch: KillSwitch
    ) -> [String: Any] {
        if killSwitch.isPaused { return killSwitchErrorPayload() }
        let parts = menuPath.split(separator: ">").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        guard let first = parts.first else {
            return ["status": "error", "action": "menu", "error": "Empty menu path"]
        }
        if let secErr = checkActuationSecurityGate(actionName: "menu", appName: appName ?? "", bundleID: bundleID) {
            return secErr
        }
        let resolvedPid: pid_t? = pid ?? resolveTargetApp(pid: nil, appName: appName).0
        guard let targetPid = resolvedPid, targetPid > 0 else {
            return ["status": "error", "action": "menu", "path": menuPath, "error": "Could not resolve target PID"]
        }

        let axApp = AXUIElementCreateApplication(targetPid)
        guard let menuBar = getAXElement(axApp, attribute: "AXMenuBar", killSwitch: killSwitch) else {
            return ["status": "error", "action": "menu", "path": menuPath, "error": "Target application has no AXMenuBar"]
        }

        let topItems = getAXElementArray(menuBar, attribute: "AXChildren", killSwitch: killSwitch)
        let firstLow = first.lowercased()
        guard var curr = topItems.first(where: {
            let t = getAXString($0, attribute: "AXTitle", killSwitch: killSwitch).lowercased()
            return t == firstLow || t.contains(firstLow)
        }) else {
            return ["status": "error", "action": "menu", "path": menuPath, "error": "Menu bar item '\(first)' not found"]
        }

        for part in parts.dropFirst() {
            var children = getAXElementArray(curr, attribute: "AXChildren", killSwitch: killSwitch)
            if children.count == 1, getAXString(children[0], attribute: "AXRole", killSwitch: killSwitch) == "AXMenu" {
                children = getAXElementArray(children[0], attribute: "AXChildren", killSwitch: killSwitch)
            }
            let pLow = part.lowercased()
            guard let next = children.first(where: {
                let t = getAXString($0, attribute: "AXTitle", killSwitch: killSwitch).lowercased()
                return t == pLow || t.contains(pLow)
            }) else {
                return ["status": "error", "action": "menu", "path": menuPath, "error": "Menu item '\(part)' not found"]
            }
            curr = next
        }

        if killSwitch.isPaused { return killSwitchErrorPayload() }
        let err = AXUIElementPerformAction(curr, kAXPressAction as CFString)
        if err == .success {
            return ["status": "success", "action": "menu", "path": menuPath, "summary": "Triggered menu item '\(menuPath)' via AXPress"]
        }
        return ["status": "error", "action": "menu", "path": menuPath, "error": "AXPress returned error code \(err.rawValue)"]
    }

    private static func performScroll(
        direction: String,
        snapshot: FovealSnapshotRecord,
        pid: pid_t?,
        allowFocusSteal: Bool,
        noCursor: Bool,
        killSwitch: KillSwitch
    ) -> [String: Any] {
        if killSwitch.isPaused { return killSwitchErrorPayload() }
        let d = direction.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let axAct = (d == "down" || d == "bottom") ? "AXScrollDownByPage" : "AXScrollUpByPage"

        for el in snapshot.interactiveElements where ["AXScrollArea", "AXWebArea", "AXList", "AXTable"].contains(el.role) {
            if let raw = el.rawElement, !killSwitch.isPaused,
               AXUIElementPerformAction(raw, axAct as CFString) == .success
            {
                return [
                    "status": "success",
                    "action": "scroll",
                    "direction": d,
                    "ax_scroll": true,
                    "quartz_scroll": false,
                    "cursor_moved": false,
                    "summary": "Scrolled \(d) via background AXScroll (0 cursor impact)"
                ]
            }
        }

        let (canCursor, guardInfo) = canUseGlobalCursor(snapshot: snapshot, pid: pid, allowFocusSteal: allowFocusSteal, noCursor: noCursor)
        if !canCursor {
            if let pid, pid > 0, !killSwitch.isPaused {
                let keycode: CGKeyCode = (d == "bottom") ? 119 : ((d == "top") ? 115 : ((d == "down") ? 121 : 116))
                if let down = CGEvent(keyboardEventSource: nil, virtualKey: keycode, keyDown: true),
                   let up = CGEvent(keyboardEventSource: nil, virtualKey: keycode, keyDown: false)
                {
                    let mode = postKeyEvent(down, pid: pid, allowFocusSteal: allowFocusSteal, killSwitch: killSwitch)
                    Thread.sleep(forTimeInterval: 0.02)
                    _ = postKeyEvent(up, pid: pid, allowFocusSteal: allowFocusSteal, killSwitch: killSwitch)
                    if mode == "CGEventPostToPid" {
                        return [
                            "status": "success",
                            "action": "scroll",
                            "direction": d,
                            "ax_scroll": false,
                            "pid_key_scroll": true,
                            "cursor_moved": false,
                            "summary": "Scrolled \(d) in background (CGEventPostToPid)"
                        ]
                    }
                }
            }
            return [
                "status": "error",
                "action": "scroll",
                "direction": d,
                "cursor_moved": false,
                "hid_guard": guardInfo,
                "summary": "Blocked global HID scroll (\(d)) to protect active cursor window"
            ]
        }

        let deltaY: Int32 = (d == "bottom") ? -20 : ((d == "top") ? 20 : ((d == "down") ? -5 : 5))
        if let ev = CGEvent(scrollWheelEvent2Source: nil, units: .line, wheelCount: 1, wheel1: deltaY, wheel2: 0, wheel3: 0),
           !killSwitch.isPaused
        {
            ev.post(tap: .cghidEventTap)
            return ["status": "success", "action": "scroll", "direction": d, "quartz_scroll": true, "summary": "Scrolled \(d) via Quartz"]
        }
        return ["status": "error", "action": "scroll", "direction": d, "error": "Scroll failed"]
    }

    private static func performAttachFile(
        filePath: String,
        snapshot: FovealSnapshotRecord,
        pid: pid_t?,
        appName: String,
        allowFocusSteal: Bool,
        killSwitch: KillSwitch
    ) -> [String: Any] {
        if killSwitch.isPaused { return killSwitchErrorPayload() }
        let expanded = (filePath as NSString).expandingTildeInPath
        let resolved = URL(fileURLWithPath: expanded).standardizedFileURL.path
        var actions: [[String: Any]] = []
        actions.append(dispatchHotkey(chord: "cmd+shift+g", pid: pid, appName: appName, allowFocusSteal: allowFocusSteal, killSwitch: killSwitch))
        Thread.sleep(forTimeInterval: 0.06)
        let pasted = atomicPasteText(resolved, targetElem: nil, pid: pid, allowFocusSteal: allowFocusSteal, killSwitch: killSwitch)
        actions.append(["status": pasted ? "success" : "error", "action": "paste_path", "path": resolved])
        actions.append(sendSubmit(snapshot: snapshot, pid: pid, allowFocusSteal: allowFocusSteal, noCursor: false, killSwitch: killSwitch))
        Thread.sleep(forTimeInterval: 0.10)
        actions.append(sendSubmit(snapshot: snapshot, pid: pid, allowFocusSteal: allowFocusSteal, noCursor: false, killSwitch: killSwitch))
        let allOK = actions.allSatisfy { ($0["status"] as? String) == "success" }
        return [
            "status": allOK ? "success" : "partial",
            "action": "attach_file",
            "file_path": resolved,
            "actions": actions,
            "summary": "Attached file '\(resolved)' via NSOpenPanel Go-to-folder sheet"
        ]
    }

    private static func parseChainSteps(_ raw: Any) -> [(String, String)] {
        if let str = raw as? String {
            var tokens: [String] = []
            var curr = ""
            var inQuote = false
            var quoteChar: Character = "\""
            for ch in str {
                if ch == "\"" || ch == "'" {
                    if !inQuote {
                        inQuote = true
                        quoteChar = ch
                    } else if ch == quoteChar {
                        inQuote = false
                    }
                    curr.append(ch)
                } else if ch == "," && !inQuote {
                    let trimmed = curr.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !trimmed.isEmpty { tokens.append(trimmed) }
                    curr = ""
                } else {
                    curr.append(ch)
                }
            }
            let tail = curr.trimmingCharacters(in: .whitespacesAndNewlines)
            if !tail.isEmpty { tokens.append(tail) }

            return tokens.map { step in
                if let colon = step.firstIndex(of: ":") {
                    let act = String(step[..<colon]).trimmingCharacters(in: .whitespaces).lowercased()
                    var arg = String(step[step.index(after: colon)...]).trimmingCharacters(in: .whitespaces)
                    if (arg.hasPrefix("\"") && arg.hasSuffix("\"")) || (arg.hasPrefix("'") && arg.hasSuffix("'")), arg.count >= 2 {
                        arg = String(arg.dropFirst().dropLast())
                    }
                    return (act, arg)
                }
                return (step.lowercased(), "")
            }
        }
        if let arr = raw as? [[String: Any]] {
            return arr.map { d in
                let act = ((d["action"] as? String) ?? (d["type"] as? String) ?? "").lowercased()
                let arg = (d["target"] as? String) ?? (d["value"] as? String) ?? (d["text"] as? String) ?? (d["path"] as? String) ?? ""
                return (act, arg)
            }
        }
        return []
    }

    private static func executeChain(
        steps: [(String, String)],
        snapshot: FovealSnapshotRecord,
        appName: String,
        pid: pid_t?,
        windowTitle: String,
        allowFocusSteal: Bool,
        noCursor: Bool,
        killSwitch: KillSwitch
    ) -> [String: Any] {
        var actions: [[String: Any]] = []
        let currSnap = snapshot
        for (act, arg) in steps {
            // Immediate mid-chain Kill Switch abort
            if killSwitch.isPaused {
                return killSwitchErrorPayload()
            }
            switch act {
            case "click", "act":
                actions.append(performClickElement(target: arg, snapshot: currSnap, allowFocusSteal: allowFocusSteal, noCursor: noCursor, killSwitch: killSwitch))
            case "click-xy", "click_xy":
                let p = arg.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
                if p.count == 2, let x = Double(p[0]), let y = Double(p[1]) {
                    actions.append(performClickXY(x: x, y: y, pid: pid, snapshot: currSnap, allowFocusSteal: allowFocusSteal, noCursor: noCursor, killSwitch: killSwitch))
                } else {
                    actions.append(["status": "error", "action": "click_xy", "error": "Invalid coordinates '\(arg)'"])
                }
            case "hover":
                actions.append(performHover(target: arg, snapshot: currSnap, pid: pid, allowFocusSteal: allowFocusSteal, noCursor: noCursor, killSwitch: killSwitch))
            case "right-click", "right_click":
                actions.append(performRightClick(target: arg, snapshot: currSnap, pid: pid, allowFocusSteal: allowFocusSteal, noCursor: noCursor, killSwitch: killSwitch))
            case "type":
                actions.append(performTypeText(text: arg, target: nil, snapshot: currSnap, pid: pid, allowFocusSteal: allowFocusSteal, noCursor: noCursor, killSwitch: killSwitch))
            case "paste":
                actions.append(performPaste(text: arg, target: nil, snapshot: currSnap, pid: pid, allowFocusSteal: allowFocusSteal, noCursor: noCursor, killSwitch: killSwitch))
            case "attach":
                actions.append(performAttachFile(filePath: arg, snapshot: currSnap, pid: pid, appName: appName, allowFocusSteal: allowFocusSteal, killSwitch: killSwitch))
            case "hotkey", "key":
                actions.append(dispatchHotkey(chord: arg, pid: pid, appName: appName, allowFocusSteal: allowFocusSteal, killSwitch: killSwitch))
            case "menu":
                actions.append(triggerMenuItem(menuPath: arg, pid: pid, appName: appName, killSwitch: killSwitch))
            case "scroll":
                actions.append(performScroll(direction: arg, snapshot: currSnap, pid: pid, allowFocusSteal: allowFocusSteal, noCursor: noCursor, killSwitch: killSwitch))
            case "submit":
                Thread.sleep(forTimeInterval: 0.04)
                actions.append(sendSubmit(snapshot: currSnap, pid: pid, allowFocusSteal: allowFocusSteal, noCursor: noCursor, killSwitch: killSwitch))
            case "wait-reply", "wait_reply":
                let tSec = Double(arg) ?? 15.0
                let wRes = waitForReply(appName: appName, pid: pid.map { Int($0) }, windowTitle: windowTitle, timeoutSec: tSec, preSnapshot: currSnap, killSwitch: killSwitch)
                actions.append(["status": "success", "action": "wait_reply", "settled": wRes["settled"] ?? false])
            case "read-last", "read_last":
                let texts = extractTextBlocks(from: currSnap)
                actions.append(["status": "success", "action": "read_last", "messages": texts, "last_message": texts.last ?? ""])
            case "sleep":
                let sec = max(0.0, min(5.0, Double(arg) ?? 0.1))
                Thread.sleep(forTimeInterval: sec)
                actions.append(["status": "success", "action": "sleep", "seconds": sec])
            default:
                actions.append(["status": "error", "action": act, "error": "Unknown chain step '\(act)'"])
            }
        }
        let allOK = actions.allSatisfy { ($0["status"] as? String) == "success" }
        return [
            "status": allOK ? "success" : "partial",
            "action": "execute_chain",
            "steps_count": steps.count,
            "actions": actions,
            "summary": actions.map { ($0["summary"] as? String) ?? ($0["status"] as? String) ?? "done" }.joined(separator: " -> ")
        ]
    }

    // MARK: - Stream-Settle, Text Extraction & Tactile Delta

    public static func extractTextBlocks(from snapshot: FovealSnapshotRecord) -> [String] {
        let buttonRoles: Set<String> = ["AXButton", "AXPopUpButton", "AXMenuButton", "AXRadioButton", "AXMenuItem", "AXCheckBox"]
        let sidebarCutoff = snapshot.windowBBox.0 + 250.0
        var blocks: [String] = []
        for el in snapshot.interactiveElements {
            if let sr = el.subrole, ignoreTextSubroles.contains(sr) { continue }
            if buttonRoles.contains(el.role) { continue }
            if snapshot.windowBBox.2 > 500.0 && el.center.0 < sidebarCutoff { continue }
            let cand = (el.value ?? (el.label.isEmpty ? (el.title.isEmpty ? el.description : el.title) : el.label))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if cand.count < 2 || cand.hasPrefix("AXEmpty") || cand.hasPrefix("AXApplicationStatus") { continue }
            if blocks.last != cand {
                blocks.append(cand)
            }
        }
        return blocks
    }

    public static func waitForReply(
        appName: String?,
        pid: Int?,
        windowTitle: String?,
        timeoutSec: Double,
        pollIntervalSec: Double = 0.25,
        preSnapshot: FovealSnapshotRecord?,
        killSwitch: KillSwitch
    ) -> [String: Any] {
        let t0 = CFAbsoluteTimeGetCurrent()
        let preBlocks = preSnapshot.map { extractTextBlocks(from: $0) } ?? []
        let preSet = Set(preBlocks)
        var prevCandidate: String?
        var sameCount = 0
        var currBlocks: [String] = []

        while (CFAbsoluteTimeGetCurrent() - t0) < timeoutSec {
            if killSwitch.isPaused {
                return killSwitchErrorPayload()
            }
            Thread.sleep(forTimeInterval: pollIntervalSec)
            if killSwitch.isPaused {
                return killSwitchErrorPayload()
            }
            let currSnap = captureSnapshot(
                pid: pid,
                appName: appName,
                windowTitleQuery: windowTitle,
                maxElements: 250,
                killSwitch: killSwitch
            )
            currBlocks = extractTextBlocks(from: currSnap)
            let latest = currBlocks.last ?? ""
            let hasNew = currBlocks.count > preBlocks.count || !preSet.contains(latest) || (!preBlocks.isEmpty && latest != preBlocks.last)
            if hasNew && !latest.isEmpty {
                if latest == prevCandidate {
                    sameCount += 1
                } else {
                    prevCandidate = latest
                    sameCount = 1
                }
                if sameCount >= 2 {
                    let elapsed = ((CFAbsoluteTimeGetCurrent() - t0) * 100000.0).rounded() / 100.0
                    var newMsgs = currBlocks.filter { !preSet.contains($0) }
                    if newMsgs.isEmpty { newMsgs = [latest] }
                    return [
                        "settled": true,
                        "elapsed_ms": elapsed,
                        "new_messages": newMsgs,
                        "last_message": latest
                    ]
                }
            }
        }

        let elapsed = ((CFAbsoluteTimeGetCurrent() - t0) * 100000.0).rounded() / 100.0
        let latest = currBlocks.last ?? ""
        var newMsgs = currBlocks.filter { !preSet.contains($0) }
        if newMsgs.isEmpty && !latest.isEmpty && latest != preBlocks.last {
            newMsgs = [latest]
        }
        return [
            "settled": false,
            "elapsed_ms": elapsed,
            "new_messages": newMsgs,
            "last_message": latest
        ]
    }

    public static func computeTactileDelta(pre: FovealSnapshotRecord, post: FovealSnapshotRecord) -> [String: Any] {
        let preLabels = Set(pre.interactiveElements.map { $0.label }.filter { !$0.isEmpty })
        let postLabels = Set(post.interactiveElements.map { $0.label }.filter { !$0.isEmpty })
        let added = Array(postLabels.subtracting(preLabels).prefix(5))
        let removed = Array(preLabels.subtracting(postLabels).prefix(5))

        let addedCount = max(0, post.interactiveElements.count - pre.interactiveElements.count) + added.count
        let removedCount = max(0, pre.interactiveElements.count - post.interactiveElements.count) + removed.count

        var preValues: [String: String] = [:]
        for el in pre.interactiveElements {
            if let v = el.value { preValues[el.label.isEmpty ? el.id : el.label] = v }
        }
        var postValues: [String: String] = [:]
        for el in post.interactiveElements {
            if let v = el.value { postValues[el.label.isEmpty ? el.id : el.label] = v }
        }
        let valueChanged = preValues != postValues
        let focusChanged = (pre.focusedElement?.label != post.focusedElement?.label) || (pre.focusedElement?.role != post.focusedElement?.role)
        let titleChanged = pre.windowTitle != post.windowTitle
        let verified = addedCount > 0 || removedCount > 0 || valueChanged || focusChanged || titleChanged

        return [
            "verified": verified,
            "elements_added_count": addedCount,
            "elements_removed_count": removedCount,
            "added_labels": added,
            "removed_labels": removed,
            "value_changed": valueChanged,
            "focus_changed": focusChanged,
            "window_title_changed": titleChanged,
            "post_element_count": post.interactiveElements.count
        ]
    }

    // MARK: - AX Helpers

    private static func resolveTargetApp(pid: Int?, appName: String?) -> (pid_t?, String, String) {
        let ws = NSWorkspace.shared
        if let pid, pid > 0 {
            let p = pid_t(pid)
            if let running = NSRunningApplication(processIdentifier: p) {
                return (p, running.localizedName ?? "PID-\(pid)", running.bundleIdentifier ?? "")
            }
            return (p, "PID-\(pid)", "")
        }

        if let appName, !appName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let q = appName.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            let apps = ws.runningApplications
            if let exact = apps.first(where: { ($0.localizedName ?? "").trimmingCharacters(in: .whitespaces).lowercased() == q }) {
                return (exact.processIdentifier, exact.localizedName ?? appName, exact.bundleIdentifier ?? "")
            }
            if let sub = apps.first(where: { ($0.localizedName ?? "").lowercased().contains(q) }) {
                return (sub.processIdentifier, sub.localizedName ?? appName, sub.bundleIdentifier ?? "")
            }
            if let bid = apps.first(where: {
                let b = ($0.bundleIdentifier ?? "").lowercased()
                let ex = ($0.executableURL?.lastPathComponent ?? "").lowercased()
                return b.contains(q) || (!ex.isEmpty && ex.contains(q))
            }) {
                return (bid.processIdentifier, bid.localizedName ?? appName, bid.bundleIdentifier ?? "")
            }
        }

        if let front = ws.frontmostApplication {
            return (front.processIdentifier, front.localizedName ?? "Unknown", front.bundleIdentifier ?? "")
        }
        return (nil, appName ?? "Unknown", "")
    }

    private static func resolveTargetCoordinates(target: String, snapshot: FovealSnapshotRecord) -> (Double?, Double?, FovealElementRecord?) {
        let clean = target.trimmingCharacters(in: .whitespacesAndNewlines)
        if clean.contains(",") {
            let parts = clean.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
            if parts.count == 2, let x = Double(parts[0]), let y = Double(parts[1]) {
                return (x, y, nil)
            }
        }
        if let elem = findMatchingElement(target: clean, in: snapshot) {
            return (elem.center.0, elem.center.1, elem)
        }
        return (nil, nil, nil)
    }

    private static func unlockAccessibilityTree(_ elem: AXUIElement, killSwitch: KillSwitch) {
        guard !killSwitch.isPaused else { return }
        _ = AXUIElementSetAttributeValue(elem, "AXManualAccessibility" as CFString, kCFBooleanTrue)
        _ = AXUIElementSetAttributeValue(elem, "AXEnhancedUserInterface" as CFString, kCFBooleanTrue)
    }

    private static func getAXRaw(_ elem: AXUIElement, attribute: String, killSwitch: KillSwitch) -> CFTypeRef? {
        guard !killSwitch.isPaused else { return nil }
        var val: CFTypeRef?
        let err = AXUIElementCopyAttributeValue(elem, attribute as CFString, &val)
        return err == .success ? val : nil
    }

    private static func getAXString(_ elem: AXUIElement, attribute: String, killSwitch: KillSwitch) -> String {
        guard let raw = getAXRaw(elem, attribute: attribute, killSwitch: killSwitch) else { return "" }
        if let s = raw as? String { return s.trimmingCharacters(in: .whitespacesAndNewlines) }
        if let url = raw as? URL { return url.absoluteString }
        if let num = raw as? NSNumber { return num.stringValue }
        return ""
    }

    private static func getAXBool(_ elem: AXUIElement, attribute: String, defaultValue: Bool, killSwitch: KillSwitch) -> Bool {
        guard let raw = getAXRaw(elem, attribute: attribute, killSwitch: killSwitch) else { return defaultValue }
        if let b = raw as? Bool { return b }
        if let n = raw as? NSNumber { return n.boolValue }
        return defaultValue
    }

    private static func getAXElement(_ elem: AXUIElement, attribute: String, killSwitch: KillSwitch) -> AXUIElement? {
        guard let raw = getAXRaw(elem, attribute: attribute, killSwitch: killSwitch) else { return nil }
        if CFGetTypeID(raw) == AXUIElementGetTypeID() {
            return unsafeBitCast(raw, to: AXUIElement.self)
        }
        return nil
    }

    private static func getAXElementArray(_ elem: AXUIElement, attribute: String, killSwitch: KillSwitch) -> [AXUIElement] {
        guard let raw = getAXRaw(elem, attribute: attribute, killSwitch: killSwitch),
              let arr = raw as? [AnyObject]
        else { return [] }
        var out: [AXUIElement] = []
        out.reserveCapacity(arr.count)
        for item in arr where CFGetTypeID(item) == AXUIElementGetTypeID() {
            out.append(unsafeBitCast(item, to: AXUIElement.self))
        }
        return out
    }

    private static func extractBBox(_ elem: AXUIElement, killSwitch: KillSwitch) -> (Double, Double, Double, Double) {
        var pt = CGPoint.zero
        var sz = CGSize.zero
        if let posRaw = getAXRaw(elem, attribute: "AXPosition", killSwitch: killSwitch),
           CFGetTypeID(posRaw) == AXValueGetTypeID()
        {
            let axVal = unsafeBitCast(posRaw, to: AXValue.self)
            _ = AXValueGetValue(axVal, .cgPoint, &pt)
        }
        if let szRaw = getAXRaw(elem, attribute: "AXSize", killSwitch: killSwitch),
           CFGetTypeID(szRaw) == AXValueGetTypeID()
        {
            let axVal = unsafeBitCast(szRaw, to: AXValue.self)
            _ = AXValueGetValue(axVal, .cgSize, &sz)
        }
        return (Double(pt.x), Double(pt.y), Double(sz.width), Double(sz.height))
    }

    private static func findPrimaryWebAreaURL(in window: AXUIElement, killSwitch: KillSwitch) -> String {
        var queue: [(AXUIElement, Int)] = [(window, 0)]
        var idx = 0
        while idx < queue.count && idx < 48 {
            if killSwitch.isPaused { return "" }
            let (curr, depth) = queue[idx]
            idx += 1
            let role = getAXString(curr, attribute: "AXRole", killSwitch: killSwitch)
            if role == "AXWebArea" {
                let urlStr = getAXString(curr, attribute: "AXURL", killSwitch: killSwitch)
                if !urlStr.isEmpty { return urlStr }
            }
            if depth < 5 {
                for ch in getAXElementArray(curr, attribute: "AXChildren", killSwitch: killSwitch).prefix(12) {
                    queue.append((ch, depth + 1))
                }
            }
        }
        return ""
    }

    private static func findTargetWindows(
        axApp: AXUIElement,
        titleQuery: String?,
        includeFloating: Bool,
        includeStatusBar: Bool,
        killSwitch: KillSwitch
    ) -> [AXUIElement] {
        var candidates: [AXUIElement] = []
        func appendUnique(_ el: AXUIElement?) {
            guard let el else { return }
            if !candidates.contains(where: { CFEqual($0, el) }) {
                candidates.append(el)
            }
        }

        appendUnique(getAXElement(axApp, attribute: "AXFocusedWindow", killSwitch: killSwitch))
        appendUnique(getAXElement(axApp, attribute: "AXMainWindow", killSwitch: killSwitch))
        for w in getAXElementArray(axApp, attribute: "AXWindows", killSwitch: killSwitch) {
            appendUnique(w)
        }

        let extrasMB = getAXElement(axApp, attribute: "AXExtrasMenuBar", killSwitch: killSwitch)
        let menuBar = getAXElement(axApp, attribute: "AXMenuBar", killSwitch: killSwitch)
        if includeStatusBar {
            appendUnique(extrasMB)
            appendUnique(menuBar)
        }
        if includeFloating {
            for ch in getAXElementArray(axApp, attribute: "AXChildren", killSwitch: killSwitch) {
                let r = getAXString(ch, attribute: "AXRole", killSwitch: killSwitch)
                let sr = getAXString(ch, attribute: "AXSubrole", killSwitch: killSwitch)
                if ["AXWindow", "AXDialog", "AXSheet", "AXPopover", "AXMenu"].contains(r)
                    || ["AXFloatingWindow", "AXSystemDialog", "NSPanel", "AXDialog"].contains(sr)
                {
                    appendUnique(ch)
                }
            }
        }
        if candidates.isEmpty {
            appendUnique(extrasMB)
            if includeStatusBar || extrasMB == nil { appendUnique(menuBar) }
            for ch in getAXElementArray(axApp, attribute: "AXChildren", killSwitch: killSwitch) {
                appendUnique(ch)
            }
        }

        guard let qRaw = titleQuery?.trimmingCharacters(in: .whitespacesAndNewlines), !qRaw.isEmpty else {
            return candidates
        }
        let q = qRaw.lowercased()
        let matched = candidates.filter { w in
            let t = getAXString(w, attribute: "AXTitle", killSwitch: killSwitch).lowercased()
            let d = getAXString(w, attribute: "AXDocument", killSwitch: killSwitch).lowercased()
            let desc = getAXString(w, attribute: "AXDescription", killSwitch: killSwitch).lowercased()
            if t.contains(q) || d.contains(q) || desc.contains(q) { return true }
            for ch in getAXElementArray(w, attribute: "AXChildren", killSwitch: killSwitch) {
                let ct = getAXString(ch, attribute: "AXTitle", killSwitch: killSwitch).lowercased()
                let cd = getAXString(ch, attribute: "AXDescription", killSwitch: killSwitch).lowercased()
                if ct.contains(q) || cd.contains(q) { return true }
            }
            return false
        }
        return matched.isEmpty ? candidates : matched
    }

    private static func parseUIElement(
        _ elem: AXUIElement,
        depth: Int,
        isFocusedOverride: Bool? = nil,
        killSwitch: KillSwitch
    ) -> FovealElementRecord? {
        if killSwitch.isPaused { return nil }
        let role = getAXString(elem, attribute: "AXRole", killSwitch: killSwitch)
        if role.isEmpty { return nil }

        let subroleRaw = getAXString(elem, attribute: "AXSubrole", killSwitch: killSwitch)
        let subrole = subroleRaw.isEmpty ? nil : subroleRaw
        let title = getAXString(elem, attribute: "AXTitle", killSwitch: killSwitch)
        let desc = getAXString(elem, attribute: "AXDescription", killSwitch: killSwitch)

        // Check AXSecureTextField BEFORE reading AXValue so raw password characters are never copied!
        let isSecure = PrivacyBlindfold.isSecureElement(role: role, subrole: subrole, title: title)
        let valStr: String
        if isSecure {
            valStr = "[REDACTED_PASSWORD]"
        } else {
            valStr = getAXString(elem, attribute: "AXValue", killSwitch: killSwitch)
        }

        let fallbackVal = (role == "AXStaticText") ? String(valStr.prefix(120)) : ""
        let label = !title.isEmpty ? title : (!desc.isEmpty ? desc : (!fallbackVal.isEmpty ? fallbackVal : (subrole ?? "")))
        let focused = isFocusedOverride ?? getAXBool(elem, attribute: "AXFocused", defaultValue: false, killSwitch: killSwitch)
        let enabled = getAXBool(elem, attribute: "AXEnabled", defaultValue: true, killSwitch: killSwitch)
        let identRaw = getAXString(elem, attribute: "AXIdentifier", killSwitch: killSwitch)
        let bbox = extractBBox(elem, killSwitch: killSwitch)
        let center: (Double, Double) = (bbox != (0, 0, 0, 0))
            ? (bbox.0 + bbox.2 / 2.0, bbox.1 + bbox.3 / 2.0)
            : (0.0, 0.0)

        return FovealElementRecord(
            id: "",
            role: role,
            title: String((title.isEmpty ? label : title).prefix(120)),
            label: String(label.prefix(120)),
            description: String(desc.prefix(120)),
            value: valStr.isEmpty ? nil : String(valStr.prefix(2000)),
            bbox: bbox,
            center: center,
            focused: focused,
            enabled: enabled,
            subrole: subrole,
            identifier: identRaw.isEmpty ? nil : identRaw,
            depth: depth,
            isSecure: isSecure,
            rawElement: elem
        )
    }

    private static func traverseWindows(
        _ wins: [AXUIElement],
        maxElements: Int,
        maxDepth: Int,
        filterQuery: String?,
        killSwitch: KillSwitch
    ) -> [FovealElementRecord] {
        var discovered: [FovealElementRecord] = []
        var queue: [(AXUIElement, Int)] = wins.map { ($0, 0) }
        var qIdx = 0
        var seen = Set<String>()
        let budgetMs = Double(ProcessInfo.processInfo.environment["WBC_FOVEAL_TRAVERSE_BUDGET_MS"] ?? "350") ?? 350.0
        let deadline = CFAbsoluteTimeGetCurrent() + max(0.05, budgetMs / 1000.0)
        let fqLow = filterQuery?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()

        while qIdx < queue.count && discovered.count < maxElements {
            if killSwitch.isPaused { break }
            if qIdx % 8 == 0 && CFAbsoluteTimeGetCurrent() >= deadline { break }
            let (curr, depth) = queue[qIdx]
            qIdx += 1
            if depth > maxDepth { continue }

            if depth <= 4 {
                unlockAccessibilityTree(curr, killSwitch: killSwitch)
            } else {
                let r = getAXString(curr, attribute: "AXRole", killSwitch: killSwitch)
                if ["AXWebArea", "AXGroup", "AXScrollArea", "AXSplitGroup"].contains(r) {
                    unlockAccessibilityTree(curr, killSwitch: killSwitch)
                }
            }

            if let parsed = parseUIElement(curr, depth: depth, killSwitch: killSwitch) {
                let shimNames: Set<String> = ["AXEmptyGroup", "AXApplicationStatus", "AXLandmarkMain", "Notifications"]
                let isShim = shimNames.contains(parsed.title)
                    || shimNames.contains(parsed.label)
                    || (parsed.bbox.2 > 0 && parsed.bbox.2 <= 3.0)
                    || (parsed.bbox.3 > 0 && parsed.bbox.3 <= 3.0)
                let isInteractive = !isShim && (
                    interactiveAXRoles.contains(parsed.role)
                    || (["AXGroup", "AXGenericElement", "AXImage", "AXRadioGroup"].contains(parsed.role)
                        && (!parsed.label.isEmpty || !parsed.description.isEmpty || parsed.value != nil))
                )
                if isInteractive {
                    var hasText = !parsed.label.isEmpty || parsed.value != nil || parsed.focused
                    if ["AXStaticText", "AXHeading"].contains(parsed.role)
                        && parsed.label.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    {
                        hasText = false
                    }
                    var matchesFilter = true
                    if let fqLow, !fqLow.isEmpty {
                        let hay = "\(parsed.role) \(parsed.label) \(parsed.title) \(parsed.value ?? "") \(parsed.description)".lowercased()
                        matchesFilter = hay.contains(fqLow)
                    }
                    let dedupKey = "\(parsed.role)|\(parsed.label)|\(Int(parsed.bbox.0)),\(Int(parsed.bbox.1)),\(Int(parsed.bbox.2)),\(Int(parsed.bbox.3))"
                    if hasText && matchesFilter && !seen.contains(dedupKey) {
                        seen.insert(dedupKey)
                        discovered.append(parsed)
                    }
                }
            }

            var children = getAXElementArray(curr, attribute: "AXChildren", killSwitch: killSwitch)
            if children.isEmpty {
                children = getAXElementArray(curr, attribute: "AXVisibleChildren", killSwitch: killSwitch)
            }
            if children.isEmpty {
                children = getAXElementArray(curr, attribute: "AXContents", killSwitch: killSwitch)
            }
            for ch in children {
                if discovered.count >= maxElements { break }
                queue.append((ch, depth + 1))
            }
        }
        return discovered
    }

    private static func isEmptyOrOnlyWindowControls(_ elems: [FovealElementRecord]) -> Bool {
        if elems.isEmpty { return true }
        let meaningful = elems.filter { el in
            el.role != "AXWindow" && !ignoreTextSubroles.contains(el.subrole ?? "")
        }
        if meaningful.isEmpty { return true }
        let shellRoles: Set<String> = ["AXGroup", "AXSplitGroup", "AXUnknown", "AXScrollArea"]
        return meaningful.allSatisfy { shellRoles.contains($0.role) }
    }

    private static func loadCustomBlindfoldOrigins() -> [String] {
        let path = (NSHomeDirectory() as NSString).appendingPathComponent(".whole_body_compute/blindfold_origins.json")
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let arr = try? JSONSerialization.jsonObject(with: data) as? [String]
        else { return [] }
        return arr
    }

    // MARK: - Headless Test Fixture Mode (Used Only Under `--headless-test-mode`)

    private static func handleMockDOMRequest(
        req: [String: Any],
        appNameArg: String?,
        titleQuery: String?,
        filterQuery: String?,
        maxElements: Int,
        hasActuation: Bool,
        click: String?,
        typeText: String?,
        pasteText: String?,
        readLast: Bool,
        waitReply: Double?,
        killSwitch: KillSwitch,
        t0: CFAbsoluteTime
    ) -> [String: Any] {
        if killSwitch.isPaused {
            return killSwitchErrorPayload()
        }
        let targetApp = appNameArg ?? ProcessInfo.processInfo.environment["WBC_EYES_MOCK_APP"] ?? "Finder"
        let targetBundle = ProcessInfo.processInfo.environment["WBC_EYES_MOCK_BUNDLE_ID"] ?? "com.apple.finder"
        let targetWin = titleQuery ?? ProcessInfo.processInfo.environment["WBC_EYES_MOCK_WINDOW_TITLE"] ?? "Workspace"
        let targetURL = ProcessInfo.processInfo.environment["WBC_EYES_MOCK_WEB_URL"] ?? ""

        let appCheck = PrivacyBlindfold.checkApplication(appName: targetApp, bundleID: targetBundle)
        if appCheck.blocked {
            if hasActuation {
                let secErr: [String: Any] = [
                    "status": "error",
                    "action": "actuate",
                    "method": "blocked_security_gate",
                    "security_blocked": true,
                    "cursor_moved": false,
                    "error": "Security violation: \(appCheck.reason)",
                    "summary": "Blocked actuate: \(appCheck.reason)"
                ]
                var snap = FovealSnapshotRecord(
                    appName: targetApp,
                    bundleID: targetBundle,
                    pid: 9998,
                    windowTitle: appCheck.placeholderTitle,
                    windowBBox: (0, 0, 0, 0),
                    focusedElement: nil,
                    interactiveElements: [],
                    timestamp: Date().timeIntervalSince1970,
                    latencyMs: 0.8,
                    headlessRoute: appCheck.reason,
                    filterQuery: filterQuery
                )
                snap.actuation = [
                    "status": "error",
                    "security_blocked": true,
                    "actions": [secErr],
                    "error": secErr["error"]!,
                    "summary": secErr["summary"]!,
                    "tactile_delta": ["verified": false, "security_blocked": true]
                ]
                return snap.toDictionary()
            }
            return FovealSnapshotRecord(
                appName: targetApp,
                bundleID: targetBundle,
                pid: 9998,
                windowTitle: appCheck.placeholderTitle,
                windowBBox: (0, 0, 0, 0),
                focusedElement: nil,
                interactiveElements: [],
                timestamp: Date().timeIntervalSince1970,
                latencyMs: 0.8,
                headlessRoute: appCheck.reason,
                filterQuery: filterQuery
            ).toDictionary()
        }

        let originCheck = PrivacyBlindfold.checkWindowOrigin(
            windowTitle: targetWin,
            webAreaURL: targetURL
        )
        if originCheck.blocked {
            var snap = FovealSnapshotRecord(
                appName: targetApp,
                bundleID: targetBundle,
                pid: 9998,
                windowTitle: originCheck.placeholderTitle,
                windowBBox: (0, 0, 0, 0),
                focusedElement: nil,
                interactiveElements: [],
                timestamp: Date().timeIntervalSince1970,
                latencyMs: 0.8,
                headlessRoute: originCheck.reason,
                filterQuery: filterQuery
            )
            if hasActuation {
                snap.actuation = [
                    "status": "error",
                    "security_blocked": true,
                    "error": "Security violation: \(originCheck.reason)",
                    "summary": "Blocked actuate: \(originCheck.reason)",
                    "tactile_delta": ["verified": false, "security_blocked": true]
                ]
            }
            return snap.toDictionary()
        }

        let runBtn = FovealElementRecord(
            id: "e1",
            role: "AXButton",
            title: "Run Tests",
            label: "Run Tests",
            description: "Execute test suite",
            value: nil,
            bbox: (120, 45, 80, 28),
            center: (160, 59),
            focused: false,
            enabled: true,
            subrole: nil,
            identifier: "btn-run",
            depth: 2,
            isSecure: false,
            rawElement: nil
        )
        let msgInput = FovealElementRecord(
            id: "e2",
            role: "AXTextArea",
            title: "Message input",
            label: "Message input",
            description: "Prompt composer",
            value: "Ready",
            bbox: (300, 700, 600, 44),
            center: (600, 722),
            focused: true,
            enabled: true,
            subrole: nil,
            identifier: "composer",
            depth: 2,
            isSecure: false,
            rawElement: nil
        )
        let pwInput = FovealElementRecord(
            id: "e3",
            role: "AXSecureTextField",
            title: "Master Password",
            label: "Master Password",
            description: "Password field",
            value: "[REDACTED_PASSWORD]",
            bbox: (300, 760, 300, 32),
            center: (450, 776),
            focused: false,
            enabled: true,
            subrole: "AXSecureTextField",
            identifier: "vault-pw",
            depth: 2,
            isSecure: true,
            rawElement: nil
        )

        var elems = [runBtn, msgInput, pwInput]
        if let fq = filterQuery?.lowercased(), !fq.isEmpty {
            elems = elems.filter {
                "\($0.role) \($0.label) \($0.description)".lowercased().contains(fq)
            }
        }
        elems = Array(elems.prefix(maxElements))

        var snap = FovealSnapshotRecord(
            appName: targetApp,
            bundleID: targetBundle,
            pid: 4242,
            windowTitle: targetWin,
            windowBBox: (0, 0, 1440, 900),
            focusedElement: msgInput,
            interactiveElements: elems,
            timestamp: Date().timeIntervalSince1970,
            latencyMs: (CFAbsoluteTimeGetCurrent() - t0) * 1000.0 + 0.5,
            headlessRoute: "",
            filterQuery: filterQuery
        )

        if hasActuation {
            if (typeText != nil || pasteText != nil), let click, click.lowercased() == "e3" {
                let secErr: [String: Any] = [
                    "status": "error",
                    "action": typeText != nil ? "type_text" : "paste",
                    "method": "blocked_secure_field",
                    "security_blocked": true,
                    "cursor_moved": false,
                    "error": "Security violation: Forbidden input into secure password field (AXSecureTextField)",
                    "summary": "Blocked input: secure password field"
                ]
                snap.actuation = [
                    "status": "error",
                    "security_blocked": true,
                    "actions": [secErr],
                    "error": secErr["error"]!,
                    "summary": secErr["summary"]!,
                    "tactile_delta": ["verified": false, "security_blocked": true]
                ]
            } else {
                let delta: [String: Any] = [
                    "verified": true,
                    "elements_added_count": 0,
                    "elements_removed_count": 0,
                    "added_labels": [String](),
                    "removed_labels": [String](),
                    "value_changed": typeText != nil || pasteText != nil,
                    "focus_changed": click != nil,
                    "window_title_changed": false,
                    "post_element_count": elems.count
                ]
                snap.actuation = [
                    "status": "success",
                    "actions": [["status": "success", "method": "AXPress", "summary": "Executed via WBC Eyes.app"]],
                    "summary": "Executed via WBC Eyes.app",
                    "tactile_delta": delta
                ]
                snap.tactileDelta = delta
            }
        }

        if readLast || waitReply != nil {
            snap.lastMessages = ["Ready"]
            snap.lastMessage = "Ready"
            snap.replySettled = true
            snap.waitReplyMs = 1.0
        }

        return snap.toDictionary()
    }

    // MARK: - Primitive Value Coercion

    private static func boolValue(_ raw: Any?) -> Bool {
        if let b = raw as? Bool { return b }
        if let n = raw as? NSNumber { return n.boolValue }
        return false
    }

    private static func nonEmptyString(_ raw: Any?) -> String? {
        guard let s = raw as? String else { return nil }
        let trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}
