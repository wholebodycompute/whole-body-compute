import AppKit
import ApplicationServices
import AVFoundation
import Foundation

/// Unified `Whole Body Compute.app` menu-bar `NSStatusItem` controller displaying live
/// Eyes & Hands + Ears status with 1-click Kill Switches (`⌥`-click pauses/resumes all organs).
@MainActor
public final class StatusBarController: NSObject {
    private let statusItem: NSStatusItem
    private let killSwitch: KillSwitch
    private let menu = NSMenu()

    private let toggleEyesMenuItem = NSMenuItem()
    private let toggleEarsMenuItem = NSMenuItem()
    private let toggleAllMenuItem = NSMenuItem()
    private let eyesStateItem = NSMenuItem()
    private let earsStateItem = NSMenuItem()
    private let grantAXMenuItem = NSMenuItem()
    private let statsMenuItem = NSMenuItem()
    private let lastCallerMenuItem = NSMenuItem()

    private var observerID: UUID?
    private var axPollTimer: Timer?

    public init(killSwitch: KillSwitch = .shared) {
        self.killSwitch = killSwitch
        self.statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        super.init()
        configureStatusButton()
        configureMenu()
        refreshUI(snapshot: killSwitch.snapshot())

        observerID = killSwitch.addObserver { [weak self] snap in
            Task { @MainActor in
                self?.refreshUI(snapshot: snap)
            }
        }

        axPollTimer = Timer.scheduledTimer(withTimeInterval: 3.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                self.refreshUI(snapshot: self.killSwitch.snapshot())
            }
        }
    }

    private func configureStatusButton() {
        guard let button = statusItem.button else { return }
        button.imagePosition = .imageLeading
        button.target = self
        button.action = #selector(handleStatusBarButtonClick(_:))
        button.sendAction(on: [.leftMouseUp, .rightMouseUp])
        button.toolTip = "Whole Body Compute - Eyes, Hands & Ears TCC Broker (Option-click for 1-Click Kill Switch)"
    }

    private func configureMenu() {
        toggleAllMenuItem.target = self
        toggleAllMenuItem.action = #selector(toggleAllOrgans(_:))
        menu.addItem(toggleAllMenuItem)

        menu.addItem(.separator())

        toggleEyesMenuItem.target = self
        toggleEyesMenuItem.action = #selector(toggleEyes(_:))
        toggleEyesMenuItem.keyEquivalent = "k"
        toggleEyesMenuItem.keyEquivalentModifierMask = [.command, .shift]
        menu.addItem(toggleEyesMenuItem)

        toggleEarsMenuItem.target = self
        toggleEarsMenuItem.action = #selector(toggleEars(_:))
        toggleEarsMenuItem.keyEquivalent = "m"
        toggleEarsMenuItem.keyEquivalentModifierMask = [.command, .shift]
        menu.addItem(toggleEarsMenuItem)

        menu.addItem(.separator())

        eyesStateItem.isEnabled = false
        menu.addItem(eyesStateItem)

        earsStateItem.isEnabled = false
        menu.addItem(earsStateItem)

        grantAXMenuItem.title = "Grant Accessibility Permission..."
        grantAXMenuItem.target = self
        grantAXMenuItem.action = #selector(promptAccessibilityPermission(_:))
        menu.addItem(grantAXMenuItem)

        menu.addItem(.separator())

        statsMenuItem.isEnabled = false
        menu.addItem(statsMenuItem)

        lastCallerMenuItem.isEnabled = false
        menu.addItem(lastCallerMenuItem)

        menu.addItem(.separator())

        let quitItem = NSMenuItem(title: "Quit Whole Body Compute", action: #selector(quitApp(_:)), keyEquivalent: "q")
        quitItem.target = self
        menu.addItem(quitItem)
    }

    @objc private func handleStatusBarButtonClick(_ sender: NSStatusBarButton) {
        let event = NSApp.currentEvent
        let isOptionClick = event?.modifierFlags.contains(.option) == true
        if isOptionClick {
            _ = killSwitch.toggleAll()
            return
        }
        refreshUI(snapshot: killSwitch.snapshot())
        statusItem.menu = menu
        statusItem.button?.performClick(nil)
        statusItem.menu = nil
    }

    @objc private func toggleAllOrgans(_ sender: Any?) {
        _ = killSwitch.toggleAll()
    }

    @objc private func toggleEyes(_ sender: Any?) {
        _ = killSwitch.toggle()
    }

    @objc private func toggleEars(_ sender: Any?) {
        _ = killSwitch.toggleEars()
    }

    @objc private func promptAccessibilityPermission(_ sender: Any?) {
        let opts = ["AXTrustedCheckOptionPrompt": kCFBooleanTrue!] as CFDictionary
        _ = AXIsProcessTrustedWithOptions(opts)
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility") {
            NSWorkspace.shared.open(url)
        }
    }

    @objc private func quitApp(_ sender: Any?) {
        NSApp.terminate(nil)
    }

    public func refreshUI(snapshot: KillSwitch.Snapshot) {
        let axTrusted = AXIsProcessTrusted()
        let micAuthorized = AVCaptureDevice.authorizationStatus(for: .audio) == .authorized
        let anyPaused = snapshot.isPaused || snapshot.isEarsPaused
        let bothPaused = snapshot.isPaused && snapshot.isEarsPaused

        if let button = statusItem.button {
            button.image = Self.makeMenuBarIcon(paused: snapshot.isPaused)
            if bothPaused {
                button.title = " Paused"
            } else if snapshot.isPaused {
                button.title = " Eyes Off"
            } else if snapshot.isEarsPaused {
                button.title = " Mic Off"
            } else if !axTrusted {
                button.title = " !"
            } else {
                button.title = ""
            }
        }

        toggleAllMenuItem.title = bothPaused
            ? "Resume All Organs (Option-Click Icon)"
            : (anyPaused ? "Pause All Organs (Option-Click Icon)" : "Pause All Organs - 1-Click Kill Switch")

        toggleEyesMenuItem.title = snapshot.isPaused
            ? "Resume Eyes & Hands (Paused)"
            : "Pause Eyes & Hands"

        toggleEarsMenuItem.title = snapshot.isEarsPaused
            ? "Resume Ears - Hold-Fn Dictation (Paused)"
            : "Pause Ears - Hold-Fn Dictation"

        if snapshot.isPaused {
            eyesStateItem.title = "Eyes & Hands: PAUSED - Refusing AX & Clicks"
        } else if axTrusted {
            eyesStateItem.title = "Eyes & Hands: Active (Sole TCC Accessibility Holder)"
        } else {
            eyesStateItem.title = "Eyes & Hands: Accessibility Permission Required"
        }

        if snapshot.isEarsPaused {
            earsStateItem.title = "Ears (Hold-Fn): PAUSED - Microphone Disabled"
        } else if micAuthorized {
            earsStateItem.title = "Ears (Hold-Fn): Ready (100% Local ANE)"
        } else {
            earsStateItem.title = "Ears (Hold-Fn): Standby (Prompts Mic on First Use)"
        }

        grantAXMenuItem.isHidden = axTrusted
        statsMenuItem.title = "Requests Served: \(snapshot.requestsServed)  •  Blocked: \(snapshot.requestsBlocked)"
        lastCallerMenuItem.title = "Last Caller: \(snapshot.lastCallerSummary)"
    }

    /// Renders the canonical Whole Body Compute monochrome menu-bar icon (`assets/wbc-menu-bar-icon.svg`).
    public static func makeMenuBarIcon(paused: Bool) -> NSImage {
        let size = NSSize(width: 18, height: 18)
        let image = NSImage(size: size, flipped: true) { rect in
            let scale = min(rect.width / 512.0, rect.height / 512.0)
            guard let ctx = NSGraphicsContext.current?.cgContext else { return false }
            ctx.saveGState()
            ctx.scaleBy(x: scale, y: scale)
            ctx.setStrokeColor(NSColor.black.withAlphaComponent(paused ? 0.45 : 1.0).cgColor)
            ctx.setLineCap(.round)
            ctx.setLineJoin(.round)

            func strokeLine(_ x1: CGFloat, _ y1: CGFloat, _ x2: CGFloat, _ y2: CGFloat, width: CGFloat) {
                ctx.setLineWidth(width)
                ctx.beginPath()
                ctx.move(to: CGPoint(x: x1, y: y1))
                ctx.addLine(to: CGPoint(x: x2, y: y2))
                ctx.strokePath()
            }

            // Broadcast crown arcs
            for (r, cy, w): (CGFloat, CGFloat, CGFloat) in [(94, 152, 22), (63, 172, 20), (35, 192, 19)] {
                ctx.setLineWidth(w)
                ctx.beginPath()
                ctx.addArc(center: CGPoint(x: 256, y: cy), radius: r, startAngle: .pi * 1.15, endAngle: .pi * 1.85, clockwise: false)
                ctx.strokePath()
            }

            // Cyber slit eyes
            strokeLine(208, 220, 238, 220, width: 18)
            strokeLine(274, 220, 304, 220, width: 18)

            // Studio head U-cradle
            ctx.setLineWidth(22)
            ctx.beginPath()
            ctx.move(to: CGPoint(x: 146, y: 186))
            ctx.addCurve(to: CGPoint(x: 256, y: 304), control1: CGPoint(x: 146, y: 282), control2: CGPoint(x: 188, y: 304))
            ctx.addCurve(to: CGPoint(x: 366, y: 186), control1: CGPoint(x: 324, y: 304), control2: CGPoint(x: 366, y: 282))
            ctx.strokePath()

            // Neck & rounded torso
            strokeLine(256, 304, 256, 326, width: 22)
            let torso = CGRect(x: 180, y: 326, width: 152, height: 104)
            let torsoPath = CGPath(roundedRect: torso, cornerWidth: 28, cornerHeight: 28, transform: nil)
            ctx.setLineWidth(22)
            ctx.addPath(torsoPath)
            ctx.strokePath()

            // Direct chest ECG waveform
            ctx.setLineWidth(18)
            ctx.beginPath()
            ctx.move(to: CGPoint(x: 204, y: 378))
            for pt in [CGPoint(x: 232, y: 378), CGPoint(x: 245, y: 346), CGPoint(x: 258, y: 410), CGPoint(x: 271, y: 360), CGPoint(x: 282, y: 378), CGPoint(x: 308, y: 378)] {
                ctx.addLine(to: pt)
            }
            ctx.strokePath()

            // Arms & legs
            strokeLine(180, 356, 132, 412, width: 20)
            strokeLine(332, 356, 380, 412, width: 20)
            strokeLine(222, 430, 222, 466, width: 20)
            strokeLine(290, 430, 290, 466, width: 20)
            strokeLine(184, 466, 244, 466, width: 20)
            strokeLine(268, 466, 328, 466, width: 20)

            // Diagonal slash overlay when Kill Switch is paused
            if paused {
                ctx.setStrokeColor(NSColor.black.cgColor)
                strokeLine(76, 436, 436, 76, width: 34)
            }

            ctx.restoreGState()
            return true
        }
        image.isTemplate = true
        return image
    }
}
