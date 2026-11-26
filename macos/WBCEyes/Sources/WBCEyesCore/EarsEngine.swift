import AppKit
import ApplicationServices
import AVFoundation
import CoreGraphics
import Darwin
import Foundation
import Speech

/// Native Swift Apple Neural Engine Speech-to-Text (`SFSpeechRecognizer`, `requiresOnDeviceRecognition = true`)
/// and listen-only `CGEventTap` hotkey streamer inside notarized `Whole Body Compute.app`.
/// Eliminates any need for Python or ad-hoc helper `.app` wrappers to hold Microphone or Accessibility TCC.
public enum EarsEngine {

    public static func defaultEarsPausedMarkerPath() -> String {
        let env = ProcessInfo.processInfo.environment
        if let explicit = env["WBC_EARS_PAUSED_PATH"], !explicit.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return (explicit as NSString).expandingTildeInPath
        }
        if let dataDir = env["WBC_DATA_DIR"], !dataDir.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return ((dataDir as NSString).expandingTildeInPath as NSString).appendingPathComponent("ears.paused")
        }
        return (NSHomeDirectory() as NSString).appendingPathComponent(".whole_body_compute/ears.paused")
    }

    public static func isEarsPausedOnDisk() -> Bool {
        FileManager.default.fileExists(atPath: defaultEarsPausedMarkerPath())
    }

    public static func persistEarsPausedState(_ paused: Bool) {
        let path = defaultEarsPausedMarkerPath()
        let parent = (path as NSString).deletingLastPathComponent
        if paused {
            try? FileManager.default.createDirectory(atPath: parent, withIntermediateDirectories: true, attributes: [
                .posixPermissions: NSNumber(value: Int16(0o700))
            ])
            FileManager.default.createFile(atPath: path, contents: Data("1\n".utf8), attributes: [
                .posixPermissions: NSNumber(value: Int16(0o600))
            ])
            _ = chmod(path, mode_t(0o600))
        } else {
            try? FileManager.default.removeItem(atPath: path)
        }
    }

    private static func emitJSONLine(_ obj: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: obj, options: []),
              let line = String(data: data, encoding: .utf8)
        else { return }
        FileHandle.standardOutput.write(Data((line + "\n").utf8))
        fflush(stdout)
    }

    /// Execute `--ears-mic` or `--ears-file <wav>` using 100% on-device ANE `SFSpeechRecognizer`.
    public static func runSpeechCLI(arguments: [String]) -> Never {
        var mode = "mic"
        var filePath: String?
        var contextStrings: [String] = []
        var silenceSec: Double = 1.4
        var maxSec: Double = 30.0

        var idx = 1
        while idx < arguments.count {
            switch arguments[idx] {
            case "--ears-file", "--file":
                if idx + 1 < arguments.count {
                    mode = "file"
                    filePath = arguments[idx + 1]
                    idx += 1
                }
            case "--ears-mic", "--mic":
                mode = "mic"
            case "--context-json":
                if idx + 1 < arguments.count,
                   let data = arguments[idx + 1].data(using: .utf8),
                   let arr = try? JSONSerialization.jsonObject(with: data) as? [String]
                {
                    contextStrings = arr
                    idx += 1
                }
            case "--silence":
                if idx + 1 < arguments.count, let v = Double(arguments[idx + 1]) {
                    silenceSec = v
                    idx += 1
                }
            case "--max-seconds":
                if idx + 1 < arguments.count, let v = Double(arguments[idx + 1]) {
                    maxSec = v
                    idx += 1
                }
            default:
                break
            }
            idx += 1
        }

        if mode == "mic" && isEarsPausedOnDisk() {
            emitJSONLine(["error": "ears_paused_by_kill_switch"])
            exit(5)
        }

        guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US")),
              recognizer.supportsOnDeviceRecognition
        else {
            emitJSONLine(["error": "on_device_speech_unavailable"])
            exit(2)
        }

        let authSem = DispatchSemaphore(value: 0)
        var authStatus = SFSpeechRecognizer.authorizationStatus()
        if authStatus == .notDetermined {
            SFSpeechRecognizer.requestAuthorization { st in
                authStatus = st
                authSem.signal()
            }
            _ = authSem.wait(timeout: .now() + 10.0)
        }
        guard authStatus == .authorized else {
            emitJSONLine(["error": "speech_permission_denied"])
            exit(3)
        }

        if mode == "file", let fp = filePath {
            let req = SFSpeechURLRecognitionRequest(url: URL(fileURLWithPath: fp))
            req.requiresOnDeviceRecognition = true
            if !contextStrings.isEmpty {
                req.contextualStrings = Array(contextStrings.prefix(100))
            }
            let done = DispatchSemaphore(value: 0)
            var finalOut = ""
            var errOut: String?
            _ = recognizer.recognitionTask(with: req) { res, err in
                if let res {
                    finalOut = res.bestTranscription.formattedString
                    if res.isFinal { done.signal() }
                }
                if let err {
                    if finalOut.isEmpty { errOut = err.localizedDescription }
                    done.signal()
                }
            }
            _ = done.wait(timeout: .now() + 60.0)
            if let errOut, finalOut.isEmpty {
                emitJSONLine(["error": errOut])
                exit(1)
            }
            emitJSONLine(["type": "final", "text": finalOut, "backend": "ane_sfspeech"])
            exit(0)
        }

        // Live RAM-only microphone capture (zero audio written to disk)
        let micSem = DispatchSemaphore(value: 0)
        var micGranted = false
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            micGranted = true
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .audio) { ok in
                micGranted = ok
                micSem.signal()
            }
            _ = micSem.wait(timeout: .now() + 10.0)
        default:
            micGranted = false
        }
        guard micGranted else {
            emitJSONLine(["error": "microphone_permission_denied"])
            exit(3)
        }

        let engine = AVAudioEngine()
        let req = SFSpeechAudioBufferRecognitionRequest()
        req.requiresOnDeviceRecognition = true
        req.shouldReportPartialResults = true
        if !contextStrings.isEmpty {
            req.contextualStrings = Array(contextStrings.prefix(100))
        }

        let inputNode = engine.inputNode
        let format = inputNode.outputFormat(forBus: 0)
        var bestText = ""
        var lastSpeechTs = Date().timeIntervalSince1970
        var heardSpeech = false
        let startTs = lastSpeechTs

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { buf, _ in
            req.append(buf)
            if let ch = buf.floatChannelData?[0] {
                let frames = Int(buf.frameLength)
                var sumSq: Float = 0
                for frameIdx in 0..<frames { sumSq += ch[frameIdx] * ch[frameIdx] }
                let rms = sqrt(sumSq / max(1, Float(frames)))
                if rms > 0.015 {
                    heardSpeech = true
                    lastSpeechTs = Date().timeIntervalSince1970
                }
            }
        }

        signal(SIGINT, SIG_IGN)
        signal(SIGTERM, SIG_IGN)
        let sigSrc = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global(qos: .userInteractive))
        let termSrc = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .global(qos: .userInteractive))

        var finished = false
        var lastEmittedPartial = ""
        func emitAndExit() {
            if finished { return }
            finished = true
            emitJSONLine(["type": "final", "text": bestText, "backend": "ane_sfspeech"])
            exit(0)
        }

        func finishMic() {
            engine.stop()
            inputNode.removeTap(onBus: 0)
            req.endAudio()
            DispatchQueue.global().asyncAfter(deadline: .now() + 1.8) {
                emitAndExit()
            }
        }

        sigSrc.setEventHandler { finishMic() }
        termSrc.setEventHandler { finishMic() }
        sigSrc.resume()
        termSrc.resume()

        do {
            engine.prepare()
            try engine.start()
        } catch {
            emitJSONLine(["error": "audio_engine_start_failed: \(error.localizedDescription)"])
            exit(4)
        }

        _ = recognizer.recognitionTask(with: req) { res, err in
            if let res {
                bestText = res.bestTranscription.formattedString
                if res.isFinal {
                    emitAndExit()
                } else if !bestText.isEmpty && bestText != lastEmittedPartial {
                    lastEmittedPartial = bestText
                    emitJSONLine(["type": "partial", "text": bestText, "backend": "ane_sfspeech"])
                }
            } else if err != nil {
                emitAndExit()
            }
        }

        DispatchQueue.global().async {
            while !finished {
                Thread.sleep(forTimeInterval: 0.1)
                if isEarsPausedOnDisk() {
                    finishMic()
                    break
                }
                let now = Date().timeIntervalSince1970
                if (heardSpeech && (now - lastSpeechTs) >= silenceSec) || (now - startTs) >= maxSec {
                    finishMic()
                    break
                }
            }
        }

        RunLoop.main.run()
        exit(0)
    }

    /// Stream listen-only modifier/Space key events for Hold-Fn / Opt+Space dictation so Python needs zero TCC.
    public static func runHotkeyStreamCLI() -> Never {
        let mask: CGEventMask = (1 << CGEventType.flagsChanged.rawValue) | (1 << CGEventType.keyDown.rawValue)

        final class TapBox: @unchecked Sendable {
            var tap: CFMachPort?
        }
        let box = TapBox()
        let refcon = Unmanaged.passRetained(box).toOpaque()

        guard let tap = CGEvent.tapCreate(
            tap: .cgSessionEventTap,
            place: .headInsertEventTap,
            options: .listenOnly,
            eventsOfInterest: mask,
            callback: { _, type, event, userInfo in
                if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
                    if let userInfo {
                        let b = Unmanaged<TapBox>.fromOpaque(userInfo).takeUnretainedValue()
                        if let t = b.tap {
                            CGEvent.tapEnable(tap: t, enable: true)
                        }
                    }
                    return Unmanaged.passUnretained(event)
                }
                if EarsEngine.isEarsPausedOnDisk() {
                    return Unmanaged.passUnretained(event)
                }
                let flagsRaw = UInt64(event.flags.rawValue)
                if type == .flagsChanged {
                    EarsEngine.emitJSONLine(["event": "flags", "flags": flagsRaw])
                } else if type == .keyDown {
                    let keycode = event.getIntegerValueField(.keyboardEventKeycode)
                    // Only emit Space (49) modifier chords - never stream arbitrary user keystrokes!
                    if keycode == 49 {
                        EarsEngine.emitJSONLine(["event": "keydown", "keycode": keycode, "flags": flagsRaw])
                    }
                }
                return Unmanaged.passUnretained(event)
            },
            userInfo: refcon
        ) else {
            emitJSONLine(["error": "event_tap_unavailable"])
            exit(2)
        }

        box.tap = tap
        let rlSource = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
        CFRunLoopAddSource(CFRunLoopGetCurrent(), rlSource, .commonModes)
        CGEvent.tapEnable(tap: tap, enable: true)
        emitJSONLine(["event": "ready"])
        CFRunLoopRun()
        exit(0)
    }

    /// Type differential dictation backspaces + Unicode text into the focused app, refusing vaults/secure fields.
    public static func typeDifferentialDictation(backspaces: Int, text: String, headlessTestMode: Bool) -> [String: Any] {
        if headlessTestMode {
            return ["status": "ok", "typed": text.count, "backspaces": backspaces]
        }
        if let front = NSWorkspace.shared.frontmostApplication {
            let appName = front.localizedName ?? ""
            let bundleID = front.bundleIdentifier ?? ""
            let appDec = PrivacyBlindfold.checkApplication(appName: appName, bundleID: bundleID)
            if appDec.blocked {
                return [
                    "status": "error",
                    "security_blocked": true,
                    "error": "PrivacyBlindfold blocked dictation typing into protected surface (\(appName))."
                ]
            }
            let appElem = AXUIElementCreateApplication(front.processIdentifier)
            var focusedRef: CFTypeRef?
            if AXUIElementCopyAttributeValue(appElem, kAXFocusedUIElementAttribute as CFString, &focusedRef) == .success,
               let focusedElem = focusedRef
            {
                let axElem = focusedElem as! AXUIElement
                var roleRef: CFTypeRef?
                var subroleRef: CFTypeRef?
                _ = AXUIElementCopyAttributeValue(axElem, kAXRoleAttribute as CFString, &roleRef)
                _ = AXUIElementCopyAttributeValue(axElem, kAXSubroleAttribute as CFString, &subroleRef)
                let roleStr = (roleRef as? String) ?? ""
                let subroleStr = subroleRef as? String
                if PrivacyBlindfold.isSecureElement(role: roleStr, subrole: subroleStr) {
                    return [
                        "status": "error",
                        "security_blocked": true,
                        "error": "PrivacyBlindfold blocked dictation typing into AXSecureTextField."
                    ]
                }
            }
        }

        guard let src = CGEventSource(stateID: .hidSystemState) else {
            return ["status": "error", "error": "CGEventSource unavailable"]
        }

        if backspaces > 0 {
            let clamped = min(backspaces, 512)
            for _ in 0..<clamped {
                if let down = CGEvent(keyboardEventSource: src, virtualKey: 51, keyDown: true),
                   let up = CGEvent(keyboardEventSource: src, virtualKey: 51, keyDown: false)
                {
                    down.flags = []
                    up.flags = []
                    down.post(tap: .cghidEventTap)
                    up.post(tap: .cghidEventTap)
                }
            }
        }

        if !text.isEmpty {
            let utf16Units = Array(text.utf16)
            let chunkSize = 16
            var offset = 0
            while offset < utf16Units.count {
                let end = min(offset + chunkSize, utf16Units.count)
                var slice = Array(utf16Units[offset..<end])
                if let down = CGEvent(keyboardEventSource: src, virtualKey: 0, keyDown: true),
                   let up = CGEvent(keyboardEventSource: src, virtualKey: 0, keyDown: false)
                {
                    down.flags = []
                    up.flags = []
                    down.keyboardSetUnicodeString(stringLength: slice.count, unicodeString: &slice)
                    up.keyboardSetUnicodeString(stringLength: slice.count, unicodeString: &slice)
                    down.post(tap: .cghidEventTap)
                    up.post(tap: .cghidEventTap)
                }
                offset = end
            }
        }

        return ["status": "ok", "typed": text.count, "backspaces": backspaces]
    }
}
