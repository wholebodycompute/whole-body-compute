import AppKit
import Foundation
import WBCEyesCore

final class WholeBodyComputeAppDelegate: NSObject, NSApplicationDelegate {
    private var socketServer: SocketServer?
    private var statusBarController: StatusBarController?
    private var sigtermSource: DispatchSourceSignal?
    private var sigintSource: DispatchSourceSignal?

    private let socketPathOverride: String?
    private let headlessMode: Bool
    private let headlessTestMode: Bool

    init(socketPathOverride: String?, headlessMode: Bool, headlessTestMode: Bool) {
        self.socketPathOverride = socketPathOverride
        self.headlessMode = headlessMode
        self.headlessTestMode = headlessTestMode
        super.init()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        startServices()
    }

    func applicationWillTerminate(_ notification: Notification) {
        socketServer?.stop()
    }

    func startServices() {
        do {
            let server = try SocketServer(
                socketPath: socketPathOverride,
                killSwitch: .shared,
                headlessTestMode: headlessTestMode
            )
            try server.start()
            self.socketServer = server
            FileHandle.standardOutput.write(
                Data("Whole Body Compute listening on \(server.socketPath) (token: \(server.authenticator.tokenPath))\n".utf8)
            )
        } catch {
            FileHandle.standardError.write(
                Data("Whole Body Compute fatal socket startup error: \(error.localizedDescription)\n".utf8)
            )
            exit(1)
        }

        if !headlessMode {
            MainActor.assumeIsolated {
                self.statusBarController = StatusBarController(killSwitch: .shared)
            }
        }

        installSignalHandlers()
    }

    private func installSignalHandlers() {
        signal(SIGTERM, SIG_IGN)
        signal(SIGINT, SIG_IGN)

        let termSrc = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
        termSrc.setEventHandler { [weak self] in
            self?.socketServer?.stop()
            exit(0)
        }
        termSrc.resume()
        self.sigtermSource = termSrc

        let intSrc = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
        intSrc.setEventHandler { [weak self] in
            self?.socketServer?.stop()
            exit(0)
        }
        intSrc.resume()
        self.sigintSource = intSrc
    }
}

// MARK: - Entrypoint

let args = CommandLine.arguments
if args.contains("--version") {
    print("Whole Body Compute 0.1.6")
    exit(0)
}

if args.contains("--ears-mic") || args.contains("--ears-file") || args.contains("--mic") || args.contains("--file") {
    EarsEngine.runSpeechCLI(arguments: args)
}

if args.contains("--ears-hotkey-stream") {
    EarsEngine.runHotkeyStreamCLI()
}

var socketOverride: String?
var i = 1
while i < args.count {
    if args[i] == "--socket", i + 1 < args.count {
        socketOverride = args[i + 1]
        i += 2
        continue
    }
    i += 1
}

let env = ProcessInfo.processInfo.environment
let headlessTestMode = args.contains("--headless-test-mode")
let headlessMode = headlessTestMode || args.contains("--headless") || env["WBC_EYES_HEADLESS"] == "1"

let delegate = WholeBodyComputeAppDelegate(
    socketPathOverride: socketOverride,
    headlessMode: headlessMode,
    headlessTestMode: headlessTestMode
)

if headlessMode {
    delegate.startServices()
    RunLoop.main.run()
} else {
    let app = NSApplication.shared
    app.setActivationPolicy(.accessory)
    app.delegate = delegate
    app.run()
}
