import ApplicationServices
import Darwin
import Foundation

/// Persistent `0600` `AF_UNIX` JSON-RPC / flat-JSON server for `~/.whole_body_compute/foveal.sock`.
public final class SocketServer: @unchecked Sendable {
    public let socketPath: String
    public let authenticator: CallerAuthenticator
    public let killSwitch: KillSwitch
    public let headlessTestMode: Bool

    private var listenFD: Int32 = -1
    private let acceptQueue = DispatchQueue(label: "com.wholebodycompute.eyes.accept", qos: .userInteractive)
    private let workQueue = DispatchQueue(label: "com.wholebodycompute.eyes.ax-serial", qos: .userInteractive)
    private var isStopping = false

    public init(
        socketPath: String? = nil,
        killSwitch: KillSwitch = .shared,
        headlessTestMode: Bool = false
    ) throws {
        let resolved = Self.resolveSocketPath(explicitPath: socketPath)
        self.socketPath = resolved
        self.killSwitch = killSwitch
        self.headlessTestMode = headlessTestMode
        self.authenticator = try CallerAuthenticator(socketPath: resolved)
    }

    public static func resolveSocketPath(explicitPath: String? = nil) -> String {
        if let explicitPath, !explicitPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return (explicitPath as NSString).expandingTildeInPath
        }
        let env = ProcessInfo.processInfo.environment
        if let envSock = env["WBC_FOVEAL_SOCKET_PATH"], !envSock.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return (envSock as NSString).expandingTildeInPath
        }
        if let dataDir = env["WBC_DATA_DIR"], !dataDir.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return ((dataDir as NSString).expandingTildeInPath as NSString).appendingPathComponent("foveal.sock")
        }
        return (NSHomeDirectory() as NSString).appendingPathComponent(".whole_body_compute/foveal.sock")
    }

    public func start() throws {
        let sockURL = URL(fileURLWithPath: socketPath)
        let parentDir = sockURL.deletingLastPathComponent().path
        try FileManager.default.createDirectory(atPath: parentDir, withIntermediateDirectories: true, attributes: [
            .posixPermissions: NSNumber(value: Int16(0o700))
        ])
        _ = chmod(parentDir, mode_t(0o700))
        unlink(socketPath)

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathUTF8 = socketPath.utf8CString
        let maxSunLen = MemoryLayout.size(ofValue: addr.sun_path)
        guard pathUTF8.count <= maxSunLen else {
            throw NSError(domain: "WBCEyes.SocketServer", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Socket path exceeds sockaddr_un limit (\(maxSunLen) bytes): \(socketPath)"
            ])
        }

        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno), userInfo: [
                NSLocalizedDescriptionKey: "socket(AF_UNIX) failed: \(String(cString: strerror(errno)))"
            ])
        }

        withUnsafeMutablePointer(to: &addr.sun_path) { sunPtr in
            let rawPtr = UnsafeMutableRawPointer(sunPtr).assumingMemoryBound(to: CChar.self)
            pathUTF8.withUnsafeBufferPointer { src in
                _ = memcpy(rawPtr, src.baseAddress!, src.count)
            }
        }

        let prevUmask = umask(mode_t(0o177))
        let bindRC = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { saPtr in
                bind(fd, saPtr, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        umask(prevUmask)

        guard bindRC == 0 else {
            let err = errno
            close(fd)
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(err), userInfo: [
                NSLocalizedDescriptionKey: "bind(\(socketPath)) failed: \(String(cString: strerror(err)))"
            ])
        }
        _ = chmod(socketPath, mode_t(0o600))

        guard listen(fd, 16) == 0 else {
            let err = errno
            close(fd)
            unlink(socketPath)
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(err), userInfo: [
                NSLocalizedDescriptionKey: "listen(\(socketPath)) failed: \(String(cString: strerror(err)))"
            ])
        }

        self.listenFD = fd
        acceptQueue.async { [weak self] in
            self?.acceptLoop()
        }
    }

    public func stop() {
        isStopping = true
        if listenFD >= 0 {
            close(listenFD)
            listenFD = -1
        }
        unlink(socketPath)
        authenticator.cleanupTokenFile()
    }

    private func acceptLoop() {
        while !isStopping && listenFD >= 0 {
            var clientAddr = sockaddr_un()
            var clientLen = socklen_t(MemoryLayout<sockaddr_un>.size)
            let clientFD = withUnsafeMutablePointer(to: &clientAddr) { ptr in
                ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { saPtr in
                    accept(listenFD, saPtr, &clientLen)
                }
            }
            if clientFD < 0 {
                if isStopping || errno == EBADF || errno == EINVAL {
                    break
                }
                continue
            }
            workQueue.async { [weak self] in
                guard let self else {
                    close(clientFD)
                    return
                }
                self.handleClient(clientFD: clientFD)
            }
        }
    }

    private func handleClient(clientFD: Int32) {
        defer { close(clientFD) }

        // Prevent SIGPIPE on abrupt client disconnect
        var noSigPipe: Int32 = 1
        _ = setsockopt(clientFD, SOL_SOCKET, SO_NOSIGPIPE, &noSigPipe, socklen_t(MemoryLayout<Int32>.size))

        var tv = timeval(tv_sec: 5, tv_usec: 0)
        _ = setsockopt(clientFD, SOL_SOCKET, SO_RCVTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))

        let maxBytes = 65536
        var buffer = Data()
        var chunk = [UInt8](repeating: 0, count: 4096)
        while buffer.count < maxBytes {
            let n = read(clientFD, &chunk, chunk.count)
            if n <= 0 { break }
            buffer.append(chunk, count: n)
            if chunk[0..<n].contains(0x0A) { break }
        }

        if buffer.isEmpty { return }
        if buffer.count >= maxBytes && !buffer.contains(0x0A) {
            sendJSON(clientFD: clientFD, payload: [
                "status": "error",
                "error": "Socket request exceeded 64 KiB limit"
            ])
            return
        }

        if let nlIdx = buffer.firstIndex(of: 0x0A) {
            buffer = buffer.subdata(in: 0..<nlIdx)
        }

        guard let rawObj = try? JSONSerialization.jsonObject(with: buffer),
              let outerDict = rawObj as? [String: Any]
        else {
            sendJSON(clientFD: clientFD, payload: [
                "status": "error",
                "error": "Invalid JSON object on foveal.sock"
            ])
            return
        }

        let isJSONRPC = (outerDict["jsonrpc"] as? String) == "2.0"
        let rpcID = outerDict["id"]
        let rpcMethod = outerDict["method"] as? String

        var effectiveReq: [String: Any]
        let methodOverride: String?
        if isJSONRPC {
            effectiveReq = (outerDict["params"] as? [String: Any]) ?? [:]
            if effectiveReq["_auth"] == nil, let outerAuth = outerDict["_auth"] {
                effectiveReq["_auth"] = outerAuth
            }
            switch rpcMethod {
            case "foveal.status", "foveal.ping", "vifi.ping":
                methodOverride = "status"
            case "foveal.ambient_status":
                methodOverride = "ambient_status"
            default:
                methodOverride = nil
            }
        } else {
            effectiveReq = outerDict
            methodOverride = nil
        }

        // 1. Mandatory Kernel Peer + 0600 HMAC + SecCode Authentication
        let caller: AuthenticatedCaller
        do {
            caller = try authenticator.authenticateSocketPeer(
                clientFD: clientFD,
                request: effectiveReq,
                methodOverride: methodOverride
            )
        } catch {
            killSwitch.recordBlocked()
            let errMsg = error.localizedDescription
            if isJSONRPC {
                sendJSON(clientFD: clientFD, payload: [
                    "jsonrpc": "2.0",
                    "id": rpcID ?? NSNull(),
                    "error": [
                        "code": -32001,
                        "message": errMsg,
                        "data": ["auth_failed": true]
                    ]
                ])
            } else {
                sendJSON(clientFD: clientFD, payload: [
                    "status": "error",
                    "auth_failed": true,
                    "error": errMsg
                ])
            }
            return
        }

        let op = methodOverride ?? (effectiveReq["op"] as? String) ?? ""

        // 2. Test-only pause control (strictly disabled unless launched with --headless-test-mode)
        if op == "test_set_paused" && headlessTestMode {
            let paused = (effectiveReq["paused"] as? Bool) ?? false
            let snap = killSwitch.setPaused(paused)
            sendResponse(
                clientFD: clientFD,
                isJSONRPC: isJSONRPC,
                rpcID: rpcID,
                result: [
                    "status": "ok",
                    "paused": snap.isPaused,
                    "ears_paused": snap.isEarsPaused,
                    "kill_switch": snap.isPaused
                ]
            )
            return
        }

        if op == "test_set_ears_paused" && headlessTestMode {
            let paused = (effectiveReq["paused"] as? Bool) ?? false
            let snap = killSwitch.setEarsPaused(paused, persist: false)
            sendResponse(
                clientFD: clientFD,
                isJSONRPC: isJSONRPC,
                rpcID: rpcID,
                result: [
                    "status": "ok",
                    "paused": snap.isPaused,
                    "ears_paused": snap.isEarsPaused
                ]
            )
            return
        }

        // 3. Helper Health / Status Inspection (0 AXUIElement or CGEvent calls)
        if op == "status" || op == "eyes_helper_status" || op == "ping" {
            let snap = killSwitch.snapshot()
            let trusted = headlessTestMode ? true : AXIsProcessTrusted()
            sendResponse(
                clientFD: clientFD,
                isJSONRPC: isJSONRPC,
                rpcID: rpcID,
                result: [
                    "status": "ok",
                    "pong": true,
                    "provider": "Whole Body Compute.app",
                    "bundle_id": "com.wholebodycompute.app",
                    "version": "0.1.6",
                    "pid": Int(getpid()),
                    "paused": snap.isPaused,
                    "ears_paused": snap.isEarsPaused,
                    "kill_switch": snap.isPaused,
                    "ax_trusted": trusted,
                    "requests_served": snap.requestsServed,
                    "requests_blocked": snap.requestsBlocked,
                    "last_caller": snap.lastCallerSummary
                ]
            )
            return
        }

        if op == "ambient_status" {
            let snap = killSwitch.snapshot()
            sendResponse(
                clientFD: clientFD,
                isJSONRPC: isJSONRPC,
                rpcID: rpcID,
                result: [
                    "status": "ok",
                    "provider": "Whole Body Compute.app",
                    "paused": snap.isPaused,
                    "ears_paused": snap.isEarsPaused,
                    "ambient_observer_running": false,
                    "events_buffered": 0,
                    "staged_in_session": 0
                ]
            )
            return
        }

        // 4. Immediate Kill Switch Enforcement Gate BEFORE any AXUIElement / CGEvent work
        if killSwitch.isPaused {
            killSwitch.recordBlocked(callerSummary: caller.summary)
            let errPayload = FovealEngine.killSwitchErrorPayload()
            if isJSONRPC {
                sendJSON(clientFD: clientFD, payload: [
                    "jsonrpc": "2.0",
                    "id": rpcID ?? NSNull(),
                    "error": [
                        "code": -32002,
                        "message": (errPayload["error"] as? String) ?? "Paused",
                        "data": errPayload
                    ]
                ])
            } else {
                sendJSON(clientFD: clientFD, payload: errPayload)
            }
            return
        }

        if op == "ears_type_diff" {
            if killSwitch.isEarsPaused {
                killSwitch.recordBlocked(callerSummary: caller.summary)
                sendResponse(
                    clientFD: clientFD,
                    isJSONRPC: isJSONRPC,
                    rpcID: rpcID,
                    result: [
                        "status": "error",
                        "ears_paused": true,
                        "kill_switch": true,
                        "error": "Ears Kill Switch is active (Paused)."
                    ]
                )
                return
            }
            let backspaces = (effectiveReq["backspaces"] as? Int) ?? 0
            let text = (effectiveReq["text"] as? String) ?? ""
            let res = EarsEngine.typeDifferentialDictation(
                backspaces: backspaces,
                text: text,
                headlessTestMode: headlessTestMode
            )
            if (res["security_blocked"] as? Bool) == true {
                killSwitch.recordBlocked(callerSummary: caller.summary)
            } else {
                killSwitch.recordServed(callerSummary: caller.summary)
            }
            sendResponse(clientFD: clientFD, isJSONRPC: isJSONRPC, rpcID: rpcID, result: res)
            return
        }

        // 5. Execute Foveal Vision / Tactile Hands Engine with native Swift Blindfolds
        let result = FovealEngine.handleRequest(
            effectiveReq,
            killSwitch: killSwitch,
            headlessTestMode: headlessTestMode
        )

        if (result["kill_switch"] as? Bool) == true
            || ((result["actuation"] as? [String: Any])?["security_blocked"] as? Bool) == true
            || (result["window_title"] as? String)?.hasPrefix("[BLOCKED_") == true
        {
            killSwitch.recordBlocked(callerSummary: caller.summary)
        } else {
            killSwitch.recordServed(callerSummary: caller.summary)
        }

        sendResponse(clientFD: clientFD, isJSONRPC: isJSONRPC, rpcID: rpcID, result: result)
    }

    private func sendResponse(clientFD: Int32, isJSONRPC: Bool, rpcID: Any?, result: [String: Any]) {
        if isJSONRPC {
            if (result["status"] as? String) == "error" {
                sendJSON(clientFD: clientFD, payload: [
                    "jsonrpc": "2.0",
                    "id": rpcID ?? NSNull(),
                    "error": [
                        "code": -32000,
                        "message": (result["error"] as? String) ?? "Foveal error",
                        "data": result
                    ]
                ])
            } else {
                sendJSON(clientFD: clientFD, payload: [
                    "jsonrpc": "2.0",
                    "id": rpcID ?? NSNull(),
                    "result": result
                ])
            }
        } else {
            sendJSON(clientFD: clientFD, payload: result)
        }
    }

    private func sendJSON(clientFD: Int32, payload: [String: Any]) {
        guard var data = try? JSONSerialization.data(withJSONObject: payload, options: []) else { return }
        data.append(0x0A)
        data.withUnsafeBytes { rawBuf in
            guard let base = rawBuf.baseAddress else { return }
            var sent = 0
            while sent < rawBuf.count {
                let n = write(clientFD, base.advanced(by: sent), rawBuf.count - sent)
                if n <= 0 { break }
                sent += n
            }
        }
    }
}
