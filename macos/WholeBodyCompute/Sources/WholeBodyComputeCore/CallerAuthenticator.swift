import CryptoKit
import Darwin
import Foundation
import Security
import os.lock

@_silgen_name("proc_pidpath")
private func wbc_proc_pidpath(_ pid: Int32, _ buffer: UnsafeMutableRawPointer, _ buffersize: UInt32) -> Int32

private let WBC_SOL_LOCAL: Int32 = 0
private let WBC_LOCAL_PEERCRED: Int32 = 0x001
private let WBC_LOCAL_PEERPID: Int32 = 0x002
private let WBC_LOCAL_PEERTOKEN: Int32 = 0x006
private let WBC_PROC_PIDPATHINFO_MAXSIZE: UInt32 = 4096

/// Kernel-verified & HMAC-authenticated caller identity on `foveal.sock`.
public struct AuthenticatedCaller: Sendable, Equatable {
    public let pid: pid_t
    public let uid: uid_t
    public let executablePath: String
    public let signingIdentifier: String?
    public let teamIdentifier: String?
    public let signingFlags: UInt32

    public var summary: String {
        let bin = URL(fileURLWithPath: executablePath).lastPathComponent
        let name = bin.isEmpty ? "pid-\(pid)" : bin
        return "\(name) (PID \(pid))"
    }
}

public enum CallerAuthError: Error, LocalizedError, Equatable {
    case peerCredUnavailable
    case uidMismatch(expected: uid_t, actual: uid_t)
    case peerPidUnavailable
    case missingAuthEnvelope
    case pidMismatch(expected: pid_t, claimed: Int)
    case invalidTimestamp
    case expiredTimestamp(skewSeconds: Double)
    case invalidNonce
    case replayedNonce
    case opBindMismatch(expected: String, claimed: String)
    case invalidHmacSignature
    case codeSignatureRejected(reason: String)

    public var errorDescription: String? {
        switch self {
        case .peerCredUnavailable:
            return "Caller authentication failed: unable to read LOCAL_PEERCRED."
        case let .uidMismatch(expected, actual):
            return "Caller authentication failed: peer UID \(actual) does not match owner UID \(expected)."
        case .peerPidUnavailable:
            return "Caller authentication failed: unable to read LOCAL_PEERPID."
        case .missingAuthEnvelope:
            return "Caller authentication failed: missing '_auth' HMAC token envelope."
        case let .pidMismatch(expected, claimed):
            return "Caller authentication failed: claimed PID \(claimed) != kernel LOCAL_PEERPID \(expected)."
        case .invalidTimestamp:
            return "Caller authentication failed: malformed '_auth.ts' timestamp."
        case let .expiredTimestamp(skew):
            return String(format: "Caller authentication failed: timestamp skew %.1fs exceeds 30s window.", skew)
        case .invalidNonce:
            return "Caller authentication failed: '_auth.nonce' must be at least 16 characters."
        case .replayedNonce:
            return "Caller authentication failed: replayed '_auth.nonce' rejected."
        case let .opBindMismatch(expected, claimed):
            return "Caller authentication failed: request parameters do not match signed op_bind (expected '\(expected)', got '\(claimed)')."
        case .invalidHmacSignature:
            return "Caller authentication failed: invalid HMAC-SHA256 signature."
        case let .codeSignatureRejected(reason):
            return "Caller authentication failed: SecCode verification rejected peer (\(reason))."
        }
    }
}

/// Enforces `LOCAL_PEERPID` + `LOCAL_PEERCRED` + `0600` per-session HMAC token
/// + `SecCodeCopyGuestWithAttributes` inspection on `foveal.sock`.
public final class CallerAuthenticator: @unchecked Sendable {
    public let tokenPath: String
    private let secretBytes: Data
    private let symmetricKey: SymmetricKey
    private let maxClockSkewSeconds: Double
    private let requireCodeSign: Bool
    private let allowedTeamIDs: Set<String>
    private let allowedSigningIdentifiers: Set<String>

    private var nonceLock = os_unfair_lock_s()
    private var seenNonces: [String: TimeInterval] = [:]
    private let maxNonceEntries = 4096

    public init(
        socketPath: String,
        explicitTokenPath: String? = nil,
        explicitSecretBytes: Data? = nil,
        maxClockSkewSeconds: Double = 30.0,
        requireCodeSign: Bool? = nil,
        allowedTeamIDs: Set<String>? = nil,
        allowedSigningIdentifiers: Set<String>? = nil
    ) throws {
        let resolvedTok = Self.resolveTokenPath(socketPath: socketPath, explicitTokenPath: explicitTokenPath)
        self.tokenPath = resolvedTok
        self.maxClockSkewSeconds = maxClockSkewSeconds

        let env = ProcessInfo.processInfo.environment
        self.requireCodeSign = requireCodeSign ?? (env["WBC_EYES_REQUIRE_CODESIGN"] == "1")
        if let allowedTeamIDs {
            self.allowedTeamIDs = allowedTeamIDs
        } else if let rawTeams = env["WBC_EYES_ALLOWED_TEAM_IDS"], !rawTeams.isEmpty {
            self.allowedTeamIDs = Set(rawTeams.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty })
        } else {
            self.allowedTeamIDs = []
        }

        if let allowedSigningIdentifiers {
            self.allowedSigningIdentifiers = allowedSigningIdentifiers
        } else if let rawIDs = env["WBC_EYES_ALLOWED_IDENTIFIERS"], !rawIDs.isEmpty {
            self.allowedSigningIdentifiers = Set(rawIDs.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty })
        } else {
            self.allowedSigningIdentifiers = []
        }

        if let explicitSecretBytes {
            self.secretBytes = explicitSecretBytes
        } else {
            var randomBytes = [UInt8](repeating: 0, count: 32)
            let rc = SecRandomCopyBytes(kSecRandomDefault, randomBytes.count, &randomBytes)
            guard rc == errSecSuccess else {
                throw NSError(domain: "WholeBodyCompute.CallerAuthenticator", code: Int(rc), userInfo: [
                    NSLocalizedDescriptionKey: "SecRandomCopyBytes failed (\(rc))"
                ])
            }
            let data = Data(randomBytes)
            self.secretBytes = data
            try Self.writeTokenFileAtomically(secret: data, to: resolvedTok)
        }
        self.symmetricKey = SymmetricKey(data: self.secretBytes)
    }

    public static func resolveTokenPath(socketPath: String, explicitTokenPath: String? = nil) -> String {
        if let explicitTokenPath, !explicitTokenPath.isEmpty {
            return (explicitTokenPath as NSString).expandingTildeInPath
        }
        if let envTok = ProcessInfo.processInfo.environment["WBC_FOVEAL_TOKEN_PATH"],
           !envTok.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            return (envTok as NSString).expandingTildeInPath
        }
        if socketPath.hasSuffix(".sock") {
            return String(socketPath.dropLast(5)) + ".token"
        }
        return socketPath + ".token"
    }

    private static func writeTokenFileAtomically(secret: Data, to path: String) throws {
        let url = URL(fileURLWithPath: path)
        let dirURL = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: dirURL, withIntermediateDirectories: true, attributes: [
            .posixPermissions: NSNumber(value: Int16(0o700))
        ])
        _ = chmod(dirURL.path, mode_t(0o700))

        let tmpPath = path + ".tmp.\(getpid())"
        unlink(tmpPath)
        let fd = open(tmpPath, O_CREAT | O_EXCL | O_WRONLY | O_NOFOLLOW, mode_t(0o600))
        guard fd >= 0 else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno), userInfo: [
                NSLocalizedDescriptionKey: "Failed creating \(tmpPath): \(String(cString: strerror(errno)))"
            ])
        }
        defer { close(fd) }
        _ = fchmod(fd, mode_t(0o600))

        var hexLine = secret.map { String(format: "%02x", $0) }.joined() + "\n"
        try hexLine.withUTF8 { ptr in
            var written = 0
            while written < ptr.count {
                let n = write(fd, ptr.baseAddress!.advanced(by: written), ptr.count - written)
                if n <= 0 {
                    throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno), userInfo: [
                        NSLocalizedDescriptionKey: "Failed writing token file"
                    ])
                }
                written += n
            }
        }

        if rename(tmpPath, path) != 0 {
            unlink(tmpPath)
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno), userInfo: [
                NSLocalizedDescriptionKey: "Failed renaming token file to \(path)"
            ])
        }
        _ = chmod(path, mode_t(0o600))
    }

    public func cleanupTokenFile() {
        unlink(tokenPath)
    }

    /// Compute canonical `op_bind` string matching Python `_compute_foveal_op_bind`.
    public static func computeOpBind(request: [String: Any], methodOverride: String? = nil) -> String {
        let opStr = methodOverride ?? (request["op"] as? String) ?? "snapshot"
        let appStr = (request["app_name"] as? String) ?? ""
        let pidStr: String
        if let pidInt = request["pid"] as? Int {
            pidStr = String(pidInt)
        } else if let pidNum = request["pid"] as? NSNumber {
            pidStr = pidNum.stringValue
        } else {
            pidStr = ""
        }

        let actKeys = [
            "click", "click_xy", "hover", "right_click",
            "type_text", "paste_text", "hotkey", "menu",
            "scroll", "attach", "chain", "submit"
        ]
        var actParts: [String] = []
        for k in actKeys {
            guard let rawVal = request[k] else { continue }
            if rawVal is NSNull { continue }
            if let b = rawVal as? Bool {
                if b { actParts.append("\(k)=true") }
                continue
            }
            if let s = rawVal as? String {
                if !s.isEmpty { actParts.append("\(k)=\(s)") }
                continue
            }
            if let num = rawVal as? NSNumber {
                // Distinguish Bool NSNumber from numeric NSNumber
                if CFGetTypeID(num) == CFBooleanGetTypeID() {
                    if num.boolValue { actParts.append("\(k)=true") }
                } else {
                    actParts.append("\(k)=\(num.stringValue)")
                }
                continue
            }
            if JSONSerialization.isValidJSONObject(rawVal),
               let jsonBytes = try? JSONSerialization.data(withJSONObject: rawVal, options: [.sortedKeys]),
               let jsonStr = String(data: jsonBytes, encoding: .utf8)
            {
                actParts.append("\(k)=\(jsonStr)")
            }
        }
        let actSummary = actParts.joined(separator: "|")
        return "\(opStr):\(appStr):\(pidStr):\(actSummary)"
    }

    /// Generate HMAC-SHA256 hex digest for testing or in-process signing.
    public func sign(pid: pid_t, timestamp: String, nonce: String, opBind: String) -> String {
        let message = "wbc-foveal-v1:\(pid):\(timestamp):\(nonce):\(opBind)"
        let mac = HMAC<SHA256>.authenticationCode(for: Data(message.utf8), using: symmetricKey)
        return mac.map { String(format: "%02x", $0) }.joined()
    }

    /// Authenticate a live `AF_UNIX` client file descriptor and JSON request dictionary.
    public func authenticateSocketPeer(
        clientFD: Int32,
        request: [String: Any],
        methodOverride: String? = nil,
        now: TimeInterval = Date().timeIntervalSince1970
    ) throws -> AuthenticatedCaller {
        // 1. Kernel LOCAL_PEERCRED check
        var cred = xucred()
        var credLen = socklen_t(MemoryLayout<xucred>.size)
        let credRC = withUnsafeMutablePointer(to: &cred) { ptr in
            getsockopt(clientFD, WBC_SOL_LOCAL, WBC_LOCAL_PEERCRED, ptr, &credLen)
        }
        guard credRC == 0 else {
            throw CallerAuthError.peerCredUnavailable
        }
        let expectedUID = getuid()
        guard cred.cr_uid == expectedUID else {
            throw CallerAuthError.uidMismatch(expected: expectedUID, actual: cred.cr_uid)
        }

        // 2. Kernel LOCAL_PEERPID check
        var peerPid: pid_t = 0
        var pidLen = socklen_t(MemoryLayout<pid_t>.size)
        let pidRC = withUnsafeMutablePointer(to: &peerPid) { ptr in
            getsockopt(clientFD, WBC_SOL_LOCAL, WBC_LOCAL_PEERPID, ptr, &pidLen)
        }
        guard pidRC == 0, peerPid > 0 else {
            throw CallerAuthError.peerPidUnavailable
        }

        // 3. Optional kernel LOCAL_PEERTOKEN (audit_token_t) to prevent PID-reuse TOCTOU
        var auditToken = audit_token_t()
        var tokenLen = socklen_t(MemoryLayout<audit_token_t>.size)
        let hasAuditToken = withUnsafeMutablePointer(to: &auditToken) { ptr in
            getsockopt(clientFD, WBC_SOL_LOCAL, WBC_LOCAL_PEERTOKEN, ptr, &tokenLen) == 0
        }

        return try verifyAuthenticatedEnvelope(
            peerPid: peerPid,
            peerUid: cred.cr_uid,
            auditToken: hasAuditToken ? auditToken : nil,
            request: request,
            methodOverride: methodOverride,
            now: now
        )
    }

    /// Verify `_auth` envelope + `SecCodeCopyGuestWithAttributes` against kernel-verified peer attributes.
    public func verifyAuthenticatedEnvelope(
        peerPid: pid_t,
        peerUid: uid_t,
        auditToken: audit_token_t?,
        request: [String: Any],
        methodOverride: String? = nil,
        now: TimeInterval = Date().timeIntervalSince1970
    ) throws -> AuthenticatedCaller {
        let expectedUID = getuid()
        guard peerUid == expectedUID else {
            throw CallerAuthError.uidMismatch(expected: expectedUID, actual: peerUid)
        }

        guard let authDict = request["_auth"] as? [String: Any] else {
            throw CallerAuthError.missingAuthEnvelope
        }

        let claimedPid: Int
        if let p = authDict["pid"] as? Int {
            claimedPid = p
        } else if let n = authDict["pid"] as? NSNumber {
            claimedPid = n.intValue
        } else {
            throw CallerAuthError.pidMismatch(expected: peerPid, claimed: -1)
        }
        guard claimedPid == Int(peerPid) else {
            throw CallerAuthError.pidMismatch(expected: peerPid, claimed: claimedPid)
        }

        let tsStr: String
        if let s = authDict["ts"] as? String {
            tsStr = s
        } else if let n = authDict["ts"] as? NSNumber {
            tsStr = n.stringValue
        } else {
            throw CallerAuthError.invalidTimestamp
        }
        guard let tsVal = Double(tsStr) else {
            throw CallerAuthError.invalidTimestamp
        }
        let skew = abs(now - tsVal)
        guard skew <= maxClockSkewSeconds else {
            throw CallerAuthError.expiredTimestamp(skewSeconds: skew)
        }

        guard let nonce = authDict["nonce"] as? String, nonce.count >= 16, nonce.count <= 128 else {
            throw CallerAuthError.invalidNonce
        }

        let expectedOpBind = Self.computeOpBind(request: request, methodOverride: methodOverride)
        let claimedOpBind = (authDict["op_bind"] as? String) ?? ""
        guard claimedOpBind == expectedOpBind else {
            throw CallerAuthError.opBindMismatch(expected: expectedOpBind, claimed: claimedOpBind)
        }

        guard let hmacHex = authDict["hmac"] as? String,
              let providedMAC = Self.decodeHex(hmacHex),
              providedMAC.count == 32
        else {
            throw CallerAuthError.invalidHmacSignature
        }

        let canonicalMsg = "wbc-foveal-v1:\(peerPid):\(tsStr):\(nonce):\(expectedOpBind)"
        let macValid = HMAC<SHA256>.isValidAuthenticationCode(
            providedMAC,
            authenticating: Data(canonicalMsg.utf8),
            using: symmetricKey
        )
        guard macValid else {
            throw CallerAuthError.invalidHmacSignature
        }

        // Check & record one-time nonce after MAC verification (so attackers can't flood replay cache)
        try checkAndInsertNonce(nonce, now: now)

        // Resolve executable path and inspect code signature via SecCodeCopyGuestWithAttributes
        let execPath = Self.resolveProcessPath(pid: peerPid)
        let codeInfo = try inspectPeerCodeSignature(peerPid: peerPid, auditToken: auditToken)

        return AuthenticatedCaller(
            pid: peerPid,
            uid: peerUid,
            executablePath: execPath,
            signingIdentifier: codeInfo.identifier,
            teamIdentifier: codeInfo.teamID,
            signingFlags: codeInfo.flags
        )
    }

    private func checkAndInsertNonce(_ nonce: String, now: TimeInterval) throws {
        os_unfair_lock_lock(&nonceLock)
        defer { os_unfair_lock_unlock(&nonceLock) }

        let cutoff = now - (maxClockSkewSeconds * 2.0 + 5.0)
        if seenNonces.count >= maxNonceEntries {
            seenNonces = seenNonces.filter { $0.value >= cutoff }
            if seenNonces.count >= maxNonceEntries, let oldestKey = seenNonces.min(by: { $0.value < $1.value })?.key {
                seenNonces.removeValue(forKey: oldestKey)
            }
        }
        if let prevTs = seenNonces[nonce], prevTs >= cutoff {
            throw CallerAuthError.replayedNonce
        }
        seenNonces[nonce] = now
    }

    private func inspectPeerCodeSignature(
        peerPid: pid_t,
        auditToken: audit_token_t?
    ) throws -> (identifier: String?, teamID: String?, flags: UInt32) {
        var guestCode: SecCode?
        var status: OSStatus = errSecParam

        if var token = auditToken {
            let tokenData = withUnsafeBytes(of: &token) { Data($0) as CFData }
            let attrs = [kSecGuestAttributeAudit: tokenData] as CFDictionary
            status = SecCodeCopyGuestWithAttributes(nil, attrs, SecCSFlags(), &guestCode)
        }
        if status != errSecSuccess || guestCode == nil {
            let attrs = [kSecGuestAttributePid: NSNumber(value: peerPid)] as CFDictionary
            status = SecCodeCopyGuestWithAttributes(nil, attrs, SecCSFlags(), &guestCode)
        }

        let strictPolicy = requireCodeSign || !allowedTeamIDs.isEmpty || !allowedSigningIdentifiers.isEmpty
        guard status == errSecSuccess, let code = guestCode else {
            if strictPolicy {
                throw CallerAuthError.codeSignatureRejected(reason: "SecCodeCopyGuestWithAttributes status=\(status)")
            }
            return (nil, nil, 0)
        }

        if requireCodeSign {
            let validStatus = SecCodeCheckValidity(code, SecCSFlags(), nil)
            if validStatus != errSecSuccess {
                throw CallerAuthError.codeSignatureRejected(reason: "SecCodeCheckValidity status=\(validStatus)")
            }
        }

        var staticCode: SecStaticCode?
        guard SecCodeCopyStaticCode(code, SecCSFlags(), &staticCode) == errSecSuccess,
              let sCode = staticCode
        else {
            if strictPolicy {
                throw CallerAuthError.codeSignatureRejected(reason: "SecCodeCopyStaticCode failed")
            }
            return (nil, nil, 0)
        }

        var rawInfo: CFDictionary?
        let infoStatus = SecCodeCopySigningInformation(
            sCode,
            SecCSFlags(rawValue: kSecCSSigningInformation),
            &rawInfo
        )
        guard infoStatus == errSecSuccess, let info = rawInfo as? [String: Any] else {
            if strictPolicy {
                throw CallerAuthError.codeSignatureRejected(reason: "SecCodeCopySigningInformation status=\(infoStatus)")
            }
            return (nil, nil, 0)
        }

        let identifier = info[kSecCodeInfoIdentifier as String] as? String
        let teamID = info[kSecCodeInfoTeamIdentifier as String] as? String
        let flags = (info[kSecCodeInfoFlags as String] as? NSNumber)?.uint32Value ?? 0

        if !allowedTeamIDs.isEmpty {
            guard let teamID, allowedTeamIDs.contains(teamID) else {
                throw CallerAuthError.codeSignatureRejected(
                    reason: "teamIdentifier '\(teamID ?? "none")' not in WBC_EYES_ALLOWED_TEAM_IDS"
                )
            }
        }

        if !allowedSigningIdentifiers.isEmpty {
            guard let identifier, allowedSigningIdentifiers.contains(identifier) else {
                throw CallerAuthError.codeSignatureRejected(
                    reason: "signingIdentifier '\(identifier ?? "none")' not in WBC_EYES_ALLOWED_IDENTIFIERS"
                )
            }
        }

        return (identifier, teamID, flags)
    }

    private static func resolveProcessPath(pid: pid_t) -> String {
        var buf = [CChar](repeating: 0, count: Int(WBC_PROC_PIDPATHINFO_MAXSIZE))
        let len = wbc_proc_pidpath(pid, &buf, WBC_PROC_PIDPATHINFO_MAXSIZE)
        if len > 0 {
            return String(cString: buf)
        }
        return ""
    }

    private static func decodeHex(_ hex: String) -> Data? {
        let clean = hex.trimmingCharacters(in: .whitespacesAndNewlines)
        guard clean.count % 2 == 0 else { return nil }
        var bytes = Data(capacity: clean.count / 2)
        var idx = clean.startIndex
        while idx < clean.endIndex {
            let next = clean.index(idx, offsetBy: 2)
            guard let b = UInt8(clean[idx..<next], radix: 16) else { return nil }
            bytes.append(b)
            idx = next
        }
        return bytes
    }
}
