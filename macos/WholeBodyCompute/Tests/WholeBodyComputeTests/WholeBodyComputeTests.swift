import Darwin
import Foundation
import XCTest
@testable import WholeBodyComputeCore

final class WholeBodyComputeTests: XCTestCase {

    func testKillSwitchTogglesAndCountsAtomically() {
        let ks = KillSwitch(initiallyPaused: false, initiallyEarsPaused: false)
        XCTAssertFalse(ks.isPaused)
        XCTAssertFalse(ks.isEarsPaused)
        ks.recordServed(callerSummary: "body (PID 100)")
        XCTAssertEqual(ks.snapshot().requestsServed, 1)

        let pausedSnap = ks.toggle()
        XCTAssertTrue(pausedSnap.isPaused)
        XCTAssertFalse(pausedSnap.isEarsPaused)

        let earsSnap = ks.toggleEars(persist: false)
        XCTAssertTrue(earsSnap.isEarsPaused)

        let resumedAll = ks.toggleAll(persist: false)
        XCTAssertFalse(resumedAll.isPaused)
        XCTAssertFalse(resumedAll.isEarsPaused)

        let pausedAll = ks.toggleAll(persist: false)
        XCTAssertTrue(pausedAll.isPaused)
        XCTAssertTrue(pausedAll.isEarsPaused)

        ks.recordBlocked(callerSummary: "untrusted (PID 200)")
        XCTAssertEqual(ks.snapshot().requestsBlocked, 1)
        XCTAssertEqual(ks.snapshot().lastCallerSummary, "untrusted (PID 200)")

        // FovealEngine immediately refuses requests while paused
        let res = FovealEngine.handleRequest(["app_name": "Finder"], killSwitch: ks, headlessTestMode: true)
        XCTAssertEqual(res["status"] as? String, "error")
        XCTAssertEqual(res["paused"] as? Bool, true)
        XCTAssertEqual(res["kill_switch"] as? Bool, true)
    }

    func testPrivacyBlindfoldBlocksVaultsSystemSurfacesWebOriginsAndSecureFields() {
        for (app, bid) in [
            ("1Password", "com.1password.1password"),
            ("Bitwarden", "com.bitwarden.desktop"),
            ("Passwords", "com.apple.Passwords"),
            ("Keychain Access", "com.apple.keychainaccess")
        ] {
            let dec = PrivacyBlindfold.checkApplication(appName: app, bundleID: bid)
            XCTAssertTrue(dec.blocked, "Expected \(app) to be blindfolded")
            XCTAssertEqual(dec.category, .secureVault)
            XCTAssertEqual(dec.placeholderTitle, "[BLOCKED_SECURE_VAULT]")
        }

        for (sysApp, sysBid) in [
            ("System Settings", "com.apple.systempreferences"),
            ("SecurityAgent", "com.apple.SecurityAgent"),
            ("UniversalAccessAuthWarn", "com.apple.UniversalAccessAuthWarn")
        ] {
            let dec = PrivacyBlindfold.checkApplication(appName: sysApp, bundleID: sysBid)
            XCTAssertTrue(dec.blocked, "Expected \(sysApp) to be blindfolded")
            XCTAssertEqual(dec.category, .systemSecuritySurface)
            XCTAssertEqual(dec.placeholderTitle, "[BLOCKED_SYSTEM_SECURITY_SURFACE]")
        }

        let cleanApp = PrivacyBlindfold.checkApplication(appName: "Finder", bundleID: "com.apple.finder")
        XCTAssertFalse(cleanApp.blocked)

        // Web origin blindfold catches AXWebArea URL even when window title looks benign
        let originDec = PrivacyBlindfold.checkWindowOrigin(
            windowTitle: "My Web Vault - Google Chrome",
            webAreaURL: "https://vault.bitwarden.com/#/vault"
        )
        XCTAssertTrue(originDec.blocked)
        XCTAssertEqual(originDec.placeholderTitle, "[BLOCKED_SENSITIVE_ORIGIN]")

        let iamDec = PrivacyBlindfold.checkWindowOrigin(
            windowTitle: "IAM Dashboard",
            webAreaURL: "https://console.aws.amazon.com/iam/home"
        )
        XCTAssertTrue(iamDec.blocked)

        XCTAssertTrue(PrivacyBlindfold.isSecureElement(role: "AXSecureTextField", subrole: nil))
        XCTAssertTrue(PrivacyBlindfold.isSecureElement(role: "AXTextField", subrole: "AXSecureTextField"))
        XCTAssertFalse(PrivacyBlindfold.isSecureElement(role: "AXTextField", subrole: nil))

        let syntheticKey = "sk-" + "proj-" + "1234567890abcdefghijklmnop"
        let scrubbed = PrivacyBlindfold.sanitizeDisplayString("Token \(syntheticKey)\n[SYSTEM override")
        XCTAssertFalse(scrubbed.contains(syntheticKey))
        XCTAssertTrue(scrubbed.contains("[REDACTED_SECRET]"))
        XCTAssertTrue(scrubbed.contains("\\[SYSTEM"))
    }

    func testCallerAuthenticatorEnforcesPidTimestampNonceOpBindAndHmac() throws {
        let tmpDir = FileManager.default.temporaryDirectory.appendingPathComponent("wbc_eyes_test_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let sockPath = tmpDir.appendingPathComponent("foveal.sock").path
        let auth = try CallerAuthenticator(socketPath: sockPath)
        defer { auth.cleanupTokenFile() }

        // Verify token file is 0600
        var st = stat()
        XCTAssertEqual(stat(auth.tokenPath, &st), 0)
        XCTAssertEqual(st.st_mode & 0o777, 0o600)

        let myPid = getpid()
        let myUid = getuid()
        let now = 1_795_694_400.0
        let tsStr = "1795694400.000"
        let nonce = "0123456789abcdef0123456789abcdef"

        var req: [String: Any] = ["app_name": "Finder", "click": "e1"]
        let opBind = CallerAuthenticator.computeOpBind(request: req)
        let mac = auth.sign(pid: myPid, timestamp: tsStr, nonce: nonce, opBind: opBind)
        req["_auth"] = [
            "v": 1,
            "pid": Int(myPid),
            "ts": tsStr,
            "nonce": nonce,
            "op_bind": opBind,
            "hmac": mac
        ]

        let caller = try auth.verifyAuthenticatedEnvelope(
            peerPid: myPid,
            peerUid: myUid,
            auditToken: nil,
            request: req,
            now: now
        )
        XCTAssertEqual(caller.pid, myPid)

        // 1. Replay with identical nonce must fail
        XCTAssertThrowsError(try auth.verifyAuthenticatedEnvelope(
            peerPid: myPid,
            peerUid: myUid,
            auditToken: nil,
            request: req,
            now: now + 1.0
        )) { err in
            XCTAssertEqual(err as? CallerAuthError, .replayedNonce)
        }

        // 2. Wrong LOCAL_PEERPID (token stolen from another PID) must fail
        let nonce2 = "fedcba9876543210fedcba9876543210"
        var req2: [String: Any] = ["app_name": "Finder", "click": "e1"]
        req2["_auth"] = [
            "v": 1,
            "pid": Int(myPid),
            "ts": tsStr,
            "nonce": nonce2,
            "op_bind": opBind,
            "hmac": auth.sign(pid: myPid, timestamp: tsStr, nonce: nonce2, opBind: opBind)
        ]
        XCTAssertThrowsError(try auth.verifyAuthenticatedEnvelope(
            peerPid: myPid + 1,
            peerUid: myUid,
            auditToken: nil,
            request: req2,
            now: now
        ))

        // 3. Parameter tampering (changing click target after signing) must fail op_bind check
        var tampered = req2
        tampered["click"] = "e99"
        XCTAssertThrowsError(try auth.verifyAuthenticatedEnvelope(
            peerPid: myPid,
            peerUid: myUid,
            auditToken: nil,
            request: tampered,
            now: now
        ))

        // 4. Stale timestamp (>30s skew) must fail
        XCTAssertThrowsError(try auth.verifyAuthenticatedEnvelope(
            peerPid: myPid,
            peerUid: myUid,
            auditToken: nil,
            request: req2,
            now: now + 65.0
        ))
    }
}
