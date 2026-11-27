"""
End-to-end integration tests for `Whole Body Compute.app` (`macos/WholeBodyCompute`):
1. `0600` socket & `0600` per-session HMAC token file permissions.
2. `CallerAuthenticator` (`LOCAL_PEERPID` + `0600` HMAC-SHA256 + timestamp/nonce replay + `SecCode` identifier allowlist).
3. 1-Click Kill Switch immediate refusal of `body eyes` and `body hands` without PyObjC fallback bypass.
4. Native Swift blindfolds (`1Password`, `Bitwarden`, `Passwords`, `Keychain Access`, `System Settings`, `SecurityAgent`, sensitive web origins, and `AXSecureTextField`).
5. Transparent `body eyes` / `body hands` / `body doctor` / JSON-RPC 2.0 compatibility.
6. Hardened Runtime (`codesign --options runtime`, Library Validation ON) & `body daemon install` LaunchAgent wiring.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import plistlib
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest

from whole_body_compute.cli import main as cli_main
from whole_body_compute.daemon import (
    WBC_EYES_TEAM_ID,
    build_launchd_plist_specs,
    download_and_install_wbc_eyes_app,
    resolve_wbc_eyes_app_executable,
    verify_wbc_eyes_app_signature,
)
from whole_body_compute.doctor import check_organ_the_eyes
from whole_body_compute.integrations.foveal_vision import (
    FovealVisionEngine,
    _compute_foveal_op_bind,
    send_foveal_socket_request,
)

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="WBC Eyes.app is macOS-specific")

REPO_ROOT = Path(__file__).resolve().parent.parent
SWIFT_PKG_DIR = REPO_ROOT / "macos" / "WholeBodyCompute"
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_wbc_eyes_app.sh"


@pytest.fixture(scope="module")
def wbc_eyes_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build ad-hoc Hardened Runtime `WBC Eyes.app` once for the test module."""
    out_dir = tmp_path_factory.mktemp("wbc_eyes_dist")
    proc = subprocess.run(
        [str(BUILD_SCRIPT), "--adhoc", "--native-only", "--output-dir", str(out_dir)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, f"build_wbc_eyes_app.sh failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    app_bundle = out_dir / "Whole Body Compute.app"
    assert (app_bundle / "Contents" / "MacOS" / "WholeBodyCompute").is_file()
    return app_bundle


def _start_headless_eyes_daemon(
    exe_path: Path,
    sock_path: Path,
    extra_env: Dict[str, str] | None = None,
) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["WBC_EYES_HEADLESS"] = "1"
    env["WBC_EYES_MOCK_DOM"] = "1"
    env["WBC_FOVEAL_SOCKET_PATH"] = str(sock_path)
    if extra_env:
        env.update(extra_env)

    proc = subprocess.Popen(
        [str(exe_path), "--headless-test-mode", "--socket", str(sock_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    tok_path = sock_path.with_suffix(".token")
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if sock_path.exists() and tok_path.is_file() and tok_path.stat().st_size >= 64:
            return proc
        if proc.poll() is not None:
            out, err = proc.communicate(timeout=2)
            raise RuntimeError(f"WBCEyes exited early ({proc.returncode}): {out}\n{err}")
        time.sleep(0.02)
    proc.kill()
    raise TimeoutError(f"Timed out waiting for {sock_path} to bind")


@pytest.fixture()
def running_eyes_socket(
    wbc_eyes_bundle: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Path]:
    """Launch isolated headless `WBCEyes` server on a short `/tmp` AF_UNIX socket."""
    # AF_UNIX sun_path limit on macOS is 104 bytes; use short /tmp path
    sock_dir = Path(f"/tmp/wbceyes_{os.getpid()}_{int(time.time() * 1000) % 1000000}")
    sock_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    sock_path = sock_dir / "foveal.sock"
    exe_path = wbc_eyes_bundle / "Contents" / "MacOS" / "WholeBodyCompute"

    proc = _start_headless_eyes_daemon(exe_path, sock_path)
    monkeypatch.setenv("WBC_FOVEAL_SOCKET_PATH", str(sock_path))
    monkeypatch.delenv("VOICEFI_MOCK_FOVEAL", raising=False)
    monkeypatch.delenv("VOICEFI_MOCK_FRONT_APP", raising=False)
    try:
        yield sock_path
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        for p in sock_dir.glob("*"):
            p.unlink(missing_ok=True)
        sock_dir.rmdir()


def _send_raw_unix(sock_path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(2.0)
        s.connect(str(sock_path))
        s.sendall(json.dumps(payload).encode("utf-8") + b"\n")
        with s.makefile("rb") as fp:
            buf = fp.readline()
    return json.loads(buf.decode("utf-8"))


def test_bundle_hardened_runtime_and_library_validation(wbc_eyes_bundle: Path) -> None:
    """Verify `Whole Body Compute.app` has LSUIElement=true, Hardened Runtime ON, Library Validation ON, and zero Python dylibs."""
    info_plist = plistlib.loads((wbc_eyes_bundle / "Contents" / "Info.plist").read_bytes())
    assert info_plist["CFBundleIdentifier"] == "com.wholebodycompute.app"
    assert info_plist["LSUIElement"] is True
    assert "NSMicrophoneUsageDescription" in info_plist

    dv = subprocess.run(
        ["codesign", "-dv", "--verbose=4", str(wbc_eyes_bundle)],
        capture_output=True,
        text=True,
        check=False,
    )
    combined = f"{dv.stdout}\n{dv.stderr}"
    assert "runtime" in combined

    ent = subprocess.run(
        ["codesign", "-d", "--entitlements", ":-", str(wbc_eyes_bundle)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert "disable-library-validation" not in ent.stdout
    assert "allow-dyld-environment-variables" not in ent.stdout

    otool = subprocess.run(
        ["otool", "-L", str(wbc_eyes_bundle / "Contents" / "MacOS" / "WholeBodyCompute")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "python" not in otool.stdout.lower()


def test_socket_permissions_and_caller_authentication(running_eyes_socket: Path) -> None:
    """Verify `0600` socket/token permissions and rejection of unauthenticated, spoofed-PID, or replayed requests."""
    tok_path = running_eyes_socket.with_suffix(".token")
    assert (running_eyes_socket.stat().st_mode & 0o777) == 0o600
    assert (tok_path.stat().st_mode & 0o777) == 0o600

    # 1. Unauthenticated raw socket connection rejected
    unauth = _send_raw_unix(running_eyes_socket, {"app_name": "Finder"})
    assert unauth["status"] == "error"
    assert unauth.get("auth_failed") is True

    # 2. Spoofed PID (mismatching kernel LOCAL_PEERPID) rejected even with valid secret
    secret = bytes.fromhex(tok_path.read_text(encoding="utf-8").strip())
    spoof_pid = os.getpid() + 999
    ts_str = f"{time.time():.3f}"
    nonce = "abcdef0123456789abcdef0123456789"
    req_spoof: Dict[str, Any] = {"app_name": "Finder"}
    op_bind = _compute_foveal_op_bind(req_spoof)
    mac = hmac.new(secret, f"wbc-foveal-v1:{spoof_pid}:{ts_str}:{nonce}:{op_bind}".encode(), hashlib.sha256).hexdigest()
    req_spoof["_auth"] = {"v": 1, "pid": spoof_pid, "ts": ts_str, "nonce": nonce, "op_bind": op_bind, "hmac": mac}
    res_spoof = _send_raw_unix(running_eyes_socket, req_spoof)
    assert res_spoof["status"] == "error"
    assert res_spoof.get("auth_failed") is True
    assert "LOCAL_PEERPID" in res_spoof["error"]

    # 3. Nonce replay rejected
    real_pid = os.getpid()
    nonce_once = "11112222333344445555666677778888"
    mac_real = hmac.new(secret, f"wbc-foveal-v1:{real_pid}:{ts_str}:{nonce_once}:{op_bind}".encode(), hashlib.sha256).hexdigest()
    req_valid = dict(req_spoof)
    req_valid["_auth"] = {"v": 1, "pid": real_pid, "ts": ts_str, "nonce": nonce_once, "op_bind": op_bind, "hmac": mac_real}
    first_ok = _send_raw_unix(running_eyes_socket, req_valid)
    assert first_ok["app_name"] == "Finder"

    replay_err = _send_raw_unix(running_eyes_socket, req_valid)
    assert replay_err["status"] == "error"
    assert replay_err.get("auth_failed") is True
    assert "replayed" in replay_err["error"].lower()

    # 4. Authenticated Python client + JSON-RPC 2.0 envelope succeed
    rpc_resp = send_foveal_socket_request({"jsonrpc": "2.0", "id": 7, "method": "foveal.status", "params": {}})
    assert rpc_resp is not None
    assert rpc_resp["jsonrpc"] == "2.0"
    assert rpc_resp["id"] == 7
    assert rpc_resp["result"]["provider"] == "Whole Body Compute.app"

    # 5. Ears dictation highlight (`op="ears_highlight"`) resolves text_box -> window cleanly in headless mode
    hl_show = send_foveal_socket_request({"op": "ears_highlight", "action": "show", "mode": "ptt", "target": "auto"})
    assert hl_show is not None
    assert hl_show["status"] == "ok"
    assert hl_show["visible"] is True
    assert hl_show["mode"] == "ptt"
    assert hl_show["target_kind"] == "text_box"

    hl_win = send_foveal_socket_request({"op": "ears_highlight", "action": "update", "mode": "hands_free", "target": "window"})
    assert hl_win is not None
    assert hl_win["visible"] is True
    assert hl_win["mode"] == "hands_free"
    assert hl_win["target_kind"] == "window"

    hl_hide = send_foveal_socket_request({"op": "ears_highlight", "action": "hide"})
    assert hl_hide is not None
    assert hl_hide["status"] == "ok"
    assert hl_hide["visible"] is False


def test_seccode_signing_identifier_allowlist_blocks_unlisted_caller(wbc_eyes_bundle: Path) -> None:
    """Verify optional `SecCodeCopyGuestWithAttributes` identifier allowlist rejects non-matching peers."""
    sock_dir = Path(f"/tmp/wbceyes_sec_{os.getpid()}_{int(time.time() * 1000) % 1000000}")
    sock_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    sock_path = sock_dir / "foveal.sock"
    exe_path = wbc_eyes_bundle / "Contents" / "MacOS" / "WholeBodyCompute"

    proc = _start_headless_eyes_daemon(
        exe_path,
        sock_path,
        extra_env={"WBC_EYES_ALLOWED_IDENTIFIERS": "com.wholebodycompute.nonexistent-signed-client"},
    )
    try:
        old_env = os.environ.get("WBC_FOVEAL_SOCKET_PATH")
        os.environ["WBC_FOVEAL_SOCKET_PATH"] = str(sock_path)
        try:
            resp = send_foveal_socket_request({"op": "status"})
            assert resp is not None
            assert resp["status"] == "error"
            assert resp.get("auth_failed") is True
            assert "SecCode" in resp["error"]
        finally:
            if old_env is None:
                os.environ.pop("WBC_FOVEAL_SOCKET_PATH", None)
            else:
                os.environ["WBC_FOVEAL_SOCKET_PATH"] = old_env
    finally:
        proc.terminate()
        proc.wait(timeout=3)
        for p in sock_dir.glob("*"):
            p.unlink(missing_ok=True)
        sock_dir.rmdir()


def test_kill_switch_immediately_blocks_eyes_and_hands_and_cli(
    running_eyes_socket: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify engaging the menu-bar Kill Switch refuses all `body eyes` and `body hands` calls without PyObjC fallback."""
    # Active: CLI `body eyes` and `body hands` succeed transparently over `foveal.sock`
    rc_ok = cli_main(["eyes", "--app", "Finder", "--json"])
    assert rc_ok == 0
    eyes_json = json.loads(capsys.readouterr().out)
    assert eyes_json["helper_provider"] == "Whole Body Compute.app"
    assert eyes_json["element_count"] == 3

    rc_hands = cli_main(["hands", "--app", "Finder", "--click", "e1", "--json"])
    assert rc_hands == 0
    hands_json = json.loads(capsys.readouterr().out)
    assert hands_json["actuation"]["status"] == "success"

    # Engage Kill Switch
    pause_res = send_foveal_socket_request({"op": "test_set_paused", "paused": True})
    assert pause_res is not None and pause_res["paused"] is True

    # Doctor reports warn + PAUSED state
    doc = check_organ_the_eyes()
    assert doc["status"] == "warn"
    assert "Whole Body Compute.app:PAUSED" in doc["details"]

    # Both `body eyes` and `body hands` return exit code 1 with kill_switch=True (never falling back to PyObjC)
    rc_paused_eyes = cli_main(["eyes", "--app", "Finder", "--json"])
    assert rc_paused_eyes == 1
    paused_out = json.loads(capsys.readouterr().out)
    assert paused_out["kill_switch"] is True

    rc_paused_hands = cli_main(["hands", "--app", "Finder", "--click", "e1"])
    assert rc_paused_hands == 1
    assert "Kill Switch is active" in capsys.readouterr().out

    # Resume Kill Switch and verify instant recovery
    resumed = send_foveal_socket_request({"op": "test_set_paused", "paused": False})
    assert resumed is not None and resumed["paused"] is False
    assert cli_main(["eyes", "--app", "Finder"]) == 0
    text_out = capsys.readouterr().out
    assert "[e1] [AXButton]" in text_out


def test_native_swift_blindfolds_vaults_system_settings_origins_and_secure_fields(
    running_eyes_socket: Path,
) -> None:
    """Verify Swift helper enforces vault, System Settings/SecurityAgent, web origin, and AXSecureTextField blindfolds."""
    # 1. Password vaults blindfolded before returning data
    for vault in ("1Password", "Bitwarden", "Passwords", "Keychain Access"):
        snap = FovealVisionEngine.capture_snapshot(app_name=vault)
        assert snap.window_title == "[BLOCKED_SECURE_VAULT]"
        assert snap.interactive_elements == []

        snap_act = FovealVisionEngine.capture_snapshot(app_name=vault, click="e1")
        assert snap_act.actuation is not None
        assert snap_act.actuation["security_blocked"] is True

    # 2. System Settings & SecurityAgent blindfolded
    for sys_app in ("System Settings", "SecurityAgent"):
        snap_sys = FovealVisionEngine.capture_snapshot(app_name=sys_app)
        assert snap_sys.window_title == "[BLOCKED_SYSTEM_SECURITY_SURFACE]"
        assert snap_sys.interactive_elements == []

    # 3. Sensitive web origins blindfolded
    for origin_title in ("Vault - vault.bitwarden.com", "my.1password.com - Chrome", "console.aws.amazon.com/iam"):
        snap_orig = FovealVisionEngine.capture_snapshot(app_name="Google Chrome", window_title_query=origin_title)
        assert snap_orig.window_title == "[BLOCKED_SENSITIVE_ORIGIN]"
        assert snap_orig.interactive_elements == []

    # 4. AXSecureTextField value redacted on read and blocked on write
    snap_clean = FovealVisionEngine.capture_snapshot(app_name="Finder")
    sec_elems = [e for e in snap_clean.interactive_elements if e.id == "e3"]
    assert len(sec_elems) == 1
    assert sec_elems[0].is_secure is True
    assert sec_elems[0].value == "[REDACTED_PASSWORD]"

    snap_sec_write = FovealVisionEngine.capture_snapshot(app_name="Finder", click="e3", paste_text="hunter2")
    assert snap_sec_write.actuation is not None
    assert snap_sec_write.actuation["security_blocked"] is True


def test_launchd_daemon_wires_wbc_eyes_app_bundle(
    wbc_eyes_bundle: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `build_launchd_plist_specs` wires `Whole Body Compute.app` into `com.wholebodycompute.foveal`."""
    monkeypatch.setenv("WBC_EYES_APP_PATH", str(wbc_eyes_bundle))
    resolved = resolve_wbc_eyes_app_executable()
    assert resolved == (wbc_eyes_bundle / "Contents" / "MacOS" / "WholeBodyCompute").resolve()

    specs = build_launchd_plist_specs(logs_dir=tmp_path / "logs", include_pro=False)
    foveal_spec = specs["com.wholebodycompute.foveal"]
    assert foveal_spec["AssociatedBundleIdentifiers"] == "com.wholebodycompute.app"
    assert foveal_spec["LimitLoadToSessionType"] == "Aqua"
    assert foveal_spec["ProgramArguments"] == [str(resolved)]


def test_wbc_eyes_pinned_sha256_team_id_verification_and_cli_install(
    wbc_eyes_bundle: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify pinned SHA-256 + Team ID `Y25SHBA3P5` enforcement, zip-slip rejection, and `body eyes --install-app`."""
    import shutil
    import zipfile

    # 1. Signature verifier rejects ad-hoc bundle when Team ID is required
    sig_adhoc_strict = verify_wbc_eyes_app_signature(
        wbc_eyes_bundle,
        expected_team_id=WBC_EYES_TEAM_ID,
        require_team_id=True,
        require_notarized=False,
    )
    assert sig_adhoc_strict["ok"] is False
    assert "TeamIdentifier mismatch" in sig_adhoc_strict["error"]

    sig_adhoc_local = verify_wbc_eyes_app_signature(
        wbc_eyes_bundle,
        require_team_id=False,
        require_notarized=False,
    )
    assert sig_adhoc_local["ok"] is True
    assert sig_adhoc_local["identifier"] == "com.wholebodycompute.app"
    assert sig_adhoc_local["hardened_runtime"] is True
    assert sig_adhoc_local["library_validation"] is True

    # 2. Downloader fails closed on unpinned/zero SHA-256, insecure HTTP URL, and SHA-256 mismatch
    install_dest = tmp_path / "Applications"
    unpinned_res = download_and_install_wbc_eyes_app(
        url="https://example.com/WBC-Eyes.zip",
        expected_sha256="0" * 64,
        install_dir=install_dest,
    )
    assert unpinned_res["status"] == "error"
    assert "not pinned" in unpinned_res["error"].lower()

    insecure_res = download_and_install_wbc_eyes_app(
        url="http://example.com/WBC-Eyes.zip",
        expected_sha256="a" * 64,
        install_dir=install_dest,
    )
    assert insecure_res["status"] == "error"
    assert "HTTPS is required" in insecure_res["error"]

    # Package wbc_eyes_bundle into a zip via ditto
    zip_path = tmp_path / "WBC-Eyes-test.zip"
    subprocess.run(
        ["ditto", "-c", "-k", "--keepParent", str(wbc_eyes_bundle), str(zip_path)],
        check=True,
        capture_output=True,
    )
    zip_sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()

    mismatch_res = download_and_install_wbc_eyes_app(
        url=zip_path.as_uri(),
        expected_sha256="b" * 64,
        install_dir=install_dest,
        require_notarized=False,
    )
    assert mismatch_res["status"] == "error"
    assert "SHA-256 mismatch" in mismatch_res["error"]

    # 3. Downloader enforces Team ID Y25SHBA3P5 on extracted bundle (rejects adhoc zip)
    team_reject_res = download_and_install_wbc_eyes_app(
        url=zip_path.as_uri(),
        expected_sha256=zip_sha,
        expected_team_id=WBC_EYES_TEAM_ID,
        install_dir=install_dest,
        require_notarized=False,
    )
    assert team_reject_res["status"] == "error"
    assert "TeamIdentifier mismatch" in team_reject_res["error"]

    # 4. Zip-slip traversal archive is rejected
    bad_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("../escape.txt", "bad")
    bad_sha = hashlib.sha256(bad_zip.read_bytes()).hexdigest()
    slip_res = download_and_install_wbc_eyes_app(
        url=bad_zip.as_uri(),
        expected_sha256=bad_sha,
        install_dir=install_dest,
        require_notarized=False,
    )
    assert slip_res["status"] == "error"
    assert "path-traversal" in slip_res["error"]

    # 5. Live Developer ID signed bundle (if built in repo dist/) passes Y25SHBA3P5 signature check
    dist_bundle = REPO_ROOT / "dist" / "Whole Body Compute.app"
    if dist_bundle.is_dir():
        dist_sig = verify_wbc_eyes_app_signature(
            dist_bundle,
            expected_team_id=WBC_EYES_TEAM_ID,
            require_team_id=True,
            require_notarized=False,
        )
        assert dist_sig["ok"] is True
        assert dist_sig["team_id"] == WBC_EYES_TEAM_ID

        dist_zip = tmp_path / "Whole-Body-Compute-devid.zip"
        subprocess.run(
            ["ditto", "-c", "-k", "--keepParent", str(dist_bundle), str(dist_zip)],
            check=True,
            capture_output=True,
        )
        dist_sha = hashlib.sha256(dist_zip.read_bytes()).hexdigest()

        orig_verify = verify_wbc_eyes_app_signature
        monkeypatch.setattr(
            "whole_body_compute.daemon.verify_wbc_eyes_app_signature",
            lambda bundle, **kw: orig_verify(bundle, **{**kw, "require_notarized": False}),
        )
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("WBC_SKIP_DAEMON_WIRE", "1")
        rc = cli_main(
            [
                "eyes",
                "--install-app",
                "--app-url",
                dist_zip.as_uri(),
                "--app-sha256",
                dist_sha,
                "--json",
            ]
        )
        assert rc == 0
        cli_out = json.loads(capsys.readouterr().out)
        assert cli_out["status"] == "ok"
        assert cli_out["team_id"] == WBC_EYES_TEAM_ID
        assert cli_out["sha256"] == dist_sha
        assert (tmp_path / "Applications" / "Whole Body Compute.app" / "Contents" / "MacOS" / "WholeBodyCompute").is_file()

    # 6. Homebrew Cask syntax check
    cask_path = REPO_ROOT / "packaging" / "homebrew" / "Casks" / "whole-body-compute.rb"
    assert cask_path.is_file()
    cask_txt = cask_path.read_text(encoding="utf-8")
    assert 'cask "whole-body-compute"' in cask_txt
    assert WBC_EYES_TEAM_ID in cask_txt
    ruby_bin = shutil.which("ruby")
    if ruby_bin:
        chk = subprocess.run([ruby_bin, "-c", str(cask_path)], capture_output=True, text=True, check=True)
        assert "Syntax OK" in chk.stdout
