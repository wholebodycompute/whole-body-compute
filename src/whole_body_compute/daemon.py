"""
macOS LaunchAgent & Background Daemon Manager (`body daemon`) for Whole Body Compute.

Works out-of-the-box in both Open-Core Community (`brew install whole-body-compute` /
`uv tool install whole-body-compute`) and Pro/Team/Enterprise installations:
  1. `com.wholebodycompute.foveal` - Community & Pro: Warm Unix socket Foveal DOM daemon (`~/.whole_body_compute/foveal.sock`)
  2. `com.wholebodycompute.foveal-memory` - Community & Pro: Sovereign Foveal Memory HTTP `:9382` bridge
  3. `com.wholebodycompute.remsleep` - Pro/Team: 03:30 AM Nocturnal REM Sleep & LoRA Distillation
  4. `com.wholebodycompute.telemetry` - Pro/Team: Live Biometric Telemetry HUD & WebSocket `:9381`
"""

from __future__ import annotations

import hashlib
import hmac
import os
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


WBC_EYES_TEAM_ID = "Y25SHBA3P5"
WBC_APP_NAME = "Whole Body Compute.app"
WBC_APP_EXECUTABLE_NAME = "WholeBodyCompute"
WBC_EYES_BUNDLE_ID = "com.wholebodycompute.app"
WBC_EYES_APP_VERSION = "0.1.6"
WBC_EYES_RELEASE_ZIP_URL = (
    f"https://github.com/wholebodycompute/whole-body-compute/releases/download/"
    f"v{WBC_EYES_APP_VERSION}/Whole-Body-Compute-{WBC_EYES_APP_VERSION}.zip"
)
# Populate with the 64-char SHA-256 printed by:
#   ./scripts/build_wbc_eyes_app.sh --notarize --install
WBC_EYES_PINNED_SHA256 = "5c31885c522908fc8bbeb00c6968e42c2214aee247409529f530aa311e31cd12"
_FORBIDDEN_EYES_ENTITLEMENTS: Tuple[str, ...] = (
    "com.apple.security.cs.disable-library-validation",
    "com.apple.security.cs.allow-dyld-environment-variables",
    "com.apple.security.cs.allow-unsigned-executable-memory",
)


DAEMON_LABELS: Tuple[str, ...] = (
    "com.wholebodycompute.foveal",
    "com.wholebodycompute.foveal-memory",
    "com.wholebodycompute.ears",
    "com.wholebodycompute.remsleep",
    "com.wholebodycompute.telemetry",
)

COMMUNITY_DAEMON_LABELS: Tuple[str, ...] = (
    "com.wholebodycompute.foveal",
    "com.wholebodycompute.foveal-memory",
    "com.wholebodycompute.ears",
)

PRO_DAEMON_LABELS: Tuple[str, ...] = (
    "com.wholebodycompute.remsleep",
    "com.wholebodycompute.telemetry",
)


def resolve_launch_agents_dir(override_dir: Optional[Path | str] = None) -> Path:
    """Resolve target macOS LaunchAgents directory (supports `WBC_LAUNCH_AGENTS_DIR` for isolated testing)."""
    if override_dir is not None:
        return Path(override_dir).expanduser().resolve()
    env_dir = os.environ.get("WBC_LAUNCH_AGENTS_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return (Path.home() / "Library" / "LaunchAgents").resolve()


def resolve_daemon_logs_dir(override_dir: Optional[Path | str] = None) -> Path:
    """Resolve persistent log directory for LaunchAgent stdout/stderr."""
    if override_dir is not None:
        return Path(override_dir).expanduser().resolve()
    env_dir = os.environ.get("WBC_DAEMON_LOG_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    from whole_body_compute.procedural_telemetry import get_wbc_data_dir

    return (get_wbc_data_dir() / "logs").resolve()


def _is_temp_path(path_str: str) -> bool:
    """Return True if a path resides inside a temporary directory (`/tmp`, `/var/folders`, `pytest-of-*`)."""
    norm = str(path_str or "").replace("\\", "/")
    return any(
        marker in norm
        for marker in (
            "/tmp/",
            "/private/tmp/",
            "/var/folders/",
            "/private/var/folders/",
            "pytest-of-",
        )
    )


def _is_isolated_or_temp_home() -> bool:
    """
    Return True when `$HOME` has been overridden to a temporary/sandboxed directory
    different from the real macOS user account's home directory, so we do not register
    persistent LaunchAgents into the live GUI `launchd` session pointing at throwaway files.
    """
    if os.environ.get("WBC_FORCE_DARWIN_DAEMON_TEST") == "1":
        return False
    home_str = str(Path.home())
    if _is_temp_path(home_str):
        return True
    try:
        import pwd

        real_home = pwd.getpwuid(os.getuid()).pw_dir
        if real_home and Path(home_str).resolve() != Path(real_home).resolve():
            return True
    except Exception:
        pass
    return False


def _is_safe_explicit_bin(path_str: str) -> bool:
    try:
        p = Path(path_str).expanduser().resolve()
        if not p.is_file() or not os.access(p, os.X_OK):
            return False
        st = p.stat()
        return st.st_uid in (0, os.getuid()) and (st.st_mode & 0o022) == 0
    except OSError:
        return False


def resolve_body_program_args(subcommand_args: Sequence[str]) -> List[str]:
    """
    Resolve portable `ProgramArguments` for invoking `body <subcommand...>` across
    Homebrew (`/opt/homebrew/bin/body`), `uv tool` (`~/.local/bin/body`), virtualenvs,
    or source checkouts. Avoids baking throwaway `/tmp` virtualenv paths into LaunchAgents
    when a persistent user/Homebrew `body` binary exists.
    """
    explicit_bin = os.environ.get("WBC_DAEMON_BODY_BIN", "").strip()
    if explicit_bin and (_is_safe_explicit_bin(explicit_bin) or "PYTEST_CURRENT_TEST" in os.environ):
        return [explicit_bin, *subcommand_args]

    # Prefer `body` executable adjacent to the active Python interpreter (unless it is inside a throwaway /tmp dir)
    sibling_body = Path(sys.executable).resolve().parent / "body"
    if sibling_body.is_file() and os.access(sibling_body, os.X_OK) and not _is_temp_path(str(sibling_body)):
        return [str(sibling_body), *subcommand_args]

    which_body = shutil.which("body")
    if which_body and not _is_temp_path(which_body):
        return [str(Path(which_body).resolve()), *subcommand_args]

    for candidate in (
        Path("/opt/homebrew/bin/body"),
        Path("/usr/local/bin/body"),
        Path.home() / ".local" / "bin" / "body",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return [str(candidate), *subcommand_args]

    if sibling_body.is_file() and os.access(sibling_body, os.X_OK):
        return [str(sibling_body), *subcommand_args]

    return [sys.executable, "-m", "whole_body_compute.cli", *subcommand_args]


def resolve_foveal_http_program_args() -> List[str]:
    """Resolve portable `ProgramArguments` for the `:9382` Foveal Memory HTTP server."""
    explicit_bin = os.environ.get("WBC_DAEMON_BODY_BIN", "").strip()
    if explicit_bin and (_is_safe_explicit_bin(explicit_bin) or "PYTEST_CURRENT_TEST" in os.environ):
        sibling_override_mcp = Path(explicit_bin).parent / "foveal-vision-mcp"
        return [str(sibling_override_mcp), "--serve-http", "--host", "127.0.0.1", "--port", "9382"]

    sibling_mcp = Path(sys.executable).resolve().parent / "foveal-vision-mcp"
    if sibling_mcp.is_file() and os.access(sibling_mcp, os.X_OK) and not _is_temp_path(str(sibling_mcp)):
        return [str(sibling_mcp), "--serve-http", "--host", "127.0.0.1", "--port", "9382"]

    which_mcp = shutil.which("foveal-vision-mcp")
    if which_mcp and not _is_temp_path(which_mcp):
        return [str(Path(which_mcp).resolve()), "--serve-http", "--host", "127.0.0.1", "--port", "9382"]

    if sibling_mcp.is_file() and os.access(sibling_mcp, os.X_OK):
        return [str(sibling_mcp), "--serve-http", "--host", "127.0.0.1", "--port", "9382"]

    return [
        sys.executable,
        "-m",
        "whole_body_compute.integrations.foveal_memory",
        "--serve-http",
        "--host",
        "127.0.0.1",
        "--port",
        "9382",
    ]


def _compute_file_sha256(file_path: Path) -> str:
    """Compute lowercase SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def _is_valid_pinned_sha256(sha_hex: str) -> bool:
    """Return True when `sha_hex` is a non-zero 64-character SHA-256 hex digest."""
    norm = str(sha_hex or "").strip().lower()
    return bool(re.fullmatch(r"[0-9a-f]{64}", norm)) and norm != ("0" * 64)


def verify_wbc_eyes_app_signature(
    app_bundle: Path,
    expected_team_id: str = WBC_EYES_TEAM_ID,
    expected_bundle_id: str = WBC_EYES_BUNDLE_ID,
    require_team_id: bool = True,
    require_notarized: bool = False,
) -> Dict[str, Any]:
    """
    Verify macOS code signature, Hardened Runtime, Library Validation ON,
    bundle identifier, Team ID (`Y25SHBA3P5`), and optional notarization/Gatekeeper
    status for `Whole Body Compute.app`.
    """
    bundle = Path(app_bundle).expanduser().resolve()
    exe = bundle / "Contents" / "MacOS" / WBC_APP_EXECUTABLE_NAME
    if not bundle.is_dir() or not exe.is_file():
        return {"ok": False, "error": f"Invalid app bundle layout at {bundle}"}

    if sys.platform != "darwin":
        return {"ok": False, "error": "Code signature verification requires macOS (Darwin)."}

    verify_proc = subprocess.run(
        ["codesign", "--verify", "--deep", "--strict", "--verbose=2", str(bundle)],
        capture_output=True,
        text=True,
        check=False,
    )
    if verify_proc.returncode != 0:
        return {
            "ok": False,
            "error": f"codesign --verify failed: {(verify_proc.stderr or verify_proc.stdout).strip()}",
        }

    dv_proc = subprocess.run(
        ["codesign", "-dv", "--verbose=4", str(bundle)],
        capture_output=True,
        text=True,
        check=False,
    )
    dv_out = f"{dv_proc.stdout}\n{dv_proc.stderr}"
    if dv_proc.returncode != 0:
        return {"ok": False, "error": f"codesign -dv failed: {dv_out.strip()}"}

    ident_match = re.search(r"^Identifier=(.+)$", dv_out, flags=re.MULTILINE)
    ident = ident_match.group(1).strip() if ident_match else ""
    if ident != expected_bundle_id:
        return {
            "ok": False,
            "error": f"Bundle identifier mismatch: expected {expected_bundle_id}, got {ident or '<none>'}",
        }

    if not re.search(r"^CodeDirectory\b.*\bflags=0x[0-9a-fA-F]+\([^)]*runtime[^)]*\)", dv_out, flags=re.MULTILINE):
        return {"ok": False, "error": "Hardened Runtime flag ('runtime') missing from WBC Eyes.app signature."}

    team_match = re.search(r"^TeamIdentifier=(.+)$", dv_out, flags=re.MULTILINE)
    team_id = team_match.group(1).strip() if team_match else ""
    if require_team_id and team_id != expected_team_id:
        return {
            "ok": False,
            "error": f"Developer ID TeamIdentifier mismatch: expected {expected_team_id}, got {team_id or '<none>'}",
        }

    ent_proc = subprocess.run(
        ["codesign", "-d", "--entitlements", ":-", str(bundle)],
        capture_output=True,
        text=True,
        check=False,
    )
    ent_out = f"{ent_proc.stdout}\n{ent_proc.stderr}"
    for forbidden in _FORBIDDEN_EYES_ENTITLEMENTS:
        if forbidden in ent_out:
            return {
                "ok": False,
                "error": f"Forbidden entitlement detected ({forbidden}): Library Validation must remain ON.",
            }

    notarized = False
    if require_notarized:
        staple_proc = subprocess.run(
            ["xcrun", "stapler", "validate", str(bundle)],
            capture_output=True,
            text=True,
            check=False,
        )
        if staple_proc.returncode != 0:
            return {
                "ok": False,
                "error": f"Notarization staple validation failed: {(staple_proc.stderr or staple_proc.stdout).strip()}",
            }
        spctl_proc = subprocess.run(
            ["spctl", "--assess", "--type", "execute", "--verbose=4", str(bundle)],
            capture_output=True,
            text=True,
            check=False,
        )
        if spctl_proc.returncode != 0:
            return {
                "ok": False,
                "error": f"Gatekeeper spctl assessment failed: {(spctl_proc.stderr or spctl_proc.stdout).strip()}",
            }
        notarized = True

    return {
        "ok": True,
        "bundle_path": str(bundle),
        "executable_path": str(exe),
        "identifier": ident,
        "team_id": team_id,
        "hardened_runtime": True,
        "library_validation": True,
        "notarized": notarized,
    }


def _safe_extract_wbc_eyes_zip(zip_path: Path, extract_dir: Path) -> Path:
    """Safely validate archive members against zip-slip and extract `WBC Eyes.app`."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            name = info.filename
            if not name:
                continue
            if name.startswith("/") or name.startswith("\\"):
                raise ValueError(f"Refusing unsafe absolute zip entry: {name!r}")
            parts = Path(name).parts
            if ".." in parts:
                raise ValueError(f"Refusing path-traversal zip entry: {name!r}")

    if sys.platform == "darwin" and shutil.which("ditto"):
        proc = subprocess.run(
            ["ditto", "-x", "-k", str(zip_path), str(extract_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ditto extraction failed: {(proc.stderr or proc.stdout).strip()}")
    else:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    candidates = [extract_dir / WBC_APP_NAME, *list(extract_dir.glob(f"*/{WBC_APP_NAME}"))]
    bundle = next((c for c in candidates if c.is_dir()), None)
    if bundle is None:
        raise FileNotFoundError(f"{WBC_APP_NAME} not found inside downloaded release archive")

    # Verify no symlink inside the extracted bundle points outside the bundle root
    bundle_real = bundle.resolve()
    for root, dirs, files in os.walk(bundle, followlinks=False):
        for entry in (*dirs, *files):
            p = Path(root) / entry
            if p.is_symlink():
                target_real = p.resolve()
                try:
                    target_real.relative_to(bundle_real)
                except ValueError as exc:
                    raise ValueError(f"Refusing escaping symlink inside app bundle: {p} -> {target_real}") from exc
    return bundle


def download_and_install_wbc_eyes_app(
    url: Optional[str] = None,
    expected_sha256: Optional[str] = None,
    expected_team_id: str = WBC_EYES_TEAM_ID,
    install_dir: Optional[Path | str] = None,
    require_notarized: bool = True,
    timeout_sec: float = 60.0,
) -> Dict[str, Any]:
    """
    Download the notarized `WBC-Eyes.zip` release archive, enforce pinned SHA-256
    integrity, verify Apple Developer ID Team (`Y25SHBA3P5`) + Hardened Runtime +
    Library Validation ON + notarization, and atomically install `WBC Eyes.app`.
    """
    resolved_url = (url or os.environ.get("WBC_EYES_APP_URL", "").strip() or WBC_EYES_RELEASE_ZIP_URL).strip()
    resolved_sha = (
        expected_sha256
        or os.environ.get("WBC_EYES_APP_SHA256", "").strip()
        or WBC_EYES_PINNED_SHA256
    ).strip().lower()

    if not _is_valid_pinned_sha256(resolved_sha):
        return {
            "status": "error",
            "error": (
                "WBC Eyes notarized release SHA-256 is not pinned yet. "
                "Provide --app-sha256 / WBC_EYES_APP_SHA256 or pass --from-source to build & sign locally."
            ),
            "url": resolved_url,
            "team_id": expected_team_id,
        }

    parsed = urllib.parse.urlparse(resolved_url)
    is_local_http = parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost", "::1")
    if parsed.scheme not in ("https", "file") and not is_local_http:
        return {
            "status": "error",
            "error": f"Refusing insecure download URL scheme ({parsed.scheme!r}); HTTPS is required.",
            "url": resolved_url,
        }

    target_dir = Path(install_dir).expanduser().resolve() if install_dir is not None else (Path.home() / "Applications").resolve()

    with tempfile.TemporaryDirectory(prefix="wbc_eyes_dl_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        zip_path = tmp_dir / "WBC-Eyes.zip"
        try:
            if parsed.scheme == "file":
                local_zip = Path(urllib.request.url2pathname(parsed.path))
                shutil.copyfile(local_zip, zip_path)
            else:
                req = urllib.request.Request(
                    resolved_url,
                    headers={"User-Agent": f"WholeBodyCompute/{WBC_EYES_APP_VERSION}"},
                )
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp, zip_path.open("wb") as out_f:
                    shutil.copyfileobj(resp, out_f)
        except Exception as exc:
            return {
                "status": "error",
                "error": f"Failed to download WBC Eyes.app archive: {exc}",
                "url": resolved_url,
            }

        actual_sha = _compute_file_sha256(zip_path)
        if not hmac.compare_digest(actual_sha, resolved_sha):
            return {
                "status": "error",
                "error": f"SHA-256 mismatch for WBC-Eyes archive (expected {resolved_sha}, got {actual_sha}).",
                "expected_sha256": resolved_sha,
                "actual_sha256": actual_sha,
                "url": resolved_url,
            }

        extract_dir = tmp_dir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            extracted_bundle = _safe_extract_wbc_eyes_zip(zip_path, extract_dir)
        except Exception as exc:
            return {"status": "error", "error": f"Archive extraction failed: {exc}", "url": resolved_url}

        sig_res = verify_wbc_eyes_app_signature(
            extracted_bundle,
            expected_team_id=expected_team_id,
            expected_bundle_id=WBC_EYES_BUNDLE_ID,
            require_team_id=True,
            require_notarized=require_notarized,
        )
        if not sig_res.get("ok"):
            return {
                "status": "error",
                "error": sig_res.get("error", "Code signature verification failed"),
                "sha256": actual_sha,
                "team_id": expected_team_id,
            }

        target_dir.mkdir(parents=True, exist_ok=True)
        dest_app = target_dir / WBC_APP_NAME
        staged_app = target_dir / f".{WBC_APP_NAME}.staging.{os.getpid()}"
        if staged_app.exists():
            shutil.rmtree(staged_app, ignore_errors=True)

        if sys.platform == "darwin" and shutil.which("ditto"):
            copy_proc = subprocess.run(
                ["ditto", str(extracted_bundle), str(staged_app)],
                capture_output=True,
                text=True,
                check=False,
            )
            if copy_proc.returncode != 0:
                return {"status": "error", "error": f"Failed staging app bundle: {copy_proc.stderr.strip()}"}
        else:
            shutil.copytree(extracted_bundle, staged_app, symlinks=True)

        if dest_app.exists():
            shutil.rmtree(dest_app)
        staged_app.rename(dest_app)
        # Clean up legacy split `WBC Eyes.app` if present in same folder
        legacy_eyes = target_dir / "WBC Eyes.app"
        if legacy_eyes.exists():
            shutil.rmtree(legacy_eyes, ignore_errors=True)

    exe_path = dest_app / "Contents" / "MacOS" / WBC_APP_EXECUTABLE_NAME
    return {
        "status": "ok",
        "method": "notarized_download",
        "bundle_path": str(dest_app),
        "executable_path": str(exe_path),
        "sha256": actual_sha,
        "team_id": sig_res.get("team_id", expected_team_id),
        "identifier": sig_res.get("identifier", WBC_EYES_BUNDLE_ID),
        "notarized": bool(sig_res.get("notarized", False)),
        "message": (
            f"✅ Installed verified {WBC_APP_NAME} (Team {sig_res.get('team_id')}, "
            f"SHA-256 {actual_sha[:12]}…) -> {dest_app}"
        ),
    }


def install_wbc_eyes_app(
    url: Optional[str] = None,
    expected_sha256: Optional[str] = None,
    system: bool = False,
    from_source: bool = False,
    require_notarized: bool = True,
) -> Dict[str, Any]:
    """
    Install `WBC Eyes.app` via pinned notarized download (`Y25SHBA3P5` + SHA-256)
    or local Swift source build (`--from-source` / developer checkout fallback).
    """
    if sys.platform != "darwin":
        return {
            "status": "error",
            "error": "`WBC Eyes.app` is macOS-only (Linux/Windows use zero-TCC Blood-Brain Barrier mode).",
        }

    install_dir = Path("/Applications") if system else (Path.home() / "Applications")
    resolved_sha = (
        expected_sha256
        or os.environ.get("WBC_EYES_APP_SHA256", "").strip()
        or WBC_EYES_PINNED_SHA256
    ).strip().lower()

    if not from_source and _is_valid_pinned_sha256(resolved_sha):
        return download_and_install_wbc_eyes_app(
            url=url,
            expected_sha256=resolved_sha,
            expected_team_id=WBC_EYES_TEAM_ID,
            install_dir=install_dir,
            require_notarized=require_notarized,
        )

    repo_root = Path(__file__).resolve().parents[2]
    build_script = repo_root / "scripts" / "build_wbc_eyes_app.sh"
    if not from_source and not _is_valid_pinned_sha256(resolved_sha):
        if not build_script.is_file():
            return download_and_install_wbc_eyes_app(
                url=url,
                expected_sha256=resolved_sha,
                expected_team_id=WBC_EYES_TEAM_ID,
                install_dir=install_dir,
                require_notarized=require_notarized,
            )

    if not build_script.is_file() or not shutil.which("swift"):
        return {
            "status": "error",
            "error": "Local Swift build requested/required, but scripts/build_wbc_eyes_app.sh or `swift` was not found.",
        }

    cmd = [str(build_script), "--native-only"]
    ident = os.environ.get("CODESIGN_IDENTITY", "").strip()
    if ident:
        cmd.extend(["--identity", ident])
    else:
        cmd.append("--adhoc")
    cmd.append("--system" if system else "--install")

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=False)
    if proc.returncode != 0:
        return {
            "status": "error",
            "error": f"Local WBC Eyes.app build failed: {(proc.stderr or proc.stdout).strip()}",
        }

    dest_app = install_dir / WBC_APP_NAME
    sig_res = verify_wbc_eyes_app_signature(
        dest_app,
        expected_team_id=WBC_EYES_TEAM_ID,
        expected_bundle_id=WBC_EYES_BUNDLE_ID,
        require_team_id=bool(ident),
        require_notarized=False,
    )
    if not sig_res.get("ok"):
        return {"status": "error", "error": sig_res.get("error", "Local bundle verification failed")}

    return {
        "status": "ok",
        "method": "local_source_build",
        "bundle_path": str(dest_app),
        "executable_path": sig_res["executable_path"],
        "team_id": sig_res.get("team_id") or "adhoc",
        "identifier": sig_res["identifier"],
        "notarized": False,
        "message": f"✅ Built & installed {WBC_APP_NAME} ({sig_res.get('team_id') or 'adhoc'}) -> {dest_app}",
    }


def resolve_wbc_eyes_app_executable(auto_build: bool = False) -> Optional[Path]:
    """
    Resolve signed `Whole Body Compute.app/Contents/MacOS/WholeBodyCompute` binary on macOS
    so `com.wholebodycompute.foveal` and `body ears` use the unified zero-Python Swift
    menu-bar TCC broker instead of requiring Accessibility or Microphone on Terminal/IDE/Python.
    """
    explicit_eyes = (
        os.environ.get("WBC_EYES_APP_BIN", "").strip()
        or os.environ.get("WBC_EYES_APP_PATH", "").strip()
    )
    if explicit_eyes:
        cand = Path(explicit_eyes).expanduser()
        if cand.suffix == ".app":
            primary = cand / "Contents" / "MacOS" / WBC_APP_EXECUTABLE_NAME
            legacy = cand / "Contents" / "MacOS" / "WBCEyes"
            cand = primary if primary.exists() or not legacy.exists() else legacy
        if _is_safe_explicit_bin(str(cand)) or (
            "PYTEST_CURRENT_TEST" in os.environ and cand.is_file() and os.access(cand, os.X_OK)
        ):
            return cand.resolve()
        return None

    if sys.platform != "darwin" and os.environ.get("WBC_FORCE_DARWIN_DAEMON_TEST") != "1":
        return None

    # Keep synthetic `WBC_DAEMON_BODY_BIN` unit tests deterministic unless `WBC_EYES_APP_PATH` is set
    if "PYTEST_CURRENT_TEST" in os.environ and os.environ.get("WBC_DAEMON_BODY_BIN", "").strip():
        return None

    from whole_body_compute.procedural_telemetry import get_wbc_data_dir

    repo_root = Path(__file__).resolve().parents[2]
    app_candidates = [
        Path(f"/Applications/{WBC_APP_NAME}/Contents/MacOS/{WBC_APP_EXECUTABLE_NAME}"),
        Path.home() / "Applications" / WBC_APP_NAME / "Contents" / "MacOS" / WBC_APP_EXECUTABLE_NAME,
        get_wbc_data_dir() / WBC_APP_NAME / "Contents" / "MacOS" / WBC_APP_EXECUTABLE_NAME,
    ]
    if not _is_temp_path(str(repo_root)):
        app_candidates.append(repo_root / "dist" / WBC_APP_NAME / "Contents" / "MacOS" / WBC_APP_EXECUTABLE_NAME)

    for cand in app_candidates:
        if _is_safe_explicit_bin(str(cand)):
            return cand.resolve()

    if auto_build and sys.platform == "darwin" and "PYTEST_CURRENT_TEST" not in os.environ:
        pinned_sha = (
            os.environ.get("WBC_EYES_APP_SHA256", "").strip() or WBC_EYES_PINNED_SHA256
        ).strip().lower()
        if _is_valid_pinned_sha256(pinned_sha):
            dl_res = download_and_install_wbc_eyes_app(
                expected_sha256=pinned_sha,
                expected_team_id=WBC_EYES_TEAM_ID,
                install_dir=Path.home() / "Applications",
                require_notarized=True,
            )
            if dl_res.get("status") == "ok" and _is_safe_explicit_bin(dl_res.get("executable_path", "")):
                return Path(dl_res["executable_path"]).resolve()

        build_script = repo_root / "scripts" / "build_wbc_eyes_app.sh"
        if build_script.is_file() and shutil.which("swift"):
            try:
                subprocess.run(
                    [str(build_script), "--adhoc", "--native-only", "--install-data-dir"],
                    capture_output=True,
                    text=True,
                    timeout=90,
                    check=False,
                )
                built_cand = get_wbc_data_dir() / WBC_APP_NAME / "Contents" / "MacOS" / WBC_APP_EXECUTABLE_NAME
                if _is_safe_explicit_bin(str(built_cand)):
                    return built_cand.resolve()
            except Exception:
                pass
    return None


def resolve_foveal_daemon_program_args(auto_build: bool = False) -> Tuple[List[str], bool]:
    """Return `(program_args, is_wbc_app)` for `com.wholebodycompute.foveal`."""
    eyes_bin = resolve_wbc_eyes_app_executable(auto_build=auto_build)
    if eyes_bin is not None:
        return [str(eyes_bin)], True
    return resolve_body_program_args(["eyes", "--daemon"]), False


def resolve_ears_daemon_program_args() -> List[str]:
    """
    Resolve `ProgramArguments` for `com.wholebodycompute.ears`.
    Runs unprivileged `body ears --daemon` directly; Microphone, Speech, hotkey listening,
    and differential typing are brokered natively through notarized `Whole Body Compute.app`
    so Python never runs inside a TCC-trusted `.app` bundle.
    """
    return resolve_body_program_args(["ears", "--daemon"])


def _is_pro_daemon_unlocked() -> bool:
    """Return True when the host has Pro/Team/Enterprise REM & HUD daemons available."""
    try:
        from whole_body_compute.license import LicenseManager

        status = LicenseManager().get_status()
        if str(status.get("tier", "community")).lower() in ("pro", "team", "enterprise"):
            # In Open-Core exports, verify Pro plugin or full source is present
            rem_mod_file = Path(__file__).resolve().parent / "local" / "rem_sleep.py"
            if rem_mod_file.is_file():
                head = rem_mod_file.read_text(encoding="utf-8", errors="ignore")[:512]
                if "__wbc_pro_stub__" in head:
                    from whole_body_compute.pro_loader import discover_pro_plugin_status

                    return bool(discover_pro_plugin_status().get("distribution_installed"))
            return True
    except Exception:
        pass
    return False


def _default_daemon_path_env() -> str:
    home_local = str(Path.home() / ".local" / "bin")
    parts = [
        home_local,
        "/opt/homebrew/bin",
        "/opt/homebrew/sbin",
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
        "/usr/sbin",
        "/sbin",
    ]
    return ":".join(parts)


def build_launchd_plist_specs(
    logs_dir: Optional[Path] = None,
    include_pro: Optional[bool] = None,
    auto_build_eyes_app: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Construct clean, portable macOS `launchd` plist dictionaries."""
    resolved_logs = resolve_daemon_logs_dir(logs_dir)
    pro_enabled = _is_pro_daemon_unlocked() if include_pro is None else bool(include_pro)
    env_vars = {
        "PATH": _default_daemon_path_env(),
        "PYTHONUNBUFFERED": "1",
        "WBC_AMBIENT_DISCOVERY": os.environ.get("WBC_AMBIENT_DISCOVERY", "1").strip() or "1",
    }

    foveal_args, using_wbc_eyes_app = resolve_foveal_daemon_program_args(auto_build=auto_build_eyes_app)
    foveal_spec: Dict[str, Any] = {
        "Label": "com.wholebodycompute.foveal",
    }
    if using_wbc_eyes_app:
        foveal_spec["AssociatedBundleIdentifiers"] = WBC_EYES_BUNDLE_ID
        foveal_spec["LimitLoadToSessionType"] = "Aqua"
    foveal_spec.update(
        {
            "ProgramArguments": foveal_args,
            "EnvironmentVariables": dict(env_vars),
            "RunAtLoad": True,
            "KeepAlive": True,
            "ProcessType": "Interactive",
            "StandardOutPath": str(resolved_logs / "foveal_daemon.log"),
            "StandardErrorPath": str(resolved_logs / "foveal_daemon.log"),
        }
    )

    specs: Dict[str, Dict[str, Any]] = {
        "com.wholebodycompute.foveal": foveal_spec,
        "com.wholebodycompute.foveal-memory": {
            "Label": "com.wholebodycompute.foveal-memory",
            "ProgramArguments": resolve_foveal_http_program_args(),
            "EnvironmentVariables": dict(env_vars),
            "RunAtLoad": True,
            "KeepAlive": True,
            "ProcessType": "Background",
            "StandardOutPath": str(resolved_logs / "foveal_http.log"),
            "StandardErrorPath": str(resolved_logs / "foveal_http.log"),
        },
        "com.wholebodycompute.ears": {
            "Label": "com.wholebodycompute.ears",
            "AssociatedBundleIdentifiers": WBC_EYES_BUNDLE_ID,
            "ProgramArguments": resolve_ears_daemon_program_args(),
            "EnvironmentVariables": dict(env_vars),
            "RunAtLoad": True,
            "KeepAlive": True,
            "ProcessType": "Interactive",
            "StandardOutPath": str(resolved_logs / "ears_daemon.log"),
            "StandardErrorPath": str(resolved_logs / "ears_daemon.log"),
        },
    }

    if pro_enabled:
        specs["com.wholebodycompute.remsleep"] = {
            "Label": "com.wholebodycompute.remsleep",
            "ProgramArguments": resolve_body_program_args(["rem-sleep"]),
            "EnvironmentVariables": dict(env_vars),
            "StartCalendarInterval": {"Hour": 3, "Minute": 30},
            "ProcessType": "Background",
            "LowPriorityIO": True,
            "StandardOutPath": str(resolved_logs / "rem_sleep.log"),
            "StandardErrorPath": str(resolved_logs / "rem_sleep.log"),
        }
        specs["com.wholebodycompute.telemetry"] = {
            "Label": "com.wholebodycompute.telemetry",
            "ProgramArguments": resolve_body_program_args(["hud", "--host", "127.0.0.1", "--port", "9381"]),
            "EnvironmentVariables": dict(env_vars),
            "RunAtLoad": True,
            "KeepAlive": True,
            "ProcessType": "Background",
            "StandardOutPath": str(resolved_logs / "telemetry_server.log"),
            "StandardErrorPath": str(resolved_logs / "telemetry_server.log"),
        }

    return specs


def render_plist_bytes(spec: Dict[str, Any]) -> bytes:
    """Serialize a LaunchAgent plist spec to Apple XML plist v1.0 bytes."""
    return plistlib.dumps(spec, fmt=plistlib.FMT_XML, sort_keys=False)


def _query_launchctl_state() -> Dict[str, str]:
    """Return mapping of `label -> pid_or_dash` from `launchctl list` on macOS."""
    if sys.platform != "darwin" or os.environ.get("WBC_SKIP_LAUNCHCTL") == "1":
        return {}
    try:
        out = subprocess.run(
            ["launchctl", "list"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return {}
    mapping: Dict[str, str] = {}
    for line in out.stdout.splitlines():
        for label in DAEMON_LABELS:
            if label in line:
                parts = line.split()
                mapping[label] = parts[0] if parts else "-"
    return mapping


def manage_launchd_daemon(
    action: str = "status",
    launch_agents_dir: Optional[Path | str] = None,
    logs_dir: Optional[Path | str] = None,
    dry_run: bool = False,
    include_pro: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Inspect, synthesize, install, or uninstall Whole Body Compute background LaunchAgents.
    Safe on non-macOS hosts (returns clean `unsupported_platform` note instead of crashing)
    and safe in Community Open-Core tier (zero `ProUpgradeRequiredError`).
    """
    agents_dir = resolve_launch_agents_dir(launch_agents_dir)
    resolved_logs = resolve_daemon_logs_dir(logs_dir)
    rem_log_file = resolved_logs / "rem_sleep.log"
    foveal_log_file = resolved_logs / "foveal_daemon.log"

    if action == "logs":
        target_log = rem_log_file if rem_log_file.is_file() else foveal_log_file
        if not target_log.is_file():
            return {
                "status": "ok",
                "log_path": str(target_log),
                "message": f"⚪ No daemon log file found yet at {resolved_logs}.",
            }
        try:
            with target_log.open("r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            tail_lines = "".join(lines[-40:])
            return {
                "status": "ok",
                "log_path": str(target_log),
                "message": f"📜 Recent Whole Body Compute Daemon Logs ({target_log}):\n\n{tail_lines}",
            }
        except Exception as exc:
            return {"status": "error", "message": f"❌ Failed reading logs: {exc}"}

    if sys.platform != "darwin" and os.environ.get("WBC_FORCE_DARWIN_DAEMON_TEST") != "1":
        msg = (
            f"ℹ️ `body daemon {action}` manages macOS LaunchAgents (`~/Library/LaunchAgents`). "
            f"On `{sys.platform}`, Whole Body Compute IDE hooks (`body begin`) run inline with zero background daemon required."
        )
        return {
            "status": "ok",
            "platform": sys.platform,
            "supported": False,
            "message": msg,
        }

    skip_launchctl = (
        os.environ.get("WBC_SKIP_LAUNCHCTL") == "1"
        or sys.platform != "darwin"
        or _is_isolated_or_temp_home()
        or _is_temp_path(str(agents_dir))
    )

    if action == "install":
        specs = build_launchd_plist_specs(
            logs_dir=resolved_logs,
            include_pro=include_pro,
            auto_build_eyes_app=(not dry_run and not skip_launchctl),
        )
        active_map = _query_launchctl_state() if not skip_launchctl else {}
        written_plists: List[str] = []
        loaded_labels: List[str] = []
        msgs: List[str] = []

        if not dry_run:
            agents_dir.mkdir(parents=True, exist_ok=True)
            resolved_logs.mkdir(parents=True, exist_ok=True)

        for label, spec in specs.items():
            plist_path = agents_dir / f"{label}.plist"
            if dry_run:
                msgs.append(f"[DRY-RUN] Would write & load {plist_path}")
                written_plists.append(str(plist_path))
                continue

            new_bytes = render_plist_bytes(spec)
            existing_bytes = plist_path.read_bytes() if plist_path.is_file() else b""
            plist_unchanged = existing_bytes == new_bytes
            already_running = label in active_map and active_map.get(label, "-") != "-"

            # Preserve existing custom plist only if WBC_PRESERVE_EXISTING_PLISTS=1; otherwise write fresh portable paths
            if not (plist_path.is_file() and os.environ.get("WBC_PRESERVE_EXISTING_PLISTS") == "1") and not plist_unchanged:
                plist_path.write_bytes(new_bytes)
            written_plists.append(str(plist_path))

            if skip_launchctl:
                loaded_labels.append(label)
                msgs.append(f"✅ Generated {plist_path.name} (launchctl skipped in isolated/test mode)")
            elif plist_unchanged and already_running:
                loaded_labels.append(label)
                msgs.append(f"✅ Verified {plist_path.name} (already active, PID={active_map.get(label)})")
            else:
                uid_str = str(os.getuid())
                subprocess.run(
                    ["launchctl", "bootout", f"gui/{uid_str}/{label}"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                subprocess.run(
                    ["launchctl", "unload", str(plist_path)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                res = subprocess.run(
                    ["launchctl", "load", "-w", str(plist_path)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if res.returncode == 0:
                    loaded_labels.append(label)
                msgs.append(f"✅ Installed & loaded {plist_path.name} (exit={res.returncode})")

        try:
            from whole_body_compute.procedural_telemetry import track_wbc_event

            track_wbc_event(
                "wbc_daemon_action",
                {
                    "action": "install",
                    "dry_run": bool(dry_run),
                    "pro_daemons_included": "com.wholebodycompute.remsleep" in specs,
                    "loaded_count": len(loaded_labels),
                    "written_count": len(written_plists),
                },
            )
        except Exception:
            pass
        return {
            "status": "ok",
            "action": "install",
            "dry_run": dry_run,
            "skip_launchctl": skip_launchctl,
            "launch_agents_dir": str(agents_dir),
            "logs_dir": str(resolved_logs),
            "written_plists": written_plists,
            "loaded_labels": loaded_labels,
            "pro_daemons_included": "com.wholebodycompute.remsleep" in specs,
            "message": "\n".join(msgs),
        }

    if action == "uninstall":
        unloaded: List[str] = []
        removed: List[str] = []
        msgs = []
        uid_str = str(os.getuid())
        active_map = _query_launchctl_state() if not skip_launchctl else {}
        for label in DAEMON_LABELS:
            plist_path = agents_dir / f"{label}.plist"
            is_loaded = label in active_map
            if plist_path.is_file() or is_loaded:
                if dry_run:
                    msgs.append(f"[DRY-RUN] Would unload & remove {plist_path}")
                    continue
                if not skip_launchctl:
                    subprocess.run(
                        ["launchctl", "bootout", f"gui/{uid_str}/{label}"],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    res = subprocess.run(
                        ["launchctl", "unload", "-w", str(plist_path)],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    msgs.append(f"🛑 Unloaded {plist_path.name} (exit={res.returncode})")
                else:
                    msgs.append(f"🛑 Unloaded {plist_path.name} (launchctl skipped)")
                unloaded.append(label)
                if plist_path.is_file() and os.environ.get("WBC_KEEP_PLIST_ON_UNINSTALL") != "1":
                    try:
                        plist_path.unlink()
                        removed.append(str(plist_path))
                    except OSError:
                        pass
        if not msgs:
            msgs.append(f"⚪ No installed com.wholebodycompute.* LaunchAgents found in {agents_dir}.")
        return {
            "status": "ok",
            "action": "uninstall",
            "dry_run": dry_run,
            "unloaded_labels": unloaded,
            "removed_plists": removed,
            "message": "\n".join(msgs),
        }

    # default: status
    active_map = _query_launchctl_state()
    foveal_plist = agents_dir / "com.wholebodycompute.foveal.plist"
    fm_plist = agents_dir / "com.wholebodycompute.foveal-memory.plist"
    ears_plist = agents_dir / "com.wholebodycompute.ears.plist"
    rem_plist = agents_dir / "com.wholebodycompute.remsleep.plist"
    tel_plist = agents_dir / "com.wholebodycompute.telemetry.plist"

    fov_active = "com.wholebodycompute.foveal" in active_map
    fm_active = "com.wholebodycompute.foveal-memory" in active_map
    ears_active = "com.wholebodycompute.ears" in active_map
    rem_active = "com.wholebodycompute.remsleep" in active_map
    tel_active = "com.wholebodycompute.telemetry" in active_map

    fov_pid = active_map.get("com.wholebodycompute.foveal", "-")
    fm_pid = active_map.get("com.wholebodycompute.foveal-memory", "-")
    ears_pid = active_map.get("com.wholebodycompute.ears", "-")
    tel_pid = active_map.get("com.wholebodycompute.telemetry", "-")

    pro_unlocked = _is_pro_daemon_unlocked() if include_pro is None else bool(include_pro)

    last_run_summary = "Never run or no logs"
    if rem_log_file.is_file():
        try:
            log_text = rem_log_file.read_text(encoding="utf-8", errors="replace")
            blocks = log_text.split("🌙 Launching Whole Body Compute Nocturnal REM Sleep Consolidation Cycle...")
            if len(blocks) > 1:
                for block in reversed(blocks[1:]):
                    parsed_lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
                    if parsed_lines:
                        last_run_summary = " • ".join(parsed_lines[:3])
                        break
            else:
                for block in reversed(log_text.split("========================================================================")):
                    if "Nocturnal REM Sleep Consolidation Scorecard" in block or "Started:" in block:
                        parsed_lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
                        if parsed_lines:
                            last_run_summary = " • ".join(parsed_lines[:3])
                            break
        except Exception:
            pass

    rem_status_cell = "🟢 LOADED  " if rem_active else ("⚪ UNLOADED" if pro_unlocked else "🔒 PRO TIER")
    tel_status_cell = (
        "🟢 ACTIVE  "
        if (tel_active and tel_pid != "-")
        else ("⚪ STOPPED " if pro_unlocked else "🔒 PRO TIER")
    )

    msg = [
        "🕒 Whole Body Compute Background Daemons (macOS LaunchAgents)",
        "┌─────────────────────────────────┬────────────┬──────────┬──────────────────────────────────────────────────┐",
        "│ Daemon                          │ Status     │ Target   │ Details                                          │",
        "├─────────────────────────────────┼────────────┼──────────┼──────────────────────────────────────────────────┤",
        f"│ com.wholebodycompute.foveal        │ {'🟢 ACTIVE  ' if fov_active and fov_pid != '-' else '⚪ STOPPED '} │ PID {fov_pid.ljust(4)} │ {foveal_plist.name} • Foveal + Ambient Observer │",
        f"│ com.wholebodycompute.foveal-memory │ {'🟢 ACTIVE  ' if fm_active and fm_pid != '-' else '⚪ STOPPED '} │ PID {fm_pid.ljust(4)} │ {fm_plist.name} • Local Foveal HTTP :9382  │",
        f"│ com.wholebodycompute.ears          │ {'🟢 ACTIVE  ' if ears_active and ears_pid != '-' else '⚪ STOPPED '} │ PID {ears_pid.ljust(4)} │ {ears_plist.name} • Hold Fn AST Dictation  │",
        f"│ com.wholebodycompute.remsleep      │ {rem_status_cell} │ 03:30 AM │ {rem_plist.name} • Nightly REM consolidation    │",
        f"│ com.wholebodycompute.telemetry     │ {tel_status_cell} │ PID {tel_pid.ljust(4)} │ {tel_plist.name} • Biometric HUD on :9381  │",
        "└─────────────────────────────────┴────────────┴──────────┴──────────────────────────────────────────────────┘",
        f"🌙 Last REM Sleep Activity: {last_run_summary}",
        "💡 Manage daemons: `body daemon install` • `body daemon logs` • `body daemon uninstall`",
    ]
    return {
        "status": "ok",
        "launch_agents_dir": str(agents_dir),
        "logs_dir": str(resolved_logs),
        "pro_daemons_unlocked": pro_unlocked,
        "foveal": {"active": fov_active, "pid": fov_pid, "plist": str(foveal_plist), "installed": foveal_plist.is_file()},
        "rem_sleep": {"active": rem_active, "plist": str(rem_plist), "installed": rem_plist.is_file()},
        "foveal_memory": {"active": fm_active, "pid": fm_pid, "plist": str(fm_plist), "installed": fm_plist.is_file()},
        "ears": {"active": ears_active, "pid": ears_pid, "plist": str(ears_plist), "installed": ears_plist.is_file()},
        "telemetry": {"active": tel_active, "pid": tel_pid, "plist": str(tel_plist), "installed": tel_plist.is_file()},
        "message": "\n".join(msg),
    }
