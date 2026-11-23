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

import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


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


def resolve_ears_daemon_program_args() -> List[str]:
    """
    Resolve `ProgramArguments` for `com.wholebodycompute.ears`.
    On macOS, wraps `body ears --daemon` inside `~/.whole_body_compute/Whole Body Compute.app`
    as the top-level LaunchAgent process so Login Items and TCC permission dialogs
    display "Whole Body Compute" + `AppIcon.icns` instead of `body`/`python3`.
    """
    body_args = resolve_body_program_args(["ears", "--daemon"])
    if sys.platform == "darwin":
        try:
            from whole_body_compute.stt.local_ears import ensure_ears_app_executable

            app_bin = ensure_ears_app_executable()
            if app_bin is not None:
                return [str(app_bin), "--launch-daemon", *body_args]
        except Exception:
            pass
    return body_args


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
) -> Dict[str, Dict[str, Any]]:
    """Construct clean, portable macOS `launchd` plist dictionaries."""
    resolved_logs = resolve_daemon_logs_dir(logs_dir)
    pro_enabled = _is_pro_daemon_unlocked() if include_pro is None else bool(include_pro)
    env_vars = {
        "PATH": _default_daemon_path_env(),
        "PYTHONUNBUFFERED": "1",
        "WBC_AMBIENT_DISCOVERY": os.environ.get("WBC_AMBIENT_DISCOVERY", "1").strip() or "1",
    }

    specs: Dict[str, Dict[str, Any]] = {
        "com.wholebodycompute.foveal": {
            "Label": "com.wholebodycompute.foveal",
            "ProgramArguments": resolve_body_program_args(["eyes", "--daemon"]),
            "EnvironmentVariables": dict(env_vars),
            "RunAtLoad": True,
            "KeepAlive": True,
            "ProcessType": "Interactive",
            "StandardOutPath": str(resolved_logs / "foveal_daemon.log"),
            "StandardErrorPath": str(resolved_logs / "foveal_daemon.log"),
        },
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
            "AssociatedBundleIdentifiers": "com.wholebodycompute.ears.app",
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
        specs = build_launchd_plist_specs(logs_dir=resolved_logs, include_pro=include_pro)
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
