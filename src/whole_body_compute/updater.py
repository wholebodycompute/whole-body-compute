"""Self-updater and 24-hour cached PyPI version checker for Whole Body Compute."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from whole_body_compute import __version__
from whole_body_compute.procedural_telemetry import get_wbc_data_dir

PYPI_JSON_URL = "https://pypi.org/pypi/whole-body-compute/json"
UPDATE_CHECK_TTL_SEC = 86400  # 24 hours


def _parse_version(ver: str) -> tuple[int, ...]:
    """Parse a dotted semantic version string into an integer tuple."""
    parts: list[int] = []
    for raw in str(ver).strip().lstrip("v").split("."):
        digits = "".join(ch for ch in raw if ch.isdigit())
        if digits:
            parts.append(int(digits))
        else:
            break
    return tuple(parts) or (0,)


def is_newer_version(latest: str, current: str = __version__) -> bool:
    """Return True if `latest` is strictly newer than `current`."""
    return _parse_version(latest) > _parse_version(current)


def detect_install_method(executable: str | None = None) -> tuple[str, list[str]]:
    """Detect how `whole-body-compute` was installed and return `(method, upgrade_argv)`."""
    exe = str(executable or sys.executable or "").replace("\\", "/")
    if "uv/tools/whole-body-compute" in exe:
        uv_bin = shutil.which("uv") or "uv"
        return "uv_tool", [uv_bin, "tool", "upgrade", "whole-body-compute"]
    if "pipx/venvs/whole-body-compute" in exe:
        pipx_bin = shutil.which("pipx") or "pipx"
        return "pipx", [pipx_bin, "upgrade", "whole-body-compute"]
    if "/Cellar/whole-body-compute" in exe or "/opt/homebrew/" in exe and "whole-body-compute" in exe:
        brew_bin = shutil.which("brew") or "brew"
        return "homebrew", [brew_bin, "upgrade", "whole-body-compute"]

    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "pyproject.toml").is_file() and (repo_root / ".git").exists():
        return "editable", ["git", "pull", "--ff-only"]

    return "pip", [sys.executable, "-m", "pip", "install", "--upgrade", "whole-body-compute"]


def _get_cache_path() -> Path:
    return get_wbc_data_dir() / "update_check.json"


def _fetch_pypi_latest_version(timeout: float = 1.5) -> str | None:
    """Fetch the latest published version from PyPI, returning None on any error."""
    req = urllib.request.Request(
        PYPI_JSON_URL,
        headers={"Accept": "application/json", "User-Agent": f"whole-body-compute/{__version__}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        latest = str(payload.get("info", {}).get("version", "")).strip()
        return latest or None
    except Exception:
        return None


def check_for_update(force: bool = False, timeout: float = 1.5) -> dict[str, Any]:
    """Check PyPI for a newer version of `whole-body-compute` (24-hour cached)."""
    cache_path = _get_cache_path()
    now = time.time()

    if not force and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and (now - float(cached.get("checked_at", 0))) < UPDATE_CHECK_TTL_SEC:
                latest = str(cached.get("latest_version") or __version__)
                method, cmd = detect_install_method()
                return {
                    "status": "ok",
                    "cached": True,
                    "current_version": __version__,
                    "latest_version": latest,
                    "update_available": is_newer_version(latest, __version__),
                    "install_method": method,
                    "upgrade_command": " ".join(cmd),
                }
        except Exception:
            pass

    if os.environ.get("PYTEST_CURRENT_TEST") and not force:
        method, cmd = detect_install_method()
        return {
            "status": "skipped",
            "reason": "pytest_isolation",
            "cached": False,
            "current_version": __version__,
            "latest_version": __version__,
            "update_available": False,
            "install_method": method,
            "upgrade_command": " ".join(cmd),
        }

    latest = _fetch_pypi_latest_version(timeout=timeout)
    method, cmd = detect_install_method()
    if not latest:
        return {
            "status": "offline",
            "cached": False,
            "current_version": __version__,
            "latest_version": __version__,
            "update_available": False,
            "install_method": method,
            "upgrade_command": " ".join(cmd),
        }

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"checked_at": round(now, 3), "latest_version": latest}, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass

    return {
        "status": "ok",
        "cached": False,
        "current_version": __version__,
        "latest_version": latest,
        "update_available": is_newer_version(latest, __version__),
        "install_method": method,
        "upgrade_command": " ".join(cmd),
    }


def format_update_notice() -> str | None:
    """Return a concise one-line update notice if a newer PyPI release is available."""
    res = check_for_update(force=False, timeout=1.0)
    if not res.get("update_available"):
        return None
    return (
        f"Update available: whole-body-compute {res['latest_version']} "
        f"(current: {res['current_version']}). Run `body update` to upgrade."
    )


def run_self_update(dry_run: bool = False, check_only: bool = False) -> dict[str, Any]:
    """Check for updates and optionally upgrade `whole-body-compute` in-place."""
    info = check_for_update(force=True, timeout=3.0)
    method, cmd = detect_install_method()
    info["install_method"] = method
    info["upgrade_command"] = " ".join(cmd)

    if check_only or not info.get("update_available"):
        return info

    if dry_run:
        info["dry_run"] = True
        return info

    if method == "editable":
        info["status"] = "editable_checkout"
        info["message"] = "Editable source checkout detected; run `git pull` to update."
        return info

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    info["returncode"] = proc.returncode
    info["stdout"] = proc.stdout.strip()
    info["stderr"] = proc.stderr.strip()
    info["status"] = "updated" if proc.returncode == 0 else "error"
    return info
