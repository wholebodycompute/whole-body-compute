"""Self-updater and 24-hour cached PyPI version checker for Whole Body Compute."""

from __future__ import annotations

import json
import os
import re
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
MAX_PYPI_JSON_BYTES = 2 * 1024 * 1024  # 2 MiB safety cap
_SAFE_VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)+(?:[A-Za-z0-9._+-]{0,32})?$")


def _is_safe_version_string(ver: Any) -> bool:
    if not isinstance(ver, str):
        return False
    clean = ver.strip()
    if not clean or len(clean) > 64:
        return False
    return bool(_SAFE_VERSION_RE.match(clean))


def _parse_version(ver: str) -> tuple[tuple[int, ...], int]:
    """Parse a semantic/PEP 440 version into `(numeric_release_tuple, stability_rank)`.

    `stability_rank` is `1` for final/post releases and `0` for pre/dev releases (`a`, `b`, `rc`, `dev`),
    preventing `"0.1.6rc1"` from concatenating digits into `(0, 1, 61)` or outranking `"0.1.6"`.
    """
    cleaned = str(ver).strip().lstrip("vV")
    parts: list[int] = []
    is_prerelease = False
    lower = cleaned.lower()
    if any(tag in lower for tag in ("rc", "alpha", "beta", "dev")) or re.search(r"\d[ab]\d", lower):
        is_prerelease = True
    for raw in cleaned.split("."):
        m = re.match(r"^(\d+)", raw)
        if m:
            parts.append(int(m.group(1)))
        else:
            break
    # Normalize trailing zeros on main release segments (e.g. 0.1 == 0.1.0)
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return (tuple(parts) or (0,), 0 if is_prerelease else 1)


def is_newer_version(latest: str, current: str = __version__) -> bool:
    """Return True if `latest` is a valid version string strictly newer than `current`."""
    if not _is_safe_version_string(latest):
        return False
    return _parse_version(latest) > _parse_version(current)


def detect_install_method(executable: str | None = None) -> tuple[str, list[str]]:
    """Detect how `whole-body-compute` was installed and return `(method, upgrade_argv)`."""
    raw_exe = str(executable or sys.executable or "")
    if "\x00" in raw_exe or raw_exe.lstrip().startswith("-"):
        raw_exe = sys.executable
    exe = raw_exe.replace("\\", "/")
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

    safe_py = raw_exe if raw_exe and not raw_exe.startswith("-") and "\x00" not in raw_exe else sys.executable
    return "pip", [safe_py, "-m", "pip", "install", "--upgrade", "whole-body-compute"]


def _get_cache_path() -> Path:
    return get_wbc_data_dir() / "update_check.json"


def _fetch_pypi_latest_version(timeout: float = 1.5) -> str | None:
    """Fetch the latest published version from PyPI, returning None on any error or unsafe string."""
    if not PYPI_JSON_URL.startswith("https://"):
        return None
    req = urllib.request.Request(
        PYPI_JSON_URL,
        headers={"Accept": "application/json", "User-Agent": f"whole-body-compute/{__version__}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read(MAX_PYPI_JSON_BYTES + 1)
        if len(raw) > MAX_PYPI_JSON_BYTES:
            return None
        payload = json.loads(raw.decode("utf-8"))
        latest = str(payload.get("info", {}).get("version", "")).strip()
        return latest if _is_safe_version_string(latest) else None
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
                raw_cached_ver = str(cached.get("latest_version") or "").strip()
                latest = raw_cached_ver if _is_safe_version_string(raw_cached_ver) else __version__
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
    if not latest or not _is_safe_version_string(latest):
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

    proc = subprocess.run(cmd, shell=False, capture_output=True, text=True, timeout=120, check=False)  # noqa: S603
    info["returncode"] = proc.returncode
    info["stdout"] = proc.stdout.strip()
    info["stderr"] = proc.stderr.strip()
    info["status"] = "updated" if proc.returncode == 0 else "error"
    return info
