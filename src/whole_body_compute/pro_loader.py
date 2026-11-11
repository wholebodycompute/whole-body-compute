"""Dynamic Pro Plugin Bridge & Wheel Installer (`whole-body-compute-pro`).

Bridges Open-Core Pro stubs (`whole_body_compute.*`) to the proprietary
`whole-body-compute-pro` companion wheel via Python entry points
(`whole_body_compute.pro_modules`) when an active `pro`, `team`, or
`enterprise` license is verified by `LicenseManager`.
"""

from __future__ import annotations

import base64
import csv
import functools
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

PRO_PACKAGE_NAME = "whole-body-compute-pro"
PRO_NAMESPACE = "whole_body_compute_pro"
PRO_ENTRY_POINT_GROUP = "whole_body_compute.pro_modules"
DEFAULT_PRO_WHEEL_URL = "https://pro.wholebodycompute.com/v1/wheel"
MAX_WHEEL_BYTES = 50 * 1024 * 1024  # 50 MiB upper bound for companion wheel download

PRO_MODULE_KEYS: tuple[str, ...] = (
    "rem_sleep",
    "dot_memory",
    "radix_cache",
    "agent_loop",
    "cognitive_sentry",
    "cognitive_capsules",
    "cognitive_glymphatic",
    "telemetry_server",
)

OPEN_CORE_MODULE_PATHS: dict[str, str] = {
    "rem_sleep": "whole_body_compute.local.rem_sleep",
    "dot_memory": "whole_body_compute.local.dot_memory",
    "radix_cache": "whole_body_compute.local.radix_cache",
    "agent_loop": "whole_body_compute.local.agent_loop",
    "cognitive_sentry": "whole_body_compute.organs.cognitive_sentry",
    "cognitive_capsules": "whole_body_compute.organs.cognitive_capsules",
    "cognitive_glymphatic": "whole_body_compute.organs.cognitive_glymphatic",
    "telemetry_server": "whole_body_compute.telemetry_server",
}

ACTIVE_PRO_TIERS = ("pro", "team", "enterprise")
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

UPGRADE_MESSAGE: str = (
    "Requires Whole Body Compute Pro ($29/mo). "
    "Run: body license --activate <POLAR_KEY> or visit https://wholebodycompute.com/pricing"
)

_LOADED_PRO_MODULES: dict[str, ModuleType] = {}
_LICENSE_ACTIVE_CACHE: dict[tuple[str, str, str], tuple[int, bool]] = {}

_SAFE_WHEEL_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+-[0-9][A-Za-z0-9._-]*-py3-none-any\.whl$")
_DIST_INFO_DIR_RE = re.compile(r"^whole_body_compute_pro-[0-9][A-Za-z0-9._-]*\.dist-info$")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class ProUpgradeRequiredError(NotImplementedError):
    """Raised when a Pro/Team capability is invoked without an active Pro+ license or plugin."""


def _normalize_dist_name(name: str) -> str:
    """Normalize a Python distribution name per PEP 503."""
    return re.sub(r"[-_.]+", "-", (name or "").strip()).lower()


def validate_http_endpoint_url(url: str, label: str = "Endpoint URL") -> urllib.parse.SplitResult:
    """Validate that an endpoint URL uses `https://` or loopback-only `http://`."""
    clean = (url or "").strip()
    if not clean:
        raise ValueError(f"{label} cannot be empty.")
    parsed = urllib.parse.urlsplit(clean)
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    if scheme not in ("https", "http") or not host:
        raise ValueError(
            f"Unsupported URL scheme in {label} ('{parsed.scheme or 'none'}'). Use https:// (or loopback http://)."
        )
    if scheme == "http" and host not in LOOPBACK_HOSTS:
        raise ValueError(
            f"Insecure non-loopback HTTP URL rejected for {label} ('{host}'). Use https://."
        )
    return parsed


class _SafeWheelRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Prevent `https -> http` downgrades and strip license credentials on cross-origin redirects."""

    def redirect_request(  # type: ignore[override]
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        orig = urllib.parse.urlsplit(req.full_url)
        dest_full = urllib.parse.urljoin(req.full_url, newurl)
        dest = validate_http_endpoint_url(dest_full, label="Redirect URL")
        if orig.scheme.lower() == "https" and dest.scheme.lower() != "https":
            raise urllib.error.URLError("Refusing HTTPS-to-HTTP downgrade redirect during wheel download.")

        new_req = super().redirect_request(req, fp, code, msg, headers, dest_full)
        if new_req is None:
            return None

        if (orig.scheme.lower(), orig.netloc.lower()) != (dest.scheme.lower(), dest.netloc.lower()):
            for sensitive in (
                "Authorization",
                "authorization",
                "X-Polar-License-Key",
                "x-polar-license-key",
                "X-Polar-Activation-Id",
                "x-polar-activation-id",
            ):
                new_req.headers.pop(sensitive, None)
                new_req.unredirected_hdrs.pop(sensitive, None)
        return new_req


@functools.lru_cache(maxsize=8)
def _is_open_core_stubbed_cached(root_str: str) -> bool:
    probe = Path(root_str) / "local" / "dot_memory.py"
    if not probe.is_file():
        return False
    try:
        head = probe.read_text(encoding="utf-8", errors="ignore")[:2048]
    except OSError:
        return False
    return "__wbc_pro_stub__" in head or "Open-Core Pro Feature Stub" in head


def is_open_core_stubbed(package_root: Path | None = None) -> bool:
    """Return True if the installed `whole_body_compute` tree uses Open-Core Pro stubs."""
    root = (Path(package_root) if package_root else Path(__file__).resolve().parent).resolve()
    return _is_open_core_stubbed_cached(str(root))


def _get_user_pro_site_packages() -> Path:
    """Return user-writable fallback directory for PEP 668 / read-only Python installations."""
    try:
        from whole_body_compute.procedural_telemetry import get_wbc_data_dir

        return (get_wbc_data_dir() / "pro_site_packages").expanduser().resolve()
    except (ImportError, OSError, ValueError, RuntimeError):
        return (Path.home() / ".voicefi" / "pro_site_packages").expanduser().resolve()


def _ensure_pro_install_paths() -> None:
    """Ensure `WBC_PRO_INSTALL_TARGET` and user fallback plugin dir are present on `sys.path`."""
    candidates: list[Path] = []
    env_target = os.environ.get("WBC_PRO_INSTALL_TARGET")
    if env_target:
        candidates.append(Path(env_target).expanduser().resolve())
    user_sp = _get_user_pro_site_packages()
    if user_sp.is_dir():
        candidates.append(user_sp)

    added = False
    for cand in candidates:
        cand_str = str(cand)
        if cand.is_dir() and cand_str not in sys.path:
            sys.path.insert(0, cand_str)
            added = True
    if added:
        importlib.invalidate_caches()


def _resolve_package_root_from_stub(stub_file: str | None) -> Path | None:
    """Locate the `whole_body_compute` directory owning a given stub file path."""
    if not stub_file:
        return None
    p = Path(stub_file).resolve()
    for parent in p.parents:
        if parent.name == "whole_body_compute" and (parent / "license.py").is_file():
            return parent
    return None


def _file_mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return -1


def is_pro_license_active(
    license_path: Path | None = None,
    stub_file: str | None = None,
) -> bool:
    """Check whether the active license tier unlocks Pro (`pro`, `team`, `enterprise`)."""
    from whole_body_compute.license import LicenseManager

    founder_env = os.environ.get("WBC_FOUNDER_MODE", "")
    stub_pkg_root = _resolve_package_root_from_stub(stub_file)
    this_pkg_root = Path(__file__).resolve().parent

    mgr = LicenseManager(license_path=license_path)
    resolved_lic = str(mgr.license_path)
    mtime_sig = _file_mtime_ns(mgr.license_path)
    cache_key = (resolved_lic, founder_env, str(stub_pkg_root or ""))

    cached = _LICENSE_ACTIVE_CACHE.get(cache_key)
    if cached is not None and cached[0] == mtime_sig:
        return cached[1]

    # If a stub was loaded directly via spec_from_file_location out of an exported tree
    # while sys.modules['whole_body_compute'] points to the dev monorepo, honor the exported
    # tree's community default unless an explicit license file or founder mode is active.
    if (
        stub_pkg_root is not None
        and stub_pkg_root != this_pkg_root
        and license_path is None
        and not os.environ.get("VOICEFI_LICENSE_PATH")
        and founder_env != "1"
        and mtime_sig == -1
    ):
        _LICENSE_ACTIVE_CACHE[cache_key] = (mtime_sig, False)
        return False

    tier = mgr.get_active_tier() if hasattr(mgr, "get_active_tier") else str(mgr.get_status().get("tier", "community"))
    is_active = tier.lower() in ACTIVE_PRO_TIERS
    _LICENSE_ACTIVE_CACHE[cache_key] = (mtime_sig, is_active)
    return is_active


def discover_pro_plugin_status() -> dict[str, Any]:
    """Inspect installed `whole-body-compute-pro` wheel metadata and entry points."""
    from whole_body_compute import __version__ as wbc_version

    _ensure_pro_install_paths()
    in_tree = not is_open_core_stubbed()
    dist_version: str | None = None
    discovered_eps: list[str] = []

    try:
        dist_version = importlib.metadata.version(PRO_PACKAGE_NAME)
    except (importlib.metadata.PackageNotFoundError, OSError, ValueError):
        dist_version = None

    try:
        eps = importlib.metadata.entry_points(group=PRO_ENTRY_POINT_GROUP)
        valid_eps: set[str] = set()
        for ep in eps:
            if not str(ep.value).startswith(f"{PRO_NAMESPACE}."):
                continue
            ep_dist = getattr(ep, "dist", None)
            if ep_dist is not None and _normalize_dist_name(getattr(ep_dist, "name", "")) != PRO_PACKAGE_NAME:
                continue
            valid_eps.add(ep.name)
        discovered_eps = sorted(valid_eps)
    except (OSError, ValueError, TypeError):
        discovered_eps = []

    ns_spec_found = False
    if dist_version is None and not discovered_eps:
        try:
            spec = importlib.util.find_spec(PRO_NAMESPACE)
            ns_spec_found = spec is not None
            if ns_spec_found:
                pro_init = importlib.import_module(PRO_NAMESPACE)
                dist_version = getattr(pro_init, "__version__", wbc_version)
        except (ImportError, OSError, ValueError, AttributeError):
            ns_spec_found = False

    dist_installed = bool(dist_version is not None or discovered_eps or ns_spec_found)
    version_mismatch = bool(dist_version is not None and dist_version != wbc_version)
    if dist_installed:
        mode = "pro_wheel"
    elif in_tree:
        mode = "in_tree_monorepo"
    else:
        mode = "open_core_stub"

    return {
        "installed": bool(dist_installed or in_tree),
        "distribution_installed": dist_installed,
        "in_tree_monorepo": in_tree,
        "version": dist_version,
        "expected_version": wbc_version,
        "version_mismatch": version_mismatch,
        "entry_points": discovered_eps,
        "mode": mode,
    }


def clear_pro_module_cache(unload_sys_modules: bool = False) -> None:
    """Invalidate cached Pro module references, license lookup cache, and Python import finders."""
    _LOADED_PRO_MODULES.clear()
    _LICENSE_ACTIVE_CACHE.clear()
    _is_open_core_stubbed_cached.cache_clear()
    importlib.invalidate_caches()
    if unload_sys_modules:
        to_remove = [
            mod_name
            for mod_name in list(sys.modules.keys())
            if mod_name == PRO_NAMESPACE or mod_name.startswith(f"{PRO_NAMESPACE}.")
        ]
        for mod_name in to_remove:
            sys.modules.pop(mod_name, None)


def _is_missing_pro_package_error(exc: ModuleNotFoundError, module_key: str) -> bool:
    """Return True if `exc` simply means `whole_body_compute_pro` is not installed (vs a broken 3rd-party dep)."""
    missing_name = getattr(exc, "name", None)
    return missing_name in (None, PRO_NAMESPACE, f"{PRO_NAMESPACE}.{module_key}")


def _import_pro_backend_module(module_key: str) -> ModuleType | None:
    """Import `whole_body_compute_pro.<module_key>` via pinned entry points or direct namespace."""
    _ensure_pro_install_paths()
    cached = _LOADED_PRO_MODULES.get(module_key)
    if cached is not None and getattr(cached, "__name__", "") in sys.modules:
        return cached

    expected_target = f"{PRO_NAMESPACE}.{module_key}"
    for attempt in range(2):
        # 1. Primary path: PEP 621 / importlib.metadata entry points (pinned to whole-body-compute-pro)
        try:
            eps = importlib.metadata.entry_points(group=PRO_ENTRY_POINT_GROUP)
            for ep in eps:
                if ep.name != module_key:
                    continue
                ep_val = str(ep.value).split(":", 1)[0].strip()
                if ep_val != expected_target:
                    continue
                ep_dist = getattr(ep, "dist", None)
                if ep_dist is not None and _normalize_dist_name(getattr(ep_dist, "name", "")) != PRO_PACKAGE_NAME:
                    continue
                loaded = ep.load()
                if isinstance(loaded, ModuleType):
                    _LOADED_PRO_MODULES[module_key] = loaded
                    return loaded
        except ModuleNotFoundError as exc:
            if not _is_missing_pro_package_error(exc, module_key):
                raise
        except (OSError, ValueError, AttributeError):
            pass

        # 2. Secondary path: direct namespace import (`whole_body_compute_pro.<module_key>`)
        try:
            loaded = importlib.import_module(expected_target)
            _LOADED_PRO_MODULES[module_key] = loaded
            return loaded
        except ModuleNotFoundError as exc:
            if not _is_missing_pro_package_error(exc, module_key):
                raise
            if attempt == 0:
                importlib.invalidate_caches()
                continue
        except (OSError, ValueError, RuntimeError):
            break

    return None


def load_pro_module(
    module_key: str,
    license_path: Path | None = None,
    stub_file: str | None = None,
) -> ModuleType | None:
    """
    Dynamically resolve and return the proprietary Pro module for `module_key`
    ONLY if:
      1. Active `LicenseManager` tier is `pro`, `team`, or `enterprise`, AND
      2. `whole-body-compute-pro` is installed (discovered via entry points or namespace).
    Returns `None` when unlicensed or when the Pro wheel is not installed.
    """
    if module_key not in PRO_MODULE_KEYS:
        return None
    if not is_pro_license_active(license_path=license_path, stub_file=stub_file):
        return None
    return _import_pro_backend_module(module_key)


def verify_pro_wheel_archive(wheel_path: Path) -> dict[str, Any]:
    """
    Strictly audit a `whole-body-compute-pro` `.whl` archive BEFORE any installer executes:
      1. Rejects duplicate zip entries, zip symlinks, path traversal, `.pth`, and `sitecustomize.py`.
      2. Enforces strict namespace confinement (`whole_body_compute_pro/` + `whole_body_compute_pro-<ver>.dist-info/`).
      3. Verifies 100% PEP 427 `RECORD` coverage with non-empty `sha256=` digests and exact byte sizes.
    """
    with zipfile.ZipFile(wheel_path, "r") as zf:
        infos = zf.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError(f"Wheel '{wheel_path.name}' contains duplicate zip entries.")

        if not any(n.startswith(f"{PRO_NAMESPACE}/") and not n.endswith("/") for n in names):
            raise ValueError(f"Wheel '{wheel_path.name}' does not contain `{PRO_NAMESPACE}` package.")

        dist_infos: set[str] = set()
        for info in infos:
            member = info.filename
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"Symlink entry forbidden in wheel archive: {member}")
            if "\\" in member or member.startswith("/") or ".." in member.split("/"):
                raise ValueError(f"Unsafe path in wheel archive: {member}")
            if member.lower().endswith(".pth"):
                raise ValueError(f"Executable .pth startup file forbidden in wheel archive: {member}")

            top_segment = member.split("/", 1)[0]
            if top_segment == PRO_NAMESPACE:
                continue
            if _DIST_INFO_DIR_RE.match(top_segment):
                dist_infos.add(top_segment)
                continue
            raise ValueError(
                f"Foreign top-level entry '{member}' forbidden in `{PRO_PACKAGE_NAME}` wheel."
            )

        if len(dist_infos) != 1:
            raise ValueError(
                f"Wheel '{wheel_path.name}' must contain exactly 1 whole_body_compute_pro .dist-info directory."
            )

        dist_info_dir = next(iter(dist_infos))
        record_arc = f"{dist_info_dir}/RECORD"
        if record_arc not in names:
            raise ValueError(f"Wheel '{wheel_path.name}' is missing PEP 427 {record_arc}.")

        expected_hashes: dict[str, tuple[str, int]] = {}
        record_text = zf.read(record_arc).decode("utf-8")
        reader = csv.reader(io.StringIO(record_text))
        for row in reader:
            if not row or not row[0]:
                continue
            arc_path = row[0]
            hash_spec = row[1] if len(row) > 1 else ""
            size_spec = row[2] if len(row) > 2 else ""
            if arc_path == record_arc:
                continue
            if not hash_spec.startswith("sha256=") or not hash_spec.split("=", 1)[1]:
                raise ValueError(
                    f"Wheel RECORD entry '{arc_path}' is missing required sha256= digest."
                )
            if not size_spec.isdigit():
                raise ValueError(
                    f"Wheel RECORD entry '{arc_path}' is missing exact byte length."
                )
            expected_hashes[arc_path] = (hash_spec.split("=", 1)[1], int(size_spec))

        verified_files = 0
        for member in names:
            if member.endswith("/") or member == record_arc:
                continue
            if member not in expected_hashes:
                raise ValueError(
                    f"Unrecorded file '{member}' in wheel '{wheel_path.name}' (missing from RECORD)."
                )
            data = zf.read(member)
            exp_b64, exp_len = expected_hashes[member]
            actual_b64 = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
            if actual_b64 != exp_b64:
                raise ValueError(f"Wheel RECORD SHA-256 mismatch for '{member}'.")
            if len(data) != exp_len:
                raise ValueError(f"Wheel RECORD byte length mismatch for '{member}'.")
            verified_files += 1

    return {"dist_info": dist_info_dir, "verified_files": verified_files}


def _verify_and_extract_wheel_purelib(wheel_path: Path, target_dir: Path) -> None:
    """Validate PEP 427 RECORD SHA-256 digests and atomically extract a pure-Python Pro wheel."""
    verify_pro_wheel_archive(wheel_path)
    target_root = target_dir.resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(wheel_path, "r") as zf:
        names = zf.namelist()
        for member in names:
            dest = (target_root / member).resolve()
            if not dest.is_relative_to(target_root):
                raise ValueError(f"Unsafe path traversal detected in wheel entry: {member}")

        # Remove stale whole_body_compute_pro package & dist-info directories before clean extraction
        stale_pkg = target_root / PRO_NAMESPACE
        if stale_pkg.is_dir():
            shutil.rmtree(stale_pkg)
        for existing in target_root.glob("whole_body_compute_pro-*.dist-info"):
            if existing.is_dir():
                shutil.rmtree(existing)

        for member in names:
            if member.endswith("/"):
                (target_root / member).mkdir(parents=True, exist_ok=True)
                continue
            dest = target_root / member
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(member))


def install_pro_wheel(
    wheel_path: Path | str,
    python_executable: str | None = None,
    target_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Audit and install a `whole-body-compute-pro` `.whl` archive into the active environment."""
    whl = Path(wheel_path).expanduser().resolve()
    if not whl.is_file():
        raise ValueError(f"Pro wheel file not found: {whl}")
    if whl.suffix.lower() != ".whl":
        raise ValueError(f"Expected a .whl archive, got: {whl.name}")
    if not zipfile.is_zipfile(whl):
        raise ValueError(f"Invalid wheel archive (not a valid zip file): {whl.name}")

    # Mandatory pre-install cryptographic & namespace confinement verification on ALL paths
    verify_pro_wheel_archive(whl)

    py_exe = python_executable or sys.executable
    env_target = target_dir or os.environ.get("WBC_PRO_INSTALL_TARGET")
    installer_used = "purelib_pep427"

    if env_target:
        dest_dir = Path(env_target).expanduser().resolve()
        _verify_and_extract_wheel_purelib(whl, dest_dir)
        if str(dest_dir) not in sys.path:
            sys.path.insert(0, str(dest_dir))
        installer_used = "target_pep427"
    else:
        installed_ok = False
        uv_bin = shutil.which("uv")
        if uv_bin:
            proc_uv = subprocess.run(
                [uv_bin, "pip", "install", "--python", py_exe, "--no-deps", "--reinstall", str(whl)],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc_uv.returncode == 0:
                installed_ok = True
                installer_used = "uv_pip"

        if not installed_ok:
            proc_pip = subprocess.run(
                [py_exe, "-m", "pip", "install", "--no-deps", "--force-reinstall", "--no-index", str(whl)],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc_pip.returncode == 0:
                installed_ok = True
                installer_used = "pip"

        if not installed_ok:
            # Fallback for minimal/pip-less venvs or PEP 668 externally-managed environments
            if Path(py_exe).resolve() == Path(sys.executable).resolve():
                purelib = Path(sysconfig.get_path("purelib"))
            else:
                probe = subprocess.run(
                    [py_exe, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                purelib = Path(probe.stdout.strip())
            try:
                _verify_and_extract_wheel_purelib(whl, purelib)
                installer_used = "purelib_pep427"
            except PermissionError:
                user_sp = _get_user_pro_site_packages()
                _verify_and_extract_wheel_purelib(whl, user_sp)
                if str(user_sp) not in sys.path:
                    sys.path.insert(0, str(user_sp))
                installer_used = "user_site_pep427"

    clear_pro_module_cache(unload_sys_modules=True)
    plugin_state = discover_pro_plugin_status()
    return {
        "installed": True,
        "status": "installed",
        "installer": installer_used,
        "wheel_name": whl.name,
        "version": plugin_state.get("version"),
        "entry_points": plugin_state.get("entry_points", []),
    }


def _extract_wheel_filename_from_headers(headers: Any, default_version: str) -> str:
    """Resolve a sanitized PEP 427 wheel filename from HTTP response headers."""
    candidates: list[str] = []
    if headers is not None:
        x_name = headers.get("X-Wheel-Filename")
        if x_name:
            candidates.append(str(x_name).strip())
        c_disp = headers.get("Content-Disposition") or ""
        m = re.search(r'filename="?([^";\r\n]+)"?', str(c_disp))
        if m:
            candidates.append(m.group(1).strip())

    for cand in candidates:
        base = Path(cand).name
        if _SAFE_WHEEL_NAME_RE.match(base):
            return base

    clean_ver = re.sub(r"[^A-Za-z0-9._-]", "", default_version or "0.1.0") or "0.1.0"
    return f"whole_body_compute_pro-{clean_ver}-py3-none-any.whl"


def download_and_install_pro_wheel(
    polar_key: str,
    wheel_url: str | None = None,
    version: str | None = None,
    activation_id: str | None = None,
    python_executable: str | None = None,
    target_dir: Path | str | None = None,
    timeout: float = 20.0,
) -> dict[str, Any]:
    """Exchange a verified Polar license key at the Cloudflare Worker gate and install the Pro wheel."""
    from whole_body_compute import __version__ as wbc_version

    clean_key = (polar_key or "").strip()
    if not clean_key:
        raise ValueError("License key is required to download whole-body-compute-pro.")

    target_ver = (version or wbc_version or "0.1.0").strip()
    raw_url = (wheel_url or os.environ.get("WBC_PRO_WHEEL_URL") or DEFAULT_PRO_WHEEL_URL).strip()
    parsed = validate_http_endpoint_url(raw_url, label="WBC_PRO_WHEEL_URL")

    query_params = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    if "version" not in query_params and not parsed.path.endswith(".whl"):
        query_params["version"] = target_ver
    final_url = urllib.parse.urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urllib.parse.urlencode(query_params),
            parsed.fragment,
        )
    )

    headers: dict[str, str] = {
        "Authorization": f"Bearer {clean_key}",
        "Accept": "application/octet-stream, application/zip, application/json",
        "User-Agent": f"WholeBodyCompute/{target_ver}",
    }
    if activation_id:
        headers["X-Polar-Activation-Id"] = str(activation_id).strip()

    req = urllib.request.Request(final_url, headers=headers, method="GET")
    opener = urllib.request.build_opener(_SafeWheelRedirectHandler())

    try:
        with opener.open(req, timeout=timeout) as resp:
            wheel_bytes = resp.read(MAX_WHEEL_BYTES + 1)
            if len(wheel_bytes) > MAX_WHEEL_BYTES:
                raise RuntimeError(
                    f"Pro wheel response exceeded maximum allowed size ({MAX_WHEEL_BYTES // (1024 * 1024)} MiB)."
                )
            resp_headers = resp.headers
    except urllib.error.HTTPError as err:
        detail = f"HTTP {err.code}"
        try:
            err_payload = json.loads(err.read(65536).decode("utf-8", errors="ignore"))
            if isinstance(err_payload, dict):
                detail = str(err_payload.get("error") or err_payload.get("detail") or detail)
        except (OSError, ValueError, TypeError):
            detail = f"HTTP {err.code}"
        raise RuntimeError(f"Pro wheel gate rejected download ({detail}).") from err
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RuntimeError(f"Failed to download Pro wheel from gate: {exc}") from exc

    raw_expected_sha = (resp_headers.get("X-Wheel-SHA256") if resp_headers else None) or ""
    expected_sha256 = raw_expected_sha.strip().lower()
    if expected_sha256:
        if not _SHA256_HEX_RE.match(expected_sha256):
            raise RuntimeError("Pro wheel gate returned malformed X-Wheel-SHA256 header.")
        actual_sha256 = hashlib.sha256(wheel_bytes).hexdigest().lower()
        if actual_sha256 != expected_sha256:
            raise RuntimeError("Downloaded Pro wheel failed SHA-256 integrity verification.")

    wheel_filename = _extract_wheel_filename_from_headers(resp_headers, default_version=target_ver)
    with tempfile.TemporaryDirectory(prefix="wbc_pro_wheel_") as tmp_dir:
        tmp_whl = Path(tmp_dir) / wheel_filename
        tmp_whl.write_bytes(wheel_bytes)
        res = install_pro_wheel(
            wheel_path=tmp_whl,
            python_executable=python_executable,
            target_dir=target_dir,
        )
        res["source"] = "remote_gate"
        return res
