"""
PLG Tier & License Manager (`body license` / `tier` / `pricing` / `upgrade` / `plan`). [direct-edit]
Covered by four (4) pending U.S. & international patent applications (see PATENTS.md and https://wholebodycompute.com/patents).

Manages Whole Body Compute's 4-Tier Product-Led Growth (PLG) Packaging:
- `community` ($0/mo - Open-Core Lead Magnet: unlimited `body audit-spend`, `body begin`, `body doctor`, The Eyes (`body eyes`), The Hands (`body hands`) + 50 daily BBB AST interceptions + 25 daily Local Silicon Offloads + 100% free Em-Dash Auto-Heal + 3 daily Anti-Cheat Blocks)
- `pro` ($29/mo - Whole Body Compute Pro: unlimited Blood-Brain Barrier (BBB) PreToolUse + PostToolOutput, 25-step Glymphatic StateCapsule flush, 3:30 AM REM LoRA distillation, Cephalic pre-warming, VoiceFi 2.5x Speed Studio)
- `team` ($79/seat/mo - Bicameral FinOps & Hive-Mind: everything in Pro + `body team-sync` shared `.agents/team_exemplars.jsonl` Hive-Mind apprenticeship, multi-seat CFO FinOps rollup, shared `.agents/hooks.json` policy enforcement)
- `enterprise` ($199/seat/mo - Zero-Egress Sovereign Fleet: MDM/Jamf fleet deployment, custom VascularTrafficMonitor domain policies, SOC2 3-Stage AST SecGate compliance logs)
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import socket
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

POLAR_API_BASE = os.environ.get("WBC_POLAR_API_BASE", "https://api.polar.sh/v1/customer-portal")
DEFAULT_POLAR_ORG_ID = os.environ.get("WBC_POLAR_ORG_ID", "c0041b8b-5453-4291-a64b-dc9a4546f5ee")

DEFAULT_POLAR_BENEFIT_IDS: Dict[str, str] = {
    "pro": os.environ.get("WBC_POLAR_BENEFIT_PRO", "5273a8c2-3989-4f67-b775-77ed25e9302f"),
    "team": os.environ.get("WBC_POLAR_BENEFIT_TEAM", "d0af5eb3-9cd1-4745-a120-8d266bb443fc"),
    "enterprise": os.environ.get("WBC_POLAR_BENEFIT_ENTERPRISE", "4bbd93bd-9f97-49e8-aca1-87e0938984c8"),
}

POLAR_CHECKOUT_URLS: Dict[str, str] = {
    "pro": os.environ.get(
        "WBC_POLAR_PRO_URL",
        "https://buy.polar.sh/polar_cl_W4re7lSF6revpXa1huLqAMbecKgnMUzT4P5xK21eobp",
    ),
    "team": os.environ.get(
        "WBC_POLAR_TEAM_URL",
        "https://buy.polar.sh/polar_cl_WdAQSkL5Y5eOdot0FtPUjzW52BwUEp4vray3G0yyDao",
    ),
    "enterprise": os.environ.get(
        "WBC_POLAR_ENTERPRISE_URL",
        "https://buy.polar.sh/polar_cl_hasEoqwmUvYwCq0VLwqNO6XB5u4avtiFnUwOG1uvitY",
    ),
}


def _is_expired_iso(expires_at: Any) -> bool:
    """Return True if an ISO-8601 `expires_at` timestamp is in the past."""
    if not expires_at or not isinstance(expires_at, str):
        return False
    raw = expires_at.strip()
    if not raw:
        return False
    try:
        normalized = raw[:-1] + "+00:00" if raw.endswith(("Z", "z")) else raw
        dt = datetime.datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.timestamp() <= time.time()
    except ValueError:
        return False


def _format_polar_error_detail(err_payload: Any, fallback: str) -> str:
    """Format Polar / FastAPI JSON error responses (including 422 validation arrays) cleanly."""
    if not isinstance(err_payload, dict):
        return fallback
    detail = err_payload.get("detail")
    if isinstance(detail, list):
        parts = []
        for item in detail:
            if isinstance(item, dict):
                parts.append(str(item.get("msg") or item.get("detail") or item))
            else:
                parts.append(str(item))
        if parts:
            return "; ".join(parts)
    if isinstance(detail, str) and detail.strip():
        return detail.strip()
    msg = err_payload.get("message") or err_payload.get("error")
    if isinstance(msg, str) and msg.strip():
        return msg.strip()
    return fallback


def _is_seat_limit_error(detail: str) -> bool:
    """Return True if an activation error indicates the Polar license seat limit was reached."""
    lower = (detail or "").lower()
    return any(
        phrase in lower
        for phrase in (
            "activation limit",
            "limit reached",
            "limit exceeded",
            "maximum activations",
            "max activations",
            "already reached",
        )
    )


def _is_isolated_home_env() -> bool:
    """
    Return True when `$HOME`, `WBC_HOME`, or `WBC_DATA_DIR` points to an isolated/temp
    new-user sandbox different from the real host user account's home directory.
    """
    if os.environ.get("WBC_FOUNDER_MODE") == "1":
        return False
    if os.environ.get("WBC_FORCE_COMMUNITY") == "1":
        return True
    if "PYTEST_CURRENT_TEST" in os.environ:
        return False
    for env_key in ("WBC_HOME", "WBC_DATA_DIR"):
        val = os.environ.get(env_key, "").strip()
        if val:
            return True
    home_str = str(Path.home())
    if any(
        marker in home_str.replace("\\", "/")
        for marker in ("/tmp/", "/private/tmp/", "/var/folders/", "/private/var/folders/", "pytest-of-")
    ):
        return True
    try:
        import pwd

        real_home = pwd.getpwuid(os.getuid()).pw_dir
        if real_home and Path(home_str).resolve() != Path(real_home).resolve():
            return True
    except Exception:
        pass
    return False


def _is_open_core_enforced() -> bool:
    """Return True when running in an exported Open-Core installation or explicit Community sandbox without founder override."""
    if os.environ.get("WBC_FOUNDER_MODE") == "1":
        return False
    if os.environ.get("WBC_FORCE_COMMUNITY") == "1":
        return True
    try:
        from whole_body_compute.pro_loader import is_open_core_stubbed

        return is_open_core_stubbed(package_root=Path(__file__).resolve().parent)
    except Exception:
        return False


def _write_secure_license_json(license_path: Path, payload: Dict[str, Any]) -> None:
    """Atomically write `license.json` with `0o600` owner-only permissions."""
    license_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(
        prefix=".license_",
        suffix=".tmp",
        dir=str(license_path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        try:
            os.fchmod(fd, 0o600)
        except (AttributeError, OSError):
            pass
        with os.fdopen(fd, "wb") as f:
            f.write(serialized)
            f.flush()
        os.replace(tmp_path, license_path)
        try:
            os.chmod(license_path, 0o600)
        except OSError:
            pass
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _resolve_tier_from_polar_key_or_benefit(
    key: str,
    benefit_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Resolve active paid tier from Polar benefit ID mappings, metadata, or key prefix."""
    if benefit_id:
        b_id = str(benefit_id).strip()
        ent_b = os.environ.get("WBC_POLAR_BENEFIT_ENTERPRISE", DEFAULT_POLAR_BENEFIT_IDS["enterprise"]).strip()
        team_b = os.environ.get("WBC_POLAR_BENEFIT_TEAM", DEFAULT_POLAR_BENEFIT_IDS["team"]).strip()
        pro_b = os.environ.get("WBC_POLAR_BENEFIT_PRO", DEFAULT_POLAR_BENEFIT_IDS["pro"]).strip()
        if ent_b and b_id == ent_b:
            return "enterprise"
        if team_b and b_id == team_b:
            return "team"
        if pro_b and b_id == pro_b:
            return "pro"

    if metadata and isinstance(metadata, dict):
        m_tier = str(metadata.get("tier", "")).lower().strip()
        if m_tier in ("community", "free"):
            raise ValueError("Polar license metadata specifies non-paid 'community' tier.")
        if m_tier in ("pro", "team", "enterprise"):
            return m_tier

    k_upper = (key or "").strip().upper()
    if k_upper.startswith(("WBC-ENT-", "ENT-", "ENTERPRISE-", "WBC-ENTERPRISE-", "WBC_ENT_", "WBC_ENTERPRISE_")):
        return "enterprise"
    if k_upper.startswith(("WBC-TEAM-", "TEAM-", "WBC_TEAM_")):
        return "team"
    if k_upper.startswith(("WBC-PRO-", "PRO-", "WBC_PRO_")):
        return "pro"

    return "pro"


def _mask_license_key(key: str) -> str:
    """Mask middle characters of a license key while keeping prefix and last 4 chars."""
    clean = (key or "").strip()
    if len(clean) <= 6:
        return "****"

    upper = clean.upper()
    known_prefixes = (
        "WBC-ENTERPRISE-",
        "WBC-TEAM-",
        "WBC-PRO-",
        "WBC-ENT-",
        "WBC-",
        "WBC_ENTERPRISE_",
        "WBC_TEAM_",
        "WBC_PRO_",
        "WBC_ENT_",
        "WBC_",
        "ENTERPRISE-",
        "TEAM-",
        "PRO-",
        "ENT-",
    )
    for pfx in known_prefixes:
        if upper.startswith(pfx):
            head = clean[: len(pfx)]
            tail = clean[-4:]
            return f"{head}****-{tail}"

    if len(clean) > 8:
        return f"{clean[:4]}-****-{clean[-4:]}"
    return f"****-{clean[-4:]}"


def _get_default_license_path() -> Path:
    env_p = os.environ.get("VOICEFI_LICENSE_PATH")
    if env_p:
        return Path(env_p).expanduser()
    try:
        from whole_body_compute.procedural_telemetry import get_wbc_data_dir

        return (get_wbc_data_dir() / "license.json").expanduser()
    except Exception:
        return (Path.home() / ".whole_body_compute" / "license.json").expanduser()


DEFAULT_LICENSE_PATH = _get_default_license_path()

PATENT_NOTICES = (
    "Covered by four (4) pending U.S. & international patent applications "
    "(see PATENTS.md and https://wholebodycompute.com/patents)."
)

TIER_ORDER = ("community", "pro", "team", "enterprise")

TIER_CATALOG: Dict[str, Dict[str, Any]] = {
    "community": {
        "id": "community",
        "name": "Community (Open-Core Lead Magnet)",
        "price_display": "$0 / mo",
        "monthly_usd": 0.0,
        "daily_interception_limit": 50,
        "daily_silicon_offload_limit": 25,
        "daily_anti_cheat_limit": 3,
        "free_skill_export_limit": 1,
        "em_dash_auto_heal_limit": None,
        "features": [
            "Unlimited `body audit-spend` 30-day FinOps receipts (`https://wholebodycompute.org/audit.py` / `https://wholebodycompute.com/audit.py`)",
            "Unlimited `body begin` & `body doctor` 5-organ vitals",
            "Unlimited The Eyes (`body eyes` <5ms Foveal AXUIElement DOM Vision) & The Hands (`body hands` deterministic `ax_act` & <50ms Tactile sensor) (100% Free Forever)",
            "1 Free Teach-by-Showing Portable Skill Export (`body skills export` + unlimited free Dry-Run Previews)",
            "50 daily Blood-Brain Barrier (BBB) AST interceptions",
            "25 daily Local Silicon Offloads (`ReconScout` & `vifi implement` on your Apple Silicon Metal GPU)",
            "Unlimited Blood-Brain Barrier Em-Dash & Typography Auto-Healing (100% Free Forever)",
            "3 daily Anti-Reward-Hacking Blocks (AST Test-Integrity & Protected-File Guard)",
            "Interactive ROI Calculator (`https://wholebodycompute.com/roi` / `https://wholebodycompute.com/pricing`)",
        ],
    },
    "pro": {
        "id": "pro",
        "name": "Pro",
        "price_display": "$29 / mo",
        "monthly_usd": 29.0,
        "daily_interception_limit": None,
        "daily_silicon_offload_limit": None,
        "daily_anti_cheat_limit": None,
        "free_skill_export_limit": None,
        "em_dash_auto_heal_limit": None,
        "features": [
            "Unlimited Hey Pal Demonstration Compiler & Portable Skill Exports (`SKILL.md` + `.zip` for Claude Duet, Claude Code & Antigravity)",
            "Unlimited 82.8% Local Silicon Offloading (`:9379` / `:9380` Metal GPU) + Blood-Brain Barrier (BBB) shielding",
            "Unlimited Anti-Reward-Hacking Guard (AST Test-Integrity Lock & Protected-File Shield)",
            "Unlimited Blood-Brain Barrier (BBB) `PreToolUse` + `PostToolOutput` shielding",
            "Automatic 25-step Glymphatic `StateCapsule` context flush",
            "3:30 AM Nocturnal REM Sleep LoRA distillation (`:9379` / `:9380`)",
            "Cephalic Keystroke & VAD pre-warming + VoiceFi 2.5x Speed Studio",
        ],
    },
    "team": {
        "id": "team",
        "name": "Team",
        "price_display": "$79 / seat / mo",
        "monthly_usd": 79.0,
        "daily_interception_limit": None,
        "daily_silicon_offload_limit": None,
        "daily_anti_cheat_limit": None,
        "free_skill_export_limit": None,
        "em_dash_auto_heal_limit": None,
        "features": [
            "Everything in Pro ($29/mo) unlocked",
            "`body team-sync` shared `.agents/team_exemplars.jsonl` Hive-Mind",
            "Multi-seat CFO FinOps rollup & 3-Point Prompt-Cache Receipt",
            "Shared `.agents/hooks.json` team policy enforcement",
        ],
    },
    "enterprise": {
        "id": "enterprise",
        "name": "Enterprise",
        "price_display": "$199 / seat / mo",
        "monthly_usd": 199.0,
        "daily_interception_limit": None,
        "daily_silicon_offload_limit": None,
        "daily_anti_cheat_limit": None,
        "free_skill_export_limit": None,
        "em_dash_auto_heal_limit": None,
        "features": [
            "Everything in Team ($79/seat/mo) unlocked",
            "MDM / Jamf Apple Silicon fleet deployment & SSO",
            "Custom `VascularTrafficMonitor` domain egress policies",
            "SOC2 / HIPAA 3-Stage AST SecGate compliance audit logs",
        ],
    },
}

FEATURE_MIN_TIER: Dict[str, str] = {
    "audit-spend": "community",
    "begin": "community",
    "doctor": "community",
    "eyes": "community",
    "hands": "community",
    "bbb_interception": "community",
    "silicon_offload": "community",
    "em_dash_auto_heal": "community",
    "anti_cheat_guard": "community",
    "pal_skill_export": "community",
    "unlimited_pal_skill_export": "pro",
    "unlimited_bbb": "pro",
    "unlimited_silicon_offload": "pro",
    "unlimited_anti_cheat_guard": "pro",
    "glymphatic_flush": "pro",
    "rem_sleep_lora": "pro",
    "cephalic_prewarm": "pro",
    "voicefi_speed_studio": "pro",
    "team_sync": "team",
    "cfo_multi_seat_rollup": "team",
    "shared_hooks_policy": "team",
    "mdm_fleet": "enterprise",
    "custom_vascular_policy": "enterprise",
    "soc2_compliance_logs": "enterprise",
}


class LicenseManager:
    """Manages PLG tier state, non-destructive feature gates, and ROI payback cards."""

    def __init__(self, license_path: Optional[Path] = None) -> None:
        self.license_path = Path(license_path).expanduser() if license_path else _get_default_license_path()

    def _sanitize_loaded_record(self, loaded: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce expiration timestamps and Open-Core Polar provider integrity on loaded records."""
        sanitized = dict(loaded)
        if _is_expired_iso(sanitized.get("expires_at")):
            sanitized["tier"] = "community"
            sanitized["mode"] = "expired"
            return sanitized

        claimed_tier = str(sanitized.get("tier", "community")).lower()
        if claimed_tier != "community" and _is_open_core_enforced():
            has_polar_proof = (
                sanitized.get("provider") == "polar.sh"
                and bool(sanitized.get("license_key") or sanitized.get("receipt_hash"))
            )
            if not has_polar_proof:
                try:
                    from whole_body_compute.procedural_telemetry import track_wbc_event

                    track_wbc_event(
                        "wbc_quota_bypass_detected",
                        {
                            "bypass_type": "unverified_license_json",
                            "claimed_tier": claimed_tier,
                            "provider": str(sanitized.get("provider") or "none"),
                        },
                    )
                except Exception:
                    pass
                sanitized["tier"] = "community"
                sanitized["mode"] = "community_default"
        return sanitized

    def get_active_tier(self) -> str:
        """Fast `<0.05ms` active tier resolution without scanning `.dist-info` plugin metadata."""
        founder_mode = os.environ.get("WBC_FOUNDER_MODE") == "1"
        if founder_mode and "PYTEST_CURRENT_TEST" not in os.environ:
            try:
                from whole_body_compute.pro_loader import is_open_core_stubbed

                if is_open_core_stubbed(package_root=Path(__file__).resolve().parent):
                    from whole_body_compute.procedural_telemetry import track_wbc_event

                    track_wbc_event(
                        "wbc_quota_bypass_detected",
                        {
                            "bypass_type": "founder_mode_env_in_open_core",
                        },
                    )
            except Exception:
                pass
        is_comm_default = (not founder_mode) and (_is_open_core_enforced() or _is_isolated_home_env())
        default_tier = "community" if is_comm_default else "enterprise"
        if self.license_path.is_file():
            try:
                loaded = json.loads(self.license_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and loaded.get("tier") in TIER_CATALOG:
                    checked = self._sanitize_loaded_record(loaded)
                    t_id = str(checked.get("tier", default_tier)).lower()
                    return t_id if t_id in TIER_CATALOG else default_tier
            except Exception:
                pass
        return default_tier

    def get_status(self) -> Dict[str, Any]:
        """Return current license configuration (defaulting to `community` tier in open-core export)."""
        founder_mode = os.environ.get("WBC_FOUNDER_MODE") == "1"
        default_tier = "enterprise" if founder_mode else "community"
        data: Dict[str, Any] = {
            "tier": "enterprise" if founder_mode else "community",
            "mode": "founder_enterprise" if founder_mode else "community_default",
            "updated_at": int(time.time()),
        }
        if self.license_path.is_file():
            try:
                loaded = json.loads(self.license_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and loaded.get("tier") in TIER_CATALOG:
                    data.update(self._sanitize_loaded_record(loaded))
            except Exception:
                pass

        tier_id = str(data.get("tier", default_tier)).lower()
        if tier_id not in TIER_CATALOG:
            tier_id = default_tier
        cat = TIER_CATALOG[tier_id]
        try:
            from whole_body_compute.pro_loader import discover_pro_plugin_status

            pro_plugin = discover_pro_plugin_status()
        except Exception:
            pro_plugin = {"installed": False, "mode": "unknown"}

        return {
            "tier": tier_id,
            "tier_name": cat["name"],
            "price_display": cat["price_display"],
            "monthly_usd": cat["monthly_usd"],
            "daily_interception_limit": cat["daily_interception_limit"],
            "daily_silicon_offload_limit": cat.get("daily_silicon_offload_limit"),
            "daily_anti_cheat_limit": cat.get("daily_anti_cheat_limit"),
            "em_dash_auto_heal_limit": cat.get("em_dash_auto_heal_limit"),
            "features": cat["features"],
            "license_path": str(self.license_path),
            "mode": data.get("mode", "active"),
            "provider": data.get("provider", "local"),
            "license_key_masked": data.get("license_key_masked"),
            "activation_id": data.get("activation_id"),
            "expires_at": data.get("expires_at"),
            "pro_plugin": pro_plugin,
            "checkout_urls": POLAR_CHECKOUT_URLS,
        }

    def activate_polar_key(
        self,
        key: str,
        organization_id: Optional[str] = None,
        label: Optional[str] = None,
        offline: bool = False,
        wheel_path: Optional[Path | str] = None,
        wheel_url: Optional[str] = None,
        install_wheel: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Activate and verify a Polar.sh license key (online or offline) and install Pro plugin wheel."""
        clean_key = (key or "").strip()
        if not clean_key:
            raise ValueError("License key cannot be empty.")
        if len(clean_key) > 256:
            raise ValueError("License key exceeds maximum allowed length (256 characters).")

        explicit_wheel_path = wheel_path or os.environ.get("WBC_PRO_WHEEL_PATH")
        explicit_wheel_url = wheel_url or os.environ.get("WBC_PRO_WHEEL_URL")

        # Validate explicit local wheel BEFORE mutating license.json
        if install_wheel is not False and explicit_wheel_path:
            from whole_body_compute.pro_loader import verify_pro_wheel_archive

            whl_candidate = Path(explicit_wheel_path).expanduser().resolve()
            if not whl_candidate.is_file():
                raise ValueError(f"Pro wheel file not found: {whl_candidate}")
            verify_pro_wheel_archive(whl_candidate)

        if organization_id is not None:
            org_id = organization_id.strip()
        elif "WBC_POLAR_ORG_ID" in os.environ:
            org_id = os.environ["WBC_POLAR_ORG_ID"].strip()
        else:
            org_id = DEFAULT_POLAR_ORG_ID.strip()

        if not offline and not org_id:
            raise ValueError(
                "Missing Polar organization_id for online activation (set WBC_POLAR_ORG_ID or pass --offline)."
            )

        activation_id: Optional[str] = None
        benefit_id: Optional[str] = None
        expires_at: Optional[str] = None
        metadata: Dict[str, Any] = {}
        status_val = "granted"

        if not offline:
            from whole_body_compute.pro_loader import validate_http_endpoint_url

            api_base = os.environ.get("WBC_POLAR_API_BASE", POLAR_API_BASE).rstrip("/")
            validate_http_endpoint_url(api_base, label="WBC_POLAR_API_BASE")
            activate_url = f"{api_base}/license-keys/activate"
            validate_url = f"{api_base}/license-keys/validate"
            client_label = (label or socket.gethostname() or "wbc-client").strip()

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "WholeBodyCompute/1.0",
            }

            resp_data: Optional[Dict[str, Any]] = None

            # Idempotent re-activation check: if this machine already holds an activation_id
            # for the exact same Polar key, validate that existing seat first instead of burning a new seat.
            prior_act_id: Optional[str] = None
            if self.license_path.is_file():
                try:
                    existing_doc = json.loads(self.license_path.read_text(encoding="utf-8"))
                    if (
                        isinstance(existing_doc, dict)
                        and existing_doc.get("license_key") == clean_key
                        and existing_doc.get("activation_id")
                    ):
                        prior_act_id = str(existing_doc["activation_id"]).strip()
                except Exception:
                    prior_act_id = None

            if prior_act_id:
                reuse_body = json.dumps(
                    {
                        "key": clean_key,
                        "organization_id": org_id,
                        "activation_id": prior_act_id,
                    }
                ).encode("utf-8")
                req_reuse = urllib.request.Request(validate_url, data=reuse_body, headers=headers, method="POST")
                try:
                    with urllib.request.urlopen(req_reuse, timeout=10.0) as reuse_resp:
                        resp_data = json.loads(reuse_resp.read().decode("utf-8"))
                        activation_id = prior_act_id
                except Exception:
                    resp_data = None

            if resp_data is None:
                act_body = json.dumps(
                    {
                        "key": clean_key,
                        "organization_id": org_id,
                        "label": client_label,
                    }
                ).encode("utf-8")
                req_act = urllib.request.Request(activate_url, data=act_body, headers=headers, method="POST")

                try:
                    with urllib.request.urlopen(req_act, timeout=10.0) as resp:
                        resp_data = json.loads(resp.read().decode("utf-8"))
                except urllib.error.HTTPError as err:
                    raw_err_text = ""
                    try:
                        raw_err_text = err.read().decode("utf-8", errors="ignore")
                        err_json = json.loads(raw_err_text)
                        act_detail = _format_polar_error_detail(err_json, str(err))
                    except Exception:
                        act_detail = raw_err_text.strip() or str(err)

                    if err.code in (400, 422) and not _is_seat_limit_error(act_detail):
                        val_body = json.dumps(
                            {
                                "key": clean_key,
                                "organization_id": org_id,
                            }
                        ).encode("utf-8")
                        req_val = urllib.request.Request(validate_url, data=val_body, headers=headers, method="POST")
                        try:
                            with urllib.request.urlopen(req_val, timeout=10.0) as val_resp:
                                resp_data = json.loads(val_resp.read().decode("utf-8"))
                        except urllib.error.HTTPError as val_err:
                            try:
                                val_json = json.loads(val_err.read().decode("utf-8", errors="ignore"))
                                detail = _format_polar_error_detail(val_json, str(val_err))
                            except Exception:
                                detail = str(val_err)
                            raise ValueError(f"Polar license validation failed: {detail}") from val_err
                        except Exception as ex:
                            raise ValueError(f"Polar license validation network error: {ex}") from ex
                    else:
                        raise ValueError(f"Polar license activation failed: {act_detail}") from err
                except Exception as ex:
                    raise ValueError(f"Polar license activation network error: {ex}") from ex

            if isinstance(resp_data, dict):
                if "license_key" in resp_data and isinstance(resp_data["license_key"], dict):
                    activation_id = resp_data.get("id") or activation_id
                    lk = resp_data["license_key"]
                    status_val = str(lk.get("status", "granted")).lower()
                    benefit_id = lk.get("benefit_id")
                    expires_at = lk.get("expires_at")
                    metadata = lk.get("metadata") or {}
                else:
                    status_val = str(resp_data.get("status", "granted")).lower()
                    benefit_id = resp_data.get("benefit_id")
                    expires_at = resp_data.get("expires_at")
                    metadata = resp_data.get("metadata") or {}
                    activation_id = resp_data.get("id") or resp_data.get("activation_id") or activation_id

            if status_val in ("revoked", "disabled"):
                raise ValueError(f"Polar license key is {status_val}.")
            if status_val != "granted":
                raise ValueError(f"Polar license key is not active (status: {status_val}).")
            if _is_expired_iso(expires_at):
                raise ValueError("Polar license key has expired.")
        else:
            masked_err_key = _mask_license_key(clean_key)
            if len(clean_key) < 8:
                raise ValueError(f"Invalid license key format: key '{masked_err_key}' is too short (min 8 characters).")
            k_upper = clean_key.upper()
            valid_prefixes = (
                "WBC-", "WBC_", "PRO-", "TEAM-", "ENT-", "ENTERPRISE-", "POLAR_"
            )
            is_valid_format = (
                any(k_upper.startswith(pfx) for pfx in valid_prefixes)
                or ("-" in clean_key and len(clean_key) >= 12)
            )
            if not is_valid_format:
                raise ValueError(
                    f"Invalid license key format: '{masked_err_key}'. Expected format like WBC-PRO-XXXX-YYYY."
                )

        tier = _resolve_tier_from_polar_key_or_benefit(clean_key, benefit_id=benefit_id, metadata=metadata)
        receipt_hash = hashlib.sha256(f"{tier}:{clean_key}:{activation_id or 'local'}".encode("utf-8")).hexdigest()[:16]
        masked_key = _mask_license_key(clean_key)
        mode = "polar_verified" if not offline else "polar_offline"

        payload = {
            "tier": tier,
            "mode": mode,
            "provider": "polar.sh",
            "license_key_masked": masked_key,
            "license_key": clean_key,
            "activation_id": activation_id,
            "benefit_id": benefit_id,
            "organization_id": org_id or None,
            "expires_at": expires_at,
            "receipt_hash": receipt_hash,
            "updated_at": int(time.time()),
        }
        _write_secure_license_json(self.license_path, payload)

        pro_install_receipt: Optional[Dict[str, Any]] = None
        from whole_body_compute.pro_loader import (
            clear_pro_module_cache,
            discover_pro_plugin_status,
            download_and_install_pro_wheel,
            install_pro_wheel,
            is_open_core_stubbed,
        )

        clear_pro_module_cache(unload_sys_modules=True)

        if install_wheel is not False:
            if explicit_wheel_path:
                pro_install_receipt = install_pro_wheel(explicit_wheel_path)
            elif explicit_wheel_url and not offline:
                pro_install_receipt = download_and_install_pro_wheel(
                    polar_key=clean_key,
                    wheel_url=explicit_wheel_url,
                    activation_id=activation_id,
                )
            elif not offline:
                plug_status = discover_pro_plugin_status()
                needs_wheel = bool(
                    not plug_status.get("distribution_installed")
                    or plug_status.get("version_mismatch")
                )
                if install_wheel is True or (is_open_core_stubbed() and needs_wheel):
                    pro_install_receipt = download_and_install_pro_wheel(
                        polar_key=clean_key,
                        activation_id=activation_id,
                    )

        status_out = self.get_status()
        if pro_install_receipt is not None:
            status_out["pro_plugin_install"] = pro_install_receipt
        return status_out

    def reinstall_pro_wheel(
        self,
        wheel_path: Optional[Path | str] = None,
        wheel_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Install or re-download `whole-body-compute-pro` into the current Python environment using stored Polar credentials."""
        from whole_body_compute.pro_loader import download_and_install_pro_wheel, install_pro_wheel

        explicit_path = wheel_path or os.environ.get("WBC_PRO_WHEEL_PATH")
        if explicit_path:
            receipt = install_pro_wheel(explicit_path)
            st = self.get_status()
            st["pro_plugin_install"] = receipt
            return st

        if not self.license_path.is_file():
            raise ValueError("No active license file found. Run: body license --activate <POLAR_KEY>")

        doc = json.loads(self.license_path.read_text(encoding="utf-8"))
        stored_key = str((doc or {}).get("license_key") or "").strip()
        stored_act_id = (doc or {}).get("activation_id")
        if not stored_key:
            raise ValueError("No stored Polar license key found. Run: body license --activate <POLAR_KEY>")

        receipt = download_and_install_pro_wheel(
            polar_key=stored_key,
            wheel_url=wheel_url,
            activation_id=str(stored_act_id) if stored_act_id else None,
        )
        st = self.get_status()
        st["pro_plugin_install"] = receipt
        return st

    def deactivate_license(self) -> Dict[str, Any]:
        """Deactivate license on this machine and revert to Community tier."""
        existing: Dict[str, Any] = {}
        if self.license_path.is_file():
            try:
                existing = json.loads(self.license_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        act_id = existing.get("activation_id")
        org_id = existing.get("organization_id")
        stored_key = existing.get("license_key")

        if act_id and org_id and stored_key:
            try:
                from whole_body_compute.pro_loader import validate_http_endpoint_url

                api_base = os.environ.get("WBC_POLAR_API_BASE", POLAR_API_BASE).rstrip("/")
                validate_http_endpoint_url(api_base, label="WBC_POLAR_API_BASE")
                deact_url = f"{api_base}/license-keys/deactivate"
                payload_bytes = json.dumps({
                    "key": stored_key,
                    "organization_id": org_id,
                    "activation_id": act_id,
                }).encode("utf-8")
                headers = {"Content-Type": "application/json", "Accept": "application/json"}
                req = urllib.request.Request(deact_url, data=payload_bytes, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=5.0):
                    pass
            except Exception:
                pass

        payload = {
            "tier": "community",
            "mode": "deactivated",
            "provider": "local",
            "updated_at": int(time.time()),
        }
        _write_secure_license_json(self.license_path, payload)
        try:
            from whole_body_compute.pro_loader import clear_pro_module_cache

            clear_pro_module_cache(unload_sys_modules=True)
        except Exception:
            pass
        return self.get_status()

    def set_tier(self, tier: str) -> Dict[str, Any]:
        """Switch active PLG tier (`community`, or dev/founder simulation tiers)."""
        clean = (tier or "").strip().lower()
        if clean not in TIER_CATALOG:
            raise ValueError(f"Unsupported tier '{tier}'. Choose from: {', '.join(TIER_ORDER)}")
        if clean != "community" and _is_open_core_enforced():
            raise ValueError(
                f"Paid tier '{clean}' requires a verified Polar.sh license key in Open-Core. "
                "Run: body license --activate <POLAR_KEY>"
            )
        payload = {
            "tier": clean,
            "mode": "founder_enterprise" if clean == "enterprise" else f"plg_{clean}",
            "updated_at": int(time.time()),
        }
        _write_secure_license_json(self.license_path, payload)
        try:
            from whole_body_compute.pro_loader import clear_pro_module_cache

            clear_pro_module_cache(unload_sys_modules=True)
        except Exception:
            pass
        return self.get_status()

    def is_feature_allowed(self, feature_name: str, **kwargs: Any) -> bool:
        """Convenience boolean check wrapping `check_feature_gate`."""
        return bool(self.check_feature_gate(feature_name, **kwargs).get("allowed"))

    def check_feature_gate(
        self,
        feature_name: str,
        daily_interceptions: int = 0,
        unbounded_lines: int = 500,
        reingestion_turns: int = 25,
        daily_silicon_offloads: int = 0,
        offloaded_tokens_today: int = 0,
        daily_anti_cheat_blocks: int = 0,
        em_dashes_healed_today: int = 0,
        skill_exports_used: int = 0,
    ) -> Dict[str, Any]:
        """
        Check whether `feature_name` is permitted under the active tier. [direct-edit]
        In `community` tier:
        - `bbb_interception` is allowed up to 50 daily interceptions
        - `silicon_offload` is allowed up to 25 daily offloads
        - `anti_cheat_guard` is allowed up to 3 daily blocks
        - `pal_skill_export` is allowed up to 1 free full portable skill export (then Dry-Run Preview)
        - `em_dash_auto_heal` is 100% Free Forever across all tiers
        Once limits are exceeded, the Blood-Brain Barrier (BBB) operates in non-destructive Pass-Through
        mode and emits a live receipt/alert nudge.
        """
        curr_tier = self.get_active_tier()
        curr_idx = TIER_ORDER.index(curr_tier)

        req_tier = FEATURE_MIN_TIER.get(feature_name, "pro")
        req_idx = TIER_ORDER.index(req_tier)

        allowed = curr_idx >= req_idx
        pass_through_on_limit = False
        nudge = ""

        # Estimate cumulative re-ingested tokens & dollar cost for a single un-sliced read
        est_tokens_per_read = max(100, int(unbounded_lines * 8.5) - 350) * max(1, reingestion_turns)
        est_waste_usd = round(est_tokens_per_read * 0.000003, 2)
        daily_saved_usd = round(50 * max(0.15, est_waste_usd), 2)

        # Estimate daily local silicon offload tokens and dollar savings
        est_offloaded_tokens_today = max(offloaded_tokens_today, max(1, daily_silicon_offloads) * 11550)

        if feature_name == "em_dash_auto_heal":
            allowed = True
            pass_through_on_limit = False

        if curr_tier == "community" and feature_name in ("pal_skill_export", "unlimited_pal_skill_export"):
            if skill_exports_used < 1 and feature_name == "pal_skill_export":
                allowed = True
                pass_through_on_limit = False
            else:
                allowed = False
                pass_through_on_limit = True
                req_tier = "pro"
                nudge = (
                    f"🎓 [WBC Community Limit: {max(1, skill_exports_used)}/1 Free Portable Skill Exports Used] "
                    "Dry-Run Preview active: unlock Unlimited Hey Pal Demonstration Compiling & `.skill.zip` Exports "
                    "for Claude Duet, Claude Code & Antigravity with Whole Body Compute Pro ($29/mo): "
                    "`body license --activate <POLAR_KEY>`"
                )

        if curr_tier == "community" and feature_name in ("bbb_interception", "unlimited_bbb"):
            if daily_interceptions >= 50 or feature_name == "unlimited_bbb":
                allowed = False
                pass_through_on_limit = True
                req_tier = "pro"
                nudge = (
                    f"⚠️ [WBC Community Limit: {max(50, daily_interceptions)}/50 BBB Shields Used Today - "
                    f"${daily_saved_usd:.2f} Saved] Pass-through active: this {unbounded_lines}-line read adds "
                    f"~{est_tokens_per_read:,} re-ingested tokens (${est_waste_usd:.2f}) to your cloud bill. "
                    "Unlock unlimited BBB + 25-step Glymphatic Flush for $29/mo: `body license --activate <POLAR_KEY>`"
                )

        if curr_tier == "community" and feature_name in ("silicon_offload", "unlimited_silicon_offload"):
            if daily_silicon_offloads < 25 and feature_name == "silicon_offload":
                allowed = True
                pass_through_on_limit = False
            else:
                allowed = False
                pass_through_on_limit = True
                req_tier = "pro"
                tokens_handled = max(offloaded_tokens_today, max(25, daily_silicon_offloads) * 11550)
                saved_usd = round(tokens_handled * 0.000015, 2)
                nudge = (
                    f"⚡ [WBC Community Limit: {max(25, daily_silicon_offloads)}/25 Local Silicon Offloads Used Today - "
                    f"{tokens_handled:,} Cloud Tokens (${saved_usd:.2f}) Handled 100% on Your Mac GPU] Passing through to "
                    "cloud frontier model. Unlock Unlimited 82.8% Local Silicon Offloading + 3:30 AM REM LoRA Distillation "
                    "for $29/mo: `body license --activate <POLAR_KEY>`"
                )

        if curr_tier == "community" and feature_name in ("anti_cheat_guard", "unlimited_anti_cheat_guard"):
            if daily_anti_cheat_blocks < 3 and feature_name == "anti_cheat_guard":
                allowed = True
                pass_through_on_limit = False
            else:
                allowed = False
                pass_through_on_limit = True
                req_tier = "pro"
                nudge = (
                    f"⚠️ [WBC Community Limit: {max(3, daily_anti_cheat_blocks)}/3 Anti-Reward-Hacking Blocks Used Today] "
                    "Pass-through alert: agent attempted a test-integrity or protected-file modification. "
                    "Unlock Unlimited AST Test-Integrity & Protected-File Lock for $29/mo: `body license --activate <POLAR_KEY>`"
                )

        if not allowed and not nudge:
            req_cat = TIER_CATALOG[req_tier]
            nudge = (
                f"Feature `{feature_name}` requires {req_cat['name']} ({req_cat['price_display']}). "
                "Activate with `body license --activate <POLAR_KEY>`."
            )

        # Track quota milestones and limit hits in PostHog
        if curr_tier == "community" and (
            pass_through_on_limit
            or not allowed
            or daily_interceptions in (10, 25, 40, 50)
            or daily_silicon_offloads in (10, 20, 25)
            or daily_anti_cheat_blocks in (1, 2, 3)
        ):
            try:
                from whole_body_compute.procedural_telemetry import track_wbc_event

                track_wbc_event(
                    "wbc_quota_event",
                    {
                        "feature": feature_name,
                        "allowed": bool(allowed),
                        "limit_reached": bool(pass_through_on_limit or not allowed),
                        "current_tier": curr_tier,
                        "required_tier": req_tier,
                        "daily_interceptions": int(daily_interceptions),
                        "daily_silicon_offloads": int(daily_silicon_offloads),
                        "daily_anti_cheat_blocks": int(daily_anti_cheat_blocks),
                        "skill_exports_used": int(skill_exports_used),
                    },
                    detached_hook=True,
                )
            except Exception:
                pass

        return {
            "feature": feature_name,
            "allowed": allowed,
            "pass_through_on_limit": pass_through_on_limit,
            "current_tier": curr_tier,
            "required_tier": req_tier,
            "daily_interceptions": int(daily_interceptions),
            "daily_silicon_offloads": int(daily_silicon_offloads),
            "daily_anti_cheat_blocks": int(daily_anti_cheat_blocks),
            "em_dashes_healed_today": int(em_dashes_healed_today),
            "estimated_offloaded_tokens_today": est_offloaded_tokens_today,
            "estimated_read_reingestion_tokens": est_tokens_per_read,
            "estimated_read_waste_usd": est_waste_usd,
            "upgrade_nudge": nudge,
            "roi_payback_summary": "Pro ($29/mo) pays for itself in <2 hours of frontier coding (473x-782x ROI • SWE-bench_Verified k=3: 1.48x context, 1.52x net context, 1.34x billed runway, -18.8% turns @ 100% Pass^3).",
        }


def format_license_and_pricing_card(
    status: Dict[str, Any],
    finops_summary: Optional[Dict[str, Any]] = None,
) -> str:
    """Render a box-drawn terminal comparison table of the 4 tiers + personalized ROI payback."""
    curr = status.get("tier", "enterprise")
    sr = (finops_summary or {}).get("savings_range", {})
    cons_mo = float(sr.get("conservative_cached", {}).get("monthly_savings_per_seat_usd", 0.0) or 0.0)
    real_mo = float(sr.get("realistic_blended", {}).get("monthly_savings_per_seat_usd", 0.0) or 0.0)
    pro_roi_lo = round(cons_mo / 29.0, 1) if 0 < cons_mo / 29.0 < 10 else round(cons_mo / 29.0)
    pro_roi_hi = round(real_mo / 29.0, 1) if 0 < real_mo / 29.0 < 10 else round(real_mo / 29.0)
    team_roi_lo = round(cons_mo / 79.0, 1) if 0 < cons_mo / 79.0 < 10 else round(cons_mo / 79.0)
    team_roi_hi = round(real_mo / 79.0, 1) if 0 < real_mo / 79.0 < 10 else round(real_mo / 79.0)

    b_lim = "Unlimited" if status.get("daily_interception_limit") is None else f"{status.get('daily_interception_limit')}/day"
    o_lim = "Unlimited" if status.get("daily_silicon_offload_limit") is None else f"{status.get('daily_silicon_offload_limit')}/day"
    offload_label = "Local Silicon Offloads" if o_lim != "Unlimited" else "Silicon Offloads"

    lines = [
        "",
        "╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗",
        "║               WHOLE BODY COMPUTE - PLG TIER PACKAGING & LICENSE STATUS (BBB & FINOPS)                ║",
        "╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝",
        f"  • Active License Tier: {status.get('tier_name', 'Enterprise')} [{curr.upper()}] ({status.get('price_display', '$199/seat/mo')})",
        f"  • Quotas & BBB Gates:  {b_lim} BBB Shields • {o_lim} {offload_label} • Eyes & Hands: 100% Free Forever",
        f"  • License File:        {status.get('license_path', '~/.whole_body_compute/license.json')}",
    ]
    if status.get("license_key_masked"):
        lines.append(
            f"  • Polar.sh License:    {status.get('license_key_masked')} ({status.get('provider', 'polar.sh')}:{status.get('mode', 'active')})"
        )
    pro_plug = status.get("pro_plugin")
    if isinstance(pro_plug, dict):
        p_mode = pro_plug.get("mode", "unknown")
        if p_mode == "pro_wheel":
            p_ver = pro_plug.get("version") or "installed"
            ep_cnt = len(pro_plug.get("entry_points") or [])
            mismatch_note = " [update available: run `body license --reinstall-wheel`]" if pro_plug.get("version_mismatch") else ""
            lines.append(f"  • Pro Plugin Wheel:    whole-body-compute-pro v{p_ver} ({ep_cnt} entry points active){mismatch_note}")
        elif p_mode == "open_core_stub":
            lines.append("  • Pro Plugin Wheel:    Not installed (run `body license --activate <POLAR_KEY>`)")
    if real_mo > 0.0:
        payback_line = (
            f"  • Live Seat Payback:   ${cons_mo:,.2f} - ${real_mo:,.2f}/mo saved -> "
            f"{pro_roi_lo:,}x-{pro_roi_hi:,}x Pro ROI • {team_roi_lo:,}x-{team_roi_hi:,}x Team ROI"
        )
    else:
        payback_line = (
            "  • Live Seat Payback:   $0.00/mo saved (0 local sessions audited yet - run `body audit-spend`)"
        )
    lines.extend([
        payback_line,
        "  • SWE-bench_Verified:  1.48x context (-32.4%) • 1.52x net context (-34.3%) • 1.34x billed runway (-25.6%) • -18.8% turns @ 100% Pass^3 (k=3, 60 trials)",
        "",
        "  ┌──────────────────────┬───────────────────┬─────────────────────────────────────────────────────────┐",
        "  │ Tier                 │ Price             │ Core Capabilities & FinOps Guarantees                   │",
        "  ├──────────────────────┼───────────────────┼─────────────────────────────────────────────────────────┤",
    ])

    rows = [
        ("community", "Community (Free)", "$0 / mo", "Unlimited audit-spend, Eyes & Hands, 50 BBB, 25/day Offload"),
        ("pro", "Pro (Builder)", "$29 / mo", "Unlimited BBB + Offload + Anti-Cheat Lock, REM LoRA"),
        ("team", "Team (Hive-Mind)", "$79 / seat / mo", "`body team-sync` shared exemplars, CFO multi-seat FinOps"),
        ("enterprise", "Enterprise (Fleet)", "$199 / seat / mo", "MDM/Jamf fleet, custom Vascular egress, SOC2 AST SecGate"),
    ]
    for tid, tlabel, tprice, tdesc in rows:
        marker = "★ " if tid == curr else "  "
        lines.append(f"  │ {marker}{tlabel:<18} │ {tprice:<17} │ {tdesc:<55} │")

    lines.extend(
        [
            "  └──────────────────────┴───────────────────┴─────────────────────────────────────────────────────────┘",
            "  💡 Eyes & Hands: 100% Free Forever | Revert to Free: `body license --deactivate`",
            "  🔑 Polar License: `body license --activate <POLAR_KEY>` | Reinstall Plugin: `body license --reinstall-wheel`",
            "  🌐 Web & ROI Calculator: https://wholebodycompute.com  •  https://wholebodycompute.com/roi  •  https://wholebodycompute.org/audit.py",
            f"  📜 Patent Notice: {PATENT_NOTICES}",
            "",
        ]
    )
    return "\n".join(lines)


class FeatureGate:
    """Convenience class for gating local compute and app execution based on tier/hardware."""

    @classmethod
    def can_use_local_compute(cls) -> bool:
        mgr = LicenseManager()
        return mgr.is_feature_allowed("silicon_offload")

    @classmethod
    def can_run_app(cls) -> bool:
        return True
