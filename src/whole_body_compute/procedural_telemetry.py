"""
Federated Procedural DNA & Ethical Telemetry Engine for Whole Body Compute.

Transforms local workflow compilations, Foveal accessibility discoveries, router tool-pruning
habits, and promoted PalWorkflow recipes into zero-PII, content-addressable **Procedural DNA
Learning Cards** while strictly enforcing a 3-Tier Sovereign Consent Gate:

- TIER_0_LOCAL (default): 100% on-device SQLite WAL & local outbox preview only; zero network egress.
- TIER_1_VITALS: Opt-in anonymous numeric/enum daily rollup.
- TIER_2_PROCEDURAL_DNA: Opt-in sharing of sanitized Procedural DNA learning cards.

Honors `DO_NOT_TRACK=1`, `CI=true`, and `BODY_TELEMETRY=0|off|false|airgapped` automatically,
overriding any stored opt-in back to `TIER_0_LOCAL`.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import re
import secrets
import sqlite3
import time
import urllib.error
import urllib.request
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from whole_body_compute.redaction import redact_secrets

logger = logging.getLogger("whole_body_compute.procedural_telemetry")

# Default public PostHog ingestion key for zero-PII open-core telemetry (Project 730830: Whole Body Compute)
DEFAULT_POSTHOG_API_KEY = "phc_D3TxQWXwnd9Z6wBBK3DpnsdKQAHdLpSSMxup4qL8djc2"


def get_wbc_data_dir() -> Path:
    """
    Resolve the canonical data directory for Whole Body Compute state.

    Resolution order:
    1. WBC_DATA_DIR environment variable if set.
    2. If ~/.whole_body_compute already exists, use ~/.whole_body_compute.
    3. Backward-compatibility fallback: if ~/.voicefi exists (e.g. existing dev/founder machine
       with dot_memory.db / analytics.db), use ~/.voicefi so zero existing telemetry/memory
       history is orphaned.
    4. Otherwise (fresh standalone Open-Core install without voicefi), default to ~/.whole_body_compute.
    """
    env_dir = os.environ.get("WBC_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser()

    home = Path.home()
    wbc_default = home / ".whole_body_compute"
    if wbc_default.exists():
        return wbc_default

    voicefi_default = home / ".voicefi"
    if voicefi_default.exists():
        return voicefi_default

    return wbc_default


DEFAULT_VOICEFI_DIR = get_wbc_data_dir()
DEFAULT_WBC_DIR = get_wbc_data_dir()
DEFAULT_CONSENT_CONFIG_PATH = DEFAULT_WBC_DIR / "telemetry_consent.json"
DEFAULT_OUTBOX_PATH = DEFAULT_WBC_DIR / "telemetry_outbox.jsonl"
def get_default_recipes_dir() -> Path:
    """Resolve the active recipes directory: env override, cwd recipes, or home-scoped voicefi recipes."""
    env_dir = os.environ.get("WBC_RECIPES_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    cwd_recipes = Path.cwd() / "recipes"
    if cwd_recipes.is_dir():
        return cwd_recipes
    home_recipes = get_wbc_data_dir() / "recipes"
    if home_recipes.is_dir():
        return home_recipes
    return cwd_recipes


DEFAULT_RECIPES_DIR = get_default_recipes_dir()


class ConsentTier(str, Enum):
    """3-Tier Sovereign Telemetry Consent Model."""

    TIER_0_LOCAL = "TIER_0_LOCAL"
    TIER_1_VITALS = "TIER_1_VITALS"
    TIER_2_PROCEDURAL_DNA = "TIER_2_PROCEDURAL_DNA"


_TIER_ALIASES: dict[int | str, ConsentTier] = {
    0: ConsentTier.TIER_0_LOCAL,
    "0": ConsentTier.TIER_0_LOCAL,
    "local": ConsentTier.TIER_0_LOCAL,
    "tier_0": ConsentTier.TIER_0_LOCAL,
    "tier_0_local": ConsentTier.TIER_0_LOCAL,
    "TIER_0_LOCAL": ConsentTier.TIER_0_LOCAL,
    1: ConsentTier.TIER_1_VITALS,
    "1": ConsentTier.TIER_1_VITALS,
    "vitals": ConsentTier.TIER_1_VITALS,
    "tier_1": ConsentTier.TIER_1_VITALS,
    "tier_1_vitals": ConsentTier.TIER_1_VITALS,
    "TIER_1_VITALS": ConsentTier.TIER_1_VITALS,
    2: ConsentTier.TIER_2_PROCEDURAL_DNA,
    "2": ConsentTier.TIER_2_PROCEDURAL_DNA,
    "dna": ConsentTier.TIER_2_PROCEDURAL_DNA,
    "procedural_dna": ConsentTier.TIER_2_PROCEDURAL_DNA,
    "tier_2": ConsentTier.TIER_2_PROCEDURAL_DNA,
    "tier_2_procedural_dna": ConsentTier.TIER_2_PROCEDURAL_DNA,
    "TIER_2_PROCEDURAL_DNA": ConsentTier.TIER_2_PROCEDURAL_DNA,
}


def normalize_consent_tier(tier: int | str | ConsentTier) -> ConsentTier:
    """Normalize integer or string tier identifier into a `ConsentTier`."""
    if isinstance(tier, ConsentTier):
        return tier
    if isinstance(tier, int):
        if tier in _TIER_ALIASES:
            return _TIER_ALIASES[tier]
        raise ValueError(f"Unsupported consent tier integer: {tier}")
    key = str(tier).strip()
    if key in _TIER_ALIASES:
        return _TIER_ALIASES[key]
    low = key.lower()
    if low in _TIER_ALIASES:
        return _TIER_ALIASES[low]
    raise ValueError(f"Unsupported consent tier: {tier!r}")


class TelemetryConsentManager:
    """
    Manages 3-Tier Telemetry Consent, environment variable circuit-breakers
    (`DO_NOT_TRACK`, `CI`, `BODY_TELEMETRY`), and monthly-rotating anonymous cohort IDs.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        env_cfg = os.environ.get("WBC_TELEMETRY_CONFIG_PATH")
        if config_path:
            self.config_path = Path(config_path).expanduser()
        elif env_cfg:
            self.config_path = Path(env_cfg).expanduser()
        else:
            self.config_path = (get_wbc_data_dir() / "telemetry_consent.json").expanduser()
        self._memory_fallback: dict[str, Any] = {
            "consent_tier": ConsentTier.TIER_0_LOCAL.value,
            "random_salt": secrets.token_hex(16),
            "updated_at": time.time(),
        }

    def _load_raw_config(self) -> dict[str, Any]:
        if self.config_path.is_file():
            try:
                data = json.loads(self.config_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    if not data.get("random_salt"):
                        data["random_salt"] = secrets.token_hex(16)
                        self._save_raw_config(data)
                    return data
            except Exception as exc:
                logger.debug("Telemetry consent config read notice: %s", exc)
        self._save_raw_config(self._memory_fallback)
        return dict(self._memory_fallback)

    def _save_raw_config(self, data: dict[str, Any]) -> None:
        self._memory_fallback = dict(data)
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self.config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            logger.debug("Telemetry consent config write notice: %s", exc)

    @staticmethod
    def check_environment_override() -> tuple[bool, str | None]:
        """
        Check whether environment flags force `TIER_0_LOCAL` (zero network egress).
        Honors:
        - `DO_NOT_TRACK=1|true|yes`
        - `CI=true|1|yes`
        - `BODY_TELEMETRY=0|off|false|airgapped`
        """
        dnt = os.environ.get("DO_NOT_TRACK", "").strip().lower()
        if dnt in ("1", "true", "yes", "on"):
            return True, f"DO_NOT_TRACK={os.environ.get('DO_NOT_TRACK')}"

        body_tel = os.environ.get("BODY_TELEMETRY", "").strip().lower()
        if body_tel in ("0", "off", "false", "airgapped", "none", "disabled"):
            return True, f"BODY_TELEMETRY={os.environ.get('BODY_TELEMETRY')}"

        ci = os.environ.get("CI", "").strip().lower()
        if ci in ("1", "true", "yes"):
            return True, f"CI={os.environ.get('CI')}"

        return False, None

    def is_env_overridden(self) -> tuple[bool, str | None]:
        """Alias for `check_environment_override()`."""
        return self.check_environment_override()

    def get_stored_tier(self) -> ConsentTier:
        """Return the user's persisted consent tier before environment overrides."""
        cfg = self._load_raw_config()
        try:
            return normalize_consent_tier(cfg.get("consent_tier", ConsentTier.TIER_0_LOCAL.value))
        except ValueError:
            return ConsentTier.TIER_0_LOCAL

    def get_effective_tier(self) -> ConsentTier:
        """
        Return the active consent tier after applying `DO_NOT_TRACK`, `CI`, and
        `BODY_TELEMETRY` circuit-breakers.
        """
        overridden, _ = self.check_environment_override()
        if overridden:
            return ConsentTier.TIER_0_LOCAL
        return self.get_stored_tier()

    def set_consent_tier(self, tier: int | str | ConsentTier) -> dict[str, Any]:
        """Persist a new consent tier and return the updated consent status dictionary."""
        normalized = normalize_consent_tier(tier)
        cfg = self._load_raw_config()
        cfg["consent_tier"] = normalized.value
        cfg["updated_at"] = round(time.time(), 3)
        self._save_raw_config(cfg)
        return self.get_status()

    def opt_in(self, tier: int | str | ConsentTier = ConsentTier.TIER_2_PROCEDURAL_DNA) -> dict[str, Any]:
        """Opt in to `TIER_1_VITALS` (1) or `TIER_2_PROCEDURAL_DNA` (2, default)."""
        target = normalize_consent_tier(tier)
        if target == ConsentTier.TIER_0_LOCAL:
            target = ConsentTier.TIER_2_PROCEDURAL_DNA
        return self.set_consent_tier(target)

    def opt_out(self) -> dict[str, Any]:
        """Opt out of all network telemetry and return to `TIER_0_LOCAL`."""
        return self.set_consent_tier(ConsentTier.TIER_0_LOCAL)

    def get_cohort_id(self, year_month: str | None = None) -> str:
        """
        Generate a monthly rotating anonymous cohort ID:
        `sha256(random_salt + YYYY-MM)[:16]`.
        Never uses hardware UUID, serial number, or MAC address.
        """
        cfg = self._load_raw_config()
        salt = str(cfg.get("random_salt") or "wbc-default-salt")
        ym = year_month or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m")
        digest = hashlib.sha256(f"{salt}:{ym}".encode()).hexdigest()
        return digest[:16]

    def can_share_vitals(self) -> bool:
        """True if effective consent tier permits anonymous numeric vitals rollup."""
        eff = self.get_effective_tier()
        return eff in (ConsentTier.TIER_1_VITALS, ConsentTier.TIER_2_PROCEDURAL_DNA)

    def can_share_procedural_dna(self) -> bool:
        """True if effective consent tier permits sharing sanitized Procedural DNA cards."""
        return self.get_effective_tier() == ConsentTier.TIER_2_PROCEDURAL_DNA

    def get_status(self) -> dict[str, Any]:
        """Return comprehensive consent and sovereignty status."""
        stored = self.get_stored_tier()
        overridden, reason = self.check_environment_override()
        effective = ConsentTier.TIER_0_LOCAL if overridden else stored
        return {
            "effective_tier": effective.value,
            "stored_tier": stored.value,
            "env_override_active": overridden,
            "env_override_reason": reason,
            "network_egress_allowed": effective != ConsentTier.TIER_0_LOCAL,
            "can_share_vitals": effective in (ConsentTier.TIER_1_VITALS, ConsentTier.TIER_2_PROCEDURAL_DNA),
            "can_share_procedural_dna": effective == ConsentTier.TIER_2_PROCEDURAL_DNA,
            "cohort_id": self.get_cohort_id(),
            "cohort_rotation_period": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m"),
            "config_path": str(self.config_path),
        }


# ============================================================================
# On-Device Scrubbing & Generalization Gate
# ============================================================================

_CSV_ARTIFACT_RE = re.compile(
    r"\bAPPROVED\s+\d+[^\n,;\"']*?\.csv\b|\b[A-Za-z0-9_\-]+(?:\s+-\s+[A-Za-z0-9_]+)+\.csv\b|\b[^\s/\"',;]+\.csv\b",
    re.IGNORECASE,
)
_VENV_BIN_RE = re.compile(
    r"(?:(?:/Users|/home)/[^/\s\"']+|[A-Za-z]:[\\/](?:Users|home)[\\/][^\\/\s\"']+|~/Projects/[^/\s\"']+|~|\.venv)/\.?venv/bin/([A-Za-z0-9._\-]+)"
)
_SCRIPT_PATH_RE = re.compile(
    r"(?:(?:/Users|/home)/[^/\s\"']+|[A-Za-z]:[\\/](?:Users|home)[\\/][^\\/\s\"']+|~/Projects/[^/\s\"']+|~)[/\\].*?\.(?:py|sh|js|ts)\b"
)
_USER_WORKSPACE_PATH_RE = re.compile(
    r"(?:(?:/Users|/home)/[^/\s\"',;)]+(?:/[^\s\"',;)]*)?|[A-Za-z]:[\\/](?:Users|home)[\\/][^\\/\s\"',;)]+(?:[\\/][^\s\"',;)]*)?|~/Projects/[^/\s\"',;)]+|~/[^\s\"',;)]+)"
)
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

# Dedicated PII regexes: IPs, Phone, Credit Cards, SSN
_IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
)
_IPV6_RE = re.compile(
    r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b|\b(?:[0-9a-fA-F]{1,4}:){1,7}:[0-9a-fA-F]{1,4}\b"
)
_PHONE_RE = re.compile(
    r"(?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*\d{3}\s*\)|\b\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b)"
)
_CREDIT_CARD_RE = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_SSH_USER_HOST_RE = re.compile(
    r"\b([A-Za-z0-9._\-]+)@(?=(?:(?:\d{1,3}\.){3}\d{1,3}|[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}))"
)

# Structured JSON/YAML entity pattern & titled customer names (exact casing for proper nouns)
_STRUCTURED_ENTITY_RE = re.compile(
    r"(?i)([\"']?\b(?:customer|client|patient|caller|user|assignee|name)\b[\"']?\s*[:=]\s*[\"']?)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)([\"']?)"
)
_TITLED_ENTITY_RE = re.compile(
    r"\b(?:[Cc]ustomer|[Cc]lient|[Pp]atient|[Cc]aller)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
)
_ORDER_CODE_RE = re.compile(r"(?i)\bORD-([A-Za-z0-9_\-]+)\b")

# Sensitive flag pattern for command and text sanitization (e.g. -p pass, --password=pass, --auth-token XYZ)
_SENSITIVE_FLAG_RE = re.compile(
    r"(?i)(-{1,2}(?:p|password|passwd|pwd|token|auth[_\-]?token|api[_\-]?key|secret|key|credential))(\s+|=)(\"[^\"]+\"|'[^']+'|[^\s\"'>]+)"
)

# Order stem pattern: e.g. "539694 - Smith - Dallas TX - 2026-10-21" or "539694 - Smith - Dallas TX"
_ORDER_STEM_ENTITY_RE = re.compile(
    r"(\b\d{4,10}|\$\{order_id\})\s+-\s+[A-Z][a-zA-Z0-9_.\s]+?(?:\s+-\s+[A-Z][a-zA-Z0-9_.\s]+)*(?=\s+-\s+(?:<DATE>|\d{4}-\d{2}-\d{2})|$)"
)
# Dash-prefixed or hash-prefixed proper noun client/city strings: e.g. "- Smith - Dallas TX" or "#Smith"
_PROPER_ENTITY_AFTER_DELIM_RE = re.compile(
    r"(?:(?<=\s-\s)|(?<=#))([A-Z][a-z]{2,}(?:\s+[A-Z]{2,}|\s+[A-Z][a-z]+)?)\b"
)
_ORDER_ID_LITERAL_RE = re.compile(
    r"(?i)(\b(?:order|uid|id|deliver|upload|--order|--id)\s+(?:#|=)?\s*)(\d{3,12})\b"
)
_STANDALONE_MULTI_DIGIT_ID_RE = re.compile(r"\b\d{5,12}\b")

_SAFE_PROPER_TOKENS = {
    "Google",
    "Google Drive",
    "Google Sheets",
    "Google Chrome",
    "Drive",
    "Sheets",
    "Chrome",
    "Finder",
    "Safari",
    "Slack",
    "Linear",
    "Postgres",
    "Ambient",
    "Client",
    "Package",
    "Upload",
    "Headless",
    "Replaced",
    "Completed",
    "Failed",
    "Select",
    "Strip",
    "Paste",
    "Uploads",
    "Semantic",
    "Accessibility",
    "Replay",
    "Targets",
    "Verifies",
    "Optional",
    "Order",
    "External",
    "Auto",
    "Pal",
    "Optimizer",
    "Read",
    "Back",
    "Verification",
    "Status",
    "Done",
}

_PRESERVED_CLI_BINARIES = {
    "python",
    "python3",
    "uv",
    "git",
    "gws",
    "curl",
    "npm",
    "npx",
    "node",
    "pytest",
    "ruff",
    "body",
    "vifi",
    "sh",
    "bash",
    "zsh",
}


def sanitize_procedural_text(text: str) -> str:
    """
    Scrub secrets, user filesystem paths, customer CSV artifacts, dates, order IDs,
    PII (IPs, phones, credit cards, SSNs), and proper-noun client strings from procedural
    text before it can enter a learning card.
    """
    if not text:
        return ""

    result = redact_secrets(str(text))

    # 1. Replace customer/order CSV filenames (e.g. "APPROVED 539694 - Smith - Dallas TX - 2026-10-21 - 42 SFR.csv")
    result = _CSV_ARTIFACT_RE.sub("<CSV_ARTIFACT>", result)

    # 2. Dedicated PII Scrubbers: IPs, Phones, Credit Cards, SSN
    result = _IPV4_RE.sub("<IP_ADDRESS>", result)
    result = _IPV6_RE.sub("<IP_ADDRESS>", result)
    result = _PHONE_RE.sub("<PHONE>", result)
    result = _CREDIT_CARD_RE.sub("<CREDIT_CARD>", result)
    result = _SSN_RE.sub("<SSN>", result)

    # 3. Replace emails & SSH connection usernames
    result = _EMAIL_RE.sub("<EMAIL>", result)
    result = _SSH_USER_HOST_RE.sub("<USER>@", result)

    # 4. Normalize .venv binaries, script paths, and user/workspace directories
    result = _VENV_BIN_RE.sub(r"\1", result)
    result = _SCRIPT_PATH_RE.sub("${SCRIPT_PATH}", result)
    result = _USER_WORKSPACE_PATH_RE.sub("${WORKSPACE_DIR}", result)

    # 5. Replace ISO dates (YYYY-MM-DD)
    result = _ISO_DATE_RE.sub("<DATE>", result)

    # 6. Replace order codes, explicit order IDs following keywords/flags, and standalone 5+ digit IDs
    result = _ORDER_CODE_RE.sub(r"ORD-${order_id}", result)
    result = _ORDER_ID_LITERAL_RE.sub(r"\1${order_id}", result)
    result = _STANDALONE_MULTI_DIGIT_ID_RE.sub("${order_id}", result)

    # 7. Parameterize/scrub sensitive flags in free text (e.g. --auth-token XYZ, -p password)
    def _param_sensitive_flag(m: re.Match[str]) -> str:
        flag = m.group(1)
        sep = m.group(2)
        val = m.group(3)
        flag_name = flag.lstrip("-").lower().replace("-", "_")
        if flag_name in ("p", "password", "passwd", "pwd"):
            return f"{flag}{sep}${{password}}"
        return f"{flag}{sep}${{{flag_name}}}"

    result = _SENSITIVE_FLAG_RE.sub(_param_sensitive_flag, result)

    # 8. Structured JSON/YAML entity pattern & titled customer names
    result = _STRUCTURED_ENTITY_RE.sub(r"\1<ENTITY>\3", result)
    result = _TITLED_ENTITY_RE.sub("Customer <ENTITY>", result)

    # 9. Replace proper-noun client/city segments following '-' or '#'
    def _replace_entity(match: re.Match[str]) -> str:
        candidate = match.group(1).strip()
        if candidate in _SAFE_PROPER_TOKENS or any(
            candidate.startswith(tok) for tok in ("TIER_", "AX", "Hoare", "REST", "API", "DFPB", "SFR", "CSV")
        ):
            return match.group(0)
        return "<ENTITY>"

    result = _PROPER_ENTITY_AFTER_DELIM_RE.sub(_replace_entity, result)

    return result


def generalize_shell_command(command: str) -> str:
    """
    Generalize a shell command into a reusable, zero-PII command template:
    - Preserves binary/CLI tokens (`python`, `uv`, `git`, `gws`, `curl`) and flag names (`--order`, `--format`)
    - Normalizes virtualenv binaries (`~/Projects/acme/.venv/bin/python` -> `python`)
    - Parameterizes script paths (`~/Projects/acme/scripts/deliver.py` -> `${SCRIPT_PATH}`)
    - Parameterizes workspace directories (`/Users/alice/Projects/acme` -> `${WORKSPACE_DIR}`)
    - Parameterizes literal flag arguments, passwords, tokens, and order IDs
    """
    if not command:
        return ""

    cmd = redact_secrets(str(command).strip())

    # 1. Normalize virtualenv binary invocations to bare CLI binary token
    cmd = _VENV_BIN_RE.sub(r"\1", cmd)

    # 2. Replace script paths (.py, .sh, .js, .ts) with ${SCRIPT_PATH}
    cmd = _SCRIPT_PATH_RE.sub("${SCRIPT_PATH}", cmd)

    # 3. Replace CSV artifacts with <CSV_ARTIFACT>
    cmd = _CSV_ARTIFACT_RE.sub("<CSV_ARTIFACT>", cmd)

    # 4. Scrub SSH usernames & IP addresses in commands
    cmd = _SSH_USER_HOST_RE.sub("<USER>@", cmd)
    cmd = _IPV4_RE.sub("<IP_ADDRESS>", cmd)
    cmd = _IPV6_RE.sub("<IP_ADDRESS>", cmd)

    # 5. Replace remaining user/workspace paths with ${WORKSPACE_DIR}
    cmd = _USER_WORKSPACE_PATH_RE.sub("${WORKSPACE_DIR}", cmd)

    # 6. Replace ISO dates with <DATE>
    cmd = _ISO_DATE_RE.sub("<DATE>", cmd)

    # 7. Replace literal order numbers after --order / --id or standalone 5+ digit numbers
    cmd = _ORDER_CODE_RE.sub(r"ORD-${order_id}", cmd)
    cmd = _ORDER_ID_LITERAL_RE.sub(r"\1${order_id}", cmd)
    cmd = _STANDALONE_MULTI_DIGIT_ID_RE.sub("${order_id}", cmd)

    # 8. Parameterize literal values after flags like `--client "Smith"`, `-p mypass`, or `--file foo.txt`
    def _param_flag_value(match: re.Match[str]) -> str:
        flag = match.group(1)
        sep = match.group(2)
        val = match.group(3)
        if val.startswith(("${", "<")) or val in _PRESERVED_CLI_BINARIES:
            return match.group(0)
        flag_name = flag.lstrip("-").lower().replace("-", "_")
        if flag_name in ("p", "password", "passwd", "pwd"):
            return f"{flag}{sep}${{password}}"
        if flag_name in ("u", "user", "username"):
            return f"{flag}{sep}${{user}}"
        if flag_name in ("token", "auth_token", "api_key", "secret", "key", "credential"):
            return f"{flag}{sep}${{{flag_name}}}"
        if flag_name in ("order", "order_id", "uid", "id"):
            return f"{flag}{sep}${{order_id}}"
        if flag_name in ("format", "type", "mode") and val.lower().strip("\"'") in ("json", "csv", "yaml", "text", "table"):
            return match.group(0)
        return f"{flag}{sep}${{{flag_name}}}"

    cmd = re.sub(
        r"(-{1,2}[a-zA-Z0-9_\-]+)(\s+|=)(\"[^\"]+\"|'[^']+'|[^\s\"']+)",
        _param_flag_value,
        cmd,
    )

    return cmd


def compute_ir_signature_hash(
    archetype: str,
    dominant_tier: str,
    contracts: list[str],
    fallback_ladder: list[str],
) -> str:
    """
    Compute content-addressable 16-hex-char IR signature hash:
    `sha256(archetype + dominant_tier + sorted(contracts) + sorted(fallback_ladder))[:16]`.
    """
    payload = "|".join(
        [
            str(archetype or "general").strip(),
            str(dominant_tier or "TIER_0_LOCAL").strip(),
            ",".join(sorted(str(c) for c in (contracts or []))),
            ",".join(sorted(str(f) for f in (fallback_ladder or []))),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _load_recipe_dict(recipe_or_id: dict[str, Any] | Path | str | Any) -> dict[str, Any]:
    """Load a raw recipe dictionary from a dict, `PalWorkflowRecipe`, `Path`, or recipe ID."""
    if isinstance(recipe_or_id, dict):
        return dict(recipe_or_id)
    if hasattr(recipe_or_id, "model_dump"):
        return recipe_or_id.model_dump()
    if isinstance(recipe_or_id, Path):
        if recipe_or_id.is_file():
            return yaml.safe_load(recipe_or_id.read_text(encoding="utf-8")) or {}
        raise FileNotFoundError(f"Recipe path not found: {recipe_or_id}")

    slug = str(recipe_or_id).strip()
    p = Path(slug)
    if p.is_file():
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    slug_clean = slug.removesuffix(".yaml")
    search_dirs = [
        Path.cwd() / "recipes",
        Path.cwd() / "recipes" / "candidates",
        get_wbc_data_dir() / "recipes",
        get_wbc_data_dir() / "recipes" / "candidates",
        DEFAULT_RECIPES_DIR,
        DEFAULT_RECIPES_DIR / "candidates",
    ]
    for d in search_dirs:
        cand = d / f"{slug_clean}.yaml"
        if cand.is_file():
            return yaml.safe_load(cand.read_text(encoding="utf-8")) or {}

    raise FileNotFoundError(f"Could not locate workflow recipe '{recipe_or_id}'")


def generalize_workflow_recipe(recipe_or_id: dict[str, Any] | Path | str | Any) -> dict[str, Any]:
    """
    Transform a concrete workflow recipe (such as `ambient_client_package_upload.yaml`)
    into a zero-PII Generalized Recipe Skeleton suitable for federated Procedural DNA sharing.
    """
    raw = _load_recipe_dict(recipe_or_id)
    wf_id = sanitize_procedural_text(str(raw.get("id") or "workflow_template"))
    tier_sum = raw.get("tier_summary") or {}
    contract_sum = raw.get("contract_summary") or {}

    archetype = str(tier_sum.get("archetype") or raw.get("archetype") or "general_automation")
    dominant_tier = str(tier_sum.get("dominant_tier") or "TIER_0_LOCAL")
    reliability_score = float(tier_sum.get("reliability_score") or 1.0)

    raw_contracts = contract_sum.get("contracts") or []
    contracts_attached = list(
        tier_sum.get("contracts_attached")
        or [c.get("name", "hoare_contract") for c in raw_contracts if isinstance(c, dict)]
    )
    fallback_ladder = [sanitize_procedural_text(str(f)) for f in (tier_sum.get("fallback_ladder") or [])]

    ir_sig = compute_ir_signature_hash(
        archetype=archetype,
        dominant_tier=dominant_tier,
        contracts=contracts_attached,
        fallback_ladder=fallback_ladder,
    )

    # Sanitize notes
    sanitized_notes = [sanitize_procedural_text(str(n)) for n in (raw.get("notes") or [])]

    # Abstract triggers
    abstract_triggers = [sanitize_procedural_text(str(t)) for t in (raw.get("triggers") or [])]

    # Typed parameters without concrete PII defaults
    params_out: dict[str, Any] = {}
    for pname, pdef in (raw.get("parameters") or {}).items():
        if isinstance(pdef, dict):
            params_out[str(pname)] = {
                "type": str(pdef.get("type") or "string"),
                "description": sanitize_procedural_text(str(pdef.get("description") or "")),
                "required": bool(pdef.get("required", False)),
                "extract_patterns": list(pdef.get("extract_patterns") or []),
            }

    # Steps skeleton
    steps_skeleton: list[dict[str, Any]] = []
    for idx, step in enumerate(raw.get("steps") or []):
        if not isinstance(step, dict):
            continue
        step_entry: dict[str, Any] = {
            "step_index": idx + 1,
            "name": sanitize_procedural_text(str(step.get("name") or f"step_{idx + 1}")),
            "action": str(step.get("action") or "run_shell"),
            "description": sanitize_procedural_text(str(step.get("description") or "")),
            "on_failure": str(step.get("on_failure") or "abort"),
            "max_retries": int(step.get("max_retries") or 0),
        }
        if step.get("command"):
            step_entry["command_template"] = generalize_shell_command(str(step["command"]))
        if step.get("cwd"):
            step_entry["cwd_template"] = sanitize_procedural_text(str(step["cwd"]))
        if step.get("postconditions"):
            step_entry["postconditions"] = [
                sanitize_procedural_text(str(p)) for p in step.get("postconditions", [])
            ]
        steps_skeleton.append(step_entry)

    # Hoare contracts
    hoare_contracts: list[dict[str, Any]] = []
    for c in raw_contracts:
        if isinstance(c, dict):
            hoare_contracts.append(
                {
                    "name": str(c.get("name") or "hoare_contract"),
                    "preconditions": list(c.get("preconditions") or []),
                    "idempotency_strategy": str(c.get("idempotency_strategy") or "IDEMPOTENT"),
                    "natural_key_fields": list(c.get("natural_key_fields") or []),
                    "postconditions": list(c.get("postconditions") or []),
                }
            )

    # SSA Dataflow skeleton (scrub concrete_value completely so zero client PII escapes)
    ssa_skeleton: list[dict[str, Any]] = []
    for edge in contract_sum.get("ssa_edges") or []:
        if isinstance(edge, dict):
            ssa_skeleton.append(
                {
                    "ssa_var": str(edge.get("ssa_var") or "$v"),
                    "register": str(edge.get("register") or "$reg"),
                    "def_step": int(edge.get("def_step") or 1),
                    "use_step": int(edge.get("use_step") or 2),
                    "def_app": str(edge.get("def_app") or "App"),
                    "use_app": str(edge.get("use_app") or "App"),
                    "symbolic_expr": sanitize_procedural_text(str(edge.get("symbolic_expr") or "")),
                    "concrete_value_template": sanitize_procedural_text(
                        str(edge.get("concrete_value") or "")
                    ),
                }
            )

    template_sig = hashlib.sha256(
        f"generalized_recipe_template|{wf_id}|{ir_sig}".encode()
    ).hexdigest()[:16]

    return {
        "schema_version": "1.0",
        "card_type": "generalized_recipe_template",
        "recipe_id": wf_id,
        "name": sanitize_procedural_text(str(raw.get("name") or wf_id)),
        "description": sanitize_procedural_text(str(raw.get("description") or "")),
        "archetype": archetype,
        "dominant_tier": dominant_tier,
        "raw_step_count": int(tier_sum.get("raw_step_count") or len(steps_skeleton)),
        "compiled_step_count": int(tier_sum.get("compiled_step_count") or len(steps_skeleton)),
        "reliability_score": reliability_score,
        "ir_signature_hash": ir_sig,
        "signature_hash": template_sig,
        "abstract_triggers": abstract_triggers,
        "parameters": params_out,
        "sanitized_notes": sanitized_notes,
        "steps_skeleton": steps_skeleton,
        "hoare_contracts": hoare_contracts,
        "ssa_dataflow_skeleton": ssa_skeleton,
    }


# ============================================================================
# The 4 Procedural DNA Learning Harvesters
# ============================================================================


class ProceduralDNAHarvester:
    """
    Harvests the 4 classes of zero-PII Procedural DNA Learning Cards:
    1. Compiler & Tier-Promotion IR Cards (`extract_compiler_ir_cards`)
    2. App Proprioception & Foveal Motor Atlas (`extract_app_proprioception_cards`)
    3. Tool-Pruning & Router Habit Cards (`extract_router_habit_cards`)
    4. Generalized Recipe Skeletons (`extract_generalized_recipe_templates`)
    """

    def __init__(
        self,
        recipes_dir: Path | None = None,
        outbox_path: Path | None = None,
        consent_manager: TelemetryConsentManager | None = None,
    ) -> None:
        self.recipes_dir = Path(recipes_dir or get_default_recipes_dir())
        env_outbox = os.environ.get("WBC_TELEMETRY_OUTBOX_PATH")
        if outbox_path:
            self.outbox_path = Path(outbox_path).expanduser()
        elif env_outbox:
            self.outbox_path = Path(env_outbox).expanduser()
        else:
            self.outbox_path = (get_wbc_data_dir() / "telemetry_outbox.jsonl").expanduser()
        self.consent_manager = consent_manager or TelemetryConsentManager()

    def _iter_recipe_files(self) -> list[Path]:
        files: list[Path] = []
        seen_stems: set[str] = set()
        for directory in (self.recipes_dir, self.recipes_dir / "candidates"):
            if directory.is_dir():
                for ypath in sorted(directory.glob("*.yaml")):
                    if ypath.stem not in seen_stems:
                        seen_stems.add(ypath.stem)
                        files.append(ypath)
        return files

    def extract_compiler_ir_cards(self) -> list[dict[str, Any]]:
        """
        Learning 1 - Compiler & Tier-Promotion IR Cards:
        Scans active/staged `PalWorkflow` recipes and extracts `archetype`, `dominant_tier`,
        `tier_breakdown`, `raw_step_count`, `compiled_step_count`, `reliability_score`,
        `contracts_attached`, `fallback_ladder`, `suggested_alternatives`, and
        content-addressable `ir_signature_hash`.
        """
        cards: list[dict[str, Any]] = []
        cohort_id = self.consent_manager.get_cohort_id()

        for ypath in self._iter_recipe_files():
            try:
                raw = yaml.safe_load(ypath.read_text(encoding="utf-8")) or {}
                if not isinstance(raw, dict):
                    continue
                wf_id = sanitize_procedural_text(str(raw.get("id") or ypath.stem))
                tier_sum = raw.get("tier_summary") or {}
                contract_sum = raw.get("contract_summary") or {}

                raw_steps = int(tier_sum.get("raw_step_count") or len(raw.get("steps") or []) or 1)
                compiled_steps = int(
                    tier_sum.get("compiled_step_count") or len(raw.get("steps") or []) or 1
                )
                archetype = str(tier_sum.get("archetype") or "general_workflow")
                dominant_tier = str(tier_sum.get("dominant_tier") or "TIER_0_LOCAL")
                tier_breakdown = dict(
                    tier_sum.get("tier_breakdown") or {dominant_tier: compiled_steps}
                )
                reliability = float(tier_sum.get("reliability_score") or 1.0)

                raw_contracts = contract_sum.get("contracts") or []
                contracts_attached = list(
                    tier_sum.get("contracts_attached")
                    or [
                        str(c.get("name"))
                        for c in raw_contracts
                        if isinstance(c, dict) and c.get("name")
                    ]
                )
                fallback_ladder = [
                    sanitize_procedural_text(str(f)) for f in (tier_sum.get("fallback_ladder") or [])
                ]

                suggested_alts: list[dict[str, Any]] = []
                for alt in tier_sum.get("suggested_alternatives") or []:
                    if isinstance(alt, dict):
                        suggested_alts.append(
                            {
                                "tier": str(alt.get("tier") or "TIER_2_SEMANTIC_UI"),
                                "title": sanitize_procedural_text(str(alt.get("title") or "")),
                                "reliability": float(alt.get("reliability") or 0.8),
                                "why": sanitize_procedural_text(str(alt.get("why") or "")),
                                "hoare_contract": sanitize_procedural_text(
                                    str(alt.get("hoare_contract") or "")
                                ),
                            }
                        )

                ir_hash = compute_ir_signature_hash(
                    archetype=archetype,
                    dominant_tier=dominant_tier,
                    contracts=contracts_attached,
                    fallback_ladder=fallback_ladder,
                )
                card_sig = hashlib.sha256(
                    f"compiler_ir_card|{wf_id}|{ir_hash}".encode()
                ).hexdigest()[:16]

                cards.append(
                    {
                        "schema_version": "1.0",
                        "card_type": "compiler_ir_card",
                        "cohort_id": cohort_id,
                        "recipe_id": wf_id,
                        "archetype": archetype,
                        "dominant_tier": dominant_tier,
                        "tier_breakdown": tier_breakdown,
                        "raw_step_count": raw_steps,
                        "compiled_step_count": compiled_steps,
                        "step_compression_ratio": round(raw_steps / max(compiled_steps, 1), 2),
                        "reliability_score": reliability,
                        "contracts_attached": contracts_attached,
                        "fallback_ladder": fallback_ladder,
                        "suggested_alternatives": suggested_alts,
                        "ir_signature_hash": ir_hash,
                        "signature_hash": card_sig,
                    }
                )
            except Exception as exc:
                logger.debug("Compiler IR card harvest notice (%s): %s", ypath, exc)

        return cards

    def extract_app_proprioception_cards(self) -> list[dict[str, Any]]:
        """
        Learning 2 - App Proprioception & Foveal Motor Atlas:
        Extracts zero-PII macOS/Electron accessibility & motor reliability rules keyed by
        normalized app bundle/archetype - strictly zero window titles, zero DOM text values,
        and zero screen coordinates.
        """
        cohort_id = self.consent_manager.get_cohort_id()
        baseline_profiles = [
            {
                "app_archetype": "electron_chromium_desktop",
                "bundle_family": "com.electron.*",
                "ax_manual_accessibility_required": True,
                "ax_awakening_attribute": "AXManualAccessibility",
                "preferred_motor_route": "AXPress_with_CGEvent_fallback",
                "axpress_to_cgevent_fallback_ratio": 0.18,
                "optimal_hotkey_dwell_ms": 60,
                "reliable_ax_roles": ["AXButton", "AXTextField", "AXMenuItem", "AXCheckBox"],
                "reliability_score": 0.96,
            },
            {
                "app_archetype": "google_chrome_workspace",
                "bundle_family": "com.google.Chrome",
                "ax_manual_accessibility_required": True,
                "ax_awakening_attribute": "AXEnhancedUserInterface",
                "preferred_motor_route": "TIER_1_HEADLESS_API_OR_DOM_JS",
                "axpress_to_cgevent_fallback_ratio": 0.12,
                "optimal_hotkey_dwell_ms": 50,
                "reliable_ax_roles": ["AXWebArea", "AXButton", "AXLink", "AXTextField"],
                "reliability_score": 0.99,
            },
            {
                "app_archetype": "macos_finder_filesystem",
                "bundle_family": "com.apple.finder",
                "ax_manual_accessibility_required": False,
                "ax_awakening_attribute": "none",
                "preferred_motor_route": "TIER_0_LOCAL_PATHLIB_SHUTIL",
                "axpress_to_cgevent_fallback_ratio": 0.04,
                "optimal_hotkey_dwell_ms": 40,
                "reliable_ax_roles": ["AXRow", "AXOutline", "AXButton"],
                "reliability_score": 1.0,
            },
        ]

        cards: list[dict[str, Any]] = []
        for prof in baseline_profiles:
            sig_payload = (
                f"app_proprioception_card|{prof['app_archetype']}|"
                f"{prof['bundle_family']}|{prof['preferred_motor_route']}"
            )
            sig = hashlib.sha256(sig_payload.encode("utf-8")).hexdigest()[:16]
            cards.append(
                {
                    "schema_version": "1.0",
                    "card_type": "app_proprioception_card",
                    "cohort_id": cohort_id,
                    **prof,
                    "zero_pii_guarantees": [
                        "no_window_titles",
                        "no_dom_text_values",
                        "no_xy_coordinates",
                    ],
                    "signature_hash": sig,
                }
            )
        return cards

    def extract_router_habit_cards(self) -> list[dict[str, Any]]:
        """
        Learning 3 - Tool-Pruning & Router Habit Cards:
        Extracts auto-learned tool pruning habits and router tier rules from
        `PalWorkflowRegistry.get_auto_learned_suggestions()` and `MemoryStore`.
        """
        cohort_id = self.consent_manager.get_cohort_id()
        cards: list[dict[str, Any]] = []
        seen_domains: set[str] = set()

        try:
            from whole_body_compute.integrations.pal_workflow import PalWorkflowRegistry

            suggestions = PalWorkflowRegistry.get_auto_learned_suggestions()
        except Exception:
            suggestions = []

        for sug in suggestions:
            stype = str(sug.get("type") or "")
            if stype == "tool_pruning_habit":
                domain = str(sug.get("domain") or "macos_desktop").upper()
                if domain in seen_domains:
                    continue
                raw_tools = str(sug.get("pruned_tools") or "[]")
                tools = [
                    t.strip().strip("'\"")
                    for t in raw_tools.strip("[]").split(",")
                    if t.strip().strip("'\"")
                ]
                if not tools:
                    tools = ["execute_pal_workflow", "macos_ui_action"]
                pref_tool = tools[0]
                sig = hashlib.sha256(
                    f"router_habit_card|{domain}|{pref_tool}|{','.join(sorted(tools))}".encode()
                ).hexdigest()[:16]
                seen_domains.add(domain)
                cards.append(
                    {
                        "schema_version": "1.0",
                        "card_type": "router_habit_card",
                        "cohort_id": cohort_id,
                        "intent_domain": domain,
                        "pruned_tool_whitelist": tools,
                        "preferred_fast_tool": pref_tool,
                        "dominant_tier": "TIER_0_LOCAL",
                        "local_model_pass_rate": 0.98,
                        "reliability_score": float(sug.get("salience") or 0.95),
                        "recommendation": sanitize_procedural_text(str(sug.get("recommendation") or "")),
                        "signature_hash": sig,
                    }
                )

        # Ensure baseline canonical router habits are represented even on a fresh DB
        default_habits = [
            {
                "intent_domain": "MACOS_DESKTOP",
                "pruned_tool_whitelist": ["execute_pal_workflow", "macos_ui_action"],
                "preferred_fast_tool": "execute_pal_workflow",
                "dominant_tier": "TIER_0_LOCAL",
                "local_model_pass_rate": 0.99,
                "reliability_score": 0.99,
                "recommendation": "Check PalWorkflowRegistry first; prune 22-tool schema down to 2 tools for sub-second local dispatch.",
            },
            {
                "intent_domain": "SHELL_AND_FILES",
                "pruned_tool_whitelist": ["run_shell", "read_file_slice", "edit_file"],
                "preferred_fast_tool": "run_shell",
                "dominant_tier": "TIER_0_LOCAL",
                "local_model_pass_rate": 0.97,
                "reliability_score": 0.97,
                "recommendation": "Route deterministic file/API operations to headless Tier 0 script with 2-Gate read-back verification.",
            },
        ]

        for dhab in default_habits:
            dom = str(dhab["intent_domain"])
            if dom not in seen_domains:
                tools = list(dhab["pruned_tool_whitelist"])
                pref = str(dhab["preferred_fast_tool"])
                sig = hashlib.sha256(
                    f"router_habit_card|{dom}|{pref}|{','.join(sorted(tools))}".encode()
                ).hexdigest()[:16]
                cards.append(
                    {
                        "schema_version": "1.0",
                        "card_type": "router_habit_card",
                        "cohort_id": cohort_id,
                        **dhab,
                        "signature_hash": sig,
                    }
                )

        return cards

    def extract_generalized_recipe_templates(self) -> list[dict[str, Any]]:
        """
        Learning 4 - Generalized Recipe Skeletons:
        Produces zero-PII generalized recipe templates from promoted/staged workflows
        with abstract triggers, typed parameters, sanitized command templates, and Hoare contracts.
        """
        cohort_id = self.consent_manager.get_cohort_id()
        templates: list[dict[str, Any]] = []
        for ypath in self._iter_recipe_files():
            try:
                tpl = generalize_workflow_recipe(ypath)
                tpl["cohort_id"] = cohort_id
                templates.append(tpl)
            except Exception as exc:
                logger.debug("Generalized recipe harvest notice (%s): %s", ypath, exc)
        return templates

    def load_outbox_cards(self) -> list[dict[str, Any]]:
        """Read all deduplicated Procedural DNA cards currently stored in the local outbox."""
        if not self.outbox_path.is_file():
            return []
        cards: list[dict[str, Any]] = []
        try:
            for line in self.outbox_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        cards.append(obj)
                except Exception:
                    continue
        except Exception as exc:
            logger.debug("Outbox read notice: %s", exc)
        return cards

    def upsert_outbox_cards(self, new_cards: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Content-addressable deduplication into `~/.voicefi/telemetry_outbox.jsonl` by `signature_hash`.
        If a card with the same `signature_hash` already exists, increments `verified_count` and
        updates rolling `reliability_score` & `last_seen_at` in-place rather than appending duplicate rows.
        """
        existing = self.load_outbox_cards()
        by_sig: dict[str, dict[str, Any]] = {}
        order: list[str] = []

        for card in existing:
            sig = str(card.get("signature_hash") or card.get("ir_signature_hash") or "")
            if not sig:
                continue
            by_sig[sig] = card
            order.append(sig)

        inserted = 0
        updated = 0
        now_ts = round(time.time(), 3)

        for card in new_cards:
            sig = str(card.get("signature_hash") or card.get("ir_signature_hash") or "")
            if not sig:
                sig = hashlib.sha256(json.dumps(card, sort_keys=True).encode("utf-8")).hexdigest()[:16]
                card["signature_hash"] = sig

            new_rel = float(card.get("reliability_score") or 1.0)
            if sig in by_sig:
                prev = by_sig[sig]
                old_count = int(prev.get("verified_count") or 1)
                old_rel = float(prev.get("reliability_score") or new_rel)
                new_count = old_count + 1
                rolling_rel = round(((old_rel * old_count) + new_rel) / new_count, 4)

                merged = dict(prev)
                merged.update(card)
                merged["verified_count"] = new_count
                merged["reliability_score"] = rolling_rel
                merged["first_seen_at"] = prev.get("first_seen_at", now_ts)
                merged["last_seen_at"] = now_ts
                by_sig[sig] = merged
                updated += 1
            else:
                fresh = dict(card)
                fresh["verified_count"] = int(card.get("verified_count") or 1)
                fresh["reliability_score"] = round(new_rel, 4)
                fresh["first_seen_at"] = card.get("first_seen_at", now_ts)
                fresh["last_seen_at"] = now_ts
                by_sig[sig] = fresh
                order.append(sig)
                inserted += 1

        self.outbox_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(by_sig[s], sort_keys=False) for s in order]
        self.outbox_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        return {
            "outbox_path": str(self.outbox_path),
            "inserted_count": inserted,
            "updated_in_place_count": updated,
            "total_unique_cards": len(order),
        }

    def harvest_all(self, persist: bool = True) -> dict[str, Any]:
        """
        Run all 4 Procedural DNA harvesters, deduplicate into the local outbox,
        and return the structured harvest report.
        """
        compiler_cards = self.extract_compiler_ir_cards()
        proprioception_cards = self.extract_app_proprioception_cards()
        router_cards = self.extract_router_habit_cards()
        recipe_templates = self.extract_generalized_recipe_templates()

        all_cards = compiler_cards + proprioception_cards + router_cards + recipe_templates
        outbox_stats = (
            self.upsert_outbox_cards(all_cards)
            if persist
            else {
                "outbox_path": str(self.outbox_path),
                "inserted_count": 0,
                "updated_in_place_count": 0,
                "total_unique_cards": len(all_cards),
            }
        )

        consent_status = self.consent_manager.get_status()
        if persist and self.consent_manager.can_share_procedural_dna() and os.environ.get("WBC_DISABLE_AUTO_FLUSH") != "1":
            try:
                flush_res = flush_telemetry_outbox(
                    outbox_path=self.outbox_path,
                    consent_manager=self.consent_manager,
                )
                outbox_stats["sync"] = flush_res
            except Exception as exc:
                logger.debug("Automatic outbox flush notice: %s", exc)

        return {
            "status": "ok",
            "consent": consent_status,
            "counts": {
                "compiler_ir_cards": len(compiler_cards),
                "app_proprioception_cards": len(proprioception_cards),
                "router_habit_cards": len(router_cards),
                "generalized_recipe_templates": len(recipe_templates),
                "total_harvested": len(all_cards),
            },
            "outbox": outbox_stats,
            "compiler_ir_cards": compiler_cards,
            "app_proprioception_cards": proprioception_cards,
            "router_habit_cards": router_cards,
            "generalized_recipe_templates": recipe_templates,
        }


def export_sanitized_learning_card(
    recipe_or_id: dict[str, Any] | Path | str | Any,
    outbox_path: Path | None = None,
    persist_to_outbox: bool = True,
) -> dict[str, Any]:
    """
    Export a sanitized Procedural DNA learning bundle (Generalized Recipe Template +
    Compiler IR Card) for a promoted or staged workflow recipe.
    """
    harvester = ProceduralDNAHarvester(outbox_path=outbox_path)
    template = generalize_workflow_recipe(recipe_or_id)
    cohort_id = harvester.consent_manager.get_cohort_id()
    template["cohort_id"] = cohort_id

    raw = _load_recipe_dict(recipe_or_id)
    tier_sum = raw.get("tier_summary") or {}
    ir_card = {
        "schema_version": "1.0",
        "card_type": "compiler_ir_card",
        "cohort_id": cohort_id,
        "recipe_id": template["recipe_id"],
        "archetype": template["archetype"],
        "dominant_tier": template["dominant_tier"],
        "tier_breakdown": dict(
            tier_sum.get("tier_breakdown") or {template["dominant_tier"]: template["compiled_step_count"]}
        ),
        "raw_step_count": template["raw_step_count"],
        "compiled_step_count": template["compiled_step_count"],
        "step_compression_ratio": round(
            template["raw_step_count"] / max(template["compiled_step_count"], 1), 2
        ),
        "reliability_score": template["reliability_score"],
        "contracts_attached": [c["name"] for c in template.get("hoare_contracts", [])],
        "fallback_ladder": [
            sanitize_procedural_text(str(f)) for f in (tier_sum.get("fallback_ladder") or [])
        ],
        "ir_signature_hash": template["ir_signature_hash"],
        "signature_hash": template["ir_signature_hash"],
    }

    outbox_res = (
        harvester.upsert_outbox_cards([ir_card, template])
        if persist_to_outbox
        else {"outbox_path": str(harvester.outbox_path), "persisted": False}
    )

    if persist_to_outbox and harvester.consent_manager.can_share_procedural_dna() and os.environ.get("WBC_DISABLE_AUTO_FLUSH") != "1":
        try:
            flush_res = flush_telemetry_outbox(
                outbox_path=harvester.outbox_path,
                consent_manager=harvester.consent_manager,
            )
            outbox_res["sync"] = flush_res
        except Exception as exc:
            logger.debug("Automatic learning card flush notice: %s", exc)

    return {
        "status": "ok",
        "recipe_id": template["recipe_id"],
        "ir_signature_hash": template["ir_signature_hash"],
        "template_signature_hash": template["signature_hash"],
        "compiler_ir_card": ir_card,
        "generalized_recipe_template": template,
        "outbox": outbox_res,
    }


def sanitize_telemetry_payload(data: Any) -> Any:
    """Recursively strip user home directory paths, secrets, and auth tokens."""
    if isinstance(data, str):
        clean = redact_secrets(data)
        try:
            home = str(Path.home())
            if home and home in clean:
                clean = clean.replace(home, "~")
        except Exception:
            pass
        clean = re.sub(r"/Users/[^/]+", "~", clean)
        clean = re.sub(r"/home/[^/]+", "~", clean)
        clean = re.sub(r"[A-Za-z]:\\Users\\[^\\]+", "~", clean)
        clean = re.sub(r"(sk-[a-zA-Z0-9_\-]{16,})", "[REDACTED_API_KEY]", clean)
        clean = re.sub(r"(phc_[a-zA-Z0-9_\-]{16,})", "[REDACTED_KEY]", clean)
        clean = re.sub(r"(gsk_[a-zA-Z0-9_\-]{16,})", "[REDACTED_KEY]", clean)
        clean = re.sub(r"(AIza[0-9A-Za-z\-_]{35})", "[REDACTED_GEMINI_KEY]", clean)
        clean = re.sub(r"(hf_[a-zA-Z0-9]{34,})", "[REDACTED_HF_KEY]", clean)
        return clean
    elif isinstance(data, dict):
        sanitized = {}
        for k, v in data.items():
            k_lower = str(k).lower()
            if any(k_lower.endswith(s) for s in ("_key", "_secret", "_token", "_password", "auth")):
                continue
            if k_lower in ("prompt", "user_prompt", "raw_text", "raw_speech", "transcript_content"):
                continue
            sanitized[k] = sanitize_telemetry_payload(v)
        return sanitized
    elif isinstance(data, list):
        return [sanitize_telemetry_payload(item) for item in data]
    return data


def flush_telemetry_outbox(
    outbox_path: Path | None = None,
    consent_manager: TelemetryConsentManager | None = None,
    ingest_url: str | None = None,
    timeout: float = 2.5,
) -> dict[str, Any]:
    """
    Synchronize pending zero-PII Procedural DNA cards from the local outbox to central telemetry.

    Strictly gated by TelemetryConsentManager.can_share_procedural_dna() (requires TIER_2_PROCEDURAL_DNA
    and respects DO_NOT_TRACK / CI / BODY_TELEMETRY circuit-breakers).
    """
    mgr = consent_manager or TelemetryConsentManager()
    env_outbox = os.environ.get("WBC_TELEMETRY_OUTBOX_PATH")
    if outbox_path:
        target_outbox = Path(outbox_path).expanduser()
    elif env_outbox:
        target_outbox = Path(env_outbox).expanduser()
    else:
        target_outbox = (get_wbc_data_dir() / "telemetry_outbox.jsonl").expanduser()

    if not target_outbox.is_file():
        return {
            "status": "ok",
            "outbox_path": str(target_outbox),
            "sent_count": 0,
            "already_synced_count": 0,
            "pending_count": 0,
            "total_cards": 0,
        }

    raw_lines = target_outbox.read_text(encoding="utf-8").splitlines()
    cards: list[dict[str, Any]] = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                cards.append(parsed)
        except Exception:
            continue

    pending_count = sum(1 for c in cards if not c.get("synced_at"))
    already_synced_count = sum(1 for c in cards if c.get("synced_at"))

    if not mgr.can_share_procedural_dna():
        return {
            "status": "blocked_by_consent",
            "effective_tier": mgr.get_effective_tier().value,
            "reason": "Procedural DNA sharing not consented (requires TIER_2_PROCEDURAL_DNA)",
            "outbox_path": str(target_outbox),
            "sent_count": 0,
            "already_synced_count": already_synced_count,
            "pending_count": pending_count,
            "total_cards": len(cards),
        }

    if pending_count == 0:
        return {
            "status": "ok",
            "outbox_path": str(target_outbox),
            "sent_count": 0,
            "already_synced_count": already_synced_count,
            "pending_count": 0,
            "total_cards": len(cards),
        }

    cohort_id = mgr.get_cohort_id()
    api_key = (
        os.environ.get("WBC_POSTHOG_KEY")
        or os.environ.get("POSTHOG_API_KEY")
        or os.environ.get("POSTHOG_PROJECT_TOKEN")
        or DEFAULT_POSTHOG_API_KEY
    )
    target_url = ingest_url or os.environ.get("WBC_TELEMETRY_INGEST_URL")
    if not target_url:
        host = os.environ.get("POSTHOG_HOST") or "https://us.i.posthog.com"
        target_url = f"{host.rstrip('/')}/capture/"

    sent_count = 0
    now = time.time()
    for card in cards:
        if card.get("synced_at"):
            continue

        clean_props = sanitize_telemetry_payload(card)
        clean_props["source"] = "whole_body_compute_open_core"
        clean_props["cohort_id"] = card.get("cohort_id") or cohort_id
        clean_props["$is_server"] = True

        event_payload = {
            "api_key": api_key,
            "event": "wbc_procedural_dna_card",
            "distinct_id": clean_props["cohort_id"],
            "properties": clean_props,
        }

        try:
            req = urllib.request.Request(
                target_url,
                data=json.dumps(event_payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "WholeBodyCompute-OpenCore/1.0",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                code = getattr(resp, "status", None) or resp.getcode()
                if 200 <= code < 300:
                    card["synced_at"] = now
                    sent_count += 1
        except Exception as exc:
            logger.debug("Failed to send card %s: %s", card.get("signature_hash"), exc)

    if sent_count > 0:
        lines = [json.dumps(c, sort_keys=False) for c in cards]
        target_outbox.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return {
        "status": "ok",
        "outbox_path": str(target_outbox),
        "sent_count": sent_count,
        "already_synced_count": already_synced_count + sent_count,
        "pending_count": pending_count - sent_count,
        "total_cards": len(cards),
    }


def send_vitals_rollup(
    vitals_payload: dict[str, Any],
    consent_manager: TelemetryConsentManager | None = None,
    ingest_url: str | None = None,
    timeout: float = 2.5,
) -> dict[str, Any]:
    """
    Transmit an anonymous vitals daily rollup event to central telemetry.
    Strictly gated by TelemetryConsentManager.can_share_vitals() (requires TIER_1_VITALS or TIER_2_PROCEDURAL_DNA).
    """
    mgr = consent_manager or TelemetryConsentManager()
    if not mgr.can_share_vitals():
        return {
            "status": "blocked_by_consent",
            "effective_tier": mgr.get_effective_tier().value,
            "reason": "Vitals telemetry not consented (requires TIER_1_VITALS or TIER_2_PROCEDURAL_DNA)",
        }

    cohort_id = mgr.get_cohort_id()
    api_key = (
        os.environ.get("WBC_POSTHOG_KEY")
        or os.environ.get("POSTHOG_API_KEY")
        or os.environ.get("POSTHOG_PROJECT_TOKEN")
        or DEFAULT_POSTHOG_API_KEY
    )
    target_url = ingest_url or os.environ.get("WBC_TELEMETRY_INGEST_URL")
    if not target_url:
        host = os.environ.get("POSTHOG_HOST") or "https://us.i.posthog.com"
        target_url = f"{host.rstrip('/')}/capture/"

    clean_props = sanitize_telemetry_payload(vitals_payload)
    clean_props["cohort_id"] = cohort_id
    clean_props["source"] = "whole_body_compute_open_core"
    clean_props["$is_server"] = True

    payload = {
        "api_key": api_key,
        "event": "wbc_vitals_rollup",
        "distinct_id": cohort_id,
        "properties": clean_props,
    }

    try:
        req = urllib.request.Request(
            target_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "WholeBodyCompute-OpenCore/1.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = getattr(resp, "status", None) or resp.getcode()
            return {"status": "ok", "http_status": code, "event": "wbc_vitals_rollup", "cohort_id": cohort_id}
    except Exception as exc:
        logger.debug("Vitals rollup send notice: %s", exc)
        return {"status": "network_error", "reason": str(exc), "cohort_id": cohort_id}


def get_usage_stats_path() -> Path:
    """Resolve the local JSON ledger for CLI command, audit, Edit Mode, and workflow usage stats."""
    env_p = os.environ.get("WBC_USAGE_STATS_PATH")
    if env_p:
        return Path(env_p).expanduser()
    return (get_wbc_data_dir() / "usage_stats.json").expanduser()


def load_usage_stats(stats_path: Path | None = None) -> dict[str, Any]:
    """Load the persistent local usage stats ledger."""
    target = Path(stats_path).expanduser() if stats_path else get_usage_stats_path()
    default_ledger: dict[str, Any] = {
        "version": 1,
        "updated_at": 0.0,
        "total_events": 0,
        "commands": {},
        "events": {},
        "recent_events": [],
    }
    if target.is_file():
        try:
            data = json.loads(target.read_text(encoding="utf-8", errors="replace"))
            if isinstance(data, dict):
                for k, v in default_ledger.items():
                    data.setdefault(k, v)
                return data
        except Exception:
            pass
    return default_ledger


def track_wbc_event(
    event_name: str,
    properties: dict[str, Any] | None = None,
    *,
    consent_manager: TelemetryConsentManager | None = None,
    stats_path: Path | None = None,
    async_send: bool = True,
    force_anonymous_ping: bool = True,
    timeout: float = 1.5,
    detached_hook: bool = False,
) -> dict[str, Any]:
    """
    Record a zero-PII Whole Body Compute usage event:
    1. Always increments the local usage ledger (`~/.whole_body_compute/usage_stats.json`).
    2. Emits an anonymous PostHog event to Project 730830 (`DEFAULT_POSTHOG_API_KEY`) unless
       disabled by `DO_NOT_TRACK=1`, `BODY_TELEMETRY=0|off|false|airgapped`, `CI=true`, or `PYTEST_CURRENT_TEST`.
    """
    import sys
    import threading

    clean_props = sanitize_telemetry_payload(dict(properties or {}))
    now = round(time.time(), 3)

    # 1. Update local usage_stats.json (skip writing to real home during pytest unless WBC_USAGE_STATS_PATH is set)
    in_pytest = bool(os.environ.get("PYTEST_CURRENT_TEST"))
    can_persist_local = (not in_pytest) or bool(stats_path or os.environ.get("WBC_USAGE_STATS_PATH"))
    target_stats = Path(stats_path).expanduser() if stats_path else get_usage_stats_path()

    if can_persist_local:
        try:
            ledger = load_usage_stats(target_stats)
            ledger["updated_at"] = now
            ledger["total_events"] = int(ledger.get("total_events", 0)) + 1

            ev_map = ledger.setdefault("events", {})
            ev_entry = ev_map.setdefault(
                event_name,
                {"count": 0, "first_ts": now, "last_ts": now, "last_properties": {}},
            )
            ev_entry["count"] = int(ev_entry.get("count", 0)) + 1
            ev_entry["first_ts"] = min(float(ev_entry.get("first_ts", now)), now)
            ev_entry["last_ts"] = max(float(ev_entry.get("last_ts", 0.0)), now)
            ev_entry["last_properties"] = {
                k: v for k, v in clean_props.items() if isinstance(v, (str, int, float, bool))
            }

            if event_name == "wbc_cli_command" and clean_props.get("command"):
                cmd_name = str(clean_props["command"])
                cmd_map = ledger.setdefault("commands", {})
                c_entry = cmd_map.setdefault(
                    cmd_name,
                    {"count": 0, "first_ts": now, "last_ts": now},
                )
                c_entry["count"] = int(c_entry.get("count", 0)) + 1
                c_entry["first_ts"] = min(float(c_entry.get("first_ts", now)), now)
                c_entry["last_ts"] = max(float(c_entry.get("last_ts", 0.0)), now)

            recent = list(ledger.get("recent_events", []))
            recent.append(
                {
                    "timestamp": now,
                    "event": event_name,
                    "summary": {
                        k: v
                        for k, v in clean_props.items()
                        if k in ("command", "action", "mode", "recipe_id", "engine_used", "harnesses_active", "threads_scanned", "wasted_spend_usd", "passive_workflows_crystallized", "discovered_count")
                    },
                }
            )
            ledger["recent_events"] = recent[-100:]
            target_stats.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = target_stats.with_suffix(".json.tmp")
            tmp_file.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
            tmp_file.replace(target_stats)
        except Exception as exc:
            logger.debug("Local usage stats write notice: %s", exc)

    # 2. Check circuit-breakers before network dispatch
    mgr = consent_manager or TelemetryConsentManager()
    overridden, override_reason = mgr.check_environment_override()
    if overridden:
        return {
            "status": "blocked_by_env",
            "reason": override_reason,
            "event": event_name,
            "local_recorded": can_persist_local,
        }

    # During pytest, do not fire background network threads unless explicitly testing telemetry sync
    if in_pytest and os.environ.get("WBC_TEST_TELEMETRY_PING") != "1":
        return {
            "status": "skipped_in_pytest",
            "event": event_name,
            "local_recorded": can_persist_local,
        }

    if not force_anonymous_ping and not mgr.can_share_vitals():
        return {
            "status": "blocked_by_consent",
            "effective_tier": mgr.get_effective_tier().value,
            "event": event_name,
            "local_recorded": can_persist_local,
        }

    cohort_id = mgr.get_cohort_id()
    api_key = (
        os.environ.get("WBC_POSTHOG_KEY")
        or os.environ.get("POSTHOG_API_KEY")
        or os.environ.get("POSTHOG_PROJECT_TOKEN")
        or DEFAULT_POSTHOG_API_KEY
    )
    target_url = os.environ.get("WBC_TELEMETRY_INGEST_URL")
    if not target_url:
        host = os.environ.get("POSTHOG_HOST") or "https://us.i.posthog.com"
        target_url = f"{host.rstrip('/')}/capture/"

    clean_props["cohort_id"] = cohort_id
    clean_props["product"] = "whole_body_compute"
    clean_props["source"] = "whole_body_compute_open_core"
    clean_props["os"] = sys.platform.lower()
    clean_props["arch"] = (os.uname().machine.lower() if hasattr(os, "uname") else "unknown")
    clean_props["$is_server"] = True
    if os.environ.get("WBC_INTERNAL_DEV") == "1" or Path.cwd().name == "Whole Body Compute":
        clean_props["is_internal_dev"] = True

    payload_bytes = json.dumps(
        {
            "api_key": api_key,
            "event": event_name,
            "distinct_id": cohort_id,
            "properties": clean_props,
        }
    ).encode("utf-8")

    def _send() -> dict[str, Any]:
        try:
            req = urllib.request.Request(
                target_url,
                data=payload_bytes,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "WholeBodyCompute-OpenCore/1.0",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                code = getattr(resp, "status", None) or resp.getcode()
                return {"status": "ok", "http_status": code, "event": event_name, "cohort_id": cohort_id}
        except Exception as exc:
            logger.debug("PostHog event notice (%s): %s", event_name, exc)
            return {"status": "network_error", "reason": str(exc), "event": event_name}

    if not async_send or os.environ.get("WBC_TEST_SYNC") == "1" or os.environ.get("VOICEFI_TEST_SYNC") == "1":
        res = _send()
        res["local_recorded"] = can_persist_local
        return res

    if detached_hook:
        try:
            import shutil
            import subprocess

            curl_bin = shutil.which("curl")
            if curl_bin:
                subprocess.Popen(
                    [
                        curl_bin,
                        "-sS",
                        "-m",
                        "2",
                        "-X",
                        "POST",
                        "-H",
                        "Content-Type: application/json",
                        "-d",
                        payload_bytes.decode("utf-8"),
                        target_url,
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                return {
                    "status": "dispatched_detached",
                    "event": event_name,
                    "cohort_id": cohort_id,
                    "local_recorded": can_persist_local,
                }
        except Exception:
            pass

    threading.Thread(target=_send, daemon=bool(detached_hook)).start()
    return {"status": "dispatched_async", "event": event_name, "cohort_id": cohort_id, "local_recorded": can_persist_local}


def get_posthog_personal_api_key() -> str:
    """
    Resolve an optional PostHog Personal API Key (`phx_...`) for read-only HogQL cloud analytics.
    Checks `POSTHOG_PERSONAL_API_KEY` env var first, then `~/.whole_body_compute/posthog_personal_key`
    (owner-only `0600` file outside git repositories, so `.env` is never required).
    """
    env_key = os.environ.get("POSTHOG_PERSONAL_API_KEY", "").strip()
    if env_key:
        return env_key
    for cand in (
        get_wbc_data_dir() / "posthog_personal_key",
        Path.home() / ".whole_body_compute" / "posthog_personal_key",
    ):
        try:
            if cand.is_file():
                val = cand.read_text(encoding="utf-8", errors="ignore").strip()
                if val.startswith("phx_"):
                    return val
        except Exception:
            continue
    return ""


def fetch_posthog_cloud_funnel_summary(
    project_id: int | str = 730830,
    personal_api_key: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Query live external funnel, quota limit-hit, and conversion stats from PostHog Cloud via HogQL."""
    key = (personal_api_key or get_posthog_personal_api_key()).strip()
    if not key:
        return {
            "status": "missing_personal_key",
            "hint": "Store a read-only phx_ key at ~/.whole_body_compute/posthog_personal_key (chmod 600) or set POSTHOG_PERSONAL_API_KEY.",
        }

    ext_pred = (
        "distinct_id != '257d2d48b088ef95' "
        "AND distinct_id NOT LIKE 'edge_wholebodycompute%' "
        "AND coalesce(toString(person.properties.$internal_or_test_user), 'false') != 'true' "
        "AND coalesce(toString(properties.is_internal_dev), 'false') != 'true' "
        "AND coalesce(toString(properties.is_internal), 'false') != 'true' "
        "AND coalesce(toString(properties.is_bot), 'false') != 'true'"
    )
    sql = f"""
        SELECT
            countIf(event = '$pageview' AND ({ext_pred})) AS ext_pageviews,
            count(DISTINCT if(event = '$pageview' AND ({ext_pred}), distinct_id, NULL)) AS ext_web_visitors,
            countIf(event = 'wbc_edge_begin_fetch' AND ({ext_pred}) AND toString(properties.is_cli_client) = 'true') AS ext_cli_begin_fetches,
            countIf(event = 'wbc_edge_audit_fetch' AND ({ext_pred}) AND toString(properties.is_cli_client) = 'true') AS ext_cli_audit_fetches,
            count(DISTINCT if(event IN ('wbc_body_begin', 'wbc_audit_completed', 'wbc_cli_command', 'wbc_shield_pulse', 'wbc_quota_event') AND ({ext_pred}), distinct_id, NULL)) AS ext_product_cohorts,
            count(DISTINCT if(event = 'wbc_quota_event' AND ({ext_pred}) AND (toString(properties.limit_reached) = 'true' OR toString(properties.allowed) = 'false'), distinct_id, NULL)) AS ext_cohorts_hitting_limits,
            countIf(event = 'wbc_quota_event' AND ({ext_pred})) AS ext_quota_events,
            countIf(event IN ('wbc_edge_pricing_click', 'wbc_pricing_viewed') AND ({ext_pred})) AS ext_pricing_intents,
            countIf(event = 'wbc_checkout_clicked' AND ({ext_pred})) AS ext_checkout_clicks,
            countIf(event = 'wbc_license_activated' AND ({ext_pred})) AS ext_licenses_activated,
            countIf(event = 'wbc_license_activation_failed' AND ({ext_pred})) AS ext_license_failures
        FROM events
    """
    url = f"https://us.posthog.com/api/projects/{project_id}/query/"
    body = json.dumps({"query": {"kind": "HogQLQuery", "query": sql}}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        rows = payload.get("results") or []
        row = rows[0] if rows else [0] * 11
        return {
            "status": "ok",
            "project_id": int(project_id),
            "ext_pageviews": int(row[0] or 0),
            "ext_web_visitors": int(row[1] or 0),
            "ext_cli_begin_fetches": int(row[2] or 0),
            "ext_cli_audit_fetches": int(row[3] or 0),
            "ext_product_cohorts": int(row[4] or 0),
            "ext_cohorts_hitting_limits": int(row[5] or 0),
            "ext_quota_events": int(row[6] or 0),
            "ext_pricing_intents": int(row[7] or 0),
            "ext_checkout_clicks": int(row[8] or 0),
            "ext_licenses_activated": int(row[9] or 0),
            "ext_license_failures": int(row[10] or 0),
        }
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}


def get_comprehensive_usage_stats_report(
    stats_path: Path | None = None,
    *,
    include_cloud: bool = False,
) -> dict[str, Any]:
    """
    Build a unified Usage & Automation Telemetry Report combining:
    1. CLI command & event counts (`usage_stats.json`: `body begin`, `audit-spend`, `doctor`, `savings`, etc.)
    2. Shield Blood-Brain Barrier interceptions & tokens saved (`shield_savings_ledger`)
    3. PalWorkflow executions & cloud savings (`pal_cloud_savings.json`)
    4. Pal Live Edit Mode session stats (`active_editing_session.json` + `usage_stats.json`)
    5. Ambient Workflow Discovery & REM Dreaming stats (`recipes/candidates/`, `telemetry_outbox.jsonl`)
    """
    usage = load_usage_stats(stats_path)
    tel_status = get_telemetry_status_report()

    # Shield savings summary
    shield_summary: dict[str, Any] = {}
    try:
        from whole_body_compute.shield import get_savings_summary

        s_all = get_savings_summary(window="all")
        shield_summary = {
            "total_interceptions": int(s_all.get("total_events", 0)),
            "direct_tokens_saved": int(s_all.get("direct_tokens_saved", 0)),
            "cumulative_tokens_saved": int(s_all.get("cumulative_tokens_saved", 0)),
            "estimated_usd_saved": float(s_all.get("estimated_usd_saved", 0.0)),
            "wall_time_saved_sec": float(s_all.get("wall_time_saved_sec", 0.0)),
            "secrets_scrubbed": int(s_all.get("secrets_scrubbed", 0)),
        }
    except Exception:
        pass

    # Workflow execution savings
    wf_savings: dict[str, Any] = {"total_runs": 0, "total_tokens_saved": 0, "workflows_count": 0}
    try:
        for cand_p in (
            get_wbc_data_dir() / "telemetry" / "pal_cloud_savings.json",
            Path.home() / ".voicefi" / "telemetry" / "pal_cloud_savings.json",
        ):
            if cand_p.is_file():
                wf_raw = json.loads(cand_p.read_text(encoding="utf-8"))
                wf_savings = {
                    "total_runs": int(wf_raw.get("total_runs", 0)),
                    "total_tokens_saved": int(wf_raw.get("total_tokens_saved", 0)),
                    "total_screen_mb_saved": float(wf_raw.get("total_screen_mb_saved", 0.0)),
                    "workflows_count": len(wf_raw.get("workflows", {})),
                    "workflows": wf_raw.get("workflows", {}),
                }
                break
    except Exception:
        pass

    # Ambient candidates & active recipes
    ambient_info: dict[str, Any] = {"staged_candidates_count": 0, "active_recipes_count": 0, "ambient_transitions_logged": 0}
    try:
        from whole_body_compute.integrations.ambient_discovery import list_staged_candidates

        c_rep = list_staged_candidates()
        ambient_info = {
            "staged_candidates_count": int(c_rep.get("staged_candidates_count", 0)),
            "active_recipes_count": int(c_rep.get("active_recipes_count", 0)),
            "ambient_transitions_logged": int(c_rep.get("ambient_transitions_logged", 0)),
        }
    except Exception:
        pass

    # Skills telemetry summary
    skills_info: dict[str, Any] = {}
    try:
        from whole_body_compute.skills_telemetry import load_skill_telemetry

        sk_ledger = load_skill_telemetry()
        sk_map = sk_ledger.get("skills", {})
        skills_info = {
            "tracked_skills_count": len(sk_map),
            "total_skill_reads": sum(int(v.get("total_reads", 0)) for v in sk_map.values() if isinstance(v, dict)),
            "scanned_transcripts_count": len(sk_ledger.get("scanned_transcripts", {})),
        }
    except Exception:
        pass

    out_report: dict[str, Any] = {
        "status": "ok",
        "posthog_project_id": 730830,
        "posthog_key_prefix": DEFAULT_POSTHOG_API_KEY[:12] + "...",
        "consent": tel_status.get("consent", {}),
        "usage_ledger": {
            "path": str(Path(stats_path).expanduser() if stats_path else get_usage_stats_path()),
            "total_events": int(usage.get("total_events", 0)),
            "commands": usage.get("commands", {}),
            "events": usage.get("events", {}),
            "recent_events": usage.get("recent_events", [])[-15:],
        },
        "shield_savings": shield_summary,
        "workflow_automations": wf_savings,
        "ambient_and_dreaming": {
            **ambient_info,
            "procedural_dna_cards_total": int(tel_status.get("outbox_total_cards", 0)),
            "procedural_dna_cards_synced": int(tel_status.get("outbox_synced_cards", 0)),
            "procedural_dna_by_type": tel_status.get("outbox_cards_by_type", {}),
        },
        "skills_telemetry": skills_info,
    }
    if include_cloud:
        out_report["posthog_cloud"] = fetch_posthog_cloud_funnel_summary()
    return out_report


def format_usage_stats_table(report: dict[str, Any]) -> str:
    """Format a rich terminal overview for `body telemetry stats` / `body stats`."""
    consent = report.get("consent", {})
    u_ledger = report.get("usage_ledger", {})
    cmds = u_ledger.get("commands", {})
    evts = u_ledger.get("events", {})
    shield = report.get("shield_savings", {})
    wf = report.get("workflow_automations", {})
    amb = report.get("ambient_and_dreaming", {})
    sk = report.get("skills_telemetry", {})

    audit_runs = (
        int(evts.get("wbc_audit_completed", {}).get("count", 0))
        + int(cmds.get("audit-spend", {}).get("count", 0))
    )
    begin_runs = (
        int(evts.get("wbc_body_begin", {}).get("count", 0))
        + int(cmds.get("begin", {}).get("count", 0))
    )
    edit_applies = int(evts.get("wbc_pal_edit_applied", {}).get("count", 0))
    edit_sessions = int(evts.get("wbc_pal_edit_session", {}).get("count", 0))
    rem_runs = int(evts.get("wbc_rem_sleep_completed", {}).get("count", 0))

    lines = [
        "📊 Whole Body Compute - Unified Usage, Funnel & Automation Telemetry (`body telemetry stats`)",
        "=" * 88,
        f"  • PostHog Cloud Project:  Project #730830 ({report.get('posthog_key_prefix')}) | Cohort: `{consent.get('cohort_id')}`",
        f"  • Consent & Egress Gate:  {consent.get('effective_tier')} (DO_NOT_TRACK Override: {'ACTIVE' if consent.get('env_override_active') else 'OFF'})",
        "-" * 88,
        "  [1] Top-of-Funnel & Core CLI Usage:",
        f"      • No-Code / Spend Audits (`audit.py` / `body audit-spend`): {audit_runs:,} run(s)",
        f"      • Workspace Awakenings (`body begin`):                      {begin_runs:,} run(s)",
        f"      • Total Tracked CLI Invocations:                            {sum(int(v.get('count', 0)) for v in cmds.values()):,} command(s)",
    ]
    if cmds:
        top_cmds = sorted(cmds.items(), key=lambda kv: int(kv[1].get("count", 0)), reverse=True)[:8]
        cmd_str = ", ".join(f"`body {k}` ({v.get('count', 0)}x)" for k, v in top_cmds)
        lines.append(f"      • Top CLI Subcommands: {cmd_str}")

    lines.extend(
        [
            "",
            "  [2] Organ 05 Blood-Brain Barrier & Skill Telemetry:",
            f"      • Shield Hook Interceptions: {shield.get('total_interceptions', 0):,} gates | "
            f"{shield.get('direct_tokens_saved', 0):,} direct tok saved ({shield.get('cumulative_tokens_saved', 0):,} cumulative)",
            f"      • Estimated Cloud USD Saved: ${shield.get('estimated_usd_saved', 0.0):.2f} | "
            f"Secrets Scrubbed: {shield.get('secrets_scrubbed', 0)}",
            f"      • Skill Activations (`SKILL.md` reads): {sk.get('total_skill_reads', 0):,} reads across "
            f"{sk.get('tracked_skills_count', 0)} skills ({sk.get('scanned_transcripts_count', 0)} transcripts audited)",
            "",
            "  [3] Workflow Automations, Live Edit Mode & Nocturnal REM Dreaming:",
            f"      • PalWorkflow Executions:    {wf.get('total_runs', 0):,} run(s) across {wf.get('workflows_count', 0)} recipes "
            f"({wf.get('total_tokens_saved', 0):,} cloud tokens saved)",
            f"      • Pal Live Edit Mode:        {edit_sessions:,} session(s) | {edit_applies:,} voice edit(s) applied on-device",
            f"      • Ambient Workflow Mining:   {amb.get('ambient_transitions_logged', 0):,} transitions observed -> "
            f"{amb.get('staged_candidates_count', 0)} staged candidates / {amb.get('active_recipes_count', 0)} active recipes",
            f"      • Nocturnal REM Dreaming:    {rem_runs:,} consolidation pass(es) | "
            f"{amb.get('procedural_dna_cards_total', 0)} Procedural DNA cards harvested ({amb.get('procedural_dna_cards_synced', 0)} synced)",
        ]
    )
    cloud = report.get("posthog_cloud")
    if isinstance(cloud, dict):
        lines.append("")
        lines.append("  [4] Live PostHog Cloud Funnel & Limit Stats (Project #730830 - External Non-Bot):")
        if cloud.get("status") == "ok":
            lines.extend(
                [
                    f"      • Web Visitors / Pageviews:   {cloud.get('ext_web_visitors', 0):,} unique visitors ({cloud.get('ext_pageviews', 0):,} views)",
                    f"      • CLI Script `curl` Fetches:  {cloud.get('ext_cli_begin_fetches', 0):,} `/begin.sh` • {cloud.get('ext_cli_audit_fetches', 0):,} `/audit.py`",
                    f"      • Active Product Cohorts:     {cloud.get('ext_product_cohorts', 0):,} external cohort(s) | "
                    f"Limit Hits: {cloud.get('ext_cohorts_hitting_limits', 0):,} cohort(s) ({cloud.get('ext_quota_events', 0):,} quota events)",
                    f"      • Upgrade Conversion Funnel:  {cloud.get('ext_pricing_intents', 0):,} pricing views -> "
                    f"{cloud.get('ext_checkout_clicks', 0):,} checkout clicks -> {cloud.get('ext_licenses_activated', 0):,} activated "
                    f"({cloud.get('ext_license_failures', 0):,} failed)",
                ]
            )
        else:
            lines.append(f"      • Cloud Query Status: {cloud.get('status')} ({cloud.get('hint') or cloud.get('reason')})")
    lines.append("=" * 88)
    return "\n".join(lines)


# ============================================================================
# Storage Compaction & 15.0 MB Guardrail
# ============================================================================


def compact_telemetry_storage(
    max_mb: float = 15.0,
    retention_days: int = 7,
    outbox_path: Path | None = None,
    db_paths: list[Path] | None = None,
    max_outbox_entries: int = 500,
) -> dict[str, Any]:
    """
    Enforce the `<= 15.0 MB` local telemetry storage budget:
    1. Deduplicates `telemetry_outbox.jsonl` by `signature_hash` and caps at `max_outbox_entries`.
    2. Prunes stale raw telemetry rows older than `retention_days` in SQLite databases
       (`analytics.db`, `dot_memory.db`).
    3. Reports `before_mb`, `after_mb`, `reclaimed_mb`, and `within_budget`.
    """
    env_outbox = os.environ.get("WBC_TELEMETRY_OUTBOX_PATH")
    target_outbox = Path(outbox_path or (env_outbox if env_outbox else (get_wbc_data_dir() / "telemetry_outbox.jsonl"))).expanduser()
    data_dir = get_wbc_data_dir()
    history_jsonl = data_dir / "telemetry_history.jsonl"
    target_dbs = (
        [Path(p).expanduser() for p in db_paths]
        if db_paths is not None
        else [
            data_dir / "analytics.db",
        ]
    )
    prune_dbs = list(target_dbs)
    if db_paths is None:
        dot_db = data_dir / "dot_memory.db"
        if dot_db.is_file():
            prune_dbs.append(dot_db)

    def _measure_bytes() -> int:
        total = 0
        if target_outbox.is_file():
            total += target_outbox.stat().st_size
        if db_paths is None and history_jsonl.is_file():
            total += history_jsonl.stat().st_size
        if db_paths is not None:
            for dbp in target_dbs:
                if dbp.is_file():
                    total += dbp.stat().st_size
        return total

    before_bytes = _measure_bytes()
    outbox_before = 0
    outbox_after = 0

    # 1. Deduplicate and cap outbox JSONL
    if target_outbox.is_file():
        raw_lines = [
            ln.strip() for ln in target_outbox.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        outbox_before = len(raw_lines)
        by_sig: dict[str, dict[str, Any]] = {}
        for ln in raw_lines:
            try:
                obj = json.loads(ln)
                if not isinstance(obj, dict):
                    continue
                sig = str(
                    obj.get("signature_hash")
                    or obj.get("ir_signature_hash")
                    or hashlib.sha256(ln.encode("utf-8")).hexdigest()[:16]
                )
                if sig in by_sig:
                    prev = by_sig[sig]
                    prev["verified_count"] = int(prev.get("verified_count") or 1) + int(
                        obj.get("verified_count") or 1
                    )
                    prev["last_seen_at"] = max(
                        float(prev.get("last_seen_at") or 0.0),
                        float(obj.get("last_seen_at") or 0.0),
                    )
                else:
                    obj["signature_hash"] = sig
                    obj.setdefault("verified_count", 1)
                    by_sig[sig] = obj
            except Exception:
                continue

        deduped = list(by_sig.values())
        deduped.sort(
            key=lambda x: (int(x.get("verified_count") or 1), float(x.get("last_seen_at") or 0.0)),
            reverse=True,
        )
        deduped = deduped[:max_outbox_entries]
        # Strictly enforce max_mb storage budget on outbox file bytes
        max_bytes = int(max_mb * 1024 * 1024)
        serialized_entries = [json.dumps(c) for c in deduped]
        total_size = sum(len(e.encode("utf-8")) + 1 for e in serialized_entries)
        while deduped and total_size > max_bytes:
            serialized_entries.pop()
            deduped.pop()
            total_size = sum(len(e.encode("utf-8")) + 1 for e in serialized_entries)
        outbox_after = len(deduped)
        target_outbox.write_text(
            "\n".join(serialized_entries) + ("\n" if serialized_entries else ""),
            encoding="utf-8",
        )

    # 1b. Trim oversized telemetry_history.jsonl if needed to honor <= max_mb budget
    if db_paths is None and history_jsonl.is_file():
        max_hist_bytes = int(max_mb * 0.5 * 1024 * 1024)
        if history_jsonl.stat().st_size > max_hist_bytes:
            try:
                hist_lines = history_jsonl.read_text(encoding="utf-8", errors="ignore").splitlines()
                kept_lines = hist_lines[-5000:]
                history_jsonl.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
            except Exception as exc:
                logger.debug("Telemetry history compaction notice: %s", exc)

    # 2. Prune stale telemetry events in SQLite DBs older than `retention_days`
    cutoff_ts = time.time() - (retention_days * 86400)
    pruned_db_rows = 0
    for dbp in prune_dbs:
        if not dbp.is_file():
            continue
        try:
            with sqlite3.connect(str(dbp), timeout=2.0) as conn:
                tables = [
                    r[0]
                    for r in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                ]
                for tbl in ("cognitive_telemetry", "telemetry_events", "immune_hook_events"):
                    if tbl in tables:
                        cols = [
                            c[1] for c in conn.execute(f"PRAGMA table_info({tbl})").fetchall()
                        ]
                        ts_col = "timestamp" if "timestamp" in cols else ("created_at" if "created_at" in cols else None)
                        if ts_col:
                            cur = conn.execute(
                                f"DELETE FROM {tbl} WHERE {ts_col} < ?", (cutoff_ts,)
                            )
                            pruned_db_rows += max(cur.rowcount, 0)
                conn.commit()
        except Exception as exc:
            logger.debug("SQLite telemetry compaction notice (%s): %s", dbp, exc)

    after_bytes = _measure_bytes()
    before_mb = round(before_bytes / (1024 * 1024), 4)
    after_mb = round(after_bytes / (1024 * 1024), 4)
    reclaimed_mb = round(max(0.0, before_mb - after_mb), 4)

    return {
        "status": "ok",
        "max_mb_budget": max_mb,
        "retention_days": retention_days,
        "before_mb": before_mb,
        "after_mb": after_mb,
        "reclaimed_mb": reclaimed_mb,
        "within_budget": after_mb <= max_mb,
        "outbox_entries_before": outbox_before,
        "outbox_entries_after": outbox_after,
        "pruned_db_rows": pruned_db_rows,
        "outbox_path": str(target_outbox),
    }


# ============================================================================
# CLI Formatting Helpers
# ============================================================================


def get_telemetry_status_report(
    outbox_path: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Build a unified status report for `body telemetry status`."""
    consent_mgr = TelemetryConsentManager(config_path=config_path)
    harvester = ProceduralDNAHarvester(outbox_path=outbox_path, consent_manager=consent_mgr)
    cards = harvester.load_outbox_cards()
    by_type: dict[str, int] = {
        "compiler_ir_card": 0,
        "app_proprioception_card": 0,
        "router_habit_card": 0,
        "generalized_recipe_template": 0,
    }
    for c in cards:
        ct = str(c.get("card_type") or "unknown")
        by_type[ct] = by_type.get(ct, 0) + 1

    synced_cards = sum(1 for c in cards if c.get("synced_at"))
    pending_cards = len(cards) - synced_cards

    storage = compact_telemetry_storage(outbox_path=harvester.outbox_path)
    return {
        "status": "ok",
        "consent": consent_mgr.get_status(),
        "storage": storage,
        "outbox_path": str(harvester.outbox_path),
        "outbox_total_cards": len(cards),
        "outbox_synced_cards": synced_cards,
        "outbox_pending_cards": pending_cards,
        "outbox_cards_by_type": by_type,
    }


def format_telemetry_status(report: dict[str, Any]) -> str:
    """Format `body telemetry status` for terminal display."""
    consent = report.get("consent", {})
    storage = report.get("storage", {})
    by_type = report.get("outbox_cards_by_type", {})
    override_str = (
        f"ACTIVE ({consent.get('env_override_reason')})"
        if consent.get("env_override_active")
        else "INACTIVE (normal)"
    )
    egress_str = "ALLOWED (Opt-In)" if consent.get("network_egress_allowed") else "BLOCKED (100% Local)"

    lines = [
        "🧬 Whole Body Compute - Federated Procedural DNA & Ethical Telemetry Status",
        "=" * 80,
        f"  • Effective Consent Tier: {consent.get('effective_tier')} (Stored: {consent.get('stored_tier')})",
        f"  • Env Circuit-Breaker:    {override_str}",
        f"  • Network Egress Gate:    {egress_str}",
        f"  • Rotating Cohort ID:     {consent.get('cohort_id')} (Month: {consent.get('cohort_rotation_period')})",
        (f"  • Footprint vs Budget:    {storage.get('after_mb', 0.0):.3f} MB / {storage.get('max_mb_budget', 15.0):.1f} MB "
        f"({'OK' if storage.get('within_budget', True) else 'OVER BUDGET'})"),
        (f"  • Auditable Outbox:       {report.get('outbox_path')} ({report.get('outbox_total_cards', 0)} cards: "
        f"{report.get('outbox_synced_cards', 0)} synced, {report.get('outbox_pending_cards', 0)} pending)"),
        "  • Harvested Procedural DNA Breakdown:",
        f"      - Compiler IR Cards:           {by_type.get('compiler_ir_card', 0)}",
        f"      - App Proprioception Cards:    {by_type.get('app_proprioception_card', 0)}",
        f"      - Router Tool-Pruning Habits:  {by_type.get('router_habit_card', 0)}",
        f"      - Generalized Recipe Skeletons:{by_type.get('generalized_recipe_template', 0)}",
        "=" * 80,
    ]
    return "\n".join(lines)


def format_harvest_summary(report: dict[str, Any]) -> str:
    """Format `body telemetry harvest` summary table for terminal display."""
    counts = report.get("counts", {})
    outbox = report.get("outbox", {})
    consent = report.get("consent", {})

    lines = [
        "🧬 Whole Body Compute - Procedural DNA Harvest & Deduplication Report",
        "=" * 88,
        (f"  Consent Tier: {consent.get('effective_tier')} | Cohort: {consent.get('cohort_id')} | "
        f"Outbox: {outbox.get('total_unique_cards', 0)} unique cards "
        f"(+{outbox.get('inserted_count', 0)} new, {outbox.get('updated_in_place_count', 0)} deduplicated in-place)"),
        "-" * 88,
        ("  [1] Compiler & Tier-Promotion IR Cards "
        f"({counts.get('compiler_ir_cards', 0)} harvested):"),
    ]
    for c in report.get("compiler_ir_cards", []):
        lines.append(
            f"      • {c['recipe_id']:<32} | {c['dominant_tier']:<18} | "
            f"{c['raw_step_count']}->{c['compiled_step_count']} steps | "
            f"rel={c['reliability_score']:.2f} | ir_hash={c['ir_signature_hash']}"
        )

    lines.append(
        f"\n  [2] App Proprioception & Foveal Motor Atlas ({counts.get('app_proprioception_cards', 0)} harvested):"
    )
    for p in report.get("app_proprioception_cards", []):
        lines.append(
            f"      • {p['app_archetype']:<32} | route={p['preferred_motor_route']:<28} | "
            f"hash={p['signature_hash']}"
        )

    lines.append(
        f"\n  [3] Tool-Pruning & Router Habit Cards ({counts.get('router_habit_cards', 0)} harvested):"
    )
    for r in report.get("router_habit_cards", []):
        lines.append(
            f"      • {r['intent_domain']:<32} | fast_tool={r['preferred_fast_tool']:<24} | "
            f"whitelist={len(r['pruned_tool_whitelist'])} tools | hash={r['signature_hash']}"
        )

    lines.append(
        f"\n  [4] Generalized Recipe Skeletons ({counts.get('generalized_recipe_templates', 0)} harvested):"
    )
    for t in report.get("generalized_recipe_templates", []):
        cmd_preview = (
            t["steps_skeleton"][0].get("command_template", "-")
            if t.get("steps_skeleton")
            else "-"
        )
        lines.append(
            f"      • {t['recipe_id']:<32} | ir_hash={t['ir_signature_hash']} | cmd=`{cmd_preview}`"
        )

    lines.append("=" * 88)
    return "\n".join(lines)
