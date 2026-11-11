"""
Procedural DNA Hub & Cryptographically Signed `PalWorkflow` Recipe Registry (`body pull`, `body search`, `body publish`, `body hub`).

Enforces 4-layer defense-in-depth before any Hub recipe can install or execute on `OrganEyes` / `Hands`:
  1. Open `PalWorkflow` + `ProceduralDNA` v1 Schema Validation (`schemas/pal_workflow_v1.json`)
  2. Zero-PII & Blood-Brain Barrier `VascularTrafficMonitor` Shell Safety Audit
  3. Dual SHA-256 Digest Binding (`canonical_payload_sha256` + `hoare_contract_sha256`)
  4. Domain-Separated RFC 8032 Ed25519 Signature Verification against Trusted Keyring
"""

from __future__ import annotations

import base64
import copy
import datetime
import hashlib
import json
import os
import re
import secrets
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from whole_body_compute import __version__
from whole_body_compute.organs.immune_vascular import VascularTrafficMonitor
from whole_body_compute.procedural_telemetry import (
    compute_ir_signature_hash,
    get_wbc_data_dir,
    sanitize_procedural_text,
)
from whole_body_compute.redaction import redact_secrets

DEFAULT_HUB_INDEX_URL = "https://hub.wholebodycompute.org/index.json"
FALLBACK_HUB_INDEX_URL = "https://wholebodycompute.org/hub/index.json"

RECIPE_SIGNING_DOMAIN = "WBC-PAL-WORKFLOW-V1"
INDEX_SIGNING_DOMAIN = "WBC-HUB-INDEX-V1"

WBC_OFFICIAL_KEY_ID = "wbc-hub-official-2026-ed25519"
WBC_OFFICIAL_HUB_PUBLIC_KEY_B64 = "BjQUeE3tG5i4omCmTBzjPeFfAFTRwbHxOgveAy1QWxA="

TRUSTED_BUILTIN_PUBLIC_KEYS: dict[str, str] = {
    WBC_OFFICIAL_KEY_ID: WBC_OFFICIAL_HUB_PUBLIC_KEY_B64,
}


class RecipeVerificationError(ValueError):
    """Raised when a PalWorkflow recipe fails schema, Zero-PII, SHA-256, or Ed25519 verification."""


class RecipeCompatibilityError(RuntimeError):
    """Raised when a PalWorkflow recipe's compatibility envelope excludes the current OS or runtime."""


# ==============================================================================
# 1. RFC 8032 Ed25519 Engine (`cryptography` fast path + pure-stdlib fallback)
# ==============================================================================

_ED25519_B = 256
_ED25519_Q = 2**255 - 19
_ED25519_L = 2**252 + 27742317777372353535851937790883648493


def _ed_inv(x: int) -> int:
    return pow(x, _ED25519_Q - 2, _ED25519_Q)


_ED25519_D = (-121665 * _ed_inv(121666)) % _ED25519_Q
_ED25519_I = pow(2, (_ED25519_Q - 1) // 4, _ED25519_Q)


def _ed_xrecover(y: int) -> int:
    xx = (y * y - 1) * _ed_inv(_ED25519_D * y * y + 1) % _ED25519_Q
    x = pow(xx, (_ED25519_Q + 3) // 8, _ED25519_Q)
    if (x * x - xx) % _ED25519_Q != 0:
        x = (x * _ED25519_I) % _ED25519_Q
    if x % 2 != 0:
        x = _ED25519_Q - x
    return x


_ED_BY = (4 * _ed_inv(5)) % _ED25519_Q
_ED_BX = _ed_xrecover(_ED_BY)
_ED_BASE = (_ED_BX % _ED25519_Q, _ED_BY % _ED25519_Q, 1, (_ED_BX * _ED_BY) % _ED25519_Q)
_ED_IDENT = (0, 1, 1, 0)


def _ed_add(
    p: tuple[int, int, int, int], q: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    x1, y1, z1, t1 = p
    x2, y2, z2, t2 = q
    a = ((y1 - x1) * (y2 - x2)) % _ED25519_Q
    b = ((y1 + x1) * (y2 + x2)) % _ED25519_Q
    c = (t1 * 2 * _ED25519_D * t2) % _ED25519_Q
    d = (z1 * 2 * z2) % _ED25519_Q
    e = (b - a) % _ED25519_Q
    f = (d - c) % _ED25519_Q
    g = (d + c) % _ED25519_Q
    h = (b + a) % _ED25519_Q
    return (
        (e * f) % _ED25519_Q,
        (g * h) % _ED25519_Q,
        (f * g) % _ED25519_Q,
        (e * h) % _ED25519_Q,
    )


def _ed_scalarmult(p: tuple[int, int, int, int], e: int) -> tuple[int, int, int, int]:
    res = _ED_IDENT
    cur = p
    scalar = e
    while scalar > 0:
        if scalar & 1:
            res = _ed_add(res, cur)
        cur = _ed_add(cur, cur)
        scalar >>= 1
    return res


def _ed_encodepoint(p: tuple[int, int, int, int]) -> bytes:
    x, y, z, _ = p
    zi = _ed_inv(z)
    x_aff = (x * zi) % _ED25519_Q
    y_aff = (y * zi) % _ED25519_Q
    encoded = bytearray(y_aff.to_bytes(32, "little"))
    encoded[31] |= (x_aff & 1) << 7
    return bytes(encoded)


def _ed_decodepoint(b: bytes) -> tuple[int, int, int, int]:
    if len(b) != 32:
        raise ValueError("Invalid Ed25519 public key length")
    y = int.from_bytes(b, "little") & ((1 << 255) - 1)
    if y >= _ED25519_Q:
        raise ValueError("Non-canonical Ed25519 point y >= q")
    x_sign = (b[31] >> 7) & 1
    xx = (y * y - 1) * _ed_inv(_ED25519_D * y * y + 1) % _ED25519_Q
    x = pow(xx, (_ED25519_Q + 3) // 8, _ED25519_Q)
    if (x * x - xx) % _ED25519_Q != 0:
        x = (x * _ED25519_I) % _ED25519_Q
    if (x * x - xx) % _ED25519_Q != 0:
        raise ValueError("Point is not on Ed25519 curve")
    if x == 0 and x_sign == 1:
        raise ValueError("Invalid negative zero x coordinate in Ed25519 point")
    if (x & 1) != x_sign:
        x = _ED25519_Q - x
    return (x, y, 1, (x * y) % _ED25519_Q)


def _pure_ed25519_public_from_seed(seed: bytes) -> bytes:
    if len(seed) != 32:
        raise ValueError("Ed25519 private seed must be 32 bytes")
    h = hashlib.sha512(seed).digest()
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    return _ed_encodepoint(_ed_scalarmult(_ED_BASE, a))


def _pure_ed25519_sign(seed: bytes, message: bytes) -> bytes:
    if len(seed) != 32:
        raise ValueError("Ed25519 private seed must be 32 bytes")
    h = hashlib.sha512(seed).digest()
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    prefix = h[32:64]
    pub = _ed_encodepoint(_ed_scalarmult(_ED_BASE, a))
    r = int.from_bytes(hashlib.sha512(prefix + message).digest(), "little") % _ED25519_L
    r_bytes = _ed_encodepoint(_ed_scalarmult(_ED_BASE, r))
    k = int.from_bytes(hashlib.sha512(r_bytes + pub + message).digest(), "little") % _ED25519_L
    s = (r + k * a) % _ED25519_L
    return r_bytes + s.to_bytes(32, "little")


def _pure_ed25519_verify(public_key: bytes, message: bytes, signature: bytes) -> bool:
    if len(public_key) != 32 or len(signature) != 64:
        return False
    r_bytes = signature[:32]
    s = int.from_bytes(signature[32:64], "little")
    if s >= _ED25519_L:
        return False
    try:
        a_pt = _ed_decodepoint(public_key)
        r_pt = _ed_decodepoint(r_bytes)
    except ValueError:
        return False
    k = int.from_bytes(hashlib.sha512(r_bytes + public_key + message).digest(), "little") % _ED25519_L
    # Cofactor-multiplied verification equation: [8][S]B == [8]R + [8][k]A (RFC 8032 §5.1.7)
    lhs = _ed_scalarmult(_ED_BASE, 8 * s)
    rhs = _ed_scalarmult(_ed_add(r_pt, _ed_scalarmult(a_pt, k)), 8)
    return _ed_encodepoint(lhs) == _ed_encodepoint(rhs)


def ed25519_public_from_seed(seed: bytes, *, use_pure_python: bool = False) -> bytes:
    """Derive 32-byte RFC 8032 Ed25519 public key from 32-byte private seed."""
    if len(seed) != 32:
        raise ValueError("Ed25519 seed must be 32 bytes")
    if not use_pure_python:
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )

            priv = Ed25519PrivateKey.from_private_bytes(seed)
            return priv.public_key().public_bytes(
                serialization.Encoding.Raw, serialization.PublicFormat.Raw
            )
        except ImportError:
            pass
    return _pure_ed25519_public_from_seed(seed)


def ed25519_generate_keypair(*, use_pure_python: bool = False) -> tuple[bytes, bytes]:
    """Generate a fresh `(seed_32b, public_key_32b)` Ed25519 keypair."""
    seed = secrets.token_bytes(32)
    pub = ed25519_public_from_seed(seed, use_pure_python=use_pure_python)
    return seed, pub


def ed25519_sign(seed: bytes, message: bytes, *, use_pure_python: bool = False) -> bytes:
    """Sign `message` using 32-byte Ed25519 `seed`, returning a 64-byte signature."""
    if len(seed) != 32:
        raise ValueError("Ed25519 seed must be 32 bytes")
    if not use_pure_python:
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )

            priv = Ed25519PrivateKey.from_private_bytes(seed)
            return priv.sign(message)
        except ImportError:
            pass
    return _pure_ed25519_sign(seed, message)


def ed25519_verify(
    public_key: bytes, message: bytes, signature: bytes, *, use_pure_python: bool = False
) -> bool:
    """Verify a 64-byte Ed25519 `signature` over `message` against 32-byte `public_key`."""
    if len(public_key) != 32 or len(signature) != 64:
        return False
    if not use_pure_python:
        try:
            from cryptography.exceptions import InvalidSignature
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PublicKey,
            )

            pub = Ed25519PublicKey.from_public_bytes(public_key)
            try:
                pub.verify(signature, message)
                return True
            except InvalidSignature:
                return False
        except ImportError:
            pass
    return _pure_ed25519_verify(public_key, message, signature)


# ==============================================================================
# 2. Schema Loader, Canonical Digests & Zero-PII / Vascular Security Gate
# ==============================================================================

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")
_NAMESPACE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,38}$")
_SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?$")
_BUNDLE_ID_RE = re.compile(r"^[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+$")
_AX_ROLE_RE = re.compile(r"^AX[A-Z][A-Za-z0-9]*$")
_HEX16_RE = re.compile(r"^[0-9a-f]{16}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

_PERSONAL_PATH_RE = re.compile(
    r"(?:/Users/[A-Za-z0-9._-]+|/home/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+)"
)
_EMAIL_PII_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_SSN_PII_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_RAW_SECRET_RE = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{16,}"
    r"|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|rk_live_[A-Za-z0-9]{16,}|sk_live_[A-Za-z0-9]{16,})"
)
_RAW_PUBLIC_IPV4_RE = re.compile(
    r"\b(?!(?:127\.0\.0\.1|0\.0\.0\.0|10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.))"
    r"(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b"
)

ALLOWED_ACTIONS = frozenset(
    {
        "activate_app",
        "wait_for_window",
        "hotkey",
        "keystroke",
        "press_key",
        "click",
        "double_click",
        "right_click",
        "drag",
        "applescript",
        "run_shell",
        "wait",
        "system_action",
        "notify",
        "open_url",
    }
)

ALLOWED_GETTER_TYPES = frozenset(
    {
        "file_exists",
        "file_nonempty",
        "window_title_contains",
        "clipboard_matches",
        "browser_url_contains",
        "dom_selector_exists",
        "shell_exit_zero",
        "app_running",
        "ax_element_exists",
        "cli_installed",
    }
)

ALLOWED_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "namespace",
        "id",
        "name",
        "version",
        "description",
        "license",
        "author",
        "tags",
        "notes",
        "triggers",
        "destructive",
        "timeout_total_ms",
        "app_compatibility",
        "parameters",
        "preconditions",
        "steps",
        "postconditions",
        "success_summary",
        "error_summary",
        "tier_summary",
        "contract_summary",
        "procedural_dna",
        "signature",
    }
)


def get_schema_path() -> Path:
    """Locate `pal_workflow_v1.json` across package data or repository root."""
    pkg_schema = Path(__file__).resolve().parent / "schemas" / "pal_workflow_v1.json"
    if pkg_schema.is_file():
        return pkg_schema
    repo_schema = Path(__file__).resolve().parents[2] / "schemas" / "pal_workflow_v1.json"
    if repo_schema.is_file():
        return repo_schema
    raise FileNotFoundError("Could not locate schemas/pal_workflow_v1.json")


def load_pal_workflow_v1_schema() -> dict[str, Any]:
    """Load the Open `PalWorkflow` + `ProceduralDNA` v1 JSON Schema."""
    return json.loads(get_schema_path().read_text(encoding="utf-8"))


def canonical_json_bytes(obj: Any) -> bytes:
    """Deterministic sorted-key UTF-8 JSON bytes for SHA-256 and Ed25519 signing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def compute_canonical_payload_sha256(recipe_dict: dict[str, Any]) -> str:
    """Compute SHA-256 hex digest of the unsigned recipe dictionary (excluding `signature`)."""
    unsigned = {k: v for k, v in recipe_dict.items() if k != "signature"}
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def compute_hoare_contract_sha256(recipe_dict: dict[str, Any]) -> str:
    """
    Compute dedicated SHA-256 hex digest binding executable steps, target app bundle IDs,
    semantic `AXRole` selectors, and Hoare preconditions/postconditions/contracts.
    """
    hoare_core = {
        "namespace": recipe_dict.get("namespace", ""),
        "id": recipe_dict.get("id", ""),
        "version": recipe_dict.get("version", ""),
        "destructive": bool(recipe_dict.get("destructive", False)),
        "app_compatibility": recipe_dict.get("app_compatibility") or {},
        "preconditions": recipe_dict.get("preconditions") or [],
        "steps": recipe_dict.get("steps") or [],
        "postconditions": recipe_dict.get("postconditions") or [],
        "contract_summary": recipe_dict.get("contract_summary") or {},
    }
    return hashlib.sha256(canonical_json_bytes(hoare_core)).hexdigest()


def build_recipe_signing_message(
    canonical_payload_sha256: str, hoare_contract_sha256: str
) -> bytes:
    """Construct domain-separated signing message for a `PalWorkflow` recipe."""
    return (
        f"{RECIPE_SIGNING_DOMAIN}\n{canonical_payload_sha256}\n{hoare_contract_sha256}"
    ).encode()


def compute_index_sha256(index_dict: dict[str, Any]) -> str:
    """Compute SHA-256 hex digest of the unsigned Hub `index.json` payload."""
    unsigned = {k: v for k, v in index_dict.items() if k != "signature"}
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def _parse_semver(ver: str) -> tuple[int, int, int]:
    clean = str(ver or "0.0.0").strip().split("-", 1)[0]
    parts = clean.split(".")
    if len(parts) < 3:
        parts = (parts + ["0", "0", "0"])[:3]
    return int(parts[0]), int(parts[1]), int(parts[2])


def validate_pal_workflow_v1_dict(
    recipe_dict: dict[str, Any], *, require_signature: bool = False
) -> list[str]:
    """
    Deterministically validate a recipe dict against `schemas/pal_workflow_v1.json`.
    Returns a list of human-readable schema violation strings (empty when valid).
    """
    errors: list[str] = []
    if not isinstance(recipe_dict, dict):
        return ["Recipe root must be a mapping/object"]

    unknown_top = set(recipe_dict.keys()) - ALLOWED_TOP_LEVEL_KEYS
    if unknown_top:
        errors.append(f"Unsupported top-level fields: {sorted(unknown_top)}")

    required = (
        "schema_version",
        "namespace",
        "id",
        "name",
        "version",
        "description",
        "app_compatibility",
        "steps",
        "contract_summary",
        "procedural_dna",
    )
    for req in required:
        if req not in recipe_dict or recipe_dict[req] is None:
            errors.append(f"Missing required field '{req}'")

    if errors:
        return errors

    if str(recipe_dict.get("schema_version")) not in ("1.0", "1.0.0"):
        errors.append(f"Invalid schema_version: {recipe_dict.get('schema_version')!r} (expected '1.0')")

    ns = str(recipe_dict.get("namespace", ""))
    if not _NAMESPACE_RE.match(ns):
        errors.append(f"Invalid namespace '{ns}': must match {_NAMESPACE_RE.pattern}")

    rid = str(recipe_dict.get("id", ""))
    if not _SLUG_RE.match(rid):
        errors.append(f"Invalid recipe id '{rid}': must match {_SLUG_RE.pattern}")

    name = recipe_dict.get("name")
    if not isinstance(name, str) or not (3 <= len(name.strip()) <= 120):
        errors.append("Field 'name' must be a string between 3 and 120 chars")

    ver = str(recipe_dict.get("version", ""))
    if not _SEMVER_RE.match(ver):
        errors.append(f"Invalid semver version '{ver}'")

    desc = recipe_dict.get("description")
    if not isinstance(desc, str) or not (10 <= len(desc.strip()) <= 1000):
        errors.append("Field 'description' must be between 10 and 1000 chars")

    triggers = recipe_dict.get("triggers")
    if triggers is not None and (
        not isinstance(triggers, list)
        or len(triggers) == 0
        or not all(isinstance(t, str) and len(t.strip()) >= 2 for t in triggers)
    ):
        errors.append("Field 'triggers' must be a non-empty list of strings")

    # App Compatibility Envelope
    compat = recipe_dict.get("app_compatibility")
    if not isinstance(compat, dict):
        errors.append("Field 'app_compatibility' must be an object")
    else:
        platforms = compat.get("platforms")
        if not isinstance(platforms, list) or not platforms or any(p not in ("macos", "linux") for p in platforms):
            errors.append("app_compatibility.platforms must be a non-empty list containing 'macos' and/or 'linux'")
        min_wbc = str(compat.get("min_wbc_version", ""))
        if not _SEMVER_RE.match(min_wbc):
            errors.append(f"app_compatibility.min_wbc_version '{min_wbc}' must be valid semver")
        target_apps = compat.get("target_apps")
        if not isinstance(target_apps, list):
            errors.append("app_compatibility.target_apps must be a list")
        else:
            for idx, app in enumerate(target_apps):
                if not isinstance(app, dict):
                    errors.append(f"app_compatibility.target_apps[{idx}] must be an object")
                    continue
                b_id = str(app.get("bundle_id", ""))
                if not _BUNDLE_ID_RE.match(b_id):
                    errors.append(
                        f"app_compatibility.target_apps[{idx}].bundle_id '{b_id}' must be reverse-DNS (e.g. 'com.apple.Safari')"
                    )
                if not isinstance(app.get("app_name"), str) or not app["app_name"].strip():
                    errors.append(f"app_compatibility.target_apps[{idx}].app_name is required")

    # Preconditions / Postconditions helper
    def _validate_conditions(conds: Any, path_label: str) -> None:
        if conds is None:
            return
        if not isinstance(conds, list):
            errors.append(f"{path_label} must be a list")
            return
        for c_idx, cond in enumerate(conds):
            if not isinstance(cond, dict):
                errors.append(f"{path_label}[{c_idx}] must be an object")
                continue
            gt = cond.get("getter_type")
            if gt not in ALLOWED_GETTER_TYPES:
                errors.append(f"{path_label}[{c_idx}].getter_type '{gt}' is unsupported")
            tgt = cond.get("target")
            if not isinstance(tgt, str) or not tgt.strip():
                errors.append(f"{path_label}[{c_idx}].target must be a non-empty string")

    _validate_conditions(recipe_dict.get("preconditions"), "preconditions")
    _validate_conditions(recipe_dict.get("postconditions"), "postconditions")

    # Steps & Semantic AXRole validation
    steps = recipe_dict.get("steps")
    if not isinstance(steps, list) or len(steps) == 0:
        errors.append("Field 'steps' must be a non-empty list")
    else:
        for s_idx, step in enumerate(steps):
            if not isinstance(step, dict):
                errors.append(f"steps[{s_idx}] must be an object")
                continue
            act = step.get("action")
            if act not in ALLOWED_ACTIONS:
                errors.append(f"steps[{s_idx}].action '{act}' is not in ALLOWED_ACTIONS")
            if not isinstance(step.get("name"), str) or len(step["name"].strip()) < 2:
                errors.append(f"steps[{s_idx}].name must be at least 2 characters")
            if step.get("bundle_id") is not None and not _BUNDLE_ID_RE.match(str(step["bundle_id"])):
                errors.append(f"steps[{s_idx}].bundle_id '{step['bundle_id']}' must be reverse-DNS")

            ct = step.get("click_target")
            if ct is not None:
                if not isinstance(ct, dict):
                    errors.append(f"steps[{s_idx}].click_target must be an object")
                else:
                    ttype = ct.get("target_type")
                    if ttype not in ("menu_path", "ui_element", "dom_selector", "coordinates"):
                        errors.append(f"steps[{s_idx}].click_target.target_type '{ttype}' is invalid")
                    if ttype == "coordinates" or any(k in ct for k in ("x", "y", "coordinates", "click_xy")):
                        errors.append(
                            f"steps[{s_idx}].click_target uses fragile raw 'coordinates'; Hub v1 recipes require semantic 'ui_element' (AXRole), 'dom_selector', or 'menu_path'"
                        )
                    role = ct.get("role")
                    if ttype == "ui_element" and not role:
                        errors.append(f"steps[{s_idx}].click_target with target_type='ui_element' requires semantic AXRole 'role'")
                    if role is not None and not _AX_ROLE_RE.match(str(role)):
                        errors.append(
                            f"steps[{s_idx}].click_target.role '{role}' must be a valid macOS AXRole (e.g. 'AXButton', 'AXOutline')"
                        )
            _validate_conditions(step.get("postconditions"), f"steps[{s_idx}].postconditions")

    # Hoare Contract Summary
    csum = recipe_dict.get("contract_summary")
    if not isinstance(csum, dict):
        errors.append("Field 'contract_summary' must be an object")
    else:
        contracts = csum.get("contracts")
        if not isinstance(contracts, list) or len(contracts) == 0:
            errors.append("contract_summary.contracts must contain at least 1 Hoare contract {P} C {Q}")
        else:
            for h_idx, hc in enumerate(contracts):
                if not isinstance(hc, dict):
                    errors.append(f"contract_summary.contracts[{h_idx}] must be an object")
                    continue
                if not hc.get("name") or not isinstance(hc.get("name"), str):
                    errors.append(f"contract_summary.contracts[{h_idx}].name is required")
                pre = hc.get("preconditions")
                post = hc.get("postconditions")
                if not isinstance(pre, list) or len(pre) == 0:
                    errors.append(f"contract_summary.contracts[{h_idx}].preconditions must be non-empty")
                if not isinstance(post, list) or len(post) == 0:
                    errors.append(f"contract_summary.contracts[{h_idx}].postconditions must be non-empty")
                if not isinstance(hc.get("idempotency_strategy"), str) or len(hc["idempotency_strategy"].strip()) < 3:
                    errors.append(f"contract_summary.contracts[{h_idx}].idempotency_strategy is required")

    # Procedural DNA Envelope
    dna = recipe_dict.get("procedural_dna")
    if not isinstance(dna, dict):
        errors.append("Field 'procedural_dna' must be an object")
    else:
        if dna.get("dominant_tier") not in (
            "TIER_0_LOCAL",
            "TIER_1_HEADLESS_API",
            "TIER_1_5_BROWSER_DOM_JS",
            "TIER_2_SEMANTIC_UI",
        ):
            errors.append(f"procedural_dna.dominant_tier '{dna.get('dominant_tier')}' is invalid for Hub v1")
        ir_sig = str(dna.get("ir_signature_hash", ""))
        if not _HEX16_RE.match(ir_sig):
            errors.append(f"procedural_dna.ir_signature_hash '{ir_sig}' must be 16 lowercase hex chars")
        attest = dna.get("zero_pii_attestation")
        if not isinstance(attest, dict):
            errors.append("procedural_dna.zero_pii_attestation must be an object")
        else:
            if attest.get("sanitized") is not True:
                errors.append("procedural_dna.zero_pii_attestation.sanitized must be true")
            if attest.get("contains_credentials") is not False:
                errors.append("procedural_dna.zero_pii_attestation.contains_credentials must be false")
            if attest.get("contains_personal_paths") is not False:
                errors.append("procedural_dna.zero_pii_attestation.contains_personal_paths must be false")

    # Signature Envelope (when present or required)
    sig = recipe_dict.get("signature")
    if require_signature and not isinstance(sig, dict):
        errors.append("Missing required cryptographic 'signature' envelope")
    elif sig is not None:
        if not isinstance(sig, dict):
            errors.append("Field 'signature' must be an object")
        else:
            if sig.get("algorithm") != "ed25519+sha256":
                errors.append(f"Unsupported signature.algorithm: {sig.get('algorithm')!r}")
            for hex_field in ("canonical_payload_sha256", "hoare_contract_sha256"):
                if not _HEX64_RE.match(str(sig.get(hex_field, ""))):
                    errors.append(f"signature.{hex_field} must be a 64-char SHA-256 hex digest")
            for str_field in ("key_id", "public_key", "ed25519_signature", "signed_at"):
                if not isinstance(sig.get(str_field), str) or not sig[str_field].strip():
                    errors.append(f"signature.{str_field} must be a non-empty string")

    return errors


def _iter_strings_with_paths(node: Any, prefix: str = "") -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if isinstance(node, str):
        out.append((prefix, node))
    elif isinstance(node, dict):
        for k, v in node.items():
            if k == "signature":
                continue
            sub = f"{prefix}.{k}" if prefix else str(k)
            out.extend(_iter_strings_with_paths(v, sub))
    elif isinstance(node, list):
        for idx, item in enumerate(node):
            out.extend(_iter_strings_with_paths(item, f"{prefix}[{idx}]"))
    return out


def audit_zero_pii_and_shell_safety(recipe_dict: dict[str, Any]) -> list[str]:
    """
    Scan every string and shell command in `recipe_dict` for PII, personal paths,
    credentials, and Organ 05 `VascularTrafficMonitor` egress/injection vectors.
    """
    violations: list[str] = []

    for field_path, text_val in _iter_strings_with_paths(recipe_dict):
        if not text_val:
            continue
        m_path = _PERSONAL_PATH_RE.search(text_val)
        if m_path:
            violations.append(
                f"Personal filesystem path '{m_path.group(0)}' detected in {field_path} (use relative paths or ${{WORKSPACE_DIR}})"
            )
        if _RAW_SECRET_RE.search(text_val) or redact_secrets(text_val) != text_val:
            violations.append(f"Credential or secret token detected in {field_path}")
        m_email = _EMAIL_PII_RE.search(text_val)
        if m_email:
            violations.append(f"Email PII '{m_email.group(0)}' detected in {field_path}")
        if _SSN_PII_RE.search(text_val):
            violations.append(f"SSN pattern detected in {field_path}")
        m_ip = _RAW_PUBLIC_IPV4_RE.search(text_val)
        if m_ip:
            violations.append(f"Raw public IPv4 '{m_ip.group(0)}' detected in {field_path}")

    # Inspect all shell execution surfaces via Blood-Brain Barrier VascularTrafficMonitor
    for s_idx, step in enumerate(recipe_dict.get("steps") or []):
        if not isinstance(step, dict):
            continue
        cmd = step.get("command")
        if isinstance(cmd, str) and cmd.strip():
            vasc = VascularTrafficMonitor.inspect_shell_command(cmd, record_antigen_on_block=False)
            if not vasc.get("allowed", True):
                violations.append(
                    f"Blood-Brain Barrier VascularTrafficMonitor blocked steps[{s_idx}].command: {vasc.get('reason')}"
                )
        for p_idx, pc in enumerate(step.get("postconditions") or []):
            if isinstance(pc, dict) and pc.get("getter_type") == "shell_exit_zero":
                vasc = VascularTrafficMonitor.inspect_shell_command(
                    str(pc.get("target") or ""), record_antigen_on_block=False
                )
                if not vasc.get("allowed", True):
                    violations.append(
                        f"Blood-Brain Barrier VascularTrafficMonitor blocked steps[{s_idx}].postconditions[{p_idx}]: {vasc.get('reason')}"
                    )

    for cond_key in ("preconditions", "postconditions"):
        for c_idx, cond in enumerate(recipe_dict.get(cond_key) or []):
            if isinstance(cond, dict) and cond.get("getter_type") == "shell_exit_zero":
                vasc = VascularTrafficMonitor.inspect_shell_command(
                    str(cond.get("target") or ""), record_antigen_on_block=False
                )
                if not vasc.get("allowed", True):
                    violations.append(
                        f"Blood-Brain Barrier VascularTrafficMonitor blocked {cond_key}[{c_idx}]: {vasc.get('reason')}"
                    )

    return violations


# ==============================================================================
# 3. Trusted Keyring, Publisher Keys & Cryptographic Signing / Verification
# ==============================================================================


def _write_private_json_0600(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    raw = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(prefix=".wbc_hub_", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        try:
            os.fchmod(fd, 0o600)
        except (AttributeError, OSError):
            pass
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
            f.flush()
        os.replace(tmp_path, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def get_trusted_keyring(
    extra_trusted_keys: dict[str, str] | None = None,
    keyring_path: Path | None = None,
) -> dict[str, str]:
    """
    Assemble the active Trusted Keyring (`key_id -> public_key_b64`):
      1. Built-in official Whole Body Compute Hub key (`wbc-hub-official-2026-ed25519`)
      2. User-pinned publisher keys in `<wbc_data_dir>/hub/trusted_publishers.json`
      3. Local self-publisher public key in `<wbc_data_dir>/keys/publisher_ed25519.json` (if present)
      4. `WBC_TRUSTED_HUB_KEYS` env override (`key_id=pub_b64,...`)
      5. Explicit `extra_trusted_keys` map
    """
    ring: dict[str, str] = dict(TRUSTED_BUILTIN_PUBLIC_KEYS)

    k_path = keyring_path or (get_wbc_data_dir() / "hub" / "trusted_publishers.json")
    if k_path.is_file():
        try:
            stored = json.loads(k_path.read_text(encoding="utf-8"))
            if isinstance(stored, dict):
                for kid, pub in (stored.get("trusted_keys") or {}).items():
                    if isinstance(kid, str) and isinstance(pub, str) and kid not in TRUSTED_BUILTIN_PUBLIC_KEYS:
                        ring[kid] = pub.strip()
        except (OSError, ValueError, TypeError):
            stored = None

    local_pub_file = get_wbc_data_dir() / "keys" / "publisher_ed25519.json"
    if local_pub_file.is_file():
        try:
            local_k = json.loads(local_pub_file.read_text(encoding="utf-8"))
            lk_id = str(local_k.get("key_id") or "").strip()
            lk_pub = str(local_k.get("public_key_b64") or "").strip()
            if lk_id and lk_pub and lk_id not in TRUSTED_BUILTIN_PUBLIC_KEYS:
                ring[lk_id] = lk_pub
        except (OSError, ValueError, TypeError):
            local_k = None

    env_keys = os.environ.get("WBC_TRUSTED_HUB_KEYS", "").strip()
    if env_keys:
        for pair in env_keys.split(","):
            if "=" in pair:
                k_id, pub_b64 = pair.split("=", 1)
                k_id = k_id.strip()
                pub_b64 = pub_b64.strip()
                if k_id and pub_b64:
                    ring[k_id] = pub_b64

    if extra_trusted_keys:
        for k_id, pub_b64 in extra_trusted_keys.items():
            if k_id and pub_b64:
                ring[str(k_id).strip()] = str(pub_b64).strip()

    return ring


def pin_trusted_publisher_key(
    key_id: str, public_key_b64: str, keyring_path: Path | None = None
) -> Path:
    """Pin an Ed25519 publisher public key in `<wbc_data_dir>/hub/trusted_publishers.json`."""
    kid = key_id.strip()
    pub_b64 = public_key_b64.strip()
    if kid in TRUSTED_BUILTIN_PUBLIC_KEYS and TRUSTED_BUILTIN_PUBLIC_KEYS[kid] != pub_b64:
        raise RecipeVerificationError(f"Cannot override built-in official Hub key '{kid}'")
    pub_raw = base64.b64decode(pub_b64)
    if len(pub_raw) != 32:
        raise ValueError("Ed25519 public key must decode to 32 bytes")

    target = keyring_path or (get_wbc_data_dir() / "hub" / "trusted_publishers.json")
    existing: dict[str, Any] = {"trusted_keys": {}}
    if target.is_file():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            existing = {"trusted_keys": {}}
    tmap = existing.setdefault("trusted_keys", {})
    tmap[kid] = pub_b64
    existing["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _write_private_json_0600(target, existing)
    return target


def ensure_publisher_keypair(
    namespace: str = "community",
    key_path: Path | None = None,
    auto_generate: bool = True,
) -> dict[str, str]:
    """Load or generate a local publisher Ed25519 keypair (`0o600` owner-only)."""
    target = key_path or (get_wbc_data_dir() / "keys" / "publisher_ed25519.json")
    if target.is_file():
        data = json.loads(target.read_text(encoding="utf-8"))
        seed = base64.b64decode(data["private_seed_b64"])
        derived_pub = base64.b64encode(ed25519_public_from_seed(seed)).decode("ascii")
        if data.get("public_key_b64") != derived_pub:
            raise RecipeVerificationError(f"Corrupted keypair in {target}: public key mismatch")
        return {
            "key_id": str(data["key_id"]),
            "private_seed_b64": str(data["private_seed_b64"]),
            "public_key_b64": derived_pub,
            "key_path": str(target),
        }

    if not auto_generate:
        raise FileNotFoundError(
            f"Publisher signing key not found at {target}. Run `body hub keygen` or pass `--keygen`."
        )

    seed_bytes, pub_bytes = ed25519_generate_keypair()
    fp12 = hashlib.sha256(pub_bytes).hexdigest()[:12]
    clean_ns = re.sub(r"[^a-z0-9_-]", "", namespace.lower()) or "publisher"
    key_id = f"publisher:{clean_ns}:{fp12}"
    payload = {
        "key_id": key_id,
        "algorithm": "ed25519+sha256",
        "private_seed_b64": base64.b64encode(seed_bytes).decode("ascii"),
        "public_key_b64": base64.b64encode(pub_bytes).decode("ascii"),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    _write_private_json_0600(target, payload)
    payload["key_path"] = str(target)
    return payload


def sign_recipe_dict(
    recipe_dict: dict[str, Any],
    *,
    private_seed_b64: str,
    key_id: str,
    signed_at: str | None = None,
    use_pure_python: bool = False,
) -> dict[str, Any]:
    """
    Validate schema + Zero-PII + Organ 05 shell safety, compute dual SHA-256 digests,
    and attach an RFC 8032 Ed25519 `signature` envelope.
    """
    unsigned = copy.deepcopy({k: v for k, v in recipe_dict.items() if k != "signature"})

    schema_errs = validate_pal_workflow_v1_dict(unsigned, require_signature=False)
    if schema_errs:
        raise RecipeVerificationError(
            "Recipe failed PalWorkflow v1 schema validation:\n  - " + "\n  - ".join(schema_errs)
        )

    pii_errs = audit_zero_pii_and_shell_safety(unsigned)
    if pii_errs:
        raise RecipeVerificationError(
            "Recipe failed Zero-PII / Blood-Brain Barrier safety audit:\n  - " + "\n  - ".join(pii_errs)
        )

    seed = base64.b64decode(private_seed_b64)
    pub_bytes = ed25519_public_from_seed(seed, use_pure_python=use_pure_python)
    pub_b64 = base64.b64encode(pub_bytes).decode("ascii")

    payload_sha = compute_canonical_payload_sha256(unsigned)
    hoare_sha = compute_hoare_contract_sha256(unsigned)
    msg = build_recipe_signing_message(payload_sha, hoare_sha)
    sig_bytes = ed25519_sign(seed, msg, use_pure_python=use_pure_python)

    ts = signed_at or datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    signed_out = copy.deepcopy(unsigned)
    signed_out["signature"] = {
        "algorithm": "ed25519+sha256",
        "key_id": key_id,
        "public_key": pub_b64,
        "canonical_payload_sha256": payload_sha,
        "hoare_contract_sha256": hoare_sha,
        "ed25519_signature": base64.b64encode(sig_bytes).decode("ascii"),
        "signed_at": ts,
    }
    return signed_out


def verify_signed_recipe(
    recipe_dict: dict[str, Any],
    *,
    require_trusted_key: bool = True,
    trusted_keys: dict[str, str] | None = None,
    expected_payload_sha256: str | None = None,
    expected_hoare_sha256: str | None = None,
    check_compatibility: bool = False,
    use_pure_python: bool = False,
) -> dict[str, Any]:
    """
    Verify a signed `PalWorkflow` v1 recipe end-to-end:
      1. Schema conformance (`schemas/pal_workflow_v1.json`)
      2. Zero-PII & Organ 05 `VascularTrafficMonitor` shell safety
      3. `canonical_payload_sha256` & `hoare_contract_sha256` integrity
      4. RFC 8032 Ed25519 signature & Trusted Keyring pin verification
      5. Optional OS / WBC version compatibility envelope check
    Raises `RecipeVerificationError` (or `RecipeCompatibilityError`) on any failure.
    """
    schema_errs = validate_pal_workflow_v1_dict(recipe_dict, require_signature=True)
    if schema_errs:
        raise RecipeVerificationError(
            "PalWorkflow v1 schema validation failed:\n  - " + "\n  - ".join(schema_errs)
        )

    pii_errs = audit_zero_pii_and_shell_safety(recipe_dict)
    if pii_errs:
        raise RecipeVerificationError(
            "Zero-PII / Blood-Brain Barrier safety verification failed:\n  - " + "\n  - ".join(pii_errs)
        )

    sig_env = recipe_dict["signature"]
    actual_payload_sha = compute_canonical_payload_sha256(recipe_dict)
    actual_hoare_sha = compute_hoare_contract_sha256(recipe_dict)

    if sig_env["canonical_payload_sha256"] != actual_payload_sha:
        raise RecipeVerificationError(
            f"SHA-256 payload mismatch: recipe body was modified after signing "
            f"(expected {sig_env['canonical_payload_sha256'][:16]}..., got {actual_payload_sha[:16]}...)"
        )

    if sig_env["hoare_contract_sha256"] != actual_hoare_sha:
        raise RecipeVerificationError(
            f"Hoare-contract SHA-256 mismatch: steps/contracts were modified after signing "
            f"(expected {sig_env['hoare_contract_sha256'][:16]}..., got {actual_hoare_sha[:16]}...)"
        )

    if expected_payload_sha256 and actual_payload_sha != expected_payload_sha256:
        raise RecipeVerificationError(
            f"Registry index SHA-256 mismatch: index expected {expected_payload_sha256[:16]}..., "
            f"downloaded payload hashed to {actual_payload_sha[:16]}..."
        )

    if expected_hoare_sha256 and actual_hoare_sha != expected_hoare_sha256:
        raise RecipeVerificationError(
            f"Registry index Hoare SHA-256 mismatch: index expected {expected_hoare_sha256[:16]}..., "
            f"downloaded recipe hashed to {actual_hoare_sha[:16]}..."
        )

    key_id = str(sig_env["key_id"]).strip()
    embedded_pub_b64 = str(sig_env["public_key"]).strip()

    ring = get_trusted_keyring(extra_trusted_keys=trusted_keys)
    pinned_pub_b64 = ring.get(key_id)

    if require_trusted_key:
        if not pinned_pub_b64:
            raise RecipeVerificationError(
                f"Untrusted publisher key '{key_id}'. Pin explicitly via `--trust-key {key_id}=<PUBLIC_KEY_B64>` "
                f"or `body hub trust {key_id} <PUBLIC_KEY_B64>` before pulling or executing."
            )
        if pinned_pub_b64 != embedded_pub_b64:
            raise RecipeVerificationError(
                f"Key spoofing detected for '{key_id}': recipe public key does not match pinned keyring entry!"
            )
        verify_pub_b64 = pinned_pub_b64
    else:
        if pinned_pub_b64 and pinned_pub_b64 != embedded_pub_b64:
            raise RecipeVerificationError(
                f"Key spoofing detected for '{key_id}': recipe public key does not match pinned keyring entry!"
            )
        verify_pub_b64 = pinned_pub_b64 or embedded_pub_b64

    try:
        pub_bytes = base64.b64decode(verify_pub_b64, validate=True)
        sig_bytes = base64.b64decode(str(sig_env["ed25519_signature"]).strip(), validate=True)
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        yaml.YAMLError,
        urllib.error.URLError,
    ) as exc:
        raise RecipeVerificationError(f"Malformed base64 key or signature encoding: {exc}") from exc

    msg = build_recipe_signing_message(actual_payload_sha, actual_hoare_sha)
    if not ed25519_verify(pub_bytes, msg, sig_bytes, use_pure_python=use_pure_python):
        raise RecipeVerificationError(
            f"Invalid Ed25519 cryptographic signature for '{recipe_dict.get('namespace')}/{recipe_dict.get('id')}' (key_id={key_id})"
        )

    compat_status = check_compatibility_envelope(recipe_dict)
    if check_compatibility and not compat_status["compatible"]:
        raise RecipeCompatibilityError(compat_status["reason"])

    return {
        "verified": True,
        "ref": f"{recipe_dict['namespace']}/{recipe_dict['id']}",
        "version": recipe_dict["version"],
        "key_id": key_id,
        "trusted_key": key_id in ring and ring[key_id] == embedded_pub_b64,
        "canonical_payload_sha256": actual_payload_sha,
        "hoare_contract_sha256": actual_hoare_sha,
        "compatibility": compat_status,
    }


def check_compatibility_envelope(
    recipe_dict: dict[str, Any],
    *,
    current_platform: str | None = None,
    current_wbc_version: str | None = None,
) -> dict[str, Any]:
    """Evaluate a recipe's `app_compatibility` envelope against the current host OS & runtime."""
    compat = recipe_dict.get("app_compatibility") or {}
    plat = current_platform or ("macos" if sys.platform == "darwin" else "linux" if sys.platform.startswith("linux") else sys.platform)
    wbc_ver = current_wbc_version or __version__

    supported_platforms = compat.get("platforms") or ["macos", "linux"]
    min_wbc = str(compat.get("min_wbc_version") or "0.1.0")

    if plat not in supported_platforms:
        return {
            "compatible": False,
            "platform": plat,
            "supported_platforms": supported_platforms,
            "wbc_version": wbc_ver,
            "min_wbc_version": min_wbc,
            "reason": f"Recipe requires platform in {supported_platforms}, current host is '{plat}'",
        }

    if _parse_semver(wbc_ver) < _parse_semver(min_wbc):
        return {
            "compatible": False,
            "platform": plat,
            "supported_platforms": supported_platforms,
            "wbc_version": wbc_ver,
            "min_wbc_version": min_wbc,
            "reason": f"Recipe requires whole-body-compute >= {min_wbc} (installed: {wbc_ver})",
        }

    return {
        "compatible": True,
        "platform": plat,
        "supported_platforms": supported_platforms,
        "wbc_version": wbc_ver,
        "min_wbc_version": min_wbc,
        "target_apps": compat.get("target_apps") or [],
        "required_clis": compat.get("required_clis") or [],
        "reason": "ok",
    }


def sign_hub_index(
    index_dict: dict[str, Any],
    *,
    private_seed_b64: str,
    key_id: str = WBC_OFFICIAL_KEY_ID,
    signed_at: str | None = None,
) -> dict[str, Any]:
    """Attach an Ed25519 + SHA-256 signature to the Hub `index.json` catalog."""
    unsigned = copy.deepcopy({k: v for k, v in index_dict.items() if k != "signature"})
    seed = base64.b64decode(private_seed_b64)
    pub_b64 = base64.b64encode(ed25519_public_from_seed(seed)).decode("ascii")
    idx_sha = compute_index_sha256(unsigned)
    msg = f"{INDEX_SIGNING_DOMAIN}\n{idx_sha}".encode()
    sig_b64 = base64.b64encode(ed25519_sign(seed, msg)).decode("ascii")
    ts = signed_at or datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    out = copy.deepcopy(unsigned)
    out["signature"] = {
        "algorithm": "ed25519+sha256",
        "key_id": key_id,
        "public_key": pub_b64,
        "canonical_index_sha256": idx_sha,
        "ed25519_signature": sig_b64,
        "signed_at": ts,
    }
    return out


def verify_hub_index(
    index_dict: dict[str, Any],
    *,
    trusted_keys: dict[str, str] | None = None,
    require_signature: bool = True,
) -> dict[str, Any]:
    """Verify the cryptographic integrity of a Hub `index.json` document."""
    if not isinstance(index_dict, dict) or not isinstance(index_dict.get("recipes"), list):
        raise RecipeVerificationError("Malformed Hub index.json: missing 'recipes' list")

    sig_env = index_dict.get("signature")
    if not sig_env:
        if require_signature:
            raise RecipeVerificationError("Hub index.json is missing required Ed25519 signature")
        return {"verified": False, "unsigned": True}

    actual_sha = compute_index_sha256(index_dict)
    if sig_env.get("canonical_index_sha256") != actual_sha:
        raise RecipeVerificationError("Hub index.json SHA-256 mismatch (tampered catalog)")

    key_id = str(sig_env.get("key_id") or "").strip()
    ring = get_trusted_keyring(extra_trusted_keys=trusted_keys)
    pinned_pub = ring.get(key_id)
    if not pinned_pub:
        raise RecipeVerificationError(f"Hub index signed by untrusted key '{key_id}'")
    if pinned_pub != str(sig_env.get("public_key") or "").strip():
        raise RecipeVerificationError(f"Hub index key spoofing detected for '{key_id}'")

    pub_bytes = base64.b64decode(pinned_pub)
    sig_bytes = base64.b64decode(str(sig_env.get("ed25519_signature") or ""))
    msg = f"{INDEX_SIGNING_DOMAIN}\n{actual_sha}".encode()
    if not ed25519_verify(pub_bytes, msg, sig_bytes):
        raise RecipeVerificationError("Invalid Ed25519 signature on Hub index.json")

    return {"verified": True, "key_id": key_id, "canonical_index_sha256": actual_sha}


# ==============================================================================
# 4. Runtime Execution Gate (Called Before `OrganEyes` / `Hands`)
# ==============================================================================


def get_hub_install_dir() -> Path:
    env_dir = os.environ.get("WBC_HUB_INSTALL_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    return get_wbc_data_dir() / "hub" / "recipes"


def get_hub_lockfile_path(install_dir: Path | None = None) -> Path:
    base = (install_dir.parent if install_dir else (get_wbc_data_dir() / "hub"))
    return base / "installed_lock.json"


def _load_lockfile(lock_path: Path) -> dict[str, Any]:
    if lock_path.is_file():
        try:
            data = json.loads(lock_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("installed"), dict):
                return data
        except (OSError, ValueError, TypeError):
            return {"schema_version": "1.0", "installed": {}}
    return {"schema_version": "1.0", "installed": {}}


def verify_recipe_for_execution(
    recipe_or_dict: Any,
    *,
    require_signed: bool = False,
    trusted_keys: dict[str, str] | None = None,
    check_compatibility: bool = True,
) -> dict[str, Any]:
    """
    Pre-execution security gate invoked by `PalWorkflowRunner.execute_recipe` and
    `OrganEyes.execute_recipe` BEFORE any step touches `OrganEyes` or `Hands`.

    - Any recipe carrying a `signature`, `namespace`, or Hub lockfile entry (or when
      `require_signed=True` / `WBC_REQUIRE_SIGNED_RECIPES=1`) MUST pass full SHA-256 +
      Ed25519 verification against the Trusted Keyring AND match its lockfile digest if
      installed from Hub.
    - Local unsigned user-authored recipes without Hub provenance pass through unless
      `require_signed=True` or `WBC_REQUIRE_SIGNED_RECIPES=1`.
    """
    enforce_all = require_signed or os.environ.get("WBC_REQUIRE_SIGNED_RECIPES") == "1"

    raw_dict: dict[str, Any] | None = None
    if isinstance(recipe_or_dict, dict):
        raw_dict = recipe_or_dict
    elif hasattr(recipe_or_dict, "raw_signed_manifest") and recipe_or_dict.raw_signed_manifest:
        raw_dict = recipe_or_dict.raw_signed_manifest
    elif hasattr(recipe_or_dict, "model_dump"):
        dumped = recipe_or_dict.model_dump(exclude_none=True)
        raw_dict = dumped

    if not isinstance(raw_dict, dict):
        raise RecipeVerificationError("Cannot verify invalid recipe object")

    has_hub_markers = bool(
        raw_dict.get("signature")
        or raw_dict.get("namespace")
        or getattr(recipe_or_dict, "hub_source", False)
    )

    # Also check if recipe id matches a pulled Hub lockfile entry
    lock_entry: dict[str, Any] | None = None
    lock_data = _load_lockfile(get_hub_lockfile_path())
    ns = raw_dict.get("namespace")
    rid = raw_dict.get("id", "")
    if ns and f"{ns}/{rid}" in lock_data["installed"]:
        lock_entry = lock_data["installed"][f"{ns}/{rid}"]
    else:
        for entry in lock_data["installed"].values():
            if entry.get("id") == rid and (ns or getattr(recipe_or_dict, "hub_source", False)):
                lock_entry = entry
                break

    if not has_hub_markers and lock_entry is None and not enforce_all:
        return {"verified": True, "mode": "local_unsigned_recipe"}

    if not raw_dict.get("signature"):
        raise RecipeVerificationError(
            f"Recipe '{rid}' requires a valid Ed25519+SHA256 signature before executing on OrganEyes / Hands."
        )

    expected_payload = lock_entry.get("canonical_payload_sha256") if lock_entry else None
    expected_hoare = lock_entry.get("hoare_contract_sha256") if lock_entry else None

    result = verify_signed_recipe(
        raw_dict,
        require_trusted_key=True,
        trusted_keys=trusted_keys,
        expected_payload_sha256=expected_payload,
        expected_hoare_sha256=expected_hoare,
        check_compatibility=check_compatibility,
    )
    result["mode"] = "signed_hub_verified"
    return result


# ==============================================================================
# 5. Seed Catalog Discovery & Registry Client (`HubClient`)
# ==============================================================================


def get_bundled_seed_dir() -> Path | None:
    """Locate bundled signed Hub seed catalog (`hub_seed/` or `docs/hub/`)."""
    pkg_seed = Path(__file__).resolve().parent / "hub_seed"
    if (pkg_seed / "index.json").is_file():
        return pkg_seed
    repo_hub = Path(__file__).resolve().parents[2] / "docs" / "hub"
    if (repo_hub / "index.json").is_file():
        return repo_hub
    return None


def _is_loopback_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    return host in ("127.0.0.1", "localhost", "::1")


def _fetch_bytes_from_url_or_path(url_or_path: str, timeout: float = 8.0) -> bytes:
    parsed = urllib.parse.urlparse(url_or_path)
    if parsed.scheme == "file":
        local_p = Path(urllib.request.url2pathname(parsed.path))
        return local_p.read_bytes()
    if parsed.scheme == "":
        local_p = Path(url_or_path).expanduser()
        if local_p.is_file():
            return local_p.read_bytes()
    if parsed.scheme == "http" and not _is_loopback_url(url_or_path):
        raise RecipeVerificationError(
            f"Insecure plain HTTP Hub URL rejected ('{url_or_path}'); HTTPS is required for non-loopback registries"
        )
    req = urllib.request.Request(
        url_or_path,
        headers={
            "Accept": "application/json, application/x-yaml, text/yaml, text/plain",
            "User-Agent": f"whole-body-compute/{__version__}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(2 * 1024 * 1024)  # 2 MB safety cap


def _upgrade_local_recipe_for_publish(
    raw: dict[str, Any], *, namespace: str, sanitize: bool = False
) -> dict[str, Any]:
    """Normalize or sanitize a local PalWorkflow YAML into PalWorkflow v1 schema shape."""
    doc = copy.deepcopy(raw)
    doc.pop("signature", None)
    doc.setdefault("schema_version", "1.0")
    doc["namespace"] = namespace or str(doc.get("namespace") or "community")
    doc.setdefault("version", "1.0.0")
    doc.setdefault("license", "Apache-2.0")

    if sanitize:
        doc["description"] = sanitize_procedural_text(str(doc.get("description") or ""))
        if doc.get("notes"):
            doc["notes"] = [sanitize_procedural_text(str(n)) for n in doc["notes"]]

    doc.setdefault(
        "app_compatibility",
        {
            "platforms": ["macos", "linux"],
            "min_wbc_version": "0.1.0",
            "target_apps": [],
            "required_clis": [],
        },
    )

    csum = doc.get("contract_summary")
    if not isinstance(csum, dict) or not csum.get("contracts"):
        doc["contract_summary"] = {
            "contracts": [
                {
                    "name": f"{doc.get('id', 'workflow')}_contract",
                    "preconditions": ["WORKSPACE_READY"],
                    "idempotency_strategy": "IDEMPOTENT_EXECUTION",
                    "natural_key_fields": [],
                    "postconditions": ["EXIT_STATUS_ZERO"],
                }
            ]
        }

    tsum = doc.get("tier_summary") or {}
    archetype = str(tsum.get("archetype") or "general_automation")
    dom_tier = str(tsum.get("dominant_tier") or "TIER_0_LOCAL")
    if dom_tier == "TIER_3_COORDINATE_GUI":
        dom_tier = "TIER_2_SEMANTIC_UI"
    contracts_names = [
        str(c.get("name"))
        for c in (doc["contract_summary"].get("contracts") or [])
        if isinstance(c, dict) and c.get("name")
    ]
    ir_hash = compute_ir_signature_hash(
        archetype=archetype,
        dominant_tier=dom_tier,
        contracts=contracts_names,
        fallback_ladder=list(tsum.get("fallback_ladder") or []),
    )
    if "procedural_dna" not in doc:
        est_tokens = max(12000, len(doc.get("steps") or []) * 16500)
        doc["procedural_dna"] = {
            "archetype": archetype,
            "dominant_tier": dom_tier,
            "ir_signature_hash": ir_hash,
            "estimated_cloud_tokens_saved": est_tokens,
            "estimated_usd_saved_per_run": round(est_tokens / 1_000_000 * 5.0, 4),
            "zero_pii_attestation": {
                "sanitized": True,
                "contains_credentials": False,
                "contains_personal_paths": False,
                "redaction_engine": "whole_body_compute.procedural_telemetry",
                "pii_patterns_checked": [
                    "credentials",
                    "personal_paths",
                    "emails",
                    "ssn",
                    "public_ipv4",
                    "vascular_egress",
                ],
            },
        }
    return doc


class HubClient:
    """Client for `hub.wholebodycompute.org` (and private/offline `PalWorkflow` registries)."""

    def __init__(
        self,
        hub_url: str | None = None,
        install_dir: Path | None = None,
        trusted_keys: dict[str, str] | None = None,
        require_signed_index: bool = True,
    ) -> None:
        env_url = os.environ.get("WBC_HUB_INDEX_URL")
        self.explicit_hub_url = hub_url or env_url
        self.hub_url = self.explicit_hub_url or DEFAULT_HUB_INDEX_URL
        self.install_dir = Path(install_dir).expanduser() if install_dir else get_hub_install_dir()
        self.trusted_keys = trusted_keys
        self.require_signed_index = require_signed_index

    def _resolve_relative_recipe_url(self, index_source: str, recipe_path: str) -> str:
        if recipe_path.startswith(("https://", "http://", "file://")):
            return recipe_path
        rel = recipe_path.lstrip("/")
        if index_source.startswith(("https://", "http://")):
            base = index_source.rsplit("/", 1)[0] + "/"
            return urllib.parse.urljoin(base, rel)
        src_path = Path(
            urllib.request.url2pathname(urllib.parse.urlparse(index_source).path)
            if index_source.startswith("file://")
            else index_source
        )
        base_dir = src_path.parent if src_path.is_file() else src_path
        candidate = base_dir / rel
        if candidate.is_file():
            return str(candidate)
        if rel.startswith("hub/") and (base_dir / rel.removeprefix("hub/")).is_file():
            return str(base_dir / rel.removeprefix("hub/"))
        return str(candidate)

    def fetch_index(self, *, offline: bool = False) -> tuple[dict[str, Any], str]:
        """Fetch and cryptographically verify `index.json`, returning `(index_dict, source_label)`."""
        if offline:
            if self.explicit_hub_url and not self.explicit_hub_url.startswith(("http://", "https://")):
                raw = _fetch_bytes_from_url_or_path(self.explicit_hub_url)
                idx = json.loads(raw.decode("utf-8"))
                verify_hub_index(
                    idx, trusted_keys=self.trusted_keys, require_signature=self.require_signed_index
                )
                return idx, self.explicit_hub_url
            seed_dir = get_bundled_seed_dir()
            if not seed_dir:
                raise FileNotFoundError("Bundled offline Hub seed catalog not found")
            seed_idx_path = seed_dir / "index.json"
            idx = json.loads(seed_idx_path.read_text(encoding="utf-8"))
            verify_hub_index(
                idx, trusted_keys=self.trusted_keys, require_signature=self.require_signed_index
            )
            return idx, str(seed_idx_path)

        # If an explicit custom URL/path was provided, fetch strictly from it without masking errors
        if self.explicit_hub_url:
            src = self.explicit_hub_url
            p = Path(src).expanduser()
            if p.is_dir() and (p / "index.json").is_file():
                src = str(p / "index.json")
            raw = _fetch_bytes_from_url_or_path(src)
            idx = json.loads(raw.decode("utf-8"))
            verify_hub_index(
                idx, trusted_keys=self.trusted_keys, require_signature=self.require_signed_index
            )
            return idx, src

        # Default public Hub resolution: primary -> org mirror -> bundled signed seed fallback
        for candidate_url in (DEFAULT_HUB_INDEX_URL, FALLBACK_HUB_INDEX_URL):
            try:
                raw = _fetch_bytes_from_url_or_path(candidate_url, timeout=4.0)
                idx = json.loads(raw.decode("utf-8"))
                verify_hub_index(
                    idx, trusted_keys=self.trusted_keys, require_signature=self.require_signed_index
                )
                return idx, candidate_url
            except RecipeVerificationError:
                raise
            except (OSError, urllib.error.URLError, ValueError) as _net_err:
                _ = _net_err
                continue

        seed_dir = get_bundled_seed_dir()
        if seed_dir and (seed_dir / "index.json").is_file():
            seed_idx_path = seed_dir / "index.json"
            idx = json.loads(seed_idx_path.read_text(encoding="utf-8"))
            verify_hub_index(
                idx, trusted_keys=self.trusted_keys, require_signature=self.require_signed_index
            )
            return idx, str(seed_idx_path)

        raise ConnectionError(f"Unable to reach {DEFAULT_HUB_INDEX_URL} and no local seed catalog found")

    def search(
        self,
        query: str = "",
        *,
        tag: str | None = None,
        app: str | None = None,
        tier: str | None = None,
        offline: bool = False,
    ) -> dict[str, Any]:
        """Search verified Hub recipes by keyword, tag, target app/bundle_id, or execution tier."""
        idx, source = self.fetch_index(offline=offline)
        entries: list[dict[str, Any]] = list(idx.get("recipes") or [])

        q_terms = [t for t in (query or "").strip().lower().split() if t]
        tag_filter = tag.strip().lower() if tag else None
        app_filter = app.strip().lower() if app else None
        tier_filter = tier.strip().upper() if tier else None

        matched: list[tuple[int, dict[str, Any]]] = []
        for entry in entries:
            if tag_filter and tag_filter not in [str(t).lower() for t in (entry.get("tags") or [])]:
                continue
            if tier_filter and str(entry.get("dominant_tier") or "").upper() != tier_filter:
                continue
            if app_filter:
                apps_blob = " ".join(
                    f"{a.get('app_name', '')} {a.get('bundle_id', '')}"
                    for a in (entry.get("target_apps") or [])
                    if isinstance(a, dict)
                ).lower()
                clis_blob = " ".join(str(c) for c in (entry.get("required_clis") or [])).lower()
                if app_filter not in apps_blob and app_filter not in clis_blob:
                    continue

            if not q_terms:
                matched.append((100, entry))
                continue

            ref_low = str(entry.get("ref") or "").lower()
            name_low = str(entry.get("name") or "").lower()
            desc_low = str(entry.get("description") or "").lower()
            tags_low = " ".join(str(t) for t in (entry.get("tags") or [])).lower()
            triggers_low = " ".join(str(t) for t in (entry.get("triggers") or [])).lower()
            apps_low = " ".join(
                f"{a.get('app_name', '')} {a.get('bundle_id', '')}"
                for a in (entry.get("target_apps") or [])
                if isinstance(a, dict)
            ).lower()
            full_blob = f"{ref_low} {name_low} {desc_low} {tags_low} {triggers_low} {apps_low}"

            if not all(term in full_blob for term in q_terms):
                continue

            score = 0
            for term in q_terms:
                if term in ref_low:
                    score += 10
                if term in name_low:
                    score += 6
                if term in tags_low:
                    score += 5
                if term in apps_low:
                    score += 4
                if term in desc_low:
                    score += 2
            matched.append((score, entry))

        matched.sort(key=lambda item: (-item[0], str(item[1].get("ref", ""))))
        results = [item[1] for item in matched]
        return {
            "status": "ok",
            "query": query,
            "hub_source": source,
            "total_catalog_recipes": len(entries),
            "count": len(results),
            "results": results,
        }

    @staticmethod
    def normalize_ref(ref: str) -> tuple[str, str]:
        raw = (ref or "").strip()
        if "@" in raw:
            raw = raw.split("@", 1)[0]
        if "/" in raw:
            ns, slug = raw.split("/", 1)
        else:
            ns, slug = "wbc", raw
        slug = slug.removesuffix(".yaml").removesuffix(".yml")
        if not _NAMESPACE_RE.match(ns) or not _SLUG_RE.match(slug):
            raise ValueError(f"Invalid recipe reference '{ref}' (expected '<org>/<recipe>', e.g. 'wbc/github-pr-triage')")
        return ns, slug

    def pull(
        self,
        ref: str,
        *,
        version: str | None = None,
        trust_key: str | None = None,
        force: bool = False,
        dry_run: bool = False,
        offline: bool = False,
        check_compatibility: bool = True,
    ) -> dict[str, Any]:
        """
        Download, cryptographically verify (SHA-256 + Ed25519 + Schema + Zero-PII + Organ 05),
        and atomically install a `PalWorkflow` recipe from the Hub.
        """
        ver_pin = version
        if "@" in ref and not ver_pin:
            ref, ver_pin = ref.split("@", 1)
        ns, slug = self.normalize_ref(ref)
        canonical_ref = f"{ns}/{slug}"

        extra_keys = dict(self.trusted_keys or {})
        if trust_key:
            if "=" not in trust_key:
                raise ValueError("--trust-key must be formatted as '<key_id>=<public_key_b64>'")
            tk_id, tk_pub = trust_key.split("=", 1)
            extra_keys[tk_id.strip()] = tk_pub.strip()

        idx, source = self.fetch_index(offline=offline)
        candidates = [
            e
            for e in (idx.get("recipes") or [])
            if e.get("ref") == canonical_ref or (e.get("namespace") == ns and e.get("id") == slug)
        ]
        if ver_pin:
            candidates = [e for e in candidates if str(e.get("version")) == ver_pin]

        if not candidates:
            ver_note = f"@{ver_pin}" if ver_pin else ""
            raise FileNotFoundError(f"Recipe '{canonical_ref}{ver_note}' not found in Hub index ({source})")

        entry = candidates[0]
        recipe_rel = str(entry.get("download_path") or f"recipes/{ns}/{slug}.yaml")
        recipe_url = self._resolve_relative_recipe_url(source, recipe_rel)

        raw_yaml_bytes = _fetch_bytes_from_url_or_path(recipe_url)
        raw_text = raw_yaml_bytes.decode("utf-8")
        recipe_dict = yaml.safe_load(raw_text)
        if not isinstance(recipe_dict, dict):
            raise RecipeVerificationError(f"Downloaded recipe '{canonical_ref}' is not a valid YAML mapping")

        if recipe_dict.get("namespace") != ns or recipe_dict.get("id") != slug:
            raise RecipeVerificationError(
                f"Namespace/slug spoofing in '{canonical_ref}': YAML declares '{recipe_dict.get('namespace')}/{recipe_dict.get('id')}'"
            )

        verification = verify_signed_recipe(
            recipe_dict,
            require_trusted_key=True,
            trusted_keys=extra_keys,
            expected_payload_sha256=entry.get("canonical_payload_sha256"),
            expected_hoare_sha256=entry.get("hoare_contract_sha256"),
            check_compatibility=check_compatibility,
        )

        dest_path = (self.install_dir / ns / f"{slug}.yaml").resolve()
        if not dest_path.is_relative_to(self.install_dir.resolve()):
            raise RecipeVerificationError("Path traversal blocked during Hub pull")

        if dest_path.is_file() and not force and not dry_run:
            existing_dict = yaml.safe_load(dest_path.read_text(encoding="utf-8")) or {}
            if (
                isinstance(existing_dict, dict)
                and (existing_dict.get("signature") or {}).get("canonical_payload_sha256")
                == verification["canonical_payload_sha256"]
            ):
                return {
                    "status": "already_up_to_date",
                    "ref": canonical_ref,
                    "version": recipe_dict["version"],
                    "installed_path": str(dest_path),
                    "verification": verification,
                }

        if not dry_run:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(prefix=f".{slug}_", suffix=".yaml", dir=str(dest_path.parent))
            tmp_p = Path(tmp_name)
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(raw_yaml_bytes)
                    f.flush()
                os.replace(tmp_p, dest_path)
            finally:
                if tmp_p.exists():
                    tmp_p.unlink(missing_ok=True)

            if trust_key and "=" in trust_key:
                tk_id, tk_pub = trust_key.split("=", 1)
                pin_trusted_publisher_key(tk_id.strip(), tk_pub.strip())

            lock_path = get_hub_lockfile_path(self.install_dir)
            lock_data = _load_lockfile(lock_path)
            lock_data["installed"][canonical_ref] = {
                "ref": canonical_ref,
                "namespace": ns,
                "id": slug,
                "version": recipe_dict["version"],
                "installed_path": str(dest_path),
                "canonical_payload_sha256": verification["canonical_payload_sha256"],
                "hoare_contract_sha256": verification["hoare_contract_sha256"],
                "key_id": verification["key_id"],
                "public_key": recipe_dict["signature"]["public_key"],
                "pulled_from": recipe_url,
                "verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            _write_private_json_0600(lock_path, lock_data)

        return {
            "status": "dry_run_verified" if dry_run else "installed",
            "ref": canonical_ref,
            "name": recipe_dict["name"],
            "version": recipe_dict["version"],
            "dominant_tier": recipe_dict["procedural_dna"]["dominant_tier"],
            "estimated_cloud_tokens_saved": recipe_dict["procedural_dna"]["estimated_cloud_tokens_saved"],
            "installed_path": str(dest_path),
            "verification": verification,
        }

    def publish(
        self,
        recipe_path: Path | str,
        *,
        namespace: str | None = None,
        out_dir: Path | None = None,
        key_path: Path | None = None,
        private_seed_b64: str | None = None,
        key_id: str | None = None,
        auto_keygen: bool = True,
        sanitize: bool = False,
        registry_url: str | None = None,
        upload: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Validate, audit for Zero-PII & Organ 05 safety, cryptographically sign (Ed25519 + SHA-256),
        and package (or upload) a `PalWorkflow` v1 recipe for the Procedural DNA Hub.
        """
        src_path = Path(recipe_path).expanduser()
        if not src_path.is_file():
            raise FileNotFoundError(f"Recipe file not found: {src_path}")

        raw = yaml.safe_load(src_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise RecipeVerificationError("Recipe file must parse to a YAML/JSON object")

        eff_ns = namespace or str(raw.get("namespace") or "community")
        prepared = _upgrade_local_recipe_for_publish(raw, namespace=eff_ns, sanitize=sanitize)

        if private_seed_b64 and key_id:
            seed_b64 = private_seed_b64
            eff_key_id = key_id
            pub_b64 = base64.b64encode(ed25519_public_from_seed(base64.b64decode(seed_b64))).decode("ascii")
        else:
            env_seed = os.environ.get("WBC_PUBLISHER_PRIVATE_SEED_B64")
            env_kid = os.environ.get("WBC_PUBLISHER_KEY_ID")
            if env_seed and env_kid:
                seed_b64 = env_seed.strip()
                eff_key_id = env_kid.strip()
                pub_b64 = base64.b64encode(ed25519_public_from_seed(base64.b64decode(seed_b64))).decode("ascii")
            else:
                kp = ensure_publisher_keypair(
                    namespace=eff_ns, key_path=key_path, auto_generate=auto_keygen
                )
                seed_b64 = kp["private_seed_b64"]
                eff_key_id = kp["key_id"]
                pub_b64 = kp["public_key_b64"]

        signed_recipe = sign_recipe_dict(
            prepared,
            private_seed_b64=seed_b64,
            key_id=eff_key_id,
        )

        # Self-verify immediately using the signing public key
        verification = verify_signed_recipe(
            signed_recipe,
            require_trusted_key=True,
            trusted_keys={eff_key_id: pub_b64},
            check_compatibility=False,
        )

        ns = signed_recipe["namespace"]
        slug = signed_recipe["id"]
        ref = f"{ns}/{slug}"

        index_entry = build_index_entry_from_signed_recipe(
            signed_recipe, download_path=f"recipes/{ns}/{slug}.yaml"
        )

        bundle_root = (
            Path(out_dir).expanduser()
            if out_dir
            else (get_wbc_data_dir() / "hub" / "published")
        )
        signed_yaml_path = bundle_root / ns / f"{slug}.yaml"
        manifest_path = bundle_root / ns / f"{slug}.manifest.json"
        signed_yaml_text = yaml.safe_dump(signed_recipe, sort_keys=False, allow_unicode=True)

        upload_result: dict[str, Any] | None = None
        if not dry_run:
            signed_yaml_path.parent.mkdir(parents=True, exist_ok=True)
            signed_yaml_path.write_text(signed_yaml_text, encoding="utf-8")
            manifest_path.write_text(json.dumps(index_entry, indent=2) + "\n", encoding="utf-8")

            if upload:
                target_reg = registry_url or os.environ.get(
                    "WBC_HUB_PUBLISH_URL", "https://hub.wholebodycompute.org/v1/publish"
                )
                if target_reg.startswith("http://") and not _is_loopback_url(target_reg):
                    raise RecipeVerificationError("Refusing to upload over plain HTTP to non-loopback host")
                body_bytes = json.dumps(
                    {
                        "ref": ref,
                        "recipe": signed_recipe,
                        "index_entry": index_entry,
                    }
                ).encode("utf-8")
                req = urllib.request.Request(
                    target_reg,
                    data=body_bytes,
                    method="POST",
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": f"whole-body-compute/{__version__}",
                    },
                )
                with urllib.request.urlopen(req, timeout=10.0) as resp:
                    resp_raw = resp.read(256 * 1024).decode("utf-8", errors="replace")
                    upload_result = {
                        "http_status": resp.status,
                        "response": json.loads(resp_raw) if resp_raw.strip().startswith("{") else resp_raw,
                    }

        return {
            "status": "dry_run_signed" if dry_run else ("uploaded" if upload_result else "published_local_bundle"),
            "ref": ref,
            "version": signed_recipe["version"],
            "key_id": eff_key_id,
            "public_key": pub_b64,
            "canonical_payload_sha256": verification["canonical_payload_sha256"],
            "hoare_contract_sha256": verification["hoare_contract_sha256"],
            "signed_yaml_path": str(signed_yaml_path),
            "manifest_path": str(manifest_path),
            "index_entry": index_entry,
            "upload": upload_result,
        }

    def list_installed(self) -> list[dict[str, Any]]:
        """List all pulled & verified Hub recipes tracked in `installed_lock.json`."""
        lock_data = _load_lockfile(get_hub_lockfile_path(self.install_dir))
        return sorted(lock_data["installed"].values(), key=lambda x: str(x.get("ref", "")))


def build_index_entry_from_signed_recipe(
    signed_recipe: dict[str, Any], *, download_path: str | None = None
) -> dict[str, Any]:
    """Extract the compact searchable Hub `index.json` entry from a verified signed recipe."""
    ns = signed_recipe["namespace"]
    slug = signed_recipe["id"]
    sig = signed_recipe["signature"]
    dna = signed_recipe["procedural_dna"]
    compat = signed_recipe["app_compatibility"]
    contracts = (signed_recipe.get("contract_summary") or {}).get("contracts") or []

    return {
        "ref": f"{ns}/{slug}",
        "namespace": ns,
        "id": slug,
        "name": signed_recipe["name"],
        "version": signed_recipe["version"],
        "description": signed_recipe["description"],
        "license": signed_recipe.get("license", "Apache-2.0"),
        "author": signed_recipe.get("author", ns),
        "price_usd": 0.0,
        "tags": list(signed_recipe.get("tags") or []),
        "triggers": list(signed_recipe.get("triggers") or []),
        "destructive": bool(signed_recipe.get("destructive", False)),
        "platforms": list(compat.get("platforms") or ["macos", "linux"]),
        "min_wbc_version": compat.get("min_wbc_version", "0.1.0"),
        "target_apps": list(compat.get("target_apps") or []),
        "required_clis": list(compat.get("required_clis") or []),
        "dominant_tier": dna["dominant_tier"],
        "archetype": dna["archetype"],
        "ir_signature_hash": dna["ir_signature_hash"],
        "estimated_cloud_tokens_saved": dna["estimated_cloud_tokens_saved"],
        "estimated_usd_saved_per_run": dna["estimated_usd_saved_per_run"],
        "steps_count": len(signed_recipe.get("steps") or []),
        "hoare_contracts_count": len(contracts),
        "download_path": download_path or f"recipes/{ns}/{slug}.yaml",
        "key_id": sig["key_id"],
        "public_key": sig["public_key"],
        "canonical_payload_sha256": sig["canonical_payload_sha256"],
        "hoare_contract_sha256": sig["hoare_contract_sha256"],
        "ed25519_signature": sig["ed25519_signature"],
        "signed_at": sig["signed_at"],
    }


# ==============================================================================
# 6. Terminal Formatters
# ==============================================================================


def format_hub_search_table(report: dict[str, Any]) -> str:
    results = report.get("results") or []
    q = report.get("query") or "*"
    lines = [
        f"🌐 [Procedural DNA Hub — {report.get('count', 0)}/{report.get('total_catalog_recipes', 0)} Signed $0.00 Recipes matching '{q}']",
        "=" * 102,
        f"  {'RECIPE REF':<30} {'VER':<8} {'TIER':<20} {'SAVED/RUN':<12} {'HOARE':<7} {'SUMMARY'}",
        "-" * 102,
    ]
    if not results:
        lines.append("  No matching signed recipes found. Try `body search` with no arguments to list all.")
    for item in results:
        ref = str(item.get("ref", ""))[:29]
        ver = str(item.get("version", "1.0.0"))[:7]
        tier = str(item.get("dominant_tier", "TIER_0_LOCAL")).replace("TIER_", "T")[:19]
        tok = int(item.get("estimated_cloud_tokens_saved") or 0)
        tok_str = f"~{round(tok / 1000)}K tok" if tok >= 1000 else f"{tok} tok"
        hc = f"{item.get('hoare_contracts_count', 1)}x {{P}}Q"
        desc = str(item.get("name") or item.get("description") or "")[:36]
        lines.append(f"  {ref:<30} {ver:<8} {tier:<20} {tok_str:<12} {hc:<7} {desc}")
    lines.append("=" * 102)
    lines.append("  Install & verify any recipe with: `body pull <org/recipe>` (e.g. `body pull wbc/github-pr-triage`)")
    return "\n".join(lines)


def format_hub_pull_receipt(result: dict[str, Any]) -> str:
    ver = result.get("verification") or {}
    status = result.get("status", "installed")
    icon = "✅" if status in ("installed", "already_up_to_date", "dry_run_verified") else "❌"
    return "\n".join(
        [
            f"{icon} [{status.upper()}] `{result.get('ref')}@{result.get('version')}`",
            f"   • Signed Key ID:          `{ver.get('key_id')}` (Trusted Keyring: {'YES' if ver.get('trusted_key') else 'NO'})",
            f"   • Payload SHA-256:        `{str(ver.get('canonical_payload_sha256', ''))[:24]}...`",
            f"   • Hoare Contract SHA-256: `{str(ver.get('hoare_contract_sha256', ''))[:24]}...`",
            f"   • Installed Path:         `{result.get('installed_path')}`",
        ]
    )


def format_hub_publish_receipt(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"🔏 [{str(result.get('status', '')).upper()}] Signed `{result.get('ref')}@{result.get('version')}` (PalWorkflow + ProceduralDNA v1)",
            f"   • Publisher Key ID:       `{result.get('key_id')}`",
            f"   • Public Key (Ed25519):   `{result.get('public_key')}`",
            f"   • Payload SHA-256:        `{str(result.get('canonical_payload_sha256', ''))[:24]}...`",
            f"   • Hoare Contract SHA-256: `{str(result.get('hoare_contract_sha256', ''))[:24]}...`",
            f"   • Signed Bundle YAML:     `{result.get('signed_yaml_path')}`",
            f"   • Registry Manifest JSON: `{result.get('manifest_path')}`",
        ]
    )


# ==============================================================================
# 7. CLI Entrypoint Dispatcher (`body pull`, `body search`, `body publish`, `body hub`)
# ==============================================================================


def run_hub_cli_command(cmd: str, args: Any) -> int:
    """Execute `body pull`, `body search`, `body publish`, or `body hub` CLI subcommands."""
    is_json = bool(getattr(args, "json", False))
    hub_url = getattr(args, "hub_url", None)
    install_dir = getattr(args, "install_dir", None)
    offline = bool(getattr(args, "offline", False))

    action = cmd
    if cmd == "hub":
        action = getattr(args, "hub_action", "list") or "list"

    client = HubClient(
        hub_url=hub_url,
        install_dir=Path(install_dir) if install_dir else None,
    )

    try:
        if action in ("search", "list"):
            query_val = ""
            if cmd == "search":
                q_parts = getattr(args, "query", None) or []
                query_val = " ".join(q_parts) if isinstance(q_parts, list) else str(q_parts)
            elif cmd == "hub":
                t_parts = getattr(args, "targets", None) or []
                query_val = " ".join(t_parts) if action == "search" else ""

            report = client.search(
                query=query_val,
                tag=getattr(args, "tag", None),
                app=getattr(args, "app", None),
                tier=getattr(args, "tier", None),
                offline=offline,
            )
            if is_json:
                print(json.dumps(report, indent=2))
            else:
                table_out = format_hub_search_table(report)
                if cmd == "hub" and action == "list":
                    installed_count = len(client.list_installed())
                    table_out += (
                        f"\n  📦 Installed Locally: {installed_count} verified recipe(s) (`body hub installed`)\n"
                        "  🧭 Hub Actions:       `body hub <list|search|pull|publish|verify|installed|keygen|trust>`"
                    )
                print("\n" + table_out + "\n")
            return 0

        if action == "pull":
            ref = getattr(args, "recipe_ref", None)
            if not ref and getattr(args, "targets", None):
                ref = args.targets[0]
            if not ref:
                raise ValueError("Specify a recipe reference to pull (e.g. `body pull wbc/github-pr-triage`)")

            res = client.pull(
                ref=ref,
                version=getattr(args, "version_pin", None),
                trust_key=getattr(args, "trust_key", None),
                force=bool(getattr(args, "force", False)),
                dry_run=bool(getattr(args, "dry_run", False)),
                offline=offline,
                check_compatibility=not bool(getattr(args, "ignore_compat", False)),
            )
            if is_json:
                print(json.dumps(res, indent=2))
            else:
                print("\n" + format_hub_pull_receipt(res) + "\n")
            return 0

        if action == "publish":
            target_file = getattr(args, "recipe_path", None)
            if not target_file and getattr(args, "targets", None):
                target_file = args.targets[0]
            if not target_file:
                raise ValueError("Specify a recipe YAML path to sign and publish (e.g. `body publish my_recipe.yaml`)")

            res = client.publish(
                recipe_path=target_file,
                namespace=getattr(args, "namespace", None),
                out_dir=Path(args.out_dir) if getattr(args, "out_dir", None) else None,
                key_path=Path(args.key_path) if getattr(args, "key_path", None) else None,
                auto_keygen=not bool(getattr(args, "no_keygen", False)),
                sanitize=bool(getattr(args, "sanitize", False)),
                registry_url=getattr(args, "registry_url", None),
                upload=bool(getattr(args, "upload", False)),
                dry_run=bool(getattr(args, "dry_run", False)),
            )
            if is_json:
                print(json.dumps(res, indent=2))
            else:
                print("\n" + format_hub_publish_receipt(res) + "\n")
            return 0

        if action == "verify":
            targets = getattr(args, "targets", None) or []
            if not targets:
                raise ValueError("Specify a signed recipe YAML path to verify (e.g. `body hub verify recipe.yaml`)")
            p = Path(targets[0]).expanduser()
            raw_data = yaml.safe_load(p.read_text(encoding="utf-8"))
            ver = verify_signed_recipe(
                raw_data,
                require_trusted_key=not bool(getattr(args, "allow_untrusted_key", False)),
                check_compatibility=False,
            )
            if is_json:
                print(json.dumps(ver, indent=2))
            else:
                print(
                    f"✅ Verified `{ver['ref']}@{ver['version']}` "
                    f"(Key: `{ver['key_id']}`, Payload SHA-256: `{ver['canonical_payload_sha256'][:20]}...`)"
                )
            return 0

        if action == "installed":
            items = client.list_installed()
            payload = {"status": "ok", "count": len(items), "installed": items}
            if is_json:
                print(json.dumps(payload, indent=2))
            else:
                print(f"\n📦 [Installed Procedural DNA Hub Recipes — {len(items)} verified]")
                for it in items:
                    print(f"  • {it['ref']}@{it['version']} ({it['key_id']}) -> `{it['installed_path']}`")
                print()
            return 0

        if action == "keygen":
            ns = getattr(args, "namespace", None) or "community"
            kp = ensure_publisher_keypair(
                namespace=ns,
                key_path=Path(args.key_path) if getattr(args, "key_path", None) else None,
                auto_generate=True,
            )
            pub_out = {
                "status": "ok",
                "key_id": kp["key_id"],
                "public_key_b64": kp["public_key_b64"],
                "key_path": kp["key_path"],
            }
            if is_json:
                print(json.dumps(pub_out, indent=2))
            else:
                print(
                    f"🔑 Publisher Ed25519 Key Ready:\n"
                    f"   • Key ID:     `{kp['key_id']}`\n"
                    f"   • Public Key: `{kp['public_key_b64']}`\n"
                    f"   • Stored At:  `{kp['key_path']}` (0600)"
                )
            return 0

        if action == "trust":
            targets = getattr(args, "targets", None) or []
            if len(targets) == 1 and "=" in targets[0]:
                kid, pub = targets[0].split("=", 1)
            elif len(targets) >= 2:
                kid, pub = targets[0], targets[1]
            else:
                raise ValueError("Usage: `body hub trust <key_id> <public_key_b64>`")
            ring_path = pin_trusted_publisher_key(kid, pub)
            out = {"status": "ok", "key_id": kid.strip(), "keyring_path": str(ring_path)}
            if is_json:
                print(json.dumps(out, indent=2))
            else:
                print(f"✅ Trusted publisher key `{kid.strip()}` pinned in `{ring_path}`")
            return 0

        raise ValueError(f"Unsupported hub action: {action}")
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        yaml.YAMLError,
        urllib.error.URLError,
    ) as exc:
        err_payload = {"status": "error", "command": cmd, "action": action, "error": str(exc)}
        if is_json:
            print(json.dumps(err_payload, indent=2))
        else:
            print(f"❌ [Hub Error] {exc}", file=sys.stderr)
        return 1
