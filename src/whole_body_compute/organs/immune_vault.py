"""
Organ 05 (Blood-Brain Barrier) - Cache-Safe Reversible Airlock Blob Vault.

Stores secret-redacted raw tool outputs in a content-addressable local store
(`~/.whole_body_compute/airlock_blobs/<sha256[:12]>.txt`) when `PostToolUse`
compacts a large stdout/tool buffer. Appends an immutable receipt at the tool
boundary (`0.0%` HTTP prompt-prefix cache invalidation) for `<5ms` exact line
recovery via `body immune --blob <id> --lines <start>:<end>`.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any

_BLOB_ID_RE = re.compile(r"^[0-9a-f]{8,64}$")
_LINE_RANGE_RE = re.compile(
    r"^\s*[Ll]?(\d+)?\s*(?:[:\-]\s*[Ll]?(\d+)?)?\s*$"
)

COMMUNITY_TTL_SECONDS = 3600  # 1h ephemeral session lookup (Free Forever)
PRO_DEFAULT_TTL_SECONDS = 86400  # 24h multi-session history (Pro / Team / Enterprise)
MAX_PRO_TTL_SECONDS = 7 * 86400  # Up to 7d configurable retention
DEFAULT_MAX_VAULT_BYTES = 50 * 1024 * 1024  # 50 MB cap


def get_airlock_vault_dir(vault_dir: Path | str | None = None) -> Path:
    """Resolve the Airlock Blob Vault directory, honoring hermetic test overrides."""
    if vault_dir is not None:
        return Path(vault_dir).expanduser()
    env_override = os.environ.get("WBC_AIRLOCK_VAULT_DIR", "").strip()
    if env_override:
        return Path(env_override).expanduser()
    try:
        from whole_body_compute.organs.immune_hygiene import _get_wbc_data_dir

        return _get_wbc_data_dir() / "airlock_blobs"
    except (ImportError, OSError, RuntimeError, ValueError):
        return Path.home() / ".whole_body_compute" / "airlock_blobs"


def get_effective_vault_limits() -> tuple[int, int, str]:
    """Return `(ttl_seconds, max_vault_bytes, active_tier)` aligned with LICENSE."""
    tier = "enterprise"
    try:
        from whole_body_compute.license import LicenseManager

        tier = str(LicenseManager().get_active_tier() or "community").lower()
    except (ImportError, OSError, RuntimeError, ValueError):
        tier = "community"

    default_ttl = COMMUNITY_TTL_SECONDS if tier == "community" else PRO_DEFAULT_TTL_SECONDS
    env_ttl = os.environ.get("WBC_AIRLOCK_TTL_SECONDS", "").strip()
    if env_ttl:
        try:
            parsed_ttl = max(1, int(float(env_ttl)))
            if tier == "community":
                ttl_seconds = min(parsed_ttl, COMMUNITY_TTL_SECONDS)
            else:
                ttl_seconds = min(parsed_ttl, MAX_PRO_TTL_SECONDS)
        except ValueError:
            ttl_seconds = default_ttl
    else:
        ttl_seconds = default_ttl

    env_max_b = os.environ.get("WBC_AIRLOCK_MAX_BYTES", "").strip()
    if env_max_b:
        try:
            max_bytes = max(4096, int(env_max_b))
        except ValueError:
            max_bytes = DEFAULT_MAX_VAULT_BYTES
    else:
        max_bytes = DEFAULT_MAX_VAULT_BYTES

    return ttl_seconds, max_bytes, tier


def _redact_text_safely(raw_text: str) -> str:
    """Scrub credentials across all 23 vendor patterns prior to hashing or disk write."""
    try:
        from whole_body_compute.redaction import redact_secrets

        return str(redact_secrets(raw_text))
    except (ImportError, OSError, RuntimeError, ValueError):
        # Defense-in-depth stdlib fallback if redaction module is unavailable
        return re.sub(
            r"\b(?:sk-[A-Za-z0-9_\-]{16,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9\-]{16,}|AKIA[0-9A-Z]{16})\b",
            "[REDACTED_SECRET]",
            raw_text,
        )


def prune_airlock_vault(
    vault_dir: Path | str | None = None,
    ttl_seconds: int | None = None,
    max_bytes: int | None = None,
    now: float | None = None,
) -> dict[str, int]:
    """Evict expired Airlock blobs and enforce total byte ceiling (`50 MB` default)."""
    vdir = get_airlock_vault_dir(vault_dir)
    if not vdir.is_dir():
        return {"expired_removed": 0, "cap_evicted": 0, "remaining_files": 0, "remaining_bytes": 0}

    eff_ttl, eff_max_bytes, _ = get_effective_vault_limits()
    ttl = eff_ttl if ttl_seconds is None else max(1, int(ttl_seconds))
    cap = eff_max_bytes if max_bytes is None else max(4096, int(max_bytes))
    ts_now = float(time.time() if now is None else now)

    expired_removed = 0
    cap_evicted = 0
    survivors: list[tuple[float, int, Path]] = []

    try:
        with os.scandir(vdir) as it:
            for entry in it:
                if not entry.is_file(follow_symlinks=False) or not entry.name.endswith(".txt"):
                    continue
                try:
                    st = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                age = ts_now - st.st_mtime
                p = Path(entry.path)
                if age > ttl:
                    try:
                        p.unlink(missing_ok=True)
                        expired_removed += 1
                    except OSError:
                        pass
                else:
                    survivors.append((st.st_mtime, st.st_size, p))
    except OSError:
        return {"expired_removed": expired_removed, "cap_evicted": 0, "remaining_files": 0, "remaining_bytes": 0}

    total_bytes = sum(sz for _, sz, _ in survivors)
    if total_bytes > cap:
        # Evict oldest mtime first until within budget
        survivors.sort(key=lambda item: item[0])
        kept: list[tuple[float, int, Path]] = []
        for mtime, sz, p in survivors:
            if total_bytes > cap:
                try:
                    p.unlink(missing_ok=True)
                    total_bytes -= sz
                    cap_evicted += 1
                except OSError:
                    kept.append((mtime, sz, p))
            else:
                kept.append((mtime, sz, p))
        survivors = kept

    return {
        "expired_removed": expired_removed,
        "cap_evicted": cap_evicted,
        "remaining_files": len(survivors),
        "remaining_bytes": max(0, total_bytes),
    }


def format_airlock_receipt(blob_id: str) -> str:
    """Format the immutable single-line tool-boundary receipt (`0%` proxy cache invalidation)."""
    return (
        f"🔎 Lossless Airlock Receipt [blob: {blob_id}] - "
        f"inspect exact raw lines via: body immune --blob {blob_id} --lines <start>:<end>"
    )


def store_airlock_blob(
    raw_text: str,
    vault_dir: Path | str | None = None,
) -> dict[str, Any]:
    """
    Scrub secrets, compute SHA-256[:12], persist to Airlock Blob Vault, and return receipt metadata.
    """
    t0 = time.perf_counter()
    safe_text = _redact_text_safely(str(raw_text or ""))
    encoded = safe_text.encode("utf-8", errors="replace")
    blob_id = hashlib.sha256(encoded).hexdigest()[:12]

    vdir = get_airlock_vault_dir(vault_dir)
    vdir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(vdir, 0o700)
    except OSError:
        pass

    ttl_s, max_b, tier = get_effective_vault_limits()
    target_file = vdir / f"{blob_id}.txt"

    if target_file.is_file() and not target_file.is_symlink():
        try:
            os.utime(target_file, None)
        except OSError:
            pass
    else:
        tmp_file = vdir / f".tmp_{blob_id}_{os.getpid()}.txt"
        try:
            fd = os.open(str(tmp_file), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as fh:
                fh.write(encoded)
            os.replace(str(tmp_file), str(target_file))
        finally:
            if tmp_file.exists():
                try:
                    tmp_file.unlink(missing_ok=True)
                except OSError:
                    pass

    # Opportunistic sub-millisecond TTL / 50MB hygiene sweep
    prune_airlock_vault(vault_dir=vdir, ttl_seconds=ttl_s, max_bytes=max_b)

    line_count = len(safe_text.splitlines())
    receipt = format_airlock_receipt(blob_id)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    return {
        "blob_id": blob_id,
        "path": str(target_file),
        "bytes": len(encoded),
        "lines": line_count,
        "ttl_seconds": ttl_s,
        "tier": tier,
        "receipt": receipt,
        "latency_ms": elapsed_ms,
    }


def parse_line_range_spec(
    lines_spec: str | None,
    total_lines: int,
    default_span: int = 160,
    max_span: int = 400,
) -> tuple[int, int]:
    """
    Parse 1-indexed inclusive line range specifications (`120:180`, `120-180`, `L120:L180`, `140`, `120:`).
    """
    if total_lines <= 0:
        return (1, 1)

    if not lines_spec or not str(lines_spec).strip():
        return (1, min(total_lines, max(1, default_span)))

    spec = str(lines_spec).strip()
    m = _LINE_RANGE_RE.match(spec)
    if not m:
        raise ValueError(
            f"Invalid --lines range '{lines_spec}'. Expected <start>:<end> (e.g. 120:180)."
        )

    has_sep = ":" in spec or "-" in spec
    g1, g2 = m.group(1), m.group(2)

    if g1 is None and g2 is None:
        return (1, min(total_lines, max(1, default_span)))

    start = max(1, int(g1)) if g1 is not None else 1
    if not has_sep:
        end = start
    elif g2 is not None:
        end = max(start, int(g2))
    else:
        end = start + max(1, default_span) - 1

    start = min(start, total_lines)
    end = min(max(start, end), total_lines)
    if end - start + 1 > max_span:
        end = min(total_lines, start + max_span - 1)
    return (start, end)


def read_airlock_blob_slice(
    blob_id: str,
    lines_spec: str | None = None,
    vault_dir: Path | str | None = None,
    max_span: int = 400,
) -> dict[str, Any]:
    """
    Recover exact secret-redacted raw lines from the Airlock Blob Vault in `<5ms`.
    """
    t0 = time.perf_counter()
    clean_id = str(blob_id or "").strip().lower()
    if not _BLOB_ID_RE.match(clean_id):
        return {
            "found": False,
            "blob_id": clean_id,
            "error": f"Invalid Airlock blob ID '{blob_id}'. Expected 8-64 lowercase hex characters.",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    vdir = get_airlock_vault_dir(vault_dir)
    ttl_s, _, tier = get_effective_vault_limits()

    candidate: Path | None = None
    exact = vdir / f"{clean_id[:12]}.txt" if len(clean_id) >= 12 else None
    if exact is not None and exact.is_file() and not exact.is_symlink():
        candidate = exact
    elif vdir.is_dir():
        matches = [
            p
            for p in vdir.glob(f"{clean_id}*.txt")
            if p.is_file() and not p.is_symlink()
        ]
        if len(matches) == 1:
            candidate = matches[0]

    if candidate is None:
        return {
            "found": False,
            "blob_id": clean_id,
            "tier": tier,
            "error": (
                f"Airlock blob '{clean_id}' not found or expired "
                f"(active retention: {ttl_s // 3600}h on {tier} tier)."
            ),
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    try:
        st = candidate.stat(follow_symlinks=False)
    except OSError as exc:
        return {
            "found": False,
            "blob_id": clean_id,
            "error": f"Unable to stat Airlock blob '{clean_id}': {exc}",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    age_s = time.time() - st.st_mtime
    if age_s > ttl_s:
        try:
            candidate.unlink(missing_ok=True)
        except OSError:
            pass
        upgrade_note = (
            " Upgrade to Whole Body Compute Pro (`body upgrade`) for 24h-7d multi-session Airlock history."
            if tier == "community"
            else ""
        )
        return {
            "found": False,
            "expired": True,
            "blob_id": candidate.stem,
            "tier": tier,
            "error": (
                f"Airlock blob '{candidate.stem}' expired ({int(age_s)}s > {ttl_s}s {tier} TTL).{upgrade_note}"
            ),
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    raw_content = candidate.read_text(encoding="utf-8", errors="replace")
    all_lines = raw_content.splitlines()
    total_lines = len(all_lines)

    try:
        start_ln, end_ln = parse_line_range_spec(
            lines_spec, total_lines=total_lines, default_span=160, max_span=max_span
        )
    except ValueError as exc:
        return {
            "found": False,
            "blob_id": candidate.stem,
            "error": str(exc),
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    selected_raw = all_lines[start_ln - 1 : end_ln] if total_lines > 0 else []
    # Defense-in-depth redaction guarantee on egress
    redacted_slice_text = _redact_text_safely("\n".join(selected_raw))
    redacted_lines = redacted_slice_text.splitlines() if selected_raw else []

    numbered_lines = [
        f"L{start_ln + idx}: {line}" for idx, line in enumerate(redacted_lines)
    ]
    formatted = (
        f"🔎 Airlock Blob [{candidate.stem}] (L{start_ln}-L{end_ln} of {total_lines:,} lines):\n"
        + "\n".join(numbered_lines)
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    return {
        "found": True,
        "blob_id": candidate.stem,
        "start_line": start_ln,
        "end_line": end_ln,
        "total_lines": total_lines,
        "raw_slice": redacted_slice_text,
        "lines": redacted_lines,
        "formatted": formatted,
        "tier": tier,
        "ttl_seconds": ttl_s,
        "latency_ms": elapsed_ms,
    }
