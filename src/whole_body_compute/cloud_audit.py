"""
Opt-in Cloud Usage & Spend Audit helpers for `body audit` / `body audit-spend`.

Supports two 100% opt-in cloud data sources without compromising local-first privacy:
  1. Offline file import (`--import <path>`):
     - Claude.ai Data Export (`conversations.json` or `.zip` containing `conversations.json`)
     - Anthropic Console billing / usage `.csv`
  2. Anthropic Organization Admin API (`--cloud` / `--anthropic-admin-key`):
     - `/v1/organizations/usage_report/messages`
     - `/v1/organizations/cost_report`
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from whole_body_compute.redaction import redact_secrets

DEFAULT_MAX_CLOUD_EXPORT_BYTES = 256 * 1024 * 1024  # 256 MiB safety cap against zip/JSON bombs
MAX_ADMIN_API_RESPONSE_BYTES = 16 * 1024 * 1024  # 16 MiB cap per API response
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]+")


def _get_max_export_bytes() -> int:
    raw = os.environ.get("WBC_MAX_CLOUD_EXPORT_BYTES", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return DEFAULT_MAX_CLOUD_EXPORT_BYTES


def _sanitize_terminal_text(val: Any) -> str:
    """Strip terminal escape/control sequences and scrub accidental API keys from status strings."""
    cleaned = _CONTROL_CHARS_RE.sub(" ", str(val or "")).strip()
    return redact_secrets(cleaned)


def _parse_iso_datetime(raw: Any) -> Optional[dt.datetime]:
    if not raw:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(s)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed
    except Exception:
        pass
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return dt.datetime.strptime(s[:10], fmt).replace(tzinfo=dt.timezone.utc)
        except Exception:
            continue
    return None


def _parse_claude_conversations_payload(
    payload: Any,
    source_label: str,
    days: int = 30,
    price_per_m: float = 5.0,
) -> Dict[str, Any]:
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=max(1, int(days)))
    convs = payload if isinstance(payload, list) else (payload.get("conversations", []) if isinstance(payload, dict) else [])

    conversations_in_window = 0
    messages_in_window = 0
    est_input_tokens = 0
    est_output_tokens = 0

    for conv in convs:
        if not isinstance(conv, dict):
            continue
        ts = _parse_iso_datetime(conv.get("updated_at") or conv.get("created_at"))
        if ts is not None and ts < cutoff:
            continue
        msgs = conv.get("chat_messages") or conv.get("messages") or []
        if not isinstance(msgs, list):
            continue
        conversations_in_window += 1
        running_context_chars = 0
        for msg in msgs:
            if not isinstance(msg, dict):
                continue
            messages_in_window += 1
            text = str(msg.get("text") or "")
            if not text and isinstance(msg.get("content"), list):
                text = " ".join(
                    str(blk.get("text") or "")
                    for blk in msg["content"]
                    if isinstance(blk, dict)
                )
            sender = str(msg.get("sender") or msg.get("role") or "").lower()
            char_len = len(text)
            running_context_chars += char_len
            if sender in ("assistant", "model"):
                est_output_tokens += max(1, char_len // 4) if char_len else 0
                est_input_tokens += max(1, running_context_chars // 4) if running_context_chars else 0

    output_price_per_m = price_per_m * 5.0
    est_spend_usd = round(
        (est_input_tokens / 1_000_000.0) * price_per_m
        + (est_output_tokens / 1_000_000.0) * output_price_per_m,
        2,
    )
    return {
        "ok": True,
        "source": "claude_export_json",
        "label": source_label,
        "window_days": days,
        "conversations": conversations_in_window,
        "messages": messages_in_window,
        "input_tokens": est_input_tokens,
        "output_tokens": est_output_tokens,
        "total_tokens": est_input_tokens + est_output_tokens,
        "estimated_spend_usd": est_spend_usd,
    }


def _parse_billing_csv(
    csv_text: str,
    source_label: str,
    days: int = 30,
    price_per_m: float = 5.0,
) -> Dict[str, Any]:
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=max(1, int(days)))
    reader = csv.DictReader(io.StringIO(csv_text))
    rows_matched = 0
    input_tokens = 0
    output_tokens = 0
    cache_read_tokens = 0
    explicit_cost_usd = 0.0
    has_explicit_cost = False

    def _num(row: Dict[str, Any], keys: tuple[str, ...]) -> float:
        for k in keys:
            for rk, rv in row.items():
                if rk and rk.strip().lower() == k and rv not in (None, ""):
                    cleaned = str(rv).strip().replace("$", "").replace(",", "")
                    try:
                        return float(cleaned)
                    except ValueError:
                        return 0.0
        return 0.0

    def _date_val(row: Dict[str, Any]) -> Optional[dt.datetime]:
        for k in ("date", "usage_date", "timestamp", "created_at", "bucket_start", "day"):
            for rk, rv in row.items():
                if rk and rk.strip().lower() == k and rv:
                    parsed = _parse_iso_datetime(rv)
                    if parsed is not None:
                        return parsed
        return None

    for row in reader:
        if not isinstance(row, dict):
            continue
        row_dt = _date_val(row)
        if row_dt is not None and row_dt < cutoff:
            continue
        rows_matched += 1
        in_t = int(_num(row, ("input_tokens", "uncached_input_tokens", "prompt_tokens")))
        out_t = int(_num(row, ("output_tokens", "completion_tokens")))
        cr_t = int(_num(row, ("cache_read_input_tokens", "cached_input_tokens")))
        cost_v = _num(row, ("cost_usd", "total_cost_usd", "cost", "amount_usd", "amount", "usd"))
        input_tokens += in_t
        output_tokens += out_t
        cache_read_tokens += cr_t
        if cost_v > 0:
            explicit_cost_usd += cost_v
            has_explicit_cost = True

    if has_explicit_cost:
        spend_usd = round(explicit_cost_usd, 2)
    else:
        output_price_per_m = price_per_m * 5.0
        spend_usd = round(
            (input_tokens / 1_000_000.0) * price_per_m
            + (cache_read_tokens / 1_000_000.0) * (price_per_m * 0.1)
            + (output_tokens / 1_000_000.0) * output_price_per_m,
            2,
        )

    return {
        "ok": True,
        "source": "billing_csv",
        "label": source_label,
        "window_days": days,
        "rows": rows_matched,
        "input_tokens": input_tokens,
        "cache_read_tokens": cache_read_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + cache_read_tokens + output_tokens,
        "estimated_spend_usd": spend_usd,
        "exact_cost_reported": has_explicit_cost,
    }


def parse_cloud_export(
    file_path: str | Path,
    days: int = 30,
    price_per_m: float = 5.0,
) -> Dict[str, Any]:
    """Parse an offline Claude.ai export (`conversations.json` / `.zip`) or Console billing `.csv`."""
    p = Path(file_path).expanduser().resolve()
    safe_label = _sanitize_terminal_text(p.name or str(file_path))
    if not p.is_file():
        return {"ok": False, "source": "import", "label": safe_label, "error": f"File not found: {safe_label}"}

    max_bytes = _get_max_export_bytes()
    try:
        file_sz = p.stat().st_size
    except OSError as exc:
        return {"ok": False, "source": "import", "label": safe_label, "error": _sanitize_terminal_text(exc)}

    if file_sz > max_bytes:
        return {
            "ok": False,
            "source": "import",
            "label": safe_label,
            "error": f"Export file exceeds safety limit ({file_sz} > {max_bytes} bytes)",
        }

    suffix = p.suffix.lower()
    try:
        if suffix == ".zip":
            with zipfile.ZipFile(p, "r") as zf:
                conv_member = next(
                    (
                        name
                        for name in zf.namelist()
                        if (name == "conversations.json" or name.endswith("/conversations.json"))
                        and ".." not in Path(name).parts
                    ),
                    None,
                )
                if not conv_member:
                    return {
                        "ok": False,
                        "source": "import",
                        "label": safe_label,
                        "error": "No conversations.json found inside zip archive",
                    }
                zinfo = zf.getinfo(conv_member)
                if zinfo.file_size > max_bytes:
                    return {
                        "ok": False,
                        "source": "import",
                        "label": safe_label,
                        "error": f"Compressed member exceeds safety limit ({zinfo.file_size} > {max_bytes} bytes)",
                    }
                with zf.open(conv_member, "r") as member_fp:
                    raw_bytes = member_fp.read(max_bytes + 1)
                if len(raw_bytes) > max_bytes:
                    return {
                        "ok": False,
                        "source": "import",
                        "label": safe_label,
                        "error": f"Decompressed stream exceeds safety limit ({max_bytes} bytes)",
                    }
                payload = json.loads(raw_bytes.decode("utf-8", errors="ignore"))
            return _parse_claude_conversations_payload(payload, safe_label, days=days, price_per_m=price_per_m)

        if suffix == ".csv":
            txt = p.read_text(encoding="utf-8", errors="ignore")
            return _parse_billing_csv(txt, safe_label, days=days, price_per_m=price_per_m)

        # Default: JSON (Claude conversations.json)
        payload = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        return _parse_claude_conversations_payload(payload, safe_label, days=days, price_per_m=price_per_m)
    except Exception as exc:
        return {"ok": False, "source": "import", "label": safe_label, "error": _sanitize_terminal_text(exc)}


def fetch_anthropic_cloud_usage(
    admin_api_key: str,
    days: int = 30,
    timeout: float = 15.0,
) -> Dict[str, Any]:
    """
    Fetch organization-level cloud token usage & USD cost from Anthropic's Admin API:
      - GET /v1/organizations/usage_report/messages
      - GET /v1/organizations/cost_report
    """
    key = (admin_api_key or "").strip()
    if not key:
        return {
            "ok": False,
            "source": "anthropic_admin_api",
            "label": "Anthropic Admin API",
            "error": "Missing ANTHROPIC_ADMIN_API_KEY",
        }

    now = dt.datetime.now(dt.timezone.utc)
    start = now - dt.timedelta(days=max(1, int(days)))
    qs = urllib.parse.urlencode(
        {
            "starting_at": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ending_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "bucket_width": "1d",
        }
    )
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "accept": "application/json",
    }

    def _get_json(url: str) -> Dict[str, Any]:
        parsed_u = urllib.parse.urlparse(url)
        if parsed_u.scheme != "https" or parsed_u.hostname != "api.anthropic.com":
            raise ValueError("Untrusted Anthropic Admin API URL")
        req = urllib.request.Request(url, headers=headers, method="GET")  # noqa: S310
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read(MAX_ADMIN_API_RESPONSE_BYTES + 1)
        if len(raw) > MAX_ADMIN_API_RESPONSE_BYTES:
            raise ValueError("Anthropic Admin API response exceeded safety limit")
        return json.loads(raw.decode("utf-8", errors="ignore"))

    input_tokens = 0
    cache_read_tokens = 0
    output_tokens = 0
    cost_usd = 0.0
    errors: List[str] = []

    try:
        usage_data = _get_json(f"https://api.anthropic.com/v1/organizations/usage_report/messages?{qs}")
        for bucket in usage_data.get("data", []) or []:
            for res in bucket.get("results", [bucket]) or []:
                if not isinstance(res, dict):
                    continue
                input_tokens += int(res.get("uncached_input_tokens", 0) or res.get("input_tokens", 0) or 0)
                cache_read_tokens += int(res.get("cache_read_input_tokens", 0) or 0)
                output_tokens += int(res.get("output_tokens", 0) or 0)
    except (urllib.error.URLError, urllib.error.HTTPError, Exception) as exc:
        errors.append(f"usage_report: {_sanitize_terminal_text(exc)}")

    try:
        cost_data = _get_json(f"https://api.anthropic.com/v1/organizations/cost_report?{qs}")
        for bucket in cost_data.get("data", []) or []:
            for res in bucket.get("results", [bucket]) or []:
                if not isinstance(res, dict):
                    continue
                amt = res.get("amount", 0)
                try:
                    # Anthropic cost_report returns decimal USD string or cents depending on version
                    cost_usd += float(amt)
                except (TypeError, ValueError):
                    pass
    except (urllib.error.URLError, urllib.error.HTTPError, Exception) as exc:
        errors.append(f"cost_report: {_sanitize_terminal_text(exc)}")

    if len(errors) == 2:
        return {
            "ok": False,
            "source": "anthropic_admin_api",
            "label": "Anthropic Admin API",
            "window_days": days,
            "error": "; ".join(errors),
        }

    return {
        "ok": True,
        "source": "anthropic_admin_api",
        "label": "Anthropic Admin API",
        "window_days": days,
        "input_tokens": input_tokens,
        "cache_read_tokens": cache_read_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + cache_read_tokens + output_tokens,
        "estimated_spend_usd": round(cost_usd, 2),
        "exact_cost_reported": cost_usd > 0,
        "warnings": errors,
    }


def format_cloud_audit_section(cloud_sources: List[Dict[str, Any]]) -> str:
    """Render opt-in cloud usage & spend lines for `body audit` terminal output."""
    if not cloud_sources:
        return ""
    lines = [
        "",
        "☁️  Cloud Usage & Spend (Opt-In - Reported Separately from Local Transcripts):",
    ]
    for src in cloud_sources:
        lbl = _sanitize_terminal_text(src.get("label", src.get("source", "cloud")))
        if not src.get("ok"):
            err_clean = _sanitize_terminal_text(src.get("error", "unknown error"))
            lines.append(f"  • {lbl}: [SKIPPED] {err_clean}")
            continue
        tot_tok = int(src.get("total_tokens", 0))
        usd = float(src.get("estimated_spend_usd", 0.0))
        extra = ""
        if "conversations" in src:
            extra = f" | {int(src['conversations']):,} convs, {int(src.get('messages', 0)):,} msgs"
        elif "rows" in src:
            extra = f" | {int(src['rows']):,} rows"
        cost_tag = "exact" if src.get("exact_cost_reported") else "est."
        lines.append(
            f"  • {lbl}: {tot_tok:,} tokens ({cost_tag} ${usd:,.2f}){extra}"
        )
    return "\n".join(lines)
