"""
Opt-in Cloud Usage & Spend Audit helpers for `body audit` / `body audit-spend`.

Supports three 100% opt-in cloud data sources without compromising local-first privacy:
  1. Offline file import (`--import <path>` or `--import auto`):
     - Claude.ai Data Export (`conversations.json` or `.zip` containing `conversations.json`)
       with per-thread Context Saturation Index (CSI), bloat triggers, and quota-stretch math
     - Anthropic Console billing / usage `.csv` with cache-hit & savable spend attribution
  2. Auto-Downloads sweep (`--auto-downloads`):
     - Opt-in scan of recent (`<= 14d`) Claude `.zip` / `conversations.json` & Console `.csv`
       exports in `~/Downloads` so users don't have to locate or unzip files manually
  3. Anthropic Organization Admin API (`--cloud` / `--anthropic-admin-key`):
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
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from whole_body_compute.finops_audit import calculate_csi, resolve_flush_step_interval
from whole_body_compute.redaction import redact_secrets

DEFAULT_MAX_CLOUD_EXPORT_BYTES = 256 * 1024 * 1024  # 256 MiB safety cap against zip/JSON bombs
MAX_ADMIN_API_RESPONSE_BYTES = 16 * 1024 * 1024  # 16 MiB cap per API response
MAX_CONTEXT_WINDOW_TOKENS = 200_000
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


def discover_recent_cloud_exports(
    downloads_dir: Optional[Path] = None,
    max_age_days: int = 14,
) -> List[Path]:
    """
    Opt-in helper (`--auto-downloads` / `--import auto`): locate recent Claude.ai data
    exports (`.zip` containing `conversations.json`, or `conversations.json`) and Anthropic
    Console billing/usage `.csv` files in `~/Downloads` without decompressing unrelated archives.
    """
    env_dl = os.environ.get("WBC_DOWNLOADS_DIR", "").strip()
    dl_root = Path(downloads_dir or env_dl or (Path.home() / "Downloads")).expanduser()
    if not dl_root.is_dir():
        return []

    now = time.time()
    max_age_sec = max(1, int(max_age_days)) * 86400.0
    max_bytes = _get_max_export_bytes()
    candidates: List[tuple[float, Path]] = []

    try:
        entries = list(dl_root.iterdir())
    except OSError:
        return []

    for entry in entries:
        try:
            if not entry.is_file() or entry.name.startswith("."):
                continue
            st = entry.stat()
            if st.st_size <= 0 or st.st_size > max_bytes:
                continue
            if (now - st.st_mtime) > max_age_sec:
                continue

            low_name = entry.name.lower()
            suf = entry.suffix.lower()
            if low_name == "conversations.json":
                candidates.append((st.st_mtime, entry))
            elif suf == ".zip":
                try:
                    with zipfile.ZipFile(entry, "r") as zf:
                        has_convs = any(
                            (n == "conversations.json" or n.endswith("/conversations.json"))
                            and ".." not in Path(n).parts
                            for n in zf.namelist()[:200]
                        )
                    if has_convs:
                        candidates.append((st.st_mtime, entry))
                except Exception:
                    continue
            elif suf == ".csv" and st.st_size <= 32 * 1024 * 1024:
                with open(entry, "r", encoding="utf-8", errors="ignore") as fh:
                    header_line = fh.readline(4096).lower()
                cols = {c.strip().strip('"') for c in header_line.split(",") if c.strip()}
                has_tok_cols = bool(
                    cols & {"input_tokens", "uncached_input_tokens", "cache_read_input_tokens", "prompt_tokens"}
                ) and bool(
                    cols & {"output_tokens", "completion_tokens", "cost_usd", "total_cost_usd", "amount_usd", "cost", "amount"}
                )
                has_console_cost_cols = (
                    any(k in low_name for k in ("anthropic", "claude"))
                    and bool(cols & {"cost_usd", "total_cost_usd", "amount_usd", "cost", "amount", "usd"})
                    and bool(cols & {"date", "usage_date", "timestamp", "created_at", "bucket_start", "day", "model", "workspace"})
                )
                if has_tok_cols or has_console_cost_cols:
                    candidates.append((st.st_mtime, entry))
        except OSError:
            continue

    candidates.sort(key=lambda item: item[0], reverse=True)
    # Keep at most the newest Claude export and newest billing CSV unless distinct types exist
    picked: List[Path] = []
    seen_kinds: set[str] = set()
    for _mtime, path in candidates:
        kind = "csv" if path.suffix.lower() == ".csv" else "claude_chat"
        if kind not in seen_kinds:
            seen_kinds.add(kind)
            picked.append(path)
    return picked


def _extract_claude_chat_message_parts(msg: Dict[str, Any]) -> tuple[str, int, bool]:
    """Return `(combined_text, extra_attachment_bytes, used_tools_or_attachments)` for a Claude.ai message."""
    chunks: List[str] = []
    used_tools = False
    base_txt = str(msg.get("text") or "")
    if base_txt:
        chunks.append(base_txt)

    content_list = msg.get("content")
    if isinstance(content_list, list):
        for blk in content_list:
            if not isinstance(blk, dict):
                continue
            btype = str(blk.get("type") or "").lower()
            if btype in ("tool_use", "tool_result", "mcp_tool_use", "mcp_tool_result"):
                used_tools = True
            b_txt = str(blk.get("text") or blk.get("content") or "")
            if b_txt and b_txt != base_txt:
                chunks.append(b_txt)
            elif btype in ("tool_use", "tool_result") and blk.get("input"):
                chunks.append(json.dumps(blk.get("input")))

    extra_bytes = 0
    for att_key in ("attachments", "files", "files_v2"):
        atts = msg.get(att_key)
        if isinstance(atts, list) and atts:
            used_tools = True
            for att in atts:
                if isinstance(att, dict):
                    ext_c = str(att.get("extracted_content") or att.get("content") or "")
                    if ext_c:
                        extra_bytes += len(ext_c.encode("utf-8"))
                    elif att.get("file_size"):
                        try:
                            extra_bytes += min(200_000, int(att["file_size"]))
                        except (TypeError, ValueError):
                            pass

    joined = " ".join(c for c in chunks if c)
    return joined, extra_bytes, used_tools


def _parse_claude_conversations_payload(
    payload: Any,
    source_label: str,
    days: int = 30,
    price_per_m: float = 5.0,
) -> Dict[str, Any]:
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=max(1, int(days)))
    convs = payload if isinstance(payload, list) else (payload.get("conversations", []) if isinstance(payload, dict) else [])
    flush_steps = resolve_flush_step_interval()
    output_price_per_m = price_per_m * 5.0

    conversations_in_window = 0
    messages_in_window = 0
    est_input_tokens = 0
    est_output_tokens = 0
    wasted_bloat_tokens = 0
    bbb_optimized_tokens = 0
    csi_red_threads = 0
    csi_yellow_threads = 0
    marathon_threads = 0
    thread_summaries: List[Dict[str, Any]] = []

    for conv in convs:
        if not isinstance(conv, dict):
            continue
        ts = _parse_iso_datetime(conv.get("updated_at") or conv.get("created_at"))
        if ts is not None and ts < cutoff:
            continue
        msgs = conv.get("chat_messages") or conv.get("messages") or []
        if not isinstance(msgs, list) or not msgs:
            continue

        valid_msgs = [m for m in msgs if isinstance(m, dict)]
        if not valid_msgs:
            continue

        conversations_in_window += 1
        running_context_chars = 0
        running_unshielded_tok = 0
        running_flushed_sliced_tok = 0
        conv_in_tok = 0
        conv_out_tok = 0
        conv_opt_in_tok = 0
        conv_bytes = 0
        conv_trunc_steps = 0
        conv_used_tools = False
        assistant_turns = 0

        for step_idx, msg in enumerate(valid_msgs, start=1):
            messages_in_window += 1
            text, att_bytes, msg_tools = _extract_claude_chat_message_parts(msg)
            if msg_tools:
                conv_used_tools = True
            sender = str(msg.get("sender") or msg.get("role") or "").lower()
            text_bytes = len(text.encode("utf-8"))
            eff_bytes = text_bytes + att_bytes
            conv_bytes += eff_bytes
            if eff_bytes > 16_000:
                conv_trunc_steps += 1

            char_len = len(text) + att_bytes
            running_context_chars += char_len
            step_tok = max(1, eff_bytes // 4) if eff_bytes > 0 else 0
            if eff_bytes > 2000:
                base_cap = min(step_tok, 500)
                compressed_tok = base_cap + int(round(max(0, step_tok - base_cap) * 0.10))
            else:
                compressed_tok = step_tok

            running_unshielded_tok = min(MAX_CONTEXT_WINDOW_TOKENS, running_unshielded_tok + step_tok)
            if (step_idx > flush_steps and (step_idx - 1) % flush_steps == 0) or (
                conv_bytes >= 120_000 and step_idx > 25 and (step_idx - 1) % 25 == 0
            ):
                running_flushed_sliced_tok = 450 + compressed_tok
            else:
                running_flushed_sliced_tok = min(
                    MAX_CONTEXT_WINDOW_TOKENS, running_flushed_sliced_tok + compressed_tok
                )

            if sender in ("assistant", "model"):
                assistant_turns += 1
                out_t = max(1, len(text) // 4) if text else 0
                in_t = min(MAX_CONTEXT_WINDOW_TOKENS, max(1, running_context_chars // 4)) if running_context_chars else 0
                est_output_tokens += out_t
                est_input_tokens += in_t
                conv_out_tok += out_t
                conv_in_tok += in_t
                conv_opt_in_tok += min(in_t, max(1, running_flushed_sliced_tok))

        if assistant_turns == 0 and conv_bytes > 0:
            conv_in_tok = max(1, conv_bytes // 4)
            conv_opt_in_tok = conv_in_tok

        conv_wasted_tok = max(0, conv_in_tok - conv_opt_in_tok)
        wasted_bloat_tokens += conv_wasted_tok
        bbb_optimized_tokens += conv_opt_in_tok

        full_kb = round(conv_bytes / 1024.0, 1)
        csi, tier = calculate_csi(
            steps=len(valid_msgs),
            full_size_kb=full_kb,
            truncated_steps=conv_trunc_steps,
        )
        if tier == "RED":
            csi_red_threads += 1
        elif tier == "YELLOW":
            csi_yellow_threads += 1
        if assistant_turns >= 25:
            marathon_threads += 1

        if conv_trunc_steps > 0:
            trigger = "TRUNCATION_CASCADE"
        elif len(valid_msgs) > 25:
            trigger = "QUADRATIC_TURN_TAX"
        elif conv_used_tools and conv_wasted_tok > 0:
            trigger = "UNBOUNDED_TOOL_DUMP"
        else:
            trigger = "NOMINAL"

        conv_spend_usd = round(
            (conv_in_tok / 1_000_000.0) * price_per_m
            + (conv_out_tok / 1_000_000.0) * output_price_per_m,
            2,
        )
        conv_savable_usd = round((conv_wasted_tok / 1_000_000.0) * price_per_m, 2)
        cid = _sanitize_terminal_text(conv.get("uuid") or conv.get("id") or f"chat-{conversations_in_window}")
        raw_name = _sanitize_terminal_text(conv.get("name") or conv.get("title") or "")[:40]
        thread_summaries.append(
            {
                "conv_id": cid,
                "short_id": cid[:8],
                "title": raw_name,
                "steps": len(valid_msgs),
                "turns": assistant_turns,
                "transcript_kb": full_kb,
                "csi": csi,
                "health_tier": tier,
                "bloat_trigger": trigger,
                "input_tokens": conv_in_tok,
                "wasted_bloat_tokens": conv_wasted_tok,
                "estimated_spend_usd": conv_spend_usd,
                "savable_usd": conv_savable_usd,
            }
        )

    thread_summaries.sort(key=lambda r: (r["estimated_spend_usd"], r["wasted_bloat_tokens"]), reverse=True)
    est_spend_usd = round(
        (est_input_tokens / 1_000_000.0) * price_per_m
        + (est_output_tokens / 1_000_000.0) * output_price_per_m,
        2,
    )
    savable_usd = round((wasted_bloat_tokens / 1_000_000.0) * price_per_m, 2)
    bloat_pct = round((wasted_bloat_tokens / float(max(1, est_input_tokens))) * 100.0, 1) if est_input_tokens > 0 else 0.0
    quota_stretch = (
        round(float(est_input_tokens) / float(max(1, bbb_optimized_tokens)), 2)
        if est_input_tokens > 0 and bbb_optimized_tokens > 0
        else 1.0
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
        "wasted_bloat_tokens": wasted_bloat_tokens,
        "bbb_optimized_tokens": bbb_optimized_tokens,
        "avoidable_bloat_pct": bloat_pct,
        "quota_stretch_multiple": quota_stretch,
        "csi_red_threads": csi_red_threads,
        "csi_yellow_threads": csi_yellow_threads,
        "marathon_threads": marathon_threads,
        "estimated_spend_usd": est_spend_usd,
        "estimated_wbc_savable_usd": savable_usd,
        "top_bloated_conversations": thread_summaries[:5],
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
    cache_create_tokens = 0
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
        cc_t = int(_num(row, ("cache_creation_input_tokens", "cache_write_input_tokens")))
        cost_v = _num(row, ("cost_usd", "total_cost_usd", "cost", "amount_usd", "amount", "usd"))
        input_tokens += in_t
        output_tokens += out_t
        cache_read_tokens += cr_t
        cache_create_tokens += cc_t
        if cost_v > 0:
            explicit_cost_usd += cost_v
            has_explicit_cost = True

    output_price_per_m = price_per_m * 5.0
    modeled_in_usd = (
        (input_tokens / 1_000_000.0) * price_per_m
        + (cache_create_tokens / 1_000_000.0) * (price_per_m * 1.25)
        + (cache_read_tokens / 1_000_000.0) * (price_per_m * 0.1)
    )
    modeled_out_usd = (output_tokens / 1_000_000.0) * output_price_per_m
    if has_explicit_cost:
        spend_usd = round(explicit_cost_usd, 2)
    else:
        spend_usd = round(modeled_in_usd + modeled_out_usd, 2)

    tot_in_pool = input_tokens + cache_create_tokens + cache_read_tokens
    cache_hit_pct = round((cache_read_tokens / float(max(1, tot_in_pool))) * 100.0, 1) if tot_in_pool > 0 else 0.0
    # Estimate WBC savings on Console API traffic: auto-ephemeral prompt caching on uncached input + BBB slicing
    if tot_in_pool > 0:
        in_share = modeled_in_usd / max(0.01, modeled_in_usd + modeled_out_usd)
        eff_in_spend = spend_usd * in_share
        save_ratio = 0.62 if cache_hit_pct < 35.0 else 0.42
        est_savable_usd = round(eff_in_spend * save_ratio, 2)
    else:
        est_savable_usd = round(spend_usd * 0.35, 2) if spend_usd > 0 else 0.0

    return {
        "ok": True,
        "source": "billing_csv",
        "label": source_label,
        "window_days": days,
        "rows": rows_matched,
        "input_tokens": input_tokens,
        "cache_create_tokens": cache_create_tokens,
        "cache_read_tokens": cache_read_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + cache_create_tokens + cache_read_tokens + output_tokens,
        "cache_hit_rate_pct": cache_hit_pct,
        "estimated_spend_usd": spend_usd,
        "estimated_wbc_savable_usd": est_savable_usd,
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

    tot_in_pool = input_tokens + cache_read_tokens
    cache_hit_pct = round((cache_read_tokens / float(max(1, tot_in_pool))) * 100.0, 1) if tot_in_pool > 0 else 0.0

    return {
        "ok": True,
        "source": "anthropic_admin_api",
        "label": "Anthropic Admin API",
        "window_days": days,
        "input_tokens": input_tokens,
        "cache_read_tokens": cache_read_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + cache_read_tokens + output_tokens,
        "cache_hit_rate_pct": cache_hit_pct,
        "estimated_spend_usd": round(cost_usd, 2),
        "exact_cost_reported": cost_usd > 0,
        "warnings": errors,
    }


def format_cloud_audit_section(cloud_sources: List[Dict[str, Any]]) -> str:
    """Render opt-in cloud usage, CSI saturation & spend lines for `body audit` terminal output."""
    if not cloud_sources:
        return ""
    lines = [
        "",
        "☁️  Cloud & Anthropic Console Audit (Opt-In):",
    ]
    for src in cloud_sources:
        lbl = _sanitize_terminal_text(src.get("label", src.get("source", "cloud")))
        if not src.get("ok"):
            err_clean = _sanitize_terminal_text(src.get("error", "unknown error"))
            lines.append(f"  • {lbl}: [SKIPPED] {err_clean}")
            continue
        tot_tok = int(src.get("total_tokens", 0))
        usd = float(src.get("estimated_spend_usd", 0.0))
        sav_usd = float(src.get("estimated_wbc_savable_usd", 0.0))
        extra = ""
        if "conversations" in src:
            extra = f" | {int(src['conversations']):,} convs, {int(src.get('messages', 0)):,} msgs"
        elif "rows" in src:
            extra = f" | {int(src['rows']):,} rows"
        cost_tag = "exact" if src.get("exact_cost_reported") else "est."
        lines.append(
            f"  • {lbl}: {tot_tok:,} tokens ({cost_tag} ${usd:,.2f}){extra}"
        )
        if src.get("source") == "claude_export_json" and int(src.get("conversations", 0)) > 0:
            bloat_pct = float(src.get("avoidable_bloat_pct", 0.0))
            q_mult = float(src.get("quota_stretch_multiple", 1.0))
            red_n = int(src.get("csi_red_threads", 0))
            yel_n = int(src.get("csi_yellow_threads", 0))
            mar_n = int(src.get("marathon_threads", 0))
            lines.append(
                f"    ├─ Context Bloat & Quota: {bloat_pct:.1f}% avoidable re-reads • ~{q_mult:.2f}x Pro quota stretch (est. save ${sav_usd:,.2f})"
            )
            lines.append(
                f"    └─ CSI Health Breakdown:  {red_n} RED (CSI >= 1.0) • {yel_n} YELLOW (CSI >= 0.7) • {mar_n} marathons (25+ turns)"
            )
            for idx, tc in enumerate(src.get("top_bloated_conversations", [])[:3], start=1):
                t_lbl = f' "{tc["title"]}"' if tc.get("title") else ""
                lines.append(
                    f"       {idx}. [{tc['short_id']}] CSI {tc['csi']:.2f} ({tc['health_tier']}) • {tc['steps']} msgs • {tc['bloat_trigger']}{t_lbl}"
                )
        elif src.get("source") in ("billing_csv", "anthropic_admin_api") and "cache_hit_rate_pct" in src:
            ch_pct = float(src.get("cache_hit_rate_pct", 0.0))
            sav_note = f" • est. WBC save ~${sav_usd:,.2f}" if sav_usd > 0 else ""
            lines.append(
                f"    └─ Console Cache Efficiency: {ch_pct:.1f}% cached input hit rate{sav_note}"
            )
    return "\n".join(lines)


def sync_edge_proxy_receipts(
    license_key: str,
    *,
    edge_base_url: str = "https://gateway.wholebodycompute.com",
    db_path: Optional[Path | str] = None,
    timeout_s: float = 5.0,
) -> Dict[str, Any]:
    """
    Pull numeric usage receipts from `gateway.wholebodycompute.com/v1/receipts/pull`,
    record them into local `proxy_finops_ledger`, and acknowledge deletion via `/v1/receipts/ack`.
    """
    key = (license_key or "").strip()
    if not key:
        return {"ok": False, "error": "missing_license_key", "synced": 0}

    base = edge_base_url.rstrip("/")
    pull_req = urllib.request.Request(
        f"{base}/v1/receipts/pull",
        headers={"x-wbc-license": key, "Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(pull_req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as exc:
        return {"ok": False, "error": f"pull_failed: {exc}", "synced": 0}

    receipts = payload.get("receipts") or []
    if not isinstance(receipts, list) or not receipts:
        return {"ok": True, "synced": 0, "deleted": 0}

    from whole_body_compute.proxy.pricing import compute_proxy_call_cost
    from whole_body_compute.shield import record_proxy_call

    synced_ids: List[str] = []
    for rec in receipts:
        if not isinstance(rec, dict):
            continue
        model = str(rec.get("model") or "claude-sonnet-4-5")
        u_in = int(rec.get("uncached_in") or 0)
        cw = int(rec.get("cache_write") or 0)
        cr = int(rec.get("cache_read") or 0)
        out = int(rec.get("out_tok") or 0)
        cost = compute_proxy_call_cost(
            model,
            uncached_input_tokens=u_in,
            cache_write_tokens=cw,
            cache_read_tokens=cr,
            output_tokens=out,
        )
        ok = record_proxy_call(
            db_path=db_path,
            client_slug=str(rec.get("client_slug") or "untagged"),
            provider="edge",
            requested_model=model,
            routed_model=model,
            route_mode="edge_cache",
            cache_breakpoints_injected=1 if (cw > 0 or cr > 0) else 0,
            uncached_input_tokens=u_in,
            cache_write_tokens=cw,
            cache_read_tokens=cr,
            output_tokens=out,
            baseline_usd=cost.baseline_uncached_usd,
            actual_usd=cost.actual_billed_usd,
            net_usd_saved=float(rec.get("saved_usd") if rec.get("saved_usd") is not None else cost.net_usd_saved),
            equivalent_input_tokens_saved=cost.equivalent_input_tokens_saved,
            timestamp=(float(rec["ts"]) / 1000.0 if rec.get("ts") and float(rec["ts"]) > 1e12 else rec.get("ts")),
        )
        if ok and rec.get("id"):
            synced_ids.append(str(rec["id"]))

    deleted = 0
    if synced_ids:
        ack_req = urllib.request.Request(
            f"{base}/v1/receipts/ack",
            data=json.dumps({"receipt_ids": synced_ids}).encode("utf-8"),
            headers={
                "x-wbc-license": key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(ack_req, timeout=timeout_s) as ack_resp:
                ack_data = json.loads(ack_resp.read().decode("utf-8", errors="replace"))
                deleted = int(ack_data.get("deleted") or len(synced_ids))
        except Exception:
            pass

    return {"ok": True, "synced": len(synced_ids), "deleted": deleted}


def format_proxy_client_receipt(
    summary: Dict[str, Any],
    *,
    clients_cfg: Optional[Dict[str, Dict[str, Any]]] = None,
    markdown: bool = False,
) -> str:
    """Render per-client FinOps Proxy cost, savings, and retainer margin breakdown."""
    clients = summary.get("clients") or []
    cfg_map = clients_cfg or {}
    if markdown:
        lines = [
            f"### FinOps API Proxy - Per-Client Receipt (`{summary.get('window', 'all')}`)",
            "",
            "| Client | Calls | Baseline USD | Actual USD | Net Saved | Retainer Margin |",
            "| :--- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for c in clients:
            slug = str(c.get("client_slug", "untagged"))
            retainer = float((cfg_map.get(slug) or {}).get("monthly_retainer_usd") or 0.0)
            margin_s = (
                f"{((retainer - float(c.get('actual_usd', 0.0))) / retainer) * 100.0:.1f}%"
                if retainer > 0.0
                else " - "
            )
            lines.append(
                f"| `{slug}` | {int(c.get('calls', 0)):,} | ${float(c.get('baseline_usd', 0.0)):,.4f} | "
                f"${float(c.get('actual_usd', 0.0)):,.4f} | ${float(c.get('net_usd_saved', 0.0)):,.4f} | {margin_s} |"
            )
        return "\n".join(lines)

    lines = [
        "",
        f"💸 FinOps API Proxy - Per-Client Receipt ({summary.get('window', 'all')}):",
        f"   Calls: {int(summary.get('total_calls', 0)):,} ({int(summary.get('cached_calls', 0)):,} cached) • "
        f"Actual: ${float(summary.get('actual_usd', 0.0)):,.4f} • Saved: ${float(summary.get('net_usd_saved', 0.0)):,.4f}",
    ]
    for c in clients:
        slug = str(c.get("client_slug", "untagged"))
        retainer = float((cfg_map.get(slug) or {}).get("monthly_retainer_usd") or 0.0)
        margin_note = (
            f" • Margin {((retainer - float(c.get('actual_usd', 0.0))) / retainer) * 100.0:.1f}%"
            if retainer > 0.0
            else ""
        )
        lines.append(
            f"   • {slug:<18} {int(c.get('calls', 0)):>4} calls | "
            f"${float(c.get('actual_usd', 0.0)):>7.4f} billed | "
            f"${float(c.get('net_usd_saved', 0.0)):>7.4f} saved{margin_note}"
        )
    if float(summary.get("missed_usd_community_cap", 0.0)) > 0.0:
        lines.append(
            f"   ⚠️ Community cap missed savings: ${float(summary['missed_usd_community_cap']):,.4f} (upgrade to Pro for unlimited caching)"
        )
    return "\n".join(lines)
