#!/usr/bin/env python3
"""
Whole Body Compute - Zero-Dependency, Zero-Install Multi-Harness FinOps Audit (`docs/audit.py`).

What this script does (100% read-only on your transcripts, macOS & Linux):
  - Zero installs, zero sudo, 100% Python 3 Standard Library only.
  - Zero source code, file paths, prompts, or secrets ever leave your machine.
  - Pass `--offline` or `DO_NOT_TRACK=1` for 100% air-gapped local-only execution.

Scans local coding agent transcripts across all 4 major harnesses:
  1. Google Antigravity (`~/.gemini/antigravity/brain/*/.system_generated/logs/transcript.jsonl`)
  2. Anthropic Claude Code (`~/.claude/projects/**/*.jsonl`)
  3. Cursor Agent / Composer (macOS & Linux `~/.config/Cursor/User/workspaceStorage/*/state.vscdb`)
  4. Block Goose (`~/.config/goose/sessions.db`)

Run via:
  curl -sSL https://wholebodycompute.com/audit.py | python3
  uv run https://wholebodycompute.com/audit.py
  python3 docs/audit.py
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import secrets
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_POSTHOG_HOST = "https://us.i.posthog.com"
DEFAULT_POSTHOG_API_KEY = "phc_D3TxQWXwnd9Z6wBBK3DpnsdKQAHdLpSSMxup4qL8djc2"
WBC_EDGE_AUDIT_SOURCE = ""

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def _visible_len(s: str) -> int:
    clean = _ANSI_ESCAPE_RE.sub("", s)
    w = 0
    for ch in clean:
        cp = ord(ch)
        if ch in ("\ufe0f", "\u200d") or (0x0300 <= cp <= 0x036F):
            continue
        if cp >= 0x1F300 or cp in (0x26A1, 0x2699, 0x2705, 0x274C, 0x2728, 0x23F0, 0x23F3):
            w += 2
        else:
            w += 1
    return w


def _pad_right(s: str, width: int) -> str:
    vis = _visible_len(s)
    if vis >= width:
        return s
    return s + (" " * (width - vis))


def _pad_left(s: str, width: int) -> str:
    vis = _visible_len(s)
    if vis >= width:
        return s
    return (" " * (width - vis)) + s

MAX_CONTEXT_WINDOW_TOKENS = 200_000
DEFAULT_FLUSH_STEP_INTERVAL = 150


def _resolve_flush_step_interval() -> int:
    env_val = os.environ.get("WBC_FLUSH_STEP_INTERVAL")
    if env_val:
        try:
            return max(10, int(env_val))
        except ValueError:
            pass
    return DEFAULT_FLUSH_STEP_INTERVAL


def _resolve_model_rates(
    model_name: str,
    default_in: float,
    default_out: float,
    explicit_override: bool = False,
) -> Tuple[float, float, bool]:
    """Return `(input_usd_per_m, output_usd_per_m, matched_known_model)` for a logged model string."""
    if explicit_override:
        return default_in, default_out, False
    m = (model_name or "").strip().lower()
    if not m or m in ("<synthetic>", "none", "unknown", "inherit", "default"):
        return default_in, default_out, False
    if any(k in m for k in ("flash-lite", "flash_lite", "hydrogen", "gpt-4.1-nano", "gpt-5-nano")):
        return 0.50, 2.50, True
    if any(k in m for k in ("gemini-2.5-flash", "gemini-2.0-flash")):
        return 0.30, 2.50, True
    if any(k in m for k in ("haiku-3-5", "haiku-3.5", "3-5-haiku", "3.5-haiku")):
        return 0.80, 4.00, True
    if any(
        k in m
        for k in ("haiku", "flash", "argon", "gpt-4o-mini", "gpt-4.1-mini", "gpt-5-mini", "o3-mini", "o4-mini")
    ):
        return 1.00, 5.00, True
    if "gpt-5" in m and "pro" not in m:
        return 1.25, 10.00, True
    if "o3" in m and "pro" not in m:
        return 2.00, 8.00, True
    if "gpt-4o" in m:
        return 2.50, 10.00, True
    if any(k in m for k in ("claude-3-opus", "opus-4-2025", "opus-4-0", "opus-4-1")):
        return 15.00, 75.00, True
    if "opus" in m:
        return 5.00, 25.00, True
    if any(
        k in m
        for k in ("sonnet", "carbon", "neon", "gemini-2.5-pro", "gemini-3", "gemini-4", "gemini 3", "gemini 4", "gpt-4.1", "gpt-5", "o3")
    ):
        return 3.00, 15.00, True
    return default_in, default_out, False


_MODEL_SELECTION_RE = re.compile(
    r"Model Selection`\s+from\s+.+?\s+to\s+(.+?)\.(?:\s|$)"
)
_SAFE_MODEL_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:\- ()/]{2,64}$")
_SECRET_LIKE_MODEL_RE = re.compile(
    r"^(?:sk-|ghp_|github_pat_|AKIA|AIza|phc_|xox[baprs]-)", re.IGNORECASE
)
_ROUTINE_OPPORTUNITY_TOOLS = frozenset(
    {
        "read",
        "read_file",
        "view_file",
        "grep",
        "grep_search",
        "find_by_name",
        "glob",
        "ls",
        "list_dir",
        "bash",
        "run_command",
        "execute_command",
        "edit",
        "edit_file",
        "write",
        "write_file",
        "multi_edit",
        "codex_shell",
        "gemini_shell",
        "cursor_composer",
        "cursor_agent",
        "aider_edit",
    }
)


def _extract_model_selection_change(text: str) -> str:
    """Extract clean model name from `<USER_SETTINGS_CHANGE>` text without truncating decimal versions."""
    if "Model Selection" not in text:
        return ""
    mm = _MODEL_SELECTION_RE.search(text)
    if not mm:
        return ""
    cand = mm.group(1).strip()
    if len(cand) < 2 or cand in ("(", ")", "None", "null"):
        return ""
    return cand


def _canonicalize_model_profile(raw_model: str) -> Optional[Dict[str, Any]]:
    """
    Normalize raw model IDs/display labels into zero-PII canonical model buckets and cost tiers:
      - `fast_low`: Haiku, Flash, Hydrogen, Nano, Mini, Local Ollama/MLX
      - `balanced`: Sonnet, Gemini Carbon/Argon/Neon (Low/Med), GPT-4o/4.1/5
      - `heavy_thinking`: Claude Opus, o1/o3-pro, Orchestrator, High/XHigh thinking variants
    """
    raw = str(raw_model or "").strip()
    if not raw or len(raw) < 2:
        return None
    low = raw.lower()
    if low in ("<synthetic>", "none", "unknown", "inherit", "default", "untagged", "auto", "(", ")"):
        return None
    if _SECRET_LIKE_MODEL_RE.match(raw):
        return None

    has_xhigh = any(k in low for k in ("xhigh", "extra-high", "extra_high"))
    has_high = has_xhigh or any(k in low for k in ("(high)", "-high", "_high", " high"))
    has_med = any(k in low for k in ("(medium)", "-medium", "_medium", " medium"))
    has_low = any(k in low for k in ("(low)", "-low", "_low", " low"))

    def _effort_suffix() -> str:
        if has_xhigh:
            return " (XHigh)"
        if has_high:
            return " (High)"
        if has_med:
            return " (Med)"
        if has_low:
            return " (Low)"
        return ""

    def _effort_key() -> str:
        if has_xhigh:
            return "-xhigh"
        if has_high:
            return "-high"
        if has_med:
            return "-medium"
        if has_low:
            return "-low"
        return ""

    if any(k in low for k in ("local", "mlx", "ollama", "qwen", "llama", "deepseek-r1-distill", "tev1")):
        return {
            "key": "local-metal",
            "label": "Local Metal / Open-Weight",
            "tier": "fast_low",
            "family": "local",
            "high_thinking": False,
        }

    # Anthropic Claude family
    if "haiku" in low:
        ver = "4.5" if ("4-5" in low or "4.5" in low) else ("3.5" if ("3-5" in low or "3.5" in low) else "")
        lbl = f"Claude Haiku {ver}".strip()
        return {
            "key": f"claude-haiku-{ver or 'x'}".replace(".", "-"),
            "label": lbl,
            "tier": "fast_low",
            "family": "claude",
            "high_thinking": False,
        }
    if "opus" in low:
        ver = (
            "5.5"
            if ("5-5" in low or "5.5" in low)
            else ("5" if ("opus-5" in low or "opus 5" in low) else ("4.1" if ("4-1" in low or "4.1" in low) else ("4" if ("opus-4" in low or "opus 4" in low) else "3")))
        )
        lbl = f"Claude Opus {ver}{_effort_suffix()}".strip()
        return {
            "key": f"claude-opus-{ver.replace('.', '-')}{_effort_key()}",
            "label": lbl,
            "tier": "heavy_thinking",
            "family": "claude",
            "high_thinking": True,
        }
    if "sonnet" in low:
        ver = (
            "5.5"
            if ("5-5" in low or "5.5" in low)
            else ("5" if ("sonnet-5" in low or "sonnet 5" in low) else ("4.5" if ("4-5" in low or "4.5" in low) else ("4" if ("sonnet-4" in low or "sonnet 4" in low) else "3.7")))
        )
        is_ht = bool(has_high or has_xhigh)
        return {
            "key": f"claude-sonnet-{ver.replace('.', '-')}{_effort_key()}",
            "label": f"Claude Sonnet {ver}{_effort_suffix()}".strip(),
            "tier": "heavy_thinking" if is_ht else "balanced",
            "family": "claude",
            "high_thinking": is_ht,
        }

    # Google Gemini / Antigravity family
    if "orchestrator" in low:
        sub = "Neon" if "neon" in low else "Pro"
        return {
            "key": f"gemini-orchestrator-{sub.lower()}",
            "label": f"Orchestrator ({sub})",
            "tier": "heavy_thinking",
            "family": "gemini",
            "high_thinking": True,
        }
    for codename in ("hydrogen", "neon", "argon", "carbon"):
        if codename in low:
            title_c = codename.capitalize()
            eff_s = _effort_suffix()
            eff_k = _effort_key()
            is_ht = bool(has_high or has_xhigh)
            if codename == "hydrogen" and not is_ht and not has_med:
                tier = "fast_low"
            elif codename == "neon" and has_low:
                tier = "fast_low"
            elif is_ht:
                tier = "heavy_thinking"
            else:
                tier = "balanced"
            return {
                "key": f"gemini-4-{codename}{eff_k}",
                "label": f"Gemini 4 {title_c}{eff_s}",
                "tier": tier,
                "family": "gemini",
                "high_thinking": is_ht,
            }
    if "flash-lite" in low or "flash_lite" in low:
        return {
            "key": "gemini-flash-lite",
            "label": "Gemini Flash-Lite",
            "tier": "fast_low",
            "family": "gemini",
            "high_thinking": False,
        }
    if "flash" in low:
        is_ht = bool(has_high or has_xhigh)
        return {
            "key": f"gemini-flash{_effort_key()}",
            "label": f"Gemini Flash{_effort_suffix()}".strip(),
            "tier": "balanced" if is_ht else "fast_low",
            "family": "gemini",
            "high_thinking": is_ht,
        }
    if "gemini" in low:
        is_ht = bool(has_high or has_xhigh or "ultra" in low)
        ver_m = re.search(r"gemini[- ]?(\d+(?:\.\d+)?)", low)
        ver_s = f" {ver_m.group(1)}" if ver_m else ""
        pro_s = " Pro" if "pro" in low else ""
        return {
            "key": f"gemini{ver_s.replace(' ', '-').replace('.', '-')}{pro_s.lower().replace(' ', '-')}{_effort_key()}",
            "label": f"Gemini{ver_s}{pro_s}{_effort_suffix()}".strip(),
            "tier": "heavy_thinking" if is_ht else "balanced",
            "family": "gemini",
            "high_thinking": is_ht,
        }

    # OpenAI / Codex / Reasoning family
    if any(k in low for k in ("gpt-5-nano", "gpt-4.1-nano", "gpt-4o-mini", "gpt-4.1-mini", "gpt-5-mini", "o3-mini", "o4-mini")):
        clean_k = re.sub(r"[^a-z0-9.\-]+", "-", low).strip("-")[:28]
        return {
            "key": clean_k,
            "label": raw[:28],
            "tier": "fast_low",
            "family": "openai",
            "high_thinking": False,
        }
    if any(k in low for k in ("o1-pro", "o3-pro", "gpt-5-pro")) or (
        any(k in low for k in ("o1", "o3", "gpt-5")) and (has_high or has_xhigh)
    ):
        clean_k = re.sub(r"[^a-z0-9.\-]+", "-", low).strip("-")[:28]
        return {
            "key": clean_k,
            "label": raw[:28],
            "tier": "heavy_thinking",
            "family": "openai",
            "high_thinking": True,
        }
    if any(k in low for k in ("gpt-4", "gpt-5", "o1", "o3", "o4", "codex")):
        clean_k = re.sub(r"[^a-z0-9.\-]+", "-", low).strip("-")[:28]
        return {
            "key": clean_k,
            "label": raw[:28],
            "tier": "balanced",
            "family": "openai",
            "high_thinking": False,
        }

    if not _SAFE_MODEL_TOKEN_RE.match(raw):
        return None
    clean_key = re.sub(r"[^a-z0-9.\-]+", "-", low).strip("-")[:32]
    if not clean_key:
        return None
    return {
        "key": clean_key,
        "label": raw[:28],
        "tier": "heavy_thinking" if (has_high or has_xhigh) else "balanced",
        "family": "other",
        "high_thinking": bool(has_high or has_xhigh),
    }


def _record_turn_model_usage(
    bucket_map: Dict[str, Dict[str, Any]],
    raw_model: str,
    turns: int = 1,
    billed_input_tokens: int = 0,
    cache_create_tokens: int = 0,
    cache_read_tokens: int = 0,
    output_tokens: int = 0,
    fresh_usd: float = 0.0,
    cache_usd: float = 0.0,
    output_usd: float = 0.0,
) -> None:
    prof = _canonicalize_model_profile(raw_model)
    if prof is None:
        return
    mkey = str(prof["key"])
    entry = bucket_map.get(mkey)
    if entry is None:
        entry = {
            "key": mkey,
            "label": prof["label"],
            "tier": prof["tier"],
            "family": prof["family"],
            "high_thinking": bool(prof["high_thinking"]),
            "turns": 0,
            "billed_input_tokens": 0,
            "cache_create_tokens": 0,
            "cache_read_tokens": 0,
            "output_tokens": 0,
            "fresh_input_usd": 0.0,
            "cache_read_usd": 0.0,
            "output_usd": 0.0,
            "spend_usd": 0.0,
        }
        bucket_map[mkey] = entry
    entry["turns"] += int(turns)
    entry["billed_input_tokens"] += int(billed_input_tokens)
    entry["cache_create_tokens"] += int(cache_create_tokens)
    entry["cache_read_tokens"] += int(cache_read_tokens)
    entry["output_tokens"] += int(output_tokens)
    entry["fresh_input_usd"] += float(fresh_usd)
    entry["cache_read_usd"] += float(cache_usd)
    entry["output_usd"] += float(output_usd)
    entry["spend_usd"] += float(fresh_usd + cache_usd + output_usd)


def _estimate_typed_prompt_tokens(raw_prompt: str) -> int:
    """Estimate human-typed prompt tokens, stripping IDE metadata wrappers, compaction summaries, and tool results."""
    txt = str(raw_prompt or "").strip()
    if (
        not txt
        or txt.startswith("output:")
        or txt.startswith("tool_output:")
        or txt.startswith("This session is being continued")
        or txt.startswith("<command-name>")
        or txt.startswith("<local-command-")
    ):
        return 0
    req_match = re.search(r"<USER_REQUEST>\s*(.*?)\s*</USER_REQUEST>", txt, flags=re.DOTALL | re.IGNORECASE)
    if req_match:
        txt = req_match.group(1).strip()
    else:
        txt = re.sub(
            r"<(ADDITIONAL_METADATA|USER_SETTINGS_CHANGE|ide_opened_file|system-reminder)>.*?</\1>",
            " ",
            txt,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()
    if not txt:
        return 0
    return max(1, int(round(len(txt.encode("utf-8")) / 4.0)))


def _clean_session_title(raw_prompt: str, max_len: int = 28) -> str:
    """Extract a clean, single-line human prompt preview stripped of wrapper tags and secrets."""
    if os.environ.get("WBC_AUDIT_REDACT_TITLES") == "1":
        return ""
    txt = str(raw_prompt or "").strip()
    if not txt or txt.startswith("[redacted"):
        return ""
    req_match = re.search(r"<USER_REQUEST>\s*(.*?)\s*</USER_REQUEST>", txt, flags=re.DOTALL | re.IGNORECASE)
    if req_match:
        txt = req_match.group(1)
    else:
        txt = re.sub(
            r"<(ADDITIONAL_METADATA|USER_SETTINGS_CHANGE|ide_opened_file|system-reminder)>.*?</\1>",
            " ",
            txt,
            flags=re.DOTALL | re.IGNORECASE,
        )
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = re.sub(r"^\[From:[^\]]*\]\s*", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"@\[[^\]]+\]", "", txt)
    txt = re.sub(r"\b(?:sk-[A-Za-z0-9_-]{12,}|ghp_[A-Za-z0-9]{12,}|AIza[A-Za-z0-9_-]{12,})\b", "[key]", txt)
    txt = "".join(ch if ord(ch) < 0x2000 else " " for ch in txt)
    txt = " ".join(txt.split()).strip(" \"'`•-")
    if not txt:
        return ""
    if len(txt) > max_len:
        return txt[: max(1, max_len - 3)].rstrip() + "..."
    return txt


def _evaluate_step_sequence(
    step_Tuples: List[Tuple[bool, int, bool]],
    conv_bytes: int,
    short_id: str,
    conv_id: str,
    harness: str,
    mtime: float,
    in_price_per_m: float,
    out_price_per_m: float,
    bloat_trigger_override: Optional[str] = None,
    top_tool_override: str = "run_command",
    first_prompt: str = "",
    exact_usage: Optional[Dict[str, Any]] = None,
    model_usage: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Evaluate a sequence of `(is_planner_turn, step_effective_bytes, is_truncated)` tuples
    using Whole Body Compute's 200k-window-capped FinOps & Blood-Brain Barrier (BBB) model.
    """
    steps = len(step_Tuples)
    planner_turns = sum(1 for is_p, _, _ in step_Tuples if is_p)
    has_planner_turns = planner_turns > 0
    if not has_planner_turns:
        planner_turns = max(1, int(round(steps * 0.38)))

    running_raw_unbounded_ctx = 0
    running_unshielded_ctx = 0
    running_sliced_no_flush_ctx = 0
    running_flushed_sliced_ctx = 0
    running_tool_bloat_ctx = 0
    flush_steps = _resolve_flush_step_interval()

    conv_raw_unbounded_tok = 0
    conv_actual_input_tok = 0
    conv_output_tok = 0
    conv_tool_bloat_tok = 0
    conv_quad_post_sat_tok = 0
    conv_flushed_sliced_cum_tok = 0
    total_bloat_step_tok = 0
    truncated_steps = 0

    for idx, (is_p, eff_bytes, is_trunc) in enumerate(step_Tuples, start=1):
        if is_trunc:
            truncated_steps += 1
        step_tok = max(1, int(round(float(eff_bytes) / 4.0)))
        if is_p:
            conv_output_tok += max(20, step_tok)

        if eff_bytes > 2000 or is_trunc:
            base_cap = min(step_tok, 500)
            excess_tok = max(0, step_tok - base_cap)
            compressed_tok = base_cap + int(round(excess_tok * 0.10))
            bloat_tok = max(0, step_tok - compressed_tok)
        else:
            compressed_tok = step_tok
            bloat_tok = 0
        total_bloat_step_tok += bloat_tok

        running_raw_unbounded_ctx += step_tok
        running_unshielded_ctx = min(MAX_CONTEXT_WINDOW_TOKENS, running_unshielded_ctx + step_tok)
        running_sliced_no_flush_ctx = min(
            MAX_CONTEXT_WINDOW_TOKENS, running_sliced_no_flush_ctx + compressed_tok
        )
        running_tool_bloat_ctx = min(
            MAX_CONTEXT_WINDOW_TOKENS, running_tool_bloat_ctx + bloat_tok
        )

        if (idx > flush_steps and (idx - 1) % flush_steps == 0) or (
            conv_bytes >= 120_000
            and steps <= flush_steps
            and idx > 25
            and (idx - 1) % 25 == 0
        ):
            running_flushed_sliced_ctx = 450 + compressed_tok
        else:
            running_flushed_sliced_ctx = min(
                MAX_CONTEXT_WINDOW_TOKENS, running_flushed_sliced_ctx + compressed_tok
            )

        if is_p or (not has_planner_turns):
            conv_raw_unbounded_tok += running_raw_unbounded_ctx
            conv_actual_input_tok += running_unshielded_ctx
            conv_tool_bloat_tok += running_tool_bloat_ctx
            conv_quad_post_sat_tok += max(0, running_sliced_no_flush_ctx - running_flushed_sliced_ctx)
            conv_flushed_sliced_cum_tok += running_flushed_sliced_ctx

    if not has_planner_turns:
        conv_raw_unbounded_tok = max(1, int(round(conv_raw_unbounded_tok * 0.38)))
        conv_actual_input_tok = max(1, int(round(conv_actual_input_tok * 0.38)))
        conv_tool_bloat_tok = int(round(conv_tool_bloat_tok * 0.38))
        conv_quad_post_sat_tok = int(round(conv_quad_post_sat_tok * 0.38))
        conv_flushed_sliced_cum_tok = max(1, int(round(conv_flushed_sliced_cum_tok * 0.38)))

    if steps <= 3 and conv_bytes < 1000 and top_tool_override in ("none", "") and conv_tool_bloat_tok == 0 and conv_quad_post_sat_tok == 0:
        conv_bbb_opt_tok = conv_actual_input_tok
        conv_bbb_opt_out_tok = conv_output_tok
    else:
        conv_bbb_opt_tok = max(1, min(conv_actual_input_tok, conv_flushed_sliced_cum_tok))
        conv_bbb_opt_out_tok = conv_output_tok
    conv_wasted_tok = max(0, conv_actual_input_tok - conv_bbb_opt_tok)

    raw_components = conv_quad_post_sat_tok + conv_tool_bloat_tok
    if raw_components > conv_wasted_tok and raw_components > 0:
        scale_c = float(conv_wasted_tok) / float(raw_components)
        conv_quad_post_sat_tok = int(round(conv_quad_post_sat_tok * scale_c))
        conv_tool_bloat_tok = max(0, conv_wasted_tok - conv_quad_post_sat_tok)
        conv_local_offload_tok = 0
    else:
        conv_local_offload_tok = max(0, conv_wasted_tok - raw_components)

    conv_cons_frontier_usd = (
        (conv_actual_input_tok / 1_000_000.0) * (in_price_per_m * 0.19)
        + (conv_output_tok / 1_000_000.0) * out_price_per_m
    )
    conv_cons_bbb_usd = (
        (conv_bbb_opt_tok / 1_000_000.0) * (in_price_per_m * 0.19)
        + (conv_bbb_opt_out_tok / 1_000_000.0) * out_price_per_m
    )
    conv_cons_saved_usd = max(0.0, conv_cons_frontier_usd - conv_cons_bbb_usd)

    conv_real_frontier_usd = (
        (conv_actual_input_tok / 1_000_000.0) * (in_price_per_m * 0.32)
        + (conv_output_tok / 1_000_000.0) * out_price_per_m
    )
    conv_real_bbb_usd = (
        (conv_bbb_opt_tok / 1_000_000.0) * (in_price_per_m * 0.19)
        + (conv_bbb_opt_out_tok / 1_000_000.0) * out_price_per_m
    )
    conv_real_saved_usd = max(0.0, conv_real_frontier_usd - conv_real_bbb_usd)

    conv_uncached_frontier_usd = (
        (conv_actual_input_tok / 1_000_000.0) * in_price_per_m
        + (conv_output_tok / 1_000_000.0) * out_price_per_m
    )
    conv_uncached_bbb_usd = (
        (conv_bbb_opt_tok / 1_000_000.0) * (in_price_per_m * 0.32)
        + (conv_bbb_opt_out_tok / 1_000_000.0) * out_price_per_m
    )
    conv_uncached_saved_usd = max(0.0, conv_uncached_frontier_usd - conv_uncached_bbb_usd)

    full_kb = round(conv_bytes / 1024.0, 1)
    if bloat_trigger_override:
        trigger = bloat_trigger_override
    elif truncated_steps > 0:
        trigger = "TRUNCATION_CASCADE"
    elif steps > 25:
        trigger = "QUADRATIC_TURN_TAX"
    elif conv_tool_bloat_tok > conv_quad_post_sat_tok:
        trigger = "UNBOUNDED_TOOL_DUMP"
    else:
        trigger = "NOMINAL"

    eu = exact_usage or {}
    m_turns = int(eu.get("metered_turns", 0))
    m_in = int(eu.get("billed_input_tokens", 0))
    m_cc = int(eu.get("cache_create_tokens", 0))
    m_cc_1h = int(eu.get("cache_create_1h_tokens", 0))
    m_cc_5m = int(eu.get("cache_create_5m_tokens", max(0, m_cc - m_cc_1h)))
    m_cr = int(eu.get("cache_read_tokens", 0))
    m_out = int(eu.get("output_tokens", 0))
    m_matched = int(eu.get("model_matched_turns", 0))
    m_typed_tok = int(eu.get("typed_prompt_tokens", 0))

    if m_turns > 0 and (m_in + m_cc + m_cr + m_out) > 0:
        # Exact server-side billing telemetry present (e.g., Antigravity / Claude Code)
        if "fresh_input_usd" in eu and "cache_read_usd" in eu and "output_usd" in eu:
            fresh_usd = float(eu["fresh_input_usd"])
            cache_usd = float(eu["cache_read_usd"])
            out_usd = float(eu["output_usd"])
        else:
            fresh_usd = (
                (m_in / 1_000_000.0) * in_price_per_m
                + (m_cc_5m / 1_000_000.0) * (in_price_per_m * 1.25)
                + (m_cc_1h / 1_000_000.0) * (in_price_per_m * 2.00)
            )
            cache_usd = (m_cr / 1_000_000.0) * (in_price_per_m * 0.10)
            out_usd = (m_out / 1_000_000.0) * out_price_per_m
        exact_api_usd = fresh_usd + cache_usd + out_usd
        metered_flag = True
        tot_fresh_tok = max(1, m_in + m_cc)
        if m_typed_tok <= 0:
            m_typed_tok = min(tot_fresh_tok, max(12, planner_turns * 35))
        else:
            m_typed_tok = min(tot_fresh_tok, m_typed_tok)
        typed_prompt_usd = fresh_usd * (float(m_typed_tok) / float(tot_fresh_tok))
    else:
        out_usd = (conv_output_tok / 1_000_000.0) * out_price_per_m
        in_total_usd = max(0.0, conv_cons_frontier_usd - out_usd)
        # Split conservative blended input (0.19x) into ~10% fresh + ~90% cached equivalent
        fresh_usd = in_total_usd * 0.526
        cache_usd = max(0.0, in_total_usd - fresh_usd)
        exact_api_usd = conv_cons_frontier_usd
        metered_flag = False
        m_typed_tok = 0
        typed_prompt_usd = 0.0

    # Per-session projected savings derived directly from this session's simulated context trajectory:
    #   - `fresh_cut`: fraction of step bytes compressible via AST slicing (>2KB tool outputs -> 500-tok outline + 10% tail)
    #   - `cache_cut`: fraction of cumulative re-read history avoided via AST slicing + periodic context flushes
    #   - `out_cut`: output reduction only when multi-turn tool/truncation retries are present
    session_regime = "marathon" if planner_turns >= 25 else "sprint"
    has_tool_activity = top_tool_override not in ("none", "", "ask_question", "send_message")
    if steps <= 3 and not has_tool_activity and total_bloat_step_tok == 0 and conv_quad_post_sat_tok == 0:
        hal_saved_usd = 0.0
        hal_wbc_usd = exact_api_usd
    else:
        fresh_cut = min(
            0.70,
            float(total_bloat_step_tok) / float(max(1, running_raw_unbounded_ctx)),
        )
        cache_cut = min(
            0.75,
            float(max(0, conv_actual_input_tok - conv_flushed_sliced_cum_tok))
            / float(max(1, conv_actual_input_tok)),
        )
        out_cut = (
            min(0.18, 0.06 + (0.06 if truncated_steps > 0 else 0.0) + (0.06 if planner_turns >= 25 else 0.0))
            if (planner_turns > 4 and (total_bloat_step_tok > 0 or truncated_steps > 0))
            else 0.0
        )
        hal_saved_usd = max(0.0, fresh_usd * fresh_cut + cache_usd * cache_cut + out_usd * out_cut)
        hal_wbc_usd = max(0.0, exact_api_usd - hal_saved_usd)

    return {
        "harness": harness,
        "conv_id": conv_id,
        "short_id": short_id[:8],
        "session_regime": session_regime,
        "steps": steps,
        "planner_turns": planner_turns,
        "truncated_steps": truncated_steps,
        "transcript_bytes": conv_bytes,
        "transcript_kb": full_kb,
        "bloat_trigger": trigger,
        "top_offender_tool": top_tool_override,
        "raw_unbounded_quadratic_tokens": conv_raw_unbounded_tok,
        "actual_cumulative_input_tokens": conv_actual_input_tok,
        "output_tokens": conv_output_tok,
        "wasted_bloat_tokens": conv_wasted_tok,
        "quadratic_post_saturation_tokens": conv_quad_post_sat_tok,
        "unfiltered_tool_bloat_tokens": conv_tool_bloat_tok,
        "bicameral_local_offload_tokens": conv_local_offload_tok,
        "bbb_optimized_tokens": conv_bbb_opt_tok,
        "bbb_optimized_output_tokens": conv_bbb_opt_out_tok,
        "metered_exact": metered_flag,
        "metered_turns": m_turns,
        "model_matched_turns": m_matched,
        "metered_billed_input_tokens": m_in,
        "metered_cache_create_tokens": m_cc,
        "metered_cache_create_1h_tokens": m_cc_1h,
        "metered_cache_create_5m_tokens": m_cc_5m,
        "metered_cache_read_tokens": m_cr,
        "metered_output_tokens": m_out,
        "_mtime": mtime,
        "_fresh_input_usd": round(fresh_usd, 4),
        "_typed_prompt_tokens": m_typed_tok,
        "_typed_prompt_usd": round(typed_prompt_usd, 4),
        "_cache_read_usd": round(cache_usd, 4),
        "_output_usd": round(out_usd, 4),
        "_local_title": _clean_session_title(first_prompt),
        "_model_usage": model_usage or {},
        "primary_model": (
            max((model_usage or {}).values(), key=lambda mu: (float(mu.get("spend_usd", 0.0)), int(mu.get("turns", 0))))["label"]
            if model_usage
            else ""
        ),
        "exact_api_spend_usd": round(exact_api_usd, 2),
        "hal_wbc_spend_usd": round(hal_wbc_usd, 2),
        "hal_saved_usd": round(hal_saved_usd, 2),
        "conservative_spend_usd": round(conv_cons_frontier_usd, 2),
        "estimated_spend_usd": round(conv_real_frontier_usd, 2),
        "uncached_spend_usd": round(conv_uncached_frontier_usd, 2),
        "bbb_spend_usd": round(conv_real_bbb_usd, 2),
        "conservative_savable_usd": round(conv_cons_saved_usd, 2),
        "usd_savable": round(conv_real_saved_usd, 2),
        "uncached_savable_usd": round(conv_uncached_saved_usd, 2),
        "last_modified_str": datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M"),
        "first_prompt": (
            first_prompt[:110]
            if os.environ.get("WBC_AUDIT_INCLUDE_PROMPTS") == "1"
            else "[redacted - set WBC_AUDIT_INCLUDE_PROMPTS=1 to include prompt text]"
        ),
    }


_WBC_TRANSCRIPT_HOOK_MARKERS = (
    "[WBC ",
    "[WBC-",
    "WBC Pre-Flight",
    "Bounded Slice",
    "body-immune-hook",
)


def _parse_iso_or_epoch_ts(val: Any) -> float:
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        fv = float(val)
        return fv / 1000.0 if fv > 1e11 else fv
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return 0.0
        try:
            fv = float(s)
            return fv / 1000.0 if fv > 1e11 else fv
        except ValueError:
            pass
        try:
            iso = s.replace("Z", "+00:00") if s.endswith("Z") else s
            dt = datetime.datetime.fromisoformat(iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            return dt.timestamp()
        except Exception:
            return 0.0
    return 0.0


def scan_antigravity_harness(
    agy_dir: Path,
    window_seconds: float,
    now: float,
    in_price_per_m: float,
    out_price_per_m: float,
    explicit_price_override: bool = False,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not agy_dir.is_dir():
        return records

    for c_dir in agy_dir.iterdir():
        if not c_dir.is_dir() or c_dir.name.startswith("."):
            continue
        logs_dir = c_dir / ".system_generated" / "logs"
        compact_p = logs_dir / "transcript.jsonl"
        full_p = logs_dir / "transcript_full.jsonl"
        primary_p = compact_p if compact_p.is_file() else (full_p if full_p.is_file() else None)
        if not primary_p:
            continue
        try:
            mtime = os.path.getmtime(primary_p)
            if window_seconds > 0 and (now - mtime) > window_seconds:
                continue
            compact_bytes = os.path.getsize(primary_p)
            full_bytes = os.path.getsize(full_p) if full_p.is_file() else compact_bytes
            conv_bytes = max(compact_bytes, full_bytes)

            raw_objs: List[Tuple[Dict[str, Any], int]] = []
            with open(primary_p, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        raw_objs.append((json.loads(s), len(s.encode("utf-8"))))
                    except Exception:
                        continue
            if not raw_objs:
                continue

            trunc_cnt = sum(1 for obj, _ in raw_objs if obj.get("truncated_fields"))
            overflow_per_trunc = (
                float(max(0, full_bytes - compact_bytes)) / float(max(1, trunc_cnt))
                if trunc_cnt > 0
                else 0.0
            )
            step_tuples: List[Tuple[bool, int, bool]] = []
            tool_counts: Dict[str, int] = {}
            first_prompt = ""
            active_model = ""
            m_turns = m_in = m_cr = m_out = m_matched = typed_prompt_tok = 0
            fresh_usd = cache_usd = out_usd = 0.0
            last_event_ts = 0.0
            wbc_hook_fired = False
            model_usage_map: Dict[str, Dict[str, Any]] = {}
            for obj, rlen in raw_objs:
                stype = str(obj.get("type") or "")
                content_str = str(obj.get("content") or "")
                ev_ts = _parse_iso_or_epoch_ts(obj.get("created_at") or obj.get("timestamp"))
                if ev_ts > last_event_ts:
                    last_event_ts = ev_ts
                if not wbc_hook_fired and stype not in ("USER_INPUT", "TOOL_RESPONSE") and content_str:
                    if any(mk in content_str for mk in _WBC_TRANSCRIPT_HOOK_MARKERS):
                        wbc_hook_fired = True
                ms_cand = _extract_model_selection_change(content_str)
                if ms_cand:
                    active_model = ms_cand
                if obj.get("model"):
                    active_model = str(obj.get("model"))
                is_p = stype == "PLANNER_RESPONSE"
                is_t = bool(obj.get("truncated_fields"))
                eff_b = int(round(float(rlen) + (overflow_per_trunc if is_t else 0.0)))
                step_tuples.append((is_p, eff_b, is_t))
                if is_p and ("input_tokens" in obj or "output_tokens" in obj or "cache_read_tokens" in obj):
                    t_in = int(obj.get("input_tokens") or 0)
                    t_cr = int(obj.get("cache_read_tokens") or 0)
                    t_out = int(obj.get("output_tokens") or 0)
                    m_turns += 1
                    m_in += t_in
                    m_cr += t_cr
                    m_out += t_out
                    tin_p, tout_p, matched = _resolve_model_rates(
                        active_model, in_price_per_m, out_price_per_m, explicit_price_override
                    )
                    if matched:
                        m_matched += 1
                    t_fresh = (t_in / 1_000_000.0) * tin_p
                    t_cache = (t_cr / 1_000_000.0) * (tin_p * 0.10)
                    t_out_u = (t_out / 1_000_000.0) * tout_p
                    fresh_usd += t_fresh
                    cache_usd += t_cache
                    out_usd += t_out_u
                    if active_model:
                        _record_turn_model_usage(
                            model_usage_map,
                            active_model,
                            turns=1,
                            billed_input_tokens=t_in,
                            cache_read_tokens=t_cr,
                            output_tokens=t_out,
                            fresh_usd=t_fresh,
                            cache_usd=t_cache,
                            output_usd=t_out_u,
                        )
                elif is_p and active_model:
                    est_out = max(20, int(round(eff_b / 4.0)))
                    tin_p, tout_p, _ = _resolve_model_rates(
                        active_model, in_price_per_m, out_price_per_m, explicit_price_override
                    )
                    _record_turn_model_usage(
                        model_usage_map,
                        active_model,
                        turns=1,
                        output_tokens=est_out,
                        output_usd=(est_out / 1_000_000.0) * tout_p,
                    )
                if stype == "USER_INPUT":
                    typed_prompt_tok += _estimate_typed_prompt_tokens(content_str)
                    if not first_prompt:
                        first_prompt = content_str.strip()
                for tc in obj.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        tname = str(tc.get("name") or "tool")
                        tool_counts[tname] = tool_counts.get(tname, 0) + 1
                if obj.get("tool_name"):
                    tname = str(obj.get("tool_name"))
                    tool_counts[tname] = tool_counts.get(tname, 0) + 1

            eff_ts = last_event_ts if last_event_ts > 0 else mtime
            if window_seconds > 0 and (now - eff_ts) > window_seconds:
                continue

            top_tool = max(tool_counts.items(), key=lambda kv: kv[1])[0] if tool_counts else "none"
            rec = _evaluate_step_sequence(
                step_Tuples=step_tuples,
                conv_bytes=conv_bytes,
                short_id=c_dir.name[:8],
                conv_id=c_dir.name,
                harness="antigravity",
                mtime=eff_ts,
                in_price_per_m=in_price_per_m,
                out_price_per_m=out_price_per_m,
                top_tool_override=top_tool,
                first_prompt=first_prompt,
                exact_usage={
                    "metered_turns": m_turns,
                    "model_matched_turns": m_matched,
                    "billed_input_tokens": m_in,
                    "cache_create_tokens": 0,
                    "cache_read_tokens": m_cr,
                    "output_tokens": m_out,
                    "typed_prompt_tokens": typed_prompt_tok,
                    "fresh_input_usd": fresh_usd,
                    "cache_read_usd": cache_usd,
                    "output_usd": out_usd,
                } if m_turns > 0 else None,
                model_usage=model_usage_map,
            )
            rec["_wbc_hook_fired"] = wbc_hook_fired
            records.append(rec)
        except Exception:
            continue
    return records


def _extract_claude_text(content_val: Any) -> str:
    if isinstance(content_val, str):
        return content_val.strip()
    if isinstance(content_val, list):
        parts: List[str] = []
        for blk in content_val:
            if isinstance(blk, dict) and blk.get("type") == "text" and blk.get("text"):
                parts.append(str(blk["text"]).strip())
            elif isinstance(blk, str):
                parts.append(blk.strip())
        return " ".join(p for p in parts if p)
    return ""


def scan_claude_code_harness(
    claude_dir: Path,
    window_seconds: float,
    now: float,
    in_price_per_m: float,
    out_price_per_m: float,
    explicit_price_override: bool = False,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not claude_dir.is_dir():
        return records

    for jpath in claude_dir.rglob("*.jsonl"):
        if not jpath.is_file():
            continue
        try:
            mtime = os.path.getmtime(jpath)
            if window_seconds > 0 and (now - mtime) > window_seconds:
                continue
            conv_bytes = os.path.getsize(jpath)
            step_tuples: List[Tuple[bool, int, bool]] = []
            tool_counts: Dict[str, int] = {}
            first_prompt = ""
            active_model = ""
            m_turns = m_in = m_cc = m_cc_1h = m_cc_5m = m_cr = m_out = m_matched = typed_prompt_tok = 0
            fresh_usd = cache_usd = out_usd = 0.0
            last_event_ts = 0.0
            wbc_hook_fired = False
            model_usage_map: Dict[str, Dict[str, Any]] = {}
            seen_usage_by_msg_key: Dict[Tuple[str, str], Tuple[Dict[str, Any], str]] = {}
            step_idx_by_msg_key: Dict[Tuple[str, str], int] = {}
            unmetered_p_bytes_by_model: Dict[str, List[int]] = {}
            line_idx = 0
            with open(jpath, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    line_idx += 1
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                    except Exception:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    msg_dict = obj.get("message") if isinstance(obj.get("message"), dict) else {}
                    ev_ts = _parse_iso_or_epoch_ts(
                        obj.get("timestamp") or obj.get("created_at") or msg_dict.get("created_at")
                    )
                    if ev_ts > last_event_ts:
                        last_event_ts = ev_ts
                    role = str(
                        obj.get("type")
                        or obj.get("role")
                        or msg_dict.get("role")
                        or ""
                    ).lower()
                    if not wbc_hook_fired:
                        if role == "attachment" and isinstance(obj.get("attachment"), dict):
                            att = obj["attachment"]
                            atype = str(att.get("type") or "")
                            if atype.startswith("hook_"):
                                acmd = str(att.get("command") or "")
                                actx = str(att.get("content") or "")
                                if (
                                    "body-immune-hook" in acmd
                                    or "whole_body_compute" in acmd
                                    or any(mk in actx for mk in _WBC_TRANSCRIPT_HOOK_MARKERS)
                                ):
                                    wbc_hook_fired = True
                        elif role in ("system", "progress") and "body-immune-hook" in s:
                            wbc_hook_fired = True
                    raw_model = str(msg_dict.get("model") or obj.get("model") or "")
                    if raw_model and raw_model != "<synthetic>":
                        active_model = raw_model
                    is_p = role in ("assistant", "model", "planner_response")
                    msg_content = msg_dict.get("content") if "content" in msg_dict else obj.get("content")
                    if isinstance(msg_content, list):
                        for blk in msg_content:
                            if isinstance(blk, dict) and blk.get("type") == "tool_use":
                                tname = str(blk.get("name") or "tool")
                                tool_counts[tname] = tool_counts.get(tname, 0) + 1
                    b_len = len(s.encode("utf-8"))
                    u_dict = msg_dict.get("usage") if isinstance(msg_dict.get("usage"), dict) else None
                    if not isinstance(u_dict, dict):
                        u_dict = obj.get("usage") if isinstance(obj.get("usage"), dict) else None
                    msg_id = str(msg_dict.get("id") or obj.get("messageId") or "").strip()
                    req_id = str(obj.get("requestId") or msg_dict.get("requestId") or "").strip()
                    dedup_key = (msg_id, req_id) if (msg_id or req_id) else (f"_line_{line_idx}", "")
                    is_new_turn_msg = dedup_key not in seen_usage_by_msg_key
                    if isinstance(u_dict, dict) and (
                        "input_tokens" in u_dict
                        or "cache_read_input_tokens" in u_dict
                        or "output_tokens" in u_dict
                    ):
                        # Claude Code writes one JSONL row per reply content block (thinking, text, tool_use)
                        # repeating the identical usage block for the same (message.id, requestId).
                        # Keep the latest usage block per unique (message.id, requestId) so each API turn is counted once.
                        seen_usage_by_msg_key[dedup_key] = (u_dict, active_model)
                    elif is_p and active_model and is_new_turn_msg:
                        unmetered_p_bytes_by_model.setdefault(active_model, []).append(b_len)
                    if role in ("user", "human", "user_input") and not obj.get("isMeta"):
                        cand = _extract_claude_text(msg_content)
                        if cand and not cand.startswith("output:") and not cand.startswith("tool_output:"):
                            typed_prompt_tok += _estimate_typed_prompt_tokens(cand)
                            if not first_prompt:
                                first_prompt = cand[:240]
                    if is_p and (msg_id or req_id) and dedup_key in step_idx_by_msg_key:
                        p_idx = step_idx_by_msg_key[dedup_key]
                        prev_p, prev_b, prev_t = step_tuples[p_idx]
                        comb_b = prev_b + b_len
                        step_tuples[p_idx] = (prev_p, comb_b, prev_t or (comb_b > 16000))
                    else:
                        if is_p and (msg_id or req_id):
                            step_idx_by_msg_key[dedup_key] = len(step_tuples)
                        step_tuples.append((is_p, b_len, b_len > 16000))

            for _dkey, (u_dict, turn_model) in seen_usage_by_msg_key.items():
                t_in = int(u_dict.get("input_tokens") or 0)
                t_cc = int(u_dict.get("cache_creation_input_tokens") or 0)
                cc_sub = u_dict.get("cache_creation") if isinstance(u_dict.get("cache_creation"), dict) else {}
                cc_1h = max(0, int(cc_sub.get("ephemeral_1h_input_tokens") or 0))
                cc_5m_explicit = max(0, int(cc_sub.get("ephemeral_5m_input_tokens") or 0))
                if cc_1h > 0 or cc_5m_explicit > 0:
                    cc_1h = min(t_cc, cc_1h) if t_cc >= cc_1h else cc_1h
                    cc_5m = max(0, t_cc - cc_1h) if t_cc >= cc_1h else cc_5m_explicit
                    t_cc = cc_1h + cc_5m
                else:
                    cc_1h = 0
                    cc_5m = t_cc
                t_cr = int(u_dict.get("cache_read_input_tokens") or 0)
                t_out = int(u_dict.get("output_tokens") or 0)
                m_turns += 1
                m_in += t_in
                m_cc += t_cc
                m_cc_1h += cc_1h
                m_cc_5m += cc_5m
                m_cr += t_cr
                m_out += t_out
                tin_p, tout_p, matched = _resolve_model_rates(
                    turn_model, in_price_per_m, out_price_per_m, explicit_price_override
                )
                if matched:
                    m_matched += 1
                # Anthropic prompt caching rates: 5m cache write = 1.25x input, 1h cache write = 2.0x input, cache read = 0.10x input
                t_fresh = (
                    (t_in / 1_000_000.0) * tin_p
                    + (cc_5m / 1_000_000.0) * (tin_p * 1.25)
                    + (cc_1h / 1_000_000.0) * (tin_p * 2.00)
                )
                t_cache = (t_cr / 1_000_000.0) * (tin_p * 0.10)
                t_out_u = (t_out / 1_000_000.0) * tout_p
                fresh_usd += t_fresh
                cache_usd += t_cache
                out_usd += t_out_u
                if turn_model:
                    _record_turn_model_usage(
                        model_usage_map,
                        turn_model,
                        turns=1,
                        billed_input_tokens=t_in,
                        cache_create_tokens=t_cc,
                        cache_read_tokens=t_cr,
                        output_tokens=t_out,
                        fresh_usd=t_fresh,
                        cache_usd=t_cache,
                        output_usd=t_out_u,
                    )
            if not seen_usage_by_msg_key:
                for turn_model, b_list in unmetered_p_bytes_by_model.items():
                    tin_p, tout_p, _ = _resolve_model_rates(
                        turn_model, in_price_per_m, out_price_per_m, explicit_price_override
                    )
                    for b_len in b_list:
                        est_out = max(20, int(round(b_len / 4.0)))
                        _record_turn_model_usage(
                            model_usage_map,
                            turn_model,
                            turns=1,
                            output_tokens=est_out,
                            output_usd=(est_out / 1_000_000.0) * tout_p,
                        )
            if not step_tuples:
                continue
            eff_ts = last_event_ts if last_event_ts > 0 else mtime
            if window_seconds > 0 and (now - eff_ts) > window_seconds:
                continue
            cid = jpath.stem
            top_tool = max(tool_counts.items(), key=lambda kv: kv[1])[0] if tool_counts else ("Bash" if len(step_tuples) > 3 else "none")
            rec = _evaluate_step_sequence(
                step_Tuples=step_tuples,
                conv_bytes=conv_bytes,
                short_id=cid[:8],
                conv_id=cid,
                harness="claude_code",
                mtime=eff_ts,
                in_price_per_m=in_price_per_m,
                out_price_per_m=out_price_per_m,
                top_tool_override=top_tool,
                first_prompt=first_prompt,
                exact_usage={
                    "metered_turns": m_turns,
                    "model_matched_turns": m_matched,
                    "billed_input_tokens": m_in,
                    "cache_create_tokens": m_cc,
                    "cache_create_1h_tokens": m_cc_1h,
                    "cache_create_5m_tokens": m_cc_5m,
                    "cache_read_tokens": m_cr,
                    "output_tokens": m_out,
                    "typed_prompt_tokens": typed_prompt_tok,
                    "fresh_input_usd": fresh_usd,
                    "cache_read_usd": cache_usd,
                    "output_usd": out_usd,
                } if m_turns > 0 else None,
                model_usage=model_usage_map,
            )
            rec["_wbc_hook_fired"] = wbc_hook_fired
            records.append(rec)
        except Exception:
            continue
    return records


def scan_goose_harness(
    goose_db: Path,
    window_seconds: float,
    now: float,
    in_price_per_m: float,
    out_price_per_m: float,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not goose_db.is_file():
        return records

    try:
        mtime = os.path.getmtime(goose_db)
        conn = sqlite3.connect(f"file:{goose_db}?mode=ro", uri=True, timeout=1.5)
        conn.row_factory = sqlite3.Row
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        if "messages" in tables:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(messages)").fetchall()}
            sess_col = "session_id" if "session_id" in cols else ("conversation_id" if "conversation_id" in cols else None)
            role_col = "role" if "role" in cols else None
            content_col = "content_json" if "content_json" in cols else ("content" if "content" in cols else None)
            ts_col = next(
                (c for c in ("created_timestamp", "created_at", "timestamp", "updated_at", "created") if c in cols),
                None,
            )
            order_col = next((c for c in ("id", "created_timestamp", "created_at", "timestamp") if c in cols), None)
            if sess_col and content_col:
                role_expr = role_col if role_col else "'assistant'"
                ts_expr = ts_col if ts_col else "NULL"
                order_clause = f" ORDER BY {order_col} DESC" if order_col else ""
                q = f"SELECT {sess_col} AS sid, {role_expr} AS role, {content_col} AS body, {ts_expr} AS ts FROM messages{order_clause} LIMIT 5000"
                rows = conn.execute(q).fetchall()
                by_sid: Dict[str, List[Tuple[bool, int, bool]]] = {}
                latest_ts_by_sid: Dict[str, float] = {}
                for r in rows:
                    sid = str(r["sid"] or "goose-0")
                    role_s = str(r["role"] or "").lower()
                    body_s = str(r["body"] or "")
                    row_ts = _parse_iso_or_epoch_ts(r["ts"])
                    if row_ts > latest_ts_by_sid.get(sid, 0.0):
                        latest_ts_by_sid[sid] = row_ts
                    b_len = max(32, len(body_s.encode("utf-8")))
                    is_p = role_s in ("assistant", "model")
                    by_sid.setdefault(sid, []).append((is_p, b_len, b_len > 16000))
                for sid, st_list in by_sid.items():
                    sess_ts = latest_ts_by_sid.get(sid, 0.0) or mtime
                    if window_seconds > 0 and (now - sess_ts) > window_seconds:
                        continue
                    tot_b = sum(x[1] for x in st_list)
                    records.append(
                        _evaluate_step_sequence(
                            step_Tuples=st_list,
                            conv_bytes=tot_b,
                            short_id=sid[:8],
                            conv_id=sid,
                            harness="goose",
                            mtime=sess_ts,
                            in_price_per_m=in_price_per_m,
                            out_price_per_m=out_price_per_m,
                            top_tool_override="developer__shell",
                        )
                    )
        conn.close()
    except Exception:
        pass
    return records


_WSL_SKIP_USERS = frozenset({"public", "default", "default user", "all users", "desktop.ini"})


def _iter_candidate_user_homes() -> List[Path]:
    """Return Path.home() plus Windows/WSL2 `/mnt/c/Users/<WinUser>` homes when present."""
    homes: List[Path] = []
    seen: set[str] = set()

    def _add(p: Path) -> None:
        try:
            ep = p.expanduser()
            if not ep.is_dir():
                return
            rp = os.path.realpath(ep)
            if rp not in seen:
                seen.add(rp)
                homes.append(ep)
        except Exception:
            pass

    _add(Path.home())
    win_prof = os.environ.get("USERPROFILE", "").strip()
    if win_prof:
        _add(Path(win_prof))

    wsl_root_env = os.environ.get("WBC_WSL_USERS_ROOT", "").strip()
    wsl_root = Path(wsl_root_env) if wsl_root_env else Path("/mnt/c/Users")
    try:
        if wsl_root.is_dir():
            for child in sorted(wsl_root.iterdir()):
                cname = child.name.strip()
                if not cname or cname.startswith(".") or cname.lower() in _WSL_SKIP_USERS:
                    continue
                _add(child)
    except Exception:
        pass

    return homes or [Path.home()]


def _dedup_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate scanned conversation records by (harness, conv_id), keeping richest token payload."""
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for rec in records:
        key = (
            str(rec.get("harness") or ""),
            str(rec.get("conversation_id") or rec.get("conv_id") or rec.get("short_id") or id(rec)),
        )
        prev = by_key.get(key)
        rec_rank = (
            1 if rec.get("metered_exact") else 0,
            int(rec.get("transcript_bytes", 0)),
            int(rec.get("actual_cumulative_input_tokens", 0)),
        )
        prev_rank = (
            (
                1 if prev.get("metered_exact") else 0,
                int(prev.get("transcript_bytes", 0)),
                int(prev.get("actual_cumulative_input_tokens", 0)),
            )
            if prev is not None
            else (-1, -1, -1)
        )
        if rec_rank > prev_rank:
            by_key[key] = rec
    return list(by_key.values())


def _extract_steps_from_json_blob(val_s: str) -> List[Tuple[bool, int, bool]]:
    """
    Extract real `(is_planner_turn, step_bytes, is_truncated)` tuples from a JSON value stored in
    a legacy SQLite `ItemTable` or thread file without slicing raw byte counts into fake alternating turns.
    """
    b_total = len(val_s.encode("utf-8"))
    if b_total < 64:
        return []
    out: List[Tuple[bool, int, bool]] = []
    try:
        parsed = json.loads(val_s)
    except Exception:
        parsed = None

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            # Look for explicit message/turn arrays
            for list_key in ("messages", "conversation", "bubbles", "turns", "steps", "history", "chat", "prompts"):
                arr = node.get(list_key)
                if isinstance(arr, list) and arr:
                    for idx, elem in enumerate(arr):
                        if isinstance(elem, dict):
                            raw_e = json.dumps(elem)
                            eb = max(32, len(raw_e.encode("utf-8")))
                            role = str(elem.get("role") or elem.get("sender") or elem.get("type") or "").lower()
                            is_p = (elem.get("type") == 2) or (
                                role in ("assistant", "model", "planner_response", "ai", "cascade", "zed")
                            )
                            out.append((is_p, eb, eb > 16000))
                        elif isinstance(elem, str) and elem.strip():
                            eb = max(32, len(elem.encode("utf-8")))
                            out.append((idx % 2 == 1, eb, eb > 16000))
            for sub_key in ("tabs", "threads", "sessions", "chats"):
                sub = node.get(sub_key)
                if isinstance(sub, list):
                    for child in sub:
                        _walk(child)
        elif isinstance(node, list):
            for idx, elem in enumerate(node):
                if isinstance(elem, dict):
                    if any(k in elem for k in ("role", "type", "text", "content", "bubbles", "turns", "messages")):
                        if any(k in elem for k in ("bubbles", "turns", "messages", "conversation")):
                            _walk(elem)
                        else:
                            raw_e = json.dumps(elem)
                            eb = max(32, len(raw_e.encode("utf-8")))
                            role = str(elem.get("role") or elem.get("sender") or elem.get("type") or "").lower()
                            is_p = (elem.get("type") == 2) or (
                                role in ("assistant", "model", "planner_response", "ai", "cascade", "zed")
                            )
                            out.append((is_p, eb, eb > 16000))
                elif isinstance(elem, str) and elem.strip():
                    eb = max(32, len(elem.encode("utf-8")))
                    out.append((idx % 2 == 1, eb, eb > 16000))

    if parsed is not None:
        _walk(parsed)
    if len(out) == 1:
        # Ensure at least one user prompt + one assistant response when a single turn container was stored
        only_p, only_b, only_t = out[0]
        prompt_b = min(256, max(32, only_b // 10))
        return [(False, prompt_b, False), (True, max(32, only_b - prompt_b), only_t)]
    return out


def _scan_cursor_global_vscdb(
    db_p: Path,
    window_seconds: float,
    now: float,
    in_price_per_m: float,
    out_price_per_m: float,
) -> List[Dict[str, Any]]:
    """Scan modern Cursor `globalStorage/state.vscdb` (`cursorDiskKV` composerData + bubbleId)."""
    records: List[Dict[str, Any]] = []
    if not db_p.is_file():
        return records
    try:
        db_mtime = os.path.getmtime(db_p)
        conn = sqlite3.connect(f"file:{db_p}?mode=ro", uri=True, timeout=1.5)
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        if "cursorDiskKV" not in tables:
            conn.close()
            return records

        composers: Dict[str, Dict[str, Any]] = {}
        for k, v in conn.execute(
            "SELECT key, value FROM cursorDiskKV WHERE key LIKE 'composerData:%' LIMIT 1500"
        ):
            if not k or not v:
                continue
            cid = str(k).split(":", 1)[1] if ":" in str(k) else str(k)
            try:
                raw_s = v.decode("utf-8", errors="ignore") if isinstance(v, bytes) else str(v)
                d = json.loads(raw_s)
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            ts = _parse_iso_or_epoch_ts(d.get("lastUpdatedAt") or d.get("createdAt"))
            prompt_s = _extract_text_snippet(d.get("text") or d.get("name") or "")
            composers[cid] = {
                "ts": ts,
                "prompt": prompt_s,
                "raw_bytes": len(raw_s.encode("utf-8")),
                "conversation": d.get("conversation") if isinstance(d.get("conversation"), list) else [],
            }

        bubbles_by_cid: Dict[str, List[Tuple[float, bool, int, int, int, str]]] = {}
        for k, v in conn.execute(
            "SELECT key, value FROM cursorDiskKV WHERE key LIKE 'bubbleId:%' LIMIT 25000"
        ):
            if not k or not v:
                continue
            parts = str(k).split(":")
            if len(parts) < 3:
                continue
            cid = parts[1]
            try:
                raw_s = v.decode("utf-8", errors="ignore") if isinstance(v, bytes) else str(v)
                b_len = max(32, len(raw_s.encode("utf-8")))
                d = json.loads(raw_s)
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            b_type = d.get("type")
            role_s = str(d.get("role") or "").lower()
            is_p = (b_type == 2) or (role_s in ("assistant", "model", "ai"))
            bts = _parse_iso_or_epoch_ts(d.get("createdAt") or d.get("timestamp"))
            tc = d.get("tokenCount") if isinstance(d.get("tokenCount"), dict) else {}
            in_tok = int(tc.get("inputTokens") or 0)
            out_tok = int(tc.get("outputTokens") or 0)
            snip = ""
            if not is_p:
                snip = _extract_text_snippet(d.get("text") or d.get("rawText") or "")
            bubbles_by_cid.setdefault(cid, []).append((bts, is_p, b_len, in_tok, out_tok, snip))
        conn.close()

        all_cids = set(bubbles_by_cid.keys()) | {
            cid for cid, meta in composers.items() if meta.get("conversation")
        }
        for cid in all_cids:
            cmeta = composers.get(cid, {})
            blist = bubbles_by_cid.get(cid, [])
            if blist:
                blist.sort(key=lambda x: x[0] if x[0] > 0 else 0.0)
                latest_b_ts = max((x[0] for x in blist if x[0] > 0), default=0.0)
                sess_ts = max(float(cmeta.get("ts") or 0.0), latest_b_ts) or db_mtime
                if window_seconds > 0 and (now - sess_ts) > window_seconds:
                    continue
                step_tuples = [(x[1], x[2], x[2] > 16000) for x in blist]
                tot_bytes = sum(x[2] for x in blist)
                m_turns = sum(1 for x in blist if x[1] and (x[3] > 0 or x[4] > 0))
                m_in = sum(x[3] for x in blist)
                m_out = sum(x[4] for x in blist)
                first_prompt = str(cmeta.get("prompt") or "") or next(
                    (x[5] for x in blist if x[5]), ""
                )
            else:
                conv_items = cmeta.get("conversation") or []
                sess_ts = float(cmeta.get("ts") or 0.0) or db_mtime
                if window_seconds > 0 and (now - sess_ts) > window_seconds:
                    continue
                step_tuples = []
                tot_bytes = 0
                m_turns = m_in = m_out = 0
                first_prompt = str(cmeta.get("prompt") or "")
                for item in conv_items:
                    if not isinstance(item, dict):
                        continue
                    raw_i = json.dumps(item)
                    ib = max(32, len(raw_i.encode("utf-8")))
                    tot_bytes += ib
                    is_p = (item.get("type") == 2) or (
                        str(item.get("role") or "").lower() in ("assistant", "model", "ai")
                    )
                    if not first_prompt and not is_p:
                        first_prompt = _extract_text_snippet(item.get("text") or "")
                    step_tuples.append((is_p, ib, ib > 16000))
            if not step_tuples:
                continue
            records.append(
                _evaluate_step_sequence(
                    step_Tuples=step_tuples,
                    conv_bytes=tot_bytes,
                    short_id=cid[:8],
                    conv_id=cid,
                    harness="cursor",
                    mtime=sess_ts,
                    in_price_per_m=in_price_per_m,
                    out_price_per_m=out_price_per_m,
                    top_tool_override="cursor_composer",
                    first_prompt=first_prompt,
                    exact_usage={
                        "metered_turns": m_turns,
                        "billed_input_tokens": m_in,
                        "cache_create_tokens": 0,
                        "cache_read_tokens": 0,
                        "output_tokens": m_out,
                    }
                    if m_turns > 0
                    else None,
                )
            )
    except Exception:
        pass
    return records


def _scan_cursor_agent_transcripts_jsonl(
    projects_root: Path,
    window_seconds: float,
    now: float,
    in_price_per_m: float,
    out_price_per_m: float,
) -> List[Dict[str, Any]]:
    """Scan modern Cursor `~/.cursor/projects/**/agent-transcripts/**/*.jsonl` transcripts."""
    records: List[Dict[str, Any]] = []
    if not projects_root.exists():
        return records
    jfiles: List[Path] = []
    try:
        if projects_root.is_file() and projects_root.suffix == ".jsonl":
            jfiles.append(projects_root)
        elif projects_root.is_dir():
            for jf in projects_root.rglob("*.jsonl"):
                if jf.is_file() and ("agent-transcripts" in jf.parts or projects_root.name == "agent-transcripts"):
                    jfiles.append(jf)
    except Exception:
        return records

    for jpath in jfiles[:400]:
        try:
            mtime = os.path.getmtime(jpath)
            if window_seconds > 0 and (now - mtime) > window_seconds:
                continue
            conv_bytes = os.path.getsize(jpath)
            step_tuples: List[Tuple[bool, int, bool]] = []
            first_prompt = ""
            with open(jpath, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                    except Exception:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    msg = obj.get("message") if isinstance(obj.get("message"), dict) else {}
                    role = str(obj.get("role") or msg.get("role") or obj.get("type") or "").lower()
                    is_p = role in ("assistant", "model", "ai", "planner_response")
                    b_len = max(32, len(s.encode("utf-8")))
                    if not first_prompt and role in ("user", "human", "user_input"):
                        first_prompt = _extract_text_snippet(msg.get("content") or obj.get("content") or "")
                    step_tuples.append((is_p, b_len, b_len > 16000))
            if not step_tuples:
                continue
            cid = jpath.stem
            records.append(
                _evaluate_step_sequence(
                    step_Tuples=step_tuples,
                    conv_bytes=conv_bytes,
                    short_id=cid[:8],
                    conv_id=cid,
                    harness="cursor",
                    mtime=mtime,
                    in_price_per_m=in_price_per_m,
                    out_price_per_m=out_price_per_m,
                    top_tool_override="cursor_agent",
                    first_prompt=first_prompt,
                )
            )
        except Exception:
            continue
    return records


def scan_cursor_harness(
    cursor_dir: Optional[Any] = None,
    window_seconds: float = 0.0,
    now: float = 0.0,
    in_price_per_m: float = 5.0,
    out_price_per_m: float = 25.0,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if now <= 0.0:
        now = time.time()

    roots: List[Path] = []
    transcript_roots: List[Path] = []
    if cursor_dir is not None:
        raw_list = list(cursor_dir) if isinstance(cursor_dir, (list, tuple)) else [cursor_dir]
        for item in raw_list:
            p = Path(item).expanduser()
            roots.append(p)
            if p.name == "workspaceStorage" and (p.parent / "globalStorage" / "state.vscdb").is_file():
                roots.append(p.parent / "globalStorage")
            elif p.is_dir() and (p / "workspaceStorage").is_dir():
                roots.append(p / "workspaceStorage")
                if (p / "globalStorage").is_dir():
                    roots.append(p / "globalStorage")
            if p.is_dir() and ((p / "projects").is_dir() or p.name in ("projects", "agent-transcripts")):
                transcript_roots.append(p / "projects" if (p / "projects").is_dir() else p)
            elif p.is_dir():
                transcript_roots.append(p)
    else:
        env_dir = os.environ.get("CURSOR_STORAGE_DIR")
        if env_dir:
            ep = Path(env_dir).expanduser()
            roots.append(ep)
            if ep.name == "workspaceStorage" and (ep.parent / "globalStorage").is_dir():
                roots.append(ep.parent / "globalStorage")
            elif ep.is_dir() and (ep / "workspaceStorage").is_dir():
                roots.append(ep / "workspaceStorage")
                if (ep / "globalStorage").is_dir():
                    roots.append(ep / "globalStorage")
            transcript_roots.append(ep)
        else:
            appdata = os.environ.get("APPDATA", "").strip()
            if appdata:
                ap_user = Path(appdata) / "Cursor" / "User"
                roots.extend([ap_user / "workspaceStorage", ap_user / "globalStorage"])
            for home in _iter_candidate_user_homes():
                for user_base in (
                    home / "Library" / "Application Support" / "Cursor" / "User",
                    home / "AppData" / "Roaming" / "Cursor" / "User",
                    home / ".config" / "Cursor" / "User",
                    home / ".cursor-server" / "data" / "User",
                ):
                    roots.extend([user_base / "workspaceStorage", user_base / "globalStorage"])
                transcript_roots.append(home / ".cursor" / "projects")

    seen_global_dbs: set[str] = set()
    seen_legacy_dbs: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        # Modern globalStorage/state.vscdb (`cursorDiskKV`)
        global_candidates: List[Path] = []
        if root.is_file() and root.suffix == ".vscdb":
            global_candidates.append(root)
        elif root.is_dir():
            if (root / "state.vscdb").is_file():
                global_candidates.append(root / "state.vscdb")
            if (root / "globalStorage" / "state.vscdb").is_file():
                global_candidates.append(root / "globalStorage" / "state.vscdb")
        for gdb in global_candidates:
            rp = os.path.realpath(gdb)
            if rp not in seen_global_dbs:
                seen_global_dbs.add(rp)
                records.extend(
                    _scan_cursor_global_vscdb(gdb, window_seconds, now, in_price_per_m, out_price_per_m)
                )

        # Legacy workspaceStorage/*/state.vscdb (`ItemTable`)
        vscdb_files = (
            [root]
            if (root.is_file() and root.suffix == ".vscdb")
            else (list(root.glob("*/state.vscdb")) + list(root.glob("*.vscdb")))
        )
        for db_p in vscdb_files[:80]:
            if not db_p.is_file() or db_p.parent.name == "globalStorage":
                continue
            rp = os.path.realpath(db_p)
            if rp in seen_legacy_dbs:
                continue
            seen_legacy_dbs.add(rp)
            try:
                mtime = os.path.getmtime(db_p)
                if window_seconds > 0 and (now - mtime) > window_seconds:
                    continue
                conn = sqlite3.connect(f"file:{db_p}?mode=ro", uri=True, timeout=1.5)
                tables = {
                    r[0]
                    for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                }
                if "ItemTable" not in tables:
                    conn.close()
                    continue
                rows = conn.execute(
                    "SELECT key, value FROM ItemTable WHERE key LIKE '%aichat%' OR key LIKE '%composer%' OR key = 'aiService.prompts' LIMIT 40"
                ).fetchall()
                conn.close()
                if not rows:
                    continue
                step_tuples: List[Tuple[bool, int, bool]] = []
                tot_bytes = 0
                for _k, val in rows:
                    val_s = str(val or "")
                    b_len = len(val_s.encode("utf-8"))
                    if b_len < 64:
                        continue
                    tot_bytes += b_len
                    step_tuples.extend(_extract_steps_from_json_blob(val_s))
                if not step_tuples:
                    continue
                cid = db_p.parent.name if db_p.parent != root else db_p.stem
                records.append(
                    _evaluate_step_sequence(
                        step_Tuples=step_tuples,
                        conv_bytes=tot_bytes,
                        short_id=cid[:8],
                        conv_id=cid,
                        harness="cursor",
                        mtime=mtime,
                        in_price_per_m=in_price_per_m,
                        out_price_per_m=out_price_per_m,
                        top_tool_override="cursor_composer",
                    )
                )
            except Exception:
                continue

    seen_t_roots: set[str] = set()
    for tr in transcript_roots:
        if not tr.exists():
            continue
        rp = os.path.realpath(tr)
        if rp in seen_t_roots:
            continue
        seen_t_roots.add(rp)
        records.extend(
            _scan_cursor_agent_transcripts_jsonl(tr, window_seconds, now, in_price_per_m, out_price_per_m)
        )

    return _dedup_records(records)


def scan_gemini_cli_harness(
    gemini_cli_dir: Optional[Any] = None,
    window_seconds: float = 0.0,
    now: float = 0.0,
    in_price_per_m: float = 5.0,
    out_price_per_m: float = 25.0,
) -> List[Dict[str, Any]]:
    """Scan Google Gemini CLI (`~/.gemini/tmp/*/chats/session-*.{json,jsonl}`) sessions."""
    records: List[Dict[str, Any]] = []
    if now <= 0.0:
        now = time.time()

    roots: List[Path] = []
    if gemini_cli_dir is not None:
        if isinstance(gemini_cli_dir, (list, tuple)):
            roots.extend(Path(p).expanduser() for p in gemini_cli_dir)
        else:
            roots.append(Path(gemini_cli_dir).expanduser())
    else:
        env_dir = os.environ.get("GEMINI_CLI_DIR")
        if env_dir:
            roots.append(Path(env_dir).expanduser())
        else:
            for home in _iter_candidate_user_homes():
                roots.append(home / ".gemini" / "tmp")

    session_files: List[Path] = []
    seen_files: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        try:
            cands = (
                [root]
                if root.is_file()
                else (list(root.glob("*/chats/session-*.*")) + list(root.glob("chats/session-*.*")) + list(root.glob("session-*.*")))
            )
            for fp in cands:
                if fp.is_file() and fp.suffix in (".json", ".jsonl"):
                    rp = os.path.realpath(fp)
                    if rp not in seen_files:
                        seen_files.add(rp)
                        session_files.append(fp)
        except Exception:
            continue

    recent_files: List[Tuple[float, Path]] = []
    for spath in session_files:
        try:
            mtime = os.path.getmtime(spath)
            if window_seconds > 0 and (now - mtime) > window_seconds:
                continue
            recent_files.append((mtime, spath))
        except Exception:
            continue
    recent_files.sort(key=lambda item: item[0], reverse=True)

    for mtime, spath in recent_files[:600]:
        try:
            conv_bytes = os.path.getsize(spath)
            msgs_by_id: Dict[str, Dict[str, Any]] = {}
            session_id = spath.stem

            if spath.suffix == ".json":
                data = json.loads(spath.read_text(encoding="utf-8", errors="ignore"))
                if isinstance(data, dict):
                    if not spath.stem.startswith("session-") and data.get("sessionId"):
                        session_id = str(data.get("sessionId"))
                    for idx, m in enumerate(data.get("messages") or []):
                        if isinstance(m, dict):
                            mid = str(m.get("id") or f"idx-{idx}")
                            msgs_by_id[mid] = m
            else:
                with open(spath, "r", encoding="utf-8", errors="ignore") as fh:
                    for idx, line in enumerate(fh):
                        s = line.strip()
                        if not s:
                            continue
                        try:
                            obj = json.loads(s)
                        except Exception:
                            continue
                        if not isinstance(obj, dict):
                            continue
                        if not spath.stem.startswith("session-") and obj.get("sessionId"):
                            session_id = str(obj["sessionId"])
                        set_obj = obj.get("$set") if isinstance(obj.get("$set"), dict) else None
                        if set_obj is not None:
                            m_val = set_obj.get("messages")
                            if isinstance(m_val, list):
                                for m_i, m in enumerate(m_val):
                                    if isinstance(m, dict):
                                        mid = str(m.get("id") or f"set-{idx}-{m_i}")
                                        msgs_by_id[mid] = m
                            for sk, sv in set_obj.items():
                                if sk.startswith("messages.") and isinstance(sv, dict):
                                    mid = str(sv.get("id") or sk)
                                    msgs_by_id[mid] = sv
                        elif obj.get("type") in ("user", "gemini", "model", "assistant", "tool"):
                            mid = str(obj.get("id") or f"ln-{idx}")
                            msgs_by_id[mid] = obj

            if not msgs_by_id:
                continue
            step_tuples: List[Tuple[bool, int, bool]] = []
            first_prompt = ""
            m_turns = m_in = m_cr = m_out = m_matched = typed_prompt_tok = 0
            fresh_usd = cache_usd = out_usd = 0.0
            model_usage_map: Dict[str, Dict[str, Any]] = {}
            active_model = ""
            for m in msgs_by_id.values():
                mtype = str(m.get("type") or m.get("role") or "").lower()
                raw_mod = str(m.get("model") or m.get("modelId") or "")
                if raw_mod:
                    active_model = raw_mod
                is_p = mtype in ("gemini", "assistant", "model", "planner_response")
                raw_m = json.dumps(m)
                b_len = max(32, len(raw_m.encode("utf-8")))
                if mtype in ("user", "human"):
                    u_snip = _extract_text_snippet(m.get("content") or "")
                    typed_prompt_tok += _estimate_typed_prompt_tokens(u_snip)
                    if not first_prompt:
                        first_prompt = u_snip
                tok = m.get("tokens") if isinstance(m.get("tokens"), dict) else None
                if is_p and isinstance(tok, dict) and (tok.get("input") or tok.get("output") or tok.get("total")):
                    in_total = int(tok.get("input") or 0)
                    cached_t = int(tok.get("cached") or 0)
                    out_t = int(tok.get("output") or 0) + int(tok.get("thoughts") or 0)
                    t_in = max(0, in_total - cached_t)
                    t_cr = max(0, cached_t)
                    t_out = max(0, out_t)
                    m_turns += 1
                    m_in += t_in
                    m_cr += t_cr
                    m_out += t_out
                    tin_p, tout_p, matched = _resolve_model_rates(
                        active_model, in_price_per_m, out_price_per_m, False
                    )
                    if matched:
                        m_matched += 1
                    t_fresh = (t_in / 1_000_000.0) * tin_p
                    t_cache = (t_cr / 1_000_000.0) * (tin_p * 0.10)
                    t_out_u = (t_out / 1_000_000.0) * tout_p
                    fresh_usd += t_fresh
                    cache_usd += t_cache
                    out_usd += t_out_u
                    if active_model:
                        _record_turn_model_usage(
                            model_usage_map,
                            active_model,
                            turns=1,
                            billed_input_tokens=t_in,
                            cache_read_tokens=t_cr,
                            output_tokens=t_out,
                            fresh_usd=t_fresh,
                            cache_usd=t_cache,
                            output_usd=t_out_u,
                        )
                elif is_p and active_model:
                    est_out = max(20, int(round(b_len / 4.0)))
                    _, tout_p, _ = _resolve_model_rates(active_model, in_price_per_m, out_price_per_m, False)
                    _record_turn_model_usage(
                        model_usage_map,
                        active_model,
                        turns=1,
                        output_tokens=est_out,
                        output_usd=(est_out / 1_000_000.0) * tout_p,
                    )
                step_tuples.append((is_p, b_len, b_len > 16000))

            if not step_tuples:
                continue
            records.append(
                _evaluate_step_sequence(
                    step_Tuples=step_tuples,
                    conv_bytes=conv_bytes,
                    short_id=session_id[:8],
                    conv_id=session_id,
                    harness="gemini_cli",
                    mtime=mtime,
                    in_price_per_m=in_price_per_m,
                    out_price_per_m=out_price_per_m,
                    top_tool_override="gemini_shell",
                    first_prompt=first_prompt,
                    exact_usage={
                        "metered_turns": m_turns,
                        "model_matched_turns": m_matched,
                        "billed_input_tokens": m_in,
                        "cache_create_tokens": 0,
                        "cache_read_tokens": m_cr,
                        "output_tokens": m_out,
                        "typed_prompt_tokens": typed_prompt_tok,
                        "fresh_input_usd": fresh_usd,
                        "cache_read_usd": cache_usd,
                        "output_usd": out_usd,
                    }
                    if m_turns > 0
                    else None,
                    model_usage=model_usage_map,
                )
            )
        except Exception:
            continue
    return _dedup_records(records)


def _extract_text_snippet(val: Any) -> str:
    if isinstance(val, str):
        return val.strip().replace("\n", " ")[:110]
    if isinstance(val, list):
        parts = []
        for item in val:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                t = item.get("text") or item.get("content") or item.get("input_text") or ""
                if isinstance(t, str) and t:
                    parts.append(t)
        return " ".join(parts).strip().replace("\n", " ")[:110]
    if isinstance(val, dict):
        return _extract_text_snippet(val.get("text") or val.get("content") or "")
    return ""


def scan_codex_harness(
    codex_dir: Optional[Any] = None,
    window_seconds: float = 0.0,
    now: float = 0.0,
    in_price_per_m: float = 5.0,
    out_price_per_m: float = 25.0,
    explicit_price_override: bool = False,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if now <= 0.0:
        now = time.time()

    candidates: List[Path] = []
    if codex_dir is not None:
        if isinstance(codex_dir, (list, tuple)):
            candidates.extend(Path(p).expanduser() for p in codex_dir)
        else:
            p = Path(codex_dir).expanduser()
            candidates.append(p)
            if p.is_dir():
                if (p / "sessions").is_dir():
                    candidates.append(p / "sessions")
                if (p / "history.jsonl").is_file():
                    candidates.append(p / "history.jsonl")
    else:
        env_sess = os.environ.get("CODEX_SESSIONS_DIR")
        env_home = os.environ.get("CODEX_HOME")
        if env_sess:
            p_s = Path(env_sess).expanduser()
            candidates.append(p_s)
            if p_s.parent.is_dir() and (p_s.parent / "history.jsonl").is_file():
                candidates.append(p_s.parent / "history.jsonl")
        elif env_home:
            base = Path(env_home).expanduser()
            candidates.extend([base / "sessions", base / "history.jsonl", base])
        else:
            for home in _iter_candidate_user_homes():
                home_codex = home / ".codex"
                candidates.extend([home_codex / "sessions", home_codex / "history.jsonl"])

    seen_files: set[str] = set()
    jsonl_files: List[Path] = []
    for cand in candidates:
        try:
            if cand.is_file() and cand.suffix == ".jsonl":
                rp = os.path.realpath(cand)
                if rp not in seen_files:
                    seen_files.add(rp)
                    jsonl_files.append(cand)
            elif cand.is_dir():
                for jf in cand.rglob("*.jsonl"):
                    if jf.is_file():
                        rp = os.path.realpath(jf)
                        if rp not in seen_files:
                            seen_files.add(rp)
                            jsonl_files.append(jf)
        except Exception:
            continue

    for jpath in jsonl_files:
        try:
            mtime = os.path.getmtime(jpath)
            if window_seconds > 0 and (now - mtime) > window_seconds:
                continue
            conv_bytes = os.path.getsize(jpath)
            step_tuples: List[Tuple[bool, int, bool]] = []
            tool_counts: Dict[str, int] = {}
            first_prompt = ""
            active_model = ""
            last_event_ts = 0.0
            m_turns = m_in = m_cc = m_cr = m_out = m_matched = typed_prompt_tok = 0
            fresh_usd = cache_usd = out_usd = 0.0
            prev_tot_tuple: Optional[Tuple[int, int, int, int]] = None
            model_usage_map: Dict[str, Dict[str, Any]] = {}
            with open(jpath, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                    except Exception:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    msg_obj = obj.get("message") if isinstance(obj.get("message"), dict) else {}
                    pay_obj = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}
                    ev_ts = _parse_iso_or_epoch_ts(
                        obj.get("timestamp") or obj.get("created_at") or pay_obj.get("timestamp")
                    )
                    if ev_ts > last_event_ts:
                        last_event_ts = ev_ts
                    raw_model = str(
                        pay_obj.get("model")
                        or msg_obj.get("model")
                        or obj.get("model")
                        or ""
                    ).strip()
                    if raw_model and raw_model != "<synthetic>":
                        eff_lvl = str(pay_obj.get("effort") or obj.get("effort") or "").strip().lower()
                        if eff_lvl in ("low", "medium", "high", "xhigh") and eff_lvl not in raw_model.lower():
                            active_model = f"{raw_model} ({eff_lvl.capitalize()})"
                        else:
                            active_model = raw_model
                    role = str(
                        obj.get("role")
                        or msg_obj.get("role")
                        or pay_obj.get("role")
                        or obj.get("type")
                        or ""
                    ).lower()
                    stype = str(obj.get("type") or "").lower()
                    ptype = str(pay_obj.get("type") or "").lower()

                    # Parse Codex token_count events and direct usage blocks
                    u_info = pay_obj.get("info") if (ptype == "token_count" and isinstance(pay_obj.get("info"), dict)) else None
                    direct_u = (
                        obj.get("usage")
                        if isinstance(obj.get("usage"), dict)
                        else (pay_obj.get("usage") if isinstance(pay_obj.get("usage"), dict) else None)
                    )
                    if isinstance(u_info, dict):
                        tot_u = u_info.get("total_token_usage") if isinstance(u_info.get("total_token_usage"), dict) else {}
                        last_u = u_info.get("last_token_usage") if isinstance(u_info.get("last_token_usage"), dict) else {}
                        tot_in = int(tot_u.get("input_tokens") or 0)
                        tot_cr = int(tot_u.get("cached_input_tokens") or 0)
                        tot_cc = int(tot_u.get("cache_write_input_tokens") or 0)
                        # In OpenAI/Codex TokenUsage, reasoning_output_tokens is already included inside output_tokens
                        tot_out = max(int(tot_u.get("output_tokens") or 0), int(tot_u.get("reasoning_output_tokens") or 0))
                        cur_tot = (tot_in, tot_cr, tot_cc, tot_out)
                        if (tot_in + tot_cr + tot_cc + tot_out) > 0 and cur_tot != prev_tot_tuple:
                            if prev_tot_tuple is not None and all(c >= p for c, p in zip(cur_tot, prev_tot_tuple)):
                                d_in = tot_in - prev_tot_tuple[0]
                                d_cr = tot_cr - prev_tot_tuple[1]
                                d_cc = tot_cc - prev_tot_tuple[2]
                                d_out = tot_out - prev_tot_tuple[3]
                            elif last_u and (int(last_u.get("input_tokens") or 0) + int(last_u.get("output_tokens") or 0)) > 0:
                                d_in = int(last_u.get("input_tokens") or 0)
                                d_cr = int(last_u.get("cached_input_tokens") or 0)
                                d_cc = int(last_u.get("cache_write_input_tokens") or 0)
                                d_out = max(int(last_u.get("output_tokens") or 0), int(last_u.get("reasoning_output_tokens") or 0))
                            else:
                                d_in, d_cr, d_cc, d_out = cur_tot
                            prev_tot_tuple = cur_tot
                            fresh_in = max(0, d_in - d_cr) if d_in >= d_cr else d_in
                            m_turns += 1
                            m_in += fresh_in
                            m_cc += d_cc
                            m_cr += d_cr
                            m_out += d_out
                            tin_p, tout_p, matched = _resolve_model_rates(
                                active_model, in_price_per_m, out_price_per_m, explicit_price_override
                            )
                            if matched:
                                m_matched += 1
                            t_fresh = (fresh_in / 1_000_000.0) * tin_p + (d_cc / 1_000_000.0) * (tin_p * 1.25)
                            t_cache = (d_cr / 1_000_000.0) * (tin_p * 0.10)
                            t_out_u = (d_out / 1_000_000.0) * tout_p
                            fresh_usd += t_fresh
                            cache_usd += t_cache
                            out_usd += t_out_u
                            if active_model:
                                _record_turn_model_usage(
                                    model_usage_map,
                                    active_model,
                                    turns=1,
                                    billed_input_tokens=fresh_in,
                                    cache_create_tokens=d_cc,
                                    cache_read_tokens=d_cr,
                                    output_tokens=d_out,
                                    fresh_usd=t_fresh,
                                    cache_usd=t_cache,
                                    output_usd=t_out_u,
                                )
                        continue
                    elif isinstance(direct_u, dict) and ("input_tokens" in direct_u or "output_tokens" in direct_u):
                        raw_in = int(direct_u.get("input_tokens") or 0)
                        in_det = direct_u.get("input_token_details") if isinstance(direct_u.get("input_token_details"), dict) else {}
                        d_cr = int(direct_u.get("cached_input_tokens") or in_det.get("cached_tokens") or 0)
                        fresh_in = max(0, raw_in - d_cr) if raw_in >= d_cr else raw_in
                        d_out = int(direct_u.get("output_tokens") or 0)
                        m_turns += 1
                        m_in += fresh_in
                        m_cr += d_cr
                        m_out += d_out
                        tin_p, tout_p, matched = _resolve_model_rates(
                            active_model, in_price_per_m, out_price_per_m, explicit_price_override
                        )
                        if matched:
                            m_matched += 1
                        t_fresh = (fresh_in / 1_000_000.0) * tin_p
                        t_cache = (d_cr / 1_000_000.0) * (tin_p * 0.10)
                        t_out_u = (d_out / 1_000_000.0) * tout_p
                        fresh_usd += t_fresh
                        cache_usd += t_cache
                        out_usd += t_out_u
                        if active_model:
                            _record_turn_model_usage(
                                model_usage_map,
                                active_model,
                                turns=1,
                                billed_input_tokens=fresh_in,
                                cache_read_tokens=d_cr,
                                output_tokens=d_out,
                                fresh_usd=t_fresh,
                                cache_usd=t_cache,
                                output_usd=t_out_u,
                            )

                    if stype in ("session_meta", "turn_context", "world_state"):
                        continue

                    eff_type = ptype or stype
                    is_p = role in ("assistant", "model", "planner_response") or eff_type in (
                        "assistant",
                        "model",
                        "planner_response",
                        "function_call",
                        "tool_call",
                        "reasoning",
                        "agent_message",
                    )
                    b_len = max(32, len(s.encode("utf-8")))
                    if (
                        role in ("user", "human", "user_input")
                        or eff_type in ("user", "user_message")
                        or ("text" in obj and role not in ("developer", "system"))
                    ):
                        snip = _extract_text_snippet(
                            obj.get("content")
                            or msg_obj.get("content")
                            or pay_obj.get("content")
                            or pay_obj.get("message")
                            or obj.get("text")
                            or ""
                        )
                        if snip and not snip.startswith("tool_output:") and not snip.startswith("<app-context>"):
                            typed_prompt_tok += _estimate_typed_prompt_tokens(snip)
                            if not first_prompt:
                                first_prompt = snip
                    for tc in obj.get("tool_calls") or msg_obj.get("tool_calls") or []:
                        if isinstance(tc, dict):
                            tname = str(
                                tc.get("name")
                                or (tc.get("function") or {}).get("name")
                                or "codex_shell"
                            )
                            tool_counts[tname] = tool_counts.get(tname, 0) + 1
                    if eff_type in ("function_call", "tool_call", "custom_tool_call", "command_execution"):
                        tname = str(obj.get("name") or pay_obj.get("name") or "codex_shell")
                        tool_counts[tname] = tool_counts.get(tname, 0) + 1
                    elif obj.get("tool_name"):
                        tname = str(obj.get("tool_name"))
                        tool_counts[tname] = tool_counts.get(tname, 0) + 1
                    step_tuples.append((is_p, b_len, b_len > 16000))
            if not step_tuples and m_turns == 0:
                continue
            if not step_tuples and m_turns > 0:
                step_tuples = [(True, max(64, conv_bytes // max(1, m_turns)), False) for _ in range(m_turns)]
            eff_ts = last_event_ts if last_event_ts > 0 else mtime
            if window_seconds > 0 and (now - eff_ts) > window_seconds:
                continue
            top_tool = (
                max(tool_counts.items(), key=lambda kv: kv[1])[0]
                if tool_counts
                else "codex_shell"
            )
            cid = jpath.stem if jpath.stem != "history" else f"codex-{jpath.parent.name or 'history'}"
            records.append(
                _evaluate_step_sequence(
                    step_Tuples=step_tuples,
                    conv_bytes=conv_bytes,
                    short_id=cid[:8],
                    conv_id=cid,
                    harness="codex",
                    mtime=eff_ts,
                    in_price_per_m=in_price_per_m,
                    out_price_per_m=out_price_per_m,
                    top_tool_override=top_tool,
                    first_prompt=first_prompt,
                    exact_usage={
                        "metered_turns": m_turns,
                        "model_matched_turns": m_matched,
                        "billed_input_tokens": m_in,
                        "cache_create_tokens": m_cc,
                        "cache_read_tokens": m_cr,
                        "output_tokens": m_out,
                        "typed_prompt_tokens": typed_prompt_tok,
                        "fresh_input_usd": fresh_usd,
                        "cache_read_usd": cache_usd,
                        "output_usd": out_usd,
                    } if m_turns > 0 else None,
                    model_usage=model_usage_map,
                )
            )
        except Exception:
            continue
    return records


def scan_cline_roo_harness(
    cline_dir: Optional[Any] = None,
    window_seconds: float = 0.0,
    now: float = 0.0,
    in_price_per_m: float = 5.0,
    out_price_per_m: float = 25.0,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if now <= 0.0:
        now = time.time()

    roots: List[Path] = []
    if cline_dir is not None:
        if isinstance(cline_dir, (list, tuple)):
            roots.extend(Path(p).expanduser() for p in cline_dir)
        else:
            roots.append(Path(cline_dir).expanduser())
    else:
        env_dir = os.environ.get("CLINE_STORAGE_DIR")
        if env_dir:
            roots.append(Path(env_dir).expanduser())
        else:
            appdata = os.environ.get("APPDATA", "").strip()
            if appdata:
                ap = Path(appdata)
                roots.extend(
                    [
                        ap / "Code" / "User" / "globalStorage" / "saoudrizwan.claude-dev" / "tasks",
                        ap / "Cursor" / "User" / "globalStorage" / "saoudrizwan.claude-dev" / "tasks",
                        ap / "Code" / "User" / "globalStorage" / "rooveterinaryinc.roo-cline" / "tasks",
                        ap / "Cursor" / "User" / "globalStorage" / "rooveterinaryinc.roo-cline" / "tasks",
                    ]
                )
            for home in _iter_candidate_user_homes():
                roots.extend(
                    [
                        home / "Library" / "Application Support" / "Code" / "User" / "globalStorage" / "saoudrizwan.claude-dev" / "tasks",
                        home / "Library" / "Application Support" / "Cursor" / "User" / "globalStorage" / "saoudrizwan.claude-dev" / "tasks",
                        home / "Library" / "Application Support" / "Code" / "User" / "globalStorage" / "rooveterinaryinc.roo-cline" / "tasks",
                        home / "Library" / "Application Support" / "Cursor" / "User" / "globalStorage" / "rooveterinaryinc.roo-cline" / "tasks",
                        home / "AppData" / "Roaming" / "Code" / "User" / "globalStorage" / "saoudrizwan.claude-dev" / "tasks",
                        home / "AppData" / "Roaming" / "Cursor" / "User" / "globalStorage" / "saoudrizwan.claude-dev" / "tasks",
                        home / "AppData" / "Roaming" / "Code" / "User" / "globalStorage" / "rooveterinaryinc.roo-cline" / "tasks",
                        home / "AppData" / "Roaming" / "Cursor" / "User" / "globalStorage" / "rooveterinaryinc.roo-cline" / "tasks",
                        home / ".config" / "Code" / "User" / "globalStorage" / "saoudrizwan.claude-dev" / "tasks",
                        home / ".config" / "Cursor" / "User" / "globalStorage" / "saoudrizwan.claude-dev" / "tasks",
                        home / ".config" / "Code" / "User" / "globalStorage" / "rooveterinaryinc.roo-cline" / "tasks",
                        home / ".config" / "Cursor" / "User" / "globalStorage" / "rooveterinaryinc.roo-cline" / "tasks",
                    ]
                )

    task_files: Dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        try:
            if root.is_file() and root.name in ("api_conversation_history.json", "ui_messages.json"):
                task_files[os.path.realpath(root.parent)] = root
                continue
            for pat in ("api_conversation_history.json", "ui_messages.json"):
                for fp in list(root.glob(f"*/{pat}")) + list(root.rglob(pat)):
                    if not fp.is_file():
                        continue
                    tdir_key = os.path.realpath(fp.parent)
                    # Prefer api_conversation_history.json when both exist in a task dir
                    if tdir_key not in task_files or fp.name == "api_conversation_history.json":
                        task_files[tdir_key] = fp
        except Exception:
            continue

    for _tdir_key, jpath in task_files.items():
        try:
            mtime = os.path.getmtime(jpath)
            if window_seconds > 0 and (now - mtime) > window_seconds:
                continue
            conv_bytes = os.path.getsize(jpath)
            raw_text = jpath.read_text(encoding="utf-8", errors="ignore")
            data = json.loads(raw_text)
            if isinstance(data, dict):
                items = data.get("messages") or data.get("history") or []
            elif isinstance(data, list):
                items = data
            else:
                items = []

            step_tuples: List[Tuple[bool, int, bool]] = []
            tool_counts: Dict[str, int] = {}
            first_prompt = ""
            for msg in items:
                if not isinstance(msg, dict):
                    continue
                raw_s = json.dumps(msg)
                b_len = max(32, len(raw_s.encode("utf-8")))
                role = str(msg.get("role") or msg.get("type") or "").lower()
                say_t = str(msg.get("say") or "").lower()
                is_p = role in ("assistant", "model", "planner_response") or say_t in (
                    "text",
                    "api_req_started",
                    "reasoning",
                    "tool",
                )
                content = msg.get("content") if "content" in msg else msg.get("text")
                if not first_prompt and (
                    role in ("user", "human") or say_t in ("task", "user_feedback") or msg.get("ask")
                ):
                    first_prompt = _extract_text_snippet(content)
                if isinstance(content, list):
                    for blk in content:
                        if isinstance(blk, dict) and blk.get("type") in ("tool_use", "tool_call"):
                            tname = str(blk.get("name") or "read_file")
                            tool_counts[tname] = tool_counts.get(tname, 0) + 1
                step_tuples.append((is_p, b_len, b_len > 16000))

            if not step_tuples:
                continue
            top_tool = (
                max(tool_counts.items(), key=lambda kv: kv[1])[0]
                if tool_counts
                else "read_file"
            )
            cid = jpath.parent.name or jpath.stem
            records.append(
                _evaluate_step_sequence(
                    step_Tuples=step_tuples,
                    conv_bytes=conv_bytes,
                    short_id=cid[:8],
                    conv_id=cid,
                    harness="cline_roo",
                    mtime=mtime,
                    in_price_per_m=in_price_per_m,
                    out_price_per_m=out_price_per_m,
                    top_tool_override=top_tool,
                    first_prompt=first_prompt,
                )
            )
        except Exception:
            continue
    return records


def scan_windsurf_harness(
    windsurf_dir: Optional[Any] = None,
    window_seconds: float = 0.0,
    now: float = 0.0,
    in_price_per_m: float = 5.0,
    out_price_per_m: float = 25.0,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if now <= 0.0:
        now = time.time()

    roots: List[Path] = []
    if windsurf_dir is not None:
        if isinstance(windsurf_dir, (list, tuple)):
            roots.extend(Path(p).expanduser() for p in windsurf_dir)
        else:
            roots.append(Path(windsurf_dir).expanduser())
    else:
        env_dir = os.environ.get("WINDSURF_STORAGE_DIR")
        if env_dir:
            roots.append(Path(env_dir).expanduser())
        else:
            appdata = os.environ.get("APPDATA", "").strip()
            if appdata:
                roots.append(Path(appdata) / "Windsurf" / "User" / "workspaceStorage")
            for home in _iter_candidate_user_homes():
                roots.extend(
                    [
                        home / "Library" / "Application Support" / "Windsurf" / "User" / "workspaceStorage",
                        home / "AppData" / "Roaming" / "Windsurf" / "User" / "workspaceStorage",
                        home / ".config" / "Windsurf" / "User" / "workspaceStorage",
                        home / ".codeium" / "windsurf" / "cascade",
                    ]
                )

    seen_paths: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        # 1. Scan state.vscdb files (SQLite ItemTable pattern)
        vscdb_files = (
            [root]
            if (root.is_file() and root.suffix == ".vscdb")
            else (list(root.glob("*/state.vscdb")) + list(root.glob("*.vscdb")))
        )
        for db_p in vscdb_files[:60]:
            if not db_p.is_file():
                continue
            rp = os.path.realpath(db_p)
            if rp in seen_paths:
                continue
            seen_paths.add(rp)
            try:
                mtime = os.path.getmtime(db_p)
                if window_seconds > 0 and (now - mtime) > window_seconds:
                    continue
                conn = sqlite3.connect(f"file:{db_p}?mode=ro", uri=True, timeout=1.5)
                rows = conn.execute(
                    "SELECT key, value FROM ItemTable WHERE key LIKE '%cascade%' OR key LIKE '%codeium%' OR key LIKE '%windsurf%' OR key LIKE '%aichat%' OR key LIKE '%composer%' LIMIT 40"
                ).fetchall()
                conn.close()
                if not rows:
                    continue
                step_tuples: List[Tuple[bool, int, bool]] = []
                tot_bytes = 0
                for _k, val in rows:
                    val_s = str(val or "")
                    b_len = len(val_s.encode("utf-8"))
                    if b_len < 64:
                        continue
                    tot_bytes += b_len
                    step_tuples.extend(_extract_steps_from_json_blob(val_s))
                if not step_tuples:
                    continue
                cid = db_p.parent.name if db_p.parent != root else db_p.stem
                records.append(
                    _evaluate_step_sequence(
                        step_Tuples=step_tuples,
                        conv_bytes=tot_bytes,
                        short_id=cid[:8],
                        conv_id=cid,
                        harness="windsurf",
                        mtime=mtime,
                        in_price_per_m=in_price_per_m,
                        out_price_per_m=out_price_per_m,
                        top_tool_override="windsurf_cascade",
                    )
                )
            except Exception:
                continue

        # 2. Scan cascade .jsonl / .json transcripts
        transcript_files = (
            [root]
            if (root.is_file() and root.suffix in (".jsonl", ".json"))
            else (list(root.rglob("*.jsonl")) + list(root.rglob("*.json")))
        )
        for tpath in transcript_files[:60]:
            if not tpath.is_file() or tpath.name == "workspace.json":
                continue
            rp = os.path.realpath(tpath)
            if rp in seen_paths:
                continue
            seen_paths.add(rp)
            try:
                mtime = os.path.getmtime(tpath)
                if window_seconds > 0 and (now - mtime) > window_seconds:
                    continue
                conv_bytes = os.path.getsize(tpath)
                step_tuples = []
                first_prompt = ""
                if tpath.suffix == ".jsonl":
                    with open(tpath, "r", encoding="utf-8", errors="ignore") as fh:
                        for line in fh:
                            s = line.strip()
                            if not s:
                                continue
                            try:
                                obj = json.loads(s)
                            except Exception:
                                continue
                            if not isinstance(obj, dict):
                                continue
                            role = str(
                                obj.get("role")
                                or obj.get("type")
                                or (obj.get("message") or {}).get("role")
                                or ""
                            ).lower()
                            is_p = role in ("assistant", "model", "planner_response", "cascade", "ai")
                            b_len = max(32, len(s.encode("utf-8")))
                            if not first_prompt and role in ("user", "human", "user_input"):
                                first_prompt = _extract_text_snippet(
                                    obj.get("content") or (obj.get("message") or {}).get("content") or ""
                                )
                            step_tuples.append((is_p, b_len, b_len > 16000))
                else:
                    data = json.loads(tpath.read_text(encoding="utf-8", errors="ignore"))
                    items = (
                        (
                            data.get("messages")
                            or data.get("turns")
                            or data.get("steps")
                            or data.get("history")
                            or data.get("trajectory")
                            or []
                        )
                        if isinstance(data, dict)
                        else (data if isinstance(data, list) else [])
                    )
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        raw_s = json.dumps(item)
                        b_len = max(32, len(raw_s.encode("utf-8")))
                        role = str(item.get("role") or item.get("type") or "").lower()
                        is_p = role in ("assistant", "model", "planner_response", "cascade", "ai")
                        if not first_prompt and role in ("user", "human", "user_input"):
                            first_prompt = _extract_text_snippet(item.get("content") or item.get("text") or "")
                        step_tuples.append((is_p, b_len, b_len > 16000))
                if not step_tuples:
                    continue
                cid = tpath.stem
                records.append(
                    _evaluate_step_sequence(
                        step_Tuples=step_tuples,
                        conv_bytes=conv_bytes,
                        short_id=cid[:8],
                        conv_id=cid,
                        harness="windsurf",
                        mtime=mtime,
                        in_price_per_m=in_price_per_m,
                        out_price_per_m=out_price_per_m,
                        top_tool_override="windsurf_cascade",
                        first_prompt=first_prompt,
                    )
                )
            except Exception:
                continue
    return records


def scan_aider_harness(
    aider_path: Optional[Any] = None,
    window_seconds: float = 0.0,
    now: float = 0.0,
    in_price_per_m: float = 5.0,
    out_price_per_m: float = 25.0,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if now <= 0.0:
        now = time.time()

    candidates: List[Path] = []
    if aider_path is not None:
        if isinstance(aider_path, (list, tuple)):
            candidates.extend(Path(p).expanduser() for p in aider_path)
        else:
            candidates.append(Path(aider_path).expanduser())
    else:
        env_p = os.environ.get("AIDER_HISTORY_PATH")
        if env_p:
            candidates.append(Path(env_p).expanduser())
        else:
            candidates.extend(
                [
                    Path.cwd() / ".aider.chat.history.md",
                    Path.home() / ".aider" / "caches",
                    Path.home() / ".aider.chat.history.md",
                ]
            )

    md_files: List[Path] = []
    seen_files: set[str] = set()
    for cand in candidates:
        try:
            if cand.is_file():
                rp = os.path.realpath(cand)
                if rp not in seen_files:
                    seen_files.add(rp)
                    md_files.append(cand)
            elif cand.is_dir():
                direct_md = cand / ".aider.chat.history.md"
                found = ([direct_md] if direct_md.is_file() else []) + list(cand.rglob("*.md")) + list(cand.rglob("*.jsonl"))
                for fp in found:
                    if fp.is_file():
                        rp = os.path.realpath(fp)
                        if rp not in seen_files:
                            seen_files.add(rp)
                            md_files.append(fp)
        except Exception:
            continue

    for fpath in md_files:
        try:
            mtime = os.path.getmtime(fpath)
            if window_seconds > 0 and (now - mtime) > window_seconds:
                continue
            conv_bytes = os.path.getsize(fpath)
            if conv_bytes <= 0:
                continue

            step_tuples: List[Tuple[bool, int, bool]] = []
            first_prompt = ""

            if fpath.suffix == ".jsonl":
                with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        s = line.strip()
                        if not s:
                            continue
                        try:
                            obj = json.loads(s)
                        except Exception:
                            continue
                        if not isinstance(obj, dict):
                            continue
                        role = str(obj.get("role") or obj.get("type") or "").lower()
                        is_p = role in ("assistant", "model", "planner_response")
                        b_len = max(32, len(s.encode("utf-8")))
                        if not first_prompt and role in ("user", "human"):
                            first_prompt = _extract_text_snippet(obj.get("content") or "")
                        step_tuples.append((is_p, b_len, b_len > 16000))
            else:
                raw_text = fpath.read_text(encoding="utf-8", errors="ignore")
                sections: List[Tuple[str, List[str]]] = []
                curr_header: Optional[str] = None
                curr_body: List[str] = []
                for line in raw_text.splitlines():
                    if line.startswith("#### "):
                        if curr_header is not None:
                            sections.append((curr_header, curr_body))
                        curr_header = line
                        curr_body = []
                    elif curr_header is not None:
                        curr_body.append(line)
                if curr_header is not None:
                    sections.append((curr_header, curr_body))

                has_explicit_role_headers = any(
                    h[5:].strip().lower().startswith(("user", "human", "assistant", "model", "ai"))
                    for h, _ in sections
                )
                for sec_idx, (hdr, body_lines) in enumerate(sections):
                    hdr_clean = hdr[5:].strip()
                    body_text = "\n".join(body_lines).strip()
                    full_sec = (hdr + ("\n" + body_text if body_text else "")).encode("utf-8")
                    if has_explicit_role_headers:
                        is_p = hdr_clean.lower().startswith(("assistant", "model", "ai"))
                        b_len = max(32, len(full_sec))
                        if not is_p and not first_prompt:
                            first_prompt = (body_text or hdr_clean).replace("\n", " ")[:110]
                        step_tuples.append((is_p, b_len, b_len > 16000))
                    else:
                        if not first_prompt and hdr_clean:
                            first_prompt = hdr_clean.replace("\n", " ")[:110]
                        if body_text:
                            u_len = max(32, len(hdr.encode("utf-8")))
                            a_len = max(32, len(body_text.encode("utf-8")))
                            step_tuples.append((False, u_len, u_len > 16000))
                            step_tuples.append((True, a_len, a_len > 16000))
                        else:
                            b_len = max(32, len(full_sec))
                            step_tuples.append((sec_idx % 2 == 1, b_len, b_len > 16000))

            if not step_tuples:
                continue
            cid = f"aider-{fpath.parent.name or fpath.stem}"
            records.append(
                _evaluate_step_sequence(
                    step_Tuples=step_tuples,
                    conv_bytes=conv_bytes,
                    short_id=cid[:8],
                    conv_id=cid,
                    harness="aider",
                    mtime=mtime,
                    in_price_per_m=in_price_per_m,
                    out_price_per_m=out_price_per_m,
                    top_tool_override="aider_edit",
                    first_prompt=first_prompt,
                )
            )
        except Exception:
            continue
    return records


def scan_continue_dev_harness(
    continue_dir: Optional[Path] = None,
    window_seconds: float = 0.0,
    now: float = 0.0,
    in_price_per_m: float = 5.0,
    out_price_per_m: float = 25.0,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if now <= 0.0:
        now = time.time()

    target_dir = Path(
        continue_dir
        or os.environ.get("CONTINUE_SESSIONS_DIR")
        or (Path.home() / ".continue" / "sessions")
    ).expanduser()

    json_files: List[Path] = []
    if target_dir.is_file() and target_dir.suffix == ".json":
        json_files.append(target_dir)
    elif target_dir.is_dir():
        json_files.extend(target_dir.glob("*.json"))

    for jpath in json_files:
        if not jpath.is_file() or jpath.name == "sessions.json":
            continue
        try:
            mtime = os.path.getmtime(jpath)
            if window_seconds > 0 and (now - mtime) > window_seconds:
                continue
            conv_bytes = os.path.getsize(jpath)
            data = json.loads(jpath.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(data, dict):
                items = data.get("history") or data.get("messages") or data.get("steps") or []
            elif isinstance(data, list):
                items = data
            else:
                items = []

            step_tuples: List[Tuple[bool, int, bool]] = []
            first_prompt = ""
            for item in items:
                if not isinstance(item, dict):
                    continue
                msg = item.get("message") if isinstance(item.get("message"), dict) else item
                if "role" not in msg and "content" not in msg and "type" not in item:
                    continue
                role = str(msg.get("role") or item.get("role") or item.get("type") or "").lower()
                is_p = role in ("assistant", "model", "planner_response", "ai")
                raw_s = json.dumps(item)
                b_len = max(32, len(raw_s.encode("utf-8")))
                if not first_prompt and role in ("user", "human", "user_input"):
                    first_prompt = _extract_text_snippet(msg.get("content") or item.get("content") or "")
                step_tuples.append((is_p, b_len, b_len > 16000))

            if not step_tuples:
                continue
            cid = str(data.get("sessionId") or jpath.stem) if isinstance(data, dict) else jpath.stem
            records.append(
                _evaluate_step_sequence(
                    step_Tuples=step_tuples,
                    conv_bytes=conv_bytes,
                    short_id=cid[:8],
                    conv_id=cid,
                    harness="continue_dev",
                    mtime=mtime,
                    in_price_per_m=in_price_per_m,
                    out_price_per_m=out_price_per_m,
                    top_tool_override="continue_chat",
                    first_prompt=first_prompt,
                )
            )
        except Exception:
            continue
    return records


def scan_zed_harness(
    zed_dir: Optional[Any] = None,
    window_seconds: float = 0.0,
    now: float = 0.0,
    in_price_per_m: float = 5.0,
    out_price_per_m: float = 25.0,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if now <= 0.0:
        now = time.time()

    roots: List[Path] = []
    if zed_dir is not None:
        if isinstance(zed_dir, (list, tuple)):
            roots.extend(Path(p).expanduser() for p in zed_dir)
        else:
            roots.append(Path(zed_dir).expanduser())
    else:
        env_d = os.environ.get("ZED_THREADS_DIR")
        if env_d:
            roots.append(Path(env_d).expanduser())
        else:
            home = Path.home()
            roots.extend(
                [
                    home / "Library" / "Application Support" / "Zed" / "threads",
                    home / ".local" / "share" / "zed" / "threads",
                ]
            )

    seen_files: set[str] = set()
    cand_files: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            rp = os.path.realpath(root)
            if rp not in seen_files:
                seen_files.add(rp)
                cand_files.append(root)
        elif root.is_dir():
            for ext in ("*.json", "*.jsonl", "*.db", "*.sqlite"):
                for fp in root.rglob(ext):
                    if fp.is_file():
                        rp = os.path.realpath(fp)
                        if rp not in seen_files:
                            seen_files.add(rp)
                            cand_files.append(fp)

    for fpath in cand_files[:80]:
        try:
            mtime = os.path.getmtime(fpath)
            if window_seconds > 0 and (now - mtime) > window_seconds:
                continue
            conv_bytes = os.path.getsize(fpath)
            step_tuples: List[Tuple[bool, int, bool]] = []
            first_prompt = ""

            if fpath.suffix in (".db", ".sqlite"):
                conn = sqlite3.connect(f"file:{fpath}?mode=ro", uri=True, timeout=1.5)
                tables = {
                    r[0]
                    for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                }
                for tbl in ("threads", "messages", "ItemTable"):
                    if tbl in tables:
                        rows = conn.execute(f"SELECT * FROM {tbl} LIMIT 100").fetchall()
                        for r in rows:
                            val_s = " ".join(str(c or "") for c in r)
                            step_tuples.extend(_extract_steps_from_json_blob(val_s))
                conn.close()
            elif fpath.suffix == ".jsonl":
                with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        s = line.strip()
                        if not s:
                            continue
                        try:
                            obj = json.loads(s)
                        except Exception:
                            continue
                        if not isinstance(obj, dict):
                            continue
                        role = str(obj.get("role") or obj.get("sender") or obj.get("type") or "").lower()
                        is_p = role in ("assistant", "model", "planner_response", "ai", "zed")
                        b_len = max(32, len(s.encode("utf-8")))
                        if not first_prompt and role in ("user", "human"):
                            first_prompt = _extract_text_snippet(obj.get("content") or obj.get("text") or "")
                        step_tuples.append((is_p, b_len, b_len > 16000))
            else:
                raw_json_txt = fpath.read_text(encoding="utf-8", errors="ignore")
                data = json.loads(raw_json_txt)
                items = (
                    (
                        data.get("messages")
                        or data.get("turns")
                        or data.get("steps")
                        or data.get("events")
                        or []
                    )
                    if isinstance(data, dict)
                    else (data if isinstance(data, list) else [])
                )
                for idx, item in enumerate(items):
                    if isinstance(item, dict):
                        raw_s = json.dumps(item)
                        b_len = max(32, len(raw_s.encode("utf-8")))
                        role = str(item.get("role") or item.get("sender") or item.get("type") or "").lower()
                        is_p = role in ("assistant", "model", "planner_response", "ai", "zed") or (
                            not role and idx % 2 == 1
                        )
                        if not first_prompt and not is_p:
                            first_prompt = _extract_text_snippet(item.get("content") or item.get("text") or "")
                        step_tuples.append((is_p, b_len, b_len > 16000))
                if not step_tuples and isinstance(data, dict) and conv_bytes >= 64:
                    step_tuples.extend(_extract_steps_from_json_blob(raw_json_txt))

            if not step_tuples:
                continue
            cid = fpath.stem
            records.append(
                _evaluate_step_sequence(
                    step_Tuples=step_tuples,
                    conv_bytes=conv_bytes,
                    short_id=cid[:8],
                    conv_id=cid,
                    harness="zed",
                    mtime=mtime,
                    in_price_per_m=in_price_per_m,
                    out_price_per_m=out_price_per_m,
                    top_tool_override="zed_assistant",
                    first_prompt=first_prompt,
                )
            )
        except Exception:
            continue
    return records


_TURN0_WORD_RE = re.compile(r"[A-Za-z]+|[0-9]{1,3}|[^\sA-Za-z0-9]")
_TURN0_INDENT_RE = re.compile(r"[ ]{2,}")
_TURN0_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.\-]")
_TURN0_SECRET_ID_RE = re.compile(
    r"^(?:sk-|ghp_|github_pat_|AKIA|AIza|phc_|xox[baprs]-)", re.IGNORECASE
)
DEFAULT_ESTIMATED_TOKENS_PER_MCP_SERVER = 1_800
DEFAULT_DEFERRED_TOKENS_PER_MCP_SERVER = 350


def _estimate_static_context_tokens(text: str, is_json: bool = False) -> int:
    """Calibrated zero-dependency BPE token estimator (<10% drift) for Markdown & JSON schemas."""
    if not text:
        return 0
    raw_bytes = len(text.encode("utf-8"))
    byte_est = raw_bytes / (3.55 if is_json else 3.95)

    sample = text[:262_144]
    indent_tokens = sum(max(1, len(m.group(0)) // 4) for m in _TURN0_INDENT_RE.finditer(sample))
    collapsed = _TURN0_INDENT_RE.sub(" ", sample)
    lex_tokens = float(indent_tokens)
    for tok in _TURN0_WORD_RE.findall(collapsed):
        c0 = tok[0]
        tlen = len(tok)
        if ("A" <= c0 <= "Z") or ("a" <= c0 <= "z"):
            if tlen <= 6:
                lex_tokens += 1.0
            elif tlen <= 11:
                lex_tokens += 1.65
            else:
                lex_tokens += float((tlen + 4) // 5)
        elif "0" <= c0 <= "9":
            lex_tokens += 1.0
        else:
            lex_tokens += 0.68 if is_json else 0.76

    if len(text) > len(sample) and len(sample) > 0:
        lex_tokens *= float(len(text)) / float(len(sample))

    return max(1, int(round((byte_est * 0.5) + (lex_tokens * 0.5))))


def _sanitize_turn0_label(raw_name: str) -> str:
    cleaned = _TURN0_SAFE_ID_RE.sub("", str(raw_name or "").strip())[:40]
    if not cleaned or _TURN0_SECRET_ID_RE.match(cleaned):
        return "redacted_mcp_server"
    return cleaned


def _format_turn0_rel_path(fpath: Path, ws_dir: Path, home_dir: Path) -> str:
    try:
        rel_ws = fpath.resolve().relative_to(ws_dir.resolve()).as_posix()
        return f"<workspace>/{rel_ws}"
    except Exception:
        pass
    try:
        rel_home = fpath.resolve().relative_to(home_dir.resolve()).as_posix()
        return f"~/{rel_home}"
    except Exception:
        return f"<local>/{fpath.name}"


def audit_turn0_static_overhead(
    workspace_dir: Optional[Path] = None,
    home_dir: Optional[Path] = None,
    sessions_count: int = 0,
    in_price_per_m: float = 5.0,
) -> Dict[str, Any]:
    """
    Blood-Brain Barrier - Turn-0 Fixed Context Tax Audit (<50ms, 100% stdlib, 0 PII leaks):
    Scans static instruction files (`CLAUDE.md`, `AGENTS.md`, `GEMINI.md`, `.cursorrules`, etc.)
    and MCP server tool configurations/schemas vs `0` MCP schema tokens for `body` OS hooks.
    """
    t0 = time.perf_counter()
    ws_dir = Path(
        workspace_dir or os.environ.get("WBC_AUDIT_WORKSPACE_DIR") or Path.cwd()
    ).expanduser()
    h_dir = Path(
        home_dir or os.environ.get("WBC_AUDIT_HOME_DIR") or Path.home()
    ).expanduser()

    instruction_candidates: List[Tuple[Path, str]] = [
        (ws_dir / "CLAUDE.md", "workspace"),
        (ws_dir / ".claude" / "CLAUDE.md", "workspace"),
        (ws_dir / "AGENTS.md", "workspace"),
        (ws_dir / ".agents" / "AGENTS.md", "workspace"),
        (ws_dir / "GEMINI.md", "workspace"),
        (ws_dir / ".gemini" / "GEMINI.md", "workspace"),
        (ws_dir / ".cursorrules", "workspace"),
        (ws_dir / ".windsurfrules", "workspace"),
        (ws_dir / ".clinerules", "workspace"),
        (ws_dir / ".github" / "copilot-instructions.md", "workspace"),
        (h_dir / ".claude" / "CLAUDE.md", "user_global"),
        (h_dir / ".gemini" / "GEMINI.md", "user_global"),
        (h_dir / ".gemini" / "antigravity" / "GEMINI.md", "user_global"),
        (h_dir / ".codex" / "AGENTS.md", "user_global"),
        (h_dir / ".codex" / "instructions.md", "user_global"),
    ]
    for rules_dir, r_scope in (
        (ws_dir / ".cursor" / "rules", "workspace"),
        (ws_dir / ".agents" / "rules", "workspace"),
        (ws_dir / ".claude" / "rules", "workspace"),
        (h_dir / ".gemini" / "rules", "user_global"),
        (h_dir / ".claude" / "rules", "user_global"),
    ):
        if rules_dir.is_dir():
            try:
                for rule_p in sorted(
                    list(rules_dir.glob("*.md")) + list(rules_dir.glob("*.mdc"))
                )[:16]:
                    instruction_candidates.append((rule_p, r_scope))
            except OSError:
                pass

    seen_realpaths: set[str] = set()
    instruction_files: List[Dict[str, Any]] = []
    instruction_bytes = 0
    instruction_tokens = 0

    for cand_p, scope in instruction_candidates:
        try:
            if not cand_p.is_file():
                continue
            rp = os.path.realpath(cand_p)
            if rp in seen_realpaths:
                continue
            seen_realpaths.add(rp)
            raw = cand_p.read_text(encoding="utf-8", errors="ignore")
            if not raw.strip():
                continue
            b_len = len(raw.encode("utf-8"))
            tok = _estimate_static_context_tokens(raw, is_json=False)
            instruction_bytes += b_len
            instruction_tokens += tok
            instruction_files.append(
                {
                    "path": _format_turn0_rel_path(cand_p, ws_dir, h_dir),
                    "scope": scope,
                    "bytes": b_len,
                    "tokens": tok,
                }
            )
        except OSError:
            continue

    disk_schema_by_server: Dict[str, Tuple[int, int]] = {}
    for schema_root in (
        h_dir / ".gemini" / "antigravity" / "mcp",
        ws_dir / ".mcp-schemas",
    ):
        if not schema_root.is_dir():
            continue
        try:
            for srv_dir in sorted(schema_root.iterdir())[:32]:
                if not srv_dir.is_dir() or srv_dir.name.startswith("."):
                    continue
                s_name = _sanitize_turn0_label(srv_dir.name)
                s_files = 0
                s_toks = 0
                for jf in sorted(srv_dir.glob("*.json"))[:64]:
                    if not jf.is_file():
                        continue
                    try:
                        j_raw = jf.read_text(encoding="utf-8", errors="ignore")
                    except OSError:
                        continue
                    if not j_raw.strip():
                        continue
                    s_files += 1
                    s_toks += _estimate_static_context_tokens(j_raw, is_json=True)
                if s_files > 0:
                    prev_f, prev_t = disk_schema_by_server.get(s_name, (0, 0))
                    disk_schema_by_server[s_name] = (prev_f + s_files, prev_t + s_toks)
        except OSError:
            continue

    mcp_config_candidates: List[Path] = [
        ws_dir / ".mcp.json",
        ws_dir / ".cursor" / "mcp.json",
        ws_dir / ".vscode" / "mcp.json",
        ws_dir / ".gemini" / "mcp_config.json",
        ws_dir / ".agents" / "mcp_config.json",
        h_dir / ".claude.json",
        h_dir / ".claude" / "mcp.json",
        h_dir / ".claude" / "claude_desktop_config.json",
        h_dir / ".cursor" / "mcp.json",
        h_dir / ".gemini" / "antigravity" / "mcp_config.json",
        h_dir / ".gemini" / "settings.json",
        h_dir / ".codeium" / "windsurf" / "mcp_config.json",
    ]

    mcp_servers_map: Dict[str, Dict[str, Any]] = {}
    seen_cfg_realpaths: set[str] = set()
    disabled_canonical_keys: set[str] = set()
    agy_mcp_config_present = False

    def _canonical_mcp_key(name: str) -> str:
        norm = re.sub(r"[^a-z0-9]+", "", str(name or "").lower())
        if norm.endswith("mcp") and len(norm) > 3:
            norm = norm[:-3]
        return norm

    def _register_servers_dict(servers_obj: Any, cfg_label: str) -> None:
        if not isinstance(servers_obj, dict):
            return
        for raw_srv_name, srv_cfg in list(servers_obj.items())[:48]:
            s_name = _sanitize_turn0_label(raw_srv_name)
            if isinstance(srv_cfg, dict) and srv_cfg.get("disabled") is True:
                disabled_canonical_keys.add(s_name)
                disabled_canonical_keys.add(_canonical_mcp_key(s_name))
                continue
            inline_tools = (
                srv_cfg.get("tools")
                if isinstance(srv_cfg, dict) and isinstance(srv_cfg.get("tools"), list)
                else None
            )
            if inline_tools:
                inline_json = json.dumps(inline_tools)
                t_cnt = len(inline_tools)
                t_tok = _estimate_static_context_tokens(inline_json, is_json=True)
                mcp_servers_map[s_name] = {
                    "name": s_name,
                    "source": cfg_label,
                    "schema_files": t_cnt,
                    "tokens": t_tok,
                    "provenance": "measured_inline_schema",
                }
            elif s_name in disk_schema_by_server:
                d_cnt, d_tok = disk_schema_by_server[s_name]
                mcp_servers_map[s_name] = {
                    "name": s_name,
                    "source": cfg_label,
                    "schema_files": d_cnt,
                    "tokens": d_tok,
                    "provenance": "measured_disk_schema",
                }
            elif s_name not in mcp_servers_map:
                low_cfg = cfg_label.lower()
                is_claude_deferred = (
                    ("claude" in low_cfg and "desktop" not in low_cfg)
                    or low_cfg.endswith("/.mcp.json")
                    or low_cfg == ".mcp.json"
                )
                est_tok = (
                    DEFAULT_DEFERRED_TOKENS_PER_MCP_SERVER
                    if is_claude_deferred
                    else DEFAULT_ESTIMATED_TOKENS_PER_MCP_SERVER
                )
                mcp_servers_map[s_name] = {
                    "name": s_name,
                    "source": cfg_label,
                    "schema_files": 0,
                    "tokens": est_tok,
                    "provenance": (
                        "deferred_tool_search_estimate"
                        if is_claude_deferred
                        else "calibrated_server_estimate"
                    ),
                }

    agy_cfg_path = h_dir / ".gemini" / "antigravity" / "mcp_config.json"
    for cfg_p in mcp_config_candidates:
        try:
            if not cfg_p.is_file():
                continue
            if cfg_p == agy_cfg_path:
                agy_mcp_config_present = True
            rp = os.path.realpath(cfg_p)
            if rp in seen_cfg_realpaths:
                continue
            seen_cfg_realpaths.add(rp)
            if os.path.getsize(cfg_p) > 4_000_000:
                continue
            parsed = json.loads(cfg_p.read_text(encoding="utf-8", errors="ignore"))
            if not isinstance(parsed, dict):
                continue
            cfg_lbl = _format_turn0_rel_path(cfg_p, ws_dir, h_dir)
            _register_servers_dict(parsed.get("mcpServers") or parsed.get("servers"), cfg_lbl)
            projects_obj = parsed.get("projects")
            if isinstance(projects_obj, dict):
                ws_key = str(ws_dir.resolve())
                proj_entry = projects_obj.get(ws_key) or projects_obj.get(str(ws_dir))
                if isinstance(proj_entry, dict):
                    _register_servers_dict(proj_entry.get("mcpServers"), cfg_lbl)
        except Exception:
            continue

    canonical_seen = {_canonical_mcp_key(k) for k in mcp_servers_map}

    if not agy_mcp_config_present or (ws_dir / ".mcp-schemas").is_dir():
        for s_name, (d_cnt, d_tok) in disk_schema_by_server.items():
            c_key = _canonical_mcp_key(s_name)
            if s_name in disabled_canonical_keys or c_key in disabled_canonical_keys:
                continue
            if agy_mcp_config_present and not (ws_dir / ".mcp-schemas" / s_name).is_dir():
                continue
            if s_name not in mcp_servers_map and c_key not in canonical_seen:
                canonical_seen.add(c_key)
                mcp_servers_map[s_name] = {
                    "name": s_name,
                    "source": "~/.gemini/antigravity/mcp",
                    "schema_files": d_cnt,
                    "tokens": d_tok,
                    "provenance": "measured_disk_schema",
                }

    mcp_servers_list = sorted(
        mcp_servers_map.values(), key=lambda x: int(x.get("tokens", 0)), reverse=True
    )
    mcp_measured_tokens = sum(
        int(s["tokens"])
        for s in mcp_servers_list
        if s["provenance"] in ("measured_disk_schema", "measured_inline_schema")
    )
    mcp_estimated_tokens = sum(
        int(s["tokens"])
        for s in mcp_servers_list
        if s["provenance"] in ("calibrated_server_estimate", "deferred_tool_search_estimate")
    )
    mcp_measured_schemas_count = sum(int(s["schema_files"]) for s in mcp_servers_list)
    mcp_turn0_tokens = mcp_measured_tokens + mcp_estimated_tokens
    total_turn0_per_session = instruction_tokens + mcp_turn0_tokens

    wbc_hook_active = any(
        p.is_file()
        for p in (
            ws_dir / "bin" / "body-immune-hook",
            h_dir / ".whole_body_compute" / "bin" / "body-immune-hook",
            h_dir / ".voicefi" / "bin" / "body-immune-hook",
        )
    )

    eff_sessions = max(1, int(sessions_count)) if sessions_count > 0 else 1
    cum_turn0_tokens = total_turn0_per_session * eff_sessions
    cum_mcp_tokens = mcp_turn0_tokens * eff_sessions
    est_30d_turn0_usd = round((cum_turn0_tokens / 1_000_000.0) * float(in_price_per_m), 2)
    est_30d_mcp_bloat_usd = round((cum_mcp_tokens / 1_000_000.0) * float(in_price_per_m), 2)

    if total_turn0_per_session >= 12_000 or mcp_turn0_tokens >= 9_000:
        status_tier = "HIGH_BLOAT"
    elif total_turn0_per_session >= 4_500 or mcp_turn0_tokens >= 3_000:
        status_tier = "MODERATE"
    else:
        status_tier = "LEAN"

    scan_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    summary = (
        f"{total_turn0_per_session:,} tok/session "
        f"({instruction_tokens:,} instruction tok across {len(instruction_files)} files + "
        f"{mcp_turn0_tokens:,} MCP schema tok across {len(mcp_servers_list)} servers vs 0 MCP tok for `body` OS hooks)"
    )

    return {
        "scan_latency_ms": scan_ms,
        "status_tier": status_tier,
        "instruction_files_count": len(instruction_files),
        "instruction_bytes": instruction_bytes,
        "instruction_tokens": instruction_tokens,
        "instruction_files": instruction_files,
        "mcp_servers_count": len(mcp_servers_list),
        "mcp_tool_schemas_measured_count": mcp_measured_schemas_count,
        "mcp_measured_schema_tokens": mcp_measured_tokens,
        "mcp_estimated_unindexed_tokens": mcp_estimated_tokens,
        "mcp_turn0_tokens": mcp_turn0_tokens,
        "mcp_servers": mcp_servers_list,
        "total_turn0_tokens_per_session": total_turn0_per_session,
        "wbc_os_hook_active": wbc_hook_active,
        "wbc_os_hook_turn0_tokens": 0,
        "avoidable_mcp_turn0_tokens_per_session": mcp_turn0_tokens,
        "cumulative_session_turn0_tokens": cum_turn0_tokens,
        "estimated_30d_turn0_usd": est_30d_turn0_usd,
        "estimated_30d_mcp_bloat_usd": est_30d_mcp_bloat_usd,
        "summary": summary,
    }


def _collect_wbc_shield_actuals(
    home_dir: Path,
    all_records: List[Dict[str, Any]],
    mtimes_by_id: Dict[str, float],
    window_seconds: float,
    now: float,
    enabled: bool = True,
    hook_fired_by_id: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """Cross-check global WBC hook configs, transcript hook records, and `immune_audit.jsonl` per session."""
    if not enabled:
        return {
            "wbc_active_on_machine": False,
            "wired_harnesses": [],
            "shielded_sessions": 0,
            "shielded_spend_usd": 0.0,
            "shielded_avg_session_usd": 0.0,
            "pre_or_unshielded_sessions": len(all_records),
            "pre_or_unshielded_spend_usd": round(sum(float(r.get("exact_api_spend_usd", 0.0)) for r in all_records), 2),
            "pre_or_unshielded_avg_session_usd": (
                round(sum(float(r.get("exact_api_spend_usd", 0.0)) for r in all_records) / len(all_records), 2)
                if all_records
                else 0.0
            ),
            "per_session_cost_drop_pct": 0.0,
            "unwired_active_harnesses": [],
            "unwired_projected_save_usd": round(sum(float(r.get("hal_saved_usd", 0.0)) for r in all_records), 2),
            "measured_bounded_slices": 0,
            "measured_direct_tokens_saved": 0,
            "measured_hook_events": 0,
        }

    hook_since_by_harness: Dict[str, float] = {}
    hook_cfg_candidates = (
        ("antigravity", home_dir / ".gemini" / "config" / "hooks.json"),
        (
            "antigravity",
            home_dir
            / ".gemini"
            / "config"
            / "plugins"
            / "whole-body-compute-plugin"
            / "hooks"
            / "hooks.json",
        ),
        ("claude_code", home_dir / ".claude" / "settings.json"),
        ("cursor", home_dir / ".cursor" / "hooks.json"),
        ("codex", home_dir / ".codex" / "hooks.json"),
    )
    for h_key, cfg_p in hook_cfg_candidates:
        try:
            if cfg_p.is_file():
                raw_cfg = cfg_p.read_text(encoding="utf-8", errors="ignore")
                if "body-immune-hook" in raw_cfg or "whole_body_compute" in raw_cfg:
                    mt = os.path.getmtime(cfg_p)
                    if h_key not in hook_since_by_harness or mt < hook_since_by_harness[h_key]:
                        hook_since_by_harness[h_key] = mt
        except OSError:
            continue

    cut_ts = (now - window_seconds) if window_seconds > 0 else 0.0
    seen_ev: set[Tuple[Any, ...]] = set()
    dedup_file_credits: Dict[Tuple[str, str], int] = {}
    immune_conv_ids: set[str] = set()
    bounded_slices = 0
    direct_tok_saved = 0
    hook_events = 0

    audit_paths: List[Path] = []
    if os.environ.get("WBC_IMMUNE_AUDIT_PATH"):
        audit_paths.append(Path(os.environ["WBC_IMMUNE_AUDIT_PATH"]).expanduser())
    else:
        audit_paths.extend(
            [
                home_dir / ".whole_body_compute" / "immune_audit.jsonl",
                home_dir / ".voicefi" / "immune_audit.jsonl",
            ]
        )

    for ap in audit_paths:
        if not ap.is_file():
            continue
        try:
            with open(ap, "r", encoding="utf-8", errors="ignore") as fh:
                for ln in fh:
                    s = ln.strip()
                    if not s:
                        continue
                    try:
                        ev = json.loads(s)
                    except Exception:
                        continue
                    if not isinstance(ev, dict):
                        continue
                    ts = float(ev.get("timestamp") or 0.0)
                    if cut_ts > 0 and ts < cut_ts:
                        continue
                    cid = str(ev.get("conversation_id") or "").strip()
                    ev_key = (round(ts, 3), ev.get("phase"), cid, ev.get("step_idx"), ev.get("tool"))
                    if ev_key in seen_ev:
                        continue
                    seen_ev.add(ev_key)
                    hook_events += 1
                    if cid and cid not in ("global", "default-session", "local"):
                        immune_conv_ids.add(cid)
                    if ev.get("bounded_slice_read") or ev.get("read_auto_sliced"):
                        bounded_slices += 1
                    raw_sav = max(0, int(ev.get("direct_tokens_saved") or 0))
                    if raw_sav > 0:
                        tgt = str(ev.get("target") or "").strip()
                        if tgt:
                            dk = (cid or "global", tgt)
                            prev_c = dedup_file_credits.get(dk, 0)
                            eff_s = max(0, max(prev_c, raw_sav) - prev_c)
                            dedup_file_credits[dk] = prev_c + eff_s
                            direct_tok_saved += eff_s
                        else:
                            direct_tok_saved += raw_sav
        except OSError:
            continue

    fired_map = hook_fired_by_id or {}
    has_per_session_telemetry = (hook_events > 0) or any(fired_map.values())
    wbc_active = bool(hook_since_by_harness or has_per_session_telemetry)
    shielded_recs: List[Dict[str, Any]] = []
    unshielded_recs: List[Dict[str, Any]] = []
    unwired_recs: List[Dict[str, Any]] = []
    unwired_harnesses_set: set[str] = set()

    for r in all_records:
        hk = str(r.get("harness", ""))
        cid = str(r.get("conv_id", ""))
        rmt = float(mtimes_by_id.get(cid, 0.0))
        if has_per_session_telemetry:
            is_sh = (cid in immune_conv_ids) or bool(fired_map.get(cid, False))
        elif hk in hook_since_by_harness:
            is_sh = rmt >= hook_since_by_harness[hk]
        else:
            is_sh = False
        if hk not in hook_since_by_harness and not is_sh:
            unwired_recs.append(r)
            unwired_harnesses_set.add(hk)
        if is_sh:
            shielded_recs.append(r)
        else:
            unshielded_recs.append(r)

    sh_n = len(shielded_recs)
    un_n = len(unshielded_recs)
    sh_usd = round(sum(float(r.get("exact_api_spend_usd", 0.0)) for r in shielded_recs), 2)
    un_usd = round(sum(float(r.get("exact_api_spend_usd", 0.0)) for r in unshielded_recs), 2)
    sh_avg = round(sh_usd / sh_n, 2) if sh_n > 0 else 0.0
    un_avg = round(un_usd / un_n, 2) if un_n > 0 else 0.0
    drop_pct = round(((un_avg - sh_avg) / un_avg) * 100.0, 1) if (sh_n >= 3 and un_n >= 3 and un_avg > sh_avg) else 0.0
    unwired_save_usd = round(sum(float(r.get("hal_saved_usd", 0.0)) for r in unwired_recs), 2)

    wired_harnesses_out = set(hook_since_by_harness.keys())
    for r in shielded_recs:
        if r.get("harness"):
            wired_harnesses_out.add(str(r["harness"]))

    return {
        "wbc_active_on_machine": wbc_active,
        "wired_harnesses": sorted(wired_harnesses_out),
        "shielded_sessions": sh_n,
        "shielded_spend_usd": sh_usd,
        "shielded_avg_session_usd": sh_avg,
        "pre_or_unshielded_sessions": un_n,
        "pre_or_unshielded_spend_usd": un_usd,
        "pre_or_unshielded_avg_session_usd": un_avg,
        "per_session_cost_drop_pct": drop_pct,
        "unwired_active_harnesses": sorted(unwired_harnesses_set),
        "unwired_projected_save_usd": unwired_save_usd,
        "measured_bounded_slices": bounded_slices,
        "measured_direct_tokens_saved": direct_tok_saved,
        "measured_hook_events": hook_events,
    }


def compute_model_opportunity_insights(
    all_records: List[Dict[str, Any]],
    model_usage_by_id: Dict[str, Dict[str, Dict[str, Any]]],
    window_days: int = 30,
) -> Dict[str, Any]:
    """
    Compute read-only model-opportunity insights across scanned transcripts (zero PII, zero mutations):
      1) Model Mix & Cost Breakdown (`models` + `tier_share_pct`)
      2) Over-Provisioned Routine Turns (`routine_downshift`: Heavy/Balanced models on short sprints or routine read/grep/bash/edit loops)
      3) Thinking / CoT Tax & Marathon Re-Read Drag (`thinking_cot_tax` + `marathon_reread_drag`)
    """
    agg: Dict[str, Dict[str, Any]] = {}
    tagged_sessions = 0
    untagged_sessions = 0
    untagged_turns = 0
    untagged_spend_usd = 0.0

    routine_sessions_set: set[str] = set()
    routine_turns = 0
    routine_current_usd = 0.0
    routine_target_usd = 0.0
    routine_models_spend: Dict[str, float] = {}
    has_claude_routine = False
    has_gemini_routine = False

    cot_sessions_set: set[str] = set()
    cot_turns = 0
    cot_output_tokens = 0
    cot_output_usd = 0.0
    cot_savable_usd = 0.0
    cot_models_spend: Dict[str, float] = {}

    drag_sessions_set: set[str] = set()
    drag_turns = 0
    drag_cache_tokens = 0
    drag_cache_usd = 0.0
    drag_savable_usd = 0.0
    drag_models_spend: Dict[str, float] = {}

    for rec in all_records:
        cid = str(rec.get("conv_id") or "")
        mu_map = model_usage_by_id.get(cid) or {}
        p_turns = int(rec.get("planner_turns", 0))
        s_spend = float(rec.get("exact_api_spend_usd", rec.get("conservative_spend_usd", 0.0)))
        top_tool = str(rec.get("top_offender_tool") or "none").strip().lower()
        is_routine_tool = top_tool in _ROUTINE_OPPORTUNITY_TOOLS
        is_short_sprint = p_turns <= 15
        is_medium_routine = p_turns <= 24 and is_routine_tool
        is_marathon_50 = p_turns >= 50

        if not mu_map:
            untagged_sessions += 1
            untagged_turns += p_turns
            untagged_spend_usd += s_spend
            continue

        tagged_sessions += 1
        mu_total_u = sum(float(v.get("spend_usd", 0.0)) for v in mu_map.values())
        mu_total_t = max(1, sum(int(v.get("turns", 0)) for v in mu_map.values()))

        for mkey, mv in mu_map.items():
            t_cnt = int(mv.get("turns", 0))
            if t_cnt <= 0:
                continue
            raw_u = float(mv.get("spend_usd", 0.0))
            fresh_u = float(mv.get("fresh_input_usd", 0.0))
            cache_u = float(mv.get("cache_read_usd", 0.0))
            out_u = float(mv.get("output_usd", 0.0))
            if mu_total_u <= 0.0 and s_spend > 0.0:
                share = float(t_cnt) / float(mu_total_t)
                raw_u = s_spend * share
                out_u = min(raw_u, out_u if out_u > 0 else raw_u * 0.22)
                rem_u = max(0.0, raw_u - out_u)
                fresh_u = rem_u * 0.42
                cache_u = rem_u * 0.58

            b_in = int(mv.get("billed_input_tokens", 0))
            b_cc = int(mv.get("cache_create_tokens", 0))
            b_cr = int(mv.get("cache_read_tokens", 0))
            b_out = int(mv.get("output_tokens", 0))
            tier = str(mv.get("tier") or "balanced")
            lbl = str(mv.get("label") or mkey)
            fam = str(mv.get("family") or "other")
            is_ht = bool(mv.get("high_thinking", False))

            row = agg.get(mkey)
            if row is None:
                row = {
                    "key": mkey,
                    "label": lbl,
                    "tier": tier,
                    "family": fam,
                    "high_thinking": is_ht,
                    "_sessions": set(),
                    "turns": 0,
                    "billed_input_tokens": 0,
                    "cache_read_tokens": 0,
                    "output_tokens": 0,
                    "fresh_input_usd": 0.0,
                    "cache_read_usd": 0.0,
                    "output_usd": 0.0,
                    "spend_usd": 0.0,
                }
                agg[mkey] = row
            row["_sessions"].add(cid)
            row["turns"] += t_cnt
            row["billed_input_tokens"] += b_in + b_cc
            row["cache_read_tokens"] += b_cr
            row["output_tokens"] += b_out
            row["fresh_input_usd"] += fresh_u
            row["cache_read_usd"] += cache_u
            row["output_usd"] += out_u
            row["spend_usd"] += raw_u

            # Opportunity 1: Over-Provisioned Routine Turns (Heavy/Balanced on short sprints or routine loops)
            target_in_p: Optional[float] = None
            target_out_p: Optional[float] = None
            if tier == "heavy_thinking" and (is_short_sprint or is_medium_routine):
                if fam == "claude":
                    target_in_p, target_out_p = 1.0, 5.0  # Haiku 4.5 ($1/$5)
                    has_claude_routine = True
                else:
                    target_in_p, target_out_p = 0.50, 2.50  # Hydrogen Low / Neon Low ($0.50/$2.50)
                    has_gemini_routine = True
            elif tier == "balanced" and (p_turns <= 12 and is_routine_tool):
                if fam == "claude":
                    target_in_p, target_out_p = 1.0, 5.0
                    has_claude_routine = True
                else:
                    target_in_p, target_out_p = 0.50, 2.50
                    has_gemini_routine = True

            if target_in_p is not None and target_out_p is not None and raw_u > 0.0:
                if (b_in + b_cc + b_cr + b_out) > 0:
                    tgt_u = (
                        (b_in / 1_000_000.0) * target_in_p
                        + (b_cc / 1_000_000.0) * (target_in_p * 1.25)
                        + (b_cr / 1_000_000.0) * (target_in_p * 0.10)
                        + (b_out / 1_000_000.0) * target_out_p
                    )
                else:
                    mult = 0.22 if tier == "heavy_thinking" else 0.33
                    tgt_u = raw_u * mult
                if raw_u > tgt_u + 0.005:
                    routine_sessions_set.add(cid)
                    routine_turns += t_cnt
                    routine_current_usd += raw_u
                    routine_target_usd += tgt_u
                    routine_models_spend[lbl] = routine_models_spend.get(lbl, 0.0) + (raw_u - tgt_u)

            # Opportunity 2A: Thinking / CoT Output Tax (high-thinking / heavy models or >28% output share)
            out_share = (out_u / raw_u) if raw_u > 0 else 0.0
            avg_out_per_t = (float(b_out) / float(t_cnt)) if t_cnt > 0 else 0.0
            if out_u > 0.02 and (is_ht or tier == "heavy_thinking" or (out_share >= 0.28 and avg_out_per_t >= 700.0)):
                # Grounded in Live Neon XHigh CoT-Bloat A/B (-23.5% cost / -19.1% CoT+out) + Low-effort routing (-50% CoT)
                cot_cut = 0.38 if (is_ht or tier == "heavy_thinking") else 0.22
                sav_cot = out_u * cot_cut
                cot_sessions_set.add(cid)
                cot_turns += t_cnt
                cot_output_tokens += b_out
                cot_output_usd += out_u
                cot_savable_usd += sav_cot
                cot_models_spend[lbl] = cot_models_spend.get(lbl, 0.0) + sav_cot

            # Opportunity 2B: Marathon Re-Read Drag (50+ turn heavy/balanced marathons re-reading cached history)
            if is_marathon_50 and tier in ("heavy_thinking", "balanced") and (cache_u + fresh_u * 0.35) > 0.05:
                # Grounded in Glymphatic Flush + AST Slicer marathon history cut (~62% cache re-read reduction)
                reread_base = cache_u + (fresh_u * 0.25 if cache_u == 0.0 else 0.0)
                sav_drag = reread_base * (0.64 if tier == "heavy_thinking" else 0.54)
                drag_sessions_set.add(cid)
                drag_turns += t_cnt
                drag_cache_tokens += b_cr
                drag_cache_usd += reread_base
                drag_savable_usd += sav_drag
                drag_models_spend[lbl] = drag_models_spend.get(lbl, 0.0) + sav_drag

    models_list: List[Dict[str, Any]] = []
    tagged_spend_total = sum(float(r["spend_usd"]) for r in agg.values())
    tagged_turns_total = sum(int(r["turns"]) for r in agg.values())

    tier_spend = {"heavy_thinking": 0.0, "balanced": 0.0, "fast_low": 0.0}
    tier_turns = {"heavy_thinking": 0, "balanced": 0, "fast_low": 0}

    for r in sorted(agg.values(), key=lambda x: (float(x["spend_usd"]), int(x["turns"])), reverse=True):
        s_cnt = len(r.pop("_sessions"))
        sp_u = round(float(r["spend_usd"]), 2)
        out_u = round(float(r["output_usd"]), 2)
        cr_u = round(float(r["cache_read_usd"]), 2)
        fr_u = round(float(r["fresh_input_usd"]), 2)
        tr_k = str(r["tier"])
        if tr_k in tier_spend:
            tier_spend[tr_k] += float(r["spend_usd"])
            tier_turns[tr_k] += int(r["turns"])
        models_list.append(
            {
                "key": r["key"],
                "label": r["label"],
                "tier": tr_k,
                "family": r["family"],
                "high_thinking": bool(r["high_thinking"]),
                "sessions": s_cnt,
                "turns": int(r["turns"]),
                "billed_input_tokens": int(r["billed_input_tokens"]),
                "cache_read_tokens": int(r["cache_read_tokens"]),
                "output_tokens": int(r["output_tokens"]),
                "fresh_input_usd": fr_u,
                "cache_read_usd": cr_u,
                "output_usd": out_u,
                "spend_usd": sp_u,
                "avg_session_usd": round(float(r["spend_usd"]) / float(max(1, s_cnt)), 2),
                "spend_share_pct": (
                    round((float(r["spend_usd"]) / tagged_spend_total) * 100.0, 1)
                    if tagged_spend_total > 0
                    else 0.0
                ),
                "output_share_pct": (
                    round((float(r["output_usd"]) / float(r["spend_usd"])) * 100.0, 1)
                    if float(r["spend_usd"]) > 0
                    else 0.0
                ),
            }
        )

    routine_savable = round(max(0.0, routine_current_usd - routine_target_usd), 2)
    cot_savable = round(max(0.0, cot_savable_usd), 2)
    drag_savable = round(max(0.0, drag_savable_usd), 2)
    # De-duplicate overlap conservatively when summing total model-routing + flush opportunity
    combined_opportunity_usd = round(
        min(
            max(0.0, tagged_spend_total * 0.78),
            routine_savable + cot_savable * 0.85 + drag_savable * 0.85,
        ),
        2,
    )
    scale_30d = 30.0 / float(max(1, window_days)) if window_days > 0 else 1.0
    projected_annual_usd = round(combined_opportunity_usd * scale_30d * 12.0, 2)

    if has_claude_routine and has_gemini_routine:
        target_driver = "Haiku 4.5 / Hydrogen (Low) + Local Metal"
    elif has_claude_routine:
        target_driver = "Claude Haiku 4.5 + Local Metal"
    else:
        target_driver = "Gemini 4 Hydrogen/Neon (Low) + Local Metal"

    top_routine_models = [
        k for k, _ in sorted(routine_models_spend.items(), key=lambda kv: kv[1], reverse=True)[:2]
    ]
    top_cot_models = [
        k for k, _ in sorted(cot_models_spend.items(), key=lambda kv: kv[1], reverse=True)[:2]
    ]
    top_drag_models = [
        k for k, _ in sorted(drag_models_spend.items(), key=lambda kv: kv[1], reverse=True)[:2]
    ]

    opportunities: List[Dict[str, Any]] = []
    if routine_savable >= 0.05 and routine_turns > 0:
        src_lbl = ", ".join(top_routine_models) if top_routine_models else "Flagship models"
        opportunities.append(
            {
                "id": "routine_downshift",
                "title": "Routine Sprint Downshift",
                "sessions": len(routine_sessions_set),
                "turns": routine_turns,
                "current_spend_usd": round(routine_current_usd, 2),
                "savable_usd": routine_savable,
                "source_models": top_routine_models,
                "recommended_target": target_driver,
                "summary": (
                    f"{len(routine_sessions_set)} short/routine sessions ({routine_turns:,}t) on {src_lbl} "
                    f"-> route first-pass edits/reads to {target_driver} (~${routine_savable:,.2f} savable)"
                ),
            }
        )
    if drag_savable >= 0.05 and len(drag_sessions_set) > 0:
        src_lbl = ", ".join(top_drag_models) if top_drag_models else "Flagship models"
        opportunities.append(
            {
                "id": "marathon_reread_drag",
                "title": "Marathon Re-Read Drag",
                "sessions": len(drag_sessions_set),
                "turns": drag_turns,
                "cache_read_tokens": drag_cache_tokens,
                "current_reread_usd": round(drag_cache_usd, 2),
                "savable_usd": drag_savable,
                "source_models": top_drag_models,
                "summary": (
                    f"{len(drag_sessions_set)} marathon sessions ({drag_turns:,}t) on {src_lbl} re-read "
                    f"${drag_cache_usd:,.2f} of history -> run `body flush` at ~50t (~${drag_savable:,.2f} savable)"
                ),
            }
        )
    if cot_savable >= 0.05 and cot_turns > 0:
        src_lbl = ", ".join(top_cot_models) if top_cot_models else "High-thinking models"
        opportunities.append(
            {
                "id": "thinking_cot_tax",
                "title": "Thinking / CoT Output Tax",
                "sessions": len(cot_sessions_set),
                "turns": cot_turns,
                "output_tokens": cot_output_tokens,
                "current_output_usd": round(cot_output_usd, 2),
                "savable_usd": cot_savable,
                "source_models": top_cot_models,
                "summary": (
                    f"{cot_turns:,} high-thinking turns ({cot_output_tokens:,} CoT/out tok, ${cot_output_usd:,.2f}) on {src_lbl} "
                    f"-> BBB AST slicing + Low-effort default cuts CoT ~24-38% (~${cot_savable:,.2f} savable)"
                ),
            }
        )

    opportunities.sort(key=lambda x: float(x.get("savable_usd", 0.0)), reverse=True)

    return {
        "read_only": True,
        "window_days": int(window_days),
        "tagged_sessions": tagged_sessions,
        "tagged_turns": tagged_turns_total,
        "tagged_spend_usd": round(tagged_spend_total, 2),
        "untagged_sessions": untagged_sessions,
        "untagged_turns": untagged_turns,
        "untagged_spend_usd": round(untagged_spend_usd, 2),
        "tier_breakdown": {
            "heavy_thinking": {
                "turns": tier_turns["heavy_thinking"],
                "spend_usd": round(tier_spend["heavy_thinking"], 2),
                "spend_share_pct": (
                    round((tier_spend["heavy_thinking"] / tagged_spend_total) * 100.0, 1)
                    if tagged_spend_total > 0
                    else 0.0
                ),
            },
            "balanced": {
                "turns": tier_turns["balanced"],
                "spend_usd": round(tier_spend["balanced"], 2),
                "spend_share_pct": (
                    round((tier_spend["balanced"] / tagged_spend_total) * 100.0, 1)
                    if tagged_spend_total > 0
                    else 0.0
                ),
            },
            "fast_low": {
                "turns": tier_turns["fast_low"],
                "spend_usd": round(tier_spend["fast_low"], 2),
                "spend_share_pct": (
                    round((tier_spend["fast_low"] / tagged_spend_total) * 100.0, 1)
                    if tagged_spend_total > 0
                    else 0.0
                ),
            },
        },
        "models": models_list,
        "routine_downshift": {
            "sessions": len(routine_sessions_set),
            "turns": routine_turns,
            "current_spend_usd": round(routine_current_usd, 2),
            "savable_usd": routine_savable,
            "recommended_target": target_driver,
            "source_models": top_routine_models,
        },
        "thinking_cot_tax": {
            "sessions": len(cot_sessions_set),
            "turns": cot_turns,
            "output_tokens": cot_output_tokens,
            "current_output_usd": round(cot_output_usd, 2),
            "savable_usd": cot_savable,
            "source_models": top_cot_models,
        },
        "marathon_reread_drag": {
            "sessions": len(drag_sessions_set),
            "turns": drag_turns,
            "cache_read_tokens": drag_cache_tokens,
            "current_reread_usd": round(drag_cache_usd, 2),
            "savable_usd": drag_savable,
            "source_models": top_drag_models,
        },
        "total_opportunity_usd": combined_opportunity_usd,
        "projected_annual_opportunity_usd": projected_annual_usd,
        "opportunities": opportunities,
    }


def run_multi_harness_finops_audit(
    days: int = 30,
    price_per_m: float = 5.0,
    seats: int = 10,
    top: int = 5,
    agy_dir: Optional[Path] = None,
    claude_dir: Optional[Path] = None,
    goose_db: Optional[Path] = None,
    cursor_dir: Optional[Path] = None,
    codex_dir: Optional[Path] = None,
    cline_dir: Optional[Path] = None,
    windsurf_dir: Optional[Path] = None,
    aider_path: Optional[Path] = None,
    continue_dir: Optional[Path] = None,
    zed_dir: Optional[Path] = None,
    gemini_cli_dir: Optional[Path] = None,
    workspace_dir: Optional[Path] = None,
    home_dir: Optional[Path] = None,
    explicit_price_override: bool = False,
) -> Dict[str, Any]:
    now = time.time()
    window_seconds = max(1, int(days)) * 86400.0 if days > 0 else 0.0
    in_price = float(price_per_m)
    out_price = round(in_price * 5.0, 2)

    any_custom_override = any(
        x is not None
        for x in (
            agy_dir,
            claude_dir,
            goose_db,
            cursor_dir,
            codex_dir,
            cline_dir,
            windsurf_dir,
            aider_path,
            continue_dir,
            zed_dir,
            gemini_cli_dir,
        )
    ) or any(
        os.environ.get(k)
        for k in (
            "AGY_BRAIN_DIR",
            "CLAUDE_CODE_DIR",
            "GOOSE_SESSIONS_DB",
            "CURSOR_STORAGE_DIR",
            "CODEX_SESSIONS_DIR",
            "CODEX_HOME",
            "CLINE_STORAGE_DIR",
            "WINDSURF_STORAGE_DIR",
            "AIDER_HISTORY_PATH",
            "CONTINUE_SESSIONS_DIR",
            "ZED_THREADS_DIR",
            "GEMINI_CLI_DIR",
        )
    )

    candidate_homes = _iter_candidate_user_homes()
    agy_roots = (
        [Path(agy_dir or os.environ["AGY_BRAIN_DIR"]).expanduser()]
        if (agy_dir is not None or os.environ.get("AGY_BRAIN_DIR"))
        else [h / ".gemini" / "antigravity" / "brain" for h in candidate_homes]
    )
    claude_roots = (
        [Path(claude_dir or os.environ["CLAUDE_CODE_DIR"]).expanduser()]
        if (claude_dir is not None or os.environ.get("CLAUDE_CODE_DIR"))
        else [h / ".claude" / "projects" for h in candidate_homes]
    )
    goose_roots = (
        [Path(goose_db or os.environ["GOOSE_SESSIONS_DB"]).expanduser()]
        if (goose_db is not None or os.environ.get("GOOSE_SESSIONS_DB"))
        else [
            p
            for h in candidate_homes
            for p in (
                h / ".config" / "goose" / "sessions.db",
                h / "AppData" / "Roaming" / "Block" / "goose" / "data" / "sessions.db",
            )
        ]
    )
    cursor_override = (
        Path(cursor_dir or os.environ["CURSOR_STORAGE_DIR"]).expanduser()
        if (cursor_dir is not None or os.environ.get("CURSOR_STORAGE_DIR"))
        else None
    )

    agy_recs = (
        _dedup_records(
            [
                r
                for root in agy_roots
                for r in scan_antigravity_harness(
                    root, window_seconds, now, in_price, out_price, explicit_price_override=explicit_price_override
                )
            ]
        )
        if (not any_custom_override or agy_dir is not None or os.environ.get("AGY_BRAIN_DIR"))
        else []
    )
    gemini_cli_recs = (
        scan_gemini_cli_harness(gemini_cli_dir, window_seconds, now, in_price, out_price)
        if (not any_custom_override or gemini_cli_dir is not None or os.environ.get("GEMINI_CLI_DIR"))
        else []
    )
    claude_recs = (
        _dedup_records(
            [
                r
                for root in claude_roots
                for r in scan_claude_code_harness(
                    root, window_seconds, now, in_price, out_price, explicit_price_override=explicit_price_override
                )
            ]
        )
        if (not any_custom_override or claude_dir is not None or os.environ.get("CLAUDE_CODE_DIR"))
        else []
    )
    goose_recs = (
        _dedup_records(
            [
                r
                for root in goose_roots
                for r in scan_goose_harness(root, window_seconds, now, in_price, out_price)
            ]
        )
        if (not any_custom_override or goose_db is not None or os.environ.get("GOOSE_SESSIONS_DB"))
        else []
    )
    cursor_recs = (
        scan_cursor_harness(cursor_override, window_seconds, now, in_price, out_price)
        if (not any_custom_override or cursor_dir is not None or os.environ.get("CURSOR_STORAGE_DIR"))
        else []
    )
    codex_recs = (
        scan_codex_harness(
            codex_dir,
            window_seconds,
            now,
            in_price,
            out_price,
            explicit_price_override=explicit_price_override,
        )
        if (
            not any_custom_override
            or codex_dir is not None
            or os.environ.get("CODEX_SESSIONS_DIR")
            or os.environ.get("CODEX_HOME")
        )
        else []
    )
    cline_recs = (
        scan_cline_roo_harness(cline_dir, window_seconds, now, in_price, out_price)
        if (not any_custom_override or cline_dir is not None or os.environ.get("CLINE_STORAGE_DIR"))
        else []
    )
    windsurf_recs = (
        scan_windsurf_harness(windsurf_dir, window_seconds, now, in_price, out_price)
        if (not any_custom_override or windsurf_dir is not None or os.environ.get("WINDSURF_STORAGE_DIR"))
        else []
    )
    aider_recs = (
        scan_aider_harness(aider_path, window_seconds, now, in_price, out_price)
        if (not any_custom_override or aider_path is not None or os.environ.get("AIDER_HISTORY_PATH"))
        else []
    )
    continue_recs = (
        scan_continue_dev_harness(continue_dir, window_seconds, now, in_price, out_price)
        if (not any_custom_override or continue_dir is not None or os.environ.get("CONTINUE_SESSIONS_DIR"))
        else []
    )
    zed_recs = (
        scan_zed_harness(zed_dir, window_seconds, now, in_price, out_price)
        if (not any_custom_override or zed_dir is not None or os.environ.get("ZED_THREADS_DIR"))
        else []
    )

    all_records = (
        agy_recs
        + gemini_cli_recs
        + claude_recs
        + goose_recs
        + cursor_recs
        + codex_recs
        + cline_recs
        + windsurf_recs
        + aider_recs
        + continue_recs
        + zed_recs
    )
    all_records.sort(
        key=lambda r: (r.get("exact_api_spend_usd", 0.0), r.get("hal_saved_usd", 0.0), r["estimated_spend_usd"]),
        reverse=True,
    )

    def _summarize_harness(label: str, recs: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "label": label,
            "threads": len(recs),
            "steps": sum(r["steps"] for r in recs),
            "planner_turns": sum(r["planner_turns"] for r in recs),
            "metered_turns": sum(int(r.get("metered_turns", 0)) for r in recs),
            "transcript_kb": round(sum(r["transcript_bytes"] for r in recs) / 1024.0, 1),
            "actual_input_tokens": sum(r["actual_cumulative_input_tokens"] for r in recs),
            "metered_billed_input_tokens": sum(int(r.get("metered_billed_input_tokens", 0)) for r in recs),
            "metered_cache_create_tokens": sum(int(r.get("metered_cache_create_tokens", 0)) for r in recs),
            "metered_cache_read_tokens": sum(int(r.get("metered_cache_read_tokens", 0)) for r in recs),
            "metered_output_tokens": sum(int(r.get("metered_output_tokens", 0)) for r in recs),
            "wasted_tokens": sum(r["wasted_bloat_tokens"] for r in recs),
            "exact_api_spend_usd": round(sum(float(r.get("exact_api_spend_usd", 0.0)) for r in recs), 2),
            "hal_wbc_spend_usd": round(sum(float(r.get("hal_wbc_spend_usd", 0.0)) for r in recs), 2),
            "hal_saved_usd": round(sum(float(r.get("hal_saved_usd", 0.0)) for r in recs), 2),
            "conservative_savable_usd": round(sum(r["conservative_savable_usd"] for r in recs), 2),
            "blended_savable_usd": round(sum(r["usd_savable"] for r in recs), 2),
        }

    harness_breakdown = {
        "antigravity": _summarize_harness("Google Antigravity", agy_recs),
        "gemini_cli": _summarize_harness("Google Gemini CLI", gemini_cli_recs),
        "claude_code": _summarize_harness("Anthropic Claude Code", claude_recs),
        "cursor": _summarize_harness("Cursor Agent / Composer", cursor_recs),
        "goose": _summarize_harness("Block Goose CLI", goose_recs),
        "codex": _summarize_harness("OpenAI Codex CLI", codex_recs),
        "cline_roo": _summarize_harness("Cline / Roo Code", cline_recs),
        "windsurf": _summarize_harness("Windsurf Cascade", windsurf_recs),
        "aider": _summarize_harness("Aider CLI", aider_recs),
        "continue_dev": _summarize_harness("Continue.dev", continue_recs),
        "zed": _summarize_harness("Zed AI", zed_recs),
    }

    total_steps = sum(r["steps"] for r in all_records)
    total_planner_turns = sum(r["planner_turns"] for r in all_records)
    total_transcript_bytes = sum(r["transcript_bytes"] for r in all_records)
    total_raw_unbounded_tokens = sum(r["raw_unbounded_quadratic_tokens"] for r in all_records)
    total_actual_input_tokens = sum(r["actual_cumulative_input_tokens"] for r in all_records)
    total_output_tokens = sum(r["output_tokens"] for r in all_records)
    total_wasted_bloat_tokens = sum(r["wasted_bloat_tokens"] for r in all_records)
    total_quadratic_post_sat_tokens = sum(r["quadratic_post_saturation_tokens"] for r in all_records)
    total_unfiltered_tool_bloat_tokens = sum(r["unfiltered_tool_bloat_tokens"] for r in all_records)
    total_bicameral_offload_tokens = sum(r["bicameral_local_offload_tokens"] for r in all_records)
    total_bbb_optimized_tokens = sum(r["bbb_optimized_tokens"] for r in all_records)
    total_bbb_optimized_output_tokens = sum(r["bbb_optimized_output_tokens"] for r in all_records)

    metered_turns_total = sum(int(r.get("metered_turns", 0)) for r in all_records)
    model_matched_turns_total = sum(int(r.get("model_matched_turns", 0)) for r in all_records)
    metered_billed_input_total = sum(int(r.get("metered_billed_input_tokens", 0)) for r in all_records)
    metered_cache_create_total = sum(int(r.get("metered_cache_create_tokens", 0)) for r in all_records)
    metered_cache_read_total = sum(int(r.get("metered_cache_read_tokens", 0)) for r in all_records)
    metered_output_total = sum(int(r.get("metered_output_tokens", 0)) for r in all_records)
    fresh_input_usd_total = round(sum(float(r.pop("_fresh_input_usd", 0.0)) for r in all_records), 2)
    typed_prompt_tokens_total = sum(int(r.pop("_typed_prompt_tokens", 0)) for r in all_records)
    typed_prompt_usd_total = round(sum(float(r.pop("_typed_prompt_usd", 0.0)) for r in all_records), 2)
    tool_and_mcp_tokens_total = max(0, (metered_billed_input_total + metered_cache_create_total) - typed_prompt_tokens_total)
    tool_and_mcp_usd_total = round(max(0.0, fresh_input_usd_total - typed_prompt_usd_total), 2)
    cache_read_usd_total = round(sum(float(r.pop("_cache_read_usd", 0.0)) for r in all_records), 2)
    output_usd_total = round(sum(float(r.pop("_output_usd", 0.0)) for r in all_records), 2)
    mtimes_by_id: Dict[str, float] = {
        str(r.get("conv_id", "")): float(r.pop("_mtime", 0.0)) for r in all_records
    }
    hook_fired_by_id: Dict[str, bool] = {
        str(r.get("conv_id", "")): bool(r.pop("_wbc_hook_fired", False)) for r in all_records
    }
    local_titles_by_id: Dict[str, str] = {
        str(r.get("conv_id", "")): str(r.pop("_local_title", "")) for r in all_records
    }
    model_usage_by_id: Dict[str, Dict[str, Dict[str, Any]]] = {
        str(r.get("conv_id", "")): (r.pop("_model_usage", None) or {}) for r in all_records
    }
    model_opportunities = compute_model_opportunity_insights(
        all_records=all_records,
        model_usage_by_id=model_usage_by_id,
        window_days=days,
    )
    exact_api_spend_total = round(sum(float(r.get("exact_api_spend_usd", 0.0)) for r in all_records), 2)
    hal_wbc_spend_total = round(sum(float(r.get("hal_wbc_spend_usd", 0.0)) for r in all_records), 2)
    hal_saved_total = round(max(0.0, exact_api_spend_total - hal_wbc_spend_total), 2)
    actuals_cut_pct = (
        round((hal_saved_total / exact_api_spend_total) * 100.0, 1)
        if exact_api_spend_total > 0
        else 0.0
    )
    scale_30d_hal = 30.0 / float(max(1, days)) if days > 0 else 1.0
    hal_monthly_saved = round(hal_saved_total * scale_30d_hal, 2)
    hal_annual_saved = round(hal_monthly_saved * 12.0, 2)

    marathon_threshold = 50 if any(int(r.get("planner_turns", 0)) >= 50 for r in all_records) else 25
    marathon_recs = [r for r in all_records if int(r.get("planner_turns", 0)) >= marathon_threshold]
    sprint_recs = [r for r in all_records if int(r.get("planner_turns", 0)) < marathon_threshold]
    marathon_spend_usd = round(sum(float(r.get("exact_api_spend_usd", 0.0)) for r in marathon_recs), 2)
    marathon_saved_usd = round(sum(float(r.get("hal_saved_usd", 0.0)) for r in marathon_recs), 2)
    sprint_spend_usd = round(sum(float(r.get("exact_api_spend_usd", 0.0)) for r in sprint_recs), 2)
    sprint_saved_usd = round(sum(float(r.get("hal_saved_usd", 0.0)) for r in sprint_recs), 2)
    marathon_cut_pct = (
        round((marathon_saved_usd / marathon_spend_usd) * 100.0, 1)
        if marathon_spend_usd > 0
        else 0.0
    )
    sprint_cut_pct = (
        round((sprint_saved_usd / sprint_spend_usd) * 100.0, 1)
        if sprint_spend_usd > 0
        else 0.0
    )
    marathon_share_pct = (
        round((marathon_spend_usd / exact_api_spend_total) * 100.0, 1)
        if exact_api_spend_total > 0
        else 0.0
    )

    tool_spend_by_name: Dict[str, float] = {}
    tool_sess_by_name: Dict[str, int] = {}
    for r in all_records:
        tname = str(r.get("top_offender_tool") or "").strip()
        if tname and tname not in ("none", "tool"):
            tool_spend_by_name[tname] = tool_spend_by_name.get(tname, 0.0) + float(r.get("exact_api_spend_usd", 0.0))
            tool_sess_by_name[tname] = tool_sess_by_name.get(tname, 0) + 1
    top_tools_ranked = [
        {"tool": k, "sessions": tool_sess_by_name[k], "spend_usd": round(v, 2)}
        for k, v in sorted(tool_spend_by_name.items(), key=lambda kv: (kv[1], tool_sess_by_name[kv[0]]), reverse=True)[:4]
    ]

    cost_breakdown = {
        "cache_read_usd": cache_read_usd_total,
        "fresh_input_usd": fresh_input_usd_total,
        "tool_and_mcp_usd": tool_and_mcp_usd_total,
        "typed_prompt_usd": typed_prompt_usd_total,
        "tool_and_mcp_tokens": tool_and_mcp_tokens_total,
        "typed_prompt_tokens": typed_prompt_tokens_total,
        "output_usd": output_usd_total,
        "model_matched_turns": model_matched_turns_total,
        "explicit_price_override": bool(explicit_price_override),
        "marathon_turn_threshold": marathon_threshold,
        "marathon_sessions_count": len(marathon_recs),
        "marathon_spend_usd": marathon_spend_usd,
        "marathon_saved_usd": marathon_saved_usd,
        "marathon_cut_pct": marathon_cut_pct,
        "marathon_spend_share_pct": marathon_share_pct,
        "sprint_sessions_count": len(sprint_recs),
        "sprint_spend_usd": sprint_spend_usd,
        "sprint_saved_usd": sprint_saved_usd,
        "sprint_cut_pct": sprint_cut_pct,
        "top_context_tools": top_tools_ranked,
    }

    # Also compute Sonnet/Flash ($3/$15) exact metered spend so developers see the standard $3/$15 baseline too
    sonnet_exact_total = 0.0
    for r in all_records:
        if r.get("metered_exact"):
            r_cc = int(r.get("metered_cache_create_tokens", 0))
            r_cc_1h = int(r.get("metered_cache_create_1h_tokens", 0))
            r_cc_5m = int(r.get("metered_cache_create_5m_tokens", max(0, r_cc - r_cc_1h)))
            sonnet_exact_total += (
                (int(r.get("metered_billed_input_tokens", 0)) / 1_000_000.0) * 3.0
                + (r_cc_5m / 1_000_000.0) * 3.75
                + (r_cc_1h / 1_000_000.0) * 6.00
                + (int(r.get("metered_cache_read_tokens", 0)) / 1_000_000.0) * 0.30
                + (int(r.get("metered_output_tokens", 0)) / 1_000_000.0) * 15.0
            )
        else:
            sonnet_exact_total += float(r.get("conservative_spend_usd", 0.0)) * (3.0 / max(0.01, in_price))
    sonnet_exact_total = round(sonnet_exact_total, 2)
    eff_cut_ratio = (actuals_cut_pct / 100.0) if actuals_cut_pct > 0 else 0.0
    sonnet_hal_wbc_total = round(sonnet_exact_total * (1.0 - eff_cut_ratio), 2)
    sonnet_hal_saved_total = round(max(0.0, sonnet_exact_total - sonnet_hal_wbc_total), 2)

    projection_assumptions = {
        "ast_slicing_threshold_bytes": 2000,
        "ast_slicing_outline_cap_tokens": 500,
        "ast_slicing_tail_retention_ratio": 0.10,
        "context_flush_step_interval": _resolve_flush_step_interval(),
        "max_fresh_input_reduction_cap": 0.70,
        "max_cache_reread_reduction_cap": 0.75,
        "max_retry_output_reduction_cap": 0.18,
    }
    projected_savings = {
        "methodology": "Bottom-up context trajectory simulation on user transcripts (AST tool-output slicing + periodic context flushes)",
        "assumptions": projection_assumptions,
        "cost_cut_pct": actuals_cut_pct,
        "efficiency_multiple": round(1.0 / max(0.10, 1.0 - eff_cut_ratio), 2) if eff_cut_ratio > 0 else 1.0,
        "sprint_cut_pct": sprint_cut_pct,
        "marathon_cut_pct": marathon_cut_pct,
        "metered_turns": metered_turns_total,
        "metered_billed_input_tokens": metered_billed_input_total,
        "metered_cache_create_tokens": metered_cache_create_total,
        "metered_cache_read_tokens": metered_cache_read_total,
        "metered_output_tokens": metered_output_total,
        "exact_api_spend_usd": exact_api_spend_total,
        "projected_wbc_spend_usd": hal_wbc_spend_total,
        "projected_saved_usd": hal_saved_total,
        "projected_monthly_saved_usd": hal_monthly_saved,
        "projected_annual_saved_usd": hal_annual_saved,
        "hal_wbc_spend_usd": hal_wbc_spend_total,
        "hal_saved_usd": hal_saved_total,
        "hal_monthly_saved_usd": hal_monthly_saved,
        "hal_annual_saved_usd": hal_annual_saved,
        "sonnet_exact_spend_usd": sonnet_exact_total,
        "sonnet_hal_wbc_spend_usd": sonnet_hal_wbc_total,
        "sonnet_hal_saved_usd": sonnet_hal_saved_total,
    }
    hal_benchmark_proof = projected_savings

    def _calc_profile(in_p: float, out_p: float, in_mult: float, bbb_in_mult: float) -> Dict[str, float]:
        f_usd = (
            (total_actual_input_tokens / 1_000_000.0) * (in_p * in_mult)
            + (total_output_tokens / 1_000_000.0) * out_p
        )
        b_usd = (
            (total_bbb_optimized_tokens / 1_000_000.0) * (in_p * bbb_in_mult)
            + (total_bbb_optimized_output_tokens / 1_000_000.0) * out_p
        )
        w_usd = max(0.0, f_usd - b_usd)
        scale_30d = 30.0 / float(max(1, days)) if days > 0 else 1.0
        monthly_saved = w_usd * scale_30d
        annual_seat = monthly_saved * 12.0
        return {
            "input_multiplier": in_mult,
            "effective_input_price_per_m": round(in_p * in_mult, 2),
            "output_price_per_m": out_p,
            "estimated_frontier_spend_usd": round(f_usd, 2),
            "bbb_optimized_spend_usd": round(b_usd, 2),
            "wasted_spend_usd": round(w_usd, 2),
            "net_savings_usd": round(w_usd, 2),
            "monthly_savings_per_seat_usd": round(monthly_saved, 2),
            "projected_annual_per_seat_usd": round(annual_seat, 2),
            "projected_team_10_seats_annual_usd": round(annual_seat * 10.0, 2),
            "projected_team_seats_annual_usd": round(annual_seat * max(1, int(seats)), 2),
            "roi_multiple_vs_pro_tier": round(monthly_saved / 29.0, 1),
            "roi_multiple_vs_team_tier": round(monthly_saved / 79.0, 1),
        }

    def _calc_tier_spend(in_p: float, out_p: float) -> Dict[str, Any]:
        cons = _calc_profile(in_p, out_p, in_mult=0.19, bbb_in_mult=0.19)
        real = _calc_profile(in_p, out_p, in_mult=0.32, bbb_in_mult=0.19)
        ceil = _calc_profile(in_p, out_p, in_mult=1.00, bbb_in_mult=0.32)
        return {
            "input_price_per_m": in_p,
            "output_price_per_m": out_p,
            "estimated_frontier_spend_usd": real["estimated_frontier_spend_usd"],
            "bbb_optimized_spend_usd": real["bbb_optimized_spend_usd"],
            "wasted_spend_usd": real["wasted_spend_usd"],
            "net_savings_usd": real["net_savings_usd"],
            "monthly_savings_per_seat_usd": real["monthly_savings_per_seat_usd"],
            "projected_annual_per_seat_usd": real["projected_annual_per_seat_usd"],
            "projected_team_10_seats_annual_usd": real["projected_team_10_seats_annual_usd"],
            "roi_multiple_vs_pro_tier": real["roi_multiple_vs_pro_tier"],
            "roi_multiple_vs_team_tier": real["roi_multiple_vs_team_tier"],
            "conservative_cached": cons,
            "realistic_blended": real,
            "uncached_ceiling": ceil,
        }

    primary_tier = _calc_tier_spend(in_price, out_price)
    cons_p = primary_tier["conservative_cached"]
    real_p = primary_tier["realistic_blended"]
    ceil_p = primary_tier["uncached_ceiling"]

    estimated_frontier_spend_usd = exact_api_spend_total if metered_turns_total > 0 else real_p["estimated_frontier_spend_usd"]
    bbb_optimized_spend_usd = hal_wbc_spend_total if metered_turns_total > 0 else real_p["bbb_optimized_spend_usd"]
    wasted_spend_usd = hal_saved_total if metered_turns_total > 0 else real_p["wasted_spend_usd"]
    net_savings_usd = wasted_spend_usd
    savings_pct = (
        actuals_cut_pct
        if metered_turns_total > 0
        else (
            round((net_savings_usd / estimated_frontier_spend_usd) * 100.0, 1)
            if estimated_frontier_spend_usd > 0
            else 0.0
        )
    )

    saturated_sessions_cnt = sum(
        1
        for r in all_records
        if int(r.get("truncated_steps", 0)) > 0
        or int(r.get("quadratic_post_saturation_tokens", 0)) > 0
        or int(r.get("planner_turns", 0)) >= marathon_threshold
    )
    if all_records and total_bbb_optimized_tokens > 0:
        quota_stretch_multiple = (
            round(1.0 / max(0.25, 1.0 - eff_cut_ratio), 2)
            if eff_cut_ratio > 0
            else round(
                min(4.0, max(1.0, float(total_actual_input_tokens) / float(max(1, total_bbb_optimized_tokens)))),
                2,
            )
        )
    else:
        quota_stretch_multiple = 0.0
    quota_multiplier_metrics = {
        "quota_stretch_multiple": quota_stretch_multiple,
        "saturated_sessions_count": saturated_sessions_cnt,
        "basis": "projected_from_simulated_context_reduction",
    }

    pricing_tiers = {
        "sonnet_flash_3_15": {
            "label": "Sonnet / Flash ($3 / $15 per 1M)",
            **_calc_tier_spend(3.0, 15.0),
        },
        "flagship_pro_5_25": {
            "label": "Frontier Flagship ($5 / $25 per 1M)",
            **_calc_tier_spend(5.0, 25.0),
        },
        "opus_heavy_15_75": {
            "label": "Claude Opus Heavy ($15 / $75 per 1M)",
            **_calc_tier_spend(15.0, 75.0),
        },
    }

    fallback_custom_root = agy_roots[0].parent if agy_roots else Path.cwd()
    resolved_turn0_ws = (
        workspace_dir
        if workspace_dir is not None
        else (
            Path(os.environ["WBC_AUDIT_WORKSPACE_DIR"])
            if os.environ.get("WBC_AUDIT_WORKSPACE_DIR")
            else (fallback_custom_root if any_custom_override else Path.cwd())
        )
    )
    resolved_turn0_home = (
        home_dir
        if home_dir is not None
        else (
            Path(os.environ["WBC_AUDIT_HOME_DIR"])
            if os.environ.get("WBC_AUDIT_HOME_DIR")
            else (
                fallback_custom_root
                if any_custom_override
                else (candidate_homes[0] if candidate_homes else Path.home())
            )
        )
    )
    turn0_static_overhead = audit_turn0_static_overhead(
        workspace_dir=resolved_turn0_ws,
        home_dir=resolved_turn0_home,
        sessions_count=len(all_records),
        in_price_per_m=in_price,
    )
    t0_per_sess_tok = int(turn0_static_overhead.get("total_turn0_tokens_per_session", 0))
    metered_sessions_cnt = sum(1 for r in all_records if int(r.get("metered_turns", 0)) >= 1)
    tot_fresh_billed_tok = max(1, metered_billed_input_total + metered_cache_create_total)
    max_non_prompt_tok = max(0, (metered_billed_input_total + metered_cache_create_total) - typed_prompt_tokens_total)
    startup_mcp_tokens_total = min(max_non_prompt_tok, t0_per_sess_tok * metered_sessions_cnt)
    startup_mcp_usd_total = round(
        min(
            tool_and_mcp_usd_total,
            fresh_input_usd_total * (float(startup_mcp_tokens_total) / float(tot_fresh_billed_tok)),
        ),
        2,
    )
    tool_outputs_tokens_total = max(0, max_non_prompt_tok - startup_mcp_tokens_total)
    # Reconcile tool_outputs_usd so the 5 display buckets sum to exact_api_spend_total to the exact cent
    reconciled_fresh_usd = max(0.0, exact_api_spend_total - cache_read_usd_total - output_usd_total)
    tool_outputs_usd_total = round(
        max(0.0, reconciled_fresh_usd - startup_mcp_usd_total - typed_prompt_usd_total),
        2,
    )
    cost_breakdown["fresh_input_usd"] = round(reconciled_fresh_usd, 2)
    cost_breakdown["tool_outputs_usd"] = tool_outputs_usd_total
    cost_breakdown["tool_outputs_tokens"] = tool_outputs_tokens_total
    cost_breakdown["startup_and_mcp_usd"] = startup_mcp_usd_total
    cost_breakdown["startup_and_mcp_tokens"] = startup_mcp_tokens_total
    wbc_shield_telemetry = _collect_wbc_shield_actuals(
        home_dir=resolved_turn0_home,
        all_records=all_records,
        mtimes_by_id=mtimes_by_id,
        window_seconds=window_seconds,
        now=now,
        enabled=(not any_custom_override or home_dir is not None or bool(os.environ.get("WBC_IMMUNE_AUDIT_PATH"))),
        hook_fired_by_id=hook_fired_by_id,
    )

    # Do not project additional savings on sessions where WBC hooks already fired
    unshielded_projected_saved = round(
        sum(
            float(r.get("hal_saved_usd", 0.0))
            for r in all_records
            if not hook_fired_by_id.get(str(r.get("conv_id", "")), False)
        ),
        2,
    )
    if wbc_shield_telemetry.get("wbc_active_on_machine") and wbc_shield_telemetry.get("shielded_sessions", 0) > 0:
        eff_saved_total = unshielded_projected_saved
        eff_wbc_spend = round(max(0.0, exact_api_spend_total - eff_saved_total), 2)
    else:
        eff_saved_total = hal_saved_total
        eff_wbc_spend = hal_wbc_spend_total
    eff_monthly_saved = round(eff_saved_total * scale_30d_hal, 2)
    eff_annual_saved = round(eff_monthly_saved * 12.0, 2)
    projected_savings["projected_saved_usd"] = eff_saved_total
    projected_savings["projected_wbc_spend_usd"] = eff_wbc_spend
    projected_savings["projected_monthly_saved_usd"] = eff_monthly_saved
    projected_savings["projected_annual_saved_usd"] = eff_annual_saved
    projected_savings["hal_saved_usd"] = eff_saved_total
    projected_savings["hal_wbc_spend_usd"] = eff_wbc_spend
    projected_savings["hal_monthly_saved_usd"] = eff_monthly_saved
    projected_savings["hal_annual_saved_usd"] = eff_annual_saved

    return {
        "window_days": int(days),
        "input_price_per_m": in_price,
        "output_price_per_m": out_price,
        "seats": int(seats),
        "conversations_scanned": len(all_records),
        "total_steps": total_steps,
        "total_planner_turns": total_planner_turns,
        "total_transcript_bytes": total_transcript_bytes,
        "total_transcript_mb": round(total_transcript_bytes / (1024.0 * 1024.0), 2),
        "raw_unbounded_quadratic_tokens": total_raw_unbounded_tokens,
        "actual_cumulative_input_tokens": total_actual_input_tokens,
        "output_tokens": total_output_tokens,
        "wasted_bloat_tokens": total_wasted_bloat_tokens,
        "quadratic_post_saturation_tokens": total_quadratic_post_sat_tokens,
        "unfiltered_tool_bloat_tokens": total_unfiltered_tool_bloat_tokens,
        "bicameral_local_offload_tokens": total_bicameral_offload_tokens,
        "bbb_optimized_tokens": total_bbb_optimized_tokens,
        "bbb_optimized_output_tokens": total_bbb_optimized_output_tokens,
        "estimated_frontier_spend_usd": estimated_frontier_spend_usd,
        "wasted_spend_usd": eff_saved_total if metered_turns_total > 0 else wasted_spend_usd,
        "bbb_optimized_spend_usd": eff_wbc_spend if metered_turns_total > 0 else bbb_optimized_spend_usd,
        "net_savings_usd": eff_saved_total if metered_turns_total > 0 else net_savings_usd,
        "savings_pct": round((eff_saved_total / exact_api_spend_total) * 100.0, 1) if (metered_turns_total > 0 and exact_api_spend_total > 0) else savings_pct,
        "monthly_savings_per_seat_usd": eff_monthly_saved if metered_turns_total > 0 else real_p["monthly_savings_per_seat_usd"],
        "projected_annual_savings_per_seat_usd": eff_annual_saved if metered_turns_total > 0 else real_p["projected_annual_per_seat_usd"],
        "projected_team_10_seats_annual_usd": round(eff_annual_saved * 10.0, 2) if metered_turns_total > 0 else real_p["projected_team_10_seats_annual_usd"],
        "projected_team_seats_annual_usd": round(eff_annual_saved * max(1, int(seats)), 2) if metered_turns_total > 0 else real_p["projected_team_seats_annual_usd"],
        "roi_multiple_vs_pro_tier": round(eff_monthly_saved / 29.0, 1) if metered_turns_total > 0 else real_p["roi_multiple_vs_pro_tier"],
        "roi_multiple_vs_team_tier": round(eff_monthly_saved / 79.0, 1) if metered_turns_total > 0 else real_p["roi_multiple_vs_team_tier"],
        "measured_spend": {
            "conversations_scanned": len(all_records),
            "total_planner_turns": total_planner_turns,
            "metered_turns": metered_turns_total,
            "model_matched_turns": model_matched_turns_total,
            "metered_billed_input_tokens": metered_billed_input_total,
            "metered_cache_create_tokens": metered_cache_create_total,
            "metered_cache_read_tokens": metered_cache_read_total,
            "metered_output_tokens": metered_output_total,
            "exact_api_spend_usd": exact_api_spend_total,
            "cache_read_usd": cache_read_usd_total,
            "fresh_input_usd": round(reconciled_fresh_usd, 2),
            "tool_outputs_usd": tool_outputs_usd_total,
            "startup_and_mcp_usd": startup_mcp_usd_total,
            "typed_prompt_usd": typed_prompt_usd_total,
            "output_usd": output_usd_total,
        },
        "projected_savings": projected_savings,
        "hal_benchmark_proof": hal_benchmark_proof,
        "cost_breakdown": cost_breakdown,
        "wbc_shield_telemetry": wbc_shield_telemetry,
        "quota_multiplier_metrics": quota_multiplier_metrics,
        "harness_breakdown": harness_breakdown,
        "model_opportunities": model_opportunities,
        "turn0_static_overhead": turn0_static_overhead,
        "top_bloated_conversations": all_records[: max(1, int(top))],
        "_local_titles_by_id": local_titles_by_id,
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _humanize_bloat_trigger(trigger: str) -> str:
    t = str(trigger or "").strip().upper()
    if t == "TRUNCATION_CASCADE":
        return "Context overflow"
    if t == "QUADRATIC_TURN_TAX":
        return "Long session history"
    if t == "UNBOUNDED_TOOL_DUMP":
        return "Large tool outputs"
    return "Active session"


def _wrap_comma_items(prefix: str, items: List[str], max_width: int = 78, cont_indent: str = "    ") -> List[str]:
    if not items:
        return []
    out: List[str] = []
    cur = prefix + items[0]
    for item in items[1:]:
        cand = f"{cur}, {item}"
        if len(cand) <= max_width:
            cur = cand
        else:
            out.append(cur + ",")
            cur = cont_indent + item
    out.append(cur)
    return out


def format_standalone_receipt(
    audit: Dict[str, Any],
    color: Optional[bool] = None,
    show_team: bool = False,
) -> str:
    if color is None:
        color = bool(
            hasattr(sys.stdout, "isatty")
            and sys.stdout.isatty()
            and not os.environ.get("NO_COLOR")
            and os.environ.get("TERM") != "dumb"
        )

    BOLD = "\033[1m" if color else ""
    DIM = "\033[2m" if color else ""
    CYAN = "\033[36m" if color else ""
    GREEN = "\033[32m" if color else ""
    YELLOW = "\033[33m" if color else ""
    RED = "\033[31m" if color else ""
    MAGENTA = "\033[35m" if color else ""
    RESET = "\033[0m" if color else ""

    days = int(audit.get("window_days", 30))
    convs = int(audit.get("conversations_scanned", 0))
    planner_turns = int(audit.get("total_planner_turns", 0))
    in_price = float(audit.get("input_price_per_m", 5.0))
    out_price = float(audit.get("output_price_per_m", round(in_price * 5.0, 2)))
    seats = int(audit.get("seats", 1))

    in_tok = int(audit.get("actual_cumulative_input_tokens", 0))
    out_tok = int(audit.get("output_tokens", 0))
    sav_pct = float(audit.get("savings_pct", 0.0))

    hal = audit.get("hal_benchmark_proof", {})
    cb = audit.get("cost_breakdown", {})
    m_turns = int(hal.get("metered_turns", 0))
    m_matched = int(cb.get("model_matched_turns", 0))
    explicit_override = bool(cb.get("explicit_price_override", False))
    m_in = int(hal.get("metered_billed_input_tokens", 0))
    m_cc = int(hal.get("metered_cache_create_tokens", 0))
    m_cr = int(hal.get("metered_cache_read_tokens", 0))
    m_out = int(hal.get("metered_output_tokens", 0))

    sr = audit.get("savings_range", {})
    cons = sr.get("conservative_cached", {})

    exact_usd = float(hal.get("exact_api_spend_usd", cons.get("estimated_frontier_spend_usd", 0.0)))
    hal_wbc_usd = float(hal.get("hal_wbc_spend_usd", round(exact_usd * 0.322, 2)))
    hal_saved_usd = float(hal.get("hal_saved_usd", round(max(0.0, exact_usd - hal_wbc_usd), 2)))
    hal_mo_saved = float(hal.get("hal_monthly_saved_usd", hal_saved_usd))
    hal_yr_saved = float(hal.get("hal_annual_saved_usd", round(hal_mo_saved * 12.0, 2)))

    qm = audit.get("quota_multiplier_metrics", {})
    q_mult = float(qm.get("quota_stretch_multiple", 0.0))

    def _fmt_tok(n: int) -> str:
        if n >= 1_000_000_000:
            return f"{n / 1_000_000_000:.2f}B"
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n / 1_000:.1f}K"
        return f"{n:,}"

    def _fmt_usd(v: float) -> str:
        return f"${v:,.0f}" if v >= 1000.0 else f"${v:,.2f}"

    def _bar(ratio: float, width: int = 18, fill_color: str = GREEN) -> str:
        clamped = max(0.0, min(1.0, ratio))
        filled = max(1 if clamped > 0 else 0, int(round(clamped * width)))
        empty = max(0, width - filled)
        return f"{fill_color}{'█' * filled}{DIM}{'░' * empty}{RESET}"

    bar_cur = _bar(1.0 if convs > 0 and exact_usd > 0 else 0.0, width=18, fill_color=RED)
    wbc_ratio = (hal_wbc_usd / exact_usd) if exact_usd > 0 else 0.0
    bar_wbc = _bar(wbc_ratio if convs > 0 and exact_usd > 0 else 0.0, width=18, fill_color=GREEN)

    if convs == 0:
        provenance_str = "Scanned locally on your Mac • 0 recent sessions"
    elif m_turns > 0 and m_turns >= int(planner_turns * 0.85):
        tot_tok = m_in + m_cc + m_cr + m_out
        provenance_str = (
            f"Scanned locally • {convs:,} sessions • {planner_turns:,} turns • "
            f"{_fmt_tok(tot_tok)} tok (exact API counts)"
        )
    elif m_turns > 0:
        provenance_str = (
            f"Scanned locally • {convs:,} sessions • {planner_turns:,} turns "
            f"({m_turns:,} exact API counts)"
        )
    else:
        provenance_str = (
            f"Scanned locally • {convs:,} sessions • {planner_turns:,} turns • "
            f"estimated from local logs"
        )

    if explicit_override or m_matched == 0:
        rate_note = f"Valued at ${in_price:g}/${out_price:g} per 1M tok API list rates • per-session actuals"
    elif m_matched >= m_turns and m_turns > 0:
        rate_note = "Valued at logged model API rates ($0.50-$5/M in) • per-session actuals"
    else:
        rate_note = (
            f"Logged model rates ($0.50-$5/M) + ${in_price:g}/${out_price:g} fallback • per-session actuals"
        )

    divider = f"{DIM}{'─' * 78}{RESET}"
    cut_pct_f = (hal_saved_usd / exact_usd * 100.0) if exact_usd > 0 else 0.0
    cut_pct = int(round(cut_pct_f)) if hal_saved_usd > 0 else 0
    cut_badge = f"(-{cut_pct}% est.)" if cut_pct > 0 else ""

    eff_mult = round(1.0 / max(0.25, 1.0 - (cut_pct_f / 100.0)), 1) if cut_pct_f > 0 else 1.0
    if q_mult > 0:
        q_lo = max(1.1, min(q_mult, eff_mult))
        q_hi = max(q_mult, eff_mult)
    else:
        q_lo = q_hi = max(1.1, eff_mult)
    mult_s = f"~{eff_mult:.1f}x" if abs(q_hi - q_lo) < 0.15 else f"{q_lo:.1f}x-{q_hi:.1f}x"
    quota_str = (
        f"{BOLD}{CYAN}{mult_s}{RESET} longer Pro/Max quota before rate limits"
        if convs > 0 and cut_pct > 0
        else f"{DIM}Longer Pro/Max quota before rate-limit lockouts{RESET}"
    )

    wbc_st = audit.get("wbc_shield_telemetry", {}) or {}
    wbc_active = bool(wbc_st.get("wbc_active_on_machine", False))
    wired_set = set(wbc_st.get("wired_harnesses", []) or [])
    sh_n = int(wbc_st.get("shielded_sessions", 0))
    un_n = int(wbc_st.get("pre_or_unshielded_sessions", convs))
    sh_usd = float(wbc_st.get("shielded_spend_usd", 0.0))
    un_usd = float(wbc_st.get("pre_or_unshielded_spend_usd", exact_usd))
    sh_avg = float(wbc_st.get("shielded_avg_session_usd", 0.0))
    un_avg = float(wbc_st.get("pre_or_unshielded_avg_session_usd", 0.0))
    drop_pct = float(wbc_st.get("per_session_cost_drop_pct", 0.0))
    unwired_harnesses = wbc_st.get("unwired_active_harnesses", []) or []
    unwired_save = float(wbc_st.get("unwired_projected_save_usd", 0.0))
    meas_slices = int(wbc_st.get("measured_bounded_slices", 0))
    meas_dtok = int(wbc_st.get("measured_direct_tokens_saved", 0))

    lines: List[str] = [
        "",
        f"{BOLD}{CYAN}WHOLE BODY COMPUTE - {days}-Day Agent Audit{RESET}",
        f"{DIM}{provenance_str}{RESET}",
        divider,
        "",
    ]

    if wbc_active and convs > 0 and sh_n > 0:
        wired_lbl = ", ".join(h.replace("_", "-") for h in sorted(wired_set))[:32] or "local hooks"
        lines.append(
            f"  {BOLD}Total {days}d API-Equiv{RESET}       {BOLD}${exact_usd:,.2f}{RESET}  {GREEN}(WBC active: {wired_lbl}){RESET}"
        )
        if un_n > 0:
            bar_sh = _bar(sh_n / max(1, convs), width=14, fill_color=GREEN)
            bar_un = _bar(un_n / max(1, convs), width=14, fill_color=YELLOW)
            sh_cnt = f"{sh_n:,}/{convs:,} sess"
            un_cnt = f"{un_n:,}/{convs:,} sess"
            cnt_w = max(len(sh_cnt), len(un_cnt))
            lines.append(
                f"  {GREEN}WBC-Shielded Sessions{RESET}     {bar_sh}  {BOLD}{GREEN}{sh_cnt:>{cnt_w}}{RESET}  {DIM}(${sh_usd:,.0f}, ${sh_avg:,.2f}/s){RESET}"
            )
            lines.append(
                f"  {YELLOW}Unshielded / Bypassed{RESET}     {bar_un}  {BOLD}{un_cnt:>{cnt_w}}{RESET}  {DIM}(${un_usd:,.0f}, ${un_avg:,.2f}/s){RESET}"
            )
        else:
            lines.append(
                f"  {GREEN}WBC-Shielded Sessions{RESET}     {BOLD}{GREEN}{sh_n:,} / {convs:,} sessions{RESET}  (${sh_avg:,.2f}/sess avg)"
            )
        lines.append("")
        if meas_dtok > 0 or meas_slices > 0:
            lines.append(
                f"  {BOLD}Measured Hook Shield{RESET}      {GREEN}{meas_slices:,} bounded slices{RESET} • {_fmt_tok(meas_dtok)} direct tok blocked"
            )
        if unwired_save > 0 and unwired_harnesses:
            uw_lbl = ", ".join(h.replace("_", "-") for h in unwired_harnesses[:2])
            lines.append(
                f"  {BOLD}Remaining Opportunity{RESET}     {BOLD}{GREEN}Save ${unwired_save:,.2f} / {days}d{RESET} by wiring {uw_lbl}"
            )
        elif un_n > 0:
            lines.append(
                f"  {BOLD}Unshielded Spend{RESET}          {DIM}${un_usd:,.2f} across {un_n:,} sessions without hook telemetry{RESET}"
            )
        else:
            lines.append(
                f"  {BOLD}Unwired Opportunity{RESET}       {DIM}$0.00 uncaptured (all active harnesses protected){RESET}"
            )
    else:
        cur_usd_s = f"${exact_usd:,.2f}"
        wbc_usd_s = f"${hal_wbc_usd:,.2f}"
        usd_w = max(len(cur_usd_s), len(wbc_usd_s))
        lines.extend(
            [
                f"  {RED}Current API-Equiv Spend{RESET}   {bar_cur}  {BOLD}{cur_usd_s:>{usd_w}}{RESET} / {days}d",
                f"  {GREEN}With Whole Body Compute{RESET}   {bar_wbc}  {BOLD}{GREEN}{wbc_usd_s:>{usd_w}}{RESET} / {days}d  {DIM}{cut_badge}{RESET}".rstrip(),
                "",
                f"  {BOLD}Projected Savings{RESET}         {BOLD}{GREEN}${hal_saved_usd:,.2f} / {days}d{RESET}  ({GREEN}${hal_yr_saved:,.0f} / yr{RESET})",
                f"  {BOLD}Subscription Impact{RESET}       {quota_str}",
            ]
        )

    if show_team or seats > 1:
        base_yr = (unwired_save * (30.0 / max(1, days)) * 12.0) if (wbc_active and unwired_save > 0) else hal_yr_saved
        team_hal_yr = base_yr * max(1, seats)
        lines.append(
            f"  {BOLD}{MAGENTA}Team Projection ({seats} Seats){RESET} {BOLD}{MAGENTA}${team_hal_yr:,.0f} / yr{RESET} projected savings"
        )

    flush_steps = _resolve_flush_step_interval()
    projection_note = (
        f"Projection: AST slicing (>2KB tool out -> 500-tok outline) + flush @{flush_steps}t"
    )
    lines.append(f"  {DIM}{rate_note}{RESET}")
    lines.append(f"  {DIM}{projection_note}{RESET}")

    hb = audit.get("harness_breakdown", {})
    active_harnesses: List[Tuple[str, Dict[str, Any]]] = []
    empty_labels: List[str] = []
    for h_key in (
        "antigravity",
        "gemini_cli",
        "claude_code",
        "cursor",
        "goose",
        "codex",
        "cline_roo",
        "windsurf",
        "aider",
        "continue_dev",
        "zed",
    ):
        if h_key not in hb:
            continue
        h_d = hb[h_key]
        if int(h_d.get("threads", 0)) > 0:
            active_harnesses.append((h_key, h_d))
        else:
            short_lbl = (
                str(h_d.get("label", h_key))
                .replace(" Agent / Composer", "")
                .replace(" CLI", "")
                .replace(" Cascade", "")
                .replace("Google ", "")
                .replace("Anthropic ", "")
                .replace("OpenAI ", "")
                .replace("Block ", "")
            )
            empty_labels.append(short_lbl)

    lines.append("")
    lines.append(f"{BOLD}BY AGENT HARNESS{RESET}")
    if active_harnesses:
        right_hdr = "Shield / Save" if wbc_active else "Projected Save"
        lines.append(
            f"  {DIM}{'Harness':<24} {'Sessions':>8} {'Turns':>9}  {'Current':>11}  {right_hdr:>16}{RESET}"
        )
        for _hk, h_d in active_harnesses:
            lbl = str(h_d.get("label", _hk))[:24]
            t_c = int(h_d.get("threads", 0))
            p_c = int(h_d.get("planner_turns", 0))
            ex_u = float(h_d.get("exact_api_spend_usd", h_d.get("conservative_savable_usd", 0.0)))
            hs_u = float(h_d.get("hal_saved_usd", 0.0))
            ex_str = f"${ex_u:,.2f}"
            if wbc_active and _hk in wired_set:
                hs_plain = "WBC Active"
                hs_col = f"{GREEN}{hs_plain:>16}{RESET}"
            else:
                hs_plain = f"Save ${hs_u:,.2f}"
                hs_col = f"{GREEN}{hs_plain:>16}{RESET}"
            lines.append(f"  {lbl:<24} {t_c:>8,} {p_c:>9,}  {ex_str:>11}  {hs_col}")
    else:
        lines.append(f"  {DIM}No active agent sessions found in the last {days} days.{RESET}")

    if empty_labels:
        for wrapped_ln in _wrap_comma_items("  Checked (0 recent): ", empty_labels, max_width=78, cont_indent="    "):
            lines.append(f"{DIM}{wrapped_ln}{RESET}")

    t0_meta = audit.get("turn0_static_overhead", {})
    t0_tot = int(t0_meta.get("total_turn0_tokens_per_session", 0))
    t0_ifiles = int(t0_meta.get("instruction_files_count", 0))
    t0_mcp = int(t0_meta.get("mcp_turn0_tokens", 0))
    t0_msrv = int(t0_meta.get("mcp_servers_count", 0))
    mcp_servers = t0_meta.get("mcp_servers", []) or []

    cache_usd = float(cb.get("cache_read_usd", 0.0))
    fresh_usd = float(cb.get("fresh_input_usd", 0.0))
    typed_usd = float(cb.get("typed_prompt_usd", 0.0))
    typed_tok = int(cb.get("typed_prompt_tokens", 0))
    tool_mcp_usd = float(cb.get("tool_and_mcp_usd", max(0.0, fresh_usd - typed_usd)))
    tool_mcp_tok = int(cb.get("tool_and_mcp_tokens", max(0, (m_in + m_cc) - typed_tok)))
    startup_usd = float(cb.get("startup_and_mcp_usd", 0.0))
    startup_tok = int(cb.get("startup_and_mcp_tokens", 0))
    tool_out_usd = float(cb.get("tool_outputs_usd", max(0.0, tool_mcp_usd - startup_usd)))
    tool_out_tok = int(cb.get("tool_outputs_tokens", max(0, tool_mcp_tok - startup_tok)))
    out_usd = float(cb.get("output_usd", 0.0))
    bucket_sum = cache_usd + fresh_usd + out_usd
    marathon_cnt = int(cb.get("marathon_sessions_count", 0))
    marathon_thr = int(cb.get("marathon_turn_threshold", 50))
    marathon_pct = float(cb.get("marathon_spend_share_pct", 0.0))
    top_tools = cb.get("top_context_tools", []) or []

    if convs > 0 or t0_tot > 0 or t0_msrv > 0:
        lines.append("")
        where_hdr = f"WHERE YOUR {_fmt_usd(exact_usd)} WENT" if exact_usd > 0 else "WHERE YOUR TOKENS GO"
        if m_turns > 0 and bucket_sum > 0:
            c_pct = int(round((cache_usd / bucket_sum) * 100.0))
            f_pct = int(round((fresh_usd / bucket_sum) * 100.0))
            o_pct = max(0, 100 - c_pct - f_pct)
            read_tax_pct = min(99, max(1, c_pct + f_pct))
            think_pct = max(1, 100 - read_tax_pct)
            hdr_line = f"{BOLD}{where_hdr}{RESET}  {DIM}({read_tax_pct}% reading tax · {think_pct}% model thinking){RESET}"
            if len(f"{where_hdr}  ({read_tax_pct}% reading tax · {think_pct}% model thinking)") <= 79:
                lines.append(hdr_line)
            else:
                lines.append(f"{BOLD}{where_hdr}{RESET}")

            tot_m_tok = max(1, m_in + m_cc + m_cr + m_out)
            cr_vol_pct = int(round((m_cr / float(tot_m_tok)) * 100.0))
            out_vol_pct = (m_out / float(tot_m_tok)) * 100.0
            out_vol_s = "<1%" if out_vol_pct < 1.0 else f"{out_vol_pct:.0f}%"

            def _fmt_share(u_val: float) -> str:
                raw_p = (u_val / bucket_sum) * 100.0 if bucket_sum > 0 else 0.0
                if u_val > 0 and raw_p < 1.0:
                    return "<1%"
                return f"{int(round(raw_p))}%"

            s_pct_int = int(round((startup_usd / bucket_sum) * 100.0))
            p_pct_int = int(round((typed_usd / bucket_sum) * 100.0))
            t_pct_int = max(0, f_pct - s_pct_int - p_pct_int)
            t_pct_s = "<1%" if (tool_out_usd > 0 and t_pct_int == 0) else f"{t_pct_int}%"
            s_pct_s = _fmt_share(startup_usd)
            p_pct_s = _fmt_share(typed_usd)
            rules_lbl = (
                f"{t0_ifiles} rules file"
                if t0_ifiles == 1
                else (f"{t0_ifiles} rules files" if t0_ifiles > 1 else "rules files")
            )
            startup_desc = (
                f"{_fmt_tok(startup_tok)} tok (MCP & {rules_lbl}; {_fmt_tok(t0_mcp)} MCP/s)"
                if t0_mcp > 0
                else f"{_fmt_tok(startup_tok)} tok ({rules_lbl}; 0 MCP)"
            )

            lines.append(
                f"  • {BOLD}History re-reads:{RESET}  {_fmt_usd(cache_usd)} ({c_pct}%) - "
                f"{_fmt_tok(m_cr)} cached tok ({cr_vol_pct}% of tok)"
            )
            lines.append(
                f"  • {BOLD}Tool & new ctx:{RESET}    {_fmt_usd(tool_out_usd)} ({t_pct_s}) - "
                f"{_fmt_tok(tool_out_tok)} tool out + cache write tok"
            )
            lines.append(
                f"  • {BOLD}Startup & MCPs:{RESET}    {_fmt_usd(startup_usd)} ({s_pct_s}) - "
                f"{startup_desc}"
            )
            lines.append(
                f"  • {BOLD}Your prompts:{RESET}      {_fmt_usd(typed_usd)} ({p_pct_s}) - "
                f"{_fmt_tok(typed_tok)} tok you actually typed"
            )
            lines.append(
                f"  • {BOLD}Model responses:{RESET}   {_fmt_usd(out_usd)} ({o_pct}%) - "
                f"{_fmt_tok(m_out)} output tok ({out_vol_s} of tok)"
            )
        else:
            lines.append(f"{BOLD}{where_hdr}{RESET}")
            if convs > 0 and in_tok > 0:
                lines.append(
                    f"  • {BOLD}Context re-reads:{RESET}  {_fmt_tok(in_tok)} input tok across turns "
                    f"({sav_pct:.0f}% avoidable bloat)"
                )
                lines.append(
                    f"  • {BOLD}Model responses:{RESET}   {_fmt_tok(out_tok)} output tokens"
                )

        if convs >= 3 and marathon_cnt > 0 and marathon_pct >= 15.0:
            lines.append(
                f"  • {BOLD}Marathon threads:{RESET}  {marathon_cnt:,} sessions ({marathon_thr}+ turns) "
                f"drove {marathon_pct:.0f}% of total spend"
            )

        if top_tools:
            has_tool_usd = any(float(t.get("spend_usd", 0.0)) >= 0.50 for t in top_tools[:3])
            if has_tool_usd:
                parts_t: List[str] = []
                for t in top_tools[:3]:
                    tn = str(t["tool"])[:14]
                    tu = float(t.get("spend_usd", 0.0))
                    cand_p = f"{tn} ({_fmt_usd(tu)})" if tu >= 0.05 else tn
                    cand_join = ", ".join(parts_t + [cand_p])
                    if len("  • Top tool drivers:  " + cand_join) <= 79:
                        parts_t.append(cand_p)
                if parts_t:
                    lines.append(f"  • {BOLD}Top tool drivers:{RESET}  {', '.join(parts_t)}")
                else:
                    tool_names = ", ".join(str(t["tool"])[:18] for t in top_tools[:3])
                    lines.append(f"  • {BOLD}Top tool drivers:{RESET}  {tool_names}")
            else:
                tool_names = ", ".join(str(t["tool"])[:18] for t in top_tools[:3])
                lines.append(f"  • {BOLD}Top tool drivers:{RESET}  {tool_names}")

        if t0_tot > 0 or t0_msrv > 0 or t0_ifiles > 0:
            if t0_mcp > 0:
                lines.append(
                    f"  • {BOLD}Startup overhead:{RESET}  {YELLOW if t0_mcp >= 3000 else ''}{t0_tot:,} tok/sess{RESET} "
                    f"before prompt 1 ({GREEN}{_fmt_tok(t0_mcp)} MCP -> 0 w/ body{RESET})"
                )
            else:
                lines.append(
                    f"  • {BOLD}Startup overhead:{RESET}  {t0_tot:,} tok/sess before prompt 1 "
                    f"({DIM}rules files; 0 MCP{RESET})"
                )
            named_mcps = [
                f"{str(s.get('name', ''))[:18]} ({_fmt_tok(int(s.get('tokens', 0)))})"
                for s in mcp_servers
                if int(s.get("tokens", 0)) > 0
            ][:3]
            if named_mcps:
                mcp_line = "                       MCPs: " + ", ".join(named_mcps)
                lines.append(f"{DIM}{mcp_line[:78]}{RESET}")

    mo = audit.get("model_opportunities", {}) or {}
    mo_models = mo.get("models", []) or []
    if mo_models and (len(mo_models) >= 2 or float(mo.get("total_opportunity_usd", 0.0)) >= 0.05):
        lines.append("")
        lines.append(f"{BOLD}MODEL MIX & ROUTING OPPORTUNITIES (READ-ONLY){RESET}")
        lines.append(
            f"  {DIM}{'Model':<26} {'Sessions':>8} {'Turns':>8}  {'Current':>10}  {'Avg/Sess':>10}  {'Out%':>6}{RESET}"
        )
        for m_row in mo_models[:6]:
            m_lbl = str(m_row.get("label", ""))[:26]
            m_sess = int(m_row.get("sessions", 0))
            m_trns = int(m_row.get("turns", 0))
            m_cur = f"${float(m_row.get('spend_usd', 0.0)):,.2f}"
            m_avg = f"${float(m_row.get('avg_session_usd', 0.0)):,.2f}"
            m_out_p = f"{float(m_row.get('output_share_pct', 0.0)):.0f}%"
            lines.append(
                f"  {m_lbl:<26} {m_sess:>8,} {m_trns:>8,}  {m_cur:>10}  {m_avg:>10}  {m_out_p:>6}"
            )
        if len(mo_models) > 6:
            rem_models = mo_models[6:]
            rem_u = sum(float(x.get("spend_usd", 0.0)) for x in rem_models)
            rem_t = sum(int(x.get("turns", 0)) for x in rem_models)
            rem_line = f"  + {len(rem_models)} more logged models ({rem_t:,} turns, ${rem_u:,.2f})"
            lines.append(f"{DIM}{rem_line[:78]}{RESET}")

        rd = mo.get("routine_downshift", {}) or {}
        md = mo.get("marathon_reread_drag", {}) or {}
        ct = mo.get("thinking_cot_tax", {}) or {}
        tot_opp_u = float(mo.get("total_opportunity_usd", 0.0))
        yr_opp_u = float(mo.get("projected_annual_opportunity_usd", 0.0))

        if float(rd.get("savable_usd", 0.0)) >= 0.05 and int(rd.get("turns", 0)) > 0:
            rd_sav = f"~{_fmt_usd(float(rd.get('savable_usd', 0.0)))}"
            rd_sess = int(rd.get("sessions", 0))
            rd_trns = int(rd.get("turns", 0))
            tgt_short = "Haiku / Hyd-Low" if "Haiku" in str(rd.get("recommended_target", "")) else "Hyd / Neon-Low"
            lines.append(
                f"  • {BOLD}Routine downshift:{RESET} {GREEN}{rd_sav}{RESET} - {rd_sess:,} sprints ({rd_trns:,}t) -> {tgt_short}"
            )
        if float(md.get("savable_usd", 0.0)) >= 0.05 and int(md.get("sessions", 0)) > 0:
            md_sav = f"~{_fmt_usd(float(md.get('savable_usd', 0.0)))}"
            md_sess = int(md.get("sessions", 0))
            md_reread = _fmt_usd(float(md.get("current_reread_usd", 0.0)))
            lines.append(
                f"  • {BOLD}Marathon re-reads:{RESET} {GREEN}{md_sav}{RESET} - {md_sess:,} marathons ({md_reread} re-read; 50+t)"
            )
        if float(ct.get("savable_usd", 0.0)) >= 0.05 and int(ct.get("turns", 0)) > 0:
            ct_sav = f"~{_fmt_usd(float(ct.get('savable_usd', 0.0)))}"
            ct_tok = _fmt_tok(int(ct.get("output_tokens", 0)))
            ct_trns = int(ct.get("turns", 0))
            lines.append(
                f"  • {BOLD}Thinking/CoT tax:{RESET}  {GREEN}{ct_sav}{RESET} - {ct_tok} CoT/out across {ct_trns:,} heavy turns"
            )
        if tot_opp_u >= 1.0:
            lines.append(
                f"  • {BOLD}Model-fit upside:{RESET}  {BOLD}{GREEN}~{_fmt_usd(tot_opp_u)} / {days}d{RESET} "
                f"({GREEN}~${yr_opp_u:,.0f}/yr{RESET} • read-only)"
            )

    top_convs = audit.get("top_bloated_conversations", [])
    titles_map = audit.get("_local_titles_by_id", {}) or {}
    if top_convs:
        lines.append("")
        lines.append(f"  {BOLD}Costliest Sessions{RESET}")
        harness_short_map = {
            "antigravity": "antigravity",
            "claude_code": "claude-code",
            "gemini_cli": "gemini-cli",
            "cursor": "cursor",
            "goose": "goose",
            "codex": "codex",
            "cline_roo": "cline/roo",
            "windsurf": "windsurf",
            "aider": "aider",
            "continue_dev": "continue",
            "zed": "zed",
        }
        for idx, c in enumerate(top_convs, start=1):
            c_cur = float(c.get("exact_api_spend_usd", c.get("conservative_spend_usd", 0.0)))
            h_raw = str(c.get("harness", "agent"))
            h_short = harness_short_map.get(h_raw, h_raw[:11])
            turns_n = int(c.get("planner_turns", 0))
            t_str = f"{turns_n:,}t"
            cid_full = str(c.get("conv_id", ""))
            raw_title = str(titles_map.get(cid_full) or _clean_session_title(str(c.get("first_prompt", "")), max_len=26))
            if raw_title:
                clipped = raw_title[:26] if len(raw_title) <= 26 else (raw_title[:23].rstrip() + "...")
                desc_plain = f'"{clipped}"'
                desc_str = f"{desc_plain:<28}"
            else:
                reason = _humanize_bloat_trigger(str(c.get("bloat_trigger", "")))[:28]
                desc_str = f"{YELLOW}{reason:<28}{RESET}"
            spend_s = f"${c_cur:,.2f}"
            sid_s = f"[{str(c.get('short_id', ''))[:8]}]"
            idx_s = f"  {idx}." if idx < 10 else f" {idx}."
            lines.append(
                f"{idx_s} {BOLD}{spend_s:>9}{RESET}  {h_short:<11} {t_str:>6}  {desc_str}  {CYAN}{sid_s}{RESET}"
            )

    lines.extend(
        [
            "",
            divider,
            f"{BOLD}{GREEN}Start Saving Now - Copy & Paste to Install (100% Free Open-Core):{RESET}",
            f"  • {BOLD}Install:{RESET}   {CYAN}curl -fsSL https://wholebodycompute.com/begin/audit | sh{RESET}",
            f"  • {BOLD}Alt:{RESET}       {DIM}uv tool install whole-body-compute && body begin{RESET}",
            f"  • {BOLD}Share:{RESET}     {DIM}curl -sSL https://wholebodycompute.com/audit.py | python3{RESET}",
            f"  • {BOLD}Privacy:{RESET}   {DIM}Read-only scan; 1 anon ping (--offline or DO_NOT_TRACK=1){RESET}",
            "",
        ]
    )
    return "\n".join(lines)


def _detect_harness_env_flags() -> Dict[str, Any]:
    """Return strictly boolean, zero-PII directory presence flags to diagnose 0-session audits."""
    try:
        homes = _iter_candidate_user_homes()
        appdata = os.environ.get("APPDATA", "").strip()
        is_wsl = bool(os.environ.get("WSL_DISTRO_NAME"))
        if not is_wsl and Path("/proc/version").is_file():
            try:
                is_wsl = "microsoft" in Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
            except Exception:
                pass
        return {
            "has_agy_dir": any((h / ".gemini" / "antigravity" / "brain").is_dir() for h in homes),
            "has_gemini_cli_dir": any((h / ".gemini" / "tmp").is_dir() for h in homes),
            "has_claude_dir": any((h / ".claude" / "projects").is_dir() or (h / ".claude").is_dir() for h in homes),
            "has_cursor_dir": bool(
                (appdata and (Path(appdata) / "Cursor").is_dir())
                or any(
                    (h / ".cursor").is_dir()
                    or (h / "Library" / "Application Support" / "Cursor").is_dir()
                    or (h / "AppData" / "Roaming" / "Cursor").is_dir()
                    or (h / ".config" / "Cursor").is_dir()
                    or (h / ".cursor-server").is_dir()
                    for h in homes
                )
            ),
            "has_codex_dir": any((h / ".codex").is_dir() for h in homes),
            "has_windsurf_dir": any(
                (h / ".codeium" / "windsurf").is_dir()
                or (h / "AppData" / "Roaming" / "Windsurf").is_dir()
                or (h / "Library" / "Application Support" / "Windsurf").is_dir()
                for h in homes
            ),
            "is_wsl": is_wsl,
            "has_wsl_win_users": Path("/mnt/c/Users").is_dir() or bool(os.environ.get("WBC_WSL_USERS_ROOT")),
        }
    except Exception:
        return {}


def _emit_audit_telemetry(audit: Dict[str, Any], no_telemetry: bool = False) -> None:
    """
    Emit an anonymous, zero-PII `wbc_audit_completed` funnel ping to PostHog (`730830`)
    and increment the local `~/.whole_body/telemetry/usage_stats.json` counter.
    Strictly honors DO_NOT_TRACK=1, CI=true, BODY_TELEMETRY=0/off/false/airgapped,
    and PYTEST_CURRENT_TEST (unless WBC_TEST_TELEMETRY_PING=1).
    """
    if no_telemetry:
        return
    if "PYTEST_CURRENT_TEST" in os.environ and not os.environ.get("WBC_TEST_TELEMETRY_PING"):
        return
    # Check DNT / CI / opt-out BEFORE writing any local stats or consent files
    dnt = os.environ.get("DO_NOT_TRACK", "").strip().lower()
    if dnt in ("1", "true", "yes", "on"):
        return
    if os.environ.get("CI", "").strip().lower() in ("1", "true", "yes"):
        return
    bt = os.environ.get("BODY_TELEMETRY", "").strip().lower()
    if bt in ("0", "off", "false", "no", "none", "local", "airgapped", "tier0"):
        return

    # 1. Local usage_stats.json increment (best-effort)
    cohort_id = ""
    try:
        wbc_dir = Path(
            os.environ.get("WBC_DATA_DIR")
            or (
                Path.home() / ".whole_body_compute"
                if (Path.home() / ".whole_body_compute").exists()
                else (
                    Path.home() / ".voicefi"
                    if (Path.home() / ".voicefi").exists()
                    else Path.home() / ".whole_body_compute"
                )
            )
        ).expanduser()
        stats_p = Path(
            os.environ.get("WBC_USAGE_STATS_PATH") or (wbc_dir / "usage_stats.json")
        ).expanduser()
        consent_p = Path(
            os.environ.get("WBC_TELEMETRY_CONFIG_PATH")
            or (wbc_dir / "telemetry_consent.json")
        ).expanduser()
        if consent_p.is_file():
            c_data = json.loads(consent_p.read_text(encoding="utf-8"))
            cohort_id = str(c_data.get("cohort_id") or "")
            if not cohort_id and c_data.get("random_salt"):
                ym = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m")
                cohort_id = hashlib.sha256(f"{c_data['random_salt']}:{ym}".encode("utf-8")).hexdigest()[:16]
        if not cohort_id:
            salt = secrets.token_hex(16)
            ym = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m")
            cohort_id = hashlib.sha256(f"{salt}:{ym}".encode("utf-8")).hexdigest()[:16]
            try:
                if consent_p.parent.is_dir() or os.environ.get("WBC_TELEMETRY_CONFIG_PATH") or os.environ.get("WBC_DATA_DIR"):
                    consent_p.parent.mkdir(parents=True, exist_ok=True)
                    consent_p.write_text(
                        json.dumps({"random_salt": salt, "cohort_id": cohort_id}, indent=2),
                        encoding="utf-8",
                    )
            except Exception:
                pass

        now = round(time.time(), 3)
        if stats_p.parent.is_dir() or os.environ.get("WBC_USAGE_STATS_PATH"):
            stats_p.parent.mkdir(parents=True, exist_ok=True)
            stats: Dict[str, Any] = {
                "version": 1,
                "updated_at": now,
                "total_events": 0,
                "commands": {},
                "events": {},
                "recent_events": [],
            }
            if stats_p.is_file():
                try:
                    loaded = json.loads(stats_p.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        stats.update(loaded)
                except Exception:
                    pass
            stats["updated_at"] = now
            stats["total_events"] = int(stats.get("total_events", 0)) + 1
            ev_map = stats.setdefault("events", {})
            ev_entry = ev_map.setdefault(
                "wbc_audit_completed",
                {"count": 0, "first_ts": now, "last_ts": now, "last_properties": {}},
            )
            if isinstance(ev_entry, int):
                ev_entry = {"count": ev_entry, "first_ts": now, "last_ts": now, "last_properties": {}}
                ev_map["wbc_audit_completed"] = ev_entry
            ev_entry["count"] = int(ev_entry.get("count", 0)) + 1
            ev_entry["last_ts"] = now
            ev_entry["last_properties"] = {
                "mode": "standalone_audit_py",
                "threads_scanned": int(audit.get("conversations_scanned", 0)),
                "wasted_spend_usd": float(audit.get("wasted_spend_usd", 0.0)),
            }
            stats_p.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    except Exception:
        if not cohort_id:
            cohort_id = "anon_audit"

    # 2. Check DNT / CI / opt-out before network call
    dnt = os.environ.get("DO_NOT_TRACK", "").strip().lower()
    if dnt in ("1", "true", "yes", "on"):
        return
    if os.environ.get("CI", "").strip().lower() in ("1", "true", "yes"):
        return
    bt = os.environ.get("BODY_TELEMETRY", "").strip().lower()
    if bt in ("0", "off", "false", "no", "none", "local", "airgapped", "tier0"):
        return

    api_key = (
        os.environ.get("WBC_POSTHOG_API_KEY", "").strip()
        or os.environ.get("POSTHOG_API_KEY", "").strip()
        or DEFAULT_POSTHOG_API_KEY
    )
    host = (
        os.environ.get("WBC_POSTHOG_HOST", "").strip()
        or os.environ.get("POSTHOG_HOST", "").strip()
        or DEFAULT_POSTHOG_HOST
    ).rstrip("/")
    if not api_key:
        return

    hb = audit.get("harness_breakdown", {}) or {}
    active_harnesses = [k for k, v in hb.items() if isinstance(v, dict) and int(v.get("threads", 0)) > 0]
    convs_scanned = int(audit.get("conversations_scanned", 0))
    raw_acq = (os.environ.get("WBC_AUDIT_SOURCE") or WBC_EDGE_AUDIT_SOURCE or "direct").strip().lower()
    acq_source = re.sub(r"[^a-z0-9_.-]", "", raw_acq)[:40] or "direct"
    is_internal = (
        os.environ.get("WBC_INTERNAL_DEV") == "1"
        or Path.cwd().name == "Whole Body Compute"
        or acq_source == "wbc_canary"
    )
    payload = {
        "api_key": api_key,
        "event": "wbc_audit_completed",
        "distinct_id": cohort_id,
        "properties": {
            "cohort_id": cohort_id,
            "mode": "standalone_audit_py",
            "acquisition_source": acq_source,
            "utm_source": acq_source if acq_source != "direct" else None,
            "is_qualified_audit": convs_scanned > 0,
            "window_days": int(audit.get("window_days", 30)),
            "conversations_scanned": convs_scanned,
            "total_steps": int(audit.get("total_steps", 0)),
            "active_harnesses": ",".join(active_harnesses),
            "active_harnesses_count": len(active_harnesses),
            "actual_cumulative_input_tokens": int(audit.get("actual_cumulative_input_tokens", 0)),
            "wasted_bloat_tokens": int(audit.get("wasted_bloat_tokens", 0)),
            "wasted_spend_usd": float(audit.get("wasted_spend_usd", 0.0)),
            "savings_pct": float(audit.get("savings_pct", 0.0)),
            "os_platform": sys.platform,
            "$lib": "whole-body-compute-standalone-audit",
            "$ip": None,
            "$geoip_disable": True,
            **_detect_harness_env_flags(),
            **({"is_internal_dev": True} if is_internal else {}),
        },
    }
    try:
        req = urllib.request.Request(
            f"{host}/capture/",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "WholeBodyCompute-Audit/1.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=1.2):
            pass
    except Exception:
        pass


def main(argv: Optional[List[str]] = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    explicit_seats = any(a == "--seats" or a.startswith("--seats=") for a in raw_argv)
    explicit_price = any(a == "--price-per-m" or a.startswith("--price-per-m=") for a in raw_argv)

    parser = argparse.ArgumentParser(
        description="Whole Body Compute - Zero-Install Multi-Harness FinOps & Quota Audit"
    )
    parser.add_argument("--days", type=int, default=30, help="Historical window in days (default: 30)")
    parser.add_argument("--price-per-m", type=float, default=5.0, dest="price_per_m", help="Frontier input USD per 1M tokens (default: 5.0)")
    parser.add_argument("--seats", type=int, default=1, help="Number of seats for team projection (default: 1 personal)")
    parser.add_argument("--top", type=int, default=5, help="Top expensive threads to show (default: 5)")
    parser.add_argument("--no-color", action="store_true", dest="no_color", help="Disable ANSI terminal colors")
    parser.add_argument("--no-telemetry", "--offline", action="store_true", dest="no_telemetry", help="Skip anonymous audit completion ping")
    parser.add_argument("--agy-dir", dest="agy_dir", default=None, help="Custom Antigravity brain directory")
    parser.add_argument("--gemini-cli-dir", dest="gemini_cli_dir", default=None, help="Custom Google Gemini CLI tmp directory")
    parser.add_argument("--claude-dir", dest="claude_dir", default=None, help="Custom Claude Code projects directory")
    parser.add_argument("--goose-db", dest="goose_db", default=None, help="Custom Block Goose sessions.db path")
    parser.add_argument("--cursor-dir", dest="cursor_dir", default=None, help="Custom Cursor storage directory")
    parser.add_argument("--codex-dir", dest="codex_dir", default=None, help="Custom OpenAI Codex sessions directory or history.jsonl")
    parser.add_argument("--cline-dir", dest="cline_dir", default=None, help="Custom Cline / Roo Code tasks storage directory")
    parser.add_argument("--windsurf-dir", dest="windsurf_dir", default=None, help="Custom Windsurf Cascade storage directory")
    parser.add_argument("--aider-path", dest="aider_path", default=None, help="Custom Aider chat history path or directory")
    parser.add_argument("--continue-dir", dest="continue_dir", default=None, help="Custom Continue.dev sessions directory")
    parser.add_argument("--zed-dir", dest="zed_dir", default=None, help="Custom Zed AI threads directory")
    parser.add_argument("--workspace-dir", dest="workspace_dir", default=None, help="Custom workspace directory for Turn-0 static instruction/MCP scan")
    parser.add_argument("--home-dir", dest="home_dir", default=None, help="Custom user home directory for Turn-0 global MCP/instruction scan")
    parser.add_argument("--export-receipt", dest="export_receipt", default=None, help="Optional path to export receipt")
    parser.add_argument("--json", action="store_true", help="Output structured JSON")
    args = parser.parse_args(argv)

    audit = run_multi_harness_finops_audit(
        days=args.days,
        price_per_m=args.price_per_m,
        seats=args.seats,
        top=args.top,
        agy_dir=Path(args.agy_dir) if args.agy_dir else None,
        claude_dir=Path(args.claude_dir) if args.claude_dir else None,
        goose_db=Path(args.goose_db) if args.goose_db else None,
        cursor_dir=Path(args.cursor_dir) if args.cursor_dir else None,
        codex_dir=Path(args.codex_dir) if args.codex_dir else None,
        cline_dir=Path(args.cline_dir) if args.cline_dir else None,
        windsurf_dir=Path(args.windsurf_dir) if args.windsurf_dir else None,
        aider_path=Path(args.aider_path) if args.aider_path else None,
        continue_dir=Path(args.continue_dir) if args.continue_dir else None,
        zed_dir=Path(args.zed_dir) if args.zed_dir else None,
        gemini_cli_dir=Path(args.gemini_cli_dir) if args.gemini_cli_dir else None,
        workspace_dir=Path(args.workspace_dir) if args.workspace_dir else None,
        home_dir=Path(args.home_dir) if args.home_dir else None,
        explicit_price_override=explicit_price,
    )
    show_team = bool(explicit_seats or args.seats > 1)
    if args.export_receipt:
        out_p = Path(args.export_receipt)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(
            format_standalone_receipt(audit, color=False, show_team=show_team),
            encoding="utf-8",
        )

    if args.json:
        json_audit = dict(audit)
        json_audit.pop("_local_titles_by_id", None)
        print(json.dumps(json_audit, indent=2), flush=True)
    else:
        use_color = False if args.no_color else None
        print(format_standalone_receipt(audit, color=use_color, show_team=show_team), flush=True)

    _emit_audit_telemetry(audit, no_telemetry=bool(getattr(args, "no_telemetry", False)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
