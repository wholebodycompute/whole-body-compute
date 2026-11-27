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
import math
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
    """Return `(input_usd_per_m, output_usd_per_m, matched_known_model)` for a logged model string.

    Input rates for Gemini 4 / Carbon / Argon / Neon / Hydrogen are aligned with
    `whole_body_compute.shield.MODEL_INPUT_PRICING_PER_M` so `body savings` and `docs/audit.py`
    never diverge on model pricing.
    """
    if explicit_override:
        return default_in, default_out, False
    m = (model_name or "").strip().lower()
    if not m or m in ("<synthetic>", "none", "unknown", "inherit", "default"):
        return default_in, default_out, False
    if any(k in m for k in ("local-metal", "local_metal", "body-coder", "voicefi-coder", "ollama", "mlx-community")):
        return 0.0, 0.0, True
    if "hydrogen" in m:
        return 1.50, 7.50, True
    if "neon" in m:
        return 3.00, 15.00, True
    if any(k in m for k in ("argon", "carbon", "gemini-4", "gemini 4")):
        # Official Gemini 4 Argon/Carbon intro ($2/$10 thru 2026-11-30 UTC) -> standard ($4/$20 from 2026-12-01 UTC)
        import time as _time
        if _time.time() >= 1796083200.0:
            return 4.00, 20.00, True
        return 2.00, 10.00, True
    if any(k in m for k in ("flash-lite", "flash_lite", "gpt-4.1-nano", "gpt-5-nano")):
        return 0.50, 2.50, True
    if any(k in m for k in ("gemini-2.5-flash", "gemini-2.0-flash")):
        return 0.30, 2.50, True
    if any(k in m for k in ("haiku-3-5", "haiku-3.5", "3-5-haiku", "3.5-haiku")):
        return 0.80, 4.00, True
    if any(
        k in m
        for k in ("haiku", "flash", "gpt-4o-mini", "gpt-4.1-mini", "gpt-5-mini", "o3-mini", "o4-mini")
    ):
        return 1.00, 5.00, True
    if "gemini-2.5-pro" in m:
        return 1.25, 10.00, True
    if any(k in m for k in ("gemini-3", "gemini 3", "gemini-pro")):
        return 2.00, 12.00, True
    if "gpt-5" in m and "pro" not in m:
        return 1.25, 10.00, True
    if ("gpt-4.1" in m or "o3" in m) and "pro" not in m:
        return 2.00, 8.00, True
    if "gpt-4o" in m:
        return 2.50, 10.00, True
    if any(k in m for k in ("opus-4-5", "opus-4.5", "opus 4.5", "opus-4-6", "opus 4.6", "opus-5", "opus 5")):
        return 5.00, 25.00, True
    if "opus" in m:
        return 15.00, 75.00, True
    if any(k in m for k in ("sonnet", "orchestrator", "gpt-5", "o3")):
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
    "[BLOB:",
    "AST_SLICE",
    "WBC_BLOOD_BRAIN_BARRIER",
    "Zero-Turn AST",
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


_BENCHMARK_PATH_TOKENS = (
    "-swebench-",
    "_swebench_",
    "/swebench",
    "swebench-",
    "-agentdojo-",
    "_agentdojo_",
    "/agentdojo",
    "agentdojo-",
    "-vanilla-arm-",
    "_vanilla_arm_",
    "-wbc-arm-",
    "_wbc_arm_",
    "scratch-claude-live-test",
    "pytest-of-",
    "wbc_onboarding_verify",
)

_AGY_BENCHMARK_PROMPT_MARKERS = (
    "/tmp/wbc-",
    "/tmp/wbc_",
    "/private/tmp/wbc",
    "princeton-nlp/swe-bench",
    "swebench",
    "swe-bench",
    "agentdojo",
    "mcptox",
    "osharm",
    "os-harm",
    "cyberseceval",
    "redcode",
    "injecagent",
    "zero_dollar_gauntlet",
    "uat_break_it",
    "wbc_onboarding_verify",
    "scratch-claude-live-test",
    "vanilla_arm",
    "vanilla-arm",
    "wbc_arm",
    "wbc-arm",
)

_AGY_HEADLESS_EVAL_TEMPLATES = (
    "available virtual tools:\n- ",
    "you are an ai language model who assists the user by using the given tools",
    "you are a helpful assistant with access to virtual tools in an isolated sandbox",
)

_AGY_BENCHMARK_TOOL_ARGS_MARKERS = (
    "/tmp/wbc-",
    "/tmp/wbc_",
    "/private/tmp/wbc",
    "princeton-nlp/swe-bench",
    "zero_dollar_gauntlet",
    "uat_break_it",
    "wbc_onboarding_verify",
)


def _resolve_include_benchmarks(include_benchmarks: bool = False) -> bool:
    if include_benchmarks:
        return True
    env_v = os.environ.get("WBC_AUDIT_INCLUDE_BENCHMARKS", "").strip().lower()
    return env_v in ("1", "true", "yes")


def _bump_excluded_stat(
    excluded_stats: Optional[Dict[str, int]], harness: str, count: int = 1
) -> None:
    if isinstance(excluded_stats, dict) and count > 0:
        excluded_stats[harness] = int(excluded_stats.get(harness, 0)) + int(count)
        excluded_stats["_total"] = int(excluded_stats.get("_total", 0)) + int(count)


def _is_benchmark_path_or_cwd(path_or_label: str) -> bool:
    s = str(path_or_label or "").strip().lower().replace("\\", "/")
    if not s:
        return False
    if s.startswith("file://"):
        s = s[7:]
    if s in ("/tmp", "/private/tmp", "/var/folders", "/private/var/folders"):
        return True
    if s.startswith(
        (
            "/tmp/",
            "/private/tmp/",
            "/var/folders/",
            "/private/var/folders/",
            "tmp/",
            "private/tmp/",
            "var/folders/",
            "private/var/folders/",
        )
    ):
        return True
    if any(seg in s for seg in ("/tmp/", "/private/tmp/", "/var/folders/")):
        return True
    parts = [p for p in s.split("/") if p]
    if any(
        p.startswith(("-tmp-", "-private-tmp-", "-var-folders-", "-private-var-folders-"))
        or p in ("-tmp", "-private-tmp")
        for p in parts
    ):
        return True
    return any(tok in s for tok in _BENCHMARK_PATH_TOKENS)


def _is_rel_transcript_benchmark(
    fpath: Path, scan_root: Optional[Path] = None, cwd_hint: str = ""
) -> bool:
    if scan_root is not None:
        try:
            if fpath == scan_root:
                rel_s = fpath.name
            else:
                rel_s = fpath.resolve().relative_to(scan_root.resolve()).as_posix()
        except Exception:
            rel_s = fpath.name
    else:
        rel_s = fpath.as_posix()
    if _is_benchmark_path_or_cwd(rel_s):
        return True
    if cwd_hint:
        c_str = str(cwd_hint).strip().replace("\\", "/")
        if scan_root is not None:
            try:
                test_base = scan_root.resolve().parent.as_posix()
                c_res = Path(c_str).resolve().as_posix() if c_str.startswith("/") else c_str
                if c_res.startswith(test_base + "/"):
                    c_str = c_res[len(test_base) + 1 :]
            except Exception:
                pass
        if _is_benchmark_path_or_cwd(c_str):
            return True
    return False


def _is_agy_conv_direct_benchmark(raw_objs: List[Tuple[Dict[str, Any], int]]) -> bool:
    if not raw_objs:
        return False
    head_objs = raw_objs[:60]
    is_subagent = str(head_objs[0][0].get("type") or "") == "SYSTEM_MESSAGE"
    prompt_txt = ""
    for idx, (obj, _) in enumerate(head_objs):
        if not isinstance(obj, dict):
            continue
        for ws_key in ("workspace_uri", "cwd", "workspace"):
            ws_val = str(obj.get(ws_key) or "").strip()
            if ws_val and _is_benchmark_path_or_cwd(ws_val):
                return True
        stype = str(obj.get("type") or "")
        content_str = str(obj.get("content") or "")
        if stype == "USER_INPUT":
            if "<ADDITIONAL_METADATA>" in content_str and "Active Workspaces:" in content_str:
                meta_part = content_str.split("<ADDITIONAL_METADATA>", 1)[1].split(
                    "</ADDITIONAL_METADATA>", 1
                )[0]
                for mln in meta_part.splitlines():
                    ms = mln.strip()
                    if ms.startswith("- /") and _is_benchmark_path_or_cwd(ms[2:].strip()):
                        return True
            if not prompt_txt:
                if "<USER_REQUEST>" in content_str and "</USER_REQUEST>" in content_str:
                    prompt_txt = content_str.split("<USER_REQUEST>", 1)[1].split(
                        "</USER_REQUEST>", 1
                    )[0]
                else:
                    prompt_txt = content_str
        elif (
            is_subagent
            and idx < 6
            and stype == "SYSTEM_MESSAGE"
            and "[Message]" in content_str
            and "[swarm] new board message" not in content_str
            and not prompt_txt
        ):
            prompt_txt = content_str

        if stype == "PLANNER_RESPONSE":
            for tc in obj.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                args_obj = tc.get("args") or tc.get("arguments") or tc.get("input") or {}
                try:
                    args_low = json.dumps(args_obj, ensure_ascii=False).lower()
                except Exception:
                    args_low = str(args_obj).lower()
                if any(mk in args_low for mk in _AGY_BENCHMARK_TOOL_ARGS_MARKERS):
                    return True
                if is_subagent and str(tc.get("name") or "") == "run_command":
                    if "cd /tmp/" in args_low or "cd /private/tmp/" in args_low:
                        return True

    if prompt_txt:
        low_p = prompt_txt.lower()
        if any(tpl in low_p for tpl in _AGY_HEADLESS_EVAL_TEMPLATES):
            return True
        is_repo_edit = "allowed files only" in low_p and "src/whole_body_compute/" in low_p
        if not is_repo_edit and any(mk in low_p for mk in _AGY_BENCHMARK_PROMPT_MARKERS):
            return True
    return False


def scan_antigravity_harness(
    agy_dir: Path,
    window_seconds: float,
    now: float,
    in_price_per_m: float,
    out_price_per_m: float,
    explicit_price_override: bool = False,
    include_benchmarks: bool = False,
    excluded_stats: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not agy_dir.is_dir():
        return records

    eff_inc_bench = _resolve_include_benchmarks(include_benchmarks)
    parsed_convs: List[Tuple[str, List[Tuple[Dict[str, Any], int]], int, int, int, float]] = []
    spawned_model_by_cid: Dict[str, str] = {}
    conv_declared_model: Dict[str, str] = {}
    parent_by_cid: Dict[str, str] = {}
    sender_re = re.compile(r"sender=([A-Za-z0-9_-]{8,})")
    cont_re = re.compile(r"Continuing from (?:conversation://)?([0-9a-fA-F\-]{36})")

    for c_dir in agy_dir.iterdir():
        if not c_dir.is_dir() or c_dir.name.startswith("."):
            continue
        if not eff_inc_bench and _is_benchmark_path_or_cwd(c_dir.name):
            _bump_excluded_stat(excluded_stats, "antigravity", 1)
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

            cid = c_dir.name
            fallback_sender = ""
            for step_i, (obj, _) in enumerate(raw_objs):
                stype = str(obj.get("type") or "")
                content_str = str(obj.get("content") or "")
                ms_cand = _extract_model_selection_change(content_str)
                if ms_cand:
                    conv_declared_model[cid] = ms_cand
                if obj.get("model"):
                    conv_declared_model[cid] = str(obj.get("model"))
                if stype == "SYSTEM_MESSAGE" and "sender=" in content_str:
                    m_snd = sender_re.search(content_str)
                    if m_snd and m_snd.group(1).lower() != "system":
                        snd_id = m_snd.group(1)
                        if "[swarm] new board message" in content_str:
                            if not fallback_sender:
                                fallback_sender = snd_id
                        elif cid not in parent_by_cid:
                            parent_by_cid[cid] = snd_id
                elif stype == "USER_INPUT" and step_i < 5 and cid not in parent_by_cid:
                    m_cont = cont_re.search(content_str)
                    if m_cont and m_cont.group(1) != cid:
                        parent_by_cid[cid] = m_cont.group(1)
                if content_str.startswith("{") and "conversation_id" in content_str:
                    try:
                        spawn_meta = json.loads(content_str)
                        if isinstance(spawn_meta, dict) and spawn_meta.get("conversation_id"):
                            s_model = str(spawn_meta.get("model_display_name") or spawn_meta.get("model") or "").strip()
                            if s_model:
                                spawned_model_by_cid[str(spawn_meta["conversation_id"])] = s_model
                    except Exception:
                        pass
            if cid not in parent_by_cid and fallback_sender:
                parent_by_cid[cid] = fallback_sender

            parsed_convs.append((cid, raw_objs, compact_bytes, full_bytes, conv_bytes, mtime))
        except Exception:
            continue

    bench_cids: set[str] = set()
    if not eff_inc_bench:
        for cid, raw_objs, *_ in parsed_convs:
            if _is_agy_conv_direct_benchmark(raw_objs):
                bench_cids.add(cid)
        # Also check any parent transcript outside window if referenced by a child in window
        unchecked_parents = {
            pid for cid, pid in parent_by_cid.items() if cid not in bench_cids and pid not in bench_cids
        }
        parsed_cid_set = {c for c, *_ in parsed_convs}
        for pid in unchecked_parents:
            if pid in parsed_cid_set:
                continue
            p_logs = agy_dir / pid / ".system_generated" / "logs"
            p_file = (
                p_logs / "transcript.jsonl"
                if (p_logs / "transcript.jsonl").is_file()
                else (p_logs / "transcript_full.jsonl" if (p_logs / "transcript_full.jsonl").is_file() else None)
            )
            if p_file:
                try:
                    p_head: List[Tuple[Dict[str, Any], int]] = []
                    with open(p_file, "r", encoding="utf-8", errors="ignore") as pfh:
                        for _ in range(60):
                            pln = pfh.readline()
                            if not pln:
                                break
                            ps = pln.strip()
                            if ps:
                                p_head.append((json.loads(ps), len(ps)))
                    if _is_agy_conv_direct_benchmark(p_head):
                        bench_cids.add(pid)
                except Exception:
                    pass
        changed = True
        while changed:
            changed = False
            for cid, pid in parent_by_cid.items():
                if cid not in bench_cids and pid in bench_cids:
                    bench_cids.add(cid)
                    changed = True

    for cid, raw_objs, compact_bytes, full_bytes, conv_bytes, mtime in parsed_convs:
        if not eff_inc_bench and cid in bench_cids:
            _bump_excluded_stat(excluded_stats, "antigravity", 1)
            continue
        try:
            trunc_cnt = sum(1 for obj, _ in raw_objs if obj.get("truncated_fields"))
            overflow_per_trunc = (
                float(max(0, full_bytes - compact_bytes)) / float(max(1, trunc_cnt))
                if trunc_cnt > 0
                else 0.0
            )
            step_tuples: List[Tuple[bool, int, bool]] = []
            tool_counts: Dict[str, int] = {}
            first_prompt = ""
            active_model = (
                spawned_model_by_cid.get(cid)
                or conv_declared_model.get(parent_by_cid.get(cid, ""), "")
            )
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
                short_id=cid[:8],
                conv_id=cid,
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


_AUDIT_SECRET_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(
        r"-----BEGIN[ A-Z0-9\-]*PRIVATE KEY[ A-Z0-9\-]*-----.*?-----END[ A-Z0-9\-]*PRIVATE KEY[ A-Z0-9\-]*-----",
        re.DOTALL | re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:"
        r"sk-[A-Za-z0-9_\-]{12,}"
        r"|sk_(?:live|test)_[A-Za-z0-9]{8,}"
        r"|pk_(?:live|test)_[A-Za-z0-9]{8,}"
        r"|rk_(?:live|test)_[A-Za-z0-9]{8,}"
        r"|whsec_[A-Za-z0-9+/=]{8,}"
        r"|ghp_[A-Za-z0-9]{16,}"
        r"|gh[osu]_[A-Za-z0-9]{16,}"
        r"|github_pat_[A-Za-z0-9_]{16,}"
        r"|glpat-[A-Za-z0-9_\-]{16,}"
        r"|xox[baprs]-[A-Za-z0-9\-]{10,}"
        r"|AKIA[0-9A-Z]{16}"
        r"|ASIA[0-9A-Z]{16}"
        r"|AIza[0-9A-Za-z_\-]{20,}"
        r"|ya29\.[0-9A-Za-z_\-]{16,}"
        r"|ph[cx]_[A-Za-z0-9]{16,}"
        r"|pypi-[A-Za-z0-9_\-]{20,}"
        r"|hf_[A-Za-z0-9]{16,}"
        r"|npm_[A-Za-z0-9]{20,}"
        r"|eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"
        r")\b"
    ),
    re.compile(r"(?i)(https?://)([^/\s:@]+:[^/\s@]+)@"),
    re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9._\-+/=]{8,}"),
    re.compile(
        r"(?i)\b(api[_\-]?key|secret|token|password|passwd|pwd|access[_\-]?key|client[_\-]?secret|"
        r"anthropic_api_key|openai_api_key|github_token|aws_secret_access_key)"
        r"(\s*[=:]\s*)(['\"]?)([^\s'\"`&;|]{4,})\3"
    ),
    re.compile(r"(?i)(\s(?:--token|--password|--api-key|-p|-u)\s+)(['\"]?)([^\s'\"`&;|]{4,})\2"),
)


def _redact_audit_secrets(text: str) -> str:
    """Stdlib-only credential redactor for Bash command previews and lowering patterns."""
    cleaned = str(text or "")
    if not cleaned:
        return ""
    for idx, pat in enumerate(_AUDIT_SECRET_PATTERNS):
        if idx == 2:
            cleaned = pat.sub(r"\1[redacted]@", cleaned)
        elif idx == 3:
            cleaned = pat.sub(r"\1[redacted]", cleaned)
        elif idx == 4:
            cleaned = pat.sub(r"\1\2[redacted]", cleaned)
        elif idx == 5:
            cleaned = pat.sub(r"\1[redacted]", cleaned)
        else:
            cleaned = pat.sub("[redacted]", cleaned)
    return cleaned


_BASH_ENV_PREFIX_RE = re.compile(r"^(?:[A-Za-z_][A-Za-z0-9_]*=(?:'[^']*'|\"[^\"]*\"|\S+)\s+)+")
_BASH_RUNNER_PREFIX_RE = re.compile(
    r"^(?:uv\s+run\s+|npx\s+|pnpm\s+(?:exec\s+)?|yarn\s+|poetry\s+run\s+|python3?(?:\.\d+)?\s+-m\s+)+"
)
_BASH_GIT_PLUMBING_RE = re.compile(
    r"^git\s+(?:-[Cc]\s+\S+\s+)*(status|diff|add|commit|push|branch|checkout|switch|log|fetch|pull|rebase|merge|stash|rev-parse|show)\b"
)
_BASH_GH_PR_RE = re.compile(r"^gh\s+pr\s+(create|view|checks|comment|merge|list|status|diff)\b")
_BASH_NOISY_TEST_BUILD_RE = re.compile(
    r"(?:^|\b)(?:pytest|py\.test|vitest|jest|mocha|cargo\s+test|go\s+test|npm\s+test|pnpm\s+test|yarn\s+test|xcodebuild|docker\s+build|make\b)"
)
_BASH_DATA_WRANGLE_TOOL_RE = re.compile(r"^(jq|grep|rg|awk|sed|cat|head|tail|wc|curl)\b")
_BASH_PY_INLINE_RE = re.compile(r"^python3?(?:\.\d+)?\s+-c\b")
_BASH_PY_DATA_HINT_RE = re.compile(r"(?i)(json|csv|sarif|yaml|xml|load\(|loads\(|dictreader|stdin|open\(|split\(|parse)")


def _classify_bash_command(cmd: str) -> Tuple[Optional[str], str, bool]:
    """Return `(lowering_category, normalized_verb, is_noisy_test_build)` for a Bash command."""
    raw = str(cmd or "").strip()
    if not raw:
        return None, "", False
    segments = [seg.strip() for seg in re.split(r"(?:&&|\|\||[;\n|])", raw) if seg.strip()]
    is_noisy = bool(_BASH_NOISY_TEST_BUILD_RE.search(raw))

    detected_cat: Optional[str] = None
    detected_verb = ""
    for seg in segments:
        norm_seg = _BASH_ENV_PREFIX_RE.sub("", seg).strip()
        runner_seg = _BASH_RUNNER_PREFIX_RE.sub("", norm_seg).strip()
        low = runner_seg.lower()
        low_orig = norm_seg.lower()

        if low.startswith("ruff"):
            return "lint_format_loops", "ruff", is_noisy
        if low.startswith("black"):
            return "lint_format_loops", "black", is_noisy
        if low.startswith("isort"):
            return "lint_format_loops", "isort", is_noisy
        if low.startswith("prettier"):
            return "lint_format_loops", "prettier", is_noisy
        if low.startswith("eslint"):
            return "lint_format_loops", "eslint", is_noisy
        if low.startswith("tsc") and "--noemit" in low:
            return "lint_format_loops", "tsc:noEmit", is_noisy
        if low.startswith("gofmt"):
            return "lint_format_loops", "gofmt", is_noisy
        if low.startswith("cargo fmt"):
            return "lint_format_loops", "cargo:fmt", is_noisy
        if low.startswith("cargo clippy"):
            return "lint_format_loops", "cargo:clippy", is_noisy

        m_git = _BASH_GIT_PLUMBING_RE.match(low_orig)
        if m_git and detected_cat is None:
            detected_cat = "git_pr_plumbing"
            detected_verb = f"git:{m_git.group(1)}"
            continue
        m_gh = _BASH_GH_PR_RE.match(low_orig)
        if m_gh and detected_cat is None:
            detected_cat = "git_pr_plumbing"
            detected_verb = f"gh:pr:{m_gh.group(1)}"
            continue

        if not is_noisy and detected_cat is None:
            m_dw = _BASH_DATA_WRANGLE_TOOL_RE.match(low_orig)
            if m_dw:
                detected_cat = "data_log_wrangling"
                detected_verb = m_dw.group(1)
                continue
            if _BASH_PY_INLINE_RE.match(low_orig) and _BASH_PY_DATA_HINT_RE.search(norm_seg):
                detected_cat = "data_log_wrangling"
                detected_verb = "python:-c"
                continue

    return detected_cat, detected_verb, is_noisy


def _canonical_mcp_server_key(name: str) -> str:
    norm = re.sub(r"[^a-z0-9]+", "", str(name or "").lower())
    if norm.endswith("mcp") and len(norm) > 3:
        norm = norm[:-3]
    return norm


def _is_safe_contained_file(path: Path, root: Path) -> bool:
    """Return True iff `path` is a non-symlink regular file strictly contained inside `root`."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    cur = root
    for part in rel.parts:
        cur = cur / part
        try:
            if cur.is_symlink():
                return False
        except OSError:
            return False
    try:
        resolved_root = root.resolve()
        resolved_path = path.resolve()
        if not resolved_path.is_relative_to(resolved_root):
            return False
        return path.is_file()
    except OSError:
        return False


def _collect_lowering_mcp_configs(
    turn0_static_overhead: Dict[str, Any],
    ws_dir: Path,
    h_dir: Path,
    ci_dir: Optional[Path],
    session_declared_mcps: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Collect configured MCP servers from Turn-0 static scan, `.mcp.json`, `.claude/settings*.json`, and session init tools."""
    servers: Dict[str, Dict[str, Any]] = {}

    def _add_srv(raw_name: str, tokens: int) -> None:
        disp = _sanitize_turn0_label(raw_name)
        ckey = _canonical_mcp_server_key(disp)
        if not ckey:
            return
        if ckey not in servers:
            servers[ckey] = {"name": disp, "tokens": max(0, int(tokens))}
        elif tokens > int(servers[ckey]["tokens"]):
            servers[ckey]["tokens"] = int(tokens)

    for srv in turn0_static_overhead.get("mcp_servers", []) or []:
        if isinstance(srv, dict) and srv.get("name"):
            _add_srv(str(srv["name"]), int(srv.get("tokens", DEFAULT_DEFERRED_TOKENS_PER_MCP_SERVER)))

    extra_cfgs: List[Path] = [
        ws_dir / ".claude" / "settings.json",
        ws_dir / ".claude" / "settings.local.json",
        h_dir / ".claude" / "settings.json",
        h_dir / ".claude" / "settings.local.json",
    ]
    ci_p: Optional[Path] = Path(ci_dir).expanduser() if ci_dir is not None else None
    if ci_p is not None:
        extra_cfgs.extend(
            [
                ci_p / ".mcp.json",
                ci_p / ".claude" / "settings.json",
                ci_p / ".claude" / "settings.local.json",
            ]
        )

    seen_rp: set[str] = set()
    for cfg_p in extra_cfgs:
        try:
            if ci_p is not None:
                if not _is_safe_contained_file(cfg_p, ci_p):
                    continue
            elif not cfg_p.is_file():
                continue
            rp = os.path.realpath(cfg_p)
            if rp in seen_rp:
                continue
            seen_rp.add(rp)
            if os.path.getsize(cfg_p) > 2_000_000:
                continue
            parsed = json.loads(cfg_p.read_text(encoding="utf-8", errors="ignore"))
            if not isinstance(parsed, dict):
                continue
            for key_name in ("mcpServers", "servers"):
                srv_dict = parsed.get(key_name)
                if isinstance(srv_dict, dict):
                    for sname, scfg in srv_dict.items():
                        if isinstance(scfg, dict) and scfg.get("disabled") is True:
                            continue
                        _add_srv(str(sname), DEFAULT_DEFERRED_TOKENS_PER_MCP_SERVER)
            enabled_list = parsed.get("enabledMcpjsonServers")
            if isinstance(enabled_list, list):
                for sname in enabled_list:
                    if isinstance(sname, str) and sname.strip():
                        _add_srv(sname.strip(), DEFAULT_DEFERRED_TOKENS_PER_MCP_SERVER)
            perms = parsed.get("permissions")
            allow_lists: List[Any] = []
            if isinstance(perms, dict):
                allow_lists.append(perms.get("allow"))
            allow_lists.append(parsed.get("allowedTools"))
            for lst in allow_lists:
                if isinstance(lst, list):
                    for entry in lst:
                        if isinstance(entry, str) and entry.startswith("mcp__"):
                            parts = entry.split("__", 2)
                            if len(parts) >= 2 and parts[1]:
                                _add_srv(parts[1], DEFAULT_DEFERRED_TOKENS_PER_MCP_SERVER)
        except Exception:
            continue

    for sname in session_declared_mcps:
        _add_srv(sname, DEFAULT_DEFERRED_TOKENS_PER_MCP_SERVER)

    return servers


def _analyze_claude_lowering_choreography(
    claude_recs: List[Dict[str, Any]],
    traces_by_id: Dict[str, Dict[str, Any]],
    turn0_static_overhead: Dict[str, Any],
    ws_dir: Path,
    h_dir: Path,
    ci_dir: Optional[Path],
    days: int,
    in_price: float,
    total_monthly_usd: float,
    hook_fired_by_id: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """Deterministic Bash choreography fingerprinter & multiplicative dual-wedge savings model."""
    ci_mode = bool(ci_dir is not None)
    all_event_ts: List[float] = []
    for rec in claude_recs:
        cid = str(rec.get("conv_id", ""))
        trace = traces_by_id.get(cid) or {}
        for ts_k in ("min_event_ts", "max_event_ts"):
            ts_v = float(trace.get(ts_k) or 0.0)
            if ts_v > 0.0:
                all_event_ts.append(ts_v)
    if len(all_event_ts) >= 2 and max(all_event_ts) > min(all_event_ts):
        observed_span_hours = round((max(all_event_ts) - min(all_event_ts)) / 3600.0, 4)
    else:
        observed_span_hours = 0.0

    use_monthly = (
        (not ci_mode)
        and (days >= 7)
        and (observed_span_hours == 0.0 or observed_span_hours >= 168.0)
    )
    period_unit = "/mo" if use_monthly else " (audited period)"
    scale = (30.0 / float(max(1, days))) if (use_monthly and days > 0) else 1.0

    claude_period_usd = round(sum(float(r.get("exact_api_spend_usd", 0.0)) for r in claude_recs), 2)
    claude_monthly_usd = round(claude_period_usd * scale, 2)
    total_claude_turns = sum(max(1, int(r.get("planner_turns", 0))) for r in claude_recs)
    if not claude_recs or (total_claude_turns == 0 and claude_monthly_usd <= 0):
        baseline_period_usd = 0.0
        baseline_monthly_usd = 0.0
    else:
        baseline_period_usd = claude_period_usd if claude_period_usd > 0 else 0.0
        baseline_monthly_usd = claude_monthly_usd if claude_monthly_usd > 0 else 0.0
    edit_tools = frozenset({"edit", "write", "multiedit", "notebookedit"})

    candidate_runs: List[Dict[str, Any]] = []
    invoked_mcp_keys: set[str] = set()
    session_declared_mcps: List[str] = []
    unbounded_read_calls = 0
    noisy_bash_calls = 0
    session_bloat_flags: Dict[str, Tuple[int, int]] = {}

    for rec in claude_recs:
        cid = str(rec.get("conv_id", ""))
        trace = traces_by_id.get(cid) or {}
        ordered_tools: List[Dict[str, Any]] = list(trace.get("ordered_tools") or [])
        for dmcp in trace.get("turn0_declared_mcps") or []:
            session_declared_mcps.append(str(dmcp))

        s_unread = int(trace.get("unbounded_read_calls", 0))
        s_nbash = int(trace.get("noisy_bash_calls", 0))
        unbounded_read_calls += s_unread
        noisy_bash_calls += s_nbash
        session_bloat_flags[cid] = (s_unread, s_nbash)

        for item in ordered_tools:
            msrv = str(item.get("mcp_server") or "")
            if msrv:
                ck = _canonical_mcp_server_key(msrv)
                if ck:
                    invoked_mcp_keys.add(ck)

        # 1. Extract contiguous runs for git_pr_plumbing and data_log_wrangling
        for cat in ("git_pr_plumbing", "data_log_wrangling"):
            cur_run: List[Dict[str, Any]] = []
            for item in ordered_tools:
                if item.get("cat") == cat:
                    cur_run.append(item)
                else:
                    if cur_run:
                        verbs = tuple(str(x.get("verb") or cat) for x in cur_run)
                        sig = (cat, verbs)
                        candidate_runs.append(
                            {
                                "cid": cid,
                                "cat": cat,
                                "items": cur_run,
                                "sig": sig,
                                "core_calls": len(cur_run),
                                "total_steps": len(cur_run),
                            }
                        )
                        cur_run = []
            if cur_run:
                verbs = tuple(str(x.get("verb") or cat) for x in cur_run)
                sig = (cat, verbs)
                candidate_runs.append(
                    {
                        "cid": cid,
                        "cat": cat,
                        "items": cur_run,
                        "sig": sig,
                        "core_calls": len(cur_run),
                        "total_steps": len(cur_run),
                    }
                )

        # 2. Extract lint_format_loops (allowing interleaved Edit/Write between lint runs)
        n_tools = len(ordered_tools)
        idx = 0
        while idx < n_tools:
            if ordered_tools[idx].get("cat") == "lint_format_loops":
                run_items: List[Dict[str, Any]] = [ordered_tools[idx]]
                j = idx + 1
                while j < n_tools:
                    nxt = ordered_tools[j]
                    if nxt.get("cat") == "lint_format_loops":
                        run_items.append(nxt)
                        j += 1
                    elif nxt.get("norm") in edit_tools:
                        # Look ahead to see if another lint_format_loops follows after Edit/Write(s)
                        k = j + 1
                        while k < n_tools and ordered_tools[k].get("norm") in edit_tools:
                            k += 1
                        if k < n_tools and ordered_tools[k].get("cat") == "lint_format_loops":
                            run_items.extend(ordered_tools[j : k + 1])
                            j = k + 1
                        else:
                            break
                    else:
                        break
                lint_core = [x for x in run_items if x.get("cat") == "lint_format_loops"]
                verbs = tuple(str(x.get("verb") or "lint") for x in lint_core)
                candidate_runs.append(
                    {
                        "cid": cid,
                        "cat": "lint_format_loops",
                        "items": run_items,
                        "sig": ("lint_format_loops", verbs),
                        "core_calls": len(lint_core),
                        "total_steps": len(run_items),
                    }
                )
                idx = j
            else:
                idx += 1

    # Count recurrence across distinct sessions per signature and per primary verb
    sig_sessions: Dict[Any, set[str]] = {}
    sig_calls: Dict[Any, int] = {}
    verb_sessions: Dict[Tuple[str, str], set[str]] = {}
    verb_calls: Dict[Tuple[str, str], int] = {}
    for run in candidate_runs:
        sig = run["sig"]
        sig_sessions.setdefault(sig, set()).add(run["cid"])
        sig_calls[sig] = sig_calls.get(sig, 0) + int(run["core_calls"])
        for it in run["items"]:
            if it.get("cat") == run["cat"]:
                vk = (run["cat"], str(it.get("verb") or run["cat"]))
                verb_sessions.setdefault(vk, set()).add(run["cid"])
                verb_calls[vk] = verb_calls.get(vk, 0) + 1

    qualifying_runs: List[Dict[str, Any]] = []
    for run in candidate_runs:
        cat = run["cat"]
        sig = run["sig"]
        is_consec = (
            (run["total_steps"] >= 3 and run["core_calls"] >= 2)
            if cat == "lint_format_loops"
            else (run["total_steps"] >= 3)
        )
        is_recurring_sig = len(sig_sessions.get(sig, set())) >= 2 and (
            sig_calls.get(sig, 0) >= 3 or run["total_steps"] >= 2
        )
        first_v = str(sig[1][0]) if sig[1] else cat
        is_recurring_verb = (
            len(verb_sessions.get((cat, first_v), set())) >= 2
            and verb_calls.get((cat, first_v), 0) >= 3
        )
        if is_consec or is_recurring_sig or is_recurring_verb:
            qualifying_runs.append(run)

    cat_stats: Dict[str, Dict[str, Any]] = {
        "git_pr_plumbing": {
            "sequences": 0,
            "calls": 0,
            "turns": 0,
            "patterns": [],
            "recommendation": "body commit + Python spine",
            "_turn_keys": set(),
            "_sigs": set(),
        },
        "data_log_wrangling": {
            "sequences": 0,
            "calls": 0,
            "turns": 0,
            "patterns": [],
            "recommendation": "Deterministic Python/jq Tier 0/1 script",
            "_turn_keys": set(),
            "_sigs": set(),
        },
        "lint_format_loops": {
            "sequences": 0,
            "calls": 0,
            "turns": 0,
            "patterns": [],
            "recommendation": "Pre-commit / Tier 0 auto-fix loop",
            "_turn_keys": set(),
            "_sigs": set(),
        },
    }

    session_lowerable_turn_keys: Dict[str, set[str]] = {}
    distinct_patterns: set[Any] = set()

    for run in qualifying_runs:
        cat = run["cat"]
        cs = cat_stats[cat]
        cs["sequences"] += 1
        cs["calls"] += int(run["total_steps"])
        distinct_patterns.add(run["sig"])
        cs["_sigs"].add(run["sig"])
        cid = run["cid"]
        tkeys_set = session_lowerable_turn_keys.setdefault(cid, set())
        for it in run["items"]:
            tk = f"{cid}:{it.get('turn_key', '')}"
            tkeys_set.add(str(it.get("turn_key", "")))
            cs["_turn_keys"].add(tk)
        if len(cs["patterns"]) < 5:
            verbs_str = " -> ".join(list(run["sig"][1])[:5])
            first_cmd = ""
            for it in run["items"]:
                if it.get("cmd"):
                    first_cmd = _redact_audit_secrets(str(it["cmd"]))
                    first_cmd = " ".join(first_cmd.split())[:84]
                    break
            preview = f"{verbs_str} ({first_cmd})" if first_cmd else verbs_str
            preview = _redact_audit_secrets(preview)[:120]
            if preview and preview not in cs["patterns"]:
                cs["patterns"].append(preview)

    for cs in cat_stats.values():
        cs["turns"] = len(cs.pop("_turn_keys"))
        cs["distinct_patterns"] = len(cs.pop("_sigs"))

    fired_map = hook_fired_by_id or {}
    already_shielded_sessions = sum(
        1 for rec in claude_recs if bool(fired_map.get(str(rec.get("conv_id", "")), False))
    )

    total_lowerable_turns = 0
    raw_code_period_usd = 0.0
    session_code_ratio_by_id: Dict[str, float] = {}

    for rec in claude_recs:
        cid = str(rec.get("conv_id", ""))
        s_turns = max(1, int(rec.get("planner_turns", 0)))
        low_tkeys = session_lowerable_turn_keys.get(cid, set())
        low_turns = min(s_turns, len(low_tkeys))
        total_lowerable_turns += low_turns

        trace = traces_by_id.get(cid) or {}
        turn_cost_map = trace.get("turn_cost_by_key")
        total_turn_cost = (
            sum(float(v) for v in turn_cost_map.values())
            if isinstance(turn_cost_map, dict) and turn_cost_map
            else 0.0
        )
        if isinstance(turn_cost_map, dict) and turn_cost_map and total_turn_cost > 0.0:
            low_cost = sum(max(0.0, float(turn_cost_map.get(tk, 0.0))) for tk in low_tkeys)
            c_ratio = min(0.94, max(0.0, low_cost / total_turn_cost))
            session_code_ratio_by_id[cid] = c_ratio
            raw_code_period_usd += min(0.94 * total_turn_cost, max(0.0, low_cost))
        else:
            c_ratio = min(0.94, float(low_turns) / float(s_turns)) if s_turns > 0 else 0.0
            session_code_ratio_by_id[cid] = c_ratio
            raw_code_period_usd += float(rec.get("exact_api_spend_usd", 0.0)) * c_ratio

    code_lowering_turn_pct = (
        round(min(100.0, (float(total_lowerable_turns) / float(max(1, total_claude_turns))) * 100.0), 1)
        if total_claude_turns > 0
        else 0.0
    )

    configured_mcps = _collect_lowering_mcp_configs(
        turn0_static_overhead=turn0_static_overhead,
        ws_dir=ws_dir,
        h_dir=h_dir,
        ci_dir=ci_dir,
        session_declared_mcps=session_declared_mcps,
    )
    dormant_entries = [
        srv_info
        for ckey, srv_info in sorted(configured_mcps.items())
        if ckey not in invoked_mcp_keys
    ]
    dormant_mcps = [str(e["name"]) for e in dormant_entries]
    dormant_mcp_tokens = sum(int(e["tokens"]) for e in dormant_entries)

    # Multiplicative Non-Overlapping Math:
    # Stage 1: Code lowering savings S_code (exact metered per-turn attribution)
    # Remaining spend: T_rem = max(0, T - S_code)
    # Stage 2: WBC Linux Shield savings strictly on remaining unshielded turns T_rem,
    # scaled quadratically by (1 - c_ratio)^2, capped at empirical 35.0% of T_rem,
    # and bounded so S_code + S_wbc <= 0.94 * T.
    max_total_usd = math.floor(0.94 * baseline_monthly_usd * 100.0) / 100.0 if baseline_monthly_usd > 0 else 0.0
    s_code = round(min(max_total_usd, raw_code_period_usd * scale), 2) if baseline_monthly_usd > 0 else 0.0
    t_rem = round(max(0.0, baseline_monthly_usd - s_code), 2)

    rem_hal_period_usd = 0.0
    bonus_bloat_period_usd = 0.0
    unshielded_recs: List[Dict[str, Any]] = []
    unshielded_rem_period_usd = 0.0
    for rec in claude_recs:
        cid = str(rec.get("conv_id", ""))
        if bool(fired_map.get(cid, False)):
            continue
        unshielded_recs.append(rec)
        c_ratio = session_code_ratio_by_id.get(cid, 0.0)
        rem_frac = max(0.0, 1.0 - c_ratio)
        s_spend = float(rec.get("exact_api_spend_usd", 0.0))
        s_hal = float(rec.get("hal_saved_usd", 0.0))
        unshielded_rem_period_usd += s_spend * rem_frac
        rem_hal_period_usd += s_hal * (rem_frac ** 2)
        if s_hal <= 0.0 and s_spend > 0.0:
            s_unread, s_nbash = session_bloat_flags.get(cid, (0, 0))
            if s_unread > 0 or s_nbash > 0:
                bonus_bloat_period_usd += (s_spend * (rem_frac ** 2)) * min(0.28, 0.12 * (s_unread + s_nbash))

    dormant_mcp_monthly_usd = 0.0
    unshielded_turns = sum(max(1, int(r.get("planner_turns", 0))) for r in unshielded_recs)
    if dormant_mcp_tokens > 0 and t_rem > 0 and unshielded_recs and unshielded_turns > 0:
        eff_sess = len(unshielded_recs)
        dormant_tok_period = dormant_mcp_tokens * (eff_sess + max(0, unshielded_turns - eff_sess) * 0.10)
        dormant_mcp_monthly_usd = ((dormant_tok_period / 1_000_000.0) * max(0.0, float(in_price))) * scale
        dormant_mcp_monthly_usd = min(t_rem * 0.35, dormant_mcp_monthly_usd)

    raw_s_wbc = (
        ((rem_hal_period_usd + bonus_bloat_period_usd) * scale + dormant_mcp_monthly_usd)
        if unshielded_recs
        else 0.0
    )
    eff_t_rem_unshielded = (
        t_rem
        if already_shielded_sessions == 0
        else min(t_rem, round(unshielded_rem_period_usd * scale, 2))
    )
    max_wbc_by_rem = (
        math.floor(0.35 * eff_t_rem_unshielded * 100.0) / 100.0
        if eff_t_rem_unshielded > 0
        else 0.0
    )
    s_wbc_cap = min(max_wbc_by_rem, max(0.0, max_total_usd - s_code))
    s_wbc = round(min(s_wbc_cap, max(0.0, raw_s_wbc)), 2) if t_rem > 0 else 0.0
    s_total = round(s_code + s_wbc, 2)

    wbc_shield_rem_pct = round((s_wbc / t_rem) * 100.0, 1) if t_rem > 0 else 0.0
    combined_net_pct = round((s_total / baseline_monthly_usd) * 100.0, 1) if baseline_monthly_usd > 0 else 0.0
    recurring_patterns_count = len(distinct_patterns)

    dormant_cat = {
        "servers_count": len(dormant_mcps),
        "servers": dormant_mcps,
        "tokens_per_session": dormant_mcp_tokens,
        "recommendation": "body skills prune-mcp",
    }

    return {
        "enabled": True,
        "ci_mode": ci_mode,
        "observed_span_hours": observed_span_hours,
        "period_unit": period_unit,
        "baseline_period_usd": baseline_period_usd,
        "already_shielded_sessions": already_shielded_sessions,
        "baseline_monthly_usd": baseline_monthly_usd,
        "code_lowering_monthly_usd": s_code,
        "code_lowering_turn_pct": code_lowering_turn_pct,
        "remaining_monthly_usd": t_rem,
        "wbc_shield_monthly_usd": s_wbc,
        "wbc_shield_rem_pct": wbc_shield_rem_pct,
        "combined_monthly_usd": s_total,
        "combined_net_pct": combined_net_pct,
        "total_claude_turns": total_claude_turns,
        "lowerable_turns": total_lowerable_turns,
        "recurring_patterns_count": recurring_patterns_count,
        "tax_breakdown": {
            "llm_as_interpreter_tax": {
                "turns": total_lowerable_turns,
                "monthly_usd": s_code,
                "turn_pct": code_lowering_turn_pct,
                "categories": cat_stats,
            },
            "context_bloat_tax": {
                "monthly_usd": s_wbc,
                "remaining_pct": wbc_shield_rem_pct,
                "unbounded_read_calls": unbounded_read_calls,
                "noisy_bash_calls": noisy_bash_calls,
                "dormant_mcp_schemas": dormant_cat,
            },
        },
        "categories": {
            "git_pr_plumbing": cat_stats["git_pr_plumbing"],
            "data_log_wrangling": cat_stats["data_log_wrangling"],
            "lint_format_loops": cat_stats["lint_format_loops"],
            "dormant_mcp_schemas": dormant_cat,
        },
    }


def scan_claude_code_harness(
    claude_dir: Path,
    window_seconds: float,
    now: float,
    in_price_per_m: float,
    out_price_per_m: float,
    explicit_price_override: bool = False,
    include_benchmarks: bool = False,
    excluded_stats: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not claude_dir.is_dir():
        return records

    eff_inc_bench = _resolve_include_benchmarks(include_benchmarks)
    for jpath in claude_dir.rglob("*.jsonl"):
        if not _is_safe_contained_file(jpath, claude_dir):
            continue
        try:
            mtime = os.path.getmtime(jpath)
            if window_seconds > 0 and (now - mtime) > window_seconds:
                continue
            if not eff_inc_bench and _is_rel_transcript_benchmark(jpath, claude_dir):
                _bump_excluded_stat(excluded_stats, "claude_code", 1)
                continue
            conv_bytes = os.path.getsize(jpath)
            step_tuples: List[Tuple[bool, int, bool]] = []
            tool_counts: Dict[str, int] = {}
            first_prompt = ""
            active_model = ""
            m_turns = m_in = m_cc = m_cc_1h = m_cc_5m = m_cr = m_out = m_matched = typed_prompt_tok = 0
            fresh_usd = cache_usd = out_usd = 0.0
            first_event_ts = 0.0
            last_event_ts = 0.0
            wbc_hook_fired = False
            skip_benchmark_cwd = False
            model_usage_map: Dict[str, Dict[str, Any]] = {}
            seen_usage_by_msg_key: Dict[Tuple[str, str], Tuple[Dict[str, Any], str]] = {}
            turn_cost_by_key: Dict[str, float] = {}
            step_idx_by_msg_key: Dict[Tuple[str, str], int] = {}
            unmetered_p_bytes_by_model: Dict[str, List[int]] = {}
            ordered_tools: List[Dict[str, Any]] = []
            seen_tool_use_ids: set[str] = set()
            tool_entry_by_id: Dict[str, Dict[str, Any]] = {}
            turn0_declared_mcps: List[str] = []
            turn0_declared_set: set[str] = set()
            unbounded_read_calls = 0
            noisy_bash_calls = 0
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
                    if not eff_inc_bench and line_idx <= 12 and obj.get("cwd"):
                        if _is_rel_transcript_benchmark(jpath, claude_dir, cwd_hint=str(obj.get("cwd"))):
                            skip_benchmark_cwd = True
                            break
                    msg_dict = obj.get("message") if isinstance(obj.get("message"), dict) else {}
                    ev_ts = _parse_iso_or_epoch_ts(
                        obj.get("timestamp") or obj.get("created_at") or msg_dict.get("created_at")
                    )
                    if ev_ts > 0.0:
                        if first_event_ts <= 0.0 or ev_ts < first_event_ts:
                            first_event_ts = ev_ts
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
                        elif role in ("system", "progress") and (
                            "body-immune-hook" in s
                            or any(mk in s for mk in _WBC_TRANSCRIPT_HOOK_MARKERS)
                        ):
                            wbc_hook_fired = True
                    # Capture Turn-0 declared MCP servers from system/init records if present
                    for t_list_key in ("tools", "mcp_servers"):
                        t_list = obj.get(t_list_key)
                        if isinstance(t_list, list):
                            for entry in t_list:
                                if isinstance(entry, str):
                                    if entry.startswith("mcp__"):
                                        parts = entry.split("__", 2)
                                        if len(parts) >= 2 and parts[1] and parts[1] not in turn0_declared_set:
                                            turn0_declared_set.add(parts[1])
                                            turn0_declared_mcps.append(parts[1])
                                    elif t_list_key == "mcp_servers" and entry.strip() and entry.strip() not in turn0_declared_set:
                                        turn0_declared_set.add(entry.strip())
                                        turn0_declared_mcps.append(entry.strip())
                                elif isinstance(entry, dict):
                                    ename = str(entry.get("name") or "")
                                    if t_list_key == "mcp_servers" and ename and ename not in turn0_declared_set:
                                        turn0_declared_set.add(ename)
                                        turn0_declared_mcps.append(ename)
                                    elif ename.startswith("mcp__"):
                                        parts = ename.split("__", 2)
                                        if len(parts) >= 2 and parts[1] and parts[1] not in turn0_declared_set:
                                            turn0_declared_set.add(parts[1])
                                            turn0_declared_mcps.append(parts[1])
                    raw_model = str(msg_dict.get("model") or obj.get("model") or "")
                    if raw_model and raw_model != "<synthetic>":
                        active_model = raw_model
                    is_p = role in ("assistant", "model", "planner_response")
                    msg_id = str(msg_dict.get("id") or obj.get("messageId") or "").strip()
                    req_id = str(obj.get("requestId") or msg_dict.get("requestId") or "").strip()
                    dedup_key = (msg_id, req_id) if (msg_id or req_id) else (f"_line_{line_idx}", "")
                    turn_key_str = f"{dedup_key[0]}|{dedup_key[1]}"
                    msg_content = msg_dict.get("content") if "content" in msg_dict else obj.get("content")
                    if not wbc_hook_fired and role in ("assistant", "model", "planner_response", "system", "progress"):
                        if isinstance(msg_content, str) and any(mk in msg_content for mk in _WBC_TRANSCRIPT_HOOK_MARKERS):
                            wbc_hook_fired = True
                    if isinstance(msg_content, list):
                        for blk in msg_content:
                            if not isinstance(blk, dict):
                                continue
                            btype = blk.get("type")
                            if not wbc_hook_fired and role in ("assistant", "model", "planner_response", "system", "progress"):
                                blk_text = str(blk.get("text") or blk.get("thinking") or blk.get("content") or "")
                                if blk_text and any(mk in blk_text for mk in _WBC_TRANSCRIPT_HOOK_MARKERS):
                                    wbc_hook_fired = True
                            if btype == "tool_use":
                                tname = str(blk.get("name") or "tool")
                                tid = str(blk.get("id") or "").strip()
                                tool_counts[tname] = tool_counts.get(tname, 0) + 1
                                if tid and tid in seen_tool_use_ids:
                                    continue
                                if tid:
                                    seen_tool_use_ids.add(tid)
                                norm_name = tname.strip().lower()
                                inp = blk.get("input") if isinstance(blk.get("input"), dict) else {}
                                cmd_str = str(inp.get("command") or "") if norm_name == "bash" else ""
                                cat_lbl = None
                                verb_lbl = ""
                                is_noisy_cmd = False
                                if norm_name == "bash" and cmd_str:
                                    cat_lbl, verb_lbl, is_noisy_cmd = _classify_bash_command(cmd_str)
                                has_read_bounds = bool(inp.get("limit") or inp.get("offset") or inp.get("lines"))
                                is_unbounded_read = False
                                if is_noisy_cmd:
                                    noisy_bash_calls += 1
                                mcp_srv = ""
                                if norm_name.startswith("mcp__"):
                                    mparts = tname.split("__", 2)
                                    if len(mparts) >= 2:
                                        mcp_srv = mparts[1]
                                t_entry = {
                                    "tool": tname,
                                    "norm": norm_name,
                                    "cmd": _redact_audit_secrets(cmd_str)[:200] if cmd_str else "",
                                    "cat": cat_lbl,
                                    "verb": verb_lbl,
                                    "turn_key": turn_key_str,
                                    "has_read_bounds": has_read_bounds,
                                    "unbounded_read": is_unbounded_read,
                                    "noisy_bash": is_noisy_cmd,
                                    "mcp_server": mcp_srv,
                                }
                                ordered_tools.append(t_entry)
                                if tid:
                                    tool_entry_by_id[tid] = t_entry
                            elif btype == "tool_result":
                                tr_id = str(blk.get("tool_use_id") or "").strip()
                                tr_content = blk.get("content")
                                tr_str = (
                                    tr_content
                                    if isinstance(tr_content, str)
                                    else (json.dumps(tr_content or "", ensure_ascii=False) if tr_content is not None else "")
                                )
                                if not wbc_hook_fired and tr_str and any(mk in tr_str for mk in _WBC_TRANSCRIPT_HOOK_MARKERS):
                                    wbc_hook_fired = True
                                tr_entry = tool_entry_by_id.get(tr_id)
                                if tr_entry is not None:
                                    tr_bytes = len(tr_str.encode("utf-8", errors="ignore"))
                                    tr_lines = len(tr_str.splitlines()) if tr_str else 0
                                    if tr_entry["norm"] in ("read", "notebookread"):
                                        if not tr_entry.get("has_read_bounds") and (tr_bytes > 2000 or tr_lines > 50):
                                            if not tr_entry["unbounded_read"]:
                                                tr_entry["unbounded_read"] = True
                                                unbounded_read_calls += 1
                                    elif tr_entry["norm"] == "bash" and tr_bytes > 2000 and not tr_entry["noisy_bash"]:
                                        tr_entry["noisy_bash"] = True
                                        noisy_bash_calls += 1
                    b_len = len(s.encode("utf-8"))
                    u_dict = msg_dict.get("usage") if isinstance(msg_dict.get("usage"), dict) else None
                    if not isinstance(u_dict, dict):
                        u_dict = obj.get("usage") if isinstance(obj.get("usage"), dict) else None
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

            if skip_benchmark_cwd:
                _bump_excluded_stat(excluded_stats, "claude_code", 1)
                continue

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
                tkey_str = f"{_dkey[0]}|{_dkey[1]}"
                turn_cost_by_key[tkey_str] = turn_cost_by_key.get(tkey_str, 0.0) + (t_fresh + t_cache + t_out_u)
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
            rec["_lowering_trace"] = {
                "ordered_tools": ordered_tools,
                "turn0_declared_mcps": turn0_declared_mcps,
                "unbounded_read_calls": unbounded_read_calls,
                "noisy_bash_calls": noisy_bash_calls,
                "turn_cost_by_key": turn_cost_by_key,
                "min_event_ts": first_event_ts,
                "max_event_ts": last_event_ts,
            }
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
    include_benchmarks: bool = False,
    excluded_stats: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not goose_db.is_file():
        return records
    eff_inc_bench = _resolve_include_benchmarks(include_benchmarks)

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
                    if not eff_inc_bench and _is_benchmark_path_or_cwd(sid):
                        _bump_excluded_stat(excluded_stats, "goose", 1)
                        continue
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
    include_benchmarks: bool = False,
    excluded_stats: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Scan modern Cursor `~/.cursor/projects/**/agent-transcripts/**/*.jsonl` transcripts."""
    records: List[Dict[str, Any]] = []
    if not projects_root.exists():
        return records
    eff_inc_bench = _resolve_include_benchmarks(include_benchmarks)
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
            if not eff_inc_bench and _is_rel_transcript_benchmark(jpath, projects_root):
                _bump_excluded_stat(excluded_stats, "cursor", 1)
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
    include_benchmarks: bool = False,
    excluded_stats: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if now <= 0.0:
        now = time.time()
    eff_inc_bench = _resolve_include_benchmarks(include_benchmarks)

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
                if not eff_inc_bench:
                    ws_json_p = db_p.parent / "workspace.json"
                    ws_folder = ""
                    if ws_json_p.is_file():
                        try:
                            ws_folder = str(json.loads(ws_json_p.read_text(encoding="utf-8", errors="ignore")).get("folder") or "")
                        except Exception:
                            ws_folder = ""
                    if _is_rel_transcript_benchmark(db_p, root, cwd_hint=ws_folder):
                        _bump_excluded_stat(excluded_stats, "cursor", 1)
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
            _scan_cursor_agent_transcripts_jsonl(
                tr,
                window_seconds,
                now,
                in_price_per_m,
                out_price_per_m,
                include_benchmarks=eff_inc_bench,
                excluded_stats=excluded_stats,
            )
        )

    return _dedup_records(records)


def scan_gemini_cli_harness(
    gemini_cli_dir: Optional[Any] = None,
    window_seconds: float = 0.0,
    now: float = 0.0,
    in_price_per_m: float = 5.0,
    out_price_per_m: float = 25.0,
    include_benchmarks: bool = False,
    excluded_stats: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Scan Google Gemini CLI (`~/.gemini/tmp/*/chats/session-*.{json,jsonl}`) sessions."""
    records: List[Dict[str, Any]] = []
    if now <= 0.0:
        now = time.time()
    eff_inc_bench = _resolve_include_benchmarks(include_benchmarks)

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
            g_root = next((r for r in roots if r.is_dir() and str(spath).startswith(str(r))), None)
            if not eff_inc_bench and _is_rel_transcript_benchmark(spath, g_root or spath):
                _bump_excluded_stat(excluded_stats, "gemini_cli", 1)
                continue
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
    include_benchmarks: bool = False,
    excluded_stats: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if now <= 0.0:
        now = time.time()
    eff_inc_bench = _resolve_include_benchmarks(include_benchmarks)

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
            codex_root = next((c for c in candidates if c.is_dir() and str(jpath).startswith(str(c))), None)
            if not eff_inc_bench and _is_rel_transcript_benchmark(jpath, codex_root or jpath):
                _bump_excluded_stat(excluded_stats, "codex", 1)
                continue
            conv_bytes = os.path.getsize(jpath)
            step_tuples: List[Tuple[bool, int, bool]] = []
            tool_counts: Dict[str, int] = {}
            first_prompt = ""
            active_model = ""
            last_event_ts = 0.0
            skip_codex_bench = False
            m_turns = m_in = m_cc = m_cr = m_out = m_matched = typed_prompt_tok = 0
            fresh_usd = cache_usd = out_usd = 0.0
            prev_tot_tuple: Optional[Tuple[int, int, int, int]] = None
            model_usage_map: Dict[str, Dict[str, Any]] = {}
            codex_ln_idx = 0
            with open(jpath, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    codex_ln_idx += 1
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
                    if not eff_inc_bench and codex_ln_idx <= 10:
                        c_cwd = str(obj.get("cwd") or pay_obj.get("cwd") or "").strip()
                        if c_cwd and _is_rel_transcript_benchmark(jpath, codex_root or jpath, cwd_hint=c_cwd):
                            skip_codex_bench = True
                            break
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
            if skip_codex_bench:
                _bump_excluded_stat(excluded_stats, "codex", 1)
                continue
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
    include_benchmarks: bool = False,
    excluded_stats: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if now <= 0.0:
        now = time.time()
    eff_inc_bench = _resolve_include_benchmarks(include_benchmarks)

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
            cl_root = next((r for r in roots if r.is_dir() and str(jpath).startswith(str(r))), None)
            if not eff_inc_bench and _is_rel_transcript_benchmark(jpath, cl_root or jpath):
                _bump_excluded_stat(excluded_stats, "cline_roo", 1)
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
    include_benchmarks: bool = False,
    excluded_stats: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if now <= 0.0:
        now = time.time()
    eff_inc_bench = _resolve_include_benchmarks(include_benchmarks)

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
                if not eff_inc_bench:
                    ws_json_p = db_p.parent / "workspace.json"
                    ws_folder = ""
                    if ws_json_p.is_file():
                        try:
                            ws_folder = str(json.loads(ws_json_p.read_text(encoding="utf-8", errors="ignore")).get("folder") or "")
                        except Exception:
                            ws_folder = ""
                    if _is_rel_transcript_benchmark(db_p, root, cwd_hint=ws_folder):
                        _bump_excluded_stat(excluded_stats, "windsurf", 1)
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
    include_benchmarks: bool = False,
    excluded_stats: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if now <= 0.0:
        now = time.time()
    eff_inc_bench = _resolve_include_benchmarks(include_benchmarks)

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
            a_root = (
                next((c for c in candidates if c.is_dir() and str(fpath).startswith(str(c))), fpath)
                if (aider_path is not None or os.environ.get("AIDER_HISTORY_PATH"))
                else None
            )
            if not eff_inc_bench and _is_rel_transcript_benchmark(fpath, a_root):
                _bump_excluded_stat(excluded_stats, "aider", 1)
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
    include_benchmarks: bool = False,
    excluded_stats: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if now <= 0.0:
        now = time.time()
    eff_inc_bench = _resolve_include_benchmarks(include_benchmarks)

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
            if not eff_inc_bench and _is_rel_transcript_benchmark(jpath, target_dir):
                _bump_excluded_stat(excluded_stats, "continue_dev", 1)
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
    include_benchmarks: bool = False,
    excluded_stats: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if now <= 0.0:
        now = time.time()
    eff_inc_bench = _resolve_include_benchmarks(include_benchmarks)

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
            z_root = next((r for r in roots if r.is_dir() and str(fpath).startswith(str(r))), fpath)
            if not eff_inc_bench and _is_rel_transcript_benchmark(fpath, z_root):
                _bump_excluded_stat(excluded_stats, "zed", 1)
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


def _parse_turn0_rule_frontmatter(raw: str) -> Tuple[bool, str, str]:
    """Parse YAML frontmatter on rule files -> (is_conditional, description, normalized_body)."""
    s = raw.lstrip()
    if not s.startswith("---"):
        return False, "", re.sub(r"\s+", " ", s.strip())
    parts = s.split("---", 2)
    if len(parts) < 3:
        return False, "", re.sub(r"\s+", " ", s.strip())
    fm = parts[1]
    body = parts[2]
    trigger_val = ""
    always_apply_val = ""
    desc_parts: List[str] = []
    in_desc = False
    for line in fm.splitlines():
        if not line.strip():
            continue
        indented = line.startswith((" ", "\t"))
        ls = line.strip()
        low = ls.lower()
        if not indented and low.startswith("trigger:"):
            in_desc = False
            trigger_val = ls.split(":", 1)[1].strip().strip("'\"").lower()
        elif not indented and low.startswith("alwaysapply:"):
            in_desc = False
            always_apply_val = ls.split(":", 1)[1].strip().strip("'\"").lower()
        elif not indented and low.startswith("description:"):
            in_desc = True
            dval = ls.split(":", 1)[1].strip().strip("'\"")
            if dval and dval not in ("|", ">", "|-", ">-"):
                desc_parts.append(dval)
        elif in_desc and (indented or ":" not in ls):
            desc_parts.append(ls)
        else:
            in_desc = False
    is_cond = trigger_val in ("model_decision", "manual", "glob", "on_demand") or always_apply_val in (
        "false",
        "no",
        "0",
    )
    return is_cond, " ".join(desc_parts).strip(), re.sub(r"\s+", " ", body.strip())


def _scan_turn0_installed_skills(
    ws_dir: Path, h_dir: Path, strict_containment: bool = False
) -> Tuple[List[Dict[str, Any]], int]:
    """Scan installed SKILL.md frontmatter across workspace, ~/.agents, enabled plugins, and builtin."""
    disabled_plugins: set[str] = set()
    cfg_path = h_dir / ".gemini" / "config" / "config.json"
    if cfg_path.is_file():
        try:
            cfg_data = json.loads(cfg_path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(cfg_data, dict):
                pmap = cfg_data.get("plugins")
                if isinstance(pmap, dict):
                    for pk, pv in pmap.items():
                        if isinstance(pv, dict) and pv.get("enabled") is False:
                            disabled_plugins.add(str(pk).split("/")[-1])
                for dkey in ("disabledPlugins", "disabled_plugins"):
                    dlist = cfg_data.get(dkey)
                    if isinstance(dlist, list):
                        for item in dlist:
                            if isinstance(item, str):
                                disabled_plugins.add(item.split("/")[-1])
        except Exception:
            pass

    skill_roots: List[Tuple[Path, str, Path]] = [
        (ws_dir / ".agents" / "skills", "workspace", ws_dir),
        (h_dir / ".agents" / "skills", "user_global", h_dir),
    ]
    plugins_dir = h_dir / ".gemini" / "config" / "plugins"
    if plugins_dir.is_dir():
        try:
            for p_dir in sorted(plugins_dir.iterdir())[:48]:
                if not p_dir.is_dir() or p_dir.name.startswith(".") or p_dir.name in disabled_plugins:
                    continue
                p_json = p_dir / "plugin.json"
                if p_json.is_file():
                    try:
                        p_meta = json.loads(p_json.read_text(encoding="utf-8", errors="ignore"))
                        if isinstance(p_meta, dict) and p_meta.get("disabled") is True:
                            continue
                    except Exception:
                        pass
                if (p_dir / "skills").is_dir():
                    skill_roots.append((p_dir / "skills", "plugin", h_dir))
        except OSError:
            pass
    skill_roots.append((h_dir / ".gemini" / "antigravity" / "builtin" / "skills", "builtin", h_dir))

    seen_skill_rp: set[str] = set()
    skills_list: List[Dict[str, Any]] = []
    total_skill_tok = 0

    for s_root, s_scope, contain_root in skill_roots:
        if not s_root.is_dir():
            continue
        try:
            for s_dir in sorted(s_root.iterdir())[:96]:
                if not s_dir.is_dir() or s_dir.name.startswith("."):
                    continue
                s_md = s_dir / "SKILL.md"
                if strict_containment:
                    if not _is_safe_contained_file(s_md, contain_root):
                        continue
                elif not s_md.is_file():
                    continue
                try:
                    rp = os.path.realpath(s_md)
                except OSError:
                    continue
                if rp in seen_skill_rp:
                    continue
                seen_skill_rp.add(rp)
                try:
                    with open(s_md, "r", encoding="utf-8", errors="ignore") as fh:
                        head = fh.read(4096)
                except OSError:
                    continue
                fm_name = s_dir.name
                desc_parts: List[str] = []
                in_desc = False
                hs = head.lstrip()
                if hs.startswith("---"):
                    fparts = hs.split("---", 2)
                    if len(fparts) >= 3:
                        for fln in fparts[1].splitlines():
                            if not fln.strip():
                                continue
                            ind = fln.startswith((" ", "\t"))
                            fs = fln.strip()
                            flow = fs.lower()
                            if not ind and flow.startswith("name:"):
                                in_desc = False
                                nval = fs.split(":", 1)[1].strip().strip("'\"")
                                if nval:
                                    fm_name = nval
                            elif not ind and flow.startswith("description:"):
                                in_desc = True
                                dval = fs.split(":", 1)[1].strip().strip("'\"")
                                if dval and dval not in ("|", ">", "|-", ">-"):
                                    desc_parts.append(dval)
                            elif in_desc and (ind or ":" not in fs):
                                desc_parts.append(fs)
                            else:
                                in_desc = False
                desc_str = " ".join(desc_parts).strip()
                safe_name = _sanitize_turn0_label(fm_name or s_dir.name)
                s_tok = max(12, (len(fm_name) + len(desc_str) + 48) // 4)
                total_skill_tok += s_tok
                skills_list.append(
                    {
                        "name": safe_name,
                        "scope": s_scope,
                        "tokens": s_tok,
                    }
                )
        except OSError:
            continue

    skills_list.sort(key=lambda x: int(x.get("tokens", 0)), reverse=True)
    return skills_list, total_skill_tok


def audit_turn0_static_overhead(
    workspace_dir: Optional[Path] = None,
    home_dir: Optional[Path] = None,
    sessions_count: int = 0,
    in_price_per_m: float = 5.0,
    strict_containment: bool = False,
) -> Dict[str, Any]:
    """
    Blood-Brain Barrier - Turn-0 Fixed Context Tax Audit (<50ms, 100% stdlib, 0 PII leaks):
    Scans static instruction files (`CLAUDE.md`, `AGENTS.md`, `GEMINI.md`, `.cursorrules`, etc.),
    conditional rules, installed `<skills>` index descriptions, and MCP server schemas
    vs `0` MCP schema tokens for `body` OS hooks.
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
    seen_rule_entries: Dict[Tuple[str, str], Dict[str, Any]] = {}
    instruction_files: List[Dict[str, Any]] = []
    duplicate_instruction_files: List[Dict[str, Any]] = []
    instruction_bytes = 0
    instruction_tokens = 0

    for cand_p, scope in instruction_candidates:
        try:
            if strict_containment:
                cand_root = ws_dir if scope == "workspace" else h_dir
                if not _is_safe_contained_file(cand_p, cand_root):
                    continue
            elif not cand_p.is_file():
                continue
            rp = os.path.realpath(cand_p)
            if rp in seen_realpaths:
                continue
            seen_realpaths.add(rp)
            raw = cand_p.read_text(encoding="utf-8", errors="ignore")
            if not raw.strip():
                continue
            b_len = len(raw.encode("utf-8"))
            full_tok = _estimate_static_context_tokens(raw, is_json=False)
            is_cond, cond_desc, norm_body = _parse_turn0_rule_frontmatter(raw)
            active_tok = max(12, len(cond_desc) // 4 + 12) if is_cond else full_tok
            rel_p = _format_turn0_rel_path(cand_p, ws_dir, h_dir)
            is_rule_dir_file = cand_p.parent.name == "rules"
            if is_rule_dir_file:
                dup_key = (cand_p.name.lower(), norm_body)
                if dup_key in seen_rule_entries:
                    canon = seen_rule_entries[dup_key]
                    canon.setdefault("duplicate_paths", []).append(rel_p)
                    duplicate_instruction_files.append(
                        {
                            "path": rel_p,
                            "scope": scope,
                            "duplicate_of": canon["path"],
                            "bytes": b_len,
                            "tokens": active_tok,
                            "full_tokens": full_tok,
                            "conditional": is_cond,
                        }
                    )
                    continue
            entry_dict: Dict[str, Any] = {
                "path": rel_p,
                "scope": scope,
                "bytes": b_len,
                "tokens": active_tok,
                "full_tokens": full_tok,
                "conditional": is_cond,
            }
            if is_rule_dir_file:
                seen_rule_entries[(cand_p.name.lower(), norm_body)] = entry_dict
            instruction_bytes += b_len
            instruction_tokens += active_tok
            instruction_files.append(entry_dict)
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
                if strict_containment and srv_dir.is_symlink():
                    continue
                s_name = _sanitize_turn0_label(srv_dir.name)
                s_files = 0
                s_toks = 0
                for jf in sorted(srv_dir.glob("*.json"))[:64]:
                    if strict_containment:
                        if not _is_safe_contained_file(jf, schema_root):
                            continue
                    elif not jf.is_file():
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
        h_dir / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",
        h_dir / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json",
        h_dir / ".config" / "Claude" / "claude_desktop_config.json",
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
            if strict_containment:
                if not (_is_safe_contained_file(cfg_p, ws_dir) or _is_safe_contained_file(cfg_p, h_dir)):
                    continue
            elif not cfg_p.is_file():
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

    skills_list, skills_turn0_tokens = _scan_turn0_installed_skills(
        ws_dir=ws_dir, h_dir=h_dir, strict_containment=strict_containment
    )

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
    total_turn0_per_session = instruction_tokens + skills_turn0_tokens + mcp_turn0_tokens

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
    skills_summary_part = (
        f"{skills_turn0_tokens:,} `<skills>` tok across {len(skills_list)} skills + "
        if skills_list
        else ""
    )
    summary = (
        f"{total_turn0_per_session:,} tok/session "
        f"({instruction_tokens:,} instruction tok across {len(instruction_files)} files + "
        f"{skills_summary_part}"
        f"{mcp_turn0_tokens:,} MCP schema tok across {len(mcp_servers_list)} servers vs 0 MCP tok for `body` OS hooks)"
    )

    return {
        "scan_latency_ms": scan_ms,
        "status_tier": status_tier,
        "instruction_files_count": len(instruction_files),
        "instruction_bytes": instruction_bytes,
        "instruction_tokens": instruction_tokens,
        "instruction_files": instruction_files,
        "conditional_rules_count": sum(1 for f in instruction_files if f.get("conditional")),
        "duplicate_rules_count": len(duplicate_instruction_files),
        "duplicate_instruction_files": duplicate_instruction_files,
        "installed_skills_count": len(skills_list),
        "skills_turn0_tokens": skills_turn0_tokens,
        "skills": skills_list[:64],
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


_AUDIT_REPO_ROOT_RESOLVED = Path(__file__).resolve().parents[1]
_AUDIT_FILE_READ_TOOLS_AGY = frozenset({"read_file", "view_file"})
_AUDIT_FILE_READ_TOOLS_CC = frozenset({"read", "notebookread"})
_AUDIT_FILE_READ_TOOLS_CURSOR = frozenset({"beforereadfile", "pre_read_code"})
_AUDIT_FILE_READ_TOOLS_CODEX = frozenset({"developer__text_editor"})


def _normalize_audit_target_label(target: str, home_dir: Optional[Path] = None) -> str:
    """CWD-independent stdlib target normalization mirroring `shield._normalize_target_label`."""
    raw = str(target or "").strip()
    if not raw:
        return "unknown"
    if raw.startswith("foveal:") or raw.endswith(":actuate"):
        return raw[:140]
    if "/" in raw or "\\" in raw:
        norm = raw.replace("\\", "/")
        h_res = (home_dir or Path.home()).resolve()
        h_norm = str(h_res).replace("\\", "/").lstrip("/")
        p_res = (h_res / "Projects").resolve()
        if h_norm and norm.startswith(h_norm + "/"):
            raw = "/" + norm
            norm = raw
        elif norm.startswith("Projects/") and len(norm.split("/")) >= 3:
            raw = str(h_res / norm)
            norm = raw.replace("\\", "/")
        elif " " in norm and not norm.startswith(("/", "~", "./", "../")):
            if not norm.split(" ", 1)[0].endswith((".py", ".ts", ".js", ".md", ".json", ".jsonl", ".log")):
                return raw[:140]
        if not norm.startswith(("/", "~")):
            while norm.startswith("./"):
                norm = norm[2:]
            return (norm or raw)[:140]
        try:
            if norm.startswith("~/"):
                p = (h_res / norm[2:]).resolve()
            else:
                p = Path(os.path.expanduser(norm)).resolve()
            try:
                return str(p.relative_to(_AUDIT_REPO_ROOT_RESOLVED)).replace("\\", "/")[:140]
            except ValueError:
                pass
            try:
                rel_proj = p.relative_to(p_res)
                if len(rel_proj.parts) >= 2:
                    return str(rel_proj).replace("\\", "/")[:140]
            except ValueError:
                pass
            return (p.name or raw)[:140]
        except Exception:
            return (Path(norm).name or raw)[:140]
    return raw[:140]


def _audit_harness_read_ceiling(tool_name: str) -> Optional[int]:
    """Return native IDE read ceiling (tokens) for file-read tools, or None for non-read tools."""
    tl = str(tool_name or "").strip().lower()
    if tl in _AUDIT_FILE_READ_TOOLS_AGY:
        return 11_520
    if tl in _AUDIT_FILE_READ_TOOLS_CC:
        return 25_000
    if tl in _AUDIT_FILE_READ_TOOLS_CURSOR:
        return 15_000
    if tl in _AUDIT_FILE_READ_TOOLS_CODEX:
        return 20_000
    return None


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
            "measured_active_slices": 0,
            "measured_voluntary_bounded_reads": 0,
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
    bare_seen_credits: Dict[Tuple[str, str], int] = {}
    immune_conv_ids: set[str] = set()
    active_slices = 0
    voluntary_bounded_reads = 0
    direct_tok_saved = 0
    hook_events = 0
    explicit_audit_env = bool(os.environ.get("WBC_IMMUNE_AUDIT_PATH"))

    audit_paths: List[Path] = []
    if explicit_audit_env:
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

                    dec_low = str(ev.get("decision") or "").strip().lower()
                    is_active_slice = bool(ev.get("read_auto_sliced") or ev.get("post_output_sliced"))
                    is_active_interception = bool(
                        is_active_slice
                        or ev.get("context_gate_triggered")
                        or ev.get("auto_healed")
                        or dec_low in ("deny", "ask", "force_ask")
                    )
                    if is_active_slice:
                        active_slices += 1
                    elif ev.get("bounded_slice_read"):
                        voluntary_bounded_reads += 1

                    if not is_active_interception:
                        continue

                    raw_sav = max(0, int(ev.get("direct_tokens_saved") or 0))
                    if raw_sav > 0:
                        raw_tgt = str(ev.get("target") or "").strip()
                        norm_raw = raw_tgt.replace("\\", "/")
                        if not explicit_audit_env and norm_raw.startswith(
                            ("/tmp/", "/private/tmp/", "/var/folders/", "/private/var/folders/")
                        ):
                            continue
                        read_ceil = _audit_harness_read_ceiling(str(ev.get("tool") or ""))
                        if read_ceil is not None:
                            raw_sav = min(read_ceil, raw_sav)
                        norm_tgt = _normalize_audit_target_label(raw_tgt, home_dir=home_dir) if raw_tgt else ""
                        if norm_tgt and norm_tgt != "unknown":
                            sess_k = cid or "global"
                            dk = (sess_k, norm_tgt)
                            b_only = Path(norm_tgt).name if "/" in norm_tgt else norm_tgt
                            bk = (sess_k, b_only)
                            if "/" not in norm_tgt:
                                prev_c = max(dedup_file_credits.get(dk, 0), bare_seen_credits.get(bk, 0))
                            else:
                                prev_c = max(dedup_file_credits.get(dk, 0), dedup_file_credits.get(bk, 0))
                            eff_s = max(0, max(prev_c, raw_sav) - prev_c)
                            new_hwm = prev_c + eff_s
                            dedup_file_credits[dk] = new_hwm
                            bare_seen_credits[bk] = max(bare_seen_credits.get(bk, 0), new_hwm)
                            if "/" in norm_tgt and bk in dedup_file_credits:
                                dedup_file_credits[bk] = max(dedup_file_credits[bk], new_hwm)
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
        "measured_bounded_slices": active_slices,
        "measured_active_slices": active_slices,
        "measured_voluntary_bounded_reads": voluntary_bounded_reads,
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


def discover_standalone_cloud_exports(
    downloads_dir: Optional[Path] = None,
    max_age_days: int = 14,
) -> List[Path]:
    """Opt-in helper (`--auto-downloads` / `--import auto`): find recent Claude `.zip`/`conversations.json` & Console `.csv` files."""
    import zipfile as _zf

    env_dl = os.environ.get("WBC_DOWNLOADS_DIR", "").strip()
    dl_root = Path(downloads_dir or env_dl or (Path.home() / "Downloads")).expanduser()
    if not dl_root.is_dir():
        return []

    now_ts = time.time()
    max_age_sec = max(1, int(max_age_days)) * 86400.0
    max_bytes = 256 * 1024 * 1024
    candidates: List[Tuple[float, Path]] = []

    try:
        entries = list(dl_root.iterdir())
    except OSError:
        return []

    for entry in entries:
        try:
            if not entry.is_file() or entry.name.startswith("."):
                continue
            st = entry.stat()
            if st.st_size <= 0 or st.st_size > max_bytes or (now_ts - st.st_mtime) > max_age_sec:
                continue
            low_name = entry.name.lower()
            suf = entry.suffix.lower()
            if low_name == "conversations.json":
                candidates.append((st.st_mtime, entry))
            elif suf == ".zip":
                try:
                    with _zf.ZipFile(entry, "r") as zf:
                        if any(
                            (n == "conversations.json" or n.endswith("/conversations.json"))
                            and ".." not in Path(n).parts
                            for n in zf.namelist()[:200]
                        ):
                            candidates.append((st.st_mtime, entry))
                except Exception:
                    continue
            elif suf == ".csv" and st.st_size <= 32 * 1024 * 1024:
                with open(entry, "r", encoding="utf-8", errors="ignore") as fh:
                    hdr = fh.readline(4096).lower()
                cols = {c.strip().strip('"') for c in hdr.split(",") if c.strip()}
                has_tok = bool(
                    cols & {"input_tokens", "uncached_input_tokens", "cache_read_input_tokens", "prompt_tokens"}
                ) and bool(
                    cols & {"output_tokens", "completion_tokens", "cost_usd", "total_cost_usd", "amount_usd", "cost", "amount"}
                )
                has_console = (
                    any(k in low_name for k in ("anthropic", "claude"))
                    and bool(cols & {"cost_usd", "total_cost_usd", "amount_usd", "cost", "amount", "usd"})
                    and bool(cols & {"date", "usage_date", "timestamp", "created_at", "bucket_start", "day", "model", "workspace"})
                )
                if has_tok or has_console:
                    candidates.append((st.st_mtime, entry))
        except OSError:
            continue

    candidates.sort(key=lambda x: x[0], reverse=True)
    picked: List[Path] = []
    seen_kinds: set[str] = set()
    for _mt, p in candidates:
        kind = "csv" if p.suffix.lower() == ".csv" else "claude_chat"
        if kind not in seen_kinds:
            seen_kinds.add(kind)
            picked.append(p)
    return picked


def scan_claude_export_records(
    import_paths: List[Any],
    window_seconds: float,
    now: float,
    in_price_per_m: float,
    out_price_per_m: float,
    explicit_price_override: bool = False,
) -> List[Dict[str, Any]]:
    """Convert imported Claude.ai `conversations.json` / `.zip` threads into full FinOps & CSI session records."""
    import zipfile as _zf

    records: List[Dict[str, Any]] = []
    max_bytes = 256 * 1024 * 1024
    seen_conv_ids: set[str] = set()

    for raw_path in import_paths or []:
        try:
            p = Path(str(raw_path)).expanduser().resolve()
            if not p.is_file():
                continue
            suf = p.suffix.lower()
            if suf == ".csv":
                continue
            if p.stat().st_size > max_bytes:
                continue

            if suf == ".zip":
                with _zf.ZipFile(p, "r") as zf:
                    member = next(
                        (
                            n
                            for n in zf.namelist()
                            if (n == "conversations.json" or n.endswith("/conversations.json"))
                            and ".." not in Path(n).parts
                        ),
                        None,
                    )
                    if not member or zf.getinfo(member).file_size > max_bytes:
                        continue
                    with zf.open(member, "r") as mfp:
                        raw_bytes = mfp.read(max_bytes + 1)
                    if len(raw_bytes) > max_bytes:
                        continue
                    payload = json.loads(raw_bytes.decode("utf-8", errors="ignore"))
            else:
                payload = json.loads(p.read_text(encoding="utf-8", errors="ignore"))

            convs = (
                payload
                if isinstance(payload, list)
                else (payload.get("conversations", []) if isinstance(payload, dict) else [])
            )
            file_mtime = os.path.getmtime(p)

            for idx_c, conv in enumerate(convs):
                if not isinstance(conv, dict):
                    continue
                conv_ts = _parse_iso_or_epoch_ts(conv.get("updated_at") or conv.get("created_at"))
                eff_mtime = conv_ts if conv_ts > 0 else file_mtime
                if window_seconds > 0 and conv_ts > 0 and (now - conv_ts) > window_seconds:
                    continue

                msgs = conv.get("chat_messages") or conv.get("messages") or []
                if not isinstance(msgs, list) or not msgs:
                    continue

                cid = str(conv.get("uuid") or conv.get("id") or f"claude-chat-{idx_c}").strip()
                if cid in seen_conv_ids:
                    continue
                seen_conv_ids.add(cid)

                step_tuples: List[Tuple[bool, int, bool]] = []
                tool_counts: Dict[str, int] = {}
                conv_bytes = 0
                first_prompt = str(conv.get("name") or conv.get("title") or "").strip()
                conv_model = str(conv.get("model") or "")
                model_usage_map: Dict[str, Dict[str, Any]] = {}

                for msg in msgs:
                    if not isinstance(msg, dict):
                        continue
                    sender = str(msg.get("sender") or msg.get("role") or "").lower()
                    is_p = sender in ("assistant", "model", "ai")

                    chunks: List[str] = []
                    base_txt = str(msg.get("text") or "")
                    if base_txt:
                        chunks.append(base_txt)
                    c_list = msg.get("content")
                    if isinstance(c_list, list):
                        for blk in c_list:
                            if not isinstance(blk, dict):
                                continue
                            btype = str(blk.get("type") or "").lower()
                            if btype in ("tool_use", "mcp_tool_use"):
                                tname = str(blk.get("name") or "mcp_tool")
                                tool_counts[tname] = tool_counts.get(tname, 0) + 1
                            b_txt = str(blk.get("text") or blk.get("content") or "")
                            if b_txt and b_txt != base_txt:
                                chunks.append(b_txt)
                            elif btype in ("tool_use", "tool_result") and blk.get("input"):
                                chunks.append(json.dumps(blk.get("input")))

                    att_bytes = 0
                    for ak in ("attachments", "files", "files_v2"):
                        atts = msg.get(ak)
                        if isinstance(atts, list) and atts:
                            tool_counts["file_attachment"] = tool_counts.get("file_attachment", 0) + len(atts)
                            for att in atts:
                                if isinstance(att, dict):
                                    ext_c = str(att.get("extracted_content") or att.get("content") or "")
                                    if ext_c:
                                        att_bytes += len(ext_c.encode("utf-8"))
                                    elif att.get("file_size"):
                                        try:
                                            att_bytes += min(200_000, int(att["file_size"]))
                                        except (TypeError, ValueError):
                                            pass

                    joined_txt = " ".join(c for c in chunks if c)
                    if not first_prompt and not is_p and joined_txt:
                        first_prompt = joined_txt[:240]

                    b_len = max(32, len(joined_txt.encode("utf-8")) + att_bytes)
                    conv_bytes += b_len
                    step_tuples.append((is_p, b_len, b_len > 16_000))

                    if is_p and conv_model:
                        est_out_tok = max(20, b_len // 4)
                        tin_p, tout_p, _ = _resolve_model_rates(
                            conv_model, in_price_per_m, out_price_per_m, explicit_price_override
                        )
                        _record_turn_model_usage(
                            model_usage_map,
                            conv_model,
                            turns=1,
                            output_tokens=est_out_tok,
                            output_usd=(est_out_tok / 1_000_000.0) * tout_p,
                        )

                if not step_tuples:
                    continue

                top_tool = (
                    max(tool_counts.items(), key=lambda kv: kv[1])[0]
                    if tool_counts
                    else "claude_chat"
                )
                eff_in_p, eff_out_p, _ = _resolve_model_rates(
                    conv_model, in_price_per_m, out_price_per_m, explicit_price_override
                )
                rec = _evaluate_step_sequence(
                    step_Tuples=step_tuples,
                    conv_bytes=conv_bytes,
                    short_id=cid[:8],
                    conv_id=cid,
                    harness="claude_chat",
                    mtime=eff_mtime,
                    in_price_per_m=eff_in_p,
                    out_price_per_m=eff_out_p,
                    top_tool_override=top_tool,
                    first_prompt=first_prompt,
                    model_usage=model_usage_map,
                )
                full_kb = round(conv_bytes / 1024.0, 1)
                csi_val = round(
                    max(len(step_tuples) / 850.0, full_kb / 1800.0)
                    + 0.05 * min(1.0, rec["truncated_steps"] / 25.0),
                    3,
                )
                rec["csi"] = csi_val
                rec["health_tier"] = (
                    "RED"
                    if (len(step_tuples) >= 850 or full_kb >= 1800.0 or csi_val >= 1.0)
                    else ("YELLOW" if (len(step_tuples) >= 600 or full_kb >= 1200.0 or csi_val >= 0.70) else "GREEN")
                )
                records.append(rec)
        except Exception:
            continue
    return records


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
    unshielded_only: bool = False,
    import_paths: Optional[List[Any]] = None,
    shield_ledger_summary: Optional[Dict[str, Any]] = None,
    ci_dir: Optional[Path] = None,
    lowering: bool = False,
    include_benchmarks: bool = False,
) -> Dict[str, Any]:
    now = time.time()
    window_seconds = max(1, int(days)) * 86400.0 if days > 0 else 0.0
    in_price = float(price_per_m)
    out_price = round(in_price * 5.0, 2)
    eff_inc_bench = _resolve_include_benchmarks(include_benchmarks)
    excluded_stats: Dict[str, int] = {}

    agy_env = os.environ.get("AGY_BRAIN_DIR") or os.environ.get("ANTIGRAVITY_BRAIN_DIR")
    any_custom_override = any(
        x is not None
        for x in (
            agy_dir,
            claude_dir,
            ci_dir,
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
            "ANTIGRAVITY_BRAIN_DIR",
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
        [Path(agy_dir or agy_env).expanduser()]
        if (agy_dir is not None or agy_env)
        else [h / ".gemini" / "antigravity" / "brain" for h in candidate_homes]
    )
    if ci_dir is not None:
        claude_roots = [Path(ci_dir).expanduser()]
        if claude_dir is not None:
            claude_roots.append(Path(claude_dir).expanduser())
    else:
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
                    root,
                    window_seconds,
                    now,
                    in_price,
                    out_price,
                    explicit_price_override=explicit_price_override,
                    include_benchmarks=eff_inc_bench,
                    excluded_stats=excluded_stats,
                )
            ]
        )
        if (not any_custom_override or agy_dir is not None or agy_env)
        else []
    )
    gemini_cli_recs = (
        scan_gemini_cli_harness(
            gemini_cli_dir,
            window_seconds,
            now,
            in_price,
            out_price,
            include_benchmarks=eff_inc_bench,
            excluded_stats=excluded_stats,
        )
        if (not any_custom_override or gemini_cli_dir is not None or os.environ.get("GEMINI_CLI_DIR"))
        else []
    )
    claude_recs = (
        _dedup_records(
            [
                r
                for root in claude_roots
                for r in scan_claude_code_harness(
                    root,
                    window_seconds,
                    now,
                    in_price,
                    out_price,
                    explicit_price_override=explicit_price_override,
                    include_benchmarks=eff_inc_bench,
                    excluded_stats=excluded_stats,
                )
            ]
        )
        if (not any_custom_override or ci_dir is not None or claude_dir is not None or os.environ.get("CLAUDE_CODE_DIR"))
        else []
    )
    goose_recs = (
        _dedup_records(
            [
                r
                for root in goose_roots
                for r in scan_goose_harness(
                    root,
                    window_seconds,
                    now,
                    in_price,
                    out_price,
                    include_benchmarks=eff_inc_bench,
                    excluded_stats=excluded_stats,
                )
            ]
        )
        if (not any_custom_override or goose_db is not None or os.environ.get("GOOSE_SESSIONS_DB"))
        else []
    )
    cursor_recs = (
        scan_cursor_harness(
            cursor_override,
            window_seconds,
            now,
            in_price,
            out_price,
            include_benchmarks=eff_inc_bench,
            excluded_stats=excluded_stats,
        )
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
            include_benchmarks=eff_inc_bench,
            excluded_stats=excluded_stats,
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
        scan_cline_roo_harness(
            cline_dir,
            window_seconds,
            now,
            in_price,
            out_price,
            include_benchmarks=eff_inc_bench,
            excluded_stats=excluded_stats,
        )
        if (not any_custom_override or cline_dir is not None or os.environ.get("CLINE_STORAGE_DIR"))
        else []
    )
    windsurf_recs = (
        scan_windsurf_harness(
            windsurf_dir,
            window_seconds,
            now,
            in_price,
            out_price,
            include_benchmarks=eff_inc_bench,
            excluded_stats=excluded_stats,
        )
        if (not any_custom_override or windsurf_dir is not None or os.environ.get("WINDSURF_STORAGE_DIR"))
        else []
    )
    aider_recs = (
        scan_aider_harness(
            aider_path,
            window_seconds,
            now,
            in_price,
            out_price,
            include_benchmarks=eff_inc_bench,
            excluded_stats=excluded_stats,
        )
        if (not any_custom_override or aider_path is not None or os.environ.get("AIDER_HISTORY_PATH"))
        else []
    )
    continue_recs = (
        scan_continue_dev_harness(
            continue_dir,
            window_seconds,
            now,
            in_price,
            out_price,
            include_benchmarks=eff_inc_bench,
            excluded_stats=excluded_stats,
        )
        if (not any_custom_override or continue_dir is not None or os.environ.get("CONTINUE_SESSIONS_DIR"))
        else []
    )
    zed_recs = (
        scan_zed_harness(
            zed_dir,
            window_seconds,
            now,
            in_price,
            out_price,
            include_benchmarks=eff_inc_bench,
            excluded_stats=excluded_stats,
        )
        if (not any_custom_override or zed_dir is not None or os.environ.get("ZED_THREADS_DIR"))
        else []
    )
    claude_chat_recs = (
        scan_claude_export_records(
            list(import_paths),
            window_seconds,
            now,
            in_price,
            out_price,
            explicit_price_override=explicit_price_override,
        )
        if import_paths
        else []
    )

    all_records = (
        agy_recs
        + gemini_cli_recs
        + claude_recs
        + claude_chat_recs
        + goose_recs
        + cursor_recs
        + codex_recs
        + cline_recs
        + windsurf_recs
        + aider_recs
        + continue_recs
        + zed_recs
    )
    if unshielded_only:
        all_records = [r for r in all_records if not r.get("wbc_shielded")]
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
    if claude_chat_recs:
        harness_breakdown["claude_chat"] = _summarize_harness(
            "Claude.ai / Desktop Chat", claude_chat_recs
        )

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
    lowering_traces_by_id: Dict[str, Dict[str, Any]] = {
        str(r.get("conv_id", "")): (r.pop("_lowering_trace", None) or {}) for r in all_records
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

    if ci_dir is not None and agy_dir is None and not agy_env:
        fallback_custom_root = Path(ci_dir).expanduser()
    else:
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
        strict_containment=(ci_dir is not None),
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
    if isinstance(shield_ledger_summary, dict):
        l_events = int(shield_ledger_summary.get("total_events", 0) or 0)
        l_turns = int(shield_ledger_summary.get("actual_turns_observed", 0) or 0)
        l_dtok = int(shield_ledger_summary.get("direct_tokens_saved", 0) or 0)
        l_ctok = int(shield_ledger_summary.get("cumulative_tokens_saved", l_dtok) or 0)
        l_usd = float(shield_ledger_summary.get("api_dollars_saved_est", 0.0) or 0.0)
        l_dusd = float(shield_ledger_summary.get("api_dollars_saved_direct", 0.0) or 0.0)
        l_sec = int(shield_ledger_summary.get("secrets_blocked", 0) or 0)
        l_bots = [str(b) for b in (shield_ledger_summary.get("bots_protected") or []) if b]
        wbc_shield_telemetry["ledger_synced"] = True
        wbc_shield_telemetry["ledger_events"] = l_events
        wbc_shield_telemetry["ledger_turns"] = l_turns
        wbc_shield_telemetry["measured_direct_tokens_saved"] = l_dtok
        wbc_shield_telemetry["measured_cumulative_tokens_saved"] = l_ctok
        wbc_shield_telemetry["measured_usd_saved"] = round(l_usd, 2)
        wbc_shield_telemetry["measured_direct_usd_saved"] = round(l_dusd, 4)
        wbc_shield_telemetry["measured_secrets_blocked"] = l_sec
        wbc_shield_telemetry["bots_protected"] = l_bots
        bot_to_harness = {
            "antigravity": "antigravity",
            "claude-code": "claude_code",
            "cursor": "cursor",
            "codex": "codex",
            "goose": "goose",
        }
        merged_wired = set(wbc_shield_telemetry.get("wired_harnesses") or [])
        for b_lbl in l_bots:
            mapped_h = bot_to_harness.get(b_lbl.strip().lower())
            if mapped_h:
                merged_wired.add(mapped_h)
        wbc_shield_telemetry["wired_harnesses"] = sorted(merged_wired)
        if l_events > 0 or l_dtok > 0 or l_sec > 0:
            wbc_shield_telemetry["wbc_active_on_machine"] = True

    # Do not project additional savings on sessions where WBC hooks already fired
    unshielded_projected_saved = round(
        sum(
            float(r.get("hal_saved_usd", 0.0))
            for r in all_records
            if not hook_fired_by_id.get(str(r.get("conv_id", "")), False)
        ),
        2,
    )
    if unshielded_only:
        wbc_shield_telemetry["wbc_active_on_machine"] = False
    if wbc_shield_telemetry.get("wbc_active_on_machine") and wbc_shield_telemetry.get("shielded_sessions", 0) > 0:
        eff_saved_total = unshielded_projected_saved
        eff_wbc_spend = round(max(0.0, exact_api_spend_total - eff_saved_total), 2)
    else:
        eff_saved_total = hal_saved_total
        eff_wbc_spend = hal_wbc_spend_total
    eff_monthly_saved = round(eff_saved_total * scale_30d_hal, 2)
    eff_annual_saved = round(eff_monthly_saved * 12.0, 2)
    projected_savings["counterfactual_saved_usd"] = hal_saved_total
    projected_savings["counterfactual_annual_saved_usd"] = hal_annual_saved
    projected_savings["projected_saved_usd"] = eff_saved_total
    projected_savings["projected_wbc_spend_usd"] = eff_wbc_spend
    projected_savings["projected_monthly_saved_usd"] = eff_monthly_saved
    projected_savings["projected_annual_saved_usd"] = eff_annual_saved
    projected_savings["hal_saved_usd"] = eff_saved_total
    projected_savings["hal_wbc_spend_usd"] = eff_wbc_spend
    projected_savings["hal_monthly_saved_usd"] = eff_monthly_saved
    projected_savings["hal_annual_saved_usd"] = eff_annual_saved

    lowering_analysis: Optional[Dict[str, Any]] = None
    if lowering:
        active_claude_recs = [r for r in all_records if r.get("harness") == "claude_code"]
        lowering_analysis = _analyze_claude_lowering_choreography(
            claude_recs=active_claude_recs,
            traces_by_id=lowering_traces_by_id,
            turn0_static_overhead=turn0_static_overhead,
            ws_dir=resolved_turn0_ws,
            h_dir=resolved_turn0_home,
            ci_dir=ci_dir,
            days=days,
            in_price=in_price,
            total_monthly_usd=round(float(estimated_frontier_spend_usd) * scale_30d_hal, 2),
            hook_fired_by_id=hook_fired_by_id,
        )

    excluded_total = int(excluded_stats.get("_total", 0))
    excluded_by_harness = {k: int(v) for k, v in excluded_stats.items() if k != "_total" and int(v) > 0}

    result_payload: Dict[str, Any] = {
        "window_days": int(days),
        "input_price_per_m": in_price,
        "output_price_per_m": out_price,
        "seats": int(seats),
        "include_benchmarks": eff_inc_bench,
        "excluded_benchmark_conversations": excluded_total,
        "excluded_benchmark_by_harness": excluded_by_harness,
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
            "excluded_benchmark_conversations": excluded_total,
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
    if lowering_analysis is not None:
        result_payload["lowering_analysis"] = lowering_analysis
    return result_payload


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
    cli_mode: bool = False,
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
    meas_ctok = int(wbc_st.get("measured_cumulative_tokens_saved", meas_dtok))
    meas_usd = float(wbc_st.get("measured_usd_saved", 0.0))
    ledger_synced = bool(wbc_st.get("ledger_synced", False))
    ledger_events = int(wbc_st.get("ledger_events", 0))
    mo_top = audit.get("model_opportunities", {}) or {}
    top_opp_usd = float(mo_top.get("total_opportunity_usd", 0.0))
    top_opp_yr = float(mo_top.get("projected_annual_opportunity_usd", 0.0))

    show_wbc_active_header = wbc_active and ((convs > 0 and sh_n > 0) or (cli_mode and ledger_events > 0))
    eff_mult = round(1.0 / max(0.25, 1.0 - (cut_pct_f / 100.0)), 1) if cut_pct_f > 0 else 1.0
    if show_wbc_active_header:
        cf_saved_usd = float(hal.get("counterfactual_saved_usd", hal_saved_usd))
        cf_cut_pct_f = (cf_saved_usd / exact_usd * 100.0) if exact_usd > 0 else cut_pct_f
        cf_mult = round(1.0 / max(0.25, 1.0 - (cf_cut_pct_f / 100.0)), 1) if cf_cut_pct_f > 0 else 1.0
        active_mult = round(max(q_mult, cf_mult), 1) if (q_mult > 1.0 or cf_mult > 1.0) else eff_mult
        if convs > 0 and active_mult > 1.0:
            if cut_pct > 0 and eff_mult >= 1.1 and eff_mult < active_mult - 0.05:
                quota_str = (
                    f"{BOLD}{CYAN}~{active_mult:.1f}x{RESET} longer Pro/Max quota "
                    f"{DIM}(+{eff_mult:.1f}x more upside){RESET}"
                )
            else:
                quota_str = f"{BOLD}{CYAN}~{active_mult:.1f}x{RESET} longer Pro/Max quota before rate limits"
        else:
            quota_str = f"{DIM}Longer Pro/Max quota before rate-limit lockouts{RESET}"
    else:
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

    lines: List[str] = [
        "",
        f"{BOLD}{CYAN}WHOLE BODY COMPUTE - {days}-Day Agent Audit{RESET}",
        f"{DIM}{provenance_str}{RESET}",
        divider,
        "",
    ]

    if show_wbc_active_header:
        wired_lbl = ", ".join(h.replace("_", "-") for h in sorted(wired_set))[:32] or "local hooks"
        tot_lbl = f"Total {days}d API-Equiv"
        lines.append(
            f"  {BOLD}{tot_lbl:<26}{RESET}{BOLD}${exact_usd:,.2f}{RESET}  {GREEN}(WBC active: {wired_lbl}){RESET}"
        )
        if convs > 0 and sh_n > 0:
            cov_ratio = max(0.0, min(1.0, sh_n / float(max(1, convs))))
            cov_pct = int(round(cov_ratio * 100.0))
            bar_sh = _bar(cov_ratio, width=12, fill_color=GREEN)
            cov_lbl = f"{cov_pct}% of sessions"
            lines.append(
                f"  {GREEN}{'Shield Coverage':<26}{RESET}{bar_sh}  {BOLD}{GREEN}{cov_lbl}{RESET}  {DIM}({sh_n:,}/{convs:,} sess){RESET}"
            )
        lines.append("")
        if ledger_synced and (ledger_events > 0 or meas_dtok > 0):
            full_sh_plain = (
                f"${meas_usd:,.2f} saved • {_fmt_tok(meas_ctok)} cum "
                f"({_fmt_tok(meas_dtok)} dir, {ledger_events:,} gates)"
            )
            if 28 + len(full_sh_plain) <= 78:
                lines.append(
                    f"  {BOLD}{'Verified WBC Shield':<26}{RESET}{GREEN}${meas_usd:,.2f} saved{RESET} • "
                    f"{_fmt_tok(meas_ctok)} cum ({_fmt_tok(meas_dtok)} dir, {ledger_events:,} gates)"
                )
            else:
                lines.append(
                    f"  {BOLD}{'Verified WBC Shield':<26}{RESET}{GREEN}${meas_usd:,.2f} saved{RESET} • "
                    f"{_fmt_tok(meas_ctok)} cum tok ({ledger_events:,} gates)"
                )
        elif meas_dtok > 0 or meas_slices > 0:
            sh_hdr = "Verified WBC Shield" if cli_mode else "Measured Hook Shield"
            lines.append(
                f"  {BOLD}{sh_hdr:<26}{RESET}{GREEN}{meas_slices:,} active AST slices{RESET} • {_fmt_tok(meas_dtok)} direct tok blocked"
            )
        elif cli_mode:
            lines.append(
                f"  {BOLD}{'Verified WBC Shield':<26}{RESET}{DIM}0 gates in {days}d (see body savings){RESET}"
            )
        lines.append(f"  {BOLD}{'Subscription Impact':<26}{RESET}{quota_str}")
        if unwired_save > 0 and unwired_harnesses:
            uw_lbl = ", ".join(h.replace("_", "-") for h in unwired_harnesses[:2])
            uw_plain = f"Save ${unwired_save:,.2f} / {days}d by wiring {uw_lbl}"
            if 28 + len(uw_plain) <= 78:
                lines.append(
                    f"  {BOLD}{'Remaining Opportunity':<26}{RESET}{BOLD}{GREEN}Save ${unwired_save:,.2f} / {days}d{RESET} by wiring {uw_lbl}"
                )
            else:
                lines.append(
                    f"  {BOLD}{'Remaining Opportunity':<26}{RESET}{BOLD}{GREEN}Save ${unwired_save:,.2f} / {days}d{RESET} (run body begin)"
                )
        if cli_mode and top_opp_usd >= 0.05:
            up_plain = f"~{_fmt_usd(top_opp_usd)} / {days}d (~${top_opp_yr:,.0f}/yr • flush + model fit)"
            if 28 + len(up_plain) <= 78:
                lines.append(
                    f"  {BOLD}{'Remaining Upside':<26}{RESET}{GREEN}~{_fmt_usd(top_opp_usd)} / {days}d{RESET} "
                    f"{DIM}(~${top_opp_yr:,.0f}/yr • flush + model fit){RESET}"
                )
            else:
                lines.append(
                    f"  {BOLD}{'Remaining Upside':<26}{RESET}{GREEN}~{_fmt_usd(top_opp_usd)} / {days}d{RESET} "
                    f"{DIM}(~${top_opp_yr:,.0f}/yr){RESET}"
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
        cf_yr = float(hal.get("counterfactual_annual_saved_usd", hal_yr_saved))
        if wbc_active and unwired_save > 0:
            base_yr = unwired_save * (30.0 / max(1, days)) * 12.0
        elif hal_yr_saved > 0:
            base_yr = hal_yr_saved
        else:
            meas_yr = meas_usd * (30.0 / max(1, days)) * 12.0
            base_yr = max(cf_yr, meas_yr + top_opp_yr)
        team_hal_yr = base_yr * max(1, seats)
        lines.append(
            f"  {BOLD}{MAGENTA}Team Projection ({seats} Seats){RESET} {BOLD}{MAGENTA}${team_hal_yr:,.0f} / yr{RESET} projected savings"
        )

    lines.append(f"  {DIM}{rate_note}{RESET}")
    if not (wbc_active and convs > 0 and sh_n > 0):
        flush_steps = _resolve_flush_step_interval()
        projection_note = (
            f"Projection: AST slicing (>2KB tool out -> 500-tok outline) + flush @{flush_steps}t"
        )
        lines.append(f"  {DIM}{projection_note}{RESET}")

    hb = audit.get("harness_breakdown", {})
    active_harnesses: List[Tuple[str, Dict[str, Any]]] = []
    empty_labels: List[str] = []
    for h_key in (
        "antigravity",
        "gemini_cli",
        "claude_code",
        "claude_chat",
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

    la = audit.get("lowering_analysis")
    if isinstance(la, dict) and la.get("enabled"):
        def _fmt_mo(val: float) -> str:
            return f"${val:,.0f}" if val >= 1000.0 else f"${val:,.2f}"

        def _fmt_pct(val: float) -> str:
            return f"{int(round(val))}%" if abs(val - round(val)) < 0.05 else f"{val:.1f}%"

        s_code = float(la.get("code_lowering_monthly_usd", 0.0))
        y_pct = _fmt_pct(float(la.get("code_lowering_turn_pct", 0.0)))
        n_pat = int(la.get("recurring_patterns_count", 0))
        s_wbc = float(la.get("wbc_shield_monthly_usd", 0.0))
        z_pct = _fmt_pct(float(la.get("wbc_shield_rem_pct", 0.0)))
        s_tot = float(la.get("combined_monthly_usd", 0.0))
        p_pct = _fmt_pct(float(la.get("combined_net_pct", 0.0)))
        p_unit = str(la.get("period_unit") or "/mo")
        lines.extend(
            [
                "",
                f"{BOLD}SAVINGS BREAKDOWN{RESET}",
                f"  1. Automate {n_pat} repeatable loops:  {_fmt_mo(s_code)}{p_unit}  ({y_pct} of turns)",
                f"  2. Cut log, file & MCP bloat:      {_fmt_mo(s_wbc)}{p_unit}  ({z_pct} on remaining turns)",
                f"  Total:                             {_fmt_mo(s_tot)}{p_unit}  ({p_pct} less spend)",
            ]
        )

    top_convs = audit.get("top_bloated_conversations", [])
    titles_map = audit.get("_local_titles_by_id", {}) or {}
    if top_convs:
        lines.append("")
        lines.append(f"  {BOLD}Costliest Sessions{RESET}")
        harness_short_map = {
            "antigravity": "antigravity",
            "claude_code": "claude-code",
            "claude_chat": "claude-chat",
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

    if cli_mode:
        lines.extend(["", divider, f"{BOLD}NEXT STEPS{RESET}"])
        if not wbc_active or unwired_harnesses:
            lines.append(
                f"  • {BOLD}Shield agents:{RESET} {CYAN}body begin{RESET}  {DIM}(wire automatic PreToolUse/PostToolUse hooks){RESET}"
            )
        if marathon_cnt > 0 or float((mo.get("marathon_reread_drag") or {}).get("savable_usd", 0.0)) >= 0.50:
            lines.append(
                f"  • {BOLD}Cut re-reads:{RESET}  {CYAN}body flush --auto{RESET}  {DIM}(compact 50+ turn marathon sessions){RESET}"
            )
        if t0_mcp >= 2000 or t0_tot >= 8000:
            lines.append(
                f"  • {BOLD}Trim Turn-0:{RESET}   {CYAN}body skills audit{RESET}  {DIM}(prune dormant MCP & skill schemas){RESET}"
            )
        sav_cmd = f"body savings --window {days}d" if days in (1, 7, 30) else "body savings"
        lines.append(
            f"  • {BOLD}Organ ledger:{RESET}  {CYAN}{sav_cmd}{RESET}  {DIM}(live organ & hook receipt){RESET}"
        )
        lines.append(
            f"  • {BOLD}Team report:{RESET}   {DIM}body audit --seats 10 --export-receipt ROI_RECEIPT.md{RESET}"
        )
        lines.append("")
    else:
        lines.extend(
            [
                "",
                divider,
                f"{BOLD}{GREEN}Start Saving Now - Copy & Paste to Install (100% Free Open-Core):{RESET}",
                f"  • {BOLD}Install:{RESET}   {CYAN}curl -fsSL https://wholebodycompute.com/begin/audit | sh{RESET}",
                f"               {DIM}Wires automatic hooks into active agents - just restart & code.{RESET}",
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
            **({"audit_channel_group": "cloud_ci"} if acq_source in ("cloud", "ci") else {}),
            "lowering_enabled": bool(audit.get("lowering_analysis")),
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
        custom_host = os.environ.get("POSTHOG_HOST", "").strip()
        ingest_target = (
            f"{custom_host.rstrip('/')}/capture/"
            if custom_host
            else "https://wholebodycompute.com/v1/telemetry"
        )
        req = urllib.request.Request(
            ingest_target,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "WholeBodyCompute-Audit/1.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=1.2):
            pass
    except Exception:
        pass


def _collect_standalone_cloud_sources(
    import_paths: List[str],
    admin_key: Optional[str],
    use_cloud_api: bool,
    days: int = 30,
    price_per_m: float = 5.0,
) -> List[Dict[str, Any]]:
    """Zero-dependency cloud export & Anthropic Admin API collector for standalone `audit.py`."""
    import csv as _csv
    import datetime as _dt
    import io as _io
    import urllib.error as _uerr
    import urllib.parse as _uparse
    import urllib.request as _ureq
    import zipfile as _zf

    def _parse_dt(raw: Any) -> Optional[_dt.datetime]:
        if not raw:
            return None
        s = str(raw).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            p = _dt.datetime.fromisoformat(s)
            return p if p.tzinfo else p.replace(tzinfo=_dt.timezone.utc)
        except Exception:
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
                try:
                    return _dt.datetime.strptime(s[:10], fmt).replace(tzinfo=_dt.timezone.utc)
                except Exception:
                    continue
        return None

    cutoff = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=max(1, int(days)))
    results: List[Dict[str, Any]] = []

    for raw_path in import_paths:
        p = Path(raw_path).expanduser().resolve()
        if not p.is_file():
            results.append({"ok": False, "source": "import", "label": str(raw_path), "error": f"File not found: {p}"})
            continue
        try:
            suf = p.suffix.lower()
            if suf == ".csv":
                rows_m = in_t = out_t = cr_t = 0
                exp_usd = 0.0
                has_cost = False
                for row in _csv.DictReader(_io.StringIO(p.read_text(encoding="utf-8", errors="ignore"))):
                    if not isinstance(row, dict):
                        continue
                    rd = None
                    for k in ("date", "usage_date", "timestamp", "created_at", "bucket_start", "day"):
                        for rk, rv in row.items():
                            if rk and rk.strip().lower() == k and rv:
                                rd = _parse_dt(rv)
                                if rd:
                                    break
                        if rd:
                            break
                    if rd is not None and rd < cutoff:
                        continue
                    rows_m += 1

                    def _n(keys: tuple[str, ...]) -> float:
                        for kk in keys:
                            for rk, rv in row.items():
                                if rk and rk.strip().lower() == kk and rv not in (None, ""):
                                    try:
                                        return float(str(rv).strip().replace("$", "").replace(",", ""))
                                    except ValueError:
                                        return 0.0
                        return 0.0

                    in_t += int(_n(("input_tokens", "uncached_input_tokens", "prompt_tokens")))
                    out_t += int(_n(("output_tokens", "completion_tokens")))
                    cr_t += int(_n(("cache_read_input_tokens", "cached_input_tokens")))
                    cv = _n(("cost_usd", "total_cost_usd", "cost", "amount_usd", "amount", "usd"))
                    if cv > 0:
                        exp_usd += cv
                        has_cost = True
                mod_in = (in_t / 1e6) * price_per_m + (cr_t / 1e6) * (price_per_m * 0.1)
                mod_out = (out_t / 1e6) * (price_per_m * 5.0)
                usd = round(exp_usd, 2) if has_cost else round(mod_in + mod_out, 2)
                tot_in = in_t + cr_t
                ch_pct = round((cr_t / float(max(1, tot_in))) * 100.0, 1) if tot_in > 0 else 0.0
                if tot_in > 0:
                    in_sh = mod_in / max(0.01, mod_in + mod_out)
                    sav_u = round(usd * in_sh * (0.62 if ch_pct < 35.0 else 0.42), 2)
                else:
                    sav_u = round(usd * 0.35, 2) if usd > 0 else 0.0
                results.append(
                    {
                        "ok": True,
                        "source": "billing_csv",
                        "label": p.name,
                        "rows": rows_m,
                        "input_tokens": in_t,
                        "cache_read_tokens": cr_t,
                        "output_tokens": out_t,
                        "total_tokens": in_t + cr_t + out_t,
                        "cache_hit_rate_pct": ch_pct,
                        "estimated_spend_usd": usd,
                        "estimated_wbc_savable_usd": sav_u,
                        "exact_cost_reported": has_cost,
                    }
                )
            else:
                chat_recs = scan_claude_export_records(
                    [str(p)],
                    max(1, int(days)) * 86400.0 if days > 0 else 0.0,
                    time.time(),
                    price_per_m,
                    round(price_per_m * 5.0, 2),
                )
                if not chat_recs:
                    if suf == ".zip":
                        with _zf.ZipFile(p, "r") as z:
                            m = next((n for n in z.namelist() if n.endswith("conversations.json")), None)
                            if not m:
                                results.append({"ok": False, "source": "import", "label": p.name, "error": "No conversations.json in zip"})
                                continue
                    results.append(
                        {
                            "ok": True,
                            "source": "claude_export_json",
                            "label": p.name,
                            "conversations": 0,
                            "messages": 0,
                            "total_tokens": 0,
                            "estimated_spend_usd": 0.0,
                            "exact_cost_reported": False,
                        }
                    )
                    continue
                c_cnt = len(chat_recs)
                m_cnt = sum(int(r["steps"]) for r in chat_recs)
                ei = sum(int(r["actual_cumulative_input_tokens"]) for r in chat_recs)
                eo = sum(int(r["output_tokens"]) for r in chat_recs)
                ew = sum(int(r["wasted_bloat_tokens"]) for r in chat_recs)
                eopt = sum(int(r["bbb_optimized_tokens"]) for r in chat_recs)
                usd = round(sum(float(r["exact_api_spend_usd"]) for r in chat_recs), 2)
                sav_u = round(sum(float(r["hal_saved_usd"]) for r in chat_recs), 2)
                red_n = sum(1 for r in chat_recs if r.get("health_tier") == "RED")
                yel_n = sum(1 for r in chat_recs if r.get("health_tier") == "YELLOW")
                mar_n = sum(1 for r in chat_recs if int(r.get("planner_turns", 0)) >= 25)
                bloat_p = round((ew / float(max(1, ei))) * 100.0, 1) if ei > 0 else 0.0
                q_mult = round(float(ei) / float(max(1, eopt)), 2) if ei > 0 and eopt > 0 else 1.0
                results.append(
                    {
                        "ok": True,
                        "source": "claude_export_json",
                        "label": p.name,
                        "conversations": c_cnt,
                        "messages": m_cnt,
                        "input_tokens": ei,
                        "output_tokens": eo,
                        "wasted_bloat_tokens": ew,
                        "avoidable_bloat_pct": bloat_p,
                        "quota_stretch_multiple": q_mult,
                        "csi_red_threads": red_n,
                        "csi_yellow_threads": yel_n,
                        "marathon_threads": mar_n,
                        "total_tokens": ei + eo,
                        "estimated_spend_usd": usd,
                        "estimated_wbc_savable_usd": sav_u,
                        "exact_cost_reported": False,
                    }
                )
        except Exception as exc:
            results.append({"ok": False, "source": "import", "label": p.name, "error": str(exc)})

    if use_cloud_api or admin_key:
        key = (admin_key or "").strip()
        if not key:
            results.append({"ok": False, "source": "anthropic_admin_api", "label": "Anthropic Admin API", "error": "Missing ANTHROPIC_ADMIN_API_KEY"})
        else:
            now = _dt.datetime.now(_dt.timezone.utc)
            qs = _uparse.urlencode({"starting_at": cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"), "ending_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "bucket_width": "1d"})
            hdrs = {"x-api-key": key, "anthropic-version": "2023-06-01", "accept": "application/json"}
            in_t = cr_t = out_t = 0
            cost_usd = 0.0
            errs: List[str] = []
            for kind, ep in (("usage", "usage_report/messages"), ("cost", "cost_report")):
                try:
                    req = _ureq.Request(f"https://api.anthropic.com/v1/organizations/{ep}?{qs}", headers=hdrs, method="GET")
                    with _ureq.urlopen(req, timeout=15.0) as resp:
                        data = json.loads(resp.read().decode("utf-8", errors="ignore"))
                    for b in data.get("data", []) or []:
                        for r in b.get("results", [b]) or []:
                            if not isinstance(r, dict):
                                continue
                            if kind == "usage":
                                in_t += int(r.get("uncached_input_tokens", 0) or r.get("input_tokens", 0) or 0)
                                cr_t += int(r.get("cache_read_input_tokens", 0) or 0)
                                out_t += int(r.get("output_tokens", 0) or 0)
                            else:
                                try:
                                    cost_usd += float(r.get("amount", 0))
                                except (TypeError, ValueError):
                                    pass
                except (_uerr.URLError, _uerr.HTTPError, Exception) as exc:
                    errs.append(f"{kind}: {exc}")
            if len(errs) == 2:
                results.append({"ok": False, "source": "anthropic_admin_api", "label": "Anthropic Admin API", "error": "; ".join(errs)})
            else:
                results.append({"ok": True, "source": "anthropic_admin_api", "label": "Anthropic Admin API", "total_tokens": in_t + cr_t + out_t, "estimated_spend_usd": round(cost_usd, 2), "exact_cost_reported": cost_usd > 0})
    return results


def _format_standalone_cloud_section(cloud_sources: List[Dict[str, Any]]) -> str:
    lines = ["", "CLOUD & ANTHROPIC CONSOLE AUDIT (OPT-IN)"]
    for src in cloud_sources:
        lbl = str(src.get("label", src.get("source", "cloud")))[:28]
        if not src.get("ok"):
            err_s = str(src.get("error", "unknown error"))[:42]
            lines.append(f"  • {lbl}: [SKIPPED] {err_s}")
            continue
        extra = ""
        if "conversations" in src:
            extra = f" | {int(src['conversations']):,} convs"
        elif "rows" in src:
            extra = f" | {int(src['rows']):,} rows"
        tag = "exact" if src.get("exact_cost_reported") else "est."
        usd = float(src.get("estimated_spend_usd", 0.0))
        tok_s = f"{int(src.get('total_tokens', 0)):,}"
        lines.append(f"  • {lbl}: {tok_s} tok ({tag} ${usd:,.2f}){extra}"[:78])
        if src.get("source") == "claude_export_json" and int(src.get("conversations", 0)) > 0:
            bp = float(src.get("avoidable_bloat_pct", 0.0))
            qm = float(src.get("quota_stretch_multiple", 1.0))
            rn = int(src.get("csi_red_threads", 0))
            yn = int(src.get("csi_yellow_threads", 0))
            mn = int(src.get("marathon_threads", 0))
            lines.append(
                f"    CSI: {rn} RED, {yn} YELLOW, {mn} marathons • {bp:.0f}% bloat (~{qm:.1f}x quota)"[:78]
            )
        elif src.get("source") == "billing_csv":
            ch = float(src.get("cache_hit_rate_pct", 0.0))
            sv = float(src.get("estimated_wbc_savable_usd", 0.0))
            lines.append(
                f"    Console API: {ch:.1f}% cache hit • est. WBC save ~${sv:,.2f}"[:78]
            )
    return "\n".join(lines)


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
    parser.add_argument(
        "--import",
        dest="import_paths",
        action="append",
        default=[],
        metavar="PATH",
        help="Opt-in offline cloud export (`conversations.json`, `.zip`, Console `.csv`, or `auto`)",
    )
    parser.add_argument(
        "--auto-downloads",
        dest="auto_downloads",
        action="store_true",
        help="Opt-in: auto-discover recent Claude `.zip`/`conversations.json` & Console `.csv` in ~/Downloads",
    )
    parser.add_argument(
        "--cloud",
        action="store_true",
        help="Opt-in: also query Anthropic Admin API (requires ANTHROPIC_ADMIN_API_KEY or --anthropic-admin-key)",
    )
    parser.add_argument(
        "--anthropic-admin-key",
        dest="anthropic_admin_key",
        default=None,
        metavar="KEY",
        help="Anthropic Organization Admin API key for cloud usage/cost report",
    )
    parser.add_argument("--json", action="store_true", help="Output structured JSON")
    parser.add_argument("--unshielded-only", action="store_true", help="Analyze only unshielded sessions as if WBC were not yet installed")
    parser.add_argument(
        "--include-benchmarks",
        action="store_true",
        dest="include_benchmarks",
        help="Include synthetic benchmark, SWE-bench, AgentDojo, and /tmp test sessions in spend audit",
    )
    parser.add_argument(
        "--lowering",
        action="store_true",
        help="Analyze deterministic Bash/tool loops for Tier 0/1 code lowering vs WBC Linux Shield savings",
    )
    parser.add_argument(
        "--ci-dir",
        dest="ci_dir",
        default=None,
        metavar="PATH",
        help="Optional path to exported CI Claude Code JSONL transcripts to audit exclusively",
    )
    args = parser.parse_args(argv)

    raw_imports = list(getattr(args, "import_paths", None) or [])
    use_auto_dl = bool(getattr(args, "auto_downloads", False)) or any(
        str(p).strip().lower() == "auto" for p in raw_imports
    )
    import_paths: List[str] = [str(p) for p in raw_imports if str(p).strip().lower() != "auto"]
    auto_dl_missing = False
    if use_auto_dl:
        env_dl_override = os.environ.get("WBC_DOWNLOADS_DIR", "").strip()
        dl_base = (
            Path(env_dl_override).expanduser()
            if env_dl_override
            else ((Path(args.home_dir).expanduser() / "Downloads") if args.home_dir else None)
        )
        found_dl = discover_standalone_cloud_exports(downloads_dir=dl_base)
        if not found_dl:
            auto_dl_missing = True
        for dp in found_dl:
            if str(dp) not in import_paths:
                import_paths.append(str(dp))

    edge_acq = (os.environ.get("WBC_AUDIT_SOURCE") or WBC_EDGE_AUDIT_SOURCE or "").strip().lower()
    enable_lowering = bool(args.lowering) or (edge_acq in ("cloud", "ci"))

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
        unshielded_only=bool(args.unshielded_only),
        import_paths=import_paths,
        ci_dir=Path(args.ci_dir) if args.ci_dir else None,
        lowering=enable_lowering,
        include_benchmarks=bool(getattr(args, "include_benchmarks", False)),
    )

    cloud_sources: List[Dict[str, Any]] = []
    use_cloud_api = bool(getattr(args, "cloud", False))
    admin_key = getattr(args, "anthropic_admin_key", None) or (
        os.environ.get("ANTHROPIC_ADMIN_API_KEY") if use_cloud_api else None
    )
    if import_paths or use_cloud_api or admin_key:
        cloud_sources = _collect_standalone_cloud_sources(
            import_paths=import_paths,
            admin_key=admin_key,
            use_cloud_api=use_cloud_api,
            days=args.days,
            price_per_m=args.price_per_m,
        )
        audit["cloud_usage"] = cloud_sources

    show_team = bool(explicit_seats or args.seats > 1)
    if args.export_receipt:
        out_p = Path(args.export_receipt)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(
            format_standalone_receipt(audit, color=False, show_team=show_team),
            encoding="utf-8",
        )

    dl_hint = "No ~/Downloads file: console.anthropic.com/settings/cost or claude.ai export"
    if auto_dl_missing:
        audit["cloud_download_hint"] = dl_hint

    if args.json:
        json_audit = dict(audit)
        json_audit.pop("_local_titles_by_id", None)
        print(json.dumps(json_audit, indent=2), flush=True)
    else:
        use_color = False if args.no_color else None
        print(format_standalone_receipt(audit, color=use_color, show_team=show_team), flush=True)
        if cloud_sources:
            print(_format_standalone_cloud_section(cloud_sources), flush=True)
        if auto_dl_missing:
            print(f"  {dl_hint}", flush=True)

    _emit_audit_telemetry(audit, no_telemetry=bool(getattr(args, "no_telemetry", False)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
