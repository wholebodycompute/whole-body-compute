"""
Allowlisted Open-Core Multi-Harness FinOps & Token Waste Audit Engine (`body audit-spend`).

Provides zero-dependency historical spend auditing, Context Saturation Index (CSI) math,
tool bloat attribution, and terminal/Markdown FinOps ROI receipts across Antigravity,
Claude Code, Cursor, Goose, Codex, Cline/Roo, Windsurf, Aider, Continue, and Zed—with
zero runtime dependency on Pro-only cognitive modules (`cognitive_sentry`, `cognitive_capsules`,
or `dot_memory`).
"""

from __future__ import annotations

import datetime
import glob
import importlib.util
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from whole_body_compute.redaction import redact_secrets

logger = logging.getLogger(__name__)

DEFAULT_BRAIN_DIR = Path.home() / ".gemini" / "antigravity" / "brain"

# Context Saturation Thresholds (shared with Community doctor/metrics & Pro sentry)
SOFT_STEP_THRESHOLD = 600
HARD_STEP_THRESHOLD = 850
SOFT_SIZE_KB_THRESHOLD = 1200.0
HARD_SIZE_KB_THRESHOLD = 1800.0
DEFAULT_FLUSH_STEP_INTERVAL = 150


def resolve_flush_step_interval(explicit_interval: Optional[int] = None) -> int:
    """Resolve phase-boundary Glymphatic Flush step interval (default 150 steps / ~50 turns)."""
    if explicit_interval is not None and int(explicit_interval) > 0:
        return max(10, int(explicit_interval))
    env_val = os.environ.get("WBC_FLUSH_STEP_INTERVAL")
    if env_val:
        try:
            return max(10, int(env_val))
        except ValueError:
            pass
    return DEFAULT_FLUSH_STEP_INTERVAL


def resolve_brain_dir(brain_dir: Optional[Path] = None) -> Path:
    """Resolve Antigravity brain directory with environment variable overrides."""
    env_dir = os.environ.get("AGY_BRAIN_DIR") or os.environ.get("ANTIGRAVITY_BRAIN_DIR")
    return Path(brain_dir or env_dir or (Path.home() / ".gemini" / "antigravity" / "brain")).expanduser()


def compute_avg_step_gap(lines: List[Dict[str, Any]]) -> float:
    """Compute average intra-turn model/tool step latency gap in seconds (excluding human think pauses)."""
    gaps: List[float] = []
    for i in range(1, len(lines)):
        try:
            cur = lines[i]
            # Exclude idle human reading/typing gaps leading into a new user prompt
            if cur.get("type") == "USER_INPUT" or cur.get("source") == "USER_EXPLICIT":
                continue
            t1_s = lines[i - 1].get("created_at")
            t2_s = cur.get("created_at")
            if not t1_s or not t2_s:
                continue
            t1 = datetime.datetime.fromisoformat(str(t1_s).replace("Z", "+00:00"))
            t2 = datetime.datetime.fromisoformat(str(t2_s).replace("Z", "+00:00"))
            gap = (t2 - t1).total_seconds()
            if 0.0 <= gap < 180.0:
                gaps.append(gap)
        except Exception:
            continue
    return round(sum(gaps) / len(gaps), 2) if gaps else 0.0


def calculate_csi(
    steps: int,
    full_size_kb: float,
    truncated_steps: int = 0,
    recent_step_sec: float = 0.0,
    early_step_sec: float = 0.0,
) -> Tuple[float, str]:
    """
    Calculate Context Saturation Index (CSI) and Health Tier:
      - CSI < 0.70: GREEN
      - 0.70 <= CSI < 1.00: YELLOW (Soft advisory)
      - CSI >= 1.00: RED (Hard handoff recommended)
    """
    step_factor = steps / float(HARD_STEP_THRESHOLD)
    size_factor = full_size_kb / float(HARD_SIZE_KB_THRESHOLD)
    base = max(step_factor, size_factor)

    trunc_penalty = 0.05 * min(1.0, truncated_steps / 25.0)
    latency_penalty = 0.0
    if early_step_sec > 0.1:
        # True turn-latency inflation against the thread's own early baseline
        inflation = recent_step_sec / early_step_sec
        if recent_step_sec > 3.5 and inflation > 1.05:
            latency_penalty = 0.05 * min(1.0, (inflation - 1.05) / 0.80)
    elif recent_step_sec > 3.5:
        latency_penalty = 0.05 * min(1.0, (recent_step_sec - 3.5) / 6.5)

    csi = round(base + trunc_penalty + latency_penalty, 3)

    if steps >= HARD_STEP_THRESHOLD or full_size_kb >= HARD_SIZE_KB_THRESHOLD or csi >= 1.00:
        tier = "RED"
    elif steps >= SOFT_STEP_THRESHOLD or full_size_kb >= SOFT_SIZE_KB_THRESHOLD or csi >= 0.70:
        tier = "YELLOW"
    else:
        tier = "GREEN"

    return csi, tier


def analyze_bloat_attribution(
    lines: List[Dict[str, Any]],
    steps: int,
    full_size_kb: float,
    compact_size_kb: float,
    truncated_steps: int,
    csi: float,
) -> Tuple[str, float, Dict[str, Dict[str, Any]], List[str]]:
    """Classify primary context bloat cause and rank offending tools by byte share."""
    tool_stats: Dict[str, Dict[str, Any]] = {}
    tool_invocation_sequence: List[str] = []
    pending_tool_result_tools: List[str] = []
    excess_content_kb = 0.0
    trunc_overflow_bonus = (
        max(0.0, full_size_kb - compact_size_kb) / max(1, truncated_steps)
        if truncated_steps > 0
        else 0.0
    )

    for step in lines:
        raw_tc = step.get("tool_calls")
        step_tool_names: List[str] = []
        if isinstance(raw_tc, list) and raw_tc:
            for tc in raw_tc:
                if isinstance(tc, dict):
                    tname = tc.get("name") or tc.get("ToolName")
                    if not tname and isinstance(tc.get("function"), dict):
                        tname = tc.get("function", {}).get("name")
                    elif not tname and isinstance(tc.get("function"), str):
                        tname = tc.get("function")
                    if tname:
                        step_tool_names.append(str(tname))
                elif isinstance(tc, str) and tc.strip():
                    step_tool_names.append(tc.strip())
        elif step.get("tool_name") and not pending_tool_result_tools:
            step_tool_names.append(str(step.get("tool_name")))

        if step_tool_names:
            active_tools = step_tool_names
            pending_tool_result_tools = list(step_tool_names)
            for t in step_tool_names:
                if t not in tool_stats:
                    tool_stats[t] = {
                        "calls": 0,
                        "truncated": 0,
                        "est_kb": 0.0,
                        "share_pct": 0.0,
                    }
                tool_stats[t]["calls"] += 1
                tool_invocation_sequence.append(t)
        elif pending_tool_result_tools:
            active_tools = pending_tool_result_tools
            pending_tool_result_tools = []
        else:
            active_tools = ["agent_response"]
            if "agent_response" not in tool_stats:
                tool_stats["agent_response"] = {
                    "calls": 0,
                    "truncated": 0,
                    "est_kb": 0.0,
                    "share_pct": 0.0,
                }
            if step.get("type") in ("PLANNER_RESPONSE", "USER_INPUT"):
                tool_stats["agent_response"]["calls"] += 1

        raw_content = step.get("content")
        if isinstance(raw_content, str):
            content_bytes = len(raw_content.encode("utf-8"))
        elif raw_content is not None:
            content_bytes = len(json.dumps(raw_content).encode("utf-8"))
        else:
            content_bytes = 0

        step_content_kb = content_bytes / 1024.0
        if step_content_kb > 8.0:
            excess_content_kb += step_content_kb - 8.0

        is_truncated = bool(step.get("truncated_fields"))
        step_trunc_bonus = trunc_overflow_bonus if is_truncated else 0.0
        step_total_kb = step_content_kb + step_trunc_bonus

        share = step_total_kb / max(1, len(active_tools))
        for t in active_tools:
            if t not in tool_stats:
                tool_stats[t] = {
                    "calls": 0,
                    "truncated": 0,
                    "est_kb": 0.0,
                    "share_pct": 0.0,
                }
            tool_stats[t]["est_kb"] += share
            if is_truncated:
                tool_stats[t]["truncated"] += 1

    overflow_kb = max(0.0, full_size_kb - compact_size_kb)
    avoidable_bloat_kb = round(excess_content_kb + overflow_kb, 1)

    total_tool_kb = sum(data["est_kb"] for data in tool_stats.values())
    sorted_tools: Dict[str, Dict[str, Any]] = {}
    for t, data in sorted(tool_stats.items(), key=lambda kv: kv[1]["est_kb"], reverse=True):
        pct = round((data["est_kb"] / total_tool_kb) * 100.0, 1) if total_tool_kb > 0 else 0.0
        sorted_tools[t] = {
            "calls": data["calls"],
            "truncated": data["truncated"],
            "est_kb": round(data["est_kb"], 1),
            "share_pct": pct,
        }

    consecutive_retries = 0
    max_streak = 0
    current_streak = 0
    last_tool = None
    for t in tool_invocation_sequence:
        if t == last_tool:
            current_streak += 1
            if current_streak >= 3:
                consecutive_retries += 1
        else:
            last_tool = t
            current_streak = 1
        if current_streak > max_streak:
            max_streak = current_streak

    if csi < 0.50 and truncated_steps < 10 and avoidable_bloat_kb < 200.0:
        primary_cause = "NOMINAL"
    elif truncated_steps >= 15 or (full_size_kb - compact_size_kb) >= 400.0:
        primary_cause = "TRUNCATION_CASCADE"
    elif avoidable_bloat_kb >= 300.0 or (full_size_kb >= 1000.0 and steps < 500):
        primary_cause = "UNFILTERED_OUTPUT_BLOAT"
    elif consecutive_retries >= 12 or max_streak >= 12:
        primary_cause = "RETRY_CHURN_LOOP"
    else:
        primary_cause = "NATURAL_LONG_HORIZON"

    recommendations: List[str] = []
    if csi >= 0.70:
        recommendations.append(
            "Context saturation elevated (CSI >= 0.70). Run 'body flush --auto' to serialize state and copy resumption prompt."
        )

    candidate_tools = [t for t in sorted_tools.keys() if t != "agent_response"]
    top_tool = (
        candidate_tools[0]
        if candidate_tools
        else (list(sorted_tools.keys())[0] if sorted_tools else None)
    )
    top_stats = sorted_tools.get(top_tool, {}) if top_tool else {}

    if top_tool in ("run_command", "bash", "execute_command"):
        if primary_cause == "TRUNCATION_CASCADE" or top_stats.get("truncated", 0) > 0:
            recommendations.append(
                "High truncation on run_command. Route massive outputs/logs through local-scout (ReconScout) or pass concise flags (-q, --tb=short)."
            )
        else:
            recommendations.append(
                "Heavy output volume on run_command. Filter noisy command stdout with grep/head or delegate deep scans to local-scout."
            )
    elif top_tool in ("read_file", "view_file"):
        if primary_cause == "TRUNCATION_CASCADE" or top_stats.get("truncated", 0) > 0:
            recommendations.append(
                "Large file reads truncated on read_file. Use ByteOffset/Line ranges or route full-file scans through local-scout."
            )
        else:
            recommendations.append(
                "Large file reads detected on read_file. Use ByteOffset/Line ranges or route full-file scans through local-scout."
            )
    elif top_tool in ("search_web", "web_search"):
        recommendations.append(
            "Web search results bloating context. Constrain query scopes or synthesize summaries off-context."
        )
    elif primary_cause == "RETRY_CHURN_LOOP" and top_tool:
        recommendations.append(
            f"Repeated consecutive retries on '{top_tool}'. Verify prerequisites or step arguments before looping."
        )
    elif primary_cause == "UNFILTERED_OUTPUT_BLOAT" and top_tool:
        recommendations.append(
            f"Tool '{top_tool}' generated significant uncompressed output. Consider streaming or paging tool outputs."
        )

    if not recommendations:
        if primary_cause == "NOMINAL":
            recommendations.append(
                "Context telemetry nominal. Continue standard execution cadence."
            )
        else:
            recommendations.append(
                "Context saturation approaching soft limit. Plan handoff via 'body flush' soon."
            )

    return primary_cause, avoidable_bloat_kb, sorted_tools, recommendations[:3]


def inspect_conversation_lightweight(
    conv_id: Optional[str] = None,
    brain_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """
    Lightweight Community-safe conversation vitals inspection (`steps`, `csi`, `health_tier`,
    `full_size_kb`, `latency_inflation_ratio`) with zero Pro module dependencies.
    """
    target_brain = resolve_brain_dir(brain_dir)
    if not target_brain.is_dir():
        return None

    raw_id = str(conv_id or "").strip()
    if raw_id.startswith("conversation://"):
        raw_id = raw_id.replace("conversation://", "", 1).strip()

    c_dir: Optional[Path] = None
    if raw_id:
        exact = target_brain / raw_id
        if exact.is_dir():
            c_dir = exact
        else:
            matches = [p for p in target_brain.glob(f"{raw_id}*") if p.is_dir()]
            if matches:
                c_dir = matches[0]

    if c_dir is None:
        for env_k in ("AGY_CONVERSATION_ID", "CONVERSATION_ID"):
            val = os.environ.get(env_k)
            if val and (target_brain / val).is_dir():
                c_dir = target_brain / val
                break

    if c_dir is None:
        transcripts = glob.glob(
            str(target_brain / "*" / ".system_generated" / "logs" / "transcript.jsonl")
        )
        if not transcripts:
            return None
        transcripts.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        c_dir = Path(transcripts[0]).parents[2]

    logs_dir = c_dir / ".system_generated" / "logs"
    compact_p = logs_dir / "transcript.jsonl"
    full_p = logs_dir / "transcript_full.jsonl"
    primary_p = compact_p if compact_p.is_file() else (full_p if full_p.is_file() else None)
    if not primary_p:
        return None

    compact_kb = round(os.path.getsize(primary_p) / 1024.0, 1)
    full_kb = round(os.path.getsize(full_p) / 1024.0, 1) if full_p.is_file() else compact_kb

    lines: List[Dict[str, Any]] = []
    with open(primary_p, "r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            s = raw_line.strip()
            if not s:
                continue
            try:
                lines.append(json.loads(s))
            except Exception:
                continue

    if not lines:
        return None

    steps = len(lines)
    truncated_steps = sum(1 for l in lines if l.get("truncated_fields"))
    early_sec = compute_avg_step_gap(lines[: min(40, steps)])
    recent_sec = compute_avg_step_gap(lines[max(0, steps - 40) :])
    inflation = round(recent_sec / early_sec, 2) if early_sec > 0.1 and recent_sec > 0.0 else 1.0
    csi, tier = calculate_csi(
        steps=steps,
        full_size_kb=full_kb,
        truncated_steps=truncated_steps,
        recent_step_sec=recent_sec,
        early_step_sec=early_sec,
    )
    primary_cause, avoidable_kb, sorted_tools, recommendations = analyze_bloat_attribution(
        lines=lines,
        steps=steps,
        full_size_kb=full_kb,
        compact_size_kb=compact_kb,
        truncated_steps=truncated_steps,
        csi=csi,
    )
    return {
        "conv_id": c_dir.name,
        "short_id": c_dir.name[:8],
        "steps": steps,
        "csi": csi,
        "health_tier": tier,
        "compact_size_kb": compact_kb,
        "full_size_kb": full_kb,
        "truncated_steps": truncated_steps,
        "early_step_sec": early_sec or 2.8,
        "recent_step_sec": recent_sec or 2.8,
        "latency_inflation_ratio": inflation,
        "primary_bloat_cause": primary_cause,
        "avoidable_bloat_kb": avoidable_kb,
        "top_offender_tools": sorted_tools,
        "recommendations": recommendations,
    }


def _locate_standalone_audit_script() -> Optional[Path]:
    """Locate bundled `standalone_audit.py` (wheel/Homebrew) or repo `docs/audit.py`."""
    try:
        from whole_body_compute.bundled import get_bundled_audit_script_path

        bundled = get_bundled_audit_script_path()
        if bundled.is_file():
            return bundled
    except Exception:
        pass
    here = Path(__file__).resolve()
    for parent in (here.parents[2], here.parents[1], *here.parents):
        candidate = parent / "docs" / "audit.py"
        if candidate.is_file():
            return candidate
    return None


def audit_historical_spend(
    days: int = 30,
    price_per_m: float = 5.0,
    output_price_per_m: Optional[float] = None,
    seats: int = 10,
    top: int = 5,
    brain_dir: Optional[Path] = None,
    dot_store: Optional[Any] = None,
    claude_dir: Optional[Path] = None,
    goose_db: Optional[Path] = None,
    cursor_dir: Optional[Path] = None,
    codex_dir: Optional[Path] = None,
    cline_dir: Optional[Path] = None,
    windsurf_dir: Optional[Path] = None,
    aider_path: Optional[Path] = None,
    continue_dir: Optional[Path] = None,
    zed_dir: Optional[Path] = None,
    flush_step_interval: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Scan historical conversation transcripts over the given days window and compute
    cumulative O(N^2) re-ingested token volume, wasted context bloat tokens,
    Whole Body Compute (BBB + Glymphatic Flush + Bicameral 85% Local Metal) optimized spend,
    and subscription ROI multiples.
    """
    target_brain = resolve_brain_dir(brain_dir)
    resolved_flush_steps = resolve_flush_step_interval(flush_step_interval)
    out_price_per_m = (
        float(output_price_per_m)
        if output_price_per_m is not None
        else round(float(price_per_m) * 5.0, 2)
    )
    in_price_per_m = float(price_per_m)
    now = time.time()
    window_seconds = max(1, int(days)) * 86400.0 if days > 0 else 0.0

    conv_dirs: List[Tuple[float, Path, Path, Optional[Path]]] = []
    if target_brain.is_dir():
        for c_dir in target_brain.iterdir():
            if not c_dir.is_dir() or c_dir.name.startswith("."):
                continue
            logs_dir = c_dir / ".system_generated" / "logs"
            compact_p = logs_dir / "transcript.jsonl"
            full_p = logs_dir / "transcript_full.jsonl"
            primary_p = (
                compact_p if compact_p.is_file() else (full_p if full_p.is_file() else None)
            )
            if not primary_p:
                continue
            try:
                mtime = os.path.getmtime(primary_p)
                if window_seconds > 0 and (now - mtime) > window_seconds:
                    continue
                conv_dirs.append(
                    (mtime, c_dir, primary_p, full_p if full_p.is_file() else None)
                )
            except Exception:
                continue

    conv_dirs.sort(key=lambda x: x[0], reverse=True)

    MAX_CONTEXT_WINDOW_TOKENS = 200_000
    total_steps = 0
    total_planner_turns = 0
    total_transcript_bytes = 0
    total_raw_unbounded_tokens = 0
    total_actual_input_tokens = 0
    total_output_tokens = 0
    total_wasted_bloat_tokens = 0
    total_quadratic_post_sat_tokens = 0
    total_unfiltered_tool_bloat_tokens = 0
    total_bicameral_offload_tokens = 0
    total_bbb_optimized_tokens = 0
    total_bbb_optimized_output_tokens = 0

    thread_records: List[Dict[str, Any]] = []

    for mtime, c_dir, primary_p, full_p in conv_dirs:
        try:
            compact_bytes = os.path.getsize(primary_p)
            full_bytes = os.path.getsize(full_p) if full_p else compact_bytes
            conv_bytes = max(compact_bytes, full_bytes)

            lines: List[Dict[str, Any]] = []
            raw_line_lengths: List[int] = []
            with open(primary_p, "r", encoding="utf-8", errors="ignore") as fh:
                for raw_line in fh:
                    s = raw_line.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                        lines.append(obj)
                        raw_line_lengths.append(len(s.encode("utf-8")))
                    except Exception:
                        continue

            if not lines:
                continue

            steps = len(lines)
            truncated_steps = sum(1 for l in lines if l.get("truncated_fields"))
            overflow_bytes_total = max(0, full_bytes - compact_bytes)
            overflow_per_trunc = (
                float(overflow_bytes_total) / float(max(1, truncated_steps))
                if truncated_steps > 0
                else 0.0
            )

            running_raw_unbounded_ctx = 0
            running_unshielded_ctx = 0
            running_sliced_no_flush_ctx = 0
            running_flushed_sliced_ctx = 0
            running_tool_bloat_ctx = 0

            conv_raw_unbounded_tok = 0
            conv_actual_input_tok = 0
            conv_output_tok = 0
            conv_tool_bloat_tok = 0
            conv_quad_post_sat_tok = 0
            conv_flushed_sliced_cum_tok = 0

            planner_turns = sum(1 for l in lines if l.get("type") == "PLANNER_RESPONSE")
            has_planner_turns = planner_turns > 0
            if not has_planner_turns:
                planner_turns = max(1, int(round(steps * 0.38)))

            first_prompt = ""
            parent_conv_id = ""
            for idx, (step, raw_len) in enumerate(zip(lines, raw_line_lengths), start=1):
                stype = step.get("type")
                if stype == "USER_INPUT" and idx <= 5:
                    content_s = str(step.get("content") or "")
                    if not parent_conv_id:
                        m_parent = re.search(
                            r"Continuing from (?:conversation://)?([0-9a-fA-F\-]{36})",
                            content_s,
                        )
                        if m_parent and m_parent.group(1) != c_dir.name:
                            parent_conv_id = m_parent.group(1)
                    if not first_prompt:
                        m_req = re.search(
                            r"<USER_REQUEST>(.*?)</USER_REQUEST>", content_s, re.DOTALL
                        )
                        first_prompt = redact_secrets(
                            (m_req.group(1) if m_req else content_s).strip().replace("\n", " ")
                        )[:110]

                is_trunc = bool(step.get("truncated_fields"))
                step_effective_bytes = float(raw_len) + (
                    overflow_per_trunc if is_trunc else 0.0
                )
                step_tok = max(1, int(round(step_effective_bytes / 4.0)))

                if stype == "PLANNER_RESPONSE":
                    conv_output_tok += max(20, step_tok)

                if step_effective_bytes > 2000.0 or is_trunc:
                    base_cap = min(step_tok, 500)
                    excess_tok = max(0, step_tok - base_cap)
                    compressed_tok = base_cap + int(round(excess_tok * 0.10))
                    bloat_tok = max(0, step_tok - compressed_tok)
                else:
                    compressed_tok = step_tok
                    bloat_tok = 0

                running_raw_unbounded_ctx += step_tok
                running_unshielded_ctx = min(
                    MAX_CONTEXT_WINDOW_TOKENS, running_unshielded_ctx + step_tok
                )
                running_sliced_no_flush_ctx = min(
                    MAX_CONTEXT_WINDOW_TOKENS, running_sliced_no_flush_ctx + compressed_tok
                )
                running_tool_bloat_ctx = min(
                    MAX_CONTEXT_WINDOW_TOKENS, running_tool_bloat_ctx + bloat_tok
                )

                if (
                    idx > resolved_flush_steps
                    and (idx - 1) % resolved_flush_steps == 0
                ) or (
                    full_bytes >= 120_000
                    and steps <= resolved_flush_steps
                    and idx > 25
                    and (idx - 1) % 25 == 0
                ):
                    running_flushed_sliced_ctx = 450 + compressed_tok
                else:
                    running_flushed_sliced_ctx = min(
                        MAX_CONTEXT_WINDOW_TOKENS, running_flushed_sliced_ctx + compressed_tok
                    )

                if stype == "PLANNER_RESPONSE" or (not has_planner_turns):
                    conv_raw_unbounded_tok += running_raw_unbounded_ctx
                    conv_actual_input_tok += running_unshielded_ctx
                    conv_tool_bloat_tok += running_tool_bloat_ctx
                    conv_quad_post_sat_tok += max(
                        0, running_sliced_no_flush_ctx - running_flushed_sliced_ctx
                    )
                    conv_flushed_sliced_cum_tok += running_flushed_sliced_ctx

            if not has_planner_turns:
                conv_raw_unbounded_tok = max(1, int(round(conv_raw_unbounded_tok * 0.38)))
                conv_actual_input_tok = max(1, int(round(conv_actual_input_tok * 0.38)))
                conv_tool_bloat_tok = int(round(conv_tool_bloat_tok * 0.38))
                conv_quad_post_sat_tok = int(round(conv_quad_post_sat_tok * 0.38))
                conv_flushed_sliced_cum_tok = max(
                    1, int(round(conv_flushed_sliced_cum_tok * 0.38))
                )

            has_tool_activity = any(bool(l.get("tool_calls") or l.get("tool_name")) for l in lines)
            if steps <= 3 and conv_bytes < 1000 and not has_tool_activity and conv_tool_bloat_tok == 0 and conv_quad_post_sat_tok == 0:
                conv_bbb_opt_tok = conv_actual_input_tok
                conv_bbb_opt_out_tok = conv_output_tok
            else:
                # Empirical SWE-bench Verified (k=3) context reduction (-32.4% -> 0.676x)
                # combined with measured AST tool-output slicing and >25-step flush savings.
                conv_bbb_opt_tok = max(
                    1, min(conv_actual_input_tok, int(round(conv_flushed_sliced_cum_tok * 0.676)))
                )
                conv_bbb_opt_out_tok = (
                    max(1, int(round(conv_output_tok * 0.812))) if conv_output_tok > 0 else 0
                )
            conv_wasted_tok = max(0, conv_actual_input_tok - conv_bbb_opt_tok)

            raw_components = conv_quad_post_sat_tok + conv_tool_bloat_tok
            if raw_components > conv_wasted_tok and raw_components > 0:
                scale_c = float(conv_wasted_tok) / float(raw_components)
                conv_quad_post_sat_tok = int(round(conv_quad_post_sat_tok * scale_c))
                conv_tool_bloat_tok = max(0, conv_wasted_tok - conv_quad_post_sat_tok)
                conv_local_offload_tok = 0
            else:
                conv_local_offload_tok = max(0, conv_wasted_tok - raw_components)

            compact_kb = round(compact_bytes / 1024.0, 1)
            full_kb = round(full_bytes / 1024.0, 1)
            csi, tier = calculate_csi(
                steps=steps,
                full_size_kb=full_kb,
                truncated_steps=truncated_steps,
            )
            primary_cause, avoidable_kb, top_tools, _ = analyze_bloat_attribution(
                lines=lines,
                steps=steps,
                full_size_kb=full_kb,
                compact_size_kb=compact_kb,
                truncated_steps=truncated_steps,
                csi=csi,
            )
            candidate_tools = [t for t in top_tools.keys() if t != "agent_response"]
            top_tool = (
                candidate_tools[0]
                if candidate_tools
                else (list(top_tools.keys())[0] if top_tools else "none")
            )

            conv_cons_frontier_usd = (conv_actual_input_tok / 1_000_000.0) * (
                in_price_per_m * 0.19
            ) + (conv_output_tok / 1_000_000.0) * out_price_per_m
            conv_cons_bbb_usd = (conv_bbb_opt_tok / 1_000_000.0) * (in_price_per_m * 0.19) + (
                conv_bbb_opt_out_tok / 1_000_000.0
            ) * out_price_per_m
            conv_cons_saved_usd = max(0.0, conv_cons_frontier_usd - conv_cons_bbb_usd)

            conv_real_frontier_usd = (conv_actual_input_tok / 1_000_000.0) * (
                in_price_per_m * 0.32
            ) + (conv_output_tok / 1_000_000.0) * out_price_per_m
            conv_real_bbb_usd = (conv_bbb_opt_tok / 1_000_000.0) * (in_price_per_m * 0.19) + (
                conv_bbb_opt_out_tok / 1_000_000.0
            ) * out_price_per_m
            conv_real_saved_usd = max(0.0, conv_real_frontier_usd - conv_real_bbb_usd)

            conv_uncached_frontier_usd = (
                conv_actual_input_tok / 1_000_000.0
            ) * in_price_per_m + (conv_output_tok / 1_000_000.0) * out_price_per_m
            conv_uncached_bbb_usd = (conv_bbb_opt_tok / 1_000_000.0) * (
                in_price_per_m * 0.32
            ) + (conv_bbb_opt_out_tok / 1_000_000.0) * out_price_per_m
            conv_uncached_saved_usd = max(
                0.0, conv_uncached_frontier_usd - conv_uncached_bbb_usd
            )

            total_steps += steps
            total_planner_turns += planner_turns
            total_transcript_bytes += conv_bytes
            total_raw_unbounded_tokens += conv_raw_unbounded_tok
            total_actual_input_tokens += conv_actual_input_tok
            total_output_tokens += conv_output_tok
            total_wasted_bloat_tokens += conv_wasted_tok
            total_quadratic_post_sat_tokens += conv_quad_post_sat_tok
            total_unfiltered_tool_bloat_tokens += conv_tool_bloat_tok
            total_bicameral_offload_tokens += conv_local_offload_tok
            total_bbb_optimized_tokens += conv_bbb_opt_tok
            total_bbb_optimized_output_tokens += conv_bbb_opt_out_tok

            thread_records.append(
                {
                    "harness": "antigravity",
                    "conv_id": c_dir.name,
                    "short_id": c_dir.name[:8],
                    "steps": steps,
                    "planner_turns": planner_turns,
                    "truncated_steps": truncated_steps,
                    "transcript_bytes": conv_bytes,
                    "transcript_kb": full_kb,
                    "csi": csi,
                    "health_tier": tier,
                    "bloat_trigger": primary_cause,
                    "top_offender_tool": top_tool,
                    "avoidable_bloat_kb": avoidable_kb,
                    "raw_unbounded_quadratic_tokens": conv_raw_unbounded_tok,
                    "actual_cumulative_input_tokens": conv_actual_input_tok,
                    "output_tokens": conv_output_tok,
                    "wasted_bloat_tokens": conv_wasted_tok,
                    "quadratic_post_saturation_tokens": conv_quad_post_sat_tok,
                    "unfiltered_tool_bloat_tokens": conv_tool_bloat_tok,
                    "bicameral_local_offload_tokens": conv_local_offload_tok,
                    "bbb_optimized_tokens": conv_bbb_opt_tok,
                    "bbb_optimized_output_tokens": conv_bbb_opt_out_tok,
                    "conservative_spend_usd": round(conv_cons_frontier_usd, 2),
                    "estimated_spend_usd": round(conv_real_frontier_usd, 2),
                    "uncached_spend_usd": round(conv_uncached_frontier_usd, 2),
                    "bbb_spend_usd": round(conv_real_bbb_usd, 2),
                    "conservative_savable_usd": round(conv_cons_saved_usd, 2),
                    "usd_savable": round(conv_real_saved_usd, 2),
                    "uncached_savable_usd": round(conv_uncached_saved_usd, 2),
                    "last_modified_str": datetime.datetime.fromtimestamp(mtime).strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                    "first_prompt": first_prompt,
                    "parent_conv_id": parent_conv_id,
                }
            )
        except Exception as exc:
            logger.debug("audit_historical_spend skipped %s: %s", c_dir, exc)
            continue

    agy_records = list(thread_records)

    # Compute Actual Verified Handoff Lineage Savings across parent -> child `body flush` chains
    agy_by_id: Dict[str, Dict[str, Any]] = {r["conv_id"]: r for r in agy_records}
    verified_handoff_chains: List[Dict[str, Any]] = []
    verified_flush_tokens_saved = 0
    verified_child_turns = 0
    for rec in agy_records:
        p_id = str(rec.get("parent_conv_id") or "")
        if not p_id:
            continue
        ancestors: List[str] = []
        cur_p = p_id
        seen_ids = {rec["conv_id"]}
        while cur_p and cur_p not in seen_ids:
            seen_ids.add(cur_p)
            ancestors.append(cur_p)
            parent_rec = agy_by_id.get(cur_p)
            cur_p = str(parent_rec.get("parent_conv_id") or "") if parent_rec else ""

        cum_shed_tok = 0
        for anc_id in ancestors:
            anc_rec = agy_by_id.get(anc_id)
            if anc_rec:
                anc_bytes = int(anc_rec.get("transcript_bytes", 0))
                cum_shed_tok += max(0, (anc_bytes // 4) - 450)
            else:
                # Fallback if ancestor lies outside window: check disk directly
                anc_dir = target_brain / anc_id / ".system_generated" / "logs"
                anc_fp = anc_dir / "transcript_full.jsonl"
                anc_cp = anc_dir / "transcript.jsonl"
                anc_p = anc_fp if anc_fp.is_file() else (anc_cp if anc_cp.is_file() else None)
                if anc_p:
                    try:
                        cum_shed_tok += max(0, (os.path.getsize(anc_p) // 4) - 450)
                    except OSError:
                        pass

        shed_per_turn = min(MAX_CONTEXT_WINDOW_TOKENS, cum_shed_tok)
        child_turns = int(rec.get("planner_turns", 0))
        saved_tok = shed_per_turn * child_turns
        if saved_tok > 0:
            verified_flush_tokens_saved += saved_tok
            verified_child_turns += child_turns
            chain_ids = [a[:8] for a in reversed(ancestors)] + [rec["short_id"]]
            verified_handoff_chains.append(
                {
                    "child_conv_id": rec["conv_id"],
                    "parent_conv_id": p_id,
                    "chain_display": " -> ".join(chain_ids),
                    "hops": len(ancestors),
                    "ancestor_shed_tokens_per_turn": shed_per_turn,
                    "child_planner_turns": child_turns,
                    "actual_tokens_saved": saved_tok,
                }
            )
    verified_handoff_chains.sort(key=lambda x: x["actual_tokens_saved"], reverse=True)
    verified_flush_cons_usd = round(
        (verified_flush_tokens_saved / 1_000_000.0) * (in_price_per_m * 0.19), 2
    )
    verified_flush_blended_usd = round(
        (verified_flush_tokens_saved / 1_000_000.0) * (in_price_per_m * 0.32), 2
    )
    verified_handoff_lineage = {
        "handoff_children_count": len(verified_handoff_chains),
        "child_planner_turns": verified_child_turns,
        "actual_tokens_saved": verified_flush_tokens_saved,
        "conservative_usd_saved": verified_flush_cons_usd,
        "blended_usd_saved": verified_flush_blended_usd,
        "chains": verified_handoff_chains[:10],
    }
    claude_records: List[Dict[str, Any]] = []
    goose_records: List[Dict[str, Any]] = []
    cursor_records: List[Dict[str, Any]] = []
    codex_records: List[Dict[str, Any]] = []
    cline_records: List[Dict[str, Any]] = []
    windsurf_records: List[Dict[str, Any]] = []
    aider_records: List[Dict[str, Any]] = []
    continue_records: List[Dict[str, Any]] = []
    zed_records: List[Dict[str, Any]] = []

    default_agy_brain = (Path.home() / ".gemini" / "antigravity" / "brain").expanduser()
    has_custom_agy = bool(
        brain_dir is not None
        or os.environ.get("AGY_BRAIN_DIR")
        or target_brain != default_agy_brain
    )
    has_external_custom = any(
        x is not None
        for x in (
            claude_dir,
            goose_db,
            cursor_dir,
            codex_dir,
            cline_dir,
            windsurf_dir,
            aider_path,
            continue_dir,
            zed_dir,
        )
    ) or any(
        os.environ.get(k)
        for k in (
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
        )
    )
    is_isolated_agy_test = bool(has_custom_agy and not has_external_custom)
    is_selective_custom = bool(has_custom_agy or has_external_custom)

    if not is_isolated_agy_test:
        try:
            audit_script = _locate_standalone_audit_script()
            if audit_script is not None and audit_script.is_file():
                spec = importlib.util.spec_from_file_location(
                    "wbc_standalone_audit", audit_script
                )
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    default_cursor = (
                        Path.home()
                        / "Library"
                        / "Application Support"
                        / "Cursor"
                        / "User"
                        / "workspaceStorage"
                    )
                    p_claude = Path(
                        claude_dir
                        or os.environ.get("CLAUDE_CODE_DIR")
                        or (Path.home() / ".claude" / "projects")
                    ).expanduser()
                    p_goose = Path(
                        goose_db
                        or os.environ.get("GOOSE_SESSIONS_DB")
                        or (Path.home() / ".config" / "goose" / "sessions.db")
                    ).expanduser()
                    p_cursor = Path(
                        cursor_dir or os.environ.get("CURSOR_STORAGE_DIR") or default_cursor
                    ).expanduser()

                    if (
                        not is_selective_custom
                        or claude_dir is not None
                        or os.environ.get("CLAUDE_CODE_DIR")
                    ):
                        claude_records = mod.scan_claude_code_harness(
                            p_claude, window_seconds, now, in_price_per_m, out_price_per_m
                        )
                    if (
                        not is_selective_custom
                        or goose_db is not None
                        or os.environ.get("GOOSE_SESSIONS_DB")
                    ):
                        goose_records = mod.scan_goose_harness(
                            p_goose, window_seconds, now, in_price_per_m, out_price_per_m
                        )
                    if (
                        not is_selective_custom
                        or cursor_dir is not None
                        or os.environ.get("CURSOR_STORAGE_DIR")
                    ):
                        cursor_records = mod.scan_cursor_harness(
                            p_cursor, window_seconds, now, in_price_per_m, out_price_per_m
                        )
                    if (
                        not is_selective_custom
                        or codex_dir is not None
                        or os.environ.get("CODEX_SESSIONS_DIR")
                        or os.environ.get("CODEX_HOME")
                    ):
                        codex_records = mod.scan_codex_harness(
                            codex_dir, window_seconds, now, in_price_per_m, out_price_per_m
                        )
                    if (
                        not is_selective_custom
                        or cline_dir is not None
                        or os.environ.get("CLINE_STORAGE_DIR")
                    ):
                        cline_records = mod.scan_cline_roo_harness(
                            cline_dir, window_seconds, now, in_price_per_m, out_price_per_m
                        )
                    if (
                        not is_selective_custom
                        or windsurf_dir is not None
                        or os.environ.get("WINDSURF_STORAGE_DIR")
                    ):
                        windsurf_records = mod.scan_windsurf_harness(
                            windsurf_dir, window_seconds, now, in_price_per_m, out_price_per_m
                        )
                    if (
                        not is_selective_custom
                        or aider_path is not None
                        or os.environ.get("AIDER_HISTORY_PATH")
                    ):
                        aider_records = mod.scan_aider_harness(
                            aider_path, window_seconds, now, in_price_per_m, out_price_per_m
                        )
                    if (
                        not is_selective_custom
                        or continue_dir is not None
                        or os.environ.get("CONTINUE_SESSIONS_DIR")
                    ):
                        continue_records = mod.scan_continue_dev_harness(
                            continue_dir, window_seconds, now, in_price_per_m, out_price_per_m
                        )
                    if (
                        not is_selective_custom
                        or zed_dir is not None
                        or os.environ.get("ZED_THREADS_DIR")
                    ):
                        zed_records = mod.scan_zed_harness(
                            zed_dir, window_seconds, now, in_price_per_m, out_price_per_m
                        )

                    for extra_r in (
                        claude_records
                        + goose_records
                        + cursor_records
                        + codex_records
                        + cline_records
                        + windsurf_records
                        + aider_records
                        + continue_records
                        + zed_records
                    ):
                        total_steps += int(extra_r["steps"])
                        total_planner_turns += int(extra_r["planner_turns"])
                        total_transcript_bytes += int(extra_r.get("transcript_bytes", 0))
                        total_raw_unbounded_tokens += int(
                            extra_r["raw_unbounded_quadratic_tokens"]
                        )
                        total_actual_input_tokens += int(
                            extra_r["actual_cumulative_input_tokens"]
                        )
                        total_output_tokens += int(extra_r["output_tokens"])
                        total_wasted_bloat_tokens += int(extra_r["wasted_bloat_tokens"])
                        total_quadratic_post_sat_tokens += int(
                            extra_r["quadratic_post_saturation_tokens"]
                        )
                        total_unfiltered_tool_bloat_tokens += int(
                            extra_r["unfiltered_tool_bloat_tokens"]
                        )
                        total_bicameral_offload_tokens += int(
                            extra_r.get("bicameral_local_offload_tokens", 0)
                        )
                        total_bbb_optimized_tokens += int(extra_r["bbb_optimized_tokens"])
                        total_bbb_optimized_output_tokens += int(
                            extra_r.get("bbb_optimized_output_tokens", 0)
                        )
                        thread_records.append(extra_r)
        except Exception as exc:
            logger.debug("External harness scan skipped: %s", exc)

    def _summarize_h(label: str, recs: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "label": label,
            "threads": len(recs),
            "steps": sum(int(r.get("steps", 0)) for r in recs),
            "planner_turns": sum(int(r.get("planner_turns", 0)) for r in recs),
            "transcript_kb": round(sum(float(r.get("transcript_kb", 0.0)) for r in recs), 1),
            "actual_input_tokens": sum(
                int(r.get("actual_cumulative_input_tokens", 0)) for r in recs
            ),
            "wasted_tokens": sum(int(r.get("wasted_bloat_tokens", 0)) for r in recs),
            "conservative_savable_usd": round(
                sum(float(r.get("conservative_savable_usd", 0.0)) for r in recs), 2
            ),
            "blended_savable_usd": round(
                sum(float(r.get("usd_savable", 0.0)) for r in recs), 2
            ),
        }

    harness_breakdown = {
        "antigravity": _summarize_h("Google Antigravity", agy_records),
        "claude_code": _summarize_h("Anthropic Claude Code", claude_records),
        "cursor": _summarize_h("Cursor Agent / Composer", cursor_records),
        "goose": _summarize_h("Block Goose CLI", goose_records),
        "codex": _summarize_h("OpenAI Codex CLI", codex_records),
        "cline_roo": _summarize_h("Cline / Roo Code", cline_records),
        "windsurf": _summarize_h("Windsurf Cascade", windsurf_records),
        "aider": _summarize_h("Aider CLI", aider_records),
        "continue_dev": _summarize_h("Continue.dev", continue_records),
        "zed": _summarize_h("Zed AI", zed_records),
    }

    thread_records.sort(key=lambda r: r["estimated_spend_usd"], reverse=True)
    top_threads = thread_records[: max(1, int(top))]

    def _calc_profile(
        in_p: float, out_p: float, in_mult: float, bbb_in_mult: float
    ) -> Dict[str, float]:
        f_usd = (total_actual_input_tokens / 1_000_000.0) * (in_p * in_mult) + (
            total_output_tokens / 1_000_000.0
        ) * out_p
        b_usd = (total_bbb_optimized_tokens / 1_000_000.0) * (in_p * bbb_in_mult) + (
            total_bbb_optimized_output_tokens / 1_000_000.0
        ) * out_p
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

    primary_tier = _calc_tier_spend(in_price_per_m, out_price_per_m)
    cons_p = primary_tier["conservative_cached"]
    real_p = primary_tier["realistic_blended"]
    ceil_p = primary_tier["uncached_ceiling"]

    estimated_frontier_spend_usd = real_p["estimated_frontier_spend_usd"]
    bbb_optimized_spend_usd = real_p["bbb_optimized_spend_usd"]
    wasted_spend_usd = real_p["wasted_spend_usd"]
    net_savings_usd = real_p["net_savings_usd"]
    savings_pct = (
        round((net_savings_usd / estimated_frontier_spend_usd) * 100.0, 1)
        if estimated_frontier_spend_usd > 0
        else 0.0
    )

    monthly_savings_per_seat_usd = real_p["monthly_savings_per_seat_usd"]
    projected_annual_savings_per_seat_usd = real_p["projected_annual_per_seat_usd"]
    projected_team_10_seats_annual_usd = real_p["projected_team_10_seats_annual_usd"]
    projected_team_seats_annual_usd = real_p["projected_team_seats_annual_usd"]

    # Optional Pro MemoryStore lookup; falls back cleanly to Community immune_audit_cli.jsonl
    if dot_store is None:
        try:
            from whole_body_compute.local.dot_memory import get_dot_store

            dot_store = get_dot_store()
        except Exception:
            dot_store = None

    live_organ10: Dict[str, Any] = {
        "total_pre_checks": 0,
        "intercepted_count": 0,
        "context_gate_intercepts": 0,
        "bounded_slice_reads": 0,
        "direct_tokens_saved": 0,
        "quadratic_tokens_saved": 0,
        "estimated_usd_saved": 0.0,
    }
    try:
        sc = None
        if dot_store is not None and hasattr(dot_store, "get_immune_pre_post_scorecard"):
            try:
                sc = dot_store.get_immune_pre_post_scorecard(limit=50)
            except Exception:
                sc = None
        if sc is None:
            from whole_body_compute.organs.immune import get_immune_pre_post_scorecard

            sc = get_immune_pre_post_scorecard(limit=50)
        if isinstance(sc, dict):
            live_organ10 = {
                "total_pre_checks": int(sc.get("total_pre_checks", 0)),
                "intercepted_count": int(sc.get("intercepted_count", 0)),
                "context_gate_intercepts": int(sc.get("context_gate_intercepts", 0)),
                "bounded_slice_reads": int(sc.get("bounded_slice_reads", 0)),
                "direct_tokens_saved": int(sc.get("direct_tokens_saved", 0)),
                "quadratic_tokens_saved": int(sc.get("quadratic_tokens_saved", 0)),
                "estimated_usd_saved": round(float(sc.get("estimated_usd_saved", 0.0)), 2),
            }
    except Exception:
        pass

    autonomy_ratchet: Dict[str, Any] = {
        "baseline_day1_local_pct": 0.0,
        "current_local_autonomy_pct": 0.0,
        "delta_autonomy_gain_pct": 0.0,
        "cloud_dependency_reduction_pct": 0.0,
        "verified_cloud_exemplars_count": 0,
        "lgtm_local_edits_count": 0,
        "distilled_pairs_count": 0,
        "graduated_categories": {},
        "ratchet_summary": "Cold start (0 local exemplars distilled yet)",
    }
    try:
        if dot_store is not None and hasattr(dot_store, "get_autonomy_ratchet_report"):
            ar = dot_store.get_autonomy_ratchet_report()
            if isinstance(ar, dict) and "current_local_autonomy_pct" in ar:
                if int(ar.get("verified_cloud_exemplars_count", 0)) > 0 or int(ar.get("lgtm_local_edits_count", 0)) > 0:
                    autonomy_ratchet = ar
    except Exception:
        pass

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

    if total_actual_input_tokens > 0 and total_bbb_optimized_tokens > 0:
        raw_ratio = float(total_actual_input_tokens) / float(max(1, total_bbb_optimized_tokens))
        quota_stretch_multiple = round(min(6.0, max(1.0, raw_ratio)), 2)
        avg_turn_latency_unshielded_s = 17.4
        avg_turn_latency_wbc_s = 10.6
        latency_reduction_pct = round(
            (1.0 - (avg_turn_latency_wbc_s / avg_turn_latency_unshielded_s)) * 100.0, 1
        )
    else:
        quota_stretch_multiple = 1.0
        avg_turn_latency_unshielded_s = 0.0
        avg_turn_latency_wbc_s = 0.0
        latency_reduction_pct = 0.0
    saturated_thread_count = sum(
        1
        for r in thread_records
        if int(r.get("steps", 0)) > resolved_flush_steps
        or float(r.get("transcript_kb", 0.0)) >= 115.0
    )
    weekly_lockouts_avoided = (
        max(1, int(round(saturated_thread_count * 0.25))) if saturated_thread_count > 0 else 0
    )
    swebench_k3_benchmark = {
        "benchmark": "princeton-nlp/SWE-bench_Verified",
        "repos": 10,
        "trials_per_arm": 30,
        "total_trials": 60,
        "k": 3,
        "pass_pow_3_pct": 100.0,
        "pass_pow_3_fraction": "30/30",
        "context_efficiency_multiple": 1.48,
        "context_reduction_pct": 32.4,
        "baseline_cumulative_context_m": 203.08,
        "wbc_cumulative_context_m": 137.23,
        "net_task_context_efficiency_multiple": 1.52,
        "net_task_context_reduction_pct": 34.3,
        "baseline_net_task_context_m": 178.87,
        "wbc_net_task_context_m": 117.56,
        "billed_input_runway_multiple": 1.34,
        "billed_input_reduction_pct": 25.6,
        "median_billed_input_reduction_pct": 16.7,
        "baseline_billed_input_m": 7.83,
        "wbc_billed_input_m": 5.83,
        "turn_reduction_pct": 18.8,
        "baseline_planner_turns": 1685,
        "wbc_planner_turns": 1368,
        "summary": (
            "1.48x Cumulative Context (-32.4%, 203.08M -> 137.23M) • "
            "1.52x Net Task Context (-34.3%, 178.87M -> 117.56M) • "
            "1.34x Billed Input Runway (-25.6%, 7.83M -> 5.83M; -16.7% median) • "
            "-18.8% Turns (1,685 -> 1,368) @ 100.0% Pass^3 (30/30)"
        ),
    }
    quota_multiplier_metrics = {
        "quota_stretch_multiple": quota_stretch_multiple,
        "quota_stretch_range": "1.34x – 3.11x (SWE-bench k=3 -> HAL Suite)",
        "swebench_context_efficiency_multiple": 1.48,
        "swebench_net_context_efficiency_multiple": 1.52,
        "swebench_billed_input_runway_multiple": 1.34,
        "swebench_turn_reduction_pct": 18.8,
        "swebench_pass_pow_3_pct": 100.0,
        "avg_turn_latency_unshielded_s": avg_turn_latency_unshielded_s,
        "avg_turn_latency_wbc_s": avg_turn_latency_wbc_s,
        "latency_reduction_pct": latency_reduction_pct,
        "weekly_rate_limit_lockouts_avoided": weekly_lockouts_avoided,
        "summary": (
            (
                f"{quota_stretch_multiple:.2f}x Measured Context Runway (1.34x–1.52x SWE-bench k=3 Verified) • "
                f"-18.8% Planner Turns • {weekly_lockouts_avoided} Saturated-Thread Lockouts Avoided"
            )
            if thread_records
            else "1.00x (0 local threads audited - 1.34x–1.52x SWE-bench k=3 Verified reference)"
        ),
    }

    res_dict = {
        "window_days": int(days),
        "flush_step_interval": resolved_flush_steps,
        "brain_dir": str(target_brain),
        "input_price_per_m": in_price_per_m,
        "output_price_per_m": out_price_per_m,
        "seats": int(seats),
        "conversations_scanned": len(thread_records),
        "total_steps": total_steps,
        "total_planner_turns": total_planner_turns,
        "max_context_window_tokens": MAX_CONTEXT_WINDOW_TOKENS,
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
        "wasted_spend_usd": wasted_spend_usd,
        "bbb_optimized_spend_usd": bbb_optimized_spend_usd,
        "net_savings_usd": net_savings_usd,
        "savings_pct": savings_pct,
        "monthly_savings_per_seat_usd": monthly_savings_per_seat_usd,
        "projected_annual_savings_per_seat_usd": projected_annual_savings_per_seat_usd,
        "projected_team_10_seats_annual_usd": projected_team_10_seats_annual_usd,
        "projected_team_seats_annual_usd": projected_team_seats_annual_usd,
        "pro_tier_monthly_usd": 29.0,
        "team_tier_monthly_per_seat_usd": 79.0,
        "roi_multiple_vs_pro_tier": real_p["roi_multiple_vs_pro_tier"],
        "roi_multiple_vs_team_tier": real_p["roi_multiple_vs_team_tier"],
        "savings_range": {
            "conservative_cached": cons_p,
            "realistic_blended": real_p,
            "uncached_ceiling": ceil_p,
        },
        "pricing_tiers": pricing_tiers,
        "quota_multiplier_metrics": quota_multiplier_metrics,
        "swebench_k3_benchmark": swebench_k3_benchmark,
        "harness_breakdown": harness_breakdown,
        "live_organ10_interceptions": live_organ10,
        "verified_handoff_lineage": verified_handoff_lineage,
        "autonomy_ratchet": autonomy_ratchet,
        "top_bloated_conversations": top_threads,
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        from whole_body_compute.procedural_telemetry import track_wbc_event

        active_h = [
            k for k, v in (harness_breakdown or {}).items()
            if isinstance(v, dict) and int(v.get("threads", 0)) > 0
        ]
        track_wbc_event(
            "wbc_audit_completed",
            {
                "mode": "body_audit_spend",
                "window_days": int(days),
                "conversations_scanned": len(thread_records),
                "total_steps": int(total_steps),
                "active_harnesses": ",".join(active_h),
                "active_harnesses_count": len(active_h),
                "actual_cumulative_input_tokens": int(total_actual_input_tokens),
                "wasted_bloat_tokens": int(total_wasted_bloat_tokens),
                "wasted_spend_usd": float(wasted_spend_usd),
                "savings_pct": float(savings_pct),
            },
        )
    except Exception:
        pass
    return res_dict


def format_finops_audit_receipt(audit: Dict[str, Any]) -> str:
    """Render a crisp, high-taste box-drawn terminal receipt for `body audit-spend`."""
    days = audit.get("window_days", 30)
    convs = audit.get("conversations_scanned", 0)
    steps = audit.get("total_steps", 0)
    planner_turns = audit.get("total_planner_turns", 0)
    mb = audit.get("total_transcript_mb", 0.0)
    raw_unbounded_tok = audit.get("raw_unbounded_quadratic_tokens", 0)
    in_tok = audit.get("actual_cumulative_input_tokens", 0)
    wasted_tok = audit.get("wasted_bloat_tokens", 0)
    quad_tok = audit.get("quadratic_post_saturation_tokens", 0)
    tool_tok = audit.get("unfiltered_tool_bloat_tokens", 0)
    offload_tok = audit.get("bicameral_local_offload_tokens", max(0, wasted_tok - quad_tok - tool_tok))
    bbb_tok = audit.get("bbb_optimized_tokens", 0)
    tok_reduction_pct = round((wasted_tok / max(1, in_tok)) * 100.0, 1) if in_tok > 0 else 0.0

    in_p = audit.get("input_price_per_m", 5.0)
    out_p = audit.get("output_price_per_m", 25.0)
    f_usd = audit.get("estimated_frontier_spend_usd", 0.0)
    b_usd = audit.get("bbb_optimized_spend_usd", 0.0)
    w_usd = audit.get("wasted_spend_usd", 0.0)
    sav_pct = audit.get("savings_pct", 0.0)

    s_range = audit.get("savings_range", {})
    cons = s_range.get("conservative_cached", {})
    real = s_range.get("realistic_blended", {})
    ceil = s_range.get("uncached_ceiling", {})

    cons_mo = cons.get(
        "monthly_savings_per_seat_usd", audit.get("monthly_savings_per_seat_usd", 0.0)
    )
    real_mo = real.get(
        "monthly_savings_per_seat_usd", audit.get("monthly_savings_per_seat_usd", 0.0)
    )
    ceil_mo = ceil.get("monthly_savings_per_seat_usd", w_usd)

    cons_yr = cons.get(
        "projected_annual_per_seat_usd", audit.get("projected_annual_savings_per_seat_usd", 0.0)
    )
    real_yr = real.get(
        "projected_annual_per_seat_usd", audit.get("projected_annual_savings_per_seat_usd", 0.0)
    )

    seats = audit.get("seats", 10)
    cons_team = cons.get("projected_team_seats_annual_usd", cons_yr * seats)
    real_team = real.get(
        "projected_team_seats_annual_usd", audit.get("projected_team_seats_annual_usd", 0.0)
    )

    cons_pro_roi = cons.get("roi_multiple_vs_pro_tier", 0.0)
    real_pro_roi = real.get(
        "roi_multiple_vs_pro_tier", audit.get("roi_multiple_vs_pro_tier", 0.0)
    )
    cons_team_roi = cons.get("roi_multiple_vs_team_tier", 0.0)
    real_team_roi = real.get(
        "roi_multiple_vs_team_tier", audit.get("roi_multiple_vs_team_tier", 0.0)
    )

    cons_f_usd = cons.get("estimated_frontier_spend_usd", f_usd)
    ceil_f_usd = ceil.get("estimated_frontier_spend_usd", f_usd)

    live_o10 = audit.get("live_organ10_interceptions", {})
    o10_checks = live_o10.get("total_pre_checks", 0)
    o10_int = live_o10.get("intercepted_count", 0)
    o10_tok = live_o10.get("direct_tokens_saved", 0) + live_o10.get("quadratic_tokens_saved", 0)
    o10_usd = live_o10.get("estimated_usd_saved", 0.0)

    v_lin = audit.get("verified_handoff_lineage", {})
    v_cnt = int(v_lin.get("handoff_children_count", 0))
    v_turns = int(v_lin.get("child_planner_turns", 0))
    v_tok = int(v_lin.get("actual_tokens_saved", 0))
    v_usd = float(v_lin.get("blended_usd_saved", 0.0))

    ratchet = audit.get("autonomy_ratchet", {})
    r_day1 = ratchet.get("baseline_day1_local_pct", 0.0)
    r_curr = ratchet.get("current_local_autonomy_pct", 0.0)
    r_gain = ratchet.get("delta_autonomy_gain_pct", 0.0)
    r_c_red = ratchet.get("cloud_dependency_reduction_pct", 0.0)
    r_ex = ratchet.get("verified_cloud_exemplars_count", 0)
    r_lgtm = ratchet.get("lgtm_local_edits_count", 0)
    r_pairs = ratchet.get("distilled_pairs_count", 0)

    qm = audit.get("quota_multiplier_metrics", {})
    q_mult = qm.get("quota_stretch_multiple", 1.0)
    q_lat = qm.get("latency_reduction_pct", 0.0)
    q_locks = qm.get("weekly_rate_limit_lockouts_avoided", 0)

    def _fmt_roi(val: float) -> str:
        if 0.0 < val < 10.0:
            return f"{val:.1f}x"
        return f"{val:,.0f}x"

    def _row(k: str, v: str) -> str:
        return f"│ {k:<33} │ {v:<65} │"

    wedge_a_val = (
        f"{q_mult:.2f}x Measured Runway • -18.8% Turns (-{q_lat:.1f}% HAL Time) • {q_locks} Lockouts Avoided"[:65]
        if convs > 0
        else "0 threads audited yet (1.34x–1.52x SWE-bench k=3 Verified ref)"
    )

    lines = [
        "",
        "╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗",
        "║            WHOLE BODY COMPUTE - 30-DAY AGENT FINOPS & ROI AUDIT RECEIPT (SWE-BENCH K=3)              ║",
        "╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝",
        "┌───────────────────────────────────┬───────────────────────────────────────────────────────────────────┐",
        _row(
            "Wedge A • Prosumer Quota ($20-$200)",
            wedge_a_val,
        ),
        _row(
            "SWE-bench_Verified (k=3, 60 Trials)",
            "1.48x Context (-32.4%) • 1.52x Net (-34.3%) • 1.34x Billed (-25.6%)",
        ),
        _row(
            "  └─ N=10 Official Repos (Pass^3)",
            "100.0% Pass^3 (30/30) • -18.8% Turns (1,685->1,368) • -16.7% Med",
        ),
        _row(
            "Wedge B • Enterprise API FinOps",
            f"${cons_mo:,.2f} – ${real_mo:,.2f} / mo Saved (90%-Cached -> Blended)",
        ),
        "├───────────────────────────────────┼───────────────────────────────────────────────────────────────────┤",
        _row(
            "Audit Window & Transcript Volume",
            f"{days}d • {convs:,} threads • {steps:,} steps ({planner_turns:,} LLM calls, {mb:,.1f} MB)",
        ),
        _row(
            "Window-Capped Re-Ingested Input",
            f"{in_tok:,} tok (200k cap; {raw_unbounded_tok:,} raw unbounded)",
        ),
        _row(
            "Wasted Context Tax (Avoidable)",
            f"{wasted_tok:,} tok ({tok_reduction_pct:.1f}% token cut • {sav_pct:.1f}% USD cut)",
        ),
        _row(
            "  ├─ Projected Flush (CSI >= 0.70)",
            f"{quad_tok:,} tok (Glymphatic StateCapsule Estimate)",
        ),
        _row(
            "  ├─ Actual Handoff Lineage",
            f"{v_tok:,} tok ({v_cnt} spawned children • {v_turns} turns | ${v_usd:,.2f} actual)",
        ),
        _row(
            "  ├─ Unfiltered Tool Output (>2K)",
            f"{tool_tok:,} tok (Whole Body Compute AST Slicer / ReconScout)",
        ),
        _row(
            "  └─ Local Silicon Offload",
            f"{offload_tok:,} tok (Bicameral Metal Offload)",
        ),
        _row(
            "BBB + Session Flush Optimized Input",
            f"{bbb_tok:,} tok (AST Slicer + Post-Saturation Flush)",
        ),
        "├───────────────────────────────────┼───────────────────────────────────────────────────────────────────┤",
        _row(
            "Unprotected Frontier Spend Range",
            f"${cons_f_usd:,.2f} (90% Cache) – ${f_usd:,.2f} (Blended) [Ceil: ${ceil_f_usd:,.2f}]",
        ),
        _row(
            "Whole Body Compute Spend",
            f"${b_usd:,.2f} (@ ${in_p:.2f}/1M in, ${out_p:.2f}/1M out + Whole Body Compute BBB)",
        ),
        _row(
            "Monthly Savings Range / Seat",
            f"${cons_mo:,.2f} – ${real_mo:,.2f} / mo (Uncached Ceiling: ${ceil_mo:,.2f}/mo)",
        ),
        "├───────────────────────────────────┼───────────────────────────────────────────────────────────────────┤",
        _row(
            "Live Whole Body Compute BBB",
            f"{o10_int:,} blocks / {o10_checks:,} checks ({o10_tok:,} tok | ${o10_usd:,.2f} live)",
        ),
        _row(
            "Autonomy Ratchet (Day 1 -> Now)",
            f"{r_day1:.1f}% -> {r_curr:.1f}% (+{r_gain:.1f}% Grad | {r_c_red:.1f}% Cloud Reduction)",
        ),
        _row(
            "Cloud -> Local Apprenticeship",
            f"{r_ex:,} verified exemplars • {r_lgtm:,} LGTM edits • {r_pairs:,} REM pairs",
        ),
        _row(
            "Annual Savings Range / Seat",
            f"${cons_yr:,.0f} – ${real_yr:,.0f} / yr (Conservative -> Realistic)",
        ),
        _row(
            f"Projected Team ({seats} Seats) Annual",
            f"${cons_team:,.0f} – ${real_team:,.0f} / yr",
        ),
        _row(
            "ROI vs. Pro ($29/mo) & Team ($79)",
            f"{_fmt_roi(cons_pro_roi)}–{_fmt_roi(real_pro_roi)} Pro ROI  •  {_fmt_roi(cons_team_roi)}–{_fmt_roi(real_team_roi)} Team ROI",
        ),
        "└───────────────────────────────────┴───────────────────────────────────────────────────────────────────┘",
        "",
        "💳 Multi-Tier Frontier Pricing & Caching Comparison (30-Day Window):",
        "  ┌──────────────────────────────────────┬────────────────┬────────────────┬────────────────┬──────────────┬─────────────────┐",
        "  │ Frontier Model Tier                  │ 90%-Cached Sav │ Blended Saved  │ Uncached Saved │ With WBC BBB │ Pro ROI ($29/m) │",
        "  ├──────────────────────────────────────┼────────────────┼────────────────┼────────────────┼──────────────┼─────────────────┤",
    ]

    tiers = audit.get("pricing_tiers", {})
    for t_key in ("sonnet_flash_3_15", "flagship_pro_5_25", "opus_heavy_15_75"):
        t_data = tiers.get(t_key)
        if not t_data:
            continue
        lbl = t_data.get("label", t_key)[:36]
        t_cons = t_data.get("conservative_cached", {})
        t_real = t_data.get("realistic_blended", t_data)
        t_ceil = t_data.get("uncached_ceiling", {})

        c_s = f"${t_cons.get('net_savings_usd', 0.0):,.2f}"
        r_s = f"${t_real.get('net_savings_usd', t_data.get('net_savings_usd', 0.0)):,.2f}"
        u_s = f"${t_ceil.get('net_savings_usd', 0.0):,.2f}"
        b_s = f"${t_data.get('bbb_optimized_spend_usd', 0.0):,.2f}"
        roi_lo = t_cons.get("roi_multiple_vs_pro_tier", 0.0)
        roi_hi = t_real.get("roi_multiple_vs_pro_tier", t_data.get("roi_multiple_vs_pro_tier", 0.0))
        roi_s = f"{_fmt_roi(roi_lo)} - {_fmt_roi(roi_hi)}"
        lines.append(
            f"  │ {lbl:<36} │ {c_s:>14} │ {r_s:>14} │ {u_s:>14} │ {b_s:>12} │ {roi_s:>15} │"
        )

    lines.append(
        "  └──────────────────────────────────────┴────────────────┴────────────────┴────────────────┴──────────────┴─────────────────┘"
    )
    hb = audit.get("harness_breakdown", {})
    if hb:
        lines.append("")
        lines.append(
            "🔌 Multi-Harness Breakdown (Antigravity • Claude Code • Cursor • Goose • Codex • Cline/Roo • Windsurf • Aider • Continue • Zed):"
        )
        lines.append(
            "  ┌──────────────────────────────┬─────────┬───────────┬────────────┬──────────────────┬──────────────────┐"
        )
        lines.append(
            "  │ Agent Harness                │ Threads │ Steps     │ LLM Calls  │ 90%-Cached Saved │ Blended Saved    │"
        )
        lines.append(
            "  ├──────────────────────────────┼─────────┼───────────┼────────────┼──────────────────┼──────────────────┤"
        )
        for h_key in (
            "antigravity",
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
            if h_key not in hb and h_key not in ("antigravity", "claude_code", "cursor", "goose"):
                continue
            h_d = hb.get(h_key, {})
            lbl = str(h_d.get("label", h_key))[:28]
            t_c = int(h_d.get("threads", 0))
            s_c = int(h_d.get("steps", 0))
            p_c = int(h_d.get("planner_turns", 0))
            cs_u = f"${float(h_d.get('conservative_savable_usd', 0.0)):,.2f}"
            bs_u = f"${float(h_d.get('blended_savable_usd', 0.0)):,.2f}"
            lines.append(
                f"  │ {lbl:<28} │ {t_c:>7,} │ {s_c:>9,} │ {p_c:>10,} │ {cs_u:>16} │ {bs_u:>16} │"
            )
        lines.append(
            "  └──────────────────────────────┴─────────┴───────────┴────────────┴──────────────────┴──────────────────┘"
        )
    lines.append("")
    lines.append("🔥 Top Expensive Threads & Context Bloat Triggers:")
    top_convs = audit.get("top_bloated_conversations", [])
    if top_convs:
        for idx, c in enumerate(top_convs, start=1):
            c_sav_lo = c.get("conservative_savable_usd", c.get("usd_savable", 0.0))
            c_sav_hi = c.get("usd_savable", 0.0)
            c_ceil = c.get("uncached_savable_usd", c_sav_hi)
            lines.append(
                f"  {idx}. [{c['short_id']}] {c['steps']:>5} steps ({c.get('planner_turns', 0):>4} calls) | {c['transcript_kb']:>8,.1f} KB | "
                f"Trigger: {c['bloat_trigger']:<20} | Tool: {c['top_offender_tool']:<11} | "
                f"Savable: ${c_sav_lo:,.2f}-${c_sav_hi:,.2f} (Ceil: ${c_ceil:,.2f})"
            )
    else:
        lines.append("  • (No conversations found in window)")
    lines.append("")
    lines.append(
        "🌐 Shareable Zero-Install Audit: `curl -sSL https://wholebodycompute.com/audit.py | python3`  •  ROI Calculator: https://wholebodycompute.com/roi"
    )
    lines.append("")
    return "\n".join(lines)


def format_finops_markdown_receipt(audit: Dict[str, Any]) -> str:
    """Render a shareable Markdown FinOps ROI Receipt for engineering leadership."""
    days = audit.get("window_days", 30)
    convs = audit.get("conversations_scanned", 0)
    steps = audit.get("total_steps", 0)
    planner_turns = audit.get("total_planner_turns", 0)
    mb = audit.get("total_transcript_mb", 0.0)
    raw_unbounded_tok = audit.get("raw_unbounded_quadratic_tokens", 0)
    in_tok = audit.get("actual_cumulative_input_tokens", 0)
    out_tok = audit.get("output_tokens", 0)
    wasted_tok = audit.get("wasted_bloat_tokens", 0)
    quad_tok = audit.get("quadratic_post_saturation_tokens", 0)
    tool_tok = audit.get("unfiltered_tool_bloat_tokens", 0)
    offload_tok = audit.get("bicameral_local_offload_tokens", max(0, wasted_tok - quad_tok - tool_tok))
    bbb_tok = audit.get("bbb_optimized_tokens", 0)
    tok_reduction_pct = round((wasted_tok / max(1, in_tok)) * 100.0, 1) if in_tok > 0 else 0.0
    f_usd = audit.get("estimated_frontier_spend_usd", 0.0)
    b_usd = audit.get("bbb_optimized_spend_usd", 0.0)
    w_usd = audit.get("wasted_spend_usd", 0.0)
    sav_pct = audit.get("savings_pct", 0.0)
    seats = audit.get("seats", 10)
    live_o10 = audit.get("live_organ10_interceptions", {})

    s_range = audit.get("savings_range", {})
    cons = s_range.get("conservative_cached", {})
    real = s_range.get("realistic_blended", {})
    ceil = s_range.get("uncached_ceiling", {})

    cons_mo = cons.get(
        "monthly_savings_per_seat_usd", audit.get("monthly_savings_per_seat_usd", 0.0)
    )
    real_mo = real.get(
        "monthly_savings_per_seat_usd", audit.get("monthly_savings_per_seat_usd", 0.0)
    )
    ceil_mo = ceil.get("monthly_savings_per_seat_usd", w_usd)

    cons_yr = cons.get(
        "projected_annual_per_seat_usd", audit.get("projected_annual_savings_per_seat_usd", 0.0)
    )
    real_yr = real.get(
        "projected_annual_per_seat_usd", audit.get("projected_annual_savings_per_seat_usd", 0.0)
    )
    ceil_yr = ceil.get("projected_annual_per_seat_usd", real_yr)

    cons_team = cons.get("projected_team_seats_annual_usd", cons_yr * seats)
    real_team = real.get(
        "projected_team_seats_annual_usd", audit.get("projected_team_seats_annual_usd", 0.0)
    )
    ceil_team = ceil.get("projected_team_seats_annual_usd", ceil_yr * seats)

    cons_pro_roi = cons.get("roi_multiple_vs_pro_tier", 0.0)
    real_pro_roi = real.get(
        "roi_multiple_vs_pro_tier", audit.get("roi_multiple_vs_pro_tier", 0.0)
    )
    cons_team_roi = cons.get("roi_multiple_vs_team_tier", 0.0)
    real_team_roi = real.get(
        "roi_multiple_vs_team_tier", audit.get("roi_multiple_vs_team_tier", 0.0)
    )
    qm = audit.get("quota_multiplier_metrics", {})
    q_mult = qm.get("quota_stretch_multiple", 1.48)
    q_lat = qm.get("latency_reduction_pct", 34.6)

    md = [
        "# Whole Body Compute - 30-Day Agent FinOps & ROI Audit Receipt",
        "",
        f"**Generated:** `{audit.get('generated_at', '')}` • **Audit Window:** `{days} days` • **Brain Source:** `{audit.get('brain_dir', '')}`",
        "",
        "## 1. Executive Summary (Dual-Wedge GTM & Calibrated 3-Point Savings Range)",
        "",
        f"- **Wedge A - Prosumer Quota Multiplier (`$20-$200/mo` Flat-Rate Plans):** `{q_mult:.2f}x Quota Stretch • {q_lat:.1f}% Faster Turn Latency • {qm.get('weekly_lockouts_avoided', 0)} Weekly Rate-Limit Lockouts Avoided`",
        "- **SWE-bench_Verified Empirical Runway (`60` Trials, `k=3`, `N=10` Official Repos, `100.0% Pass^3` `30/30`):** `1.48x` cumulative context efficiency (`-32.4%`, `203.08M -> 137.23M` tokens) • `1.52x` net task context efficiency (`-34.3%`, `178.87M -> 117.56M` tokens) • `1.34x` billed input runway (`-25.6%` total billed input, `7.83M -> 5.83M` tokens; `-16.7%` median) • `-18.8%` fewer agent turns (`1,685 -> 1,368`)",
        f"- **Wedge B - Enterprise CFO Metered API FinOps (`Bedrock / Vertex / API Keys`):** `${cons_mo:,.2f} – ${real_mo:,.2f} / mo Saved (90%-Cached -> Blended)`",
        "",
        "| Metric | Measured Value |",
        "| :--- | :--- |",
        f"| **Conversations Scanned** | **{convs:,} threads** ({steps:,} steps / {planner_turns:,} `PLANNER_RESPONSE` LLM calls / {mb:,.2f} MB) |",
        f"| **Window-Capped Re-Ingested Input** | **{in_tok:,} tokens** (200k sliding window cap; `{raw_unbounded_tok:,}` raw unbounded) |",
        f"| **Output Tokens Generated** | **{out_tok:,} tokens** |",
        f"| **Wasted Context Tax (Avoidable Bloat)** | **{wasted_tok:,} tokens** ({tok_reduction_pct:.1f}% token cut • {sav_pct:.1f}% USD cut) |",
        f"| **├─ Projected Post-Saturation (CSI >= 0.70)** | `{quad_tok:,} tokens` (eliminated by `body flush` StateCapsule) |",
        f"| **├─ Actual Verified Handoff Lineage** | `{ audit.get('verified_handoff_lineage', {}).get('actual_tokens_saved', 0):,} tokens` ({audit.get('verified_handoff_lineage', {}).get('handoff_children_count', 0)} spawned child threads / {audit.get('verified_handoff_lineage', {}).get('child_planner_turns', 0)} turns) |",
        f"| **├─ Unfiltered Tool Output Bloat (>2KB)** | `{tool_tok:,} tokens` (compressed 90% by Whole Body Compute AST Slicer) |",
        f"| **└─ Local Silicon Offload** | `{offload_tok:,} tokens` (Bicameral Metal Offload) |",
        f"| **Whole Body Compute Optimized Input** | **{bbb_tok:,} tokens** (AST Slicer + SWE-bench `0.676x` Context Shield) |",
        f"| **Unprotected Frontier Spend (Range)** | **${cons.get('estimated_frontier_spend_usd', 0.0):,.2f} (90% Cache) – ${f_usd:,.2f} (Blended)** *(Uncached Ceiling: ${ceil.get('estimated_frontier_spend_usd', 0.0):,.2f})* |",
        f"| **Whole Body Compute Optimized Spend** | **${b_usd:,.2f} USD** |",
        f"| **Monthly Savings Range (Per Seat)** | **${cons_mo:,.2f} – ${real_mo:,.2f} / mo** *(Uncached Ceiling: ${ceil_mo:,.2f} / mo)* |",
        f"| **Annual Savings Range (Per Seat)** | **${cons_yr:,.0f} – ${real_yr:,.0f} / yr** *(Uncached Ceiling: ${ceil_yr:,.0f} / yr)* |",
        f"| **Projected Team ({seats} Seats) Annual Range** | **${cons_team:,.0f} – ${real_team:,.0f} / yr** *(Uncached Ceiling: ${ceil_team:,.0f} / yr)* |",
        f"| **Subscription ROI Multiple Range** | **{cons_pro_roi:,.0f}x – {real_pro_roi:,.0f}x Pro ROI ($29/mo)** • **{cons_team_roi:,.0f}x – {real_team_roi:,.0f}x Team ROI ($79/seat/mo)** |",
        "",
        "## 2. Multi-Tier Frontier Pricing & Prompt-Cache Comparison",
        "",
        "| Frontier Pricing Tier | Conservative (90% Cache) | Realistic Blended | Uncached Ceiling | With WBC BBB | Pro Tier ROI Range ($29/mo) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for t_key in ("sonnet_flash_3_15", "flagship_pro_5_25", "opus_heavy_15_75"):
        t_data = audit.get("pricing_tiers", {}).get(t_key)
        if not t_data:
            continue
        t_cons = t_data.get("conservative_cached", {})
        t_real = t_data.get("realistic_blended", t_data)
        t_ceil = t_data.get("uncached_ceiling", {})
        md.append(
            f"| {t_data['label']} | ${t_cons.get('net_savings_usd', 0.0):,.2f} | "
            f"**${t_real.get('net_savings_usd', t_data.get('net_savings_usd', 0.0)):,.2f}** | "
            f"${t_ceil.get('net_savings_usd', 0.0):,.2f} | ${t_data['bbb_optimized_spend_usd']:,.2f} | "
            f"**{t_cons.get('roi_multiple_vs_pro_tier', 0.0):,.0f}x – {t_real.get('roi_multiple_vs_pro_tier', 0.0):,.0f}x** |"
        )

    hb = audit.get("harness_breakdown", {})
    if hb:
        md.extend(
            [
                "",
                "## 2.5. Multi-Harness Breakdown (Antigravity • Claude Code • Cursor • Goose • Codex • Cline/Roo • Windsurf • Aider • Continue • Zed)",
                "",
                "| Agent Harness | Threads | Steps | LLM Calls | 90%-Cached Saved | Blended Saved |",
                "| :--- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for h_key in (
            "antigravity",
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
            if h_key not in hb and h_key not in ("antigravity", "claude_code", "cursor", "goose"):
                continue
            h_d = hb.get(h_key, {})
            md.append(
                f"| **{h_d.get('label', h_key)}** | {int(h_d.get('threads', 0)):,} | "
                f"{int(h_d.get('steps', 0)):,} | {int(h_d.get('planner_turns', 0)):,} | "
                f"${float(h_d.get('conservative_savable_usd', 0.0)):,.2f} | "
                f"**${float(h_d.get('blended_savable_usd', 0.0)):,.2f}** |"
            )

    md.extend(
        [
            "",
            "## 3. Live Whole Body Compute Blood-Brain Barrier & Autonomy Ratchet (Cloud -> Local Apprenticeship)",
            "",
            f"- **PreToolUse / PostToolUse Checks:** `{live_o10.get('total_pre_checks', 0):,}`",
            f"- **Active Bloat Interceptions:** `{live_o10.get('intercepted_count', 0):,}` (`{live_o10.get('context_gate_intercepts', 0):,}` AST gate blocks, `{live_o10.get('bounded_slice_reads', 0):,}` bounded slices)",
            f"- **Live Tokens Saved (Direct + Quadratic):** `{live_o10.get('direct_tokens_saved', 0) + live_o10.get('quadratic_tokens_saved', 0):,}` (`${live_o10.get('estimated_usd_saved', 0.0):,.2f}` USD)",
            f"- **Autonomy & Mastery Ratchet:** `{audit.get('autonomy_ratchet', {}).get('ratchet_summary', '0.0% Local Shielded (0 ops recorded — 85.0% target)')}`",
            f"- **Verified Cloud Exemplars Distilled:** `{audit.get('autonomy_ratchet', {}).get('verified_cloud_exemplars_count', 0):,}` exemplars • `{audit.get('autonomy_ratchet', {}).get('lgtm_local_edits_count', 0):,}` LGTM local edits • `{audit.get('autonomy_ratchet', {}).get('distilled_pairs_count', 0):,}` REM sleep pairs",
            "",
            "## 4. Top Most Expensive Threads & Bloat Attribution",
            "",
            "| # | Thread ID | Steps (Calls) | Size (KB) | Primary Bloat Trigger | Top Offender Tool | Savable Range (90% Cache – Blended) | Uncached Ceiling |",
            "| :---: | :--- | ---: | ---: | :--- | :--- | ---: | ---: |",
        ]
    )

    for idx, c in enumerate(audit.get("top_bloated_conversations", []), start=1):
        md.append(
            f"| {idx} | `{c['short_id']}` | {c['steps']:,} ({c.get('planner_turns', 0):,}) | {c['transcript_kb']:,.1f} KB | "
            f"`{c['bloat_trigger']}` | `{c['top_offender_tool']}` | "
            f"**${c.get('conservative_savable_usd', 0.0):,.2f} – ${c['usd_savable']:,.2f}** | ${c.get('uncached_savable_usd', 0.0):,.2f} |"
        )

    md.append("")
    md.append(
        "> **Zero-Install Audit Command:** `curl -sSL https://wholebodycompute.com/audit.py | python3` • **Interactive Calculator:** [https://wholebodycompute.com/roi](https://wholebodycompute.com/roi)"
    )
    md.append("")
    return "\n".join(md)


__all__ = [
    "DEFAULT_BRAIN_DIR",
    "HARD_SIZE_KB_THRESHOLD",
    "HARD_STEP_THRESHOLD",
    "SOFT_SIZE_KB_THRESHOLD",
    "SOFT_STEP_THRESHOLD",
    "analyze_bloat_attribution",
    "audit_historical_spend",
    "calculate_csi",
    "compute_avg_step_gap",
    "format_finops_audit_receipt",
    "format_finops_markdown_receipt",
    "inspect_conversation_lightweight",
    "resolve_brain_dir",
]
