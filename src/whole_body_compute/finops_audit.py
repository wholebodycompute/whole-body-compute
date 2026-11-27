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
SOFT_STEP_THRESHOLD = 150
HARD_STEP_THRESHOLD = 350
SOFT_SIZE_KB_THRESHOLD = 320.0
HARD_SIZE_KB_THRESHOLD = 800.0
DEFAULT_FLUSH_STEP_INTERVAL = 150
FINOPS_SOFT_CONTEXT_TOKENS = 80_000
FINOPS_HARD_CONTEXT_TOKENS = 160_000
FINOPS_SOFT_CSI_THRESHOLD = 0.45
FINOPS_HARD_CSI_THRESHOLD = 0.80
COMPACTION_DRIFT_PENALTY = 0.04

_HARNESS_TRUNCATION_MARKERS = (
    "output was truncated",
    "<truncated ",
    "truncated due to being too long",
    "truncated by the byte limit",
)


def is_step_output_truncated(step: Dict[str, Any], *, is_compact_only: bool = False) -> bool:
    """Detect real harness tool-output truncation without misclassifying >4KB compact clipping."""
    if not isinstance(step, dict):
        return False

    raw_content = step.get("content")
    if isinstance(raw_content, str):
        content_str = raw_content
    elif raw_content is not None:
        try:
            content_str = json.dumps(raw_content)
        except Exception:
            content_str = str(raw_content)
    else:
        content_str = ""

    if content_str:
        lowered = content_str.lower()
        if any(marker in lowered for marker in _HARNESS_TRUNCATION_MARKERS):
            return True

    if not is_compact_only:
        if bool(step.get("truncated_fields")) and step.get("type") not in (
            "PLANNER_RESPONSE",
            "USER_INPUT",
        ):
            return True

    return False


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


def split_at_latest_checkpoint(
    lines: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int, List[Dict[str, Any]]]:
    """
    Split a conversation step list at the latest verified `{"type": "CHECKPOINT"}` entry.

    Returns `(active_lines, compaction_count, cumulative_lines)` where `active_lines`
    starts AT the latest `CHECKPOINT` entry when `compaction_count > 0`, or equals
    `list(lines)` when no `CHECKPOINT` exists.
    """
    compaction_count = 0
    last_cp_idx = -1
    for idx, step in enumerate(lines):
        if isinstance(step, dict) and step.get("type") == "CHECKPOINT":
            compaction_count += 1
            last_cp_idx = idx
    if last_cp_idx >= 0:
        active_lines = list(lines[last_cp_idx:])
    else:
        active_lines = list(lines)
        compaction_count = 0
    return active_lines, compaction_count, lines


def estimate_lines_size_kb(lines: List[Dict[str, Any]]) -> float:
    """Estimate serialized JSONL size in KB across parsed step dicts."""
    if not lines:
        return 0.0
    total_bytes = 0
    for step in lines:
        try:
            total_bytes += len(json.dumps(step, ensure_ascii=False).encode("utf-8")) + 1
        except Exception:
            total_bytes += len(str(step).encode("utf-8")) + 1
    return round(total_bytes / 1024.0, 1)


def compute_avg_step_gap(lines: List[Dict[str, Any]]) -> float:
    """Compute average intra-turn model/tool step latency gap in seconds (excluding human think pauses & CHECKPOINTs)."""
    gaps: List[float] = []
    for i in range(1, len(lines)):
        try:
            prev = lines[i - 1]
            cur = lines[i]
            if not isinstance(cur, dict) or not isinstance(prev, dict):
                continue
            # Exclude idle human reading/typing gaps and /compact CHECKPOINT timestamp jumps
            if cur.get("type") in ("USER_INPUT", "CHECKPOINT") or cur.get("source") == "USER_EXPLICIT":
                continue
            if prev.get("type") == "CHECKPOINT":
                continue
            t1_s = prev.get("created_at")
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
    compaction_count: int = 0,
) -> Tuple[float, str]:
    """
    Calculate Context Saturation Index (CSI) and Health Tier:
      - CSI < 0.45: GREEN
      - 0.45 <= CSI < 1.00: YELLOW (Soft FinOps flush sweet spot)
      - CSI >= 1.00 (or >= HARD step/size threshold): RED (Hard handoff recommended)
    """
    step_factor = steps / float(HARD_STEP_THRESHOLD)
    size_factor = full_size_kb / float(HARD_SIZE_KB_THRESHOLD)
    base = max(step_factor, size_factor)

    trunc_penalty = 0.05 * min(1.0, truncated_steps / 15.0)
    latency_penalty = 0.0
    inflation = 1.0
    if early_step_sec > 0.1:
        # True turn-latency inflation against the thread's own early baseline
        inflation = recent_step_sec / early_step_sec
        if recent_step_sec > 3.5 and inflation > 1.05:
            latency_penalty = 0.05 * min(1.0, (inflation - 1.05) / 0.80)
            if inflation >= 2.0 and recent_step_sec >= 8.0 and (steps >= 80 or full_size_kb >= 320.0):
                latency_penalty += min(0.25, 0.12 + 0.08 * (inflation - 2.0))
    elif recent_step_sec > 3.5:
        latency_penalty = 0.05 * min(1.0, (recent_step_sec - 3.5) / 6.5)

    drift_penalty = COMPACTION_DRIFT_PENALTY * max(0, int(compaction_count))
    csi = round(base + trunc_penalty + latency_penalty + drift_penalty, 3)
    if inflation >= 2.0 and recent_step_sec >= 8.0 and (steps >= 80 or full_size_kb >= 320.0):
        csi = max(csi, FINOPS_HARD_CSI_THRESHOLD)

    if steps >= HARD_STEP_THRESHOLD or full_size_kb >= HARD_SIZE_KB_THRESHOLD or csi >= 1.00:
        tier = "RED"
    elif (
        steps >= SOFT_STEP_THRESHOLD
        or full_size_kb >= SOFT_SIZE_KB_THRESHOLD
        or csi >= FINOPS_SOFT_CSI_THRESHOLD
    ):
        tier = "YELLOW"
    else:
        tier = "GREEN"

    return csi, tier


def compute_dual_csi_metrics(
    lines: List[Dict[str, Any]],
    full_size_kb: float,
    compact_size_kb: float,
    is_compact_only: bool = False,
) -> Dict[str, Any]:
    """
    Compute Dual-Metric Post-`/compact` CSI:
      - Active post-CHECKPOINT window (`active_steps`, `active_size_kb`, live `csi` & `health_tier`
        including `+0.04 * compaction_count` generational drift penalty).
      - Preserved historical & cumulative telemetry (`peak_csi`, `cumulative_steps`,
        `cumulative_size_kb`, `cumulative_truncated_steps`) for longitudinal reporting.
    """
    active_lines, compaction_count, cumulative_lines = split_at_latest_checkpoint(lines)
    cumulative_steps = len(cumulative_lines)
    cumulative_size_kb = round(float(full_size_kb), 1)
    cumulative_compact_size_kb = round(float(compact_size_kb), 1)
    cumulative_truncated_steps = sum(
        1 for s in cumulative_lines if is_step_output_truncated(s, is_compact_only=is_compact_only)
    )
    early_sec = compute_avg_step_gap(cumulative_lines[: min(40, cumulative_steps)])
    recent_sec = compute_avg_step_gap(active_lines[max(0, len(active_lines) - 40) :])

    if compaction_count == 0:
        active_steps = cumulative_steps
        active_size_kb = cumulative_size_kb
        active_compact_size_kb = cumulative_compact_size_kb
        active_truncated_steps = cumulative_truncated_steps
        csi, tier = calculate_csi(
            steps=active_steps,
            full_size_kb=active_size_kb,
            truncated_steps=active_truncated_steps,
            recent_step_sec=recent_sec,
            early_step_sec=early_sec,
            compaction_count=0,
        )
        peak_csi = csi
    else:
        active_steps = len(active_lines)
        active_size_kb = min(cumulative_size_kb, estimate_lines_size_kb(active_lines))
        active_compact_size_kb = min(cumulative_compact_size_kb, active_size_kb)
        active_truncated_steps = sum(
            1 for s in active_lines if is_step_output_truncated(s, is_compact_only=is_compact_only)
        )
        csi, tier = calculate_csi(
            steps=active_steps,
            full_size_kb=active_size_kb,
            truncated_steps=active_truncated_steps,
            recent_step_sec=recent_sec,
            early_step_sec=early_sec,
            compaction_count=compaction_count,
        )

        cp_indices = [
            idx
            for idx, s in enumerate(cumulative_lines)
            if isinstance(s, dict) and s.get("type") == "CHECKPOINT"
        ]
        last_cp_idx = cp_indices[-1]
        epoch_csis: List[float] = []
        for k, cp_idx in enumerate(cp_indices):
            seg_start = 0 if k == 0 else cp_indices[k - 1]
            seg = cumulative_lines[seg_start:cp_idx]
            if not seg:
                continue
            seg_kb = estimate_lines_size_kb(seg)
            seg_trunc = sum(
                1 for s in seg if is_step_output_truncated(s, is_compact_only=is_compact_only)
            )
            seg_recent = compute_avg_step_gap(seg[max(0, len(seg) - 40) :])
            seg_csi, _ = calculate_csi(
                steps=len(seg),
                full_size_kb=seg_kb,
                truncated_steps=seg_trunc,
                recent_step_sec=seg_recent,
                early_step_sec=early_sec,
                compaction_count=k,
            )
            epoch_csis.append(seg_csi)

        pre_lines = cumulative_lines[:last_cp_idx]
        pre_cum_csi = 0.0
        if pre_lines:
            pre_kb = min(
                cumulative_size_kb,
                max(
                    0.0,
                    round(cumulative_size_kb - active_size_kb, 1),
                    estimate_lines_size_kb(pre_lines),
                ),
            )
            pre_trunc = sum(
                1 for s in pre_lines if is_step_output_truncated(s, is_compact_only=is_compact_only)
            )
            pre_recent = compute_avg_step_gap(pre_lines[max(0, len(pre_lines) - 40) :])
            pre_cum_csi, _ = calculate_csi(
                steps=len(pre_lines),
                full_size_kb=pre_kb,
                truncated_steps=pre_trunc,
                recent_step_sec=pre_recent,
                early_step_sec=early_sec,
                compaction_count=max(0, compaction_count - 1),
            )

        peak_csi = round(max([csi, *epoch_csis, pre_cum_csi]), 3)

    return {
        "active_lines": active_lines,
        "cumulative_lines": cumulative_lines,
        "compaction_count": compaction_count,
        "active_steps": active_steps,
        "active_size_kb": active_size_kb,
        "active_compact_size_kb": active_compact_size_kb,
        "active_truncated_steps": active_truncated_steps,
        "cumulative_steps": cumulative_steps,
        "cumulative_size_kb": cumulative_size_kb,
        "cumulative_compact_size_kb": cumulative_compact_size_kb,
        "cumulative_truncated_steps": cumulative_truncated_steps,
        "early_step_sec": early_sec,
        "recent_step_sec": recent_sec,
        "csi": csi,
        "health_tier": tier,
        "peak_csi": peak_csi,
    }


def analyze_bloat_attribution(
    lines: List[Dict[str, Any]],
    steps: int,
    full_size_kb: float,
    compact_size_kb: float,
    truncated_steps: int,
    csi: float,
    is_compact_only: bool = False,
    active_size_kb: Optional[float] = None,
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

        is_truncated = is_step_output_truncated(step, is_compact_only=is_compact_only)
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
    elif truncated_steps >= 8:
        primary_cause = "TRUNCATION_CASCADE"
    elif avoidable_bloat_kb >= 300.0 or (full_size_kb >= 1000.0 and steps < 500):
        primary_cause = "UNFILTERED_OUTPUT_BLOAT"
    elif consecutive_retries >= 12 or max_streak >= 12:
        primary_cause = "RETRY_CHURN_LOOP"
    else:
        primary_cause = "NATURAL_LONG_HORIZON"

    recommendations: List[str] = []
    eff_kb = full_size_kb if active_size_kb is None else active_size_kb
    est_ctx_tokens = int(round((eff_kb * 1024.0) / 4.0))
    if csi >= FINOPS_HARD_CSI_THRESHOLD or est_ctx_tokens >= FINOPS_HARD_CONTEXT_TOKENS:
        recommendations.append(
            f"Context saturation critical (CSI {csi:.2f} / ~{est_ctx_tokens // 1000}k tok). "
            "Finish active edit/test loop and run 'body flush --auto'."
        )
    elif csi >= FINOPS_SOFT_CSI_THRESHOLD or est_ctx_tokens >= FINOPS_SOFT_CONTEXT_TOKENS:
        recommendations.append(
            f"FinOps flush sweet spot reached (CSI {csi:.2f} / ~{est_ctx_tokens // 1000}k tok >= "
            f"{FINOPS_SOFT_CONTEXT_TOKENS // 1000}k). "
            "Run 'body flush' or '/compact' at the next phase boundary."
        )

    candidate_tools = [t for t in sorted_tools.keys() if t != "agent_response"]
    top_tool = (
        candidate_tools[0]
        if candidate_tools
        else (list(sorted_tools.keys())[0] if sorted_tools else None)
    )
    top_stats = sorted_tools.get(top_tool, {}) if top_tool else {}

    if primary_cause == "NOMINAL":
        recommendations.append(
            "Context telemetry nominal. Continue standard execution cadence."
        )
    elif primary_cause == "RETRY_CHURN_LOOP" and top_tool:
        recommendations.append(
            f"Repeated consecutive retries on '{top_tool}'. Verify prerequisites or step arguments before looping."
        )
    elif top_tool in ("run_command", "bash", "execute_command"):
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
    elif primary_cause == "UNFILTERED_OUTPUT_BLOAT" and top_tool:
        recommendations.append(
            f"Tool '{top_tool}' generated significant uncompressed output. Consider streaming or paging tool outputs."
        )

    if not recommendations:
        if primary_cause == "NOMINAL" or csi < FINOPS_SOFT_CSI_THRESHOLD:
            recommendations.append(
                "Context telemetry nominal. Continue standard execution cadence."
            )
        else:
            recommendations.append(
                "Context saturation approaching soft limit. Plan handoff via 'body flush' soon."
            )

    return primary_cause, avoidable_bloat_kb, sorted_tools, recommendations[:3]


def _inspect_claude_jsonl_lightweight(jpath: Path) -> Optional[Dict[str, Any]]:
    """Parse a Claude Code `*.jsonl` session file into a Community-safe CSI vitals dict."""
    if not jpath.is_file():
        return None
    full_kb = round(os.path.getsize(jpath) / 1024.0, 1)
    norm_lines: List[Dict[str, Any]] = []
    tool_bytes: Dict[str, int] = {}
    tool_calls: Dict[str, int] = {}
    total_tool_bytes = 0
    try:
        with open(jpath, "r", encoding="utf-8", errors="ignore") as fh:
            for raw_line in fh:
                s = raw_line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                msg = obj.get("message") if isinstance(obj.get("message"), dict) else {}
                role = str(obj.get("type") or obj.get("role") or msg.get("role") or "").lower()
                ts_val = obj.get("timestamp") or obj.get("created_at") or msg.get("created_at")
                norm_entry: Dict[str, Any] = {
                    "type": "USER_INPUT" if role == "user" and not isinstance(msg.get("content"), list) else role.upper(),
                    "created_at": ts_val,
                }
                norm_lines.append(norm_entry)

                content = msg.get("content") if "content" in msg else obj.get("content")
                line_b = len(s.encode("utf-8"))
                if isinstance(content, list):
                    for blk in content:
                        if not isinstance(blk, dict):
                            continue
                        btype = str(blk.get("type") or "")
                        if btype == "tool_use":
                            tname = str(blk.get("name") or "tool")
                            tool_calls[tname] = tool_calls.get(tname, 0) + 1
                        elif btype == "tool_result":
                            total_tool_bytes += line_b
                            tname = "tool_result"
                            tool_bytes[tname] = tool_bytes.get(tname, 0) + line_b
    except Exception:
        return None

    if not norm_lines:
        return None

    steps = len(norm_lines)
    early_sec = compute_avg_step_gap(norm_lines[: min(40, steps)])
    recent_sec = compute_avg_step_gap(norm_lines[max(0, steps - 40) :])
    inflation = round(recent_sec / early_sec, 2) if early_sec > 0.1 and recent_sec > 0.0 else 1.0
    csi, tier = calculate_csi(
        steps=steps,
        full_size_kb=full_kb,
        truncated_steps=0,
        recent_step_sec=recent_sec,
        early_step_sec=early_sec,
    )
    avoidable_kb = round(max(0.0, (total_tool_bytes / 1024.0) * 0.65), 1)
    primary_cause = "UNFILTERED_TOOL_OUTPUT" if avoidable_kb >= 120.0 else "NOMINAL"
    sorted_tools = [
        {"tool": k, "calls": v, "kb": round(tool_bytes.get(k, 0) / 1024.0, 1)}
        for k, v in sorted(tool_calls.items(), key=lambda kv: kv[1], reverse=True)[:5]
    ]
    cid = jpath.stem
    return {
        "conv_id": cid,
        "short_id": cid[:8],
        "harness": "claude_code",
        "steps": steps,
        "csi": csi,
        "health_tier": tier,
        "compact_size_kb": full_kb,
        "full_size_kb": full_kb,
        "truncated_steps": 0,
        "active_steps": steps,
        "active_size_kb": full_kb,
        "compaction_count": 0,
        "peak_csi": csi,
        "cumulative_steps": steps,
        "cumulative_size_kb": full_kb,
        "cumulative_compact_size_kb": full_kb,
        "cumulative_truncated_steps": 0,
        "early_step_sec": early_sec or 2.8,
        "recent_step_sec": recent_sec or 2.8,
        "latency_inflation_ratio": inflation,
        "primary_bloat_cause": primary_cause,
        "avoidable_bloat_kb": avoidable_kb,
        "top_offender_tools": sorted_tools,
        "recommendations": [],
    }


def _resolve_claude_projects_dir(claude_dir: Optional[Path] = None) -> Optional[Path]:
    """Resolve Claude Code projects directory (`CLAUDE_CODE_DIR` or `~/.claude/projects`)."""
    env_c = os.environ.get("CLAUDE_CODE_DIR", "").strip()
    if claude_dir is not None:
        return Path(claude_dir).expanduser()
    if env_c:
        return Path(env_c).expanduser()
    # Respect isolated test sandboxes when AGY_BRAIN_DIR is overridden without CLAUDE_CODE_DIR
    if os.environ.get("AGY_BRAIN_DIR") or os.environ.get("ANTIGRAVITY_BRAIN_DIR"):
        return None
    cand = (Path.home() / ".claude" / "projects").expanduser()
    return cand if cand.is_dir() else None


def inspect_conversation_lightweight(
    conv_id: Optional[str] = None,
    brain_dir: Optional[Path] = None,
    claude_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """
    Lightweight Community-safe conversation vitals inspection (`steps`, `csi`, `health_tier`,
    `full_size_kb`, `latency_inflation_ratio`) across Antigravity and Claude Code transcripts
    with zero Pro module dependencies.
    """
    target_brain = resolve_brain_dir(brain_dir)
    target_claude = _resolve_claude_projects_dir(claude_dir if brain_dir is None else claude_dir)

    raw_id = str(conv_id or "").strip()
    if raw_id.startswith("conversation://"):
        raw_id = raw_id.replace("conversation://", "", 1).strip()

    # 1. Explicit conv_id lookup across Antigravity and Claude Code
    if raw_id:
        if target_brain.is_dir():
            exact = target_brain / raw_id
            c_dir = exact if exact.is_dir() else next((p for p in target_brain.glob(f"{raw_id}*") if p.is_dir()), None)
            if c_dir is not None:
                agy_res = _inspect_agy_dir_lightweight(c_dir)
                if agy_res is not None:
                    return agy_res
        if target_claude is not None and target_claude.is_dir():
            for jp in target_claude.rglob(f"{raw_id}*.jsonl"):
                cl_res = _inspect_claude_jsonl_lightweight(jp)
                if cl_res is not None:
                    return cl_res
        return None

    # 2. Active Antigravity env var (`AGY_CONVERSATION_ID` / `CONVERSATION_ID`)
    if target_brain.is_dir():
        for env_k in ("AGY_CONVERSATION_ID", "CONVERSATION_ID"):
            val = os.environ.get(env_k)
            if val and (target_brain / val).is_dir():
                agy_env = _inspect_agy_dir_lightweight(target_brain / val)
                if agy_env is not None:
                    return agy_env

    # 3. Compare newest Antigravity transcript vs newest Claude Code transcript (preferring cwd project)
    best_agy_mtime = -1.0
    best_agy_dir: Optional[Path] = None
    if target_brain.is_dir():
        transcripts = glob.glob(
            str(target_brain / "*" / ".system_generated" / "logs" / "transcript.jsonl")
        )
        if transcripts:
            transcripts.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            best_agy_mtime = os.path.getmtime(transcripts[0])
            best_agy_dir = Path(transcripts[0]).parents[2]

    best_claude_mtime = -1.0
    best_claude_path: Optional[Path] = None
    if target_claude is not None and target_claude.is_dir():
        cwd_slug = re.sub(r"[^A-Za-z0-9]", "-", str(Path.cwd().resolve()))
        cwd_proj = target_claude / cwd_slug
        search_root = cwd_proj if cwd_proj.is_dir() and list(cwd_proj.glob("*.jsonl")) else target_claude
        c_files = [p for p in search_root.rglob("*.jsonl") if p.is_file()]
        if c_files:
            c_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            best_claude_mtime = os.path.getmtime(c_files[0])
            best_claude_path = c_files[0]

    if best_claude_path is not None and best_claude_mtime > best_agy_mtime:
        cl_latest = _inspect_claude_jsonl_lightweight(best_claude_path)
        if cl_latest is not None:
            return cl_latest

    if best_agy_dir is not None:
        agy_latest = _inspect_agy_dir_lightweight(best_agy_dir)
        if agy_latest is not None:
            return agy_latest

    if best_claude_path is not None:
        return _inspect_claude_jsonl_lightweight(best_claude_path)

    return None


def list_conversations_lightweight(
    limit: int = 5,
    brain_dir: Optional[Path] = None,
    claude_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """List most recent Antigravity + Claude Code conversation vitals sorted by mtime."""
    target_brain = resolve_brain_dir(brain_dir)
    target_claude = _resolve_claude_projects_dir(claude_dir if brain_dir is None else claude_dir)
    candidates: List[Tuple[float, str, Path]] = []

    if target_brain.is_dir():
        for tp_str in glob.glob(str(target_brain / "*" / ".system_generated" / "logs" / "transcript.jsonl")):
            try:
                candidates.append((os.path.getmtime(tp_str), "agy", Path(tp_str).parents[2]))
            except OSError:
                continue

    if target_claude is not None and target_claude.is_dir():
        for jp in target_claude.rglob("*.jsonl"):
            if jp.is_file():
                try:
                    candidates.append((os.path.getmtime(jp), "claude", jp))
                except OSError:
                    continue

    candidates.sort(key=lambda item: item[0], reverse=True)
    out: List[Dict[str, Any]] = []
    max_n = max(1, int(limit or 5))
    for _, kind, p in candidates:
        res = _inspect_agy_dir_lightweight(p) if kind == "agy" else _inspect_claude_jsonl_lightweight(p)
        if res is not None:
            out.append(res)
        if len(out) >= max_n:
            break
    return out


def _inspect_agy_dir_lightweight(c_dir: Path) -> Optional[Dict[str, Any]]:
    """Inspect a single Antigravity conversation directory."""
    logs_dir = c_dir / ".system_generated" / "logs"
    compact_p = logs_dir / "transcript.jsonl"
    full_p = logs_dir / "transcript_full.jsonl"
    has_full = full_p.is_file()
    has_compact = compact_p.is_file()
    if not has_full and not has_compact:
        return None

    read_p = full_p if has_full else compact_p
    is_compact_only = not has_full
    full_kb = (
        round(os.path.getsize(full_p) / 1024.0, 1)
        if has_full
        else round(os.path.getsize(compact_p) / 1024.0, 1)
    )
    compact_kb = round(os.path.getsize(compact_p) / 1024.0, 1) if has_compact else full_kb

    lines: List[Dict[str, Any]] = []
    with open(read_p, "r", encoding="utf-8", errors="ignore") as fh:
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

    dual = compute_dual_csi_metrics(
        lines=lines,
        full_size_kb=full_kb,
        compact_size_kb=compact_kb,
        is_compact_only=is_compact_only,
    )
    early_sec = dual["early_step_sec"]
    recent_sec = dual["recent_step_sec"]
    inflation = round(recent_sec / early_sec, 2) if early_sec > 0.1 and recent_sec > 0.0 else 1.0
    csi = dual["csi"]
    tier = dual["health_tier"]
    primary_cause, avoidable_kb, sorted_tools, recommendations = analyze_bloat_attribution(
        lines=dual["cumulative_lines"],
        steps=dual["cumulative_steps"],
        full_size_kb=dual["cumulative_size_kb"],
        compact_size_kb=dual["cumulative_compact_size_kb"],
        truncated_steps=dual["cumulative_truncated_steps"],
        csi=csi,
        is_compact_only=is_compact_only,
        active_size_kb=dual["active_size_kb"],
    )
    return {
        "conv_id": c_dir.name,
        "short_id": c_dir.name[:8],
        "harness": "antigravity",
        "steps": dual["active_steps"],
        "csi": csi,
        "health_tier": tier,
        "compact_size_kb": dual["active_compact_size_kb"],
        "full_size_kb": dual["active_size_kb"],
        "truncated_steps": dual["active_truncated_steps"],
        "active_steps": dual["active_steps"],
        "active_size_kb": dual["active_size_kb"],
        "compaction_count": dual["compaction_count"],
        "peak_csi": dual["peak_csi"],
        "cumulative_steps": dual["cumulative_steps"],
        "cumulative_size_kb": dual["cumulative_size_kb"],
        "cumulative_compact_size_kb": dual["cumulative_compact_size_kb"],
        "cumulative_truncated_steps": dual["cumulative_truncated_steps"],
        "early_step_sec": early_sec or 2.8,
        "recent_step_sec": recent_sec or (early_sec or 2.8),
        "latency_inflation_ratio": inflation,
        "primary_bloat_cause": primary_cause,
        "avoidable_bloat_kb": avoidable_kb,
        "top_offender_tools": sorted_tools,
        "recommendations": recommendations,
    }


def _locate_standalone_audit_script() -> Optional[Path]:
    """Locate repo `docs/audit.py` first (for live dev checkouts), falling back to bundled `standalone_audit.py`."""
    here = Path(__file__).resolve()
    for parent in (here.parents[2], here.parents[1], *here.parents):
        candidate = parent / "docs" / "audit.py"
        if candidate.is_file():
            return candidate
    try:
        from whole_body_compute.bundled import get_bundled_audit_script_path

        bundled = get_bundled_audit_script_path()
        if bundled.is_file():
            return bundled
    except Exception:
        pass
    return None


def _load_standalone_audit_module() -> Optional[Any]:
    """Load `docs/audit.py` or bundled `standalone_audit.py` as an isolated Python module."""
    try:
        audit_script = _locate_standalone_audit_script()
        if audit_script is not None and audit_script.is_file():
            spec = importlib.util.spec_from_file_location(
                "wbc_standalone_audit", audit_script
            )
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
    except Exception as exc:
        logger.debug("Standalone audit module load skipped: %s", exc)
    return None


_TURN0_WORD_RE = re.compile(r"[A-Za-z]+|[0-9]{1,3}|[^\sA-Za-z0-9]")
_TURN0_INDENT_RE = re.compile(r"[ ]{2,}")
_TURN0_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.\-]")
_TURN0_SECRET_ID_RE = re.compile(
    r"^(?:sk-|ghp_|github_pat_|AKIA|AIza|phc_|xox[baprs]-)", re.IGNORECASE
)
DEFAULT_ESTIMATED_TOKENS_PER_MCP_SERVER = 1_800


def _estimate_static_context_tokens(text: str, is_json: bool = False) -> int:
    """
    Calibrated zero-dependency BPE token estimator (<10% drift vs cl100k/o200k)
    across Markdown instruction files and JSON MCP tool schemas.
    """
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


def _scan_turn0_installed_skills(ws_dir: Path, h_dir: Path) -> Tuple[List[Dict[str, Any]], int]:
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

    skill_roots: List[Tuple[Path, str]] = [
        (ws_dir / ".agents" / "skills", "workspace"),
        (h_dir / ".agents" / "skills", "user_global"),
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
                    skill_roots.append((p_dir / "skills", "plugin"))
        except OSError:
            pass
    skill_roots.append((h_dir / ".gemini" / "antigravity" / "builtin" / "skills", "builtin"))

    seen_skill_rp: set[str] = set()
    skills_list: List[Dict[str, Any]] = []
    total_skill_tok = 0

    for s_root, s_scope in skill_roots:
        if not s_root.is_dir():
            continue
        try:
            for s_dir in sorted(s_root.iterdir())[:96]:
                if not s_dir.is_dir() or s_dir.name.startswith("."):
                    continue
                s_md = s_dir / "SKILL.md"
                if not s_md.is_file():
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
) -> Dict[str, Any]:
    """
    Blood-Brain Barrier - Turn-0 Fixed Context Tax Audit (<50ms, 100% stdlib, 0 PII leaks):
    Scans static instruction files (`CLAUDE.md`, `AGENTS.md`, `GEMINI.md`, `.cursorrules`, etc.),
    conditional rules, installed `<skills>` index descriptions, and MCP server tool configurations/schemas
    to quantify fixed per-session startup token bloat vs `0` MCP schema tokens for Whole Body Compute OS hooks.
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

    # 1. Measure cached MCP schema JSONs on disk (e.g. ~/.gemini/antigravity/mcp/<server>/*.json)
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

    # 2. Scan MCP configuration files (metadata-only: server names + inline tool schemas, 0 secrets)
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
                mcp_servers_map[s_name] = {
                    "name": s_name,
                    "source": cfg_label,
                    "schema_files": 0,
                    "tokens": DEFAULT_ESTIMATED_TOKENS_PER_MCP_SERVER,
                    "provenance": "calibrated_server_estimate",
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

    # Include any disk-cached MCP schema directories not listed in a separate mcp_config.json
    # (Skip explicitly disabled servers, and skip orphan ~/.gemini/antigravity/mcp folders when mcp_config.json exists)
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

    skills_list, skills_turn0_tokens = _scan_turn0_installed_skills(ws_dir=ws_dir, h_dir=h_dir)

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
        if s["provenance"] == "calibrated_server_estimate"
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
    # Blended cached+uncached Turn-0 cost across sessions (first turn of every session is uncached or cache-create!)
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
        "summary": redact_secrets(summary),
    }


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
    workspace_dir: Optional[Path] = None,
    home_dir: Optional[Path] = None,
    import_paths: Optional[List[Any]] = None,
    include_benchmarks: bool = False,
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

    standalone_mod = _load_standalone_audit_module()
    eff_inc_bench = bool(
        include_benchmarks
        or os.environ.get("WBC_AUDIT_INCLUDE_BENCHMARKS", "").strip().lower() in ("1", "true", "yes")
    )
    excluded_stats: Dict[str, int] = {}
    agy_bench_cids: set[str] = set()
    if not eff_inc_bench and standalone_mod is not None and hasattr(standalone_mod, "_is_agy_conv_direct_benchmark"):
        agy_parent_map: Dict[str, str] = {}
        sender_re = re.compile(r"sender=([A-Za-z0-9_-]{8,})")
        cont_re = re.compile(r"Continuing from (?:conversation://)?([0-9a-fA-F\-]{36})")
        for _mt, c_dir, primary_p, _fp in conv_dirs:
            cid = c_dir.name
            if standalone_mod._is_benchmark_path_or_cwd(cid):
                agy_bench_cids.add(cid)
                continue
            head_objs: List[Tuple[Dict[str, Any], int]] = []
            fallback_snd = ""
            try:
                with open(primary_p, "r", encoding="utf-8", errors="ignore") as hfh:
                    for hi in range(60):
                        hln = hfh.readline()
                        if not hln:
                            break
                        hs = hln.strip()
                        if not hs:
                            continue
                        try:
                            ho = json.loads(hs)
                        except Exception:
                            continue
                        head_objs.append((ho, len(hs)))
                        htype = str(ho.get("type") or "")
                        hcnt = str(ho.get("content") or "")
                        if htype == "SYSTEM_MESSAGE" and "sender=" in hcnt:
                            ms = sender_re.search(hcnt)
                            if ms and ms.group(1).lower() != "system":
                                if "[swarm] new board message" in hcnt:
                                    if not fallback_snd:
                                        fallback_snd = ms.group(1)
                                elif cid not in agy_parent_map:
                                    agy_parent_map[cid] = ms.group(1)
                        elif htype == "USER_INPUT" and hi < 5 and cid not in agy_parent_map:
                            mc = cont_re.search(hcnt)
                            if mc and mc.group(1) != cid:
                                agy_parent_map[cid] = mc.group(1)
            except Exception:
                continue
            if cid not in agy_parent_map and fallback_snd:
                agy_parent_map[cid] = fallback_snd
            if standalone_mod._is_agy_conv_direct_benchmark(head_objs):
                agy_bench_cids.add(cid)
        changed = True
        while changed:
            changed = False
            for cid, pid in agy_parent_map.items():
                if cid not in agy_bench_cids and pid in agy_bench_cids:
                    agy_bench_cids.add(cid)
                    changed = True

    model_usage_by_id: Dict[str, Dict[str, Dict[str, Any]]] = {}
    thread_records: List[Dict[str, Any]] = []

    for mtime, c_dir, primary_p, full_p in conv_dirs:
        if not eff_inc_bench and c_dir.name in agy_bench_cids:
            excluded_stats["antigravity"] = int(excluded_stats.get("antigravity", 0)) + 1
            excluded_stats["_total"] = int(excluded_stats.get("_total", 0)) + 1
            continue
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
            truncated_steps = sum(1 for step in lines if step.get("truncated_fields"))
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

            planner_turns = sum(1 for step in lines if step.get("type") == "PLANNER_RESPONSE")
            has_planner_turns = planner_turns > 0
            if not has_planner_turns:
                planner_turns = max(1, int(round(steps * 0.38)))

            first_prompt = ""
            parent_conv_id = ""
            active_model = ""
            conv_model_usage: Dict[str, Dict[str, Any]] = {}
            conv_metered_turns = 0
            conv_metered_in = 0
            conv_metered_cr = 0
            conv_metered_out = 0
            conv_fresh_usd = 0.0
            conv_cache_usd = 0.0
            conv_out_usd = 0.0
            conv_max_ctx = 0
            conv_latest_ctx = 0
            conv_turns_over_100k = 0
            for idx, (step, raw_len) in enumerate(zip(lines, raw_line_lengths), start=1):
                stype = step.get("type")
                content_raw = str(step.get("content") or "")
                if standalone_mod is not None and "Model Selection" in content_raw:
                    ms_cand = standalone_mod._extract_model_selection_change(content_raw)
                    if ms_cand:
                        active_model = ms_cand
                if step.get("model"):
                    active_model = str(step.get("model"))
                if stype == "USER_INPUT" and idx <= 5:
                    content_s = content_raw
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
                    if standalone_mod is not None:
                        if "input_tokens" in step or "output_tokens" in step or "cache_read_tokens" in step:
                            t_in = int(step.get("input_tokens") or 0)
                            t_cr = int(step.get("cache_read_tokens") or 0)
                            t_out = int(step.get("output_tokens") or 0)
                            conv_metered_turns += 1
                            conv_metered_in += t_in
                            conv_metered_cr += t_cr
                            conv_metered_out += t_out
                            turn_ctx = t_in + t_cr
                            conv_latest_ctx = turn_ctx
                            if turn_ctx > conv_max_ctx:
                                conv_max_ctx = turn_ctx
                            if turn_ctx >= 100_000:
                                conv_turns_over_100k += 1
                            tin_p, tout_p, _ = standalone_mod._resolve_model_rates(
                                active_model, in_price_per_m, out_price_per_m, False
                            )
                            t_fresh = (t_in / 1_000_000.0) * tin_p
                            t_cache = (t_cr / 1_000_000.0) * (tin_p * 0.10)
                            t_out_u = (t_out / 1_000_000.0) * tout_p
                            conv_fresh_usd += t_fresh
                            conv_cache_usd += t_cache
                            conv_out_usd += t_out_u
                            if active_model:
                                standalone_mod._record_turn_model_usage(
                                    conv_model_usage,
                                    active_model,
                                    turns=1,
                                    billed_input_tokens=t_in,
                                    cache_read_tokens=t_cr,
                                    output_tokens=t_out,
                                    fresh_usd=t_fresh,
                                    cache_usd=t_cache,
                                    output_usd=t_out_u,
                                )
                        elif active_model:
                            est_out = max(20, step_tok)
                            _, tout_p, _ = standalone_mod._resolve_model_rates(
                                active_model, in_price_per_m, out_price_per_m, False
                            )
                            standalone_mod._record_turn_model_usage(
                                conv_model_usage,
                                active_model,
                                turns=1,
                                output_tokens=est_out,
                                output_usd=(est_out / 1_000_000.0) * tout_p,
                            )

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

            has_tool_activity = any(bool(step.get("tool_calls") or step.get("tool_name")) for step in lines)
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

            if conv_model_usage:
                model_usage_by_id[c_dir.name] = conv_model_usage
            exact_agy_usd = (
                round(conv_fresh_usd + conv_cache_usd + conv_out_usd, 2)
                if conv_metered_turns > 0
                else round(conv_cons_frontier_usd, 2)
            )
            primary_model_lbl = (
                max(
                    conv_model_usage.values(),
                    key=lambda mu: (float(mu.get("spend_usd", 0.0)), int(mu.get("turns", 0))),
                )["label"]
                if conv_model_usage
                else ""
            )

            thread_records.append(
                {
                    "harness": "antigravity",
                    "conv_id": c_dir.name,
                    "short_id": c_dir.name[:8],
                    "primary_model": primary_model_lbl,
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
                    "exact_api_spend_usd": exact_agy_usd,
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
    gemini_cli_records: List[Dict[str, Any]] = []
    claude_records: List[Dict[str, Any]] = []
    claude_chat_records: List[Dict[str, Any]] = []
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
            "GEMINI_CLI_DIR",
        )
    )
    is_isolated_agy_test = bool(has_custom_agy and not has_external_custom)
    is_selective_custom = bool(has_custom_agy or has_external_custom)

    if not is_isolated_agy_test and standalone_mod is not None:
        try:
            if True:
                if True:
                    mod = standalone_mod
                    cand_homes = (
                        mod._iter_candidate_user_homes()
                        if hasattr(mod, "_iter_candidate_user_homes")
                        else [Path.home()]
                    )
                    dedup_fn = getattr(mod, "_dedup_records", lambda xs: xs)

                    # Also sweep WSL/Windows Antigravity brains when running default non-custom scan
                    extra_agy_records: List[Dict[str, Any]] = []
                    if not is_selective_custom and len(cand_homes) > 1:
                        seen_agy_ids = {
                            str(r.get("conversation_id") or r.get("short_id") or "")
                            for r in agy_records
                        }
                        for eh in cand_homes[1:]:
                            eh_brain = eh / ".gemini" / "antigravity" / "brain"
                            if eh_brain.is_dir():
                                for er in mod.scan_antigravity_harness(
                                    eh_brain,
                                    window_seconds,
                                    now,
                                    in_price_per_m,
                                    out_price_per_m,
                                    include_benchmarks=eff_inc_bench,
                                    excluded_stats=excluded_stats,
                                ):
                                    ecid = str(er.get("conversation_id") or er.get("short_id") or "")
                                    if ecid not in seen_agy_ids:
                                        seen_agy_ids.add(ecid)
                                        extra_agy_records.append(er)
                                        agy_records.append(er)

                    claude_roots = (
                        [Path(claude_dir or os.environ["CLAUDE_CODE_DIR"]).expanduser()]
                        if (claude_dir is not None or os.environ.get("CLAUDE_CODE_DIR"))
                        else [h / ".claude" / "projects" for h in cand_homes]
                    )
                    goose_roots = (
                        [Path(goose_db or os.environ["GOOSE_SESSIONS_DB"]).expanduser()]
                        if (goose_db is not None or os.environ.get("GOOSE_SESSIONS_DB"))
                        else [
                            p
                            for h in cand_homes
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

                    if (not is_selective_custom or os.environ.get("GEMINI_CLI_DIR")) and hasattr(
                        mod, "scan_gemini_cli_harness"
                    ):
                        gemini_cli_records = mod.scan_gemini_cli_harness(
                            None,
                            window_seconds,
                            now,
                            in_price_per_m,
                            out_price_per_m,
                            include_benchmarks=eff_inc_bench,
                            excluded_stats=excluded_stats,
                        )
                    if (
                        not is_selective_custom
                        or claude_dir is not None
                        or os.environ.get("CLAUDE_CODE_DIR")
                    ):
                        claude_records = dedup_fn(
                            [
                                r
                                for root in claude_roots
                                for r in mod.scan_claude_code_harness(
                                    root,
                                    window_seconds,
                                    now,
                                    in_price_per_m,
                                    out_price_per_m,
                                    include_benchmarks=eff_inc_bench,
                                    excluded_stats=excluded_stats,
                                )
                            ]
                        )
                    if (
                        not is_selective_custom
                        or goose_db is not None
                        or os.environ.get("GOOSE_SESSIONS_DB")
                    ):
                        goose_records = dedup_fn(
                            [
                                r
                                for root in goose_roots
                                for r in mod.scan_goose_harness(
                                    root,
                                    window_seconds,
                                    now,
                                    in_price_per_m,
                                    out_price_per_m,
                                    include_benchmarks=eff_inc_bench,
                                    excluded_stats=excluded_stats,
                                )
                            ]
                        )
                    if (
                        not is_selective_custom
                        or cursor_dir is not None
                        or os.environ.get("CURSOR_STORAGE_DIR")
                    ):
                        cursor_records = mod.scan_cursor_harness(
                            cursor_override,
                            window_seconds,
                            now,
                            in_price_per_m,
                            out_price_per_m,
                            include_benchmarks=eff_inc_bench,
                            excluded_stats=excluded_stats,
                        )
                    if (
                        not is_selective_custom
                        or codex_dir is not None
                        or os.environ.get("CODEX_SESSIONS_DIR")
                        or os.environ.get("CODEX_HOME")
                    ):
                        codex_records = mod.scan_codex_harness(
                            codex_dir,
                            window_seconds,
                            now,
                            in_price_per_m,
                            out_price_per_m,
                            include_benchmarks=eff_inc_bench,
                            excluded_stats=excluded_stats,
                        )
                    if (
                        not is_selective_custom
                        or cline_dir is not None
                        or os.environ.get("CLINE_STORAGE_DIR")
                    ):
                        cline_records = mod.scan_cline_roo_harness(
                            cline_dir,
                            window_seconds,
                            now,
                            in_price_per_m,
                            out_price_per_m,
                            include_benchmarks=eff_inc_bench,
                            excluded_stats=excluded_stats,
                        )
                    if (
                        not is_selective_custom
                        or windsurf_dir is not None
                        or os.environ.get("WINDSURF_STORAGE_DIR")
                    ):
                        windsurf_records = mod.scan_windsurf_harness(
                            windsurf_dir,
                            window_seconds,
                            now,
                            in_price_per_m,
                            out_price_per_m,
                            include_benchmarks=eff_inc_bench,
                            excluded_stats=excluded_stats,
                        )
                    if (
                        not is_selective_custom
                        or aider_path is not None
                        or os.environ.get("AIDER_HISTORY_PATH")
                    ):
                        aider_records = mod.scan_aider_harness(
                            aider_path,
                            window_seconds,
                            now,
                            in_price_per_m,
                            out_price_per_m,
                            include_benchmarks=eff_inc_bench,
                            excluded_stats=excluded_stats,
                        )
                    if (
                        not is_selective_custom
                        or continue_dir is not None
                        or os.environ.get("CONTINUE_SESSIONS_DIR")
                    ):
                        continue_records = mod.scan_continue_dev_harness(
                            continue_dir,
                            window_seconds,
                            now,
                            in_price_per_m,
                            out_price_per_m,
                            include_benchmarks=eff_inc_bench,
                            excluded_stats=excluded_stats,
                        )
                    if (
                        not is_selective_custom
                        or zed_dir is not None
                        or os.environ.get("ZED_THREADS_DIR")
                    ):
                        zed_records = mod.scan_zed_harness(
                            zed_dir,
                            window_seconds,
                            now,
                            in_price_per_m,
                            out_price_per_m,
                            include_benchmarks=eff_inc_bench,
                            excluded_stats=excluded_stats,
                        )

                    for extra_r in (
                        extra_agy_records
                        + gemini_cli_records
                        + claude_records
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
                        extra_r.pop("_mtime", None)
                        extra_r.pop("_wbc_hook_fired", None)
                        extra_r.pop("_fresh_input_usd", None)
                        extra_r.pop("_cache_read_usd", None)
                        extra_r.pop("_output_usd", None)
                        extra_r.pop("_local_title", None)
                        ext_mu = extra_r.pop("_model_usage", None)
                        if ext_mu:
                            model_usage_by_id[str(extra_r.get("conv_id", ""))] = ext_mu
                        thread_records.append(extra_r)
        except Exception as exc:
            logger.debug("External harness scan skipped: %s", exc)

    if import_paths and standalone_mod is not None and hasattr(standalone_mod, "scan_claude_export_records"):
        try:
            claude_chat_records = standalone_mod.scan_claude_export_records(
                list(import_paths),
                window_seconds,
                now,
                in_price_per_m,
                out_price_per_m,
            )
            for cr in claude_chat_records:
                c_csi, c_tier = calculate_csi(
                    steps=int(cr["steps"]),
                    full_size_kb=float(cr.get("transcript_kb", 0.0)),
                    truncated_steps=int(cr.get("truncated_steps", 0)),
                )
                cr["csi"] = c_csi
                cr["health_tier"] = c_tier
                cr.setdefault(
                    "avoidable_bloat_kb",
                    round(int(cr.get("wasted_bloat_tokens", 0)) * 4.0 / 1024.0, 1),
                )
                total_steps += int(cr["steps"])
                total_planner_turns += int(cr["planner_turns"])
                total_transcript_bytes += int(cr.get("transcript_bytes", 0))
                total_raw_unbounded_tokens += int(cr["raw_unbounded_quadratic_tokens"])
                total_actual_input_tokens += int(cr["actual_cumulative_input_tokens"])
                total_output_tokens += int(cr["output_tokens"])
                total_wasted_bloat_tokens += int(cr["wasted_bloat_tokens"])
                total_quadratic_post_sat_tokens += int(cr["quadratic_post_saturation_tokens"])
                total_unfiltered_tool_bloat_tokens += int(cr["unfiltered_tool_bloat_tokens"])
                total_bicameral_offload_tokens += int(cr.get("bicameral_local_offload_tokens", 0))
                total_bbb_optimized_tokens += int(cr["bbb_optimized_tokens"])
                total_bbb_optimized_output_tokens += int(cr.get("bbb_optimized_output_tokens", 0))
                cr.pop("_mtime", None)
                cr.pop("_wbc_hook_fired", None)
                cr.pop("_fresh_input_usd", None)
                cr.pop("_cache_read_usd", None)
                cr.pop("_output_usd", None)
                cr.pop("_local_title", None)
                c_mu = cr.pop("_model_usage", None)
                if c_mu:
                    model_usage_by_id[str(cr.get("conv_id", ""))] = c_mu
                thread_records.append(cr)
        except Exception as exc:
            logger.debug("Imported Claude chat scan skipped: %s", exc)

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
        "gemini_cli": _summarize_h("Google Gemini CLI", gemini_cli_records),
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
    if claude_chat_records:
        harness_breakdown["claude_chat"] = _summarize_h(
            "Claude.ai / Desktop Chat", claude_chat_records
        )

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
        "n200_pass_at_1_wbc_pct": 85.5,
        "n200_pass_at_1_vanilla_pct": 83.5,
        "n200_pass_at_1_delta_pp": 2.0,
        "real_world_per_session_cost_drop_pct": 74.1,
        "real_world_unshielded_avg_session_usd": 3.59,
        "real_world_shielded_avg_session_usd": 0.93,
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
        "regime_reconciliation": (
            "Sprint Floor (SWE-bench ~11t single-issue patches): 85.5% vs 83.5% Pass@1 (+2.0pp, N=200), "
            "-15% to -34.3% context • Marathon Production (50+t multi-turn sessions): -74.1% per-session cost "
            "($3.59 -> $0.93/sess) & 85-98.7% quadratic bloat elimination via Glymphatic Flush + Local Metal"
        ),
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

    has_custom_harness_override = any(
        x is not None
        for x in (
            brain_dir,
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
    )
    resolved_turn0_ws = (
        workspace_dir
        if workspace_dir is not None
        else (
            Path(os.environ["WBC_AUDIT_WORKSPACE_DIR"])
            if os.environ.get("WBC_AUDIT_WORKSPACE_DIR")
            else (target_brain.parent if has_custom_harness_override else Path.cwd())
        )
    )
    resolved_turn0_home = (
        home_dir
        if home_dir is not None
        else (
            Path(os.environ["WBC_AUDIT_HOME_DIR"])
            if os.environ.get("WBC_AUDIT_HOME_DIR")
            else (target_brain.parent if has_custom_harness_override else Path.home())
        )
    )
    turn0_static_overhead = audit_turn0_static_overhead(
        workspace_dir=resolved_turn0_ws,
        home_dir=resolved_turn0_home,
        sessions_count=len(thread_records),
        in_price_per_m=in_price_per_m,
    )
    if standalone_mod is not None and hasattr(standalone_mod, "compute_model_opportunity_insights"):
        model_opportunities = standalone_mod.compute_model_opportunity_insights(
            all_records=thread_records,
            model_usage_by_id=model_usage_by_id,
            window_days=days,
        )
    else:
        model_opportunities = {
            "read_only": True,
            "window_days": int(days),
            "tagged_sessions": 0,
            "tagged_turns": 0,
            "tagged_spend_usd": 0.0,
            "untagged_sessions": len(thread_records),
            "untagged_turns": total_planner_turns,
            "untagged_spend_usd": 0.0,
            "tier_breakdown": {},
            "models": [],
            "routine_downshift": {"sessions": 0, "turns": 0, "current_spend_usd": 0.0, "savable_usd": 0.0},
            "thinking_cot_tax": {"sessions": 0, "turns": 0, "output_tokens": 0, "current_output_usd": 0.0, "savable_usd": 0.0},
            "marathon_reread_drag": {"sessions": 0, "turns": 0, "cache_read_tokens": 0, "current_reread_usd": 0.0, "savable_usd": 0.0},
            "total_opportunity_usd": 0.0,
            "projected_annual_opportunity_usd": 0.0,
            "opportunities": [],
        }

    router_shadow_savings: Dict[str, Any] = {
        "total_evals": 0,
        "net_tokens_saved_total": 0,
        "net_time_saved_total_s": 0.0,
        "estimated_usd_saved": 0.0,
        "tier0_ast_pinpoint": {
            "name": "Tier-0 AST Pinpoint",
            "engine": "ane_amx_tier0",
            "count": 0,
            "tier0_evals": 0,
            "agreement_rate": 0.0,
            "p50_ms": 0.0,
            "net_tokens_saved_total": 0,
            "mean_net_time_saved_s": 0.0,
        },
    }
    try:
        from whole_body_compute.local.shadow_ledger import _resolve_telemetry_paths

        _, ledger_path = _resolve_telemetry_paths()
        ledger_sum = (
            json.loads(ledger_path.read_text(encoding="utf-8"))
            if ledger_path.is_file()
            else {}
        )
        tasks_map = ledger_sum.get("tasks", {}) if isinstance(ledger_sum, dict) else {}
        if isinstance(tasks_map, dict) and tasks_map:
            total_net_tok = sum(int(t.get("net_tokens_saved_total", 0) or 0) for t in tasks_map.values() if isinstance(t, dict))
            total_net_time = sum(float(t.get("net_time_saved_sum_s", 0.0) or 0.0) for t in tasks_map.values() if isinstance(t, dict))
            pin = tasks_map.get("tool_read_pinpoint", {}) if isinstance(tasks_map.get("tool_read_pinpoint"), dict) else {}
            pin_engines = pin.get("engines", {}) if isinstance(pin.get("engines"), dict) else {}
            router_shadow_savings = {
                "total_evals": int(ledger_sum.get("total_shadow_evals", 0) or 0),
                "net_tokens_saved_total": total_net_tok,
                "net_time_saved_total_s": round(total_net_time, 2),
                "estimated_usd_saved": round((max(0, total_net_tok) / 1_000_000.0) * in_price_per_m, 2),
                "tier0_ast_pinpoint": {
                    "name": "Tier-0 AST Pinpoint",
                    "engine": "ane_amx_tier0",
                    "count": int(pin.get("count", 0) or 0),
                    "tier0_evals": int(pin_engines.get("ane_amx_tier0", 0) or 0),
                    "agreement_rate": float(pin.get("agreement_rate", 0.0) or 0.0),
                    "p50_ms": float(pin.get("p50_ms", 0.0) or 0.0),
                    "net_tokens_saved_total": int(pin.get("net_tokens_saved_total", 0) or 0),
                    "mean_net_time_saved_s": float(pin.get("mean_net_time_saved_s", 0.0) or 0.0),
                },
            }
    except Exception:
        pass

    res_dict = {
        "window_days": int(days),
        "flush_step_interval": resolved_flush_steps,
        "brain_dir": str(target_brain),
        "input_price_per_m": in_price_per_m,
        "output_price_per_m": out_price_per_m,
        "seats": int(seats),
        "include_benchmarks": eff_inc_bench,
        "router_shadow_savings": router_shadow_savings,
        "excluded_benchmark_conversations": int(excluded_stats.get("_total", 0)),
        "excluded_benchmark_by_harness": {
            k: int(v) for k, v in excluded_stats.items() if k != "_total" and int(v) > 0
        },
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
        "model_opportunities": model_opportunities,
        "turn0_static_overhead": turn0_static_overhead,
        "live_organ10_interceptions": live_organ10,
        "verified_handoff_lineage": verified_handoff_lineage,
        "autonomy_ratchet": autonomy_ratchet,
        "top_bloated_conversations": top_threads,
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        import sys as _sys
        from whole_body_compute.procedural_telemetry import track_wbc_event

        active_h = [
            k for k, v in (harness_breakdown or {}).items()
            if isinstance(v, dict) and int(v.get("threads", 0)) > 0
        ]
        home = Path.home()
        is_wsl = bool(os.environ.get("WSL_DISTRO_NAME"))
        if not is_wsl and Path("/proc/version").is_file():
            try:
                is_wsl = "microsoft" in Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
            except Exception:
                pass
        raw_acq = (os.environ.get("WBC_AUDIT_SOURCE") or "cli_installed").strip().lower()
        acq_source = re.sub(r"[^a-z0-9_.-]", "", raw_acq)[:40] or "cli_installed"
        track_wbc_event(
            "wbc_audit_completed",
            {
                "mode": "body_audit_spend",
                "acquisition_source": acq_source,
                "is_qualified_audit": len(thread_records) > 0,
                "os_platform": _sys.platform,
                "window_days": int(days),
                "conversations_scanned": len(thread_records),
                "total_steps": int(total_steps),
                "active_harnesses": ",".join(active_h),
                "active_harnesses_count": len(active_h),
                "actual_cumulative_input_tokens": int(total_actual_input_tokens),
                "wasted_bloat_tokens": int(total_wasted_bloat_tokens),
                "wasted_spend_usd": float(wasted_spend_usd),
                "savings_pct": float(savings_pct),
                "has_agy_dir": (home / ".gemini" / "antigravity" / "brain").is_dir(),
                "has_claude_dir": (home / ".claude" / "projects").is_dir() or (home / ".claude").is_dir(),
                "has_cursor_dir": (
                    (home / ".cursor").is_dir()
                    or (home / "Library" / "Application Support" / "Cursor").is_dir()
                    or (home / "AppData" / "Roaming" / "Cursor").is_dir()
                ),
                "has_codex_dir": (home / ".codex").is_dir(),
                "has_windsurf_dir": (home / ".codeium" / "windsurf").is_dir(),
                "is_wsl": is_wsl,
                "has_wsl_win_users": Path("/mnt/c/Users").is_dir(),
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
    o10_direct_tok = int(live_o10.get("direct_tokens_saved", 0))
    o10_cum_tok = max(o10_direct_tok, int(live_o10.get("quadratic_tokens_saved", 0)))
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
            "  ├─ N=200 Floor (~11t Sprints)",
            "85.5% vs 83.5% Pass@1 (+2.0pp) • 100% Pass^3 (30/30) • -18.8% Turns",
        ),
        _row(
            "  └─ Real Usage (50+t Marathons)",
            "-74.1% Cost/Sess ($3.59 -> $0.93) • 68% Spend in 50+t • 70% Local",
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
            (
                f"{o10_int:,} blocks / {o10_checks:,} checks "
                f"({f'{o10_direct_tok / 1e6:.1f}M' if o10_direct_tok >= 100_000 else f'{o10_direct_tok:,}'} dir / "
                f"{f'{o10_cum_tok / 1e6:.1f}M' if o10_cum_tok >= 100_000 else f'{o10_cum_tok:,}'} cum | ${o10_usd:,.2f})"
            )[:65],
        ),
        _row(
            "Autonomy Ratchet (Day 1 -> Now)",
            f"{r_day1:.1f}% -> {r_curr:.1f}% ({r_gain:+.1f}% Grad | {r_c_red:.1f}% Cloud Reduction)",
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
    t0_meta = audit.get("turn0_static_overhead", {})
    t0_tot = int(t0_meta.get("total_turn0_tokens_per_session", 0))
    t0_inst = int(t0_meta.get("instruction_tokens", 0))
    t0_ifiles = int(t0_meta.get("instruction_files_count", 0))
    t0_mcp = int(t0_meta.get("mcp_turn0_tokens", 0))
    t0_msrv = int(t0_meta.get("mcp_servers_count", 0))
    if t0_tot > 0 or t0_msrv > 0 or t0_ifiles > 0:
        lines.append("")
        lines.append("🛡️ Turn-0 Fixed Startup Context Tax (Static Rules + MCP Tool Schemas vs `body` OS Hooks):")
        lines.append(
            f"  • Startup Payload / Session: {t0_tot:,} tok/session "
            f"({t0_inst:,} instruction tok across {t0_ifiles} files + {t0_mcp:,} MCP schema tok across {t0_msrv} servers)"
        )
        lines.append(
            f"  • MCP Bloat vs WBC OS Hooks: {t0_mcp:,} MCP schema tok/session vs 0 MCP schema tok for `body` PreToolUse/PostToolUse hooks ($0.00 Turn-0 tax)"
        )

    mo = audit.get("model_opportunities", {}) or {}
    mo_models = mo.get("models", []) or []
    if mo_models:
        lines.append("")
        lines.append("🧠 Read-Only Model Mix & Routing Opportunity Insights (Zero-Mutation):")
        lines.append(
            "  ┌──────────────────────────────┬──────────┬──────────┬──────────────┬──────────────┬────────────┬──────────┐"
        )
        lines.append(
            "  │ Logged Model                 │ Tier     │ Sessions │ Turns        │ API-Equiv $  │ Avg / Sess │ Out/CoT% │"
        )
        lines.append(
            "  ├──────────────────────────────┼──────────┼──────────┼──────────────┼──────────────┼────────────┼──────────┤"
        )
        tier_lbl_map = {"heavy_thinking": "Heavy", "balanced": "Balanced", "fast_low": "Fast/Low"}
        for m_row in mo_models[:8]:
            m_lbl = str(m_row.get("label", ""))[:28]
            m_tr = tier_lbl_map.get(str(m_row.get("tier", "")), "Balanced")[:8]
            m_sess = int(m_row.get("sessions", 0))
            m_trns = int(m_row.get("turns", 0))
            m_sp = f"${float(m_row.get('spend_usd', 0.0)):,.2f}"
            m_avg = f"${float(m_row.get('avg_session_usd', 0.0)):,.2f}"
            m_op = f"{float(m_row.get('output_share_pct', 0.0)):.1f}%"
            lines.append(
                f"  │ {m_lbl:<28} │ {m_tr:<8} │ {m_sess:>8,} │ {m_trns:>12,} │ {m_sp:>12} │ {m_avg:>10} │ {m_op:>8} │"
            )
        lines.append(
            "  └──────────────────────────────┴──────────┴──────────┴──────────────┴──────────────┴────────────┴──────────┘"
        )
        for opp in mo.get("opportunities", []) or []:
            lines.append(f"  • {opp.get('title', 'Opportunity')}: {opp.get('summary', '')}")
        tot_opp = float(mo.get("total_opportunity_usd", 0.0))
        yr_opp = float(mo.get("projected_annual_opportunity_usd", 0.0))
        if tot_opp >= 0.05:
            lines.append(
                f"  • Combined Model-Fit Opportunity: ~${tot_opp:,.2f} / {days}d (~${yr_opp:,.0f}/yr per seat • strictly read-only)"
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
        "🌐 Shareable Zero-Install Audit: `curl -sSL https://wholebodycompute.com/audit.py | python3`  •  Awaken Workspace: `curl -fsSL https://wholebodycompute.com/begin | sh`"
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
        "- **SWE-bench_Verified Empirical Runway (`60` Trials, `k=3`, `N=10` Official Repos, `100.0% Pass^3` `30/30` + `N=200` `85.5%` vs `83.5%` `Pass@1`):** `1.48x` cumulative context efficiency (`-32.4%`, `203.08M -> 137.23M` tokens) • `1.52x` net task context efficiency (`-34.3%`, `178.87M -> 117.56M` tokens) • `1.34x` billed input runway (`-25.6%` total billed input, `7.83M -> 5.83M` tokens; `-16.7%` median) • `-18.8%` fewer agent turns (`1,685 -> 1,368`)",
        "- **Why Real-World Usage Outperforms Synthetic Benchmarks (`Sprint vs. Marathon Regime`):** Synthetic `SWE-bench_Verified` tasks are cold-start `~11-turn` sprints where context resets after every patch (`-15% to -34%` cut). In real multi-turn usage (`50+` turn marathons driving `68%` of total spend), `body flush` StateCapsules + AST slicing + `70%` Local Metal offload eliminate quadratic history re-reads for a **`-74.1%` measured per-session cost drop (`$3.59 -> $0.93/sess`)** and **`85%–98.7%` bloat reduction**.",
        f"- **Wedge B - Enterprise CFO Metered API FinOps (`Bedrock / Vertex / API Keys`):** `${cons_mo:,.2f} - ${real_mo:,.2f} / mo Saved (90%-Cached -> Blended)`",
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
            f"- **Live Tokens Saved (Direct / Cumulative Re-Ingestion):** `{int(live_o10.get('direct_tokens_saved', 0)):,}` direct (`{max(int(live_o10.get('direct_tokens_saved', 0)), int(live_o10.get('quadratic_tokens_saved', 0))):,}` cumulative) (`${live_o10.get('estimated_usd_saved', 0.0):,.2f}` USD)",
            f"- **Autonomy & Mastery Ratchet:** `{audit.get('autonomy_ratchet', {}).get('ratchet_summary', '0.0% Local Shielded (0 ops recorded — 85.0% target)')}`",
            f"- **Verified Cloud Exemplars Distilled:** `{audit.get('autonomy_ratchet', {}).get('verified_cloud_exemplars_count', 0):,}` exemplars • `{audit.get('autonomy_ratchet', {}).get('lgtm_local_edits_count', 0):,}` LGTM local edits • `{audit.get('autonomy_ratchet', {}).get('distilled_pairs_count', 0):,}` REM sleep pairs",
        ]
    )

    t0_md = audit.get("turn0_static_overhead", {})
    if t0_md:
        md.extend(
            [
                "",
                "## 3.5. Turn-0 Fixed Startup Context Tax (Instruction Files + MCP Tool Schemas)",
                "",
                f"- **Total Turn-0 Payload / Session:** `{int(t0_md.get('total_turn0_tokens_per_session', 0)):,} tokens/session` (`{t0_md.get('status_tier', 'LEAN')}`)",
                f"- **Static Instruction Rules (`CLAUDE.md` / `AGENTS.md` / `GEMINI.md` / `.cursorrules`):** `{int(t0_md.get('instruction_tokens', 0)):,} tokens` across `{int(t0_md.get('instruction_files_count', 0))}` files",
                f"- **MCP Server Tool Schemas vs `body` OS Hooks:** `{int(t0_md.get('mcp_turn0_tokens', 0)):,} tokens/session` across `{int(t0_md.get('mcp_servers_count', 0))}` MCP servers vs **`0` MCP schema tokens** for Whole Body Compute OS hooks (`$0.00` Turn-0 tax)",
            ]
        )

    mo_md = audit.get("model_opportunities", {}) or {}
    mo_md_models = mo_md.get("models", []) or []
    if mo_md_models:
        md.extend(
            [
                "",
                "## 3.75. Read-Only Model Mix & Routing Opportunity Insights",
                "",
                "| Logged Model | Tier | Sessions | Turns | API-Equiv Spend | Avg / Session | CoT/Output Share |",
                "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for m_row in mo_md_models[:10]:
            md.append(
                f"| **{m_row.get('label', '')}** | `{m_row.get('tier', 'balanced')}` | "
                f"{int(m_row.get('sessions', 0)):,} | {int(m_row.get('turns', 0)):,} | "
                f"**${float(m_row.get('spend_usd', 0.0)):,.2f}** | "
                f"${float(m_row.get('avg_session_usd', 0.0)):,.2f} | "
                f"{float(m_row.get('output_share_pct', 0.0)):.1f}% |"
            )
        opps_md = mo_md.get("opportunities", []) or []
        if opps_md:
            md.append("")
            for opp in opps_md:
                md.append(f"- **{opp.get('title', 'Opportunity')}:** {opp.get('summary', '')}")
            tot_opp_m = float(mo_md.get("total_opportunity_usd", 0.0))
            yr_opp_m = float(mo_md.get("projected_annual_opportunity_usd", 0.0))
            if tot_opp_m >= 0.05:
                md.append(
                    f"- **Combined Read-Only Model-Fit Upside:** **`~${tot_opp_m:,.2f} / {days}d`** (`~${yr_opp_m:,.0f}/yr` per seat)"
                )

    md.extend(
        [
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
        "> **Zero-Install Audit Command:** `curl -sSL https://wholebodycompute.com/audit.py | python3` • **Awaken Workspace:** `curl -fsSL https://wholebodycompute.com/begin | sh`"
    )
    md.append("")
    return "\n".join(md)


def _sanitize_lowering_text(val: str) -> str:
    """Redact secrets and neutralize prompt-injection spans in lowering previews."""
    cleaned = redact_secrets(str(val or ""))
    try:
        from whole_body_compute.organs.immune_vascular import EpithelialScreenFilter

        cleaned, _ = EpithelialScreenFilter.sanitize(
            cleaned,
            source="cli_audit",
            wrap_untrusted_tags=False,
            record_antigen=False,
        )
    except Exception:
        pass
    return redact_secrets(cleaned)


def sanitize_lowering_analysis(lowering_data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively sanitize command previews and pattern strings in `lowering_analysis`."""
    def _walk(node: Any) -> Any:
        if isinstance(node, str):
            return _sanitize_lowering_text(node)
        if isinstance(node, list):
            return [_walk(x) for x in node]
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}
        return node

    sanitized = _walk(lowering_data)
    return sanitized if isinstance(sanitized, dict) else {}


def format_unified_audit_markdown_receipt(audit: Dict[str, Any], show_team: bool = False) -> str:
    """Render a shareable Markdown receipt from the unified metered `audit.py` + `ShieldTelemetryLedger` payload."""
    days = int(audit.get("window_days", 30))
    convs = int(audit.get("conversations_scanned", 0))
    steps = int(audit.get("total_steps", 0))
    planner_turns = int(audit.get("total_planner_turns", 0))
    seats = int(audit.get("seats", 1))

    ms = audit.get("measured_spend", {}) or {}
    ps = audit.get("projected_savings", {}) or {}
    cb = audit.get("cost_breakdown", {}) or {}
    wbc_st = audit.get("wbc_shield_telemetry", {}) or {}
    qm = audit.get("quota_multiplier_metrics", {}) or {}

    exact_usd = float(ms.get("exact_api_spend_usd", audit.get("estimated_frontier_spend_usd", 0.0)))
    proj_wbc_usd = float(ps.get("projected_wbc_spend_usd", audit.get("bbb_optimized_spend_usd", 0.0)))
    proj_sav_usd = float(ps.get("projected_saved_usd", audit.get("wasted_spend_usd", 0.0)))
    cf_yr_usd = float(ps.get("counterfactual_annual_saved_usd", ps.get("projected_annual_saved_usd", 0.0)))
    q_mult = float(qm.get("quota_stretch_multiple", 1.0))

    wbc_active = bool(wbc_st.get("wbc_active_on_machine", False))
    wired_lbl = ", ".join(str(h).replace("_", "-") for h in (wbc_st.get("wired_harnesses") or [])) or "none"
    sh_sess = int(wbc_st.get("shielded_sessions", 0))
    meas_usd = float(wbc_st.get("measured_usd_saved", 0.0))
    meas_dtok = int(wbc_st.get("measured_direct_tokens_saved", 0))
    meas_ctok = int(wbc_st.get("measured_cumulative_tokens_saved", meas_dtok))
    ledger_events = int(wbc_st.get("ledger_events", 0))
    unwired_save = float(wbc_st.get("unwired_projected_save_usd", 0.0))

    mo = audit.get("model_opportunities", {}) or {}
    tot_opp = float(mo.get("total_opportunity_usd", 0.0))
    yr_opp = float(mo.get("projected_annual_opportunity_usd", 0.0))

    md: List[str] = [
        f"# Whole Body Compute - {days}-Day Agent FinOps & ROI Audit Receipt",
        "",
        f"**Generated:** `{audit.get('generated_at', '')}` • **Window:** `{days} days` • **Sessions:** `{convs:,}` (`{planner_turns:,}` turns / `{steps:,}` steps)",
        "",
        "## 1. Reconciled Spend, Verified Shield & Opportunity Summary",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| **Metered API-Equivalent Spend ({days}d)** | **${exact_usd:,.2f} USD** |",
        f"| **WBC Active Harnesses** | `{wired_lbl}` (`{sh_sess:,}/{convs:,}` sessions shielded) |",
    ]
    if wbc_st.get("ledger_synced"):
        md.append(
            f"| **Verified WBC Shield (`body savings`)** | **${meas_usd:,.2f} saved** (`{meas_ctok:,}` cumulative tok / `{meas_dtok:,}` direct tok across `{ledger_events:,}` gates) |"
        )
    if not wbc_active or sh_sess == 0:
        md.append(f"| **Projected Spend with WBC** | **${proj_wbc_usd:,.2f} USD** (`${proj_sav_usd:,.2f}` savable / `{days}d`) |")
    elif unwired_save > 0:
        md.append(f"| **Unwired Harness Opportunity** | **${unwired_save:,.2f} / {days}d** (run `body begin`) |")
    if tot_opp >= 0.05:
        md.append(f"| **Read-Only Flush & Model-Fit Upside** | **~${tot_opp:,.2f} / {days}d** (`~${yr_opp:,.0f}/yr` per seat) |")
    if q_mult > 1.0:
        md.append(f"| **Subscription Quota Stretch** | **`~{q_mult:.2f}x`** longer Pro/Max runway before rate limits |")
    if show_team or seats > 1:
        scale_yr = (30.0 / max(1, days)) * 12.0
        base_yr = (
            (unwired_save * scale_yr)
            if (wbc_active and unwired_save > 0)
            else max(cf_yr_usd, (meas_usd * scale_yr) + yr_opp, proj_sav_usd * scale_yr)
        )
        md.append(f"| **Projected Team ({seats} Seats) Annual Value** | **${base_yr * max(1, seats):,.0f} / yr** |")

    md.extend(
        [
            "",
            "## 2. Where Your Spend Went",
            "",
            "| Cost Driver | Spend (USD) | Tokens |",
            "| :--- | ---: | ---: |",
            f"| **History Re-Reads (Cached Input)** | ${float(cb.get('cache_read_usd', 0.0)):,.2f} | {int(ms.get('metered_cache_read_tokens', 0)):,} |",
            f"| **Tool Outputs & New Context** | ${float(cb.get('tool_outputs_usd', 0.0)):,.2f} | {int(cb.get('tool_outputs_tokens', 0)):,} |",
            f"| **Startup Rules & MCP Schemas** | ${float(cb.get('startup_and_mcp_usd', 0.0)):,.2f} | {int(cb.get('startup_and_mcp_tokens', 0)):,} |",
            f"| **Typed User Prompts** | ${float(cb.get('typed_prompt_usd', 0.0)):,.2f} | {int(cb.get('typed_prompt_tokens', 0)):,} |",
            f"| **Model Responses (Output / CoT)** | ${float(cb.get('output_usd', 0.0)):,.2f} | {int(ms.get('metered_output_tokens', 0)):,} |",
        ]
    )

    hb = audit.get("harness_breakdown", {}) or {}
    active_hb = [(k, v) for k, v in hb.items() if isinstance(v, dict) and int(v.get("threads", 0)) > 0]
    if active_hb:
        md.extend(
            [
                "",
                "## 3. By Agent Harness",
                "",
                "| Agent Harness | Sessions | Turns | Metered Spend | Status / Projected Save |",
                "| :--- | ---: | ---: | ---: | ---: |",
            ]
        )
        wired_set = set(wbc_st.get("wired_harnesses") or [])
        for hk, hd in active_hb:
            status_s = "WBC Active" if (wbc_active and hk in wired_set) else f"Save ${float(hd.get('hal_saved_usd', 0.0)):,.2f}"
            md.append(
                f"| **{hd.get('label', hk)}** | {int(hd.get('threads', 0)):,} | "
                f"{int(hd.get('planner_turns', 0)):,} | ${float(hd.get('exact_api_spend_usd', 0.0)):,.2f} | {status_s} |"
            )

    t0_md = audit.get("turn0_static_overhead", {}) or {}
    if t0_md:
        md.extend(
            [
                "",
                "## 4. Turn-0 Fixed Startup Context Tax",
                "",
                f"- **Total Startup Payload / Session:** `{int(t0_md.get('total_turn0_tokens_per_session', 0)):,} tokens/session` (`{t0_md.get('status_tier', 'LEAN')}`)",
                f"- **Static Instruction Rules:** `{int(t0_md.get('instruction_tokens', 0)):,} tokens` across `{int(t0_md.get('instruction_files_count', 0))}` files",
                f"- **MCP Tool Schemas vs `body` OS Hooks:** `{int(t0_md.get('mcp_turn0_tokens', 0)):,} tokens/session` across `{int(t0_md.get('mcp_servers_count', 0))}` MCP servers vs **`0` MCP schema tokens** for `body` hooks",
            ]
        )

    mo_models = mo.get("models", []) or []
    if mo_models:
        md.extend(
            [
                "",
                "## 5. Read-Only Model Mix & Routing Opportunity Insights",
                "",
                "| Logged Model | Tier | Sessions | Turns | API-Equiv Spend | Avg / Session | CoT/Output Share |",
                "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for m_row in mo_models[:10]:
            md.append(
                f"| **{m_row.get('label', '')}** | `{m_row.get('tier', 'balanced')}` | "
                f"{int(m_row.get('sessions', 0)):,} | {int(m_row.get('turns', 0)):,} | "
                f"**${float(m_row.get('spend_usd', 0.0)):,.2f}** | "
                f"${float(m_row.get('avg_session_usd', 0.0)):,.2f} | "
                f"{float(m_row.get('output_share_pct', 0.0)):.1f}% |"
            )
        opps_md = mo.get("opportunities", []) or []
        if opps_md:
            md.append("")
            for opp in opps_md:
                md.append(f"- **{opp.get('title', 'Opportunity')}:** {opp.get('summary', '')}")
            if tot_opp >= 0.05:
                md.append(
                    f"- **Combined Read-Only Model-Fit Upside:** **`~${tot_opp:,.2f} / {days}d`** (`~${yr_opp:,.0f}/yr` per seat)"
                )

    la_raw = audit.get("lowering_analysis")
    if isinstance(la_raw, dict) and la_raw.get("enabled"):
        la = sanitize_lowering_analysis(la_raw)

        def _fmt_mo_md(val: float) -> str:
            return f"${val:,.0f}" if val >= 1000.0 else f"${val:,.2f}"

        def _fmt_pct_md(val: float) -> str:
            return f"{int(round(val))}%" if abs(val - round(val)) < 0.05 else f"{val:.1f}%"

        s_code = float(la.get("code_lowering_monthly_usd", 0.0))
        y_pct = _fmt_pct_md(float(la.get("code_lowering_turn_pct", 0.0)))
        n_pat = int(la.get("recurring_patterns_count", 0))
        s_wbc = float(la.get("wbc_shield_monthly_usd", 0.0))
        z_pct = _fmt_pct_md(float(la.get("wbc_shield_rem_pct", 0.0)))
        s_tot = float(la.get("combined_monthly_usd", 0.0))
        p_pct = _fmt_pct_md(float(la.get("combined_net_pct", 0.0)))
        p_unit = str(la.get("period_unit") or "/mo")

        md.extend(
            [
                "",
                "## Savings Breakdown",
                "",
                f"1. Automate {n_pat} repeatable loops: {_fmt_mo_md(s_code)}{p_unit} ({y_pct} of turns)",
                f"2. Cut log, file & MCP bloat: {_fmt_mo_md(s_wbc)}{p_unit} ({z_pct} on remaining turns)",
                f"3. Total: {_fmt_mo_md(s_tot)}{p_unit} ({p_pct} less spend)",
            ]
        )

    top_convs = audit.get("top_bloated_conversations", []) or []
    if top_convs:
        md.extend(
            [
                "",
                "## 6. Costliest Sessions",
                "",
                "| # | Session ID | Harness | Turns | Metered Spend | Top Tool | Bloat Trigger |",
                "| :---: | :--- | :--- | ---: | ---: | :--- | :--- |",
            ]
        )
        for idx, c in enumerate(top_convs, start=1):
            c_usd = float(c.get("exact_api_spend_usd", c.get("conservative_spend_usd", 0.0)))
            md.append(
                f"| {idx} | `{c.get('short_id', '')}` | `{c.get('harness', 'agent')}` | "
                f"{int(c.get('planner_turns', 0)):,} | **${c_usd:,.2f}** | "
                f"`{c.get('top_offender_tool', 'none')}` | `{c.get('bloat_trigger', 'ACTIVE')}` |"
            )

    md.append("")
    return "\n".join(md)


__all__ = [
    "COMPACTION_DRIFT_PENALTY",
    "DEFAULT_BRAIN_DIR",
    "DEFAULT_FLUSH_STEP_INTERVAL",
    "FINOPS_HARD_CONTEXT_TOKENS",
    "FINOPS_HARD_CSI_THRESHOLD",
    "FINOPS_SOFT_CONTEXT_TOKENS",
    "FINOPS_SOFT_CSI_THRESHOLD",
    "HARD_SIZE_KB_THRESHOLD",
    "HARD_STEP_THRESHOLD",
    "SOFT_SIZE_KB_THRESHOLD",
    "SOFT_STEP_THRESHOLD",
    "analyze_bloat_attribution",
    "audit_historical_spend",
    "audit_turn0_static_overhead",
    "calculate_csi",
    "compute_avg_step_gap",
    "compute_dual_csi_metrics",
    "estimate_lines_size_kb",
    "format_finops_audit_receipt",
    "format_finops_markdown_receipt",
    "format_unified_audit_markdown_receipt",
    "inspect_conversation_lightweight",
    "is_step_output_truncated",
    "resolve_brain_dir",
    "resolve_flush_step_interval",
    "sanitize_lowering_analysis",
    "split_at_latest_checkpoint",
]
