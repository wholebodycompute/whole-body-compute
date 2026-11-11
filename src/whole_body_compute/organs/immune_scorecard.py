"""
Blood-Brain Barrier (Organ 05): Pre/Post Immune Gate Scorecard, Bicameral Compute Split & OrganImmuneSystem Facade.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from whole_body_compute.organs.immune_shell import (
    DB_RESOURCE_WEB_BROWSER_TOOL_NAMES,
    READ_TOOL_NAMES,
    SEARCH_DIR_TOOL_NAMES,
    SHELL_TOOL_NAMES,
    WRITE_EDIT_TOOL_NAMES,
    inspect_shell_command_bloat,
    parse_window_to_seconds,
)
from whole_body_compute.organs.immune_slicers import (
    _get_audit_log_path,
    slice_post_tool_output_buffer,
)
from whole_body_compute.organs.immune_vascular import AntibodyRegistry


def compute_percentage_split(
    local_ops: int, subagent_ops: int, cloud_ops: int
) -> Tuple[int, int, int]:
    """
    Compute integer percentages for Bicameral Compute Split that always sum to exactly 100%
    using the Largest Remainder Method (Hare-Niemeyer method).
    When total_ops == 0, returns honest cold-start (0, 0, 0).
    """
    total = local_ops + subagent_ops + cloud_ops
    if total <= 0:
        return 0, 0, 0

    raw = [
        (local_ops / total) * 100.0,
        (subagent_ops / total) * 100.0,
        (cloud_ops / total) * 100.0,
    ]
    floors = [int(math.floor(x)) for x in raw]
    remainder = 100 - sum(floors)

    diffs = [(raw[i] - floors[i], i) for i in range(3)]
    diffs.sort(reverse=True, key=lambda x: x[0])

    for _, idx in diffs[:remainder]:
        floors[idx] += 1

    return floors[0], floors[1], floors[2]


def _classify_bicameral_event(e: Dict[str, Any]) -> Optional[str]:
    """
    Classify a single audit pre-event into `"local"`, `"subagent"`, `"cloud"`, or `None` (no-op heartbeat).
    Normalizes multi-harness tool aliases across Antigravity and Claude Code.
    """
    subagent_tool_names = {"invoke_subagent", "send_message", "manage_subagents", "task", "agent"}
    read_low_names = {x.lower() for x in READ_TOOL_NAMES}
    write_edit_low_names = {x.lower() for x in WRITE_EDIT_TOOL_NAMES}
    shell_low_names = {x.lower() for x in SHELL_TOOL_NAMES}
    search_dir_low_names = {x.lower() for x in SEARCH_DIR_TOOL_NAMES}
    db_web_browser_low_names = {x.lower() for x in DB_RESOURCE_WEB_BROWSER_TOOL_NAMES} | {
        "search_web",
        "ask_question",
        "call_mcp_tool",
    }
    local_cli_keywords = (
        "whole_body_compute.local",
        "whole_body_compute",
        "local.scout",
        "local.implementer",
        "local.agent_loop",
        "daily_efficiency_audit.py",
        "local_qa.py",
        "vifi",
        "body",
        "scout",
        "implementer",
        "supervisor",
        "agent_loop",
        "rem_sleep",
        "dot_memory",
        "pytest",
        "ruff",
        "git",
    )

    t_name = str(e.get("tool") or "").strip().lower()
    target = str(e.get("target") or "").lower()
    ev_stage = str(e.get("event") or e.get("gate_stage") or "").lower()
    is_slice = bool(
        e.get("bounded_slice_read")
        or ev_stage in ("pre_tool_slice_allowed", "pre_tool_ast_slicer_block")
        or (e.get("fast_path") and t_name in read_low_names)
    )
    is_healed = bool(e.get("auto_healed") or int(e.get("em_dash_healed", 0)) > 0)
    is_blocked = (
        e.get("decision") in ("ask", "force_ask", "deny")
        or bool(e.get("context_gate_triggered"))
    )

    if t_name in ("tool", "unknown", "", "pre_invocation") and not (
        is_slice or is_blocked or is_healed
    ):
        return None

    if t_name in subagent_tool_names or "subagent" in t_name:
        return "subagent"
    if is_slice or is_blocked or is_healed:
        return "local"
    if t_name in ("whole_body_compute", "local_scout", "local_implementer", "vifi", "body", "supervisor"):
        return "local"
    if t_name in shell_low_names:
        if any(kw in target for kw in local_cli_keywords):
            return "local"
        if "curl" in target or "http" in target or "api." in target:
            return "cloud"
        return "local"
    if (
        t_name in write_edit_low_names
        or t_name in read_low_names
        or t_name in search_dir_low_names
        or t_name in db_web_browser_low_names
    ):
        return "cloud"
    return "cloud"


def compute_bicameral_split_metrics(
    pre_events: List[Dict[str, Any]],
    window_sec: Optional[float] = None,
    include_transcript_subagents: bool = True,
) -> Dict[str, Any]:
    """
    Compute Bicameral Compute Split metrics (Layer 1 Local $0 vs Layer 2 Subagent vs Layer 3 Cloud).
    Also tracks `local_model_shield_ops` vs `local_cli_ops` so callers can distinguish active
    Local Metal LLM/AST/Foveal offloads from routine shell verification commands.
    """
    local_ops = 0
    local_model_shield_ops = 0
    local_cli_ops = 0
    subagent_ops = 0
    cloud_ops = 0

    subagent_tool_names = {"invoke_subagent", "send_message", "manage_subagents", "task", "agent"}
    cloud_tool_names = {"write_file", "edit_file", "search_web", "ask_question", "call_mcp_tool"}
    local_model_cli_keywords = (
        "whole_body_compute.local",
        "local.scout",
        "local.implementer",
        "local.agent_loop",
        "body scout",
        "body implement",
        "body loop",
        "body eyes",
        "body airlock",
        "vifi voice",
    )
    local_cli_keywords = (
        "whole_body_compute.local",
        "whole_body_compute",
        "local.scout",
        "local.implementer",
        "local.agent_loop",
        "daily_efficiency_audit.py",
        "local_qa.py",
        "vifi",
        "body",
        "scout",
        "implementer",
        "supervisor",
        "agent_loop",
        "rem_sleep",
        "dot_memory",
        "pytest",
        "ruff",
        "git",
    )

    audit_subagent_ops = 0
    for e in pre_events:
        t_name = str(e.get("tool") or "").strip().lower()
        target = str(e.get("target") or "").lower()
        ev_stage = str(e.get("event") or e.get("gate_stage") or "").lower()
        is_slice = bool(
            e.get("bounded_slice_read")
            or ev_stage in ("pre_tool_slice_allowed", "pre_tool_ast_slicer_block")
            or (e.get("fast_path") and t_name in ("read_file", "view_file"))
        )
        is_healed = bool(e.get("auto_healed") or int(e.get("em_dash_healed", 0)) > 0)
        is_blocked = (
            e.get("decision") in ("ask", "force_ask", "deny")
            or bool(e.get("context_gate_triggered"))
        )

        if t_name in ("tool", "unknown", "", "pre_invocation") and not (
            is_slice or is_blocked or is_healed or target
        ):
            continue

        if t_name in subagent_tool_names or "subagent" in t_name:
            subagent_ops += 1
            audit_subagent_ops += 1
        elif is_slice or is_blocked or is_healed:
            local_ops += 1
            local_model_shield_ops += 1
        elif t_name in ("whole_body_compute", "local_scout", "local_implementer", "vifi", "body", "supervisor"):
            local_ops += 1
            local_model_shield_ops += 1
        elif t_name == "run_command":
            if any(kw in target for kw in local_model_cli_keywords):
                local_ops += 1
                local_model_shield_ops += 1
            elif any(kw in target for kw in local_cli_keywords):
                local_ops += 1
                local_cli_ops += 1
            elif "curl" in target or "http" in target or "api." in target:
                cloud_ops += 1
            else:
                local_ops += 1
                local_cli_ops += 1
        elif t_name in cloud_tool_names:
            cloud_ops += 1
        elif t_name in ("read_file", "view_file"):
            cloud_ops += 1
        elif t_name in ("tool", "unknown", ""):
            continue
        else:
            cloud_ops += 1

    # Only fall back to parsing actual PLANNER_RESPONSE tool_calls in transcripts when the
    # PreToolUse audit log did not capture subagent tool calls directly (never substring-match raw lines).
    if include_transcript_subagents and audit_subagent_ops == 0 and os.environ.get("VOICEFI_IMMUNE_AUDIT_PATH") is None:
        try:
            brain_dir = Path(os.path.expanduser("~/.gemini/antigravity/brain"))
            if brain_dir.is_dir():
                cutoff_ts = (time.time() - window_sec) if window_sec is not None else (time.time() - 86400.0)
                import glob

                transcript_sub_ops = 0
                for tpath_str in glob.glob(str(brain_dir / "*" / ".system_generated" / "logs" / "transcript.jsonl")):
                    try:
                        mtime = os.path.getmtime(tpath_str)
                        if mtime < cutoff_ts:
                            continue
                        with open(tpath_str, "r", encoding="utf-8", errors="replace") as tf:
                            for tline in tf:
                                if '"PLANNER_RESPONSE"' not in tline:
                                    continue
                                if "invoke_subagent" not in tline and "send_message" not in tline:
                                    continue
                                obj = json.loads(tline)
                                if obj.get("type") != "PLANNER_RESPONSE":
                                    continue
                                for tc in obj.get("tool_calls") or []:
                                    tc_name = str(tc.get("name") or tc.get("tool") or "").lower()
                                    if tc_name in ("invoke_subagent", "send_message"):
                                        transcript_sub_ops += 1
                    except Exception:
                        continue
                subagent_ops = max(subagent_ops, transcript_sub_ops)
        except Exception:
            pass

    total_ops = local_ops + subagent_ops + cloud_ops
    is_live = total_ops > 0

    local_pct, subagent_pct, cloud_pct = compute_percentage_split(
        local_ops, subagent_ops, cloud_ops
    )
    model_shield_pct = round((local_model_shield_ops / total_ops) * 100.0, 1) if total_ops > 0 else 0.0

    if is_live:
        formatted = f"{local_pct}% Local $0 • {subagent_pct}% Sub • {cloud_pct}% Cloud"
    else:
        formatted = "0% Local • 0% Sub • 0% Cloud (0 ops — 85/12/3 target)"
    target_formatted = "85% Local $0 • 12% Sub • 3% Cloud"

    return {
        "local_ops": local_ops,
        "local_model_shield_ops": local_model_shield_ops,
        "local_cli_ops": local_cli_ops,
        "local_model_shield_pct": model_shield_pct,
        "subagent_ops": subagent_ops,
        "cloud_ops": cloud_ops,
        "total_ops": total_ops,
        "local_pct": local_pct,
        "subagent_pct": subagent_pct,
        "cloud_pct": cloud_pct,
        "is_live": is_live,
        "formatted": formatted,
        "target_formatted": target_formatted,
    }


def get_immune_pre_post_scorecard(
    limit: int = 50,
    window: str = "all",
    show_tools: bool = False,
    show_blocked: bool = False,
    conv_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute empirical Pre/Post Immune Gate & Context-Shield metrics from `~/.voicefi/immune_audit.jsonl`.
    """
    log_path = _get_audit_log_path()
    pre_events: List[Dict[str, Any]] = []
    post_events: List[Dict[str, Any]] = []
    pre_inv_events: List[Dict[str, Any]] = []
    recent_events: List[Dict[str, Any]] = []
    session_pre_events: List[Dict[str, Any]] = []
    cid_prefix = str(conv_id).strip()[:8] if conv_id else ""

    window_sec = parse_window_to_seconds(window)
    cutoff = (time.time() - window_sec) if window_sec is not None else None

    if log_path.is_file():
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in lines:
                if not line.strip():
                    continue
                try:
                    ev = json.loads(line)
                    ev_ts = float(ev.get("timestamp", 0.0))
                    if cutoff is not None and ev_ts < cutoff:
                        continue
                    recent_events.append(ev)
                    if ev.get("phase") == "pre":
                        pre_events.append(ev)
                        if cid_prefix and str(ev.get("conversation_id") or "").startswith(cid_prefix):
                            session_pre_events.append(ev)
                    elif ev.get("phase") == "post":
                        post_events.append(ev)
                    elif ev.get("phase") == "pre_invocation":
                        pre_inv_events.append(ev)
                except Exception:
                    continue
        except Exception:
            pass

    avg_pre_ms = (
        round(sum(float(e.get("latency_ms", 0.0)) for e in pre_events) / len(pre_events), 2)
        if pre_events
        else 0.0
    )
    avg_post_ms = (
        round(sum(float(e.get("latency_ms", 0.0)) for e in post_events) / len(post_events), 2)
        if post_events
        else 0.0
    )
    fast_path_count = sum(1 for e in pre_events if e.get("fast_path"))
    deep_ast_count = len(pre_events) - fast_path_count
    allow_count = sum(1 for e in pre_events if e.get("decision") == "allow")
    auto_healed_count = sum(1 for e in pre_events if e.get("auto_healed"))
    em_dashes_scrubbed = sum(int(e.get("em_dash_healed", 0)) for e in pre_events)
    machine_marks_scrubbed = sum(
        int(e.get("machine_marks_scrubbed", e.get("em_dash_healed", 0)))
        for e in pre_events
    ) + sum(int(e.get("machine_marks_scrubbed", 0)) for e in recent_events if e.get("phase") == "stop")
    intercept_count = sum(1 for e in pre_events if e.get("decision") in ("ask", "force_ask", "deny"))
    net_assert_delta = sum(
        int(e.get("assert_delta", 0)) for e in pre_events if e.get("decision") == "allow"
    )
    secret_alerts = sum(1 for e in post_events if e.get("secret_in_output"))

    context_gate_intercepts = sum(1 for e in pre_events if e.get("context_gate_triggered"))
    bounded_slice_reads = sum(
        1 for e in pre_events if e.get("bounded_slice_read") or e.get("read_auto_sliced")
    )
    post_sliced_count = sum(1 for e in post_events if e.get("post_output_sliced"))
    direct_tokens_saved = sum(int(e.get("direct_tokens_saved", 0)) for e in pre_events) + sum(
        int(e.get("direct_tokens_saved", 0)) for e in post_events
    )
    quadratic_tokens_saved = sum(
        int(e.get("quadratic_tokens_saved", 0)) for e in pre_events
    ) + sum(int(e.get("quadratic_tokens_saved", 0)) for e in post_events)
    direct_usd_saved = round((direct_tokens_saved / 1_000_000.0) * 3.00, 2)
    reingestion_tokens = max(0, quadratic_tokens_saved - direct_tokens_saved)
    cached_reingestion_usd = round((reingestion_tokens / 1_000_000.0) * 0.30, 2)
    estimated_usd_saved = round(direct_usd_saved + cached_reingestion_usd, 2)
    pre_invocation_nudges = len(pre_inv_events)

    cutoff_24h = time.time() - 86400.0
    pre_24h = [e for e in pre_events if float(e.get("timestamp", 0.0)) >= cutoff_24h]
    intercepted_24h = sum(
        1 for e in pre_24h if e.get("decision") in ("ask", "force_ask", "deny")
    )

    tools_stats: Dict[str, Dict[str, Any]] = {}
    for e in pre_events:
        t_name = str(e.get("tool") or "unknown").strip() or "unknown"
        if t_name not in tools_stats:
            tools_stats[t_name] = {
                "total": 0,
                "allowed": 0,
                "intercepted": 0,
                "auto_healed": 0,
                "slices": 0,
                "direct_tokens_saved": 0,
                "quadratic_tokens_saved": 0,
                "estimated_usd_saved": 0.0,
                "allowed_pct": 0.0,
                "intercepted_pct": 0.0,
            }
        st = tools_stats[t_name]
        st["total"] += 1
        dec = str(e.get("decision", "allow")).lower()
        if dec == "allow":
            st["allowed"] += 1
        elif dec in ("ask", "force_ask", "deny"):
            st["intercepted"] += 1
        if e.get("auto_healed"):
            st["auto_healed"] += 1
        if e.get("bounded_slice_read") or e.get("read_auto_sliced"):
            st["slices"] += 1
        st["direct_tokens_saved"] += int(e.get("direct_tokens_saved", 0) or 0)
        st["quadratic_tokens_saved"] += int(e.get("quadratic_tokens_saved", 0) or 0)

    for e in post_events:
        d_post = int(e.get("direct_tokens_saved", 0) or 0)
        q_post = int(e.get("quadratic_tokens_saved", 0) or 0)
        if d_post <= 0 and q_post <= 0 and not e.get("post_output_sliced"):
            continue
        t_name = str(e.get("tool") or "unknown").strip() or "unknown"
        if t_name not in tools_stats:
            tools_stats[t_name] = {
                "total": 1,
                "allowed": 1,
                "intercepted": 0,
                "auto_healed": 0,
                "slices": 0,
                "direct_tokens_saved": 0,
                "quadratic_tokens_saved": 0,
                "estimated_usd_saved": 0.0,
                "allowed_pct": 100.0,
                "intercepted_pct": 0.0,
            }
        st = tools_stats[t_name]
        if e.get("post_output_sliced"):
            st["slices"] += 1
        st["direct_tokens_saved"] += d_post
        st["quadratic_tokens_saved"] += q_post

    for t_name, st in tools_stats.items():
        tot = st["total"]
        if tot > 0:
            st["allowed_pct"] = round((st["allowed"] / tot) * 100.0, 1)
            st["intercepted_pct"] = round((st["intercepted"] / tot) * 100.0, 1)
        d_tok = int(st["direct_tokens_saved"])
        q_tok = int(st["quadratic_tokens_saved"])
        d_usd = (d_tok / 1_000_000.0) * 3.00
        c_usd = (max(0, q_tok - d_tok) / 1_000_000.0) * 0.30
        st["estimated_usd_saved"] = round(d_usd + c_usd, 2)

    tools_breakdown = dict(
        sorted(
            tools_stats.items(),
            key=lambda kv: (
                kv[1]["estimated_usd_saved"],
                kv[1]["direct_tokens_saved"],
                kv[1]["total"],
            ),
            reverse=True,
        )
    )

    blocked_dict: Dict[str, Dict[str, Any]] = {}
    for e in pre_events:
        dec = str(e.get("decision", "allow")).lower()
        if dec in ("ask", "force_ask", "deny"):
            t_name = str(e.get("tool") or "unknown").strip()
            raw_tgt = str(e.get("target") or "unknown").strip()
            key = f"{t_name}::{raw_tgt}"
            reason_str = str(e.get("reason") or "Blocked by Blood-Brain Barrier")
            r_low = reason_str.lower()

            if "context-tax gate" in r_low or "context shield" in r_low:
                cat = "unbounded read_file" if t_name in ("read_file", "view_file") else "shell context bloat"
            elif "vascular" in r_low or "egress" in r_low:
                cat = "vascular egress"
            elif "ast secgate" in r_low:
                cat = "ast secgate"
            elif "antibody" in r_low:
                cat = "adaptive antibody"
            elif "credential" in r_low:
                cat = "credential exfiltration"
            else:
                cat = "policy violation"

            if key not in blocked_dict:
                blocked_dict[key] = {
                    "tool": t_name,
                    "target": raw_tgt,
                    "category": cat,
                    "count": 0,
                    "reasons": set(),
                    "last_timestamp": float(e.get("timestamp", 0.0)),
                }
            entry = blocked_dict[key]
            entry["count"] += 1
            entry["reasons"].add(reason_str)
            entry["last_timestamp"] = max(entry["last_timestamp"], float(e.get("timestamp", 0.0)))

    top_blocked = []
    for entry in sorted(blocked_dict.values(), key=lambda x: x["count"], reverse=True):
        top_blocked.append(
            {
                "tool": entry["tool"],
                "target": entry["target"],
                "category": entry["category"],
                "count": entry["count"],
                "reasons": sorted(list(entry["reasons"])),
                "last_timestamp": entry["last_timestamp"],
            }
        )

    return {
        "audit_log_path": str(log_path),
        "window": window,
        "window_seconds": window_sec,
        "total_pre_checks": len(pre_events),
        "total_post_verifications": len(post_events),
        "fast_path_count": fast_path_count,
        "deep_ast_count": deep_ast_count,
        "checks_24h": len(pre_24h),
        "intercepted_24h": intercepted_24h,
        "avg_pre_latency_ms": avg_pre_ms,
        "avg_post_latency_ms": avg_post_ms,
        "allowed_count": allow_count,
        "auto_healed_count": auto_healed_count,
        "em_dashes_scrubbed": em_dashes_scrubbed,
        "machine_marks_scrubbed": max(machine_marks_scrubbed, em_dashes_scrubbed),
        "intercepted_count": intercept_count,
        "net_assertion_delta": net_assert_delta,
        "secret_output_alerts": secret_alerts,
        "active_antibodies": len(AntibodyRegistry.get_instance().list_antibodies()),
        "context_gate_intercepts": context_gate_intercepts,
        "bounded_slice_reads": bounded_slice_reads,
        "post_sliced_count": post_sliced_count,
        "direct_tokens_saved": direct_tokens_saved,
        "quadratic_tokens_saved": quadratic_tokens_saved,
        "direct_usd_saved": direct_usd_saved,
        "cached_reingestion_usd": cached_reingestion_usd,
        "estimated_usd_saved": estimated_usd_saved,
        "pre_invocation_nudges": pre_invocation_nudges,
        "recent_events": recent_events[-limit:],
        "session_pre_events": session_pre_events,
        "session_compute_split": (
            compute_bicameral_split_metrics(
                session_pre_events, window_sec=window_sec, include_transcript_subagents=False
            )
            if cid_prefix
            else None
        ),
        "tools_breakdown": tools_breakdown,
        "top_blocked": top_blocked,
        "compute_split": compute_bicameral_split_metrics(pre_events, window_sec=window_sec),
    }


def get_context_shield_summary(
    limit: int = 50,
    window: str = "all",
    show_tools: bool = False,
    show_blocked: bool = False,
) -> Dict[str, Any]:
    """
    Expose empirical Pre/Post Immune Gate & Context-Shield metrics including Bicameral Compute Split.
    """
    return get_immune_pre_post_scorecard(
        limit=limit,
        window=window,
        show_tools=show_tools,
        show_blocked=show_blocked,
    )


def format_tools_breakdown_table(sc: Dict[str, Any]) -> str:
    """Format tool invocation breakdown table ranked by estimated USD & token savings."""
    tools_bd = sc.get("tools_breakdown") or {}
    win = sc.get("window", "all")
    win_str = f"Window: {win}" if win != "all" else "All Time"
    lines = [
        f"\n🔧 Tool Invocation Breakdown ({win_str} • Ranked by Savings):",
        "┌──────────────────────┬────────┬─────────┬──────────────┬────────┬────────────────┬──────────────┐",
        "│ Tool Name            │ Invocs │ Allowed │ Intercepted  │ Slices │ Direct Tokens  │ Est. Saved $ │",
        "├──────────────────────┼────────┼─────────┼──────────────┼────────┼────────────────┼──────────────┤",
    ]
    if not tools_bd:
        lines.append("│ (No tool invocations recorded in this window)                                                   │")
    else:
        for tool_name, st in tools_bd.items():
            t_name = str(tool_name)[:20].ljust(20)
            inv = str(st.get("total", 0)).rjust(6)
            alw = str(st.get("allowed", 0)).rjust(7)
            itc = str(st.get("intercepted", 0)).rjust(12)
            slc = str(st.get("slices", 0)).rjust(6)
            d_tok = f"{int(st.get('direct_tokens_saved', 0)):,}".rjust(14)
            usd_s = f"${float(st.get('estimated_usd_saved', 0.0)):,.2f}".rjust(12)
            lines.append(f"│ {t_name} │ {inv} │ {alw} │ {itc} │ {slc} │ {d_tok} │ {usd_s} │")
    lines.append("└──────────────────────┴────────┴─────────┴──────────────┴────────┴────────────────┴──────────────┘")
    return "\n".join(lines)


def format_top_blocked_table(sc: Dict[str, Any]) -> str:
    """Format top blocked files/commands table."""
    top_blkd = sc.get("top_blocked") or []
    win = sc.get("window", "all")
    win_str = f"Window: {win}" if win != "all" else "All Time"
    lines = [
        f"\n🛑 Top Blocked Files & Commands ({win_str}):",
        "┌──────┬─────────────┬─────────────────────┬──────────────────────────────┬──────────────────────────────────────────┐",
        "│ Blkd │ Tool        │ Category            │ Target                       │ Primary Reason                           │",
        "├──────┼─────────────┼─────────────────────┼──────────────────────────────┼──────────────────────────────────────────┤",
    ]
    if not top_blkd:
        lines.append("│ (No actions intercepted or blocked in this window)                                                     │")
    else:
        for entry in top_blkd[:10]:
            cnt = str(entry.get("count", 1)).rjust(4)
            tool = str(entry.get("tool", ""))[:11].ljust(11)
            cat = str(entry.get("category", ""))[:19].ljust(19)
            raw_tgt = str(entry.get("target", "")).replace("\n", " ").replace("\r", " ")
            tgt = Path(raw_tgt).name if "/" in raw_tgt and " " not in raw_tgt else raw_tgt
            tgt_disp = tgt[:28].ljust(28)
            reasons = entry.get("reasons") or ["Blocked by Blood-Brain Barrier"]
            primary_reason = str(reasons[0]).replace("\n", " ")[:40].ljust(40)
            lines.append(f"│ {cnt} │ {tool} │ {cat} │ {tgt_disp} │ {primary_reason} │")
    lines.append("└──────┴─────────────┴─────────────────────┴──────────────────────────────┴──────────────────────────────────────────┘")
    return "\n".join(lines)


def format_immune_scorecard_table(
    limit: int = 10,
    window: str = "all",
    show_tools: bool = False,
    show_blocked: bool = False,
) -> str:
    """Format the Pre/Post Immune Audit & Context Shield Scorecard for terminal display."""
    sc = get_immune_pre_post_scorecard(
        limit=limit,
        window=window,
        show_tools=show_tools,
        show_blocked=show_blocked,
    )
    win_label = f"Window: {window}" if window and window.lower() != "all" else "All Time"
    lines = [
        f"🛡️  Organ 05 (Blood-Brain Barrier) - Pre/Post Immune Gate, Context Shield & Vascular Scorecard ({win_label})",
        "┌──────────────────────────────┬──────────────────────────────────────────────────────────┐",
        f"│ Pre-Tool Checks / Post-Audits│ {sc['total_pre_checks']} pre-checks ({sc['fast_path_count']} fast-sh / {sc['deep_ast_count']} deep-AST) • {sc['total_post_verifications']} post │",
        f"│ Avg Overhead (Pre / Post)    │ {sc['avg_pre_latency_ms']:.2f}ms pre-gate  •  {sc['avg_post_latency_ms']:.2f}ms post-tactile{'':<19}│",
        f"│ Gate Decisions (All / 24h)   │ 🟢 {sc['allowed_count']} allowed  •  🩹 {sc['auto_healed_count']} healed  •  🛑 {sc['intercepted_count']} intercepted ({sc['intercepted_24h']} 24h)│",
        f"│ Context Shield (AST / Slices)│ 🛡️ {sc['context_gate_intercepts']} full-reads blocked • ⚡ {sc['bounded_slice_reads']} slices • 💬 {sc['pre_invocation_nudges']} nudges│",
        f"│ Bicameral Compute Split      │ ⚖️ {sc.get('compute_split', {}).get('formatted', '0% Local • 0% Sub • 0% Cloud (0 ops)'):<54}│",
        f"│ Cloud Tokens Saved (Shield)  │ 📉 {sc['direct_tokens_saved']:,} dir (${sc.get('direct_usd_saved', 0):.2f}) • 🔥 {sc['quadratic_tokens_saved']:,} quad (~${sc['estimated_usd_saved']:.2f}){'':<3}│",
        f"│ Net Test Assertion Delta     │ {sc['net_assertion_delta']:+d} assertions across audited Python edits{'':<16}│",
        f"│ Active Adaptive Antibodies   │ {sc['active_antibodies']} synthesized signatures in ~/.whole_body_compute/antibodies.json │",
        "└──────────────────────────────┴──────────────────────────────────────────────────────────┘",
    ]

    if show_tools:
        lines.append(format_tools_breakdown_table(sc))

    if show_blocked:
        lines.append(format_top_blocked_table(sc))

    recent = sc.get("recent_events") or []
    if recent:
        lines.append("  Recent Pre/Post Audit Trail:")
        for ev in recent[-limit:]:
            phase = str(ev.get("phase", "")).upper().ljust(4)
            tool = str(ev.get("tool", ""))[:12].ljust(12)
            lat = f"{float(ev.get('latency_ms', 0.0)):5.2f}ms"
            raw_tgt = str(ev.get("target", "")).replace("\n", " ").replace("\r", " ")
            target = Path(raw_tgt).name if "/" in raw_tgt and " " not in raw_tgt else raw_tgt[:38]
            if ev.get("phase") == "pre":
                dec = ev.get("decision", "allow")
                if ev.get("context_gate_triggered"):
                    badge = "🛡️ MAP"
                elif ev.get("bounded_slice_read") and int(ev.get("direct_tokens_saved", 0)) > 0:
                    badge = "⚡ SLICE"
                elif ev.get("auto_healed"):
                    badge = "🩹 HEAL"
                else:
                    badge = "🟢 ALLOW" if dec == "allow" else f"🛑 {dec.upper()}"
                lines.append(f"    [{phase}] {badge:<8} | {tool} | {lat} | {target[:40]}")
            elif ev.get("phase") == "pre_invocation":
                lines.append(f"    [PINV] 💬 NUDGE | {tool} | {lat} | {target[:40]}")
            else:
                chg = "Δfile=True" if ev.get("file_changed") else "Δfile=False"
                lines.append(f"    [{phase}] ✅ VERIF | {tool} | {lat} | {chg} ({target[:32]})")
    return "\n".join(lines)


class OrganImmuneSystem:
    """
    Organ 05: Blood-Brain Barrier (BBB), Adaptive Immune System & Context Shield.
    Provides PreToolUse / PostToolUse security gates, AST-aware file inspection,
    Vascular traffic egress monitoring, shell bloat prevention, and audit telemetry.
    """

    @classmethod
    def inspect_shell_command_bloat(cls, cmd_line: str) -> Dict[str, Any]:
        """
        Context-Shield Pre-Gate: Intercept shell commands that attempt to bypass
        the read_file slice gate with unbounded dumps, directory crawls, or test floods.
        """
        return inspect_shell_command_bloat(cmd_line)

    @classmethod
    def classify_airlock_permeability(
        cls,
        task_query: str = "",
        file_count_est: int = 1,
        retry_strikes: int = 0,
        target_files: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Classify a task into STRICT_AIRLOCK, SELECTIVE_PERFUSION, or FULL_PERFUSION."""
        from whole_body_compute.organs.immune import classify_airlock_permeability

        return classify_airlock_permeability(
            task_query=task_query,
            file_count_est=file_count_est,
            retry_strikes=retry_strikes,
            target_files=target_files,
        )

    @classmethod
    def evaluate_pre_tool(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate pre-tool invocation through BBB gates."""
        from whole_body_compute.organs.immune import run_ide_pre_tool_hook

        return run_ide_pre_tool_hook(payload)

    @classmethod
    def evaluate_post_tool(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate post-tool execution through BBB verification."""
        from whole_body_compute.organs.immune import run_ide_post_tool_hook

        return run_ide_post_tool_hook(payload)

    @classmethod
    def slice_post_tool_output_buffer(
        cls,
        raw_output: Any,
        tool_name: str = "tool_output",
        target_path: Optional[str | Path] = None,
        max_lines: int = 250,
        max_bytes: int = 8000,
        step_idx: int = 0,
        head_lines: int = 30,
        tail_lines: int = 15,
    ) -> Dict[str, Any]:
        """Slice oversized post-tool output buffers prior to LLM context ingestion."""
        return slice_post_tool_output_buffer(
            raw_output=raw_output,
            tool_name=tool_name,
            target_path=target_path,
            max_lines=max_lines,
            max_bytes=max_bytes,
            step_idx=step_idx,
            head_lines=head_lines,
            tail_lines=tail_lines,
        )

    @classmethod
    def get_scorecard(
        cls,
        limit: int = 50,
        window: str = "all",
        show_tools: bool = False,
        show_blocked: bool = False,
    ) -> Dict[str, Any]:
        """Retrieve the empirical Pre/Post Immune Gate & Context-Shield metrics."""
        return get_immune_pre_post_scorecard(
            limit=limit,
            window=window,
            show_tools=show_tools,
            show_blocked=show_blocked,
        )

    @classmethod
    def get_context_shield_summary(
        cls,
        limit: int = 50,
        window: str = "all",
        show_tools: bool = False,
        show_blocked: bool = False,
    ) -> Dict[str, Any]:
        """Retrieve the empirical Context-Shield Scorecard & Bicameral Compute Split."""
        return get_context_shield_summary(
            limit=limit,
            window=window,
            show_tools=show_tools,
            show_blocked=show_blocked,
        )

    @classmethod
    def format_scorecard_table(
        cls,
        limit: int = 10,
        window: str = "all",
        show_tools: bool = False,
        show_blocked: bool = False,
    ) -> str:
        """Format the scorecard table with optional tool and blocked breakdowns."""
        return format_immune_scorecard_table(
            limit=limit,
            window=window,
            show_tools=show_tools,
            show_blocked=show_blocked,
        )
