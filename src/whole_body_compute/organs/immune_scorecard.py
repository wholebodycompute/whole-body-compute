"""
Blood-Brain Barrier (Organ 05): Pre/Post Immune Gate Scorecard, Bicameral Compute Split & OrganImmuneSystem Facade.
"""

from __future__ import annotations

import json
import math
import os
import re
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
        e.get("read_auto_sliced")
        or ev_stage in ("pre_tool_ast_slicer_block", "pre_tool_auto_slice")
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
            e.get("read_auto_sliced")
            or ev_stage in ("pre_tool_ast_slicer_block", "pre_tool_auto_slice")
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
        elif t_name in ("run_command", "bash", "execute_command", "developer__shell"):
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


_SLIM_KEEP_KEYS = (
    "timestamp",
    "phase",
    "conversation_id",
    "step_idx",
    "stage",
    "tool",
    "target",
    "decision",
    "event",
    "gate_stage",
    "latency_ms",
    "fast_path",
    "auto_healed",
    "em_dash_healed",
    "machine_marks_scrubbed",
    "assert_delta",
    "secret_in_output",
    "context_gate_triggered",
    "bounded_slice_read",
    "read_auto_sliced",
    "post_output_sliced",
    "direct_tokens_saved",
    "quadratic_tokens_saved",
    "model_name",
    "model",
)
_MEM_SLIM_CACHE: Dict[str, Any] = {}


def _to_slim_event(d: Dict[str, Any]) -> Dict[str, Any]:
    slim = {
        k: d[k]
        for k in _SLIM_KEEP_KEYS
        if k in d and d[k] not in (False, 0, 0.0, None, "")
    }
    if "timestamp" in d:
        slim["timestamp"] = float(d.get("timestamp") or 0.0)
    if d.get("decision") not in ("allow", None) and d.get("reason"):
        slim["reason"] = d["reason"]
    return slim


def _load_windowed_audit_events(
    log_path: Path, cutoff: Optional[float]
) -> List[Dict[str, Any]]:
    if not log_path.is_file():
        return []
    try:
        file_size = log_path.stat().st_size
    except Exception:
        return []

    if cutoff is None or file_size < 1_000_000:
        out: List[Dict[str, Any]] = []
        try:
            for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    ev = json.loads(line)
                    if cutoff is not None and float(ev.get("timestamp", 0.0)) < cutoff:
                        continue
                    out.append(ev)
                except Exception:
                    continue
        except Exception:
            pass
        return out

    path_key = str(log_path.resolve())
    cache_file = log_path.with_suffix(".slim_cache.json")
    now_ts = time.time()
    target_min_cutoff = min(cutoff, now_ts - 8 * 86400.0)
    max_cache_bytes = 64 * 1024 * 1024

    def _write_slim_cache_json(data: Dict[str, Any]) -> None:
        tmp_cache = cache_file.with_suffix(f".slim_cache.{os.getpid()}.tmp")
        try:
            fd = os.open(str(tmp_cache), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fp:
                json.dump(data, fp, separators=(",", ":"))
            os.replace(str(tmp_cache), str(cache_file))
        except Exception:
            try:
                tmp_cache.unlink(missing_ok=True)
            except Exception:
                pass

    try:
        with open(log_path, "rb") as f:
            head_sig_hex = f.read(256).hex()

            mem = _MEM_SLIM_CACHE.get(path_key)
            if (
                mem
                and mem.get("head_sig") == head_sig_hex
                and file_size >= int(mem.get("size", 0))
                and cutoff >= float(mem.get("min_cutoff", now_ts)) - 60.0
            ):
                cached_size = int(mem["size"])
                events: List[Dict[str, Any]] = mem["events"]
                if file_size > cached_size:
                    f.seek(cached_size)
                    for raw_ln in f.read().splitlines():
                        if not raw_ln.strip():
                            continue
                        try:
                            events.append(_to_slim_event(json.loads(raw_ln)))
                        except Exception:
                            continue
                    mem["size"] = file_size
                return [e for e in events if float(e.get("timestamp", 0.0)) >= cutoff]

            if cache_file.is_file() and not cache_file.is_symlink():
                try:
                    st = os.lstat(cache_file)
                    owner_ok = not hasattr(os, "getuid") or st.st_uid == os.getuid()
                    if owner_ok and 0 < st.st_size <= max_cache_bytes:
                        disk = json.loads(cache_file.read_text(encoding="utf-8"))
                    else:
                        disk = None
                    if (
                        isinstance(disk, dict)
                        and disk.get("head_sig") == head_sig_hex
                        and file_size >= int(disk.get("size", 0))
                        and isinstance(disk.get("events"), list)
                        and cutoff >= float(disk.get("min_cutoff", now_ts)) - 60.0
                    ):
                        cached_size = int(disk["size"])
                        events = disk["events"]
                        delta_bytes = file_size - cached_size
                        if delta_bytes > 0:
                            f.seek(cached_size)
                            for raw_ln in f.read().splitlines():
                                if not raw_ln.strip():
                                    continue
                                try:
                                    events.append(_to_slim_event(json.loads(raw_ln)))
                                except Exception:
                                    continue
                            disk["size"] = file_size
                            if delta_bytes > 65536:
                                prune_ts = now_ts - 8 * 86400.0
                                events = [
                                    e for e in events if float(e.get("timestamp", 0.0)) >= prune_ts
                                ]
                                disk["events"] = events
                                disk["min_cutoff"] = prune_ts
                                _write_slim_cache_json(disk)
                        _MEM_SLIM_CACHE[path_key] = disk
                        return [e for e in events if float(e.get("timestamp", 0.0)) >= cutoff]
                except Exception:
                    pass

            # Cold cache: binary-seek to target_min_cutoff
            lo, hi = 0, file_size
            while hi - lo > 65536:
                mid = (lo + hi) // 2
                f.seek(mid)
                f.readline()
                probe = f.readline()
                if not probe:
                    hi = mid
                    continue
                m = re.search(rb'"timestamp":\s*([0-9.]+)', probe)
                if m and float(m.group(1)) < target_min_cutoff:
                    lo = mid
                else:
                    hi = mid
            f.seek(lo)
            if lo > 0:
                f.readline()
            tail_bytes = f.read()

        events = []
        for raw_ln in tail_bytes.splitlines():
            if not raw_ln.strip():
                continue
            try:
                ev = json.loads(raw_ln)
                if float(ev.get("timestamp", 0.0)) >= target_min_cutoff:
                    events.append(_to_slim_event(ev))
            except Exception:
                continue

        payload = {
            "head_sig": head_sig_hex,
            "size": file_size,
            "min_cutoff": target_min_cutoff,
            "events": events,
        }
        _MEM_SLIM_CACHE[path_key] = payload
        _write_slim_cache_json(payload)
        return [e for e in events if float(e.get("timestamp", 0.0)) >= cutoff]
    except Exception:
        return []


def get_immune_pre_post_scorecard(
    limit: int = 50,
    window: str = "all",
    show_tools: bool = False,
    show_blocked: bool = False,
    conv_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute empirical Pre/Post Immune Gate & Context-Shield metrics from `~/.whole_body_compute/immune_audit.jsonl`.
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

    for ev in _load_windowed_audit_events(log_path, cutoff):
        recent_events.append(ev)
        ph = ev.get("phase")
        if ph == "pre":
            pre_events.append(ev)
            if cid_prefix and str(ev.get("conversation_id") or "").startswith(cid_prefix):
                session_pre_events.append(ev)
        elif ph == "post":
            post_events.append(ev)
        elif ph == "pre_invocation":
            pre_inv_events.append(ev)

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
    active_ast_slices = sum(1 for e in pre_events if e.get("read_auto_sliced"))
    voluntary_bounded_reads = sum(
        1 for e in pre_events if e.get("bounded_slice_read") and not e.get("read_auto_sliced")
    )
    bounded_slice_reads = active_ast_slices
    post_sliced_count = sum(1 for e in post_events if e.get("post_output_sliced"))

    from whole_body_compute.shield import (
        _lookup_model_rate,
        _normalize_target_label,
        detect_active_model_and_rate,
        get_harness_file_token_ceiling,
    )

    _, active_rate_per_m = detect_active_model_and_rate()

    def _tool_read_ceiling(tool_raw: str) -> Optional[int]:
        tl = str(tool_raw or "").strip().lower()
        if tl in ("read_file", "view_file"):
            return get_harness_file_token_ceiling(bot_name="antigravity")
        if tl in ("read", "notebookread"):
            return get_harness_file_token_ceiling(bot_name="claude-code")
        if tl in ("beforereadfile", "pre_read_code"):
            return get_harness_file_token_ceiling(bot_name="cursor")
        if tl == "developer__text_editor":
            return get_harness_file_token_ceiling(bot_name="codex")
        return None

    def _coalesce_session_cumulative(ev_rows: List[Tuple[float, int, float]]) -> Tuple[int, float]:
        if not ev_rows:
            return 0, 0.0
        ordered = sorted(ev_rows, key=lambda item: item[0])
        buckets: List[Tuple[float, int, float]] = []
        for ts_v, s_v, r_v in ordered:
            if not buckets or (ts_v - buckets[-1][0]) >= 25.0:
                buckets.append((ts_v, s_v, r_v))
            else:
                b_ts, b_sav, b_r = buckets[-1]
                buckets[-1] = (b_ts, b_sav + s_v, b_r)
        n_t = len(buckets)
        cum_tok = sum((n_t - idx) * b_sav for idx, (_, b_sav, _) in enumerate(buckets))
        direct_t = sum(b_sav for _, b_sav, _ in buckets)
        avg_r = sum(r_v for _, _, r_v in ordered) / max(1, len(ordered))
        reingest_usd = (max(0, cum_tok - direct_t) / 1_000_000.0) * (avg_r * 0.10)
        return cum_tok, reingest_usd

    # Chronological, ceiling-capped, per-(session, canonical_target) deduplicated savings
    # Exclude voluntary bounded_slice_read (without read_auto_sliced / active interception).
    sorted_pre = sorted(pre_events, key=lambda e: float(e.get("timestamp") or 0.0))
    dedup_credits: Dict[Tuple[str, str], int] = {}
    bare_seen_credits: Dict[Tuple[str, str], int] = {}
    catchall_last_ts: Dict[str, float] = {}
    catchall_epoch: Dict[str, int] = {}
    session_ev_groups: Dict[Tuple[str, int], List[Tuple[float, int, float]]] = {}
    tool_session_ev_groups: Dict[Tuple[str, str, int], List[Tuple[float, int, float]]] = {}

    direct_tokens_saved = 0
    total_direct_usd = 0.0
    tools_stats: Dict[str, Dict[str, Any]] = {}

    for e in sorted_pre:
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
                "direct_usd": 0.0,
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
        if e.get("read_auto_sliced"):
            st["slices"] += 1

        is_active = bool(
            e.get("read_auto_sliced")
            or e.get("context_gate_triggered")
            or e.get("auto_healed")
            or dec in ("ask", "force_ask", "deny")
        )
        if not is_active:
            continue

        d_sav = max(0, int(e.get("direct_tokens_saved", 0) or 0))
        if d_sav <= 0:
            continue

        read_ceil = _tool_read_ceiling(t_name)
        if read_ceil is not None:
            d_sav = min(read_ceil, d_sav)

        ts_v = float(e.get("timestamp") or 0.0)
        cid_raw = str(e.get("conversation_id") or "").strip() or "global"
        is_catchall = cid_raw in ("global", "default-session", "local", "cli-standalone", "")
        cur_ep = 0
        if is_catchall:
            prev_ts = catchall_last_ts.get(cid_raw)
            cur_ep = catchall_epoch.get(cid_raw, 0)
            if prev_ts is not None and (ts_v - prev_ts) > 7200.0:
                cur_ep += 1
                catchall_epoch[cid_raw] = cur_ep
            catchall_last_ts[cid_raw] = ts_v
        eff_sess = f"{cid_raw}#{cur_ep}" if is_catchall and cur_ep > 0 else cid_raw

        raw_tgt = str(e.get("target") or "").strip()
        if not os.environ.get("VOICEFI_IMMUNE_AUDIT_PATH") and raw_tgt.replace("\\", "/").startswith(
            ("/tmp/", "/private/tmp/", "/var/folders/", "/private/var/folders/")
        ):
            continue
        norm_tgt = _normalize_target_label(raw_tgt) if raw_tgt else ""
        if norm_tgt and norm_tgt != "unknown":
            dk = (eff_sess, norm_tgt)
            b_only = Path(norm_tgt).name if "/" in norm_tgt else norm_tgt
            bk = (eff_sess, b_only)
            if "/" not in norm_tgt:
                prev_c = max(dedup_credits.get(dk, 0), bare_seen_credits.get(bk, 0))
            else:
                prev_c = max(dedup_credits.get(dk, 0), dedup_credits.get(bk, 0))
            eff_saved = max(0, max(prev_c, d_sav) - prev_c)
            new_hwm = prev_c + eff_saved
            dedup_credits[dk] = new_hwm
            bare_seen_credits[bk] = max(bare_seen_credits.get(bk, 0), new_hwm)
            if "/" in norm_tgt and bk in dedup_credits:
                dedup_credits[bk] = max(dedup_credits[bk], new_hwm)
        else:
            eff_saved = d_sav

        if eff_saved <= 0:
            continue

        ev_model = str(e.get("model_name") or e.get("model") or "").strip()
        row_rate = _lookup_model_rate(ev_model) if ev_model else active_rate_per_m
        row_direct_usd = (eff_saved / 1_000_000.0) * row_rate

        direct_tokens_saved += eff_saved
        total_direct_usd += row_direct_usd
        st["direct_tokens_saved"] += eff_saved
        st["direct_usd"] += row_direct_usd
        session_ev_groups.setdefault((cid_raw, cur_ep), []).append((ts_v, eff_saved, row_rate))
        tool_session_ev_groups.setdefault((t_name, cid_raw, cur_ep), []).append((ts_v, eff_saved, row_rate))

    for e in sorted(post_events, key=lambda ev: float(ev.get("timestamp") or 0.0)):
        d_post = max(0, int(e.get("direct_tokens_saved", 0) or 0))
        if d_post <= 0 and not e.get("post_output_sliced"):
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
                "direct_usd": 0.0,
                "estimated_usd_saved": 0.0,
                "allowed_pct": 100.0,
                "intercepted_pct": 0.0,
            }
        st = tools_stats[t_name]
        if e.get("post_output_sliced"):
            st["slices"] += 1
        if d_post <= 0:
            continue
        ts_v = float(e.get("timestamp") or 0.0)
        cid_raw = str(e.get("conversation_id") or "").strip() or "global"
        is_catchall_post = cid_raw in ("global", "default-session", "local", "cli-standalone", "")
        cur_ep_post = 0
        if is_catchall_post:
            prev_ts = catchall_last_ts.get(cid_raw)
            cur_ep_post = catchall_epoch.get(cid_raw, 0)
            if prev_ts is not None and (ts_v - prev_ts) > 7200.0:
                cur_ep_post += 1
                catchall_epoch[cid_raw] = cur_ep_post
            catchall_last_ts[cid_raw] = ts_v
        ev_model = str(e.get("model_name") or e.get("model") or "").strip()
        row_rate = _lookup_model_rate(ev_model) if ev_model else active_rate_per_m
        row_direct_usd = (d_post / 1_000_000.0) * row_rate
        direct_tokens_saved += d_post
        total_direct_usd += row_direct_usd
        st["direct_tokens_saved"] += d_post
        st["direct_usd"] += row_direct_usd
        session_ev_groups.setdefault((cid_raw, cur_ep_post), []).append((ts_v, d_post, row_rate))
        tool_session_ev_groups.setdefault((t_name, cid_raw, cur_ep_post), []).append((ts_v, d_post, row_rate))

    quadratic_tokens_saved = 0
    total_reingest_usd = 0.0
    for ev_rows in session_ev_groups.values():
        c_tok, r_usd = _coalesce_session_cumulative(ev_rows)
        quadratic_tokens_saved += c_tok
        total_reingest_usd += r_usd

    tool_cum_map: Dict[str, Tuple[int, float]] = {}
    for (t_name, _cid, _ep), ev_rows in tool_session_ev_groups.items():
        c_tok, r_usd = _coalesce_session_cumulative(ev_rows)
        prev_ct, prev_ru = tool_cum_map.get(t_name, (0, 0.0))
        tool_cum_map[t_name] = (prev_ct + c_tok, prev_ru + r_usd)

    dedup_direct_tokens_saved = direct_tokens_saved
    direct_usd_saved = round(total_direct_usd, 2)
    dedup_direct_usd_saved = direct_usd_saved
    cached_reingestion_usd = round(total_reingest_usd, 2)
    estimated_usd_saved = round(total_direct_usd + total_reingest_usd, 2)
    pre_invocation_nudges = len(pre_inv_events)

    cutoff_24h = time.time() - 86400.0
    pre_24h = [e for e in pre_events if float(e.get("timestamp", 0.0)) >= cutoff_24h]
    intercepted_24h = sum(
        1 for e in pre_24h if e.get("decision") in ("ask", "force_ask", "deny")
    )

    for t_name, st in tools_stats.items():
        tot = st["total"]
        if tot > 0:
            st["allowed_pct"] = round((st["allowed"] / tot) * 100.0, 1)
            st["intercepted_pct"] = round((st["intercepted"] / tot) * 100.0, 1)
        t_cum, t_reingest_usd = tool_cum_map.get(t_name, (int(st["direct_tokens_saved"]), 0.0))
        st["quadratic_tokens_saved"] = t_cum
        st["estimated_usd_saved"] = round(float(st.pop("direct_usd", 0.0)) + t_reingest_usd, 2)

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
        "active_ast_slices": active_ast_slices,
        "voluntary_bounded_reads": voluntary_bounded_reads,
        "bounded_slice_reads": bounded_slice_reads,
        "post_sliced_count": post_sliced_count,
        "direct_tokens_saved": direct_tokens_saved,
        "quadratic_tokens_saved": quadratic_tokens_saved,
        "dedup_direct_tokens_saved": dedup_direct_tokens_saved,
        "dedup_direct_usd_saved": dedup_direct_usd_saved,
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


def _format_compact_target(raw_tgt: str, max_len: int = 28) -> str:
    """Format file paths (including workspace paths with spaces) or CLI commands cleanly without mid-directory truncation."""
    s = str(raw_tgt or "").replace("\n", " ").replace("\r", " ").strip()
    if not s:
        return "unknown"
    is_path_like = (
        s.startswith(("/", "~/"))
        and not s.startswith(("/bin/", "/usr/bin/"))
        and " --" not in s
        and " | " not in s
        and " && " not in s
    ) or (
        "/" in s
        and " " not in s
        and not s.startswith(("http://", "https://"))
    )
    if is_path_like:
        p = Path(s)
        name = p.name or s
        parent = p.parent.name if p.parent else ""
        if parent and parent not in ("", "/", ".") and len(f"{parent}/{name}") <= max_len:
            return f"{parent}/{name}"
        return name[:max_len]
    return s[:max_len]


def _normalize_legacy_reason(reason: str) -> str:
    """Normalize legacy historical organ audit log prefixes to 'BBB'."""
    s = str(reason or "Blocked by Blood-Brain Barrier").replace("\n", " ").strip()
    s = re.sub(r"\bOrgan\s+(?:10|11)\s*:?\s*", "BBB ", s)
    s = s.replace("Blood-Brain Barrier ", "BBB ")
    return s


def format_top_blocked_table(sc: Dict[str, Any]) -> str:
    """Format top blocked files/commands table."""
    top_blkd = sc.get("top_blocked") or []
    win = sc.get("window", "all")
    win_str = f"Window: {win}" if win != "all" else "All Time"
    lines = [
        f"\n🛑 Top Blocked Files & Commands ({win_str}):",
        "┌──────┬─────────────┬─────────────────────┬──────────────────────────────┬──────────────────────────────────────────────────┐",
        "│ Blkd │ Tool        │ Category            │ Target                       │ Primary Reason                                   │",
        "├──────┼─────────────┼─────────────────────┼──────────────────────────────┼──────────────────────────────────────────────────┤",
    ]
    if not top_blkd:
        lines.append("│ (No actions intercepted or blocked in this window)                                                             │")
    else:
        for entry in top_blkd[:10]:
            cnt = str(entry.get("count", 1)).rjust(4)
            tool = str(entry.get("tool", ""))[:11].ljust(11)
            cat = str(entry.get("category", ""))[:19].ljust(19)
            raw_tgt = str(entry.get("target", ""))
            tgt_disp = _format_compact_target(raw_tgt, max_len=28).ljust(28)
            reasons = entry.get("reasons") or ["Blocked by Blood-Brain Barrier"]
            primary_reason = _normalize_legacy_reason(str(reasons[0]))[:48].ljust(48)
            lines.append(f"│ {cnt} │ {tool} │ {cat} │ {tgt_disp} │ {primary_reason} │")
    lines.append("└──────┴─────────────┴─────────────────────┴──────────────────────────────┴──────────────────────────────────────────────────┘")
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
    total_slices = int(sc.get("bounded_slice_reads", 0)) + int(sc.get("post_sliced_count", 0))
    lines = [
        f"🛡️  The Blood-Brain Barrier (BBB) - Pre/Post Context Shield, AST Gate & Vascular Scorecard ({win_label})",
        "┌──────────────────────────────┬──────────────────────────────────────────────────────────┐",
        f"│ Pre-Tool Checks / Post-Audits│ {sc['total_pre_checks']} pre-checks ({sc['fast_path_count']} fast-sh / {sc['deep_ast_count']} deep-AST) • {sc['total_post_verifications']} post │",
        f"│ Avg Overhead (Pre / Post)    │ {sc['avg_pre_latency_ms']:.2f}ms pre-gate  •  {sc['avg_post_latency_ms']:.2f}ms post-tactile{'':<19}│",
        f"│ Gate Decisions (All / 24h)   │ 🟢 {sc['allowed_count']} allowed  •  🩹 {sc['auto_healed_count']} healed  •  🛑 {sc['intercepted_count']} intercepted ({sc['intercepted_24h']} 24h)│",
        f"│ Context Shield (AST / Slices)│ 🛡️ {sc['context_gate_intercepts']} full-reads blocked • ⚡ {total_slices} slices • 💬 {sc['pre_invocation_nudges']} nudges│",
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
            raw_tgt = str(ev.get("target", ""))
            target = _format_compact_target(raw_tgt, max_len=42)
            if ev.get("phase") == "pre":
                dec = ev.get("decision", "allow")
                if ev.get("context_gate_triggered"):
                    badge = "🛡️ MAP"
                elif (ev.get("bounded_slice_read") or ev.get("read_auto_sliced")) and int(ev.get("direct_tokens_saved", 0)) > 0:
                    badge = "⚡ SLICE"
                elif ev.get("auto_healed"):
                    badge = "🩹 HEAL"
                else:
                    badge = "🟢 ALLOW" if dec == "allow" else f"🛑 {dec.upper()}"
                lines.append(f"    [{phase}] {badge:<8} | {tool} | {lat} | {target[:42]}")
            elif ev.get("phase") == "pre_invocation":
                lines.append(f"    [PINV] 💬 NUDGE | {tool} | {lat} | {target[:42]}")
            else:
                chg = "Δfile=True" if ev.get("file_changed") else "Δfile=False"
                if ev.get("post_output_sliced"):
                    sav = int(ev.get("direct_tokens_saved", 0) or 0)
                    lines.append(f"    [{phase}] ⚡ SLICE | {tool} | {lat} | -{sav:,} tok ({target[:32]})")
                else:
                    lines.append(f"    [{phase}] ✅ VERIF | {tool} | {lat} | {chg} ({target[:36]})")
    return "\n".join(lines)


class OrganImmuneSystem:
    """
    Organ 05: The Blood-Brain Barrier (BBB) - Context Shield, AST Gate & Vascular Defense.
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
