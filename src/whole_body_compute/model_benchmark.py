"""
Empirical Multi-Model Leaderboard Telemetry (`body scorecard --by-model` / `body context --by-model`).

Benchmarks all Antigravity models head-to-head across `transcript.jsonl` logs, including:
- Exact `<USER_SETTINGS_CHANGE>` model parsing (preserving decimal names like `Gemini 3.8 Flash (High)`).
- Selective line-index fallback to `transcript_full.jsonl` when `USER_INPUT` lines have `truncated_fields`.
- `invoke_subagent` child `conversationId` lineage attribution into a separate `subagent_cohorts` table
  (or merged when `include_subagents=True`).
- Pure LLM tool-dispatch latency (`GENERIC.created_at - PLANNER_RESPONSE.created_at` on tool turns <= 350 output tok),
  end-to-end `est_task_llm_s` (`med_tool_dispatch_s * med_turns_per_req`), `med_peak_csi`, and Daily Driver selection.
"""

from __future__ import annotations

import datetime
import glob
import json
import os
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from whole_body_compute.finops_audit import DEFAULT_BRAIN_DIR

MODEL_CHANGE_RE = re.compile(
    r"changed setting `Model Selection` from (.+?) to (.+?)\.\s+No need"
)
SUBAGENT_ID_RE = re.compile(
    r'"conversationId"\s*:\s*"([0-9a-zA-Z_-]{8,64})"|conversationId:\s*([0-9a-zA-Z_-]{8,64})'
)

UNTAGGED_MAIN_LABEL = "Pre-Nov-4 / Untagged Main Thread"


def resolve_brain_dir(brain_dir: Optional[Path] = None) -> Path:
    """Resolve Antigravity brain directory matching CognitiveThermalSentry."""
    env_dir = os.environ.get("AGY_BRAIN_DIR") or os.environ.get("ANTIGRAVITY_BRAIN_DIR")
    return Path(brain_dir or env_dir or DEFAULT_BRAIN_DIR).expanduser()


def parse_ts(ts: Optional[str]) -> Optional[float]:
    """Parse ISO-8601 timestamp string to epoch seconds."""
    if not ts:
        return None
    try:
        return datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def window_to_seconds(window: Optional[str], default_sec: Optional[float] = 30.0 * 86400.0) -> Optional[float]:
    """
    Convert window string ('1h', '24h', '7d', '30d', 'all') into cutoff duration seconds.
    Returns None when window is 'all' (no time filtering).
    """
    w = (window or "30d").strip().lower()
    if w in ("all", "0", "inf", "forever", "*"):
        return None
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([smhdw]?)$", w)
    if not m:
        return default_sec
    val = float(m.group(1))
    unit = m.group(2) or "h"
    if unit == "s":
        return val
    if unit == "m":
        return val * 60.0
    if unit == "h":
        return val * 3600.0
    if unit == "d":
        return val * 86400.0
    if unit == "w":
        return val * 7.0 * 86400.0
    return default_sec


def read_specific_lines(full_path: Path, target_indices: Set[int]) -> Dict[int, Dict[str, Any]]:
    """Read only specific 0-based line indices from `transcript_full.jsonl`."""
    found: Dict[int, Dict[str, Any]] = {}
    if not target_indices or not full_path.is_file():
        return found
    try:
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            for line_no, raw_line in enumerate(f):
                if line_no in target_indices:
                    raw_s = raw_line.strip()
                    if raw_s:
                        try:
                            obj = json.loads(raw_s)
                            if isinstance(obj, dict):
                                found[line_no] = obj
                        except Exception:
                            pass
                    if len(found) == len(target_indices):
                        break
    except Exception:
        pass
    return found


def _new_bucket() -> Dict[str, Any]:
    return {
        "threads": set(),
        "planner_turns": 0,
        "tool_turns": 0,
        "total_tool_calls": 0,
        "tool_errors": 0,
        "truncations": 0,
        "llm_latencies_s": [],
        "pure_tool_latencies_s": [],
        "output_toks_per_sec": [],
        "input_tokens": [],
        "cache_read_tokens": [],
        "output_tokens": [],
        "thinking_chars": [],
        "thinking_turns": 0,
        "turns_per_user_msg": [],
        "thread_csis": [],
        "first_seen": None,
        "last_seen": None,
        "is_subagent": False,
        "parent_model": None,
    }


def _summarize_bucket(model: str, m: Dict[str, Any]) -> Dict[str, Any]:
    lats = sorted(m["llm_latencies_s"])
    pure_lats = sorted(m["pure_tool_latencies_s"])
    in_sum = sum(m["input_tokens"])
    cache_sum = sum(m["cache_read_tokens"])
    out_sum = sum(m["output_tokens"])
    total_prompt = in_sum + cache_sum

    med_llm_lat_s = round(statistics.median(lats), 2) if lats else None
    med_tool_dispatch_s = round(statistics.median(pure_lats), 2) if pure_lats else None
    p90_llm_lat_s = round(lats[int(0.9 * (len(lats) - 1))], 2) if len(lats) >= 5 else None
    med_tok_per_s = (
        round(statistics.median(m["output_toks_per_sec"]), 1)
        if m["output_toks_per_sec"]
        else None
    )
    med_out_tok = (
        int(round(statistics.median(m["output_tokens"]), 0))
        if m["output_tokens"]
        else None
    )
    cache_hit_pct = round(100.0 * cache_sum / total_prompt, 1) if total_prompt > 0 else 0.0
    thinking_pct = (
        round(100.0 * m["thinking_turns"] / m["planner_turns"], 1)
        if m["planner_turns"] > 0
        else 0.0
    )
    mean_thinking_chars = (
        int(round(statistics.mean(m["thinking_chars"]), 0))
        if m["thinking_chars"]
        else 0
    )
    tools_per_turn = (
        round(m["total_tool_calls"] / m["tool_turns"], 2)
        if m["tool_turns"] > 0
        else 0.0
    )
    tool_err_pct = (
        round(100.0 * m["tool_errors"] / m["total_tool_calls"], 2)
        if m["total_tool_calls"] > 0
        else 0.0
    )
    med_turns_per_req = (
        round(statistics.median(m["turns_per_user_msg"]), 1)
        if m["turns_per_user_msg"]
        else None
    )
    est_task_llm_s = (
        round(med_tool_dispatch_s * med_turns_per_req, 2)
        if (med_tool_dispatch_s is not None and med_turns_per_req is not None)
        else None
    )
    med_peak_csi = (
        round(statistics.median(m["thread_csis"]), 2)
        if m["thread_csis"]
        else None
    )

    first_s = (m["first_seen"] or "")[:10]
    last_s = (m["last_seen"] or "")[:10]
    date_span = f"{first_s} -> {last_s}" if (first_s or last_s) else "—"

    return {
        "model": model,
        "is_subagent_cohort": bool(m.get("is_subagent", False)),
        "parent_model": m.get("parent_model"),
        "threads": len(m["threads"]),
        "user_requests": len(m["turns_per_user_msg"]),
        "planner_turns": m["planner_turns"],
        "tool_turns": m["tool_turns"],
        "total_tool_calls": m["total_tool_calls"],
        "tool_errors": m["tool_errors"],
        "truncations": m["truncations"],
        "date_span": date_span,
        "med_tool_dispatch_s": med_tool_dispatch_s,
        "med_llm_lat_s": med_llm_lat_s,
        "p90_llm_lat_s": p90_llm_lat_s,
        "med_tok_per_s": med_tok_per_s,
        "med_out_tok": med_out_tok,
        "thinking_pct": thinking_pct,
        "mean_thinking_chars": mean_thinking_chars,
        "tools_per_turn": tools_per_turn,
        "tool_err_pct": tool_err_pct,
        "med_turns_per_req": med_turns_per_req,
        "est_task_llm_s": est_task_llm_s,
        "spawned_subagents": 0,
        "spawned_subagent_turns": 0,
        "subagent_turn_ratio": 0.0,
        "subagent_tax_s": 0.0,
        "fleet_task_llm_s": est_task_llm_s,
        "med_peak_csi": med_peak_csi,
        "cache_hit_pct": cache_hit_pct,
        "total_out_m_tok": round(out_sum / 1_000_000.0, 3),
    }


def _apply_subagent_laundering_tax(
    main_rows: List[Dict[str, Any]],
    sub_summaries: List[Dict[str, Any]],
) -> None:
    """
    Attribute hidden `invoke_subagent` child turns back onto the spawning parent model
    so models cannot artificially lower `est_task_llm_s` via subagent laundering.
    """
    by_parent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in sub_summaries:
        p_raw = s.get("parent_model")
        if not p_raw:
            continue
        p_key = UNTAGGED_MAIN_LABEL if p_raw == "Legacy Main" else str(p_raw)
        by_parent[p_key].append(s)

    for r in main_rows:
        if r.get("is_subagent_cohort"):
            continue
        cohorts = by_parent.get(str(r["model"])) or []
        if not cohorts:
            continue

        spawned_threads = sum(int(c.get("threads") or 0) for c in cohorts)
        spawned_turns = sum(int(c.get("planner_turns") or 0) for c in cohorts)
        main_turns = int(r.get("planner_turns") or 0)
        user_reqs = int(r.get("user_requests") or 0)

        r["spawned_subagents"] = spawned_threads
        r["spawned_subagent_turns"] = spawned_turns
        r["subagent_turn_ratio"] = (
            round(spawned_turns / main_turns, 2) if main_turns > 0 else 0.0
        )

        if user_reqs > 0 and spawned_turns > 0:
            weighted_sec = 0.0
            for c in cohorts:
                c_turns = int(c.get("planner_turns") or 0)
                c_disp = c.get("med_tool_dispatch_s")
                if c_disp is None:
                    c_disp = c.get("med_llm_lat_s")
                if c_disp is None:
                    c_disp = 4.0
                weighted_sec += c_turns * float(c_disp)
            # Exact per-request subagent LLM cost (total subagent dispatch sec / parent user requests)
            tax_s = round(weighted_sec / user_reqs, 2)
        else:
            tax_s = 0.0

        r["subagent_tax_s"] = tax_s
        if r.get("est_task_llm_s") is not None:
            r["fleet_task_llm_s"] = round(float(r["est_task_llm_s"]) + tax_s, 2)


def select_recommended_daily_driver(models: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Crown Recommended Daily Driver:
    Lowest subagent-adjusted `fleet_task_llm_s` among non-legacy/non-subagent models with:
      - `planner_turns >= 10`
      - `med_peak_csi <= 1.0`
      - `fleet_task_llm_s is not None`
    """
    candidates: List[Dict[str, Any]] = []
    for row in models:
        if row.get("is_subagent_cohort"):
            continue
        name = str(row.get("model") or "")
        if name == UNTAGGED_MAIN_LABEL or name.startswith("Subagent"):
            continue
        if int(row.get("planner_turns") or 0) < 10:
            continue
        eff = row.get("fleet_task_llm_s")
        if eff is None:
            eff = row.get("est_task_llm_s")
        if eff is None:
            continue
        csi = row.get("med_peak_csi")
        if csi is not None and float(csi) > 1.0:
            continue
        candidates.append(row)

    if not candidates:
        return None

    winner = min(
        candidates,
        key=lambda r: (
            float(
                r["fleet_task_llm_s"]
                if r.get("fleet_task_llm_s") is not None
                else r["est_task_llm_s"]
            ),
            float(r.get("tool_err_pct") or 0.0),
            -int(r.get("planner_turns") or 0),
        ),
    )
    fleet_s = (
        winner["fleet_task_llm_s"]
        if winner.get("fleet_task_llm_s") is not None
        else winner["est_task_llm_s"]
    )
    tax_s = float(winner.get("subagent_tax_s") or 0.0)
    return {
        "model": winner["model"],
        "fleet_task_llm_s": fleet_s,
        "est_task_llm_s": winner["est_task_llm_s"],
        "subagent_tax_s": tax_s,
        "subagent_turn_ratio": float(winner.get("subagent_turn_ratio") or 0.0),
        "med_tool_dispatch_s": winner["med_tool_dispatch_s"],
        "med_turns_per_req": winner["med_turns_per_req"],
        "med_peak_csi": winner["med_peak_csi"],
        "tool_err_pct": winner["tool_err_pct"],
        "planner_turns": winner["planner_turns"],
        "threads": winner["threads"],
        "reason": (
            f"Lowest fleet-adjusted task LLM time ({fleet_s:.1f}s = "
            f"{winner['est_task_llm_s']:.1f}s main + {tax_s:.1f}s subagent tax; "
            f"{winner['med_tool_dispatch_s']:.1f}s dispatch x {winner['med_turns_per_req']:.1f} turns/req) "
            f"with healthy context saturation (CSI {winner['med_peak_csi'] if winner['med_peak_csi'] is not None else 'N/A'} <= 1.00)."
        ),
    }


def build_model_benchmark_report(
    brain_dir: Optional[Path] = None,
    window: str = "30d",
    include_subagents: bool = False,
    now_ts: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Scan Antigravity `transcript.jsonl` logs and compute empirical multi-model KPIs.
    """
    t_start = time.perf_counter()
    resolved_brain = resolve_brain_dir(brain_dir)
    win_norm = (window or "30d").strip().lower()
    win_sec = window_to_seconds(win_norm)

    if not resolved_brain.is_dir():
        return {
            "schema_version": "1.0.0",
            "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "brain_dir": str(resolved_brain),
            "window": win_norm,
            "include_subagents": include_subagents,
            "total_transcripts_scanned": 0,
            "total_planner_turns": 0,
            "main_thread_turns": 0,
            "subagent_turns": 0,
            "recommended_daily_driver": None,
            "models": [],
            "subagent_cohorts": [],
            "elapsed_ms": round((time.perf_counter() - t_start) * 1000.0, 2),
        }

    pattern = str(resolved_brain / "*" / ".system_generated" / "logs" / "transcript.jsonl")
    paths = sorted(glob.glob(pattern))

    # First pass: load transcripts and map subagent child -> (parent_conv_id, active_parent_model, label)
    loaded: Dict[str, List[Dict[str, Any]]] = {}
    file_mtimes: Dict[str, float] = {}
    subagent_parent: Dict[str, Tuple[str, Optional[str], str]] = {}
    thread_explicit_model: Dict[str, str] = {}

    for path_str in paths:
        p = Path(path_str)
        conv_id = p.parents[2].name
        try:
            file_mtimes[conv_id] = p.stat().st_mtime
        except Exception:
            file_mtimes[conv_id] = 0.0

        full_path = p.with_name("transcript_full.jsonl")
        steps: List[Dict[str, Any]] = []
        trunc_user_lines: Set[int] = set()

        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for raw_line in f:
                    s = raw_line.strip()
                    if not s:
                        steps.append({})
                        continue
                    try:
                        obj = json.loads(s)
                        if not isinstance(obj, dict):
                            steps.append({})
                            continue
                        line_idx = len(steps)
                        steps.append(obj)
                        if obj.get("type") == "USER_INPUT" and obj.get("truncated_fields"):
                            trunc_user_lines.add(line_idx)
                    except Exception:
                        steps.append({})
        except Exception:
            continue

        if trunc_user_lines:
            for line_no, full_obj in read_specific_lines(full_path, trunc_user_lines).items():
                if 0 <= line_no < len(steps):
                    steps[line_no] = full_obj

        loaded[conv_id] = steps
        active_m: Optional[str] = None
        for idx, step in enumerate(steps):
            if not step:
                continue
            stype = step.get("type")
            if stype == "USER_INPUT":
                matches = MODEL_CHANGE_RE.findall(str(step.get("content") or ""))
                if matches:
                    active_m = matches[-1][1].strip()
                    if conv_id not in thread_explicit_model:
                        thread_explicit_model[conv_id] = active_m
            elif stype == "PLANNER_RESPONSE":
                for tc in step.get("tool_calls") or []:
                    if isinstance(tc, dict) and tc.get("name") == "invoke_subagent":
                        args_d = tc.get("args") if isinstance(tc.get("args"), dict) else {}
                        sub_m = str(args_d.get("Model", "inherit")).strip('"')
                        j = idx + 1
                        while j < len(steps) and steps[j].get("type") == "GENERIC":
                            txt = str(steps[j].get("content") or "")
                            for m_id in SUBAGENT_ID_RE.findall(txt):
                                cid = m_id[0] or m_id[1]
                                if cid:
                                    subagent_parent[cid] = (
                                        conv_id,
                                        active_m,
                                        f"{active_m or 'Parent'} -> subagent({sub_m})",
                                    )
                            j += 1

    if now_ts is None:
        now_ts = time.time()

    cutoff_ts: Optional[float] = (now_ts - win_sec) if win_sec is not None else None

    main_stats: Dict[str, Dict[str, Any]] = defaultdict(_new_bucket)
    sub_stats: Dict[str, Dict[str, Any]] = defaultdict(_new_bucket)
    scanned_in_window = 0

    for conv_id, steps in loaded.items():
        if cutoff_ts is not None:
            latest_step_ts: Optional[float] = None
            for s in reversed(steps):
                if s and s.get("created_at"):
                    ts_val = parse_ts(s.get("created_at"))
                    if ts_val is not None:
                        latest_step_ts = ts_val
                        break
            ref_ts = latest_step_ts if latest_step_ts is not None else file_mtimes.get(conv_id, 0.0)
            if ref_ts < cutoff_ts:
                continue

        is_subagent_thread = (conv_id in subagent_parent) and (conv_id not in thread_explicit_model)
        parent_model_name: Optional[str] = None
        if is_subagent_thread:
            p_cid, p_active_m, _ = subagent_parent[conv_id]
            parent_model_name = p_active_m or thread_explicit_model.get(p_cid, "Legacy Main")
            current_model = f"Subagents spawned by {parent_model_name}"
        else:
            current_model = UNTAGGED_MAIN_LABEL

        target_map = main_stats if (include_subagents or not is_subagent_thread) else sub_stats

        turns_in_current_req = 0
        req_model = current_model
        models_in_thread: Dict[str, int] = defaultdict(int)
        thread_had_turn_in_window = False

        for idx, step in enumerate(steps):
            if not step:
                continue
            stype = step.get("type", "")
            content = str(step.get("content") or "")
            created_at = step.get("created_at")
            step_ts = parse_ts(created_at) if created_at else None

            if stype == "USER_INPUT":
                if turns_in_current_req > 0:
                    b_prev = target_map[req_model]
                    if is_subagent_thread:
                        b_prev["is_subagent"] = True
                        b_prev["parent_model"] = parent_model_name
                    b_prev["turns_per_user_msg"].append(turns_in_current_req)
                    turns_in_current_req = 0
                matches = MODEL_CHANGE_RE.findall(content)
                if matches:
                    current_model = matches[-1][1].strip()
                req_model = current_model

            elif stype == "PLANNER_RESPONSE":
                if cutoff_ts is not None and step_ts is not None and step_ts < cutoff_ts:
                    continue

                thread_had_turn_in_window = True
                m = target_map[current_model]
                if is_subagent_thread:
                    m["is_subagent"] = True
                    m["parent_model"] = parent_model_name

                m["threads"].add(conv_id)
                m["planner_turns"] += 1
                models_in_thread[current_model] += 1
                turns_in_current_req += 1

                if created_at:
                    c_str = str(created_at)
                    if m["first_seen"] is None or c_str < m["first_seen"]:
                        m["first_seen"] = c_str
                    if m["last_seen"] is None or c_str > m["last_seen"]:
                        m["last_seen"] = c_str

                in_tok = step.get("input_tokens")
                cache_tok = step.get("cache_read_tokens")
                out_tok = step.get("output_tokens")
                if isinstance(in_tok, int):
                    m["input_tokens"].append(in_tok)
                if isinstance(cache_tok, int):
                    m["cache_read_tokens"].append(cache_tok)
                if isinstance(out_tok, int):
                    m["output_tokens"].append(out_tok)

                thinking = str(step.get("thinking") or "")
                if thinking.strip():
                    m["thinking_turns"] += 1
                    m["thinking_chars"].append(len(thinking))
                else:
                    m["thinking_chars"].append(0)

                tcalls = step.get("tool_calls") or []
                if isinstance(tcalls, list) and tcalls:
                    m["tool_turns"] += 1
                    m["total_tool_calls"] += len(tcalls)

                    if idx + 1 < len(steps) and steps[idx + 1].get("type") == "GENERIC":
                        t0 = step_ts
                        t1 = parse_ts(steps[idx + 1].get("created_at"))
                        if t0 is not None and t1 is not None:
                            dt = t1 - t0
                            if 0.0 <= dt < 300.0:
                                m["llm_latencies_s"].append(dt)
                                if isinstance(out_tok, int) and out_tok <= 350:
                                    m["pure_tool_latencies_s"].append(dt)
                                if isinstance(out_tok, int) and out_tok > 0 and dt >= 1.0:
                                    m["output_toks_per_sec"].append(out_tok / dt)

                    j = idx + 1
                    while j < len(steps) and steps[j].get("type") == "GENERIC":
                        if steps[j].get("status") == "ERROR":
                            m["tool_errors"] += 1
                        if steps[j].get("truncated_fields"):
                            m["truncations"] += 1
                        j += 1

        if turns_in_current_req > 0:
            b_last = target_map[req_model]
            if is_subagent_thread:
                b_last["is_subagent"] = True
                b_last["parent_model"] = parent_model_name
            b_last["turns_per_user_msg"].append(turns_in_current_req)

        if thread_had_turn_in_window:
            scanned_in_window += 1

        if models_in_thread:
            dominant = max(models_in_thread.items(), key=lambda kv: kv[1])[0]
            planner_ctxs = [
                (int(s.get("input_tokens") or 0) + int(s.get("cache_read_tokens") or 0))
                for s in steps
                if s and s.get("type") == "PLANNER_RESPONSE"
                and (
                    cutoff_ts is None
                    or (parse_ts(s.get("created_at")) or cutoff_ts) >= cutoff_ts
                )
            ]
            peak_ctx = max(planner_ctxs) if planner_ctxs else 0
            if peak_ctx > 0:
                target_map[dominant]["thread_csis"].append(round(peak_ctx / 200_000.0, 3))

    main_rows = [
        _summarize_bucket(model, bucket)
        for model, bucket in main_stats.items()
        if bucket["planner_turns"] > 0
    ]
    sub_rows = [
        _summarize_bucket(model, bucket)
        for model, bucket in sub_stats.items()
        if bucket["planner_turns"] > 0
    ]

    all_sub_summaries = sub_rows + [r for r in main_rows if r.get("is_subagent_cohort")]
    _apply_subagent_laundering_tax(main_rows, all_sub_summaries)

    def _leaderboard_sort_key(r: Dict[str, Any]) -> Tuple[int, int, int, float, int]:
        is_sub = 1 if r.get("is_subagent_cohort") else 0
        is_legacy = 1 if r["model"] == UNTAGGED_MAIN_LABEL else 0
        eff = (
            r.get("fleet_task_llm_s")
            if r.get("fleet_task_llm_s") is not None
            else r.get("est_task_llm_s")
        )
        has_no_est = 1 if eff is None else 0
        return (
            is_sub,
            is_legacy,
            has_no_est,
            float(eff) if eff is not None else 9999.0,
            -int(r["planner_turns"]),
        )

    main_rows.sort(key=_leaderboard_sort_key)
    sub_rows.sort(key=lambda r: int(r["planner_turns"]), reverse=True)

    main_turns = sum(int(r["planner_turns"]) for r in main_rows if not r.get("is_subagent_cohort"))
    sub_turns = sum(int(r["planner_turns"]) for r in sub_rows) + sum(
        int(r["planner_turns"]) for r in main_rows if r.get("is_subagent_cohort")
    )

    daily_driver = select_recommended_daily_driver(main_rows)

    return {
        "schema_version": "1.0.0",
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "brain_dir": str(resolved_brain),
        "window": win_norm,
        "include_subagents": include_subagents,
        "total_transcripts_scanned": scanned_in_window,
        "total_planner_turns": main_turns + sub_turns,
        "main_thread_turns": main_turns,
        "subagent_turns": sub_turns,
        "recommended_daily_driver": daily_driver,
        "models": main_rows,
        "subagent_cohorts": sub_rows,
        "elapsed_ms": round((time.perf_counter() - t_start) * 1000.0, 2),
    }


def _fmt_sec(v: Optional[float]) -> str:
    return f"{v:.1f}s" if v is not None else "—"


def _fmt_num(v: Optional[float], decimals: int = 1, suffix: str = "") -> str:
    if v is None:
        return "—"
    return f"{v:.{decimals}f}{suffix}"


def format_model_comparison_terminal(report: Dict[str, Any]) -> str:
    """Format box-drawing terminal leaderboard for `body scorecard --by-model`."""
    win = str(report.get("window", "30d")).upper()
    scanned = report.get("total_transcripts_scanned", 0)
    tot_turns = report.get("total_planner_turns", 0)
    main_turns = report.get("main_thread_turns", 0)
    sub_turns = report.get("subagent_turns", 0)
    elapsed_ms = report.get("elapsed_ms", 0.0)
    driver = report.get("recommended_daily_driver")
    models: List[Dict[str, Any]] = report.get("models") or []
    sub_cohorts: List[Dict[str, Any]] = report.get("subagent_cohorts") or []

    header = (
        f"║ {'MODEL':<28} │ {'THD':>3} │ {'TURNS':>5} │ {'DISP':>6} │ "
        f"{'MED/P90':>11} │ {'TOK/S':>5} │ {'THINK %':>10} │ {'BTCH':>4} │ "
        f"{'ERR%':>5} │ {'T/RQ':>4} │ {'MAIN':>6} │ {'+SUB':>7} │ {'FLEET':>7} │ {'CSI':>6} ║"
    )
    width = len(header)
    inner = width - 2

    def _box_line(body: str, extra_visual_cols: int = 0) -> str:
        target = inner - extra_visual_cols
        clipped = body[:target]
        return "║" + clipped.ljust(target) + "║"

    lines = [
        "",
        "╔" + "═" * inner + "╗",
        _box_line(f"  🏆 WHOLE BODY COMPUTE — EMPIRICAL MULTI-MODEL LEADERBOARD ({win} WINDOW)", extra_visual_cols=1),
        _box_line(
            f"  Transcripts: {scanned:,}  │  Total Turns: {tot_turns:,} "
            f"(Main: {main_turns:,} • Subagents: {sub_turns:,})  │  Scan: {elapsed_ms:.1f}ms"
        ),
        "╠" + "═" * inner + "╣",
    ]

    if driver:
        fleet_s = driver.get("fleet_task_llm_s", driver["est_task_llm_s"])
        tax_s = float(driver.get("subagent_tax_s") or 0.0)
        crown_body = (
            f"  👑 RECOMMENDED DAILY DRIVER: {driver['model']}  —  "
            f"{fleet_s:.1f}s fleet LLM "
            f"({driver['est_task_llm_s']:.1f}s main + {tax_s:.1f}s sub tax │ "
            f"{driver['med_tool_dispatch_s']:.1f}s × {driver['med_turns_per_req']:.1f} turns/req, "
            f"CSI {driver['med_peak_csi'] if driver['med_peak_csi'] is not None else '—'}, "
            f"{driver['tool_err_pct']:.2f}% err)"
        )
        lines.append(_box_line(crown_body, extra_visual_cols=1))
        lines.append("╠" + "═" * inner + "╣")

    lines.append(header)
    lines.append("╟" + "─" * inner + "╢")

    if not models:
        lines.append(_box_line("  No model transcript turns found in the selected window."))
    else:
        driver_name = driver["model"] if driver else None
        for r in models:
            m_name = str(r["model"])
            prefix = "* " if m_name == driver_name else "  "
            disp_name = (prefix + m_name)[:28]
            med_p90 = f"{_fmt_sec(r.get('med_llm_lat_s'))}/{_fmt_sec(r.get('p90_llm_lat_s'))}"
            think_s = f"{r.get('thinking_pct', 0.0):>3.0f}%({r.get('mean_thinking_chars', 0):>3}c)"
            csi_val = r.get("med_peak_csi")
            if csi_val is None:
                csi_s = "  —   "
            elif csi_val >= 1.0:
                csi_s = f"{csi_val:>5.2f}!"
            else:
                csi_s = f"{csi_val:>5.2f} "

            tax_val = float(r.get("subagent_tax_s") or 0.0)
            sub_tax_str = f"+{tax_val:.1f}s" if tax_val > 0 else "+0.0s"
            fleet_val = (
                r.get("fleet_task_llm_s")
                if r.get("fleet_task_llm_s") is not None
                else r.get("est_task_llm_s")
            )

            row_str = (
                f"║ {disp_name:<28} │ {r['threads']:>3} │ {r['planner_turns']:>5} │ "
                f"{_fmt_sec(r.get('med_tool_dispatch_s')):>6} │ {med_p90:>11} │ "
                f"{_fmt_num(r.get('med_tok_per_s'), 1):>5} │ {think_s:>10} │ "
                f"{r.get('tools_per_turn', 0.0):>4.2f} │ {r.get('tool_err_pct', 0.0):>4.1f}% │ "
                f"{_fmt_num(r.get('med_turns_per_req'), 1):>4} │ "
                f"{_fmt_sec(r.get('est_task_llm_s')):>6} │ {sub_tax_str:>7} │ "
                f"{_fmt_sec(fleet_val):>7} │ {csi_s:>6} ║"
            )
            lines.append(row_str)

    if sub_cohorts and not report.get("include_subagents"):
        lines.append("╠" + "═" * inner + "╣")
        lines.append(
            _box_line(
                "  🐝 SUBAGENT SWARM COHORTS (Attributed into `+SUB` & `FLEET` above; pass `--include-subagents` to merge)",
                extra_visual_cols=1,
            )
        )
        lines.append("╟" + "─" * inner + "╢")
        for s in sub_cohorts:
            s_name = ("  " + str(s["model"]))[:48]
            sub_body = (
                f" {s_name:<48} │ Child Threads: {s['threads']:>4} │ "
                f"Turns: {s['planner_turns']:>6} │ Dispatch: {_fmt_sec(s.get('med_tool_dispatch_s')):>6} │ "
                f"Turns/Task: {_fmt_num(s.get('med_turns_per_req'), 1):>5} │ "
                f"Out: {s.get('total_out_m_tok', 0.0):>6.3f}M tok"
            )
            lines.append(_box_line(sub_body))

    lines.append("╚" + "═" * inner + "╝")
    lines.append("")
    return "\n".join(lines)


def format_model_comparison_markdown(report: Dict[str, Any]) -> str:
    """Format compact Markdown receipt / report for `--receipt` and `--output <path>`."""
    win = str(report.get("window", "30d")).upper()
    scanned = report.get("total_transcripts_scanned", 0)
    tot_turns = report.get("total_planner_turns", 0)
    driver = report.get("recommended_daily_driver")
    models: List[Dict[str, Any]] = report.get("models") or []
    sub_cohorts: List[Dict[str, Any]] = report.get("subagent_cohorts") or []

    md = [
        f"# 🏆 Whole Body Compute — Multi-Model Leaderboard Receipt (`{win}`)",
        (
            f"**Generated:** `{report.get('generated_at')}`  |  "
            f"**Transcripts:** `{scanned:,}`  |  "
            f"**Planner Turns:** `{tot_turns:,}` (`{report.get('main_thread_turns', 0):,}` main / "
            f"`{report.get('subagent_turns', 0):,}` subagent)"
        ),
        "",
    ]

    if driver:
        fleet_s = driver.get("fleet_task_llm_s", driver["est_task_llm_s"])
        tax_s = float(driver.get("subagent_tax_s") or 0.0)
        md.extend(
            [
                f"> **👑 Recommended Daily Driver:** **{driver['model']}** — `{fleet_s:.1f}s` fleet task LLM time "
                f"(`{driver['est_task_llm_s']:.1f}s` main + `{tax_s:.1f}s` subagent tax; "
                f"`{driver['med_tool_dispatch_s']:.1f}s` tool dispatch × `{driver['med_turns_per_req']:.1f}` med turns/req, "
                f"Med Peak CSI `{driver['med_peak_csi']}`, Tool Err `{driver['tool_err_pct']:.2f}%`)",
                "",
            ]
        )

    md.extend(
        [
            "## 1. Main-Thread Model Scorecard (Subagent-Tax Adjusted)",
            "",
            "| Model | Threads | Turns | Dispatch (`s`) | LLM Med / P90 (`s`) | Stream (`tok/s`) | Thinking (`%` / chars) | Batch (`tools/turn`) | Tool Err (`%`) | Turns / Req | Main LLM (`s`) | +Sub Tax (`s`) | Fleet LLM (`s`) | Med Peak `CSI` | Cache Hit (`%`) |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    driver_name = driver["model"] if driver else None
    for r in models:
        label = f"👑 **{r['model']}**" if r["model"] == driver_name else f"**{r['model']}**"
        med_p90 = f"{_fmt_sec(r.get('med_llm_lat_s'))} / {_fmt_sec(r.get('p90_llm_lat_s'))}"
        csi_v = r.get("med_peak_csi")
        if csi_v is None:
            csi_s = "—"
        elif csi_v >= 1.0:
            csi_s = f"{csi_v:.2f} 🔴"
        else:
            csi_s = f"{csi_v:.2f}"
        tax_val = float(r.get("subagent_tax_s") or 0.0)
        ratio_val = float(r.get("subagent_turn_ratio") or 0.0)
        tax_cell = f"+{tax_val:.1f}s (`{ratio_val:.2f}x`)" if tax_val > 0 else "+0.0s"
        fleet_val = (
            r.get("fleet_task_llm_s")
            if r.get("fleet_task_llm_s") is not None
            else r.get("est_task_llm_s")
        )
        md.append(
            f"| {label} | {r['threads']} | {r['planner_turns']:,} | {_fmt_sec(r.get('med_tool_dispatch_s'))} | "
            f"{med_p90} | {_fmt_num(r.get('med_tok_per_s'), 1)} | "
            f"{r.get('thinking_pct', 0.0):.1f}% (`{r.get('mean_thinking_chars', 0)}c`) | "
            f"{r.get('tools_per_turn', 0.0):.2f} | {r.get('tool_err_pct', 0.0):.2f}% | "
            f"{_fmt_num(r.get('med_turns_per_req'), 1)} | {_fmt_sec(r.get('est_task_llm_s'))} | "
            f"{tax_cell} | **{_fmt_sec(fleet_val)}** | {csi_s} | {r.get('cache_hit_pct', 0.0):.1f}% |"
        )

    if sub_cohorts and not report.get("include_subagents"):
        md.extend(
            [
                "",
                "## 2. Subagent Swarm Cohorts (`invoke_subagent` Lineage)",
                "",
                "| Subagent Cohort | Child Threads | Planner Turns | Med Dispatch (`s`) | Turns / Task | Total Output (`M tok`) |",
                "| :--- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for s in sub_cohorts:
            md.append(
                f"| **{s['model']}** | {s['threads']} | {s['planner_turns']:,} | "
                f"{_fmt_sec(s.get('med_tool_dispatch_s'))} | {_fmt_num(s.get('med_turns_per_req'), 1)} | "
                f"{s.get('total_out_m_tok', 0.0):.3f}M |"
            )

    md.append("")
    return "\n".join(md)
