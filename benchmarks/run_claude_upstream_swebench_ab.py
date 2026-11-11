#!/usr/bin/env python3
"""
Upstream `princeton-nlp/SWE-bench_Verified` Cross-Harness Benchmark on Claude Code (`2,000L` Default Read Ceiling).

Evaluates untouched historical GitHub commit SHAs (`base_commit`), verbatim upstream `problem_statement` prompts,
and official held-out `test_patch` (`FAIL_TO_PASS` + `PASS_TO_PASS`) verification across all `N=10` official repos
using `claude --setting-sources local --model sonnet -p` (20 live runs: 10 Vanilla vs. 10 `+ Whole Body Compute`).

Directly quantifies how much wider the Whole Body Compute context/token savings gap grows on Claude Code's
2,000-line native `Read` ceiling compared to Antigravity's 800-line built-in ceiling.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from benchmarks.upstream_swebench_verified_harness import (
    CACHE_BASE,
    TARGET_INSTANCES,
    compute_distribution_stats,
    compute_pass_metrics,
    copy_tree_cow,
    evaluate_sandbox,
    extract_model_patch,
    load_parquet_metadata,
    setup_all,
)

CLAUDE_SANDBOX_BASE = Path("/tmp/wbc_claude_upstream_swebench")
RESULTS_JSON = REPO_ROOT / "benchmarks" / "live_claude_upstream_swebench_results.json"
REPORT_MD = REPO_ROOT / "benchmarks" / "LIVE_CLAUDE_UPSTREAM_SWEBENCH_REPORT_2026_11_10.md"
RESULTS_K3_JSON = REPO_ROOT / "benchmarks" / "live_claude_upstream_swebench_k3_results.json"
REPORT_K3_MD = REPO_ROOT / "benchmarks" / "LIVE_CLAUDE_UPSTREAM_SWEBENCH_K3_REPORT_2026_11_10.md"
CROSS_HARNESS_120_MD = REPO_ROOT / "benchmarks" / "CROSS_HARNESS_120_TRIAL_SWEBENCH_K3_REPORT.md"
GEMINI_K3_RESULTS_JSON = REPO_ROOT / "benchmarks" / "live_upstream_swebench_k3_results.json"
CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"


def parse_claude_project_transcript(arm_dir: Path) -> dict[str, int]:
    """Parse tool-use telemetry from `~/.claude/projects/<encoded_cwd>/*.jsonl` for a sandbox."""
    stats = {
        "total_tools": 0,
        "read_total": 0,
        "read_bounded": 0,
        "read_unbounded": 0,
        "bash_calls": 0,
        "edit_calls": 0,
        "write_calls": 0,
        "web_search_calls": 0,
        "web_fetch_calls": 0,
        "tool_search_calls": 0,
        "bbb_hook_interceptions": 0,
        "bbb_bounded_fast_path": 0,
        "bbb_autonomic_cues": 0,
    }
    try:
        resolved_str = str(arm_dir.resolve())
    except Exception:
        resolved_str = str(arm_dir)
    candidates = [
        resolved_str.replace("/", "-").replace("_", "-"),
        str(arm_dir).replace("/", "-").replace("_", "-"),
    ]
    proj_dir: Path | None = None
    for cand in candidates:
        p = CLAUDE_PROJECTS_DIR / cand
        if p.is_dir():
            proj_dir = p
            break
    if proj_dir is None:
        return stats

    jsonl_files = sorted(proj_dir.glob("*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not jsonl_files:
        return stats

    seen_tool_ids: set[str] = set()
    for ln in jsonl_files[0].read_text(encoding="utf-8", errors="replace").splitlines():
        if not ln.strip():
            continue
        try:
            d = json.loads(ln)
        except Exception:
            continue
        dtype = d.get("type")
        if dtype == "assistant":
            msg = d.get("message", {})
            if not isinstance(msg, dict):
                continue
            for block in msg.get("content", []):
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                tid = str(block.get("id") or "")
                if tid and tid in seen_tool_ids:
                    continue
                if tid:
                    seen_tool_ids.add(tid)
                tname = str(block.get("name") or "")
                tinp = block.get("input", {}) if isinstance(block.get("input"), dict) else {}
                stats["total_tools"] += 1
                if tname == "Read":
                    stats["read_total"] += 1
                    if "offset" in tinp or "limit" in tinp:
                        stats["read_bounded"] += 1
                    else:
                        stats["read_unbounded"] += 1
                elif tname == "Bash":
                    stats["bash_calls"] += 1
                elif tname in ("Edit", "MultiEdit"):
                    stats["edit_calls"] += 1
                elif tname == "Write":
                    stats["write_calls"] += 1
                elif tname == "WebSearch":
                    stats["web_search_calls"] += 1
                elif tname == "WebFetch":
                    stats["web_fetch_calls"] += 1
                elif tname == "ToolSearch":
                    stats["tool_search_calls"] += 1
        elif dtype == "attachment":
            att = d.get("attachment", {})
            if isinstance(att, dict) and "hook" in str(att.get("type", "")).lower():
                stats["bbb_hook_interceptions"] += 1
                att_str = json.dumps(att)
                if "CSI" in att_str or "Retry-Churn" in att_str:
                    stats["bbb_autonomic_cues"] += 1
    return stats


def get_model_paths(model_alias: str) -> tuple[Path, Path, Path]:
    """Return (sandbox_base, results_json, report_md) parameterized by model_alias."""
    m = model_alias.strip().lower()
    if m == "sonnet":
        return CLAUDE_SANDBOX_BASE, RESULTS_JSON, REPORT_MD
    sandbox_base = Path(f"/tmp/wbc_claude_upstream_swebench_{m}")
    results_json = REPO_ROOT / "benchmarks" / f"live_claude_{m}_upstream_swebench_results.json"
    report_md = REPO_ROOT / "benchmarks" / f"LIVE_CLAUDE_{m.upper()}_UPSTREAM_SWEBENCH_REPORT_2026_11_10.md"
    return sandbox_base, results_json, report_md


def provision_claude_sandboxes(
    instances: list[dict[str, Any]],
    meta: dict[str, dict[str, Any]],
    sandbox_base: Path = CLAUDE_SANDBOX_BASE,
    trial_ids: list[int] | None = None,
) -> None:
    """Provision APFS CoW sandboxes for Vanilla and `+ Whole Body Compute` Claude Code arms across specified trial_ids."""
    t_ids = trial_ids if trial_ids is not None else [1]
    # Ensure base repo_cache and venv_cache exist
    missing_cache = any(
        not (CACHE_BASE / f"repo_{item['short_id']}" / ".git").exists()
        or not (CACHE_BASE / f"venv_{item['short_id']}" / "bin" / "python").exists()
        for item in instances
    )
    if missing_cache:
        setup_all(instances=instances, trials=1, sandbox_base=sandbox_base)

    sandbox_base.mkdir(parents=True, exist_ok=True)
    hook_path = (REPO_ROOT / "bin" / "body-immune-hook").resolve()

    for t_id in t_ids:
        for item in instances:
            iid = item["instance_id"]
            sid = item["short_id"]
            rec = meta[iid]
            repo = rec["repo"]
            sha = rec["base_commit"]
            repo_cache = CACHE_BASE / f"repo_{sid}"
            venv_dir = CACHE_BASE / f"venv_{sid}"

            problem_text = (
                f"# Official SWE-bench Verified Issue: `{iid}` (`{repo}` @ `{sha[:10]}`)\n\n"
                + rec["problem_statement"].strip()
                + "\n"
            )

            for arm_sub, is_wbc in (("wbc_vanilla_arm", False), ("wbc_wbc_arm", True)):
                arm_dir = sandbox_base / arm_sub / f"{sid}_t{t_id}"
                if arm_dir.exists():
                    shutil.rmtree(arm_dir)
                arm_dir.parent.mkdir(parents=True, exist_ok=True)
                copy_tree_cow(repo_cache, arm_dir)

                if not (arm_dir / ".venv").exists():
                    os.symlink(venv_dir, arm_dir / ".venv")
                (arm_dir / "PROBLEM_STATEMENT.md").write_text(problem_text, encoding="utf-8")

                claude_dir = arm_dir / ".claude"
                claude_dir.mkdir(parents=True, exist_ok=True)
                if is_wbc:
                    settings = {
                        "hooks": {
                            "PreToolUse": [
                                {
                                    "matcher": ".*",
                                    "hooks": [
                                        {
                                            "type": "command",
                                            "command": f"'{hook_path}'",
                                            "timeout": 10,
                                        }
                                    ],
                                }
                            ],
                            "PostToolUse": [
                                {
                                    "matcher": ".*",
                                    "hooks": [
                                        {
                                            "type": "command",
                                            "command": f"'{hook_path}' --phase post",
                                            "timeout": 10,
                                        }
                                    ],
                                }
                            ],
                        }
                    }
                else:
                    settings = {"hooks": {}}

                (claude_dir / "settings.local.json").write_text(
                    json.dumps(settings, indent=2) + "\n", encoding="utf-8"
                )


def build_claude_prompt(item: dict[str, Any], arm_dir: Path) -> str:
    test_hint = item.get("worker_test_hint", "pytest -q")
    return (
        f"Solve the official SWE-bench Verified issue in this repository (`{arm_dir}`):\n"
        f"1. Read `PROBLEM_STATEMENT.md`.\n"
        f"2. Inspect the relevant source files using `Read` and apply the minimal root-cause fix using `Edit`.\n"
        f"3. Run targeted verification via `Bash`: `{test_hint}`\n"
        f"4. Rules: Work ONLY inside `{arm_dir}`. Never modify `PROBLEM_STATEMENT.md` or `.claude/settings.local.json`, "
        f"and never run destructive git commands (`git checkout`, `git reset`, `git clean`). Stop as soon as tests pass."
    )


def run_single_claude_trial(
    item: dict[str, Any],
    rec: dict[str, Any],
    is_wbc: bool,
    model_alias: str = "sonnet",
    sandbox_base: Path = CLAUDE_SANDBOX_BASE,
    timeout_s: int = 900,
    trial_id: int = 1,
) -> dict[str, Any]:
    sid = item["short_id"]
    iid = item["instance_id"]
    arm_sub = "wbc_wbc_arm" if is_wbc else "wbc_vanilla_arm"
    arm_key = "wbc" if is_wbc else "vanilla"
    arm_prefix = "wbc" if is_wbc else "van"
    worker_name = f"{arm_prefix}-{sid}-t{trial_id}"
    arm_label = f"Claude {model_alias.capitalize()} + Whole Body Compute" if is_wbc else f"Vanilla Claude {model_alias.capitalize()}"
    arm_dir = sandbox_base / arm_sub / f"{sid}_t{trial_id}"

    claude_bin = shutil.which("claude") or str(Path.home() / ".nvm/versions/node/v22.22.0/bin/claude")
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["WBC_IMMUNE_DEDUP_FILE"] = str(arm_dir / ".wbc_dedup")
    env["WBC_DATA_DIR"] = str(arm_dir / ".wbc_data")
    venv_bin = str((REPO_ROOT / ".venv" / "bin").resolve())
    env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"

    prompt = build_claude_prompt(item, arm_dir)
    t0 = time.perf_counter()
    timed_out = False
    try:
        proc = subprocess.run(
            [
                claude_bin,
                "--setting-sources",
                "local",
                "--model",
                model_alias,
                "-p",
                prompt,
                "--output-format",
                "json",
                "--dangerously-skip-permissions",
            ],
            cwd=arm_dir,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_s,
            check=False,
        )
        stdout_str = proc.stdout.strip()
        stderr_str = proc.stderr.strip()
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout_str = (exc.stdout or "").decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr_str = f"TimeoutExpired after {timeout_s}s"

    wall_s = round(time.perf_counter() - t0, 2)

    raw_json: dict[str, Any] = {}
    if stdout_str:
        try:
            raw_json = json.loads(stdout_str)
        except (json.JSONDecodeError, ValueError):
            # Fallback if trailing line contains JSON
            for ln in reversed(stdout_str.splitlines()):
                ln = ln.strip()
                if ln.startswith("{") and ln.endswith("}"):
                    try:
                        raw_json = json.loads(ln)
                        break
                    except Exception:
                        pass

    tool_telemetry = parse_claude_project_transcript(arm_dir)

    # Strip untracked runtime state files before evaluating sandbox so git diff stays clean
    for aux in (".claude", ".wbc_data", ".wbc_dedup", ".wbc_dedup.res"):
        p_aux = arm_dir / aux
        if p_aux.is_dir():
            shutil.rmtree(p_aux, ignore_errors=True)
        elif p_aux.exists():
            p_aux.unlink(missing_ok=True)

    f2p_passed, eval_log = evaluate_sandbox(item, rec, arm_dir)
    patch_diff = extract_model_patch(arm_dir)

    usage = raw_json.get("usage", {}) if isinstance(raw_json, dict) else {}
    input_tok = int(usage.get("input_tokens", 0) or 0)
    cache_create_tok = int(usage.get("cache_creation_input_tokens", 0) or 0)
    cache_read_tok = int(usage.get("cache_read_input_tokens", 0) or 0)
    output_tok = int(usage.get("output_tokens", 0) or 0)
    billed_input = input_tok + cache_create_tok
    total_context = billed_input + cache_read_tok
    num_turns = int(raw_json.get("num_turns", 0) or 0)
    total_cost_usd = round(float(raw_json.get("total_cost_usd", 0.0) or 0.0), 5)

    print(
        f"[{arm_key.upper():7s} t{trial_id}] {iid:28s} | {'PASS' if f2p_passed else 'FAIL'} | "
        f"turns={num_turns:3d} | wall={wall_s:6.1f}s | ctx={total_context:10,d} | "
        f"billed={billed_input:8,d} | cost=${total_cost_usd:.4f}",
        flush=True,
    )

    return {
        "worker_name": worker_name,
        "trial": trial_id,
        "short_id": sid,
        "instance_id": iid,
        "primary_file": item["primary_file"],
        "primary_file_lines": item["primary_file_lines"],
        "arm": arm_key,
        "arm_label": arm_label,
        "model": model_alias,
        "resolved": bool(f2p_passed),
        "timed_out": timed_out,
        "turns": num_turns,
        "wall_seconds": wall_s,
        "input_tokens": input_tok,
        "cache_creation_input_tokens": cache_create_tok,
        "billed_input_tokens": billed_input,
        "cache_read_tokens": cache_read_tok,
        "total_context_tokens": total_context,
        "output_tokens": output_tok,
        "total_cost_usd": total_cost_usd,
        "read_file_calls": tool_telemetry["read_total"],
        "read_bounded_calls": tool_telemetry["read_bounded"],
        "read_unbounded_calls": tool_telemetry["read_unbounded"],
        "bash_calls": tool_telemetry["bash_calls"],
        "edit_calls": tool_telemetry["edit_calls"] + tool_telemetry["write_calls"],
        "web_search_calls": tool_telemetry["web_search_calls"],
        "web_fetch_calls": tool_telemetry["web_fetch_calls"],
        "total_tool_calls": tool_telemetry["total_tools"],
        "patch_lines": len(patch_diff.splitlines()),
        "eval_tail": eval_log[-400:],
        "stderr_tail": stderr_str[-300:] if stderr_str else "",
    }


def enrich_run_with_tool_telemetry(
    run: dict[str, Any], sandbox_base: Path = CLAUDE_SANDBOX_BASE
) -> dict[str, Any]:
    """Ensure a run dict has `trial`, `worker_name`, and parsed tool-use telemetry from `~/.claude/projects`."""
    enriched = dict(run)
    t_id = int(enriched.get("trial", 1) or 1)
    enriched["trial"] = t_id
    sid = enriched["short_id"]
    arm_key = enriched["arm"]
    arm_prefix = "wbc" if arm_key == "wbc" else "van"
    arm_sub = "wbc_wbc_arm" if arm_key == "wbc" else "wbc_vanilla_arm"
    enriched.setdefault("worker_name", f"{arm_prefix}-{sid}-t{t_id}")
    if "read_file_calls" not in enriched or enriched.get("total_tool_calls", 0) == 0:
        arm_dir = sandbox_base / arm_sub / f"{sid}_t{t_id}"
        tt = parse_claude_project_transcript(arm_dir)
        enriched["read_file_calls"] = tt["read_total"]
        enriched["read_bounded_calls"] = tt["read_bounded"]
        enriched["read_unbounded_calls"] = tt["read_unbounded"]
        enriched["bash_calls"] = tt["bash_calls"]
        enriched["edit_calls"] = tt["edit_calls"] + tt["write_calls"]
        enriched["web_search_calls"] = tt["web_search_calls"]
        enriched["web_fetch_calls"] = tt["web_fetch_calls"]
        enriched["total_tool_calls"] = tt["total_tools"]
    return enriched


def pct_delta(van_val: float, wbc_val: float) -> float:
    if van_val == 0:
        return 0.0
    return round(((wbc_val - van_val) / van_val) * 100.0, 2)


def compile_results_and_report(
    runs: list[dict[str, Any]],
    model_alias: str,
    instances: list[dict[str, Any]],
    results_json_path: Path | None = None,
    report_md_path: Path | None = None,
) -> dict[str, Any]:
    _, default_results_json, default_report_md = get_model_paths(model_alias)
    out_json = results_json_path or default_results_json
    out_md = report_md_path or default_report_md
    van_runs = [r for r in runs if r["arm"] == "vanilla"]
    wbc_runs = [r for r in runs if r["arm"] == "wbc"]

    def arm_agg(arm_list: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "runs_total": len(arm_list),
            "runs_resolved": sum(1 for r in arm_list if r["resolved"]),
            "pass_at_1_pct": round(
                (sum(1 for r in arm_list if r["resolved"]) / max(1, len(arm_list))) * 100.0, 2
            ),
            "total_context_stats": compute_distribution_stats([r["total_context_tokens"] for r in arm_list]),
            "billed_input_stats": compute_distribution_stats([r["billed_input_tokens"] for r in arm_list]),
            "turns_stats": compute_distribution_stats([r["turns"] for r in arm_list]),
            "wall_stats": compute_distribution_stats([r["wall_seconds"] for r in arm_list]),
            "output_stats": compute_distribution_stats([r["output_tokens"] for r in arm_list]),
            "cost_usd_stats": compute_distribution_stats([r["total_cost_usd"] for r in arm_list]),
        }

    van_summary = arm_agg(van_runs)
    wbc_summary = arm_agg(wbc_runs)

    # Also compute >2,000L monster-file subset (directly exceeding Claude Code's 2,000L ceiling)
    monster_van = [r for r in van_runs if r["primary_file_lines"] > 2000]
    monster_wbc = [r for r in wbc_runs if r["primary_file_lines"] > 2000]
    monster_van_ctx = sum(r["total_context_tokens"] for r in monster_van)
    monster_wbc_ctx = sum(r["total_context_tokens"] for r in monster_wbc)
    monster_van_billed = sum(r["billed_input_tokens"] for r in monster_van)
    monster_wbc_billed = sum(r["billed_input_tokens"] for r in monster_wbc)
    monster_van_cost = sum(r["total_cost_usd"] for r in monster_van)
    monster_wbc_cost = sum(r["total_cost_usd"] for r in monster_wbc)

    deltas = {
        "total_context_sum_delta_pct": pct_delta(
            van_summary["total_context_stats"]["sum"], wbc_summary["total_context_stats"]["sum"]
        ),
        "total_context_median_delta_pct": pct_delta(
            van_summary["total_context_stats"]["median"], wbc_summary["total_context_stats"]["median"]
        ),
        "billed_input_sum_delta_pct": pct_delta(
            van_summary["billed_input_stats"]["sum"], wbc_summary["billed_input_stats"]["sum"]
        ),
        "billed_input_median_delta_pct": pct_delta(
            van_summary["billed_input_stats"]["median"], wbc_summary["billed_input_stats"]["median"]
        ),
        "turns_sum_delta_pct": pct_delta(
            van_summary["turns_stats"]["sum"], wbc_summary["turns_stats"]["sum"]
        ),
        "wall_sum_delta_pct": pct_delta(
            van_summary["wall_stats"]["sum"], wbc_summary["wall_stats"]["sum"]
        ),
        "cost_usd_sum_delta_pct": pct_delta(
            van_summary["cost_usd_stats"]["sum"], wbc_summary["cost_usd_stats"]["sum"]
        ),
        "monster_gt2000l_context_delta_pct": pct_delta(monster_van_ctx, monster_wbc_ctx),
        "monster_gt2000l_billed_delta_pct": pct_delta(monster_van_billed, monster_wbc_billed),
        "monster_gt2000l_cost_delta_pct": pct_delta(monster_van_cost, monster_wbc_cost),
    }

    per_repo: list[dict[str, Any]] = []
    by_key = {(r["instance_id"], r["arm"]): r for r in runs}
    for item in instances:
        iid = item["instance_id"]
        rv = by_key.get((iid, "vanilla"))
        rw = by_key.get((iid, "wbc"))
        if rv and rw:
            per_repo.append(
                {
                    "instance_id": iid,
                    "primary_file": item["primary_file"],
                    "primary_file_lines": item["primary_file_lines"],
                    "vanilla_resolved": rv["resolved"],
                    "wbc_resolved": rw["resolved"],
                    "vanilla_turns": rv["turns"],
                    "wbc_turns": rw["turns"],
                    "vanilla_context": rv["total_context_tokens"],
                    "wbc_context": rw["total_context_tokens"],
                    "context_delta_pct": pct_delta(rv["total_context_tokens"], rw["total_context_tokens"]),
                    "vanilla_billed_input": rv["billed_input_tokens"],
                    "wbc_billed_input": rw["billed_input_tokens"],
                    "billed_delta_pct": pct_delta(rv["billed_input_tokens"], rw["billed_input_tokens"]),
                    "vanilla_cost_usd": rv["total_cost_usd"],
                    "wbc_cost_usd": rw["total_cost_usd"],
                    "cost_delta_pct": pct_delta(rv["total_cost_usd"], rw["total_cost_usd"]),
                }
            )

    payload = {
        "generated_at": datetime.now().isoformat(),
        "dataset": "princeton-nlp/SWE-bench_Verified",
        "harness": "Claude Code CLI (2,000-line default Read ceiling)",
        "model": model_alias,
        "instances_count": len(instances),
        "total_runs": len(runs),
        "summary": {"vanilla": van_summary, "wbc": wbc_summary},
        "deltas": deltas,
        "per_repo": per_repo,
        "runs": runs,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        f"# Upstream `princeton-nlp/SWE-bench_Verified` on Claude Code (`2,000L` Ceiling, `N={len(instances)}` Official Repos)",
        f"**Date**: `{payload['generated_at']}`  ",
        f"**Harness**: `Claude Code CLI` (`2,000`-line default `Read` ceiling, held-out official `test_patch` verification)  ",
        f"**Model**: `Claude {model_alias.capitalize()}` (`{len(runs)}` live isolated evaluations)  ",
        "",
        "## 1. Executive Cross-Harness Scorecard",
        "",
        "| Metric | Vanilla Claude Code | `+ Whole Body Compute` | Delta |",
        "| :--- | :---: | :---: | :---: |",
        f"| **`Pass@1` Resolution** | `{van_summary['runs_resolved']}/{van_summary['runs_total']}` (`{van_summary['pass_at_1_pct']}%`) | **`{wbc_summary['runs_resolved']}/{wbc_summary['runs_total']}` (`{wbc_summary['pass_at_1_pct']}%`)** | `{wbc_summary['pass_at_1_pct'] - van_summary['pass_at_1_pct']:+.1f}pp` |",
        f"| **Total Context Ingested** | `{int(van_summary['total_context_stats']['sum']):,}` | **`{int(wbc_summary['total_context_stats']['sum']):,}`** | **`{deltas['total_context_sum_delta_pct']:+.1f}%`** (Median `{deltas['total_context_median_delta_pct']:+.1f}%`) |",
        f"| **Billed Input Tokens** | `{int(van_summary['billed_input_stats']['sum']):,}` | **`{int(wbc_summary['billed_input_stats']['sum']):,}`** | **`{deltas['billed_input_sum_delta_pct']:+.1f}%`** (Median `{deltas['billed_input_median_delta_pct']:+.1f}%`) |",
        f"| **API-Equivalent Cost (`USD`)** | `${van_summary['cost_usd_stats']['sum']:.2f}` | **`${wbc_summary['cost_usd_stats']['sum']:.2f}`** | **`{deltas['cost_usd_sum_delta_pct']:+.1f}%`** |",
        f"| **Planner Turns** | `{int(van_summary['turns_stats']['sum']):,}` | **`{int(wbc_summary['turns_stats']['sum']):,}`** | **`{deltas['turns_sum_delta_pct']:+.1f}%`** |",
        f"| **Wall Time (`s`)** | `{van_summary['wall_stats']['sum']:.1f}s` | **`{wbc_summary['wall_stats']['sum']:.1f}s`** | **`{deltas['wall_sum_delta_pct']:+.1f}%`** |",
        f"| **`>2,000L` Monster Repos Context** | `{monster_van_ctx:,}` | **`{monster_wbc_ctx:,}`** | **`{deltas['monster_gt2000l_context_delta_pct']:+.1f}%`** (Billed `{deltas['monster_gt2000l_billed_delta_pct']:+.1f}%`, Cost `{deltas['monster_gt2000l_cost_delta_pct']:+.1f}%`) |",
        "",
        "## 2. Per-Repository Head-to-Head Breakdown",
        "",
        "| Official Instance | Primary File (`Lines`) | Vanilla (`Pass`, Turns, Ctx, Cost) | `+ Whole Body Compute` (`Pass`, Turns, Ctx, Cost) | Context Δ | Cost Δ |",
        "| :--- | :--- | :--- | :--- | :---: | :---: |",
    ]
    for pr in per_repo:
        v_ok = "✅" if pr["vanilla_resolved"] else "❌"
        w_ok = "✅" if pr["wbc_resolved"] else "❌"
        md_lines.append(
            f"| `{pr['instance_id']}` | `{pr['primary_file']}` (`{pr['primary_file_lines']:,}L`) | "
            f"{v_ok} `{pr['vanilla_turns']}t`, `{pr['vanilla_context']:,}`, `${pr['vanilla_cost_usd']:.2f}` | "
            f"{w_ok} `{pr['wbc_turns']}t`, `{pr['wbc_context']:,}`, `${pr['wbc_cost_usd']:.2f}` | "
            f"**`{pr['context_delta_pct']:+.1f}%`** | **`{pr['cost_delta_pct']:+.1f}%`** |"
        )

    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return payload


def _fmt_med_iqr_val(dist: dict[str, float], is_int: bool = True, prefix: str = "", suffix: str = "") -> str:
    if is_int:
        return f"`{prefix}{int(round(dist['median'])):,}{suffix}` (`IQR {prefix}{int(round(dist['iqr'])):,}{suffix}`)"
    if prefix == "$":
        return f"`${dist['median']:.2f}` (`IQR ${dist['iqr']:.2f}`)"
    return f"`{prefix}{dist['median']:.1f}{suffix}` (`IQR {prefix}{dist['iqr']:.1f}{suffix}`)"


def compile_k3_results_and_report(
    runs: list[dict[str, Any]],
    model_alias: str,
    instances: list[dict[str, Any]],
    k: int = 3,
    sandbox_base: Path = CLAUDE_SANDBOX_BASE,
    results_k3_json_path: Path = RESULTS_K3_JSON,
    report_k3_md_path: Path = REPORT_K3_MD,
    cross_harness_md_path: Path = CROSS_HARNESS_120_MD,
) -> dict[str, Any]:
    """Compile k=3 multi-trial Claude Code results + 120-trial cross-harness synthesis."""
    enriched_runs = [enrich_run_with_tool_telemetry(r, sandbox_base=sandbox_base) for r in runs]
    order_map = {it["instance_id"]: idx for idx, it in enumerate(instances)}
    enriched_runs.sort(
        key=lambda r: (
            order_map.get(r["instance_id"], 999),
            0 if r["arm"] == "vanilla" else 1,
            int(r.get("trial", 1)),
        )
    )

    van_runs = [r for r in enriched_runs if r["arm"] == "vanilla"]
    wbc_runs = [r for r in enriched_runs if r["arm"] == "wbc"]

    def build_arm_k3_summary(arm_list: list[dict[str, Any]]) -> dict[str, Any]:
        outcomes_by_iid: dict[str, list[bool]] = {it["instance_id"]: [] for it in instances}
        for r in arm_list:
            iid = r["instance_id"]
            if iid in outcomes_by_iid:
                outcomes_by_iid[iid].append(bool(r["resolved"]))
        pass_m = compute_pass_metrics(outcomes_by_iid, k=k)
        return {
            "runs_total": len(arm_list),
            "runs_resolved": sum(1 for r in arm_list if r["resolved"]),
            "instances_evaluated": pass_m["instances_evaluated"],
            "pass_at_1_pct": pass_m["pass_at_1_pct"],
            "pass_hat_3_count": pass_m["pass_hat_3_count"],
            "pass_hat_3_pct": pass_m["pass_hat_3_pct"],
            "pass_at_3_count": pass_m["pass_at_3_count"],
            "pass_at_3_pct": pass_m["pass_at_3_pct"],
            "per_instance_pass": pass_m["per_instance"],
            "total_context_stats": compute_distribution_stats([r["total_context_tokens"] for r in arm_list]),
            "billed_input_stats": compute_distribution_stats([r["billed_input_tokens"] for r in arm_list]),
            "turns_stats": compute_distribution_stats([r["turns"] for r in arm_list]),
            "wall_stats": compute_distribution_stats([r["wall_seconds"] for r in arm_list]),
            "output_stats": compute_distribution_stats([r["output_tokens"] for r in arm_list]),
            "cost_usd_stats": compute_distribution_stats([r["total_cost_usd"] for r in arm_list]),
            "read_file_calls_stats": compute_distribution_stats([r.get("read_file_calls", 0) for r in arm_list]),
            "bash_calls_stats": compute_distribution_stats([r.get("bash_calls", 0) for r in arm_list]),
            "edit_calls_stats": compute_distribution_stats([r.get("edit_calls", 0) for r in arm_list]),
            "web_leakage_total": sum(
                int(r.get("web_search_calls", 0)) + int(r.get("web_fetch_calls", 0)) for r in arm_list
            ),
        }

    van_s = build_arm_k3_summary(van_runs)
    wbc_s = build_arm_k3_summary(wbc_runs)

    # Matched-solved trial pairs (same instance_id & trial where both Vanilla and WBC solved)
    by_trial_key = {(r["instance_id"], int(r.get("trial", 1)), r["arm"]): r for r in enriched_runs}
    matched_van: list[dict[str, Any]] = []
    matched_wbc: list[dict[str, Any]] = []
    for it in instances:
        iid = it["instance_id"]
        for t_id in range(1, k + 1):
            rv = by_trial_key.get((iid, t_id, "vanilla"))
            rw = by_trial_key.get((iid, t_id, "wbc"))
            if rv and rw and rv["resolved"] and rw["resolved"]:
                matched_van.append(rv)
                matched_wbc.append(rw)

    matched_summary = {
        "paired_trials_count": len(matched_van),
        "vanilla_context_sum": sum(r["total_context_tokens"] for r in matched_van),
        "wbc_context_sum": sum(r["total_context_tokens"] for r in matched_wbc),
        "context_sum_delta_pct": pct_delta(
            sum(r["total_context_tokens"] for r in matched_van),
            sum(r["total_context_tokens"] for r in matched_wbc),
        ),
        "vanilla_billed_sum": sum(r["billed_input_tokens"] for r in matched_van),
        "wbc_billed_sum": sum(r["billed_input_tokens"] for r in matched_wbc),
        "billed_sum_delta_pct": pct_delta(
            sum(r["billed_input_tokens"] for r in matched_van),
            sum(r["billed_input_tokens"] for r in matched_wbc),
        ),
        "vanilla_cost_sum": round(sum(r["total_cost_usd"] for r in matched_van), 4),
        "wbc_cost_sum": round(sum(r["total_cost_usd"] for r in matched_wbc), 4),
        "cost_sum_delta_pct": pct_delta(
            sum(r["total_cost_usd"] for r in matched_van),
            sum(r["total_cost_usd"] for r in matched_wbc),
        ),
        "vanilla_turns_sum": sum(r["turns"] for r in matched_van),
        "wbc_turns_sum": sum(r["turns"] for r in matched_wbc),
        "turns_sum_delta_pct": pct_delta(
            sum(r["turns"] for r in matched_van),
            sum(r["turns"] for r in matched_wbc),
        ),
        "vanilla_wall_sum": round(sum(r["wall_seconds"] for r in matched_van), 2),
        "wbc_wall_sum": round(sum(r["wall_seconds"] for r in matched_wbc), 2),
        "wall_sum_delta_pct": pct_delta(
            sum(r["wall_seconds"] for r in matched_van),
            sum(r["wall_seconds"] for r in matched_wbc),
        ),
    }

    monster_van = [r for r in van_runs if r["primary_file_lines"] > 2000]
    monster_wbc = [r for r in wbc_runs if r["primary_file_lines"] > 2000]

    deltas = {
        "pass_at_1_delta_pp": round(wbc_s["pass_at_1_pct"] - van_s["pass_at_1_pct"], 2),
        "pass_hat_3_delta_pp": round(wbc_s["pass_hat_3_pct"] - van_s["pass_hat_3_pct"], 2),
        "pass_at_3_delta_pp": round(wbc_s["pass_at_3_pct"] - van_s["pass_at_3_pct"], 2),
        "total_context_sum_delta_pct": pct_delta(
            van_s["total_context_stats"]["sum"], wbc_s["total_context_stats"]["sum"]
        ),
        "total_context_median_delta_pct": pct_delta(
            van_s["total_context_stats"]["median"], wbc_s["total_context_stats"]["median"]
        ),
        "billed_input_sum_delta_pct": pct_delta(
            van_s["billed_input_stats"]["sum"], wbc_s["billed_input_stats"]["sum"]
        ),
        "billed_input_median_delta_pct": pct_delta(
            van_s["billed_input_stats"]["median"], wbc_s["billed_input_stats"]["median"]
        ),
        "turns_sum_delta_pct": pct_delta(
            van_s["turns_stats"]["sum"], wbc_s["turns_stats"]["sum"]
        ),
        "turns_median_delta_pct": pct_delta(
            van_s["turns_stats"]["median"], wbc_s["turns_stats"]["median"]
        ),
        "wall_sum_delta_pct": pct_delta(
            van_s["wall_stats"]["sum"], wbc_s["wall_stats"]["sum"]
        ),
        "wall_median_delta_pct": pct_delta(
            van_s["wall_stats"]["median"], wbc_s["wall_stats"]["median"]
        ),
        "output_sum_delta_pct": pct_delta(
            van_s["output_stats"]["sum"], wbc_s["output_stats"]["sum"]
        ),
        "output_median_delta_pct": pct_delta(
            van_s["output_stats"]["median"], wbc_s["output_stats"]["median"]
        ),
        "cost_usd_sum_delta_pct": pct_delta(
            van_s["cost_usd_stats"]["sum"], wbc_s["cost_usd_stats"]["sum"]
        ),
        "cost_usd_median_delta_pct": pct_delta(
            van_s["cost_usd_stats"]["median"], wbc_s["cost_usd_stats"]["median"]
        ),
        "read_file_calls_sum_delta_pct": pct_delta(
            van_s["read_file_calls_stats"]["sum"], wbc_s["read_file_calls_stats"]["sum"]
        ),
        "read_file_calls_median_delta_pct": pct_delta(
            van_s["read_file_calls_stats"]["median"], wbc_s["read_file_calls_stats"]["median"]
        ),
        "monster_gt2000l_context_delta_pct": pct_delta(
            sum(r["total_context_tokens"] for r in monster_van),
            sum(r["total_context_tokens"] for r in monster_wbc),
        ),
        "monster_gt2000l_billed_delta_pct": pct_delta(
            sum(r["billed_input_tokens"] for r in monster_van),
            sum(r["billed_input_tokens"] for r in monster_wbc),
        ),
        "monster_gt2000l_cost_delta_pct": pct_delta(
            sum(r["total_cost_usd"] for r in monster_van),
            sum(r["total_cost_usd"] for r in monster_wbc),
        ),
    }

    per_instance_k3: list[dict[str, Any]] = []
    for it in instances:
        iid = it["instance_id"]
        iv_runs = [r for r in van_runs if r["instance_id"] == iid]
        iw_runs = [r for r in wbc_runs if r["instance_id"] == iid]
        vp = van_s["per_instance_pass"].get(iid, {})
        wp = wbc_s["per_instance_pass"].get(iid, {})
        v_ctx_st = compute_distribution_stats([r["total_context_tokens"] for r in iv_runs])
        w_ctx_st = compute_distribution_stats([r["total_context_tokens"] for r in iw_runs])
        v_bill_st = compute_distribution_stats([r["billed_input_tokens"] for r in iv_runs])
        w_bill_st = compute_distribution_stats([r["billed_input_tokens"] for r in iw_runs])
        v_cost_st = compute_distribution_stats([r["total_cost_usd"] for r in iv_runs])
        w_cost_st = compute_distribution_stats([r["total_cost_usd"] for r in iw_runs])
        v_turns_st = compute_distribution_stats([r["turns"] for r in iv_runs])
        w_turns_st = compute_distribution_stats([r["turns"] for r in iw_runs])
        v_reads_st = compute_distribution_stats([r.get("read_file_calls", 0) for r in iv_runs])
        w_reads_st = compute_distribution_stats([r.get("read_file_calls", 0) for r in iw_runs])
        per_instance_k3.append(
            {
                "instance_id": iid,
                "short_id": it["short_id"],
                "primary_file": it["primary_file"],
                "primary_file_lines": it["primary_file_lines"],
                "vanilla_pass_rate": vp.get("pass_rate_str", "0/0"),
                "vanilla_pass_hat_3": bool(vp.get("pass_hat_k", False)),
                "wbc_pass_rate": wp.get("pass_rate_str", "0/0"),
                "wbc_pass_hat_3": bool(wp.get("pass_hat_k", False)),
                "vanilla_median_context": v_ctx_st["median"],
                "wbc_median_context": w_ctx_st["median"],
                "median_context_delta_pct": pct_delta(v_ctx_st["median"], w_ctx_st["median"]),
                "vanilla_sum_context": v_ctx_st["sum"],
                "wbc_sum_context": w_ctx_st["sum"],
                "sum_context_delta_pct": pct_delta(v_ctx_st["sum"], w_ctx_st["sum"]),
                "vanilla_median_billed": v_bill_st["median"],
                "wbc_median_billed": w_bill_st["median"],
                "median_billed_delta_pct": pct_delta(v_bill_st["median"], w_bill_st["median"]),
                "vanilla_median_cost_usd": v_cost_st["median"],
                "wbc_median_cost_usd": w_cost_st["median"],
                "median_cost_delta_pct": pct_delta(v_cost_st["median"], w_cost_st["median"]),
                "vanilla_sum_cost_usd": v_cost_st["sum"],
                "wbc_sum_cost_usd": w_cost_st["sum"],
                "sum_cost_delta_pct": pct_delta(v_cost_st["sum"], w_cost_st["sum"]),
                "vanilla_median_turns": v_turns_st["median"],
                "wbc_median_turns": w_turns_st["median"],
                "vanilla_median_reads": v_reads_st["median"],
                "wbc_median_reads": w_reads_st["median"],
            }
        )

    payload = {
        "generated_at": datetime.now().isoformat(),
        "dataset": "princeton-nlp/SWE-bench_Verified (Official HuggingFace Test Split, Untouched GitHub Commit SHAs)",
        "harness": "Claude Code CLI (2,000-line default Read ceiling, held-out official test_patch verification)",
        "model": f"Claude {model_alias.capitalize()}",
        "trials_per_instance_k": k,
        "instances_count": len(instances),
        "total_runs": len(enriched_runs),
        "summary": {"vanilla": van_s, "wbc": wbc_s},
        "matched_solved_summary": matched_summary,
        "deltas": deltas,
        "per_instance": per_instance_k3,
        "runs": enriched_runs,
    }
    results_k3_json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Render LIVE_CLAUDE_UPSTREAM_SWEBENCH_K3_REPORT_2026_11_10.md
    md = [
        f"# Verbatim Upstream `princeton-nlp/SWE-bench_Verified` Multi-Trial (`k={k}`) Report on Claude Code (`2,000L` Ceiling, `N={len(instances)}` Official Repos, `{len(enriched_runs)}` Live Trials)",
        f"**Date**: `{payload['generated_at']}`  ",
        f"**Dataset**: `{payload['dataset']}`  ",
        f"**Harness**: `{payload['harness']}`  ",
        f"**Model**: `{payload['model']}` (`Zero Synthetic Padding`, Held-Out Official `test_patch` + `FAIL_TO_PASS` Evaluation)  ",
        "",
        f"## 1. Aggregate Multi-Trial (`k={k}`) Reliability & Efficiency Scorecard",
        "",
        f"| Arm (`N={len(instances)}` Official Orgs, `{van_s['runs_total']}` Trials/Arm) | `Pass@1` (Single-Trial) | `Pass^{k}` (Strict `{k}/{k}`) | `Pass@{k}` (`>=1/{k}`) | Median Context (`IQR`) | Median Billed Input (`IQR`) | Median Cost (`IQR`) | Total Cost (`{van_s['runs_total']}` Runs) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        (
            f"| Vanilla `Claude {model_alias.capitalize()}` *(Unshielded)* | "
            f"`{van_s['runs_resolved']}/{van_s['runs_total']}` (`{van_s['pass_at_1_pct']:.1f}%`) | "
            f"`{van_s['pass_hat_3_count']}/{van_s['instances_evaluated']}` (`{van_s['pass_hat_3_pct']:.1f}%`) | "
            f"`{van_s['pass_at_3_count']}/{van_s['instances_evaluated']}` (`{van_s['pass_at_3_pct']:.1f}%`) | "
            f"{_fmt_med_iqr_val(van_s['total_context_stats'])} | "
            f"{_fmt_med_iqr_val(van_s['billed_input_stats'])} | "
            f"{_fmt_med_iqr_val(van_s['cost_usd_stats'], is_int=False, prefix='$')} | "
            f"`${van_s['cost_usd_stats']['sum']:.2f}` |"
        ),
        (
            f"| **`Claude {model_alias.capitalize()} + Whole Body Compute`** | "
            f"**`{wbc_s['runs_resolved']}/{wbc_s['runs_total']}` (`{wbc_s['pass_at_1_pct']:.1f}%`, `{deltas['pass_at_1_delta_pp']:+.1f}pp`)** | "
            f"**`{wbc_s['pass_hat_3_count']}/{wbc_s['instances_evaluated']}` (`{wbc_s['pass_hat_3_pct']:.1f}%`, `{deltas['pass_hat_3_delta_pp']:+.1f}pp`)** | "
            f"**`{wbc_s['pass_at_3_count']}/{wbc_s['instances_evaluated']}` (`{wbc_s['pass_at_3_pct']:.1f}%`, `{deltas['pass_at_3_delta_pp']:+.1f}pp`)** | "
            f"**{_fmt_med_iqr_val(wbc_s['total_context_stats'])}** (`{deltas['total_context_median_delta_pct']:+.1f}%`) | "
            f"**{_fmt_med_iqr_val(wbc_s['billed_input_stats'])}** (`{deltas['billed_input_median_delta_pct']:+.1f}%`) | "
            f"**{_fmt_med_iqr_val(wbc_s['cost_usd_stats'], is_int=False, prefix='$')}** (`{deltas['cost_usd_median_delta_pct']:+.1f}%`) | "
            f"**`$ {wbc_s['cost_usd_stats']['sum']:.2f}`** (`{deltas['cost_usd_sum_delta_pct']:+.1f}%`) |"
        ),
        "",
        "## 2. Non-Parametric (`Median` / `IQR [Q1, Q3]`) & Parametric (`Mean ± SD`) Error-Bar Telemetry",
        "",
        "| Telemetry Metric | Vanilla `Median [Q1, Q3]` (`IQR`) | `+ Whole Body Compute` `Median [Q1, Q3]` (`IQR`) | Median Δ | Vanilla `Mean ± SD` | `+ Whole Body Compute` `Mean ± SD` | Total Arm Sum Δ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    stat_rows = [
        ("Wall Time (`s`)", "wall_stats", "wall_median_delta_pct", "wall_sum_delta_pct", False, ""),
        ("Planner Turns", "turns_stats", "turns_median_delta_pct", "turns_sum_delta_pct", False, ""),
        ("Billed Input Tokens", "billed_input_stats", "billed_input_median_delta_pct", "billed_input_sum_delta_pct", True, ""),
        ("Total Context Tokens", "total_context_stats", "total_context_median_delta_pct", "total_context_sum_delta_pct", True, ""),
        ("CoT + Code Output Tokens", "output_stats", "output_median_delta_pct", "output_sum_delta_pct", True, ""),
        ("API Cost (`USD`)", "cost_usd_stats", "cost_usd_median_delta_pct", "cost_usd_sum_delta_pct", False, "$"),
        ("`Read` Calls", "read_file_calls_stats", "read_file_calls_median_delta_pct", "read_file_calls_sum_delta_pct", False, ""),
    ]
    for label, skey, med_k, sum_k, is_int, pref in stat_rows:
        vd = van_s[skey]
        wd = wbc_s[skey]
        if is_int:
            vq = f"`{int(round(vd['median'])):,}` [`{int(round(vd['q1'])):,}`, `{int(round(vd['q3'])):,}`] (`{int(round(vd['iqr'])):,}`)"
            wq = f"`{int(round(wd['median'])):,}` [`{int(round(wd['q1'])):,}`, `{int(round(wd['q3'])):,}`] (`{int(round(wd['iqr'])):,}`)"
            vm = f"`{vd['mean']:,.0f} ± {vd['sd']:,.0f}`"
            wm = f"`{wd['mean']:,.0f} ± {wd['sd']:,.0f}`"
        elif pref == "$":
            vq = f"`${vd['median']:.2f}` [`${vd['q1']:.2f}`, `${vd['q3']:.2f}`] (`${vd['iqr']:.2f}`)"
            wq = f"`${wd['median']:.2f}` [`${wd['q1']:.2f}`, `${wd['q3']:.2f}`] (`${wd['iqr']:.2f}`)"
            vm = f"`${vd['mean']:.2f} ± ${vd['sd']:.2f}`"
            wm = f"`${wd['mean']:.2f} ± ${wd['sd']:.2f}`"
        else:
            vq = f"`{vd['median']:.1f}` [`{vd['q1']:.1f}`, `{vd['q3']:.1f}`] (`{vd['iqr']:.1f}`)"
            wq = f"`{wd['median']:.1f}` [`{wd['q1']:.1f}`, `{wd['q3']:.1f}`] (`{wd['iqr']:.1f}`)"
            vm = f"`{vd['mean']:.1f} ± {vd['sd']:.1f}`"
            wm = f"`{wd['mean']:.1f} ± {wd['sd']:.1f}`"
        md.append(
            f"| {label} | {vq} | {wq} | `{deltas[med_k]:+.1f}%` | {vm} | {wm} | `{deltas[sum_k]:+.1f}%` |"
        )

    md.extend(
        [
            "",
            f"### 2.1 Matched-Solved Trial Pairs (`N={matched_summary['paired_trials_count']}` Paired Trials Solved by Both Arms)",
            "",
            "| Matched-Solved Metric | Vanilla `Claude Sonnet` | `Claude Sonnet + Whole Body Compute` | Matched Delta |",
            "| :--- | :---: | :---: | :---: |",
            f"| **Total Context Ingested** | `{matched_summary['vanilla_context_sum']:,}` | **`{matched_summary['wbc_context_sum']:,}`** | **`{matched_summary['context_sum_delta_pct']:+.1f}%`** |",
            f"| **Billed Input Tokens** | `{matched_summary['vanilla_billed_sum']:,}` | **`{matched_summary['wbc_billed_sum']:,}`** | **`{matched_summary['billed_sum_delta_pct']:+.1f}%`** |",
            f"| **API Cost (`USD`)** | `${matched_summary['vanilla_cost_sum']:.2f}` | **`${matched_summary['wbc_cost_sum']:.2f}`** | **`{matched_summary['cost_sum_delta_pct']:+.1f}%`** |",
            f"| **Planner Turns** | `{matched_summary['vanilla_turns_sum']:,}` | **`{matched_summary['wbc_turns_sum']:,}`** | **`{matched_summary['turns_sum_delta_pct']:+.1f}%`** |",
            f"| **Wall Time (`s`)** | `{matched_summary['vanilla_wall_sum']:.1f}s` | **`{matched_summary['wbc_wall_sum']:.1f}s`** | **`{matched_summary['wall_sum_delta_pct']:+.1f}%`** |",
            "",
            f"## 3. Per-Instance Multi-Trial (`k={k}`) Consistency Matrix (`N={len(instances)}` Distinct Official Repos)",
            "",
            f"| Instance ID | Primary File (`Lines`) | Vanilla (`Pass`, `Pass^{k}`, Med Ctx, Med Cost) | `+ Whole Body Compute` (`Pass`, `Pass^{k}`, Med Ctx, Med Cost) | Med Ctx Δ | Med Cost Δ |",
            "| :--- | :--- | :--- | :--- | :---: | :---: |",
        ]
    )
    for pi in per_instance_k3:
        v_hat = "✅" if pi["vanilla_pass_hat_3"] else "❌"
        w_hat = "✅" if pi["wbc_pass_hat_3"] else "❌"
        md.append(
            f"| `{pi['instance_id']}` | `{pi['primary_file']}` (`{pi['primary_file_lines']:,}L`) | "
            f"`{pi['vanilla_pass_rate']}` (`Pass^{k}` {v_hat}), `{int(round(pi['vanilla_median_context'])):,}`, `${pi['vanilla_median_cost_usd']:.2f}` | "
            f"**`{pi['wbc_pass_rate']}` (`Pass^{k}` {w_hat})**, `{int(round(pi['wbc_median_context'])):,}`, `${pi['wbc_median_cost_usd']:.2f}` | "
            f"`{pi['median_context_delta_pct']:+.1f}%` | `{pi['median_cost_delta_pct']:+.1f}%` |"
        )

    md.extend(
        [
            "",
            f"## 4. Complete Live Trial Receipts (`{len(enriched_runs)}` Sessions)",
            "",
            "| Worker | Instance ID | Trial | Arm | Official `F2P` | Turns | Reads | Wall | Billed Input | Total Context | Output | Cost (`USD`) |",
            "| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for r in enriched_runs:
        arm_lbl = "**`+ Whole Body Compute`**" if r["arm"] == "wbc" else "Vanilla"
        ver = "✅ PASS" if r["resolved"] else "❌ FAIL"
        md.append(
            f"| `{r['worker_name']}` | `{r['instance_id']}` | `t{r['trial']}` | {arm_lbl} | {ver} | "
            f"`{r['turns']}` | `{r.get('read_file_calls', 0)}` | `{r['wall_seconds']:.1f}s` | "
            f"`{r['billed_input_tokens']:,}` | `{r['total_context_tokens']:,}` | `{r['output_tokens']:,}` | `${r['total_cost_usd']:.4f}` |"
        )

    report_k3_md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    if GEMINI_K3_RESULTS_JSON.exists():
        generate_cross_harness_120_report(payload, cross_harness_md_path=cross_harness_md_path)

    return payload


def generate_cross_harness_120_report(
    claude_k3_payload: dict[str, Any],
    gemini_k3_path: Path = GEMINI_K3_RESULTS_JSON,
    cross_harness_md_path: Path = CROSS_HARNESS_120_MD,
) -> str:
    """Generate the 120-trial cross-harness synthesis report (60 Gemini 4 Carbon + 60 Claude Sonnet)."""
    gem_data = json.loads(gemini_k3_path.read_text(encoding="utf-8"))
    g_van = gem_data["summary"]["vanilla"]
    g_wbc = gem_data["summary"]["wbc"]
    g_dl = gem_data["deltas"]

    c_van = claude_k3_payload["summary"]["vanilla"]
    c_wbc = claude_k3_payload["summary"]["wbc"]
    c_dl = claude_k3_payload["deltas"]
    c_m = claude_k3_payload["matched_solved_summary"]

    total_trials = int(gem_data.get("completed_runs", 60)) + int(claude_k3_payload.get("total_runs", 60))
    van_total_resolved = int(g_van["tasks_resolved"]) + int(c_van["runs_resolved"])
    van_total_runs = int(g_van["tasks_total"]) + int(c_van["runs_total"])
    wbc_total_resolved = int(g_wbc["tasks_resolved"]) + int(c_wbc["runs_resolved"])
    wbc_total_runs = int(g_wbc["tasks_total"]) + int(c_wbc["runs_total"])

    van_pass_hat_3_sum = int(g_van["pass_hat_k_count"]) + int(c_van["pass_hat_3_count"])
    wbc_pass_hat_3_sum = int(g_wbc["pass_hat_k_count"]) + int(c_wbc["pass_hat_3_count"])
    total_inst_pairs = int(g_van["instances_evaluated"]) + int(c_van["instances_evaluated"])

    g_van_ctx_sum = int(g_van["distributions"]["total_context_tokens"]["sum"])
    g_wbc_ctx_sum = int(g_wbc["distributions"]["total_context_tokens"]["sum"])
    c_van_ctx_sum = int(c_van["total_context_stats"]["sum"])
    c_wbc_ctx_sum = int(c_wbc["total_context_stats"]["sum"])
    comb_van_ctx = g_van_ctx_sum + c_van_ctx_sum
    comb_wbc_ctx = g_wbc_ctx_sum + c_wbc_ctx_sum

    g_van_bill_sum = int(g_van["distributions"]["billed_input_tokens"]["sum"])
    g_wbc_bill_sum = int(g_wbc["distributions"]["billed_input_tokens"]["sum"])
    c_van_bill_sum = int(c_van["billed_input_stats"]["sum"])
    c_wbc_bill_sum = int(c_wbc["billed_input_stats"]["sum"])
    comb_van_bill = g_van_bill_sum + c_van_bill_sum
    comb_wbc_bill = g_wbc_bill_sum + c_wbc_bill_sum

    g_van_turns_sum = int(g_van["distributions"]["turns"]["sum"])
    g_wbc_turns_sum = int(g_wbc["distributions"]["turns"]["sum"])
    c_van_turns_sum = int(c_van["turns_stats"]["sum"])
    c_wbc_turns_sum = int(c_wbc["turns_stats"]["sum"])
    comb_van_turns = g_van_turns_sum + c_van_turns_sum
    comb_wbc_turns = g_wbc_turns_sum + c_wbc_turns_sum

    lines = [
        f"# Cross-Harness Frontier `SWE-bench_Verified` Multi-Trial (`k=3`) Report (`{total_trials}` Live Trials: `60` Gemini 4 Carbon + `60` Claude Sonnet)",
        f"**Date**: `{datetime.now().isoformat()}`  ",
        "**Dataset**: Official `princeton-nlp/SWE-bench_Verified` (`N=10` distinct official GitHub organizations/repositories, untouched historical `base_commit` SHAs, verbatim `problem_statement`, held-out official `test_patch` `FAIL_TO_PASS` + `PASS_TO_PASS` evaluation)  ",
        "**Experimental Design**: `2 Harnesses × 2 Frontier Models × 2 Scaffolds (Vanilla vs. + Whole Body Compute) × 10 Official Repos × k=3 Independent Trials = 120 Live Trials`  ",
        "",
        "## 1. Unified 120-Trial Cross-Harness Executive Scorecard",
        "",
        "| Frontier Harness & Model (`k=3`, `N=10` Repos) | Native Read Ceiling | Vanilla `Pass@1` | `+ WBC` `Pass@1` | Vanilla `Pass^3` (Strict `3/3`) | `+ WBC` `Pass^3` (Strict `3/3`) | Context Ingested (`Vanilla` → `+ WBC`) | Billed Input (`Vanilla` → `+ WBC`) | Planner Turns (`Vanilla` → `+ WBC`) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        (
            f"| **Antigravity (`Gemini 4 Carbon`)** (`60` trials) | `800L` (`46KB`) | "
            f"`{g_van['tasks_resolved']}/{g_van['tasks_total']}` (`{g_van['pass_at_1_pct']:.1f}%`) | "
            f"**`{g_wbc['tasks_resolved']}/{g_wbc['tasks_total']}` (`{g_wbc['pass_at_1_pct']:.1f}%`)** | "
            f"`{g_van['pass_hat_k_count']}/{g_van['instances_evaluated']}` (`{g_van['pass_hat_k_pct']:.1f}%`) | "
            f"**`{g_wbc['pass_hat_k_count']}/{g_wbc['instances_evaluated']}` (`{g_wbc['pass_hat_k_pct']:.1f}%`)** | "
            f"`{g_van_ctx_sum:,}` → **`{g_wbc_ctx_sum:,}` (`{g_dl['total_context_delta_pct']:+.1f}%`)** | "
            f"`{g_van_bill_sum:,}` → **`{g_wbc_bill_sum:,}` (`{g_dl['billed_input_delta_pct']:+.1f}%`)** | "
            f"`{g_van_turns_sum:,}` → **`{g_wbc_turns_sum:,}` (`{g_dl['turns_delta_pct']:+.1f}%`)** |"
        ),
        (
            f"| **Claude Code (`Claude Sonnet`)** (`60` trials) | `2,000L` (`200KB`) | "
            f"`{c_van['runs_resolved']}/{c_van['runs_total']}` (`{c_van['pass_at_1_pct']:.1f}%`) | "
            f"**`{c_wbc['runs_resolved']}/{c_wbc['runs_total']}` (`{c_wbc['pass_at_1_pct']:.1f}%`, `{c_dl['pass_at_1_delta_pp']:+.1f}pp`)** | "
            f"`{c_van['pass_hat_3_count']}/{c_van['instances_evaluated']}` (`{c_van['pass_hat_3_pct']:.1f}%`) | "
            f"**`{c_wbc['pass_hat_3_count']}/{c_wbc['instances_evaluated']}` (`{c_wbc['pass_hat_3_pct']:.1f}%`, `{c_dl['pass_hat_3_delta_pp']:+.1f}pp`)** | "
            f"`{c_van_ctx_sum:,}` → **`{c_wbc_ctx_sum:,}` (`{c_dl['total_context_sum_delta_pct']:+.1f}%` all / `{c_m['context_sum_delta_pct']:+.1f}%` matched)** | "
            f"`{c_van_bill_sum:,}` → **`{c_wbc_bill_sum:,}` (`{c_dl['billed_input_sum_delta_pct']:+.1f}%` all / `{c_m['billed_sum_delta_pct']:+.1f}%` matched)** | "
            f"`{c_van_turns_sum:,}` → **`{c_wbc_turns_sum:,}` (`{c_dl['turns_sum_delta_pct']:+.1f}%` all / `{c_m['turns_sum_delta_pct']:+.1f}%` matched)** |"
        ),
        (
            f"| **Combined Cross-Harness Frontier (`120` trials)** | `800L` & `2,000L` | "
            f"`{van_total_resolved}/{van_total_runs}` (`{(van_total_resolved / max(1, van_total_runs)) * 100:.1f}%`) | "
            f"**`{wbc_total_resolved}/{wbc_total_runs}` (`{(wbc_total_resolved / max(1, wbc_total_runs)) * 100:.1f}%`)** | "
            f"`{van_pass_hat_3_sum}/{total_inst_pairs}` (`{(van_pass_hat_3_sum / max(1, total_inst_pairs)) * 100:.1f}%`) | "
            f"**`{wbc_pass_hat_3_sum}/{total_inst_pairs}` (`{(wbc_pass_hat_3_sum / max(1, total_inst_pairs)) * 100:.1f}%`)** | "
            f"`{comb_van_ctx:,}` → **`{comb_wbc_ctx:,}` (`{pct_delta(comb_van_ctx, comb_wbc_ctx):+.1f}%`)** | "
            f"`{comb_van_bill:,}` → **`{comb_wbc_bill:,}` (`{pct_delta(comb_van_bill, comb_wbc_bill):+.1f}%`)** | "
            f"`{comb_van_turns:,}` → **`{comb_wbc_turns:,}` (`{pct_delta(comb_van_turns, comb_wbc_turns):+.1f}%`)** |"
        ),
        "",
        "## 2. Per-Repository Cross-Harness `Pass^3` & Median Context Matrix (`120` Trials)",
        "",
        "| Official Instance (`Primary File`) | `Gemini 4 Carbon` Vanilla (`Pass^3`, Med Ctx) | `Gemini 4 Carbon + WBC` (`Pass^3`, Med Ctx) | `Claude Sonnet` Vanilla (`Pass^3`, Med Ctx, Med Cost) | `Claude Sonnet + WBC` (`Pass^3`, Med Ctx, Med Cost) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    gem_per_inst = gem_data.get("per_instance_comparison", {})
    claude_per_inst = {pi["instance_id"]: pi for pi in claude_k3_payload.get("per_instance", [])}

    for it in TARGET_INSTANCES:
        iid = it["instance_id"]
        g_pi = gem_per_inst.get(iid, {}).get("arms", {})
        gv = g_pi.get("vanilla", {})
        gw = g_pi.get("wbc", {})
        cp = claude_per_inst.get(iid, {})

        gv_hat = "✅" if gv.get("pass_hat_k") else "❌"
        gw_hat = "✅" if gw.get("pass_hat_k") else "❌"
        cv_hat = "✅" if cp.get("vanilla_pass_hat_3") else "❌"
        cw_hat = "✅" if cp.get("wbc_pass_hat_3") else "❌"

        gv_ctx = int(round(gv.get("distributions", {}).get("total_context_tokens", {}).get("median", 0)))
        gw_ctx = int(round(gw.get("distributions", {}).get("total_context_tokens", {}).get("median", 0)))
        cv_ctx = int(round(cp.get("vanilla_median_context", 0)))
        cw_ctx = int(round(cp.get("wbc_median_context", 0)))
        cv_cost = float(cp.get("vanilla_median_cost_usd", 0.0))
        cw_cost = float(cp.get("wbc_median_cost_usd", 0.0))

        lines.append(
            f"| `{iid}` (`{it['primary_file_lines']:,}L`) | "
            f"`{gv.get('pass_rate_str', '3/3')}` {gv_hat} (`{gv_ctx:,}`) | "
            f"**`{gw.get('pass_rate_str', '3/3')}` {gw_hat} (`{gw_ctx:,}`)** | "
            f"`{cp.get('vanilla_pass_rate', '0/3')}` {cv_hat} (`{cv_ctx:,}`, `${cv_cost:.2f}`) | "
            f"**`{cp.get('wbc_pass_rate', '0/3')}` {cw_hat} (`{cw_ctx:,}`, `${cw_cost:.2f}`)** |"
        )

    text = "\n".join(lines) + "\n"
    cross_harness_md_path.write_text(text, encoding="utf-8")
    return text


def _parse_trial_ids_arg(trial_ids_str: str, trials_k: int) -> list[int]:
    if trial_ids_str.strip():
        return sorted({int(x.strip()) for x in trial_ids_str.split(",") if x.strip()})
    return list(range(1, trials_k + 1))


def load_banked_runs(
    model_alias: str,
    sandbox_base: Path,
    results_json_path: Path,
    results_k3_json_path: Path = RESULTS_K3_JSON,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Load existing verified runs keyed by `(instance_id, arm, trial)`."""
    banked: dict[tuple[str, str, int], dict[str, Any]] = {}
    for cand_path in (results_json_path, results_k3_json_path):
        if not cand_path.exists():
            continue
        try:
            data = json.loads(cand_path.read_text(encoding="utf-8"))
            for r in data.get("runs", []):
                if str(r.get("model", model_alias)).lower() != model_alias.lower():
                    continue
                t_id = int(r.get("trial", 1) or 1)
                key = (r["instance_id"], r["arm"], t_id)
                # Prefer runs with turns > 0
                if key not in banked or (banked[key].get("turns", 0) == 0 and r.get("turns", 0) > 0):
                    banked[key] = enrich_run_with_tool_telemetry(r, sandbox_base=sandbox_base)
        except Exception:
            pass
    return banked


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Upstream SWE-bench_Verified on Claude Code CLI")
    parser.add_argument("--model", default="sonnet", help="Claude model alias (sonnet or haiku)")
    parser.add_argument("--workers", type=int, default=6, help="Max concurrent Claude CLI workers")
    parser.add_argument("--trials", type=int, default=1, help="Total trials k per instance (e.g. 3 for k=3)")
    parser.add_argument("--trial-ids", default="", help="Comma-separated trial IDs to execute live (e.g. '2,3')")
    parser.add_argument("--instances", default="", help="Optional comma-separated short_id or instance_id filter")
    parser.add_argument("--only-unresolved", action="store_true", help="Skip (instance_id, arm, trial) entries that are already resolved=True in banked results")
    parser.add_argument("--compile-only", action="store_true", help="Only compile existing banked runs without launching Claude CLI")
    parser.add_argument("--monster-only", action="store_true", help="Only run the 5 >2,000L monster repos")
    parser.add_argument("--timeout", type=int, default=900, help="Per-trial timeout in seconds")
    args = parser.parse_args()

    sandbox_base, results_json_path, report_md_path = get_model_paths(args.model)
    trial_ids_to_run = _parse_trial_ids_arg(args.trial_ids, args.trials)
    effective_k = max([args.trials, *trial_ids_to_run])

    instances = [
        it for it in TARGET_INSTANCES if (not args.monster_only or it["primary_file_lines"] > 2000)
    ]
    if args.instances.strip():
        wanted = {tok.strip() for tok in args.instances.split(",") if tok.strip()}
        instances = [
            it for it in instances if it["short_id"] in wanted or it["instance_id"] in wanted
        ]

    meta = load_parquet_metadata()
    banked_map = load_banked_runs(args.model, sandbox_base, results_json_path)

    if not args.compile_only:
        print(
            f"Provisioning {len(instances) * 2 * len(trial_ids_to_run)} APFS CoW sandboxes for Claude Code ({args.model}, trials={trial_ids_to_run}) at {sandbox_base}...",
            flush=True,
        )
        provision_claude_sandboxes(
            instances, meta, sandbox_base=sandbox_base, trial_ids=trial_ids_to_run
        )

        jobs: list[tuple[dict[str, Any], dict[str, Any], bool, int]] = []
        for t_id in trial_ids_to_run:
            for item in instances:
                iid = item["instance_id"]
                rec = meta[iid]
                for is_wbc, arm_k in ((False, "vanilla"), (True, "wbc")):
                    existing = banked_map.get((iid, arm_k, t_id))
                    if args.only_unresolved and existing and existing.get("resolved"):
                        continue
                    jobs.append((item, rec, is_wbc, t_id))

        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(
                    run_single_claude_trial,
                    item,
                    rec,
                    is_wbc,
                    args.model,
                    sandbox_base,
                    args.timeout,
                    t_id,
                ): (item["instance_id"], is_wbc, t_id)
                for (item, rec, is_wbc, t_id) in jobs
            }
            for fut in as_completed(futs):
                res = fut.result()
                key = (res["instance_id"], res["arm"], int(res.get("trial", 1)))
                banked_map[key] = res
                # Save incremental checkpoint to RESULTS_K3_JSON if multi-trial
                if effective_k > 1:
                    all_current = [
                        v
                        for (iid_k, arm_k, t_k), v in banked_map.items()
                        if t_k <= effective_k
                    ]
                    compile_k3_results_and_report(
                        all_current,
                        args.model,
                        TARGET_INSTANCES if not args.monster_only else instances,
                        k=effective_k,
                        sandbox_base=sandbox_base,
                    )
        elapsed_total = time.perf_counter() - t_start
    else:
        elapsed_total = 0.0

    if effective_k == 1:
        runs_t1 = [
            v for (iid_k, arm_k, t_k), v in banked_map.items() if t_k == 1
        ]
        order_map = {it["instance_id"]: idx for idx, it in enumerate(instances)}
        runs_t1.sort(key=lambda r: (order_map.get(r["instance_id"], 999), 0 if r["arm"] == "vanilla" else 1))
        summary_payload = compile_results_and_report(
            runs_t1, args.model, instances, results_json_path=results_json_path, report_md_path=report_md_path
        )
        print(
            f"\nCompleted {len(runs_t1)} Claude Code (k=1) runs in {elapsed_total:.1f}s.\n"
            f"Context Δ: {summary_payload['deltas']['total_context_sum_delta_pct']:+.1f}% | "
            f"Billed Input Δ: {summary_payload['deltas']['billed_input_sum_delta_pct']:+.1f}% | "
            f"Cost Δ: {summary_payload['deltas']['cost_usd_sum_delta_pct']:+.1f}%",
            flush=True,
        )
    else:
        full_instances = TARGET_INSTANCES if not args.monster_only else instances
        all_k_runs = [
            v for (iid_k, arm_k, t_k), v in banked_map.items() if 1 <= t_k <= effective_k
        ]
        k3_payload = compile_k3_results_and_report(
            all_k_runs,
            args.model,
            full_instances,
            k=effective_k,
            sandbox_base=sandbox_base,
        )
        print(
            f"\nCompiled {len(all_k_runs)} Claude Code (k={effective_k}) SWE-bench_Verified runs (live elapsed {elapsed_total:.1f}s).\n"
            f"Vanilla Pass@1: {k3_payload['summary']['vanilla']['pass_at_1_pct']}% | Pass^{effective_k}: {k3_payload['summary']['vanilla']['pass_hat_3_pct']}%\n"
            f"WBC     Pass@1: {k3_payload['summary']['wbc']['pass_at_1_pct']}% | Pass^{effective_k}: {k3_payload['summary']['wbc']['pass_hat_3_pct']}%\n"
            f"All-Runs Context Δ: {k3_payload['deltas']['total_context_sum_delta_pct']:+.1f}% (Med {k3_payload['deltas']['total_context_median_delta_pct']:+.1f}%) | "
            f"Matched-Solved Context Δ: {k3_payload['matched_solved_summary']['context_sum_delta_pct']:+.1f}% | "
            f"Matched-Solved Cost Δ: {k3_payload['matched_solved_summary']['cost_sum_delta_pct']:+.1f}%",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
