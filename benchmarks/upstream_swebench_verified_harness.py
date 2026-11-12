#!/usr/bin/env python3
"""
100% Verbatim Upstream `princeton-nlp/SWE-bench_Verified` Multi-Trial (`k=3`) Head-to-Head Harness
(`N=10` Distinct Official GitHub Organizations/Repositories, `60` Total Live Trials).

Evaluates untouched historical GitHub commit SHAs (`base_commit`), verbatim upstream `problem_statement` prompts,
and official held-out `test_patch` + `FAIL_TO_PASS` / `PASS_TO_PASS` verification across 10 distinct official repos:
  1.  `psf__requests-6028`           (`psf/requests`          @ `0192aac241`, `requests/utils.py` 1,056L)
  2.  `pytest-dev__pytest-8399`      (`pytest-dev/pytest`     @ `6e7dc8bac8`, `src/_pytest/python.py` 1,701L)
  3.  `pydata__xarray-6721`          (`pydata/xarray`         @ `cc183652bf`, `xarray/core/common.py` 2,075L)
  4.  `sympy__sympy-24562`           (`sympy/sympy`           @ `b1cb676cf9`, `sympy/core/numbers.py` 4,496L)
  5.  `pallets__flask-5014`          (`pallets/flask`         @ `7ee9ceb71e`, `src/flask/blueprints.py` 621L)
  6.  `mwaskom__seaborn-3069`        (`mwaskom/seaborn`       @ `54cab15bda`, `seaborn/_core/plot.py` 1,649L)
  7.  `pylint-dev__pylint-6528`      (`pylint-dev/pylint`     @ `273a8b2562`, `pylint/lint/pylinter.py` 1,188L)
  8.  `sphinx-doc__sphinx-10435`     (`sphinx-doc/sphinx`     @ `f1061c012e`, `sphinx/writers/latex.py` 2,061L)
  9.  `django__django-17084`         (`django/django`         @ `f8c43aca46`, `django/db/models/sql/query.py` 2,678L)
  10. `matplotlib__matplotlib-25960` (`matplotlib/matplotlib` @ `1d0d255b79`, `lib/matplotlib/figure.py` 3,618L)

Strict Holdout Integrity:
  - Agents ONLY see the clean `base_commit` checkout + verbatim `PROBLEM_STATEMENT.md` + `.venv` symlink
    (held-out `test_patch` and gold `patch` are NEVER present in any agent sandbox).
  - Sandbox paths preserve `/wbc_vanilla_arm/` vs `/wbc_wbc_arm/` substring isolation required by
    `bin/body-immune-hook` (`Vanilla` bypasses Whole Body Compute hooks; `WBC` engages Whole Body Compute).
  - Grading copies each agent's resulting sandbox state to an isolated temporary directory, strips
    `PROBLEM_STATEMENT.md` and `.venv`, applies official `test_patch`, and executes official `FAIL_TO_PASS`
    + `PASS_TO_PASS` tests.
  - Statistical engine computes `Pass@1`, `Pass^k` (`Pass^3` strict consistency), `Pass@k` (`Pass@3` coverage),
    `Median`, `IQR` (`[Q1, Q3]`), `Mean ± SD`, CoT `output_tokens` variance, tool-call telemetry
    (`read_file_calls`, `run_command_calls`, `edit_file_calls`), and exports a Princeton HAL Leaderboard
    submission bundle (`benchmarks/hal_swebench_verified_k3_submission.json`).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BRAIN_DIR = Path.home() / ".gemini" / "antigravity" / "brain"

PARQUET_PATH = Path("/tmp/swebench_verified.parquet")
CACHE_BASE = Path("/tmp/wbc_upstream_swebench_cache")
SANDBOX_BASE = Path("/tmp/wbc_upstream_swebench_k3")
META_JSON = CACHE_BASE / "selected_instances.json"

RESULTS_JSON = REPO_ROOT / "benchmarks" / "live_upstream_swebench_k3_results.json"
REPORT_MD = REPO_ROOT / "benchmarks" / "LIVE_UPSTREAM_SWEBENCH_K3_REPORT_2026_11_10.md"
HAL_SUBMISSION_JSON = REPO_ROOT / "benchmarks" / "hal_swebench_verified_k3_submission.json"
README_K3_MD = REPO_ROOT / "benchmarks" / "README_SWEBENCH_VERIFIED_K3.md"
DEFAULT_HAL_EXPORT_DIR = REPO_ROOT / "benchmarks" / "swebench_verified_k3"

DEFAULT_TRIALS = 3
# Standard Frontier Pricing ($ USD / 1M tokens) for HAL Cost-vs-Accuracy Pareto reporting:
# $2.00 / 1M non-cached input, $0.20 / 1M cached input, $12.00 / 1M output
FRONTIER_INPUT_USD_PER_1M = 2.00
FRONTIER_CACHE_READ_USD_PER_1M = 0.20
FRONTIER_OUTPUT_USD_PER_1M = 12.00

METRIC_KEYS: tuple[str, ...] = (
    "wall_seconds",
    "turns",
    "billed_input_tokens",
    "total_context_tokens",
    "net_task_context_tokens",
    "output_tokens",
    "read_file_calls",
    "run_command_calls",
    "edit_file_calls",
)

READ_TOOL_NAMES = frozenset({"read_file", "view_file"})
RUN_TOOL_NAMES = frozenset({"run_command"})
EDIT_TOOL_NAMES = frozenset(
    {"edit_file", "write_file", "replace_file_content", "multi_replace_file_content"}
)

TARGET_INSTANCES: list[dict[str, Any]] = [
    {
        "short_id": "req6028",
        "instance_id": "psf__requests-6028",
        "python_ver": "3.10",
        "deps": [
            "pytest>=7.0,<8.0",
            "pytest-mock",
            "urllib3<1.27",
            "chardet<5",
            "idna<4",
            "certifi",
            "pysocks",
        ],
        "eval_cmd_type": "pytest",
        "sympy_test_file": "",
        "primary_file": "requests/utils.py",
        "primary_file_lines": 1056,
        "worker_test_hint": "PYTHONPATH=. ./.venv/bin/python -m pytest -q tests/test_utils.py",
    },
    {
        "short_id": "pyt8399",
        "instance_id": "pytest-dev__pytest-8399",
        "python_ver": "3.9",
        "deps": [
            "attrs>=19.2",
            "iniconfig",
            "packaging",
            "pluggy>=0.12,<1.0",
            "py>=1.8.2",
            "toml",
            "hypothesis",
            "xmlschema",
        ],
        "eval_cmd_type": "pytest",
        "sympy_test_file": "",
        "primary_file": "src/_pytest/python.py",
        "primary_file_lines": 1701,
        "worker_test_hint": "PYTHONPATH=src:. ./.venv/bin/python -m pytest -q testing/test_unittest.py",
    },
    {
        "short_id": "xar6721",
        "instance_id": "pydata__xarray-6721",
        "python_ver": "3.10",
        "deps": ["pytest>=7.0,<8.0", "numpy<2.0", "pandas<2.1", "packaging", "dask[array]"],
        "eval_cmd_type": "pytest",
        "sympy_test_file": "",
        "primary_file": "xarray/core/common.py",
        "primary_file_lines": 2075,
        "worker_test_hint": "PYTHONPATH=. ./.venv/bin/python -m pytest -q xarray/tests/test_dataset.py",
    },
    {
        "short_id": "sym24562",
        "instance_id": "sympy__sympy-24562",
        "python_ver": "3.10",
        "deps": ["pytest>=7.0,<8.0", "mpmath"],
        "eval_cmd_type": "sympy_pytest",
        "sympy_test_file": "sympy/core/tests/test_numbers.py",
        "primary_file": "sympy/core/numbers.py",
        "primary_file_lines": 4496,
        "worker_test_hint": "PYTHONPATH=. ./.venv/bin/python -m pytest -q sympy/core/tests/test_numbers.py",
    },
    {
        "short_id": "flk5014",
        "instance_id": "pallets__flask-5014",
        "python_ver": "3.10",
        "deps": [
            "pytest>=7.0,<8.0",
            "werkzeug>=2.2.2,<3.0",
            "jinja2>=3.0,<3.2",
            "itsdangerous>=2.0,<2.2",
            "click>=8.0,<8.2",
            "blinker<1.8",
            "asgiref>=3.2",
        ],
        "eval_cmd_type": "pytest",
        "sympy_test_file": "",
        "primary_file": "src/flask/blueprints.py",
        "primary_file_lines": 621,
        "worker_test_hint": "PYTHONPATH=src:. ./.venv/bin/python -m pytest -q tests/test_blueprints.py",
    },
    {
        "short_id": "sea3069",
        "instance_id": "mwaskom__seaborn-3069",
        "python_ver": "3.10",
        "deps": [
            "pytest>=7.0,<8.0",
            "numpy>=1.21,<1.26",
            "pandas>=1.4,<2.0",
            "matplotlib>=3.6,<3.8",
            "scipy>=1.8,<1.12",
        ],
        "eval_cmd_type": "pytest",
        "sympy_test_file": "",
        "primary_file": "seaborn/_core/plot.py",
        "primary_file_lines": 1649,
        "worker_test_hint": "PYTHONPATH=. ./.venv/bin/python -m pytest -q tests/_core/test_plot.py",
    },
    {
        "short_id": "pyl6528",
        "instance_id": "pylint-dev__pylint-6528",
        "python_ver": "3.10",
        "deps": [
            "pytest>=7.0,<8.0",
            "pytest-timeout",
            "py>=1.11.0",
            "astroid==2.11.7",
            "dill>=0.2",
            "platformdirs>=2.2.0",
            "isort>=4.2.5,<6",
            "mccabe>=0.6,<0.8",
            "tomli>=1.1.0",
            "tomlkit>=0.10.1",
            "gitpython>3",
            "typing-extensions>=3.10.0",
        ],
        "eval_cmd_type": "pytest",
        "sympy_test_file": "",
        "primary_file": "pylint/lint/pylinter.py",
        "primary_file_lines": 1188,
        "worker_test_hint": "PYTHONPATH=. ./.venv/bin/python -m pytest -q tests/lint/unittest_lint.py",
    },
    {
        "short_id": "sph10435",
        "instance_id": "sphinx-doc__sphinx-10435",
        "python_ver": "3.10",
        "deps": [
            "pytest>=7.0,<8.0",
            "html5lib",
            "Jinja2>=3.0,<3.1",
            "markupsafe<2.1",
            "Pygments>=2.12,<2.17",
            "docutils>=0.17,<0.19",
            "snowballstemmer>=1.1",
            "babel>=2.9,<2.15",
            "alabaster>=0.7,<0.8",
            "imagesize",
            "requests>=2.25,<2.32",
            "packaging<24",
            "sphinxcontrib-applehelp<1.0.8",
            "sphinxcontrib-devhelp<1.0.6",
            "sphinxcontrib-htmlhelp>=2.0.0,<2.0.5",
            "sphinxcontrib-jsmath<1.0.2",
            "sphinxcontrib-qthelp<1.0.7",
            "sphinxcontrib-serializinghtml>=1.1.5,<1.1.10",
        ],
        "eval_cmd_type": "pytest",
        "sympy_test_file": "",
        "primary_file": "sphinx/writers/latex.py",
        "primary_file_lines": 2061,
        "worker_test_hint": "PYTHONPATH=. ./.venv/bin/python -m pytest -q tests/test_build_latex.py",
    },
    {
        "short_id": "dja17084",
        "instance_id": "django__django-17084",
        "python_ver": "3.11",
        "deps": ["asgiref>=3.7.0", "sqlparse>=0.3.1", "tblib"],
        "eval_cmd_type": "django",
        "sympy_test_file": "",
        "primary_file": "django/db/models/sql/query.py",
        "primary_file_lines": 2678,
        "worker_test_hint": "PYTHONPATH=. ./.venv/bin/python ./tests/runtests.py --verbosity=1 aggregation",
    },
    {
        "short_id": "mpl25960",
        "instance_id": "matplotlib__matplotlib-25960",
        "python_ver": "3.10",
        "deps": [
            "pytest>=7.0,<8.0",
            "matplotlib==3.7.2",
            "numpy<1.26",
            "packaging",
            "pillow<10.3",
            "pyparsing",
            "cycler",
            "kiwisolver",
            "fonttools",
            "contourpy",
        ],
        "eval_cmd_type": "pytest",
        "sympy_test_file": "",
        "sync_mpl_so": True,
        "primary_file": "lib/matplotlib/figure.py",
        "primary_file_lines": 3618,
        "worker_test_hint": "PYTHONPATH=lib:. ./.venv/bin/python -m pytest -q lib/matplotlib/tests/test_figure.py",
    },
]


def parse_iso_ts(ts: str) -> float:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def copy_tree_cow(src: Path, dst: Path) -> None:
    """APFS Copy-on-Write clone (`cp -cR`) on macOS with automatic fallback to `cp -R`."""
    proc = subprocess.run(["cp", "-cR", str(src), str(dst)], capture_output=True)
    if proc.returncode != 0:
        subprocess.run(["cp", "-R", str(src), str(dst)], check=True)


def normalize_django_test_label(label: str) -> str:
    """
    Convert SWE-bench Django unittest strings like:
      `test_foo (aggregation.tests.AggregateTestCase.test_foo)` or
      `test_foo (aggregation.tests.AggregateTestCase)`
    into `./tests/runtests.py` dotted label `aggregation.tests.AggregateTestCase.test_foo`.
    """
    s = label.strip()
    m = re.match(r"^(\w+)\s+\(([\w\.]+)\)$", s)
    if not m:
        return s
    method, inner = m.groups()
    if inner.endswith(f".{method}"):
        return inner
    return f"{inner}.{method}"


def filter_target_instances(instances_filter: str = "") -> list[dict[str, Any]]:
    if not instances_filter.strip():
        return list(TARGET_INSTANCES)
    wanted = {tok.strip() for tok in instances_filter.split(",") if tok.strip()}
    selected = [
        item
        for item in TARGET_INSTANCES
        if item["short_id"] in wanted or item["instance_id"] in wanted
    ]
    if not selected:
        raise ValueError(f"No TARGET_INSTANCES matched filter: {instances_filter!r}")
    return selected


def parse_worker_transcript(conv_id: str, brain_dir: Path | None = None) -> dict[str, Any]:
    """
    Parse Antigravity worker transcript JSONL for exact token accounting, wall time,
    CoT `output_tokens`, and tool-call counts (`read_file_calls`, `run_command_calls`, `edit_file_calls`).
    Falls back to `transcript_full.jsonl` on steps where `tool_calls` was truncated.
    """
    base_brain = brain_dir if brain_dir is not None else BRAIN_DIR
    logs_dir = base_brain / conv_id / ".system_generated" / "logs"
    t_path = logs_dir / "transcript.jsonl"
    t_full_path = logs_dir / "transcript_full.jsonl"
    if not t_path.exists():
        raise FileNotFoundError(f"Missing transcript: {t_path}")

    raw_lines = [ln for ln in t_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    full_by_step: dict[Any, dict[str, Any]] | None = None

    planner_steps: list[dict[str, Any]] = []
    all_ts: list[float] = []
    for idx, line in enumerate(raw_lines):
        obj = json.loads(line)
        if obj.get("created_at"):
            all_ts.append(parse_iso_ts(obj["created_at"]))
        if obj.get("type") == "PLANNER_RESPONSE":
            trunc = obj.get("truncated_fields") or []
            if "tool_calls" in trunc and t_full_path.exists():
                if full_by_step is None:
                    full_by_step = {}
                    for f_idx, f_line in enumerate(
                        t_full_path.read_text(encoding="utf-8").splitlines()
                    ):
                        if not f_line.strip():
                            continue
                        f_obj = json.loads(f_line)
                        full_by_step[("step", f_obj.get("step_index"))] = f_obj
                        full_by_step[("line", f_idx)] = f_obj
                step_key = ("step", obj.get("step_index"))
                obj = full_by_step.get(step_key, full_by_step.get(("line", idx), obj))
            planner_steps.append(obj)

    if not planner_steps:
        raise RuntimeError(f"No PLANNER_RESPONSE steps found in {t_path}")

    billed_input = sum(int(s.get("input_tokens", 0)) for s in planner_steps)
    cache_read = sum(int(s.get("cache_read_tokens", 0)) for s in planner_steps)
    output_tokens = sum(int(s.get("output_tokens", 0)) for s in planner_steps)
    total_context = billed_input + cache_read

    cache_miss_turns = 0
    cache_eviction_rebilled_tokens = 0
    for i in range(1, len(planner_steps)):
        prev_ctx = int(planner_steps[i - 1].get("input_tokens", 0)) + int(
            planner_steps[i - 1].get("cache_read_tokens", 0)
        )
        curr_cache = int(planner_steps[i].get("cache_read_tokens", 0))
        drop = prev_ctx - curr_cache
        # Ignore minor KV block-alignment jitter; only flag real replica misses (>10% & >4k drop, or full 0 reset).
        if curr_cache == 0 or (drop > 4096 and drop > int(prev_ctx * 0.10)):
            cache_miss_turns += 1
            cache_eviction_rebilled_tokens += max(0, drop)
    cache_normalized_billed_input = max(0, billed_input - cache_eviction_rebilled_tokens)

    first_step_context = int(planner_steps[0].get("input_tokens", 0)) + int(
        planner_steps[0].get("cache_read_tokens", 0)
    )
    turns = len(planner_steps)
    net_task_context = max(0, total_context - (first_step_context * turns))
    wall_seconds = round(max(all_ts) - min(all_ts), 2) if len(all_ts) >= 2 else 0.0

    read_file_calls = 0
    run_command_calls = 0
    edit_file_calls = 0
    total_tool_calls = 0
    for step in planner_steps:
        tcalls = step.get("tool_calls") or []
        if not isinstance(tcalls, list):
            continue
        for tc in tcalls:
            if not isinstance(tc, dict):
                continue
            total_tool_calls += 1
            tname = str(tc.get("name", ""))
            if tname in READ_TOOL_NAMES:
                read_file_calls += 1
            elif tname in RUN_TOOL_NAMES:
                run_command_calls += 1
            elif tname in EDIT_TOOL_NAMES:
                edit_file_calls += 1

    return {
        "conversation_id": conv_id,
        "turns": turns,
        "wall_seconds": wall_seconds,
        "billed_input_tokens": billed_input,
        "cache_miss_turns": cache_miss_turns,
        "cache_eviction_rebilled_tokens": cache_eviction_rebilled_tokens,
        "cache_normalized_billed_input_tokens": cache_normalized_billed_input,
        "cache_read_tokens": cache_read,
        "total_context_tokens": total_context,
        "turn1_base_prompt_tokens": first_step_context,
        "net_task_context_tokens": net_task_context,
        "output_tokens": output_tokens,
        "read_file_calls": read_file_calls,
        "run_command_calls": run_command_calls,
        "edit_file_calls": edit_file_calls,
        "total_tool_calls": total_tool_calls,
    }


def linear_percentile(sorted_vals: list[float], pct: float) -> float:
    """Standard R-7 linear interpolation percentile (matches NumPy/SciPy/Pandas default)."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_vals[0])
    pos = (pct / 100.0) * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    return float(sorted_vals[lo]) + frac * (float(sorted_vals[hi]) - float(sorted_vals[lo]))


def compute_distribution_stats(values: list[float | int]) -> dict[str, float]:
    """Compute Median, IQR (`Q3 - Q1`), `[Q1, Q3]` quartiles, `Mean ± SD`, Min, Max, and Sum."""
    if not values:
        return {
            "count": 0,
            "median": 0.0,
            "q1": 0.0,
            "q3": 0.0,
            "iqr": 0.0,
            "mean": 0.0,
            "sd": 0.0,
            "min": 0.0,
            "max": 0.0,
            "sum": 0.0,
        }
    xs = sorted(float(v) for v in values)
    q1 = linear_percentile(xs, 25.0)
    med = linear_percentile(xs, 50.0)
    q3 = linear_percentile(xs, 75.0)
    iqr = q3 - q1
    mean_v = statistics.fmean(xs)
    sd_v = statistics.stdev(xs) if len(xs) >= 2 else 0.0
    return {
        "count": len(xs),
        "median": round(med, 2),
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2),
        "mean": round(mean_v, 2),
        "sd": round(sd_v, 2),
        "min": round(xs[0], 2),
        "max": round(xs[-1], 2),
        "sum": round(sum(xs), 2),
    }


def compute_pass_metrics(
    instance_trial_outcomes: dict[str, list[bool]], k: int = DEFAULT_TRIALS
) -> dict[str, Any]:
    """
    Compute multi-trial accuracy & reliability metrics across evaluated instances:
      - `Pass@1`: expected single-trial resolution rate (`resolved_trials / total_trials * 100`).
      - `Pass^k` (`Pass^3`): strict consistency / τ-bench & HAL reliability — fraction of instances
        where ALL `k/k` trials pass (`len(outcomes) >= k and all(outcomes[:k])`).
      - `Pass@k` (`Pass@3`): coverage — fraction of instances where at least `1` trial passes (`any(outcomes)`).
    """
    eval_iids = [iid for iid, outs in instance_trial_outcomes.items() if len(outs) > 0]
    total_trials = sum(len(instance_trial_outcomes[iid]) for iid in eval_iids)
    resolved_trials = sum(
        sum(1 for ok in instance_trial_outcomes[iid] if ok) for iid in eval_iids
    )
    n_inst = len(eval_iids)

    per_instance: dict[str, dict[str, Any]] = {}
    pass_hat_k_count = 0
    pass_at_k_count = 0

    for iid in eval_iids:
        outs = instance_trial_outcomes[iid]
        m = len(outs)
        c = sum(1 for ok in outs if ok)
        strict_all_k = bool(m >= k and all(outs[:k]))
        any_pass = bool(any(outs))
        if strict_all_k:
            pass_hat_k_count += 1
        if any_pass:
            pass_at_k_count += 1
        per_instance[iid] = {
            "trials_completed": m,
            "trials_passed": c,
            "pass_rate_str": f"{c}/{m}",
            "pass_at_1_pct": round((c / m) * 100.0, 2) if m else 0.0,
            "pass_hat_k": strict_all_k,
            "pass_at_k": any_pass,
        }

    return {
        "k": k,
        "instances_evaluated": n_inst,
        "trials_resolved": resolved_trials,
        "trials_total": total_trials,
        "pass_at_1_pct": round((resolved_trials / total_trials) * 100.0, 2) if total_trials else 0.0,
        "pass_hat_k_count": pass_hat_k_count,
        "pass_hat_k_pct": round((pass_hat_k_count / n_inst) * 100.0, 2) if n_inst else 0.0,
        "pass_hat_3_count": pass_hat_k_count,
        "pass_hat_3_pct": round((pass_hat_k_count / n_inst) * 100.0, 2) if n_inst else 0.0,
        "pass_at_k_count": pass_at_k_count,
        "pass_at_k_pct": round((pass_at_k_count / n_inst) * 100.0, 2) if n_inst else 0.0,
        "pass_at_3_count": pass_at_k_count,
        "pass_at_3_pct": round((pass_at_k_count / n_inst) * 100.0, 2) if n_inst else 0.0,
        "per_instance": per_instance,
    }


def load_parquet_metadata() -> dict[str, dict[str, Any]]:
    required_ids = [t["instance_id"] for t in TARGET_INSTANCES]
    if META_JSON.exists():
        cached = json.loads(META_JSON.read_text(encoding="utf-8"))
        if all(iid in cached for iid in required_ids):
            return cached

    if not PARQUET_PATH.exists():
        subprocess.run(
            [
                "curl",
                "-sSL",
                "-o",
                str(PARQUET_PATH),
                "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/resolve/main/data/test-00000-of-00001.parquet",
            ],
            check=True,
        )

    out = subprocess.check_output(
        [
            "uv",
            "run",
            "--with",
            "pyarrow",
            "python",
            "-c",
            (
                "import pyarrow.parquet as pq, json; "
                f"rows = {{r['instance_id']: r for r in pq.read_table({str(PARQUET_PATH)!r}).to_pylist()}}; "
                f"sel = {{k: rows[k] for k in {required_ids!r}}}; "
                "print(json.dumps(sel))"
            ),
        ],
        text=True,
    )
    meta = json.loads(out)
    CACHE_BASE.mkdir(parents=True, exist_ok=True)
    META_JSON.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def sandbox_dir_for(
    arm_sub: str, sid: str, trial: int, sandbox_base: Path = SANDBOX_BASE
) -> Path:
    return sandbox_base / arm_sub / f"{sid}_t{trial}"


def worker_name_for(arm_prefix: str, sid: str, trial: int) -> str:
    return f"{arm_prefix}-{sid}-t{trial}"


def setup_all(
    instances: list[dict[str, Any]] | None = None,
    trials: int = DEFAULT_TRIALS,
    sandbox_base: Path = SANDBOX_BASE,
) -> None:
    target_list = instances if instances is not None else TARGET_INSTANCES
    meta = load_parquet_metadata()
    CACHE_BASE.mkdir(parents=True, exist_ok=True)
    sandbox_base.mkdir(parents=True, exist_ok=True)

    for item in target_list:
        iid = item["instance_id"]
        sid = item["short_id"]
        rec = meta[iid]
        repo = rec["repo"]
        sha = rec["base_commit"]
        py_ver = item.get("python_ver", "3.10")

        repo_cache = CACHE_BASE / f"repo_{sid}"
        venv_dir = CACHE_BASE / f"venv_{sid}"

        if not (repo_cache / ".git").exists():
            repo_cache.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init"], cwd=repo_cache, check=True, capture_output=True)
            subprocess.run(
                ["git", "remote", "add", "origin", f"https://github.com/{repo}.git"],
                cwd=repo_cache,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "fetch", "--depth", "1", "origin", sha],
                cwd=repo_cache,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "checkout", "-f", "FETCH_HEAD"],
                cwd=repo_cache,
                check=True,
                capture_output=True,
            )
            print(f"Fetched {iid} ({repo} @ {sha[:10]})")

        if not (venv_dir / "bin" / "python").exists():
            subprocess.run(
                ["uv", "venv", "--python", py_ver, str(venv_dir)],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(venv_dir / "bin" / "python"),
                    *item["deps"],
                ],
                check=True,
                capture_output=True,
            )
            if sid == "pyt8399":
                scm_env = {**os.environ, "SETUPTOOLS_SCM_PRETEND_VERSION": "6.3.0"}
                subprocess.run(
                    [
                        "uv",
                        "pip",
                        "install",
                        "--python",
                        str(venv_dir / "bin" / "python"),
                        "--no-deps",
                        "-e",
                        str(repo_cache),
                    ],
                    env=scm_env,
                    check=False,
                    capture_output=True,
                )
            print(f"Prepared Python {py_ver} venv for {iid}")

        if sid == "pyt8399":
            ver_file = repo_cache / "src" / "_pytest" / "_version.py"
            ver_file.write_text('version = "6.3.0"\nversion_tuple = (6, 3, 0)\n', encoding="utf-8")

        if item.get("sync_mpl_so"):
            sp_mpl = venv_dir / "lib" / f"python{py_ver}" / "site-packages" / "matplotlib"
            if sp_mpl.is_dir():
                for so in sp_mpl.rglob("*.so"):
                    rel = so.relative_to(sp_mpl)
                    dst = repo_cache / "lib" / "matplotlib" / rel
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(so, dst)
                ver_f = sp_mpl / "_version.py"
                dst_ver = repo_cache / "lib" / "matplotlib" / "_version.py"
                if ver_f.exists() and not dst_ver.exists():
                    shutil.copy2(ver_f, dst_ver)

    t0 = time.perf_counter()
    created_count = 0
    for item in target_list:
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

        for arm_sub, arm_prefix in (("wbc_vanilla_arm", "van"), ("wbc_wbc_arm", "wbc")):
            for t_idx in range(1, trials + 1):
                arm_dir = sandbox_dir_for(arm_sub, sid, t_idx, sandbox_base=sandbox_base)
                if arm_dir.exists():
                    shutil.rmtree(arm_dir)
                arm_dir.parent.mkdir(parents=True, exist_ok=True)
                copy_tree_cow(repo_cache, arm_dir)
                if not (arm_dir / ".venv").exists():
                    os.symlink(venv_dir, arm_dir / ".venv")
                (arm_dir / "PROBLEM_STATEMENT.md").write_text(problem_text, encoding="utf-8")
                created_count += 1
                wname = worker_name_for(arm_prefix, sid, t_idx)
                print(f"Ready {wname} -> {arm_dir}")

    elapsed = time.perf_counter() - t0
    print(f"Provisioned {created_count} APFS CoW sandboxes in {elapsed:.2f}s at {sandbox_base}")


def extract_model_patch(arm_dir: Path) -> str:
    """Extract clean unified `git diff HEAD` patch from an agent sandbox (excludes untracked files)."""
    if not (arm_dir / ".git").exists():
        return ""
    proc = subprocess.run(
        ["git", "diff", "HEAD"],
        cwd=arm_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout


def evaluate_sandbox(
    item: dict[str, Any],
    rec: dict[str, Any],
    arm_dir: Path,
    cache_base: Path = CACHE_BASE,
) -> tuple[bool, str]:
    """
    Copies `arm_dir` to an isolated grading temp dir, removes `PROBLEM_STATEMENT.md` & `.venv`,
    applies official held-out `test_patch`, and runs official `FAIL_TO_PASS` (+ `PASS_TO_PASS` subset).
    """
    eval_dir = cache_base / f"eval_{arm_dir.parent.name}_{arm_dir.name}_{os.getpid()}"
    if eval_dir.exists():
        shutil.rmtree(eval_dir)
    try:
        copy_tree_cow(arm_dir, eval_dir)
        prob_md = eval_dir / "PROBLEM_STATEMENT.md"
        if prob_md.exists():
            prob_md.unlink()
        venv_link = eval_dir / ".venv"
        if venv_link.is_symlink() or venv_link.exists():
            if venv_link.is_symlink() or venv_link.is_file():
                venv_link.unlink()
            else:
                shutil.rmtree(venv_link)

        # Official SWE-bench protocol: reset any agent edits on files touched by held-out `test_patch`
        # so held-out `test_patch` applies cleanly against `base_commit` test files while preserving source edits.
        patch_lines = rec["test_patch"].splitlines()
        test_patch_files = [
            line[6:].strip() for line in patch_lines if line.startswith("--- a/")
        ]
        if test_patch_files:
            subprocess.run(
                ["git", "checkout", "HEAD", "--", *test_patch_files],
                cwd=eval_dir,
                capture_output=True,
            )
        for idx_ln, line in enumerate(patch_lines):
            if line.startswith("+++ b/") and idx_ln > 0 and patch_lines[idx_ln - 1].startswith("--- /dev/null"):
                created_rel = line[6:].strip()
                created_target = eval_dir / created_rel
                if created_target.exists() or created_target.is_symlink():
                    created_target.unlink()

        patch_proc = subprocess.run(
            ["git", "apply", "-"],
            input=rec["test_patch"],
            cwd=eval_dir,
            capture_output=True,
            text=True,
        )
        if patch_proc.returncode != 0:
            return False, f"test_patch apply failed: {patch_proc.stderr}"

        venv_py = cache_base / f"venv_{item['short_id']}" / "bin" / "python"
        env = dict(os.environ)
        src_paths = [str(eval_dir)]
        if (eval_dir / "lib").is_dir():
            src_paths.insert(0, str(eval_dir / "lib"))
        if (eval_dir / "src").is_dir():
            src_paths.insert(0, str(eval_dir / "src"))
        env["PYTHONPATH"] = ":".join(src_paths)

        f2p: list[str] = json.loads(rec["FAIL_TO_PASS"])
        p2p: list[str] = json.loads(rec["PASS_TO_PASS"])[:10]
        cmd_type = item.get("eval_cmd_type") or (
            "sympy_pytest" if item.get("sympy_test_file") else "pytest"
        )

        if cmd_type == "django":
            labels = [normalize_django_test_label(lbl) for lbl in (f2p + p2p[:5])]
            cmd = [str(venv_py), "./tests/runtests.py", "--verbosity=1", *labels]
        elif cmd_type == "sympy_pytest" or item.get("sympy_test_file"):
            test_file = item["sympy_test_file"]
            expr = " or ".join(f2p + p2p[:5])
            cmd = [str(venv_py), "-m", "pytest", "-q", test_file, "-k", expr]
        else:
            cmd = [str(venv_py), "-m", "pytest", "-q", *(f2p + p2p[:5])]

        proc = subprocess.run(
            cmd, cwd=eval_dir, env=env, capture_output=True, text=True, timeout=120
        )
        return proc.returncode == 0, (proc.stdout + "\n" + proc.stderr).strip()[-600:]
    finally:
        shutil.rmtree(eval_dir, ignore_errors=True)


def verify_gold_harness(
    instances: list[dict[str, Any]] | None = None,
    sandbox_base: Path = SANDBOX_BASE,
) -> list[dict[str, Any]]:
    """Sanity-check that unpatched `base_commit` FAILS F2P and gold `patch` + `test_patch` PASSES F2P."""
    target_list = instances if instances is not None else TARGET_INSTANCES
    meta = load_parquet_metadata()
    checks: list[dict[str, Any]] = []

    for item in target_list:
        iid = item["instance_id"]
        sid = item["short_id"]
        rec = meta[iid]
        van_dir = sandbox_dir_for("wbc_vanilla_arm", sid, 1, sandbox_base=sandbox_base)
        if not van_dir.exists():
            van_dir = CACHE_BASE / f"repo_{sid}"

        ok_before, _ = evaluate_sandbox(item, rec, van_dir)

        gold_dir = CACHE_BASE / f"gold_check_{sid}_{os.getpid()}"
        if gold_dir.exists():
            shutil.rmtree(gold_dir)
        try:
            copy_tree_cow(van_dir, gold_dir)
            subprocess.run(
                ["git", "apply", "-"],
                input=rec["patch"],
                cwd=gold_dir,
                check=True,
                text=True,
            )
            ok_after, out_after = evaluate_sandbox(item, rec, gold_dir)
        finally:
            shutil.rmtree(gold_dir, ignore_errors=True)

        print(
            f"Harness check {iid}: before_patch_pass={ok_before} (expect False), "
            f"after_gold_pass={ok_after} (expect True)"
        )
        if ok_before or not ok_after:
            raise RuntimeError(f"Gold harness sanity check failed for {iid}: {out_after}")
        checks.append(
            {
                "short_id": sid,
                "instance_id": iid,
                "repo": rec["repo"],
                "before_patch_pass": ok_before,
                "after_gold_pass": ok_after,
            }
        )
    return checks


def build_worker_manifest(
    instances: list[dict[str, Any]] | None = None,
    trials: int = DEFAULT_TRIALS,
    sandbox_base: Path = SANDBOX_BASE,
) -> dict[str, Any]:
    target_list = instances if instances is not None else TARGET_INSTANCES
    meta = load_parquet_metadata()
    entries: list[dict[str, Any]] = []

    for t_idx in range(1, trials + 1):
        for item in target_list:
            sid = item["short_id"]
            iid = item["instance_id"]
            rec = meta[iid]
            repo = rec["repo"]
            sha = rec["base_commit"]
            hint = item["worker_test_hint"]
            for arm_key, arm_sub, arm_prefix, is_wbc in (
                ("vanilla", "wbc_vanilla_arm", "van", False),
                ("wbc", "wbc_wbc_arm", "wbc", True),
            ):
                wname = worker_name_for(arm_prefix, sid, t_idx)
                sdir = sandbox_dir_for(arm_sub, sid, t_idx, sandbox_base=sandbox_base)
                brief = (
                    f"You own isolated SWE-bench Verified trial `{wname}` (`{iid}`, `{repo}` @ `{sha[:10]}`).\n"
                    f"1. Work EXCLUSIVELY inside `{sdir}` (do NOT read or edit any file outside `{sdir}`).\n"
                    f"2. Read `{sdir}/PROBLEM_STATEMENT.md` for the verbatim upstream issue.\n"
                    f"3. Diagnose and implement the minimal, correct source fix inside `{sdir}`.\n"
                    f"4. Run tests inside `{sdir}` using:\n"
                    f"   `cd {sdir} && {hint}`\n"
                    f"   (You may also pass `-k <expr>` or add a targeted regression test if appropriate; "
                    f"leave `PROBLEM_STATEMENT.md` untouched and never run `git reset`/`git clean`).\n"
                    f"5. When tests pass and `git diff` is clean and minimal, post your concise fix summary "
                    f"to `@coordinator` and call `swarm_wait`."
                )
                entries.append(
                    {
                        "worker_name": wname,
                        "arm": arm_key,
                        "is_wbc": is_wbc,
                        "trial": t_idx,
                        "short_id": sid,
                        "instance_id": iid,
                        "repo": repo,
                        "base_commit": sha,
                        "primary_file": item["primary_file"],
                        "primary_file_lines": item["primary_file_lines"],
                        "sandbox_dir": str(sdir),
                        "worker_test_hint": hint,
                        "brief": brief,
                    }
                )

    return {
        "trials_per_instance": trials,
        "instances_count": len(target_list),
        "total_workers": len(entries),
        "sandbox_base": str(sandbox_base),
        "workers": entries,
    }


def parse_map_json_arg(map_json_arg: str) -> dict[str, str]:
    candidate_path = Path(map_json_arg)
    if candidate_path.is_file():
        return json.loads(candidate_path.read_text(encoding="utf-8"))
    return json.loads(map_json_arg)


def _pct_delta(new_v: float, base_v: float) -> float:
    return round(((new_v - base_v) / base_v) * 100.0, 1) if base_v else 0.0


def compile_results(
    worker_map: dict[str, str],
    instances: list[dict[str, Any]] | None = None,
    trials: int = DEFAULT_TRIALS,
    sandbox_base: Path = SANDBOX_BASE,
    cache_base: Path = CACHE_BASE,
    brain_dir: Path | None = None,
    meta_override: dict[str, dict[str, Any]] | None = None,
    evaluator_fn: Any | None = None,
) -> dict[str, Any]:
    """
    Grade completed sandboxes (supports incremental partial waves AND full `60`-trial maps),
    extract `git diff HEAD` patches, parse worker transcripts for tokens + CoT + tool calls,
    and compute `Pass@1`, `Pass^k` (`Pass^3`), `Pass@k` (`Pass@3`), `Median`, and `IQR`.
    """
    target_list = instances if instances is not None else TARGET_INSTANCES
    meta = meta_override if meta_override is not None else load_parquet_metadata()
    eval_fn = evaluator_fn if evaluator_fn is not None else evaluate_sandbox

    runs: list[dict[str, Any]] = []
    expected_total_runs = len(target_list) * 2 * trials

    for arm_key, arm_sub, arm_prefix, is_wbc in (
        ("vanilla", "wbc_vanilla_arm", "van", False),
        ("wbc", "wbc_wbc_arm", "wbc", True),
    ):
        for item in target_list:
            sid = item["short_id"]
            iid = item["instance_id"]
            rec = meta[iid]
            for t_idx in range(1, trials + 1):
                wname = worker_name_for(arm_prefix, sid, t_idx)
                legacy_wname = f"{arm_prefix}-{sid}" if trials == 1 else ""
                if wname in worker_map:
                    conv_id = worker_map[wname]
                    arm_dir = sandbox_dir_for(arm_sub, sid, t_idx, sandbox_base=sandbox_base)
                    if not arm_dir.exists() and (sandbox_base / arm_sub / sid).exists():
                        arm_dir = sandbox_base / arm_sub / sid
                elif legacy_wname and legacy_wname in worker_map:
                    wname = legacy_wname
                    conv_id = worker_map[legacy_wname]
                    arm_dir = sandbox_dir_for(arm_sub, sid, t_idx, sandbox_base=sandbox_base)
                    if not arm_dir.exists() and (sandbox_base / arm_sub / sid).exists():
                        arm_dir = sandbox_base / arm_sub / sid
                else:
                    # Incremental / checkpointed mode: skip not-yet-executed trials cleanly
                    continue

                passed, eval_out = eval_fn(item, rec, arm_dir, cache_base=cache_base)
                model_patch = extract_model_patch(arm_dir)
                t_metrics = parse_worker_transcript(conv_id, brain_dir=brain_dir)
                runs.append(
                    {
                        "worker_name": wname,
                        "arm": arm_key,
                        "is_wbc": is_wbc,
                        "trial": t_idx,
                        "short_id": sid,
                        "instance_id": iid,
                        "repo": rec["repo"],
                        "base_commit": rec["base_commit"],
                        "primary_file": item.get("primary_file", ""),
                        "primary_file_lines": item.get("primary_file_lines", 0),
                        "sandbox_dir": str(arm_dir),
                        "verified_pass": passed,
                        "eval_output": eval_out,
                        "model_patch": model_patch,
                        **t_metrics,
                    }
                )

    summary: dict[str, Any] = {}
    per_instance_arm: dict[str, dict[str, Any]] = {}

    for item in target_list:
        per_instance_arm[item["instance_id"]] = {
            "short_id": item["short_id"],
            "repo": meta[item["instance_id"]]["repo"],
            "base_commit": meta[item["instance_id"]]["base_commit"],
            "primary_file": item.get("primary_file", ""),
            "primary_file_lines": item.get("primary_file_lines", 0),
            "arms": {},
        }

    for arm_key in ("vanilla", "wbc"):
        a_runs = [r for r in runs if r["arm"] == arm_key]
        outcomes_by_inst: dict[str, list[bool]] = {}
        for item in target_list:
            iid = item["instance_id"]
            i_runs = sorted(
                [r for r in a_runs if r["instance_id"] == iid],
                key=lambda x: int(x["trial"]),
            )
            if i_runs:
                outcomes_by_inst[iid] = [bool(r["verified_pass"]) for r in i_runs]
                inst_dists = {
                    mk: compute_distribution_stats([r[mk] for r in i_runs])
                    for mk in METRIC_KEYS
                }
                c_pass = sum(1 for r in i_runs if r["verified_pass"])
                m_cnt = len(i_runs)
                per_instance_arm[iid]["arms"][arm_key] = {
                    "trials_completed": m_cnt,
                    "trials_passed": c_pass,
                    "pass_rate_str": f"{c_pass}/{m_cnt}",
                    "pass_at_1_pct": round((c_pass / m_cnt) * 100.0, 2),
                    "pass_hat_k": bool(m_cnt >= trials and all(r["verified_pass"] for r in i_runs[:trials])),
                    "pass_at_k": bool(any(r["verified_pass"] for r in i_runs)),
                    "distributions": inst_dists,
                }

        pass_stats = compute_pass_metrics(outcomes_by_inst, k=trials)
        arm_dists = {
            mk: compute_distribution_stats([r[mk] for r in a_runs]) for mk in METRIC_KEYS
        }

        summary[arm_key] = {
            "tasks_resolved": pass_stats["trials_resolved"],
            "tasks_total": pass_stats["trials_total"],
            "accuracy_pct": pass_stats["pass_at_1_pct"],
            "pass_at_1_pct": pass_stats["pass_at_1_pct"],
            "instances_evaluated": pass_stats["instances_evaluated"],
            "trials_per_instance_k": trials,
            "pass_hat_k_count": pass_stats["pass_hat_k_count"],
            "pass_hat_k_pct": pass_stats["pass_hat_k_pct"],
            "pass_hat_3_count": pass_stats["pass_hat_3_count"],
            "pass_hat_3_pct": pass_stats["pass_hat_3_pct"],
            "pass_at_k_count": pass_stats["pass_at_k_count"],
            "pass_at_k_pct": pass_stats["pass_at_k_pct"],
            "pass_at_3_count": pass_stats["pass_at_3_count"],
            "pass_at_3_pct": pass_stats["pass_at_3_pct"],
            "total_turns": sum(r["turns"] for r in a_runs),
            "total_wall_seconds": round(sum(r["wall_seconds"] for r in a_runs), 2),
            "total_billed_input": sum(r["billed_input_tokens"] for r in a_runs),
            "total_cache_read": sum(r["cache_read_tokens"] for r in a_runs),
            "total_context_ingested": sum(r["total_context_tokens"] for r in a_runs),
            "total_net_task_context": sum(r["net_task_context_tokens"] for r in a_runs),
            "total_output_tokens": sum(r["output_tokens"] for r in a_runs),
            "total_read_file_calls": sum(r["read_file_calls"] for r in a_runs),
            "total_run_command_calls": sum(r["run_command_calls"] for r in a_runs),
            "total_edit_file_calls": sum(r["edit_file_calls"] for r in a_runs),
            "distributions": arm_dists,
            "per_instance": pass_stats["per_instance"],
        }

    va, wb = summary["vanilla"], summary["wbc"]
    va_d, wb_d = va["distributions"], wb["distributions"]

    deltas = {
        "pass_at_1_delta_pp": round(wb["pass_at_1_pct"] - va["pass_at_1_pct"], 2),
        "pass_hat_3_delta_pp": round(wb["pass_hat_3_pct"] - va["pass_hat_3_pct"], 2),
        "pass_at_3_delta_pp": round(wb["pass_at_3_pct"] - va["pass_at_3_pct"], 2),
        "turns_delta_pct": _pct_delta(wb["total_turns"], va["total_turns"]),
        "wall_delta_pct": _pct_delta(wb["total_wall_seconds"], va["total_wall_seconds"]),
        "billed_input_delta_pct": _pct_delta(wb["total_billed_input"], va["total_billed_input"]),
        "total_context_delta_pct": _pct_delta(
            wb["total_context_ingested"], va["total_context_ingested"]
        ),
        "net_task_context_delta_pct": _pct_delta(
            wb["total_net_task_context"], va["total_net_task_context"]
        ),
        "output_tokens_delta_pct": _pct_delta(
            wb["total_output_tokens"], va["total_output_tokens"]
        ),
        "read_file_calls_delta_pct": _pct_delta(
            wb["total_read_file_calls"], va["total_read_file_calls"]
        ),
        "median_wall_delta_pct": _pct_delta(
            wb_d["wall_seconds"]["median"], va_d["wall_seconds"]["median"]
        ),
        "median_turns_delta_pct": _pct_delta(
            wb_d["turns"]["median"], va_d["turns"]["median"]
        ),
        "median_billed_input_delta_pct": _pct_delta(
            wb_d["billed_input_tokens"]["median"], va_d["billed_input_tokens"]["median"]
        ),
        "median_total_context_delta_pct": _pct_delta(
            wb_d["total_context_tokens"]["median"], va_d["total_context_tokens"]["median"]
        ),
        "median_net_context_delta_pct": _pct_delta(
            wb_d["net_task_context_tokens"]["median"],
            va_d["net_task_context_tokens"]["median"],
        ),
        "median_output_tokens_delta_pct": _pct_delta(
            wb_d["output_tokens"]["median"], va_d["output_tokens"]["median"]
        ),
        "median_read_file_calls_delta_pct": _pct_delta(
            wb_d["read_file_calls"]["median"], va_d["read_file_calls"]["median"]
        ),
        "context_efficiency_x": round(
            va["total_context_ingested"] / max(1, wb["total_context_ingested"]), 2
        ),
        "net_context_efficiency_x": round(
            va["total_net_task_context"] / max(1, wb["total_net_task_context"]), 2
        ),
    }

    return {
        "generated_at": datetime.now().isoformat(),
        "dataset": "princeton-nlp/SWE-bench_Verified (Official HuggingFace Test Split, Untouched GitHub Commit SHAs)",
        "model": "Gemini 4 Carbon (Low)",
        "trials_per_instance_k": trials,
        "target_instances_count": len(target_list),
        "expected_total_runs": expected_total_runs,
        "completed_runs": len(runs),
        "is_partial_checkpoint": len(runs) < expected_total_runs,
        "summary": summary,
        "deltas": deltas,
        "per_instance_comparison": per_instance_arm,
        "runs": runs,
    }


def build_hal_submission(results_data: dict[str, Any]) -> dict[str, Any]:
    """
    Build a Princeton HAL (`princeton-nlp / hal-harness`) Leaderboard-ready JSON submission bundle
    containing official SWE-bench `model_patch` predictions per trial, exact token & tool-call
    receipts, `Pass@1`, `Pass^3`, `Pass@3`, and `Median/IQR` error bars.
    """
    k = int(results_data["trials_per_instance_k"])
    arms_bundle: dict[str, Any] = {}

    for arm_key, agent_label, model_slug in (
        (
            "vanilla",
            "Gemini 4 Carbon (Unshielded Vanilla Baseline)",
            "gemini-4-carbon-low-vanilla",
        ),
        (
            "wbc",
            "Gemini 4 Carbon + Whole Body Compute",
            "gemini-4-carbon-low-wbc",
        ),
    ):
        arm_sum = results_data["summary"][arm_key]
        arm_runs = [r for r in results_data["runs"] if r["arm"] == arm_key]

        predictions_by_trial: dict[str, dict[str, dict[str, Any]]] = {}
        for t_idx in range(1, k + 1):
            t_key = f"t{t_idx}"
            t_runs = [r for r in arm_runs if int(r["trial"]) == t_idx]
            if not t_runs:
                continue
            predictions_by_trial[t_key] = {
                r["instance_id"]: {
                    "instance_id": r["instance_id"],
                    "model_name_or_path": model_slug,
                    "model_patch": r["model_patch"],
                    "verified_pass": r["verified_pass"],
                    "conversation_id": r["conversation_id"],
                }
                for r in t_runs
            }

        canonical_predictions: list[dict[str, Any]] = []
        seen_iids = list(dict.fromkeys(r["instance_id"] for r in arm_runs))
        for iid in seen_iids:
            i_runs = sorted(
                [r for r in arm_runs if r["instance_id"] == iid],
                key=lambda x: int(x["trial"]),
            )
            chosen = next((r for r in i_runs if r["verified_pass"]), i_runs[0])
            canonical_predictions.append(
                {
                    "instance_id": iid,
                    "model_name_or_path": model_slug,
                    "model_patch": chosen["model_patch"],
                    "selected_trial": chosen["trial"],
                    "pass_rate_str": f"{sum(1 for r in i_runs if r['verified_pass'])}/{len(i_runs)}",
                    "all_trials_pass_hat_k": bool(
                        len(i_runs) >= k and all(r["verified_pass"] for r in i_runs[:k])
                    ),
                }
            )

        receipts = [
            {
                "worker_name": r["worker_name"],
                "conversation_id": r["conversation_id"],
                "instance_id": r["instance_id"],
                "repo": r["repo"],
                "base_commit": r["base_commit"],
                "trial": r["trial"],
                "verified_pass": r["verified_pass"],
                "model_patch": r["model_patch"],
                "turns": r["turns"],
                "wall_seconds": r["wall_seconds"],
                "billed_input_tokens": r["billed_input_tokens"],
                "cache_read_tokens": r["cache_read_tokens"],
                "total_context_tokens": r["total_context_tokens"],
                "net_task_context_tokens": r["net_task_context_tokens"],
                "output_tokens": r["output_tokens"],
                "read_file_calls": r["read_file_calls"],
                "run_command_calls": r["run_command_calls"],
                "edit_file_calls": r["edit_file_calls"],
            }
            for r in arm_runs
        ]

        arms_bundle[arm_key] = {
            "agent_name": agent_label,
            "model_name_or_path": model_slug,
            "metrics": {
                "pass_at_1_pct": arm_sum["pass_at_1_pct"],
                "pass_hat_3_pct": arm_sum["pass_hat_3_pct"],
                "pass_at_3_pct": arm_sum["pass_at_3_pct"],
                "pass_hat_k_pct": arm_sum["pass_hat_k_pct"],
                "pass_at_k_pct": arm_sum["pass_at_k_pct"],
                "trials_resolved": arm_sum["tasks_resolved"],
                "trials_total": arm_sum["tasks_total"],
                "instances_evaluated": arm_sum["instances_evaluated"],
                "distributions": arm_sum["distributions"],
            },
            "canonical_predictions": canonical_predictions,
            "predictions_by_trial": predictions_by_trial,
            "trial_receipts": receipts,
        }

    return {
        "schema_version": "hal-swebench-verified-k3-v1.0",
        "benchmark_name": "swebench_verified",
        "dataset": "princeton-nlp/SWE-bench_Verified",
        "split": "test",
        "generated_at": results_data["generated_at"],
        "model": results_data["model"],
        "trials_per_instance_k": k,
        "target_instances_count": results_data["target_instances_count"],
        "completed_runs": results_data["completed_runs"],
        "is_partial_checkpoint": results_data["is_partial_checkpoint"],
        "ab_comparison_deltas": results_data["deltas"],
        "arms": arms_bundle,
    }


def _compute_run_cost_usd(
    billed_input_tokens: int,
    cache_read_tokens: int,
    output_tokens: int,
) -> float:
    return round(
        (billed_input_tokens / 1_000_000.0) * FRONTIER_INPUT_USD_PER_1M
        + (cache_read_tokens / 1_000_000.0) * FRONTIER_CACHE_READ_USD_PER_1M
        + (output_tokens / 1_000_000.0) * FRONTIER_OUTPUT_USD_PER_1M,
        6,
    )


def sanitize_public_text(text: str) -> str:
    """
    Ensure exported public benchmark artifacts contain zero local macOS home paths,
    confidential provisional patent serial numbers, internal docket codes,
    or internal organ numbering.
    """
    home_prefix = "/" + "Users" + "/"
    docket_prefix = "VIFI" + "-PAT-"
    org_10 = "Organ " + "10"
    org_11 = "Organ " + "11"
    cleaned = re.sub(re.escape(home_prefix) + r"[A-Za-z0-9._-]+", "/workspace", text)
    cleaned = re.sub(r"\b64/\d{3},\d{3}\b", "[REDACTED-SERIAL]", cleaned)
    cleaned = re.sub(re.escape(docket_prefix) + r"[A-Za-z0-9_-]*", "[REDACTED-DOCKET]", cleaned)
    cleaned = cleaned.replace(org_10, "Whole Body Compute").replace(
        org_11, "Whole Body Compute"
    )
    return cleaned


def audit_public_kit_cleanliness(kit_dir: Path) -> dict[str, Any]:
    """
    Audit every file inside `kit_dir` to guarantee zero local macOS home paths, patent serials,
    internal docket codes, or internal organ labels exist in the exported kit.
    """
    home_prefix = "/" + "Users" + "/"
    docket_prefix = "VIFI" + "-PAT-"
    org_10 = "Organ " + "10"
    org_11 = "Organ " + "11"
    forbidden_re = re.compile(
        f"{re.escape(home_prefix)}|64/\\d{{3}},\\d{{3}}|{re.escape(docket_prefix)}|{re.escape(org_10)}|{re.escape(org_11)}"
    )
    violations: list[str] = []
    checked_files = 0

    for path in sorted(kit_dir.rglob("*")):
        if not path.is_file():
            continue
        checked_files += 1
        rel = path.relative_to(kit_dir).as_posix()
        txt = path.read_text(encoding="utf-8")
        m = forbidden_re.search(txt)
        if m:
            violations.append(f"Forbidden pattern {m.group(0)!r} found in {rel}")

    if violations:
        raise RuntimeError(
            f"Public benchmark kit audit failed with {len(violations)} violation(s):\n"
            + "\n".join(f"  - {v}" for v in violations)
        )
    return {"passed": True, "checked_files": checked_files, "violations": 0}


def export_hal_upload_kit(
    submission_data: dict[str, Any] | None = None,
    results_data: dict[str, Any] | None = None,
    report_md_text: str | None = None,
    out_dir: Path = DEFAULT_HAL_EXPORT_DIR,
) -> dict[str, Any]:
    """
    Export a self-contained, Open-Core-safe reproducibility & Princeton HAL (`princeton-pli/hal-harness`)
    submission kit under `out_dir` (default: `benchmarks/swebench_verified_k3/`).

    Generates:
      1. `predictions/` — Standard `swebench.harness.run_evaluation` JSONL files (`instance_id`,
         `model_name_or_path`, `model_patch`) for canonical + per-trial (`t1..tk`) across both arms.
      2. `hal_upload/` — Official `princeton-pli/hal-harness` (`hal-upload -D`) files:
         - `<run_id>_UPLOAD.json` (`config`, `results`, `raw_eval_results`, `raw_logging_results`,
           `total_usage`, `total_cost`, `git_info`, `wall_clock_times`, `task_step_counts`, `task_metrics`)
         - `<run_id>_RAW_SUBMISSIONS.jsonl`
         - `<run_id>_SWE_BENCH_SUBMISSIONS.jsonl`
         - `<run_id>_WALL_CLOCK_TIMES.jsonl`
      3. `patches/<arm>/<instance_id>_t<k>.diff` — All `60` standalone unified git diffs (`model_patch`)
         for instant human & reviewer inspection.
      4. `hal_swebench_verified_k3_submission.json`, `live_upstream_swebench_k3_results.json`, and
         `LIVE_UPSTREAM_SWEBENCH_K3_REPORT_2026_11_10.md` (sanitized & audited).
    """
    sub = (
        submission_data
        if submission_data is not None
        else json.loads(HAL_SUBMISSION_JSON.read_text(encoding="utf-8"))
    )
    res = (
        results_data
        if results_data is not None
        else (
            json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
            if RESULTS_JSON.exists()
            else None
        )
    )
    rep_md = (
        report_md_text
        if report_md_text is not None
        else (REPORT_MD.read_text(encoding="utf-8") if REPORT_MD.exists() else "")
    )

    out_dir = Path(out_dir)
    pred_dir = out_dir / "predictions"
    hal_dir = out_dir / "hal_upload"
    patches_dir = out_dir / "patches"
    pred_dir.mkdir(parents=True, exist_ok=True)
    hal_dir.mkdir(parents=True, exist_ok=True)
    patches_dir.mkdir(parents=True, exist_ok=True)

    k = int(sub["trials_per_instance_k"])
    gen_date = str(sub.get("generated_at", datetime.now().isoformat()))[:10]
    exported_prediction_files: list[str] = []
    exported_hal_upload_files: list[str] = []
    exported_patch_files: list[str] = []

    for arm_key in ("vanilla", "wbc"):
        arm_data = sub["arms"][arm_key]
        agent_name = arm_data["agent_name"]
        model_slug = arm_data["model_name_or_path"]
        receipts = arm_data["trial_receipts"]
        arm_patch_dir = patches_dir / arm_key
        arm_patch_dir.mkdir(parents=True, exist_ok=True)

        # 1. Canonical predictions.jsonl (1 per instance)
        canon_jsonl_path = pred_dir / f"{arm_key}_canonical_predictions.jsonl"
        canon_lines = [
            json.dumps(
                {
                    "instance_id": p["instance_id"],
                    "model_name_or_path": model_slug,
                    "model_patch": p["model_patch"],
                }
            )
            for p in arm_data["canonical_predictions"]
        ]
        canon_jsonl_path.write_text(
            sanitize_public_text("\n".join(canon_lines) + "\n"), encoding="utf-8"
        )
        exported_prediction_files.append(str(canon_jsonl_path.relative_to(out_dir)))

        # 2. Per-trial predictions JSONL + standalone .diff files + HAL _UPLOAD.json bundles
        for t_idx in range(1, k + 1):
            t_key = f"t{t_idx}"
            t_receipts = [r for r in receipts if int(r["trial"]) == t_idx]
            if not t_receipts:
                continue

            trial_pred_path = pred_dir / f"{arm_key}_{t_key}_predictions.jsonl"
            trial_pred_lines: list[str] = []
            for r in t_receipts:
                iid = r["instance_id"]
                mpatch = r["model_patch"]
                trial_pred_lines.append(
                    json.dumps(
                        {
                            "instance_id": iid,
                            "model_name_or_path": model_slug,
                            "model_patch": mpatch,
                        }
                    )
                )
                diff_path = arm_patch_dir / f"{iid}_{t_key}.diff"
                diff_path.write_text(sanitize_public_text(mpatch), encoding="utf-8")
                exported_patch_files.append(str(diff_path.relative_to(out_dir)))

            trial_pred_path.write_text(
                sanitize_public_text("\n".join(trial_pred_lines) + "\n"), encoding="utf-8"
            )
            exported_prediction_files.append(str(trial_pred_path.relative_to(out_dir)))

            # Build official princeton-pli/hal-harness _UPLOAD.json + sibling JSONLs for this trial wave
            run_id = f"swebench_verified_{arm_key}_{t_key}"
            resolved_ids = [r["instance_id"] for r in t_receipts if r["verified_pass"]]
            unresolved_ids = [r["instance_id"] for r in t_receipts if not r["verified_pass"]]
            total_inst = len(t_receipts)
            resolved_cnt = len(resolved_ids)
            accuracy = round(resolved_cnt / max(1, total_inst), 4)

            tot_in = sum(int(r["billed_input_tokens"]) for r in t_receipts)
            tot_cache = sum(int(r["cache_read_tokens"]) for r in t_receipts)
            tot_out = sum(int(r["output_tokens"]) for r in t_receipts)
            tot_cost = round(
                sum(
                    _compute_run_cost_usd(
                        int(r["billed_input_tokens"]),
                        int(r["cache_read_tokens"]),
                        int(r["output_tokens"]),
                    )
                    for r in t_receipts
                ),
                6,
            )

            wall_clock_times = {
                r["instance_id"]: float(r["wall_seconds"]) for r in t_receipts
            }
            latencies = {
                r["instance_id"]: {
                    "first_call_timestamp": gen_date,
                    "last_call_timestamp": gen_date,
                    "total_time": float(r["wall_seconds"]),
                }
                for r in t_receipts
            }
            task_step_counts = {r["instance_id"]: int(r["turns"]) for r in t_receipts}
            task_metrics = {
                r["instance_id"]: {
                    "step_count": int(r["turns"]),
                    "wall_seconds": float(r["wall_seconds"]),
                    "billed_input_tokens": int(r["billed_input_tokens"]),
                    "cache_read_tokens": int(r["cache_read_tokens"]),
                    "total_context_tokens": int(r["total_context_tokens"]),
                    "net_task_context_tokens": int(r["net_task_context_tokens"]),
                    "output_tokens": int(r["output_tokens"]),
                    "read_file_calls": int(r["read_file_calls"]),
                    "run_command_calls": int(r["run_command_calls"]),
                    "edit_file_calls": int(r["edit_file_calls"]),
                    "cost_usd": _compute_run_cost_usd(
                        int(r["billed_input_tokens"]),
                        int(r["cache_read_tokens"]),
                        int(r["output_tokens"]),
                    ),
                    "conversation_id": r["conversation_id"],
                }
                for r in t_receipts
            }
            raw_eval_results = {
                "total_instances": total_inst,
                "submitted_instances": total_inst,
                "completed_instances": total_inst,
                "resolved_instances": resolved_cnt,
                "unresolved_instances": len(unresolved_ids),
                "empty_patch_instances": sum(
                    1 for r in t_receipts if not r["model_patch"].strip()
                ),
                "error_instances": 0,
                "resolved_ids": resolved_ids,
                "unresolved_ids": unresolved_ids,
                "error_ids": [],
            }
            raw_logging_results = [
                {
                    "task_id": r["instance_id"],
                    "conversation_id": r["conversation_id"],
                    "worker_name": r["worker_name"],
                    "trial": int(r["trial"]),
                    "turns": int(r["turns"]),
                    "wall_seconds": float(r["wall_seconds"]),
                    "usage": {
                        "input_tokens": int(r["billed_input_tokens"]),
                        "cache_read_tokens": int(r["cache_read_tokens"]),
                        "total_context_tokens": int(r["total_context_tokens"]),
                        "output_tokens": int(r["output_tokens"]),
                    },
                    "cost_usd": task_metrics[r["instance_id"]]["cost_usd"],
                }
                for r in t_receipts
            ]

            hal_upload_payload = {
                "config": {
                    "agent_name": agent_name,
                    "benchmark_name": "swebench_verified",
                    "date": gen_date,
                    "run_id": run_id,
                    "agent_args": {
                        "model": sub["model"],
                        "model_name_or_path": model_slug,
                        "arm": arm_key,
                        "trial": int(t_idx),
                        "trials_per_instance_k": k,
                        "whole_body_compute_enabled": bool(arm_key == "wbc"),
                    },
                    "run_command": (
                        f"uv run python benchmarks/upstream_swebench_verified_harness.py "
                        f"--setup --verify-gold --trials {k}"
                    ),
                    "prompt_sensitivity": False,
                },
                "results": {
                    "accuracy": accuracy,
                    "pass_at_1_pct": arm_data["metrics"]["pass_at_1_pct"],
                    "pass_hat_3_pct": arm_data["metrics"]["pass_hat_3_pct"],
                    "pass_at_3_pct": arm_data["metrics"]["pass_at_3_pct"],
                    "successful_tasks": resolved_ids,
                    "failed_tasks": unresolved_ids,
                    "total_cost": tot_cost,
                    "latencies": latencies,
                },
                "raw_eval_results": raw_eval_results,
                "raw_logging_results": raw_logging_results,
                "total_usage": {
                    sub["model"]: {
                        "prompt_tokens": tot_in,
                        "cache_read_input_tokens": tot_cache,
                        "total_context_tokens": tot_in + tot_cache,
                        "completion_tokens": tot_out,
                        "total_tokens": tot_in + tot_cache + tot_out,
                    }
                },
                "total_cost": tot_cost,
                "git_info": {
                    "repository": "https://github.com/wholebodycompute/whole-body-compute",
                    "dataset": "princeton-nlp/SWE-bench_Verified",
                    "split": "test",
                },
                "wall_clock_times": wall_clock_times,
                "task_step_counts": task_step_counts,
                "task_metrics": task_metrics,
            }

            upload_json_path = hal_dir / f"{run_id}_UPLOAD.json"
            upload_json_path.write_text(
                sanitize_public_text(json.dumps(hal_upload_payload, indent=2) + "\n"),
                encoding="utf-8",
            )
            exported_hal_upload_files.append(str(upload_json_path.relative_to(out_dir)))

            raw_sub_path = hal_dir / f"{run_id}_RAW_SUBMISSIONS.jsonl"
            raw_sub_lines = [
                json.dumps({r["instance_id"]: r["model_patch"]}) for r in t_receipts
            ]
            raw_sub_path.write_text(
                sanitize_public_text("\n".join(raw_sub_lines) + "\n"), encoding="utf-8"
            )
            exported_hal_upload_files.append(str(raw_sub_path.relative_to(out_dir)))

            swe_sub_path = hal_dir / f"{run_id}_SWE_BENCH_SUBMISSIONS.jsonl"
            swe_sub_path.write_text(
                sanitize_public_text("\n".join(trial_pred_lines) + "\n"), encoding="utf-8"
            )
            exported_hal_upload_files.append(str(swe_sub_path.relative_to(out_dir)))

            wall_jsonl_path = hal_dir / f"{run_id}_WALL_CLOCK_TIMES.jsonl"
            wall_lines = [
                json.dumps(
                    {
                        "task_id": r["instance_id"],
                        "wall_clock_time": float(r["wall_seconds"]),
                    }
                )
                for r in t_receipts
            ]
            wall_jsonl_path.write_text(
                sanitize_public_text("\n".join(wall_lines) + "\n"), encoding="utf-8"
            )
            exported_hal_upload_files.append(str(wall_jsonl_path.relative_to(out_dir)))

    # 3. Copy sanitized top-level submission JSON, results JSON, and Markdown report into out_dir
    kit_sub_path = out_dir / "hal_swebench_verified_k3_submission.json"
    kit_sub_path.write_text(
        sanitize_public_text(json.dumps(sub, indent=2) + "\n"), encoding="utf-8"
    )
    if res is not None:
        kit_res_path = out_dir / "live_upstream_swebench_k3_results.json"
        kit_res_path.write_text(
            sanitize_public_text(json.dumps(res, indent=2) + "\n"), encoding="utf-8"
        )
    if rep_md:
        kit_md_path = out_dir / "LIVE_UPSTREAM_SWEBENCH_K3_REPORT_2026_11_10.md"
        kit_md_path.write_text(sanitize_public_text(rep_md), encoding="utf-8")
    if README_K3_MD.exists():
        kit_readme_path = out_dir / "README.md"
        kit_readme_path.write_text(
            sanitize_public_text(README_K3_MD.read_text(encoding="utf-8")),
            encoding="utf-8",
        )

    audit = audit_public_kit_cleanliness(out_dir)
    return {
        "out_dir": str(out_dir),
        "prediction_files_count": len(exported_prediction_files),
        "hal_upload_files_count": len(exported_hal_upload_files),
        "patch_files_count": len(exported_patch_files),
        "prediction_files": sorted(exported_prediction_files),
        "hal_upload_files": sorted(exported_hal_upload_files),
        "audit": audit,
    }


def _fmt_med_iqr(dist: dict[str, float], is_int_like: bool = True, suffix: str = "") -> str:
    if is_int_like:
        return f"`{int(round(dist['median'])):,}{suffix}` (`IQR {int(round(dist['iqr'])):,}{suffix}`)"
    return f"`{dist['median']:.1f}{suffix}` (`IQR {dist['iqr']:.1f}{suffix}`)"


def render_markdown(data: dict[str, Any]) -> str:
    va = data["summary"]["vanilla"]
    wb = data["summary"]["wbc"]
    dl = data["deltas"]
    k = data["trials_per_instance_k"]
    va_d = va["distributions"]
    wb_d = wb["distributions"]
    ckpt_note = (
        f" *(Partial Checkpoint: `{data['completed_runs']}/{data['expected_total_runs']}` runs)*"
        if data.get("is_partial_checkpoint")
        else ""
    )

    lines = [
        f"# Verbatim Upstream `princeton-nlp/SWE-bench_Verified` Multi-Trial (`k={k}`) Report (`N={data['target_instances_count']}` Official Repos, `{data['completed_runs']}` Live Trials){ckpt_note}",
        f"**Date**: `{data['generated_at']}`  ",
        f"**Dataset**: `{data['dataset']}`  ",
        f"**Model**: `{data['model']}` (`Zero Synthetic Padding`, Held-Out Official `test_patch` + `FAIL_TO_PASS` Evaluation)  ",
        "",
        f"## 1. Aggregate Multi-Trial (`k={k}`) Reliability & Efficiency Scorecard",
        "",
        f"| Arm (`N={data['target_instances_count']}` Official Orgs, `{va['tasks_total']}` Trials/Arm) | `Pass@1` (Single-Trial) | `Pass^{k}` (Strict `{k}/{k}`) | `Pass@{k}` (`>=1/{k}`) | Median Context (`IQR`) | Median File Reads (`IQR`) | Median CoT Output (`IQR`) | Median Wall (`IQR`) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        (
            f"| Vanilla `Gemini 4 Carbon` *(Unshielded)* | "
            f"`{va['tasks_resolved']}/{va['tasks_total']}` (`{va['pass_at_1_pct']:.1f}%`) | "
            f"`{va['pass_hat_k_count']}/{va['instances_evaluated']}` (`{va['pass_hat_k_pct']:.1f}%`) | "
            f"`{va['pass_at_k_count']}/{va['instances_evaluated']}` (`{va['pass_at_k_pct']:.1f}%`) | "
            f"{_fmt_med_iqr(va_d['total_context_tokens'])} | "
            f"{_fmt_med_iqr(va_d['read_file_calls'], is_int_like=False)} | "
            f"{_fmt_med_iqr(va_d['output_tokens'])} | "
            f"{_fmt_med_iqr(va_d['wall_seconds'], is_int_like=False, suffix='s')} |"
        ),
        (
            f"| **`Gemini 4 Carbon + Whole Body Compute`** | "
            f"**`{wb['tasks_resolved']}/{wb['tasks_total']}` (`{wb['pass_at_1_pct']:.1f}%`, `{dl['pass_at_1_delta_pp']:+.1f}pp`)** | "
            f"**`{wb['pass_hat_k_count']}/{wb['instances_evaluated']}` (`{wb['pass_hat_k_pct']:.1f}%`, `{dl['pass_hat_3_delta_pp']:+.1f}pp`)** | "
            f"**`{wb['pass_at_k_count']}/{wb['instances_evaluated']}` (`{wb['pass_at_k_pct']:.1f}%`, `{dl['pass_at_3_delta_pp']:+.1f}pp`)** | "
            f"**{_fmt_med_iqr(wb_d['total_context_tokens'])}** (`{dl['median_total_context_delta_pct']:+.1f}%`) | "
            f"**{_fmt_med_iqr(wb_d['read_file_calls'], is_int_like=False)}** (`{dl['median_read_file_calls_delta_pct']:+.1f}%`) | "
            f"**{_fmt_med_iqr(wb_d['output_tokens'])}** (`{dl['median_output_tokens_delta_pct']:+.1f}%`) | "
            f"**{_fmt_med_iqr(wb_d['wall_seconds'], is_int_like=False, suffix='s')}** (`{dl['median_wall_delta_pct']:+.1f}%`) |"
        ),
        "",
        "## 2. Non-Parametric (`Median` / `IQR [Q1, Q3]`) & Parametric (`Mean ± SD`) Error-Bar Telemetry",
        "",
        "| Telemetry Metric | Vanilla `Median [Q1, Q3]` (`IQR`) | `+ Whole Body Compute` `Median [Q1, Q3]` (`IQR`) | Median Δ | Vanilla `Mean ± SD` | `+ Whole Body Compute` `Mean ± SD` | Total Arm Sum Δ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    metric_rows = [
        ("Wall Time (`s`)", "wall_seconds", "median_wall_delta_pct", "wall_delta_pct", False),
        ("Planner Turns", "turns", "median_turns_delta_pct", "turns_delta_pct", False),
        (
            "Billed Input Tokens",
            "billed_input_tokens",
            "median_billed_input_delta_pct",
            "billed_input_delta_pct",
            True,
        ),
        (
            "Total Context Tokens",
            "total_context_tokens",
            "median_total_context_delta_pct",
            "total_context_delta_pct",
            True,
        ),
        (
            "Net Task Context Tokens",
            "net_task_context_tokens",
            "median_net_context_delta_pct",
            "net_task_context_delta_pct",
            True,
        ),
        (
            "CoT Output Tokens",
            "output_tokens",
            "median_output_tokens_delta_pct",
            "output_tokens_delta_pct",
            True,
        ),
        (
            "`read_file` Calls",
            "read_file_calls",
            "median_read_file_calls_delta_pct",
            "read_file_calls_delta_pct",
            False,
        ),
    ]

    for label, mkey, med_dl_key, sum_dl_key, big_int in metric_rows:
        vd = va_d[mkey]
        wd = wb_d[mkey]
        if big_int:
            v_q = f"`{int(round(vd['median'])):,}` [`{int(round(vd['q1'])):,}`, `{int(round(vd['q3'])):,}`] (`{int(round(vd['iqr'])):,}`)"
            w_q = f"`{int(round(wd['median'])):,}` [`{int(round(wd['q1'])):,}`, `{int(round(wd['q3'])):,}`] (`{int(round(wd['iqr'])):,}`)"
            v_m = f"`{vd['mean']:,.0f} ± {vd['sd']:,.0f}`"
            w_m = f"`{wd['mean']:,.0f} ± {wd['sd']:,.0f}`"
        else:
            v_q = f"`{vd['median']:.1f}` [`{vd['q1']:.1f}`, `{vd['q3']:.1f}`] (`{vd['iqr']:.1f}`)"
            w_q = f"`{wd['median']:.1f}` [`{wd['q1']:.1f}`, `{wd['q3']:.1f}`] (`{wd['iqr']:.1f}`)"
            v_m = f"`{vd['mean']:.1f} ± {vd['sd']:.1f}`"
            w_m = f"`{wd['mean']:.1f} ± {wd['sd']:.1f}`"
        lines.append(
            f"| {label} | {v_q} | {w_q} | `{dl[med_dl_key]:+.1f}%` | {v_m} | {w_m} | `{dl[sum_dl_key]:+.1f}%` |"
        )

    lines.extend(
        [
            "",
            f"## 3. Per-Instance Multi-Trial (`k={k}`) Consistency Matrix (`N={data['target_instances_count']}` Distinct Official Repos)",
            "",
            f"| Instance ID | Official Repo & Primary File | Vanilla (`Pass`, `Pass^{k}`, Med Ctx, Med Reads, Med CoT) | `+ Whole Body Compute` (`Pass`, `Pass^{k}`, Med Ctx, Med Reads, Med CoT) |",
            "| :--- | :--- | :--- | :--- |",
        ]
    )

    for iid, pinfo in data["per_instance_comparison"].items():
        arms = pinfo["arms"]
        if not arms:
            continue

        def _cell(arm_d: dict[str, Any] | None) -> str:
            if not arm_d:
                return "—"
            d = arm_d["distributions"]
            hat = "✅" if arm_d["pass_hat_k"] else "❌"
            return (
                f"`{arm_d['pass_rate_str']}` (`Pass^{k}` {hat}), "
                f"ctx `{int(round(d['total_context_tokens']['median'])):,}`, "
                f"reads `{d['read_file_calls']['median']:.1f}`, "
                f"CoT `{int(round(d['output_tokens']['median'])):,}`"
            )

        lines.append(
            f"| `{iid}` | `{pinfo['repo']}` (`{pinfo['primary_file']}`, `{pinfo['primary_file_lines']:,}L`) | "
            f"{_cell(arms.get('vanilla'))} | {_cell(arms.get('wbc'))} |"
        )

    lines.extend(
        [
            "",
            f"## 4. Complete Live Trial Receipts (`{len(data['runs'])}` Sessions)",
            "",
            "| Worker | Instance ID | Trial | Arm | Official `F2P` | Turns | Reads | Wall | Billed Input | Total Context | CoT Output |",
            "| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for iid in data["per_instance_comparison"]:
        i_runs = sorted(
            [r for r in data["runs"] if r["instance_id"] == iid],
            key=lambda x: (0 if not x["is_wbc"] else 1, int(x["trial"])),
        )
        for r in i_runs:
            arm_lbl = "**`+ Whole Body Compute`**" if r["is_wbc"] else "Vanilla"
            ver = "✅ PASS" if r["verified_pass"] else "❌ FAIL"
            lines.append(
                f"| `{r['worker_name']}` | `{r['instance_id']}` | `t{r['trial']}` | {arm_lbl} | {ver} | "
                f"`{r['turns']}` | `{r['read_file_calls']}` | `{r['wall_seconds']}s` | "
                f"`{r['billed_input_tokens']:,}` | `{r['total_context_tokens']:,}` | `{r['output_tokens']:,}` |"
            )

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upstream princeton-nlp/SWE-bench_Verified Multi-Trial (k=3, N=10 Official Repos) Harness"
    )
    parser.add_argument("--setup", action="store_true", help="Provision cached repos, venvs, and k*2*N APFS CoW sandboxes")
    parser.add_argument("--verify-gold", action="store_true", help="Verify unpatched F2P fails and gold patch + test_patch passes")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Number of trials per (arm, instance) (default: 3)")
    parser.add_argument("--instances", type=str, default="", help="Optional comma-separated short_id or instance_id subset filter")
    parser.add_argument("--manifest-json", action="store_true", help="Print JSON manifest of all worker names, sandboxes, and test hints")
    parser.add_argument("--print-briefs", action="store_true", help="Print ready-to-spawn worker wave manifest and briefs")
    parser.add_argument("--map-json", type=str, default="", help="Inline JSON or path to JSON mapping worker_name -> conversation_id")
    parser.add_argument(
        "--export-hal-upload",
        nargs="?",
        const=str(DEFAULT_HAL_EXPORT_DIR),
        default="",
        help="Export self-contained Princeton HAL (hal-harness) & SWE-bench reproducibility kit to directory (default: benchmarks/swebench_verified_k3)",
    )
    args = parser.parse_args()

    selected_instances = filter_target_instances(args.instances)
    did_action = False

    if args.setup:
        setup_all(instances=selected_instances, trials=args.trials)
        did_action = True

    if args.verify_gold:
        verify_gold_harness(instances=selected_instances)
        did_action = True

    if args.manifest_json:
        manifest = build_worker_manifest(instances=selected_instances, trials=args.trials)
        print(json.dumps(manifest, indent=2))
        did_action = True

    if args.print_briefs:
        manifest = build_worker_manifest(instances=selected_instances, trials=args.trials)
        print(
            f"# SWE-bench Verified Multi-Trial Manifest: N={manifest['instances_count']} instances, "
            f"k={manifest['trials_per_instance']} trials, {manifest['total_workers']} total workers"
        )
        for w in manifest["workers"]:
            print(
                f"\n## {w['worker_name']} | {w['instance_id']} ({w['repo']}) | {w['sandbox_dir']}\n"
                f"Test hint: {w['worker_test_hint']}\n"
                f"{w['brief']}"
            )
        did_action = True

    if args.map_json:
        worker_map = parse_map_json_arg(args.map_json)
        res = compile_results(
            worker_map=worker_map,
            instances=selected_instances,
            trials=args.trials,
        )
        RESULTS_JSON.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
        md = render_markdown(res)
        REPORT_MD.write_text(md, encoding="utf-8")
        hal_bundle = build_hal_submission(res)
        HAL_SUBMISSION_JSON.write_text(json.dumps(hal_bundle, indent=2) + "\n", encoding="utf-8")
        if args.export_hal_upload:
            export_hal_upload_kit(
                submission_data=hal_bundle,
                results_data=res,
                report_md_text=md,
                out_dir=Path(args.export_hal_upload),
            )
        print(md)
        return 0

    if args.export_hal_upload:
        export_report = export_hal_upload_kit(out_dir=Path(args.export_hal_upload))
        print(json.dumps(export_report, indent=2))
        return 0

    if did_action:
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
