from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

HARNESS_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "upstream_swebench_verified_harness.py"
)


def _load_harness_module():
    spec = importlib.util.spec_from_file_location("upstream_swebench_k3_harness", HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


harness = _load_harness_module()


def test_target_instances_n10_distinct_official_repos() -> None:
    instances = harness.TARGET_INSTANCES
    assert len(instances) == 10

    short_ids = [it["short_id"] for it in instances]
    instance_ids = [it["instance_id"] for it in instances]
    repo_prefixes = [it["instance_id"].split("__")[0] for it in instances]

    assert len(set(short_ids)) == 10
    assert len(set(instance_ids)) == 10
    assert len(set(repo_prefixes)) == 10

    for it in instances:
        hint = it["worker_test_hint"]
        assert ".venv/bin/python" in hint
        # Ensure worker_test_hint never leaks held-out test function node IDs
        assert "::test_" not in hint
        assert it["primary_file"]
        assert int(it["primary_file_lines"]) >= 600


def test_normalize_django_test_label() -> None:
    # Python 3.11+ unittest format
    assert (
        harness.normalize_django_test_label(
            "test_referenced_window_requires_wrapping "
            "(aggregation.tests.AggregateAnnotationPruningTests.test_referenced_window_requires_wrapping)"
        )
        == "aggregation.tests.AggregateAnnotationPruningTests.test_referenced_window_requires_wrapping"
    )
    # Python 3.10 unittest format
    assert (
        harness.normalize_django_test_label(
            "test_clear_cache (apps.tests.AppsTests)"
        )
        == "apps.tests.AppsTests.test_clear_cache"
    )
    # Already-dotted pass-through
    assert (
        harness.normalize_django_test_label("aggregation.tests.AggregateTestCase.test_alias")
        == "aggregation.tests.AggregateTestCase.test_alias"
    )


def test_compute_pass_metrics_pass_at_1_hat_3_and_at_3() -> None:
    outcomes = {
        "repoA__pkg-1": [True, True, True],    # 3/3 -> pass^3=True,  pass@3=True
        "repoB__pkg-2": [True, False, True],   # 2/3 -> pass^3=False, pass@3=True
        "repoC__pkg-3": [False, False, True],  # 1/3 -> pass^3=False, pass@3=True
        "repoD__pkg-4": [False, False, False], # 0/3 -> pass^3=False, pass@3=False
    }
    res = harness.compute_pass_metrics(outcomes, k=3)
    assert res["instances_evaluated"] == 4
    assert res["trials_total"] == 12
    assert res["trials_resolved"] == 6
    assert res["pass_at_1_pct"] == 50.0
    assert res["pass_hat_3_count"] == 1
    assert res["pass_hat_3_pct"] == 25.0
    assert res["pass_at_3_count"] == 3
    assert res["pass_at_3_pct"] == 75.0

    assert res["per_instance"]["repoA__pkg-1"]["pass_rate_str"] == "3/3"
    assert res["per_instance"]["repoA__pkg-1"]["pass_hat_k"] is True
    assert res["per_instance"]["repoB__pkg-2"]["pass_rate_str"] == "2/3"
    assert res["per_instance"]["repoB__pkg-2"]["pass_hat_k"] is False
    assert res["per_instance"]["repoB__pkg-2"]["pass_at_k"] is True
    assert res["per_instance"]["repoD__pkg-4"]["pass_at_k"] is False

    # Partial trial guard: 1/1 when k=3 should NOT satisfy strict Pass^3
    partial = harness.compute_pass_metrics({"repoE__pkg-5": [True]}, k=3)
    assert partial["pass_at_1_pct"] == 100.0
    assert partial["pass_hat_3_count"] == 0
    assert partial["pass_at_3_count"] == 1


def test_compute_distribution_stats_median_iqr_and_outliers() -> None:
    empty = harness.compute_distribution_stats([])
    assert empty["count"] == 0
    assert empty["median"] == 0.0
    assert empty["iqr"] == 0.0

    k3 = harness.compute_distribution_stats([10, 20, 30])
    assert k3["median"] == 20.0
    assert k3["q1"] == 15.0
    assert k3["q3"] == 25.0
    assert k3["iqr"] == 10.0
    assert k3["mean"] == 20.0
    assert k3["sd"] == 10.0

    # Heavy tail outlier where Median/IQR stays robust compared to Mean/SD
    skewed = harness.compute_distribution_stats([100, 110, 120, 130, 5000])
    assert skewed["median"] == 120.0
    assert skewed["q1"] == 110.0
    assert skewed["q3"] == 130.0
    assert skewed["iqr"] == 20.0
    assert skewed["mean"] > 1000.0


def test_parse_worker_transcript_tokens_cot_and_truncated_tool_calls(tmp_path: Path) -> None:
    conv_id = "conv-unit-test-001"
    logs_dir = tmp_path / conv_id / ".system_generated" / "logs"
    logs_dir.mkdir(parents=True)

    compact_steps = [
        {
            "step_index": 0,
            "type": "USER_INPUT",
            "created_at": "2026-11-10T10:00:00Z",
        },
        {
            "step_index": 1,
            "type": "PLANNER_RESPONSE",
            "created_at": "2026-11-10T10:00:05Z",
            "input_tokens": 10000,
            "cache_read_tokens": 4000,
            "output_tokens": 250,
            "tool_calls": [
                {"name": "read_file", "args": {"Path": "PROBLEM_STATEMENT.md"}},
                {"name": "view_file", "args": {"Path": "foo.py"}},
            ],
        },
        {
            "step_index": 2,
            "type": "PLANNER_RESPONSE",
            "created_at": "2026-11-10T10:00:20Z",
            "input_tokens": 2000,
            "cache_read_tokens": 14000,
            "output_tokens": 600,
            "truncated_fields": ["tool_calls"],
            "tool_calls": [],  # Truncated in compact transcript!
        },
        {
            "step_index": 3,
            "type": "PLANNER_RESPONSE",
            "created_at": "2026-11-10T10:00:35Z",
            "input_tokens": 1500,
            "cache_read_tokens": 16000,
            "output_tokens": 150,
            "tool_calls": [
                {"name": "run_command", "args": {"CommandLine": "pytest -q"}},
                {"name": "post_message", "args": {"text": "done"}},
            ],
        },
    ]

    full_steps = list(compact_steps)
    full_steps[2] = {
        "step_index": 2,
        "type": "PLANNER_RESPONSE",
        "created_at": "2026-11-10T10:00:20Z",
        "input_tokens": 2000,
        "cache_read_tokens": 14000,
        "output_tokens": 600,
        "tool_calls": [
            {"name": "read_file", "args": {"Path": "bar.py"}},
            {"name": "edit_file", "args": {"Path": "foo.py"}},
            {"name": "write_file", "args": {"Path": "helper.py"}},
            {"name": "run_command", "args": {"CommandLine": "pytest -q"}},
        ],
    }

    (logs_dir / "transcript.jsonl").write_text(
        "\n".join(json.dumps(s) for s in compact_steps) + "\n", encoding="utf-8"
    )
    (logs_dir / "transcript_full.jsonl").write_text(
        "\n".join(json.dumps(s) for s in full_steps) + "\n", encoding="utf-8"
    )

    metrics = harness.parse_worker_transcript(conv_id, brain_dir=tmp_path)
    assert metrics["turns"] == 3
    assert metrics["wall_seconds"] == 35.0
    assert metrics["billed_input_tokens"] == 13500
    assert metrics["cache_read_tokens"] == 34000
    assert metrics["total_context_tokens"] == 47500
    assert metrics["turn1_base_prompt_tokens"] == 14000
    assert metrics["net_task_context_tokens"] == 47500 - (14000 * 3)
    assert metrics["output_tokens"] == 1000
    assert metrics["read_file_calls"] == 3
    assert metrics["run_command_calls"] == 2
    assert metrics["edit_file_calls"] == 2
    assert metrics["total_tool_calls"] == 8


def _init_git_sandbox(sdir: Path, modified_body: str) -> None:
    sdir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=sdir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@wbc.local"], cwd=sdir, check=True)
    subprocess.run(["git", "config", "user.name", "WBC Test"], cwd=sdir, check=True)
    src = sdir / "pkg.py"
    src.write_text("def compute():\n    return 0\n", encoding="utf-8")
    subprocess.run(["git", "add", "pkg.py"], cwd=sdir, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=sdir, check=True, capture_output=True)
    # Apply agent modification + untracked PROBLEM_STATEMENT.md
    src.write_text(modified_body, encoding="utf-8")
    (sdir / "PROBLEM_STATEMENT.md").write_text("Issue text\n", encoding="utf-8")


def _write_synth_transcript(
    brain_dir: Path,
    conv_id: str,
    *,
    billed_in: int,
    cache_read: int,
    out_toks: int,
    reads: int,
    wall_s: int,
) -> None:
    logs_dir = brain_dir / conv_id / ".system_generated" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    tcalls = [{"name": "read_file", "args": {"Path": f"f{i}.py"}} for i in range(reads)] + [
        {"name": "edit_file", "args": {"Path": "pkg.py"}},
        {"name": "run_command", "args": {"CommandLine": "pytest -q"}},
    ]
    steps = [
        {"step_index": 0, "type": "USER_INPUT", "created_at": "2026-11-10T12:00:00Z"},
        {
            "step_index": 1,
            "type": "PLANNER_RESPONSE",
            "created_at": f"2026-11-10T12:00:{wall_s:02d}Z",
            "input_tokens": billed_in,
            "cache_read_tokens": cache_read,
            "output_tokens": out_toks,
            "tool_calls": tcalls,
        },
    ]
    (logs_dir / "transcript.jsonl").write_text(
        "\n".join(json.dumps(s) for s in steps) + "\n", encoding="utf-8"
    )


def test_compile_results_partial_and_full_k3_and_hal_submission(tmp_path: Path) -> None:
    sandbox_base = tmp_path / "sandboxes"
    cache_base = tmp_path / "cache"
    brain_dir = tmp_path / "brain"

    instances = [
        {
            "short_id": "inst1",
            "instance_id": "org1__repo1-101",
            "primary_file": "pkg.py",
            "primary_file_lines": 1500,
            "worker_test_hint": "pytest -q",
        },
        {
            "short_id": "inst2",
            "instance_id": "org2__repo2-202",
            "primary_file": "pkg.py",
            "primary_file_lines": 2400,
            "worker_test_hint": "pytest -q",
        },
    ]
    meta = {
        "org1__repo1-101": {"repo": "org1/repo1", "base_commit": "aaaa1111bbbb2222"},
        "org2__repo2-202": {"repo": "org2/repo2", "base_commit": "cccc3333dddd4444"},
    }

    pass_table: dict[str, bool] = {}
    full_map: dict[str, str] = {}

    for item in instances:
        sid = item["short_id"]
        for arm_sub, prefix, is_wbc in (
            ("wbc_vanilla_arm", "van", False),
            ("wbc_wbc_arm", "wbc", True),
        ):
            for t in (1, 2, 3):
                wname = f"{prefix}-{sid}-t{t}"
                cid = f"cid-{wname}"
                full_map[wname] = cid
                sdir = sandbox_base / arm_sub / f"{sid}_t{t}"
                # Vanilla fails trial 3 on inst2 (Pass^3 = 1/2 = 50%), WBC passes all 6/6 (Pass^3 = 2/2 = 100%)
                ok = not (not is_wbc and sid == "inst2" and t == 3)
                pass_table[str(sdir)] = ok
                _init_git_sandbox(
                    sdir,
                    f"def compute():\n    return {1 if ok else -1}  # {wname}\n",
                )
                _write_synth_transcript(
                    brain_dir,
                    cid,
                    billed_in=20000 if is_wbc else 40000,
                    cache_read=60000 if is_wbc else 140000,
                    out_toks=(1200 + t * 50) if is_wbc else (2400 + t * 100),
                    reads=2 if is_wbc else 7,
                    wall_s=15 + t if is_wbc else 30 + t,
                )

    def fake_eval(
        item: dict[str, Any], rec: dict[str, Any], arm_dir: Path, cache_base: Path = cache_base
    ) -> tuple[bool, str]:
        ok = pass_table[str(arm_dir)]
        return ok, "1 passed" if ok else "1 failed"

    # 1. Partial wave checkpoint (only inst1 completed: 6 of 12 runs)
    partial_map = {k: v for k, v in full_map.items() if "inst1" in k}
    partial_res = harness.compile_results(
        worker_map=partial_map,
        instances=instances,
        trials=3,
        sandbox_base=sandbox_base,
        cache_base=cache_base,
        brain_dir=brain_dir,
        meta_override=meta,
        evaluator_fn=fake_eval,
    )
    assert partial_res["is_partial_checkpoint"] is True
    assert partial_res["completed_runs"] == 6
    assert partial_res["expected_total_runs"] == 12
    assert partial_res["summary"]["wbc"]["instances_evaluated"] == 1
    assert partial_res["summary"]["wbc"]["pass_hat_3_pct"] == 100.0

    # 2. Full 12-run k=3 compilation
    full_res = harness.compile_results(
        worker_map=full_map,
        instances=instances,
        trials=3,
        sandbox_base=sandbox_base,
        cache_base=cache_base,
        brain_dir=brain_dir,
        meta_override=meta,
        evaluator_fn=fake_eval,
    )
    assert full_res["is_partial_checkpoint"] is False
    assert full_res["completed_runs"] == 12

    va = full_res["summary"]["vanilla"]
    wb = full_res["summary"]["wbc"]
    dl = full_res["deltas"]

    # Vanilla: 5/6 trials resolved (83.33%), Pass^3 = 1/2 (50%), Pass@3 = 2/2 (100%)
    assert va["tasks_resolved"] == 5
    assert va["pass_at_1_pct"] == pytest.approx(83.33, abs=0.01)
    assert va["pass_hat_3_pct"] == 50.0
    assert va["pass_at_3_pct"] == 100.0

    # WBC: 6/6 trials resolved (100%), Pass^3 = 2/2 (100%), Pass@3 = 2/2 (100%)
    assert wb["tasks_resolved"] == 6
    assert wb["pass_at_1_pct"] == 100.0
    assert wb["pass_hat_3_pct"] == 100.0
    assert wb["pass_at_3_pct"] == 100.0

    assert dl["pass_hat_3_delta_pp"] == 50.0
    assert dl["median_read_file_calls_delta_pct"] < -50.0
    assert dl["median_output_tokens_delta_pct"] < -40.0

    # Verify Markdown report & HAL Leaderboard submission bundle
    md = harness.render_markdown(full_res)
    assert "Pass^3" in md
    assert "Median File Reads" in md
    assert "CoT Output Tokens" in md
    assert "Gemini 4 Carbon + Whole Body Compute" in md
    assert "+ Whole Body Compute" in md
    assert "Blood-Brain Barrier" not in md
    assert "(BBB)" not in md
    assert "Organ 10" not in md
    assert "Organ 11" not in md

    hal = harness.build_hal_submission(full_res)
    hal_json_str = json.dumps(hal)
    assert hal["benchmark_name"] == "swebench_verified"
    assert hal["trials_per_instance_k"] == 3
    assert hal["arms"]["wbc"]["agent_name"] == "Gemini 4 Carbon + Whole Body Compute"
    assert "Blood-Brain Barrier" not in hal_json_str
    assert "(BBB)" not in hal_json_str
    assert "Organ 10" not in hal_json_str
    assert "Organ 11" not in hal_json_str
    assert "Organ 10" not in HARNESS_PATH.read_text(encoding="utf-8")
    assert "Organ 11" not in HARNESS_PATH.read_text(encoding="utf-8")
    assert len(hal["arms"]["wbc"]["canonical_predictions"]) == 2
    assert len(hal["arms"]["wbc"]["trial_receipts"]) == 6
    for pred in hal["arms"]["wbc"]["canonical_predictions"]:
        assert "diff --git a/pkg.py b/pkg.py" in pred["model_patch"]
        assert "PROBLEM_STATEMENT.md" not in pred["model_patch"]

    # 3. Verify export_hal_upload_kit (--export-hal-upload) schema & clean-room audit
    export_dir = tmp_path / "exported_kit"
    export_info = harness.export_hal_upload_kit(
        submission_data=hal,
        results_data=full_res,
        report_md_text=md,
        out_dir=export_dir,
    )
    assert export_info["audit"]["passed"] is True
    assert export_info["prediction_files_count"] == 8
    assert export_info["hal_upload_files_count"] == 24
    assert export_info["patch_files_count"] == 12

    upload_wbc_t1 = json.loads(
        (export_dir / "hal_upload" / "swebench_verified_wbc_t1_UPLOAD.json").read_text(
            encoding="utf-8"
        )
    )
    assert upload_wbc_t1["config"]["agent_name"] == "Gemini 4 Carbon + Whole Body Compute"
    assert upload_wbc_t1["config"]["benchmark_name"] == "swebench_verified"
    assert upload_wbc_t1["results"]["accuracy"] == 1.0
    assert upload_wbc_t1["raw_eval_results"]["resolved_instances"] == 2
    assert "total_usage" in upload_wbc_t1
    assert "wall_clock_times" in upload_wbc_t1
    assert "task_step_counts" in upload_wbc_t1

    pred_lines = [
        json.loads(line)
        for line in (export_dir / "predictions" / "wbc_canonical_predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(pred_lines) == 2
    assert set(pred_lines[0].keys()) == {"instance_id", "model_name_or_path", "model_patch"}


def test_committed_swebench_verified_k3_kit_and_hal_schema() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    kit_dir = repo_root / "benchmarks" / "swebench_verified_k3"
    readme_k3 = repo_root / "benchmarks" / "README_SWEBENCH_VERIFIED_K3.md"

    assert kit_dir.is_dir()
    assert readme_k3.is_file()
    assert (kit_dir / "README.md").is_file()
    assert (kit_dir / "hal_swebench_verified_k3_submission.json").is_file()
    assert (kit_dir / "live_upstream_swebench_k3_results.json").is_file()
    assert (kit_dir / "LIVE_UPSTREAM_SWEBENCH_K3_REPORT_2026_11_10.md").is_file()

    audit = harness.audit_public_kit_cleanliness(kit_dir)
    assert audit["passed"] is True
    assert audit["violations"] == 0

    readme_txt = readme_k3.read_text(encoding="utf-8")
    assert "/Users/" not in readme_txt
    assert "Organ 10" not in readme_txt
    assert "Organ 11" not in readme_txt
    assert "Gemini 4 Carbon + Whole Body Compute" in readme_txt

    # Verify all 60 standalone .diff files exist (30 vanilla + 30 wbc)
    van_diffs = list((kit_dir / "patches" / "vanilla").glob("*.diff"))
    wbc_diffs = list((kit_dir / "patches" / "wbc").glob("*.diff"))
    assert len(van_diffs) == 30
    assert len(wbc_diffs) == 30
    for dpath in van_diffs + wbc_diffs:
        dtxt = dpath.read_text(encoding="utf-8")
        assert "diff --git" in dtxt
        assert "PROBLEM_STATEMENT.md" not in dtxt
        assert "/Users/" not in dtxt

