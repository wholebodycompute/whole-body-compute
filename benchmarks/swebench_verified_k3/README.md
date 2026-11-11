# Verbatim Upstream `princeton-nlp/SWE-bench_Verified` (`N=10, k=3`, `60` Live Trials) Reproducibility & Princeton HAL Submission Kit

This directory and companion kit (`benchmarks/swebench_verified_k3/`) contain the complete, self-contained empirical artifacts, unified `git diff` patches, token/tool-call telemetry, and **Princeton HAL (`princeton-pli/hal-harness`)** submission files for our **60-trial (`N=10` official repositories × `2` arms × `k=3` independent trials)** head-to-head evaluation of **`Gemini 4 Carbon + Whole Body Compute`** vs. **`Gemini 4 Carbon` (Unshielded Vanilla Baseline)** on `princeton-nlp/SWE-bench_Verified`.

---

## 1. Headline Empirical Results (`60` Live Trials, Zero Simulation)

| Arm (`N=10` Official Orgs, `30` Trials/Arm) | `Pass@1` (Single-Trial) | `Pass^3` (Strict `3/3`) | `Pass@3` (`>=1/3`) | Total Context Ingested (`Δ`) | Billed Input Tokens (`Δ`) | Planner Turns (`Δ`) | Wall Time (`Δ`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Vanilla `Gemini 4 Carbon` *(Unshielded)* | `30/30` (`100.0%`) | `10/10` (`100.0%`) | `10/10` (`100.0%`) | `203,080,170` (Med `2.16M`, IQR `7.19M`) | `7,826,980` (Med `153.3k`) | `1,685` (Med `34.0`) | `22,470s` (Med `393.0s`) |
| **`Gemini 4 Carbon + Whole Body Compute`** | **`30/30` (`100.0%`, `+0.0pp`)** | **`10/10` (`100.0%`, `+0.0pp`)** | **`10/10` (`100.0%`, `+0.0pp`)** | **`137,226,568` (`-32.4%`, `1.48x` leaner; Med `-7.3%`, IQR `-28.8%`)** | **`5,826,005` (`-25.6%`; Med `-16.7%`)** | **`1,368` (`-18.8%`; Med `-8.8%`)** | **`21,621s` (`-3.8%`)** |

---

## 2. Kit Layout (`benchmarks/swebench_verified_k3/`)

Regenerate or verify the entire kit at any time in 1 command:

```bash
uv run python benchmarks/upstream_swebench_verified_harness.py --export-hal-upload
```

```text
benchmarks/
├── README_SWEBENCH_VERIFIED_K3.md                 # This reproducibility & HAL submission guide
├── upstream_swebench_verified_harness.py          # Official setup, holdout verifier, k=3 compiler & HAL exporter
├── hal_swebench_verified_k3_submission.json       # Unified multi-trial (k=3) HAL leaderboard bundle
├── live_upstream_swebench_k3_results.json         # Full 60-trial raw telemetry, distributions & diffs
├── LIVE_UPSTREAM_SWEBENCH_K3_REPORT_2026_11_10.md # Full Markdown scorecard & 60-session receipt table
└── swebench_verified_k3/                          # Self-contained Open-Core & Princeton HAL kit
    ├── README.md                                  # Kit quickstart copy
    ├── hal_swebench_verified_k3_submission.json   # Sanitized HAL multi-trial bundle
    ├── live_upstream_swebench_k3_results.json     # Sanitized 60-trial results JSON
    ├── LIVE_UPSTREAM_SWEBENCH_K3_REPORT_2026_11_10.md
    ├── predictions/                               # Standard swebench.harness.run_evaluation JSONLs
    │   ├── vanilla_canonical_predictions.jsonl
    │   ├── vanilla_t1_predictions.jsonl
    │   ├── vanilla_t2_predictions.jsonl
    │   ├── vanilla_t3_predictions.jsonl
    │   ├── wbc_canonical_predictions.jsonl
    │   ├── wbc_t1_predictions.jsonl
    │   ├── wbc_t2_predictions.jsonl
    │   └── wbc_t3_predictions.jsonl
    ├── hal_upload/                                # Official princeton-pli/hal-harness (`hal-upload -D`) files
    │   ├── swebench_verified_vanilla_t{1,2,3}_UPLOAD.json
    │   ├── swebench_verified_vanilla_t{1,2,3}_RAW_SUBMISSIONS.jsonl
    │   ├── swebench_verified_vanilla_t{1,2,3}_SWE_BENCH_SUBMISSIONS.jsonl
    │   ├── swebench_verified_vanilla_t{1,2,3}_WALL_CLOCK_TIMES.jsonl
    │   ├── swebench_verified_wbc_t{1,2,3}_UPLOAD.json
    │   ├── swebench_verified_wbc_t{1,2,3}_RAW_SUBMISSIONS.jsonl
    │   ├── swebench_verified_wbc_t{1,2,3}_SWE_BENCH_SUBMISSIONS.jsonl
    │   └── swebench_verified_wbc_t{1,2,3}_WALL_CLOCK_TIMES.jsonl
    └── patches/                                   # All 60 standalone unified git diffs (`model_patch`)
        ├── vanilla/<instance_id>_t{1,2,3}.diff    # 30 Vanilla diffs
        └── wbc/<instance_id>_t{1,2,3}.diff        # 30 Gemini 4 Carbon + Whole Body Compute diffs
```

---

## 3. How to Inspect All 60 `model_patch` Diffs

Every trial's exact `git diff HEAD` patch is available both as standalone `.diff` files under `benchmarks/swebench_verified_k3/patches/{vanilla,wbc}/` and inside the JSONL/JSON bundles:

```bash
# List all 60 standalone unified git diffs (30 vanilla + 30 wbc)
ls -la benchmarks/swebench_verified_k3/patches/vanilla/ benchmarks/swebench_verified_k3/patches/wbc/

# Inspect any specific trial's patch (e.g., Django #17084, Trial 1, + Whole Body Compute)
cat benchmarks/swebench_verified_k3/patches/wbc/django__django-17084_t1.diff

# Verify all 60 patches are non-empty unified diffs touching real repository files
uv run python -c '
import json
data = json.loads(open("benchmarks/swebench_verified_k3/live_upstream_swebench_k3_results.json").read())
assert len(data["runs"]) == 60
for r in data["runs"]:
    assert r["verified_pass"] is True
    assert "diff --git" in r["model_patch"], f"Missing diff in {r[\"worker_name\"]}"
print("Verified 60/60 non-empty passing unified git diffs!")
'
```

---

## 4. How to Audit Zero Holdout Leakage

Our harness enforces strict separation between the agent workspace (`sandbox_dir`) and the grading harness (`evaluate_sandbox`):

1. **Agent Sandboxes Only Receive `base_commit` + `PROBLEM_STATEMENT.md`**:
   - During `--setup` (`setup_all` in [`benchmarks/upstream_swebench_verified_harness.py`](../upstream_swebench_verified_harness.py)), each repository is checked out at its exact historical `base_commit` SHA and cloned via APFS Copy-on-Write (`cp -cR`).
   - Only `PROBLEM_STATEMENT.md` (containing the verbatim upstream `problem_statement` string) and a `.venv` symlink are placed in the sandbox.
   - Neither the gold `patch` nor the held-out `test_patch` nor `FAIL_TO_PASS` test node names (`::test_...`) are ever written to any agent sandbox or `worker_test_hint`.
2. **Isolated Post-Hoc Held-Out `test_patch` Grading**:
   - During grading (`evaluate_sandbox`), the harness clones the finished agent sandbox to an ephemeral grading directory (`eval_<arm>_<sid>_<pid>`), removes `PROBLEM_STATEMENT.md` and `.venv`, resets any files touched by the held-out `test_patch` (`git checkout HEAD -- <test_patch_files>`), pipes the official held-out `rec["test_patch"]` into `git apply -`, and executes the official `FAIL_TO_PASS` + `PASS_TO_PASS` test suite.

Run the automated holdout & leak audit in 1 command:

```bash
uv run pytest -q --tb=short tests/test_upstream_swebench_k3_harness.py
```

---

## 5. Reproduce `--verify-gold` & Held-Out `test_patch` Evaluation in 1 Command

To fetch the 10 official upstream repositories at their untouched historical GitHub commit SHAs, build their isolated virtual environments, provision all `60` APFS CoW sandboxes, and verify that **every unpatched `base_commit` fails `FAIL_TO_PASS` (`before_patch_pass=False`)** and **every upstream gold `patch` + held-out `test_patch` passes `FAIL_TO_PASS` (`after_gold_pass=True`)**:

```bash
uv run python benchmarks/upstream_swebench_verified_harness.py --setup --verify-gold --trials 3
```

To re-apply our saved `model_patch` diffs from `benchmarks/swebench_verified_k3/predictions/wbc_t1_predictions.jsonl` (or any trial `t1..t3`) against fresh `base_commit` checkouts and verify held-out `test_patch` resolution in 1 command:

```bash
uv run python -c '
import json, shutil, subprocess
from pathlib import Path
import benchmarks.upstream_swebench_verified_harness as h

meta = h.load_parquet_metadata()
preds = {
    json.loads(line)["instance_id"]: json.loads(line)["model_patch"]
    for line in Path("benchmarks/swebench_verified_k3/predictions/wbc_canonical_predictions.jsonl").read_text().splitlines()
    if line.strip()
}
for item in h.TARGET_INSTANCES:
    iid, sid = item["instance_id"], item["short_id"]
    repo_cache = h.CACHE_BASE / f"repo_{sid}"
    tmp_dir = h.CACHE_BASE / f"repro_verify_{sid}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    h.copy_tree_cow(repo_cache, tmp_dir)
    subprocess.run(["git", "apply", "-"], input=preds[iid], cwd=tmp_dir, check=True, text=True)
    ok, out = h.evaluate_sandbox(item, meta[iid], tmp_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"{iid}: verified_pass={ok}")
    assert ok, f"Failed on {iid}: {out}"
'
```

---

## 6. Compatibility with Official `swebench.harness.run_evaluation` & Princeton `hal-upload`

### 6.1 Official `swebench.harness.run_evaluation` (Docker Harness)
All files in `benchmarks/swebench_verified_k3/predictions/*.jsonl` conform 100% to the upstream `swebench.harness.run_evaluation` schema (`instance_id`, `model_name_or_path`, `model_patch`):

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --split test \
  --predictions_path benchmarks/swebench_verified_k3/predictions/wbc_canonical_predictions.jsonl \
  --max_workers 6 \
  --run_id wbc_swebench_verified_k3
```

### 6.2 Official Princeton HAL Upload (`princeton-pli/hal-harness`)
All files in `benchmarks/swebench_verified_k3/hal_upload/` match the exact file naming and schema expectations of `hal.benchmarks.base_benchmark.BaseBenchmark.process_results` and `hal.utils.upload:upload_results`:
- `<run_id>_UPLOAD.json` (`config`, `results`, `raw_eval_results`, `raw_logging_results`, `total_usage`, `total_cost`, `git_info`, `wall_clock_times`, `task_step_counts`, `task_metrics`)
- `<run_id>_RAW_SUBMISSIONS.jsonl` (`{instance_id: model_patch}` per line, bundled automatically by `hal-upload`)
- `<run_id>_SWE_BENCH_SUBMISSIONS.jsonl` (`{instance_id, model_patch, model_name_or_path}` per line)
- `<run_id>_WALL_CLOCK_TIMES.jsonl` (`{task_id, wall_clock_time}` per line)

```bash
# Upload all 6 trial waves (Vanilla t1..t3 + Whole Body Compute t1..t3) via hal-upload
hal-upload -D benchmarks/swebench_verified_k3/hal_upload
```

---

## 7. Scaling to Full `N=500` `SWE-bench_Verified` API Batch Sweeps

The harness is designed to scale cleanly from our 10-organization stratified vertical (`60` trials) or `swebench_verified_mini` (`N=50`) to the full `N=500` `princeton-nlp/SWE-bench_Verified` split:

1. **Parquet Metadata & Shard Filtering (`--instances` & `--trials`)**:
   - `benchmarks/upstream_swebench_verified_harness.py` automatically downloads and caches the full 500-instance `princeton-nlp/SWE-bench_Verified` Parquet table (`test-00000-of-00001.parquet`).
   - Use `--instances <comma_separated_ids>` to shard execution across worker waves (e.g., 10–25 instances per wave) while `--map-json` incrementally checkpoints partial waves (`is_partial_checkpoint: true`) until all `N * 2 * k` runs complete.
2. **Running via `hal-eval` or Batch Workers**:
   ```bash
   # Generate JSON worker manifest for automated batch dispatch
   uv run python benchmarks/upstream_swebench_verified_harness.py \
     --trials 3 \
     --manifest-json > /tmp/swebench_wave_manifest.json

   # Compile partial or completed wave receipts & export HAL upload kit
   uv run python benchmarks/upstream_swebench_verified_harness.py \
     --trials 3 \
     --map-json /path/to/worker_conversation_map.json \
     --export-hal-upload benchmarks/swebench_verified_k3
   ```
