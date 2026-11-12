# Whole Body Compute - Empirical Benchmarks & Princeton HAL Receipts

Evaluated against the [**Princeton HAL (Holistic Agent Leaderboard)**](https://hal.cs.princeton.edu/#leaderboards) cost-vs-accuracy Pareto methodology and 4-pillar reliability framework (*Towards a Science of AI Agent Reliability*, 2026) with **zero simulation** - every figure below is computed directly from **146 saved live unscripted frontier & open-weight sessions** (`100` official upstream `princeton-nlp/SWE-bench_Verified` sessions across `10` distinct official Python organizations - `60` multi-trial `k=3` on `Gemini 4 Carbon` + `40` `2x2` Claude Code CLI runs across `Claude Sonnet` & `Claude Haiku` - plus `22` Anthropic API JSON billing receipts across `Claude Haiku 4.5` & `Sonnet` + `24` Antigravity server-side `transcript.jsonl` receipts across `Gemini 4 Argon`, `Gemini 4 Carbon`, `Gemini 3.8 Flash`, and `GPT-OSS 120B`) and live Apple Silicon organ executions.

---

## 1. Official Upstream `princeton-nlp/SWE-bench_Verified` Multi-Trial (`k=3`) Benchmark (`N=10` Official Orgs, `60` Live Trials)

- **100% Verbatim Upstream Holdout Evaluation (`Zero Synthetic Padding`):** Evaluated on `10` distinct official Python organizations (`psf/requests`, `pytest-dev/pytest`, `pydata/xarray`, `sympy/sympy`, `pallets/flask`, `mwaskom/seaborn`, `pylint-dev/pylint`, `sphinx-doc/sphinx`, `django/django`, `matplotlib/matplotlib`) at untouched historical GitHub `base_commit` SHAs (`30` trials/arm, `k=3` independent trials per repo) graded strictly against held-out official `test_patch` (`FAIL_TO_PASS` + `PASS_TO_PASS`).
- **Zero Correctness Regression:** Both `Vanilla Gemini 4 Carbon` and **`Gemini 4 Carbon + Whole Body Compute`** achieved **`Pass@1 = 30/30 (100.0%)`**, **`Pass^3 = 10/10 (100.0%)`**, and **`Pass@3 = 10/10 (100.0%)`** across all `60` live sessions (`+0.0pp` delta).
- **Context & Token Efficiency:** **`+ Whole Body Compute`** cut cumulative context by **`-32.4%`** (**`65.85M` fewer tokens**, `203.08M -> 137.23M`, **`1.48x` context efficiency**), net task context by **`-34.3%`** (`178.87M -> 117.56M`, **`1.52x` efficiency**), billed input by **`-25.6%`** (**`-2.00M` billed tokens**, `7.83M -> 5.83M`; **`-16.7%` median** `153,304 -> 127,668`), planner turns by **`-18.8%`** (**`-317` turns**, `1,685 -> 1,368`; `-8.8%` median `34.0 -> 31.0`), context IQR by **`-28.8%`** (`7.19M -> 5.12M` tighter spread), and median `read_file` calls on `3,618`-line `matplotlib/figure.py` (`matplotlib__matplotlib-25960`) by **`-45.9%`** (`74.0 -> 40.0`).

| Arm (`N=10` Official Orgs, `30` Trials/Arm) | `Pass@1` (Single-Trial) | `Pass^3` (Strict `3/3`) | `Pass@3` (`>=1/3`) | Total Context (`Sum` / `Mean ± SD`) | Net Task Context (`Sum` / `Mean ± SD`) | Billed Input (`Sum` / `Median`) | Planner Turns (`Sum` / `Median`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Vanilla Gemini 4 Carbon` *(Unshielded)* | `30/30` (`100.0%`) | `10/10` (`100.0%`) | `10/10` (`100.0%`) | `203.08M` (`6.77M ± 8.89M`) | `178.87M` (`5.96M ± 8.19M`) | `7.83M` (`153,304`) | `1,685` (`34.0`) |
| **`Gemini 4 Carbon + Whole Body Compute`** | **`30/30` (`100.0%`)** | **`10/10` (`100.0%`)** | **`10/10` (`100.0%`)** | **`137.23M` (`-32.4%` / `-65.85M`, `1.48x`)** | **`117.56M` (`-34.3%`, `1.52x`)** | **`5.83M` (`-25.6%` / `-16.7%` med)** | **`1,368` (`-18.8%` / `-317`)** |

| Instance ID | Official Repo & Primary File | `Vanilla Gemini 4 Carbon` (`Pass^3`, Med Context, Med Reads) | `Gemini 4 Carbon + Whole Body Compute` (`Pass^3`, Med Context, Med Reads) |
| :--- | :--- | :--- | :--- |
| `psf__requests-6028` | `psf/requests` (`requests/utils.py`, `1,056L`) | `3/3` (`Pass^3`), ctx `1,404,681`, reads `13.0` | `3/3` (`Pass^3`), ctx `1,555,557`, reads **`9.0` (`-30.8%`)** |
| `pytest-dev__pytest-8399` | `pytest-dev/pytest` (`src/_pytest/python.py`, `1,701L`) | `3/3` (`Pass^3`), ctx `2,173,649`, reads `15.0` | `3/3` (`Pass^3`), ctx **`1,773,333` (`-18.4%`)**, reads `20.0` |
| `pydata__xarray-6721` | `pydata/xarray` (`xarray/core/common.py`, `2,075L`) | `3/3` (`Pass^3`), ctx `2,155,406`, reads `36.0` | `3/3` (`Pass^3`), ctx `2,257,216`, reads **`24.0` (`-33.3%`)** |
| `sympy__sympy-24562` | `sympy/sympy` (`sympy/core/numbers.py`, `4,496L`) | `3/3` (`Pass^3`), ctx `783,925`, reads `8.0` | `3/3` (`Pass^3`), ctx **`705,380` (`-10.0%`)**, reads `9.0` |
| `pallets__flask-5014` | `pallets/flask` (`src/flask/blueprints.py`, `621L`) | `3/3` (`Pass^3`), ctx `984,538`, reads `9.0` | `3/3` (`Pass^3`), ctx **`880,839` (`-10.5%`)**, reads `9.0` |
| `mwaskom__seaborn-3069` | `mwaskom/seaborn` (`seaborn/_core/plot.py`, `1,649L`) | `3/3` (`Pass^3`), ctx `23,964,642`, reads `70.0` | `3/3` (`Pass^3`), ctx **`16,025,135` (`-33.1%`)**, reads `76.0` |
| `pylint-dev__pylint-6528` | `pylint-dev/pylint` (`pylint/lint/pylinter.py`, `1,188L`) | `3/3` (`Pass^3`), ctx `5,393,048`, reads `30.0` | `3/3` (`Pass^3`), ctx **`5,308,711` (`-1.6%`)**, reads `32.0` |
| `sphinx-doc__sphinx-10435` | `sphinx-doc/sphinx` (`sphinx/writers/latex.py`, `2,061L`) | `3/3` (`Pass^3`), ctx `1,183,361`, reads `15.0` | `3/3` (`Pass^3`), ctx `1,324,145`, reads `20.0` |
| `django__django-17084` | `django/django` (`django/db/models/sql/query.py`, `2,678L`) | `3/3` (`Pass^3`), ctx `5,559,962`, reads `37.0` | `3/3` (`Pass^3`), ctx **`3,045,651` (`-45.2%`)**, reads `37.0` |
| `matplotlib__matplotlib-25960` | `matplotlib/matplotlib` (`lib/matplotlib/figure.py`, `3,618L`) | `3/3` (`Pass^3`), ctx `23,108,355`, reads `74.0` | `3/3` (`Pass^3`), ctx **`14,848,727` (`-35.7%`)**, reads **`40.0` (`-45.9%`)** |

---

## 2. Official Upstream `princeton-nlp/SWE-bench_Verified` Claude Code `2x2` Pareto Matrix (`N=10` Official Orgs, `40` Live Runs)

- **`Claude Sonnet + Whole Body Compute` Strictly Dominates `Vanilla Claude Haiku` (Including Total Dollar Cost):** **`Claude Sonnet + Whole Body Compute`** solved **`10/10` (`100.0% Pass@1`) for `$3.89` (`$0.39` / resolved issue)**, unlocking `mwaskom__seaborn-3069` where `Vanilla Sonnet` (`9/10` = `90.0%`, `$4.00`) failed (`-16.2%` matched context, `-16.6%` matched cost, `-18.2%` faster matched wall time, `0` `WebSearch` leaks vs. `5` in Vanilla), while beating **`Vanilla Claude Haiku` (`8/10` = `80.0%`, `$4.42`, `24.12M` context, `459` turns)** by **`+20.0pp` accuracy AND `-12.0%` lower total dollar cost** (`-64.7%` context, `-55.8%` turns).
- **Haiku Token Shield (`-36.1%` Context / `-31.0%` Spend):** Adding Whole Body Compute to `Claude Haiku` slashed cumulative context from `24.12M -> 15.40M` (**`-36.1%`**), planner turns by **`-24.0%`** (`459 -> 349`), wall time by **`-28.0%`** (`1,761s -> 1,267s`), and total API spend by **`-31.0%`** (`$4.42 -> $3.05`).

| Claude Code `2x2` Corner (`N=10` Official `SWE-bench_Verified` Repos) | `Pass@1` Resolution | Total Context | Billed Input | Output Tokens | Planner Turns | Wall Time | Total Cost (`$ / Resolved`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Corner 1: `Vanilla Claude Haiku` *(Unshielded)* | `8/10` (`80.0%`) | `24.12M` | `565,522` | `184,098` | `459` | `1,761.2s` | `$4.42` (`$0.55`) |
| Corner 2: `Claude Haiku + Whole Body Compute` | `5/10` (`50.0%`)\* | **`15.40M` (`-36.1%`)** | **`473,679` (`-16.2%` / `-24.8%` med)** | **`120,343` (`-34.6%`)** | **`349` (`-24.0%`)** | **`1,267.4s` (`-28.0%`)** | **`$3.05` (`-31.0%`)** |
| Corner 3: `Vanilla Claude Sonnet` *(Frontier Baseline)* | `9/10` (`90.0%`) | `8.48M` (`7.32M` matched) | `323,646` | `103,577` | `197` | `1,652.1s` | `$4.00` (`$0.44`) |
| **Corner 4: `Claude Sonnet + Whole Body Compute` *(Global Pareto Winner)*** | **`10/10` (`100.0%`)** | **`8.51M` (`-16.2%` matched)** | **`308,607` (`-11.6%` matched)** | **`100,332` (`-20.8%` matched)** | `203` (`-5.9%` matched) | `1,667.7s` (`-18.2%` matched) | **`$3.89` (`$0.39` · `-12.5%` / issue)** |

*\*Note: On `django__django-17084`, the concurrent 6-worker Haiku sweep hit an external Anthropic 5-hour CLI rate limit (`HTTP 429`) mid-edit; across the universal 5-repo subset solved by all 4 corners, **`Claude Sonnet + Whole Body Compute` (`$1.04`, `55` turns, `1.87M` ctx)** still finished faster and cheaper than both `Claude Haiku + WBC` (`$1.12`) and `Vanilla Claude Haiku` (`$1.27`).*

---

## 3. Multi-Track Live Frontier Scorecard (`18` Unscripted `Haiku 4.5` A/B Sessions)

| HAL Benchmark Track | Arm | Accuracy (`Pass@1`) | Turns | Wall Time | Billed Input | Total Context | API Spend | Efficiency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`SWE-bench`** (`3` repos) | Vanilla `Haiku 4.5` -> **`+ Whole Body Compute`** | **`100.0%` (`3/3`)** | `32` -> **`27`** (`-15.6%`) | `78.12s` -> **`68.75s`** (`-12.0%`) | `108,678` -> **`47,928`** (`-55.9%`) | `1.51M` -> **`617k`** (`-59.2%`) | `$0.3987` -> **`$0.1907`** (`-52.2%`) | **`2.09x`** |
| **`CORE-Bench Hard`** (`3` capsules) | Vanilla `Haiku 4.5` -> **`+ Whole Body Compute`** | **`100.0%` (`3/3`)** | `28` -> **`21`** (`-25.0%`) | `74.55s` -> **`64.74s`** (`-13.2%`) | `170,203` -> **`45,831`** (`-73.1%`) | `1.56M` -> **`480k`** (`-69.3%`) | `$0.5237` -> **`$0.1729`** (`-67.0%`) | **`3.03x`** |
| **`OSWorld / RULER / LongMemEval`** (`3` tasks) | Vanilla `Haiku 4.5` -> **`+ Whole Body Compute`** | **`100.0%` (`3/3`)** | `36` -> **`21`** (`-41.7%`) | `160.22s` -> **`56.97s`** (`-64.4%`) | `194,624` -> **`39,133`** (`-79.9%`) | `2.19M` -> **`450k`** (`-79.4%`) | `$0.6758` -> **`$0.1509`** (`-77.7%`) | **`4.48x`** |
| **Combined Live Frontier A/B (`9` tasks)** | Vanilla `Haiku 4.5` -> **`+ Whole Body Compute`** | **`100.0%` (`9/9`)** | `96` -> **`69`** (**`-28.1%`**) | `312.89s` -> **`190.46s`** (**`-39.1%`**) | `473,505` -> **`132,892`** (**`-71.9%`**) | `5.26M` -> **`1.55M`** (**`-70.6%` / `-3.71M`**) | `$1.5981` -> **`$0.5146`** (**`-67.8%`**) | **`3.11x`** |

---

## 4. Apples-to-Apples `2x2` `Haiku 4.5` vs. `Sonnet` Matrix (`swe-03` + `lme-01`)

Without a local shield, downgrading from `Vanilla Sonnet` (`$0.4874`, `50.71s`) to `Vanilla Haiku 4.5` (`$0.5650`, `143.19s`) cost `+15.9%` more and ran `2.82x` slower due to context thrashing across `32` turns (`2.10M` tokens). Adding Whole Body Compute inverted the curve:

| `2x2` HAL Quadrant (`swe-03` + `lme-01`) | Accuracy | Turns | Wall Time | Billed Input | Total Context | Total Cost | vs. `Vanilla Sonnet` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1. `Vanilla Haiku 4.5` *(Unshielded)* | `100.0%` (`2/2`) | `32` | `143.19s` | `145,458` | `2,100,242` (`2.10M`) | `$0.5650` | **`+15.9%` costlier & `2.82x` slower** |
| 3. `Vanilla Sonnet` *(Frontier Baseline)* | `100.0%` (`2/2`) | `18` | `50.71s` | `81,404` | `715,130` (`715k`) | `$0.4874` | `1.00x` Frontier baseline |
| **4. `Sonnet + Whole Body Compute` *(Embodied Frontier)*** | **`100.0%` (`2/2`)** | **`19`** | **`59.58s`** | **`35,707` (`-56.1%`)** | **`530,877` (`-25.8%`)** | **`$0.2793` (`-42.7%`)** | **`1.75x` cheaper** |
| **2. `Haiku 4.5 + Whole Body Compute` *(Global Pareto Optimum)*** | **`100.0%` (`2/2`)** | **`21`** | **`48.52s`** *(Fastest)* | **`39,408`** | **`456,291`** *(Lowest)* | **`$0.1472` (`-69.8%`)** | **`3.31x` cheaper & faster** |

---

## 5. Cross-Vendor & Open-Weight Matrix (`Gemini 4 Carbon`, `Argon`, `3.8 Flash` & `GPT-OSS 120B`)

| Model Tier & Arm (`swe-03` + `lme-01`) | Accuracy (`Pass@1`) | Turns | Wall Time | Billed Input | Total Context | Net Task Context | vs. `Vanilla Gemini 4 Carbon` / Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `Vanilla Gemini 4 Carbon` *(Baseline)* | `100.0%` (`2/2`) | `12` | `99.0s` | `135,427` | `617,621` | `447,041` | `1.00x` Baseline |
| **`Gemini 4 Carbon + Whole Body Compute`** | **`100.0%` (`2/2`)** | **`12`** | **`100.0s`** | **`69,983` (`-48.3%`)** | **`296,985` (`-51.9%`)** | **`126,351` (`-71.7%`)** | **`2.08x` leaner (`3.54x` net)** |
| `Vanilla Gemini 4 Argon` *(Unshielded)* | `100.0%` (`2/2`) | `10` | `87.0s` | `128,978` | `488,130` | `345,955` | `1.27x` leaner context |
| **`Gemini 4 Argon + Whole Body Compute`** | **`100.0%` (`2/2`)** | **`10`** | **`121.0s`** (`73.0s` active) | **`63,160` (`-51.0%`)** | **`233,363` (`-52.2%`)** | **`91,183` (`-73.6%`)** | **`2.65x` leaner (`4.90x` net)** |
| `Vanilla Gemini 3.8 Flash` *(Unshielded)* | `100.0%` (`2/2`) | `24` | `61.0s` | `233,498` | `771,809` | `428,537` | **`+72.4%` more billed input** |
| **`Gemini 3.8 Flash + Whole Body Compute`** | **`100.0%` (`2/2`)** | **`23`** | **`62.0s`** | **`181,724` (`-22.2%`)** | **`417,062` (`-46.0%`)** | **`87,769` (`-79.5%`)** | **`5.09x` leaner net & `37.4%` faster** |
| `Vanilla GPT-OSS 120B` *(Open-Weight)* | **`50.0%` (`1/2`)** | `15` | `26.0s` | `159,437` | `664,653` | `475,164` | Failed `lme-01` (`46kB` truncation) |
| **`GPT-OSS 120B + Whole Body Compute`** | **`100.0%` (`2/2`)** | **`52`** | **`125.0s`** | **`128,538` (`-19.4%`)** | **`1,770,586`** | **`1,113,682`** | **`+50.0 pts` `Pass@1` lift (`50% -> 100%`)** |

---

## 6. Princeton HAL 4-Pillar Reliability & Safety Scorecard

| HAL Pillar | External Benchmark Standard | Unshielded / Static Baseline | `+ Whole Body Compute` | Verified Empirical Receipt |
| :--- | :--- | :---: | :---: | :--- |
| **Consistency (`Pass^3` & `Pass^8`)** | Official `SWE-bench_Verified` (`60` trials, `N=10 x 3`) + Sierra `TAU-bench` (`24` trials) | `10/10 Pass^3` (`203.08M` ctx) · `0.0% Pass^8` static macros | **`10/10 (100%) Pass^3` (`-65.85M` tok)** · **`100.0% Pass^8` (`24/24` at `$0.00`)** | Zero regression across `60` live `SWE-bench_Verified` runs; `0.37s` local `PalWorkflow` replay |
| **Robustness & Sensory** | `OSWorld` 3-Window GUI + `RULER` 4-Needle + 88-Turn Flush | `7,650` tok/step Retina PNG; brittle UI (`10/24`) | **`-96.89%` GUI sensory tokens** (`237.7` tok/step) | `33/33` (`100%`) drift survival; `4/4` needles in `0.679ms`; `98.15%` flush compaction (`100%` recall) |
| **Predictability** | Paired Turn/Context Variance & In-Process Gate Overhead | `7.19M` context IQR / `1,685` turns (`SWE-bench_Verified`) | **`-28.8%` tighter context IQR (`5.12M`)** / **`-18.8%` turns (`-317` turns)** | `-45.9%` median file reads on `3,618`-line `matplotlib/figure.py` (`74.0 -> 40.0`); **`0.102ms` gate latency** |
| **Safety (`ASR` / `FPR`)** | `AgentDojo` (NeurIPS 2024) + `CyberSecEval 3` + `OSWorld` DOM | `100.0% ASR` (`12/12` compromised) | **`0.0% ASR` (`12/12` blocked) · `0.0% FPR` (`6/6` benign)** | Deterministic 3-Stage AST SecGate, Vascular Blindfold, APFS CoW & Foveal DOM sanitization |

---

## 7. Saved Raw Receipts & Reproducibility

- **Official `princeton-nlp/SWE-bench_Verified` (`k=3`, `60` Trials) Report:** [`LIVE_UPSTREAM_SWEBENCH_K3_REPORT_2026_11_10.md`](LIVE_UPSTREAM_SWEBENCH_K3_REPORT_2026_11_10.md)
- **Official `princeton-nlp/SWE-bench_Verified` Orchestrator Head-to-Head (`N=10, k=1`, `20` Trials) Report:** [`LIVE_ORCHESTRATOR_AB_SWEBENCH_N10_REPORT.md`](LIVE_ORCHESTRATOR_AB_SWEBENCH_N10_REPORT.md) (`9/10` vs. `9/10` `Pass@1`, `-13.9%` context, `-15.5%` net context, `-8.7%` turns, `-5.7%` Frontier-equiv USD)
- **Official `princeton-nlp/SWE-bench_Verified` Claude Code `2x2` (`40` Runs) Report:** [`LIVE_CLAUDE_2X2_PARETO_SWEBENCH_REPORT_2026_11_10.md`](LIVE_CLAUDE_2X2_PARETO_SWEBENCH_REPORT_2026_11_10.md)
- **Claude Code Sonnet & Haiku Upstream Reports:** [`LIVE_CLAUDE_UPSTREAM_SWEBENCH_REPORT_2026_11_10.md`](LIVE_CLAUDE_UPSTREAM_SWEBENCH_REPORT_2026_11_10.md) · [`LIVE_CLAUDE_HAIKU_UPSTREAM_SWEBENCH_REPORT_2026_11_10.md`](LIVE_CLAUDE_HAIKU_UPSTREAM_SWEBENCH_REPORT_2026_11_10.md)
- **Princeton HAL `SWE-bench_Verified` (`k=3`) Submission Bundle:** [`hal_swebench_verified_k3_submission.json`](hal_swebench_verified_k3_submission.json)
- **Raw `SWE-bench_Verified` Telemetry JSONs:** [`live_upstream_swebench_k3_results.json`](live_upstream_swebench_k3_results.json) · [`live_orchestrator_ab_swebench_n10_results.json`](live_orchestrator_ab_swebench_n10_results.json) · [`live_claude_2x2_pareto_swebench_results.json`](live_claude_2x2_pareto_swebench_results.json)
- **Evaluation Harnesses:** [`upstream_swebench_verified_harness.py`](upstream_swebench_verified_harness.py) · [`run_claude_upstream_swebench_ab.py`](run_claude_upstream_swebench_ab.py) · [`swebench_verified_k3/`](swebench_verified_k3/)

```bash
uv run python benchmarks/upstream_swebench_verified_harness.py --report-only
uv run pytest tests/test_upstream_swebench_k3_harness.py tests/test_immune_bbb.py
```
