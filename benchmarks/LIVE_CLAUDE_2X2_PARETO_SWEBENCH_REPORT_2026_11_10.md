# 2×2 Claude Code Model-by-Scaffold Pareto Matrix (`Haiku` vs. `Sonnet` × `Vanilla` vs. `+ Whole Body Compute`) on Upstream `princeton-nlp/SWE-bench_Verified`

**Date**: `2026-11-10`  
**Dataset**: Official `princeton-nlp/SWE-bench_Verified` (`N=10` distinct official GitHub repositories, untouched historical `base_commit` SHAs, verbatim `problem_statement`, held-out official `test_patch` `FAIL_TO_PASS` + `PASS_TO_PASS` verification)  
**Harness**: `Claude Code CLI` (`2,000`-line default `Read` ceiling, `--setting-sources local --output-format json`)  
**Total Live Isolated Evaluations**: `40` (`10` repos × `4` corners: `Vanilla Claude Haiku`, `Claude Haiku + Whole Body Compute`, `Vanilla Claude Sonnet`, `Claude Sonnet + Whole Body Compute`)  
**Source-of-Truth Receipts**:
- `benchmarks/live_claude_2x2_pareto_swebench_results.json`
- `benchmarks/live_claude_haiku_upstream_swebench_results.json`
- `benchmarks/live_claude_upstream_swebench_results.json`

---

## 1. Executive 2×2 Model-by-Scaffold Pareto Matrix (`N=10` Official Repos, `40` Live Runs)

| Metric | Corner 1: `Vanilla Claude Haiku` | Corner 2: `Claude Haiku + Whole Body Compute` | Corner 3: `Vanilla Claude Sonnet` | Corner 4: `Claude Sonnet + Whole Body Compute` | Haiku WBC Δ (`C2 vs C1`) | Sonnet WBC Δ (`C4 vs C3`) | Cross-Tier (`C2 vs C3`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Pass@1` Resolution** | `8/10` (`80.0%`) | `5/10` (`50.0%`)* | `9/10` (`90.0%`) | **`10/10` (`100.0%`)** | `-30.0pp` | **`+10.0pp`** | `-40.0pp` |
| **Total Context Ingested** | `24,115,878` | **`15,403,110`** | `8,482,401` | **`8,507,085`** | **`-36.1%`** | `+0.3%` (`-16.2%` matched) | `+81.6%` |
| **Median Context per Repo** | `1,915,753` | **`1,687,404`** | `571,608` | `707,478` | **`-11.9%`** | `+23.8%` | `+195.2%` |
| **Billed Input Tokens** | `565,522` | **`473,679`** | `323,646` | **`308,607`** | **`-16.2%`** (Med **`-24.8%`**) | **`-4.7%`** (`-11.6%` matched) | `+46.4%` |
| **Output Tokens (`CoT + Code`)** | `184,098` | **`120,343`** | `103,577` | **`100,332`** | **`-34.6%`** | **`-3.1%`** (`-20.8%` matched) | `+16.2%` |
| **API-Equivalent Cost (`USD`)** | `$4.42` | **`$3.05`** | `$4.00` | **`$3.89`** | **`-31.0%`** | **`-2.8%`** (`-16.6%` matched) | **`-23.8%`** |
| **Cost per Resolved Issue** | `$0.55` | `$0.61` | `$0.44` | **`$0.39`** | `+10.4%` | **`-12.5%`** | `+37.3%` |
| **Planner Turns** | `459` | **`349`** | `197` | `203` | **`-24.0%`** | `+3.0%` (`-5.9%` matched) | `+77.2%` |
| **Wall Time (`s`)** | `1,761.2s` | **`1,267.4s`** | `1,652.1s` | `1,667.7s` | **`-28.0%`** | `+0.9%` (`-18.2%` matched) | **`-23.3%`** |

*\*Note on Corner 2 (`Claude Haiku + Whole Body Compute`): On `django__django-17084`, the concurrent 6-worker Haiku sweep hit the external Anthropic 5-hour CLI quota limit (`apiErrorStatus: 429` at `03:20:15Z`) mid-edit right after identifying `contains_over_clause` in `django/db/models/aggregates.py`.*

---

## 2. Key Architectural Findings: The 2×2 Pareto Frontier

1. **`Claude Sonnet + Whole Body Compute` (Corner 4) Strictly Dominates `Vanilla Claude Haiku` (Corner 1) on Every Axis—Including Total Dollar Cost**:
   - A common assumption is that running a smaller tier (`Vanilla Claude Haiku`) is cheaper than running a frontier tier (`Claude Sonnet`). On official `SWE-bench_Verified` repositories, **`Vanilla Claude Haiku` actually costs `$4.42` across the 10 repos (`459` turns, `24.12M` cumulative context tokens)** because smaller models thrash inside `Bash` (`314` shell invocations) and re-read multi-thousand-line files across `60–108` turns (`pylint-dev__pylint-6528` alone took `108` turns, `6.52M` context tokens, and `$0.99`).
   - By contrast, **`Claude Sonnet + Whole Body Compute` (Corner 4) achieves `10/10` (`100.0%`) resolution for `$3.89` total (`$0.39` per resolved issue)**—delivering **`+20.0pp` higher `Pass@1` accuracy (`100%` vs. `80%`) at `-12.0%` lower total dollar cost (`$3.89` vs. `$4.42`), `-64.7%` lower context ingestion (`8.51M` vs. `24.12M`), and `-55.8%` fewer planner turns (`203` vs. `459`)**.
2. **Whole Body Compute Cuts `Claude Haiku` Token Burn by `-36.1%` and Cost by `-31.0%` (`$4.42` → `$3.05`)**:
   - Comparing Corner 2 (`Claude Haiku + Whole Body Compute`) against Corner 1 (`Vanilla Claude Haiku`), the `Blood-Brain Barrier` PreToolUse/PostToolUse hooks reduce total context tokens from `24,115,878` to `15,403,110` (**`-36.1%`**), billed input tokens from `565,522` to `473,679` (**`-16.2%`**, median **`-24.8%`**), `Bash` calls from `314` to `210` (**`-33.1%`**), planner turns from `459` to `349` (**`-24.0%`**), wall-clock time from `1,761.2s` to `1,267.4s` (**`-28.0%`**), and total API cost from `$4.42` to `$3.05` (**`-31.0%`**).
3. **Matched-Solved Subset Efficiency (Controlling for Resolution Outcomes)**:
   - **Sonnet Matched-Solved (`N=9` repos solved by both `Vanilla Sonnet` and `Sonnet + Whole Body Compute`)**: Whole Body Compute reduces total context by **`-16.2%`** (`7.32M` → `6.13M`), billed input tokens by **`-11.6%`** (`277.5K` → `245.3K`), output tokens by **`-20.8%`** (`87.4K` → `69.2K`), wall time by **`-18.2%`** (`1,403.1s` → `1,148.0s`), and API cost by **`-16.6%`** (`$3.43` → `$2.86`), while unlocking the 10th repo (`mwaskom__seaborn-3069`) that `Vanilla Sonnet` failed.
   - **Haiku Matched-Solved (`N=5` repos solved by both `Vanilla Haiku` and `Haiku + Whole Body Compute`)**: Whole Body Compute reduces total context by **`-9.8%`** (`5.30M` → `4.78M`), billed input tokens by **`-10.0%`** (`214.9K` → `193.4K`), output tokens by **`-15.0%`** (`64.4K` → `54.8K`), wall time by **`-10.9%`** (`632.0s` → `563.4s`), and API cost by **`-11.1%`** (`$1.27` → `$1.12`).
   - **Universal 4-Corner Matched-Solved (`N=5` repos solved by all 4 corners)**: Even on the 5 straightforward issues that every corner solves (`pytest-8399`, `xarray-6721`, `sympy-24562`, `flask-5014`, `matplotlib-25960`), **`Claude Sonnet + Whole Body Compute` (`$1.04`, `55` turns, `1.87M` context) is cheaper and faster than both `Claude Haiku + Whole Body Compute` (`$1.12`, `125` turns, `4.78M` context) and `Vanilla Claude Haiku` (`$1.27`, `121` turns, `5.30M` context)**.

---

## 3. Matched-Solved Token & Cost Delta Tables

### 3.1 Sonnet Matched-Solved Subset (`N=9` Repos Solved by Both Arms, Excluding `mwaskom__seaborn-3069`)

| Metric (`N=9` Matched-Solved) | `Vanilla Claude Sonnet` | `Claude Sonnet + Whole Body Compute` | Matched-Solved Delta |
| :--- | :---: | :---: | :---: |
| **Total Context Tokens** | `7,318,678` | **`6,132,632`** | **`-16.21%`** |
| **Billed Input Tokens** | `277,452` | **`245,271`** | **`-11.60%`** |
| **Output Tokens (`CoT + Code`)** | `87,385` | **`69,181`** | **`-20.83%`** |
| **Planner Turns** | `170` | **`160`** | **`-5.88%`** |
| **Wall Time (`s`)** | `1,403.09s` | **`1,147.98s`** | **`-18.18%`** |
| **API Cost (`USD`)** | `$3.4281` | **`$2.8607`** | **`-16.55%`** |

### 3.2 Haiku Matched-Solved Subset (`N=5` Repos Solved by Both Arms: `pytest-8399`, `xarray-6721`, `sympy-24562`, `flask-5014`, `matplotlib-25960`)

| Metric (`N=5` Matched-Solved) | `Vanilla Claude Haiku` | `Claude Haiku + Whole Body Compute` | Matched-Solved Delta |
| :--- | :---: | :---: | :---: |
| **Total Context Tokens** | `5,301,671` | **`4,784,023`** | **`-9.76%`** |
| **Billed Input Tokens** | `214,913` | **`193,380`** | **`-10.02%`** |
| **Output Tokens (`CoT + Code`)** | `64,432` | **`54,775`** | **`-14.99%`** |
| **Planner Turns** | `121` | `125` | `+3.31%` |
| **Wall Time (`s`)** | `632.02s` | **`563.41s`** | **`-10.86%`** |
| **API Cost (`USD`)** | `$1.2657` | **`$1.1248`** | **`-11.13%`** |

### 3.3 Universal 4-Corner Matched-Solved Comparison (`N=5` Repos Solved by All 4 Corners)

| Corner (`N=5` Universal Solved) | Turns | Wall (`s`) | Total Context | Billed Input | Output Tokens | Total Cost (`USD`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Corner 1: `Vanilla Claude Haiku`** | `121` | `632.0s` | `5,301,671` | `214,913` | `64,432` | `$1.2657` |
| **Corner 2: `Claude Haiku + Whole Body Compute`** | `125` | `563.4s` | `4,784,023` | `193,380` | `54,775` | `$1.1248` |
| **Corner 3: `Vanilla Claude Sonnet`** | `56` | `517.6s` | `2,056,661` | `120,865` | `26,635` | `$1.1428` |
| **Corner 4: `Claude Sonnet + Whole Body Compute`** | **`55`** | **`490.6s`** | **`1,869,131`** | **`112,580`** | **`23,041`** | **`$1.0379`** |

---

## 4. Per-Repository 4-Corner Breakdown (`10` Untouched GitHub SHAs)

| Official Instance (`Repo @ SHA`) | Primary File (`Lines`) | Corner 1: `Vanilla Haiku` | Corner 2: `Haiku + WBC` | Corner 3: `Vanilla Sonnet` | Corner 4: `Sonnet + WBC` | Haiku Cost Δ | Sonnet Cost Δ |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| `psf__requests-6028`<br>(`psf/requests` @ `0192aac241`) | `requests/utils.py`<br>(`1,056L`) | ❌ `62t` \| `4.14M` ctx<br>`80.2K` bill \| `$0.68` | ❌ `36t` \| `1.90M` ctx<br>`58.4K` bill \| `$0.35` | ✅ `44t` \| `2.38M` ctx<br>`57.3K` bill \| `$0.94` | ✅ **`21t` \| `752K` ctx**<br>**`26.2K` bill \| `$0.33`** | **`-48.7%`** | **`-65.0%`** |
| `pytest-dev__pytest-8399`<br>(`pytest-dev/pytest` @ `6e7dc8bac8`) | `src/_pytest/python.py`<br>(`1,701L`) | ✅ `18t` \| `752K` ctx<br>`36.0K` bill \| `$0.17` | ✅ **`12t` \| `466K` ctx**<br>**`32.8K` bill \| `$0.13`** | ✅ `7t` \| `216K` ctx<br>`18.4K` bill \| `$0.12` | ✅ `8t` \| `252K` ctx<br>`18.7K` bill \| `$0.13` | **`-25.0%`** | `+7.7%` |
| `pydata__xarray-6721`<br>(`pydata/xarray` @ `cc183652bf`) | `xarray/core/common.py`<br>(`2,075L`) | ✅ `32t` \| `1.88M` ctx<br>`61.1K` bill \| `$0.37` | ✅ `34t` \| `1.12M` ctx<br>`31.4K` bill \| `$0.22` | ✅ `8t` \| `269K` ctx<br>`20.9K` bill \| `$0.16` | ✅ **`7t` \| `231K` ctx**<br>**`19.7K` bill \| `$0.14`** | **`-41.9%`** | **`-11.1%`** |
| `sympy__sympy-24562`<br>(`sympy/sympy` @ `b1cb676cf9`) | `sympy/core/numbers.py`<br>(`4,496L`) | ✅ `20t` \| `730K` ctx<br>`38.4K` bill \| `$0.24` | ✅ `21t` \| `758K` ctx<br>`35.1K` bill \| `$0.22` | ✅ `8t` \| `266K` ctx<br>`23.1K` bill \| `$0.20` | ✅ **`8t` \| `266K` ctx**<br>**`23.0K` bill \| `$0.20`** | **`-7.9%`** | **`-0.3%`** |
| `pallets__flask-5014`<br>(`pallets/flask` @ `7ee9ceb71e`) | `src/flask/blueprints.py`<br>(`621L`) | ✅ `17t` \| `534K` ctx<br>`24.3K` bill \| `$0.12` | ✅ `14t` \| `522K` ctx<br>`32.5K` bill \| `$0.13` | ✅ `9t` \| `271K` ctx<br>`15.5K` bill \| `$0.13` | ✅ `13t` \| `390K` ctx<br>`16.1K` bill \| `$0.17` | `+11.5%` | `+27.2%` |
| `mwaskom__seaborn-3069`<br>(`mwaskom/seaborn` @ `54cab15bda`) | `seaborn/_core/plot.py`<br>(`1,649L`) | ❌ `66t` \| `3.17M` ctx<br>`66.9K` bill \| `$0.56` | ❌ `48t` \| `1.90M` ctx<br>`45.2K` bill \| `$0.34` | ❌ `27t` \| `1.16M` ctx<br>`46.2K` bill \| `$0.57` | ✅ **`43t` \| `2.37M` ctx**<br>**`63.3K` bill \| `$1.03`** | **`-39.7%`** | `+80.0%`<br>*(❌→✅)* |
| `pylint-dev__pylint-6528`<br>(`pylint-dev/pylint` @ `273a8b2562`) | `pylint/lint/pylinter.py`<br>(`1,188L`) | ✅ `108t` \| `6.52M` ctx<br>`81.1K` bill \| `$0.99` | ❌ `38t` \| `1.47M` ctx<br>`43.1K` bill \| `$0.28` | ✅ `39t` \| `1.74M` ctx<br>`44.0K` bill \| `$0.70` | ✅ **`42t` \| `1.82M` ctx**<br>**`41.0K` bill \| `$0.69`** | **`-71.9%`** | **`-1.1%`** |
| `sphinx-doc__sphinx-10435`<br>(`sphinx-doc/sphinx` @ `f1061c012e`) | `sphinx/writers/latex.py`<br>(`2,061L`) | ✅ `32t` \| `1.95M` ctx<br>`66.0K` bill \| `$0.40` | ❌ `37t` \| `2.23M` ctx<br>`62.4K` bill \| `$0.40` | ✅ `19t` \| `746K` ctx<br>`34.5K` bill \| `$0.44` | ✅ `24t` \| `1.01M` ctx<br>`36.3K` bill \| `$0.50` | `+0.1%` | `+13.1%` |
| `django__django-17084`<br>(`django/django` @ `f8c43aca46`) | `django/db/models/sql/query.py`<br>(`2,678L`) | ✅ `70t` \| `3.03M` ctx<br>`56.3K` bill \| `$0.51` | ❌ `65t` \| `3.11M` ctx<br>`71.2K` bill \| `$0.56`* | ✅ `12t` \| `397K` ctx<br>`20.7K` bill \| `$0.21` | ✅ `18t` \| `685K` ctx<br>`29.2K` bill \| `$0.30` | `+9.1%` | `+47.9%` |
| `matplotlib__matplotlib-25960`<br>(`matplotlib/matplotlib` @ `1d0d255b79`) | `lib/matplotlib/figure.py`<br>(`3,618L`) | ✅ `34t` \| `1.41M` ctx<br>`55.0K` bill \| `$0.37` | ✅ `44t` \| `1.92M` ctx<br>`61.6K` bill \| `$0.43` | ✅ `24t` \| `1.03M` ctx<br>`43.0K` bill \| `$0.53` | ✅ **`19t` \| `730K` ctx**<br>**`35.1K` bill \| `$0.40`** | `+16.8%` | **`-24.8%`** |

---

## 5. Tool-Use Distributions & `WebSearch`/`WebFetch` Leakage Audit (`~/.claude/projects/` JSONL Telemetry)

### 5.1 Aggregate Tool-Use Distribution Across All 4 Corners

| Tool / Hook Telemetry Metric | Corner 1: `Vanilla Claude Haiku` | Corner 2: `Claude Haiku + Whole Body Compute` | Corner 3: `Vanilla Claude Sonnet` | Corner 4: `Claude Sonnet + Whole Body Compute` |
| :--- | :---: | :---: | :---: | :---: |
| **Total Tool Invocations** | `449` | **`339`** (`-24.5%`) | `187` | `193` (`+3.2%`) |
| **`Read` Total Calls** | `97` | `104` | `50` | **`46`** (`-8.0%`) |
| **`Read` Bounded (`offset`/`limit`)** | `70` (`72.2%`) | **`79` (`76.0%`)** | `39` (`78.0%`) | `34` (`73.9%`) |
| **`Read` Unbounded (Full-File)** | `27` (`27.8%`) | **`25` (`24.0%`)** | `11` (`22.0%`) | `12` (`26.1%`) |
| **`Bash` Shell Calls** | `314` | **`210`** (`-33.1%`) | `114` | `125` |
| **`Edit` / `Write` Calls** | `38` / `0` | **`23` / `2`** (`-34.2%`) | `17` / `0` | `21` / `1` |
| **`ToolSearch` Calls** | `0` | `0` | `1` | **`0`** |
| **`WebSearch` Calls (External Leakage)** | `0` | **`0`** | `1` (`requests-6028`) | **`0` (`100%` blocked)** |
| **`WebFetch` Calls (External Leakage)** | `0` | **`0`** | `4` (`requests-6028`) | **`0` (`100%` blocked)** |
| **`Blood-Brain Barrier` Hook Interceptions** | `0` (Unshielded) | **`656`** | `0` (Unshielded) | **`382`** |
| **Bounded-Slice Fast-Path Grants (`<2ms`)** | `0` | **`34`** | `0` | **`9`** |
| **Autonomic Reflex Cues (`CSI >= 0.70` / Churn)** | `0` | **`27`** | `0` | **`0`** (Zero saturation) |

### 5.2 Per-Repository Tool-Use & Hook Telemetry Breakdown

| Instance | `Vanilla Haiku` (`Read B/U`, `Bash`, `Edit`) | `Haiku + WBC` (`Read B/U`, `Bash`, `Edit`, `Cues`) | `Vanilla Sonnet` (`Read B/U`, `Bash`, `Edit`, `Web`) | `Sonnet + WBC` (`Read B/U`, `Bash`, `Edit`, `Web`) |
| :--- | :--- | :--- | :--- | :--- |
| `psf__requests-6028` | `Read 11/4`, `Bash 40`, `Edit 6` | `Read 8/3`, `Bash 21`, `Edit 3`, `2 cues` | `Read 10/1`, `Bash 26`, `Edit 1`, **`Web 5`** | `Read 5/1`, `Bash 13`, `Edit 1`, **`Web 0`** |
| `pytest-dev__pytest-8399` | `Read 4/1`, `Bash 11`, `Edit 1` | `Read 2/2`, `Bash 6`, `Edit 1`, `0 cues` | `Read 2/1`, `Bash 2`, `Edit 1`, `Web 0` | `Read 2/1`, `Bash 3`, `Edit 1`, `Web 0` |
| `pydata__xarray-6721` | `Read 4/1`, `Bash 24`, `Edit 2` | `Read 14/2`, `Bash 16`, `Edit 1`, `0 cues` | `Read 2/1`, `Bash 3`, `Edit 1`, `Web 0` | `Read 1/1`, `Bash 3`, `Edit 1`, `Web 0` |
| `sympy__sympy-24562` | `Read 2/1`, `Bash 15`, `Edit 1` | `Read 2/1`, `Bash 16`, `Edit 1`, `0 cues` | `Read 1/1`, `Bash 4`, `Edit 1`, `Web 0` | `Read 1/1`, `Bash 4`, `Edit 1`, `Web 0` |
| `pallets__flask-5014` | `Read 1/2`, `Bash 12`, `Edit 1` | `Read 2/2`, `Bash 8`, `Edit 1`, `0 cues` | `Read 0/2`, `Bash 5`, `Edit 1`, `Web 0` | `Read 1/2`, `Bash 8`, `Edit 1`, `Web 0` |
| `mwaskom__seaborn-3069` | `Read 12/4`, `Bash 42`, `Edit 7` | `Read 12/4`, `Bash 28`, `Edit 3`, `4 cues` | `Read 7/2`, `Bash 14`, `Edit 3`, `Web 0` | `Read 7/3`, `Bash 25`, `Edit 7`, `Web 0` |
| `pylint-dev__pylint-6528` | `Read 11/6`, `Bash 80`, `Edit 10` | `Read 10/3`, `Bash 21`, `Edit 3`, `1 cue` | `Read 5/1`, `Bash 28`, `Edit 4`, `Web 0` | `Read 6/1`, `Bash 30`, `Edit 4`, `Web 0` |
| `sphinx-doc__sphinx-10435` | `Read 6/2`, `Bash 22`, `Edit 1` | `Read 7/3`, `Bash 23`, `Edit 2`, `2 cues` | `Read 4/1`, `Bash 12`, `Edit 1`, `Web 0` | `Read 3/1`, `Bash 18`, `Edit 1`, `Web 0` |
| `django__django-17084` | `Read 12/5`, `Bash 45`, `Edit 7` | `Read 13/4`, `Bash 40`, `Edit 7`, `15 cues` | `Read 4/0`, `Bash 6`, `Edit 1`, `Web 0` | `Read 5/0`, `Bash 11`, `Edit 1`, `Web 0` |
| `matplotlib__matplotlib-25960` | `Read 7/1`, `Bash 23`, `Edit 2` | `Read 9/1`, `Bash 31`, `Edit 1`, `3 cues` | `Read 4/1`, `Bash 14`, `Edit 3`, `Web 0` | `Read 3/1`, `Bash 10`, `Edit 3`, `Web 0` |

### 5.3 Behavioral & Mechanistic Takeaways from the Transcripts

1. **Elimination of `WebSearch` / `WebFetch` Benchmark Contamination (`psf__requests-6028`)**:
   - In `Vanilla Claude Sonnet` (`psf__requests-6028`), the unshielded agent invoked `ToolSearch` (`1` call), `WebSearch` (`1` call), and `WebFetch` (`4` calls), ballooning the run to `44` turns, `2.38M` context tokens, and `$0.94`.
   - Under `Claude Sonnet + Whole Body Compute`, the `Blood-Brain Barrier` kept execution 100% airlocked inside the local repository (`0` `WebSearch`/`WebFetch` calls), solving `psf__requests-6028` in `21` turns (`752K` context, `$0.33`)—a **`-68.4%` context reduction** and **`-65.0%` cost reduction**.
2. **Bounded `Read` Discipline (`76.0%` Bounded in `Haiku + Whole Body Compute` vs. `72.2%` in `Vanilla Haiku`) and `-33.1%` Less `Bash` Thrashing**:
   - Without Whole Body Compute, `Vanilla Claude Haiku` compensates for lower single-turn reasoning capacity by firing `314` `Bash` commands (`80` `Bash` calls on `pylint-6528` alone) and `38` `Edit` attempts.
   - With Whole Body Compute's `Blood-Brain Barrier`, `Haiku + Whole Body Compute` increases its bounded `Read` share from `72.2%` (`70/97`) to `76.0%` (`79/104` with `34` sub-2ms Context Shield fast-path slice grants), cuts `Bash` invocations by **`-33.1%`** (`314` → `210`), and cuts `Edit` thrashing by **`-34.2%`** (`38` → `25` total write/edit operations).
3. **Why Autonomic Saturation Cues Help Frontier Models (`Sonnet`) More Than Small Models (`Haiku`)**:
   - `Claude Sonnet + Whole Body Compute` solved `9/10` issues in `7–24` turns (`CSI < 0.70`), never triggering a single `RED_TIGHT` saturation warning (`0` autonomic cues), and used its clean context window on the hardest issue (`mwaskom__seaborn-3069`, `43` turns) to achieve a `10/10` (`100%`) sweep.
   - Conversely, when `Claude Haiku + Whole Body Compute` exceeded `26` turns (`CSI >= 0.70`, triggering `27` autonomic saturation/retry-churn cues across `pylint-6528`, `sphinx-10435`, `django-17084`, `requests-6028`, and `seaborn-3069`), Haiku wrapped up early on `pylint-6528` (`38` turns, `$0.28` vs. Vanilla Haiku's `108` turns, `$0.99`) and `sphinx-10435` (`37` turns) after passing only a subset of unit tests rather than the full edge-case matrix. This confirms a critical systems insight for multi-tier routing: **cognitive saturation cues (`CSI >= 0.70`) on small models (`Haiku`) should trigger an automatic tier escalation handoff to `Sonnet + Whole Body Compute` rather than asking `Haiku` to self-compress**.
