# Cross-Harness Frontier `SWE-bench_Verified` Multi-Trial (`k=3`) Report (`120` Live Trials: `60` Gemini 4 Carbon + `60` Claude Sonnet)
**Date**: `2026-11-11T13:08:57.323206`  
**Dataset**: Official `princeton-nlp/SWE-bench_Verified` (`N=10` distinct official GitHub organizations/repositories, untouched historical `base_commit` SHAs, verbatim `problem_statement`, held-out official `test_patch` `FAIL_TO_PASS` + `PASS_TO_PASS` evaluation)  
**Experimental Design**: `2 Harnesses × 2 Frontier Models × 2 Scaffolds (Vanilla vs. + Whole Body Compute) × 10 Official Repos × k=3 Independent Trials = 120 Live Trials`  

## 1. Unified 120-Trial Cross-Harness Executive Scorecard

| Frontier Harness & Model (`k=3`, `N=10` Repos) | Native Read Ceiling | Vanilla `Pass@1` | `+ WBC` `Pass@1` | Vanilla `Pass^3` (Strict `3/3`) | `+ WBC` `Pass^3` (Strict `3/3`) | Context Ingested (`Vanilla` → `+ WBC`) | Billed Input (`Vanilla` → `+ WBC`) | Planner Turns (`Vanilla` → `+ WBC`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Antigravity (`Gemini 4 Carbon`)** (`60` trials) | `800L` (`46KB`) | `30/30` (`100.0%`) | **`30/30` (`100.0%`)** | `10/10` (`100.0%`) | **`10/10` (`100.0%`)** | `203,080,170` → **`137,226,568` (`-32.4%`)** | `7,826,980` → **`5,826,005` (`-25.6%`)** | `1,685` → **`1,368` (`-18.8%`)** |
| **Claude Code (`Claude Sonnet`)** (`60` trials) | `2,000L` (`200KB`) | `29/30` (`96.7%`) | **`30/30` (`100.0%`, `+3.3pp`)** | `9/10` (`90.0%`) | **`10/10` (`100.0%`, `+10.0pp`)** | `21,999,799` → **`22,075,253` (`+0.3%` all / `-5.5%` matched)** | `900,017` → **`895,066` (`-0.6%` all / `-2.6%` matched)** | `549` → **`547` (`-0.4%` all / `-3.5%` matched)** |
| **Combined Cross-Harness Frontier (`120` trials)** | `800L` & `2,000L` | `59/60` (`98.3%`) | **`60/60` (`100.0%`)** | `19/20` (`95.0%`) | **`20/20` (`100.0%`)** | `225,079,969` → **`159,301,821` (`-29.2%`)** | `8,726,997` → **`6,721,071` (`-23.0%`)** | `2,234` → **`1,915` (`-14.3%`)** |

## 2. Per-Repository Cross-Harness `Pass^3` & Median Context Matrix (`120` Trials)

| Official Instance (`Primary File`) | `Gemini 4 Carbon` Vanilla (`Pass^3`, Med Ctx) | `Gemini 4 Carbon + WBC` (`Pass^3`, Med Ctx) | `Claude Sonnet` Vanilla (`Pass^3`, Med Ctx, Med Cost) | `Claude Sonnet + WBC` (`Pass^3`, Med Ctx, Med Cost) |
| :--- | :---: | :---: | :---: | :---: |
| `psf__requests-6028` (`1,056L`) | `3/3` ✅ (`1,404,681`) | **`3/3` ✅ (`1,555,557`)** | `3/3` ✅ (`1,045,426`, `$0.43`) | **`3/3` ✅ (`752,144`, `$0.33`)** |
| `pytest-dev__pytest-8399` (`1,701L`) | `3/3` ✅ (`2,173,649`) | **`3/3` ✅ (`1,773,333`)** | `3/3` ✅ (`216,362`, `$0.12`) | **`3/3` ✅ (`212,020`, `$0.12`)** |
| `pydata__xarray-6721` (`2,075L`) | `3/3` ✅ (`2,155,406`) | **`3/3` ✅ (`2,257,216`)** | `3/3` ✅ (`301,576`, `$0.16`) | **`3/3` ✅ (`267,040`, `$0.15`)** |
| `sympy__sympy-24562` (`4,496L`) | `3/3` ✅ (`783,925`) | **`3/3` ✅ (`705,380`)** | `3/3` ✅ (`330,322`, `$0.21`) | **`3/3` ✅ (`364,511`, `$0.23`)** |
| `pallets__flask-5014` (`621L`) | `3/3` ✅ (`984,538`) | **`3/3` ✅ (`880,839`)** | `3/3` ✅ (`271,166`, `$0.13`) | **`3/3` ✅ (`271,246`, `$0.13`)** |
| `mwaskom__seaborn-3069` (`1,649L`) | `3/3` ✅ (`23,964,642`) | **`3/3` ✅ (`16,025,135`)** | `2/3` ❌ (`867,260`, `$0.44`) | **`3/3` ✅ (`1,646,226`, `$0.62`)** |
| `pylint-dev__pylint-6528` (`1,188L`) | `3/3` ✅ (`5,393,048`) | **`3/3` ✅ (`5,308,711`)** | `3/3` ✅ (`1,530,980`, `$0.62`) | **`3/3` ✅ (`1,820,687`, `$0.69`)** |
| `sphinx-doc__sphinx-10435` (`2,061L`) | `3/3` ✅ (`1,183,361`) | **`3/3` ✅ (`1,324,145`)** | `3/3` ✅ (`1,151,584`, `$0.48`) | **`3/3` ✅ (`929,431`, `$0.46`)** |
| `django__django-17084` (`2,678L`) | `3/3` ✅ (`5,559,962`) | **`3/3` ✅ (`3,045,651`)** | `3/3` ✅ (`397,127`, `$0.19`) | **`3/3` ✅ (`380,367`, `$0.22`)** |
| `matplotlib__matplotlib-25960` (`3,618L`) | `3/3` ✅ (`23,108,355`) | **`3/3` ✅ (`14,848,727`)** | `3/3` ✅ (`969,677`, `$0.51`) | **`3/3` ✅ (`672,788`, `$0.38`)** |
