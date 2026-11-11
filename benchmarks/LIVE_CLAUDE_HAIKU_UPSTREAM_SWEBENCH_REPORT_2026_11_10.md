# Upstream `princeton-nlp/SWE-bench_Verified` on Claude Code (`2,000L` Ceiling, `N=10` Official Repos)
**Date**: `2026-11-10T21:20:33.858947`  
**Harness**: `Claude Code CLI` (`2,000`-line default `Read` ceiling, held-out official `test_patch` verification)  
**Model**: `Claude Haiku` (`20` live isolated evaluations)  

## 1. Executive Cross-Harness Scorecard

| Metric | Vanilla Claude Code | `+ Whole Body Compute` | Delta |
| :--- | :---: | :---: | :---: |
| **`Pass@1` Resolution** | `8/10` (`80.0%`) | **`5/10` (`50.0%`)** | `-30.0pp` |
| **Total Context Ingested** | `24,115,878` | **`15,403,110`** | **`-36.1%`** (Median `-11.9%`) |
| **Billed Input Tokens** | `565,522` | **`473,679`** | **`-16.2%`** (Median `-24.8%`) |
| **API-Equivalent Cost (`USD`)** | `$4.42` | **`$3.05`** | **`-31.0%`** |
| **Planner Turns** | `459` | **`349`** | **`-24.0%`** |
| **Wall Time (`s`)** | `1761.2s` | **`1267.4s`** | **`-28.0%`** |
| **`>2,000L` Monster Repos Context** | `9,001,416` | **`9,139,492`** | **`+1.5%`** (Billed `-5.5%`, Cost `-3.5%`) |

## 2. Per-Repository Head-to-Head Breakdown

| Official Instance | Primary File (`Lines`) | Vanilla (`Pass`, Turns, Ctx, Cost) | `+ Whole Body Compute` (`Pass`, Turns, Ctx, Cost) | Context Δ | Cost Δ |
| :--- | :--- | :--- | :--- | :---: | :---: |
| `psf__requests-6028` | `requests/utils.py` (`1,056L`) | ❌ `62t`, `4,142,929`, `$0.68` | ❌ `36t`, `1,900,891`, `$0.35` | **`-54.1%`** | **`-48.7%`** |
| `pytest-dev__pytest-8399` | `src/_pytest/python.py` (`1,701L`) | ✅ `18t`, `751,704`, `$0.17` | ✅ `12t`, `466,182`, `$0.13` | **`-38.0%`** | **`-25.0%`** |
| `pydata__xarray-6721` | `xarray/core/common.py` (`2,075L`) | ✅ `32t`, `1,878,766`, `$0.37` | ✅ `34t`, `1,117,072`, `$0.22` | **`-40.5%`** | **`-41.9%`** |
| `sympy__sympy-24562` | `sympy/core/numbers.py` (`4,496L`) | ✅ `20t`, `730,008`, `$0.24` | ✅ `21t`, `757,556`, `$0.22` | **`+3.8%`** | **`-7.9%`** |
| `pallets__flask-5014` | `src/flask/blueprints.py` (`621L`) | ✅ `17t`, `534,098`, `$0.12` | ✅ `14t`, `521,737`, `$0.13` | **`-2.3%`** | **`+11.5%`** |
| `mwaskom__seaborn-3069` | `seaborn/_core/plot.py` (`1,649L`) | ❌ `66t`, `3,165,766`, `$0.56` | ❌ `48t`, `1,900,584`, `$0.34` | **`-40.0%`** | **`-39.7%`** |
| `pylint-dev__pylint-6528` | `pylint/lint/pylinter.py` (`1,188L`) | ✅ `108t`, `6,519,965`, `$0.99` | ❌ `38t`, `1,474,224`, `$0.28` | **`-77.4%`** | **`-71.9%`** |
| `sphinx-doc__sphinx-10435` | `sphinx/writers/latex.py` (`2,061L`) | ✅ `32t`, `1,952,740`, `$0.40` | ❌ `37t`, `2,233,710`, `$0.40` | **`+14.4%`** | **`+0.1%`** |
| `django__django-17084` | `django/db/models/sql/query.py` (`2,678L`) | ✅ `70t`, `3,032,807`, `$0.51` | ❌ `65t`, `3,109,678`, `$0.56` | **`+2.5%`** | **`+9.1%`** |
| `matplotlib__matplotlib-25960` | `lib/matplotlib/figure.py` (`3,618L`) | ✅ `34t`, `1,407,095`, `$0.37` | ✅ `44t`, `1,921,476`, `$0.43` | **`+36.6%`** | **`+16.8%`** |
