# Upstream `princeton-nlp/SWE-bench_Verified` on Claude Code (`2,000L` Ceiling, `N=10` Official Repos)
**Date**: `2026-11-10T18:36:14.181489`  
**Harness**: `Claude Code CLI` (`2,000`-line default `Read` ceiling, held-out official `test_patch` verification)  
**Model**: `Claude Sonnet` (`20` live isolated evaluations)  

## 1. Executive Cross-Harness Scorecard

| Metric | Vanilla Claude Code | `+ Whole Body Compute` | Delta |
| :--- | :---: | :---: | :---: |
| **`Pass@1` Resolution** | `9/10` (`90.0%`) | **`10/10` (`100.0%`)** | `+10.0pp` |
| **Total Context Ingested** | `8,482,401` | **`8,507,085`** | **`+0.3%`** (Median `+23.8%`) |
| **Billed Input Tokens** | `323,646` | **`308,607`** | **`-4.7%`** (Median `-3.7%`) |
| **API-Equivalent Cost (`USD`)** | `$4.00` | **`$3.89`** | **`-2.8%`** |
| **Planner Turns** | `197` | **`203`** | **`+3.0%`** |
| **Wall Time (`s`)** | `1652.1s` | **`1667.7s`** | **`+0.9%`** |
| **`>2,000L` Monster Repos Context** | `2,712,349` | **`2,917,880`** | **`+7.6%`** (Billed `+0.8%`, Cost `+0.4%`) |

## 2. Per-Repository Head-to-Head Breakdown

| Official Instance | Primary File (`Lines`) | Vanilla (`Pass`, Turns, Ctx, Cost) | `+ Whole Body Compute` (`Pass`, Turns, Ctx, Cost) | Context Δ | Cost Δ |
| :--- | :--- | :--- | :--- | :---: | :---: |
| `psf__requests-6028` | `requests/utils.py` (`1,056L`) | ✅ `44t`, `2,381,194`, `$0.94` | ✅ `21t`, `752,144`, `$0.33` | **`-68.4%`** | **`-65.0%`** |
| `pytest-dev__pytest-8399` | `src/_pytest/python.py` (`1,701L`) | ✅ `7t`, `216,362`, `$0.12` | ✅ `8t`, `251,502`, `$0.13` | **`+16.2%`** | **`+7.7%`** |
| `pydata__xarray-6721` | `xarray/core/common.py` (`2,075L`) | ✅ `8t`, `269,443`, `$0.16` | ✅ `7t`, `231,325`, `$0.14` | **`-14.2%`** | **`-11.1%`** |
| `sympy__sympy-24562` | `sympy/core/numbers.py` (`4,496L`) | ✅ `8t`, `266,085`, `$0.20` | ✅ `8t`, `266,162`, `$0.20` | **`+0.0%`** | **`-0.3%`** |
| `pallets__flask-5014` | `src/flask/blueprints.py` (`621L`) | ✅ `9t`, `271,166`, `$0.13` | ✅ `13t`, `390,419`, `$0.17` | **`+44.0%`** | **`+27.2%`** |
| `mwaskom__seaborn-3069` | `seaborn/_core/plot.py` (`1,649L`) | ❌ `27t`, `1,163,723`, `$0.57` | ✅ `43t`, `2,374,453`, `$1.03` | **`+104.0%`** | **`+80.0%`** |
| `pylint-dev__pylint-6528` | `pylint/lint/pylinter.py` (`1,188L`) | ✅ `39t`, `1,737,607`, `$0.70` | ✅ `42t`, `1,820,687`, `$0.69` | **`+4.8%`** | **`-1.1%`** |
| `sphinx-doc__sphinx-10435` | `sphinx/writers/latex.py` (`2,061L`) | ✅ `19t`, `746,089`, `$0.44` | ✅ `24t`, `1,005,438`, `$0.50` | **`+34.8%`** | **`+13.1%`** |
| `django__django-17084` | `django/db/models/sql/query.py` (`2,678L`) | ✅ `12t`, `397,127`, `$0.21` | ✅ `18t`, `685,232`, `$0.30` | **`+72.5%`** | **`+47.9%`** |
| `matplotlib__matplotlib-25960` | `lib/matplotlib/figure.py` (`3,618L`) | ✅ `24t`, `1,033,605`, `$0.53` | ✅ `19t`, `729,723`, `$0.40` | **`-29.4%`** | **`-24.8%`** |
