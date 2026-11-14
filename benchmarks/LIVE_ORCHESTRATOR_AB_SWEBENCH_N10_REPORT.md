# Vanilla Orchestrator vs. Orchestrator + WBC — Official SWE-bench Verified Head-to-Head (N=10, k=1, 20 Trials)

**Generated:** `2026-11-12T20:02:36.381062+00:00`  
**Arms:** `Vanilla Orchestrator` (`/wbc_vanilla_arm/`, hooks bypassed) vs. `Orchestrator + WBC` (`/wbc_wbc_arm/`, WBC immune + AST hooks active)  
**Protocol:** Identical Frontier Planner + Fast Execution Worker tiers, identical briefs in 2 waves of 5, zero post-holdout nudges, graded post-hoc against upstream `princeton-nlp/SWE-bench_Verified` held-out `test_patch`.

---

## 1. Controlled Head-to-Head Headline Results (Workers Only, 10 Trials / Arm)

| Metric | `Vanilla Orchestrator` | **`Orchestrator + WBC`** | Delta (`+WBC` vs. Vanilla) |
| :--- | ---: | ---: | ---: |
| **Holdout Pass@1** | `9/10` (90.0%) | **`9/10` (90.0%)** | `+0.0 pp` |
| **Non-Empty Clean Patches** | `10/10` | `10/10` | `0` |
| **Total Turns** | `357` | **`326`** | `-8.7%` |
| **Total Wall Clock (s)** | `2,518.0s` | **`2,421.0s`** | `-3.9%` |
| **Billed Input Tokens** | `1,196,977` | **`1,277,356`** | `+6.7%` (`0.94x` leaner) |
| **Cache-Read Tokens** | `20,343,816` | **`17,276,778`** | `-15.1%` |
| **KV-Cache-Normalized Billed Input** *(excl. 4 backend evictions)* | `1,196,977` | **`1,166,222`** | `-2.6%` |
| **Total Context Ingested** | `21,540,793` | **`18,554,134`** | `-13.9%` (**`1.16x` leaner**) |
| **Net Task Context** | `16,538,551` | **`13,978,630`** | `-15.5%` |
| **Output Tokens** | `293,597` | **`284,172`** | `-3.2%` |
| **File Reads (`read_file`)** | `143` | **`167`** | `+16.8%` |
| **Total Tool Calls** | `412` | **`417`** | `+1.2%` |
| **Median Turns (IQR)** | `25.5` (20.2) | **`28.0` (15.5)** | `+9.8%` |
| **Median Wall (IQR)** | `163.0s` (173.0s) | **`196.0s` (133.2s)** | `+20.2%` |
| **Median Billed Input** | `91,388` | **`95,650`** | `+4.7%` |
| **Median Total Context** | `1,116,762` | **`1,090,469`** | `-2.4%` |
| **Workers Frontier-Equiv USD** | `$9.99` | **`$9.42`** | `-5.7%` |

**Paired Outcomes:** Both Pass: `9/10` · Vanilla-only Pass: `None` · +WBC-only Pass: `None` · Both Fail: `['sea3069']`

---

## 2. End-to-End Orchestrator Overhead & Pre-Run Estimate Calibration

### End-to-End Comparison (Workers + Coordinator Phase-1/2 + Monitor)

| Metric | `Vanilla E2E` | **`+WBC E2E`** | Delta | Combined 20-Trial E2E |
| :--- | ---: | ---: | ---: | ---: |
| **Turns** | `432` | **`388`** | `-10.2%` | `824` |
| **Billed Input Tokens** | `2,780,096` | **`2,339,385`** | `-15.9%` | `5,226,146` |
| **Cache-Read Tokens** | `31,077,391` | **`21,636,054`** | `-30.4%` | `53,010,466` |
| **Total Context Ingested** | `33,857,487` | **`23,975,439`** | `-29.2%` | `58,236,612` |
| **Output Tokens** | `353,117` | **`344,146`** | `-2.5%` | `701,600` |
| **Frontier-Equiv USD** | `$16.01` | **`$13.14`** | `-18.0%` | `$29.47` |

### Pre-Run Estimate vs. Actual Telemetry

| Scope | Metric | Pre-Run Estimate | Actual | Delta |
| :--- | :--- | ---: | ---: | ---: |
| `Vanilla Orchestrator` (10 Workers) | Billed Input / Total Context | `2,450,000` / `61,000,000` | `1,196,977` / `21,540,793` | `-51.1%` / `-64.7%` |
| `Orchestrator + WBC` (10 Workers) | Billed Input / Total Context | `1,800,000` / `41,000,000` | `1,277,356` / `18,554,134` | `-29.0%` / `-54.7%` |
| **Combined 20-Trial A/B** (Full E2E) | Billed Input / Total Context | `4,250,000` / `102,000,000` | `5,226,146` / `58,236,612` | `+23.0%` / `-42.9%` |

---

## 3. Paired Per-Repository Comparison (`Vanilla` vs. `+WBC`)

| Instance | Holdout (Van / +WBC) | Turns (Van / +WBC) | Wall s (Van / +WBC) | Billed In (Van / +WBC) | Total Ctx (Van / +WBC) | `read_file` (Van / +WBC) | Ctx Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | ---: |
| `psf__requests-6028` | `PASS` / **`PASS`** | `18` / `23` | `107` / `139` | `61,920` / `75,260` | `534,345` / `785,619` | `6` / `11` | `+47.0%` |
| `pytest-dev__pytest-8399` | `PASS` / **`PASS`** | `31` / `31` | `160` / `157` | `100,926` / `96,655` | `1,349,421` / `1,000,181` | `18` / `15` | `-25.9%` |
| `pydata__xarray-6721` | `PASS` / **`PASS`** | `23` / `26` | `166` / `200` | `81,851` / `94,646` | `884,103` / `1,164,270` | `13` / `13` | `+31.7%` |
| `sympy__sympy-24562` | `PASS` / **`PASS`** | `17` / `17` | `89` / `95` | `59,270` / `63,067` | `449,448` / `472,607` | `5` / `6` | `+5.2%` |
| `pallets__flask-5014` | `PASS` / **`PASS`** | `17` / `14` | `70` / `84` | `56,933` / `54,415` | `438,001` / `358,956` | `10` / `8` | `-18.0%` |
| `mwaskom__seaborn-3069` | `FAIL` / **`FAIL`** | `97` / `64` | `762` / `593` | `302,097` / `233,995` | `9,814,064` / `6,743,349` | `31` / `32` | `-31.3%` |
| `pylint-dev__pylint-6528` | `PASS` / **`PASS`** | `62` / `42` | `473` / `416` | `195,820` / `280,081` | `3,662,387` / `2,795,904` | `21` / `25` | `-23.7%` |
| `sphinx-doc__sphinx-10435` | `PASS` / **`PASS`** | `22` / `28` | `145` / `192` | `79,295` / `91,432` | `814,744` / `1,016,668` | `9` / `11` | `+24.8%` |
| `django__django-17084` | `PASS` / **`PASS`** | `42` / `53` | `240` / `264` | `131,487` / `153,931` | `1,780,306` / `2,451,904` | `13` / `23` | `+37.7%` |
| `matplotlib__matplotlib-25960` | `PASS` / **`PASS`** | `28` / `28` | `306` / `281` | `127,378` / `133,874` | `1,813,974` / `1,764,676` | `17` / `23` | `-2.7%` |
