# Verbatim Upstream `princeton-nlp/SWE-bench_Verified` Multi-Trial (`k=3`) Report on Claude Code (`2,000L` Ceiling, `N=10` Official Repos, `60` Live Trials)
**Date**: `2026-11-11T13:08:57.321552`  
**Dataset**: `princeton-nlp/SWE-bench_Verified (Official HuggingFace Test Split, Untouched GitHub Commit SHAs)`  
**Harness**: `Claude Code CLI (2,000-line default Read ceiling, held-out official test_patch verification)`  
**Model**: `Claude Sonnet` (`Zero Synthetic Padding`, Held-Out Official `test_patch` + `FAIL_TO_PASS` Evaluation)  

## 1. Aggregate Multi-Trial (`k=3`) Reliability & Efficiency Scorecard

| Arm (`N=10` Official Orgs, `30` Trials/Arm) | `Pass@1` (Single-Trial) | `Pass^3` (Strict `3/3`) | `Pass@3` (`>=1/3`) | Median Context (`IQR`) | Median Billed Input (`IQR`) | Median Cost (`IQR`) | Total Cost (`30` Runs) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Vanilla `Claude Sonnet` *(Unshielded)* | `29/30` (`96.7%`) | `9/10` (`90.0%`) | `10/10` (`100.0%`) | `538,619` (`IQR 743,661`) | `28,303` (`IQR 19,437`) | `$0.29` (`IQR $0.33`) | `$10.47` |
| **`Claude Sonnet + Whole Body Compute`** | **`30/30` (`100.0%`, `+3.3pp`)** | **`10/10` (`100.0%`, `+10.0pp`)** | **`10/10` (`100.0%`, `+0.0pp`)** | **`636,978` (`IQR 644,385`)** (`+18.3%`) | **`28,643` (`IQR 16,562`)** (`+1.2%`) | **`$0.32` (`IQR $0.29`)** (`+10.3%`) | **`$ 10.41`** (`-0.6%`) |

## 2. Non-Parametric (`Median` / `IQR [Q1, Q3]`) & Parametric (`Mean ± SD`) Error-Bar Telemetry

| Telemetry Metric | Vanilla `Median [Q1, Q3]` (`IQR`) | `+ Whole Body Compute` `Median [Q1, Q3]` (`IQR`) | Median Δ | Vanilla `Mean ± SD` | `+ Whole Body Compute` `Mean ± SD` | Total Arm Sum Δ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Wall Time (`s`) | `136.0` [`50.3`, `197.9`] (`147.7`) | `125.1` [`52.6`, `184.5`] (`131.9`) | `-8.0%` | `138.6 ± 105.6` | `131.7 ± 101.7` | `-4.9%` |
| Planner Turns | `15.0` [`9.0`, `26.0`] (`17.0`) | `16.5` [`9.0`, `23.5`] (`14.5`) | `+10.0%` | `18.3 ± 10.7` | `18.2 ± 11.3` | `-0.4%` |
| Billed Input Tokens | `28,303` [`20,759`, `40,196`] (`19,437`) | `28,643` [`20,189`, `36,751`] (`16,562`) | `+1.2%` | `30,001 ± 12,164` | `29,836 ± 11,967` | `-0.6%` |
| Total Context Tokens | `538,619` [`298,810`, `1,042,471`] (`743,661`) | `636,978` [`279,300`, `923,686`] (`644,385`) | `+18.3%` | `733,327 ± 541,668` | `735,842 ± 585,896` | `+0.3%` |
| CoT + Code Output Tokens | `7,276` [`2,147`, `15,450`] (`13,303`) | `7,342` [`2,078`, `12,373`] (`10,295`) | `+0.9%` | `8,642 ± 6,645` | `8,367 ± 7,029` | `-3.2%` |
| API Cost (`USD`) | `$0.29` [`$0.16`, `$0.49`] (`$0.33`) | `$0.32` [`$0.16`, `$0.45`] (`$0.29`) | `+10.3%` | `$0.35 ± $0.22` | `$0.35 ± $0.23` | `-0.6%` |
| `Read` Calls | `4.0` [`3.0`, `7.0`] (`4.0`) | `3.5` [`2.0`, `5.8`] (`3.8`) | `-12.5%` | `4.9 ± 2.8` | `4.3 ± 2.7` | `-12.3%` |

### 2.1 Matched-Solved Trial Pairs (`N=29` Paired Trials Solved by Both Arms)

| Matched-Solved Metric | Vanilla `Claude Sonnet` | `Claude Sonnet + Whole Body Compute` | Matched Delta |
| :--- | :---: | :---: | :---: |
| **Total Context Ingested** | `20,836,076` | **`19,700,800`** | **`-5.5%`** |
| **Billed Input Tokens** | `853,823` | **`831,730`** | **`-2.6%`** |
| **API Cost (`USD`)** | `$9.90` | **`$9.38`** | **`-5.3%`** |
| **Planner Turns** | `522` | **`504`** | **`-3.5%`** |
| **Wall Time (`s`)** | `3908.1s` | **`3432.2s`** | **`-12.2%`** |

## 3. Per-Instance Multi-Trial (`k=3`) Consistency Matrix (`N=10` Distinct Official Repos)

| Instance ID | Primary File (`Lines`) | Vanilla (`Pass`, `Pass^3`, Med Ctx, Med Cost) | `+ Whole Body Compute` (`Pass`, `Pass^3`, Med Ctx, Med Cost) | Med Ctx Δ | Med Cost Δ |
| :--- | :--- | :--- | :--- | :---: | :---: |
| `psf__requests-6028` | `requests/utils.py` (`1,056L`) | `3/3` (`Pass^3` ✅), `1,045,426`, `$0.43` | **`3/3` (`Pass^3` ✅)**, `752,144`, `$0.33` | `-28.1%` | `-23.3%` |
| `pytest-dev__pytest-8399` | `src/_pytest/python.py` (`1,701L`) | `3/3` (`Pass^3` ✅), `216,362`, `$0.12` | **`3/3` (`Pass^3` ✅)**, `212,020`, `$0.12` | `-2.0%` | `+0.0%` |
| `pydata__xarray-6721` | `xarray/core/common.py` (`2,075L`) | `3/3` (`Pass^3` ✅), `301,576`, `$0.16` | **`3/3` (`Pass^3` ✅)**, `267,040`, `$0.15` | `-11.4%` | `-6.2%` |
| `sympy__sympy-24562` | `sympy/core/numbers.py` (`4,496L`) | `3/3` (`Pass^3` ✅), `330,322`, `$0.21` | **`3/3` (`Pass^3` ✅)**, `364,511`, `$0.23` | `+10.3%` | `+9.5%` |
| `pallets__flask-5014` | `src/flask/blueprints.py` (`621L`) | `3/3` (`Pass^3` ✅), `271,166`, `$0.13` | **`3/3` (`Pass^3` ✅)**, `271,246`, `$0.13` | `+0.0%` | `+0.0%` |
| `mwaskom__seaborn-3069` | `seaborn/_core/plot.py` (`1,649L`) | `2/3` (`Pass^3` ❌), `867,260`, `$0.44` | **`3/3` (`Pass^3` ✅)**, `1,646,226`, `$0.62` | `+89.8%` | `+40.9%` |
| `pylint-dev__pylint-6528` | `pylint/lint/pylinter.py` (`1,188L`) | `3/3` (`Pass^3` ✅), `1,530,980`, `$0.62` | **`3/3` (`Pass^3` ✅)**, `1,820,687`, `$0.69` | `+18.9%` | `+11.3%` |
| `sphinx-doc__sphinx-10435` | `sphinx/writers/latex.py` (`2,061L`) | `3/3` (`Pass^3` ✅), `1,151,584`, `$0.48` | **`3/3` (`Pass^3` ✅)**, `929,431`, `$0.46` | `-19.3%` | `-4.2%` |
| `django__django-17084` | `django/db/models/sql/query.py` (`2,678L`) | `3/3` (`Pass^3` ✅), `397,127`, `$0.19` | **`3/3` (`Pass^3` ✅)**, `380,367`, `$0.22` | `-4.2%` | `+15.8%` |
| `matplotlib__matplotlib-25960` | `lib/matplotlib/figure.py` (`3,618L`) | `3/3` (`Pass^3` ✅), `969,677`, `$0.51` | **`3/3` (`Pass^3` ✅)**, `672,788`, `$0.38` | `-30.6%` | `-25.5%` |

## 4. Complete Live Trial Receipts (`60` Sessions)

| Worker | Instance ID | Trial | Arm | Official `F2P` | Turns | Reads | Wall | Billed Input | Total Context | Output | Cost (`USD`) |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `van-req6028-t1` | `psf__requests-6028` | `t1` | Vanilla | ✅ PASS | `44` | `11` | `382.1s` | `57,307` | `2,381,194` | `21,855` | `$0.9396` |
| `van-req6028-t2` | `psf__requests-6028` | `t2` | Vanilla | ✅ PASS | `26` | `7` | `141.8s` | `31,898` | `1,045,426` | `9,375` | `$0.4251` |
| `van-req6028-t3` | `psf__requests-6028` | `t3` | Vanilla | ✅ PASS | `25` | `5` | `149.3s` | `33,421` | `997,513` | `10,618` | `$0.4338` |
| `wbc-req6028-t1` | `psf__requests-6028` | `t1` | **`+ Whole Body Compute`** | ✅ PASS | `21` | `4` | `125.3s` | `26,221` | `752,144` | `7,768` | `$0.3289` |
| `wbc-req6028-t2` | `psf__requests-6028` | `t2` | **`+ Whole Body Compute`** | ✅ PASS | `20` | `4` | `194.4s` | `35,098` | `815,859` | `12,459` | `$0.4684` |
| `wbc-req6028-t3` | `psf__requests-6028` | `t3` | **`+ Whole Body Compute`** | ✅ PASS | `19` | `2` | `132.2s` | `35,997` | `682,803` | `5,928` | `$0.3338` |
| `van-pyt8399-t1` | `pytest-dev__pytest-8399` | `t1` | Vanilla | ✅ PASS | `7` | `2` | `31.4s` | `18,433` | `216,362` | `1,015` | `$0.1246` |
| `van-pyt8399-t2` | `pytest-dev__pytest-8399` | `t2` | Vanilla | ✅ PASS | `7` | `2` | `24.0s` | `14,407` | `206,070` | `1,118` | `$0.1083` |
| `van-pyt8399-t3` | `pytest-dev__pytest-8399` | `t3` | Vanilla | ✅ PASS | `10` | `2` | `40.3s` | `15,293` | `297,888` | `2,169` | `$0.1406` |
| `wbc-pyt8399-t1` | `pytest-dev__pytest-8399` | `t1` | **`+ Whole Body Compute`** | ✅ PASS | `8` | `2` | `38.1s` | `18,654` | `251,502` | `1,190` | `$0.1343` |
| `wbc-pyt8399-t2` | `pytest-dev__pytest-8399` | `t2` | **`+ Whole Body Compute`** | ✅ PASS | `7` | `2` | `28.4s` | `15,713` | `212,020` | `1,395` | `$0.1173` |
| `wbc-pyt8399-t3` | `pytest-dev__pytest-8399` | `t3` | **`+ Whole Body Compute`** | ✅ PASS | `7` | `2` | `26.5s` | `13,596` | `201,246` | `1,039` | `$0.1035` |
| `van-xar6721-t1` | `pydata__xarray-6721` | `t1` | Vanilla | ✅ PASS | `8` | `3` | `146.2s` | `20,884` | `269,443` | `2,140` | `$0.1558` |
| `van-xar6721-t2` | `pydata__xarray-6721` | `t2` | Vanilla | ✅ PASS | `9` | `3` | `57.6s` | `21,521` | `308,375` | `2,013` | `$0.1648` |
| `van-xar6721-t3` | `pydata__xarray-6721` | `t3` | Vanilla | ✅ PASS | `9` | `4` | `57.6s` | `20,818` | `301,576` | `1,945` | `$0.1600` |
| `wbc-xar6721-t1` | `pydata__xarray-6721` | `t1` | **`+ Whole Body Compute`** | ✅ PASS | `7` | `1` | `144.9s` | `19,725` | `231,325` | `1,610` | `$0.1385` |
| `wbc-xar6721-t2` | `pydata__xarray-6721` | `t2` | **`+ Whole Body Compute`** | ✅ PASS | `9` | `3` | `56.7s` | `20,745` | `303,463` | `1,840` | `$0.1591` |
| `wbc-xar6721-t3` | `pydata__xarray-6721` | `t3` | **`+ Whole Body Compute`** | ✅ PASS | `8` | `3` | `54.9s` | `20,545` | `267,040` | `1,878` | `$0.1514` |
| `van-sym24562-t1` | `sympy__sympy-24562` | `t1` | Vanilla | ✅ PASS | `8` | `2` | `77.8s` | `23,054` | `266,085` | `5,536` | `$0.1973` |
| `van-sym24562-t2` | `sympy__sympy-24562` | `t2` | Vanilla | ✅ PASS | `10` | `3` | `58.2s` | `24,708` | `330,322` | `4,777` | `$0.2089` |
| `van-sym24562-t3` | `sympy__sympy-24562` | `t3` | Vanilla | ✅ PASS | `11` | `3` | `70.8s` | `23,546` | `374,735` | `5,541` | `$0.2210` |
| `wbc-sym24562-t1` | `sympy__sympy-24562` | `t1` | **`+ Whole Body Compute`** | ✅ PASS | `8` | `2` | `83.2s` | `23,034` | `266,162` | `5,486` | `$0.1968` |
| `wbc-sym24562-t2` | `sympy__sympy-24562` | `t2` | **`+ Whole Body Compute`** | ✅ PASS | `11` | `3` | `83.7s` | `22,334` | `364,511` | `6,915` | `$0.2281` |
| `wbc-sym24562-t3` | `sympy__sympy-24562` | `t3` | **`+ Whole Body Compute`** | ✅ PASS | `11` | `4` | `98.0s` | `26,883` | `388,526` | `7,901` | `$0.2600` |
| `van-flk5014-t1` | `pallets__flask-5014` | `t1` | Vanilla | ✅ PASS | `9` | `2` | `32.3s` | `15,523` | `271,166` | `1,704` | `$0.1314` |
| `van-flk5014-t2` | `pallets__flask-5014` | `t2` | Vanilla | ✅ PASS | `7` | `3` | `20.1s` | `15,933` | `209,309` | `1,041` | `$0.1140` |
| `van-flk5014-t3` | `pallets__flask-5014` | `t3` | Vanilla | ✅ PASS | `10` | `5` | `32.3s` | `18,702` | `314,724` | `1,802` | `$0.1532` |
| `wbc-flk5014-t1` | `pallets__flask-5014` | `t1` | **`+ Whole Body Compute`** | ✅ PASS | `13` | `4` | `49.4s` | `16,109` | `390,419` | `2,679` | `$0.1673` |
| `wbc-flk5014-t2` | `pallets__flask-5014` | `t2` | **`+ Whole Body Compute`** | ✅ PASS | `9` | `3` | `28.9s` | `15,983` | `271,246` | `1,601` | `$0.1322` |
| `wbc-flk5014-t3` | `pallets__flask-5014` | `t3` | **`+ Whole Body Compute`** | ✅ PASS | `6` | `2` | `17.0s` | `15,308` | `176,832` | `837` | `$0.1031` |
| `van-sea3069-t1` | `mwaskom__seaborn-3069` | `t1` | Vanilla | ❌ FAIL | `27` | `5` | `249.0s` | `46,194` | `1,163,723` | `16,192` | `$0.5713` |
| `van-sea3069-t2` | `mwaskom__seaborn-3069` | `t2` | Vanilla | ✅ PASS | `26` | `9` | `154.2s` | `39,627` | `867,260` | `11,682` | `$0.4420` |
| `van-sea3069-t3` | `mwaskom__seaborn-3069` | `t3` | Vanilla | ✅ PASS | `19` | `7` | `130.2s` | `33,914` | `670,797` | `9,011` | `$0.3543` |
| `wbc-sea3069-t1` | `mwaskom__seaborn-3069` | `t1` | **`+ Whole Body Compute`** | ✅ PASS | `43` | `10` | `519.7s` | `63,336` | `2,374,453` | `31,151` | `$1.0281` |
| `wbc-sea3069-t2` | `mwaskom__seaborn-3069` | `t2` | **`+ Whole Body Compute`** | ✅ PASS | `33` | `9` | `194.6s` | `51,335` | `1,646,226` | `9,768` | `$0.6231` |
| `wbc-sea3069-t3` | `mwaskom__seaborn-3069` | `t3` | **`+ Whole Body Compute`** | ✅ PASS | `24` | `5` | `169.4s` | `36,914` | `933,020` | `8,873` | `$0.4167` |
| `van-pyl6528-t1` | `pylint-dev__pylint-6528` | `t1` | Vanilla | ✅ PASS | `39` | `11` | `233.2s` | `44,008` | `1,737,607` | `18,376` | `$0.6996` |
| `van-pyl6528-t2` | `pylint-dev__pylint-6528` | `t2` | Vanilla | ✅ PASS | `31` | `7` | `148.4s` | `32,567` | `1,239,046` | `11,509` | `$0.4877` |
| `van-pyl6528-t3` | `pylint-dev__pylint-6528` | `t3` | Vanilla | ✅ PASS | `36` | `11` | `195.6s` | `41,595` | `1,530,980` | `15,635` | `$0.6217` |
| `wbc-pyl6528-t1` | `pylint-dev__pylint-6528` | `t1` | **`+ Whole Body Compute`** | ✅ PASS | `42` | `11` | `238.0s` | `40,997` | `1,820,687` | `17,063` | `$0.6916` |
| `wbc-pyl6528-t2` | `pylint-dev__pylint-6528` | `t2` | **`+ Whole Body Compute`** | ✅ PASS | `44` | `9` | `262.1s` | `49,056` | `2,051,740` | `20,885` | `$0.8066` |
| `wbc-pyl6528-t3` | `pylint-dev__pylint-6528` | `t3` | **`+ Whole Body Compute`** | ✅ PASS | `35` | `7` | `190.7s` | `39,005` | `1,414,904` | `14,399` | `$0.5763` |
| `van-sph10435-t1` | `sphinx-doc__sphinx-10435` | `t1` | Vanilla | ✅ PASS | `19` | `4` | `198.7s` | `34,533` | `746,089` | `15,908` | `$0.4406` |
| `van-sph10435-t2` | `sphinx-doc__sphinx-10435` | `t2` | Vanilla | ✅ PASS | `27` | `5` | `354.1s` | `54,383` | `1,263,960` | `18,786` | `$0.6484` |
| `van-sph10435-t3` | `sphinx-doc__sphinx-10435` | `t3` | Vanilla | ✅ PASS | `28` | `4` | `344.3s` | `34,458` | `1,151,584` | `11,504` | `$0.4774` |
| `wbc-sph10435-t1` | `sphinx-doc__sphinx-10435` | `t1` | **`+ Whole Body Compute`** | ✅ PASS | `24` | `4` | `213.1s` | `36,261` | `1,005,438` | `15,829` | `$0.4983` |
| `wbc-sph10435-t2` | `sphinx-doc__sphinx-10435` | `t2` | **`+ Whole Body Compute`** | ✅ PASS | `24` | `3` | `177.8s` | `30,856` | `929,431` | `12,115` | `$0.4254` |
| `wbc-sph10435-t3` | `sphinx-doc__sphinx-10435` | `t3` | **`+ Whole Body Compute`** | ✅ PASS | `22` | `4` | `186.8s` | `37,115` | `906,449` | `13,175` | `$0.4552` |
| `van-dja17084-t1` | `django__django-17084` | `t1` | Vanilla | ✅ PASS | `12` | `3` | `71.6s` | `20,739` | `397,127` | `4,611` | `$0.2055` |
| `van-dja17084-t2` | `django__django-17084` | `t2` | Vanilla | ✅ PASS | `12` | `2` | `47.8s` | `21,287` | `406,441` | `3,034` | `$0.1937` |
| `van-dja17084-t3` | `django__django-17084` | `t3` | Vanilla | ✅ PASS | `9` | `2` | `38.0s` | `17,514` | `279,806` | `2,610` | `$0.1498` |
| `wbc-dja17084-t1` | `django__django-17084` | `t1` | **`+ Whole Body Compute`** | ✅ PASS | `18` | `2` | `80.9s` | `29,212` | `685,232` | `5,480` | `$0.3040` |
| `wbc-dja17084-t2` | `django__django-17084` | `t2` | **`+ Whole Body Compute`** | ✅ PASS | `11` | `2` | `51.9s` | `20,070` | `348,530` | `3,383` | `$0.1810` |
| `wbc-dja17084-t3` | `django__django-17084` | `t3` | **`+ Whole Body Compute`** | ✅ PASS | `11` | `2` | `51.4s` | `28,074` | `380,367` | `3,221` | `$0.2161` |
| `van-mpl25960-t1` | `matplotlib__matplotlib-25960` | `t1` | Vanilla | ✅ PASS | `24` | `7` | `229.9s` | `42,971` | `1,033,605` | `16,240` | `$0.5335` |
| `van-mpl25960-t2` | `matplotlib__matplotlib-25960` | `t2` | Vanilla | ✅ PASS | `18` | `7` | `180.2s` | `40,394` | `751,909` | `14,897` | `$0.4540` |
| `van-mpl25960-t3` | `matplotlib__matplotlib-25960` | `t3` | Vanilla | ✅ PASS | `22` | `5` | `260.2s` | `40,385` | `969,677` | `16,628` | `$0.5148` |
| `wbc-mpl25960-t1` | `matplotlib__matplotlib-25960` | `t1` | **`+ Whole Body Compute`** | ✅ PASS | `19` | `6` | `175.1s` | `35,058` | `729,723` | `12,076` | `$0.4011` |
| `wbc-mpl25960-t2` | `matplotlib__matplotlib-25960` | `t2` | **`+ Whole Body Compute`** | ✅ PASS | `16` | `6` | `154.0s` | `34,502` | `601,167` | `12,898` | `$0.3814` |
| `wbc-mpl25960-t3` | `matplotlib__matplotlib-25960` | `t3` | **`+ Whole Body Compute`** | ✅ PASS | `17` | `7` | `125.0s` | `37,330` | `672,788` | `10,172` | `$0.3793` |
