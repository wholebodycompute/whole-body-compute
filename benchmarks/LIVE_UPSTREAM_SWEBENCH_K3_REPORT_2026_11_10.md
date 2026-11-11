# Verbatim Upstream `princeton-nlp/SWE-bench_Verified` Multi-Trial (`k=3`) Report (`N=10` Official Repos, `60` Live Trials)
**Date**: `2026-11-10T17:42:57.634115`  
**Dataset**: `princeton-nlp/SWE-bench_Verified (Official HuggingFace Test Split, Untouched GitHub Commit SHAs)`  
**Model**: `Gemini 4 Carbon (Low)` (`Zero Synthetic Padding`, Held-Out Official `test_patch` + `FAIL_TO_PASS` Evaluation)  

## 1. Aggregate Multi-Trial (`k=3`) Reliability & Efficiency Scorecard

| Arm (`N=10` Official Orgs, `30` Trials/Arm) | `Pass@1` (Single-Trial) | `Pass^3` (Strict `3/3`) | `Pass@3` (`>=1/3`) | Median Context (`IQR`) | Median File Reads (`IQR`) | Median CoT Output (`IQR`) | Median Wall (`IQR`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Vanilla `Gemini 4 Carbon` *(Unshielded)* | `30/30` (`100.0%`) | `10/10` (`100.0%`) | `10/10` (`100.0%`) | `2,164,528` (`IQR 7,191,700`) | `22.5` (`IQR 27.8`) | `6,588` (`IQR 12,990`) | `393.0s` (`IQR 683.8s`) |
| **`Gemini 4 Carbon + Whole Body Compute`** | **`30/30` (`100.0%`, `+0.0pp`)** | **`10/10` (`100.0%`, `+0.0pp`)** | **`10/10` (`100.0%`, `+0.0pp`)** | **`2,005,532` (`IQR 5,118,270`)** (`-7.3%`) | **`24.0` (`IQR 23.2`)** (`+6.7%`) | **`14,689` (`IQR 13,472`)** (`+123.0%`) | **`425.5s` (`IQR 656.0s`)** (`+8.3%`) |

## 2. Non-Parametric (`Median` / `IQR [Q1, Q3]`) & Parametric (`Mean ± SD`) Error-Bar Telemetry

| Telemetry Metric | Vanilla `Median [Q1, Q3]` (`IQR`) | `+ Whole Body Compute` `Median [Q1, Q3]` (`IQR`) | Median Δ | Vanilla `Mean ± SD` | `+ Whole Body Compute` `Mean ± SD` | Total Arm Sum Δ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Wall Time (`s`) | `393.0` [`272.5`, `956.2`] (`683.8`) | `425.5` [`318.2`, `974.2`] (`656.0`) | `+8.3%` | `749.0 ± 776.8` | `720.7 ± 602.2` | `-3.8%` |
| Planner Turns | `34.0` [`22.2`, `68.0`] (`45.8`) | `31.0` [`23.2`, `58.5`] (`35.2`) | `-8.8%` | `56.2 ± 49.4` | `45.6 ± 33.4` | `-18.8%` |
| Billed Input Tokens | `153,304` [`111,467`, `285,112`] (`173,645`) | `127,668` [`103,491`, `247,712`] (`144,222`) | `-16.7%` | `260,899 ± 237,261` | `194,200 ± 145,143` | `-25.6%` |
| Total Context Tokens | `2,164,528` [`1,238,691`, `8,430,390`] (`7,191,700`) | `2,005,532` [`1,275,646`, `6,393,916`] (`5,118,270`) | `-7.3%` | `6,769,339 ± 8,892,864` | `4,574,219 ± 5,229,460` | `-32.4%` |
| Net Task Context Tokens | `1,757,117` [`912,999`, `7,352,759`] (`6,439,760`) | `1,633,668` [`829,228`, `5,289,290`] (`4,460,062`) | `-7.0%` | `5,962,389 ± 8,193,150` | `3,918,800 ± 4,770,299` | `-34.3%` |
| CoT Output Tokens | `6,588` [`4,134`, `17,124`] (`12,990`) | `14,689` [`5,670`, `19,142`] (`13,472`) | `+123.0%` | `18,432 ± 23,859` | `23,301 ± 29,819` | `+26.4%` |
| `read_file` Calls | `22.5` [`13.0`, `40.8`] (`27.8`) | `24.0` [`12.0`, `35.2`] (`23.2`) | `+6.7%` | `30.2 ± 24.2` | `27.0 ± 18.9` | `-10.6%` |

## 3. Per-Instance Multi-Trial (`k=3`) Consistency Matrix (`N=10` Distinct Official Repos)

| Instance ID | Official Repo & Primary File | Vanilla (`Pass`, `Pass^3`, Med Ctx, Med Reads, Med CoT) | `+ Whole Body Compute` (`Pass`, `Pass^3`, Med Ctx, Med Reads, Med CoT) |
| :--- | :--- | :--- | :--- |
| `psf__requests-6028` | `psf/requests` (`requests/utils.py`, `1,056L`) | `3/3` (`Pass^3` ✅), ctx `1,404,681`, reads `13.0`, CoT `2,811` | `3/3` (`Pass^3` ✅), ctx `1,555,557`, reads `9.0`, CoT `3,684` |
| `pytest-dev__pytest-8399` | `pytest-dev/pytest` (`src/_pytest/python.py`, `1,701L`) | `3/3` (`Pass^3` ✅), ctx `2,173,649`, reads `15.0`, CoT `4,708` | `3/3` (`Pass^3` ✅), ctx `1,773,333`, reads `20.0`, CoT `13,408` |
| `pydata__xarray-6721` | `pydata/xarray` (`xarray/core/common.py`, `2,075L`) | `3/3` (`Pass^3` ✅), ctx `2,155,406`, reads `36.0`, CoT `6,348` | `3/3` (`Pass^3` ✅), ctx `2,257,216`, reads `24.0`, CoT `16,386` |
| `sympy__sympy-24562` | `sympy/sympy` (`sympy/core/numbers.py`, `4,496L`) | `3/3` (`Pass^3` ✅), ctx `783,925`, reads `8.0`, CoT `3,960` | `3/3` (`Pass^3` ✅), ctx `705,380`, reads `9.0`, CoT `17,591` |
| `pallets__flask-5014` | `pallets/flask` (`src/flask/blueprints.py`, `621L`) | `3/3` (`Pass^3` ✅), ctx `984,538`, reads `9.0`, CoT `1,952` | `3/3` (`Pass^3` ✅), ctx `880,839`, reads `9.0`, CoT `2,742` |
| `mwaskom__seaborn-3069` | `mwaskom/seaborn` (`seaborn/_core/plot.py`, `1,649L`) | `3/3` (`Pass^3` ✅), ctx `23,964,642`, reads `70.0`, CoT `56,484` | `3/3` (`Pass^3` ✅), ctx `16,025,135`, reads `76.0`, CoT `71,818` |
| `pylint-dev__pylint-6528` | `pylint-dev/pylint` (`pylint/lint/pylinter.py`, `1,188L`) | `3/3` (`Pass^3` ✅), ctx `5,393,048`, reads `30.0`, CoT `14,839` | `3/3` (`Pass^3` ✅), ctx `5,308,711`, reads `32.0`, CoT `18,973` |
| `sphinx-doc__sphinx-10435` | `sphinx-doc/sphinx` (`sphinx/writers/latex.py`, `2,061L`) | `3/3` (`Pass^3` ✅), ctx `1,183,361`, reads `15.0`, CoT `5,325` | `3/3` (`Pass^3` ✅), ctx `1,324,145`, reads `20.0`, CoT `5,476` |
| `django__django-17084` | `django/django` (`django/db/models/sql/query.py`, `2,678L`) | `3/3` (`Pass^3` ✅), ctx `5,559,962`, reads `37.0`, CoT `14,946` | `3/3` (`Pass^3` ✅), ctx `3,045,651`, reads `37.0`, CoT `13,662` |
| `matplotlib__matplotlib-25960` | `matplotlib/matplotlib` (`lib/matplotlib/figure.py`, `3,618L`) | `3/3` (`Pass^3` ✅), ctx `23,108,355`, reads `74.0`, CoT `43,740` | `3/3` (`Pass^3` ✅), ctx `14,848,727`, reads `40.0`, CoT `50,183` |

## 4. Complete Live Trial Receipts (`60` Sessions)

| Worker | Instance ID | Trial | Arm | Official `F2P` | Turns | Reads | Wall | Billed Input | Total Context | CoT Output |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `van-req6028-t1` | `psf__requests-6028` | `t1` | Vanilla | ✅ PASS | `17` | `14` | `162.0s` | `139,839` | `1,404,681` | `2,538` |
| `van-req6028-t2` | `psf__requests-6028` | `t2` | Vanilla | ✅ PASS | `22` | `13` | `304.0s` | `123,969` | `1,573,545` | `3,956` |
| `van-req6028-t3` | `psf__requests-6028` | `t3` | Vanilla | ✅ PASS | `16` | `11` | `135.0s` | `107,300` | `1,060,354` | `2,811` |
| `wbc-req6028-t1` | `psf__requests-6028` | `t1` | **`+ Whole Body Compute`** | ✅ PASS | `20` | `9` | `338.0s` | `103,530` | `1,247,096` | `2,735` |
| `wbc-req6028-t2` | `psf__requests-6028` | `t2` | **`+ Whole Body Compute`** | ✅ PASS | `27` | `12` | `390.0s` | `130,246` | `2,083,356` | `18,845` |
| `wbc-req6028-t3` | `psf__requests-6028` | `t3` | **`+ Whole Body Compute`** | ✅ PASS | `24` | `8` | `241.0s` | `116,983` | `1,555,557` | `3,684` |
| `van-pyt8399-t1` | `pytest-dev__pytest-8399` | `t1` | Vanilla | ✅ PASS | `32` | `19` | `357.0s` | `239,086` | `2,721,030` | `6,828` |
| `van-pyt8399-t2` | `pytest-dev__pytest-8399` | `t2` | Vanilla | ✅ PASS | `24` | `15` | `308.0s` | `145,455` | `2,173,649` | `4,658` |
| `van-pyt8399-t3` | `pytest-dev__pytest-8399` | `t3` | Vanilla | ✅ PASS | `23` | `14` | `211.0s` | `130,099` | `1,745,724` | `4,708` |
| `wbc-pyt8399-t1` | `pytest-dev__pytest-8399` | `t1` | **`+ Whole Body Compute`** | ✅ PASS | `29` | `20` | `412.0s` | `114,814` | `1,773,333` | `11,298` |
| `wbc-pyt8399-t2` | `pytest-dev__pytest-8399` | `t2` | **`+ Whole Body Compute`** | ✅ PASS | `31` | `24` | `404.0s` | `144,851` | `2,563,672` | `16,116` |
| `wbc-pyt8399-t3` | `pytest-dev__pytest-8399` | `t3` | **`+ Whole Body Compute`** | ✅ PASS | `24` | `17` | `439.0s` | `117,131` | `1,757,994` | `13,408` |
| `van-xar6721-t1` | `pydata__xarray-6721` | `t1` | Vanilla | ✅ PASS | `46` | `43` | `454.0s` | `161,152` | `3,766,934` | `6,348` |
| `van-xar6721-t2` | `pydata__xarray-6721` | `t2` | Vanilla | ✅ PASS | `33` | `36` | `518.0s` | `202,667` | `2,155,406` | `20,493` |
| `van-xar6721-t3` | `pydata__xarray-6721` | `t3` | Vanilla | ✅ PASS | `32` | `25` | `271.0s` | `124,674` | `1,907,575` | `5,124` |
| `wbc-xar6721-t1` | `pydata__xarray-6721` | `t1` | **`+ Whole Body Compute`** | ✅ PASS | `38` | `26` | `543.0s` | `125,091` | `2,257,216` | `18,329` |
| `wbc-xar6721-t2` | `pydata__xarray-6721` | `t2` | **`+ Whole Body Compute`** | ✅ PASS | `23` | `24` | `323.0s` | `79,692` | `964,598` | `3,931` |
| `wbc-xar6721-t3` | `pydata__xarray-6721` | `t3` | **`+ Whole Body Compute`** | ✅ PASS | `30` | `24` | `443.0s` | `131,054` | `2,272,922` | `16,386` |
| `van-sym24562-t1` | `sympy__sympy-24562` | `t1` | Vanilla | ✅ PASS | `19` | `8` | `326.0s` | `91,274` | `982,422` | `17,330` |
| `van-sym24562-t2` | `sympy__sympy-24562` | `t2` | Vanilla | ✅ PASS | `18` | `8` | `208.0s` | `69,635` | `636,092` | `2,647` |
| `van-sym24562-t3` | `sympy__sympy-24562` | `t3` | Vanilla | ✅ PASS | `24` | `7` | `277.0s` | `78,314` | `783,925` | `3,960` |
| `wbc-sym24562-t1` | `sympy__sympy-24562` | `t1` | **`+ Whole Body Compute`** | ✅ PASS | `11` | `9` | `164.0s` | `53,563` | `353,370` | `10,223` |
| `wbc-sym24562-t2` | `sympy__sympy-24562` | `t2` | **`+ Whole Body Compute`** | ✅ PASS | `20` | `9` | `368.0s` | `78,233` | `897,039` | `21,317` |
| `wbc-sym24562-t3` | `sympy__sympy-24562` | `t3` | **`+ Whole Body Compute`** | ✅ PASS | `16` | `8` | `319.0s` | `73,815` | `705,380` | `17,591` |
| `van-flk5014-t1` | `pallets__flask-5014` | `t1` | Vanilla | ✅ PASS | `12` | `9` | `124.0s` | `84,193` | `655,088` | `1,765` |
| `van-flk5014-t2` | `pallets__flask-5014` | `t2` | Vanilla | ✅ PASS | `16` | `9` | `200.0s` | `93,392` | `984,538` | `1,952` |
| `van-flk5014-t3` | `pallets__flask-5014` | `t3` | Vanilla | ✅ PASS | `19` | `11` | `157.0s` | `124,788` | `1,508,961` | `2,638` |
| `wbc-flk5014-t1` | `pallets__flask-5014` | `t1` | **`+ Whole Body Compute`** | ✅ PASS | `11` | `9` | `110.0s` | `103,478` | `748,447` | `1,566` |
| `wbc-flk5014-t2` | `pallets__flask-5014` | `t2` | **`+ Whole Body Compute`** | ✅ PASS | `23` | `15` | `281.0s` | `133,086` | `1,901,963` | `3,690` |
| `wbc-flk5014-t3` | `pallets__flask-5014` | `t3` | **`+ Whole Body Compute`** | ✅ PASS | `14` | `7` | `129.0s` | `97,154` | `880,839` | `2,742` |
| `van-sea3069-t1` | `mwaskom__seaborn-3069` | `t1` | Vanilla | ✅ PASS | `69` | `26` | `1041.0s` | `288,935` | `9,875,936` | `16,499` |
| `van-sea3069-t2` | `mwaskom__seaborn-3069` | `t2` | Vanilla | ✅ PASS | `194` | `70` | `2920.0s` | `708,210` | `32,947,915` | `101,529` |
| `van-sea3069-t3` | `mwaskom__seaborn-3069` | `t3` | Vanilla | ✅ PASS | `147` | `87` | `2147.0s` | `585,662` | `23,964,642` | `56,484` |
| `wbc-sea3069-t1` | `mwaskom__seaborn-3069` | `t1` | **`+ Whole Body Compute`** | ✅ PASS | `118` | `81` | `1917.0s` | `561,155` | `16,025,135` | `76,617` |
| `wbc-sea3069-t2` | `mwaskom__seaborn-3069` | `t2` | **`+ Whole Body Compute`** | ✅ PASS | `68` | `52` | `801.0s` | `308,536` | `11,263,270` | `15,266` |
| `wbc-sea3069-t3` | `mwaskom__seaborn-3069` | `t3` | **`+ Whole Body Compute`** | ✅ PASS | `118` | `76` | `2098.0s` | `562,134` | `18,361,536` | `71,818` |
| `van-pyl6528-t1` | `pylint-dev__pylint-6528` | `t1` | Vanilla | ✅ PASS | `81` | `42` | `996.0s` | `413,933` | `10,296,200` | `14,839` |
| `van-pyl6528-t2` | `pylint-dev__pylint-6528` | `t2` | Vanilla | ✅ PASS | `53` | `27` | `1021.0s` | `216,002` | `5,342,421` | `64,308` |
| `van-pyl6528-t3` | `pylint-dev__pylint-6528` | `t3` | Vanilla | ✅ PASS | `62` | `30` | `675.0s` | `307,403` | `5,393,048` | `13,962` |
| `wbc-pyl6528-t1` | `pylint-dev__pylint-6528` | `t1` | **`+ Whole Body Compute`** | ✅ PASS | `69` | `36` | `810.0s` | `238,510` | `6,755,651` | `18,973` |
| `wbc-pyl6528-t2` | `pylint-dev__pylint-6528` | `t2` | **`+ Whole Body Compute`** | ✅ PASS | `60` | `32` | `1029.0s` | `205,376` | `5,308,711` | `14,112` |
| `wbc-pyl6528-t3` | `pylint-dev__pylint-6528` | `t3` | **`+ Whole Body Compute`** | ✅ PASS | `48` | `28` | `1307.0s` | `311,210` | `4,702,002` | `67,498` |
| `van-sph10435-t1` | `sphinx-doc__sphinx-10435` | `t1` | Vanilla | ✅ PASS | `35` | `15` | `313.0s` | `100,799` | `1,183,361` | `5,325` |
| `van-sph10435-t2` | `sphinx-doc__sphinx-10435` | `t2` | Vanilla | ✅ PASS | `29` | `13` | `349.0s` | `91,261` | `1,052,425` | `4,705` |
| `van-sph10435-t3` | `sphinx-doc__sphinx-10435` | `t3` | Vanilla | ✅ PASS | `44` | `23` | `441.0s` | `130,744` | `1,859,325` | `5,999` |
| `wbc-sph10435-t1` | `sphinx-doc__sphinx-10435` | `t1` | **`+ Whole Body Compute`** | ✅ PASS | `37` | `20` | `314.0s` | `107,516` | `1,324,145` | `5,476` |
| `wbc-sph10435-t2` | `sphinx-doc__sphinx-10435` | `t2` | **`+ Whole Body Compute`** | ✅ PASS | `31` | `12` | `452.0s` | `122,454` | `1,927,707` | `6,252` |
| `wbc-sph10435-t3` | `sphinx-doc__sphinx-10435` | `t3` | **`+ Whole Body Compute`** | ✅ PASS | `32` | `22` | `306.0s` | `103,474` | `1,259,479` | `5,075` |
| `van-dja17084-t1` | `django__django-17084` | `t1` | Vanilla | ✅ PASS | `44` | `22` | `429.0s` | `172,247` | `3,800,193` | `10,905` |
| `van-dja17084-t2` | `django__django-17084` | `t2` | Vanilla | ✅ PASS | `78` | `63` | `837.0s` | `273,644` | `9,387,200` | `16,508` |
| `van-dja17084-t3` | `django__django-17084` | `t3` | Vanilla | ✅ PASS | `65` | `37` | `758.0s` | `215,127` | `5,559,962` | `14,946` |
| `wbc-dja17084-t1` | `django__django-17084` | `t1` | **`+ Whole Body Compute`** | ✅ PASS | `54` | `37` | `733.0s` | `156,119` | `3,045,651` | `13,662` |
| `wbc-dja17084-t2` | `django__django-17084` | `t2` | **`+ Whole Body Compute`** | ✅ PASS | `83` | `48` | `1508.0s` | `250,780` | `6,769,401` | `19,198` |
| `wbc-dja17084-t3` | `django__django-17084` | `t3` | **`+ Whole Body Compute`** | ✅ PASS | `30` | `26` | `318.0s` | `98,482` | `1,355,472` | `8,011` |
| `van-mpl25960-t1` | `matplotlib__matplotlib-25960` | `t1` | Vanilla | ✅ PASS | `138` | `74` | `2383.0s` | `906,510` | `23,108,355` | `59,217` |
| `van-mpl25960-t2` | `matplotlib__matplotlib-25960` | `t2` | Vanilla | ✅ PASS | `130` | `90` | `1754.0s` | `638,825` | `21,911,176` | `36,248` |
| `van-mpl25960-t3` | `matplotlib__matplotlib-25960` | `t3` | Vanilla | ✅ PASS | `163` | `44` | `2394.0s` | `861,841` | `23,338,087` | `43,740` |
| `wbc-mpl25960-t1` | `matplotlib__matplotlib-25960` | `t1` | **`+ Whole Body Compute`** | ✅ PASS | `119` | `45` | `1478.0s` | `368,886` | `14,848,727` | `25,974` |
| `wbc-mpl25960-t2` | `matplotlib__matplotlib-25960` | `t2` | **`+ Whole Body Compute`** | ✅ PASS | `112` | `40` | `1802.0s` | `528,311` | `15,148,234` | `50,183` |
| `wbc-mpl25960-t3` | `matplotlib__matplotlib-25960` | `t3` | **`+ Whole Body Compute`** | ✅ PASS | `48` | `33` | `1844.0s` | `300,341` | `7,168,665` | `139,053` |
