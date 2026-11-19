# Upstream `princeton-nlp/SWE-bench_Verified` on Claude Code (`2,000L` Ceiling, `N=100` Official Repos)
**Date**: `2026-11-18T13:16:28.082131`  
**Harness**: `Claude Code CLI` (`2,000`-line default `Read` ceiling, held-out official `test_patch` verification)  
**Model**: `Claude Haiku` (`150` live isolated evaluations)  

## 1. Executive Cross-Harness Scorecard

| Metric | Vanilla Claude Code | `+ Whole Body Compute` | Delta |
| :--- | :---: | :---: | :---: |
| **`Pass@1` Resolution** | `39/50` (`78.0%`) | **`74/100` (`74.0%`)** | `-4.0pp` |
| **Total Context Ingested** | `74,539,913` | **`101,839,332`** | **`-14.4%`** (Median `-10.1%`) |
| **Billed Input Tokens** | `2,080,733` | **`3,565,146`** | **`-3.0%`** (Median `-1.1%`) |
| **API-Equivalent Cost (`USD`)** | `$14.73` | **`$21.64`** | **`-11.0%`** |
| **Planner Turns** | `1,679` | **`2,464`** | **`-11.3%`** |
| **Wall Time (`s`)** | `8008.1s` | **`11416.7s`** | **`-1.3%`** |
| **`>2,000L` Monster Repos Context** | `17,184,624` | **`16,438,567`** | **`-4.3%`** (Billed `-7.5%`, Cost `-7.7%`) |

## 2. Per-Repository Head-to-Head Breakdown

| Official Instance | Primary File (`Lines`) | Vanilla (`Pass`, Turns, Ctx, Cost) | `+ Whole Body Compute` (`Pass`, Turns, Ctx, Cost) | Context Δ | Cost Δ |
| :--- | :--- | :--- | :--- | :---: | :---: |
| `psf__requests-6028` | `requests/utils.py` (`1,056L`) | ❌ `62t`, `4,142,929`, `$0.68` | ✅ `28t`, `919,898`, `$0.19` | **`-77.8%`** | **`-72.4%`** |
| `pytest-dev__pytest-8399` | `src/_pytest/python.py` (`1,701L`) | ✅ `18t`, `751,704`, `$0.17` | ✅ `12t`, `466,182`, `$0.13` | **`-38.0%`** | **`-25.0%`** |
| `pydata__xarray-6721` | `xarray/core/common.py` (`2,075L`) | ✅ `32t`, `1,878,766`, `$0.37` | ✅ `34t`, `1,117,072`, `$0.22` | **`-40.5%`** | **`-41.9%`** |
| `sympy__sympy-24562` | `sympy/core/numbers.py` (`4,496L`) | ✅ `20t`, `730,008`, `$0.24` | ✅ `21t`, `757,556`, `$0.22` | **`+3.8%`** | **`-7.9%`** |
| `pallets__flask-5014` | `src/flask/blueprints.py` (`621L`) | ✅ `17t`, `534,098`, `$0.12` | ✅ `14t`, `521,737`, `$0.13` | **`-2.3%`** | **`+11.5%`** |
| `mwaskom__seaborn-3069` | `seaborn/_core/plot.py` (`1,649L`) | ❌ `66t`, `3,165,766`, `$0.56` | ✅ `38t`, `1,528,386`, `$0.30` | **`-51.7%`** | **`-46.2%`** |
| `pylint-dev__pylint-6528` | `pylint/lint/pylinter.py` (`1,188L`) | ✅ `108t`, `6,519,965`, `$0.99` | ✅ `58t`, `2,848,529`, `$0.48` | **`-56.3%`** | **`-51.7%`** |
| `sphinx-doc__sphinx-10435` | `sphinx/writers/latex.py` (`2,061L`) | ✅ `32t`, `1,952,740`, `$0.40` | ✅ `17t`, `492,750`, `$0.15` | **`-74.8%`** | **`-63.6%`** |
| `django__django-17084` | `django/db/models/sql/query.py` (`2,678L`) | ✅ `70t`, `3,032,807`, `$0.51` | ❌ `66t`, `3,965,757`, `$0.64` | **`+30.8%`** | **`+26.1%`** |
| `matplotlib__matplotlib-25960` | `lib/matplotlib/figure.py` (`3,618L`) | ✅ `34t`, `1,407,095`, `$0.37` | ✅ `44t`, `1,921,476`, `$0.43` | **`+36.6%`** | **`+16.8%`** |
| `scikit-learn__scikit-learn-25102` | `sklearn/base.py` (`1,081L`) | ✅ `43t`, `1,870,260`, `$0.35` | ✅ `42t`, `1,810,883`, `$0.35` | **`-3.2%`** | **`+1.4%`** |
| `astropy__astropy-13236` | `astropy/table/table.py` (`3,972L`) | ✅ `25t`, `801,425`, `$0.19` | ✅ `34t`, `1,325,018`, `$0.28` | **`+65.3%`** | **`+47.1%`** |
| `django__django-16454` | `django/core/management/base.py` (`678L`) | ❌ `8t`, `206,715`, `$0.07` | ✅ `24t`, `727,848`, `$0.17` | **`+252.1%`** | **`+134.6%`** |
| `django__django-15731` | `django/db/models/manager.py` (`213L`) | ✅ `7t`, `160,025`, `$0.05` | ✅ `7t`, `163,239`, `$0.05` | **`+2.0%`** | **`+3.9%`** |
| `sympy__sympy-23824` | `sympy/physics/hep/gamma_matrices.py` (`717L`) | ✅ `11t`, `348,898`, `$0.12` | ✅ `10t`, `277,128`, `$0.09` | **`-20.6%`** | **`-25.4%`** |
| `sphinx-doc__sphinx-10323` | `sphinx/directives/code.py` (`461L`) | ✅ `22t`, `829,934`, `$0.18` | ✅ `16t`, `593,300`, `$0.15` | **`-28.5%`** | **`-16.6%`** |
| `matplotlib__matplotlib-25287` | `lib/matplotlib/axis.py` (`2,742L`) | ✅ `21t`, `558,701`, `$0.14` | ✅ `20t`, `552,540`, `$0.13` | **`-1.1%`** | **`-7.6%`** |
| `scikit-learn__scikit-learn-25232` | `sklearn/impute/_iterative.py` (`889L`) | ✅ `38t`, `1,979,743`, `$0.34` | ✅ `46t`, `2,572,534`, `$0.48` | **`+29.9%`** | **`+38.0%`** |
| `astropy__astropy-14182` | `astropy/io/ascii/rst.py` (`66L`) | ❌ `11t`, `271,831`, `$0.07` | ❌ `25t`, `917,306`, `$0.23` | **`+237.4%`** | **`+216.6%`** |
| `pydata__xarray-7229` | `xarray/core/computation.py` (`2,151L`) | ❌ `69t`, `3,280,069`, `$0.56` | ❌ `55t`, `2,860,082`, `$0.51` | **`-12.8%`** | **`-7.8%`** |
| `pytest-dev__pytest-10051` | `src/_pytest/logging.py` (`826L`) | ✅ `12t`, `424,435`, `$0.13` | ✅ `12t`, `452,056`, `$0.17` | **`+6.5%`** | **`+31.1%`** |
| `pylint-dev__pylint-8898` | `pylint/config/argument.py` (`502L`) | ✅ `30t`, `1,208,984`, `$0.24` | ✅ `29t`, `1,185,418`, `$0.24` | **`-1.9%`** | **`+0.1%`** |
| `django__django-16485` | `django/template/defaultfilters.py` (`978L`) | ✅ `25t`, `767,265`, `$0.17` | ✅ `20t`, `653,051`, `$0.17` | **`-14.9%`** | **`+0.7%`** |
| `django__django-15732` | `django/db/backends/base/schema.py` (`1,724L`) | ✅ `59t`, `2,682,153`, `$0.45` | ✅ `58t`, `2,715,316`, `$0.46` | **`+1.2%`** | **`+2.4%`** |
| `sympy__sympy-23950` | `sympy/sets/contains.py` (`48L`) | ✅ `39t`, `1,367,874`, `$0.28` | ✅ `42t`, `1,563,408`, `$0.29` | **`+14.3%`** | **`+4.4%`** |
| `sphinx-doc__sphinx-11445` | `sphinx/util/rst.py` (`110L`) | ✅ `28t`, `896,013`, `$0.20` | ✅ `8t`, `215,055`, `$0.08` | **`-76.0%`** | **`-58.8%`** |
| `matplotlib__matplotlib-25311` | `lib/matplotlib/offsetbox.py` (`1,621L`) | ✅ `49t`, `1,993,149`, `$0.38` | ✅ `48t`, `2,011,015`, `$0.40` | **`+0.9%`** | **`+6.9%`** |
| `scikit-learn__scikit-learn-25747` | `sklearn/utils/_set_output.py` (`279L`) | ❌ `16t`, `491,951`, `$0.13` | ❌ `11t`, `308,699`, `$0.09` | **`-37.2%`** | **`-31.5%`** |
| `pydata__xarray-7233` | `xarray/core/rolling.py` (`1,151L`) | ✅ `16t`, `584,195`, `$0.16` | ✅ `12t`, `325,431`, `$0.10` | **`-44.3%`** | **`-37.1%`** |
| `pytest-dev__pytest-10081` | `src/_pytest/unittest.py` (`414L`) | ✅ `23t`, `1,113,982`, `$0.25` | ✅ `33t`, `1,177,253`, `$0.24` | **`+5.7%`** | **`-5.7%`** |
| `pylint-dev__pylint-6903` | `pylint/lint/run.py` (`232L`) | ✅ `8t`, `193,829`, `$0.06` | ✅ `7t`, `169,364`, `$0.06` | **`-12.6%`** | **`-0.8%`** |
| `django__django-15814` | `django/db/models/sql/query.py` (`2,693L`) | ✅ `12t`, `312,682`, `$0.10` | ✅ `33t`, `1,182,004`, `$0.24` | **`+278.0%`** | **`+144.5%`** |
| `django__django-15741` | `django/utils/formats.py` (`310L`) | ✅ `34t`, `1,152,643`, `$0.22` | ✅ `36t`, `1,234,741`, `$0.23` | **`+7.1%`** | **`+3.9%`** |
| `django__django-15851` | `django/db/backends/postgresql/client.py` (`64L`) | ✅ `13t`, `324,052`, `$0.09` | ✅ `12t`, `297,095`, `$0.08` | **`-8.3%`** | **`-8.6%`** |
| `django__django-15863` | `django/template/defaultfilters.py` (`978L`) | ✅ `13t`, `445,070`, `$0.13` | ✅ `13t`, `419,983`, `$0.12` | **`-5.6%`** | **`-12.6%`** |
| `django__django-15916` | `django/forms/models.py` (`1,666L`) | ❌ `31t`, `1,612,428`, `$0.35` | ❌ `27t`, `1,453,718`, `$0.36` | **`-9.8%`** | **`+4.3%`** |
| `django__django-15930` | `django/db/models/expressions.py` (`1,803L`) | ✅ `66t`, `2,985,138`, `$0.52` | ✅ `36t`, `1,968,412`, `$0.39` | **`-34.1%`** | **`-24.3%`** |
| `django__django-15957` | `django/db/models/fields/related_descriptors.py` (`1,419L`) | ✅ `66t`, `3,038,528`, `$0.51` | ❌ `86t`, `5,194,176`, `$0.82` | **`+70.9%`** | **`+61.2%`** |
| `django__django-15973` | `django/db/migrations/autodetector.py` (`1,736L`) | ❌ `44t`, `1,509,454`, `$0.28` | ❌ `23t`, `636,610`, `$0.13` | **`-57.8%`** | **`-52.2%`** |
| `django__django-16032` | `django/db/models/fields/related_lookups.py` (`210L`) | ✅ `80t`, `5,211,891`, `$0.97` | ✅ `45t`, `2,185,599`, `$0.38` | **`-58.1%`** | **`-60.5%`** |
| `sympy__sympy-23262` | `sympy/utilities/lambdify.py` (`1,402L`) | ✅ `33t`, `1,168,651`, `$0.23` | ✅ `25t`, `832,690`, `$0.18` | **`-28.8%`** | **`-19.4%`** |
| `sympy__sympy-23413` | `sympy/polys/matrices/normalforms.py` (`403L`) | ❌ `34t`, `1,884,864`, `$0.54` | ❌ `22t`, `1,178,505`, `$0.48` | **`-37.5%`** | **`-11.0%`** |
| `sympy__sympy-23534` | `sympy/core/symbol.py` (`925L`) | ✅ `17t`, `596,408`, `$0.15` | ✅ `16t`, `587,068`, `$0.15` | **`-1.6%`** | **`+1.9%`** |
| `sympy__sympy-24066` | `sympy/physics/units/unitsystem.py` (`206L`) | ✅ `23t`, `779,834`, `$0.20` | ✅ `19t`, `823,883`, `$0.25` | **`+5.7%`** | **`+25.0%`** |
| `matplotlib__matplotlib-25332` | `lib/matplotlib/cbook.py` (`2,278L`) | ✅ `38t`, `1,348,367`, `$0.26` | ✅ `31t`, `1,128,503`, `$0.23` | **`-16.3%`** | **`-11.9%`** |
| `matplotlib__matplotlib-25479` | `lib/matplotlib/cm.py` (`723L`) | ❌ `17t`, `483,083`, `$0.12` | ❌ `22t`, `762,839`, `$0.18` | **`+57.9%`** | **`+54.8%`** |
| `django__django-16082` | `django/db/models/expressions.py` (`1,835L`) | ✅ `19t`, `597,020`, `$0.14` | ✅ `17t`, `463,096`, `$0.11` | **`-22.4%`** | **`-21.7%`** |
| `pydata__xarray-6938` | `xarray/core/dataset.py` (`8,882L`) | ✅ `46t`, `1,881,964`, `$0.40` | ✅ `35t`, `1,135,809`, `$0.21` | **`-39.6%`** | **`-46.7%`** |
| `pydata__xarray-6744` | `xarray/core/rolling.py` (`1,144L`) | ✅ `16t`, `609,723`, `$0.17` | ✅ `37t`, `1,413,755`, `$0.32` | **`+131.9%`** | **`+89.8%`** |
| `pytest-dev__pytest-10356` | `src/_pytest/mark/structures.py` (`590L`) | ❌ `58t`, `2,450,804`, `$0.41` | ❌ `64t`, `3,751,487`, `$0.57` | **`+53.1%`** | **`+39.6%`** |
| `django__django-16100` | `django/contrib/admin/options.py` (`2,498L`) |  - *(WBC-Only)* | ✅ `26t`, `871,638`, `$0.20` |  - | - |
| `django__django-16333` | `django/contrib/auth/forms.py` (`487L`) |  - *(WBC-Only)* | ✅ `23t`, `822,662`, `$0.15` |  - | - |
| `django__django-16136` | `django/views/generic/base.py` (`275L`) |  - *(WBC-Only)* | ✅ `7t`, `179,635`, `$0.06` |  - | - |
| `django__django-16139` | `django/contrib/auth/forms.py` (`485L`) |  - *(WBC-Only)* | ✅ `7t`, `188,820`, `$0.07` |  - | - |
| `django__django-16145` | `django/core/management/commands/runserver.py` (`183L`) |  - *(WBC-Only)* | ❌ `9t`, `232,853`, `$0.06` |  - | - |
| `django__django-16255` | `django/contrib/sitemaps/__init__.py` (`258L`) |  - *(WBC-Only)* | ✅ `6t`, `145,973`, `$0.06` |  - | - |
| `django__django-16256` | `django/contrib/contenttypes/fields.py` (`766L`) |  - *(WBC-Only)* | ❌ `26t`, `1,312,689`, `$0.27` |  - | - |
| `django__django-16263` | `django/db/models/expressions.py` (`1,849L`) |  - *(WBC-Only)* | ❌ `54t`, `2,161,523`, `$0.41` |  - | - |
| `django__django-16315` | `django/db/models/query.py` (`2,650L`) |  - *(WBC-Only)* | ✅ `21t`, `562,580`, `$0.14` |  - | - |
| `sympy__sympy-22080` | `sympy/printing/codeprinter.py` (`810L`) |  - *(WBC-Only)* | ❌ `63t`, `4,035,749`, `$0.69` |  - | - |
| `sympy__sympy-22456` | `sympy/codegen/ast.py` (`1,863L`) |  - *(WBC-Only)* | ✅ `47t`, `2,119,388`, `$0.39` |  - | - |
| `sympy__sympy-22714` | `sympy/geometry/point.py` (`1,376L`) |  - *(WBC-Only)* | ✅ `11t`, `366,685`, `$0.11` |  - | - |
| `sympy__sympy-22914` | `sympy/printing/pycode.py` (`640L`) |  - *(WBC-Only)* | ✅ `6t`, `163,142`, `$0.06` |  - | - |
| `django__django-16429` | `django/utils/timesince.py` (`137L`) |  - *(WBC-Only)* | ✅ `5t`, `117,581`, `$0.05` |  - | - |
| `matplotlib__matplotlib-26113` | `lib/matplotlib/axes/_axes.py` (`8,451L`) |  - *(WBC-Only)* | ✅ `7t`, `181,843`, `$0.08` |  - | - |
| `pydata__xarray-6461` | `xarray/core/computation.py` (`1,997L`) |  - *(WBC-Only)* | ✅ `43t`, `1,569,206`, `$0.33` |  - | - |
| `pydata__xarray-6599` | `xarray/core/computation.py` (`2,070L`) |  - *(WBC-Only)* | ❌ `26t`, `1,488,234`, `$0.32` |  - | - |
| `django__django-16493` | `django/db/models/fields/files.py` (`510L`) |  - *(WBC-Only)* | ✅ `6t`, `149,824`, `$0.06` |  - | - |
| `django__django-16502` | `django/core/servers/basehttp.py` (`246L`) |  - *(WBC-Only)* | ❌ `23t`, `853,024`, `$0.22` |  - | - |
| `django__django-16527` | `django/contrib/admin/templatetags/admin_modify.py` (`150L`) |  - *(WBC-Only)* | ✅ `9t`, `230,914`, `$0.07` |  - | - |
| `django__django-16560` | `django/db/models/constraints.py` (`405L`) |  - *(WBC-Only)* | ✅ `50t`, `2,682,291`, `$0.46` |  - | - |
| `django__django-16569` | `django/forms/formsets.py` (`575L`) |  - *(WBC-Only)* | ✅ `6t`, `132,638`, `$0.05` |  - | - |
| `django__django-16595` | `django/db/migrations/operations/fields.py` (`357L`) |  - *(WBC-Only)* | ✅ `25t`, `1,070,904`, `$0.23` |  - | - |
| `django__django-16612` | `django/contrib/admin/sites.py` (`611L`) |  - *(WBC-Only)* | ✅ `5t`, `117,073`, `$0.05` |  - | - |
| `django__django-16631` | `django/contrib/auth/__init__.py` (`230L`) |  - *(WBC-Only)* | ❌ `40t`, `1,699,843`, `$0.31` |  - | - |
| `django__django-16642` | `django/http/response.py` (`749L`) |  - *(WBC-Only)* | ❌ `13t`, `430,218`, `$0.13` |  - | - |
| `django__django-16661` | `django/contrib/admin/options.py` (`2,513L`) |  - *(WBC-Only)* | ✅ `30t`, `999,110`, `$0.22` |  - | - |
| `django__django-16662` | `django/db/migrations/writer.py` (`311L`) |  - *(WBC-Only)* | ✅ `20t`, `785,572`, `$0.19` |  - | - |
| `django__django-16667` | `django/forms/widgets.py` (`1,171L`) |  - *(WBC-Only)* | ❌ `9t`, `216,385`, `$0.07` |  - | - |
| `sympy__sympy-24213` | `sympy/physics/units/unitsystem.py` (`205L`) |  - *(WBC-Only)* | ✅ `8t`, `219,755`, `$0.08` |  - | - |
| `sympy__sympy-24443` | `sympy/combinatorics/homomorphisms.py` (`560L`) |  - *(WBC-Only)* | ✅ `15t`, `483,974`, `$0.14` |  - | - |
| `sympy__sympy-24539` | `sympy/polys/rings.py` (`2,467L`) |  - *(WBC-Only)* | ✅ `6t`, `134,813`, `$0.05` |  - | - |
| `sympy__sympy-24661` | `sympy/parsing/sympy_parser.py` (`1,241L`) |  - *(WBC-Only)* | ✅ `19t`, `595,884`, `$0.15` |  - | - |
| `pydata__xarray-6992` | `xarray/core/dataset.py` (`8,935L`) |  - *(WBC-Only)* | ❌ `14t`, `428,684`, `$0.11` |  - | - |
| `django__django-16801` | `django/db/models/fields/files.py` (`510L`) |  - *(WBC-Only)* | ✅ `11t`, `331,464`, `$0.10` |  - | - |
| `django__django-16819` | `django/db/migrations/operations/models.py` (`1,137L`) |  - *(WBC-Only)* | ✅ `32t`, `1,535,672`, `$0.30` |  - | - |
| `django__django-16877` | `django/template/defaultfilters.py` (`983L`) |  - *(WBC-Only)* | ❌ `22t`, `708,601`, `$0.16` |  - | - |
| `django__django-16899` | `django/contrib/admin/checks.py` (`1,353L`) |  - *(WBC-Only)* | ✅ `28t`, `843,768`, `$0.18` |  - | - |
| `django__django-16901` | `django/db/models/sql/where.py` (`355L`) |  - *(WBC-Only)* | ✅ `23t`, `754,386`, `$0.18` |  - | - |
| `django__django-16938` | `django/core/serializers/python.py` (`190L`) |  - *(WBC-Only)* | ❌ `5t`, `121,608`, `$0.05` |  - | - |
| `django__django-16950` | `django/forms/models.py` (`1,668L`) |  - *(WBC-Only)* | ❌ `24t`, `934,925`, `$0.29` |  - | - |
| `django__django-17029` | `django/apps/registry.py` (`436L`) |  - *(WBC-Only)* | ✅ `7t`, `177,258`, `$0.06` |  - | - |
| `django__django-17087` | `django/db/migrations/serializer.py` (`400L`) |  - *(WBC-Only)* | ✅ `5t`, `123,665`, `$0.05` |  - | - |
| `django__django-15987` | `django/core/management/commands/loaddata.py` (`432L`) |  - *(WBC-Only)* | ❌ `8t`, `256,620`, `$0.10` |  - | - |
| `django__django-16116` | `django/core/management/commands/makemigrations.py` (`511L`) |  - *(WBC-Only)* | ✅ `23t`, `849,641`, `$0.18` |  - | - |
| `sympy__sympy-21379` | `sympy/core/mod.py` (`239L`) |  - *(WBC-Only)* | ✅ `7t`, `204,339`, `$0.07` |  - | - |
| `sympy__sympy-21596` | `sympy/sets/handlers/intersection.py` (`490L`) |  - *(WBC-Only)* | ❌ `38t`, `1,428,643`, `$0.32` |  - | - |
| `sympy__sympy-21612` | `sympy/printing/str.py` (`997L`) |  - *(WBC-Only)* | ✅ `10t`, `289,396`, `$0.09` |  - | - |
| `sympy__sympy-21847` | `sympy/polys/monomials.py` (`636L`) |  - *(WBC-Only)* | ✅ `7t`, `196,863`, `$0.07` |  - | - |
| `sympy__sympy-21930` | `sympy/physics/secondquant.py` (`3,106L`) |  - *(WBC-Only)* | ❌ `39t`, `1,566,116`, `$0.30` |  - | - |

## 3. Bicameral Routing Cascade Across `N=100` (`Haiku 4.5 + WBC` Primary -> `Sonnet + WBC` Escalation)

Following the **WBC-Only Bicameral Routing** strategy (`--wbc-only` from Shard 2 onward), all `100` official `princeton-nlp/SWE-bench_Verified` instances (`N=1..100`) were evaluated first with `Claude Haiku 4.5 + Whole Body Compute`, and only the `26` unresolved tasks (`10` in Shard 1 + `16` in Shard 2) were escalated to `Claude Sonnet + Whole Body Compute`:

| Stage / Configuration | Instances Evaluated | Resolved (`Pass@1`) | Total Context Tokens | Total Cost (USD) | Cost / Task | Cost / Resolved |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Shard 1 (`N=1..50`) `Haiku 4.5 + WBC` -> `Sonnet + WBC`** | `50` (`+10` esc) | **`49 / 50` (`98.0%`)** | `71,491,203` | `$16.92` | `$0.338` | `$0.345` |
| **Shard 2 (`N=51..100`) `Haiku 4.5 + WBC` -> `Sonnet + WBC`** | `50` (`+16` esc) | **`38 / 50` (`76.0%`)** | `48,966,109` | `$14.13` | `$0.283` | `$0.372` |
| **Combined `N=100` `Haiku 4.5 + WBC` (Stage 1 Primary Only)** | `100` | **`74 / 100` (`74.0%`)** | `101,839,332` | `$21.64` | `$0.216` | `$0.292` |
| **Combined `N=100` `Sonnet + WBC` (Stage 2 Unresolved Escalations)** | `26` | **`+13 / 26` (`50.0%` rescue)** | `18,617,980` | `$9.40` | `$0.362` | `$0.723` |
| **Combined `N=100` Bicameral Total (`Haiku 4.5 + WBC` -> `Sonnet + WBC`)** | **`100` (`126` runs)** | **`87 / 100` (`87.0%`)** | **`120,457,312`** | **`$31.04`** | **`$0.310`** | **`$0.357`** |

### All 26 `Sonnet + WBC` Escalation Runs (`N=1..100`)

| # | Instance | Shard | `Sonnet + WBC` Result | Turns | Context Tokens | Cost (USD) | Wall Time |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `dja17084` (`django__django-17084`) | Shard 1 (1-50) | **PASS** | 18 | 685,232 | $0.3040 | 80.9s |
| 2 | `ast14182` (`astropy__astropy-14182`) | Shard 1 (1-50) | **PASS** | 11 | 414,993 | $0.2729 | 92.2s |
| 3 | `xar7229` (`pydata__xarray-7229`) | Shard 1 (1-50) | **PASS** | 29 | 1,348,860 | $0.5854 | 201.6s |
| 4 | `skl25747` (`scikit-learn__scikit-learn-25747`) | Shard 1 (1-50) | **PASS** | 26 | 1,188,985 | $0.4912 | 137.3s |
| 5 | `dja15916` (`django__django-15916`) | Shard 1 (1-50) | **PASS** | 12 | 487,095 | $0.4528 | 208.8s |
| 6 | `dja15957` (`django__django-15957`) | Shard 1 (1-50) | **PASS** | 32 | 1,455,039 | $0.5601 | 151.3s |
| 7 | `dja15973` (`django__django-15973`) | Shard 1 (1-50) | **PASS** | 8 | 272,989 | $0.1777 | 51.8s |
| 8 | `sym23413` (`sympy__sympy-23413`) | Shard 1 (1-50) | **PASS** | 23 | 991,932 | $0.5287 | 224.0s |
| 9 | `mpl25479` (`matplotlib__matplotlib-25479`) | Shard 1 (1-50) | **FAIL** | 7 | 233,103 | $0.1426 | 35.1s |
| 10 | `pyt10356` (`pytest-dev__pytest-10356`) | Shard 1 (1-50) | **PASS** | 17 | 649,084 | $0.2941 | 80.6s |
| 11 | `dja16145` (`django__django-16145`) | Shard 2 (51-100) | **PASS** | 7 | 220,530 | $0.1384 | 36.5s |
| 12 | `dja16256` (`django__django-16256`) | Shard 2 (51-100) | **FAIL** | 12 | 402,369 | $0.2040 | 52.6s |
| 13 | `dja16263` (`django__django-16263`) | Shard 2 (51-100) | **PASS** | 29 | 1,616,617 | $0.6987 | 322.8s |
| 14 | `sym22080` (`sympy__sympy-22080`) | Shard 2 (51-100) | **FAIL** | 40 | 2,071,770 | $0.8471 | 308.2s |
| 15 | `xar6599` (`pydata__xarray-6599`) | Shard 2 (51-100) | **PASS** | 9 | 301,299 | $0.1784 | 59.9s |
| 16 | `dja16502` (`django__django-16502`) | Shard 2 (51-100) | **FAIL** | 5 | 159,205 | $0.1105 | 20.6s |
| 17 | `dja16631` (`django__django-16631`) | Shard 2 (51-100) | **FAIL** | 14 | 528,863 | $0.2717 | 71.4s |
| 18 | `dja16642` (`django__django-16642`) | Shard 2 (51-100) | **PASS** | 9 | 270,299 | $0.1495 | 45.4s |
| 19 | `dja16667` (`django__django-16667`) | Shard 2 (51-100) | **FAIL** | 7 | 205,777 | $0.1084 | 25.1s |
| 20 | `xar6992` (`pydata__xarray-6992`) | Shard 2 (51-100) | **FAIL** | 10 | 337,919 | $0.1910 | 63.5s |
| 21 | `dja16877` (`django__django-16877`) | Shard 2 (51-100) | **FAIL** | 12 | 373,783 | $0.1903 | 64.6s |
| 22 | `dja16938` (`django__django-16938`) | Shard 2 (51-100) | **FAIL** | 5 | 158,058 | $0.1108 | 23.2s |
| 23 | `dja16950` (`django__django-16950`) | Shard 2 (51-100) | **FAIL** | 14 | 531,481 | $0.3240 | 127.4s |
| 24 | `dja15987` (`django__django-15987`) | Shard 2 (51-100) | **FAIL** | 11 | 334,414 | $0.1663 | 48.7s |
| 25 | `sym21596` (`sympy__sympy-21596`) | Shard 2 (51-100) | **FAIL** | 40 | 2,585,672 | $1.3284 | 571.8s |
| 26 | `sym21930` (`sympy__sympy-21930`) | Shard 2 (51-100) | **FAIL** | 15 | 500,119 | $0.2758 | 119.0s |
