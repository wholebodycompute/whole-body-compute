# Whole Body Compute (`body`)

**Cut agent token burn and latency with zero accuracy loss on SWE-bench Verified.**  
Whole Body Compute makes AI computing simpler, faster, and local: giving Claude Code, Cursor, Windsurf, Cline, Goose, and Antigravity a `<3ms` AST context shield, native macOS vision, and on-device memory right on your Mac (with cross-platform Blood-Brain Barrier & FinOps audit support on Linux).

[Open-Core Spec (`wholebodycompute.com/open-core`)](https://wholebodycompute.com/open-core) · [Zero-Install Spend Audit (`audit.py`)](https://wholebodycompute.com/audit.py) · [Commercial Platform (`wholebodycompute.com`)](https://wholebodycompute.com) · [Benchmarks (`benchmarks/`)](https://github.com/wholebodycompute/whole-body-compute/blob/main/benchmarks/README.md) · [Architecture Wiki](https://github.com/wholebodycompute/whole-body-compute/blob/main/docs/WIKI_AND_GLOSSARY.md)

---

## Quickstart

```bash
# Option 1a: 1-line installer (macOS, Linux & Windows WSL2 Ubuntu)
curl -fsSL https://wholebodycompute.com/begin | sh

# Option 1b: 1-line installer (Windows PowerShell)
irm https://wholebodycompute.com/begin.ps1 | iex

# Option 2: uv / pipx directly from PyPI or GitHub
uv tool install whole-body-compute && body begin

# Option 3: Source checkout
git clone https://github.com/wholebodycompute/whole-body-compute.git
cd whole-body-compute
python3 -m venv .venv && source .venv/bin/activate && pip install -e .
body begin
```

Or run a read-only 30-day token spend audit across your local agent logs with zero installation:

```bash
curl -sSL https://wholebodycompute.com/audit.py | python3
```

---

## What It Does

Cloud coding agents burn massive context windows re-ingesting full files, noisy test logs, and screenshots on every turn. **Whole Body Compute (`body`)** sits locally between your coding agent and macOS/Linux hardware, intercepting bloated reads and tool outputs in `<3ms` before they hit your cloud billing meter.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                            WHOLE BODY COMPUTE (5 CORE ORGANS)                          │
├──────────────────────────┬──────────────────────────┬──────────────────────────────────┤
│ 01. THE HEART            │ 02. THE BRAINSTEM        │ 03. THE EYES & HANDS             │
│ Thermal, Perfusion & CSI │ Local Metal, Router & KV │ Foveal AX DOM, Tactile & Vagus   │
├──────────────────────────┼──────────────────────────┴──────────────────────────────────┤
│ 04. THE HIPPOCAMPUS      │ 05. THE IMMUNE SYSTEM (BLOOD-BRAIN BARRIER)                 │
│ Memory WAL, Flush & REM  │ Pre/Post Hooks, SensoryVector, <3ms AST Slicer & SecGate    │
└──────────────────────────┴─────────────────────────────────────────────────────────────┘
```

| Capability | Standard Unshielded Cloud Agent | With Whole Body Compute (`body`) | Measured Impact |
| :--- | :--- | :--- | :--- |
| **Harness Wiring** | Manual per-editor rules & scattered scripts | `body begin` auto-wires `PreToolUse` / `PostToolUse` hooks across Claude Code, Cursor, Antigravity, Windsurf, Cline & Goose | **60-second setup + 30-day FinOps receipt** |
| **File & Log Reads** | Unbounded `read_file` (`2,000+` lines) & raw `pytest`/`cat` dumps | **Blood-Brain Barrier (`<3ms` AST Symbol Slicer)** + local Metal `ReconScout` | **Up to 98.7% token reduction** on large reads |
| **Desktop Vision** | Full-screen PNG loops (`7,650+` tokens/step) or heavy 24/7 video recorders | **Foveal Vision (`AXUIElement` DOM)** + `<50ms` Tactile git/file delta sensor (`<50MB` RAM) | **96.9%–99.5% sensory token reduction** |
| **Session Hygiene** | $O(N^2)$ turn accumulation until context degrades | **Cognitive Sentry (`body context`)** + **StateCapsule Flush (`body flush`)** `[Pro]` | **Zero-loss handoff** in `<30` steps |
| **Pre-Egress Security** | Cloud classifiers run *after* data leaves your machine | `<1ms` 23-vendor secret redaction, `1Password`/Keychain blindfold, `<2ms` APFS CoW airlock & AST SecGate | **`0` secret leaks · `0.0%` ASR** on `AgentDojo` |

---

## Benchmarks (`SWE-bench Verified` & Princeton HAL)

Evaluated across **180+ live unscripted agent sessions** - including **`120` cross-harness `k=3` frontier trials** (`60` `Gemini 4 Carbon` + `60` `Claude Sonnet`) and **`20` `Claude Haiku` runs** on official upstream `princeton-nlp/SWE-bench_Verified` across `10` Python organizations at untouched historical commits, graded strictly against held-out `FAIL_TO_PASS` + `PASS_TO_PASS` test patches:

| Evaluation Suite | Baseline (Unshielded) | `+ Whole Body Compute` | Key Empirical Result |
| :--- | :--- | :--- | :--- |
| **Combined `SWE-bench Verified` (`k=3`, `120` Trials)**<br>`60` `Gemini 4 Carbon` + `60` `Claude Sonnet` | `59/60` (`98.3% Pass@1`)<br>`19/20 Pass^3` · `225.08M` ctx | **`60/60` (`100.0% Pass@1`)**<br>**`20/20 Pass^3` · `159.30M` ctx** | **`100%` strict `3/3` consistency** (`20/20 Pass^3`)<br>**`-29.2%` total context** (`-65.78M` tokens)<br>**`-23.0%` billed input**<br>**`-14.3%` turns** |
| **`SWE-bench Verified` (`k=3`, `60` Trials) - Antigravity**<br>`Gemini 4 Carbon` (`800L` ceiling, `10` official orgs) | `30/30` (`100% Pass@1`)<br>`10/10 Pass^3` · `203.08M` ctx | **`30/30` (`100% Pass@1`)**<br>**`10/10 Pass^3` · `137.23M` ctx** | **Zero accuracy loss** (`100% Pass^3`)<br>**`-32.4%` total context** (`-65.85M` tokens)<br>**`-25.6%` billed input**<br>**`-18.8%` fewer turns** |
| **`SWE-bench Verified` (`k=3`, `60` Trials) - Claude Code**<br>`Claude Sonnet` (`2,000L` ceiling, `10` official orgs) | `29/30` (`96.7% Pass@1`)<br>`9/10 Pass^3` · `$10.47` | **`30/30` (`100% Pass@1`)**<br>**`10/10 Pass^3` · `$10.41`** | **`+10.0pp Pass^3`** (`mwaskom/seaborn-3069` solved `3/3`)<br>**`-12.2%` matched wall time**<br>**`-5.5%` matched ctx**<br>**`-5.3%` matched $** |
| **`SWE-bench Verified` (`Claude Code 2x2`, `40` Runs)**<br>`Claude Sonnet` & `Claude Haiku` CLI (`k=1`) | `Vanilla Sonnet`: `9/10` (`$4.00`)<br>`Vanilla Haiku`: `8/10` (`$4.42`) | **`Sonnet + WBC`: `10/10` (`$3.89`)**<br>**`Haiku + WBC`: `-36.1%` ctx (`$3.05`)** | **`Sonnet + WBC` solved `10/10` (`100%`)**<br>**`-12.0%` LESS cost than `Vanilla Haiku`** (`$3.89` vs. `$4.42`)<br>**`+20.0pp` higher accuracy** vs. `Vanilla Haiku` |
| **Multi-Track Live A/B (`18` Sessions)**<br>`SWE-bench` + `CORE-Bench Hard` + `OSWorld` | `Haiku 4.5`: `9/9` (`100%`)<br>`5.26M` ctx · `$1.60` · `312.9s` | **`Haiku 4.5 + WBC`: `9/9` (`100%`)**<br>**`1.55M` ctx · `$0.51` · `190.5s`** | **`-67.8%` API spend** (`3.11x` efficiency)<br>**`-71.9%` billed input**<br>**`-39.1%` faster wall-clock time** |
| **Security & Reliability (`AgentDojo` + `CyberSecEval 3`)** | `100.0%` Attack Success (`12/12`) | **`0.0%` Attack Success (`12/12` blocked)** | **`0.0%` false-positive rate** (`6/6` benign passed)<br>**`0.102ms` in-process gate overhead** |

Full per-repository tables, `120`-trial cross-harness synthesis ([`CROSS_HARNESS_120_TRIAL_SWEBENCH_K3_REPORT.md`](https://github.com/wholebodycompute/whole-body-compute/blob/main/benchmarks/CROSS_HARNESS_120_TRIAL_SWEBENCH_K3_REPORT.md)), `2x2` model matrices, raw JSON telemetry receipts, and 1-command verification harnesses: **[`benchmarks/README.md`](https://github.com/wholebodycompute/whole-body-compute/blob/main/benchmarks/README.md)**.

---

## CLI Reference (`body`)

```bash
# Core Onboarding, Audit & Organ Primitives (Free Forever)
body begin          # Wire Blood-Brain Barrier hooks & print 30-day FinOps ROI receipt
body audit-spend    # Scan Claude Code, Cursor, Antigravity, Windsurf, Cline, Goose, Codex & Aider spend
body savings        # Inspect live Blood-Brain Barrier token savings & active shield status
body report         # 5-Pillar operational & FinOps scorecard (Session, 24h & 7d trends)
body doctor         # Sub-second 5-organ health check (--exercise runs closed-loop reflex verification)
body heart          # Thermal state, Unified RAM headroom & model perfusion status
body eyes           # Live macOS AXUIElement Foveal DOM snapshot + on-demand Apple Vision OCR
body hands          # Deterministic UI actuation (ax_act) & <50ms tactile git/file delta sensor
body immune         # Blood-Brain Barrier audit log (--window 24h --tools --blocked), APFS CoW & SecGate
body daemon status  # Manage portable macOS LaunchAgents (install / status / logs)

# Sovereign Context, Memory & Autonomous Consolidation [Pro / Team]
body license        # Activate Pro/Team Polar key (--activate <KEY>) or inspect tier quota
body context        # Context Saturation Index (CSI) & bloat attribution (--insights) [Pro]
body flush --auto   # Grounded StateCapsule session handoff & metabolic noise purge [Pro]
body memory         # Persistent Hippocampus SQLite WAL & FTS5 recall (--remember / --recall) [Pro]
body perfusion      # Evict idle local Metal models (--evict) or pre-warm weights (--prewarm) [Pro]
body sleep          # Nocturnal REM consolidation & Radix KV-Cache prefix digest [Pro]
```

Deep-dive component reference and glossary: **[`docs/WIKI_AND_GLOSSARY.md`](https://github.com/wholebodycompute/whole-body-compute/blob/main/docs/WIKI_AND_GLOSSARY.md)**.

---

## License

Free forever for individual developers, researchers, and internal non-competing business use under the [Open-Core Community License (`LICENSE`)](https://github.com/wholebodycompute/whole-body-compute/blob/main/LICENSE) and [`PATENTS.md`](https://github.com/wholebodycompute/whole-body-compute/blob/main/PATENTS.md) - commercial resale, managed hosting, or embedding into competing IDEs, agent harnesses, gateways, or FinOps products requires a separate commercial license (`hello@wholebodycompute.com`).
