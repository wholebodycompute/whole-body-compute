# Whole Body Compute (`body`) - Architecture Wiki, Component Catalog & Feature Glossary

> **The 5-Organ Physiological Runtime underneath every AI agent - optimized for Apple Silicon & Linux.**  
> This Wiki & Glossary is organized by **Anatomical Components (The 5 Core Organs + Fast-Path Subsystems)** and includes a complete **Feature & CLI Command Matrix**, **Local vs. Cloud Execution Breakdown**, **Time & Cost Savings Ledger**, and **A-Z Technical Glossary**.

---

## Table of Contents

1. [System Overview: Why Agents Need a Body](#1-system-overview-why-agents-need-a-body)
2. [Component Catalog (Organized by Anatomical Organ & Subsystem)](#2-component-catalog-organized-by-anatomical-organ--subsystem)
   - [Organ 1 - The Heart & Circulatory System (Hardware, Perfusion & Cognitive Sentry)](#organ-1---the-heart--circulatory-system-hardware-perfusion--cognitive-sentry)
   - [Organ 2 - The Brainstem & Prefrontal Router (Local Metal & Specialist Routing)](#organ-2---the-brainstem--prefrontal-router-local-metal--specialist-routing)
   - [Organ 3 - The Eyes, Hands & Nervous System (Foveal DOM, Tactile Deltas & Vagus Brake)](#organ-3---the-eyes-hands--nervous-system-foveal-dom-tactile-deltas--vagus-brake)
   - [Organ 4 - The Hippocampus & Circadian Loop (Memory WAL, Ambient Discovery & REM Sleep)](#organ-4---the-hippocampus--circadian-loop-memory-wal-ambient-discovery--rem-sleep)
   - [Organ 5 - The Immune System / Blood-Brain Barrier (AST Slicer, Elastic Membrane & SecGate)](#organ-5---the-immune-system--blood-brain-barrier-ast-slicer-elastic-membrane--secgate)
   - [Subsystem 6 - Multi-Agent Git Fast-Path (`body commit` & `body checkpoint`)](#subsystem-6---multi-agent-git-fast-path-body-commit--body-checkpoint)
   - [Subsystem 7 - Onboarding, Daemons & 5-Pillar FinOps Telemetry](#subsystem-7---onboarding-daemons--5-pillar-finops-telemetry)
3. [Deep Dive: `body commit` vs. Standard Agent Commit ("Just Commit")](#3-deep-dive-body-commit-vs-standard-agent-commit-just-commit)
4. [Master Feature & CLI Command Savings Ledger (Local % vs. Cloud Driving)](#4-master-feature--cli-command-savings-ledger-local--vs-cloud-driving)
5. [A-Z Glossary of Whole Body Compute Terms](#5-a-z-glossary-of-whole-body-compute-terms)

---

## 1. System Overview: Why Agents Need a Body

Standard cloud agents ("Brains in a Jar") stream raw 2,000-line source files, unfiltered `git diff` outputs, verbose test logs, and 4K base64 screenshots directly into frontier LLM context windows. Because transformer attention scales quadratically ($O(N^2)$) and re-bills the entire conversation transcript on every turn, unshielded agents suffer from:
1. **The Context Tax**: Re-ingesting 10,000+ tokens of boilerplate, diffs, and log noise on every subsequent turn.
2. **Blind Truncation & Retry Churn**: Hitting IDE tool ceilings (800 lines in Antigravity, 2,000 lines in Claude Code) or oscillating edits without structural AST progress.
3. **Lock Contention**: Colliding on `.git/index.lock` when multiple subagents or terminals work in parallel.

**Whole Body Compute (`body`)** solves this with **Bicameral Compute**: pairing any **Frontier Cloud Prefrontal Cortex** (Claude Code, Antigravity, Cursor, Windsurf, Cline, Goose) with an always-on **Local Apple Silicon Body** (`$0.00` local Metal GPU/Unified RAM, `<5ms` deterministic AST/DOM reflexes) guarded by the **Blood-Brain Barrier (BBB)**.

---

## 2. Component Catalog (Organized by Anatomical Organ & Subsystem)

### Organ 1 - The Heart & Circulatory System (Hardware, Perfusion & Cognitive Sentry)

Governs Apple Silicon thermal state, Unified RAM pressure, model warm/cold lifecycles, and conversation context saturation.

| Feature / Mechanism | CLI Command | Source Module | What Happens Locally vs. Cloud | Time & Cost Savings |
| :--- | :--- | :--- | :--- | :--- |
| **Apple Silicon Thermal & RAM Supervisor** (`ThermalSupervisor`) | `body heart` | [`local/supervisor.py`](../src/whole_body_compute/local/supervisor.py) | **100% Local (`<2ms`)**: Reads `pmset -g therm`, `vm_stat` Unified RAM headroom, swap pressure, and flags runaway tachycardic processes (`>=500MB` or `>=20% CPU`). | Prevents macOS swap thrashing and thermal throttling before launching local or cloud jobs (`$0.00`). |
| **Selective Vascular Perfusion** (`Vasoconstriction` & `Cephalic Prewarm`) | `body perfusion` (`--evict`, `--prewarm`, `--cold-starts`) | [`local/supervisor.py`](../src/whole_body_compute/local/supervisor.py) | **100% Local**: Protects Tier-0 autonomic processes while evicting idle Tier-1/2 specialist models (`14B-70B`) to Darwin's Inactive File Cache; pre-warms models on speech/focus onset. | Reclaims **4-18 GB Unified RAM** in `<50ms`; masks cold-start latency to **`0ms`**. |
| **Circulatory Radix KV-Cache** (`RadixKVCacheManager`) | `body radix-cache` (`--status`) | [`local/radix_cache.py`](../src/whole_body_compute/local/radix_cache.py) | **100% Local (`:9379` / Ollama)**: Hashes and pins static prompt prefixes (`.goosehints`, system prompts, active `StateCapsule`) in Metal Unified Memory. | Cuts local tool-loop Time-To-First-Token (TTFB) from `~65ms` to **`<20ms`** (`$0.00`). |
| **Cognitive Thermal Sentry & CSI** (`CognitiveThermalSentry`) | `body context` (`--all`, `--insights`) | [`organs/cognitive_sentry.py`](../src/whole_body_compute/organs/cognitive_sentry.py) | **100% Local (`<15ms`)**: Parses `transcript.jsonl` to compute the **Context Saturation Index (CSI)** (`GREEN < 0.70`, `YELLOW 0.70-0.99`, `RED >= 1.00`), excluding human reading/thinking pauses, and attributes bloat (`UNFILTERED_OUTPUT_BLOAT`, `TRUNCATION_CASCADE`, `RETRY_CHURN_LOOP`). | Detects quadratic attention decay before the model hallucinates or burns `$1-$5/turn` on bloated 600KB+ threads. |
| **Ultradian Glymphatic Context Flush & Zero-Click Handoff** (`UltradianGlymphaticFlush`) | `body flush --auto` | [`organs/cognitive_glymphatic.py`](../src/whole_body_compute/organs/cognitive_glymphatic.py), [`organs/cognitive_capsules.py`](../src/whole_body_compute/organs/cognitive_capsules.py) | **100% Local (`<250ms`)**: Serializes a grounded `StateCapsule` (`git status --porcelain` + SHA-256 artifact digests + `<500`-token resumption prompt), purges `dot_memory.db` noise, copies to clipboard, and via `ZeroClickHandoffBridge` clicks `"New Conversation"` + pastes/submits into `Message input`. | **98.15% transcript compaction** with **100% state recall**; **4.76x faster** session handoff (`107.7s -> 22.6s`). |

---

### Organ 2 - The Brainstem & Prefrontal Router (Local Metal & Specialist Routing)

Executes high-frequency reconnaissance, code slicing, and surgical 1-file edits on local Apple Silicon Metal GPU (`$0.00`) so the frontier cloud model only handles high-level architecture and multi-file reasoning.

| Feature / Mechanism | CLI Command | Source Module | What Happens Locally vs. Cloud | Time & Cost Savings |
| :--- | :--- | :--- | :--- | :--- |
| **Active Recon Scout** (`ReconScout`) | `body scout <target> -q "<query>"` / `body scout --slice file:Sym` | [`local/scout.py`](../src/whole_body_compute/local/scout.py), [`local/symbols.py`](../src/whole_body_compute/local/symbols.py) | **100% Local (`<0.1ms` RAM ingress, `<3ms` AST slice, `<3s` Metal summary)**: Pre-digests massive logs, tracebacks, or multi-file symbol slices on local Ollama/MLX (`qwen2.5-coder`, `gemma2`). | **85%-98.7% token compression** at **`$0.00` cloud cost**. |
| **Local Metal Implementer** (`ReconImplementer`) | `body implement <file> -i "<instruction>"` | [`local/implementer.py`](../src/whole_body_compute/local/implementer.py) | **100% Local (`1.1s-3.5s` on Metal)**: Extracts the exact target function via `<7ms` deterministic AST slicing, applies the edit locally, verifies AST syntax, and returns only a clean unified git diff. | **98.7% cloud token savings** (`$0.00` inference; cloud sees a 20-line diff instead of a 1,500-line file). |
| **Hardware-Gated Hybrid Router** (`HybridRouter`) | `body router` / `body brainstem` | [`local/router.py`](../src/whole_body_compute/local/router.py) | **100% Local (`<1ms`)**: Routes tasks across `70B -> 32B/26B -> 7B -> 1.5B -> Cloud Frontier` based on real-time Unified RAM headroom, swap pressure, and file mastery score (`body airlock mastery`). | Offloads **67%-85%** of routine tool/recon turns to `$0.00` local Metal. |

---

### Organ 3 - The Eyes, Hands & Nervous System (Foveal DOM, Tactile Deltas & Vagus Brake)

Replaces 24/7 video recording and 8,000-token Retina screenshot loops with `<5ms` event-driven macOS Accessibility DOM inspection, closed-loop tactile verification, and a `<50ms` spoken kill-switch.

| Feature / Mechanism | CLI Command | Source Module | What Happens Locally vs. Cloud | Time & Cost Savings |
| :--- | :--- | :--- | :--- | :--- |
| **Foveal Vision Engine & Warm Socket Daemon** (`FovealVisionEngine`) | `body eyes` (`--app`, `--filter`, `--click`, `--paste`, `--chain`, `--daemon`) | [`integrations/foveal_vision.py`](../src/whole_body_compute/integrations/foveal_vision.py) | **100% Local (`<5ms` via `foveal.sock`)**: Extracts structured interactive `AXUIElement` DOM (buttons, inputs, bounding boxes, Electron/Chromium deep DOM awakening) with prompt-injection sanitization. | **96.9%-99.56% sensory token reduction** (`~50-237` tokens vs. `7,650+` tokens per Retina PNG screenshot) & `<50MB` RAM vs. `2-4GB` for 24/7 video recorders. |
| **Closed-Loop Tactile & Visceral Proprioception** (`TactileSensor`) | `body hands` | [`integrations/tactile.py`](../src/whole_body_compute/integrations/tactile.py), [`organs/proprioception.py`](../src/whole_body_compute/organs/proprioception.py) | **100% Local (`0.09ms-2ms`)**: Computes pre/post SHA-256 + AST structural digests on every file edit (`mechanical_noop`, `net_ast_progress`, assertion preservation). Triggers a Vagus Retry-Churn cue if a file is touched `>=3x` with zero AST progress. | Eliminates 3-10 turn blind edit/retry loops; auto-escalates `AXPress` $\to$ Quartz `CGEvent` click when UI state doesn't change. |
| **Vagus Nerve Brake & Acoustic Cognitive Safety** (`ActiveListeningEngine`) | `body nervous` / `body voice` | [`audio/cognitive_safety.py`](../src/whole_body_compute/audio/cognitive_safety.py), [`integrations/active_listening.py`](../src/whole_body_compute/integrations/active_listening.py) | **100% Local (`<1.5ms` FFT/ZCR/lexical check, `<5ms` `SIGINT`)**: Discriminates intentional abort speech (`"stop"`, `"wait"`, `"cancel"` - while ignoring negations like `"don't stop"`) from backchannels (`"uh-huh"`), ducks audio `-12dB`, kills runaway subprocesses, and rolls back the APFS CoW sandbox. | **`<50ms` emergency stop & instant rollback** before a runaway agent can corrupt files or burn more tokens. |

---

### Organ 4 - The Hippocampus & Circadian Loop (Memory WAL, Ambient Discovery & REM Sleep)

Provides persistent episodic/semantic memory, passive cross-app workflow discovery, and nocturnal LoRA self-distillation.

| Feature / Mechanism | CLI Command | Source Module | What Happens Locally vs. Cloud | Time & Cost Savings |
| :--- | :--- | :--- | :--- | :--- |
| **Memory Hippocampus & Déjà Vu Recall** | `body memory` (`--remember`, `--recall`, `--deja-vu`, `--rewind`) | [`local/dot_memory.py`](../src/whole_body_compute/local/dot_memory.py) | **100% Local (`<10ms` SQLite WAL + Metal 768-d embeddings)**: Stores facts, LGTM edit diffs, file mastery scores, and episodic git timelines; matches recurring stack traces via `--deja-vu`. | Instant recall of prior bug fixes and workspace conventions without re-searching the repo (`$0.00`). |
| **Default-On Passive Ambient Workflow Observer** (`AmbientWorkflowObserver` & `ShadowModeStager`) | `body candidates` (`--discover`, `--promote <id>`, `--as-skill`) | [`integrations/ambient_discovery.py`](../src/whole_body_compute/integrations/ambient_discovery.py) | **100% Local (Background Daemon)**: Passively observes recurring $\ge 3$-step cross-app copy/paste/focus motifs (suppressing `1Password`/`Keychain`/secure fields/PII via `AmbientPrivacyGuard`), logs to `ambient_transitions.jsonl`, and auto-stages YAML recipes in `recipes/candidates/`. | Turns repetitive multi-app human/agent routines into **deterministic `$0.00` Tier-0/1 recipes (`0.37s` execution)** + portable `.skill.zip` bundles. |
| **Nocturnal REM Sleep & LoRA Distillation** (`REMSleepDaemon`) | `body rem-sleep` / `body sleep` | [`local/rem_sleep.py`](../src/whole_body_compute/local/rem_sleep.py) | **100% Local (3:30 AM LaunchAgent)**: Purges duplicate memories, harvests golden error-free turns into ChatML, runs MLX Metal LoRA distillation, and gates promotion on a 30/30 (`100%`) LocalBench pass. | Continuously ratchets local model autonomy overnight at `$0.00` cloud cost. |
| **Sovereign Duet Morning Briefing** (`SovereignDuetBriefingEngine`) | `body brief` / `body duet` (`--speak`, `--approve`) | [`brief.py`](../src/whole_body_compute/brief.py) | **100% Local (`<5ms` fallback or `:9380` Metal)**: Synthesizes window focus, Hippocampus memories, staged workflow candidates, git branch state, and 24h FinOps savings into a `<30s` (`<=75` words) spoken/terminal briefing. | Zero-token morning context alignment + 1-keystroke candidate promotion. |

---

### Organ 5 - The Immune System / Blood-Brain Barrier (AST Slicer, Elastic Membrane & SecGate)

The inline `PreInvocation`, `PreToolUse`, and `PostToolUse` choke point wired into `.agents/hooks.json` and `.claude/settings.local.json`.

| Feature / Mechanism | CLI Command | Source Module | What Happens Locally vs. Cloud | Time & Cost Savings |
| :--- | :--- | :--- | :--- | :--- |
| **Subconscious `SensoryVector` & Interoceptive Elastic Membrane** | `body immune` (`--window 24h --tools --blocked`) | [`organs/proprioception.py`](../src/whole_body_compute/organs/proprioception.py), [`organs/immune.py`](../src/whole_body_compute/organs/immune.py) | **100% Local (`0.004ms-0.10ms`)**: Auto-detects host harness ceilings (`800L` Antigravity / `2,000L` Claude Code / `1,000L` Goose) and homeostatic regime (`GREEN_ELASTIC`, `YELLOW_BALANCED`, `RED_TIGHT`). Injects **0 meta-commentary tokens** on nominal turns while auto-slicing oversized reads (`>800L`/`>2,000L`) into `<3ms` AST Symbol Maps + Predictive Query-Matched AST slices. | **-32.4% cumulative context (`1.48x` efficiency)** and **-25.6% billed input (`1.34x` runway)** across 60 official `SWE-bench_Verified` trials (`100% Pass^3`); **-54.6% $O(N^2)$ context** in the Proprioception Arena. |
| **In-Flight Shell Anti-Bloat Auto-Healing & `<5ms` Traceback Slicer** | Automatic via `body-immune-hook` | [`organs/immune_shell.py`](../src/whole_body_compute/organs/immune_shell.py), [`organs/immune.py`](../src/whole_body_compute/organs/immune.py) | **100% Local (`<5ms`)**: Rewrites unbounded `cat`, `head`, `find`, `tree`, and verbose `pytest` commands into bounded slices or `pytest -q --tb=short`; extracts innermost failing traceback frames + `±20L` AST source slices with `# <--- [CRASH FRAME: L<n>]`. | Slashed debugging turns by **-52.2%** (`23 -> 11` turns), wall time by **-36.3%**, and cost by **-39.5%** in live Claude Code benchmarks. |
| **APFS Copy-on-Write Sandbox & 3-Stage AST SecGate** (`APFSCoWSandbox`) | `body doctor --chaos` | [`integrations/goose_runner.py`](../src/whole_body_compute/integrations/goose_runner.py), [`organs/immune_vascular.py`](../src/whole_body_compute/organs/immune_vascular.py) | **100% Local (`<2ms` APFS `cp -c -R` clone + `<3ms` AST audit)**: Blocks tautological assertions (`assert True`, `x == x`), test deletion (`test_*`), unsafe dynamic execution (`eval`, `getattr(os, "system")`), symlink traversal, and unauthorized egress domains (`VascularTrafficMonitor`). | **`0.0%` Attack Success Rate (`12/12` blocked)** and **`0.0%` False Positive Rate** on `AgentDojo` + `CyberSecEval 3` + 682 production domains. |

---

### Subsystem 6 - Multi-Agent Git Fast-Path (`body commit` & `body checkpoint`)

Race-free, zero-lock Git operations for multi-agent and human-agent pair programming.

| Feature / Mechanism | CLI Command | Source Module | What Happens Locally vs. Cloud | Time & Cost Savings |
| :--- | :--- | :--- | :--- | :--- |
| **Shadow-Index Atomic CAS Commit** (`run_fast_commit`) | `body commit [paths...] [-m "..."]` | [`local/git_fastpath.py`](../src/whole_body_compute/local/git_fastpath.py) | **99%+ Local (`0.4s-1.8s`)**: Stages in a process-private `/tmp/wbc_shadow_idx_*.idx`, scopes to current session touches, runs fail-closed secret + AST + `ruff` gates, synthesizes Conventional Commit messages on-device, updates `HEAD` via atomic `git update-ref` CAS, and logs to `dot_memory.db`. | **Saves 15s-30s (`85-90%` faster)** and **$0.10-$0.43+ per commit** (`-75%` to `-100%` turns; **99.8% less diff bloat** in cloud context). |
| **Zero-Risk Working-Tree Checkpoints** (`run_checkpoint`) | `body checkpoint [label]` (`--list`, `--restore`) | [`local/git_fastpath.py`](../src/whole_body_compute/local/git_fastpath.py) | **100% Local (`<30ms`)**: Snapshots working tree state into `refs/checkpoints/<timestamp>-<label>` via shadow index without disturbing `.git/index` or `HEAD`. | Instant zero-token safety net before risky refactors. |

---

### Subsystem 7 - Onboarding, Daemons & 5-Pillar FinOps Telemetry

| Feature / Mechanism | CLI Command | Source Module | What Happens Locally vs. Cloud | Time & Cost Savings |
| :--- | :--- | :--- | :--- | :--- |
| **60-Second Workspace Awakening** (`run_body_begin`) | `body begin` (`init`, `awaken`, `start`) | [`begin.py`](../src/whole_body_compute/begin.py) | **100% Local (`<4s`)**: Wires `.agents/hooks.json`, `.claude/settings.local.json`, Cursor/Windsurf/Cline rules, `.git/info/exclude` hygiene, installs macOS LaunchAgents (`com.wholebodycompute.foveal` + default-on `AmbientWorkflowObserver`), runs live `<5ms` AST/secret self-test, and prints a 30-day FinOps ROI receipt. | Eliminates manual per-repo hook configuration; keeps `git status --porcelain` 100% clean. |
| **macOS LaunchAgent Daemon Manager** | `body daemon` (`status`, `install`, `uninstall`, `logs`) | [`daemon.py`](../src/whole_body_compute/daemon.py) | **100% Local**: Manages `com.wholebodycompute.foveal` (`foveal.sock` + `AmbientWorkflowObserver`), `screenpipe` (`:3030` Foveal HTTP), `remsleep` (`3:30 AM` REM), and `telemetry` (`:9381` HUD). | Keeps `<5ms` warm socket reflexes and passive workflow discovery running across reboots (`<15W` idle). |
| **Retroactive Multi-Harness Spend Audit & Canonical 5-Pillar Report** | `body audit-spend`, `body report`, `body savings`, `body context --by-model` | [`finops_audit.py`](../src/whole_body_compute/finops_audit.py), [`metrics.py`](../src/whole_body_compute/metrics.py), [`model_benchmark.py`](../src/whole_body_compute/model_benchmark.py) | **100% Local**: Scans local transcripts across Antigravity, Claude Code, Cursor, Goose, Windsurf, and Cline to compute preventable Context-Tax spend, 5-Pillar KPIs (`.agents/METRICS.md`), and empirical multi-model leaderboards. | Instant visibility into dollar ROI, Local Metal offload %, First-Time-Right quality, and per-model efficiency at `$0.00`. |

---

## 3. Deep Dive: `body commit` vs. Standard Agent Commit ("Just Commit")

When you ask a standard cloud agent to *"commit my changes"*, the agent typically drives a **3-to-5 turn shell loop** (`git status` $\to$ `git diff` $\to$ `ruff` $\to$ `git add` $\to$ `git commit`) and ingests the entire `git diff` into its cloud context window.

With **`body commit`** ([`src/whole_body_compute/local/git_fastpath.py`](../src/whole_body_compute/local/git_fastpath.py)), **almost 100% of the pipeline executes locally on your Mac in a single `<1.5s` pass**:

### Step-by-Step Comparison

| Step | Standard Cloud Agent Driving ("Just Commit") | `body commit` (On-Device Fast-Path) |
| :--- | :--- | :--- |
| **1. File Scoping & Multi-Agent Safety** | Agent runs `git status` (Turn 1), then often runs `git add .` or `git commit -a`, accidentally sweeping in untracked scratch files or edits another subagent is actively making. | **100% Local (`<10ms`)**: [`resolve_smart_scope`](../src/whole_body_compute/local/git_fastpath.py#L217-L331) checks `.git/wbc_active_touches.json`. If multiple sessions are active within 120s, it **only stages files touched by the current session** and aborts on same-file collisions. |
| **2. Index Lock Contention (`.git/index.lock`)** | Directly mutates `.git/index`. Collides with IDE background git fetchers or parallel subagents (`fatal: Unable to create '.git/index.lock'`). | **100% Local (`<15ms`)**: Stages inside a **process-private ephemeral shadow index** (`GIT_INDEX_FILE=/tmp/wbc_shadow_idx_<pid>_<uuid>.idx`). Zero `.git/index.lock` contention. |
| **3. Diff Inspection & Commit Message** | Agent runs `git diff` (Turn 2), uploading **4,000-18,000+ diff tokens** into the frontier LLM context window (which get re-billed on every subsequent turn of the thread). | **100% Local (`<15ms` Cold AST or `~250ms` Local Metal `:9379`)**: [`synthesize_commit_message`](../src/whole_body_compute/local/git_fastpath.py#L477-L567) inspects the diff in local RAM and generates a Conventional Commit message (`feat(scope): ...`) without uploading a single diff line to the cloud. |
| **4. Secret, Syntax & Lint Gates** | Agent either forgets to lint, or runs `ruff`/`pytest` in another turn (Turn 3) and reads verbose stdout into context. | **100% Local (`~150-400ms`)**: Automatically runs fail-closed [`check_secrets`](../src/whole_body_compute/local/git_fastpath.py#L400-L419) (API keys, `.env`), `<5ms` [`check_ast_syntax`](../src/whole_body_compute/local/git_fastpath.py#L422-L438) (`ast.parse`), and staged-only [`check_ruff`](../src/whole_body_compute/local/git_fastpath.py#L441-L465). |
| **5. Torn-Write Protection & Atomic Commit** | If a file is saved mid-stage, git commits a half-written state. | **100% Local (`<20ms`)**: Verifies `(mtime_ns, size, sha256)` via [`verify_no_torn_writes`](../src/whole_body_compute/local/git_fastpath.py#L371-L397), writes the tree via `git write-tree` + `git commit-tree`, and updates the branch pointer via atomic Compare-And-Swap (`git update-ref <branch> <new_sha> <old_sha>`). |
| **6. Episodic Memory Logging** | Only exists in `git log`. | **100% Local (`<5ms`)**: Logs a `git_commit` episodic anchor to `dot_memory.db` for `body memory --rewind` and `body brief`. |

### Quantified Time & Cost Savings per Commit

| Metric | Standard Cloud Agent Commit Loop | `body commit` | Net Savings |
| :--- | :---: | :---: | :--- |
| **Cloud LLM Turns** | `3 - 5 turns` | `1 turn` (or `0` from terminal) | **`-75%` to `-100%` turns** |
| **Wall-Clock Time** | `18s - 35s` | **`0.4s - 1.8s`** local (`~3s` via agent) | **`15s - 30s` saved (`85-90%` faster)** |
| **Cloud Context Bloat Added** | `+4,000 - +18,000 tokens` (persists for rest of thread) | **`~15 tokens`** (`ok` receipt) | **`99.8%` less context bloat** |
| **Direct + Downstream API Cost** | `$0.12 - $0.45+` per commit | **`$0.00`** (terminal) / **`~$0.02`** (1 agent turn) | **`$0.10 - $0.43+` saved (`90-100%` cheaper)** |

---

## 4. Master Feature & CLI Command Savings Ledger (Local % vs. Cloud Driving)

| Feature / Command | Local Execution % | Cloud Agent Role | Typical Wall-Clock Speedup | Typical Token / Cost Savings |
| :--- | :---: | :--- | :--- | :--- |
| **`body begin`** (`init` / `awaken`) | **100% Local** | 0 turns (run once per repo) | `<4s` vs. `15-30 min` manual hook/rule setup | Enables `-32.4%` context / `-25.6%` billed input across all future sessions |
| **Organ 5 BBB AST Slicer** (`read_file` hook) | **100% Local (`<3ms`)** | Receives `<3ms` AST Symbol Map + query-matched slice instead of truncated 2,000L dump | Eliminates 2-5 blind paging turns (`-18.8%` to `-45.9%` reads) | **95%-98.7% token reduction** on monster files (`$0.00` local AST) |
| **Organ 5 Traceback + Source Auto-Slicer** | **100% Local (`<5ms`)** | Receives exact failing frame + `±20L` AST source slice on Turn 1 | **-36.3% wall time** (`133.5s -> 85.0s`) | **-52.2% turns** (`23 -> 11`) & **-39.5% API cost** |
| **`body scout <path> -q "..."`** | **100% Local (`<3s`)** | Reads 150-token distilled answer instead of raw log/directory | **3x-5x faster** than multi-turn cloud `grep`/`read_file` | **85%+ token compression** at `$0.00` |
| **`body implement <file> -i "..."`** | **100% Local (`1.2s-3.5s`)** | Reviews 20-line unified diff | **4x-8x faster** (`1.2s` vs. `12-25s` cloud edit) | **98.7% token reduction** at `$0.00` |
| **`body eyes`** (`Foveal DOM`) | **100% Local (`<5ms`)** | Reads `~50-237` structured DOM tokens + dispatches 1-turn `--chain` macro | **-64.4% wall time** on GUI tasks (`160.2s -> 57.0s`) | **-96.89% to -99.56% sensory tokens** vs. Retina screenshots |
| **`body commit`** (`Shadow-Index CAS`) | **99-100% Local (`0.4s-1.8s`)** | 1 tool call (`ok` receipt) or 0 turns from CLI | **85-90% faster** (`0.8s` vs. `18-35s`) | **90-100% cost reduction** (`0` diff tokens in cloud transcript) |
| **`body flush --auto`** (`Glymphatic Flush`) | **100% Local (`<250ms`)** | Resumes in fresh thread from `<500`-token grounded `StateCapsule` | **4.76x faster** (`22.6s` vs. `107.7s` on 88-turn threads) | **98.15% context compaction** with `100%` state recall |
| **`body candidates --promote`** (`PalWorkflow`) | **100% Local (`0.37s`)** | 0 cloud turns once promoted to Tier-0/1 YAML recipe | **`0.37s`** local replay vs. `30-90s` cloud browser loop | **100% cloud token elimination (`$0.00/run`, `100% Pass^8`)** |
| **`body doctor --exercise`** | **100% Local (`<1s`)** | 0 turns (or 1 diagnostic turn) | Sub-second 5-Organ + 11-step reflex verification | `$0.00` (replaces ad-hoc `ps aux` / `vm_stat` shell debugging) |

---

## 5. A-Z Glossary of Whole Body Compute Terms

- **Active Touches Ledger (`.git/wbc_active_touches.json`)**: An `fcntl`-locked JSON ledger managed by [`git_fastpath.py`](../src/whole_body_compute/local/git_fastpath.py) that records which agent `session_id` modified which file over a rolling 120-second window, preventing multi-agent commit collisions.
- **Ambient Privacy Guard (`AmbientPrivacyGuard`)**: Zero-leak security filter in [`ambient_discovery.py`](../src/whole_body_compute/integrations/ambient_discovery.py) that automatically blocks password vaults (`1Password`, `Bitwarden`, `Keychain Access`), `AXSecureTextField` password inputs, API keys, SSNs, and credit cards from ever entering transition logs or candidate recipes.
- **Ambient Workflow Observer (`AmbientWorkflowObserver`)**: Default-on background sensor running inside `com.wholebodycompute.foveal` (`body eyes --daemon`) that monitors cross-app copy/paste and window focus transitions, clusters recurring $\ge 3$-step sequences into DAG motifs, and stages executable YAML candidates into `recipes/candidates/`.
- **APFS Copy-on-Write (CoW) Sandbox (`APFSCoWSandbox`)**: Creates an instant `<2ms` shadow clone of the workspace using Apple File System `cp -c -R` clones before an autonomous agent mutates code, promoting changes to the live tree only if the **3-Stage AST SecGate** passes.
- **AST SecGate (3-Stage)**: Deterministic static security and anti-reward-hacking gate in [`goose_runner.py`](../src/whole_body_compute/integrations/goose_runner.py) verifying: (1) valid Python AST syntax, (2) zero deleted `test_*` functions and zero tautological assertions (`assert True`, `x == x`), and (3) zero unsafe dynamic execution calls (`eval`, `exec`, `os.system`, `getattr(os, "system")`, `subprocess.*(shell=True)`).
- **Bicameral Compute**: Architectural division of labor pairing a **Frontier Cloud Prefrontal Cortex** (high-level reasoning, architecture, complex multi-file synthesis) with a **Local Apple Silicon Brainstem** (`$0.00` Metal GPU recon, `<3ms` AST slicing, `<5ms` Foveal DOM vision, and atomic Git fast-paths).
- **Blood-Brain Barrier (BBB / Organ 5)**: The bidirectional `PreInvocation`, `PreToolUse`, and `PostToolUse` firewall between the local machine and cloud LLMs. Strips secrets, blocks unauthorized network egress, auto-heals noisy shell commands, and slices oversized files/logs before they can bloat cloud context.
- **Cephalic Phase Pre-Warm (`trigger_cephalic_prewarm`)**: Biological reflex equivalent where detecting wakeword speech onset or IDE focus shifts triggers asynchronous `mmap` loading of local Metal models *while the human is still speaking or typing*, masking model cold-start latency to `0ms`.
- **Cognitive Saturation Index (CSI)**: Deterministic 0.00-1.00+ metric computed by `CognitiveThermalSentry` (`body context`) from conversation steps, full transcript byte size, truncated tool outputs, and true intra-turn model latency inflation (excluding human reading pauses). Thresholds: `GREEN (<0.70)`, `YELLOW (0.70-0.99)`, `RED (>=1.00)`.
- **Compare-And-Swap (CAS) Ref Update**: Lock-free git branch update (`git update-ref <branch> <new_sha> <old_sha>`) used by `body commit` so concurrent agents never corrupt `HEAD` or block on `.git/index.lock`.
- **Ephemeral Shadow Index (`GIT_INDEX_FILE`)**: Process-private temporary index file (`/tmp/wbc_shadow_idx_<pid>_<uuid>.idx`) used by `body commit` and `body checkpoint` to stage and write git trees without touching the shared `.git/index`.
- **Foveal Vision (`AXUIElement` DOM)**: Direct macOS Accessibility tree inspection (`body eyes`) that extracts interactive UI elements, screen coordinates, and text values in `<5ms` at `<50` tokens/turn - replacing 24/7 screen recording and 8,000-token screenshot loops.
- **Glymphatic Context Flush (`body flush --auto`)**: The cross-organ transaction that cures high-CSI context bloat by serializing a grounded **StateCapsule**, pruning duplicate memories in `dot_memory.db`, and spawning a fresh agent conversation pre-populated with a `<500`-token resumption prompt.
- **Hoare Logic Contracts (`Precondition` / `PostConditionCheck`)**: Formal verification assertions compiled into `PalWorkflow` YAML recipes (e.g., `clipboard_matches`, `url_contains`, `element_exists`) ensuring automated workflows fail safely and deterministically if UI state drifts.
- **Interoceptive Elastic Membrane**: Dynamic BBB threshold controller in [`proprioception.py`](../src/whole_body_compute/organs/proprioception.py) that adapts file-read and tool-output ceilings across three physiological regimes: `GREEN_ELASTIC` (full native 800L/2,000L ceiling), `YELLOW_BALANCED` (160L auto-slice on blind reads), and `RED_TIGHT` (80L auto-slice + `<=15`-word reflex cue).
- **PalCompiler & PalWorkflow**: The 4-pass demonstration-to-recipe compiler (`PalCompiler`) and execution engine (`PalWorkflow`) that lowers recorded GUI demonstrations or ambiently discovered motifs into **Tier 0 (Local CLI/File)**, **Tier 1 (Headless HTTP/API)**, or **Tier 1.5 (Foveal AX DOM)** deterministic recipes.
- **Procedural DNA (`body telemetry`)**: Zero-PII, structurally hashed workflow skeleton cards (`ir_signature_hash`) that capture reusable automation patterns without exposing user text, URLs, or credentials (governed by `TIER_0_LOCAL` default consent).
- **Radix KV-Cache (`body radix-cache`)**: Circulatory prompt cache manager that pins static system instructions and workspace `StateCapsule` prefixes in Apple Silicon Unified Memory for `<20ms` Time-To-First-Token.
- **SensoryVector (`ProprioceptiveState` + `InteroceptiveState`)**: Sub-`0.02ms` subconscious state vector computed on every tool call, tracking host harness envelope (`antigravity` 800L vs. `claude_code` 2,000L), read intent (`deliberate_bounded` vs. `blind_unbounded`), CSI adenosine level, and thermal state - enforcing **0 distraction meta-tokens** on nominal turns.
- **StateCapsule (`StateCapsule`)**: Non-lossy session handoff record stored in `dot_memory.db` containing the active goal, milestone, live `git status --porcelain` file vectors, SHA-256 artifact digests, and canonical 5-Pillar metrics receipt.
- **Tactile Sensor & Retry-Churn Reflex (`TactileSensor`)**: Closed-loop motor verification sensor (`<2ms`) that checks whether a file edit or UI action produced real structural change (`net_ast_progress`). If an agent touches the same file $\ge 3$ times without AST progress, it automatically attaches a `.pyi` contract slice and a 15-word Vagus course-correction cue.
- **Vagus Nerve Brake (`trigger_vagus_nerve_brake`)**: Sub-`50ms` acoustic barge-in kill-switch that intercepts spoken abort commands (`"stop"`, `"wait"`, `"cancel"`), sends `SIGINT` to active agent subprocesses in `<5ms`, and rolls back the active APFS CoW shadow sandbox.
- **Vascular Perfusion (`run_vascular_perfusion_sweep`)**: Hardware memory hygiene pass (`body perfusion`) that evicts idle specialist LLM weights from Active Unified RAM into Darwin's Inactive File Cache while protecting Tier-0 autonomic daemons.
- **Zero-Click Handoff Bridge (`ZeroClickHandoffBridge`)**: Automated thread-spawning bridge in [`cognitive_capsules.py`](../src/whole_body_compute/organs/cognitive_capsules.py) that raises the active IDE (`Antigravity`, `Cursor`, `Claude`), clicks `"New Conversation"`, locates the fresh `Message input` composer via Foveal Vision, pastes the `<500`-token `StateCapsule` resumption prompt, and presses Return with zero human clicks.
