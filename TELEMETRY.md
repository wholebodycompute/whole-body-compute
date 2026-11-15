# Sovereign Telemetry & Hybrid Compute Dataset Specification (`TELEMETRY.md`)

Whole Body Compute (`body` / `wbc`) is built on a clear architectural division: **your code, context, files, prompts, and personal data stay 100% sovereign on your machine**, while **anonymous operational vitals and zero-PII Procedural DNA cards** help build the industry's open empirical benchmark of how local compute collaborates with frontier Cloud LLMs.

---

## 1. The Sovereign Red-Line Guarantee

Whole Body Compute **NEVER** collects, logs, or transmits:

- **No Source Code or AST Content:** Never file contents, diffs, symbol bodies, or AST slice text.
- **No Paths or Identifiers:** Never file paths (`/Users/...`, `/home/...`), directory names, repository names, git remotes, branch names, or commit SHAs.
- **No Prompts or Conversations:** Never user prompts, LLM responses, spoken transcripts, or clipboard contents.
- **No Screen or Window Data:** Never screenshots, window titles, DOM text values, or `(x, y)` screen coordinates.
- **No Secrets or Credentials:** Protected by a 23-vendor credential scrubber (`redaction.py`) plus recursive key stripping (`*_key`, `*_secret`, `*_token`, `*_password`, `auth`).
- **No PII or IP Addresses:** Emails, phone numbers, SSNs, credit cards, customer names, order IDs, usernames, hostnames, hardware serials/UUIDs, and MAC addresses are scrubbed on-device before serialization. Every outbound payload sets `"$ip": null` and `"$geoip_disable": true` so client IP addresses are discarded at ingestion.

---

## 2. Quick Commands: Inspect, Control, or Opt Out

You have complete visibility and one-command control at all times:

```bash
# Inspect the exact JSON payload last sent from your machine
body telemetry show

# View active consent tier, rotating cohort ID, and local outbox card counts
body telemetry status

# Inspect every deduplicated Procedural DNA card staged locally
body telemetry inspect

# Rotate your anonymous monthly cohort salt immediately
body telemetry reset-id

# Opt out completely (sets TIER_0_LOCAL -> 100% local, zero network calls)
body telemetry disable
```

### Environment Variable Circuit-Breakers (Instant Zero-Egress)

Setting **any** of the following environment variables automatically overrides telemetry to `TIER_0_LOCAL` (zero network egress across all CLI commands, hooks, and background tasks):

```bash
export DO_NOT_TRACK=1              # Universal CLI Do-Not-Track standard (consoledonottrack.com)
export WBC_TELEMETRY_DISABLED=1    # Whole Body Compute explicit disable flag
export BODY_TELEMETRY=airgapped    # Also accepts: 0, off, false, none, disabled
```

Telemetry is also **automatically disabled** in CI/test runners (`CI=true`, `GITHUB_ACTIONS=true`, `PYTEST_CURRENT_TEST`).

---

## 3. Consent Tiers (`~/.whole_body_compute/telemetry_consent.json`)

| Tier | CLI Command | What Runs Locally | What Syncs Over Network |
| :--- | :--- | :--- | :--- |
| **`TIER_2_PROCEDURAL_DNA`** *(Default, Opt-Out)* | `body telemetry enable` | Full SQLite WAL, FinOps ROI ledger, and `<= 15.0 MB` capped outbox | Anonymous operational vitals, hybrid step Pareto metrics, and sanitized Procedural DNA workflow cards (`$ip: null`) |
| **`TIER_1_VITALS`** | `body telemetry opt-in --tier 1` | Full local ledgers & outbox | Anonymous numeric/enum operational vitals & hybrid step metrics only (no workflow skeleton cards) |
| **`TIER_0_LOCAL`** *(Air-Gapped / Opt-Out)* | `body telemetry disable` | Full local ledgers & outbox (`body stats` still works 100% offline) | **Zero network calls** |

On the first interactive terminal session, `body` prints a one-time notice to `stderr` (never polluting `--json` `stdout` pipelines) reminding you of `body telemetry show` and `body telemetry disable`.

---

## 4. Exact Outbound JSON Schemas

All anonymous telemetry uses a **monthly-rotating cohort ID** (`sha256(local_random_salt + ":" + YYYY-MM)[:16]`), preventing permanent cross-month device tracking while enabling short-window reliability analysis.

### A. Anonymous Hardware & Harness Context Envelope

Attached to operational events so hybrid efficiency can be benchmarked across Apple Silicon generations, Unified RAM tiers, and coding agent harnesses:

```json
{
  "wbc_version": "0.1.1",
  "license_tier": "community",
  "os": "darwin",
  "arch": "arm64",
  "chip_family": "Apple M3 Max",
  "ram_gb_bucket": 64,
  "cpu_cores": 16,
  "python_minor": "3.12",
  "active_agent_harness": "claude_code",
  "cohort_id": "9f4c2e81b03a7d12",
  "$geoip_disable": true,
  "$ip": null
}
```

### B. Hybrid Local + Cloud Step Telemetry (`wbc_hybrid_step` & `wbc_shield_pulse`)

Measures how local silicon offloads work from frontier Cloud LLMs:

```json
{
  "event": "wbc_hybrid_step",
  "distinct_id": "9f4c2e81b03a7d12",
  "properties": {
    "stage": "local_scout",
    "model_name": "qwen2.5-coder:7b",
    "is_local": true,
    "duration_ms": 142,
    "input_tokens": 4800,
    "output_tokens": 210,
    "tokens_saved": 4590,
    "savings_pct": 95.6,
    "estimated_usd_saved": 0.0161,
    "tok_per_sec": 84.2,
    "thermal_state": "nominal",
    "free_ram_gb": 38.4,
    "cascaded": false,
    "success": true,
    "$geoip_disable": true,
    "$ip": null
  }
}
```

### C. Workflow Automation & Habit Intelligence (`wbc_workflow_executed`)

Reveals what multi-app workflows developers automate, which ones become daily habits, and where UI or headless steps fail or self-heal - with **zero window titles, prompts, or parameter values**:

```json
{
  "event": "wbc_workflow_executed",
  "distinct_id": "9f4c2e81b03a7d12",
  "properties": {
    "recipe_id": "post_call_actions",
    "is_custom_user_recipe": false,
    "outcome": "success",
    "archetype": "multi_app_handoff",
    "dominant_tier": "TIER_0_LOCAL",
    "steps_count": 4,
    "headless_step_share": 0.75,
    "app_chain": ["Google Chrome", "Slack"],
    "action_chain": ["activate_app", "ax_press", "run_shell", "http_request"],
    "lifetime_runs_for_recipe": 12,
    "distinct_days_active": 5,
    "self_healed_steps": 0,
    "tokens_saved": 22950,
    "screen_mb_saved": 4.2,
    "elapsed_s": 0.84,
    "$geoip_disable": true,
    "$ip": null
  }
}
```

### D. The 5 Content-Addressable Procedural DNA Cards (`wbc_procedural_dna_card`)

Deduplicated in `~/.whole_body_compute/telemetry_outbox.jsonl` (strictly capped at `<= 15.0 MB` with 7-day SQLite compaction):

1. **`compiler_ir_card`** - Workflow compression (`raw_step_count -> compiled_step_count`), `dominant_tier`, Hoare contract names, and reliability score.
2. **`app_proprioception_card`** - macOS/Electron accessibility motor reliability (`AXPress` vs. `CGEvent` fallback ratio, optimal hotkey dwell ms) keyed strictly by app bundle family (`com.google.Chrome`, `com.electron.*`, `com.apple.finder`) with zero window titles or coordinates.
3. **`router_habit_card`** - Intent-domain tool-pruning whitelists and local model pass rates.
4. **`generalized_recipe_template`** - Zero-PII workflow skeletons where paths, scripts, dates, entities, order numbers, and flags are generalized into placeholders (`python ${SCRIPT_PATH} --order ${order_id}`, `${WORKSPACE_DIR}`, `<ENTITY>`, `<DATE>`).
5. **`hybrid_compute_efficiency_card`** - Aggregated Pareto rollup of Shield AST token savings by tool category, local-vs-cloud step share, average local Metal latency/throughput, and habitual workflow counts (`3+` runs).

---

## 5. Automated CI Privacy & Redaction Verification

Our privacy guarantees are continuously enforced in public CI rather than relying on policy prose alone:

- **[`tests/test_procedural_telemetry_sync.py`](tests/test_procedural_telemetry_sync.py)** - Verifies Sovereign Red-Line PII/prompt/code/path/window-title scrubbing, `$ip: null` & `$geoip_disable: true` zero-IP headers, `DO_NOT_TRACK=1` / `body telemetry disable` zero-egress enforcement, workflow anatomy fingerprinting, local outbox deduplication, and `body telemetry show` inspectability.
- **[`tests/test_immune_bbb.py`](tests/test_immune_bbb.py)** - Verifies `<1ms` 23-vendor secret redaction and Blood-Brain Barrier pre-egress protection.

To run the telemetry & redaction privacy suite locally in any GitHub clone:

```bash
pytest -q tests/test_procedural_telemetry_sync.py tests/test_immune_bbb.py
```
