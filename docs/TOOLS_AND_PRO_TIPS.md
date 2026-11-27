# Whole Body Compute (`body`) - Tools & Pro Tips Playbook

> **Interactive Web Version**: **[`https://wholebodycompute.com/docs`](https://wholebodycompute.com/docs)**  
> **In-Terminal Quick Reference**: Run `body tips` (or `body tips --rules` to print copy-paste rules for `CLAUDE.md` / `.cursorrules`).

Whole Body Compute (`body`) gives **Claude Code**, **Google Antigravity**, **Cursor**, **Windsurf**, **Cline**, and **Goose** fast local reflexes on your Mac (with cross-platform Blood-Brain Barrier and FinOps audit support on Linux).

To get the most out of WBC, keep **two layers** in mind:

1. **Layer 1 - Passive Autopilot (0 Effort after `body begin`)**: Once wired into your workspace hooks, the Blood-Brain Barrier automatically slices oversized `read_file` calls in `<3ms`, compacts noisy `pytest`/`git`/`rg` logs, redacts 23-vendor secrets before egress, and blocks reward-hacking test edits (`assert True`, deleted `test_*`).
2. **Layer 2 - Active Power Moves (5x-10x Leverage)**: High-speed local commands you (or your coding agent) run on demand for `$0` git commits, zero-cursor Chrome & macOS UI control, AST-grounded voice dictation, local Metal pre-digests, and zero-click thread resets.

---

## 1. Your First 5 Minutes

Run these three commands inside any repository to confirm everything is wired and working:

```bash
# 1. Wire PreToolUse/PostToolUse hooks across your coding agents + print 30-day FinOps ROI receipt
body begin

# 2. Verify all 5 Organs + run closed-loop reflex exercise in <1s
body doctor --exercise

# 3. Inspect live tokens, dollars, and seconds saved across organs, files, and bots
body savings --leaderboard
```

---

## 2. Top 8 Power-User Pro Tips

### Pro Tip #1: Never ask your cloud agent to "just git commit" (`body commit` + `body checkpoint`)
* **Tier**: `Free Forever` · **Impact**: `85-90% faster` (`~0.8s` vs `18-35s`), `$0` diff bloat, `0` `.git/index.lock` crashes.
* **Why**: Standard cloud agent commits take 3-5 turns (`git status` -> `git diff` -> `git add` -> `git commit`) and dump `4,000-18,000` diff tokens into your transcript that get re-billed on every turn after.
* **Do this instead**:
  ```bash
  # Optional <30ms safety snapshot before risky refactors (never touches .git/index or HEAD)
  body checkpoint before-auth-refactor

  # Shadow-index atomic commit (auto-scoped to session touches + secret/AST/ruff gates)
  body commit
  ```

### Pro Tip #2: Beat "late-thread brain fog" before Turn 50 (`body context` + `body flush --auto`)
* **Tier**: `CSI Free` / `Auto-Flush Pro` · **Impact**: `98.15%` context compaction, `4.76x` faster handoff, `100%` state recall.
* **Why**: Quadratic transformer attention makes 500KB+ threads slow, forgetful, and expensive (`$1-$5/turn`).
* **Do this instead**:
  ```bash
  # Check real-time Context Saturation Index (GREEN < 0.70, YELLOW 0.70-0.99, RED >= 1.00)
  body context --insights

  # Serialize live git status + SHA-256 artifacts into a <500-token StateCapsule & spawn a fresh chat
  body flush --auto
  ```

### Pro Tip #3: Control live macOS apps & logged-in Chrome without moving your mouse (`body eyes` + `body hands --no-cursor`)
* **Tier**: `Free Forever` · **Impact**: `<15ms` DOM vs `7,650+` token screenshots (`-96.9%` sensory tokens), zero cursor hijack.
* **Why**: Full-screen PNG loops burn thousands of tokens, and headless browsers aren't signed into your real apps.
* **Do this instead**:
  ```bash
  # One-time setup: install notarized Whole Body Compute.app (Terminal & IDEs need zero TCC permissions)
  body eyes --install-app

  # Inspect interactive elements in <15ms
  body eyes --app "Google Chrome" --filter "Submit"

  # Click, paste, and submit in the background without moving your physical cursor
  body hands --app "Google Chrome" --no-cursor --chain "click:e5,paste:Ready to ship,hotkey:enter"
  ```

### Pro Tip #4: `$0` repo-aware voice-to-code dictation (`body ears`)
* **Tier**: `Free Forever` · **Impact**: `$0/mo` (vs `$15/mo` cloud dictation apps), zero audio uploaded, exact symbol spelling.
* **Why**: Generic voice tools turn `SpinalReflexGate` into `"spinal reflex gate"`. `body ears` hydrates your repo's live `symbols.db` AST in `<5ms` and snaps spoken identifiers to exact `CamelCase`, `snake_case`, and file paths.
* **Do this instead**:
  ```bash
  # Install background push-to-talk LaunchAgent (Hold Fn to talk, Fn+Space for hands-free)
  body ears --install

  # Add custom team acronyms or product names to ~/.whole_body_compute/vocab.txt
  body ears --add-word "StateCapsule"

  # Verify hotkey & indexed symbol count
  body ears --status
  ```

### Pro Tip #5: Optional - Pre-digest multi-megabyte `.log` dumps on Apple Silicon (`body scout`)
* **Tier**: `Opt-In (Requires Ollama or MLX)` · **Impact**: `85%-98.7%` raw log token reduction at `$0.00` local cost.
* **Why**: For normal source code, your native `Read` tool + `<3ms` Blood-Brain Barrier hook already slices oversized files automatically with `0 GB` local LLM RAM. If you already run Ollama or MLX locally, `body scout` is great as an opt-in human tool for multi-megabyte raw `.log` / crash dumps (and `body scout --slice` runs pure `<3ms` AST slicing with zero LLM required).
* **Do this instead**:
  ```bash
  # Ask a warm local Metal/Ollama model to distill a massive raw log
  body scout server.log -q "What caused the 502 spike?"

  # Extract exact AST symbol slices across multiple files in <3ms (zero LLM required)
  body scout --slice src/auth.py:verify_token src/db.py:Session
  ```

### Pro Tip #6: Zoom into compacted logs with the lossless Airlock Blob Vault (`body bbb --blob`)
* **Tier**: `Free Forever` · **Impact**: `<5ms` raw line recovery, `0%` prompt-prefix KV-cache invalidation.
* **Why**: Whenever Blood-Brain Barrier compresses a verbose `pytest` or `git diff` dump, it stores a secret-redacted copy locally in `~/.whole_body_compute/airlock_blobs/` and tags the summary with `[BLOB:<id>]`.
* **Do this instead**:
  ```bash
  # Inspect recent Blood-Brain Barrier interceptions
  body bbb --window 24h --tools

  # Recover exact raw lines in <5ms without re-running the test suite
  body bbb --blob <blob_id> --lines 120:180
  ```

### Pro Tip #7: Promote repeated multi-app chores into `$0`, `0.37s` skills (`body candidates` & `body hub`)
* **Tier**: `Discovery Free` / `Promote Pro` · **Impact**: `0.37s` deterministic execution, `100%` cloud token elimination.
* **Why**: WBC passively observes recurring 3+ step cross-app routines (while blindfolding `1Password`, `Keychain`, and password fields) and stages executable YAML candidates in `recipes/candidates/`.
* **Do this instead**:
  ```bash
  # Discover recurring cross-app habits
  body candidates --discover

  # Promote a candidate into an active $0 recipe + portable .skill.zip bundle [Pro]
  body candidates --promote <candidate_id> --as-skill

  # Or search & pull Ed25519-signed community recipes from the Hub
  body hub search github
  ```

### Pro Tip #8: Shelve dormant MCP servers & unused skills (`body skills audit` / `prune`)
* **Tier**: `Free Forever` · **Impact**: Reclaims `3,000-18,000+` Turn-0 schema tokens per session.
* **Why**: Dormant MCP servers and unused skills inject their tool schemas on Turn 0 and re-bill them on every turn.
* **Do this instead**:
  ```bash
  # Measure real usage vs. prompt token tax across your transcripts
  body skills audit

  # Preview & reversibly shelve 0-use skills and MCP servers (restore anytime with `restore`)
  body skills prune --dry-run
  body skills prune-mcp --dry-run
  ```

---

## 3. Complete Tool Reference by Job-to-be-Done

| Goal | Command | Tier | What It Does |
| :--- | :--- | :---: | :--- |
| **Setup & Health** | `body begin` | Free | Wire agent hooks across IDEs + print 30-day FinOps ROI receipt |
| | `body doctor --exercise` | Free | `<1s` 5-Organ health check + closed-loop reflex verification (`--chaos` fuzzes) |
| | `body shield --status` | Free | Audit or install (`--install --all`) zero-schema `Pre`/`PostToolUse` hooks |
| | `body tips` | Free | Print terminal Pro Tips cheat sheet (`--rules`, `--random`, `--open`) |
| | `body heart` | Free | Check Apple Silicon thermals, Unified RAM headroom, and swap pressure |
| | `body daemon status` | Free | Manage macOS LaunchAgents (`install`, `status`, `logs`) |
| **Git & Multi-Agent** | `body commit` | Free | `~0.8s` shadow-index atomic commit with secret + AST + `ruff` gates |
| | `body checkpoint <label>` | Free | `<30ms` zero-lock working-tree snapshot (`--list`, `--restore <ref>`) |
| **Desktop, Chrome & Voice** | `body eyes --app "..."` | Free | `<15ms` Foveal `AXUIElement` DOM snapshot (`--install-app` installs broker app) |
| | `body hands --app "..." --no-cursor` | Free | Background clicks, typing, paste, hotkeys & `--chain` macros |
| | `body ears` (`body transcribe`) | Free | `$0` local AST-grounded dictation (`Hold Fn` / `Fn+Space`, `--install`, `--add-word`) |
| | `body meeting` | Free | Local call & mic meeting text capture (`status`, `start`, `stop`, `show latest`) |
| | `body brief --speak` | Free | `<30s` (`<=75`-word) morning context, git & FinOps briefing |
| **FinOps & Context** | `body savings --leaderboard` | Free | Live shield token/USD/time ledger & 4-board leaderboard |
| | `body audit-spend --days 30` | Free | 30-day multi-harness spend & preventable context-tax invoice |
| | `body scorecard --by-model` | Free | 5-Pillar KPI scorecard & empirical multi-model leaderboard (`alias: body report`) |
| | `body context --insights` | Free | Context Saturation Index (`CSI`) & root-cause bloat debugger |
| | `body bbb --blob <id>` | Free | Blood-Brain Barrier audit (`--tools`, `--blocked`) & lossless log line recovery |
| **Local Metal & Skills** | `body scout <target> -q "..."` | Free | `$0` Local Metal log/file pre-digest & `--slice file.py:Sym` AST slicer |
| | `body implement <file> -i "..."` | Free | Surgical 1-file Local Metal edit + AST syntax check |
| | `body skills audit` | Free | Audit & reversibly prune (`prune`, `prune-mcp`, `restore`) dormant skills/MCPs |
| | `body candidates --discover` | Free / Pro | Discover repeated cross-app habits (Free) & `--promote` to `$0` skills (Pro) |
| | `body hub <search\|pull>` | Free | Search & pull Ed25519-signed `$0` `PalWorkflow` recipes |
| **Memory & Autonomous Pro** | `body flush --auto` | Pro | Grounded `StateCapsule` compaction & zero-click fresh chat handoff |
| | `body memory` | Pro | Hippocampus SQLite WAL (`--remember`, `--recall`, `--deja-vu`, `--rewind`) |
| | `body perfusion --evict` | Pro | Evict idle 14B-70B local Metal weights (`<50ms`) or `--prewarm` |
| | `body sleep` | Pro | Nocturnal REM memory deduplication & LocalBench-gated Metal distillation |

---

## 4. Copy-Paste Snippet to Teach Your Agent (`CLAUDE.md` / Rules)

`body begin` wires your `PreToolUse` and `PostToolUse` hooks automatically. Adding this short, zero-RAM-overhead block to your global `CLAUDE.md`, `.cursorrules`, or workspace rules (also available via `body tips --rules`) teaches your agent to stick to fast, deterministic wins on every machine:

```markdown
# Whole Body Compute (`body`) Power-User Rules
- **Commits & Checkpoints**: Prefer `body commit` (or `body commit <paths> -m "..."`) instead of multi-turn `git status` + `git diff` + `git add` + `git commit` loops. Before risky refactors, run `body checkpoint <label>`.
- **Read Code Natively (Zero-RAM AST Shield)**: Call native `Read` / `read_file` directly on source files instead of `cat`/`head`/`sed` shell loops. The `<3ms` Blood-Brain Barrier hook automatically slices oversized files with `0 GB` local LLM RAM.
- **Compacted Logs (`[BLOB:<id>]`)**: When a test or CLI log shows `[BLOB:<id>]`, recover exact lines locally via `body bbb --blob <id> --lines START:END` instead of re-running the full test suite.
- **Live macOS Apps & Logged-In Chrome (macOS)**: Prefer `body eyes --app "..."` and `body hands --app "..." --no-cursor` over full-screen screenshots (`screencapture`) or AppleScript.
```

---

## 5. Fast Troubleshooting

* **`body eyes` doesn't see inside Chrome or Electron**: Run `body eyes --install-app` once and toggle **Whole Body Compute.app** ON in `System Settings -> Privacy & Security -> Accessibility`. Your Terminal and IDEs require zero TCC permissions.
* **Pressing `Fn` opens the macOS Emoji/Globe picker**: Run `body ears --status` to inspect your `Fn` setting. Either set `System Settings -> Keyboard -> Press Globe key to: Do Nothing`, or use a custom chord (`body ears --hotkey opt+space`).
* **Unsure whether hooks are active in your repo**: Run `body shield --status` followed by `body bbb --window 1h --tools`.
* **Mac feels warm or low on Unified RAM**: Run `body heart` to check pressure, then `body perfusion --evict` to reclaim 4-18 GB of Unified RAM from idle local models in `<50ms`.

Deep architectural details & A-Z glossary: **[`docs/WIKI_AND_GLOSSARY.md`](WIKI_AND_GLOSSARY.md)** · Security & TCC specs: **[`docs/SECURITY_AND_TCC_FAQ.md`](SECURITY_AND_TCC_FAQ.md)**.
