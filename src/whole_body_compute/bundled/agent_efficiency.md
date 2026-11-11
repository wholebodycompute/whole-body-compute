# Whole Body Compute: Bicameral Compute & Blood Brain Barrier (Organ 05)

This workspace uses **Whole Body Compute (WBC)** (**Bicameral Compute** & **Organ 05 Blood Brain Barrier**) to give frontier cloud models fast local Apple Silicon reflexes, `<5ms` crash/log triage, and hardware/git safety—**without restricting frontier intelligence or starving the model of source code context**.

## 1. Zero-Polling on Background Tasks
- **NEVER poll background task status in a loop.** Wait for reactive completion notifications when a task runs asynchronously.
- **Set `WaitMsBeforeAsync: 10000`** (when available) for fast synchronous commands (`pytest`, `body doctor`, linters, builds `<10s`) so they finish in a single turn.

## 2. Full-Context Code Reading & `<5ms` Log/Monster-File Triage
- **Normal Source Files (`<= 800 lines` in Antigravity / `<= 2,000 lines` in Claude Code)**: Read normal code files directly in a single turn so you have full visibility into imports, cross-function state, and edge cases before editing. Do not artificially page through small or medium files.
- **Monster Files (`> 800` / `> 2,000` lines)**: Unbounded reads on oversized files are automatically sliced inline with a `<3ms` AST Symbol Map (`class`/`def` line ranges) so symbols at the bottom of large files are never silently truncated. Jump straight to the exact symbol with bounded line ranges (`StartLine`/`EndLine` or `offset`/`limit`).
- **Raw Log & Crash Files (`*.log`, `*.err`, `transcript.jsonl`)**: Organ 05 (`Blood Brain Barrier`) automatically extracts `Traceback`/`ERROR` blocks inline in `<5ms` and attaches the referenced source slice. For targeted log queries, run:
  ```bash
  body scout <log_path> -q "<question/objective>"
  ```

## 3. Hardware, Vitals & Verified Token Savings
- Check Apple Silicon RAM, swap, thermal state, or 5-organ health directly via CLI instead of ad-hoc `ps aux` / `vm_stat` scripts:
  ```bash
  body doctor
  body savings
  ```

## 4. Frontier Quality & Safe Git Hygiene
- **Write code directly with Frontier intelligence**: Use native `edit_file` / `write_file` (`Edit` / `Write`) with full context.
- **Never run destructive git commands** (`git reset --hard`, `git checkout --`, `git clean -fd`) without explicit user confirmation.
