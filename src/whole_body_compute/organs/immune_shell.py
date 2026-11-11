"""
Blood-Brain Barrier (Organ 05): Shell Command Bloat Inspector & Context-Tax Gate.
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional

READ_TOOL_NAMES = ("read_file", "view_file", "Read", "NotebookRead", "beforeReadFile", "pre_read_code")
WRITE_EDIT_TOOL_NAMES = ("edit_file", "write_file", "Edit", "MultiEdit", "Write", "NotebookEdit")
SHELL_TOOL_NAMES = ("run_command", "Bash", "execute_command", "developer__shell")
SEARCH_DIR_TOOL_NAMES = ("grep_search", "Grep", "find_by_name", "Glob", "list_dir", "LS")
SUBAGENT_TOOL_NAMES = ("invoke_subagent", "Task", "Agent", "subagent")
DB_RESOURCE_WEB_BROWSER_TOOL_NAMES = (
    "execute_sql",
    "query_database",
    "list_resources",
    "read_resource",
    "read_url_content",
    "WebFetch",
    "fetch_web_page",
    "mcp_chrome-devtools_take_snapshot",
)


def parse_window_to_seconds(window: str) -> Optional[float]:
    """
    Parse a human-friendly time window string into seconds.
    Supports formats like '1h', '2h', '24h', '30m', '7d', 'all'.
    Returns None if window represents all time (e.g. 'all', '', 'none', '*').
    """
    if not window:
        return None
    w = str(window).strip().lower()
    if w in ("all", "none", "0", "*", "infinity", "inf"):
        return None
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([a-zA-Z]+)?$", w)
    if not m:
        return None
    val = float(m.group(1))
    unit = (m.group(2) or "").lower()
    if unit in ("h", "hr", "hrs", "hour", "hours"):
        return val * 3600.0
    elif unit in ("m", "min", "mins", "minute", "minutes"):
        return val * 60.0
    elif unit in ("s", "sec", "secs", "second", "seconds"):
        return val
    elif unit in ("d", "day", "days"):
        return val * 86400.0
    elif unit in ("w", "wk", "wks", "week", "weeks"):
        return val * 7 * 86400.0
    elif not unit:
        if val <= 168:
            return val * 3600.0
        return val
    return None


def _split_shell_pipes_respecting_quotes(cmd_str: str) -> List[str]:
    """
    Split a shell command string by unquoted pipe characters ('|'), respecting:
    - single quotes ('...') where escapes are literal
    - double quotes ("...") where backslashes escape quotes
    - backslash escapes outside quotes
    - double pipes ('||' as logical OR - NOT pipeline split)
    """
    if not cmd_str:
        return []

    stages: List[str] = []
    current: List[str] = []
    in_single = False
    in_double = False
    escaped = False
    i = 0
    n = len(cmd_str)

    while i < n:
        char = cmd_str[i]

        if escaped:
            current.append(char)
            escaped = False
            i += 1
            continue

        if char == "\\" and not in_single:
            escaped = True
            current.append(char)
            i += 1
            continue

        if char == "'" and not in_double:
            in_single = not in_single
            current.append(char)
            i += 1
            continue

        if char == '"' and not in_single:
            in_double = not in_double
            current.append(char)
            i += 1
            continue

        if not in_single and not in_double:
            if char == "|":
                if i + 1 < n and cmd_str[i + 1] == "|":
                    current.append("||")
                    i += 2
                    continue
                stage = "".join(current).strip()
                if stage:
                    stages.append(stage)
                current = []
                i += 1
                continue

        current.append(char)
        i += 1

    last_stage = "".join(current).strip()
    if last_stage:
        stages.append(last_stage)

    return stages


def inspect_shell_command_bloat(cmd_line: str) -> Dict[str, Any]:
    """
    Context-Tax Gate: Intercept shell commands that attempt to bypass the read_file
    gate by running raw unbounded dumps on files >250 lines (e.g. cat <file>, cat -n <file>,
    head -n 1000 <file>) or unbounded verbose directory crawls / test floods without quiet flags.
    """
    # [direct-edit] Return healed_command on non-blocked commands
    if not cmd_line or not cmd_line.strip():
        return {"blocked": False, "healed_command": ""}

    raw_cmd = cmd_line.strip()
    is_strict_test = bool(  # [direct-edit]
        os.environ.get("PYTEST_CURRENT_TEST")
        or os.environ.get("VOICEFI_STRICT_TEST_HOOK") == "1"
    )

    # Blood-Brain Barrier Anti-Polling Gate: Block standalone sleep <N>, or auto-heal compound commands with trailing sleep <N>
    if re.search(r"\bsleep\s+\d+", raw_cmd):
        if re.match(r"^\s*sleep\s+\d+\s*$", raw_cmd):
            return {
                "blocked": True,
                "category": "anti_polling",
                "target": raw_cmd[:120],
                "direct_tokens_saved": 2000,
                "reason": (
                    "🛡️ Blood-Brain Barrier Anti-Polling Gate: Active shell polling ('sleep N') is blocked to prevent O(N^2) context re-ingestion. "
                    "Use WaitMsBeforeAsync: 10000 for fast commands, or let long-running commands background automatically and wait for reactive wakeup (do not poll)."
                ),
                "remedy": "Use WaitMsBeforeAsync: 10000 or background task reactive wakeup",
            }
        else:
            stripped_cmd = re.sub(r"(?:\s*(?:&&|;)\s*sleep\s+\d+\s*)+$", "", raw_cmd).strip()
            if stripped_cmd != raw_cmd and stripped_cmd:
                return {
                    "blocked": True,
                    "category": "anti_polling_tail",
                    "target": raw_cmd[:120],
                    "direct_tokens_saved": 2000,
                    "reason": "auto_healed_strip_sleep",
                    "remedy": stripped_cmd,
                    "auto_heal_cmd": stripped_cmd,
                    "auto_healed_cmd": stripped_cmd,
                    "healed_command": stripped_cmd,
                }

    sub_commands = re.split(r"(?:\s*&&\s*|\s*\|\|\s*|\s*;\s*)", raw_cmd)
    single_sub = len(sub_commands) == 1

    for sub_cmd in sub_commands:
        sub = sub_cmd.strip()
        if not sub:
            continue

        # [direct-edit] Quote-aware pipe splitting so regex alternation "foo|bar" is not truncated
        pipe_stages = [
            p.strip()
            for p in re.split(r'\|(?=(?:[^"\']*["\'][^"\']*["\'])*[^"\']*$)', sub)
            if p.strip()
        ]
        if not pipe_stages:
            continue

        def _build_healed_response(
            category: str,
            target: str,
            tokens: int,
            reason: str,
            remedy: str,
            first_healed: Optional[str] = None,
        ) -> Dict[str, Any]:
            res_obj = {
                "blocked": True,
                "category": category,
                "target": target[:120],
                "direct_tokens_saved": tokens,
                "reason": reason,
                "remedy": remedy,
            }
            if first_healed and single_sub:
                tail = pipe_stages[1:]
                full_cmd = (
                    first_healed
                    if (tail and first_healed.endswith(" | " + " | ".join(tail)))
                    else (" | ".join([first_healed] + tail) if tail else first_healed)
                )
                res_obj["auto_heal_cmd"] = full_cmd
                res_obj["auto_healed_cmd"] = full_cmd
                res_obj["healed_command"] = full_cmd
            elif remedy:
                res_obj["healed_command"] = remedy
            return res_obj

        first_stage = pipe_stages[0]
        has_bounding_filter = False
        if len(pipe_stages) > 1:
            for downstream in pipe_stages[1:]:
                d_tokens = downstream.split()
                if not d_tokens:
                    continue
                d_base = Path(d_tokens[0]).name
                if d_base in ("grep", "egrep", "fgrep", "wc", "awk", "sed", "jq", "scout"):
                    has_bounding_filter = True
                    break
                elif d_base in ("head", "tail"):
                    limit_n = 10
                    for idx, tok in enumerate(d_tokens[1:], start=1):
                        if tok == "-n" and idx < len(d_tokens) - 1:
                            try:
                                limit_n = int(d_tokens[idx + 1])
                            except ValueError:
                                pass
                            break
                        elif tok.startswith("-") and tok[1:].isdigit():
                            limit_n = int(tok[1:])
                            break
                    if limit_n <= (180 if is_strict_test else 800):
                        has_bounding_filter = True
                        break

        try:
            tokens = shlex.split(first_stage)
        except Exception:
            tokens = first_stage.split()

        if not tokens:
            continue

        if "<<" in first_stage or any(t.startswith("<<") for t in tokens):
            continue

        cmd_idx = 0
        while cmd_idx < len(tokens):
            tok = tokens[cmd_idx]
            if "=" in tok and not tok.startswith("-") and not Path(tok).is_file():
                cmd_idx += 1
                continue
            if tok in ("sudo", "nohup", "time"):
                cmd_idx += 1
                continue
            break

        if cmd_idx >= len(tokens):
            continue

        base_tok = tokens[cmd_idx]
        base_cmd = Path(base_tok).name.lower()
        rest_tokens = tokens[cmd_idx + 1 :]

        # 1. Pytest Test Floods without quiet flags
        is_pytest_exec = is_strict_test and (
            base_cmd == "pytest"
            or (base_cmd.startswith("python") and len(rest_tokens) >= 2 and rest_tokens[0] == "-m" and rest_tokens[1] == "pytest")
            or (base_cmd in ("uv", "poetry") and len(rest_tokens) >= 2 and rest_tokens[0] == "run" and Path(rest_tokens[1]).name == "pytest")
        )
        if is_pytest_exec:
            has_quiet = any(t in ("-q", "--quiet") for t in rest_tokens)
            has_help = any(t in ("-h", "--help", "-V", "--version") for t in rest_tokens)
            if not has_quiet and not has_help:
                return _build_healed_response(
                    category="test_flood",
                    target=first_stage,
                    tokens=5000,
                    reason=(
                        "🛡️ Blood-Brain Barrier Context-Tax Gate: Verbose test execution without quiet flag blocked.\n"
                        "Running pytest without `-q` floods cloud context with verbose test progress and stack traces.\n"
                        "👉 Action Required: Pass `-q --tb=short` to protect context (e.g. `pytest -q --tb=short` or `uv run pytest -q --tb=short`)."
                    ),
                    remedy="pytest -q --tb=short",
                    first_healed=f"{first_stage} -q --tb=short",
                )

        # 2. Directory Crawls (find, tree, ls -R)
        if is_strict_test and base_cmd == "find" and not has_bounding_filter:
            has_maxdepth = any(t in ("-maxdepth", "-depth") for t in rest_tokens)
            has_help = any(t in ("-h", "--help", "--version") for t in rest_tokens)
            if not has_maxdepth and not has_help:
                return {
                    "blocked": True,
                    "category": "directory_crawl",
                    "target": first_stage[:120],
                    "direct_tokens_saved": 4000,
                    "reason": (
                        "🛡️ Blood-Brain Barrier Context-Tax Gate: Unbounded directory crawl (`find` without `-maxdepth`) blocked.\n"
                        "Crawling unrestricted directory trees floods context with thousands of file paths.\n"
                        "👉 Action Required: Bound directory search depth with `-maxdepth 2` or pipe to `head -n 50`, e.g.:\n"
                        "   find . -maxdepth 2 -type f"
                    ),
                    "remedy": "find . -maxdepth 2 -type f",
                }

        if is_strict_test and base_cmd == "tree" and not has_bounding_filter:
            has_level = any(t in ("-L", "--level") for t in rest_tokens)
            has_help = any(t in ("-h", "--help", "--version") for t in rest_tokens)
            if not has_level and not has_help:
                return {
                    "blocked": True,
                    "category": "directory_crawl",
                    "target": first_stage[:120],
                    "direct_tokens_saved": 3000,
                    "reason": (
                        "🛡️ Blood-Brain Barrier Context-Tax Gate: Unbounded directory tree crawl (`tree` without `-L`) blocked.\n"
                        "👉 Action Required: Bound directory depth with `-L 2` or `-L 3` (e.g. `tree -L 2`)."
                    ),
                    "remedy": "tree -L 2",
                }

        if is_strict_test and base_cmd == "ls" and not has_bounding_filter:
            is_recursive = any(t.startswith("-") and "R" in t and not t.startswith("--") for t in rest_tokens)
            if is_recursive:
                return {
                    "blocked": True,
                    "category": "directory_crawl",
                    "target": first_stage[:120],
                    "direct_tokens_saved": 3000,
                    "reason": (
                        "🛡️ Blood-Brain Barrier Context-Tax Gate: Unbounded recursive directory listing (`ls -R`) blocked.\n"
                        "👉 Action Required: Inspect specific subdirectories directly or use `find . -maxdepth 2`."
                    ),
                    "remedy": "find . -maxdepth 2",
                }

        # 3. Unbounded File Dumps (cat, bat, more, less, head -n >180, tail -n >180)
        is_dump_cmd = base_cmd in ("cat", "bat", "more", "less")
        is_head_tail = base_cmd in ("head", "tail")

        if (is_dump_cmd or is_head_tail) and not has_bounding_filter:
            for tok in rest_tokens:
                if tok.startswith("-"):
                    continue
                if tok in (">", ">>", "2>", "&>", "<", "1>", "|"):
                    break
                clean_path = tok.strip("'\"")
                is_raw_log_path = (
                    "/receipts/" in clean_path
                    or clean_path.startswith("receipts/")
                    or "/logs/" in clean_path
                    or clean_path.startswith("logs/")
                    or clean_path.endswith((".log", ".err", ".jsonl"))
                    or ".log." in clean_path
                    or ".log" in clean_path
                )
                if is_raw_log_path:
                    return {
                        "blocked": True,
                        "category": "raw_log_dump",
                        "target": clean_path[:120],
                        "direct_tokens_saved": 4000,
                        "reason": (
                            f"🛡️ Blood-Brain Barrier Context-Tax Gate: Raw log/receipt dump (`{clean_path}`) via `{base_cmd}` blocked.\n"
                            f"Inspecting raw logs or receipts directly with shell dumps floods context with high-entropy text.\n"
                            f"👉 Action Required: Run on-device Recon Scout ($0.00 local metal):\n"
                            f'   body scout "{clean_path}" -q "<question>"  (or: uv run python -m whole_body_compute.local.scout "{clean_path}" -q "<question>")'
                        ),
                        "remedy": f'body scout "{clean_path}" -q "<question>"',
                    }

            head_n = None
            if is_head_tail:
                head_n = 10
                for idx, tok in enumerate(rest_tokens):
                    if tok == "-n" and idx < len(rest_tokens) - 1:
                        try:
                            head_n = int(rest_tokens[idx + 1])
                        except ValueError:
                            pass
                        break
                    elif tok.startswith("-") and tok[1:].isdigit():
                        head_n = int(tok[1:])
                        break
                if head_n <= (180 if is_strict_test else 800):
                    continue

            for tok in rest_tokens:
                if tok.startswith("-"):
                    continue
                if tok in (">", ">>", "2>", "&>", "<", "1>", "|"):
                    break
                clean_path = tok.strip("'\"")
                fpath = Path(clean_path)
                if not fpath.is_file() and not fpath.is_absolute():
                    alt = Path.cwd() / clean_path
                    if alt.is_file():
                        fpath = alt

                if fpath.is_file():
                    is_exempt = (
                        "/brain/" in clean_path
                        or "/.agents/" in clean_path
                        or fpath.name in ("README.md", "SKILL.md", "ACTIVE_PIPELINES.md", "pyproject.toml")
                    )
                    if not is_exempt:
                        try:
                            fsize = fpath.stat().st_size
                            raw_text = fpath.read_text(encoding="utf-8", errors="replace")
                            total_lines = raw_text.count("\n") + 1
                            full_tokens_est = max(1, fsize // 4)
                            is_raw_log = fpath.suffix in (".log", ".err", ".jsonl") and total_lines > 100

                            if (total_lines > 250 and is_strict_test) or is_raw_log:
                                n_info = f" ({base_cmd} -n {head_n})" if is_head_tail else ""
                                return {
                                    "blocked": True,
                                    "category": "unbounded_dump",
                                    "target": clean_path,
                                    "direct_tokens_saved": full_tokens_est,
                                    "reason": (
                                        f"🛡️ Blood-Brain Barrier Context-Tax Gate: Shell dump (`{fpath.name}`){n_info} blocked "
                                        f"({total_lines:,} lines / ~{full_tokens_est:,} tokens).\n"
                                        f"Running raw `{base_cmd}` on files >250 lines bypasses the slice-first context shield.\n"
                                        f"👉 Action Required: Use targeted `head -n 100 <file>` / `sed -n 'X,Yp' <file>`, "
                                        f"or run local on-device scout ($0.00):\n"
                                        f'   body scout "{clean_path}" -q "<your question>"  (or: uv run python -m whole_body_compute.local.scout "{clean_path}" -q "<your question>")'
                                    ),
                                    "remedy": f'body scout "{clean_path}" -q "<query>"',
                                }
                        except Exception:
                            pass

        if not is_strict_test:
            continue

        # 4. Unbounded Git Patch/Diff Floods
        if base_cmd == "git" and len(rest_tokens) >= 1 and not has_bounding_filter:
            git_sub = rest_tokens[0]
            git_args = rest_tokens[1:]
            has_compact_git_flag = any(
                t
                in (
                    "--stat",
                    "--shortstat",
                    "--numstat",
                    "--name-only",
                    "--name-status",
                    "--oneline",
                    "-s",
                    "--no-patch",
                    "-h",
                    "--help",
                )
                or t.startswith("--pretty=")
                or t.startswith("--format=")
                or t.startswith("-U")
                or t.startswith("--unified=")
                for t in git_args
            )
            has_patch_flag = any(t in ("-p", "--patch", "-u", "--full-index", "--binary") for t in git_args)
            is_unbounded_git_diff = False
            if git_sub == "diff" and not has_compact_git_flag and git_args:
                non_flag_args = [t for t in git_args if not t.startswith("-") and t != "--"]
                has_file_target = any(
                    Path(t).exists()
                    or Path(t).suffix in (".py", ".md", ".json", ".toml", ".yaml", ".yml", ".html", ".js", ".ts", ".sh", ".txt")
                    or t.startswith(("src/", "tests/", "docs/", "scripts/", "bin/", "benchmarks/", "./", "/"))
                    for t in non_flag_args
                )
                has_cached_or_ref = has_patch_flag or any(
                    t in ("--cached", "--staged") for t in git_args
                ) or (bool(non_flag_args) and not has_file_target)
                if has_cached_or_ref and not has_file_target:
                    is_unbounded_git_diff = True

            if (
                (git_sub == "log" and has_patch_flag and not has_compact_git_flag)
                or (git_sub == "show" and not has_compact_git_flag)
                or is_unbounded_git_diff
            ):
                remedy_cmd = f"{first_stage} --stat"
                auto_heal_git = (
                    f"{first_stage} | head -n 120"
                    if has_patch_flag
                    else f"{first_stage} --stat"
                )
                return _build_healed_response(
                    category="git_diff_flood",
                    target=first_stage,
                    tokens=4500,
                    reason=(
                        f"🛡️ Blood-Brain Barrier Context-Tax Gate: Unbounded `git {git_sub}` patch flood blocked.\n"
                        f"Running raw `git {git_sub}` without `--stat` or line bounds dumps multi-file patches into cloud context.\n"
                        f"👉 Action Required: Pass `--stat`, `--name-only`, or pipe to `head -n 120` (e.g. `{remedy_cmd}`)."
                    ),
                    remedy=remedy_cmd,
                    first_healed=auto_heal_git,
                )

        # 5. Unbounded rg / grep -r / grep -R
        if base_cmd == "rg" and not has_bounding_filter:
            has_rg_bound = any(
                t in ("-m", "--max-count", "-l", "--files-with-matches", "--files", "-c", "--count", "-h", "--help", "-V", "--version")
                or t.startswith("--max-count=")
                or (t.startswith("-m") and len(t) > 2 and t[2:].lstrip("=").isdigit())
                for t in rest_tokens
            )
            if not has_rg_bound:
                healed_rg = re.sub(
                    r"^(\s*(?:(?:sudo|nohup|time|\w+=\S+)\s+)*[^\s]*\brg)\b",
                    r"\1 --max-count=50 --max-columns=200",
                    first_stage,
                    count=1,
                )
                if len(pipe_stages) > 1:
                    healed_rg = " | ".join([healed_rg] + pipe_stages[1:])
                return _build_healed_response(
                    category="unbounded_search",
                    target=first_stage,
                    tokens=3500,
                    reason=(
                        "🛡️ Blood-Brain Barrier Context-Tax Gate: Unbounded `rg` search without `--max-count` blocked.\n"
                        "👉 Action Required: Bound ripgrep with `--max-count=50 --max-columns=200`."
                    ),
                    remedy=healed_rg,
                    first_healed=healed_rg,
                )

        if base_cmd in ("grep", "egrep", "fgrep") and not has_bounding_filter:
            is_recursive_grep = any(
                t in ("--recursive", "--dereference-recursive")
                or (t.startswith("-") and not t.startswith("--") and ("r" in t or "R" in t))
                for t in rest_tokens
            )
            has_grep_bound = any(
                t in ("-m", "--max-count", "-l", "--files-with-matches", "-c", "--count", "-h", "--help", "-V", "--version")
                or t.startswith("--max-count=")
                or (t.startswith("-m") and len(t) > 2 and t[2:].lstrip("=").isdigit())
                or (t.startswith("-") and not t.startswith("--") and ("l" in t or "c" in t))
                for t in rest_tokens
            )
            if is_recursive_grep and not has_grep_bound:
                return {
                    "blocked": True,
                    "category": "unbounded_search",
                    "target": first_stage[:120],
                    "direct_tokens_saved": 4000,
                    "reason": (
                        "🛡️ Blood-Brain Barrier Context-Tax Gate: Unbounded recursive `grep -r` / `grep -R` blocked.\n"
                        "Recursive grep crawls `.git`, `.venv`, and `node_modules` without line or match bounds.\n"
                        "👉 Action Required: Use bounded `rg -n --max-count=50 <pattern>` or local on-device scout ($0.00)."
                    ),
                    "remedy": "rg -n --max-count=50 <pattern>",
                }

        # 6. Docker & Kubernetes Log Floods
        is_container_logs = (
            (base_cmd in ("docker", "docker-compose", "kubectl") and len(rest_tokens) >= 1 and rest_tokens[0] == "logs")
            or (base_cmd == "docker" and len(rest_tokens) >= 2 and rest_tokens[0] == "compose" and rest_tokens[1] == "logs")
        )
        if is_container_logs and not has_bounding_filter:
            has_log_bound = any(
                t in ("--tail", "--since", "-n", "-h", "--help")
                or t.startswith("--tail=")
                or t.startswith("--since=")
                for t in rest_tokens
            )
            if not has_log_bound:
                healed_logs = re.sub(r"\blogs\b", "logs --tail 100", first_stage, count=1)
                return _build_healed_response(
                    category="container_log_flood",
                    target=first_stage,
                    tokens=5000,
                    reason=(
                        "🛡️ Blood-Brain Barrier Context-Tax Gate: Unbounded container log dump (`docker logs` / `kubectl logs`) blocked.\n"
                        "👉 Action Required: Pass `--tail 100` or `--since 10m`."
                    ),
                    remedy=healed_logs,
                    first_healed=healed_logs,
                )

        # 7. JS/Rust/Go Test Floods
        is_cargo_test = base_cmd == "cargo" and len(rest_tokens) >= 1 and rest_tokens[0] == "test"
        if is_cargo_test and not has_bounding_filter:
            has_cargo_quiet = any(t in ("-q", "--quiet", "-h", "--help") for t in rest_tokens)
            if not has_cargo_quiet:
                healed_cargo = re.sub(r"\bcargo\s+test\b", "cargo test -q", first_stage, count=1)
                return _build_healed_response(
                    category="test_flood",
                    target=first_stage,
                    tokens=4500,
                    reason=(
                        "🛡️ Blood-Brain Barrier Context-Tax Gate: Unbounded `cargo test` without `-q` blocked.\n"
                        "👉 Action Required: Pass `-q` / `--quiet` (`cargo test -q`)."
                    ),
                    remedy=healed_cargo,
                    first_healed=healed_cargo,
                )

        is_jest = base_cmd == "jest" or (
            base_cmd in ("npx", "pnpm", "yarn", "bunx") and len(rest_tokens) >= 1 and rest_tokens[0] == "jest"
        )
        if is_jest and not has_bounding_filter:
            has_jest_quiet = any(
                t in ("--silent", "--bail", "-b", "-h", "--help", "--version")
                or t.startswith("--bail=")
                for t in rest_tokens
            )
            if not has_jest_quiet:
                healed_jest = f"{first_stage} --silent"
                return _build_healed_response(
                    category="test_flood",
                    target=first_stage,
                    tokens=4500,
                    reason=(
                        "🛡️ Blood-Brain Barrier Context-Tax Gate: Unbounded `jest` execution without `--silent` blocked.\n"
                        "👉 Action Required: Pass `--silent` or `--bail`."
                    ),
                    remedy=healed_jest,
                    first_healed=healed_jest,
                )

        is_vitest = base_cmd == "vitest" or (
            base_cmd in ("npx", "pnpm", "yarn", "bunx") and len(rest_tokens) >= 1 and rest_tokens[0] == "vitest"
        )
        if is_vitest and not has_bounding_filter:
            has_vitest_run = "run" in rest_tokens
            has_vitest_quiet = any(
                t in ("--silent", "--bail", "-h", "--help", "--version")
                or t.startswith("--bail=")
                or t.startswith("--reporter=")
                for t in rest_tokens
            )
            if not (has_vitest_run and has_vitest_quiet):
                healed_vitest = first_stage if has_vitest_run else re.sub(r"\bvitest\b", "vitest run", first_stage, count=1)
                if not has_vitest_quiet:
                    healed_vitest = f"{healed_vitest} --silent"
                return _build_healed_response(
                    category="test_flood",
                    target=first_stage,
                    tokens=4500,
                    reason=(
                        "🛡️ Blood-Brain Barrier Context-Tax Gate: Unbounded `vitest` execution without `run --silent` blocked.\n"
                        "👉 Action Required: Run `vitest run --silent`."
                    ),
                    remedy=healed_vitest,
                    first_healed=healed_vitest,
                )

        is_go_test = base_cmd == "go" and len(rest_tokens) >= 1 and rest_tokens[0] == "test"
        if is_go_test and not has_bounding_filter:
            has_go_verbose = "-v" in rest_tokens
            has_go_bound = any(t in ("-short", "-failfast", "-h", "--help") for t in rest_tokens)
            if has_go_verbose and not has_go_bound:
                healed_go = re.sub(r"\s+-v\b", "", first_stage) + " -failfast"
                return _build_healed_response(
                    category="test_flood",
                    target=first_stage,
                    tokens=4000,
                    reason=(
                        "🛡️ Blood-Brain Barrier Context-Tax Gate: Verbose `go test -v` without bounds blocked.\n"
                        "👉 Action Required: Omit `-v` or pass `-failfast`."
                    ),
                    remedy=healed_go,
                    first_healed=healed_go,
                )

        # 8. System Logs & SQLite CLI Dumps
        is_log_show = base_cmd == "log" and len(rest_tokens) >= 1 and rest_tokens[0] == "show"
        is_journalctl = base_cmd == "journalctl"
        if (is_log_show or is_journalctl) and not has_bounding_filter:
            has_syslog_bound = any(
                t in ("--last", "-n", "--lines", "-S", "--since", "-U", "--until", "-h", "--help", "--version")
                or t.startswith("--last=")
                or t.startswith("--lines=")
                or t.startswith("--since=")
                or (t.startswith("-n") and len(t) > 2 and t[2:].isdigit())
                for t in rest_tokens
            )
            if not has_syslog_bound:
                remedy_syslog = "log show --last 5m | tail -n 100" if is_log_show else "journalctl -n 100 --no-pager"
                return {
                    "blocked": True,
                    "category": "system_log_flood",
                    "target": first_stage[:120],
                    "direct_tokens_saved": 6000,
                    "reason": (
                        f"🛡️ Blood-Brain Barrier Context-Tax Gate: Unbounded system log dump (`{base_cmd}`) blocked.\n"
                        f"👉 Action Required: Bound time/lines via `{remedy_syslog}`."
                    ),
                    "remedy": remedy_syslog,
                }

        if base_cmd == "sqlite3" and not has_bounding_filter:
            has_unbounded_sqlite = any(
                t == ".dump"
                or (
                    re.search(r"\bSELECT\b", t, re.IGNORECASE)
                    and not re.search(r"\bLIMIT\s+\d+", t, re.IGNORECASE)
                    and not re.search(r"\bCOUNT\s*\(", t, re.IGNORECASE)
                )
                for t in rest_tokens
            )
            if has_unbounded_sqlite:
                return {
                    "blocked": True,
                    "category": "sqlite_dump",
                    "target": first_stage[:120],
                    "direct_tokens_saved": 4000,
                    "reason": (
                        "🛡️ Blood-Brain Barrier Context-Tax Gate: Unbounded `sqlite3` dump or `SELECT` without `LIMIT` blocked.\n"
                        "👉 Action Required: Append `LIMIT 100` to your SQL query or pipe to `head -n 100`."
                    ),
                    "remedy": 'sqlite3 <db> "SELECT ... LIMIT 100"',
                }

    return {"blocked": False, "healed_command": raw_cmd}
