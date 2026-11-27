"""
Track 2.2: `body lower` Transcript-to-Code Scaffolder.

Extracts the top recurring deterministic Bash/tool loop from Claude Code session
transcripts (`<session.jsonl>` or `run_multi_harness_finops_audit(..., lowering=True)`)
and lowers it into a self-contained Tier 0/1 Python runner (`scripts/<workflow>_runner.py`)
with Hoare `{P} C {Q}` precondition/postcondition gates and a single bounded `claude -p`
callout strictly for the irreducible reasoning step.

Enforces PLG packaging:
  - `community` ($0/mo): 1 free lowered script per month (`YYYY-MM`) + unlimited dry-run previews.
  - `pro` ($29/mo) / `team` ($79/seat/mo) / `enterprise`: unlimited `body lower` compilation.
"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import stat
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from whole_body_compute.bundled import standalone_audit
from whole_body_compute.bundled.standalone_audit import run_multi_harness_finops_audit
from whole_body_compute.finops_audit import sanitize_lowering_analysis
from whole_body_compute.license import POLAR_CHECKOUT_URLS, LicenseManager
from whole_body_compute.organs.immune_vascular import EpithelialScreenFilter
from whole_body_compute.procedural_telemetry import get_wbc_data_dir
from whole_body_compute.redaction import redact_secrets

_ORIG_RUN_MULTI_HARNESS_FINOPS_AUDIT = run_multi_harness_finops_audit

PROCEDURAL_CATEGORIES: Tuple[str, ...] = (
    "git_pr_plumbing",
    "data_log_wrangling",
    "lint_format_loops",
)

_CANONICAL_FALLBACK_STEPS: Dict[str, List[List[str]]] = {
    "git_pr_plumbing": [
        ["git", "status", "--short"],
        ["git", "diff", "--stat"],
    ],
    "data_log_wrangling": [
        ["python3", "-c", "import json, sys; print('Tier 0 deterministic data pipeline ready')"],
    ],
    "lint_format_loops": [
        ["ruff", "check", "."],
    ],
}

_CANONICAL_VERB_MAP: Dict[str, List[str]] = {
    "git:status": ["git", "status", "--short"],
    "git:diff": ["git", "diff", "--stat"],
    "git:add": ["git", "add", "-A"],
    "git:log": ["git", "log", "-n", "5", "--oneline"],
    "git:branch": ["git", "branch", "--show-current"],
    "git:rev-parse": ["git", "rev-parse", "--abbrev-ref", "HEAD"],
    "gh:pr:checks": ["gh", "pr", "checks"],
    "gh:pr:view": ["gh", "pr", "view"],
    "gh:pr:status": ["gh", "pr", "status"],
    "ruff": ["ruff", "check", "."],
    "black": ["black", "--check", "."],
    "isort": ["isort", "--check-only", "."],
    "prettier": ["npx", "prettier", "--check", "."],
    "eslint": ["npx", "eslint", "."],
    "tsc:noEmit": ["npx", "tsc", "--noEmit"],
    "gofmt": ["gofmt", "-l", "."],
    "cargo:fmt": ["cargo", "fmt", "--check"],
    "cargo:clippy": ["cargo", "clippy"],
    "jq": ["jq", "."],
}

_REQUIRED_TOOLS_BY_CAT: Dict[str, List[str]] = {
    "git_pr_plumbing": ["git"],
    "data_log_wrangling": ["python3"],
    "lint_format_loops": ["ruff"],
}

_IRREDUCIBLE_PROMPT_BY_CAT: Dict[str, str] = {
    "git_pr_plumbing": (
        "Given the deterministic git status/diff summary above, draft a concise, accurate "
        "commit/PR summary or identify any blocking risk in <=6 lines."
    ),
    "data_log_wrangling": (
        "Given the structured JSON/log extract above, identify the root anomaly or "
        "required state transition in <=6 lines."
    ),
    "lint_format_loops": (
        "Given the deterministic lint/typecheck diagnostics above, emit only the minimal "
        "semantic fix plan needed for remaining non-auto-fixable errors."
    ),
}

_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")


def sanitize_lowering_output_text(text: str) -> str:
    """Redact credentials and neutralize prompt-injection spans before disk write or CLI output."""
    cleaned = redact_secrets(str(text or ""))
    try:
        cleaned, _ = EpithelialScreenFilter.sanitize(
            cleaned,
            source="cli_lower",
            wrap_untrusted_tags=False,
            record_antigen=False,
        )
    except Exception:
        pass
    return redact_secrets(cleaned)


def sanitize_lowering_payload(data: Any) -> Any:
    """Recursively sanitize all string leaves in a lowering result payload."""
    if isinstance(data, str):
        return sanitize_lowering_output_text(data)
    if isinstance(data, list):
        return [sanitize_lowering_payload(item) for item in data]
    if isinstance(data, dict):
        return {k: sanitize_lowering_payload(v) for k, v in data.items()}
    return data


def _current_month_key(now: Optional[float] = None) -> str:
    ts = float(now if now is not None else time.time())
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m")


def _is_system_root_symlink(path: Path) -> bool:
    return str(path) in ("/tmp", "/var", "/etc")


def _assert_no_symlink_components(path: Path, stop_at: Optional[Path] = None) -> None:
    if path.is_symlink():
        raise ValueError(f"Refusing symlink target: {path}")
    for parent in path.parents:
        if stop_at is not None and parent == stop_at:
            break
        if parent == parent.parent:
            break
        if _is_system_root_symlink(parent):
            continue
        if parent.is_symlink():
            raise ValueError(f"Refusing symlink parent directory: {parent}")


def get_lowering_quota_path(ledger_path: Optional[Path] = None) -> Path:
    """Resolve the monthly `lowering_quota.json` path under `WBC_DATA_DIR` or `~/.whole_body_compute`."""
    if ledger_path is not None:
        return Path(ledger_path).expanduser()
    env_p = os.environ.get("WBC_LOWERING_QUOTA_PATH", "").strip()
    if env_p:
        return Path(env_p).expanduser()
    return (get_wbc_data_dir() / "lowering_quota.json").expanduser()


def load_lowering_quota(
    ledger_path: Optional[Path] = None,
    month: Optional[str] = None,
) -> Dict[str, Any]:
    """Load the monthly `body lower` quota record keyed by `YYYY-MM`."""
    month_key = month or _current_month_key()
    path = get_lowering_quota_path(ledger_path)
    _assert_no_symlink_components(path)
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                months = raw.get("months") if isinstance(raw.get("months"), dict) else {}
                bucket = months.get(month_key) if isinstance(months.get(month_key), dict) else {}
                return {
                    "month": month_key,
                    "scripts_used": max(0, int(bucket.get("scripts_used", 0))),
                    "entries": list(bucket.get("entries") or []),
                }
        except Exception:
            pass
    return {"month": month_key, "scripts_used": 0, "entries": []}


def record_lowering_quota_usage(
    workflow: str,
    out_path: Path,
    ledger_path: Optional[Path] = None,
    month: Optional[str] = None,
) -> Dict[str, Any]:
    """Increment and persist the monthly `YYYY-MM` lowering quota ledger."""
    month_key = month or _current_month_key()
    path = get_lowering_quota_path(ledger_path)
    _assert_no_symlink_components(path)

    full_doc: Dict[str, Any] = {"months": {}}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                full_doc = loaded
        except Exception:
            full_doc = {"months": {}}

    months = full_doc.setdefault("months", {})
    if not isinstance(months, dict):
        months = {}
        full_doc["months"] = months

    bucket = months.get(month_key)
    if not isinstance(bucket, dict):
        bucket = {"scripts_used": 0, "entries": []}
    bucket["scripts_used"] = max(0, int(bucket.get("scripts_used", 0))) + 1
    entries = list(bucket.get("entries") or [])
    entries.append(
        {
            "workflow": sanitize_lowering_output_text(workflow),
            "out_path": sanitize_lowering_output_text(str(out_path)),
            "created_at": int(time.time()),
        }
    )
    bucket["entries"] = entries[-50:]
    months[month_key] = bucket

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(full_doc, indent=2) + "\n", encoding="utf-8")
    return {
        "month": month_key,
        "scripts_used": int(bucket["scripts_used"]),
        "entries": bucket["entries"],
    }


def _slugify_workflow_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", (name or "lowered_workflow").strip().lower()).strip("_")
    return slug or "lowered_workflow"


def _assert_safe_input_file(path: Path) -> Path:
    raw = Path(path).expanduser()
    _assert_no_symlink_components(raw)
    if not raw.is_file():
        raise FileNotFoundError(f"Session transcript not found: {raw}")
    return raw.resolve()


def _resolve_safe_output_path(
    *,
    workflow_slug: str,
    out_path: Optional[Union[Path, str]],
    base_dir: Path,
) -> Path:
    resolved_base = base_dir.resolve()
    if out_path is not None:
        raw = Path(out_path).expanduser()
        is_rel = not raw.is_absolute()
        candidate = (resolved_base / raw) if is_rel else raw
        _assert_no_symlink_components(candidate, stop_at=resolved_base if is_rel else None)
        if is_rel and not candidate.resolve().is_relative_to(resolved_base):
            raise ValueError(f"Refusing relative `--out` outside workspace containment: {out_path}")
    else:
        scripts_dir = resolved_base / "scripts"
        if scripts_dir.is_symlink():
            raise ValueError(f"Refusing symlink scripts directory: {scripts_dir}")
        fname = workflow_slug if workflow_slug.endswith("_runner") else f"{workflow_slug}_runner"
        candidate = scripts_dir / f"{fname}.py"
        _assert_no_symlink_components(candidate, stop_at=resolved_base)
        if not candidate.resolve().is_relative_to(resolved_base):
            raise ValueError(f"Refusing output path outside workspace containment: {candidate}")

    if candidate.exists() and candidate.is_dir():
        raise ValueError(f"Output target is a directory, expected file: {candidate}")
    return candidate


def _extract_session_trace(session_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, List[str]]]:
    """Parse a single Claude Code `.jsonl` transcript into lowering trace + raw command buckets."""
    ordered_tools: List[Dict[str, Any]] = []
    seen_tool_ids: set[str] = set()
    turn0_declared_mcps: List[str] = []
    turn0_declared_set: set[str] = set()
    unbounded_read_calls = 0
    noisy_bash_calls = 0
    cmd_by_cat: Dict[str, List[str]] = {c: [] for c in PROCEDURAL_CATEGORIES}
    planner_turns = 0
    conv_bytes = 0

    raw_text = session_path.read_text(encoding="utf-8", errors="ignore")
    conv_bytes = len(raw_text.encode("utf-8", errors="ignore"))

    for line_idx, line in enumerate(raw_text.splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue

        msg_dict = obj.get("message") if isinstance(obj.get("message"), dict) else {}
        role = str(obj.get("type") or obj.get("role") or msg_dict.get("role") or "").lower()
        if role in ("assistant", "model", "planner_response"):
            planner_turns += 1

        for t_list_key in ("tools", "mcp_servers"):
            t_list = obj.get(t_list_key)
            if isinstance(t_list, list):
                for entry in t_list:
                    ename = str(entry.get("name") or "") if isinstance(entry, dict) else str(entry or "")
                    if ename.startswith("mcp__"):
                        parts = ename.split("__", 2)
                        if len(parts) >= 2 and parts[1] and parts[1] not in turn0_declared_set:
                            turn0_declared_set.add(parts[1])
                            turn0_declared_mcps.append(parts[1])

        msg_id = str(msg_dict.get("id") or obj.get("messageId") or "").strip()
        req_id = str(obj.get("requestId") or msg_dict.get("requestId") or "").strip()
        turn_key_str = f"{msg_id or f'_line_{line_idx}'}|{req_id}"

        msg_content = msg_dict.get("content") if "content" in msg_dict else obj.get("content")
        if not isinstance(msg_content, list):
            continue
        for blk in msg_content:
            if not isinstance(blk, dict):
                continue
            if blk.get("type") != "tool_use":
                continue
            tname = str(blk.get("name") or "tool")
            tid = str(blk.get("id") or "").strip()
            if tid and tid in seen_tool_ids:
                continue
            if tid:
                seen_tool_ids.add(tid)

            norm_name = tname.strip().lower()
            inp = blk.get("input") if isinstance(blk.get("input"), dict) else {}
            cmd_str = str(inp.get("command") or "") if norm_name == "bash" else ""
            cat_lbl: Optional[str] = None
            verb_lbl = ""
            is_noisy_cmd = False
            if norm_name == "bash" and cmd_str:
                cat_lbl, verb_lbl, is_noisy_cmd = standalone_audit._classify_bash_command(cmd_str)
            is_unbounded_read = (
                norm_name in ("read", "notebookread")
                and not (inp.get("limit") or inp.get("offset") or inp.get("lines"))
            )
            if is_unbounded_read:
                unbounded_read_calls += 1
            if is_noisy_cmd:
                noisy_bash_calls += 1

            mcp_srv = ""
            if norm_name.startswith("mcp__"):
                mparts = tname.split("__", 2)
                if len(mparts) >= 2:
                    mcp_srv = mparts[1]

            safe_cmd = sanitize_lowering_output_text(cmd_str)[:240] if cmd_str else ""
            ordered_tools.append(
                {
                    "tool": tname,
                    "norm": norm_name,
                    "cmd": safe_cmd,
                    "cat": cat_lbl,
                    "verb": verb_lbl,
                    "turn_key": turn_key_str,
                    "unbounded_read": is_unbounded_read,
                    "noisy_bash": is_noisy_cmd,
                    "mcp_server": mcp_srv,
                }
            )
            if cat_lbl in cmd_by_cat and safe_cmd:
                cmd_by_cat[cat_lbl].append(safe_cmd)

    cid = session_path.stem
    rec = {
        "conv_id": cid,
        "planner_turns": max(1, planner_turns) if ordered_tools else planner_turns,
        "exact_api_spend_usd": 0.0,
        "hal_saved_usd": 0.0,
        "conv_bytes": conv_bytes,
    }
    trace = {
        "ordered_tools": ordered_tools,
        "turn0_declared_mcps": turn0_declared_mcps,
        "unbounded_read_calls": unbounded_read_calls,
        "noisy_bash_calls": noisy_bash_calls,
    }
    lowering_analysis = standalone_audit._analyze_claude_lowering_choreography(
        claude_recs=[rec],
        traces_by_id={cid: trace},
        turn0_static_overhead={},
        ws_dir=session_path.parent,
        h_dir=Path.home(),
        ci_dir=session_path.parent,
        days=30,
        in_price=5.0,
        total_monthly_usd=0.0,
    )
    return sanitize_lowering_analysis(lowering_analysis), trace, cmd_by_cat


def _collect_commands_from_dir(search_dir: Optional[Path]) -> Dict[str, List[str]]:
    buckets: Dict[str, List[str]] = {c: [] for c in PROCEDURAL_CATEGORIES}
    if search_dir is None or not search_dir.is_dir():
        return buckets
    for jpath in sorted(search_dir.rglob("*.jsonl"))[:50]:
        if not standalone_audit._is_safe_contained_file(jpath, search_dir):
            continue
        try:
            _, _, per_file = _extract_session_trace(jpath)
            for cat in PROCEDURAL_CATEGORIES:
                buckets[cat].extend(per_file.get(cat) or [])
        except Exception:
            continue
    return buckets


def _command_to_argv_steps(cmd: str) -> List[List[str]]:
    """Convert a sanitized shell command preview into safe `subprocess.run(..., shell=False)` argv lists."""
    safe_cmd = sanitize_lowering_output_text(cmd).strip()
    if not safe_cmd:
        return []
    try:
        tokens = shlex.split(safe_cmd)
    except ValueError:
        return []
    if not tokens:
        return []

    steps: List[List[str]] = []
    cur: List[str] = []
    for tok in tokens:
        if tok in ("&&", "||", ";"):
            if cur:
                steps.append(cur)
                cur = []
            continue
        if tok == "|":
            # Avoid shell piping; keep the primary producer command segment.
            break
        cur.append(tok)
    if cur:
        steps.append(cur)

    cleaned_steps: List[List[str]] = []
    for st in steps:
        while st and _ENV_ASSIGN_RE.match(st[0]):
            st = st[1:]
        if not st:
            continue
        # Never embed an interactive/nested `claude` invocation inside deterministic C steps.
        if Path(st[0]).name == "claude":
            continue
        cleaned_steps.append(st)
    return cleaned_steps


def _commands_from_patterns(category: str, cat_info: Dict[str, Any], raw_cmds: List[str]) -> List[List[str]]:
    """Build deterministic argv steps from extracted commands or pattern previews."""
    argv_steps: List[List[str]] = []
    seen: set[Tuple[str, ...]] = set()

    for cmd in raw_cmds:
        for argv in _command_to_argv_steps(cmd):
            key = tuple(argv)
            if key not in seen:
                seen.add(key)
                argv_steps.append(argv)
        if len(argv_steps) >= 12:
            break

    if not argv_steps:
        for pat in cat_info.get("patterns") or []:
            pat_str = str(pat)
            m = re.search(r"\(([^()]+)\)\s*$", pat_str)
            if m:
                for argv in _command_to_argv_steps(m.group(1)):
                    key = tuple(argv)
                    if key not in seen:
                        seen.add(key)
                        argv_steps.append(argv)
            verb_part = pat_str.split("(", 1)[0]
            for vtok in [x.strip() for x in verb_part.split("->") if x.strip()]:
                mapped = _CANONICAL_VERB_MAP.get(vtok)
                if mapped:
                    key = tuple(mapped)
                    if key not in seen:
                        seen.add(key)
                        argv_steps.append(list(mapped))
            if len(argv_steps) >= 12:
                break

    if not argv_steps:
        argv_steps = [list(x) for x in _CANONICAL_FALLBACK_STEPS.get(category, [["git", "status", "--short"]])]

    return argv_steps[:12]


def _select_top_workflow(
    lowering_analysis: Dict[str, Any],
    cmd_by_cat: Dict[str, List[str]],
) -> Optional[Dict[str, Any]]:
    """Select the top recurring procedural tool loop across the 3 lowerable categories."""
    cats = lowering_analysis.get("categories") if isinstance(lowering_analysis.get("categories"), dict) else {}
    scored: List[Tuple[Tuple[int, int, int, int], str, Dict[str, Any]]] = []
    for idx, cat in enumerate(PROCEDURAL_CATEGORIES):
        info = cats.get(cat) if isinstance(cats.get(cat), dict) else {}
        turns = int(info.get("turns", 0) or 0)
        calls = int(info.get("calls", 0) or 0)
        seqs = int(info.get("sequences", 0) or 0)
        if calls > 0 or seqs > 0 or turns > 0:
            scored.append(((turns, calls, seqs, -idx), cat, info))

    if scored:
        scored.sort(reverse=True)
        _, top_cat, top_info = scored[0]
        raw_cmds = list(cmd_by_cat.get(top_cat) or [])
        argv_steps = _commands_from_patterns(top_cat, top_info, raw_cmds)
        return {
            "category": top_cat,
            "turns": int(top_info.get("turns", 0) or 0),
            "calls": int(top_info.get("calls", 0) or 0),
            "sequences": int(top_info.get("sequences", 0) or 0),
            "patterns": list(top_info.get("patterns") or []),
            "recommendation": str(top_info.get("recommendation") or ""),
            "argv_steps": argv_steps,
        }

    # Single-session fallback when <3 turns occurred in a lone transcript
    fallback_cat = max(
        PROCEDURAL_CATEGORIES,
        key=lambda c: len(cmd_by_cat.get(c) or []),
    )
    raw_cmds = list(cmd_by_cat.get(fallback_cat) or [])
    if not raw_cmds:
        return None
    argv_steps = _commands_from_patterns(fallback_cat, {}, raw_cmds)
    return {
        "category": fallback_cat,
        "turns": len(raw_cmds),
        "calls": len(raw_cmds),
        "sequences": 1,
        "patterns": [sanitize_lowering_output_text(" -> ".join(" ".join(s[:2]) for s in argv_steps[:5]))],
        "recommendation": "Tier 0/1 deterministic Python spine",
        "argv_steps": argv_steps,
    }


def _infer_required_tools(category: str, argv_steps: List[List[str]]) -> List[str]:
    tools: List[str] = list(_REQUIRED_TOOLS_BY_CAT.get(category, []))
    for argv in argv_steps:
        if not argv:
            continue
        exe = Path(argv[0]).name
        if exe in ("git", "gh", "ruff", "jq", "python3", "black", "isort", "npx", "uv", "cargo", "gofmt"):
            if exe not in tools:
                tools.append(exe)
    return tools


def render_lowered_runner_script(
    *,
    workflow_name: str,
    category: str,
    argv_steps: List[List[str]],
    source_label: str,
    patterns: Optional[List[str]] = None,
) -> str:
    """Render a deterministic Tier 0/1 Python runner with `{P} C {Q}` gates and 1 bounded `claude -p` callout."""
    safe_wf = sanitize_lowering_output_text(workflow_name)
    safe_cat = sanitize_lowering_output_text(category)
    safe_src = sanitize_lowering_output_text(source_label)
    safe_patterns = [sanitize_lowering_output_text(p) for p in (patterns or [])][:5]
    safe_steps = [
        [sanitize_lowering_output_text(tok) for tok in argv]
        for argv in argv_steps
        if argv
    ] or [list(x) for x in _CANONICAL_FALLBACK_STEPS.get(category, [["git", "status", "--short"]])]
    req_tools = _infer_required_tools(category, safe_steps)
    irreducible_prompt = sanitize_lowering_output_text(
        _IRREDUCIBLE_PROMPT_BY_CAT.get(
            category,
            "Summarize the deterministic step output and propose only the minimal next action.",
        )
    )

    safe_src_doc = safe_src.replace("\\", "\\\\").replace('"""', '\\"\\"\\"').replace("'''", "\\'\\'\\'").replace("\r", " ").replace("\n", " ")

    template = f'''#!/usr/bin/env python3
"""
Deterministic Tier 0/1 Runner generated by `body lower`.

Workflow : {safe_wf}
Category : {safe_cat}
Source   : {safe_src_doc}

Hoare Contract {{P}} C {{Q}}:
  - Precondition {{P}} : Verify required CLI tools, workspace/git repository state & branch guard.
  - Deterministic C  : Execute lowered commands directly via subprocess (zero LLM turn burn).
  - Reasoning Call   : Single bounded `claude -p --max-turns 1` callout (`WBC_HEADLESS=1` / `body begin --ci` aware).
  - Postcondition {{Q}}: Verify all deterministic steps succeeded and repository/output invariants hold.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple

WORKFLOW_NAME: str = {json.dumps(safe_wf)}
WORKFLOW_CATEGORY: str = {json.dumps(safe_cat)}
REQUIRED_TOOLS: List[str] = {json.dumps(req_tools)}
DETECTED_PATTERNS: List[str] = {json.dumps(safe_patterns)}
LOWERED_ARGV_STEPS: List[List[str]] = {json.dumps(safe_steps, indent=4)}
IRREDUCIBLE_REASONING_PROMPT: str = {json.dumps(irreducible_prompt)}


def check_preconditions(
    cwd: Path,
    expected_branch: Optional[str] = None,
    require_clean_git: bool = False,
) -> Tuple[bool, List[str]]:
    """Hoare Precondition {{P}}: validate workspace, required binaries, and git repo/branch state."""
    notes: List[str] = []
    if not cwd.is_dir():
        return False, [f"Workspace directory does not exist: {{cwd}}"]

    missing = [tool for tool in REQUIRED_TOOLS if shutil.which(tool) is None]
    if missing:
        return False, [f"Missing required CLI tool(s): {{', '.join(missing)}}"]
    notes.append(f"tools_ok:{{','.join(REQUIRED_TOOLS)}}")

    if WORKFLOW_CATEGORY == "git_pr_plumbing" or "git" in REQUIRED_TOOLS:
        in_repo = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        if in_repo.returncode != 0 or in_repo.stdout.strip().lower() != "true":
            return False, ["Not inside a git work tree"]

        branch_proc = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        branch = branch_proc.stdout.strip()
        notes.append(f"git_branch:{{branch or 'unknown'}}")
        if expected_branch and branch != expected_branch:
            return False, [f"Expected branch {{expected_branch!r}}, found {{branch!r}}"]

        st_proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        dirty = bool(st_proc.stdout.strip())
        notes.append("git_dirty" if dirty else "git_clean")
        if require_clean_git and dirty:
            return False, ["Git working tree is dirty (require_clean_git=True)"]

    return True, notes


def run_deterministic_steps(
    cwd: Path,
    dry_run: bool = False,
    fail_fast: bool = True,
) -> List[Dict[str, Any]]:
    """Deterministic body C: run lowered Tier 0/1 commands directly without an LLM loop."""
    results: List[Dict[str, Any]] = []
    for idx, argv in enumerate(LOWERED_ARGV_STEPS, start=1):
        if dry_run:
            results.append({{"step": idx, "argv": argv, "returncode": 0, "dry_run": True}})
            continue
        proc = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        results.append(
            {{
                "step": idx,
                "argv": argv,
                "returncode": proc.returncode,
                "stdout": proc.stdout[-2000:],
                "stderr": proc.stderr[-1000:],
            }}
        )
        if fail_fast and proc.returncode != 0:
            break
    return results


def ensure_ci_shield_context(cwd: Path, auto_init: bool = False) -> Dict[str, Any]:
    """Ensure headless WBC Blood-Brain Barrier context (`body begin --ci` / `WBC_HEADLESS=1`)."""
    settings_path = cwd / ".claude" / "settings.json"
    if auto_init and not settings_path.is_file() and shutil.which("body"):
        env = {{**os.environ, "WBC_HEADLESS": "1"}}
        subprocess.run(
            ["body", "begin", "--ci", "--skip-audit", "--json"],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    return {{
        "wbc_headless": os.environ.get("WBC_HEADLESS", "1"),
        "claude_settings_present": settings_path.is_file(),
        "ci_bootstrap_cmd": "body begin --ci",
    }}


def run_bounded_claude_callout(
    cwd: Path,
    step_results: List[Dict[str, Any]],
    *,
    dry_run: bool = False,
    skip_claude: bool = False,
    ensure_ci_shield: bool = False,
) -> Dict[str, Any]:
    """Single bounded `claude -p` callout strictly for the irreducible reasoning step."""
    shield_info = ensure_ci_shield_context(cwd, auto_init=ensure_ci_shield and not dry_run)
    if dry_run or skip_claude:
        return {{"invoked": False, "reason": "dry_run_or_skipped", "shield": shield_info}}

    if shutil.which("claude") is None:
        return {{"invoked": False, "reason": "claude_cli_not_found", "shield": shield_info}}

    digest = json.dumps(
        [{{"step": r["step"], "rc": r["returncode"], "out": str(r.get("stdout", ""))[:400]}} for r in step_results]
    )
    bounded_prompt = f"{{IRREDUCIBLE_REASONING_PROMPT}}\\n\\nDeterministic context:\\n{{digest}}"
    env = {{**os.environ, "WBC_HEADLESS": "1"}}
    proc = subprocess.run(
        ["claude", "-p", bounded_prompt, "--max-turns", "1", "--output-format", "json"],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    return {{
        "invoked": True,
        "bounded_max_turns": 1,
        "returncode": proc.returncode,
        "output": proc.stdout[-2000:],
        "shield": shield_info,
    }}


def verify_postconditions(
    cwd: Path,
    step_results: List[Dict[str, Any]],
    dry_run: bool = False,
) -> Tuple[bool, List[str]]:
    """Hoare Postcondition {{Q}}: verify deterministic steps and workspace invariants passed."""
    if dry_run:
        return True, ["dry_run_postcondition_ok"]

    failed = [r for r in step_results if int(r.get("returncode", 1)) != 0]
    if failed:
        return False, [f"Step {{r['step']}} failed with exit code {{r['returncode']}}" for r in failed]

    if WORKFLOW_CATEGORY == "git_pr_plumbing" or "git" in REQUIRED_TOOLS:
        chk = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        if chk.returncode != 0:
            return False, ["Git repository invariant failed after execution"]

    return True, [f"all_{{len(step_results)}}_steps_passed"]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=f"Lowered Tier 0/1 runner: {{WORKFLOW_NAME}}")
    parser.add_argument("--cwd", default=".", help="Target workspace directory")
    parser.add_argument("--expected-branch", default=None, help="Optional git branch precondition")
    parser.add_argument("--require-clean-git", action="store_true", help="Fail precondition if git tree is dirty")
    parser.add_argument("--dry-run", action="store_true", help="Preview {{P}} C {{Q}} without mutating state")
    parser.add_argument("--skip-claude", action="store_true", help="Skip the bounded `claude -p` reasoning callout")
    parser.add_argument("--ensure-ci-shield", action="store_true", help="Run `body begin --ci` if `.claude/settings.json` is absent")
    parser.add_argument("--json", action="store_true", help="Output structured JSON")
    args = parser.parse_args(argv)

    cwd = Path(args.cwd).expanduser().resolve()
    p_ok, p_notes = check_preconditions(
        cwd,
        expected_branch=args.expected_branch,
        require_clean_git=bool(args.require_clean_git),
    )
    if not p_ok:
        payload = {{"ok": False, "phase": "precondition_P", "notes": p_notes}}
        print(json.dumps(payload, indent=2) if args.json else f"Precondition {{{{P}}}} failed: {{p_notes}}")
        return 1

    step_results = run_deterministic_steps(cwd, dry_run=bool(args.dry_run))
    claude_res = run_bounded_claude_callout(
        cwd,
        step_results,
        dry_run=bool(args.dry_run),
        skip_claude=bool(args.skip_claude),
        ensure_ci_shield=bool(args.ensure_ci_shield),
    )
    q_ok, q_notes = verify_postconditions(cwd, step_results, dry_run=bool(args.dry_run))

    payload = {{
        "ok": q_ok,
        "workflow": WORKFLOW_NAME,
        "category": WORKFLOW_CATEGORY,
        "precondition_P": {{"ok": p_ok, "notes": p_notes}},
        "steps_C": step_results,
        "claude_callout": claude_res,
        "postcondition_Q": {{"ok": q_ok, "notes": q_notes}},
    }}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"[{{WORKFLOW_NAME}}] P={{p_ok}} steps={{len(step_results)}} Q={{q_ok}}")
    return 0 if q_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''

    sanitized_src = sanitize_lowering_output_text(template)
    # Validate syntax strictly after redaction/sanitization
    ast.parse(sanitized_src)
    return sanitized_src


def lower_workflow(
    session_path: Optional[Union[Path, str]] = None,
    *,
    from_audit: bool = False,
    ci_dir: Optional[Union[Path, str]] = None,
    out_path: Optional[Union[Path, str]] = None,
    name: Optional[str] = None,
    dry_run: bool = False,
    workspace_dir: Optional[Union[Path, str]] = None,
    license_manager: Optional[LicenseManager] = None,
    ledger_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Lower the top recurring procedural tool loop from a Claude Code `.jsonl` session
    or `run_multi_harness_finops_audit(..., lowering=True)` into a deterministic Python runner.
    """
    base_dir = Path(workspace_dir).expanduser() if workspace_dir else Path.cwd()

    if session_path is None and not from_audit and ci_dir is None:
        raise ValueError("Specify a <session.jsonl> path or `--from-audit` (with optional `--ci-dir`).")

    ci_dir_path: Optional[Path] = None
    if ci_dir is not None:
        raw_ci = Path(ci_dir).expanduser()
        _assert_no_symlink_components(raw_ci)
        if not raw_ci.is_dir():
            raise FileNotFoundError(f"CI transcript directory not found: {raw_ci}")
        ci_dir_path = raw_ci.resolve()

    if session_path is not None:
        safe_session = _assert_safe_input_file(Path(session_path))
        lowering_analysis, _, cmd_by_cat = _extract_session_trace(safe_session)
        source_label = f"session:{safe_session.name}"
    else:
        audit_fn = (
            standalone_audit.run_multi_harness_finops_audit
            if run_multi_harness_finops_audit is _ORIG_RUN_MULTI_HARNESS_FINOPS_AUDIT
            else run_multi_harness_finops_audit
        )
        audit = audit_fn(
            days=30,
            ci_dir=ci_dir_path,
            lowering=True,
        )
        raw_la = audit.get("lowering_analysis") if isinstance(audit, dict) else None
        lowering_analysis = sanitize_lowering_analysis(raw_la if isinstance(raw_la, dict) else {})
        cmd_by_cat = _collect_commands_from_dir(ci_dir_path)
        source_label = f"audit:{ci_dir_path}" if ci_dir_path else "audit:local"

    selected = _select_top_workflow(lowering_analysis, cmd_by_cat)
    if selected is None:
        raise ValueError("No lowerable procedural tool loops detected in the provided transcript(s).")

    category = selected["category"]
    workflow_slug = _slugify_workflow_name(name or category)
    target_path = _resolve_safe_output_path(
        workflow_slug=workflow_slug,
        out_path=out_path,
        base_dir=base_dir,
    )

    script_text = render_lowered_runner_script(
        workflow_name=workflow_slug,
        category=category,
        argv_steps=selected["argv_steps"],
        source_label=source_label,
        patterns=selected.get("patterns"),
    )

    mgr = license_manager or LicenseManager()
    ledger = load_lowering_quota(ledger_path)
    month_key = str(ledger.get("month") or _current_month_key())
    scripts_used = int(ledger.get("scripts_used", 0))

    gate = mgr.check_feature_gate(
        "code_lowering",
        lowering_scripts_used=scripts_used,
    )
    allowed = bool(gate.get("allowed", False))
    current_tier = str(gate.get("current_tier", "community"))

    if dry_run or not allowed:
        result: Dict[str, Any] = {
            "ok": True,
            "written": False,
            "dry_run_preview": True,
            "gated_by_license": not allowed,
            "current_tier": current_tier,
            "required_tier": str(gate.get("required_tier", "pro")),
            "month": month_key,
            "monthly_scripts_used": scripts_used,
            "monthly_limit": 1 if current_tier == "community" else None,
            "workflow": workflow_slug,
            "category": category,
            "target_path": str(target_path),
            "out_path": None,
            "script_preview": script_text,
            "patterns": selected.get("patterns") or [],
            "step_count": len(selected["argv_steps"]),
            "upgrade_nudge": str(gate.get("upgrade_nudge") or ""),
            "checkout_url": POLAR_CHECKOUT_URLS.get("pro"),
            "lowering_analysis": lowering_analysis,
        }
        return sanitize_lowering_payload(result)

    # Re-verify target path safety immediately before writing
    _assert_no_symlink_components(
        target_path,
        stop_at=base_dir.resolve() if out_path is None else None,
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(script_text, encoding="utf-8")
    try:
        target_path.chmod(target_path.stat().st_mode | stat.S_IXUSR)
    except OSError:
        pass

    updated_ledger = record_lowering_quota_usage(
        workflow=workflow_slug,
        out_path=target_path,
        ledger_path=ledger_path,
        month=month_key,
    )

    result = {
        "ok": True,
        "written": True,
        "dry_run_preview": False,
        "gated_by_license": False,
        "current_tier": current_tier,
        "required_tier": str(gate.get("required_tier", "community")),
        "month": month_key,
        "monthly_scripts_used": int(updated_ledger["scripts_used"]),
        "monthly_limit": 1 if current_tier == "community" else None,
        "workflow": workflow_slug,
        "category": category,
        "target_path": str(target_path),
        "out_path": str(target_path),
        "script_preview": script_text,
        "patterns": selected.get("patterns") or [],
        "step_count": len(selected["argv_steps"]),
        "upgrade_nudge": "",
        "lowering_analysis": lowering_analysis,
    }
    return sanitize_lowering_payload(result)
