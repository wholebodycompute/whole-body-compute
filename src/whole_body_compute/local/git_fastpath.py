"""Ephemeral Shadow Index & Atomic CAS Git Fast-Path for Multi-Agent Concurrency.

Provides zero-lock, race-free git operations across multiple concurrent OS processes
and subagents by leveraging temporary process-private shadow index files (GIT_INDEX_FILE)
and atomic branch ref Compare-And-Swap (git update-ref <ref> <new> <old>).
"""

from __future__ import annotations

import ast
import contextlib

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows native fallback
    fcntl = None  # type: ignore[assignment]
import hashlib
import json
import os
import random
import re
import socket
import subprocess
import tempfile
import time
import urllib.request
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from whole_body_compute.local.shadow_ledger import (
    is_external_models_drive_mounted,
    is_infra_error,
    record_shadow_eval,
    strip_llm_fences_and_thoughts,
)

# Fast secret scanning regex (detects common API keys, private keys, and tokens)
_FAST_SECRET_RE = re.compile(
    r"("
    r"sk-[a-zA-Z0-9_\-]{15,}"
    r"|ghp_[a-zA-Z0-9]{20,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|AIza[0-9A-Za-z\-_]{35}"
    r"|-----BEGIN (?:RSA |EC |OPENSSH |PRIVATE )?KEY-----"
    r")"
)

FORBIDDEN_SECRET_BASENAMES = {
    ".env",
    ".env.local",
    "id_rsa",
    "id_rsa.pub",
    "credentials.json",
}


def resolve_repo_root(repo_root: Path | str | None = None) -> Path:
    """Resolve the git repository root directory."""
    if repo_root is not None:
        p = Path(repo_root).resolve()
        if (p / ".git").exists():
            return p
        curr = p
        while curr != curr.parent:
            if (curr / ".git").exists():
                return curr
            curr = curr.parent
        return p

    with contextlib.suppress(OSError, subprocess.SubprocessError):
        res = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        return Path(res.stdout.strip()).resolve()

    curr = Path.cwd().resolve()
    while curr != curr.parent:
        if (curr / ".git").exists():
            return curr
        curr = curr.parent
    return Path.cwd().resolve()


def get_git_dir(repo_root: Path) -> Path:
    """Resolve the actual .git directory (handles git worktrees/submodules)."""
    git_dir = repo_root / ".git"
    if git_dir.is_file():
        with contextlib.suppress(OSError):
            content = git_dir.read_text(encoding="utf-8").strip()
            if content.startswith("gitdir:"):
                target = content.split("gitdir:", 1)[1].strip()
                return (repo_root / target).resolve()
    return git_dir


def normalize_repo_path(path: str | Path, repo_root: Path) -> str:
    """Normalize a path to a repo-relative POSIX string."""
    p = Path(path)
    if not p.is_absolute():
        candidate = (Path.cwd() / p).resolve()
        if candidate.is_relative_to(repo_root):
            p = candidate
        else:
            p = (repo_root / p).resolve()
    else:
        p = p.resolve()
    try:
        return p.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def resolve_session_id(session_id: str | None = None) -> str:
    """Resolve active agent/conversation session identifier."""
    return (
        session_id
        or os.environ.get("WBC_SESSION_ID")
        or os.environ.get("AGY_CONVERSATION_ID")
        or os.environ.get("CONVERSATION_ID")
        or "default"
    )


def _load_repo_zone_config(repo_root: Path) -> dict[str, Any]:
    """Load optional repo-local `.wbc_reflex.json` (zones + blocked_paths)."""
    cfg_path = repo_root / ".wbc_reflex.json"
    if not cfg_path.is_file():
        return {}
    with contextlib.suppress(OSError, json.JSONDecodeError, TypeError, ValueError):
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    return {}


def _is_blocked_repo_path(rel_path: str, blocked_paths: list[str]) -> str | None:
    norm = rel_path.replace("\\", "/").lstrip("./")
    for bp in blocked_paths:
        bp_norm = str(bp).strip().replace("\\", "/").lstrip("./")
        if bp_norm and (norm == bp_norm or norm.startswith(bp_norm.rstrip("/") + "/")):
            return bp_norm
    return None


def _matches_zone_prefixes(rel_path: str, prefixes: list[Any]) -> bool:
    norm = rel_path.replace("\\", "/").lstrip("./")
    for pref in prefixes:
        p_norm = str(pref).strip().replace("\\", "/").lstrip("./")
        if not p_norm:
            continue
        if norm == p_norm or (p_norm.endswith("/") and norm.startswith(p_norm)) or norm.startswith(p_norm.rstrip("/") + "/"):
            return True
    return False


def record_file_touch(
    paths: list[str] | str,
    session_id: str | None = None,
    repo_root: Path | None = None,
    *,
    record_session_edit: bool = True,
) -> None:
    """Atomically record file touches for an agent session under .git lock.

    Maintains:
      - `touches` (120.0s TTL) for live multi-agent collision detection.
      - `session_edits` (86,400.0s / 24h TTL) so bare `body commit` inside a
        conversation scopes automatically to files edited in that conversation.
    """
    root = resolve_repo_root(repo_root)
    git_dir = get_git_dir(root)
    git_dir.mkdir(parents=True, exist_ok=True)

    sid = resolve_session_id(session_id)
    if isinstance(paths, (str, Path)):
        path_list = [str(paths)]
    else:
        path_list = [str(p) for p in paths]

    rel_paths = [normalize_repo_path(p, root) for p in path_list]

    lock_path = git_dir / "wbc_active_touches.lock"
    touches_path = git_dir / "wbc_active_touches.json"
    now = time.time()

    with open(lock_path, "w", encoding="utf-8") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            data: dict[str, Any] = {"touches": {}, "session_edits": {}}
            if touches_path.exists():
                try:
                    loaded = json.loads(touches_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict) and "touches" in loaded:
                        data = loaded
                except (json.JSONDecodeError, OSError):
                    data = {"touches": {}, "session_edits": {}}

            # Prune live collision entries older than 120.0s
            active_touches: dict[str, Any] = {}
            for p, entry in data.get("touches", {}).items():
                if isinstance(entry, dict) and (now - float(entry.get("ts", 0))) < 120.0:
                    active_touches[p] = entry

            # Update entries for touched files
            for p in rel_paths:
                if p in active_touches:
                    entry = active_touches[p]
                    sessions = list(entry.get("sessions", []))
                    old_sid = entry.get("session_id")
                    if old_sid and old_sid not in sessions:
                        sessions.append(old_sid)
                    if sid not in sessions:
                        sessions.append(sid)
                    active_touches[p] = {
                        "session_id": sid,
                        "ts": now,
                        "pid": os.getpid(),
                        "sessions": sessions,
                    }
                else:
                    active_touches[p] = {
                        "session_id": sid,
                        "ts": now,
                        "pid": os.getpid(),
                        "sessions": [sid],
                    }

            # Prune 24h session_edits and append edited paths for real conversation IDs
            raw_sess_edits = data.get("session_edits") if isinstance(data.get("session_edits"), dict) else {}
            session_edits: dict[str, Any] = {}
            for s_key, s_val in raw_sess_edits.items():
                if isinstance(s_val, dict) and (now - float(s_val.get("ts", 0))) < 86400.0:
                    files_clean = [str(f) for f in (s_val.get("files") or []) if str(f)]
                    if files_clean:
                        session_edits[str(s_key)] = {"ts": float(s_val.get("ts", now)), "files": files_clean}

            if record_session_edit and sid and sid not in ("default", "local", "global"):
                existing_entry = session_edits.get(sid, {"ts": now, "files": []})
                merged_files = list(existing_entry.get("files") or [])
                for rp in rel_paths:
                    if rp and not rp.startswith("..") and rp not in merged_files:
                        merged_files.append(rp)
                if merged_files:
                    session_edits[sid] = {"ts": now, "files": merged_files}

            data["touches"] = active_touches
            data["session_edits"] = session_edits
            touches_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file, fcntl.LOCK_UN)


def remove_touches(repo_root: Path, rel_paths: list[str]) -> None:
    """Remove committed files from both the 120s active touches and 24h session_edits ledgers."""
    git_dir = get_git_dir(repo_root)
    lock_path = git_dir / "wbc_active_touches.lock"
    touches_path = git_dir / "wbc_active_touches.json"
    if not touches_path.exists():
        return

    removed_set = set(rel_paths)
    with (
        contextlib.suppress(OSError, json.JSONDecodeError),
        open(lock_path, "w", encoding="utf-8") as lock_file,
    ):
        if fcntl is not None:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            loaded = json.loads(touches_path.read_text(encoding="utf-8"))
            touches = loaded.get("touches", {})
            for p in rel_paths:
                touches.pop(p, None)
            loaded["touches"] = touches

            sess_edits = loaded.get("session_edits")
            if isinstance(sess_edits, dict):
                cleaned_sess: dict[str, Any] = {}
                for s_key, s_val in sess_edits.items():
                    if not isinstance(s_val, dict):
                        continue
                    rem = [f for f in (s_val.get("files") or []) if f not in removed_set]
                    if rem:
                        cleaned_sess[s_key] = {"ts": s_val.get("ts", time.time()), "files": rem}
                loaded["session_edits"] = cleaned_sess

            touches_path.write_text(json.dumps(loaded, indent=2), encoding="utf-8")
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file, fcntl.LOCK_UN)


def resolve_smart_scope(
    explicit_paths: list[str] | None,
    session_id: str | None,
    all_tracked: bool,
    repo_root: Path,
    zone: str | None = None,
) -> tuple[list[str], str | None]:
    """Resolve file scope for commit, guarding against multi-agent collisions,
    respecting `.wbc_reflex.json` zones/blocked_paths, and auto-scoping to 24h session edits.

    Returns:
        (scoped_rel_paths, error_message)
    """
    root = resolve_repo_root(repo_root)
    git_dir = get_git_dir(root)
    sid = resolve_session_id(session_id)
    repo_cfg = _load_repo_zone_config(root)
    blocked_paths = [str(p) for p in (repo_cfg.get("blocked_paths") or []) if str(p).strip()]
    zones_map = repo_cfg.get("zones") if isinstance(repo_cfg.get("zones"), dict) else {}

    # Load active touches (within 120s) and 24h session_edits under shared lock
    lock_path = git_dir / "wbc_active_touches.lock"
    touches_path = git_dir / "wbc_active_touches.json"
    active_touches: dict[str, Any] = {}
    session_edits: dict[str, Any] = {}
    now = time.time()

    if touches_path.exists():
        with (
            contextlib.suppress(OSError, json.JSONDecodeError),
            open(lock_path, "w", encoding="utf-8") as lock_file,
        ):
            if fcntl is not None:
                fcntl.flock(lock_file, fcntl.LOCK_SH)
            try:
                loaded = json.loads(touches_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    for p, entry in (loaded.get("touches") or {}).items():
                        if isinstance(entry, dict) and (now - float(entry.get("ts", 0))) < 120.0:
                            active_touches[p] = entry
                    for s_key, s_val in (loaded.get("session_edits") or {}).items():
                        if isinstance(s_val, dict) and (now - float(s_val.get("ts", 0))) < 86400.0:
                            session_edits[str(s_key)] = s_val
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)

    if explicit_paths:
        normalized = [normalize_repo_path(p, root) for p in explicit_paths]
        for np in normalized:
            hit_blk = _is_blocked_repo_path(np, blocked_paths)
            if hit_blk:
                return [], f"Blocked private path '{np}' matches .wbc_reflex.json blocked_paths ('{hit_blk}')."
        if not all_tracked:
            for p in normalized:
                if p in active_touches:
                    entry = active_touches[p]
                    sessions = entry.get("sessions", [])
                    if len(sessions) > 1:
                        other = [s for s in sessions if s != sid]
                        return [], (
                            f"Concurrent collision detected on '{p}': "
                            f"touched by multiple active sessions {sessions} (including {other}). "
                            "Resolve conflicts or pass all_tracked=True."
                        )
        return normalized, None

    # Query modified/added/deleted tracked files and non-ignored untracked files
    res = subprocess.run(
        ["git", "status", "--porcelain", "-uall"],
        check=False,
        cwd=root,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        return [], f"git status failed: {res.stderr}"

    modified_tracked: list[str] = []
    untracked_non_ignored: list[str] = []
    for line in res.stdout.splitlines():
        if len(line) < 3:
            continue
        code = line[:2]
        path_str = line[3:].strip()
        if " -> " in path_str:
            path_str = path_str.split(" -> ", 1)[1].strip()
        if path_str.startswith('"') and path_str.endswith('"'):
            path_str = path_str[1:-1]

        if code == "!!":
            continue
        norm_p = normalize_repo_path(path_str, root)
        if _is_blocked_repo_path(norm_p, blocked_paths):
            continue
        if code == "??":
            untracked_non_ignored.append(norm_p)
        else:
            modified_tracked.append(norm_p)

    # Named `--zone <name>` filter (e.g. `body commit --zone core` or `"commit site"`)
    if zone:
        z_key = zone.strip()
        if z_key not in zones_map:
            avail = ", ".join(sorted(zones_map.keys())) if zones_map else "none (.wbc_reflex.json missing)"
            return [], f"Unknown commit zone '{z_key}'. Available zones: {avail}."
        prefixes = zones_map.get(z_key) or []
        zone_scoped = [f for f in modified_tracked if _matches_zone_prefixes(f, prefixes)]
        for zp in zone_scoped:
            if zp in active_touches:
                sessions = active_touches[zp].get("sessions", [])
                if len(sessions) > 1:
                    other = [s for s in sessions if s != sid]
                    return [], (
                        f"Concurrent collision detected on '{zp}' in zone '{z_key}': "
                        f"touched by multiple active sessions {sessions} (including {other})."
                    )
        return zone_scoped, None

    distinct_sessions = set()
    for entry in active_touches.values():
        for s in entry.get("sessions", []):
            if s:
                distinct_sessions.add(s)
        if entry.get("session_id"):
            distinct_sessions.add(entry["session_id"])

    if all_tracked:
        return modified_tracked, None

    if len(distinct_sessions) <= 1:
        # 24h per-conversation auto-scoping: if this conversation recorded edits, commit ONLY its dirty/created files
        if sid and sid not in ("default", "local", "global") and sid in session_edits:
            sess_file_set = set(session_edits[sid].get("files") or [])
            sess_scoped = [
                f for f in (modified_tracked + untracked_non_ignored) if f in sess_file_set
            ]
            if sess_scoped:
                return sess_scoped, None
        return modified_tracked, None

    # Multi-agent active (>1 distinct active session_id in last 120s):
    # Check if any file was touched by BOTH session_id and another session
    for p, entry in active_touches.items():
        sessions = entry.get("sessions", [])
        if sid in sessions and len(sessions) > 1:
            other = [s for s in sessions if s != sid]
            return [], (
                f"Same-file collision detected on '{p}': "
                f"touched by both current session '{sid}' and {other}. "
                "Commit with explicit paths or resolve conflicts."
            )

    session_touched_files = {
        p
        for p, entry in active_touches.items()
        if sid in entry.get("sessions", []) or entry.get("session_id") == sid
    }
    scoped = [f for f in modified_tracked if f in session_touched_files]
    if not scoped:
        return [], (
            f"No modified files associated with session '{sid}' in active touches ledger. "
            "Specify explicit file paths to commit."
        )

    return scoped, None


def snapshot_file_states(
    repo_root: Path,
    rel_paths: list[str],
) -> dict[str, tuple[int, int, str] | str]:
    """Snapshot file states (mtime, size, sha256) for torn-write CAS checking.

    If any target file was modified < 50ms ago, briefly sleeps to allow flushing.
    """
    now = time.time()
    for rel_path in rel_paths:
        full_path = repo_root / rel_path
        if full_path.exists():
            with contextlib.suppress(OSError):
                st = full_path.stat()
                if now - st.st_mtime < 0.05:
                    for _ in range(3):
                        time.sleep(0.05)
                        st = full_path.stat()
                        if time.time() - st.st_mtime >= 0.05:
                            break

    states: dict[str, tuple[int, int, str] | str] = {}
    for rel_path in rel_paths:
        full_path = repo_root / rel_path
        if not full_path.exists():
            states[rel_path] = "DELETED"
        else:
            try:
                st = full_path.stat()
                content = full_path.read_bytes()
                sha = hashlib.sha256(content).hexdigest()
                states[rel_path] = (st.st_mtime_ns, st.st_size, sha)
            except OSError:
                states[rel_path] = "DELETED"
    return states


def verify_no_torn_writes(
    repo_root: Path,
    pre_states: dict[str, tuple[int, int, str] | str],
    check_fn: Callable[[], None] | None = None,
) -> bool:
    """Verify that no monitored files changed state since pre_states was taken."""
    if check_fn is not None:
        try:
            check_fn()
        except (OSError, RuntimeError, ValueError):
            return False

    for rel_path, expected in pre_states.items():
        full_path = repo_root / rel_path
        if not full_path.exists():
            curr: tuple[int, int, str] | str = "DELETED"
        else:
            try:
                st = full_path.stat()
                content = full_path.read_bytes()
                sha = hashlib.sha256(content).hexdigest()
                curr = (st.st_mtime_ns, st.st_size, sha)
            except OSError:
                curr = "DELETED"
        if curr != expected:
            return False
    return True


def check_secrets(
    repo_root: Path,
    scoped_rel_paths: list[str],
    staged_diff: str,
) -> tuple[bool, str | None]:
    """Scan staged diff and filenames for secrets. Fail-closed gate."""
    for p in scoped_rel_paths:
        name = Path(p).name
        if name in FORBIDDEN_SECRET_BASENAMES or name == ".env" or name.startswith(".env."):
            return False, f"Forbidden secret file staged: '{p}'"

    for line in staged_diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added_content = line[1:]
            match = _FAST_SECRET_RE.search(added_content)
            if match:
                token_preview = match.group(0)[:8]
                return False, f"Secret pattern matched in staged diff: '{token_preview}...'"

    return True, None


def check_ast_syntax(
    repo_root: Path,
    scoped_rel_paths: list[str],
) -> tuple[bool, str | None]:
    """Fast on-device Python AST syntax verification (<5ms)."""
    for p in scoped_rel_paths:
        if p.endswith(".py"):
            full_path = repo_root / p
            if full_path.exists():
                try:
                    source = full_path.read_text(encoding="utf-8")
                    ast.parse(source)
                except SyntaxError as e:
                    return False, f"AST SyntaxError in {p}: {e}"
                except OSError as e:
                    return False, f"AST inspection error in {p}: {e}"
    return True, None


def check_ruff(
    repo_root: Path,
    scoped_rel_paths: list[str],
) -> tuple[bool, str | None]:
    """Staged-only ruff linter check (<5s timeout)."""
    py_files = [p for p in scoped_rel_paths if p.endswith(".py") and (repo_root / p).exists()]
    if not py_files:
        return True, None

    try:
        res = subprocess.run(
            ["ruff", "check", "--select", "E,F", "--ignore", "BLE001,S110,E501,E741,E402,F841,F401,F541", "--quiet", "--"] + py_files,
            check=False,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if res.returncode != 0:
            output = res.stdout or res.stderr
            return False, f"Ruff lint failed:\n{output.strip()}"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    return True, None


def _is_port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.015) -> bool:
    """Check TCP port with sub-15ms timeout."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def synthesize_commit_message(
    scoped_rel_paths: list[str],
    staged_diff: str,
    repo_root: Path,
    no_llm: bool = False,
) -> str:
    """Synthesize Conventional Commit message using Cold AST Delta or Warm Local Metal."""
    # 1. Cold Deterministic AST Delta (<15ms)
    symbols: list[str] = []
    for line in staged_diff.splitlines():
        line_s = line.strip()
        if line_s.startswith(("+def ", "def ")):
            after_def = line_s.split("def ", 1)[1]
            name = re.split(r"[\(: ]", after_def)[0].strip()
            if name and name not in symbols:
                symbols.append(name)
        elif line_s.startswith(("+class ", "class ")):
            after_cls = line_s.split("class ", 1)[1]
            name = re.split(r"[\(: ]", after_cls)[0].strip()
            if name and name not in symbols:
                symbols.append(name)
        elif line.startswith("@@"):
            m = re.search(r"@@.*@@\s*(?:def|class)\s+([a-zA-Z0-9_]+)", line)
            if m:
                name = m.group(1)
                if name and name not in symbols:
                    symbols.append(name)

    if any(p.startswith("tests/") or "test" in Path(p).name for p in scoped_rel_paths):
        prefix = "test"
    elif len(scoped_rel_paths) == 1:
        prefix = "feat"
    else:
        prefix = "chore"

    if len(scoped_rel_paths) == 1:
        p0 = Path(scoped_rel_paths[0])
        scope = p0.parent.name if p0.parent.name not in (".", "") else p0.stem
    else:
        parents = {Path(p).parent.name for p in scoped_rel_paths if Path(p).parent.name not in (".", "")}
        scope = next(iter(parents)) if len(parents) == 1 else "core"

    if symbols:
        primary_file = Path(scoped_rel_paths[0]).name if scoped_rel_paths else "files"
        sym_str = ", ".join(symbols[:3])
        cold_msg = f"{prefix}({scope}): update {sym_str} in {primary_file}"
    else:
        file_names = ", ".join(Path(p).name for p in scoped_rel_paths[:3])
        cold_msg = f"{prefix}({scope}): update {file_names}"

    if no_llm or os.environ.get("WBC_DISABLE_LOCAL_LLM"):
        return cold_msg

    # 2. Warm Local Metal (:9379 body-coder-mlx or :9380 gemma4-26b / localbench-specialist-router)
    model_9380 = (
        "gemma4-26b"
        if is_external_models_drive_mounted()
        else "localbench-specialist-router"
    )
    for port, model_name in ((9379, "body-coder-mlx"), (9380, model_9380)):
        if _is_port_open(port):
            t0 = time.perf_counter()
            first_line = ""
            is_valid_cc = False
            infra_skipped = False
            try:
                payload = json.dumps({
                    "model": model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a git commit generator. Output ONLY a single-line "
                                "Conventional Commit message (feat|fix|refactor|chore|test(scope): summary)."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Files: {scoped_rel_paths}\nDiff:\n{staged_diff[:1200]}",
                        },
                    ],
                    "temperature": 0.1,
                    "max_tokens": 48,
                }).encode("utf-8")

                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=0.6) as resp:
                    if resp.status == 200:
                        res_json = json.loads(resp.read().decode("utf-8"))
                        raw_text = str(res_json["choices"][0]["message"]["content"])
                        text = strip_llm_fences_and_thoughts(raw_text)
                        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                        first_line = lines[0].strip('"').strip("'") if lines else ""
                        is_valid_cc = bool(
                            re.match(
                                r"^(feat|fix|refactor|chore|test|docs|perf)\(?[a-zA-Z0-9_\-]*\)?:\s*.+",
                                first_line,
                            )
                        )
                    elif is_infra_error(status_code=resp.status):
                        infra_skipped = True
            except (
                OSError,
                json.JSONDecodeError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ) as exc:
                if is_infra_error(exc=exc):
                    infra_skipped = True
            if os.environ.get("WBC_ROUTER_SHADOW") == "1":
                record_shadow_eval(
                    task_type="fast_commit",
                    engine=f"metal_{port}",
                    model=model_name,
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                    valid_guardrail=is_valid_cc,
                    baseline_used="live_fallback",
                    sla_ms=600.0,
                    tokens_saved_est=80,
                    infra_skip=infra_skipped,
                )
            if is_valid_cc:
                return first_line
            if not infra_skipped:
                break

    return cold_msg


def sync_main_index(repo_root: Path, scoped_rel_paths: list[str]) -> bool:
    """Best-effort synchronize main .git/index for scoped paths under flock."""
    git_dir = get_git_dir(repo_root)
    sync_lock_path = git_dir / "wbc_index_sync.lock"

    deadline = time.time() + 5.0
    fd = None
    try:
        fd = os.open(str(sync_lock_path), os.O_CREAT | os.O_RDWR)
        acquired = False
        while time.time() < deadline:
            if fcntl is None:
                acquired = True
                break
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except (BlockingIOError, OSError):
                time.sleep(0.01)
        if not acquired:
            return False

        for _ in range(15):
            res = subprocess.run(
                ["git", "reset", "-q", "HEAD", "--"] + scoped_rel_paths,
                check=False,
                cwd=repo_root,
                capture_output=True,
            )
            if res.returncode == 0:
                return True
            time.sleep(0.02)
        return False
    except OSError:
        return False
    finally:
        if fd is not None:
            if fcntl is not None:
                with contextlib.suppress(OSError):
                    fcntl.flock(fd, fcntl.LOCK_UN)
            with contextlib.suppress(OSError):
                os.close(fd)


def record_commit_to_dot_memory(
    repo_root: Path,
    commit_sha: str,
    message: str,
    files: list[str],
    session_id: str | None = None,
) -> None:
    """Best-effort persistence of commit event into local DotMemory."""
    with contextlib.suppress(ImportError, OSError, RuntimeError, ValueError):
        from whole_body_compute.local.dot_memory import record_episodic_anchor

        record_episodic_anchor(
            tag="git_commit",
            summary=f"Commit {commit_sha[:8]}: {message}",
            metadata={"commit_sha": commit_sha, "files": files, "session_id": session_id},
        )


def run_fast_commit(
    paths: list[str] | str | None = None,
    message: str | None = None,
    session_id: str | None = None,
    all_tracked: bool = False,
    wip: bool = False,
    no_lint: bool = False,
    no_llm: bool = False,
    zone: str | None = None,
    repo_root: Path | None = None,
    _mid_flight_hook: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Execute race-free atomic git commit using ephemeral shadow index & update-ref CAS."""
    root = resolve_repo_root(repo_root)
    sid = resolve_session_id(session_id)

    if isinstance(paths, (str, Path)):
        explicit_paths = [str(paths)]
    elif paths is not None:
        explicit_paths = [str(p) for p in paths]
    else:
        explicit_paths = None

    scoped_rel_paths, err = resolve_smart_scope(explicit_paths, sid, all_tracked, root, zone=zone)
    if err is not None:
        return {"status": "error", "error": err}

    if not scoped_rel_paths:
        clean_msg = f"Nothing to commit in zone '{zone}'" if zone else "Nothing to commit"
        return {
            "status": "clean",
            "message": clean_msg,
            "files": [],
            "receipt": f"⚪ {clean_msg}.",
        }

    # Record active 120s collision touch without overwriting 24h session_edits scope
    record_file_touch(scoped_rel_paths, session_id=sid, repo_root=root, record_session_edit=False)

    # Pre-staging snapshot
    pre_states = snapshot_file_states(root, scoped_rel_paths)

    max_cas_retries = 12
    cached_commit_msg = message
    commit_sha: str | None = None
    branch_ref: str = "HEAD"
    success = False

    # Ensure author/committer env vars fallback
    author_env = {
        "GIT_AUTHOR_NAME": os.environ.get("GIT_AUTHOR_NAME", "Whole Body Compute"),
        "GIT_AUTHOR_EMAIL": os.environ.get("GIT_AUTHOR_EMAIL", "wbc@local.internal"),
        "GIT_COMMITTER_NAME": os.environ.get("GIT_COMMITTER_NAME", "Whole Body Compute"),
        "GIT_COMMITTER_EMAIL": os.environ.get("GIT_COMMITTER_EMAIL", "wbc@local.internal"),
    }

    for attempt in range(max_cas_retries):
        # 1. Read current HEAD commit SHA and symbolic ref
        res_head = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            check=False,
            cwd=root,
            capture_output=True,
            text=True,
        )
        old_head = res_head.stdout.strip() if res_head.returncode == 0 else None

        res_sym = subprocess.run(
            ["git", "symbolic-ref", "-q", "HEAD"],
            check=False,
            cwd=root,
            capture_output=True,
            text=True,
        )
        branch_ref = res_sym.stdout.strip() if res_sym.returncode == 0 and res_sym.stdout.strip() else "HEAD"

        # 2. Ephemeral shadow index
        shadow_idx_path = Path(tempfile.gettempdir()) / f"wbc_shadow_idx_{os.getpid()}_{uuid.uuid4().hex}.idx"
        env = {**os.environ, **author_env, "GIT_INDEX_FILE": str(shadow_idx_path)}

        try:
            if old_head:
                res_rt = subprocess.run(
                    ["git", "read-tree", old_head],
                    check=False,
                    cwd=root,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                if res_rt.returncode != 0:
                    time.sleep(random.uniform(0.015, 0.060))
                    continue

            # Update only scoped_rel_paths in shadow index (split existing vs deleted so
            # deleted files and .gitignore'd tracked paths like .agents/* never collide)
            existing_rel = [p for p in scoped_rel_paths if (root / p).exists() or (root / p).is_symlink()]
            deleted_rel = [p for p in scoped_rel_paths if p not in existing_rel]

            if deleted_rel:
                res_rm = subprocess.run(
                    ["git", "rm", "--cached", "--ignore-unmatch", "--quiet", "--"] + deleted_rel,
                    check=False,
                    cwd=root,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                if res_rm.returncode != 0:
                    return {"status": "error", "error": f"Failed to stage deletions in shadow index: {res_rm.stderr}"}

            if existing_rel:
                res_add = subprocess.run(
                    ["git", "add", "-f", "-A", "--"] + existing_rel,
                    check=False,
                    cwd=root,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                if res_add.returncode != 0:
                    return {"status": "error", "error": f"Failed to stage files in shadow index: {res_add.stderr}"}

            # 3. Extract staged diff
            res_diff = subprocess.run(
                ["git", "diff", "--cached", "-U0"],
                check=False,
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
            )
            staged_diff = res_diff.stdout
            if not staged_diff.strip():
                return {"status": "clean", "message": "Nothing to commit", "files": []}

            # 4. Attempt 0: Security & Quality Gates
            if attempt == 0:
                # Fail-Closed Secret Gate (NEVER skippable)
                sec_ok, sec_err = check_secrets(root, scoped_rel_paths, staged_diff)
                if not sec_ok:
                    return {"status": "error", "error": sec_err}

                # AST Syntax Gate
                if not wip:
                    ast_ok, ast_err = check_ast_syntax(root, scoped_rel_paths)
                    if not ast_ok:
                        return {"status": "error", "error": ast_err}

                # Staged-Only Ruff Gate
                if not no_lint and not wip:
                    ruff_ok, ruff_err = check_ruff(root, scoped_rel_paths)
                    if not ruff_ok:
                        return {"status": "error", "error": ruff_err}

                # Message synthesis
                if not cached_commit_msg:
                    if wip:
                        cached_commit_msg = f"wip(checkpoint): snapshot {len(scoped_rel_paths)} file(s)"
                    else:
                        cached_commit_msg = synthesize_commit_message(
                            scoped_rel_paths,
                            staged_diff,
                            root,
                            no_llm=no_llm,
                        )

            # 5. Pre/Post Torn-Write CAS
            if _mid_flight_hook is not None:
                _mid_flight_hook()

            if not verify_no_torn_writes(root, pre_states):
                return {
                    "status": "error",
                    "error": "Torn write detected: file modified mid-commit",
                }

            # 6. Write tree from shadow index
            res_wt = subprocess.run(
                ["git", "write-tree"],
                check=False,
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
            )
            if res_wt.returncode != 0:
                time.sleep(random.uniform(0.015, 0.060))
                continue
            tree_sha = res_wt.stdout.strip()

            # 7. Create commit object
            commit_cmd = ["git", "commit-tree", tree_sha]
            if old_head:
                commit_cmd += ["-p", old_head]
            commit_cmd += ["-m", cached_commit_msg]
            res_ct = subprocess.run(
                commit_cmd,
                check=False,
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
            )
            if res_ct.returncode != 0:
                time.sleep(random.uniform(0.015, 0.060))
                continue
            commit_sha = res_ct.stdout.strip()

            # 8. Atomic Compare-And-Swap on branch ref
            if old_head:
                update_cmd = ["git", "update-ref", branch_ref, commit_sha, old_head]
            else:
                update_cmd = ["git", "update-ref", branch_ref, commit_sha, ""]
            res_ur = subprocess.run(update_cmd, check=False, cwd=root, capture_output=True, text=True)

            if res_ur.returncode == 0:
                success = True
                break
            else:
                # CAS contention: retry with fresh HEAD
                time.sleep(random.uniform(0.015, 0.060))
                continue
        finally:
            if shadow_idx_path.exists():
                with contextlib.suppress(OSError):
                    shadow_idx_path.unlink()

    if not success or commit_sha is None:
        return {
            "status": "error",
            "error": "Exceeded max CAS retries due to concurrent commit contention",
        }

    # Best-effort sync main index for scoped paths
    sync_main_index(root, scoped_rel_paths)

    # Remove committed touches
    remove_touches(root, scoped_rel_paths)

    # Best-effort DotMemory persistence
    record_commit_to_dot_memory(root, commit_sha, cached_commit_msg, scoped_rel_paths, sid)

    short_branch = branch_ref.replace("refs/heads/", "")
    head_msg = (cached_commit_msg or "").splitlines()[0] if cached_commit_msg else "commit"
    preview_files = ", ".join(scoped_rel_paths[:5])
    if len(scoped_rel_paths) > 5:
        preview_files += f", +{len(scoped_rel_paths) - 5} more"
    receipt_line = f"✅ {commit_sha[:7]} ({short_branch}) - {head_msg} [{len(scoped_rel_paths)} file(s): {preview_files}]"

    return {
        "status": "ok",
        "commit_sha": commit_sha,
        "branch": branch_ref,
        "message": cached_commit_msg,
        "files": scoped_rel_paths,
        "receipt": receipt_line,
    }


def run_checkpoint(
    label: str | None = None,
    commit_mode: bool = False,
    list_mode: bool = False,
    restore_ref: str | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Capture or restore ultra-fast, non-destructive git plumbing checkpoints (<15ms)."""
    root = resolve_repo_root(repo_root)

    if list_mode:
        res = subprocess.run(
            [
                "git",
                "for-each-ref",
                "--sort=-creatordate",
                "--format=%(refname)|%(objectname:short)|%(subject)",
                "refs/checkpoints/",
            ],
            check=False,
            cwd=root,
            capture_output=True,
            text=True,
        )
        checkpoints = []
        for line in res.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", 2)
            if len(parts) == 3:
                checkpoints.append({
                    "ref": parts[0],
                    "short_ref": parts[0].replace("refs/checkpoints/", ""),
                    "sha": parts[1],
                    "subject": parts[2],
                })
        return {"status": "ok", "checkpoints": checkpoints}

    if restore_ref:
        if restore_ref.startswith("-"):
            return {
                "status": "error",
                "error": f"Invalid checkpoint reference {restore_ref!r}: must not start with '-'",
            }
        res = subprocess.run(
            ["git", "checkout", restore_ref, "--", "."],
            check=False,
            cwd=root,
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            return {
                "status": "error",
                "error": f"Failed to restore checkpoint {restore_ref}: {res.stderr}",
            }
        return {"status": "ok", "restored": restore_ref}

    if commit_mode:
        return run_fast_commit(
            all_tracked=True,
            wip=True,
            message=f"wip(checkpoint): {label or 'snapshot'}",
            repo_root=root,
        )

    # Default: non-destructive plumbing checkpoint (<15ms, zero mutation of .git/index or worktree)
    res_head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        check=False,
        cwd=root,
        capture_output=True,
        text=True,
    )
    old_head = res_head.stdout.strip() if res_head.returncode == 0 else None

    shadow_idx = Path(tempfile.gettempdir()) / f"wbc_ckpt_idx_{os.getpid()}_{uuid.uuid4().hex}.idx"
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": os.environ.get("GIT_AUTHOR_NAME", "Whole Body Compute"),
        "GIT_AUTHOR_EMAIL": os.environ.get("GIT_AUTHOR_EMAIL", "wbc@local.internal"),
        "GIT_COMMITTER_NAME": os.environ.get("GIT_COMMITTER_NAME", "Whole Body Compute"),
        "GIT_COMMITTER_EMAIL": os.environ.get("GIT_COMMITTER_EMAIL", "wbc@local.internal"),
        "GIT_INDEX_FILE": str(shadow_idx),
    }

    try:
        if old_head:
            res_rt = subprocess.run(
                ["git", "read-tree", old_head],
                check=False,
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
            )
            if res_rt.returncode != 0:
                return {"status": "error", "error": f"read-tree failed: {res_rt.stderr}"}

        # Capture all modified tracked files in shadow index
        res_add = subprocess.run(
            ["git", "add", "-u"],
            check=False,
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
        )
        if res_add.returncode != 0:
            return {"status": "error", "error": f"git add -u failed: {res_add.stderr}"}

        res_wt = subprocess.run(
            ["git", "write-tree"],
            check=False,
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
        )
        if res_wt.returncode != 0:
            return {"status": "error", "error": f"write-tree failed: {res_wt.stderr}"}
        tree_sha = res_wt.stdout.strip()

        ts_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_label = re.sub(r"[^a-zA-Z0-9_\-]", "-", label or "snapshot")
        ref_name = f"refs/checkpoints/{ts_str}-{safe_label}"
        commit_msg = f"checkpoint: {label or 'snapshot'}"

        commit_cmd = ["git", "commit-tree", tree_sha]
        if old_head:
            commit_cmd += ["-p", old_head]
        commit_cmd += ["-m", commit_msg]
        res_ct = subprocess.run(commit_cmd, check=False, cwd=root, env=env, capture_output=True, text=True)
        if res_ct.returncode != 0:
            return {"status": "error", "error": f"commit-tree failed: {res_ct.stderr}"}
        commit_sha = res_ct.stdout.strip()

        res_ur = subprocess.run(
            ["git", "update-ref", ref_name, commit_sha],
            check=False,
            cwd=root,
            capture_output=True,
            text=True,
        )
        if res_ur.returncode != 0:
            return {"status": "error", "error": f"update-ref failed: {res_ur.stderr}"}

        return {
            "status": "ok",
            "ref": ref_name,
            "commit_sha": commit_sha,
            "label": label or "snapshot",
            "receipt": f"📌 Saved checkpoint {ref_name} ({commit_sha[:7]})",
        }
    finally:
        if shadow_idx.exists():
            with contextlib.suppress(OSError):
                shadow_idx.unlink()
