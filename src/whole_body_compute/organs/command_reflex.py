"""
Whole Body Compute - On-Demand Command Reflex Engine (`PreInvocation` + `PreToolUse`).

Provides a configurable 2-layer reflex system (`WBC_COMMAND_REFLEX_MODE=hybrid|guided|off`
or `~/.whole_body_compute/reflex_config.json`) with SQLite WAL telemetry in
`shield_savings_ledger` (`event_type='command_reflex'`):

1. Layer 1 (`PreInvocation` / `UserPromptSubmit`):
   - `spinal_short_circuit` (active in `hybrid` mode on strict standalone imperatives
     like `"commit"`, `"commit this"`, `"wip"`, `"checkpoint"`): executes `run_fast_commit`
     (`no_llm=True`) or `run_checkpoint` in-process in ~50-250ms and returns
     `terminationBehavior: "terminate"` + inline `metadata.user_markdown` (`0` cloud tokens).
     If a lint, syntax, secret, or multi-agent collision error occurs, automatically
     **fails through** (`spinal_failthrough`) to the LLM with the exact error context so
     the agent self-heals on Turn 1.
   - `preinv_directive` (active in `hybrid` and `guided` modes): injects a 1-line
     `systemMessage` contract for nuanced `body commit`, `body checkpoint`, `body flush --auto`,
     `body eyes`/`body hands`, `body memory`, and `body doctor`/`savings`/`context`/`scorecard`
     requests, paying `0` tokens on non-matching turns.

2. Layer 2 (`PreToolUse` Shell Auto-Rewrite):
   - `pretool_git_rewrite`: intercepts raw `git commit` / `git add ... && git commit`
     commands emitted mid-task by any coding agent and rewrites `overwrite` / `updatedInput`
     to `body commit` (preserving `-m`, `-a`, explicit paths, and trailing `&&` stages),
     guaranteeing shadow-index concurrency safety, secret/AST/ruff gates, and `dot_memory.db`
     persistence while leaving `--amend`, `--allow-empty`, rebase/merge, and `/tmp` sandboxes untouched.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any

VALID_REFLEX_MODES = ("hybrid", "guided", "off")

_USER_REQUEST_RE = re.compile(r"<USER_REQUEST>\s*(.*?)\s*</USER_REQUEST>", re.DOTALL | re.IGNORECASE)

# Discussion / question / analytical prefixes that must NEVER trigger imperative reflexes
_LEADING_QUOTE_PREFIX_RE = re.compile(r"^(?:@\[[^\]]+\]\s*)+")

_DISCUSSION_PREFIX_RE = re.compile(
    r"^(?:why|how|what|when|where|which|who|should|consider|would|could|is|are|was|were|"
    r"do\s+we|does|did|explain|describe|tell\s+me\s+(?:why|how|about)|compare|think\s+about|"
    r"let'?s\s+(?:discuss|consider|think)|anything|anyghint|any\s+(?:open|other|pending))\b",
    re.IGNORECASE,
)

# Prose false-positive phrases containing "commit" that are NOT git commit requests
_COMMIT_PROSE_EXCLUSION_RE = re.compile(
    r"\b(?:commit(?:ted|ting|s)?\s+to\b|commit\s+(?:crime|fraud|suicide|memory|time|resources)|"
    r"commit\s+(?:history|log|hash|sha|id|author|date|graph|count|stats)|"
    r"(?:last|previous|prior|recent|initial|first|parent|merge|revert|cherry-pick)\s+commits?\b|"
    r"pre-commit|post-commit|reflex\s+for\s+commit|git\s+(?:log|show|rev-parse|blame|bisect))\b",
    re.IGNORECASE,
)

# Prose/code false-positive phrases containing "flush" that are NOT context handoff requests
_FLUSH_PROSE_EXCLUSION_RE = re.compile(
    r"\b(?:fflush|sys\.stdout\.flush|stdout|stderr|buffer|socket|pipe|stream|dns|arp|iptables|"
    r"redis|memcached|postgres|sqlite|wal_checkpoint|cache\s+flush|flush\s+(?:cache|dns|buffer|stdout|stderr|logs?|disk|queue| toilet))\b",
    re.IGNORECASE,
)

# Strict full-utterance standalone imperatives eligible for 0-token Spinal Short-Circuit in `hybrid` mode
_STANDALONE_COMMIT_RE = re.compile(
    r"^(?:please\s+)?(?:(?:uv\s+run\s+)?(?:body|wbc)\s+commit|git\s+commit|commit)"
    r"(?:\s+(?:this|all|it|changes|our\s+changes|the\s+changes|everything|now))?[.!]*$",
    re.IGNORECASE,
)

_STANDALONE_ZONE_COMMIT_RE = re.compile(
    r"^(?:please\s+)?(?:(?:uv\s+run\s+)?(?:body|wbc)\s+commit\s+(?:(?:--zone|-z)\s+)?|git\s+commit\s+|commit\s+)"
    r"([A-Za-z][A-Za-z0-9_-]{1,23})[.!]*$",
    re.IGNORECASE,
)

_STANDALONE_WIP_RE = re.compile(
    r"^(?:please\s+)?(?:(?:uv\s+run\s+)?(?:body|wbc)\s+commit\s+--wip|wip\s+commit|commit\s+wip|save\s+wip|wip)[.!]*$",
    re.IGNORECASE,
)

_STANDALONE_CHECKPOINT_RE = re.compile(
    r"^(?:please\s+)?(?:(?:uv\s+run\s+)?(?:body|wbc)\s+checkpoint|git\s+checkpoint|(?:save|create|make)\s+(?:a\s+)?checkpoint|checkpoint)"
    r"(?:\s+([A-Za-z0-9][A-Za-z0-9_-]{0,31}))?[.!]*$",
    re.IGNORECASE,
)

# Nuanced / multi-clause intent patterns for Mode B (`preinv_directive`)
_NUANCED_COMMIT_RE = re.compile(
    r"(?:^\s*(?:please\s+|go\s+ahead\s+and\s+|can\s+you\s+)?(?:(?:uv\s+run\s+)?(?:body|wbc)\s+commit|git\s+commit|commit)\b|"
    r"\b(?:body\s+commit|commit\s+(?:these|the|our|my|staged|modified|all)\s+(?:changes|files|fixes|edits|work)|"
    r"stage\s+and\s+commit|commit\s+and\s+push)\b)",
    re.IGNORECASE,
)

_NUANCED_CHECKPOINT_RE = re.compile(
    r"\b(?:body\s+checkpoint|(?:save|create|take|make)\s+(?:a\s+)?(?:git\s+)?checkpoint)\b",
    re.IGNORECASE,
)

_FLUSH_INTENT_RE = re.compile(
    r"(?:^\s*(?:(?:yes|yep|yeah|ok|okay|sure)[,!\s]+)?(?:please\s+|let'?s\s+|go\s+ahead\s+and\s+|time\s+to\s+)?"
    r"(?:(?:uv\s+run\s+)?(?:body|wbc)\s+)?(?:flush(?:\s+(?:it|now|this))?(?!\s+out\b)|(?:do\s+(?:the|a)\s+)?handoff|hand\s+off(?:\s+now)?)[.!]*$|"
    r"\b(?:body\s+flush|wbc\s+flush|flush\s+(?:the\s+)?(?:context|thread|convo|conversation|session)|"
    r"flush\s+it(?!\s+out\b)|(?:let'?s|go\s+ahead\s+and|time\s+to)\s+flush(?!\s+(?:it\s+)?out\b)|"
    r"(?:yes|ok|okay)[,\s]+flush(?!\s+(?:it\s+)?out\b)|glymphatic\s+flush|"
    r"(?:do\s+(?:the|a)\s+|run\s+)handoff|hand\s+off\s+now|"
    r"(?:do\s+a?\s*|run\s+)?handoff\s+to\s+(?:a\s+)?(?:new|fresh)\s+(?:thread|convo|conversation)|"
    r"start\s+(?:a\s+)?(?:new|fresh)\s+(?:convo|conversation|thread)\s+(?:for|with|and\s+handoff))\b)",
    re.IGNORECASE,
)

_EYES_HANDS_INTENT_RE = re.compile(
    r"\b(?:look\s+at\s+my\s+screen|what'?s\s+on\s+my\s+screen|inspect\s+my\s+screen|"
    r"logged-in\s+chrome|in\s+my\s+chrome(?:\s+tab|\s+window)?|open\s+in\s+my\s+chrome|"
    r"body\s+eyes|body\s+hands|eyes\s+and\s+hands|hands\s+and\s+eyes)\b",
    re.IGNORECASE,
)

_MEMORY_INTENT_RE = re.compile(
    r"(?:^\s*(?:please\s+)?remember\s+that\b|"
    r"\b(?:body\s+memory|(?:save|store)\s+(?:this\s+)?(?:in|to)\s+(?:dot_?memory|hippocampus|wbc\s+memory)|"
    r"have\s+we\s+seen\s+this\s+(?:bug|error|traceback|crash)\s+before)\b)",
    re.IGNORECASE,
)

_DIAG_INTENT_RE = re.compile(
    r"^(?:please\s+|can\s+you\s+)?(?:run\s+|show\s+|check\s+)?"
    r"(?:(?:uv\s+run\s+)?(?:body|wbc)\s+(?:doctor|savings|scorecard|context|csi|leaderboard)"
    r"(?:\s+(?:--?[A-Za-z0-9][A-Za-z0-9_-]*(?:=[^\s]+)?|json|all|full|top|live|strict))*|"
    r"(?:token|shield)\s+savings|context\s+saturation)\s*[.!]*$",
    re.IGNORECASE,
)

_USER_NAMED_BODY_RE = re.compile(
    r"\b(?:body|wbc)\s+(?:commit|checkpoint|flush|eyes|hands|memory|doctor|savings|scorecard|context|csi|leaderboard)\b",
    re.IGNORECASE,
)


def _get_wbc_data_dir() -> Path:
    env_dir = os.environ.get("WBC_DATA_DIR", "").strip()
    if env_dir:
        return Path(os.path.expanduser(env_dir))
    home = Path.home()
    wbc_dir = home / ".whole_body_compute"
    if wbc_dir.exists() or not (home / ".voicefi").exists():
        return wbc_dir
    return home / ".voicefi"


def resolve_command_reflex_mode() -> str:
    """
    Resolve active command reflex mode:
      - `'hybrid'` (default): Spinal Short-Circuit (`0` LLM tokens) on standalone commit/wip/checkpoint
        with fail-through on error, plus 1-line `PreInvocation` directives & `PreToolUse` git rewrite.
      - `'guided'`: Never terminates early; emits 1-line `PreInvocation` directives + `PreToolUse` git rewrite.
      - `'off'`: Disables command reflexes.
    """
    env_mode = os.environ.get("WBC_COMMAND_REFLEX_MODE", "").strip().lower()
    if env_mode in VALID_REFLEX_MODES:
        return env_mode

    cfg_path = _get_wbc_data_dir() / "reflex_config.json"
    if cfg_path.is_file():
        with contextlib.suppress(OSError, json.JSONDecodeError, TypeError, ValueError):
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                cfg_mode = str(data.get("mode") or data.get("command_reflex_mode") or "").strip().lower()
                if cfg_mode in VALID_REFLEX_MODES:
                    return cfg_mode
    return "hybrid"


def extract_clean_user_prompt(
    payload: dict[str, Any] | None,
    transcript_path_str: str = "",
) -> tuple[str, bool]:
    """
    Extract `(clean_user_text, is_fresh_user_turn)`.
    Ensures `PreInvocation` only fires on the initial `USER_INPUT` step of a turn,
    never re-firing on subsequent tool-result model invocations within the same turn.
    """
    payload = payload if isinstance(payload, dict) else {}
    raw_input = str(
        payload.get("lastUserInput")
        or payload.get("prompt")
        or payload.get("user_prompt")
        or payload.get("composer_prompt")
        or ""
    ).strip()
    inv_num = payload.get("invocationNum")
    inv_int: int | None = None
    if inv_num is not None:
        with contextlib.suppress(ValueError, TypeError):
            inv_int = int(inv_num)

    # In Antigravity protojson, invocationNum is 0-indexed (omitted or 0 on the initial user turn,
    # and >= 1 on subsequent mid-turn tool-result model invocations).
    if inv_int is not None and inv_int >= 1:
        return ("", False)
    if raw_input and (inv_int == 0 or ("lastUserInput" in payload and inv_int is None)):
        m = _USER_REQUEST_RE.search(raw_input)
        clean = m.group(1).strip() if m else raw_input
        return (clean, bool(clean))

    tp_str = (
        transcript_path_str
        or str(payload.get("transcriptPath") or payload.get("transcript_path") or "").strip()
    )

    if tp_str:
        tp = Path(tp_str)
        if tp.is_file():
            try:
                size = tp.stat().st_size
                with tp.open("rb") as fh:
                    if size > 12288:
                        fh.seek(size - 12288)
                    tail_text = fh.read().decode("utf-8", errors="replace")
                last_obj = None
                for raw_ln in reversed(tail_text.splitlines()):
                    raw_ln = raw_ln.strip()
                    if not raw_ln:
                        continue
                    try:
                        cand = json.loads(raw_ln)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(cand, dict):
                        last_obj = cand
                        break
                if not last_obj or last_obj.get("type") != "USER_INPUT":
                    return ("", False)
                raw_content = str(last_obj.get("content") or "")
                m = _USER_REQUEST_RE.search(raw_content)
                text = m.group(1).strip() if m else raw_content.strip()
                return (text, bool(text))
            except OSError:
                return ("", False)

    # Fallback when no transcript file exists yet (e.g. Claude Code UserPromptSubmit or unit tests)
    raw_input = str(
        payload.get("lastUserInput")
        or payload.get("prompt")
        or payload.get("user_prompt")
        or payload.get("composer_prompt")
        or ""
    ).strip()
    if not raw_input:
        return ("", False)
    # If invocationNum >= 1 without a transcript check, avoid re-firing mid-turn
    inv_num = payload.get("invocationNum")
    if inv_num is not None:
        try:
            if int(inv_num) >= 1:
                return ("", False)
        except (ValueError, TypeError):
            pass
    m = _USER_REQUEST_RE.search(raw_input)
    clean = m.group(1).strip() if m else raw_input
    return (clean, bool(clean))


def _resolve_live_context_tokens(
    transcript_path_str: str,
    transcript_kb: float,
) -> tuple[int, float]:
    """
    Resolve `(active_ctx_tokens, cache_ratio)` from the latest `PLANNER_RESPONSE` in `transcript.jsonl`
    (reflecting exact post-`/compact` active context and real prompt-cache split), falling back to
    `transcript_kb * 256` with an empirical 0.85 prompt-cache ratio when unavailable.
    """
    if transcript_path_str:
        tp = Path(transcript_path_str)
        if tp.is_file():
            with contextlib.suppress(OSError):
                size = tp.stat().st_size
                with tp.open("rb") as fh:
                    if size > 32768:
                        fh.seek(size - 32768)
                    tail_text = fh.read().decode("utf-8", errors="replace")
                for raw_ln in reversed(tail_text.splitlines()):
                    raw_ln = raw_ln.strip()
                    if not raw_ln or "PLANNER_RESPONSE" not in raw_ln:
                        continue
                    try:
                        obj = json.loads(raw_ln)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict) and obj.get("type") == "PLANNER_RESPONSE":
                        in_tok = int(obj.get("input_tokens") or 0)
                        cache_tok = int(obj.get("cache_read_tokens") or 0)
                        total_ctx = in_tok + cache_tok
                        if total_ctx > 0:
                            clamped_ctx = max(2_500, min(120_000, total_ctx))
                            # If cache_read_tokens > 0 use observed ratio; otherwise default to 0.85
                            # for subsequent cached tool round-trips within the same turn.
                            c_ratio = min(0.95, cache_tok / total_ctx) if cache_tok > 0 else 0.85
                            return (clamped_ctx, c_ratio)
    fallback_ctx = (
        max(4_500, min(120_000, int((transcript_kb * 1024.0) / 4.0)))
        if transcript_kb > 0
        else 12_000
    )
    return (fallback_ctx, 0.85)


def _estimate_avoided_commit_tokens(
    transcript_kb: float,
    action_mode: str,
    *,
    transcript_path_str: str = "",
    user_named_body: bool = False,
) -> int:
    """
    Ground counterfactual token savings in the actual active context size, adjusted for
    prompt caching (cached reads billed at ~10% of uncached input tokens) and whether the
    user already explicitly named `body <cmd>`.
    """
    active_ctx, cache_ratio = _resolve_live_context_tokens(transcript_path_str, transcript_kb)
    # Effective uncached-equivalent cost of 1 avoided cached round-trip
    roundtrip_equiv = min(22_000, max(2_500, int(active_ctx * (1.0 - 0.90 * cache_ratio)) + 1_200))

    if action_mode == "spinal_short_circuit":
        # Conservative 1.2 avoided tool-prep round-trips (does not assume host halted Turn 1
        # if the harness still invokes a 1-line echo turn).
        return min(28_000, max(3_500, int(roundtrip_equiv * 1.25)))
    if action_mode == "preinv_directive":
        if user_named_body:
            # User already requested `body commit`; directive adds concise contract/flag guidance only.
            return min(4_500, max(1_200, int(roundtrip_equiv * 0.25)))
        # Bare imperative ("commit ..."): avoids ~0.85 exploratory git status/diff/add turn.
        return min(18_000, max(3_500, int(roundtrip_equiv * 0.85)))
    # pretool_git_rewrite: probabilistic expected retry avoidance (~15% collision/lint retry rate)
    return 900


def _record_reflex_telemetry(
    *,
    action_mode: str,
    target: str,
    raw_tokens_est: int,
    compressed_tokens_est: int,
    latency_ms: float,
    payload: dict[str, Any] | None = None,
) -> None:
    with contextlib.suppress(ImportError, OSError, sqlite3.Error, ValueError, TypeError):
        from whole_body_compute.shield import record_shield_event

        saved = max(0, int(raw_tokens_est) - int(compressed_tokens_est))
        record_shield_event(
            event_type="command_reflex",
            target=target,
            raw_tokens_est=int(raw_tokens_est),
            compressed_tokens_est=int(compressed_tokens_est),
            tokens_saved=saved,
            latency_ms=round(float(latency_ms), 3),
            action_mode=action_mode,
            payload=payload,
        )


def _can_run_spinal_short_circuit(payload: dict[str, Any] | None, mode: str) -> bool:
    if mode != "hybrid":
        return False
    if os.environ.get("WBC_ENABLE_SPINAL_SHORT_CIRCUIT") == "1":
        return True
    if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("VOICEFI_STRICT_TEST_HOOK") == "1":
        # In unit tests, only execute live git commits when explicitly testing spinal short-circuit
        return False
    # Claude Code UserPromptSubmit does not use Antigravity's terminationBehavior+user_markdown contract
    return not (
        isinstance(payload, dict)
        and ("hook_event_name" in payload or payload.get("hookEventName") == "UserPromptSubmit")
    )


def _load_repo_reflex_config(repo_root: Path) -> dict[str, Any]:
    """Load optional repo-local `.wbc_reflex.json` (zones + blocked_paths) if present."""
    cfg_path = repo_root / ".wbc_reflex.json"
    if not cfg_path.is_file():
        return {}
    with contextlib.suppress(OSError, json.JSONDecodeError, TypeError, ValueError):
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    return {}


def check_repo_commit_zones(repo_root: Path, rel_paths: list[str]) -> tuple[bool, str]:
    """
    Validate candidate commit paths against optional repo-local `.wbc_reflex.json`.
    Returns `(ok, error_message)`. When no `.wbc_reflex.json` exists, always returns `(True, "")`.
    """
    cfg = _load_repo_reflex_config(repo_root)
    if not cfg or not rel_paths:
        return (True, "")

    blocked = [str(p).strip() for p in (cfg.get("blocked_paths") or []) if str(p).strip()]
    for rp in rel_paths:
        norm = rp.replace("\\", "/").lstrip("./")
        for bp in blocked:
            bp_norm = bp.replace("\\", "/").lstrip("./")
            if norm == bp_norm or norm.startswith(bp_norm.rstrip("/") + "/"):
                return (False, f"Blocked private path `{norm}` matches `.wbc_reflex.json` blocked_paths (`{bp}`).")

    zones_cfg = cfg.get("zones")
    if isinstance(zones_cfg, dict) and zones_cfg:
        hit_zones: dict[str, list[str]] = {}
        for rp in rel_paths:
            norm = rp.replace("\\", "/").lstrip("./")
            for z_name, prefixes in zones_cfg.items():
                if not isinstance(prefixes, list):
                    continue
                for pref in prefixes:
                    p_norm = str(pref).replace("\\", "/").lstrip("./")
                    if not p_norm:
                        continue
                    if norm == p_norm or (p_norm.endswith("/") and norm.startswith(p_norm)):
                        hit_zones.setdefault(str(z_name), []).append(norm)
                        break
        if len(hit_zones) > 1:
            summary = "; ".join(f"{z} ({len(fs)} files: {', '.join(fs[:3])})" for z, fs in hit_zones.items())
            return (
                False,
                (f"Cross-zone commit guard triggered across {len(hit_zones)} zones ({summary}). "
                "Commit each zone separately with explicit paths, e.g. `uv run body commit src/ tests/`."),
            )
    return (True, "")


def _resolve_repo_cwd(payload: dict[str, Any] | None) -> Path:
    if isinstance(payload, dict):
        cwd_val = str(payload.get("cwd") or "").strip()
        if cwd_val and Path(cwd_val).is_dir():
            return Path(cwd_val)
        ws = payload.get("workspacePaths")
        if isinstance(ws, list) and ws:
            first_ws = str(ws[0] or "").strip()
            if first_ws and Path(first_ws).is_dir():
                return Path(first_ws)
    return Path.cwd()


def evaluate_preinvocation_command_reflex(
    payload: dict[str, Any] | None,
    transcript_path_str: str = "",
    transcript_kb: float = 0.0,
) -> dict[str, Any]:
    """
    Evaluate `PreInvocation` / `UserPromptSubmit` user prompt for deterministic `body` reflexes.

    Returns dict:
      {
        "matched": bool,
        "action_mode": str,          # "spinal_short_circuit" | "spinal_failthrough" | "preinv_directive" | ""
        "target": str,               # e.g. "body commit", "body flush", "body checkpoint"
        "system_message": str,       # model-visible systemMessage ("" on pure short-circuit)
        "user_markdown": str,        # user-visible inline markdown receipt (non-empty on short-circuit)
        "termination_behavior": str, # "terminate" on short-circuit success, else ""
        "latency_ms": float,
        "tokens_saved": int,
      }
    """
    t0 = time.perf_counter()
    empty_res: dict[str, Any] = {
        "matched": False,
        "action_mode": "",
        "target": "",
        "system_message": "",
        "user_markdown": "",
        "termination_behavior": "",
        "latency_ms": 0.0,
        "tokens_saved": 0,
    }

    mode = resolve_command_reflex_mode()
    if mode == "off":
        return empty_res

    user_text, is_fresh = extract_clean_user_prompt(payload, transcript_path_str=transcript_path_str)
    if not is_fresh or not user_text or len(user_text) > 600:
        return empty_res

    stripped = user_text.strip()
    stripped_no_quote = _LEADING_QUOTE_PREFIX_RE.sub("", stripped).strip()
    low = stripped.lower()

    # Never hijack analytical discussions or open-ended questions (including mid-prompt '?' clauses)
    is_question = "?" in stripped
    if _DISCUSSION_PREFIX_RE.match(stripped_no_quote):
        return empty_res

    user_named_body = bool(_USER_NAMED_BODY_RE.search(low))

    # -------------------------------------------------------------------------
    # 1. Standalone Spinal Short-Circuit Candidates (`commit`, `wip`, `checkpoint`)
    # -------------------------------------------------------------------------
    is_standalone_wip = not is_question and bool(_STANDALONE_WIP_RE.match(stripped))
    is_standalone_commit = (
        not is_question
        and not is_standalone_wip
        and not _COMMIT_PROSE_EXCLUSION_RE.search(stripped)
        and bool(_STANDALONE_COMMIT_RE.match(stripped))
    )
    zone_commit_name: str | None = None
    if not is_question and not is_standalone_commit and not is_standalone_wip and not _COMMIT_PROSE_EXCLUSION_RE.search(stripped):
        zm = _STANDALONE_ZONE_COMMIT_RE.match(stripped)
        if zm:
            cand_zone = zm.group(1).strip()
            repo_cfg_z = _load_repo_reflex_config(_resolve_repo_cwd(payload))
            zones_dict = repo_cfg_z.get("zones") if isinstance(repo_cfg_z.get("zones"), dict) else {}
            if cand_zone in zones_dict:
                zone_commit_name = cand_zone
    ckpt_match = None if is_question else _STANDALONE_CHECKPOINT_RE.match(stripped)

    if (is_standalone_commit or is_standalone_wip or zone_commit_name) and _can_run_spinal_short_circuit(payload, mode):
        repo_cwd = _resolve_repo_cwd(payload)
        try:
            from whole_body_compute.local.git_fastpath import (
                resolve_repo_root,
                resolve_smart_scope,
                run_fast_commit,
            )

            resolved_repo = resolve_repo_root(repo_cwd)
            if (resolved_repo / ".git").exists():
                session_id = str((payload or {}).get("conversationId") or (payload or {}).get("conversation_id") or "") or None
                if not zone_commit_name:
                    scoped_preview, _ = resolve_smart_scope(None, session_id, False, resolved_repo)
                    zone_ok, zone_err = check_repo_commit_zones(resolved_repo, scoped_preview)
                    if not zone_ok:
                        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                        fail_msg = (
                            f"🛡️ [WBC Repo Zone Guard (`{elapsed_ms:.0f}ms`)]: {zone_err}"
                        )
                        _record_reflex_telemetry(
                            action_mode="spinal_failthrough",
                            target="body commit",
                            raw_tokens_est=2_500,
                            compressed_tokens_est=180,
                            latency_ms=elapsed_ms,
                            payload=payload,
                        )
                        return {
                            "matched": True,
                            "action_mode": "spinal_failthrough",
                            "target": "body commit",
                            "system_message": fail_msg,
                            "user_markdown": "",
                            "termination_behavior": "",
                            "latency_ms": elapsed_ms,
                            "tokens_saved": 2_320,
                        }

                res = run_fast_commit(
                    wip=is_standalone_wip,
                    no_llm=True,
                    session_id=session_id,
                    zone=zone_commit_name,
                    repo_root=resolved_repo,
                )
                elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                status = res.get("status")
                if status in ("ok", "clean"):
                    avoided_tok = _estimate_avoided_commit_tokens(
                        transcript_kb,
                        "spinal_short_circuit",
                        transcript_path_str=transcript_path_str,
                        user_named_body=user_named_body,
                    )
                    if status == "clean":
                        md = (
                            f"⚡ **WBC Spinal Commit Reflex** (`{elapsed_ms:.0f}ms` • `0` cloud tokens): "
                            "Working tree has no modified tracked files to commit."
                        )
                    else:
                        short_sha = str(res.get("commit_sha") or "")[:7]
                        branch = str(res.get("branch") or "HEAD").replace("refs/heads/", "")
                        msg = str(res.get("message") or "").splitlines()[0]
                        files = res.get("files") or []
                        file_preview = ", ".join(f"`{f}`" for f in files[:6])
                        if len(files) > 6:
                            file_preview += f" (+{len(files) - 6} more)"
                        md = (
                            f"✅ **WBC Spinal Commit Reflex** (`{elapsed_ms:.0f}ms` • `0` cloud tokens)\n"
                            f"- **Commit:** `{short_sha}` on `{branch}` - {msg}\n"
                            f"- **Scoped Files ({len(files)}):** {file_preview}"
                        )
                    receipt_comp = max(60, len(md) // 4)
                    _record_reflex_telemetry(
                        action_mode="spinal_short_circuit",
                        target="body commit",
                        raw_tokens_est=avoided_tok,
                        compressed_tokens_est=receipt_comp,
                        latency_ms=elapsed_ms,
                        payload=payload,
                    )
                    return {
                        "matched": True,
                        "action_mode": "spinal_short_circuit",
                        "target": "body commit",
                        "system_message": (
                            f"Already executed by WBC Spinal Commit Reflex:\n{md}\n"
                            "Echo this concise receipt only; do NOT call any tools or re-run commit."
                        ),
                        "user_markdown": md,
                        "termination_behavior": "terminate",
                        "latency_ms": elapsed_ms,
                        "tokens_saved": max(0, avoided_tok - receipt_comp),
                    }
                else:
                    err_detail = str(res.get("error") or "unknown error").strip()
                    fail_msg = (
                        f"⚡ [WBC Commit Reflex Pre-Check (`{elapsed_ms:.0f}ms`)]: `body commit` caught a pre-commit gate issue:\n"
                        f"{err_detail}\n"
                        "Fix the reported issue (or specify explicit untracked file paths if needed) and commit via `uv run body commit`."
                    )
                    _record_reflex_telemetry(
                        action_mode="spinal_failthrough",
                        target="body commit",
                        raw_tokens_est=2_500,
                        compressed_tokens_est=220,
                        latency_ms=elapsed_ms,
                        payload=payload,
                    )
                    return {
                        "matched": True,
                        "action_mode": "spinal_failthrough",
                        "target": "body commit",
                        "system_message": fail_msg,
                        "user_markdown": "",
                        "termination_behavior": "",
                        "latency_ms": elapsed_ms,
                        "tokens_saved": 2_280,
                    }
        except (ImportError, OSError, subprocess.SubprocessError, ValueError, RuntimeError):
            pass

    if ckpt_match and _can_run_spinal_short_circuit(payload, mode):
        repo_cwd = _resolve_repo_cwd(payload)
        with contextlib.suppress(ImportError, OSError, subprocess.SubprocessError, ValueError, RuntimeError):
            from whole_body_compute.local.git_fastpath import (
                resolve_repo_root,
                run_checkpoint,
            )

            resolved_repo = resolve_repo_root(repo_cwd)
            if (resolved_repo / ".git").exists():
                label = ckpt_match.group(1) or None
                res = run_checkpoint(label=label, repo_root=resolved_repo)
                elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                if res.get("status") == "ok":
                    avoided_tok = _estimate_avoided_commit_tokens(
                        transcript_kb,
                        "spinal_short_circuit",
                        transcript_path_str=transcript_path_str,
                        user_named_body=user_named_body,
                    )
                    ref_name = str(res.get("ref") or "")
                    short_sha = str(res.get("commit_sha") or "")[:7]
                    md = (
                        f"📌 **WBC Spinal Checkpoint Reflex** (`{elapsed_ms:.0f}ms` • `0` cloud tokens)\n"
                        f"- **Saved Ref:** `{ref_name}` (`{short_sha}`) - working tree & `.git/index` untouched."
                    )
                    receipt_comp = max(60, len(md) // 4)
                    _record_reflex_telemetry(
                        action_mode="spinal_short_circuit",
                        target="body checkpoint",
                        raw_tokens_est=avoided_tok,
                        compressed_tokens_est=receipt_comp,
                        latency_ms=elapsed_ms,
                        payload=payload,
                    )
                    return {
                        "matched": True,
                        "action_mode": "spinal_short_circuit",
                        "target": "body checkpoint",
                        "system_message": (
                            f"Already executed by WBC Spinal Checkpoint Reflex:\n{md}\n"
                            "Echo this concise receipt only; do NOT call any tools or re-run checkpoint."
                        ),
                        "user_markdown": md,
                        "termination_behavior": "terminate",
                        "latency_ms": elapsed_ms,
                        "tokens_saved": max(0, avoided_tok - receipt_comp),
                    }

    # -------------------------------------------------------------------------
    # 2. Mode B - 1-Line `PreInvocation` Directives (`systemMessage`)
    # -------------------------------------------------------------------------
    directive = ""
    target_cmd = ""
    raw_est = 6_000

    if not is_question and not _FLUSH_PROSE_EXCLUSION_RE.search(stripped) and _FLUSH_INTENT_RE.search(stripped):
        target_cmd = "body flush"
        raw_est = _estimate_avoided_commit_tokens(
            transcript_kb,
            "preinv_directive",
            transcript_path_str=transcript_path_str,
            user_named_body=user_named_body,
        )
        conv_id = str((payload or {}).get("conversationId") or (payload or {}).get("conversation_id") or "").strip()
        conv_flag = f" --conv-id {shlex.quote(conv_id)}" if conv_id and conv_id not in ("local", "global") else ""
        directive = (
            f"🌊 [WBC Glymphatic Flush Reflex]: Execute `uv run body flush --auto{conv_flag} --goal \"<active goal>\" "
            "--milestone \"<completed milestone>\" --next-step \"<immediate next action>\"` directly via `run_command` "
            "instead of writing a manual chat summary. Include the emitted `conversation://` handoff link in your 1-line reply."
        )
    elif (
        not is_question
        and not _COMMIT_PROSE_EXCLUSION_RE.search(stripped)
        and (is_standalone_commit or is_standalone_wip or _NUANCED_COMMIT_RE.search(stripped))
    ):
        target_cmd = "body commit"
        raw_est = _estimate_avoided_commit_tokens(
            transcript_kb,
            "preinv_directive",
            transcript_path_str=transcript_path_str,
            user_named_body=user_named_body,
        )
        directive = (
            "⚡ [WBC Git Fast-Path Reflex]: Commit in a single command via `uv run body commit` "
            "(or `uv run body commit <paths...> -m \"...\"` / `--wip`) instead of multi-turn `git status` -> `git diff` -> `git add` -> `git commit`. "
            "`body commit` uses an isolated shadow index (`GIT_INDEX_FILE`) with multi-agent collision, secret, AST, and staged `ruff` gates."
        )
    elif not is_question and (ckpt_match or _NUANCED_CHECKPOINT_RE.search(stripped)):
        target_cmd = "body checkpoint"
        raw_est = 2_000 if user_named_body else 4_000
        directive = (
            "📌 [WBC Checkpoint Reflex]: Save a non-destructive `<15ms` git plumbing snapshot via "
            "`uv run body checkpoint [label]` (`refs/checkpoints/*`, zero `.git/index` or worktree mutation) instead of `git stash`."
        )
    elif not is_question and "https://" not in low and _EYES_HANDS_INTENT_RE.search(stripped):
        # Note: URL + PostHog cases continue to be handled by `_build_preinv_url_reflex`
        target_cmd = "body eyes"
        raw_est = 2_500 if user_named_body else 6_000
        directive = (
            "👁️✋ [WBC Foveal Eyes & Hands Reflex]: Use `uv run body eyes --app \"<App>\" --max-elements 60` "
            "and `uv run body hands --app \"<App>\" --click <e_id>` (`<350ms` AXUIElement/Chrome DOM) "
            "instead of spawning an isolated `browser` subagent, `osascript`, or `screencapture`."
        )
    elif not is_question and _MEMORY_INTENT_RE.search(stripped):
        target_cmd = "body memory"
        raw_est = 1_500 if user_named_body else 3_000
        directive = (
            "🧠 [WBC Hippocampus Memory Reflex]: Use `uv run body memory --remember \"<fact>\"`, "
            "`--recall \"<query>\"`, or `--deja-vu \"<error>\"` to persist/search SQLite WAL `dot_memory.db`."
        )
    elif not is_question and _DIAG_INTENT_RE.match(stripped):
        target_cmd = "body diagnostics"
        raw_est = 1_500 if user_named_body else 3_000
        directive = (
            "🩺 [WBC Diagnostics Reflex]: Run `uv run body <doctor|savings|context|scorecard|leaderboard>` "
            "directly via `run_command` without exploratory repo searches."
        )

    if not directive:
        return empty_res

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    comp_est = max(45, len(directive) // 4)
    _record_reflex_telemetry(
        action_mode="preinv_directive",
        target=target_cmd,
        raw_tokens_est=raw_est,
        compressed_tokens_est=comp_est,
        latency_ms=elapsed_ms,
        payload=payload,
    )
    return {
        "matched": True,
        "action_mode": "preinv_directive",
        "target": target_cmd,
        "system_message": directive,
        "user_markdown": "",
        "termination_behavior": "",
        "latency_ms": elapsed_ms,
        "tokens_saved": max(0, raw_est - comp_est),
    }


def _resolve_body_cli_prefix(cwd: Path | None = None, *, absolute: bool = False) -> str:
    """Resolve the cleanest executable prefix for `body` in shell rewrites."""
    work_dir = cwd or Path.cwd()
    venv_body = work_dir / ".venv" / "bin" / "body"
    if venv_body.is_file() and os.access(venv_body, os.X_OK):
        if absolute or " " in str(venv_body):
            return shlex.quote(str(venv_body.resolve()))
        return ".venv/bin/body"
    global_body = shutil.which("body")
    if global_body:
        return shlex.quote(global_body) if " " in global_body else "body"
    if (work_dir / "pyproject.toml").is_file() and shutil.which("uv"):
        return "uv run body"
    return "body"


def _extract_cd_target(cd_stage: str, base_cwd: Path) -> Path | None:
    """Resolve target directory of a leading `cd <dir>` stage if it is a concrete directory."""
    try:
        toks = shlex.split(cd_stage)
    except ValueError:
        return None
    if len(toks) < 2 or toks[0] != "cd":
        return None
    raw_dir = toks[-1]
    if raw_dir.startswith("-") or "$" in raw_dir:
        return None
    cand = Path(os.path.expanduser(raw_dir))
    if not cand.is_absolute():
        cand = base_cwd / cand
    return cand if cand.is_dir() else None


def rewrite_git_commit_to_body_commit(
    cmd_line: str,
    *,
    payload: dict[str, Any] | None = None,
    record_telemetry: bool = True,
) -> dict[str, Any] | None:
    """
    Inspect a `PreToolUse` shell command (`run_command` / `Bash`) and rewrite standard
    `git commit` (or `git add ... && git commit ...`) calls into `body commit`.

    Preserves legitimate raw git plumbing escape hatches (`--amend`, `--allow-empty`,
    `--fixup`, `--squash`, `-C /tmp/...`, heredocs, rebase/merge/cherry-pick).
    """
    if not cmd_line or "git" not in cmd_line or "commit" not in cmd_line:
        return None
    if "body commit" in cmd_line or "wbc commit" in cmd_line or "git_fastpath" in cmd_line:
        return None

    mode = resolve_command_reflex_mode()
    if mode == "off":
        return None

    # Do not rewrite synthetic unit-test git repo setup commands unless explicitly enabled
    if (
        (os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("VOICEFI_STRICT_TEST_HOOK") == "1")
        and os.environ.get("WBC_INTERCEPT_GIT_COMMIT") != "1"
    ):
        return None

    raw = cmd_line.strip()
    if "<<" in raw or "|" in raw or ";" in raw or "||" in raw:
        return None

    t0 = time.perf_counter()
    cwd = _resolve_repo_cwd(payload)
    cwd_str = str(cwd).replace("\\", "/")
    if (
        cwd_str.startswith(("/tmp/", "/private/tmp/", "/var/folders/", "/private/var/folders/"))
        and os.environ.get("WBC_INTERCEPT_GIT_COMMIT") != "1"
    ):
        return None

    stages = [s.strip() for s in re.split(r"\s*&&\s*", raw) if s.strip()]
    if not stages:
        return None

    cd_prefix: str | None = None
    idx = 0
    if stages and re.match(r"^cd\s+", stages[0]):
        cd_stage = stages[0]
        if (
            any(tmp_p in cd_stage for tmp_p in ("/tmp/", "/var/folders/"))
            and os.environ.get("WBC_INTERCEPT_GIT_COMMIT") != "1"
        ):
            return None
        cd_prefix = cd_stage
        idx = 1

    staged_all = False
    explicit_paths: list[str] = []

    # Optional `git status` / `git add ...` prep stages before `git commit`
    while idx < len(stages):
        st = stages[idx]
        try:
            toks = shlex.split(st)
        except ValueError:
            return None
        if not toks:
            idx += 1
            continue
        if Path(toks[0]).name != "git" or len(toks) < 2:
            break
        sub = toks[1]
        if sub == "status":
            idx += 1
            continue
        if sub == "add":
            add_args = toks[2:]
            if any(f in add_args for f in ("-p", "--patch", "-i", "--interactive", "-e", "--edit")):
                return None
            for a in add_args:
                if a in ("-A", "--all", "-u", "--update", "."):
                    staged_all = True
                elif a == "--":
                    continue
                elif a.startswith("-"):
                    if a not in ("-f", "--force", "-v", "--verbose"):
                        return None
                else:
                    explicit_paths.append(a)
            idx += 1
            continue
        break

    if idx >= len(stages):
        return None

    commit_stage = stages[idx]
    trailing_stages = stages[idx + 1 :]
    try:
        c_toks = shlex.split(commit_stage)
    except ValueError:
        return None

    if len(c_toks) < 2 or Path(c_toks[0]).name != "git":
        return None
    # Reject `git -C ... commit` or `git -c ... commit` custom git flags
    if c_toks[1] != "commit":
        return None

    escape_flags = {
        "--amend",
        "--allow-empty",
        "--allow-empty-message",
        "--no-edit",
        "--fixup",
        "--squash",
        "-F",
        "--file",
        "-C",
        "--reuse-message",
        "-c",
        "--reedit-message",
        "--reset-author",
        "--author",
        "--date",
        "-S",
        "--gpg-sign",
    }

    messages: list[str] = []
    no_lint = False
    commit_paths: list[str] = []
    j = 2
    while j < len(c_toks):
        tok = c_toks[j]
        if tok in escape_flags or any(tok.startswith(ef + "=") for ef in escape_flags):
            return None
        if tok in ("-m", "--message"):
            if j + 1 >= len(c_toks):
                return None
            messages.append(c_toks[j + 1])
            j += 2
            continue
        if tok.startswith("--message="):
            messages.append(tok.split("=", 1)[1])
            j += 1
            continue
        if tok.startswith("-m") and len(tok) > 2:
            messages.append(tok[2:])
            j += 1
            continue
        if tok in ("-am", "-ma"):
            if j + 1 >= len(c_toks):
                return None
            staged_all = True
            messages.append(c_toks[j + 1])
            j += 2
            continue
        if tok in ("-a", "--all"):
            staged_all = True
            j += 1
            continue
        if tok in ("-n", "--no-verify"):
            no_lint = True
            j += 1
            continue
        if tok in ("-q", "--quiet", "-v", "--verbose", "-s", "--signoff"):
            j += 1
            continue
        if tok == "--":
            commit_paths.extend(c_toks[j + 1 :])
            break
        if tok.startswith("-"):
            return None
        commit_paths.append(tok)
        j += 1

    merged_paths: list[str] = []
    for p in explicit_paths + commit_paths:
        if p and p not in merged_paths:
            merged_paths.append(p)

    cd_target_dir = _extract_cd_target(cd_prefix, cwd) if cd_prefix else None
    if cd_target_dir is not None:
        # Prefer target repo's venv/global `body`, falling back to an absolute path to current workspace's `body`
        if (cd_target_dir / ".venv" / "bin" / "body").is_file():
            body_prefix = _resolve_body_cli_prefix(cd_target_dir, absolute=True)
        else:
            body_prefix = _resolve_body_cli_prefix(cwd, absolute=True)
    elif cd_prefix:
        body_prefix = _resolve_body_cli_prefix(cwd, absolute=True)
    else:
        body_prefix = _resolve_body_cli_prefix(cwd, absolute=False)
    parts: list[str] = [body_prefix, "commit"]
    if staged_all and not merged_paths:
        parts.append("-a")
    if no_lint:
        parts.append("--no-lint")
    if messages:
        full_msg = "\n\n".join(m for m in messages if m.strip())
        if full_msg:
            parts.extend(["-m", shlex.quote(full_msg)])
    for mp in merged_paths:
        parts.append(shlex.quote(mp))

    rewritten_core = " ".join(parts)
    chain: list[str] = []
    if cd_prefix:
        chain.append(cd_prefix)
    chain.append(rewritten_core)
    chain.extend(trailing_stages)
    healed_cmd = " && ".join(chain)

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    if record_telemetry:
        _record_reflex_telemetry(
            action_mode="pretool_git_rewrite",
            target="body commit",
            raw_tokens_est=900,
            compressed_tokens_est=150,
            latency_ms=elapsed_ms,
            payload=payload,
        )
    return {
        "rewritten_command": healed_cmd,
        "original_command": raw,
        "tokens_saved": 750,
        "latency_ms": elapsed_ms,
        "reason": (
            f"⚡ Blood-Brain Barrier Git Fast-Path Reflex: Upgraded raw `git commit` -> `{rewritten_core}` "
            "(isolated GIT_INDEX_FILE shadow index, multi-agent collision guard, secret + AST + ruff gates)."
        ),
    }


_ENV_PREFIX_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")
_UV_PASSTHROUGH_FLAGS = {
    "--no-sync",
    "--frozen",
    "--offline",
    "-q",
    "--quiet",
    "-s",
    "--no-project",
    "--no-dev",
    "--all-extras",
    "--active",
}


def _has_unquoted_shell_control(raw: str) -> bool:
    """Return True if `raw` contains shell pipes, redirects, chains, or command substitutions."""
    in_single = False
    in_double = False
    escaped = False
    n = len(raw)
    i = 0
    while i < n:
        ch = raw[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if ch == "\\" and not in_single:
            escaped = True
            i += 1
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            i += 1
            continue
        if not in_single:
            if ch in ("`", "$"):
                return True
            if not in_double and ch in ("|", ">", "<", ";", "&", "\n"):
                return True
        i += 1
    return False


def _format_env_prefix(env_tokens: list[str]) -> str:
    formatted: list[str] = []
    for tok in env_tokens:
        k, v = tok.split("=", 1)
        if v == "" or any(c.isspace() or c in "\"'\\$`|&;<>()*" for c in v):
            formatted.append(f"{k}={shlex.quote(v)}")
        else:
            formatted.append(f"{k}={v}")
    return " ".join(formatted)


def _join_cmd_tokens(env_tokens: list[str], cmd_tokens: list[str]) -> str:
    core = " ".join(shlex.quote(t) for t in cmd_tokens)
    if not env_tokens:
        return core
    return f"{_format_env_prefix(env_tokens)} {core}"


def _has_short_flag_char(args: list[str], chars: set[str]) -> bool:
    for arg in args:
        if arg == "--":
            break
        if (
            arg.startswith("-")
            and not arg.startswith("--")
            and len(arg) > 1
            and any(c in chars for c in arg[1:])
        ):
            return True
    return False


def rewrite_read_only_inspection_command(
    cmd_line: str,
    *,
    payload: dict[str, Any] | None = None,
    record_telemetry: bool = True,
) -> dict[str, Any] | None:
    """
    Layer 2 (`PreToolUse` shell rewrite) fast-path for high-frequency read-only
    inspection, search, lint, and test commands (`git diff`, `git status`, `git show`,
    `ls -la`, `rg`, `grep`, `uv run pytest`, `uv run ruff check`) when emitted without
    explicit narrowing flags, unquoted pipes, redirects, or subshells.
    """
    if not cmd_line:
        return None
    if resolve_command_reflex_mode() == "off":
        return None

    raw = cmd_line.strip()
    if not raw or _has_unquoted_shell_control(raw):
        return None

    try:
        toks = shlex.split(raw)
    except ValueError:
        return None
    if not toks:
        return None

    env_tokens: list[str] = []
    cmd_idx = 0
    while cmd_idx < len(toks) and _ENV_PREFIX_RE.match(toks[cmd_idx]):
        env_tokens.append(toks[cmd_idx])
        cmd_idx += 1
    core_toks = toks[cmd_idx:]
    if not core_toks:
        return None

    t0 = time.perf_counter()
    exe = Path(core_toks[0]).name
    uv_prefix: list[str] = []
    target_toks = core_toks
    if exe == "uv" and len(core_toks) >= 3 and core_toks[1] == "run":
        uv_idx = 2
        while uv_idx < len(core_toks) and core_toks[uv_idx] in _UV_PASSTHROUGH_FLAGS:
            uv_idx += 1
        if uv_idx < len(core_toks):
            uv_prefix = core_toks[:uv_idx]
            target_toks = core_toks[uv_idx:]

    inner_exe = Path(target_toks[0]).name
    healed_cmd: str | None = None
    target_label = ""
    tokens_saved = 0

    if not uv_prefix and inner_exe == "git" and len(target_toks) >= 2:
        sub = target_toks[1]
        args = target_toks[2:]
        pre_dd = args[: args.index("--")] if "--" in args else args
        if sub == "diff" and not any(a.startswith("-") for a in pre_dd):
            healed_cmd = _join_cmd_tokens(env_tokens, ["git", "diff", "--stat", *args])
            target_label = "git diff --stat"
            tokens_saved = 600
        elif sub == "status" and not any(a.startswith("-") for a in pre_dd):
            healed_cmd = _join_cmd_tokens(env_tokens, ["git", "status", "--short", *args])
            target_label = "git status --short"
            tokens_saved = 250
        elif sub == "show" and not any(a.startswith("-") for a in pre_dd):
            # Preserve `git show rev:path` file-blob lookups where `--stat` suppresses content.
            if not any(":" in a for a in pre_dd):
                healed_cmd = _join_cmd_tokens(env_tokens, ["git", "show", "--stat", *args])
                target_label = "git show --stat"
                tokens_saved = 600
    elif not uv_prefix and inner_exe == "ls" and len(target_toks) >= 2:
        args = target_toks[1:]
        flags = [a for a in args if a.startswith("-")]
        paths = [a for a in args if not a.startswith("-")]
        if (flags in (["-la"], ["-al"]) or set(flags) == {"-l", "-a"}) and len(paths) <= 1:
            healed_cmd = f"{raw} | head -n 60"
            target_label = "ls -la | head -n 60"
            tokens_saved = 400
    elif inner_exe == "rg":
        if os.environ.get("WBC_STRICT_SHELL_BLOAT_BLOCK") == "1":
            return None
        args = target_toks[1:]
        rg_pre_dd = args[: args.index("--")] if "--" in args else args
        rg_skip_exact = {
            "-m", "--max-count", "-l", "--files-with-matches", "--files", "-c", "--count",
            "--max-columns", "--json", "--vimgrep", "--stats", "--trace", "--debug",
            "--type-list", "--pcre2-version", "-h", "--help", "-V", "--version", "-q", "--quiet",
        }
        has_rg_bound = any(
            a in rg_skip_exact
            or a.startswith(("--max-count=", "--max-columns="))
            or (a.startswith("-m") and len(a) > 2 and a[2:].lstrip("=").isdigit())
            for a in rg_pre_dd
        ) or _has_short_flag_char(args, {"m", "l", "c", "q", "h", "V"})
        if not has_rg_bound and args:
            healed_cmd = _join_cmd_tokens(
                env_tokens,
                [*uv_prefix, target_toks[0], "--max-count=50", "--max-columns=200", *args],
            )
            target_label = "uv run rg --max-count=50" if uv_prefix else "rg --max-count=50"
            tokens_saved = 500
    elif inner_exe in ("grep", "egrep", "fgrep"):
        args = target_toks[1:]
        grep_pre_dd = args[: args.index("--")] if "--" in args else args
        has_line_num = _has_short_flag_char(args, {"n"}) or "--line-number" in grep_pre_dd
        is_recursive = _has_short_flag_char(args, {"r", "R"}) or any(
            a in ("--recursive", "--dereference-recursive") for a in grep_pre_dd
        )
        grep_skip_exact = {
            "-m", "--max-count", "-l", "--files-with-matches", "-c", "--count",
            "-q", "--quiet", "-h", "--help", "-V", "--version",
        }
        has_grep_bound = any(
            a in grep_skip_exact
            or a.startswith("--max-count=")
            or (a.startswith("-m") and len(a) > 2 and a[2:].lstrip("=").isdigit())
            for a in grep_pre_dd
        ) or _has_short_flag_char(args, {"m", "l", "c", "q", "h", "V"})
        if has_line_num and not is_recursive and not has_grep_bound and args:
            healed_cmd = _join_cmd_tokens(
                env_tokens,
                [*uv_prefix, target_toks[0], "-m", "50", *args],
            )
            target_label = f"{inner_exe} -m 50"
            tokens_saved = 400
    else:
        # Unwrap `pytest` / `ruff check` directly, under `uv run`, or via `python3 -m <tool>`
        py_module_prefix: list[str] = []
        tool_toks = target_toks
        if (
            inner_exe.startswith("python")
            and len(target_toks) >= 3
            and target_toks[1] == "-m"
        ):
            py_module_prefix = target_toks[:2]
            tool_toks = target_toks[2:]
        tool_name = Path(tool_toks[0]).name
        tool_args = tool_toks[1:]

        if tool_name == "pytest":
            if os.environ.get("WBC_STRICT_SHELL_BLOAT_BLOCK") == "1":
                return None
            pytest_skip_exact = {
                "-q", "--quiet", "-h", "--help", "-V", "--version",
                "--collect-only", "--co", "-v", "--verbose",
            }
            has_pytest_flag = any(
                a in pytest_skip_exact for a in tool_args
            ) or _has_short_flag_char(tool_args, {"q", "v"})
            if not has_pytest_flag:
                has_tb = any(a == "--tb" or a.startswith("--tb=") for a in tool_args)
                inject = ["-q"] if has_tb else ["-q", "--tb=short"]
                if "--" in tool_args:
                    dd = tool_args.index("--")
                    ordered_args = [*tool_args[:dd], *inject, *tool_args[dd:]]
                else:
                    ordered_args = [*tool_args, *inject]
                healed_cmd = _join_cmd_tokens(
                    env_tokens,
                    [*uv_prefix, *py_module_prefix, tool_toks[0], *ordered_args],
                )
                target_label = "uv run pytest -q" if uv_prefix else "pytest -q"
                tokens_saved = 800
        elif tool_name == "ruff" and len(tool_args) >= 1 and tool_args[0] == "check":
            ruff_args = tool_args[1:]
            ruff_skip_exact = {
                "--output-format", "--format", "-q", "--quiet", "-s", "--silent",
                "--fix", "--unsafe-fixes", "--diff", "--show-fixes", "--statistics",
                "-h", "--help", "-V", "--version", "-w", "--watch", "--add-noqa",
            }
            has_ruff_flag = any(
                a in ruff_skip_exact or a.startswith(("--output-format=", "--format="))
                for a in ruff_args
            ) or _has_short_flag_char(ruff_args, {"q", "s", "h", "V", "w"})
            if not has_ruff_flag:
                healed_cmd = _join_cmd_tokens(
                    env_tokens,
                    [*uv_prefix, *py_module_prefix, tool_toks[0], "check", "--output-format=concise", *ruff_args],
                )
                target_label = (
                    "uv run ruff check --output-format=concise"
                    if uv_prefix
                    else "ruff check --output-format=concise"
                )
                tokens_saved = 450

    if not healed_cmd:
        return None

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    if record_telemetry:
        _record_reflex_telemetry(
            action_mode="pretool_inspection_fastpath",
            target=target_label,
            raw_tokens_est=tokens_saved + 100,
            compressed_tokens_est=100,
            latency_ms=elapsed_ms,
            payload=payload,
        )
    return {
        "rewritten_command": healed_cmd,
        "original_command": raw,
        "target": target_label,
        "tokens_saved": tokens_saved,
        "latency_ms": elapsed_ms,
        "reason": f"⚡ Blood-Brain Barrier Inspection Fast-Path: `{raw}` -> `{healed_cmd}`",
    }


def get_command_reflex_telemetry_breakdown(
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """
    Query `shield_savings_ledger` grouped by `action_mode` and `target` for `event_type='command_reflex'`,
    applying the exact same 120s cooldown and per-session `body flush` high-water-mark deduplication
    as `shield.py`'s `get_summary()` so totals match `body savings` 1:1.
    """
    from whole_body_compute.shield import get_default_shield_db_path

    resolved_db = Path(os.path.expanduser(str(db_path))) if db_path else get_default_shield_db_path()
    out: dict[str, Any] = {
        "active_mode": resolve_command_reflex_mode(),
        "db_path": str(resolved_db),
        "total_reflex_events": 0,
        "total_tokens_saved": 0,
        "raw_logged_tokens_saved": 0,
        "by_mode": {},
        "by_target": {},
    }
    if not resolved_db.is_file():
        return out
    with contextlib.suppress(sqlite3.Error, OSError, ValueError, TypeError):
        conn = sqlite3.connect(str(resolved_db), timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT
                    bot_name,
                    session_id,
                    timestamp,
                    action_mode,
                    target,
                    tokens_saved,
                    latency_ms
                FROM shield_savings_ledger
                WHERE event_type = 'command_reflex'
                ORDER BY timestamp ASC, id ASC;
                """
            ).fetchall()
        finally:
            conn.close()

        reflex_last_ts: dict[tuple[str, str, str], float] = {}
        flush_hwm: dict[tuple[str, str, str], int] = {}
        mode_lats: dict[str, list[float]] = {}
        tm_lats: dict[tuple[str, str], list[float]] = {}

        for r in rows:
            b_name = str(r["bot_name"] or "unknown")
            s_id = str(r["session_id"] or "default-session").strip() or "default-session"
            ts_val = float(r["timestamp"] or 0.0)
            am = str(r["action_mode"] or "unknown")
            tgt = str(r["target"] or "unknown")
            saved_db = max(0, int(r["tokens_saved"] or 0))
            lat = float(r["latency_ms"] or 0.0)

            r_key = (b_name, s_id, tgt)
            prev_ts = reflex_last_ts.get(r_key)
            reflex_last_ts[r_key] = ts_val
            if tgt == "body flush":
                prev_c = flush_hwm.get(r_key, 0)
                ceiling = max(prev_c, saved_db)
                eff_saved = min(saved_db, max(0, ceiling - prev_c))
                flush_hwm[r_key] = prev_c + eff_saved
            elif prev_ts is not None and 0.0 <= (ts_val - prev_ts) < 120.0:
                eff_saved = 0
            else:
                eff_saved = saved_db

            out["total_reflex_events"] += 1
            out["total_tokens_saved"] += eff_saved
            out["raw_logged_tokens_saved"] += saved_db

            mode_lats.setdefault(am, []).append(lat)
            tm_lats.setdefault((tgt, am), []).append(lat)

            m_bucket = out["by_mode"].setdefault(am, {"events": 0, "tokens_saved": 0, "avg_latency_ms": 0.0})
            m_bucket["events"] += 1
            m_bucket["tokens_saved"] += eff_saved

            t_bucket = out["by_target"].setdefault(tgt, {"events": 0, "tokens_saved": 0, "modes": {}})
            t_bucket["events"] += 1
            t_bucket["tokens_saved"] += eff_saved
            tm_bucket = t_bucket["modes"].setdefault(am, {"events": 0, "tokens_saved": 0, "avg_latency_ms": 0.0})
            tm_bucket["events"] += 1
            tm_bucket["tokens_saved"] += eff_saved

        for am, lats in mode_lats.items():
            out["by_mode"][am]["avg_latency_ms"] = round(sum(lats) / len(lats), 2) if lats else 0.0
        for (tgt, am), lats in tm_lats.items():
            out["by_target"][tgt]["modes"][am]["avg_latency_ms"] = round(sum(lats) / len(lats), 2) if lats else 0.0
    return out
