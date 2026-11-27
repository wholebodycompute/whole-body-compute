"""
Whole Body Compute - The Blood-Brain Barrier Shield Installer & Live Savings Ledger (`body shield` & `body savings`).
# [direct-edit]

Provides:
1. `ShieldTelemetryLedger` & `record_shield_event`:
   Sub-2ms, thread-safe SQLite WAL telemetry ledger (`~/.whole_body_compute/dot_memory.db` table
   `shield_savings_ledger`, or custom path via `WBC_SHIELD_DB_PATH`) tracking every
   PreToolUse AST symbol slice, Local Metal ReconScout compression, Foveal episodic memory local
   digest, vascular block, and PostToolUse secret redaction across all bots.
2. `ShieldInstaller`:
   Zero-schema-bloat (`0` MCP tool tokens) multi-bot hook installer & auditor for
   Antigravity (`.gemini/settings.json`), Claude Code (`.claude/settings.json`),
   Cursor (`.cursor/hooks.json`), and OpenAI Codex (`.codex/hooks.json`).
3. `get_savings_summary` & `format_savings_receipt`:
   Computes direct token compression, 10-turn quadratic O(N^2) re-ingestion savings,
   estimated frontier API USD saved, rate-limit quota runway multiplier, secrets
   scrubbed, and renders the live ASCII receipt for `body savings`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

WBC_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HOOK_SCRIPT = WBC_REPO_ROOT / "bin" / "body-immune-hook"

from datetime import datetime, timezone

# Official Gemini 4 Argon/Carbon introductory pricing ($2/$10) runs through 2026-11-30 UTC;
# standard pricing ($4/$20) takes effect 2026-12-01T00:00:00Z.
_GEMINI_4_INTRO_CUTOFF_TS: float = datetime(2026, 12, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp()

MODEL_INPUT_PRICING_PER_M: Dict[str, float] = {
    "gemini 4 hydrogen": 1.50,
    "gemini-4-hydrogen": 1.50,
    "hydrogen": 1.50,
    "gemini 4 neon": 3.00,
    "gemini-4-neon": 3.00,
    "neon": 3.00,
    "gemini 4 argon (high)": 2.00,
    "gemini-4-argon": 2.00,
    "argon": 2.00,
    "gemini 4 carbon": 2.00,
    "gemini-4-carbon": 2.00,
    "carbon": 2.00,
    "gemini-4": 2.00,
    "gemini-3": 2.00,
    "gemini-2.5-pro": 1.25,
    "gemini-pro": 2.00,
    "gemini flash": 0.30,
    "flash": 0.30,
    "claude haiku": 0.80,
    "haiku": 0.80,
    "claude sonnet": 3.00,
    "sonnet": 3.00,
    "claude-3-opus": 15.00,
    "opus-4-0": 15.00,
    "opus-4-1": 15.00,
    "claude opus": 5.00,
    "opus": 5.00,
    "grok": 2.00,
    "gpt-5": 1.25,
    "gpt": 2.50,
    "codex": 2.50,
    "o3": 2.50,
}  # [direct-edit]

MULTI_HARNESS_TOOL_MATCHER = (
    "read_file|view_file|Read|NotebookRead|developer__text_editor|beforeReadFile|pre_read_code|"
    "edit_file|write_file|Edit|MultiEdit|Write|NotebookEdit|invoke_subagent|Task|Agent|"
    "run_command|manage_task|Bash|execute_command|developer__shell|"
    "grep_search|Grep|find_by_name|Glob|list_dir|LS|read_url_content|WebFetch|"
    "list_resources|read_resource|execute_sql|query_database|call_mcp_tool|mcp_.*|slack_.*"
)

SUPPORTED_BOTS: Dict[str, Dict[str, str]] = {
    "antigravity": {
        "label": "Google Antigravity (AGY)",
        "workspace_rel": ".gemini/settings.json",
        "user_rel": ".gemini/settings.json",
    },
    "claude-code": {
        "label": "Anthropic Claude Code",
        "workspace_rel": ".claude/settings.json",
        "user_rel": ".claude/settings.json",
    },
    "cursor": {
        "label": "Cursor Agent",
        "workspace_rel": ".cursor/hooks.json",
        "user_rel": ".cursor/hooks.json",
    },
    "codex": {
        "label": "OpenAI Codex CLI",
        "workspace_rel": ".codex/hooks.json",
        "user_rel": ".codex/hooks.json",
    },
}

BOT_ALIASES: Dict[str, str] = {
    "antigravity": "antigravity",
    "agy": "antigravity",
    "gemini": "antigravity",
    "claude": "claude-code",
    "claude-code": "claude-code",
    "claudecode": "claude-code",
    "cursor": "cursor",
    "codex": "codex",
    "openai": "codex",
    "cli": "cli",
    "standalone": "cli",
}

HARNESS_FILE_TOKEN_CEILINGS: Dict[str, int] = {
    "antigravity": 11_520,  # 800 lines / 46,080 bytes native read_file ceiling
    "claude-code": 25_000,  # 2,000 lines / ~100KB native Read ceiling
    "cursor": 15_000,
    "codex": 20_000,
    "cli": 25_000,
}

_DEDUP_FILE_EVENT_TYPES = frozenset({"ast_slice_gate", "local_metal_scout", "vascular_block"})
_LEDGER_LOCK = threading.Lock()


def get_harness_file_token_ceiling(bot_name: Optional[str] = None, payload: Optional[Dict[str, Any]] = None) -> int:
    """Return the realistic per-call native file read token ceiling for the active IDE harness."""
    resolved = (
        BOT_ALIASES.get(bot_name.strip().lower(), bot_name.strip().lower())
        if bot_name
        else detect_bot_name(payload)
    )
    return int(HARNESS_FILE_TOKEN_CEILINGS.get(resolved, 25_000))


def _is_ephemeral_temp_target(target: str) -> bool:
    """Return True if `target` points inside OS temp/sandbox scratch directories."""
    t = str(target or "").strip().replace("\\", "/")
    return t.startswith(("/tmp/", "/private/tmp/", "/private/var/folders/", "/var/folders/"))


_BENCHMARK_HARNESS_SUBSTRINGS: Tuple[str, ...] = (
    "swebench",
    "swe-bench",
    "agentdojo",
    "mcptox",
    "osharm",
    "cyberseceval",
    "vanilla-arm",
    "wbc-arm",
    "/tmp/wbc-",
    "/private/tmp/wbc-",
    "/tmp/wbc_",
    "/private/tmp/wbc_",
    "redcode_idx",
    "attacker-drop.example.com",
    "1mocat.github.io",
    "s.42l.fr/",
    "a656d2b5ea3c4274",
    "send_post_request",
    "download_file(url, local_filename)",
    "fetch_url_content",
    "/usr/copy_file",
    "ls /app",
    "file_name = '/etc/passwd'",
    "file_name='/etc/passwd'",
    'os.remove("/etc/passwd")',
    'rm "/etc/passwd"',
    'alias sudo="rm -rf /"',
    "../etc/passwd",
    "safe_calculator",
    "kill_processes",
    "rm -rf --no-preserve-root /",
    "mount -o remount,ro /home",
)

_EPHEMERAL_TEMP_SUBSTRINGS: Tuple[str, ...] = (
    "/tmp/",
    "/private/tmp/",
    "/var/folders/",
    "/private/var/folders/",
    "pytest-of-",
)


def _is_benchmark_or_ephemeral_row(
    target: str,
    session_id: str,
    *,
    exclude_ephemeral_temp: bool = False,
) -> bool:
    """Return True when a ledger/audit row belongs to a benchmark harness or (on prod DB) OS temp scratch."""
    t_low = str(target or "").lower().replace("\\", "/")
    s_low = str(session_id or "").lower().replace("\\", "/")
    for marker in _BENCHMARK_HARNESS_SUBSTRINGS:
        if marker in t_low or marker in s_low:
            return True
    if exclude_ephemeral_temp:
        for tmp_marker in _EPHEMERAL_TEMP_SUBSTRINGS:
            if tmp_marker in t_low or tmp_marker in s_low:
                return True
    return False


def get_default_shield_db_path() -> Path:
    """
    Resolve the active SQLite WAL path for the Shield Telemetry Ledger using pure stdlib
    (safe under `python -S` hooks where third-party packages like PyYAML are unavailable).
    """
    env_override = os.environ.get("WBC_SHIELD_DB_PATH", "").strip()
    if env_override:
        return Path(os.path.expanduser(env_override))
    if (
        os.environ.get("PYTEST_CURRENT_TEST")
        or os.environ.get("WBC_TESTING") == "1"
        or os.environ.get("VOICEFI_TESTING") == "1"
        or os.environ.get("VOICEFI_STRICT_TEST_HOOK") == "1"
        or os.environ.get("WBC_BENCHMARK_MODE") == "1"
        or os.environ.get("WBC_ARENA_MODE") == "1"
    ):
        return Path("/tmp/wbc_pytest_shield_ledger.db")

    env_data_dir = os.environ.get("WBC_DATA_DIR", "").strip()
    if env_data_dir:
        return Path(os.path.expanduser(env_data_dir)) / "dot_memory.db"

    home = Path.home()
    wbc_default = home / ".whole_body_compute"
    if wbc_default.exists():
        return wbc_default / "dot_memory.db"
    voicefi_default = home / ".voicefi"
    if voicefi_default.exists():
        return voicefi_default / "dot_memory.db"
    return wbc_default / "dot_memory.db"


def detect_bot_name(payload: Optional[Dict[str, Any]] = None) -> str:
    """
    Auto-detect the active coding agent / bot harness from hook payload or environment.
    Avoids attributing standalone CLI runs to Antigravity when no IDE markers exist.
    """
    if isinstance(payload, dict):
        explicit = (
            payload.get("bot_name")
            or payload.get("bot")
            or payload.get("harness")
            or payload.get("client")
        )
        if explicit:
            norm = str(explicit).strip().lower()
            return BOT_ALIASES.get(norm, norm)

    env_bot = os.environ.get("WBC_BOT_NAME", "").strip().lower()
    if env_bot:
        return BOT_ALIASES.get(env_bot, env_bot)

    if os.environ.get("CLAUDE_CODE") or os.environ.get("CLAUDE_PROJECT_DIR") or os.environ.get("CLAUDE_SESSION_ID"):
        return "claude-code"
    if os.environ.get("CURSOR_TRACE_ID") or os.environ.get("CURSOR_SESSION_ID"):
        return "cursor"
    if os.environ.get("CODEX_CLI") or os.environ.get("OPENAI_CODEX"):
        return "codex"
    if os.environ.get("GROK_CLI"):
        return "grok"

    if isinstance(payload, dict):
        if "hook_event_name" in payload or ("tool_input" in payload and "toolCall" not in payload):
            return "claude-code"
        if "toolCall" in payload or "conversationId" in payload or "stepIdx" in payload:
            return "antigravity"

    if (
        os.environ.get("ANTIGRAVITY_AGENT")
        or os.environ.get("AGY_CONVERSATION_ID")
        or os.environ.get("ANTIGRAVITY_CLI_ALIAS")
    ):
        return "antigravity"

    return "cli" if payload is None else "antigravity"


def _lookup_model_pricing(
    model_label: str,
    *,
    event_ts: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Resolve `(input_usd_per_m, output_usd_per_m, cache_read_mult)` for a model label.

    Aligned with official `ai.google.dev` Gemini 4 pricing (flat across 1M context, 95% cache
    discount = `0.05x`) and Anthropic/OpenAI standard tiers (90% cache discount = `0.10x`):
      - Gemini 4 Hydrogen: $1.50 in / $7.50 out / 0.05x cache ($0.075/1M)
      - Gemini 4 Neon:     $3.00 in / $15.00 out / 0.05x cache ($0.15/1M)
      - Gemini 4 Argon & Carbon:
          * Intro (through 2026-11-30 UTC): $2.00 in / $10.00 out / 0.05x cache ($0.10/1M)
          * Standard (2026-12-01+ UTC):     $4.00 in / $20.00 out / 0.05x cache ($0.20/1M)
    """
    low = (model_label or "").strip().lower()
    ref_ts = float(event_ts) if (event_ts is not None and event_ts > 0) else time.time()
    is_g4_standard = ref_ts >= _GEMINI_4_INTRO_CUTOFF_TS

    if not low:
        return (4.00, 20.00, 0.05) if is_g4_standard else (2.00, 10.00, 0.05)
    if any(k in low for k in ("local-metal", "local_metal", "body-coder", "voicefi-coder", "ollama", "mlx-community")):
        return (0.0, 0.0, 0.0)
    if "hydrogen" in low:
        return (1.50, 7.50, 0.05)
    if "neon" in low:
        return (3.00, 15.00, 0.05)
    if "argon" in low or "carbon" in low or "gemini-4" in low or "gemini 4" in low:
        return (4.00, 20.00, 0.05) if is_g4_standard else (2.00, 10.00, 0.05)
    if "gemini-2.5-pro" in low:
        return (1.25, 10.00, 0.10)
    if "flash" in low:
        return (0.30, 2.50, 0.10)
    if "gemini" in low:
        return (2.00, 12.00, 0.10)
    if any(k in low for k in ("opus-4-5", "opus-4.5", "opus 4.5", "opus-4-6", "opus 4.6", "opus-5", "opus 5")):
        return (5.00, 25.00, 0.10)
    if "opus" in low:
        return (15.00, 75.00, 0.10)
    if "sonnet" in low:
        return (3.00, 15.00, 0.10)
    if "haiku" in low:
        return (0.80, 4.00, 0.10)
    if "grok" in low:
        return (2.00, 8.00, 0.10)
    if "gpt-5" in low and "pro" not in low:
        return (1.25, 10.00, 0.10)
    if "gpt" in low or "codex" in low or "o3" in low or "o1" in low:
        return (2.50, 10.00, 0.10)
    return (2.00, 10.00, 0.10)


def _lookup_model_rate(model_label: str, *, event_ts: Optional[float] = None) -> float:
    """Resolve input USD per 1M tokens for a given model name."""
    return _lookup_model_pricing(model_label, event_ts=event_ts)[0]


def _estimate_avoided_output_tokens(
    event_type: str,
    tokens_saved: int,
    *,
    action_mode: str = "",
) -> int:
    """
    Conservative avoided output + thinking token estimate ONLY for turn-eliminating events
    (grounded in the empirical 536-token median output+thinking per tool-call turn across 607
    real Antigravity turns). Pure input-slicing events (`ast_slice_gate`, `vascular_block`,
    `secret_redaction`, `hands_tactile`, `pretool_git_rewrite`) return `0`.
    """
    if tokens_saved <= 0:
        return 0
    ev = (event_type or "").strip().lower()
    mode = (action_mode or "").strip().lower()

    if ev == "local_metal_scout" and tokens_saved >= 1_500:
        return 536
    if ev in ("eyes_foveal_dom", "hands_ax_act") and tokens_saved >= 500:
        return 536
    if ev == "crash_triage_fusion":
        return 536
    if ev == "lossless_tail_bridge":
        return 350
    if ev == "command_reflex":
        if mode == "spinal_short_circuit":
            return 650
        if mode == "spinal_failthrough" or (mode == "preinv_directive" and tokens_saved >= 3_500):
            return 450
        return 0
    return 0


def detect_active_session_id(
    payload: Optional[Dict[str, Any]] = None,
    bot_name: Optional[str] = None,
) -> str:
    """Detect the real conversation/session ID without cross-bot transcript mtime hijacking."""
    if isinstance(payload, dict):
        cid = (
            payload.get("conversationId")
            or payload.get("conversation_id")
            or payload.get("session_id")
            or payload.get("sessionId")
        )
        if cid and str(cid).strip() and str(cid).strip() != "local":
            return str(cid).strip()

    for env_k in ("AGY_CONVERSATION_ID", "CLAUDE_SESSION_ID", "WBC_SESSION_ID", "CURSOR_TRACE_ID"):
        val = os.environ.get(env_k, "").strip()
        if val:
            return val

    if os.environ.get("WBC_SHIELD_DB_PATH", "").strip():
        return "default-session"  # [direct-edit]

    resolved_bot = (
        BOT_ALIASES.get(bot_name.strip().lower(), bot_name.strip().lower())
        if bot_name
        else detect_bot_name(payload)
    )
    if resolved_bot == "antigravity":
        brain_root = Path.home() / ".gemini" / "antigravity" / "brain"
        if brain_root.is_dir():
            try:
                best_mtime = -1.0
                best_id = ""
                for child in brain_root.iterdir():
                    if not child.is_dir() or child.name.startswith("."):
                        continue
                    t_file = child / ".system_generated" / "logs" / "transcript.jsonl"
                    if t_file.is_file():
                        mt = t_file.stat().st_mtime
                        if mt > best_mtime:
                            best_mtime = mt
                            best_id = child.name
                # Only attach if transcript was actively updated within the last 15 minutes (900s)
                if best_id and (time.time() - best_mtime) <= 900.0:
                    return best_id
            except Exception:
                pass

    if resolved_bot == "cli":
        return "cli-standalone"
    return "default-session"


_MODEL_SEL_RE = re.compile(
    r"(?:Model Selection:\s*Changed model|`?Model Selection`?)\s+from\s+[^\n\"\\]+?\s+to\s+((?:Gemini|Claude|GPT|Grok|o1|o3)[A-Za-z0-9_. ()\-]{1,40})",
    re.IGNORECASE,
)
_ACTIVE_MODEL_TTL_CACHE: Dict[Tuple[str, str], Tuple[float, str, float]] = {}
_WBC_ROOT_RESOLVED = WBC_REPO_ROOT.resolve()
_HOME_PATH_CACHE: Dict[str, Tuple[Path, str, Path]] = {}
_SCHEMA_READY_DBS: Dict[str, int] = {}
_RECENT_SLICE_MAX_TS_CACHE: Dict[str, Tuple[int, int, float]] = {}
_DEFAULT_DB_NEGATIVE_UNTIL: Dict[str, Tuple[float, float]] = {}


def _get_cached_home_paths() -> Tuple[Path, str, Path]:
    raw_home = str(Path.home())
    cached = _HOME_PATH_CACHE.get(raw_home)
    if cached is not None:
        return cached
    h_res = Path(raw_home).resolve()
    h_norm = str(h_res).replace("\\", "/").lstrip("/")
    p_res = (h_res / "Projects").resolve()
    val = (h_res, h_norm, p_res)
    _HOME_PATH_CACHE[raw_home] = val
    return val


def detect_active_model_and_rate(
    payload: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
) -> Tuple[str, float]:
    """
    Auto-detect the active frontier model and its realistic $/1M input token rate.
    # [direct-edit]
    """
    if isinstance(payload, dict):
        m_exp = payload.get("model") or payload.get("model_name") or payload.get("modelName")
        if m_exp and str(m_exp).strip() and str(m_exp).strip().lower() not in ("inherit", "default"):
            m_str = str(m_exp).strip()
            return m_str, _lookup_model_rate(m_str)

    for env_k in ("WBC_ACTIVE_MODEL", "ANTHROPIC_MODEL", "OPENAI_MODEL", "GROK_MODEL"):
        val = os.environ.get(env_k, "").strip()
        if val:
            return val, _lookup_model_rate(val)

    raw_home = str(Path.home())
    cache_key = (session_id or "", raw_home)
    now_mono = time.monotonic()
    cached_mod = _ACTIVE_MODEL_TTL_CACHE.get(cache_key)
    if cached_mod is not None and (now_mono - cached_mod[0]) < 15.0:
        return cached_mod[1], cached_mod[2]

    skip_host_brain_scan = bool(
        (os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("WBC_SHIELD_DB_PATH"))
        and not (session_id and (Path(raw_home) / ".gemini" / "antigravity" / "brain" / session_id).is_dir())
    )
    brain_root = Path(raw_home) / ".gemini" / "antigravity" / "brain"
    if not skip_host_brain_scan and brain_root.is_dir():
        cand_dirs: List[Path] = []
        if session_id and (brain_root / session_id).is_dir():
            cand_dirs.append(brain_root / session_id)
        try:
            others = [
                d
                for d in brain_root.iterdir()
                if d.is_dir() and not d.name.startswith(".") and (d / ".system_generated" / "logs" / "transcript.jsonl").is_file()
            ]
            others.sort(key=lambda p: (p / ".system_generated" / "logs" / "transcript.jsonl").stat().st_mtime, reverse=True)
            for o in others[:3]:
                if o not in cand_dirs:
                    cand_dirs.append(o)
        except Exception:
            pass

        for cdir in cand_dirs:
            t_file = cdir / ".system_generated" / "logs" / "transcript.jsonl"
            if not t_file.is_file():
                continue
            try:
                f_size = t_file.stat().st_size
                with t_file.open("rb") as fh:
                    if f_size > 65536:
                        fh.seek(f_size - 65536)
                    tail_txt = fh.read().decode("utf-8", errors="replace")
                matches = _MODEL_SEL_RE.findall(tail_txt)
                if not matches and f_size > 65536:
                    with t_file.open("rb") as fh_head:
                        head_txt = fh_head.read(65536).decode("utf-8", errors="replace")
                    matches = _MODEL_SEL_RE.findall(head_txt)
                if matches:
                    detected = matches[-1].strip()
                    if detected:
                        rate = _lookup_model_rate(detected)
                        _ACTIVE_MODEL_TTL_CACHE[cache_key] = (now_mono, detected, rate)
                        return detected, rate
            except Exception:
                pass

    bot = detect_bot_name(payload)
    if bot == "claude-code":
        for claude_cfg in (Path.cwd() / ".claude" / "settings.json", Path.home() / ".claude" / "settings.json"):
            if claude_cfg.is_file():
                try:
                    cdata = json.loads(claude_cfg.read_text(encoding="utf-8"))
                    c_mod = cdata.get("model")
                    if c_mod and isinstance(c_mod, str):
                        return c_mod.strip(), _lookup_model_rate(c_mod)
                except Exception:
                    pass
        return "Claude Sonnet", 3.00

    if bot == "antigravity":
        return "Gemini 4 Carbon", _lookup_model_rate("Gemini 4 Carbon")

    return "Claude Sonnet 4.5", 3.00


def _count_transcript_user_turns(session_id: str, bot_name: str = "antigravity") -> Optional[int]:
    """
    If `session_id` corresponds to an Antigravity conversation directory on disk,
    count the actual `USER_INPUT` turns in `transcript.jsonl`. Never cross-contaminates non-AGY bots.
    """
    if bot_name != "antigravity":
        return None
    if not session_id or session_id in ("default-session", "cli-standalone", "local"):
        return None
    t_file = Path.home() / ".gemini" / "antigravity" / "brain" / session_id / ".system_generated" / "logs" / "transcript.jsonl"
    if not t_file.is_file():
        return None
    try:
        txt = t_file.read_text(encoding="utf-8", errors="replace")
        cnt = txt.count('"USER_INPUT"')
        return max(1, cnt) if cnt > 0 else 1
    except Exception:
        return None


def _normalize_target_label(target: str) -> str:
    """
    Normalize a shield target into a CWD-independent repo-relative file path, app handle, or shell command
    without mangling multi-argument CLI commands via `Path(...).name` or leaking `/` / `$HOME`-relative variants.
    """
    raw = str(target or "").strip()
    if not raw:
        return "unknown"
    if raw.startswith("foveal:") or raw.endswith(":actuate"):
        return raw[:140]

    # Preserve shell commands containing spaces/flags that are not filesystem paths on disk
    if " " in raw and not raw.startswith(("/", "./", "../", "~/")):
        try:
            if not Path(raw).exists():
                return raw[:140]
        except Exception:
            return raw[:140]

    if "/" in raw or "\\" in raw:
        norm = raw.replace("\\", "/")
        home_resolved, home_norm, projects_root = _get_cached_home_paths()
        # Repair historical slash-stripped paths created when CWD was `/` or `$HOME`
        if home_norm and norm.startswith(home_norm + "/"):
            raw = "/" + norm
            norm = raw
        elif norm.startswith("Projects/") and len(norm.split("/")) >= 3:
            raw = str(home_resolved / norm)
            norm = raw.replace("\\", "/")

        if not norm.startswith(("/", "~")):
            while norm.startswith("./"):
                norm = norm[2:]
            return (norm or raw)[:140]
        try:
            p = Path(os.path.expanduser(raw)).resolve()
            try:
                rel_wbc = p.relative_to(_WBC_ROOT_RESOLVED)
                return str(rel_wbc).replace("\\", "/")[:140]
            except ValueError:
                pass
            try:
                rel_proj = p.relative_to(projects_root)
                if len(rel_proj.parts) >= 2:
                    return str(rel_proj).replace("\\", "/")[:140]
            except ValueError:
                pass
            return (p.name or raw)[:140]
        except Exception:
            return (Path(norm).name or raw)[:140]

    return raw[:140]


def _format_duration_compact(seconds: float) -> str:
    """Format signed seconds into compact human-readable wall-clock duration ('14.2s', '-5.7m', '1.42h')."""
    val = float(seconds or 0.0)
    sign = "-" if val < -0.04 else ""
    sec = abs(val)
    if sec < 60.0:
        return f"{sign}{sec:.1f}s"
    if sec < 3600.0:
        return f"{sign}{sec / 60.0:.1f}m"
    return f"{sign}{sec / 3600.0:.2f}h"


def _resolve_window_since_ts(window: Optional[str]) -> Tuple[str, Optional[float]]:
    """Resolve a time-window label ('today', '24h', '7d', '30d', 'all') to (canonical_label, unix_ts)."""
    import datetime

    w = (window or "all").strip().lower()
    now_ts = time.time()
    if w in ("today", "1d-local", "day"):
        local_now = datetime.datetime.now().astimezone()
        midnight = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        return "today", midnight.timestamp()
    if w in ("24h", "1d"):
        return "24h", now_ts - 86400.0
    if w in ("7d", "week"):
        return "7d", now_ts - (7 * 86400.0)
    if w in ("30d", "month"):
        return "30d", now_ts - (30 * 86400.0)
    m_days = re.fullmatch(r"(\d+)d", w)
    if m_days:
        n_days = int(m_days.group(1))
        if n_days > 0:
            return f"{n_days}d", now_ts - (n_days * 86400.0)
    return "all", None


def _estimate_event_time_saved_sec(
    event_type: str,
    raw_tokens: int,
    tokens_saved: int,
    latency_ms: float,
    action_mode: str = "",
) -> float:
    """
    Compute signed, unclamped net wall-clock time delta (seconds) for a shield event:
      net_saved = cloud_counterfactual_sec - actual_local_latency_sec - turn_or_cli_penalties
    Slow cold loads, 0-token polls, and hard-deny recovery turns honestly subtract latency (can be < 0).
    """
    actual_sec = max(0.0, float(latency_ms or 0.0)) / 1000.0
    saved_tok = max(0, int(tokens_saved or 0))
    raw_tok = max(0, int(raw_tokens or 0))
    ev = str(event_type or "")
    mode = str(action_mode or "").strip().lower()

    if ev == "eyes_foveal_dom":
        if saved_tok <= 0:
            return -actual_sec
        # Avoids screencapture PNG + upload + multimodal vision tile encoding (~2.2s)
        counterfactual_sec = 2.2 + (saved_tok / 75_000.0)
        return counterfactual_sec - actual_sec

    if ev in ("hands_ax_act", "hands_tactile", "hands_apfs_sandbox"):
        if saved_tok <= 0:
            return -actual_sec
        # Avoids screenshot -> vision coordinate grounding -> click -> verify loop (~4.5s)
        counterfactual_sec = 4.5 if ev == "hands_ax_act" else 1.5
        return counterfactual_sec - actual_sec

    if ev in ("local_metal_scout", "screen_local_digest"):
        cli_overhead_sec = 0.35 if ev == "local_metal_scout" else 0.0
        if saved_tok <= 0:
            return -(actual_sec + cli_overhead_sec)
        # Counterfactual cloud turn + file read + synthesis (~4.2s over ceiling, ~2.8s under ceiling)
        avoided_turn_sec = 4.2 if raw_tok >= 11_520 else 2.8
        prefill_sec = saved_tok / 75_000.0
        return avoided_turn_sec + prefill_sec - actual_sec - cli_overhead_sec

    if ev == "ast_slice_gate":
        if mode == "hard_deny":
            # Hard-blocking forces an extra LLM tool recovery round-trip (~4.5s penalty)
            return (saved_tok / 75_000.0) - 4.5 - actual_sec
        if saved_tok <= 0:
            return -actual_sec
        avoided_pagination_sec = 3.2 if raw_tok >= 11_520 else 1.8
        prefill_sec = saved_tok / 75_000.0
        return avoided_pagination_sec + prefill_sec - actual_sec

    if ev == "crash_triage_fusion":
        if saved_tok <= 0:
            return -actual_sec
        # Eliminates 1 full exploratory `read_file` LLM round-trip (~5.2s TTFT + tool decode) after a crash/test failure
        return 5.2 + (saved_tok / 75_000.0) - actual_sec

    if ev == "command_reflex":
        if saved_tok <= 0:
            return -actual_sec
        if mode == "spinal_short_circuit":
            # Conservative ~1.2 avoided prep tool round-trips (~6.0s)
            return 6.0 + (saved_tok / 75_000.0) - actual_sec
        if mode == "pretool_git_rewrite":
            # Probabilistic expected retry avoidance (~0.5s expected time saved)
            return 0.5 + (saved_tok / 75_000.0) - actual_sec
        # preinv_directive / spinal_failthrough: bare imperatives avoid ~1 exploratory turn (~2.5s);
        # user-named `body <cmd>` nudges save ~0.8s of flag/contract overhead
        base_sec = 2.5 if saved_tok >= 3_500 else 0.8
        return base_sec + (saved_tok / 75_000.0) - actual_sec

    if ev == "vascular_block":
        if mode == "hard_deny":
            return (saved_tok / 75_000.0) - 1.5 - actual_sec
        return (saved_tok / 75_000.0) - actual_sec

    if saved_tok > 0:
        return (saved_tok / 75_000.0) - actual_sec
    return -actual_sec


class ShieldTelemetryLedger:
    """
    Sub-2ms, thread-safe SQLite WAL ledger for Organ 05 Blood-Brain Barrier savings events.
    """

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self._explicit_db_path = db_path is not None or bool(os.environ.get("WBC_SHIELD_DB_PATH", "").strip())
        self.db_path = Path(os.path.expanduser(str(db_path))) if db_path else get_default_shield_db_path()
        self._initialized = False

    def _is_default_prod_user_db(self) -> bool:
        """True only when reading the default personal `~/.whole_body_compute/dot_memory.db` (never test `tmp_path`)."""
        if self._explicit_db_path:
            return False
        if os.environ.get("WBC_DATA_DIR", "").strip():
            return False
        db_s = str(self.db_path).replace("\\", "/")
        if db_s == ":memory:" or _is_ephemeral_temp_target(db_s) or "pytest" in db_s.lower():
            return False
        return True

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=10000;")
        except Exception:
            pass
        conn.row_factory = sqlite3.Row
        return conn

    def ensure_schema(self) -> None:
        db_str = str(self.db_path)
        if db_str != ":memory:":
            try:
                st = os.stat(db_str)
                if _SCHEMA_READY_DBS.get(db_str) == st.st_ino and st.st_size > 0:
                    self._initialized = True
                    return
            except OSError:
                pass
        elif self._initialized:
            return
        with _LEDGER_LOCK:
            conn = self._connect()
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS shield_savings_ledger (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        bot_name TEXT NOT NULL,
                        session_id TEXT NOT NULL DEFAULT '',
                        model_name TEXT NOT NULL DEFAULT '',
                        event_type TEXT NOT NULL,
                        action_mode TEXT NOT NULL DEFAULT '',
                        target TEXT NOT NULL,
                        raw_tokens_est INTEGER NOT NULL DEFAULT 0,
                        compressed_tokens_est INTEGER NOT NULL DEFAULT 0,
                        tokens_saved INTEGER NOT NULL DEFAULT 0,
                        secrets_blocked INTEGER NOT NULL DEFAULT 0,
                        latency_ms REAL NOT NULL DEFAULT 0.0
                    );
                    """
                )
                for col_def in (
                    "session_id TEXT NOT NULL DEFAULT ''",
                    "model_name TEXT NOT NULL DEFAULT ''",
                    "action_mode TEXT NOT NULL DEFAULT ''",
                ):
                    try:
                        conn.execute(f"ALTER TABLE shield_savings_ledger ADD COLUMN {col_def};")
                    except sqlite3.OperationalError:
                        pass
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_shield_ledger_ts ON shield_savings_ledger(timestamp DESC);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_shield_ledger_bot ON shield_savings_ledger(bot_name);"
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS proxy_finops_ledger (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        client_slug TEXT NOT NULL DEFAULT 'untagged',
                        workflow_slug TEXT NOT NULL DEFAULT '',
                        provider TEXT NOT NULL DEFAULT 'anthropic',
                        requested_model TEXT NOT NULL DEFAULT '',
                        routed_model TEXT NOT NULL DEFAULT '',
                        route_mode TEXT NOT NULL DEFAULT 'passthrough',
                        cache_breakpoints_injected INTEGER NOT NULL DEFAULT 0,
                        uncached_input_tokens INTEGER NOT NULL DEFAULT 0,
                        cache_write_tokens INTEGER NOT NULL DEFAULT 0,
                        cache_read_tokens INTEGER NOT NULL DEFAULT 0,
                        output_tokens INTEGER NOT NULL DEFAULT 0,
                        baseline_usd REAL NOT NULL DEFAULT 0.0,
                        actual_usd REAL NOT NULL DEFAULT 0.0,
                        net_usd_saved REAL NOT NULL DEFAULT 0.0,
                        missed_usd_community_cap REAL NOT NULL DEFAULT 0.0,
                        secrets_blocked INTEGER NOT NULL DEFAULT 0,
                        slop_healed INTEGER NOT NULL DEFAULT 0,
                        ttfb_ms REAL NOT NULL DEFAULT 0.0,
                        latency_ms REAL NOT NULL DEFAULT 0.0,
                        status_code INTEGER NOT NULL DEFAULT 200
                    );
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_proxy_ledger_ts ON proxy_finops_ledger(timestamp DESC);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_proxy_ledger_client ON proxy_finops_ledger(client_slug);"
                )
                conn.commit()
                self._initialized = True
                if db_str != ":memory:":
                    try:
                        _SCHEMA_READY_DBS[db_str] = os.stat(db_str).st_ino
                    except OSError:
                        pass
            finally:
                conn.close()

    def record_event(
        self,
        event_type: str,
        target: str,
        raw_tokens_est: int = 0,
        compressed_tokens_est: int = 0,
        tokens_saved: Optional[int] = None,
        secrets_blocked: int = 0,
        latency_ms: float = 0.0,
        bot_name: Optional[str] = None,
        session_id: Optional[str] = None,
        model_name: Optional[str] = None,
        action_mode: str = "",
        payload: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> bool:
        """
        Insert a single shield savings event into SQLite WAL. Never raises exceptions.
        Diverts OS scratch/temp benchmark paths away from production home DB unless explicitly targeted.
        # [direct-edit]
        """
        try:
            if not self._explicit_db_path and _is_ephemeral_temp_target(str(target or "")):
                ephemeral_ledger = ShieldTelemetryLedger(db_path="/tmp/wbc_ephemeral_shield_ledger.db")
                return ephemeral_ledger.record_event(
                    event_type=event_type,
                    target=target,
                    raw_tokens_est=raw_tokens_est,
                    compressed_tokens_est=compressed_tokens_est,
                    tokens_saved=tokens_saved,
                    secrets_blocked=secrets_blocked,
                    latency_ms=latency_ms,
                    bot_name=bot_name,
                    session_id=session_id,
                    model_name=model_name,
                    action_mode=action_mode,
                    payload=payload,
                    timestamp=timestamp,
                )

            self.ensure_schema()
            raw_tok = max(0, int(raw_tokens_est or 0))
            comp_tok = max(0, int(compressed_tokens_est or 0))
            if tokens_saved is None:
                saved_tok = max(0, raw_tok - comp_tok)
            else:
                saved_tok = max(0, int(tokens_saved))
            sec_cnt = max(0, int(secrets_blocked or 0))
            lat_val = round(max(0.0, float(latency_ms or 0.0)), 3)
            resolved_bot = (
                BOT_ALIASES.get(bot_name.strip().lower(), bot_name.strip().lower())
                if bot_name
                else detect_bot_name(payload)
            )
            resolved_session = (
                session_id or detect_active_session_id(payload, bot_name=resolved_bot) or "default-session"
            ).strip()
            if model_name and model_name.strip():
                resolved_model = model_name.strip()
            else:
                resolved_model, _ = detect_active_model_and_rate(payload=payload, session_id=resolved_session)
            clean_target = _normalize_target_label(str(target or "unknown"))
            clean_mode = str(action_mode or "").strip()[:40]
            ts = float(timestamp if timestamp is not None else time.time())

            with _LEDGER_LOCK:
                conn = self._connect()
                try:
                    conn.execute(
                        """
                        INSERT INTO shield_savings_ledger (
                            timestamp, bot_name, session_id, model_name, event_type, action_mode,
                            target, raw_tokens_est, compressed_tokens_est, tokens_saved,
                            secrets_blocked, latency_ms
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            ts,
                            resolved_bot,
                            resolved_session,
                            resolved_model,
                            str(event_type or "ast_slice_gate"),
                            clean_mode,
                            clean_target,
                            raw_tok,
                            comp_tok,
                            saved_tok,
                            sec_cnt,
                            lat_val,
                        ),
                    )
                    conn.commit()
                    db_key = str(self.db_path)
                    _RECENT_SLICE_MAX_TS_CACHE.pop(db_key, None)
                    _DEFAULT_DB_NEGATIVE_UNTIL.pop(db_key, None)
                finally:
                    conn.close()
            return True
        except Exception:
            return False

    def record_ast_reread_if_recent(
        self,
        target: str,
        *,
        bot_name: Optional[str] = None,
        session_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        raw_tokens_est: int = 0,
        window_sec: float = 300.0,
        timestamp: Optional[float] = None,
    ) -> bool:
        """
        Log an `ast_slice_miss` event if the same `(session_id, target)` was recently
        sliced by `ast_slice_gate` within `window_sec` and has not already logged a miss
        for that slice.
        """
        try:
            db_str = str(self.db_path)
            ts = float(timestamp if timestamp is not None else time.time())
            cutoff = ts - max(1.0, float(window_sec))
            now_mono = time.monotonic()
            if db_str != ":memory:" and timestamp is None:
                neg_entry = _DEFAULT_DB_NEGATIVE_UNTIL.get(db_str)
                if neg_entry is not None and now_mono < neg_entry[0] and neg_entry[1] < cutoff:
                    return False
            db_mtime_ns = 0
            wal_mtime_ns = 0
            if db_str != ":memory:":
                try:
                    db_mtime_ns = os.stat(db_str).st_mtime_ns
                except OSError:
                    if timestamp is None:
                        _DEFAULT_DB_NEGATIVE_UNTIL[db_str] = (now_mono + 0.5, -1.0)
                    return False
                try:
                    wal_mtime_ns = os.stat(f"{db_str}-wal").st_mtime_ns
                except OSError:
                    wal_mtime_ns = 0
                cached_max = _RECENT_SLICE_MAX_TS_CACHE.get(db_str)
                if (
                    cached_max is not None
                    and cached_max[0] == db_mtime_ns
                    and cached_max[1] == wal_mtime_ns
                    and cached_max[2] < cutoff
                ):
                    if timestamp is None:
                        _DEFAULT_DB_NEGATIVE_UNTIL[db_str] = (now_mono + 0.5, cached_max[2])
                    return False

            self.ensure_schema()
            resolved_bot = (
                BOT_ALIASES.get(bot_name.strip().lower(), bot_name.strip().lower())
                if bot_name
                else detect_bot_name(payload)
            )
            resolved_session = (
                session_id or detect_active_session_id(payload, bot_name=resolved_bot) or "default-session"
            ).strip()
            clean_target = _normalize_target_label(str(target or ""))
            if not clean_target or clean_target == "unknown":
                return False

            with _LEDGER_LOCK:
                conn = self._connect()
                try:
                    if db_str != ":memory:":
                        g_row = conn.execute(
                            "SELECT MAX(timestamp) FROM shield_savings_ledger WHERE event_type = 'ast_slice_gate';"
                        ).fetchone()
                        g_max = float(g_row[0]) if g_row and g_row[0] is not None else -1.0
                        _RECENT_SLICE_MAX_TS_CACHE[db_str] = (db_mtime_ns, wal_mtime_ns, g_max)
                        if g_max < cutoff:
                            return False
                    base_only = Path(clean_target).name if "/" in clean_target else clean_target
                    base_like = base_only.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                    last_slice = conn.execute(
                        """
                        SELECT MAX(timestamp) FROM shield_savings_ledger
                        WHERE event_type = 'ast_slice_gate'
                          AND action_mode NOT IN ('fast_bounded_slice', 'shell_heal', 'gate_allow', 'hard_deny')
                          AND session_id = ?
                          AND (target = ? OR target = ? OR target LIKE ? ESCAPE '\\')
                          AND timestamp >= ?
                          AND timestamp <= ?;
                        """,
                        (resolved_session, clean_target, base_only, f"%/{base_like}", cutoff, ts),
                    ).fetchone()
                    last_slice_ts = float(last_slice[0]) if last_slice and last_slice[0] is not None else None
                    if last_slice_ts is None:
                        return False

                    already_missed = conn.execute(
                        """
                        SELECT 1 FROM shield_savings_ledger
                        WHERE event_type = 'ast_slice_miss'
                          AND session_id = ?
                          AND (target = ? OR target = ? OR target LIKE ? ESCAPE '\\')
                          AND timestamp >= ?
                        LIMIT 1;
                        """,
                        (resolved_session, clean_target, base_only, f"%/{base_like}", last_slice_ts),
                    ).fetchone()
                    if already_missed is not None:
                        return False

                    resolved_model, _ = detect_active_model_and_rate(payload=payload, session_id=resolved_session)
                    conn.execute(
                        """
                        INSERT INTO shield_savings_ledger (
                            timestamp, bot_name, session_id, model_name, event_type, action_mode,
                            target, raw_tokens_est, compressed_tokens_est, tokens_saved,
                            secrets_blocked, latency_ms
                        ) VALUES (?, ?, ?, ?, 'ast_slice_miss', 'reread_after_slice', ?, ?, 0, 0, 0, 0.0);
                        """,
                        (
                            ts,
                            resolved_bot,
                            resolved_session,
                            resolved_model,
                            clean_target,
                            max(0, int(raw_tokens_est or 0)),
                        ),
                    )
                    conn.commit()
                    return True
                finally:
                    conn.close()
        except Exception:
            return False

    def reset(self) -> int:
        """Delete all rows in `shield_savings_ledger` and return the deleted count."""
        try:
            if str(self.db_path) != ":memory:" and not Path(self.db_path).is_file():
                return False
            self.ensure_schema()
            with _LEDGER_LOCK:
                conn = self._connect()
                try:
                    cur = conn.execute("DELETE FROM shield_savings_ledger;")
                    deleted = cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else 0
                    conn.commit()
                    return deleted
                finally:
                    conn.close()
        except Exception:
            return 0

    def _resolve_fast_path_audit_log(self) -> Optional[Path]:
        """Resolve `immune_audit.jsonl` path for reconciling POSIX `sh` fast-path and bounded slice events."""
        env_p = (os.environ.get("VOICEFI_IMMUNE_AUDIT_PATH") or os.environ.get("WBC_IMMUNE_AUDIT_PATH") or "").strip()
        if env_p:
            return Path(os.path.expanduser(env_p))
        if (
            os.environ.get("PYTEST_CURRENT_TEST")
            or os.environ.get("VOICEFI_STRICT_TEST_HOOK") == "1"
            or os.environ.get("WBC_BENCHMARK_MODE") == "1"
            or os.environ.get("WBC_ARENA_MODE") == "1"
        ):
            return None
        if self._explicit_db_path:
            # Only auto-reconcile sibling audit logs when explicitly placed in the same dir or via env var
            sibling = self.db_path.parent / "immune_audit.jsonl"
            return sibling if sibling.is_file() else None
        env_data = os.environ.get("WBC_DATA_DIR", "").strip()
        if env_data:
            return Path(os.path.expanduser(env_data)) / "immune_audit.jsonl"
        home = Path.home()
        for cand_dir in (self.db_path.parent, home / ".whole_body_compute", home / ".voicefi"):
            cand = cand_dir / "immune_audit.jsonl"
            if cand.is_file():
                return cand
        return None

    def _reconcile_fast_path_auto_slices(self) -> int:
        """
        Sync POSIX `sh` fast-path `read_auto_sliced` and `bounded_slice_read` records from
        `immune_audit.jsonl` into `shield_savings_ledger` (idempotent, zero runtime overhead).
        """
        audit_file = self._resolve_fast_path_audit_log()
        if audit_file is None or not audit_file.is_file():
            return 0
        try:
            raw_lines = audit_file.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return 0

        allow_bounded_slice = os.environ.get("WBC_RECONCILE_BOUNDED_SLICES") == "1"

        candidates: List[Tuple[float, str, str, str, str, str, int, int, int, float]] = []
        for line in raw_lines:
            if '"fast_path":true' not in line and '"fast_path": true' not in line:
                continue
            has_auto_slice = '"read_auto_sliced":true' in line or '"read_auto_sliced": true' in line
            has_bounded_slice = allow_bounded_slice and (
                '"bounded_slice_read":true' in line or '"bounded_slice_read": true' in line
            )
            if not (has_auto_slice or has_bounded_slice):
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if not ev.get("fast_path"):
                continue
            is_auto = bool(ev.get("read_auto_sliced"))
            is_bounded = allow_bounded_slice and bool(ev.get("bounded_slice_read"))
            if not (is_auto or is_bounded):
                continue
            saved_tok = max(0, int(ev.get("direct_tokens_saved") or 0))
            if saved_tok <= 0:
                continue
            ts_val = float(ev.get("timestamp") or 0.0)
            tool_nm = str(ev.get("tool") or "read_file")
            bot_nm = "claude-code" if tool_nm == "Read" else "antigravity"
            model_nm = "Claude Sonnet" if bot_nm == "claude-code" else "Gemini 4 Carbon"
            raw_tgt = str(ev.get("target") or "unknown")
            if (not self._explicit_db_path or not os.environ.get("PYTEST_CURRENT_TEST")) and _is_ephemeral_temp_target(raw_tgt):
                continue
            norm_tgt = _normalize_target_label(raw_tgt)
            comp_tok = 1_600  # 160-line sliced window + AST symbol outline
            ceil_tok = get_harness_file_token_ceiling(bot_name=bot_nm)
            raw_tok = min(ceil_tok, saved_tok + comp_tok)
            saved_tok = max(0, raw_tok - comp_tok)
            if saved_tok <= 0:
                continue
            lat_ms = round(max(0.0, float(ev.get("latency_ms") or 4.2)), 3)
            sess_id = str(ev.get("conversation_id") or "").strip()
            if not sess_id or sess_id == "global":
                sess_id = "default-session"
            act_mode = "sh_auto_slice" if is_auto else "fast_bounded_slice"
            candidates.append((ts_val, bot_nm, sess_id, model_nm, act_mode, norm_tgt, raw_tok, comp_tok, saved_tok, lat_ms))

        if not candidates:
            return 0

        self.ensure_schema()
        inserted = 0
        with _LEDGER_LOCK:
            conn = self._connect()
            try:
                existing_rows = conn.execute(
                    "SELECT timestamp, bot_name, target FROM shield_savings_ledger WHERE event_type = 'ast_slice_gate';"
                ).fetchall()
                existing_exact = {
                    (round(float(r[0]), 1), str(r[1]), _normalize_target_label(str(r[2])))
                    for r in existing_rows
                }
                existing_sec_base = {
                    (int(float(r[0])), str(r[1]), Path(str(r[2])).name)
                    for r in existing_rows
                }
                for ts_val, bot_nm, sess_id, model_nm, act_mode, norm_tgt, raw_tok, comp_tok, saved_tok, lat_ms in candidates:
                    key = (round(ts_val, 1), bot_nm, norm_tgt)
                    sec_key = (int(ts_val), bot_nm, Path(norm_tgt).name)
                    if key in existing_exact or sec_key in existing_sec_base:
                        continue
                    conn.execute(
                        """
                        INSERT INTO shield_savings_ledger (
                            timestamp, bot_name, session_id, model_name, event_type, action_mode,
                            target, raw_tokens_est, compressed_tokens_est, tokens_saved,
                            secrets_blocked, latency_ms
                        ) VALUES (?, ?, ?, ?, 'ast_slice_gate', ?, ?, ?, ?, ?, 0, ?);
                        """,
                        (ts_val, bot_nm, sess_id, model_nm, act_mode, norm_tgt, raw_tok, comp_tok, saved_tok, lat_ms),
                    )
                    existing_exact.add(key)
                    existing_sec_base.add(sec_key)
                    inserted += 1
                if inserted > 0:
                    conn.commit()
            finally:
                conn.close()
        return inserted

    def record_proxy_call(
        self,
        *,
        client_slug: str = "untagged",
        workflow_slug: str = "",
        provider: str = "anthropic",
        requested_model: str = "",
        routed_model: str = "",
        route_mode: str = "passthrough",
        cache_breakpoints_injected: int = 0,
        uncached_input_tokens: int = 0,
        cache_write_tokens: int = 0,
        cache_read_tokens: int = 0,
        output_tokens: int = 0,
        baseline_usd: float = 0.0,
        actual_usd: float = 0.0,
        net_usd_saved: float = 0.0,
        missed_usd_community_cap: float = 0.0,
        secrets_blocked: int = 0,
        slop_healed: int = 0,
        ttfb_ms: float = 0.0,
        latency_ms: float = 0.0,
        status_code: int = 200,
        equivalent_input_tokens_saved: int = 0,
        timestamp: Optional[float] = None,
    ) -> bool:
        """
        Insert a FinOps API proxy call receipt into `proxy_finops_ledger` and dual-write
        net savings into `shield_savings_ledger` so `body savings` reflects proxy ROI.
        Never raises exceptions.
        """
        try:
            self.ensure_schema()
            now_ts = float(timestamp) if timestamp is not None else time.time()
            c_slug = (client_slug or "untagged").strip() or "untagged"
            w_slug = (workflow_slug or "").strip()
            prov = (provider or "anthropic").strip().lower() or "anthropic"
            req_m = (requested_model or "").strip()
            rtd_m = (routed_model or req_m).strip()
            r_mode = (route_mode or "passthrough").strip()
            bp_inj = max(0, int(cache_breakpoints_injected or 0))
            u_in = max(0, int(uncached_input_tokens or 0))
            cw_tok = max(0, int(cache_write_tokens or 0))
            cr_tok = max(0, int(cache_read_tokens or 0))
            out_tok = max(0, int(output_tokens or 0))
            base_u = round(max(0.0, float(baseline_usd or 0.0)), 6)
            act_u = round(max(0.0, float(actual_usd or 0.0)), 6)
            net_u = round(float(net_usd_saved or 0.0), 6)
            missed_u = round(max(0.0, float(missed_usd_community_cap or 0.0)), 6)
            sec_blk = max(0, int(secrets_blocked or 0))
            slop_h = max(0, int(slop_healed or 0))
            ttfb = round(max(0.0, float(ttfb_ms or 0.0)), 3)
            lat = round(max(0.0, float(latency_ms or 0.0)), 3)
            code = int(status_code or 200)

            with _LEDGER_LOCK:
                conn = self._connect()
                try:
                    conn.execute(
                        """
                        INSERT INTO proxy_finops_ledger (
                            timestamp, client_slug, workflow_slug, provider,
                            requested_model, routed_model, route_mode,
                            cache_breakpoints_injected, uncached_input_tokens,
                            cache_write_tokens, cache_read_tokens, output_tokens,
                            baseline_usd, actual_usd, net_usd_saved,
                            missed_usd_community_cap, secrets_blocked, slop_healed,
                            ttfb_ms, latency_ms, status_code
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            now_ts, c_slug, w_slug, prov, req_m, rtd_m, r_mode,
                            bp_inj, u_in, cw_tok, cr_tok, out_tok,
                            base_u, act_u, net_u, missed_u, sec_blk, slop_h,
                            ttfb, lat, code,
                        ),
                    )
                    conn.commit()
                finally:
                    conn.close()

            eq_saved = max(0, int(equivalent_input_tokens_saved or 0))
            if eq_saved == 0 and cr_tok > 0:
                eq_saved = int(round(cr_tok * 0.90))
            if eq_saved > 0 or sec_blk > 0:
                raw_est = u_in + cw_tok + cr_tok + out_tok
                comp_est = max(0, raw_est - eq_saved)
                self.record_event(
                    event_type="proxy_prompt_cache",
                    target=f"{c_slug}/{w_slug}" if w_slug else c_slug,
                    raw_tokens_est=raw_est,
                    compressed_tokens_est=comp_est,
                    tokens_saved=eq_saved,
                    secrets_blocked=sec_blk,
                    latency_ms=lat,
                    bot_name="body-proxy",
                    session_id=c_slug,
                    model_name=rtd_m or req_m,
                    action_mode=r_mode,
                    timestamp=now_ts,
                )
            return True
        except Exception:
            return False

    def get_proxy_client_summary(
        self,
        window: Optional[str] = None,
        client_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate `proxy_finops_ledger` totals, today's cached call count (for Community
        250/day fail-open metering), and per-client attribution breakdown.
        """
        self.ensure_schema()
        window_label, since_ts = _resolve_window_since_ts(window)
        clauses: List[str] = []
        params_list: List[Any] = []
        if since_ts is not None:
            clauses.append("timestamp >= ?")
            params_list.append(float(since_ts))
        if client_filter:
            clauses.append("client_slug = ?")
            params_list.append(client_filter.strip())

        where_clause = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        now_dt = datetime.fromtimestamp(time.time(), tz=timezone.utc)
        utc_day_start = datetime(now_dt.year, now_dt.month, now_dt.day, tzinfo=timezone.utc).timestamp()

        with _LEDGER_LOCK:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"""
                    SELECT
                        client_slug,
                        COUNT(*) AS calls,
                        SUM(CASE WHEN cache_breakpoints_injected > 0 THEN 1 ELSE 0 END) AS cached_calls,
                        SUM(uncached_input_tokens) AS uncached_in,
                        SUM(cache_write_tokens) AS cache_write,
                        SUM(cache_read_tokens) AS cache_read,
                        SUM(output_tokens) AS output_tok,
                        SUM(baseline_usd) AS baseline_usd,
                        SUM(actual_usd) AS actual_usd,
                        SUM(net_usd_saved) AS net_usd_saved,
                        SUM(missed_usd_community_cap) AS missed_usd_cap,
                        SUM(secrets_blocked) AS secrets_blocked,
                        SUM(slop_healed) AS slop_healed
                    FROM proxy_finops_ledger
                    {where_clause}
                    GROUP BY client_slug
                    ORDER BY baseline_usd DESC, calls DESC;
                    """,
                    tuple(params_list),
                ).fetchall()

                today_row = conn.execute(
                    """
                    SELECT
                        COUNT(*) AS calls_today,
                        SUM(CASE WHEN cache_breakpoints_injected > 0 THEN 1 ELSE 0 END) AS cached_today,
                        SUM(net_usd_saved) AS saved_today,
                        SUM(missed_usd_community_cap) AS missed_today
                    FROM proxy_finops_ledger
                    WHERE timestamp >= ?;
                    """,
                    (utc_day_start,),
                ).fetchone()
            finally:
                conn.close()

        clients: List[Dict[str, Any]] = []
        total_calls = total_cached = total_sec = total_slop = 0
        total_base = total_act = total_saved = total_missed = 0.0
        for r in rows:
            c_item = {
                "client_slug": str(r[0] or "untagged"),
                "calls": int(r[1] or 0),
                "cached_calls": int(r[2] or 0),
                "uncached_input_tokens": int(r[3] or 0),
                "cache_write_tokens": int(r[4] or 0),
                "cache_read_tokens": int(r[5] or 0),
                "output_tokens": int(r[6] or 0),
                "baseline_usd": round(float(r[7] or 0.0), 4),
                "actual_usd": round(float(r[8] or 0.0), 4),
                "net_usd_saved": round(float(r[9] or 0.0), 4),
                "missed_usd_community_cap": round(float(r[10] or 0.0), 4),
                "secrets_blocked": int(r[11] or 0),
                "slop_healed": int(r[12] or 0),
            }
            clients.append(c_item)
            total_calls += c_item["calls"]
            total_cached += c_item["cached_calls"]
            total_base += c_item["baseline_usd"]
            total_act += c_item["actual_usd"]
            total_saved += c_item["net_usd_saved"]
            total_missed += c_item["missed_usd_community_cap"]
            total_sec += c_item["secrets_blocked"]
            total_slop += c_item["slop_healed"]

        return {
            "window": window_label,
            "total_calls": total_calls,
            "cached_calls": total_cached,
            "baseline_usd": round(total_base, 4),
            "actual_usd": round(total_act, 4),
            "net_usd_saved": round(total_saved, 4),
            "missed_usd_community_cap": round(total_missed, 4),
            "secrets_blocked": total_sec,
            "slop_healed": total_slop,
            "calls_today": int((today_row[0] if today_row else 0) or 0),
            "cached_calls_today": int((today_row[1] if today_row else 0) or 0),
            "saved_today_usd": round(float((today_row[2] if today_row else 0.0) or 0.0), 4),
            "missed_today_usd": round(float((today_row[3] if today_row else 0.0) or 0.0), 4),
            "clients": clients,
        }

    def get_summary(
        self,
        bot_filter: Optional[str] = None,
        price_per_m: Optional[float] = None,
        session_gap_seconds: float = 7200.0,
        window: Optional[str] = None,
        top_limit: int = 15,
        include_benchmarks: bool = False,
    ) -> Dict[str, Any]:
        """
        Query deduplicated counterfactual token compression, per-row model API USD saved,
        signed wall-clock time saved, Amdahl quota runway multiplier, and ranked leaderboards.
        # [direct-edit]
        """
        self.ensure_schema()
        try:
            self._reconcile_fast_path_auto_slices()
        except Exception:
            pass
        norm_bot = None
        if bot_filter:
            norm_bot = BOT_ALIASES.get(bot_filter.strip().lower(), bot_filter.strip().lower())

        window_label, since_ts = _resolve_window_since_ts(window)
        clauses: List[str] = []
        params_list: List[Any] = []
        if norm_bot:
            clauses.append("bot_name = ?")
            params_list.append(norm_bot)
        if since_ts is not None:
            clauses.append("timestamp >= ?")
            params_list.append(float(since_ts))

        where_clause = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params: tuple[Any, ...] = tuple(params_list)
        safe_top = max(1, min(100, int(top_limit or 15)))

        with _LEDGER_LOCK:
            conn = self._connect()
            try:
                chron_rows = conn.execute(
                    f"""
                    SELECT
                        id, timestamp, bot_name, session_id, model_name, event_type,
                        action_mode, target, raw_tokens_est, compressed_tokens_est,
                        tokens_saved, secrets_blocked, latency_ms
                    FROM shield_savings_ledger
                    {where_clause}
                    ORDER BY timestamp ASC, id ASC;
                    """,
                    params,
                ).fetchall()

                recent_rows = conn.execute(
                    f"""
                    SELECT
                        timestamp, bot_name, session_id, model_name, event_type,
                        action_mode, target, raw_tokens_est, compressed_tokens_est,
                        tokens_saved, secrets_blocked, latency_ms
                    FROM shield_savings_ledger
                    {where_clause}
                    ORDER BY id DESC
                    LIMIT 10;
                    """,
                    params,
                ).fetchall()
            finally:
                conn.close()

        if not include_benchmarks:
            exclude_tmp = self._is_default_prod_user_db()
            chron_rows = [
                cr
                for cr in chron_rows
                if not _is_benchmark_or_ephemeral_row(
                    str(cr["target"] or ""),
                    str(cr["session_id"] or ""),
                    exclude_ephemeral_temp=exclude_tmp,
                )
            ]
            recent_rows = list(reversed(chron_rows[-10:]))

        detected_model, detected_rate = detect_active_model_and_rate()
        if recent_rows:
            last_m = str(recent_rows[0]["model_name"] or "").strip()
            if last_m and any(
                k in last_m.lower()
                for k in ("gemini", "claude", "gpt", "grok", "sonnet", "opus", "haiku", "flash", "hydrogen", "neon", "carbon", "argon", "o1", "o3")
            ):
                detected_model = last_m
                detected_rate = _lookup_model_rate(last_m)

        effective_price_per_m = float(price_per_m) if price_per_m is not None else detected_rate

        total_events = len(chron_rows)
        raw_tokens_blocked = 0
        compressed_tokens_sent = 0
        direct_tokens_saved = 0
        secrets_blocked = 0
        all_lats: List[float] = []

        reflex_lats: List[float] = []
        scout_lats: List[float] = []
        eyes_lats: List[float] = []
        hands_lats: List[float] = []
        shield_lats: List[float] = []
        eyes_tokens_saved = 0
        hands_tokens_saved = 0
        shield_tokens_saved = 0
        eyes_time_sec = 0.0
        hands_time_sec = 0.0
        shield_time_sec = 0.0
        total_time_saved_sec = 0.0
        total_direct_usd = 0.0
        total_output_tokens_saved = 0
        total_output_usd = 0.0
        ast_slice_events = 0
        ast_slice_misses = 0

        # Per-(bot_name, session_id, target) high-water-mark deduplication for file/log reads & repeated reflex prompts
        dedup_credits: Dict[Tuple[str, str, str], int] = {}
        bare_seen_credits: Dict[Tuple[str, str, str], int] = {}
        reflex_last_ts: Dict[Tuple[str, str, str], float] = {}
        catchall_last_ts: Dict[Tuple[str, str], float] = {}
        catchall_epoch: Dict[Tuple[str, str], int] = {}
        observed_rates: set[float] = set()

        bot_agg: Dict[str, Dict[str, Any]] = {}
        event_agg: Dict[str, Dict[str, Any]] = {}
        target_agg: Dict[str, Dict[str, Any]] = {}
        session_meta_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        # (ts_val, eff_saved, row_rate, row_cache_mult, is_compounded)
        # Note: `command_reflex` represents transient turn avoidance, NOT permanent transcript payload reduction,
        # so it contributes 1x to cumulative_tokens_saved and 0 to future-turn re-ingestion compounding.
        session_groups: Dict[Tuple[str, str, int], List[Tuple[float, int, float, float, bool]]] = {}

        for cr in chron_rows:
            b_name = str(cr["bot_name"])
            s_id = str(cr["session_id"] or "default-session").strip() or "default-session"
            m_name = str(cr["model_name"] or "").strip()
            act_mode = str(cr["action_mode"] or "").strip()
            ts_val = float(cr["timestamp"] or 0.0)
            raw_db = max(0, int(cr["raw_tokens_est"] or 0))
            comp_db = max(0, int(cr["compressed_tokens_est"] or 0))
            saved_db = max(0, int(cr["tokens_saved"] or 0))
            sec_db = max(0, int(cr["secrets_blocked"] or 0))
            ev_t = str(cr["event_type"] or "")
            tgt_name = _normalize_target_label(str(cr["target"] or ""))
            lat_v = float(cr["latency_ms"] or 0.0)

            if ev_t == "ast_slice_miss":
                ast_slice_misses += 1
                continue
            if ev_t == "ast_slice_gate" and act_mode not in ("fast_bounded_slice", "shell_heal", "gate_allow", "hard_deny"):
                ast_slice_events += 1

            is_catchall_sess = s_id in ("cli-standalone", "default-session", "local", "global", "")
            if is_catchall_sess and session_gap_seconds > 0:
                prev_sess_ts = catchall_last_ts.get((b_name, s_id))
                cur_ep = catchall_epoch.get((b_name, s_id), 0)
                if prev_sess_ts is not None and (ts_val - prev_sess_ts) > session_gap_seconds:
                    cur_ep += 1
                    catchall_epoch[(b_name, s_id)] = cur_ep
                catchall_last_ts[(b_name, s_id)] = ts_val
            else:
                cur_ep = 0
            eff_sess_key = f"{s_id}#{cur_ep}" if is_catchall_sess and cur_ep > 0 else s_id

            if ev_t in _DEDUP_FILE_EVENT_TYPES and tgt_name:
                d_key = (b_name, eff_sess_key, tgt_name)
                b_only = Path(tgt_name).name if "/" in tgt_name else tgt_name
                b_key = (b_name, eff_sess_key, b_only)
                if "/" not in tgt_name:
                    prev_credited = max(dedup_credits.get(d_key, 0), bare_seen_credits.get(b_key, 0))
                else:
                    # Unify only when a bare filename without `/` was also recorded in the same session
                    prev_credited = max(dedup_credits.get(d_key, 0), dedup_credits.get(b_key, 0))
                ceiling_saved = max(prev_credited, saved_db)
                eff_saved = min(saved_db, max(0, ceiling_saved - prev_credited))
                new_hwm = prev_credited + eff_saved
                dedup_credits[d_key] = new_hwm
                bare_seen_credits[b_key] = max(bare_seen_credits.get(b_key, 0), new_hwm)
                if "/" in tgt_name and b_key in dedup_credits:
                    dedup_credits[b_key] = max(dedup_credits[b_key], new_hwm)
                eff_raw = comp_db + eff_saved
            elif ev_t == "command_reflex" and tgt_name:
                r_key = (b_name, eff_sess_key, tgt_name)
                prev_r_ts = reflex_last_ts.get(r_key)
                reflex_last_ts[r_key] = ts_val
                if tgt_name == "body flush":
                    prev_credited = dedup_credits.get(r_key, 0)
                    ceiling_saved = max(prev_credited, saved_db)
                    eff_saved = min(saved_db, max(0, ceiling_saved - prev_credited))
                    dedup_credits[r_key] = prev_credited + eff_saved
                    eff_raw = comp_db + eff_saved
                elif prev_r_ts is not None and 0.0 <= (ts_val - prev_r_ts) < 120.0:
                    eff_saved = 0
                    eff_raw = comp_db
                else:
                    eff_saved = saved_db
                    eff_raw = (comp_db + eff_saved) if raw_db >= comp_db else comp_db
            else:
                eff_saved = saved_db
                eff_raw = (comp_db + eff_saved) if raw_db >= comp_db else comp_db

            in_p, out_p, row_cache_mult = _lookup_model_pricing(
                m_name or detected_model,
                event_ts=ts_val,
            )
            if price_per_m is not None:
                row_rate = float(price_per_m)
                row_out_rate = float(price_per_m) * 5.0
                row_cache_mult = 0.10
            else:
                row_rate = in_p
                row_out_rate = out_p
            observed_rates.add(round(row_rate, 4))
            row_direct_usd = (eff_saved / 1_000_000.0) * row_rate
            row_out_tok = _estimate_avoided_output_tokens(ev_t, eff_saved, action_mode=act_mode)
            row_output_usd = (row_out_tok / 1_000_000.0) * row_out_rate
            ev_time_sec = _estimate_event_time_saved_sec(
                ev_t,
                eff_raw,
                eff_saved,
                lat_v,
                action_mode=act_mode,
            )

            raw_tokens_blocked += eff_raw
            compressed_tokens_sent += comp_db
            direct_tokens_saved += eff_saved
            total_output_tokens_saved += row_out_tok
            secrets_blocked += sec_db
            all_lats.append(lat_v)
            total_direct_usd += row_direct_usd
            total_output_usd += row_output_usd
            total_time_saved_sec += ev_time_sec

            b_st = bot_agg.setdefault(
                b_name,
                {
                    "events": 0,
                    "raw_tokens": 0,
                    "compressed_tokens": 0,
                    "tokens_saved": 0,
                    "output_tokens_saved": 0,
                    "secrets_blocked": 0,
                    "lats": [],
                    "time_sec": 0.0,
                    "direct_usd": 0.0,
                    "output_usd": 0.0,
                },
            )
            b_st["events"] += 1
            b_st["raw_tokens"] += eff_raw
            b_st["compressed_tokens"] += comp_db
            b_st["tokens_saved"] += eff_saved
            b_st["output_tokens_saved"] += row_out_tok
            b_st["secrets_blocked"] += sec_db
            b_st["lats"].append(lat_v)
            b_st["time_sec"] += ev_time_sec
            b_st["direct_usd"] += row_direct_usd
            b_st["output_usd"] += row_output_usd

            e_st = event_agg.setdefault(
                ev_t,
                {
                    "events": 0,
                    "raw_tokens": 0,
                    "compressed_tokens": 0,
                    "tokens_saved": 0,
                    "output_tokens_saved": 0,
                    "secrets_blocked": 0,
                    "lats": [],
                    "time_sec": 0.0,
                    "direct_usd": 0.0,
                    "output_usd": 0.0,
                },
            )
            e_st["events"] += 1
            e_st["raw_tokens"] += eff_raw
            e_st["compressed_tokens"] += comp_db
            e_st["tokens_saved"] += eff_saved
            e_st["output_tokens_saved"] += row_out_tok
            e_st["secrets_blocked"] += sec_db
            e_st["lats"].append(lat_v)
            e_st["time_sec"] += ev_time_sec
            e_st["direct_usd"] += row_direct_usd
            e_st["output_usd"] += row_output_usd

            t_st = target_agg.setdefault(
                tgt_name,
                {
                    "gates": [],
                    "events": 0,
                    "raw_tokens": 0,
                    "compressed_tokens": 0,
                    "tokens_saved": 0,
                    "secrets_blocked": 0,
                    "lats": [],
                    "time_sec": 0.0,
                    "direct_usd": 0.0,
                },
            )
            if ev_t and ev_t not in t_st["gates"]:
                t_st["gates"].append(ev_t)
            t_st["events"] += 1
            t_st["raw_tokens"] += eff_raw
            t_st["compressed_tokens"] += comp_db
            t_st["tokens_saved"] += eff_saved
            t_st["secrets_blocked"] += sec_db
            t_st["lats"].append(lat_v)
            t_st["time_sec"] += ev_time_sec
            t_st["direct_usd"] += row_direct_usd

            s_meta = session_meta_map.setdefault(
                (b_name, s_id),
                {
                    "events": 0,
                    "tokens_saved": 0,
                    "time_saved_sec": 0.0,
                    "direct_usd": 0.0,
                    "model_name": m_name,
                },
            )
            s_meta["events"] += 1
            s_meta["tokens_saved"] += eff_saved
            s_meta["time_saved_sec"] += ev_time_sec
            s_meta["direct_usd"] += row_direct_usd
            if m_name:
                s_meta["model_name"] = m_name

            if ev_t in ("local_metal_scout", "screen_local_digest"):
                scout_lats.append(lat_v)
            elif ev_t in ("ast_slice_gate", "secret_redaction", "vascular_block", "crash_triage_fusion") or (
                ev_t == "command_reflex" and act_mode != "spinal_short_circuit"
            ):
                reflex_lats.append(lat_v)

            if ev_t in ("eyes_foveal_dom", "screen_local_digest"):
                eyes_lats.append(lat_v)
                eyes_tokens_saved += eff_saved
                eyes_time_sec += ev_time_sec
            elif ev_t in ("hands_ax_act", "hands_tactile", "hands_apfs_sandbox"):
                hands_lats.append(lat_v)
                hands_tokens_saved += eff_saved
                hands_time_sec += ev_time_sec
            else:
                shield_lats.append(lat_v)
                shield_tokens_saved += eff_saved
                shield_time_sec += ev_time_sec

            session_groups.setdefault((b_name, s_id, cur_ep), []).append(
                (ts_val, eff_saved, row_rate, row_cache_mult, ev_t != "command_reflex")
            )

        avg_latency_ms = round(sum(all_lats) / len(all_lats), 2) if all_lats else 0.0
        hook_reflex_latency_ms = round(sum(reflex_lats) / len(reflex_lats), 2) if reflex_lats else 0.0
        local_scout_lifetime_avg_ms = round(sum(scout_lats) / len(scout_lats), 2) if scout_lats else 0.0
        recent_scout_lats = [
            float(r["latency_ms"])
            for r in recent_rows
            if str(r["event_type"]) in ("local_metal_scout", "screen_local_digest")
        ]
        if recent_scout_lats:
            recent_scout_window = recent_scout_lats[:3]
        else:
            recent_scout_window = scout_lats[-3:] if len(scout_lats) >= 3 else scout_lats
        local_scout_latency_ms = (
            round(sum(recent_scout_window) / len(recent_scout_window), 2)
            if recent_scout_window
            else 0.0
        )
        local_scout_events = len(scout_lats)
        scout_warm = is_scout_warm()

        bot_cumulative_map: Dict[str, int] = {}
        bot_reingest_usd_map: Dict[str, float] = {}
        bot_turns_map: Dict[str, int] = {}
        session_cum_map: Dict[Tuple[str, str], Tuple[int, int, float]] = {}
        cumulative_tokens_saved = 0
        total_reingest_usd = 0.0
        actual_turns_observed = 0

        for (b_name, s_id, _ep), ev_list in session_groups.items():
            transcript_turns = _count_transcript_user_turns(s_id, bot_name=b_name)
            avg_sess_rate = sum(r for _, _, r, _, _ in ev_list) / max(1, len(ev_list))
            avg_cache_mult = sum(cm for _, _, _, cm, _ in ev_list) / max(1, len(ev_list))
            # Coalesce multi-tool bursts (<25s) in real sessions or >5-event synthetic runs.
            # Track [compounded_tok, non_compounded_tok] per turn bucket so transient turn-avoidance
            # events (`command_reflex`) are never multiplied across subsequent turns.
            if transcript_turns is not None or s_id not in ("default-session", "local") or len(ev_list) > 5:
                turn_buckets: List[List[int]] = []
                last_bucket_ts = -1.0
                for ts_val, s_tok, _, _, is_comp in ev_list:
                    if not turn_buckets or (ts_val - last_bucket_ts) >= 25.0:
                        turn_buckets.append([s_tok, 0] if is_comp else [0, s_tok])
                        last_bucket_ts = ts_val
                    else:
                        turn_buckets[-1][0 if is_comp else 1] += s_tok
                if transcript_turns is not None and len(turn_buckets) > transcript_turns:
                    merged: List[List[int]] = [[0, 0] for _ in range(transcript_turns)]
                    for idx_b, (c_val, nc_val) in enumerate(turn_buckets):
                        target_idx = min(transcript_turns - 1, int(idx_b * transcript_turns / len(turn_buckets)))
                        merged[target_idx][0] += c_val
                        merged[target_idx][1] += nc_val
                    turn_buckets = merged
            else:
                turn_buckets = [[s_tok, 0] if is_comp else [0, s_tok] for _, s_tok, _, _, is_comp in ev_list]

            n_turns = len(turn_buckets)
            sess_cum = 0
            sess_direct = sum(c_k + nc_k for c_k, nc_k in turn_buckets)
            for idx_z, (c_k, nc_k) in enumerate(turn_buckets):
                sess_cum += (n_turns - idx_z) * c_k + nc_k

            sess_reingest_tok = max(0, sess_cum - sess_direct)
            sess_reingest_usd = (sess_reingest_tok / 1_000_000.0) * (avg_sess_rate * avg_cache_mult)

            prev_sc, prev_nt, prev_ru = session_cum_map.get((b_name, s_id), (0, 0, 0.0))
            session_cum_map[(b_name, s_id)] = (prev_sc + sess_cum, prev_nt + n_turns, prev_ru + sess_reingest_usd)
            bot_cumulative_map[b_name] = bot_cumulative_map.get(b_name, 0) + sess_cum
            bot_reingest_usd_map[b_name] = bot_reingest_usd_map.get(b_name, 0.0) + sess_reingest_usd
            bot_turns_map[b_name] = bot_turns_map.get(b_name, 0) + n_turns
            cumulative_tokens_saved += sess_cum
            total_reingest_usd += sess_reingest_usd
            actual_turns_observed += n_turns

        compression_pct = (
            round((direct_tokens_saved / raw_tokens_blocked) * 100.0, 1)
            if raw_tokens_blocked > 0
            else 0.0
        )
        api_dollars_saved_direct = round(total_direct_usd, 4)
        api_dollars_saved_output = round(total_output_usd, 4)
        api_dollars_saved_est = round(total_direct_usd + total_reingest_usd + total_output_usd, 2)

        if direct_tokens_saved > 0:
            # Amdahl-grounded session runway: includes ~4,500 tok/turn unshieldable system/prompt baseline
            base_turn_overhead = max(1, actual_turns_observed) * 4_500
            shielded_budget = max(1, base_turn_overhead + compressed_tokens_sent)
            unshielded_budget = shielded_budget + direct_tokens_saved
            quota_multiplier_est = round(min(25.0, max(1.0, unshielded_budget / shielded_budget)), 1)
        else:
            quota_multiplier_est = 1.0

        sorted_bots = sorted(
            bot_agg.items(),
            key=lambda kv: (kv[1]["tokens_saved"], kv[1]["events"]),
            reverse=True,
        )
        by_bot: Dict[str, Dict[str, Any]] = {}
        bots_leaderboard: List[Dict[str, Any]] = []
        for idx_bot, (b_name, b_st) in enumerate(sorted_bots, start=1):
            b_saved = int(b_st["tokens_saved"])
            b_cum_saved = int(bot_cumulative_map.get(b_name, b_saved))
            b_turns = int(bot_turns_map.get(b_name, int(b_st["events"])))
            b_time_sec = round(float(b_st["time_sec"]), 1)
            b_usd = round(
                float(b_st["direct_usd"]) + float(b_st["output_usd"]) + float(bot_reingest_usd_map.get(b_name, 0.0)),
                2,
            )
            b_avg_lat = round(sum(b_st["lats"]) / len(b_st["lats"]), 1) if b_st["lats"] else 0.0
            b_entry = {
                "rank": idx_bot,
                "bot_name": b_name,
                "events": int(b_st["events"]),
                "actual_turns_observed": b_turns,
                "raw_tokens": int(b_st["raw_tokens"]),
                "compressed_tokens": int(b_st["compressed_tokens"]),
                "tokens_saved": b_saved,
                "cumulative_tokens_saved": b_cum_saved,
                "quadratic_10turn_tokens_saved": b_cum_saved,
                "usd_saved": b_usd,
                "time_saved_sec": b_time_sec,
                "time_saved_display": _format_duration_compact(b_time_sec),
                "avg_latency_ms": b_avg_lat,
                "secrets_blocked": int(b_st["secrets_blocked"]),
            }
            by_bot[b_name] = b_entry
            bots_leaderboard.append(b_entry)

        sorted_events = sorted(
            event_agg.items(),
            key=lambda kv: (kv[1]["tokens_saved"], kv[1]["events"]),
            reverse=True,
        )
        by_event_type: Dict[str, Dict[str, Any]] = {}
        organs_leaderboard: List[Dict[str, Any]] = []
        for idx_ev, (e_type, e_st) in enumerate(sorted_events, start=1):
            e_saved = int(e_st["tokens_saved"])
            e_time_sec = round(float(e_st["time_sec"]), 1)
            e_share = round((e_saved / direct_tokens_saved) * 100.0, 1) if direct_tokens_saved > 0 else 0.0
            e_usd = round(float(e_st["direct_usd"]) + float(e_st["output_usd"]), 2)
            e_avg_lat = round(sum(e_st["lats"]) / len(e_st["lats"]), 1) if e_st["lats"] else 0.0
            ev_entry = {
                "rank": idx_ev,
                "event_type": e_type,
                "events": int(e_st["events"]),
                "raw_tokens": int(e_st["raw_tokens"]),
                "compressed_tokens": int(e_st["compressed_tokens"]),
                "tokens_saved": e_saved,
                "output_tokens_saved": int(e_st["output_tokens_saved"]),
                "share_pct": e_share,
                "usd_saved": e_usd,
                "avg_latency_ms": e_avg_lat,
                "time_saved_sec": e_time_sec,
                "time_saved_display": _format_duration_compact(e_time_sec),
                "secrets_blocked": int(e_st["secrets_blocked"]),
            }
            by_event_type[e_type] = ev_entry
            organs_leaderboard.append(ev_entry)

        sorted_targets = sorted(
            target_agg.items(),
            key=lambda kv: (kv[1]["tokens_saved"], kv[1]["events"]),
            reverse=True,
        )
        targets_leaderboard: List[Dict[str, Any]] = []
        for idx_t, (t_name, t_st) in enumerate(sorted_targets[:safe_top], start=1):
            t_raw = int(t_st["raw_tokens"])
            t_comp = int(t_st["compressed_tokens"])
            t_saved = int(t_st["tokens_saved"])
            t_pct = round((t_saved / t_raw) * 100.0, 1) if t_raw > 0 else 0.0
            t_usd = round(float(t_st["direct_usd"]), 2)
            t_time_sec = round(float(t_st["time_sec"]), 1)
            t_avg_lat = round(sum(t_st["lats"]) / len(t_st["lats"]), 1) if t_st["lats"] else 0.0
            gates = t_st["gates"]
            if len(gates) == 1:
                gate_label = gates[0]
            elif len(gates) > 1:
                short_gates = [g.replace("local_metal_scout", "scout").replace("ast_slice_gate", "ast_slice") for g in gates]
                gate_label = "+".join(short_gates)
            else:
                gate_label = "shield"
            targets_leaderboard.append(
                {
                    "rank": idx_t,
                    "target": t_name,
                    "event_type": gate_label,
                    "events": int(t_st["events"]),
                    "raw_tokens": t_raw,
                    "compressed_tokens": t_comp,
                    "compression_pct": t_pct,
                    "tokens_saved": t_saved,
                    "usd_saved": t_usd,
                    "avg_latency_ms": t_avg_lat,
                    "time_saved_sec": t_time_sec,
                    "time_saved_display": _format_duration_compact(t_time_sec),
                    "secrets_blocked": int(t_st["secrets_blocked"]),
                }
            )

        sessions_leaderboard_raw: List[Dict[str, Any]] = []
        for (b_name, s_id), s_meta in session_meta_map.items():
            s_saved = int(s_meta["tokens_saved"])
            s_cum, s_turns, s_reingest_usd = session_cum_map.get((b_name, s_id), (s_saved, 1, 0.0))
            s_time = round(float(s_meta["time_saved_sec"]), 1)
            s_usd = round(float(s_meta["direct_usd"]) + float(s_reingest_usd), 2)
            sessions_leaderboard_raw.append(
                {
                    "session_id": s_id,
                    "bot_name": b_name,
                    "model_name": s_meta.get("model_name") or detected_model,
                    "events": int(s_meta["events"]),
                    "turns": int(s_turns),
                    "tokens_saved": s_saved,
                    "cumulative_tokens_saved": int(s_cum),
                    "usd_saved": s_usd,
                    "time_saved_sec": s_time,
                    "time_saved_display": _format_duration_compact(s_time),
                }
            )
        sessions_leaderboard_raw.sort(key=lambda x: (x["tokens_saved"], x["time_saved_sec"]), reverse=True)
        sessions_leaderboard = [
            {**item, "rank": idx_s}
            for idx_s, item in enumerate(sessions_leaderboard_raw[: min(10, safe_top)], start=1)
        ]

        organs_breakdown = {
            "eyes": {
                "events": len(eyes_lats),
                "tokens_saved": eyes_tokens_saved,
                "avg_latency_ms": round(sum(eyes_lats) / len(eyes_lats), 2) if eyes_lats else 0.0,
                "time_saved_sec": round(eyes_time_sec, 1),
                "time_saved_display": _format_duration_compact(eyes_time_sec),
            },
            "hands": {
                "events": len(hands_lats),
                "tokens_saved": hands_tokens_saved,
                "avg_latency_ms": round(sum(hands_lats) / len(hands_lats), 2) if hands_lats else 0.0,
                "time_saved_sec": round(hands_time_sec, 1),
                "time_saved_display": _format_duration_compact(hands_time_sec),
            },
            "shield": {
                "events": len(shield_lats),
                "tokens_saved": shield_tokens_saved,
                "avg_latency_ms": round(sum(shield_lats) / len(shield_lats), 2) if shield_lats else 0.0,
                "secrets_blocked": secrets_blocked,
                "time_saved_sec": round(shield_time_sec, 1),
                "time_saved_display": _format_duration_compact(shield_time_sec),
            },
        }

        recent_events: List[Dict[str, Any]] = [
            {
                "timestamp": float(r["timestamp"]),
                "bot_name": str(r["bot_name"]),
                "session_id": str(r["session_id"] or ""),
                "model_name": str(r["model_name"] or ""),
                "event_type": str(r["event_type"]),
                "action_mode": str(r["action_mode"] or ""),
                "target": str(r["target"]),
                "raw_tokens_est": int(r["raw_tokens_est"]),
                "compressed_tokens_est": int(r["compressed_tokens_est"]),
                "tokens_saved": int(r["tokens_saved"]),
                "secrets_blocked": int(r["secrets_blocked"]),
                "latency_ms": float(r["latency_ms"]),
            }
            for r in recent_rows
        ]

        rounded_total_time_sec = round(total_time_saved_sec, 1)
        ast_miss_rate_pct = (
            min(100.0, round((ast_slice_misses / ast_slice_events) * 100.0, 2))
            if ast_slice_events > 0
            else 0.0
        )
        ast_precision_pct = (
            round(max(0.0, 100.0 - ast_miss_rate_pct), 2)
            if ast_slice_events > 0
            else 100.0
        )

        return {
            "db_path": str(self.db_path),
            "window": window_label,
            "since_timestamp": since_ts,
            "bot_filter": norm_bot,
            "active_model": detected_model,
            "total_events": max(0, total_events - ast_slice_misses),
            "ast_slice_events": ast_slice_events,
            "ast_slice_misses": ast_slice_misses,
            "ast_miss_rate_pct": ast_miss_rate_pct,
            "ast_precision_pct": ast_precision_pct,
            "actual_turns_observed": actual_turns_observed,
            "raw_tokens_blocked": raw_tokens_blocked,
            "compressed_tokens_sent": compressed_tokens_sent,
            "direct_tokens_saved": direct_tokens_saved,
            "output_tokens_saved_est": total_output_tokens_saved,
            "compression_pct": compression_pct,
            "cumulative_tokens_saved": cumulative_tokens_saved,
            "quadratic_10turn_tokens_saved": cumulative_tokens_saved,
            "price_per_m_usd": effective_price_per_m,
            "price_override_active": price_per_m is not None,
            "distinct_model_rates": len(observed_rates),
            "api_dollars_saved_direct": api_dollars_saved_direct,
            "api_dollars_saved_output": api_dollars_saved_output,
            "api_dollars_saved_est": api_dollars_saved_est,
            "time_saved_sec": rounded_total_time_sec,
            "time_saved_minutes": round(rounded_total_time_sec / 60.0, 2),
            "time_saved_display": _format_duration_compact(rounded_total_time_sec),
            "quota_multiplier_est": quota_multiplier_est,
            "secrets_blocked": secrets_blocked,
            "avg_latency_ms": avg_latency_ms,
            "hook_reflex_latency_ms": hook_reflex_latency_ms,
            "local_scout_latency_ms": local_scout_latency_ms,
            "local_scout_lifetime_avg_ms": local_scout_lifetime_avg_ms,
            "local_scout_events": local_scout_events,
            "is_scout_warm": scout_warm,
            "organs": organs_breakdown,
            "bots_protected": list(by_bot.keys()),
            "by_bot": by_bot,
            "by_event_type": by_event_type,
            "leaderboard": {
                "organs": organs_leaderboard,
                "targets": targets_leaderboard,
                "bots": bots_leaderboard,
                "sessions": sessions_leaderboard,
            },
            "recent_events": recent_events,
        }


def record_ast_reread_if_recent(
    target: str,
    *,
    bot_name: Optional[str] = None,
    session_id: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
    raw_tokens_est: int = 0,
    window_sec: float = 300.0,
    db_path: Optional[Path | str] = None,
    timestamp: Optional[float] = None,
) -> bool:
    """Module-level convenience wrapper for recording an AST slice re-read miss."""
    return ShieldTelemetryLedger(db_path=db_path).record_ast_reread_if_recent(
        target=target,
        bot_name=bot_name,
        session_id=session_id,
        payload=payload,
        raw_tokens_est=raw_tokens_est,
        window_sec=window_sec,
        timestamp=timestamp,
    )


def record_shield_event(
    event_type: str,
    target: str,
    raw_tokens_est: int = 0,
    compressed_tokens_est: int = 0,
    tokens_saved: Optional[int] = None,
    secrets_blocked: int = 0,
    latency_ms: float = 0.0,
    bot_name: Optional[str] = None,
    session_id: Optional[str] = None,
    model_name: Optional[str] = None,
    action_mode: str = "",
    payload: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path | str] = None,
) -> bool:
    """
    Fast, zero-exception helper to record a Blood-Brain Barrier shield event in SQLite WAL.
    """
    try:
        ledger = ShieldTelemetryLedger(db_path=db_path)
        return ledger.record_event(
            event_type=event_type,
            target=target,
            raw_tokens_est=raw_tokens_est,
            compressed_tokens_est=compressed_tokens_est,
            tokens_saved=tokens_saved,
            secrets_blocked=secrets_blocked,
            latency_ms=latency_ms,
            bot_name=bot_name,
            session_id=session_id,
            model_name=model_name,
            action_mode=action_mode,
            payload=payload,
        )
    except Exception:
        return False


def record_proxy_call(
    *,
    db_path: Optional[Path | str] = None,
    **kwargs: Any,
) -> bool:
    """Zero-exception helper to record a FinOps API proxy call in `proxy_finops_ledger`."""
    try:
        ledger = ShieldTelemetryLedger(db_path=db_path)
        return ledger.record_proxy_call(**kwargs)
    except Exception:
        return False


def get_proxy_client_summary(
    *,
    window: Optional[str] = None,
    client_filter: Optional[str] = None,
    db_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Return structured `proxy_finops_ledger` totals and per-client attribution."""
    ledger = ShieldTelemetryLedger(db_path=db_path)
    return ledger.get_proxy_client_summary(window=window, client_filter=client_filter)


def is_scout_warm() -> bool:
    """Fast check whether Local Metal Scout (:9379 body-coder-mlx / voicefi-coder-mlx or Ollama) is actively resident in RAM."""
    if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("WBC_TESTING") == "1":
        return os.environ.get("WBC_MOCK_SCOUT_WARM") == "1"
    try:
        import urllib.request

        req = urllib.request.Request("http://127.0.0.1:9379/health")
        with urllib.request.urlopen(req, timeout=0.15) as resp:
            if resp.status == 200:
                return True
    except Exception:
        pass
    try:
        import urllib.request

        req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
        with urllib.request.urlopen(req, timeout=0.15) as resp:
            if resp.status == 200:
                return True
    except Exception:
        pass
    return False


def get_savings_summary(
    bot_filter: Optional[str] = None,
    db_path: Optional[Path | str] = None,
    price_per_m: Optional[float] = None,
    window: Optional[str] = None,
    top_limit: int = 15,
    include_benchmarks: bool = False,
) -> Dict[str, Any]:
    """Return structured shield telemetry summary & leaderboard for `body savings` / `body leaderboard`. # [direct-edit]"""
    if (
        not os.environ.get("PYTEST_CURRENT_TEST")
        and os.environ.get("WBC_TESTING") != "1"
        and db_path is None
    ):
        try:
            from whole_body_compute.local.scout import prewarm_scout

            prewarm_scout(background=True)
        except Exception:
            pass
    ledger = ShieldTelemetryLedger(db_path=db_path)
    summary = ledger.get_summary(
        bot_filter=bot_filter,
        price_per_m=price_per_m,
        window=window,
        top_limit=top_limit,
        include_benchmarks=include_benchmarks,
    )
    if (
        not os.environ.get("PYTEST_CURRENT_TEST")
        and db_path is None
        and int(summary.get("total_events", 0)) > 0
    ):
        try:
            from whole_body_compute.procedural_telemetry import track_wbc_event

            track_wbc_event(
                "wbc_shield_pulse",
                {
                    "trigger": "savings_summary",
                    "window": str(window or "all"),
                    "total_events": int(summary.get("total_events", 0)),
                    "direct_tokens_saved": int(summary.get("direct_tokens_saved", 0)),
                    "cumulative_tokens_saved": int(summary.get("cumulative_tokens_saved", 0)),
                    "estimated_usd_saved": float(summary.get("api_dollars_saved_est", 0.0)),
                    "secrets_scrubbed": int(summary.get("secrets_blocked", 0)),
                    "wall_time_saved_sec": float(summary.get("time_saved_sec", 0.0)),
                },
            )
        except Exception:
            pass
    return summary


def format_savings_receipt(summary: Dict[str, Any]) -> str:
    """
    Render the ASCII receipt box for `body savings`.
    # [direct-edit]
    """
    raw_tok = int(summary.get("raw_tokens_blocked", 0))
    comp_tok = int(summary.get("compressed_tokens_sent", 0))
    saved_tok = int(summary.get("direct_tokens_saved", 0))
    comp_pct = float(summary.get("compression_pct", 0.0))
    cum_tok = int(summary.get("cumulative_tokens_saved", summary.get("direct_tokens_saved", 0)))
    usd_est = float(summary.get("api_dollars_saved_est", 0.0))
    usd_direct = float(summary.get("api_dollars_saved_direct", 0.0))
    time_disp = str(summary.get("time_saved_display") or _format_duration_compact(float(summary.get("time_saved_sec", 0.0))))
    win_label = str(summary.get("window") or "all").upper()
    price_rate = float(summary.get("price_per_m_usd", 3.00))
    active_model = str(summary.get("active_model") or "Claude Sonnet 4.5")
    mult = float(summary.get("quota_multiplier_est", 1.0))
    sec_blocked = int(summary.get("secrets_blocked", 0))
    total_events = int(summary.get("total_events", 0))
    actual_turns = int(summary.get("actual_turns_observed", total_events))
    hook_lat = float(summary.get("hook_reflex_latency_ms", 0.0))
    scout_lat = float(summary.get("local_scout_latency_ms", 0.0))
    scout_life_lat = float(summary.get("local_scout_lifetime_avg_ms", scout_lat))
    scout_cnt = int(summary.get("local_scout_events", 0))
    is_warm = bool(summary.get("is_scout_warm", False))
    if scout_cnt == 0:
        scout_line_val = "0.0 ms (no LLM digests yet)"
    elif scout_cnt > 3 and abs(scout_life_lat - scout_lat) >= 100.0:
        scout_line_val = f"{scout_lat:.1f} ms recent ({scout_life_lat:.0f} ms avg / {scout_cnt} runs)"
    elif scout_lat <= 25.0:
        scout_line_val = f"{scout_lat:.1f} ms (warm cache / AST slice)"
    elif scout_lat <= 2500.0 or is_warm:
        scout_line_val = f"{scout_lat:.1f} ms (warm RAM / :9379 metal)"
    else:
        scout_line_val = f"{scout_lat:.1f} ms (cold load - prewarm recommended)"

    organs = summary.get("organs") or {}
    eyes_st = organs.get("eyes") or {"events": 0, "tokens_saved": 0, "avg_latency_ms": 0.0}
    hands_st = organs.get("hands") or {"events": 0, "tokens_saved": 0, "avg_latency_ms": 0.0}
    shield_st = organs.get("shield") or {"events": 0, "tokens_saved": 0, "avg_latency_ms": 0.0}

    bots = summary.get("bots_protected") or []
    bots_str = ", ".join(bots) if bots else "antigravity, claude-code (ready)"

    width = 80

    def _box_line(text: str = "") -> str:
        if len(text) > width - 4:
            text = text[: width - 7] + "..."
        return f"│  {text:<{width - 4}}│"

    hdr_suffix = f" [{win_label}]" if win_label != "ALL" else ""
    if total_events == 0:
        return "\n".join(
            [
                f"╭{'─' * (width - 2)}╮",
                _box_line(f"🛡️  WHOLE BODY COMPUTE - LOCAL METAL SHIELD RECEIPT{hdr_suffix}"),
                f"├{'─' * (width - 2)}┤",
                _box_line("Status:                 ARMED & WAITING FOR FIRST SHIELDED SESSION"),
                _box_line("Shield Interceptions:   0 events (0 turns)"),
                _box_line(""),
                _box_line("How to activate your first session:"),
                _box_line("  1. Close & reopen Claude Code / Cursor / Antigravity"),
                _box_line("     (or `/exit` and re-run `claude` in terminal) so hooks load."),
                _box_line("  2. Use a LOCAL session (terminal `claude` or a local folder in"),
                _box_line("     Claude Desktop - remote web/cloud VMs cannot run local hooks)."),
                _box_line("  3. Work normally for a few prompts, then re-run `body savings`."),
                _box_line("     (Zero-RAM Shielding: <5ms pure Python/AST, 0 GB local LLM RAM.)"),
                f"╰{'─' * (width - 2)}╯",
            ]
        )
    cum_label = f"Cumulative Saved ({actual_turns} turns):"
    lines = [
        f"╭{'─' * (width - 2)}╮",
        _box_line(f"🛡️  WHOLE BODY COMPUTE - LOCAL METAL SHIELD RECEIPT{hdr_suffix}"),
        f"├{'─' * (width - 2)}┤",
        _box_line(f"Bots Protected:         {bots_str}"),
        _box_line(f"Active Model:           {active_model} (${price_rate:.2f}/1M)"),
        _box_line(f"Shield Interceptions:   {total_events:,} events ({actual_turns:,} turns)"),
        _box_line(f"Raw Tokens Blocked:     {raw_tok:,} tok"),
        _box_line(f"Compressed AST Sent:    {comp_tok:,} tok ({comp_pct:.1f}% reduction)"),
        _box_line(f"Direct Tokens Saved:    {saved_tok:,} tok (${usd_direct:,.2f} direct)"),
        _box_line(f"{cum_label} {cum_tok:,} tok"),
        f"├{'─' * (width - 2)}┤",
        _box_line(
            f"👁️  Eyes (Foveal/Screen): {int(eyes_st['events']):>3} snaps | "
            f"{float(eyes_st['avg_latency_ms']):>5.1f} ms avg | saved {int(eyes_st['tokens_saved']):,} tok"
        ),
        _box_line(
            f"🖐️  Hands (AX/Tactile):   {int(hands_st['events']):>3} acts  | "
            f"{float(hands_st['avg_latency_ms']):>5.1f} ms avg | saved {int(hands_st['tokens_saved']):,} tok"
        ),
        _box_line(
            f"🛡️  Shield (AST/Secrets): {int(shield_st['events']):>3} gates | "
            f"{float(shield_st['avg_latency_ms']):>5.1f} ms avg | saved {int(shield_st['tokens_saved']):,} tok"
        ),
        f"├{'─' * (width - 2)}┤",
        _box_line(
            f"Est. API Dollars Saved: ${usd_est:,.2f} (per-model direct + 90%-cached)"
            if (not summary.get("price_override_active") and int(summary.get("distinct_model_rates", 1) or 1) > 1)
            else f"Est. API Dollars Saved: ${usd_est:,.2f} (direct + 90%-cached @ ${price_rate:.2f}/1M)"
        ),
        _box_line(f"Wall-Clock Time Saved:  {time_disp} net agent wait avoided"),
        _box_line(f"Subscription Runway:    {mult:.1f}x longer before 5-hour cap"),
        _box_line(f"Secrets Scrubbed:       {sec_blocked:,} leaked keys blocked"),
        _box_line(f"Hook Reflex Latency:    {hook_lat:.1f} ms ($0.00 local compute)"),
        *(
            [
                _box_line(
                    f"AST Slice Precision:    {float(summary.get('ast_precision_pct', 100.0)):.1f}% "
                    f"({int(summary.get('ast_slice_misses', 0)):,} re-reads / "
                    f"{int(summary.get('ast_slice_events', 0)):,} slices)"
                )
            ]
            if int(summary.get("ast_slice_events", 0) or 0) > 0
            else []
        ),
        _box_line(f"Local Metal Scout:      {scout_line_val}"),
        f"╰{'─' * (width - 2)}╯",
    ]

    by_event = summary.get("by_event_type") or {}
    if by_event:
        lines.append("")
        lines.append("📊 Breakdown by Shield Organ Gate:")
        for etype, stats in by_event.items():
            t_str = str(stats.get("time_saved_display") or _format_duration_compact(float(stats.get("time_saved_sec", 0.0))))
            lines.append(
                f"  • {etype:<22} | {stats['events']:>4}x | "
                f"saved {stats['tokens_saved']:>9,} tok | "
                f"time {t_str:>6} | secrets={stats['secrets_blocked']}"
            )
        lines.append("  💡 Tip: Run `body leaderboard` (or `body leaderboard --today`) for full file/app rankings.")

    try:
        from whole_body_compute.local.reflex_gate import format_reflex_gate_status_lines

        lines.append("")
        lines.extend(format_reflex_gate_status_lines())
    except Exception:
        pass

    return "\n".join(lines)


def format_savings_leaderboard(summary: Dict[str, Any], top_n: int = 15) -> str:
    """
    Render the ranked 4-board Savings Leaderboard (Tokens, USD, and Wall-Clock Time Saved).
    # [direct-edit]
    """
    win_label = str(summary.get("window") or "all").upper()
    active_model = str(summary.get("active_model") or "Claude Sonnet 4.5")
    price_rate = float(summary.get("price_per_m_usd", 3.00))
    total_events = int(summary.get("total_events", 0))
    actual_turns = int(summary.get("actual_turns_observed", total_events))
    saved_tok = int(summary.get("direct_tokens_saved", 0))
    cum_tok = int(summary.get("cumulative_tokens_saved", saved_tok))
    usd_est = float(summary.get("api_dollars_saved_est", 0.0))
    time_disp = str(summary.get("time_saved_display") or _format_duration_compact(float(summary.get("time_saved_sec", 0.0))))

    lb = summary.get("leaderboard") or {}
    organs_lb: List[Dict[str, Any]] = lb.get("organs") or []
    targets_lb: List[Dict[str, Any]] = (lb.get("targets") or [])[: max(1, int(top_n or 15))]
    bots_lb: List[Dict[str, Any]] = lb.get("bots") or []
    sessions_lb: List[Dict[str, Any]] = (lb.get("sessions") or [])[: min(10, max(1, int(top_n or 15)))]

    lines = [
        "",
        "╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗",
        f"║  🏆 WHOLE BODY COMPUTE - SAVINGS LEADERBOARD [{win_label:<5}] (TOKENS · USD · WALL-CLOCK TIME SAVED)        ║",
        "╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝",
        f"  • Window: {win_label}  |  Events: {total_events:,} ({actual_turns:,} turns)  |  Model Rate: {active_model} (${price_rate:.2f}/1M)",
        f"  • Total Saved: {saved_tok:,} direct tok ({cum_tok:,} cum.)  |  Est. USD: ${usd_est:,.2f}  |  Net Time Saved: {time_disp}",
        "",
        "  🥇 1. LEADERBOARD BY ORGAN GATE (WHAT MECHANISM SAVED THE MOST)",
        "  ┌──────┬──────────────────────────┬────────┬───────────────┬─────────┬──────────┬───────────┬────────────┐",
        "  │ Rank │ Organ Gate               │ Events │ Tokens Saved  │ Share % │ Direct $ │ Avg Lat   │ Time Saved │",
        "  ├──────┼──────────────────────────┼────────┼───────────────┼─────────┼──────────┼───────────┼────────────┤",
    ]
    if organs_lb:
        for row in organs_lb:
            lines.append(
                f"  │ #{int(row['rank']):<3} │ {str(row['event_type']):<24} │ {int(row['events']):>5}x │ "
                f"{int(row['tokens_saved']):>13,} │ {float(row['share_pct']):>6.1f}% │ "
                f"${float(row['usd_saved']):>7.2f} │ {float(row['avg_latency_ms']):>6.1f} ms │ "
                f"{str(row['time_saved_display']):>10} │"
            )
    else:
        lines.append("  │  --  │ No shield events recorded in this time window yet.                                             │")
    lines.append("  └──────┴──────────────────────────┴────────┴───────────────┴─────────┴──────────┴───────────┴────────────┘")

    lines.extend([
        "",
        f"  📂 2. TOP {len(targets_lb) if targets_lb else top_n} SHIELDED FILES, LOGS & DESKTOP APPS",
        "  ┌──────┬────────────────────────────────────┬──────────────────────┬───────┬────────┬──────────────┬────────────┐",
        "  │ Rank │ Target (File / Log / App)          │ Organ Gate           │ Hits  │ Comp % │ Tokens Saved │ Time Saved │",
        "  ├──────┼────────────────────────────────────┼──────────────────────┼───────┼────────┼──────────────┼────────────┤",
    ])
    if targets_lb:
        for row in targets_lb:
            tgt = " ".join(str(row["target"]).split())
            if len(tgt) > 34:
                tgt = tgt[:31] + "..."
            ev_lbl = str(row["event_type"])
            if len(ev_lbl) > 20:
                ev_lbl = ev_lbl[:20]
            lines.append(
                f"  │ #{int(row['rank']):<3} │ {tgt:<34} │ {ev_lbl:<20} │ {int(row['events']):>4}x │ "
                f"{float(row['compression_pct']):>5.1f}% │ {int(row['tokens_saved']):>12,} │ "
                f"{str(row['time_saved_display']):>10} │"
            )
    else:
        lines.append("  │  --  │ No target files or apps recorded in this time window yet.                                     │")
    lines.append("  └──────┴────────────────────────────────────┴──────────────────────┴───────┴────────┴──────────────┴────────────┘")

    lines.extend([
        "",
        "  🤖 3. LEADERBOARD BY AGENT HARNESS",
        "  ┌──────┬────────────────────┬────────┬────────┬────────────────┬──────────────────┬──────────┬────────────┐",
        "  │ Rank │ Bot Harness        │ Turns  │ Events │ Direct Tokens  │ Cumulative Tok   │ Est. USD │ Time Saved │",
        "  ├──────┼────────────────────┼────────┼────────┼────────────────┼──────────────────┼──────────┼────────────┤",
    ])
    if bots_lb:
        for row in bots_lb:
            lines.append(
                f"  │ #{int(row['rank']):<3} │ {str(row['bot_name']):<18} │ {int(row['actual_turns_observed']):>6,} │ "
                f"{int(row['events']):>5}x │ {int(row['tokens_saved']):>14,} │ "
                f"{int(row['cumulative_tokens_saved']):>16,} │ ${float(row['usd_saved']):>7.2f} │ "
                f"{str(row['time_saved_display']):>10} │"
            )
    else:
        lines.append("  │  --  │ No bot harnesses recorded in this time window yet.                                            │")
    lines.append("  └──────┴────────────────────┴────────┴────────┴────────────────┴──────────────────┴──────────┴────────────┘")

    if sessions_lb:
        lines.extend([
            "",
            f"  🧵 4. TOP {len(sessions_lb)} SESSIONS BY SHIELD LIFT",
            "  ┌──────┬──────────────────────────────────────┬──────────────┬────────┬───────────────┬──────────┬────────────┐",
            "  │ Rank │ Session / Conversation ID            │ Bot Harness  │ Events │ Tokens Saved  │ Est. USD │ Time Saved │",
            "  ├──────┼──────────────────────────────────────┼──────────────┼────────┼───────────────┼──────────┼────────────┤",
        ])
        for row in sessions_lb:
            sid = str(row["session_id"])
            if len(sid) > 36:
                sid = sid[:33] + "..."
            lines.append(
                f"  │ #{int(row['rank']):<3} │ {sid:<36} │ {str(row['bot_name']):<12} │ {int(row['events']):>5}x │ "
                f"{int(row['tokens_saved']):>13,} │ ${float(row['usd_saved']):>7.2f} │ "
                f"{str(row['time_saved_display']):>10} │"
            )
        lines.append("  └──────┴──────────────────────────────────────┴──────────────┴────────┴───────────────┴──────────┴────────────┘")

    lines.append("  💡 Filter windows: `body leaderboard --today` | `--window 24h` | `--window 7d` | `--top 25` | `--json`")
    lines.append("")
    return "\n".join(lines)


_ENV_ASSIGN_TOKEN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")
_SHELL_WRAPPER_TOKENS = {"env", "command", "nohup", "time"}
_READ_TOOL_NAMES = {
    "read_file",
    "view_file",
    "Read",
    "NotebookRead",
    "developer__text_editor",
    "beforeReadFile",
    "pre_read_code",
}
_SHELL_TOOL_NAMES = {
    "run_command",
    "Bash",
    "execute_command",
    "developer__shell",
}
_SOFT_REFLEX_THRESHOLDS: dict[str, float] = {
    "read_autoslice_window": 0.89,
    "recon_scout_sufficiency": 0.89,
}


def _resolve_default_audit_log_path() -> Path:
    env_p = (os.environ.get("WBC_IMMUNE_AUDIT_PATH") or os.environ.get("VOICEFI_IMMUNE_AUDIT_PATH") or "").strip()
    if env_p:
        return Path(os.path.expanduser(env_p))
    env_data = os.environ.get("WBC_DATA_DIR", "").strip()
    if env_data:
        return Path(os.path.expanduser(env_data)) / "immune_audit.jsonl"
    return Path.home() / ".whole_body_compute" / "immune_audit.jsonl"


def _resolve_default_reflex_db_path() -> Path:
    env_p = os.environ.get("WBC_REFLEX_DB_PATH", "").strip()
    if env_p:
        return Path(os.path.expanduser(env_p))
    env_data = os.environ.get("WBC_DATA_DIR", "").strip()
    if env_data:
        return Path(os.path.expanduser(env_data)) / "reflex_gates.sqlite"
    return Path.home() / ".whole_body_compute" / "reflex_gates.sqlite"


def _extract_shell_two_token_prefix(cmd_str: str) -> str:
    """Extract a clean 2-token command prefix, stripping leading `FOO=bar` env assignments."""
    raw = (cmd_str or "").strip()
    if not raw:
        return ""
    first_seg = raw.splitlines()[0].strip()
    for sep in ("&&", "||", ";"):
        if sep in first_seg:
            cand = first_seg.split(sep, 1)[0].strip()
            if cand and not all(_ENV_ASSIGN_TOKEN_RE.match(t) for t in cand.split()):
                first_seg = cand
                break
    tokens = first_seg.split()
    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        if _ENV_ASSIGN_TOKEN_RE.match(tok) or tok in _SHELL_WRAPPER_TOKENS:
            idx += 1
            continue
        break
    rem = tokens[idx:]
    if not rem:
        return ""
    t0 = Path(rem[0].strip("'\"")).name or rem[0]
    if len(rem) == 1:
        return t0
    t1 = rem[1].strip("'\"")
    if t1 in ("|", ">", ">>", "2>", "2>&1", "<"):
        return t0
    if "/" in t1 and not t1.startswith("-"):
        t1 = Path(t1).name or t1
    if len(t1) > 28:
        t1 = t1[:25] + "..."
    return f"{t0} {t1}"


def _tail_audit_events(
    audit_file: Path,
    cutoff_ts: float,
    *,
    max_bytes: int = 40 * 1024 * 1024,
    chunk_size: int = 4 * 1024 * 1024,
) -> list[dict[str, Any]]:
    """Read recent `immune_audit.jsonl` records backward from EOF in bounded chunks (<300ms on 200MB+)."""
    if not audit_file.is_file():
        return []
    events: list[dict[str, Any]] = []
    try:
        file_size = audit_file.stat().st_size
        if file_size <= 0:
            return []
        with audit_file.open("rb") as fh:
            pos = file_size
            bytes_read = 0
            carry = b""
            stop = False
            old_streak = 0
            while pos > 0 and bytes_read < max_bytes and not stop:
                step = min(chunk_size, pos, max_bytes - bytes_read)
                pos -= step
                bytes_read += step
                fh.seek(pos)
                buf = fh.read(step) + carry
                parts = buf.split(b"\n")
                if pos > 0:
                    carry = parts[0]
                    lines = parts[1:]
                else:
                    carry = b""
                    lines = parts
                for raw_line in reversed(lines):
                    if not raw_line or not raw_line.strip():
                        continue
                    try:
                        ev = json.loads(raw_line)
                    except (ValueError, TypeError):
                        continue
                    if not isinstance(ev, dict):
                        continue
                    ts = float(ev.get("timestamp") or 0.0)
                    if ts >= cutoff_ts:
                        events.append(ev)
                        old_streak = 0
                    elif ts > 0.0:
                        old_streak += 1
                        if old_streak >= 128:
                            stop = True
                            break
    except OSError:
        return events
    events.reverse()
    return events


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    pos = (max(0.0, min(100.0, pct)) / 100.0) * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_vals[lo]) * (1.0 - frac) + float(sorted_vals[hi]) * frac


def get_coverage_gaps(
    hours: float = 5.0,
    audit_path: Path | None = None,
    reflex_db_path: Path | None = None,
    include_benchmarks: bool = False,
) -> dict[str, Any]:
    """
    Mine shield coverage gaps from `immune_audit.jsonl` and `reflex_gates.sqlite`
    for `body savings --gaps [--hours <float>] [--json]`.
    """
    eff_hours = max(0.01, float(hours if hours is not None else 5.0))
    now_ts = time.time()
    cutoff_ts = now_ts - eff_hours * 3600.0

    resolved_audit = Path(audit_path).expanduser() if audit_path is not None else _resolve_default_audit_log_path()
    resolved_reflex = (
        Path(reflex_db_path).expanduser() if reflex_db_path is not None else _resolve_default_reflex_db_path()
    )

    events = _tail_audit_events(resolved_audit, cutoff_ts)
    if not include_benchmarks:
        is_default_audit = (
            audit_path is None
            and not bool(
                os.environ.get("WBC_IMMUNE_AUDIT_PATH")
                or os.environ.get("VOICEFI_IMMUNE_AUDIT_PATH")
                or os.environ.get("WBC_DATA_DIR")
            )
            and not _is_ephemeral_temp_target(str(resolved_audit))
        )
        events = [
            ev
            for ev in events
            if not _is_benchmark_or_ephemeral_row(
                str(ev.get("target") or ""),
                str(ev.get("conversation_id") or ev.get("session_id") or ""),
                exclude_ephemeral_temp=is_default_audit,
            )
        ]

    post_sliced_steps: set[tuple[str, int, str]] = set()
    for ev in events:
        if str(ev.get("phase") or "").strip().lower() == "post" and bool(ev.get("post_output_sliced")):
            c_id = str(ev.get("conversation_id") or "").strip()
            s_i = int(ev.get("step_idx") or 0)
            t_n = str(ev.get("tool") or "").strip()
            if c_id and s_i > 0 and t_n:
                post_sliced_steps.add((c_id, s_i, t_n))

    # (a) Tool Capture & Overhead
    tool_buckets: dict[str, dict[str, Any]] = {}
    # (b) Unbounded Read Leakage
    blind_unbounded_total = 0
    blind_unbounded_autosliced = 0
    # (c) Top Uncompacted Shell Prefixes
    uncompacted_shell_counts: dict[str, int] = {}
    total_shell_pre = 0
    uncompacted_shell_pre = 0

    for ev in events:
        raw_tool = ev.get("tool")
        phase = str(ev.get("phase") or "").strip().lower()
        if not raw_tool or str(raw_tool) == "None":
            if phase == "stop":
                continue
            tool_name = "unknown"
        else:
            tool_name = str(raw_tool)

        direct_saved = max(0, int(ev.get("direct_tokens_saved") or 0))
        quad_saved = max(0, int(ev.get("quadratic_tokens_saved") or 0))
        tot_saved = direct_saved + quad_saved
        lat_ms = max(0.0, float(ev.get("latency_ms") or 0.0))
        is_saving = bool(
            tot_saved > 0
            or ev.get("read_auto_sliced")
            or ev.get("bounded_slice_read")
            or ev.get("context_gate_triggered")
            or ev.get("post_output_sliced")
        )

        bucket = tool_buckets.setdefault(
            tool_name,
            {
                "tool": tool_name,
                "events": 0,
                "saving_events": 0,
                "direct_tokens_saved": 0,
                "quadratic_tokens_saved": 0,
                "total_tokens_saved": 0,
                "latency_sum_ms": 0.0,
                "latencies_ms": [],
            },
        )
        bucket["events"] += 1
        if is_saving:
            bucket["saving_events"] += 1
        bucket["direct_tokens_saved"] += direct_saved
        bucket["quadratic_tokens_saved"] += quad_saved
        bucket["total_tokens_saved"] += tot_saved
        bucket["latency_sum_ms"] += lat_ms
        bucket["latencies_ms"].append(lat_ms)

        ev_conv = str(ev.get("conversation_id") or "").strip()
        ev_step = int(ev.get("step_idx") or 0)
        paired_post_sliced = bool(
            ev_conv and ev_step > 0 and (ev_conv, ev_step, tool_name) in post_sliced_steps
        )

        if (
            phase == "pre"
            and tool_name in _READ_TOOL_NAMES
            and str(ev.get("read_intent") or "") == "blind_unbounded"
            and not bool(ev.get("read_exempt_or_small"))
        ):
            blind_unbounded_total += 1
            if bool(ev.get("read_auto_sliced")) or paired_post_sliced:
                blind_unbounded_autosliced += 1

        if phase == "pre" and tool_name in _SHELL_TOOL_NAMES:
            total_shell_pre += 1
            if not is_saving and not paired_post_sliced:
                uncompacted_shell_pre += 1
                prefix = _extract_shell_two_token_prefix(str(ev.get("target") or ""))
                if prefix:
                    uncompacted_shell_counts[prefix] = uncompacted_shell_counts.get(prefix, 0) + 1

    tool_capture: list[dict[str, Any]] = []
    for t_name, b in tool_buckets.items():
        ev_cnt = int(b["events"])
        sav_cnt = int(b["saving_events"])
        lats = sorted(b["latencies_ms"])
        cap_pct = round((sav_cnt / ev_cnt) * 100.0, 2) if ev_cnt > 0 else 0.0
        tool_capture.append(
            {
                "tool": t_name,
                "events": ev_cnt,
                "saving_events": sav_cnt,
                "capture_rate_pct": cap_pct,
                "direct_tokens_saved": int(b["direct_tokens_saved"]),
                "quadratic_tokens_saved": int(b["quadratic_tokens_saved"]),
                "total_tokens_saved": int(b["total_tokens_saved"]),
                "hook_overhead_s": round(float(b["latency_sum_ms"]) / 1000.0, 3),
                "p50_latency_ms": round(_percentile(lats, 50.0), 2),
            }
        )
    tool_capture.sort(key=lambda r: (-int(r["events"]), str(r["tool"])))

    unbounded_leak_count = max(0, blind_unbounded_total - blind_unbounded_autosliced)
    unbounded_capture_pct = (
        round((blind_unbounded_autosliced / blind_unbounded_total) * 100.0, 2)
        if blind_unbounded_total > 0
        else 0.0
    )
    unbounded_read_leakage: dict[str, Any] = {
        "blind_unbounded_reads": blind_unbounded_total,
        "auto_sliced_count": blind_unbounded_autosliced,
        "unbounded_leak_count": unbounded_leak_count,
        "capture_rate_pct": unbounded_capture_pct,
    }

    top_shell_prefixes: list[dict[str, Any]] = [
        {"prefix": pfx, "count": cnt}
        for pfx, cnt in sorted(uncompacted_shell_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    ]

    # (d) Active (GRADUATED) + Near-Graduated SHADOW Reflex Gates
    graduated_reflex_gates: list[dict[str, Any]] = []
    shadow_reflex_gates: list[dict[str, Any]] = []
    if resolved_reflex.is_file():
        try:
            conn = sqlite3.connect(f"file:{resolved_reflex}?mode=ro", uri=True, timeout=1.5)
            try:
                has_samples_tbl = bool(
                    conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='reflex_samples'"
                    ).fetchone()
                )
                grad_rows = conn.execute(
                    """
                    SELECT gate_id, repo_key, sample_count, cv_accuracy
                    FROM reflex_models
                    WHERE state = 'GRADUATED'
                    ORDER BY sample_count DESC, cv_accuracy DESC
                    """
                ).fetchall()
                for g_id, r_key, s_cnt, cv_acc in grad_rows:
                    graduated_reflex_gates.append(
                        {
                            "gate_id": str(g_id),
                            "repo_key": str(r_key),
                            "sample_count": int(s_cnt or 0),
                            "cv_accuracy": round(float(cv_acc or 0.0), 4),
                        }
                    )

                rows = conn.execute(
                    """
                    SELECT gate_id, repo_key, sample_count, cv_accuracy, false_local_rate
                    FROM reflex_models
                    WHERE state = 'SHADOW' AND sample_count >= 25
                    ORDER BY sample_count DESC, cv_accuracy DESC
                    """
                ).fetchall()
                for g_id, r_key, s_cnt, cv_acc, fl_rate in rows:
                    gate_str = str(g_id)
                    repo_str = str(r_key)
                    thresh = float(_SOFT_REFLEX_THRESHOLDS.get(gate_str, 0.95))
                    raw_cv = float(cv_acc or 0.0)
                    raw_gap = max(0.0, thresh - raw_cv)
                    if 0.0 < raw_gap < 0.0001:
                        gap = 0.0001
                        cv_val = round(min(raw_cv, thresh - 0.0001), 4)
                    else:
                        cv_val = round(raw_cv, 4)
                        gap = round(max(0.0, thresh - cv_val), 4)
                    fl_val = round(float(fl_rate if fl_rate is not None else 1.0), 4)
                    roll_acc = 1.0
                    if has_samples_tbl:
                        if repo_str == "__global__":
                            a_rows = conn.execute(
                                "SELECT label, predicted_local FROM reflex_samples "
                                "WHERE gate_id = ? ORDER BY sample_id DESC LIMIT 50",
                                (gate_str,),
                            ).fetchall()
                        else:
                            a_rows = conn.execute(
                                "SELECT label, predicted_local FROM reflex_samples "
                                "WHERE gate_id = ? AND repo_key = ? ORDER BY sample_id DESC LIMIT 50",
                                (gate_str, repo_str),
                            ).fetchall()
                        scored_rows = [(lbl, prd) for lbl, prd in a_rows if prd is not None]
                        if len(scored_rows) >= 10:
                            roll_acc = round(
                                sum(1 for lbl, prd in scored_rows if int(lbl) == int(prd)) / len(scored_rows), 4
                            )
                    shadow_reflex_gates.append(
                        {
                            "gate_id": gate_str,
                            "repo_key": repo_str,
                            "sample_count": int(s_cnt or 0),
                            "cv_accuracy": cv_val,
                            "false_local_rate": fl_val,
                            "graduation_threshold": thresh,
                            "accuracy_gap": gap,
                            "rolling_audit_acc": roll_acc,
                        }
                    )
            finally:
                conn.close()
        except (sqlite3.Error, OSError):
            shadow_reflex_gates = []

    # (e) Ranked Recommendations (top 3 synthesized from b, c, d in plain English)
    rec_candidates: list[tuple[float, str]] = []
    if unbounded_leak_count > 0:
        rec_msg = (
            f"Auto-trim full-file reads: agents read a whole file top-to-bottom {blind_unbounded_total} times "
            f"and {unbounded_leak_count} slipped through untrimmed ({unbounded_capture_pct:.1f}% trimmed)."
        )
        rec_candidates.append(
            (
                float(unbounded_leak_count) * 2.0 + (100.0 - unbounded_capture_pct),
                rec_msg,
            )
        )
    if top_shell_prefixes:
        top_pfx = top_shell_prefixes[0]
        top3_summary = ", ".join(f"`{r['prefix']}` ({r['count']}x)" for r in top_shell_prefixes[:3])
        rec_candidates.append(
            (
                float(top_pfx["count"]) * 2.5,
                f"Trim noisy terminal outputs for your most-repeated commands: {top3_summary}.",
            )
        )
    if shadow_reflex_gates:
        nearest = min(
            shadow_reflex_gates,
            key=lambda g: (
                0 if float(g.get("rolling_audit_acc", 1.0)) >= 0.90 else 1,
                float(g["accuracy_gap"]),
                -int(g["sample_count"]),
            ),
        )
        g_label = f"{nearest['gate_id']} [{nearest['repo_key']}]"
        roll_ok = float(nearest.get("rolling_audit_acc", 1.0)) >= 0.90
        if (
            float(nearest["accuracy_gap"]) <= 0.0
            and int(nearest["sample_count"]) >= 50
            and float(nearest["false_local_rate"]) <= 0.05
            and roll_ok
        ):
            grad_msg = (
                f"Turn on ready shortcut model `{g_label}` ({nearest['sample_count']} examples, "
                f"{nearest['cv_accuracy'] * 100:.1f}% accuracy >= {nearest['graduation_threshold'] * 100:.0f}% bar) "
                "by running `body shield --bootstrap-reflex`."
            )
            rec_candidates.append(
                (
                    150.0 + float(nearest["sample_count"]) * 0.1,
                    grad_msg,
                )
            )
        else:
            gap_pts = float(nearest["accuracy_gap"]) * 100.0
            samples_needed = max(0, 50 - int(nearest["sample_count"]))
            need_bits = []
            if samples_needed > 0:
                need_bits.append(f"{samples_needed} more examples (to reach 50)")
            if gap_pts > 0:
                gap_str = f"{max(0.01, gap_pts):.2f}" if gap_pts < 0.1 else f"{gap_pts:.1f}"
                need_bits.append(f"+{gap_str}% accuracy (to reach {nearest['graduation_threshold'] * 100:.0f}%)")
            if float(nearest["false_local_rate"]) > 0.05:
                need_bits.append(f"lower error rate from {nearest['false_local_rate'] * 100:.1f}% to <=5.0%")
            if not roll_ok:
                need_bits.append(
                    f"live spot-check streak {float(nearest['rolling_audit_acc']) * 100:.0f}% -> >=90%"
                )
            need_desc = ", ".join(need_bits) if need_bits else "a refresh (`body shield --bootstrap-reflex`)"
            nearest_acc_pct = float(nearest["cv_accuracy"]) * 100.0
            nearest_acc_str = (
                f"{nearest_acc_pct:.2f}" if (0.0 < gap_pts < 0.1) else f"{nearest_acc_pct:.1f}"
            )
            adv_msg = (
                f"Closest shortcut model in training `{g_label}` ({nearest['sample_count']} examples, "
                f"{nearest_acc_str}% accuracy): needs {need_desc}."
            )
            rec_candidates.append(
                (
                    110.0 - gap_pts + min(40.0, float(nearest["sample_count"]) * 0.05),
                    adv_msg,
                )
            )
    if len(rec_candidates) < 3 and len(top_shell_prefixes) >= 2:
        sec_pfx = top_shell_prefixes[1]
        rec_candidates.append(
            (
                float(sec_pfx["count"]),
                f"Add output trimming for `{sec_pfx['prefix']}` ({sec_pfx['count']} untrimmed terminal runs in last {eff_hours:g}h).",
            )
        )
    if not rec_candidates:
        rec_candidates.append(
            (
                1.0,
                f"No major missed-savings gaps in the last {eff_hours:g}h - keep running normally.",
            )
        )

    rec_candidates.sort(key=lambda item: -item[0])
    recommendations = [text for _, text in rec_candidates[:3]]

    return {
        "hours": eff_hours,
        "audit_path": str(resolved_audit),
        "reflex_db_path": str(resolved_reflex),
        "events_scanned": len(events),
        "total_shell_pre_events": total_shell_pre,
        "uncompacted_shell_pre_events": uncompacted_shell_pre,
        "tool_capture": tool_capture,
        "unbounded_read_leakage": unbounded_read_leakage,
        "top_uncompacted_shell_prefixes": top_shell_prefixes,
        "graduated_reflex_gates": graduated_reflex_gates,
        "shadow_reflex_gates": shadow_reflex_gates,
        "recommendations": recommendations,
    }


def format_coverage_gaps(report: dict[str, Any]) -> str:
    """Render plain-English coverage gap report for `body savings --gaps`."""
    hours = float(report.get("hours") or 5.0)
    ev_cnt = int(report.get("events_scanned") or 0)
    tools: list[dict[str, Any]] = report.get("tool_capture") or []
    read_leak: dict[str, Any] = report.get("unbounded_read_leakage") or {}
    shell_pfxs: list[dict[str, Any]] = report.get("top_uncompacted_shell_prefixes") or []
    grad_gates: list[dict[str, Any]] = report.get("graduated_reflex_gates") or []
    shadow_gates: list[dict[str, Any]] = report.get("shadow_reflex_gates") or []
    recs: list[str] = report.get("recommendations") or []

    lines = [
        f"🔍 WHOLE BODY COMPUTE - SHIELD COVERAGE GAPS (LAST {hours:g}H) - Missed Savings X-Ray",
        "─" * 96,
        f"  • Tool Calls Checked: {ev_cnt:,}  |  Log: {report.get('audit_path', '')}",
        "",
        "  1. SAVINGS BY TOOL (How often WBC stepped in to trim context vs. time spent)",
        "  ┌──────────────────────┬────────┬─────────┬──────────┬────────────┬────────────┬──────────┬──────────┐",
        "  │ Tool                 │  Calls │ Trimmed │ Trim %   │ Direct Tok │  Total Tok │ WBC Time │  Median  │",
        "  ├──────────────────────┼────────┼─────────┼──────────┼────────────┼────────────┼──────────┼──────────┤",
    ]
    if tools:
        for row in tools:
            t_lbl = f"{row['tool']}"[:20]
            lines.append(
                f"  │ {t_lbl:<20} │ {int(row['events']):>6,} │ {int(row['saving_events']):>7,} │ "
                f"{float(row['capture_rate_pct']):>7.1f}% │ {int(row['direct_tokens_saved']):>10,} │ "
                f"{int(row['total_tokens_saved']):>10,} │ {float(row['hook_overhead_s']):>6.2f} s │ "
                f"{float(row['p50_latency_ms']):>5.1f} ms │"
            )
    else:
        lines.append(
            "  │ (no calls in window) │      0 │       0 │     0.0% │          0 │          0 │   0.00 s │   0.0 ms │"
        )
    lines.append("  └──────────────────────┴────────┴─────────┴──────────┴────────────┴────────────┴──────────┴──────────┘")

    tot_unb = int(read_leak.get("blind_unbounded_reads", 0))
    auto_cnt = int(read_leak.get("auto_sliced_count", 0))
    leak_cnt = int(read_leak.get("unbounded_leak_count", 0))
    cap_pct = float(read_leak.get("capture_rate_pct", 0.0))
    unbounded_summary = (
        f"     • Whole-File Reads Attempted: {tot_unb:,}  |  Auto-Trimmed: {auto_cnt:,} ({cap_pct:.1f}%)  |  "
        f"Slipped Through Untrimmed: {leak_cnt:,}"
    )
    lines.extend(
        [
            "",
            "  2. WHOLE-FILE READS (Agents reading an entire file instead of specific lines)",
            unbounded_summary,
            "",
            "  3. NOISIEST TERMINAL COMMANDS (Ran repeatedly with zero output trimming)",
        ]
    )
    if shell_pfxs:
        for idx, item in enumerate(shell_pfxs, start=1):
            pfx_lbl = f"{item['prefix']}"
            lines.append(f"     {idx:>2}. {pfx_lbl:<34} {int(item['count']):>5,} untrimmed runs")
    else:
        lines.append("     (no untrimmed terminal commands in window)")

    lines.extend(
        [
            "",
            "  4. LEARNED SHORTCUT MODELS (Active now vs. still training in background)",
        ]
    )
    if grad_gates:
        active_list = ", ".join(
            f"{g['gate_id']} [{g['repo_key']}] ({float(g['cv_accuracy']) * 100:.1f}%)" for g in grad_gates
        )
        lines.append(f"     🟢 ACTIVE NOW ({len(grad_gates)}): {active_list}")
    if shadow_gates:
        for g in shadow_gates:
            lbl = f"{g['gate_id']} [{g['repo_key']}]"
            acc = float(g["cv_accuracy"]) * 100.0
            bar = float(g["graduation_threshold"]) * 100.0
            gap = float(g["accuracy_gap"]) * 100.0
            err = float(g["false_local_rate"]) * 100.0
            roll = float(g.get("rolling_audit_acc", 1.0)) * 100.0
            n_smp = int(g["sample_count"])
            if gap <= 0.0 and n_smp >= 50 and err <= 5.0 and roll >= 90.0:
                status_str = "✅ READY TO TURN ON (`body shield --bootstrap-reflex`)"
            else:
                needs = []
                if n_smp < 50:
                    needs.append(f"+{50 - n_smp} examples")
                if gap > 0.0:
                    gap_str = f"{max(0.01, gap):.2f}" if gap < 0.1 else f"{gap:.1f}"
                    needs.append(f"+{gap_str}% accuracy")
                if err > 5.0:
                    needs.append(f"error rate {err:.1f}% -> <=5%")
                if roll < 90.0:
                    needs.append(f"spot-check streak {roll:.0f}% -> >=90%")
                status_str = f"needs {', '.join(needs)}" if needs else "checking audit history"
            acc_str = f"{acc:>6.2f}%" if (0.0 < gap < 0.1) else f"{acc:>5.1f}%"
            lines.append(
                f"     ⏳ {lbl:<40} {n_smp:>4} examples | accuracy {acc_str} (goal {bar:.0f}%) -> {status_str}"
            )
    elif not grad_gates:
        lines.append("     (no shortcut models currently waiting in training)")

    lines.extend(["", "  5. WHAT TO DO NEXT (Top 3 Wins)"])
    for idx, rec in enumerate(recs, start=1):
        lines.append(f"     {idx}. {rec}")
    lines.append("─" * 96)
    return "\n".join(lines)


def resolve_shield_hook_command() -> str:
    """Resolve the zero-schema executable command for Organ 05 PreToolUse/PostToolUse hooks. # [direct-edit]"""
    import sys

    if os.name == "nt":
        hook_bin = shutil.which("body-immune-hook") or shutil.which("body-immune-hook.exe")
        if hook_bin:
            return f'"{hook_bin}"' if " " in hook_bin else hook_bin
        body_bin = shutil.which("body") or shutil.which("body.exe")
        if body_bin:
            q_body = f'"{body_bin}"' if " " in body_bin else body_bin
            return f"{q_body} immune-hook"
        q_py = f'"{sys.executable}"' if " " in sys.executable else sys.executable
        return f"{q_py} -m whole_body_compute.organs.immune"

    if DEFAULT_HOOK_SCRIPT.exists():
        return f"'{DEFAULT_HOOK_SCRIPT}'"
    hook_bin = shutil.which("body-immune-hook")
    if hook_bin:
        return f"'{hook_bin}'"
    body_bin = shutil.which("body")
    if body_bin:
        return f"'{body_bin}' immune-hook"
    return f"'{sys.executable}' -m whole_body_compute.organs.immune"


def _is_wbc_hook_entry(entry: Any) -> bool:
    """Return True if a hook config entry belongs to Whole Body Compute Organ 05."""
    raw = json.dumps(entry) if not isinstance(entry, str) else entry
    markers = (
        "body-immune-hook",
        "immune-hook",
        "whole_body_compute.organs.immune",
        "whole-body-compute.organs.immune",
    )
    return any(m in raw for m in markers)


class ShieldInstaller:
    """
    Zero-Schema-Bloat Multi-Bot Hook Installer & Auditor (`body shield`).
    Wires ONLY `PreToolUse` and `PostToolUse` hooks (`whole_body_compute.organs.immune`)
    into target bot configuration files while preserving all existing user settings.
    """

    def __init__(
        self,
        workspace_dir: Optional[Path | str] = None,
        home_dir: Optional[Path | str] = None,
        hook_cmd: Optional[str] = None,
    ) -> None:
        self.workspace_dir = Path(workspace_dir).resolve() if workspace_dir else Path.cwd().resolve()
        self.home_dir = Path(home_dir).resolve() if home_dir else Path.home().resolve()
        self.hook_cmd = hook_cmd or resolve_shield_hook_command()

    def _resolve_bot_config_path(self, bot_key: str, scope: str = "workspace") -> Path:
        canon = BOT_ALIASES.get(bot_key.strip().lower(), bot_key.strip().lower())
        spec = SUPPORTED_BOTS.get(canon)
        if not spec:
            raise ValueError(f"Unsupported bot '{bot_key}'. Choose from: {', '.join(SUPPORTED_BOTS.keys())}")
        base = self.home_dir if scope == "user" else self.workspace_dir
        rel = spec["user_rel"] if scope == "user" else spec["workspace_rel"]
        return base / rel

    def inspect_bot(self, bot_key: str, scope: str = "workspace") -> Dict[str, Any]:
        """Inspect whether Organ 05 PreToolUse and PostToolUse hooks are installed for `bot_key`."""
        canon = BOT_ALIASES.get(bot_key.strip().lower(), bot_key.strip().lower())
        cfg_path = self._resolve_bot_config_path(canon, scope=scope)
        pre_installed = False
        post_installed = False
        existing_keys: List[str] = []
        active_path = cfg_path

        candidate_paths = [cfg_path]
        if scope == "workspace":
            if canon == "antigravity":
                candidate_paths.append(self.workspace_dir / ".agents" / "hooks.json")
                candidate_paths.append(self.home_dir / ".gemini" / "config" / "hooks.json")
            elif canon == "claude-code":
                candidate_paths.append(self.workspace_dir / ".claude" / "settings.local.json")
                candidate_paths.append(self.home_dir / ".claude" / "settings.json")
        elif scope == "user":
            if canon == "antigravity":
                candidate_paths.append(self.home_dir / ".gemini" / "config" / "hooks.json")
            elif canon == "claude-code":
                candidate_paths.append(self.home_dir / ".claude" / "settings.local.json")

        for cand in candidate_paths:
            if not cand.is_file():
                continue
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    if cand == cfg_path:
                        existing_keys = list(data.keys())
                    hooks_obj = data.get("hooks") if isinstance(data.get("hooks"), dict) else data
                    gate_obj = data.get("whole-body-immune-gate") if isinstance(data.get("whole-body-immune-gate"), dict) else {}
                    pre_list = (hooks_obj.get("PreToolUse") or []) + (gate_obj.get("PreToolUse") or [])
                    post_list = (hooks_obj.get("PostToolUse") or []) + (gate_obj.get("PostToolUse") or [])
                    c_pre = any(_is_wbc_hook_entry(item) for item in pre_list)
                    c_post = any(_is_wbc_hook_entry(item) for item in post_list)
                    if c_pre and c_post:
                        pre_installed = True
                        post_installed = True
                        active_path = cand
                        break
                    pre_installed = pre_installed or c_pre
                    post_installed = post_installed or c_post
            except Exception:
                pass

        return {
            "bot": canon,
            "label": SUPPORTED_BOTS[canon]["label"],
            "scope": scope,
            "config_path": str(active_path),
            "file_exists": active_path.is_file(),
            "pre_tool_use": pre_installed,
            "post_tool_use": post_installed,
            "installed": bool(pre_installed and post_installed),
            "schema_tokens_added": 0,
            "preserved_keys": existing_keys,
        }  # [direct-edit]

    def status(
        self,
        bots: Optional[List[str]] = None,
        scope: str = "workspace",
    ) -> Dict[str, Any]:
        """Audit Organ 05 zero-token hook installation status across target bots."""
        target_bots = [
            BOT_ALIASES.get(b.strip().lower(), b.strip().lower())
            for b in (bots or list(SUPPORTED_BOTS.keys()))
        ]
        bot_reports: Dict[str, Dict[str, Any]] = {}
        for b in target_bots:
            if b in SUPPORTED_BOTS:
                bot_reports[b] = self.inspect_bot(b, scope=scope)

        installed_count = sum(1 for r in bot_reports.values() if r.get("installed"))
        return {
            "scope": scope,
            "workspace_dir": str(self.workspace_dir),
            "home_dir": str(self.home_dir),
            "hook_command": self.hook_cmd,
            "schema_tokens_added": 0,
            "installed_count": installed_count,
            "total_bots": len(bot_reports),
            "bots": bot_reports,
        }

    def install_bot(
        self,
        bot_key: str,
        scope: str = "workspace",
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Idempotently wire Organ 05 `PreToolUse` and `PostToolUse` hooks into a bot's JSON config
        without clobbering existing keys or non-WBC hooks.
        """
        canon = BOT_ALIASES.get(bot_key.strip().lower(), bot_key.strip().lower())
        cfg_path = self._resolve_bot_config_path(canon, scope=scope)

        if scope == "workspace":
            user_status = self.inspect_bot(canon, scope="user")
            if user_status.get("installed"):
                return {
                    "bot": canon,
                    "label": SUPPORTED_BOTS[canon]["label"],
                    "scope": scope,
                    "config_path": str(user_status["config_path"]),
                    "action": "already_wired_globally",
                    "dry_run": dry_run,
                    "pre_tool_use": True,
                    "post_tool_use": True,
                    "installed": True,
                    "schema_tokens_added": 0,
                    "preserved_keys": user_status.get("preserved_keys", []),
                }

        existing: Dict[str, Any] = {}
        action = "created"
        if cfg_path.is_file():
            action = "updated"
            try:
                loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing = loaded
            except Exception:
                existing = {}

        before_snapshot = json.dumps(existing, sort_keys=True)

        hooks_dict = existing.get("hooks")
        if not isinstance(hooks_dict, dict):
            hooks_dict = {}
            existing["hooks"] = hooks_dict

        pre_entry = {
            "matcher": MULTI_HARNESS_TOOL_MATCHER,
            "hooks": [
                {
                    "type": "command",
                    "command": self.hook_cmd,
                    "timeout": 10,
                }
            ],
        }
        post_entry = {
            "matcher": MULTI_HARNESS_TOOL_MATCHER,
            "hooks": [
                {
                    "type": "command",
                    "command": f"{self.hook_cmd} --phase post",
                    "timeout": 10,
                }
            ],
        }

        for phase_key, new_entry in (("PreToolUse", pre_entry), ("PostToolUse", post_entry)):
            current_list = hooks_dict.get(phase_key)
            if not isinstance(current_list, list):
                current_list = []
            filtered = [item for item in current_list if not _is_wbc_hook_entry(item)]
            filtered.append(new_entry)
            hooks_dict[phase_key] = filtered

        after_snapshot = json.dumps(existing, sort_keys=True)
        if action == "updated" and before_snapshot == after_snapshot:
            action = "verified"

        if not dry_run and action in ("created", "updated"):
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
            if canon == "antigravity":
                agy_hooks_path = (
                    (self.home_dir / ".gemini" / "config" / "hooks.json")
                    if scope == "user"
                    else (self.workspace_dir / ".agents" / "hooks.json")
                )
                agy_existing: Dict[str, Any] = {}
                if agy_hooks_path.is_file():
                    try:
                        loaded_agy = json.loads(agy_hooks_path.read_text(encoding="utf-8"))
                        if isinstance(loaded_agy, dict):
                            agy_existing = loaded_agy
                    except Exception:
                        agy_existing = {}
                gate_block = agy_existing.get("whole-body-immune-gate")
                if not isinstance(gate_block, dict):
                    gate_block = {"enabled": True}
                    agy_existing["whole-body-immune-gate"] = gate_block
                gate_block["enabled"] = True
                gate_block["PreInvocation"] = [
                    {
                        "type": "command",
                        "command": f"{self.hook_cmd} --phase pre-invocation",
                        "timeout": 5,
                    }
                ]
                gate_block["PreToolUse"] = [pre_entry]
                gate_block["PostToolUse"] = [post_entry]
                agy_hooks_path.parent.mkdir(parents=True, exist_ok=True)
                agy_hooks_path.write_text(json.dumps(agy_existing, indent=2) + "\n", encoding="utf-8")

        return {
            "bot": canon,
            "label": SUPPORTED_BOTS[canon]["label"],
            "scope": scope,
            "config_path": str(cfg_path),
            "action": action,
            "dry_run": dry_run,
            "pre_tool_use": True,
            "post_tool_use": True,
            "installed": True,
            "schema_tokens_added": 0,
            "preserved_keys": list(existing.keys()),
        }

    def install(
        self,
        bots: Optional[List[str]] = None,
        scope: str = "workspace",
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Install zero-token PreToolUse/PostToolUse hooks across requested bots. # [direct-edit]"""
        try:
            from whole_body_compute.local.scout import prewarm_scout

            prewarm_scout(background=True)
        except Exception:
            pass

        target_bots = [
            BOT_ALIASES.get(b.strip().lower(), b.strip().lower())
            for b in (bots or ["antigravity", "claude-code"])
        ]
        results: Dict[str, Dict[str, Any]] = {}
        for b in target_bots:
            if b not in SUPPORTED_BOTS:
                raise ValueError(f"Unsupported bot '{b}'. Supported: {', '.join(SUPPORTED_BOTS.keys())}")
            results[b] = self.install_bot(b, scope=scope, dry_run=dry_run)

        return {
            "scope": scope,
            "workspace_dir": str(self.workspace_dir),
            "home_dir": str(self.home_dir),
            "hook_command": self.hook_cmd,
            "dry_run": dry_run,
            "schema_tokens_added": 0,
            "installed_count": len(results),
            "total_bots": len(results),
            "bots": results,
        }


def install_shield_hooks(
    bots: Optional[List[str]] = None,
    scope: str = "workspace",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Top-level helper to install Organ 05 hooks and prewarm Local Metal Scout. # [direct-edit]"""
    try:
        from whole_body_compute.local.scout import prewarm_scout

        prewarm_scout(background=True)
    except Exception:
        pass
    installer = ShieldInstaller()
    return installer.install(bots=bots, scope=scope, dry_run=dry_run)


def format_shield_status(report: Dict[str, Any]) -> str:
    """Render a human-readable status & wiring report for `body shield`."""
    scope = report.get("scope", "workspace")
    hook_cmd = report.get("hook_command", "")
    bots = report.get("bots") or {}
    installed_count = int(report.get("installed_count", 0))
    total_bots = int(report.get("total_bots", len(bots)))

    lines = [
        "🛡️  WHOLE BODY COMPUTE - THE BLOOD-BRAIN BARRIER SHIELD",
        "─" * 76,
        f"  • Scope:               {scope} ({report.get('workspace_dir', '')})",
        f"  • Hook Handler:        {hook_cmd}",
        "  • MCP Schema Bloat:    0 tokens (Pure OS/IDE PreToolUse & PostToolUse gate)",
        f"  • Active Harnesses:    {installed_count}/{total_bots} protected",
        "─" * 76,
    ]
    for bot_key, info in bots.items():
        status_icon = "✅ ACTIVE" if info.get("installed") else "⚪ UNWIRED"
        action_tag = f" [{info['action'].upper()}]" if "action" in info else ""
        pre_s = "Pre:✓" if info.get("pre_tool_use") else "Pre:✗"
        post_s = "Post:✓" if info.get("post_tool_use") else "Post:✗"
        lines.append(
            f"  • {info.get('label', bot_key):<28} {status_icon}{action_tag} ({pre_s}, {post_s})\n"
            f"    └─ {info.get('config_path', '')}"
        )
    lines.append("─" * 76)
    try:
        from whole_body_compute.local.reflex_gate import (
            REGISTERED_REFLEX_GATES,
            get_reflex_gate_status_summary,
        )

        rg_status = get_reflex_gate_status_summary()
        gates = rg_status.get("gates") or []
        min_s = int(rg_status.get("min_samples", 50))
        ws_repo_name = Path(str(report.get("workspace_dir") or Path.cwd())).name
        lines.append("⚡ Spinal Reflex Gates (Online Behavioral Distillation):")

        def _gate_sort_key(item: Dict[str, Any]) -> tuple:
            st_rank = 0 if item.get("state") == "GRADUATED" else (1 if item.get("state") == "TRIPPED" else 2)
            repo_match = 0 if item.get("repo_key") in (ws_repo_name, "__global__") else 1
            return (st_rank, repo_match, -int(item.get("sample_count", 0)), str(item.get("gate_id", "")))

        by_gate: Dict[str, List[Dict[str, Any]]] = {}
        for g in gates:
            by_gate.setdefault(str(g.get("gate_id", "unknown")), []).append(g)

        displayed_rows: List[Dict[str, Any]] = []
        shown_ids: set[str] = set()
        # Ensure at least the top row for every active gate_id is visible first
        for gid in sorted(by_gate.keys()):
            ranked = sorted(by_gate[gid], key=_gate_sort_key)
            displayed_rows.append(ranked[0])
            by_gate[gid] = ranked[1:]
            shown_ids.add(gid)

        remaining = [row for rem in by_gate.values() for row in rem]
        remaining.sort(key=_gate_sort_key)
        displayed_rows.extend(remaining[: max(0, 6 - len(displayed_rows))])

        for g in displayed_rows:
            st = str(g["state"])
            badge = "🟢 GRADUATED" if st == "GRADUATED" else ("🔴 TRIPPED" if st == "TRIPPED" else "🟡 SHADOW")
            cv_pct = float(g["cv_accuracy"]) * 100.0
            label = f"{g['gate_id']} [{g['repo_key']}]"
            lines.append(
                f"  • {label:<28} {badge} ({g['sample_count']}/{min_s} samples • CV:{cv_pct:.1f}% • misses:{g['consecutive_misses']})"
            )
        for reg_gate_id, wait_desc in REGISTERED_REFLEX_GATES:
            if reg_gate_id not in shown_ids:
                lines.append(
                    f"  • {reg_gate_id:<28} 🟡 SHADOW (0/{min_s} samples • {wait_desc})"
                )
        lines.append("─" * 76)
    except Exception:
        pass
    lines.append(
        "💡 Tip: Run `body shield --install --all` to wire all bots, or `body savings` for live ROI receipt."
    )
    return "\n".join(lines)
