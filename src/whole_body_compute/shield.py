"""
Whole Body Compute - Organ 05 Blood-Brain Barrier Shield Installer & Live Savings Ledger (`body shield` & `body savings`).
# [direct-edit]

Provides:
1. `ShieldTelemetryLedger` & `record_shield_event`:
   Sub-2ms, thread-safe SQLite WAL telemetry ledger (`~/.voicefi/dot_memory.db` table
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

MODEL_INPUT_PRICING_PER_M: Dict[str, float] = {
    "gemini 4 argon (high)": 1.25,
    "gemini 4 carbon": 1.25,
    "gemini-4-carbon": 1.25,
    "carbon": 1.25,
    "gemini-4": 1.25,
    "gemini-pro": 1.25,
    "gemini flash": 0.15,
    "flash": 0.15,
    "claude haiku": 0.80,
    "haiku": 0.80,
    "claude sonnet": 3.00,
    "sonnet": 3.00,
    "claude opus": 15.00,
    "opus": 15.00,
    "grok": 2.00,
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


def _lookup_model_rate(model_label: str) -> float:
    """Resolve input USD per 1M tokens for a given model name."""
    low = (model_label or "").strip().lower()
    if not low:
        return 1.25
    for k, rate in MODEL_INPUT_PRICING_PER_M.items():
        if k in low:
            return rate
    if "opus" in low:
        return 15.00
    if "sonnet" in low:
        return 3.00
    if "haiku" in low:
        return 0.80
    if "flash" in low:
        return 0.15
    if "grok" in low:
        return 2.00
    if "gpt" in low or "codex" in low or "o3" in low or "o1" in low:
        return 2.50
    if "gemini" in low or "argon" in low or "carbon" in low:
        return 1.25
    return 1.25


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

    brain_root = Path.home() / ".gemini" / "antigravity" / "brain"
    if brain_root.is_dir():
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

        model_sel_re = re.compile(
            r"Model Selection:\s*Changed model from [^\n\"\\]+? to ((?:Gemini|Claude|GPT|Grok|o1|o3)[A-Za-z0-9_. ()\-]{1,40})"
        )  # [direct-edit]
        for cdir in cand_dirs:
            t_file = cdir / ".system_generated" / "logs" / "transcript.jsonl"
            if not t_file.is_file():
                continue
            try:
                txt_all = t_file.read_text(encoding="utf-8", errors="replace")
                matches = model_sel_re.findall(txt_all)
                if matches:
                    detected = matches[-1].strip()
                    if detected:
                        return detected, _lookup_model_rate(detected)
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
    Normalize a shield target into a clean repo-relative file path, app handle, or shell command
    without mangling multi-argument CLI commands via `Path(...).name`.
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
        # Already a clean workspace-relative path (no leading slash)
        if not norm.startswith(("/", "~")):
            while norm.startswith("./"):
                norm = norm[2:]
            return (norm or raw)[:140]
        try:
            p = Path(os.path.expanduser(raw)).resolve()
            for root_cand in (WBC_REPO_ROOT.resolve(), Path.cwd().resolve()):
                try:
                    rel = p.relative_to(root_cand)
                    return str(rel).replace("\\", "/")[:140]
                except ValueError:
                    continue
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
        if saved_tok <= 0 and ev == "hands_ax_act":
            return -actual_sec
        # Avoids screenshot -> vision coordinate grounding -> click -> verify loop (~4.5s)
        counterfactual_sec = 4.5 if ev == "hands_ax_act" else 1.5
        return counterfactual_sec - actual_sec

    if ev in ("local_metal_scout", "screen_local_digest"):
        cli_overhead_sec = 0.35 if ev == "local_metal_scout" else 0.0
        if saved_tok <= 0:
            return -(actual_sec + cli_overhead_sec)
        # Only files exceeding the native single-read ceiling (>=11,520 tok) avoid a pagination/recon LLM turn (~3.2s)
        avoided_turn_sec = 3.2 if raw_tok >= 11_520 else 0.8
        prefill_sec = saved_tok / 75_000.0
        return avoided_turn_sec + prefill_sec - actual_sec - cli_overhead_sec

    if ev == "ast_slice_gate":
        if mode == "hard_deny":
            # Hard-blocking forces an extra LLM tool recovery round-trip (~4.5s penalty)
            return (saved_tok / 75_000.0) - 4.5 - actual_sec
        if saved_tok <= 0:
            return -actual_sec
        avoided_pagination_sec = 3.2 if raw_tok >= 11_520 else 0.6
        prefill_sec = saved_tok / 75_000.0
        return avoided_pagination_sec + prefill_sec - actual_sec

    if ev == "crash_triage_fusion":
        if saved_tok <= 0:
            return -actual_sec
        # Eliminates 1 full exploratory `read_file` LLM round-trip (~5.2s TTFT + tool decode) after a crash/test failure
        return 5.2 + (saved_tok / 75_000.0) - actual_sec

    if ev == "vascular_block":
        return (1.5 - actual_sec) if mode != "hard_deny" else (-1.5 - actual_sec)

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
        if self._initialized and self.db_path.exists():
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
                conn.commit()
                self._initialized = True
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
                finally:
                    conn.close()
            return True
        except Exception:
            return False

    def reset(self) -> int:
        """Delete all rows in `shield_savings_ledger` and return the deleted count."""
        try:
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
        """Resolve `immune_audit.jsonl` path for reconciling POSIX `sh` fast-path auto-slices."""
        env_p = (os.environ.get("VOICEFI_IMMUNE_AUDIT_PATH") or os.environ.get("WBC_IMMUNE_AUDIT_PATH") or "").strip()
        if env_p:
            return Path(os.path.expanduser(env_p))
        if self._explicit_db_path:
            # Only auto-reconcile sibling audit logs when explicitly placed in the same dir or via env var
            sibling = self.db_path.parent / "immune_audit.jsonl"
            return sibling if sibling.is_file() else None
        if (
            os.environ.get("PYTEST_CURRENT_TEST")
            or os.environ.get("VOICEFI_STRICT_TEST_HOOK") == "1"
            or os.environ.get("WBC_BENCHMARK_MODE") == "1"
            or os.environ.get("WBC_ARENA_MODE") == "1"
        ):
            return None
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
        Sync POSIX `sh` fast-path `read_auto_sliced` records from `immune_audit.jsonl` into
        `shield_savings_ledger` (idempotent, zero runtime overhead added to the <15ms shell hook).
        """
        audit_file = self._resolve_fast_path_audit_log()
        if audit_file is None or not audit_file.is_file():
            return 0
        try:
            raw_lines = audit_file.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return 0

        candidates: List[Tuple[float, str, str, str, str, int, int, int, float]] = []
        for line in raw_lines:
            if '"read_auto_sliced":true' not in line and '"read_auto_sliced": true' not in line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if not ev.get("fast_path") or not ev.get("read_auto_sliced"):
                continue
            saved_tok = max(0, int(ev.get("direct_tokens_saved") or 0))
            if saved_tok <= 0:
                continue
            ts_val = float(ev.get("timestamp") or 0.0)
            tool_nm = str(ev.get("tool") or "read_file")
            bot_nm = "claude-code" if tool_nm == "Read" else "antigravity"
            model_nm = "Claude Sonnet" if bot_nm == "claude-code" else "Gemini 4 Carbon"
            raw_tgt = str(ev.get("target") or "unknown")
            if not self._explicit_db_path and _is_ephemeral_temp_target(raw_tgt):
                continue
            norm_tgt = _normalize_target_label(raw_tgt)
            comp_tok = 1_600  # 160-line auto-sliced window + AST symbol outline
            raw_tok = saved_tok + comp_tok
            lat_ms = round(max(0.0, float(ev.get("latency_ms") or 4.2)), 3)
            sess_id = str(ev.get("conversation_id") or "").strip()
            if not sess_id or sess_id == "global":
                sess_id = "default-session"
            candidates.append((ts_val, bot_nm, sess_id, model_nm, norm_tgt, raw_tok, comp_tok, saved_tok, lat_ms))

        if not candidates:
            return 0

        self.ensure_schema()
        inserted = 0
        with _LEDGER_LOCK:
            conn = self._connect()
            try:
                existing_keys = {
                    (round(float(r[0]), 1), str(r[1]), str(r[2]))
                    for r in conn.execute(
                        "SELECT timestamp, bot_name, target FROM shield_savings_ledger WHERE action_mode = 'sh_auto_slice';"
                    ).fetchall()
                }
                for ts_val, bot_nm, sess_id, model_nm, norm_tgt, raw_tok, comp_tok, saved_tok, lat_ms in candidates:
                    key = (round(ts_val, 1), bot_nm, norm_tgt)
                    if key in existing_keys:
                        continue
                    conn.execute(
                        """
                        INSERT INTO shield_savings_ledger (
                            timestamp, bot_name, session_id, model_name, event_type, action_mode,
                            target, raw_tokens_est, compressed_tokens_est, tokens_saved,
                            secrets_blocked, latency_ms
                        ) VALUES (?, ?, ?, ?, 'ast_slice_gate', 'sh_auto_slice', ?, ?, ?, ?, 0, ?);
                        """,
                        (ts_val, bot_nm, sess_id, model_nm, norm_tgt, raw_tok, comp_tok, saved_tok, lat_ms),
                    )
                    existing_keys.add(key)
                    inserted += 1
                if inserted > 0:
                    conn.commit()
            finally:
                conn.close()
        return inserted

    def get_summary(
        self,
        bot_filter: Optional[str] = None,
        price_per_m: Optional[float] = None,
        session_gap_seconds: float = 7200.0,
        window: Optional[str] = None,
        top_limit: int = 15,
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
                    ORDER BY id ASC;
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

        detected_model, detected_rate = detect_active_model_and_rate()
        if recent_rows:
            last_m = str(recent_rows[0]["model_name"] or "").strip()
            if last_m and any(
                k in last_m.lower()
                for k in ("gemini", "claude", "gpt", "grok", "sonnet", "opus", "haiku", "flash", "carbon", "argon", "o1", "o3")
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

        # Per-(bot_name, session_id, target) high-water-mark deduplication for file/log reads
        dedup_credits: Dict[Tuple[str, str, str], int] = {}

        bot_agg: Dict[str, Dict[str, Any]] = {}
        event_agg: Dict[str, Dict[str, Any]] = {}
        target_agg: Dict[str, Dict[str, Any]] = {}
        session_meta_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        session_groups: Dict[Tuple[str, str], List[Tuple[float, int, float]]] = {}

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

            if ev_t in _DEDUP_FILE_EVENT_TYPES and tgt_name:
                d_key = (b_name, s_id, tgt_name)
                prev_credited = dedup_credits.get(d_key, 0)
                ceiling_saved = max(prev_credited, saved_db)
                eff_saved = min(saved_db, max(0, ceiling_saved - prev_credited))
                dedup_credits[d_key] = prev_credited + eff_saved
                eff_raw = (comp_db + eff_saved) if raw_db >= comp_db else raw_db
            else:
                eff_saved = saved_db
                eff_raw = raw_db

            row_rate = (
                float(price_per_m)
                if price_per_m is not None
                else _lookup_model_rate(m_name or detected_model)
            )
            row_direct_usd = (eff_saved / 1_000_000.0) * row_rate
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
            secrets_blocked += sec_db
            all_lats.append(lat_v)
            total_direct_usd += row_direct_usd
            total_time_saved_sec += ev_time_sec

            b_st = bot_agg.setdefault(
                b_name,
                {
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
            b_st["events"] += 1
            b_st["raw_tokens"] += eff_raw
            b_st["compressed_tokens"] += comp_db
            b_st["tokens_saved"] += eff_saved
            b_st["secrets_blocked"] += sec_db
            b_st["lats"].append(lat_v)
            b_st["time_sec"] += ev_time_sec
            b_st["direct_usd"] += row_direct_usd

            e_st = event_agg.setdefault(
                ev_t,
                {
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
            e_st["events"] += 1
            e_st["raw_tokens"] += eff_raw
            e_st["compressed_tokens"] += comp_db
            e_st["tokens_saved"] += eff_saved
            e_st["secrets_blocked"] += sec_db
            e_st["lats"].append(lat_v)
            e_st["time_sec"] += ev_time_sec
            e_st["direct_usd"] += row_direct_usd

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
            elif ev_t in ("ast_slice_gate", "secret_redaction", "vascular_block", "crash_triage_fusion"):
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

            session_groups.setdefault((b_name, s_id), []).append((ts_val, eff_saved, row_rate))

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

        for (b_name, s_id), ev_list in session_groups.items():
            transcript_turns = _count_transcript_user_turns(s_id, bot_name=b_name)
            avg_sess_rate = sum(r for _, _, r in ev_list) / max(1, len(ev_list))
            # Coalesce multi-tool bursts (<25s) in real sessions or >5-event synthetic runs
            if transcript_turns is not None or s_id not in ("default-session", "local") or len(ev_list) > 5:
                turn_buckets: List[int] = []
                last_bucket_ts = -1.0
                for ts_val, s_tok, _ in ev_list:
                    if not turn_buckets or (ts_val - last_bucket_ts) >= 25.0:
                        turn_buckets.append(s_tok)
                        last_bucket_ts = ts_val
                    else:
                        turn_buckets[-1] += s_tok
                if transcript_turns is not None and len(turn_buckets) > transcript_turns:
                    merged: List[int] = [0] * transcript_turns
                    for idx_b, val_b in enumerate(turn_buckets):
                        target_idx = min(transcript_turns - 1, int(idx_b * transcript_turns / len(turn_buckets)))
                        merged[target_idx] += val_b
                    turn_buckets = merged
            else:
                turn_buckets = [s_tok for _, s_tok, _ in ev_list]

            n_turns = len(turn_buckets)
            sess_cum = 0
            sess_direct = sum(turn_buckets)
            for idx_z, s_k in enumerate(turn_buckets):
                sess_cum += (n_turns - idx_z) * s_k

            sess_reingest_tok = max(0, sess_cum - sess_direct)
            sess_reingest_usd = (sess_reingest_tok / 1_000_000.0) * (avg_sess_rate * 0.10)

            session_cum_map[(b_name, s_id)] = (sess_cum, n_turns, sess_reingest_usd)
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
        api_dollars_saved_est = round(total_direct_usd + total_reingest_usd, 2)

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
            b_usd = round(float(b_st["direct_usd"]) + float(bot_reingest_usd_map.get(b_name, 0.0)), 2)
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
            e_usd = round(float(e_st["direct_usd"]), 2)
            e_avg_lat = round(sum(e_st["lats"]) / len(e_st["lats"]), 1) if e_st["lats"] else 0.0
            ev_entry = {
                "rank": idx_ev,
                "event_type": e_type,
                "events": int(e_st["events"]),
                "raw_tokens": int(e_st["raw_tokens"]),
                "compressed_tokens": int(e_st["compressed_tokens"]),
                "tokens_saved": e_saved,
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

        return {
            "db_path": str(self.db_path),
            "window": window_label,
            "since_timestamp": since_ts,
            "bot_filter": norm_bot,
            "active_model": detected_model,
            "total_events": total_events,
            "actual_turns_observed": actual_turns_observed,
            "raw_tokens_blocked": raw_tokens_blocked,
            "compressed_tokens_sent": compressed_tokens_sent,
            "direct_tokens_saved": direct_tokens_saved,
            "compression_pct": compression_pct,
            "cumulative_tokens_saved": cumulative_tokens_saved,
            "quadratic_10turn_tokens_saved": cumulative_tokens_saved,
            "price_per_m_usd": effective_price_per_m,
            "api_dollars_saved_direct": api_dollars_saved_direct,
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


def is_scout_warm() -> bool:
    """Fast check whether Local Metal Scout (:9379 voicefi-coder-mlx or Ollama) is actively resident in RAM. # [direct-edit]"""
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
) -> Dict[str, Any]:
    """Return structured shield telemetry summary & leaderboard for `body savings` / `body leaderboard`. # [direct-edit]"""
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
                    "estimated_usd_saved": float(summary.get("estimated_usd_saved", 0.0)),
                    "secrets_scrubbed": int(summary.get("secrets_scrubbed", 0)),
                    "wall_time_saved_sec": float(summary.get("wall_time_saved_sec", 0.0)),
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
    scout_cnt = int(summary.get("local_scout_events", 0))
    is_warm = bool(summary.get("is_scout_warm", False))
    if scout_cnt == 0:
        scout_line_val = "0.0 ms (no LLM digests yet)"
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

    width = 74

    def _box_line(text: str = "") -> str:
        if len(text) > width - 4:
            text = text[: width - 7] + "..."
        return f"│  {text:<{width - 4}}│"

    hdr_suffix = f" [{win_label}]" if win_label != "ALL" else ""
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
        _box_line(f"Est. API Dollars Saved: ${usd_est:,.2f} (direct + 90%-cached @ ${price_rate:.2f}/1M)"),
        _box_line(f"Wall-Clock Time Saved:  {time_disp} net agent wait avoided"),
        _box_line(f"Subscription Runway:    {mult:.1f}x longer before 5-hour cap"),
        _box_line(f"Secrets Scrubbed:       {sec_blocked:,} leaked keys blocked"),
        _box_line(f"Hook Reflex Latency:    {hook_lat:.1f} ms ($0.00 local compute)"),
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
            tgt = str(row["target"])
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
        "🛡️  WHOLE BODY COMPUTE - ORGAN 05 BLOOD-BRAIN BARRIER SHIELD",
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
    lines.append(
        "💡 Tip: Run `body shield --install --all` to wire all bots, or `body savings` for live ROI receipt."
    )
    return "\n".join(lines)
