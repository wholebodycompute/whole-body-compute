"""
Headless Goose CLI Runner & Live Activity Watcher for VoiceFi (Zero-Egress).

Wires VoiceFi ("Ears & Mouth") + Sovereign Foveal Vision & Episodic Memory ("Eyes") + Goose CLI ("Hands")
on top of a local Ollama/MLX LLM ("Brain" at 127.0.0.1:11434) with:
- Zero cloud API keys / zero external network egress
- Zero macOS window focus stealing (100% background subprocess execution)
- Live activity watching via Goose's `stream-json` event stream and `~/.local/share/goose/sessions/sessions.db`
- Automatic deictic Foveal Vision episodic OCR/audio enrichment ("this error", "on my screen", "that meeting")
- Strict <25-word spoken TTS summary normalization
"""

from __future__ import annotations

import ast
import datetime
import errno
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import signal
import sqlite3
import subprocess
import tempfile
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from whole_body_compute.organs.immune_polyglot import (
    POLYGLOT_EXTENSIONS,
    POLYGLOT_SECGATE_EXTENSIONS,
    PROTECTED_POLYGLOT_CONFIG_FILES,
    run_polyglot_three_stage_secgate,
)

from whole_body_compute.integrations.screenpipe import (
    DEFAULT_SCREENPIPE_DB,
    DEFAULT_SCREENPIPE_URL,
    extract_visible_errors,
    search_screen_content,
)


def _clean_text_for_speech(text: str, max_words: int = 24) -> str:
    """Clean markdown and code syntax for spoken TTS without requiring audio/sounddevice imports."""
    try:
        from voicefi.integrations.antigravity import clean_markdown_for_speech

        return clean_markdown_for_speech(text, max_words=max_words)
    except Exception:
        cleaned = re.sub(r"```[\s\S]*?```", " ", text)
        cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
        cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned)
        cleaned = re.sub(r"[#*_~>|]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        words = cleaned.split()
        if len(words) > max_words:
            cleaned = " ".join(words[:max_words]).rstrip(",;:-") + "."
        return cleaned

logger = logging.getLogger("whole_body_compute.integrations.goose_runner")

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_SPECIALIST_ROUTER_URL = "http://127.0.0.1:9380/v1/chat/completions"
DEFAULT_GOOSE_SESSIONS_DB = Path.home() / ".local" / "share" / "goose" / "sessions" / "sessions.db"
DEFAULT_VIFI_SOCKET = "/tmp/vifi.sock"
DEFAULT_GOOSE_ACTIVE_PID_FILE = Path("/tmp/voicefi_goose_active.json")

PROTECTED_GRADER_PATTERNS = (
    "eval_models.py",
    "evaluate.py",
    "eval_localbench.py",
    "BENCHMARK_REPORT.md",
)

# Preferred local models in priority order (configurable via GOOSE_MODEL)
PREFERRED_LOCAL_MODELS = (
    "qwen3:32b",
    "gemma4:27b",
    "qwen2.5-coder:1.5b",
    "llama3.2:1b",
    "nimble:latest",
    "llama3.3:70b",
    "tev1:latest",
)

DEICTIC_SCREEN_PATTERNS = re.compile(
    r"\b("
    r"this error|on my screen|from my screen|my screen|terminal error|latest error|"
    r"stack trace|what i was just looking at|what was i looking at|on screen|"
    r"that meeting|last call|post[- ]call|action items from|"
    r"this button|this input|this field|active window|focused window|active app"
    r")\b",
    re.IGNORECASE,
)

ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
GOOSE_BANNER_RE = re.compile(
    r"^\s*(?:__\( O\)>|\\____\)|L L\s+goose is ready|●\s+new session|session id:|working dir:).*$",
    re.IGNORECASE,
)


@dataclass
class GooseToolEvent:
    """Represents a live tool call or status event observed from Goose."""

    session_id: str
    call_id: str
    extension: str
    tool_name: str
    arguments: Dict[str, Any]
    status: str  # "started", "completed", "error"
    summary: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "call_id": self.call_id,
            "extension": self.extension,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "status": self.status,
            "summary": self.summary,
            "timestamp": self.timestamp,
        }


@dataclass
class GooseTurnResult:
    """Structured result from a headless Goose execution."""

    success: bool
    spoken_summary: str
    raw_response: str
    session_id: str
    model: str
    recipe_used: Optional[str] = None
    tool_events: List[GooseToolEvent] = field(default_factory=list)
    duration_seconds: float = 0.0
    error: Optional[str] = None
    sandbox_verified: bool = True
    secgate_report: Optional[Dict[str, Any]] = None
    interrupted: bool = False
    dot_memories_injected: int = 0
    tactile_delta: Optional[Dict[str, Any]] = None
    foveal_snapshot: Optional[Dict[str, Any]] = None
    bridge_crumb: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "spoken_summary": self.spoken_summary,
            "raw_response": self.raw_response,
            "session_id": self.session_id,
            "model": self.model,
            "recipe_used": self.recipe_used,
            "tool_events": [e.to_dict() for e in self.tool_events],
            "duration_seconds": round(self.duration_seconds, 3),
            "error": self.error,
            "sandbox_verified": self.sandbox_verified,
            "secgate_report": self.secgate_report,
            "interrupted": self.interrupted,
            "dot_memories_injected": self.dot_memories_injected,
            "tactile_delta": self.tactile_delta,
            "foveal_snapshot": self.foveal_snapshot,
            "bridge_crumb": self.bridge_crumb,
        }


def find_goose_binary() -> Optional[str]:
    """Locate the `goose` CLI binary on macOS."""
    found = shutil.which("goose")
    if found and os.access(found, os.X_OK):
        return found
    for cand in (
        Path("/opt/homebrew/bin/goose"),
        Path("/usr/local/bin/goose"),
        Path.home() / ".local" / "bin" / "goose",
        Path.home() / ".cargo" / "bin" / "goose",
    ):
        if cand.is_file() and os.access(str(cand), os.X_OK):
            return str(cand)
    return None


def resolve_local_ollama_model(
    preferred: Optional[str] = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> str:
    """
    Query HybridRouter and local endpoints (:9380 SpecialistRouter / gemma4-26b,
    :9379 voicefi-coder-mlx, or Ollama) to pick a thermally-approved local model,
    falling back to preferred or qwen3:32b.
    """
    env_model = preferred or os.environ.get("GOOSE_MODEL") or os.environ.get("VIFI_GOOSE_MODEL")
    try:
        from whole_body_compute.local.router import default_router

        decision = default_router.resolve_model(task_type="coder", preferred_model=env_model)
        if decision and decision.selected_model:
            cand = decision.selected_model.replace("cloud:", "").strip()
            if cand:
                return cand
    except Exception as e:
        logger.debug(f"HybridRouter model resolution error: {e}")

    try:
        req = urllib.request.Request(f"{ollama_url.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            installed = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
            if env_model and env_model in installed:
                return env_model
            for candidate in PREFERRED_LOCAL_MODELS:
                if candidate in installed:
                    return candidate
            if installed:
                # Prefer non-embedding models
                non_embed = [m for m in installed if "embed" not in m.lower()]
                return non_embed[0] if non_embed else installed[0]
    except Exception:
        pass
    return env_model or "qwen3:32b"


def format_goose_tool_announcement(extension: str, tool_name: str, arguments: Dict[str, Any]) -> str:
    """Produce a concise, human-readable status string for what Goose is doing right now."""
    ext = (extension or "goose").strip()
    tname = (tool_name or "tool").strip()
    if tname == "shell" or tname.endswith("__shell"):
        cmd = str(arguments.get("command") or "").strip()
        short_cmd = (cmd[:42] + "...") if len(cmd) > 45 else cmd
        return f"Goose [{ext}]: running shell `{short_cmd}`" if short_cmd else f"Goose [{ext}]: running shell command"
    if tname in ("edit", "str_replace", "write") or tname.endswith(("__edit", "__write")):
        path = str(arguments.get("path") or arguments.get("file_path") or "").strip()
        base = Path(path).name if path else "file"
        return f"Goose [{ext}]: updating {base}"
    if "screen" in tname or "ocr" in tname or "terminal_error" in tname:
        q = str(arguments.get("q") or arguments.get("query") or arguments.get("app_name") or "screen").strip()
        return f"Goose [screenpipe]: scanning OCR for '{q[:30]}'"
    if "audio" in tname:
        return "Goose [screenpipe]: pulling recent audio transcript"
    return f"Goose [{ext}]: calling {tname}"


def enforce_spoken_word_budget(text: str, max_words: int = 24) -> str:
    """
    Strip noisy logs/markdown/JSON wrappers and cap response to <25 words for TTS playback.
    """
    if not text:
        return ""

    cleaned = ANSI_ESCAPE_RE.sub("", text).strip()

    # If the model wrapped its answer in a JSON object like {"name": "...", "summary": "..."}, unwrap it
    if cleaned.startswith("{") and cleaned.endswith("}"):
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, dict):
                for key in ("spoken_summary", "summary", "response", "text", "result", "message", "name"):
                    val = obj.get(key)
                    if isinstance(val, str) and val.strip():
                        cleaned = val.strip()
                        break
        except Exception:
            pass

    # Filter out Goose ASCII banner lines and raw JSON event lines
    kept_lines: List[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if GOOSE_BANNER_RE.match(line):
            continue
        if line.startswith('{"type":') and line.endswith("}"):
            continue
        if line.startswith("```"):
            continue
        kept_lines.append(line)

    merged = " ".join(kept_lines).strip()
    spoken = _clean_text_for_speech(merged, max_words=max_words) if merged else ""
    if not spoken:
        spoken = re.sub(r"\s+", " ", merged).strip()

    words = spoken.split()
    if len(words) > max_words:
        spoken = " ".join(words[:max_words]).rstrip(",;:-") + "."
    return spoken


def parse_goose_output_stream(
    raw_output: str,
    session_id: str = "headless",
    on_tool_event: Optional[Callable[[GooseToolEvent], None]] = None,
) -> Tuple[str, List[GooseToolEvent]]:
    """
    Parse output from `goose run` (supports `--output-format stream-json`, `--output-format json`,
    and plain text output), stripping all banners and tool logs and returning `(final_text, tool_events)`.
    """
    tool_events: List[GooseToolEvent] = []
    text_chunks_by_msg: Dict[str, List[str]] = {}
    msg_order: List[str] = []
    plain_lines: List[str] = []

    raw_stripped = ANSI_ESCAPE_RE.sub("", raw_output or "").strip()
    if not raw_stripped:
        return "", tool_events

    # Case 1: Full JSON document from `--output-format json`
    json_start = raw_stripped.find("{\n  \"messages\":")
    if json_start != -1:
        try:
            doc = json.loads(raw_stripped[json_start:])
            if isinstance(doc, dict) and isinstance(doc.get("messages"), list):
                last_assistant_text = ""
                for msg in doc["messages"]:
                    role = msg.get("role")
                    for item in msg.get("content") or []:
                        ctype = item.get("type")
                        if ctype == "toolRequest":
                            tc = (item.get("toolCall") or {}).get("value") or {}
                            meta = item.get("_meta") or {}
                            ext = str(meta.get("goose_extension") or "developer")
                            tname = str(tc.get("name") or "tool")
                            targs = tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {}
                            evt = GooseToolEvent(
                                session_id=session_id,
                                call_id=str(item.get("id") or ""),
                                extension=ext,
                                tool_name=tname,
                                arguments=targs,
                                status="started",
                                summary=format_goose_tool_announcement(ext, tname, targs),
                            )
                            tool_events.append(evt)
                            if on_tool_event:
                                on_tool_event(evt)
                        elif ctype == "text" and role == "assistant":
                            txt = str(item.get("text") or "").strip()
                            if txt:
                                last_assistant_text = txt
                return last_assistant_text, tool_events
        except Exception:
            pass

    # Case 2: Line-by-line `stream-json` or mixed stdout
    for raw_line in raw_stripped.splitlines():
        line = raw_line.strip()
        if not line or GOOSE_BANNER_RE.match(line):
            continue
        if line.startswith("{") and line.endswith("}"):
            try:
                evt_obj = json.loads(line)
                etype = evt_obj.get("type")
                if etype == "message":
                    msg = evt_obj.get("message") or {}
                    role = msg.get("role")
                    msg_id = str(msg.get("id") or f"msg_{len(msg_order)}")
                    for item in msg.get("content") or []:
                        ctype = item.get("type")
                        if ctype == "toolRequest":
                            tc = (item.get("toolCall") or {}).get("value") or {}
                            meta = item.get("_meta") or {}
                            ext = str(meta.get("goose_extension") or "developer")
                            tname = str(tc.get("name") or "tool")
                            targs = tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {}
                            tevt = GooseToolEvent(
                                session_id=session_id,
                                call_id=str(item.get("id") or ""),
                                extension=ext,
                                tool_name=tname,
                                arguments=targs,
                                status="started",
                                summary=format_goose_tool_announcement(ext, tname, targs),
                            )
                            tool_events.append(tevt)
                            if on_tool_event:
                                on_tool_event(tevt)
                        elif ctype == "toolResponse":
                            tr = (item.get("toolResult") or {}).get("value") or {}
                            is_err = bool(tr.get("isError"))
                            tevt = GooseToolEvent(
                                session_id=session_id,
                                call_id=str(item.get("id") or ""),
                                extension="goose",
                                tool_name="tool_response",
                                arguments={},
                                status="error" if is_err else "completed",
                                summary="Goose tool failed" if is_err else "Goose tool completed",
                            )
                            tool_events.append(tevt)
                            if on_tool_event:
                                on_tool_event(tevt)
                        elif ctype == "text" and role == "assistant":
                            piece = str(item.get("text") or "")
                            if msg_id not in text_chunks_by_msg:
                                text_chunks_by_msg[msg_id] = []
                                msg_order.append(msg_id)
                            text_chunks_by_msg[msg_id].append(piece)
                        elif ctype == "error":
                            err_txt = str(item.get("message") or "")
                            if err_txt:
                                plain_lines.append(err_txt)
                    continue
                if etype in ("complete", "notifications"):
                    continue
                if etype == "error":
                    err_str = str(evt_obj.get("error") or "")
                    if err_str:
                        plain_lines.append(err_str)
                    continue
            except Exception:
                pass

        # Ignore noisy CLI log headers from Goose text mode
        if line.startswith(("───", "◓", "◐", "◑", "◒", "[developer]", "[screenpipe]", "[memory]")):
            continue
        plain_lines.append(line)

    if msg_order:
        last_msg_id = msg_order[-1]
        joined = "".join(text_chunks_by_msg[last_msg_id]).strip()
        if joined:
            return joined, tool_events

    return "\n".join(plain_lines).strip(), tool_events


class GooseActivityWatcher:
    """
    Real-time watcher for Goose CLI sessions (`~/.local/share/goose/sessions/sessions.db`).
    Monitors new sessions, tool requests, tool completions, and assistant responses so VoiceFi
    can display or speak what Goose is doing in real time without stealing window focus.
    """

    def __init__(
        self,
        sessions_db: Optional[Path] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
        poll_interval: float = 0.25,
        update_hud: bool = True,
    ):
        self.sessions_db = Path(sessions_db or DEFAULT_GOOSE_SESSIONS_DB).expanduser()
        self.on_event = on_event
        self.poll_interval = poll_interval
        self.update_hud = update_hud
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_message_id: int = self._get_max_message_id()

    def _get_max_message_id(self) -> int:
        if not self.sessions_db.is_file():
            return 0
        try:
            conn = sqlite3.connect(f"file:{self.sessions_db}?mode=ro", uri=True, timeout=2.0)
            try:
                row = conn.execute("SELECT COALESCE(MAX(id), 0) FROM messages").fetchone()
                return int(row[0] or 0) if row else 0
            finally:
                conn.close()
        except Exception:
            return 0

    def poll_once(self) -> List[Dict[str, Any]]:
        """Inspect `sessions.db` for any newly inserted Goose messages or tool calls since last poll."""
        if not self.sessions_db.is_file():
            return []

        events: List[Dict[str, Any]] = []
        try:
            conn = sqlite3.connect(f"file:{self.sessions_db}?mode=ro", uri=True, timeout=2.0)
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    """
                    SELECT m.id, m.session_id, m.role, m.content_json, m.timestamp,
                           s.name AS session_name, s.working_dir, s.provider_name, s.model_config_json
                    FROM messages m
                    LEFT JOIN sessions s ON m.session_id = s.id
                    WHERE m.id > ?
                    ORDER BY m.id ASC
                    """,
                    (self._last_message_id,),
                ).fetchall()
            finally:
                conn.close()
        except Exception:
            return []

        for row in rows:
            msg_row_id = int(row["id"])
            if msg_row_id > self._last_message_id:
                self._last_message_id = msg_row_id

            session_id = str(row["session_id"] or "")
            role = str(row["role"] or "")
            try:
                content_items = json.loads(row["content_json"] or "[]")
            except Exception:
                content_items = []

            for item in content_items:
                if not isinstance(item, dict):
                    continue
                ctype = item.get("type")
                if ctype == "toolRequest":
                    tc = (item.get("toolCall") or {}).get("value") or {}
                    meta = item.get("_meta") or {}
                    ext = str(meta.get("goose_extension") or "developer")
                    tname = str(tc.get("name") or "tool")
                    targs = tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {}
                    summary = format_goose_tool_announcement(ext, tname, targs)
                    evt = {
                        "event_type": "tool_start",
                        "session_id": session_id,
                        "role": role,
                        "extension": ext,
                        "tool_name": tname,
                        "arguments": targs,
                        "summary": summary,
                        "timestamp": str(row["timestamp"] or ""),
                    }
                    events.append(evt)
                    self._emit(evt)
                elif ctype == "toolResponse":
                    tr = (item.get("toolResult") or {}).get("value") or {}
                    is_err = bool(tr.get("isError"))
                    evt = {
                        "event_type": "tool_error" if is_err else "tool_complete",
                        "session_id": session_id,
                        "role": role,
                        "summary": "Goose tool error" if is_err else "Goose tool finished",
                        "timestamp": str(row["timestamp"] or ""),
                    }
                    events.append(evt)
                    self._emit(evt)
                elif ctype == "text" and role == "assistant":
                    txt = str(item.get("text") or "").strip()
                    if txt and not txt.startswith("<turn-context>"):
                        spoken = enforce_spoken_word_budget(txt, max_words=24)
                        evt = {
                            "event_type": "assistant_turn",
                            "session_id": session_id,
                            "role": "assistant",
                            "text": txt,
                            "spoken_summary": spoken,
                            "summary": f"Goose: {spoken}",
                            "timestamp": str(row["timestamp"] or ""),
                        }
                        events.append(evt)
                        self._emit(evt)

        return events

    def _emit(self, evt: Dict[str, Any]) -> None:
        if self.update_hud and not os.environ.get("PYTEST_CURRENT_TEST"):
            try:
                from voicefi.tts.base import set_cross_process_hud_state

                state = "thinking" if evt.get("event_type") != "assistant_turn" else "speaking"
                set_cross_process_hud_state(
                    state,
                    str(evt.get("summary") or "")[:90],
                    agent_name="Goose",
                    app_name="Goose",
                    persona_name="Christopher",
                )
            except Exception:
                pass
        if self.on_event:
            try:
                self.on_event(evt)
            except Exception:
                pass

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="GooseActivityWatcher")
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        while self._running:
            try:
                self.poll_once()
            except Exception:
                pass
            time.sleep(self.poll_interval)


def select_recipe_for_prompt(prompt: str, repo_root: Optional[Path] = None) -> Optional[Tuple[Path, Dict[str, str]]]:
    """
    Match a spoken user command to one of the declarative YAML recipes in `recipes/`
    when explicit recipe keywords are present.
    """
    root = Path(repo_root or Path(__file__).resolve().parents[3])
    recipes_dir = root / "recipes"
    if not recipes_dir.is_dir():
        return None

    low = prompt.lower().strip()

    # 1. Post-call / meeting action items
    if any(k in low for k in ("post call", "post-call", "action items", "meeting notes", "from that call", "from the call")):
        rpath = recipes_dir / "post_call_actions.yaml"
        if rpath.is_file():
            m = re.search(r"(\d+)\s*minute", low)
            mins = m.group(1) if m else "15"
            return rpath, {"minutes": mins, "output_path": str(root / "POST_CALL_ACTIONS.md")}

    # 2. Time-travel recall ("what was I looking at", "find the command I ran", "time travel")
    if any(k in low for k in ("what i was just looking at", "what was i looking at", "time travel", "copy to clipboard", "recall")):
        rpath = recipes_dir / "time_travel_recall.yaml"
        if rpath.is_file():
            return rpath, {"query": prompt}

    # 3. Screen debug ("patch the bug", "fix this error on my screen")
    if any(k in low for k in ("fix this error", "patch this error", "fix the terminal error", "debug my screen")):
        rpath = recipes_dir / "screen_debug.yaml"
        if rpath.is_file():
            return rpath, {"workspace_dir": str(root)}

    return None


def build_deictic_screenpipe_context(
    prompt: str,
    screenpipe_url: str = DEFAULT_SCREENPIPE_URL,
    db_path: Optional[Path] = None,
    include_foveal_dom: bool = True,
) -> str:
    """
    Whenever the user prompt references deictic context ("this error", "on my screen",
    "terminal error", "what I was just looking at", "that meeting", "this button", "active window"),
    automatically query Sovereign Foveal Vision episodic OCR/audio and Foveal Vision DOM so Goose and the local LLM
    have immediate ground-truth context.
    """
    if not DEICTIC_SCREEN_PATTERNS.search(prompt):
        return ""

    low = prompt.lower()
    if any(k in low for k in ("meeting", "call", "said", "audio")):
        records = search_screen_content(
            query="",
            content_type="audio",
            minutes_ago=30,
            limit=10,
            base_url=screenpipe_url,
            db_path=db_path,
            capture_live_if_empty=False,
        )
        if records:
            from whole_body_compute.organs.immune import sanitize_untrusted_screen_text

            lines = [
                f"[{r.timestamp}] ({r.app_name}): {sanitize_untrusted_screen_text(r.text, source='screenpipe_audio')}"
                for r in records[:5]
            ]
            return "Recent Foveal Vision Episodic Audio Transcription:\n" + "\n".join(lines)

    records = search_screen_content(
        query="",
        content_type="ocr",
        app_name="Terminal,iTerm2,Chrome,Godot",
        minutes_ago=10,
        limit=10,
        base_url=screenpipe_url,
        db_path=db_path,
    )
    if not records:
        records = search_screen_content(
            query="",
            content_type="ocr",
            minutes_ago=None,
            limit=5,
            base_url=screenpipe_url,
            db_path=db_path,
        )

    from whole_body_compute.organs.immune import sanitize_untrusted_screen_text

    ocr_ctx = ""
    errors = extract_visible_errors(records)
    if errors:
        top = errors[0]
        clean_snip = sanitize_untrusted_screen_text(top["snippet"], source="screenpipe_ocr")
        ocr_ctx = (
            f"[Foveal Vision Episodic OCR Context - {top['app_name']} ({top['window_name']}) @ {top['timestamp']}]\n"
            f"{clean_snip}"
        )
    elif records:
        top_rec = records[0]
        clean_txt = sanitize_untrusted_screen_text(top_rec.text[:800], source="screenpipe_ocr")
        ocr_ctx = (
            f"[Foveal Vision Episodic OCR Context - {top_rec.app_name} ({top_rec.window_name}) @ {top_rec.timestamp}]\n"
            f"{clean_txt}"
        )

    if include_foveal_dom:
        try:
            from whole_body_compute.integrations.foveal_vision import build_foveal_screen_context

            return build_foveal_screen_context(ocr_context=ocr_ctx, include_foveal_dom=True)
        except Exception:
            pass

    return ocr_ctx


def _is_tautological_assertion(test_node: ast.AST) -> bool:
    """Detect if an assert test condition is trivially true (reward-hacking bypass)."""
    # 1. Direct Truthy Constants: assert True, assert 1, assert "passed"
    if isinstance(test_node, ast.Constant):
        return bool(test_node.value) is True

    # 2. Tautological Compares: assert 1 == 1, assert "a" == "a", assert x == x, assert True is True
    if isinstance(test_node, ast.Compare):
        if len(test_node.ops) == 1 and isinstance(test_node.ops[0], (ast.Eq, ast.Is)):
            left = test_node.left
            right = test_node.comparators[0]
            if isinstance(left, ast.Constant) and isinstance(right, ast.Constant):
                return left.value == right.value
            if isinstance(left, ast.Name) and isinstance(right, ast.Name):
                return left.id == right.id

    # 3. Negated Falsey Constants: assert not False, assert not 0, assert not None
    if isinstance(test_node, ast.UnaryOp) and isinstance(test_node.op, ast.Not):
        if isinstance(test_node.operand, ast.Constant):
            return bool(test_node.operand.value) is False

    # 4. Calls to bool(True) or bool(1)
    if (
        isinstance(test_node, ast.Call)
        and isinstance(test_node.func, ast.Name)
        and test_node.func.id == "bool"
    ):
        if test_node.args and isinstance(test_node.args[0], ast.Constant):
            return bool(test_node.args[0].value) is True

    return False


def _extract_ast_audit_summary(tree: ast.AST) -> Dict[str, Any]:
    """Extract assertions, test functions, public symbols, and unsafe call sites from an AST."""
    assert_count = 0
    tautological_asserts = 0
    test_funcs: Set[str] = set()
    public_symbols: Set[str] = set()
    unsafe_calls: List[str] = []

    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                public_symbols.add(node.name)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            assert_count += 1
            if _is_tautological_assertion(node.test):
                tautological_asserts += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                test_funcs.add(node.name)
        elif isinstance(node, ast.Call):
            # 1. Direct dangerous builtin calls
            if isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec", "compile", "__import__"):
                unsafe_calls.append(node.func.id)
            # 2. Dynamic attribute extraction: getattr(..., 'system' / 'popen' / 'exec' / 'eval')
            elif isinstance(node.func, ast.Name) and node.func.id == "getattr":
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    if node.args[1].value in ("system", "popen", "exec", "eval", "Popen"):
                        unsafe_calls.append(f"getattr(..., \"{node.args[1].value}\")")
            # 3. Attribute calls like os.system, pty.spawn, or subprocess with shell=True
            elif isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                val = node.func.value
                val_repr = val.id if isinstance(val, ast.Name) else ("call()" if isinstance(val, ast.Call) else "")
                qual = f"{val_repr}.{attr}" if val_repr else attr
                if qual in ("os.system", "os.popen", "pty.spawn") or attr in ("system", "popen", "spawn"):
                    unsafe_calls.append(qual)
                elif qual in ("subprocess.Popen", "subprocess.run", "subprocess.call", "subprocess.check_call", "subprocess.check_output"):
                    for kw in node.keywords:
                        if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and bool(kw.value.value) is True:
                            unsafe_calls.append(f"{qual}(shell=True)")
                elif qual == "shutil.rmtree" and node.args:
                    arg0 = node.args[0]
                    if isinstance(arg0, ast.Constant) and str(arg0.value).strip() in ("/", "~", ".."):
                        unsafe_calls.append("shutil.rmtree(root)")

    return {
        "assert_count": assert_count,
        "tautological_asserts": tautological_asserts,
        "test_funcs": test_funcs,
        "public_symbols": public_symbols,
        "unsafe_calls": unsafe_calls,
    }


def run_three_stage_ast_secgate(
    parent_code: str,
    candidate_code: str,
    file_path: str = "",
    required_symbols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Phase 13.2 — 3-Stage Zero-LLM AST SecGate (Anti-Reward-Hacking & Safety Gate).

    Stage 1: Syntax & Interface Contract (`ast.parse` + public symbol preservation + grader lock).
    Stage 2: Anti-Reward-Hacking (Assertion & `test_*` count preservation; rejects `assert True`).
    Stage 3: Unsafe Call & Boundary Audit (Rejects newly introduced `eval`, `exec`, `os.system`).
    """
    fname = Path(file_path).name if file_path else ""
    if fname and any(fname == pat or file_path.endswith(pat) for pat in PROTECTED_GRADER_PATTERNS):
        return {
            "passed": False,
            "stage": 1,
            "reason": f"Stage 1 (Protected Grader Lock): Modification of '{fname}' is forbidden.",
        }

    ext = Path(file_path).suffix.lower() if file_path else ""
    if ext in POLYGLOT_EXTENSIONS:
        poly_gate = run_polyglot_three_stage_secgate(
            parent_code=parent_code,
            candidate_code=candidate_code,
            file_path=file_path,
            required_symbols=required_symbols,
        )
        if not poly_gate.get("passed") and poly_gate.get("stage") == 3:
            try:
                from whole_body_compute.organs.immune import AntibodyRegistry

                AntibodyRegistry.get_instance().record_antigen(
                    vector="ast_secgate",
                    payload=candidate_code,
                    reason=str(poly_gate.get("reason", "Stage 3 unsafe call")),
                    stage=3,
                )
            except Exception:
                pass
        return poly_gate

    # Stage 1: Syntax & Interface Contract
    try:
        cand_tree = ast.parse(candidate_code)
    except SyntaxError as exc:
        return {
            "passed": False,
            "stage": 1,
            "reason": f"Stage 1 (Syntax Error): {exc.msg} at line {exc.lineno}",
        }

    parent_summary = {
        "assert_count": 0,
        "tautological_asserts": 0,
        "test_funcs": set(),
        "public_symbols": set(),
        "unsafe_calls": [],
    }
    if parent_code.strip():
        try:
            parent_tree = ast.parse(parent_code)
            parent_summary = _extract_ast_audit_summary(parent_tree)
        except SyntaxError:
            pass

    cand_summary = _extract_ast_audit_summary(cand_tree)

    if required_symbols:
        missing_req = [s for s in required_symbols if s not in cand_summary["public_symbols"]]
        if missing_req:
            return {
                "passed": False,
                "stage": 1,
                "reason": f"Stage 1 (Interface Contract): Missing required symbols {missing_req}",
            }

    # Stage 2: Anti-Reward-Hacking & Assertion Preservation
    if cand_summary["assert_count"] < parent_summary["assert_count"]:
        return {
            "passed": False,
            "stage": 2,
            "reason": (
                f"Stage 2 (Anti-Reward-Hacking): Candidate reduced assert count "
                f"({cand_summary['assert_count']} < {parent_summary['assert_count']})."
            ),
        }

    missing_tests = parent_summary["test_funcs"] - cand_summary["test_funcs"]
    if missing_tests:
        return {
            "passed": False,
            "stage": 2,
            "reason": (
                f"Stage 2 (Anti-Reward-Hacking): Candidate deleted test function(s): "
                f"{sorted(missing_tests)}"
            ),
        }

    if cand_summary["tautological_asserts"] > parent_summary["tautological_asserts"]:
        return {
            "passed": False,
            "stage": 2,
            "reason": "Stage 2 (Anti-Reward-Hacking): Tautological assertion (`assert True`) detected.",
        }

    # Stage 3: Unsafe Call & Destructive Boundary Audit
    new_unsafe = [
        call for call in cand_summary["unsafe_calls"] if call not in parent_summary["unsafe_calls"]
    ]
    if new_unsafe:
        reason_str = f"Stage 3 (Unsafe Call Audit): Forbidden call(s) introduced: {new_unsafe}"
        try:
            from whole_body_compute.organs.immune import AntibodyRegistry

            AntibodyRegistry.get_instance().record_antigen(
                vector="ast_secgate",
                payload=candidate_code,
                reason=reason_str,
                stage=3,
            )
        except Exception:
            pass
        return {
            "passed": False,
            "stage": 3,
            "reason": reason_str,
        }

    return {
        "passed": True,
        "stage": 3,
        "reason": "All 3 AST SecGate stages passed.",
        "parent_asserts": parent_summary["assert_count"],
        "candidate_asserts": cand_summary["assert_count"],
        "tests_verified": len(cand_summary["test_funcs"]),
    }


class APFSCoWSandbox:
    """
    Phase 13.2 & Open-Core Phase 3 — Cross-Platform CoW & Shadow Sandbox.

    Auto-detects OS and underlying filesystem to isolate workspace mutations with near-zero overhead:
      1. macOS APFS (`Darwin`): `cp -c -R` (`clonefile(2)`) -> `clone_strategy="apfs_cow"`
      2. Linux Btrfs / XFS / bcachefs: `cp --reflink=always -R` (`FICLONE`) -> `clone_strategy="linux_reflink"`
      3. Linux `ext4` / OverlayFS / WSL2:
         - Detached `git worktree add --detach` + working-tree dirty overlay when `.git` has commits -> `"git_worktree"`
         - Same-device `os.link` hardlink shadow with in-memory rollback guard -> `"hardlink_shadow"`
         - Cross-device (`EXDEV`, e.g. `/workspace` -> `/tmp` `tmpfs`) `cp --reflink=auto -R` / copy -> `"reflink_auto_copy"`
    All mutations inside `sandbox_dir` are verified through the polyglot 3-Stage AST SecGate
    (and optional test target) before atomic promotion back to `source_dir`.
    """

    SKIP_TOP_LEVEL = {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".wrangler",
        "dist",
        "build",
        ".next",
        "target",
    }

    MAX_HARDLINK_SNAPSHOT_BYTES = 64 * 1024 * 1024  # 64 MB

    def __init__(
        self,
        source_dir: Path,
        session_id: Optional[str] = None,
        sandbox_root: Optional[Path] = None,
        prefer_strategy: Optional[str] = None,
        platform_system: Optional[str] = None,
    ) -> None:
        self.source_dir = Path(source_dir).resolve()
        self.session_id = session_id or f"sbx_{uuid.uuid4().hex[:8]}"
        root = Path(sandbox_root) if sandbox_root else Path(tempfile.gettempdir())
        self.sandbox_dir = root / f"voicefi_apfs_sandbox_{self.session_id}"
        self.prefer_strategy = prefer_strategy
        self._platform_system = platform_system or platform.system()
        self.clone_strategy: str = "uninitialized"
        self.filesystem_type: str = "unknown"
        self._worktree_registered: bool = False
        self.setup_ms: float = 0.0
        self.rolled_back: bool = False
        self.promoted: bool = False
        self._initial_digests: Dict[str, str] = {}
        self._initial_sources: Dict[str, str] = {}
        self._initial_py_sources: Dict[str, str] = {}
        self._initial_bytes: Dict[str, bytes] = {}

    @classmethod
    def detect_filesystem_type(
        cls,
        path: Path,
        mounts_text: Optional[str] = None,
        platform_system: Optional[str] = None,
    ) -> str:
        """Detect filesystem type (`apfs`, `btrfs`, `xfs`, `ext4`, `overlay`, `tmpfs`, etc.) for `path`."""
        sys_name = platform_system or platform.system()
        raw_p = Path(path)
        if mounts_text is not None:
            target_str = os.path.normpath(str(raw_p))
        else:
            missing_parts: List[str] = []
            curr = raw_p
            try:
                while not curr.exists() and curr.parent != curr:
                    missing_parts.append(curr.name)
                    curr = curr.parent
                resolved_base = curr.resolve()
                for part in reversed(missing_parts):
                    resolved_base = resolved_base / part
                target_str = str(resolved_base)
            except Exception:
                target_str = os.path.abspath(str(raw_p))

        raw_mounts = mounts_text
        if raw_mounts is None and sys_name == "Linux":
            for mp_file in ("/proc/mounts", "/etc/mtab"):
                try:
                    if os.path.exists(mp_file):
                        raw_mounts = Path(mp_file).read_text(encoding="utf-8", errors="replace")
                        break
                except Exception:
                    continue

        if raw_mounts:
            best_mp = ""
            best_fs = "unknown"
            for line in raw_mounts.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                mp = parts[1].replace("\\040", " ")
                fstype = parts[2].lower()
                # Ignore pseudo rootfs entry when a concrete root mount follows
                if fstype == "rootfs" and mp == "/":
                    continue
                norm_mp = mp.rstrip("/") or "/"
                matches = (
                    target_str == norm_mp
                    or norm_mp == "/"
                    or target_str.startswith(norm_mp + "/")
                )
                if matches and len(norm_mp) >= len(best_mp):
                    best_mp = norm_mp
                    best_fs = fstype
            if best_fs != "unknown":
                return best_fs

        if sys_name == "Darwin":
            try:
                res = subprocess.run(
                    ["/sbin/mount"],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=0.5,
                )
                best_mp = ""
                best_fs = "apfs"
                for line in res.stdout.splitlines():
                    m = re.search(r"\son\s+(.+?)\s+\(([^,\)]+)", line)
                    if m:
                        mp = m.group(1).strip().rstrip("/") or "/"
                        fstype = m.group(2).strip().lower()
                        if (target_str == mp or mp == "/" or target_str.startswith(mp + "/")) and len(mp) >= len(best_mp):
                            best_mp = mp
                            best_fs = fstype
                return best_fs
            except Exception:
                return "apfs"

        return "unknown"

    @classmethod
    def probe_cow_support(
        cls,
        target_dir: Optional[Path] = None,
        platform_system: Optional[str] = None,
        mounts_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Probe Copy-on-Write / shadow sandbox support on the host OS & filesystem.
        Used by `body doctor` (`check_organ_immune_system`) and sandbox telemetry.
        """
        sys_name = platform_system or platform.system()
        probe_root = Path(target_dir) if target_dir else Path(tempfile.gettempdir())
        probe_root.mkdir(parents=True, exist_ok=True)
        fs_type = cls.detect_filesystem_type(
            probe_root, mounts_text=mounts_text, platform_system=sys_name
        )
        token = f"{os.getpid()}_{uuid.uuid4().hex[:6]}"
        t_src = probe_root / f".wbc_cow_src_{token}"
        t_dst = probe_root / f".wbc_cow_dst_{token}"

        try:
            t_src.write_text("cow_check", encoding="utf-8")
            if sys_name == "Darwin":
                res = subprocess.run(
                    ["cp", "-c", str(t_src), str(t_dst)],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    timeout=0.3,
                )
                if res.returncode == 0:
                    return {
                        "supported": True,
                        "native_cow": True,
                        "strategy": "apfs_cow",
                        "filesystem": fs_type or "apfs",
                        "label": "APFS CoW active (<2ms clone)",
                    }
                return {
                    "supported": False,
                    "native_cow": False,
                    "strategy": "copy_fallback",
                    "filesystem": fs_type,
                    "label": f"APFS CoW fallback ({fs_type})",
                }

            if sys_name == "Linux":
                res_reflink = subprocess.run(
                    ["cp", "--reflink=always", str(t_src), str(t_dst)],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    timeout=0.3,
                )
                if res_reflink.returncode == 0:
                    return {
                        "supported": True,
                        "native_cow": True,
                        "strategy": "linux_reflink",
                        "filesystem": fs_type,
                        "label": f"Linux FICLONE reflink active ({fs_type})",
                    }
                t_dst.unlink(missing_ok=True)
                try:
                    os.link(t_src, t_dst)
                    return {
                        "supported": True,
                        "native_cow": False,
                        "strategy": "hardlink_or_worktree",
                        "filesystem": fs_type,
                        "label": f"Linux CoW shadow active ({fs_type}: git-worktree / hardlink)",
                    }
                except OSError:
                    return {
                        "supported": True,
                        "native_cow": False,
                        "strategy": "reflink_auto_copy",
                        "filesystem": fs_type,
                        "label": f"Linux shadow sandbox active ({fs_type}: cp --reflink=auto)",
                    }

            return {
                "supported": True,
                "native_cow": False,
                "strategy": "copy_fallback",
                "filesystem": fs_type,
                "label": f"Shadow copy sandbox ({sys_name})",
            }
        except Exception:
            return {
                "supported": False,
                "native_cow": False,
                "strategy": "copy_fallback",
                "filesystem": fs_type,
                "label": "CoW sandbox fallback",
            }
        finally:
            t_src.unlink(missing_ok=True)
            t_dst.unlink(missing_ok=True)

    def _iter_workspace_files(self, base_dir: Path) -> List[Path]:
        files: List[Path] = []
        if not base_dir.exists():
            return files
        for root_str, dirnames, filenames in os.walk(base_dir):
            dirnames[:] = [d for d in dirnames if d not in self.SKIP_TOP_LEVEL]
            root_path = Path(root_str)
            for fname in filenames:
                if fname in self.SKIP_TOP_LEVEL or fname.endswith((".pyc", ".pyo", ".DS_Store")):
                    continue
                files.append(root_path / fname)
        return files

    @staticmethod
    def _file_sha256(path: Path) -> str:
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except Exception:
            return ""

    @staticmethod
    def _is_protected_or_test_file(rel_path: str) -> bool:
        p = Path(rel_path)
        fname = p.name
        if any(fname == pat or rel_path.endswith(pat) for pat in PROTECTED_GRADER_PATTERNS):
            return True
        if any(fname == pat or rel_path.endswith(pat) for pat in PROTECTED_POLYGLOT_CONFIG_FILES):
            return True
        ext = p.suffix.lower()
        if ext not in POLYGLOT_SECGATE_EXTENSIONS:
            return False
        low_name = fname.lower()
        parts_low = {part.lower() for part in p.parts[:-1]}
        if "tests" in parts_low or "__tests__" in parts_low:
            return True
        if (
            low_name.startswith("test_")
            or "_test." in low_name
            or ".test." in low_name
            or ".spec." in low_name
            or fname.endswith(("Test.java", "Tests.java", "Test.kt", "Tests.kt"))
        ):
            return True
        return False

    def _copy_entries_python(self, entries: List[str], use_hardlinks: bool = False) -> None:
        copy_fn = os.link if use_hardlinks else shutil.copy2
        for name in entries:
            src_item = self.source_dir / name
            dst_item = self.sandbox_dir / name
            if src_item.is_symlink():
                os.symlink(os.readlink(src_item), dst_item)
            elif src_item.is_dir():
                shutil.copytree(
                    src_item,
                    dst_item,
                    symlinks=True,
                    copy_function=copy_fn,
                    ignore=shutil.ignore_patterns(*self.SKIP_TOP_LEVEL, "*.pyc", "*.pyo", ".DS_Store"),
                )
            elif src_item.is_file():
                copy_fn(src_item, dst_item)

    def _try_create_git_worktree(self) -> bool:
        """Create a detached git worktree at `sandbox_dir` and overlay uncommitted working-tree state."""
        if not (self.source_dir / ".git").exists():
            return False
        try:
            if self.sandbox_dir.exists():
                shutil.rmtree(self.sandbox_dir, ignore_errors=True)
            res = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.source_dir),
                    "worktree",
                    "add",
                    "--detach",
                    "--quiet",
                    str(self.sandbox_dir),
                    "HEAD",
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=3.0,
            )
            if res.returncode != 0:
                return False
            self._worktree_registered = True

            # Overlay uncommitted modified/untracked files and working-tree deletions so
            # sandbox_dir reflects the exact live working tree rather than stale HEAD.
            st_res = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.source_dir),
                    "status",
                    "--porcelain",
                    "-uall",
                    "-z",
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=3.0,
            )
            if st_res.returncode == 0 and st_res.stdout:
                tokens = st_res.stdout.decode("utf-8", errors="replace").split("\x00")
                idx = 0
                while idx < len(tokens):
                    rec = tokens[idx]
                    idx += 1
                    if len(rec) < 4:
                        continue
                    xy = rec[:2]
                    rel_str = rec[3:]
                    # In `-z` porcelain mode, renames/copies emit `XY new_path\0old_path\0`
                    if "R" in xy or "C" in xy:
                        old_rel = tokens[idx] if idx < len(tokens) else ""
                        idx += 1
                        if old_rel:
                            (self.sandbox_dir / old_rel).unlink(missing_ok=True)
                    rel_parts = Path(rel_str).parts
                    if rel_parts and rel_parts[0] in self.SKIP_TOP_LEVEL:
                        continue
                    src_f = self.source_dir / rel_str
                    dst_f = self.sandbox_dir / rel_str
                    if src_f.is_symlink():
                        dst_f.parent.mkdir(parents=True, exist_ok=True)
                        dst_f.unlink(missing_ok=True)
                        os.symlink(os.readlink(src_f), dst_f)
                    elif src_f.is_file():
                        dst_f.parent.mkdir(parents=True, exist_ok=True)
                        dst_f.unlink(missing_ok=True)
                        shutil.copy2(src_f, dst_f)
                    elif not src_f.exists() and dst_f.exists():
                        if dst_f.is_dir() and not dst_f.is_symlink():
                            shutil.rmtree(dst_f, ignore_errors=True)
                        else:
                            dst_f.unlink(missing_ok=True)
            return True
        except Exception:
            if self._worktree_registered:
                self._cleanup_worktree()
            return False

    def _cleanup_worktree(self) -> None:
        if not self._worktree_registered:
            return
        self._worktree_registered = False
        try:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.source_dir),
                    "worktree",
                    "remove",
                    "--force",
                    str(self.sandbox_dir),
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=2.5,
            )
            subprocess.run(
                ["git", "-C", str(self.source_dir), "worktree", "prune"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=1.5,
            )
        except Exception:
            pass

    def _try_create_hardlink_shadow(self, entries: List[str]) -> bool:
        """
        Create a fast `os.link` hardlink shadow tree on same-mount `ext4`/OverlayFS and
        snapshot original bytes (`<= 64MB`) so in-place writes in `sandbox_dir` can be
        atomically rolled back without corrupting `source_dir`.
        """
        try:
            src_dev = self.source_dir.stat().st_dev
            dst_dev = self.sandbox_dir.stat().st_dev
            if src_dev != dst_dev:
                return False

            src_files = self._iter_workspace_files(self.source_dir)
            total_bytes = 0
            byte_map: Dict[str, bytes] = {}
            for sf in src_files:
                if sf.is_symlink():
                    continue
                st = sf.stat()
                total_bytes += st.st_size
                if total_bytes > self.MAX_HARDLINK_SNAPSHOT_BYTES:
                    return False

            self._copy_entries_python(entries, use_hardlinks=True)
            for sf in src_files:
                if not sf.is_symlink():
                    rel = str(sf.relative_to(self.source_dir))
                    byte_map[rel] = sf.read_bytes()
            self._initial_bytes = byte_map
            return True
        except OSError as exc:
            if exc.errno == errno.EXDEV or True:
                shutil.rmtree(self.sandbox_dir, ignore_errors=True)
                self.sandbox_dir.mkdir(parents=True, exist_ok=True)
                self._initial_bytes.clear()
                return False

    def _copy_with_reflink_auto_or_fallback(self, entries: List[str]) -> str:
        """Copy workspace entries using GNU `cp --reflink=auto -R`, BSD `cp -R`, or Python `copy2`."""
        src_paths = [str(self.source_dir / name) for name in entries]
        if self._platform_system == "Linux":
            res = subprocess.run(
                ["cp", "--reflink=auto", "-R", *src_paths, str(self.sandbox_dir)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if res.returncode == 0:
                return "reflink_auto_copy"

        res_cp = subprocess.run(
            ["cp", "-R", *src_paths, str(self.sandbox_dir)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if res_cp.returncode == 0:
            return "copy_fallback"

        self._copy_entries_python(entries, use_hardlinks=False)
        return "copy_fallback"

    def create(self) -> Path:
        """Create the OS-aware CoW / shadow clone and snapshot initial file digests."""
        t0 = time.perf_counter()
        self._cleanup_worktree()
        if self.sandbox_dir.exists():
            shutil.rmtree(self.sandbox_dir, ignore_errors=True)
        self.sandbox_dir.parent.mkdir(parents=True, exist_ok=True)
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)

        self.filesystem_type = self.detect_filesystem_type(
            self.sandbox_dir, platform_system=self._platform_system
        )
        self._initial_bytes.clear()

        if self.source_dir.exists():
            entries = [
                e.name for e in self.source_dir.iterdir() if e.name not in self.SKIP_TOP_LEVEL
            ]
            src_paths = [str(self.source_dir / name) for name in entries]
            pref = self.prefer_strategy

            if pref == "git_worktree":
                if self._try_create_git_worktree():
                    self.clone_strategy = "git_worktree"
                else:
                    self.sandbox_dir.mkdir(parents=True, exist_ok=True)
                    self.clone_strategy = (
                        self._copy_with_reflink_auto_or_fallback(entries) if entries else "empty"
                    )
            elif pref == "hardlink_shadow":
                if not entries:
                    self.clone_strategy = "hardlink_shadow"
                elif self._try_create_hardlink_shadow(entries):
                    self.clone_strategy = "hardlink_shadow"
                else:
                    self.clone_strategy = self._copy_with_reflink_auto_or_fallback(entries)
            elif pref == "linux_reflink":
                if not entries:
                    self.clone_strategy = "linux_reflink"
                else:
                    res = subprocess.run(
                        ["cp", "--reflink=always", "-R", *src_paths, str(self.sandbox_dir)],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    if res.returncode == 0:
                        self.clone_strategy = "linux_reflink"
                    else:
                        shutil.rmtree(self.sandbox_dir, ignore_errors=True)
                        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
                        self.clone_strategy = self._copy_with_reflink_auto_or_fallback(entries)
            elif pref == "reflink_auto_copy":
                self.clone_strategy = (
                    self._copy_with_reflink_auto_or_fallback(entries) if entries else "reflink_auto_copy"
                )
            elif self._platform_system == "Darwin":
                if not entries:
                    self.clone_strategy = "apfs_cow"
                else:
                    res = subprocess.run(
                        ["cp", "-c", "-R", *src_paths, str(self.sandbox_dir)],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    if res.returncode == 0:
                        self.clone_strategy = "apfs_cow"
                    else:
                        shutil.rmtree(self.sandbox_dir, ignore_errors=True)
                        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
                        self.clone_strategy = self._copy_with_reflink_auto_or_fallback(entries)
            elif self._platform_system == "Linux":
                if not entries:
                    self.clone_strategy = "linux_reflink"
                else:
                    # 1. Native kernel FICLONE CoW on Btrfs / XFS / bcachefs
                    res_ref = subprocess.run(
                        ["cp", "--reflink=always", "-R", *src_paths, str(self.sandbox_dir)],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    if res_ref.returncode == 0:
                        self.clone_strategy = "linux_reflink"
                    else:
                        shutil.rmtree(self.sandbox_dir, ignore_errors=True)
                        # 2. Detached git worktree + dirty overlay on ext4 / OverlayFS git repos
                        if self._try_create_git_worktree():
                            self.clone_strategy = "git_worktree"
                        else:
                            self.sandbox_dir.mkdir(parents=True, exist_ok=True)
                            # 3. Same-device hardlink shadow with rollback byte-guard
                            if self._try_create_hardlink_shadow(entries):
                                self.clone_strategy = "hardlink_shadow"
                            else:
                                # 4. Cross-mount (EXDEV / tmpfs) reflink=auto / copy fallback
                                self.clone_strategy = self._copy_with_reflink_auto_or_fallback(entries)
            else:
                self.clone_strategy = (
                    self._copy_with_reflink_auto_or_fallback(entries) if entries else "copy_fallback"
                )

        self.setup_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        self._initial_digests.clear()
        self._initial_sources.clear()
        self._initial_py_sources.clear()
        for fpath in self._iter_workspace_files(self.sandbox_dir):
            rel = str(fpath.relative_to(self.sandbox_dir))
            self._initial_digests[rel] = self._file_sha256(fpath)
            ext_low = fpath.suffix.lower()
            if ext_low in POLYGLOT_SECGATE_EXTENSIONS:
                try:
                    txt = fpath.read_text(encoding="utf-8")
                except Exception:
                    txt = ""
                self._initial_sources[rel] = txt
                if ext_low == ".py":
                    self._initial_py_sources[rel] = txt

        return self.sandbox_dir

    def detect_modified_files(self) -> List[str]:
        """Return relative paths of files modified or created inside the shadow sandbox."""
        modified: List[str] = []
        if not self.sandbox_dir.exists():
            return modified
        for fpath in self._iter_workspace_files(self.sandbox_dir):
            rel = str(fpath.relative_to(self.sandbox_dir))
            digest = self._file_sha256(fpath)
            if self._initial_digests.get(rel) != digest:
                modified.append(rel)
        return sorted(modified)

    def detect_deleted_files(self) -> List[str]:
        """Return relative paths of tracked files that were deleted inside the shadow sandbox."""
        if not self.sandbox_dir.exists():
            return []
        deleted: List[str] = []
        for rel in self._initial_digests:
            if not (self.sandbox_dir / rel).exists():
                deleted.append(rel)
        return sorted(deleted)

    def rollback(self) -> None:
        """Instantly discard the shadow sandbox and restore any shared-inode hardlink writes."""
        self.rolled_back = True
        if self.clone_strategy == "hardlink_shadow" and self._initial_bytes:
            for rel, orig_bytes in self._initial_bytes.items():
                live_file = self.source_dir / rel
                expected_sha = self._initial_digests.get(rel)
                if expected_sha and (not live_file.exists() or self._file_sha256(live_file) != expected_sha):
                    live_file.parent.mkdir(parents=True, exist_ok=True)
                    tmp_restore = live_file.with_name(f".{live_file.name}.wbc_rb_{os.getpid()}.tmp")
                    tmp_restore.write_bytes(orig_bytes)
                    os.replace(tmp_restore, live_file)
            self._initial_bytes.clear()

        if self._worktree_registered:
            self._cleanup_worktree()
        if self.sandbox_dir.exists():
            shutil.rmtree(self.sandbox_dir, ignore_errors=True)

    def verify_and_promote(
        self,
        run_pytest_target: Optional[str] = None,
        pytest_timeout: int = 30,
    ) -> Dict[str, Any]:
        """
        Audit all shadow mutations through the polyglot 3-Stage SecGate (and optional pytest target).
        Promotes verified diffs to `source_dir` only if 100% green; otherwise rolls back.
        """
        if self.rolled_back or not self.sandbox_dir.exists():
            return {
                "passed": False,
                "reason": "Sandbox was already rolled back or interrupted.",
                "promoted_files": [],
                "setup_ms": self.setup_ms,
                "clone_strategy": self.clone_strategy,
            }

        deleted_files = self.detect_deleted_files()
        for del_rel in deleted_files:
            if self._is_protected_or_test_file(del_rel):
                self.rollback()
                return {
                    "passed": False,
                    "stage": 2,
                    "reason": f"Stage 2 (Anti-Reward-Hacking): Protected or test file deleted: {del_rel}",
                    "failed_file": del_rel,
                    "promoted_files": [],
                    "setup_ms": self.setup_ms,
                    "clone_strategy": self.clone_strategy,
                }

        modified_files = self.detect_modified_files()
        file_reports: Dict[str, Any] = {}

        for rel_path in modified_files:
            cand_file = self.sandbox_dir / rel_path
            fname = cand_file.name
            if any(fname == pat or rel_path.endswith(pat) for pat in PROTECTED_GRADER_PATTERNS) or any(
                fname == pat or rel_path.endswith(pat) for pat in PROTECTED_POLYGLOT_CONFIG_FILES
            ):
                self.rollback()
                return {
                    "passed": False,
                    "stage": 1,
                    "reason": f"Stage 1 (Protected Grader Lock): Attempted to modify {rel_path}",
                    "failed_file": rel_path,
                    "promoted_files": [],
                    "setup_ms": self.setup_ms,
                    "clone_strategy": self.clone_strategy,
                }

            if cand_file.suffix.lower() in POLYGLOT_SECGATE_EXTENSIONS and not cand_file.is_symlink():
                parent_src = self._initial_sources.get(
                    rel_path, self._initial_py_sources.get(rel_path, "")
                )
                try:
                    cand_src = cand_file.read_text(encoding="utf-8")
                except Exception as exc:
                    self.rollback()
                    return {
                        "passed": False,
                        "stage": 1,
                        "reason": f"Failed to read candidate source file {rel_path}: {exc}",
                        "failed_file": rel_path,
                        "promoted_files": [],
                        "setup_ms": self.setup_ms,
                        "clone_strategy": self.clone_strategy,
                    }

                gate = run_three_stage_ast_secgate(
                    parent_code=parent_src,
                    candidate_code=cand_src,
                    file_path=rel_path,
                )
                file_reports[rel_path] = gate
                if not gate.get("passed"):
                    self.rollback()
                    return {
                        "passed": False,
                        "stage": gate.get("stage", 1),
                        "reason": gate.get("reason", "AST SecGate rejected change"),
                        "failed_file": rel_path,
                        "secgate": gate,
                        "promoted_files": [],
                        "setup_ms": self.setup_ms,
                        "clone_strategy": self.clone_strategy,
                    }

        if run_pytest_target:
            env = dict(os.environ)
            env["VOICEFI_MOCK_AUDIO"] = "1"
            env["VOICEFI_HEADLESS"] = "1"
            env["PYTHONPATH"] = str(self.sandbox_dir / "src") + os.pathsep + str(self.sandbox_dir)
            py_res = subprocess.run(
                ["pytest", run_pytest_target, "-q"],
                cwd=str(self.sandbox_dir),
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=pytest_timeout,
            )
            if py_res.returncode != 0:
                self.rollback()
                return {
                    "passed": False,
                    "stage": 3,
                    "reason": f"Shadow sandbox pytest failed on {run_pytest_target}",
                    "pytest_stdout": (py_res.stdout or "")[-500:],
                    "pytest_stderr": (py_res.stderr or "")[-500:],
                    "promoted_files": [],
                    "setup_ms": self.setup_ms,
                    "clone_strategy": self.clone_strategy,
                }

        # All checks green -> Atomically promote modified/created files back to live source_dir
        promoted: List[str] = []
        for rel_path in modified_files:
            src_file = self.sandbox_dir / rel_path
            dst_file = self.source_dir / rel_path

            # Symlink Exfiltration Guard: Disallow symlinks escaping workspace boundaries
            if src_file.is_symlink():
                try:
                    resolved = src_file.resolve()
                    inside_sandbox = True
                    try:
                        resolved.relative_to(self.sandbox_dir.resolve())
                    except ValueError:
                        inside_sandbox = False

                    inside_source = True
                    try:
                        resolved.relative_to(self.source_dir.resolve())
                    except ValueError:
                        inside_source = False

                    if not (inside_sandbox or inside_source):
                        self.rollback()
                        return {
                            "passed": False,
                            "stage": 3,
                            "reason": (
                                f"Security Exception: Symlink '{rel_path}' points outside workspace "
                                f"boundary ({resolved})"
                            ),
                            "failed_file": rel_path,
                            "promoted_files": [],
                            "setup_ms": self.setup_ms,
                            "clone_strategy": self.clone_strategy,
                        }
                except Exception as exc:
                    self.rollback()
                    return {
                        "passed": False,
                        "stage": 3,
                        "reason": f"Security Exception: Unresolvable symlink '{rel_path}': {exc}",
                        "failed_file": rel_path,
                        "promoted_files": [],
                        "setup_ms": self.setup_ms,
                        "clone_strategy": self.clone_strategy,
                    }

                dst_file.parent.mkdir(parents=True, exist_ok=True)
                dst_file.unlink(missing_ok=True)
                os.symlink(os.readlink(src_file), dst_file)
                promoted.append(rel_path)
                continue

            dst_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_dst = dst_file.with_name(f".{dst_file.name}.wbc_promote_{os.getpid()}.tmp")
            shutil.copy2(src_file, tmp_dst, follow_symlinks=False)
            os.replace(tmp_dst, dst_file)
            promoted.append(rel_path)

        self.promoted = True
        self._initial_bytes.clear()
        if self._worktree_registered:
            self._cleanup_worktree()
        if self.sandbox_dir.exists():
            shutil.rmtree(self.sandbox_dir, ignore_errors=True)
        return {
            "passed": True,
            "reason": "All shadow sandbox checks passed.",
            "promoted_files": promoted,
            "files_checked": len(modified_files),
            "file_reports": file_reports,
            "setup_ms": self.setup_ms,
            "clone_strategy": self.clone_strategy,
        }


# Cross-platform alias for Linux / WSL2 / Devcontainer callers
CoWSandbox = APFSCoWSandbox


def trigger_vagus_nerve_brake(
    reason: str = "spoken_barge_in",
    sig: int = signal.SIGINT,
) -> Dict[str, Any]:
    """
    Phase 13.3 — Module-level Vagus Nerve Brake entrypoint.
    Sends `os.killpg(pgid, sig)` to any active Goose process group in <50ms
    and rolls back any active APFS Copy-on-Write shadow sandbox.
    """
    return GooseHeadlessRunner.get_instance().trigger_vagus_nerve_brake(reason=reason, sig=sig)


class GooseHeadlessRunner:
    """
    Thread-safe, focus-preserving headless executor for Goose CLI (`goose run`)
    backed by local Ollama/MLX (`127.0.0.1:11434`) and Sovereign Foveal Vision & Episodic Memory (`127.0.0.1:3030`).
    """

    _instance: Optional["GooseHeadlessRunner"] = None
    _global_lock = threading.Lock()

    def __init__(self) -> None:
        self._conv_locks: Dict[str, threading.Lock] = {}
        self._conv_locks_guard = threading.Lock()
        self._active_procs: Dict[str, subprocess.Popen] = {}
        self._active_sandboxes: Dict[str, APFSCoWSandbox] = {}
        self._interrupted_sessions: Set[str] = set()
        self._active_guard = threading.Lock()

    @classmethod
    def get_instance(cls) -> "GooseHeadlessRunner":
        with cls._global_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _get_conv_lock(self, canonical_id: str) -> threading.Lock:
        with self._conv_locks_guard:
            if canonical_id not in self._conv_locks:
                self._conv_locks[canonical_id] = threading.Lock()
            return self._conv_locks[canonical_id]

    def register_active_process(
        self,
        session_id: str,
        proc: subprocess.Popen,
        sandbox: Optional[APFSCoWSandbox] = None,
    ) -> None:
        """Track an active Goose subprocess group and its APFS CoW shadow sandbox."""
        with self._active_guard:
            self._active_procs[session_id] = proc
            if sandbox is not None:
                self._active_sandboxes[session_id] = sandbox
            try:
                pgid = os.getpgid(proc.pid)
            except Exception:
                pgid = proc.pid
            try:
                DEFAULT_GOOSE_ACTIVE_PID_FILE.write_text(
                    json.dumps(
                        {
                            "session_id": session_id,
                            "pid": proc.pid,
                            "pgid": pgid,
                            "sandbox_dir": str(sandbox.sandbox_dir) if sandbox else None,
                            "timestamp": time.time(),
                        }
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass

    def unregister_active_process(self, session_id: str) -> None:
        """Remove a completed Goose session from the active process registry."""
        with self._active_guard:
            self._active_procs.pop(session_id, None)
            self._active_sandboxes.pop(session_id, None)
            try:
                if DEFAULT_GOOSE_ACTIVE_PID_FILE.exists():
                    data = json.loads(DEFAULT_GOOSE_ACTIVE_PID_FILE.read_text(encoding="utf-8"))
                    if data.get("session_id") == session_id:
                        DEFAULT_GOOSE_ACTIVE_PID_FILE.unlink(missing_ok=True)
            except Exception:
                pass

    def trigger_vagus_nerve_brake(
        self,
        reason: str = "spoken_barge_in",
        sig: int = signal.SIGINT,
    ) -> Dict[str, Any]:
        """
        Phase 13.3 — Sub-50ms Vagus Nerve Brake.
        Immediately dispatches `os.killpg(pgid, sig)` to all active Goose process groups
        and rolls back any active APFS Copy-on-Write shadow sandboxes.
        """
        t0 = time.perf_counter()
        killed_sessions: List[str] = []
        killed_pids: List[int] = []
        rolled_back: List[str] = []

        with self._active_guard:
            for session_id, proc in list(self._active_procs.items()):
                self._interrupted_sessions.add(session_id)
                if proc.poll() is None:
                    try:
                        pgid = os.getpgid(proc.pid)
                        os.killpg(pgid, sig)
                    except Exception:
                        try:
                            proc.send_signal(sig)
                        except Exception:
                            pass
                    killed_sessions.append(session_id)
                    killed_pids.append(proc.pid)

                sbx = self._active_sandboxes.get(session_id)
                if sbx is not None and not sbx.rolled_back:
                    sbx.rollback()
                    rolled_back.append(str(sbx.sandbox_dir))

            # Also check cross-process PID file in `/tmp/voicefi_goose_active.json`
            if DEFAULT_GOOSE_ACTIVE_PID_FILE.exists():
                try:
                    info = json.loads(DEFAULT_GOOSE_ACTIVE_PID_FILE.read_text(encoding="utf-8"))
                    ext_pgid = int(info.get("pgid") or info.get("pid") or 0)
                    ext_sid = str(info.get("session_id") or "")
                    if ext_pgid > 0 and ext_pgid not in killed_pids:
                        try:
                            os.killpg(ext_pgid, sig)
                            killed_pids.append(ext_pgid)
                            if ext_sid and ext_sid not in killed_sessions:
                                killed_sessions.append(ext_sid)
                        except ProcessLookupError:
                            pass
                        except Exception:
                            pass
                    ext_sbx = info.get("sandbox_dir")
                    if ext_sbx and Path(ext_sbx).exists():
                        shutil.rmtree(ext_sbx, ignore_errors=True)
                        if ext_sbx not in rolled_back:
                            rolled_back.append(ext_sbx)
                    DEFAULT_GOOSE_ACTIVE_PID_FILE.unlink(missing_ok=True)
                except Exception:
                    pass

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return {
            "halted": len(killed_pids) > 0 or len(rolled_back) > 0,
            "killed_sessions": killed_sessions,
            "killed_pids": killed_pids,
            "sandboxes_rolled_back": rolled_back,
            "reason": reason,
            "signal": int(sig),
            "latency_ms": latency_ms,
        }

    def run_voice_turn(
        self,
        prompt: str,
        recipe_path: Optional[Path] = None,
        recipe_params: Optional[Dict[str, str]] = None,
        model: Optional[str] = None,
        cwd: Optional[Path] = None,
        timeout: int = 90,
        with_extensions: bool = False,
        on_tool_event: Optional[Callable[[GooseToolEvent], None]] = None,
        screenpipe_db: Optional[Path] = None,
        speak_aloud: bool = False,
        use_apfs_sandbox: bool = True,
        use_zero_egress_seatbelt: bool = False,
        inject_dot_memory: bool = True,
        pytest_verify_target: Optional[str] = None,
        capture_tactile: bool = True,
        with_foveal: bool = True,
        session_id: Optional[str] = None,
        from_conv_id: Optional[str] = None,
        from_conv_name: Optional[str] = None,
        from_engine: Optional[str] = None,
        conv_name: Optional[str] = None,
        include_envelope: bool = True,
    ) -> GooseTurnResult:
        """
        Execute a voice command headlessly through Goose CLI without stealing window focus,
        returning a clean <25-word spoken summary string for TTS.
        """
        t0 = time.time()
        session_id = session_id or f"goose_{uuid.uuid4().hex[:8]}"

        recorded_crumb: Optional[Dict[str, Any]] = None
        if from_conv_id or include_envelope:
            try:
                from whole_body_compute.integrations.agent_bridge import (
                    record_bridge_crumb,
                    wrap_prompt_with_crumbs,
                )

                crumb_obj = record_bridge_crumb(
                    source_engine=from_engine or "antigravity",
                    source_conv_id=from_conv_id,
                    source_conv_name=from_conv_name,
                    target_engine="goose",
                    target_conv_id=session_id,
                    target_conv_name=conv_name or f"Goose ({session_id[:8]})",
                    prompt=prompt,
                )
                recorded_crumb = crumb_obj.to_dict()
                if include_envelope:
                    prompt = wrap_prompt_with_crumbs(prompt, crumb_obj)
            except Exception:
                pass

        goose_bin = find_goose_binary()

        # HybridRouter resolution based on thermal state and available endpoints
        provider = "ollama"
        endpoint_url = DEFAULT_OLLAMA_URL
        resolved_model = resolve_local_ollama_model(preferred=model)

        if resolved_model == "voicefi-coder-mlx" or "9379" in resolved_model:
            provider = "openai"
            endpoint_url = "http://127.0.0.1:9379/v1"
        elif resolved_model == "gemma4-26b" or "9380" in resolved_model:
            provider = "openai"
            endpoint_url = "http://127.0.0.1:9380/v1"

        if not goose_bin:
            err = "Goose CLI binary not found. Run setup/install_stack.sh first."
            return GooseTurnResult(
                success=False,
                spoken_summary=err,
                raw_response="",
                session_id=session_id,
                model=resolved_model,
                error=err,
                bridge_crumb=recorded_crumb,
            )

        working_dir = Path(cwd or Path(__file__).resolve().parents[3])

        # Phase 14.2 — Tactile Baseline (Pre-Action Snapshot)
        tactile_base = None
        if capture_tactile and not os.environ.get("VOICEFI_DISABLE_TACTILE"):
            try:
                from whole_body_compute.integrations.tactile import capture_tactile_baseline

                tactile_base = capture_tactile_baseline(working_dir=working_dir)
            except Exception:
                tactile_base = None

        deictic_ctx = build_deictic_screenpipe_context(
            prompt, db_path=screenpipe_db, include_foveal_dom=with_foveal
        )

        # Phase 14.1 — Foveal Vision DOM Snapshot (if deictic context is present)
        foveal_snap = None
        if with_foveal and deictic_ctx:
            try:
                from whole_body_compute.integrations.foveal_vision import capture_foveal_snapshot

                foveal_snap = capture_foveal_snapshot(max_elements=25)
            except Exception:
                foveal_snap = None

        # Phase 13.1 — Dot Hippocampus Personal Preference & Habit Injection
        dot_ctx = ""
        dot_memories_count = 0
        if inject_dot_memory and not os.environ.get("VOICEFI_DISABLE_DOT_MEMORY"):
            try:
                from whole_body_compute.local.dot_memory import get_dot_store

                store = get_dot_store()
                dot_ctx = store.build_context_injection(prompt, top_k=3)
                if dot_ctx:
                    dot_memories_count = sum(
                        1 for line in dot_ctx.splitlines() if line.strip().startswith("- [")
                    )
            except Exception:
                dot_ctx = ""

        auto_recipe = None
        if recipe_path is None:
            auto_recipe = select_recipe_for_prompt(prompt, repo_root=working_dir)
            if auto_recipe:
                recipe_path, recipe_params = auto_recipe

        system_guard = (
            "You are the VoiceFi Sovereign OS voice assistant running 100% locally on macOS. "
            "Respond with ONLY a single concise spoken sentence under 25 words suitable for TTS playback. "
            "Do not output markdown blocks, JSON wrappers, or bullet lists in your final answer. "
            "Never execute destructive shell commands (rm -rf, git reset --hard, external POSTs)."
        )

        context_blocks: List[str] = []
        if dot_ctx:
            context_blocks.append(dot_ctx)
        if deictic_ctx:
            context_blocks.append(deictic_ctx)

        effective_prompt = prompt.strip()
        if context_blocks:
            joined_ctx = "\n\n".join(context_blocks)
            effective_prompt = (
                f"{joined_ctx}\n\n"
                f"User Voice Command: {prompt.strip()}\n"
                f"Instruction: Answer in one concise spoken sentence under 25 words."
            )

        # Phase 13.2 — Zero-Cost APFS Copy-on-Write Shadow Sandbox
        sandbox: Optional[APFSCoWSandbox] = None
        exec_dir = working_dir
        if use_apfs_sandbox:
            sandbox = APFSCoWSandbox(working_dir, session_id=session_id)
            exec_dir = sandbox.create()

        cmd: List[str] = [
            goose_bin,
            "run",
            "--no-session",
            "--provider",
            provider,
            "--model",
            resolved_model,
            "--output-format",
            "stream-json",
            "--max-turns",
            "4",
        ]

        if not with_extensions and recipe_path is None:
            cmd.append("--no-profile")
        elif with_extensions and recipe_path is None:
            cmd.extend(["--with-builtin", "developer"])
            mcp_script = working_dir / "bin" / "screenpipe-mcp"
            if mcp_script.is_file():
                cmd.extend(["--with-extension", f"screenpipe:{mcp_script}"])

        if recipe_path is not None and Path(recipe_path).is_file():
            cmd.extend(["--no-profile", "--recipe", str(recipe_path)])
            for k, v in (recipe_params or {}).items():
                cmd.extend(["--params", f"{k}={v}"])
        else:
            cmd.extend(["--system", system_guard, "--text", effective_prompt])

        env = dict(os.environ)
        env["GOOSE_PROVIDER"] = provider
        env["GOOSE_MODEL"] = resolved_model
        env["GOOSE_MODE"] = env.get("GOOSE_MODE", "smart_approve")
        if provider == "openai":
            env["OPENAI_BASE_URL"] = endpoint_url
            env["OPENAI_API_KEY"] = env.get("OPENAI_API_KEY", "local-mlx")
        else:
            env["OLLAMA_HOST"] = env.get("OLLAMA_HOST", DEFAULT_OLLAMA_URL)
        # Ensure zero GUI prompts or browser popups
        env["BROWSER"] = "false"
        env["DISPLAY"] = ""

        from whole_body_compute.organs.immune import VascularTrafficMonitor

        cmd = VascularTrafficMonitor.wrap_command_with_seatbelt(
            cmd, enable=use_zero_egress_seatbelt
        )

        tool_events: List[GooseToolEvent] = []
        collected_lines: List[str] = []
        vascular_violation: Optional[str] = None

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=str(exec_dir),
                start_new_session=True,
            )
            self.register_active_process(session_id, proc, sandbox=sandbox)
            assert proc.stdout is not None
            for line in proc.stdout:
                collected_lines.append(line)
                # Live-parse toolRequest events as they stream in
                stripped = line.strip()
                if stripped.startswith("{") and '"toolRequest"' in stripped:
                    _, live_evts = parse_goose_output_stream(
                        stripped, session_id=session_id, on_tool_event=on_tool_event
                    )
                    tool_events.extend(live_evts)
                    # Organ 05 Vascular Traffic Monitor: Live Auto-Immune Intercept
                    for evt in live_evts:
                        check = VascularTrafficMonitor.inspect_tool_event(
                            evt.tool_name, evt.arguments
                        )
                        if not check.get("allowed", True):
                            vascular_violation = check.get(
                                "reason", "Blocked by Vascular Traffic Monitor"
                            )
                            self.trigger_vagus_nerve_brake(
                                reason=f"auto_immune_intercept: {vascular_violation}"
                            )
                            break
                    if vascular_violation:
                        break

            proc.wait(timeout=timeout)
            stderr_out = (proc.stderr.read() if proc.stderr else "") or ""
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
            if sandbox is not None:
                sandbox.rollback()
            err_msg = f"Goose timed out after {timeout} seconds."
            return GooseTurnResult(
                success=False,
                spoken_summary=err_msg,
                raw_response="".join(collected_lines),
                session_id=session_id,
                model=resolved_model,
                recipe_used=str(recipe_path) if recipe_path else None,
                tool_events=tool_events,
                duration_seconds=time.time() - t0,
                error=err_msg,
                sandbox_verified=False,
                dot_memories_injected=dot_memories_count,
            )
        except Exception as exc:
            if sandbox is not None:
                sandbox.rollback()
            err_msg = f"Goose execution failed: {exc}"
            return GooseTurnResult(
                success=False,
                spoken_summary=err_msg,
                raw_response="",
                session_id=session_id,
                model=resolved_model,
                duration_seconds=time.time() - t0,
                error=err_msg,
                sandbox_verified=False,
                dot_memories_injected=dot_memories_count,
            )
        finally:
            self.unregister_active_process(session_id)

        # Phase 13.3 — Check if Vagus Nerve Brake interrupted this session
        if session_id in self._interrupted_sessions:
            self._interrupted_sessions.discard(session_id)
            if sandbox is not None and not sandbox.rolled_back:
                sandbox.rollback()
            if vascular_violation:
                halt_msg = enforce_spoken_word_budget(
                    f"Auto-immune intercept halted Goose and rolled back sandbox: {vascular_violation}",
                    max_words=24,
                )
                err_detail = f"Auto-Immune Intercept (Vagus Brake): {vascular_violation}"
            else:
                halt_msg = "Stopped Goose task and rolled back shadow sandbox."
                err_detail = "Interrupted by Vagus Nerve Brake (SIGINT)"
            return GooseTurnResult(
                success=False,
                spoken_summary=halt_msg,
                raw_response="".join(collected_lines),
                session_id=session_id,
                model=resolved_model,
                recipe_used=str(recipe_path) if recipe_path else None,
                tool_events=tool_events,
                duration_seconds=time.time() - t0,
                error=err_detail,
                sandbox_verified=False,
                interrupted=True,
                dot_memories_injected=dot_memories_count,
                bridge_crumb=recorded_crumb,
            )

        # Phase 13.2 — Verify and promote APFS CoW shadow sandbox mutations
        sandbox_verified = True
        secgate_report: Optional[Dict[str, Any]] = None
        if sandbox is not None:
            secgate_report = sandbox.verify_and_promote(run_pytest_target=pytest_verify_target)
            sandbox_verified = bool(secgate_report.get("passed", False))
            if not sandbox_verified:
                reject_reason = secgate_report.get("reason", "AST SecGate rejected change.")
                spoken_reject = enforce_spoken_word_budget(
                    f"Blocked unsafe change in shadow sandbox: {reject_reason}", max_words=24
                )
                return GooseTurnResult(
                    success=False,
                    spoken_summary=spoken_reject,
                    raw_response="".join(collected_lines),
                    session_id=session_id,
                    model=resolved_model,
                    recipe_used=str(recipe_path) if recipe_path else None,
                    tool_events=tool_events,
                    duration_seconds=time.time() - t0,
                    error=reject_reason,
                    sandbox_verified=False,
                    secgate_report=secgate_report,
                    dot_memories_injected=dot_memories_count,
                    bridge_crumb=recorded_crumb,
                )

        raw_stdout = "".join(collected_lines)
        parsed_text, all_events = parse_goose_output_stream(raw_stdout, session_id=session_id)
        if not tool_events and all_events:
            tool_events = all_events

        spoken = enforce_spoken_word_budget(parsed_text, max_words=24)

        # Fallback if model returned empty or non-natural text when deictic context was present
        if (not spoken or len(spoken.split()) < 2) and deictic_ctx:
            records = search_screen_content(
                query="",
                content_type="ocr",
                app_name="Terminal,iTerm2,Chrome,Godot",
                limit=5,
                db_path=screenpipe_db,
            )
            errs = extract_visible_errors(records)
            if errs:
                top_err = errs[0]
                spoken = enforce_spoken_word_budget(
                    f"Found {top_err['error_line']} in {top_err['app_name']} window {top_err['window_name']}.",
                    max_words=24,
                )

        # Phase 14.2 — Tactile Closed-Loop Post-Action Verification Delta (<50ms)
        tactile_delta_dict = None
        if tactile_base is not None:
            try:
                from whole_body_compute.integrations.tactile import capture_post_action_tactile_delta

                delta_res = capture_post_action_tactile_delta(tactile_base, working_dir=working_dir)
                tactile_delta_dict = delta_res.to_dict()
                if delta_res.git_changed and (not spoken or spoken == "Goose completed the local task."):
                    spoken = f"Goose completed the local task ({delta_res.summary})."
            except Exception:
                tactile_delta_dict = None

        if not spoken:
            if proc.returncode != 0:
                err_clean = enforce_spoken_word_budget(stderr_out or raw_stdout, max_words=20)
                spoken = err_clean or "Goose encountered an error while running the local model."
            else:
                spoken = "Goose completed the local task."

        if speak_aloud and not os.environ.get("PYTEST_CURRENT_TEST"):
            self._speak_summary(spoken)

        return GooseTurnResult(
            success=(proc.returncode == 0 or bool(spoken)),
            spoken_summary=spoken,
            raw_response=parsed_text or raw_stdout,
            session_id=session_id,
            model=resolved_model,
            recipe_used=str(recipe_path) if recipe_path else None,
            tool_events=tool_events,
            duration_seconds=time.time() - t0,
            error=None if proc.returncode == 0 else stderr_out.strip(),
            sandbox_verified=sandbox_verified,
            secgate_report=secgate_report,
            interrupted=False,
            dot_memories_injected=dot_memories_count,
            tactile_delta=tactile_delta_dict,
            foveal_snapshot=foveal_snap.to_dict() if foveal_snap else None,
            bridge_crumb=recorded_crumb,
        )

    def _speak_summary(self, summary: str) -> None:
        """Speak the concise summary via VoiceFi TTS engine (or macOS `say` fallback)."""
        try:
            from whole_body_compute.config import load_config
            from voicefi.tts import get_tts_engine

            cfg = load_config()
            tts = get_tts_engine(cfg, agent_name="goose")
            tts.stream_speak(summary, block=True)
            return
        except Exception:
            pass
        try:
            subprocess.run(["say", summary], stdin=subprocess.DEVNULL, timeout=15)
        except Exception:
            pass
