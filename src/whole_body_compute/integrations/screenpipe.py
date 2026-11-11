"""
VoiceFi Screenpipe Integration & Local Episodic Memory Engine (Zero-Egress).

Provides:
1. Direct client & SQLite reader/writer for Screenpipe (`http://127.0.0.1:3030` and `~/.screenpipe/db.sqlite`).
2. Headless, focus-preserving macOS screen/window OCR capture (`screencapture -x` + Vision/AppleScript)
   so VoiceFi has continuous visual & audio recall even when standalone Screenpipe is not running.
3. Zero-egress stdio MCP server exposing `search_screen_content` (and `search-content`),
   `get_latest_terminal_error`, and `get_recent_audio_transcript` to Goose CLI and VoiceFi agents.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from whole_body_compute.integrations.foveal_vision import (
        FovealUIElement,
        FovealVisionEngine,
        FovealWindowSnapshot,
        build_foveal_screen_context,
        capture_foveal_snapshot,
    )
except ImportError:
    FovealUIElement = Any  # type: ignore[misc,assignment]
    FovealVisionEngine = None  # type: ignore[assignment]
    FovealWindowSnapshot = Any  # type: ignore[misc,assignment]
    build_foveal_screen_context = None  # type: ignore[assignment]
    capture_foveal_snapshot = None  # type: ignore[assignment]

DEFAULT_SCREENPIPE_URL = "http://127.0.0.1:3030"
DEFAULT_SCREENPIPE_DB = Path.home() / ".screenpipe" / "db.sqlite"
DEFAULT_FOVEAL_MEMORY_DB = Path.home() / ".whole_body" / "foveal_memory.sqlite"

TARGET_DEBUG_APPS = ("Terminal", "iTerm2", "iTerm", "Google Chrome", "Chrome", "Godot", "Code", "Cursor")

# Regex patterns for stripping macOS menu bar / button chrome so OCR never steers agents into GUI clicking
UI_CHROME_LINE_RE = re.compile(
    r"^("
    r"(?:[A-Za-z0-9_]+\s+)?File(?:\s+Edit)?(?:\s+View)?$|"
    r"Edit\s+View(?:\s+Window)?$|"
    r"Window\s+Help$|"
    r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d+\s+\d+:\d+\s*(?:AM|PM)?|"
    r"(?:Click\s+here|Submit|Cancel|OK|Close|Minimize|Zoom|Back|Forward|Reload)$"
    r")",
    re.IGNORECASE,
)


def strip_ui_chrome_from_ocr(raw_text: str) -> str:
    """
    Remove macOS menu bar items, status bar clocks, and clickable button chrome from OCR text
    so downstream models only see actionable data payloads (paths, URLs, commands, errors, records).
    """
    if not raw_text:
        return ""
    kept: List[str] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if UI_CHROME_LINE_RE.search(line):
            continue
        if line in ("File", "Edit", "View", "Window", "Help", "History", "Bookmarks", "Profiles", "Tab"):
            continue
        # Drop pure icon-garble lines from menu bar symbols
        if len(line) < 25 and "\uffff" in line:
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def extract_semantic_screen_entities(ocr_text: str) -> Dict[str, List[str]]:
    """
    Extract structured, non-UI data entities from OCR text so agents can act directly
    via Tier 0 (Filesystem/Python) or Tier 1 (CLI/API) instead of clicking GUI coordinates.
    """
    cleaned = strip_ui_chrome_from_ocr(ocr_text)
    file_paths = list(
        dict.fromkeys(
            re.findall(
                r"(?:(?:/Users/|~/|\./|src/|tests/|recipes/|bin/|config/)[A-Za-z0-9_./-]+\.[A-Za-z0-9]+|\b[A-Za-z0-9_-]+\.(?:py|yaml|yml|json|jsonl|csv|md|sh|sql|ts|js)\b)",
                cleaned,
            )
        )
    )
    urls = list(dict.fromkeys(re.findall(r"https?://[^\s\"')>]+", cleaned)))
    commands = list(
        dict.fromkeys(
            m.group(1).strip()
            for m in re.finditer(
                r"(?:^|\n)\s*(?:[$%]|[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+\s+[^\n%$]*[%$])\s+([^\n]+)",
                cleaned,
            )
            if m.group(1).strip()
        )
    )
    channels_or_ids = list(
        dict.fromkeys(
            re.findall(
                r"(?:#[a-z0-9_-]{2,30}|\b[A-Z]{2,10}-\d{1,6}\b|\b00[a-zA-Z0-9]{18}\b)",
                cleaned,
            )
        )
    )
    return {
        "file_paths": file_paths[:10],
        "urls": urls[:5],
        "commands": commands[:5],
        "channels_or_ids": channels_or_ids[:10],
    }


def resolve_direct_headless_route(app_name: str, window_name: str, ocr_text: str = "") -> str:
    """
    Map a GUI application/window seen in Screenpipe OCR to its Tier 0 / Tier 1 Direct Headless Route
    so Goose and Hey Pal never synthesize mouse clicks when a direct CLI/API/filesystem route exists.
    """
    app_low = (app_name or "").lower()
    win_low = (window_name or "").lower()
    ocr_low = (ocr_text or "").lower()

    if any(k in win_low or k in ocr_low for k in ("bill.com", "create invoice", "qqtdaabp")):
        return (
            "DIRECT ROUTE (Tier 1 Billing API): Use `python scripts/billing.py create-invoice` "
            "(via `scripts/billcom_client.py` REST API) — never automate Bill.com login or invoice form via browser clicks."
        )
    if "finder" in app_low or ".csv" in ocr_low or "downloads" in win_low:
        return (
            "DIRECT ROUTE (Tier 0 Filesystem): Use Python `pathlib`/`shutil`/`csv` directly on disk "
            "(never drag-and-drop or click Finder windows)."
        )
    if "github" in win_low or "github.com" in ocr_low or "pull request" in ocr_low:
        return (
            "DIRECT ROUTE (Tier 1 CLI): Use `gh pr`, `gh issue`, or `git` CLI commands "
            "(never click GitHub buttons in the browser)."
        )
    if "drive" in win_low or "sheets" in win_low or "docs.google.com" in ocr_low:
        return (
            "DIRECT ROUTE (Tier 1 API/CLI): Use `gws` / `gsheets` CLI or local CSV/file manipulation "
            "(never click Google Drive/Sheets UI menus)."
        )
    if "slack" in app_low or "slack" in win_low:
        return (
            "DIRECT ROUTE (Tier 1 API): Use direct Slack API/webhook dispatch "
            "(never click or type into the Slack desktop window)."
        )
    if any(k in app_low for k in ("terminal", "iterm", "code", "cursor", "antigravity", "godot")):
        return (
            "DIRECT ROUTE (Tier 0/1 Developer): Edit files directly with `edit`/`write` and run `pytest` or shell commands "
            "in the background (never click IDE panes or terminal windows)."
        )
    return (
        "DIRECT ROUTE (Tier 0/1 Headless): Extract the data entities (paths, URLs, IDs) from OCR and execute "
        "via Python, filesystem, `curl`, `gh`, or `pbcopy` — never synthesize GUI mouse clicks."
    )


@dataclass
class ScreenpipeRecord:
    record_type: str  # "OCR" or "Audio"
    timestamp: str
    app_name: str
    window_name: str
    text: str
    confidence: float = 1.0
    frame_id: Optional[int] = None

    def to_api_item(self) -> Dict[str, Any]:
        if self.record_type.lower() == "audio":
            return {
                "type": "Audio",
                "content": {
                    "chunk_id": self.frame_id or 0,
                    "transcription": self.text,
                    "timestamp": self.timestamp,
                    "device": self.app_name or "MacBook Pro Microphone",
                    "is_input_device": True,
                },
            }
        return {
            "type": "OCR",
            "content": {
                "frame_id": self.frame_id or 0,
                "text": self.text,
                "timestamp": self.timestamp,
                "app_name": self.app_name,
                "window_name": self.window_name,
                "focused": True,
            },
        }


# Canonical alias for sovereign foveal episodic memory
FovealMemoryRecord = ScreenpipeRecord


def ensure_foveal_memory_db(db_path: Optional[Path] = None) -> Path:
    """Ensure `~/.whole_body/foveal_memory.sqlite` exists with compatible tables."""
    target = Path(db_path or os.environ.get("FOVEAL_MEMORY_DB_PATH") or DEFAULT_FOVEAL_MEMORY_DB).expanduser()
    return ensure_screenpipe_db(target)


def import_legacy_screenpipe_db(
    source_db_path: Optional[Path] = None,
    target_db_path: Optional[Path] = None,
    redact_pii: bool = True,
) -> Dict[str, Any]:
    """1-Way Trojan Horse migration from Screenpipe SQLite to sovereign Foveal Memory."""
    from whole_body_compute.integrations.foveal_memory import (
        import_legacy_screenpipe_db as _imp,
    )

    return _imp(source_db_path=source_db_path, target_db_path=target_db_path, redact_pii=redact_pii)


def ensure_screenpipe_db(db_path: Optional[Path] = None) -> Path:
    """Ensure `~/.screenpipe/db.sqlite` exists with compatible tables for OCR and Audio history."""
    target = Path(db_path or os.environ.get("SCREENPIPE_DB_PATH") or DEFAULT_SCREENPIPE_DB).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target), timeout=5.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                app_name TEXT DEFAULT '',
                window_name TEXT DEFAULT '',
                focused BOOLEAN DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS ocr_text (
                frame_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                text_json TEXT DEFAULT '{}',
                app_name TEXT DEFAULT '',
                ocr_engine TEXT DEFAULT 'apple-vision',
                window_name TEXT DEFAULT '',
                focused BOOLEAN DEFAULT 1,
                text_length INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS audio_transcriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_chunk_id INTEGER DEFAULT 0,
                offset_index INTEGER DEFAULT 0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                transcription TEXT NOT NULL,
                device TEXT DEFAULT 'default',
                is_input_device BOOLEAN DEFAULT 1,
                speaker_id INTEGER DEFAULT 0,
                transcription_engine TEXT DEFAULT 'whisper'
            );
            CREATE INDEX IF NOT EXISTS idx_frames_timestamp ON frames(timestamp);
            CREATE INDEX IF NOT EXISTS idx_ocr_frame_id ON ocr_text(frame_id);
            CREATE INDEX IF NOT EXISTS idx_audio_timestamp ON audio_transcriptions(timestamp);
            """
        )
        conn.commit()
    finally:
        conn.close()
    return target


def record_ocr_snapshot(
    text: str,
    app_name: str = "Terminal",
    window_name: str = "zsh",
    timestamp: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Insert an OCR snapshot into `~/.screenpipe/db.sqlite` without stealing focus."""
    from whole_body_compute.integrations.ambient_discovery import AmbientPrivacyGuard
    from whole_body_compute.redaction import redact_secrets
    from whole_body_compute.organs.immune import EpithelialScreenFilter

    if AmbientPrivacyGuard.is_app_blocked(app_name):
        logger.warning(f"Blocked OCR snapshot for sensitive application: {app_name}")
        return 0

    sanitized = redact_secrets(text)
    sanitized, _ = EpithelialScreenFilter.sanitize(sanitized, source="screenpipe_ocr")

    target = ensure_screenpipe_db(db_path)
    ts = timestamp or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn = sqlite3.connect(str(target), timeout=5.0)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO frames (timestamp, app_name, window_name, focused) VALUES (?, ?, ?, 1)",
            (ts, app_name, window_name),
        )
        frame_id = int(cur.lastrowid or 0)
        cur.execute(
            """
            INSERT INTO ocr_text (frame_id, text, app_name, window_name, focused, text_length)
            VALUES (?, ?, ?, ?, 1, ?)
            """,
            (frame_id, sanitized, app_name, window_name, len(sanitized)),
        )
        conn.commit()
        return frame_id
    finally:
        conn.close()


def record_audio_transcription(
    transcription: str,
    device: str = "MacBook Microphone",
    timestamp: Optional[str] = None,
    db_path: Optional[Path] = None,
    is_input_device: bool = True,
    speaker_id: int = 0,
    transcription_engine: str = "whisper",
) -> int:
    """Insert an audio transcript segment into `~/.screenpipe/db.sqlite` (with recent-utterance deduplication)."""
    clean_txt = (transcription or "").strip()
    if not clean_txt:
        return 0

    from whole_body_compute.redaction import redact_secrets
    from whole_body_compute.organs.immune import EpithelialScreenFilter

    clean_txt = redact_secrets(clean_txt)
    clean_txt, _ = EpithelialScreenFilter.sanitize(clean_txt, source="screenpipe_audio")

    target = ensure_screenpipe_db(db_path)
    ts = timestamp or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn = sqlite3.connect(str(target), timeout=5.0)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, transcription, device FROM audio_transcriptions ORDER BY id DESC LIMIT 5"
        )
        for existing_id, existing_txt, existing_dev in cur.fetchall():
            if (existing_txt or "").strip() == clean_txt and (existing_dev or "") == device:
                return int(existing_id)
        cur.execute(
            """
            INSERT INTO audio_transcriptions (timestamp, transcription, device, is_input_device, speaker_id, transcription_engine)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (ts, clean_txt, device, 1 if is_input_device else 0, speaker_id, transcription_engine),
        )
        row_id = int(cur.lastrowid or 0)
        conn.commit()
        return row_id
    finally:
        conn.close()


def record_companion_event(
    text: str,
    event_type: str = "voice_command",
    device: str = "Companion Phone Mic",
    is_input: bool = True,
    app_name: str = "VoiceFi Companion",
    window_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None,
    db_path: Optional[Path] = None,
    record_audio: bool = True,
    record_frame: bool = True,
) -> Dict[str, int]:
    """
    Record an interaction, prompt, action, or agent turn occurring over the VoiceFi Companion connection into Screenpipe.

    Dual-ingests into:
    1. `audio_transcriptions`: for voice/audio timeline and workflow audio notes (e.g. PalTrainer, Goose MCP).
    2. `frames` + `ocr_text`: for visual action timeline and workflow step reconstruction (PalTrainer.from_screenpipe_history).

    Returns a dict with `{"audio_id": int, "frame_id": int}`.
    """
    clean_txt = (text or "").strip()
    if not clean_txt:
        return {"audio_id": 0, "frame_id": 0}

    target = ensure_screenpipe_db(db_path)
    ts = timestamp or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    win = window_name or f"Mobile Companion / {event_type.replace('_', ' ').title()}"
    meta = metadata or {}
    meta_json = json.dumps(meta, default=str) if meta else "{}"

    audio_id = 0
    frame_id = 0

    if record_audio:
        try:
            audio_id = record_audio_transcription(
                transcription=clean_txt,
                device=device,
                timestamp=ts,
                db_path=target,
                is_input_device=is_input,
                transcription_engine="voicefi-companion",
            )
        except Exception as e:
            logger.warning("[Screenpipe] Error recording companion audio: %s", e)

    if record_frame:
        conn = sqlite3.connect(str(target), timeout=5.0)
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO frames (timestamp, app_name, window_name, focused) VALUES (?, ?, ?, 1)",
                (ts, app_name, win),
            )
            frame_id = int(cur.lastrowid or 0)
            frame_text = f"[{event_type.upper()}] {clean_txt}" if not clean_txt.startswith("[") else clean_txt
            cur.execute(
                """
                INSERT INTO ocr_text (frame_id, text, text_json, app_name, ocr_engine, window_name, focused, text_length)
                VALUES (?, ?, ?, ?, 'voicefi-companion', ?, 1, ?)
                """,
                (frame_id, frame_text, meta_json, app_name, win, len(frame_text)),
            )
            conn.commit()
        except Exception as e:
            logger.warning("[Screenpipe] Error recording companion frame: %s", e)
        finally:
            conn.close()

    return {"audio_id": audio_id, "frame_id": frame_id}


def get_frontmost_window_metadata() -> Tuple[str, str]:
    """Return `(app_name, window_name)` of the active on-screen macOS window in <1ms via Quartz."""
    try:
        import Quartz

        wins = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID,
        )
        if wins:
            for w in wins:
                if w.get("kCGWindowLayer", 0) == 0:
                    owner = str(w.get("kCGWindowOwnerName") or "").strip()
                    if owner and owner not in ("Window Server", "SystemUIServer", "Control Center"):
                        w_name = str(w.get("kCGWindowName") or "").strip() or owner
                        return owner, w_name
    except Exception:
        pass
    return "macOS Screen", "Active Display"


_VISION_CLASSES: Optional[Tuple[Any, Any]] = None


def _get_vision_classes() -> Tuple[Optional[Any], Optional[Any]]:
    """Cache Apple Vision framework classes once per process to avoid repeated bundle loading."""
    global _VISION_CLASSES
    if _VISION_CLASSES is not None:
        return _VISION_CLASSES
    try:
        import objc

        vision_ns: Dict[str, Any] = {}
        objc.loadBundle(
            "Vision",
            bundle_path="/System/Library/Frameworks/Vision.framework",
            module_globals=vision_ns,
        )
        _VISION_CLASSES = (
            vision_ns.get("VNRecognizeTextRequest"),
            vision_ns.get("VNImageRequestHandler"),
        )
        return _VISION_CLASSES
    except Exception:
        return (None, None)


def get_seconds_since_last_user_input() -> float:
    """Return seconds since last keyboard/mouse/trackpad event via Quartz (<0.2ms)."""
    try:
        import Quartz

        return float(
            Quartz.CGEventSourceSecondsSinceLastEventType(
                Quartz.kCGEventSourceStateCombinedSessionState,
                Quartz.kCGAnyInputEventType,
            )
        )
    except Exception:
        return 0.0


def capture_apple_vision_ocr(fast_level: int = 1) -> Optional[Tuple[str, str, str]]:
    """
    Capture the main macOS display silently (`screencapture -x -m`) without stealing focus
    and extract all visible text in ~350ms using Apple's Neural Engine `Vision.framework`
    (`VNRecognizeTextRequest`).
    Returns `(app_name, window_name, ocr_text)` or None.
    """
    if sys.platform != "darwin" or os.environ.get("PYTEST_CURRENT_TEST"):
        return None

    try:
        from Foundation import NSURL

        VNRecognizeTextRequest, VNImageRequestHandler = _get_vision_classes()
        if VNRecognizeTextRequest is None or VNImageRequestHandler is None:
            return None

        app_name, win_name = get_frontmost_window_metadata()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            subprocess.run(
                ["screencapture", "-x", "-m", tmp.name],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            )
            url = NSURL.fileURLWithPath_(tmp.name)
            handler = VNImageRequestHandler.alloc().initWithURL_options_(url, None)
            req = VNRecognizeTextRequest.alloc().init()
            req.setRecognitionLevel_(int(fast_level))  # 1 = Fast (~350ms), 0 = Accurate
            ok = handler.performRequests_error_([req], None)
            if not ok:
                return None
            obs = req.results() or []
            lines: List[str] = []
            for o in obs:
                cand = o.topCandidates_(1)
                if cand and len(cand) > 0:
                    s = str(cand[0].string()).strip()
                    if s:
                        lines.append(s)
            full_text = strip_ui_chrome_from_ocr("\n".join(lines).strip())
            if full_text:
                return (app_name, win_name, full_text)
    except Exception:
        pass
    return None


def capture_headless_window_text(include_vision_ocr: bool = True) -> List[Tuple[str, str, str]]:
    """
    Read visible text from Terminal, iTerm2, and active macOS windows (via Apple Vision OCR)
    completely in the background without changing active application focus.
    """
    if sys.platform != "darwin" or os.environ.get("PYTEST_CURRENT_TEST"):
        return []

    snapshots: List[Tuple[str, str, str]] = []
    # 1. Inspect Terminal.app contents if running (zero focus change)
    script_terminal = """
    if application "Terminal" is running then
        tell application "Terminal"
            if (count of windows) > 0 then
                set winName to name of front window
                set tabText to contents of selected tab of front window
                return winName & "|||" & tabText
            end if
        end tell
    end if
    return ""
    """
    try:
        res = subprocess.run(
            ["osascript", "-e", script_terminal],
            capture_output=True,
            text=True,
            timeout=1.5,
        )
        out = (res.stdout or "").strip()
        if "|||" in out:
            win_name, tab_text = out.split("|||", 1)
            trimmed = "\n".join(tab_text.strip().splitlines()[-40:]).strip()
            if trimmed:
                snapshots.append(("Terminal", win_name.strip(), trimmed))
    except Exception:
        pass

    # 2. Inspect iTerm2 session text if running (zero focus change)
    script_iterm = """
    if application "iTerm2" is running then
        tell application "iTerm2"
            if (count of windows) > 0 then
                tell current session of current window
                    set sessName to name
                    set sessText to text
                    return sessName & "|||" & sessText
                end tell
            end if
        end tell
    end if
    return ""
    """
    try:
        res = subprocess.run(
            ["osascript", "-e", script_iterm],
            capture_output=True,
            text=True,
            timeout=1.5,
        )
        out = (res.stdout or "").strip()
        if "|||" in out:
            win_name, tab_text = out.split("|||", 1)
            trimmed = "\n".join(tab_text.strip().splitlines()[-40:]).strip()
            if trimmed:
                snapshots.append(("iTerm2", win_name.strip(), trimmed))
    except Exception:
        pass

    # 3. Capture full-screen Apple Vision Neural Engine OCR for Chrome, Godot, PDFs, Slack, etc.
    if include_vision_ocr:
        vis = capture_apple_vision_ocr(fast_level=1)
        if vis is not None:
            snapshots.append(vis)

    return snapshots


def search_screenpipe_sqlite(
    query: str = "",
    content_type: str = "all",
    app_name: Optional[str] = None,
    minutes_ago: Optional[int] = None,
    limit: int = 10,
    db_path: Optional[Path] = None,
    start_time: Optional[str] = None,
) -> List[ScreenpipeRecord]:
    """
    Search `~/.screenpipe/db.sqlite` directly for OCR and/or Audio history.
    """
    target = ensure_screenpipe_db(db_path)
    records: List[ScreenpipeRecord] = []

    cutoff_iso: Optional[str] = start_time.strip() if start_time and start_time.strip() else None
    if not cutoff_iso and minutes_ago is not None and minutes_ago > 0:
        cutoff_dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=minutes_ago)
        cutoff_iso = cutoff_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    conn = sqlite3.connect(str(target), timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        ctype = (content_type or "all").lower()
        q_like = f"%{query.strip()}%" if query and query.strip() else "%"

        if ctype in ("all", "ocr"):
            sql = """
                SELECT
                    f.id AS frame_id,
                    f.timestamp AS timestamp,
                    COALESCE(NULLIF(o.app_name, ''), f.app_name, 'Unknown') AS app_name,
                    COALESCE(NULLIF(o.window_name, ''), f.window_name, '') AS window_name,
                    o.text AS text
                FROM ocr_text o
                LEFT JOIN frames f ON o.frame_id = f.id
                WHERE o.text LIKE ?
            """
            params: List[Any] = [q_like]
            if app_name:
                apps = [a.strip() for a in app_name.split(",") if a.strip()]
                if len(apps) == 1:
                    sql += " AND (LOWER(o.app_name) LIKE LOWER(?) OR LOWER(f.app_name) LIKE LOWER(?))"
                    params.extend([f"%{apps[0]}%", f"%{apps[0]}%"])
                elif len(apps) > 1:
                    clauses = " OR ".join(
                        "(LOWER(o.app_name) LIKE LOWER(?) OR LOWER(f.app_name) LIKE LOWER(?))"
                        for _ in apps
                    )
                    sql += f" AND ({clauses})"
                    for a in apps:
                        params.extend([f"%{a}%", f"%{a}%"])
            if cutoff_iso:
                sql += " AND f.timestamp >= ?"
                params.append(cutoff_iso)
            sql += " ORDER BY f.id DESC LIMIT ?"
            params.append(int(limit))

            for row in conn.execute(sql, params):
                records.append(
                    ScreenpipeRecord(
                        record_type="OCR",
                        timestamp=str(row["timestamp"] or ""),
                        app_name=str(row["app_name"] or ""),
                        window_name=str(row["window_name"] or ""),
                        text=str(row["text"] or ""),
                        frame_id=int(row["frame_id"] or 0),
                    )
                )

        if ctype in ("all", "audio"):
            sql_audio = """
                SELECT id, timestamp, device, transcription
                FROM audio_transcriptions
                WHERE transcription LIKE ?
            """
            params_audio: List[Any] = [q_like]
            if app_name:
                apps = [a.strip() for a in app_name.split(",") if a.strip()]
                if apps:
                    clauses = []
                    for a in apps:
                        clauses.append("LOWER(device) LIKE LOWER(?)")
                        params_audio.append(f"%{a}%")
                        if "companion" in a.lower():
                            clauses.append("LOWER(device) LIKE '%companion%'")
                    if clauses:
                        sql_audio += f" AND ({' OR '.join(clauses)})"
            if cutoff_iso:
                sql_audio += " AND timestamp >= ?"
                params_audio.append(cutoff_iso)
            sql_audio += " ORDER BY id DESC LIMIT ?"
            params_audio.append(int(limit))

            for row in conn.execute(sql_audio, params_audio):
                records.append(
                    ScreenpipeRecord(
                        record_type="Audio",
                        timestamp=str(row["timestamp"] or ""),
                        app_name=str(row["device"] or "Microphone"),
                        window_name="Audio Capture",
                        text=str(row["transcription"] or ""),
                        frame_id=int(row["id"] or 0),
                    )
                )
    finally:
        conn.close()

    return records[:limit]


def check_screenpipe_health(base_url: str = DEFAULT_SCREENPIPE_URL, timeout: float = 1.5) -> bool:
    """Return True if a Screenpipe HTTP server is responding on localhost:3030."""
    for path in ("/health", "/search?limit=1"):
        try:
            req = urllib.request.Request(f"{base_url.rstrip('/')}{path}", method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if 200 <= resp.status < 300:
                    return True
        except Exception:
            continue
    return False


def search_screen_content(
    query: str = "",
    content_type: str = "all",
    app_name: Optional[str] = None,
    minutes_ago: Optional[int] = None,
    limit: int = 10,
    base_url: str = DEFAULT_SCREENPIPE_URL,
    db_path: Optional[Path] = None,
    capture_live_if_empty: bool = True,
) -> List[ScreenpipeRecord]:
    """
    Primary entry point for searching Screenpipe visual & audio memory.
    1. Queries `http://127.0.0.1:3030/search` if reachable.
    2. Falls back to (or merges with) `~/.screenpipe/db.sqlite`.
    3. Optionally captures live headless Terminal/iTerm2 content if no recent OCR frames match.
    """
    results: List[ScreenpipeRecord] = []
    cutoff_iso: Optional[str] = None
    if minutes_ago is not None and minutes_ago > 0:
        start_dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=minutes_ago)
        cutoff_iso = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # 1. Try localhost:3030 REST endpoint first (unless custom db_path or SCREENPIPE_DB_PATH is specified)
    if db_path is None and "SCREENPIPE_DB_PATH" not in os.environ:
        try:
            params: Dict[str, Any] = {"limit": limit, "content_type": content_type or "all"}
            if query:
                params["q"] = query
            if app_name and "," not in app_name:
                params["app_name"] = app_name
            if cutoff_iso:
                params["start_time"] = cutoff_iso
                params["minutes_ago"] = minutes_ago

            qs = urllib.parse.urlencode(params)
            req = urllib.request.Request(f"{base_url.rstrip('/')}/search?{qs}", method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    payload = json.loads(resp.read().decode("utf-8"))
                    for item in payload.get("data", []):
                        itype = item.get("type", "OCR")
                        content = item.get("content", {})
                        ts_val = str(content.get("timestamp", ""))
                        if cutoff_iso and ts_val and ts_val < cutoff_iso:
                            continue
                        if itype == "Audio":
                            results.append(
                                ScreenpipeRecord(
                                    record_type="Audio",
                                    timestamp=ts_val,
                                    app_name=str(content.get("device", "Microphone")),
                                    window_name="Audio",
                                    text=str(content.get("transcription", "")),
                                    frame_id=content.get("chunk_id"),
                                )
                            )
                        else:
                            results.append(
                                ScreenpipeRecord(
                                    record_type="OCR",
                                    timestamp=ts_val,
                                    app_name=str(content.get("app_name", "")),
                                    window_name=str(content.get("window_name", "")),
                                    text=str(content.get("text", "")),
                                    frame_id=content.get("frame_id"),
                                )
                            )
        except Exception:
            pass

    # 2. Query local SQLite directly if HTTP returned nothing or had app list filters
    if not results:
        results = search_screenpipe_sqlite(
            query=query,
            content_type=content_type,
            app_name=app_name,
            minutes_ago=minutes_ago,
            limit=limit,
            db_path=db_path,
        )

    # 3. If still empty and looking for OCR, capture headless live window text
    if not results and capture_live_if_empty and content_type.lower() in ("all", "ocr"):
        from whole_body_compute.integrations.ambient_discovery import AmbientPrivacyGuard

        for snap_app, snap_win, snap_text in capture_headless_window_text():
            if AmbientPrivacyGuard.is_app_blocked(snap_app):
                continue
            if query and query.lower() not in snap_text.lower():
                continue
            fid = record_ocr_snapshot(snap_text, app_name=snap_app, window_name=snap_win, db_path=db_path)
            if fid > 0:
                results.append(
                    ScreenpipeRecord(
                        record_type="OCR",
                        timestamp=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        app_name=snap_app,
                        window_name=snap_win,
                        text=snap_text,
                        frame_id=fid,
                    )
                )

    # Sanitize egress: scrub secrets and prompt injections, filter blocked apps
    from whole_body_compute.integrations.ambient_discovery import AmbientPrivacyGuard
    from whole_body_compute.redaction import redact_secrets
    from whole_body_compute.organs.immune import EpithelialScreenFilter

    sanitized_results: List[ScreenpipeRecord] = []
    for r in results:
        if AmbientPrivacyGuard.is_app_blocked(r.app_name):
            continue
        c_text = redact_secrets(r.text)
        c_text, _ = EpithelialScreenFilter.sanitize(c_text, source=f"screenpipe_{r.record_type.lower()}")
        r.text = c_text
        sanitized_results.append(r)

    return sanitized_results[:limit]


def extract_visible_errors(records: List[ScreenpipeRecord]) -> List[Dict[str, str]]:
    """
    Scan OCR records for stack traces, exceptions, compiler errors, or failed commands.
    """
    error_patterns = re.compile(
        r"(Traceback \(most recent call last\)|"
        r"[A-Za-z0-9_]*Error:|"
        r"[A-Za-z0-9_]*Exception:|"
        r"FAILED\s+|"
        r"panic:|"
        r"error\[E\d+\]:|"
        r"SyntaxError:|TypeError:|ValueError:|KeyError:|ImportError:|FileNotFoundError:|ZeroDivisionError:)",
        re.IGNORECASE,
    )
    specific_exc_re = re.compile(
        r"([A-Za-z0-9_]*Error:.*|[A-Za-z0-9_]*Exception:.*|FAILED\s+.*|panic:.*|error\[E\d+\]:.*)",
        re.IGNORECASE,
    )
    extracted: List[Dict[str, str]] = []
    for rec in records:
        if rec.record_type != "OCR":
            continue
        lines = rec.text.splitlines()
        for idx, line in enumerate(lines):
            if error_patterns.search(line):
                start = max(0, idx - 3)
                end = min(len(lines), idx + 12)
                snippet = "\n".join(lines[start:end]).strip()
                best_line = line.strip()
                for sline in lines[start:end]:
                    if specific_exc_re.search(sline.strip()):
                        best_line = sline.strip()
                        break
                extracted.append(
                    {
                        "app_name": rec.app_name,
                        "window_name": rec.window_name,
                        "timestamp": rec.timestamp,
                        "error_line": best_line,
                        "snippet": snippet,
                    }
                )
                break
    return extracted


class _ScreenpipeHTTPHandler(BaseHTTPRequestHandler):
    """Lightweight localhost:3030 Screenpipe-compatible HTTP handler backed by `~/.screenpipe/db.sqlite`."""

    db_path: Path = DEFAULT_SCREENPIPE_DB

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress noisy access logs in headless daemon mode
        return

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/health":
            self._send_json(
                200,
                {
                    "status": "healthy",
                    " status_code": 200,
                    "frame_status": "ok",
                    "audio_status": "ok",
                    "message": "VoiceFi Local Screenpipe Service healthy",
                },
            )
            return

        if parsed.path == "/search":
            qs = urllib.parse.parse_qs(parsed.query)
            q = (qs.get("q") or [""])[0]
            ctype = (qs.get("content_type") or ["all"])[0]
            app_name = (qs.get("app_name") or [None])[0]
            start_time = (qs.get("start_time") or [None])[0]
            minutes_ago_str = (qs.get("minutes_ago") or [None])[0]
            minutes_ago_val: Optional[int] = None
            if minutes_ago_str:
                try:
                    minutes_ago_val = int(minutes_ago_str)
                except ValueError:
                    minutes_ago_val = None
            limit_str = (qs.get("limit") or ["10"])[0]
            try:
                limit = max(1, min(100, int(limit_str)))
            except ValueError:
                limit = 10

            recs = search_screenpipe_sqlite(
                query=q,
                content_type=ctype,
                app_name=app_name,
                minutes_ago=minutes_ago_val,
                start_time=start_time,
                limit=limit,
                db_path=self.db_path,
            )
            self._send_json(
                200,
                {
                    "data": [r.to_api_item() for r in recs],
                    "pagination": {"limit": limit, "offset": 0, "total": len(recs)},
                },
            )
            return

        self._send_json(404, {"error": f"Unknown path: {parsed.path}"})

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0") or "0")
        body_raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            body = json.loads(body_raw)
        except Exception:
            body = {}

        if parsed.path == "/add_ocr":
            fid = record_ocr_snapshot(
                text=str(body.get("text", "")),
                app_name=str(body.get("app_name", "Terminal")),
                window_name=str(body.get("window_name", "zsh")),
                db_path=self.db_path,
            )
            self._send_json(200, {"ok": True, "frame_id": fid})
            return

        if parsed.path == "/add_audio":
            aid = record_audio_transcription(
                transcription=str(body.get("transcription", "")),
                device=str(body.get("device", "Microphone")),
                db_path=self.db_path,
            )
            self._send_json(200, {"ok": True, "audio_id": aid})
            return

        self._send_json(404, {"error": f"Unknown POST path: {parsed.path}"})


def serve_local_screenpipe_http(
    host: str = "127.0.0.1",
    port: int = 3030,
    db_path: Optional[Path] = None,
    poll_windows: bool = True,
) -> ThreadingHTTPServer:
    """Start the zero-egress Screenpipe HTTP service on 127.0.0.1:3030."""
    resolved_db = ensure_screenpipe_db(db_path)

    class BoundHandler(_ScreenpipeHTTPHandler):
        pass

    BoundHandler.db_path = resolved_db
    server = ThreadingHTTPServer((host, port), BoundHandler)

    if poll_windows:
        def _window_poller() -> None:
            last_seen: Dict[str, str] = {}
            last_window_sig: Tuple[str, str] = ("", "")
            sleep_sec: float = 15.0
            while True:
                try:
                    cur_sig = get_frontmost_window_metadata()
                    idle_sec = get_seconds_since_last_user_input()
                    # Skip expensive screencapture + Vision OCR when user is idle and window hasn't changed
                    if idle_sec >= 45.0 and cur_sig == last_window_sig and last_seen:
                        time.sleep(30.0)
                        continue

                    changed = False
                    for app_n, win_n, txt in capture_headless_window_text():
                        if txt and last_seen.get(app_n) != txt:
                            last_seen[app_n] = txt
                            changed = True
                            record_ocr_snapshot(txt, app_name=app_n, window_name=win_n, db_path=resolved_db)

                    if cur_sig != last_window_sig or changed:
                        last_window_sig = cur_sig
                        sleep_sec = 15.0
                    else:
                        sleep_sec = min(45.0, sleep_sec + 15.0)
                except Exception:
                    pass
                time.sleep(sleep_sec)

        t = threading.Thread(target=_window_poller, daemon=True, name="ScreenpipeWindowPoller")
        t.start()

    return server


def perform_ax_action(
    app_name: str,
    role: str,
    title: str = "",
    action: str = "AXPress",
    value: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Deterministic Tier-2 accessibility action via PyObjC AXUIElement (<10ms, no cursor movement).
    Records Organ 09 (The Hands) telemetry in ShieldTelemetryLedger. # [direct-edit]
    """
    res = _perform_ax_action_impl(
        app_name=app_name,
        role=role,
        title=title,
        action=action,
        value=value,
    )
    try:
        from whole_body_compute.shield import record_shield_event

        sec_blocked = 1 if "Security violation" in str(res.get("error", "")) else 0
        record_shield_event(
            event_type="hands_ax_act",
            target=f"{app_name or 'app'}:{action}",
            raw_tokens_est=3600,
            compressed_tokens_est=35,
            tokens_saved=3565,
            secrets_blocked=sec_blocked,
            latency_ms=float(res.get("latency_ms", 0.0)),
        )
    except Exception:
        pass
    return res


def _perform_ax_action_impl(
    app_name: str,
    role: str,
    title: str = "",
    action: str = "AXPress",
    value: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Deterministic Tier-2 accessibility action via PyObjC AXUIElement (<10ms, no cursor movement).
    Checks AmbientPrivacyGuard before accessing password managers or writing to secure fields.
    """
    t0 = time.perf_counter()
    from whole_body_compute.integrations.ambient_discovery import AmbientPrivacyGuard

    if AmbientPrivacyGuard.is_app_blocked(app_name):
        return {
            "success": False,
            "error": f"Security violation: Application '{app_name}' is in BLOCKED_PASSWORD_MANAGERS.",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    role_clean = (role or "").strip()
    title_clean = (title or "").strip()

    if action == "AXSetValue" and any(
        k in role_clean.lower() or k in title_clean.lower() for k in ("password", "secure")
    ):
        return {
            "success": False,
            "error": "Security violation: Setting value into secure/password field is forbidden by AmbientPrivacyGuard.",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    # Headless / mock testing bypass
    if os.environ.get("VOICEFI_MOCK_FOVEAL") == "1" or os.environ.get("VOICEFI_HEADLESS") == "1":
        return {
            "success": True,
            "message": f"[MOCK] Successfully performed '{action}' on {role_clean} '{title_clean}' in {app_name}.",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    try:
        import ApplicationServices as AS
        from AppKit import NSWorkspace
    except ImportError:
        return {
            "success": False,
            "error": "PyObjC ApplicationServices / AppKit not available.",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    ws = NSWorkspace.sharedWorkspace()
    target_app = None
    for a in ws.runningApplications():
        n = a.localizedName()
        if n and (n.lower() == app_name.lower() or app_name.lower() in n.lower()):
            target_app = a
            break

    if not target_app:
        return {
            "success": False,
            "error": f"Application '{app_name}' is not running.",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    app_pid = int(target_app.processIdentifier())
    app_ax = AS.AXUIElementCreateApplication(app_pid)
    if not app_ax:
        return {
            "success": False,
            "error": f"Failed to acquire AXUIElement for '{app_name}' (PID: {app_pid}).",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    target_win = None
    for attr in ("AXFocusedWindow", "AXMainWindow"):
        err, win = AS.AXUIElementCopyAttributeValue(app_ax, attr, None)
        if err == 0 and win:
            target_win = win
            break

    if not target_win:
        err, wins = AS.AXUIElementCopyAttributeValue(app_ax, "AXWindows", None)
        if err == 0 and wins:
            target_win = wins[0]

    root = target_win or app_ax
    from collections import deque

    queue = deque([root])
    matched_elem = None
    target_role = role_clean.lower()
    target_title = title_clean.lower()

    checked = 0
    curr_role = ""
    curr_title = ""
    while queue and checked < 150:
        curr = queue.popleft()
        checked += 1

        err, r = AS.AXUIElementCopyAttributeValue(curr, "AXRole", None)
        curr_role = str(r or "").lower()

        err, t = AS.AXUIElementCopyAttributeValue(curr, "AXTitle", None)
        curr_title = str(t or "").lower()
        if not curr_title:
            err, d = AS.AXUIElementCopyAttributeValue(curr, "AXDescription", None)
            curr_title = str(d or "").lower()

        role_matches = not target_role or target_role in curr_role
        title_matches = not target_title or target_title in curr_title
        if role_matches and title_matches and (target_title or target_role):
            matched_elem = curr
            break

        err, children = AS.AXUIElementCopyAttributeValue(curr, "AXChildren", None)
        if err == 0 and children:
            for ch in children:
                queue.append(ch)

    if not matched_elem:
        return {
            "success": False,
            "error": f"Element matching role='{role}' title='{title}' not found in '{app_name}'.",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    err, subrole = AS.AXUIElementCopyAttributeValue(matched_elem, "AXSubrole", None)
    elem_dict = {"role": str(curr_role), "subrole": str(subrole or ""), "title": str(curr_title)}
    if AmbientPrivacyGuard.is_element_secure(elem_dict):
        if action == "AXSetValue":
            return {
                "success": False,
                "error": "Security violation: Setting value into secure/password field is forbidden by AmbientPrivacyGuard.",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            }

    if action == "AXSetValue":
        set_err = AS.AXUIElementSetAttributeValue(matched_elem, "AXValue", value or "")
        success = (set_err == 0)
        action_err = set_err
    else:
        act_err = AS.AXUIElementPerformAction(matched_elem, action)
        success = (act_err == 0)
        action_err = act_err

    latency = round((time.perf_counter() - t0) * 1000.0, 3)
    if success:
        return {
            "success": True,
            "message": f"Successfully performed '{action}' on {role} '{title}' in {app_name}.",
            "latency_ms": latency,
        }
    else:
        return {
            "success": False,
            "error": f"AX action '{action}' returned error code {action_err}.",
            "latency_ms": latency,
        }


class ScreenpipeMCPServer:
    """
    Zero-Egress Stdio MCP Server for Goose (`screenpipe-mcp`).
    Exposes `search_screen_content`, `search-content`, `get_latest_terminal_error`,
    and `get_recent_audio_transcript` over JSON-RPC 2.0.
    """

    def __init__(self, db_path: Optional[Path] = None, base_url: str = DEFAULT_SCREENPIPE_URL):
        self.db_path = ensure_screenpipe_db(db_path)
        self.base_url = base_url

    def get_tools(self) -> List[Dict[str, Any]]:
        search_schema = {
            "type": "object",
            "properties": {
                "q": {
                    "type": "string",
                    "description": "Search query keyword or error string (leave empty for most recent).",
                },
                "query": {
                    "type": "string",
                    "description": "Alias for q (search query).",
                },
                "content_type": {
                    "type": "string",
                    "enum": ["all", "ocr", "audio"],
                    "default": "all",
                    "description": "Filter by 'ocr' (screen text), 'audio' (spoken transcripts), or 'all'.",
                },
                "app_name": {
                    "type": "string",
                    "description": "Optional app name filter (e.g., 'Terminal', 'iTerm2', 'Chrome', 'Godot').",
                },
                "minutes_ago": {
                    "type": "integer",
                    "description": "Look back window in minutes (e.g., 2 for last 2 minutes).",
                },
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "description": "Max records to return.",
                },
            },
        }
        return [
            {
                "name": "search_screen_content",
                "description": (
                    "Search Screenpipe local OCR and audio history. Use whenever the user references "
                    "'this error', 'on my screen', 'what I was just looking at', or 'that meeting'."
                ),
                "inputSchema": search_schema,
            },
            {
                "name": "search-content",
                "description": "Alias for search_screen_content (Screenpipe MCP compatibility).",
                "inputSchema": search_schema,
            },
            {
                "name": "get_latest_terminal_error",
                "description": (
                    "Queries recent Screenpipe OCR from Terminal, iTerm2, Chrome, or Godot and extracts "
                    "any visible stack trace, exception, or error message."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "minutes_ago": {
                            "type": "integer",
                            "default": 5,
                            "description": "How many minutes back to search (default: 5).",
                        },
                        "app_name": {
                            "type": "string",
                            "default": "Terminal,iTerm2,Chrome,Godot",
                            "description": "Comma-separated application names to inspect.",
                        },
                    },
                },
            },
            {
                "name": "get_recent_audio_transcript",
                "description": "Fetches the last X minutes of Screenpipe audio transcriptions for meeting notes or action items.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "minutes_ago": {
                            "type": "integer",
                            "default": 30,
                            "description": "Minutes of audio transcription to retrieve.",
                        },
                        "limit": {
                            "type": "integer",
                            "default": 50,
                        },
                    },
                },
            },
            {
                "name": "capture_live_screen_ocr",
                "description": (
                    "Captures the current macOS display headlessly in ~350ms using Apple Vision Neural Engine "
                    "(VNRecognizeTextRequest) without stealing focus, stores it in ~/.screenpipe/db.sqlite, "
                    "and returns all visible text."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "accurate": {
                            "type": "boolean",
                            "default": False,
                            "description": "Use accurate (0) vs fast (1) Vision recognition level.",
                        }
                    },
                },
            },
            {
                "name": "inspect_foveal_window_dom",
                "description": (
                    "Phase 14.1 — Zero-GPU Foveal Vision: Captures the active macOS window's structured "
                    "interactive Accessibility DOM in <5ms (buttons, text inputs, sliders, exact bounding boxes, "
                    "focused/enabled states, and direct headless routes) via PyObjC ApplicationServices (AXUIElement)."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "max_elements": {
                            "type": "integer",
                            "default": 25,
                            "description": "Maximum number of interactive UI elements to return.",
                        }
                    },
                },
            },
            {
                "name": "ask_screen_local",
                "description": (
                    "Query recent Screenpipe OCR/audio history and summarize it on-device using a local model "
                    "(:9380 SpecialistRouter / :9379 MLX / Ollama via HybridRouter) without dumping raw multi-frame OCR to cloud context."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Question to answer from recent screen or audio context.",
                        },
                        "app_name": {
                            "type": "string",
                            "description": "Optional application name to focus on (e.g. 'Terminal', 'Chrome').",
                        },
                        "minutes_back": {
                            "type": "integer",
                            "default": 5,
                            "description": "How many minutes of recent screen/audio context to inspect.",
                        },
                    },
                    "required": ["question"],
                },
            },
            {
                "name": "ax_act",
                "description": (
                    "Perform deterministic Tier-2 accessibility action in <10ms without cursor movement "
                    "via PyObjC ApplicationServices (AXUIElement). Supports AXPress, AXSetValue, and AXConfirm."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "app_name": {
                            "type": "string",
                            "description": "Target application name (e.g. 'Google Chrome', 'Terminal').",
                        },
                        "role": {
                            "type": "string",
                            "description": "AXUIElement role (e.g. 'AXButton', 'AXTextField').",
                        },
                        "title": {
                            "type": "string",
                            "description": "Title, label, or description of the target UI element.",
                        },
                        "action": {
                            "type": "string",
                            "enum": ["AXPress", "AXSetValue", "AXConfirm"],
                            "default": "AXPress",
                            "description": "Action to perform on target element.",
                        },
                        "value": {
                            "type": "string",
                            "description": "Value to set if action is AXSetValue.",
                        },
                    },
                    "required": ["app_name", "role", "action"],
                },
            },
        ]

    def _query_local_screen_summary(self, question: str, context_block: str) -> str:
        """Route to local model via HybridRouter (:9380 / :9379 / Ollama)."""
        prompt = (
            f"You are the VoiceFi Local Screen Analyst running on Apple Silicon.\n"
            f"Based strictly on the following recent screen/audio context, answer the user's question concisely in 1-3 sentences without leaking secrets.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        selected_model = "qwen2.5-coder:1.5b"
        try:
            from whole_body_compute.local.router import default_router
            decision = default_router.resolve_model(task_type="scout")
            if decision and decision.selected_model:
                selected_model = decision.selected_model.replace("cloud:", "").strip()
        except Exception:
            pass

        # 1. Try :9380 SpecialistRouter
        try:
            body = json.dumps({
                "model": "gemma4-26b",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 160,
                "temperature": 0.1,
            }).encode("utf-8")
            req = urllib.request.Request(
                "http://127.0.0.1:9380/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    ans = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    if ans:
                        return f"Local Screen Summary: {ans}"
        except Exception:
            pass

        # 2. Try :9379 MLX
        try:
            body = json.dumps({
                "model": "voicefi-coder-mlx",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 160,
                "temperature": 0.1,
            }).encode("utf-8")
            req = urllib.request.Request(
                "http://127.0.0.1:9379/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    ans = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    if ans:
                        return f"Local Screen Summary: {ans}"
        except Exception:
            pass

        # 3. Try Ollama :11434
        for cand_model in (selected_model, "tev1:0.8b", "qwen2.5-coder:1.5b"):
            try:
                body = json.dumps({
                    "model": cand_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 120, "temperature": 0.1},
                }).encode("utf-8")
                req = urllib.request.Request(
                    "http://127.0.0.1:11434/api/generate",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        ans = data.get("response", "").strip()
                        if ans:
                            return f"Local Screen Summary: {ans}"
            except Exception:
                continue

        # 4. Fallback keyword heuristic
        q_words = [w.lower() for w in re.findall(r"[A-Za-z0-9_]+", question) if len(w) > 2]
        matches = []
        for line in context_block.splitlines():
            line_lower = line.lower()
            if any(w in line_lower for w in q_words):
                matches.append(line.strip())
        if matches:
            return f"Local Screen Summary: " + " | ".join(matches[:3])
        return f"Local Screen Summary: Evaluated {len(context_block.splitlines())} lines of screen context."

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool and ensure all returned text is sanitized against secrets and prompt injections."""
        res = self._dispatch_tool(name, arguments)
        from whole_body_compute.redaction import redact_secrets
        from whole_body_compute.organs.immune import EpithelialScreenFilter

        if isinstance(res, dict) and "content" in res and isinstance(res["content"], list):
            for item in res["content"]:
                if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                    raw = str(item["text"])
                    scrubbed = redact_secrets(raw)
                    scrubbed, _ = EpithelialScreenFilter.sanitize(scrubbed, source=f"screenpipe_mcp_{name}")
                    item["text"] = scrubbed
        return res

    def _dispatch_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        args = arguments or {}
        if name in ("search_screen_content", "search-content"):
            q = str(args.get("q") or args.get("query") or "").strip()
            ctype = str(args.get("content_type") or "all")
            app_name = args.get("app_name")
            minutes_ago = args.get("minutes_ago")
            limit = int(args.get("limit") or 10)

            records = search_screen_content(
                query=q,
                content_type=ctype,
                app_name=str(app_name) if app_name else None,
                minutes_ago=int(minutes_ago) if minutes_ago is not None else None,
                limit=limit,
                base_url=self.base_url,
                db_path=self.db_path,
            )
            if not records:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": "No matching Screenpipe OCR or audio records found in the requested window.",
                        }
                    ]
                }
            formatted = []
            for r in records:
                clean_txt = strip_ui_chrome_from_ocr(r.text) if r.record_type == "OCR" else r.text
                route_hint = (
                    f"\n{resolve_direct_headless_route(r.app_name, r.window_name, clean_txt)}"
                    if r.record_type == "OCR"
                    else ""
                )
                entities = extract_semantic_screen_entities(clean_txt) if r.record_type == "OCR" else {}
                entity_parts = [f"{k}={v}" for k, v in entities.items() if v]
                entity_line = f"\nExtracted Entities: {', '.join(entity_parts)}" if entity_parts else ""
                formatted.append(
                    f"[{r.timestamp}] ({r.record_type}) {r.app_name} — {r.window_name}:{entity_line}{route_hint}\n{clean_txt}"
                )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Results: {len(records)}\n\n" + "\n---\n".join(formatted),
                    }
                ]
            }

        if name == "get_latest_terminal_error":
            minutes_ago = int(args.get("minutes_ago") or 5)
            app_filter = str(args.get("app_name") or "Terminal,iTerm2,Chrome,Godot")
            records = search_screen_content(
                query="",
                content_type="ocr",
                app_name=app_filter,
                minutes_ago=minutes_ago,
                limit=20,
                base_url=self.base_url,
                db_path=self.db_path,
            )
            if not records:
                # Fallback without strict time cutoff so recent errors aren't missed
                records = search_screen_content(
                    query="",
                    content_type="ocr",
                    app_name=app_filter,
                    minutes_ago=None,
                    limit=10,
                    base_url=self.base_url,
                    db_path=self.db_path,
                )
            errors = extract_visible_errors(records)
            if errors:
                top = errors[0]
                text = (
                    f"Latest Error in {top['app_name']} ({top['window_name']}) at {top['timestamp']}:\n"
                    f"Error Line: {top['error_line']}\n\n"
                    f"Stack Trace / Context:\n{top['snippet']}"
                )
                return {"content": [{"type": "text", "text": text}]}
            if records:
                latest = records[0]
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Most recent screen OCR in {latest.app_name} ({latest.window_name}):\n"
                                f"{latest.text}"
                            ),
                        }
                    ]
                }
            return {"content": [{"type": "text", "text": "No terminal or application errors found on screen."}]}

        if name == "get_recent_audio_transcript":
            minutes_ago = int(args.get("minutes_ago") or 30)
            limit = int(args.get("limit") or 50)
            records = search_screen_content(
                query="",
                content_type="audio",
                minutes_ago=minutes_ago,
                limit=limit,
                base_url=self.base_url,
                db_path=self.db_path,
                capture_live_if_empty=False,
            )
            if not records:
                records = search_screen_content(
                    query="",
                    content_type="audio",
                    minutes_ago=None,
                    limit=limit,
                    base_url=self.base_url,
                    db_path=self.db_path,
                    capture_live_if_empty=False,
                )
            if not records:
                return {"content": [{"type": "text", "text": "No recent audio transcriptions found in Screenpipe."}]}
            lines = [f"[{r.timestamp}] ({r.app_name}): {r.text}" for r in reversed(records)]
            return {"content": [{"type": "text", "text": "\n".join(lines)}]}

        if name == "capture_live_screen_ocr":
            accurate = bool(args.get("accurate", False))
            vis = capture_apple_vision_ocr(fast_level=0 if accurate else 1)
            if vis is not None:
                app_n, win_n, txt = vis
                fid = record_ocr_snapshot(txt, app_name=app_n, window_name=win_n, db_path=self.db_path)
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"[Frame #{fid}] Live Apple Vision OCR ({app_n} — {win_n}):\n{txt}",
                        }
                    ]
                }
            return {"content": [{"type": "text", "text": "No live screen text captured."}]}

        if name == "inspect_foveal_window_dom":
            max_elems = int(args.get("max_elements") or 25)
            snapshot = capture_foveal_snapshot(max_elements=max_elems)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": snapshot.to_context_block(max_elements=max_elems),
                    }
                ]
            }

        if name == "ask_screen_local":
            question = str(args.get("question") or "").strip()
            if not question:
                return {
                    "isError": True,
                    "content": [{"type": "text", "text": "Question parameter is required for ask_screen_local."}],
                }
            app_filter = args.get("app_name")
            minutes_back = int(args.get("minutes_back") or args.get("minutes_ago") or 5)

            records = search_screen_content(
                query="",
                content_type="all",
                app_name=str(app_filter) if app_filter else None,
                minutes_ago=minutes_back,
                limit=15,
                base_url=self.base_url,
                db_path=self.db_path,
            )
            if not records:
                records = search_screen_content(
                    query="",
                    content_type="all",
                    app_name=str(app_filter) if app_filter else None,
                    minutes_ago=None,
                    limit=5,
                    base_url=self.base_url,
                    db_path=self.db_path,
                )

            if not records:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": "No recent screen OCR or audio records found to analyze.",
                        }
                    ]
                }

            context_lines = []
            for r in records:
                t = strip_ui_chrome_from_ocr(r.text) if r.record_type == "OCR" else r.text
                context_lines.append(f"[{r.timestamp}] {r.app_name} ({r.window_name}): {t}")
            context_block = "\n".join(context_lines)
            if len(context_block) > 8000:
                context_block = context_block[:5000] + "\n...[truncated]...\n" + context_block[-2000:]

            t_screen0 = time.perf_counter()
            summary = self._query_local_screen_summary(question, context_block)
            try:
                from whole_body_compute.shield import record_shield_event

                raw_scr_tok = max(1, len(context_block) // 4)
                comp_scr_tok = max(1, len(summary) // 4)
                record_shield_event(
                    event_type="screen_local_digest",
                    target=f"ask_screen_local:{app_filter or 'all'}",
                    raw_tokens_est=raw_scr_tok,
                    compressed_tokens_est=comp_scr_tok,
                    tokens_saved=max(0, raw_scr_tok - comp_scr_tok),
                    secrets_blocked=0,
                    latency_ms=round((time.perf_counter() - t_screen0) * 1000.0, 2),
                )  # [direct-edit]
            except Exception:
                pass
            return {
                "content": [
                    {
                        "type": "text",
                        "text": summary,
                    }
                ]
            }

        if name == "ax_act":
            app_n = str(args.get("app_name") or "")
            role_n = str(args.get("role") or "")
            title_n = str(args.get("title") or "")
            act_n = str(args.get("action") or "AXPress")
            val_n = args.get("value")
            res = perform_ax_action(
                app_name=app_n,
                role=role_n,
                title=title_n,
                action=act_n,
                value=str(val_n) if val_n is not None else None,
            )
            is_err = not res.get("success", False)
            msg_text = res.get("message") if res.get("success") else res.get("error", "AX action failed")
            return {
                "isError": is_err,
                "content": [{"type": "text", "text": msg_text}],
            }

        return {
            "isError": True,
            "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
        }

    def handle_rpc_message(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        method = msg.get("method", "")
        msg_id = msg.get("id")

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "voicefi-screenpipe-mcp", "version": "1.0.0"},
                },
            }
        if method.startswith("notifications/"):
            return None
        if method == "ping":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": self.get_tools()},
            }
        if method == "tools/call":
            params = msg.get("params") or {}
            tool_name = str(params.get("name") or "")
            tool_args = params.get("arguments") or {}
            res = self.call_tool(tool_name, tool_args)
            return {"jsonrpc": "2.0", "id": msg_id, "result": res}

        if msg_id is not None:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        return None

    def run_stdio(self) -> None:
        """Run the MCP server loop over stdin/stdout."""
        for line in sys.stdin:
            raw = line.strip()
            if not raw:
                continue
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            resp = self.handle_rpc_message(msg)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()


# Canonical alias
FovealVisionMCPServer = ScreenpipeMCPServer
