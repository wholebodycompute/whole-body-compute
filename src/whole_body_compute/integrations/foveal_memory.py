"""
Sovereign Foveal Memory Engine (Organ 4 / Organ 2 Vision) & Trojan Horse Importer.

Zero-egress Apple Neural Engine OCR & episodic memory recall. Replaces external
Screenpipe dependencies with pure on-device SQLite storage and local Apple Vision
capture, while providing a 1-way "Trojan Horse" importer to migrate legacy Screenpipe
databases seamlessly into sovereign Whole Body Compute storage.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

DEFAULT_FOVEAL_MEMORY_DB = Path.home() / ".whole_body" / "foveal_memory.sqlite"
DEFAULT_SCREENPIPE_DB = Path.home() / ".screenpipe" / "db.sqlite"
DEFAULT_SCREENPIPE_URL = "http://127.0.0.1:3030"

# Re-export core primitives and helpers from screenpipe
from whole_body_compute.integrations.screenpipe import (
    ScreenpipeRecord,
    ScreenpipeMCPServer,
    ensure_screenpipe_db,
    record_ocr_snapshot,
    record_audio_transcription,
    record_companion_event,
    search_screenpipe_sqlite,
    check_screenpipe_health,
    search_screen_content,
    extract_visible_errors,
    serve_local_screenpipe_http,
    perform_ax_action,
    strip_ui_chrome_from_ocr,
    extract_semantic_screen_entities,
    resolve_direct_headless_route,
    get_frontmost_window_metadata,
    capture_apple_vision_ocr,
    capture_headless_window_text,
)

# Canonical aliases
FovealMemoryRecord = ScreenpipeRecord


def ensure_foveal_memory_db(db_path: Optional[Path] = None) -> Path:
    """Ensure `~/.whole_body/foveal_memory.sqlite` exists with full schema and indexes."""
    target = Path(db_path or os.environ.get("FOVEAL_MEMORY_DB_PATH") or DEFAULT_FOVEAL_MEMORY_DB).expanduser()
    return ensure_screenpipe_db(target)


def search_foveal_memory_sqlite(
    query: str = "",
    content_type: str = "all",
    app_name: Optional[str] = None,
    minutes_ago: Optional[int] = None,
    limit: int = 10,
    db_path: Optional[Path] = None,
    start_time: Optional[str] = None,
) -> List[FovealMemoryRecord]:
    """Search sovereign Foveal Memory SQLite database directly."""
    target = Path(db_path or os.environ.get("FOVEAL_MEMORY_DB_PATH") or DEFAULT_FOVEAL_MEMORY_DB).expanduser()
    return search_screenpipe_sqlite(
        query=query,
        content_type=content_type,
        app_name=app_name,
        minutes_ago=minutes_ago,
        limit=limit,
        db_path=target,
        start_time=start_time,
    )


def record_foveal_ocr_snapshot(
    text: str,
    app_name: str = "Terminal",
    window_name: str = "zsh",
    timestamp: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Record an OCR snapshot directly into sovereign Foveal Memory."""
    target = Path(db_path or os.environ.get("FOVEAL_MEMORY_DB_PATH") or DEFAULT_FOVEAL_MEMORY_DB).expanduser()
    return record_ocr_snapshot(
        text=text,
        app_name=app_name,
        window_name=window_name,
        timestamp=timestamp,
        db_path=target,
    )


def record_foveal_audio_transcription(
    transcription: str,
    device: str = "Microphone",
    timestamp: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Record an audio transcription directly into sovereign Foveal Memory."""
    target = Path(db_path or os.environ.get("FOVEAL_MEMORY_DB_PATH") or DEFAULT_FOVEAL_MEMORY_DB).expanduser()
    return record_audio_transcription(
        transcription=transcription,
        device=device,
        timestamp=timestamp,
        db_path=target,
    )


class FovealVisionMCPServer(ScreenpipeMCPServer):
    """
    Sovereign Foveal Vision & Episodic Memory MCP Server.
    Provides zero-egress screen memory search to Goose, VoiceFi, and subagents.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        base_url: str = DEFAULT_SCREENPIPE_URL,
    ) -> None:
        target = Path(db_path or os.environ.get("FOVEAL_MEMORY_DB_PATH") or DEFAULT_FOVEAL_MEMORY_DB).expanduser()
        super().__init__(db_path=target, base_url=base_url)

    def handle_rpc_message(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        resp = super().handle_rpc_message(msg)
        if resp and isinstance(resp.get("result"), dict) and "serverInfo" in resp["result"]:
            resp["result"]["serverInfo"] = {
                "name": "voicefi-foveal-vision-mcp",
                "version": "2.0.0",
            }
        return resp


def import_legacy_screenpipe_db(
    source_db_path: Optional[Path] = None,
    target_db_path: Optional[Path] = None,
    redact_pii: bool = True,
) -> Dict[str, Any]:
    """
    1-Way "Trojan Horse" importer from legacy Screenpipe SQLite to sovereign Foveal Memory.

    Supports:
    - Upstream Screenpipe schema (`ocr_text` + `frames`, `audio_transcriptions`)
    - Local Whole Body Compute schema (`ocr_snapshots` or `ocr_text`, `audio_transcriptions`)
    - PII / secret scrubbing via `whole_body_compute.redaction.redact_secrets`
    - Idempotent deduplication against existing target entries
    """
    from whole_body_compute.redaction import redact_secrets

    source = Path(source_db_path or os.environ.get("SCREENPIPE_DB_PATH") or DEFAULT_SCREENPIPE_DB).expanduser()
    target = Path(target_db_path or os.environ.get("FOVEAL_MEMORY_DB_PATH") or DEFAULT_FOVEAL_MEMORY_DB).expanduser()

    if not source.exists():
        return {
            "status": "not_found",
            "source_db": str(source),
            "target_db": str(target),
            "ocr_imported": 0,
            "audio_imported": 0,
            "total_imported": 0,
            "redacted_count": 0,
            "message": f"Source Screenpipe SQLite database not found at {source}",
        }

    ensure_foveal_memory_db(target)

    # 1. Inspect existing target records to prevent duplicate re-import
    existing_ocr_sigs: Set[Tuple[str, str, str]] = set()
    existing_audio_sigs: Set[Tuple[str, str]] = set()

    conn_target = sqlite3.connect(str(target), timeout=10.0)
    conn_target.row_factory = sqlite3.Row
    try:
        cur_t = conn_target.cursor()
        cur_t.execute(
            """
            SELECT f.timestamp, COALESCE(NULLIF(o.app_name, ''), f.app_name, '') as app, o.text
            FROM ocr_text o
            LEFT JOIN frames f ON o.frame_id = f.id
            """
        )
        for r in cur_t.fetchall():
            existing_ocr_sigs.add((str(r["timestamp"] or "").strip(), str(r["app"] or "").strip(), str(r["text"] or "").strip()))

        cur_t.execute("SELECT timestamp, transcription FROM audio_transcriptions")
        for r in cur_t.fetchall():
            existing_audio_sigs.add((str(r["timestamp"] or "").strip(), str(r["transcription"] or "").strip()))
    finally:
        conn_target.close()

    ocr_imported = 0
    audio_imported = 0
    redacted_count = 0

    conn_source = sqlite3.connect(str(source), timeout=10.0)
    conn_source.row_factory = sqlite3.Row
    conn_target = sqlite3.connect(str(target), timeout=10.0)
    try:
        cur_s = conn_source.cursor()
        cur_t = conn_target.cursor()

        # Discover tables in source
        cur_s.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur_s.fetchall()}

        # 2. Migrate OCR Text / Frames
        if "ocr_text" in tables:
            has_frames = "frames" in tables
            cur_s.execute("PRAGMA table_info(ocr_text)")
            ocr_cols = {row[1] for row in cur_s.fetchall()}
            
            frame_cols: Set[str] = set()
            if has_frames:
                cur_s.execute("PRAGMA table_info(frames)")
                frame_cols = {row[1] for row in cur_s.fetchall()}

            ts_expr = "f.timestamp" if (has_frames and "timestamp" in frame_cols) else ("o.timestamp" if "timestamp" in ocr_cols else "CURRENT_TIMESTAMP")
            app_expr = "COALESCE(NULLIF(o.app_name, ''), f.app_name, '')" if (has_frames and "app_name" in frame_cols) else "COALESCE(o.app_name, '')"
            win_expr = "COALESCE(NULLIF(o.window_name, ''), f.window_name, '')" if (has_frames and "window_name" in frame_cols) else "COALESCE(o.window_name, '')"
            json_expr = "o.text_json" if "text_json" in ocr_cols else "'{}'"
            engine_expr = "o.ocr_engine" if "ocr_engine" in ocr_cols else "'apple-vision'"

            query = f"""
                SELECT
                    {ts_expr} AS timestamp,
                    {app_expr} AS app_name,
                    {win_expr} AS window_name,
                    o.text AS text,
                    {json_expr} AS text_json,
                    {engine_expr} AS ocr_engine
                FROM ocr_text o
                {"LEFT JOIN frames f ON o.frame_id = f.id" if has_frames else ""}
            """
            cur_s.execute(query)
            for row in cur_s.fetchall():
                raw_text = str(row["text"] or "").strip()
                if not raw_text:
                    continue
                app_name = str(row["app_name"] or "").strip()
                window_name = str(row["window_name"] or "").strip()
                timestamp = str(row["timestamp"] or "").strip() or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                text_json = str(row["text_json"] or "{}")

                if redact_pii:
                    sanitized = redact_secrets(raw_text)
                    if sanitized != raw_text:
                        redacted_count += 1
                else:
                    sanitized = raw_text

                sig = (timestamp, app_name, sanitized)
                if sig in existing_ocr_sigs:
                    continue

                # Insert into frames
                cur_t.execute(
                    "INSERT INTO frames (timestamp, app_name, window_name, focused) VALUES (?, ?, ?, 1)",
                    (timestamp, app_name, window_name),
                )
                frame_id = int(cur_t.lastrowid or 0)

                # Insert into ocr_text
                cur_t.execute(
                    """
                    INSERT INTO ocr_text (frame_id, text, text_json, app_name, ocr_engine, window_name, focused, text_length)
                    VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                    """,
                    (frame_id, sanitized, text_json, app_name, str(row["ocr_engine"] or "apple-vision"), window_name, len(sanitized)),
                )
                existing_ocr_sigs.add(sig)
                ocr_imported += 1

        elif "ocr_snapshots" in tables:
            cur_s.execute("PRAGMA table_info(ocr_snapshots)")
            cols = {row[1] for row in cur_s.fetchall()}
            text_col = "text" if "text" in cols else ("ocr_text" if "ocr_text" in cols else "content")
            app_col = "app_name" if "app_name" in cols else ("app" if "app" in cols else "''")
            win_col = "window_name" if "window_name" in cols else ("window" if "window" in cols else "''")
            ts_col = "timestamp" if "timestamp" in cols else "CURRENT_TIMESTAMP"

            cur_s.execute(f"SELECT {ts_col} AS timestamp, {app_col} AS app_name, {win_col} AS window_name, {text_col} AS text FROM ocr_snapshots")
            for row in cur_s.fetchall():
                raw_text = str(row["text"] or "").strip()
                if not raw_text:
                    continue
                app_name = str(row["app_name"] or "").strip()
                window_name = str(row["window_name"] or "").strip()
                timestamp = str(row["timestamp"] or "").strip() or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

                if redact_pii:
                    sanitized = redact_secrets(raw_text)
                    if sanitized != raw_text:
                        redacted_count += 1
                else:
                    sanitized = raw_text

                sig = (timestamp, app_name, sanitized)
                if sig in existing_ocr_sigs:
                    continue

                cur_t.execute(
                    "INSERT INTO frames (timestamp, app_name, window_name, focused) VALUES (?, ?, ?, 1)",
                    (timestamp, app_name, window_name),
                )
                frame_id = int(cur_t.lastrowid or 0)
                cur_t.execute(
                    """
                    INSERT INTO ocr_text (frame_id, text, text_json, app_name, ocr_engine, window_name, focused, text_length)
                    VALUES (?, ?, '{}', ?, 'apple-vision', ?, 1, ?)
                    """,
                    (frame_id, sanitized, app_name, window_name, len(sanitized)),
                )
                existing_ocr_sigs.add(sig)
                ocr_imported += 1

        # 3. Migrate Audio Transcriptions
        audio_table = None
        for cand in ("audio_transcriptions", "audio_chunks", "audio_transcripts"):
            if cand in tables:
                audio_table = cand
                break

        if audio_table:
            cur_s.execute(f"PRAGMA table_info({audio_table})")
            a_cols = {row[1] for row in cur_s.fetchall()}
            t_col = "transcription" if "transcription" in a_cols else ("text" if "text" in a_cols else "content")
            dev_col = "device" if "device" in a_cols else "''"
            ts_col = "timestamp" if "timestamp" in a_cols else "CURRENT_TIMESTAMP"

            cur_s.execute(f"SELECT {ts_col} AS timestamp, {dev_col} AS device, {t_col} AS transcription FROM {audio_table}")
            for row in cur_s.fetchall():
                raw_audio = str(row["transcription"] or "").strip()
                if not raw_audio:
                    continue
                device = str(row["device"] or "default").strip()
                timestamp = str(row["timestamp"] or "").strip() or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

                if redact_pii:
                    sanitized_audio = redact_secrets(raw_audio)
                    if sanitized_audio != raw_audio:
                        redacted_count += 1
                else:
                    sanitized_audio = raw_audio

                sig_a = (timestamp, sanitized_audio)
                if sig_a in existing_audio_sigs:
                    continue

                cur_t.execute(
                    """
                    INSERT INTO audio_transcriptions (timestamp, transcription, device, is_input_device, transcription_engine)
                    VALUES (?, ?, ?, 1, 'whisper')
                    """,
                    (timestamp, sanitized_audio, device),
                )
                existing_audio_sigs.add(sig_a)
                audio_imported += 1

        conn_target.commit()
    finally:
        conn_source.close()
        conn_target.close()

    return {
        "status": "migrated",
        "source_db": str(source),
        "target_db": str(target),
        "ocr_imported": ocr_imported,
        "audio_imported": audio_imported,
        "total_imported": ocr_imported + audio_imported,
        "redacted_count": redacted_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="VoiceFi Sovereign Foveal Memory MCP & Engine")
    parser.add_argument("--mcp-stdio", action="store_true", help="Run as JSON-RPC 2.0 stdio MCP server")
    parser.add_argument("--import-screenpipe", action="store_true", help="Run 1-way migration from legacy Screenpipe DB")
    parser.add_argument("--source", default=None, help="Source Screenpipe DB path")
    parser.add_argument("--target", default=None, help="Target Foveal Memory DB path")
    parser.add_argument("--no-redact", action="store_true", help="Skip PII redaction during import")
    parser.add_argument("--serve-http", action="store_true", help="Run localhost:3030 compatible HTTP server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=3030, help="Bind port")
    parser.add_argument("--db-path", default=os.environ.get("FOVEAL_MEMORY_DB_PATH", str(DEFAULT_FOVEAL_MEMORY_DB)))
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args()

    db_path = Path(args.db_path).expanduser()

    if args.import_screenpipe:
        res = import_legacy_screenpipe_db(
            source_db_path=Path(args.source) if args.source else None,
            target_db_path=Path(args.target) if args.target else db_path,
            redact_pii=not args.no_redact,
        )
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(f"Migrated {res.get('total_imported', 0)} records ({res.get('ocr_imported', 0)} OCR, {res.get('audio_imported', 0)} Audio) to {res.get('target_db')}")
            if res.get("redacted_count", 0) > 0:
                print(f"Sanitized {res['redacted_count']} sensitive/PII entries during import.")
        return 0 if res.get("status") in ("migrated", "skipped") else 1

    if args.serve_http:
        server = serve_local_screenpipe_http(host=args.host, port=args.port, db_path=db_path)
        print(f"[FovealMemoryLocal] Listening on http://{args.host}:{args.port} (db: {db_path})", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.server_close()
        return 0

    # Default to MCP stdio
    mcp = FovealVisionMCPServer(db_path=db_path)
    mcp.run_stdio()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
