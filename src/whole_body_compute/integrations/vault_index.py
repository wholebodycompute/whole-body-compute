"""
voicefi/integrations/vault_index.py
Sub-20ms Local-First Spoken RAG Index & Socratic Brainstorming Partner.

Indexes three developer memory streams into Unified RAM / SQLite FTS5 WAL:
1. Obsidian Vault Markdown Notes (heading-scoped chunks with exact file:Lstart-Lend citations)
2. Voice Memos & Meeting Transcripts
3. Recent Git Commits & Working-Tree Diffs

Synthesizes answers 100% locally on Apple Silicon Metal GPU via HybridRouter & Ollama
(or zero-LLM instant citation fallback), with Socratic Partner brainstorming mode
and Studio Neural Persona playback (Christopher, Sonia, Guy, Aria, or offline Ava).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qsl, urlparse, urlunparse

from whole_body_compute.redaction import redact_secrets

logger = logging.getLogger("whole_body_compute.integrations.vault_index")

_EXCLUDED_DIRS = {".obsidian", ".trash", ".git", "node_modules", ".venv", "__pycache__"}

STUDIO_VAULT_PERSONAS: Dict[str, str] = {
    "christopher": "en-US-ChristopherNeural",
    "sonia": "en-GB-SoniaNeural",
    "guy": "en-US-GuyNeural",
    "aria": "en-US-AriaNeural",
    "ava": "Ava (Premium)",
    "viv": "en-IE-EmilyNeural",
}


@dataclass
class VaultChunk:
    chunk_id: str
    source_type: str  # "note", "voice_memo", "git_context"
    note_name: str
    file_path: str
    heading: str
    line_start: int
    line_end: int
    content: str
    citation: str
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VaultRAGResponse:
    query: str
    mode: str  # "qa" or "socratic"
    spoken_response: str
    citations: List[str]
    chunks: List[VaultChunk]
    retrieval_latency_ms: float
    synthesis_latency_ms: float
    total_latency_ms: float
    model_used: str
    is_local: bool
    persona: str = "Christopher"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["chunks"] = [c.to_dict() for c in self.chunks]
        return d


def split_markdown_into_chunks(
    file_path: Path,
    source_type: str = "note",
    max_chunk_lines: int = 35,
) -> List[VaultChunk]:
    """
    Split a markdown file into heading-scoped chunks with exact 1-indexed line ranges
    and automatic secret redaction.
    """
    try:
        raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    if not raw_text.strip():
        return []

    lines = raw_text.splitlines()
    note_name = file_path.stem
    rel_name = file_path.name

    chunks: List[VaultChunk] = []
    current_heading = note_name
    buf: List[str] = []
    chunk_start = 1

    def _flush(end_line: int):
        nonlocal buf
        body = "\n".join(buf).strip()
        if not body:
            buf = []
            return
        clean_body = redact_secrets(body)
        cid = f"{file_path}:{chunk_start}-{end_line}"
        citation = f"{rel_name}#L{chunk_start}-L{end_line}"
        chunks.append(
            VaultChunk(
                chunk_id=cid,
                source_type=source_type,
                note_name=note_name,
                file_path=str(file_path),
                heading=current_heading,
                line_start=chunk_start,
                line_end=end_line,
                content=clean_body[:1500],
                citation=citation,
            )
        )
        buf = []

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            if buf:
                _flush(idx - 1)
            current_heading = re.sub(r"^#+\s*", "", stripped).strip() or note_name
            chunk_start = idx
            buf.append(line)
        else:
            if not buf:
                chunk_start = idx
            buf.append(line)
            if len(buf) >= max_chunk_lines:
                _flush(idx)
                chunk_start = idx + 1

    if buf:
        _flush(len(lines))

    return chunks


class VaultRAGIndex:
    """
    Sub-20ms SQLite FTS5 + Unified RAM knowledge index over Obsidian notes,
    voice memos, and recent git commits/diffs.
    """

    def __init__(self, db_path: Optional[Path] = None):
        default_dir = Path.home() / ".voicefi"
        default_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or (default_dir / "vault_index.db")
        self._local = threading.local()
        self._has_fts5 = True
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=5.0,
                check_same_thread=False,
                isolation_level=None,
            )
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA journal_mode = WAL;")
                conn.execute("PRAGMA synchronous = NORMAL;")
                conn.execute("PRAGMA temp_store = MEMORY;")
                conn.execute("PRAGMA mmap_size = 268435456;")  # 256MB Unified RAM mmap
            except Exception:
                pass
            self._local.conn = conn
        return self._local.conn

    def _init_schema(self) -> None:
        conn = self._get_conn()
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS file_meta (
                    file_path TEXT PRIMARY KEY,
                    mtime REAL NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    source_type TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    note_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    heading TEXT NOT NULL,
                    line_start INTEGER NOT NULL,
                    line_end INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    citation TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vault_links (
                    source_path TEXT,
                    target_note TEXT,
                    snippet TEXT
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_vault_links_source ON vault_links(source_path);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_vault_links_target ON vault_links(target_note);"
            )
            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                        chunk_id UNINDEXED,
                        note_name,
                        heading,
                        content,
                        tokenize = 'porter unicode61'
                    );
                    """
                )
            except sqlite3.OperationalError:
                self._has_fts5 = False

    def _init_db(self) -> None:
        """Alias for _init_schema."""
        self._init_schema()

    def index_vault(
        self,
        vault_path: Path,
        source_type: str = "note",
        force: bool = False,
    ) -> int:
        """
        Incrementally index all markdown files in `vault_path`.
        Only parses files whose mtime or byte size changed since the last sync.
        Returns number of chunks indexed/updated.
        """
        vault = Path(vault_path).expanduser().resolve()
        if not vault.is_dir():
            return 0

        conn = self._get_conn()
        existing_meta: Dict[str, Tuple[float, int]] = {}
        if not force:
            for row in conn.execute("SELECT file_path, mtime, size_bytes FROM file_meta;"):
                existing_meta[row["file_path"]] = (float(row["mtime"]), int(row["size_bytes"]))

        updated_chunks = 0
        for md_file in vault.rglob("*.md"):
            if any(part in _EXCLUDED_DIRS for part in md_file.parts):
                continue
            try:
                st = md_file.stat()
            except OSError:
                continue
            if st.st_size > 2_000_000:
                continue

            fp_str = str(md_file)
            prev = existing_meta.get(fp_str)
            if not force and prev and abs(prev[0] - st.st_mtime) < 1e-3 and prev[1] == st.st_size:
                continue

            file_chunks = split_markdown_into_chunks(md_file, source_type=source_type)
            try:
                raw_text = md_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                raw_text = ""

            with conn:
                conn.execute("DELETE FROM chunks WHERE file_path = ?;", (fp_str,))
                conn.execute("DELETE FROM vault_links WHERE source_path = ?;", (fp_str,))
                for match in re.finditer(r"\[\[([^\]\|]+)(?:\|([^\]]+))?\]\]", raw_text):
                    raw_target = match.group(1).strip()
                    if not raw_target:
                        continue
                    # Strip heading anchor if present
                    target_note = raw_target.split("#")[0].strip()
                    if target_note:
                        start_pos = max(0, match.start() - 60)
                        end_pos = min(len(raw_text), match.end() + 60)
                        snippet = raw_text[start_pos:end_pos].strip().replace("\n", " ")
                        conn.execute(
                            "INSERT INTO vault_links (source_path, target_note, snippet) VALUES (?, ?, ?);",
                            (fp_str, target_note, snippet),
                        )
                if self._has_fts5:
                    conn.execute(
                        "DELETE FROM chunks_fts WHERE chunk_id LIKE ?;", (f"{fp_str}:%",)
                    )
                for ch in file_chunks:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO chunks (
                            chunk_id, source_type, note_name, file_path,
                            heading, line_start, line_end, content, citation
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            ch.chunk_id,
                            ch.source_type,
                            ch.note_name,
                            ch.file_path,
                            ch.heading,
                            ch.line_start,
                            ch.line_end,
                            ch.content,
                            ch.citation,
                        ),
                    )
                    if self._has_fts5:
                        conn.execute(
                            """
                            INSERT INTO chunks_fts (chunk_id, note_name, heading, content)
                            VALUES (?, ?, ?, ?);
                            """,
                            (ch.chunk_id, ch.note_name, ch.heading, ch.content),
                        )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO file_meta (file_path, mtime, size_bytes, source_type)
                    VALUES (?, ?, ?, ?);
                    """,
                    (fp_str, st.st_mtime, st.st_size, source_type),
                )
            updated_chunks += len(file_chunks)

        return updated_chunks

    def index_git_context(self, repo_path: Optional[Path] = None, max_commits: int = 8) -> int:
        """Index recent git commits and working-tree diff summary into the vault index."""
        repo = Path(repo_path or Path.cwd()).resolve()
        if not (repo / ".git").exists():
            return 0

        try:
            log_res = subprocess.run(
                ["git", "log", f"-n{max_commits}", "--oneline", "--no-decorate"],
                cwd=str(repo),
                capture_output=True,
                text=True,
                timeout=1.5,
            )
            commits_text = log_res.stdout.strip()
        except Exception:
            commits_text = ""

        try:
            diff_res = subprocess.run(
                ["git", "diff", "--stat", "HEAD"],
                cwd=str(repo),
                capture_output=True,
                text=True,
                timeout=1.5,
            )
            diff_stat = diff_res.stdout.strip()
        except Exception:
            diff_stat = ""

        if not commits_text and not diff_stat:
            return 0

        combined = redact_secrets(
            f"Recent Git Commits ({repo.name}):\n{commits_text}\n\nWorking Tree Diff Stat:\n{diff_stat}"
        )
        cid = f"git:{repo.name}:recent"
        conn = self._get_conn()
        with conn:
            conn.execute("DELETE FROM chunks WHERE chunk_id = ?;", (cid,))
            if self._has_fts5:
                conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?;", (cid,))
            conn.execute(
                """
                INSERT INTO chunks (
                    chunk_id, source_type, note_name, file_path,
                    heading, line_start, line_end, content, citation
                ) VALUES (?, 'git_context', ?, ?, 'Recent Git Commits & Diff', 1, 20, ?, ?);
                """,
                (
                    cid,
                    f"git:{repo.name}",
                    str(repo),
                    combined[:1500],
                    f"{repo.name}@HEAD",
                ),
            )
            if self._has_fts5:
                conn.execute(
                    """
                    INSERT INTO chunks_fts (chunk_id, note_name, heading, content)
                    VALUES (?, ?, 'Recent Git Commits & Diff', ?);
                    """,
                    (cid, f"git:{repo.name}", combined[:1500]),
                )
        return 1

    @staticmethod
    def _clean_browser_url(raw_url: str) -> str:
        """
        Strip query-string tokens/secrets and enforce redact_secrets() on URL.
        """
        if not raw_url:
            return ""
        try:
            parsed = urlparse(raw_url)
            if parsed.query:
                sensitive_keywords = (
                    "token", "api_key", "apikey", "secret", "auth", "key",
                    "password", "passwd", "pwd", "credential", "bearer",
                    "access_token", "refresh_token", "session", "code",
                    "private_key", "client_secret", "jwt",
                )
                clean_pairs = []
                for k, v in parse_qsl(parsed.query, keep_blank_values=True):
                    k_lower = k.lower()
                    if any(kw in k_lower for kw in sensitive_keywords) or any(
                        pfx in v.lower() for pfx in ("bearer", "sk-", "ghp_", "eyj")
                    ):
                        clean_pairs.append(f"{k}=[redacted]")
                    else:
                        clean_pairs.append(f"{k}={redact_secrets(v)}")
                new_query = "&".join(clean_pairs)
                raw_url = urlunparse((
                    parsed.scheme,
                    parsed.netloc,
                    parsed.path,
                    parsed.params,
                    new_query,
                    parsed.fragment,
                ))
        except Exception:
            pass
        return redact_secrets(raw_url)

    @staticmethod
    @contextmanager
    def _snapshot_sqlite(db_path: Path):
        """
        Safely open a read-only SQLite snapshot of a potentially locked live database.
        Copies the SQLite file (and any -wal/-shm files) to a temporary directory
        to ensure zero lock contention with active browser or note apps.
        Falls back to direct immutable URI if copy is impractical.
        """
        temp_dir = None
        conn = None
        try:
            temp_dir = tempfile.TemporaryDirectory()
            dest_file = Path(temp_dir.name) / db_path.name
            shutil.copy2(db_path, dest_file)

            # Copy WAL / SHM files if present for live consistency
            wal_file = db_path.with_name(db_path.name + "-wal")
            if wal_file.exists():
                try:
                    shutil.copy2(wal_file, Path(temp_dir.name) / wal_file.name)
                except OSError:
                    pass
            shm_file = db_path.with_name(db_path.name + "-shm")
            if shm_file.exists():
                try:
                    shutil.copy2(shm_file, Path(temp_dir.name) / shm_file.name)
                except OSError:
                    pass

            conn = sqlite3.connect(f"file:{dest_file}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            yield conn
        except (PermissionError, sqlite3.OperationalError, sqlite3.DatabaseError, OSError) as e:
            logger.debug(f"Failed to create/read SQLite snapshot for {db_path}: {e}")
            fallback_conn = None
            try:
                if db_path.exists():
                    fallback_conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
                    fallback_conn.row_factory = sqlite3.Row
            except Exception as e2:
                logger.debug(f"Direct immutable fallback also failed for {db_path}: {e2}")

            if fallback_conn is not None:
                try:
                    yield fallback_conn
                finally:
                    try:
                        fallback_conn.close()
                    except Exception:
                        pass
            else:
                yield None
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            if temp_dir is not None:
                try:
                    temp_dir.cleanup()
                except Exception:
                    pass

    def index_browser_history_sqlite(
        self,
        history_db_path: Optional[Path] = None,
        limit: int = 100,
    ) -> int:
        """
        Collect and index recent browser history from Chrome or Arc SQLite databases
        into the VaultRAGIndex (FTS5) using zero-lock read-only snapshotting.

        Default path discovery checks:
        - Chrome: ~/Library/Application Support/Google/Chrome/Default/History
        - Arc: ~/Library/Application Support/Arc/User Data/Default/History

        Strips query-string tokens/secrets and redacts secrets in both title and URL
        prior to insertion.
        """
        if history_db_path is not None:
            target_paths = [Path(history_db_path).expanduser().resolve()]
        else:
            candidates = [
                Path.home() / "Library" / "Application Support" / "Google" / "Chrome" / "Default" / "History",
                Path.home() / "Library" / "Application Support" / "Arc" / "User Data" / "Default" / "History",
            ]
            target_paths = []
            for p in candidates:
                try:
                    if p.exists():
                        target_paths.append(p)
                except (PermissionError, OSError):
                    pass

        if not target_paths:
            return 0

        total_indexed = 0
        conn = self._get_conn()

        for db_file in target_paths:
            try:
                with self._snapshot_sqlite(db_file) as db:
                    if db is None:
                        continue

                    # Query urls table
                    try:
                        cursor = db.execute(
                            """
                            SELECT url, title, last_visit_time
                            FROM urls
                            WHERE title IS NOT NULL AND title != ''
                            ORDER BY last_visit_time DESC
                            LIMIT ?
                            """,
                            (limit,),
                        )
                        rows = cursor.fetchall()
                    except sqlite3.OperationalError:
                        try:
                            cursor = db.execute(
                                """
                                SELECT url, title
                                FROM urls
                                WHERE title IS NOT NULL AND title != ''
                                ORDER BY rowid DESC
                                LIMIT ?
                                """,
                                (limit,),
                            )
                            rows = cursor.fetchall()
                        except Exception as e:
                            logger.debug(f"Failed to query urls from {db_file}: {e}")
                            continue

                    with conn:
                        for row in rows:
                            raw_url = row["url"] if "url" in row.keys() else row[0]
                            raw_title = row["title"] if "title" in row.keys() else row[1]

                            clean_url = self._clean_browser_url(str(raw_url or ""))
                            clean_title = redact_secrets(str(raw_title or "")).strip()

                            if not clean_title or not clean_url:
                                continue

                            try:
                                parsed = urlparse(clean_url)
                                domain = parsed.netloc or "browser"
                            except Exception:
                                domain = "browser"

                            virtual_path = f"browser://history/{domain}"
                            citation = f"browser://history/{domain}#L1-L1"
                            cid = f"browser:{clean_url}"
                            content = f"{clean_title}\n{clean_url}"

                            conn.execute("DELETE FROM chunks WHERE chunk_id = ?;", (cid,))
                            if self._has_fts5:
                                conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?;", (cid,))
                            conn.execute(
                                """
                                INSERT INTO chunks (
                                    chunk_id, source_type, note_name, file_path,
                                    heading, line_start, line_end, content, citation
                                ) VALUES (?, 'browser_history', ?, ?, ?, 1, 1, ?, ?);
                                """,
                                (cid, clean_title, virtual_path, domain, content, citation),
                            )
                            if self._has_fts5:
                                conn.execute(
                                    """
                                    INSERT INTO chunks_fts (chunk_id, note_name, heading, content)
                                    VALUES (?, ?, ?, ?);
                                    """,
                                    (cid, clean_title, domain, content),
                                )
                            total_indexed += 1
            except (PermissionError, sqlite3.OperationalError, sqlite3.DatabaseError, OSError) as e:
                logger.debug(f"Failed to index browser history from {db_file}: {e}")
                continue

        return total_indexed

    def index_apple_notes_sqlite(
        self,
        notes_db_path: Optional[Path] = None,
        limit: int = 100,
    ) -> int:
        """
        Collect and index Apple Notes from NoteStore.sqlite into VaultRAGIndex (FTS5)
        using zero-lock read-only snapshotting.

        Default path discovery:
        ~/Library/Group Containers/group.com.apple.notes/NoteStore.sqlite

        Supports querying ZICCLOUDSYNCINGOBJECT (ZTITLE1, ZSNIPPET, ZIDENTIFIER)
        or fallback simplified notes tables (title, snippet, identifier) gracefully,
        handling macOS TCC permission denial (PermissionError / OperationalError)
        with a clean 0 return and no crash.
        """
        if notes_db_path is not None:
            target_path = Path(notes_db_path).expanduser().resolve()
        else:
            target_path = (
                Path.home()
                / "Library"
                / "Group Containers"
                / "group.com.apple.notes"
                / "NoteStore.sqlite"
            )

        try:
            if not target_path.exists():
                return 0
        except (PermissionError, OSError):
            logger.debug(f"Permission denied checking Apple Notes path {target_path}")
            return 0

        total_indexed = 0
        conn = self._get_conn()

        try:
            with self._snapshot_sqlite(target_path) as db:
                if db is None:
                    return 0

                rows: List[Dict[str, Any]] = []
                # 1. Standard Apple Notes table
                try:
                    cursor = db.execute(
                        """
                        SELECT ZIDENTIFIER, ZTITLE1, ZSNIPPET
                        FROM ZICCLOUDSYNCINGOBJECT
                        WHERE ZTITLE1 IS NOT NULL AND ZTITLE1 != ''
                        ORDER BY rowid DESC
                        LIMIT ?
                        """,
                        (limit,),
                    )
                    for r in cursor.fetchall():
                        rows.append({
                            "identifier": r["ZIDENTIFIER"] if "ZIDENTIFIER" in r.keys() else r[0],
                            "title": r["ZTITLE1"] if "ZTITLE1" in r.keys() else r[1],
                            "snippet": r["ZSNIPPET"] if "ZSNIPPET" in r.keys() else r[2],
                        })
                except sqlite3.OperationalError:
                    # 2. Fallback to simplified notes tables (e.g. title, snippet, identifier)
                    try:
                        table_rows = db.execute(
                            "SELECT name FROM sqlite_master WHERE type='table';"
                        ).fetchall()
                        table_names = [r[0] for r in table_rows]
                        found = False
                        for tname in table_names:
                            col_info = db.execute(f"PRAGMA table_info({tname});").fetchall()
                            col_names = {c[1].lower(): c[1] for c in col_info}
                            if "title" in col_names:
                                id_col = col_names.get("identifier") or col_names.get("id") or "rowid"
                                title_col = col_names["title"]
                                snippet_col = col_names.get("snippet", "''")
                                cursor = db.execute(
                                    f"SELECT {id_col} AS identifier, {title_col} AS title, {snippet_col} AS snippet FROM {tname} WHERE {title_col} IS NOT NULL AND {title_col} != '' LIMIT ?",
                                    (limit,),
                                )
                                for r in cursor.fetchall():
                                    rows.append({
                                        "identifier": str(r["identifier"] or ""),
                                        "title": str(r["title"] or ""),
                                        "snippet": str(r["snippet"] or ""),
                                    })
                                found = True
                                break
                        if not found:
                            return 0
                    except Exception as e_inner:
                        logger.debug(f"Fallback table search failed: {e_inner}")
                        return 0

                with conn:
                    for item in rows:
                        raw_id = item["identifier"]
                        raw_title = item["title"]
                        raw_snippet = item["snippet"]

                        clean_title = redact_secrets(str(raw_title or "")).strip()
                        clean_snippet = redact_secrets(str(raw_snippet or "")).strip()
                        ident = str(raw_id or "").strip() or f"note_{total_indexed}"

                        if not clean_title:
                            continue

                        virtual_path = f"apple-notes://{ident}"
                        citation = f"apple-notes://{ident}#L1-L1"
                        cid = f"apple_notes:{ident}"
                        content = f"{clean_title}\n{clean_snippet}".strip()

                        conn.execute("DELETE FROM chunks WHERE chunk_id = ?;", (cid,))
                        if self._has_fts5:
                            conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?;", (cid,))
                        conn.execute(
                            """
                            INSERT INTO chunks (
                                chunk_id, source_type, note_name, file_path,
                                heading, line_start, line_end, content, citation
                            ) VALUES (?, 'apple_notes', ?, ?, ?, 1, 1, ?, ?);
                            """,
                            (cid, clean_title, virtual_path, clean_title, content, citation),
                        )
                        if self._has_fts5:
                            conn.execute(
                                """
                                INSERT INTO chunks_fts (chunk_id, note_name, heading, content)
                                VALUES (?, ?, ?, ?);
                                """,
                                (cid, clean_title, clean_title, content),
                            )
                        total_indexed += 1
        except (PermissionError, sqlite3.OperationalError, sqlite3.DatabaseError, OSError) as e:
            logger.debug(f"Failed to read Apple Notes SQLite at {target_path}: {e}")
            return 0

        return total_indexed

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[str] = None,
    ) -> Tuple[List[VaultChunk], float]:
        """
        Execute sub-20ms BM25 + keyword search in Unified RAM.
        Returns (ranked_chunks, retrieval_latency_ms).
        """
        from whole_body_compute.integrations.vault_agent import VaultAgent

        t0 = time.perf_counter()
        clean_q = (query or "").replace("\x00", " ").strip()
        terms = VaultAgent.extract_search_terms(clean_q)
        if not terms:
            terms = re.findall(r"\b\w{2,}\b", clean_q.lower())
        if not terms:
            return [], round((time.perf_counter() - t0) * 1000.0, 3)

        conn = self._get_conn()
        results: List[VaultChunk] = []

        if self._has_fts5:
            fts_query = " OR ".join(f'"{t}"*' for t in terms[:8])
            sql = """
                SELECT c.*, bm25(chunks_fts, 3.0, 2.5, 1.0) AS rank_score
                FROM chunks_fts
                JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
                WHERE chunks_fts MATCH ?
            """
            params: List[Any] = [fts_query]
            if source_filter:
                sql += " AND c.source_type = ?"
                params.append(source_filter)
            sql += " ORDER BY rank_score ASC LIMIT ?;"
            params.append(top_k)

            try:
                rows = conn.execute(sql, tuple(params)).fetchall()
                for r in rows:
                    # FTS5 bm25 returns negative numbers where more negative = better match
                    raw_bm25 = float(r["rank_score"] or 0.0)
                    results.append(
                        VaultChunk(
                            chunk_id=r["chunk_id"],
                            source_type=r["source_type"],
                            note_name=r["note_name"],
                            file_path=r["file_path"],
                            heading=r["heading"],
                            line_start=int(r["line_start"]),
                            line_end=int(r["line_end"]),
                            content=r["content"],
                            citation=r["citation"],
                            score=round(abs(raw_bm25) + 1.0, 3),
                        )
                    )
            except Exception as e:
                logger.debug(f"FTS5 query fallback triggered: {e}")

        # Fallback if FTS5 yielded nothing or wasn't available
        if not results:
            rows = conn.execute("SELECT * FROM chunks;").fetchall()
            scored: List[VaultChunk] = []
            for r in rows:
                if source_filter and r["source_type"] != source_filter:
                    continue
                haystack = f"{r['note_name']} {r['heading']} {r['content']}".lower()
                matched = sum(1 for t in terms if t in haystack)
                if matched > 0:
                    scored.append(
                        VaultChunk(
                            chunk_id=r["chunk_id"],
                            source_type=r["source_type"],
                            note_name=r["note_name"],
                            file_path=r["file_path"],
                            heading=r["heading"],
                            line_start=int(r["line_start"]),
                            line_end=int(r["line_end"]),
                            content=r["content"],
                            citation=r["citation"],
                            score=float(matched),
                        )
                    )
            scored.sort(key=lambda c: c.score, reverse=True)
            results = scored[:top_k]

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return results, latency_ms

    def search_vault(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[str] = None,
    ) -> List[VaultChunk]:
        """Convenience method returning ranked chunks from Unified RAM search."""
        chunks, _ = self.search(query=query, top_k=top_k, source_filter=source_filter)
        return chunks

    def get_backlinks(self, note_name: str) -> List[Dict[str, Any]]:
        """Return incoming links pointing to `note_name`."""
        conn = self._get_conn()
        clean = note_name[:-3] if note_name.endswith(".md") else note_name
        stem = Path(clean).stem
        rows = conn.execute(
            """
            SELECT source_path, target_note, snippet
            FROM vault_links
            WHERE LOWER(target_note) = LOWER(?)
               OR LOWER(target_note) = LOWER(? || '.md')
               OR LOWER(target_note) = LOWER(?)
               OR LOWER(target_note) = LOWER(? || '.md')
               OR LOWER(target_note) LIKE LOWER('%/' || ?)
               OR LOWER(target_note) LIKE LOWER('%/' || ? || '.md')
            """,
            (clean, clean, stem, stem, stem, stem),
        ).fetchall()
        return [
            {
                "source_path": r["source_path"],
                "target_note": r["target_note"],
                "snippet": r["snippet"],
            }
            for r in rows
        ]

    def search_graph_hops(
        self,
        query: str,
        max_hops: int = 2,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search vault and expand 1-to-2 hops across outgoing [[WikiLinks]] and incoming backlinks.
        Marks each result with `hop_distance` (0, 1, or 2) and `linked_via`.
        """
        hop0_chunks = self.search_vault(query, top_k=limit)
        results: List[Dict[str, Any]] = []
        visited_notes: Set[str] = set()
        visited_paths: Set[str] = set()

        def _norm(name: str) -> str:
            n = (name or "").strip()
            n = n.split("#")[0].strip()
            if n.lower().endswith(".md"):
                n = n[:-3]
            return Path(n).name.lower()

        current_layer: List[Dict[str, Any]] = []
        base_chunks = hop0_chunks
        if max_hops > 0 and len(base_chunks) >= limit and limit > 1:
            base_chunks = hop0_chunks[: max(1, limit - 2 if limit >= 3 else 1)]

        for ch in base_chunks:
            norm_name = _norm(ch.note_name)
            visited_notes.add(norm_name)
            visited_paths.add(ch.file_path)
            item = {
                "note_name": ch.note_name,
                "file_path": ch.file_path,
                "heading": ch.heading,
                "content": ch.content,
                "snippet": ch.content[:240].strip(),
                "score": ch.score,
                "hop_distance": 0,
                "linked_via": None,
            }
            results.append(item)
            current_layer.append(item)

        conn = self._get_conn()

        for hop in range(1, max_hops + 1):
            next_layer: List[Dict[str, Any]] = []
            for parent in current_layer:
                parent_name = parent["note_name"]
                parent_path = parent["file_path"]

                # 1. Outgoing links from this note
                out_links = conn.execute(
                    "SELECT target_note, snippet FROM vault_links WHERE source_path = ?",
                    (parent_path,),
                ).fetchall()

                for ol in out_links:
                    tgt = ol["target_note"]
                    norm_tgt = _norm(tgt)
                    if norm_tgt in visited_notes:
                        continue
                    visited_notes.add(norm_tgt)

                    clean_tgt = tgt.split("#")[0].strip()
                    stem_tgt = Path(clean_tgt).stem
                    chunk_row = conn.execute(
                        """
                        SELECT chunk_id, source_type, note_name, file_path, heading, content, citation
                        FROM chunks
                        WHERE LOWER(note_name) = ?
                           OR LOWER(note_name) = ? || '.md'
                           OR LOWER(note_name) = LOWER(?)
                           OR LOWER(note_name) = LOWER(?) || '.md'
                           OR LOWER(file_path) LIKE LOWER('%/' || ? || '.md')
                           OR LOWER(file_path) LIKE LOWER('%/' || ? || '.md')
                        LIMIT 1
                        """,
                        (norm_tgt, norm_tgt, stem_tgt, stem_tgt, clean_tgt, stem_tgt),
                    ).fetchone()

                    if chunk_row:
                        visited_paths.add(chunk_row["file_path"])
                        item = {
                            "note_name": chunk_row["note_name"],
                            "file_path": chunk_row["file_path"],
                            "heading": chunk_row["heading"],
                            "content": chunk_row["content"],
                            "snippet": chunk_row["content"][:240].strip(),
                            "score": round(1.0 / (hop + 1), 3),
                            "hop_distance": hop,
                            "linked_via": parent_name,
                        }
                    else:
                        item = {
                            "note_name": tgt,
                            "file_path": "",
                            "heading": "",
                            "content": ol["snippet"],
                            "snippet": ol["snippet"],
                            "score": round(1.0 / (hop + 1), 3),
                            "hop_distance": hop,
                            "linked_via": parent_name,
                        }
                    next_layer.append(item)
                    results.append(item)

                # 2. Incoming backlinks pointing to this note
                backlinks = self.get_backlinks(parent_name)
                for bl in backlinks:
                    src_path = bl["source_path"]
                    src_name = Path(src_path).stem
                    norm_src = _norm(src_name)
                    if norm_src in visited_notes or src_path in visited_paths:
                        continue
                    visited_notes.add(norm_src)
                    visited_paths.add(src_path)

                    chunk_row = conn.execute(
                        """
                        SELECT chunk_id, source_type, note_name, file_path, heading, content, citation
                        FROM chunks
                        WHERE file_path = ?
                        LIMIT 1
                        """,
                        (src_path,),
                    ).fetchone()

                    if chunk_row:
                        item = {
                            "note_name": chunk_row["note_name"],
                            "file_path": chunk_row["file_path"],
                            "heading": chunk_row["heading"],
                            "content": chunk_row["content"],
                            "snippet": chunk_row["content"][:240].strip(),
                            "score": round(1.0 / (hop + 1), 3),
                            "hop_distance": hop,
                            "linked_via": parent_name,
                        }
                    else:
                        item = {
                            "note_name": src_name,
                            "file_path": src_path,
                            "heading": "",
                            "content": bl["snippet"],
                            "snippet": bl["snippet"],
                            "score": round(1.0 / (hop + 1), 3),
                            "hop_distance": hop,
                            "linked_via": parent_name,
                        }
                    next_layer.append(item)
                    results.append(item)

            current_layer = next_layer
            if not current_layer:
                break

        if len(results) < limit:
            for ch in hop0_chunks[len(base_chunks):]:
                norm_name = _norm(ch.note_name)
                if norm_name not in visited_notes:
                    visited_notes.add(norm_name)
                    results.append({
                        "note_name": ch.note_name,
                        "file_path": ch.file_path,
                        "heading": ch.heading,
                        "content": ch.content,
                        "snippet": ch.content[:240].strip(),
                        "score": ch.score,
                        "hop_distance": 0,
                        "linked_via": None,
                    })
                if len(results) >= limit:
                    break

        return results[:limit] if limit and limit > 0 else results


class LocalVaultSynthesizer:
    """
    100% Local-First Q&A and Socratic Brainstorming Partner powered by Apple Silicon
    Ollama + HybridRouter (with zero-LLM instant citation fallback).
    """

    QA_SYSTEM_PROMPT = (
        "You are VoiceFi's local knowledge partner reading the developer's own Obsidian notes and git history aloud.\n"
        "Answer the question in at most 2 concise spoken sentences using ONLY the provided excerpts.\n"
        "Explicitly mention the note name and line numbers (e.g., 'In ROADMAP line 110...').\n"
        "Do not use markdown bullets or symbols."
    )

    SOCRATIC_SYSTEM_PROMPT = (
        "You are VoiceFi's Socratic Brainstorming Partner running locally on Apple Silicon.\n"
        "Cross-examine the developer's question or idea against their Obsidian notes and recent git history.\n"
        "In 2-3 punchy spoken sentences: 1) Cite any relevant past decision or contradiction in their notes/code, "
        "and 2) Ask one sharp, clarifying architectural question to challenge their assumptions.\n"
        "Do not use markdown formatting."
    )

    def __init__(self, index: Optional[VaultRAGIndex] = None):
        self.index = index or VaultRAGIndex()

    def ask(
        self,
        query: str,
        vault_path: Optional[Path] = None,
        mode: str = "qa",  # "qa" or "socratic"
        persona: str = "Christopher",
        include_git: bool = True,
        use_local_llm: bool = True,
        speak_aloud: bool = False,
    ) -> VaultRAGResponse:
        """
        Execute end-to-end Spoken RAG or Socratic Brainstorming against the vault.
        """
        t_total = time.perf_counter()

        if vault_path and Path(vault_path).is_dir():
            self.index.index_vault(Path(vault_path))
        if include_git:
            self.index.index_git_context()

        chunks, retrieval_ms = self.index.search(query, top_k=4)
        citations = [c.citation for c in chunks]

        if not chunks:
            total_ms = round((time.perf_counter() - t_total) * 1000.0, 3)
            spoken = f"I searched your vault in {retrieval_ms:.1f} milliseconds, but found no notes matching '{query}'."
            return VaultRAGResponse(
                query=query,
                mode=mode,
                spoken_response=spoken,
                citations=[],
                chunks=[],
                retrieval_latency_ms=retrieval_ms,
                synthesis_latency_ms=0.0,
                total_latency_ms=total_ms,
                model_used="none",
                is_local=True,
                persona=persona,
            )

        t_synth = time.perf_counter()
        spoken_answer: Optional[str] = None
        model_used = "retrieval-only"
        is_local = True

        if use_local_llm and os.environ.get("VOICEFI_TESTING") != "1":
            spoken_answer, model_used, is_local = self._query_hybrid_model(query, chunks, mode=mode)

        if not spoken_answer:
            top = chunks[0]
            clean_snip = re.sub(r"[*_#`>\[\]]", "", top.content)
            clean_snip = re.sub(r"\s+", " ", clean_snip).strip()[:220]
            if mode == "socratic":
                spoken_answer = (
                    f"In {top.note_name} lines {top.line_start} to {top.line_end}, you noted: {clean_snip}. "
                    f"How does that align with what you're proposing for {query}?"
                )
            else:
                spoken_answer = (
                    f"From {top.note_name} lines {top.line_start} to {top.line_end}: {clean_snip}"
                )

        synth_ms = round((time.perf_counter() - t_synth) * 1000.0, 3)
        total_ms = round((time.perf_counter() - t_total) * 1000.0, 3)

        # Record in Hybrid Telemetry Ledger
        try:
            from whole_body_compute.local.telemetry import record_hybrid_step

            in_tok = sum(max(1, len(c.content) // 4) for c in chunks)
            out_tok = max(1, len(spoken_answer) // 4)
            record_hybrid_step(
                stage="vault_socratic" if mode == "socratic" else "vault_rag",
                model_name=model_used,
                is_local=is_local,
                duration_ms=int(max(1.0, total_ms)),
                input_tokens=in_tok,
                output_tokens=out_tok,
                ingress_latency_ms=retrieval_ms,
                success=True,
                target=chunks[0].citation if chunks else "vault",
                notes=f"mode={mode} citations={len(citations)}",
            )
        except Exception:
            pass

        if speak_aloud and os.environ.get("VOICEFI_MOCK_AUDIO") != "1":
            self._speak_persona(spoken_answer, persona=persona)

        return VaultRAGResponse(
            query=query,
            mode=mode,
            spoken_response=spoken_answer,
            citations=citations,
            chunks=chunks,
            retrieval_latency_ms=retrieval_ms,
            synthesis_latency_ms=synth_ms,
            total_latency_ms=total_ms,
            model_used=model_used,
            is_local=is_local,
            persona=persona,
        )

    def _query_hybrid_model(
        self,
        query: str,
        chunks: List[VaultChunk],
        mode: str = "qa",
    ) -> Tuple[Optional[str], str, bool]:
        """Route synthesis through HybridRouter to local Ollama (or Cloud fallback)."""
        from whole_body_compute.local.router import default_router

        task_role = "coder" if mode == "socratic" else "scout"
        decision = default_router.resolve_model(task_type=task_role)

        context_block = "\n\n".join(
            f"[{c.citation} | Heading: {c.heading}]\n{c.content[:500]}" for c in chunks
        )
        sys_prompt = self.SOCRATIC_SYSTEM_PROMPT if mode == "socratic" else self.QA_SYSTEM_PROMPT
        user_prompt = f"Query: {query}\n\nVault & Git Excerpts:\n{context_block}"

        if decision.is_local:
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
            # For fast interactive response (<2.5s), prefer fast local tier if 26b isn't warm
            candidate_models = [decision.selected_model, "qwen2.5-coder:1.5b", "gemma2:2b"]
            seen = set()
            for mdl in candidate_models:
                if not mdl or mdl in seen or mdl.startswith("cloud:"):
                    continue
                seen.add(mdl)
                payload = {
                    "model": mdl,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.3 if mode == "socratic" else 0.1, "num_predict": 160},
                }
                try:
                    req = urllib.request.Request(
                        f"{base_url}/api/chat",
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=4.5) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        text = data.get("message", {}).get("content", "").strip()
                        if text:
                            clean = re.sub(r"[*_#`>]", "", text).strip()
                            return clean, f"ollama/{mdl}", True
                except Exception:
                    continue

        return None, "retrieval-only", True

    @staticmethod
    def _speak_persona(text: str, persona: str = "Christopher") -> None:
        """Synthesize spoken answer using the requested Studio Neural Persona."""
        try:
            from voicefi.tts import get_tts_engine
            from whole_body_compute.config import load_config

            cfg = load_config()
            voice_id = STUDIO_VAULT_PERSONAS.get(persona.lower().strip(), persona)
            cfg.tts.voice = voice_id
            tts = get_tts_engine(cfg)
            tts.speak(text)
        except Exception as e:
            logger.debug(f"Studio persona playback skipped: {e}")
