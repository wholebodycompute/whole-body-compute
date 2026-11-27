"""
Ultra-compact JSON-RPC 2.0 stdio MCP server for Claude Desktop ("Quota Bouncer").

Exposes 4 lean tools (<450 schema tokens total) so Claude Desktop can offload
local file/folder/CSV/transcript pre-digests, maintain per-client FTS5 memory,
compile <400-token StateCapsule handoffs, and read inline FinOps receipts.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

from whole_body_compute import __version__
from whole_body_compute.redaction import REDACTION_PLACEHOLDER, redact_secrets
from whole_body_compute.shield import (
    get_proxy_client_summary,
    get_savings_summary,
    record_shield_event,
)

DEFAULT_CLIENT_MEMORY_DB = Path(
    os.environ.get("WBC_CLIENT_MEMORY_DB") or "~/.whole_body_compute/client_memory.sqlite"
).expanduser()

_SKIP_DIRS = frozenset(
    {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
    }
)

_MAX_READ_BYTES = 512 * 1024
_MAX_DIGEST_CHARS = 2400
_MAX_CAPSULE_CHARS = 1520  # < 400 tokens at ~4 chars/token
_QUERY_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}")

# Strictly < 450 estimated tokens (`len(json.dumps(TOOLS)) // 4 < 450`).
TOOLS: list[dict[str, Any]] = [
    {
        "name": "wbc_scout_digest",
        "description": "Local file/dir/CSV digest with secret redaction.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File or dir path"},
                "question": {"type": "string", "description": "Focus query"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "wbc_client_memory",
        "description": "Per-client voice & fact FTS5 store.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["remember", "recall", "list"]},
                "client": {"type": "string", "description": "Client slug"},
                "text_or_query": {"type": "string", "description": "Fact or query"},
            },
            "required": ["action", "client"],
        },
    },
    {
        "name": "wbc_handoff_capsule",
        "description": "Compile <400-tok StateCapsule handoff; pbcopy if GUI.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "Active goal"},
                "key_decisions": {"type": "string", "description": "Decisions"},
                "next_step": {"type": "string", "description": "Next action"},
            },
            "required": ["goal"],
        },
    },
    {
        "name": "wbc_finops_receipt",
        "description": "Inline WBC savings & proxy FinOps receipt.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "window": {"type": "string", "description": "24h|7d|30d|all"},
                "client": {"type": "string", "description": "Client slug"},
            },
        },
    },
]


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3].rstrip() + "..."


def _query_terms(question: str) -> list[str]:
    return [t.lower() for t in _QUERY_TOKEN_RE.findall(question or "")][:8]


def _format_ast_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = [a.arg for a in node.args.posonlyargs + node.args.args]
    if node.args.vararg:
        args.append(f"*{node.args.vararg.arg}")
    if node.args.kwarg:
        args.append(f"**{node.args.kwarg.arg}")
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({', '.join(args[:8])})"


def _digest_python(path: Path, raw_text: str, terms: list[str]) -> str:
    try:
        tree = ast.parse(raw_text)
    except SyntaxError:
        return _digest_text(path, raw_text, terms)

    lines = raw_text.splitlines()
    out: list[str] = [f"Python AST outline ({len(lines)} lines):"]
    doc = ast.get_docstring(tree)
    if doc:
        out.append(f"Doc: {_truncate(doc.splitlines()[0].strip(), 140)}")

    imports: list[str] = []
    symbols: list[str] = []
    matched_snippets: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            imports.append(mod)
        elif isinstance(node, ast.ClassDef):
            methods = [
                m.name
                for m in node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            m_str = f" [{', '.join(methods[:10])}]" if methods else ""
            symbols.append(f"L{node.lineno}: class {node.name}{m_str}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(f"L{node.lineno}: {_format_ast_signature(node)}")

    if imports:
        uniq_imports = list(dict.fromkeys(imports))
        out.append(f"Imports ({len(uniq_imports)}): {', '.join(uniq_imports[:12])}")
    if symbols:
        out.append("Symbols:")
        out.extend(f"- {s}" for s in symbols[:30])
    if terms:
        for idx, line in enumerate(lines, start=1):
            low = line.lower()
            if any(t in low for t in terms):
                matched_snippets.append(f"L{idx}: {line.strip()[:140]}")
                if len(matched_snippets) >= 10:
                    break
        if matched_snippets:
            out.append("Query matches:")
            out.extend(f"  {m}" for m in matched_snippets)
    return "\n".join(out)


def _digest_csv(path: Path, raw_text: str, terms: list[str]) -> str:
    rows: list[list[str]] = []
    delim = "\t" if path.suffix.lower() == ".tsv" else ","
    reader = csv.reader(raw_text.splitlines(), delimiter=delim)
    for row in reader:
        rows.append(row)
        if len(rows) >= 500:
            break
    if not rows:
        return "Empty CSV/TSV file."
    header = rows[0]
    data_rows = rows[1:]
    out: list[str] = [
        f"CSV/TSV summary ({len(data_rows)}+ data rows, {len(header)} cols):",
        f"Columns: {', '.join(c.strip() for c in header[:20])}",
        "Sample rows:",
    ]
    for idx, row in enumerate(data_rows[:6], start=1):
        out.append(f"- #{idx}: {' | '.join(c[:40] for c in row[:10])}")
    if terms and data_rows:
        hits: list[str] = []
        for idx, row in enumerate(data_rows, start=1):
            joined = " ".join(row).lower()
            if any(t in joined for t in terms):
                hits.append(f"- #{idx}: {' | '.join(c[:40] for c in row[:10])}")
                if len(hits) >= 6:
                    break
        if hits:
            out.append("Query matches:")
            out.extend(hits)
    return "\n".join(out)


def _digest_text(path: Path, raw_text: str, terms: list[str]) -> str:
    lines = raw_text.splitlines()
    out: list[str] = [f"Text preview ({len(lines)} lines, {len(raw_text)} chars):"]
    headings = [ln.strip() for ln in lines if ln.lstrip().startswith("#")][:10]
    if headings:
        out.append("Headings:")
        out.extend(f"- {h[:120]}" for h in headings)
    if terms:
        hits: list[str] = []
        for idx, line in enumerate(lines, start=1):
            if any(t in line.lower() for t in terms):
                hits.append(f"L{idx}: {line.strip()[:160]}")
                if len(hits) >= 12:
                    break
        if hits:
            out.append("Query matches:")
            out.extend(f"  {h}" for h in hits)
    head_lines = [ln[:160] for ln in lines[:12] if ln.strip()]
    if head_lines:
        out.append("Head:")
        out.extend(head_lines)
    if len(lines) > 16:
        tail_lines = [ln[:160] for ln in lines[-5:] if ln.strip()]
        if tail_lines:
            out.append("Tail:")
            out.extend(tail_lines)
    return "\n".join(out)


def _digest_directory(root: Path, terms: list[str]) -> tuple[str, int]:
    entries: list[str] = []
    ext_counts: dict[str, int] = {}
    matches: list[str] = []
    raw_bytes = 0
    file_count = 0
    dir_count = 0

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        rel_dir = Path(dirpath).relative_to(root)
        if str(rel_dir) != ".":
            dir_count += 1
        for fname in sorted(filenames):
            file_count += 1
            fpath = Path(dirpath) / fname
            try:
                raw_bytes += fpath.stat().st_size
            except OSError:
                pass
            ext = fpath.suffix.lower() or "(noext)"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
            rel_file = str(fpath.relative_to(root))
            if len(entries) < 40:
                entries.append(rel_file)
            if terms and any(t in rel_file.lower() for t in terms) and len(matches) < 12:
                matches.append(rel_file)
            if file_count >= 250:
                break
        if file_count >= 250:
            break

    top_exts = ", ".join(
        f"{k}:{v}" for k, v in sorted(ext_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
    )
    out = [
        f"Directory digest ({file_count} files, {dir_count} dirs, ~{raw_bytes:,} bytes):",
        f"Extensions: {top_exts or 'none'}",
    ]
    if matches:
        out.append("Query-matching paths:")
        out.extend(f"- {m}" for m in matches)
    if entries:
        out.append("Files:")
        out.extend(f"- {e}" for e in entries[:30])
    return "\n".join(out), max(raw_bytes, len("\n".join(out)))


class _ClientMemoryStore:
    """SQLite FTS5 per-client memory store with automatic secret redaction."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._fts_enabled: bool | None = None

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
        except sqlite3.Error:
            # In-memory or read-only WAL pragma failures are non-fatal
            return conn
        return conn

    def ensure_schema(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS client_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_client_facts_client ON client_facts(client, id DESC);"
            )
            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS client_facts_fts
                    USING fts5(client, content, content='client_facts', content_rowid='id');
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS client_facts_ai AFTER INSERT ON client_facts BEGIN
                        INSERT INTO client_facts_fts(rowid, client, content)
                        VALUES (new.id, new.client, new.content);
                    END;
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS client_facts_ad AFTER DELETE ON client_facts BEGIN
                        INSERT INTO client_facts_fts(client_facts_fts, rowid, client, content)
                        VALUES ('delete', old.id, old.client, old.content);
                    END;
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS client_facts_au AFTER UPDATE ON client_facts BEGIN
                        INSERT INTO client_facts_fts(client_facts_fts, rowid, client, content)
                        VALUES ('delete', old.id, old.client, old.content);
                        INSERT INTO client_facts_fts(rowid, client, content)
                        VALUES (new.id, new.client, new.content);
                    END;
                    """
                )
                self._fts_enabled = True
            except sqlite3.OperationalError:
                self._fts_enabled = False
            conn.commit()
        finally:
            conn.close()

    def remember(self, client: str, text: str) -> dict[str, Any]:
        self.ensure_schema()
        clean_client = client.strip().lower()
        clean_text = redact_secrets(text.strip())
        now = time.time()
        conn = self._connect()
        try:
            cur = conn.execute(
                "INSERT INTO client_facts(client, content, created_at) VALUES (?, ?, ?);",
                (clean_client, clean_text, now),
            )
            conn.commit()
            return {"id": int(cur.lastrowid or 0), "client": clean_client, "content": clean_text}
        finally:
            conn.close()

    def recall(self, client: str, query: str, limit: int = 10) -> list[dict[str, Any]]:
        self.ensure_schema()
        clean_client = client.strip().lower()
        tokens = _QUERY_TOKEN_RE.findall(query or "")
        if not tokens:
            return self.list_facts(clean_client, limit=limit)

        conn = self._connect()
        try:
            if self._fts_enabled:
                fts_expr = " OR ".join(f'"{t}"' for t in tokens[:10])
                try:
                    rows = conn.execute(
                        """
                        SELECT f.id, f.client, f.content, f.created_at
                        FROM client_facts_fts
                        JOIN client_facts f ON f.id = client_facts_fts.rowid
                        WHERE client_facts_fts MATCH ? AND f.client = ?
                        ORDER BY bm25(client_facts_fts), f.id DESC
                        LIMIT ?;
                        """,
                        (fts_expr, clean_client, limit),
                    ).fetchall()
                    return [
                        {
                            "id": int(r["id"]),
                            "client": str(r["client"]),
                            "content": redact_secrets(str(r["content"])),
                        }
                        for r in rows
                    ]
                except sqlite3.OperationalError:
                    pass

            like_clauses = " OR ".join("LOWER(content) LIKE ?" for _ in tokens[:6])
            params: list[Any] = [clean_client] + [f"%{t.lower()}%" for t in tokens[:6]] + [limit]
            rows = conn.execute(
                f"""
                SELECT id, client, content, created_at
                FROM client_facts
                WHERE client = ? AND ({like_clauses})
                ORDER BY id DESC
                LIMIT ?;
                """,
                tuple(params),
            ).fetchall()
            return [
                {
                    "id": int(r["id"]),
                    "client": str(r["client"]),
                    "content": redact_secrets(str(r["content"])),
                }
                for r in rows
            ]
        finally:
            conn.close()

    def list_facts(self, client: str, limit: int = 20) -> list[dict[str, Any]]:
        self.ensure_schema()
        clean_client = client.strip().lower()
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT id, client, content, created_at
                FROM client_facts
                WHERE client = ?
                ORDER BY id DESC
                LIMIT ?;
                """,
                (clean_client, limit),
            ).fetchall()
            return [
                {
                    "id": int(r["id"]),
                    "client": str(r["client"]),
                    "content": redact_secrets(str(r["content"])),
                }
                for r in rows
            ]
        finally:
            conn.close()


class ClaudeDesktopBouncerServer:
    """Ultra-compact JSON-RPC 2.0 stdio MCP server for Claude Desktop quota savings."""

    def __init__(
        self,
        *,
        db_path: Path | str | None = None,
        memory_db_path: Path | str | None = None,
    ) -> None:
        self.db_path: Path | None = Path(db_path).expanduser() if db_path is not None else None
        mem_path = (
            Path(memory_db_path).expanduser()
            if memory_db_path is not None
            else DEFAULT_CLIENT_MEMORY_DB
        )
        self.memory_db_path = mem_path
        self._memory = _ClientMemoryStore(mem_path)

    def get_tools(self) -> list[dict[str, Any]]:
        return TOOLS

    def _tool_scout_digest(self, args: dict[str, Any]) -> dict[str, Any]:
        raw_path = str(args.get("path") or "").strip()
        question = str(args.get("question") or "").strip()
        if not raw_path:
            return {"isError": True, "content": [{"type": "text", "text": "Missing required `path`."}]}

        target = Path(raw_path).expanduser()
        if not target.exists():
            return {"isError": True, "content": [{"type": "text", "text": f"Path not found: {raw_path}"}]}

        t0 = time.perf_counter()
        terms = _query_terms(question)
        raw_bytes_est = 0

        if target.is_dir():
            body, raw_bytes_est = _digest_directory(target, terms)
        else:
            try:
                raw_bytes_est = target.stat().st_size
            except OSError:
                raw_bytes_est = 0
            raw_bytes = target.read_bytes()[:_MAX_READ_BYTES]
            raw_text = raw_bytes.decode("utf-8", errors="replace")
            if raw_bytes_est <= 0:
                raw_bytes_est = len(raw_text.encode("utf-8"))
            suffix = target.suffix.lower()
            if suffix == ".py":
                body = _digest_python(target, raw_text, terms)
            elif suffix in (".csv", ".tsv"):
                body = _digest_csv(target, raw_text, terms)
            else:
                body = _digest_text(target, raw_text, terms)

        bounded_body = _truncate(body, _MAX_DIGEST_CHARS)
        before_placeholders = bounded_body.count(REDACTION_PLACEHOLDER)
        scrubbed_body = redact_secrets(bounded_body)
        after_placeholders = scrubbed_body.count(REDACTION_PLACEHOLDER)
        secrets_blocked = max(0, after_placeholders - before_placeholders)

        raw_tokens_est = max(1, raw_bytes_est // 4)
        compressed_tokens_est = max(1, len(scrubbed_body) // 4)
        raw_tokens_est = max(raw_tokens_est, compressed_tokens_est)
        tokens_saved = max(0, raw_tokens_est - compressed_tokens_est)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        record_shield_event(
            event_type="local_metal_scout",
            target=str(target),
            raw_tokens_est=raw_tokens_est,
            compressed_tokens_est=compressed_tokens_est,
            tokens_saved=tokens_saved,
            secrets_blocked=secrets_blocked,
            latency_ms=latency_ms,
            bot_name="claude-desktop",
            action_mode="mcp_scout_digest",
            db_path=self.db_path,
        )

        safe_q = redact_secrets(question) if question else ""
        header = (
            f"### WBC Scout Digest: `{target.name or str(target)}`\n"
            f"- Raw est: `{raw_tokens_est:,}` tok | Digest: `{compressed_tokens_est:,}` tok | "
            f"Saved: `{tokens_saved:,}` tok | Secrets scrubbed: `{secrets_blocked}`"
        )
        if safe_q:
            header += f"\n- Focus: `{_truncate(safe_q, 120)}`"
        return {"isError": False, "content": [{"type": "text", "text": f"{header}\n\n{scrubbed_body}"}]}

    def _tool_client_memory(self, args: dict[str, Any]) -> dict[str, Any]:
        action = str(args.get("action") or "").strip().lower()
        client = str(args.get("client") or "").strip()
        text_or_query = str(args.get("text_or_query") or "").strip()
        if not client:
            return {"isError": True, "content": [{"type": "text", "text": "Missing required `client`."}]}

        if action == "remember":
            if not text_or_query:
                return {
                    "isError": True,
                    "content": [{"type": "text", "text": "Missing `text_or_query` for remember."}],
                }
            saved = self._memory.remember(client, text_or_query)
            return {
                "isError": False,
                "content": [
                    {
                        "type": "text",
                        "text": f"Stored fact #{saved['id']} for `{saved['client']}`: {saved['content']}",
                    }
                ],
            }

        if action == "recall":
            hits = self._memory.recall(client, text_or_query)
            if not hits:
                return {
                    "isError": False,
                    "content": [{"type": "text", "text": f"No matching facts for `{client.lower()}`."}],
                }
            lines = [f"Client `{client.lower()}` matches ({len(hits)}):"]
            lines.extend(f"- [#{h['id']}] {h['content']}" for h in hits)
            return {"isError": False, "content": [{"type": "text", "text": "\n".join(lines)}]}

        if action == "list":
            rows = self._memory.list_facts(client)
            if not rows:
                return {
                    "isError": False,
                    "content": [{"type": "text", "text": f"No stored facts for `{client.lower()}`."}],
                }
            lines = [f"Client `{client.lower()}` facts ({len(rows)}):"]
            lines.extend(f"- [#{r['id']}] {r['content']}" for r in rows)
            return {"isError": False, "content": [{"type": "text", "text": "\n".join(lines)}]}

        return {
            "isError": True,
            "content": [{"type": "text", "text": f"Unsupported action `{action}`. Use remember|recall|list."}],
        }

    def _tool_handoff_capsule(self, args: dict[str, Any]) -> dict[str, Any]:
        raw_goal = str(args.get("goal") or "").strip()
        if not raw_goal:
            return {"isError": True, "content": [{"type": "text", "text": "Missing required `goal`."}]}

        raw_decisions = args.get("key_decisions")
        if isinstance(raw_decisions, Sequence) and not isinstance(raw_decisions, (str, bytes)):
            decisions_str = "; ".join(str(x).strip() for x in raw_decisions if str(x).strip())
        else:
            decisions_str = str(raw_decisions or "").strip()
        raw_next = str(args.get("next_step") or "").strip()

        goal = _truncate(redact_secrets(raw_goal), 360)
        decisions = _truncate(redact_secrets(decisions_str or "None recorded"), 560)
        next_step = _truncate(redact_secrets(raw_next or "Continue from current state"), 360)

        capsule = (
            "## StateCapsule Handoff\n"
            f"- **Goal:** {goal}\n"
            f"- **Key Decisions:** {decisions}\n"
            f"- **Next Step:** {next_step}\n\n"
            "Paste into a fresh thread to resume with zero history bloat."
        )
        capsule = _truncate(capsule, _MAX_CAPSULE_CHARS)

        copied = False
        is_headless = os.environ.get("WBC_HEADLESS") == "1"
        is_testing = os.environ.get("WBC_TESTING") == "1"
        if not is_headless and not is_testing and sys.platform == "darwin":
            pbcopy = shutil.which("pbcopy")
            if pbcopy:
                try:
                    subprocess.run(
                        [pbcopy],
                        input=capsule.encode("utf-8"),
                        check=True,
                        timeout=2.0,
                    )
                    copied = True
                except (subprocess.SubprocessError, OSError):
                    copied = False

        tok_est = max(1, len(capsule) // 4)
        footer = f"\n\n[tokens_est={tok_est} | clipboard_copied={str(copied).lower()}]"
        return {"isError": False, "content": [{"type": "text", "text": capsule + footer}]}

    def _tool_finops_receipt(self, args: dict[str, Any]) -> dict[str, Any]:
        window = str(args.get("window") or "all").strip() or "all"
        client = str(args.get("client") or "").strip() or None

        sav = get_savings_summary(window=window, db_path=self.db_path)
        proxy = get_proxy_client_summary(window=window, client_filter=client, db_path=self.db_path)

        lines = [
            f"### WBC FinOps Receipt (window={sav.get('window', window)})",
            "**Local Shield (`body savings`):**",
            f"- Events: `{int(sav.get('total_events', 0)):,}` | Direct tokens saved: `{int(sav.get('direct_tokens_saved', 0)):,}`",
            f"- Cumulative saved: `{int(sav.get('cumulative_tokens_saved', 0)):,}` tok | Est USD saved: `${float(sav.get('api_dollars_saved_est', 0.0)):.4f}`",
            f"- Compression: `{float(sav.get('compression_pct', 0.0)):.1f}%` | Secrets blocked: `{int(sav.get('secrets_blocked', 0)):,}`",
            "**API Proxy (`proxy_finops_ledger`):**",
            f"- Calls: `{int(proxy.get('total_calls', 0)):,}` (cached: `{int(proxy.get('cached_calls', 0)):,}`) | Today: `{int(proxy.get('calls_today', 0)):,}` calls (`${float(proxy.get('saved_today_usd', 0.0)):.4f}` saved)",
            f"- Baseline: `${float(proxy.get('baseline_usd', 0.0)):.4f}` -> Actual: `${float(proxy.get('actual_usd', 0.0)):.4f}` | Net saved: `${float(proxy.get('net_usd_saved', 0.0)):.4f}`",
        ]
        clients = proxy.get("clients") or []
        if clients:
            lines.append("**By Client:**")
            for c in clients[:10]:
                lines.append(
                    f"- `{c.get('client_slug', 'untagged')}`: {int(c.get('calls', 0))} calls, "
                    f"{int(c.get('cache_read_tokens', 0)):,} cache_read tok, "
                    f"net saved `${float(c.get('net_usd_saved', 0.0)):.4f}`"
                )
        return {"isError": False, "content": [{"type": "text", "text": "\n".join(lines)}]}

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        args = arguments if isinstance(arguments, dict) else {}
        if name == "wbc_scout_digest":
            return self._tool_scout_digest(args)
        if name == "wbc_client_memory":
            return self._tool_client_memory(args)
        if name == "wbc_handoff_capsule":
            return self._tool_handoff_capsule(args)
        if name == "wbc_finops_receipt":
            return self._tool_finops_receipt(args)
        return {"isError": True, "content": [{"type": "text", "text": f"Unknown tool: {name}"}]}

    def handle_rpc_message(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(msg, dict):
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "Invalid Request"},
            }
        method = str(msg.get("method") or "")
        msg_id = msg.get("id")

        if method == "initialize":
            params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
            proto = str(params.get("protocolVersion") or "2024-11-05")
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": proto,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "whole-body-compute", "version": __version__},
                },
            }
        if method.startswith("notifications/"):
            return None
        if method == "ping":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": self.get_tools()}}
        if method == "tools/call":
            params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
            tool_name = str(params.get("name") or "")
            tool_args = params.get("arguments")
            res = self.call_tool(tool_name, tool_args if isinstance(tool_args, dict) else {})
            return {"jsonrpc": "2.0", "id": msg_id, "result": res}

        if msg_id is not None:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        return None

    def run_stdio(
        self,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
    ) -> None:
        in_stream = stdin if stdin is not None else sys.stdin
        out_stream = stdout if stdout is not None else sys.stdout
        for line in in_stream:
            raw = line.strip()
            if not raw:
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                out_stream.write(
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {"code": -32700, "message": "Parse error"},
                        }
                    )
                    + "\n"
                )
                out_stream.flush()
                continue
            resp = self.handle_rpc_message(msg)
            if resp is not None:
                out_stream.write(json.dumps(resp) + "\n")
                out_stream.flush()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WBC Claude Desktop Quota Bouncer MCP server")
    parser.add_argument("--db-path", default=None, help="Optional shield ledger SQLite path")
    parser.add_argument("--memory-db", default=None, help="Optional client memory SQLite path")
    args = parser.parse_args(list(argv) if argv is not None else None)
    server = ClaudeDesktopBouncerServer(db_path=args.db_path, memory_db_path=args.memory_db)
    server.run_stdio()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
