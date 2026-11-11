"""
VoiceFi Local Symbol Indexer & Sovereign Airlock (Phase 1).
Extracts and indexes rich AST contracts (signatures, 1-line docstrings, call graphs,
and deterministic body invariants) into SQLite for sub-millisecond lookups, .pyi
skeleton rendering, and Selective Perfusion bundles without raw file bloat.
"""

import ast
import json
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from whole_body_compute.organs.immune_polyglot import (
    POLYGLOT_EXTENSIONS,
    extract_polyglot_symbols,
    fence_language_for_path,
)

logger = logging.getLogger("whole_body_compute.local.symbols")

def _get_default_symbols_db() -> Path:
    try:
        from whole_body_compute.procedural_telemetry import get_wbc_data_dir

        return (get_wbc_data_dir() / "symbols.db").expanduser()
    except Exception:
        return Path.home() / ".whole_body_compute" / "symbols.db"


DB_PATH = _get_default_symbols_db()
_INITIAL_DB_PATH = DB_PATH

_QUERY_STOPWORDS: Set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "analyze",
    "file",
    "find",
    "identify",
    "extract",
    "any",
}


@dataclass
class SymbolRecord:
    name: str
    kind: str  # "function", "class", "method", "async_function", "async_method", "constant"
    file_path: str
    start_line: int
    end_line: int
    parent_class: Optional[str] = None
    signature: str = ""
    docstring: str = ""
    callees: List[str] = field(default_factory=list)
    invariants: List[str] = field(default_factory=list)

    @property
    def parent(self) -> Optional[str]:
        return self.parent_class

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "parent_class": self.parent_class,
            "signature": self.signature,
            "docstring": self.docstring,
            "callees": list(self.callees),
            "invariants": list(self.invariants),
        }


# Alias for Sovereign Airlock compatibility
SymbolEntry = SymbolRecord


def _tokenize_query(query: str) -> List[str]:
    """Split a query into normalized search tokens (including snake_case/camelCase parts)."""
    raw_tokens = re.findall(r"[A-Za-z0-9_]+", query or "")
    tokens: List[str] = []
    for tok in raw_tokens:
        low = tok.lower()
        if len(low) >= 2 and low not in _QUERY_STOPWORDS and low not in tokens:
            tokens.append(low)
        # Split camelCase and snake_case into sub-tokens
        sub_parts = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", tok).lower().split("_")
        for sp in sub_parts:
            if len(sp) >= 3 and sp not in _QUERY_STOPWORDS and sp not in tokens:
                tokens.append(sp)
    return tokens


class LocalSymbolIndex:
    """Fast SQLite-backed symbol index and deterministic .pyi Airlock for codebases."""

    SUPPORTED_EXTENSIONS = (
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
        ".rs",
        ".go",
        ".java",
        ".kt",
        ".kts",
        ".sh",
        ".bash",
        ".zsh",
        ".yaml",
        ".yml",
    )

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        root_dir: Optional[Union[str, Path]] = None,
    ):
        if db_path is not None:
            self.db_path = Path(db_path)
        elif DB_PATH != _INITIAL_DB_PATH:
            self.db_path = Path(DB_PATH)
        else:
            self.db_path = _get_default_symbols_db()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.root_dir = Path(root_dir).resolve() if root_dir else None
        self._last_check: float = 0.0
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS file_meta (
                    path TEXT PRIMARY KEY,
                    mtime REAL,
                    size INTEGER DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS files_meta (
                    path TEXT PRIMARY KEY,
                    mtime REAL,
                    size INTEGER DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS symbols (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    kind TEXT,
                    file_path TEXT,
                    start_line INTEGER,
                    end_line INTEGER,
                    parent_class TEXT,
                    signature TEXT DEFAULT '',
                    docstring TEXT DEFAULT '',
                    callees TEXT DEFAULT '[]',
                    invariants TEXT DEFAULT '[]'
                )
                """
            )
            # Seamless schema auto-migration for existing ~/.voicefi/symbols.db files
            migrations = [
                "ALTER TABLE file_meta ADD COLUMN size INTEGER DEFAULT 0",
                "ALTER TABLE files_meta ADD COLUMN size INTEGER DEFAULT 0",
                "ALTER TABLE symbols ADD COLUMN signature TEXT DEFAULT ''",
                "ALTER TABLE symbols ADD COLUMN docstring TEXT DEFAULT ''",
                "ALTER TABLE symbols ADD COLUMN callees TEXT DEFAULT '[]'",
                "ALTER TABLE symbols ADD COLUMN invariants TEXT DEFAULT '[]'",
            ]
            for stmt in migrations:
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError:
                    pass

            conn.execute("CREATE INDEX IF NOT EXISTS idx_sym_name ON symbols(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sym_path ON symbols(file_path)")

    def _row_to_record(self, r: sqlite3.Row) -> SymbolRecord:
        keys = r.keys()
        callees_raw = r["callees"] if "callees" in keys and r["callees"] else "[]"
        invariants_raw = r["invariants"] if "invariants" in keys and r["invariants"] else "[]"
        try:
            callees = json.loads(callees_raw) if isinstance(callees_raw, str) else list(callees_raw)
        except Exception:
            callees = []
        try:
            invariants = (
                json.loads(invariants_raw)
                if isinstance(invariants_raw, str)
                else list(invariants_raw)
            )
        except Exception:
            invariants = []

        return SymbolRecord(
            name=r["name"],
            kind=r["kind"],
            file_path=r["file_path"],
            start_line=r["start_line"],
            end_line=r["end_line"],
            parent_class=r["parent_class"],
            signature=r["signature"] or "",
            docstring=r["docstring"] or "",
            callees=callees,
            invariants=invariants,
        )

    def _store_file_symbols(
        self,
        conn: sqlite3.Connection,
        file_path_str: str,
        mtime: float,
        size: int,
        symbols: List[SymbolRecord],
    ) -> None:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM symbols WHERE file_path = ?", (file_path_str,))
        for s in symbols:
            cursor.execute(
                """
                INSERT INTO symbols (
                    name, kind, file_path, start_line, end_line,
                    parent_class, signature, docstring, callees, invariants
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    s.name,
                    s.kind,
                    s.file_path,
                    s.start_line,
                    s.end_line,
                    s.parent_class,
                    s.signature,
                    s.docstring,
                    json.dumps(s.callees),
                    json.dumps(s.invariants),
                ),
            )
        cursor.execute(
            "INSERT OR REPLACE INTO file_meta (path, mtime, size) VALUES (?, ?, ?)",
            (file_path_str, mtime, size),
        )
        cursor.execute(
            "INSERT OR REPLACE INTO files_meta (path, mtime, size) VALUES (?, ?, ?)",
            (file_path_str, mtime, size),
        )

    def index_file(self, file_path: Union[str, Path]) -> List[SymbolRecord]:
        """Parse and index a single file (.py via AST, .ts/.js/.sh/.yaml via regex fallback)."""
        p = Path(file_path).resolve()
        if not p.is_file():
            return []
        try:
            stat = p.stat()
            symbols = self._parse_file(p)
            with self._get_connection() as conn:
                self._store_file_symbols(conn, str(p), stat.st_mtime, stat.st_size, symbols)
                conn.commit()
            return symbols
        except Exception as e:
            logger.debug(f"Error indexing file {p}: {e}")
            return []

    def index_directory(self, root_dir: Optional[Union[str, Path]] = None) -> int:
        """Scan and index all supported source files under root_dir incrementally."""
        return self.index_repo(root_dir)

    def index_repo(self, root_dir: Optional[Union[str, Path]] = None) -> int:
        """Scan and index all python and polyglot files under root_dir incrementally."""
        target = root_dir or self.root_dir or "."
        root = Path(target).resolve()
        if root.is_file():
            return len(self.index_file(root))

        indexed_count = 0
        candidate_files: List[Path] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d
                for d in dirnames
                if not d.startswith(".")
                and d not in ("venv", ".venv", "node_modules", "dist", "build", "__pycache__")
            ]
            for f in filenames:
                if f.endswith(self.SUPPORTED_EXTENSIONS):
                    candidate_files.append(Path(dirpath) / f)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            for p in candidate_files:
                try:
                    p_resolved = p.resolve()
                    stat = p_resolved.stat()
                    cursor.execute(
                        "SELECT mtime, size FROM files_meta WHERE path = ?",
                        (str(p_resolved),),
                    )
                    row = cursor.fetchone()
                    if (
                        row
                        and row["mtime"] == stat.st_mtime
                        and (row["size"] == stat.st_size or row["size"] == 0)
                    ):
                        continue

                    symbols = self._parse_file(p_resolved)
                    self._store_file_symbols(
                        conn, str(p_resolved), stat.st_mtime, stat.st_size, symbols
                    )
                    indexed_count += len(symbols)
                except Exception as e:
                    logger.debug(f"Error indexing {p}: {e}")

            conn.commit()

        return indexed_count

    def get_file_symbols(self, file_path: Union[str, Path]) -> List[SymbolRecord]:
        """
        Return all indexed symbols for file_path.
        Always verifies stat().st_mtime and st_size against files_meta and auto-reindexes
        inline (<1.5ms) if missing or stale.
        """
        p = Path(file_path).resolve()
        if not p.is_file():
            return []

        try:
            stat = p.stat()
        except OSError:
            return []

        needs_reindex = False
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT mtime, size FROM files_meta WHERE path = ?",
                (str(p),),
            )
            meta = cursor.fetchone()
            if not meta:
                cursor.execute(
                    "SELECT mtime, size FROM file_meta WHERE path = ?",
                    (str(p),),
                )
                meta = cursor.fetchone()

            if (
                not meta
                or meta["mtime"] != stat.st_mtime
                or (meta["size"] not in (0, None) and meta["size"] != stat.st_size)
            ):
                needs_reindex = True
            else:
                cursor.execute(
                    "SELECT * FROM symbols WHERE file_path = ? ORDER BY start_line ASC, id ASC",
                    (str(p),),
                )
                rows = cursor.fetchall()
                return [self._row_to_record(r) for r in rows]

        if needs_reindex:
            return self.index_file(p)
        return []

    def format_symbol_map(self, file_path: Union[str, Path]) -> str:
        """Format a compact AST symbol map with line bounds and signatures."""
        p = Path(file_path).resolve()
        symbols = self.get_file_symbols(p)
        if not symbols:
            return f"# No symbols indexed for {p.name}"
        lines = [f"# Symbol Map: {p.name}"]
        for s in symbols:
            prefix = "  " if s.parent_class else ""
            sig = s.signature or f"{s.kind} {s.name}"
            doc_part = f" — {s.docstring}" if s.docstring else ""
            lines.append(f"{prefix}- [L{s.start_line}-L{s.end_line}] {sig}{doc_part}")
        return "\n".join(lines)

    @staticmethod
    def _summarize_docstring(doc: Optional[str]) -> str:
        if not doc:
            return ""
        for line in doc.strip().splitlines():
            cleaned = line.strip()
            if cleaned:
                return cleaned[:120]
        return ""

    def _parse_file(self, path: Path) -> List[SymbolRecord]:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()
        except Exception:
            return []

        ext = path.suffix.lower()
        if ext in POLYGLOT_EXTENSIONS:
            poly_items, _ = extract_polyglot_symbols(
                path, source, prefer_treesitter=True, include_details=True
            )
            path_str = str(path)
            return [
                SymbolRecord(
                    name=item.name,
                    kind=item.kind,
                    file_path=path_str,
                    start_line=item.start_line,
                    end_line=item.end_line,
                    parent_class=item.parent_class,
                    signature=item.signature,
                    docstring=item.docstring,
                    callees=list(item.callees),
                    invariants=list(item.invariants),
                )
                for item in poly_items
            ]
        if ext != ".py":
            return self._parse_regex_fallback(path, source)

        try:
            tree = ast.parse(source, filename=str(path))
        except Exception:
            return []

        return self._extract_ast_symbols(path, tree)

    def _extract_ast_symbols(self, path: Path, tree: ast.Module) -> List[SymbolRecord]:
        symbols: List[SymbolRecord] = []
        path_str = str(path)

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                doc = self._summarize_docstring(ast.get_docstring(node))
                bases = [ast.unparse(b) for b in node.bases]
                cls_sig = (
                    f"class {node.name}({', '.join(bases)})"
                    if bases
                    else f"class {node.name}"
                )
                cls_invariants = self._extract_invariants(node, shallow_class=True)
                cls_callees = self._extract_callees(node, shallow_class=True)
                symbols.append(
                    SymbolRecord(
                        name=node.name,
                        kind="class",
                        file_path=path_str,
                        start_line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        signature=cls_sig,
                        docstring=doc,
                        callees=cls_callees,
                        invariants=cls_invariants,
                    )
                )
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        m_doc = self._summarize_docstring(ast.get_docstring(item))
                        kind = (
                            "async_method"
                            if isinstance(item, ast.AsyncFunctionDef)
                            else "method"
                        )
                        symbols.append(
                            SymbolRecord(
                                name=item.name,
                                kind=kind,
                                file_path=path_str,
                                start_line=item.lineno,
                                end_line=getattr(item, "end_lineno", item.lineno),
                                parent_class=node.name,
                                signature=self._format_signature(item),
                                docstring=m_doc,
                                callees=self._extract_callees(item),
                                invariants=self._extract_invariants(item),
                            )
                        )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                f_doc = self._summarize_docstring(ast.get_docstring(node))
                kind = (
                    "async_function"
                    if isinstance(node, ast.AsyncFunctionDef)
                    else "function"
                )
                symbols.append(
                    SymbolRecord(
                        name=node.name,
                        kind=kind,
                        file_path=path_str,
                        start_line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        signature=self._format_signature(node),
                        docstring=f_doc,
                        callees=self._extract_callees(node),
                        invariants=self._extract_invariants(node),
                    )
                )
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and not target.id.startswith("__"):
                        if target.id == "logger":
                            continue
                        val_str = self._compact_value(node.value)
                        symbols.append(
                            SymbolRecord(
                                name=target.id,
                                kind="constant",
                                file_path=path_str,
                                start_line=node.lineno,
                                end_line=getattr(node, "end_lineno", node.lineno),
                                signature=f"{target.id} = {val_str}",
                            )
                        )
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                name = node.target.id
                if not name.startswith("__") and name != "logger":
                    ann_str = ast.unparse(node.annotation)
                    if node.value is not None:
                        val_str = self._compact_value(node.value)
                        sig = f"{name}: {ann_str} = {val_str}"
                    else:
                        sig = f"{name}: {ann_str}"
                    symbols.append(
                        SymbolRecord(
                            name=name,
                            kind="constant",
                            file_path=path_str,
                            start_line=node.lineno,
                            end_line=getattr(node, "end_lineno", node.lineno),
                            signature=sig,
                        )
                    )

        return symbols

    @staticmethod
    def _compact_value(val_node: ast.AST, max_len: int = 48) -> str:
        try:
            raw = ast.unparse(val_node).strip()
        except Exception:
            return "..."
        if "\n" in raw or len(raw) > max_len:
            if isinstance(val_node, ast.Dict):
                return "{...}"
            if isinstance(val_node, (ast.List, ast.Set, ast.Tuple)):
                return "[...]"
            return raw.splitlines()[0][:max_len] + "..."
        return raw

    def _format_args(self, args_node: ast.arguments) -> str:
        """Legacy helper returning (arg1, arg2, *args, **kwargs)."""
        args_list = [a.arg for a in args_node.args]
        if args_node.vararg:
            args_list.append(f"*{args_node.vararg.arg}")
        if args_node.kwarg:
            args_list.append(f"**{args_node.kwarg.arg}")
        return "(" + ", ".join(args_list) + ")"

    def _format_signature(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
        prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
        args_node = node.args
        parts: List[str] = []

        pos_args = list(getattr(args_node, "posonlyargs", [])) + list(args_node.args)
        num_defaults = len(args_node.defaults)
        default_offset = len(pos_args) - num_defaults

        for idx, arg in enumerate(pos_args):
            piece = arg.arg
            if arg.annotation is not None:
                try:
                    piece += f": {ast.unparse(arg.annotation)}"
                except Exception:
                    pass
            if idx >= default_offset:
                def_node = args_node.defaults[idx - default_offset]
                piece += f" = {self._compact_value(def_node, max_len=24)}"
            parts.append(piece)

        if args_node.vararg is not None:
            v_piece = f"*{args_node.vararg.arg}"
            if args_node.vararg.annotation is not None:
                try:
                    v_piece += f": {ast.unparse(args_node.vararg.annotation)}"
                except Exception:
                    pass
            parts.append(v_piece)
        elif args_node.kwonlyargs:
            parts.append("*")

        for kw_arg, kw_def in zip(args_node.kwonlyargs, args_node.kw_defaults):
            k_piece = kw_arg.arg
            if kw_arg.annotation is not None:
                try:
                    k_piece += f": {ast.unparse(kw_arg.annotation)}"
                except Exception:
                    pass
            if kw_def is not None:
                k_piece += f" = {self._compact_value(kw_def, max_len=24)}"
            parts.append(k_piece)

        if args_node.kwarg is not None:
            kw_piece = f"**{args_node.kwarg.arg}"
            if args_node.kwarg.annotation is not None:
                try:
                    kw_piece += f": {ast.unparse(args_node.kwarg.annotation)}"
                except Exception:
                    pass
            parts.append(kw_piece)

        ret_suffix = ""
        if node.returns is not None:
            try:
                ret_suffix = f" -> {ast.unparse(node.returns)}"
            except Exception:
                pass

        return f"{prefix}{node.name}({', '.join(parts)}){ret_suffix}"

    @staticmethod
    def _iter_scope_nodes(node: ast.AST, shallow_class: bool = False):
        """Yield AST nodes inside `node`, optionally skipping nested method definitions on classes."""
        stack = [node]
        while stack:
            curr = stack.pop()
            yield curr
            for child in reversed(list(ast.iter_child_nodes(curr))):
                if shallow_class and curr is node and isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    continue
                stack.append(child)

    def _extract_callees(self, node: ast.AST, shallow_class: bool = False) -> List[str]:
        callees: List[str] = []
        seen: Set[str] = set()
        for sub in self._iter_scope_nodes(node, shallow_class=shallow_class):
            if isinstance(sub, ast.Call):
                callee_name: Optional[str] = None
                if isinstance(sub.func, ast.Name):
                    callee_name = sub.func.id
                elif isinstance(sub.func, ast.Attribute):
                    callee_name = sub.func.attr
                if callee_name and callee_name not in seen:
                    seen.add(callee_name)
                    callees.append(callee_name)
        return callees

    def _extract_invariants(self, node: ast.AST, shallow_class: bool = False) -> List[str]:
        invariants: List[str] = []
        seen: Set[str] = set()

        def _add(marker: str) -> None:
            if marker and marker not in seen:
                seen.add(marker)
                invariants.append(marker)

        for sub in self._iter_scope_nodes(node, shallow_class=shallow_class):
            if isinstance(sub, ast.Raise):
                if sub.exc is None:
                    _add("raises")
                elif isinstance(sub.exc, ast.Call):
                    if isinstance(sub.exc.func, ast.Name):
                        _add(f"raises {sub.exc.func.id}")
                    elif isinstance(sub.exc.func, ast.Attribute):
                        _add(f"raises {sub.exc.func.attr}")
                    else:
                        _add(f"raises {ast.unparse(sub.exc.func)}")
                elif isinstance(sub.exc, ast.Name):
                    _add(f"raises {sub.exc.id}")
                else:
                    _add(f"raises {ast.unparse(sub.exc)[:32]}")

            elif isinstance(sub, ast.Assert):
                _add("asserts")

            elif isinstance(sub, (ast.With, ast.AsyncWith)):
                for item in sub.items:
                    try:
                        ctx_str = ast.unparse(item.context_expr).strip()
                    except Exception:
                        ctx_str = "context"
                    _add(f"with {ctx_str[:40]}")

            elif isinstance(sub, ast.Call):
                try:
                    func_unparsed = ast.unparse(sub.func)
                except Exception:
                    func_unparsed = ""
                func_short = (
                    sub.func.id
                    if isinstance(sub.func, ast.Name)
                    else (sub.func.attr if isinstance(sub.func, ast.Attribute) else "")
                )

                # Environment variable lookup: os.environ.get("KEY") / os.getenv("KEY")
                if func_unparsed in (
                    "os.environ.get",
                    "environ.get",
                    "os.getenv",
                    "getenv",
                ) and sub.args:
                    first = sub.args[0]
                    if isinstance(first, ast.Constant) and isinstance(first.value, str):
                        _add(f"env:{first.value}")

                # I/O markers: open, urlopen, subprocess
                if func_short == "open" or func_unparsed in ("os.open", "io.open", "Path.open"):
                    _add("io:open")
                elif func_short == "urlopen" or "urlopen" in func_unparsed:
                    _add("io:urlopen")
                elif func_unparsed.startswith("subprocess.") or func_short in (
                    "Popen",
                    "check_output",
                    "check_call",
                ):
                    _add("io:subprocess")

            elif isinstance(sub, ast.Subscript):
                try:
                    val_unparsed = ast.unparse(sub.value)
                except Exception:
                    val_unparsed = ""
                if val_unparsed in ("os.environ", "environ"):
                    sl = sub.slice
                    if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                        _add(f"env:{sl.value}")

        return invariants

    @staticmethod
    def _compute_polyglot_end_line(
        lines: List[str], start_line: int, max_bound: int, suffix: str
    ) -> int:
        """
        Compute accurate end_line for brace-based and indentation-based files.
        Avoids spanning to EOF or trailing comments.
        """
        ext = suffix.lower()
        total_lines = len(lines)
        max_limit = min(max_bound, total_lines)
        start_idx = max(0, start_line - 1)

        if ext in (".yaml", ".yml"):
            if start_idx >= total_lines:
                return start_line
            first_line = lines[start_idx]
            base_indent = len(first_line) - len(first_line.lstrip())
            last_content_ln = start_line
            for idx in range(start_idx + 1, max_limit):
                line = lines[idx]
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                curr_indent = len(line) - len(line.lstrip())
                if curr_indent <= base_indent:
                    return last_content_ln
                last_content_ln = idx + 1
            return last_content_ln

        # Brace-based files (.ts, .tsx, .js, .jsx, .go, .rs, .c, .cpp, .java, .sh, .bash, .zsh)
        saw_open_brace = False
        brace_depth = 0
        in_block_comment = False
        last_non_blank_idx = start_idx
        is_shell = ext in (".sh", ".bash", ".zsh")

        for idx in range(start_idx, max_limit):
            line = lines[idx]
            stripped = line.strip()
            if stripped:
                last_non_blank_idx = idx

            i = 0
            n = len(line)
            while i < n:
                if in_block_comment:
                    if not is_shell and i + 1 < n and line[i : i + 2] == "*/":
                        in_block_comment = False
                        i += 2
                        continue
                    i += 1
                    continue

                if not is_shell and i + 1 < n and line[i : i + 2] == "/*":
                    in_block_comment = True
                    i += 2
                    continue

                if not is_shell and i + 1 < n and line[i : i + 2] == "//":
                    break

                if is_shell and line[i] == "#":
                    break

                if is_shell and line[i] == "$" and i + 1 < n and line[i + 1] == "{":
                    i += 2
                    while i < n and line[i] != "}":
                        if line[i] == "\\":
                            i += 1
                        i += 1
                    if i < n and line[i] == "}":
                        i += 1
                    continue

                ch = line[i]
                if ch in ('"', "'", "`"):
                    quote = ch
                    i += 1
                    while i < n:
                        if line[i] == "\\":
                            i += 2
                            continue
                        if line[i] == quote:
                            i += 1
                            break
                        i += 1
                    continue

                if ch == "{":
                    saw_open_brace = True
                    brace_depth += 1
                elif ch == "}":
                    if saw_open_brace:
                        brace_depth -= 1
                        if brace_depth <= 0:
                            return idx + 1
                i += 1

            if saw_open_brace and brace_depth <= 0:
                return idx + 1

            if not saw_open_brace:
                if stripped.endswith(";") and brace_depth == 0:
                    return idx + 1
                if idx > start_idx and not stripped:
                    return last_non_blank_idx + 1

        if saw_open_brace and brace_depth <= 0:
            return idx + 1
        return max(start_line, last_non_blank_idx + 1)

    @staticmethod
    def _extract_polyglot_callees(body_lines: List[str], own_name: str) -> List[str]:
        text = "\n".join(body_lines)
        keywords = {
            "if", "for", "while", "switch", "catch", "function", "return",
            "typeof", "sizeof", "await", "new", "import", "export", "class",
            "interface", own_name,
        }
        callees: List[str] = []
        seen = set()
        for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", text):
            name = m.group(1)
            if name not in keywords and name not in seen:
                seen.add(name)
                callees.append(name)
                if len(callees) >= 8:
                    break
        return callees

    @staticmethod
    def _extract_polyglot_invariants(body_lines: List[str]) -> List[str]:
        text = "\n".join(body_lines)
        invariants: List[str] = []
        seen = set()

        def _add(inv: str) -> None:
            if inv not in seen and len(invariants) < 6:
                seen.add(inv)
                invariants.append(inv)

        # 1. Error throwing: throw new (\w+) -> raises <Err>
        for m in re.finditer(r"throw\s+new\s+([A-Za-z0-9_]+)", text):
            _add(f"raises {m.group(1)}")

        # 2. Environment variables: process.env.KEY, import.meta.env.KEY, ${KEY} or $KEY
        for m in re.finditer(r"(?:process\.env|import\.meta\.env)\.([A-Z0-9_]+)", text):
            _add(f"env:{m.group(1)}")
        for m in re.finditer(r"\$\{?([A-Z][A-Z0-9_]{2,})\}?", text):
            _add(f"env:{m.group(1)}")

        # 3. Network IO: \bfetch\s*\(|axios\. -> io:network
        if re.search(r"\bfetch\s*\(|axios\.", text):
            _add("io:network")

        # 4. Filesystem IO: \bfs\. -> io:fs
        if re.search(r"\bfs\.", text):
            _add("io:fs")

        # 5. Subprocess IO: child_process|execSync|spawnSync -> io:subprocess
        if re.search(r"child_process|execSync|spawnSync", text):
            _add("io:subprocess")

        return invariants

    def _parse_regex_fallback(self, path: Path, source: str) -> List[SymbolRecord]:
        """Lightweight regex fallback for .ts, .js, .sh, and .yaml files."""
        ext = path.suffix.lower()
        lines = source.splitlines()
        raw_hits: List[Tuple[str, str, int, str]] = []  # (name, kind, start_line, signature)

        if ext in (".ts", ".tsx", ".js", ".jsx"):
            cls_re = re.compile(
                r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?(?:class|interface)\s+([A-Za-z_$][A-Za-z0-9_$]*)"
            )
            fn_re = re.compile(
                r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(([^)]*)\)"
            )
            arrow_re = re.compile(
                r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*(?::[^=]+)?=\s*(?:async\s*)?\([^)]*\)\s*=>"
            )
            const_re = re.compile(
                r"^\s*(?:export\s+)?(?:const|type)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*(?::[^=]+)?="
            )
            brace_depth = 0
            for idx, line in enumerate(lines, start=1):
                stripped = line.strip()
                is_top = (brace_depth == 0) or stripped.startswith("export ")
                if is_top and (m := cls_re.match(line)):
                    raw_hits.append((m.group(1), "class", idx, line.strip().rstrip("{").strip()))
                elif is_top and (m := fn_re.match(line)):
                    raw_hits.append((m.group(1), "function", idx, line.strip().rstrip("{").strip()))
                elif is_top and (m := arrow_re.match(line)):
                    raw_hits.append((m.group(1), "function", idx, line.strip().rstrip("{").strip()))
                elif is_top and (m := const_re.match(line)):
                    raw_hits.append((m.group(1), "constant", idx, line.strip()[:80]))

                if not stripped.startswith("//") and not stripped.startswith("/*"):
                    for ch in line:
                        if ch == "{":
                            brace_depth += 1
                        elif ch == "}":
                            brace_depth = max(0, brace_depth - 1)

        elif ext in (".sh", ".bash", ".zsh"):
            fn_re = re.compile(
                r"^\s*(?:function\s+([A-Za-z_][A-Za-z0-9_\-]*)(?:\s*\(\s*\))?|([A-Za-z_][A-Za-z0-9_\-]*)\s*\(\s*\))\s*\{?"
            )
            const_re = re.compile(r"^(?:export\s+)?([A-Z_][A-Z0-9_]*)=(.*)")
            brace_depth = 0
            for idx, line in enumerate(lines, start=1):
                stripped = line.strip()
                is_top = (brace_depth == 0)
                if m := fn_re.match(line):
                    name = m.group(1) or m.group(2)
                    raw_hits.append((name, "function", idx, f"{name}()"))
                elif is_top and (m := const_re.match(line)):
                    raw_hits.append((m.group(1), "constant", idx, line.strip()[:80]))

                if not stripped.startswith("#"):
                    for ch in line:
                        if ch == "{":
                            brace_depth += 1
                        elif ch == "}":
                            brace_depth = max(0, brace_depth - 1)

        elif ext in (".yaml", ".yml"):
            yaml_re = re.compile(r"^([A-Za-z0-9_\-]+)\s*:")
            for idx, line in enumerate(lines, start=1):
                if m := yaml_re.match(line):
                    raw_hits.append((m.group(1), "constant", idx, line.strip()[:80]))

        total_lines = max(1, len(lines))
        records: List[SymbolRecord] = []
        for i, (name, kind, start_ln, sig) in enumerate(raw_hits):
            max_bound = raw_hits[i + 1][2] - 1 if i + 1 < len(raw_hits) else total_lines
            end_ln = self._compute_polyglot_end_line(lines, start_ln, max_bound, ext)
            end_ln = max(start_ln, min(end_ln, max_bound, total_lines))

            body_lines = lines[start_ln - 1 : end_ln]
            callees = self._extract_polyglot_callees(body_lines, name)
            invariants = self._extract_polyglot_invariants(body_lines)

            records.append(
                SymbolRecord(
                    name=name,
                    kind=kind,
                    file_path=str(path),
                    start_line=start_ln,
                    end_line=end_ln,
                    signature=sig,
                    callees=callees,
                    invariants=invariants,
                )
            )
        return records

    # -------------------------------------------------------------------------
    # Sovereign Airlock Deterministic Methods (Phase 1)
    # -------------------------------------------------------------------------

    def render_skeleton(self, file_path: Union[str, Path]) -> str:
        """
        Generate a .pyi-style deterministic contract stub for file_path in <5ms.
        Includes module imports summary, constants, classes, method type signatures,
        [Lstart-Lend] line annotations, 1-line docstrings, # callees: ..., and
        # invariants: ... comments—with 0 lines of raw implementation boilerplate.

        SyntaxError Fail-Open Safeguard:
        If a Python file has a SyntaxError(lineno=X), returns a deterministic
        [X-25, X+25] raw line window around line X with an explanatory header.
        """
        p = Path(file_path).resolve()
        if not p.is_file():
            return f"# File not found: {file_path}"

        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()
        except Exception as e:
            return f"# Error reading {p.name}: {e}"

        ext = p.suffix.lower()
        if ext == ".py":
            try:
                tree = ast.parse(source, filename=str(p))
            except SyntaxError as exc:
                lineno = exc.lineno or 1
                lines = source.splitlines()
                total = len(lines)
                start_ln = max(1, lineno - 25)
                end_ln = min(total, lineno + 25)
                out = [
                    f"# [SYNTAX ERROR FAIL-OPEN WINDOW] {p.name}",
                    f"# SyntaxError at line {lineno} (offset {exc.offset}): {exc.msg}",
                    f"# Showing raw line window [L{start_ln}-L{end_ln}] around line {lineno}:",
                ]
                for idx in range(start_ln, end_ln + 1):
                    marker = ">>> " if idx == lineno else "    "
                    out.append(f"{marker}L{idx}: {lines[idx - 1]}")
                return "\n".join(out)

            # Parse and ensure cache is warm
            symbols = self.get_file_symbols(p)
            if not symbols:
                symbols = self._extract_ast_symbols(p, tree)

            sym_memories: Dict[str, List[str]] = {}
            try:
                from whole_body_compute.local.dot_memory import get_dot_store

                sym_names_query = [s.name for s in symbols]
                sym_memories = get_dot_store().get_symbol_memories(str(file_path), sym_names_query)
            except Exception:
                sym_memories = {}

            out_lines: List[str] = [f"# .pyi Airlock Skeleton: {p.name}"]
            mod_doc = self._summarize_docstring(ast.get_docstring(tree))
            if mod_doc:
                out_lines.append(f'"""{mod_doc}"""')
            for file_lesson in sym_memories.get("__file__", []):
                out_lines.append(f"# memory [file]: {file_lesson}")

            # Summarize module imports
            imports_summary: List[str] = []
            for node in tree.body:
                if isinstance(node, ast.Import):
                    names = ", ".join(a.name for a in node.names)
                    imports_summary.append(f"import {names}")
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    names = ", ".join(a.name for a in node.names[:6])
                    if len(node.names) > 6:
                        names += ", ..."
                    imports_summary.append(f"from {mod} import {names}")
            if imports_summary:
                out_lines.append("# imports: " + " | ".join(imports_summary[:12]))

            constants = [s for s in symbols if s.kind == "constant"]
            for c in constants:
                sig = c.signature or f"{c.name} = ..."
                out_lines.append(f"{sig}  # [L{c.start_line}-L{c.end_line}]")

            top_classes = [s for s in symbols if s.kind == "class"]
            methods_by_class: Dict[str, List[SymbolRecord]] = {}
            for s in symbols:
                if s.parent_class:
                    methods_by_class.setdefault(s.parent_class, []).append(s)

            for cls_sym in top_classes:
                out_lines.append("")
                cls_sig = cls_sym.signature or f"class {cls_sym.name}"
                out_lines.append(f"{cls_sig}:  # [L{cls_sym.start_line}-L{cls_sym.end_line}]")
                if cls_sym.docstring:
                    out_lines.append(f'    """{cls_sym.docstring}"""')
                if cls_sym.invariants:
                    out_lines.append(f"    # invariants: {', '.join(cls_sym.invariants)}")
                for cls_lesson in sym_memories.get(cls_sym.name, []):
                    out_lines.append(f"    # memory: {cls_lesson}")
                cls_methods = methods_by_class.get(cls_sym.name, [])
                if not cls_methods and not cls_sym.docstring:
                    out_lines.append("    ...")
                for m in cls_methods:
                    m_sig = m.signature or f"def {m.name}(...)"
                    out_lines.append(
                        f"    {m_sig}: ...  # [L{m.start_line}-L{m.end_line}]"
                    )
                    if m.docstring:
                        out_lines.append(f'        """{m.docstring}"""')
                    if m.callees:
                        out_lines.append(f"        # callees: {', '.join(m.callees)}")
                    if m.invariants:
                        out_lines.append(f"        # invariants: {', '.join(m.invariants)}")
                    qual_m = f"{cls_sym.name}.{m.name}"
                    m_lessons: List[str] = []
                    for l_str in sym_memories.get(qual_m, []) + sym_memories.get(m.name, []):
                        if l_str not in m_lessons:
                            m_lessons.append(l_str)
                    for m_lesson in m_lessons:
                        out_lines.append(f"        # memory: {m_lesson}")

            top_funcs = [
                f_sym
                for f_sym in symbols
                if f_sym.kind in ("function", "async_function") and not f_sym.parent_class
            ]
            for fn in top_funcs:
                out_lines.append("")
                fn_sig = fn.signature or f"def {fn.name}(...)"
                out_lines.append(f"{fn_sig}: ...  # [L{fn.start_line}-L{fn.end_line}]")
                if fn.docstring:
                    out_lines.append(f'    """{fn.docstring}"""')
                if fn.callees:
                    out_lines.append(f"    # callees: {', '.join(fn.callees)}")
                if fn.invariants:
                    out_lines.append(f"    # invariants: {', '.join(fn.invariants)}")
                for fn_lesson in sym_memories.get(fn.name, []):
                    out_lines.append(f"    # memory: {fn_lesson}")

            ext_contracts = self.resolve_external_callee_signatures(p, symbols)
            if ext_contracts:
                out_lines.append("")
                out_lines.append("# External Callee Contracts:")
                out_lines.extend(ext_contracts)

            return "\n".join(out_lines)

        # Non-Python fallback skeleton
        symbols = self.get_file_symbols(p)
        sym_memories_fb: Dict[str, List[str]] = {}
        try:
            from whole_body_compute.local.dot_memory import get_dot_store

            sym_memories_fb = get_dot_store().get_symbol_memories(
                str(file_path), [s.name for s in symbols]
            )
        except Exception:
            sym_memories_fb = {}
        out_lines = [f"# Airlock Skeleton: {p.name}"]
        for file_lesson in sym_memories_fb.get("__file__", []):
            out_lines.append(f"# memory [file]: {file_lesson}")
        for s in symbols:
            out_lines.append(f"{s.signature or s.name}  # [L{s.start_line}-L{s.end_line}]")
            for s_lesson in sym_memories_fb.get(s.name, []):
                out_lines.append(f"    # memory: {s_lesson}")

        ext_contracts = self.resolve_external_callee_signatures(p, symbols)
        if ext_contracts:
            out_lines.append("")
            out_lines.append("# External Callee Contracts:")
            out_lines.extend(ext_contracts)

        return "\n".join(out_lines)

    def resolve_external_callee_signatures(
        self,
        file_path: Union[str, Path],
        symbols: List[SymbolRecord],
        max_items: int = 8,
    ) -> List[str]:
        """
        Collect callees across symbols not defined locally in symbols,
        query the SQLite index for their signatures in other files,
        and format each hit as: # ext: {sig}  [{Path(fp).name}:L{s}-L{e}]
        """
        norm_fp = str(Path(file_path).resolve())
        local_names = {s.name for s in symbols}

        external_callees: List[str] = []
        seen = set()
        for s in symbols:
            for c in s.callees:
                if c not in local_names and c not in seen:
                    seen.add(c)
                    external_callees.append(c)

        if not external_callees:
            return []

        # Extract caller file imports to disambiguate same-named symbols across files
        caller_imports: Dict[str, str] = {}
        try:
            caller_p = Path(norm_fp)
            if not caller_p.exists() and Path(file_path).exists():
                caller_p = Path(file_path)
            if caller_p.suffix == ".py" and caller_p.exists():
                src_content = caller_p.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(src_content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        mod = (node.module or "").split(".")[-1]
                        for alias in node.names:
                            caller_imports[alias.asname or alias.name] = mod
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            caller_imports[alias.asname or alias.name] = alias.name.split(".")[-1]
        except Exception:
            pass

        results: List[str] = []
        try:
            with self._get_connection() as conn:
                for c_name in external_callees:
                    rows = conn.execute(
                        "SELECT name, signature, file_path, start_line, end_line FROM symbols WHERE name = ? AND file_path != ?",
                        (c_name, norm_fp),
                    ).fetchall()
                    if not rows and norm_fp != str(file_path):
                        rows = conn.execute(
                            "SELECT name, signature, file_path, start_line, end_line FROM symbols WHERE name = ? AND file_path != ?",
                            (c_name, str(file_path)),
                        ).fetchall()
                    if not rows:
                        continue

                    matched_row = None
                    if len(rows) == 1:
                        matched_row = rows[0]
                    else:
                        # Ambiguous: multiple candidates across files. Disambiguate via caller imports
                        imported_mod = caller_imports.get(c_name)
                        if imported_mod:
                            for r in rows:
                                if Path(r["file_path"]).stem == imported_mod:
                                    matched_row = r
                                    break
                        if not matched_row:
                            # Still ambiguous: do not assert an arbitrary signature
                            cand_names = ", ".join(Path(r["file_path"]).name for r in rows[:3])
                            results.append(f"# ext (ambiguous, {len(rows)} candidates): def {c_name}(...)  [{cand_names}]")
                            if len(results) >= max_items:
                                break
                            continue

                    sig = matched_row["signature"] or f"def {matched_row['name']}(...)"
                    fp = matched_row["file_path"]
                    s = matched_row["start_line"]
                    e = matched_row["end_line"]
                    results.append(f"# ext: {sig}  [{Path(fp).name}:L{s}-L{e}]")
                    if len(results) >= max_items:
                        break
        except Exception:
            return []

        return results

    def zoom_symbol(
        self,
        file_path: Union[str, Path],
        symbol_name: str,
        max_lines: int = 80,
    ) -> str:
        """
        Locate symbol_name (supports 'ClassName.method' or 'method') in file_path
        and return its exact source lines [start_line..end_line] (capped at max_lines
        with line numbers prefixed).
        """
        p = Path(file_path).resolve()
        if not p.is_file():
            return f"# File not found: {file_path}"

        symbols = self.get_file_symbols(p)
        target_sym: Optional[SymbolRecord] = None

        if "." in symbol_name:
            cls_part, meth_part = symbol_name.split(".", 1)
            for s in symbols:
                if s.parent_class == cls_part and s.name == meth_part:
                    target_sym = s
                    break
            if not target_sym:
                for s in symbols:
                    if s.name == meth_part:
                        target_sym = s
                        break
        else:
            for s in symbols:
                if s.name == symbol_name:
                    target_sym = s
                    break
            if not target_sym:
                low_q = symbol_name.lower()
                for s in symbols:
                    if s.name.lower() == low_q or low_q in s.name.lower():
                        target_sym = s
                        break

        if not target_sym:
            return f"# Symbol '{symbol_name}' not found in {p.name}"

        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines()
        except Exception as e:
            return f"# Error reading {p.name}: {e}"

        start_ln = max(1, target_sym.start_line)
        full_end_ln = min(len(lines), max(start_ln, target_sym.end_line))
        capped_end_ln = min(full_end_ln, start_ln + max(1, max_lines) - 1)

        qual_name = (
            f"{target_sym.parent_class}.{target_sym.name}"
            if target_sym.parent_class
            else target_sym.name
        )
        out = [
            f"# Zoom: {qual_name} in {p.name} [L{start_ln}-L{full_end_ln}]"
        ]
        try:
            from whole_body_compute.local.dot_memory import get_dot_store

            zoom_mems = get_dot_store().get_symbol_memories(
                str(p), [target_sym.name, qual_name]
            )
            seen_z_lessons: List[str] = []
            for z_l in zoom_mems.get(qual_name, []) + zoom_mems.get(target_sym.name, []):
                if z_l not in seen_z_lessons:
                    seen_z_lessons.append(z_l)
                    out.append(f"# memory: {z_l}")
        except Exception:
            pass
        for ln in range(start_ln, capped_end_ln + 1):
            out.append(f"L{ln}: {lines[ln - 1]}")

        if full_end_ln > capped_end_ln:
            out.append(
                f"# ... [truncated at max_lines={max_lines}; full symbol spans L{start_ln}-L{full_end_ln}]"
            )
        return "\n".join(out)

    def build_selective_bundle(
        self,
        target_path: Optional[Union[str, Path]] = None,
        query: str = "",
        max_tokens: int = 3500,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Mode 2 (SELECTIVE_PERFUSION):
        Deterministically score symbols against query tokens (identifier match +
        docstring/signature overlap + 2-hop callee/caller neighborhood + fast lexical
        substring match), pack exact raw source bodies of the top matching functions
        alongside .pyi skeletons of neighboring context up to max_tokens.
        """
        if target_path is None:
            target_path = kwargs.get("target") or kwargs.get("target_dir") or self.root_dir or Path.cwd()
        if not query and "query" in kwargs:
            query = kwargs["query"]

        t0 = time.perf_counter()
        target = Path(target_path).resolve()
        query_tokens = _tokenize_query(query)

        candidate_files: List[Path] = []
        if target.is_file():
            candidate_files.append(target)
        elif target.is_dir():
            for dirpath, dirnames, filenames in os.walk(target):
                dirnames[:] = [
                    d
                    for d in dirnames
                    if not d.startswith(".")
                    and d not in ("venv", ".venv", "node_modules", "dist", "build", "__pycache__")
                ]
                for f in filenames:
                    if f.endswith(self.SUPPORTED_EXTENSIONS):
                        candidate_files.append(Path(dirpath) / f)

        file_sources: Dict[str, List[str]] = {}
        all_symbols: List[SymbolRecord] = []
        for fp in candidate_files:
            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    file_sources[str(fp)] = f.read().splitlines()
                all_symbols.extend(self.get_file_symbols(fp))
            except Exception:
                pass

        # Score each symbol
        direct_scores: Dict[int, float] = {}
        for idx, sym in enumerate(all_symbols):
            score = 0.0
            name_low = sym.name.lower()
            name_parts = set(name_low.split("_"))
            sig_low = sym.signature.lower()
            doc_low = sym.docstring.lower()
            inv_low = " ".join(sym.invariants).lower()
            callees_low = {c.lower() for c in sym.callees}

            src_lines = file_sources.get(sym.file_path, [])
            body_slice = "\n".join(
                src_lines[max(0, sym.start_line - 1) : min(len(src_lines), sym.end_line)]
            ).lower()

            for tok in query_tokens:
                if tok == name_low or tok in name_parts:
                    score += 12.0
                elif tok in name_low:
                    score += 6.0
                if sym.parent_class and tok in sym.parent_class.lower():
                    score += 3.0
                if tok in sig_low:
                    score += 3.5
                if tok in doc_low:
                    score += 3.0
                if tok in inv_low:
                    score += 2.5
                if tok in callees_low:
                    score += 2.5
                if tok in body_slice:
                    score += 2.0

            # Prefer concrete functions/methods over outer class wrappers
            if sym.kind == "class":
                score *= 0.35
            elif sym.kind == "constant":
                score *= 0.5
            direct_scores[idx] = score

        # 2-hop decayed callee / caller neighborhood propagation
        final_scores: Dict[int, float] = dict(direct_scores)
        name_to_indices: Dict[str, List[int]] = {}
        for idx, sym in enumerate(all_symbols):
            name_to_indices.setdefault(sym.name, []).append(idx)

        callers_map: Dict[str, List[int]] = {}
        for caller_idx, other_sym in enumerate(all_symbols):
            for c_name in other_sym.callees:
                callers_map.setdefault(c_name, []).append(caller_idx)

        for idx, sym in enumerate(all_symbols):
            base = direct_scores.get(idx, 0.0)
            if base <= 0:
                continue

            hop1_callees: Set[int] = set()
            hop1_callers: Set[int] = set()

            # Hop 1: callees get + base_score * 0.30
            for callee_name in sym.callees:
                for c_idx in name_to_indices.get(callee_name, []):
                    if c_idx != idx:
                        hop1_callees.add(c_idx)
                        final_scores[c_idx] = final_scores.get(c_idx, 0.0) + (base * 0.30)

            # Hop 1: callers get + base_score * 0.25
            for caller_idx in callers_map.get(sym.name, []):
                if caller_idx != idx:
                    hop1_callers.add(caller_idx)
                    final_scores[caller_idx] = final_scores.get(caller_idx, 0.0) + (base * 0.25)

            hop1_all = hop1_callees | hop1_callers

            # Hop 2: 2-hop callees get + base_score * 0.12, 2-hop callers get + base_score * 0.10
            hop2_seen_callees: Set[int] = set()
            hop2_seen_callers: Set[int] = set()

            for h_idx in hop1_all:
                h_sym = all_symbols[h_idx]
                for h_callee_name in h_sym.callees:
                    for c2_idx in name_to_indices.get(h_callee_name, []):
                        if c2_idx != idx and c2_idx not in hop1_all and c2_idx not in hop2_seen_callees:
                            hop2_seen_callees.add(c2_idx)
                            final_scores[c2_idx] = final_scores.get(c2_idx, 0.0) + (base * 0.12)

                for r2_idx in callers_map.get(h_sym.name, []):
                    if r2_idx != idx and r2_idx not in hop1_all and r2_idx not in hop2_seen_callers:
                        hop2_seen_callers.add(r2_idx)
                        final_scores[r2_idx] = final_scores.get(r2_idx, 0.0) + (base * 0.10)

        ranked_indices = sorted(
            range(len(all_symbols)),
            key=lambda i: (
                final_scores.get(i, 0.0),
                1 if all_symbols[i].kind not in ("class", "constant") else 0,
                -all_symbols[i].start_line,
            ),
            reverse=True,
        )

        max_chars = max(400, max_tokens * 4)
        body_budget_chars = int(max_chars * 0.68)

        sections: List[str] = [
            f"# SELECTIVE_PERFUSION Bundle (Query: {query})"
        ]
        used_chars = len(sections[0])
        files_included: List[str] = []
        symbols_included: List[str] = []

        # 1. Pack exact raw source bodies of top matching functions/methods
        zoom_blocks: List[str] = []
        for idx in ranked_indices:
            sym = all_symbols[idx]
            if sym.kind in ("class", "constant") and len(all_symbols) > 1:
                continue
            if final_scores.get(idx, 0.0) <= 0 and zoom_blocks:
                break

            qual_sym_name = f"{sym.parent_class}.{sym.name}" if sym.parent_class else sym.name
            zoom_str = self.zoom_symbol(sym.file_path, qual_sym_name, max_lines=90)
            if used_chars + len(zoom_str) + 2 > body_budget_chars and zoom_blocks:
                continue

            zoom_blocks.append(zoom_str)
            used_chars += len(zoom_str) + 2
            if sym.name not in symbols_included:
                symbols_included.append(sym.name)
            if sym.parent_class and qual_sym_name not in symbols_included:
                symbols_included.append(qual_sym_name)
            if sym.file_path not in files_included:
                files_included.append(sym.file_path)

            if len(zoom_blocks) >= 8:
                break

        if zoom_blocks:
            sections.append("## Zoomed Target Symbols (Raw Implementation)")
            sections.extend(zoom_blocks)

        # 2. Pack .pyi skeletons for neighboring context in included files
        skeleton_files = list(files_included) if files_included else [str(fp) for fp in candidate_files[:3]]
        skeleton_blocks: List[str] = []
        for fp_str in skeleton_files:
            skel = self.render_skeleton(fp_str)
            remaining = max_chars - used_chars - 40
            if remaining <= 120:
                break
            if len(skel) > remaining:
                skel = skel[:remaining] + "\n# ... [skeleton capped to max_tokens]"
            skeleton_blocks.append(skel)
            used_chars += len(skel) + 2
            if fp_str not in files_included:
                files_included.append(fp_str)

        if skeleton_blocks:
            sections.append("## Neighboring Context (.pyi Contract Skeletons)")
            sections.extend(skeleton_blocks)

        bundle_text = "\n\n".join(sections)
        if len(bundle_text) > max_chars:
            bundle_text = bundle_text[:max_chars]

        tokens_est = max(1, len(bundle_text) // 4)
        duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        return {
            "bundle_text": bundle_text,
            "tokens_est": tokens_est,
            "files_included": files_included,
            "symbols_included": symbols_included,
            "duration_ms": duration_ms,
        }

    # Alias for compatibility with Perfusion Bundle specifications
    build_perfusion_bundle = build_selective_bundle

    # -------------------------------------------------------------------------
    # Existing Repository Lookup Methods
    # -------------------------------------------------------------------------

    def ensure_fresh(
        self,
        root_dir: Optional[Union[str, Path]] = None,
        min_interval: float = 5.0,
    ) -> int:
        """Incrementally re-index files if min_interval seconds have elapsed since last check."""
        now = time.time()
        if now - self._last_check < min_interval:
            return 0
        self._last_check = now
        target_dir = root_dir or self.root_dir or Path.cwd()
        return self.index_repo(target_dir)

    def find_symbol(
        self,
        name: str,
        root_dir: Optional[Union[str, Path]] = None,
        auto_refresh: bool = True,
    ) -> List[SymbolRecord]:
        """Lookup exact symbol name across the codebase."""
        target_root = root_dir or self.root_dir
        if auto_refresh:
            try:
                self.ensure_fresh(target_root)
            except Exception:
                pass

        with self._get_connection() as conn:
            cursor = conn.cursor()
            if target_root:
                r_prefix = f"{Path(target_root).resolve()}%"
                cursor.execute(
                    "SELECT * FROM symbols WHERE name = ? AND file_path LIKE ? ORDER BY kind DESC LIMIT 10",
                    (name, r_prefix),
                )
                rows = cursor.fetchall()
                if rows:
                    return [self._row_to_record(r) for r in rows]

            cursor.execute(
                "SELECT * FROM symbols WHERE name = ? ORDER BY kind DESC LIMIT 10",
                (name,),
            )
            rows = cursor.fetchall()
            return [self._row_to_record(r) for r in rows]

    def search_symbols(
        self,
        query: str,
        limit: int = 15,
        root_dir: Optional[Union[str, Path]] = None,
        auto_refresh: bool = True,
    ) -> List[SymbolRecord]:
        """Fuzzy search symbol names, signatures, and docstrings matching query."""
        target_root = root_dir or self.root_dir
        if auto_refresh:
            try:
                self.ensure_fresh(target_root)
            except Exception:
                pass

        with self._get_connection() as conn:
            cursor = conn.cursor()
            like_q = f"%{query}%"
            if target_root:
                r_prefix = f"{Path(target_root).resolve()}%"
                cursor.execute(
                    """
                    SELECT * FROM symbols
                    WHERE file_path LIKE ? AND (name LIKE ? OR signature LIKE ? OR docstring LIKE ?)
                    ORDER BY CASE WHEN name LIKE ? THEN 0 ELSE 1 END, kind DESC
                    LIMIT ?
                    """,
                    (r_prefix, like_q, like_q, like_q, like_q, limit),
                )
                rows = cursor.fetchall()
                if rows:
                    return [self._row_to_record(r) for r in rows]

            cursor.execute(
                """
                SELECT * FROM symbols
                WHERE name LIKE ? OR signature LIKE ? OR docstring LIKE ?
                ORDER BY CASE WHEN name LIKE ? THEN 0 ELSE 1 END, kind DESC
                LIMIT ?
                """,
                (like_q, like_q, like_q, like_q, limit),
            )
            rows = cursor.fetchall()
            return [self._row_to_record(r) for r in rows]


def extract_batch_symbol_slices(
    specs: List[str],
    root_dir: Optional[Union[Path, str]] = None,
    max_lines: int = 120,
) -> Dict[str, Any]:
    """
    Multi-File Batch AST Slicer.
    Extracts multiple symbol definitions or bounded line ranges across files in <10ms.
    Each spec format:
      - 'path/to/file.py:SymbolName' (e.g. 'src/billing.py:compute_tax')
      - 'path/to/file.py:start-end' (e.g. 'src/billing.py:10-50')
    """
    root_path = Path(root_dir or Path.cwd()).resolve()
    results: Dict[str, Any] = {}
    total_lines = 0
    formatted_parts: List[str] = [f"### Multi-File Batch AST Slices ({len(specs)} requested)"]

    for spec in specs:
        spec_str = str(spec).strip()
        if not spec_str:
            continue

        if ":" not in spec_str:
            results[spec_str] = {
                "status": "error",
                "reason": "Invalid spec format. Expected 'file:Symbol' or 'file:start-end'",
            }
            continue

        file_part, target_part = spec_str.rsplit(":", 1)
        rel_path = Path(file_part)
        fpath = (root_path / rel_path).resolve() if not rel_path.is_absolute() else rel_path

        if not fpath.is_file():
            results[spec_str] = {
                "status": "error",
                "reason": f"File not found: {fpath}",
            }
            formatted_parts.append(f"\n❌ `{spec_str}`: File not found ({fpath})")
            continue

        try:
            raw_text = fpath.read_text(encoding="utf-8", errors="replace")
            lines = raw_text.splitlines()
        except Exception as exc:
            results[spec_str] = {
                "status": "error",
                "reason": f"Read error: {exc}",
            }
            continue

        # Case 1: Line range (e.g. "10-50")
        range_match = re.match(r"^(\d+)-(\d+)$", target_part)
        if range_match:
            s_line = max(1, int(range_match.group(1)))
            e_line = min(len(lines), int(range_match.group(2)))
            if e_line < s_line:
                e_line = s_line
            span = min(e_line - s_line + 1, max_lines)
            eff_end = s_line + span - 1
            slice_lines = lines[s_line - 1 : eff_end]
            slice_content = "\n".join(slice_lines)
            total_lines += len(slice_lines)

            results[spec_str] = {
                "status": "ok",
                "type": "range",
                "file": str(rel_path),
                "start_line": s_line,
                "end_line": eff_end,
                "line_count": len(slice_lines),
                "content": slice_content,
            }
            fence_lang = fence_language_for_path(fpath) or "python"
            formatted_parts.append(
                f"\n#### `{rel_path}` (L{s_line}-L{eff_end})\n```{fence_lang}\n{slice_content}\n```"
            )
            continue

        # Case 2: AST Symbol Name (e.g. "compute_tax" or "BillingService.compute_tax")
        sym_name = target_part.strip()
        found_slice: Optional[Tuple[str, int, int, str]] = None
        ext_low = fpath.suffix.lower()

        if ext_low == ".py":
            try:
                tree = ast.parse(raw_text, filename=str(fpath))
                candidates: List[Tuple[str, int, int]] = []
                for node in tree.body:
                    if isinstance(node, ast.ClassDef):
                        c_end = getattr(node, "end_lineno", node.lineno)
                        candidates.append((node.name, node.lineno, c_end))
                        for m in node.body:
                            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                m_end = getattr(m, "end_lineno", m.lineno)
                                candidates.append((f"{node.name}.{m.name}", m.lineno, m_end))
                                candidates.append((m.name, m.lineno, m_end))
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        f_end = getattr(node, "end_lineno", node.lineno)
                        candidates.append((node.name, node.lineno, f_end))

                # Exact match first, then case-insensitive
                for c_name, c_start, c_end in candidates:
                    if c_name == sym_name:
                        eff_end = min(c_end, c_start + max_lines - 1)
                        slice_lines = lines[c_start - 1 : eff_end]
                        found_slice = (c_name, c_start, eff_end, "\n".join(slice_lines))
                        break
                if not found_slice:
                    for c_name, c_start, c_end in candidates:
                        if c_name.lower() == sym_name.lower():
                            eff_end = min(c_end, c_start + max_lines - 1)
                            slice_lines = lines[c_start - 1 : eff_end]
                            found_slice = (c_name, c_start, eff_end, "\n".join(slice_lines))
                            break
            except Exception:
                pass
        elif ext_low in POLYGLOT_EXTENSIONS:
            poly_syms, _ = extract_polyglot_symbols(
                fpath, raw_text, prefer_treesitter=True, include_details=False
            )
            poly_candidates: List[Tuple[str, int, int]] = []
            for ps in poly_syms:
                if ps.parent_class:
                    poly_candidates.append((f"{ps.parent_class}.{ps.name}", ps.start_line, ps.end_line))
                    poly_candidates.append((ps.name, ps.start_line, ps.end_line))
                else:
                    poly_candidates.append((ps.name, ps.start_line, ps.end_line))

            for c_name, c_start, c_end in poly_candidates:
                if c_name == sym_name:
                    eff_end = min(c_end, c_start + max_lines - 1)
                    slice_lines = lines[c_start - 1 : eff_end]
                    found_slice = (c_name, c_start, eff_end, "\n".join(slice_lines))
                    break
            if not found_slice:
                for c_name, c_start, c_end in poly_candidates:
                    if c_name.lower() == sym_name.lower():
                        eff_end = min(c_end, c_start + max_lines - 1)
                        slice_lines = lines[c_start - 1 : eff_end]
                        found_slice = (c_name, c_start, eff_end, "\n".join(slice_lines))
                        break

        # Fallback to regex symbol search if AST did not resolve
        if not found_slice:
            pattern = re.compile(
                rf"^\s*(?:export\s+)?(?:pub\s+)?(?:async\s+)?(?:def|class|fn|func|function|struct|interface|trait)\s+({re.escape(sym_name)})\b",
                re.MULTILINE,
            )
            m = pattern.search(raw_text)
            if m:
                s_pos = m.start()
                s_line = raw_text[:s_pos].count("\n") + 1
                e_line = min(len(lines), s_line + max_lines - 1)
                slice_lines = lines[s_line - 1 : e_line]
                found_slice = (sym_name, s_line, e_line, "\n".join(slice_lines))

        if found_slice:
            s_name, s_start, s_end, s_content = found_slice
            total_lines += (s_end - s_start + 1)
            results[spec_str] = {
                "status": "ok",
                "type": "symbol",
                "symbol": s_name,
                "file": str(rel_path),
                "start_line": s_start,
                "end_line": s_end,
                "line_count": (s_end - s_start + 1),
                "content": s_content,
            }
            fence_lang = fence_language_for_path(fpath) or "python"
            formatted_parts.append(
                f"\n#### `{rel_path}:{s_name}` (L{s_start}-L{s_end})\n```{fence_lang}\n{s_content}\n```"
            )
        else:
            results[spec_str] = {
                "status": "not_found",
                "reason": f"Symbol '{sym_name}' not found in {rel_path}",
            }
            formatted_parts.append(f"\n⚠️ `{spec_str}`: Symbol '{sym_name}' not found in `{rel_path}`")

    return {
        "count": len(results),
        "total_lines": total_lines,
        "slices": results,
        "formatted": "\n".join(formatted_parts),
    }


# Module-level aliases for polyglot AST helpers
_compute_polyglot_end_line = LocalSymbolIndex._compute_polyglot_end_line
_extract_polyglot_callees = LocalSymbolIndex._extract_polyglot_callees
_extract_polyglot_invariants = LocalSymbolIndex._extract_polyglot_invariants

