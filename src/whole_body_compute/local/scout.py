"""
VoiceFi Recon Scout Engine & Sovereign Airlock (Phase 1).
Pre-digests massive files, logs, and stack traces on-device using Local Metal
Daemons (:9380 gemma4-26b, :9379 voicefi-coder-mlx), Ollama, LiteRT, or
sub-5ms deterministic AST/trace slicing when thermally throttled.
"""

# [direct-edit]
import ast
import asyncio
import json
import logging
import os
import re
import sqlite3
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from whole_body_compute.local.engine import LocalModelEngine, is_litert_available
from whole_body_compute.local.supervisor import (
    HardwareTelemetry,
    ThermalSupervisor,
    default_supervisor,
)
from whole_body_compute.local.symbols import (
    LocalSymbolIndex,
    SymbolRecord,
    _tokenize_query,
)
from whole_body_compute.local.trace_parser import (
    extract_error_line,
    extract_js_frames,
    extract_python_frames,
    parse_traceback,
)

logger = logging.getLogger("whole_body_compute.local.scout")

# Ensure ThermalSupervisor.check_health().should_throttle contract is always available
if not hasattr(HardwareTelemetry, "should_throttle"):
    HardwareTelemetry.should_throttle = property(lambda self: not self.is_safe)  # type: ignore[attr-defined]

if not hasattr(ThermalSupervisor, "check_health"):
    ThermalSupervisor.check_health = ThermalSupervisor.get_telemetry  # type: ignore[attr-defined]


@dataclass
class ScoutResult:
    target: str
    query: str
    findings: str
    input_tokens_est: int
    output_tokens: int
    tokens_saved: int
    savings_pct: float
    duration_seconds: float
    model_name: str
    is_local: bool
    read_latency_ms: float = 0.0
    ingress_mode: str = "unified_ram"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def estimate_tokens(text: str) -> int:
    """Rough estimation of token count (~4 characters per token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


_DAEMON_CIRCUIT_BREAKER: Dict[str, float] = {}
_CIRCUIT_BREAKER_FILE = Path("/tmp/wbc_metal_circuit_breaker.json")
_CIRCUIT_BREAKER_LOCK = threading.Lock()
CIRCUIT_BREAKER_TTL: float = 60.0


def reset_daemon_circuit_breaker() -> None:
    """Reset the daemon circuit breaker state in memory and on disk (for tests). # [direct-edit]"""
    global _DAEMON_CIRCUIT_BREAKER
    with _CIRCUIT_BREAKER_LOCK:
        _DAEMON_CIRCUIT_BREAKER.clear()
        try:
            if _CIRCUIT_BREAKER_FILE.exists():
                _CIRCUIT_BREAKER_FILE.unlink(missing_ok=True)
        except Exception:
            pass


def _is_daemon_circuit_open(model_id: str) -> bool:
    """Return True if daemon tripped circuit breaker within the last 60 seconds."""
    now = time.time()
    with _CIRCUIT_BREAKER_LOCK:
        ts = _DAEMON_CIRCUIT_BREAKER.get(model_id)
        if ts and (now - ts) < CIRCUIT_BREAKER_TTL:
            return True
        try:
            if _CIRCUIT_BREAKER_FILE.is_file():
                data = json.loads(_CIRCUIT_BREAKER_FILE.read_text(encoding="utf-8"))
                disk_ts = float(data.get(model_id, 0.0))
                if (now - disk_ts) < CIRCUIT_BREAKER_TTL:
                    _DAEMON_CIRCUIT_BREAKER[model_id] = disk_ts
                    return True
        except Exception:
            pass
    return False


def _trip_daemon_circuit_breaker(model_id: str) -> None:
    """Trip circuit breaker for model_id for 60 seconds with atomic disk persistence."""
    now = time.time()
    with _CIRCUIT_BREAKER_LOCK:
        _DAEMON_CIRCUIT_BREAKER[model_id] = now
        try:
            data = {}
            if _CIRCUIT_BREAKER_FILE.is_file():
                try:
                    data = json.loads(_CIRCUIT_BREAKER_FILE.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
            data[model_id] = now
            _tmp = Path(f"/tmp/wbc_cb_{os.getpid()}_{time.time_ns()}.tmp")
            _tmp.write_text(json.dumps(data), encoding="utf-8")
            os.replace(str(_tmp), str(_CIRCUIT_BREAKER_FILE))
        except Exception:
            pass


def _clear_daemon_circuit_breaker(model_id: str) -> None:
    """Clear tripped circuit breaker on successful response with atomic disk persistence."""
    with _CIRCUIT_BREAKER_LOCK:
        _DAEMON_CIRCUIT_BREAKER.pop(model_id, None)
        try:
            if _CIRCUIT_BREAKER_FILE.is_file():
                try:
                    data = json.loads(_CIRCUIT_BREAKER_FILE.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
                if model_id in data:
                    del data[model_id]
                    _tmp = Path(f"/tmp/wbc_cb_{os.getpid()}_{time.time_ns()}.tmp")
                    _tmp.write_text(json.dumps(data), encoding="utf-8")
                    os.replace(str(_tmp), str(_CIRCUIT_BREAKER_FILE))
        except Exception:
            pass


def _connect_scout_db(db_path: Path, timeout: float = 10.0) -> sqlite3.Connection:
    """Connect to SQLite with WAL mode and robust busy timeout for concurrent dogpiling."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=timeout)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=10000;")
    except Exception:
        pass
    return conn


class ReconScout:
    """
    On-device scout that reads large local files/logs and extracts concise,
    actionable insights for Antigravity and Claude Code with zero Donut-Hole truncation.
    """

    DEFAULT_SCOUT_INSTRUCTIONS = (
        "You are the VoiceFi Recon Scout. You run on-device to pre-digest raw files, logs, and code.\n"
        "Your mission is to find the exact root cause, key symbols, or answers requested.\n"
        "Be extremely concise, technical, and precise. Avoid fluff. Include relevant line numbers and code snippets."
    )

    CODE_EXTENSIONS = {
        ".py",
        ".pyi",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".sh",
        ".bash",
        ".zsh",
        ".yaml",
        ".yml",
        ".rs",
        ".go",
        ".swift",
    }
    LOG_EXTENSIONS = {".log", ".err", ".out", ".jsonl", ".trace"}

    def __init__(
        self,
        engine: Optional[LocalModelEngine] = None,
        symbol_index: Optional[LocalSymbolIndex] = None,
    ):
        self.engine = engine or LocalModelEngine()
        self.symbol_index = symbol_index or LocalSymbolIndex()

    def read_target_content(
        self,
        target_path: Union[str, Path],
        max_bytes: int = 500_000,
    ) -> tuple[str, int, bool]:
        """Read target file content safely up to max_bytes (reading head+tail if file exceeds max_bytes)."""
        path = Path(os.path.expanduser(str(target_path)))
        if not path.exists():
            raise FileNotFoundError(f"Target path does not exist: {path}")

        if path.is_file():
            size = path.stat().st_size
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                if size <= max_bytes:
                    content = f.read()
                    truncated = False
                else:
                    head_bytes = max_bytes // 2
                    tail_bytes = max_bytes - head_bytes
                    head = f.read(head_bytes)
                    try:
                        f.seek(max(0, size - tail_bytes))
                        tail = f.read(tail_bytes)
                        content = head + "\n... [oversized file middle skipped at byte level] ...\n" + tail
                    except Exception:
                        content = head
                    truncated = True
            return content, size, truncated
        elif path.is_dir():
            content_parts = [f"Directory listing for: {path}"]
            total_size = 0
            file_count = 0
            ignore_dirs = {
                ".git",
                ".venv",
                "__pycache__",
                "node_modules",
                ".gemini",
                ".pytest_cache",
                ".ruff_cache",
                ".mypy_cache",
                "garageband_session",
                "parody_takes",
                "dist",
                "build",
                ".eggs",
                ".tox",
                "site-packages",
            }
            truncated = False
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith(".")]
                for file in files:
                    if file.startswith(".") or file.endswith(
                        (
                            ".pyc",
                            ".png",
                            ".jpg",
                            ".jpeg",
                            ".wav",
                            ".mp3",
                            ".dmg",
                            ".dylib",
                            ".so",
                            ".DS_Store",
                        )
                    ):
                        continue
                    fp = Path(root) / file
                    try:
                        sz = fp.stat().st_size
                        total_size += sz
                        file_count += 1
                        content_parts.append(f" - {fp.relative_to(path)} ({sz} bytes)")
                        if file_count >= 60:
                            truncated = True
                            break
                    except Exception:
                        pass
                if truncated:
                    content_parts.append(" ... [listing capped at 60 files for fast local context]")
                    break
            return "\n".join(content_parts), total_size, truncated
        else:
            raise ValueError(f"Unsupported path type: {path}")

    def _is_log_or_trace(self, content: str, target_path: str) -> bool:
        ext = Path(target_path).suffix.lower()
        if ext in self.LOG_EXTENSIONS:
            return True
        if ext in self.CODE_EXTENSIONS:
            return False
        if "Traceback (most recent call last):" in content:
            return True
        if re.search(r"^(?:\d{4}-\d{2}-\d{2}|\[\d+|\w+\s+\d+\s+\d{2}:\d{2}:\d{2}).*\b(?:ERROR|CRITICAL|EXCEPTION|FATAL)\b", content, re.MULTILINE):
            return True
        return False

    def _render_in_memory_skeleton(
        self,
        content: str,
        target_path: str,
        symbols: List[SymbolRecord],
    ) -> str:
        """Render a deterministic .pyi skeleton directly from in-memory content & symbols."""
        p = Path(target_path)
        if p.is_file():
            try:
                return self.symbol_index.render_skeleton(p)
            except Exception:
                pass

        sym_memories: Dict[str, List[str]] = {}
        try:
            from whole_body_compute.local.dot_memory import get_dot_store

            sym_memories = get_dot_store().get_symbol_memories(
                str(target_path), [s.name for s in symbols]
            )
        except Exception:
            sym_memories = {}

        out_lines = [f"# .pyi Airlock Skeleton: {p.name}"]
        for file_lesson in sym_memories.get("__file__", []):
            out_lines.append(f"# memory [file]: {file_lesson}")
        for s in symbols:
            if s.kind == "constant":
                out_lines.append(f"{s.signature or s.name}  # [L{s.start_line}-L{s.end_line}]")
            elif s.kind == "class":
                out_lines.append(f"\n{s.signature or ('class ' + s.name)}:  # [L{s.start_line}-L{s.end_line}]")
                if s.docstring:
                    out_lines.append(f'    """{s.docstring}"""')
                for cls_l in sym_memories.get(s.name, []):
                    out_lines.append(f"    # memory: {cls_l}")
            elif s.parent_class:
                out_lines.append(f"    {s.signature or ('def ' + s.name)}: ...  # [L{s.start_line}-L{s.end_line}]")
                if s.callees:
                    out_lines.append(f"        # callees: {', '.join(s.callees)}")
                if s.invariants:
                    out_lines.append(f"        # invariants: {', '.join(s.invariants)}")
                qual_s = f"{s.parent_class}.{s.name}"
                m_lessons: List[str] = []
                for l_str in sym_memories.get(qual_s, []) + sym_memories.get(s.name, []):
                    if l_str not in m_lessons:
                        m_lessons.append(l_str)
                for m_l in m_lessons:
                    out_lines.append(f"        # memory: {m_l}")
            else:
                out_lines.append(f"\n{s.signature or ('def ' + s.name)}: ...  # [L{s.start_line}-L{s.end_line}]")
                if s.docstring:
                    out_lines.append(f'    """{s.docstring}"""')
                if s.callees:
                    out_lines.append(f"    # callees: {', '.join(s.callees)}")
                if s.invariants:
                    out_lines.append(f"    # invariants: {', '.join(s.invariants)}")
                for fn_l in sym_memories.get(s.name, []):
                    out_lines.append(f"    # memory: {fn_l}")
        return "\n".join(out_lines)

    def select_deterministic_chunks(
        self,
        content: str,
        target_path: str,
        query: str,
        max_chars: int = 16_000,
        symbols: Optional[List[SymbolRecord]] = None,
    ) -> tuple[str, bool, List[str]]:
        """
        Eliminate the 'Donut Hole' truncation by deterministically selecting high-signal chunks:
        - For Python/code files: parse AST (or function blocks), score every AST node (class/def)
          against query tokens + callees, and pack the highest-scoring complete AST functions
          (with exact line numbers Lstart-Lend) plus the file's .pyi contract header so functions
          in the middle of a 2,000-line file are 100% visible.
        - For log/trace files: use trace_parser + reverse-tail sliding-window deduplication
          across the entire file to extract unique stack frames and error windows [-20, +10].
        """
        if not content:
            return "", False, []

        if self._is_log_or_trace(content, target_path):
            return self._select_log_chunks(content, target_path, query, max_chars=max_chars)

        return self._select_code_chunks(
            content, target_path, query, max_chars=max_chars, symbols=symbols
        )

    def _select_code_chunks(
        self,
        content: str,
        target_path: str,
        query: str,
        max_chars: int = 16_000,
        symbols: Optional[List[SymbolRecord]] = None,
    ) -> tuple[str, bool, List[str]]:
        p = Path(target_path)
        ext = p.suffix.lower()
        lines = content.splitlines()
        total_lines = len(lines)

        if symbols is None:
            symbols = []
            if ext == ".py" or not ext:
                try:
                    tree = ast.parse(content, filename=str(p))
                    symbols = self.symbol_index._extract_ast_symbols(p, tree)
                except SyntaxError as exc:
                    lineno = exc.lineno or 1
                    start_ln = max(1, lineno - 25)
                    end_ln = min(total_lines, lineno + 25)
                    window_lines = [
                        f"# [SYNTAX ERROR FAIL-OPEN WINDOW] {p.name}",
                        f"# SyntaxError at line {lineno}: {exc.msg}",
                        f"# Showing raw lines [L{start_ln}-L{end_ln}] around line {lineno}:",
                    ]
                    for idx in range(start_ln, end_ln + 1):
                        marker = ">>> " if idx == lineno else "    "
                        window_lines.append(f"{marker}L{idx}: {lines[idx - 1]}")
                    text = "\n".join(window_lines)
                    return text[:max_chars], True, [f"SyntaxError@L{lineno}"]
                except Exception:
                    symbols = []
            else:
                symbols = self.symbol_index._parse_regex_fallback(p, content)

        if not symbols:
            if len(content) <= max_chars:
                return content, False, []
            half = max_chars // 2 - 100
            return (
                content[:half] + "\n... [truncated] ...\n" + content[-half:],
                True,
                [],
            )

        query_tokens = _tokenize_query(query)
        direct_scores: Dict[int, float] = {}

        for idx, sym in enumerate(symbols):
            score = 0.0
            name_low = sym.name.lower()
            name_parts = set(name_low.split("_"))
            sig_low = sym.signature.lower()
            doc_low = sym.docstring.lower()
            inv_low = " ".join(sym.invariants).lower()
            callees_low = {c.lower() for c in sym.callees}

            s_start = max(1, sym.start_line)
            s_end = min(total_lines, max(s_start, sym.end_line))
            body_text = "\n".join(lines[s_start - 1 : s_end]).lower()

            for tok in query_tokens:
                if tok == name_low or tok in name_parts:
                    score += 15.0
                elif tok in name_low:
                    score += 8.0
                if sym.parent_class and tok in sym.parent_class.lower():
                    score += 3.5
                if tok in sig_low:
                    score += 4.0
                if tok in doc_low:
                    score += 3.5
                if tok in callees_low:
                    score += 3.0
                if tok in inv_low:
                    score += 2.5
                if tok in body_text:
                    score += 2.5

            if sym.kind == "class":
                score *= 0.3
            elif sym.kind == "constant":
                score *= 0.25
            direct_scores[idx] = score

        # 2-hop decayed callee / caller neighborhood propagation
        final_scores: Dict[int, float] = dict(direct_scores)
        name_map: Dict[str, List[int]] = {}
        for idx, sym in enumerate(symbols):
            name_map.setdefault(sym.name, []).append(idx)

        callers_map: Dict[str, List[int]] = {}
        for caller_idx, other in enumerate(symbols):
            for c_name in other.callees:
                callers_map.setdefault(c_name, []).append(caller_idx)

        for idx, sym in enumerate(symbols):
            base = direct_scores.get(idx, 0.0)
            if base <= 0:
                continue

            hop1_callees: Set[int] = set()
            hop1_callers: Set[int] = set()

            # Hop 1: Boost callees (+0.30)
            for callee in sym.callees:
                for c_idx in name_map.get(callee, []):
                    if c_idx != idx:
                        hop1_callees.add(c_idx)
                        final_scores[c_idx] = final_scores.get(c_idx, 0.0) + (base * 0.30)

            # Hop 1: Boost callers (+0.25)
            for caller_idx in callers_map.get(sym.name, []):
                if caller_idx != idx:
                    hop1_callers.add(caller_idx)
                    final_scores[caller_idx] = final_scores.get(caller_idx, 0.0) + (base * 0.25)

            hop1_all = hop1_callees | hop1_callers

            # Hop 2: 2-hop callees (+0.12), 2-hop callers (+0.10)
            hop2_seen_callees: Set[int] = set()
            hop2_seen_callers: Set[int] = set()

            for h_idx in hop1_all:
                h_sym = symbols[h_idx]
                for h_callee in h_sym.callees:
                    for c2_idx in name_map.get(h_callee, []):
                        if c2_idx != idx and c2_idx not in hop1_all and c2_idx not in hop2_seen_callees:
                            hop2_seen_callees.add(c2_idx)
                            final_scores[c2_idx] = final_scores.get(c2_idx, 0.0) + (base * 0.12)

                for r2_idx in callers_map.get(h_sym.name, []):
                    if r2_idx != idx and r2_idx not in hop1_all and r2_idx not in hop2_seen_callers:
                        hop2_seen_callers.add(r2_idx)
                        final_scores[r2_idx] = final_scores.get(r2_idx, 0.0) + (base * 0.10)

        ranked = sorted(
            range(len(symbols)),
            key=lambda i: (
                final_scores.get(i, 0.0),
                1 if symbols[i].kind not in ("class", "constant") else 0,
                -symbols[i].start_line,
            ),
            reverse=True,
        )

        # Build .pyi contract header (capped at ~35% of max_chars so zoomed AST functions have priority)
        skeleton = self._render_in_memory_skeleton(content, target_path, symbols)
        skel_budget = max(600, int(max_chars * 0.35))
        if len(skeleton) > skel_budget:
            skeleton = skeleton[:skel_budget] + "\n# ... [.pyi skeleton condensed]"

        sections: List[str] = [
            "### .pyi Contract Header",
            skeleton,
            "### Highest-Scoring AST Slices",
        ]
        used_chars = sum(len(s) + 2 for s in sections)
        included_symbols: List[str] = []

        func_candidates = [
            i for i in ranked if symbols[i].kind not in ("class", "constant")
        ] or ranked

        for idx in func_candidates:
            sym = symbols[idx]
            s_start = max(1, sym.start_line)
            s_end = min(total_lines, max(s_start, sym.end_line))
            qual_name = f"{sym.parent_class}.{sym.name}" if sym.parent_class else sym.name

            block_lines = [
                f"# AST Symbol: {qual_name} [L{s_start}-L{s_end}] (score={final_scores.get(idx, 0.0):.1f})"
            ]
            for ln in range(s_start, s_end + 1):
                block_lines.append(f"L{ln}: {lines[ln - 1]}")
            block_str = "\n".join(block_lines)

            if used_chars + len(block_str) + 2 > max_chars:
                if not included_symbols:
                    # Ensure at least the top-1 matching function is included
                    avail = max(200, max_chars - used_chars - 40)
                    sections.append(block_str[:avail])
                    included_symbols.append(sym.name)
                break

            sections.append(block_str)
            used_chars += len(block_str) + 2
            included_symbols.append(sym.name)

            # Stop once we have packed the relevant matches if remaining symbols have 0 score
            if len(included_symbols) >= 6 and final_scores.get(idx, 0.0) <= 0:
                break

        chunked = "\n\n".join(sections)
        was_truncated = len(content) > max_chars or len(included_symbols) < len(func_candidates)
        return chunked[:max_chars], was_truncated, included_symbols

    def _select_log_chunks(
        self,
        content: str,
        target_path: str,
        query: str,
        max_chars: int = 16_000,
    ) -> tuple[str, bool, List[str]]:
        """
        Use trace_parser + reverse-tail sliding-window deduplication across the entire
        file to extract unique stack frames and error windows [-20, +10].
        """
        lines = content.splitlines()
        total_lines = len(lines)
        parsed = parse_traceback(content, fallback_target=target_path)
        py_frames = extract_python_frames(content)
        js_frames = extract_js_frames(content)
        err_type, err_msg = extract_error_line(content)

        markers: List[str] = []
        header_lines = [
            f"### Trace & Log Digest for `{Path(target_path).name}` ({total_lines} lines)"
        ]
        if err_type or parsed.error_type:
            et = err_type or parsed.error_type
            em = err_msg or parsed.error_message or ""
            header_lines.append(f"- Primary Error: `{et}: {em}`")
            markers.append(f"{et}: {em}"[:80])
        if parsed.target_file:
            loc = f"{parsed.target_file}:{parsed.line_number}" if parsed.line_number else parsed.target_file
            header_lines.append(f"- Deepest User Target: `{loc}`")
            markers.append(loc)
        if parsed.instruction:
            header_lines.append(f"- Actionable Instruction: {parsed.instruction}")

        # Unique stack frames (reverse order from tail)
        unique_frames: List[str] = []
        seen_frames: Set[Tuple[str, int]] = set()
        for fp, ln in reversed(py_frames + js_frames):
            if (fp, ln) not in seen_frames:
                seen_frames.add((fp, ln))
                unique_frames.append(f"{fp}:L{ln}")
                if f"{Path(fp).name}:L{ln}" not in markers:
                    markers.append(f"{Path(fp).name}:L{ln}")
                if len(unique_frames) >= 12:
                    break
        if unique_frames:
            header_lines.append("- Unique Stack Frames (tail-first): " + ", ".join(unique_frames))

        # Extract structured incident identifiers & verbatim high-severity error lines (<2ms pre-filter)
        incident_meta: Dict[str, str] = {}
        verbatim_errors: List[str] = []
        seen_verbatim_sigs: Set[str] = set()
        high_sev_markers = (
            "[error]", " error ", "error:", "[critical]", "[fatal]", "panic",
            "zerodivisionerror", "valueerror", "typeerror", "keyerror", "runtimeerror",
            "attributeerror", "connectionrefusederror", "permissionerror", "assertionerror",
            "unhandled exception",
        )
        for idx_ln, raw_l in enumerate(lines):
            low_l = raw_l.lower()
            if any(m in low_l for m in high_sev_markers):
                if not incident_meta.get("request_id"):
                    req_m = re.search(r"\b(req_[a-zA-Z0-9_-]+)\b", raw_l) or re.search(r"request_id[=:]\s*([a-zA-Z0-9_-]+)", raw_l)
                    if req_m:
                        incident_meta["request_id"] = req_m.group(1)
                if not incident_meta.get("account_id"):
                    acct_m = re.search(r"\b(acct_[a-zA-Z0-9_-]+)\b", raw_l) or re.search(r"account(?:_id)?[=:]\s*([a-zA-Z0-9_-]+)", raw_l)
                    if acct_m:
                        incident_meta["account_id"] = acct_m.group(1)
                if not incident_meta.get("exception"):
                    exc_m = re.search(r"\b([A-Za-z0-9_]*(?:Error|Exception))\b", raw_l)
                    if exc_m:
                        incident_meta["exception"] = exc_m.group(1)
                if not incident_meta.get("function"):
                    fn_m = re.search(r"\bin\s+([A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+)\b", raw_l)
                    if fn_m:
                        incident_meta["function"] = fn_m.group(1)
                if not incident_meta.get("location"):
                    loc_m = re.search(r"\b([A-Za-z0-9_./-]+\.py:\d+)\b", raw_l)
                    if loc_m:
                        incident_meta["location"] = loc_m.group(1)

                v_sig = re.sub(r"0x[0-9a-fA-F]+|\d+", "#", low_l.strip())[:100]
                if v_sig not in seen_verbatim_sigs and len(verbatim_errors) < 8:
                    seen_verbatim_sigs.add(v_sig)
                    verbatim_errors.append(f"L{idx_ln + 1}: {raw_l.strip()}")

        if incident_meta:
            header_lines.append("- Extracted Incident Metadata: `" + json.dumps(incident_meta) + "`")
        if verbatim_errors:
            header_lines.append("### Verbatim High-Severity Error Lines")
            header_lines.extend(verbatim_errors)

        # Reverse-tail sliding-window deduplication across the ENTIRE file (2-tier severity priority)
        query_tokens = [
            t for t in _tokenize_query(query)
            if t not in {"find", "line", "lines", "extract", "type", "name", "function", "file", "log", "show", "get"}
        ]
        error_keywords = ("traceback", "error", "exception", "critical", "fatal", "panic", "failed", "assert")
        seen_signatures: Set[str] = set()
        windows: List[Tuple[int, int, str]] = []  # (start_ln, end_ln, label)
        covered_lines: Set[int] = set()

        # Pass 1: High-severity error/traceback lines first; Pass 2: Query-token matches
        for pass_num in (1, 2):
            if len(windows) >= 10:
                break
            for idx in range(total_lines - 1, -1, -1):
                line = lines[idx]
                low = line.lower()
                is_err = any(k in low for k in error_keywords)
                is_q = any(t in low for t in query_tokens) if query_tokens else False
                if pass_num == 1 and not is_err:
                    continue
                if pass_num == 2 and (is_err or not is_q):
                    continue

                ln_num = idx + 1
                if ln_num in covered_lines:
                    continue

                # Normalize line to deduplicate repeated identical log entries (including prefixed IDs like req_88201)
                norm_sig = re.sub(r"0x[0-9a-fA-F]+|\d+", "#", low.strip())[:100]
                if norm_sig in seen_signatures:
                    continue
                seen_signatures.add(norm_sig)

                w_start = max(1, ln_num - 20)
                w_end = min(total_lines, ln_num + 10)
                for covered in range(w_start, w_end + 1):
                    covered_lines.add(covered)
                    cov_sig = re.sub(
                        r"0x[0-9a-fA-F]+|\d+", "#", lines[covered - 1].lower().strip()
                    )[:100]
                    seen_signatures.add(cov_sig)

                label = line.strip()[:100]
                windows.append((w_start, w_end, label))
                if label not in markers and len(markers) < 15:
                    markers.append(label)

                if len(windows) >= 10:
                    break

        sections = ["\n".join(header_lines)]
        used_chars = len(sections[0])

        if windows:
            sections.append("### Deduplicated Error Windows [-20, +10] (Reverse-Tail Order)")
            used_chars += len(sections[-1]) + 2
            for w_start, w_end, label in windows:
                win_lines = [f"# Window [L{w_start}-L{w_end}]: {label}"]
                prev_sig = None
                repeat_count = 0
                for ln in range(w_start, w_end + 1):
                    raw_ln = lines[ln - 1]
                    cur_sig = re.sub(r"0x[0-9a-fA-F]+|\d+", "#", raw_ln.lower().strip())[:100]
                    if cur_sig and cur_sig == prev_sig:
                        repeat_count += 1
                        if repeat_count <= 2:
                            win_lines.append(f"L{ln}: {raw_ln}")
                        elif repeat_count == 3:
                            win_lines.append("    ... [repeated similar log lines collapsed] ...")
                    else:
                        prev_sig = cur_sig
                        repeat_count = 1
                        win_lines.append(f"L{ln}: {raw_ln}")
                win_block = "\n".join(win_lines)
                if used_chars + len(win_block) + 2 > max_chars:
                    break
                sections.append(win_block)
                used_chars += len(win_block) + 2
        else:
            # No error keywords found; include tail window of log
            tail_start = max(1, total_lines - 60)
            tail_lines = [f"# Log Tail [L{tail_start}-L{total_lines}]"]
            for ln in range(tail_start, total_lines + 1):
                tail_lines.append(f"L{ln}: {lines[ln - 1]}")
            sections.append("\n".join(tail_lines))

        out_text = "\n\n".join(sections)
        was_truncated = len(content) > max_chars or len(out_text) < len(content)
        return out_text[:max_chars], was_truncated, markers

    def _is_thermally_throttled(self) -> bool:
        """Check if ThermalSupervisor reports active throttling via check_health().should_throttle."""
        try:
            health = default_supervisor.check_health()
            if isinstance(health, dict):
                if "should_throttle" in health:
                    return bool(health["should_throttle"])
                return not bool(health.get("healthy", health.get("is_safe", True)))
            return bool(getattr(health, "should_throttle", not getattr(health, "is_safe", True)))
        except Exception:
            return False

    def _find_similar_files(self, missing_path: Union[str, Path], max_matches: int = 5) -> List[str]:
        """Search workspace for files with similar names or stem tokens when a target file is not found."""
        p = Path(str(missing_path))
        stem_tokens = [
            t.lower()
            for t in re.split(r"[_\-\./\s]+", p.stem)
            if len(t) >= 3 and t.lower() not in {"src", "whole", "body", "compute", "test", "tests", "local"}
        ]
        target_name_lower = p.name.lower()
        scored: List[Tuple[int, str]] = []
        ignore_dirs = {
            ".git", ".venv", "__pycache__", "node_modules", ".gemini",
            ".pytest_cache", ".ruff_cache", ".mypy_cache", "dist", "build",
        }
        search_root = Path.cwd()
        try:
            for root, dirs, files in os.walk(search_root):
                dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith(".")]
                for f in files:
                    if f.endswith((".pyc", ".DS_Store")):
                        continue
                    fl = f.lower()
                    rel = str(Path(root, f).relative_to(search_root))
                    score = 0
                    if fl == target_name_lower:
                        score += 100
                    elif p.stem.lower() and p.stem.lower() in fl:
                        score += 50
                    for tok in stem_tokens:
                        if tok in rel.lower():
                            score += 15
                    if score > 0:
                        scored.append((score, rel))
        except Exception:
            pass
        scored.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
        return [rel for _, rel in scored[:max_matches]]

    @staticmethod
    def _is_port_open(host: str, port: int, timeout: float = 0.025) -> bool:
        """Fast <25ms TCP socket check before issuing HTTP requests to local daemons."""
        import socket

        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except Exception:
            return False

    def _query_local_metal_daemons(
        self,
        prompt: str,
        system_prompt: str,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Query Local Metal Daemons on :9379 (voicefi-coder-mlx) first for fast ~900ms response (2.2s timeout, 220 tokens),
        then :9380 (gemma4-26b) with a 0.8s timeout before falling back to Ollama.
        # [direct-edit]
        """
        daemons = [
            ("voicefi-coder-mlx", 9379, "http://127.0.0.1:9379/v1/chat/completions", 2.2, 220),
            ("gemma4-26b", 9380, "http://127.0.0.1:9380/v1/chat/completions", 0.8, 220),
        ]
        for model_id, port, endpoint, timeout_sec, max_toks in daemons:
            if _is_daemon_circuit_open(model_id):
                continue
            if not self._is_port_open("127.0.0.1", port, timeout=0.025):
                continue
            payload = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": max_toks,
                "stream": False,
            }
            try:
                req = urllib.request.Request(
                    endpoint,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8", errors="replace"))
                        choices = data.get("choices", [])
                        if choices:
                            msg = choices[0].get("message", {})
                            content = (msg.get("content") or "").strip()
                            if content:
                                _clear_daemon_circuit_breaker(model_id)
                                return model_id, content
            except Exception as e:
                _trip_daemon_circuit_breaker(model_id)
                logger.debug(f"Local Metal daemon {model_id} ({endpoint}) tripped circuit breaker: {e}")

        return None, None

    async def scout(
        self,
        target_path: Union[str, Path],
        query: str = "Analyze this file, identify any errors or anomalies, and extract key functions/logic.",
        max_bytes: int = 500_000,
        record_shield: bool = True,
    ) -> ScoutResult:
        """
        Execute an on-device scout run on the target file/directory.
        """
        start_time = time.perf_counter()

        def _record_and_return(result: ScoutResult) -> ScoutResult:
            try:
                from whole_body_compute.local.telemetry import record_hybrid_step

                record_hybrid_step(
                    stage="local_scout",
                    model_name=result.model_name,
                    is_local=result.is_local,
                    duration_ms=int(result.duration_seconds * 1000),
                    input_tokens=result.input_tokens_est,
                    output_tokens=result.output_tokens,
                    tokens_saved=result.tokens_saved,
                    savings_pct=result.savings_pct,
                    ingress_latency_ms=result.read_latency_ms,
                    success=(result.error is None),
                    target=str(target_path),
                )
            except Exception:
                pass
            try:
                is_speculative_bg = os.environ.get("WBC_SPECULATIVE_PRESCOUT") == "1"
                if (
                    record_shield
                    and not is_speculative_bg
                    and result.error is None
                    and result.input_tokens_est > 0
                ):
                    from whole_body_compute.shield import (
                        get_harness_file_token_ceiling,
                        record_shield_event,
                    )

                    harness_ceiling = get_harness_file_token_ceiling()
                    capped_raw = min(harness_ceiling, max(1, int(result.input_tokens_est)))
                    comp_out = max(1, int(result.output_tokens))
                    capped_saved = max(0, capped_raw - comp_out)
                    if capped_saved > 0:
                        record_shield_event(
                            event_type="local_metal_scout",
                            target=str(target_path),
                            raw_tokens_est=capped_raw,
                            compressed_tokens_est=comp_out,
                            tokens_saved=capped_saved,
                            secrets_blocked=0,
                            latency_ms=round(float(result.duration_seconds) * 1000.0, 2),
                            action_mode="warm_cache" if result.ingress_mode == "warm_digest_cache" else "local_exec",
                        )  # [direct-edit]
            except Exception:
                pass
            try:
                from whole_body_compute.local.supervisor import default_supervisor

                default_supervisor.maybe_opportunistic_reap()
            except Exception:
                pass
            return result

        read_start = time.perf_counter()
        try:
            raw_content, file_size, truncated = self.read_target_content(
                target_path, max_bytes=max_bytes
            )
            read_latency_ms = round((time.perf_counter() - read_start) * 1000.0, 3)
        except FileNotFoundError:
            duration = time.perf_counter() - start_time
            suggestions = self._find_similar_files(target_path)
            err_msg = f"[Error] File not found: {target_path} (Target path does not exist)"
            findings_lines = [err_msg]
            if suggestions:
                findings_lines.append("Did you mean one of these existing files?")
                for s in suggestions:
                    findings_lines.append(f"  - {s}")
            findings_text = "\n".join(findings_lines)
            return _record_and_return(
                ScoutResult(
                    target=str(target_path),
                    query=query,
                    findings=findings_text,
                    input_tokens_est=0,
                    output_tokens=estimate_tokens(findings_text),
                    tokens_saved=0,
                    savings_pct=0.0,
                    duration_seconds=round(duration, 3),
                    model_name=self.engine.model_name,
                    is_local=False,
                    read_latency_ms=0.0,
                    error=err_msg,
                )
            )
        except Exception as e:
            duration = time.perf_counter() - start_time
            err_msg = f"[Error] {e}"
            return _record_and_return(
                ScoutResult(
                    target=str(target_path),
                    query=query,
                    findings=err_msg,
                    input_tokens_est=0,
                    output_tokens=estimate_tokens(err_msg),
                    tokens_saved=0,
                    savings_pct=0.0,
                    duration_seconds=round(duration, 3),
                    model_name=self.engine.model_name,
                    is_local=False,
                    read_latency_ms=0.0,
                    error=err_msg,
                )
            )

        raw_input_tokens = estimate_tokens(raw_content)
        target_suffix = Path(str(target_path)).suffix.lower()

        # Tiny-File Instant Bypass (<1ms vs 2.3s): never invoke an LLM on small data/config files (<= 400 tokens)
        if (
            raw_content.strip()
            and raw_input_tokens <= 400
            and target_suffix in {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env", ".csv", ".txt", ".md"}
            and not self._is_log_or_trace(raw_content, str(target_path))
            and os.environ.get("WBC_DISABLE_TINY_SCOUT_BYPASS") != "1"
        ):
            duration = time.perf_counter() - start_time
            compact_body = f"### Direct File Pass-Through (`{Path(str(target_path)).name}`, {raw_input_tokens} tok)\n```\n{raw_content.strip()}\n```"
            out_tok = estimate_tokens(compact_body)
            return _record_and_return(
                ScoutResult(
                    target=str(target_path),
                    query=query,
                    findings=compact_body,
                    input_tokens_est=raw_input_tokens,
                    output_tokens=out_tok,
                    tokens_saved=max(0, raw_input_tokens - out_tok),
                    savings_pct=0.0,
                    duration_seconds=round(duration, 4),
                    model_name="deterministic-tiny-bypass [sub-1ms]",
                    is_local=True,
                    read_latency_ms=read_latency_ms,
                    ingress_mode="tiny_file_bypass",
                )
            )

        # Check Warm Content-Hash + Query Digest Cache (<1ms repeat turnaround) # [direct-edit]
        import hashlib
        import sqlite3
        from whole_body_compute.shield import get_default_shield_db_path

        cache_key = hashlib.sha256(
            f"{raw_content}\0{query.strip().lower()}".encode("utf-8", errors="replace")
        ).hexdigest()
        def_cache_key = hashlib.sha256(
            raw_content.encode("utf-8", errors="replace") + b"::default"
        ).hexdigest()

        mem_cache = getattr(self.__class__, "_WARM_DIGEST_CACHE", None)
        if mem_cache is None:
            mem_cache = {}
            setattr(self.__class__, "_WARM_DIGEST_CACHE", mem_cache)

        cached_entry = mem_cache.get(cache_key) or mem_cache.get(def_cache_key)
        if cached_entry is None:
            try:
                db_p = get_default_shield_db_path()
                with _connect_scout_db(db_p, timeout=5.0) as conn:
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS scout_digest_cache (
                            cache_key TEXT PRIMARY KEY,
                            findings TEXT NOT NULL,
                            model_name TEXT NOT NULL,
                            input_tokens_est INTEGER NOT NULL,
                            output_tokens INTEGER NOT NULL,
                            tokens_saved INTEGER NOT NULL,
                            savings_pct REAL NOT NULL,
                            is_local INTEGER NOT NULL,
                            updated_at REAL NOT NULL
                        );
                        """
                    )
                    row = conn.execute(
                        """
                        SELECT findings, model_name, input_tokens_est, output_tokens, tokens_saved, savings_pct, is_local 
                        FROM scout_digest_cache 
                        WHERE cache_key = ? OR cache_key = ?
                        ORDER BY (cache_key = ?) DESC LIMIT 1;
                        """,
                        (cache_key, def_cache_key, cache_key),
                    ).fetchone()
                    if row:
                        cached_entry = {
                            "findings": str(row[0]),
                            "model_name": str(row[1]),
                            "input_tokens_est": int(row[2]),
                            "output_tokens": int(row[3]),
                            "tokens_saved": int(row[4]),
                            "savings_pct": float(row[5]),
                            "is_local": bool(row[6]),
                        }
                        mem_cache[cache_key] = cached_entry
                        mem_cache[def_cache_key] = cached_entry
            except Exception:
                pass

        if cached_entry is not None and (
            not (raw_content.strip() and self._is_log_or_trace(raw_content, str(target_path)))
            or "deterministic-log-airlock" in str(cached_entry.get("model_name", ""))
        ):
            duration = time.perf_counter() - start_time
            base_m = str(cached_entry["model_name"]).replace(" [warm-cache]", "")
            return _record_and_return(
                ScoutResult(
                    target=str(target_path),
                    query=query,
                    findings=cached_entry["findings"],
                    input_tokens_est=cached_entry["input_tokens_est"],
                    output_tokens=cached_entry["output_tokens"],
                    tokens_saved=cached_entry["tokens_saved"],
                    savings_pct=cached_entry["savings_pct"],
                    duration_seconds=round(duration, 4),
                    model_name=f"{base_m} [warm-cache]",
                    is_local=cached_entry["is_local"],
                    read_latency_ms=read_latency_ms,
                    ingress_mode="warm_digest_cache",
                )
            )

        def _cache_and_record(result: ScoutResult) -> ScoutResult:
            if result.error is None and result.findings:
                entry = {
                    "findings": result.findings,
                    "model_name": result.model_name,
                    "input_tokens_est": result.input_tokens_est,
                    "output_tokens": result.output_tokens,
                    "tokens_saved": result.tokens_saved,
                    "savings_pct": result.savings_pct,
                    "is_local": result.is_local,
                }
                mem_cache[cache_key] = entry
                mem_cache[def_cache_key] = entry
                try:
                    db_p = get_default_shield_db_path()
                    with _connect_scout_db(db_p, timeout=5.0) as conn:
                        conn.execute(
                            """
                            CREATE TABLE IF NOT EXISTS scout_digest_cache (
                                cache_key TEXT PRIMARY KEY,
                                findings TEXT NOT NULL,
                                model_name TEXT NOT NULL,
                                input_tokens_est INTEGER NOT NULL,
                                output_tokens INTEGER NOT NULL,
                                tokens_saved INTEGER NOT NULL,
                                savings_pct REAL NOT NULL,
                                is_local INTEGER NOT NULL,
                                updated_at REAL NOT NULL
                            );
                            """
                        )
                        for ck in (cache_key, def_cache_key):
                            conn.execute(
                                """
                                INSERT OR REPLACE INTO scout_digest_cache (
                                    cache_key, findings, model_name, input_tokens_est,
                                    output_tokens, tokens_saved, savings_pct, is_local, updated_at
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                                """,
                                (
                                    ck,
                                    result.findings,
                                    result.model_name,
                                    int(result.input_tokens_est),
                                    int(result.output_tokens),
                                    int(result.tokens_saved),
                                    float(result.savings_pct),
                                    1 if result.is_local else 0,
                                    time.time(),
                                ),
                            )
                        conn.commit()
                except Exception:
                    pass
            return _record_and_return(result)

        # Deterministic Airlock chunk selection (eliminates Donut-Hole middle truncation)
        selected_content, chunk_truncated, selected_markers = self.select_deterministic_chunks(
            raw_content,
            str(target_path),
            query,
            max_chars=16_000,
        )
        truncated = truncated or chunk_truncated
        input_tokens = max(raw_input_tokens, estimate_tokens(selected_content))

        # Deterministic Log Airlock Pre-Filter: For non-empty log/trace files, return the
        # lossless <5ms structured incident metadata + verbatim error lines + deduplicated windows
        # immediately without waiting on LLM prose generation (which can drop exact IDs).
        if raw_content.strip() and self._is_log_or_trace(raw_content, str(target_path)):
            output_tokens = estimate_tokens(selected_content)
            tokens_saved = max(0, input_tokens - output_tokens)
            savings_pct = round((tokens_saved / max(1, input_tokens)) * 100, 1)
            duration = time.perf_counter() - start_time
            return _cache_and_record(
                ScoutResult(
                    target=str(target_path),
                    query=query,
                    findings=selected_content,
                    input_tokens_est=input_tokens,
                    output_tokens=output_tokens,
                    tokens_saved=tokens_saved,
                    savings_pct=savings_pct,
                    duration_seconds=round(duration, 4),
                    model_name="deterministic-log-airlock [sub-5ms]",
                    is_local=True,
                    read_latency_ms=read_latency_ms,
                    ingress_mode="airlock_log_prefilter",
                )
            )

        # Thermal Safeguard: If ThermalSupervisor reports active throttling, skip LLM prose
        # generation immediately and return the deterministic .pyi skeleton + scored AST slices /
        # trace_parser digest in <5ms.
        if self._is_thermally_throttled():
            output_tokens = estimate_tokens(selected_content)
            tokens_saved = max(0, input_tokens - output_tokens)
            savings_pct = round((tokens_saved / max(1, input_tokens)) * 100, 1)
            duration = time.perf_counter() - start_time
            return _record_and_return(
                ScoutResult(
                    target=str(target_path),
                    query=query,
                    findings=selected_content,
                    input_tokens_est=input_tokens,
                    output_tokens=output_tokens,
                    tokens_saved=tokens_saved,
                    savings_pct=savings_pct,
                    duration_seconds=round(duration, 4),
                    model_name="deterministic-airlock [thermal-throttle-bypass]",
                    is_local=True,
                    read_latency_ms=read_latency_ms,
                    ingress_mode="airlock_deterministic",
                )
            )

        _structured_code_exts = {
            ".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".rs", ".go",
            ".c", ".cpp", ".h", ".hpp", ".java", ".kt", ".swift",
            ".json", ".yaml", ".yml", ".toml", ".sql", ".sh",
        }
        _default_structural_queries = {
            "",
            "summarize key structure and findings",
            "analyze this file, identify any errors or anomalies, and extract key functions/logic.",
            "extract key functions and classes",
            "list functions and classes",
            "outline",
            "skeleton",
        }
        is_default_structural_scan = query.strip().lower() in _default_structural_queries
        if (
            is_default_structural_scan
            and target_suffix in _structured_code_exts
            and os.environ.get("WBC_SCOUT_FORCE_LLM") != "1"
            and os.environ.get("WBC_SCOUT_FORCE_OLLAMA") != "1"
        ):
            if selected_content:
                ast_findings = (
                    f"### Recon Diagnosis for `{Path(str(target_path)).name}`\n"
                    f"- Matched Symbols/Frames: {', '.join(selected_markers[:8]) if selected_markers else 'Full scan'}\n\n"
                    f"{selected_content[:12_000]}"
                )
            else:
                ast_findings = self._heuristic_extract(raw_content, str(target_path), query)
            output_tokens = estimate_tokens(ast_findings)
            tokens_saved = max(0, input_tokens - output_tokens)
            savings_pct = round((tokens_saved / max(1, input_tokens)) * 100, 1)
            duration = time.perf_counter() - start_time
            return _cache_and_record(
                ScoutResult(
                    target=str(target_path),
                    query=query,
                    findings=ast_findings,
                    input_tokens_est=input_tokens,
                    output_tokens=output_tokens,
                    tokens_saved=tokens_saved,
                    savings_pct=savings_pct,
                    duration_seconds=round(duration, 4),
                    model_name="deterministic-ast-airlock [sub-10ms]",
                    is_local=True,
                    read_latency_ms=read_latency_ms,
                    ingress_mode="airlock_ast_prefilter",
                )
            )

        # Formulate prompt using deterministic chunks
        truncation_note = " [Airlock selective slice]" if truncated else ""
        resolved_path = Path(os.path.expanduser(str(target_path))).resolve()
        prompt = (
            f"Objective: {query}\n\n"
            f"Target: {resolved_path}{truncation_note} ({file_size} bytes, ~{input_tokens} tokens)\n\n"
            f"```\n{selected_content}\n```\n\n"
            "Provide: 1) One-line Diagnosis/Summary, 2) Key Functions/Lines, 3) Actionable Fix or Findings."
        )

        # 1. Check Direct Local Metal Daemons (:9380 gemma4-26b & :9379 voicefi-coder-mlx) # [direct-edit]
        metal_model, metal_findings = self._query_local_metal_daemons(
            prompt, self.DEFAULT_SCOUT_INSTRUCTIONS
        )
        if metal_findings:
            output_tokens = estimate_tokens(metal_findings)
            tokens_saved = max(0, input_tokens - output_tokens)
            savings_pct = round((tokens_saved / max(1, input_tokens)) * 100, 1)
            duration = time.perf_counter() - start_time
            return _cache_and_record(
                ScoutResult(
                    target=str(target_path),
                    query=query,
                    findings=metal_findings,
                    input_tokens_est=input_tokens,
                    output_tokens=output_tokens,
                    tokens_saved=tokens_saved,
                    savings_pct=savings_pct,
                    duration_seconds=round(duration, 2),
                    model_name=f"metal/{metal_model}",
                    is_local=True,
                    read_latency_ms=read_latency_ms,
                    ingress_mode="unified_ram",
                )
            )

        # 1b. AST-First Code & Structured-File Airlock (<10ms vs 3-4s Ollama prose generation):
        # When warm :9379 MLX is offline, never stall 3-4s in Ollama on tiny files or structured source code
        # where deterministic AST chunking + .pyi contracts preserve 100% exact symbols & line numbers.
        _structured_code_exts = {
            ".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".rs", ".go",
            ".c", ".cpp", ".h", ".hpp", ".java", ".kt", ".swift",
            ".json", ".yaml", ".yml", ".toml", ".sql", ".sh",
        }
        if (
            os.environ.get("WBC_SCOUT_FORCE_OLLAMA") != "1"
            and (raw_input_tokens <= 600 or target_suffix in _structured_code_exts)
        ):
            if selected_content:
                ast_findings = (
                    f"### Recon Diagnosis for `{Path(str(target_path)).name}`\n"
                    f"- Matched Symbols/Frames: {', '.join(selected_markers[:8]) if selected_markers else 'Full scan'}\n\n"
                    f"{selected_content[:12_000]}"
                )
            else:
                ast_findings = self._heuristic_extract(raw_content, str(target_path), query)
            output_tokens = estimate_tokens(ast_findings)
            tokens_saved = max(0, input_tokens - output_tokens)
            savings_pct = round((tokens_saved / max(1, input_tokens)) * 100, 1)
            duration = time.perf_counter() - start_time
            return _cache_and_record(
                ScoutResult(
                    target=str(target_path),
                    query=query,
                    findings=ast_findings,
                    input_tokens_est=input_tokens,
                    output_tokens=output_tokens,
                    tokens_saved=tokens_saved,
                    savings_pct=savings_pct,
                    duration_seconds=round(duration, 4),
                    model_name="deterministic-ast-airlock [sub-10ms]",
                    is_local=True,
                    read_latency_ms=read_latency_ms,
                    ingress_mode="airlock_ast_prefilter",
                )
            )

        # 2. Check local Ollama instance
        ollama_model, ollama_findings = self._query_ollama(prompt, self.DEFAULT_SCOUT_INSTRUCTIONS)
        if ollama_findings:
            output_tokens = estimate_tokens(ollama_findings)
            tokens_saved = max(0, input_tokens - output_tokens)
            savings_pct = round((tokens_saved / max(1, input_tokens)) * 100, 1)
            duration = time.perf_counter() - start_time

            return _cache_and_record(
                ScoutResult(
                    target=str(target_path),
                    query=query,
                    findings=ollama_findings,
                    input_tokens_est=input_tokens,
                    output_tokens=output_tokens,
                    tokens_saved=tokens_saved,
                    savings_pct=savings_pct,
                    duration_seconds=round(duration, 2),
                    model_name=f"ollama/{ollama_model}",
                    is_local=True,
                    read_latency_ms=read_latency_ms,
                    ingress_mode="unified_ram",
                )
            )

        # 3. If local model weights are present and LiteRT is ready, execute on-device
        if self.engine.is_installed and self.engine.model_exists:
            try:
                findings = await self.engine.chat_text(
                    prompt=prompt,
                    system_instructions=self.DEFAULT_SCOUT_INSTRUCTIONS,
                    tools=[],
                )
                output_tokens = estimate_tokens(findings)
                tokens_saved = max(0, input_tokens - output_tokens)
                savings_pct = round((tokens_saved / max(1, input_tokens)) * 100, 1)
                duration = time.perf_counter() - start_time

                return _cache_and_record(
                    ScoutResult(
                        target=str(target_path),
                        query=query,
                        findings=findings,
                        input_tokens_est=input_tokens,
                        output_tokens=output_tokens,
                        tokens_saved=tokens_saved,
                        savings_pct=savings_pct,
                        duration_seconds=round(duration, 2),
                        model_name=self.engine.model_name,
                        is_local=True,
                        read_latency_ms=read_latency_ms,
                        ingress_mode="unified_ram",
                    )
                )
            except Exception as e:
                logger.warning(
                    f"Local scout execution error: {e}. Falling back to heuristic extraction."
                )

        # 4. Fallback / Fast Deterministic Rule-Based Scout
        duration = time.perf_counter() - start_time
        simulated_findings = self._heuristic_extract(raw_content, str(target_path), query)
        output_tokens = estimate_tokens(simulated_findings)
        tokens_saved = max(0, input_tokens - output_tokens)
        savings_pct = round((tokens_saved / max(1, input_tokens)) * 100, 1)

        note = ""
        if not self.engine.model_exists:
            note = f"\n(Heuristic mode: model weights not found at {self.engine.model_path})"

        return _cache_and_record(
            ScoutResult(
                target=str(target_path),
                query=query,
                findings=simulated_findings + note,
                input_tokens_est=input_tokens,
                output_tokens=output_tokens,
                tokens_saved=tokens_saved,
                savings_pct=savings_pct,
                duration_seconds=round(duration, 3),
                model_name=f"{self.engine.model_name} [heuristic]",
                is_local=False,
                read_latency_ms=read_latency_ms,
                ingress_mode="unified_ram",
            )
        )

    def _query_ollama(self, prompt: str, system_prompt: str) -> tuple[Optional[str], Optional[str]]:
        """Query local Ollama instance with auto-discovery and thermal safety guard."""
        if self._is_thermally_throttled():
            return None, None

        if not self._is_port_open("127.0.0.1", 11434, timeout=0.025):
            return None, None

        base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        preferred = [
            "qwen2.5-coder:1.5b",
            "tev1:0.8b",
            "gemma2:2b",
            "llama3.2:1b",
            "tev1:latest",
            "nimble:latest",
        ]
        try:
            from whole_body_compute.local.router import default_router

            route = default_router.resolve_model("scout")
            if route and route.is_local and route.selected_model:
                approved_model = route.selected_model.replace("ollama/", "").strip()
                if approved_model in preferred:
                    preferred.remove(approved_model)
                preferred.insert(0, approved_model)
        except Exception:
            pass

        selected_model = None
        try:
            req = urllib.request.Request(f"{base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                data = json.loads(resp.read().decode())
                installed = [
                    m.get("name", "")
                    for m in data.get("models", [])
                    if "embed" not in m.get("name", "").lower()
                ]
                for p in preferred:
                    if p in installed:
                        selected_model = p
                        break
                if not selected_model and installed:
                    selected_model = installed[0]
        except Exception:
            return None, None

        if not selected_model:
            return None, None

        keep_alive_val: Union[int, str] = "30m"
        try:
            from whole_body_compute.local.supervisor import default_supervisor
            keep_alive_val = default_supervisor.get_adaptive_keep_alive(selected_model).get(
                "recommended_keep_alive", "30m"
            )
        except Exception:
            keep_alive_val = "5m" if self._is_thermally_throttled() else -1

        payload = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "keep_alive": keep_alive_val,
            "options": {
                "temperature": 0.2,
                "num_predict": 220,
            },
        }
        try:
            req = urllib.request.Request(
                f"{base_url}/api/chat",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                if resp.status == 200:
                    res = json.loads(resp.read().decode())
                    if "load_duration" in res:
                        try:
                            from whole_body_compute.local.supervisor import default_supervisor
                            default_supervisor.record_model_load_telemetry(
                                model_name=selected_model,
                                load_duration_ms=res["load_duration"] / 1e6,
                                total_duration_ms=res.get("total_duration", 0) / 1e6,
                                runtime="ollama",
                            )
                        except Exception:
                            pass
                    content = res.get("message", {}).get("content", "").strip()
                    if content:
                        return selected_model, content
        except Exception as e:
            logger.debug(f"Ollama scout query error: {e}")
        return None, None

    def _heuristic_extract(self, content: str, path: str, query: str) -> str:
        """
        Fast deterministic AST / trace_parser extractor when model weights or daemons
        are unavailable. Uses select_deterministic_chunks across the full file.
        """
        chunked, _, markers = self.select_deterministic_chunks(
            content,
            path,
            query,
            max_chars=12_000,
        )
        if chunked:
            return (
                f"### Recon Diagnosis for `{Path(path).name}`\n"
                f"- Matched Symbols/Frames: {', '.join(markers[:8]) if markers else 'Full scan'}\n\n"
                f"{chunked}"
            )

        lines = content.splitlines()
        return (
            f"### Recon Summary for `{Path(path).name}`\n"
            f"- Scanned {len(lines)} lines ({len(content)} characters).\n"
            f"- Ready for task: {query}"
        )

    def prewarm(self, background: bool = True) -> Dict[str, Any]:
        """
        Prewarm the on-device scout engine in Apple Silicon Unified RAM.
        Pings :9379 (voicefi-coder-mlx) and pins Ollama scout model weights (keep_alive: -1)
        when thermally safe so initial scout executions avoid cold-start penalties.
        # [direct-edit]
        """
        if self._is_thermally_throttled():
            return {"prewarmed": False, "reason": "thermally_throttled"}

        def _do_prewarm() -> Dict[str, Any]:
            results: Dict[str, Any] = {"metal_daemons": {}, "ollama": {}}

            # 1. Ping :9379 (voicefi-coder-mlx) and :9380 (gemma4-26b)
            for model_id, endpoint in [
                ("voicefi-coder-mlx", "http://127.0.0.1:9379"),
                ("gemma4-26b", "http://127.0.0.1:9380"),
            ]:
                try:
                    req = urllib.request.Request(
                        f"{endpoint}/health",
                        headers={"User-Agent": "scout-prewarm"},
                    )
                    with urllib.request.urlopen(req, timeout=0.6) as resp:
                        results["metal_daemons"][model_id] = resp.status == 200
                except Exception:
                    results["metal_daemons"][model_id] = False

            # 2. Ping Ollama scout model with keep_alive: -1
            try:
                base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
                req_tags = urllib.request.Request(f"{base_url}/api/tags")
                with urllib.request.urlopen(req_tags, timeout=1.0) as resp:
                    if resp.status == 200:
                        tags_data = json.loads(resp.read().decode())
                        installed = [
                            m.get("name", "")
                            for m in tags_data.get("models", [])
                            if "embed" not in m.get("name", "").lower()
                        ]
                        target_model = None
                        for pref in ("voicefi-coder:1.5b", "qwen2.5-coder:1.5b", "gemma2:2b"):
                            if pref in installed:
                                target_model = pref
                                break
                        if not target_model and installed:
                            target_model = installed[0]

                        if target_model:
                            keep_alive_val: Union[int, str] = (
                                "5m" if self._is_thermally_throttled() else -1
                            )
                            ping_payload = {
                                "model": target_model,
                                "messages": [],
                                "keep_alive": keep_alive_val,
                            }
                            ping_req = urllib.request.Request(
                                f"{base_url}/api/chat",
                                data=json.dumps(ping_payload).encode("utf-8"),
                                headers={"Content-Type": "application/json"},
                                method="POST",
                            )
                            with urllib.request.urlopen(ping_req, timeout=5.0) as ping_resp:
                                results["ollama"]["pinned_model"] = target_model
                                results["ollama"]["status"] = ping_resp.status == 200
            except Exception as e:
                results["ollama"]["error"] = str(e)

            return results

        if background:
            import threading

            t = threading.Thread(target=_do_prewarm, name="scout-prewarm", daemon=True)
            t.start()
            return {"prewarmed": True, "mode": "background"}
        return _do_prewarm()

    async def scout_many(
        self,
        target_paths: List[Union[str, Path]],
        query: str = "Analyze this file, identify any errors or anomalies, and extract key functions/logic.",
        max_bytes: int = 500_000,
    ) -> List[ScoutResult]:
        """
        Execute concurrent on-device scout runs across multiple files/directories in a single turn.
        Resilient against individual missing/corrupted file exceptions so the batch never crashes.
        # [direct-edit]
        """
        if not target_paths:
            return []
        tasks = [
            self.scout(p, query=query, max_bytes=max_bytes)
            for p in target_paths
        ]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        results: List[ScoutResult] = []
        for p, res in zip(target_paths, gathered):
            if isinstance(res, Exception):
                err_msg = f"[Error] {type(res).__name__}: {res}"
                results.append(
                    ScoutResult(
                        target=str(p),
                        query=query,
                        findings=err_msg,
                        input_tokens_est=0,
                        output_tokens=estimate_tokens(err_msg),
                        tokens_saved=0,
                        savings_pct=0.0,
                        duration_seconds=0.0,
                        model_name=getattr(self.engine, "model_name", "local_scout"),
                        is_local=False,
                        read_latency_ms=0.0,
                        error=err_msg,
                    )
                )
            else:
                results.append(res)
        return results


def prewarm_scout(background: bool = True) -> Dict[str, Any]:
    """Module-level helper to trigger non-blocking scout prewarm. # [direct-edit]"""
    scout = ReconScout()
    return scout.prewarm(background=background)


def speculative_prescout(target_path: Union[str, Path], query: Optional[str] = None) -> bool:
    """
    Speculatively pre-digest a file in the background upon ast_slice_gate denial.
    Populates scout_digest_cache under both the query key and the file default key SHA-256(raw_content + "::default").
    # [direct-edit]
    """
    try:
        resolved = Path(os.path.expanduser(str(target_path))).resolve()
        if not resolved.is_file():
            return False

        # If thermally throttled, do not launch speculative LLM work
        try:
            from whole_body_compute.local.supervisor import default_supervisor

            health = default_supervisor.check_health()
            if getattr(health, "should_throttle", False) or not getattr(health, "is_safe", True):
                return False
        except Exception:
            pass

        # Check if already cached in scout_digest_cache
        try:
            import hashlib
            import sqlite3
            from whole_body_compute.shield import get_default_shield_db_path

            raw_head = resolved.read_bytes()[:500000]
            def_key = hashlib.sha256(raw_head + b"::default").hexdigest()
            db_p = get_default_shield_db_path()
            if db_p.exists():
                with _connect_scout_db(db_p, timeout=5.0) as conn:
                    row = conn.execute(
                        "SELECT 1 FROM scout_digest_cache WHERE cache_key = ? LIMIT 1;",
                        (def_key,),
                    ).fetchone()
                    if row:
                        return True  # Already cached
        except Exception:
            pass

        q_str = query or "Analyze this file, identify any errors or anomalies, and extract key functions/logic."

        # If running under pytest or subprocesses disabled, run via daemon thread
        is_test = "PYTEST_CURRENT_TEST" in os.environ or os.environ.get("WBC_DISABLE_SPECULATIVE_SUBPROCESS") == "1"
        if is_test:
            import threading

            def _in_thread():
                try:
                    scout_obj = ReconScout()
                    asyncio.run(scout_obj.scout(resolved, query=q_str, record_shield=False))
                except Exception:
                    pass

            th = threading.Thread(target=_in_thread, name="speculative-prescout", daemon=True)
            th.start()
            return True
        else:
            # Spawn detached background subprocess that survives CLI hook process exit
            import subprocess

            bg_env = os.environ.copy()
            bg_env["WBC_SPECULATIVE_PRESCOUT"] = "1"
            subprocess.Popen(
                [sys.executable, "-m", "whole_body_compute.local.scout", str(resolved), "-q", q_str],
                env=bg_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True
    except Exception:
        return False


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for VoiceFi On-Device Recon Scout."""
    import argparse
    import asyncio
    import sys

    parser = argparse.ArgumentParser(description="VoiceFi On-Device Recon Scout")
    parser.add_argument("target", nargs="*", default=[], help="Target file(s) or directory(ies) to pre-digest")
    parser.add_argument(
        "--slice",
        "--slices",
        dest="slices",
        nargs="+",
        default=[],
        help="Multi-file batch AST symbol slices (e.g. file.py:SymbolName file.py:10-50)",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="Analyze this file, identify any errors or anomalies, and extract key functions/logic.",
        help="Recon objective",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Optional local model override",
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args(argv)

    if args.slices:
        from whole_body_compute.local.symbols import extract_batch_symbol_slices
        res = extract_batch_symbol_slices(args.slices)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(res.get("formatted", ""))
        return 0

    if not args.target:
        parser.print_help()
        return 1

    engine = LocalModelEngine(model_name=args.model) if args.model else None
    scout = ReconScout(engine=engine)

    def _run(coro):
        # [direct-edit] Run safely even if an event loop is already active in current thread
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        return asyncio.run(coro)

    if len(args.target) == 1:
        res = _run(scout.scout(args.target[0], query=args.query))
        if args.json:
            print(json.dumps(res.to_dict(), indent=2))
        else:
            print(f"=== VoiceFi Recon Scout [{res.model_name}] ===")
            print(f"Target:       {res.target} (~{res.input_tokens_est} tokens)")
            print(f"Duration:     {res.duration_seconds}s (Read RAM Latency: {res.read_latency_ms}ms)")
            print(f"Tokens Saved: {res.tokens_saved} tokens ({res.savings_pct}% context compression)")
            print(f"Local Compute: {res.is_local} (Ingress: {res.ingress_mode})")
            if res.error and res.error not in (res.findings or ""):
                print(f"Error:        {res.error}")
            print("\n--- Scout Findings ---")
            print(res.findings)
        return 1 if res.error else 0
    else:
        results = _run(scout.scout_many(args.target, query=args.query))
        if args.json:
            print(json.dumps([r.to_dict() for r in results], indent=2))
        else:
            print(f"=== VoiceFi Batch Recon Scout ({len(results)} targets) ===")
            for r in results:
                print(f"\nTarget:       {r.target} [{r.model_name}] ({r.duration_seconds}s)")
                print(f"Tokens Saved: {r.tokens_saved} tokens ({r.savings_pct}% context compression) | Ingress: {r.ingress_mode}")
                if r.error and r.error not in (r.findings or ""):
                    print(f"Error:        {r.error}")
                print("--- Findings ---")
                print(r.findings)
        return 1 if any(r.error for r in results) else 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
