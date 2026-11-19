"""
AST Vocab-Fusion & Deterministic Symbol Snapping (`whole_body_compute.stt.vocab_fusion`).

Bridges Organ 03 Foveal Focus (`OrganEyes` active window/tab), `TactileSensor` git delta,
and Organ 02 `LocalSymbolIndex` (`~/.whole_body_compute/symbols.db`) into a sub-5ms
dual-stage Speech-to-Text grounding pipeline:

  - Stage 1 (Pre-Decode Bias): Prioritizes active-file symbols (P0), uncommitted git
    working-set symbols (P1), and repo-spine symbols (P2) into compact `contextualStrings`
    (Apple Neural Engine `SFSpeechRecognizer`) and `initial_prompt` (MLX / Faster-Whisper).
  - Stage 2 (Post-Decode Deterministic AST Snap): Splits CamelCase / snake_case / file
    paths into spoken word tuples and snaps 2..6-word spans (plus constrained fuzzy hits
    on long multi-word identifiers) directly to exact AST identifiers, guarded against
    false-positive snapping of common single English words.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from whole_body_compute.stt.biasing import PhoneticNormalizer

logger = logging.getLogger("whole_body_compute.stt.vocab_fusion")

# Common single English words that must NEVER be snapped on their own unless preceded
# by an explicit code cue (e.g., "class ...", "function ...", "in ...", "dot py").
COMMON_ENGLISH_BLOCKLIST: Set[str] = {
    "a", "an", "and", "app", "are", "args", "as", "at", "audio", "auto",
    "base", "be", "begin", "body", "box", "brief", "build", "by",
    "cache", "call", "check", "class", "clean", "clear", "cli", "click", "client",
    "close", "code", "command", "config", "context", "core", "count", "create",
    "data", "date", "db", "debug", "default", "delete", "delta", "device", "diff",
    "dir", "do", "doc", "doctor", "docs", "dom", "done", "draft", "drive",
    "ears", "edit", "embed", "end", "engine", "entry", "env", "error", "event", "eyes",
    "fast", "fetch", "field", "file", "files", "filter", "find", "fix", "flag",
    "flush", "focus", "for", "format", "frame", "from", "func", "function",
    "gate", "get", "git", "go", "guard",
    "hand", "hands", "has", "hash", "head", "health", "heart", "help", "hook", "hooks", "hub",
    "id", "immune", "import", "in", "index", "info", "init", "input", "inspect", "install", "is", "it", "item",
    "json", "key", "keys", "kind",
    "last", "launch", "layout", "level", "lib", "license", "line", "lines", "link", "list", "live", "load", "local", "lock", "log", "logs", "loop",
    "main", "make", "manager", "map", "match", "max", "meeting", "memory", "menu", "merge", "message", "meta", "metal", "method", "metric", "metrics", "min", "mode", "model", "models", "module", "monitor",
    "name", "names", "new", "next", "node", "none", "note", "notes",
    "of", "ok", "on", "open", "opt", "option", "or", "order", "organ", "organs", "out", "output",
    "pack", "page", "panel", "param", "parse", "parser", "pass", "paste", "patch", "path", "paths", "payload", "people", "pill", "plan", "plugin", "point", "policy", "port", "post", "pre", "prefix", "preview", "print", "pro", "process", "profile", "project", "prompt", "proxy", "pull", "push", "py",
    "query", "queue",
    "ram", "range", "rate", "raw", "read", "ready", "record", "red", "redact", "ref", "refresh", "regex", "register", "release", "reload", "remote", "remove", "render", "replace", "reply", "repo", "report", "req", "request", "reset", "resolve", "response", "restore", "result", "results", "resume", "retry", "return", "review", "root", "route", "router", "row", "rows", "rule", "rules", "run", "runner", "runtime",
    "safe", "sample", "save", "scan", "schema", "score", "screen", "script", "scroll", "search", "secret", "section", "security", "seed", "select", "self", "send", "sensor", "server", "service", "session", "set", "setup", "share", "shell", "shield", "show", "sign", "signal", "size", "skill", "skills", "skip", "sleep", "slice", "slicer", "slot", "snap", "snapshot", "socket", "sort", "sound", "source", "space", "spec", "speech", "speed", "split", "stage", "start", "state", "stats", "status", "step", "steps", "stop", "store", "stream", "string", "stub", "style", "submit", "summary", "sync", "system",
    "tab", "table", "tag", "tap", "target", "task", "tasks", "team", "telemetry", "temp", "template", "term", "test", "tests", "text", "that", "the", "theme", "thermal", "this", "thread", "tier", "time", "timer", "title", "to", "token", "tokens", "tool", "tools", "top", "trace", "track", "train", "transcribe", "transcript", "tree", "trigger", "trim", "try", "turn", "type",
    "ui", "unit", "update", "upgrade", "url", "use", "user", "users", "util", "utils",
    "val", "valid", "value", "values", "var", "vault", "vector", "verify", "version", "view", "vision", "vocab", "voice",
    "wait", "watch", "web", "width", "window", "with", "word", "words", "work", "worker", "workflow", "workspace", "wrap", "write",
    "yaml", "zero", "zone",
}

CODE_CUE_WORDS: Set[str] = {
    "class", "def", "function", "method", "module", "file", "import", "from",
    "in", "inspect", "open", "edit", "update", "call", "run", "symbol",
}

EXTENSION_SPOKEN_MAP: Dict[str, Tuple[str, ...]] = {
    ".py": ("dot py", "dot pie"),
    ".ts": ("dot ts", "dot t s"),
    ".tsx": ("dot tsx", "dot t s x"),
    ".js": ("dot js", "dot j s"),
    ".jsx": ("dot jsx",),
    ".rs": ("dot rs", "dot rust"),
    ".go": ("dot go",),
    ".md": ("dot md", "dot markdown"),
    ".json": ("dot json",),
    ".yaml": ("dot yaml",),
    ".yml": ("dot yml", "dot yaml"),
    ".sh": ("dot sh",),
    ".toml": ("dot toml",),
}


@dataclass
class VocabBundle:
    """Hydrated AST vocabulary context for a single dictation/transcription turn."""

    workspace: str
    focused_app: Optional[str] = None
    active_file: Optional[str] = None
    git_branch: Optional[str] = None
    p0_focus_symbols: List[str] = field(default_factory=list)
    p1_git_symbols: List[str] = field(default_factory=list)
    p2_repo_symbols: List[str] = field(default_factory=list)
    file_paths: List[str] = field(default_factory=list)
    hydration_ms: float = 0.0

    @property
    def all_symbols(self) -> List[str]:
        """Deduplicated priority-ordered list of symbols and key filenames."""
        seen: Set[str] = set()
        ordered: List[str] = []
        for item in (
            self.p0_focus_symbols
            + self.p1_git_symbols
            + self.p2_repo_symbols
            + [Path(p).name for p in self.file_paths]
        ):
            if item and item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered

    def to_ane_contextual_strings(self, limit: int = 100) -> List[str]:
        """Build `contextualStrings` for macOS SFSpeechRecognizer (canonical + spoken forms)."""
        out: List[str] = []
        seen: Set[str] = set()
        for sym in self.all_symbols:
            if len(out) >= limit:
                break
            if sym not in seen:
                seen.add(sym)
                out.append(sym)
            spoken = " ".join(split_identifier_to_words(sym))
            if spoken and spoken.lower() != sym.lower() and spoken not in seen and len(out) < limit:
                seen.add(spoken)
                out.append(spoken)
        return out[:limit]

    def to_whisper_prompt(self, max_terms: int = 64) -> str:
        """Build compact <=180-token `initial_prompt` for MLX / Faster-Whisper."""
        core = ["pytest", "git", "CLI", "API", "AST", "macOS", "Python", "TypeScript"]
        if self.git_branch:
            core.append(self.git_branch)
        terms = list(dict.fromkeys(self.all_symbols[:max_terms] + core))[:max_terms]
        return f"Codebase symbols and files: {', '.join(terms)}."


@dataclass
class SnapReceipt:
    """Audit record of a single deterministic AST symbol snap."""

    spoken: str
    snapped: str
    match_type: str  # "exact" | "file" | "fuzzy"


def split_identifier_to_words(identifier: str) -> List[str]:
    """Split CamelCase, PascalCase, snake_case, or kebab-case into lowercase spoken tokens."""
    if not identifier:
        return []
    stem = identifier
    # Strip leading underscores
    stem = stem.lstrip("_")
    # Split on separators and CamelCase boundaries
    raw = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|[A-Z]?[a-z]+|\d+", stem)
    return [w.lower() for w in raw if w]


def _levenshtein_bounded(a: str, b: str, max_dist: int = 1) -> int:
    """Fast bounded Levenshtein distance; returns `max_dist + 1` immediately if exceeded."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > max_dist:
        return max_dist + 1
    if max_dist == 1:
        # O(N) single-edit check
        if la == lb:
            diffs = sum(1 for x, y in zip(a, b) if x != y)
            return diffs if diffs <= 1 else 2
        short, long_s = (a, b) if la < lb else (b, a)
        i = j = diffs = 0
        while i < len(short) and j < len(long_s):
            if short[i] == long_s[j]:
                i += 1
                j += 1
            else:
                diffs += 1
                if diffs > 1:
                    return 2
                j += 1
        return 1
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * lb
        row_min = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            if curr[j] < row_min:
                row_min = curr[j]
        if row_min > max_dist:
            return max_dist + 1
        prev = curr
    return prev[lb]


class ASTVocabHydrator:
    """
    Sub-5ms vocabulary hydrator combining:
      P0: Active IDE/terminal window & file (`OrganEyes` AXUIElement)
      P1: Modified/staged git working set & current branch
      P2: Repo-wide public symbols from `LocalSymbolIndex` (`symbols.db`)
    """

    def __init__(
        self,
        workspace: Optional[Path] = None,
        db_path: Optional[Path] = None,
    ):
        self.workspace = Path(workspace).resolve() if workspace else Path.cwd().resolve()
        self.db_path = Path(db_path) if db_path else None

    def _detect_focus_context(self) -> Tuple[Optional[str], Optional[str]]:
        """Best-effort <2ms focused app & active file title via OrganEyes (safe on non-macOS)."""
        if os.environ.get("WBC_EARS_DISABLE_EYES") == "1":
            return None, None
        try:
            from whole_body_compute.organs.eyes import OrganEyes

            snap = OrganEyes.inspect(max_elements=8)
            app_name = getattr(snap, "app_name", None) or getattr(snap, "app", None)
            title = getattr(snap, "window_title", None) or getattr(snap, "title", None) or ""
            # Extract filename-like token from IDE window title (e.g. "immune_vault.py - Cursor")
            m = re.search(r"([A-Za-z0-9_./-]+\.(?:py|ts|tsx|js|jsx|rs|go|md|json|yaml|yml|sh|toml))", title)
            active_file = m.group(1) if m else None
            return app_name, active_file
        except Exception:
            return None, None

    def _detect_git_context(self) -> Tuple[Optional[str], List[str]]:
        """Read git branch and modified/staged file paths in <2ms."""
        branch: Optional[str] = None
        changed_files: List[str] = []
        try:
            head = self.workspace / ".git" / "HEAD"
            if head.is_file():
                raw = head.read_text(encoding="utf-8").strip()
                if raw.startswith("ref: refs/heads/"):
                    branch = raw.replace("ref: refs/heads/", "", 1).strip()
        except Exception:
            pass

        try:
            proc = subprocess.run(
                ["git", "status", "--porcelain", "-uall"],
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=0.4,
            )
            if proc.returncode == 0:
                for line in proc.stdout.splitlines()[:40]:
                    if len(line) > 3:
                        rel = line[3:].strip().split(" -> ")[-1].strip('"')
                        if rel:
                            changed_files.append(rel)
        except Exception:
            pass
        return branch, changed_files

    def _query_symbols_db(
        self,
        active_file: Optional[str],
        changed_files: Sequence[str],
        p0_limit: int = 30,
        p1_limit: int = 25,
        p2_limit: int = 25,
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Query `~/.whole_body_compute/symbols.db` via LocalSymbolIndex in <2ms."""
        p0: List[str] = []
        p1: List[str] = []
        p2: List[str] = []
        file_paths: List[str] = []

        try:
            from whole_body_compute.local.symbols import LocalSymbolIndex

            idx = LocalSymbolIndex(db_path=self.db_path, root_dir=self.workspace)
            ws_prefix = str(self.workspace)
            with idx._get_connection() as conn:
                # Helper to filter rows to this workspace when db is shared across repos
                like_prefix = f"{ws_prefix}%"

                # P0: Active file symbols
                if active_file:
                    active_base = Path(active_file).name
                    rows = conn.execute(
                        """
                        SELECT DISTINCT name FROM symbols
                        WHERE (file_path LIKE ? OR file_path LIKE ?)
                          AND name NOT LIKE '\\_%' ESCAPE '\\'
                        LIMIT ?
                        """,
                        (f"%{active_file}", f"%/{active_base}", p0_limit * 2),
                    ).fetchall()
                    for r in rows:
                        n = str(r["name"])
                        if len(n) >= 3 and n.lower() not in COMMON_ENGLISH_BLOCKLIST and n not in p0:
                            p0.append(n)
                            if len(p0) >= p0_limit:
                                break

                # P1: Git working-set file symbols
                for cf in changed_files[:15]:
                    file_paths.append(cf)
                    cf_base = Path(cf).name
                    rows = conn.execute(
                        """
                        SELECT DISTINCT name FROM symbols
                        WHERE (file_path LIKE ? OR file_path LIKE ?)
                          AND name NOT LIKE '\\_%' ESCAPE '\\'
                        LIMIT ?
                        """,
                        (f"%{cf}", f"%/{cf_base}", p1_limit),
                    ).fetchall()
                    for r in rows:
                        n = str(r["name"])
                        if (
                            len(n) >= 3
                            and n.lower() not in COMMON_ENGLISH_BLOCKLIST
                            and n not in p0
                            and n not in p1
                        ):
                            p1.append(n)
                            if len(p1) >= p1_limit:
                                break
                    if len(p1) >= p1_limit:
                        break

                # P2: Repo-spine public classes & multi-word functions
                rows = conn.execute(
                    """
                    SELECT DISTINCT name, file_path FROM symbols
                    WHERE file_path LIKE ?
                      AND name NOT LIKE '\\_%' ESCAPE '\\'
                      AND kind IN ('class', 'function', 'async_function', 'method', 'async_method')
                    ORDER BY CASE kind WHEN 'class' THEN 0 WHEN 'function' THEN 1 ELSE 2 END, length(name) DESC
                    LIMIT ?
                    """,
                    (like_prefix, p2_limit * 4),
                ).fetchall()
                for r in rows:
                    n = str(r["name"])
                    fp = str(r["file_path"])
                    try:
                        rel_fp = str(Path(fp).resolve().relative_to(self.workspace))
                    except Exception:
                        rel_fp = Path(fp).name
                    if rel_fp not in file_paths and len(file_paths) < 80:
                        file_paths.append(rel_fp)
                    if (
                        len(n) >= 4
                        and n.lower() not in COMMON_ENGLISH_BLOCKLIST
                        and n not in p0
                        and n not in p1
                        and n not in p2
                    ):
                        p2.append(n)
                        if len(p2) >= p2_limit:
                            break
        except Exception as exc:
            logger.debug("ASTVocabHydrator symbols.db query skipped: %s", exc)

        return p0, p1, p2, file_paths

    def hydrate(
        self,
        *,
        active_file_override: Optional[str] = None,
        skip_eyes: bool = False,
    ) -> VocabBundle:
        """Build a priority-ranked VocabBundle in <5ms."""
        t0 = time.perf_counter()
        focused_app: Optional[str] = None
        active_file: Optional[str] = active_file_override
        if not skip_eyes and active_file is None:
            focused_app, active_file = self._detect_focus_context()

        branch, changed_files = self._detect_git_context()
        p0, p1, p2, file_paths = self._query_symbols_db(active_file, changed_files)

        if active_file and active_file not in file_paths:
            file_paths.insert(0, active_file)

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return VocabBundle(
            workspace=str(self.workspace),
            focused_app=focused_app,
            active_file=active_file,
            git_branch=branch,
            p0_focus_symbols=p0,
            p1_git_symbols=p1,
            p2_repo_symbols=p2,
            file_paths=file_paths,
            hydration_ms=elapsed_ms,
        )


class ASTSymbolSnapper:
    """
    Stage-2 Post-Decode Deterministic AST Symbol Snapper.

    Maps spoken multi-word phrases ("spinal reflex gate", "immune vault dot py")
    back to exact AST identifiers and repository file paths in <1ms, while guarding
    against false-positive snapping of common single English words.
    """

    def __init__(self, bundle: VocabBundle):
        self.bundle = bundle
        # tuple[str, ...] -> (canonical_str, match_type)
        self._exact_map: Dict[Tuple[str, ...], Tuple[str, str]] = {}
        # Long multi-word identifiers eligible for constrained edit-distance<=1 fuzzy snap
        self._fuzzy_candidates: List[Tuple[Tuple[str, ...], str, str]] = []
        self._max_ngram: int = 2
        self._build_index()

    def _register(self, tokens: Sequence[str], canonical: str, match_type: str) -> None:
        tok_tuple = tuple(t.lower() for t in tokens if t)
        if not tok_tuple or len(tok_tuple) > 7:
            return
        if len(tok_tuple) == 1 and tok_tuple[0] in COMMON_ENGLISH_BLOCKLIST:
            return
        if tok_tuple not in self._exact_map:
            self._exact_map[tok_tuple] = (canonical, match_type)
            if len(tok_tuple) > self._max_ngram:
                self._max_ngram = len(tok_tuple)
            joined = " ".join(tok_tuple)
            if len(tok_tuple) >= 2 and len(joined) >= 10:
                self._fuzzy_candidates.append((tok_tuple, joined, canonical))

    def _build_index(self) -> None:
        # 1. File paths & basenames (with spoken extension variants like "dot py" / "dot pie")
        for fp in self.bundle.file_paths:
            p = Path(fp)
            base = p.name
            stem_words = split_identifier_to_words(p.stem)
            ext = p.suffix.lower()
            if not stem_words:
                continue
            ext_variants = EXTENSION_SPOKEN_MAP.get(ext, ())
            for variant in ext_variants:
                v_words = variant.split()
                # Prefer relative path if it's in tests/ or has a distinguishing parent; else basename
                target = fp if fp.startswith("tests/") else base
                self._register(stem_words + v_words, target, "file")
                # Also register full path spoken form ("tests test immune bbb dot py")
                if p.parent.name and p.parent.name not in (".", "src"):
                    parent_words = split_identifier_to_words(p.parent.name)
                    if parent_words:
                        self._register(parent_words + stem_words + v_words, fp, "file")

        # 2. Priority-ordered AST symbols (P0 > P1 > P2)
        for sym in self.bundle.p0_focus_symbols + self.bundle.p1_git_symbols + self.bundle.p2_repo_symbols:
            words = split_identifier_to_words(sym)
            if not words:
                continue
            if len(words) == 1:
                # Single-word identifiers only snap if distinctive (>=6 chars, not common English)
                if len(sym) >= 6 and sym.lower() not in COMMON_ENGLISH_BLOCKLIST:
                    self._register(words, sym, "exact")
            else:
                self._register(words, sym, "exact")

    @staticmethod
    def _strip_punct(word: str) -> Tuple[str, str, str]:
        """Split a token into (leading_punct, core_alnum, trailing_punct)."""
        m = re.match(r"^([^A-Za-z0-9_]*)(.*?)([^A-Za-z0-9_]*)$", word)
        if not m:
            return "", word, ""
        return m.group(1), m.group(2), m.group(3)

    def snap(self, raw_text: str) -> Tuple[str, List[SnapReceipt]]:
        """
        Apply Stage-2 AST symbol snapping + PhoneticNormalizer to a raw transcript.
        Returns `(normalized_text, list_of_snap_receipts)`.
        """
        if not raw_text or not raw_text.strip():
            return "", []

        # Pre-normalize spoken extensions ("dot pie" -> "dot pie" preserved for file snap)
        words = raw_text.strip().split()
        cores: List[str] = []
        leads: List[str] = []
        trails: List[str] = []
        for w in words:
            ld, core, tr = self._strip_punct(w)
            leads.append(ld)
            cores.append(core.lower())
            trails.append(tr)

        out_tokens: List[str] = []
        receipts: List[SnapReceipt] = []
        i = 0
        n = len(words)

        while i < n:
            matched = False
            max_span = min(self._max_ngram, n - i)
            # Longest-match-first exact lookup
            for span in range(max_span, 0, -1):
                # Do not cross sentence-ending punctuation inside a multi-word span
                if span > 1 and any(trails[k] and trails[k][-1] in ".!?;," for k in range(i, i + span - 1)):
                    continue
                key = tuple(cores[i : i + span])
                if key in self._exact_map:
                    canonical, match_type = self._exact_map[key]
                    # For 1-word matches, require code cue if lowercase-only
                    if span == 1 and canonical.islower():
                        prev_core = cores[i - 1] if i > 0 else ""
                        if prev_core not in CODE_CUE_WORDS:
                            continue
                    spoken_phrase = " ".join(words[i : i + span])
                    out_tokens.append(f"{leads[i]}{canonical}{trails[i + span - 1]}")
                    if spoken_phrase != canonical:
                        receipts.append(SnapReceipt(spoken=spoken_phrase, snapped=canonical, match_type=match_type))
                    i += span
                    matched = True
                    break

            if matched:
                continue

            # Constrained fuzzy match (edit distance <= 1 on >=10-char multi-word identifiers)
            for span in range(min(5, n - i), 1, -1):
                if any(trails[k] and trails[k][-1] in ".!?;," for k in range(i, i + span - 1)):
                    continue
                cand_tuple = tuple(cores[i : i + span])
                cand_joined = " ".join(cand_tuple)
                if len(cand_joined) < 10:
                    continue
                for sym_tuple, sym_joined, canonical in self._fuzzy_candidates:
                    if len(sym_tuple) != span:
                        continue
                    # First word must share initial letter to avoid spurious cross-word snaps
                    if sym_tuple[0][0] != cand_tuple[0][0]:
                        continue
                    if _levenshtein_bounded(cand_joined, sym_joined, max_dist=1) <= 1:
                        spoken_phrase = " ".join(words[i : i + span])
                        out_tokens.append(f"{leads[i]}{canonical}{trails[i + span - 1]}")
                        receipts.append(SnapReceipt(spoken=spoken_phrase, snapped=canonical, match_type="fuzzy"))
                        i += span
                        matched = True
                        break
                if matched:
                    break

            if not matched:
                out_tokens.append(words[i])
                i += 1

        snapped_text = " ".join(out_tokens)
        final_text = PhoneticNormalizer.normalize(snapped_text)
        return final_text, receipts
