"""
Pal Live Edit Mode - Local Site File Resolver & Surgical Local Edit Engine.

Provides:
1. LocalSiteFileResolver: Maps browser selections (URL, window title, selected text,
   surrounding text) to exact local source files and line indices in <15ms.
2. LocalEditEngine: Executes single-line and multi-line surgical edits using
   sub-2ms deterministic fast-path rules and on-device local LLMs (Metal MLX / Ollama).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

from whole_body_compute.integrations.pal_editor_rules import (
    DEFAULT_WORKSPACE_ROOTS,
    _normalize_whitespace_and_entities,
    apply_color_change_to_line,
    apply_smart_replacement_in_line,
    extract_relational_line_actions,
    pal_log,
    parse_fast_path_edit_instruction,
)


class LocalSiteFileResolver:
    """
    Maps a live browser selection (URL, window title, selected text, surrounding text)
    to the exact local source file and line range on disk.
    Optimized with root prioritization, sub-millisecond direct entry-point checks,
    case-insensitive tag-stripped matching, and blazing-fast ripgrep fallback (< 15ms).
    """

    TEXT_EXTENSIONS = (
        ".html",
        ".htm",
        ".tsx",
        ".jsx",
        ".vue",
        ".svelte",
        ".md",
        ".mdx",
        ".js",
        ".ts",
        ".css",
    )
    COMMON_ENTRY_POINTS = (
        "site/index.html",
        "dist_cloudflare/index.html",
        "index.html",
        "public/index.html",
        "src/index.html",
        "dist/index.html",
        "site/styles.css",
        "dist_cloudflare/styles.css",
        "styles.css",
        "site/app.js",
        "dist_cloudflare/app.js",
        "app.js",
        "site/main.js",
        "main.js",
        "src/App.tsx",
        "src/App.vue",
        "src/App.jsx",
        "dashboard/index.html",
        "site/dashboard/index.html",
        "dist_cloudflare/dashboard/index.html",
    )
    SKIP_DIRS = {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        "dist",
        "build",
        ".next",
        ".pytest_cache",
        ".ruff_cache",
    }

    def __init__(self, workspace_roots: Optional[List[Path]] = None):
        if workspace_roots is not None:
            self.roots = [Path(p) for p in workspace_roots if Path(p).exists()]
        else:
            self.roots = [p for p in DEFAULT_WORKSPACE_ROOTS if p.exists()]

    def _prioritize_roots(self, window_title: str = "", page_url: str = "") -> List[Path]:
        """Sort workspace roots so the project matching window_title or page_url is checked first."""
        scored_roots: List[Tuple[float, Path]] = []
        w_lower = (window_title or "").lower()
        u_lower = (page_url or "").lower()

        for r in self.roots:
            score = 0.0
            r_name = r.name.lower()
            if r_name in w_lower:
                score += 100.0
            if r_name in u_lower:
                score += 50.0

            if "enterprise_corpus" in w_lower and r_name == "enterprise_corpus":
                score += 150.0
            elif "vifi" in w_lower and r_name == "vifi.co":
                score += 150.0
            elif "whole_body_compute.org" in w_lower and r_name == "whole_body_compute.org":
                score += 150.0
            elif "voicefi" in w_lower and "enterprise_corpus" not in w_lower and r_name == "voicefi":
                score += 100.0

            if "8080" in u_lower and r_name == "enterprise_corpus":
                score += 80.0
            elif "5141" in u_lower and r_name == "voicefi":
                score += 80.0
            elif "3000" in u_lower and r_name in ("vifi.co", "whole_body_compute.org"):
                score += 40.0

            scored_roots.append((score, r))

        scored_roots.sort(key=lambda item: item[0], reverse=True)
        return [r for _, r in scored_roots]

    def find_best_file_and_line(
        self,
        selected_text: str,
        surrounding_text: str = "",
        page_url: str = "",
        window_title: str = "",
        explicit_subtarget: Optional[str] = None,
    ) -> Optional[Tuple[Path, int]]:
        """
        Locate the best matching local file and 0-based line index for the given selection.
        """
        t0 = time.perf_counter()
        sel_clean = (selected_text or "").strip()
        pal_log(
            f"🔍 Resolving file for selection: '{sel_clean[:40]}' | explicit: '{explicit_subtarget or ''}' | win: '{window_title[:35]}' | url: '{page_url}'"
        )

        # 1. Check direct file:// URL
        if page_url and page_url.startswith("file://"):
            parsed = urllib.parse.urlparse(page_url)
            local_p = Path(urllib.parse.unquote(parsed.path))
            if local_p.is_file():
                line_idx, score = self._score_file_match(
                    local_p,
                    selected_text=sel_clean,
                    surrounding_text=surrounding_text,
                    url_path="",
                    url_basename="",
                    title_clean="",
                    explicit_subtarget=explicit_subtarget,
                )
                if line_idx is not None and score > 0:
                    ms = (time.perf_counter() - t0) * 1000
                    pal_log(
                        f"✅ Direct file:// URL matched: {local_p}:{line_idx+1} in {ms:.1f}ms (score: {score:.1f})"
                    )
                    return (local_p, line_idx)

        # 2. Check VoiceFi Companion localhost:5141 routes
        url_path = ""
        if page_url:
            try:
                parsed = urllib.parse.urlparse(page_url)
                url_path = (parsed.path or "").strip("/")
                if "5141" in (parsed.netloc or ""):
                    static_dir = (
                        Path.home()
                        / "Projects"
                        / "VoiceFi"
                        / "src"
                        / "voicefi"
                        / "companion"
                        / "static"
                    )
                    route_map = {
                        "": static_dir / "index.html",
                        "training": static_dir / "training_mode.html",
                        "studio": static_dir / "studio.html",
                        "factory": static_dir / "factory.html",
                        "pair": static_dir / "pair.html",
                    }
                    mapped = route_map.get(url_path)
                    if mapped and mapped.is_file():
                        line_idx, score = self._score_file_match(
                            mapped,
                            selected_text=sel_clean,
                            surrounding_text=surrounding_text,
                            url_path=url_path,
                            url_basename=mapped.name.lower(),
                            title_clean="",
                            explicit_subtarget=explicit_subtarget,
                        )
                        if line_idx is not None and score > 0:
                            ms = (time.perf_counter() - t0) * 1000
                            pal_log(
                                f"✅ Companion route matched: {mapped}:{line_idx+1} in {ms:.1f}ms (score: {score:.1f})"
                            )
                            return (mapped, line_idx)
            except Exception:
                pass

        # 3. Prioritize workspace roots
        sorted_roots = self._prioritize_roots(window_title, page_url)
        url_basename = Path(url_path).name.lower() if url_path else ""
        title_clean = (
            _normalize_whitespace_and_entities(window_title).lower() if window_title else ""
        )
        pal_log(f"📂 Workspace search order: {[r.name for r in sorted_roots]}")

        # 4. Phase 1: Sub-millisecond direct entry point check on prioritized roots
        best_direct: Optional[Tuple[float, Path, int]] = None
        for root in sorted_roots:
            candidates_to_check: List[Path] = []
            if url_path:
                candidates_to_check.extend(
                    [
                        root / "site" / url_path / "index.html",
                        root / url_path / "index.html",
                        root / "public" / url_path / "index.html",
                        root / f"{url_path}.html",
                        root / "site" / f"{url_path}.html",
                    ]
                )
            for ep in self.COMMON_ENTRY_POINTS:
                candidates_to_check.append(root / ep)

            for cand in candidates_to_check:
                if cand.is_file():
                    line_idx, score = self._score_file_match(
                        cand,
                        selected_text=sel_clean,
                        surrounding_text=surrounding_text,
                        url_path=url_path,
                        url_basename=url_basename,
                        title_clean=title_clean,
                        explicit_subtarget=explicit_subtarget,
                    )
                    if line_idx is not None and score > 0:
                        pal_log(
                            f"  Direct check candidate: {cand.name} line {line_idx+1} (score: {score:.1f})"
                        )
                        if best_direct is None or score > best_direct[0]:
                            best_direct = (score, cand, line_idx)
                            if score >= 35.0:
                                break
            if best_direct is not None and best_direct[0] >= 35.0:
                ms = (time.perf_counter() - t0) * 1000
                pal_log(
                    f"⚡ Direct candidate hit in {ms:.1f}ms: {best_direct[1]}:{best_direct[2]+1} (score: {best_direct[0]:.1f})"
                )
                return (best_direct[1], best_direct[2])

        # 5. Phase 2: Ripgrep fast indexed search across roots (< 15ms)
        rg_match = self._find_via_ripgrep(
            sorted_roots,
            selected_text=sel_clean,
            surrounding_text=surrounding_text,
            url_path=url_path,
            url_basename=url_basename,
            title_clean=title_clean,
            explicit_subtarget=explicit_subtarget,
        )
        if rg_match is not None:
            ms = (time.perf_counter() - t0) * 1000
            pal_log(
                f"⚡ Ripgrep match in {ms:.1f}ms: {rg_match[0]}:{rg_match[1]+1} (score: {rg_match[2]:.1f})"
            )
            return (rg_match[0], rg_match[1])

        if best_direct is not None:
            ms = (time.perf_counter() - t0) * 1000
            pal_log(
                f"✅ Using best direct match: {best_direct[1]}:{best_direct[2]+1} in {ms:.1f}ms (score: {best_direct[0]:.1f})"
            )
            return (best_direct[1], best_direct[2])

        # 6. Phase 3: Bounded shallow walk fallback (max depth 3, max 60 files)
        shallow_match = self._shallow_walk_fallback(
            sorted_roots,
            selected_text=sel_clean,
            surrounding_text=surrounding_text,
            url_path=url_path,
            url_basename=url_basename,
            title_clean=title_clean,
            explicit_subtarget=explicit_subtarget,
        )
        if shallow_match is not None:
            ms = (time.perf_counter() - t0) * 1000
            pal_log(
                f"✅ Shallow walk match in {ms:.1f}ms: {shallow_match[0]}:{shallow_match[1]+1} (score: {shallow_match[2]:.1f})"
            )
            return (shallow_match[0], shallow_match[1])

        ms = (time.perf_counter() - t0) * 1000
        pal_log(
            f"❌ File resolution found no match for '{sel_clean[:40]}' after {ms:.1f}ms. Roots: {[r.name for r in sorted_roots]}",
            level="warning",
        )
        return None

    def _find_via_ripgrep(
        self,
        roots: List[Path],
        selected_text: str,
        surrounding_text: str,
        url_path: str,
        url_basename: str,
        title_clean: str,
        explicit_subtarget: Optional[str] = None,
    ) -> Optional[Tuple[Path, int, float]]:
        """Use ripgrep to rapidly locate candidate files matching selected_text or significant words (< 15ms)."""
        rg_bin = shutil.which("rg") or "/opt/homebrew/bin/rg"
        if not os.path.isfile(rg_bin):
            return None

        sel_clean = (selected_text or "").strip()
        tokens = [
            w
            for w in re.split(r"[\s•—–;:,\"]+", sel_clean)
            if len(w) >= 3 and w.lower() not in {"the", "and", "for", "with", "this", "that"}
        ]

        search_terms: List[str] = []
        if len(tokens) >= 2:
            search_terms.append(f"{tokens[0]} {tokens[1]}")
        if tokens:
            search_terms.append(tokens[0])
        if explicit_subtarget and len(explicit_subtarget) > 1:
            search_terms.append(explicit_subtarget)

        best_match: Optional[Tuple[float, Path, int]] = None

        for root in roots:
            if not root.is_dir():
                continue
            for term in search_terms:
                cmd = [
                    rg_bin,
                    "-i",
                    "-l",
                    "--max-count",
                    "5",
                    "-g",
                    "*.html",
                    "-g",
                    "*.htm",
                    "-g",
                    "*.tsx",
                    "-g",
                    "*.jsx",
                    "-g",
                    "*.vue",
                    "-g",
                    "*.svelte",
                    "-g",
                    "*.js",
                    "-g",
                    "*.ts",
                    "-g",
                    "*.css",
                    "-g",
                    "!node_modules",
                    "-g",
                    "!.git",
                    "-g",
                    "!.venv",
                    "-g",
                    "!dist",
                    "-g",
                    "!build",
                    term,
                    str(root),
                ]
                try:
                    res = subprocess.run(cmd, capture_output=True, text=True, timeout=2.0)
                    found_paths = [Path(p.strip()) for p in res.stdout.splitlines() if p.strip()]
                    for fpath in found_paths:
                        if fpath.is_file():
                            line_idx, score = self._score_file_match(
                                fpath,
                                selected_text=sel_clean,
                                surrounding_text=surrounding_text,
                                url_path=url_path,
                                url_basename=url_basename,
                                title_clean=title_clean,
                                explicit_subtarget=explicit_subtarget,
                            )
                            if line_idx is not None and score > 0:
                                if best_match is None or score > best_match[0]:
                                    best_match = (score, fpath, line_idx)
                                    if score >= 35.0:
                                        return (best_match[1], best_match[2], best_match[0])
                except Exception:
                    pass

        if best_match is not None and best_match[0] > 0:
            return (best_match[1], best_match[2], best_match[0])
        return None

    def _shallow_walk_fallback(
        self,
        roots: List[Path],
        selected_text: str,
        surrounding_text: str,
        url_path: str,
        url_basename: str,
        title_clean: str,
        explicit_subtarget: Optional[str] = None,
    ) -> Optional[Tuple[Path, int, float]]:
        """Bounded shallow walk fallback: max depth 3, max 60 files checked."""
        best_match: Optional[Tuple[float, Path, int]] = None
        files_checked = 0

        for root in roots:
            root_depth = len(root.parts)
            for dirpath, dirnames, filenames in os.walk(root):
                curr_depth = len(Path(dirpath).parts) - root_depth
                if curr_depth > 3:
                    dirnames.clear()
                    continue
                dirnames[:] = [
                    d for d in dirnames if d not in self.SKIP_DIRS and not d.startswith(".")
                ]
                for fname in filenames:
                    if fname.endswith(self.TEXT_EXTENSIONS):
                        fpath = Path(dirpath) / fname
                        files_checked += 1
                        line_idx, score = self._score_file_match(
                            fpath,
                            selected_text=selected_text,
                            surrounding_text=surrounding_text,
                            url_path=url_path,
                            url_basename=url_basename,
                            title_clean=title_clean,
                            explicit_subtarget=explicit_subtarget,
                        )
                        if line_idx is not None and score > 0:
                            if best_match is None or score > best_match[0]:
                                best_match = (score, fpath, line_idx)
                                if score >= 40.0:
                                    return (best_match[1], best_match[2], best_match[0])
                        if files_checked >= 60:
                            break
                if files_checked >= 60:
                    break
            if files_checked >= 60:
                break

        if best_match is not None and best_match[0] > 0:
            return (best_match[1], best_match[2], best_match[0])
        return None

    def _score_file_match(
        self,
        fpath: Path,
        selected_text: str,
        surrounding_text: str,
        url_path: str,
        url_basename: str,
        title_clean: str,
        explicit_subtarget: Optional[str] = None,
    ) -> Tuple[Optional[int], float]:
        try:
            if fpath.stat().st_size > 2_000_000:
                return None, 0.0
            content = fpath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None, 0.0

        if not content:
            return None, 0.0

        file_bonus = 0.0
        fname_lower = fpath.name.lower()
        fpath_str_lower = str(fpath).lower()

        if url_basename:
            if fname_lower == url_basename or fname_lower == f"{url_basename}.html":
                file_bonus += 50.0
            elif url_basename in fpath_str_lower:
                file_bonus += 25.0
        elif page_is_root := (url_path == "" and fname_lower == "index.html"):
            if page_is_root:
                file_bonus += 15.0

        if title_clean:
            m_title = re.search(r"<title[^>]*>(.*?)</title>", content, re.IGNORECASE | re.DOTALL)
            if m_title:
                file_title = _normalize_whitespace_and_entities(m_title.group(1)).lower()
                if file_title and (file_title in title_clean or title_clean in file_title):
                    file_bonus += 40.0

        sel_clean = (selected_text or "").strip()
        surr_clean = _normalize_whitespace_and_entities(surrounding_text) if surrounding_text else ""
        sel_norm = _normalize_whitespace_and_entities(sel_clean) if sel_clean else ""
        sel_lower = sel_norm.lower() if sel_norm else ""

        lines = content.splitlines()
        best_line_idx: Optional[int] = None
        best_line_score = 0.0

        surr_words = [
            w.lower()
            for w in re.findall(r"[a-zA-Z0-9]{3,}", surr_clean)
            if w.lower() not in {"the", "and", "for", "with", "that", "this", "from", "your"}
        ]

        for i, line in enumerate(lines):
            if not line.strip():
                continue

            line_score = 0.0
            line_norm = _normalize_whitespace_and_entities(line)
            line_lower = line_norm.lower()

            has_sel = False
            if sel_clean and (sel_clean in line or (sel_norm and sel_norm in line_norm)):
                has_sel = True
                line_score += 40.0 + min(len(sel_clean), 60) * 0.5
            elif sel_lower and (sel_lower in line_lower):
                has_sel = True
                line_score += 35.0 + min(len(sel_clean), 60) * 0.5
            else:
                line_no_tags = re.sub(r"<[^>]+>", " ", line)
                line_no_tags_norm = _normalize_whitespace_and_entities(line_no_tags).lower()
                if sel_lower and sel_lower in line_no_tags_norm:
                    has_sel = True
                    line_score += 35.0 + min(len(sel_clean), 60) * 0.5
                elif explicit_subtarget and (
                    explicit_subtarget in line
                    or explicit_subtarget in line_norm
                    or explicit_subtarget.lower() in line_lower
                ):
                    has_sel = True
                    line_score += 25.0
                elif sel_lower:
                    sel_trimmed = sel_lower.strip(" \t\r\n.,;:!?-—–•")
                    if (
                        len(sel_trimmed) >= 4
                        and sel_trimmed in line_no_tags_norm
                        and (sel_lower.endswith("•") or "•" in sel_lower or "—" in sel_lower)
                    ):
                        has_sel = True
                        line_score += 30.0

            if not has_sel:
                continue

            if surr_clean:
                window_raw = " ".join(lines[max(0, i - 2) : min(len(lines), i + 3)])
                window_norm = _normalize_whitespace_and_entities(window_raw)
                if surr_clean in line_norm:
                    line_score += 100.0
                elif surr_clean in window_norm:
                    line_score += 80.0
                elif surr_words:
                    window_lower = window_norm.lower()
                    matched_words = sum(1 for w in surr_words if w in window_lower)
                    if matched_words > 0:
                        overlap_ratio = matched_words / len(surr_words)
                        line_score += overlap_ratio * 70.0

            total_score = file_bonus + line_score
            if total_score > best_line_score:
                best_line_score = total_score
                best_line_idx = i

        return best_line_idx, best_line_score


class LocalEditEngine:
    """
    Executes surgical single-line or snippet edits using:
      1. Sub-2ms deterministic fast-path rules
      2. On-device local LLM (Ollama qwen2.5-coder:1.5b / gemma2:2b or LiteRT)
    """

    OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
    LITERT_URL = "http://127.0.0.1:9379/v1/chat/completions"
    SPECIALIST_ROUTER_URL = "http://127.0.0.1:9380/v1/chat/completions"

    @classmethod
    def apply_edit_to_line(
        cls,
        line_text: str,
        selected_text: str,
        instruction: str,
    ) -> Tuple[Optional[str], str]:
        """
        Apply `instruction` to `line_text` focused on `selected_text`.
        Supports both single instructions and multi-sentence instructions ("clause 1 • clause 2").
        Returns (new_line_text, engine_used) where engine_used is 'fast_path' or 'local_llm'.
        """
        raw_clauses = re.split(
            r"(?:[•;]|\.\s+|\s+(?:and\s+then|and\s+also|and\s+(?=(?:remove|delete|drop|erase|add|put|insert|pad|change|replace|make|turn|swap|break)))\s+)",
            instruction or "",
            flags=re.IGNORECASE,
        )
        clauses = [c.strip() for c in raw_clauses if c and c.strip()]
        if len(clauses) > 1:
            curr_line = line_text
            all_fast = True
            for clause in clauses:
                step_line, _ = cls._apply_single_clause_fast_path(
                    curr_line, selected_text, clause
                )
                if step_line is not None and step_line != curr_line:
                    curr_line = step_line
                else:
                    all_fast = False
                    break
            if all_fast and curr_line != line_text:
                return curr_line, "fast_path"
        else:
            fast_line, fast_engine = cls._apply_single_clause_fast_path(
                line_text, selected_text, instruction
            )
            if fast_line is not None and fast_line != line_text:
                return fast_line, fast_engine

        # 2. Fallback to On-Device Local Model (Ollama / LiteRT) for creative/multi-part edits
        llm_updated = cls._call_local_llm_line_edit(
            line_text=line_text,
            selected_text=selected_text,
            instruction=instruction,
        )
        if llm_updated is not None and llm_updated != line_text:
            return llm_updated, "local_llm"

        return None, "none"

    @classmethod
    def _apply_single_clause_fast_path(
        cls,
        line_text: str,
        selected_text: str,
        instruction: str,
    ) -> Tuple[Optional[str], str]:
        parsed = parse_fast_path_edit_instruction(instruction, selected_text)
        if parsed is not None:
            op_type, explicit_src, rep_text = parsed
            if op_type == "delete_selection":
                updated = apply_smart_replacement_in_line(
                    line_text, selected_text, "", explicit_subtarget=explicit_src
                )
                if updated is not None and updated != line_text:
                    return updated, "fast_path"
            elif op_type == "change_color":
                at_end = explicit_src == "at_end"
                updated = apply_color_change_to_line(
                    line_text, selected_text, rep_text, at_end=at_end
                )
                if updated is not None and updated != line_text:
                    return updated, "fast_path"
            elif op_type == "transform_case":
                target = (selected_text or "").strip()
                if target and target in line_text:
                    if rep_text == "upper":
                        transformed = target.upper()
                    elif rep_text == "lower":
                        transformed = target.lower()
                    else:
                        transformed = target.title()
                    return line_text.replace(target, transformed, 1), "fast_path"
            elif op_type in ("add_line_break_before", "add_line_break_after"):
                target = (explicit_src or selected_text or "").strip()
                if target:
                    tag_before_re = re.compile(
                        rf"(<[a-zA-Z0-9]+[^>]*>\s*)?{re.escape(target)}", re.IGNORECASE
                    )
                    m = tag_before_re.search(line_text)
                    if m:
                        if op_type == "add_line_break_before":
                            idx = m.start()
                            return line_text[:idx] + "<br>" + line_text[idx:], "fast_path"
                        else:
                            idx = m.end()
                            return line_text[:idx] + "<br>" + line_text[idx:], "fast_path"
            elif op_type == "replace_selection":
                updated = apply_smart_replacement_in_line(
                    line_text, selected_text, rep_text, explicit_subtarget=explicit_src
                )
                if updated is not None and updated != line_text:
                    return updated, "fast_path"
        return None, "none"

    @classmethod
    def _call_local_llm_line_edit(
        cls,
        line_text: str,
        selected_text: str,
        instruction: str,
    ) -> Optional[str]:
        """
        Send a compact ~60-token prompt to the local model (LiteRT or Ollama)
        to rewrite `line_text` according to `instruction`.
        """
        leading_ws = line_text[: len(line_text) - len(line_text.lstrip(" \t"))]
        dot_ctx = ""
        try:
            from whole_body_compute.local.dot_memory import get_dot_store

            dot_ctx = get_dot_store().build_context_injection(
                f"{selected_text} {instruction}",
                top_k=2,
                categories=["lgtm_edit_lesson", "preference"],
            )
        except Exception:
            pass
        dot_block = f"{dot_ctx}\n\n" if dot_ctx else ""
        prompt = (
            "You are a surgical local code & copy editor. "
            "Modify the following single line of code/HTML according to the user's spoken instruction.\n"
            " Preserve all HTML tags, classes, and attributes unless instructed otherwise.\n"
            "Output ONLY the modified line with no markdown fences, no quotes, and no explanation.\n\n"
            f"{dot_block}"
            f"ORIGINAL LINE:\n{line_text.strip()}\n\n"
            f"SELECTED TEXT:\n{selected_text.strip()}\n\n"
            f"INSTRUCTION:\n{instruction.strip()}\n\n"
            "MODIFIED LINE:"
        )

        for router_url in (cls.SPECIALIST_ROUTER_URL, cls.LITERT_URL):
            try:
                payload = json.dumps(
                    {
                        "model": "localbench-specialist-router",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 200,
                    }
                ).encode("utf-8")
                req = urllib.request.Request(
                    router_url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        choices = data.get("choices") or []
                        if choices:
                            raw_out = (choices[0].get("message", {}).get("content") or "").strip()
                            cleaned = cls._clean_llm_single_line(raw_out, leading_ws)
                            if cleaned:
                                return cleaned
            except Exception:
                continue

        for model_name in (
            "voicefi-coder:1.5b",
            "qwen2.5-coder:1.5b",
            "gemma2:2b",
            "qwen2.5-coder:7b",
        ):
            try:
                payload = json.dumps(
                    {
                        "model": model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.1, "num_predict": 200},
                    }
                ).encode("utf-8")
                req = urllib.request.Request(
                    cls.OLLAMA_URL,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=4.0) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        raw_out = (data.get("response") or "").strip()
                        cleaned = cls._clean_llm_single_line(raw_out, leading_ws)
                        if cleaned:
                            return cleaned
            except Exception:
                continue

        return None

    @staticmethod
    def _clean_llm_single_line(raw_out: str, leading_ws: str) -> Optional[str]:
        if "<|im_end|>" in raw_out:
            raw_out = raw_out.split("<|im_end|>", 1)[0]
        cleaned = re.sub(r"<think>.*?</think>", "", raw_out, flags=re.DOTALL).strip()
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
        if not cleaned:
            return None
        first_line = cleaned.splitlines()[0].strip()
        if not first_line:
            return None
        return f"{leading_ws}{first_line}"

    @classmethod
    def _call_local_llm_multi_line_edit(
        cls,
        snippet_lines: List[str],
        focus_offset: int,
        selected_text: str,
        instruction: str,
    ) -> Optional[List[str]]:
        """
        Send a multi-line snippet to local LLM (tev1:latest / gemma2:2b / LiteRT)
        when edits require multi-element or contextual understanding beyond single-line rules.
        """
        snippet = "\n".join(snippet_lines)
        focus_line = snippet_lines[focus_offset] if 0 <= focus_offset < len(snippet_lines) else ""
        dot_ctx = ""
        try:
            from whole_body_compute.local.dot_memory import get_dot_store

            dot_ctx = get_dot_store().build_context_injection(
                f"{selected_text} {instruction}",
                top_k=2,
                categories=["lgtm_edit_lesson", "preference"],
            )
        except Exception:
            pass
        dot_block = f"{dot_ctx}\n\n" if dot_ctx else ""
        prompt = (
            "You are a precision surgical local code & copy editor.\n"
            "Modify the following code snippet according to the user's spoken instruction.\n"
            "The user highlighted text in the FOCUS LINE.\n"
            "Apply all requested changes (editing copy, modifying classes, removing lines or elements).\n"
            "Preserve enclosing container tags unless explicitly asked to delete them.\n"
            "Output ONLY the replacement code snippet with NO markdown fences, NO quotes, and NO explanations.\n\n"
            f"{dot_block}"
            f"ORIGINAL SNIPPET:\n{snippet}\n\n"
            f"FOCUS LINE:\n{focus_line.strip()}\n\n"
            f"USER HIGHLIGHTED:\n{selected_text.strip()}\n\n"
            f"INSTRUCTION:\n{instruction.strip()}\n\n"
            "REPLACEMENT SNIPPET:"
        )

        for router_url, model_id in (
            (cls.SPECIALIST_ROUTER_URL, "localbench-specialist-router"),
            (cls.LITERT_URL, "voicefi-coder-mlx"),
            (
                "http://127.0.0.1:8080/v1/chat/completions",
                "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
            ),
        ):
            try:
                max_tok = 360 if model_id == "voicefi-coder-mlx" else 400
                payload = json.dumps(
                    {
                        "model": model_id,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": max_tok,
                    }
                ).encode("utf-8")
                req = urllib.request.Request(
                    router_url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=4.0) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        choices = data.get("choices") or []
                        if choices:
                            raw_out = (choices[0].get("message", {}).get("content") or "").strip()
                            if "<|im_end|>" in raw_out:
                                raw_out = raw_out.split("<|im_end|>", 1)[0].strip()
                            if "</think>" in raw_out:
                                raw_out = raw_out.split("</think>", 1)[-1].strip()
                            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", raw_out)
                            cleaned = re.sub(r"\n?```$", "", cleaned).strip()
                            if cleaned:
                                return cleaned.splitlines()
            except Exception:
                continue

        for model_name in ("voicefi-coder:1.5b", "gemma2:2b", "tev1:latest"):
            try:
                payload = json.dumps(
                    {
                        "model": model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.1, "num_predict": 400},
                    }
                ).encode("utf-8")
                req = urllib.request.Request(
                    cls.OLLAMA_URL,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=6.0) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        raw_out = (data.get("response") or "").strip()
                        if "</think>" in raw_out:
                            raw_out = raw_out.split("</think>", 1)[-1].strip()
                        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", raw_out)
                        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
                        if cleaned:
                            return cleaned.splitlines()
            except Exception:
                continue

        return None

    @classmethod
    def apply_edit(
        cls,
        lines: List[str],
        line_idx: int,
        selected_text: str,
        instruction: str,
    ) -> Tuple[Optional[List[str]], int, int, str]:
        """
        Apply `instruction` to a file's lines array focused on `line_idx` and `selected_text`.
        Returns (replacement_chunk, start_idx, end_idx, engine_used).
        """
        if line_idx < 0 or line_idx >= len(lines):
            return None, 0, 0, "none"

        clean_inst = (instruction or "").strip()
        remaining_inst, offsets, remove_current = extract_relational_line_actions(clean_inst)

        if offsets or remove_current:
            new_focus_line = None
            engine_used = "relational_fast_path"

            if remaining_inst:
                step_line, step_engine = cls.apply_edit_to_line(
                    lines[line_idx], selected_text, remaining_inst
                )
                if step_line is not None and step_line != lines[line_idx]:
                    new_focus_line = step_line
                    engine_used = f"relational+{step_engine}"
                else:
                    new_focus_line = lines[line_idx]
            else:
                new_focus_line = "" if remove_current else lines[line_idx]

            active_offsets = set(offsets)
            active_offsets.add(0)

            min_offset = min(active_offsets)
            max_offset = max(active_offsets)

            start_idx = max(0, line_idx + min_offset)
            end_idx = min(len(lines), line_idx + max_offset + 1)

            replacement_chunk: List[str] = []
            for off in range(min_offset, max_offset + 1):
                idx = line_idx + off
                if idx < 0 or idx >= len(lines):
                    continue
                if off in offsets:
                    continue
                if off == 0:
                    if not remove_current:
                        if new_focus_line is not None and (
                            new_focus_line.strip() != "" or lines[line_idx].strip() == ""
                        ):
                            replacement_chunk.append(new_focus_line)
                else:
                    replacement_chunk.append(lines[idx])

            return replacement_chunk, start_idx, end_idx, engine_used

        new_line, engine_used = cls.apply_edit_to_line(
            lines[line_idx], selected_text, clean_inst
        )
        if new_line is not None and new_line != lines[line_idx]:
            if new_line.strip() == "" and lines[line_idx].strip() != "":
                return [], line_idx, line_idx + 1, engine_used
            return [new_line], line_idx, line_idx + 1, engine_used

        window_above = 3
        window_below = 4
        start_idx = max(0, line_idx - window_above)
        end_idx = min(len(lines), line_idx + window_below + 1)
        snippet_lines = lines[start_idx:end_idx]
        llm_chunk = cls._call_local_llm_multi_line_edit(
            snippet_lines, line_idx - start_idx, selected_text, clean_inst
        )
        if llm_chunk is not None:
            return llm_chunk, start_idx, end_idx, "local_llm_tev1"

        return None, line_idx, line_idx + 1, "none"
