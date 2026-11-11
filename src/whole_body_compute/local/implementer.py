"""
VoiceFi Recon Implementer Engine.
Executes an on-device 2-tier cascade (Gemma 4 2B Scout -> Gemma 4 26B Coder)
to surgically modify code and output clean unified diffs, saving >90% of cloud tokens.
"""

import os
import re
import time
import difflib
import logging
import json
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from whole_body_compute.local.engine import LocalModelEngine, is_litert_available
from whole_body_compute.local.scout import ReconScout, estimate_tokens

logger = logging.getLogger("whole_body_compute.local.implementer")

LOCAL_SERVER_URL = "http://127.0.0.1:9379/v1/chat/completions"


@dataclass
class ImplementResult:
    target_path: str
    instruction: str
    diff: str
    applied: bool
    scout_duration: float
    coder_duration: float
    total_duration: float
    tokens_saved: int
    savings_pct: float
    input_tokens_est: int
    diff_tokens_est: int
    model_scout: str
    model_coder: str
    scout_findings: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def apply_search_replace(
    content: str,
    search: str,
    replace: str,
    preferred_line: Optional[int] = None,
) -> Tuple[str, bool]:
    """Apply a SEARCH/REPLACE block to text content with indentation-aware fallback."""
    # Normalize line endings
    search_clean = search.replace("\r\n", "\n")
    replace_clean = replace.replace("\r\n", "\n")
    content_clean = content.replace("\r\n", "\n")

    padded_content = "\n" + content_clean + "\n"
    target_search = "\n" + search_clean + "\n"
    count = padded_content.count(target_search)

    if count == 1 or (count > 1 and preferred_line is None):
        if target_search in padded_content:
            return content_clean.replace(search_clean, replace_clean, 1), True
    elif count > 1 and preferred_line is not None:
        matches = [m.start() for m in re.finditer(re.escape(target_search), padded_content)]
        best_pos = None
        best_dist = float("inf")
        for pos in matches:
            line_num = padded_content[:pos].count("\n") + 1
            dist = abs(line_num - preferred_line)
            if dist < best_dist:
                best_dist = dist
                best_pos = pos
        if best_pos is not None:
            new_content = (
                content_clean[:best_pos]
                + replace_clean
                + content_clean[best_pos + len(search_clean) :]
            )
            return new_content, True

    # Fallback: strip leading/trailing blank lines in search block (requiring full-line boundary)
    search_stripped = search_clean.strip("\n")
    if search_stripped and ("\n" + search_stripped + "\n") in padded_content:
        return content_clean.replace(
            search_stripped, replace_clean.strip("\n"), 1
        ), True

    # Fallback: line-by-line whitespace-tolerant match with indentation preservation
    raw_search_lines = search_clean.strip("\n").splitlines()
    search_lines_stripped = [l.strip() for l in raw_search_lines]
    if not any(search_lines_stripped):
        return content, False

    content_lines = content_clean.splitlines()
    n_search = len(raw_search_lines)
    for i in range(len(content_lines) - n_search + 1):
        window = [content_lines[i + j].strip() for j in range(n_search)]
        if window == search_lines_stripped:
            # Determine indentation delta from the first non-empty line
            indent_delta = 0
            for j in range(n_search):
                if search_lines_stripped[j]:
                    orig_indent = len(content_lines[i + j]) - len(content_lines[i + j].lstrip(" "))
                    search_indent = len(raw_search_lines[j]) - len(raw_search_lines[j].lstrip(" "))
                    indent_delta = orig_indent - search_indent
                    break

            rep_lines = replace_clean.strip("\n").splitlines()
            if indent_delta > 0:
                adjusted_rep = [
                    (" " * indent_delta + rl) if rl.strip() else rl
                    for rl in rep_lines
                ]
            elif indent_delta < 0:
                trim = abs(indent_delta)
                adjusted_rep = [
                    rl[trim:] if rl.startswith(" " * trim) else rl
                    for rl in rep_lines
                ]
            else:
                adjusted_rep = rep_lines

            new_lines = content_lines[:i] + adjusted_rep + content_lines[i + n_search :]
            return "\n".join(new_lines) + ("\n" if content_clean.endswith("\n") else ""), True

    return content, False


def parse_search_replace_blocks(text: str) -> List[Tuple[str, str]]:
    """Parse <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE blocks (stripping <think> tags first)."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    pattern = re.compile(
        r"<{3,}\s*SEARCH\s*\n(.*?)\n={3,}\s*\n(.*?)\n>{3,}\s*REPLACE",
        re.DOTALL,
    )
    blocks = []
    for match in pattern.finditer(cleaned):
        blocks.append((match.group(1), match.group(2)))
    if not blocks:
        # Fallback for small distilled models (1.5B) that close the REPLACE section with ``` or EOF
        fallback_pattern = re.compile(
            r"<{3,}\s*SEARCH\s*\n(.*?)\n={3,}\s*\n(.*?)(?:\n>{3,}\s*\w*|\n```|\Z)",
            re.DOTALL,
        )
        for match in fallback_pattern.finditer(cleaned):
            s_part = match.group(1).strip("\n")
            r_part = re.sub(r"\n```\s*\Z", "", match.group(2)).strip("\n")
            if s_part:
                blocks.append((s_part, r_part))
    return blocks


def extract_line_hint_or_keyword_window(
    original_content: str,
    instruction: str,
    window_radius: int = 80,
    large_file_threshold: int = 250,
    top_window_threshold: int = 80,
    file_path: Optional[Union[str, Path]] = None,
) -> Optional[str]:
    """
    Extract a focused verbatim code window (±window_radius lines) around a target line
    identified by explicit line number hints or literal identifiers in `instruction`.
    Designed to prevent context truncation and ensure SEARCH/REPLACE blocks on large
    files (>250 lines) match verbatim against original full file content.
    """
    lines = original_content.splitlines()
    total_lines = len(lines)
    if total_lines <= top_window_threshold:
        return None

    target_line: Optional[int] = None

    # 1. Explicit line number hints: (line|lines|L) 1201 or lines 1200-1250
    line_pattern = re.compile(
        r"(?:line|lines|L)\s*(\d+)(?:\s*[-–to]+\s*(\d+))?",
        re.IGNORECASE,
    )
    for m in line_pattern.finditer(instruction):
        try:
            l1 = int(m.group(1))
            l2 = int(m.group(2)) if m.group(2) else l1
            mid = (l1 + l2) // 2
            if 1 <= mid <= total_lines:
                target_line = mid
                break
        except (ValueError, TypeError):
            continue

    # 2. Exact code/identifier literals in backticks or quotes
    file_basename = Path(file_path).name.lower() if file_path else ""
    backticked = re.findall(r"`([^`]+)`", instruction)
    quoted_raw = re.findall(r'"([^"]+)"|\'([^\']+)\'', instruction)
    quoted = [q1 or q2 for q1, q2 in quoted_raw]
    literals = [
        lit.strip()
        for lit in backticked + quoted
        if lit.strip() and lit.strip().lower() != file_basename and len(lit.strip()) >= 2
    ]
    literals.sort(key=len, reverse=True)

    if target_line is None and literals:
        for lit in literals:
            matched_indices = [idx for idx, line in enumerate(lines) if lit in line]
            if matched_indices:
                def_matches = [
                    i
                    for i in matched_indices
                    if any(kw in lines[i] for kw in ("def ", "class ", "async def "))
                ]
                target_line = (def_matches[0] if def_matches else matched_indices[0]) + 1
                break

    if target_line is None:
        return None

    # Only apply window slicing if file is large (>250 lines) or target is beyond top-of-file window
    if total_lines > large_file_threshold or target_line > top_window_threshold:
        w_start = max(0, target_line - 1 - window_radius)
        w_end = min(total_lines, target_line - 1 + window_radius + 1)
        slice_lines = lines[w_start:w_end]
        return "\n".join(slice_lines)

    return None


def extract_deterministic_symbol_slice(
    original_content: str,
    instruction: str,
    context_before: int = 4,
    context_after: int = 24,
    max_slice_lines: int = 90,
    file_path: Optional[Union[str, Path]] = None,
    min_lines: int = 80,
) -> Optional[str]:
    """
    Tier-1 Deterministic AST & Identifier Slicer (<1ms, 0 LLM tokens).
    Extracts verbatim code windows around functions, classes, routes, or identifiers
    referenced in `instruction` so the local coder model receives a <300-token slice
    instead of a 10,000-token full file dump. Enriches matched symbols with 1-hop
    callee signatures and DotMemory warnings.
    """
    import ast as _ast

    lines = original_content.splitlines()
    if len(lines) <= min_lines and not file_path:
        return None

    # For large files (>250 lines) or explicit line hints, check focused window slicing first
    if len(lines) > 250:
        window = extract_line_hint_or_keyword_window(
            original_content,
            instruction,
            window_radius=context_after + 56,
            file_path=file_path,
        )
        if window:
            return window

    # Extract candidate identifiers, backticked tokens, and route paths from instruction
    backticked = re.findall(r"`([^`]+)`", instruction)
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', instruction)
    identifiers = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]{3,})\b", instruction)

    stop_words = {
        "async", "def", "class", "return", "import", "from", "with", "inside",
        "after", "before", "right", "function", "method", "file", "add", "update",
        "modify", "implement", "request", "response", "true", "false", "none",
        "that", "this", "where", "when", "into", "block", "argument", "parser",
    }

    anchors: List[str] = []
    for tok in backticked:
        t = tok.strip()
        if len(t) >= 3 and t not in anchors:
            anchors.append(t)
    for q1, q2 in quoted:
        t = (q1 or q2).strip()
        if len(t) >= 3 and t not in anchors:
            anchors.append(t)
    for ident in identifiers:
        if ident.lower() not in stop_words and ident not in anchors:
            if "_" in ident or any(c.isupper() for c in ident[1:]):
                anchors.append(ident)

    if not anchors:
        return None

    # Prioritize longest/most specific anchors first so specific identifiers match before generic flags
    anchors.sort(key=len, reverse=True)

    matched_ranges: List[Tuple[int, int]] = []
    matched_symbol_names: List[str] = []
    node_callees: List[str] = []

    # 1. AST node span matching for exact function/class names
    try:
        tree = _ast.parse(original_content)
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)) and hasattr(node, "end_lineno"):
                if node.name in anchors:
                    start = max(0, node.lineno - 1 - context_before)
                    end = min(len(lines), (node.end_lineno or node.lineno) + context_after)
                    matched_ranges.append((start, end))
                    if node.name not in matched_symbol_names:
                        matched_symbol_names.append(node.name)
                        for sub in _ast.walk(node):
                            if isinstance(sub, _ast.Call):
                                c_name = ""
                                if isinstance(sub.func, _ast.Name):
                                    c_name = sub.func.id
                                elif isinstance(sub.func, _ast.Attribute):
                                    c_name = sub.func.attr
                                if c_name and c_name not in matched_symbol_names and c_name not in node_callees:
                                    node_callees.append(c_name)
    except SyntaxError:
        pass

    # 2. Line-level substring matching for variables, routes, or subparsers (e.g. p_context)
    for idx, line in enumerate(lines):
        for anc in anchors[:8]:
            if anc in line:
                start = max(0, idx - context_before)
                end = min(len(lines), idx + 12)
                matched_ranges.append((start, end))
                break

    if not matched_ranges:
        return None

    # Prefer narrow line-level matches over giant whole-function AST spans (>35 lines)
    narrow_ranges = [r for r in matched_ranges if (r[1] - r[0]) <= 35]
    if narrow_ranges:
        matched_ranges = narrow_ranges

    # Merge overlapping line ranges
    matched_ranges.sort(key=lambda r: r[0])
    merged: List[Tuple[int, int]] = [matched_ranges[0]]
    for cur_s, cur_e in matched_ranges[1:]:
        prev_s, prev_e = merged[-1]
        if cur_s <= prev_e + 3:
            merged[-1] = (prev_s, max(prev_e, cur_e))
        else:
            merged.append((cur_s, cur_e))

    slices: List[str] = []
    total_collected = 0
    for s_idx, e_idx in merged:
        chunk = lines[s_idx:e_idx]
        if total_collected + len(chunk) > max_slice_lines:
            remaining = max(10, max_slice_lines - total_collected)
            chunk = chunk[:remaining]
        slices.append("\n".join(chunk))
        total_collected += len(chunk)
        if total_collected >= max_slice_lines:
            break

    body_text = "\n\n".join(slices) if slices else None
    if not body_text:
        return None

    # Query LocalSymbolIndex and DotMemory for context enrichment
    context_headers: List[str] = []
    if matched_symbol_names:
        helper_sigs: List[str] = []
        try:
            from whole_body_compute.local.symbols import LocalSymbolIndex

            idx_engine = LocalSymbolIndex()
            with idx_engine._get_connection() as conn:
                db_callees = list(node_callees)
                for s_name in matched_symbol_names:
                    rows = conn.execute(
                        "SELECT callees FROM symbols WHERE name = ?", (s_name,)
                    ).fetchall()
                    for r in rows:
                        raw_c = r["callees"] or "[]"
                        try:
                            c_list = (
                                json.loads(raw_c)
                                if isinstance(raw_c, str)
                                else list(raw_c)
                            )
                        except Exception:
                            c_list = []
                        for c in c_list:
                            if c not in matched_symbol_names and c not in db_callees:
                                db_callees.append(c)

                for callee in db_callees:
                    row = conn.execute(
                        "SELECT name, signature, file_path, start_line, end_line FROM symbols WHERE name = ? LIMIT 1",
                        (callee,),
                    ).fetchone()
                    if row:
                        sig = row["signature"] or f"def {row['name']}(...)"
                        c_fp = Path(row["file_path"]).name
                        helper_sigs.append(
                            f"# callee: {sig}  [{c_fp}:L{row['start_line']}-L{row['end_line']}]"
                        )
                        if len(helper_sigs) >= 6:
                            break
        except Exception:
            pass

        if helper_sigs:
            context_headers.append("# [AST Context: 1-Hop Callee Signatures]")
            context_headers.extend(helper_sigs)

        memory_warnings: List[str] = []
        try:
            from whole_body_compute.local.dot_memory import get_dot_store

            store = get_dot_store()
            fp_str = str(file_path) if file_path else ""
            mems = store.get_symbol_memories(fp_str, matched_symbol_names)
            for s_name in matched_symbol_names:
                for w in mems.get(s_name, []):
                    memory_warnings.append(f"# [Memory Warning for {s_name}]: {w}")
        except Exception:
            pass

        if memory_warnings:
            context_headers.extend(memory_warnings)

    if context_headers:
        return "\n".join(context_headers) + "\n\n" + body_text
    return body_text


# Exported alias for compatibility with upgrade specs
extract_relevant_ast_slice = extract_deterministic_symbol_slice


def apply_code_block_ast_fallback(
    original_content: str, coder_output: str
) -> Tuple[str, bool]:
    """
    Fallback for small local models (1B-2B) that emit fenced ```python ... ``` blocks
    instead of SEARCH/REPLACE blocks. Matches AST functions/classes by name and splices
    their line ranges into original_content.
    """
    import ast

    cleaned = re.sub(r"<think>.*?</think>", "", coder_output, flags=re.DOTALL).strip()
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", cleaned, re.DOTALL)
    if not code_blocks:
        return original_content, False

    try:
        orig_tree = ast.parse(original_content)
    except SyntaxError:
        return original_content, False

    orig_nodes = {
        node.name: node
        for node in ast.walk(orig_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and hasattr(node, "end_lineno")
    }
    if not orig_nodes:
        return original_content, False

    orig_lines = original_content.splitlines()
    replacements: List[Tuple[int, int, List[str]]] = []

    for block in code_blocks:
        try:
            sub_tree = ast.parse(block)
        except SyntaxError:
            continue
        sub_lines = block.splitlines()
        for node in sub_tree.body:
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name in orig_nodes
                and hasattr(node, "end_lineno")
            ):
                target = orig_nodes[node.name]
                t_start = target.lineno - 1
                t_end = target.end_lineno or target.lineno
                # Include decorators if present
                if getattr(target, "decorator_list", None):
                    t_start = min(t_start, min(d.lineno - 1 for d in target.decorator_list))

                s_start = node.lineno - 1
                s_end = node.end_lineno or node.lineno
                if getattr(node, "decorator_list", None):
                    s_start = min(s_start, min(d.lineno - 1 for d in node.decorator_list))

                new_chunk = sub_lines[s_start:s_end]
                indent_diff = target.col_offset - node.col_offset
                if indent_diff > 0:
                    new_chunk = [(" " * indent_diff + l) if l.strip() else l for l in new_chunk]
                replacements.append((t_start, t_end, new_chunk))

    if not replacements:
        return original_content, False

    # Apply replacements in reverse line order to preserve offsets
    for start_idx, end_idx, new_chunk in sorted(replacements, key=lambda x: x[0], reverse=True):
        orig_lines = orig_lines[:start_idx] + new_chunk + orig_lines[end_idx:]

    updated = "\n".join(orig_lines) + ("\n" if original_content.endswith("\n") else "")
    try:
        ast.parse(updated)
    except SyntaxError:
        return original_content, False

    return updated, updated != original_content



def _is_tcp_port_open(host: str, port: int, timeout: float = 0.025) -> bool:
    """Fast <25ms TCP socket probe for local inference endpoints."""
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def resolve_best_local_coder_endpoint(
    preferred_model: Optional[str] = None,
    timeout_sec: float = 0.025,
) -> Dict[str, Any]:
    """
    Fast (<30ms socket/HTTP probe) endpoint resolver checking live local coder endpoints
    in priority order:
      1. :9380 (gemma4-26b)
      2. :9379 (voicefi-coder-mlx)
      3. :11434 Ollama (preferring gemma4, qwen2.5-coder:32b/14b/7b, llama3.3, voicefi-coder:1.5b
         before falling back to qwen2.5-coder:1.5b)
    """
    t0 = time.perf_counter()
    req_clean = (preferred_model or "").strip()

    # If caller explicitly requested voicefi-coder-mlx or 2b, check :9379 first
    if req_clean in ("voicefi-coder-mlx", "gemma4-2b"):
        if _is_tcp_port_open("127.0.0.1", 9379, timeout=timeout_sec):
            return {
                "available": True,
                "provider": "metal",
                "endpoint": "http://127.0.0.1:9379/v1/chat/completions",
                "model": req_clean,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

    # Priority 1: :9380 (gemma4-26b)
    if _is_tcp_port_open("127.0.0.1", 9380, timeout=timeout_sec):
        return {
            "available": True,
            "provider": "metal",
            "endpoint": "http://127.0.0.1:9380/v1/chat/completions",
            "model": "gemma4-26b",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    # Priority 2: :9379 (voicefi-coder-mlx)
    if _is_tcp_port_open("127.0.0.1", 9379, timeout=timeout_sec):
        return {
            "available": True,
            "provider": "metal",
            "endpoint": "http://127.0.0.1:9379/v1/chat/completions",
            "model": "voicefi-coder-mlx",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    # Priority 3: :11434 Ollama
    if _is_tcp_port_open("127.0.0.1", 11434, timeout=timeout_sec):
        ollama_model = _resolve_ollama_fallback(req_clean or "auto")
        if ollama_model:
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
            return {
                "available": True,
                "provider": "ollama",
                "endpoint": f"{base_url}/api/chat",
                "model": ollama_model,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

    return {
        "available": False,
        "provider": "none",
        "endpoint": "",
        "model": req_clean if req_clean and req_clean != "auto" else "qwen2.5-coder:1.5b",
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }


def _resolve_ollama_fallback(requested_model: str) -> Optional[str]:
    """Resolve a requested model name to an installed Ollama generative model."""
    if not _is_tcp_port_open("127.0.0.1", 11434, timeout=0.025):
        return None
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    try:
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=0.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            installed = [
                m.get("name", "")
                for m in data.get("models", [])
                if "embed" not in m.get("name", "").lower()
            ]
    except Exception:
        return None

    if not installed:
        return None

    clean_req = requested_model.replace("ollama/", "").strip()
    if clean_req and clean_req != "auto" and clean_req in installed:
        return clean_req

    # Tier-aware mapping when gemma4-2b (scout) or gemma4-26b / auto (coder) is requested
    if "2b" in clean_req or "scout" in clean_req or "0.8b" in clean_req:
        candidates = [
            "voicefi-coder:1.5b",
            "qwen2.5-coder:1.5b",
            "tev1:0.8b",
            "llama3.2:1b",
            "gemma2:2b",
            "tev1:latest",
        ]
    else:
        candidates = [
            "qwen2.5-coder:1.5b",
            "voicefi-coder:1.5b",
            "gemma4:26b",
            "gemma4:latest",
            "qwen2.5-coder:32b",
            "qwen2.5-coder:14b",
            "qwen2.5-coder:7b",
            "nimble:latest",
            "tev1:latest",
            "llama3.3:70b",
            "llama3.3:latest",
            "dolphin-llama3:latest",
            "llama3.2:1b",
        ]

    for c in candidates:
        if c in installed:
            return c
    return installed[0]


def resolve_available_ollama_coder_models(timeout: float = 0.05) -> List[str]:
    """
    Probe 127.0.0.1:11434 and return available Ollama coder models in priority order.
    """
    if not _is_tcp_port_open("127.0.0.1", 11434, timeout=timeout):
        return []
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    installed: List[str] = []
    try:
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=0.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            installed = [
                m.get("name", "")
                for m in data.get("models", [])
                if m.get("name") and "embed" not in m.get("name", "").lower()
            ]
    except Exception:
        installed = []

    preferred = [
        "qwen2.5-coder:1.5b",
        "voicefi-coder:1.5b",
        "nimble:latest",
        "qwen2.5-coder:7b",
        "qwen2.5-coder:14b",
        "qwen2.5-coder:32b",
        "gemma4:26b",
        "gemma4:latest",
        "llama3.3:70b",
        "llama3.3:latest",
        "dolphin-llama3:latest",
        "llama3.2:1b",
        "tev1:latest",
    ]

    if installed:
        ordered = [m for m in preferred if m in installed]
        for m in installed:
            if m not in ordered and "coder" in m.lower():
                ordered.append(m)
        for m in installed:
            if m not in ordered:
                ordered.append(m)
        return ordered

    # Fallback to standard local coder tiers if port is open but tags could not be enumerated
    return ["qwen2.5-coder:1.5b", "voicefi-coder:1.5b", "nimble:latest"]


async def query_ollama(
    model_name: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 3000,
    temperature: float = 0.2,
    timeout: float = 15.0,
) -> Optional[str]:
    """Query local Ollama daemon directly on :11434 with timeout guard."""
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    clean_model = model_name.replace("ollama/", "").strip()
    payload = {
        "model": clean_model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    try:
        req = urllib.request.Request(
            f"{base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data.get("message", {}).get("content", "").strip()
            if not content:
                choices = data.get("choices", [])
                if choices and "message" in choices[0]:
                    content = choices[0]["message"].get("content", "").strip()
            return content if content else None
    except Exception as e:
        logger.debug(f"[query_ollama] Request failed for {clean_model}: {e}")
        return None


async def query_model(
    model_name: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.2,
) -> str:
    """
    Query local model via warm litert-lm HTTP server if available,
    then local Ollama daemon, falling back to direct LocalModelEngine.
    """
    from whole_body_compute.local.supervisor import default_supervisor

    if hasattr(default_supervisor, "resolve_loadable_model"):
        approved, resolved_model, reason = default_supervisor.resolve_loadable_model(model_name)
        if not approved:
            default_supervisor.assert_can_load(model_name)
        elif resolved_model != model_name:
            logger.warning(
                f"[ReconImplementer] query_model auto-fallback '{model_name}' -> '{resolved_model}': {reason}"
            )
            model_name = resolved_model
    else:
        default_supervisor.assert_can_load(model_name)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    # 1. Query dedicated warm local Metal GPU HTTP servers (9379 for voicefi-coder-mlx & 2b, 9380 for 26b)
    if model_name in ("voicefi-coder-mlx", "gemma4-2b", "gemma4-26b"):
        server_url = (
            "http://127.0.0.1:9380/v1/chat/completions"
            if model_name == "gemma4-26b"
            else "http://127.0.0.1:9379/v1/chat/completions"
        )
        try:
            req = urllib.request.Request(
                server_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choices = data.get("choices", [])
                if choices and "message" in choices[0]:
                    return choices[0]["message"].get("content", "")
        except Exception as e:
            logger.debug(
                f"HTTP litert-lm server query failed ({e}), checking Ollama fallback."
            )

    # 2. Query local Ollama daemon (with automatic fallback for offline LiteRT tiers)
    ollama_target = _resolve_ollama_fallback(model_name)
    if ollama_target:
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        ollama_payload: Dict[str, Any] = {
            "model": ollama_target,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            req = urllib.request.Request(
                f"{base_url}/api/chat",
                data=json.dumps(ollama_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data.get("message", {}).get("content", "").strip()
                if content:
                    return content
        except Exception as e:
            logger.debug(f"Ollama query failed for {ollama_target} ({e}), falling back to LocalModelEngine.")

    # 3. Direct in-process LocalModelEngine
    engine = LocalModelEngine(model_name=model_name)
    return await engine.chat_text(prompt=prompt, system_instructions=system_prompt)


_DEFAULT_WBC_CODER_PROMPT = (
    "You are an expert systems programmer. Implement the requested code changes accurately.\n"
    "Format your output strictly using one or more SEARCH/REPLACE blocks like this:\n\n"
    "<<<<<<< SEARCH\n"
    "[exact lines from the original file to replace]\n"
    "=======\n"
    "[new replacement lines]\n"
    ">>>>>>> REPLACE\n\n"
    "Maintain identical indentation and coding conventions. Output ONLY the SEARCH/REPLACE block(s)."
)

HARNESS_PROFILES: Dict[str, Dict[str, Any]] = {
    "mini_swe_bash": {
        "profile_id": "mini_swe_bash",
        "matched_prefixes": ("glm", "mini-swe"),
        "tool_surface": ["bash"],
        "max_obs_chars": 1200,
        "compaction_strategy": "head_tail_elision",
        "system_prompt": (
            "You are a helpful assistant that can interact with a computer shell to solve programming tasks.\n"
            "Emit ONLY minimal <<<<<<< SEARCH / ======= / >>>>>>> REPLACE blocks with exact indentation. "
            "No prose or tool-schema bloat."
        ),
    },
    "hermes_structured": {
        "profile_id": "hermes_structured",
        "matched_prefixes": ("deepseek", "hermes", "nous"),
        "tool_surface": ["terminal", "read_file", "patch", "search_files"],
        "max_obs_chars": 1600,
        "compaction_strategy": "stale_turn_pruning",
        "system_prompt": (
            "You are Hermes Agent executing a structured Action -> Observation loop.\n"
            "Analyze the target code slice and emit ONLY deterministic SEARCH/REPLACE blocks:\n\n"
            "<<<<<<< SEARCH\n"
            "[exact lines from the original file to replace]\n"
            "=======\n"
            "[new replacement lines]\n"
            ">>>>>>> REPLACE"
        ),
    },
    "kimi_flint": {
        "profile_id": "kimi_flint",
        "matched_prefixes": ("flint", "kimi", "moonshot"),
        "tool_surface": ["shell", "read", "edit", "write"],
        "max_obs_chars": 1400,
        "compaction_strategy": "window_85pct_handoff",
        "system_prompt": (
            "You are Kimi Code performing a surgical single-file edit.\n"
            "Match exact lines without line numbers and emit ONLY <<<<<<< SEARCH / ======= / >>>>>>> REPLACE blocks."
        ),
    },
    "pi_minimal": {
        "profile_id": "pi_minimal",
        "matched_prefixes": ("pi", "oh-my-pi"),
        "tool_surface": ["read", "bash", "edit", "write"],
        "max_obs_chars": 1200,
        "compaction_strategy": "early_reserve_summary",
        "system_prompt": (
            "Active primitives: read, bash, edit, write. Be concise. Show file paths clearly.\n"
            "Output ONLY exact <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE edit blocks."
        ),
    },
    "wbc_ast_slice": {
        "profile_id": "wbc_ast_slice",
        "matched_prefixes": ("qwen", "voicefi-coder", "gemma", "tev"),
        "tool_surface": ["ast_slice", "search_replace", "ast_secgate"],
        "max_obs_chars": 2400,
        "compaction_strategy": "ast_symbol_slice",
        "system_prompt": _DEFAULT_WBC_CODER_PROMPT,
    },
}


def resolve_harness_profile(model_name: str) -> Dict[str, Any]:
    """
    Resolve the optimal coding-agent harness profile for a given model family,
    informed by Harness Leaderboard & Harness Lens empirical findings:
    - GLM-5.3-Flash / mini-swe* -> mini_swe_bash (ultra-minimal prompt, head/tail stdout masking)
    - DeepSeek-V4.1-Flash / hermes* -> hermes_structured (structured loop, stale-turn pruning)
    - Flint / kimi* -> kimi_flint (concise surgical edit framing)
    - Pi -> pi_minimal (4-primitive minimal core without persona bloat)
    - Default (qwen*, voicefi-coder*, gemma*, tev*) -> wbc_ast_slice (<7ms AST symbol slice)
    """
    raw = (model_name or "").strip().lower()
    # Strip common provider/router prefixes (e.g. zai-org/, deepseek-ai/, RecursiveSI/, ollama/)
    base = raw.split("/")[-1]

    if base.startswith(("glm", "mini-swe")) or "glm-" in raw or "mini-swe" in raw:
        profile = dict(HARNESS_PROFILES["mini_swe_bash"])
    elif base.startswith(("deepseek", "hermes", "nous")) or "deepseek" in raw or "hermes" in raw:
        profile = dict(HARNESS_PROFILES["hermes_structured"])
    elif base.startswith(("flint", "kimi", "moonshot")) or "flint" in raw or "kimi" in raw:
        profile = dict(HARNESS_PROFILES["kimi_flint"])
    elif base == "pi" or base.startswith(("pi-", "pi_", "pi:", "oh-my-pi")):
        profile = dict(HARNESS_PROFILES["pi_minimal"])
    else:
        profile = dict(HARNESS_PROFILES["wbc_ast_slice"])

    profile["model_name"] = model_name
    return profile


def compact_tool_observation(output: str, max_chars: int = 1200) -> str:
    """
    Observation masking & head/tail elision inspired by Harness Lens (mini-swe-agent
    head/tail observation template & Hermes Agent stale-turn tool output pruning).
    Preserves leading command context and trailing error tracebacks while masking
    middle stdout bloat to prevent quadratic O(N^2) context ballooning.
    """
    if not output or max_chars <= 0 or len(output) <= max_chars:
        return output

    elided_chars = len(output) - max_chars
    marker = f"\n... [HARNESS_LENS_MASKED: {elided_chars} chars elided] ...\n"
    available = max(64, max_chars - len(marker))

    # Allocate 40% to head (command invocation / headers) and 60% to tail (tracebacks / assertions)
    head_len = max(24, int(available * 0.4))
    tail_len = max(24, available - head_len)

    return output[:head_len] + marker + output[-tail_len:]


def _apply_search_replace_or_fallbacks(
    content: str,
    coder_output: str,
    target_line_hint: Optional[int] = None,
    instruction: str = "",
) -> Tuple[str, bool]:
    blocks = parse_search_replace_blocks(coder_output)
    updated_content = content
    applied_any = False

    if blocks:
        for search_chunk, replace_chunk in blocks:
            if search_chunk.strip() == replace_chunk.strip():
                continue
            updated_content, matched = apply_search_replace(
                updated_content,
                search_chunk,
                replace_chunk,
                preferred_line=target_line_hint,
            )
            if matched:
                applied_any = True

    if not applied_any:
        updated_content, applied_any = apply_code_block_ast_fallback(
            content, coder_output
        )

    if not applied_any and instruction:
        m_after = re.search(
            r"after\s+`([^`]+)`.*?add\s+`([^`]+)`",
            instruction,
            re.IGNORECASE | re.DOTALL,
        )
        if m_after:
            anchor_str = m_after.group(1).strip()
            new_str = m_after.group(2).strip()
            if anchor_str in content and new_str not in content:
                updated_content, applied_any = apply_search_replace(
                    content,
                    anchor_str,
                    f"{anchor_str}\n{new_str}",
                )

    return updated_content, applied_any


class ReconImplementer:
    """
    On-device 2-tier implementation pipeline:
    - Tier 1: Gemma 4 2B extracts the target slice and line coordinates.
    - Tier 2: Gemma 4 26B generates the precise code edit.
    - Applies the edit locally and outputs only the clean unified git diff.
    """

    DEFAULT_SCOUT_PROMPT = (
        "You are the VoiceFi Recon Scout. Your job is to locate the exact functions, "
        "classes, or lines in the provided code that need to be modified for the user's task.\n"
        "Return ONLY the relevant code slice with line context, plus a 1-sentence note of what needs editing. "
        "Do NOT write the implementation yourself."
    )

    DEFAULT_CODER_PROMPT = _DEFAULT_WBC_CODER_PROMPT

    def __init__(
        self,
        model_scout: str = "voicefi-coder-mlx",
        model_coder: str = "voicefi-coder-mlx",
        dot_store: Optional[Any] = None,
    ):
        self.model_scout = model_scout
        self.model_coder = model_coder
        self.harness_profile = resolve_harness_profile(model_coder)
        self._dot_store = dot_store

    def build_coder_prompt(
        self,
        file_path: Union[str, Path],
        instruction: str,
        scout_slice: str,
        dot_store: Optional[Any] = None,
    ) -> str:
        """
        Construct the local coder prompt augmented with <150-token outcome-verified
        cloud exemplars retrieved from `MemoryStore`.
        """
        fname = Path(str(file_path)).name
        exemplar_block = ""
        try:
            store = dot_store or self._dot_store
            if store is None:
                from whole_body_compute.local.dot_memory import get_dot_store

                store = get_dot_store()
            if hasattr(store, "retrieve_cloud_exemplars"):
                exemplar_block = store.retrieve_cloud_exemplars(
                    file_path=str(file_path),
                    instruction=instruction,
                    top_k=2,
                    max_tokens=150,
                )
        except Exception:
            exemplar_block = ""

        parts = [f"File: {fname}", f"Task: {instruction}"]
        if exemplar_block:
            parts.append(f"\n{exemplar_block}")
        parts.append(f"\nRelevant Code Slice:\n```python\n{scout_slice}\n```")
        parts.append("\nProduce the exact SEARCH/REPLACE block(s) to achieve this task.")
        return "\n".join(parts)

    async def implement(
        self,
        target_path: Union[str, Path],
        instruction: str,
        apply: bool = True,
        diff_only: bool = True,
        max_file_bytes: int = 500_000,
    ) -> ImplementResult:
        """
        Execute full on-device scout -> implement -> diff workflow.
        """
        start_total = time.perf_counter()
        path = Path(os.path.expanduser(str(target_path))).resolve()

        if not path.exists() or not path.is_file():
            return ImplementResult(
                target_path=str(path),
                instruction=instruction,
                diff="",
                applied=False,
                scout_duration=0.0,
                coder_duration=0.0,
                total_duration=0.0,
                tokens_saved=0,
                savings_pct=0.0,
                input_tokens_est=0,
                diff_tokens_est=0,
                model_scout=self.model_scout,
                model_coder=self.model_coder,
                error=f"File not found: {path}",
            )

        # Check resource headroom for both scout and coder models (with auto-sweep + fallback cascade)
        from whole_body_compute.local.supervisor import default_supervisor

        for role_attr in ("model_scout", "model_coder"):
            m_name = getattr(self, role_attr)
            if hasattr(default_supervisor, "resolve_loadable_model"):
                approved, resolved_model, reason = default_supervisor.resolve_loadable_model(m_name)
            else:
                approved, reason, _ = default_supervisor.can_load_model(m_name)
                resolved_model = m_name

            if not approved:
                _, _, alert = default_supervisor.can_load_model(m_name)
                alert_box = default_supervisor.format_resource_alert(alert) if alert else reason
                logger.warning(
                    f"[ReconImplementer] Model '{m_name}' blocked by ResourceGate: {reason}"
                )
                return ImplementResult(
                    target_path=str(path),
                    instruction=instruction,
                    diff="",
                    applied=False,
                    scout_duration=0.0,
                    coder_duration=0.0,
                    total_duration=round(time.perf_counter() - start_total, 3),
                    tokens_saved=0,
                    savings_pct=0.0,
                    input_tokens_est=0,
                    diff_tokens_est=0,
                    model_scout=self.model_scout,
                    model_coder=self.model_coder,
                    scout_findings="",
                    error=f"ResourceGate Blocked: {reason}\n\n{alert_box}",
                )
            elif resolved_model != m_name:
                logger.warning(
                    f"[ReconImplementer] Auto-fallback for {role_attr}: '{m_name}' -> '{resolved_model}' ({reason})"
                )
                setattr(self, role_attr, resolved_model)

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            original_content = f.read(max_file_bytes)

        input_tokens = estimate_tokens(original_content)
        lines = original_content.splitlines()
        total_lines = len(lines)

        # -------------------------------------------------------------
        # Phase 1: Scout (Deterministic AST Slicer -> LLM Fallback) for files > 80 lines
        # -------------------------------------------------------------
        scout_start = time.perf_counter()
        scout_slice = original_content
        scout_findings = ""
        target_line_hint: Optional[int] = None

        if total_lines > 80:
            focused_slice = extract_line_hint_or_keyword_window(
                original_content,
                instruction,
                window_radius=80,
                file_path=path,
            )
            if focused_slice:
                scout_slice = focused_slice
                scout_findings = focused_slice
                lh_match = re.search(r"(?:line|lines|L)\s*(\d+)", instruction, re.IGNORECASE)
                if lh_match:
                    try:
                        target_line_hint = int(lh_match.group(1))
                    except ValueError:
                        pass
            else:
                det_slice = extract_deterministic_symbol_slice(
                    original_content, instruction, file_path=path
                )
                if det_slice:
                    scout_slice = det_slice
                    scout_findings = det_slice
                else:
                    scout_query = (
                        f"Task: {instruction}\n\n"
                        f"File: {path.name} ({total_lines} lines)\n\n"
                        f"```\n{original_content}\n```\n\n"
                        "Identify the exact section or lines to edit for this task."
                    )
                    try:
                        scout_findings = await query_model(
                            model_name=self.model_scout,
                            prompt=scout_query,
                            system_prompt=self.DEFAULT_SCOUT_PROMPT,
                            max_tokens=1024,
                        )
                        scout_slice = scout_findings
                    except Exception as e:
                        logger.warning(f"Scout tier failed ({e}), using full file for coder tier.")
                        scout_slice = original_content

        scout_duration = round(time.perf_counter() - scout_start, 3)

        # -------------------------------------------------------------
        # Phase 2: Coder (Targeted 200-Token Slice + Distilled Cloud Exemplars)
        # -------------------------------------------------------------
        coder_start = time.perf_counter()
        profile = resolve_harness_profile(self.model_coder)
        coder_prompt = self.build_coder_prompt(path, instruction, scout_slice)

        active_coder_model = self.model_coder
        coder_output = ""
        coder_error: Optional[str] = None

        try:
            coder_output = await query_model(
                model_name=self.model_coder,
                prompt=coder_prompt,
                system_prompt=profile.get("system_prompt", self.DEFAULT_CODER_PROMPT),
                max_tokens=3000,
            )
        except Exception as e:
            coder_error = str(e)
            logger.warning(f"[ReconImplementer] Primary coder query failed ({e})")

        # -------------------------------------------------------------
        # Phase 3: Apply Changes & Generate Unified Diff (with Ollama Fallback)
        # -------------------------------------------------------------
        updated_content = original_content
        applied_any = False

        if coder_output:
            updated_content, applied_any = _apply_search_replace_or_fallbacks(
                original_content,
                coder_output,
                target_line_hint=target_line_hint,
                instruction=instruction,
            )

        # Fallback to available local Ollama coder models if primary endpoint
        # (:9380 or :9379) failed to produce matching SEARCH/REPLACE blocks
        # or suffered connection/format failure
        if not applied_any:
            ollama_candidates = resolve_available_ollama_coder_models()
            for cand_model in ollama_candidates:
                if cand_model == active_coder_model:
                    continue
                logger.info(
                    f"[ReconImplementer] Primary endpoint failed to produce matching blocks or failed; "
                    f"trying Ollama coder fallback '{cand_model}' on :11434"
                )
                try:
                    fb_output = await query_ollama(
                        model_name=cand_model,
                        prompt=coder_prompt,
                        system_prompt=profile.get("system_prompt", self.DEFAULT_CODER_PROMPT),
                        max_tokens=3000,
                    )
                    if not fb_output:
                        try:
                            fb_output = await query_model(
                                model_name=cand_model,
                                prompt=coder_prompt,
                                system_prompt=profile.get("system_prompt", self.DEFAULT_CODER_PROMPT),
                                max_tokens=3000,
                            )
                        except Exception:
                            fb_output = None
                    if fb_output:
                        cand_content, cand_applied = _apply_search_replace_or_fallbacks(
                            original_content,
                            fb_output,
                            target_line_hint=target_line_hint,
                            instruction=instruction,
                        )
                        if cand_applied:
                            updated_content = cand_content
                            applied_any = True
                            coder_output = fb_output
                            active_coder_model = cand_model
                            coder_error = None
                            break
                except Exception as fb_err:
                    logger.debug(f"[ReconImplementer] Fallback to Ollama '{cand_model}' failed: {fb_err}")

        coder_duration = round(time.perf_counter() - coder_start, 3)

        # Generate unified diff
        diff_lines = list(
            difflib.unified_diff(
                original_content.splitlines(keepends=True),
                updated_content.splitlines(keepends=True),
                fromfile=f"a/{path.name}",
                tofile=f"b/{path.name}",
            )
        )
        diff_text = "".join(diff_lines)

        syntax_error_msg: Optional[str] = None
        if applied_any and path.suffix == ".py":
            try:
                import ast

                cand_tree = ast.parse(updated_content)
                try:
                    orig_tree = ast.parse(original_content)
                    orig_asserts = sum(1 for n in ast.walk(orig_tree) if isinstance(n, ast.Assert))
                    cand_asserts = sum(1 for n in ast.walk(cand_tree) if isinstance(n, ast.Assert))
                    if cand_asserts < orig_asserts:
                        syntax_error_msg = (
                            f"3-Stage AST SecGate rejected edit: assertion count dropped "
                            f"({cand_asserts} < {orig_asserts})"
                        )
                        logger.warning(f"[ReconImplementer] {syntax_error_msg}")
                        applied_any = False
                except SyntaxError:
                    pass
            except SyntaxError as syn_err:
                syntax_error_msg = (
                    f"AST SyntaxGuard rejected edit at line {syn_err.lineno}: {syn_err.msg}"
                )
                logger.warning(f"[ReconImplementer] {syntax_error_msg}")
                applied_any = False

        # If git repo exists, run git diff if applied to disk
        if apply and applied_any and diff_text.strip():
            # Backup original
            bak_path = path.with_suffix(f"{path.suffix}.vifi_bak")
            try:
                with open(bak_path, "w", encoding="utf-8") as f:
                    f.write(original_content)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(updated_content)
            except Exception as e:
                logger.error(f"Failed to write file {path}: {e}")

        diff_tokens = estimate_tokens(diff_text)
        tokens_saved = max(0, input_tokens - diff_tokens)
        savings_pct = round((tokens_saved / max(1, input_tokens)) * 100, 1)
        total_duration = round(time.perf_counter() - start_total, 3)

        res = ImplementResult(
            target_path=str(path),
            instruction=instruction,
            diff=diff_text if diff_text.strip() else coder_output,
            applied=applied_any if apply else False,
            scout_duration=scout_duration,
            coder_duration=coder_duration,
            total_duration=total_duration,
            tokens_saved=tokens_saved,
            savings_pct=savings_pct,
            input_tokens_est=input_tokens,
            diff_tokens_est=diff_tokens,
            model_scout=self.model_scout,
            model_coder=active_coder_model,
            scout_findings=scout_findings,
            error=None
            if applied_any
            else (
                syntax_error_msg
                or coder_error
                or "No matching SEARCH/REPLACE blocks could be applied automatically."
            ),
        )
        try:
            from whole_body_compute.local.telemetry import record_hybrid_step

            record_hybrid_step(
                stage="local_implement",
                model_name=res.model_coder,
                is_local=True,
                duration_ms=int(res.total_duration * 1000),
                input_tokens=res.input_tokens_est,
                output_tokens=res.diff_tokens_est,
                tokens_saved=res.tokens_saved,
                savings_pct=res.savings_pct,
                success=(res.error is None),
                target=str(target_path),
            )
        except Exception:
            pass
        try:
            from whole_body_compute.local.supervisor import default_supervisor

            default_supervisor.maybe_opportunistic_reap()
        except Exception:
            pass
        return res


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for VoiceFi On-Device ReconImplementer."""
    import argparse
    import asyncio
    import sys

    parser = argparse.ArgumentParser(
        description="VoiceFi On-Device ReconImplementer (2-tier local AST slice + coder diff)"
    )
    parser.add_argument("target", type=str, help="Target file path to modify")
    parser.add_argument(
        "-i",
        "--instruction",
        type=str,
        required=True,
        help="Surgical code modification instruction",
    )
    parser.add_argument(
        "--model-scout",
        type=str,
        default="voicefi-coder-mlx",
        help="Scout model name (default: voicefi-coder-mlx)",
    )
    parser.add_argument(
        "--model-coder",
        "-m",
        type=str,
        default="auto",
        help="Coder model name or 'auto' (default: auto -> :9380 gemma4-26b / :9379 voicefi-coder-mlx / Ollama)",
    )
    parser.add_argument(
        "--no-apply",
        "--dry-run",
        dest="no_apply",
        action="store_true",
        help="Preview unified diff without writing changes to disk",
    )
    parser.add_argument("--json", action="store_true", help="Output result as JSON")
    args = parser.parse_args(argv)

    probe = resolve_best_local_coder_endpoint(
        None if args.model_coder == "auto" else args.model_coder
    )
    if not probe["available"]:
        msg = (
            "[Fast-Fail] No local coder endpoint (:9380, :9379, or :11434 Ollama) is online "
            f"(probed in {probe['latency_ms']}ms). Fallback to native `edit_file` immediately."
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "target_path": args.target,
                        "instruction": args.instruction,
                        "applied": False,
                        "error": msg,
                        "probe": probe,
                    },
                    indent=2,
                )
            )
        else:
            print(msg, file=sys.stderr)
        return 1

    coder_model = probe["model"] if args.model_coder == "auto" else args.model_coder
    impl = ReconImplementer(model_scout=args.model_scout, model_coder=coder_model)
    res = asyncio.run(
        impl.implement(
            target_path=args.target,
            instruction=args.instruction,
            apply=not args.no_apply,
        )
    )
    if args.json:
        print(json.dumps(res.to_dict(), indent=2))
    else:
        print(
            f"=== VoiceFi ReconImplementer [coder={res.model_coder}, applied={res.applied}, {res.total_duration}s] ==="
        )
        if res.error:
            print(f"[Error] {res.error} (Fallback to native `edit_file` immediately)")
        if res.diff:
            print(res.diff)
    return 0 if (res.applied or (args.no_apply and not res.error)) else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
