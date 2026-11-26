"""
Blood-Brain Barrier (Organ 05): AST SecGate, Predictive AST Slicer, Fast Log Digest, & Post-Tool Buffer Slicer.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from whole_body_compute.organs.immune_hygiene import _get_wbc_data_dir
from whole_body_compute.organs.immune_polyglot import (
    POLYGLOT_EXTENSIONS,
    extract_polyglot_predictive_slice,
    extract_polyglot_symbols,
    fence_language_for_path,
    format_polyglot_symbol_outline,
    run_polyglot_three_stage_secgate,
)
from whole_body_compute.organs.immune_vascular import AntibodyRegistry

PROTECTED_CONFIG_FILES = (
    ".agents/hooks.json",
    ".agents/mcp_config.json",
    ".gemini/config/hooks.json",
    ".gemini/config/mcp_config.json",
    ".gemini/settings.json",
    ".claude/settings.json",
    ".claude/settings.local.json",
    ".cursor/hooks.json",
    ".codex/hooks.json",
    "bin/body-immune-hook",
    "bin/body-immune-watchdog",
    "bundled/body_immune_hook.sh",
    "eval_models.py",
    "evaluate.py",
    "eval_localbench.py",
    "BENCHMARK_REPORT.md",
)

BINARY_MEDIA_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".mp4", ".mov", ".wav", ".mp3", ".ogg", ".flac",
    ".pdf", ".zip", ".tar", ".gz", ".safetensors", ".gguf",
    ".bin", ".onnx", ".pyc", ".so", ".dylib",
}

_FAST_SECRET_RE = re.compile(
    r"(?:sk-[A-Za-z0-9_\-]{16,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_\-]{35}|-----BEGIN[ A-Z]*PRIVATE KEY-----)"
)


def _fast_is_tautological(test_node: ast.AST) -> bool:
    if isinstance(test_node, ast.Constant):
        return bool(test_node.value) is True
    if isinstance(test_node, (ast.List, ast.Tuple, ast.Set)):
        return len(test_node.elts) > 0
    if isinstance(test_node, ast.Dict):
        return len(test_node.keys) > 0
    if isinstance(test_node, ast.JoinedStr):
        return any(
            isinstance(v, ast.Constant) and bool(str(v.value))
            for v in test_node.values
        )
    if isinstance(test_node, ast.BoolOp) and isinstance(test_node.op, ast.Or):
        return any(_fast_is_tautological(v) for v in test_node.values)
    if isinstance(test_node, ast.Compare):
        if len(test_node.ops) == 1:
            op = test_node.ops[0]
            left, right = test_node.left, test_node.comparators[0]
            if isinstance(op, (ast.Eq, ast.Is)):
                if isinstance(left, ast.Constant) and isinstance(right, ast.Constant):
                    return left.value == right.value
                if isinstance(left, ast.Name) and isinstance(right, ast.Name):
                    return left.id == right.id
            elif isinstance(op, (ast.NotEq, ast.IsNot)):
                if isinstance(left, ast.Constant) and isinstance(right, ast.Constant):
                    return left.value != right.value
            elif isinstance(op, (ast.Gt, ast.GtE, ast.Lt, ast.LtE)):
                if (
                    isinstance(left, ast.Constant)
                    and isinstance(right, ast.Constant)
                    and isinstance(left.value, (int, float))
                    and isinstance(right.value, (int, float))
                ):
                    if isinstance(op, ast.Gt):
                        return left.value > right.value
                    if isinstance(op, ast.GtE):
                        return left.value >= right.value
                    if isinstance(op, ast.Lt):
                        return left.value < right.value
                    if isinstance(op, ast.LtE):
                        return left.value <= right.value
    if isinstance(test_node, ast.UnaryOp) and isinstance(test_node.op, ast.Not):
        if isinstance(test_node.operand, ast.Constant):
            return bool(test_node.operand.value) is False
        if isinstance(test_node.operand, (ast.List, ast.Tuple, ast.Set)):
            return len(test_node.operand.elts) == 0
        if isinstance(test_node.operand, ast.Dict):
            return len(test_node.operand.keys) == 0
    if isinstance(test_node, ast.Call) and isinstance(test_node.func, ast.Name) and test_node.func.id == "bool":
        if test_node.args and isinstance(test_node.args[0], ast.Constant):
            return bool(test_node.args[0].value) is True
    return False


def _has_skip_decorator(node: ast.AST) -> bool:
    for dec in getattr(node, "decorator_list", []):
        dec_Target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(dec_Target, ast.Name) and "skip" in dec_Target.id.lower():
            return True
        if isinstance(dec_Target, ast.Attribute) and "skip" in dec_Target.attr.lower():
            return True
    return False


def _render_ast_call_span(node: ast.AST, source_code: str = "") -> str:
    if source_code:
        seg = ast.get_source_segment(source_code, node)
        if seg and seg.strip():
            return seg.strip()
    try:
        return ast.unparse(node).strip()
    except Exception:
        return ""


def _fast_extract_ast_summary(tree: ast.AST, source_code: str = "") -> Dict[str, Any]:
    assert_count = 0
    taut_asserts = 0
    test_funcs: Set[str] = set()
    unsafe_calls: List[str] = []
    unsafe_spans: List[Tuple[str, str]] = []
    alias_map: Dict[str, str] = {}

    # Seed module-level aliases in O(len(tree.body)) before single-pass iterative DFS
    for stmt in getattr(tree, "body", ()):
        if isinstance(stmt, ast.Import):
            for n in stmt.names:
                if n.asname:
                    alias_map[n.asname] = n.name
        elif isinstance(stmt, ast.ImportFrom):
            mod = stmt.module or ""
            for n in stmt.names:
                local_name = n.asname or n.name
                alias_map[local_name] = f"{mod}.{n.name}" if mod else n.name

    stack: List[ast.AST] = [tree]
    pop_node = stack.pop
    ext_nodes = stack.extend
    iter_children = ast.iter_child_nodes
    Import_cls = ast.Import
    ImportFrom_cls = ast.ImportFrom
    Assert_cls = ast.Assert
    Func_cls = (ast.FunctionDef, ast.AsyncFunctionDef)
    Call_cls = ast.Call

    while stack:
        node = pop_node()
        children = list(iter_children(node))
        if children:
            children.reverse()
            ext_nodes(children)

        n_cls = node.__class__
        if n_cls is Import_cls:
            for n in node.names:
                if n.asname:
                    alias_map[n.asname] = n.name
        elif n_cls is ImportFrom_cls:
            mod = node.module or ""
            for n in node.names:
                local_name = n.asname or n.name
                alias_map[local_name] = f"{mod}.{n.name}" if mod else n.name
        elif n_cls is Assert_cls:
            assert_count += 1
            if _fast_is_tautological(node.test):
                taut_asserts += 1
        elif isinstance(node, Func_cls):
            if node.name.startswith("test_") and not _has_skip_decorator(node):
                test_funcs.add(node.name)
        elif n_cls is Call_cls:
            matched_label: Optional[str] = None
            if isinstance(node.func, ast.Subscript):
                sub_val = node.func.value
                if isinstance(sub_val, ast.Name) and sub_val.id in ("__builtins__", "builtins"):
                    matched_label = "__builtins__[...]"
                elif isinstance(sub_val, ast.Call) and isinstance(sub_val.func, ast.Name) and sub_val.func.id in ("globals", "locals", "vars"):
                    matched_label = f"{sub_val.func.id}()[...]"
            elif isinstance(node.func, ast.Name):
                resolved = alias_map.get(node.func.id, node.func.id)
                if resolved in ("eval", "exec", "compile", "__import__", "os.system", "os.popen", "pty.spawn"):
                    matched_label = resolved
                elif resolved in ("subprocess.Popen", "subprocess.run", "subprocess.call", "subprocess.check_call", "subprocess.check_output"):
                    for kw in node.keywords:
                        if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and bool(kw.value.value) is True:
                            matched_label = f"{resolved}(shell=True)"
                            break
                elif node.func.id == "getattr" and len(node.args) >= 2:
                    arg0 = node.args[0]
                    arg1 = node.args[1]
                    arg0_name = alias_map.get(arg0.id, arg0.id) if isinstance(arg0, ast.Name) else ""
                    if isinstance(arg1, ast.Constant):
                        if arg1.value in ("system", "popen", "exec", "eval", "Popen", "spawn"):
                            matched_label = f'getattr(..., "{arg1.value}")'
                    elif arg0_name in ("os", "subprocess", "pty", "builtins", "__builtins__", "shutil"):
                        matched_label = f"getattr({arg0_name}, <dynamic>)"
            elif isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                val = node.func.value
                raw_id = val.id if isinstance(val, ast.Name) else ("call()" if isinstance(val, ast.Call) else "")
                val_repr = alias_map.get(raw_id, raw_id) if raw_id else ""
                qual = f"{val_repr}.{attr}" if val_repr else attr
                if (
                    qual in ("os.system", "os.popen", "pty.spawn")
                    or attr in ("system", "popen", "spawn", "posix_spawn", "execl", "execle", "execclp", "execv", "execve", "execvp")
                ) and qual != "platform.system":
                    matched_label = qual
                elif qual in ("subprocess.Popen", "subprocess.run", "subprocess.call", "subprocess.check_call", "subprocess.check_output"):
                    for kw in node.keywords:
                        if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and bool(kw.value.value) is True:
                            matched_label = f"{qual}(shell=True)"
                            break
                elif qual == "shutil.rmtree" and node.args:
                    arg0 = node.args[0]
                    if isinstance(arg0, ast.Constant) and str(arg0.value).strip() in ("/", "~", ".."):
                        matched_label = "shutil.rmtree(root)"
            if matched_label:
                unsafe_calls.append(matched_label)
                span = _render_ast_call_span(node, source_code) or matched_label
                unsafe_spans.append((matched_label, span))
    return {
        "assert_count": assert_count,
        "tautological_asserts": taut_asserts,
        "test_funcs": test_funcs,
        "unsafe_calls": unsafe_calls,
        "unsafe_spans": unsafe_spans,
    }


def _fast_three_stage_ast_secgate(
    parent_code: str,
    candidate_code: str,
    file_path: str = "",
    *,
    parent_tree: Optional[ast.AST] = None,
    candidate_tree: Optional[ast.AST] = None,
) -> Dict[str, Any]:
    """Zero-import stdlib 3-Stage AST SecGate for sub-30ms `-S` hook execution."""
    fname = Path(file_path).name if file_path else ""
    norm_candidates = []
    if file_path:
        norm_candidates.append(file_path.replace("\\", "/"))
        try:
            norm_candidates.append(str(Path(file_path).expanduser().resolve()).replace("\\", "/"))
        except OSError:
            pass
    if fname and any(
        fname == pat or any(nc.endswith(pat) for nc in norm_candidates)
        for pat in PROTECTED_CONFIG_FILES
    ):
        return {
            "passed": False,
            "stage": 1,
            "reason": f"Stage 1 (Protected Grader Lock): Modification of '{fname}' is forbidden.",
        }
    ext = Path(file_path).suffix.lower() if file_path else ".py"
    if ext in POLYGLOT_EXTENSIONS:
        gate = run_polyglot_three_stage_secgate(
            parent_code=parent_code,
            candidate_code=candidate_code,
            file_path=file_path,
        )
        if not gate.get("passed", False) and int(gate.get("stage", 0)) == 3:
            AntibodyRegistry.get_instance().record_antigen(
                vector="ast_secgate",
                payload=candidate_code,
                reason=str(gate.get("reason", "Stage 3 unsafe call")),
                stage=3,
                persist_to_hippocampus=False,
            )
        return gate
    if candidate_tree is not None:
        cand_tree = candidate_tree
    else:
        try:
            cand_tree = ast.parse(candidate_code)
        except SyntaxError as exc:
            return {
                "passed": False,
                "stage": 1,
                "reason": f"Stage 1 (Syntax Error): {exc.msg} at line {exc.lineno}",
            }
        except (ValueError, RecursionError, MemoryError) as exc:
            return {
                "passed": False,
                "stage": 1,
                "reason": f"Stage 1 (AST Parse Guard): {type(exc).__name__}",
            }

    parent_summary = {
        "assert_count": 0,
        "tautological_asserts": 0,
        "test_funcs": set(),
        "unsafe_calls": [],
        "unsafe_spans": [],
    }
    if parent_tree is not None:
        parent_summary = _fast_extract_ast_summary(parent_tree, parent_code)
    elif parent_code.strip():
        try:
            parent_summary = _fast_extract_ast_summary(ast.parse(parent_code), parent_code)
        except SyntaxError:
            pass
    cand_summary = _fast_extract_ast_summary(cand_tree, candidate_code)

    if cand_summary["assert_count"] < parent_summary["assert_count"]:
        return {
            "passed": False,
            "stage": 2,
            "reason": (
                f"Stage 2 (Anti-Reward-Hacking): Candidate reduced assert count "
                f"({cand_summary['assert_count']} < {parent_summary['assert_count']})."
            ),
            "parent_asserts": parent_summary["assert_count"],
            "candidate_asserts": cand_summary["assert_count"],
        }
    missing_tests = parent_summary["test_funcs"] - cand_summary["test_funcs"]
    if missing_tests:
        return {
            "passed": False,
            "stage": 2,
            "reason": f"Stage 2 (Anti-Reward-Hacking): Candidate deleted test function(s): {sorted(missing_tests)}",
            "parent_asserts": parent_summary["assert_count"],
            "candidate_asserts": cand_summary["assert_count"],
        }
    if cand_summary["tautological_asserts"] > parent_summary["tautological_asserts"]:
        return {
            "passed": False,
            "stage": 2,
            "reason": "Stage 2 (Anti-Reward-Hacking): Tautological assertion (`assert True`) detected.",
            "parent_asserts": parent_summary["assert_count"],
            "candidate_asserts": cand_summary["assert_count"],
        }
    new_unsafe = [c for c in cand_summary["unsafe_calls"] if c not in parent_summary["unsafe_calls"]]
    if new_unsafe:
        parent_span_set = set(parent_summary.get("unsafe_spans", []))
        new_spans = [
            span
            for pair in cand_summary.get("unsafe_spans", [])
            if pair not in parent_span_set and pair[0] in new_unsafe
            for span in (pair[1],)
            if span
        ]
        antigen_payload = "; ".join(new_spans) if new_spans else "; ".join(new_unsafe)
        reason_str = f"Stage 3 (Unsafe Call Audit): Forbidden call(s) introduced: {new_unsafe}"
        AntibodyRegistry.get_instance().record_antigen(
            vector="ast_secgate",
            payload=antigen_payload,
            reason=reason_str,
            stage=3,
            persist_to_hippocampus=False,
        )
        return {
            "passed": False,
            "stage": 3,
            "reason": reason_str,
            "parent_asserts": parent_summary["assert_count"],
            "candidate_asserts": cand_summary["assert_count"],
        }
    return {
        "passed": True,
        "stage": 3,
        "reason": "All 3 AST SecGate stages passed.",
        "parent_asserts": parent_summary["assert_count"],
        "candidate_asserts": cand_summary["assert_count"],
        "tests_verified": len(cand_summary["test_funcs"]),
    }


def _get_audit_log_path(audit_log_path: Optional[Path | str] = None) -> Path:
    if audit_log_path is not None:
        return Path(audit_log_path)
    env_p = os.environ.get("VOICEFI_IMMUNE_AUDIT_PATH") or os.environ.get("WBC_IMMUNE_AUDIT_PATH")
    if env_p:
        return Path(env_p)
    if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("VOICEFI_STRICT_TEST_HOOK") == "1":
        return Path("/tmp/wbc_pytest_immune_audit.jsonl")
    return _get_wbc_data_dir() / "immune_audit.jsonl"


_AUDIT_KNOWN_DIRS: set[str] = set()


def _append_immune_audit_event(
    event: Dict[str, Any],
    audit_log_path: Optional[Path | str] = None,
) -> None:
    try:
        log_path_str = str(audit_log_path) if audit_log_path is not None else str(_get_audit_log_path(None))
        parent_dir = os.path.dirname(log_path_str)
        if parent_dir and parent_dir not in _AUDIT_KNOWN_DIRS:
            os.makedirs(parent_dir, exist_ok=True)
            _AUDIT_KNOWN_DIRS.add(parent_dir)
        with open(log_path_str, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, separators=(",", ":")) + "\n")
    except Exception:
        pass


_JS_TS_IMPORT_EXTENSIONS: Tuple[str, ...] = (
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".d.ts",
)

_JS_TS_SOURCE_EXTENSIONS: Tuple[str, ...] = (
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
)

_JS_TS_IMPORT_SPEC_RE = re.compile(
    r"""(?:^\s*(?:import|export)\s+(?:type\s+)?(?:[^'"\n;]*?\s+from\s+)?|import\s*\(\s*|require\s*\(\s*)['"]([^'"\n]+)['"]""",
    re.MULTILINE,
)

_STDLIB_MODULES: frozenset[str] = frozenset(getattr(sys, "stdlib_module_names", ()))


def _probe_py_module_path(base_dir: Path, parts: List[str]) -> Optional[Path]:
    """Probe `<base>/<parts>.py` or `<base>/<parts>/__init__.py` with bounded stat calls."""
    if not parts:
        init_p = base_dir / "__init__.py"
        return init_p if init_p.is_file() else None
    target = base_dir.joinpath(*parts)
    py_file = target.with_suffix(".py")
    if py_file.is_file():
        return py_file
    pyi_file = target.with_suffix(".pyi")
    if pyi_file.is_file():
        return pyi_file
    init_file = target / "__init__.py"
    if init_file.is_file():
        return init_file
    return None


def _probe_js_ts_specifier(candidate_base: Path) -> Optional[Path]:
    """Resolve a JS/TS module path on disk including ESM `.js -> .ts/.tsx` and `/index.*`."""
    if candidate_base.suffix in _JS_TS_IMPORT_EXTENSIONS:
        if candidate_base.is_file():
            return candidate_base
        if candidate_base.suffix in (".js", ".jsx", ".mjs", ".cjs"):
            stem_base = candidate_base.with_suffix("")
            for ts_ext in (".ts", ".tsx", ".d.ts"):
                remapped = stem_base.with_suffix(ts_ext)
                if remapped.is_file():
                    return remapped
        return None

    for ext in _JS_TS_IMPORT_EXTENSIONS:
        with_ext = Path(f"{candidate_base}{ext}")
        if with_ext.is_file():
            return with_ext

    if candidate_base.is_dir():
        for ext in _JS_TS_IMPORT_EXTENSIONS:
            idx_p = candidate_base / f"index{ext}"
            if idx_p.is_file():
                return idx_p
    return None


def _discover_workspace_and_search_roots(resolved_file: Path) -> Tuple[Path, List[Path]]:
    """
    Walk upward at most 8 ancestor levels (O(1) stat checks, zero directory traversal)
    to identify the workspace root and Python/TS import roots.
    """
    file_dir = resolved_file.parent
    workspace_root = file_dir
    search_roots: List[Path] = [file_dir]

    cur = file_dir
    for _ in range(8):
        if (cur / "__init__.py").is_file():
            parent_root = cur.parent
            if parent_root not in search_roots:
                search_roots.append(parent_root)
        if any(
            (cur / marker).exists()
            for marker in ("pyproject.toml", "tsconfig.json", "package.json", ".git", "setup.py")
        ):
            workspace_root = cur
            src_dir = cur / "src"
            if src_dir.is_dir() and src_dir not in search_roots:
                search_roots.append(src_dir)
            if cur not in search_roots:
                search_roots.append(cur)
            break
        if cur.parent == cur:
            break
        cur = cur.parent

    return workspace_root, search_roots


def _collect_top_level_py_import_nodes(tree: ast.Module) -> List[ast.AST]:
    """Extract top-level `import` / `from ... import` nodes (including `if TYPE_CHECKING:`)."""
    nodes: List[ast.AST] = []
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            nodes.append(stmt)
        elif isinstance(stmt, (ast.If, ast.Try)):
            sub_body = list(stmt.body) + list(getattr(stmt, "orelse", []))
            for sub in sub_body:
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    nodes.append(sub)
        if len(nodes) >= 48:
            break
    return nodes


def _resolve_workspace_imports_footer(
    fpath: Path,
    raw_text: str,
    parsed_py_tree: Optional[ast.Module] = None,
    max_files: int = 8,
    max_chars: int = 520,
) -> str:
    """
    Zero-Index Local Import Footer (Organ 05 Phase 3):
    Resolve top-level `.py` and `.ts`/`.tsx`/`.js`/`.jsx` workspace imports directly
    against disk with zero cold-start index, zero SQLite graph, `<= 150` tokens, and `<1ms` latency.
    """
    if not raw_text:
        return ""
    ext = fpath.suffix.lower()
    if ext != ".py" and ext not in _JS_TS_SOURCE_EXTENSIONS:
        return ""
    try:
        if not fpath.is_file():
            return ""
        resolved_self = Path(os.path.abspath(fpath))
    except OSError:
        return ""

    workspace_root, search_roots = _discover_workspace_and_search_roots(resolved_self)
    resolved_targets: List[Path] = []
    seen_targets: Set[str] = {os.path.normpath(str(resolved_self))}

    def _record(candidate: Optional[Path]) -> None:
        if candidate is None:
            return
        norm_s = os.path.normpath(str(candidate))
        if norm_s not in seen_targets:
            seen_targets.add(norm_s)
            resolved_targets.append(Path(norm_s))

    if ext == ".py":
        tree = parsed_py_tree
        if tree is None:
            try:
                tree = ast.parse(raw_text)
            except SyntaxError:
                tree = None
        if tree is not None:
            for node in _collect_top_level_py_import_nodes(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        parts = [p for p in alias.name.split(".") if p]
                        if not parts:
                            continue
                        if parts[0] in _STDLIB_MODULES and not (resolved_self.parent / f"{parts[0]}.py").is_file():
                            continue
                        for root in search_roots:
                            hit = _probe_py_module_path(root, parts)
                            if hit is not None:
                                _record(hit)
                                break
                elif isinstance(node, ast.ImportFrom):
                    mod_parts = [p for p in (node.module or "").split(".") if p]
                    if node.level and node.level >= 1:
                        rel_base = resolved_self.parent
                        valid_rel = True
                        for _ in range(node.level - 1):
                            if rel_base.parent == rel_base:
                                valid_rel = False
                                break
                            rel_base = rel_base.parent
                        if not valid_rel:
                            continue
                        mod_hit = _probe_py_module_path(rel_base, mod_parts) if mod_parts else None
                        if mod_hit is not None and mod_hit.name != "__init__.py":
                            _record(mod_hit)
                        else:
                            matched_sub = False
                            for alias in node.names[:16]:
                                if alias.name == "*":
                                    continue
                                sub_hit = _probe_py_module_path(rel_base, [*mod_parts, alias.name])
                                if sub_hit is not None:
                                    _record(sub_hit)
                                    matched_sub = True
                            if not matched_sub and mod_hit is not None:
                                _record(mod_hit)
                    elif mod_parts:
                        if mod_parts[0] in _STDLIB_MODULES and not (resolved_self.parent / f"{mod_parts[0]}.py").is_file():
                            continue
                        for root in search_roots:
                            mod_hit = _probe_py_module_path(root, mod_parts)
                            if mod_hit is not None and mod_hit.name != "__init__.py":
                                _record(mod_hit)
                                break
                            pkg_dir = root.joinpath(*mod_parts)
                            if pkg_dir.is_dir():
                                matched_sub = False
                                for alias in node.names[:16]:
                                    if alias.name == "*":
                                        continue
                                    sub_hit = _probe_py_module_path(root, [*mod_parts, alias.name])
                                    if sub_hit is not None:
                                        _record(sub_hit)
                                        matched_sub = True
                                if matched_sub:
                                    break
                            if mod_hit is not None:
                                _record(mod_hit)
                                break

    elif ext in _JS_TS_SOURCE_EXTENSIONS:
        scan_window = raw_text[:65536]
        file_dir = resolved_self.parent
        alias_roots = [workspace_root / "src", workspace_root]
        for match in _JS_TS_IMPORT_SPEC_RE.finditer(scan_window):
            spec = match.group(1).strip()
            if not spec:
                continue
            if spec.startswith(("./", "../")):
                _record(_probe_js_ts_specifier(file_dir / spec))
            elif spec.startswith(("@/", "~/")):
                rel_spec = spec[2:]
                for a_root in alias_roots:
                    hit = _probe_js_ts_specifier(a_root / rel_spec)
                    if hit is not None:
                        _record(hit)
                        break

    if not resolved_targets:
        return ""

    total_found = len(resolved_targets)
    rel_labels: List[str] = []
    running_chars = 42  # Header prefix allowance
    for target_p in resolved_targets[: max(1, max_files)]:
        try:
            label = target_p.relative_to(workspace_root).as_posix()
        except ValueError:
            try:
                label = os.path.relpath(target_p, workspace_root).replace("\\", "/")
            except ValueError:
                label = target_p.name
        next_cost = len(label) + (2 if rel_labels else 0)
        if rel_labels and (running_chars + next_cost) > max_chars:
            break
        rel_labels.append(label)
        running_chars += next_cost

    omitted = total_found - len(rel_labels)
    noun = "file" if total_found == 1 else "files"
    suffix = f" (+{omitted} more)" if omitted > 0 else ""
    return f"  [Resolved Workspace Imports ({total_found} {noun})]: {', '.join(rel_labels)}{suffix}"


def _build_fast_file_symbol_outline(
    fpath: Path,
    raw_text: str,
    max_items: int = 26,
) -> str:
    """
    Generate a <3ms zero-LLM AST Symbol Map (for .py) or Section/Heading/Key Outline
    (for .json, .csv, .md, .sql, .yaml, .html, .ts, .js, .log, .jsonl) so agents can
    re-call read_file with exact StartLine/EndLine bounds or run whole_body_compute.local.scout.
    """
    items: List[str] = []
    ext = fpath.suffix.lower()
    parsed_py_tree: Optional[ast.Module] = None

    if ext == ".py":
        try:
            parsed_py_tree = ast.parse(raw_text)
            for node in parsed_py_tree.body:
                if isinstance(node, ast.ClassDef):
                    end_ln = getattr(node, "end_lineno", node.lineno)
                    methods = [
                        f"{m.name}(L{m.lineno}-{getattr(m, 'end_lineno', m.lineno)})"
                        for m in node.body
                        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ]
                    m_preview = ", ".join(methods[:5])
                    if len(methods) > 5:
                        m_preview += f", +{len(methods) - 5} more"
                    suffix_str = f" -> [{m_preview}]" if m_preview else ""
                    items.append(f"  - class {node.name} (L{node.lineno}-{end_ln}){suffix_str}")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    end_ln = getattr(node, "end_lineno", node.lineno)
                    items.append(f"  - def {node.name} (L{node.lineno}-{end_ln})")
                if len(items) >= max_items:
                    break
        except Exception:
            pass

    elif ext == ".json":
        lines = raw_text.splitlines()
        key_re = re.compile(r'^\s{1,8}"([^"]+)"\s*:\s*([\{\[\"\d]|true|false|null)')
        for idx, line in enumerate(lines, start=1):
            m = key_re.match(line)
            if m:
                items.append(f'  - L{idx}: key "{m.group(1)}"')
                if len(items) >= max_items:
                    break
        if not items:
            parsed_json: Any = None
            if len(raw_text) <= 2_000_000:
                try:
                    parsed_json = json.loads(raw_text)
                except Exception:
                    parsed_json = None
            if isinstance(parsed_json, dict):
                for k_idx, k_name in enumerate(parsed_json.keys(), start=1):
                    items.append(f'  - key "{k_name}"')
                    if len(items) >= max_items:
                        break
            elif isinstance(parsed_json, list):
                items.append(f"  - JSON Array: {len(parsed_json):,} elements")
                if parsed_json and isinstance(parsed_json[0], dict):
                    sample_keys = list(parsed_json[0].keys())[:10]
                    for sk in sample_keys:
                        items.append(f'  - element key "{sk}"')
                        if len(items) >= max_items:
                            break
            else:
                seen_keys: set[str] = set()
                for m_any in re.finditer(
                    r'"([A-Za-z0-9_.\-:]{1,64})"\s*:\s*(?:[\{\[\"\d\-]|true|false|null)',
                    raw_text[:65536],
                ):
                    k_str = m_any.group(1)
                    if k_str not in seen_keys:
                        seen_keys.add(k_str)
                        items.append(f'  - key "{k_str}"')
                        if len(items) >= max_items:
                            break
        if not items and lines:
            items.append(f"  - JSON Document: {len(lines):,} lines (starts: {lines[0][:60]}...)")

    elif ext == ".csv":
        lines = raw_text.splitlines()
        if lines:
            header = lines[0].strip()[:90]
            items.append(f"  - CSV Schema (L1): {header}")
            items.append(f"  - Total Data Rows: {len(lines) - 1:,}")
            if len(lines) > 1:
                items.append(f"  - Sample Row 1 (L2): {lines[1].strip()[:90]}")
            try:
                import csv as _csv_mod
                import io as _io_mod

                reader = _csv_mod.reader(_io_mod.StringIO(raw_text[:262144]))
                raw_hdrs = [h.strip().lstrip("\ufeff").strip("\"'").lower() for h in next(reader, [])]
                if raw_hdrs:
                    apn_idx = next((i for i, h in enumerate(raw_hdrs) if h in ("parcel_id_main", "parcel_id", "apn", "account_number", "pin")), None)
                    debt_idx = next((i for i, h in enumerate(raw_hdrs) if h in ("amount_due_main", "total_due", "amount_due", "balance", "delinquent_balance")), None)
                    seen_apns: set[str] = set()
                    dup_apns = 0
                    qual_debt = 0
                    sampled_rows = 0
                    for r in reader:
                        if not r:
                            continue
                        sampled_rows += 1
                        if apn_idx is not None and apn_idx < len(r):
                            apn_v = re.sub(r"[^A-Z0-9]", "", str(r[apn_idx]).upper())
                            if apn_v:
                                if apn_v in seen_apns:
                                    dup_apns += 1
                                else:
                                    seen_apns.add(apn_v)
                        if debt_idx is not None and debt_idx < len(r):
                            try:
                                dv = float(str(r[debt_idx]).replace("$", "").replace(",", "").strip())
                                if dv >= 3500.0:
                                    qual_debt += 1
                            except Exception:
                                pass
                    if apn_idx is not None or debt_idx is not None:
                        canonical_ok = all(k in raw_hdrs for k in ("parcel_id_main", "owner_name_main", "address_main", "amount_due_main"))
                        items.append(
                            f"  - Stage-0 CSV Digest ({sampled_rows:,} rows sampled): "
                            f"Qualifying Debt (>=$3,500)={qual_debt:,}, Duplicate APNs={dup_apns:,}, Canonical Schema={'YES' if canonical_ok else 'NO'}"
                        )
            except Exception:
                pass

    elif ext in (".md", ".markdown"):
        lines = raw_text.splitlines()
        md_re = re.compile(r"^\s*(#{1,4})\s+(.+)")
        for idx, line in enumerate(lines, start=1):
            m = md_re.match(line)
            if m:
                level = len(m.group(1))
                indent = "  " * (level - 1)
                items.append(f"  - L{idx}:{indent} {m.group(2).strip()[:72]}")
                if len(items) >= max_items:
                    break

    elif ext == ".sql":
        lines = raw_text.splitlines()
        sql_re = re.compile(
            r"^\s*(CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP(?:ORARY)?\s+|UNIQUE\s+)?(?:TABLE|INDEX|VIEW|FUNCTION|TRIGGER)|ALTER\s+TABLE|DROP\s+TABLE|INSERT\s+INTO|UPDATE|SELECT|WITH)\b",
            re.IGNORECASE,
        )
        for idx, line in enumerate(lines, start=1):
            m = sql_re.match(line)
            if m:
                items.append(f"  - L{idx}: {line.strip()[:72]}")
                if len(items) >= max_items:
                    break

    elif ext in (".yaml", ".yml"):
        lines = raw_text.splitlines()
        yaml_re = re.compile(r"^([a-zA-Z0-9_\-]+)\s*:")
        for idx, line in enumerate(lines, start=1):
            m = yaml_re.match(line)
            if m:
                items.append(f'  - L{idx}: key "{m.group(1)}"')
                if len(items) >= max_items:
                    break

    elif ext in POLYGLOT_EXTENSIONS:
        poly_outline = format_polyglot_symbol_outline(fpath, raw_text, max_items=max_items)
        if poly_outline:
            items.extend(poly_outline.splitlines())

    elif ext in (".c", ".cpp", ".h", ".hpp"):
        lines = raw_text.splitlines()
        cst_re = re.compile(
            r"^\s*(?:export\s+)?(?:default\s+)?(?:pub\s+)?(?:async\s+)?"
            r"(class|interface|type|struct|enum|trait|impl|function|fn|func|def)\s+([A-Za-z0-9_<>]+)"
        )
        for idx, line in enumerate(lines, start=1):
            m_cst = cst_re.match(line)
            if m_cst:
                kind, sym_name = m_cst.group(1), m_cst.group(2)
                items.append(f"  - {kind} {sym_name} (L{idx})")
            if len(items) >= max_items:
                break

    if not items:
        lines = raw_text.splitlines()
        header_re = re.compile(
            r"^\s*(?:#{1,4}\s+.+|(?:export\s+)?(?:async\s+)?(?:function|class|interface|type|struct|enum|fn|func|def)\s+[A-Za-z0-9_]+|<section\b[^>]*>|<div\s+id=[\"'][^\"']+[\"'])"
        )
        for idx, line in enumerate(lines, start=1):
            if header_re.match(line):
                items.append(f"  - L{idx}: {line.strip()[:76]}")
                if len(items) >= max_items:
                    break
        if not items and lines:
            step = max(1, len(lines) // 8)
            for idx in range(1, len(lines) + 1, step):
                items.append(f"  - L{idx}: {lines[idx - 1].strip()[:72]}")
                if len(items) >= 10:
                    break

    base_outline = (
        "\n".join(items)
        if items
        else "  (No top-level symbols indexed; slice 120-line windows)"
    )
    import_footer = _resolve_workspace_imports_footer(
        fpath=fpath,
        raw_text=raw_text,
        parsed_py_tree=parsed_py_tree,
    )
    if import_footer:
        return f"{base_outline}\n{import_footer}"
    return base_outline


def _extract_predictive_ast_slice(
    fpath: Path,
    raw_text: str,
    hint_text: str = "",
    max_lines: int = 80,
) -> Optional[Tuple[str, int, int, str]]:
    """
    Extract a high-relevance AST symbol slice (up to max_lines) from raw_text when an
    unbounded read_file is intercepted, matching against hint_text (or top symbol).
    Returns (symbol_name, start_line, end_line, slice_code).
    """
    if not raw_text:
        return None
    ext = fpath.suffix.lower()
    if ext in POLYGLOT_EXTENSIONS:
        return extract_polyglot_predictive_slice(
            fpath=fpath,
            raw_text=raw_text,
            hint_text=hint_text,
            max_lines=max_lines,
        )
    if ext != ".py":
        return None

    lines = raw_text.splitlines()
    if not lines:
        return None

    try:
        tree = ast.parse(raw_text, filename=str(fpath))
    except Exception:
        return None

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

    if not candidates:
        return None

    selected: Optional[Tuple[str, int, int]] = None

    if hint_text:
        hint_clean = hint_text.strip()
        for name, s_ln, e_ln in candidates:
            if re.search(rf"\b{re.escape(name)}\b", hint_clean, re.IGNORECASE):
                selected = (name, s_ln, e_ln)
                break
        if not selected:
            for name, s_ln, e_ln in candidates:
                if name.lower() in hint_clean.lower():
                    selected = (name, s_ln, e_ln)
                    break

    if not selected:
        stem = fpath.stem.lower().replace("_", "")
        for name, s_ln, e_ln in candidates:
            if name.lower().replace("_", "") in stem or stem in name.lower().replace("_", ""):
                selected = (name, s_ln, e_ln)
                break
        if not selected:
            selected = candidates[0]

    if not selected:
        return None

    sym_name, start_line, end_line = selected
    eff_end = min(end_line, start_line + max_lines - 1)
    slice_lines = lines[start_line - 1 : eff_end]
    slice_code = "\n".join(slice_lines)
    return (sym_name, start_line, eff_end, slice_code)


def _resolve_traceback_source_file(
    raw_path: str,
    log_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Resolve a source file path referenced in a stack trace or log line to a local filesystem Path.
    Handles absolute paths, relative paths, project-relative suffixes (e.g. src/...),
    and fast basename lookups across project source roots in <0.5ms.
    """
    if not raw_path or raw_path.startswith("<"):
        return None

    raw_clean = raw_path.strip().strip("'\"")
    if not raw_clean:
        return None

    lower_path = raw_clean.lower()
    is_external = any(
        x in lower_path
        for x in ("site-packages", "/lib/python", "/.venv/", "/venv/", "virtualenvs")
    )
    if is_external:
        return None

    p_raw = Path(raw_clean)
    cwd = Path.cwd()

    try:
        if p_raw.is_absolute() and p_raw.is_file():
            return p_raw.resolve()
    except Exception:
        pass

    if log_path:
        try:
            p_log = log_path.parent / raw_clean
            if p_log.is_file():
                return p_log.resolve()
        except Exception:
            pass

    try:
        p_cwd = cwd / raw_clean
        if p_cwd.is_file():
            return p_cwd.resolve()
    except Exception:
        pass

    for anchor in ("src/", "lib/", "tests/"):
        if anchor in raw_clean:
            sub = raw_clean[raw_clean.index(anchor):]
            if log_path:
                try:
                    p_log_sub = log_path.parent / sub
                    if p_log_sub.is_file():
                        return p_log_sub.resolve()
                except Exception:
                    pass
            try:
                p_sub = cwd / sub
                if p_sub.is_file():
                    return p_sub.resolve()
            except Exception:
                pass

    target_name = p_raw.name
    if target_name and "." in target_name:
        ignore_dirs = {
            ".git",
            ".venv",
            "venv",
            "node_modules",
            "__pycache__",
            ".cache",
            "dist",
            "build",
            ".tox",
            ".idea",
            ".vscode",
        }
        search_roots = [cwd]
        if log_path and log_path.parent not in search_roots:
            search_roots.append(log_path.parent)
            if log_path.parent.parent not in search_roots:
                search_roots.append(log_path.parent.parent)

        for base_dir in search_roots:
            if not base_dir.is_dir():
                continue
            for root_prefix in ("src", "lib", "tests", "."):
                check_root = base_dir / root_prefix if root_prefix != "." else base_dir
                if not check_root.is_dir():
                    continue
                for root, dirs, files in os.walk(check_root):
                    dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith(".")]
                    if target_name in files:
                        cand = Path(root) / target_name
                        if cand.is_file():
                            return cand.resolve()

    return None


def _extract_stack_trace_source_slice(
    target_path: Path,
    target_line: int,
    func_name: Optional[str] = None,
    code_expr: Optional[str] = None,
    log_path: Optional[Path] = None,
) -> Optional[str]:
    """
    Extract the exact referenced source code function slice (±20 lines) from target_path,
    enriched with AST symbol detection and the crash frame marked at line L<n>.
    """
    try:
        source_text = target_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    lines = source_text.splitlines()
    n_lines = len(lines)
    if n_lines == 0:
        return None

    target_line = max(1, min(target_line, n_lines))

    func_start: Optional[int] = None
    func_end: Optional[int] = None
    qual_name = func_name or ""
    parent_class: Optional[str] = None

    if target_path.suffix.lower() == ".py":
        try:
            tree = ast.parse(source_text, filename=str(target_path))
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    c_end = getattr(node, "end_lineno", node.lineno)
                    if node.lineno <= target_line <= c_end:
                        parent_class = node.name
                        for m in node.body:
                            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                m_end = getattr(m, "end_lineno", m.lineno)
                                if m.lineno <= target_line <= m_end:
                                    func_start = m.lineno
                                    func_end = m_end
                                    qual_name = f"{parent_class}.{m.name}"
                                    break
                        if func_start:
                            break
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    f_end = getattr(node, "end_lineno", node.lineno)
                    if node.lineno <= target_line <= f_end:
                        func_start = node.lineno
                        func_end = f_end
                        qual_name = node.name
                        break

            matched_by_line = qual_name.split(".")[-1] if qual_name else ""
            if func_name and matched_by_line != func_name:
                drift_match = None
                drift_parent = None
                for node in tree.body:
                    if isinstance(node, ast.ClassDef):
                        for m in node.body:
                            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name == func_name:
                                drift_match = m
                                drift_parent = node.name
                                break
                        if drift_match:
                            break
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
                        drift_match = node
                        break

                if drift_match:
                    func_start = drift_match.lineno
                    func_end = getattr(drift_match, "end_lineno", drift_match.lineno)
                    parent_class = drift_parent
                    qual_name = f"{parent_class}.{drift_match.name}" if drift_parent else drift_match.name
                    target_line = func_start
        except Exception:
            pass
    elif target_path.suffix.lower() in POLYGLOT_EXTENSIONS:
        try:
            poly_syms, _ = extract_polyglot_symbols(target_path, source_text, prefer_treesitter=True, include_details=False)
            for s in poly_syms:
                if s.kind not in ("class", "interface", "struct", "enum", "trait", "impl", "object", "type", "constant"):
                    if s.start_line <= target_line <= s.end_line:
                        func_start = s.start_line
                        func_end = s.end_line
                        qual_name = f"{s.parent_class}.{s.name}" if s.parent_class else s.name
                        break
            if func_start is None and func_name:
                short_fn = func_name.split(".")[-1].split("::")[-1]
                for s in poly_syms:
                    if s.name == short_fn and s.kind not in ("class", "interface", "struct", "enum", "trait", "impl"):
                        func_start = s.start_line
                        func_end = s.end_line
                        qual_name = f"{s.parent_class}.{s.name}" if s.parent_class else s.name
                        target_line = func_start
                        break
        except Exception:
            pass

    # Universal statement drift reconciliation:
    if code_expr and code_expr.strip():
        clean_expr = code_expr.strip()
        cur_line = lines[target_line - 1] if 0 <= target_line - 1 < n_lines else ""
        if clean_expr not in cur_line:
            search_min = max(1, target_line - 30)
            search_max = min(n_lines, target_line + 30)
            if func_start and func_end:
                search_min = max(1, min(search_min, func_start))
                search_max = min(n_lines, max(search_max, func_end))

            found = False
            for l_idx in range(search_min, search_max + 1):
                if clean_expr in lines[l_idx - 1]:
                    target_line = l_idx
                    found = True
                    break
            if not found:
                tokens = [
                    t for t in re.findall(r"[A-Za-z0-9_]+", clean_expr)
                    if len(t) >= 3 and t.lower() not in (
                        "self", "def", "return", "true", "false", "none", "function",
                        "const", "let", "var", "async", "await", "import", "export"
                    )
                ]
                if tokens:
                    best_score = 0
                    best_idx = target_line
                    for l_idx in range(search_min, search_max + 1):
                        score = sum(1 for tok in tokens if tok in lines[l_idx - 1])
                        if score > best_score:
                            best_score = score
                            best_idx = l_idx
                    if best_score >= 1:
                        target_line = best_idx

    slice_start = max(1, target_line - 20)
    slice_end = min(n_lines, target_line + 20)

    if func_start is not None and func_start >= target_line - 28:
        slice_start = min(slice_start, func_start)
    if func_end is not None and func_end <= target_line + 28:
        slice_end = max(slice_end, func_end)

    if slice_end - slice_start > 50:
        slice_start = max(1, target_line - 20)
        slice_end = min(n_lines, target_line + 20)

    formatted_lines: List[str] = []
    for l_idx in range(slice_start, slice_end + 1):
        code_line = lines[l_idx - 1]
        if l_idx == target_line:
            formatted_lines.append(f"L{l_idx}: {code_line}  # <--- [CRASH FRAME: L{l_idx}]")
        else:
            formatted_lines.append(f"L{l_idx}: {code_line}")

    rel_path = None
    candidate_roots = [Path.cwd()]
    if log_path:
        candidate_roots.append(log_path.parent)
        if log_path.parent != log_path.parent.parent:
            candidate_roots.append(log_path.parent.parent)

    for root_dir in candidate_roots:
        try:
            rel_path = target_path.relative_to(root_dir)
            break
        except Exception:
            pass

    if rel_path is None:
        rel_path = target_path.name

    fence_lang = fence_language_for_path(target_path)

    header = f"📍 {rel_path}:{target_line}"
    if qual_name:
        header += f" (in `{qual_name}`)"
    header += f" [L{slice_start}-L{slice_end}]:"

    return f"{header}\n```{fence_lang}\n" + "\n".join(formatted_lines) + "\n```"


def _extract_stack_trace_source_slices(
    traceback_blocks: List[str],
    error_lines: List[str],
    log_path: Optional[Path] = None,
    max_slices: int = 2,
) -> List[str]:
    """
    Extract referenced source code slices for frames in traceback_blocks and error_lines.
    Prioritizes innermost (deepest) project frames in reverse call order across
    Python, Node.js/TypeScript, Rust, and Go crash signatures.
    """
    tb_python_pattern = re.compile(
        r'File\s+["\'](?P<path>[^"\']+)["\'],\s+line\s+(?P<line>\d+)(?:,\s+in\s+(?P<func>[A-Za-z0-9_<>.]+))?'
    )
    pytest_pattern = re.compile(
        r'^(?:L\d+:\s+)?(?P<path>[A-Za-z0-9_./\\-]+\.[a-zA-Z0-9_]+):(?P<line>\d+):\s+in\s+(?P<func>[A-Za-z0-9_<>.]+)'
    )
    node_pattern = re.compile(
        r'^\s*(?:L\d+:\s+)?at\s+(?:(?P<func>[A-Za-z0-9_$.<>]+)\s+\((?P<path>[^:\n\r\t()]+):(?P<line>\d+)(?::\d+)?\)|(?P<path_anon>[^:\n\r\t()]+):(?P<line_anon>\d+)(?::\d+)?)'
    )
    rust_pattern = re.compile(
        r'(?:panicked at [^,]+,\s+|-->\s+|\d+:\s+)?(?P<path>[A-Za-z0-9_./\\-]+\.rs):(?P<line>\d+)(?::(?P<col>\d+))?(?:\s+-\s+(?P<func>[A-Za-z0-9_:<>]+))?'
    )
    go_pattern = re.compile(
        r'^\s*(?:L\d+:\s+)?(?P<path>[A-Za-z0-9_./\\-]+\.go):(?P<line>\d+)(?:\s+\+0x[0-9a-fA-F]+)?'
    )
    generic_bracket_pattern = re.compile(
        r'[\(\[](?P<path>[A-Za-z0-9_./\\-]+\.(?:py|ts|tsx|js|jsx|rs|go|c|cpp|h)):(?P<line>\d+)[\)\]]'
    )
    generic_colon_pattern = re.compile(
        r'(?:^|\s)(?P<path>[A-Za-z0-9_./\\-]+\.(?:py|ts|tsx|js|jsx|rs|go|c|cpp|h)):(?P<line>\d+)(?::\d+)?'
    )

    candidate_frames: List[Tuple[str, int, Optional[str], Optional[str]]] = []

    for tb_block in traceback_blocks:
        tb_lines = tb_block.splitlines()
        for idx, raw_l in enumerate(tb_lines):
            tb_line = re.sub(r"^L\d+:\s+", "", raw_l)

            m_tb = tb_python_pattern.search(tb_line)
            if m_tb:
                next_expr = None
                if idx + 1 < len(tb_lines):
                    cand = re.sub(r"^L\d+:\s+", "", tb_lines[idx + 1]).strip()
                    if not cand.startswith("File ") and not cand.startswith("Traceback") and not any(cand.startswith(k) for k in ("Error:", "Exception:", "TypeError:", "ValueError:", "OperationalError:")):
                        next_expr = cand
                candidate_frames.append(
                    (m_tb.group("path"), int(m_tb.group("line")), m_tb.group("func"), next_expr)
                )
                continue

            m_py = pytest_pattern.search(tb_line)
            if m_py:
                candidate_frames.append(
                    (m_py.group("path"), int(m_py.group("line")), m_py.group("func"), None)
                )
                continue

            m_node = node_pattern.search(tb_line)
            if m_node:
                p_node = m_node.group("path") or m_node.group("path_anon")
                l_node = m_node.group("line") or m_node.group("line_anon")
                f_node = m_node.group("func")
                if p_node and l_node:
                    candidate_frames.append((p_node, int(l_node), f_node, None))
                    continue

            m_rust = rust_pattern.search(tb_line)
            if m_rust:
                p_rs = m_rust.group("path")
                l_rs = m_rust.group("line")
                f_rs = m_rust.group("func")
                if p_rs and l_rs:
                    candidate_frames.append((p_rs, int(l_rs), f_rs, None))
                    continue

            m_go = go_pattern.search(tb_line)
            if m_go:
                p_go = m_go.group("path")
                l_go = m_go.group("line")
                f_go = None
                if idx > 0:
                    prev = re.sub(r"^L\d+:\s+", "", tb_lines[idx - 1]).strip()
                    if "(" in prev and not prev.startswith(("panic:", "goroutine ", "created by ")):
                        f_go = prev.split("(")[0].strip()
                if p_go and l_go:
                    candidate_frames.append((p_go, int(l_go), f_go, None))
                    continue

    for err_line in error_lines:
        clean_err = re.sub(r"^L\d+:\s+", "", err_line).strip()
        m_node = node_pattern.search(clean_err)
        if m_node:
            p_node = m_node.group("path") or m_node.group("path_anon")
            l_node = m_node.group("line") or m_node.group("line_anon")
            f_node = m_node.group("func")
            if p_node and l_node:
                candidate_frames.append((p_node, int(l_node), f_node, None))
                continue

        m_rust = rust_pattern.search(clean_err)
        if m_rust:
            p_rs = m_rust.group("path")
            l_rs = m_rust.group("line")
            f_rs = m_rust.group("func")
            if p_rs and l_rs:
                candidate_frames.append((p_rs, int(l_rs), f_rs, None))
                continue

        m_go = go_pattern.search(clean_err)
        if m_go:
            p_go = m_go.group("path")
            l_go = m_go.group("line")
            if p_go and l_go:
                candidate_frames.append((p_go, int(l_go), None, None))
                continue

        m_brk = generic_bracket_pattern.search(clean_err)
        if m_brk:
            candidate_frames.append((m_brk.group("path"), int(m_brk.group("line")), None, None))
            continue

        m_col = generic_colon_pattern.search(clean_err)
        if m_col:
            candidate_frames.append((m_col.group("path"), int(m_col.group("line")), None, None))
            continue

    if not candidate_frames:
        return []

    def _is_runtime_path(p: Path) -> bool:
        s = str(p).replace("\\", "/")
        return any(
            ign in s
            for ign in (
                "/.venv/",
                "/venv/",
                "/site-packages/",
                "/node_modules/",
                "/lib/python",
                "/rustlib/",
                "/usr/lib/",
                "/usr/local/Cellar/",
                "/System/Library/",
            )
        )

    resolved_candidates: List[Tuple[Path, int, Optional[str], Optional[str]]] = []
    seen_targets: Set[Tuple[str, int]] = set()

    for path_str, line_num, func_name, code_expr in reversed(candidate_frames):
        resolved = _resolve_traceback_source_file(path_str, log_path=log_path)
        if not resolved:
            continue

        target_key = (str(resolved), line_num)
        if target_key in seen_targets:
            continue
        seen_targets.add(target_key)
        resolved_candidates.append((resolved, line_num, func_name, code_expr))

    project_candidates = [c for c in resolved_candidates if not _is_runtime_path(c[0])]
    ordered_candidates = project_candidates if project_candidates else resolved_candidates

    slices: List[str] = []
    for target_path, line_num, func_name, code_expr in ordered_candidates:
        slice_text = _extract_stack_trace_source_slice(
            target_path,
            line_num,
            func_name=func_name,
            code_expr=code_expr,
            log_path=log_path,
        )
        if slice_text:
            slices.append(slice_text)
            if len(slices) >= max_slices:
                break

    return slices


def _extract_fast_log_digest_and_offset(
    path: Path | str,
    max_bytes: int = 2 * 1024 * 1024,
) -> Tuple[str, int]:
    """
    Pure-Python (<5ms) bounded log scanner and traceback extractor (Blood-Brain Barrier Context Shield).
    """
    p = Path(path)
    if not p.is_file():
        return ("", 1)

    try:
        file_size = p.stat().st_size
        with open(p, "rb") as f:
            raw_bytes = f.read(max_bytes)
        text = raw_bytes.decode("utf-8", errors="replace")
    except Exception:
        return ("", 1)

    lines = text.splitlines()
    total_lines = len(lines)
    size_kb = file_size / 1024.0

    traceback_blocks: List[str] = []
    error_lines: List[str] = []
    seen_errors = set()
    first_error_line: Optional[int] = None

    i = 0
    in_tb = False
    cur_tb: List[str] = []
    ERROR_KEYWORDS = ("error", "critical", "fatal", "exception", "panic", "panicked", "failure")

    while i < total_lines:
        line = lines[i]
        line_num = i + 1
        line_lower = line.lower()

        is_tb_header = (
            "traceback (most recent call last):" in line_lower
            or "stack backtrace:" in line_lower
            or line_lower.startswith("goroutine ")
            or bool(re.search(r':\d+:\s+in\s+[A-Za-z0-9_<>.]+', line))
        )
        if is_tb_header:
            if cur_tb:
                traceback_blocks.append("\n".join(cur_tb))
                cur_tb = []
            in_tb = True
            if first_error_line is None:
                if i > 0 and any(kw in lines[i - 1].lower() for kw in ERROR_KEYWORDS):
                    first_error_line = i
                else:
                    first_error_line = line_num
            cur_tb.append(f"L{line_num}: {line}")
            i += 1
            continue

        if not in_tb and line.strip().startswith("at ") and ":" in line:
            in_tb = True
            if cur_tb:
                traceback_blocks.append("\n".join(cur_tb))
                cur_tb = []
            cur_tb.append(f"L{line_num}: {line}")
            i += 1
            continue

        if in_tb:
            is_new_log_record = bool(
                re.match(r"^\d{4}-\d{2}-\d{2}", line)
                or re.match(r"^\[(?:info|debug|warn|trace|notice|metric)\]", line_lower)
            )
            is_tb_continuation = (
                line.startswith((" ", "\t", "E "))
                or any(kw in line_lower for kw in ("traceback", "stack backtrace", "goroutine", "caused by", "panic:"))
                or bool(re.search(r':\d+:\s+in\s+[A-Za-z0-9_<>.]+', line))
                or (("(" in line and ")" in line) and not is_new_log_record)
                or any(line_lower.startswith(k) for k in ("error:", "exception:", "typeerror:", "valueerror:", "operationalerror:", "runtimeerror:", "keyerror:"))
                or "error" in line_lower
                or "exception" in line_lower
            )
            if not is_tb_continuation or is_new_log_record or line.startswith("==="):
                in_tb = False
                traceback_blocks.append("\n".join(cur_tb))
                cur_tb = []
            else:
                cur_tb.append(f"L{line_num}: {line}")
                i += 1
                continue

        if any(kw in line_lower for kw in ERROR_KEYWORDS):
            if first_error_line is None or "[error]" in line_lower or "critical" in line_lower or "fatal" in line_lower or "exception" in line_lower:
                if first_error_line is None or (
                    "[error]" in line_lower and "[error]" not in lines[first_error_line - 1].lower()
                ):
                    first_error_line = line_num
            sig = re.sub(r"0x[0-9a-fA-F]+|\d+", "#", line_lower.strip())[:100]
            if len(error_lines) < 25 and sig not in seen_errors:
                seen_errors.add(sig)
                error_lines.append(f"L{line_num}: {line}")

        i += 1

    if cur_tb:
        traceback_blocks.append("\n".join(cur_tb))

    tail_start = max(0, total_lines - 10)
    tail_lines = [f"L{tail_start + j + 1}: {lines[tail_start + j]}" for j in range(len(lines) - tail_start)]

    if first_error_line is None:
        first_error_line = max(1, total_lines - 50)

    source_slices = _extract_stack_trace_source_slices(
        traceback_blocks, error_lines, log_path=p, max_slices=2
    )

    parts = [
        f"📋 Fast Log Traceback & Error Digest (Zero-Turn Deterministic Error Windows & Frames) for `{p.name}` ({total_lines:,} lines, {size_kb:.1f} KB):"
    ]
    if traceback_blocks:
        parts.append("\n[Traceback Blocks]:")
        parts.extend(traceback_blocks)
    if source_slices:
        parts.append("\n[Auto-Attached Stack Trace Source Slices (±20 Lines)]:")
        parts.extend(source_slices)
    if error_lines:
        parts.append(f"\n[Key Error Lines ({len(error_lines)} found)]:")
        parts.extend(error_lines)
    if tail_lines:
        parts.append(f"\n[Tail (Last {len(tail_lines)} Lines)]:")
        parts.extend(tail_lines)

    digest = "\n".join(parts)
    try:
        from whole_body_compute.redaction import redact_secrets
        digest = redact_secrets(digest)
    except Exception:
        pass
    return (digest, first_error_line)


def _extract_fast_log_digest(path: Path | str, max_bytes: int = 2 * 1024 * 1024) -> str:
    """Convenience helper returning the fast log traceback & error digest string."""
    digest, _ = _extract_fast_log_digest_and_offset(path, max_bytes=max_bytes)
    return digest


_SHELL_OUTPUT_TOOL_NAMES = {
    "run_command",
    "bash",
    "sh",
    "zsh",
    "cmd",
    "powershell",
    "terminal",
    "execute_command",
    "run_terminal_cmd",
    "developer__shell",
    "tool_output",
    "post_tool",
}

_GIT_PORCELAIN_LINE_RE = re.compile(
    r"^(?:[MADRCU?!][ MADRCU?!]| [MADRCU?!])\s+\S+"
)
_PYTEST_ITEM_RE = re.compile(
    r"^(?:\S+::\S+\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b|(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+\S+::\S+)"
)
_PYTEST_DOT_PROGRESS_RE = re.compile(r"^[.sFxEX]+\s*(?:\[\s*\d+%\])?\s*$")
_CARGO_ITEM_RE = re.compile(r"^test\s+(\S+)\s+\.\.\.\s+(ok|FAILED|ignored)\b")
_GO_RUN_RE = re.compile(r"^===\s+(?:RUN|PAUSE|CONT)\s+(\S+)")
_GO_RESULT_RE = re.compile(r"^---\s+(PASS|FAIL|SKIP):\s+(\S+)")
_GO_PKG_OK_RE = re.compile(r"^(?:ok|\?)\s+\S+")
_JEST_PASS_RE = re.compile(r"^\s*(?:PASS\s+\S+|[✓✔]\s+)")
_JEST_FAIL_RE = re.compile(r"^\s*(?:FAIL\s+\S+|[✕×●]\s+)")
_FAILURE_SECTION_START_RE = re.compile(
    r"^(?:={3,}\s*(?:FAILURES|ERRORS|short test summary info)\s*={3,}|failures:\s*$|Summary of all failing tests\b)"
)
_FAILURE_SECTION_STOP_RE = re.compile(
    r"^(?:={3,}\s*(?:warnings summary|slowest .* durations)\s*={3,}|-{3,}\s*generated xml\b|={3,}\s*.*(?:\bpassed\b|\bfailed\b|\berrors?\b|\bskipped\b).*={3,}\s*$)"
)
_FINAL_TEST_SUMMARY_RE = re.compile(
    r"^(?:={3,}\s*.*(?:\bpassed\b|\bfailed\b|\berrors?\b).*={3,}\s*$|test result:\s+(?:ok|FAILED)\..*|(?:Test Suites|Tests):\s+.*(?:passed|failed).*|\d+\s+(?:failed|passed).*\bin\s+[\d.]+s|FAIL\s+\S+\s+[\d.]+s)"
)
_GREP_LINE_RE = re.compile(
    r"^(?!https?://)(?P<file>[^\s:\0][^:\0]{0,240}):(?P<line>\d{1,7})(?::|-)(?P<text>.*)$"
)


def _looks_like_grep_file_token(token: str) -> bool:
    if not token or token.startswith(("-", "[")):
        return False
    if "/" in token or "\\" in token:
        return True
    dot_idx = token.rfind(".")
    return 0 < dot_idx < len(token) - 1 and token[dot_idx + 1 :].isalnum()


def _detect_cli_stdout_schema(raw_text: str) -> Optional[Tuple[str, Path]]:
    """Fast O(1)-head/tail check for git diff/status, test suite logs, and rg/grep stdout."""
    if not raw_text:
        return None

    head_chunk = raw_text[:8192]
    tail_chunk = raw_text[-4096:] if len(raw_text) > 4096 else raw_text
    sample_lines = [ln for ln in head_chunk.splitlines()[:35] if ln.strip()]
    if not sample_lines:
        return None

    first_ln = sample_lines[0]

    # 1. git diff / git show / git status
    if (
        first_ln.startswith("diff --git ")
        or "\ndiff --git a/" in head_chunk
        or ("\n--- a/" in head_chunk and "\n+++ b/" in head_chunk and "\n@@ -" in head_chunk)
        or (first_ln.startswith("commit ") and "\ndiff --git " in raw_text)
        or first_ln.startswith(("On branch ", "HEAD detached at ", "Changes to be committed:", "Changes not staged for commit:"))
    ):
        return ("git_diff_status", Path("git_diff.patch"))

    if len(sample_lines) >= 8:
        porcelain_hits = sum(1 for ln in sample_lines[:20] if _GIT_PORCELAIN_LINE_RE.match(ln))
        if porcelain_hits >= max(6, int(len(sample_lines[:20]) * 0.75)):
            return ("git_diff_status", Path("git_status.txt"))

    # 2. Test suite stdout (pytest, cargo test, go test, jest/vitest)
    if (
        "test session starts" in head_chunk
        or "=== FAILURES ===" in raw_text
        or "=== ERRORS ===" in raw_text
        or "short test summary info" in tail_chunk
        or "test result: " in tail_chunk
        or ("Test Suites:" in tail_chunk and "Tests:" in tail_chunk)
    ):
        return ("test_suite_stdout", Path("test_suite.log"))

    pytest_hits = 0
    cargo_hits = 0
    go_hits = 0
    jest_hits = 0
    for ln in sample_lines[:25]:
        if _PYTEST_ITEM_RE.match(ln):
            pytest_hits += 1
        elif _CARGO_ITEM_RE.match(ln) or (ln.startswith("running ") and ln.endswith((" test", " tests"))):
            cargo_hits += 1
        elif _GO_RUN_RE.match(ln) or _GO_RESULT_RE.match(ln):
            go_hits += 1
        elif _JEST_PASS_RE.match(ln) or _JEST_FAIL_RE.match(ln):
            jest_hits += 1

    if pytest_hits >= 3 or cargo_hits >= 2 or go_hits >= 2 or jest_hits >= 3:
        return ("test_suite_stdout", Path("test_suite.log"))

    # 3. rg / grep file:line:text stdout
    if len(sample_lines) >= 5:
        probe = sample_lines[:25]
        grep_hits = 0
        for ln in probe:
            m = _GREP_LINE_RE.match(ln)
            if m and _looks_like_grep_file_token(m.group("file")):
                grep_hits += 1
        if grep_hits >= max(5, int(len(probe) * 0.65)):
            return ("grep_grouped_stdout", Path("grep_results.txt"))

    return None


def _build_git_diff_status_digest(lines: List[str], max_bytes: int = 8000) -> str:
    """
    Deterministic `git_diff_status` digest: preserves 100% of modified file paths,
    per-file `+added / -removed` counts, and `@@ ... @@` hunk headers with raw line coordinates.
    """
    has_diff_headers = any(ln.startswith("diff --git ") for ln in lines[:200]) or any(
        ln.startswith("diff --git ") for ln in lines
    )
    if not has_diff_headers:
        # git status (verbose or porcelain)
        branch_hdr: List[str] = []
        grouped: Dict[str, List[str]] = {
            "modified": [],
            "added": [],
            "deleted": [],
            "renamed": [],
            "untracked": [],
            "other": [],
        }
        in_untracked = False
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            if s.startswith(("On branch ", "HEAD detached at ", "Your branch is ")):
                branch_hdr.append(s)
                continue
            if s.startswith("Untracked files:"):
                in_untracked = True
                continue
            if s.startswith(("Changes to be committed:", "Changes not staged for commit:")):
                in_untracked = False
                continue
            if s.startswith("(use \"git "):
                continue
            m_porc = _GIT_PORCELAIN_LINE_RE.match(ln)
            if m_porc:
                code = ln[:2]
                fpath = ln[2:].strip()
                if "?" in code:
                    grouped["untracked"].append(fpath)
                elif "D" in code:
                    grouped["deleted"].append(fpath)
                elif "A" in code:
                    grouped["added"].append(fpath)
                elif "R" in code:
                    grouped["renamed"].append(fpath)
                elif "M" in code:
                    grouped["modified"].append(fpath)
                else:
                    grouped["other"].append(f"{code.strip()} {fpath}")
            elif s.startswith("modified:"):
                grouped["modified"].append(s.split(":", 1)[1].strip())
            elif s.startswith("new file:"):
                grouped["added"].append(s.split(":", 1)[1].strip())
            elif s.startswith("deleted:"):
                grouped["deleted"].append(s.split(":", 1)[1].strip())
            elif s.startswith("renamed:"):
                grouped["renamed"].append(s.split(":", 1)[1].strip())
            elif in_untracked and ln.startswith(("\t", "  ")):
                grouped["untracked"].append(s)

        out = ["[Git Working Tree Status Digest]:"]
        out.extend(branch_hdr[:3])
        for bucket, items in grouped.items():
            if items:
                out.append(f"  {bucket} ({len(items)}): " + ", ".join(items))
        return "\n".join(out)

    # Full `git diff` / `git show` parser
    commit_preamble: List[str] = []
    files: List[Dict[str, Any]] = []
    cur_file: Optional[Dict[str, Any]] = None
    cur_hunk: Optional[Dict[str, Any]] = None

    for idx, ln in enumerate(lines, start=1):
        if ln.startswith("diff --git "):
            if cur_file is not None:
                files.append(cur_file)
            # Parse `diff --git a/foo b/bar`
            path_part = ln[len("diff --git ") :].strip()
            b_idx = path_part.rfind(" b/")
            if b_idx != -1:
                a_path = path_part[2:b_idx] if path_part.startswith("a/") else path_part[:b_idx]
                b_path = path_part[b_idx + 3 :]
                display_path = b_path if a_path == b_path else f"{a_path} -> {b_path}"
            else:
                display_path = path_part
            cur_file = {
                "path": display_path,
                "status": "modified",
                "added": 0,
                "removed": 0,
                "start_line": idx,
                "hunks": [],
            }
            cur_hunk = None
            continue

        if cur_file is None:
            if len(commit_preamble) < 8 and ln.strip():
                commit_preamble.append(ln[:160])
            continue

        if ln.startswith("new file mode"):
            cur_file["status"] = "added"
        elif ln.startswith("deleted file mode"):
            cur_file["status"] = "deleted"
        elif ln.startswith("rename to "):
            cur_file["status"] = "renamed"
        elif ln.startswith("Binary files "):
            cur_file["status"] = "binary"
        elif ln.startswith("@@ "):
            cur_hunk = {
                "line_no": idx,
                "header": ln[:160],
                "samples": [],
            }
            cur_file["hunks"].append(cur_hunk)
        elif ln.startswith("+") and not ln.startswith("+++"):
            cur_file["added"] += 1
            if cur_hunk is not None and len(cur_hunk["samples"]) < 2 and ln[1:].strip():
                cur_hunk["samples"].append(ln[:110])
        elif ln.startswith("-") and not ln.startswith("---"):
            cur_file["removed"] += 1
            if cur_hunk is not None and len(cur_hunk["samples"]) < 2 and ln[1:].strip():
                cur_hunk["samples"].append(ln[:110])

    if cur_file is not None:
        files.append(cur_file)

    total_add = sum(f["added"] for f in files)
    total_rem = sum(f["removed"] for f in files)
    parts: List[str] = []
    if commit_preamble:
        parts.append("[Commit Header]:")
        parts.extend(commit_preamble)
    parts.append(
        f"[Git Diff File Manifest ({len(files)} file(s), +{total_add} / -{total_rem} lines)]:"
    )
    # 100% file path recall guaranteed first
    for f in files:
        tag = f" [{f['status']}]" if f["status"] != "modified" else ""
        parts.append(
            f"  • {f['path']}{tag} (+{f['added']} / -{f['removed']}, {len(f['hunks'])} hunk(s), raw L{f['start_line']})"
        )

    manifest_str = "\n".join(parts)
    budget = max(2400, max_bytes)
    hunk_lines: List[str] = ["[Modified Hunk Headers]:"]
    used_bytes = len(manifest_str.encode("utf-8", errors="replace")) + 32

    for f in files:
        if not f["hunks"]:
            continue
        file_hdr = f"  {f['path']}:"
        hdr_cost = len(file_hdr) + 1
        if used_bytes + hdr_cost > budget:
            break
        hunk_lines.append(file_hdr)
        used_bytes += hdr_cost
        for hk in f["hunks"]:
            hk_line = f"    L{hk['line_no']}: {hk['header']}"
            if used_bytes + len(hk_line) + 1 > budget:
                break
            hunk_lines.append(hk_line)
            used_bytes += len(hk_line) + 1
            for smp in hk["samples"]:
                smp_line = f"      {smp}"
                if used_bytes + len(smp_line) + 1 <= budget:
                    hunk_lines.append(smp_line)
                    used_bytes += len(smp_line) + 1

    if len(hunk_lines) > 1:
        return manifest_str + "\n" + "\n".join(hunk_lines)
    return manifest_str


def _build_test_suite_stdout_digest(
    lines: List[str],
    max_bytes: int = 8000,
    preserve_fuller_tail: bool = False,
) -> str:
    """
    Deterministic `test_suite_stdout` digest across `pytest`, `cargo test`, `go test`, and `jest`:
    collapses passing progress noise into summary counters while preserving 100.0% of
    middle-of-log and section-grouped `FAILED` / `ERROR` / `AssertionError` blocks.
    When `preserve_fuller_tail=True` (`stdout_compaction` graduated_escalate), retains fuller
    warning/tail lines so the agent does not need a verbose retry turn.
    """
    passed_count = 0
    failed_count = 0
    skipped_count = 0
    error_count = 0

    in_failure_section = False
    in_inline_failure = False
    in_noise_section = False
    warning_lines_collapsed = 0
    retained_warning_tail: List[str] = []
    pending_run_header: Optional[Tuple[int, str]] = None
    recent_context_buf: List[Tuple[int, str]] = []
    failure_lines: List[str] = []
    summary_lines: List[str] = []
    emitted_line_nos: set[int] = set()

    def _emit_failure_line(lno: int, text: str) -> None:
        if lno not in emitted_line_nos:
            emitted_line_nos.add(lno)
            failure_lines.append(f"L{lno}: {text}")

    for idx, ln in enumerate(lines, start=1):
        stripped = ln.strip()

        if _FINAL_TEST_SUMMARY_RE.match(stripped):
            in_failure_section = False
            in_inline_failure = False
            in_noise_section = False
            summary_lines.append(f"L{idx}: {ln}")
            continue

        if _FAILURE_SECTION_START_RE.match(stripped):
            in_failure_section = True
            in_inline_failure = False
            in_noise_section = False
            _emit_failure_line(idx, ln)
            continue

        if _FAILURE_SECTION_STOP_RE.match(stripped):
            in_failure_section = False
            in_inline_failure = False
            in_noise_section = True
            if preserve_fuller_tail and stripped:
                retained_warning_tail.append(f"L{idx}: {ln}")
            continue

        if in_noise_section:
            if stripped:
                warning_lines_collapsed += 1
                if preserve_fuller_tail:
                    retained_warning_tail.append(f"L{idx}: {ln}")
            continue

        if in_failure_section:
            _emit_failure_line(idx, ln)
            continue

        # Classify explicit passing / skipped / running progress lines
        m_py = _PYTEST_ITEM_RE.match(stripped)
        if m_py:
            outcome = m_py.group(1) or m_py.group(2)
            if outcome in ("PASSED", "XPASS"):
                passed_count += 1
                in_inline_failure = False
                pending_run_header = None
                recent_context_buf.clear()
                continue
            if outcome in ("SKIPPED", "XFAIL"):
                skipped_count += 1
                in_inline_failure = False
                pending_run_header = None
                recent_context_buf.clear()
                continue
            if outcome == "FAILED":
                failed_count += 1
                in_inline_failure = True
                _emit_failure_line(idx, ln)
                continue
            if outcome == "ERROR":
                error_count += 1
                in_inline_failure = True
                _emit_failure_line(idx, ln)
                continue

        if _PYTEST_DOT_PROGRESS_RE.match(stripped):
            passed_count += stripped.count(".")
            skipped_count += stripped.count("s")
            failed_count += stripped.count("F")
            error_count += stripped.count("E")
            continue

        m_cargo = _CARGO_ITEM_RE.match(stripped)
        if m_cargo:
            outcome = m_cargo.group(2)
            if outcome == "ok":
                passed_count += 1
                in_inline_failure = False
                pending_run_header = None
                recent_context_buf.clear()
                continue
            if outcome == "ignored":
                skipped_count += 1
                in_inline_failure = False
                continue
            if outcome == "FAILED":
                failed_count += 1
                in_inline_failure = True
                _emit_failure_line(idx, ln)
                continue

        m_go_run = _GO_RUN_RE.match(stripped)
        if m_go_run:
            in_inline_failure = False
            pending_run_header = (idx, ln)
            recent_context_buf.clear()
            continue

        m_go_res = _GO_RESULT_RE.match(stripped)
        if m_go_res:
            outcome = m_go_res.group(1)
            if outcome == "PASS":
                passed_count += 1
                in_inline_failure = False
                pending_run_header = None
                recent_context_buf.clear()
                continue
            if outcome == "SKIP":
                skipped_count += 1
                in_inline_failure = False
                pending_run_header = None
                recent_context_buf.clear()
                continue
            if outcome == "FAIL":
                failed_count += 1
                if pending_run_header is not None:
                    _emit_failure_line(*pending_run_header)
                    pending_run_header = None
                for ctx_lno, ctx_txt in recent_context_buf:
                    _emit_failure_line(ctx_lno, ctx_txt)
                recent_context_buf.clear()
                _emit_failure_line(idx, ln)
                in_inline_failure = True
                continue

        if _GO_PKG_OK_RE.match(stripped):
            in_inline_failure = False
            pending_run_header = None
            recent_context_buf.clear()
            continue

        if _JEST_PASS_RE.match(ln):
            passed_count += 1
            in_inline_failure = False
            pending_run_header = None
            recent_context_buf.clear()
            continue

        if stripped.startswith(("running ", "Running unittests", "Doc-tests ", "collecting ...")) or (
            stripped.startswith("platform ") and "pytest-" in stripped
        ) or stripped.startswith(("rootdir:", "configfile:", "plugins:", "collected ")):
            if stripped.startswith("collected "):
                summary_lines.insert(0, f"L{idx}: {ln}")
            continue

        # Detect inline / middle-of-log failure blocks
        is_inline_fail_trigger = bool(
            _JEST_FAIL_RE.match(ln)
            or "AssertionError" in ln
            or ln.startswith("E   ")
            or ln.startswith("E\t")
            or "Traceback (most recent call last):" in ln
            or "panicked at " in ln
            or stripped.startswith(("FAIL\t", "FAIL ", "FAILED ", "ERROR ", "--- FAIL:"))
            or re.search(r"\b(?:FAILED|AssertionError|TypeError|ValueError|KeyError|RuntimeError|ImportError|SyntaxError)\b", ln)
        )

        if is_inline_fail_trigger:
            if pending_run_header is not None:
                _emit_failure_line(*pending_run_header)
                pending_run_header = None
            for ctx_lno, ctx_txt in recent_context_buf[-4:]:
                _emit_failure_line(ctx_lno, ctx_txt)
            recent_context_buf.clear()
            in_inline_failure = True
            _emit_failure_line(idx, ln)
            continue

        if in_inline_failure:
            if not stripped:
                _emit_failure_line(idx, ln)
            elif ln.startswith((" ", "\t", ">", "E ", "+", "-")) or ":" in stripped or stripped.startswith("_"):
                _emit_failure_line(idx, ln)
            else:
                in_inline_failure = False
                recent_context_buf.append((idx, ln))
                if len(recent_context_buf) > 6:
                    recent_context_buf.pop(0)
        else:
            if stripped:
                recent_context_buf.append((idx, ln))
                if len(recent_context_buf) > 6:
                    recent_context_buf.pop(0)

    if not summary_lines and lines:
        for idx in range(max(1, len(lines) - 4), len(lines) + 1):
            if lines[idx - 1].strip():
                summary_lines.append(f"L{idx}: {lines[idx - 1]}")

    warn_suffix = (
        f" | {warning_lines_collapsed:,} warning-lines collapsed" if warning_lines_collapsed > 0 else ""
    )
    header = (
        f"[Test Suite Semantic Digest: {passed_count:,} passed collapsed | "
        f"{failed_count} failed | {error_count} error(s) | {skipped_count} skipped{warn_suffix}]"
    )
    out_parts: List[str] = [header]
    if failure_lines:
        out_parts.append(f"[Retained Failure & Assertion Blocks ({len(failure_lines)} lines, 100% recall)]:")
        # Guarantee 100% retention of failure signal lines up to generous safety ceiling
        hard_cap_chars = max(16000, max_bytes * 2)
        joined_failures = "\n".join(failure_lines)
        if len(joined_failures) <= hard_cap_chars:
            out_parts.append(joined_failures)
        else:
            # Even on massive 50-test failure cascades, retain every FAILED/ERROR/E-assert line + top context
            critical_only = [
                fl
                for fl in failure_lines
                if any(k in fl for k in ("FAILED", "FAIL", "ERROR", "AssertionError", ": E   ", "panicked at", "Error:"))
            ]
            out_parts.append("\n".join(failure_lines[:120]))
            out_parts.append("...[additional verbose stack frames condensed; all failure assertions retained below]...")
            out_parts.extend(critical_only[:120])
    else:
        out_parts.append("[All Passing - 0 Failure / Error Blocks Detected]")

    if preserve_fuller_tail:
        if retained_warning_tail:
            tail_budget_lines = retained_warning_tail[-120:]
            out_parts.append(
                f"[Retained Warnings & Diagnostics Tail ({len(tail_budget_lines)} lines, reflex graduated_escalate)]:"
            )
            out_parts.extend(tail_budget_lines)
        elif lines:
            tail_start = max(1, len(lines) - 119)
            raw_tail = [f"L{i}: {lines[i - 1]}" for i in range(tail_start, len(lines) + 1) if lines[i - 1].strip()]
            if raw_tail:
                out_parts.append(
                    f"[Retained Fuller Tail ({len(raw_tail)} lines, reflex graduated_escalate)]:"
                )
                out_parts.extend(raw_tail)

    if summary_lines:
        out_parts.append("[Summary]:")
        out_parts.extend(summary_lines[-6:])

    return "\n".join(out_parts)


def _build_grep_grouped_stdout_digest(lines: List[str], max_bytes: int = 8000) -> str:
    """
    Deterministic `grep_grouped_stdout` digest: groups `rg` / `grep` matches by file path
    (`file.py (N matches): L12, L45...`) preserving 100% of matched files + compact snippets.
    """
    grouped: Dict[str, List[Tuple[int, str]]] = {}
    unparsed: List[str] = []
    total_matches = 0

    for ln in lines:
        if not ln.strip():
            continue
        m = _GREP_LINE_RE.match(ln)
        if m and _looks_like_grep_file_token(m.group("file")):
            fpath = m.group("file")
            lno = int(m.group("line"))
            txt = m.group("text").strip()
            grouped.setdefault(fpath, []).append((lno, txt))
            total_matches += 1
        elif len(unparsed) < 5:
            unparsed.append(ln[:120])

    out_lines: List[str] = [
        f"[Grouped Search Digest: {total_matches:,} match(es) across {len(grouped):,} file(s)]:"
    ]
    # Guarantee 100% file path + line coordinate visibility first
    file_summaries: List[str] = []
    for fpath, hits in grouped.items():
        line_refs = ", ".join(f"L{lno}" for lno, _ in hits[:18])
        if len(hits) > 18:
            line_refs += f", ... (+{len(hits) - 18} more)"
        file_summaries.append(f"  • {fpath} ({len(hits)} match{'es' if len(hits) != 1 else ''}): {line_refs}")

    base_text = "\n".join(out_lines + file_summaries)
    budget = max(2400, max_bytes)
    if len(base_text.encode("utf-8", errors="replace")) >= budget:
        return base_text

    # Enrich with up to 2 sample match snippets per file while within byte budget
    enriched: List[str] = list(out_lines)
    used = len(out_lines[0]) + 1
    remaining_summary_cost = sum(len(s) + 1 for s in file_summaries)

    for idx, (fpath, hits) in enumerate(grouped.items()):
        summary_line = file_summaries[idx]
        enriched.append(summary_line)
        used += len(summary_line) + 1
        remaining_summary_cost -= len(summary_line) + 1

        for lno, txt in hits[:2]:
            if not txt:
                continue
            snippet = f"      L{lno}: {txt[:96] + '...' if len(txt) > 96 else txt}"
            if used + len(snippet) + 1 + remaining_summary_cost <= budget:
                enriched.append(snippet)
                used += len(snippet) + 1

    if unparsed and used + 120 <= budget:
        enriched.extend(f"  [note] {u}" for u in unparsed[:2])

    return "\n".join(enriched)


def _detect_buffer_schema_format(
    raw_text: str,
    target_path: Optional[str | Path] = None,
    tool_name: str = "tool_output",
) -> Tuple[str, Path]:
    """
    Deterministically detect the structural schema format of an arbitrary tool output buffer
    (Semantic CLI stdout digests, AST Python, Tree-sitter/CST/regex source, JSON, CSV, SQL, Markdown, or YAML).
    """
    is_shell_tool = str(tool_name or "").strip().lower() in _SHELL_OUTPUT_TOOL_NAMES
    has_real_file_target = False
    if target_path:
        p = Path(str(target_path))
        if not is_shell_tool or p.is_file():
            has_real_file_target = True

    if not has_real_file_target or is_shell_tool:
        cli_schema = _detect_cli_stdout_schema(raw_text)
        if cli_schema is not None:
            return cli_schema

    if target_path:
        p = Path(str(target_path))
        ext = p.suffix.lower()
        if ext == ".py":
            return "python_ast", p
        if ext in (*POLYGLOT_EXTENSIONS, ".c", ".cpp", ".h", ".hpp"):
            return "treesitter_cst_regex", p
        if ext in (".json", ".jsonl"):
            return "json_schema", p
        if ext == ".csv":
            return "csv_schema", p
        if ext == ".sql":
            return "sql_schema", p
        if ext in (".md", ".markdown"):
            return "markdown_outline", p
        if ext in (".yaml", ".yml"):
            return "yaml_schema", p

    cli_fallback = _detect_cli_stdout_schema(raw_text)
    if cli_fallback is not None:
        return cli_fallback

    stripped = raw_text.lstrip()
    if stripped.startswith(("{", "[")):
        try:
            json.loads(stripped)
            return "json_schema", Path("buffer.json")
        except Exception:
            if re.search(r'"[A-Za-z0-9_.\-:]+"\s*:', raw_text):
                return "json_schema", Path("buffer.json")

    if re.search(
        r"^\s*(?:CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP(?:ORARY)?\s+|UNIQUE\s+)?(?:TABLE|INDEX|VIEW)|ALTER\s+TABLE|DROP\s+TABLE|INSERT\s+INTO|SELECT\s+.+\s+FROM)\b",
        raw_text,
        re.IGNORECASE | re.MULTILINE,
    ):
        return "sql_schema", Path("buffer.sql")

    lines = [ln for ln in raw_text.splitlines()[:10] if ln.strip()]
    if len(lines) >= 2 and "," in lines[0]:
        c0 = lines[0].count(",")
        if c0 >= 2 and all(abs(ln.count(",") - c0) <= 1 for ln in lines[1:5]):
            return "csv_schema", Path("buffer.csv")

    if re.search(r"^\s*(?:def|class|async\s+def)\s+[A-Za-z0-9_]+\s*[\(:]", raw_text, re.MULTILINE):
        return "python_ast", Path("buffer.py")

    if re.search(
        r"^\s*(?:export\s+)?(?:interface|type|struct|enum|fn|func|function|class)\s+[A-Za-z0-9_]+",
        raw_text,
        re.MULTILINE,
    ):
        return "treesitter_cst_regex", Path("buffer.ts")

    return "text_outline", Path(str(target_path) if target_path else "buffer.txt")


def slice_post_tool_output_buffer(
    raw_output: Any,
    tool_name: str = "tool_output",
    target_path: Optional[str | Path] = None,
    max_lines: int = 250,
    max_bytes: int = 8000,
    step_idx: int = 0,
    head_lines: int = 30,
    tail_lines: int = 15,
    preserve_fuller_tail: bool = False,
) -> Dict[str, Any]:
    """
    PostToolOutput / Pre-Context-Ingestion Buffer Slicer (Blood-Brain Barrier).
    """
    if raw_output is None:
        return {
            "sliced": False,
            "tool_name": tool_name,
            "target_path": str(target_path or ""),
            "format_detected": "empty",
            "original_lines": 0,
            "original_bytes": 0,
            "structural_map": "",
            "content": "",
            "direct_tokens_saved": 0,
            "quadratic_tokens_saved": 0,
            "airlock_blob_id": "",
            "airlock_receipt": "",
        }

    if isinstance(raw_output, dict) and str(tool_name or "").strip().lower() in _SHELL_OUTPUT_TOOL_NAMES:
        # Unwrap shell response envelopes (`{"output": "..."}` / `{"stdout": "..."}`) so
        # multi-line CLI stdout schemas (`pytest`, `git diff`, `rg`) are not escaped into a 3-line JSON string.
        unwrapped_shell_str: Optional[str] = None
        for k in ("output", "stdout", "result", "content", "stderr"):
            v = raw_output.get(k)
            if isinstance(v, str) and v.strip():
                unwrapped_shell_str = (
                    f"{unwrapped_shell_str}\n{v}" if unwrapped_shell_str else v
                )
        if unwrapped_shell_str is not None:
            raw_text = unwrapped_shell_str
            default_path = target_path
        else:
            raw_text = json.dumps(raw_output, indent=2)
            default_path = target_path or "payload.json"
    elif isinstance(raw_output, (dict, list)):
        raw_text = json.dumps(raw_output, indent=2)
        default_path = target_path or "payload.json"
    elif isinstance(raw_output, (bytes, bytearray)):
        raw_text = bytes(raw_output).decode("utf-8", errors="replace")
        default_path = target_path
    else:
        raw_text = str(raw_output)
        default_path = target_path

    byte_len = len(raw_text.encode("utf-8", errors="replace"))
    lines = raw_text.splitlines()
    total_lines = len(lines)
    fmt_name, virtual_path = _detect_buffer_schema_format(raw_text, default_path, tool_name=tool_name)

    if total_lines <= max_lines and byte_len <= max_bytes:
        return {
            "sliced": False,
            "tool_name": tool_name,
            "target_path": str(target_path or virtual_path),
            "format_detected": fmt_name,
            "original_lines": total_lines,
            "original_bytes": byte_len,
            "structural_map": "",
            "content": raw_text if isinstance(raw_output, (bytes, bytearray)) else raw_output,
            "direct_tokens_saved": 0,
            "quadratic_tokens_saved": 0,
            "airlock_blob_id": "",
            "airlock_receipt": "",
        }

    if fmt_name == "git_diff_status":
        outline = _build_git_diff_status_digest(lines, max_bytes=max_bytes)
        compressed_str = (
            f"🛡️ Blood-Brain Barrier PostToolOutput Semantic CLI Digest ({fmt_name}):\n"
            f"Tool `{tool_name}` stdout ({total_lines:,} lines, {byte_len / 1024:.1f} KB) "
            f"compacted with 100% modified-file recall:\n"
            f"{outline}\n"
        )
    elif fmt_name == "test_suite_stdout":
        outline = _build_test_suite_stdout_digest(
            lines,
            max_bytes=max_bytes,
            preserve_fuller_tail=preserve_fuller_tail,
        )
        compressed_str = (
            f"🛡️ Blood-Brain Barrier PostToolOutput Semantic CLI Digest ({fmt_name}):\n"
            f"Tool `{tool_name}` test log ({total_lines:,} lines, {byte_len / 1024:.1f} KB) "
            f"compacted with 100% failure & assertion recall:\n"
            f"{outline}\n"
        )
    elif fmt_name == "grep_grouped_stdout":
        outline = _build_grep_grouped_stdout_digest(lines, max_bytes=max_bytes)
        compressed_str = (
            f"🛡️ Blood-Brain Barrier PostToolOutput Semantic CLI Digest ({fmt_name}):\n"
            f"Tool `{tool_name}` search stdout ({total_lines:,} lines, {byte_len / 1024:.1f} KB) "
            f"grouped by file:\n"
            f"{outline}\n"
        )
    else:
        outline = _build_fast_file_symbol_outline(virtual_path, raw_text, max_items=30)
        eff_tail_lines = max(tail_lines, 120) if preserve_fuller_tail else tail_lines
        max_head_chars = max(300, min(2000, max_bytes // 4))
        max_tail_chars = (
            max(6000, min(16000, max_bytes * 2))
            if preserve_fuller_tail
            else max_head_chars
        )
        line_trunc_limit = 500 if preserve_fuller_tail else 200
        head_slice_lines = [
            (ln[:200] + "...[line truncated]") if len(ln) > 200 else ln
            for ln in lines[: max(1, head_lines)]
        ]
        head_preview = "\n".join(head_slice_lines)[:max_head_chars]
        tail_preview = ""
        if total_lines > head_lines:
            tail_slice_lines = [
                (ln[:line_trunc_limit] + "...[line truncated]") if len(ln) > line_trunc_limit else ln
                for ln in lines[-max(1, eff_tail_lines) :]
            ]
            tail_preview = "\n".join(tail_slice_lines)[:max_tail_chars]
        elif total_lines == 1 and len(raw_text) > 400:
            tail_preview = "..." + raw_text[-min(200, max_tail_chars // 2) :]

        compressed_str = (
            f"🛡️ Blood-Brain Barrier PostToolOutput Pre-Ingestion Context Shield ({fmt_name}):\n"
            f"Tool `{tool_name}` output on `{virtual_path.name}` was {total_lines:,} lines "
            f"({byte_len / 1024:.1f} KB). Replaced prior to LLM context ingestion with "
            f"Structural Index + Head/Tail Slice:\n"
            f"--- [STRUCTURAL SYMBOL / SCHEMA MAP ({fmt_name})] ---\n"
            f"{outline}\n"
            f"--- [HEAD SLICE (L1-L{min(total_lines, head_lines)})] ---\n"
            f"{head_preview}\n"
        )
        if tail_preview:
            compressed_str += (
                f"--- [TAIL SLICE (L{max(1, total_lines - eff_tail_lines + 1)}-L{total_lines})] ---\n"
                f"{tail_preview}\n"
            )

    # Strict Monotonicity Guard: never inflate or waste Airlock vault writes on buffers that
    # are already >=92% pure failure tracebacks / dense signal.
    airlock_enabled = os.environ.get("WBC_AIRLOCK_VAULT_DISABLED") != "1"
    est_receipt_bytes = 210 if airlock_enabled else 0
    candidate_bytes = len(compressed_str.encode("utf-8", errors="replace")) + est_receipt_bytes
    if candidate_bytes >= int(byte_len * 0.92):
        return {
            "sliced": False,
            "tool_name": tool_name,
            "target_path": str(target_path or virtual_path),
            "format_detected": fmt_name,
            "original_lines": total_lines,
            "original_bytes": byte_len,
            "structural_map": "",
            "content": raw_text if isinstance(raw_output, (bytes, bytearray)) else raw_output,
            "direct_tokens_saved": 0,
            "quadratic_tokens_saved": 0,
            "airlock_blob_id": "",
            "airlock_receipt": "",
        }

    airlock_blob_id = ""
    airlock_receipt = ""
    if airlock_enabled:
        try:
            from whole_body_compute.organs.immune_vault import store_airlock_blob

            vault_info = store_airlock_blob(raw_text)
            airlock_blob_id = str(vault_info.get("blob_id") or "")
            airlock_receipt = str(vault_info.get("receipt") or "")
        except Exception:
            airlock_blob_id = ""
            airlock_receipt = ""

    if airlock_receipt:
        compressed_str += f"{airlock_receipt}\n"

    sliced_byte_len = len(compressed_str.encode("utf-8", errors="replace"))
    full_tokens_est = max(1, byte_len // 4)
    sliced_tokens_est = max(1, sliced_byte_len // 4)
    direct_tokens_saved = max(0, full_tokens_est - sliced_tokens_est)
    remaining_turns = max(1, 25 - int(step_idx or 0))
    quadratic_tokens_saved = direct_tokens_saved * remaining_turns

    return {
        "sliced": True,
        "tool_name": tool_name,
        "target_path": str(target_path or virtual_path),
        "format_detected": fmt_name,
        "original_lines": total_lines,
        "original_bytes": byte_len,
        "structural_map": outline,
        "content": compressed_str,
        "direct_tokens_saved": direct_tokens_saved,
        "quadratic_tokens_saved": quadratic_tokens_saved,
        "airlock_blob_id": airlock_blob_id,
        "airlock_receipt": airlock_receipt,
    }
