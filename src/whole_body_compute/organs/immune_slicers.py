"""
Blood-Brain Barrier (Organ 05): AST SecGate, Predictive AST Slicer, Fast Log Digest, & Post-Tool Buffer Slicer.
"""

from __future__ import annotations

import ast
import json
import os
import re
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
    "bin/body-immune-hook",
    "bin/body-immune-watchdog",
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


def _fast_extract_ast_summary(tree: ast.AST) -> Dict[str, Any]:
    assert_count = 0
    taut_asserts = 0
    test_funcs: Set[str] = set()
    unsafe_calls: List[str] = []
    alias_map: Dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                if n.asname:
                    alias_map[n.asname] = n.name
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for n in node.names:
                local_name = n.asname or n.name
                alias_map[local_name] = f"{mod}.{n.name}" if mod else n.name

    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            assert_count += 1
            if _fast_is_tautological(node.test):
                taut_asserts += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_") and not _has_skip_decorator(node):
                test_funcs.add(node.name)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Subscript):
                sub_val = node.func.value
                if isinstance(sub_val, ast.Name) and sub_val.id in ("__builtins__", "builtins"):
                    unsafe_calls.append("__builtins__[...]")
                elif isinstance(sub_val, ast.Call) and isinstance(sub_val.func, ast.Name) and sub_val.func.id in ("globals", "locals", "vars"):
                    unsafe_calls.append(f"{sub_val.func.id}()[...]")
            elif isinstance(node.func, ast.Name):
                resolved = alias_map.get(node.func.id, node.func.id)
                if resolved in ("eval", "exec", "compile", "__import__", "os.system", "os.popen", "pty.spawn"):
                    unsafe_calls.append(resolved)
                elif resolved in ("subprocess.Popen", "subprocess.run", "subprocess.call", "subprocess.check_call", "subprocess.check_output"):
                    for kw in node.keywords:
                        if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and bool(kw.value.value) is True:
                            unsafe_calls.append(f"{resolved}(shell=True)")
                elif node.func.id == "getattr" and len(node.args) >= 2:
                    arg0 = node.args[0]
                    arg1 = node.args[1]
                    arg0_name = alias_map.get(arg0.id, arg0.id) if isinstance(arg0, ast.Name) else ""
                    if isinstance(arg1, ast.Constant):
                        if arg1.value in ("system", "popen", "exec", "eval", "Popen", "spawn"):
                            unsafe_calls.append(f'getattr(..., "{arg1.value}")')
                    elif arg0_name in ("os", "subprocess", "pty", "builtins", "__builtins__", "shutil"):
                        unsafe_calls.append(f"getattr({arg0_name}, <dynamic>)")
            elif isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                val = node.func.value
                raw_id = val.id if isinstance(val, ast.Name) else ("call()" if isinstance(val, ast.Call) else "")
                val_repr = alias_map.get(raw_id, raw_id) if raw_id else ""
                qual = f"{val_repr}.{attr}" if val_repr else attr
                if qual in ("os.system", "os.popen", "pty.spawn") or attr in ("system", "popen", "spawn", "posix_spawn", "execl", "execle", "execlp", "execv", "execve", "execvp"):
                    unsafe_calls.append(qual)
                elif qual in ("subprocess.Popen", "subprocess.run", "subprocess.call", "subprocess.check_call", "subprocess.check_output"):
                    for kw in node.keywords:
                        if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and bool(kw.value.value) is True:
                            unsafe_calls.append(f"{qual}(shell=True)")
                elif qual == "shutil.rmtree" and node.args:
                    arg0 = node.args[0]
                    if isinstance(arg0, ast.Constant) and str(arg0.value).strip() in ("/", "~", ".."):
                        unsafe_calls.append("shutil.rmtree(root)")
    return {
        "assert_count": assert_count,
        "tautological_asserts": taut_asserts,
        "test_funcs": test_funcs,
        "unsafe_calls": unsafe_calls,
    }


def _fast_three_stage_ast_secgate(
    parent_code: str,
    candidate_code: str,
    file_path: str = "",
) -> Dict[str, Any]:
    """Zero-import stdlib 3-Stage AST SecGate for sub-30ms `-S` hook execution."""
    fname = Path(file_path).name if file_path else ""
    if fname and any(fname == pat or file_path.endswith(pat) for pat in PROTECTED_CONFIG_FILES):
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
    try:
        cand_tree = ast.parse(candidate_code)
    except SyntaxError as exc:
        return {
            "passed": False,
            "stage": 1,
            "reason": f"Stage 1 (Syntax Error): {exc.msg} at line {exc.lineno}",
        }

    parent_summary = {"assert_count": 0, "tautological_asserts": 0, "test_funcs": set(), "unsafe_calls": []}
    if parent_code.strip():
        try:
            parent_summary = _fast_extract_ast_summary(ast.parse(parent_code))
        except SyntaxError:
            pass
    cand_summary = _fast_extract_ast_summary(cand_tree)

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
        reason_str = f"Stage 3 (Unsafe Call Audit): Forbidden call(s) introduced: {new_unsafe}"
        AntibodyRegistry.get_instance().record_antigen(
            vector="ast_secgate",
            payload=candidate_code,
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


def _append_immune_audit_event(
    event: Dict[str, Any],
    audit_log_path: Optional[Path | str] = None,
) -> None:
    try:
        log_path = _get_audit_log_path(audit_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception:
        pass


def _build_fast_file_symbol_outline(
    fpath: Path,
    raw_text: str,
    max_items: int = 26,
) -> str:
    """
    Generate a <3ms zero-LLM AST Symbol Map (for .py) or Section/Heading/Key Outline
    (for .json, .csv, .md, .sql, .yaml, .html, .ts, .js, .log, .jsonl) so agents can
    re-call read_file with exact StartLine/EndLine bounds or run voicefi.local.scout.
    """
    items: List[str] = []
    ext = fpath.suffix.lower()

    if ext == ".py":
        try:
            tree = ast.parse(raw_text)
            for node in tree.body:
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

    return "\n".join(items) if items else "  (No top-level symbols indexed; slice 120-line windows)"


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


def _detect_buffer_schema_format(raw_text: str, target_path: Optional[str | Path] = None) -> Tuple[str, Path]:
    """
    Deterministically detect the structural schema format of an arbitrary tool output buffer
    (AST Python, Tree-sitter/CST/regex source, JSON, CSV, SQL, Markdown, or YAML).
    """
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
        }

    if isinstance(raw_output, (dict, list)):
        raw_text = json.dumps(raw_output, indent=2)
        default_path: Optional[str | Path] = target_path or "payload.json"
    elif isinstance(raw_output, (bytes, bytearray)):
        raw_text = bytes(raw_output).decode("utf-8", errors="replace")
        default_path = target_path
    else:
        raw_text = str(raw_output)
        default_path = target_path

    byte_len = len(raw_text.encode("utf-8", errors="replace"))
    lines = raw_text.splitlines()
    total_lines = len(lines)
    fmt_name, virtual_path = _detect_buffer_schema_format(raw_text, default_path)

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
        }

    outline = _build_fast_file_symbol_outline(virtual_path, raw_text, max_items=30)
    max_preview_chars = max(300, min(2000, max_bytes // 4))
    head_slice_lines = [
        (ln[:200] + "...[line truncated]") if len(ln) > 200 else ln
        for ln in lines[: max(1, head_lines)]
    ]
    head_preview = "\n".join(head_slice_lines)[:max_preview_chars]
    tail_preview = ""
    if total_lines > head_lines:
        tail_slice_lines = [
            (ln[:200] + "...[line truncated]") if len(ln) > 200 else ln
            for ln in lines[-max(1, tail_lines) :]
        ]
        tail_preview = "\n".join(tail_slice_lines)[:max_preview_chars]
    elif total_lines == 1 and len(raw_text) > 400:
        tail_preview = "..." + raw_text[-min(200, max_preview_chars // 2) :]

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
            f"--- [TAIL SLICE (L{max(1, total_lines - tail_lines + 1)}-L{total_lines})] ---\n"
            f"{tail_preview}\n"
        )

    full_tokens_est = max(1, byte_len // 4)
    sliced_tokens_est = max(1, len(compressed_str.encode("utf-8", errors="replace")) // 4)
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
    }
