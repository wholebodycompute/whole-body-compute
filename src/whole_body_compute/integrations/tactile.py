"""
Tactile Nerve Endings: Sub-50ms Post-Action Closed-Loop State Delta (Phase 14.2 & Proprioception Phase 3).

Captures a baseline before an agent action and computes the exact post-action
closed-loop state delta:
1. Target-file SHA-256 mutation, AST syntax validity, structural AST digest, and assertion preservation (<2ms)
2. Filesystem/Git status delta (modified/added files, insertions, deletions)
3. macOS surface delta (focused window, active application, focused AXUIElement)
Delivers immediate closed-loop verification so the agent feels the consequences
of its actions in <50ms without waiting for a human readout.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger("whole_body_compute.integrations.tactile")


def _is_tautological_assert(test_node: ast.AST) -> bool:
    """Detect reward-hacking tautological assertions (`assert True`, `assert 1 == 1`, `assert x == x`)."""
    if isinstance(test_node, ast.Constant):
        return bool(test_node.value) is True
    if isinstance(test_node, (ast.List, ast.Tuple, ast.Set)):
        return len(test_node.elts) > 0
    if isinstance(test_node, ast.Dict):
        return len(test_node.keys) > 0
    if isinstance(test_node, ast.BoolOp) and isinstance(test_node.op, ast.Or):
        return any(_is_tautological_assert(v) for v in test_node.values)
    if isinstance(test_node, ast.Compare) and len(test_node.ops) == 1:
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
    if isinstance(test_node, ast.UnaryOp) and isinstance(test_node.op, ast.Not):
        if isinstance(test_node.operand, ast.Constant):
            return bool(test_node.operand.value) is False
    if isinstance(test_node, ast.Call) and isinstance(test_node.func, ast.Name) and test_node.func.id == "bool":
        if test_node.args and isinstance(test_node.args[0], ast.Constant):
            return bool(test_node.args[0].value) is True
    return False


def _has_skip_decorator(node: ast.AST) -> bool:
    for dec in getattr(node, "decorator_list", []):
        dec_target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(dec_target, ast.Name) and "skip" in dec_target.id.lower():
            return True
        if isinstance(dec_target, ast.Attribute) and "skip" in dec_target.attr.lower():
            return True
    return False


def compute_ast_structural_digest(code: str, is_python: bool = True) -> Tuple[bool, Optional[str], str, str, int, int, int, List[str]]:
    """
    Parse code in <1ms and return:
    `(ast_syntax_valid, ast_error, ast_digest, topology_digest, symbol_count, assert_count, tautological_asserts, test_funcs)`.
    """
    if not is_python:
        digest = hashlib.sha256(code.encode("utf-8", errors="replace")).hexdigest()[:16] if code else ""
        return True, None, digest, digest, 0, 0, 0, []

    if not code or not code.strip():
        return True, None, "", "", 0, 0, 0, []

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        err_msg = f"SyntaxError: {exc.msg} at line {exc.lineno}"
        return False, err_msg, "", "", 0, 0, 0, []

    ast_dump_str = ast.dump(tree, include_attributes=False)
    ast_digest = hashlib.sha256(ast_dump_str.encode("utf-8")).hexdigest()[:16]

    symbol_sigs: List[str] = []
    assert_count = 0
    taut_count = 0
    test_funcs: Set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = [
                f"{item.name}({len(item.args.args)})"
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            symbol_sigs.append(f"class:{node.name}:[{','.join(sorted(methods))}]")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arg_names = ",".join(a.arg for a in node.args.args)
            symbol_sigs.append(f"def:{node.name}({arg_names})")
            if node.name.startswith("test_") and not _has_skip_decorator(node):
                test_funcs.add(node.name)
        elif isinstance(node, ast.Assert):
            assert_count += 1
            if _is_tautological_assert(node.test):
                taut_count += 1

    sorted_tests = sorted(test_funcs)
    valid_asserts = max(0, assert_count - taut_count)
    topo_payload = json.dumps(
        {
            "symbols": sorted(symbol_sigs),
            "valid_asserts": valid_asserts,
            "tests": sorted_tests,
        },
        sort_keys=True,
    )
    topology_digest = hashlib.sha256(topo_payload.encode("utf-8")).hexdigest()[:16]
    return (
        True,
        None,
        ast_digest,
        topology_digest,
        len(symbol_sigs),
        assert_count,
        taut_count,
        sorted_tests,
    )


@dataclass
class FileTactileSnapshot:
    """Sub-2ms single-file tactile & AST structural baseline."""

    file_path: str
    exists: bool
    sha256: str = ""
    size_bytes: int = 0
    ast_syntax_valid: bool = True
    ast_error: Optional[str] = None
    ast_digest: str = ""
    topology_digest: str = ""
    symbol_count: int = 0
    assert_count: int = 0
    tautological_asserts: int = 0
    test_funcs: List[str] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def ast_structural_digest(self) -> str:
        return self.ast_digest

    @property
    def ast_topology_digest(self) -> str:
        return self.topology_digest

    @property
    def test_func_names(self) -> List[str]:
        return list(self.test_funcs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "exists": self.exists,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "ast_syntax_valid": self.ast_syntax_valid,
            "ast_error": self.ast_error,
            "ast_digest": self.ast_digest,
            "topology_digest": self.topology_digest,
            "symbol_count": self.symbol_count,
            "assert_count": self.assert_count,
            "tautological_asserts": self.tautological_asserts,
            "test_funcs": list(self.test_funcs),
            "latency_ms": round(self.latency_ms, 3),
        }


@dataclass
class TactileBaseline:
    """Snapshot taken immediately prior to an action."""

    working_dir: Path
    timestamp: float
    app_name: str
    window_title: str
    focused_element_label: str
    git_status_map: Dict[str, str] = field(default_factory=dict)
    git_shortstat: str = ""
    target_file: Optional[str] = None
    file_snapshot: Optional[FileTactileSnapshot] = None

    def compute_delta(
        self,
        working_dir: Optional[Path] = None,
        timeout_s: float = 1.5,
    ) -> "TactileDelta":
        """Compute the post-action delta against this baseline."""
        return TactileSensor.capture_delta(self, working_dir=working_dir, timeout_s=timeout_s)


@dataclass
class TactileDelta:
    """Closed-loop verification delta comparing post-action state against baseline."""

    git_changed: bool
    git_files_modified: List[str] = field(default_factory=list)
    git_files_added: List[str] = field(default_factory=list)
    git_files_deleted: List[str] = field(default_factory=list)
    git_shortstat: str = ""
    window_changed: bool = False
    app_before: str = ""
    app_after: str = ""
    window_title_before: str = ""
    window_title_after: str = ""
    focused_before: str = ""
    focused_after: str = ""
    latency_ms: float = 0.0
    summary: str = ""
    # Phase 3 Closed-Loop File & AST Proprioception fields
    target_file: str = ""
    pre_sha256: str = ""
    post_sha256: str = ""
    file_changed: bool = False
    mechanical_noop: bool = False
    ast_syntax_valid: bool = True
    ast_error: Optional[str] = None
    pre_ast_digest: str = ""
    post_ast_digest: str = ""
    ast_mutated: bool = False
    pre_topology_digest: str = ""
    post_topology_digest: str = ""
    topology_mutated: bool = False
    pre_symbol_count: int = 0
    post_symbol_count: int = 0
    symbol_delta: int = 0
    pre_assert_count: int = 0
    post_assert_count: int = 0
    assert_delta: int = 0
    tautological_asserts_introduced: int = 0
    deleted_test_funcs: List[str] = field(default_factory=list)
    assertions_preserved: bool = True
    net_ast_progress: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "git_changed": self.git_changed,
            "git_files_modified": self.git_files_modified,
            "git_files_added": self.git_files_added,
            "git_files_deleted": self.git_files_deleted,
            "git_shortstat": self.git_shortstat,
            "window_changed": self.window_changed,
            "app_before": self.app_before,
            "app_after": self.app_after,
            "window_title_before": self.window_title_before,
            "window_title_after": self.window_title_after,
            "focused_before": self.focused_before,
            "focused_after": self.focused_after,
            "latency_ms": round(self.latency_ms, 3),
            "summary": self.summary,
            "target_file": self.target_file,
            "pre_sha256": self.pre_sha256,
            "post_sha256": self.post_sha256,
            "file_changed": self.file_changed,
            "mechanical_noop": self.mechanical_noop,
            "ast_syntax_valid": self.ast_syntax_valid,
            "ast_error": self.ast_error,
            "pre_ast_digest": self.pre_ast_digest,
            "post_ast_digest": self.post_ast_digest,
            "ast_mutated": self.ast_mutated,
            "pre_topology_digest": self.pre_topology_digest,
            "post_topology_digest": self.post_topology_digest,
            "topology_mutated": self.topology_mutated,
            "pre_symbol_count": self.pre_symbol_count,
            "post_symbol_count": self.post_symbol_count,
            "symbol_delta": self.symbol_delta,
            "pre_assert_count": self.pre_assert_count,
            "post_assert_count": self.post_assert_count,
            "assert_delta": self.assert_delta,
            "tautological_asserts_introduced": self.tautological_asserts_introduced,
            "deleted_test_funcs": list(self.deleted_test_funcs),
            "assertions_preserved": self.assertions_preserved,
            "net_ast_progress": self.net_ast_progress,
        }

    def to_context_block(self) -> str:
        """Format a compact closed-loop tactile feedback block for LLM prompts."""
        lines = [f"[Tactile Nerve Delta — {self.latency_ms:.1f}ms Closed-Loop Verification]"]
        if self.git_changed:
            files_str = ", ".join(
                self.git_files_modified[:5]
                + self.git_files_added[:3]
                + self.git_files_deleted[:2]
            )
            stat = f" ({self.git_shortstat})" if self.git_shortstat else ""
            lines.append(f"- Filesystem/Git: Modified [{files_str}]{stat}")
        else:
            lines.append("- Filesystem/Git: No file changes observed.")

        if self.window_changed:
            lines.append(
                f"- Surface: Focus shifted from '{self.app_before}' (\"{self.window_title_before}\") -> "
                f"'{self.app_after}' (\"{self.window_title_after}\")"
            )
        else:
            focus_note = f" @ {self.focused_after}" if self.focused_after else ""
            lines.append(f"- Surface: Focus retained on '{self.app_after}' (\"{self.window_title_after}\"){focus_note}")

        if self.summary:
            lines.append(f"- Status: {self.summary}")
        return "\n".join(lines)


class TactileSensor:
    """
    Sub-50ms closed-loop state sensor probing filesystem git status and macOS UI focus.
    """

    def __init__(self, working_dir: Optional[Path | str] = None) -> None:
        self.working_dir = Path(working_dir).resolve() if working_dir else Path.cwd().resolve()

    @classmethod
    def _get_git_status_map(cls, repo_dir: Path, timeout_s: float = 1.0) -> Dict[str, str]:
        if not repo_dir.exists():
            return {}
        try:
            res = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            if res.returncode != 0:
                return {}
            status_map: Dict[str, str] = {}
            for line in res.stdout.splitlines():
                if len(line) >= 4:
                    code = line[:2]
                    file_path = line[3:].strip()
                    status_map[file_path] = code
            return status_map
        except Exception:
            return {}

    @classmethod
    def _get_git_shortstat(cls, repo_dir: Path, timeout_s: float = 1.0) -> str:
        try:
            res = subprocess.run(
                ["git", "diff", "--shortstat"],
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            return res.stdout.strip() if res.returncode == 0 else ""
        except Exception:
            return ""

    @classmethod
    def _get_active_surface_info(cls) -> Tuple[str, str, str]:
        """Return (app_name, window_title, focused_element_label) via PyObjC or fallback."""
        if os.environ.get("VOICEFI_MOCK_TACTILE") == "1" or (
            os.environ.get("VOICEFI_MOCK_AUDIO") == "1" and os.environ.get("VOICEFI_HEADLESS") == "1"
        ):
            return ("Mock App", "Mock Window", "AXTextArea [Message input]")

        try:
            from AppKit import NSWorkspace
            import ApplicationServices as AS

            ws = NSWorkspace.sharedWorkspace()
            front_app = ws.frontmostApplication()
            if front_app is None:
                return ("", "", "")

            app_name = str(front_app.localizedName() or "").strip()
            pid = int(front_app.processIdentifier())
            ax_app = AS.AXUIElementCreateApplication(pid)
            if ax_app is None:
                return (app_name, "", "")

            err, win = AS.AXUIElementCopyAttributeValue(ax_app, "AXFocusedWindow", None)
            if err != 0 or win is None:
                err, win = AS.AXUIElementCopyAttributeValue(ax_app, "AXMainWindow", None)

            window_title = ""
            if win is not None:
                err, t = AS.AXUIElementCopyAttributeValue(win, "AXTitle", None)
                if err == 0 and t:
                    window_title = str(t).strip()

            focused_label = ""
            err, foc = AS.AXUIElementCopyAttributeValue(ax_app, "AXFocusedUIElement", None)
            if err == 0 and foc is not None:
                err, r = AS.AXUIElementCopyAttributeValue(foc, "AXRole", None)
                err, d = AS.AXUIElementCopyAttributeValue(foc, "AXDescription", None)
                err, title = AS.AXUIElementCopyAttributeValue(foc, "AXTitle", None)
                tag = str(title or d or "").strip()
                focused_label = f"{r} [{tag}]" if tag else str(r or "")

            return (app_name, window_title, focused_label)
        except Exception:
            return ("", "", "")

    @classmethod
    def capture_file_snapshot(
        cls,
        file_path: Path | str,
        raw_text: Optional[str] = None,
        raw_bytes: Optional[bytes] = None,
    ) -> FileTactileSnapshot:
        """
        Capture a <2ms single-file SHA-256, AST syntax validity, structural AST digest,
        and assertion baseline without spawning git subprocesses.
        """
        t0 = time.perf_counter()
        fpath = Path(file_path) if file_path else Path()
        fpath_str = str(file_path or "")

        if raw_bytes is None and raw_text is None:
            if not fpath_str or not fpath.is_file():
                return FileTactileSnapshot(
                    file_path=fpath_str,
                    exists=False,
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                )
            try:
                raw_bytes = fpath.read_bytes()
            except Exception:
                return FileTactileSnapshot(
                    file_path=fpath_str,
                    exists=False,
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                )

        if raw_bytes is None and raw_text is not None:
            raw_bytes = raw_text.encode("utf-8", errors="replace")
        if raw_text is None and raw_bytes is not None:
            raw_text = raw_bytes.decode("utf-8", errors="replace")

        b_data = raw_bytes or b""
        t_data = raw_text or ""
        sha256_hex = hashlib.sha256(b_data).hexdigest()[:16] if b_data else ""
        is_py = fpath.suffix.lower() == ".py"

        (
            ast_valid,
            ast_err,
            ast_dig,
            topo_dig,
            sym_cnt,
            assert_cnt,
            taut_cnt,
            test_fn_list,
        ) = compute_ast_structural_digest(t_data, is_python=is_py)

        return FileTactileSnapshot(
            file_path=fpath_str,
            exists=True,
            sha256=sha256_hex,
            size_bytes=len(b_data),
            ast_syntax_valid=ast_valid,
            ast_error=ast_err,
            ast_digest=ast_dig,
            topology_digest=topo_dig,
            symbol_count=sym_cnt,
            assert_count=assert_cnt,
            tautological_asserts=taut_cnt,
            test_funcs=test_fn_list,
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
        )

    @classmethod
    def verify_file_delta(
        cls,
        file_path_or_pre: Path | str | FileTactileSnapshot | Dict[str, Any],
        pre_snapshot: Optional[FileTactileSnapshot | Dict[str, Any] | Path | str] = None,
        *,
        seen_ast_digests: Optional[Iterable[str]] = None,
        post_text: Optional[str] = None,
        post_raw_text: Optional[str] = None,
        post_bytes: Optional[bytes] = None,
    ) -> TactileDelta:
        """
        Compute the closed-loop post-edit tactile delta for a single target file in <2ms (<50ms SLA):
          1. File SHA-256 mutation (`file_changed` vs `mechanical_noop`)
          2. AST syntax validity (`ast_syntax_valid`, `ast_error`)
          3. Structural AST mutation (`ast_mutated`, `topology_mutated`, `net_ast_progress`)
          4. Assertion & test function preservation (`assertions_preserved`, `assert_delta`)
        """
        t0 = time.perf_counter()
        if isinstance(file_path_or_pre, (FileTactileSnapshot, dict)):
            resolved_pre = file_path_or_pre
            file_path = pre_snapshot
            if file_path is None and isinstance(file_path_or_pre, FileTactileSnapshot):
                file_path = file_path_or_pre.file_path
            elif file_path is None and isinstance(file_path_or_pre, dict):
                file_path = file_path_or_pre.get("file_path") or file_path_or_pre.get("target")
        else:
            file_path = file_path_or_pre
            resolved_pre = (
                pre_snapshot
                if isinstance(pre_snapshot, (FileTactileSnapshot, dict))
                else None
            )

        effective_post_text = post_text if post_text is not None else post_raw_text
        fpath_str = str(file_path or "")

        if isinstance(resolved_pre, FileTactileSnapshot):
            pre_dict = resolved_pre.to_dict()
        elif isinstance(resolved_pre, dict):
            pre_dict = resolved_pre
        else:
            pre_dict = {}

        pre_sha = str(pre_dict.get("pre_sha256") or pre_dict.get("sha256") or "")
        pre_ast_dig = str(pre_dict.get("pre_ast_digest") or pre_dict.get("ast_digest") or "")
        pre_topo_dig = str(pre_dict.get("pre_topology_digest") or pre_dict.get("topology_digest") or "")
        pre_sym_cnt = int(pre_dict.get("pre_symbol_count", pre_dict.get("symbol_count", 0)) or 0)
        pre_assert_cnt = int(pre_dict.get("pre_assert_count", pre_dict.get("assert_count", 0)) or 0)
        pre_taut_cnt = int(pre_dict.get("pre_tautological_asserts", pre_dict.get("tautological_asserts", 0)) or 0)
        raw_pre_tests = pre_dict.get("pre_test_funcs", pre_dict.get("test_funcs")) or []
        pre_tests: Set[str] = {str(x) for x in raw_pre_tests if x}

        post_snap = cls.capture_file_snapshot(
            file_path=fpath_str,
            raw_text=effective_post_text,
            raw_bytes=post_bytes,
        )

        post_sha = post_snap.sha256
        file_changed = bool(post_sha != pre_sha) if (post_sha or pre_sha) else False
        mechanical_noop = not file_changed

        ast_valid = post_snap.ast_syntax_valid
        ast_err = post_snap.ast_error
        post_ast_dig = post_snap.ast_digest
        post_topo_dig = post_snap.topology_digest
        post_sym_cnt = post_snap.symbol_count
        post_assert_cnt = post_snap.assert_count
        post_taut_cnt = post_snap.tautological_asserts
        post_tests: Set[str] = set(post_snap.test_funcs)

        ast_mutated = bool(
            ast_valid
            and file_changed
            and (post_ast_dig != pre_ast_dig if (post_ast_dig or pre_ast_dig) else True)
        )
        topology_mutated = bool(
            ast_valid
            and file_changed
            and (post_topo_dig != pre_topo_dig if (post_topo_dig or pre_topo_dig) else False)
        )
        symbol_delta = post_sym_cnt - pre_sym_cnt
        assert_delta = post_assert_cnt - pre_assert_cnt
        taut_intro = max(0, post_taut_cnt - pre_taut_cnt)
        deleted_tests = sorted(pre_tests - post_tests)
        assertions_preserved = bool(
            ast_valid
            and post_assert_cnt >= pre_assert_cnt
            and taut_intro == 0
            and len(deleted_tests) == 0
        )

        seen_set: Set[str] = {str(d) for d in (seen_ast_digests or ()) if d}
        is_reverted_ast = bool(post_ast_dig and post_ast_dig in seen_set and post_ast_dig != pre_ast_dig)
        net_ast_progress = bool(
            file_changed
            and ast_valid
            and ast_mutated
            and assertions_preserved
            and not is_reverted_ast
        )

        summary_parts: List[str] = []
        if file_changed:
            summary_parts.append(f"SHA256 {pre_sha[:8] or 'new'}->{post_sha[:8]}")
        else:
            summary_parts.append("No-op (SHA256 unchanged)")
        if not ast_valid:
            summary_parts.append(ast_err or "AST SyntaxError")
        elif ast_mutated:
            summary_parts.append(f"AST mutated (assert_delta={assert_delta:+d})")
        else:
            summary_parts.append("AST unchanged")
        if not assertions_preserved:
            summary_parts.append("Assertions violated")

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        fname = Path(fpath_str).name if fpath_str else ""
        return TactileDelta(
            git_changed=file_changed,
            git_files_modified=[fname] if (file_changed and pre_sha) else [],
            git_files_added=[fname] if (file_changed and not pre_sha) else [],
            git_files_deleted=[],
            latency_ms=elapsed_ms,
            summary=" | ".join(summary_parts),
            target_file=fpath_str,
            pre_sha256=pre_sha,
            post_sha256=post_sha,
            file_changed=file_changed,
            mechanical_noop=mechanical_noop,
            ast_syntax_valid=ast_valid,
            ast_error=ast_err,
            pre_ast_digest=pre_ast_dig,
            post_ast_digest=post_ast_dig,
            ast_mutated=ast_mutated,
            pre_topology_digest=pre_topo_dig,
            post_topology_digest=post_topo_dig,
            topology_mutated=topology_mutated,
            pre_symbol_count=pre_sym_cnt,
            post_symbol_count=post_sym_cnt,
            symbol_delta=symbol_delta,
            pre_assert_count=pre_assert_cnt,
            post_assert_count=post_assert_cnt,
            assert_delta=assert_delta,
            tautological_asserts_introduced=taut_intro,
            deleted_test_funcs=deleted_tests,
            assertions_preserved=assertions_preserved,
            net_ast_progress=net_ast_progress,
        )

    @classmethod
    def capture_baseline(
        cls,
        working_dir: Optional[Path] = None,
        target_file: Optional[Path | str] = None,
    ) -> TactileBaseline:
        """Capture pre-action baseline of filesystem, optional target file, and window focus."""
        t0 = time.time()
        cwd = Path(working_dir or Path.cwd()).resolve()
        app_name, window_title, focused_label = cls._get_active_surface_info()
        git_map = cls._get_git_status_map(cwd)
        shortstat = cls._get_git_shortstat(cwd)
        f_snap = cls.capture_file_snapshot(target_file) if target_file else None

        return TactileBaseline(
            working_dir=cwd,
            timestamp=t0,
            app_name=app_name,
            window_title=window_title,
            focused_element_label=focused_label,
            git_status_map=git_map,
            git_shortstat=shortstat,
            target_file=str(target_file) if target_file else None,
            file_snapshot=f_snap,
        )

    @classmethod
    def capture_delta(
        cls,
        baseline: TactileBaseline,
        working_dir: Optional[Path] = None,
        timeout_s: float = 1.5,
    ) -> TactileDelta:
        """Compute the post-action state delta in <50ms."""
        t0 = time.perf_counter()
        cwd = Path(working_dir or baseline.working_dir).resolve()

        # 1. macOS Surface probe
        app_after, win_after, foc_after = cls._get_active_surface_info()
        window_changed = (
            (app_after != baseline.app_name)
            or (win_after != baseline.window_title)
            or (bool(baseline.focused_element_label and foc_after and baseline.focused_element_label != foc_after))
        )

        # 2. Git/Filesystem probe
        git_after = cls._get_git_status_map(cwd, timeout_s=timeout_s)
        git_shortstat = cls._get_git_shortstat(cwd, timeout_s=timeout_s)

        modified: List[str] = []
        added: List[str] = []
        deleted: List[str] = []

        all_keys = set(baseline.git_status_map.keys()) | set(git_after.keys())
        for fpath in sorted(all_keys):
            st_before = baseline.git_status_map.get(fpath)
            st_after = git_after.get(fpath)
            if st_before != st_after:
                if st_after is None:
                    continue
                if "D" in st_after:
                    deleted.append(fpath)
                elif "?" in st_after or "A" in st_after:
                    added.append(fpath)
                else:
                    modified.append(fpath)

        git_stat_delta = git_shortstat != baseline.git_shortstat
        git_changed = bool(modified or added or deleted or git_stat_delta)

        # 3. Optional target-file AST & SHA-256 verification
        file_delta: Optional[TactileDelta] = None
        if baseline.target_file and baseline.file_snapshot is not None:
            file_delta = cls.verify_file_delta(baseline.target_file, baseline.file_snapshot)
            if file_delta.file_changed:
                git_changed = True

        summary_parts = []
        if git_changed:
            summary_parts.append(
                f"{len(modified)} modified, {len(added)} added, {len(deleted)} deleted"
            )
        if window_changed:
            summary_parts.append(f"focus shifted to {app_after}")
        if not summary_parts:
            summary_parts.append("No observable state change (deterministic no-op)")

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        delta = TactileDelta(
            git_changed=git_changed,
            git_files_modified=modified,
            git_files_added=added,
            git_files_deleted=deleted,
            git_shortstat=git_shortstat,
            window_changed=window_changed,
            app_before=baseline.app_name,
            app_after=app_after,
            window_title_before=baseline.window_title,
            window_title_after=win_after,
            focused_before=baseline.focused_element_label,
            focused_after=foc_after,
            latency_ms=elapsed_ms,
            summary=" | ".join(summary_parts),
        )
        if file_delta is not None:
            delta.target_file = file_delta.target_file
            delta.pre_sha256 = file_delta.pre_sha256
            delta.post_sha256 = file_delta.post_sha256
            delta.file_changed = file_delta.file_changed
            delta.mechanical_noop = file_delta.mechanical_noop
            delta.ast_syntax_valid = file_delta.ast_syntax_valid
            delta.ast_error = file_delta.ast_error
            delta.pre_ast_digest = file_delta.pre_ast_digest
            delta.post_ast_digest = file_delta.post_ast_digest
            delta.ast_mutated = file_delta.ast_mutated
            delta.pre_topology_digest = file_delta.pre_topology_digest
            delta.post_topology_digest = file_delta.post_topology_digest
            delta.topology_mutated = file_delta.topology_mutated
            delta.pre_symbol_count = file_delta.pre_symbol_count
            delta.post_symbol_count = file_delta.post_symbol_count
            delta.symbol_delta = file_delta.symbol_delta
            delta.pre_assert_count = file_delta.pre_assert_count
            delta.post_assert_count = file_delta.post_assert_count
            delta.assert_delta = file_delta.assert_delta
            delta.tautological_asserts_introduced = file_delta.tautological_asserts_introduced
            delta.deleted_test_funcs = file_delta.deleted_test_funcs
            delta.assertions_preserved = file_delta.assertions_preserved
            delta.net_ast_progress = file_delta.net_ast_progress
        return delta


def capture_tactile_baseline(
    working_dir: Optional[Path] = None,
    target_file: Optional[Path | str] = None,
) -> TactileBaseline:
    """Helper entrypoint to record a tactile pre-action baseline."""
    return TactileSensor.capture_baseline(working_dir=working_dir, target_file=target_file)


def capture_post_action_tactile_delta(
    baseline: TactileBaseline,
    working_dir: Optional[Path] = None,
) -> TactileDelta:
    """Helper entrypoint to compute tactile feedback delta in <50ms."""
    return baseline.compute_delta(working_dir=working_dir)


def capture_file_tactile_snapshot(
    file_path: Path | str,
    raw_text: Optional[str] = None,
    raw_bytes: Optional[bytes] = None,
) -> FileTactileSnapshot:
    """Helper entrypoint to capture a <2ms single-file SHA-256 and AST baseline."""
    return TactileSensor.capture_file_snapshot(
        file_path=file_path,
        raw_text=raw_text,
        raw_bytes=raw_bytes,
    )


def verify_post_edit_tactile_delta(
    file_path_or_pre: Path | str | FileTactileSnapshot | Dict[str, Any],
    pre_snapshot: Optional[FileTactileSnapshot | Dict[str, Any] | Path | str] = None,
    *,
    seen_ast_digests: Optional[Iterable[str]] = None,
    post_text: Optional[str] = None,
    post_raw_text: Optional[str] = None,
    post_bytes: Optional[bytes] = None,
) -> TactileDelta:
    """Helper entrypoint to verify post-edit SHA-256, AST syntax, and assertions in <2ms (<50ms SLA)."""
    return TactileSensor.verify_file_delta(
        file_path_or_pre,
        pre_snapshot,
        seen_ast_digests=seen_ast_digests,
        post_text=post_text,
        post_raw_text=post_raw_text,
        post_bytes=post_bytes,
    )
