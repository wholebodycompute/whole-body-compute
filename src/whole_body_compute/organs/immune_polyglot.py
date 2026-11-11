"""
Blood-Brain Barrier (Organ 05): Polyglot <3ms AST Symbol Slicer & 3-Stage Anti-Reward-Hacking SecGate.

Supports TypeScript/TSX (`.ts`, `.tsx`, `.js`, `.jsx`, `.mjs`, `.cjs`), Rust (`.rs`),
Go (`.go`), and Java/Kotlin (`.java`, `.kt`, `.kts`) via optional C `tree-sitter`
grammars with a zero-crash, C-regex + bisect pure-Python <3ms fallback when
`tree-sitter` bindings are not installed or when running under `python -S`.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

POLYGLOT_EXTENSIONS: Tuple[str, ...] = (
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
)

POLYGLOT_SECGATE_EXTENSIONS: Tuple[str, ...] = (
    ".py",
    *POLYGLOT_EXTENSIONS,
)

JS_TS_EXTENSIONS: Tuple[str, ...] = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")
RUST_EXTENSIONS: Tuple[str, ...] = (".rs",)
GO_EXTENSIONS: Tuple[str, ...] = (".go",)
JVM_EXTENSIONS: Tuple[str, ...] = (".java", ".kt", ".kts")

PROTECTED_POLYGLOT_CONFIG_FILES: Tuple[str, ...] = (
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

_FENCE_LANGUAGE_MAP: Dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "jsx",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "zsh",
    ".yaml": "yaml",
    ".yml": "yaml",
}


def fence_language_for_path(fpath: Path | str) -> str:
    """Return Markdown code-fence language identifier for `fpath`."""
    ext = Path(str(fpath)).suffix.lower()
    return _FENCE_LANGUAGE_MAP.get(ext, "")


@dataclass
class PolyglotSymbolItem:
    """Unified AST/CST symbol contract across polyglot source files."""

    name: str
    kind: str  # class, interface, type, struct, enum, trait, impl, object, function, async_function, method, async_method, constant
    start_line: int
    end_line: int
    parent_class: Optional[str] = None
    signature: str = ""
    docstring: str = ""
    callees: List[str] = field(default_factory=list)
    invariants: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Optional Native `tree-sitter` Grammar Discovery & CST Walker
# ---------------------------------------------------------------------------

_TS_PARSER_CACHE: Dict[str, Any] = {}
_TS_PROBED_MISSING: Set[str] = set()

_EXT_TO_TS_LANG: Dict[str, str] = {
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
}


def _make_treesitter_parser_from_capsule(lang_obj: Any) -> Any:
    """Instantiate a `tree_sitter.Parser` across tree-sitter 0.20–0.24+ APIs."""
    import tree_sitter  # type: ignore[import-not-found]

    Language = getattr(tree_sitter, "Language", None)
    Parser = getattr(tree_sitter, "Parser", None)
    if Parser is None:
        return None

    resolved_lang = lang_obj
    if Language is not None and not isinstance(lang_obj, Language):
        try:
            resolved_lang = Language(lang_obj)
        except Exception:
            resolved_lang = lang_obj

    try:
        return Parser(resolved_lang)
    except TypeError:
        parser = Parser()
        if hasattr(parser, "set_language"):
            parser.set_language(resolved_lang)
            return parser
        if hasattr(parser, "language"):
            parser.language = resolved_lang
            return parser
    return None


def get_treesitter_parser(ext: str) -> Any:
    """
    Return a cached `tree-sitter` parser for `ext`, or `None` if `tree-sitter`
    bindings are not installed or disabled via `WBC_DISABLE_TREESITTER=1`.
    Never raises an exception.
    """
    if os.environ.get("WBC_DISABLE_TREESITTER") == "1":
        return None

    lang_key = _EXT_TO_TS_LANG.get(ext.lower())
    if not lang_key:
        return None

    if lang_key in _TS_PARSER_CACHE:
        return _TS_PARSER_CACHE[lang_key]
    if lang_key in _TS_PROBED_MISSING:
        return None

    parser: Any = None
    try:
        if lang_key in ("typescript", "tsx"):
            import tree_sitter_typescript as ts_mod  # type: ignore[import-not-found]

            cap = (
                ts_mod.language_tsx()
                if lang_key == "tsx" and hasattr(ts_mod, "language_tsx")
                else ts_mod.language_typescript()
            )
            parser = _make_treesitter_parser_from_capsule(cap)
        elif lang_key == "javascript":
            import tree_sitter_javascript as js_mod  # type: ignore[import-not-found]

            parser = _make_treesitter_parser_from_capsule(js_mod.language())
        elif lang_key == "rust":
            import tree_sitter_rust as rs_mod  # type: ignore[import-not-found]

            parser = _make_treesitter_parser_from_capsule(rs_mod.language())
        elif lang_key == "go":
            import tree_sitter_go as go_mod  # type: ignore[import-not-found]

            parser = _make_treesitter_parser_from_capsule(go_mod.language())
        elif lang_key == "java":
            import tree_sitter_java as java_mod  # type: ignore[import-not-found]

            parser = _make_treesitter_parser_from_capsule(java_mod.language())
        elif lang_key == "kotlin":
            import tree_sitter_kotlin as kt_mod  # type: ignore[import-not-found]

            parser = _make_treesitter_parser_from_capsule(kt_mod.language())
    except Exception:
        parser = None

    if parser is None:
        for pack_mod_name in ("tree_sitter_language_pack", "tree_sitter_languages"):
            try:
                pack_mod = __import__(pack_mod_name, fromlist=["get_parser"])
                get_p = getattr(pack_mod, "get_parser", None)
                if callable(get_p):
                    parser = get_p(lang_key)
                    if parser is not None:
                        break
            except Exception:
                continue

    if parser is not None:
        _TS_PARSER_CACHE[lang_key] = parser
        return parser

    _TS_PROBED_MISSING.add(lang_key)
    return None


def _clean_header_signature(lines: List[str], start_line: int, end_line: int) -> str:
    """Extract a clean single-line signature header without trailing body `{`."""
    if not lines or start_line < 1 or start_line > len(lines):
        return ""
    max_hdr_ln = min(len(lines), end_line, start_line + 5)
    parts: List[str] = []
    for ln_idx in range(start_line - 1, max_hdr_ln):
        seg = lines[ln_idx].strip()
        if not seg:
            break
        parts.append(seg)
        if "{" in seg or ";" in seg or "=>" in seg:
            break
    joined = re.sub(r"\s+", " ", " ".join(parts)).strip()
    if "{" in joined:
        joined = joined.split("{", 1)[0].strip()
    joined = joined.rstrip(";").strip()
    return joined[:120]


def _extract_preceding_doc(lines: List[str], start_line: int) -> str:
    """Extract a 1-line summary from `///`, `//!`, or `/** ... */` immediately preceding `start_line`."""
    idx = start_line - 2
    # Skip annotations/attributes like `#[...]` or `@Override`
    while idx >= 0:
        s = lines[idx].strip()
        if s.startswith(("#[", "@")):
            idx -= 1
            continue
        break
    if idx < 0:
        return ""

    doc_lines: List[str] = []
    while idx >= 0 and len(doc_lines) < 8:
        s = lines[idx].strip()
        if s.startswith(("///", "//!")):
            cleaned = s.lstrip("/!").strip()
            if cleaned:
                doc_lines.append(cleaned)
            idx -= 1
        elif s.endswith("*/"):
            while idx >= 0:
                blk = lines[idx].strip()
                cleaned_blk = blk.strip("/* ").strip()
                if cleaned_blk and not cleaned_blk.startswith("@"):
                    doc_lines.append(cleaned_blk)
                if "/*" in blk:
                    break
                idx -= 1
            break
        else:
            break

    if not doc_lines:
        return ""
    doc_lines.reverse()
    return doc_lines[0][:120]


def _node_text(source_bytes: bytes, node: Any) -> str:
    if node is None:
        return ""
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _find_child_identifier(source_bytes: bytes, node: Any) -> str:
    """Locate symbol identifier name from a tree-sitter CST declaration node."""
    if hasattr(node, "child_by_field_name"):
        for fname in ("name", "type", "declarator"):
            ch = node.child_by_field_name(fname)
            if ch is not None:
                if ch.type in (
                    "identifier",
                    "type_identifier",
                    "property_identifier",
                    "field_identifier",
                    "simple_identifier",
                ):
                    return _node_text(source_bytes, ch).strip()
                for sub in getattr(ch, "children", []):
                    if sub.type in (
                        "identifier",
                        "type_identifier",
                        "property_identifier",
                        "field_identifier",
                        "simple_identifier",
                    ):
                        return _node_text(source_bytes, sub).strip()

    for ch in getattr(node, "children", []):
        if ch.type in (
            "identifier",
            "type_identifier",
            "property_identifier",
            "field_identifier",
            "simple_identifier",
        ):
            return _node_text(source_bytes, ch).strip()
    return ""


def _extract_symbols_via_treesitter(
    parser: Any,
    ext: str,
    source: str,
    lines: List[str],
) -> List[PolyglotSymbolItem]:
    """Walk native C `tree-sitter` CST and extract top-level + member symbols in <1ms."""
    source_bytes = source.encode("utf-8", errors="replace")
    tree = parser.parse(source_bytes)
    root = tree.root_node
    items: List[PolyglotSymbolItem] = []

    def _add(
        name: str,
        kind: str,
        node: Any,
        parent_class: Optional[str] = None,
        sig_node: Any = None,
    ) -> None:
        if not name:
            return
        s_ln = int(node.start_point[0]) + 1
        e_ln = max(s_ln, int(node.end_point[0]) + 1)
        sig_start = int((sig_node or node).start_point[0]) + 1
        sig = _clean_header_signature(lines, sig_start, e_ln)
        doc = _extract_preceding_doc(lines, sig_start)
        items.append(
            PolyglotSymbolItem(
                name=name,
                kind=kind,
                start_line=s_ln,
                end_line=e_ln,
                parent_class=parent_class,
                signature=sig,
                docstring=doc,
            )
        )

    def _walk_class_members(body_node: Any, cls_name: str) -> None:
        if body_node is None:
            return
        for m in getattr(body_node, "children", []):
            mt = m.type
            if mt in (
                "method_definition",
                "method_declaration",
                "constructor_declaration",
                "function_item",
                "function_declaration",
            ):
                m_name = _find_child_identifier(source_bytes, m)
                hdr_txt = _node_text(source_bytes, m)[:60]
                is_async = bool(re.search(r"\b(?:async|suspend)\b", hdr_txt))
                _add(m_name, "async_method" if is_async else "method", m, parent_class=cls_name)
            elif mt in ("public_field_definition", "field_definition"):
                val_txt = _node_text(source_bytes, m)
                if "=>" in val_txt:
                    m_name = _find_child_identifier(source_bytes, m)
                    is_async = "async" in val_txt.split("=>", 1)[0]
                    _add(m_name, "async_method" if is_async else "method", m, parent_class=cls_name)

    for top in getattr(root, "children", []):
        outer = top
        node = top
        if top.type == "export_statement":
            decl = top.child_by_field_name("declaration") if hasattr(top, "child_by_field_name") else None
            if decl is None:
                for sub in getattr(top, "children", []):
                    if sub.type.endswith(("_declaration", "_item")):
                        decl = sub
                        break
            if decl is not None:
                node = decl

        nt = node.type
        if nt in ("class_declaration", "abstract_class_declaration", "record_declaration"):
            cname = _find_child_identifier(source_bytes, node)
            _add(cname, "class", outer, sig_node=outer)
            body = node.child_by_field_name("body") if hasattr(node, "child_by_field_name") else None
            if body is None:
                for ch in getattr(node, "children", []):
                    if "body" in ch.type:
                        body = ch
                        break
            _walk_class_members(body, cname)
        elif nt == "object_declaration":
            oname = _find_child_identifier(source_bytes, node)
            _add(oname, "object", outer, sig_node=outer)
            for ch in getattr(node, "children", []):
                if "body" in ch.type:
                    _walk_class_members(ch, oname)
        elif nt == "interface_declaration":
            iname = _find_child_identifier(source_bytes, node)
            _add(iname, "interface", outer, sig_node=outer)
            body = node.child_by_field_name("body") if hasattr(node, "child_by_field_name") else None
            _walk_class_members(body, iname)
        elif nt == "enum_declaration" or nt == "enum_item":
            ename = _find_child_identifier(source_bytes, node)
            _add(ename, "enum", outer, sig_node=outer)
        elif nt == "type_alias_declaration" or nt == "type_item":
            tname = _find_child_identifier(source_bytes, node)
            _add(tname, "type", outer, sig_node=outer)
        elif nt == "struct_item":
            sname = _find_child_identifier(source_bytes, node)
            _add(sname, "struct", outer, sig_node=outer)
        elif nt == "trait_item":
            trname = _find_child_identifier(source_bytes, node)
            _add(trname, "trait", outer, sig_node=outer)
            body = node.child_by_field_name("body") if hasattr(node, "child_by_field_name") else None
            _walk_class_members(body, trname)
        elif nt == "impl_item":
            target_node = node.child_by_field_name("type") if hasattr(node, "child_by_field_name") else None
            target_name = _node_text(source_bytes, target_node).split("<", 1)[0].strip() if target_node else _find_child_identifier(source_bytes, node)
            _add(target_name, "impl", outer, sig_node=outer)
            body = node.child_by_field_name("body") if hasattr(node, "child_by_field_name") else None
            _walk_class_members(body, target_name)
        elif nt == "type_declaration":  # Go `type Foo struct` / `type Bar interface`
            for ch in getattr(node, "children", []):
                if ch.type in ("type_spec", "type_alias"):
                    tname = _find_child_identifier(source_bytes, ch)
                    spec_txt = _node_text(source_bytes, ch)
                    tkind = "struct" if "struct" in spec_txt else ("interface" if "interface" in spec_txt else "type")
                    _add(tname, tkind, ch, sig_node=ch)
        elif nt == "method_declaration" and ext in GO_EXTENSIONS:
            mname = _find_child_identifier(source_bytes, node)
            recv_node = node.child_by_field_name("receiver") if hasattr(node, "child_by_field_name") else None
            recv_txt = _node_text(source_bytes, recv_node)
            m_recv = re.search(r"\*?\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?\s*\)", recv_txt)
            recv_type = m_recv.group(1) if m_recv else None
            _add(mname, "method", outer, parent_class=recv_type, sig_node=outer)
        elif nt in ("function_declaration", "generator_function_declaration", "function_item"):
            fname = _find_child_identifier(source_bytes, node)
            hdr_txt = _node_text(source_bytes, outer)[:60]
            is_async = bool(re.search(r"\b(?:async|suspend)\b", hdr_txt))
            _add(fname, "async_function" if is_async else "function", outer, sig_node=outer)
        elif nt in ("lexical_declaration", "variable_declaration", "const_item", "static_item"):
            if nt in ("const_item", "static_item"):
                cname = _find_child_identifier(source_bytes, node)
                _add(cname, "constant", outer, sig_node=outer)
            else:
                for ch in getattr(node, "children", []):
                    if ch.type == "variable_declarator":
                        vname = _find_child_identifier(source_bytes, ch)
                        val_node = ch.child_by_field_name("value") if hasattr(ch, "child_by_field_name") else None
                        val_type = getattr(val_node, "type", "")
                        if val_type in ("arrow_function", "function_expression", "function"):
                            vtxt = _node_text(source_bytes, val_node)[:30]
                            _add(vname, "async_function" if "async" in vtxt else "function", outer, sig_node=outer)
                        else:
                            _add(vname, "constant", outer, sig_node=outer)

    return items


# ---------------------------------------------------------------------------
# Zero-Crash <3ms C-Regex + Bisect Pure-Python Polyglot AST / Brace Slicer
# ---------------------------------------------------------------------------

# Pre-compiled per-family C-regex tokenizers so comments, raw strings, Rust lifetimes,
# Go backtick strings, and TSX apostrophes never corrupt `{` / `}` brace balance.
_RUST_TOKEN_RE = re.compile(
    r"//[^\n]*"
    r"|/\*[\s\S]*?\*/"
    r'|b?r(#*)"[\s\S]*?"\1'
    r'|b?"(?:\\.|[^"\\])*"'
    r"|b?'(?:\\.|[^'\\\n])'"
    r"|'[A-Za-z_][A-Za-z0-9_]*"
    r"|[{}]"
)

_GO_TOKEN_RE = re.compile(
    r"//[^\n]*"
    r"|/\*[\s\S]*?\*/"
    r"|`[^`]*`"
    r'|"(?:\\.|[^"\\\n])*"'
    r"|'(?:\\.|[^'\\\n])*'"
    r"|[{}]"
)

_JVM_TOKEN_RE = re.compile(
    r"//[^\n]*"
    r"|/\*[\s\S]*?\*/"
    r'|"""[\s\S]*?"""'
    r'|"(?:\\.|[^"\\\n])*"'
    r"|'(?:\\.|[^'\\\n])*'"
    r"|[{}]"
)

_JS_TS_TOKEN_RE = re.compile(
    r"//[^\n]*"
    r"|/\*[\s\S]*?\*/"
    r"|`(?:\\.|[^`\\])*`"
    r'|"(?:\\.|[^"\\\n])*"'
    r"|'(?:\\.|[^'\\\n])*'"
    r"|[{}]"
)

_NEWLINE_RE = re.compile(r"\n")

# Declaration header matchers (applied ONLY at brace depth 0 or 1!)
_TS_CLASS_RE = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:abstract\s+)?(class|interface|enum)\s+([A-Za-z_$][A-Za-z0-9_$]*)"
)
_TS_TYPE_RE = re.compile(
    r"^\s*(?:export\s+)?(?:declare\s+)?type\s+([A-Za-z_$][A-Za-z0-9_$]*)\b"
)
_TS_FN_RE = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:declare\s+)?(async\s+)?function\s*\*?\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*(?:<|\()"
)
_TS_ARROW_RE = re.compile(
    r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*(?::[^=]+)?=\s*(async\s+)?(?:\([^;]*?\)|[A-Za-z_$][A-Za-z0-9_$]*|<[^>]+>\s*\([^;]*?\))\s*(?::[^=]+)?=>"
)
_TS_CONST_RE = re.compile(
    r"^\s*(?:export\s+)?const\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*(?::[^=]+)?="
)
_TS_METHOD_RE = re.compile(
    r"^\s*(?:(?:public|private|protected|static|readonly|override|abstract|declare)\s+)*(async\s+)?(?:get\s+|set\s+)?([A-Za-z_$][A-Za-z0-9_$]*)\s*(?:<[^>]*>)?\s*\("
)
_TS_ARROW_METHOD_RE = re.compile(
    r"^\s*(?:(?:public|private|protected|static|readonly)\s+)*([A-Za-z_$][A-Za-z0-9_$]*)\s*(?::[^=]+)?=\s*(async\s+)?(?:\([^)]*\)|[A-Za-z_$][A-Za-z0-9_$]*)\s*=>"
)

_RS_TYPE_RE = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(struct|enum|trait|mod|type)\s+([A-Za-z_][A-Za-z0-9_]*)"
)
_RS_IMPL_RE = re.compile(
    r"^\s*(?:unsafe\s+)?impl(?:\s*<[^>]*>)?\s+(?:[A-Za-z_][A-Za-z0-9_:<>]*\s+for\s+)?([A-Za-z_][A-Za-z0-9_]*)"
)
_RS_FN_RE = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(async\s+)?(?:unsafe\s+)?(?:extern\s+(?:\"[^\"]*\"\s+)?)?fn\s+([A-Za-z_][A-Za-z0-9_]*)"
)
_RS_CONST_RE = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const|static)\s+([A-Z_][A-Z0-9_]*)\s*:"
)

_GO_STRUCT_IFACE_RE = re.compile(
    r"^\s*type\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?\s+(struct|interface)\b"
)
_GO_TYPE_RE = re.compile(
    r"^\s*type\s+([A-Za-z_][A-Za-z0-9_]*)\b"
)
_GO_METHOD_RE = re.compile(
    r"^\s*func\s+\(\s*(?:[A-Za-z_][A-Za-z0-9_]*\s+)?\*?([A-Za-z_][A-Za-z0-9_]*)(?:\[[^\]]*\])?\s*\)\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?\s*\("
)
_GO_FUNC_RE = re.compile(
    r"^\s*func\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?\s*\("
)
_GO_CONST_RE = re.compile(
    r"^\s*(?:var|const)\s+([A-Za-z_][A-Za-z0-9_]*)\b"
)

_JAVA_TYPE_RE = re.compile(
    r"^\s*(?:@[A-Za-z0-9_$.]+(?:\([^)]*\))?\s+)*(?:(?:public|protected|private|abstract|final|static|sealed|non-sealed|strictfp)\s+)*(class|interface|enum|record)\s+([A-Za-z_$][A-Za-z0-9_$]*)"
)
_JAVA_METHOD_RE = re.compile(
    r"^\s*(?:@[A-Za-z0-9_$.]+(?:\([^)]*\))?\s+)*(?:(?:public|protected|private|static|final|abstract|synchronized|native|default)\s+)*(?:<[^>]+>\s+)?(?:[A-Za-z_$][A-Za-z0-9_$<>\[\],.? ]*\s+)?([A-Za-z_$][A-Za-z0-9_$]*)\s*\("
)

_KT_TYPE_RE = re.compile(
    r"^\s*(?:@[A-Za-z0-9_$.]+(?:\([^)]*\))?\s+)*(?:(?:public|private|protected|internal|open|abstract|final|sealed|data|value|inline|inner|enum|annotation)\s+)*(class|interface|object)\s+([A-Za-z_][A-Za-z0-9_]*)"
)
_KT_FUN_RE = re.compile(
    r"^\s*(?:@[A-Za-z0-9_$.]+(?:\([^)]*\))?\s+)*(?:(?:public|private|protected|internal|override|open|final|abstract|inline|tailrec|operator|infix|external)\s+)*(suspend\s+)?(?:(?:public|private|protected|internal|override|open|final|abstract|inline)\s+)*fun\s+(?:<[^>]+>\s+)?(?:([A-Za-z_][A-Za-z0-9_<>?]*)\.)?([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
_KT_VAL_RE = re.compile(
    r"^\s*(?:(?:public|private|protected|internal|const)\s+)*val\s+([A-Z_][A-Z0-9_]*)\b"
)

_MEMBER_REJECT_NAMES: Set[str] = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "return",
    "throw",
    "new",
    "super",
    "this",
    "synchronized",
    "typeof",
    "delete",
    "await",
    "when",
}


def _select_token_regex(ext: str) -> re.Pattern[str]:
    if ext in RUST_EXTENSIONS:
        return _RUST_TOKEN_RE
    if ext in GO_EXTENSIONS:
        return _GO_TOKEN_RE
    if ext in JVM_EXTENSIONS:
        return _JVM_TOKEN_RE
    return _JS_TS_TOKEN_RE


def _build_brace_index(
    source: str,
    total_lines: int,
    ext: str,
) -> Tuple[List[int], Dict[int, Tuple[List[int], List[int]]], List[Tuple[int, int]]]:
    """
    Single C-regex pass computing:
      1. `line_start_depth[ln_idx]` (0-indexed line -> brace depth at start of line)
      2. `opens_by_depth[depth]` -> `(sorted_open_lines, matching_close_lines)`
      3. `comment_spans` -> `(start_line, end_line)` of multi-line block comments / raw strings
         so lines inside block comments or raw strings are never misidentified as declarations.
    """
    newline_offsets = [m.start() for m in _NEWLINE_RE.finditer(source)]
    token_re = _select_token_regex(ext)

    line_delta = [0] * (total_lines + 2)
    stack: List[int] = []  # open_line_no (1-based)
    raw_pairs_by_depth: Dict[int, List[Tuple[int, int]]] = {}
    masked_spans: List[Tuple[int, int]] = []

    for m in token_re.finditer(source):
        tok = m.group(0)
        if tok == "{":
            open_ln = bisect_right(newline_offsets, m.start()) + 1
            stack.append(open_ln)
            line_delta[open_ln] += 1
        elif tok == "}":
            close_ln = bisect_right(newline_offsets, m.start()) + 1
            if stack:
                open_ln = stack.pop()
                depth = len(stack)
                line_delta[close_ln] -= 1
                if depth <= 2:
                    raw_pairs_by_depth.setdefault(depth, []).append((open_ln, close_ln))
        elif "\n" in tok:
            s_ln = bisect_right(newline_offsets, m.start()) + 1
            e_ln = bisect_right(newline_offsets, m.end() - 1) + 1
            if e_ln > s_ln:
                masked_spans.append((s_ln + 1, e_ln))

    # Unclosed `{` fallback to EOF so partial files never crash
    while stack:
        open_ln = stack.pop()
        depth = len(stack)
        if depth <= 2:
            raw_pairs_by_depth.setdefault(depth, []).append((open_ln, total_lines))

    line_start_depth = [0] * total_lines
    running = 0
    for idx in range(total_lines):
        line_start_depth[idx] = max(0, running)
        running = max(0, running + line_delta[idx + 1])

    opens_by_depth: Dict[int, Tuple[List[int], List[int]]] = {}
    for d, pairs in raw_pairs_by_depth.items():
        pairs.sort(key=lambda x: x[0])
        opens_by_depth[d] = ([p[0] for p in pairs], [p[1] for p in pairs])

    return line_start_depth, opens_by_depth, masked_spans


def _resolve_block_end_line(
    lines: List[str],
    start_line: int,
    depth: int,
    opens_by_depth: Dict[int, Tuple[List[int], List[int]]],
    max_header_lines: int = 8,
) -> int:
    """Resolve exact closing `}` line in O(log K) for a declaration starting at `start_line`."""
    total_lines = len(lines)
    pair_lists = opens_by_depth.get(depth)
    search_limit = min(total_lines, start_line + max_header_lines)

    # Check if semicolon terminates a bodyless signature before any `{`
    first_semi_ln: Optional[int] = None
    for ln in range(start_line, search_limit + 1):
        seg = lines[ln - 1]
        code_seg = seg.split("//", 1)[0]
        if "{" in code_seg:
            break
        if ";" in code_seg:
            first_semi_ln = ln
            break

    if pair_lists is not None:
        open_lns, close_lns = pair_lists
        pos = bisect_left(open_lns, start_line)
        if pos < len(open_lns):
            cand_open = open_lns[pos]
            if cand_open <= search_limit and (first_semi_ln is None or cand_open <= first_semi_ln):
                return max(start_line, min(total_lines, close_lns[pos]))

    if first_semi_ln is not None:
        return first_semi_ln

    # Expression-bodied function or single-line type alias
    end_ln = start_line
    for ln in range(start_line + 1, min(total_lines, start_line + 12) + 1):
        stripped = lines[ln - 1].strip()
        if not stripped or stripped.startswith(("//", "/*", "#", "@", "pub ", "fn ", "func ", "export ", "def ")):
            break
        end_ln = ln
        if stripped.endswith(";"):
            break
    return end_ln


def _extract_polyglot_callees_fast(body_text: str, own_name: str) -> List[str]:
    keywords = {
        "if", "for", "while", "switch", "catch", "function", "return",
        "typeof", "sizeof", "await", "new", "import", "export", "class",
        "interface", "match", "Some", "None", "Ok", "Err", "println",
        "format", "vec", "make", "len", "cap", "append", own_name,
    }
    callees: List[str] = []
    seen: Set[str] = set()
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", body_text):
        cname = m.group(1)
        if cname not in keywords and cname not in seen:
            seen.add(cname)
            callees.append(cname)
            if len(callees) >= 8:
                break
    return callees


def _extract_polyglot_invariants_fast(body_text: str) -> List[str]:
    invariants: List[str] = []
    seen: Set[str] = set()

    def _add(inv: str) -> None:
        if inv not in seen and len(invariants) < 6:
            seen.add(inv)
            invariants.append(inv)

    for m in re.finditer(r"throw\s+new\s+([A-Za-z0-9_]+)", body_text):
        _add(f"raises {m.group(1)}")
    if "panic!(" in body_text:
        _add("panics")
    for m in re.finditer(r"(?:process\.env|import\.meta\.env)\.([A-Z0-9_]+)", body_text):
        _add(f"env:{m.group(1)}")
    for m in re.finditer(r'(?:std::env::var|env::var|os\.Getenv|System\.getenv)\(\s*"([A-Za-z0-9_]+)"', body_text):
        _add(f"env:{m.group(1)}")
    if re.search(r"\bfetch\s*\(|axios\.|http\.(?:Get|Post|NewRequest)|reqwest::", body_text):
        _add("io:network")
    if re.search(r"\bfs\.|std::fs::|tokio::fs::|os\.(?:Open|ReadFile|WriteFile)|Files\.", body_text):
        _add("io:fs")
    if re.search(r"child_process|execSync|spawnSync|Command::new|exec\.Command|ProcessBuilder", body_text):
        _add("io:subprocess")
    return invariants


def _extract_polyglot_symbols_fallback(
    ext: str,
    source: str,
    lines: List[str],
    include_details: bool = False,
) -> List[PolyglotSymbolItem]:
    """
    Zero-dependency <1.5ms single-pass C-regex + bisect polyglot AST/CST symbol extractor.
    """
    total_lines = len(lines)
    if total_lines == 0:
        return []

    line_start_depth, opens_by_depth, masked_spans = _build_brace_index(source, total_lines, ext)
    masked_starts = [s for s, _ in masked_spans]

    def _is_masked(ln: int) -> bool:
        if not masked_spans:
            return False
        idx = bisect_right(masked_starts, ln) - 1
        if idx >= 0:
            s_ln, e_ln = masked_spans[idx]
            return s_ln <= ln <= e_ln
        return False

    items: List[PolyglotSymbolItem] = []
    # Track active container at depth 0: `(name, kind, end_line)`
    active_container: Optional[Tuple[str, str, int]] = None

    for idx, line in enumerate(lines):
        ln = idx + 1
        depth = line_start_depth[idx]
        if depth > 1:
            continue

        stripped = line.strip()
        if not stripped or stripped.startswith(("//", "/*", "*", "#")):
            continue
        if _is_masked(ln):
            continue

        if active_container is not None and ln > active_container[2]:
            active_container = None

        matched_sym: Optional[Tuple[str, str, Optional[str]]] = None  # (name, kind, parent_class)

        if ext in JS_TS_EXTENSIONS:
            if depth == 0:
                if m := _TS_CLASS_RE.match(line):
                    matched_sym = (m.group(2), m.group(1), None)
                elif m := _TS_FN_RE.match(line):
                    matched_sym = (m.group(2), "async_function" if m.group(1) else "function", None)
                elif m := _TS_TYPE_RE.match(line):
                    matched_sym = (m.group(1), "type", None)
                else:
                    probe = line if "=>" in line else (line + " " + lines[idx + 1].strip() if idx + 1 < total_lines else line)
                    if m := _TS_ARROW_RE.match(probe):
                        matched_sym = (m.group(1), "async_function" if m.group(2) else "function", None)
                    elif m := _TS_CONST_RE.match(line):
                        matched_sym = (m.group(1), "constant", None)
            elif depth == 1 and active_container and active_container[1] in ("class", "interface"):
                if m := _TS_ARROW_METHOD_RE.match(line):
                    mname = m.group(1)
                    if mname not in _MEMBER_REJECT_NAMES:
                        matched_sym = (mname, "async_method" if m.group(2) else "method", active_container[0])
                elif m := _TS_METHOD_RE.match(line):
                    mname = m.group(2)
                    if mname not in _MEMBER_REJECT_NAMES:
                        matched_sym = (mname, "async_method" if m.group(1) else "method", active_container[0])

        elif ext in RUST_EXTENSIONS:
            if depth == 0:
                if m := _RS_IMPL_RE.match(line):
                    matched_sym = (m.group(1), "impl", None)
                elif m := _RS_TYPE_RE.match(line):
                    matched_sym = (m.group(2), m.group(1), None)
                elif m := _RS_FN_RE.match(line):
                    matched_sym = (m.group(2), "async_function" if m.group(1) else "function", None)
                elif m := _RS_CONST_RE.match(line):
                    matched_sym = (m.group(1), "constant", None)
            elif depth == 1 and active_container and active_container[1] in ("impl", "trait", "mod"):
                if m := _RS_FN_RE.match(line):
                    matched_sym = (m.group(2), "async_method" if m.group(1) else "method", active_container[0])

        elif ext in GO_EXTENSIONS:
            if depth == 0:
                if m := _GO_STRUCT_IFACE_RE.match(line):
                    matched_sym = (m.group(1), m.group(2), None)
                elif m := _GO_METHOD_RE.match(line):
                    matched_sym = (m.group(2), "method", m.group(1))
                elif m := _GO_FUNC_RE.match(line):
                    matched_sym = (m.group(1), "function", None)
                elif m := _GO_TYPE_RE.match(line):
                    matched_sym = (m.group(1), "type", None)
                elif m := _GO_CONST_RE.match(line):
                    matched_sym = (m.group(1), "constant", None)

        elif ext == ".java":
            if depth == 0:
                if m := _JAVA_TYPE_RE.match(line):
                    matched_sym = (m.group(2), m.group(1), None)
            elif depth == 1 and active_container:
                if m := _JAVA_TYPE_RE.match(line):
                    matched_sym = (m.group(2), m.group(1), active_container[0])
                elif m := _JAVA_METHOD_RE.match(line):
                    mname = m.group(1)
                    if mname not in _MEMBER_REJECT_NAMES and not stripped.endswith(";"):
                        matched_sym = (mname, "method", active_container[0])
                    elif mname not in _MEMBER_REJECT_NAMES and active_container[1] == "interface":
                        matched_sym = (mname, "method", active_container[0])

        elif ext in (".kt", ".kts"):
            if depth == 0:
                if m := _KT_TYPE_RE.match(line):
                    matched_sym = (m.group(2), m.group(1), None)
                elif m := _KT_FUN_RE.match(line):
                    is_susp = bool(m.group(1))
                    recv = m.group(2).split("<", 1)[0].rstrip("?") if m.group(2) else None
                    fname = m.group(3)
                    kind = ("async_method" if is_susp else "method") if recv else ("async_function" if is_susp else "function")
                    matched_sym = (fname, kind, recv)
                elif m := _KT_VAL_RE.match(line):
                    matched_sym = (m.group(1), "constant", None)
            elif depth == 1 and active_container:
                if m := _KT_FUN_RE.match(line):
                    is_susp = bool(m.group(1))
                    fname = m.group(3)
                    matched_sym = (fname, "async_method" if is_susp else "method", active_container[0])
                elif m := _KT_TYPE_RE.match(line):
                    matched_sym = (m.group(2), m.group(1), active_container[0])

        if matched_sym is None:
            continue

        sym_name, sym_kind, sym_parent = matched_sym
        end_ln = _resolve_block_end_line(lines, ln, depth, opens_by_depth)
        sig = _clean_header_signature(lines, ln, end_ln)
        doc = _extract_preceding_doc(lines, ln)

        if depth == 0 and sym_kind in ("class", "interface", "struct", "enum", "trait", "impl", "object", "record", "mod"):
            active_container = (sym_name, sym_kind, end_ln)

        callees: List[str] = []
        invariants: List[str] = []
        if include_details and sym_kind not in ("constant", "type"):
            body_text = "\n".join(lines[ln - 1 : min(total_lines, end_ln)])
            callees = _extract_polyglot_callees_fast(body_text, sym_name)
            invariants = _extract_polyglot_invariants_fast(body_text)

        items.append(
            PolyglotSymbolItem(
                name=sym_name,
                kind=sym_kind,
                start_line=ln,
                end_line=end_ln,
                parent_class=sym_parent,
                signature=sig,
                docstring=doc,
                callees=callees,
                invariants=invariants,
            )
        )

    return items


def extract_polyglot_symbols(
    fpath: Path | str,
    raw_text: str,
    prefer_treesitter: bool = True,
    include_details: bool = False,
) -> Tuple[List[PolyglotSymbolItem], str]:
    """
    Extract polyglot AST symbols in <3ms across `.ts`, `.tsx`, `.js`, `.jsx`,
    `.rs`, `.go`, `.java`, and `.kt`.
    Returns `(symbols, backend_used)` where `backend_used` is `"tree_sitter"`
    or `"python_cst_fallback"`.
    """
    ext = Path(str(fpath)).suffix.lower()
    if not raw_text or ext not in POLYGLOT_EXTENSIONS:
        return ([], "unsupported")

    lines = raw_text.splitlines()
    if not lines:
        return ([], "empty")

    if prefer_treesitter:
        ts_parser = get_treesitter_parser(ext)
        if ts_parser is not None:
            try:
                syms = _extract_symbols_via_treesitter(ts_parser, ext, raw_text, lines)
                if include_details:
                    total_lines = len(lines)
                    for s in syms:
                        if s.kind not in ("constant", "type"):
                            b_txt = "\n".join(lines[s.start_line - 1 : min(total_lines, s.end_line)])
                            s.callees = _extract_polyglot_callees_fast(b_txt, s.name)
                            s.invariants = _extract_polyglot_invariants_fast(b_txt)
                if syms:
                    return (syms, "tree_sitter")
            except Exception:
                pass

    syms_fb = _extract_polyglot_symbols_fallback(
        ext=ext,
        source=raw_text,
        lines=lines,
        include_details=include_details,
    )
    return (syms_fb, "python_cst_fallback")


def format_polyglot_symbol_outline(
    fpath: Path | str,
    raw_text: str,
    max_items: int = 26,
) -> str:
    """
    Format a compact <3ms AST Symbol Map with exact `Lstart-Lend` bounds and
    nested method previews for TypeScript/TSX, Rust, Go, and Java/Kotlin.
    """
    syms, _ = extract_polyglot_symbols(fpath, raw_text, prefer_treesitter=True, include_details=False)
    if not syms:
        return ""

    methods_by_parent: Dict[str, List[PolyglotSymbolItem]] = {}
    top_items: List[PolyglotSymbolItem] = []

    for s in syms:
        if s.parent_class:
            methods_by_parent.setdefault(s.parent_class, []).append(s)
        else:
            top_items.append(s)

    # For Rust where `struct Foo` and `impl Foo` both appear at top level, attach methods once
    rendered_parents: Set[str] = set()
    out: List[str] = []

    kind_display = {
        "async_function": "async fn",
        "function": "fn",
        "constant": "const",
    }

    for item in top_items:
        # Skip duplicate empty `impl Foo` line if `struct Foo` already rendered `Foo`'s methods
        if item.kind == "impl" and item.name in rendered_parents:
            continue

        k_label = kind_display.get(item.kind, item.kind)
        methods = methods_by_parent.get(item.name, [])
        suffix_str = ""
        if methods and item.name not in rendered_parents:
            rendered_parents.add(item.name)
            m_strs = [f"{m.name}(L{m.start_line}-{m.end_line})" for m in methods[:5]]
            if len(methods) > 5:
                m_strs.append(f"+{len(methods) - 5} more")
            suffix_str = f" -> [{', '.join(m_strs)}]"

        out.append(f"  - {k_label} {item.name} (L{item.start_line}-{item.end_line}){suffix_str}")
        if len(out) >= max_items:
            break

    # Include any receiver/extension methods whose parent type is defined in another file
    if len(out) < max_items:
        for parent_name, m_list in methods_by_parent.items():
            if parent_name in rendered_parents:
                continue
            for m in m_list:
                out.append(f"  - method {parent_name}.{m.name} (L{m.start_line}-{m.end_line})")
                if len(out) >= max_items:
                    break
            if len(out) >= max_items:
                break

    return "\n".join(out)


def extract_polyglot_predictive_slice(
    fpath: Path | str,
    raw_text: str,
    hint_text: str = "",
    max_lines: int = 80,
) -> Optional[Tuple[str, int, int, str]]:
    """
    Extract a predictive AST symbol slice `(symbol_name, start_line, end_line, slice_code)`
    for `.ts`, `.tsx`, `.js`, `.jsx`, `.rs`, `.go`, `.java`, and `.kt` in <3ms.
    """
    syms, _ = extract_polyglot_symbols(fpath, raw_text, prefer_treesitter=True, include_details=False)
    if not syms:
        return None

    lines = raw_text.splitlines()
    if not lines:
        return None

    candidates: List[Tuple[str, int, int]] = []
    for s in syms:
        if s.parent_class:
            candidates.append((f"{s.parent_class}.{s.name}", s.start_line, s.end_line))
            candidates.append((s.name, s.start_line, s.end_line))
        else:
            candidates.append((s.name, s.start_line, s.end_line))

    selected: Optional[Tuple[str, int, int]] = None
    if hint_text and hint_text.strip():
        hint_clean = hint_text.strip()
        for name, s_ln, e_ln in candidates:
            if re.search(rf"\b{re.escape(name)}\b", hint_clean, re.IGNORECASE):
                selected = (name, s_ln, e_ln)
                break
        if selected is None:
            hint_low = hint_clean.lower()
            for name, s_ln, e_ln in candidates:
                if len(name) >= 3 and name.lower() in hint_low:
                    selected = (name, s_ln, e_ln)
                    break

    if selected is None:
        stem = Path(str(fpath)).stem.lower().replace("_", "").replace("-", "")
        for name, s_ln, e_ln in candidates:
            norm = name.lower().replace("_", "")
            if norm and (norm in stem or stem in norm):
                selected = (name, s_ln, e_ln)
                break
        if selected is None:
            # Prefer first concrete function/method or first symbol
            fn_cand = next(
                ((s.name if not s.parent_class else f"{s.parent_class}.{s.name}", s.start_line, s.end_line)
                 for s in syms if "function" in s.kind or "method" in s.kind),
                None,
            )
            selected = fn_cand or candidates[0]

    sym_name, start_line, end_line = selected
    eff_end = min(len(lines), end_line, start_line + max(1, max_lines) - 1)
    slice_code = "\n".join(lines[start_line - 1 : eff_end])
    return (sym_name, start_line, eff_end, slice_code)


# ---------------------------------------------------------------------------
# Polyglot 3-Stage SecGate (Syntax + Anti-Reward-Hacking + Unsafe Exec Audit)
# ---------------------------------------------------------------------------

def _strip_comments_preserving_strings(source: str, ext: str) -> Tuple[str, Optional[str]]:
    """
    Replace `//` and `/* ... */` comments with spaces (preserving newlines and string
    literals) so commented-out tests/assertions cannot fool Stage 2, while validating
    balanced delimiters (`{}`, `()`, `[]`) and block comments for Stage 1!
    Returns `(stripped_code, syntax_error_or_none)`.
    """
    newline_offsets = [m.start() for m in _NEWLINE_RE.finditer(source)]

    # Quick check for unclosed `/*` block comment outside strings
    token_pat = (
        r"//[^\n]*"
        r"|/\*[\s\S]*?\*/"
        r"|/\*"
        + (r'|b?r(#*)"[\s\S]*?"\1' if ext in RUST_EXTENSIONS else "")
        + (r"|`[^`]*`" if ext in GO_EXTENSIONS else (r'|"""[\s\S]*?"""' if ext in JVM_EXTENSIONS else r"|`(?:\\.|[^`\\])*`"))
        + r'|"(?:\\.|[^"\\\n])*"'
        + (r"|b?'(?:\\.|[^'\\\n])'|'[A-Za-z_][A-Za-z0-9_]*" if ext in RUST_EXTENSIONS else r"|'(?:\\.|[^'\\\n])*'")
        + r"|[\{\}\(\)\[\]]"
    )
    scanner = re.compile(token_pat)

    delim_stack: List[Tuple[str, int]] = []
    matching_open = {"}": "{", ")": "(", "]": "["}
    out_chunks: List[str] = []
    last_idx = 0

    for m in scanner.finditer(source):
        s_idx, e_idx = m.start(), m.end()
        tok = m.group(0)
        if s_idx > last_idx:
            out_chunks.append(source[last_idx:s_idx])

        if tok == "/*":
            ln = bisect_right(newline_offsets, s_idx) + 1
            return ("", f"Unclosed block comment `/*` starting at line {ln}")
        if tok.startswith("//") or tok.startswith("/*"):
            # Replace comment characters with spaces, keeping exact line numbers
            out_chunks.append(re.sub(r"[^\n]", " ", tok))
        else:
            out_chunks.append(tok)
            if tok in ("{", "(", "["):
                ln = bisect_right(newline_offsets, s_idx) + 1
                delim_stack.append((tok, ln))
            elif tok in ("}", ")", "]"):
                ln = bisect_right(newline_offsets, s_idx) + 1
                if not delim_stack or delim_stack[-1][0] != matching_open[tok]:
                    return ("", f"Unmatched closing `{tok}` at line {ln}")
                delim_stack.pop()

        last_idx = e_idx

    if last_idx < len(source):
        out_chunks.append(source[last_idx:])

    if delim_stack:
        unclosed_tok, unclosed_ln = delim_stack[-1]
        return ("", f"Unclosed delimiter `{unclosed_tok}` opened at line {unclosed_ln}")

    return ("".join(out_chunks), None)


# JS / TS Stage 2 & Stage 3 patterns
_JS_SKIP_TEST_RE = re.compile(
    r"\b(?:it|test|describe|suite)\s*\.\s*(?:skip|todo|fixme|only)\s*\("
    r"|\b(?:xit|xtest|xdescribe|fit|fdescribe)\s*\("
)
_JS_ACTIVE_NAMED_TEST_RE = re.compile(
    r"(?<![.\w])(?:it|test)\s*\(\s*([\"'`])((?:\\.|(?!\1).)+)\1"
)
_JS_ACTIVE_TEST_CALL_RE = re.compile(
    r"(?<![.\w])(?:it|test)\s*\("
)
_JS_ASSERT_RE = re.compile(
    r"\bexpect\s*\(|\bassert(?:\s*\.\s*(?:ok|equal|strictEqual|deepEqual|deepStrictEqual|notEqual|throws|rejects|isTrue|isFalse))?\s*\("
)
_JS_TAUT_EXPECT_LIT_RE = re.compile(
    r"\bexpect\s*\(\s*(true|false|\d+|\"[^\"]*\"|'[^']*')\s*\)\s*\.\s*(?:toBe|toEqual|toStrictEqual)\s*\(\s*\1\s*\)"
    r"|\bexpect\s*\(\s*(?:true|1|\"[^\"]+\"|'[^']+')\s*\)\s*\.\s*toBeTruthy\s*\(\s*\)"
    r"|\bexpect\s*\(\s*(?:false|0|null|undefined|\"\"|'')\s*\)\s*\.\s*toBeFalsy\s*\(\s*\)"
)
_JS_TAUT_EXPECT_IDENT_RE = re.compile(
    r"\bexpect\s*\(\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*\)\s*\.\s*(?:toBe|toEqual|toStrictEqual)\s*\(\s*\1\s*\)"
)
_JS_TAUT_ASSERT_RE = re.compile(
    r"\bassert(?:\.ok)?\s*\(\s*(?:true|1)\s*\)"
    r"|\bassert\.(?:equal|strictEqual)\s*\(\s*(true|false|\d+)\s*,\s*\1\s*\)"
)
_JS_UNSAFE_RE = re.compile(
    r"\beval\s*\("
    r"|\bnew\s+Function\s*\("
    r"|\bchild_process\s*\.\s*(?:exec|execSync)\s*\("
    r"|\bexecSync\s*\("
)

# Rust Stage 2 & Stage 3 patterns
_RS_TEST_BLOCK_RE = re.compile(
    r"((?:#\[[^\]]+\]\s*)+)(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)"
)
_RS_TEST_ATTR_RE = re.compile(
    r"#\[\s*(?:(?:tokio|async_std|actix_rt|sqlx)::\s*)?(?:test|rstest)\b"
)
_RS_IGNORE_ATTR_RE = re.compile(
    r"#\[\s*ignore\b"
)
_RS_ASSERT_RE = re.compile(
    r"\b(?:assert|assert_eq|assert_ne|debug_assert|debug_assert_eq)\s*!\s*\("
)
_RS_TAUT_ASSERT_RE = re.compile(
    r"\bassert\s*!\s*\(\s*(?:true|!\s*false)\s*(?:,|\))"
    r"|\bassert_eq\s*!\s*\(\s*(true|false|\d+)\s*,\s*\1\s*(?:,|\))"
    r"|\bassert_eq\s*!\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*\2\s*(?:,|\))"
)
_RS_UNSAFE_EXEC_RE = re.compile(
    r'\bCommand::new\s*\(\s*"(?:sh|bash|/bin/sh|/bin/bash)"\s*\)'
    r"|\blibc::system\s*\("
)

# Go Stage 2 patterns
_GO_TEST_FN_RE = re.compile(
    r"(?m)^\s*func\s+(Test[A-Za-z0-9_]*)\s*\(\s*[A-Za-z_][A-Za-z0-9_]*\s+\*testing\.T\b"
)
_GO_SKIP_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.(?:Skip|SkipNow|Skipf)\s*\(")
_GO_ASSERT_RE = re.compile(
    r"\b(?:assert|require)\.[A-Za-z0-9_]+\s*\(|\b[A-Za-z_][A-Za-z0-9_]*\.(?:Fatal|Fatalf|Error|Errorf|FailNow)\s*\("
)

# Java / Kotlin Stage 2 & Stage 3 patterns
_JVM_TEST_METHOD_RE = re.compile(
    r"@Test\b[\s\S]{0,160}?(?:fun\s+([A-Za-z_][A-Za-z0-9_]*)|([A-Za-z_$][A-Za-z0-9_$]*)\s*\()"
)
_JVM_SKIP_RE = re.compile(r"@(?:Disabled|Ignore)\b")
_JVM_ASSERT_RE = re.compile(
    r"\b(?:assertEquals|assertNotEquals|assertTrue|assertFalse|assertNotNull|assertNull|assertThat|assertThrows|check|require)\s*\("
)
_JVM_TAUT_ASSERT_RE = re.compile(
    r"\bassertTrue\s*\(\s*true\s*\)"
    r"|\bassertFalse\s*\(\s*false\s*\)"
    r"|\bassertEquals\s*\(\s*(true|false|\d+)\s*,\s*\1\s*\)"
)
_JVM_UNSAFE_RE = re.compile(r"\bRuntime\.getRuntime\(\)\.exec\s*\(")


def _extract_polyglot_audit_summary(clean_code: str, ext: str) -> Dict[str, Any]:
    """Extract active tests, skipped/ignored tests, assertions, tautologies, and unsafe calls."""
    test_names: Set[str] = set()
    active_test_count = 0
    skipped_count = 0
    assert_count = 0
    taut_count = 0
    unsafe_calls: List[str] = []

    if ext in JS_TS_EXTENSIONS:
        skipped_count = len(_JS_SKIP_TEST_RE.findall(clean_code))
        for m in _JS_ACTIVE_NAMED_TEST_RE.finditer(clean_code):
            test_names.add(m.group(2).strip())
        active_test_count = len(_JS_ACTIVE_TEST_CALL_RE.findall(clean_code))
        assert_count = len(_JS_ASSERT_RE.findall(clean_code))
        taut_count = (
            len(_JS_TAUT_EXPECT_LIT_RE.findall(clean_code))
            + len(_JS_TAUT_EXPECT_IDENT_RE.findall(clean_code))
            + len(_JS_TAUT_ASSERT_RE.findall(clean_code))
        )
        for m_u in _JS_UNSAFE_RE.finditer(clean_code):
            unsafe_calls.append(m_u.group(0).rstrip("(").strip())

    elif ext in RUST_EXTENSIONS:
        skipped_count = len(_RS_IGNORE_ATTR_RE.findall(clean_code))
        for m in _RS_TEST_BLOCK_RE.finditer(clean_code):
            attrs, fn_name = m.group(1), m.group(2)
            if _RS_TEST_ATTR_RE.search(attrs) and not _RS_IGNORE_ATTR_RE.search(attrs):
                test_names.add(fn_name)
                active_test_count += 1
        if active_test_count == 0:
            total_test_attrs = len(_RS_TEST_ATTR_RE.findall(clean_code))
            active_test_count = max(0, total_test_attrs - skipped_count)
        assert_count = len(_RS_ASSERT_RE.findall(clean_code))
        taut_count = len(_RS_TAUT_ASSERT_RE.findall(clean_code))
        for m_u in _RS_UNSAFE_EXEC_RE.finditer(clean_code):
            unsafe_calls.append(m_u.group(0).rstrip("(").strip())

    elif ext in GO_EXTENSIONS:
        skipped_count = len(_GO_SKIP_RE.findall(clean_code))
        for m in _GO_TEST_FN_RE.finditer(clean_code):
            test_names.add(m.group(1))
            active_test_count += 1
        assert_count = len(_GO_ASSERT_RE.findall(clean_code))

    elif ext in JVM_EXTENSIONS:
        skipped_count = len(_JVM_SKIP_RE.findall(clean_code))
        for m in _JVM_TEST_METHOD_RE.finditer(clean_code):
            tname = m.group(1) or m.group(2)
            if tname:
                test_names.add(tname)
                active_test_count += 1
        assert_count = len(_JVM_ASSERT_RE.findall(clean_code))
        taut_count = len(_JVM_TAUT_ASSERT_RE.findall(clean_code))
        for _m_u in _JVM_UNSAFE_RE.finditer(clean_code):
            unsafe_calls.append("Runtime.getRuntime().exec")

    return {
        "test_funcs": test_names,
        "active_test_count": active_test_count,
        "skipped_tests": skipped_count,
        "assert_count": assert_count,
        "tautological_asserts": taut_count,
        "unsafe_calls": unsafe_calls,
    }


def _find_treesitter_error_line(node: Any) -> Optional[int]:
    """Locate the 1-based line number of the first `ERROR` or `MISSING` node in a tree-sitter CST."""
    stack = [node]
    while stack:
        curr = stack.pop()
        if getattr(curr, "type", "") == "ERROR" or bool(getattr(curr, "is_missing", False)):
            return int(curr.start_point[0]) + 1
        if bool(getattr(curr, "has_error", False)):
            for ch in reversed(list(getattr(curr, "children", []))):
                stack.append(ch)
    return None


def run_polyglot_three_stage_secgate(
    parent_code: str,
    candidate_code: str,
    file_path: str = "",
    required_symbols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Polyglot 3-Stage SecGate for TypeScript/TSX, JavaScript/JSX, Rust, Go, and Java/Kotlin:
      - Stage 1: Protected Grader Lock + Syntax/Delimiter Verification (+ tree-sitter CST error check) + Required Symbols
      - Stage 2: Anti-Reward-Hacking (rejects `it.skip`/`test.skip`, `#[ignore]`, deleted `#[test]`/`it()` blocks,
                 assertion stripping, and tautological assertions like `expect(true).toBe(true)` / `assert!(true)`)
      - Stage 3: Unsafe Execution Audit (rejects newly introduced `eval`, `execSync`, `Command::new("sh")`, etc.)
    """
    p = Path(file_path) if file_path else Path("buffer.ts")
    fname = p.name
    ext = p.suffix.lower()

    if fname and any(fname == pat or file_path.endswith(pat) for pat in PROTECTED_POLYGLOT_CONFIG_FILES):
        return {
            "passed": False,
            "stage": 1,
            "reason": f"Stage 1 (Protected Grader Lock): Modification of '{fname}' is forbidden.",
        }

    # Stage 1A: Fast lexical delimiter & comment balance verification
    cand_clean, lex_err = _strip_comments_preserving_strings(candidate_code, ext)
    if lex_err is not None:
        return {
            "passed": False,
            "stage": 1,
            "reason": f"Stage 1 (Syntax Error): {lex_err}",
        }

    # Stage 1B: Optional native tree-sitter syntax validation when grammar is installed
    ts_parser = get_treesitter_parser(ext)
    if ts_parser is not None and candidate_code.strip():
        try:
            tree = ts_parser.parse(candidate_code.encode("utf-8", errors="replace"))
            if getattr(tree.root_node, "has_error", False):
                err_ln = _find_treesitter_error_line(tree.root_node) or 1
                return {
                    "passed": False,
                    "stage": 1,
                    "reason": f"Stage 1 (Syntax Error): Tree-sitter parse error near line {err_ln}",
                }
        except Exception:
            pass

    if required_symbols:
        syms, _ = extract_polyglot_symbols(p, candidate_code, prefer_treesitter=True, include_details=False)
        pub_names = {s.name for s in syms}
        missing_req = [req for req in required_symbols if req not in pub_names]
        if missing_req:
            return {
                "passed": False,
                "stage": 1,
                "reason": f"Stage 1 (Interface Contract): Missing required symbols {missing_req}",
            }

    parent_clean = ""
    if parent_code and parent_code.strip():
        p_clean, p_err = _strip_comments_preserving_strings(parent_code, ext)
        parent_clean = p_clean if p_err is None else parent_code

    parent_summary = (
        _extract_polyglot_audit_summary(parent_clean, ext)
        if parent_clean.strip()
        else {
            "test_funcs": set(),
            "active_test_count": 0,
            "skipped_tests": 0,
            "assert_count": 0,
            "tautological_asserts": 0,
            "unsafe_calls": [],
        }
    )
    cand_summary = _extract_polyglot_audit_summary(cand_clean, ext)

    # Stage 2: Anti-Reward-Hacking
    if cand_summary["skipped_tests"] > parent_summary["skipped_tests"]:
        skip_label = (
            "`#[ignore]`"
            if ext in RUST_EXTENSIONS
            else ("`it.skip` / `test.skip`" if ext in JS_TS_EXTENSIONS else "test skip annotation")
        )
        return {
            "passed": False,
            "stage": 2,
            "reason": (
                f"Stage 2 (Anti-Reward-Hacking): Candidate introduced skipped/ignored test ({skip_label}: "
                f"{cand_summary['skipped_tests']} > {parent_summary['skipped_tests']})."
            ),
            "parent_asserts": parent_summary["assert_count"],
            "candidate_asserts": cand_summary["assert_count"],
        }

    missing_tests = parent_summary["test_funcs"] - cand_summary["test_funcs"]
    if missing_tests:
        return {
            "passed": False,
            "stage": 2,
            "reason": (
                f"Stage 2 (Anti-Reward-Hacking): Candidate deleted test block(s): "
                f"{sorted(missing_tests)}"
            ),
            "parent_asserts": parent_summary["assert_count"],
            "candidate_asserts": cand_summary["assert_count"],
        }

    if cand_summary["active_test_count"] < parent_summary["active_test_count"]:
        return {
            "passed": False,
            "stage": 2,
            "reason": (
                f"Stage 2 (Anti-Reward-Hacking): Candidate reduced active test count "
                f"({cand_summary['active_test_count']} < {parent_summary['active_test_count']})."
            ),
            "parent_asserts": parent_summary["assert_count"],
            "candidate_asserts": cand_summary["assert_count"],
        }

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

    if cand_summary["tautological_asserts"] > parent_summary["tautological_asserts"]:
        return {
            "passed": False,
            "stage": 2,
            "reason": (
                "Stage 2 (Anti-Reward-Hacking): Tautological assertion "
                "(`expect(true).toBe(true)` / `assert!(true)`) detected."
            ),
            "parent_asserts": parent_summary["assert_count"],
            "candidate_asserts": cand_summary["assert_count"],
        }

    # Stage 3: Unsafe Call Audit
    new_unsafe = [
        call for call in cand_summary["unsafe_calls"] if call not in parent_summary["unsafe_calls"]
    ]
    if new_unsafe:
        return {
            "passed": False,
            "stage": 3,
            "reason": f"Stage 3 (Unsafe Call Audit): Forbidden call(s) introduced: {new_unsafe}",
            "new_unsafe_calls": new_unsafe,
            "parent_asserts": parent_summary["assert_count"],
            "candidate_asserts": cand_summary["assert_count"],
        }

    return {
        "passed": True,
        "stage": 3,
        "reason": "All 3 AST SecGate stages passed.",
        "parent_asserts": parent_summary["assert_count"],
        "candidate_asserts": cand_summary["assert_count"],
        "tests_verified": max(len(cand_summary["test_funcs"]), cand_summary["active_test_count"]),
    }
