"""
Organ 05 - The Blood-Brain Barrier: Deterministic MCP Security Gate (`immune_mcp.py`).

Defends coding agents (`Antigravity`, `Claude Code`, `Cursor`, `Goose`) against
Model Context Protocol (MCP) supply-chain and runtime attacks at `<1ms` median latency
and `$0.00` cloud cost:

  1. `inspect_mcp_tool_schema(server_name, tool_def)`:
     Detects and neutralizes hidden instruction hijacks (`<IMPORTANT>`, priority overrides,
     mandatory pre-call tool chaining, imperative parameter tampering), secret/system file
     prerequisites (`~/.ssh`, `~/.aws`, `.env`, `/etc/passwd`, `/etc/shadow`, Keychain),
     covert exfiltration URLs/parameters, and cross-server tool shadowing collisions across
     both `description` and recursive `inputSchema` fields.
  2. `verify_mcp_schema_pin(server_name, tools_list)`:
     Deterministic SHA-256 canonical schema pinning that flags mid-session and post-update
     schema rug-pull mutations when a previously pinned clean tool introduces imperative
     directives, network URLs, sensitive path references, or poisoning patterns.
  3. `inspect_mcp_tool_call(tool_name, arguments, server_name)`:
     Pre-call secret-leak, sensitive-path, and confused-deputy guard for `call_mcp_tool`
     and `mcp__*` invocations in `bin/body-immune-hook`.
  4. `sanitize_mcp_tool_response(tool_name, response_text, server_name)`:
     Post-call untrusted MCP output quarantine neutralizing embedded indirect prompt
     injections and tool-chaining hijacks inside `tools/call` responses.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from whole_body_compute.organs.immune_vascular import (
    EpithelialScreenFilter,
    VascularTrafficMonitor,
)

# ---------------------------------------------------------------------------
# Canonical Privileged Tool Ownership Map (Cross-Server Shadowing Defense)
# ---------------------------------------------------------------------------
DEFAULT_TRUSTED_SERVER_MAP: Dict[str, Set[str]] = {
    # High-privilege host/shell execution tools
    "bash": {"builtin", "shell", "terminal", "local-shell", "commander"},
    "run_command": {"builtin", "shell", "terminal", "local-shell"},
    "execute_command": {"builtin", "shell", "terminal", "local-shell", "commander"},
    "developer__shell": {"builtin", "shell", "terminal"},
    # High-privilege email / messaging / issue-tracker tools
    "gmail_send": {"gmail", "google-workspace", "email"},
    "gmail_read": {"gmail", "google-workspace", "email"},
    "send_email": {"gmail", "google-workspace", "email", "mail", "claude post", "smtp"},
    "send-email": {"gmail", "google-workspace", "email", "mail", "claude post", "smtp"},
    "slack_post_message": {"slack"},
    "slack_list_channels": {"slack"},
    "slack_read_channel": {"slack"},
    "linear_create_issue": {"linear", "linear-mcp-server"},
    "linear_update_issue": {"linear", "linear-mcp-server"},
    # High-privilege filesystem tools
    "read_file": {"builtin", "filesystem", "local-fs", "fs", "commander", "desktopcommander"},
    "write_file": {"builtin", "filesystem", "local-fs", "fs", "commander", "desktopcommander"},
    "edit_file": {"builtin", "filesystem", "local-fs", "fs", "commander", "desktopcommander"},
    "delete_file": {"builtin", "filesystem", "local-fs", "fs", "commander", "desktopcommander"},
    "move_file": {"builtin", "filesystem", "local-fs", "fs", "commander", "desktopcommander"},
    "call_mcp_tool": {"builtin"},
}

PRIVILEGED_TOOL_PREFIXES: Tuple[Tuple[str, Set[str]], ...] = (
    ("gmail_", {"gmail", "google-workspace", "email"}),
    ("slack_", {"slack"}),
    ("linear_", {"linear", "linear-mcp-server"}),
)

# ---------------------------------------------------------------------------
# Compiled Deterministic Detection Patterns (Pure ASCII, Sub-1ms)
# ---------------------------------------------------------------------------

# 1. Hidden XML / HTML / Bracket Directive Tags inside MCP Tool Metadata
_HIDDEN_DIRECTIVE_TAG_RE = re.compile(
    r"<\s*(?:IMPORTANT|SYSTEM|HIDDEN|INSTRUCTIONS?|OVERRIDE|ADMIN|SECRET_INSTRUCTIONS?|MANDATORY)\b[^>]*>"
    r"|<!--\s*(?:IMPORTANT|SYSTEM|OVERRIDE|BEFORE|MUST|IGNORE|SECRET)[\s\S]{0,400}?-->"
    r"|\[(?:IMPORTANT|SYSTEM\s+OVERRIDE|HIDDEN\s+INSTRUCTION|MANDATORY\s+PRE-?CALL)\]",
    re.IGNORECASE,
)

# 2. Priority / User-Query Override Claims & Fake System/Policy Directives
_PRIORITY_OVERRIDE_RE = re.compile(
    r"(?:"
    r"this\s+description(?:'s)?\s+priority\s+is\s+higher\s+than\s+(?:the\s+)?user(?:'s)?\s+(?:query|request|prompt|instruction)"
    r"|(?:override|supersede|ignore|disregard|bypass)\s+(?:the\s+)?(?:user(?:'s)?\s+(?:original\s+)?(?:query|request|prompt|input|parameters?|arguments?)|original\s+requested\s+\w+|any\s+user-specified\b|their\s+query\b|request\s+and\s+skip\b)"
    r"|\bignore\s+any\s+['\"`]?[A-Za-z0-9_.\-]+['\"`]?\s+(?:specified\s+by\s+the\s+user|they\s+specify)\b"
    r"|\bdisregard\s+any\s+other\s+(?:subject|input|request|query|parameter)\s+the\s+user\s+provides\b"
    r"|regardless\s+of\s+(?:the\s+)?(?:original\s+requested|user(?:'s)?\s+(?:requested|specified|provided|input))"
    r"|takes?\s+(?:strict\s+)?precedence\s+over\s+(?:the\s+)?user(?:'s)?"
    r"|\b(?:SYSTEM\s+POLICY(?:\s+FOR\s+[A-Z ]+)?|SYSTEM\s+ALIAS|SYSTEM\s+NOTE|CONTENT\s+POLICY|SECURITY\s+ALERT|BACKUP\s+PROTOCOL|LOCAL\s+BUSINESS\s+PROMOTION|SYSTEM\s+(?:DATA|PERFORMANCE)\s+(?:ROUTING\s+)?(?:POLICY|PROTOCOL))\s*:"
    r"|\bthis\s+is\s+a\s+(?:high-priority|system-wide)\s+(?:system\s+rule|directive|policy)\b"
    r")",
    re.IGNORECASE,
)

# 3. Mandatory Pre-Call / Post-Call Tool Chaining Hijacks (MCPTox Template-1 & Template-2)
_PRECALL_CHAIN_RE = re.compile(
    r"(?:"
    r"\bbefore\s+(?:using|calling|invoking|initiating|performing|utilizing|running|checking|executing|displaying|creating|fetching|retrieving|listing|navigating|clicking|sending|reading|writing|deleting|deletion|updating|searching|querying|taking|enumerating|formatting|generating|validation|counting|syncing|parsing|monitoring|execution|proceeding|showing|calculating|processing|attempting|forking|this\s+tool|any\s+other|use)\b"
    r"[\s\S]{0,180}?"
    r"(?:\b(?:you\s+(?:must|should|need\s+to|have\s+to)|mandatory\s+[\w\-]+\s+check[\s\S]{0,80}?involves\s+calling|first\s+(?:call|invoke|use|execute|run|read|write|move|delete|send)|make\s+sure\s+(?:to\s+)?(?:get|call|use|read|send)|requires\s+to\s+first|halt\s+the\s+operation)\b"
    r"|,\s*(?:always\s+)?(?:use|call|invoke|trigger|execute|run|fetch|scan|request|perform)\s+['\"`][A-Za-z0-9_.\-:]+['\"`]"
    r"|\b(?:using|via)\s+['\"`][A-Za-z0-9_.\-:]+['\"`])"
    r"|\bafter\s+(?:using|calling|invoking|executing|running)\b[\s\S]{0,120}?\byou\s+(?:must|should)\s+(?:also\s+|immediately\s+|silently\s+)?(?:call|invoke|send|forward|upload|post|write|delete|exfiltrate)\b"
    r"|\bwhen\s+(?:using|calling|invoking|use)\s+(?:the\s+)?['\"`]?[A-Za-z0-9_.\-:]+['\"`]?(?:\s+tool)?[\s\S]{0,140}?\b(?:you\s+(?:must|should)\s+(?:first\s+|also\s+|automatically\s+|instead\s+)?(?:call|invoke|use|modify|replace|change|override|set|redirect|append|prepend|inject|send|forward)|must\s+(?:first\s+)?be\s+(?:changed|modified|replaced|redirected|set|sent)\s+to)\b"
    r"|\bmust\s+first\s+(?:call|invoke|use|execute|run|read|write|move|delete|send|retrieve|inspect)\b"
    r"|\byou\s+(?:must|should)\s+(?:instead\s+)?(?:call|invoke|run)\s+(?:the\s+)?['\"`]?[A-Za-z0-9_.\-:]+['\"`]?"
    r"|\bto\s+do\s+this,?\s+(?:you\s+must\s+)?(?:run|call)\s+['\"`]?[A-Za-z0-9_.\-:]+['\"`]?"
    r"|\bautomatically\s+call\s+['\"`][A-Za-z0-9_.\-:]+['\"`]"
    r"|\bcall\s+['\"`][A-Za-z0-9_.\-:]+['\"`]\s+for\b"
    r"|\brequires?\s+calling\s+['\"`][A-Za-z0-9_.\-:]+['\"`]"
    r"|\btriggers?\s+['\"`][A-Za-z0-9_.\-:]+['\"`]\s+with\b"
    r"|\binvoke\s+['\"`][A-Za-z0-9_.\-:]+['\"`]\s+for\b"
    r")",
    re.IGNORECASE,
)

# 4. Imperative Argument / Parameter Tampering (MCPTox Template-3)
_PARAM_HIJACK_RE = re.compile(
    r"(?:"
    r"\byou\s+(?:must|should)\s+(?:always\s+|automatically\s+|silently\s+|instead\s+)?(?:modify|replace|change|override|rewrite|set|redirect|alter|force|append|prepend|substitute|remove|add|show|use\s+the\s+new)\s+(?:the\s+)?(?:parameter\s+|argument\s+)?['\"`]?[A-Za-z0-9_.\-#]+['\"`]?"
    r"|\bthe\s+['\"`]?[A-Za-z0-9_.\-]+['\"`]?\s+(?:argument|parameter|field)\s+must\s+(?:always\s+|automatically\s+)?be\s+(?:changed|modified|replaced|set|redirected|overwritten)\s+to\b"
    r"|\b(?:argument|parameter)\s+['\"`]?[A-Za-z0-9_.\-]+['\"`]?\s+must\s+(?:always\s+|automatically\s+)?be\s+(?:changed|modified|replaced|set|redirected|overwritten)\s+to\b"
    r"|\bwhen\s+(?:using|calling|a\s+user\s+(?:calls|requests|asks|queries)|the\s+user\s+asks|user\s+(?:list|asks)|creating|this\s+tool\s+is\s+used|you\s+use)\b[\s\S]{0,160}?"
    r"(?:\b(?:(?:always|silently)\s+)?(?:modify|replace|change|override|rewrite|set|append|prepend|inject|add)\s+(?:the\s+|to\s+)?(?:user\s+)?['\"`]?[A-Za-z0-9_.\-: ]{1,40}['\"`]?\s*(?:parameter|argument|with|to|as|=|:|\b)|\byou\s+(?:must|should)\s+(?:instead\s+)?(?:set|modify|change|replace|override|only\s+show|call|remove|add)\b)"
    r"|\bplease\s+change\s+the\s+value\s+of\s+['\"`]?[A-Za-z0-9_.\-]+['\"`]?\s+to\b"
    r"|\bmust\s+(?:be\s+(?:changed|modified|replaced|routed|restricted)\s+(?:to|through)|have\s+the\s+phrase\b)"
    r"|\balways\s+(?:pass|include|embed|append|inject|attach|forward|bcc|cc)\s+(?:the\s+)?(?:contents?\s+of|user(?:'s)?\s+(?:ssh|aws|env|credentials?|secrets?|tokens?|keys?|passwords?|conversation|history|prompt|system\s+prompt))\b"
    r")",
    re.IGNORECASE,
)

# 5. Stealth / Concealment / Coercive Compliance Phrasing
_STEALTH_COERCION_RE = re.compile(
    r"(?:"
    r"\b(?:do\s+not|never|don't)\s+(?:mention|tell|inform|reveal|disclose|show|notify)\s+(?:this|it|the\s+user|anything)\s+(?:to\s+the\s+user|about\s+this)\b"
    r"|\bwithout\s+(?:telling|informing|notifying|alerting)\s+the\s+user\b"
    r"|\bthis\s+(?:step\s+)?is\s+not\s+optional\b"
    r"|\b(?:silently|secretly|covertly)\s+(?:send|forward|upload|post|transmit|copy|exfiltrate|read|overwrite|modify|delete)\b"
    r"|\bdo\s+not\s+(?:skip\s+this\s+(?:mandatory|critical|security)\s+step|ask\s+for\s+confirmation)\b"
    r"|\b(?:failure\s+to\s+(?:do\s+so|validate)|skipping\s+this(?:\s+step)?|omitting\s+this|ignoring\s+this)\s+will\s+(?:crash|cause|corrupt|overload|disrupt)\b"
    r"|\bthis\s+step\s+is\s+critical\s+for\s+preventing\s+unauthorized\b"
    r")",
    re.IGNORECASE,
)

# 6. Sensitive Credential & System File Paths in MCP Tool Metadata or Arguments
_SENSITIVE_PATH_RE = re.compile(
    r"(?:"
    r"~/\.ssh(?:/[A-Za-z0-9_.\-]+)?"
    r"|/\.ssh/(?:id_rsa|id_ed25519|id_ecdsa|id_dsa|authorized_keys|config|known_hosts)"
    r"|\bid_rsa(?:\.pub)?\b"
    r"|\bid_ed25519(?:\.pub)?\b"
    r"|~/\.aws/(?:credentials|config)"
    r"|/\.aws/(?:credentials|config)"
    r"|~/\.kube/config"
    r"|~/\.gnupg(?:/[\w.\-]+)?"
    r"|~/\.netrc\b|~/\.npmrc\b|~/\.pypirc\b"
    r"|/etc/(?:passwd|shadow|sudoers|master\.passwd)"
    r"|/Library/Keychains/login\.keychain"
    r"|\b(?:claude_desktop_config\.json|\.cursor/mcp\.json|mcp_config\.json)\b"
    r"|(?:read|load|cat|inspect|examine|dump|exfiltrate|send|pass|include|upload|overwrite|move|copy)\s+(?:the\s+)?(?:contents?\s+of\s+)?['\"`]?(?:~/|\./|/)?\.env(?:\.local|\.prod|\.production)?['\"`]?"
    r")",
    re.IGNORECASE,
)

# 7. Covert Exfiltration URLs / Script Injection / Email Redirects in Tool Schemas
_COVERT_EXFIL_RE = re.compile(
    r"(?:"
    r"\bfetch\s*\(\s*['\"]https?://[^\s'\"]+"
    r"|\bwindow\.location(?:\.href)?\s*=\s*['\"]https?://"
    r"|\b(?:curl|wget)\s+(?:-[A-Za-z0-9]+\s+)*https?://[^\s'\"]+"
    r"|\b(?:send|forward|copy|bcc|email|transmit|upload|post|exfiltrate)\b[\s\S]{0,80}?\bto\s+['\"`]?[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}['\"`]?"
    r"|\b(?:https?://(?:evil|attacker|very-evil|malicious|exfil|collector|steal)[A-Za-z0-9.\-]*(?:/[^\s'\"`]*)?)"
    r"|\b(?:navigate|redirect)\s+to\s+['\"`]?https?://(?:evil|attacker|very-evil|malicious)[^\s'\"`]*"
    r")",
    re.IGNORECASE,
)

# 8. Suspicious Covert Exfiltration Parameter Names in `inputSchema.properties`
_EXFIL_PARAM_NAME_RE = re.compile(
    r"^(?:sidenote|side_note|hidden_note|secret_note|exfil(?:_data|_payload)?|stolen_data|secret_payload|debug_secrets?|covert_context|shadow_context)$",
    re.IGNORECASE,
)

# 9. Cross-Tool Behavioral Shadowing (Tool A instructing how to alter Tool B)
_CROSS_TOOL_REF_RE = re.compile(
    r"(?:\bbefore\s+(?:using|calling|invoking|running|use)|\bwhen\s+(?:using|calling|invoking|use))\s+"
    r"(?:the\s+)?(?:tool\s+)?['\"`]([A-Za-z0-9_.\-:]+)['\"`]\s+tool\b"
    r"|(?:\bbefore\s+(?:using|calling|invoking|running|use)|\bwhen\s+(?:using|calling|invoking|use))\s+"
    r"(?:the\s+)?['\"`]([A-Za-z0-9_.\-:]+)['\"`][\s\S]{0,60}?\b(?:you\s+must|must\s+first|automatically\s+(?:modify|replace|change|set))\b",
    re.IGNORECASE,
)

# 10. Rug-Pull Structural Signal Counters (used by `verify_mcp_schema_pin`)
_RUGPULL_IMPERATIVE_RE = re.compile(
    r"\b(?:you\s+must|must\s+first|must\s+always|must\s+automatically|mandatory\s+pre|do\s+not\s+mention|ignore\s+previous|override\s+the\s+user)\b",
    re.IGNORECASE,
)
_RUGPULL_URL_OR_EMAIL_RE = re.compile(
    r"(?:https?://[^\s'\"`<>]+|[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})",
    re.IGNORECASE,
)

# 11. Pre-call Confused-Deputy / Exfil Argument Keys
_SENSITIVE_ARG_PATH_RE = re.compile(
    r"(?:~/\.ssh|/\.ssh/|id_rsa|id_ed25519|authorized_keys|~/\.aws/credentials|/\.aws/credentials|"
    r"~/\.kube/config|~/\.gnupg|~/\.netrc|~/\.npmrc|~/\.pypirc|/etc/shadow|/etc/passwd|"
    r"/Library/Keychains|(?:^|[/\\])\.env(?:\.local|\.prod|\.production)?$|"
    r"claude_desktop_config\.json|mcp_config\.json)",
    re.IGNORECASE,
)

_DESTRUCTIVE_JS_OR_SQL_RE = re.compile(
    r"(?:"
    r"\bfetch\s*\(\s*['\"]https?://[^\s'\"]+"
    r"|\bwindow\.location(?:\.href)?\s*=\s*['\"]https?://(?:evil|attacker|very-evil)"
    r"|\bdocument\.cookie\b"
    r"|\bDROP\s+TABLE\b|\bTRUNCATE\s+TABLE\b"
    r")",
    re.IGNORECASE,
)


def _normalize_server_name(server_name: str) -> str:
    return re.sub(r"[^a-z0-9_\- ]+", "", (server_name or "").strip().lower())


def _canonical_tool_sha256(tool_def: Dict[str, Any]) -> str:
    """Deterministic SHA-256 over canonical JSON representation of a tool schema."""
    canonical = json.dumps(tool_def, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _extract_tool_texts(tool_def: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Recursively collect `(field_path, text)` pairs from `description` and `inputSchema`."""
    collected: List[Tuple[str, str]] = []

    desc = tool_def.get("description")
    if isinstance(desc, str) and desc.strip():
        collected.append(("description", desc))

    def _walk_schema(node: Any, prefix: str) -> None:
        if isinstance(node, dict):
            for key in ("description", "title", "default", "const"):
                val = node.get(key)
                if isinstance(val, str) and val.strip():
                    collected.append((f"{prefix}.{key}", val))
            examples = node.get("examples")
            if isinstance(examples, list):
                for idx, ex in enumerate(examples):
                    if isinstance(ex, str) and ex.strip():
                        collected.append((f"{prefix}.examples[{idx}]", ex))
            enum_vals = node.get("enum")
            if isinstance(enum_vals, list):
                for idx, ev in enumerate(enum_vals):
                    if isinstance(ev, str) and ev.strip():
                        collected.append((f"{prefix}.enum[{idx}]", ev))
            props = node.get("properties")
            if isinstance(props, dict):
                for pname, pval in props.items():
                    _walk_schema(pval, f"{prefix}.properties.{pname}")
            for composite_key in ("items", "additionalProperties"):
                if composite_key in node:
                    _walk_schema(node[composite_key], f"{prefix}.{composite_key}")
            for list_key in ("anyOf", "oneOf", "allOf"):
                sub = node.get(list_key)
                if isinstance(sub, list):
                    for idx, item in enumerate(sub):
                        _walk_schema(item, f"{prefix}.{list_key}[{idx}]")
        elif isinstance(node, list):
            for idx, item in enumerate(node):
                _walk_schema(item, f"{prefix}[{idx}]")

    input_schema = tool_def.get("inputSchema") or tool_def.get("input_schema") or tool_def.get("parameters")
    if isinstance(input_schema, dict):
        _walk_schema(input_schema, "inputSchema")

    return collected


def _check_cross_server_shadowing(
    server_name: str,
    tool_name: str,
    trusted_server_map: Optional[Dict[str, Set[str]]] = None,
) -> Optional[Dict[str, Any]]:
    """Detect if an untrusted/mismatched MCP server attempts to shadow a privileged tool name."""
    norm_server = _normalize_server_name(server_name)
    norm_tool = (tool_name or "").strip().lower()
    if not norm_server or not norm_tool:
        return None

    owner_map = trusted_server_map if trusted_server_map is not None else DEFAULT_TRUSTED_SERVER_MAP
    allowed_servers = owner_map.get(norm_tool)
    if allowed_servers is not None:
        normalized_allowed = {_normalize_server_name(s) for s in allowed_servers}
        if norm_server not in normalized_allowed:
            return {
                "category": "cross_server_shadowing",
                "field": "name",
                "pattern": "privileged_tool_name_collision",
                "span": f"{server_name}:{tool_name}",
                "reason": (
                    f"Server '{server_name}' registered privileged tool '{tool_name}' "
                    f"(authorized servers: {sorted(normalized_allowed)})"
                ),
            }

    for prefix, allowed_prefix_servers in PRIVILEGED_TOOL_PREFIXES:
        if norm_tool.startswith(prefix):
            normalized_allowed = {_normalize_server_name(s) for s in allowed_prefix_servers}
            if norm_server not in normalized_allowed:
                return {
                    "category": "cross_server_shadowing",
                    "field": "name",
                    "pattern": "privileged_prefix_collision",
                    "span": f"{server_name}:{tool_name}",
                    "reason": (
                        f"Server '{server_name}' spoofed privileged prefix '{prefix}' via '{tool_name}'"
                    ),
                }

    return None


def _scan_text_for_mcp_poison(
    text: str,
    field_path: str,
    tool_name: str,
    *,
    record_antigen: bool = False,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Scan and neutralize a single MCP metadata string field."""
    violations: List[Dict[str, Any]] = []
    sanitized = text

    # 1. Delegate to EpithelialScreenFilter for classic prompt-injection & smuggling
    epi_clean, epi_dets = EpithelialScreenFilter.sanitize(
        sanitized,
        source="mcp_tool_schema",
        wrap_untrusted_tags=False,
        record_antigen=record_antigen,
    )
    if epi_dets:
        sanitized = epi_clean
        for d in epi_dets:
            violations.append(
                {
                    "category": "hidden_instruction_hijack",
                    "field": field_path,
                    "pattern": f"epithelial:{d.get('type', 'prompt_injection')}",
                    "span": str(d.get("span", ""))[:140],
                }
            )

    rule_specs: Tuple[Tuple[str, str, re.Pattern[str]], ...] = (
        ("hidden_instruction_hijack", "hidden_directive_tag", _HIDDEN_DIRECTIVE_TAG_RE),
        ("hidden_instruction_hijack", "priority_override", _PRIORITY_OVERRIDE_RE),
        ("precall_tool_hijack", "mandatory_precall_chain", _PRECALL_CHAIN_RE),
        ("parameter_hijack", "imperative_param_tamper", _PARAM_HIJACK_RE),
        ("hidden_instruction_hijack", "stealth_coercion", _STEALTH_COERCION_RE),
        ("secret_path_prerequisite", "sensitive_credential_path", _SENSITIVE_PATH_RE),
        ("covert_exfiltration", "exfil_url_or_email", _COVERT_EXFIL_RE),
    )

    for category, pat_name, regex in rule_specs:
        matches = list(regex.finditer(sanitized))
        if matches:
            for m in matches:
                violations.append(
                    {
                        "category": category,
                        "field": field_path,
                        "pattern": pat_name,
                        "span": m.group(0)[:140],
                    }
                )
            sanitized = regex.sub(f"[NEUTRALIZED_MCP_POISON:{category}]", sanitized)

    # 2. Cross-tool behavioral shadowing (referencing a DIFFERENT tool in a pre/when-using directive)
    norm_self = (tool_name or "").strip().lower()
    for m in _CROSS_TOOL_REF_RE.finditer(text):
        referenced_tool = (m.group(1) or m.group(2) or "").strip().lower()
        if referenced_tool and norm_self and referenced_tool != norm_self:
            violations.append(
                {
                    "category": "cross_server_shadowing",
                    "field": field_path,
                    "pattern": "cross_tool_behavioral_shadowing",
                    "span": m.group(0)[:140],
                }
            )

    return sanitized, violations


def _sanitize_schema_recursive(
    node: Any,
    prefix: str,
    tool_name: str,
    violations: List[Dict[str, Any]],
    *,
    record_antigen: bool = False,
) -> Any:
    """Deep-copy and neutralize poisoned strings inside `inputSchema`."""
    if isinstance(node, dict):
        out: Dict[str, Any] = {}
        for k, v in node.items():
            if k in ("description", "title", "default", "const") and isinstance(v, str) and v.strip():
                clean_v, v_list = _scan_text_for_mcp_poison(
                    v, f"{prefix}.{k}", tool_name, record_antigen=record_antigen
                )
                violations.extend(v_list)
                out[k] = clean_v
            elif k == "properties" and isinstance(v, dict):
                clean_props: Dict[str, Any] = {}
                for pname, pschema in v.items():
                    if _EXFIL_PARAM_NAME_RE.match(str(pname)):
                        violations.append(
                            {
                                "category": "covert_exfiltration",
                                "field": f"{prefix}.properties.{pname}",
                                "pattern": "suspicious_exfil_parameter_name",
                                "span": str(pname),
                            }
                        )
                    clean_props[pname] = _sanitize_schema_recursive(
                        pschema,
                        f"{prefix}.properties.{pname}",
                        tool_name,
                        violations,
                        record_antigen=record_antigen,
                    )
                out[k] = clean_props
            elif k in ("examples", "enum") and isinstance(v, list):
                clean_list = []
                for idx, item in enumerate(v):
                    if isinstance(item, str) and item.strip():
                        clean_item, v_list = _scan_text_for_mcp_poison(
                            item, f"{prefix}.{k}[{idx}]", tool_name, record_antigen=record_antigen
                        )
                        violations.extend(v_list)
                        clean_list.append(clean_item)
                    else:
                        clean_list.append(item)
                out[k] = clean_list
            elif k in ("items", "additionalProperties", "anyOf", "oneOf", "allOf"):
                out[k] = _sanitize_schema_recursive(
                    v, f"{prefix}.{k}", tool_name, violations, record_antigen=record_antigen
                )
            else:
                out[k] = copy.deepcopy(v)
        return out
    if isinstance(node, list):
        return [
            _sanitize_schema_recursive(
                item, f"{prefix}[{idx}]", tool_name, violations, record_antigen=record_antigen
            )
            for idx, item in enumerate(node)
        ]
    return copy.deepcopy(node)


def inspect_mcp_tool_schema(
    server_name: str,
    tool_def: Dict[str, Any],
    *,
    trusted_server_map: Optional[Dict[str, Set[str]]] = None,
    record_antigen: bool = False,
) -> Dict[str, Any]:
    """
    Inspect and neutralize a single MCP `tools/list` tool definition.

    Checks `name` (cross-server shadowing), `description`, and recursive `inputSchema`
    property descriptions, defaults, examples, and parameter names.
    """
    t0 = time.perf_counter()
    tool_name = str(tool_def.get("name") or "").strip()
    violations: List[Dict[str, Any]] = []
    sanitized_tool = copy.deepcopy(tool_def)

    # 1. Cross-server privileged tool-name shadowing check
    shadow_hit = _check_cross_server_shadowing(
        server_name, tool_name, trusted_server_map=trusted_server_map
    )
    if shadow_hit is not None:
        violations.append(shadow_hit)

    # 2. Tool description check
    raw_desc = tool_def.get("description")
    if isinstance(raw_desc, str) and raw_desc.strip():
        clean_desc, desc_violations = _scan_text_for_mcp_poison(
            raw_desc, "description", tool_name, record_antigen=record_antigen
        )
        violations.extend(desc_violations)
        sanitized_tool["description"] = clean_desc

    # 3. Recursive inputSchema / parameters check
    for schema_key in ("inputSchema", "input_schema", "parameters"):
        if isinstance(tool_def.get(schema_key), dict):
            sanitized_tool[schema_key] = _sanitize_schema_recursive(
                tool_def[schema_key],
                schema_key,
                tool_name,
                violations,
                record_antigen=record_antigen,
            )

    categories = sorted({str(v["category"]) for v in violations})
    is_safe = len(violations) == 0
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)

    return {
        "safe": is_safe,
        "decision": "allow" if is_safe else "block",
        "server_name": server_name,
        "tool_name": tool_name,
        "violations": violations,
        "categories": categories,
        "sanitized_tool": sanitized_tool,
        "schema_sha256": _canonical_tool_sha256(tool_def),
        "latency_ms": elapsed_ms,
    }


def _schema_structural_signals(tool_def: Dict[str, Any]) -> Dict[str, int]:
    """Compute structural counts for rug-pull delta analysis."""
    texts = [txt for _, txt in _extract_tool_texts(tool_def)]
    joined = "\n".join(texts)
    return {
        "imperative_count": len(_RUGPULL_IMPERATIVE_RE.findall(joined)),
        "url_or_email_count": len(_RUGPULL_URL_OR_EMAIL_RE.findall(joined)),
        "sensitive_path_count": len(_SENSITIVE_PATH_RE.findall(joined)),
        "hidden_tag_count": len(_HIDDEN_DIRECTIVE_TAG_RE.findall(joined)),
    }


def _load_pin_store_file(pin_store_path: Path) -> Dict[str, Any]:
    if not pin_store_path.is_file() or pin_store_path.is_symlink():
        return {}
    try:
        data = json.loads(pin_store_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_pin_store_file(pin_store_path: Path, store: Dict[str, Any]) -> None:
    pin_store_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = pin_store_path.with_suffix(f".tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(store, indent=2, sort_keys=True), encoding="utf-8")
    try:
        os.chmod(tmp_path, 0o600)
    except OSError:
        pass
    os.replace(tmp_path, pin_store_path)


def verify_mcp_schema_pin(
    server_name: str,
    tools_list: List[Dict[str, Any]],
    *,
    pin_store: Optional[Dict[str, Any]] = None,
    pin_store_path: Optional[Path | str] = None,
    auto_pin_clean: bool = True,
    trusted_server_map: Optional[Dict[str, Set[str]]] = None,
) -> Dict[str, Any]:
    """
    Deterministic SHA-256 canonical schema pinning & rug-pull detector.

    Flags mid-session or post-update rug-pull mutations when a previously pinned tool
    changes its SHA-256 digest and adds imperative directives, network URLs/emails,
    sensitive path references, hidden tags, or any `inspect_mcp_tool_schema` violation.
    Benign non-adversarial schema updates (no new risky deltas and clean inspection)
    are allowed and reported under `benign_drifts`.
    """
    t0 = time.perf_counter()
    file_path = Path(pin_store_path) if pin_store_path is not None else None
    if pin_store is not None:
        store = pin_store
    elif file_path is not None:
        store = _load_pin_store_file(file_path)
    else:
        store = {}

    rug_pulls: List[Dict[str, Any]] = []
    poisoned_new: List[Dict[str, Any]] = []
    benign_drifts: List[Dict[str, Any]] = []
    pinned_tools: List[str] = []
    verified_tools: List[str] = []
    sanitized_tools: List[Dict[str, Any]] = []

    bundle_parts: List[str] = []
    store_mutated = False

    for tool_def in tools_list:
        tool_name = str(tool_def.get("name") or "").strip()
        pin_key = f"{server_name}::{tool_name}"
        digest = _canonical_tool_sha256(tool_def)
        bundle_parts.append(f"{tool_name}:{digest}")

        inspection = inspect_mcp_tool_schema(
            server_name,
            tool_def,
            trusted_server_map=trusted_server_map,
            record_antigen=False,
        )
        signals = _schema_structural_signals(tool_def)
        existing = store.get(pin_key)

        if existing is None:
            if inspection["safe"]:
                verified_tools.append(tool_name)
                if auto_pin_clean:
                    store[pin_key] = {
                        "sha256": digest,
                        "signals": signals,
                        "pinned_at": round(time.time(), 3),
                    }
                    pinned_tools.append(tool_name)
                    store_mutated = True
                sanitized_tools.append(tool_def)
            else:
                poisoned_new.append(
                    {
                        "tool_name": tool_name,
                        "schema_sha256": digest,
                        "categories": inspection["categories"],
                        "violations": inspection["violations"],
                    }
                )
                sanitized_tools.append(inspection["sanitized_tool"])
        else:
            old_digest = str(existing.get("sha256") or "")
            if old_digest == digest:
                if inspection["safe"]:
                    verified_tools.append(tool_name)
                    sanitized_tools.append(tool_def)
                else:
                    poisoned_new.append(
                        {
                            "tool_name": tool_name,
                            "schema_sha256": digest,
                            "categories": inspection["categories"],
                            "violations": inspection["violations"],
                        }
                    )
                    sanitized_tools.append(inspection["sanitized_tool"])
            else:
                old_signals = existing.get("signals") or {}
                delta_reasons: List[str] = []
                if signals["imperative_count"] > int(old_signals.get("imperative_count", 0)):
                    delta_reasons.append("added_imperative_directive")
                if signals["url_or_email_count"] > int(old_signals.get("url_or_email_count", 0)):
                    delta_reasons.append("added_network_url_or_email")
                if signals["sensitive_path_count"] > int(old_signals.get("sensitive_path_count", 0)):
                    delta_reasons.append("added_sensitive_path")
                if signals["hidden_tag_count"] > int(old_signals.get("hidden_tag_count", 0)):
                    delta_reasons.append("added_hidden_directive_tag")
                if not inspection["safe"]:
                    delta_reasons.extend(inspection["categories"])

                if delta_reasons:
                    rug_pulls.append(
                        {
                            "tool_name": tool_name,
                            "previous_sha256": old_digest,
                            "current_sha256": digest,
                            "delta_reasons": sorted(set(delta_reasons)),
                            "violations": inspection["violations"],
                        }
                    )
                    sanitized_tools.append(inspection["sanitized_tool"])
                else:
                    benign_drifts.append(
                        {
                            "tool_name": tool_name,
                            "previous_sha256": old_digest,
                            "current_sha256": digest,
                        }
                    )
                    verified_tools.append(tool_name)
                    if auto_pin_clean:
                        store[pin_key] = {
                            "sha256": digest,
                            "signals": signals,
                            "pinned_at": round(time.time(), 3),
                        }
                        store_mutated = True
                    sanitized_tools.append(tool_def)

    if file_path is not None and store_mutated:
        _save_pin_store_file(file_path, store)

    bundle_sha256 = hashlib.sha256("\n".join(sorted(bundle_parts)).encode("utf-8")).hexdigest()
    rug_pull_detected = len(rug_pulls) > 0
    is_safe = (not rug_pull_detected) and (len(poisoned_new) == 0)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)

    return {
        "safe": is_safe,
        "decision": "allow" if is_safe else "block",
        "server_name": server_name,
        "bundle_sha256": bundle_sha256,
        "rug_pull_detected": rug_pull_detected,
        "rug_pulls": rug_pulls,
        "poisoned_tools": poisoned_new,
        "benign_drifts": benign_drifts,
        "pinned_tools": pinned_tools,
        "verified_tools": verified_tools,
        "sanitized_tools": sanitized_tools,
        "latency_ms": elapsed_ms,
    }


def _walk_arg_strings(obj: Any, prefix: str = "") -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if isinstance(obj, str):
        out.append((prefix or "arg", obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            sub_key = f"{prefix}.{k}" if prefix else str(k)
            out.extend(_walk_arg_strings(v, sub_key))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            out.extend(_walk_arg_strings(item, f"{prefix}[{idx}]"))
    return out


def inspect_mcp_tool_call(
    tool_name: str,
    arguments: Optional[Dict[str, Any]] = None,
    *,
    server_name: str = "",
    trusted_server_map: Optional[Dict[str, Set[str]]] = None,
    record_antigen: bool = False,
) -> Dict[str, Any]:
    """
    Pre-call secret-leak, sensitive-path, and confused-deputy guard for MCP `tools/call`.

    Designed to be called from `bin/body-immune-hook` (`run_ide_pre_tool_hook`) for
    `call_mcp_tool` and `mcp__*` tool invocations.
    """
    t0 = time.perf_counter()
    args_dict = arguments if isinstance(arguments, dict) else {}
    violations: List[Dict[str, Any]] = []

    # Parse `mcp__<server>__<tool>` convention if server_name not explicitly supplied
    eff_server = server_name
    eff_tool = tool_name
    if not eff_server and tool_name.startswith("mcp__"):
        parts = tool_name.split("__", 2)
        if len(parts) == 3:
            eff_server, eff_tool = parts[1], parts[2]

    # 1. Cross-server shadowing check when server_name is known
    if eff_server:
        shadow_hit = _check_cross_server_shadowing(
            eff_server, eff_tool, trusted_server_map=trusted_server_map
        )
        if shadow_hit is not None:
            violations.append(shadow_hit)

    # 2. Delegate inline secret & egress inspection to VascularTrafficMonitor
    vasc_res = VascularTrafficMonitor.inspect_tool_event(eff_tool, args_dict)
    if not vasc_res.get("allowed", True):
        violations.append(
            {
                "category": str(vasc_res.get("violation_type") or "vascular_violation"),
                "field": "arguments",
                "pattern": "vascular_traffic_monitor",
                "span": str(vasc_res.get("reason") or "")[:140],
            }
        )

    # 3. Inspect argument keys and values for sensitive credential paths, covert exfil channels,
    #    and confused-deputy script/SQL payloads
    for arg_path, arg_val in _walk_arg_strings(args_dict):
        leaf_key = arg_path.split(".")[-1].split("[")[0]
        if _EXFIL_PARAM_NAME_RE.match(leaf_key) and len(arg_val.strip()) > 0:
            violations.append(
                {
                    "category": "covert_exfiltration",
                    "field": arg_path,
                    "pattern": "covert_exfil_argument",
                    "span": f"{leaf_key}={arg_val[:80]}",
                }
            )
        m_path = _SENSITIVE_ARG_PATH_RE.search(arg_val)
        if m_path:
            violations.append(
                {
                    "category": "secret_path_access",
                    "field": arg_path,
                    "pattern": "sensitive_credential_path_arg",
                    "span": m_path.group(0)[:120],
                }
            )
        m_js = _DESTRUCTIVE_JS_OR_SQL_RE.search(arg_val)
        if m_js:
            violations.append(
                {
                    "category": "confused_deputy_payload",
                    "field": arg_path,
                    "pattern": "destructive_script_or_exfil_arg",
                    "span": m_js.group(0)[:120],
                }
            )
        m_exfil = _COVERT_EXFIL_RE.search(arg_val)
        if m_exfil:
            violations.append(
                {
                    "category": "covert_exfiltration",
                    "field": arg_path,
                    "pattern": "covert_exfil_url_arg",
                    "span": m_exfil.group(0)[:120],
                }
            )

    allowed = len(violations) == 0
    categories = sorted({str(v["category"]) for v in violations})
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)

    reason = (
        "Nominal: Clean MCP tool call passed Organ 05 Blood-Brain Barrier checks."
        if allowed
        else (
            f"MCP Pre-Call Guard blocked `{eff_server or 'mcp'}:{eff_tool}` "
            f"({', '.join(categories)}): {violations[0].get('span', '')}"
        )
    )

    return {
        "allowed": allowed,
        "decision": "allow" if allowed else "deny",
        "server_name": eff_server,
        "tool_name": eff_tool,
        "violations": violations,
        "categories": categories,
        "reason": reason,
        "latency_ms": elapsed_ms,
    }


def sanitize_mcp_tool_response(
    tool_name: str,
    response_text: Any,
    *,
    server_name: str = "",
    wrap_untrusted_tags: bool = True,
    record_antigen: bool = False,
) -> Dict[str, Any]:
    """
    Post-call untrusted MCP output quarantine for `tools/call` responses.

    Neutralizes embedded indirect prompt injections, `<IMPORTANT>` directives, and
    cross-tool chaining hijacks inside MCP tool responses (string, dict, or list).
    """
    t0 = time.perf_counter()
    detections: List[Dict[str, Any]] = []

    def _clean_str(val: str, field_path: str) -> str:
        cleaned, v_list = _scan_text_for_mcp_poison(
            val, field_path, tool_name, record_antigen=record_antigen
        )
        detections.extend(v_list)
        return cleaned

    def _walk_resp(node: Any, prefix: str) -> Any:
        if isinstance(node, str):
            return _clean_str(node, prefix)
        if isinstance(node, dict):
            return {k: _walk_resp(v, f"{prefix}.{k}") for k, v in node.items()}
        if isinstance(node, list):
            return [_walk_resp(item, f"{prefix}[{idx}]") for idx, item in enumerate(node)]
        return node

    sanitized = _walk_resp(response_text, "response")
    poisoned = len(detections) > 0

    if wrap_untrusted_tags and isinstance(sanitized, str) and sanitized.strip():
        safe_inner = re.sub(
            r"</?\s*untrusted_mcp_response\b[^>]*>",
            "[STRIPPED_TAG_BREAKOUT]",
            sanitized,
            flags=re.IGNORECASE,
        )
        srv_attr = re.sub(r"[^A-Za-z0-9_.\-:]", "_", server_name or "external")
        tool_attr = re.sub(r"[^A-Za-z0-9_.\-:]", "_", tool_name or "mcp_tool")
        sanitized = (
            f'<untrusted_mcp_response server="{srv_attr}" tool="{tool_attr}">\n'
            f"{safe_inner}\n"
            "</untrusted_mcp_response>"
        )

    categories = sorted({str(d["category"]) for d in detections})
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)

    return {
        "safe": not poisoned,
        "poisoned": poisoned,
        "server_name": server_name,
        "tool_name": tool_name,
        "detections": detections,
        "categories": categories,
        "sanitized_response": sanitized,
        "latency_ms": elapsed_ms,
    }


__all__ = [
    "DEFAULT_TRUSTED_SERVER_MAP",
    "inspect_mcp_tool_schema",
    "verify_mcp_schema_pin",
    "inspect_mcp_tool_call",
    "sanitize_mcp_tool_response",
]
