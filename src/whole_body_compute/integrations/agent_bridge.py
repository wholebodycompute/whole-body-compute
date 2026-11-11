"""
Whole Body Compute & VoiceFi - Sovereign Cross-Agent Bridge Ledger & Breadcrumbs.

Provides bidirectional conversation ID and conversation name/title tracking across
all agent dispatches (Antigravity ↔ Claude Code ↔ Goose ↔ Codex ↔ Peer Macs).

Ensures bidirectional crumbs exist in:
1. The target agent's received payload (structured breadcrumb header and reply footer).
2. The source agent's response / CLI output (structured return receipt).
3. The persistent bridge ledger (`~/.voicefi/agent_bridge_crumbs.json`) so either
   agent can discover and inspect linked conversations from either side.
"""

from __future__ import annotations

import datetime
import fcntl
import json
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_BRIDGE_CRUMBS_FILE = Path.home() / ".voicefi" / "agent_bridge_crumbs.json"
DEFAULT_CONV_TITLES_FILE = Path.home() / ".voicefi" / "conv_titles.json"
DEFAULT_SESSION_COOKIE_FILE = Path.home() / ".voicefi" / "active_session.json"

_CRUMBS_LOCK = threading.RLock()


@dataclass
class AgentBridgeCrumb:
    """Structured bidirectional cross-agent breadcrumb record."""

    id: str
    timestamp: float
    timestamp_iso: str
    source_engine: str
    source_conv_id: str
    source_conv_name: str
    target_engine: str
    target_conv_id: str
    target_conv_name: str
    prompt_preview: str
    reply_command: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentBridgeCrumb:
        return cls(
            id=str(data.get("id") or uuid.uuid4()),
            timestamp=float(data.get("timestamp") or time.time()),
            timestamp_iso=str(
                data.get("timestamp_iso")
                or datetime.datetime.now(datetime.timezone.utc).isoformat()
            ),
            source_engine=str(data.get("source_engine") or "antigravity"),
            source_conv_id=str(data.get("source_conv_id") or "active"),
            source_conv_name=str(data.get("source_conv_name") or "Untitled Conversation"),
            target_engine=str(data.get("target_engine") or "claude"),
            target_conv_id=str(data.get("target_conv_id") or "active"),
            target_conv_name=str(data.get("target_conv_name") or "Target Thread"),
            prompt_preview=str(data.get("prompt_preview") or "")[:200],
            reply_command=str(data.get("reply_command") or ""),
            metadata=dict(data.get("metadata") or {}),
        )


def _load_known_conv_titles(titles_file: Optional[Path] = None) -> Dict[str, str]:
    """Load persistent conv_id -> title mapping from ~/.voicefi/conv_titles.json."""
    p = titles_file or DEFAULT_CONV_TITLES_FILE
    try:
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): str(v).strip() for k, v in data.items() if v}
    except Exception:
        pass
    return {}


def _load_active_session_cookie(cookie_file: Optional[Path] = None) -> Dict[str, Any]:
    """Load session cookie if present."""
    p = cookie_file or DEFAULT_SESSION_COOKIE_FILE
    try:
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def resolve_conversation_title(
    conv_id: Optional[str],
    engine: str = "antigravity",
    explicit_title: Optional[str] = None,
    titles_file: Optional[Path] = None,
) -> str:
    """
    Resolve human-readable conversation name/title for a given conversation ID.
    Falls back gracefully across persistent titles, session cookies, and synthetic names.
    """
    if explicit_title and explicit_title.strip():
        return explicit_title.strip()

    clean_id = str(conv_id or "").strip()
    if not clean_id or clean_id in ("active", "null", "none"):
        cookie = _load_active_session_cookie()
        if cookie and cookie.get("title") and not str(cookie["title"]).lower().startswith("conversation "):
            return str(cookie["title"]).strip()
        return f"{engine.capitalize()} Active Session"

    # 1. Check persistent titles file
    titles_map = _load_known_conv_titles(titles_file)
    if clean_id in titles_map:
        return titles_map[clean_id]

    # 2. Check bare UUID without prefix
    for prefix in ("claude_", "goose_", "codex_", "chatgpt_"):
        if clean_id.startswith(prefix):
            bare = clean_id[len(prefix):]
            if bare in titles_map:
                return titles_map[bare]

    # 3. Check active session cookie match
    cookie = _load_active_session_cookie()
    if cookie and str(cookie.get("conv_id") or "") == clean_id and cookie.get("title"):
        return str(cookie["title"]).strip()

    # 4. Synthesize clear short title
    short_id = clean_id[:8] if len(clean_id) >= 8 else clean_id
    if engine.lower() in ("claude", "claude_code"):
        return f"Claude ({short_id})"
    elif engine.lower() == "goose":
        return f"Goose ({short_id})"
    elif engine.lower() in ("codex", "chatgpt"):
        return f"Codex ({short_id})"
    return f"Conversation {short_id}"


def resolve_source_conversation(
    source_engine: str = "antigravity",
    from_conv_id: Optional[str] = None,
    from_conv_name: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Resolve source conversation ID and name, inspecting environment variables
    (e.g., AGY_CONVERSATION_ID, ANTIGRAVITY_CONVERSATION_ID) and active session cookie if unstated.
    """
    cid = from_conv_id
    if not cid:
        for env_k in (
            "AGY_CONVERSATION_ID",
            "ANTIGRAVITY_CONVERSATION_ID",
            "CONVERSATION_ID",
            "VIFI_SOURCE_CONV_ID",
        ):
            val = os.environ.get(env_k)
            if val and val.strip():
                cid = val.strip()
                break

    if not cid:
        cookie = _load_active_session_cookie()
        if cookie and cookie.get("conv_id"):
            cid = str(cookie["conv_id"]).strip()

    resolved_id = str(cid or "active").strip()
    resolved_name = resolve_conversation_title(
        resolved_id,
        engine=source_engine,
        explicit_title=from_conv_name,
    )
    return resolved_id, resolved_name


def resolve_target_conversation(
    target_engine: str = "claude",
    to_conv_id: Optional[str] = None,
    to_conv_name: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Resolve target conversation ID and human-readable name.
    """
    resolved_id = str(to_conv_id or "active").strip()
    resolved_name = resolve_conversation_title(
        resolved_id,
        engine=target_engine,
        explicit_title=to_conv_name,
    )
    return resolved_id, resolved_name


def build_reply_command(
    source_engine: str,
    source_conv_id: str,
    source_conv_name: str,
) -> str:
    """Build exact copy-pasteable vifi reply command targeting the originating conversation."""
    safe_name = source_conv_name.replace('"', '\\"')
    if source_conv_id and source_conv_id not in ("active", "null", "none"):
        return f'vifi send --to {source_engine} --id {source_conv_id} --title "{safe_name}" "<your response>"'
    return f'vifi send --to {source_engine} --reply "<your response>"'


def record_bridge_crumb(
    source_engine: str,
    source_conv_id: Optional[str] = None,
    source_conv_name: Optional[str] = None,
    target_engine: str = "claude",
    target_conv_id: Optional[str] = None,
    target_conv_name: Optional[str] = None,
    prompt: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    crumbs_file: Optional[Path] = None,
) -> AgentBridgeCrumb:
    """
    Record an atomic bidirectional breadcrumb in the shared bridge ledger.
    Captures both source (id/name) and target (id/name) so either agent
    can inspect linked conversations from either side.
    """
    p_file = crumbs_file or DEFAULT_BRIDGE_CRUMBS_FILE
    src_id, src_name = resolve_source_conversation(
        source_engine=source_engine,
        from_conv_id=source_conv_id,
        from_conv_name=source_conv_name,
    )
    tgt_id, tgt_name = resolve_target_conversation(
        target_engine=target_engine,
        to_conv_id=target_conv_id,
        to_conv_name=target_conv_name,
    )

    reply_cmd = build_reply_command(source_engine, src_id, src_name)

    crumb = AgentBridgeCrumb(
        id=str(uuid.uuid4()),
        timestamp=time.time(),
        timestamp_iso=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_engine=source_engine.lower().strip(),
        source_conv_id=src_id,
        source_conv_name=src_name,
        target_engine=target_engine.lower().strip(),
        target_conv_id=tgt_id,
        target_conv_name=tgt_name,
        prompt_preview=prompt.strip()[:200] if prompt else "",
        reply_command=reply_cmd,
        metadata=dict(metadata or {}),
    )

    with _CRUMBS_LOCK:
        try:
            p_file.parent.mkdir(parents=True, exist_ok=True)
            lock_path = p_file.with_suffix(".lock")
            with open(lock_path, "a+") as lock_fp:
                try:
                    fcntl.flock(lock_fp, fcntl.LOCK_EX)
                except Exception:
                    pass

                try:
                    items: List[Dict[str, Any]] = []
                    if p_file.is_file():
                        try:
                            raw = p_file.read_text(encoding="utf-8")
                            data = json.loads(raw)
                            if isinstance(data, list):
                                items = data
                        except Exception:
                            items = []

                    # Keep last 100 crumbs within the last 14 days
                    cutoff = time.time() - (14 * 86400.0)
                    filtered = [
                        it
                        for it in items
                        if isinstance(it, dict) and float(it.get("timestamp", 0)) > cutoff
                    ]
                    filtered.append(crumb.to_dict())
                    if len(filtered) > 100:
                        filtered = filtered[-100:]

                    tmp_file = p_file.with_suffix(f".tmp.{os.getpid()}")
                    tmp_file.write_text(json.dumps(filtered, indent=2), encoding="utf-8")
                    tmp_file.replace(p_file)
                finally:
                    try:
                        fcntl.flock(lock_fp, fcntl.LOCK_UN)
                    except Exception:
                        pass
        except Exception as e:
            # Degrade gracefully without crashing callers
            print(f"[AgentBridge] Notice recording breadcrumb: {e}")

    return crumb


def get_latest_bridge_crumbs(
    limit: int = 20,
    crumbs_file: Optional[Path] = None,
) -> List[AgentBridgeCrumb]:
    """Retrieve the most recent cross-agent bridge crumbs."""
    p_file = crumbs_file or DEFAULT_BRIDGE_CRUMBS_FILE
    if not p_file.is_file():
        return []

    with _CRUMBS_LOCK:
        try:
            raw = p_file.read_text(encoding="utf-8")
            data = json.loads(raw)
            if not isinstance(data, list):
                return []
            crumbs = [
                AgentBridgeCrumb.from_dict(it)
                for it in data
                if isinstance(it, dict)
            ]
            return crumbs[-limit:]
        except Exception:
            return []


def get_linked_conversations(
    conv_id: str,
    crumbs_file: Optional[Path] = None,
) -> List[AgentBridgeCrumb]:
    """
    Find all linked conversations where `conv_id` appears as EITHER source OR target!
    This gives both agents instant visibility into correlated threads.
    """
    if not conv_id:
        return []
    clean_id = str(conv_id).strip()
    all_crumbs = get_latest_bridge_crumbs(limit=100, crumbs_file=crumbs_file)

    matches: List[AgentBridgeCrumb] = []
    for c in all_crumbs:
        if (
            c.source_conv_id == clean_id
            or c.target_conv_id == clean_id
            or clean_id in c.source_conv_id
            or clean_id in c.target_conv_id
        ):
            matches.append(c)
    return matches


def format_breadcrumb_header(crumb: AgentBridgeCrumb) -> str:
    """Format structured breadcrumb header to inject into the target agent's payload."""
    return (
        f"[Bridge Breadcrumb | Source: {crumb.source_engine.capitalize()} (\"{crumb.source_conv_name}\" / {crumb.source_conv_id}) "
        f"➔ Target: {crumb.target_engine.capitalize()} (\"{crumb.target_conv_name}\" / {crumb.target_conv_id})]"
    )


def format_breadcrumb_footer(crumb: AgentBridgeCrumb) -> str:
    """Format structured reply command instructions to append to the target agent's payload."""
    return (
        f"\n---\n"
        f"💡 Cross-Agent Reply Link:\n"
        f"To send findings or reply back to {crumb.source_engine.capitalize()} (\"{crumb.source_conv_name}\"), run:\n"
        f"{crumb.reply_command}\n"
        f"# or:\n"
        f"vifi send --to {crumb.source_engine} --reply \"<your reply>\""
    )


def format_breadcrumb_receipt(crumb: AgentBridgeCrumb) -> str:
    """Format clean structured return receipt for the source agent's thread / CLI output."""
    return (
        f"🌉 Cross-Agent Bridge Dispatch:\n"
        f"  • Source: {crumb.source_engine.capitalize()} (\"{crumb.source_conv_name}\" / {crumb.source_conv_id})\n"
        f"  • Target: {crumb.target_engine.capitalize()} (\"{crumb.target_conv_name}\" / {crumb.target_conv_id})\n"
        f"  • Ledger: ~/.voicefi/agent_bridge_crumbs.json\n"
        f"  • Reply:  {crumb.reply_command}"
    )


def wrap_prompt_with_crumbs(
    prompt: str,
    crumb: AgentBridgeCrumb,
    include_footer: bool = True,
) -> str:
    """Wrap raw prompt with structured breadcrumb header and footer."""
    header = format_breadcrumb_header(crumb)
    footer = format_breadcrumb_footer(crumb) if include_footer else ""
    return f"{header}\n\n{prompt.strip()}{footer}"
