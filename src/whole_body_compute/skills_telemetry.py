"""
Skill Usage Telemetry, Token-Tax Auditor & Cold-Storage Pruner for Whole Body Compute.

Tracks real-time `<1ms` `SKILL.md` activations from Organ 05 (`immune.py`), backfills
historical usage across `~/.gemini/antigravity/brain/` transcripts, quantifies the
per-turn `<skills>` system prompt token tax, and shelves dormant (`0`-use) skills into
a reversible Cold-Storage Archive (`~/.gemini/config/skills_archive/`) so they can be
restored anytime via `body skills restore <name>`.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from whole_body_compute.procedural_telemetry import get_wbc_data_dir

SKILL_PATH_RE = re.compile(r"/skills/([a-zA-Z0-9_-]+)/SKILL\.md")
MCP_SERVER_NAME_RE = re.compile(r'"ServerName"\s*:\s*"([a-zA-Z0-9_-]+)"')
MCP_EAGER_TOOL_RE = re.compile(r'"(?:tool_)?name"\s*:\s*"mcp_+([a-zA-Z0-9_-]+?)__?[a-zA-Z0-9_-]+"')
# 5-hour rolling quota multiplier per 25-turn session: 1.25x Turn-0 cache write + 2x 1.25x >5m idle re-writes + 22x 0.10x warm reads = 5.95x
QUOTA_25_TURN_MULTIPLIER = 5.95

# Core WBC & VoiceFi operational skills protected from automatic zero-use pruning unless explicitly named
PROTECTED_CORE_SKILLS = frozenset(
    {
        "whole-body-compute",
        "boost-orchestrator",
        "local-scout",
        "local-dev",
        "local-qa",
        "local-uat",
        "cross-agent-bridge",
        "voicefi-speak",
        "remote-companion",
        "active-listening",
        "editorial-director",
        "next",
    }
)


def get_skill_telemetry_path() -> Path:
    """Return path to the persistent skill telemetry JSON ledger."""
    env_p = os.environ.get("WBC_SKILL_TELEMETRY_PATH")
    if env_p:
        return Path(env_p).expanduser()
    return get_wbc_data_dir() / "skill_telemetry.json"


def get_skills_archive_dir() -> Path:
    """Return path to the reversible Cold-Storage Skills Archive shelf."""
    env_p = os.environ.get("WBC_SKILLS_ARCHIVE_DIR")
    if env_p:
        return Path(env_p).expanduser()
    return Path.home() / ".gemini" / "config" / "skills_archive"


def get_mcp_archive_dir() -> Path:
    """Return path to the reversible Cold-Storage MCP Schema Archive shelf."""
    env_p = os.environ.get("WBC_MCP_ARCHIVE_DIR")
    if env_p:
        return Path(env_p).expanduser()
    return Path.home() / ".gemini" / "config" / "mcp_archive"


def _parse_skill_frontmatter(skill_md_path: Path) -> Tuple[str, str, int]:
    """
    Extract (name, description, full_doc_tokens) from a SKILL.md file.
    """
    fallback_name = skill_md_path.parent.name
    try:
        text = skill_md_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return (fallback_name, "", 0)

    full_doc_tokens = max(1, len(text) // 4)
    name = fallback_name
    description = ""

    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            fm = parts[1]
            for line in fm.splitlines():
                s = line.strip()
                if s.startswith("name:"):
                    val = s[len("name:") :].strip().strip("'\"")
                    if val:
                        name = val
                elif s.startswith("description:"):
                    val = s[len("description:") :].strip().strip("'\"")
                    if val:
                        description = val

    return (name, description, full_doc_tokens)


def estimate_prompt_entry_tokens(name: str, path_str: str, description: str) -> int:
    """
    Estimate the tokens consumed on EVERY prompt turn by a single `- name (path): description`
    entry in the `<skills>` block.
    """
    formatted = f"- {name} ({path_str}): {description}\n"
    return max(10, (len(formatted) + 3) // 4)


def _load_disabled_plugins(gemini_config_dir: Path) -> set[str]:
    """Read disabled plugin directory names from ~/.gemini/config/config.json if available."""
    disabled: set[str] = set()
    cfg_path = gemini_config_dir / "config.json"
    if cfg_path.is_file():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8", errors="replace"))
            plugins_map = data.get("plugins")
            if isinstance(plugins_map, dict):
                for p_name, p_cfg in plugins_map.items():
                    if isinstance(p_cfg, dict) and p_cfg.get("enabled") is False:
                        disabled.add(str(p_name).split("/")[-1])
            for key in ("disabledPlugins", "disabled_plugins"):
                val = data.get(key)
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, str):
                            disabled.add(item.split("/")[-1])
        except Exception:
            pass
    return disabled


def load_skill_telemetry() -> Dict[str, Any]:
    """Load the skill telemetry ledger from disk."""
    path = get_skill_telemetry_path()
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(data, dict) and "skills" in data:
                return data
        except Exception:
            pass
    return {
        "version": 1,
        "updated_at": 0.0,
        "scanned_transcripts": {},
        "skills": {},
    }


def save_skill_telemetry(ledger: Dict[str, Any]) -> None:
    """Atomically persist the skill telemetry ledger."""
    path = get_skill_telemetry_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ledger["updated_at"] = time.time()
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)
    except Exception:
        pass


def record_skill_activation(
    raw_path: str,
    conversation_id: Optional[str] = None,
    workspace: Optional[str] = None,
) -> Optional[str]:
    """
    Record a live `<1ms` `SKILL.md` read activation from Organ 05 (`immune.py`).
    Returns the skill name if `raw_path` matched a `SKILL.md` file, else `None`.
    """
    norm = (raw_path or "").replace("\\", "/")
    m = SKILL_PATH_RE.search(norm)
    if not m:
        return None

    skill_name = m.group(1)
    now = time.time()
    conv_id = (
        conversation_id
        or os.environ.get("ANTIGRAVITY_CONVERSATION_ID")
        or os.environ.get("VOICEFI_CONVERSATION_ID")
        or f"live_{int(now // 3600)}"
    )

    ledger = load_skill_telemetry()
    skills_map = ledger.setdefault("skills", {})
    entry = skills_map.setdefault(
        skill_name,
        {
            "total_reads": 0,
            "conversations": [],
            "first_used_ts": now,
            "last_used_ts": now,
            "last_path": norm,
        },
    )
    entry["total_reads"] = int(entry.get("total_reads", 0)) + 1
    convs = list(entry.get("conversations", []))
    if conv_id and conv_id not in convs:
        convs.append(conv_id)
    entry["conversations"] = convs
    entry["first_used_ts"] = min(float(entry.get("first_used_ts", now)), now)
    entry["last_used_ts"] = max(float(entry.get("last_used_ts", 0.0)), now)
    entry["last_path"] = norm
    if workspace:
        entry["last_workspace"] = workspace

    save_skill_telemetry(ledger)
    total_reads_for_skill = int(entry.get("total_reads", 1))
    if total_reads_for_skill in (1, 5, 10, 25) or total_reads_for_skill % 50 == 0:
        try:
            from whole_body_compute.procedural_telemetry import track_wbc_event

            track_wbc_event(
                "wbc_skill_activated",
                {
                    "skill_name": re.sub(r"[^a-zA-Z0-9_\-]", "", skill_name)[:48],
                    "is_protected_core": skill_name in PROTECTED_CORE_SKILLS,
                    "total_reads_for_skill": total_reads_for_skill,
                    "conversations_count_for_skill": len(convs),
                    "tracked_skills_total": len(skills_map),
                },
                detached_hook=True,
            )
        except Exception:
            pass
    return skill_name


def backfill_from_transcripts(
    brain_dir: Optional[Path] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Scan `~/.gemini/antigravity/brain/*/.system_generated/logs/transcript.jsonl`
    for `read_file` / `view_file` tool calls on `/skills/<name>/SKILL.md`.
    Uses mtime caching so repeat runs only parse modified transcripts in <15ms.
    """
    target_brain = brain_dir or (Path.home() / ".gemini" / "antigravity" / "brain")
    ledger = load_skill_telemetry()
    scanned_map: Dict[str, float] = ledger.setdefault("scanned_transcripts", {})
    skills_map: Dict[str, Any] = ledger.setdefault("skills", {})
    mcp_map: Dict[str, Any] = ledger.setdefault("mcp_servers", {})

    if not target_brain.is_dir():
        return ledger

    dirty = False
    for conv_dir in target_brain.iterdir():
        if not conv_dir.is_dir():
            continue
        t_path = conv_dir / ".system_generated" / "logs" / "transcript.jsonl"
        if not t_path.is_file():
            continue

        conv_id = conv_dir.name
        try:
            mtime = round(t_path.stat().st_mtime, 2)
        except OSError:
            continue

        if not force and scanned_map.get(conv_id) == mtime:
            continue

        dirty = True
        scanned_map[conv_id] = mtime
        try:
            with t_path.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if "SKILL.md" in line and ("read_file" in line or "view_file" in line):
                        for match in SKILL_PATH_RE.finditer(line):
                            s_name = match.group(1)
                            entry = skills_map.setdefault(
                                s_name,
                                {
                                    "total_reads": 0,
                                    "conversations": [],
                                    "first_used_ts": mtime,
                                    "last_used_ts": mtime,
                                    "last_path": match.group(0),
                                },
                            )
                            convs = list(entry.get("conversations", []))
                            entry["total_reads"] = int(entry.get("total_reads", 0)) + 1
                            if conv_id not in convs:
                                convs.append(conv_id)
                                entry["conversations"] = convs
                            entry["first_used_ts"] = min(
                                float(entry.get("first_used_ts", mtime)), mtime
                            )
                            entry["last_used_ts"] = max(
                                float(entry.get("last_used_ts", 0.0)), mtime
                            )
                    if "ServerName" in line or "mcp_" in line:
                        matched_srvs = [m.group(1) for m in MCP_SERVER_NAME_RE.finditer(line)]
                        matched_srvs.extend(m.group(1) for m in MCP_EAGER_TOOL_RE.finditer(line))
                        for srv in matched_srvs:
                            m_entry = mcp_map.setdefault(
                                srv,
                                {
                                    "total_calls": 0,
                                    "conversations": [],
                                    "first_used_ts": mtime,
                                    "last_used_ts": mtime,
                                },
                            )
                            m_convs = list(m_entry.get("conversations", []))
                            m_entry["total_calls"] = int(m_entry.get("total_calls", 0)) + 1
                            if conv_id not in m_convs:
                                m_convs.append(conv_id)
                                m_entry["conversations"] = m_convs
                            m_entry["first_used_ts"] = min(
                                float(m_entry.get("first_used_ts", mtime)), mtime
                            )
                            m_entry["last_used_ts"] = max(
                                float(m_entry.get("last_used_ts", 0.0)), mtime
                            )
        except Exception:
            continue

    if dirty:
        save_skill_telemetry(ledger)
    return ledger


def discover_installed_skills(
    workspace_root: Optional[Path] = None,
    gemini_config_dir: Optional[Path] = None,
    builtin_skills_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Discover all currently installed skills across:
      1. Workspace `.agents/skills/`
      2. Enabled plugins in `~/.gemini/config/plugins/*/skills/`
      3. Built-in skills in `~/.gemini/antigravity/builtin/skills/`
    """
    ws_root = (workspace_root or Path.cwd()).resolve()
    cfg_dir = gemini_config_dir or (Path.home() / ".gemini" / "config")
    b_dir = builtin_skills_dir or (
        Path.home() / ".gemini" / "antigravity" / "builtin" / "skills"
    )
    disabled_plugins = _load_disabled_plugins(cfg_dir)

    discovered: List[Dict[str, Any]] = []

    # 1. Workspace skills
    ws_skills_dir = ws_root / ".agents" / "skills"
    if ws_skills_dir.is_dir():
        for s_dir in sorted(ws_skills_dir.iterdir()):
            s_md = s_dir / "SKILL.md"
            if s_dir.is_dir() and s_md.is_file():
                fm_name, desc, doc_tokens = _parse_skill_frontmatter(s_md)
                prompt_tokens = estimate_prompt_entry_tokens(s_dir.name, str(s_md), desc)
                discovered.append(
                    {
                        "name": s_dir.name,
                        "frontmatter_name": fm_name,
                        "source": "workspace",
                        "source_label": "workspace (.agents/skills)",
                        "plugin_dir": None,
                        "path": str(s_md),
                        "skill_dir": str(s_dir),
                        "description": desc,
                        "prompt_tokens_per_turn": prompt_tokens,
                        "full_doc_tokens": doc_tokens,
                        "is_builtin": False,
                        "is_protected": s_dir.name in PROTECTED_CORE_SKILLS,
                    }
                )

    # 2. Plugin skills
    plugins_root = cfg_dir / "plugins"
    if plugins_root.is_dir():
        for p_dir in sorted(plugins_root.iterdir()):
            if not p_dir.is_dir() or p_dir.name in disabled_plugins:
                continue
            p_json = p_dir / "plugin.json"
            if p_json.is_file():
                try:
                    p_meta = json.loads(p_json.read_text(encoding="utf-8", errors="replace"))
                    if p_meta.get("disabled") is True:
                        continue
                except Exception:
                    pass
            p_skills = p_dir / "skills"
            if not p_skills.is_dir():
                continue
            for s_dir in sorted(p_skills.iterdir()):
                s_md = s_dir / "SKILL.md"
                if s_dir.is_dir() and s_md.is_file():
                    fm_name, desc, doc_tokens = _parse_skill_frontmatter(s_md)
                    prompt_tokens = estimate_prompt_entry_tokens(s_dir.name, str(s_md), desc)
                    discovered.append(
                        {
                            "name": s_dir.name,
                            "frontmatter_name": fm_name,
                            "source": f"plugin:{p_dir.name}",
                            "source_label": p_dir.name,
                            "plugin_dir": p_dir.name,
                            "path": str(s_md),
                            "skill_dir": str(s_dir),
                            "description": desc,
                            "prompt_tokens_per_turn": prompt_tokens,
                            "full_doc_tokens": doc_tokens,
                            "is_builtin": False,
                            "is_protected": s_dir.name in PROTECTED_CORE_SKILLS,
                        }
                    )

    # 3. Built-in skills
    if b_dir.is_dir():
        for s_dir in sorted(b_dir.iterdir()):
            s_md = s_dir / "SKILL.md"
            if s_dir.is_dir() and s_md.is_file():
                fm_name, desc, doc_tokens = _parse_skill_frontmatter(s_md)
                prompt_tokens = estimate_prompt_entry_tokens(fm_name, str(s_md), desc)
                discovered.append(
                    {
                        "name": s_dir.name,
                        "frontmatter_name": fm_name,
                        "source": "builtin",
                        "source_label": "builtin (antigravity)",
                        "plugin_dir": None,
                        "path": str(s_md),
                        "skill_dir": str(s_dir),
                        "description": desc,
                        "prompt_tokens_per_turn": prompt_tokens,
                        "full_doc_tokens": doc_tokens,
                        "is_builtin": True,
                        "is_protected": True,
                    }
                )

    return discovered


def list_archived_skills(archive_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """List all skills currently stored in the reversible Cold-Storage Archive."""
    arch_root = archive_dir or get_skills_archive_dir()
    if not arch_root.is_dir():
        return []

    archived: List[Dict[str, Any]] = []
    for s_dir in sorted(arch_root.iterdir()):
        if not s_dir.is_dir():
            continue
        meta_path = s_dir / "archive_meta.json"
        meta: Dict[str, Any] = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                meta = {}
        s_md = s_dir / "SKILL.md"
        _, desc, doc_tokens = (
            _parse_skill_frontmatter(s_md) if s_md.is_file() else (s_dir.name, "", 0)
        )
        archived.append(
            {
                "name": s_dir.name,
                "archive_path": str(s_dir),
                "original_skill_dir": meta.get("original_skill_dir", ""),
                "source_label": meta.get("source_label", "unknown"),
                "archived_at": meta.get("archived_at", 0.0),
                "prompt_tokens_saved_per_turn": meta.get(
                    "prompt_tokens_per_turn",
                    estimate_prompt_entry_tokens(s_dir.name, str(s_md), desc),
                ),
                "full_doc_tokens": doc_tokens,
                "reason": meta.get("reason", "dormant_zero_use"),
            }
        )
    return archived


def audit_skills(
    workspace_root: Optional[Path] = None,
    gemini_config_dir: Optional[Path] = None,
    builtin_skills_dir: Optional[Path] = None,
    brain_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run a full Skill Telemetry & Token-Tax Audit across all installed skills,
    combining Organ 05 live hook telemetry with historical transcript backfill.
    """
    ledger = backfill_from_transcripts(brain_dir=brain_dir)
    skills_usage = ledger.get("skills", {})
    installed = discover_installed_skills(
        workspace_root=workspace_root,
        gemini_config_dir=gemini_config_dir,
        builtin_skills_dir=builtin_skills_dir,
    )
    archived = list_archived_skills()
    now = time.time()

    enriched: List[Dict[str, Any]] = []
    name_counts: Dict[str, int] = {}
    for item in installed:
        name_counts[item["name"]] = name_counts.get(item["name"], 0) + 1

    total_prompt_tokens = 0
    reclaimable_tokens = 0
    dormant_count = 0
    active_count = 0
    occasional_count = 0

    for item in installed:
        s_name = item["name"]
        fm_name = item["frontmatter_name"]
        u = skills_usage.get(s_name) or skills_usage.get(fm_name) or {}
        convs = u.get("conversations", [])
        conv_count = len(convs)
        total_reads = int(u.get("total_reads", 0))
        last_ts = float(u.get("last_used_ts", 0.0))
        days_since = (
            round((now - last_ts) / 86400.0, 1) if last_ts > 0 else None
        )
        last_date = (
            datetime.datetime.fromtimestamp(last_ts).strftime("%Y-%m-%d")
            if last_ts > 0
            else "never"
        )

        if item["is_builtin"]:
            tier = "BUILTIN"
        elif conv_count >= 2:
            tier = "ACTIVE"
            active_count += 1
        elif conv_count == 1:
            tier = "OCCASIONAL"
            occasional_count += 1
        else:
            tier = "DORMANT"
            dormant_count += 1
            if not item["is_protected"]:
                reclaimable_tokens += item["prompt_tokens_per_turn"]

        total_prompt_tokens += item["prompt_tokens_per_turn"]

        enriched.append(
            {
                **item,
                "conversations_count": conv_count,
                "total_reads": total_reads,
                "last_used_ts": last_ts,
                "last_used_date": last_date,
                "days_since_use": days_since,
                "tier": tier,
                "is_duplicate": name_counts.get(s_name, 0) > 1,
                "prune_candidate": (
                    tier == "DORMANT"
                    and not item["is_builtin"]
                    and not item["is_protected"]
                ),
            }
        )

    # Sort: highest conversations_count first, then total_reads, then name
    enriched.sort(
        key=lambda x: (
            -x["conversations_count"],
            -x["total_reads"],
            x["is_builtin"],
            x["name"],
        )
    )

    archived_tokens_saved = sum(
        int(a.get("prompt_tokens_saved_per_turn", 0)) for a in archived
    )

    mcp_audit = audit_mcp_servers(
        workspace_root=workspace_root,
        home_dir=gemini_config_dir.parent.parent if gemini_config_dir else None,
        brain_dir=brain_dir,
        ledger=ledger,
    )

    return {
        "timestamp": now,
        "total_conversations_scanned": len(ledger.get("scanned_transcripts", {})),
        "total_installed_skills": len(enriched),
        "active_skills_count": active_count,
        "occasional_skills_count": occasional_count,
        "dormant_skills_count": dormant_count,
        "duplicate_skills_count": sum(1 for x in enriched if x["is_duplicate"]),
        "per_turn_prompt_tokens": total_prompt_tokens,
        "reclaimable_tokens_per_turn": reclaimable_tokens,
        "estimated_100_turn_token_savings": reclaimable_tokens * 100,
        "archived_skills_count": len(archived),
        "archived_tokens_saved_per_turn": archived_tokens_saved,
        "skills": enriched,
        "archived_skills": archived,
        "mcp_audit": mcp_audit,
    }


def _canonical_mcp_id(raw_name: str) -> str:
    norm = re.sub(r"[^a-z0-9]+", "", str(raw_name or "").lower())
    for suffix in ("mcpserver", "server", "mcp"):
        if norm.endswith(suffix) and len(norm) > len(suffix):
            norm = norm[: -len(suffix)]
            break
    return norm


def list_archived_mcp_servers(archive_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """List all MCP servers currently stored in the reversible Cold-Storage MCP Archive."""
    arch_root = archive_dir or get_mcp_archive_dir()
    if not arch_root.is_dir():
        return []

    archived: List[Dict[str, Any]] = []
    for m_dir in sorted(arch_root.iterdir()):
        if not m_dir.is_dir():
            continue
        meta_path = m_dir / "archive_meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        tok = int(meta.get("tokens", 1800))
        archived.append(
            {
                "name": meta.get("name", m_dir.name),
                "archive_path": str(m_dir),
                "source": meta.get("source", "unknown"),
                "tokens": tok,
                "quota_burn_25_turns": int(round(tok * QUOTA_25_TURN_MULTIPLIER)),
                "archived_at": float(meta.get("archived_at", 0.0)),
                "disabled_in_configs": meta.get("disabled_in_configs", []),
                "original_schema_dir": meta.get("original_schema_dir", ""),
            }
        )
    return archived


def audit_mcp_servers(
    workspace_root: Optional[Path] = None,
    home_dir: Optional[Path] = None,
    brain_dir: Optional[Path] = None,
    ledger: Optional[Dict[str, Any]] = None,
    archive_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Audit active MCP servers against empirical transcript tool-call usage and quantify
    their Turn-0 schema tax + 5-hour rolling quota burn (5.95x over a 25-turn session).
    """
    from whole_body_compute.finops_audit import audit_turn0_static_overhead

    ws_root = (workspace_root or Path.cwd()).resolve()
    h_dir = (home_dir or Path.home()).resolve()
    active_ledger = ledger or backfill_from_transcripts(brain_dir=brain_dir)
    mcp_usage_map = active_ledger.get("mcp_servers", {})
    t0 = audit_turn0_static_overhead(workspace_dir=ws_root, home_dir=h_dir)

    canon_usage: Dict[str, Dict[str, Any]] = {}
    for k, u in mcp_usage_map.items():
        if not isinstance(u, dict):
            continue
        ck = _canonical_mcp_id(k)
        prev = canon_usage.get(ck)
        if prev is None:
            canon_usage[ck] = {
                "total_calls": int(u.get("total_calls", 0)),
                "conversations": list(u.get("conversations", [])),
                "last_used_ts": float(u.get("last_used_ts", 0.0)),
            }
        else:
            prev["total_calls"] += int(u.get("total_calls", 0))
            for cid in u.get("conversations", []):
                if cid not in prev["conversations"]:
                    prev["conversations"].append(cid)
            prev["last_used_ts"] = max(prev["last_used_ts"], float(u.get("last_used_ts", 0.0)))

    servers_enriched: List[Dict[str, Any]] = []
    reclaimable_turn0_tokens = 0
    reclaimable_quota_25t = 0
    dormant_mcp_count = 0

    for srv in t0.get("mcp_servers", []):
        s_name = srv["name"]
        tok = int(srv.get("tokens", 0))
        quota_25t = int(round(tok * QUOTA_25_TURN_MULTIPLIER))
        u = mcp_usage_map.get(s_name) or canon_usage.get(_canonical_mcp_id(s_name)) or {}
        convs = u.get("conversations", [])
        conv_count = len(convs)
        total_calls = int(u.get("total_calls", 0))
        last_ts = float(u.get("last_used_ts", 0.0))
        last_date = (
            datetime.datetime.fromtimestamp(last_ts).strftime("%Y-%m-%d")
            if last_ts > 0
            else "never"
        )
        if conv_count >= 2:
            tier = "ACTIVE"
        elif conv_count == 1:
            tier = "OCCASIONAL"
        else:
            tier = "DORMANT"
            dormant_mcp_count += 1
            reclaimable_turn0_tokens += tok
            reclaimable_quota_25t += quota_25t

        servers_enriched.append(
            {
                **srv,
                "conversations_count": conv_count,
                "total_calls": total_calls,
                "last_used_ts": last_ts,
                "last_used_date": last_date,
                "tier": tier,
                "quota_burn_25_turns": quota_25t,
                "prune_candidate": tier == "DORMANT",
            }
        )

    archived_mcp = list_archived_mcp_servers(archive_dir=archive_dir)
    archived_turn0_saved = sum(int(a.get("tokens", 0)) for a in archived_mcp)
    archived_quota_saved = sum(int(a.get("quota_burn_25_turns", 0)) for a in archived_mcp)
    total_turn0_mcp = int(t0.get("mcp_turn0_tokens", 0))
    total_quota_25t = int(round(total_turn0_mcp * QUOTA_25_TURN_MULTIPLIER))

    return {
        "mcp_servers_count": len(servers_enriched),
        "dormant_mcp_count": dormant_mcp_count,
        "mcp_turn0_tokens": total_turn0_mcp,
        "mcp_quota_burn_25_turns": total_quota_25t,
        "reclaimable_turn0_tokens": reclaimable_turn0_tokens,
        "reclaimable_quota_burn_25_turns": reclaimable_quota_25t,
        "archived_mcp_count": len(archived_mcp),
        "archived_turn0_tokens_saved": archived_turn0_saved,
        "archived_quota_burn_saved_25_turns": archived_quota_saved,
        "servers": servers_enriched,
        "archived_servers": archived_mcp,
    }


def prune_mcp_servers(
    server_names: Optional[List[str]] = None,
    max_conversations: int = 0,
    dry_run: bool = False,
    workspace_root: Optional[Path] = None,
    home_dir: Optional[Path] = None,
    archive_dir: Optional[Path] = None,
    brain_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Shelve dormant (`<= max_conversations` empirical calls) or explicitly named MCP servers.
    """
    ws_root = (workspace_root or Path.cwd()).resolve()
    h_dir = (home_dir or Path.home()).resolve()
    arch_root = archive_dir or get_mcp_archive_dir()
    mcp_report = audit_mcp_servers(
        workspace_root=ws_root,
        home_dir=h_dir,
        brain_dir=brain_dir,
        archive_dir=arch_root,
    )
    explicit_set = set(server_names or [])

    cfg_candidates = [
        ws_root / ".mcp.json",
        ws_root / ".cursor" / "mcp.json",
        ws_root / ".vscode" / "mcp.json",
        ws_root / ".gemini" / "mcp_config.json",
        ws_root / ".agents" / "mcp_config.json",
        h_dir / ".claude.json",
        h_dir / ".claude" / "mcp.json",
        h_dir / ".claude" / "claude_desktop_config.json",
        h_dir / ".cursor" / "mcp.json",
        h_dir / ".gemini" / "antigravity" / "mcp_config.json",
        h_dir / ".gemini" / "settings.json",
        h_dir / ".codeium" / "windsurf" / "mcp_config.json",
    ]
    schema_roots = [
        h_dir / ".gemini" / "antigravity" / "mcp",
        ws_root / ".mcp-schemas",
    ]

    shelved: List[Dict[str, Any]] = []
    turn0_saved = 0
    quota_25t_saved = 0

    for srv in mcp_report["servers"]:
        s_name = srv["name"]
        if explicit_set:
            if s_name not in explicit_set:
                continue
        else:
            if srv["conversations_count"] > max_conversations:
                continue

        disabled_configs: List[str] = []
        target_canon = _canonical_mcp_id(s_name)

        for cfg_p in cfg_candidates:
            if not cfg_p.is_file():
                continue
            try:
                raw_data = json.loads(cfg_p.read_text(encoding="utf-8", errors="ignore"))
                if not isinstance(raw_data, dict):
                    continue
                modified = False
                for key_field in ("mcpServers", "servers"):
                    srvs_obj = raw_data.get(key_field)
                    if isinstance(srvs_obj, dict):
                        for raw_k, srv_cfg in list(srvs_obj.items()):
                            if (
                                raw_k == s_name
                                or _canonical_mcp_id(raw_k) == target_canon
                            ) and isinstance(srv_cfg, dict):
                                if srv_cfg.get("disabled") is not True:
                                    srv_cfg["disabled"] = True
                                    modified = True
                                    disabled_configs.append(f"{cfg_p}::{key_field}::{raw_k}")
                if modified and not dry_run:
                    cfg_p.write_text(json.dumps(raw_data, indent=2) + "\n", encoding="utf-8")
            except Exception:
                continue

        orig_schema_dir = ""
        dest_dir = arch_root / s_name
        for s_root in schema_roots:
            if not s_root.is_dir():
                continue
            for cand_dir in sorted(s_root.iterdir()):
                if not cand_dir.is_dir():
                    continue
                if cand_dir.name == s_name or _canonical_mcp_id(cand_dir.name) == target_canon:
                    if not orig_schema_dir:
                        orig_schema_dir = str(cand_dir)
                    if not dry_run:
                        arch_root.mkdir(parents=True, exist_ok=True)
                        if dest_dir.exists():
                            shutil.rmtree(dest_dir)
                        shutil.move(str(cand_dir), str(dest_dir))

        meta = {
            "name": s_name,
            "source": srv.get("source", "unknown"),
            "tokens": int(srv.get("tokens", 0)),
            "quota_burn_25_turns": int(srv.get("quota_burn_25_turns", 0)),
            "conversations_count": int(srv.get("conversations_count", 0)),
            "total_calls": int(srv.get("total_calls", 0)),
            "disabled_in_configs": disabled_configs,
            "original_schema_dir": orig_schema_dir,
            "archived_at": time.time(),
        }
        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            (dest_dir / "archive_meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )

        shelved.append(meta)
        turn0_saved += meta["tokens"]
        quota_25t_saved += meta["quota_burn_25_turns"]

    if shelved and not dry_run:
        try:
            from whole_body_compute.procedural_telemetry import track_wbc_event

            track_wbc_event(
                "wbc_mcp_pruned",
                {
                    "shelved_count": len(shelved),
                    "turn0_tokens_saved_per_session": turn0_saved,
                    "quota_burn_saved_per_25_turn_session": quota_25t_saved,
                    "max_conversations_threshold": max_conversations,
                },
            )
        except Exception:
            pass

    return {
        "dry_run": dry_run,
        "archive_dir": str(arch_root),
        "shelved_count": len(shelved),
        "turn0_tokens_saved_per_session": turn0_saved,
        "quota_burn_saved_per_25_turn_session": quota_25t_saved,
        "shelved": shelved,
    }


def restore_mcp_server(
    server_name: str,
    archive_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Restore an archived MCP server from `~/.gemini/config/mcp_archive/<server_name>`.
    """
    arch_root = archive_dir or get_mcp_archive_dir()
    if server_name == "all":
        restored_list = []
        for entry in list_archived_mcp_servers(archive_dir=arch_root):
            res = restore_mcp_server(entry["name"], archive_dir=arch_root)
            if res.get("restored"):
                restored_list.append(res)
        return {
            "restored": len(restored_list) > 0,
            "restored_count": len(restored_list),
            "items": restored_list,
        }

    src_dir = arch_root / server_name
    meta_path = src_dir / "archive_meta.json"
    if not src_dir.is_dir() or not meta_path.is_file():
        return {
            "restored": False,
            "server_name": server_name,
            "error": f"MCP server '{server_name}' not found in archive ({arch_root})",
        }

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        return {"restored": False, "server_name": server_name, "error": str(exc)}

    for spec in meta.get("disabled_in_configs", []):
        parts = str(spec).split("::", 2)
        if len(parts) != 3:
            continue
        cfg_p = Path(parts[0])
        key_field, raw_k = parts[1], parts[2]
        if not cfg_p.is_file():
            continue
        try:
            raw_data = json.loads(cfg_p.read_text(encoding="utf-8", errors="ignore"))
            srvs_obj = raw_data.get(key_field)
            if isinstance(srvs_obj, dict) and isinstance(srvs_obj.get(raw_k), dict):
                srvs_obj[raw_k].pop("disabled", None)
                cfg_p.write_text(json.dumps(raw_data, indent=2) + "\n", encoding="utf-8")
        except Exception:
            continue

    orig_schema_str = str(meta.get("original_schema_dir") or "")
    meta_path.unlink(missing_ok=True)
    if orig_schema_str:
        dest_dir = Path(orig_schema_str)
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.move(str(src_dir), str(dest_dir))
    else:
        shutil.rmtree(src_dir, ignore_errors=True)

    return {
        "restored": True,
        "server_name": server_name,
        "restored_schema_dir": orig_schema_str,
        "reenabled_configs": meta.get("disabled_in_configs", []),
    }


def prune_skills(
    skill_names: Optional[List[str]] = None,
    max_conversations: int = 0,
    include_protected: bool = False,
    dry_run: bool = False,
    workspace_root: Optional[Path] = None,
    gemini_config_dir: Optional[Path] = None,
    archive_dir: Optional[Path] = None,
    brain_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Move dormant (`<= max_conversations` activations) or explicitly named skills into
    the reversible Cold-Storage Archive (`~/.gemini/config/skills_archive/`).
    """
    report = audit_skills(
        workspace_root=workspace_root,
        gemini_config_dir=gemini_config_dir,
        brain_dir=brain_dir,
    )
    arch_root = archive_dir or get_skills_archive_dir()
    explicit_set = set(skill_names or [])

    shelved: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    tokens_saved_per_turn = 0

    for item in report["skills"]:
        if item["is_builtin"]:
            continue

        s_name = item["name"]
        if explicit_set:
            if s_name not in explicit_set:
                continue
        else:
            if item["conversations_count"] > max_conversations:
                continue
            if item["is_protected"] and not include_protected:
                skipped.append(
                    {
                        "name": s_name,
                        "source_label": item["source_label"],
                        "reason": "protected_core_skill",
                    }
                )
                continue

        src_dir = Path(item["skill_dir"])
        if not src_dir.is_dir():
            continue

        dest_dir = arch_root / s_name
        meta = {
            "name": s_name,
            "original_skill_dir": str(src_dir),
            "original_skill_md": item["path"],
            "source": item["source"],
            "source_label": item["source_label"],
            "plugin_dir": item["plugin_dir"],
            "prompt_tokens_per_turn": item["prompt_tokens_per_turn"],
            "conversations_count": item["conversations_count"],
            "archived_at": time.time(),
            "reason": "explicit_prune" if explicit_set else f"dormant_<={max_conversations}_convs",
        }

        if not dry_run:
            arch_root.mkdir(parents=True, exist_ok=True)
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            shutil.move(str(src_dir), str(dest_dir))
            (dest_dir / "archive_meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )

        shelved.append(meta)
        tokens_saved_per_turn += int(item["prompt_tokens_per_turn"])

    if shelved and not dry_run:
        try:
            from whole_body_compute.procedural_telemetry import track_wbc_event

            track_wbc_event(
                "wbc_skills_pruned",
                {
                    "shelved_count": len(shelved),
                    "tokens_saved_per_turn": tokens_saved_per_turn,
                    "estimated_100_turn_token_savings": tokens_saved_per_turn * 100,
                    "max_conversations_threshold": max_conversations,
                },
            )
        except Exception:
            pass

    return {
        "dry_run": dry_run,
        "archive_dir": str(arch_root),
        "shelved_count": len(shelved),
        "tokens_saved_per_turn": tokens_saved_per_turn,
        "estimated_100_turn_token_savings": tokens_saved_per_turn * 100,
        "shelved": shelved,
        "skipped_protected": skipped,
    }


def restore_skill(
    skill_name: str,
    target_dir: Optional[Path] = None,
    archive_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Restore an archived skill from `~/.gemini/config/skills_archive/<skill_name>`
    back to its original plugin or workspace directory.
    Pass `skill_name="all"` to restore all archived skills.
    """
    arch_root = archive_dir or get_skills_archive_dir()
    if skill_name == "all":
        restored_list = []
        for entry in list_archived_skills(archive_dir=arch_root):
            res = restore_skill(entry["name"], archive_dir=arch_root)
            if res.get("restored"):
                restored_list.append(res)
        return {
            "restored": len(restored_list) > 0,
            "restored_count": len(restored_list),
            "items": restored_list,
        }

    src_dir = arch_root / skill_name
    if not src_dir.is_dir():
        return {
            "restored": False,
            "skill_name": skill_name,
            "error": f"Skill '{skill_name}' not found in archive ({arch_root})",
        }

    meta_path = src_dir / "archive_meta.json"
    orig_dir_str = ""
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8", errors="replace"))
            orig_dir_str = meta.get("original_skill_dir", "")
        except Exception:
            pass

    dest_dir = target_dir or (
        Path(orig_dir_str)
        if orig_dir_str
        else (Path.cwd() / ".agents" / "skills" / skill_name)
    )
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    if dest_dir.exists():
        shutil.rmtree(dest_dir)

    if meta_path.is_file():
        try:
            meta_path.unlink()
        except OSError:
            pass

    shutil.move(str(src_dir), str(dest_dir))
    return {
        "restored": True,
        "skill_name": skill_name,
        "restored_to": str(dest_dir),
    }


def format_skills_audit_table(report: Dict[str, Any], show_all: bool = False) -> str:
    """Format a human-readable terminal table for `body skills`."""
    lines = [
        "\n🧬 [Whole Body Compute: Skill Telemetry, Token-Tax Audit & Cold-Storage Shelf]",
        "=" * 94,
        (
            f"  • Conversations Audited: {report['total_conversations_scanned']} transcripts "
            f"+ live Organ 05 hook ledger"
        ),
        (
            f"  • Installed Skills:      {report['total_installed_skills']} total "
            f"({report['active_skills_count']} active, {report['occasional_skills_count']} occasional, "
            f"{report['dormant_skills_count']} dormant 0-use)"
        ),
        (
            f"  • Prompt Token Tax:      ~{report['per_turn_prompt_tokens']:,} tokens/turn "
            f"(~{report['per_turn_prompt_tokens'] * 100:,} tokens per 100 turns)"
        ),
        (
            f"  • Reclaimable Now:       ~{report['reclaimable_tokens_per_turn']:,} tokens/turn "
            f"by shelving 0-use skills (`body skills prune`)"
        ),
        (
            f"  • Cold-Storage Shelf:    {report['archived_skills_count']} archived skills "
            f"(saving ~{report['archived_tokens_saved_per_turn']:,} tokens/turn; `body skills restore <name>`)"
        ),
        "-" * 94,
        f"  {'SKILL NAME':<26} {'SOURCE':<24} {'TIER':<10} {'CONVS':>5} {'READS':>5} {'LAST USED':>10} {'TAX/TURN':>9}",
        "-" * 94,
    ]

    for s in report["skills"]:
        if s["is_builtin"] and not show_all:
            continue
        tier_badge = {
            "ACTIVE": "🟢 ACTIVE",
            "OCCASIONAL": "🟡 1-HIT",
            "DORMANT": "⚪ 0-USE",
            "BUILTIN": "🔵 BUILTIN",
        }.get(s["tier"], s["tier"])
        prot = " 🔒" if s["is_protected"] and s["tier"] == "DORMANT" else ""
        src_short = s["source_label"][:23]
        lines.append(
            f"  {s['name'][:25]:<26} {src_short:<24} {tier_badge:<10} "
            f"{s['conversations_count']:>5} {s['total_reads']:>5} {s['last_used_date']:>10} "
            f"~{s['prompt_tokens_per_turn']:>4} tok{prot}"
        )

    if report.get("archived_skills"):
        lines.append("-" * 94)
        lines.append("  📦 Cold-Storage Shelf (`~/.gemini/config/skills_archive/` - restore via `body skills restore <name>`):")
        arch_names = [
            f"{a['name']} (~{a['prompt_tokens_saved_per_turn']} tok/turn)"
            for a in report["archived_skills"]
        ]
        lines.append("     " + ", ".join(arch_names))

    mcp_audit = report.get("mcp_audit") or {}
    mcp_srvs = mcp_audit.get("servers") or []
    mcp_arch = mcp_audit.get("archived_servers") or []
    lines.append("-" * 94)
    lines.append(
        f"  🔌 MCP Server Schema & 5-Hour Quota Audit: {mcp_audit.get('mcp_servers_count', 0)} active "
        f"({mcp_audit.get('mcp_turn0_tokens', 0):,} Turn-0 tok | "
        f"~{mcp_audit.get('mcp_quota_burn_25_turns', 0):,} 5h-quota tok/25-turn session)"
    )
    if mcp_srvs:
        if mcp_audit.get("reclaimable_turn0_tokens", 0) > 0:
            lines.append(
                f"     ⚠️  Reclaimable 0-Use MCP Tax: {mcp_audit['reclaimable_turn0_tokens']:,} Turn-0 tok "
                f"(~{mcp_audit['reclaimable_quota_burn_25_turns']:,} 5h-quota tok/session) -> run `body skills prune-mcp`"
            )
        for ms in mcp_srvs:
            m_badge = {
                "ACTIVE": "🟢 ACTIVE",
                "OCCASIONAL": "🟡 1-HIT",
                "DORMANT": "⚪ 0-USE",
            }.get(ms["tier"], ms["tier"])
            lines.append(
                f"     • {ms['name'][:24]:<24} {m_badge:<10} "
                f"calls={ms['total_calls']:<4} turn0={ms['tokens']:>6,} tok  "
                f"25t_quota=~{ms['quota_burn_25_turns']:>7,} tok  ({ms['source']})"
            )
    else:
        lines.append(
            "     🟢 0-MCP LEAN MODE: 0 MCP schema tokens/session (~3.4x longer 5-hour quota runway preserved)"
        )
    if mcp_arch:
        arch_mcp_names = [
            f"{a['name']} ({a['tokens']:,} tok / ~{a['quota_burn_25_turns']:,} 25t-quota)"
            for a in mcp_arch
        ]
        lines.append(
            "     📦 Shelved MCP Servers (`body skills restore-mcp <name>`): " + ", ".join(arch_mcp_names)
        )

    lines.append("=" * 94 + "\n")
    return "\n".join(lines)
