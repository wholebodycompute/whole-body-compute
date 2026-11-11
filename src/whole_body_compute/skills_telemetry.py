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
                    if "SKILL.md" not in line or (
                        "read_file" not in line and "view_file" not in line
                    ):
                        continue
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
        lines.append("  📦 Cold-Storage Shelf (`~/.gemini/config/skills_archive/` — restore via `body skills restore <name>`):")
        arch_names = [
            f"{a['name']} (~{a['prompt_tokens_saved_per_turn']} tok/turn)"
            for a in report["archived_skills"]
        ]
        lines.append("     " + ", ".join(arch_names))

    lines.append("=" * 94 + "\n")
    return "\n".join(lines)
