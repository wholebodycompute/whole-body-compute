"""
Sovereign Relationship Memory ("People Hippocampus" - Organ 03 Eyes -> Organ 04 Hippocampus).

Distills historical social/comment archives (`.jsonl`, `.csv`, `.zip` exports from Facebook, LinkedIn,
X/Twitter, or GitHub) and live browser engagements (Organ 03 `FovealVisionEngine` AXUIElement DOM)
through a 5-Gate Signal-vs-Nonsense filter into Organ 04 `~/.whole_body_compute/dot_memory.db`
(`relationship_threads` + `relationship_events`):

  - Gate 1 (Action Trigger): Drops self-comments (`his own post/comment` or `owner_name`) and bare URLs;
    in live browser mode (`Organ 03`), only captures active outbound replies/DMs (`>= 12 words`).
  - Gate 2 (Substance Floor): Drops `< 15`-word drive-by reactions ("Thank you!", "Beautiful!").
  - Gate 3 (Peer Identity): Resolves named human peers from reply headers and leading vocatives while
    filtering out brand/company/celebrity pages (`Cursor`, `Fireflies.ai`, `*Community`).
  - Gate 4 (Warmth & Sparring Classification): Distinguishes warm builder exchanges, philosophical
    sparring partners (`sparring_partner`), and quarantined hostility (`quarantined`), while honoring
    explicit `--allow` / `--quarantine` user overrides.
  - Gate 5 (Circle Separation): Separates `tech_builder`, `personal_sacred`, `sparring_partner`, and any
    user-defined `custom_circles` in `~/.whole_body_compute/people_config.json`.

Tier Packaging:
  - Community ($0 Forever): Full 5-Gate archive import (`--import` / `--sync`), Top 5 results per query
    (`--limit 5`), and 3/day contextual DM opener copies (`--copy`).
  - Pro ($29/mo): Unlimited `--limit`, unlimited `--copy`, custom circles/CTAs, and Organ 03 Live
    Browser DOM Capture (`--capture`).
"""

from __future__ import annotations

import csv
import datetime
import io
import json
import os
import re
import sqlite3
import subprocess
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from whole_body_compute.local.dot_memory import _resolve_default_dot_db_path
from whole_body_compute.procedural_telemetry import get_wbc_data_dir
from whole_body_compute.redaction import redact_secrets

COMMUNITY_MAX_QUERY_LIMIT = 5
COMMUNITY_DAILY_COPY_LIMIT = 3

BRAND_OR_PAGE_NAMES = {
    "Cursor",
    "Fireflies.ai",
    "Tim Ferriss",
    "AITX Community",
    "Breakwater Strategy",
    "Google",
    "OpenAI",
    "Anthropic",
    "Meta",
    "Apple",
    "Microsoft",
}

NON_PERSON_FIRST_WORDS = {
    "Beautiful", "Unfortunately", "Indeed", "Definitely", "Yeah", "Yes", "No", "Sure",
    "Sweet", "Both", "Nice", "Super", "Awesome", "Congrats", "Love", "Wait", "Here",
    "Behind", "My", "The", "This", "That", "These", "Those", "What", "When", "Where",
    "Why", "How", "For", "In", "On", "At", "To", "From", "With", "Without", "If",
    "But", "And", "Or", "So", "Because", "Although", "While", "Until", "Since", "After",
    "Before", "Today", "Yesterday", "Tomorrow", "Currently", "FWIW", "Idk", "Hmm", "Oh",
    "Well", "Maybe", "Perhaps", "Honestly", "Literally", "Basically", "Actually",
    "Especially", "Warning", "Note", "Update", "Check", "Try", "Run", "Stop", "Start",
    "Build", "Building", "Saves", "Add", "Taking", "Talk", "Go", "Way", "Thank",
    "Thanks", "Appreciate", "Also", "Even", "Still", "Just", "Only", "Always", "Never",
}

HOSTILE_FRICTION_RE = re.compile(
    r"\bbelow\s+surface\s+level\b|\bfile\s+a\s+complaint\b|\bleave\s+children\s+out\b",
    re.IGNORECASE,
)

SPARRING_DEBATE_RE = re.compile(
    r"\bgaslight|\bprojecting\b|\bdeflected\b|\bpersonal\s+[\"']?attack|\bmalnutrition\b|\bworld\s+hunger\b",
    re.IGNORECASE,
)

WARMTH_RE = re.compile(
    r"\bthanks\s+for\s+asking\b|\bappreciate\s+(?:that|you|your)\b|\bsuper\s+helpful\b|"
    r"\bhmu\b|\blet\s+ya\s+know\b|\blmk\b|\bhappy\s+to\s+consider\b|\bawesome\b|"
    r"\binspiring\b|\bsaves\s+so\s+much\s+time\b|\blove\s+it\b|\bcongrats\b|"
    r"\bgood\s+sir\b|\byou\s+got\s+that\s+right\b|\byeah,\s+this\b",
    re.IGNORECASE,
)

KINSHIP_SACRED_OVERRIDE_RE = re.compile(
    r"\bthank\s+you\s+(?:sister|brother)\b|\bi\s+love\s+you\b",
    re.IGNORECASE,
)

SACRED_POETRY_RE = re.compile(
    r"\bpoem\b|\bgrief\b|\bcrying\b|\bprayer\b|\byoga\b|\btenderness\b|\bsoftness\b|\bhealing\b|\bnon-duality\b",
    re.IGNORECASE,
)

TECH_RE = re.compile(
    r"\bclaude\b|\bgemini\b|\bantigravity\b|\bagy\b|\bcursor\b|\blocal\s+model|"
    r"\blocal\s+llm|\blocal\s+stack\b|\blocal\s+compute\b|\bqwen\b|\bgemma\b|\btev\b|"
    r"\bmcp\b|\bharness|\bbenchmark|\bvifi\b|\bvoicefi\b|\bvi-fi\b|\bdavinci\b|"
    r"\bwhisper\b|\bopen\s+source\b|\bcode\b|\bcoding\b|\bagent|\btoken|\bpipeline\b|"
    r"\bqa\b|\bsoftware\b|\bpm\b|\bdata\s+business\b|\blienlogic\b|\bollama\b|\bmlx\b|\bm5\s+pro\b",
    re.IGNORECASE,
)

TOPIC_TAG_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("local-models", re.compile(r"\blocal\s+model|\blocal\s+llm|\blocal\s+compute|\blocal\s+stack|\bqwen|\bgemma|\btev|\bollama|\bmlx", re.I)),
    ("harnesses", re.compile(r"\bharness|\bclaude|\bantigravity|\bagy|\bcursor", re.I)),
    ("mcp-deterministic", re.compile(r"\bmcp\b|\bdavinci\b|\bdeterministic", re.I)),
    ("voicefi", re.compile(r"\bvifi\b|\bvoicefi\b|\bvi-fi\b|\bvoice\s+layer|\btalk\s+with", re.I)),
    ("audit-qa", re.compile(r"\baudit|\b255\s+files|\b90%\s+token|\bqa\b|\bdata\s+pipeline|\blienlogic", re.I)),
    ("software-pm", re.compile(r"\bsoftware|\bproject\s+management|\bpm\b|\bpipeline\b", re.I)),
)


def get_people_config_path() -> Path:
    """Resolve user-customizable `people_config.json` path."""
    env_p = os.environ.get("WBC_PEOPLE_CONFIG_PATH")
    if env_p:
        return Path(env_p).expanduser()
    ws_cfg = Path.cwd() / ".agents" / "people_config.json"
    if ws_cfg.is_file():
        return ws_cfg
    return (get_wbc_data_dir() / "people_config.json").expanduser()


def _detect_git_or_os_owner_name() -> str:
    """Auto-detect the machine owner's full name for Gate 1 self-comment filtering."""
    try:
        proc = subprocess.run(
            ["git", "config", "user.name"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        val = (proc.stdout or "").strip()
        if val and " " in val:
            return val
    except Exception:
        pass
    try:
        import pwd

        gecos = pwd.getpwuid(os.getuid()).pw_gecos.split(",")[0].strip()
        if gecos and " " in gecos:
            return gecos
    except Exception:
        pass
    return ""


def load_people_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load user-customizable relationship memory settings (`people_config.json`)."""
    cfg_p = Path(config_path or get_people_config_path()).expanduser()
    defaults: Dict[str, Any] = {
        "owner_name": "auto",
        "archive_path": "",
        "cooldown_days": 30,
        "default_cta": "curl -sSL https://wholebodycompute.com/audit/dm | python3",
        "custom_circles": {},
        "custom_topic_keywords": {},
    }
    if cfg_p.is_file():
        try:
            loaded = json.loads(cfg_p.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                defaults.update(loaded)
        except Exception:
            pass
    if not defaults.get("owner_name") or defaults["owner_name"] == "auto":
        defaults["owner_name"] = _detect_git_or_os_owner_name()
    return defaults


def save_people_config(updates: Dict[str, Any], config_path: Optional[Path] = None) -> Path:
    """Save or update `~/.whole_body_compute/people_config.json` with `0600` permissions."""
    cfg_p = Path(config_path or get_people_config_path()).expanduser()
    curr = load_people_config(cfg_p)
    curr.update({k: v for k, v in updates.items() if v is not None})
    cfg_p.parent.mkdir(parents=True, exist_ok=True)
    cfg_p.write_text(json.dumps(curr, indent=2) + "\n", encoding="utf-8")
    try:
        os.chmod(cfg_p, 0o600)
    except OSError:
        pass
    return cfg_p


def _is_pro_or_higher() -> bool:
    """Check whether active license tier is Pro, Team, or Enterprise."""
    try:
        from whole_body_compute.license import LicenseManager

        tier = LicenseManager().get_active_tier()
        return tier in ("pro", "team", "enterprise")
    except Exception:
        return False


def _normalize_person_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def ensure_relationship_tables(conn: sqlite3.Connection) -> None:
    """Create `relationship_threads` and `relationship_events` tables in Organ 04 `dot_memory.db`."""
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS relationship_threads (
            person_key TEXT PRIMARY KEY,
            person_name TEXT NOT NULL,
            circle TEXT NOT NULL DEFAULT 'tech_builder',
            status TEXT NOT NULL DEFAULT 'active',
            score INTEGER NOT NULL DEFAULT 0,
            interaction_count INTEGER NOT NULL DEFAULT 0,
            direct_replies INTEGER NOT NULL DEFAULT 0,
            warmth_hits INTEGER NOT NULL DEFAULT 0,
            first_seen_at TEXT,
            last_interaction_at TEXT,
            last_outbound_dm_at TEXT,
            outbound_note TEXT DEFAULT '',
            surfaces_json TEXT NOT NULL DEFAULT '[]',
            topic_tags_json TEXT NOT NULL DEFAULT '[]',
            last_excerpt TEXT NOT NULL DEFAULT '',
            updated_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS relationship_events (
            event_hash TEXT PRIMARY KEY,
            person_key TEXT NOT NULL,
            person_name TEXT NOT NULL,
            event_date TEXT NOT NULL,
            surface TEXT NOT NULL,
            interaction_kind TEXT NOT NULL,
            word_count INTEGER NOT NULL,
            verbatim_text TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_rel_threads_circle_score ON relationship_threads(circle, status, score DESC);"
    )
    conn.commit()


def extract_person_and_surface(
    title: str,
    text: str,
    owner_name: str = "",
) -> Tuple[Optional[str], str, str]:
    """Extract `(person_name, surface, interaction_kind)` from a comment/reply record."""
    title_s = (title or "").strip()
    text_s = (text or "").strip()
    group_m = re.search(r"Comment in (.+)$", title_s)
    surface = group_m.group(1).strip() if group_m else "Direct Thread"

    m_reply = re.search(r"replied to ([^']+)'s comment", title_s)
    m_comm = re.search(r"commented on ([^']+)'s (?:post|photo|reel|video|link)", title_s)
    interaction_kind = "group_or_edited"
    person: Optional[str] = None
    if m_reply:
        person = m_reply.group(1).strip()
        interaction_kind = "direct_reply"
    elif m_comm:
        person = m_comm.group(1).strip()
        interaction_kind = "post_comment"

    m_voc = re.match(
        r"^([A-Z][a-z]{1,15}\s+[A-Z][a-z]{1,18}(?:\s+(?:[A-Z][a-z]{1,18}|Jr\.|Sr\.))?)\b\s+([a-z].*)",
        text_s,
    )
    if m_voc:
        cand = m_voc.group(1).strip()
        words = cand.split()
        if not any(w in NON_PERSON_FIRST_WORDS for w in words):
            if not person or (cand.startswith(person) and len(words) == 3):
                person = cand
            if interaction_kind == "group_or_edited":
                interaction_kind = "vocative_reply"

    if person and owner_name and person.lower() == owner_name.strip().lower():
        return None, surface, interaction_kind

    return person, surface, interaction_kind


def build_contextual_opener(
    person_name: str,
    last_excerpt: str,
    topic_tags: List[str],
    surfaces: List[str],
    cta_override: Optional[str] = None,
) -> str:
    """
    Generate a 1-to-2 sentence human DM opener grounded in the actual prior exchange.
    Uses `default_cta` from `people_config.json` (or pure reconnect if `default_cta` is empty).
    """
    first_name = person_name.split()[0]
    cfg = load_people_config()
    cta = cta_override if cta_override is not None else str(cfg.get("default_cta") or "").strip()
    low = (last_excerpt or "").lower()
    surf_str = " ".join(surfaces).lower()

    if "benchmarking lately for harnesses" in low or "harness" in low:
        hook = (
            f"Hey {first_name}, remember when we talked about benchmarking agent harnesses? "
            "I finally ran the numbers and built a 1-second local check that shows how much context "
            "Claude or Cursor wasted on your machine over the last 30 days:"
        )
    elif "davinci" in low or "mcp" in low:
        hook = (
            f"Hey {first_name}, remember when we talked about skipping MCP for deterministic code? "
            "I audited my 19 MCP servers and found they were dumping 101k tokens into context before Turn 1. "
            "Built a 1-second local check if you want to see what your setup is burning:"
        )
    elif "homepage make it more clear" in low or "cords" in low:
        hook = (
            f"Hey {first_name}, appreciated your feedback on vifi a while back. "
            "Just shipped the local context shield side of it (wholebodycompute.com) with a 1-second terminal check "
            "that shows how much token bloat Claude or Cursor burned in the last 30 days:"
        )
    elif "qwen" in low or "m5 pro" in low or "gemma" in low:
        hook = (
            f"Hey {first_name}, appreciated trading local model notes with you in Vibe Coding. "
            "I built a 1-second local check that shows how many cloud tokens Claude or Cursor wasted "
            "re-reading dead history on your machine over the last 30 days:"
        )
    elif "255 files" in low or "auditable" in low or "qa" in low:
        hook = (
            f"Hey {first_name}, turned that token-savings setup from our AI audit thread into a "
            "1-second local terminal check. If you're using Claude Code or Cursor, run this and tell me what your 30-day receipt prints:"
        )
    elif "local ai" in surf_str or "local-models" in topic_tags:
        hook = (
            f"Hey {first_name}, following up on our local models chat. "
            "I built a 1-second local audit that checks your last 30 days of Claude or Cursor sessions "
            "and shows how much quota went to re-reading dead context:"
        )
    else:
        hook = (
            f"Hey {first_name}, I built a 1-second local check that shows how many tokens Claude Code or Cursor "
            "wasted re-reading dead history on your machine over the last 30 days. Tell me what yours prints:"
        )

    if not cta:
        return hook.rstrip(":") + "."
    return f"{hook}\n\n{cta}"


def _iter_normalized_records_from_path(src_path: Path) -> List[Dict[str, Any]]:
    """Load comment/message records from `.jsonl`, `.csv` (LinkedIn), or `.zip` archive."""
    records: List[Dict[str, Any]] = []
    suffix = src_path.suffix.lower()

    if suffix == ".jsonl":
        with src_path.open(encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        continue
    elif suffix == ".csv":
        with src_path.open(encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dt = str(row.get("Date") or row.get("date") or row.get("created_at") or "")[:10]
                recipient = str(row.get("TO") or row.get("Recipient") or row.get("person_name") or "").strip()
                txt = str(row.get("CONTENT") or row.get("Message") or row.get("Comment") or row.get("text") or "").strip()
                title = f"replied to {recipient}'s comment" if recipient else str(row.get("title") or "CSV Import")
                records.append({"source_type": "fb_comment", "date": dt, "title": title, "text": txt})
    elif suffix == ".zip":
        max_zip_entry_bytes = 64 * 1024 * 1024
        max_zip_total_bytes = 256 * 1024 * 1024
        total_read_bytes = 0
        with zipfile.ZipFile(src_path, "r") as zf:
            for zinfo in zf.infolist():
                if zinfo.file_size > max_zip_entry_bytes:
                    raise ValueError(f"Archive entry exceeds 64 MiB uncompressed limit: {zinfo.filename}")
                total_read_bytes += zinfo.file_size
                if total_read_bytes > max_zip_total_bytes:
                    raise ValueError("Archive exceeds 256 MiB total uncompressed limit")
                name = zinfo.filename
                low = name.lower()
                if low.endswith(".jsonl"):
                    raw = zf.read(name).decode("utf-8", errors="ignore")
                    for line in raw.splitlines():
                        if line.strip():
                            try:
                                records.append(json.loads(line))
                            except Exception:
                                continue
                elif low.endswith(".csv") and any(k in low for k in ("comment", "message")):
                    raw = zf.read(name).decode("utf-8", errors="ignore")
                    reader = csv.DictReader(io.StringIO(raw))
                    for row in reader:
                        dt = str(row.get("Date") or row.get("date") or "")[:10]
                        recipient = str(row.get("TO") or row.get("Recipient") or "").strip()
                        txt = str(row.get("CONTENT") or row.get("Message") or row.get("Comment") or "").strip()
                        title = f"replied to {recipient}'s comment" if recipient else "CSV Import"
                        records.append({"source_type": "fb_comment", "date": dt, "title": title, "text": txt})
    return records


def sync_relationship_archive(
    archive_path: Optional[Path] = None,
    db_path: Optional[Path] = None,
    since_date: str = "2026-08-01",
) -> Dict[str, Any]:
    """
    Run the 5-Gate Signal-vs-Nonsense indexer over an archive (`.jsonl`, `.csv`, `.zip`) and
    persist `relationship_threads` + `relationship_events` into Organ 04 `dot_memory.db`.
    """
    cfg = load_people_config()
    owner_name = str(cfg.get("owner_name") or "").strip()
    raw_arch = str(archive_path or cfg.get("archive_path") or "").strip()
    src_path = Path(raw_arch).expanduser() if raw_arch else None
    target_db = Path(db_path or _resolve_default_dot_db_path()).expanduser()
    target_db.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(target_db), timeout=10.0)
    ensure_relationship_tables(conn)

    if not src_path or not src_path.is_file():
        conn.close()
        return {
            "status": "no_archive",
            "archive_path": str(src_path or ""),
            "threads_upserted": 0,
            "hint": "Pass `body people --import <path.jsonl|.csv|.zip>` or set `archive_path` in ~/.whole_body_compute/people_config.json",
        }

    existing_meta: Dict[str, Dict[str, Any]] = {}
    for row in conn.execute(
        "SELECT person_key, circle, status, last_outbound_dm_at, outbound_note FROM relationship_threads"
    ).fetchall():
        existing_meta[str(row[0])] = {
            "circle": row[1],
            "status": row[2],
            "last_outbound_dm_at": row[3],
            "outbound_note": row[4] or "",
        }

    stats: Dict[str, int] = defaultdict(int)
    people: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "display_name": "",
            "interactions": [],
            "surfaces": set(),
            "topic_tags": set(),
            "warmth_hits": 0,
            "hostile_hits": 0,
            "sparring_hits": 0,
            "tech_hits": 0,
            "sacred_hits": 0,
            "kinship_override": False,
            "direct_replies": 0,
            "custom_circle_match": "",
        }
    )

    custom_circles_cfg: Dict[str, List[str]] = cfg.get("custom_circles") or {}
    records = _iter_normalized_records_from_path(src_path)

    for r in records:
        if r.get("source_type") and r.get("source_type") not in ("fb_comment", "fb_group_comment", "comment", "message"):
            continue
        stats["total_raw"] += 1
        title = str(r.get("title") or "").strip()
        text = redact_secrets(str(r.get("text") or "").strip())
        dt = str(r.get("date") or "")[:10] or datetime.date.today().isoformat()

        # Gate 1: Self-comments & bare links
        if "his own " in title or "her own " in title or "their own " in title or text.startswith("http"):
            stats["gate1_self_or_url_dropped"] += 1
            continue

        # Gate 2: Substantive floor (>= 15 words)
        wc = len(text.split())
        if wc < 15:
            stats["gate2_short_driveby_dropped"] += 1
            continue

        # Gate 3: Real human peer vs brand/page
        person, surface, kind = extract_person_and_surface(title, text, owner_name=owner_name)
        if not person:
            stats["gate3_no_named_peer_dropped"] += 1
            continue
        if person in BRAND_OR_PAGE_NAMES or any(
            b in person for b in ("Community", "Strategy", ".ai", "Club")
        ):
            stats["gate3_brand_or_page_dropped"] += 1
            continue

        pkey = _normalize_person_key(person)
        for existing_k, existing_v in list(people.items()):
            if existing_k != pkey and (
                existing_k.startswith(pkey + "_") or pkey.startswith(existing_k + "_")
            ):
                longer_name = (
                    person
                    if len(person) >= len(existing_v["display_name"])
                    else existing_v["display_name"]
                )
                pkey = _normalize_person_key(longer_name)
                if existing_k != pkey:
                    people[pkey] = people.pop(existing_k)
                person = longer_name
                break

        p = people[pkey]
        if len(person) > len(p["display_name"]):
            p["display_name"] = person
        p["surfaces"].add(surface)
        if kind in ("direct_reply", "vocative_reply"):
            p["direct_replies"] += 1
        if WARMTH_RE.search(text):
            p["warmth_hits"] += 1
        if HOSTILE_FRICTION_RE.search(text):
            p["hostile_hits"] += 1
        if SPARRING_DEBATE_RE.search(text):
            p["sparring_hits"] += 1
        if KINSHIP_SACRED_OVERRIDE_RE.search(text):
            p["kinship_override"] = True
        if TECH_RE.search(text) or any(
            g in surface for g in ("Vibe Coding", "Local AI", "Moda AI", "Open Source")
        ):
            p["tech_hits"] += 1
        if SACRED_POETRY_RE.search(text):
            p["sacred_hits"] += 1

        for c_name, kw_list in custom_circles_cfg.items():
            if isinstance(kw_list, list) and any(k.lower() in text.lower() for k in kw_list):
                p["custom_circle_match"] = c_name

        for tag_name, tag_re in TOPIC_TAG_PATTERNS:
            if tag_re.search(text) or tag_re.search(surface):
                p["topic_tags"].add(tag_name)

        p["interactions"].append(
            {
                "date": dt,
                "surface": surface,
                "kind": kind,
                "words": wc,
                "text": text,
            }
        )

    conn.execute("DELETE FROM relationship_threads")
    conn.execute("DELETE FROM relationship_events")
    now_ts = time.time()
    upserted = 0
    circle_counts: Dict[str, int] = defaultdict(int)

    for pkey, info in people.items():
        recent = [i for i in info["interactions"] if i["date"] >= since_date]
        if not recent:
            continue
        recent.sort(key=lambda x: x["date"], reverse=True)
        name = info["display_name"]

        if info["custom_circle_match"]:
            circle = info["custom_circle_match"]
            status = "active"
        elif info["kinship_override"] or (info["sacred_hits"] > info["tech_hits"]):
            circle = "personal_sacred"
            status = "active"
        elif info["hostile_hits"] > 0 and info["warmth_hits"] == 0:
            circle = "quarantined"
            status = "quarantined"
        elif info["sparring_hits"] > 0 and info["tech_hits"] == 0:
            circle = "sparring_partner"
            status = "active"
        elif info["tech_hits"] > 0:
            circle = "tech_builder"
            status = "active"
        else:
            circle = "personal_sacred"
            status = "active"

        prev = existing_meta.get(pkey, {})
        if prev.get("status") in ("allowed_override", "quarantined_override"):
            status = "active" if prev["status"] == "allowed_override" else "quarantined"
            if prev.get("circle"):
                circle = str(prev["circle"])
        last_dm = prev.get("last_outbound_dm_at")
        out_note = prev.get("outbound_note", "")

        score = (
            len(recent) * 2
            + info["direct_replies"] * 3
            + info["warmth_hits"] * 4
            + info["tech_hits"] * 2
        )
        first_dt = min(i["date"] for i in recent)
        last_dt = max(i["date"] for i in recent)
        surfaces_list = sorted(info["surfaces"])
        tags_list = sorted(info["topic_tags"])
        last_excerpt = recent[0]["text"]

        conn.execute(
            """
            INSERT INTO relationship_threads (
                person_key, person_name, circle, status, score,
                interaction_count, direct_replies, warmth_hits,
                first_seen_at, last_interaction_at, last_outbound_dm_at,
                outbound_note, surfaces_json, topic_tags_json, last_excerpt, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pkey,
                name,
                circle,
                status,
                score,
                len(recent),
                info["direct_replies"],
                info["warmth_hits"],
                first_dt,
                last_dt,
                last_dm,
                out_note,
                json.dumps(surfaces_list),
                json.dumps(tags_list),
                last_excerpt,
                now_ts,
            ),
        )

        for ev in recent:
            ev_hash = f"{pkey}:{ev['date']}:{abs(hash(ev['text'][:120]))}"
            conn.execute(
                """
                INSERT OR IGNORE INTO relationship_events (
                    event_hash, person_key, person_name, event_date, surface,
                    interaction_kind, word_count, verbatim_text, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ev_hash,
                    pkey,
                    name,
                    ev["date"],
                    ev["surface"],
                    ev["kind"],
                    ev["words"],
                    ev["text"],
                    now_ts,
                ),
            )

        upserted += 1
        circle_counts[circle] += 1

    conn.commit()
    conn.close()
    return {
        "status": "ok",
        "db_path": str(target_db),
        "archive_path": str(src_path),
        "threads_upserted": upserted,
        "circles": dict(circle_counts),
        "gate_stats": dict(stats),
    }


def record_manual_engagement_or_sent(
    person_name: str,
    *,
    mark_sent: bool = False,
    note: str = "",
    circle: Optional[str] = None,
    status_override: Optional[str] = None,
    surface: str = "Direct DM",
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Record an outbound DM (`--sent`) or update a contact's circle/status in Organ 04 `dot_memory.db`."""
    target_db = Path(db_path or _resolve_default_dot_db_path()).expanduser()
    conn = sqlite3.connect(str(target_db), timeout=10.0)
    ensure_relationship_tables(conn)

    pkey = _normalize_person_key(person_name)
    row = conn.execute(
        "SELECT person_key, person_name, circle, status, last_outbound_dm_at, outbound_note FROM relationship_threads WHERE person_key = ? OR lower(person_name) LIKE ?",
        (pkey, f"%{person_name.strip().lower()}%"),
    ).fetchone()

    today_str = datetime.date.today().isoformat()
    now_ts = time.time()

    if row:
        pkey = str(row[0])
        resolved_name = str(row[1])
        new_circle = circle or str(row[2])
        new_status = status_override or str(row[3])
        new_dm_at = today_str if mark_sent else row[4]
        new_note = redact_secrets(note) if note else str(row[5] or "")
        conn.execute(
            """
            UPDATE relationship_threads
            SET circle = ?, status = ?, last_outbound_dm_at = ?, outbound_note = ?, updated_at = ?
            WHERE person_key = ?
            """,
            (new_circle, new_status, new_dm_at, new_note, now_ts, pkey),
        )
    else:
        resolved_name = person_name.strip()
        new_circle = circle or "tech_builder"
        new_status = status_override or "active"
        new_dm_at = today_str if mark_sent else None
        new_note = redact_secrets(note)
        conn.execute(
            """
            INSERT INTO relationship_threads (
                person_key, person_name, circle, status, score, interaction_count,
                direct_replies, warmth_hits, first_seen_at, last_interaction_at,
                last_outbound_dm_at, outbound_note, surfaces_json, topic_tags_json, last_excerpt, updated_at
            ) VALUES (?, ?, ?, ?, 15, 1, 1, 1, ?, ?, ?, ?, ?, '[]', ?, ?)
            """,
            (
                pkey,
                resolved_name,
                new_circle,
                new_status,
                today_str,
                today_str,
                new_dm_at,
                new_note,
                json.dumps([surface]),
                new_note or f"Contacted via {surface}",
                now_ts,
            ),
        )

    conn.commit()
    conn.close()
    return {
        "status": "ok",
        "person_key": pkey,
        "person_name": resolved_name,
        "circle": new_circle,
        "thread_status": new_status,
        "last_outbound_dm_at": new_dm_at,
        "note": new_note,
    }


def capture_live_browser_engagement(
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Organ 03 (`The Eyes` - `FovealVisionEngine`) -> Organ 04 (`The Hippocampus` - `dot_memory.db`)
    live browser engagement capture. Requires Whole Body Compute Pro ($29/mo).
    """
    if not _is_pro_or_higher():
        return {
            "status": "upgrade_required",
            "feature": "people_live_capture",
            "required_tier": "pro",
            "upgrade_url": "https://wholebodycompute.com/checkout/pro",
            "message": "Organ 03 live browser relationship capture (`body people --capture`) requires Whole Body Compute Pro ($29/mo).",
        }

    try:
        from whole_body_compute.integrations.foveal_vision import capture_foveal_snapshot

        snap = capture_foveal_snapshot()
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc)}

    if not snap:
        return {"status": "no_window", "reason": "No active window snapshot available"}

    cfg = load_people_config()
    owner_name = str(cfg.get("owner_name") or "").strip()
    win_title = str(getattr(snap, "window_title", "") or "").strip()
    app_name = str(getattr(snap, "app_name", "") or "").strip()
    elements = getattr(snap, "elements", []) or []

    reply_candidates: List[str] = []
    heading_candidates: List[str] = []
    for el in elements:
        role = str(getattr(el, "role", "") or "")
        val = str(getattr(el, "value", "") or getattr(el, "title", "") or "").strip()
        if role in ("AXTextArea", "AXTextField") and len(val.split()) >= 12:
            reply_candidates.append(redact_secrets(val))
        elif role in ("AXHeading", "AXStaticText", "AXLink") and 2 <= len(val.split()) <= 5:
            if re.match(r"^[A-Z][a-z]+\s+[A-Z][a-z]+$", val) and val not in BRAND_OR_PAGE_NAMES:
                if not owner_name or val.lower() != owner_name.lower():
                    heading_candidates.append(val)

    if not reply_candidates:
        return {
            "status": "skipped_gate1_and_2",
            "app_name": app_name,
            "window_title": win_title,
            "reason": "No active outbound composer reply >= 12 words detected in focused Organ 03 DOM.",
        }

    reply_text = reply_candidates[0]
    person, surface, _ = extract_person_and_surface(win_title, reply_text, owner_name=owner_name)
    if not person and heading_candidates:
        person = heading_candidates[0]
    if not person:
        return {
            "status": "skipped_gate3",
            "app_name": app_name,
            "window_title": win_title,
            "reason": "Could not resolve named human peer from Organ 03 browser DOM.",
        }

    default_cta = str(cfg.get("default_cta") or "").strip()
    mark_sent = (
        "wholebodycompute.com/audit" in reply_text
        or "voicefi.org" in reply_text
        or (bool(default_cta) and default_cta in reply_text)
    )
    res = record_manual_engagement_or_sent(
        person,
        mark_sent=mark_sent,
        note=reply_text,
        surface=surface if surface != "Direct Thread" else (win_title[:60] or app_name),
        db_path=db_path,
    )
    return {
        "status": "captured",
        "person_name": res["person_name"],
        "surface": surface,
        "marked_outbound_sent": mark_sent,
        "reply_preview": reply_text[:140],
    }


def query_people(
    query: str = "",
    *,
    circle: str = "tech_builder",
    include_contacted: bool = False,
    limit: int = 10,
    cta_override: Optional[str] = None,
    db_path: Optional[Path] = None,
    archive_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Query ranked contacts from Organ 04 `dot_memory.db` (enforcing Community Top-5 vs. Pro unlimited)."""
    target_db = Path(db_path or _resolve_default_dot_db_path()).expanduser()
    conn = sqlite3.connect(str(target_db), timeout=10.0)
    ensure_relationship_tables(conn)

    cnt = conn.execute("SELECT count() FROM relationship_threads").fetchone()[0]
    if cnt == 0:
        conn.close()
        sync_relationship_archive(archive_path=archive_path, db_path=target_db)
        conn = sqlite3.connect(str(target_db), timeout=10.0)

    is_pro = _is_pro_or_higher()
    effective_limit = max(1, limit) if is_pro else min(max(1, limit), COMMUNITY_MAX_QUERY_LIMIT)
    capped_by_community = (not is_pro) and (limit > COMMUNITY_MAX_QUERY_LIMIT)

    sql = """
        SELECT
            person_key, person_name, circle, status, score,
            interaction_count, direct_replies, warmth_hits,
            first_seen_at, last_interaction_at, last_outbound_dm_at,
            outbound_note, surfaces_json, topic_tags_json, last_excerpt
        FROM relationship_threads
        WHERE status != 'quarantined'
    """
    params: List[Any] = []
    if circle and circle != "all":
        sql += " AND circle = ?"
        params.append(circle)
    if not include_contacted:
        sql += " AND (last_outbound_dm_at IS NULL OR last_outbound_dm_at = '')"

    sql += " ORDER BY score DESC, last_interaction_at DESC"
    rows = conn.execute(sql, params).fetchall()
    conn.close()

    q_low = (query or "").strip().lower()
    results: List[Dict[str, Any]] = []
    total_matched = 0
    for r in rows:
        surfaces = json.loads(r[12] or "[]")
        tags = json.loads(r[13] or "[]")
        excerpt = str(r[14] or "")
        name = str(r[1])

        if q_low:
            haystack = f"{name} {' '.join(surfaces)} {' '.join(tags)} {excerpt}".lower()
            if not all(tok in haystack for tok in q_low.split()):
                continue

        total_matched += 1
        if len(results) >= effective_limit:
            continue

        opener = build_contextual_opener(
            person_name=name,
            last_excerpt=excerpt,
            topic_tags=tags,
            surfaces=surfaces,
            cta_override=cta_override,
        )
        results.append(
            {
                "rank": len(results) + 1,
                "person_key": r[0],
                "person_name": name,
                "circle": r[2],
                "status": r[3],
                "score": int(r[4]),
                "interaction_count": int(r[5]),
                "direct_replies": int(r[6]),
                "warmth_hits": int(r[7]),
                "first_seen_at": r[8],
                "last_interaction_at": r[9],
                "last_outbound_dm_at": r[10],
                "outbound_note": r[11] or "",
                "surfaces": surfaces,
                "topic_tags": tags,
                "last_excerpt": excerpt,
                "suggested_dm": opener,
            }
        )

    return {
        "status": "ok",
        "query": query,
        "circle": circle,
        "tier": "pro" if is_pro else "community",
        "count": len(results),
        "total_matched": total_matched,
        "capped_by_community_tier": capped_by_community and total_matched > COMMUNITY_MAX_QUERY_LIMIT,
        "people": results,
    }


def format_people_table(report: Dict[str, Any], copied_rank: Optional[int] = None) -> str:
    """Format `body people` output for terminal display."""
    people = report.get("people", [])
    circle = report.get("circle", "tech_builder")
    q = report.get("query") or "all topics"
    total_m = int(report.get("total_matched", len(people)))
    lines = [
        f"🤝 Whole Body Compute - Organ 04 Relationship Hippocampus (`body people` | Circle: `{circle}` | Query: `{q}`)",
        "=" * 98,
    ]
    if not people:
        lines.append("  No matching contacts found (run `body people --import <path>` or try `--circle all`).")
        lines.append("=" * 98)
        return "\n".join(lines)

    for p in people:
        rk = p["rank"]
        surfs = ", ".join(p["surfaces"][:2])
        tags = ", ".join(p["topic_tags"][:4]) or "builder"
        sent_badge = f" [SENT: {p['last_outbound_dm_at']}]" if p.get("last_outbound_dm_at") else ""
        lines.append(
            f"  #{rk:<2d} {p['person_name']:<24} | Score: {p['score']:<2d} | Last: {p['last_interaction_at']} | "
            f"{p['interaction_count']} msg(s) | {surfs}{sent_badge}"
        )
        lines.append(f"      • Topics:  {tags}")
        snippet = p["last_excerpt"].replace("\n", " ")[:125]
        lines.append(f"      • Memory:  \"{snippet}...\"")
        dm_one_line = p["suggested_dm"].splitlines()[0]
        lines.append(f"      • Opener:  {dm_one_line}")
        lines.append("-" * 98)

    if report.get("capped_by_community_tier"):
        hidden = max(0, total_m - len(people))
        lines.append(
            f"  🔒 Community Preview: Showing top {len(people)} of {total_m} warm contacts (+{hidden} more unlocked in Pro $29/mo -> `body upgrade`)."
        )
    if copied_rank is not None:
        lines.append(f"  📋 Copied #{copied_rank}'s 2-line DM to macOS clipboard (Cmd+V ready)!")
    else:
        lines.append("  💡 Tip: Run `body people --copy 1` to copy #1's DM, `body people --sent \"Name\"` after sending, or `--import <file>`.")
    lines.append("=" * 98)
    return "\n".join(lines)


def copy_person_dm_to_clipboard(person_entry: Dict[str, Any]) -> bool:
    """Copy `suggested_dm` to macOS clipboard via `pbcopy`."""
    dm_text = str(person_entry.get("suggested_dm") or "").strip()
    if not dm_text:
        return False
    try:
        subprocess.run(["pbcopy"], input=dm_text.encode("utf-8"), check=True, timeout=5)
        return True
    except Exception:
        return False
