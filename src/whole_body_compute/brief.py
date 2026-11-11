"""
Phase 18.12: Sovereign Duet on Local Metal (`body brief` / `body duet` / `body morning`).

Synthesizes all 4 Sovereign OS Intelligence Pillars into a <30-second (<= 75 words)
proactive morning/channel briefing on Local Metal (:9380 gemma4-26b / :9379 voicefi-coder-mlx
/ local Ollama at $0.00 cloud cost):
1. 👁️ Pillar 1 - The Eyes (Organ 07): Recent Foveal episodic OCR & Foveal AXUIElement focus,
   with optional `--channel <name>` filtering and mandatory `redact_secrets` scrubbing.
2. 🧠 Pillar 2 - The Hippocampus (Organ 08): Recent high-salience facts/habits from
   `~/.voicefi/dot_memory.db` + auto-learned `PalWorkflow` candidates (`PalWorkflowRegistry`).
3. 🦴 Pillar 3 - Workspace & Git State (Organ 09): Live Git branch, porcelain delta count,
   modified/untracked file summary, latest commit, and active pipelines (`.agents/ACTIVE_PIPELINES.md`).
4. 💰 Pillar 4 - FinOps & Organ Vitals (`body report`): 24h tokens & USD saved at $0.00 Local Metal,
   local offload ratio, Blood-Brain Barrier (Organ 05) shield interceptions, CSI, and thermal headroom.

Includes:
- `--speak`: VoiceFi neural TTS playback (`vifi speak` with graceful fallback) of the <30s duet script.
- `--approve`: Human-in-the-Loop approval gate promoting staged `PalWorkflow` candidates via
  `PalWorkflowRegistry.promote_candidate(candidate_id)` and recording approval telemetry.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from whole_body_compute.redaction import redact_secrets

logger = logging.getLogger("whole_body_compute.brief")

WBC_REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_SPOKEN_WORDS = 72  # Strictly < 75 words (< 30 seconds spoken aloud)

_CACHED_BRIEF_METRICS: Dict[str, Dict[str, Any]] = {}
_CACHED_BRIEF_METRICS_TS: Dict[str, float] = {}

CHANNEL_KEYWORD_MAP: Dict[str, Tuple[str, ...]] = {
    "enterprise": ("enterprise", "cad", "tax", "county", "scraper", "parcel", "deed"),
    "voicefi": ("voicefi", "vifi", "tts", "voice", "audio", "cognitive", "barge", "live", "mlx"),
    "carousel": ("carousel", "editorial", "magazine", "oren", "slide", "spread", "issue"),
    "immune": ("immune", "bbb", "blood-brain", "secgate", "sandbox", "shield", "hook"),
    "finops": ("finops", "metrics", "savings", "tokens", "roi", "report", "scorecard", "cost"),
    "pal": ("pal", "workflow", "candidate", "recipe", "ambient", "foveal", "osworld"),
}


def _plural(count: int, singular: str, plural: Optional[str] = None) -> str:
    """Format a count + noun with proper singular/plural (never '(s)')."""
    n = int(count)
    word = singular if n == 1 else (plural or f"{singular}s")
    return f"{n} {word}"


def _format_spoken_number(n: float | int) -> str:
    """
    Convert a numeric value into natural spoken English for TTS playback
    (e.g., 163_867_900 -> '164 million', 425_000 -> '425 thousand').
    """
    val = float(n or 0.0)
    if val < 0:
        return f"minus {_format_spoken_number(-val)}"
    if val >= 1_000_000_000:
        b = val / 1_000_000_000.0
        if b >= 10.0 or abs(b - round(b)) < 0.05:
            return f"{int(round(b))} billion"
        return f"{b:.1f} billion"
    if val >= 1_000_000:
        m = val / 1_000_000.0
        if m >= 10.0 or abs(m - round(m)) < 0.05:
            return f"{int(round(m))} million"
        return f"{m:.1f} million"
    if val >= 1_000:
        k = int(round(val / 1_000.0))
        if k >= 1000:
            return "1 million"
        return f"{k} thousand"
    return str(int(round(val)))


def _format_compact_number(n: float | int) -> str:
    """Format a number compactly for terminal display (e.g., 163.9M, 425K)."""
    val = float(n or 0.0)
    if val >= 1_000_000_000:
        return f"{val / 1_000_000_000.0:.1f}B"
    if val >= 1_000_000:
        return f"{val / 1_000_000.0:.1f}M"
    if val >= 1_000:
        return f"{val / 1_000.0:.1f}K"
    return str(int(round(val)))


def _format_spoken_dollars(usd: float) -> str:
    """
    Convert a USD amount into natural spoken English for TTS playback
    (e.g., 154.40 -> '154 dollars', 14.80 -> '15 dollars', 0.0 -> 'zero dollars').
    """
    val = float(usd or 0.0)
    if val <= 0.005:
        return "zero dollars"
    if val >= 10.0 or abs(val - round(val)) < 0.1:
        rounded = int(round(val))
        return f"{rounded} {'dollar' if rounded == 1 else 'dollars'}"
    if val >= 1.0:
        whole = int(round(val))
        return f"{whole} {'dollar' if whole == 1 else 'dollars'}"
    cents = max(1, int(round(val * 100)))
    return f"{cents} {'cent' if cents == 1 else 'cents'}"


def _clean_memory_fact_for_display(fact: str, category: str = "") -> str:
    """
    Clean raw memory strings (especially `lgtm_edit_lesson` rows containing raw HTML tags)
    into crisp, human-readable one-line insights for terminal display.
    """
    cleaned = redact_secrets(fact or "").strip()
    if not cleaned:
        return ""

    # Summarize noisy `lgtm_edit_lesson` rows that contain raw HTML snippets
    m_lgtm = re.match(
        r"When drafting\s+([A-Za-z0-9_.-]+)\s+to\s+'([^']+)',\s*user preferred phrasing:\s*(.*)",
        cleaned,
        re.IGNORECASE | re.DOTALL,
    )
    if m_lgtm:
        tool_name, target_file, snippet = m_lgtm.group(1), m_lgtm.group(2), m_lgtm.group(3)
        # Strip HTML tags from snippet
        text_only = re.sub(r"<[^>]+>", " ", snippet)
        text_only = re.sub(r"\s+", " ", text_only).strip(" '\"\t\r\n")
        if text_only and len(text_only) >= 4 and not text_only.startswith(("class=", "span", "div")):
            return f"Preferred phrasing in {target_file} ({tool_name}): \"{text_only[:75]}\""
        return f"Learned UI & copy styling preference for {target_file} ({tool_name})"

    # Clean verbose `[Sovereign RSI Rule `...` (...)]: ` prefixes
    cleaned = re.sub(
        r"^\[Sovereign\s+RSI\s+Rule\s+`?[^`\]]+`?(?:\s*\([^)]+\))?\]\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Strip raw HTML tags from general memories
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _is_valid_spoken_briefing(text: str) -> bool:
    """
    Guardrail against code-completion / `.pyi` skeleton endpoints on :9379/:9380
    leaking Python code or AST skeletons into spoken TTS briefings.
    """
    if not text or len(text.split()) < 15:
        return False
    low = text.lower()
    code_markers = (
        "skeleton:",
        "imports:",
        "import ",
        "def ",
        "class ",
        "path.home",
        ".py",
        ".json",
        '"""',
        "l55-",
        "traceback",
    )
    if any(m in low for m in code_markers):
        return False
    conversational_markers = ("morning", "briefing", "branch", "roadmap", "saved", "tokens", "memory", "workflow")
    return sum(1 for m in conversational_markers if m in low) >= 2


def _summarize_roadmap_prompt(prompt_text: str) -> str:
    """
    Convert a raw `Next Resume Prompt` block into a conversational, human-readable title
    suitable for both terminal display and spoken TTS.
    """
    raw = redact_secrets(prompt_text or "").strip()
    if not raw:
        return ""
    first_line = raw.splitlines()[0].strip()

    # Match Editorial Carousel issues cleanly
    m_issue = re.search(
        r"(Issue\s*#\d+)\s+of\s+the\s+.*?Editorial\s+Magazine\s+Carousel\s+Series:\s*\"([^\"]+)\"",
        first_line,
        re.IGNORECASE,
    )
    if m_issue:
        issue_num = m_issue.group(1).replace("#", "")
        issue_title = m_issue.group(2).strip()
        return f"Editorial Carousel {issue_num}: {issue_title}"

    # Strip trailing file paths / `following .agents/...` clauses
    cleaned = re.sub(r"\s+following\s+\S+.*$", "", first_line, flags=re.IGNORECASE)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:115]


def _channel_terms(channel: Optional[str]) -> List[str]:
    """Expand a channel name into lowercase matching keywords."""
    if not channel:
        return []
    ch = channel.strip().lower()
    if ch in ("", "all", "morning", "default", "general", "duet"):
        return []
    terms = [ch]
    for k, aliases in CHANNEL_KEYWORD_MAP.items():
        if ch == k or ch in aliases:
            for a in aliases:
                if a not in terms:
                    terms.append(a)
    for part in re.split(r"[\s,_\-/]+", ch):
        if part and part not in terms:
            terms.append(part)
    return terms


def _matches_channel(text: str, terms: List[str]) -> Tuple[bool, int]:
    """Return (matched, score) for text against channel terms."""
    if not terms:
        return True, 1
    low = (text or "").lower()
    score = sum(1 for t in terms if t in low)
    return score > 0, score


def _clean_for_speech(text: str, max_words: int = MAX_SPOKEN_WORDS) -> str:
    """
    Sanitize text for natural Walkie-Talkie / Duet TTS playback:
    - Applies `redact_secrets`
    - Replaces `(s)` with `s` so TTS never pronounces 'open paren s close paren'
    - Strips HTML tags, markdown symbols, CLI flags, and raw file paths
    - Clamps strictly to `max_words` words (<30 seconds)
    """
    scrubbed = redact_secrets(text or "")
    scrubbed = scrubbed.replace("(s)", "s").replace("(S)", "s")
    # Strip raw HTML tags
    scrubbed = re.sub(r"<[^>]+>", " ", scrubbed)
    # Convert markdown links [label](url) -> label
    scrubbed = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", scrubbed)
    # Remove CLI flags like --promote or --auto
    scrubbed = re.sub(r"\s--[A-Za-z0-9_-]+", " ", scrubbed)
    # Replace underscores and slashes in identifiers with spaces so TTS reads words naturally
    scrubbed = re.sub(r"([A-Za-z0-9])[_/]([A-Za-z0-9])", r"\1 \2", scrubbed)
    # Remove markdown formatting characters and brackets/parens while preserving readable words
    scrubbed = re.sub(r"[`*_#~>|\(\)\[\]\{\}]+", " ", scrubbed)
    scrubbed = re.sub(r"\s+", " ", scrubbed).strip()
    words = scrubbed.split()
    if len(words) > max_words:
        clipped = " ".join(words[:max_words]).rstrip(",;:-")
        # Trim back to last complete sentence boundary if possible so we never cut off mid-sentence
        last_sent = max(clipped.rfind(". "), clipped.rfind("! "), clipped.rfind("? "))
        if last_sent > len(clipped) // 2:
            return clipped[: last_sent + 1].strip()
        if not clipped.endswith((".", "!", "?")):
            clipped += "."
        return clipped
    return scrubbed


LINEAR_ISSUE_KEY_PATTERN = re.compile(r"\b([A-Z]{2,8}-\d+)\b")
NOISE_ISSUE_PREFIXES = (
    "UTF-",
    "SHA-",
    "ISO-",
    "CVE-",
    "PEP-",
    "HTTP-",
    "TIER-",
    "VIFI-PAT",
    "PAT-",
    "RFC-",
    "UUID-",
)


def _extract_linear_issue_keys(text: str) -> List[str]:
    """Extract valid Linear issue keys (e.g. DEA-299) while ignoring non-issue noise prefixes."""
    if not text:
        return []
    matches = LINEAR_ISSUE_KEY_PATTERN.findall(text)
    valid: List[str] = []
    for m in matches:
        upper_m = m.upper()
        if any(upper_m.startswith(noise) for noise in NOISE_ISSUE_PREFIXES):
            continue
        if upper_m not in valid:
            valid.append(upper_m)
    return valid


def collect_github_pr_status(
    workspace_root: Optional[Path] = None,
    timeout_sec: float = 2.0,
) -> Dict[str, Any]:
    """
    Collect GitHub PR and review status via `gh pr status`.
    Gracefully handles missing `gh`, non-GitHub remotes, unauthenticated `gh`,
    or timeouts in <2s without crashing. All strings are scrubbed with `redact_secrets`.
    """
    empty_result: Dict[str, Any] = {
        "available": False,
        "current_branch_pr": None,
        "needs_review": [],
        "needs_review_count": 0,
        "created_open_count": 0,
    }

    if shutil.which("gh") is None:
        return empty_result

    cwd = str(workspace_root) if workspace_root else str(WBC_REPO_ROOT)

    candidates = [
        ["gh", "pr", "status", "--json", "currentBranch,createdBy,needsReview"],
        ["gh", "pr", "status", "--json", "number,title,state,isDraft,reviewDecision,url,headRefName"],
    ]

    data: Optional[Dict[str, Any]] = None
    for cmd in candidates:
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                stdout_text = proc.stdout.strip()
                start_idx = stdout_text.find("{")
                end_idx = stdout_text.rfind("}")
                if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
                    data = json.loads(stdout_text[start_idx : end_idx + 1])
                    break
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, Exception) as exc:
            logger.debug("gh pr status candidate failed: %s", exc)
            continue

    if not isinstance(data, dict):
        return empty_result

    # 1. Current Branch PR
    cb_raw = data.get("currentBranch")
    current_branch_pr: Optional[Dict[str, Any]] = None
    if isinstance(cb_raw, dict) and cb_raw.get("number"):
        current_branch_pr = {
            "number": int(cb_raw.get("number", 0)),
            "title": redact_secrets(str(cb_raw.get("title", ""))),
            "state": str(cb_raw.get("state", "OPEN")).upper(),
            "is_draft": bool(cb_raw.get("isDraft", cb_raw.get("is_draft", False))),
            "review_decision": str(cb_raw.get("reviewDecision", cb_raw.get("review_decision", ""))),
            "url": redact_secrets(str(cb_raw.get("url", ""))),
        }

    # 2. Needs Review
    needs_review: List[Dict[str, Any]] = []
    nr_raw = data.get("needsReview") or []
    if isinstance(nr_raw, list):
        for item in nr_raw:
            if isinstance(item, dict) and item.get("number"):
                needs_review.append(
                    {
                        "number": int(item.get("number", 0)),
                        "title": redact_secrets(str(item.get("title", ""))),
                        "url": redact_secrets(str(item.get("url", ""))),
                    }
                )

    # 3. Created By
    cb_list = data.get("createdBy") or []
    created_open_count = len([x for x in cb_list if isinstance(x, dict)])

    return {
        "available": True,
        "current_branch_pr": current_branch_pr,
        "needs_review": needs_review,
        "needs_review_count": len(needs_review),
        "created_open_count": created_open_count,
    }


def collect_linear_standup_state(
    foveal: Optional[Dict[str, Any]] = None,
    eyes: Optional[Dict[str, Any]] = None,
    hippocampus: Optional[Dict[str, Any]] = None,
    workspace: Optional[Dict[str, Any]] = None,
    linear_api_key: Optional[str] = None,
    timeout_sec: float = 1.5,
) -> Dict[str, Any]:
    """
    Multi-source sovereign discovery of active Linear tickets (<1.5s, zero-crash if offline):
    1. Live Linear Desktop Window / Foveal Eyes (Quartz & AXUIElement)
    2. Git branch, recent commit subject, or hippocampus memory facts
    3. Optional Linear GraphQL API (if LINEAR_API_KEY is present)
    """
    active_issues: List[Dict[str, Any]] = []
    in_pytest = bool(os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("VOICEFI_TESTING") == "1")

    def _add_issue(issue_id: str, title: str, source: str, status: str = "in_progress") -> None:
        issue_id_clean = issue_id.strip().upper()
        if not issue_id_clean:
            return
        if any(issue_id_clean.startswith(noise) for noise in NOISE_ISSUE_PREFIXES):
            return
        clean_title = redact_secrets(title.strip()) or issue_id_clean
        existing = next((x for x in active_issues if x["id"] == issue_id_clean), None)
        if existing:
            if existing.get("title") == issue_id_clean and clean_title != issue_id_clean:
                existing["title"] = clean_title
            return
        active_issues.append(
            {
                "id": issue_id_clean,
                "title": clean_title,
                "source": source,
                "status": status,
            }
        )

    # Source 1: Live Linear Desktop Window / Foveal Eyes
    if isinstance(foveal, dict):
        f_app = str(foveal.get("app_name", ""))
        f_title = str(foveal.get("window_title", ""))
        if "linear" in f_app.lower() or "linear" in f_title.lower() or LINEAR_ISSUE_KEY_PATTERN.search(f_title):
            for k in _extract_linear_issue_keys(f_title):
                _add_issue(k, f_title, "linear_desktop", "in_progress")

    # Check Quartz on macOS (if not in pytest or if Quartz is mocked)
    try:
        import Quartz

        is_mocked = (
            hasattr(Quartz.CGWindowListCopyWindowInfo, "mock")
            or hasattr(Quartz.CGWindowListCopyWindowInfo, "assert_called")
            or type(Quartz.CGWindowListCopyWindowInfo).__name__ in ("MagicMock", "Mock")
        )
        if not in_pytest or is_mocked:
            wins = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionAll,
                Quartz.kCGNullWindowID,
            )
            if wins:
                for w in wins:
                    owner = str(w.get("kCGWindowOwnerName") or "").strip()
                    name = str(w.get("kCGWindowName") or "").strip()
                    if "Linear" in owner or owner.lower() == "linear" or "Linear" in name:
                        if name:
                            for k in _extract_linear_issue_keys(name):
                                _add_issue(k, name, "linear_desktop", "in_progress")
    except Exception as exc:
        logger.debug("Quartz Linear window discovery fallback: %s", exc)

    if isinstance(eyes, dict):
        for frame in eyes.get("recent_frames", []):
            if isinstance(frame, dict):
                w_name = str(frame.get("window_name", ""))
                txt = str(frame.get("text", ""))
                app = str(frame.get("app_name", ""))
                if "linear" in app.lower() or "linear" in w_name.lower():
                    combined = f"{w_name} {txt}"
                    for k in _extract_linear_issue_keys(combined):
                        _add_issue(k, w_name or txt[:80] or k, "linear_desktop", "in_progress")

    # Source 2: Git Branch / Recent Commit / Hippocampus
    if isinstance(workspace, dict):
        branch = str(workspace.get("git_branch", ""))
        if branch:
            for k in _extract_linear_issue_keys(branch):
                _add_issue(k, f"{k} (from git branch {branch})", "git_branch", "active")

        commit = str(workspace.get("latest_commit", ""))
        if commit:
            for k in _extract_linear_issue_keys(commit):
                _add_issue(k, commit, "git_commit", "active")

    if isinstance(hippocampus, dict):
        for mem in hippocampus.get("recent_memories", []):
            if isinstance(mem, dict):
                fact = str(mem.get("fact", ""))
                for k in _extract_linear_issue_keys(fact):
                    _add_issue(k, fact[:100], "hippocampus", "memory")

    # Source 3: Optional Linear GraphQL API
    key = (linear_api_key or os.environ.get("LINEAR_API_KEY", "")).strip()
    if key:
        try:
            url = "https://api.linear.app/graphql"
            query = """
            query {
              viewer {
                assignedIssues(filter: { state: { type: { in: ["started", "unstarted"] } } }, first: 10) {
                  nodes {
                    identifier
                    title
                    state {
                      name
                      type
                    }
                  }
                }
              }
            }
            """
            payload = json.dumps({"query": query}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": key,
                    "User-Agent": "WholeBodyCompute/1.0",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                nodes = (
                    data.get("data", {})
                    .get("viewer", {})
                    .get("assignedIssues", {})
                    .get("nodes", [])
                )
                for node in nodes:
                    if isinstance(node, dict):
                        ident = str(node.get("identifier", "")).strip()
                        title = str(node.get("title", "")).strip()
                        st_name = (node.get("state") or {}).get("name", "active")
                        if ident:
                            full_title = f"{ident} {title}".strip() if title else ident
                            _add_issue(ident, full_title, "linear_api", st_name)
        except Exception as exc:
            logger.debug("Linear GraphQL query failed: %s", exc)

    primary = active_issues[0] if active_issues else None
    return {
        "available": len(active_issues) > 0,
        "active_issues": active_issues,
        "primary_issue": primary,
    }


class SovereignDuetBriefingEngine:
    """
    Synthesizes the 4-Pillar Sovereign Duet Briefing on Apple Silicon Local Metal ($0.00).
    """

    def __init__(
        self,
        workspace_root: Optional[Path] = None,
        dot_db_path: Optional[Path] = None,
        screenpipe_db_path: Optional[Path] = None,
    ) -> None:
        self.workspace_root = Path(workspace_root or os.getcwd()).resolve()
        self.dot_db_path = Path(
            dot_db_path
            or os.environ.get("VOICEFI_DOT_MEMORY_DB", str(Path.home() / ".voicefi" / "dot_memory.db"))
        )
        foveal_default = Path.home() / ".whole_body" / "foveal_memory.sqlite"
        screenpipe_default = Path.home() / ".screenpipe" / "db.sqlite"
        default_db = foveal_default if foveal_default.exists() and not screenpipe_default.exists() else screenpipe_default
        self.screenpipe_db_path = Path(
            screenpipe_db_path
            or os.environ.get("FOVEAL_MEMORY_DB_PATH")
            or os.environ.get("SCREENPIPE_DB_PATH")
            or default_db
        )

    # ------------------------------------------------------------------
    # Pillar 1: The Eyes (Organ 07 - Foveal Episodic Memory & AXUIElement)
    # ------------------------------------------------------------------
    def collect_pillar1_eyes(
        self,
        channel: str = "morning",
        screen_override: Optional[Dict[str, Any]] = None,
        foveal_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Inspect recent Screenpipe episodic context and Foveal AXUIElement window focus.
        All strings are scrubbed through `redact_secrets`.
        """
        from whole_body_compute.integrations.screenpipe import (
            extract_semantic_screen_entities,
            strip_ui_chrome_from_ocr,
        )

        terms = _channel_terms(channel)
        in_pytest = bool(os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("VOICEFI_TESTING") == "1")

        # 1. Foveal AXUIElement Focus
        foveal_app = "Terminal"
        foveal_title = self.workspace_root.name
        foveal_elements_count = 0
        foveal_focused_element = ""
        foveal_status = "standby"

        if isinstance(foveal_override, dict):
            foveal_app = redact_secrets(str(foveal_override.get("app_name", foveal_app)))
            foveal_title = redact_secrets(str(foveal_override.get("window_title", foveal_title)))
            foveal_elements_count = int(foveal_override.get("elements_count", 0))
            foveal_focused_element = redact_secrets(str(foveal_override.get("focused_element", "")))
            foveal_status = str(foveal_override.get("status", "active"))
        elif not in_pytest:
            try:
                from whole_body_compute.integrations.foveal_vision import FovealVisionEngine

                snap = FovealVisionEngine.capture_snapshot(max_elements=15, record_telemetry=False)
                if snap:
                    foveal_app = redact_secrets(str(getattr(snap, "app_name", "") or foveal_app))
                    foveal_title = redact_secrets(str(getattr(snap, "window_title", "") or foveal_title))
                    elems = getattr(snap, "elements", []) or []
                    foveal_elements_count = len(elems)
                    foveal_status = "active"
                    for el in elems:
                        if getattr(el, "focused", False) or getattr(el, "title", ""):
                            foveal_focused_element = redact_secrets(
                                f"{getattr(el, 'role', 'AXElement')}:{getattr(el, 'title', '')}"
                            )
                            break
            except Exception as exc:
                logger.debug("Foveal snapshot fallback: %s", exc)

        # 2. Screenpipe Episodic OCR & Recent Windows
        recent_frames: List[Dict[str, str]] = []
        raw_ocr_blobs: List[str] = []
        screenpipe_source = "fallback"

        if isinstance(screen_override, dict):
            screenpipe_source = str(screen_override.get("source", "override"))
            for item in screen_override.get("recent_frames", []):
                if isinstance(item, dict):
                    app_s = redact_secrets(str(item.get("app_name", "Workspace")))
                    win_s = redact_secrets(str(item.get("window_name", "")))
                    txt_s = redact_secrets(strip_ui_chrome_from_ocr(str(item.get("text", ""))))
                    combined = f"{app_s} {win_s} {txt_s}"
                    matched, score = _matches_channel(combined, terms)
                    if matched or not terms:
                        recent_frames.append(
                            {
                                "app_name": app_s,
                                "window_name": win_s,
                                "text": txt_s[:280],
                                "channel_score": score,
                            }
                        )
                        raw_ocr_blobs.append(txt_s)
            if screen_override.get("ocr_text"):
                cleaned_ocr = redact_secrets(strip_ui_chrome_from_ocr(str(screen_override["ocr_text"])))
                raw_ocr_blobs.append(cleaned_ocr)
                if not recent_frames:
                    matched, score = _matches_channel(cleaned_ocr, terms)
                    if matched or not terms:
                        recent_frames.append(
                            {
                                "app_name": foveal_app,
                                "window_name": foveal_title,
                                "text": cleaned_ocr[:280],
                                "channel_score": score,
                            }
                        )
        elif self.screenpipe_db_path.is_file():
            try:
                with sqlite3.connect(
                    f"file:{self.screenpipe_db_path}?mode=ro", uri=True, timeout=1.0
                ) as conn:
                    conn.row_factory = sqlite3.Row
                    rows = conn.execute(
                        """
                        SELECT app_name, window_name, text
                        FROM ocr_text
                        WHERE text IS NOT NULL AND TRIM(text) != ''
                        ORDER BY rowid DESC
                        LIMIT 25
                        """
                    ).fetchall()
                    screenpipe_source = "sqlite_ro"
                    for r in rows:
                        app_s = redact_secrets(str(r["app_name"] or "Mac"))
                        win_s = redact_secrets(str(r["window_name"] or ""))
                        txt_s = redact_secrets(strip_ui_chrome_from_ocr(str(r["text"] or "")))
                        if not txt_s:
                            continue
                        combined = f"{app_s} {win_s} {txt_s}"
                        matched, score = _matches_channel(combined, terms)
                        if matched:
                            recent_frames.append(
                                {
                                    "app_name": app_s,
                                    "window_name": win_s,
                                    "text": txt_s[:280],
                                    "channel_score": score,
                                }
                            )
                            raw_ocr_blobs.append(txt_s)
                            if len(recent_frames) >= 5:
                                break
            except Exception as exc:
                logger.debug("Screenpipe SQLite read fallback: %s", exc)

        if terms and recent_frames:
            recent_frames.sort(key=lambda x: int(x.get("channel_score", 0)), reverse=True)

        merged_ocr = redact_secrets("\n".join(raw_ocr_blobs))
        entities = extract_semantic_screen_entities(merged_ocr) if merged_ocr else {
            "file_paths": [],
            "urls": [],
            "commands": [],
            "channels_or_ids": [],
        }
        entities = {
            k: [redact_secrets(str(v)) for v in vals[:6]]
            for k, vals in entities.items()
        }

        active_apps = list(
            dict.fromkeys(
                [
                    f["app_name"]
                    for f in recent_frames
                    if f.get("app_name") and f["app_name"].lower() not in ("python3", "python", "uv", "zsh", "bash")
                ]
                + ([foveal_app] if foveal_app and foveal_app.lower() not in ("python3", "python", "uv", "zsh", "bash") else [])
            )
        )[:5]

        if foveal_app.lower() in ("python3", "python", "uv", "zsh", "bash"):
            foveal_app = active_apps[0] if active_apps else "Terminal"
            if not active_apps:
                active_apps = ["Terminal"]

        focus_summary = (
            f"{foveal_app} - {foveal_title}"
            if foveal_title and foveal_title.lower() != foveal_app.lower()
            else foveal_app
        )
        if recent_frames:
            top_f = recent_frames[0]
            snippet = top_f.get("text", "").splitlines()[0][:110] if top_f.get("text") else ""
            if snippet:
                focus_summary = f"{focus_summary} ({snippet})"

        return {
            "organ_id": "organ_07_eyes",
            "channel": channel,
            "channel_terms": terms,
            "screenpipe_source": screenpipe_source,
            "foveal_focus": {
                "status": foveal_status,
                "app_name": foveal_app,
                "window_title": foveal_title,
                "elements_count": foveal_elements_count,
                "focused_element": foveal_focused_element,
            },
            "active_apps": active_apps,
            "recent_frames_count": len(recent_frames),
            "recent_frames": recent_frames[:5],
            "semantic_entities": entities,
            "focus_summary": redact_secrets(focus_summary),
            "secrets_redacted": True,
        }

    # ------------------------------------------------------------------
    # Pillar 2: The Hippocampus (Organ 08 - DotMemory & PalWorkflow)
    # ------------------------------------------------------------------
    def collect_pillar2_hippocampus(
        self,
        channel: str = "morning",
        suggestions_override: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Query recent high-salience facts/habits from `~/.voicefi/dot_memory.db`
        and auto-learned `PalWorkflow` candidates via `PalWorkflowRegistry.get_auto_learned_suggestions()`.
        """
        from whole_body_compute.integrations.pal_workflow import PalWorkflowRegistry

        terms = _channel_terms(channel)
        recent_memories: List[Dict[str, Any]] = []
        total_active_memories = 0
        seen_display_Summaries: set[str] = set()

        if self.dot_db_path.is_file():
            try:
                with sqlite3.connect(
                    f"file:{self.dot_db_path}?mode=ro", uri=True, timeout=1.5
                ) as conn:
                    conn.row_factory = sqlite3.Row
                    cnt_row = conn.execute(
                        "SELECT COUNT(*) FROM memories WHERE COALESCE(archived, 0) = 0"
                    ).fetchone()
                    if cnt_row and cnt_row[0] is not None:
                        total_active_memories = int(cnt_row[0])

                    rows = conn.execute(
                        """
                        SELECT id, category, tags, fact, COALESCE(salience, 1.0) AS salience
                        FROM memories
                        WHERE COALESCE(archived, 0) = 0
                        ORDER BY
                            CASE
                                WHEN category IN ('habit_pattern', 'architecture', 'preference', 'rsi_superintelligence_rule') THEN 1
                                WHEN category = 'lgtm_edit_lesson' THEN 3
                                ELSE 2
                            END ASC,
                            COALESCE(salience, 1.0) DESC,
                            id DESC
                        LIMIT 45
                        """
                    ).fetchall()
                    for r in rows:
                        fact_s = redact_secrets(str(r["fact"] or "").strip())
                        tags_s = redact_secrets(str(r["tags"] or "").strip())
                        cat_s = str(r["category"] or "general").strip()
                        if not fact_s:
                            continue
                        matched, score = _matches_channel(f"{cat_s} {tags_s} {fact_s}", terms)
                        if matched or not terms:
                            display_fact = _clean_memory_fact_for_display(fact_s, cat_s)
                            dedup_key = display_fact.lower()[:90]
                            if dedup_key in seen_display_Summaries:
                                continue
                            seen_display_Summaries.add(dedup_key)
                            recent_memories.append(
                                {
                                    "id": int(r["id"]),
                                    "category": cat_s,
                                    "tags": tags_s,
                                    "fact": display_fact[:260],
                                    "raw_fact": fact_s[:260],
                                    "salience": round(float(r["salience"] or 1.0), 3),
                                    "channel_score": score,
                                }
                            )
            except Exception as exc:
                logger.debug("DotMemory read fallback: %s", exc)

        if terms and recent_memories:
            recent_memories.sort(
                key=lambda x: (int(x.get("channel_score", 0)), float(x.get("salience", 0.0))),
                reverse=True,
            )

        # Query PalWorkflowRegistry auto-learned suggestions
        query_arg = channel if terms else None
        if suggestions_override is not None:
            raw_suggestions = list(suggestions_override)
        else:
            try:
                raw_suggestions = PalWorkflowRegistry.get_auto_learned_suggestions(query=query_arg)
                if not raw_suggestions and query_arg:
                    raw_suggestions = PalWorkflowRegistry.get_auto_learned_suggestions(query=None)
            except Exception as exc:
                logger.debug("PalWorkflowRegistry suggestions fallback: %s", exc)
                raw_suggestions = []

        staged_candidates: List[Dict[str, Any]] = []
        active_workflows: List[Dict[str, Any]] = []
        learned_habits: List[Dict[str, Any]] = []

        for s in raw_suggestions:
            if not isinstance(s, dict):
                continue
            s_clean = {
                "id": str(s.get("id", "")),
                "type": str(s.get("type", "ambient_workflow")),
                "title": redact_secrets(str(s.get("title", s.get("id", "Workflow")))),
                "status": str(s.get("status", "staged")),
                "dominant_tier": str(s.get("dominant_tier", "TIER_0_LOCAL")),
                "reliability_score": float(s.get("reliability_score", s.get("salience", 1.0)) or 1.0),
                "recommendation": redact_secrets(str(s.get("recommendation", ""))),
                "trigger_example": str(s.get("trigger_example", "")),
                "promotion_ready": bool(
                    s.get("status") == "staged"
                    and float(s.get("reliability_score", 1.0) or 1.0) >= 0.8
                ),
            }
            if s_clean["type"] == "ambient_workflow":
                if s_clean["status"] == "staged":
                    staged_candidates.append(s_clean)
                else:
                    active_workflows.append(s_clean)
            else:
                learned_habits.append(s_clean)

        return {
            "organ_id": "organ_08_hippocampus",
            "dot_memory_db_exists": self.dot_db_path.is_file(),
            "total_active_memories": total_active_memories,
            "recent_memories": recent_memories[:5],
            "auto_learned_suggestions_count": len(raw_suggestions),
            "staged_candidates_count": len(staged_candidates),
            "staged_candidates": staged_candidates,
            "active_workflows": active_workflows[:5],
            "learned_habits": learned_habits[:5],
        }

    # ------------------------------------------------------------------
    # Pillar 3: Workspace & Git State (Organ 09 - Hands & Active Pipelines)
    # ------------------------------------------------------------------
    def collect_pillar3_workspace(self, channel: str = "morning") -> Dict[str, Any]:
        """
        Inspect live Git branch, porcelain delta count, modified/untracked file summary,
        latest commit subject, and active milestones from `.agents/ACTIVE_PIPELINES.md`.
        """
        terms = _channel_terms(channel)
        branch = "main"
        latest_commit = ""
        latest_commit_subject = ""
        modified_files: List[str] = []
        untracked_files: List[str] = []
        staged_files: List[str] = []
        porcelain_count = 0

        try:
            proc_b = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(self.workspace_root),
                capture_output=True,
                text=True,
                timeout=1.5,
                check=False,
            )
            if proc_b.returncode == 0 and proc_b.stdout.strip():
                branch = proc_b.stdout.strip()
        except Exception:
            pass

        try:
            proc_c = subprocess.run(
                ["git", "log", "-1", "--pretty=format:%h %s"],
                cwd=str(self.workspace_root),
                capture_output=True,
                text=True,
                timeout=1.5,
                check=False,
            )
            if proc_c.returncode == 0 and proc_c.stdout.strip():
                latest_commit = redact_secrets(proc_c.stdout.strip())
                parts = latest_commit.split(" ", 1)
                latest_commit_subject = parts[1].strip() if len(parts) > 1 else latest_commit
        except Exception:
            pass

        try:
            proc_s = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(self.workspace_root),
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            if proc_s.returncode == 0:
                lines = [ln for ln in proc_s.stdout.splitlines() if ln.strip()]
                porcelain_count = len(lines)
                for ln in lines:
                    status_xy = ln[:2]
                    fpath = ln[3:].strip()
                    if status_xy == "??":
                        untracked_files.append(fpath)
                    elif status_xy[0] not in (" ", "?"):
                        staged_files.append(fpath)
                    else:
                        modified_files.append(fpath)
        except Exception:
            pass

        # Parse .agents/ACTIVE_PIPELINES.md
        pipelines_path = self.workspace_root / ".agents" / "ACTIVE_PIPELINES.md"
        if not pipelines_path.is_file():
            pipelines_path = WBC_REPO_ROOT / ".agents" / "ACTIVE_PIPELINES.md"

        active_next_items: List[str] = []
        completed_milestones_count = 0
        last_completed_milestone = "Phase 18.12"
        next_resume_prompt = ""
        queued_resume_prompt = ""
        current_status_banner = ""
        up_next_summary = ""

        if pipelines_path.is_file():
            try:
                content = pipelines_path.read_text(encoding="utf-8", errors="ignore")
                m_status = re.search(r"-\s*\*\*Current Status\*\*:\s*`?([^\n`]+)`?", content)
                if m_status:
                    current_status_banner = redact_secrets(m_status.group(1).strip())

                checked_titles = re.findall(r"^\s*-\s*\[x\]\s*\*\*([^\n*]+)\*\*", content, re.MULTILINE)
                completed_milestones_count = len(checked_titles)
                if checked_titles:
                    # Extract short milestone label e.g. "Phase 18.12" or "Issue #4"
                    for raw_t in reversed(checked_titles):
                        m_ph = re.search(r"\b(Phase\s+\d+(?:\.\d+)?|18\.\d+|Issue\s*#\d+)\b", raw_t, re.IGNORECASE)
                        if m_ph:
                            tok = m_ph.group(1)
                            last_completed_milestone = f"Phase {tok}" if re.match(r"^\d+\.\d+$", tok) else tok
                            break

                unchecked = re.findall(r"^\s*-\s*\[\s*\]\s*\*\*([^\n*]+)\*\*([^\n]*)", content, re.MULTILINE)
                for title_part, rest_part in unchecked:
                    item_str = redact_secrets(f"{title_part.strip()}{rest_part.strip()}")
                    active_next_items.append(item_str)

                m_next = re.search(
                    r"####\s*Next Resume Prompt[^\n]*\n+```(?:markdown|text|bash)?\n(.*?)```",
                    content,
                    re.DOTALL | re.IGNORECASE,
                )
                if m_next:
                    next_resume_prompt = redact_secrets(m_next.group(1).strip())

                m_queued = re.search(
                    r"####\s*Queued Resume Prompt[^\n]*\n+```(?:markdown|text|bash)?\n(.*?)```",
                    content,
                    re.DOTALL | re.IGNORECASE,
                )
                if m_queued:
                    queued_resume_prompt = redact_secrets(m_queued.group(1).strip())
            except Exception as exc:
                logger.debug("ACTIVE_PIPELINES parse fallback: %s", exc)

        if terms and active_next_items:
            scored = []
            for idx, it in enumerate(active_next_items):
                _, sc = _matches_channel(it, terms)
                scored.append((sc, -idx, it))
            scored.sort(reverse=True)
            active_next_items = [it for _, _, it in scored]

        if next_resume_prompt:
            up_next_summary = _summarize_roadmap_prompt(next_resume_prompt)
        elif active_next_items:
            up_next_summary = _summarize_roadmap_prompt(active_next_items[0])
        elif queued_resume_prompt:
            up_next_summary = _summarize_roadmap_prompt(queued_resume_prompt)
        else:
            up_next_summary = "Continue active workspace milestone"

        return {
            "organ_id": "organ_09_workspace",
            "workspace_name": self.workspace_root.name,
            "workspace_path": str(self.workspace_root),
            "git_branch": branch,
            "latest_commit": latest_commit,
            "latest_commit_subject": latest_commit_subject,
            "porcelain_count": porcelain_count,
            "is_dirty": porcelain_count > 0,
            "modified_count": len(modified_files),
            "staged_count": len(staged_files),
            "untracked_count": len(untracked_files),
            "modified_files": modified_files[:8],
            "staged_files": staged_files[:8],
            "untracked_files": untracked_files[:8],
            "current_status_banner": current_status_banner,
            "completed_milestones_count": completed_milestones_count,
            "last_completed_milestone": last_completed_milestone,
            "up_next_summary": up_next_summary,
            "active_next_items": active_next_items[:5],
            "next_resume_prompt": next_resume_prompt,
            "queued_resume_prompt": queued_resume_prompt,
        }

    # ------------------------------------------------------------------
    # Pillar 4: FinOps & Organ Vitals (`body report`)
    # ------------------------------------------------------------------
    def collect_pillar4_finops_vitals(
        self,
        window: str = "24h",
        metrics_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Pull 24h tokens saved, dollars saved ($0.00 Local Metal vs Cloud cost avoided),
        local offload ratio, Blood-Brain Barrier (Organ 05) shield interceptions,
        and cognitive/thermal vitals from `build_canonical_metrics_report`.
        """
        if isinstance(metrics_override, dict):
            report = metrics_override
        else:
            now_ts = time.time()
            cached = _CACHED_BRIEF_METRICS.get(window)
            cached_ts = _CACHED_BRIEF_METRICS_TS.get(window, 0.0)
            if cached and (now_ts - cached_ts) <= 30.0:
                report = cached
            else:
                from whole_body_compute.metrics import build_canonical_metrics_report

                report = build_canonical_metrics_report(window=window, trend_window="7d")
                _CACHED_BRIEF_METRICS[window] = report
                _CACHED_BRIEF_METRICS_TS[window] = now_ts

        p1 = report.get("pillars", {}).get("finops_shield", {}) or report.get("finops_shield", {})
        p2 = report.get("pillars", {}).get("cognitive_health", {}) or report.get("cognitive_health", {})
        p3 = report.get("pillars", {}).get("velocity_quality", {}) or report.get("velocity_quality", {})
        p4 = report.get("pillars", {}).get("hardware_vitals", {}) or report.get("hardware_vitals", {})

        tokens_saved_24h = int(p1.get("cloud_tokens_saved_24h", 0))
        tokens_digested_24h = int(p1.get("tokens_digested_locally_24h", 0))
        usd_saved_24h = float(p1.get("estimated_usd_saved_24h", 0.0))
        usd_saved_7d = float(p1.get("estimated_usd_saved_7d", 0.0))
        retro_30d_usd = float((p1.get("retroactive_30d_roi") or {}).get("potential_usd_saved", 0.0))
        local_offload_pct = float(p1.get("local_execution_pct_24h", 85.0))
        bbb_intercepted_24h = int(p1.get("bbb_intercepted_24h", 0))
        bbb_auto_healed_24h = int(p1.get("bbb_auto_healed_24h", 0))
        bbb_total_calls_24h = int(p1.get("bbb_total_calls_24h", 0))

        active_csi = float(p2.get("active_csi", 0.15))
        active_health_tier = str(p2.get("active_health_tier", "GREEN"))
        hours_reclaimed_24h = float(p3.get("hours_reclaimed_24h", 0.0))
        thermal_state = str(p4.get("thermal_state", "NORMAL"))
        ram_free_gb = float(p4.get("ram_free_gb", 8.0))
        swap_used_gb = float(p4.get("swap_used_gb", 0.0))

        return {
            "organ_id": "organ_10_finops_vitals",
            "window": window,
            "local_metal_cost_usd": 0.0,
            "local_offload_ratio_pct": round(local_offload_pct, 1),
            "tokens_digested_locally_24h": tokens_digested_24h,
            "cloud_tokens_saved_24h": tokens_saved_24h,
            "estimated_usd_saved_24h": round(usd_saved_24h, 2),
            "estimated_usd_saved_7d": round(usd_saved_7d, 2),
            "retroactive_30d_usd_saved": round(retro_30d_usd, 2),
            "bbb_total_calls_24h": bbb_total_calls_24h,
            "bbb_intercepted_24h": bbb_intercepted_24h,
            "bbb_auto_healed_24h": bbb_auto_healed_24h,
            "hours_reclaimed_24h": round(hours_reclaimed_24h, 2),
            "active_csi": round(active_csi, 3),
            "active_health_tier": active_health_tier,
            "thermal_state": thermal_state,
            "ram_free_gb": round(ram_free_gb, 2),
            "swap_used_gb": round(swap_used_gb, 2),
        }

    # ------------------------------------------------------------------
    # Suggested Actions Builder
    # ------------------------------------------------------------------
    def build_suggested_actions(
        self,
        eyes: Dict[str, Any],
        hippocampus: Dict[str, Any],
        workspace: Dict[str, Any],
        finops_vitals: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Synthesize ranked, actionable suggestions across the 4 pillars for the
        Human-in-the-Loop approval gate (`--approve`).
        """
        actions: List[Dict[str, Any]] = []

        # 1. Staged PalWorkflow candidates ready for promotion
        for cand in hippocampus.get("staged_candidates", []):
            cid = str(cand.get("id", "")).strip()
            if not cid:
                continue
            rel = float(cand.get("reliability_score", 1.0))
            actions.append(
                {
                    "action_id": f"promote_candidate:{cid}",
                    "action_type": "promote_candidate",
                    "candidate_id": cid,
                    "title": f"Promote staged PalWorkflow candidate '{cid}' ({rel * 100:.0f}% reliability)",
                    "command": f"body candidates --promote {cid}",
                    "priority": "high",
                    "requires_approval": True,
                }
            )

        # 2. Cognitive Saturation Flush recommendation if CSI >= 0.70 (YELLOW / RED)
        csi = float(finops_vitals.get("active_csi", 0.0))
        tier = str(finops_vitals.get("active_health_tier", "GREEN"))
        if csi >= 0.70 or tier in ("YELLOW", "RED"):
            actions.append(
                {
                    "action_id": "glymphatic_flush",
                    "action_type": "glymphatic_flush",
                    "title": f"Run Ultradian Glymphatic Context Flush (CSI {csi:.2f} {tier})",
                    "command": "body flush --auto",
                    "priority": "high",
                    "requires_approval": True,
                }
            )

        # 3. Active Next Roadmap Milestone (always surface the top roadmap item)
        up_next = str(workspace.get("up_next_summary", "")).strip()
        if up_next:
            actions.append(
                {
                    "action_id": "execute_next_milestone",
                    "action_type": "pipeline_milestone",
                    "title": f"Start next roadmap milestone: {up_next[:95]}",
                    "command": "body context",
                    "priority": "high" if not actions else "medium",
                    "requires_approval": False,
                }
            )

        # 4. Dirty Git working tree checkpoint (proper singular/plural, never '(s)')
        porcelain = int(workspace.get("porcelain_count", 0))
        if porcelain > 0:
            mod_c = int(workspace.get("modified_count", 0))
            untr_c = int(workspace.get("untracked_count", 0))
            change_phrase = _plural(porcelain, "uncommitted workspace change")
            actions.append(
                {
                    "action_id": "inspect_git_deltas",
                    "action_type": "git_hygiene",
                    "title": (
                        f"Review {change_phrase} ({mod_c} modified, {untr_c} untracked) on branch '{workspace.get('git_branch', 'main')}'"
                    ),
                    "command": "git status -s",
                    "priority": "low",
                    "requires_approval": False,
                }
            )

        return actions

    # ------------------------------------------------------------------
    # Standup Collectors: GitHub PR Status & Linear Active Tickets
    # ------------------------------------------------------------------
    def collect_github_state(self, workspace_root: Optional[Path] = None) -> Dict[str, Any]:
        """Collect current GitHub PR and review status via `gh pr status`."""
        return collect_github_pr_status(workspace_root=workspace_root or self.workspace_root)

    _collect_github_state = collect_github_state

    def collect_linear_state(
        self,
        foveal: Optional[Dict[str, Any]] = None,
        eyes: Optional[Dict[str, Any]] = None,
        hippocampus: Optional[Dict[str, Any]] = None,
        workspace: Optional[Dict[str, Any]] = None,
        linear_api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Collect active Linear tickets from desktop window titles, git state, memories, or API."""
        return collect_linear_standup_state(
            foveal=foveal,
            eyes=eyes,
            hippocampus=hippocampus,
            workspace=workspace,
            linear_api_key=linear_api_key,
        )

    _collect_linear_state = collect_linear_state

    # ------------------------------------------------------------------
    # Local Metal Synthesis (:9380 / :9379 / Ollama -> <5ms Deterministic Fallback)
    # ------------------------------------------------------------------
    def synthesize_duet_script(
        self,
        channel: str,
        eyes: Dict[str, Any],
        hippocampus: Dict[str, Any],
        workspace: Dict[str, Any],
        finops_vitals: Dict[str, Any],
        suggested_actions: List[Dict[str, Any]],
        use_local_llm: Optional[bool] = None,
        github: Optional[Dict[str, Any]] = None,
        linear: Optional[Dict[str, Any]] = None,
        timeout_sec: float = 1.8,
    ) -> Dict[str, Any]:
        """
        Synthesize a <30-second (<= 75 words) spoken duet briefing and terminal executive summary.
        Tries warm Local Metal (:9380 gemma4-26b / :9379 voicefi-coder-mlx) at $0.00
        when `use_local_llm=True`, falling back in <5ms to deterministic 4-beat conversational synthesis.
        """
        t0 = time.perf_counter()
        if use_local_llm is None:
            use_local_llm = bool(os.environ.get("WBC_BRIEF_USE_LOCAL_LLM") == "1")

        det_spoken = self._build_deterministic_spoken_script(
            channel=channel,
            eyes=eyes,
            hippocampus=hippocampus,
            workspace=workspace,
            finops_vitals=finops_vitals,
            suggested_actions=suggested_actions,
            github=github,
            linear=linear,
        )

        engine_used = "deterministic_local_template"
        model_used = "tier0-deterministic-template"
        spoken_script = det_spoken

        if use_local_llm:
            llm_text, llm_model, llm_engine = self._try_local_metal_completion(
                channel=channel,
                eyes=eyes,
                hippocampus=hippocampus,
                workspace=workspace,
                finops_vitals=finops_vitals,
                suggested_actions=suggested_actions,
                github=github,
                linear=linear,
                timeout_sec=timeout_sec,
            )
            if llm_text and _is_valid_spoken_briefing(llm_text):
                spoken_script = _clean_for_speech(llm_text, max_words=MAX_SPOKEN_WORDS)
                model_used = llm_model
                engine_used = llm_engine

        spoken_script = _clean_for_speech(spoken_script, max_words=MAX_SPOKEN_WORDS)
        word_count = len(spoken_script.split())
        estimated_duration_sec = round(word_count / 2.6, 1)
        synthesis_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "spoken_script": spoken_script,
            "word_count": word_count,
            "max_words_budget": MAX_SPOKEN_WORDS,
            "estimated_duration_sec": estimated_duration_sec,
            "within_30s_sla": estimated_duration_sec <= 30.0 and word_count <= 75,
            "engine": engine_used,
            "model": model_used,
            "synthesis_latency_ms": synthesis_ms,
            "cloud_cost_usd": 0.0,
        }

    def _build_deterministic_spoken_script(
        self,
        channel: str,
        eyes: Dict[str, Any],
        hippocampus: Dict[str, Any],
        workspace: Dict[str, Any],
        finops_vitals: Dict[str, Any],
        suggested_actions: List[Dict[str, Any]],
        github: Optional[Dict[str, Any]] = None,
        linear: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build a natural, 4-beat conversational executive briefing (~45-65 words)
        optimized for human ears and VoiceFi TTS playback:
        - Beat 1: Where you are & what's on screen
        - Beat 2: What you just finished & what is up next on the roadmap (plus active Linear/PR status)
        - Beat 3: Workflows ready to approve OR context/memory health
        - Beat 4: 24h Local Metal savings in plain spoken English
        """
        ch_low = channel.strip().lower()
        greeting = (
            "Good morning."
            if ch_low in ("morning", "all", "default", "duet")
            else f"Here is your {channel.title()} briefing."
        )
        ws_name = workspace.get("workspace_name") or "Whole Body Compute"
        app_focus = eyes.get("foveal_focus", {}).get("app_name") or "Terminal"
        branch = workspace.get("git_branch", "main")
        mod_count = int(workspace.get("modified_count", 0))
        porcelain = int(workspace.get("porcelain_count", 0))

        if mod_count > 0:
            git_phrase = f"and {_plural(mod_count, 'modified file')}"
        elif porcelain > 0:
            git_phrase = f"and {_plural(porcelain, 'uncommitted file')}"
        else:
            git_phrase = "and a clean working tree"

        # Beat 1: Where you are & what's on screen
        beat1 = (
            f"{greeting} You are in {ws_name} on branch {branch}, "
            f"with {app_focus} active on screen {git_phrase}."
        )

        # Standup phrase (Linear & GitHub PR status)
        linear_info = linear or {}
        github_info = github or {}
        primary_issue = linear_info.get("primary_issue")
        linear_id = primary_issue.get("id") if primary_issue else ""
        needs_rev_cnt = int(github_info.get("needs_review_count", 0))

        standup_phrase = ""
        if linear_id and needs_rev_cnt > 0:
            standup_phrase = f"Linear ticket {linear_id} is open, with {_plural(needs_rev_cnt, 'pull request')} awaiting review."
        elif linear_id:
            standup_phrase = f"Linear ticket {linear_id} is open."
        elif needs_rev_cnt > 0:
            standup_phrase = f"You have {_plural(needs_rev_cnt, 'pull request')} awaiting review."

        # Beat 2: What you just finished & what is up next on the roadmap
        last_done = workspace.get("last_completed_milestone") or "Phase 18.12"
        up_next = workspace.get("up_next_summary") or "continuing your active pipeline"
        if standup_phrase:
            beat2 = f"{last_done} is complete. {standup_phrase} Up next on your roadmap is {up_next}."
        else:
            beat2 = f"{last_done} is complete. Up next on your roadmap is {up_next}."

        # Beat 3: Staged workflow candidates OR Context & RAM health
        staged_count = int(hippocampus.get("staged_candidates_count", 0))
        if staged_count > 0:
            beat3 = f"You have {_plural(staged_count, 'auto-learned workflow')} ready to approve."
        else:
            health_tier = str(finops_vitals.get("active_health_tier", "GREEN")).lower()
            ram_gb = max(1, int(round(float(finops_vitals.get("ram_free_gb", 16.0)))))
            beat3 = f"Context health is {health_tier} with {_plural(ram_gb, 'gigabyte')} of free memory."

        # Beat 4: 24h Savings in Plain English
        spoken_usd = _format_spoken_dollars(float(finops_vitals.get("estimated_usd_saved_24h", 0.0)))
        spoken_tokens = _format_spoken_number(int(finops_vitals.get("cloud_tokens_saved_24h", 0)))
        beat4 = (
            f"In the last 24 hours, local Metal saved {spoken_usd} "
            f"and {spoken_tokens} cloud tokens at zero cloud cost."
        )

        return _clean_for_speech(f"{beat1} {beat2} {beat3} {beat4}", max_words=MAX_SPOKEN_WORDS)

    def _try_local_metal_completion(
        self,
        channel: str,
        eyes: Dict[str, Any],
        hippocampus: Dict[str, Any],
        workspace: Dict[str, Any],
        finops_vitals: Dict[str, Any],
        suggested_actions: List[Dict[str, Any]],
        github: Optional[Dict[str, Any]] = None,
        linear: Optional[Dict[str, Any]] = None,
        timeout_sec: float = 1.8,
    ) -> Tuple[Optional[str], str, str]:
        """
        Attempt fast sub-2s synthesis against warm Local Metal (:9380 SpecialistRouter,
        :9379 voicefi-coder-mlx). Returns (text, model, engine).
        """
        spoken_usd = _format_spoken_dollars(float(finops_vitals.get("estimated_usd_saved_24h", 0.0)))
        spoken_tokens = _format_spoken_number(int(finops_vitals.get("cloud_tokens_saved_24h", 0)))
        up_next = workspace.get("up_next_summary") or "continuing active pipeline"

        standup_context = ""
        linear_info = linear or {}
        github_info = github or {}
        primary_issue = linear_info.get("primary_issue")
        if primary_issue and primary_issue.get("id"):
            standup_context += f" Linear ticket {primary_issue['id']} is open."
        if github_info.get("needs_review_count", 0) > 0:
            standup_context += f" {github_info['needs_review_count']} PRs awaiting review."

        prompt = (
            f"Write a warm, clear 4-sentence spoken morning executive briefing (strictly 45 to 65 words, "
            f"plain conversational English for TTS, no markdown, no parentheses, no decimals) for '{channel}'. "
            f"1) Workspace: {workspace.get('workspace_name')} on branch {workspace.get('git_branch')}, "
            f"active app {eyes.get('foveal_focus', {}).get('app_name')}, {workspace.get('modified_count')} modified files. "
            f"2) Just completed {workspace.get('last_completed_milestone')}.{standup_context} Up next on roadmap is {up_next}. "
            f"3) Context health is {finops_vitals.get('active_health_tier')} with {int(round(float(finops_vitals.get('ram_free_gb', 16))))} gigabytes free memory. "
            f"4) Last 24 hours on local Metal saved {spoken_usd} and {spoken_tokens} cloud tokens."
        )

        for port, default_model, eng_name in (
            (9380, "gemma4-26b", "mlx_metal_9380"),
            (9379, "voicefi-coder-mlx", "mlx_metal_9379"),
        ):
            url = f"http://127.0.0.1:{port}/v1/chat/completions"
            payload = json.dumps(
                {
                    "model": default_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 110,
                    "temperature": 0.2,
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    choices = data.get("choices") or []
                    if choices:
                        content = (choices[0].get("message") or {}).get("content", "").strip()
                        if content:
                            return content, str(data.get("model", default_model)), eng_name
            except Exception:
                continue

        return None, "tier0-deterministic-template", "deterministic_local_template"

    # ------------------------------------------------------------------
    # Full 4-Pillar Briefing Assembly
    # ------------------------------------------------------------------
    def build_briefing(
        self,
        channel: str = "morning",
        window: str = "24h",
        use_local_llm: Optional[bool] = None,
        screen_override: Optional[Dict[str, Any]] = None,
        foveal_override: Optional[Dict[str, Any]] = None,
        suggestions_override: Optional[List[Dict[str, Any]]] = None,
        metrics_override: Optional[Dict[str, Any]] = None,
        github_override: Optional[Dict[str, Any]] = None,
        linear_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the complete 4-Pillar Sovereign Duet Briefing payload with Standup expansion."""
        t0 = time.perf_counter()
        norm_channel = (channel or "morning").strip() or "morning"

        eyes = self.collect_pillar1_eyes(
            channel=norm_channel,
            screen_override=screen_override,
            foveal_override=foveal_override,
        )
        hippocampus = self.collect_pillar2_hippocampus(
            channel=norm_channel,
            suggestions_override=suggestions_override,
        )
        workspace = self.collect_pillar3_workspace(channel=norm_channel)
        finops_vitals = self.collect_pillar4_finops_vitals(
            window=window,
            metrics_override=metrics_override,
        )
        github = (
            github_override
            if github_override is not None
            else self.collect_github_state()
        )
        linear = (
            linear_override
            if linear_override is not None
            else self.collect_linear_state(
                foveal=eyes.get("foveal_focus"),
                eyes=eyes,
                hippocampus=hippocampus,
                workspace=workspace,
            )
        )
        suggested_actions = self.build_suggested_actions(
            eyes=eyes,
            hippocampus=hippocampus,
            workspace=workspace,
            finops_vitals=finops_vitals,
        )
        synthesis = self.synthesize_duet_script(
            channel=norm_channel,
            eyes=eyes,
            hippocampus=hippocampus,
            workspace=workspace,
            finops_vitals=finops_vitals,
            suggested_actions=suggested_actions,
            use_local_llm=use_local_llm,
            github=github,
            linear=linear,
        )

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "status": "ok",
            "phase": "18.12",
            "title": "Sovereign Duet on Local Metal",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "channel": norm_channel,
            "window": window,
            "elapsed_ms": elapsed_ms,
            "cloud_cost_usd": 0.0,
            "pillars": {
                "eyes": eyes,
                "hippocampus": hippocampus,
                "workspace": workspace,
                "finops_vitals": finops_vitals,
            },
            "github": github,
            "linear": linear,
            "synthesis": synthesis,
            "spoken_script": synthesis["spoken_script"],
            "suggested_actions": suggested_actions,
            "approval_results": {
                "enabled": False,
                "approved_count": 0,
                "skipped_count": 0,
                "promoted_candidates": [],
                "decisions": [],
            },
            "speech_playback": {
                "requested": False,
                "spoken": False,
                "backend": "none",
            },
        }


def build_sovereign_briefing(
    channel: str = "morning",
    window: str = "24h",
    workspace_root: Optional[Path] = None,
    dot_db_path: Optional[Path] = None,
    screenpipe_db_path: Optional[Path] = None,
    use_local_llm: Optional[bool] = None,
    screen_override: Optional[Dict[str, Any]] = None,
    foveal_override: Optional[Dict[str, Any]] = None,
    suggestions_override: Optional[List[Dict[str, Any]]] = None,
    metrics_override: Optional[Dict[str, Any]] = None,
    github_override: Optional[Dict[str, Any]] = None,
    linear_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convenience entrypoint to build the 4-Pillar Sovereign Duet Briefing."""
    engine = SovereignDuetBriefingEngine(
        workspace_root=workspace_root,
        dot_db_path=dot_db_path,
        screenpipe_db_path=screenpipe_db_path,
    )
    return engine.build_briefing(
        channel=channel,
        window=window,
        use_local_llm=use_local_llm,
        screen_override=screen_override,
        foveal_override=foveal_override,
        suggestions_override=suggestions_override,
        metrics_override=metrics_override,
        github_override=github_override,
        linear_override=linear_override,
    )


def speak_briefing(
    spoken_script: str,
    voice: str = "Christopher",
    speed: float = 1.35,
    runner: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """
    Speak the concise <30-second duet script aloud via VoiceFi TTS (`vifi speak`)
    with graceful fallback if `vifi` daemon/CLI is offline or during headless tests.
    """
    clean_script = _clean_for_speech(spoken_script, max_words=MAX_SPOKEN_WORDS)
    word_count = len(clean_script.split())
    in_pytest = bool(os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("VOICEFI_TESTING") == "1")

    if runner is not None:
        try:
            res = runner(clean_script, voice=voice, speed=speed)
            return {
                "requested": True,
                "spoken": True,
                "backend": "custom_runner",
                "voice": voice,
                "speed": speed,
                "word_count": word_count,
                "script": clean_script,
                "runner_result": res,
            }
        except Exception as exc:
            return {
                "requested": True,
                "spoken": False,
                "backend": "custom_runner_error",
                "voice": voice,
                "speed": speed,
                "word_count": word_count,
                "script": clean_script,
                "error": str(exc),
            }

    if in_pytest:
        return {
            "requested": True,
            "spoken": True,
            "backend": "test_dry_run",
            "voice": voice,
            "speed": speed,
            "word_count": word_count,
            "script": clean_script,
        }

    vifi_bin = shutil.which("vifi")
    if not vifi_bin:
        for cand in (
            str(Path(os.sys.executable).resolve().parent / "vifi"),
            os.path.expanduser("~/.local/bin/vifi"),
            "/opt/homebrew/bin/vifi",
        ):
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                vifi_bin = cand
                break

    if vifi_bin:
        try:
            speed_arg = f"{speed}x" if not str(speed).endswith("x") else str(speed)
            proc = subprocess.run(
                [vifi_bin, "speak", clean_script, "-v", voice, "-s", speed_arg],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=45.0,
                check=False,
            )
            if proc.returncode == 0:
                return {
                    "requested": True,
                    "spoken": True,
                    "backend": "vifi_speak",
                    "voice": voice,
                    "speed": speed,
                    "word_count": word_count,
                    "script": clean_script,
                }
        except Exception as exc:
            logger.debug("vifi speak fallback: %s", exc)

    say_bin = shutil.which("say") or ("/usr/bin/say" if os.path.isfile("/usr/bin/say") else None)
    if say_bin:
        try:
            wpm = str(max(160, int(round(175 * float(speed or 1.25)))))
            proc_say = subprocess.run(
                [say_bin, "-r", wpm, clean_script],
                capture_output=True,
                text=True,
                timeout=45.0,
                check=False,
            )
            if proc_say.returncode == 0:
                return {
                    "requested": True,
                    "spoken": True,
                    "backend": "macos_say",
                    "voice": voice,
                    "speed": speed,
                    "word_count": word_count,
                    "script": clean_script,
                }
        except Exception as exc:
            logger.debug("macos say fallback: %s", exc)

    return {
        "requested": True,
        "spoken": False,
        "backend": "offline_fallback",
        "voice": voice,
        "speed": speed,
        "word_count": word_count,
        "script": clean_script,
    }


def run_interactive_approval_gate(
    briefing: Dict[str, Any],
    auto_approve: bool = False,
    input_fn: Optional[Callable[[str], str]] = None,
    promote_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Human-in-the-Loop approval gate (`--approve`) for suggested actions and staged
    `PalWorkflow` candidate promotions.
    """
    if promote_fn is None:
        from whole_body_compute.integrations.pal_workflow import PalWorkflowRegistry

        promote_fn = PalWorkflowRegistry.promote_candidate

    actions = briefing.get("suggested_actions", []) or []
    decisions: List[Dict[str, Any]] = []
    promoted_candidates: List[Dict[str, Any]] = []
    approved_count = 0
    skipped_count = 0

    for action in actions:
        action_id = str(action.get("action_id", ""))
        action_type = str(action.get("action_type", ""))
        title = str(action.get("title", action_id))
        cand_id = str(action.get("candidate_id", ""))

        approved = False
        if auto_approve:
            approved = True
        elif input_fn is not None:
            try:
                ans = str(input_fn(f"Approve action '{title}'? [y/N]: ") or "").strip().lower()
                approved = ans in ("y", "yes", "approve", "1", "true")
            except (EOFError, KeyboardInterrupt):
                approved = False
        else:
            if os.environ.get("PYTEST_CURRENT_TEST") or not os.isatty(0):
                approved = False
            else:
                try:
                    ans = input(f"  ⚡ Approve '{title}'? [y/N]: ").strip().lower()
                    approved = ans in ("y", "yes", "approve", "1", "true")
                except (EOFError, KeyboardInterrupt):
                    approved = False

        decision_entry: Dict[str, Any] = {
            "action_id": action_id,
            "action_type": action_type,
            "title": title,
            "approved": approved,
            "status": "approved" if approved else "skipped",
        }

        if approved:
            approved_count += 1
            if action_type == "promote_candidate" and cand_id:
                try:
                    promo_res = promote_fn(cand_id)
                    decision_entry["promotion_result"] = promo_res
                    promoted_candidates.append(
                        {
                            "candidate_id": cand_id,
                            "result": promo_res,
                        }
                    )
                except Exception as exc:
                    decision_entry["status"] = "error"
                    decision_entry["error"] = str(exc)
        else:
            skipped_count += 1

        decisions.append(decision_entry)

    approval_summary = {
        "enabled": True,
        "auto_approve": auto_approve,
        "total_actions": len(actions),
        "approved_count": approved_count,
        "skipped_count": skipped_count,
        "promoted_candidates": promoted_candidates,
        "decisions": decisions,
    }
    briefing["approval_results"] = approval_summary
    return approval_summary


def format_briefing_spoken_script(briefing: Dict[str, Any]) -> str:
    """Return the <30-second spoken script string from a briefing payload."""
    return str(briefing.get("spoken_script") or briefing.get("synthesis", {}).get("spoken_script", ""))


def format_briefing_terminal(briefing: Dict[str, Any]) -> str:
    """Format the 4-Pillar Sovereign Duet Briefing as a crisp, human-readable terminal dashboard."""
    pillars = briefing.get("pillars", {})
    eyes = pillars.get("eyes", {})
    hippo = pillars.get("hippocampus", {})
    ws = pillars.get("workspace", {})
    fin = pillars.get("finops_vitals", {})
    syn = briefing.get("synthesis", {})
    actions = briefing.get("suggested_actions", [])
    approvals = briefing.get("approval_results", {})
    speech = briefing.get("speech_playback", {})

    foveal = eyes.get("foveal_focus", {})
    app_name = foveal.get("app_name", "Mac")
    win_title = foveal.get("window_title", "")
    focus_label = f"{app_name} - {win_title}" if win_title and win_title != app_name else app_name

    last_done = ws.get("last_completed_milestone") or "Phase 18.12"
    up_next = ws.get("up_next_summary") or "Continue active workspace milestone"
    mod_count = int(ws.get("modified_count", 0))
    untr_count = int(ws.get("untracked_count", 0))
    porcelain_count = int(ws.get("porcelain_count", 0))

    tokens_compact = _format_compact_number(fin.get("cloud_tokens_saved_24h", 0))
    usd_24h = float(fin.get("estimated_usd_saved_24h", 0.0))

    lines: List[str] = []
    lines.append("")
    lines.append(
        f"🌅 [SOVEREIGN DUET ON LOCAL METAL - CHANNEL: {str(briefing.get('channel', 'morning')).upper()} · $0.00 CLOUD COST]"
    )
    lines.append("═" * 88)

    # Top Executive Summary Block (Plain English: Where You Left Off / Up Next / Savings)
    lines.append("  📋 EXECUTIVE SUMMARY (WHAT HAPPENED & WHAT'S NEXT)")
    lines.append(
        f"     • Where You Left Off: {last_done} completed on `{ws.get('git_branch', 'main')}` "
        f"| Active Window: {focus_label} ({mod_count} modified, {untr_count} untracked)"
    )
    lines.append(f"     • Up Next on Roadmap: {up_next}")
    lines.append(
        f"     • 24h Local Savings:  ${usd_24h:,.2f} saved ({tokens_compact} cloud tokens avoided at $0.00) "
        f"| Context CSI: {fin.get('active_csi', 0.0):.2f} ({fin.get('active_health_tier', 'GREEN')}) "
        f"| Free RAM: {fin.get('ram_free_gb', 0.0):.1f} GB"
    )
    lines.append("─" * 88)

    # Pillar 1: Eyes
    frames_cnt = int(eyes.get("recent_frames_count", 0))
    lines.append(
        f"  👁️  Pillar 1 · The Eyes (Organ 07):      Focus: {focus_label}"
    )
    lines.append(
        f"     • Foveal Recall: {_plural(frames_cnt, 'recent frame')} ({eyes.get('screenpipe_source', 'fallback')}) "
        f"| Active Apps: {', '.join(eyes.get('active_apps', [])) or 'none'} | Secret Redaction: ON"
    )

    # Pillar 2: Hippocampus
    mem_cnt = int(hippo.get("total_active_memories", 0))
    cand_cnt = int(hippo.get("staged_candidates_count", 0))
    lines.append(
        f"  🧠 Pillar 2 · Hippocampus (Organ 08):   {_plural(mem_cnt, 'active memory', 'active memories')} | "
        f"{_plural(cand_cnt, 'staged PalWorkflow candidate')}"
    )
    for mem in (hippo.get("recent_memories") or [])[:2]:
        lines.append(f"     • [{mem.get('category', 'fact')}] {mem.get('fact', '')[:105]}")
    for cand in (hippo.get("staged_candidates") or [])[:2]:
        lines.append(
            f"     • [Staged Candidate] {cand.get('id')}: {cand.get('title')} "
            f"({float(cand.get('reliability_score', 1.0)) * 100:.0f}% reliability)"
        )

    # Pillar 3: Workspace & Git State
    lines.append(
        f"  🦴 Pillar 3 · Workspace & Git (09):     Branch `{ws.get('git_branch', 'main')}` | "
        f"{_plural(porcelain_count, 'uncommitted change')} "
        f"({mod_count} modified, {int(ws.get('staged_count', 0))} staged, {untr_count} untracked)"
    )
    if ws.get("latest_commit"):
        lines.append(f"     • Latest Commit: {ws.get('latest_commit')}")
    lines.append(f"     • Up Next:       {up_next[:105]}")

    # Pillar 4: FinOps & Organ Vitals
    lines.append(
        f"  💰 Pillar 4 · FinOps & Vitals (10):     "
        f"{tokens_compact} tokens saved (${usd_24h:,.2f} saved / 24h) | "
        f"{fin.get('local_offload_ratio_pct', 85.0):.1f}% Local Metal ($0.00)"
    )
    lines.append(
        f"     • BBB Shield: {fin.get('bbb_intercepted_24h', 0)} intercepted / {fin.get('bbb_auto_healed_24h', 0)} auto-healed | "
        f"CSI: {fin.get('active_csi', 0.0):.2f} ({fin.get('active_health_tier', 'GREEN')}) | "
        f"Thermal: {fin.get('thermal_state', 'NORMAL')} ({fin.get('ram_free_gb', 0.0):.1f} GB free)"
    )

    # Standup: Linear & GitHub PRs
    github = briefing.get("github", {})
    linear = briefing.get("linear", {})
    linear_issues = linear.get("active_issues", [])
    primary_issue = linear.get("primary_issue")
    gh_avail = bool(github.get("available", False))
    curr_pr = github.get("current_branch_pr")
    needs_rev = github.get("needs_review", [])
    needs_cnt = int(github.get("needs_review_count", len(needs_rev)))
    created_cnt = int(github.get("created_open_count", 0))

    lines.append("  🎫 Standup (Linear & GitHub PRs):")
    if primary_issue:
        lines.append(
            f"     • Linear Issue:  {primary_issue.get('id')} · {primary_issue.get('title')} "
            f"[{primary_issue.get('source', 'linear')}]"
        )
        for other in linear_issues[1:3]:
            lines.append(f"       - {other.get('id')} · {other.get('title')}")
    else:
        lines.append("     • Linear Issue:  No active tickets detected")

    if gh_avail:
        if curr_pr:
            draft_flag = " [Draft]" if curr_pr.get("is_draft") else ""
            rev_dec = f" · {curr_pr.get('review_decision')}" if curr_pr.get("review_decision") else ""
            lines.append(
                f"     • Branch PR:     #{curr_pr.get('number')} ({curr_pr.get('state', 'OPEN')}){draft_flag} "
                f"{curr_pr.get('title')}{rev_dec}"
            )
        else:
            lines.append(
                f"     • Branch PR:     None on `{ws.get('git_branch', 'main')}` "
                f"({_plural(created_cnt, 'open PR')} created)"
            )
        if needs_cnt > 0:
            lines.append(f"     • Review Queue:  {_plural(needs_cnt, 'PR')} awaiting your review")
            for rev_pr in needs_rev[:2]:
                lines.append(f"       - #{rev_pr.get('number')}: {rev_pr.get('title')}")
        else:
            lines.append("     • Review Queue:  0 PRs awaiting your review")
    else:
        lines.append("     • GitHub PRs:    Status unavailable (gh cli offline or unauthenticated)")

    lines.append("─" * 88)
    lines.append(
        f"  🎙️  Spoken Briefing ({syn.get('word_count', 0)} words · ~{syn.get('estimated_duration_sec', 0.0):.1f}s "
        f"· Engine: {syn.get('model', 'local')}):"
    )
    lines.append(f"     \"{briefing.get('spoken_script', '')}\"")

    if speech.get("requested"):
        lines.append(
            f"     🔊 Playback: {'SPOKEN' if speech.get('spoken') else 'FALLBACK'} via {speech.get('backend', 'none')}"
        )

    if actions:
        lines.append("─" * 88)
        lines.append("  ⚡ Suggested Next Actions:")
        for idx, act in enumerate(actions, 1):
            lines.append(f"     {idx}. [{act.get('priority', 'medium').upper()}] {act.get('title')} -> `{act.get('command')}`")

    if approvals.get("enabled"):
        lines.append("─" * 88)
        promo_cnt = len(approvals.get("promoted_candidates", []))
        lines.append(
            f"  ✅ Approval Gate: {approvals.get('approved_count', 0)} approved, "
            f"{approvals.get('skipped_count', 0)} skipped "
            f"({_plural(promo_cnt, 'candidate')} promoted)"
        )

    lines.append("═" * 88)
    lines.append("")
    return "\n".join(lines)
