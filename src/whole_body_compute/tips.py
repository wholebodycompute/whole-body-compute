"""
Whole Body Compute (`body tips`) - Terminal Pro-Tips Playbook, Agent Rules & Contextual Nudges.
"""

from __future__ import annotations

import os
import random
import webbrowser
from typing import Any

DOCS_URL = "https://wholebodycompute.com/docs"

PRO_TIPS: list[dict[str, Any]] = [
    {
        "id": "git-commit",
        "number": 1,
        "topic": "git",
        "keywords": ("git", "commit", "checkpoint", "diff", "lock", "multi-agent"),
        "title": "Never ask your cloud agent to 'just git commit'",
        "tier": "Free Forever",
        "impact": "85-90% faster (~0.8s) • $0 diff bloat • 0 index.lock crashes",
        "why": (
            "Cloud agent commit loops take 3-5 turns and upload 4k-18k diff tokens "
            "that get re-billed on every later turn."
        ),
        "commands": [
            "body checkpoint pre-refactor",
            "body commit",
        ],
    },
    {
        "id": "session-hygiene",
        "number": 2,
        "topic": "context",
        "keywords": ("context", "csi", "flush", "handoff", "slow", "bloat", "capsule"),
        "title": "Beat 'late-thread brain fog' before Turn 50",
        "tier": "CSI Free / Auto-Flush Pro",
        "impact": "98.15% compaction • 4.76x faster handoff • 100% state recall",
        "why": (
            "Quadratic transformer attention makes 500KB+ threads slow and forgetful. "
            "Flush when CSI hits YELLOW (>=0.70) or RED (>=1.00)."
        ),
        "commands": [
            "body context --insights",
            "body flush --auto",
        ],
    },
    {
        "id": "zero-cursor-desktop",
        "number": 3,
        "topic": "desktop",
        "keywords": ("desktop", "vision", "eyes", "hands", "chrome", "browser", "cursor", "ui", "dom"),
        "title": "Control live apps & logged-in Chrome without moving your mouse",
        "tier": "Free Forever",
        "impact": "<15ms DOM vs 7,650+ token screenshots (-96.9% tokens)",
        "why": (
            "Inspect live macOS Accessibility DOM elements directly and actuate in the "
            "background with --no-cursor while keeping Terminal/IDEs at zero TCC permissions."
        ),
        "commands": [
            "body eyes --install-app",
            'body eyes --app "Google Chrome" --filter "Submit"',
            'body hands --app "Google Chrome" --no-cursor --chain "click:e5,paste:Ready,hotkey:enter"',
        ],
    },
    {
        "id": "ast-voice-dictation",
        "number": 4,
        "topic": "voice",
        "keywords": ("voice", "ears", "dictate", "transcribe", "stt", "fn", "vocab", "speech"),
        "title": "Dictate code for $0 with Hold Fn & snap real repo symbols",
        "tier": "Free Forever",
        "impact": "$0/mo (vs $15/mo cloud apps) • 0 cloud audio • Exact AST symbols",
        "why": (
            "Hydrates your repo's live symbols.db AST in <5ms so saying 'spinal reflex gate' "
            "types `SpinalReflexGate` right the first time."
        ),
        "commands": [
            "body ears --install",
            'body ears --add-word "StateCapsule"',
            "body ears --status",
        ],
    },
    {
        "id": "local-metal-scout",
        "number": 5,
        "topic": "metal",
        "keywords": ("metal", "scout", "implement", "local", "slice", "logs", "ast"),
        "title": "Optional: Pre-digest multi-megabyte .log dumps locally for $0",
        "tier": "Opt-In (Ollama / MLX)",
        "impact": "85%-98.7% raw log cut at $0.00 local cost",
        "why": (
            "Normal source files are already sliced in <3ms via native Read (0 GB RAM). "
            "If you run Ollama/MLX locally, use `body scout` on huge raw logs or `--slice` for pure AST slices."
        ),
        "commands": [
            'body scout server.log -q "What caused the 502 spike?"',
            "body scout --slice src/auth.py:verify_token src/db.py:Session",
        ],
    },
    {
        "id": "airlock-blob-vault",
        "number": 6,
        "topic": "bbb",
        "keywords": ("bbb", "immune", "blob", "airlock", "vault", "pytest", "truncated"),
        "title": "Recover exact raw lines from compacted logs in <5ms",
        "tier": "Free Forever",
        "impact": "<5ms lossless recovery • 0% prompt-prefix cache break",
        "why": (
            "Every compacted CLI output stores a secret-redacted copy in airlock_blobs/ "
            "tagged with [BLOB:<id>] so you never have to re-run a slow test suite."
        ),
        "commands": [
            "body bbb --window 24h --tools",
            "body bbb --blob <blob_id> --lines 120:180",
        ],
    },
    {
        "id": "ambient-candidates",
        "number": 7,
        "topic": "workflows",
        "keywords": ("workflows", "candidates", "recipes", "hub", "promote", "pal"),
        "title": "Compile repeated cross-app chores into $0, 0.37s skills",
        "tier": "Discovery Free / Promote Pro",
        "impact": "0.37s deterministic execution • 100% cloud token elimination",
        "why": (
            "Passively spots recurring 3+ step cross-app habits (blindfolding 1Password/Keychain) "
            "and stages YAML candidates you can promote into portable skills."
        ),
        "commands": [
            "body candidates --discover",
            "body candidates --promote <candidate_id> --as-skill",
        ],
    },
    {
        "id": "prune-schema-tax",
        "number": 8,
        "topic": "skills",
        "keywords": ("skills", "mcp", "prune", "schema", "turn-0", "quota"),
        "title": "Shelve dormant MCP servers & unused skills eating Turn-0 quota",
        "tier": "Free Forever",
        "impact": "Reclaims 3,000-18,000+ Turn-0 schema tokens per session",
        "why": (
            "Unused MCP servers and dormant skills re-bill their tool schemas on every turn. "
            "`body skills audit` scans your transcripts, then `prune` / `prune-mcp` moves 0-use "
            "items to cold storage (`body skills restore <name>` brings anything back). "
            'Agent prompt: "Run `body skills audit` and `body skills prune --dry-run`, '
            'show me what is 0-use, and prune it if I approve."'
        ),
        "commands": [
            "body skills audit",
            "body skills prune --dry-run",
            "body skills prune-mcp --dry-run",
        ],
    },
]

AGENT_RULES_SNIPPET = """# Whole Body Compute (`body`) Power-User Rules
- **Commits & Checkpoints**: Prefer `body commit` (or `body commit <paths> -m "..."`) instead of multi-turn `git status` + `git diff` + `git add` + `git commit` loops. Before risky refactors, run `body checkpoint <label>`.
- **Read Code Natively (Zero-RAM AST Shield)**: Call native `Read` / `read_file` directly on source files instead of `cat`/`head`/`sed` shell loops. The `<3ms` Blood-Brain Barrier hook automatically slices oversized files with `0 GB` local LLM RAM.
- **Compacted Logs (`[BLOB:<id>]`)**: When a test or CLI log shows `[BLOB:<id>]`, recover exact lines locally via `body bbb --blob <id> --lines START:END` instead of re-running the full test suite.
- **Live macOS Apps & Logged-In Chrome (macOS)**: Prefer `body eyes --app "..."` and `body hands --app "..." --no-cursor` over full-screen screenshots (`screencapture`) or AppleScript."""


def _is_headless_or_testing() -> bool:
    for key in ("WBC_HEADLESS", "WBC_TESTING", "VOICEFI_HEADLESS", "VOICEFI_TESTING", "PYTEST_CURRENT_TEST"):
        if os.environ.get(key):
            return True
    return False


def filter_tips(topic: str | None = None) -> list[dict[str, Any]]:
    """Filter Pro Tips by topic slug, keyword, or substring."""
    if not topic:
        return list(PRO_TIPS)
    q = topic.strip().lower()
    if not q or q == "all":
        return list(PRO_TIPS)

    matched: list[dict[str, Any]] = []
    for tip in PRO_TIPS:
        if (
            q == tip["topic"]
            or q == tip["id"]
            or any(q in kw or kw in q for kw in tip["keywords"])
            or q in tip["title"].lower()
            or q in tip["why"].lower()
            or any(q in cmd.lower() for cmd in tip["commands"])
        ):
            matched.append(tip)
    return matched


def open_docs_in_browser(url: str = DOCS_URL) -> dict[str, Any]:
    """Open the web docs & pro-tips guide unless running under headless/test guards."""
    if _is_headless_or_testing():
        return {"opened": False, "url": url, "reason": "headless_or_testing"}
    opened = bool(webbrowser.open(url))
    return {"opened": opened, "url": url}


def format_tips_playbook(
    topic: str | None = None,
    random_one: bool = False,
    rng_seed: int | None = None,
) -> str:
    """Format the terminal Pro-Tips playbook."""
    tips = filter_tips(topic)
    if not tips:
        valid_topics = ", ".join(sorted({t["topic"] for t in PRO_TIPS}))
        return (
            f"No Pro Tips matched '{topic}'.\n"
            f"Available topics: {valid_topics}\n"
            f"Web guide & interactive CLI explorer: {DOCS_URL}"
        )

    if random_one:
        rng = random.Random(rng_seed)
        tips = [rng.choice(tips)]

    lines: list[str] = [
        "⚡ WHOLE BODY COMPUTE (`body`) - TOOLS & PRO TIPS PLAYBOOK",
        f"   Web Guide & Interactive CLI Explorer: {DOCS_URL}",
        "─" * 78,
    ]
    for tip in tips:
        lines.append(f"#{tip['number']} {tip['title']}  [{tip['tier']}]")
        lines.append(f"   Impact : {tip['impact']}")
        lines.append(f"   Why    : {tip['why']}")
        for cmd in tip["commands"]:
            lines.append(f"   $ {cmd}")
        lines.append("")

    lines.append("─" * 78)
    lines.append("💡 Copy rules for CLAUDE.md / .cursorrules : `body tips --rules`")
    lines.append(f"🌐 Open interactive web guide in browser   : `body tips --open` ({DOCS_URL})")
    return "\n".join(lines)


def get_contextual_tip_for_savings(summary: dict[str, Any]) -> str:
    """Return a concise 1-line contextual Pro Tip tailored to a `body savings` summary."""
    total_events = int(summary.get("total_events") or summary.get("interceptions") or 0)
    if total_events <= 0:
        return (
            f"💡 Pro Tip: Run `body begin` inside your repo to start shielding reads & logs, "
            f"or run `body tips` ({DOCS_URL})."
        )

    by_organ = summary.get("by_organ") or {}
    has_eyes = any("eye" in str(k).lower() or "foveal" in str(k).lower() for k in by_organ)
    if not has_eyes:
        return (
            f'💡 Pro Tip: Use `body commit` for ~0.8s $0 git commits and `body eyes --app "Google Chrome"` '
            f"for <15ms UI snapshots • More: `body tips` | {DOCS_URL}"
        )
    return (
        f"💡 Pro Tip: Run `body context` on long chats and `body skills audit` to cut Turn-0 MCP/skill tax "
        f"• More: `body tips` | {DOCS_URL}"
    )


def get_contextual_tip_for_audit(audit: dict[str, Any]) -> str:
    """Return a concise 1-line contextual Pro Tip tailored to a `body audit-spend` report."""
    preventable_usd = float(audit.get("preventable_waste_usd") or audit.get("estimated_savings_usd") or 0.0)
    if preventable_usd >= 10.0:
        return (
            f"💡 Pro Tip: Capture these savings fast - run `body begin` in active repos, use `body commit` "
            f"(0 diff bloat), and `body skills prune --dry-run` • Playbook: {DOCS_URL}"
        )
    return (
        f"💡 Pro Tip: Explore 8 power-user workflows (`body commit`, `body ears`, `body flush --auto`) "
        f"via `body tips` or {DOCS_URL}"
    )
