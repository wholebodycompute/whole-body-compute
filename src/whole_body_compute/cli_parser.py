"""
Argument parser and subcommand definitions for the Whole Body Compute CLI (`body` / `wbc`).
"""

from __future__ import annotations

import argparse

from whole_body_compute import __version__
from whole_body_compute.doctor import (
    check_organ_brainstem,
    check_organ_circadian,
    check_organ_heart,
    check_organ_hippocampus,
    check_organ_immune_system,
    check_organ_mouth_and_ears,
    check_organ_nervous_system,
    check_organ_prefrontal,
    check_organ_skeleton,
    check_organ_the_eyes,
    check_organ_the_hands,
)


SINGLE_ORGAN_DISPATCH = {
    "heart": check_organ_heart,
    "skeleton": check_organ_skeleton,
    "brainstem": check_organ_brainstem,
    "prefrontal": check_organ_prefrontal,
    "nervous": check_organ_nervous_system,
    "voice": check_organ_mouth_and_ears,
    "eyes": check_organ_the_eyes,
    "hippocampus": check_organ_hippocampus,
    "memory": check_organ_hippocampus,
    "hands": check_organ_the_hands,
    "immune": check_organ_immune_system,
    "circadian": check_organ_circadian,
    "sleep": check_organ_circadian,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="body",
        description="Whole Body Compute — 5-Organ Sovereign Agent Runtime",
    )
    parser.add_argument("--version", action="version", version=f"body (whole-body-compute) {__version__}")
    parser.add_argument("--json", action="store_true", help="Output structured JSON")

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # body begin (aliases: init, awaken, start)
    p_beg = subparsers.add_parser(
        "begin",
        aliases=["init", "awaken", "start"],
        help="60-second workspace awakening: wire Whole Body Compute Blood-Brain Barrier, Bicameral rules & FinOps ROI receipt",
    )
    p_beg.add_argument("target", nargs="?", default=None, help="Target workspace directory (default: current directory)")
    p_beg.add_argument("--workspace", "-w", dest="workspace_opt", default=None, help="Target workspace directory (alias for target)")
    p_beg.add_argument("--dry-run", action="store_true", help="Preview hook & rule wiring without writing files")
    p_beg.add_argument("--skip-audit", action="store_true", help="Skip retroactive 30-day FinOps spend audit")
    p_beg.add_argument("--force", "--force-rules", dest="force_rules", action="store_true", help="Overwrite existing .agents/rules/agent_efficiency.md")
    p_beg.add_argument("--days", type=int, default=30, help="Historical window in days for FinOps receipt (default: 30)")
    p_beg.add_argument("--price-per-m", type=float, default=5.0, dest="price_per_m", help="Frontier input USD per 1M tokens (default: 5.0)")
    p_beg.add_argument("--seats", type=int, default=10, help="Number of team seats for annual projection (default: 10)")
    p_beg.add_argument(
        "--telemetry-tier",
        type=int,
        choices=[0, 1, 2],
        default=None,
        dest="telemetry_tier",
        help="Sovereign telemetry consent tier (0=Local 100%% on-device, 1=Vitals daily rollup, 2=Procedural DNA learning cards)",
    )
    p_beg.add_argument("--no-daemon", action="store_true", dest="no_daemon", help="Skip auto-installing background LaunchAgents")
    p_beg.add_argument("--json", action="store_true", help="Output structured JSON")

    # body doctor (aliases: status, organs)
    p_doc = subparsers.add_parser(
        "doctor",
        aliases=["status", "organs"],
        help="Run concurrent 5-Organ anatomical health audit",
    )
    p_doc.add_argument("--json", action="store_true", help="Output structured JSON")
    p_doc.add_argument("--exercise", action="store_true", help="Also run closed-loop 5-step reflex exercise")
    p_doc.add_argument("--chaos", "--gauntlet", dest="chaos", action="store_true", help="Also run adversarial ChaosOrganFuzzer stress suite")

    # body exercise (aliases: reflex)
    p_ex = subparsers.add_parser(
        "exercise",
        aliases=["reflex"],
        help="Run closed-loop 5-step physiological reflex pass across all core organs",
    )
    p_ex.add_argument("--json", action="store_true", help="Output structured JSON")

    # body chaos (aliases: gauntlet)
    p_chaos = subparsers.add_parser(
        "chaos",
        aliases=["gauntlet"],
        help="Run 11-scenario adversarial ChaosOrganFuzzer stress suite",
    )
    p_chaos.add_argument("--json", action="store_true", help="Output structured JSON")

    # body rem-sleep (aliases: remsleep, consolidate, nightly)
    p_rem = subparsers.add_parser(
        "rem-sleep",
        aliases=["remsleep", "consolidate", "nightly"],
        help="Run nocturnal REM sleep consolidation pass (memory hygiene, distillation, LoRA trigger, 30/30 gate)",
    )
    p_rem.add_argument("--dry-run", action="store_true", help="Run without triggering long GPU training runs")
    p_rem.add_argument("--force-lora", action="store_true", help="Force full Metal GPU LoRA distillation even if 0 new pairs were harvested")
    p_rem.add_argument("--min-eval", type=float, default=1.0, help="Minimum LocalBench SMB gate threshold")
    p_rem.add_argument("--json", action="store_true", help="Output structured JSON")

    # body candidates (aliases: candidate, recipes) # [direct-edit]
    p_cand = subparsers.add_parser(
        "candidates",
        aliases=["candidate", "recipes"],
        help="Inspect, discover, or promote staged ambient workflow candidates in recipes/candidates/",
    )
    p_cand.add_argument("--discover", action="store_true", help="Run on-demand motif discovery across ambient transitions journal & Screenpipe OCR")
    p_cand.add_argument("--promote", metavar="CANDIDATE_ID", help="Validate and promote a candidate YAML into active recipes/")
    p_cand.add_argument("--as-skill", action="store_true", dest="as_skill", help="Also export the promoted recipe as a portable Skill Bundle (.agents/skills/ + .skill.zip)")
    p_cand.add_argument("--share", action="store_true", help="Also harvest & export sanitized Procedural DNA learning card to telemetry outbox")
    p_cand.add_argument("--json", action="store_true", help="Output structured JSON")

    # body pull <org/recipe> [--version VER] [--hub-url URL] [--install-dir DIR] [--trust-key KEY=PUB] [--force] [--dry-run] [--offline] [--json]
    p_pull = subparsers.add_parser(
        "pull",
        help="Download, verify (SHA-256 + Ed25519 + Hoare contracts), and install a signed PalWorkflow recipe from the Procedural DNA Hub",
    )
    p_pull.add_argument("recipe_ref", help="Recipe reference (<org>/<recipe> or <recipe>, e.g. wbc/github-pr-triage)")
    p_pull.add_argument("--version", dest="version_pin", default=None, help="Pin exact semver version to pull")
    p_pull.add_argument("--hub-url", dest="hub_url", default=None, help="Override Hub index.json URL or local path")
    p_pull.add_argument("--install-dir", dest="install_dir", default=None, help="Custom installation directory")
    p_pull.add_argument("--trust-key", dest="trust_key", default=None, help="Trust third-party publisher key (<key_id>=<public_key_b64>)")
    p_pull.add_argument("--ignore-compat", dest="ignore_compat", action="store_true", help="Allow pulling recipes targeting another OS")
    p_pull.add_argument("--force", action="store_true", help="Re-verify and overwrite even if already installed")
    p_pull.add_argument("--dry-run", action="store_true", help="Verify cryptographic signatures without writing to disk")
    p_pull.add_argument("--offline", action="store_true", help="Resolve strictly against bundled offline signed Hub seed catalog")
    p_pull.add_argument("--json", action="store_true", help="Output structured JSON")

    # body search [query...] [--tag TAG] [--app APP] [--tier TIER] [--hub-url URL] [--offline] [--json]
    p_search = subparsers.add_parser(
        "search",
        help="Search signed $0.00 PalWorkflow recipes in the Procedural DNA Hub (hub.wholebodycompute.org)",
    )
    p_search.add_argument("query", nargs="*", default=[], help="Search keywords (e.g. github, linear, stripe, slack)")
    p_search.add_argument("--tag", default=None, help="Filter by recipe tag (e.g. git, finops, devops, browser)")
    p_search.add_argument("--app", default=None, help="Filter by macOS bundle ID, app name, or CLI binary")
    p_search.add_argument("--tier", default=None, help="Filter by dominant execution tier (TIER_0_LOCAL, TIER_1_HEADLESS_API, TIER_2_SEMANTIC_UI)")
    p_search.add_argument("--hub-url", dest="hub_url", default=None, help="Override Hub index.json URL or local path")
    p_search.add_argument("--offline", action="store_true", help="Search bundled offline signed Hub seed catalog")
    p_search.add_argument("--json", action="store_true", help="Output structured JSON")

    # body publish <recipe.yaml> [--namespace ORG] [--out-dir DIR] [--sanitize] [--upload] [--registry-url URL] [--dry-run] [--json]
    p_pub = subparsers.add_parser(
        "publish",
        help="Validate PalWorkflow v1 schema + Zero-PII guarantees, sign with Ed25519 + SHA-256, and package/publish to the Hub",
    )
    p_pub.add_argument("recipe_path", help="Path to PalWorkflow YAML/JSON file to validate and sign")
    p_pub.add_argument("--namespace", "-n", default=None, help="Publisher namespace/org slug (default: recipe namespace or 'community')")
    p_pub.add_argument("--out-dir", dest="out_dir", default=None, help="Output directory for signed bundle (.yaml + .manifest.json)")
    p_pub.add_argument("--key-path", dest="key_path", default=None, help="Custom Ed25519 publisher key JSON path")
    p_pub.add_argument("--no-keygen", dest="no_keygen", action="store_true", help="Fail instead of auto-generating a publisher key if none exists")
    p_pub.add_argument("--sanitize", action="store_true", help="Run Procedural DNA text sanitizer on notes/descriptions before Zero-PII audit")
    p_pub.add_argument("--upload", action="store_true", help="POST signed recipe bundle to remote registry endpoint")
    p_pub.add_argument("--registry-url", dest="registry_url", default=None, help="Remote registry publish endpoint URL")
    p_pub.add_argument("--dry-run", action="store_true", help="Validate and sign in-memory without writing files")
    p_pub.add_argument("--json", action="store_true", help="Output structured JSON")

    # body hub <list|search|pull|publish|verify|installed|keygen|trust> [targets...]
    p_hub = subparsers.add_parser(
        "hub",
        help="Unified Procedural DNA Hub & Signed Recipe Registry manager (list, search, pull, publish, verify, installed, keygen, trust)",
    )
    p_hub.add_argument(
        "hub_action",
        nargs="?",
        default="list",
        choices=["list", "search", "pull", "publish", "verify", "installed", "keygen", "trust"],
        help="Hub subcommand (default: list)",
    )
    p_hub.add_argument("targets", nargs="*", default=[], help="Target arguments for the selected Hub subcommand")
    p_hub.add_argument("--namespace", "-n", default=None, help="Publisher namespace/org slug")
    p_hub.add_argument("--version", dest="version_pin", default=None, help="Exact semver version pin")
    p_hub.add_argument("--tag", default=None, help="Tag filter")
    p_hub.add_argument("--app", default=None, help="App/bundle_id/CLI filter")
    p_hub.add_argument("--tier", default=None, help="Execution tier filter")
    p_hub.add_argument("--hub-url", dest="hub_url", default=None, help="Override Hub index.json URL or local path")
    p_hub.add_argument("--install-dir", dest="install_dir", default=None, help="Custom installation directory")
    p_hub.add_argument("--out-dir", dest="out_dir", default=None, help="Output directory for signed publish bundle")
    p_hub.add_argument("--key-path", dest="key_path", default=None, help="Custom Ed25519 publisher key JSON path")
    p_hub.add_argument("--trust-key", dest="trust_key", default=None, help="Trust third-party publisher key (<key_id>=<public_key_b64>)")
    p_hub.add_argument("--allow-untrusted-key", dest="allow_untrusted_key", action="store_true", help="Verify cryptographic self-consistency without requiring keyring pin")
    p_hub.add_argument("--ignore-compat", dest="ignore_compat", action="store_true", help="Allow pulling recipes targeting another OS")
    p_hub.add_argument("--sanitize", action="store_true", help="Sanitize descriptions/notes during publish")
    p_hub.add_argument("--upload", action="store_true", help="Upload signed bundle to registry endpoint")
    p_hub.add_argument("--registry-url", dest="registry_url", default=None, help="Remote registry publish endpoint")
    p_hub.add_argument("--force", action="store_true", help="Force overwrite on pull")
    p_hub.add_argument("--dry-run", action="store_true", help="Dry-run verification without disk writes")
    p_hub.add_argument("--offline", action="store_true", help="Use bundled offline signed Hub seed catalog")
    p_hub.add_argument("--json", action="store_true", help="Output structured JSON")

    # body telemetry (aliases: dna, learnings) # [direct-edit]
    p_dna = subparsers.add_parser(
        "telemetry",
        aliases=["dna", "learnings"],
        help="Federated Procedural DNA & Ethical Telemetry Engine (status, harvest, inspect, generalize, compact, opt-in, opt-out)",
    )
    p_dna.add_argument(
        "action",
        nargs="?",
        default="status",
        choices=["status", "harvest", "sync", "flush", "inspect", "generalize", "compact", "opt-in", "opt-out"],
        help="Telemetry action (default: status)",
    )
    p_dna.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Recipe ID or YAML path for 'generalize' (e.g. ambient_client_package_upload)",
    )
    p_dna.add_argument(
        "--tier",
        type=int,
        choices=[0, 1, 2],
        default=None,
        help="Consent tier for opt-in/opt-out (0=Local, 1=Vitals, 2=Procedural DNA)",
    )
    p_dna.add_argument(
        "--max-mb",
        dest="max_mb",
        type=float,
        default=15.0,
        help="Maximum telemetry storage budget in MB for compact (default: 15.0)",
    )
    p_dna.add_argument(
        "--retention-days",
        dest="retention_days",
        type=int,
        default=7,
        help="Retention window in days for compact (default: 7)",
    )
    p_dna.add_argument("--port", type=int, default=9381, help="Dashboard server port (default: 9381)")
    p_dna.add_argument("--host", default="127.0.0.1", help="Dashboard server host (default: 127.0.0.1)")
    p_dna.add_argument("--open", action="store_true", help="Automatically open dashboard in default browser")
    p_dna.add_argument("--logs", action="store_true", help="Print historical Readiness & Biometric state logs")
    p_dna.add_argument("--limit", type=int, default=20, help="Number of log entries to display")
    p_dna.add_argument("--json", action="store_true", help="Output structured JSON")

    # body radix-cache (aliases: radix, prewarm, kv-cache)
    p_rad = subparsers.add_parser(
        "radix-cache",
        aliases=["radix", "prewarm", "kv-cache"],
        help="Pre-warm and pin circulatory Radix KV-Cache (.goosehints, system guard, tool schemas) on Metal GPU",
    )
    p_rad.add_argument("--url", default="http://127.0.0.1:9379/v1/chat/completions", help="Target Metal GPU URL")
    p_rad.add_argument("--status", action="store_true", help="Display cached prefix metadata and warm TTFB status")
    p_rad.add_argument("--json", action="store_true", help="Output structured JSON")

    # body daemon [status|logs|install|uninstall] [--dry-run] [--json]
    p_daemon = subparsers.add_parser(
        "daemon",
        help="Inspect or manage Whole Body Compute background LaunchAgents (Foveal Socket, Foveal HTTP, REM Sleep & HUD)",
    )
    p_daemon.add_argument(
        "action",
        nargs="?",
        default="status",
        choices=["status", "logs", "install", "uninstall"],
        help="Daemon action (default: status)",
    )
    p_daemon.add_argument("--dry-run", action="store_true", help="Preview LaunchAgent plist creation/removal without invoking launchctl")
    p_daemon.add_argument("--json", action="store_true", help="Output structured JSON")

    # body hud (aliases: dashboard) # [direct-edit]
    p_tel = subparsers.add_parser(
        "hud",
        aliases=["dashboard"],
        help="Launch living health app telemetry dashboard & WebSocket server",
    )
    p_tel.add_argument("--port", type=int, default=9381, help="Server port (default: 9381)")
    p_tel.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    p_tel.add_argument("--open", action="store_true", help="Automatically open dashboard in default browser")
    p_tel.add_argument("--logs", action="store_true", help="Print historical Readiness & Biometric state logs")
    p_tel.add_argument("--limit", type=int, default=20, help="Number of log entries to display")
    p_tel.add_argument("--json", action="store_true", help="Output structured JSON")

    # body logs (aliases: history)
    p_hist = subparsers.add_parser(
        "logs",
        aliases=["history"],
        help="Inspect persistent Apple Silicon Readiness, Thermal, and Strain state history (~/.whole_body_compute/telemetry_history.jsonl)",
    )
    p_hist.add_argument("--limit", "-n", type=int, default=25, help="Number of entries to show (default: 25)")
    p_hist.add_argument("--transitions", "-t", action="store_true", help="Only show state transition events")
    p_hist.add_argument("--json", action="store_true", help="Output structured JSON")

    # body perfusion (aliases: glymphatic, ram-hygiene)
    p_perf = subparsers.add_parser(
        "perfusion",
        aliases=["glymphatic", "ram-hygiene"],
        help="Selective Vascular Perfusion & Ultradian Glymphatic RAM Hygiene (evict idle Tier-1/2 models or trigger Cephalic Pre-Warm)",
    )
    p_perf.add_argument("--evict", action="store_true", help="Evict idle Tier-1/Tier-2 model servers into Darwin Inactive File Cache")
    p_perf.add_argument("--recycle-tachycardic", "--recycle", action="store_true", help="Recycle Tier-0 autonomic processes exhibiting tachycardia (>120%% CPU)")
    p_perf.add_argument("--dry-run", action="store_true", help="Preview evictable model RAM without terminating processes")
    p_perf.add_argument("--prewarm", metavar="MODEL", help="Trigger Cephalic Phase Predictive Pre-Warm for a specific model")
    p_perf.add_argument("--ttl", default="auto", help="Idle keep_alive TTL for pre-warmed model (default: auto)")
    p_perf.add_argument("--cold-starts", action="store_true", help="Display cold-start telemetry and adaptive keep-warm policies")
    p_perf.add_argument("--json", action="store_true", help="Output structured JSON")

    # body immune-hook --phase [pre|post]
    p_ihook = subparsers.add_parser(
        "immune-hook",
        help="Execute Whole Body Compute PreToolUse / PostToolUse IDE hook handler (reads JSON on stdin, emits JSON on stdout)",
    )
    p_ihook.add_argument("--phase", choices=["pre", "post"], default="pre", help="Hook lifecycle phase")

    # body context (aliases: csi, watchdog) - Drill-down bloat & step debugger
    p_ctx = subparsers.add_parser(
        "context",
        aliases=["csi", "watchdog"],
        help="Drill-down Context Bloat Debugger & Watchdog (top bloat tools, auto-flush trigger)",
    )
    p_ctx.add_argument("--conv-id", metavar="CONV_ID", help="Target conversation ID or prefix")
    p_ctx.add_argument("--all", action="store_true", help="Inspect all recent conversations")
    p_ctx.add_argument("--limit", type=int, default=5, help="Number of recent conversations to inspect (default: 5)")
    p_ctx.add_argument("--auto", action="store_true", help="Automatically trigger 'body flush --auto' if context is YELLOW or RED")
    p_ctx.add_argument("--insights", "--analytics", action="store_true", dest="insights", help="Generate longitudinal root-cause bloat attribution and continuous improvement insights")
    p_ctx.add_argument("--by-model", "--models", action="store_true", dest="by_model", help="Empirical multi-model leaderboard across transcript.jsonl logs")
    p_ctx.add_argument("--include-subagents", action="store_true", dest="include_subagents", help="Merge invoke_subagent child cohorts into the main model table")
    p_ctx.add_argument("--window", default=None, help="Reporting window for --by-model (1h, 24h, 7d, 30d, all; default: 30d)")
    p_ctx.add_argument("--receipt", action="store_true", help="Print compact Markdown receipt when paired with --by-model")
    p_ctx.add_argument("--output", dest="output_path", metavar="PATH", help="Optional output path to write Markdown leaderboard report")
    p_ctx.add_argument("--json", action="store_true", help="Output structured JSON")

    # body flush (aliases: handoff)
    p_fl = subparsers.add_parser(
        "flush",
        aliases=["handoff"],
        help="Ultradian Glymphatic Flush & State Capsule Handoff (grounded serialization to SQLite WAL + Zero-Click bridge)",
    )
    p_fl.add_argument("--conv-id", metavar="CONV_ID", help="Target conversation ID or prefix")
    p_fl.add_argument("--goal", help="Active goal override")
    p_fl.add_argument("--milestone", help="Active milestone override")
    p_fl.add_argument("--next-step", help="Immediate next action override")
    p_fl.add_argument("--artifact", action="append", dest="artifacts", help="Additional artifact file path to SHA-256 hash (repeatable)")
    p_fl.add_argument("--slack-thread", help="External Slack thread URL or thread_ts")
    p_fl.add_argument("--auto", action="store_true", help="Zero-Click Handoff: spawn fresh conversation + paste & submit prompt, pbcopy, IPC, TTS")
    p_fl.add_argument("--spawn", dest="spawn", action="store_true", default=None, help="Explicitly spawn a fresh conversation and paste/submit the handoff prompt")
    p_fl.add_argument("--no-spawn", dest="no_spawn", action="store_true", help="Copy to clipboard without opening/submitting a new conversation tab")
    p_fl.add_argument("--no-submit", dest="no_submit", action="store_true", help="Paste into the new conversation composer without pressing Enter")
    p_fl.add_argument("--app", dest="spawn_app", default="Antigravity", help="Target IDE/Agent app for --auto / --spawn (default: Antigravity)")
    p_fl.add_argument("--evict-models", action="store_true", help="Also trigger Selective Vascular Perfusion idle model eviction")
    p_fl.add_argument("--json", action="store_true", help="Output structured JSON")

    # body scorecard (aliases: report, metrics, kpi) - Primary Consolidated Evaluation Command
    p_rep = subparsers.add_parser(
        "scorecard",
        aliases=["report", "metrics", "kpi"],
        help="Primary Consolidated KPI Scorecard (Turn Inflation, CSI, BBB Shield, Local %%, & Gate Attempt-#1 FTR)",
    )
    p_rep.add_argument("--window", default=None, help="Primary reporting window (1h, 24h, 7d, 30d, all; default: 24h, or 30d with --by-model)")
    p_rep.add_argument("--trend", default="7d", help="Trend comparison window (default: 7d)")
    p_rep.add_argument("--conv-id", dest="conv_id", metavar="CONV_ID", help="Target conversation ID or prefix")
    p_rep.add_argument("--by-model", "--models", action="store_true", dest="by_model", help="Empirical multi-model leaderboard across transcript.jsonl logs")
    p_rep.add_argument("--include-subagents", action="store_true", dest="include_subagents", help="Merge invoke_subagent child cohorts into the main model table")
    p_rep.add_argument("--write", "--sync", dest="write_ledger", action="store_true", default=True, help="Write .agents/METRICS.md ledger (default: True)")
    p_rep.add_argument("--no-write", dest="write_ledger", action="store_false", help="Do not write .agents/METRICS.md")
    p_rep.add_argument("--output", dest="output_path", metavar="PATH", help="Optional custom output path for Markdown ledger")
    p_rep.add_argument("--receipt", action="store_true", help="Print compact End-of-Session Handoff Receipt format")
    p_rep.add_argument("--json", action="store_true", help="Output structured JSON")

    # body audit-spend (aliases: roi, finops) - 30-day Historical FinOps Invoice
    p_fin = subparsers.add_parser(
        "audit-spend",
        aliases=["roi", "finops"],
        help="30-Day Historical Token Waste & USD Savings Calculator (Monthly FinOps ROI Invoice)",
    )
    p_fin.add_argument("--days", type=int, default=30, help="Historical window in days to scan (default: 30)")
    p_fin.add_argument("--price-per-m", type=float, default=5.0, dest="price_per_m", help="Frontier input USD per 1M tokens (default: 5.0)")
    p_fin.add_argument("--seats", type=int, default=10, help="Number of team seats for annual projection (default: 10)")
    p_fin.add_argument("--top", type=int, default=5, help="Number of top expensive threads to display (default: 5)")
    p_fin.add_argument("--export-receipt", dest="export_receipt", metavar="PATH", help="Optional path to write a shareable Markdown FinOps ROI receipt")
    p_fin.add_argument("--json", action="store_true", help="Output structured JSON")

    # body team-sync (aliases: hive-sync, share-exemplars)
    p_sync = subparsers.add_parser(
        "team-sync",
        aliases=["hive-sync", "share-exemplars"],
        help="Synchronize verified cloud exemplars & LGTM lessons with git-committable .agents/team_exemplars.jsonl",
    )
    p_sync.add_argument("--path", default=".agents/team_exemplars.jsonl", help="Path to shared team exemplars JSONL ledger")
    p_sync.add_argument("--json", action="store_true", help="Output structured JSON")

    # body license (aliases: tier, pricing, upgrade, plan)
    p_lic = subparsers.add_parser(
        "license",
        aliases=["tier", "pricing", "upgrade", "plan"],
        help="Inspect or switch Whole Body Compute PLG license tier (community, pro, team, enterprise) and ROI payback",
    )
    p_lic.add_argument(
        "--set-tier",
        dest="set_tier",
        choices=["community", "pro", "team", "enterprise"],
        help="Switch active PLG tier (community, pro, team, enterprise)",
    )
    p_lic.add_argument(
        "--activate",
        "--key",
        dest="activate_key",
        metavar="POLAR_KEY",
        help="Activate a Polar.sh license key (e.g. WBC-PRO-XXXX-YYYY)",
    )
    p_lic.add_argument(
        "--org-id",
        dest="polar_org_id",
        metavar="ORG_UUID",
        help="Polar.sh Organization UUID (or set WBC_POLAR_ORG_ID)",
    )
    p_lic.add_argument(
        "--deactivate",
        dest="deactivate",
        action="store_true",
        help="Deactivate license on this machine and revert to Community tier",
    )
    p_lic.add_argument(
        "--offline",
        dest="offline",
        action="store_true",
        help="Verify license key offline without contacting api.polar.sh",
    )
    p_lic.add_argument(
        "--wheel-path",
        dest="wheel_path",
        metavar="WHEEL_PATH",
        help="Install whole-body-compute-pro from a local .whl archive (air-gapped Enterprise support)",
    )
    p_lic.add_argument(
        "--wheel-url",
        dest="wheel_url",
        metavar="WHEEL_URL",
        help="Override Polar-gated wheel endpoint URL (default: https://pro.wholebodycompute.com/v1/wheel)",
    )
    p_lic.add_argument(
        "--no-install-wheel",
        dest="no_install_wheel",
        action="store_true",
        help="Activate license key without installing or downloading the Pro plugin wheel",
    )
    p_lic.add_argument(
        "--reinstall-wheel",
        dest="reinstall_wheel",
        action="store_true",
        help="Re-download or reinstall whole-body-compute-pro into the current Python environment",
    )
    p_lic.add_argument(
        "--check-feature",
        dest="check_feature",
        metavar="FEATURE",
        help="Check if a specific feature is allowed under the current tier",
    )
    p_lic.add_argument(
        "--daily-interceptions",
        dest="daily_interceptions",
        type=int,
        default=0,
        help="Simulated daily BBB interception count for feature gate check",
    )
    p_lic.add_argument(
        "--daily-silicon-offloads",
        dest="daily_silicon_offloads",
        type=int,
        default=0,
        help="Simulated daily local silicon offloads count for feature gate check",
    )
    p_lic.add_argument(
        "--daily-anti-cheat-blocks",
        dest="daily_anti_cheat_blocks",
        type=int,
        default=0,
        help="Simulated daily Anti-Reward-Hacking blocks count for feature gate check",
    )
    p_lic.add_argument("--json", action="store_true", help="Output structured JSON")

    # body brief [--channel <name>] [--speak] [--approve] [--yes] [--window <window>] [--json]
    p_brief = subparsers.add_parser(
        "brief",
        aliases=["duet", "morning"],
        help="Sovereign Duet on Local Metal: <30s 4-pillar morning/channel briefing (Eyes + Hippocampus + Git + FinOps)",
    )
    p_brief.add_argument(
        "--channel",
        "-c",
        default="morning",
        help="Optional channel/topic filter (e.g. morning, engineering, finops, security, all; default: morning)",
    )
    p_brief.add_argument(
        "--speak",
        action="store_true",
        help="Speak the <30-second briefing aloud via local neural TTS",
    )
    p_brief.add_argument(
        "--voice",
        default="Christopher",
        help="Local neural TTS persona for --speak (default: Christopher)",
    )
    p_brief.add_argument(
        "--speed",
        type=float,
        default=1.35,
        help="Spoken playback velocity for --speak (default: 1.35)",
    )
    p_brief.add_argument(
        "--approve",
        action="store_true",
        help="Run Human-in-the-Loop approval gate for suggested actions & staged PalWorkflow promotions",
    )
    p_brief.add_argument(
        "--yes",
        "-y",
        "--auto-approve",
        dest="auto_approve",
        action="store_true",
        help="Automatically approve suggested actions and candidate promotions when paired with --approve",
    )
    p_brief.add_argument(
        "--window",
        "-w",
        default="24h",
        help="FinOps/metrics window slice (default: 24h)",
    )
    p_brief.add_argument("--json", action="store_true", help="Output structured JSON")

    # body airlock <skeleton|zoom|bundle|classify|benchmark|learn|mastery> [target] [-q query] [--symbol sym]
    p_airlock = subparsers.add_parser(
        "airlock",
        help="Sovereign Airlock & 3-Mode Biological BBB Permeability Valve (skeleton, zoom, bundle, classify, benchmark, learn, mastery)",
    )
    p_airlock.add_argument(
        "action",
        nargs="?",
        default="benchmark",
        choices=["skeleton", "zoom", "bundle", "classify", "benchmark", "learn", "mastery"],
        help="Airlock action (skeleton, zoom, bundle, classify, benchmark, learn, mastery)",
    )
    p_airlock.add_argument("target", nargs="?", default=None, help="Target file or directory path")
    p_airlock.add_argument("-q", "--query", default="", help="Task query or search objective")
    p_airlock.add_argument("--symbol", "-s", default="", help="Symbol name to zoom into")
    p_airlock.add_argument("--files", type=int, default=1, help="Estimated file count for permeability classification")
    p_airlock.add_argument("--strikes", type=int, default=0, help="Prior retry strikes for permeability classification")
    p_airlock.add_argument("--json", action="store_true", help="Output structured JSON")

    # body shield [--status] [--install] [--bot <bot>] [--all] [--scope workspace|user] [--dry-run] [--json] # [direct-edit]
    p_shield = subparsers.add_parser(
        "shield",
        help="Zero-Schema-Bloat Whole Body Compute Hook Installer & Status Auditor (Antigravity, Claude Code, Cursor, Codex)",
    )
    p_shield.add_argument("--status", action="store_true", help="Audit zero-token hook installation status across bots (default)")
    p_shield.add_argument("--install", action="store_true", help="Idempotently wire PreToolUse and PostToolUse hooks into bot configs")
    p_shield.add_argument("--bot", action="append", dest="bots", metavar="BOT", help="Target bot (claude, antigravity, cursor, codex; repeatable)")
    p_shield.add_argument("--all", action="store_true", dest="all_bots", help="Install or audit across all supported bots")
    p_shield.add_argument("--scope", choices=["workspace", "user"], default="workspace", help="Target config scope (default: workspace)")
    p_shield.add_argument("--workspace", default=None, help="Target workspace directory (default: current directory)")
    p_shield.add_argument("--dry-run", action="store_true", help="Preview hook installation without modifying files")
    p_shield.add_argument("--json", action="store_true", help="Output structured JSON")

    # body savings [--json] [--bot <bot>] [--today] [--window <w>] [--leaderboard] [--reset] # [direct-edit]
    p_savings = subparsers.add_parser(
        "savings",
        help="Live Local Metal Shield Telemetry Ledger & Token/USD/Time Receipt",
    )
    p_savings.add_argument("--bot", default=None, help="Filter ledger by bot name (e.g. antigravity, claude-code, cursor, codex)")
    p_savings.add_argument("--price-per-m", type=float, default=None, dest="price_per_m", help="Override frontier input USD per 1M tokens (default: auto-detected active model rate)")
    p_savings.add_argument("--today", action="store_true", help="Filter to today since local midnight")
    p_savings.add_argument("--window", choices=["today", "24h", "7d", "30d", "all"], default=None, help="Time window filter (today, 24h, 7d, 30d, all)")
    p_savings.add_argument("-l", "--leaderboard", action="store_true", help="Render ranked 4-board Savings Leaderboard (Organs, Files/Apps, Bots, Sessions)")
    p_savings.add_argument("--top", type=int, default=15, help="Number of top targets to display in leaderboard (default: 15)")
    p_savings.add_argument("--reset", action="store_true", help="Reset the shield savings ledger")
    p_savings.add_argument("--json", action="store_true", help="Output structured JSON")

    # body leaderboard (aliases: rank) # [direct-edit]
    p_lb = subparsers.add_parser(
        "leaderboard",
        aliases=["rank"],
        help="Ranked Savings Leaderboard across Organs, Files/Apps, Bots & Sessions (Tokens, USD & Time Saved)",
    )
    p_lb.add_argument("--bot", default=None, help="Filter ledger by bot name (e.g. antigravity, claude-code, cursor, codex)")
    p_lb.add_argument("--price-per-m", type=float, default=None, dest="price_per_m", help="Override frontier input USD per 1M tokens")
    p_lb.add_argument("--today", action="store_true", help="Filter to today since local midnight")
    p_lb.add_argument("--24h", action="store_true", dest="last_24h", help="Filter to rolling last 24 hours")
    p_lb.add_argument("--7d", action="store_true", dest="last_7d", help="Filter to rolling last 7 days")
    p_lb.add_argument("--window", choices=["today", "24h", "7d", "30d", "all"], default=None, help="Time window filter (today, 24h, 7d, 30d, all)")
    p_lb.add_argument("--top", type=int, default=15, help="Number of top targets to display (default: 15)")
    p_lb.add_argument("--json", action="store_true", help="Output structured JSON")

    # body scout <target> [-q <query>] [--slice <specs...>] [--json]
    p_scout = subparsers.add_parser(
        "scout",
        help="On-device Local Metal ReconScout context digest & AST symbol slicer",
    )
    p_scout.add_argument("target", nargs="?", default=None, help="Target file or log path to scout")
    p_scout.add_argument(
        "--slice",
        "--slices",
        dest="slices",
        nargs="+",
        default=[],
        help="Multi-file batch AST symbol slices (e.g. file.py:Symbol file.py:10-50)",
    )
    p_scout.add_argument("-q", "--query", default="Summarize key structure and findings", help="Question or objective")
    p_scout.add_argument("-m", "--model", default="qwen2.5-coder:1.5b", help="Local model identifier")
    p_scout.add_argument("--json", action="store_true", help="Output structured JSON")

    # body implement <target> -i <instruction> [--model-coder <model>] [--no-apply] [--json]
    p_impl = subparsers.add_parser(
        "implement",
        help="On-device Local Metal ReconImplementer surgical single-file code edit",
    )
    p_impl.add_argument("target", help="Target file path to modify")
    p_impl.add_argument("-i", "--instruction", required=True, help="Surgical edit instruction")
    p_impl.add_argument("-m", "--model-coder", default="auto", help="Local coder model or 'auto'")
    p_impl.add_argument("--no-apply", "--dry-run", dest="no_apply", action="store_true", help="Preview diff without writing to disk")
    p_impl.add_argument("--json", action="store_true", help="Output structured JSON")

    # body loop <goal> [-m <model>] [-t <max_turns>] [--json]
    p_loop = subparsers.add_parser(
        "loop",
        help="On-device LocalAutonomousLoop multi-turn engineering agent",
    )
    p_loop.add_argument("goal", help="Engineering goal or bug-fix objective")
    p_loop.add_argument("-m", "--model", default="auto", help="Local model identifier or 'auto'")
    p_loop.add_argument("-t", "--max-turns", type=int, default=5, help="Maximum autonomous turns")
    p_loop.add_argument("--json", action="store_true", help="Output structured JSON")

    # body arena [--arena-dir <path>] [--setup-only] [--no-claude-parity] [--json]
    p_arena = subparsers.add_parser(
        "arena",
        help="Run the 4-Way WBC Head-to-Head Benchmark Arena & Claude Code Parity Suite",
    )
    p_arena.add_argument("--arena-dir", default="/tmp/wbc_arena", help="Arena sandbox directory")
    p_arena.add_argument("--setup-only", action="store_true", help="Only create Arena fixtures")
    p_arena.add_argument("--no-claude-parity", action="store_true", help="Skip Claude Code parity check")
    p_arena.add_argument("--json", action="store_true", help="Output structured JSON")

    # body commit [paths...] [-m message] [-a] [--no-lint] [--no-llm] [--wip] [--session-id ID] [--json]
    p_commit = subparsers.add_parser(
        "commit",
        help="Sub-20ms AST-guided smart git commit with pre-flight secret/lint SecGate",
    )
    p_commit.add_argument(
        "paths",
        nargs="*",
        default=[],
        help="Optional file paths to commit (auto-scoped if omitted)",
    )
    p_commit.add_argument(
        "-m",
        "--message",
        default=None,
        help="Explicit commit message (auto-synthesized if omitted)",
    )
    p_commit.add_argument(
        "-a",
        "--all-tracked",
        action="store_true",
        help="Stage all modified tracked files across the repo",
    )
    p_commit.add_argument(
        "--no-lint",
        action="store_true",
        help="Skip staged ruff lint check (secret & AST gates still run)",
    )
    p_commit.add_argument(
        "--no-llm",
        action="store_true",
        help="Use sub-15ms deterministic AST delta commit message without querying local LLM",
    )
    p_commit.add_argument(
        "--wip",
        action="store_true",
        help="Create a fast WIP snapshot commit on HEAD",
    )
    p_commit.add_argument(
        "--session-id",
        default=None,
        help="Optional session/agent ID for smart auto-scoping",
    )
    p_commit.add_argument("--json", action="store_true", help="Emit JSON output")

    # body checkpoint [label] [--commit] [--list] [--restore REF] [--json]
    p_checkpoint = subparsers.add_parser(
        "checkpoint",
        help="Zero-risk working-tree snapshot to refs/checkpoints/* or HEAD",
    )
    p_checkpoint.add_argument(
        "label",
        nargs="?",
        default=None,
        help="Optional label for the checkpoint",
    )
    p_checkpoint.add_argument(
        "--commit",
        action="store_true",
        help="Commit dirty state directly to HEAD (for pre-branch subagent prep) instead of refs/checkpoints/*",
    )
    p_checkpoint.add_argument(
        "--list",
        action="store_true",
        help="List saved checkpoints in refs/checkpoints/*",
    )
    p_checkpoint.add_argument(
        "--restore",
        default=None,
        metavar="REF",
        help="Restore a saved checkpoint ref into the working tree",
    )
    p_checkpoint.add_argument("--json", action="store_true", help="Emit JSON output")

    # 1-Way Trojan Horse migration from legacy Screenpipe DB to Sovereign Foveal Memory
    p_import_sp = subparsers.add_parser(
        "import-screenpipe",
        help="1-Way Trojan Horse migration from Screenpipe SQLite to sovereign Foveal Memory (~/.whole_body/foveal_memory.sqlite)",
    )
    p_import_sp.add_argument(
        "--source",
        default=None,
        help="Source Screenpipe SQLite DB path (default: ~/.screenpipe/db.sqlite)",
    )
    p_import_sp.add_argument(
        "--target",
        default=None,
        help="Target Foveal Memory SQLite DB path (default: ~/.whole_body/foveal_memory.sqlite)",
    )
    p_import_sp.add_argument(
        "--no-redact",
        action="store_true",
        help="Skip PII and secret redaction during import",
    )
    p_import_sp.add_argument("--json", action="store_true", help="Emit JSON output")

    # Skills Telemetry, Token-Tax Audit & Cold-Storage Pruner
    p_skills = subparsers.add_parser(
        "skills",
        aliases=["skill"],
        help="Audit skill usage frequency, export PalWorkflow recipes as portable skills, and prune/restore dormant skills",
    )
    p_skills.add_argument(
        "action",
        nargs="?",
        choices=["audit", "prune", "restore", "archive", "export"],
        default="audit",
        help="Action: audit (default), export (<recipe_or_id>), prune (shelve 0-use skills), restore (<name> or 'all'), archive (list shelf)",
    )
    p_skills.add_argument(
        "targets",
        nargs="*",
        help="Skill name(s) to prune/restore, or recipe path/name to export",
    )
    p_skills.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Output root directory for `body skills export` (default: .agents/skills)",
    )
    p_skills.add_argument(
        "--no-zip",
        dest="no_zip",
        action="store_true",
        help="Skip creating `<slug>.skill.zip` archive during `body skills export`",
    )
    p_skills.add_argument(
        "--max-convs",
        dest="max_convs",
        type=int,
        default=0,
        help="Maximum conversation count threshold for automatic pruning (default: 0)",
    )
    p_skills.add_argument(
        "--include-protected",
        dest="include_protected",
        action="store_true",
        help="Allow pruning protected core skills when 0-use",
    )
    p_skills.add_argument(
        "--all",
        dest="show_all",
        action="store_true",
        help="Include built-in Antigravity skills in the audit table",
    )
    p_skills.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Preview skills that would be shelved without moving files",
    )
    p_skills.add_argument("--json", action="store_true", help="Emit structured JSON output")

    # Individual organ subcommands
    for organ_key in SINGLE_ORGAN_DISPATCH:
        p_org = subparsers.add_parser(organ_key, help=f"Inspect organ '{organ_key}' vital signs and latency")
        p_org.add_argument("--json", action="store_true", help="Output structured JSON")
        if organ_key == "eyes":
            p_org.add_argument("--app", type=str, default=None, help="Target application name (e.g. 'Claude', 'Muse', 'Google Chrome')")
            p_org.add_argument("--title", type=str, default=None, help="Target window or browser tab title substring")
            p_org.add_argument("--pid", type=int, default=None, help="Target process ID")
            p_org.add_argument("--max-elements", type=int, default=None, help="Maximum interactive elements to capture (default: 10 or 40)")
            p_org.add_argument("--filter", type=str, default=None, help="Filter elements by case-insensitive substring on role, title, value, or description")
            p_org.add_argument("--click", type=str, default=None, help="Click element by ID (e.g. 'e1') or label substring")
            p_org.add_argument("--click-xy", type=str, default=None, help="Click exact coordinates 'x,y'")
            p_org.add_argument("--hover", type=str, default=None, help="Hover over element ID ('e1') or coordinates 'x,y' to reveal context controls")
            p_org.add_argument("--right-click", dest="right_click", type=str, default=None, help="Right-click element ID or 'x,y' to open context menu")
            p_org.add_argument("--attach", type=str, default=None, help="Attach file via macOS NSOpenPanel Go-to-folder sheet injection")
            p_org.add_argument("--chain", type=str, default=None, help="Compound macro chain (e.g. 'click:e5,paste:Hello,hotkey:enter,wait-reply:5')")
            p_org.add_argument("--floating", action="store_true", help="Include floating windows, system dialogs, and popovers")
            p_org.add_argument("--status-bar", dest="status_bar", action="store_true", help="Include menu bar items and extras (AXExtrasMenuBar)")
            p_org.add_argument("--type", dest="type_text", type=str, default=None, help="Type text into targeted or focused input")
            p_org.add_argument("--paste", dest="paste_text", type=str, default=None, help="Atomically paste text via clipboard Cmd+V (fast React/ProseMirror injection)")
            p_org.add_argument("--scroll", dest="scroll", choices=["up", "down", "top", "bottom"], default=None, help="Scroll window/view up/down/top/bottom via AXScroll + Quartz")
            p_org.add_argument("--hotkey", dest="hotkey", type=str, default=None, help="Dispatch key chord (e.g. 'cmd+n', 'cmd+shift+p', 'escape', 'enter')")
            p_org.add_argument("--menu", dest="menu", type=str, default=None, help="Trigger menu item (e.g. 'View > Hide Sidebar') via AXPress in <3ms")
            p_org.add_argument("--submit", action="store_true", help="Send Return/Enter keystroke after clicking or typing")
            p_org.add_argument("--read-last", dest="read_last", action="store_true", help="Extract latest meaningful text blocks from window")
            p_org.add_argument("--wait-reply", dest="wait_reply", type=float, default=None, help="Wait up to TIMEOUT seconds for target app response stream to settle")
            p_org.add_argument("--daemon", dest="daemon", action="store_true", help="Run warm Unix socket daemon (~/.whole_body_compute/foveal.sock) + default-on AmbientWorkflowObserver")
            p_org.add_argument("--no-ambient", dest="no_ambient", action="store_true", help="Disable passive AmbientWorkflowObserver when running `body eyes --daemon`")
        if organ_key in ("memory", "hippocampus"):
            p_org.add_argument(
                "--remember",
                "--add",
                dest="remember",
                metavar="FACT",
                help="Store a technical fact or preference in Whole Body Compute Hippocampus DotMemory",
            )
            p_org.add_argument(
                "--recall",
                "--query",
                dest="recall",
                metavar="QUERY",
                help="Search and recall facts from Whole Body Compute Hippocampus DotMemory",
            )
            p_org.add_argument(
                "--deja-vu",
                dest="deja_vu",
                metavar="ERROR",
                help="Check memories and episodic timeline for prior bug occurrences",
            )
            p_org.add_argument(
                "--rewind",
                dest="rewind",
                metavar="QUERY",
                help="Rewind episodic timeline by query",
            )
            p_org.add_argument(
                "--decay",
                action="store_true",
                help="Apply Ebbinghaus salience decay and archive stale memories",
            )
            p_org.add_argument(
                "--decompose",
                dest="decompose",
                metavar="TRANSCRIPT",
                help="Decompose voice memo transcript into atomic notes, tickets, and questions",
            )
        if organ_key == "immune":
            p_org.add_argument(
                "--window",
                default="all",
                help="Filter audit records by time window (e.g. 1h, 2h, 24h, all; default: all)",
            )
            p_org.add_argument(
                "--tools",
                action="store_true",
                help="Breakdown hook invocations by tool name (counts, allowed, intercepted, percentages)",
            )
            p_org.add_argument(
                "--blocked",
                action="store_true",
                help="Show top blocked files / commands and their reasons",
            )

    return parser


def setup_parser() -> argparse.ArgumentParser:
    """Construct and return the CLI argument parser."""
    return build_parser()
