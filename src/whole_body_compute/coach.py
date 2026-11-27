"""Adaptive Progress & Coach Engine (`body coach` / `/api/coach`).

Synthesizes live hardware headroom, license/version readiness, 30d FinOps audit
waste vs. live Blood-Brain Barrier shield savings, per-harness hook coverage,
skills/Turn-0 overhead, and nocturnal REM / reflex gaps into an adaptive coaching
scorecard for both:
  - `starter_trial`: 14-day Trial / <=16GB Apple Silicon users (Zero-RAM safe)
  - `power_builder`: Experienced / 100+ shield event builders (deep gap telemetry)
"""

from __future__ import annotations

import os
import platform
import subprocess
import time
from typing import Any, Dict, List, Optional

import psutil

from whole_body_compute import __version__

_STARTER_GRADUATION_SHIELD_EVENTS = 100


def _detect_hardware_profile() -> Dict[str, Any]:
    """Read local Apple Silicon chip and Unified RAM capacity cleanly."""
    chip = "Apple Silicon"
    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                timeout=0.8,
            ).decode("utf-8", errors="ignore").strip()
            if out:
                chip = out
        except Exception:
            pass
    else:
        chip = platform.processor() or platform.machine() or "Unknown CPU"

    mem = psutil.virtual_memory()
    total_gb = round(mem.total / (1024**3), 1)
    free_gb = round(mem.available / (1024**3), 1)
    used_pct = round(float(mem.percent), 1)
    is_low_ram = total_gb <= 16.5
    p_cores = psutil.cpu_count(logical=False) or 8
    return {
        "chip": chip,
        "physical_cores": int(p_cores),
        "ram_total_gb": total_gb,
        "ram_free_gb": free_gb,
        "ram_used_pct": used_pct,
        "is_low_ram_mac": is_low_ram,
        "ram_tier_label": "16GB Zero-RAM Safe" if is_low_ram else f"{int(round(total_gb))}GB Metal",
        "zero_ram_only": is_low_ram,
    }


def _safe_license_status() -> Dict[str, Any]:
    try:
        from whole_body_compute.license import LicenseManager

        st = LicenseManager().get_status()
        if isinstance(st, dict):
            return st
    except Exception:
        pass
    return {
        "tier": "community",
        "tier_name": "Community",
        "mode": "fallback",
        "license_key_masked": None,
        "expires_at": None,
        "pro_plugin": {"installed": False, "mode": "unknown"},
    }


def _safe_update_status() -> Dict[str, Any]:
    try:
        from whole_body_compute.updater import check_for_update

        res = check_for_update(force=False, timeout=0.6)
        if isinstance(res, dict):
            return res
    except Exception:
        pass
    return {
        "status": "unknown",
        "current_version": __version__,
        "latest_version": __version__,
        "update_available": False,
        "upgrade_command": "uv tool upgrade whole-body-compute",
    }


def _safe_audit_report(days: int, include_benchmarks: bool) -> Dict[str, Any]:
    try:
        from whole_body_compute.finops_audit import audit_historical_spend

        res = audit_historical_spend(
            days=days,
            include_benchmarks=include_benchmarks,
        )
        if isinstance(res, dict):
            return res
    except Exception:
        pass
    return {}


def _safe_shield_summary(window: Optional[str], include_benchmarks: bool) -> Dict[str, Any]:
    try:
        from whole_body_compute.shield import get_savings_summary

        res = get_savings_summary(window=window, include_benchmarks=include_benchmarks)
        if isinstance(res, dict):
            return res
    except Exception:
        pass
    return {}


def _safe_coverage_gaps(hours: float, include_benchmarks: bool) -> Dict[str, Any]:
    try:
        from whole_body_compute.shield import get_coverage_gaps

        res = get_coverage_gaps(hours=hours, include_benchmarks=include_benchmarks)
        if isinstance(res, dict):
            return res
    except Exception:
        pass
    return {}


def _safe_skills_summary() -> Dict[str, Any]:
    try:
        from whole_body_compute.skills_telemetry import summarize_installed_skills_telemetry

        res = summarize_installed_skills_telemetry()
        if isinstance(res, dict):
            return res
    except Exception:
        pass
    return {}


def _safe_usage_stats() -> Dict[str, Any]:
    try:
        from whole_body_compute.procedural_telemetry import get_comprehensive_usage_stats_report

        res = get_comprehensive_usage_stats_report(include_cloud=False)
        if isinstance(res, dict):
            return res
    except Exception:
        pass
    return {}


def _safe_candidates() -> Dict[str, Any]:
    try:
        from whole_body_compute.integrations.ambient_discovery import list_staged_candidates

        res = list_staged_candidates()
        if isinstance(res, dict):
            return res
    except Exception:
        pass
    return {"staged_candidates_count": 0, "active_recipes_count": 0, "candidates": []}


def _safe_hooks_status() -> Dict[str, Any]:
    try:
        from whole_body_compute.doctor import _check_workspace_hooks_wired

        res = _check_workspace_hooks_wired()
        if isinstance(res, dict):
            return res
    except Exception:
        pass
    return {"hooks_wired": False, "agents_hooks_wired": False, "claude_hooks_wired": False}


def _normalize_harness_bot_key(harness_key: str) -> str:
    mapping = {
        "claude_code": "claude-code",
        "antigravity": "antigravity",
        "cursor": "cursor",
        "codex": "codex",
        "windsurf": "windsurf",
        "gemini_cli": "gemini-cli",
        "cline_roo": "cline",
        "goose": "goose",
    }
    return mapping.get(harness_key, harness_key)


def _build_harness_coverage(
    audit_rep: Dict[str, Any],
    shield_all: Dict[str, Any],
    hooks_info: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Reconcile audited harnesses with live Shield ledger events and local hook wiring."""
    hb = audit_rep.get("harness_breakdown") or {}
    by_bot = shield_all.get("by_bot") or {}
    agents_wired = bool(hooks_info.get("agents_hooks_wired"))
    claude_wired = bool(hooks_info.get("claude_hooks_wired"))
    any_wired = bool(hooks_info.get("hooks_wired"))

    rows: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()

    for h_key, h_data in hb.items():
        if not isinstance(h_data, dict):
            continue
        threads = int(h_data.get("threads", 0) or 0)
        bot_key = _normalize_harness_bot_key(h_key)
        bot_stat = by_bot.get(bot_key) or by_bot.get(h_key) or {}
        shield_events = int(bot_stat.get("events", 0) or 0)
        shield_usd = round(float(bot_stat.get("usd_saved", 0.0) or 0.0), 2)
        if threads == 0 and shield_events == 0 and h_key not in ("antigravity", "claude_code", "codex", "cursor"):
            continue
        seen_keys.add(h_key)

        is_cloud = h_key in ("claude_chat", "claude_web")
        if is_cloud:
            hook_status = "cloud_unhookable"
            hook_label = "Cloud/Web (no local hooks)"
        elif shield_events > 0:
            hook_status = "shielded"
            hook_label = "Active & Shielding"
        elif (h_key == "claude_code" and claude_wired) or (
            h_key in ("antigravity", "codex", "gemini_cli") and agents_wired
        ) or any_wired:
            hook_status = "wired_idle"
            hook_label = "Hooks Wired (awaiting local turns)"
        else:
            hook_status = "unwired"
            hook_label = "Needs `body begin`"

        rows.append(
            {
                "harness": h_key,
                "label": str(h_data.get("label") or h_key),
                "threads": threads,
                "steps": int(h_data.get("steps", 0) or 0),
                "wasted_usd": round(float(h_data.get("blended_savable_usd", 0.0) or 0.0), 2),
                "shield_events": shield_events,
                "shield_usd_saved": shield_usd,
                "hook_status": hook_status,
                "hook_label": hook_label,
                "local_hookable": not is_cloud,
            }
        )

    for b_key, b_stat in by_bot.items():
        if not isinstance(b_stat, dict):
            continue
        norm_h = b_key.replace("-", "_")
        if norm_h in seen_keys or b_key in seen_keys:
            continue
        rows.append(
            {
                "harness": norm_h,
                "label": b_key,
                "threads": 0,
                "steps": 0,
                "wasted_usd": 0.0,
                "shield_events": int(b_stat.get("events", 0) or 0),
                "shield_usd_saved": round(float(b_stat.get("usd_saved", 0.0) or 0.0), 2),
                "hook_status": "shielded",
                "hook_label": "Active & Shielding",
                "local_hookable": True,
            }
        )

    rows.sort(key=lambda r: (r["shield_events"] + r["threads"] * 10), reverse=True)
    return rows


def _build_audit_vs_shield_explainer(
    audit_summary: Dict[str, Any],
    shield_all: Dict[str, Any],
    harness_coverage: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Explain the difference between historical 30d Audit waste and live Shield savings."""
    wasted_usd = float(audit_summary.get("wasted_spend_usd", 0.0) or 0.0)
    shield_usd = float(shield_all.get("api_dollars_saved_est", 0.0) or 0.0)
    shield_events = int(shield_all.get("total_events", 0) or 0)
    avg_steps = float(audit_summary.get("avg_steps_per_thread", 0.0) or 0.0)
    has_cloud = any(not r.get("local_hookable", True) for r in harness_coverage)
    wired_idle = sum(1 for r in harness_coverage if r.get("hook_status") == "wired_idle")

    if wasted_usd >= 25.0 and shield_events < 30:
        headline = (
            f"Why Audit shows ${wasted_usd:,.2f} waste while Shield shows ${shield_usd:,.2f} saved"
        )
        reason = (
            "Your Audit scans historical 30-day transcripts BEFORE hooks were active"
            + (" (plus Claude Web/Cloud sessions that cannot run local hooks)" if has_cloud else "")
            + f", where threads averaged {avg_steps:.0f} steps. Shield only counts live local tool interceptions since hooks loaded"
            + (f" ({wired_idle} harness(es) wired & waiting for fresh local turns)." if wired_idle else ".")
        )
    elif shield_events >= 100:
        headline = (
            f"Live Shield has already blocked ${shield_usd:,.2f} across {shield_events:,} local gates"
        )
        reason = (
            f"Audit tracks remaining historical/unflushed bloat (${wasted_usd:,.2f} across "
            f"{audit_summary.get('conversations_scanned', 0):,} threads); Shield measures live hook savings."
        )
    else:
        headline = (
            f"Audit (${wasted_usd:,.2f} opportunity) vs. Live Shield (${shield_usd:,.2f} saved)"
        )
        reason = (
            "Audit measures past thread bloat across local transcripts; Shield tracks live `<5ms` AST/hook savings going forward."
        )

    return {
        "headline": headline,
        "explanation": reason,
        "audit_wasted_usd": round(wasted_usd, 2),
        "shield_saved_usd": round(shield_usd, 2),
        "shield_events": shield_events,
        "cloud_note": (
            "Claude.ai / Web sessions do not run local OS hooks - use local Claude Code / Antigravity / Codex or import exports via `body audit --import`."
        ),
    }


def _build_starter_trial_actions(
    *,
    hardware: Dict[str, Any],
    license_info: Dict[str, Any],
    update_info: Dict[str, Any],
    audit_summary: Dict[str, Any],
    shield_all: Dict[str, Any],
    usage_stats: Dict[str, Any],
    harness_coverage: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build Zero-RAM safe, high-ROI next actions for Trial / <=16GB users."""
    actions: List[Dict[str, Any]] = []
    is_low_ram = bool(hardware.get("is_low_ram_mac"))
    tier = str(license_info.get("tier") or "community").lower()
    cmds = (usage_stats.get("usage_ledger") or {}).get("commands") or {}
    evts = (usage_stats.get("usage_ledger") or {}).get("events") or {}
    mem = usage_stats.get("memory_learning") or {}

    if bool(update_info.get("update_available")):
        actions.append(
            {
                "id": "update_version",
                "priority": 100,
                "title": f"Update Whole Body Compute ({update_info.get('current_version')} -> {update_info.get('latest_version')})",
                "why": "Pulls the latest Progress & Coach dashboard, leak fixes, and hook accuracy improvements.",
                "command": "body update",
                "impact_tag": "0 GB RAM • <15s",
                "zero_ram_safe": True,
                "category": "setup",
            }
        )

    if tier == "community":
        actions.append(
            {
                "id": "activate_pro_trial",
                "priority": 98,
                "title": "Activate your 14-Day Pro Trial key",
                "why": "Unlocks `body hud`, Hippocampus cross-session `body memory`, `body flush` state handoffs, and nocturnal `body sleep`.",
                "command": "body license --activate <YOUR_PRO_KEY>",
                "impact_tag": "Unlocks Pro HUD & Memory",
                "zero_ram_safe": True,
                "category": "activation",
            }
        )

    avg_steps = float(audit_summary.get("avg_steps_per_thread", 0.0) or 0.0)
    bloat_pct = float(audit_summary.get("bloat_pct", 0.0) or 0.0)
    wasted_usd = float(audit_summary.get("wasted_spend_usd", 0.0) or 0.0)
    if avg_steps >= 45.0 or bloat_pct >= 35.0 or wasted_usd >= 20.0:
        actions.append(
            {
                "id": "reset_marathon_threads",
                "priority": 95,
                "title": "Reset marathon threads before Turn 50",
                "why": (
                    f"Threads average {avg_steps:.0f} steps ({bloat_pct:.1f}% bloat / ${wasted_usd:,.2f} wasted re-reading old context). "
                    "Hand off state cleanly into a fresh thread instead of paying quadratic cache re-reads."
                ),
                "command": "body context --insights && body flush --auto",
                "impact_tag": f"-{min(85, max(30, int(round(bloat_pct))))}% Bloat • 0 GB RAM",
                "zero_ram_safe": True,
                "category": "bloat",
            }
        )

    total_mem = int(mem.get("total_memories", 0) or 0)
    rem_runs = int((evts.get("wbc_rem_sleep_completed") or {}).get("count", 0) or 0)
    if total_mem == 0 or rem_runs == 0:
        actions.append(
            {
                "id": "activate_hippocampus",
                "priority": 85,
                "title": "Start Hippocampus Memory & Overnight Dreaming",
                "why": "Stop re-explaining your repo conventions when opening fresh short threads.",
                "command": "body memory --status && body sleep --dry-run",
                "impact_tag": "Cross-Thread Continuity",
                "zero_ram_safe": True,
                "category": "memory",
            }
        )

    commit_runs = int((cmds.get("commit") or {}).get("count", 0) or 0)
    scout_runs = int((cmds.get("scout") or {}).get("count", 0) or 0)
    if commit_runs == 0 or scout_runs == 0:
        actions.append(
            {
                "id": "zero_ram_git_scout",
                "priority": 75,
                "title": "Stop paying diff-token tax on Git commits & big logs",
                "why": "`body commit` writes AST-checked commits locally (`$0`) and `body scout` digests big files/logs without flooding chat.",
                "command": "body commit --wip" if commit_runs == 0 else "body scout <file> -q \"summary\"",
                "impact_tag": "0 GB RAM • Pure Python/AST",
                "zero_ram_safe": True,
                "category": "workflow",
            }
        )

    unwired = [r for r in harness_coverage if r.get("hook_status") == "unwired" and r.get("local_hookable")]
    ears_runs = int((cmds.get("ears") or {}).get("count", 0) or 0)
    if unwired:
        names = ", ".join(r["label"] for r in unwired[:2])
        actions.append(
            {
                "id": "wire_harnesses",
                "priority": 70,
                "title": f"Wire local Blood-Brain Barrier hooks in {names}",
                "why": "Hooks run in `<5ms` (0 GB RAM) to slice oversized file reads before they enter your prompt.",
                "command": "body begin",
                "impact_tag": "0 GB RAM • <5ms AST Shield",
                "zero_ram_safe": True,
                "category": "hooks",
            }
        )
    elif ears_runs <= 2:
        actions.append(
            {
                "id": "voice_zero_cursor",
                "priority": 60,
                "title": "Turn on Hold-Fn local dictation (`body ears --install`)",
                "why": "Speaks code symbols accurately from your AST index with zero cloud audio upload.",
                "command": "body ears --install",
                "impact_tag": "Local ANE/MLX • Privacy First",
                "zero_ram_safe": not is_low_ram,
                "category": "voice",
            }
        )

    actions.sort(key=lambda item: int(item.get("priority", 0)), reverse=True)
    return actions[:5]


def _build_power_builder_actions(
    *,
    audit_rep: Dict[str, Any],
    gaps_rep: Dict[str, Any],
    skills_rep: Dict[str, Any],
    candidates_rep: Dict[str, Any],
    usage_stats: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build deep telemetry optimization actions for Power Builders (100+ shield events)."""
    actions: List[Dict[str, Any]] = []

    # 1. Live Shield Coverage Gaps & Reflex Gate Promotions
    gap_recs = gaps_rep.get("recommendations") or []
    shell_pfxs = gaps_rep.get("top_uncompacted_shell_prefixes") or []
    shadow_gates = gaps_rep.get("shadow_reflex_gates") or []
    graduated_gates = gaps_rep.get("graduated_reflex_gates") or []
    ready_gates = [g for g in shadow_gates if bool(g.get("ready_to_graduate"))]
    if ready_gates:
        g0 = ready_gates[0]
        actions.append(
            {
                "id": "promote_reflex_gate",
                "priority": 100,
                "title": f"Graduate ready reflex shortcut `{g0.get('gate_id')} [{g0.get('repo_key')}]`",
                "why": f"{g0.get('sample_count', 0)} samples at {float(g0.get('cv_accuracy', 0.0)) * 100:.1f}% CV accuracy - ready for instant local short-circuit.",
                "command": "body shield --bootstrap-reflex",
                "impact_tag": "Reflex Promotion Ready",
                "zero_ram_safe": True,
                "category": "reflex",
            }
        )
    elif shell_pfxs or gap_recs:
        top_pfx = shell_pfxs[0] if shell_pfxs else {}
        why_txt = str(gap_recs[0]) if gap_recs else (
            f"Top uncompacted shell prefix `{top_pfx.get('prefix')}` ({top_pfx.get('count', 0)}x untrimmed)."
        )
        actions.append(
            {
                "id": "close_shield_gaps",
                "priority": 94,
                "title": "Close live Shield & `run_command` output leaks",
                "why": why_txt,
                "command": "body savings --gaps",
                "impact_tag": f"{len(graduated_gates)} graduated / {len(shadow_gates)} shadow",
                "zero_ram_safe": True,
                "category": "gaps",
            }
        )

    # 2. Turn-0 & Unread Skills Bloat
    unread_cnt = int(skills_rep.get("unread_skills_count", 0) or 0)
    unread_tok = int(skills_rep.get("unread_turn0_tokens_per_turn", 0) or 0)
    installed_cnt = int(skills_rep.get("installed_skills_count", 0) or 0)
    if unread_cnt >= 5 and unread_tok > 0:
        actions.append(
            {
                "id": "prune_unread_skills",
                "priority": 90,
                "title": f"Archive {unread_cnt} unread skills burning {unread_tok:,} Turn-0 tok/turn",
                "why": f"{unread_cnt} of {installed_cnt} installed skills have 0 reads across audited transcripts.",
                "command": "body skills audit",
                "impact_tag": f"-{unread_tok:,} tok/turn Turn-0 Tax",
                "zero_ram_safe": True,
                "category": "turn0",
            }
        )

    # 3. Live Marathon Outliers
    top_threads = audit_rep.get("top_bloated_conversations") or []
    outliers = [t for t in top_threads if int(t.get("steps", 0) or 0) >= 60]
    if outliers:
        t0 = outliers[0]
        actions.append(
            {
                "id": "flush_marathon_outlier",
                "priority": 86,
                "title": f"Flush top marathon outlier `{t0.get('short_id', 'thread')}` ({int(t0.get('steps', 0))} steps)",
                "why": f"Estimated ${float(t0.get('estimated_spend_usd', 0.0)):,.2f} spend (${float(t0.get('usd_savable', 0.0)):,.2f} savable) via `{t0.get('top_tool', 'read_file')}` bloat.",
                "command": f"body flush --conv-id {t0.get('short_id', '')}",
                "impact_tag": f"-${float(t0.get('usd_savable', 0.0)):,.2f} Savable",
                "zero_ram_safe": True,
                "category": "bloat",
            }
        )

    # 4. Staged Ambient Candidates & REM Sleep
    staged_cnt = int(candidates_rep.get("staged_candidates_count", 0) or 0)
    cands = candidates_rep.get("candidates") or []
    evts = (usage_stats.get("usage_ledger") or {}).get("events") or {}
    rem_runs = int((evts.get("wbc_rem_sleep_completed") or {}).get("count", 0) or 0)
    if staged_cnt > 0 and cands:
        cid = str(cands[0].get("id") or "candidate")
        actions.append(
            {
                "id": "promote_ambient_candidate",
                "priority": 82,
                "title": f"Promote {staged_cnt} staged ambient recipe(s) (`{cid}`)",
                "why": f"Detected repeatable desktop/workflow trajectory ({rem_runs} REM pass(es) recorded).",
                "command": f"body candidates --promote {cid}",
                "impact_tag": "Zero-Token Automation",
                "zero_ram_safe": True,
                "category": "ambient",
            }
        )

    # 5. Model Tier Spend Mix Opportunity
    model_opps = (audit_rep.get("model_opportunities") or {}).get("opportunities") or []
    if model_opps:
        opp = model_opps[0]
        actions.append(
            {
                "id": "optimize_model_mix",
                "priority": 78,
                "title": str(opp.get("title") or "Optimize Frontier Model Tier Mix"),
                "why": str(opp.get("summary") or "Downshift routine tool loops to lighter/local tiers."),
                "command": "body context --by-model",
                "impact_tag": f"~${float(opp.get('savable_usd', 0.0)):,.2f} Savable",
                "zero_ram_safe": True,
                "category": "models",
            }
        )

    actions.sort(key=lambda item: int(item.get("priority", 0)), reverse=True)
    return actions[:5]


def build_coach_report(
    *,
    mode: str = "auto",
    window_days: int = 30,
    shield_window: str = "all",
    gaps_hours: float = 5.0,
    include_benchmarks: bool = False,
    hardware_override: Optional[Dict[str, Any]] = None,
    license_override: Optional[Dict[str, Any]] = None,
    update_override: Optional[Dict[str, Any]] = None,
    audit_override: Optional[Dict[str, Any]] = None,
    shield_all_override: Optional[Dict[str, Any]] = None,
    shield_24h_override: Optional[Dict[str, Any]] = None,
    shield_7d_override: Optional[Dict[str, Any]] = None,
    gaps_override: Optional[Dict[str, Any]] = None,
    skills_override: Optional[Dict[str, Any]] = None,
    usage_override: Optional[Dict[str, Any]] = None,
    candidates_override: Optional[Dict[str, Any]] = None,
    hooks_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the stage-adaptive, RAM-aware Progress & Coach report."""
    hardware = dict(hardware_override) if hardware_override is not None else _detect_hardware_profile()
    license_info = dict(license_override) if license_override is not None else _safe_license_status()
    update_info = dict(update_override) if update_override is not None else _safe_update_status()
    audit_rep = dict(audit_override) if audit_override is not None else _safe_audit_report(window_days, include_benchmarks)
    shield_all = dict(shield_all_override) if shield_all_override is not None else _safe_shield_summary(shield_window, include_benchmarks)
    shield_24h = dict(shield_24h_override) if shield_24h_override is not None else _safe_shield_summary("24h", include_benchmarks)
    shield_7d = dict(shield_7d_override) if shield_7d_override is not None else _safe_shield_summary("7d", include_benchmarks)
    gaps_rep = dict(gaps_override) if gaps_override is not None else _safe_coverage_gaps(gaps_hours, include_benchmarks)
    skills_rep = dict(skills_override) if skills_override is not None else _safe_skills_summary()
    usage_stats = dict(usage_override) if usage_override is not None else _safe_usage_stats()
    candidates_rep = dict(candidates_override) if candidates_override is not None else _safe_candidates()
    hooks_info = dict(hooks_override) if hooks_override is not None else _safe_hooks_status()

    shield_events = int(shield_all.get("total_events", 0) or 0)
    tier = str(license_info.get("tier") or "community").lower()

    req_mode = (mode or "auto").strip().lower()
    if req_mode in ("trial", "starter", "starter_trial"):
        effective_mode = "starter_trial"
    elif req_mode in ("power", "builder", "power_builder"):
        effective_mode = "power_builder"
    else:
        effective_mode = (
            "power_builder"
            if (shield_events >= _STARTER_GRADUATION_SHIELD_EVENTS and tier != "community")
            else "starter_trial"
        )

    convs_scanned = int(audit_rep.get("conversations_scanned", 0) or 0)
    total_steps = int(audit_rep.get("total_steps", 0) or 0)
    avg_steps = round(total_steps / max(1, convs_scanned), 1) if convs_scanned > 0 else 0.0
    bloat_pct = round(float(audit_rep.get("savings_pct", 0.0) or 0.0), 1)

    actual_in_tok = max(0, int(audit_rep.get("actual_cumulative_input_tokens", 0) or 0))
    quad_bloat_tok = max(0, int(audit_rep.get("quadratic_post_saturation_tokens", 0) or 0))
    tool_bloat_tok = max(0, int(audit_rep.get("unfiltered_tool_bloat_tokens", 0) or 0))
    if quad_bloat_tok + tool_bloat_tok > actual_in_tok and actual_in_tok > 0:
        scale = float(actual_in_tok) / float(quad_bloat_tok + tool_bloat_tok)
        quad_bloat_tok = int(round(quad_bloat_tok * scale))
        tool_bloat_tok = max(0, actual_in_tok - quad_bloat_tok)
    typed_core_tok = max(0, actual_in_tok - quad_bloat_tok - tool_bloat_tok)
    denom_tok = max(1, actual_in_tok)

    cost_mix = {
        "total_input_tokens": actual_in_tok,
        "typed_prompt_tokens": typed_core_tok,
        "typed_prompt_pct": round((typed_core_tok / denom_tok) * 100.0, 1) if actual_in_tok else 0.0,
        "tool_bloat_tokens": tool_bloat_tok,
        "tool_bloat_pct": round((tool_bloat_tok / denom_tok) * 100.0, 1) if actual_in_tok else 0.0,
        "cache_reread_tokens": quad_bloat_tok,
        "cache_reread_pct": round((quad_bloat_tok / denom_tok) * 100.0, 1) if actual_in_tok else 0.0,
    }

    audit_summary = {
        "window_days": int(audit_rep.get("window_days", window_days) or window_days),
        "conversations_scanned": convs_scanned,
        "excluded_benchmark_conversations": int(audit_rep.get("excluded_benchmark_conversations", 0) or 0),
        "total_steps": total_steps,
        "avg_steps_per_thread": avg_steps,
        "bloat_pct": bloat_pct,
        "estimated_frontier_spend_usd": round(float(audit_rep.get("estimated_frontier_spend_usd", 0.0) or 0.0), 2),
        "wasted_spend_usd": round(float(audit_rep.get("wasted_spend_usd", 0.0) or 0.0), 2),
        "bbb_optimized_spend_usd": round(float(audit_rep.get("bbb_optimized_spend_usd", 0.0) or 0.0), 2),
        "net_savings_usd": round(float(audit_rep.get("net_savings_usd", 0.0) or 0.0), 2),
    }

    shield_windows = {
        "today": {
            "events": int(shield_24h.get("total_events", 0) or 0),
            "tokens_saved": int(shield_24h.get("cumulative_tokens_saved", 0) or 0),
            "direct_tokens_saved": int(shield_24h.get("direct_tokens_saved", 0) or 0),
            "usd_saved": round(float(shield_24h.get("api_dollars_saved_est", 0.0) or 0.0), 2),
            "time_saved_display": str(shield_24h.get("time_saved_display") or "0.0s"),
        },
        "7d": {
            "events": int(shield_7d.get("total_events", 0) or 0),
            "tokens_saved": int(shield_7d.get("cumulative_tokens_saved", 0) or 0),
            "direct_tokens_saved": int(shield_7d.get("direct_tokens_saved", 0) or 0),
            "usd_saved": round(float(shield_7d.get("api_dollars_saved_est", 0.0) or 0.0), 2),
            "time_saved_display": str(shield_7d.get("time_saved_display") or "0.0s"),
        },
        "all": {
            "events": shield_events,
            "tokens_saved": int(shield_all.get("cumulative_tokens_saved", 0) or 0),
            "direct_tokens_saved": int(shield_all.get("direct_tokens_saved", 0) or 0),
            "usd_saved": round(float(shield_all.get("api_dollars_saved_est", 0.0) or 0.0), 2),
            "time_saved_sec": round(float(shield_all.get("time_saved_sec", 0.0) or 0.0), 1),
            "time_saved_display": str(shield_all.get("time_saved_display") or "0.0s"),
            "bots_protected": list(shield_all.get("bots_protected") or []),
        },
    }

    harness_coverage = _build_harness_coverage(audit_rep, shield_all, hooks_info)
    explainer = _build_audit_vs_shield_explainer(audit_summary, shield_all, harness_coverage)

    if effective_mode == "starter_trial":
        coach_actions = _build_starter_trial_actions(
            hardware=hardware,
            license_info=license_info,
            update_info=update_info,
            audit_summary=audit_summary,
            shield_all=shield_all,
            usage_stats=usage_stats,
            harness_coverage=harness_coverage,
        )
    else:
        coach_actions = _build_power_builder_actions(
            audit_rep=audit_rep,
            gaps_rep=gaps_rep,
            skills_rep=skills_rep,
            candidates_rep=candidates_rep,
            usage_stats=usage_stats,
        )

    # Power-builder deep diagnostics (also safe to inspect via JSON)
    top_outliers = [
        {
            "short_id": str(t.get("short_id") or ""),
            "harness": str(t.get("harness") or "antigravity"),
            "steps": int(t.get("steps", 0) or 0),
            "csi": float(t.get("csi", 0.0) or 0.0),
            "health_tier": str(t.get("health_tier") or "GREEN"),
            "estimated_spend_usd": round(float(t.get("estimated_spend_usd", 0.0) or 0.0), 2),
            "usd_savable": round(float(t.get("usd_savable", 0.0) or 0.0), 2),
            "top_tool": str(t.get("top_tool") or "none"),
        }
        for t in (audit_rep.get("top_bloated_conversations") or [])[:5]
    ]

    turn0_audit = audit_rep.get("turn0_static_overhead") or {}
    skills_block = {
        "installed_skills_count": int(skills_rep.get("installed_skills_count", turn0_audit.get("installed_skills_count", 0)) or 0),
        "unread_skills_count": int(skills_rep.get("unread_skills_count", 0) or 0),
        "unread_skill_names": list(skills_rep.get("unread_skill_names") or [])[:15],
        "turn0_skills_tokens_per_turn": int(skills_rep.get("turn0_skills_tokens_per_turn", turn0_audit.get("skills_turn0_tokens", 0)) or 0),
        "unread_turn0_tokens_per_turn": int(skills_rep.get("unread_turn0_tokens_per_turn", 0) or 0),
        "total_turn0_tokens_per_session": int(turn0_audit.get("total_turn0_tokens_per_session", 0) or 0),
        "turn0_tier": str(turn0_audit.get("status_tier") or "LEAN"),
    }

    evts = (usage_stats.get("usage_ledger") or {}).get("events") or {}
    mem = usage_stats.get("memory_learning") or {}
    ambient_and_rem = {
        "staged_candidates_count": int(candidates_rep.get("staged_candidates_count", 0) or 0),
        "active_recipes_count": int(candidates_rep.get("active_recipes_count", 0) or 0),
        "staged_candidates": list(candidates_rep.get("candidates") or [])[:5],
        "rem_sleep_runs": int((evts.get("wbc_rem_sleep_completed") or {}).get("count", 0) or 0),
        "total_memories": int(mem.get("total_memories", 0) or 0),
        "state_capsules_saved": int(mem.get("state_capsules_saved", 0) or 0),
        "autonomy_ratchet": audit_rep.get("autonomy_ratchet") or {},
    }

    coverage_gaps_block = {
        "hours": float(gaps_rep.get("hours", gaps_hours) or gaps_hours),
        "events_scanned": int(gaps_rep.get("events_scanned", 0) or 0),
        "tool_capture": list(gaps_rep.get("tool_capture") or [])[:6],
        "unbounded_read_leakage": gaps_rep.get("unbounded_read_leakage") or {},
        "top_uncompacted_shell_prefixes": list(gaps_rep.get("top_uncompacted_shell_prefixes") or [])[:5],
        "graduated_reflex_gates": list(gaps_rep.get("graduated_reflex_gates") or []),
        "shadow_reflex_gates": list(gaps_rep.get("shadow_reflex_gates") or [])[:5],
        "recommendations": list(gaps_rep.get("recommendations") or [])[:3],
    }

    model_opps_raw = audit_rep.get("model_opportunities") or {}
    model_mix = {
        "tagged_sessions": int(model_opps_raw.get("tagged_sessions", 0) or 0),
        "tagged_spend_usd": round(float(model_opps_raw.get("tagged_spend_usd", 0.0) or 0.0), 2),
        "tier_breakdown": model_opps_raw.get("tier_breakdown") or {},
        "top_models": list(model_opps_raw.get("models") or [])[:5],
        "opportunities": list(model_opps_raw.get("opportunities") or [])[:3],
    }

    return {
        "status": "ok",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": __version__,
        "mode": effective_mode,
        "mode_label": "14-Day Trial Coach" if effective_mode == "starter_trial" else "Power Builder Coach",
        "graduation_threshold_events": _STARTER_GRADUATION_SHIELD_EVENTS,
        "include_benchmarks": bool(include_benchmarks),
        "hardware": hardware,
        "license": {
            "tier": tier,
            "tier_name": license_info.get("tier_name", "Community"),
            "mode": license_info.get("mode", "community_default"),
            "license_key_masked": license_info.get("license_key_masked"),
            "expires_at": license_info.get("expires_at"),
            "pro_plugin_installed": bool((license_info.get("pro_plugin") or {}).get("installed", False)),
        },
        "update": update_info,
        "audit_summary": audit_summary,
        "cost_mix": cost_mix,
        "shield_windows": shield_windows,
        "audit_vs_shield_explainer": explainer,
        "harness_coverage": harness_coverage,
        "coach_actions": coach_actions,
        "power_diagnostics": {
            "coverage_gaps": coverage_gaps_block,
            "skills_and_turn0": skills_block,
            "marathon_outliers": top_outliers,
            "ambient_and_rem": ambient_and_rem,
            "model_mix": model_mix,
        },
    }


def format_coach_report(report: Dict[str, Any]) -> str:
    """Render an adaptive terminal Progress & Coach scorecard for `body coach`."""
    hw = report.get("hardware") or {}
    lic = report.get("license") or {}
    upd = report.get("update") or {}
    aud = report.get("audit_summary") or {}
    mix = report.get("cost_mix") or {}
    sw = report.get("shield_windows") or {}
    sw_24 = sw.get("today") or {}
    sw_all = sw.get("all") or {}
    exp = report.get("audit_vs_shield_explainer") or {}
    cov = report.get("harness_coverage") or []
    actions = report.get("coach_actions") or []
    pdiag = report.get("power_diagnostics") or {}

    mode_lbl = report.get("mode_label", "Coach")
    upd_note = (
        f"update -> v{upd.get('latest_version')}"
        if upd.get("update_available")
        else f"v{report.get('version', __version__)}"
    )

    lines = [
        "=" * 88,
        f"  🎯 WHOLE BODY COMPUTE - {mode_lbl.upper()} (`body coach`)",
        f"  Hardware: {hw.get('chip', 'Apple Silicon')} ({hw.get('ram_total_gb', 0)} GB RAM · {hw.get('ram_free_gb', 0)} GB Free · {hw.get('ram_tier_label', '')})",
        f"  License:  {lic.get('tier_name', 'Community')} ({lic.get('tier', 'community')}) | Version: {upd_note}",
        "=" * 88,
        "",
        "  [1] Universal Progress & Bloat Scorecard (Tests/Benchmarks Excluded):",
        f"      • 30d Thread Audit:   {aud.get('conversations_scanned', 0):,} threads | avg {aud.get('avg_steps_per_thread', 0.0):.1f} steps/thread | {aud.get('bloat_pct', 0.0):.1f}% bloat (${aud.get('wasted_spend_usd', 0.0):,.2f} wasted)",
        f"      • True Cost Mix:      {mix.get('typed_prompt_pct', 0.0):.1f}% core prompt | {mix.get('tool_bloat_pct', 0.0):.1f}% tool output bloat | {mix.get('cache_reread_pct', 0.0):.1f}% quadratic re-reads",
        f"      • Live Shield Value:  Today ${sw_24.get('usd_saved', 0.0):,.2f} ({sw_24.get('events', 0):,} gates) | All-Time ${sw_all.get('usd_saved', 0.0):,.2f} ({sw_all.get('events', 0):,} gates, {sw_all.get('time_saved_display', '0.0s')})",
        "",
        f"  [2] {exp.get('headline', 'Audit vs. Shield Reconciliation')}:",
        f"      {exp.get('explanation', '')}",
        "",
        "  [3] Harness & Hook Coverage:",
    ]

    if cov:
        for row in cov[:6]:
            lines.append(
                f"      • {row.get('label', row.get('harness')):<24} "
                f"{int(row.get('threads', 0)):>4} threads | "
                f"{int(row.get('shield_events', 0)):>5} shielded (${float(row.get('shield_usd_saved', 0.0)):,.2f}) | "
                f"{row.get('hook_label', '')}"
            )
    else:
        lines.append("      • No local sessions scanned yet - run `body begin` in a repo to wire hooks.")

    lines.append("")
    lines.append(f"  [4] Ranked Next Actions ({mode_lbl}):")
    for idx, act in enumerate(actions, start=1):
        lines.append(f"      #{idx} {act.get('title', '')}  [{act.get('impact_tag', '')}]")
        lines.append(f"          Why: {act.get('why', '')}")
        lines.append(f"          Run: `{act.get('command', '')}`")

    if report.get("mode") == "power_builder":
        sk = pdiag.get("skills_and_turn0") or {}
        amb = pdiag.get("ambient_and_rem") or {}
        gaps = pdiag.get("coverage_gaps") or {}
        lines.extend(
            [
                "",
                "  [5] Power Builder Deep Telemetry:",
                f"      • Turn-0 & Skills:    {sk.get('installed_skills_count', 0)} installed | {sk.get('unread_skills_count', 0)} unread (0 reads, {sk.get('unread_turn0_tokens_per_turn', 0):,} tok/turn) | Tier {sk.get('turn0_tier', 'LEAN')}",
                f"      • Ambient & Memory:   {amb.get('staged_candidates_count', 0)} staged candidates | {amb.get('rem_sleep_runs', 0)} REM sleep pass(es) | {amb.get('total_memories', 0)} memories",
                f"      • Shield Gaps ({gaps.get('hours', 5.0):g}h):  {gaps.get('events_scanned', 0):,} audit events | {len(gaps.get('graduated_reflex_gates', []))} graduated / {len(gaps.get('shadow_reflex_gates', []))} shadow reflex gates",
            ]
        )

    lines.append("=" * 88)
    return "\n".join(lines)
