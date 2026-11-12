"""
Command handlers for Organ inspection, Cognitive Sentry, FinOps, Airlock, Hippocampus,
Foveal Eyes, and Git Fastpath subcommands in the Whole Body Compute CLI.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Optional

from whole_body_compute.cli_parser import SINGLE_ORGAN_DISPATCH
from whole_body_compute.doctor import (
    check_organ_immune_system,
    format_organ_doctor_table,
)


def render_pro_upgrade_card(
    cmd: str,
    exc: BaseException,
    is_json: bool = False,
) -> int:
    """
    Render a clean, friendly PLG Upgrade Card (or structured JSON payload) when an
    Open-Core Community user invokes a Pro ($29/mo) or Team ($79/seat/mo) command.
    Returns exit code 2 (zero raw Python stack trace).
    """
    is_team_cmd = cmd in ("team-sync", "hive-sync", "share-exemplars")
    required_tier = "team" if is_team_cmd else "pro"
    tier_label = "Team Hive-Mind ($79/seat/mo)" if is_team_cmd else "Pro ($29/mo)"
    monthly_usd = 79.0 if is_team_cmd else 29.0
    upgrade_url = "https://wholebodycompute.com/pricing"
    activate_cmd = "body license --activate <POLAR_KEY>"
    raw_msg = str(exc).strip() or f"Requires Whole Body Compute {tier_label} -> {upgrade_url}"
    companion_missing = False

    try:
        from whole_body_compute.license import LicenseManager
        from whole_body_compute.pro_loader import discover_pro_plugin_status

        gate_key = "team_sync" if is_team_cmd else "glymphatic_flush"
        if LicenseManager().is_feature_allowed(gate_key) and not discover_pro_plugin_status().get("distribution_installed"):
            companion_missing = True
            activate_cmd = "body license --reinstall-wheel"
            raw_msg = (
                "Active license verified, but `whole-body-compute-pro` is not installed in this Python environment. "
                "Run: body license --reinstall-wheel"
            )
    except Exception:
        companion_missing = False

    if is_json:
        payload_out = {
            "status": "upgrade_required",
            "command": cmd,
            "required_tier": required_tier,
            "tier_label": tier_label,
            "monthly_usd": monthly_usd,
            "message": raw_msg,
            "activate_command": activate_cmd,
            "upgrade_url": upgrade_url,
            "swebench_k3_benchmark": {
                "context_efficiency_multiple": 1.48,
                "context_reduction_pct": 32.4,
                "net_task_context_efficiency_multiple": 1.52,
                "net_task_context_reduction_pct": 34.3,
                "billed_input_runway_multiple": 1.34,
                "billed_input_reduction_pct": 25.6,
                "turn_reduction_pct": 18.8,
                "pass_pow_3_pct": 100.0,
            },
        }
        if companion_missing:
            payload_out["reason"] = "companion_wheel_missing"
        print(json.dumps(payload_out, indent=2))
        return 2

    cmd_display = f"`body {cmd}`"
    lines = [
        "",
        "╔══════════════════════════════════════════════════════════════════════════════╗",
        "║         WHOLE BODY COMPUTE - PRO / TEAM SOVEREIGN UPGRADE REQUIRED           ║",
        "╠══════════════════════════════════════════════════════════════════════════════╣",
        f"║  Command:        {cmd_display:<59} ║",
        f"║  Required Tier:  {tier_label:<59} ║",
        "╠══════════════════════════════════════════════════════════════════════════════╣",
        "║  Open-Core Community includes unlimited `body audit-spend`, `body doctor`,  ║",
        "║  `body begin`, `body eyes`, `body hands`, 50/day BBB slices & 25 offloads.  ║",
        "║                                                                              ║",
        "║  SWE-bench_Verified (k=3, 60 Trials, N=10 Repos, 100% Pass^3 30/30):         ║",
        "║    • 1.48x Context (-32.4%, 203.08M->137.23M) • 1.52x Net Context (-34.3%)   ║",
        "║    • 1.34x Billed Input Runway (-25.6%, -16.7% med) • -18.8% Agent Turns    ║",
        "║                                                                              ║",
        "║  Unlock Glymphatic Context Flush, 3:30 AM REM LoRA Distillation, Unlimited  ║",
        "║  In-Flight BBB Hooks, Promoted Recipes & Team Hive-Mind Sync:               ║",
        "║                                                                              ║",
        f"║    1. Upgrade Online:    {upgrade_url:<51} ║",
        f"║    2. Activate Key:      {activate_cmd:<51} ║",
        "╚══════════════════════════════════════════════════════════════════════════════╝",
        "",
    ]
    print("\n".join(lines))
    return 2


def cmd_immune(args: argparse.Namespace) -> int:
    """Execute Organ 05 (Immune System) check with Context-Shield audit scorecard."""
    from whole_body_compute.organs.immune import (
        format_immune_scorecard_table,
        get_immune_pre_post_scorecard,
    )

    organ_res = check_organ_immune_system()
    window = getattr(args, "window", "all") or "all"
    show_tools = getattr(args, "tools", False)
    show_blocked = getattr(args, "blocked", False)
    is_json = getattr(args, "json", False)

    sc = get_immune_pre_post_scorecard(
        limit=50,
        window=window,
        show_tools=show_tools,
        show_blocked=show_blocked,
    )

    if is_json:
        organ_res["pre_post_scorecard"] = sc
        if show_tools:
            organ_res["tools_breakdown"] = sc.get("tools_breakdown", {})
        if show_blocked:
            organ_res["top_blocked"] = sc.get("top_blocked", [])
        print(json.dumps(organ_res, indent=2))
    else:
        fake_report = {
            "timestamp": 0,
            "elapsed_ms": organ_res.get("latency_ms", 0.0),
            "overall_status": organ_res.get("status", "pass"),
            "total_organs": 1,
            "organs": [organ_res],
        }
        print(format_organ_doctor_table(fake_report))
        print(
            format_immune_scorecard_table(
                limit=10,
                window=window,
                show_tools=show_tools,
                show_blocked=show_blocked,
            )
        )
        print()

    return 0 if organ_res.get("status") != "fail" else 1


def cmd_commit(args: argparse.Namespace) -> int:
    """Sub-20ms AST-guided smart git commit with pre-flight secret/lint SecGate."""
    from whole_body_compute.local.git_fastpath import run_fast_commit

    res = run_fast_commit(
        paths=args.paths or None,
        message=args.message,
        all_tracked=args.all_tracked,
        no_lint=args.no_lint,
        no_llm=args.no_llm,
        wip=args.wip,
        session_id=args.session_id,
    )
    if getattr(args, "json", False):
        print(json.dumps(res, indent=2))
    else:
        output = res.get("receipt") or res.get("error") or res.get("status")
        if output:
            print(output)
    return 0 if res.get("status") in ("ok", "clean") else 1


def cmd_checkpoint(args: argparse.Namespace) -> int:
    """Zero-risk working-tree snapshot to refs/checkpoints/* or HEAD."""
    from whole_body_compute.local.git_fastpath import run_checkpoint

    res = run_checkpoint(
        label=args.label,
        commit_mode=args.commit,
        list_mode=getattr(args, "list", False),
        restore_ref=args.restore,
    )
    if getattr(args, "json", False):
        print(json.dumps(res, indent=2))
    elif getattr(args, "list", False) and res.get("checkpoints"):
        for line in res["checkpoints"]:
            print(line)
    else:
        output = res.get("receipt") or res.get("error")
        if output:
            print(output)
    return 0 if res.get("status") in ("ok", "clean") else 1


def _handle_by_model_leaderboard(args: argparse.Namespace) -> int:
    """Execute empirical multi-model leaderboard telemetry across transcript.jsonl logs."""
    from pathlib import Path

    from whole_body_compute.model_benchmark import (
        build_model_benchmark_report,
        format_model_comparison_markdown,
        format_model_comparison_terminal,
    )

    raw_win = getattr(args, "window", None)
    window = raw_win if raw_win else "30d"
    include_subagents = bool(getattr(args, "include_subagents", False))
    output_path = getattr(args, "output_path", None)
    show_receipt = bool(getattr(args, "receipt", False))
    is_json = bool(getattr(args, "json", False))

    report = build_model_benchmark_report(
        window=window,
        include_subagents=include_subagents,
    )

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(format_model_comparison_markdown(report), encoding="utf-8")
        report["output_path"] = str(out_p)

    if is_json:
        print(json.dumps(report, indent=2))
    elif show_receipt:
        print(format_model_comparison_markdown(report))
    else:
        print(format_model_comparison_terminal(report))
        if output_path:
            print(f"📄 Exported Multi-Model Leaderboard Markdown -> `{output_path}`\n")
    return 0


def _require_pro_cli_feature(feature_key: str = "glymphatic_flush") -> None:
    """
    Enforce Pro/Team feature gating on CLI commands (`context`, `flush`, `perfusion`, `sleep`, `rem-sleep`).
    In normal dev/founder runs on the host machine, `LicenseManager` defaults to `enterprise` so
    commands run unrestricted. In exported Open-Core or isolated new-user Community sandboxes
    (`HOME=/tmp/...`, `WBC_HOME=/tmp/...`, `WBC_FORCE_COMMUNITY=1`, or `--set-tier community` without
    `PYTEST_CURRENT_TEST`), raises `ProUpgradeRequiredError` so the CLI renders the PLG Upgrade Card.
    """
    import os

    from whole_body_compute.license import LicenseManager, _is_isolated_home_env, _is_open_core_enforced
    from whole_body_compute.pro_loader import ProUpgradeRequiredError, UPGRADE_MESSAGE

    # During unit tests in the dev monorepo (where tests exercise cognitive_sentry/supervisor directly),
    # only enforce when Open-Core stubs are active or WBC_FORCE_COMMUNITY=1 is explicitly set.
    if "PYTEST_CURRENT_TEST" in os.environ and not _is_open_core_enforced() and os.environ.get("WBC_FORCE_COMMUNITY") != "1":
        return

    mgr = LicenseManager()
    if not (_is_open_core_enforced() or _is_isolated_home_env() or mgr.license_path.is_file()):
        return

    gate = mgr.check_feature_gate(feature_key)
    if not gate.get("allowed"):
        raise ProUpgradeRequiredError(gate.get("upgrade_nudge") or UPGRADE_MESSAGE)


def dispatch_organ_and_cognitive_command(cmd: str, args: argparse.Namespace) -> Optional[int]:
    """
    Dispatch Cognitive Sentry, FinOps, Airlock, Hippocampus, Foveal Eyes, and single-organ commands.
    Returns an integer exit code if handled, or None if `cmd` was not matched.
    """
    if cmd in ("context", "csi", "watchdog"):
        _require_pro_cli_feature("glymphatic_flush")
        if getattr(args, "by_model", False):
            return _handle_by_model_leaderboard(args)

        from whole_body_compute.organs.cognitive_sentry import (
            CognitiveThermalSentry,
            UltradianGlymphaticFlush,
            format_cognitive_insights_report,
            format_context_banner,
        )

        sentry = CognitiveThermalSentry()
        show_all = getattr(args, "all", False)
        show_insights = getattr(args, "insights", False)
        target_cid = getattr(args, "conv_id", None)
        auto_flush = getattr(args, "auto", False)
        is_json = getattr(args, "json", False)

        if show_insights:
            insights = sentry.generate_longitudinal_insights(limit=getattr(args, "limit", 25))
            if is_json:
                print(json.dumps(insights, indent=2))
            else:
                print(format_cognitive_insights_report(insights))
            return 0

        if show_all:
            convs = sentry.list_active_conversations(limit=getattr(args, "limit", 5))
            if is_json:
                print(json.dumps([c.to_dict() for c in convs], indent=2))
            else:
                if not convs:
                    print("\n⚪ No active Antigravity conversations found in brain logs.\n")
                for c in convs:
                    print(format_context_banner(c))
            return 0

        metrics = sentry.inspect_conversation(target_cid)
        if not metrics:
            if is_json:
                print(
                    json.dumps(
                        {"status": "not_found", "message": "No conversation transcript found"},
                        indent=2,
                    )
                )
            else:
                print(
                    "\n⚪ No conversation transcript found. Ensure you are running within an Antigravity project.\n"
                )
            return 1

        # Record inspection in MemoryStore telemetry ledger
        try:
            from whole_body_compute.local.dot_memory import get_dot_store

            get_dot_store().record_cognitive_telemetry_event(
                metrics.to_dict(), event_type="CLI_CONTEXT_CHECK"
            )
        except Exception:
            pass

        if is_json:
            print(json.dumps(metrics.to_dict(), indent=2))
        else:
            print(format_context_banner(metrics))

        # Auto-flush if requested and saturated (YELLOW or RED)
        if auto_flush and metrics.health_tier in ("YELLOW", "RED"):
            flush_res = UltradianGlymphaticFlush.run_flush(
                conv_id=metrics.conv_id,
                auto_bridge=True,
            )
            if not is_json:
                print(
                    f"✨ [Zero-Click Handoff] State Capsule `{flush_res['capsule_id']}` saved to dot_memory.db."
                )
                if flush_res.get("new_conversation_link"):
                    print(f"🚀 New Thread:     {flush_res['new_conversation_link']}")
                if flush_res.get("handoff_announcement"):
                    print(
                        f"\n📣 [AGENT HANDOFF ANNOUNCEMENT - INCLUDE THIS LINK IN YOUR REPLY TO THE USER]:\n"
                        f"{flush_res['handoff_announcement']}\n"
                    )
                print("📋 Resumption prompt copied to macOS clipboard via pbcopy.")
                print("=" * 80)
                print(flush_res["resumption_prompt"])
                print("=" * 80 + "\n")

        return 0 if metrics.health_tier != "RED" else 1

    if cmd in ("flush", "handoff"):
        _require_pro_cli_feature("glymphatic_flush")
        from whole_body_compute.organs.cognitive_sentry import UltradianGlymphaticFlush

        target_cid = getattr(args, "conv_id", None)
        goal = getattr(args, "goal", None)
        milestone = getattr(args, "milestone", None)
        next_step = getattr(args, "next_step", None)
        artifacts = getattr(args, "artifacts", None)
        slack_thread = getattr(args, "slack_thread", None)
        auto_bridge = getattr(args, "auto", False)
        no_spawn = getattr(args, "no_spawn", False)
        explicit_spawn = getattr(args, "spawn", None)
        spawn_conv: Optional[bool] = (
            False if no_spawn else (True if explicit_spawn else None)
        )
        spawn_app = getattr(args, "spawn_app", "Antigravity") or "Antigravity"
        submit_spawned = not bool(getattr(args, "no_submit", False))
        evict_models = getattr(args, "evict_models", False)
        is_json = getattr(args, "json", False)

        res = UltradianGlymphaticFlush.run_flush(
            conv_id=target_cid,
            goal=goal,
            milestone=milestone,
            next_step=next_step,
            artifacts=artifacts,
            slack_thread=slack_thread,
            auto_bridge=auto_bridge,
            spawn_conversation=spawn_conv,
            spawn_app=spawn_app,
            submit_spawned=submit_spawned,
            evict_models=evict_models,
        )

        if is_json:
            print(json.dumps(res, indent=2))
        else:
            print("\n🧠 [Ultradian Glymphatic Flush & Context Handoff]")
            print("=" * 80)
            print(f"  • Capsule ID:     {res['capsule_id']} (Conv: {res['short_id']})")
            print(
                f"  • Saturation:     CSI {res['csi']:.3f} • Tier: {res['health_tier']} "
                f"({res['steps']} steps, {res['transcript_kb']:.0f} KB)"
            )
            print(f"  • Grounding:      {res['capsule']['git_status_summary']}")
            print(f"  • Artifacts:      {len(res['capsule']['artifact_digests'])} cryptographic SHA-256 digests")
            if res["capsule"]["slack_metadata"]:
                slack_s = (
                    res["capsule"]["slack_metadata"].get("permalink")
                    or res["capsule"]["slack_metadata"].get("thread_ts")
                )
                print(f"  • External Comms: Slack thread {slack_s}")
            print("  • Hippocampus:    Persisted to SQLite WAL (~/.whole_body_compute/dot_memory.db)")
            print(
                f"  • Metabolic Flush: {res['hygiene'].get('purged_noise', 0)} noise rows purged, "
                f"{res['hygiene'].get('purged_duplicates', 0)} duplicates removed"
            )
            if auto_bridge or spawn_conv:
                bridge_info = res.get("bridge") or {}
                new_link = res.get("new_conversation_link") or bridge_info.get("new_conversation_link")
                spawn_badge = (
                    f"🚀 Spawned fresh thread in {spawn_app}"
                    if bridge_info.get("conversation_spawned")
                    else "⚪ Auto-spawn skipped"
                )
                print(
                    f"  • Zero-Click:     📋 Copied to clipboard (pbcopy) • {spawn_badge} • ⚡ IPC emitted • 🔊 Spoken alert"
                )
                if new_link:
                    print(f"  • New Thread:     {new_link}")
            if evict_models and res.get("perfusion"):
                print(
                    f"  • Perfusion:      Reclaimed {res['perfusion'].get('reclaimed_ram_gb', 0):.1f} GB Unified RAM"
                )
            print("=" * 80)
            if res.get("handoff_announcement"):
                print(
                    "\n📣 [AGENT HANDOFF ANNOUNCEMENT - INCLUDE THIS LINK IN YOUR REPLY TO THE USER]:"
                )
                print(res["handoff_announcement"])
            print("\n📋 [Ready-to-Paste Resumption Prompt for Fresh Thread]:\n")
            print(res["resumption_prompt"])
            print("\n" + "=" * 80 + "\n")

        return 0

    if cmd == "immune":
        return cmd_immune(args)

    if cmd in ("brief", "duet", "morning"):
        from whole_body_compute.brief import (
            build_sovereign_briefing,
            format_briefing_terminal,
            run_interactive_approval_gate,
            speak_briefing,
        )

        channel = getattr(args, "channel", "morning") or "morning"
        window = getattr(args, "window", "24h") or "24h"
        do_speak = getattr(args, "speak", False)
        voice = getattr(args, "voice", "Christopher") or "Christopher"
        speed = float(getattr(args, "speed", 1.35) or 1.35)
        do_approve = getattr(args, "approve", False)
        auto_approve = getattr(args, "auto_approve", False)
        is_json = getattr(args, "json", False)

        briefing = build_sovereign_briefing(
            channel=channel,
            window=window,
        )

        if do_speak:
            briefing["speech_playback"] = speak_briefing(
                briefing.get("spoken_script", ""),
                voice=voice,
                speed=speed,
            )

        if do_approve:
            run_interactive_approval_gate(
                briefing=briefing,
                auto_approve=auto_approve,
            )

        if is_json:
            print(json.dumps(briefing, indent=2))
        else:
            print(format_briefing_terminal(briefing))
        return 0

    if cmd in ("report", "metrics", "scorecard", "kpi"):
        if getattr(args, "by_model", False):
            return _handle_by_model_leaderboard(args)

        from pathlib import Path
        from whole_body_compute.metrics import (
            build_canonical_metrics_report,
            format_metrics_terminal_table,
            format_session_handoff_receipt,
            write_metrics_ledger,
        )

        window = getattr(args, "window", None) or "24h"
        trend = getattr(args, "trend", "7d") or "7d"
        conv_id = getattr(args, "conv_id", None)
        write_ledger = getattr(args, "write_ledger", True)
        output_path = getattr(args, "output_path", None)
        show_receipt = getattr(args, "receipt", False)
        is_json = getattr(args, "json", False)

        report = build_canonical_metrics_report(
            window=window,
            trend_window=trend,
            conv_id=conv_id,
        )

        if write_ledger:
            ledger_p = write_metrics_ledger(
                report=report,
                path=Path(output_path) if output_path else None,
            )
            report["ledger_path"] = str(ledger_p)

        if is_json:
            print(json.dumps(report, indent=2))
        elif show_receipt:
            print(format_session_handoff_receipt(report))
        else:
            print(format_metrics_terminal_table(report))
            if write_ledger and report.get("ledger_path"):
                print(f"📄 Synchronized Canonical 5-Pillar Metrics Ledger -> `{report['ledger_path']}`\n")
        return 0

    if cmd in ("audit-spend", "roi", "finops"):
        from pathlib import Path
        from whole_body_compute.finops_audit import (
            audit_historical_spend,
            format_finops_audit_receipt,
            format_finops_markdown_receipt,
        )

        days = getattr(args, "days", 30)
        price_per_m = getattr(args, "price_per_m", 5.0)
        seats = getattr(args, "seats", 10)
        top = getattr(args, "top", 5)
        export_path = getattr(args, "export_receipt", None)
        is_json = getattr(args, "json", False)

        audit = audit_historical_spend(
            days=days,
            price_per_m=price_per_m,
            seats=seats,
            top=top,
        )

        if export_path:
            out_p = Path(export_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(format_finops_markdown_receipt(audit), encoding="utf-8")
            audit["exported_receipt_path"] = str(out_p)

        if is_json:
            print(json.dumps(audit, indent=2))
        else:
            print(format_finops_audit_receipt(audit))
            if export_path:
                print(f"📄 Exported shareable Markdown FinOps ROI receipt to `{export_path}`\n")
        return 0

    if cmd in ("team-sync", "hive-sync", "share-exemplars"):
        from whole_body_compute.license import LicenseManager
        from whole_body_compute.pro_loader import ProUpgradeRequiredError

        gate = LicenseManager().check_feature_gate("team_sync")
        if not gate.get("allowed"):
            raise ProUpgradeRequiredError(gate.get("upgrade_nudge") or "Requires Whole Body Compute Team ($79/seat/mo).")

        from whole_body_compute.local.dot_memory import get_dot_store

        sync_path = getattr(args, "path", ".agents/team_exemplars.jsonl")
        is_json = getattr(args, "json", False)
        res = get_dot_store().sync_team_exemplars(sync_path=sync_path)
        if is_json:
            print(json.dumps(res, indent=2))
        else:
            print(
                f"🧠 [Team Hive-Mind Exemplar Sync] {res['sync_path']}\n"
                f"  • Imported Teammate Exemplars: +{res['imported_count']}\n"
                f"  • Exported Local Exemplars:    +{res['exported_count']}\n"
                f"  • Total Shared Team Ledger:    {res['total_team_exemplars']} verified patterns\n"
                f"  • Team Local Autonomy Rate:    {res['current_local_autonomy_pct']:.1f}% (+{res['team_autonomy_boost_pct']:.1f}% Hive-Mind Boost)"
            )
        return 0

    if cmd in ("license", "tier", "pricing", "upgrade", "plan"):
        from whole_body_compute.license import LicenseManager, format_license_and_pricing_card

        mgr = LicenseManager()
        set_t = getattr(args, "set_tier", None)
        act_key = getattr(args, "activate_key", None)
        polar_org = getattr(args, "polar_org_id", None)
        deact = getattr(args, "deactivate", False)
        offline = getattr(args, "offline", False)
        wheel_path = getattr(args, "wheel_path", None)
        wheel_url = getattr(args, "wheel_url", None)
        no_install_wheel = getattr(args, "no_install_wheel", False)
        reinstall_wheel = getattr(args, "reinstall_wheel", False)
        chk_f = getattr(args, "check_feature", None)
        daily_int = getattr(args, "daily_interceptions", 0)
        daily_off = getattr(args, "daily_silicon_offloads", 0)
        daily_ac = getattr(args, "daily_anti_cheat_blocks", 0)
        is_json = getattr(args, "json", False)

        try:
            if act_key:
                status = mgr.activate_polar_key(
                    act_key,
                    organization_id=polar_org,
                    offline=offline,
                    wheel_path=wheel_path,
                    wheel_url=wheel_url,
                    install_wheel=False if no_install_wheel else None,
                )
            elif reinstall_wheel:
                status = mgr.reinstall_pro_wheel(wheel_path=wheel_path, wheel_url=wheel_url)
            elif wheel_path:
                from whole_body_compute.pro_loader import install_pro_wheel

                install_res = install_pro_wheel(wheel_path)
                status = mgr.get_status()
                status["pro_plugin_install"] = install_res
            elif deact:
                status = mgr.deactivate_license()
            elif set_t:
                status = mgr.set_tier(set_t)
            else:
                status = mgr.get_status()
        except (ValueError, RuntimeError, OSError) as err:
            if is_json:
                print(json.dumps({"error": str(err)}, indent=2))
            else:
                print(f"Error: {err}")
            return 1

        if chk_f:
            gate = mgr.check_feature_gate(
                chk_f,
                daily_interceptions=daily_int,
                daily_silicon_offloads=daily_off,
                daily_anti_cheat_blocks=daily_ac,
            )
            status["feature_gate"] = gate

        if is_json:
            print(json.dumps(status, indent=2))
        else:
            print(format_license_and_pricing_card(status))
            if chk_f and status.get("feature_gate", {}).get("upgrade_nudge"):
                print(f"  {status['feature_gate']['upgrade_nudge']}\n")
        return 0

    if cmd == "airlock":
        from pathlib import Path
        from whole_body_compute.local.benchmark import (
            _extract_selective_bundle_text,
            format_airlock_3arm_benchmark_table,
            run_airlock_3arm_benchmark,
        )
        from whole_body_compute.local.symbols import LocalSymbolIndex
        from whole_body_compute.organs.immune import (
            _render_pyi_skeleton_fallback,
            classify_airlock_permeability,
        )

        action = getattr(args, "action", "benchmark") or "benchmark"
        target = getattr(args, "target", None)
        query = getattr(args, "query", "") or ""
        symbol = getattr(args, "symbol", "") or ""
        is_json = getattr(args, "json", False)

        if action == "skeleton":
            if not target:
                print("Error: target file path is required for `body airlock skeleton <target>`.", file=sys.stderr)
                return 1
            idx = LocalSymbolIndex()
            if hasattr(idx, "render_skeleton"):
                res = idx.render_skeleton(target)
            else:
                res = _render_pyi_skeleton_fallback(Path(target))
            if is_json:
                print(json.dumps({"action": "skeleton", "target": target, "skeleton": res}, indent=2))
            else:
                print(res if isinstance(res, str) else json.dumps(res, indent=2))
            return 0

        if action == "zoom":
            if not target or not symbol:
                print("Error: target and --symbol are required for `body airlock zoom <target> --symbol <sym>`.", file=sys.stderr)
                return 1
            idx = LocalSymbolIndex()
            if hasattr(idx, "zoom_symbol"):
                res = idx.zoom_symbol(target, symbol)
            else:
                import ast

                fpath = Path(target)
                raw = fpath.read_text(encoding="utf-8", errors="replace")
                lines = raw.splitlines()
                tree = ast.parse(raw, filename=str(fpath))
                matched_slice = f"Symbol '{symbol}' not found in {target}."
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol:
                        s_ln = max(1, node.lineno)
                        e_ln = min(len(lines), getattr(node, "end_lineno", s_ln + 50))
                        numbered = [f"{s_ln + i:4d}: {l}" for i, l in enumerate(lines[s_ln - 1 : e_ln])]
                        matched_slice = f"# Zoom Symbol `{symbol}` in {fpath.name} (L{s_ln}-L{e_ln})\n" + "\n".join(numbered)
                        break
                res = matched_slice
            if is_json:
                print(json.dumps({"action": "zoom", "target": target, "symbol": symbol, "result": res}, indent=2))
            else:
                print(res if isinstance(res, str) else json.dumps(res, indent=2))
            return 0

        if action == "bundle":
            if not target:
                print("Error: target is required for `body airlock bundle <target> -q '<query>'`.", file=sys.stderr)
                return 1
            idx = LocalSymbolIndex()
            if hasattr(idx, "build_selective_bundle"):
                res = idx.build_selective_bundle(target, query)
            else:
                res = _extract_selective_bundle_text(Path(target), query, [])
            if is_json or isinstance(res, dict):
                print(json.dumps(res if isinstance(res, dict) else {"action": "bundle", "target": target, "query": query, "bundle": res}, indent=2))
            else:
                print(res)
            return 0

        if action == "classify":
            q_str = query or (target or "")
            res = classify_airlock_permeability(
                task_query=q_str,
                file_count_est=getattr(args, "files", 1),
                retry_strikes=getattr(args, "strikes", 0),
            )
            if is_json:
                print(json.dumps(res, indent=2))
            else:
                print(f"🛡️  BBB Airlock Permeability Mode: {res['mode']}")
                print(f"  • Reason:              {res['reason']}")
                print(f"  • Allow Raw Read:      {res['allow_raw_read']}")
                print(f"  • Recommended Command: {res['recommended_command']}")
            return 0

        if action == "benchmark":
            report = run_airlock_3arm_benchmark()
            if is_json:
                print(json.dumps(report, indent=2))
            else:
                print(format_airlock_3arm_benchmark_table(report))
            return 0

        if action == "learn":
            if not target or not query:
                print(
                    'Error: target and -q "<lesson>" are required for `body airlock learn <target> [--symbol <sym>] -q "<lesson>"`.',
                    file=sys.stderr,
                )
                return 1
            from whole_body_compute.local.dot_memory import get_dot_store

            mastery = get_dot_store().record_airlock_outcome(
                target_file=target,
                outcome="escalated_to_cloud",
                symbol_name=symbol,
                lesson=query,
            )
            payload = {
                "action": "learn",
                "target": target,
                "symbol": symbol,
                "lesson": query,
                "mastery": mastery,
            }
            if is_json:
                print(json.dumps(payload, indent=2))
            else:
                sym_tag = f"::{symbol}" if symbol else ""
                print(f"🧠 Recorded Airlock Lesson for `{target}{sym_tag}`: {query}")
                print(
                    f"  • Mastery Score:     {mastery['mastery_score'] * 100:.1f}% "
                    f"({mastery['local_passes']}/{mastery['total_runs']} local passes, {mastery['escalations']} escalations)"
                )
                print(f"  • Recommended Valve: {mastery['recommended_valve']}")
            return 0

        if action == "mastery":
            from whole_body_compute.local.dot_memory import get_dot_store

            store = get_dot_store()
            if target:
                m_res = store.get_file_mastery(target)
                if is_json:
                    print(json.dumps(m_res, indent=2))
                else:
                    print(f"📊 Airlock File Mastery: {m_res['target_file']}")
                    print(f"  • Total Runs:        {m_res['total_runs']}")
                    print(f"  • Local Passes:      {m_res['local_passes']}")
                    print(f"  • Escalations:       {m_res['escalations']}")
                    print(f"  • Mastery Score:     {m_res['mastery_score'] * 100:.1f}%")
                    print(f"  • Recommended Valve: {m_res['recommended_valve']}")
            else:
                all_m = store.get_all_file_mastery()
                if is_json:
                    print(json.dumps({"files": all_m}, indent=2))
                else:
                    print(f"📊 Airlock File Mastery Ledger ({len(all_m)} tracked files)")
                    for m_res in all_m:
                        print(
                            f"  • {m_res['target_file']:<28} | "
                            f"score={m_res['mastery_score'] * 100:5.1f}% "
                            f"({m_res['local_passes']}/{m_res['total_runs']} local, {m_res['escalations']} esc) "
                            f"-> {m_res['recommended_valve']}"
                        )
            return 0

    if cmd in ("memory", "hippocampus"):
        remember_fact = getattr(args, "remember", None)
        recall_query = getattr(args, "recall", None)
        deja_vu = getattr(args, "deja_vu", None)
        rewind_query = getattr(args, "rewind", None)
        decay = getattr(args, "decay", False)
        decompose_transcript = getattr(args, "decompose", None)
        is_json = getattr(args, "json", False)

        if (
            remember_fact is not None
            or recall_query is not None
            or deja_vu is not None
            or rewind_query is not None
            or decay
            or decompose_transcript is not None
        ):
            from whole_body_compute.local.dot_memory import get_dot_store

            store = get_dot_store()

            if remember_fact is not None:
                mem_res = store.add_memory(fact=remember_fact)
                with store._connect() as conn:
                    mem_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                out_data = {
                    "status": "stored",
                    "fact": remember_fact,
                    "memory_id": mem_res.get("memory_id"),
                    "memory_count": mem_count,
                }
                if is_json:
                    print(json.dumps(out_data, indent=2))
                else:
                    print("🧠 [Whole Body Compute Hippocampus: Fact Stored]")
                    print(f"  • Fact:         {remember_fact}")
                    print(f"  • Memory Count: {mem_count}")
                    if mem_res.get("memory_id"):
                        print(f"  • Memory ID:    {mem_res['memory_id']}")
                return 0

            if recall_query is not None:
                matches = store.recall(recall_query, top_k=5)
                if not matches:
                    matches = store.search(recall_query, top_k=5, use_embeddings=False)
                with store._connect() as conn:
                    mem_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                out_data = {
                    "query": recall_query,
                    "memory_count": mem_count,
                    "matches": matches,
                    "count": len(matches),
                }
                if is_json:
                    print(json.dumps(out_data, indent=2))
                else:
                    print(f"🔍 [Whole Body Compute Hippocampus: Recall for '{recall_query}'] ({len(matches)} matches)")
                    for m in matches:
                        print(f"  • {m.get('fact') or m.get('content')}")
                return 0

            if deja_vu is not None:
                match = store.check_deja_vu(deja_vu)
                if is_json:
                    print(json.dumps(match or {"matched": False, "query": deja_vu}, indent=2))
                else:
                    if match:
                        print("🔮 [Déjà Vu Match Detected]")
                        print(f"  • Memory ID:   {match['memory_id']}")
                        print(f"  • Score:       {match['score']:.2f}")
                        print(f"  • Commit Ref:  {match.get('commit_ref') or 'None'}")
                        print(f"  • Content:     {match['content']}")
                    else:
                        print("⚪ No prior bug or stack trace match found in memory/timeline.")
                return 0

            if rewind_query is not None:
                timeline = store.rewind_timeline(query=rewind_query)
                if is_json:
                    print(json.dumps(timeline, indent=2))
                else:
                    print(
                        f"⏪ [Episodic Timeline Rewind for '{rewind_query}'] ({len(timeline)} events)"
                    )
                    for ev in timeline:
                        ts_str = time.strftime(
                            "%Y-%m-%d %H:%M:%S", time.localtime(ev["timestamp"])
                        )
                        app_s = f"[{ev['app_name']}]" if ev["app_name"] else ""
                        print(f"  • {ts_str} {app_s} {ev['summary']}")
                        if ev.get("git_commit"):
                            print(f"    (git: {ev.get('git_branch', '')}@{ev['git_commit']})")
                return 0

            if decay:
                decay_res = store.apply_salience_decay()
                if is_json:
                    print(json.dumps(decay_res, indent=2))
                else:
                    print("🧠 [Ebbinghaus Salience Decay Applied]")
                    print(f"  • Evaluated: {decay_res['evaluated']}")
                    print(f"  • Active:    {decay_res['active']}")
                    print(f"  • Archived:  {decay_res['archived']}")
                return 0

            if decompose_transcript is not None:
                from whole_body_compute.integrations.vault_agent import VaultAgent

                agent = VaultAgent()
                decomposed = agent.decompose_voice_memo(
                    decompose_transcript, memory_store=store
                )
                for note in decomposed.get("atomic_notes", []):
                    title = note.get("title", "Note")
                    content = note.get("content", "")
                    store.add_memory(fact=f"{title}: {content}", category="note", tags="decomposed")
                if is_json:
                    print(json.dumps(decomposed, indent=2))
                else:
                    print("🎙️  [Voice-to-Zettelkasten Decomposed]")
                    print(f"  • Atomic Notes:       {len(decomposed['atomic_notes'])}")
                    for n in decomposed["atomic_notes"]:
                        print(
                            f"    - Title: {n['title']} (links: {', '.join(n.get('wikilinks', []))})"
                        )
                    print(
                        f"  • Action Tickets:     {len(decomposed['engineering_tickets'])}"
                    )
                    for t in decomposed["engineering_tickets"]:
                        print(f"    - [ ] {t}")
                    print(f"  • Open Questions:     {len(decomposed['open_questions'])}")
                    for q in decomposed["open_questions"]:
                        print(f"    - ? {q}")
                return 0

    if cmd in ("eyes", "hands"):
        if getattr(args, "daemon", False):
            from whole_body_compute.integrations.foveal_vision import run_foveal_daemon

            enable_ambient = None if not getattr(args, "no_ambient", False) else False
            run_foveal_daemon(enable_ambient_discovery=enable_ambient)
            return 0

        raw_click = getattr(args, "click", None)
        raw_click_xy = getattr(args, "click_xy", None)
        # Auto-heal `body hands --click "1151,884"` -> click_xy="1151,884"
        if raw_click and not raw_click_xy and "," in raw_click:
            parts = [p.strip() for p in raw_click.split(",")]
            if len(parts) == 2:
                try:
                    float(parts[0])
                    float(parts[1])
                    raw_click_xy = f"{parts[0]},{parts[1]}"
                    raw_click = None
                except ValueError:
                    pass

        has_target = bool(
            getattr(args, "app", None)
            or getattr(args, "pid", None)
            or getattr(args, "title", None)
            or getattr(args, "filter", None)
            or getattr(args, "max_elements", None)
            or raw_click
            or raw_click_xy
            or getattr(args, "hover", None)
            or getattr(args, "right_click", None)
            or getattr(args, "attach", None)
            or getattr(args, "chain", None)
            or getattr(args, "floating", False)
            or getattr(args, "status_bar", False)
            or getattr(args, "type_text", None)
            or getattr(args, "paste_text", None)
            or getattr(args, "scroll", None)
            or getattr(args, "hotkey", None)
            or getattr(args, "menu", None)
            or getattr(args, "submit", False)
            or getattr(args, "read_last", False)
            or getattr(args, "wait_reply", None) is not None
        )
        if has_target:
            from whole_body_compute.integrations.foveal_vision import (
                FovealVisionEngine,
                send_foveal_socket_request,
            )

            organ_num = "04" if cmd == "hands" else "03"
            organ_name = "The Hands" if cmd == "hands" else "The Eyes"
            organ_sub = "Goose, Tactile & Foveal Actuation" if cmd == "hands" else "Foveal DOM, Tactile & Vagus"

            req_payload = {
                "pid": getattr(args, "pid", None),
                "app_name": getattr(args, "app", None),
                "window_title_query": getattr(args, "title", None),
                "max_elements": getattr(args, "max_elements", None),
                "filter_query": getattr(args, "filter", None),
                "click": raw_click,
                "click_xy": raw_click_xy,
                "hover": getattr(args, "hover", None),
                "right_click": getattr(args, "right_click", None),
                "attach": getattr(args, "attach", None),
                "chain": getattr(args, "chain", None),
                "include_floating": getattr(args, "floating", False),
                "include_status_bar": getattr(args, "status_bar", False),
                "type_text": getattr(args, "type_text", None),
                "paste_text": getattr(args, "paste_text", None),
                "scroll": getattr(args, "scroll", None),
                "hotkey": getattr(args, "hotkey", None),
                "menu": getattr(args, "menu", None),
                "submit": getattr(args, "submit", False),
                "wait_reply": getattr(args, "wait_reply", None),
                "read_last": getattr(args, "read_last", False),
            }

            socket_resp = send_foveal_socket_request(req_payload)
            if socket_resp is not None:
                if getattr(args, "json", False):
                    socket_resp["organ_num"] = organ_num
                    socket_resp["name"] = organ_name
                    socket_resp["subsystem"] = organ_sub
                    socket_resp["status"] = "pass"
                    print(json.dumps(socket_resp, indent=2))
                else:
                    print(f"[Foveal Vision (Daemon)] App: {socket_resp.get('app_name')} | Title: {socket_resp.get('window_title')}")
                    if socket_resp.get("last_messages"):
                        print("Latest Messages:")
                        for msg in socket_resp.get("last_messages")[-5:]:
                            print(f'  - "{msg}"')
                    elif socket_resp.get("last_message"):
                        print(f"Last Message: \"{socket_resp.get('last_message')}\"")
                return 0

            snap = FovealVisionEngine.capture_snapshot(
                pid=getattr(args, "pid", None),
                app_name=getattr(args, "app", None),
                window_title_query=getattr(args, "title", None),
                max_elements=getattr(args, "max_elements", None),
                filter_query=getattr(args, "filter", None),
                click=raw_click,
                click_xy=raw_click_xy,
                hover=getattr(args, "hover", None),
                right_click=getattr(args, "right_click", None),
                attach=getattr(args, "attach", None),
                chain=getattr(args, "chain", None),
                include_floating=getattr(args, "floating", False),
                include_status_bar=getattr(args, "status_bar", False),
                type_text=getattr(args, "type_text", None),
                paste_text=getattr(args, "paste_text", None),
                scroll=getattr(args, "scroll", None),
                hotkey=getattr(args, "hotkey", None),
                menu=getattr(args, "menu", None),
                submit=getattr(args, "submit", False),
                wait_reply=getattr(args, "wait_reply", None),
                read_last=getattr(args, "read_last", False),
            )
            if getattr(args, "json", False):
                res = snap.to_dict()
                res["organ_num"] = organ_num
                res["name"] = organ_name
                res["subsystem"] = organ_sub
                res["status"] = "pass"
                print(json.dumps(res, indent=2))
                return 0
            max_show = getattr(args, "max_elements", None) or 40
            print(snap.to_context_block(max_elements=max_show))
            return 0

    if cmd in SINGLE_ORGAN_DISPATCH:
        if cmd == "sleep":
            _require_pro_cli_feature("rem_sleep_lora")
            from whole_body_compute.pro_loader import (
                UPGRADE_MESSAGE,
                ProUpgradeRequiredError,
                is_open_core_stubbed,
                load_pro_module,
            )

            if is_open_core_stubbed() and load_pro_module("rem_sleep") is None:
                raise ProUpgradeRequiredError(UPGRADE_MESSAGE)
        organ_res = SINGLE_ORGAN_DISPATCH[cmd]()
        if getattr(args, "json", False):
            if cmd == "immune":
                from whole_body_compute.organs.immune import get_immune_pre_post_scorecard

                organ_res["pre_post_scorecard"] = get_immune_pre_post_scorecard()
            print(json.dumps(organ_res, indent=2))
        else:
            fake_report = {
                "timestamp": 0,
                "elapsed_ms": organ_res.get("latency_ms", 0.0),
                "overall_status": organ_res.get("status", "pass"),
                "total_organs": 1,
                "organs": [organ_res],
            }
            print(format_organ_doctor_table(fake_report))
            if cmd == "immune":
                from whole_body_compute.organs.immune import format_immune_scorecard_table

                print(format_immune_scorecard_table())
                print()
            elif cmd == "heart":
                vitals = organ_res.get("vital_signs", {})
                large_procs = vitals.get("large_processes", [])
                tachy_procs = vitals.get("tachycardic_processes", [])
                print()
                print("📡 [High-RAM / High-CPU Process Radar (>= 500 MB or >= 20% CPU)]")
                print("─" * 84)
                if large_procs or tachy_procs:
                    for p in tachy_procs:
                        print(f"  • ⚠️  [RUNAWAY TACHYCARDIA] [{p['pid']}] {p['name']} - {p.get('rss_mb', 0):.1f} MB RSS ({p.get('rss_gb', 0):.2f} GB), {p.get('cpu_pct', 0):.1f}% CPU (up {p.get('elapsed', '0s')})")
                    for p in large_procs:
                        if any(tp.get("pid") == p.get("pid") for tp in tachy_procs):
                            continue
                        tag = "⚠️ HIGH-LOAD" if p.get("cpu_pct", 0) >= 20.0 or p.get("rss_mb", 0) >= 1500.0 else "ℹ️ HEAVY"
                        print(f"  • {tag} [{p['pid']}] {p['name']} - {p.get('rss_mb', 0):.1f} MB RSS ({p.get('rss_gb', 0):.2f} GB), {p.get('cpu_pct', 0):.1f}% CPU (up {p.get('elapsed', '0s')})")
                else:
                    print("  • Status: Clean - 0 runaway or oversized (>= 500 MB / >= 20% CPU) processes detected.")
                print("─" * 84)
                print()
            elif cmd == "eyes":
                print("  💡 Tip: Pass `--app <App>` or `--max-elements 20` to capture a live Foveal AXUIElement DOM snapshot.")
                print()
        return 0 if organ_res.get("status") != "fail" else 1

    return None
