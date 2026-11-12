"""
Whole Body Compute CLI (`body` / `wbc`)
Unified entrypoint for the 5-Organ Sovereign Agent Runtime.
"""

import argparse
import json
import sys
import time
from typing import List, Optional

from whole_body_compute import __version__
from whole_body_compute.cli_handlers import (
    cmd_checkpoint,
    cmd_commit,
    cmd_immune,
    dispatch_organ_and_cognitive_command,
    render_pro_upgrade_card,
)
from whole_body_compute.cli_parser import (
    SINGLE_ORGAN_DISPATCH,
    build_parser,
    setup_parser,
)
from whole_body_compute.doctor import (
    check_10_organs,
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
    exercise_10_organ_reflex_loop,
    format_organ_doctor_table,
    format_reflex_exercise_table,
)


def main(argv: Optional[List[str]] = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    if "--json" in raw_argv:
        args.json = True
    cmd = args.command or "doctor"
    try:
        return _dispatch_main(cmd, args, parser)
    except NotImplementedError as exc:
        return render_pro_upgrade_card(
            cmd=cmd,
            exc=exc,
            is_json=bool(getattr(args, "json", False)),
        )


def _dispatch_main(
    cmd: str,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    if cmd != "immune-hook":
        try:
            from whole_body_compute.procedural_telemetry import track_wbc_event

            track_wbc_event("wbc_cli_command", {"command": cmd})
        except Exception:
            pass

    if cmd in ("stats", "usage", "analytics"):
        from whole_body_compute.procedural_telemetry import (
            format_usage_stats_table,
            get_comprehensive_usage_stats_report,
        )

        report = get_comprehensive_usage_stats_report()
        if getattr(args, "json", False):
            print(json.dumps(report, indent=2))
        else:
            print("\n" + format_usage_stats_table(report) + "\n")
        return 0

    if cmd in ("begin", "init", "awaken", "start"):
        from whole_body_compute.begin import format_body_begin_banner, run_body_begin

        target_dir = getattr(args, "workspace_opt", None) or getattr(args, "target", None)
        res = run_body_begin(
            target_dir=target_dir,
            dry_run=getattr(args, "dry_run", False),
            skip_audit=getattr(args, "skip_audit", False),
            force_rules=getattr(args, "force_rules", False),
            days=getattr(args, "days", 30),
            price_per_m=getattr(args, "price_per_m", 5.0),
            seats=getattr(args, "seats", 10),
            telemetry_tier=getattr(args, "telemetry_tier", None),
            no_daemon=getattr(args, "no_daemon", False),
        )
        if getattr(args, "json", False):
            print(json.dumps(res, indent=2))
        else:
            print(format_body_begin_banner(res))
        return 0

    if cmd == "scout":
        from whole_body_compute.local.scout import main as scout_main

        if getattr(args, "slices", None):
            scout_argv = ["--slice"] + args.slices
        elif args.target:
            scout_argv = [args.target, "-q", args.query, "-m", args.model]
        else:
            scout_argv = ["--help"]

        if getattr(args, "json", False):
            scout_argv.append("--json")
        return scout_main(scout_argv)

    if cmd == "implement":
        from whole_body_compute.local.implementer import main as implement_main

        impl_argv = [args.target, "-i", args.instruction, "-m", args.model_coder]
        if getattr(args, "no_apply", False):
            impl_argv.append("--no-apply")
        if getattr(args, "json", False):
            impl_argv.append("--json")
        return implement_main(impl_argv)

    if cmd == "loop":
        from whole_body_compute.local.agent_loop import main as loop_main

        loop_argv = [args.goal, "-m", args.model, "-t", str(args.max_turns)]
        if getattr(args, "json", False):
            loop_argv.append("--json")
        return loop_main(loop_argv)

    if cmd == "arena":
        from whole_body_compute.local.arena import main as arena_main

        arena_argv = ["--arena-dir", str(getattr(args, "arena_dir", "/tmp/wbc_arena"))]
        if getattr(args, "setup_only", False):
            arena_argv.append("--setup-only")
        if getattr(args, "no_claude_parity", False):
            arena_argv.append("--no-claude-parity")
        if getattr(args, "json", False):
            arena_argv.append("--json")
        return arena_main(arena_argv)

    if cmd == "commit":
        return cmd_commit(args)

    if cmd == "checkpoint":
        return cmd_checkpoint(args)

    if cmd in ("import-screenpipe", "import_screenpipe"):
        from pathlib import Path

        from whole_body_compute.integrations.foveal_memory import (
            import_legacy_screenpipe_db,
        )

        res = import_legacy_screenpipe_db(
            source_db_path=Path(args.source) if args.source else None,
            target_db_path=Path(args.target) if args.target else None,
            redact_pii=not getattr(args, "no_redact", False),
        )
        if getattr(args, "json", False):
            print(json.dumps(res, indent=2))
        else:
            if res.get("status") == "not_found":
                print(f"[FovealMemory] Error: {res.get('message', 'Source DB not found')}")
                return 1
            print(
                f"[FovealMemory] Migrated {res.get('total_imported', 0)} records "
                f"({res.get('ocr_imported', 0)} OCR, {res.get('audio_imported', 0)} Audio) "
                f"from {res.get('source_db')} -> {res.get('target_db')}"
            )
            if res.get("redacted_count", 0) > 0:
                print(f"[FovealMemory] Sanitized {res['redacted_count']} sensitive/PII entries during import.")
        return 0 if res.get("status") in ("migrated", "skipped") else 1

    if cmd in ("skills", "skill"):
        from whole_body_compute.skills_telemetry import (
            audit_skills,
            format_skills_audit_table,
            list_archived_skills,
            prune_skills,
            restore_skill,
        )

        action = getattr(args, "action", "audit") or "audit"
        targets = getattr(args, "targets", None) or []
        is_json = bool(getattr(args, "json", False))

        if action == "export":
            from whole_body_compute.integrations.pal_skill_exporter import (
                export_recipe_as_skill_bundle,
            )

            if not targets:
                if is_json:
                    print(json.dumps({"error": "Specify a recipe YAML path or name to export"}, indent=2))
                else:
                    print("Usage: body skills export <recipe_yaml_or_name> [--output-dir .agents/skills] [--dry-run]")
                return 1

            try:
                res = export_recipe_as_skill_bundle(
                    recipe_or_path=targets[0],
                    output_root=getattr(args, "output_dir", None),
                    zip_bundle=not bool(getattr(args, "no_zip", False)),
                    dry_run=bool(getattr(args, "dry_run", False)),
                )
            except Exception as exc:
                if is_json:
                    print(json.dumps({"exported": False, "error": str(exc)}, indent=2))
                else:
                    print(f"❌ Skill export failed: {exc}")
                return 1

            if is_json:
                print(json.dumps(res, indent=2))
            elif res.get("exported"):
                port = res["portability"]
                print(
                    f"✅ Exported portable skill `{res['skill_slug']}` "
                    f"({port['cloud_portable_steps']}/{port['total_steps']} cloud-portable steps) -> `{res['skill_dir']}`"
                )
                if res.get("zip_bundle_path"):
                    print(f"📦 Claude Duet / Cloud Bot archive: `{res['zip_bundle_path']}`")
            else:
                port = res["portability"]
                print(
                    f"🔍 [Dry-Run Skill Preview: `{res['skill_slug']}` — "
                    f"{port['cloud_portable_steps']}/{port['total_steps']} cloud-portable steps, "
                    f"~${port['estimated_monthly_usd_saved']:.2f}/mo saved]"
                )
                if res.get("upgrade_nudge"):
                    print(f"  {res['upgrade_nudge']}")
            return 0

        if action == "prune":
            res = prune_skills(
                skill_names=targets or None,
                max_conversations=int(getattr(args, "max_convs", 0)),
                include_protected=bool(getattr(args, "include_protected", False)),
                dry_run=bool(getattr(args, "dry_run", False)),
            )
            if is_json:
                print(json.dumps(res, indent=2))
            else:
                mode = "[DRY-RUN] Would shelve" if res["dry_run"] else "📦 Shelved"
                print(
                    f"\n{mode} {res['shelved_count']} dormant skills -> `{res['archive_dir']}` "
                    f"(reclaiming ~{res['tokens_saved_per_turn']:,} prompt tokens/turn, "
                    f"~{res['estimated_100_turn_token_savings']:,} tokens per 100 turns):"
                )
                for item in res["shelved"]:
                    print(
                        f"  • {item['name']:<26} [{item['source_label']}] "
                        f"(~{item['prompt_tokens_per_turn']} tok/turn)"
                    )
                if res["skipped_protected"]:
                    prot_names = ", ".join(x["name"] for x in res["skipped_protected"])
                    print(f"  🔒 Protected core skills kept active: {prot_names}")
                print("  💡 Restore any skill anytime with: `body skills restore <name>` (or `all`)\n")
            return 0

        if action == "restore":
            if not targets:
                if is_json:
                    print(json.dumps({"error": "Specify skill name(s) or 'all' to restore"}, indent=2))
                else:
                    print("Usage: body skills restore <skill_name|all>")
                return 1
            results = [restore_skill(t) for t in targets]
            if is_json:
                print(json.dumps(results if len(results) > 1 else results[0], indent=2))
            else:
                for r in results:
                    if r.get("restored"):
                        if "items" in r:
                            print(f"✅ Restored {r['restored_count']} archived skills:")
                            for sub in r["items"]:
                                print(f"  • {sub['skill_name']} -> `{sub['restored_to']}`")
                        else:
                            print(f"✅ Restored `{r['skill_name']}` -> `{r['restored_to']}`")
                    else:
                        print(f"❌ {r.get('error', 'Failed to restore skill')}")
            return 0 if all(r.get("restored") for r in results) else 1

        if action == "archive":
            archived = list_archived_skills()
            if is_json:
                print(json.dumps({"archived_skills": archived}, indent=2))
            else:
                print(f"\n📦 [Cold-Storage Skills Archive (`~/.gemini/config/skills_archive/`) — {len(archived)} skills]")
                for a in archived:
                    print(
                        f"  • {a['name']:<26} (from {a['source_label']}, saves ~{a['prompt_tokens_saved_per_turn']} tok/turn) "
                        f"-> `{a['original_skill_dir']}`"
                    )
                print()
            return 0

        report = audit_skills()
        if is_json:
            print(json.dumps(report, indent=2))
        else:
            print(format_skills_audit_table(report, show_all=bool(getattr(args, "show_all", False))))
        return 0

    if cmd == "shield":
        from whole_body_compute.shield import (
            SUPPORTED_BOTS,
            ShieldInstaller,
            format_shield_status,
        )

        installer = ShieldInstaller(workspace_dir=getattr(args, "workspace", None))
        scope = getattr(args, "scope", "workspace") or "workspace"
        bots_arg = getattr(args, "bots", None)
        all_bots = getattr(args, "all_bots", False)
        do_install = getattr(args, "install", False)
        dry_run = getattr(args, "dry_run", False)

        if all_bots:
            target_bots = list(SUPPORTED_BOTS.keys())
        elif bots_arg:
            target_bots = bots_arg
        else:
            target_bots = list(SUPPORTED_BOTS.keys()) if not do_install else ["antigravity", "claude-code"]

        if do_install:
            report = installer.install(bots=target_bots, scope=scope, dry_run=dry_run)
        else:
            report = installer.status(bots=target_bots, scope=scope)

        if getattr(args, "json", False):
            print(json.dumps(report, indent=2))
        else:
            print(format_shield_status(report))
        return 0  # [direct-edit]

    if cmd in ("savings", "leaderboard", "rank"):
        from whole_body_compute.shield import (
            ShieldTelemetryLedger,
            format_savings_leaderboard,
            format_savings_receipt,
            get_savings_summary,
        )

        if getattr(args, "reset", False):
            ledger = ShieldTelemetryLedger()
            deleted = ledger.reset()
            if getattr(args, "json", False):
                print(json.dumps({"reset": True, "deleted_rows": deleted}, indent=2))
            else:
                print(f"🧹 Reset Shield Telemetry Ledger ({deleted} records cleared).")
            return 0

        eff_win = getattr(args, "window", None)
        if not eff_win:
            if getattr(args, "today", False):
                eff_win = "today"
            elif getattr(args, "last_24h", False):
                eff_win = "24h"
            elif getattr(args, "last_7d", False):
                eff_win = "7d"

        top_n = int(getattr(args, "top", 15) or 15)
        summary = get_savings_summary(
            bot_filter=getattr(args, "bot", None),
            price_per_m=getattr(args, "price_per_m", None),
            window=eff_win,
            top_limit=top_n,
        )  # [direct-edit]
        if getattr(args, "json", False):
            print(json.dumps(summary, indent=2))
        elif cmd in ("leaderboard", "rank") or getattr(args, "leaderboard", False):
            print(format_savings_leaderboard(summary, top_n=top_n))
        else:
            print(format_savings_receipt(summary))
        return 0

    if cmd in ("doctor", "status", "organs"):
        try:
            from whole_body_compute.local.scout import prewarm_scout

            prewarm_scout(background=True)
        except Exception:
            pass
        report = check_10_organs()
        is_json = bool(getattr(args, "json", False))
        if not is_json:
            print(format_organ_doctor_table(report))

        if getattr(args, "exercise", False):
            reflex = exercise_10_organ_reflex_loop()
            if is_json:
                report["exercise"] = reflex
            else:
                print()
                print(format_reflex_exercise_table(reflex))

        if getattr(args, "chaos", False):
            from whole_body_compute.chaos import ChaosOrganFuzzer

            fuzzer = ChaosOrganFuzzer(verbose=not is_json)
            chaos_res = fuzzer.run_all(quiet=is_json)
            if is_json:
                report["chaos"] = chaos_res

        if is_json:
            print(json.dumps(report, indent=2))
        return 0 if report.get("overall_status") != "fail" else 1

    if cmd in ("exercise", "reflex"):
        reflex = exercise_10_organ_reflex_loop()
        if getattr(args, "json", False):
            print(json.dumps(reflex, indent=2))
        else:
            print(format_reflex_exercise_table(reflex))
        return 0 if reflex.get("all_passed") else 1

    if cmd in ("chaos", "gauntlet"):
        from whole_body_compute.chaos import ChaosOrganFuzzer

        is_json = bool(getattr(args, "json", False))
        fuzzer = ChaosOrganFuzzer(verbose=not is_json)
        chaos_res = fuzzer.run_all(quiet=is_json)
        if is_json:
            print(json.dumps(chaos_res, indent=2))
        return 0 if chaos_res.get("all_passed") else 1

    if cmd in ("rem-sleep", "remsleep", "consolidate", "nightly"):
        from whole_body_compute.cli_handlers import _require_pro_cli_feature

        _require_pro_cli_feature("rem_sleep_lora")
        from whole_body_compute.local.rem_sleep import run_rem_sleep_cycle

        dry_run = getattr(args, "dry_run", False)
        force_lora = getattr(args, "force_lora", False)
        as_json = getattr(args, "json", False)
        if not as_json and getattr(run_rem_sleep_cycle, "__name__", "") != "ProFeatureStub":
            print("🌙 Launching Whole Body Compute Nocturnal REM Sleep Consolidation Cycle...")
        scorecard = run_rem_sleep_cycle(dry_run=dry_run, force_lora=force_lora)
        if as_json:
            print(json.dumps(scorecard.to_dict(), indent=2))
        else:
            print("\n" + scorecard.to_summary() + "\n")
        return 0 if scorecard.promotion_status == "PROMOTED" or dry_run else 1

    if cmd in ("candidates", "candidate", "recipes"):
        from whole_body_compute.integrations.ambient_discovery import (
            discover_now_from_ambient_sources,
            format_candidates_table,
            list_staged_candidates,
            promote_staged_candidate,
        )

        if getattr(args, "discover", False):
            disc = discover_now_from_ambient_sources()
            if not getattr(args, "json", False) and disc.get("discovered_count"):
                print(
                    f"🔍 Discovered & staged {disc['discovered_count']} ambient candidate(s): "
                    f"{', '.join(disc['staged_ids'])}"
                )

        promote_id = getattr(args, "promote", None)
        share_learnings = bool(getattr(args, "share", False))
        as_skill = bool(getattr(args, "as_skill", False))
        if promote_id:
            res = promote_staged_candidate(promote_id, share_learnings=share_learnings)
            if res.get("promoted") and as_skill and res.get("promoted_path"):
                from whole_body_compute.integrations.pal_skill_exporter import (
                    export_recipe_as_skill_bundle,
                )

                res["skill_bundle"] = export_recipe_as_skill_bundle(res["promoted_path"])
            if getattr(args, "json", False):
                print(json.dumps(res, indent=2))
            elif res.get("promoted"):
                print(
                    f"✅ Promoted candidate `{res['id']}` ({res['steps_count']} steps, "
                    f"{res['postconditions_count']} postconditions) -> `{res['promoted_path']}`"
                )
                if res.get("skill_bundle", {}).get("exported"):
                    sb = res["skill_bundle"]
                    print(f"🎓 Exported portable Skill Bundle -> `{sb['skill_dir']}` (`{sb.get('zip_bundle_path')}`)")
                elif res.get("skill_bundle", {}).get("upgrade_nudge"):
                    print(f"  {res['skill_bundle']['upgrade_nudge']}")
                if res.get("learning_card"):
                    lc = res["learning_card"]
                    print(
                        f"🧬 Exported sanitized Procedural DNA card (IR hash: `{lc.get('ir_signature_hash')}`)"
                    )
            else:
                print(f"❌ Promotion failed: {res.get('reason')}")
            return 0 if res.get("promoted") else 1

        report = list_staged_candidates()
        if getattr(args, "json", False):
            print(json.dumps(report, indent=2))
        else:
            print("\n" + format_candidates_table(report) + "\n")
        return 0

    if cmd in ("pull", "search", "publish", "hub"):
        from whole_body_compute.hub import run_hub_cli_command

        return run_hub_cli_command(cmd, args)

    if cmd in ("telemetry", "dna", "learnings"):  # [direct-edit]
        from whole_body_compute.procedural_telemetry import (
            ProceduralDNAHarvester,
            TelemetryConsentManager,
            compact_telemetry_storage,
            format_harvest_summary,
            format_telemetry_status,
            generalize_workflow_recipe,
            get_telemetry_status_report,
        )

        action = getattr(args, "action", "status") or "status"
        is_json = bool(getattr(args, "json", False))

        if action == "status":
            report = get_telemetry_status_report()
            if is_json:
                print(json.dumps(report, indent=2))
            else:
                print("\n" + format_telemetry_status(report) + "\n")
            return 0

        if action == "stats":
            from whole_body_compute.procedural_telemetry import (
                format_usage_stats_table,
                get_comprehensive_usage_stats_report,
            )

            report = get_comprehensive_usage_stats_report()
            if is_json:
                print(json.dumps(report, indent=2))
            else:
                print("\n" + format_usage_stats_table(report) + "\n")
            return 0

        if action == "harvest":
            harvester = ProceduralDNAHarvester()
            report = harvester.harvest_all(persist=True)
            if is_json:
                print(json.dumps(report, indent=2))
            else:
                print("\n" + format_harvest_summary(report) + "\n")
            return 0

        if action in ("sync", "flush"):
            from whole_body_compute.procedural_telemetry import flush_telemetry_outbox

            res = flush_telemetry_outbox()
            if is_json:
                print(json.dumps(res, indent=2))
            else:
                st = res.get("status", "unknown")
                if st == "ok":
                    print(
                        f"\n🚀 Procedural DNA Outbox Synchronized:\n"
                        f"   Sent: {res.get('sent_count', 0)} | "
                        f"Already Synced: {res.get('already_synced_count', 0)} | "
                        f"Pending: {res.get('pending_count', 0)}\n"
                        f"   Outbox: {res.get('outbox_path')}\n"
                    )
                elif st == "blocked_by_consent":
                    print(
                        f"\n🔒 Telemetry Sync Blocked by Consent Gate:\n"
                        f"   {res.get('reason', 'Opt-in required')}\n"
                        f"   Current tier: {res.get('effective_tier', 'TIER_0_LOCAL')}. "
                        f"Run `body telemetry opt-in --tier 2` to enable Procedural DNA sharing.\n"
                    )
                else:
                    print(f"\n⚠️ Telemetry Sync Result: {st} ({res.get('reason', '')})\n")
            return 0 if res.get("status") in ("ok", "blocked_by_consent") else 1

        if action == "inspect":
            harvester = ProceduralDNAHarvester()
            cards = harvester.load_outbox_cards()
            payload = {
                "status": "ok",
                "outbox_path": str(harvester.outbox_path),
                "total_cards": len(cards),
                "cards": cards,
            }
            print(json.dumps(payload, indent=2))
            return 0

        if action == "generalize":
            target = getattr(args, "target", None) or "ambient_client_package_upload"
            try:
                tpl = generalize_workflow_recipe(target)
                harvester = ProceduralDNAHarvester()
                if harvester.consent_manager.can_share_procedural_dna() or getattr(args, "share", False) or getattr(args, "save", False):
                    tpl["cohort_id"] = harvester.consent_manager.get_cohort_id()
                    harvester.upsert_outbox_cards([tpl])
            except Exception as exc:
                err = {"status": "error", "reason": str(exc), "target": target}
                if is_json:
                    print(json.dumps(err, indent=2))
                else:
                    print(f"❌ Failed to generalize recipe `{target}`: {exc}")
                return 1

            if is_json:
                print(json.dumps(tpl, indent=2))
            else:
                print(
                    f"\n🧬 Generalized Recipe Skeleton: `{tpl['recipe_id']}` "
                    f"(IR Hash: `{tpl['ir_signature_hash']}` | Template Hash: `{tpl['signature_hash']}`)"
                )
                print("=" * 88)
                print(json.dumps(tpl, indent=2))
                print("=" * 88 + "\n")
            return 0

        if action == "compact":
            max_mb = float(getattr(args, "max_mb", 15.0) or 15.0)
            retention_days = int(getattr(args, "retention_days", 7) or 7)
            comp = compact_telemetry_storage(max_mb=max_mb, retention_days=retention_days)
            if is_json:
                print(json.dumps(comp, indent=2))
            else:
                print(
                    f"\n🧹 Telemetry Storage Compaction: {comp['before_mb']:.3f} MB -> {comp['after_mb']:.3f} MB "
                    f"(reclaimed {comp['reclaimed_mb']:.3f} MB; budget <= {comp['max_mb_budget']:.1f} MB: "
                    f"{'OK' if comp['within_budget'] else 'OVER'})"
                )
                print(
                    f"   Outbox entries: {comp['outbox_entries_before']} -> {comp['outbox_entries_after']} | "
                    f"Pruned DB rows: {comp['pruned_db_rows']}\n"
                )
            return 0

        if action in ("opt-in", "opt-out"):
            mgr = TelemetryConsentManager()
            tier_arg = getattr(args, "tier", None)
            if action == "opt-out":
                status = mgr.opt_out()
            else:
                status = mgr.opt_in(tier=tier_arg if tier_arg is not None else 2)
            if is_json:
                print(json.dumps(status, indent=2))
            else:
                print(
                    f"✅ Telemetry consent updated -> Effective: `{status['effective_tier']}` "
                    f"(Stored: `{status['stored_tier']}`, Cohort: `{status['cohort_id']}`)"
                )
            return 0

    if cmd in ("radix-cache", "radix", "prewarm", "kv-cache"):
        from whole_body_compute.local.radix_cache import get_radix_cache_manager

        mgr = get_radix_cache_manager()
        if getattr(args, "status", False):
            stats = mgr.get_cache_stats()
            if stats:
                print("\n" + stats.to_summary() + "\n")
            else:
                print("⚪ No active Radix KV-Cache metadata found. Run `body radix-cache` to pre-warm.")
            return 0
        target_url = getattr(args, "url", None) or "http://127.0.0.1:9379/v1/chat/completions"
        print(f"🩸 Pre-warming circulatory Radix KV-Cache on `{target_url}`...")
        stats = mgr.prewarm_cache(target_url=target_url)
        print("\n" + stats.to_summary() + "\n")
        return 0 if stats.status == "pinned" else 1

    if cmd in ("perfusion", "glymphatic", "ram-hygiene"):
        from whole_body_compute.cli_handlers import _require_pro_cli_feature

        _require_pro_cli_feature("cephalic_prewarm")
        from whole_body_compute.local.supervisor import default_supervisor

        prewarm_model = getattr(args, "prewarm", None)
        if prewarm_model:
            ttl = getattr(args, "ttl", "auto") or "auto"
            res = default_supervisor.trigger_cephalic_prewarm(
                model_name=prewarm_model, keep_alive=ttl, background=False
            )
            if getattr(args, "json", False):
                print(json.dumps(res, indent=2))
            else:
                print(
                    f"🧠 [Cephalic Phase Pre-Warm] {res.get('status').upper()} • Model: {res.get('model')} "
                    f"• TTL: {res.get('keep_alive_ttl')} ({res.get('duration_ms', 0)}ms)"
                )
            return 0 if res.get("status") != "blocked" else 1

        show_cold_starts = getattr(args, "cold_starts", False)
        do_evict = getattr(args, "evict", False)
        recycle_tachy = getattr(args, "recycle_tachycardic", False)

        if show_cold_starts and not do_evict and not recycle_tachy:
            cs_report = default_supervisor.get_cold_start_report()
            if getattr(args, "json", False):
                print(json.dumps(cs_report, indent=2))
            else:
                print("\n📊 [Whole Body Compute: Model Cold-Start Telemetry & Full Adaptive Keep-Warm Policy]")
                print("=" * 84)
                print(
                    f"  • Tracked Models: {cs_report.get('total_models', 0)} | "
                    f"Cold Starts: {cs_report.get('total_cold_starts', 0)} | "
                    f"TTL Misses: {cs_report.get('total_ttl_misses', 0)}"
                )
                print(f"  • Total Cold-Start Latency Penalty: {cs_report.get('total_cold_penalty_ms', 0.0):.1f} ms")
                pinned = cs_report.get("pinned_models", [])
                print(f"  • Pinned Models (Warm Lock): {', '.join(pinned) if pinned else 'none'}")
                print("-" * 84)
                policies = cs_report.get("policies", [])
                if not policies:
                    print("  • No model invocations recorded in the current window.")
                for pol in policies:
                    active_tag = "ACTIVE NOW" if pol.get("in_warm_window") else "inactive"
                    pin_tag = "PINNED" if pol.get("should_pin_warm") else "standard"
                    print(
                        f"  • Model: {pol['model_name']} ({pol.get('runtime', 'ollama')}) [{pin_tag}]"
                    )
                    print(
                        f"    - Cold-Start Rate: {pol.get('cold_start_rate', 0.0)*100:.1f}% "
                        f"({pol.get('cold_starts', 0)}/{pol.get('total_invocations', 0)}) | "
                        f"TTL Misses: {pol.get('ttl_misses', 0)}"
                    )
                    print(
                        f"    - Load Latency: Avg Cold {pol.get('avg_cold_load_ms', 0.0):.1f} ms | "
                        f"Max Cold {pol.get('max_cold_load_ms', 0.0):.1f} ms | "
                        f"Avg Warm {pol.get('avg_warm_load_ms', 0.0):.1f} ms"
                    )
                    print(f"    - Total Cold Penalty: {pol.get('total_cold_penalty_ms', 0.0):.1f} ms")
                    warm_hrs = pol.get("active_warm_hours", [])
                    print(
                        f"    - Circadian Warm Hours: {warm_hrs if warm_hrs else 'none'} ({active_tag})"
                    )
                    print(
                        f"    - Recommended TTL: {pol.get('recommended_keep_alive', '5m')} "
                        f"({pol.get('recommended_ttl_sec', 300.0):.0f}s)"
                    )
                    print(f"    - Eviction Protection Score: {pol.get('eviction_protection_score', 0.0):.2f}")
                print("=" * 84 + "\n")
            return 0

        dry_run = getattr(args, "dry_run", False) or (not do_evict and not recycle_tachy)
        res = default_supervisor.run_vascular_perfusion_sweep(
            dry_run=dry_run, recycle_tachycardic=recycle_tachy
        )
        if show_cold_starts:
            res["cold_start_telemetry"] = default_supervisor.get_cold_start_report()

        if getattr(args, "json", False):
            print(json.dumps(res, indent=2))
        else:
            mode_tag = "DRY-RUN AUDIT" if dry_run else "VASOCONSTRICTION EVICTION SWEEP"
            print(f"\n🫀 [Whole Body Compute: Selective Vascular Perfusion & Glymphatic RAM Hygiene ({mode_tag})]")
            print("=" * 84)
            print(
                f"  • Unified RAM Headroom: {res['before_free_ram_gb']:.2f} GB free -> "
                f"{res['after_free_ram_gb']:.2f} GB free (+{res['reclaimed_ram_gb']:.2f} GB reclaimed)"
            )
            print(f"  • Evictable Specialist RAM: {res['estimated_reclaimable_gb']:.2f} GB")
            print(
                f"  • Protected Tier-0 Autonomic Processes: {len(res['protected_autonomic_processes'])} "
                f"({', '.join(p['name'] for p in res['protected_autonomic_processes']) or 'none'})"
            )
            for tachy in res.get("tachycardic_autonomic", []):
                t_tag = "⚠️ Autonomic Tachycardia Flagged" if dry_run else "Recycled Tachycardic Autonomic Process"
                print(f"  • ⚠️  {t_tag}: [{tachy['pid']}] {tachy['name']} - {tachy['cpu_pct']:.1f}% CPU (up {tachy['elapsed']})")
            if res["evicted_processes"]:
                action_word = "Evictable" if dry_run else "Evicted to Darwin Inactive File Cache"
                for p in res["evicted_processes"]:
                    print(f"  • {action_word}: [{p['pid']}] {p['name']} ({p['tier']}) - {p['rss_gb']:.2f} GB RSS (up {p['elapsed']})")
            else:
                print("  • Status: Clean - 0 idle Tier-1/Tier-2 specialist models bloating Unified RAM.")
            for t in res.get("reaped_tests", []):
                tag = "Would Reap Hung Test" if dry_run else "Reaped Hung Test"
                print(f"  • 🛑 {tag}: [{t['pid']}] {t['name']} - {t['cpu_pct']:.1f}% CPU (up {t['elapsed']})")
            for c in res.get("clamped_helpers", []):
                tag = "Would Clamp to E-Cores" if dry_run else "Clamped to Apple Silicon E-Cores (taskpolicy -b)"
                print(f"  • ❄️  {tag}: [{c['pid']}] {c['name']} - {c['cpu_pct']:.1f}% CPU (up {c['elapsed']})")
            print(f"  • Note: {res['darwin_file_cache_note']}")
            if show_cold_starts and "cold_start_telemetry" in res:
                cs = res["cold_start_telemetry"]
                print("-" * 84)
                print(
                    f"  • Cold-Start Telemetry: {cs.get('total_models', 0)} tracked models, "
                    f"{cs.get('total_cold_starts', 0)} cold starts ({cs.get('total_cold_penalty_ms', 0):.1f}ms penalty)"
                )
            print("=" * 84 + "\n")
        return 0

    if cmd == "daemon":
        from whole_body_compute.daemon import manage_launchd_daemon

        action = getattr(args, "action", "status") or "status"
        res = manage_launchd_daemon(
            action=action,
            dry_run=bool(getattr(args, "dry_run", False)),
        )
        if getattr(args, "json", False):
            print(json.dumps(res, indent=2))
        else:
            print("\n" + res.get("message", "") + "\n")
        return 0 if res.get("status") == "ok" else 1

    if cmd in ("logs", "history"):
        from whole_body_compute.telemetry_server import (
            format_telemetry_history_table,
            get_live_apple_silicon_vitals,
            history_logger,
        )

        limit = getattr(args, "limit", 25)
        transitions_only = getattr(args, "transitions", False)
        if getattr(args, "json", False):
            get_live_apple_silicon_vitals()
            print(json.dumps(history_logger.get_history_report(limit=limit), indent=2))
        else:
            print(format_telemetry_history_table(limit=limit, transitions_only=transitions_only))
        return 0

    if cmd in ("hud", "dashboard"):
        import subprocess
        from whole_body_compute.telemetry_server import (
            format_telemetry_history_table,
            serve_telemetry,
        )

        if getattr(args, "logs", False):
            print(format_telemetry_history_table(limit=getattr(args, "limit", 20)))
            return 0

        host = getattr(args, "host", "127.0.0.1")
        port = getattr(args, "port", 9381)
        if getattr(args, "open", False):
            subprocess.Popen(["open", f"http://{host}:{port}"])
        serve_telemetry(host=host, port=port)
        return 0

    if cmd == "immune-hook":
        from whole_body_compute.organs.immune import (
            run_ide_post_tool_hook,
            run_ide_pre_tool_hook,
        )

        phase = getattr(args, "phase", "pre")
        try:
            raw_in = sys.stdin.read().strip()
            payload = json.loads(raw_in) if raw_in else {}
        except Exception:
            payload = {}

        if phase == "post":
            out = run_ide_post_tool_hook(payload)
        else:
            out = run_ide_pre_tool_hook(payload)
        print(json.dumps(out))
        return 0

    dispatched = dispatch_organ_and_cognitive_command(cmd, args)
    if dispatched is not None:
        return dispatched

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
