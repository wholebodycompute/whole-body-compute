"""
Tests for Native Core Telemetry, Data Directory Resolution, and Outbound Sync
(Open-Core standalone support without voicefi package dependency).
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from whole_body_compute.cli import main as cli_main
from whole_body_compute.license import LicenseManager, _get_default_license_path
from whole_body_compute.local.telemetry import (
    capture_wbc_hybrid_step,
    get_analytics_db_path,
    record_hybrid_step,
)
from whole_body_compute.procedural_telemetry import (
    ConsentTier,
    DEFAULT_POSTHOG_API_KEY,
    ProceduralDNAHarvester,
    TelemetryConsentManager,
    flush_telemetry_outbox,
    format_usage_stats_table,
    get_comprehensive_usage_stats_report,
    get_usage_stats_path,
    get_wbc_data_dir,
    load_usage_stats,
    send_vitals_rollup,
    track_wbc_event,
)


def test_wbc_data_dir_resolution_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify resolution order: WBC_DATA_DIR > ~/.whole_body_compute > ~/.voicefi > default."""
    custom_dir = tmp_path / "custom_wbc_data"
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    # 1. WBC_DATA_DIR environment variable takes precedence
    monkeypatch.setenv("WBC_DATA_DIR", str(custom_dir))
    assert get_wbc_data_dir() == custom_dir

    # Clear env
    monkeypatch.delenv("WBC_DATA_DIR", raising=False)

    # 2. When ~/.whole_body_compute exists, use it
    wbc_dir = fake_home / ".whole_body_compute"
    wbc_dir.mkdir()
    assert get_wbc_data_dir() == wbc_dir

    # 3. If ~/.whole_body_compute does NOT exist, but ~/.voicefi exists, use ~/.voicefi (backward-compat)
    wbc_dir.rmdir()
    vifi_dir = fake_home / ".voicefi"
    vifi_dir.mkdir()
    assert get_wbc_data_dir() == vifi_dir

    # 4. If neither exists (fresh standalone Open-Core install), default to ~/.whole_body_compute
    vifi_dir.rmdir()
    assert get_wbc_data_dir() == wbc_dir


def test_license_and_analytics_default_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify license and analytics paths resolve to get_wbc_data_dir() while respecting env vars."""
    custom_dir = tmp_path / "wbc_state"
    custom_dir.mkdir()
    monkeypatch.setenv("WBC_DATA_DIR", str(custom_dir))
    monkeypatch.delenv("VOICEFI_LICENSE_PATH", raising=False)
    monkeypatch.delenv("VOICEFI_ANALYTICS_DB", raising=False)

    # Default paths resolve inside WBC_DATA_DIR
    assert _get_default_license_path() == custom_dir / "license.json"
    assert get_analytics_db_path() == custom_dir / "analytics.db"

    # Explicit env vars override
    monkeypatch.setenv("VOICEFI_LICENSE_PATH", str(tmp_path / "override_lic.json"))
    monkeypatch.setenv("VOICEFI_ANALYTICS_DB", str(tmp_path / "override_adb.db"))
    assert _get_default_license_path() == tmp_path / "override_lic.json"
    assert get_analytics_db_path() == tmp_path / "override_adb.db"

    mgr = LicenseManager()
    assert mgr.license_path == tmp_path / "override_lic.json"


def test_flush_outbox_blocked_by_consent_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify flush_telemetry_outbox strictly blocks network egress when opted out or under DO_NOT_TRACK."""
    outbox_file = tmp_path / "telemetry_outbox.jsonl"
    cfg_file = tmp_path / "telemetry_consent.json"
    monkeypatch.setenv("WBC_TELEMETRY_OUTBOX_PATH", str(outbox_file))
    monkeypatch.setenv("WBC_TELEMETRY_CONFIG_PATH", str(cfg_file))
    for k in ("DO_NOT_TRACK", "WBC_TELEMETRY_DISABLED", "BODY_TELEMETRY", "CI", "GITHUB_ACTIONS"):
        monkeypatch.delenv(k, raising=False)

    # Create dummy cards in outbox
    card1 = {"signature_hash": "abc1", "card_type": "compiler_ir_card", "recipe_id": "test_recipe"}
    outbox_file.write_text(json.dumps(card1) + "\n", encoding="utf-8")

    mgr = TelemetryConsentManager(config_path=cfg_file)
    assert mgr.get_effective_tier() == ConsentTier.TIER_2_PROCEDURAL_DNA

    # Opt-out (`disable`) strictly blocks egress
    mgr.disable()
    assert mgr.get_effective_tier() == ConsentTier.TIER_0_LOCAL

    with patch("urllib.request.urlopen") as mock_urlopen:
        res = flush_telemetry_outbox(outbox_path=outbox_file, consent_manager=mgr)
        assert res["status"] == "blocked_by_consent"
        assert res["sent_count"] == 0
        assert res["pending_count"] == 1
        mock_urlopen.assert_not_called()

    # Re-enable Tier 2, then trigger DO_NOT_TRACK circuit breaker
    mgr.enable(tier=2)
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    with patch("urllib.request.urlopen") as mock_urlopen:
        res = flush_telemetry_outbox(outbox_path=outbox_file, consent_manager=mgr)
        assert res["status"] == "blocked_by_consent"
        assert res["sent_count"] == 0
        mock_urlopen.assert_not_called()


def test_flush_outbox_transmits_and_stamps_synced_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify flush_telemetry_outbox transmits unsynced cards, stamps synced_at, and handles deduplication."""
    outbox_file = tmp_path / "telemetry_outbox.jsonl"
    cfg_file = tmp_path / "telemetry_consent.json"
    monkeypatch.setenv("WBC_TELEMETRY_OUTBOX_PATH", str(outbox_file))
    monkeypatch.setenv("WBC_TELEMETRY_CONFIG_PATH", str(cfg_file))
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.delenv("BODY_TELEMETRY", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("WBC_TELEMETRY_DISABLED", raising=False)

    card1 = {"signature_hash": "card_one", "card_type": "compiler_ir_card", "recipe_id": "rec_one"}
    card2 = {"signature_hash": "card_two", "card_type": "router_habit_card", "intent_domain": "SHELL"}
    outbox_file.write_text(json.dumps(card1) + "\n" + json.dumps(card2) + "\n", encoding="utf-8")

    mgr = TelemetryConsentManager(config_path=cfg_file)
    mgr.opt_in(tier=2)
    assert mgr.can_share_procedural_dna() is True

    # Mock HTTP response
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.getcode.return_value = 200
    mock_resp.__enter__.return_value = mock_resp

    posted_requests = []

    def fake_urlopen(req, timeout=None):
        posted_requests.append(req)
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        res = flush_telemetry_outbox(
            outbox_path=outbox_file,
            consent_manager=mgr,
            ingest_url="https://mock.posthog.com/capture/",
        )
        assert res["status"] == "ok"
        assert res["sent_count"] == 2
        assert res["already_synced_count"] == 2
        assert res["pending_count"] == 0
        assert len(posted_requests) == 2

        # Verify payload structure
        req_data = json.loads(posted_requests[0].data.decode("utf-8"))
        assert req_data["event"] == "wbc_procedural_dna_card"
        assert req_data["properties"]["source"] == "whole_body_compute_open_core"
        assert req_data["properties"]["signature_hash"] == "card_one"

    # Verify persisted outbox file has synced_at stamped on both cards
    updated_lines = [json.loads(line) for line in outbox_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(updated_lines) == 2
    assert "synced_at" in updated_lines[0]
    assert "synced_at" in updated_lines[1]

    # Re-running flush should now find 0 pending cards without making any HTTP requests
    with patch("urllib.request.urlopen") as mock_urlopen:
        res2 = flush_telemetry_outbox(outbox_path=outbox_file, consent_manager=mgr)
        assert res2["status"] == "ok"
        assert res2["sent_count"] == 0
        assert res2["already_synced_count"] == 2
        assert res2["pending_count"] == 0
        mock_urlopen.assert_not_called()


def test_send_vitals_rollup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify send_vitals_rollup sends wbc_vitals_rollup event gated by can_share_vitals."""
    cfg_file = tmp_path / "telemetry_consent.json"
    for k in ("DO_NOT_TRACK", "WBC_TELEMETRY_DISABLED", "BODY_TELEMETRY", "CI", "GITHUB_ACTIONS"):
        monkeypatch.delenv(k, raising=False)

    mgr = TelemetryConsentManager(config_path=cfg_file)

    # 1. Opt-out Tier 0 -> blocked
    mgr.opt_out()
    res0 = send_vitals_rollup({"cpu_load": 0.42}, consent_manager=mgr)
    assert res0["status"] == "blocked_by_consent"

    # 2. Opt-in Tier 1 -> allowed with zero-IP guarantee
    mgr.opt_in(tier=1)
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.getcode.return_value = 200
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        res1 = send_vitals_rollup({"cpu_load": 0.42, "free_ram_gb": 16.5}, consent_manager=mgr)
        assert res1["status"] == "ok"
        assert res1["event"] == "wbc_vitals_rollup"
        assert mock_urlopen.call_count == 1
        call_req = mock_urlopen.call_args[0][0]
        payload = json.loads(call_req.data.decode("utf-8"))
        assert payload["event"] == "wbc_vitals_rollup"
        assert payload["properties"]["source"] == "whole_body_compute_open_core"
        assert payload["properties"]["free_ram_gb"] == 16.5
        assert payload["properties"]["$ip"] is None
        assert payload["properties"]["$geoip_disable"] is True


def test_native_hybrid_step_telemetry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify capture_wbc_hybrid_step dispatches zero-PII hybrid step natively via urllib."""
    cfg_file = tmp_path / "telemetry_consent.json"
    adb_file = tmp_path / "analytics.db"
    monkeypatch.setenv("WBC_TELEMETRY_CONFIG_PATH", str(cfg_file))
    monkeypatch.setenv("VOICEFI_ANALYTICS_DB", str(adb_file))
    monkeypatch.setenv("WBC_TEST_SYNC", "1")
    for k in ("DO_NOT_TRACK", "WBC_TELEMETRY_DISABLED", "BODY_TELEMETRY", "CI", "GITHUB_ACTIONS"):
        monkeypatch.delenv(k, raising=False)

    mgr = TelemetryConsentManager(config_path=cfg_file)

    # Under explicit Tier 0 opt-out, capture_wbc_hybrid_step returns False
    mgr.disable()
    assert capture_wbc_hybrid_step({"stage": "local_scout"}, async_send=False) is False

    # Under Tier 1+, capture_wbc_hybrid_step dispatches with $ip: None and hardware envelope
    mgr.opt_in(tier=1)

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.getcode.return_value = 200
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        success = capture_wbc_hybrid_step(
            {
                "stage": "local_scout",
                "model_name": "qwen2.5-coder:1.5b",
                "is_local": True,
                "input_tokens": 1500,
                "output_tokens": 120,
            },
            async_send=False,
        )
        assert success is True
        assert mock_urlopen.call_count == 1
        req = mock_urlopen.call_args[0][0]
        data = json.loads(req.data.decode("utf-8"))
        assert data["event"] == "wbc_hybrid_step"
        assert data["properties"]["source"] == "whole_body_compute_open_core"
        assert data["properties"]["stage"] == "local_scout"
        assert data["properties"]["$ip"] is None
        assert data["properties"]["$geoip_disable"] is True
        assert "chip_family" in data["properties"]


def test_cli_telemetry_sync_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """Verify CLI `body telemetry sync/show/disable/enable` executes and outputs structured JSON report."""
    outbox_file = tmp_path / "telemetry_outbox.jsonl"
    cfg_file = tmp_path / "telemetry_consent.json"
    last_payload_file = tmp_path / "last_telemetry_payload.json"
    monkeypatch.setenv("WBC_TELEMETRY_OUTBOX_PATH", str(outbox_file))
    monkeypatch.setenv("WBC_TELEMETRY_CONFIG_PATH", str(cfg_file))
    monkeypatch.setenv("WBC_LAST_TELEMETRY_PAYLOAD_PATH", str(last_payload_file))
    for k in ("DO_NOT_TRACK", "WBC_TELEMETRY_DISABLED", "BODY_TELEMETRY", "CI", "GITHUB_ACTIONS"):
        monkeypatch.delenv(k, raising=False)

    # 1. Disable via CLI `body telemetry disable`, then verify sync reports blocked_by_consent
    card = {"signature_hash": "cli_card_1", "card_type": "router_habit_card"}
    outbox_file.write_text(json.dumps(card) + "\n", encoding="utf-8")

    assert cli_main(["telemetry", "disable", "--json"]) == 0
    capsys.readouterr()

    code = cli_main(["telemetry", "sync", "--json"])
    assert code == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["status"] == "blocked_by_consent"

    # 2. Re-enable via CLI `body telemetry enable`, then run sync with mocked HTTP
    assert cli_main(["telemetry", "enable", "--json"]) == 0
    capsys.readouterr()

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.getcode.return_value = 200
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        code = cli_main(["dna", "sync", "--json"])
        assert code == 0
        captured = capsys.readouterr()
        report = json.loads(captured.out)
        assert report["status"] == "ok"
        assert report["sent_count"] == 1
        assert report["pending_count"] == 0

    # 3. Inspect exact last sent payload via `body telemetry show --json`
    assert cli_main(["telemetry", "show", "--json"]) == 0
    show_report = json.loads(capsys.readouterr().out)
    assert show_report["status"] == "ok"
    assert show_report["last_outbound_payload"]["event"] == "wbc_procedural_dna_card"
    assert show_report["last_outbound_payload"]["properties"]["$ip"] is None


def test_track_wbc_event_and_usage_stats_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Verify track_wbc_event increments usage_stats.json, sends PostHog funnel pings, and renders via `body stats`."""
    stats_file = tmp_path / "usage_stats.json"
    cfg_file = tmp_path / "telemetry_consent.json"
    monkeypatch.setenv("WBC_USAGE_STATS_PATH", str(stats_file))
    monkeypatch.setenv("WBC_TELEMETRY_CONFIG_PATH", str(cfg_file))
    monkeypatch.setenv("WBC_TEST_TELEMETRY_PING", "1")
    monkeypatch.setenv("WBC_INTERNAL_DEV", "1")
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.delenv("BODY_TELEMETRY", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("WBC_TELEMETRY_DISABLED", raising=False)

    assert DEFAULT_POSTHOG_API_KEY == "phc_D3TxQWXwnd9Z6wBBK3DpnsdKQAHdLpSSMxup4qL8djc2"
    assert get_usage_stats_path() == stats_file

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.getcode.return_value = 200
    mock_resp.__enter__.return_value = mock_resp

    posted_events = []

    def fake_urlopen(req, timeout=None):
        posted_events.append(json.loads(req.data.decode("utf-8")))
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        res_audit = track_wbc_event(
            "wbc_audit_completed",
            {
                "mode": "standalone_audit_py",
                "conversations_scanned": 12,
                "wasted_bloat_tokens": 250000,
                "wasted_spend_usd": 14.5,
            },
            async_send=False,
        )
        assert res_audit["local_recorded"] is True
        assert res_audit["status"] == "ok"

        res_begin = track_wbc_event(
            "wbc_body_begin",
            {"dry_run": False, "wired_harnesses_count": 3},
            async_send=False,
        )
        assert res_begin["status"] == "ok"

        track_wbc_event(
            "wbc_pal_edit_applied",
            {"engine_used": "fast_regex", "elapsed_ms": 12.4, "app_name": "Google Chrome"},
            async_send=False,
        )
        track_wbc_event(
            "wbc_rem_sleep_completed",
            {
                "dry_run": False,
                "passive_workflows_crystallized": 2,
                "distillation_pairs_harvested": 8,
                "agy_trajectories_harvested": 4,
            },
            async_send=False,
        )

    assert len(posted_events) == 4
    assert posted_events[0]["api_key"] == DEFAULT_POSTHOG_API_KEY
    assert posted_events[0]["event"] == "wbc_audit_completed"
    assert posted_events[0]["properties"]["wasted_bloat_tokens"] == 250000
    assert posted_events[0]["properties"]["is_internal_dev"] is True
    assert posted_events[0]["properties"]["$internal_or_test_user"] is True
    assert posted_events[0]["properties"]["$set"]["$internal_or_test_user"] is True

    loaded = load_usage_stats(stats_file)
    assert loaded["total_events"] == 4
    assert loaded["events"]["wbc_audit_completed"]["count"] == 1
    assert loaded["events"]["wbc_body_begin"]["count"] == 1
    assert loaded["events"]["wbc_pal_edit_applied"]["count"] == 1
    assert loaded["events"]["wbc_rem_sleep_completed"]["count"] == 1

    # Verify DO_NOT_TRACK=1 blocks remote PostHog ping while still recording local counter
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    with patch("urllib.request.urlopen") as mock_dnt_url:
        res_dnt = track_wbc_event("wbc_cli_command", {"command": "doctor"}, async_send=False)
        assert res_dnt["local_recorded"] is True
        assert res_dnt["status"] == "blocked_by_env"
        mock_dnt_url.assert_not_called()

    # Verify `body stats --json` and `body telemetry stats`
    monkeypatch.delenv("WBC_TEST_TELEMETRY_PING", raising=False)
    capsys.readouterr()
    code_json = cli_main(["stats", "--json"])
    assert code_json == 0
    stats_report = json.loads(capsys.readouterr().out)
    assert stats_report["posthog_project_id"] == 730830
    assert stats_report["usage_ledger"]["events"]["wbc_audit_completed"]["count"] == 1
    assert stats_report["usage_ledger"]["events"]["wbc_pal_edit_applied"]["count"] == 1

    code_txt = cli_main(["telemetry", "stats"])
    assert code_txt == 0
    txt_out = capsys.readouterr().out
    assert "Whole Body Compute - Unified Usage, Funnel & Automation Telemetry" in txt_out
    assert "No-Code / Spend Audits" in txt_out
    assert "Nocturnal REM Dreaming" in txt_out


def test_python_dash_s_hook_telemetry_imports_and_activate_alias(tmp_path: Path) -> None:
    """Regression: `python -S` hooks must import telemetry/license/skills without PyYAML, and `body activate` must parse."""
    import os
    import subprocess
    import sys

    src_dir = Path(__file__).resolve().parents[1] / "src"
    code = (
        f"import sys; sys.path.insert(0, {str(src_dir)!r}); "
        "import whole_body_compute.procedural_telemetry as pt; "
        "import whole_body_compute.license as lic; "
        "import whole_body_compute.skills_telemetry as st; "
        "assert pt.get_wbc_data_dir() is not None; "
        "assert lic.LicenseManager().get_active_tier() in ('community', 'pro', 'team', 'enterprise'); "
        "assert st.get_skill_telemetry_path() is not None"
    )
    proc = subprocess.run(
        [sys.executable, "-S", "-c", code],
        env={**os.environ, "WBC_DATA_DIR": str(tmp_path / ".wbc_data")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"python -S import failed: {proc.stderr}"

    from whole_body_compute.cli_parser import build_parser

    parser = build_parser()
    ns_flag = parser.parse_args(["activate", "--key", "WBC-PRO-TEST-KEY"])
    assert ns_flag.command == "activate"
    assert ns_flag.activate_key == "WBC-PRO-TEST-KEY"
    ns_pos = parser.parse_args(["activate", "WBC-PRO-POS-KEY"])
    assert ns_pos.command == "activate"
    assert ns_pos.license_key_pos == "WBC-PRO-POS-KEY"


def test_open_core_sovereign_red_line_and_workflow_anatomy_privacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify open-core Sovereign Red-Line sanitizer, workflow anatomy fingerprinting, and hybrid card harvesting."""
    from whole_body_compute.integrations.pal_workflow import (
        ActionType,
        PalWorkflowRecipe,
        PalWorkflowRunner,
        WorkflowStep,
    )
    from whole_body_compute.procedural_telemetry import (
        generalize_workflow_recipe,
        sanitize_telemetry_payload,
    )

    dirty = {
        "prompt": "Summarize Jane Doe's private tax return",
        "code": "def secret_algo(): pass",
        "window_title": "Jane Doe - Chase Bank Account #99887766",
        "email": "jane.doe@example.com",
        "ip_address": "192.168.1.42",
        "api_secret": "super_secret_val",
        "note": "Contact bob@corp.io or 512-555-0199 from /Users/alice/secret_repo on 10.0.0.8 with SSN 123-45-6789",
        "$ip": "203.0.113.5",
    }
    cleaned = sanitize_telemetry_payload(dirty)
    serialized = json.dumps(cleaned)
    for forbidden_key in ("prompt", "code", "window_title", "email", "ip_address", "api_secret"):
        assert forbidden_key not in cleaned
    assert cleaned["$ip"] is None
    for leaked in (
        "Jane Doe",
        "secret_algo",
        "bob@corp.io",
        "512-555-0199",
        "/Users/alice",
        "10.0.0.8",
        "123-45-6789",
        "203.0.113.5",
    ):
        assert leaked not in serialized

    recipe = PalWorkflowRecipe(
        id="custom_crm_sync",
        name="Sync CRM Notes",
        description="Automates Chrome to Slack handoff",
        triggers=["sync notes"],
        steps=[
            WorkflowStep(
                name="focus_browser",
                action=ActionType.ACTIVATE_APP,
                app_name="Google Chrome",
                window_title="Private Customer Doc",
            ),
            WorkflowStep(
                name="run_transform",
                action=ActionType.RUN_SHELL,
                command="python3 transform.py",
            ),
            WorkflowStep(
                name="notify_slack",
                action=ActionType.ACTIVATE_APP,
                app_name="Slack",
                window_title="Confidential DM",
            ),
        ],
    )
    anatomy = PalWorkflowRunner.summarize_recipe_anatomy(recipe)
    assert anatomy["is_custom_user_recipe"] is True
    assert anatomy["app_chain"] == ["Google Chrome", "Slack"]
    assert anatomy["action_chain"] == ["activate_app", "run_shell", "activate_app"]
    assert "Private Customer Doc" not in json.dumps(anatomy)
    assert "Confidential DM" not in json.dumps(anatomy)

    gen = generalize_workflow_recipe(
        {
            "id": "inline_recipe",
            "notes": ["Customer Jane Smith order ORD-98234-XYZ"],
            "steps": [{"command": "python ~/Projects/secret-repo/scripts/run.py --token sk-liveSecretKey1234567890abcd"}],
        }
    )
    gen_str = json.dumps(gen)
    assert "Jane Smith" not in gen_str
    assert "secret-repo" not in gen_str
    assert "sk-liveSecretKey" not in gen_str

    monkeypatch.setenv("WBC_DATA_DIR", str(tmp_path))
    harvester = ProceduralDNAHarvester(
        recipes_dir=tmp_path / "recipes",
        outbox_path=tmp_path / "telemetry_outbox.jsonl",
    )
    hybrid_cards = harvester.extract_hybrid_efficiency_cards()
    assert len(hybrid_cards) == 1
    assert hybrid_cards[0]["card_type"] == "hybrid_compute_efficiency_card"
    assert "no_pii" in hybrid_cards[0]["zero_pii_guarantees"]


def test_first_run_telemetry_notice_and_tier_enforcement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Verify:
    1. First-run stderr transparency notice emits once, skips --json, skips opted-out/DO_NOT_TRACK.
    2. TIER_0_LOCAL blocks 100% of network calls across all 4 outbound APIs.
    3. TIER_1_VITALS allows vitals/hybrid step pings but blocks Procedural DNA cards.
    4. All env circuit-breakers (`DO_NOT_TRACK`, `WBC_TELEMETRY_DISABLED`, `BODY_TELEMETRY`, `CI`, `GITHUB_ACTIONS`) halt egress.
    """
    from whole_body_compute.procedural_telemetry import maybe_emit_first_run_telemetry_notice

    cfg_file = tmp_path / "telemetry_consent.json"
    outbox_file = tmp_path / "telemetry_outbox.jsonl"
    stats_file = tmp_path / "usage_stats.json"
    adb_file = tmp_path / "analytics.db"
    monkeypatch.setenv("WBC_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("WBC_TELEMETRY_CONFIG_PATH", str(cfg_file))
    monkeypatch.setenv("WBC_TELEMETRY_OUTBOX_PATH", str(outbox_file))
    monkeypatch.setenv("WBC_USAGE_STATS_PATH", str(stats_file))
    monkeypatch.setenv("VOICEFI_ANALYTICS_DB", str(adb_file))
    monkeypatch.setenv("WBC_TEST_SYNC", "1")
    monkeypatch.setenv("WBC_TEST_TELEMETRY_PING", "1")
    monkeypatch.setenv("WBC_INTERNAL_DEV", "1")
    for k in (
        "DO_NOT_TRACK",
        "WBC_TELEMETRY_DISABLED",
        "VOICEFI_TELEMETRY_DISABLED",
        "BODY_TELEMETRY",
        "CI",
        "GITHUB_ACTIONS",
        "WBC_SUPPRESS_TELEMETRY_NOTICE",
    ):
        monkeypatch.delenv(k, raising=False)

    outbox_file.write_text(
        json.dumps({"signature_hash": "tier_gate_1", "card_type": "compiler_ir_card"}) + "\n",
        encoding="utf-8",
    )

    mgr = TelemetryConsentManager(config_path=cfg_file)
    # Remove PYTEST_CURRENT_TEST temporarily to test interactive first-run notice emission
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    buf_json = io.StringIO()
    assert maybe_emit_first_run_telemetry_notice(mgr, is_json=True, stream=buf_json) is False
    assert buf_json.getvalue() == ""

    buf_err = io.StringIO()
    assert maybe_emit_first_run_telemetry_notice(mgr, is_json=False, stream=buf_err) is True
    notice_txt = buf_err.getvalue()
    assert "never code, paths, prompts, or PII" in notice_txt
    assert "body telemetry disable" in notice_txt
    assert mgr.has_shown_notice() is True

    # Second call must not emit twice
    buf_second = io.StringIO()
    assert maybe_emit_first_run_telemetry_notice(mgr, is_json=False, stream=buf_second) is False
    assert buf_second.getvalue() == ""
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_first_run_telemetry_notice_and_tier_enforcement")

    # TIER_0_LOCAL: zero urllib calls from all 4 outbound paths
    mgr.disable()
    with patch("urllib.request.urlopen") as mock_u0:
        assert track_wbc_event("wbc_cli_command", {"command": "doctor"}, consent_manager=mgr, async_send=False)["status"] == "blocked_by_consent"
        assert send_vitals_rollup({"free_ram_gb": 12.0}, consent_manager=mgr)["status"] == "blocked_by_consent"
        assert flush_telemetry_outbox(outbox_path=outbox_file, consent_manager=mgr)["status"] == "blocked_by_consent"
        assert capture_wbc_hybrid_step({"stage": "local_scout"}, async_send=False) is False
        mock_u0.assert_not_called()

    # TIER_1_VITALS: vitals/events/hybrid step transmit; DNA outbox flush stays blocked
    mgr.opt_in(tier=1)
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.getcode.return_value = 200
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_u1:
        assert track_wbc_event("wbc_cli_command", {"command": "doctor"}, consent_manager=mgr, async_send=False)["status"] == "ok"
        assert send_vitals_rollup({"free_ram_gb": 12.0}, consent_manager=mgr)["status"] == "ok"
        assert capture_wbc_hybrid_step({"stage": "local_scout"}, async_send=False) is True
        assert mock_u1.call_count == 3
        blocked_dna = flush_telemetry_outbox(outbox_path=outbox_file, consent_manager=mgr)
        assert blocked_dna["status"] == "blocked_by_consent"
        assert mock_u1.call_count == 3

    # Every circuit-breaker env var blocks TIER_2_PROCEDURAL_DNA egress immediately
    mgr.enable(tier=2)
    breakers = [
        ("DO_NOT_TRACK", "1"),
        ("WBC_TELEMETRY_DISABLED", "1"),
        ("VOICEFI_TELEMETRY_DISABLED", "true"),
        ("BODY_TELEMETRY", "0"),
        ("BODY_TELEMETRY", "off"),
        ("BODY_TELEMETRY", "false"),
        ("BODY_TELEMETRY", "airgapped"),
        ("BODY_TELEMETRY", "none"),
        ("BODY_TELEMETRY", "disabled"),
        ("CI", "true"),
        ("GITHUB_ACTIONS", "true"),
    ]
    for env_k, env_v in breakers:
        monkeypatch.setenv(env_k, env_v)
        with patch("urllib.request.urlopen") as mock_ub:
            assert track_wbc_event("wbc_cli_command", {"command": "status"}, consent_manager=mgr, async_send=False)["status"] == "blocked_by_env"
            assert flush_telemetry_outbox(outbox_path=outbox_file, consent_manager=mgr)["status"] == "blocked_by_consent"
            mock_ub.assert_not_called()
        monkeypatch.delenv(env_k, raising=False)


def test_nested_payload_redaction_and_harness_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify deep nested list/dict PII stripping and coding-agent harness envelope detection."""
    from whole_body_compute.procedural_telemetry import (
        _detect_active_agent_harness,
        sanitize_telemetry_payload,
    )

    nested_dirty = {
        "stage": "local_qa",
        "nested": {
            "prompt": "Private user ask",
            "user_email": "alice@secret.org",
            "access_token": "tok_secret",
            "client_secret": "cs_123",
            "items": [
                "See /Users/alice/dev/repo/main.py or C:\\Users\\Alice\\secrets.txt",
                {"window_title": "Confidential Merger.docx", "safe_metric": 42},
                "Call (512) 555-0199 SSN 123-45-6789 IPv6 2001:db8::1 key ghp_1234567890abcdefghijklmnopqrstuv",
            ],
        },
    }
    cleaned = sanitize_telemetry_payload(nested_dirty)
    dumped = json.dumps(cleaned)
    for forbidden in (
        "Private user ask",
        "alice@secret.org",
        "tok_secret",
        "cs_123",
        "/Users/alice",
        "Alice\\secrets.txt",
        "Confidential Merger.docx",
        "512) 555-0199",
        "123-45-6789",
        "2001:db8::1",
        "ghp_1234567890",
    ):
        assert forbidden not in dumped
    assert cleaned["nested"]["items"][1]["safe_metric"] == 42

    import os

    for k in list(os.environ.keys()):
        if k.startswith(("ANTIGRAVITY", "CLAUDE", "CURSOR", "WINDSURF", "CLINE", "GOOSE")) or k in (
            "WBC_CALLER_HARNESS",
            "AGY_SESSION_ID",
        ):
            monkeypatch.delenv(k, raising=False)

    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")
    assert _detect_active_agent_harness() == "claude_code"
    monkeypatch.delenv("CLAUDE_CODE_ENTRYPOINT", raising=False)

    monkeypatch.setenv("ANTIGRAVITY_AGENT", "1")
    assert _detect_active_agent_harness() == "antigravity"
    monkeypatch.delenv("ANTIGRAVITY_AGENT", raising=False)

    monkeypatch.setenv("CURSOR_TRACE_ID", "abc")
    assert _detect_active_agent_harness() == "cursor"
    monkeypatch.delenv("CURSOR_TRACE_ID", raising=False)

    monkeypatch.setenv("WBC_CALLER_HARNESS", "Custom-Harness_v2")
    assert _detect_active_agent_harness() == "custom-harness_v2"


def test_learning_telemetry_memory_skills_and_workflows_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Verify zero-PII learning telemetry events and Procedural DNA harvest across:
    1. DotMemory fact learning (`wbc_memory_fact_learned`) & habit consolidation (`wbc_habits_consolidated`).
    2. Skill activations (`wbc_skill_activated`) & skill pruning (`wbc_skills_pruned`).
    3. Custom user workflows saved in `~/.whole_body_compute/workflows/` harvested by ProceduralDNAHarvester.
    """
    from whole_body_compute.local.dot_memory import MemoryStore
    from whole_body_compute.pro_loader import ProUpgradeRequiredError
    from whole_body_compute.skills_telemetry import prune_skills, record_skill_activation

    stats_file = tmp_path / "usage_stats.json"
    cfg_file = tmp_path / "telemetry_consent.json"
    dot_db = tmp_path / "dot_memory.db"
    skill_tel_file = tmp_path / "skill_telemetry.json"
    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("WBC_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("WBC_USAGE_STATS_PATH", str(stats_file))
    monkeypatch.setenv("WBC_TELEMETRY_CONFIG_PATH", str(cfg_file))
    monkeypatch.setenv("WBC_DOT_MEMORY_PATH", str(dot_db))
    monkeypatch.setenv("WBC_SKILL_TELEMETRY_PATH", str(skill_tel_file))

    has_dot_memory = True
    try:
        store = MemoryStore(db_path=dot_db, auto_seed=False)
        store.remember("Always use uv run pytest for test verification", category="preference", tags="pytest,uv")
        store.remember("Draft email reply to client", category="shipped_action", tags="send_email_reply")
        store.remember("Second email reply to vendor", category="shipped_action", tags="send_email_reply")
        consol = store.consolidate_habits(min_occurrences=2)
        assert consol["consolidated_habits_count"] >= 1
    except ProUpgradeRequiredError:
        has_dot_memory = False

    # Record skill activation milestone (read #1)
    activated = record_skill_activation(
        "/Users/alice/.gemini/config/plugins/test-plugin/skills/custom-skill/SKILL.md",
        conversation_id="conv_learning_1",
    )
    assert activated == "custom-skill"

    # Prune a dormant custom skill
    ws_root = tmp_path / "ws"
    dormant_skill_dir = ws_root / ".agents" / "skills" / "unused-skill"
    dormant_skill_dir.mkdir(parents=True, exist_ok=True)
    (dormant_skill_dir / "SKILL.md").write_text(
        "---\nname: unused-skill\ndescription: Dormant test skill\n---\nBody\n",
        encoding="utf-8",
    )
    prune_res = prune_skills(
        workspace_root=ws_root,
        gemini_config_dir=tmp_path / "gemini_cfg",
        archive_dir=tmp_path / "skills_archive",
        brain_dir=tmp_path / "empty_brain",
    )
    assert prune_res["shelved_count"] == 1

    # Verify events recorded in usage_stats.json with zero raw fact text
    usage = load_usage_stats(stats_file)
    usage_str = json.dumps(usage)
    assert "Always use uv run pytest" not in usage_str
    if has_dot_memory:
        assert usage["events"]["wbc_memory_fact_learned"]["count"] >= 3
        assert usage["events"]["wbc_habits_consolidated"]["count"] == 1
    assert usage["events"]["wbc_skill_activated"]["count"] == 1
    assert usage["events"]["wbc_skills_pruned"]["count"] == 1

    # Write a taught workflow in `workflows/` (PalTrainer default dir) and verify harvester picks it up
    (workflows_dir / "taught_custom_flow.yaml").write_text(
        "id: taught_custom_flow\n"
        "name: Taught Custom Flow\n"
        "steps:\n"
        "  - name: step_1\n"
        "    action: run_shell\n"
        "    command: echo hello\n",
        encoding="utf-8",
    )
    harvester = ProceduralDNAHarvester(
        recipes_dir=tmp_path / "recipes",
        outbox_path=tmp_path / "telemetry_outbox.jsonl",
    )
    harvest_rep = harvester.harvest_all(persist=True)
    ir_ids = {c["recipe_id"] for c in harvest_rep["compiler_ir_cards"]}
    assert "taught_custom_flow" in ir_ids

    eff_card = harvest_rep["hybrid_efficiency_cards"][0]
    if has_dot_memory:
        assert eff_card["learning_summary"]["total_memories"] >= 4
        assert eff_card["learning_summary"]["by_category"]["preference"] == 1
    assert eff_card["learning_summary"]["tracked_skills_count"] == 1
    assert eff_card["learning_summary"]["total_skill_reads"] == 1


def test_fetch_posthog_cloud_funnel_summary_excludes_pypi_scanners() -> None:
    """Verify HogQL cloud summary predicate excludes pp-fuzz and <=2GB Linux post-PyPI scanner bots."""
    from whole_body_compute.procedural_telemetry import (
        fetch_posthog_cloud_funnel_summary,
    )

    captured_sql: list[str] = []

    def _fake_urlopen(req: MagicMock, timeout: float = 10.0) -> MagicMock:
        body = json.loads(req.data.decode("utf-8"))
        captured_sql.append(body["query"]["query"])
        row = [1396, 939, 11, 37, 1, 1, 1, 39, 4, 0, 0, 9, 19, 1, 4, 1, 3, 3, 1, 0, 0, 56]
        resp = MagicMock()
        resp.__enter__.return_value = io.BytesIO(json.dumps({"results": [row]}).encode("utf-8"))
        return resp

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        summary = fetch_posthog_cloud_funnel_summary(personal_api_key="phx_test_readonly")

    assert summary["status"] == "ok"
    assert summary["ext_product_cohorts"] == 1
    assert summary["ext_cohorts_hitting_limits"] == 1
    assert summary["ext_scanner_cohorts_filtered"] == 56
    assert len(captured_sql) == 1
    sql = captured_sql[0]
    assert "event LIKE '/tmp/%'" in sql
    assert "IntelR CoreTM i7-10700 CPU @ 2.90GHz" in sql
    assert "ram_gb_bucket" in sql
    assert "actual_cumulative_input_tokens" in sql

    table_str = format_usage_stats_table(
        {
            "consent": {"cohort_id": "abc12345", "effective_tier": "TIER_2_PROCEDURAL_DNA"},
            "usage_ledger": {"commands": {}, "events": {}},
            "shield_savings": {},
            "workflow_automations": {},
            "ambient_and_dreaming": {},
            "skills_telemetry": {},
            "memory_learning": {},
            "posthog_cloud": summary,
        }
    )
    assert "(+56 PyPI/sandbox scanners filtered)" in table_str


def test_pro_conversion_friction_fixes_doctor_paywall_and_free_csi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    Verify all 4 PLG conversion-friction fixes:
      1. `check_organ_hippocampus()` reports `pro_locked=True` on Community instead of `WAL active`.
      2. `render_pro_upgrade_card(cmd='memory')` logs `feature='hippocampus_memory'`, shows
         `body upgrade --open` + cohort-attributed checkout URL, and injects local Shield ROI.
      3. Read-only `body context` runs free on Community with a Pro flush teaser, while
         `body context --auto` remains Pro-gated (exit code 2).
    """
    from whole_body_compute.cli_handlers import render_pro_upgrade_card
    from whole_body_compute.doctor import check_organ_hippocampus
    from whole_body_compute.pro_loader import ProUpgradeRequiredError

    wbc_dir = tmp_path / "wbc_comm"
    wbc_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("WBC_DATA_DIR", str(wbc_dir))
    monkeypatch.setenv("WBC_TELEMETRY_CONFIG_PATH", str(wbc_dir / "telemetry_consent.json"))
    monkeypatch.setenv("VOICEFI_LICENSE_PATH", str(wbc_dir / "license.json"))
    monkeypatch.setenv("WBC_FORCE_COMMUNITY", "1")
    monkeypatch.setenv("BODY_TELEMETRY", "0")

    # 1. Honest Hippocampus status in `body doctor` on Community tier
    hippo = check_organ_hippocampus()
    assert hippo["status"] == "pass"
    assert hippo["vital_signs"]["pro_locked"] is True
    assert hippo["vital_signs"]["wal_enabled"] is False
    assert "Pro Tier ($29/mo)" in hippo["details"]

    # 2. Feature-specific, cohort-attributed & personalized `render_pro_upgrade_card`
    consent_mgr = TelemetryConsentManager(config_path=wbc_dir / "telemetry_consent.json")
    cohort_id = consent_mgr.get_cohort_id()

    rc_json = render_pro_upgrade_card("memory", ProUpgradeRequiredError("Pro required"), is_json=True)
    assert rc_json == 2
    card_json = json.loads(capsys.readouterr().out)
    assert card_json["feature"] == "hippocampus_memory"
    assert card_json["open_checkout_command"] == "body upgrade --open"
    assert f"cohort_id={cohort_id}" in card_json["checkout_url"]
    assert "cli_paywall_memory" in card_json["checkout_url"]

    fake_savings = {
        "total_events": 11,
        "cumulative_tokens_saved": 85_300,
        "estimated_usd_saved": 1.42,
    }
    with patch("whole_body_compute.shield.get_savings_summary", return_value=fake_savings):
        rc_txt = render_pro_upgrade_card("memory", ProUpgradeRequiredError("Pro required"), is_json=False)
    assert rc_txt == 2
    card_txt = capsys.readouterr().out
    assert "Your Local Shield Ledger: 11 BBB gates" in card_txt
    assert "Unlock Hippocampus Memory Graph" in card_txt
    assert "body upgrade --open" in card_txt
    assert f"cohort_id={cohort_id}" in card_txt

    # 3. Read-only `body context` runs free on Community with synthetic transcript
    brain_dir = tmp_path / "brain"
    conv_logs = brain_dir / "conv-free-csi-001" / ".system_generated" / "logs"
    conv_logs.mkdir(parents=True, exist_ok=True)
    (conv_logs / "transcript.jsonl").write_text(
        "\n".join(
            json.dumps({"type": "MODEL_TURN", "created_at": f"2026-11-26T12:00:{i:02d}Z"})
            for i in range(10)
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AGY_BRAIN_DIR", str(brain_dir))

    rc_ctx = cli_main(["context", "--json"])
    assert rc_ctx == 0
    ctx_data = json.loads(capsys.readouterr().out)
    assert ctx_data["conv_id"] == "conv-free-csi-001"
    assert ctx_data["pro_flush_locked"] is True
    assert "csi" in ctx_data

    rc_ctx_txt = cli_main(["context"])
    assert rc_ctx_txt == 0
    ctx_banner = capsys.readouterr().out
    assert "Cognitive Thermal Sentry" in ctx_banner
    assert "body flush --auto" in ctx_banner

    # Action mode (`--auto`) remains Pro-gated (exit code 2)
    rc_auto = cli_main(["context", "--auto", "--json"])
    assert rc_auto == 2
    auto_err = json.loads(capsys.readouterr().out)
    assert auto_err["status"] == "upgrade_required"
    assert auto_err["feature"] == "glymphatic_flush"

    # 4. Pure Claude Code workspace (`CLAUDE_CODE_DIR` with no Antigravity transcripts) works in `body context`
    empty_agy = tmp_path / "empty_agy_brain"
    claude_projects = tmp_path / "claude_projects" / "-Users-alice-myrepo"
    claude_projects.mkdir(parents=True, exist_ok=True)
    (claude_projects / "claude-sess-777.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "type": "assistant" if i % 2 else "user",
                    "timestamp": f"2026-11-26T12:10:{i:02d}Z",
                    "message": {
                        "role": "assistant" if i % 2 else "user",
                        "content": [{"type": "tool_use", "name": "Read"}] if i % 2 else "check files",
                    },
                }
            )
            for i in range(12)
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AGY_BRAIN_DIR", str(empty_agy))
    monkeypatch.setenv("CLAUDE_CODE_DIR", str(tmp_path / "claude_projects"))

    rc_claude = cli_main(["context", "--json"])
    assert rc_claude == 0
    claude_ctx = json.loads(capsys.readouterr().out)
    assert claude_ctx["conv_id"] == "claude-sess-777"
    assert claude_ctx["harness"] == "claude_code"
    assert claude_ctx["steps"] == 12
    assert claude_ctx["pro_flush_locked"] is True

    # 5. Outbound telemetry defaults to zero-IP Cloudflare edge relay (`/v1/telemetry`) when POSTHOG_HOST unset
    monkeypatch.delenv("BODY_TELEMETRY", raising=False)
    monkeypatch.delenv("WBC_TELEMETRY_INGEST_URL", raising=False)
    monkeypatch.delenv("POSTHOG_HOST", raising=False)
    sent_urls: list[str] = []

    def _capture_req(req: MagicMock, timeout: float = 2.0) -> MagicMock:
        sent_urls.append(req.full_url)
        resp = MagicMock()
        resp.status = 200
        resp.__enter__.return_value = resp
        return resp

    with patch("urllib.request.urlopen", side_effect=_capture_req):
        send_vitals_rollup({"local_steps": 5}, consent_manager=consent_mgr)
    assert sent_urls == ["https://wholebodycompute.com/v1/telemetry"]








