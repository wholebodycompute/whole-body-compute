"""
Unit tests for Organ 10 (Blood Brain Barrier) enhancements:
- Time-window filtering (--window 1h, 2h, 24h, all)
- Tool-level breakdown (--tools)
- Top blocked target analysis (--blocked)
- Shell anti-bloat context shield (run_command pre-gate)
- CLI dispatch and JSON output
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from whole_body_compute.cli import build_parser, cmd_immune, setup_parser
from whole_body_compute.organs.immune import (
    OrganImmuneSystem,
    format_immune_scorecard_table,
    get_context_shield_summary,
    get_immune_pre_post_scorecard,
    inspect_shell_command_bloat,
    parse_window_to_seconds,
    run_ide_pre_tool_hook,
)


def test_parse_window_to_seconds() -> None:
    """Verify human-friendly time window parsing into seconds."""
    assert parse_window_to_seconds("1h") == 3600.0
    assert parse_window_to_seconds("2h") == 7200.0
    assert parse_window_to_seconds("24h") == 86400.0
    assert parse_window_to_seconds("30m") == 1800.0
    assert parse_window_to_seconds("45s") == 45.0
    assert parse_window_to_seconds("7d") == 604800.0
    assert parse_window_to_seconds("1w") == 604800.0
    assert parse_window_to_seconds("2") == 7200.0  # numeric <= 168 defaults to hours
    assert parse_window_to_seconds("all") is None
    assert parse_window_to_seconds("") is None
    assert parse_window_to_seconds("none") is None
    assert parse_window_to_seconds("*") is None
    assert parse_window_to_seconds("invalid_xyz") is None


def test_shell_command_bloat_unbounded_cat(tmp_path: Path) -> None:
    """Verify raw unbounded cat on files >250 lines is blocked."""
    # Create large file (>250 lines)
    large_file = tmp_path / "large_module.py"
    large_file.write_text("\n".join(f"def func_{i}(): pass" for i in range(300)), encoding="utf-8")

    # Create small file (<=250 lines)
    small_file = tmp_path / "small_module.py"
    small_file.write_text("\n".join(f"def func_{i}(): pass" for i in range(20)), encoding="utf-8")

    # cat on large file -> blocked
    res = inspect_shell_command_bloat(f"cat {large_file}")
    assert res["blocked"] is True
    assert res["category"] == "unbounded_dump"
    assert "Context-Tax Gate" in res["reason"]
    assert "whole_body_compute.local.scout" in res["reason"]

    # cat -n on large file -> blocked
    res_n = inspect_shell_command_bloat(f"cat -n {large_file}")
    assert res_n["blocked"] is True

    # cat on small file -> allowed
    res_small = inspect_shell_command_bloat(f"cat {small_file}")
    assert res_small["blocked"] is False


def test_shell_command_bloat_head_tail(tmp_path: Path) -> None:
    """Verify unbounded head/tail (>180 lines) is blocked, but bounded slices pass."""
    large_file = tmp_path / "large_data.py"
    large_file.write_text("\n".join(f"ROW_{i} = {i}" for i in range(400)), encoding="utf-8")

    # head -n 1000 on large file -> blocked
    res_large = inspect_shell_command_bloat(f"head -n 1000 {large_file}")
    assert res_large["blocked"] is True
    assert res_large["category"] == "unbounded_dump"

    # head -n 50 on large file -> bounded slice, allowed
    res_slice = inspect_shell_command_bloat(f"head -n 50 {large_file}")
    assert res_slice["blocked"] is False

    # tail -n 100 on large file -> allowed
    res_tail = inspect_shell_command_bloat(f"tail -n 100 {large_file}")
    assert res_tail["blocked"] is False


def test_shell_command_bloat_directory_crawls() -> None:
    """Verify unbounded recursive crawls (find, tree, ls -R) are blocked unless bounded."""
    # find without -maxdepth -> blocked
    res_find = inspect_shell_command_bloat("find .")
    assert res_find["blocked"] is True
    assert res_find["category"] == "directory_crawl"

    # find with -maxdepth -> allowed
    assert inspect_shell_command_bloat("find . -maxdepth 2 -type f")["blocked"] is False

    # tree without -L -> blocked
    res_tree = inspect_shell_command_bloat("tree")
    assert res_tree["blocked"] is True
    assert res_tree["category"] == "directory_crawl"

    # tree with -L -> allowed
    assert inspect_shell_command_bloat("tree -L 2")["blocked"] is False

    # ls -R -> blocked
    res_ls = inspect_shell_command_bloat("ls -R")
    assert res_ls["blocked"] is True

    res_ls_la = inspect_shell_command_bloat("ls -laR")
    assert res_ls_la["blocked"] is True

    # normal ls -> allowed
    assert inspect_shell_command_bloat("ls -la")["blocked"] is False


def test_shell_command_bloat_pytest() -> None:
    """Verify pytest test floods without quiet flags are blocked with remedy."""
    # pytest without -q -> blocked
    res = inspect_shell_command_bloat("pytest")
    assert res["blocked"] is True
    assert res["category"] == "test_flood"
    assert "-q --tb=short" in res["reason"]

    # pytest with path, without -q -> blocked
    assert inspect_shell_command_bloat(".venv/bin/pytest tests/test_foo.py")["blocked"] is True
    assert inspect_shell_command_bloat("python -m pytest tests/")["blocked"] is True
    assert inspect_shell_command_bloat("uv run pytest tests/")["blocked"] is True

    # pytest with -q -> allowed
    assert inspect_shell_command_bloat("pytest -q")["blocked"] is False
    assert inspect_shell_command_bloat(".venv/bin/pytest tests/test_foo.py -q")["blocked"] is False
    assert inspect_shell_command_bloat("pytest -q --tb=short")["blocked"] is False
    assert inspect_shell_command_bloat("python -m pytest -q")["blocked"] is False

    # help and version -> allowed
    assert inspect_shell_command_bloat("pytest --help")["blocked"] is False
    assert inspect_shell_command_bloat("pytest --version")["blocked"] is False


def test_shell_command_bloat_safe_commands() -> None:
    """Ensure git, builds, scout, heredocs, and pipelines continue to pass cleanly."""
    assert inspect_shell_command_bloat("git status")["blocked"] is False
    assert inspect_shell_command_bloat("git diff")["blocked"] is False
    assert inspect_shell_command_bloat("cargo build")["blocked"] is False
    assert inspect_shell_command_bloat("npm run build")["blocked"] is False
    assert inspect_shell_command_bloat("python -m whole_body_compute.cli doctor")["blocked"] is False
    assert inspect_shell_command_bloat('uv run python -m whole_body_compute.local.scout file.py -q "test"')["blocked"] is False
    assert inspect_shell_command_bloat("cat << 'EOF' > output.py\nprint(1)\nEOF")["blocked"] is False
    assert inspect_shell_command_bloat("cat src/whole_body_compute/cli.py | grep 'def '")["blocked"] is False
    assert inspect_shell_command_bloat("cat src/whole_body_compute/cli.py | wc -l")["blocked"] is False
    assert inspect_shell_command_bloat("cat src/whole_body_compute/cli.py | head -n 50")["blocked"] is False


def test_run_ide_pre_tool_hook_run_command_intercept(tmp_path: Path) -> None:
    """Verify PreToolUse hook intercepts bloated shell commands and allows safe ones."""
    large_file = tmp_path / "huge_script.py"
    large_file.write_text("\n".join(f"# Line {i}" for i in range(500)), encoding="utf-8")

    # 1. Unbounded cat -> Deny
    deny_res = run_ide_pre_tool_hook(
        {
            "toolCall": {
                "name": "run_command",
                "args": {"CommandLine": f"cat {large_file}"},
            },
            "stepIdx": 5,
            "conversationId": "test_conv",
        }
    )
    assert deny_res["decision"] == "deny"
    assert "Context-Tax Gate" in deny_res["reason"]

    # 2. Safe command -> Allow
    allow_res = run_ide_pre_tool_hook(
        {
            "toolCall": {
                "name": "run_command",
                "args": {"CommandLine": "git status"},
            },
            "stepIdx": 5,
            "conversationId": "test_conv",
        }
    )
    assert allow_res["decision"] == "allow"


def test_immune_scorecard_time_window_filtering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify time window filtering (--window 1h, 2h, all) and tool/blocked analytics."""
    audit_file = tmp_path / "test_immune_audit.jsonl"
    monkeypatch.setenv("VOICEFI_IMMUNE_AUDIT_PATH", str(audit_file))

    now = time.time()
    events = [
        # Event 1: 5 hours ago (run_command, allowed)
        {
            "timestamp": now - (5 * 3600),
            "phase": "pre",
            "conversation_id": "c1",
            "step_idx": 1,
            "tool": "run_command",
            "target": "git status",
            "decision": "allow",
            "latency_ms": 1.2,
            "reason": "Organ 10 Pre-Gate passed",
        },
        # Event 2: 3 hours ago (read_file, denied context bloat)
        {
            "timestamp": now - (3 * 3600),
            "phase": "pre",
            "conversation_id": "c1",
            "step_idx": 2,
            "tool": "read_file",
            "target": "giant_log.txt",
            "decision": "deny",
            "context_gate_triggered": True,
            "direct_tokens_saved": 12000,
            "quadratic_tokens_saved": 240000,
            "latency_ms": 2.1,
            "reason": "🛡️ Organ 10 Context-Tax Gate: `giant_log.txt` is 1,200 lines",
        },
        # Event 3: 30 minutes ago (run_command, allowed)
        {
            "timestamp": now - 1800,
            "phase": "pre",
            "conversation_id": "c1",
            "step_idx": 3,
            "tool": "run_command",
            "target": "pytest -q",
            "decision": "allow",
            "latency_ms": 0.5,
            "reason": "Organ 10 Pre-Gate passed",
        },
        # Event 4: 10 minutes ago (run_command, denied pytest flood)
        {
            "timestamp": now - 600,
            "phase": "pre",
            "conversation_id": "c1",
            "step_idx": 4,
            "tool": "run_command",
            "target": "pytest tests/test_foo.py",
            "decision": "deny",
            "context_gate_triggered": True,
            "direct_tokens_saved": 5000,
            "quadratic_tokens_saved": 100000,
            "latency_ms": 1.0,
            "reason": "🛡️ Organ 10 Context-Tax Gate: Verbose test execution without quiet flag blocked.",
        },
    ]

    with open(audit_file, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")

    # Window = 1h: Should capture events 3 and 4 only
    sc_1h = get_immune_pre_post_scorecard(window="1h")
    assert sc_1h["window"] == "1h"
    assert sc_1h["total_pre_checks"] == 2
    assert sc_1h["allowed_count"] == 1
    assert sc_1h["intercepted_count"] == 1
    assert "run_command" in sc_1h["tools_breakdown"]
    tb_rc = sc_1h["tools_breakdown"]["run_command"]
    assert tb_rc["total"] == 2
    assert tb_rc["allowed"] == 1
    assert tb_rc["intercepted"] == 1
    assert tb_rc["allowed_pct"] == 50.0
    assert tb_rc["intercepted_pct"] == 50.0

    assert len(sc_1h["top_blocked"]) == 1
    assert sc_1h["top_blocked"][0]["tool"] == "run_command"
    assert sc_1h["top_blocked"][0]["target"] == "pytest tests/test_foo.py"
    assert sc_1h["top_blocked"][0]["category"] == "shell context bloat"

    # Window = 4h: Should capture events 2, 3, and 4
    sc_4h = get_immune_pre_post_scorecard(window="4h")
    assert sc_4h["total_pre_checks"] == 3
    assert sc_4h["allowed_count"] == 1
    assert sc_4h["intercepted_count"] == 2

    # Window = all: Captures all 4 events
    sc_all = get_immune_pre_post_scorecard(window="all")
    assert sc_all["total_pre_checks"] == 4
    assert sc_all["allowed_count"] == 2
    assert sc_all["intercepted_count"] == 2

    # Verify formatting with --tools and --blocked
    table_output = format_immune_scorecard_table(window="1h", show_tools=True, show_blocked=True)
    assert "Window: 1h" in table_output
    assert "Tool Invocation Breakdown" in table_output
    assert "Top Blocked Files & Commands" in table_output
    assert "pytest tests/test_foo.py" in table_output


def test_cli_immune_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `body immune` CLI parsing and execution with new flags."""
    monkeypatch.setenv("VOICEFI_IMMUNE_AUDIT_PATH", str(tmp_path / "immune_audit_cli.jsonl"))
    parser = setup_parser()

    # Verify options are recognized
    args = parser.parse_args(["immune", "--window", "2h", "--tools", "--blocked"])
    assert args.command == "immune"
    assert args.window == "2h"
    assert args.tools is True
    assert args.blocked is True
    assert args.json is False

    # Execute cmd_immune and ensure return code 0
    ret = cmd_immune(args)
    assert ret == 0

    # Execute with --json
    args_json = parser.parse_args(["immune", "--window", "2h", "--tools", "--blocked", "--json"])
    assert args_json.json is True
    ret_json = cmd_immune(args_json)
    assert ret_json == 0


def test_organ_immune_system_facade() -> None:
    """Verify OrganImmuneSystem facade class provides all required methods."""
    assert hasattr(OrganImmuneSystem, "inspect_shell_command_bloat")
    assert hasattr(OrganImmuneSystem, "evaluate_pre_tool")
    assert hasattr(OrganImmuneSystem, "evaluate_post_tool")
    assert hasattr(OrganImmuneSystem, "get_scorecard")
    assert hasattr(OrganImmuneSystem, "format_scorecard_table")

    sc = OrganImmuneSystem.get_scorecard(limit=5, window="24h")
    assert isinstance(sc, dict)
    assert "total_pre_checks" in sc
    assert "tools_breakdown" in sc
    assert "top_blocked" in sc
    assert "compute_split" in sc

    # Verify OrganImmuneSystem.get_context_shield_summary alias
    sc_summary = OrganImmuneSystem.get_context_shield_summary(window="24h")
    assert sc_summary["compute_split"]["target_formatted"] == "85% Local $0 • 12% Sub • 3% Cloud"


def test_compute_split_fallback_when_zero_ops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Bicameral Compute Split reports honest 0/0/0 with is_live=False (preserving 85/12/3 target) when 0 ops recorded."""
    empty_audit = tmp_path / "empty_audit.jsonl"
    empty_audit.write_text("", encoding="utf-8")
    monkeypatch.setenv("VOICEFI_IMMUNE_AUDIT_PATH", str(empty_audit))

    sc = get_context_shield_summary(window="1h")
    assert "compute_split" in sc
    split = sc["compute_split"]
    assert split["is_live"] is False
    assert split["total_ops"] == 0
    assert split["local_ops"] == 0
    assert split["subagent_ops"] == 0
    assert split["cloud_ops"] == 0
    assert split["local_pct"] == 0
    assert split["subagent_pct"] == 0
    assert split["cloud_pct"] == 0
    assert split["formatted"] == "0% Local • 0% Sub • 0% Cloud (0 ops — 85/12/3 target)"
    assert split["target_formatted"] == "85% Local $0 • 12% Sub • 3% Cloud"


def test_compute_split_dynamic_from_live_telemetry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Bicameral Compute Split dynamically calculates integer percentages summing to 100% from real events."""
    audit_file = tmp_path / "live_audit.jsonl"
    monkeypatch.setenv("VOICEFI_IMMUNE_AUDIT_PATH", str(audit_file))

    now = time.time()
    events = [
        # Op 1: Local - bounded slice read
        {
            "timestamp": now - 100,
            "phase": "pre",
            "tool": "read_file",
            "target": "src/foo.py",
            "bounded_slice_read": True,
            "decision": "allow",
        },
        # Op 2: Local - pytest CLI execution
        {
            "timestamp": now - 80,
            "phase": "pre",
            "tool": "run_command",
            "target": "pytest -q tests/test_foo.py",
            "decision": "allow",
        },
        # Op 3: Local - AST gate blocked context bloat
        {
            "timestamp": now - 60,
            "phase": "pre",
            "tool": "read_file",
            "target": "giant.log",
            "context_gate_triggered": True,
            "decision": "deny",
        },
        # Op 4: Subagent - invoke_subagent
        {
            "timestamp": now - 40,
            "phase": "pre",
            "tool": "invoke_subagent",
            "target": "research_agent",
            "decision": "allow",
        },
        # Op 5: Cloud - write_file direct frontier edit
        {
            "timestamp": now - 20,
            "phase": "pre",
            "tool": "write_file",
            "target": "out.txt",
            "decision": "allow",
        },
    ]

    with open(audit_file, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")

    sc = get_context_shield_summary(window="1h")
    assert "compute_split" in sc
    split = sc["compute_split"]

    assert split["is_live"] is True
    assert split["total_ops"] == 5
    assert split["local_ops"] == 3
    assert split["subagent_ops"] == 1
    assert split["cloud_ops"] == 1

    # 3/5 = 60%, 1/5 = 20%, 1/5 = 20%
    assert split["local_pct"] == 60
    assert split["subagent_pct"] == 20
    assert split["cloud_pct"] == 20
    assert split["local_pct"] + split["subagent_pct"] + split["cloud_pct"] == 100
    assert split["formatted"] == "60% Local $0 • 20% Sub • 20% Cloud"
    assert split["target_formatted"] == "85% Local $0 • 12% Sub • 3% Cloud"


def test_shell_command_bloat_git_diff_flood() -> None:
    """Verify unbounded git log -p and git show are blocked while bounded git commands pass."""
    res_log_p = inspect_shell_command_bloat("git log -p")
    assert res_log_p["blocked"] is True
    assert res_log_p["category"] == "git_diff_flood"
    assert "--stat" in res_log_p["remedy"]

    res_log_n_p = inspect_shell_command_bloat("git log -n 5 -p")
    assert res_log_n_p["blocked"] is True
    assert res_log_n_p["category"] == "git_diff_flood"

    res_show = inspect_shell_command_bloat("git show")
    assert res_show["blocked"] is True
    assert res_show["category"] == "git_diff_flood"
    assert "--stat" in res_show["remedy"]

    res_show_head = inspect_shell_command_bloat("git show HEAD")
    assert res_show_head["blocked"] is True
    assert res_show_head["category"] == "git_diff_flood"

    # Safe and bounded developer git commands must remain allowed
    assert inspect_shell_command_bloat("git status")["blocked"] is False
    assert inspect_shell_command_bloat("git log -n 10 --stat")["blocked"] is False
    assert inspect_shell_command_bloat("git diff")["blocked"] is False
    assert inspect_shell_command_bloat("git diff --stat")["blocked"] is False
    assert inspect_shell_command_bloat("git diff src/whole_body_compute/cli.py")["blocked"] is False
    assert inspect_shell_command_bloat("git show --stat")["blocked"] is False
    assert inspect_shell_command_bloat("git show --name-only")["blocked"] is False
    assert inspect_shell_command_bloat("git show HEAD | head -n 120")["blocked"] is False


def test_run_ide_pre_tool_hook_pytest_auto_heal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify verbose pytest commands are auto-healed with -q --tb=short when VOICEFI_AUTO_HEAL_SHELL=1."""
    monkeypatch.setenv("VOICEFI_AUTO_HEAL_SHELL", "1")
    healed_res = run_ide_pre_tool_hook(
        {
            "toolCall": {
                "name": "run_command",
                "args": {"CommandLine": "uv run pytest tests/test_immune_bbb.py"},
            },
            "stepIdx": 3,
            "conversationId": "test_auto_heal_conv",
        }
    )
    assert healed_res["decision"] == "allow"
    assert "overwrite" in healed_res
    assert (
        healed_res["overwrite"]["CommandLine"]
        == "uv run pytest tests/test_immune_bbb.py -q --tb=short"
    )

    monkeypatch.setenv("VOICEFI_AUTO_HEAL_SHELL", "0")
    denied_res = run_ide_pre_tool_hook(
        {
            "toolCall": {
                "name": "run_command",
                "args": {"CommandLine": "uv run pytest tests/test_immune_bbb.py"},
            },
            "stepIdx": 4,
            "conversationId": "test_auto_heal_conv",
        }
    )
    assert denied_res["decision"] == "deny"
    assert "Context-Tax Gate" in denied_res["reason"]


def test_git_diff_and_show_auto_heal_and_read_auto_slice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify git diff/show bloat auto_heal_cmd, overwrite_args auto-healing, and WBC_AUTO_SLICE_READ."""
    res_diff_ref = inspect_shell_command_bloat("git diff HEAD~2")
    assert res_diff_ref["blocked"] is True
    assert res_diff_ref["category"] == "git_diff_flood"
    assert res_diff_ref["auto_heal_cmd"] == "git diff HEAD~2 --stat"

    res_log_patch = inspect_shell_command_bloat("git log -p")
    assert res_log_patch["blocked"] is True
    assert res_log_patch["auto_heal_cmd"] == "git log -p | head -n 120"

    monkeypatch.setenv("VOICEFI_AUTO_HEAL_SHELL", "1")
    monkeypatch.delenv("WBC_STRICT_SHELL_BLOAT_BLOCK", raising=False)
    hook_git = run_ide_pre_tool_hook(
        {
            "toolCall": {
                "name": "run_command",
                "args": {"CommandLine": "git show HEAD"},
            },
            "stepIdx": 2,
            "conversationId": "test_git_auto_heal",
        },
        audit_log_path=tmp_path / "audit_git.jsonl",
    )
    assert hook_git["decision"] == "allow"
    assert hook_git["overwrite"]["CommandLine"] == "git show HEAD --stat"
    assert hook_git["overwrite_args"]["CommandLine"] == "git show HEAD --stat"

    # Verify WBC_AUTO_SLICE_READ=1 auto-slices >250 line file reads and injects AST map
    large_py = tmp_path / "big_service.py"
    large_py.write_text(
        "\n".join(f"def handler_{i}(x: int) -> int:\n    return x + {i}" for i in range(160)),
        encoding="utf-8",
    )
    monkeypatch.setenv("WBC_AUTO_SLICE_READ", "1")
    slice_res = run_ide_pre_tool_hook(
        {
            "toolCall": {
                "name": "read_file",
                "args": {"Path": str(large_py)},
            },
            "stepIdx": 3,
            "conversationId": "test_auto_slice",
        },
        audit_log_path=tmp_path / "audit_slice.jsonl",
    )
    assert slice_res["decision"] == "allow"
    assert slice_res["overwrite_args"] == {"StartLine": 1, "EndLine": 120}
    assert "handler_0" in slice_res["ephemeralMessage"]


def test_pre_invocation_two_step_subagent_firewall_escalation(tmp_path: Path) -> None:
    """Verify >2 consecutive exploratory tool calls trigger the 2-Step Subagent Firewall nudge."""
    audit_log = tmp_path / "streak_audit.jsonl"
    conv_id = "conv_streak_test"

    # Record 3 consecutive exploratory tool calls
    for idx, cmd in enumerate(["git status", "ls -la", "git diff --stat"], start=1):
        run_ide_pre_tool_hook(
            {
                "toolCall": {"name": "run_command", "args": {"CommandLine": cmd}},
                "stepIdx": idx,
                "conversationId": conv_id,
            },
            audit_log_path=audit_log,
        )

    pre_inv_res = run_ide_pre_tool_hook(
        {
            "phase": "pre_invocation",
            "conversationId": conv_id,
            "initialNumSteps": 4,
        },
        audit_log_path=audit_log,
    )
    assert pre_inv_res["subagent_escalation_nudge"] is True
    assert pre_inv_res["exploratory_streak"] >= 3
    assert "[⚡ 2-STEP SUBAGENT FIREWALL TRIGGERED" in pre_inv_res["ephemeralMessage"]
    assert "invoke_subagent" in pre_inv_res["ephemeralMessage"]
    assert "whole_body_compute.local.scout" in pre_inv_res["ephemeralMessage"]

    # Verify audit log recorded subagent_escalation_nudge=True
    audit_events = [
        json.loads(line)
        for line in audit_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        ev.get("phase") == "pre_invocation" and ev.get("subagent_escalation_nudge") is True
        for ev in audit_events
    )


def test_hybrid_router_warm_first_routing() -> None:
    """Verify HybridRouter prefers warm qwen2.5-coder:1.5b / gemma2:2b over cold endpoints."""
    from unittest.mock import MagicMock
    from whole_body_compute.local.router import HybridRouter

    mock_sup = MagicMock()
    mock_sup.get_telemetry.return_value = MagicMock(
        total_ram_gb=24.0,
        free_ram_gb=8.5,
        available_ram_gb=8.5,
        used_ram_gb=15.5,
        ram_percent=64.5,
        swap_total_gb=4.0,
        swap_used_gb=0.2,
        cpu_percent=15.0,
        load_avg_1m=1.5,
        load_avg_5m=1.4,
        thermal_state="nominal",
        safe_for_70b=False,
        safe_for_32b=False,
        safe_for_7b=True,
        safe_for_2b=True,
        recommended_tier="local_7b_9b_compact",
        active_ollama_models=["qwen2.5-coder:1.5b", "gemma2:2b"],
        advisory="Healthy",
    )
    mock_sup.can_load_model.return_value = (True, "Approved", {})

    router = HybridRouter(supervisor=mock_sup)
    router._mlx_warm_override = False
    router._litert_warm_override = False
    router._warm_ps_cache = ["qwen2.5-coder:1.5b", "gemma2:2b"]
    router._warm_ps_ts = time.time()
    router._installed_cache = [
        "voicefi-coder-mlx",
        "tev1:0.8b",
        "qwen2.5-coder:1.5b",
        "gemma2:2b",
    ]
    router._cache_ts = time.time()

    # 1. get_warm_fast_model returns warm qwen2.5-coder:1.5b in <15ms
    warm_model = router.get_warm_fast_model(task_type="code")
    assert warm_model == "qwen2.5-coder:1.5b"

    # 2. resolve_model for scout / tier0_reflex selects warm qwen2.5-coder:1.5b over cold MLX/LiteRT
    decision_scout = router.resolve_model(task_type="scout")
    assert decision_scout.selected_model == "qwen2.5-coder:1.5b"
    assert decision_scout.is_local is True
    assert decision_scout.cascaded is True
    assert decision_scout.cascaded_from == "voicefi-coder-mlx"
    assert "Warm-first fast-path" in (decision_scout.cascade_reason or "")

    decision_reflex = router.resolve_model(task_type="tier0_reflex")
    assert decision_reflex.selected_model == "qwen2.5-coder:1.5b"
    assert decision_reflex.cascaded_from == "tev1:0.8b"

    # 3. When only gemma2:2b is warm in Ollama ps, selects gemma2:2b
    router._warm_ps_cache = ["gemma2:2b"]
    router._warm_ps_ts = time.time()
    assert router.get_warm_fast_model(task_type="scout") == "gemma2:2b"
    decision_gemma = router.resolve_model(task_type="scout")
    assert decision_gemma.selected_model == "gemma2:2b"


def test_split_shifting_organ_10_upgrades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Verify the 4 Split-Shifting Organ 10 upgrades:
    1. VOICEFI_AUTO_SLICE_OVERWRITE=1 + repeated-slice ReconScout nudge (>2 reads on same file).
    2. 2-Step Exploratory Firewall (streak >= 3) + single-file edit offload (:9379/:9380) + huggingface.co tolerance.
    3. Accurate 3-Layer Bicameral Split attribution (excluding bare 'tool'/'unknown' heartbeats from cloud_ops).
    """
    from whole_body_compute.organs.immune import VascularTrafficMonitor

    # 1. Vascular tolerance for huggingface.co
    assert "huggingface.co" in VascularTrafficMonitor.TRUSTED_HOSTS
    assert VascularTrafficMonitor.is_trusted_host("huggingface.co") is True
    assert (
        VascularTrafficMonitor.inspect_shell_command(
            'curl -sL "https://huggingface.co/api/models/Qwen/Qwen2.5-Coder-7B"'
        )["allowed"]
        is True
    )

    streak_file = tmp_path / "bbb_streak.json"
    audit_file = tmp_path / "bbb_upgrades_audit.jsonl"
    monkeypatch.setenv("VOICEFI_BBB_STREAK_PATH", str(streak_file))
    monkeypatch.setenv("VOICEFI_IMMUNE_AUDIT_PATH", str(audit_file))

    large_py = tmp_path / "engine_core.py"
    large_py.write_text(
        "\n".join(f"def compute_step_{i}(val: int) -> int:\n    return val * {i}" for i in range(150)),
        encoding="utf-8",
    )

    # 2. VOICEFI_AUTO_SLICE_OVERWRITE=1 auto-rewrites StartLine=1, EndLine=120 + AST map in additionalContext
    monkeypatch.setenv("VOICEFI_AUTO_SLICE_OVERWRITE", "1")
    r1 = run_ide_pre_tool_hook(
        {
            "toolCall": {"name": "read_file", "args": {"Path": str(large_py)}},
            "stepIdx": 1,
            "conversationId": "conv_split_shift",
        },
        audit_log_path=audit_file,
    )
    assert r1["decision"] == "allow"
    assert r1["overwrite"] == {"StartLine": 1, "EndLine": 120}
    assert "compute_step_0" in r1["additionalContext"]

    # Read 2 and Read 3 on same file -> triggers Repeated-Slice Guard + 2-Step Exploratory Firewall (streak=3)
    run_ide_pre_tool_hook(
        {
            "toolCall": {
                "name": "read_file",
                "args": {"Path": str(large_py), "StartLine": 121, "EndLine": 240},
            },
            "stepIdx": 2,
            "conversationId": "conv_split_shift",
        },
        audit_log_path=audit_file,
    )
    r3 = run_ide_pre_tool_hook(
        {
            "toolCall": {
                "name": "read_file",
                "args": {"Path": str(large_py), "StartLine": 241, "EndLine": 300},
            },
            "stepIdx": 3,
            "conversationId": "conv_split_shift",
        },
        audit_log_path=audit_file,
    )
    assert "Repeated-Slice Guard" in r3["additionalContext"]
    assert "whole_body_compute.local.scout" in r3["additionalContext"]
    assert "2-Step Exploratory Firewall" in r3["additionalContext"]

    # 3. Single-file repeated edit_file (>1 time on same file) -> :9379 / :9380 offload nudge
    target_py = tmp_path / "small_helper.py"
    target_py.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    run_ide_pre_tool_hook(
        {
            "toolCall": {
                "name": "edit_file",
                "args": {
                    "Path": str(target_py),
                    "Target": "return a + b",
                    "Replacement": "return a + b + 0",
                },
            },
            "stepIdx": 4,
            "conversationId": "conv_split_shift",
        },
        audit_log_path=audit_file,
    )
    e2 = run_ide_pre_tool_hook(
        {
            "toolCall": {
                "name": "edit_file",
                "args": {
                    "Path": str(target_py),
                    "Target": "return a + b + 0",
                    "Replacement": "return a + b + 1",
                },
            },
            "stepIdx": 5,
            "conversationId": "conv_split_shift",
        },
        audit_log_path=audit_file,
    )
    assert "Local Metal Offload" in e2["additionalContext"]
    assert ":9379" in e2["additionalContext"]
    assert ":9380" in e2["additionalContext"]

    # 4. Accurate 3-Layer Bicameral Split attribution (ignoring bare 'tool'/'unknown' heartbeats)
    split_audit = tmp_path / "split_attribution.jsonl"
    monkeypatch.setenv("VOICEFI_IMMUNE_AUDIT_PATH", str(split_audit))
    now = time.time()
    raw_events = [
        {"timestamp": now - 50, "phase": "pre", "tool": "tool", "target": "fast-path local op", "decision": "allow"},
        {"timestamp": now - 49, "phase": "pre", "tool": "unknown", "target": "", "decision": "allow"},
        {
            "timestamp": now - 40,
            "phase": "pre",
            "tool": "run_command",
            "target": "uv run python scripts/daily_efficiency_audit.py",
            "decision": "allow",
        },
        {
            "timestamp": now - 30,
            "phase": "pre",
            "tool": "read_file",
            "target": "src/foo.py",
            "event": "pre_tool_slice_allowed",
            "decision": "allow",
        },
        {
            "timestamp": now - 20,
            "phase": "pre",
            "tool": "send_message",
            "target": "subagent_1",
            "decision": "allow",
        },
        {
            "timestamp": now - 10,
            "phase": "pre",
            "tool": "edit_file",
            "target": "src/bar.py",
            "decision": "allow",
        },
    ]
    split_audit.write_text("\n".join(json.dumps(x) for x in raw_events) + "\n", encoding="utf-8")
    sc = get_immune_pre_post_scorecard(window="1h")
    sp = sc["compute_split"]
    assert sp["local_ops"] == 2
    assert sp["subagent_ops"] == 1
    assert sp["cloud_ops"] == 1
    assert sp["total_ops"] == 4


def test_truncation_cascade_clamps_read_file_span(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that when read_file is an active truncation offender, read spans > 80 lines are clamped."""
    # Create a 200-line test file
    test_file = tmp_path / "service_200.py"
    test_file.write_text(
        "\n".join(f"def func_{i}():\n    return {i}" for i in range(100)),
        encoding="utf-8",
    )
    assert (test_file.read_text().count("\n") + 1) == 200

    # 1. Under nominal conditions (not flagged), a 140-line read is allowed (< 160/180 max allowed span)
    monkeypatch.delenv("WBC_TRUNCATION_OFFENDERS_JSON", raising=False)
    test_db = tmp_path / "dot_memory_clean.db"
    monkeypatch.setenv("WBC_DOT_MEMORY_PATH", str(test_db))
    nom_res = run_ide_pre_tool_hook(
        {
            "toolCall": {
                "name": "read_file",
                "args": {"Path": str(test_file), "StartLine": 1, "EndLine": 140},
            },
            "stepIdx": 3,
            "conversationId": "test_nominal_read",
        },
        audit_log_path=tmp_path / "audit_nom.jsonl",
    )
    assert nom_res["decision"] == "allow"
    assert nom_res.get("truncation_cascade_clamp") is not True

    # 2. When read_file is flagged in WBC_TRUNCATION_OFFENDERS_JSON, the 140-line read is clamped (<= 80 lines)
    monkeypatch.setenv(
        "WBC_TRUNCATION_OFFENDERS_JSON",
        json.dumps({
            "read_file": {"truncated": 5, "est_kb": 250.0, "primary_cause": "TRUNCATION_CASCADE"}
        }),
    )
    clamped_res = run_ide_pre_tool_hook(
        {
            "toolCall": {
                "name": "read_file",
                "args": {"Path": str(test_file), "StartLine": 1, "EndLine": 140},
            },
            "stepIdx": 4,
            "conversationId": "test_clamped_read",
        },
        audit_log_path=tmp_path / "audit_clamp.jsonl",
    )
    assert clamped_res["decision"] == "deny"
    assert "🛡️ Blood-Brain Barrier Truncation-Cascade Auto-Clamp" in clamped_res["reason"]
    assert "<= 80" in clamped_res["reason"]
    assert clamped_res.get("truncation_cascade_clamp") is True

    # 3. Reading within <= 80 lines remains allowed even under clamp
    valid_res = run_ide_pre_tool_hook(
        {
            "toolCall": {
                "name": "read_file",
                "args": {"Path": str(test_file), "StartLine": 1, "EndLine": 80},
            },
            "stepIdx": 5,
            "conversationId": "test_valid_clamp_read",
        },
        audit_log_path=tmp_path / "audit_valid.jsonl",
    )
    assert valid_res["decision"] == "allow"

    # 4. Also verify SQLite-based triggering from cognitive_telemetry_events
    monkeypatch.delenv("WBC_TRUNCATION_OFFENDERS_JSON", raising=False)
    import sqlite3

    db_path = tmp_path / "dot_memory_live.db"
    monkeypatch.setenv("WBC_DOT_MEMORY_PATH", str(db_path))
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE cognitive_telemetry_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT,
                primary_cause TEXT,
                top_bloat_tools_json TEXT,
                created_at REAL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO cognitive_telemetry_events (event_type, primary_cause, top_bloat_tools_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                "GLYMPHATIC_FLUSH",
                "TRUNCATION_CASCADE",
                json.dumps({"read_file": {"calls": 40, "truncated": 15, "est_kb": 220.0}}),
                time.time(),
            ),
        )

    sqlite_clamped = run_ide_pre_tool_hook(
        {
            "toolCall": {
                "name": "read_file",
                "args": {"Path": str(test_file), "StartLine": 1, "EndLine": 140},
            },
            "stepIdx": 6,
            "conversationId": "test_sqlite_clamped",
        },
        audit_log_path=tmp_path / "audit_sqlite.jsonl",
    )
    assert sqlite_clamped["decision"] == "deny"
    assert "🛡️ Blood-Brain Barrier Truncation-Cascade Auto-Clamp" in sqlite_clamped["reason"]
    assert "<= 80" in sqlite_clamped["reason"]


def test_truncation_cascade_auto_heals_run_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify when run_command is flagged, unbounded git diff or rg commands are auto-healed with | head -n 60."""
    monkeypatch.setenv(
        "WBC_TRUNCATION_OFFENDERS_JSON",
        json.dumps({
            "run_command": {"truncated": 6, "est_kb": 180.0, "primary_cause": "TRUNCATION_CASCADE"}
        }),
    )

    # 1. Unbounded git diff -> auto-heals with | head -n 60
    diff_audit = tmp_path / "audit_diff.jsonl"
    res_diff = run_ide_pre_tool_hook(
        {
            "toolCall": {
                "name": "run_command",
                "args": {"CommandLine": "git diff"},
            },
            "stepIdx": 2,
            "conversationId": "test_cmd_clamp_conv",
        },
        audit_log_path=diff_audit,
    )
    assert res_diff["decision"] == "allow"
    assert res_diff["overwrite"]["CommandLine"] == "git diff | head -n 60"
    assert res_diff["overwrite_args"]["CommandLine"] == "git diff | head -n 60"
    assert res_diff["updatedInput"]["CommandLine"] == "git diff | head -n 60"
    assert res_diff.get("truncation_cascade_clamp") is True

    # Verify audit event recorded auto_healed and truncation_cascade_clamp
    audit_lines = [json.loads(line) for line in diff_audit.read_text(encoding="utf-8").strip().splitlines()]
    assert any(e.get("auto_healed") and e.get("truncation_cascade_clamp") for e in audit_lines)

    # 2. Unbounded rg search -> auto-heals with | head -n 60
    res_rg = run_ide_pre_tool_hook(
        {
            "toolCall": {
                "name": "run_command",
                "args": {"CommandLine": "rg 'class Organ'"},
            },
            "stepIdx": 3,
            "conversationId": "test_cmd_clamp_conv",
        },
        audit_log_path=tmp_path / "audit_rg.jsonl",
    )
    assert res_rg["decision"] == "allow"
    assert res_rg["overwrite_args"]["CommandLine"] == "rg 'class Organ' | head -n 60"
    assert res_rg["updatedInput"]["CommandLine"] == "rg 'class Organ' | head -n 60"

    # 3. High-output pytest -> auto-heals with -q --tb=short
    res_pytest = run_ide_pre_tool_hook(
        {
            "toolCall": {
                "name": "run_command",
                "args": {"CommandLine": "uv run pytest tests/test_immune_bbb.py"},
            },
            "stepIdx": 4,
            "conversationId": "test_cmd_clamp_conv",
        },
        audit_log_path=tmp_path / "audit_pytest.jsonl",
    )
    assert res_pytest["decision"] == "allow"
    assert res_pytest["overwrite_args"]["CommandLine"] == "uv run pytest tests/test_immune_bbb.py -q --tb=short"

    # 4. Already bounded commands (e.g. piped to head) are NOT modified
    res_bounded = run_ide_pre_tool_hook(
        {
            "toolCall": {
                "name": "run_command",
                "args": {"CommandLine": "git diff | head -n 30"},
            },
            "stepIdx": 5,
            "conversationId": "test_cmd_clamp_conv",
        },
        audit_log_path=tmp_path / "audit_bounded.jsonl",
    )
    assert res_bounded["decision"] == "allow"
    assert "overwrite_args" not in res_bounded or "CommandLine" not in res_bounded["overwrite_args"]




