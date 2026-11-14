"""
Regression tests for Zero-Cursor Background Driving in `body hands` / `FovealVisionEngine`.

Ensures driving background SPAs (such as PostHog in Google Chrome) while working in
Antigravity never hijacks the physical mouse cursor or steals window focus.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from whole_body_compute.cli import main as cli_main
from whole_body_compute.integrations.foveal_vision import (
    FovealUIElement,
    FovealVisionEngine,
    FovealWindowSnapshot,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOICEFI_MOCK_FOVEAL", raising=False)
    monkeypatch.delenv("VOICEFI_HEADLESS", raising=False)
    monkeypatch.delenv("VOICEFI_MOCK_FRONT_APP", raising=False)
    monkeypatch.delenv("WBC_HANDS_NO_CURSOR", raising=False)


def _mock_background_workspace(frontmost_pid: int = 99999, human_idle_sec: float = 0.15):
    """Simulate Antigravity as frontmost app (PID 99999) while human is actively typing."""
    mock_quartz = MagicMock()
    mock_quartz.kCGHIDEventTap = 0
    mock_quartz.kCGMouseButtonLeft = 0
    mock_quartz.kCGMouseButtonRight = 1
    mock_quartz.kCGEventMouseMoved = 5
    mock_quartz.kCGEventRightMouseDown = 3
    mock_quartz.kCGEventRightMouseUp = 4
    mock_quartz.kCGScrollEventUnitLine = 1
    mock_quartz.CGEventSourceSecondsSinceLastEventType.return_value = human_idle_sec

    mock_appkit = MagicMock()
    mock_front = MagicMock()
    mock_front.processIdentifier.return_value = frontmost_pid
    mock_appkit.NSWorkspace.sharedWorkspace.return_value.frontmostApplication.return_value = mock_front
    return mock_quartz, mock_appkit


def _install_hid_mocks(
    monkeypatch: pytest.MonkeyPatch,
    mock_as: MagicMock | None,
    mock_quartz: MagicMock,
    mock_appkit: MagicMock,
) -> None:
    if mock_as is not None:
        monkeypatch.setitem(sys.modules, "ApplicationServices", mock_as)
    monkeypatch.setitem(sys.modules, "Quartz", mock_quartz)
    monkeypatch.setitem(sys.modules, "AppKit", mock_appkit)


def test_actuate_never_escalates_axpress_to_click_xy_when_backgrounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Leak #1 regression: When AXPress succeeds on an async SPA (PostHog) in background Chrome
    and 60ms tactile_delta is still unverified, actuate() must NOT call click_xy / kCGHIDEventTap.
    """
    mock_quartz, mock_appkit = _mock_background_workspace(frontmost_pid=99999, human_idle_sec=5.0)
    mock_as = MagicMock()
    mock_as.AXUIElementPerformAction.return_value = 0  # AXPress succeeds
    mock_as.AXUIElementSetAttributeValue.return_value = 0
    _install_hid_mocks(monkeypatch, mock_as, mock_quartz, mock_appkit)

    btn = FovealUIElement(
        role="AXButton",
        label="Run HogQL Query",
        id="e12",
        bbox=(1050.0, 420.0, 140.0, 34.0),
        raw_element=MagicMock(),
    )
    snap = FovealWindowSnapshot(
        app_name="Google Chrome",
        bundle_id="com.google.Chrome",
        pid=15741,  # Background Chrome != frontmost AGY (99999)
        window_title="SQL Insights - PostHog - Google Chrome",
        window_bbox=(0.0, 0.0, 1440.0, 900.0),
        interactive_elements=[btn],
    )

    # Post-snapshots remain identical (simulating >200ms async PostHog network fetch)
    with patch.object(FovealVisionEngine, "capture_snapshot", return_value=snap):
        res = FovealVisionEngine.actuate(click="e12", snapshot=snap)

        assert res["status"] == "success"
        click_act = res["actions"][0]
        assert click_act["method"] == "AXPress"
        assert click_act["escalated_to_cgevent"] is False
        assert click_act["escalation_blocked_zero_cursor"] is True
        assert not mock_quartz.CGEventCreateMouseEvent.called
        assert not mock_quartz.CGEventPost.called


def test_background_chrome_dom_fallback_when_axpress_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    """When AXPress fails (-25200) on a React div in background Chrome, fall back to AppleScript JS (0 cursor move)."""
    mock_quartz, mock_appkit = _mock_background_workspace(frontmost_pid=99999, human_idle_sec=0.1)
    mock_as = MagicMock()
    mock_as.AXUIElementPerformAction.return_value = -25200  # AXPress unsupported on custom div
    _install_hid_mocks(monkeypatch, mock_as, mock_quartz, mock_appkit)

    div_btn = FovealUIElement(
        role="AXGroup",
        label="Need to set up manually?",
        id="e92",
        bbox=(1062.0, 867.0, 178.0, 33.0),
        raw_element=MagicMock(),
    )
    snap = FovealWindowSnapshot(
        app_name="Google Chrome",
        bundle_id="com.google.Chrome",
        pid=15741,
        window_title="Install - Web Analytics - PostHog - Google Chrome",
        window_bbox=(0.0, 25.0, 1440.0, 875.0),
        interactive_elements=[div_btn],
    )

    with patch("subprocess.run") as mock_subproc:
        mock_subproc.return_value = MagicMock(returncode=0, stdout="clicked_bg_dom\n", stderr="")
        res = FovealVisionEngine.click_element("e92", snapshot=snap)

        assert res["status"] == "success"
        assert res["method"] == "ChromeBackgroundDOM"
        assert res["cursor_moved"] is False
        assert not mock_quartz.CGEventPost.called
        # Verify osascript received stripped PostHog tab title query without calling activate
        osa_args = mock_subproc.call_args[0][0]
        osa_input = mock_subproc.call_args[1]["input"]
        assert osa_args[0] == "osascript"
        assert osa_args[2] == "Google Chrome"
        assert "PostHog" in osa_args[3]
        assert "Google Chrome" not in osa_args[3]
        assert "activate" not in osa_input


def test_background_click_xy_reroutes_to_axpress_without_moving_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Coordinate click (`--click 1151,884`) on background Chrome reroutes to element AXPress instead of kCGHIDEventTap."""
    mock_quartz, mock_appkit = _mock_background_workspace(frontmost_pid=99999, human_idle_sec=0.2)
    mock_as = MagicMock()
    mock_as.AXUIElementPerformAction.return_value = 0
    _install_hid_mocks(monkeypatch, mock_as, mock_quartz, mock_appkit)

    btn = FovealUIElement(
        role="AXButton",
        label="Need to set up manually?",
        id="e92",
        bbox=(1062.0, 867.0, 178.0, 33.0),
        raw_element=MagicMock(),
    )
    snap = FovealWindowSnapshot(
        app_name="Google Chrome",
        bundle_id="com.google.Chrome",
        pid=15741,
        window_title="PostHog",
        window_bbox=(0.0, 25.0, 1440.0, 875.0),
        interactive_elements=[btn],
    )

    res = FovealVisionEngine.click_xy(1151.0, 884.0, pid=15741, snapshot=snap)
    assert res["status"] == "success"
    assert res["method"] == "AXPress"
    assert res["rerouted_from_xy"] is True
    assert res["cursor_moved"] is False
    assert not mock_quartz.CGEventPost.called


def test_scroll_never_fires_global_hid_wheel_when_axscroll_succeeds_or_backgrounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leak #2 regression: scroll() must not scroll Antigravity via kCGHIDEventTap when driving background Chrome."""
    mock_quartz, mock_appkit = _mock_background_workspace(frontmost_pid=99999, human_idle_sec=0.1)
    mock_as = MagicMock()
    mock_as.AXUIElementPerformAction.return_value = 0  # AXScrollDownByPage succeeds
    _install_hid_mocks(monkeypatch, mock_as, mock_quartz, mock_appkit)

    web_area = FovealUIElement(
        role="AXWebArea",
        label="PostHog Dashboard",
        id="e1",
        bbox=(0.0, 80.0, 1440.0, 820.0),
        raw_element=MagicMock(),
    )
    snap = FovealWindowSnapshot(
        app_name="Google Chrome",
        bundle_id="com.google.Chrome",
        pid=15741,
        window_title="PostHog",
        window_bbox=(0.0, 25.0, 1440.0, 875.0),
        interactive_elements=[web_area],
    )

    # 1. Native AXScroll succeeds -> 0 global Quartz scroll wheel events
    res_ax = FovealVisionEngine.scroll("down", amount=5, snapshot=snap)
    assert res_ax["status"] == "success"
    assert res_ax["ax_scroll"] is True
    assert res_ax["quartz_scroll"] is False
    assert not mock_quartz.CGEventCreateScrollWheelEvent.called
    assert not mock_quartz.CGEventPost.called

    # 2. When AXScroll fails (-25200), falls back to ChromeDOM / CGEventPostToPid (still 0 global HID wheel)
    mock_as.AXUIElementPerformAction.return_value = -25200
    with patch("subprocess.run") as mock_subproc:
        mock_subproc.return_value = MagicMock(returncode=0, stdout="scrolled_bg_dom\n", stderr="")
        res_bg = FovealVisionEngine.scroll("down", amount=5, snapshot=snap)
        assert res_bg["status"] == "success"
        assert res_bg["bg_dom_scroll"] is True
        assert res_bg["quartz_scroll"] is False
        assert not mock_quartz.CGEventPost.called


def test_hover_and_right_click_zero_cursor_when_backgrounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leak #3 regression: hover() and right_click() must not warp cursor when target app is in the background."""
    mock_quartz, mock_appkit = _mock_background_workspace(frontmost_pid=99999, human_idle_sec=0.2)
    mock_as = MagicMock()
    mock_as.AXUIElementSetAttributeValue.return_value = 0
    mock_as.AXUIElementPerformAction.return_value = 0  # AXShowMenu succeeds
    _install_hid_mocks(monkeypatch, mock_as, mock_quartz, mock_appkit)

    elem = FovealUIElement(
        role="AXButton",
        label="Funnel Options",
        id="e4",
        bbox=(400.0, 300.0, 120.0, 32.0),
        raw_element=MagicMock(),
    )
    snap = FovealWindowSnapshot(
        app_name="Google Chrome",
        bundle_id="com.google.Chrome",
        pid=15741,
        window_title="PostHog",
        interactive_elements=[elem],
    )

    with patch.object(FovealVisionEngine, "capture_snapshot", return_value=snap):
        with patch("subprocess.run") as mock_subproc:
            mock_subproc.return_value = MagicMock(returncode=0, stdout="hovered_bg_dom\n", stderr="")
            h_res = FovealVisionEngine.hover("e4", snapshot=snap, settle_ms=5)
            assert h_res["status"] == "success"
            assert h_res["cursor_moved"] is False
            assert h_res["method"] == "ChromeBackgroundDOM_hover"
            assert not mock_quartz.CGEventPost.called

        rc_res = FovealVisionEngine.right_click("e4", snapshot=snap, settle_ms=5)
        assert rc_res["status"] == "success"
        assert rc_res["cursor_moved"] is False
        assert rc_res["method"] == "AXShowMenu"
        assert not mock_quartz.CGEventPost.called


def test_cli_no_cursor_flag_locks_zero_cursor_even_when_frontmost(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `body hands --no-cursor` / `--background` passes `no_cursor=True` and blocks cursor clicks even on frontmost apps."""
    mock_quartz, mock_appkit = _mock_background_workspace(frontmost_pid=15741, human_idle_sec=10.0)
    _install_hid_mocks(monkeypatch, None, mock_quartz, mock_appkit)

    # Even with frontmost_pid == target_pid and synthetic raw_element=None, no_cursor=True blocks kCGHIDEventTap
    xy_res = FovealVisionEngine.click_xy(300.0, 400.0, pid=15741, no_cursor=True)
    assert xy_res["status"] == "error"
    assert xy_res["method"] == "CGEvent_xy_blocked_focus_steal"
    assert xy_res["cursor_moved"] is False
    assert not mock_quartz.CGEventPost.called

    mock_snap = FovealWindowSnapshot(
        app_name="Google Chrome",
        bundle_id="com.google.Chrome",
        pid=15741,
        window_title="PostHog",
    )
    with patch("whole_body_compute.integrations.foveal_vision.send_foveal_socket_request", return_value=None):
        with patch.object(FovealVisionEngine, "capture_snapshot", return_value=mock_snap) as mock_cap:
            rc = cli_main(["hands", "--app", "Google Chrome", "--title", "PostHog", "--click", "e1", "--no-cursor"])
            assert rc == 0
            _, kwargs = mock_cap.call_args
            assert kwargs["no_cursor"] is True
