"""
Pal Teach Mode Event Observer & Demonstration Synthesizer for VoiceFi.
Unified, passive, thread-safe macOS event capture with in-process AXUIElement resolution.
Captures user desktop demonstrations into clean, declarative PalWorkflow recipes.
"""

import json
import math
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml
from AppKit import NSWorkspace
from ApplicationServices import (
    AXUIElementCopyAttributeValue,
    AXUIElementCopyElementAtPosition,
    AXUIElementCreateApplication,
    AXUIElementCreateSystemWide,
)
import Quartz

from whole_body_compute.integrations.pal_workflow import (
    ActionType,
    ClickTarget,
    PalWorkflowRecipe,
    PalWorkflowRegistry,
    ParameterDefinition,
    WorkflowStep,
)

# Standard Carbon virtual keycodes for macOS hotkey normalization
VK_MAP = {
    36: "Return",
    48: "Tab",
    49: "Space",
    51: "Backspace",
    53: "Escape",
    115: "Home",
    117: "Delete",
    119: "End",
    123: "Left",
    124: "Right",
    125: "Down",
    126: "Up",
    0: "A",
    1: "S",
    2: "D",
    3: "F",
    4: "H",
    5: "G",
    6: "Z",
    7: "X",
    8: "C",
    9: "V",
    11: "B",
    12: "Q",
    13: "W",
    14: "E",
    15: "R",
    16: "Y",
    17: "T",
    31: "O",
    32: "U",
    34: "I",
    35: "P",
    37: "L",
    38: "J",
    40: "K",
    45: "N",
    46: "M",
    43: ",",
    47: ".",
    44: "/",
    24: "=",
    27: "-",
}


@dataclass
class PalTeachRawStep:
    step_number: int
    action: str  # "click", "double_click", "type_text", "hotkey", "press_key", "drag", "switch_app"
    app_name: str
    window_title: Optional[str]
    params: Dict[str, Any]
    spoken_summary: str
    voice_note: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    frame_path: Optional[str] = None
    grounding_prompt: Optional[str] = None


class PalTeachRecorder:
    """
    Passive macOS event observer that captures and denoises user demonstrations
    into clean discrete steps for Pal and VoiceFi ReAct loops.
    """

    def __init__(self, on_step: Optional[Callable[[PalTeachRawStep], None]] = None):
        self.on_step = on_step
        self.steps: List[PalTeachRawStep] = []
        self._running = False
        self._tap = None
        self._loop = None
        self._loop_source = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Denoising & State Machine
        self._current_app = ""
        self._current_window: Optional[str] = None
        self._current_window_origin: Optional[Tuple[float, float]] = None
        self._pending_text: List[str] = []
        self._mouse_down_pos: Optional[Tuple[float, float]] = None
        self._flush_timer: Optional[threading.Timer] = None
        self._last_voice_note: Optional[str] = None
        self.voice_notes: List[str] = []
        self._ignore_rects: List[Tuple[float, float, float, float]] = []

        # Auth-Pause & Privacy Boundary Guard (Patent Claim 7)
        self._auth_paused: bool = False
        self._auth_pause_reason: Optional[str] = None

    AUTH_PAUSE_PATTERNS = (
        re.compile(
            r"\b(?:pause\s+(?:for\s+)?auth(?:entication)?|entering\s+(?:credentials?|passwords?|2fa|otp|tokens?)|pause\s+recording)\b",
            re.I,
        ),
    )
    AUTH_RESUME_PATTERNS = (
        re.compile(
            r"\b(?:authenticated|resume\s+recording|auth(?:entication)?\s+complete|logged\s+in|finished\s+authenticating)\b",
            re.I,
        ),
    )

    def pause_auth(self, reason: str = "manual") -> None:
        """Explicitly pause event capture and scrub pending keystrokes/OCR for authentication/MFA."""
        with self._lock:
            self._auth_paused = True
            self._auth_pause_reason = reason
            self._pending_text.clear()

    def resume_auth(self) -> None:
        """Resume event capture after authentication has completed."""
        with self._lock:
            self._auth_paused = False
            self._auth_pause_reason = None

    def is_auth_paused(self) -> bool:
        with self._lock:
            return self._auth_paused

    def set_ignore_rect(
        self, x: float, y: float, width: float, height: float
    ) -> None:
        """Register a screen bounding box (e.g. Granola Training Card) where clicks should not be recorded as steps."""
        with self._lock:
            self._ignore_rects = [(float(x), float(y), float(width), float(height))]

    def clear_ignore_rects(self) -> None:
        with self._lock:
            self._ignore_rects.clear()

    def _is_point_ignored(self, x: float, y: float) -> bool:
        for rx, ry, rw, rh in self._ignore_rects:
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                return True
        return False

    def remove_step(self, step_number: int) -> List[PalTeachRawStep]:
        """Remove a step by its 1-based step_number and renumber remaining steps."""
        with self._lock:
            self.steps = [s for s in self.steps if s.step_number != step_number]
            for idx, s in enumerate(self.steps, start=1):
                s.step_number = idx
            return list(self.steps)

    def start(self) -> None:
        """Start passive background event tap."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_tap, daemon=True, name="PalTeachRecorder")
        self._thread.start()

    def stop(self) -> List[PalTeachRawStep]:
        """Stop observer and return consolidated steps."""
        self._running = False
        with self._lock:
            self._flush_text_buffer()
        if self._tap is not None:
            try:
                Quartz.CGEventTapEnable(self._tap, False)
                Quartz.CFMachPortInvalidate(self._tap)
            except Exception:
                pass
        if self._loop is not None:
            try:
                Quartz.CFRunLoopStop(self._loop)
            except Exception:
                pass
        return list(self.steps)

    def attach_voice_note(self, note: str) -> None:
        """Attach active speech note from Whisper stream to next or most recent step and sync to episodic audio store."""
        clean_note = (note or "").strip()
        if not clean_note:
            return

        # Acoustic Auth-Pause Boundary Gate (Patent Claim 7)
        if any(pat.search(clean_note) for pat in self.AUTH_PAUSE_PATTERNS):
            self.pause_auth(reason=f"voice: {clean_note}")
            with self._lock:
                self.voice_notes.append("[AUTH_PAUSE_INVOKED]")
            return
        if any(pat.search(clean_note) for pat in self.AUTH_RESUME_PATTERNS):
            self.resume_auth()
            with self._lock:
                self.voice_notes.append("[AUTH_RESUME_INVOKED]")
            return

        try:
            from whole_body_compute.integrations.screenpipe import record_audio_transcription

            record_audio_transcription(clean_note, device="Pal Training Mic")
        except Exception:
            pass
        with self._lock:
            self.voice_notes.append(clean_note)
            if self._last_voice_note:
                self._last_voice_note = f"{self._last_voice_note} | {clean_note}"
            else:
                self._last_voice_note = clean_note
            if self.steps:
                prev = self.steps[-1].voice_note
                self.steps[-1].voice_note = f"{prev} | {clean_note}" if prev else clean_note

    def attach_network_event(
        self,
        method: str,
        url: str,
        body: Any = None,
        max_delta_s: float = 0.5,
    ) -> bool:
        """
        Attach a contemporaneous outbound HTTP/XHR/Fetch network event (CDP / HAR) to the
        most recent UI step if it occurred within `max_delta_s` (default 500ms, Patent Claim 11).
        """
        with self._lock:
            if self._auth_paused or not self.steps:
                return False
            last_step = self.steps[-1]
            delta = abs(time.time() - float(getattr(last_step, "timestamp", 0.0) or 0.0))
            if delta > max_delta_s:
                return False
            if isinstance(last_step.params, dict):
                last_step.params["captured_xhr"] = {
                    "method": str(method or "POST").upper(),
                    "url": str(url or "").strip(),
                    "body": body if body is not None else {},
                    "delta_ms": round(delta * 1000.0, 1),
                }
                return True
            return False

    def _capture_browser_network_entries(self, app_name: str) -> List[Dict[str, Any]]:
        """
        Passively query the active browser tab's PerformanceResourceTiming buffer for recent
        outbound XHR/Fetch calls (<35ms, zero DevTools required) when recording browser steps.
        """
        from whole_body_compute.integrations.pal_harness import PalHarness

        if self._auth_paused or PalHarness._is_test_env():
            return []
        if app_name not in ("Google Chrome", "Brave Browser", "Arc", "Safari"):
            return []
        js_probe = (
            "JSON.stringify((window.performance && window.performance.getEntriesByType"
            " ? window.performance.getEntriesByType('resource')"
            ".filter(function(e){return e.initiatorType==='fetch'||e.initiatorType==='xmlhttprequest';})"
            ".slice(-15).map(function(e){return {method:'POST',url:e.name,initiatorType:e.initiatorType};})"
            " : []))"
        )
        if app_name == "Safari":
            script = f'tell application "Safari" to do JavaScript "{js_probe}" in current tab of front window'
        else:
            script = f'tell application "{app_name}" to execute active tab of front window javascript "{js_probe}"'
        try:
            ok, out = PalHarness.run_applescript(script)
            if ok and out and out.strip().startswith("["):
                import json

                parsed = json.loads(out.strip())
                if isinstance(parsed, list):
                    return [e for e in parsed if isinstance(e, dict) and e.get("url")]
        except Exception:
            pass
        return []

    def _emit_step(self, action: str, params: Dict[str, Any], spoken: str) -> None:
        if self._auth_paused:
            return
        from whole_body_compute.integrations.pal_harness import PalHarness

        enriched_params = dict(params or {})
        is_secure = self._is_focused_secure_field()
        if is_secure:
            if "screenpipe_ocr" in enriched_params:
                del enriched_params["screenpipe_ocr"]
            if "text" in enriched_params:
                enriched_params["text"] = "[REDACTED_SECURE_FIELD]"
        else:
            if (
                action in ("click", "double_click", "press_key", "hotkey")
                and "captured_xhr" not in enriched_params
                and "har_entries" not in enriched_params
            ):
                net_entries = self._capture_browser_network_entries(self._current_app or "")
                if net_entries:
                    enriched_params["har_entries"] = net_entries
            if "screenpipe_ocr" not in enriched_params and not PalHarness._is_test_env():
                try:
                    from whole_body_compute.integrations.screenpipe import search_screen_content

                    recs = search_screen_content(
                        query="",
                        content_type="ocr",
                        app_name=self._current_app or None,
                        minutes_ago=2,
                        limit=1,
                        capture_live_if_empty=(action in ("select_text", "switch_app")),
                    )
                    if recs and recs[0].text:
                        enriched_params["screenpipe_ocr"] = recs[0].text[:1500]
                except Exception:
                    pass

        # Visual Frame & Grounding Capsule for Codified Workflows
        frame_path = None
        grounding_prompt = None
        if not self._auth_paused:
            try:
                from whole_body_compute.integrations.screen_bridge import get_screen_bridge

                bridge = get_screen_bridge()
                capsule = bridge.get_latest_frame(fresh=False, max_age_s=2.0)
                if capsule and not capsule.redacted_reason and capsule.image_path:
                    frame_path = capsule.image_path
                    grounding_prompt = capsule.grounding_prompt
                    enriched_params["frame_path"] = frame_path
                    enriched_params["grounding_prompt"] = grounding_prompt
            except Exception:
                pass

        step = PalTeachRawStep(
            step_number=len(self.steps) + 1,
            action=action,
            app_name=self._current_app,
            window_title=self._current_window,
            params=enriched_params,
            spoken_summary=spoken,
            voice_note=self._last_voice_note,
            frame_path=frame_path,
            grounding_prompt=grounding_prompt,
        )
        self._last_voice_note = None
        self.steps.append(step)
        if self.on_step:
            try:
                self.on_step(step)
            except Exception as e:
                print(f"[PalTeachRecorder] on_step error: {e}")

    def _refresh_active_context(self) -> None:
        """Query frontmost app and window title directly from Quartz WindowServer (<0.2ms, no NSRunLoop lag)."""
        app_name = ""
        window_title: Optional[str] = None
        bundle_id = ""
        pid = 0

        try:
            wins = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly
                | Quartz.kCGWindowListExcludeDesktopElements,
                Quartz.kCGNullWindowID,
            )
            if wins:
                for w in wins:
                    if w.get("kCGWindowLayer", 0) == 0:
                        owner = str(w.get("kCGWindowOwnerName") or "").strip()
                        if owner and owner not in ("Window Server", "SystemUIServer", "Control Center"):
                            app_name = owner
                            w_name = str(w.get("kCGWindowName") or "").strip()
                            if w_name:
                                window_title = w_name
                            pid = int(w.get("kCGWindowOwnerPID") or 0)
                            bounds = w.get("kCGWindowBounds") or {}
                            if "X" in bounds and "Y" in bounds:
                                self._current_window_origin = (
                                    float(bounds["X"]),
                                    float(bounds["Y"]),
                                )
                            break
        except Exception:
            pass

        if not app_name:
            ws = NSWorkspace.sharedWorkspace()
            front = ws.frontmostApplication()
            if not front:
                return
            app_name = front.localizedName() or ""
            pid = front.processIdentifier()
            bundle_id = front.bundleIdentifier() or ""

        if pid and not window_title:
            try:
                app_ax = AXUIElementCreateApplication(pid)
                err, win = AXUIElementCopyAttributeValue(app_ax, "AXFocusedWindow", None)
                if err == 0 and win:
                    err, title = AXUIElementCopyAttributeValue(win, "AXTitle", None)
                    if err == 0 and title:
                        window_title = str(title)
            except Exception:
                pass

        if app_name != self._current_app:
            self._flush_text_buffer()
            self._current_app = app_name
            self._current_window = window_title
            self._emit_step(
                action="switch_app",
                params={"app_name": app_name, "bundle_id": bundle_id},
                spoken=f"Switched to {app_name}",
            )
        else:
            self._current_window = window_title

    def _flush_text_buffer(self) -> None:
        if not self._pending_text:
            return
        text = "".join(self._pending_text)
        self._pending_text.clear()
        if text.strip() or "\n" in text:
            self._emit_step(
                action="type_text",
                params={"text": text},
                spoken=f"Typed '{text.strip()}'",
            )

    def _schedule_flush(self) -> None:
        if self._flush_timer:
            self._flush_timer.cancel()
        self._flush_timer = threading.Timer(0.8, self._flush_text_buffer)
        self._flush_timer.start()

    def _inspect_element_at(self, x: float, y: float) -> Dict[str, Any]:
        """
        Resolve UI role, title, WorkArena DOM selector/ARIA attributes, and OSWorld
        window-relative anchor offsets at coordinates using system-wide AXUIElement.
        """
        info: Dict[str, Any] = {}
        if self._current_window_origin is not None:
            info["window_anchor_dx"] = round(float(x) - self._current_window_origin[0], 1)
            info["window_anchor_dy"] = round(float(y) - self._current_window_origin[1], 1)
        try:
            sys_elem = AXUIElementCreateSystemWide()
            err, elem = AXUIElementCopyElementAtPosition(sys_elem, x, y, None)
            if err == 0 and elem:
                _, role = AXUIElementCopyAttributeValue(elem, "AXRole", None)
                _, title = AXUIElementCopyAttributeValue(elem, "AXTitle", None)
                info["role"] = str(role or "")
                info["title"] = str(title or "")

                # Capture browser DOM / ARIA attributes when exposed by Chromium / WebKit AXTree
                _, dom_id = AXUIElementCopyAttributeValue(elem, "AXDOMIdentifier", None)
                _, desc = AXUIElementCopyAttributeValue(elem, "AXDescription", None)
                dom_id_str = str(dom_id or "").strip()
                desc_str = str(desc or "").strip()
                if dom_id_str:
                    info["dom_id"] = dom_id_str
                    info["dom_selector"] = f"#{dom_id_str}"
                elif desc_str:
                    info["aria_label"] = desc_str
                    safe_desc = desc_str.replace('"', '\\"')
                    info["dom_selector"] = f'[aria-label="{safe_desc}"]'
                return info
        except Exception:
            pass
        return info

    def _inspect_focused_selection(self) -> Optional[str]:
        """
        Query AXSelectedText from the focused AXUIElement in the frontmost application.
        Eliminates the 'highlighted without copying' blind spot during demonstrations.
        """
        try:
            ws = NSWorkspace.sharedWorkspace()
            front = ws.frontmostApplication()
            if not front:
                return None
            app_ax = AXUIElementCreateApplication(front.processIdentifier())
            err, focused = AXUIElementCopyAttributeValue(app_ax, "AXFocusedUIElement", None)
            if err == 0 and focused:
                err, sel = AXUIElementCopyAttributeValue(focused, "AXSelectedText", None)
                if err == 0 and sel:
                    sel_str = str(sel).strip()
                    if sel_str:
                        return sel_str
        except Exception:
            pass
        return None

    def _is_focused_secure_field(self) -> bool:
        """Query if the focused element in frontmost app is an AXSecureTextField or password input."""
        try:
            ws = NSWorkspace.sharedWorkspace()
            front = ws.frontmostApplication()
            if not front:
                return False
            app_ax = AXUIElementCreateApplication(front.processIdentifier())
            err, focused = AXUIElementCopyAttributeValue(app_ax, "AXFocusedUIElement", None)
            if err == 0 and focused:
                _, role = AXUIElementCopyAttributeValue(focused, "AXRole", None)
                _, subrole = AXUIElementCopyAttributeValue(focused, "AXSubrole", None)
                r = str(role or "").lower()
                sr = str(subrole or "").lower()
                if "secure" in r or "secure" in sr or "password" in r or "password" in sr:
                    return True
        except Exception:
            pass
        return False

    def _event_callback(self, proxy, event_type, event, refcon):
        with self._lock:
            # Privacy & Auth-Pause Gate: completely drop all event capture when auth paused
            if self._auth_paused:
                return event

            self._refresh_active_context()

            # 1. Keyboard Events
            if event_type == Quartz.kCGEventKeyDown:
                # Hardware/AX Privacy Guard: drop key logging if user is focused on a secure text field
                if self._is_focused_secure_field():
                    self._pending_text.clear()
                    return event
                flags = Quartz.CGEventGetFlags(event)
                keycode = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
                is_cmd = bool(flags & Quartz.kCGEventFlagMaskCommand)
                is_ctrl = bool(flags & Quartz.kCGEventFlagMaskControl)
                is_alt = bool(flags & Quartz.kCGEventFlagMaskAlternate)
                is_shift = bool(flags & Quartz.kCGEventFlagMaskShift)

                # Hotkey check (Cmd, Ctrl, or Option/Shift paired with navigation/deletion keys)
                nav_or_del_keys = (36, 48, 51, 53, 115, 117, 119, 123, 124, 125, 126)
                if (
                    is_cmd
                    or is_ctrl
                    or (is_alt and keycode in nav_or_del_keys)
                    or (is_shift and keycode in (48, 115, 119, 123, 124, 125, 126))
                ):
                    self._flush_text_buffer()
                    key_name = VK_MAP.get(keycode, f"Key_{keycode}")
                    mods = []
                    if is_cmd:
                        mods.append("Cmd")
                    if is_ctrl:
                        mods.append("Ctrl")
                    if is_alt:
                        mods.append("Option")
                    if is_shift:
                        mods.append("Shift")
                    combo = "+".join(mods + [key_name])
                    hotkey_params: Dict[str, Any] = {
                        "combo": combo,
                        "keycode": keycode,
                        "key": key_name,
                        "modifiers": mods,
                    }
                    if combo in ("Cmd+C", "Cmd+V"):
                        try:
                            from AppKit import NSPasteboard, NSPasteboardTypeString

                            pb = NSPasteboard.generalPasteboard()
                            clip_str = pb.stringForType_(NSPasteboardTypeString)
                            if clip_str:
                                hotkey_params["clipboard_text"] = str(clip_str)[:500]
                        except Exception:
                            pass
                    self._emit_step(
                        action="hotkey",
                        params=hotkey_params,
                        spoken=f"Pressed {combo}",
                    )
                    return event

                # Navigation & Forward Delete keys (Return, Tab, Escape, Home, Delete, End, Left, Right, Down, Up)
                if keycode in (36, 48, 53, 115, 117, 119, 123, 124, 125, 126):
                    self._flush_text_buffer()
                    key_name = VK_MAP.get(keycode, "Key")
                    self._emit_step(
                        action="press_key",
                        params={"key": key_name, "keycode": keycode},
                        spoken=f"Pressed {key_name}",
                    )
                    return event

                if keycode == 51:  # Backspace
                    if self._pending_text:
                        self._pending_text.pop()
                        self._schedule_flush()
                    else:
                        # Editing pre-filled text (e.g., stripping count suffix in Google Drive rename box)
                        self._emit_step(
                            action="press_key",
                            params={"key": "Backspace", "keycode": 51},
                            spoken="Pressed Backspace",
                        )
                    return event

                # Safe Unicode extraction (no Carbon TIS asserts on macOS 15)
                _, chars = Quartz.CGEventKeyboardGetUnicodeString(event, 4, None, None)
                if chars:
                    self._pending_text.append(chars)
                    self._schedule_flush()

            # 2. Mouse Events
            elif event_type == Quartz.kCGEventRightMouseDown:
                loc = Quartz.CGEventGetLocation(event)
                if self._is_point_ignored(loc.x, loc.y):
                    return event
                self._flush_text_buffer()
                elem_info = self._inspect_element_at(loc.x, loc.y)
                label = elem_info.get("title") or elem_info.get("role") or "screen"
                self._emit_step(
                    action="right_click",
                    params={
                        "x": round(loc.x),
                        "y": round(loc.y),
                        "click_count": 1,
                        "element": elem_info,
                    },
                    spoken=f"Right-clicked {label}",
                )

            elif event_type == Quartz.kCGEventLeftMouseDown:
                loc = Quartz.CGEventGetLocation(event)
                if self._is_point_ignored(loc.x, loc.y):
                    self._mouse_down_pos = None
                    return event
                self._flush_text_buffer()
                self._mouse_down_pos = (loc.x, loc.y)

            elif event_type == Quartz.kCGEventLeftMouseUp:
                loc = Quartz.CGEventGetLocation(event)
                if self._is_point_ignored(loc.x, loc.y):
                    self._mouse_down_pos = None
                    return event
                click_state = Quartz.CGEventGetIntegerValueField(
                    event, Quartz.kCGMouseEventClickState
                )
                if self._mouse_down_pos:
                    dx = loc.x - self._mouse_down_pos[0]
                    dy = loc.y - self._mouse_down_pos[1]
                    dist = math.hypot(dx, dy)
                    if dist < 5:
                        elem_info = self._inspect_element_at(loc.x, loc.y)
                        action_type = "double_click" if click_state == 2 else "click"
                        label = elem_info.get("title") or elem_info.get("role") or "screen"
                        self._emit_step(
                            action=action_type,
                            params={
                                "x": round(loc.x),
                                "y": round(loc.y),
                                "click_count": click_state,
                                "element": elem_info,
                            },
                            spoken=f"{action_type.replace('_', ' ').capitalize()}ed {label}",
                        )
                    else:
                        selected_text = self._inspect_focused_selection()
                        if selected_text:
                            self._emit_step(
                                action="select_text",
                                params={
                                    "selected_text": selected_text,
                                    "start_x": round(self._mouse_down_pos[0]),
                                    "start_y": round(self._mouse_down_pos[1]),
                                    "end_x": round(loc.x),
                                    "end_y": round(loc.y),
                                },
                                spoken=(
                                    f"Selected '{selected_text[:30]}...'"
                                    if len(selected_text) > 30
                                    else f"Selected '{selected_text}'"
                                ),
                            )
                        else:
                            self._emit_step(
                                action="drag",
                                params={
                                    "start_x": round(self._mouse_down_pos[0]),
                                    "start_y": round(self._mouse_down_pos[1]),
                                    "end_x": round(loc.x),
                                    "end_y": round(loc.y),
                                },
                                spoken="Dragged and dropped",
                            )
                self._mouse_down_pos = None

        return event

    def _run_tap(self) -> None:
        mask = (
            Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown)
            | Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseDown)
            | Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseUp)
            | Quartz.CGEventMaskBit(Quartz.kCGEventRightMouseDown)
            | Quartz.CGEventMaskBit(Quartz.kCGEventScrollWheel)
        )
        self._tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionListenOnly,
            mask,
            self._event_callback,
            None,
        )
        if not self._tap:
            print(
                "[PalTeachRecorder] Failed to create Quartz Event Tap. Check Accessibility permissions."
            )
            return

        self._loop_source = Quartz.CFMachPortCreateRunLoopSource(None, self._tap, 0)
        self._loop = Quartz.CFRunLoopGetCurrent()
        Quartz.CFRunLoopAddSource(self._loop, self._loop_source, Quartz.kCFRunLoopDefaultMode)
        Quartz.CGEventTapEnable(self._tap, True)

        while self._running:
            res = Quartz.CFRunLoopRunInMode(Quartz.kCFRunLoopDefaultMode, 0.5, False)
            if res not in (Quartz.kCFRunLoopRunTimedOut, Quartz.kCFRunLoopRunHandledSource):
                break


ACTIVE_TRAINING_JOURNAL = Path.home() / ".voicefi" / "active_training.json"


class PalTrainer:
    """
    Session orchestrator for Demonstration-Based Learning ('Hey Pal, watch me do this').
    Synthesizes recorded OS event streams into clean, declarative YAML workflows.
    """

    def __init__(
        self, workflow_name: str, on_step: Optional[Callable[[PalTeachRawStep], None]] = None
    ):
        clean_name = (workflow_name or "").replace("\x00", "").strip()
        self.workflow_name = clean_name or "Workflow"
        self.workflow_slug = (
            re.sub(r"[^a-zA-Z0-9_-]+", "_", self.workflow_name.lower()).strip("_-") or "workflow"
        )
        self._user_on_step = on_step
        self.recorder = PalTeachRecorder(on_step=self._handle_step)
        self.start_time: float = 0.0

    def _handle_step(self, step: PalTeachRawStep) -> None:
        self._write_journal(active=True)
        if self._user_on_step is not None:
            try:
                self._user_on_step(step)
            except Exception:
                pass

    def pause_auth(self, reason: str = "manual") -> None:
        """Pause capture for authentication/MFA (Patent Claim 7)."""
        self.recorder.pause_auth(reason=reason)
        self._write_journal(active=True)

    def resume_auth(self) -> None:
        """Resume capture after authentication (Patent Claim 7)."""
        self.recorder.resume_auth()
        self._write_journal(active=True)

    def is_auth_paused(self) -> bool:
        return self.recorder.is_auth_paused()

    def _write_journal(self, active: bool = True) -> None:
        """Persist live training session state to ~/.voicefi/active_training.json for cross-process resilience."""
        try:
            ACTIVE_TRAINING_JOURNAL.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "active": active,
                "auth_paused": self.recorder.is_auth_paused(),
                "pid": os.getpid(),
                "workflow_name": self.workflow_name,
                "workflow_slug": self.workflow_slug,
                "start_time": self.start_time,
                "updated_at": time.time(),
                "voice_notes": list(getattr(self.recorder, "voice_notes", [])),
                "steps": [asdict(s) for s in getattr(self.recorder, "steps", [])],
            }
            tmp_path = ACTIVE_TRAINING_JOURNAL.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp_path.replace(ACTIVE_TRAINING_JOURNAL)
        except Exception:
            pass

    @classmethod
    def clear_journal(cls) -> None:
        try:
            ACTIVE_TRAINING_JOURNAL.unlink(missing_ok=True)
        except Exception:
            pass

    @classmethod
    def load_journal(cls) -> Optional[Dict[str, Any]]:
        try:
            if ACTIVE_TRAINING_JOURNAL.is_file():
                data = json.loads(ACTIVE_TRAINING_JOURNAL.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("active"):
                    return data
        except Exception:
            pass
        return None

    @classmethod
    def from_journal(cls) -> Optional["PalTrainer"]:
        """Reconstruct a PalTrainer instance from the active on-disk journal."""
        data = cls.load_journal()
        if not data:
            return None
        wf_name = str(data.get("workflow_name") or "workflow")
        trainer = cls(wf_name)
        trainer.start_time = float(data.get("start_time") or time.time())
        trainer.recorder.voice_notes = [str(n) for n in (data.get("voice_notes") or [])]
        restored_steps: List[PalTeachRawStep] = []
        for idx, raw_s in enumerate(data.get("steps") or [], start=1):
            if isinstance(raw_s, dict):
                restored_steps.append(
                    PalTeachRawStep(
                        step_number=int(raw_s.get("step_number") or idx),
                        action=str(raw_s.get("action") or "click"),
                        app_name=str(raw_s.get("app_name") or ""),
                        window_title=raw_s.get("window_title"),
                        params=dict(raw_s.get("params") or {}),
                        spoken_summary=str(raw_s.get("spoken_summary") or ""),
                        voice_note=raw_s.get("voice_note"),
                        timestamp=float(raw_s.get("timestamp") or time.time()),
                    )
                )
        trainer.recorder.steps = restored_steps
        return trainer

    def rename_workflow(self, new_name: str) -> None:
        """Update the workflow title and slug dynamically during training."""
        clean = (new_name or "").strip()
        if not clean:
            return
        self.workflow_name = clean
        self.workflow_slug = (
            re.sub(r"[^a-zA-Z0-9_-]+", "_", self.workflow_name.lower()).strip("_") or "workflow"
        )
        self._write_journal(active=True)

    def remove_step(self, step_number: int) -> List[PalTeachRawStep]:
        """Remove a step by its 1-based step_number and return updated steps list."""
        updated = self.recorder.remove_step(step_number)
        self._write_journal(active=True)
        return updated

    def start(self) -> None:
        """Start the training session."""
        self.start_time = time.time()
        self.recorder.start()
        self._write_journal(active=True)

    def record_voice_note(self, text: str) -> None:
        """Attach spoken commentary to the active demonstration."""
        self.recorder.attach_voice_note(text)
        self._write_journal(active=True)

    def cancel(self) -> None:
        """Abort without saving."""
        self.recorder.stop()
        self.clear_journal()

    def _try_promote_to_headless_recipe(
        self,
        raw_steps: List[PalTeachRawStep],
        notes_list: List[str],
        unique_triggers: List[str],
    ) -> Optional[PalWorkflowRecipe]:
        """
        Pal Demonstration-to-Deterministic Compiler (Tier 3 UI -> Tier 0/1/1.5 Promotion):
        Decompiles recorded UI demonstrations (window titles, dragged files, clipboard diffs,
        and spoken voice notes) into a canonical Semantic IR with SSA data-flow edges and
        lowers recognized archetypes (client_upload, lead_gen_*, slack_qa_handoff, git_pr_release,
        email_triage) into contract-verified recipes.
        """
        try:
            from whole_body_compute.integrations.pal_compiler import PalCompiler

            recipe, ir_prog, _ = PalCompiler.compile_demonstration(
                raw_steps=raw_steps,
                voice_notes=notes_list,
                workflow_name=self.workflow_name,
                unique_triggers=unique_triggers,
            )
            if ir_prog.archetype in (
                "client_upload",
                "lead_gen_harvester",
                "lead_gen_enrichment",
                "lead_gen_outbound",
                "slack_qa_handoff",
                "git_pr_release",
                "email_triage",
            ):
                return recipe
        except Exception as e:
            print(f"[PalTrainer] PalCompiler optimization fallback: {e}")

        return None

    def finish(self, save_dir: Optional[Path] = None, optimize: bool = True) -> PalWorkflowRecipe:
        """
        Stop recording, denoise the action stream, compile to a PalWorkflowRecipe,
        save to ~/.voicefi/workflows/<slug>.yaml, and hot-reload into the registry.
        If optimize=True, attempts to promote brittle UI sequences into headless API/CLI steps.
        """
        raw_steps = self.recorder.stop()
        self.clear_journal()
        compiled_steps: List[WorkflowStep] = []

        for s in raw_steps:
            action = str(getattr(s, "action", "") or "click")
            step_name = getattr(s, "spoken_summary", "") or "Step"
            desc = getattr(s, "voice_note", None)
            s_params = s.params if isinstance(getattr(s, "params", None), dict) else {}

            if action == "switch_app":
                compiled_steps.append(
                    WorkflowStep(
                        name=f"Focus {s_params.get('app_name')}",
                        action=ActionType.ACTIVATE_APP,
                        app_name=s_params.get("app_name"),
                        bundle_id=s_params.get("bundle_id"),
                        description=desc,
                    )
                )
            elif action == "hotkey":
                compiled_steps.append(
                    WorkflowStep(
                        name=f"Press {s_params.get('combo')}",
                        action=ActionType.HOTKEY,
                        combo=s_params.get("combo"),
                        key=s_params.get("key"),
                        modifiers=s_params.get("modifiers") if isinstance(s_params.get("modifiers"), list) else None,
                        description=desc,
                    )
                )
            elif action == "press_key":
                compiled_steps.append(
                    WorkflowStep(
                        name=f"Press {s_params.get('key')}",
                        action=ActionType.PRESS_KEY,
                        key=str(s_params.get("key") or ""),
                        description=desc,
                    )
                )
            elif action == "type_text":
                text = str(s_params.get("text") or "")
                compiled_steps.append(
                    WorkflowStep(
                        name=f"Type '{text[:20]}...'" if len(text) > 20 else f"Type '{text}'",
                        action=ActionType.KEYSTROKE,
                        text=text,
                        description=desc,
                    )
                )
            elif action in ("click", "double_click", "right_click"):
                raw_elem = s_params.get("element")
                elem = raw_elem if isinstance(raw_elem, dict) else {}
                role = str(elem.get("role") or "") or None
                title = str(elem.get("title") or "") or None
                dom_selector = str(elem.get("dom_selector") or "") or None
                aria_label = str(elem.get("aria_label") or "") or None
                bid = str(elem.get("bid") or "") or None
                win_dx = elem.get("window_anchor_dx")
                win_dy = elem.get("window_anchor_dy")
                target = ClickTarget(
                    target_type="dom_selector" if (dom_selector or bid) else "coordinates",
                    x=s_params.get("x"),
                    y=s_params.get("y"),
                    role=role,
                    title=title,
                    dom_selector=dom_selector,
                    aria_label=aria_label,
                    bid=bid,
                    window_anchor_dx=float(win_dx) if win_dx is not None else None,
                    window_anchor_dy=float(win_dy) if win_dy is not None else None,
                )
                if action == "double_click":
                    act_type = ActionType.DOUBLE_CLICK
                    step_prefix = "Double-click"
                elif action == "right_click":
                    act_type = ActionType.RIGHT_CLICK
                    step_prefix = "Right-click"
                else:
                    act_type = ActionType.CLICK
                    step_prefix = "Click"
                compiled_steps.append(
                    WorkflowStep(
                        name=f"{step_prefix} {title or role or 'screen'}",
                        action=act_type,
                        click_target=target,
                        description=desc,
                    )
                )
            elif action == "drag":
                target = ClickTarget(
                    target_type="coordinates",
                    x=s.params.get("start_x"),
                    y=s.params.get("start_y"),
                    end_x=s.params.get("end_x"),
                    end_y=s.params.get("end_y"),
                )
                compiled_steps.append(
                    WorkflowStep(
                        name="Drag and drop",
                        action=ActionType.DRAG,
                        click_target=target,
                        description=desc,
                    )
                )
            elif action == "select_text":
                sel_text = str(s_params.get("selected_text") or "")
                target = ClickTarget(
                    target_type="coordinates",
                    x=s_params.get("start_x"),
                    y=s_params.get("start_y"),
                    end_x=s_params.get("end_x"),
                    end_y=s_params.get("end_y"),
                )
                compiled_steps.append(
                    WorkflowStep(
                        name=f"Select '{sel_text[:20]}...'" if len(sel_text) > 20 else f"Select '{sel_text}'",
                        action=ActionType.CLICK,
                        click_target=target,
                        text=sel_text,
                        description=desc,
                    )
                )
            elif action in ("companion_interaction", "voice_instruction"):
                compiled_steps.append(
                    WorkflowStep(
                        name=f"Companion step: {s.spoken_summary[:40]}",
                        action=ActionType.NOTIFY,
                        parameters={"message": s.spoken_summary, "voice_note": s.voice_note or ""},
                        description=desc,
                    )
                )

        # Build recipe
        triggers = [
            self.workflow_name.lower(),
            f"run {self.workflow_name.lower()}",
            f"do {self.workflow_name.lower()}",
            f"start {self.workflow_name.lower()}",
        ]
        # Remove duplicates while preserving order
        unique_triggers = list(dict.fromkeys(triggers))

        notes_list = list(getattr(self.recorder, "voice_notes", []))
        desc_text = f"Demonstrated workflow for '{self.workflow_name}' ({len(compiled_steps)} steps)"
        if notes_list:
            desc_text += f" — Notes: {' | '.join(notes_list)}"

        promoted_recipe = None
        compiler_report_dict = None
        compiler_contract_dict = None
        if optimize:
            try:
                from whole_body_compute.integrations.pal_compiler import PalCompiler
                from whole_body_compute.integrations.pal_harness import PalHarness
                from whole_body_compute.integrations.screenpipe import (
                    extract_semantic_screen_entities,
                    search_screen_content,
                )

                # Multimodal Fusion: Cross-reference recent Foveal Vision episodic OCR frames for semantic entities
                sp_entities_found: List[str] = []
                for s in raw_steps:
                    ocr_txt = str((s.params or {}).get("screenpipe_ocr") or "")
                    if ocr_txt:
                        ents = extract_semantic_screen_entities(ocr_txt)
                        for val_list in ents.values():
                            for v in val_list:
                                if v and v not in sp_entities_found:
                                    sp_entities_found.append(v)
                if not PalHarness._is_test_env():
                    try:
                        recent_recs = search_screen_content(
                            query="",
                            content_type="ocr",
                            minutes_ago=15,
                            limit=8,
                            capture_live_if_empty=False,
                        )
                        for rec in recent_recs:
                            ents = extract_semantic_screen_entities(rec.text or "")
                            for val_list in ents.values():
                                for v in val_list:
                                    if v and v not in sp_entities_found:
                                        sp_entities_found.append(v)
                    except Exception:
                        pass
                if sp_entities_found:
                    fusion_note = f"Episodic OCR Cross-Reference Entities: {', '.join(sp_entities_found[:12])}"
                    if fusion_note not in notes_list:
                        notes_list.append(fusion_note)

                comp_recipe, ir_prog, comp_report = PalCompiler.compile_demonstration(
                    raw_steps=raw_steps,
                    voice_notes=notes_list,
                    workflow_name=self.workflow_name,
                    unique_triggers=unique_triggers,
                )
                compiler_report_dict = comp_report.to_dict()
                compiler_contract_dict = comp_recipe.contract_summary
                has_synthesized_api_node = any(
                    "taint_lowered_api" in n.node_id or "schema_lowered_" in n.node_id
                    for n in ir_prog.nodes
                )
                if ir_prog.archetype in (
                    "invoice_batch",
                    "client_upload",
                    "lead_gen_harvester",
                    "lead_gen_enrichment",
                    "lead_gen_outbound",
                    "slack_qa_handoff",
                    "git_pr_release",
                    "email_triage",
                ) or has_synthesized_api_node:
                    promoted_recipe = comp_recipe
                    # Persist any synthesized Tier-1 Python script to disk
                    target_base = save_dir or (Path.home() / ".voicefi" / "workflows")
                    scripts_dir = target_base / "scripts"
                    for n in ir_prog.nodes:
                        syn_code = (n.inputs or {}).get("synthesized_script")
                        if syn_code and isinstance(syn_code, str):
                            scripts_dir.mkdir(parents=True, exist_ok=True)
                            script_file = scripts_dir / f"{self.workflow_slug}_api.py"
                            script_file.write_text(syn_code, encoding="utf-8")
                            script_note = f"Synthesized Tier-1 Python Script: {script_file}"
                            if script_note not in promoted_recipe.notes:
                                promoted_recipe.notes.append(script_note)
                            for st in promoted_recipe.steps:
                                if st.action == ActionType.RUN_SHELL and st.name.startswith("Headless Tier-1 API:"):
                                    st.command = f"python3 {str(script_file)!r}"
                            break
            except Exception as e:
                print(f"[PalTrainer] PalCompiler analysis fallback: {e}")

        if promoted_recipe is not None:
            recipe = promoted_recipe
        else:
            recipe = PalWorkflowRecipe(
                id=self.workflow_slug,
                name=self.workflow_name,
                description=desc_text,
                notes=notes_list,
                triggers=unique_triggers,
                steps=compiled_steps,
                success_summary=f"Completed {self.workflow_name} successfully",
                error_summary=f"Failed during {self.workflow_name} at {{step_name}}",
                tier_summary=compiler_report_dict,
                contract_summary=compiler_contract_dict,
            )

        # Save to disk
        target_dir = save_dir or (Path.home() / ".voicefi" / "workflows")
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{self.workflow_slug}.yaml"

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(recipe.model_dump(mode="json", exclude_none=True), f, sort_keys=False)

        # Hot-reload in memory
        PalWorkflowRegistry.register_recipe(recipe)

        # Persist demonstrated workflow triggers & parameters into DotMemory Hippocampus
        try:
            from whole_body_compute.local.dot_memory import get_dot_store

            get_dot_store().record_workflow_compiled(
                workflow_name=recipe.name,
                workflow_slug=recipe.id,
                triggers=list(recipe.triggers or []),
                step_count=len(recipe.steps or []),
                parameters=list((recipe.parameters or {}).keys()),
                notes=list(recipe.notes or []),
            )
        except Exception as dot_err:
            print(f"[PalTrainer] DotMemory notice: {dot_err}")

        # Export headless Goose YAML recipe & local small-model JSONL training example
        try:
            self._export_goose_recipe_and_training_pair(
                recipe=recipe,
                raw_steps=raw_steps,
                notes_list=notes_list,
                target_dir=target_dir,
            )
        except Exception as exp_err:
            print(f"[PalTrainer] Goose/JSONL export notice: {exp_err}")

        return recipe

    def _export_goose_recipe_and_training_pair(
        self,
        recipe: PalWorkflowRecipe,
        raw_steps: List[PalTeachRawStep],
        notes_list: List[str],
        target_dir: Path,
    ) -> Tuple[Path, Path]:
        """
        1. Export a headless Goose YAML recipe (`~/.voicefi/goose_recipes/<slug>.yaml`) so Goose
           can replay the demonstrated process via MCP without moving the mouse or stealing window focus.
        2. Append a fine-tuning example (`~/.voicefi/training_data/pal_spoken_processes.jsonl`) pairing
           the user's spoken process narration + Foveal Vision episodic OCR with the compiled step plan for local models.
        """
        from whole_body_compute.integrations.screenpipe import (
            extract_semantic_screen_entities,
            resolve_direct_headless_route,
            strip_ui_chrome_from_ocr,
        )

        step_lines: List[str] = []
        ocr_snippets: List[str] = []
        direct_routes: List[str] = []
        for idx, s in enumerate(raw_steps, start=1):
            note_suffix = f" (Voice note: {s.voice_note})" if s.voice_note else ""
            ocr_txt = strip_ui_chrome_from_ocr(str((s.params or {}).get("screenpipe_ocr") or ""))
            route_hint = resolve_direct_headless_route(s.app_name, s.window_title or "", ocr_txt)
            if route_hint not in direct_routes:
                direct_routes.append(route_hint)
            entities = extract_semantic_screen_entities(ocr_txt) if ocr_txt else {}
            ent_summary = ", ".join(f"{k}={v}" for k, v in entities.items() if v)
            ent_suffix = f" [Entities: {ent_summary}]" if ent_summary else ""
            step_lines.append(
                f"{idx}. [{s.app_name or 'macOS'}] {s.spoken_summary}{note_suffix}{ent_suffix}"
            )
            if ocr_txt:
                ocr_snippets.append(f"[{s.app_name}] {ocr_txt[:240]}")

        if not step_lines and recipe.steps:
            for idx, ws in enumerate(recipe.steps, start=1):
                step_lines.append(f"{idx}. {ws.name} ({ws.action.value})")

        mcp_bin = str(Path(__file__).resolve().parents[3] / "bin" / "screenpipe-mcp")
        goose_recipe_doc = {
            "version": "1.0.0",
            "title": f"Pal Learned Workflow: {self.workflow_name}",
            "description": recipe.description,
            "instructions": (
                f"Execute the user's taught workflow '{self.workflow_name}' headlessly on macOS "
                "without stealing window focus.\n"
                "CRITICAL STRENGTH-REDUCTION RULE: Treat Foveal Vision episodic OCR and recorded UI steps strictly as "
                "a READ-ONLY DATA SENSOR (for extracting filenames, paths, URLs, IDs, and commands). "
                "NEVER execute coordinate mouse clicks, `cliclick`, or `System Events` UI clicking. "
                "Always use the Tier 0/1 direct route:\n- "
                + "\n- ".join(direct_routes or ["Use direct Python, filesystem, CLI (gh, git, pytest, curl, pbcopy), or API calls."])
                + "\n\nObserved Demonstration Trace:\n"
                + ("\n".join(step_lines) if step_lines else "Follow the user's spoken notes.")
                + (
                    "\n\nSpoken Process Notes:\n- " + "\n- ".join(notes_list)
                    if notes_list
                    else ""
                )
                + "\n\nReturn ONLY a single spoken sentence under 25 words for TTS playback."
            ),
            "prompt": f"Run the learned workflow '{self.workflow_name}' and summarize the result in under 25 words.",
            "extensions": [
                {"type": "builtin", "name": "developer"},
                {"type": "stdio", "name": "screenpipe", "cmd": mcp_bin, "args": []},
            ],
        }

        goose_dir = target_dir.parent / "goose_recipes"
        goose_dir.mkdir(parents=True, exist_ok=True)
        goose_file = goose_dir / f"{self.workflow_slug}.yaml"
        with open(goose_file, "w", encoding="utf-8") as gf:
            yaml.dump(goose_recipe_doc, gf, sort_keys=False)

        train_dir = target_dir.parent / "training_data"
        train_dir.mkdir(parents=True, exist_ok=True)
        jsonl_file = train_dir / "pal_spoken_processes.jsonl"
        user_context = f"Workflow: {self.workflow_name}\n"
        if notes_list:
            user_context += "Spoken Process Narration: " + " | ".join(notes_list) + "\n"
        if ocr_snippets:
            user_context += "Foveal Vision Episodic OCR Context:\n" + "\n".join(ocr_snippets[:3]) + "\n"

        training_row = {
            "workflow_id": recipe.id,
            "workflow_name": self.workflow_name,
            "timestamp": time.time(),
            "messages": [
                {
                    "role": "system",
                    "content": "You are VoiceFi Pal, an on-device macOS workflow compiler trained on the user's spoken processes.",
                },
                {"role": "user", "content": user_context.strip()},
                {
                    "role": "assistant",
                    "content": "\n".join(step_lines) if step_lines else recipe.description,
                },
            ],
        }
        with open(jsonl_file, "a", encoding="utf-8") as jf:
            jf.write(json.dumps(training_row) + "\n")

        return goose_file, jsonl_file

    @classmethod
    def from_screenpipe_history(
        cls,
        workflow_name: str,
        minutes: int = 3,
        db_path: Optional[Path] = None,
    ) -> "PalTrainer":
        """
        Retroactive Training Mode ('Hey Pal, learn what I just did in the last 3 minutes'):
        Reconstructs a PalTrainer session from Foveal Vision episodic rolling OCR + audio SQLite history
        even if the user did not turn on Training Mode beforehand.
        """
        from whole_body_compute.integrations.screenpipe import search_screen_content

        trainer = cls(workflow_name=workflow_name)
        trainer.start_time = time.time() - (minutes * 60)

        ocr_records = search_screen_content(
            query="",
            content_type="ocr",
            minutes_ago=minutes,
            limit=12,
            db_path=db_path,
            capture_live_if_empty=True,
        )
        audio_records = search_screen_content(
            query="",
            content_type="audio",
            minutes_ago=minutes,
            limit=12,
            db_path=db_path,
            capture_live_if_empty=False,
        )

        notes = [r.text.strip() for r in reversed(audio_records) if r.text and r.text.strip()]
        trainer.recorder.voice_notes = notes

        steps: List[PalTeachRawStep] = []
        if ocr_records:
            for idx, rec in enumerate(reversed(ocr_records), start=1):
                app = rec.app_name or "Terminal"
                win = rec.window_name or app
                first_line = next((ln.strip() for ln in rec.text.splitlines() if ln.strip()), win)
                if "companion" in app.lower():
                    action_name = "companion_interaction"
                    spoken_desc = f"Companion: {first_line[:50]}"
                else:
                    action_name = "switch_app"
                    spoken_desc = f"Inspect {app} ({first_line[:50]})"
                steps.append(
                    PalTeachRawStep(
                        step_number=idx,
                        action=action_name,
                        app_name=app,
                        window_title=win,
                        params={
                            "app_name": app,
                            "window_title": win,
                            "screenpipe_ocr": rec.text[:600],
                        },
                        spoken_summary=spoken_desc,
                        voice_note=notes[idx - 1] if idx - 1 < len(notes) else None,
                    )
                )
        elif audio_records:
            # Reconstruct workflow purely from spoken companion notes when no GUI windows were focused
            for idx, arec in enumerate(reversed(audio_records), start=1):
                txt = arec.text.strip()
                steps.append(
                    PalTeachRawStep(
                        step_number=idx,
                        action="voice_instruction",
                        app_name=arec.app_name or "VoiceFi Companion",
                        window_title="Spoken Workflow",
                        params={"instruction": txt, "screenpipe_ocr": txt[:600]},
                        spoken_summary=f"Voice: {txt[:50]}",
                        voice_note=txt,
                    )
                )

        trainer.recorder.steps = steps
        return trainer

    @classmethod
    def from_ambient_motif(
        cls,
        motif: Any,
        workflow_name: Optional[str] = None,
    ) -> "PalTrainer":
        """
        Synthesize a PalTrainer demonstration session from an ambiently discovered WorkflowMotif.
        Converts passive cross-app transition steps into editable raw steps ready for review,
        headless promotion, or refinement without requiring manual Teach Mode activation.
        """
        name = workflow_name or getattr(motif, "name", "Ambient Workflow")
        trainer = cls(workflow_name=name)

        steps: List[PalTeachRawStep] = []
        step_idx = 1
        exemplars = getattr(motif, "exemplar_events", [])

        for event in exemplars:
            src_app = getattr(event, "source_app", "") or "App"
            src_win = getattr(event, "source_window", "") or src_app
            tgt_app = getattr(event, "target_app", "") or src_app
            tgt_win = getattr(event, "target_window", "") or tgt_app
            entity = getattr(event, "copied_entity", "") or ""
            etype = getattr(event, "entity_type", "text")
            act = getattr(event, "action_type", "copy_paste")

            # 1. Source focus / copy step
            steps.append(
                PalTeachRawStep(
                    step_number=step_idx,
                    action="switch_app",
                    app_name=src_app,
                    window_title=src_win,
                    params={"app_name": src_app, "window_title": src_win},
                    spoken_summary=f"Focus {src_app}",
                    voice_note=f"Copy {etype} from {src_win}",
                )
            )
            step_idx += 1

            if entity:
                steps.append(
                    PalTeachRawStep(
                        step_number=step_idx,
                        action="hotkey",
                        app_name=src_app,
                        window_title=src_win,
                        params={"combo": "Cmd+C", "key": "C", "modifiers": ["Cmd"], "entity_type": etype},
                        spoken_summary=f"Copy {etype}",
                        voice_note=f"Copied {etype} ({entity[:20]})",
                    )
                )
                step_idx += 1

            # 2. Target focus / paste step
            if tgt_app != src_app or tgt_win != src_win:
                steps.append(
                    PalTeachRawStep(
                        step_number=step_idx,
                        action="switch_app",
                        app_name=tgt_app,
                        window_title=tgt_win,
                        params={
                            "app_name": tgt_app,
                            "window_title": tgt_win,
                            "target_url": getattr(event, "target_url", None),
                        },
                        spoken_summary=f"Switch to {tgt_app}",
                        voice_note=f"Navigate to {tgt_win}",
                    )
                )
                step_idx += 1

            steps.append(
                PalTeachRawStep(
                    step_number=step_idx,
                    action="hotkey" if act in ("paste", "copy_paste") else "type_text",
                    app_name=tgt_app,
                    window_title=tgt_win,
                    params={
                        "combo": "Cmd+V" if act in ("paste", "copy_paste") else None,
                        "text": entity if act not in ("paste", "copy_paste") else None,
                        "entity_type": etype,
                        "element": getattr(event, "target_element", None),
                    },
                    spoken_summary=f"Paste {etype} into {tgt_app}",
                    voice_note=f"Pasted into {tgt_win}",
                )
            )
            step_idx += 1

        trainer.recorder.steps = steps
        return trainer

