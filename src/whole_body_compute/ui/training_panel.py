"""
Granola-Style Floating Training Notepad & Live Step Feed Card for VoiceFi Pal Teach Mode.

Renders a sleek, native macOS frosted-glass floating notepad card anchored in the top-right
corner (just below the menu bar, like Granola) whenever Pal Teach Mode ("watch me do this" /
"start training") is active.

Key UX Elements:
- Pulsing Recording Header: Crimson status dot, live MM:SS elapsed timer, and reactive 5-bar mic waveform
- Editable Workflow Title: Click to rename the workflow inline at any time before saving
- Live Chronological Step Feed: Real-time cards for every captured OS action (app switches, clicks,
  shortcuts, typing, drag & drop) with action badges and hover/inline delete (×) buttons
- Inline Spoken Voice Notes: Any spoken explanation during demonstration automatically attaches to
  the active step as a warm amber/gold callout quote
- Interactive Action Footer: "Cancel" and emerald "Save Workflow" buttons (alongside voice commands)
- Self-Click Exclusion: Automatically registers its Quartz screen bounds with PalTeachRecorder so
  clicking buttons or editing the title inside the card never pollutes the recorded workflow steps.
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
import warnings
from typing import Any, Callable, Dict, List, Optional

try:
    import objc

    if hasattr(objc, "ObjCPointerWarning"):
        warnings.filterwarnings("ignore", category=objc.ObjCPointerWarning)
except Exception:
    pass

from AppKit import (
    NSApplication,
    NSBackingStoreBuffered,
    NSBezierPath,
    NSButton,
    NSColor,
    NSFloatingWindowLevel,
    NSFont,
    NSFontAttributeName,
    NSForegroundColorAttributeName,
    NSLineBreakByTruncatingTail,
    NSLineBreakByWordWrapping,
    NSMakeRect,
    NSPanel,
    NSPoint,
    NSRect,
    NSScreen,
    NSScrollView,
    NSSize,
    NSStatusWindowLevel,
    NSString,
    NSTextAlignmentCenter,
    NSTextAlignmentLeft,
    NSTextAlignmentRight,
    NSTextField,
    NSView,
    NSVisualEffectBlendingModeBehindWindow,
    NSVisualEffectMaterialHUDWindow,
    NSVisualEffectView,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
)
from PyObjCTools import AppHelper


def is_headless() -> bool:
    """Return True if running in headless or automated test environment."""
    return bool(
        os.getenv("VOICEFI_HEADLESS") == "1"
        or os.getenv("HEADLESS") == "1"
        or os.getenv("PYTEST_CURRENT_TEST") is not None
        or os.getenv("VOICEFI_TESTING") == "1"
        or "pytest" in sys.modules
    )


ACTION_ICONS: Dict[str, str] = {
    "switch_app": "⌘",
    "click": "🖱️",
    "double_click": "🖱️",
    "right_click": "🖱️",
    "hotkey": "⌨️",
    "press_key": "⌨️",
    "type_text": "✎",
    "drag": "✥",
    "active_item": "🎙️",
    "applying": "⏳",
    "select": "✨",
    "edit_applied": "✅",
    "edit_failed": "⚠️",
    "undo": "↩",
}


try:
    GranolaHeaderPulseView = objc.lookUpClass("GranolaHeaderPulseView")
except objc.nosuchclass_error:

    class GranolaHeaderPulseView(NSView):
        """Renders the pulsing crimson recording indicator and 5-bar mini audio waveform."""

        def initWithFrame_(self, frame: NSRect):
            self = objc.super(GranolaHeaderPulseView, self).initWithFrame_(frame)
            if self is not None:
                self._phase = 0.0
                self._levels = [0.25, 0.45, 0.35, 0.55, 0.25]
                self._multipliers = [0.75, 1.25, 1.05, 1.35, 0.80]
                self._is_speech = False
            return self

        def tickPulse(self) -> None:
            self._phase = (self._phase + 0.35) % (2.0 * math.pi)
            if not self._is_speech:
                for i in range(len(self._levels)):
                    idle_wave = 0.20 + 0.10 * math.sin(self._phase + i * 0.9)
                    self._levels[i] = 0.7 * self._levels[i] + 0.3 * idle_wave
            self.setNeedsDisplay_(True)

        def setAudioLevel_prob_speech_(
            self, level: float, prob: float, is_speech: bool
        ) -> None:
            loudness = min(1.0, max(0.0, math.sqrt(max(0.0, float(level))) * 4.2))
            self._is_speech = bool(is_speech or prob >= 0.40)
            self._phase = (self._phase + 0.45) % (2.0 * math.pi)
            for i, mult in enumerate(self._multipliers):
                osc = 0.80 + 0.20 * math.sin(self._phase + i * 1.1)
                tgt = max(0.18, min(1.0, loudness * mult * osc))
                self._levels[i] = 0.65 * tgt + 0.35 * self._levels[i]
            self.setNeedsDisplay_(True)

        def drawRect_(self, dirtyRect: NSRect) -> None:
            bounds = self.bounds()
            w = float(bounds.size.width)
            h = float(bounds.size.height)
            if w <= 10 or h <= 10:
                return

            # 1. Outer soft crimson halo + inner recording dot on the left (x=10, y=h/2)
            cy = h / 2.0
            cx = 10.0
            pulse = 0.5 + 0.5 * math.sin(self._phase)
            halo_r = 5.5 + 2.5 * pulse
            halo_alpha = 0.18 + 0.22 * pulse
            halo_rect = NSRect(
                NSPoint(cx - halo_r, cy - halo_r), NSSize(halo_r * 2.0, halo_r * 2.0)
            )
            halo_path = NSBezierPath.bezierPathWithOvalInRect_(halo_rect)
            NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.96, 0.22, 0.28, halo_alpha
            ).setFill()
            halo_path.fill()

            dot_r = 4.2
            dot_rect = NSRect(
                NSPoint(cx - dot_r, cy - dot_r), NSSize(dot_r * 2.0, dot_r * 2.0)
            )
            dot_path = NSBezierPath.bezierPathWithOvalInRect_(dot_rect)
            NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.98, 0.24, 0.30, 1.0
            ).setFill()
            dot_path.fill()

            # 2. Mini 5-bar waveform on the right side of this view
            num_bars = len(self._levels)
            bar_w = 2.4
            bar_gap = 2.0
            total_bars_w = num_bars * bar_w + (num_bars - 1) * bar_gap
            start_x = w - total_bars_w - 2.0

            if self._is_speech:
                bar_col = NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.20, 0.92, 0.58, 0.98
                )
            else:
                bar_col = NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.98, 0.38, 0.42, 0.80
                )
            bar_col.setFill()

            max_bar_h = h - 6.0
            for i, lvl in enumerate(self._levels):
                bh = max(3.5, min(max_bar_h, lvl * max_bar_h))
                bx = start_x + i * (bar_w + bar_gap)
                by = (h - bh) / 2.0
                b_rect = NSRect(NSPoint(bx, by), NSSize(bar_w, bh))
                b_path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    b_rect, bar_w / 2.0, bar_w / 2.0
                )
                b_path.fill()


try:
    GranolaPillButtonView = objc.lookUpClass("GranolaPillButtonView")
except objc.nosuchclass_error:

    class GranolaPillButtonView(NSView):
        """Clean custom-drawn pill button for Save Workflow, Cancel, and Step Delete."""

        def initWithFrame_title_style_callback_(
            self, frame: NSRect, title: str, style: str, callback: Callable[[], None]
        ):
            self = objc.super(GranolaPillButtonView, self).initWithFrame_(frame)
            if self is not None:
                self._title = title
                self._style = style  # "primary", "secondary", "delete", "add_note"
                self._callback = callback
                self._hovered = False
                try:
                    options = 0x01 | 0x02 | 0x80
                    ta = (
                        objc.lookUpClass("NSTrackingArea")
                        .alloc()
                        .initWithRect_options_owner_userInfo_(
                            self.bounds(), options, self, None
                        )
                    )
                    self.addTrackingArea_(ta)
                except Exception:
                    pass
            return self

        def acceptsFirstMouse_(self, event) -> bool:
            return True

        def mouseDownCanMoveWindow(self) -> bool:
            return False

        def mouseEntered_(self, event) -> None:
            self._hovered = True
            self.setNeedsDisplay_(True)

        def mouseExited_(self, event) -> None:
            self._hovered = False
            self.setNeedsDisplay_(True)

        def mouseDown_(self, event) -> None:
            if self._callback:
                try:
                    self._callback()
                except Exception:
                    pass

        def setTitle_(self, title: str) -> None:
            self._title = title
            self.setNeedsDisplay_(True)

        def drawRect_(self, dirtyRect: NSRect) -> None:
            bounds = self.bounds()
            w = float(bounds.size.width)
            h = float(bounds.size.height)
            radius = 7.0 if self._style != "delete" else h / 2.0

            path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                bounds, radius, radius
            )

            if self._style == "primary":
                # Granola Emerald / Forest Green accent button
                if self._hovered:
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(
                        0.16, 0.80, 0.46, 1.0
                    ).setFill()
                else:
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(
                        0.13, 0.72, 0.41, 0.96
                    ).setFill()
                path.fill()
                txt_col = NSColor.whiteColor()
                font = NSFont.boldSystemFontOfSize_(11.5)
            elif self._style == "delete":
                if self._hovered:
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(
                        0.95, 0.25, 0.30, 0.28
                    ).setFill()
                    path.fill()
                    txt_col = NSColor.colorWithCalibratedRed_green_blue_alpha_(
                        1.0, 0.45, 0.48, 1.0
                    )
                else:
                    txt_col = NSColor.colorWithCalibratedRed_green_blue_alpha_(
                        0.60, 0.64, 0.70, 0.65
                    )
                font = NSFont.boldSystemFontOfSize_(12.0)
            else:
                # Secondary ghost/subtle button
                if self._hovered:
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(
                        1.0, 1.0, 1.0, 0.14
                    ).setFill()
                else:
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(
                        1.0, 1.0, 1.0, 0.08
                    ).setFill()
                path.fill()
                txt_col = NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.85, 0.88, 0.92, 0.95
                )
                font = NSFont.systemFontOfSize_(11.5)

            attrs = {
                NSFontAttributeName: font,
                NSForegroundColorAttributeName: txt_col,
            }
            ns_str = NSString.stringWithString_(self._title)
            sz = ns_str.sizeWithAttributes_(attrs)
            tx = (w - float(sz.width)) / 2.0
            ty = (h - float(sz.height)) / 2.0
            ns_str.drawAtPoint_withAttributes_(NSPoint(tx, ty), attrs)


try:
    GranolaStepRowView = objc.lookUpClass("GranolaStepRowView")
except objc.nosuchclass_error:

    class GranolaStepRowView(NSView):
        """Renders a single clean Granola-style step card with number pill, action summary, voice note, and delete button."""

        def initWithFrame_step_isLatest_onDelete_(
            self,
            frame: NSRect,
            step_data: Dict[str, Any],
            is_latest: bool,
            on_delete: Callable[[int], None],
        ):
            self = objc.super(GranolaStepRowView, self).initWithFrame_(frame)
            if self is not None:
                self._step = step_data
                self._is_latest = is_latest
                self._on_delete = on_delete
                self._build_subviews()
            return self

        def _build_subviews(self) -> None:
            w = float(self.bounds().size.width)
            h = float(self.bounds().size.height)
            step_num = int(self._step.get("step_number", 1))
            spoken = str(self._step.get("spoken_summary", "Action recorded"))
            app_name = str(self._step.get("app_name", "") or "")
            action = str(self._step.get("action", "click"))
            voice_note = self._step.get("voice_note")

            icon = ACTION_ICONS.get(action, "•")
            is_active_edit = action == "active_item"
            is_applied_edit = action == "edit_applied"

            # Top line Y position
            top_y = h - 24.0

            # 1. Step Number Badge Label
            num_lbl = NSTextField.alloc().initWithFrame_(
                NSRect(NSPoint(10.0, top_y), NSSize(24.0, 16.0))
            )
            num_lbl.setStringValue_(f"#{step_num}" if (is_active_edit or is_applied_edit) else f"{step_num}.")
            num_lbl.setFont_(NSFont.monospacedDigitSystemFontOfSize_weight_(11.0, 0.5))
            if is_active_edit:
                num_col = NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.22, 0.76, 0.98, 1.0
                )
            elif is_applied_edit or self._is_latest:
                num_col = NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.25, 0.90, 0.58, 0.95
                )
            else:
                num_col = NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.56, 0.62, 0.70, 0.85
                )
            num_lbl.setTextColor_(num_col)
            num_lbl.setBezeled_(False)
            num_lbl.setDrawsBackground_(False)
            num_lbl.setEditable_(False)
            num_lbl.setSelectable_(False)
            self.addSubview_(num_lbl)

            # 2. Main Step Action Text
            summary_str = f"{icon}  {spoken}"
            if app_name and app_name.lower() not in spoken.lower():
                summary_str += f"  ({app_name})"

            act_lbl = NSTextField.alloc().initWithFrame_(
                NSRect(NSPoint(34.0, top_y - 1.0), NSSize(w - 64.0, 18.0))
            )
            act_lbl.setStringValue_(summary_str)
            act_lbl.setFont_(
                NSFont.boldSystemFontOfSize_(12.0)
                if (self._is_latest or is_active_edit or is_applied_edit)
                else NSFont.systemFontOfSize_(12.0)
            )
            act_lbl.setTextColor_(
                NSColor.whiteColor()
                if (self._is_latest or is_active_edit)
                else NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.88, 0.92, 0.95, 0.95
                )
            )
            act_lbl.setBezeled_(False)
            act_lbl.setDrawsBackground_(False)
            act_lbl.setEditable_(False)
            act_lbl.setSelectable_(False)
            if hasattr(act_lbl, "setUsesSingleLineMode_"):
                act_lbl.setUsesSingleLineMode_(True)
            if hasattr(act_lbl, "cell") and hasattr(
                act_lbl.cell(), "setLineBreakMode_"
            ):
                act_lbl.cell().setLineBreakMode_(NSLineBreakByTruncatingTail)
            self.addSubview_(act_lbl)

            # 3. Delete (×) Button on the right
            del_btn = GranolaPillButtonView.alloc().initWithFrame_title_style_callback_(
                NSRect(NSPoint(w - 26.0, top_y - 1.0), NSSize(18.0, 18.0)),
                "×",
                "delete",
                lambda n=step_num: self._on_delete(n) if self._on_delete else None,
            )
            self.addSubview_(del_btn)

            # 4. Voice Note Sub-Row (or active recording prompt for active_item)
            sub_text = ""
            sub_col = NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.98, 0.78, 0.35, 0.95
            )
            if voice_note:
                prefix = "🎙️ REC: " if is_active_edit else "🎙️ "
                sub_text = f"{prefix}“{voice_note}”"
            elif is_active_edit:
                sub_text = "🎙️ Recording for this item — press Enter or say “Next change”"
                sub_col = NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.38, 0.78, 0.98, 0.92
                )

            if sub_text:
                note_lbl = NSTextField.alloc().initWithFrame_(
                    NSRect(NSPoint(34.0, 5.0), NSSize(w - 46.0, 16.0))
                )
                note_lbl.setStringValue_(sub_text)
                note_lbl.setFont_(NSFont.systemFontOfSize_(11.0))
                note_lbl.setTextColor_(sub_col)
                note_lbl.setBezeled_(False)
                note_lbl.setDrawsBackground_(False)
                note_lbl.setEditable_(False)
                note_lbl.setSelectable_(False)
                if hasattr(note_lbl, "setUsesSingleLineMode_"):
                    note_lbl.setUsesSingleLineMode_(True)
                if hasattr(note_lbl, "cell") and hasattr(
                    note_lbl.cell(), "setLineBreakMode_"
                ):
                    note_lbl.cell().setLineBreakMode_(NSLineBreakByTruncatingTail)
                self.addSubview_(note_lbl)

        def drawRect_(self, dirtyRect: NSRect) -> None:
            bounds = self.bounds()
            action = str(self._step.get("action", "click"))
            path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                bounds, 8.0, 8.0
            )
            if action == "active_item":
                # Sky-blue glowing active item card
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.06, 0.65, 0.91, 0.16
                ).setFill()
                path.fill()
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.18, 0.76, 0.98, 0.65
                ).setStroke()
                path.setLineWidth_(1.2)
                path.stroke()
            elif action == "edit_applied":
                # Completed item with subtle emerald checkmark tint
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.13, 0.78, 0.46, 0.10
                ).setFill()
                path.fill()
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.18, 0.85, 0.52, 0.32
                ).setStroke()
                path.setLineWidth_(1.0)
                path.stroke()
            elif self._is_latest:
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.16, 0.85, 0.50, 0.11
                ).setFill()
                path.fill()
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.20, 0.88, 0.54, 0.35
                ).setStroke()
                path.setLineWidth_(1.0)
                path.stroke()
            else:
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    1.0, 1.0, 1.0, 0.045
                ).setFill()
                path.fill()


try:
    GranolaTitleFieldDelegate = objc.lookUpClass("GranolaTitleFieldDelegate")
except objc.nosuchclass_error:

    class GranolaTitleFieldDelegate(objc.lookUpClass("NSObject")):
        """Delegate for the editable workflow title field."""

        def initWithCallback_panel_(
            self, on_rename: Callable[[str], None], panel: Any
        ):
            self = objc.super(GranolaTitleFieldDelegate, self).init()
            if self is not None:
                self._on_rename = on_rename
                self._panel = panel
            return self

        def controlTextDidChange_(self, notification) -> None:
            try:
                field = notification.object()
                if field and self._on_rename:
                    val = str(field.stringValue() or "").strip()
                    if val:
                        self._on_rename(val)
            except Exception:
                pass

        def controlTextDidEndEditing_(self, notification) -> None:
            try:
                field = notification.object()
                if field and self._on_rename:
                    val = str(field.stringValue() or "").strip()
                    if val:
                        self._on_rename(val)
                if self._panel:
                    self._panel.makeFirstResponder_(None)
            except Exception:
                pass


try:
    GranolaWindowMoveDelegate = objc.lookUpClass("GranolaWindowMoveDelegate")
except objc.nosuchclass_error:

    class GranolaWindowMoveDelegate(objc.lookUpClass("NSObject")):
        """Updates the self-click exclusion rectangle whenever the user drags the Granola card."""

        def initWithPanel_(self, card: "GranolaTrainingPanel"):
            self = objc.super(GranolaWindowMoveDelegate, self).init()
            if self is not None:
                self._card = card
            return self

        def windowDidMove_(self, notification) -> None:
            if self._card:
                self._card._sync_ignore_rect()


class GranolaTrainingPanel:
    """
    Granola-style floating training notepad card anchored in the top-right corner of the screen.
    Displays live captured OS steps, inline spoken voice annotations, editable workflow title,
    live elapsed timer, reactive waveform, and interactive Save / Cancel controls.
    """

    _INSTANCE: Optional["GranolaTrainingPanel"] = None
    _LOCK = threading.Lock()

    CARD_WIDTH = 356.0
    CARD_HEIGHT = 340.0

    @classmethod
    def get_instance(cls) -> "GranolaTrainingPanel":
        with cls._LOCK:
            if cls._INSTANCE is None:
                cls._INSTANCE = cls()
            return cls._INSTANCE

    def __init__(self):
        self._panel: Optional[NSPanel] = None
        self._root_view: Optional[NSView] = None
        self._effect_view: Optional[NSVisualEffectView] = None
        self._pulse_view: Optional[Any] = None
        self._timer_lbl: Optional[NSTextField] = None
        self._title_field: Optional[NSTextField] = None
        self._title_delegate: Optional[Any] = None
        self._win_delegate: Optional[Any] = None
        self._subtitle_lbl: Optional[NSTextField] = None
        self._feed_container: Optional[NSView] = None
        self._empty_state_view: Optional[NSView] = None
        self._voice_banner_lbl: Optional[NSTextField] = None
        self._save_btn: Optional[Any] = None
        self._cancel_btn: Optional[Any] = None
        self._new_change_btn: Optional[Any] = None
        self._hint_lbl: Optional[NSTextField] = None

        self._is_visible = False
        self._mode = "training"
        self._workflow_name = "Untitled Workflow"
        self._start_time = 0.0
        self._steps: List[Dict[str, Any]] = []
        self._pending_voice_note: Optional[str] = None
        self._tick_timer: Optional[threading.Timer] = None

        if not is_headless():
            if threading.current_thread() is threading.main_thread():
                self._init_window()
            else:
                AppHelper.callAfter(self._init_window)

    # ------------------------------------------------------------------
    # Window Initialization (Main Thread)
    # ------------------------------------------------------------------

    def _get_top_right_frame(self) -> NSRect:
        screen = NSScreen.mainScreen()
        if screen:
            vf = screen.visibleFrame()
            x = float(vf.origin.x + vf.size.width - self.CARD_WIDTH - 20.0)
            y = float(vf.origin.y + vf.size.height - self.CARD_HEIGHT - 14.0)
            try:
                from whole_body_compute.ui.unified_hud import UnifiedDynamicIslandHUD

                hud = UnifiedDynamicIslandHUD._instance
                if hud is not None:
                    hud_rect = hud._get_target_frame(hud.STANDARD_WIDTH, hud.STANDARD_HEIGHT)
                    hx = float(hud_rect.origin.x)
                    hy = float(hud_rect.origin.y)
                    hw = float(hud_rect.size.width)
                    if (hx + hw) > x and hy > float(vf.origin.y + vf.size.height * 0.5):
                        below_hud_y = hy - self.CARD_HEIGHT - 12.0
                        if below_hud_y >= float(vf.origin.y) + 20.0:
                            y = min(y, below_hud_y)
            except Exception:
                pass
            return NSRect(NSPoint(x, y), NSSize(self.CARD_WIDTH, self.CARD_HEIGHT))
        return NSRect(NSPoint(900.0, 500.0), NSSize(self.CARD_WIDTH, self.CARD_HEIGHT))

    def _init_window(self) -> None:
        if self._panel is not None:
            return
        try:
            rect = self._get_top_right_frame()
            style = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
            self._panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                rect,
                style,
                NSBackingStoreBuffered,
                False,
            )
            self._panel.setOpaque_(False)
            self._panel.setBackgroundColor_(NSColor.clearColor())
            self._panel.setHasShadow_(True)
            self._panel.setMovableByWindowBackground_(True)
            self._panel.setHidesOnDeactivate_(False)
            self._panel.setFloatingPanel_(True)
            self._panel.setBecomesKeyOnlyIfNeeded_(True)
            self._panel.setLevel_(NSStatusWindowLevel + 3)
            self._panel.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces
                | NSWindowCollectionBehaviorFullScreenAuxiliary
            )

            self._win_delegate = GranolaWindowMoveDelegate.alloc().initWithPanel_(self)
            self._panel.setDelegate_(self._win_delegate)

            w = self.CARD_WIDTH
            h = self.CARD_HEIGHT

            # Root rounded container
            self._root_view = NSView.alloc().initWithFrame_(
                NSRect(NSPoint(0.0, 0.0), NSSize(w, h))
            )
            self._root_view.setWantsLayer_(True)
            self._root_view.layer().setCornerRadius_(16.0)
            self._root_view.layer().setMasksToBounds_(True)
            self._root_view.layer().setBorderWidth_(1.2)
            self._root_view.layer().setBorderColor_(
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    1.0, 1.0, 1.0, 0.16
                ).CGColor()
            )
            self._root_view.layer().setBackgroundColor_(
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.07, 0.08, 0.11, 0.94
                ).CGColor()
            )

            # Frosted glass vibrancy layer
            self._effect_view = NSVisualEffectView.alloc().initWithFrame_(
                NSRect(NSPoint(0.0, 0.0), NSSize(w, h))
            )
            self._effect_view.setMaterial_(NSVisualEffectMaterialHUDWindow)
            self._effect_view.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
            self._effect_view.setState_(1)
            self._root_view.addSubview_(self._effect_view)

            # --- 1. Top Status Bar (Granola Recording Pill + Timer + Waveform) ---
            top_bar_y = h - 34.0
            self._pulse_view = GranolaHeaderPulseView.alloc().initWithFrame_(
                NSRect(NSPoint(14.0, top_bar_y), NSSize(w - 28.0, 22.0))
            )
            self._root_view.addSubview_(self._pulse_view)

            rec_badge = NSTextField.alloc().initWithFrame_(
                NSRect(NSPoint(36.0, top_bar_y + 2.0), NSSize(155.0, 16.0))
            )
            rec_badge.setStringValue_("PAL TEACH • CAPTURING")
            rec_badge.setFont_(NSFont.boldSystemFontOfSize_(10.0))
            rec_badge.setTextColor_(
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.98, 0.36, 0.40, 0.95
                )
            )
            rec_badge.setBezeled_(False)
            rec_badge.setDrawsBackground_(False)
            rec_badge.setEditable_(False)
            rec_badge.setSelectable_(False)
            self._root_view.addSubview_(rec_badge)

            self._timer_lbl = NSTextField.alloc().initWithFrame_(
                NSRect(NSPoint(w - 92.0, top_bar_y + 2.0), NSSize(48.0, 16.0))
            )
            self._timer_lbl.setStringValue_("00:00")
            self._timer_lbl.setAlignment_(NSTextAlignmentRight)
            self._timer_lbl.setFont_(
                NSFont.monospacedDigitSystemFontOfSize_weight_(11.0, 0.3)
            )
            self._timer_lbl.setTextColor_(
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.72, 0.76, 0.82, 0.90
                )
            )
            self._timer_lbl.setBezeled_(False)
            self._timer_lbl.setDrawsBackground_(False)
            self._timer_lbl.setEditable_(False)
            self._timer_lbl.setSelectable_(False)
            self._root_view.addSubview_(self._timer_lbl)

            # --- 2. Editable Workflow Title & Subtitle ---
            title_y = h - 62.0
            self._title_field = NSTextField.alloc().initWithFrame_(
                NSRect(NSPoint(14.0, title_y), NSSize(w - 28.0, 24.0))
            )
            self._title_field.setStringValue_(self._workflow_name)
            self._title_field.setFont_(NSFont.boldSystemFontOfSize_(15.5))
            self._title_field.setTextColor_(NSColor.whiteColor())
            self._title_field.setBezeled_(False)
            self._title_field.setDrawsBackground_(False)
            self._title_field.setEditable_(True)
            self._title_field.setSelectable_(True)
            self._title_field.setFocusRingType_(1)  # NSFocusRingTypeNone
            self._title_field.setToolTip_("Click to rename workflow")
            self._title_delegate = GranolaTitleFieldDelegate.alloc().initWithCallback_panel_(
                self._on_user_renamed_workflow, self._panel
            )
            self._title_field.setDelegate_(self._title_delegate)
            self._root_view.addSubview_(self._title_field)

            self._subtitle_lbl = NSTextField.alloc().initWithFrame_(
                NSRect(NSPoint(14.0, title_y - 18.0), NSSize(w - 28.0, 15.0))
            )
            self._subtitle_lbl.setStringValue_(
                "Perform actions on your Mac or speak out loud to add context notes"
            )
            self._subtitle_lbl.setFont_(NSFont.systemFontOfSize_(11.0))
            self._subtitle_lbl.setTextColor_(
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.58, 0.64, 0.72, 0.92
                )
            )
            self._subtitle_lbl.setBezeled_(False)
            self._subtitle_lbl.setDrawsBackground_(False)
            self._subtitle_lbl.setEditable_(False)
            self._subtitle_lbl.setSelectable_(False)
            self._root_view.addSubview_(self._subtitle_lbl)

            # Subtle divider line below header
            div_top = NSView.alloc().initWithFrame_(
                NSRect(NSPoint(14.0, title_y - 24.0), NSSize(w - 28.0, 1.0))
            )
            div_top.setWantsLayer_(True)
            div_top.layer().setBackgroundColor_(
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    1.0, 1.0, 1.0, 0.08
                ).CGColor()
            )
            self._root_view.addSubview_(div_top)

            # --- 3. Live Chronological Step Feed Area ---
            feed_y = 52.0
            feed_h = (title_y - 28.0) - feed_y
            self._feed_container = NSView.alloc().initWithFrame_(
                NSRect(NSPoint(14.0, feed_y), NSSize(w - 28.0, feed_h))
            )
            self._root_view.addSubview_(self._feed_container)

            # --- 4. Footer Bar (Cancel & Save Workflow Buttons) ---
            div_bot = NSView.alloc().initWithFrame_(
                NSRect(NSPoint(14.0, 46.0), NSSize(w - 28.0, 1.0))
            )
            div_bot.setWantsLayer_(True)
            div_bot.layer().setBackgroundColor_(
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    1.0, 1.0, 1.0, 0.08
                ).CGColor()
            )
            self._root_view.addSubview_(div_bot)

            self._cancel_btn = GranolaPillButtonView.alloc().initWithFrame_title_style_callback_(
                NSRect(NSPoint(14.0, 11.0), NSSize(80.0, 26.0)),
                "Cancel",
                "secondary",
                self._on_cancel_clicked,
            )
            self._root_view.addSubview_(self._cancel_btn)

            self._hint_lbl = NSTextField.alloc().initWithFrame_(
                NSRect(NSPoint(98.0, 16.0), NSSize(120.0, 15.0))
            )
            self._hint_lbl.setStringValue_("or say “stop training”")
            self._hint_lbl.setAlignment_(NSTextAlignmentCenter)
            self._hint_lbl.setFont_(NSFont.systemFontOfSize_(10.0))
            self._hint_lbl.setTextColor_(
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.50, 0.56, 0.64, 0.85
                )
            )
            self._hint_lbl.setBezeled_(False)
            self._hint_lbl.setDrawsBackground_(False)
            self._hint_lbl.setEditable_(False)
            self._hint_lbl.setSelectable_(False)
            self._root_view.addSubview_(self._hint_lbl)

            self._new_change_btn = GranolaPillButtonView.alloc().initWithFrame_title_style_callback_(
                NSRect(NSPoint(100.0, 11.0), NSSize(122.0, 26.0)),
                "+ New Change (↵)",
                "secondary",
                self._on_new_change_clicked,
            )
            self._new_change_btn.setHidden_(True)
            self._root_view.addSubview_(self._new_change_btn)

            self._save_btn = GranolaPillButtonView.alloc().initWithFrame_title_style_callback_(
                NSRect(NSPoint(w - 126.0, 11.0), NSSize(112.0, 26.0)),
                "✓ Save Workflow",
                "primary",
                self._on_save_clicked,
            )
            self._root_view.addSubview_(self._save_btn)

            self._panel.setContentView_(self._root_view)
            self._panel.orderOut_(None)
            self._rebuild_feed_ui()
        except Exception as e:
            print(f"[GranolaTrainingPanel] Init error: {e}")

    # ------------------------------------------------------------------
    # Self-Click Exclusion (Quartz Top-Left Screen Coordinates)
    # ------------------------------------------------------------------

    def _sync_ignore_rect(self) -> None:
        """Convert Cocoa bottom-left window frame to Quartz top-left coordinates and register with active recorder/editor."""
        try:
            from whole_body_compute.integrations.pal_harness import PalHarness

            if not self._panel or not self._is_visible:
                trainer = PalHarness.get_active_trainer()
                if trainer and getattr(trainer, "recorder", None):
                    trainer.recorder.clear_ignore_rects()
                return

            frame = self._panel.frame()
            screen = NSScreen.mainScreen()
            screen_h = float(screen.frame().size.height) if screen else 1080.0
            qx = float(frame.origin.x)
            qy = screen_h - (float(frame.origin.y) + float(frame.size.height))
            qw = float(frame.size.width)
            qh = float(frame.size.height)
            trainer = PalHarness.get_active_trainer()
            if trainer and getattr(trainer, "recorder", None):
                trainer.recorder.set_ignore_rect(qx, qy, qw, qh)
            editor = PalHarness.get_active_editor()
            if editor and hasattr(editor, "set_ignore_rect"):
                editor.set_ignore_rect(qx, qy, qw, qh)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Feed Rendering
    # ------------------------------------------------------------------

    def _rebuild_feed_ui(self) -> None:
        if not self._feed_container:
            return
        for sub in list(self._feed_container.subviews()):
            sub.removeFromSuperview()

        fw = float(self._feed_container.bounds().size.width)
        fh = float(self._feed_container.bounds().size.height)

        total = len(self._steps)
        if self._subtitle_lbl:
            if total == 0:
                if self._pending_voice_note:
                    self._subtitle_lbl.setStringValue_(
                        f"🎙️ Note queued: “{self._pending_voice_note[:42]}”"
                    )
                elif self._mode == "editing":
                    self._subtitle_lbl.setStringValue_(
                        "Highlight text on your page & say what to change (100% Local)"
                    )
                else:
                    self._subtitle_lbl.setStringValue_(
                        "Waiting for first action • Click, type, or narrate out loud"
                    )
            else:
                if self._mode == "editing":
                    plural = "edit" if total == 1 else "edits"
                    self._subtitle_lbl.setStringValue_(
                        f"{total} {plural} in session • Click × or Undo to revert any change"
                    )
                else:
                    plural = "step" if total == 1 else "steps"
                    self._subtitle_lbl.setStringValue_(
                        f"{total} {plural} captured • Click title to rename or × to remove a step"
                    )

        if total == 0:
            # Render clean Granola-style empty notepad state
            empty_box = NSView.alloc().initWithFrame_(
                NSRect(NSPoint(0.0, 12.0), NSSize(fw, fh - 24.0))
            )
            empty_box.setWantsLayer_(True)
            empty_box.layer().setCornerRadius_(10.0)
            empty_box.layer().setBackgroundColor_(
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    1.0, 1.0, 1.0, 0.03
                ).CGColor()
            )
            empty_box.layer().setBorderWidth_(1.0)
            empty_box.layer().setBorderColor_(
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    1.0, 1.0, 1.0, 0.06
                ).CGColor()
            )

            p1 = NSTextField.alloc().initWithFrame_(
                NSRect(NSPoint(16.0, (fh - 24.0) / 2.0 + 10.0), NSSize(fw - 32.0, 20.0))
            )
            p1.setStringValue_(
                "✏️ Select text & speak your edit..."
                if self._mode == "editing"
                else "👀 Watching your demonstration..."
            )
            p1.setAlignment_(NSTextAlignmentCenter)
            p1.setFont_(NSFont.boldSystemFontOfSize_(12.5))
            p1.setTextColor_(
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.85, 0.88, 0.94, 0.95
                )
            )
            p1.setBezeled_(False)
            p1.setDrawsBackground_(False)
            p1.setEditable_(False)
            p1.setSelectable_(False)
            empty_box.addSubview_(p1)

            sub_msg = (
                f"🎙️ “{self._pending_voice_note}”"
                if self._pending_voice_note
                else (
                    "Example: Select an em-dash '—' and say “Change this to a colon”"
                    if self._mode == "editing"
                    else "Every app switch, click, shortcut & spoken note will appear here live."
                )
            )
            p2 = NSTextField.alloc().initWithFrame_(
                NSRect(NSPoint(16.0, (fh - 24.0) / 2.0 - 22.0), NSSize(fw - 32.0, 30.0))
            )
            p2.setStringValue_(sub_msg)
            p2.setAlignment_(NSTextAlignmentCenter)
            p2.setFont_(NSFont.systemFontOfSize_(11.0))
            p2.setTextColor_(
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.98, 0.78, 0.35, 0.95
                )
                if self._pending_voice_note
                else NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.55, 0.60, 0.68, 0.90
                )
            )
            p2.setBezeled_(False)
            p2.setDrawsBackground_(False)
            p2.setEditable_(False)
            p2.setSelectable_(False)
            empty_box.addSubview_(p2)

            self._feed_container.addSubview_(empty_box)
            return

        # Render the most recent steps fitting cleanly in the notepad feed (chronological top-to-bottom)
        # Calculate heights (30px standard, 46px if step has a voice note)
        visible_steps: List[Dict[str, Any]] = []
        used_h = 0.0
        gap = 5.0
        for s in reversed(self._steps):
            row_h = 46.0 if (s.get("voice_note") or s.get("action") == "active_item") else 30.0
            if used_h + row_h > fh and visible_steps:
                break
            visible_steps.insert(0, s)
            used_h += row_h + gap

        hidden_count = total - len(visible_steps)
        cursor_y = fh

        if hidden_count > 0:
            more_lbl = NSTextField.alloc().initWithFrame_(
                NSRect(NSPoint(4.0, cursor_y - 16.0), NSSize(fw - 8.0, 14.0))
            )
            more_lbl.setStringValue_(
                f"↑ {hidden_count} earlier {'step' if hidden_count == 1 else 'steps'} recorded"
            )
            more_lbl.setFont_(NSFont.systemFontOfSize_(10.0))
            more_lbl.setTextColor_(
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.50, 0.56, 0.64, 0.85
                )
            )
            more_lbl.setBezeled_(False)
            more_lbl.setDrawsBackground_(False)
            more_lbl.setEditable_(False)
            more_lbl.setSelectable_(False)
            self._feed_container.addSubview_(more_lbl)
            cursor_y -= 18.0

        for idx, s in enumerate(visible_steps):
            is_latest = idx == len(visible_steps) - 1
            row_h = 46.0 if (s.get("voice_note") or s.get("action") == "active_item") else 30.0
            cursor_y -= row_h
            if cursor_y < -4.0:
                break
            row_view = GranolaStepRowView.alloc().initWithFrame_step_isLatest_onDelete_(
                NSRect(NSPoint(0.0, max(0.0, cursor_y)), NSSize(fw, row_h)),
                s,
                is_latest,
                self._on_delete_step,
            )
            self._feed_container.addSubview_(row_view)
            cursor_y -= gap

    # ------------------------------------------------------------------
    # Interactive Callbacks
    # ------------------------------------------------------------------

    def _on_user_renamed_workflow(self, new_name: str) -> None:
        clean = (new_name or "").strip()
        if not clean:
            return
        self._workflow_name = clean
        try:
            from whole_body_compute.integrations.pal_harness import PalHarness

            trainer = PalHarness.get_active_trainer()
            if trainer and hasattr(trainer, "rename_workflow"):
                trainer.rename_workflow(clean)
        except Exception:
            pass

    def _on_delete_step(self, step_number: int) -> None:
        try:
            from pathlib import Path
            from whole_body_compute.integrations.pal_harness import PalHarness

            editor = PalHarness.get_active_editor()
            trainer = PalHarness.get_active_trainer()
            if editor and hasattr(editor, "remove_step"):
                updated = editor.remove_step(step_number)
                self._steps = [
                    {
                        "step_number": r.step_number,
                        "action": r.action,
                        "app_name": Path(r.file_path).name if r.file_path else r.app_name,
                        "spoken_summary": r.spoken_summary,
                        "voice_note": r.instruction if r.instruction else None,
                    }
                    for r in updated
                ]
            elif trainer and hasattr(trainer, "remove_step"):
                updated = trainer.remove_step(step_number)
                self._steps = [
                    {
                        "step_number": s.step_number,
                        "action": s.action,
                        "app_name": s.app_name,
                        "spoken_summary": s.spoken_summary,
                        "voice_note": s.voice_note,
                    }
                    for s in updated
                ]
            else:
                self._steps = [
                    s for s in self._steps if int(s.get("step_number", 0)) != step_number
                ]
                for i, s in enumerate(self._steps, start=1):
                    s["step_number"] = i
        except Exception:
            pass

        def _refresh():
            self._rebuild_feed_ui()

        if threading.current_thread() is threading.main_thread():
            _refresh()
        else:
            AppHelper.callAfter(_refresh)

    def _on_new_change_clicked(self) -> None:
        """Commit the currently active edit item and open the next numbered item."""
        try:
            from whole_body_compute.integrations.pal_harness import PalHarness

            if PalHarness.is_editing_active():
                PalHarness.next_edit_item()
        except Exception:
            pass

    def _on_save_clicked(self) -> None:
        try:
            if self._title_field:
                edited_name = str(self._title_field.stringValue() or "").strip()
                if edited_name:
                    self._on_user_renamed_workflow(edited_name)
        except Exception:
            pass

        try:
            from whole_body_compute.integrations.pal_harness import PalHarness

            if PalHarness.is_editing_active():
                PalHarness.stop_edit_mode()
            elif PalHarness.is_training_active():
                PalHarness.stop_training()
            else:
                self.hide()
        except Exception:
            self.hide()

    def _on_cancel_clicked(self) -> None:
        try:
            from whole_body_compute.integrations.pal_harness import PalHarness

            if PalHarness.is_editing_active():
                PalHarness.undo_edit()
            elif PalHarness.is_training_active():
                PalHarness.cancel_training()
            else:
                self.hide()
        except Exception:
            self.hide()

    # ------------------------------------------------------------------
    # Public Lifecycle API
    # ------------------------------------------------------------------

    def show_training(self, workflow_name: str) -> None:
        """Open the Granola-style training notepad card in the top-right corner."""
        self._mode = "training"
        self._workflow_name = (workflow_name or "Untitled Workflow").strip().title()
        self._start_time = time.time()
        self._steps = []
        self._pending_voice_note = None
        self._is_visible = True
        self._start_tick_loop()

        if is_headless():
            return

        def _present():
            if not self._panel:
                self._init_window()
            if not self._panel:
                return
            try:
                self._panel.setFrame_display_(self._get_top_right_frame(), True)
            except Exception:
                pass
            if self._title_field:
                self._title_field.setStringValue_(self._workflow_name)
            if self._timer_lbl:
                self._timer_lbl.setStringValue_("00:00")
            if self._save_btn:
                self._save_btn._title = "✓ Save Workflow"
                self._save_btn.setNeedsDisplay_(True)
            if self._cancel_btn:
                self._cancel_btn._title = "Cancel"
                self._cancel_btn.setNeedsDisplay_(True)
            if self._new_change_btn:
                self._new_change_btn.setHidden_(True)
            if self._hint_lbl:
                self._hint_lbl.setHidden_(False)
            self._rebuild_feed_ui()
            self._panel.orderFrontRegardless()
            self._panel.setAlphaValue_(1.0)
            self._sync_ignore_rect()

        if threading.current_thread() is threading.main_thread():
            _present()
        else:
            AppHelper.callAfter(_present)

    def show_edit_mode(self, session_name: str = "Live Website Edit") -> None:
        """Open the Granola-style notepad card in Pal Edit Mode ('Select & Speak')."""
        self._mode = "editing"
        self._workflow_name = f"✏️ {(session_name or 'Live Website Edit').strip()}"
        self._start_time = time.time()
        self._steps = []
        self._pending_voice_note = None
        self._is_visible = True
        self._start_tick_loop()

        if is_headless():
            return

        def _present():
            if not self._panel:
                self._init_window()
            if not self._panel:
                return
            try:
                self._panel.setFrame_display_(self._get_top_right_frame(), True)
            except Exception:
                pass
            if self._title_field:
                self._title_field.setStringValue_(self._workflow_name)
            if self._timer_lbl:
                self._timer_lbl.setStringValue_("00:00")
            if self._save_btn:
                self._save_btn._title = "✓ Done"
                self._save_btn.setNeedsDisplay_(True)
            if self._cancel_btn:
                self._cancel_btn._title = "↩ Undo"
                self._cancel_btn.setNeedsDisplay_(True)
            if self._new_change_btn:
                self._new_change_btn._title = "+ New Change (↵)"
                self._new_change_btn.setHidden_(False)
                self._new_change_btn.setNeedsDisplay_(True)
            if self._hint_lbl:
                self._hint_lbl.setHidden_(True)
            self._rebuild_feed_ui()
            self._panel.orderFrontRegardless()
            self._panel.setAlphaValue_(1.0)
            self._sync_ignore_rect()

        if threading.current_thread() is threading.main_thread():
            _present()
        else:
            AppHelper.callAfter(_present)

    def add_or_update_step(
        self,
        step_number: int,
        action: str,
        spoken_summary: str,
        app_name: str = "",
        voice_note: Optional[str] = None,
    ) -> None:
        """Append a newly captured step (or update existing step) in the live Granola feed."""
        self._pending_voice_note = None
        entry = {
            "step_number": int(step_number),
            "action": str(action or "click"),
            "app_name": str(app_name or ""),
            "spoken_summary": str(spoken_summary or "Action recorded"),
            "voice_note": voice_note,
        }
        # Replace if step_number already exists, otherwise append
        replaced = False
        for i, existing in enumerate(self._steps):
            if int(existing.get("step_number", -1)) == int(step_number):
                self._steps[i] = entry
                replaced = True
                break
        if not replaced:
            self._steps.append(entry)

        if is_headless():
            return

        def _update():
            self._rebuild_feed_ui()

        if threading.current_thread() is threading.main_thread():
            _update()
        else:
            AppHelper.callAfter(_update)

    def record_voice_note(self, note: str) -> None:
        """Attach a spoken explanation inline to the latest step (or show as pending note)."""
        clean = (note or "").strip()
        if not clean:
            return
        if self._steps:
            prev = self._steps[-1].get("voice_note")
            self._steps[-1]["voice_note"] = f"{prev} | {clean}" if prev else clean
        else:
            self._pending_voice_note = (
                f"{self._pending_voice_note} | {clean}"
                if self._pending_voice_note
                else clean
            )

        if is_headless():
            return

        def _update():
            self._rebuild_feed_ui()

        if threading.current_thread() is threading.main_thread():
            _update()
        else:
            AppHelper.callAfter(_update)

    def update_audio_level(
        self, energy: float, prob: float = 0.0, is_speech: bool = False
    ) -> None:
        """Feed real-time microphone energy to the header's 5-bar waveform."""
        if not self._is_visible or is_headless():
            return

        def _do():
            if self._pulse_view and self._panel and self._panel.isVisible():
                self._pulse_view.setAudioLevel_prob_speech_(energy, prob, is_speech)

        if threading.current_thread() is threading.main_thread():
            _do()
        else:
            AppHelper.callAfter(_do)

    def hide(self) -> None:
        """Close the Granola training notepad card."""
        self._is_visible = False
        self._stop_tick_loop()
        if is_headless():
            return

        def _do_hide():
            if self._panel:
                self._panel.makeFirstResponder_(None)
                self._panel.orderOut_(None)

        if threading.current_thread() is threading.main_thread():
            _do_hide()
        else:
            AppHelper.callAfter(_do_hide)

    # ------------------------------------------------------------------
    # Live Elapsed Timer & Pulse Loop
    # ------------------------------------------------------------------

    def _start_tick_loop(self) -> None:
        self._stop_tick_loop()

        def _tick():
            if not self._is_visible:
                return
            elapsed = max(0.0, time.time() - self._start_time)
            mins = int(elapsed) // 60
            secs = int(elapsed) % 60
            time_str = f"{mins:02d}:{secs:02d}"

            if not is_headless():

                def _ui_tick():
                    if not self._is_visible or not self._panel:
                        return
                    if self._timer_lbl:
                        self._timer_lbl.setStringValue_(time_str)
                    if self._pulse_view:
                        self._pulse_view.tickPulse()

                AppHelper.callAfter(_ui_tick)

            if self._is_visible:
                self._tick_timer = threading.Timer(0.25, _tick)
                self._tick_timer.daemon = True
                self._tick_timer.start()

        self._tick_timer = threading.Timer(0.25, _tick)
        self._tick_timer.daemon = True
        self._tick_timer.start()

    def _stop_tick_loop(self) -> None:
        if self._tick_timer:
            try:
                self._tick_timer.cancel()
            except Exception:
                pass
            self._tick_timer = None


def _run_standalone_preview() -> None:
    """
    Launch the real native macOS GranolaTrainingPanel on screen with sample steps and voice notes
    so the user can inspect and iterate on the exact Cocoa UI design live.
    """
    from AppKit import NSApplicationActivationPolicyAccessory
    from PyObjCTools import AppHelper

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    panel = GranolaTrainingPanel.get_instance()
    panel.show_training("Client Report Upload")

    # Populate realistic sample steps after a brief beat so the user sees the live feed populate
    sample_steps = [
        (
            1,
            "switch_app",
            "Switched to Google Chrome",
            "Google Chrome",
            "First focus Chrome on the Q4 Drive folder",
        ),
        (
            2,
            "right_click",
            "Right-clicked Q4_SFR_Portfolio_42.csv",
            "Google Chrome",
            None,
        ),
        (
            3,
            "click",
            "Clicked Rename",
            "Google Chrome",
            None,
        ),
        (
            4,
            "press_key",
            "Pressed Backspace",
            "Google Chrome",
            "Strip the trailing count suffix before saving",
        ),
    ]

    def _populate():
        for num, act, summary, app_name, note in sample_steps:
            panel.add_or_update_step(
                step_number=num,
                action=act,
                spoken_summary=summary,
                app_name=app_name,
                voice_note=note,
            )

    threading.Timer(0.4, lambda: AppHelper.callAfter(_populate)).start()

    # Override Save / Cancel in standalone preview mode to cleanly exit the preview process
    orig_save = panel._on_save_clicked
    orig_cancel = panel._on_cancel_clicked

    def _preview_exit():
        panel.hide()
        app.terminate_(None)

    panel._on_save_clicked = _preview_exit
    panel._on_cancel_clicked = _preview_exit

    AppHelper.runEventLoop()


if __name__ == "__main__":
    _run_standalone_preview()

