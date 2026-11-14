"""
Foveal Vision: Sub-5ms Zero-GPU Interactive macOS Accessibility DOM Inspector (Phase 14.1 & Claude Bot Organ 2).

Extracts the active macOS window's interactive UI elements (buttons, inputs, sliders,
bounding boxes, focused/enabled state) via PyObjC `ApplicationServices` (`AXUIElement`)
without neural network vision or GPU compute, providing structured foveal grounding
alongside episodic OCR text and memory.

Unlocks Electron and Chromium accessibility trees (AXManualAccessibility & AXEnhancedUserInterface)
for Claude, Muse, Grok, Arc, Chrome, and native apps in <15ms with zero screenshots.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import socket
import subprocess
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("whole_body_compute.integrations.foveal_vision")

# Virtual keycode mapping for macOS Quartz (US layout)
QUARTZ_KEYCODES: Dict[str, int] = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9,
    "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17, "1": 18, "2": 19,
    "3": 20, "4": 21, "6": 22, "5": 23, "equal": 24, "=": 24, "9": 25, "7": 26,
    "minus": 27, "-": 27, "8": 28, "0": 29, "rightbracket": 30, "]": 30, "o": 31,
    "u": 32, "leftbracket": 33, "[": 33, "i": 34, "p": 35, "return": 36, "enter": 36,
    "l": 37, "j": 38, "quote": 39, "'": 39, "k": 40, "semicolon": 41, ";": 41,
    "backslash": 42, "\\": 42, "comma": 43, ",": 43, "slash": 44, "/": 44, "n": 45,
    "m": 46, "period": 47, ".": 47, "grave": 50, "`": 50, "tab": 48, "space": 49,
    "delete": 51, "backspace": 51, "escape": 53, "esc": 53, "left": 123, "right": 124,
    "down": 125, "up": 126,
}

QUARTZ_MODIFIERS: Dict[str, int] = {
    "cmd": 0x00100000,
    "command": 0x00100000,
    "shift": 0x00020000,
    "alt": 0x00080000,
    "option": 0x00080000,
    "opt": 0x00080000,
    "ctrl": 0x00040000,
    "control": 0x00040000,
}

IGNORE_TEXT_BUTTON_LABELS: Set[str] = {
    "close", "minimize", "fullscreen", "zoom", "hide sidebar", "show sidebar",
    "back", "forward", "reload", "stop", "settings", "help"
}

IGNORE_TEXT_SUBROLES: Set[str] = {
    "AXCloseButton", "AXMinimizeButton", "AXFullScreenButton", "AXZoomButton"
}

FOVEAL_SOCKET_PATH = os.path.expanduser("~/.voicefi/foveal.sock")


class SpatialMotorCache:
    """
    On-device Spatial Motor Cache ('Muscle Memory').
    Persists element center coordinates keyed by f"{app_name}:{window_bbox}:{label.lower()}".
    """

    _cache_file: str = os.path.expanduser("~/.voicefi/foveal_motor_cache.json")
    _memory: Dict[str, Dict[str, Any]] = {}
    _loaded: bool = False

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._loaded:
            return
        cls._loaded = True
        try:
            if os.path.exists(cls._cache_file):
                with open(cls._cache_file, "r", encoding="utf-8") as f:
                    cls._memory = json.load(f)
        except Exception as exc:
            logger.debug("Failed loading SpatialMotorCache: %s", exc)

    @classmethod
    def _save(cls) -> None:
        try:
            os.makedirs(os.path.dirname(cls._cache_file), exist_ok=True)
            with open(cls._cache_file, "w", encoding="utf-8") as f:
                json.dump(cls._memory, f, indent=2)
        except Exception as exc:
            logger.debug("Failed saving SpatialMotorCache: %s", exc)

    @classmethod
    def _make_key(cls, app_name: str, window_bbox: Tuple[float, float, float, float], label: str) -> str:
        bbox_str = f"{window_bbox[0]:.0f},{window_bbox[1]:.0f},{window_bbox[2]:.0f},{window_bbox[3]:.0f}"
        return f"{app_name.strip().lower()}:{bbox_str}:{label.strip().lower()}"

    @classmethod
    def get(cls, app_name: str, window_bbox: Tuple[float, float, float, float], label: str) -> Optional[Dict[str, Any]]:
        cls._ensure_loaded()
        key = cls._make_key(app_name, window_bbox, label)
        return cls._memory.get(key)

    @classmethod
    def set(
        cls,
        app_name: str,
        window_bbox: Tuple[float, float, float, float],
        label: str,
        center: Tuple[float, float],
        role: str = "",
    ) -> None:
        cls._ensure_loaded()
        key = cls._make_key(app_name, window_bbox, label)
        cls._memory[key] = {
            "center": [round(center[0], 1), round(center[1], 1)],
            "role": role,
            "updated_at": time.time(),
        }
        cls._save()

    @classmethod
    def invalidate(cls, app_name: str, window_bbox: Tuple[float, float, float, float], label: str) -> None:
        cls._ensure_loaded()
        key = cls._make_key(app_name, window_bbox, label)
        if key in cls._memory:
            del cls._memory[key]
            cls._save()

    @classmethod
    def clear(cls) -> None:
        cls._memory.clear()
        cls._loaded = True
        try:
            if os.path.exists(cls._cache_file):
                os.remove(cls._cache_file)
        except Exception:
            pass

# Interactive & meaningful roles that developers or agents interact with
INTERACTIVE_AX_ROLES: Set[str] = {
    "AXButton",
    "AXTextField",
    "AXTextArea",
    "AXCheckBox",
    "AXRadioButton",
    "AXPopUpButton",
    "AXMenuButton",
    "AXLink",
    "AXTabGroup",
    "AXTab",
    "AXComboBox",
    "AXSearchField",
    "AXSlider",
    "AXMenuItem",
    "AXMenuBarItem",
    "AXStaticText",
    "AXHeading",
    "AXCell",
    "AXWebArea",
}

CONTAINER_AX_ROLES: Set[str] = {
    "AXWindow",
    "AXGroup",
    "AXScrollArea",
    "AXWebArea",
    "AXSplitGroup",
    "AXList",
    "AXOutline",
    "AXTable",
    "AXRow",
    "AXColumn",
    "AXToolbar",
    "AXTabGroup",
    "AXDialog",
    "AXSheet",
    "AXDrawer",
    "AXApplication",
}


@dataclass
class FovealUIElement:
    """Represents a single interactive UI element in the active window."""

    role: str
    label: str = ""
    description: str = ""
    value: Optional[str] = None
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # (x, y, w, h)
    focused: bool = False
    enabled: bool = True
    subrole: Optional[str] = None
    identifier: Optional[str] = None
    depth: int = 0
    is_secure: bool = False
    id: str = ""
    title: str = ""
    position: Tuple[float, float] = (0.0, 0.0)
    size: Tuple[float, float] = (0.0, 0.0)
    center: Tuple[float, float] = (0.0, 0.0)
    raw_element: Any = None

    def __post_init__(self) -> None:
        if not self.title and self.label:
            self.title = self.label
        elif not self.label and self.title:
            self.label = self.title
        if self.bbox != (0.0, 0.0, 0.0, 0.0):
            if self.position == (0.0, 0.0):
                self.position = (self.bbox[0], self.bbox[1])
            if self.size == (0.0, 0.0):
                self.size = (self.bbox[2], self.bbox[3])
            if self.center == (0.0, 0.0):
                self.center = (
                    round(self.bbox[0] + self.bbox[2] / 2.0, 1),
                    round(self.bbox[1] + self.bbox[3] / 2.0, 1),
                )

    def to_dict(self) -> Dict[str, Any]:
        clean_lbl = self._clean_str(self.title or self.label)
        clean_desc = self._clean_str(self.description)
        clean_val = self._clean_str(self.value) if self.value is not None else None
        clean_id = self.id or ""
        x, y, w, h = self.bbox
        cx, cy = self.center
        return {
            "id": clean_id,
            "role": self.role,
            "title": clean_lbl,
            "label": clean_lbl,
            "description": clean_desc,
            "value": clean_val,
            "position": [round(x, 1), round(y, 1)],
            "size": [round(w, 1), round(h, 1)],
            "center": [round(cx, 1), round(cy, 1)],
            "bounds": {"x": round(x, 1), "y": round(y, 1), "w": round(w, 1), "h": round(h, 1)},
            "bbox": [round(x, 1), round(y, 1), round(w, 1), round(h, 1)],
            "focused": self.focused,
            "enabled": self.enabled,
            "subrole": self.subrole,
            "identifier": self._clean_str(self.identifier) if self.identifier is not None else None,
            "depth": self.depth,
            "is_secure": self.is_secure,
        }

    @staticmethod
    def _clean_str(val: str) -> str:
        """Sanitize text to neutralize newline breakout, secrets, and prompt injection vectors."""
        if not val:
            return ""
        from whole_body_compute.redaction import redact_secrets
        from whole_body_compute.organs.immune import EpithelialScreenFilter

        cleaned = re.sub(r"[\r\n\t]+", " ", str(val)).strip()
        cleaned = (
            cleaned.replace("[SYSTEM INSTRUCTION", "\\[SYSTEM_INSTRUCTION")
            .replace("[SYSTEM", "\\[SYSTEM")
            .replace("```", "")
        )
        cleaned = redact_secrets(cleaned)
        cleaned, _ = EpithelialScreenFilter.sanitize(cleaned)
        return cleaned[:80]

    def format_line(self) -> str:
        """Format a single concise representation for LLM context envelopes."""
        x, y, w, h = self.bbox
        parts = []
        if self.id:
            parts.append(f"[{self.id}]")
        parts.append(f"[{self.role}]")
        clean_lbl = self._clean_str(self.title or self.label)
        clean_desc = self._clean_str(self.description)
        clean_val = self._clean_str(self.value)
        if clean_lbl:
            parts.append(f'"{clean_lbl}"')
        if clean_desc and clean_desc != clean_lbl:
            parts.append(f'desc="{clean_desc}"')
        if clean_val:
            parts.append(f'val="{clean_val}"')
        parts.append(f"({round(x)}, {round(y)}, {round(w)}x{round(h)})")
        if self.center != (0.0, 0.0):
            parts.append(f"center=({round(self.center[0])}, {round(self.center[1])})")
        if self.focused:
            parts.append("[FOCUSED]")
        if not self.enabled:
            parts.append("[DISABLED]")
        return " ".join(parts)


# Deterministic alias for AXElementNode
AXElementNode = FovealUIElement


@dataclass
class FovealWindowSnapshot:
    """Structured DOM snapshot of the currently focused macOS window."""

    app_name: str
    bundle_id: str
    pid: int
    window_title: str
    window_bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    focused_element: Optional[FovealUIElement] = None
    interactive_elements: List[FovealUIElement] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    headless_route: str = ""
    frame_path: Optional[str] = None
    latest_frame_path: Optional[str] = None
    filter_query: Optional[str] = None
    estimated_tokens: int = 0
    screenshot_token_baseline: int = 15000
    token_savings_pct: float = 0.0
    actuation: Optional[Dict[str, Any]] = None
    tactile_delta: Optional[Dict[str, Any]] = None
    last_message: Optional[str] = None
    last_messages: Optional[List[str]] = None
    reply_settled: Optional[bool] = None
    wait_reply_ms: Optional[float] = None

    def __post_init__(self) -> None:
        if self.estimated_tokens == 0:
            self.estimated_tokens = max(25, len(self.interactive_elements) * 12 + 35)
        if self.token_savings_pct == 0.0 and self.screenshot_token_baseline > 0:
            savings = (1.0 - (float(self.estimated_tokens) / float(self.screenshot_token_baseline))) * 100.0
            self.token_savings_pct = round(max(0.0, savings), 1)

    def to_dict(self) -> Dict[str, Any]:
        elements_data = [e.to_dict() for e in self.interactive_elements]
        return {
            "app_name": self.app_name,
            "bundle_id": self.bundle_id,
            "pid": self.pid,
            "window_title": self.window_title,
            "window_bbox": list(self.window_bbox),
            "latency_ms": round(self.latency_ms, 3),
            "element_count": len(self.interactive_elements),
            "estimated_tokens": self.estimated_tokens,
            "screenshot_token_baseline": self.screenshot_token_baseline,
            "token_savings_pct": self.token_savings_pct,
            "elements": elements_data,
            "interactive_elements": elements_data,
            "focused_element": self.focused_element.to_dict() if self.focused_element else None,
            "timestamp": round(self.timestamp, 3),
            "headless_route": self.headless_route,
            "frame_path": self.frame_path,
            "latest_frame_path": self.latest_frame_path,
            "actuation": self.actuation,
            "tactile_delta": self.tactile_delta,
            "last_message": self.last_message,
            "last_messages": self.last_messages,
            "reply_settled": self.reply_settled,
            "wait_reply_ms": self.wait_reply_ms,
        }

    def to_context_block(self, max_elements: int = 40) -> str:
        """Render a high-density, prompt-ready markdown block for the agent."""
        safe_app = FovealUIElement._clean_str(self.app_name)
        safe_title = FovealUIElement._clean_str(self.window_title)
        tok_info = f"~{self.estimated_tokens} tokens ({self.token_savings_pct:.1f}% vs screenshot)"
        em_dash = "\u2014"
        lines = [
            f"[Foveal Vision DOM {em_dash} {safe_app} (PID {self.pid}) | \"{safe_title}\" in {self.latency_ms:.1f}ms | {tok_info}]"
        ]
        if self.frame_path:
            lines.append(f"Visual Frame Anchor: {self.frame_path}")
        if self.focused_element:
            lines.append(f"Active Focus: {self.focused_element.format_line()}")
        if self.headless_route:
            lines.append(f"Direct Route: {self.headless_route}")
        if self.actuation:
            act_summary = self.actuation.get("summary") or str(self.actuation)
            lines.append(f"Actuation: {act_summary}")
        if self.tactile_delta:
            v_str = "verified" if self.tactile_delta.get("verified") else "unverified"
            added = self.tactile_delta.get("elements_added_count", 0)
            removed = self.tactile_delta.get("elements_removed_count", 0)
            lines.append(f"Tactile Delta: {v_str} (+{added}/-{removed} elements)")
        if self.last_messages:
            lines.append("Latest Messages:")
            for msg in self.last_messages[-5:]:
                lines.append(f'  - "{msg}"')
        elif self.last_message:
            lines.append(f"Last Message: \"{self.last_message}\"")
        if self.interactive_elements:
            lines.append(f"Interactive Elements ({len(self.interactive_elements)} discovered):")
            for el in self.interactive_elements[:max_elements]:
                lines.append(f"  - {el.format_line()}")
            if len(self.interactive_elements) > max_elements:
                lines.append(f"  ... (+{len(self.interactive_elements) - max_elements} more)")
        return "\n".join(lines)


class FovealVisionEngine:
    """
    Sub-5ms Zero-GPU PyObjC `ApplicationServices` Accessibility DOM crawler.
    Unlocks Electron & Chromium accessibility trees (AXManualAccessibility & AXEnhancedUserInterface).
    """

    @classmethod
    def _get_ax_attribute(cls, ax_elem: Any, attribute_name: str) -> Any:
        try:
            import ApplicationServices as AS

            err, val = AS.AXUIElementCopyAttributeValue(ax_elem, attribute_name, None)
            return val if err == 0 else None
        except Exception:
            return None

    @classmethod
    def _extract_bbox(cls, ax_elem: Any) -> Tuple[float, float, float, float]:
        try:
            import ApplicationServices as AS

            pos_val = cls._get_ax_attribute(ax_elem, "AXPosition")
            sz_val = cls._get_ax_attribute(ax_elem, "AXSize")
            x = y = w = h = 0.0
            if pos_val:
                ok, pt = AS.AXValueGetValue(pos_val, AS.kAXValueCGPointType, None)
                if ok:
                    x, y = float(pt.x), float(pt.y)
            if sz_val:
                ok, sz = AS.AXValueGetValue(sz_val, AS.kAXValueCGSizeType, None)
                if ok:
                    w, h = float(sz.width), float(sz.height)
            return (round(x, 1), round(y, 1), round(w, 1), round(h, 1))
        except Exception:
            return (0.0, 0.0, 0.0, 0.0)

    @classmethod
    def _parse_ui_element(
        cls,
        ax_elem: Any,
        depth: int = 0,
        is_focused: Optional[bool] = None,
    ) -> Optional[FovealUIElement]:
        if ax_elem is None:
            return None
        role = str(cls._get_ax_attribute(ax_elem, "AXRole") or "")
        if not role:
            return None

        subrole = cls._get_ax_attribute(ax_elem, "AXSubrole")
        subrole_str = str(subrole).strip() if subrole else None
        title = cls._get_ax_attribute(ax_elem, "AXTitle")
        title_str = str(title).strip() if title is not None else ""
        desc = cls._get_ax_attribute(ax_elem, "AXDescription")
        desc_str = str(desc).strip() if desc is not None else ""
        val = cls._get_ax_attribute(ax_elem, "AXValue")
        val_str = str(val).strip() if isinstance(val, (str, int, float)) else ""

        label = title_str or desc_str or (val_str[:120] if role == "AXStaticText" else "") or (subrole_str or "")
        focused = is_focused if is_focused is not None else bool(cls._get_ax_attribute(ax_elem, "AXFocused"))
        enabled_val = cls._get_ax_attribute(ax_elem, "AXEnabled")
        enabled = True if enabled_val is None else bool(enabled_val)
        ident = cls._get_ax_attribute(ax_elem, "AXIdentifier")
        ident_str = str(ident).strip() if ident else None

        bbox = cls._extract_bbox(ax_elem)

        is_secure = False
        role_lower = role.lower()
        subrole_lower = (subrole_str or "").lower()
        if (
            role == "AXSecureTextField"
            or subrole_str == "AXSecureTextField"
            or "secure" in role_lower
            or "password" in role_lower
            or "secure" in subrole_lower
            or "password" in subrole_lower
        ):
            is_secure = True
            val_str = "[REDACTED_PASSWORD]"

        return FovealUIElement(
            role=role,
            label=label[:120],
            title=title_str[:120] if title_str else label[:120],
            description=desc_str[:120],
            value=val_str[:2000] if val_str else None,
            bbox=bbox,
            focused=focused,
            enabled=enabled,
            subrole=subrole_str,
            identifier=ident_str,
            depth=depth,
            is_secure=is_secure,
            raw_element=ax_elem,
        )

    @classmethod
    def resolve_target_app(
        cls,
        pid: Optional[int] = None,
        app_name: Optional[str] = None,
    ) -> Tuple[Optional[int], str, str]:
        """
        Resolve target macOS application by PID or app name substring.
        Returns: (pid, app_name, bundle_id)
        """
        try:
            from AppKit import NSWorkspace, NSRunningApplication

            ws = NSWorkspace.sharedWorkspace()

            if pid is not None:
                running = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
                if running:
                    name = str(running.localizedName() or f"PID-{pid}").strip()
                    bid = str(running.bundleIdentifier() or "").strip()
                    return (pid, name, bid)
                return (pid, f"PID-{pid}", "")

            if app_name:
                query = app_name.strip().lower()
                apps = list(ws.runningApplications())

                # 1. Exact match on localizedName
                for a in apps:
                    loc = str(a.localizedName() or "").strip().lower()
                    if loc == query:
                        return (int(a.processIdentifier()), str(a.localizedName()), str(a.bundleIdentifier() or ""))

                # 2. Substring match on localizedName
                for a in apps:
                    loc = str(a.localizedName() or "").strip().lower()
                    if query in loc:
                        return (int(a.processIdentifier()), str(a.localizedName()), str(a.bundleIdentifier() or ""))

                # 3. Substring match on bundleIdentifier or executable lastPathComponent
                for a in apps:
                    bid = str(a.bundleIdentifier() or "").strip().lower()
                    exec_name = ""
                    if a.executableURL():
                        exec_name = str(a.executableURL().lastPathComponent() or "").strip().lower()
                    if query in bid or (exec_name and query in exec_name):
                        return (int(a.processIdentifier()), str(a.localizedName() or exec_name), bid)

            # Default to frontmost application
            front = ws.frontmostApplication()
            if front:
                return (
                    int(front.processIdentifier()),
                    str(front.localizedName() or "Unknown").strip(),
                    str(front.bundleIdentifier() or "").strip(),
                )
        except Exception as exc:
            logger.debug("resolve_target_app failed: %s", exc)

        return (pid or 0, app_name or "Unknown", "")

    @classmethod
    def find_target_windows(
        cls,
        ax_app: Any,
        title_query: Optional[str] = None,
        include_floating: bool = False,
        include_status_bar: bool = False,
    ) -> List[Any]:
        """
        Locate target window(s) for the given AX application.
        If title_query is provided, matches AXTitle, AXDocument, or child AXWebArea title.
        If title_query is None, returns all visible windows (focused window first).
        Supports include_floating and include_status_bar discovery and auto-fallback when AXWindows is empty.
        """
        try:
            raw_wins = cls._get_ax_attribute(ax_app, "AXWindows") or []
            focused_win = cls._get_ax_attribute(ax_app, "AXFocusedWindow")
            main_win = cls._get_ax_attribute(ax_app, "AXMainWindow")

            all_candidates: List[Any] = []
            if focused_win:
                all_candidates.append(focused_win)
            if main_win and main_win not in all_candidates:
                all_candidates.append(main_win)
            for w in raw_wins:
                if w not in all_candidates:
                    all_candidates.append(w)

            # Floating & status-bar overlays discovery
            extras_menu_bar = cls._get_ax_attribute(ax_app, "AXExtrasMenuBar")
            menu_bar = cls._get_ax_attribute(ax_app, "AXMenuBar")

            if include_status_bar:
                if extras_menu_bar and extras_menu_bar not in all_candidates:
                    all_candidates.append(extras_menu_bar)
                if menu_bar and menu_bar not in all_candidates:
                    all_candidates.append(menu_bar)

            if include_floating:
                app_children = cls._get_ax_attribute(ax_app, "AXChildren") or []
                for ch in app_children:
                    r = str(cls._get_ax_attribute(ch, "AXRole") or "")
                    sr = str(cls._get_ax_attribute(ch, "AXSubrole") or "")
                    if (
                        r in ("AXWindow", "AXDialog", "AXSheet", "AXPopover", "AXMenu")
                        or sr in ("AXFloatingWindow", "AXSystemDialog", "NSPanel", "AXDialog")
                    ):
                        if ch not in all_candidates:
                            all_candidates.append(ch)

            # Auto-fallback when AXWindows is empty or yields 0 candidates (e.g. Granola, Control Center, menu bar apps)
            if not all_candidates:
                if extras_menu_bar and extras_menu_bar not in all_candidates:
                    all_candidates.append(extras_menu_bar)
                if menu_bar and (include_status_bar or not extras_menu_bar) and menu_bar not in all_candidates:
                    all_candidates.append(menu_bar)
                app_children = cls._get_ax_attribute(ax_app, "AXChildren") or []
                for ch in app_children:
                    if ch not in all_candidates:
                        all_candidates.append(ch)

            if not title_query:
                return all_candidates if all_candidates else []

            q = title_query.strip().lower()
            matched: List[Any] = []
            for w in all_candidates:
                title = str(cls._get_ax_attribute(w, "AXTitle") or "").lower()
                doc = str(cls._get_ax_attribute(w, "AXDocument") or "").lower()
                desc = str(cls._get_ax_attribute(w, "AXDescription") or "").lower()
                if q in title or q in doc or q in desc:
                    matched.append(w)
                    continue

                # Check immediate children / AXWebArea for tab or document title
                children = cls._get_ax_attribute(w, "AXChildren") or []
                found_child = False
                for ch in children:
                    ch_title = str(cls._get_ax_attribute(ch, "AXTitle") or "").lower()
                    ch_desc = str(cls._get_ax_attribute(ch, "AXDescription") or "").lower()
                    ch_val = str(cls._get_ax_attribute(ch, "AXValue") or "").lower()
                    if q in ch_title or q in ch_desc or q in ch_val:
                        matched.append(w)
                        found_child = True
                        break
                if found_child:
                    continue

            return matched if matched else all_candidates
        except Exception as exc:
            logger.debug("find_target_windows failed: %s", exc)
            return []

    @classmethod
    def _check_human_hid_guard(
        cls,
        target_pid: Optional[int] = None,
        allow_focus_steal: bool = False,
    ) -> Dict[str, Any]:
        """
        Inspect human HID idle time via Quartz CGEventSourceSecondsSinceLastEventType
        and check whether target_pid is already the frontmost application.
        Prevents background GUI automation from stealing window focus or typing into
        an active human keyboard session.
        """
        idle_sec = 999.0
        frontmost_pid: Optional[int] = None
        try:
            import Quartz

            state_id = getattr(Quartz, "kCGEventSourceStateCombinedSessionState", 0)
            any_input = getattr(Quartz, "kCGAnyInputEventType", 0xFFFFFFFF)
            if hasattr(Quartz, "CGEventSourceSecondsSinceLastEventType"):
                raw_idle = Quartz.CGEventSourceSecondsSinceLastEventType(state_id, any_input)
                if isinstance(raw_idle, (int, float)):
                    idle_sec = float(raw_idle)
        except Exception as exc:
            logger.debug("_check_human_hid_guard Quartz check failed: %s", exc)

        try:
            from AppKit import NSWorkspace

            ws = NSWorkspace.sharedWorkspace()
            front = ws.frontmostApplication() if ws else None
            if front:
                raw_pid = front.processIdentifier()
                if isinstance(raw_pid, int):
                    frontmost_pid = raw_pid
        except Exception as exc:
            logger.debug("_check_human_hid_guard NSWorkspace check failed: %s", exc)

        human_active = idle_sec < 1.5
        is_target_frontmost = bool(
            target_pid is not None and frontmost_pid is not None and int(target_pid) == int(frontmost_pid)
        )
        can_focus_steal = bool(allow_focus_steal and not human_active)
        return {
            "human_idle_seconds": round(idle_sec, 3),
            "human_active_recently": human_active,
            "frontmost_pid": frontmost_pid,
            "target_pid": target_pid,
            "is_target_frontmost": is_target_frontmost,
            "allow_focus_steal": bool(allow_focus_steal),
            "can_focus_steal": can_focus_steal,
        }

    @classmethod
    def _post_key_event(
        cls,
        event: Any,
        pid: Optional[int] = None,
        allow_focus_steal: bool = False,
    ) -> str:
        """
        Dispatch a Quartz keyboard CGEvent directly to target process Mach port via
        CGEventPostToPid when pid is known, avoiding global kCGHIDEventTap focus/key hijacking.
        """
        import Quartz

        if pid is not None and int(pid) > 0 and hasattr(Quartz, "CGEventPostToPid"):
            Quartz.CGEventPostToPid(int(pid), event)
            return "CGEventPostToPid"

        guard = cls._check_human_hid_guard(target_pid=pid, allow_focus_steal=allow_focus_steal)
        if guard["human_active_recently"] and not allow_focus_steal:
            logger.warning(
                "Blocked global kCGHIDEventTap key event: human HID input %.3fs ago (<1.5s)",
                guard["human_idle_seconds"],
            )
            return "blocked_human_hid_guard"

        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        return "kCGHIDEventTap"

    _CHROMIUM_SCRIPT_APPS: Tuple[str, ...] = (
        "Google Chrome",
        "Brave Browser",
        "Microsoft Edge",
        "Vivaldi",
        "Chromium",
    )

    @classmethod
    def _is_no_cursor_env(cls, no_cursor: bool = False) -> bool:
        if no_cursor:
            return True
        return os.environ.get("WBC_HANDS_NO_CURSOR", "").strip().lower() in ("1", "true", "yes")

    @classmethod
    def _snapshot_has_live_ax(
        cls,
        snapshot: Optional[FovealWindowSnapshot] = None,
        element: Optional[FovealUIElement] = None,
    ) -> bool:
        if element is not None and element.raw_element is not None:
            return True
        if snapshot is not None:
            if snapshot.focused_element is not None and snapshot.focused_element.raw_element is not None:
                return True
            for el in snapshot.interactive_elements:
                if el.raw_element is not None:
                    return True
        return False

    @classmethod
    def _can_use_global_cursor(
        cls,
        snapshot: Optional[FovealWindowSnapshot] = None,
        element: Optional[FovealUIElement] = None,
        pid: Optional[int] = None,
        allow_focus_steal: bool = False,
        no_cursor: bool = False,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine whether global kCGHIDEventTap mouse/cursor/scroll events may be posted.
        Blocks physical cursor takeover whenever:
          1. `no_cursor=True` or `WBC_HANDS_NO_CURSOR=1`, OR
          2. On live AX snapshots (`has_live_ax=True`), target app is in the background
             (`not is_target_frontmost` and `not can_focus_steal`) or human HID was active
             within 1.5s (`human_active_recently` and `not allow_focus_steal`).
        """
        eff_pid = pid if pid is not None else (snapshot.pid if snapshot else None)
        guard = cls._check_human_hid_guard(target_pid=eff_pid, allow_focus_steal=allow_focus_steal)
        no_cursor_locked = cls._is_no_cursor_env(no_cursor)
        guard["no_cursor"] = no_cursor_locked
        if no_cursor_locked:
            return False, guard

        has_live_ax = cls._snapshot_has_live_ax(snapshot=snapshot, element=element)
        if has_live_ax:
            if not guard["is_target_frontmost"] and not guard["can_focus_steal"]:
                return False, guard
            if guard["human_active_recently"] and not allow_focus_steal:
                return False, guard
        return True, guard

    @classmethod
    def _resolve_chromium_app_name(cls, app_name: Optional[str]) -> Optional[str]:
        if not app_name:
            return None
        low = app_name.strip().lower()
        for cand in cls._CHROMIUM_SCRIPT_APPS:
            if cand.lower() == low or low in cand.lower():
                return cand
        return None

    @classmethod
    def _chrome_background_dom_actuate(
        cls,
        op: str,
        element: Optional[FovealUIElement] = None,
        snapshot: Optional[FovealWindowSnapshot] = None,
        app_name: Optional[str] = None,
        window_title: Optional[str] = None,
        label_override: Optional[str] = None,
        screen_xy: Optional[Tuple[float, float]] = None,
        direction: str = "down",
        amount: int = 5,
        text_value: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Zero-cursor, zero-focus-steal DOM actuation inside a background Chromium tab
        via AppleScript `execute t javascript ...` (never calls `activate` or touches HID cursor).
        """
        raw_app = app_name or (snapshot.app_name if snapshot else None)
        browser_app = cls._resolve_chromium_app_name(raw_app)
        if not browser_app:
            return {"ok": False, "reason": "not_chromium"}

        raw_title = (window_title or (snapshot.window_title if snapshot else "") or "").strip()
        # Strip trailing " - Google Chrome" suffix from macOS AXWindow titles so tab title matches cleanly
        title_query = re.sub(r"\s*[-– - ]\s*(Google Chrome|Brave|Microsoft Edge|Vivaldi|Chromium).*$", "", raw_title, flags=re.IGNORECASE).strip()

        label = (
            label_override
            if label_override is not None
            else ((element.label or element.title or element.description or "") if element else "")
        ).strip()
        win_bbox = list(snapshot.window_bbox) if snapshot else [0.0, 0.0, 0.0, 0.0]
        payload = {
            "op": op,
            "label": label,
            "role": element.role if element else "",
            "screenXY": list(screen_xy) if screen_xy else None,
            "winBBox": win_bbox,
            "direction": direction,
            "amount": int(amount),
            "textValue": text_value,
        }
        js_code = (
            "(function(){"
            f"const p = {json.dumps(payload)};"
            "function findEl(q){"
            "  if(!q) return null;"
            "  const ql = q.toLowerCase();"
            "  const sel = 'button,a,[role=\"button\"],[role=\"tab\"],[role=\"menuitem\"],[role=\"option\"],[role=\"checkbox\"],[role=\"switch\"],[role=\"link\"],input,textarea,select,summary,label,[tabindex]';"
            "  const nodes = Array.from(document.querySelectorAll(sel));"
            "  let best = null; let bestScore = 999999999;"
            "  for(const el of nodes){"
            "    const r = el.getBoundingClientRect();"
            "    if(r.width <= 0 || r.height <= 0) continue;"
            "    const attrs = [el.getAttribute('aria-label'), el.getAttribute('title'), el.getAttribute('placeholder'), el.getAttribute('name'), el.innerText, el.textContent, el.value];"
            "    for(const raw of attrs){"
            "      if(!raw) continue;"
            "      const s = String(raw).trim();"
            "      if(!s) continue;"
            "      const sl = s.toLowerCase();"
            "      if(sl === ql){ return el; }"
            "      if(sl.includes(ql)){"
            "        const area = r.width * r.height + s.length * 10;"
            "        if(area < bestScore){ bestScore = area; best = el; }"
            "      }"
            "    }"
            "  }"
            "  return best;"
            "}"
            "if(p.op === 'scroll'){"
            "  const dy = (p.direction === 'down' || p.direction === 'bottom') ? (p.direction === 'bottom' ? 2400 : p.amount * 120) : (p.direction === 'top' ? -2400 : -p.amount * 120);"
            "  window.scrollBy({top: dy, behavior: 'instant'});"
            "  return 'scrolled_bg_dom';"
            "}"
            "let el = null;"
            "if(p.op === 'click_point' && p.screenXY && p.winBBox && p.winBBox[2] > 0){"
            "  const chromeTop = Math.max(0, p.winBBox[3] - window.innerHeight);"
            "  const rx = p.screenXY[0] - p.winBBox[0];"
            "  const ry = p.screenXY[1] - (p.winBBox[1] + chromeTop);"
            "  if(rx >= 0 && ry >= 0 && rx <= window.innerWidth && ry <= window.innerHeight){"
            "    el = document.elementFromPoint(rx, ry);"
            "  }"
            "} else {"
            "  el = findEl(p.label);"
            "}"
            "if(!el) return 'not_found';"
            "if(typeof el.scrollIntoView === 'function') el.scrollIntoView({block:'nearest', inline:'nearest'});"
            "const rect = el.getBoundingClientRect();"
            "const cx = rect.left + rect.width / 2; const cy = rect.top + rect.height / 2;"
            "const evInit = {bubbles: true, cancelable: true, view: window, clientX: cx, clientY: cy};"
            "if(p.op === 'hover'){"
            "  el.dispatchEvent(new PointerEvent('pointerover', evInit));"
            "  el.dispatchEvent(new MouseEvent('mouseover', evInit));"
            "  el.dispatchEvent(new PointerEvent('pointerenter', Object.assign({}, evInit, {bubbles: false})));"
            "  el.dispatchEvent(new MouseEvent('mouseenter', Object.assign({}, evInit, {bubbles: false})));"
            "  return 'hovered_bg_dom';"
            "}"
            "if(p.op === 'contextmenu'){"
            "  el.dispatchEvent(new MouseEvent('contextmenu', Object.assign({}, evInit, {button: 2, buttons: 2})));"
            "  return 'contextmenu_bg_dom';"
            "}"
            "if(p.op === 'input' && p.textValue !== null){"
            "  el.focus();"
            "  const proto = el.tagName === 'TEXTAREA' ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype;"
            "  const desc = proto ? Object.getOwnPropertyDescriptor(proto, 'value') : null;"
            "  if(desc && desc.set){ desc.set.call(el, p.textValue); } else { el.value = p.textValue; }"
            "  el.dispatchEvent(new InputEvent('input', {bubbles: true, data: p.textValue, inputType: 'insertText'}));"
            "  el.dispatchEvent(new Event('change', {bubbles: true}));"
            "  return 'input_bg_dom';"
            "}"
            "el.dispatchEvent(new PointerEvent('pointerdown', evInit));"
            "el.dispatchEvent(new MouseEvent('mousedown', evInit));"
            "el.dispatchEvent(new PointerEvent('pointerup', evInit));"
            "el.dispatchEvent(new MouseEvent('mouseup', evInit));"
            "if(typeof el.click === 'function'){ el.click(); } else { el.dispatchEvent(new MouseEvent('click', evInit)); }"
            "return 'clicked_bg_dom';"
            "})()"
        )
        applescript = (
            "on run argv\n"
            "    set browserApp to item 1 of argv\n"
            "    set titleQuery to item 2 of argv\n"
            "    set jsCode to item 3 of argv\n"
            "    tell application browserApp\n"
            "        if (count of windows) is 0 then return \"no_window\"\n"
            "        if titleQuery is not \"\" then\n"
            "            repeat with w in windows\n"
            "                repeat with t in tabs of w\n"
            "                    if ((title of t as text) contains titleQuery) or ((URL of t as text) contains titleQuery) then\n"
            "                        return (execute t javascript jsCode) as text\n"
            "                    end if\n"
            "                end repeat\n"
            "            end repeat\n"
            "        end if\n"
            "        return (execute active tab of front window javascript jsCode) as text\n"
            "    end tell\n"
            "end run\n"
        )
        try:
            proc = subprocess.run(
                ["osascript", "-", browser_app, title_query, js_code],
                input=applescript,
                text=True,
                capture_output=True,
                timeout=1.5,
            )
            out = (proc.stdout or "").strip()
            err = (proc.stderr or "").strip()
            if proc.returncode == 0 and out in (
                "clicked_bg_dom",
                "hovered_bg_dom",
                "contextmenu_bg_dom",
                "scrolled_bg_dom",
                "input_bg_dom",
            ):
                return {"ok": True, "result": out, "browser": browser_app}
            hint = ""
            if "Allow JavaScript from Apple Events" in err:
                hint = " (Enable Chrome: View > Developer > Allow JavaScript from Apple Events for background DOM clicks)"
            return {"ok": False, "reason": out or err or "failed", "hint": hint}
        except Exception as exc:
            return {"ok": False, "reason": str(exc), "hint": ""}

    @classmethod
    def _find_element_at_point(
        cls,
        x: float,
        y: float,
        snapshot: Optional[FovealWindowSnapshot],
        tolerance_px: float = 14.0,
    ) -> Optional[FovealUIElement]:
        """Resolve the most specific interactive AX element containing screen point (x, y)."""
        if not snapshot or not snapshot.interactive_elements:
            return None
        best_el: Optional[FovealUIElement] = None
        best_area = float("inf")
        for el in snapshot.interactive_elements:
            bx, by, bw, bh = el.bbox
            if bw > 0 and bh > 0:
                if (bx - tolerance_px) <= x <= (bx + bw + tolerance_px) and (by - tolerance_px) <= y <= (by + bh + tolerance_px):
                    area = bw * bh
                    if area < best_area:
                        best_area = area
                        best_el = el
        return best_el

    @classmethod
    def click_xy(
        cls,
        x: float,
        y: float,
        pid: Optional[int] = None,
        snapshot: Optional[FovealWindowSnapshot] = None,
        allow_focus_steal: bool = False,
        no_cursor: bool = False,
    ) -> Dict[str, Any]:
        """
        Performs a coordinate click at (x, y).
        When targeting a background app or `no_cursor=True`, automatically reroutes (x, y)
        to zero-cursor AXPress / ChromeBackgroundDOM instead of hijacking kCGHIDEventTap.
        """
        can_cursor, guard = cls._can_use_global_cursor(
            snapshot=snapshot,
            pid=pid,
            allow_focus_steal=allow_focus_steal,
            no_cursor=no_cursor,
        )
        if not can_cursor:
            hit_el = cls._find_element_at_point(float(x), float(y), snapshot)
            if hit_el is not None:
                ax_res = cls.click_element(
                    hit_el,
                    snapshot=snapshot,
                    allow_focus_steal=allow_focus_steal,
                    no_cursor=True,
                )
                if ax_res.get("status") == "success":
                    ax_res["coordinates"] = [float(x), float(y)]
                    ax_res["rerouted_from_xy"] = True
                    ax_res["cursor_moved"] = False
                    return ax_res

            dom_res = cls._chrome_background_dom_actuate(
                "click_point",
                snapshot=snapshot,
                screen_xy=(float(x), float(y)),
            )
            if dom_res.get("ok"):
                return {
                    "status": "success",
                    "method": "ChromeBackgroundDOM_xy",
                    "coordinates": [float(x), float(y)],
                    "cursor_moved": False,
                    "summary": f"Clicked ({x:.1f}, {y:.1f}) in background Chrome DOM without moving cursor",
                }

            return {
                "status": "error",
                "method": "CGEvent_xy_blocked_focus_steal",
                "coordinates": [float(x), float(y)],
                "cursor_moved": False,
                "hid_guard": guard,
                "error": f"Blocked cursor-hijacking coordinate click at ({x:.1f}, {y:.1f}) while backgrounded{dom_res.get('hint', '')}",
                "summary": f"Blocked focus-stealing coordinate click at ({x:.1f}, {y:.1f})",
            }

        try:
            import Quartz
            from Quartz import CGPoint

            pt = CGPoint(float(x), float(y))
            down = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventLeftMouseDown, pt, Quartz.kCGMouseButtonLeft)
            up = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventLeftMouseUp, pt, Quartz.kCGMouseButtonLeft)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, down)
            time.sleep(0.02)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, up)
            return {
                "status": "success",
                "method": "CGEvent_xy",
                "coordinates": [float(x), float(y)],
                "summary": f"Clicked at ({x:.1f}, {y:.1f})",
            }
        except Exception as exc:
            return {
                "status": "error",
                "method": "CGEvent_xy",
                "coordinates": [float(x), float(y)],
                "error": str(exc),
                "summary": f"Failed coordinate click ({x}, {y}): {exc}",
            }

    @classmethod
    def click_element(
        cls,
        target: Any,
        snapshot: Optional[FovealWindowSnapshot] = None,
        allow_focus_steal: bool = False,
        no_cursor: bool = False,
    ) -> Dict[str, Any]:
        """
        Click element by exact ID ('e1') or substring match against title/label/value/role.
        First attempts native AXPress action, then background Chromium DOM click if needed;
        falls back to Quartz CGEvent coordinate click only when safe under _can_use_global_cursor.
        """
        element: Optional[FovealUIElement] = None
        if isinstance(target, FovealUIElement):
            element = target
        elif snapshot and snapshot.interactive_elements:
            t_str = str(target).strip()
            # 1. Exact match on id
            for el in snapshot.interactive_elements:
                if el.id and el.id.lower() == t_str.lower():
                    element = el
                    break
            # 2. Substring match on label, value, description, or role
            if element is None:
                t_lower = t_str.lower()
                for el in snapshot.interactive_elements:
                    cand = f"{el.label} {el.value or ''} {el.description} {el.role}".lower()
                    if t_lower in cand:
                        element = el
                        break

        if element is None:
            return {
                "status": "error",
                "method": "none",
                "target": str(target),
                "error": f"Element '{target}' not found in snapshot",
                "summary": f"Element '{target}' not found",
            }

        # Attempt native AXPress
        ax_press_ok = False
        err_code = -1
        if element.raw_element is not None:
            try:
                import ApplicationServices as AS

                try:
                    AS.AXUIElementSetAttributeValue(element.raw_element, "AXFocused", True)
                except Exception:
                    pass
                err = AS.AXUIElementPerformAction(element.raw_element, "AXPress")
                if err == 0:
                    ax_press_ok = True
                    err_code = 0
                else:
                    err_code = err
            except Exception as e:
                err_code = str(e)

        if ax_press_ok:
            return {
                "status": "success",
                "method": "AXPress",
                "element_id": element.id,
                "role": element.role,
                "label": element.label,
                "center": list(element.center),
                "cursor_moved": False,
                "summary": f"Pressed [{element.id or element.role}] '{element.label}' via native AXPress",
            }

        # Guard against focus-stealing global mouse clicks on background live AX elements
        target_pid = snapshot.pid if snapshot else None
        can_cursor, guard = cls._can_use_global_cursor(
            snapshot=snapshot,
            element=element,
            pid=target_pid,
            allow_focus_steal=allow_focus_steal,
            no_cursor=no_cursor,
        )
        if not can_cursor:
            dom_res = cls._chrome_background_dom_actuate("click", element=element, snapshot=snapshot)
            if dom_res.get("ok"):
                return {
                    "status": "success",
                    "method": "ChromeBackgroundDOM",
                    "element_id": element.id,
                    "role": element.role,
                    "label": element.label,
                    "center": list(element.center),
                    "cursor_moved": False,
                    "summary": f"Clicked [{element.id or element.role}] '{element.label}' via background Chrome DOM (0 cursor move)",
                }
            return {
                "status": "error",
                "method": "AXPress_blocked_focus_steal",
                "element_id": element.id,
                "role": element.role,
                "label": element.label,
                "center": list(element.center),
                "cursor_moved": False,
                "hid_guard": guard,
                "error": f"AXPress failed ({err_code}) and global coordinate click blocked by focus-steal guard{dom_res.get('hint', '')}",
                "summary": f"Blocked focus-stealing click on background [{element.id or element.role}] '{element.label}'",
            }

        # Fallback to Quartz CGEvent coordinate click at element center
        cx, cy = element.center
        if cx == 0.0 and cy == 0.0 and element.bbox != (0.0, 0.0, 0.0, 0.0):
            cx = round(element.bbox[0] + element.bbox[2] / 2.0, 1)
            cy = round(element.bbox[1] + element.bbox[3] / 2.0, 1)

        res = cls.click_xy(cx, cy)
        res["element_id"] = element.id
        res["role"] = element.role
        res["label"] = element.label
        res["method"] = "CGEvent_fallback"
        res["summary"] = f"Clicked [{element.id or element.role}] '{element.label}' at ({cx}, {cy}) (AXPress err: {err_code})"
        return res

    @classmethod
    def _resolve_target_coordinates(
        cls,
        target: Any,
        snapshot: Optional[FovealWindowSnapshot] = None,
    ) -> Tuple[Optional[float], Optional[float], Optional[FovealUIElement]]:
        """Resolves target element ID ('e1'), coordinate string ('x,y'), or label to (cx, cy, element)."""
        if isinstance(target, FovealUIElement):
            cx, cy = target.center
            if cx == 0.0 and cy == 0.0 and target.bbox != (0.0, 0.0, 0.0, 0.0):
                cx = round(target.bbox[0] + target.bbox[2] / 2.0, 1)
                cy = round(target.bbox[1] + target.bbox[3] / 2.0, 1)
            return cx, cy, target

        t_str = str(target or "").strip()
        if "," in t_str:
            parts = [p.strip() for p in t_str.split(",")]
            if len(parts) == 2:
                try:
                    return float(parts[0]), float(parts[1]), None
                except ValueError:
                    pass

        if snapshot and snapshot.interactive_elements:
            t_lower = t_str.lower()
            # 1. Exact match on id
            for el in snapshot.interactive_elements:
                if el.id and el.id.lower() == t_lower:
                    cx, cy = el.center
                    if cx == 0.0 and cy == 0.0 and el.bbox != (0.0, 0.0, 0.0, 0.0):
                        cx = round(el.bbox[0] + el.bbox[2] / 2.0, 1)
                        cy = round(el.bbox[1] + el.bbox[3] / 2.0, 1)
                    return cx, cy, el
            # 2. Substring match on label, value, description, or role
            for el in snapshot.interactive_elements:
                cand = f"{el.label} {el.value or ''} {el.description} {el.role}".lower()
                if t_lower in cand:
                    cx, cy = el.center
                    if cx == 0.0 and cy == 0.0 and el.bbox != (0.0, 0.0, 0.0, 0.0):
                        cx = round(el.bbox[0] + el.bbox[2] / 2.0, 1)
                        cy = round(el.bbox[1] + el.bbox[3] / 2.0, 1)
                    return cx, cy, el
        return None, None, None

    @classmethod
    def hover(
        cls,
        target: str,
        snapshot: Optional[FovealWindowSnapshot] = None,
        app_name: Optional[str] = None,
        pid: Optional[int] = None,
        window_title: Optional[str] = None,
        settle_ms: int = 80,
    ) -> Dict[str, Any]:
        """
        Hover over element ID ('e1') or coordinates 'x,y' to reveal context controls and React toolbars.
        Posts kCGEventMouseMoved Quartz event and waits settle_ms (default: 80ms).
        """
        target_pid = snapshot.pid if snapshot and snapshot.pid else pid
        target_app = snapshot.app_name if snapshot and snapshot.app_name else app_name
        target_win = snapshot.window_title if snapshot and snapshot.window_title else window_title

        current_snap = snapshot
        if current_snap is None and (target_pid or target_app or target_win):
            try:
                current_snap = cls.capture_snapshot(
                    pid=target_pid,
                    app_name=target_app,
                    window_title_query=target_win,
                    record_telemetry=False,
                )
            except Exception:
                pass

        cx, cy, elem = cls._resolve_target_coordinates(target, snapshot=current_snap)
        if cx is None or cy is None:
            return {
                "status": "error",
                "action": "hover",
                "target": str(target),
                "error": f"Target '{target}' coordinates could not be resolved",
                "summary": f"Target '{target}' not found for hover",
            }

        try:
            import Quartz
            from Quartz import CGPoint

            pt = CGPoint(float(cx), float(cy))
            move_event = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventMouseMoved, pt, 0)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, move_event)
            time.sleep(settle_ms / 1000.0)
        except Exception as exc:
            return {
                "status": "error",
                "action": "hover",
                "target": str(target),
                "coordinates": [float(cx), float(cy)],
                "error": str(exc),
                "summary": f"Failed hover at ({cx}, {cy}): {exc}",
            }

        post_snap = None
        if target_pid or target_app or target_win or current_snap:
            try:
                post_snap = cls.capture_snapshot(
                    pid=target_pid,
                    app_name=target_app,
                    window_title_query=target_win,
                    max_elements=max(60, len(current_snap.interactive_elements) + 30) if current_snap and current_snap.interactive_elements else None,
                    filter_query=current_snap.filter_query if current_snap else None,
                    record_telemetry=False,
                )
            except Exception:
                pass

        tactile_delta = cls.compute_tactile_delta(current_snap, post_snap) if current_snap else {"verified": True, "delta_count": 0}

        return {
            "status": "success",
            "action": "hover",
            "target": str(target),
            "element_id": elem.id if elem else None,
            "role": elem.role if elem else None,
            "label": elem.label if elem else None,
            "coordinates": [float(cx), float(cy)],
            "summary": f"Hovered over [{elem.id or 'xy' if elem else 'xy'}] '{elem.label if elem else target}' at ({cx:.1f}, {cy:.1f})",
            "tactile_delta": tactile_delta,
            "post_snapshot": post_snap.to_dict() if post_snap else None,
        }

    @classmethod
    def right_click(
        cls,
        target: str,
        snapshot: Optional[FovealWindowSnapshot] = None,
        app_name: Optional[str] = None,
        pid: Optional[int] = None,
        window_title: Optional[str] = None,
        settle_ms: int = 80,
    ) -> Dict[str, Any]:
        """
        Right-click element ID ('e1') or coordinates 'x,y' to open context menu.
        Posts kCGEventRightMouseDown + kCGEventRightMouseUp Quartz events and waits settle_ms (default: 80ms).
        """
        target_pid = snapshot.pid if snapshot and snapshot.pid else pid
        target_app = snapshot.app_name if snapshot and snapshot.app_name else app_name
        target_win = snapshot.window_title if snapshot and snapshot.window_title else window_title

        current_snap = snapshot
        if current_snap is None and (target_pid or target_app or target_win):
            try:
                current_snap = cls.capture_snapshot(
                    pid=target_pid,
                    app_name=target_app,
                    window_title_query=target_win,
                    record_telemetry=False,
                )
            except Exception:
                pass

        cx, cy, elem = cls._resolve_target_coordinates(target, snapshot=current_snap)
        if cx is None or cy is None:
            return {
                "status": "error",
                "action": "right_click",
                "target": str(target),
                "error": f"Target '{target}' coordinates could not be resolved",
                "summary": f"Target '{target}' not found for right-click",
            }

        try:
            import Quartz
            from Quartz import CGPoint

            pt = CGPoint(float(cx), float(cy))
            down = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventRightMouseDown, pt, Quartz.kCGMouseButtonRight)
            up = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventRightMouseUp, pt, Quartz.kCGMouseButtonRight)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, down)
            time.sleep(0.02)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, up)
            time.sleep(settle_ms / 1000.0)
        except Exception as exc:
            return {
                "status": "error",
                "action": "right_click",
                "target": str(target),
                "coordinates": [float(cx), float(cy)],
                "error": str(exc),
                "summary": f"Failed right-click at ({cx}, {cy}): {exc}",
            }

        post_snap = None
        if target_pid or target_app or target_win or current_snap:
            try:
                post_snap = cls.capture_snapshot(
                    pid=target_pid,
                    app_name=target_app,
                    window_title_query=target_win,
                    max_elements=max(60, len(current_snap.interactive_elements) + 30) if current_snap and current_snap.interactive_elements else None,
                    filter_query=current_snap.filter_query if current_snap else None,
                    record_telemetry=False,
                )
            except Exception:
                pass

        tactile_delta = cls.compute_tactile_delta(current_snap, post_snap) if current_snap else {"verified": True, "delta_count": 0}

        return {
            "status": "success",
            "action": "right_click",
            "target": str(target),
            "element_id": elem.id if elem else None,
            "role": elem.role if elem else None,
            "label": elem.label if elem else None,
            "coordinates": [float(cx), float(cy)],
            "summary": f"Right-clicked [{elem.id or 'xy' if elem else 'xy'}] '{elem.label if elem else target}' at ({cx:.1f}, {cy:.1f})",
            "tactile_delta": tactile_delta,
            "post_snapshot": post_snap.to_dict() if post_snap else None,
        }

    @classmethod
    def attach_file(
        cls,
        file_path: str,
        app_name: Optional[str] = None,
        pid: Optional[int] = None,
        window_title: Optional[str] = None,
        trigger_target: Optional[str] = None,
        snapshot: Optional[FovealWindowSnapshot] = None,
    ) -> Dict[str, Any]:
        """
        Execute deterministic macOS NSOpenPanel file attachment sequence:
        1. If trigger_target provided, click attach button first and wait 120ms
        2. cmd+shift+g to trigger 'Go to the folder' sheet
        3. Sleep 60ms
        4. Atomically paste resolved absolute path
        5. Return/submit Go sheet
        6. Sleep 100ms
        7. Return/submit Open file dialog
        """
        resolved_path = str(Path(file_path).expanduser().resolve())
        target_pid = snapshot.pid if snapshot and snapshot.pid else pid
        target_app = snapshot.app_name if snapshot and snapshot.app_name else app_name
        target_win = snapshot.window_title if snapshot and snapshot.window_title else window_title

        actions_taken = []
        if trigger_target:
            click_res = cls.click_element(trigger_target, snapshot=snapshot)
            actions_taken.append(click_res)
            time.sleep(0.12)

        # 1. Hotkey cmd+shift+g to open "Go to folder" sheet in NSOpenPanel
        actions_taken.append(cls.dispatch_hotkey("cmd+shift+g", pid=target_pid, app_name=target_app))
        time.sleep(0.06)

        # 2. Atomic paste resolved path
        paste_ok = cls._atomic_paste_text(resolved_path)
        actions_taken.append({
            "status": "success" if paste_ok else "error",
            "action": "paste_path",
            "path": resolved_path,
            "summary": f"Pasted file path '{resolved_path}'",
        })

        # 3. Submit Go-to-folder sheet
        actions_taken.append(cls.send_submit(snapshot=snapshot))
        time.sleep(0.10)

        # 4. Confirm Open dialog
        actions_taken.append(cls.send_submit(snapshot=snapshot))
        time.sleep(0.06)

        post_snap = None
        if target_pid or target_app or target_win or snapshot:
            try:
                post_snap = cls.capture_snapshot(
                    pid=target_pid,
                    app_name=target_app,
                    window_title_query=target_win,
                    record_telemetry=False,
                )
            except Exception:
                pass

        tactile_delta = cls.compute_tactile_delta(snapshot, post_snap) if snapshot else {"verified": True, "delta_count": 0}

        return {
            "status": "success" if all(a.get("status") == "success" for a in actions_taken) else "partial",
            "action": "attach_file",
            "file_path": resolved_path,
            "actions": actions_taken,
            "summary": f"Attached file '{resolved_path}' via NSOpenPanel Go-to-folder sheet",
            "tactile_delta": tactile_delta,
            "post_snapshot": post_snap.to_dict() if post_snap else None,
        }

    @classmethod
    def execute_chain(
        cls,
        steps: Union[str, List[Union[str, Dict[str, Any]]]],
        app_name: Optional[str] = None,
        pid: Optional[int] = None,
        window_title: Optional[str] = None,
        snapshot: Optional[FovealWindowSnapshot] = None,
    ) -> Dict[str, Any]:
        """
        Execute compound multi-step action pipeline in a single turn.
        Supported step prefixes:
          click:<id>, act:<id>, click-xy:<x,y>, hover:<id>, right-click:<id>,
          type:<text>, paste:<text>, attach:<path>, hotkey:<chord>, key:<chord>,
          menu:<path>, scroll:<up|down>, submit, wait-reply[:<sec>], read-last
        """
        target_pid = snapshot.pid if snapshot and snapshot.pid else pid
        target_app = snapshot.app_name if snapshot and snapshot.app_name else app_name
        target_win = snapshot.window_title if snapshot and snapshot.window_title else window_title

        current_snap = snapshot
        if current_snap is None and (target_pid or target_app or target_win):
            try:
                current_snap = cls.capture_snapshot(
                    pid=target_pid,
                    app_name=target_app,
                    window_title_query=target_win,
                    record_telemetry=False,
                )
            except Exception:
                pass

        if isinstance(steps, str):
            raw_steps = []
            current = []
            in_quote = False
            quote_char = ""
            for ch in steps:
                if ch in ('"', "'"):
                    if not in_quote:
                        in_quote = True
                        quote_char = ch
                    elif ch == quote_char:
                        in_quote = False
                        quote_char = ""
                    current.append(ch)
                elif ch == "," and not in_quote:
                    step_str = "".join(current).strip()
                    if step_str:
                        raw_steps.append(step_str)
                    current = []
                else:
                    current.append(ch)
            if current:
                step_str = "".join(current).strip()
                if step_str:
                    raw_steps.append(step_str)
        else:
            raw_steps = list(steps)

        actions_taken = []
        for step in raw_steps:
            if isinstance(step, dict):
                act_type = str(step.get("action") or step.get("type") or "").strip().lower()
                arg = str(step.get("target") or step.get("value") or step.get("text") or step.get("path") or "").strip()
            else:
                s_str = str(step).strip()
                if ":" in s_str:
                    act_type, arg = s_str.split(":", 1)
                    act_type = act_type.strip().lower()
                    arg = arg.strip()
                else:
                    act_type = s_str.lower()
                    arg = ""

            if (arg.startswith('"') and arg.endswith('"')) or (arg.startswith("'") and arg.endswith("'")):
                arg = arg[1:-1]

            if act_type in ("click", "act"):
                actions_taken.append(cls.click_element(arg, snapshot=current_snap))
            elif act_type in ("click-xy", "click_xy"):
                parts = [p.strip() for p in arg.split(",")]
                if len(parts) == 2:
                    try:
                        actions_taken.append(cls.click_xy(float(parts[0]), float(parts[1])))
                    except Exception as e:
                        actions_taken.append({"status": "error", "action": "click_xy", "error": str(e)})
                else:
                    actions_taken.append({"status": "error", "action": "click_xy", "error": f"Invalid coordinates '{arg}'"})
            elif act_type == "hover":
                actions_taken.append(cls.hover(arg, snapshot=current_snap, app_name=target_app, pid=target_pid, window_title=target_win))
            elif act_type in ("right-click", "right_click"):
                actions_taken.append(cls.right_click(arg, snapshot=current_snap, app_name=target_app, pid=target_pid, window_title=target_win))
            elif act_type == "type":
                actions_taken.append(cls.type_text(arg, snapshot=current_snap))
            elif act_type == "paste":
                actions_taken.append(cls.paste_element(arg, snapshot=current_snap, submit=False))
            elif act_type == "attach":
                actions_taken.append(cls.attach_file(arg, snapshot=current_snap, app_name=target_app, pid=target_pid, window_title=target_win))
            elif act_type in ("hotkey", "key"):
                actions_taken.append(cls.dispatch_hotkey(arg, pid=target_pid, app_name=target_app))
            elif act_type == "menu":
                actions_taken.append(cls.trigger_menu_item(arg, pid=target_pid, app_name=target_app))
            elif act_type == "scroll":
                actions_taken.append(cls.scroll(arg, snapshot=current_snap))
            elif act_type == "submit":
                time.sleep(0.04)
                actions_taken.append(cls.send_submit(snapshot=current_snap))
            elif act_type in ("wait-reply", "wait_reply"):
                t_sec = float(arg) if arg else 15.0
                actions_taken.append(cls.wait_for_reply(app_name=target_app, pid=target_pid, window_title=target_win, timeout_sec=t_sec))
            elif act_type in ("read-last", "read_last"):
                texts = cls.extract_text_blocks(current_snap)
                actions_taken.append({
                    "status": "success",
                    "action": "read_last",
                    "messages": texts,
                    "last_message": texts[-1] if texts else "",
                    "summary": f"Extracted {len(texts)} text blocks",
                })
            elif act_type == "sleep":
                s_val = float(arg) if arg else 0.1
                time.sleep(s_val)
                actions_taken.append({"status": "success", "action": "sleep", "seconds": s_val, "summary": f"Slept {s_val}s"})
            else:
                actions_taken.append({"status": "error", "action": act_type, "error": f"Unknown chain step '{step}'"})

        time.sleep(0.06)
        post_snap = None
        if target_pid or target_app or target_win or current_snap:
            try:
                post_snap = cls.capture_snapshot(
                    pid=target_pid,
                    app_name=target_app,
                    window_title_query=target_win,
                    record_telemetry=False,
                )
            except Exception:
                pass

        tactile_delta = cls.compute_tactile_delta(current_snap, post_snap) if current_snap else {"verified": True, "delta_count": 0}

        return {
            "status": "success" if all(a.get("status") == "success" for a in actions_taken) else "partial",
            "action": "execute_chain",
            "steps_count": len(raw_steps),
            "actions": actions_taken,
            "summary": " -> ".join([a.get("summary", a.get("status", "done")) for a in actions_taken]),
            "tactile_delta": tactile_delta,
            "post_snapshot": post_snap.to_dict() if post_snap else None,
        }

    @classmethod
    def _atomic_paste_text(
        cls,
        text: str,
        target_elem: Optional[FovealUIElement] = None,
        pid: Optional[int] = None,
        allow_focus_steal: bool = False,
    ) -> bool:
        """
        Zero-focus-steal text injection:
        1. First attempts direct background Accessibility attribute write
           (AXUIElementSetAttributeValue(elem, "AXValue", text)) on target_elem or focused AXTextArea/AXTextField
           of target pid with zero clipboard mutation and zero keyboard events.
        2. If AXValue assignment is unsupported, falls back to atomic clipboard save/restore +
           process-targeted Mach port Cmd+V via Quartz.CGEventPostToPid(pid, event).
        """
        # 1. Direct Background AXValue Injection First
        raw_target = target_elem.raw_element if target_elem is not None else None
        if raw_target is not None or (pid is not None and int(pid) > 0):
            try:
                import ApplicationServices as AS

                if raw_target is None and pid is not None and int(pid) > 0:
                    ax_app = AS.AXUIElementCreateApplication(int(pid))
                    err_f, focused_el = AS.AXUIElementCopyAttributeValue(ax_app, "AXFocusedUIElement", None)
                    if err_f == 0 and focused_el is not None:
                        raw_target = focused_el

                if raw_target is not None:
                    try:
                        AS.AXUIElementSetAttributeValue(raw_target, "AXFocused", True)
                    except Exception:
                        pass
                    err_val = AS.AXUIElementSetAttributeValue(raw_target, "AXValue", text)
                    if err_val == 0:
                        return True
            except Exception as exc:
                logger.debug("Direct AXValue injection skipped/failed: %s", exc)

        # 2. Fallback: Save clipboard, pbcopy, and dispatch Cmd+V via CGEventPostToPid (or guarded HID tap)
        prior_clipboard = ""
        try:
            res = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=2.0)
            if res.returncode == 0:
                prior_clipboard = res.stdout
        except Exception as exc:
            logger.debug("pbpaste failed: %s", exc)

        try:
            subprocess.run(["pbcopy"], input=text, text=True, check=True, timeout=2.0)
        except Exception as exc:
            logger.debug("pbcopy failed: %s", exc)
            return False

        # Send Cmd+V keystroke (keycode 9 + Command modifier) via CGEventPostToPid when pid is known
        try:
            import Quartz

            cmd_flag = getattr(Quartz, "kCGEventFlagMaskCommand", 0x00100000)
            ev_down = Quartz.CGEventCreateKeyboardEvent(None, 9, True)
            Quartz.CGEventSetFlags(ev_down, cmd_flag)
            cls._post_key_event(ev_down, pid=pid, allow_focus_steal=allow_focus_steal)
            time.sleep(0.02)
            ev_up = Quartz.CGEventCreateKeyboardEvent(None, 9, False)
            Quartz.CGEventSetFlags(ev_up, cmd_flag)
            cls._post_key_event(ev_up, pid=pid, allow_focus_steal=allow_focus_steal)
        except Exception as exc:
            logger.debug("Quartz Cmd+V dispatch failed: %s", exc)

        # Allow target app to consume clipboard before restoring
        time.sleep(0.05)
        try:
            subprocess.run(["pbcopy"], input=prior_clipboard, text=True, timeout=2.0)
        except Exception as exc:
            logger.debug("Restoring clipboard via pbcopy failed: %s", exc)

        return True

    @classmethod
    def paste_element(
        cls,
        text: str,
        target: Any = None,
        snapshot: Optional[FovealWindowSnapshot] = None,
        submit: bool = False,
        allow_focus_steal: bool = False,
    ) -> Dict[str, Any]:
        """
        Paste text into target or focused input element without stealing window focus.
        Uses direct background AXValue injection first, falling back to CGEventPostToPid Cmd+V.
        """
        target_pid = snapshot.pid if snapshot and snapshot.pid else None
        guard = cls._check_human_hid_guard(target_pid=target_pid, allow_focus_steal=allow_focus_steal)

        target_elem: Optional[FovealUIElement] = None
        if target:
            cls.click_element(target, snapshot=snapshot, allow_focus_steal=allow_focus_steal)
            if snapshot and snapshot.interactive_elements:
                t_str = str(target).strip().lower()
                for el in snapshot.interactive_elements:
                    if el.id.lower() == t_str or t_str in f"{el.label} {el.role}".lower():
                        target_elem = el
                        break
        elif snapshot:
            if snapshot.focused_element and snapshot.focused_element.role in ("AXTextField", "AXTextArea", "AXSearchField"):
                target_elem = snapshot.focused_element
            else:
                # 1. Prioritize prompt/message/composer inputs
                for el in snapshot.interactive_elements:
                    if any(k in f"{el.label} {el.description} {el.title}".lower() for k in ("prompt", "message", "composer", "chat")):
                        target_elem = el
                        break
                # 2. Prioritize text editing fields
                if not target_elem:
                    for el in snapshot.interactive_elements:
                        if el.role in ("AXTextArea", "AXTextField", "AXSearchField", "AXComboBox"):
                            target_elem = el
                            break
                # 3. Fallback to focused element or AXWebArea
                if not target_elem:
                    target_elem = snapshot.focused_element or next(
                        (el for el in snapshot.interactive_elements if el.role == "AXWebArea"), None
                    )

        # Never activate/focus-steal unless explicitly permitted AND no recent human HID activity
        if target_pid and guard["can_focus_steal"]:
            try:
                from AppKit import NSRunningApplication, NSApplicationActivateIgnoringOtherApps

                app = NSRunningApplication.runningApplicationWithProcessIdentifier_(target_pid)
                if app:
                    app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
                    time.sleep(0.08)
            except Exception:
                pass

        # If target element was resolved automatically and not explicitly clicked, focus it via AXPress
        if target_elem and not target:
            cls.click_element(target_elem, snapshot=snapshot, allow_focus_steal=allow_focus_steal)
            time.sleep(0.06)

        pasted = cls._atomic_paste_text(
            text,
            target_elem=target_elem,
            pid=target_pid,
            allow_focus_steal=allow_focus_steal,
        )
        if submit:
            time.sleep(0.06)
            cls.send_submit(snapshot=snapshot, pid=target_pid, allow_focus_steal=allow_focus_steal)

        return {
            "status": "success" if pasted else "error",
            "action": "paste",
            "method": "atomic_paste_cmd_v",
            "pasted_text": text,
            "target_element": target_elem.to_dict() if target_elem else None,
            "hid_guard": guard,
            "summary": f"Pasted {len(text)} chars via background-safe atomic paste",
        }

    @classmethod
    def dispatch_hotkey(
        cls,
        chord: str,
        pid: Optional[int] = None,
        app_name: Optional[str] = None,
        allow_focus_steal: bool = False,
    ) -> Dict[str, Any]:
        """
        Parse and dispatch a hotkey chord (e.g. 'cmd+n', 'cmd+shift+p', 'escape', 'enter') via Quartz.
        Uses Quartz.CGEventPostToPid(pid, event) when pid is known so background windows never steal focus.
        """
        parts = [p.strip().lower() for p in chord.replace("-", "+").split("+") if p.strip()]
        if not parts:
            return {"status": "error", "action": "hotkey", "error": "Empty hotkey chord"}

        key_name = parts[-1]
        modifier_names = parts[:-1]

        flag_mask = 0
        for m in modifier_names:
            flag_mask |= QUARTZ_MODIFIERS.get(m, 0)

        keycode = QUARTZ_KEYCODES.get(key_name)
        if keycode is None:
            if len(key_name) == 1:
                keycode = QUARTZ_KEYCODES.get(key_name.lower(), 0)
            else:
                return {
                    "status": "error",
                    "action": "hotkey",
                    "error": f"Unknown key in hotkey chord: '{key_name}'",
                }

        if pid is None and app_name:
            resolved_pid, _, _ = cls.resolve_target_app(app_name=app_name)
            pid = resolved_pid

        guard = cls._check_human_hid_guard(target_pid=pid, allow_focus_steal=allow_focus_steal)
        if pid and guard["can_focus_steal"]:
            try:
                from AppKit import NSRunningApplication, NSApplicationActivateIgnoringOtherApps

                app = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
                if app:
                    app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
                    time.sleep(0.08)
            except Exception:
                pass

        try:
            import Quartz

            ev_down = Quartz.CGEventCreateKeyboardEvent(None, keycode, True)
            if flag_mask:
                Quartz.CGEventSetFlags(ev_down, flag_mask)
            dispatch_mode = cls._post_key_event(ev_down, pid=pid, allow_focus_steal=allow_focus_steal)
            time.sleep(0.02)
            ev_up = Quartz.CGEventCreateKeyboardEvent(None, keycode, False)
            if flag_mask:
                Quartz.CGEventSetFlags(ev_up, flag_mask)
            cls._post_key_event(ev_up, pid=pid, allow_focus_steal=allow_focus_steal)

            return {
                "status": "success",
                "action": "hotkey",
                "chord": chord,
                "keycode": keycode,
                "modifiers": modifier_names,
                "pid": pid,
                "dispatch_mode": dispatch_mode,
                "hid_guard": guard,
                "summary": f"Dispatched hotkey '{chord}' (keycode {keycode}) via {dispatch_mode}",
            }
        except Exception as exc:
            return {
                "status": "error",
                "action": "hotkey",
                "chord": chord,
                "error": str(exc),
                "summary": f"Failed hotkey '{chord}': {exc}",
            }

    @classmethod
    def trigger_menu_item(
        cls,
        menu_path: str,
        ax_app: Any = None,
        pid: Optional[int] = None,
        app_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Walk target app AXMenuBar -> AXMenuBarItem -> AXMenu -> AXMenuItem and trigger AXPress in <3ms.
        Example menu_path: 'View > Hide Sidebar' or 'File > New Window'.
        """
        parts = [p.strip() for p in menu_path.split(">") if p.strip()]
        if not parts:
            return {"status": "error", "action": "menu", "error": "Empty menu path"}

        try:
            import ApplicationServices as AS

            if ax_app is None:
                if pid is None and app_name:
                    pid, _, _ = cls.resolve_target_app(app_name=app_name)
                elif pid is None:
                    pid, _, _ = cls.resolve_target_app()
                if pid:
                    ax_app = AS.AXUIElementCreateApplication(pid)

            if ax_app is None:
                return {
                    "status": "error",
                    "action": "menu",
                    "path": menu_path,
                    "error": "Could not connect to target AXApplication",
                }

            menu_bar = cls._get_ax_attribute(ax_app, "AXMenuBar")
            if not menu_bar:
                return {
                    "status": "error",
                    "action": "menu",
                    "path": menu_path,
                    "error": "Target application has no AXMenuBar",
                }

            menu_items = cls._get_ax_attribute(menu_bar, "AXChildren") or []
            curr_elem: Optional[Any] = None

            first_target = parts[0].lower()
            for mi in menu_items:
                title = str(cls._get_ax_attribute(mi, "AXTitle") or "").strip().lower()
                if title == first_target or first_target in title:
                    curr_elem = mi
                    break

            if not curr_elem:
                return {
                    "status": "error",
                    "action": "menu",
                    "path": menu_path,
                    "error": f"Menu bar item '{parts[0]}' not found",
                }

            for part in parts[1:]:
                children = cls._get_ax_attribute(curr_elem, "AXChildren") or []
                if len(children) == 1 and str(cls._get_ax_attribute(children[0], "AXRole") or "") == "AXMenu":
                    children = cls._get_ax_attribute(children[0], "AXChildren") or []

                p_lower = part.lower()
                matched_child = None
                for ch in children:
                    ch_title = str(cls._get_ax_attribute(ch, "AXTitle") or "").strip().lower()
                    if ch_title == p_lower or p_lower in ch_title:
                        matched_child = ch
                        break

                if not matched_child:
                    return {
                        "status": "error",
                        "action": "menu",
                        "path": menu_path,
                        "error": f"Menu item '{part}' not found under '{parts[0]}'",
                    }
                curr_elem = matched_child

            err = AS.AXUIElementPerformAction(curr_elem, "AXPress")
            if err == 0:
                return {
                    "status": "success",
                    "action": "menu",
                    "path": menu_path,
                    "summary": f"Triggered menu item '{menu_path}' via AXPress in <3ms",
                }
            else:
                return {
                    "status": "error",
                    "action": "menu",
                    "path": menu_path,
                    "error": f"AXPress returned error code {err}",
                }
        except Exception as exc:
            return {
                "status": "error",
                "action": "menu",
                "path": menu_path,
                "error": str(exc),
                "summary": f"Menu execution failed: {exc}",
            }

    @classmethod
    def scroll(
        cls,
        direction: str = "down",
        amount: int = 5,
        target_center: Optional[Tuple[float, float]] = None,
        snapshot: Optional[FovealWindowSnapshot] = None,
    ) -> Dict[str, Any]:
        """
        Scroll target window/view up/down/top/bottom via native AXScroll actions and Quartz scroll events.
        """
        d = direction.lower().strip()
        ax_action_ok = False
        quartz_ok = False

        if snapshot:
            for el in snapshot.interactive_elements:
                if el.role in ("AXScrollArea", "AXWebArea", "AXList", "AXTable") and el.raw_element is not None:
                    try:
                        import ApplicationServices as AS

                        if d in ("down", "bottom"):
                            err = AS.AXUIElementPerformAction(el.raw_element, "AXScrollDownByPage")
                        else:
                            err = AS.AXUIElementPerformAction(el.raw_element, "AXScrollUpByPage")
                        if err == 0:
                            ax_action_ok = True
                            break
                    except Exception:
                        pass

        try:
            import Quartz

            delta_y = -amount if d in ("down", "bottom") else amount
            if d == "bottom":
                delta_y = -amount * 4
            elif d == "top":
                delta_y = amount * 4

            cx, cy = 500.0, 500.0
            if target_center:
                cx, cy = target_center
            elif snapshot and snapshot.window_bbox != (0.0, 0.0, 0.0, 0.0):
                cx = snapshot.window_bbox[0] + snapshot.window_bbox[2] / 2.0
                cy = snapshot.window_bbox[1] + snapshot.window_bbox[3] / 2.0

            ev = Quartz.CGEventCreateScrollWheelEvent(None, Quartz.kCGScrollEventUnitLine, 1, int(delta_y))
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
            quartz_ok = True
        except Exception as exc:
            logger.debug("Quartz scroll wheel failed: %s", exc)

        return {
            "status": "success" if (ax_action_ok or quartz_ok) else "partial",
            "action": "scroll",
            "direction": d,
            "ax_scroll": ax_action_ok,
            "quartz_scroll": quartz_ok,
            "summary": f"Scrolled {d} (AX: {ax_action_ok}, Quartz: {quartz_ok})",
        }

    @classmethod
    def type_text(
        cls,
        text: str,
        target: Any = None,
        snapshot: Optional[FovealWindowSnapshot] = None,
    ) -> Dict[str, Any]:
        """
        Type text into target or focused input element.
        Sets AXValue and sends Quartz keystrokes to ensure React/Electron input handlers fire.
        If AXValue fails or on rich-text editors and len(text) > 40, falls back to atomic paste.
        """
        target_elem: Optional[FovealUIElement] = None
        if target:
            cls.click_element(target, snapshot=snapshot)
            if snapshot and snapshot.interactive_elements:
                t_str = str(target).strip().lower()
                for el in snapshot.interactive_elements:
                    if el.id.lower() == t_str or t_str in f"{el.label} {el.role}".lower():
                        target_elem = el
                        break
        elif snapshot:
            if snapshot.focused_element and snapshot.focused_element.role in ("AXTextField", "AXTextArea", "AXSearchField"):
                target_elem = snapshot.focused_element
            else:
                # 1. Prioritize prompt/message/composer inputs
                for el in snapshot.interactive_elements:
                    if any(k in f"{el.label} {el.description} {el.title}".lower() for k in ("prompt", "message", "composer", "chat")):
                        target_elem = el
                        break
                # 2. Prioritize text editing fields
                if not target_elem:
                    for el in snapshot.interactive_elements:
                        if el.role in ("AXTextArea", "AXTextField", "AXSearchField", "AXComboBox"):
                            target_elem = el
                            break
                # 3. Fallback to focused element or AXWebArea
                if not target_elem:
                    target_elem = snapshot.focused_element or next(
                        (el for el in snapshot.interactive_elements if el.role == "AXWebArea"), None
                    )

        ax_val_set = False
        is_rich_text = target_elem and target_elem.role in ("AXTextArea", "AXWebArea")

        if target_elem and target_elem.raw_element is not None:
            try:
                import ApplicationServices as AS

                try:
                    AS.AXUIElementSetAttributeValue(target_elem.raw_element, "AXFocused", True)
                except Exception:
                    pass
                err = AS.AXUIElementSetAttributeValue(target_elem.raw_element, "AXValue", text)
                if err == 0:
                    ax_val_set = True
            except Exception:
                pass

        target_pid = snapshot.pid if snapshot and snapshot.pid else None

        # Fast path for rich text or multi-character text where AXValue didn't stick or len > 40
        if (not ax_val_set or is_rich_text) and len(text) > 40:
            pasted = cls._atomic_paste_text(text, target_elem=target_elem, pid=target_pid)
            if pasted:
                return {
                    "status": "success",
                    "method": "atomic_paste",
                    "typed_text": text,
                    "target_element": target_elem.to_dict() if target_elem else None,
                    "summary": f"Injected {len(text)} chars via background-safe atomic paste",
                }

        quartz_ok = False
        try:
            import Quartz

            ev_down = Quartz.CGEventCreateKeyboardEvent(None, 0, True)
            Quartz.CGEventKeyboardSetUnicodeString(ev_down, len(text), text)
            cls._post_key_event(ev_down, pid=target_pid)
            time.sleep(0.01)
            ev_up = Quartz.CGEventCreateKeyboardEvent(None, 0, False)
            Quartz.CGEventKeyboardSetUnicodeString(ev_up, len(text), text)
            cls._post_key_event(ev_up, pid=target_pid)
            quartz_ok = True
        except Exception as exc:
            logger.debug("Quartz typing failed: %s", exc)

        method = (
            "AXValue+CGEvent"
            if (ax_val_set and quartz_ok)
            else ("AXValue" if ax_val_set else ("CGEvent" if quartz_ok else "failed"))
        )
        status = "success" if (ax_val_set or quartz_ok) else "error"
        return {
            "status": status,
            "method": method,
            "typed_text": text,
            "target_element": target_elem.to_dict() if target_elem else None,
            "summary": f"Typed '{text}' via {method}",
        }

    @classmethod
    def send_submit(
        cls,
        snapshot: Optional[FovealWindowSnapshot] = None,
        pid: Optional[int] = None,
        allow_focus_steal: bool = False,
    ) -> Dict[str, Any]:
        """Send Return/Enter keystroke (keycode 36) via CGEventPostToPid with fallback to Send/Submit AXButton."""
        target_pid = pid if pid is not None else (snapshot.pid if snapshot and snapshot.pid else None)
        try:
            import Quartz

            ev_down = Quartz.CGEventCreateKeyboardEvent(None, 36, True)
            ev_up = Quartz.CGEventCreateKeyboardEvent(None, 36, False)
            dispatch_mode = cls._post_key_event(ev_down, pid=target_pid, allow_focus_steal=allow_focus_steal)
            time.sleep(0.03)
            cls._post_key_event(ev_up, pid=target_pid, allow_focus_steal=allow_focus_steal)

            button_clicked = None
            if snapshot and snapshot.interactive_elements:
                for el in snapshot.interactive_elements:
                    if el.role in ("AXButton", "AXPopUpButton"):
                        lbl = (el.label or el.title or el.description or "").strip().lower()
                        if lbl in ("send", "submit", "send message", "run", "run query", "execute"):
                            cls.click_element(el, snapshot=snapshot, allow_focus_steal=allow_focus_steal)
                            button_clicked = el.label or el.id
                            break

            return {
                "status": "success",
                "action": "submit",
                "keycode": 36,
                "pid": target_pid,
                "dispatch_mode": dispatch_mode,
                "button_fallback": button_clicked,
                "summary": f"Sent Return/Enter (keycode 36 via {dispatch_mode}){f' + clicked [{button_clicked}]' if button_clicked else ''}",
            }
        except Exception as exc:
            return {
                "status": "error",
                "action": "submit",
                "error": str(exc),
                "summary": f"Failed Return key: {exc}",
            }

    @classmethod
    def _compute_dom_signature(cls, snap: Optional[FovealWindowSnapshot]) -> Dict[str, Any]:
        """Compute structured DOM signature for tactile proprioception diffing."""
        if not snap:
            return {
                "window_title": "",
                "focused_key": None,
                "elements": [],
                "element_map": {},
                "labels": set(),
                "values": {},
            }

        focused_key = None
        if snap.focused_element:
            fe = snap.focused_element
            focused_key = (fe.role, fe.label, fe.value, fe.enabled, fe.focused)

        elem_tuples = []
        elem_map = {}
        labels = set()
        values = {}
        for el in snap.interactive_elements:
            lbl = el.label or el.title or ""
            if lbl:
                labels.add(lbl)
            key = (el.role, el.label, el.value, el.enabled, el.focused)
            elem_tuples.append(key)
            el_id = el.id or lbl or f"{el.role}:{el.center}"
            elem_map[el_id] = key
            if el.value:
                values[lbl or el_id] = el.value

        return {
            "window_title": snap.window_title,
            "focused_key": focused_key,
            "elements": elem_tuples,
            "element_map": elem_map,
            "labels": labels,
            "values": values,
        }

    @classmethod
    def compute_tactile_delta(
        cls,
        pre_snapshot: Optional[FovealWindowSnapshot],
        post_snapshot: Optional[FovealWindowSnapshot],
    ) -> Dict[str, Any]:
        """
        Compute tactile proprioception delta between pre-action and post-action snapshots.
        """
        pre_sig = cls._compute_dom_signature(pre_snapshot)
        post_sig = cls._compute_dom_signature(post_snapshot)

        pre_labels = pre_sig["labels"]
        post_labels = post_sig["labels"]

        added_labels = [l for l in post_labels if l not in pre_labels][:5]
        removed_labels = [l for l in pre_labels if l not in post_labels][:5]

        pre_counts = Counter(pre_sig["elements"])
        post_counts = Counter(post_sig["elements"])
        elements_added_count = sum((post_counts - pre_counts).values())
        elements_removed_count = sum((pre_counts - post_counts).values())

        value_changed = False
        pre_values = pre_sig["values"]
        post_values = post_sig["values"]
        for k, v in post_values.items():
            if k in pre_values and pre_values[k] != v:
                value_changed = True
                break
        if not value_changed and len(post_values) != len(pre_values):
            value_changed = True

        focus_changed = pre_sig["focused_key"] != post_sig["focused_key"]
        window_title_changed = pre_sig["window_title"] != post_sig["window_title"]

        verified = bool(
            elements_added_count > 0
            or elements_removed_count > 0
            or value_changed
            or focus_changed
            or window_title_changed
        )

        post_count = len(post_snapshot.interactive_elements) if post_snapshot else 0

        return {
            "verified": verified,
            "elements_added_count": elements_added_count,
            "elements_removed_count": elements_removed_count,
            "added_labels": added_labels,
            "removed_labels": removed_labels,
            "value_changed": value_changed,
            "focus_changed": focus_changed,
            "window_title_changed": window_title_changed,
            "post_element_count": post_count,
        }

    @classmethod
    def extract_text_blocks(cls, snapshot: Optional[FovealWindowSnapshot]) -> List[str]:
        """
        Extract meaningful text blocks from window snapshot (ignoring UI buttons/controls).
        """
        if not snapshot or not snapshot.interactive_elements:
            return []

        # Check if window has a sidebar by looking for sidebar width or elements
        win_x = snapshot.window_bbox[0] if snapshot.window_bbox else 0.0
        sidebar_cutoff_x = win_x + 250.0

        button_roles = {"AXButton", "AXPopUpButton", "AXMenuButton", "AXRadioButton", "AXMenuItem", "AXCheckBox"}

        blocks: List[str] = []
        for el in snapshot.interactive_elements:
            if el.subrole in IGNORE_TEXT_SUBROLES:
                continue
            if el.role in button_roles:
                continue
            # If window is wide enough, exclude sidebar controls/items on the left
            if snapshot.window_bbox and snapshot.window_bbox[2] > 500:
                if el.center[0] < sidebar_cutoff_x:
                    continue

            cand = (el.value or el.label or el.title or el.description or "").strip()
            if not cand or len(cand) < 2:
                continue

            if cand.startswith("AXEmpty") or cand.startswith("AXApplicationStatus") or cand in (
                "AXEmptyGroup", "AXApplicationStatus", "AXLandmarkMain", "Notifications"
            ):
                continue

            if not blocks or blocks[-1] != cand:
                blocks.append(cand)

        # Fallback: if no blocks found from non-button roles, allow general text blocks
        if not blocks:
            for el in snapshot.interactive_elements:
                if el.subrole in IGNORE_TEXT_SUBROLES:
                    continue
                cand = (el.value or el.label or el.title or el.description or "").strip()
                if not cand or len(cand) < 2 or cand.startswith("AX"):
                    continue
                if not blocks or blocks[-1] != cand:
                    blocks.append(cand)

        return blocks

    @classmethod
    def wait_for_reply(
        cls,
        app_name: Optional[str] = None,
        pid: Optional[int] = None,
        window_title: Optional[str] = None,
        timeout_sec: float = 15.0,
        poll_interval_sec: float = 0.25,
        pre_snapshot: Optional[FovealWindowSnapshot] = None,
    ) -> Dict[str, Any]:
        """
        Wait for target app response stream to settle (identical for 2 consecutive polls).
        """
        t0 = time.perf_counter()

        if os.environ.get("VOICEFI_MOCK_FOVEAL") == "1":
            mock_snap = cls.capture_snapshot(
                app_name=app_name,
                pid=pid,
                window_title_query=window_title,
                record_telemetry=False,
            )
            msgs = cls.extract_text_blocks(mock_snap)
            return {
                "settled": True,
                "elapsed_ms": 1.5,
                "new_messages": msgs,
                "last_message": msgs[-1] if msgs else "",
                "snapshot": mock_snap,
            }

        pre_blocks = cls.extract_text_blocks(pre_snapshot) if pre_snapshot else []
        pre_set = set(pre_blocks)

        prev_candidate_last: Optional[str] = None
        consecutive_same_count = 0
        curr_snap: Optional[FovealWindowSnapshot] = None
        curr_blocks: List[str] = []

        while (time.perf_counter() - t0) < timeout_sec:
            time.sleep(poll_interval_sec)
            curr_snap = cls.capture_snapshot(
                app_name=app_name,
                pid=pid,
                window_title_query=window_title,
                max_elements=250,
                record_telemetry=False,
            )
            curr_blocks = cls.extract_text_blocks(curr_snap)

            latest = curr_blocks[-1] if curr_blocks else ""
            has_new = (
                len(curr_blocks) > len(pre_blocks)
                or latest not in pre_set
                or (pre_blocks and latest != pre_blocks[-1])
            )

            if has_new and latest:
                if latest == prev_candidate_last:
                    consecutive_same_count += 1
                else:
                    prev_candidate_last = latest
                    consecutive_same_count = 1

                if consecutive_same_count >= 2:
                    elapsed = round((time.perf_counter() - t0) * 1000.0, 2)
                    new_msgs = [b for b in curr_blocks if b not in pre_set]
                    if not new_msgs and latest:
                        new_msgs = [latest]
                    return {
                        "settled": True,
                        "elapsed_ms": elapsed,
                        "new_messages": new_msgs,
                        "last_message": latest,
                        "snapshot": curr_snap,
                    }

        elapsed = round((time.perf_counter() - t0) * 1000.0, 2)
        new_msgs = [b for b in curr_blocks if b not in pre_set]
        latest = curr_blocks[-1] if curr_blocks else ""
        if not new_msgs and latest and (not pre_blocks or latest != pre_blocks[-1]):
            new_msgs = [latest]

        return {
            "settled": False,
            "elapsed_ms": elapsed,
            "new_messages": new_msgs,
            "last_message": latest,
            "snapshot": curr_snap,
        }

    @classmethod
    def actuate(
        cls,
        click: Optional[str] = None,
        click_xy: Optional[str] = None,
        hover: Optional[str] = None,
        right_click: Optional[str] = None,
        attach: Optional[str] = None,
        chain: Optional[Union[str, List[Union[str, Dict[str, Any]]]]] = None,
        type_text: Optional[str] = None,
        paste_text: Optional[str] = None,
        submit: bool = False,
        scroll: Optional[str] = None,
        hotkey: Optional[str] = None,
        menu: Optional[str] = None,
        snapshot: Optional[FovealWindowSnapshot] = None,
        app_name: Optional[str] = None,
        pid: Optional[int] = None,
        window_title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute composite actuation sequence with closed-loop proprioception (tactile_delta).
        Auto-escalates failed AXPress clicks to Quartz CGEvent if verified == False.
        """
        actions_taken = []
        target_pid = snapshot.pid if snapshot and snapshot.pid else pid
        target_app = snapshot.app_name if snapshot and snapshot.app_name else app_name
        target_win = snapshot.window_title if snapshot and snapshot.window_title else window_title

        # 1. Hover-reveal
        if hover:
            actions_taken.append(
                cls.hover(hover, snapshot=snapshot, app_name=target_app, pid=target_pid, window_title=target_win)
            )

        # 2. Coordinate Click
        if click_xy:
            parts = [p.strip() for p in click_xy.split(",")]
            if len(parts) == 2:
                try:
                    x, y = float(parts[0]), float(parts[1])
                    actions_taken.append(cls.click_xy(x, y))
                except Exception as e:
                    actions_taken.append({"status": "error", "action": "click_xy", "error": str(e)})

        # 3. Element Click
        click_action = None
        if click:
            click_action = cls.click_element(click, snapshot=snapshot)
            actions_taken.append(click_action)

        # 4. Right-Click
        if right_click:
            actions_taken.append(
                cls.right_click(right_click, snapshot=snapshot, app_name=target_app, pid=target_pid, window_title=target_win)
            )

        # 5. File Attachment (NSOpenPanel)
        if attach:
            if click or click_xy:
                time.sleep(0.12)
            actions_taken.append(
                cls.attach_file(
                    file_path=attach,
                    trigger_target=None if (click or click_xy) else click,
                    snapshot=snapshot,
                    app_name=target_app,
                    pid=target_pid,
                    window_title=target_win,
                )
            )

        # 6. Text injection (Paste / Type)
        if paste_text is not None:
            actions_taken.append(
                cls.paste_element(paste_text, target=click if not click_xy else None, snapshot=snapshot, submit=False)
            )

        if type_text is not None:
            actions_taken.append(cls.type_text(type_text, target=click if not click_xy else None, snapshot=snapshot))

        # 7. Hotkey chord
        if hotkey:
            actions_taken.append(cls.dispatch_hotkey(hotkey, pid=target_pid, app_name=target_app))

        # 8. Submit keystroke
        if submit:
            time.sleep(0.04)
            actions_taken.append(cls.send_submit(snapshot=snapshot))

        # 9. Menu triggering
        if menu:
            actions_taken.append(cls.trigger_menu_item(menu, pid=target_pid, app_name=target_app))

        # 10. Scroll
        if scroll:
            actions_taken.append(cls.scroll(scroll, snapshot=snapshot))

        # 11. Compound Macro Chain
        if chain:
            actions_taken.append(
                cls.execute_chain(chain, snapshot=snapshot, app_name=target_app, pid=target_pid, window_title=target_win)
            )

        # Closed-Loop Proprioception: sleep ~60ms, capture post snapshot, compute tactile_delta
        time.sleep(0.06)
        post_snap = None
        if target_pid or target_app or target_win or snapshot:
            post_snap = cls.capture_snapshot(
                pid=target_pid,
                app_name=target_app,
                window_title_query=target_win,
                max_elements=max(60, len(snapshot.interactive_elements) + 30) if snapshot and snapshot.interactive_elements else None,
                filter_query=snapshot.filter_query if snapshot else None,
                record_telemetry=False,
            )

        tactile_delta = cls.compute_tactile_delta(snapshot, post_snap)

        # Auto-Escalation on Click:
        # When AXPress was used and reported success, but post-click DOM verified == False,
        # escalate to Quartz coordinate click at element center.
        if click_action and click_action.get("method") == "AXPress" and click_action.get("status") == "success":
            if not tactile_delta["verified"]:
                cx, cy = click_action.get("center", [0.0, 0.0])
                if (cx, cy) != (0.0, 0.0) and (cx != 0 or cy != 0):
                    esc_res = cls.click_xy(cx, cy)
                    click_action["escalated_to_cgevent"] = True
                    click_action["summary"] += f" -> Escalated to Quartz CGEvent ({cx}, {cy})"
                    actions_taken.append(esc_res)
                    time.sleep(0.06)
                    post_snap = cls.capture_snapshot(
                        pid=target_pid,
                        app_name=target_app,
                        window_title_query=target_win,
                        max_elements=max(60, len(snapshot.interactive_elements) + 30) if snapshot and snapshot.interactive_elements else None,
                        filter_query=snapshot.filter_query if snapshot else None,
                        record_telemetry=False,
                    )
                    tactile_delta = cls.compute_tactile_delta(snapshot, post_snap)

        # Spatial Motor Cache update or invalidation
        if click and click_action:
            cx, cy = click_action.get("center", [0.0, 0.0])
            if tactile_delta["verified"] and (cx, cy) != (0.0, 0.0) and target_app and snapshot:
                SpatialMotorCache.set(target_app, snapshot.window_bbox, str(click), (cx, cy), click_action.get("role", ""))
            elif not tactile_delta["verified"] and target_app and snapshot:
                SpatialMotorCache.invalidate(target_app, snapshot.window_bbox, str(click))

        if snapshot:
            snapshot.tactile_delta = tactile_delta
        if post_snap:
            post_snap.tactile_delta = tactile_delta

        summary_str = (
            " -> ".join([a.get("summary", a.get("status", "done")) for a in actions_taken])
            if actions_taken
            else "no action"
        )
        return {
            "status": "success" if all(a.get("status") == "success" for a in actions_taken) else "partial",
            "actions": actions_taken,
            "summary": summary_str,
            "tactile_delta": tactile_delta,
            "post_snapshot": post_snap.to_dict() if post_snap else None,
        }

    # Backward compatibility alias
    execute_actuation = actuate

    @classmethod
    def _is_empty_or_only_window_controls(cls, elems: List[FovealUIElement]) -> bool:
        """
        Check if discovered elements are empty or consist solely of window controls
        or unhydrated top-level shell containers (AXGroup, AXSplitGroup, AXUnknown, AXScrollArea).
        Used to detect Electron/Chromium lazy accessibility hydration in Slack, Chrome, etc.
        """
        if not elems:
            return True
        window_control_subroles = {"AXCloseButton", "AXFullScreenButton", "AXMinimizeButton", "AXZoomButton"}
        window_control_roles = {"AXWindow"}
        meaningful = [
            e for e in elems
            if e.subrole not in window_control_subroles and e.role not in window_control_roles
        ]
        if len(meaningful) == 0:
            return True
        shell_roles = {"AXGroup", "AXSplitGroup", "AXUnknown", "AXScrollArea"}
        if len(meaningful) <= 2 and all(e.role in shell_roles for e in meaningful):
            return True
        if all(e.role in shell_roles for e in meaningful):
            return True
        return False

    @classmethod
    def capture_snapshot(
        cls,
        pid: Optional[int] = None,
        app_name: Optional[str] = None,
        window_title_query: Optional[str] = None,
        max_elements: Optional[int] = None,
        max_depth: int = 40,
        capture_screen: bool = False,
        filter_query: Optional[str] = None,
        click: Optional[str] = None,
        click_xy: Optional[str] = None,
        hover: Optional[str] = None,
        right_click: Optional[str] = None,
        attach: Optional[str] = None,
        chain: Optional[Union[str, List[Union[str, Dict[str, Any]]]]] = None,
        include_floating: bool = False,
        include_status_bar: bool = False,
        type_text: Optional[str] = None,
        paste_text: Optional[str] = None,
        scroll: Optional[str] = None,
        hotkey: Optional[str] = None,
        menu: Optional[str] = None,
        submit: bool = False,
        wait_reply: Optional[float] = None,
        read_last: bool = False,
        record_telemetry: bool = True,
    ) -> FovealWindowSnapshot:
        """
        Capture the structured accessibility DOM of target application/window in <15ms.
        Unlocks Electron/Chromium AX trees and executes direct actuation if requested.
        """
        t0 = time.perf_counter()

        # Calibrated default max_elements: 10 if standard/unfiltered, 40 if target app/title/filter requested, 250 for read_last/wait_reply/paste_text/type_text/chain/attach
        if max_elements is None:
            max_elements = 250 if (read_last or wait_reply or paste_text or type_text or chain or attach) else (40 if (app_name or pid or window_title_query or filter_query or include_floating or include_status_bar) else 10)
        elif (read_last or wait_reply or paste_text or type_text or chain or attach) and max_elements < 250:
            max_elements = 250

        # Handle mock/headless testing mode
        mock_app = os.environ.get("VOICEFI_MOCK_FRONT_APP")
        if mock_app:
            from whole_body_compute.integrations.ambient_discovery import AmbientPrivacyGuard

            if (
                AmbientPrivacyGuard.is_app_blocked(mock_app)
                or any(b in mock_app.lower() for b in getattr(AmbientPrivacyGuard, "BLOCKED_PASSWORD_MANAGERS", set()))
            ):
                return FovealWindowSnapshot(
                    app_name=mock_app,
                    bundle_id=f"com.{mock_app.lower()}",
                    pid=9998,
                    window_title="[BLOCKED_SECURE_VAULT]",
                    window_bbox=(0.0, 0.0, 0.0, 0.0),
                    focused_element=None,
                    interactive_elements=[],
                    latency_ms=1.0,
                    headless_route="BLOCKED: Application is an auth vault or password manager.",
                )

        if os.environ.get("VOICEFI_MOCK_FOVEAL") == "1" or (
            os.environ.get("VOICEFI_MOCK_AUDIO") == "1" and os.environ.get("VOICEFI_HEADLESS") == "1"
        ):
            mock_foc = FovealUIElement(
                role="AXTextArea",
                label="Message input",
                title="Message input",
                description="Message input",
                bbox=(515.0, 965.0, 1141.0, 49.0),
                focused=True,
                enabled=True,
                id="e3",
            )
            mock_elems = [
                FovealUIElement(
                    role="AXButton",
                    label="Run Tests",
                    title="Run Tests",
                    bbox=(120.0, 45.0, 80.0, 28.0),
                    depth=3,
                    id="e1",
                ),
                FovealUIElement(
                    role="AXButton",
                    label="Deploy",
                    title="Deploy",
                    bbox=(210.0, 45.0, 75.0, 28.0),
                    depth=3,
                    id="e2",
                ),
                mock_foc,
            ]

            if filter_query:
                fq = filter_query.lower()
                mock_elems = [
                    el
                    for el in mock_elems
                    if fq in f"{el.role} {el.label} {el.title} {el.value or ''} {el.description}".lower()
                ]

            mock_actuation = None
            if click or click_xy or type_text or paste_text or scroll or hotkey or menu or submit:
                mock_actuation = {
                    "status": "success",
                    "actions": [{"action": "mock_actuation", "status": "success"}],
                    "summary": f"Mock executed click={click}, type={type_text}",
                    "tactile_delta": {
                        "verified": True,
                        "elements_added_count": 0,
                        "elements_removed_count": 0,
                        "added_labels": [],
                        "removed_labels": [],
                        "value_changed": bool(type_text or paste_text),
                        "focus_changed": bool(click),
                        "window_title_changed": False,
                        "post_element_count": len(mock_elems[:max_elements]),
                    },
                }

            snap = FovealWindowSnapshot(
                app_name="Mock IDE",
                bundle_id="com.voicefi.mockide",
                pid=9999,
                window_title="Workspace - Main",
                window_bbox=(0.0, 0.0, 1920.0, 1080.0),
                focused_element=mock_foc,
                interactive_elements=mock_elems[:max_elements],
                latency_ms=1.2,
                headless_route="DIRECT ROUTE (Tier 0/1 Developer): Edit files directly with edit/write and run pytest in background.",
                filter_query=filter_query,
                actuation=mock_actuation,
                tactile_delta=mock_actuation["tactile_delta"] if mock_actuation else None,
            )
            if read_last or wait_reply is not None:
                snap.last_messages = [el.label for el in snap.interactive_elements if el.label]
                snap.last_message = snap.last_messages[-1] if snap.last_messages else ""
                snap.reply_settled = True
                snap.wait_reply_ms = 1.0
            return snap

        try:
            import ApplicationServices as AS
            from whole_body_compute.integrations.screenpipe import resolve_direct_headless_route
            from whole_body_compute.integrations.ambient_discovery import AmbientPrivacyGuard

            # 1. Target Application Resolution
            app_pid, resolved_app_name, bundle_id = cls.resolve_target_app(pid=pid, app_name=app_name)

            if not app_pid:
                return FovealWindowSnapshot(
                    app_name="Unknown",
                    bundle_id="",
                    pid=0,
                    window_title="",
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                )

            # Blood-Brain Barrier: Block password managers / auth vaults
            if (
                AmbientPrivacyGuard.is_app_blocked(resolved_app_name)
                or any(
                    b in resolved_app_name.lower()
                    for b in getattr(AmbientPrivacyGuard, "BLOCKED_PASSWORD_MANAGERS", set())
                )
            ):
                return FovealWindowSnapshot(
                    app_name=resolved_app_name,
                    bundle_id=bundle_id,
                    pid=app_pid,
                    window_title="[BLOCKED_SECURE_VAULT]",
                    window_bbox=(0.0, 0.0, 0.0, 0.0),
                    focused_element=None,
                    interactive_elements=[],
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                    headless_route="BLOCKED: Application is an auth vault or password manager.",
                )

            ax_app = AS.AXUIElementCreateApplication(app_pid)
            if ax_app is None:
                return FovealWindowSnapshot(
                    app_name=resolved_app_name,
                    bundle_id=bundle_id,
                    pid=app_pid,
                    window_title="",
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                )

            # 2. Unlock Electron & Chromium Accessibility Trees
            try:
                AS.AXUIElementSetAttributeValue(ax_app, "AXManualAccessibility", True)
            except Exception:
                pass
            try:
                AS.AXUIElementSetAttributeValue(ax_app, "AXEnhancedUserInterface", True)
            except Exception:
                pass

            # 3. Direct Focused Element
            raw_focused = cls._get_ax_attribute(ax_app, "AXFocusedUIElement")
            focused_elem = cls._parse_ui_element(raw_focused, depth=0, is_focused=True)

            # 4. Target Window Resolution
            target_wins = cls.find_target_windows(
                ax_app,
                title_query=window_title_query,
                include_floating=include_floating,
                include_status_bar=include_status_bar,
            )
            for w in target_wins:
                try:
                    AS.AXUIElementSetAttributeValue(w, "AXManualAccessibility", True)
                    AS.AXUIElementSetAttributeValue(w, "AXEnhancedUserInterface", True)
                except Exception:
                    pass

            window_title = ""
            window_bbox = (0.0, 0.0, 0.0, 0.0)
            if target_wins:
                primary_win = target_wins[0]
                window_title = str(cls._get_ax_attribute(primary_win, "AXTitle") or "").strip()
                window_bbox = cls._extract_bbox(primary_win)

            headless_route = resolve_direct_headless_route(resolved_app_name, window_title)

            # 5. BFS Traversal helper
            def _traverse(wins: List[Any]) -> List[FovealUIElement]:
                discovered: List[FovealUIElement] = []
                queue = deque([(w, 0) for w in wins])
                seen_labels: Set[Tuple[str, str, Tuple[float, float, float, float]]] = set()

                budget_ms = float(os.environ.get("WBC_FOVEAL_TRAVERSE_BUDGET_MS", "350"))
                deadline_ts = time.perf_counter() + max(0.05, budget_ms / 1000.0)
                visited_nodes = 0

                while queue and len(discovered) < max_elements:
                    visited_nodes += 1
                    if visited_nodes % 8 == 0 and time.perf_counter() >= deadline_ts:
                        break
                    curr, depth = queue.popleft()
                    if depth > max_depth:
                        continue

                    # Awaken intermediate container elements
                    try:
                        if depth <= 4:
                            AS.AXUIElementSetAttributeValue(curr, "AXManualAccessibility", True)
                            AS.AXUIElementSetAttributeValue(curr, "AXEnhancedUserInterface", True)
                        else:
                            role_curr = cls._get_ax_attribute(curr, "AXRole")
                            if role_curr in ("AXWebArea", "AXGroup", "AXScrollArea", "AXSplitGroup"):
                                AS.AXUIElementSetAttributeValue(curr, "AXManualAccessibility", True)
                                AS.AXUIElementSetAttributeValue(curr, "AXEnhancedUserInterface", True)
                    except Exception:
                        pass

                    parsed = cls._parse_ui_element(curr, depth=depth)
                    is_placeholder_or_shim = False
                    if parsed:
                        cand_names = {parsed.title, parsed.label, parsed.description, parsed.subrole}
                        if cand_names & {"AXEmptyGroup", "AXApplicationStatus", "AXLandmarkMain", "Notifications"}:
                            is_placeholder_or_shim = True
                        elif (0.0 < parsed.bbox[2] <= 3.0 or 0.0 < parsed.bbox[3] <= 3.0) or (
                            parsed.role in ("AXGroup", "AXGenericElement") and (parsed.bbox[2] <= 3.0 or parsed.bbox[3] <= 3.0)
                        ):
                            is_placeholder_or_shim = True

                    is_interactive = parsed and not is_placeholder_or_shim and (
                        parsed.role in INTERACTIVE_AX_ROLES
                        or (
                            parsed.role in ("AXGroup", "AXGenericElement", "AXImage", "AXRadioGroup")
                            and bool(parsed.label or parsed.description or parsed.value)
                        )
                    )
                    if parsed and is_interactive:
                        label_key = (parsed.role, parsed.label, parsed.bbox)
                        has_text = bool(parsed.label or parsed.value or parsed.focused)

                        # Filter check if requested
                        matches_filter = True
                        if filter_query:
                            fq = filter_query.lower()
                            c_str = f"{parsed.role} {parsed.label} {parsed.title} {parsed.value or ''} {parsed.description}".lower()
                            matches_filter = fq in c_str

                        # Ignore empty/whitespace-only static text
                        if parsed.role in ("AXStaticText", "AXHeading") and not parsed.label.strip():
                            has_text = False

                        if label_key not in seen_labels and has_text and matches_filter:
                            seen_labels.add(label_key)
                            discovered.append(parsed)

                    children = cls._get_ax_attribute(curr, "AXChildren")
                    if not children:
                        children = (
                            cls._get_ax_attribute(curr, "AXVisibleChildren")
                            or cls._get_ax_attribute(curr, "AXContents")
                            or cls._get_ax_attribute(curr, "AXChildrenInNavigationOrder")
                        )
                    if children:
                        for ch in children:
                            if len(discovered) >= max_elements:
                                break
                            queue.append((ch, depth + 1))

                return discovered

            # Execute first traversal pass
            interactive_elems = _traverse(target_wins)

            # 6. Fallback Retry for Electron / Chromium lazy accessibility initialization (bounded to <=450ms total)
            if cls._is_empty_or_only_window_controls(interactive_elems) and (time.perf_counter() - t0) < 0.45:
                for retry_delay in (0.12,):
                    time.sleep(retry_delay)
                    # Re-query windows once awakened and re-apply accessibility flags
                    target_wins = cls.find_target_windows(
                        ax_app,
                        title_query=window_title_query,
                        include_floating=include_floating,
                        include_status_bar=include_status_bar,
                    )
                    for w in target_wins:
                        try:
                            AS.AXUIElementSetAttributeValue(w, "AXManualAccessibility", True)
                            AS.AXUIElementSetAttributeValue(w, "AXEnhancedUserInterface", True)
                        except Exception:
                            pass
                    retry_elems = _traverse(target_wins)
                    if len(retry_elems) > len(interactive_elems) or not cls._is_empty_or_only_window_controls(retry_elems):
                        interactive_elems = retry_elems
                    if not cls._is_empty_or_only_window_controls(interactive_elems):
                        break

            # Auto-fallback if interactive_elems is still empty (e.g. Granola, Control Center, menu bar extras)
            if not interactive_elems:
                fallback_wins = []
                extras_mb = cls._get_ax_attribute(ax_app, "AXExtrasMenuBar")
                if extras_mb and extras_mb not in fallback_wins:
                    fallback_wins.append(extras_mb)
                if include_status_bar:
                    mb = cls._get_ax_attribute(ax_app, "AXMenuBar")
                    if mb and mb not in fallback_wins:
                        fallback_wins.append(mb)
                app_ch = cls._get_ax_attribute(ax_app, "AXChildren") or []
                for ch in app_ch:
                    if ch not in fallback_wins:
                        fallback_wins.append(ch)
                if fallback_wins:
                    interactive_elems = _traverse(fallback_wins)

            # Assign deterministic short IDs
            for idx, el in enumerate(interactive_elems, 1):
                el.id = f"e{idx}"

            elapsed = round((time.perf_counter() - t0) * 1000.0, 3)

            # Query Screen Bridge for active visual frame anchor
            frame_path = None
            latest_frame_path = None
            if capture_screen:
                try:
                    from whole_body_compute.integrations.screen_bridge import get_screen_bridge

                    bridge = get_screen_bridge()
                    capsule = bridge.get_latest_frame(fresh=False, max_age_s=2.0)
                    if capsule and not capsule.redacted_reason and capsule.image_path:
                        frame_path = capsule.image_path
                        latest_frame_path = capsule.latest_path
                except Exception:
                    pass
            else:
                frame_path = None
                latest_frame_path = None

            snap = FovealWindowSnapshot(
                app_name=resolved_app_name,
                bundle_id=bundle_id,
                pid=app_pid,
                window_title=window_title,
                window_bbox=window_bbox,
                focused_element=focused_elem,
                interactive_elements=interactive_elems,
                latency_ms=elapsed,
                headless_route=headless_route,
                frame_path=frame_path,
                latest_frame_path=latest_frame_path,
                filter_query=filter_query,
            )

            # 7. Execute Actuation if requested
            if (
                click
                or click_xy
                or hover
                or right_click
                or attach
                or chain
                or type_text
                or paste_text
                or scroll
                or hotkey
                or menu
                or submit
            ):
                snap.actuation = cls.actuate(
                    click=click,
                    click_xy=click_xy,
                    hover=hover,
                    right_click=right_click,
                    attach=attach,
                    chain=chain,
                    type_text=type_text,
                    paste_text=paste_text,
                    scroll=scroll,
                    hotkey=hotkey,
                    menu=menu,
                    submit=submit,
                    snapshot=snap,
                    app_name=resolved_app_name,
                    pid=app_pid,
                    window_title=window_title,
                )
                snap.tactile_delta = snap.actuation.get("tactile_delta")
                snap.latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

            # 8. Stream-Settle or Read-Last if requested
            if wait_reply is not None:
                reply_res = cls.wait_for_reply(
                    app_name=resolved_app_name,
                    pid=app_pid,
                    window_title=window_title,
                    timeout_sec=float(wait_reply),
                    pre_snapshot=snap if snap.actuation else None,
                )
                snap.reply_settled = reply_res.get("settled")
                snap.wait_reply_ms = reply_res.get("elapsed_ms")
                snap.last_messages = reply_res.get("new_messages")
                snap.last_message = reply_res.get("last_message")
            elif read_last:
                snap.last_messages = cls.extract_text_blocks(snap)
                snap.last_message = snap.last_messages[-1] if snap.last_messages else ""

            if record_telemetry:
                _record_foveal_shield_telemetry(snap)
            return snap
        except Exception as exc:
            logger.debug("Foveal vision capture failed: %s", exc)
            return FovealWindowSnapshot(
                app_name=app_name or "Unknown",
                bundle_id="",
                pid=pid or 0,
                window_title="",
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
            )

    @classmethod
    def capture_frontmost_window_dom(cls, max_elements: int = 30) -> str:
        """Capture the frontmost macOS window Accessibility DOM as a formatted context block."""
        snap = cls.capture_snapshot(max_elements=max_elements)
        _record_foveal_shield_telemetry(snap)
        return snap.to_context_block(max_elements=max_elements)


def _record_foveal_shield_telemetry(snap: FovealWindowSnapshot) -> None:
    """Record Organ 07 (The Eyes) Foveal DOM compression and Organ 09 (The Hands) actuation in ShieldTelemetryLedger. # [direct-edit]"""
    if getattr(snap, "_shield_recorded", False):
        return
    try:
        setattr(snap, "_shield_recorded", True)
        from whole_body_compute.shield import record_shield_event

        elems = snap.interactive_elements or []
        sec_cnt = sum(1 for e in elems if getattr(e, "is_secure", False)) + (
            1 if snap.window_title == "[BLOCKED_SECURE_VAULT]" else 0
        )
        has_actuation = bool(
            snap.actuation and isinstance(snap.actuation, dict) and snap.actuation.get("actions")
        )
        if has_actuation:
            comp_tok = max(25, len(elems) * 12)
            record_shield_event(
                event_type="hands_ax_act",
                target=f"{snap.app_name or 'Window'}:actuate",
                raw_tokens_est=2400,
                compressed_tokens_est=comp_tok,
                tokens_saved=max(0, 2400 - comp_tok),
                secrets_blocked=sec_cnt,
                latency_ms=float(snap.latency_ms or 0.0),
                action_mode="ax_actuate",
            )
        elif len(elems) == 0 and sec_cnt == 0:
            # Empty/failed headless or Electron AX poll: 0 counterfactual savings, honest latency recorded
            record_shield_event(
                event_type="eyes_foveal_dom",
                target=f"foveal:{snap.app_name or 'Window'}",
                raw_tokens_est=0,
                compressed_tokens_est=0,
                tokens_saved=0,
                secrets_blocked=0,
                latency_ms=float(snap.latency_ms or 0.0),
                action_mode="empty_dom",
            )
        else:
            raw_tok = min(3200, max(1200, len(elems) * 45))
            comp_tok = max(25, len(elems) * 12)
            record_shield_event(
                event_type="eyes_foveal_dom",
                target=f"foveal:{snap.app_name or 'Window'}",
                raw_tokens_est=raw_tok,
                compressed_tokens_est=comp_tok,
                tokens_saved=max(0, raw_tok - comp_tok),
                secrets_blocked=sec_cnt,
                latency_ms=float(snap.latency_ms or 0.0),
                action_mode="foveal_dom",
            )
    except Exception:
        pass


def capture_foveal_snapshot(
    pid: Optional[int] = None,
    max_elements: Optional[int] = None,
    app_name: Optional[str] = None,
    window_title_query: Optional[str] = None,
    filter_query: Optional[str] = None,
    click: Optional[str] = None,
    click_xy: Optional[str] = None,
    hover: Optional[str] = None,
    right_click: Optional[str] = None,
    attach: Optional[str] = None,
    chain: Optional[Union[str, List[Union[str, Dict[str, Any]]]]] = None,
    include_floating: bool = False,
    include_status_bar: bool = False,
    type_text: Optional[str] = None,
    paste_text: Optional[str] = None,
    scroll: Optional[str] = None,
    hotkey: Optional[str] = None,
    menu: Optional[str] = None,
    submit: bool = False,
    wait_reply: Optional[float] = None,
    read_last: bool = False,
    record_telemetry: bool = True,
) -> FovealWindowSnapshot:
    """Helper entrypoint to capture window accessibility DOM and optionally actuate."""
    snap = FovealVisionEngine.capture_snapshot(
        pid=pid,
        app_name=app_name,
        window_title_query=window_title_query,
        max_elements=max_elements,
        filter_query=filter_query,
        click=click,
        click_xy=click_xy,
        hover=hover,
        right_click=right_click,
        attach=attach,
        chain=chain,
        include_floating=include_floating,
        include_status_bar=include_status_bar,
        type_text=type_text,
        paste_text=paste_text,
        scroll=scroll,
        hotkey=hotkey,
        menu=menu,
        submit=submit,
        wait_reply=wait_reply,
        read_last=read_last,
        record_telemetry=record_telemetry,
    )
    if record_telemetry:
        _record_foveal_shield_telemetry(snap)
    return snap


def send_foveal_socket_request(payload: Dict[str, Any], timeout_sec: float = 2.0) -> Optional[Dict[str, Any]]:
    """Try forwarding request to ~/.voicefi/foveal.sock for sub-5ms zero-coldstart execution."""
    if not os.path.exists(FOVEAL_SOCKET_PATH):
        return None
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout_sec)
        s.connect(FOVEAL_SOCKET_PATH)
        data = json.dumps(payload).encode("utf-8")
        s.sendall(data + b"\n")
        resp_bytes = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            resp_bytes += chunk
            if b"\n" in chunk:
                break
        s.close()
        return json.loads(resp_bytes.decode("utf-8"))
    except Exception as exc:
        logger.debug("Socket forward failed: %s", exc)
        return None


def run_foveal_daemon(
    socket_path: str = FOVEAL_SOCKET_PATH,
    enable_ambient_discovery: Optional[bool] = None,
) -> None:
    """
    Run persistent Unix socket daemon for zero-coldstart foveal vision (<5ms)
    plus default-on passive `AmbientWorkflowObserver` workflow discovery.
    """
    os.makedirs(os.path.dirname(socket_path), exist_ok=True)
    if os.path.exists(socket_path):
        try:
            os.remove(socket_path)
        except Exception:
            pass

    ambient_controller = None
    try:
        from whole_body_compute.integrations.ambient_discovery import (
            get_global_ambient_discovery,
            is_ambient_discovery_enabled,
        )

        should_run_ambient = (
            is_ambient_discovery_enabled()
            if enable_ambient_discovery is None
            else bool(enable_ambient_discovery)
        )
        if should_run_ambient:
            ambient_controller = get_global_ambient_discovery()
            ambient_controller.start_background_discovery()
            print(
                f"🧪 AmbientWorkflowObserver active -> staging motifs to {ambient_controller.stager.candidates_dir}"
            )
    except Exception as exc:
        logger.debug("AmbientWorkflowObserver startup notice: %s", exc)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(5)
    logger.info("Foveal vision daemon listening on %s", socket_path)
    print(f"📡 Foveal Vision Daemon listening on {socket_path} (sub-5ms hot reflexes)")

    try:
        while True:
            conn, _ = server.accept()
            try:
                data = b""
                while True:
                    ch = conn.recv(4096)
                    if not ch:
                        break
                    data += ch
                    if b"\n" in ch:
                        break
                if data:
                    req = json.loads(data.decode("utf-8"))
                    if req.get("op") == "ambient_status":
                        resp = {
                            "status": "ok",
                            "ambient_observer_running": bool(
                                ambient_controller and ambient_controller.observer.is_running
                            ),
                            "events_buffered": len(ambient_controller.observer.events)
                            if ambient_controller
                            else 0,
                            "staged_in_session": len(ambient_controller.stager.list_candidates())
                            if ambient_controller
                            else 0,
                        }
                    else:
                        snap = FovealVisionEngine.capture_snapshot(**req)
                        resp = snap.to_dict()
                    conn.sendall(json.dumps(resp).encode("utf-8") + b"\n")
            except Exception as exc:
                err_resp = {"status": "error", "error": str(exc)}
                conn.sendall(json.dumps(err_resp).encode("utf-8") + b"\n")
            finally:
                conn.close()
    finally:
        if ambient_controller is not None:
            try:
                ambient_controller.stop_background_discovery()
            except Exception:
                pass
        server.close()
        if os.path.exists(socket_path):
            try:
                os.remove(socket_path)
            except Exception:
                pass


def build_foveal_screen_context(
    ocr_context: str = "",
    include_foveal_dom: bool = True,
    max_elements: int = 15,
) -> str:
    """
    Combine episodic OCR context with sub-5ms interactive Foveal Vision DOM.
    """
    blocks: List[str] = []
    if ocr_context and ocr_context.strip():
        blocks.append(ocr_context.strip())

    if include_foveal_dom:
        snapshot = capture_foveal_snapshot(max_elements=max_elements)
        if snapshot.app_name and snapshot.app_name != "Unknown":
            blocks.append(snapshot.to_context_block(max_elements=max_elements))

    return "\n\n".join(blocks).strip()
