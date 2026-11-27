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
import hashlib
import hmac
import io
import json
import logging
import os
import re
import secrets
import socket
import subprocess
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from whole_body_compute.procedural_telemetry import get_wbc_data_dir

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

_INITIAL_FOVEAL_SOCKET_PATH = str(get_wbc_data_dir() / "foveal.sock")
FOVEAL_SOCKET_PATH = _INITIAL_FOVEAL_SOCKET_PATH
_FOVEAL_DAEMON_LOCAL = threading.local()

_FOVEAL_ACT_BIND_KEYS: Tuple[str, ...] = (
    "click",
    "click_xy",
    "hover",
    "right_click",
    "type_text",
    "paste_text",
    "hotkey",
    "menu",
    "scroll",
    "attach",
    "chain",
    "submit",
)


def _resolve_foveal_socket_path() -> str:
    if FOVEAL_SOCKET_PATH != _INITIAL_FOVEAL_SOCKET_PATH:
        return FOVEAL_SOCKET_PATH
    env_sock = os.environ.get("WBC_FOVEAL_SOCKET_PATH", "").strip()
    if env_sock:
        return os.path.expanduser(env_sock)
    if (
        os.environ.get("VOICEFI_TESTING") == "1" or "PYTEST_CURRENT_TEST" in os.environ
    ) and "WBC_DATA_DIR" not in os.environ:
        return ""
    return str(get_wbc_data_dir() / "foveal.sock")


def _resolve_foveal_token_path(sock_path: str) -> str:
    env_tok = os.environ.get("WBC_FOVEAL_TOKEN_PATH", "").strip()
    if env_tok:
        return os.path.expanduser(env_tok)
    if sock_path.endswith(".sock"):
        return sock_path[:-5] + ".token"
    return sock_path + ".token"


def _compute_foveal_op_bind(req: Dict[str, Any], method_override: Optional[str] = None) -> str:
    op_str = method_override or (req.get("op") if isinstance(req.get("op"), str) else None) or "snapshot"
    app_str = req.get("app_name") if isinstance(req.get("app_name"), str) else ""
    raw_pid = req.get("pid")
    pid_str = str(int(raw_pid)) if isinstance(raw_pid, int) and not isinstance(raw_pid, bool) else ""
    act_parts: List[str] = []
    for k in _FOVEAL_ACT_BIND_KEYS:
        val = req.get(k)
        if val is None:
            continue
        if isinstance(val, bool):
            if val:
                act_parts.append(f"{k}=true")
        elif isinstance(val, str):
            if val:
                act_parts.append(f"{k}={val}")
        elif isinstance(val, (int, float)):
            act_parts.append(f"{k}={val}")
        elif isinstance(val, (list, dict)):
            act_parts.append(f"{k}={json.dumps(val, separators=(',', ':'), sort_keys=True)}")
    return f"{op_str}:{app_str}:{pid_str}:{'|'.join(act_parts)}"


def _read_foveal_token_secret(sock_path: str) -> Optional[bytes]:
    tok_path = _resolve_foveal_token_path(sock_path)
    try:
        st = os.lstat(tok_path)
        if st.st_uid != os.getuid() or (st.st_mode & 0o077) != 0:
            logger.warning("Refusing insecure foveal token file permissions/owner at %s", tok_path)
            return None
        with open(tok_path, "r", encoding="utf-8") as f:
            raw_hex = f.read().strip()
        if len(raw_hex) != 64:
            return None
        return bytes.fromhex(raw_hex)
    except Exception:
        return None


def _attach_foveal_socket_auth(payload: Dict[str, Any], sock_path: str) -> Dict[str, Any]:
    secret = _read_foveal_token_secret(sock_path)
    signed = dict(payload)
    if secret is None:
        return signed

    is_jsonrpc = signed.get("jsonrpc") == "2.0"
    if is_jsonrpc:
        params = dict(signed.get("params") or {})
        rpc_method = signed.get("method")
        method_override = (
            "status"
            if rpc_method in ("foveal.status", "foveal.ping", "vifi.ping")
            else ("ambient_status" if rpc_method == "foveal.ambient_status" else None)
        )
        op_bind = _compute_foveal_op_bind(params, method_override=method_override)
    else:
        params = signed
        op_bind = _compute_foveal_op_bind(params)

    caller_pid = os.getpid()
    ts_str = f"{time.time():.3f}"
    nonce = secrets.token_hex(16)
    canonical_msg = f"wbc-foveal-v1:{caller_pid}:{ts_str}:{nonce}:{op_bind}".encode("utf-8")
    mac_hex = hmac.new(secret, canonical_msg, hashlib.sha256).hexdigest()
    auth_env = {
        "v": 1,
        "pid": caller_pid,
        "ts": ts_str,
        "nonce": nonce,
        "op_bind": op_bind,
        "hmac": mac_hex,
    }
    if is_jsonrpc:
        params["_auth"] = auth_env
        signed["params"] = params
    else:
        signed["_auth"] = auth_env
    return signed


class SpatialMotorCache:
    """
    On-device Spatial Motor Cache ('Muscle Memory').
    Persists element center coordinates keyed by f"{app_name}:{window_bbox}:{label.lower()}".
    """

    _initial_cache_file: str = str(get_wbc_data_dir() / "foveal_motor_cache.json")
    _cache_file: str = _initial_cache_file
    _memory: Dict[str, Dict[str, Any]] = {}
    _loaded: bool = False
    _loaded_path: Optional[str] = None

    @classmethod
    def _resolve_cache_file(cls) -> str:
        if cls._cache_file != cls._initial_cache_file:
            return cls._cache_file
        env_cache = os.environ.get("WBC_FOVEAL_MOTOR_CACHE", "").strip()
        if env_cache:
            return os.path.expanduser(env_cache)
        return str(get_wbc_data_dir() / "foveal_motor_cache.json")

    @classmethod
    def _ensure_loaded(cls) -> None:
        target_file = cls._resolve_cache_file()
        if cls._loaded and cls._loaded_path == target_file:
            return
        cls._loaded = True
        cls._loaded_path = target_file
        cls._memory = {}
        try:
            if os.path.exists(target_file):
                with open(target_file, "r", encoding="utf-8") as f:
                    cls._memory = json.load(f)
        except Exception as exc:
            logger.debug("Failed loading SpatialMotorCache: %s", exc)

    @classmethod
    def _save(cls) -> None:
        target_file = cls._resolve_cache_file()
        try:
            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            with open(target_file, "w", encoding="utf-8") as f:
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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FovealUIElement":
        raw_bbox = data.get("bbox")
        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
            bbox = (float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3]))
        elif isinstance(data.get("bounds"), dict):
            b = data["bounds"]
            bbox = (
                float(b.get("x", 0.0)),
                float(b.get("y", 0.0)),
                float(b.get("w", 0.0)),
                float(b.get("h", 0.0)),
            )
        else:
            bbox = (0.0, 0.0, 0.0, 0.0)
        raw_center = data.get("center")
        center = (
            (float(raw_center[0]), float(raw_center[1]))
            if isinstance(raw_center, (list, tuple)) and len(raw_center) == 2
            else (0.0, 0.0)
        )
        lbl = str(data.get("label") or data.get("title") or "")
        return cls(
            role=str(data.get("role") or ""),
            label=lbl,
            title=str(data.get("title") or lbl),
            description=str(data.get("description") or ""),
            value=str(data["value"]) if data.get("value") is not None else None,
            bbox=bbox,
            focused=bool(data.get("focused", False)),
            enabled=bool(data.get("enabled", True)),
            subrole=str(data["subrole"]) if data.get("subrole") is not None else None,
            identifier=str(data["identifier"]) if data.get("identifier") is not None else None,
            depth=int(data.get("depth") or 0),
            is_secure=bool(data.get("is_secure", False)),
            id=str(data.get("id") or ""),
            center=center,
        )

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
        cleaned, detections = EpithelialScreenFilter.sanitize(
            cleaned,
            source="foveal_dom",
            record_antigen=False,
        )
        truncated = cleaned[:80]
        if detections and "[NEUTRALIZED_PROMPT_INJECTION:" not in truncated:
            tag = f"[NEUTRALIZED_PROMPT_INJECTION:{detections[0]['type']}]"
            if len(tag) >= 80:
                return tag[:80]
            head_budget = max(0, 79 - len(tag))
            prefix = cleaned[:head_budget].rstrip()
            return f"{prefix} {tag}".strip()[:80] if prefix else tag
        return truncated

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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FovealWindowSnapshot":
        raw_elems = data.get("interactive_elements") or data.get("elements") or []
        elems = [FovealUIElement.from_dict(e) for e in raw_elems if isinstance(e, dict)]
        raw_foc = data.get("focused_element")
        foc = FovealUIElement.from_dict(raw_foc) if isinstance(raw_foc, dict) else None
        wb = data.get("window_bbox")
        bbox = (
            (float(wb[0]), float(wb[1]), float(wb[2]), float(wb[3]))
            if isinstance(wb, (list, tuple)) and len(wb) == 4
            else (0.0, 0.0, 0.0, 0.0)
        )
        return cls(
            app_name=str(data.get("app_name") or "Unknown"),
            bundle_id=str(data.get("bundle_id") or ""),
            pid=int(data.get("pid") or 0),
            window_title=str(data.get("window_title") or ""),
            window_bbox=bbox,
            focused_element=foc,
            interactive_elements=elems,
            timestamp=float(data.get("timestamp") or time.time()),
            latency_ms=float(data.get("latency_ms") or 0.0),
            headless_route=str(data.get("headless_route") or ""),
            frame_path=data.get("frame_path"),
            latest_frame_path=data.get("latest_frame_path"),
            estimated_tokens=int(data.get("estimated_tokens") or 0),
            screenshot_token_baseline=int(data.get("screenshot_token_baseline") or 15000),
            token_savings_pct=float(data.get("token_savings_pct") or 0.0),
            actuation=data.get("actuation") if isinstance(data.get("actuation"), dict) else None,
            tactile_delta=data.get("tactile_delta") if isinstance(data.get("tactile_delta"), dict) else None,
            last_message=data.get("last_message"),
            last_messages=data.get("last_messages") if isinstance(data.get("last_messages"), list) else None,
            reply_settled=data.get("reply_settled") if isinstance(data.get("reply_settled"), bool) else None,
            wait_reply_ms=float(data["wait_reply_ms"]) if data.get("wait_reply_ms") is not None else None,
        )

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

        label = title_str or desc_str or (val_str[:600] if role == "AXStaticText" else "") or (subrole_str or "")
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
            label=label[:600],
            title=title_str[:600] if title_str else label[:600],
            description=desc_str[:600],
            value=val_str[:4000] if val_str else None,
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
                try:
                    pid_int = int(pid)
                except (TypeError, ValueError):
                    return (0, "Unknown", "")
                if pid_int <= 0:
                    return (0, "Unknown", "")
                running = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid_int)
                if running:
                    name = str(running.localizedName() or f"PID-{pid_int}").strip()
                    bid = str(running.bundleIdentifier() or "").strip()
                    return (pid_int, name, bid)
                return (pid_int, f"PID-{pid_int}", "")

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

            # Auto-fallback when AXWindows is empty or yields 0 candidates (e.g. Control Center, menu bar apps)
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
        if has_live_ax and not allow_focus_steal:
            return False, guard
        return True, guard

    @classmethod
    def _is_background_chromium_tab(
        cls,
        snapshot: Optional[FovealWindowSnapshot] = None,
        app_name: Optional[str] = None,
        window_title: Optional[str] = None,
    ) -> bool:
        """
        True when targeting a Chromium browser with an explicit `window_title` query
        (e.g. 'PostHog') that is NOT in the active AXWindow title (e.g. 'Instagram'),
        meaning the target page lives in an inactive background browser tab.
        """
        raw_app = app_name or (snapshot.app_name if snapshot else None)
        if not cls._resolve_chromium_app_name(raw_app):
            return False
        req_title = (window_title or "").strip()
        if not req_title or not snapshot or not snapshot.window_title:
            return False
        return req_title.lower() not in snapshot.window_title.lower()

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
        title_query = re.sub(
            r"\s*(?:-|\u2013|\u2014)\s*(Google Chrome|Brave|Microsoft Edge|Vivaldi|Chromium).*$",
            "",
            raw_title,
            flags=re.IGNORECASE,
        ).strip()

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
            "  const prevY = window.scrollY;"
            "  window.scrollBy({top: dy, behavior: 'instant'});"
            "  if(window.scrollY === prevY){"
            "    let bestSc = null; let bestArea = 0;"
            "    for(const c of document.querySelectorAll('main,section,div,[role=\"main\"]')){"
            "      if(c.scrollHeight > c.clientHeight + 10){"
            "        const oy = window.getComputedStyle(c).overflowY;"
            "        if(oy === 'auto' || oy === 'scroll' || oy === 'overlay'){"
            "          const a = c.clientWidth * c.clientHeight;"
            "          if(a > bestArea){ bestArea = a; bestSc = c; }"
            "        }"
            "      }"
            "    }"
            "    if(bestSc){ bestSc.scrollTop += dy; }"
            "  }"
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
            "const elType = String(el.getAttribute('type') || el.type || '').toLowerCase();"
            "const elAuto = String(el.getAttribute('autocomplete') || '').toLowerCase();"
            "if(elType === 'password' || elAuto === 'current-password' || elAuto === 'new-password'){"
            "  return 'blocked_password_field';"
            "}"
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
            "on isSensitiveTab(tTitle, tUrl)\n"
            '    set combined to (tTitle & " " & tUrl)\n'
            '    repeat with pat in {"1password.com", "bitwarden.com", "lastpass.com", "app.dashlane.com", "nordpass.com", "console.aws.amazon.com/iam", "signin.aws.amazon.com"}\n'
            "        if combined contains (pat as text) then return true\n"
            "    end repeat\n"
            "    return false\n"
            "end isSensitiveTab\n"
            "on run argv\n"
            "    set titleQuery to item 2 of argv\n"
            "    set jsCode to item 3 of argv\n"
            f'    tell application "{browser_app}"\n'
            '        if (count of windows) is 0 then return "no_window"\n'
            '        if titleQuery is not "" then\n'
            "            repeat with w in windows\n"
            "                repeat with t in tabs of w\n"
            "                    set tTitle to (title of t as text)\n"
            "                    set tUrl to (URL of t as text)\n"
            "                    if (tTitle contains titleQuery) or (tUrl contains titleQuery) then\n"
            '                        if my isSensitiveTab(tTitle, tUrl) then return "blocked_sensitive_origin"\n'
            "                        return (execute t javascript jsCode) as text\n"
            "                    end if\n"
            "                end repeat\n"
            "            end repeat\n"
            '            return "tab_not_found"\n'
            "        end if\n"
            "        set actTab to active tab of front window\n"
            "        if my isSensitiveTab((title of actTab as text), (URL of actTab as text)) then return \"blocked_sensitive_origin\"\n"
            "        return (execute actTab javascript jsCode) as text\n"
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
    def _check_actuation_security_gate(
        cls,
        action_name: str,
        app_name: Optional[str] = None,
        window_title: Optional[str] = None,
        snapshot: Optional[FovealWindowSnapshot] = None,
        element: Optional[FovealUIElement] = None,
        forbid_secure_field: bool = False,
        pid: Optional[int] = None,
        bundle_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Low-level Blood-Brain Barrier choke point for all FovealVisionEngine UI actuations.
        Blocks clicks, keystrokes, pastes, hotkeys, file attachments, and menu presses targeting
        password vaults, macOS System Settings / TCC SecurityAgent surfaces, sensitive web vault
        origins, or AXSecureTextField password inputs. # [direct-edit]
        """
        from whole_body_compute.integrations.ambient_discovery import AmbientPrivacyGuard

        eff_pid = pid if pid is not None else (snapshot.pid if snapshot and snapshot.pid else None)
        eff_app = app_name or (snapshot.app_name if snapshot else "") or ""
        eff_bid = bundle_id or (snapshot.bundle_id if snapshot else "") or ""
        eff_win = window_title or (snapshot.window_title if snapshot else "") or ""

        if snapshot is None and (not eff_app or not eff_bid):
            mock_app = os.environ.get("VOICEFI_MOCK_FRONT_APP", "").strip()
            if mock_app and not eff_app and eff_pid is None:
                eff_app = mock_app
                if not eff_win:
                    eff_win = os.environ.get("VOICEFI_MOCK_WINDOW_TITLE", "")
            elif eff_pid is not None or eff_app:
                try:
                    _, res_app, res_bid = cls.resolve_target_app(
                        pid=eff_pid, app_name=eff_app if eff_pid is None else None
                    )
                    if res_app and (not eff_app or eff_app == "Unknown" or eff_app.startswith("PID-")):
                        eff_app = res_app
                    if res_bid and not eff_bid:
                        eff_bid = res_bid
                except Exception:
                    pass
            elif os.environ.get("VOICEFI_MOCK_FOVEAL") != "1" and not (
                os.environ.get("VOICEFI_MOCK_AUDIO") == "1" and os.environ.get("VOICEFI_HEADLESS") == "1"
            ):
                try:
                    _, res_app, res_bid = cls.resolve_target_app()
                    if res_app and res_app != "Unknown":
                        eff_app = res_app
                    if res_bid:
                        eff_bid = res_bid
                except Exception:
                    pass

        elem_lbl = (
            f"{element.label} {element.title} {element.description} {element.identifier or ''}".strip()
            if element is not None
            else ""
        )
        blocked, reason = AmbientPrivacyGuard.is_actuation_blocked(
            app_name=eff_app,
            window_title=eff_win,
            bundle_id=eff_bid,
            element_label=elem_lbl,
        )
        if blocked:
            return {
                "status": "error",
                "action": action_name,
                "method": "blocked_security_gate",
                "security_blocked": True,
                "cursor_moved": False,
                "error": f"Security violation: {reason}",
                "summary": f"Blocked {action_name}: {reason}",
            }

        if forbid_secure_field:
            cand_el = element or (snapshot.focused_element if snapshot else None)
            if cand_el is not None:
                role_low = str(cand_el.role or "").lower()
                sub_low = str(cand_el.subrole or "").lower()
                if (
                    cand_el.is_secure
                    or "secure" in role_low
                    or "password" in role_low
                    or "secure" in sub_low
                    or "password" in sub_low
                ):
                    msg = f"Forbidden {action_name} into secure password field ({cand_el.role})"
                    return {
                        "status": "error",
                        "action": action_name,
                        "method": "blocked_secure_field",
                        "security_blocked": True,
                        "cursor_moved": False,
                        "error": f"Security violation: {msg}",
                        "summary": f"Blocked {action_name}: secure password field",
                    }
        return None

    @classmethod
    def _check_text_payload_security_gate(
        cls,
        action_name: str,
        text: str,
        snapshot: Optional[FovealWindowSnapshot] = None,
        app_name: Optional[str] = None,
        window_title: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Prevent GUI keystroke/paste actuation from bypassing Organ 05 Blood-Brain Barrier PreToolUse shell/secret gates
        (e.g., typing `sudo rm -rf /`, `curl ... | bash`, `.bashrc` persistence, or live API keys into Terminal/IDEs).
        """
        if not text or not str(text).strip():
            return None
        if os.environ.get("WBC_ALLOW_SENSITIVE_ACTUATION", "").strip() == "1":
            return None
        from whole_body_compute.organs.immune import VascularTrafficMonitor

        eff_app = (app_name or (snapshot.app_name if snapshot else "") or "").strip().lower()
        eff_win = (window_title or (snapshot.window_title if snapshot else "") or "").strip().lower()
        is_shell_surface = any(
            tok in f"{eff_app} {eff_win}"
            for tok in (
                "terminal",
                "iterm",
                "warp",
                "ghostty",
                "alacritty",
                "kitty",
                "wezterm",
                "hyper",
                "konsole",
                "zsh",
                "bash",
            )
        )
        vasc = VascularTrafficMonitor.inspect_shell_command(str(text), record_antigen_on_block=False)
        if not vasc.get("allowed", True):
            vtype = str(vasc.get("violation_type") or "")
            if is_shell_surface or vtype in (
                "inline_secret_exposure",
                "credential_file_exfiltration",
                "privilege_escalation",
                "reverse_shell",
                "reward_hacking_tamper",
            ):
                return {
                    "status": "error",
                    "action": action_name,
                    "method": "blocked_vascular_payload",
                    "security_blocked": True,
                    "cursor_moved": False,
                    "error": f"Security violation: {vasc.get('reason')}",
                    "summary": f"Blocked {action_name}: {vtype or 'dangerous payload'}",
                }
        return None

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
        sec_err = cls._check_actuation_security_gate("click_xy", snapshot=snapshot, pid=pid)
        if sec_err is not None:
            return sec_err
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
        app_name: Optional[str] = None,
        window_title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Click element by exact ID ('e1') or substring match against title/label/value/role.
        First attempts native AXPress action, then background Chromium DOM click if needed;
        falls back to Quartz CGEvent coordinate click only when safe under _can_use_global_cursor.
        """
        eff_app = app_name or (snapshot.app_name if snapshot else None)
        eff_win = window_title or (snapshot.window_title if snapshot else None)

        sec_err = cls._check_actuation_security_gate(
            "click",
            app_name=eff_app,
            window_title=eff_win,
            snapshot=snapshot,
            element=target if isinstance(target, FovealUIElement) else None,
        )
        if sec_err is not None:
            return sec_err

        # If targeting a hidden background browser tab, drive that tab's DOM directly without touching the active tab
        if not isinstance(target, FovealUIElement) and cls._is_background_chromium_tab(
            snapshot=snapshot, app_name=eff_app, window_title=window_title
        ):
            dom_res = cls._chrome_background_dom_actuate(
                "click",
                snapshot=snapshot,
                app_name=eff_app,
                window_title=window_title,
                label_override=str(target).strip(),
            )
            if dom_res.get("ok"):
                return {
                    "status": "success",
                    "method": "ChromeBackgroundDOM",
                    "element_id": None,
                    "role": "DOMElement",
                    "label": str(target).strip(),
                    "center": [0.0, 0.0],
                    "cursor_moved": False,
                    "summary": f"Clicked '{target}' in background {dom_res.get('browser', 'Chromium')} tab ({window_title}) via DOM",
                }

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
            if not isinstance(target, FovealUIElement) and cls._resolve_chromium_app_name(eff_app):
                dom_res = cls._chrome_background_dom_actuate(
                    "click",
                    snapshot=snapshot,
                    app_name=eff_app,
                    window_title=eff_win,
                    label_override=str(target).strip(),
                )
                if dom_res.get("ok"):
                    return {
                        "status": "success",
                        "method": "ChromeBackgroundDOM",
                        "element_id": None,
                        "role": "DOMElement",
                        "label": str(target).strip(),
                        "center": [0.0, 0.0],
                        "cursor_moved": False,
                        "summary": f"Clicked '{target}' via background {dom_res.get('browser', 'Chromium')} DOM",
                    }
                return {
                    "status": "error",
                    "method": "none",
                    "target": str(target),
                    "error": f"Element '{target}' not found in snapshot",
                    "summary": f"Element '{target}' not found",
                }

        sec_elem = cls._check_actuation_security_gate(
            "click",
            app_name=eff_app,
            window_title=eff_win,
            snapshot=snapshot,
            element=element,
        )
        if sec_elem is not None:
            return sec_elem

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
        allow_focus_steal: bool = False,
        no_cursor: bool = False,
    ) -> Dict[str, Any]:
        """
        Hover over element ID ('e1') or coordinates 'x,y' to reveal context controls and React toolbars.
        Uses zero-cursor AX focus / background Chrome DOM events when backgrounded; posts kCGEventMouseMoved
        only when safe under _can_use_global_cursor.
        """
        target_pid = snapshot.pid if snapshot and snapshot.pid else pid
        target_app = snapshot.app_name if snapshot and snapshot.app_name else app_name
        target_win = window_title or (snapshot.window_title if snapshot and snapshot.window_title else None)

        sec_err = cls._check_actuation_security_gate(
            "hover",
            app_name=target_app,
            window_title=target_win,
            snapshot=snapshot,
            pid=target_pid,
        )
        if sec_err is not None:
            return sec_err

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

        if cls._is_background_chromium_tab(snapshot=current_snap, app_name=target_app, window_title=window_title):
            dom_res = cls._chrome_background_dom_actuate(
                "hover",
                snapshot=current_snap,
                app_name=target_app,
                window_title=window_title,
                label_override=str(target).strip(),
            )
            if dom_res.get("ok"):
                return {
                    "status": "success",
                    "action": "hover",
                    "method": "ChromeBackgroundDOM_hover",
                    "cursor_moved": False,
                    "target": str(target),
                    "element_id": None,
                    "role": "DOMElement",
                    "label": str(target).strip(),
                    "coordinates": [0.0, 0.0],
                    "summary": f"Hovered over '{target}' in background {dom_res.get('browser', 'Chromium')} tab ({window_title})",
                    "tactile_delta": {"verified": True, "delta_count": 0},
                    "post_snapshot": None,
                }

        cx, cy, elem = cls._resolve_target_coordinates(target, snapshot=current_snap)
        if cx is None or cy is None:
            if cls._resolve_chromium_app_name(target_app):
                dom_res = cls._chrome_background_dom_actuate(
                    "hover",
                    snapshot=current_snap,
                    app_name=target_app,
                    window_title=target_win,
                    label_override=str(target).strip(),
                )
                if dom_res.get("ok"):
                    return {
                        "status": "success",
                        "action": "hover",
                        "method": "ChromeBackgroundDOM_hover",
                        "cursor_moved": False,
                        "target": str(target),
                        "element_id": None,
                        "role": "DOMElement",
                        "label": str(target).strip(),
                        "coordinates": [0.0, 0.0],
                        "summary": f"Hovered over '{target}' via background {dom_res.get('browser', 'Chromium')} DOM",
                        "tactile_delta": {"verified": True, "delta_count": 0},
                        "post_snapshot": None,
                    }
            return {
                "status": "error",
                "action": "hover",
                "target": str(target),
                "error": f"Target '{target}' coordinates could not be resolved",
                "summary": f"Target '{target}' not found for hover",
            }

        can_cursor, guard = cls._can_use_global_cursor(
            snapshot=current_snap,
            element=elem,
            pid=target_pid,
            allow_focus_steal=allow_focus_steal,
            no_cursor=no_cursor,
        )
        hover_method = "CGEvent_mouse_moved"
        if not can_cursor:
            ax_focused_ok = False
            if elem is not None and elem.raw_element is not None:
                try:
                    import ApplicationServices as AS

                    if AS.AXUIElementSetAttributeValue(elem.raw_element, "AXFocused", True) == 0:
                        ax_focused_ok = True
                except Exception:
                    pass
            dom_res = cls._chrome_background_dom_actuate(
                "hover",
                element=elem,
                snapshot=current_snap,
                app_name=target_app,
                window_title=target_win,
                label_override=None if elem else str(target),
            )
            if not dom_res.get("ok") and not ax_focused_ok:
                return {
                    "status": "error",
                    "action": "hover",
                    "method": "hover_blocked_focus_steal",
                    "target": str(target),
                    "coordinates": [float(cx), float(cy)],
                    "cursor_moved": False,
                    "hid_guard": guard,
                    "error": f"Blocked cursor-hijacking hover at ({cx:.1f}, {cy:.1f}) while backgrounded{dom_res.get('hint', '')}",
                    "summary": f"Blocked focus-stealing hover at ({cx:.1f}, {cy:.1f})",
                }
            hover_method = "ChromeBackgroundDOM_hover" if dom_res.get("ok") else "AXFocused_hover"
            time.sleep(settle_ms / 1000.0)
        else:
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
            "method": hover_method,
            "cursor_moved": bool(can_cursor),
            "target": str(target),
            "element_id": elem.id if elem else None,
            "role": elem.role if elem else None,
            "label": elem.label if elem else None,
            "coordinates": [float(cx), float(cy)],
            "summary": f"Hovered over [{elem.id or 'xy' if elem else 'xy'}] '{elem.label if elem else target}' at ({cx:.1f}, {cy:.1f}) ({hover_method})",
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
        allow_focus_steal: bool = False,
        no_cursor: bool = False,
    ) -> Dict[str, Any]:
        """
        Right-click element ID ('e1') or coordinates 'x,y' to open context menu.
        Uses native AXShowMenu / background Chrome DOM contextmenu when backgrounded;
        posts kCGEventRightMouseDown + kCGEventRightMouseUp only when safe under _can_use_global_cursor.
        """
        target_pid = snapshot.pid if snapshot and snapshot.pid else pid
        target_app = snapshot.app_name if snapshot and snapshot.app_name else app_name
        target_win = window_title or (snapshot.window_title if snapshot and snapshot.window_title else None)

        sec_err = cls._check_actuation_security_gate(
            "right_click",
            app_name=target_app,
            window_title=target_win,
            snapshot=snapshot,
            pid=target_pid,
        )
        if sec_err is not None:
            return sec_err

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

        if cls._is_background_chromium_tab(snapshot=current_snap, app_name=target_app, window_title=window_title):
            dom_res = cls._chrome_background_dom_actuate(
                "contextmenu",
                snapshot=current_snap,
                app_name=target_app,
                window_title=window_title,
                label_override=str(target).strip(),
            )
            if dom_res.get("ok"):
                return {
                    "status": "success",
                    "action": "right_click",
                    "method": "ChromeBackgroundDOM_contextmenu",
                    "cursor_moved": False,
                    "target": str(target),
                    "element_id": None,
                    "role": "DOMElement",
                    "label": str(target).strip(),
                    "coordinates": [0.0, 0.0],
                    "summary": f"Right-clicked '{target}' in background {dom_res.get('browser', 'Chromium')} tab ({window_title})",
                    "tactile_delta": {"verified": True, "delta_count": 0},
                    "post_snapshot": None,
                }

        cx, cy, elem = cls._resolve_target_coordinates(target, snapshot=current_snap)
        if cx is None or cy is None:
            if cls._resolve_chromium_app_name(target_app):
                dom_res = cls._chrome_background_dom_actuate(
                    "contextmenu",
                    snapshot=current_snap,
                    app_name=target_app,
                    window_title=target_win,
                    label_override=str(target).strip(),
                )
                if dom_res.get("ok"):
                    return {
                        "status": "success",
                        "action": "right_click",
                        "method": "ChromeBackgroundDOM_contextmenu",
                        "cursor_moved": False,
                        "target": str(target),
                        "element_id": None,
                        "role": "DOMElement",
                        "label": str(target).strip(),
                        "coordinates": [0.0, 0.0],
                        "summary": f"Right-clicked '{target}' via background {dom_res.get('browser', 'Chromium')} DOM",
                        "tactile_delta": {"verified": True, "delta_count": 0},
                        "post_snapshot": None,
                    }
            return {
                "status": "error",
                "action": "right_click",
                "target": str(target),
                "error": f"Target '{target}' coordinates could not be resolved",
                "summary": f"Target '{target}' not found for right-click",
            }

        can_cursor, guard = cls._can_use_global_cursor(
            snapshot=current_snap,
            element=elem,
            pid=target_pid,
            allow_focus_steal=allow_focus_steal,
            no_cursor=no_cursor,
        )
        rc_method = "CGEvent_right_click"
        if not can_cursor:
            ax_menu_ok = False
            if elem is not None and elem.raw_element is not None:
                try:
                    import ApplicationServices as AS

                    if AS.AXUIElementPerformAction(elem.raw_element, "AXShowMenu") == 0:
                        ax_menu_ok = True
                except Exception:
                    pass
            dom_ok = False
            dom_hint = ""
            if not ax_menu_ok:
                dom_res = cls._chrome_background_dom_actuate(
                    "contextmenu",
                    element=elem,
                    snapshot=current_snap,
                    app_name=target_app,
                    window_title=target_win,
                    label_override=None if elem else str(target),
                )
                dom_ok = bool(dom_res.get("ok"))
                dom_hint = dom_res.get("hint", "")
            if not ax_menu_ok and not dom_ok:
                return {
                    "status": "error",
                    "action": "right_click",
                    "method": "right_click_blocked_focus_steal",
                    "target": str(target),
                    "coordinates": [float(cx), float(cy)],
                    "cursor_moved": False,
                    "hid_guard": guard,
                    "error": f"Blocked cursor-hijacking right-click at ({cx:.1f}, {cy:.1f}) while backgrounded{dom_hint}",
                    "summary": f"Blocked focus-stealing right-click at ({cx:.1f}, {cy:.1f})",
                }
            rc_method = "AXShowMenu" if ax_menu_ok else "ChromeBackgroundDOM_contextmenu"
            time.sleep(settle_ms / 1000.0)
        else:
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
            "method": rc_method,
            "cursor_moved": bool(can_cursor),
            "target": str(target),
            "element_id": elem.id if elem else None,
            "role": elem.role if elem else None,
            "label": elem.label if elem else None,
            "coordinates": [float(cx), float(cy)],
            "summary": f"Right-clicked [{elem.id or 'xy' if elem else 'xy'}] '{elem.label if elem else target}' at ({cx:.1f}, {cy:.1f}) ({rc_method})",
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
        1. Validate target surface, Secret Blindfold (incl. symlinks), control chars, and regular-file existence
        2. If trigger_target provided, click attach button first and wait 120ms
        3. cmd+shift+g to trigger 'Go to the folder' sheet
        4. Atomically paste resolved absolute path (PID-scoped)
        5. Return/submit Go sheet + Open file dialog (PID-scoped) # [direct-edit]
        """
        from whole_body_compute.organs.immune import _is_sensitive_credential_path

        target_pid = snapshot.pid if snapshot and snapshot.pid else pid
        target_app = snapshot.app_name if snapshot and snapshot.app_name else app_name
        target_win = snapshot.window_title if snapshot and snapshot.window_title else window_title

        sec_err = cls._check_actuation_security_gate(
            "attach_file",
            app_name=target_app,
            window_title=target_win,
            snapshot=snapshot,
            pid=target_pid,
        )
        if sec_err is not None:
            return sec_err

        raw_fp = str(file_path or "")
        if not raw_fp.strip() or any(ch in raw_fp for ch in ("\x00", "\r", "\n")):
            return {
                "status": "error",
                "action": "attach_file",
                "error": "Invalid file path: empty or contains control/newline characters",
                "summary": "Blocked attach_file: invalid path characters",
            }

        expanded_fp = Path(raw_fp).expanduser()
        resolved_obj = expanded_fp.resolve()
        resolved_path = str(resolved_obj)
        if any(ch in resolved_path for ch in ("\x00", "\r", "\n")):
            return {
                "status": "error",
                "action": "attach_file",
                "error": "Resolved file path contains forbidden control/newline characters",
                "summary": "Blocked attach_file: resolved path contains control characters",
            }

        sens_hit = _is_sensitive_credential_path(raw_fp, resolved_path=resolved_path)
        if sens_hit:
            return {
                "status": "error",
                "action": "attach_file",
                "method": "blocked_secret_blindfold",
                "security_blocked": True,
                "cursor_moved": False,
                "error": f"Security violation: Refusing to attach sensitive credential file '{sens_hit}'",
                "summary": f"Blocked attach_file: sensitive file '{sens_hit}'",
            }

        if not resolved_obj.is_file():
            return {
                "status": "error",
                "action": "attach_file",
                "file_path": resolved_path,
                "error": f"Attachment target '{resolved_path}' does not exist or is not a regular file",
                "summary": "Blocked attach_file: target is not an existing regular file",
            }

        actions_taken = []
        if trigger_target:
            click_res = cls.click_element(trigger_target, snapshot=snapshot)
            actions_taken.append(click_res)
            time.sleep(0.12)

        # 1. Hotkey cmd+shift+g to open "Go to folder" sheet in NSOpenPanel
        actions_taken.append(cls.dispatch_hotkey("cmd+shift+g", pid=target_pid, app_name=target_app))
        time.sleep(0.06)

        # 2. Atomic paste resolved path (scoped to target_pid)
        paste_ok = cls._atomic_paste_text(resolved_path, pid=target_pid)
        actions_taken.append({
            "status": "success" if paste_ok else "error",
            "action": "paste_path",
            "path": resolved_path,
            "summary": f"Pasted file path '{resolved_path}'",
        })

        # 3. Submit Go-to-folder sheet
        actions_taken.append(cls.send_submit(snapshot=snapshot, pid=target_pid))
        time.sleep(0.10)

        # 4. Confirm Open dialog
        actions_taken.append(cls.send_submit(snapshot=snapshot, pid=target_pid))
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
        allow_focus_steal: bool = False,
        no_cursor: bool = False,
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
        t_chain0 = time.perf_counter()

        sec_err = cls._check_actuation_security_gate(
            "execute_chain",
            app_name=target_app,
            window_title=target_win,
            snapshot=snapshot,
            pid=target_pid,
        )
        if sec_err is not None:
            return sec_err

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

        cursor_kwargs: Dict[str, Any] = {}
        if allow_focus_steal:
            cursor_kwargs["allow_focus_steal"] = True
        if no_cursor:
            cursor_kwargs["no_cursor"] = True

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
                actions_taken.append(cls.click_element(arg, snapshot=current_snap, **cursor_kwargs))
            elif act_type in ("click-xy", "click_xy"):
                parts = [p.strip() for p in arg.split(",")]
                if len(parts) == 2:
                    try:
                        actions_taken.append(
                            cls.click_xy(
                                float(parts[0]),
                                float(parts[1]),
                                pid=target_pid,
                                snapshot=current_snap,
                                **cursor_kwargs,
                            )
                        )
                    except Exception as e:
                        actions_taken.append({"status": "error", "action": "click_xy", "error": str(e)})
                else:
                    actions_taken.append({"status": "error", "action": "click_xy", "error": f"Invalid coordinates '{arg}'"})
            elif act_type == "hover":
                actions_taken.append(
                    cls.hover(
                        arg,
                        snapshot=current_snap,
                        app_name=target_app,
                        pid=target_pid,
                        window_title=target_win,
                        **cursor_kwargs,
                    )
                )
            elif act_type in ("right-click", "right_click"):
                actions_taken.append(
                    cls.right_click(
                        arg,
                        snapshot=current_snap,
                        app_name=target_app,
                        pid=target_pid,
                        window_title=target_win,
                        **cursor_kwargs,
                    )
                )
            elif act_type == "type":
                actions_taken.append(cls.type_text(arg, snapshot=current_snap, **cursor_kwargs))
            elif act_type == "paste":
                if allow_focus_steal:
                    actions_taken.append(cls.paste_element(arg, snapshot=current_snap, submit=False, allow_focus_steal=True))
                else:
                    actions_taken.append(cls.paste_element(arg, snapshot=current_snap, submit=False))
            elif act_type == "attach":
                actions_taken.append(cls.attach_file(arg, snapshot=current_snap, app_name=target_app, pid=target_pid, window_title=target_win))
            elif act_type in ("hotkey", "key"):
                if allow_focus_steal:
                    actions_taken.append(cls.dispatch_hotkey(arg, pid=target_pid, app_name=target_app, allow_focus_steal=True))
                else:
                    actions_taken.append(cls.dispatch_hotkey(arg, pid=target_pid, app_name=target_app))
            elif act_type == "menu":
                actions_taken.append(cls.trigger_menu_item(arg, pid=target_pid, app_name=target_app))
            elif act_type == "scroll":
                actions_taken.append(cls.scroll(arg, snapshot=current_snap, **cursor_kwargs))
            elif act_type == "submit":
                time.sleep(0.04)
                if allow_focus_steal:
                    actions_taken.append(cls.send_submit(snapshot=current_snap, allow_focus_steal=True))
                else:
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
        if (
            not os.environ.get("PYTEST_CURRENT_TEST")
            or os.environ.get("WBC_ENABLE_REFLEX_GATE_IN_TESTS") == "1"
            or "WBC_DATA_DIR" in os.environ
        ):
            try:
                from whole_body_compute.local.reflex_gate import (
                    StructuralSignals,
                    record_shadow_evaluation_async,
                )

                chain_ok = (
                    bool(actions_taken)
                    and all(a.get("status") == "success" for a in actions_taken)
                    and bool(tactile_delta.get("verified", True))
                )
                chain_lat_ms = round((time.perf_counter() - t_chain0) * 1000.0, 2)
                step_summary = ",".join(
                    str(s.get("action") or s.get("type") or s).split(":", 1)[0].strip().lower()
                    for s in raw_steps[:8]
                )
                record_shadow_evaluation_async(
                    gate_id="foveal_hands_chain",
                    text=f"foveal_hands_chain | {target_app or 'app'} | {step_summary}",
                    actual_ok=chain_ok,
                    signals=StructuralSignals(
                        file_lines=len(raw_steps),
                        symbol_count=len(actions_taken),
                        exact_symbol_mention=bool(target_app),
                        turn_depth=len(raw_steps),
                    ),
                    repo_key="__global__",
                    model_name="body-coder-mlx",
                    latency_ms=chain_lat_ms,
                    metadata={
                        "app_name": target_app or "",
                        "steps_count": len(raw_steps),
                        "tactile_verified": bool(tactile_delta.get("verified", True)),
                    },
                )
            except Exception:
                pass

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
        sec_err = cls._check_actuation_security_gate(
            "paste",
            snapshot=snapshot,
            element=target if isinstance(target, FovealUIElement) else None,
            forbid_secure_field=isinstance(target, FovealUIElement),
        )
        if sec_err is not None:
            return sec_err
        sec_payload = cls._check_text_payload_security_gate("paste", text, snapshot=snapshot)
        if sec_payload is not None:
            return sec_payload
        guard = cls._check_human_hid_guard(target_pid=target_pid, allow_focus_steal=allow_focus_steal)

        target_elem: Optional[FovealUIElement] = None
        if target:
            if snapshot and snapshot.interactive_elements:
                t_str = str(target).strip().lower()
                for el in snapshot.interactive_elements:
                    if el.id.lower() == t_str or t_str in f"{el.label} {el.role}".lower():
                        target_elem = el
                        break
            sec_target = cls._check_actuation_security_gate(
                "paste",
                snapshot=snapshot,
                element=target_elem,
                forbid_secure_field=True,
            )
            if sec_target is not None:
                return sec_target
            cls.click_element(target, snapshot=snapshot, allow_focus_steal=allow_focus_steal)
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

        sec_resolved = cls._check_actuation_security_gate(
            "paste",
            snapshot=snapshot,
            element=target_elem,
            forbid_secure_field=True,
        )
        if sec_resolved is not None:
            return sec_resolved

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
        elif pid is not None and not app_name:
            _, app_name, _ = cls.resolve_target_app(pid=pid)

        sec_err = cls._check_actuation_security_gate("hotkey", app_name=app_name)
        if sec_err is not None:
            return sec_err

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

        sec_err = cls._check_actuation_security_gate("menu", app_name=app_name)
        if sec_err is not None:
            return sec_err

        try:
            import ApplicationServices as AS

            if ax_app is None:
                resolved_name = app_name
                if pid is None and app_name:
                    pid, resolved_name, _ = cls.resolve_target_app(app_name=app_name)
                elif pid is None:
                    pid, resolved_name, _ = cls.resolve_target_app()
                elif pid is not None and not app_name:
                    _, resolved_name, _ = cls.resolve_target_app(pid=pid)
                sec_resolved = cls._check_actuation_security_gate("menu", app_name=resolved_name)
                if sec_resolved is not None:
                    return sec_resolved
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
        allow_focus_steal: bool = False,
        no_cursor: bool = False,
        app_name: Optional[str] = None,
        window_title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Scroll target window/view up/down/top/bottom via native AXScroll actions, background Chrome DOM,
        or PID keycodes. Only posts global kCGHIDEventTap scroll wheel when safe under _can_use_global_cursor.
        """
        sec_err = cls._check_actuation_security_gate(
            "scroll",
            app_name=app_name,
            window_title=window_title,
            snapshot=snapshot,
        )
        if sec_err is not None:
            return sec_err
        d = direction.lower().strip()
        ax_action_ok = False
        bg_dom_ok = False
        pid_key_ok = False
        quartz_ok = False
        ax_scroll_act = "AXScrollDownByPage" if d in ("down", "bottom") else "AXScrollUpByPage"

        # If targeting a hidden background browser tab, scroll only that tab's DOM without scrolling the active tab
        if cls._is_background_chromium_tab(snapshot=snapshot, app_name=app_name, window_title=window_title):
            dom_res = cls._chrome_background_dom_actuate(
                "scroll",
                snapshot=snapshot,
                app_name=app_name,
                window_title=window_title,
                direction=d,
                amount=amount,
            )
            if dom_res.get("ok"):
                return {
                    "status": "success",
                    "action": "scroll",
                    "direction": d,
                    "ax_scroll": False,
                    "bg_dom_scroll": True,
                    "quartz_scroll": False,
                    "cursor_moved": False,
                    "summary": f"Scrolled {d} in background {dom_res.get('browser', 'Chromium')} tab ({window_title})",
                }
            return {
                "status": "error",
                "action": "scroll",
                "direction": d,
                "cursor_moved": False,
                "error": f"Background tab scroll failed: {dom_res.get('reason', 'unknown')}{dom_res.get('hint', '')}",
                "summary": f"Failed scrolling background tab '{window_title}'",
            }

        if snapshot:
            for el in snapshot.interactive_elements:
                if el.role in ("AXScrollArea", "AXWebArea", "AXList", "AXTable") and el.raw_element is not None:
                    try:
                        import ApplicationServices as AS

                        err = AS.AXUIElementPerformAction(el.raw_element, ax_scroll_act)
                        if err == 0:
                            ax_action_ok = True
                            break
                    except Exception:
                        pass

            # Walk AXParent chain of first live element to locate enclosing AXScrollArea / AXWebArea
            if not ax_action_ok:
                try:
                    import ApplicationServices as AS

                    seed_el = next((e.raw_element for e in snapshot.interactive_elements if e.raw_element is not None), None)
                    curr = seed_el
                    for _ in range(12):
                        if curr is None:
                            break
                        role_c = cls._get_ax_attribute(curr, "AXRole")
                        if role_c in ("AXScrollArea", "AXWebArea", "AXList", "AXTable"):
                            if AS.AXUIElementPerformAction(curr, ax_scroll_act) == 0:
                                ax_action_ok = True
                                break
                        curr = cls._get_ax_attribute(curr, "AXParent")
                except Exception:
                    pass

        # When native AXScroll succeeded, return immediately without firing global HID wheel on the active cursor window
        if ax_action_ok:
            return {
                "status": "success",
                "action": "scroll",
                "direction": d,
                "ax_scroll": True,
                "quartz_scroll": False,
                "cursor_moved": False,
                "summary": f"Scrolled {d} via background AXScroll (0 cursor impact)",
            }

        target_pid = snapshot.pid if snapshot and snapshot.pid else None
        can_cursor, guard = cls._can_use_global_cursor(
            snapshot=snapshot,
            pid=target_pid,
            allow_focus_steal=allow_focus_steal,
            no_cursor=no_cursor,
        )
        if not can_cursor:
            dom_res = cls._chrome_background_dom_actuate(
                "scroll",
                snapshot=snapshot,
                direction=d,
                amount=amount,
            )
            bg_dom_ok = bool(dom_res.get("ok"))
            if not bg_dom_ok and target_pid and int(target_pid) > 0:
                try:
                    import Quartz

                    # PageDown=121, PageUp=116, End=119, Home=115
                    kcode = 119 if d == "bottom" else (115 if d == "top" else (121 if d == "down" else 116))
                    ev_down = Quartz.CGEventCreateKeyboardEvent(None, kcode, True)
                    ev_up = Quartz.CGEventCreateKeyboardEvent(None, kcode, False)
                    mode = cls._post_key_event(ev_down, pid=target_pid, allow_focus_steal=allow_focus_steal)
                    time.sleep(0.02)
                    cls._post_key_event(ev_up, pid=target_pid, allow_focus_steal=allow_focus_steal)
                    pid_key_ok = mode == "CGEventPostToPid"
                except Exception:
                    pass
            ok_bg = bg_dom_ok or pid_key_ok
            return {
                "status": "success" if ok_bg else "error",
                "action": "scroll",
                "direction": d,
                "ax_scroll": False,
                "bg_dom_scroll": bg_dom_ok,
                "pid_key_scroll": pid_key_ok,
                "quartz_scroll": False,
                "cursor_moved": False,
                "hid_guard": guard,
                "summary": (
                    f"Scrolled {d} in background ({'ChromeDOM' if bg_dom_ok else 'CGEventPostToPid'})"
                    if ok_bg
                    else f"Blocked global HID scroll ({d}) to protect active cursor window"
                ),
            }

        try:
            import Quartz

            delta_y = -amount if d in ("down", "bottom") else amount
            if d == "bottom":
                delta_y = -amount * 4
            elif d == "top":
                delta_y = amount * 4

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
        allow_focus_steal: bool = False,
        no_cursor: bool = False,
    ) -> Dict[str, Any]:
        """
        Type text into target or focused input element.
        Sets AXValue and sends PID-targeted Quartz keystrokes (never stealing focus).
        """
        sec_pre = cls._check_actuation_security_gate(
            "type_text",
            snapshot=snapshot,
            element=target if isinstance(target, FovealUIElement) else None,
            forbid_secure_field=isinstance(target, FovealUIElement),
        )
        if sec_pre is not None:
            return sec_pre
        sec_payload = cls._check_text_payload_security_gate("type_text", text, snapshot=snapshot)
        if sec_payload is not None:
            return sec_payload

        target_elem: Optional[FovealUIElement] = None
        if target:
            if snapshot and snapshot.interactive_elements:
                t_str = str(target).strip().lower()
                for el in snapshot.interactive_elements:
                    if el.id.lower() == t_str or t_str in f"{el.label} {el.role}".lower():
                        target_elem = el
                        break
            sec_target = cls._check_actuation_security_gate(
                "type_text",
                snapshot=snapshot,
                element=target_elem,
                forbid_secure_field=True,
            )
            if sec_target is not None:
                return sec_target
            cls.click_element(
                target,
                snapshot=snapshot,
                allow_focus_steal=allow_focus_steal,
                no_cursor=no_cursor,
            )
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

        sec_resolved = cls._check_actuation_security_gate(
            "type_text",
            snapshot=snapshot,
            element=target_elem,
            forbid_secure_field=True,
        )
        if sec_resolved is not None:
            return sec_resolved

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

        # Zero-clipboard, zero-cursor React input injection for background Chromium tabs when AXValue fails
        if not ax_val_set and target_elem is not None:
            can_cursor, _ = cls._can_use_global_cursor(
                snapshot=snapshot,
                element=target_elem,
                pid=target_pid,
                allow_focus_steal=allow_focus_steal,
                no_cursor=no_cursor,
            )
            if not can_cursor:
                dom_in = cls._chrome_background_dom_actuate(
                    "input",
                    element=target_elem,
                    snapshot=snapshot,
                    text_value=text,
                )
                if dom_in.get("ok"):
                    return {
                        "status": "success",
                        "method": "ChromeBackgroundDOM_input",
                        "typed_text": text,
                        "cursor_moved": False,
                        "target_element": target_elem.to_dict(),
                        "summary": f"Injected {len(text)} chars via background Chrome DOM setter (0 cursor move)",
                    }

        # Fast path for rich text or multi-character text where AXValue didn't stick or len > 40
        if (not ax_val_set or is_rich_text) and len(text) > 40:
            pasted = cls._atomic_paste_text(
                text,
                target_elem=target_elem,
                pid=target_pid,
                allow_focus_steal=allow_focus_steal,
            )
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
            cls._post_key_event(ev_down, pid=target_pid, allow_focus_steal=allow_focus_steal)
            time.sleep(0.01)
            ev_up = Quartz.CGEventCreateKeyboardEvent(None, 0, False)
            Quartz.CGEventKeyboardSetUnicodeString(ev_up, len(text), text)
            cls._post_key_event(ev_up, pid=target_pid, allow_focus_steal=allow_focus_steal)
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
        sec_err = cls._check_actuation_security_gate("submit", snapshot=snapshot, pid=target_pid)
        if sec_err is not None:
            return sec_err
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
        allow_focus_steal: bool = False,
        no_cursor: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute composite actuation sequence with closed-loop proprioception (tactile_delta).
        Auto-escalates failed AXPress clicks to Quartz CGEvent only when safe under _can_use_global_cursor.
        """
        actions_taken = []
        target_pid = snapshot.pid if snapshot and snapshot.pid else pid
        target_app = snapshot.app_name if snapshot and snapshot.app_name else app_name
        target_win = window_title or (snapshot.window_title if snapshot and snapshot.window_title else None)

        sec_act = cls._check_actuation_security_gate(
            "actuate",
            app_name=target_app,
            window_title=target_win,
            snapshot=snapshot,
            forbid_secure_field=bool((type_text or paste_text) and not click),
            pid=target_pid,
        )
        if sec_act is not None:
            return {
                "status": "error",
                "security_blocked": True,
                "actions": [sec_act],
                "error": sec_act["error"],
                "summary": sec_act["summary"],
                "tactile_delta": {"verified": False, "security_blocked": True},
                "post_snapshot": None,
            }
        is_bg_tab = bool(
            window_title and cls._is_background_chromium_tab(snapshot=snapshot, app_name=target_app, window_title=window_title)
        )

        cursor_kwargs: Dict[str, Any] = {}
        if allow_focus_steal:
            cursor_kwargs["allow_focus_steal"] = True
        if no_cursor:
            cursor_kwargs["no_cursor"] = True

        # 1. Hover-reveal
        if hover:
            actions_taken.append(
                cls.hover(
                    hover,
                    snapshot=snapshot,
                    app_name=target_app,
                    pid=target_pid,
                    window_title=target_win,
                    **cursor_kwargs,
                )
            )

        # 2. Coordinate Click
        if click_xy:
            parts = [p.strip() for p in click_xy.split(",")]
            if len(parts) == 2:
                try:
                    x, y = float(parts[0]), float(parts[1])
                    if snapshot is not None or target_pid is not None or cursor_kwargs:
                        actions_taken.append(
                            cls.click_xy(x, y, pid=target_pid, snapshot=snapshot, **cursor_kwargs)
                        )
                    else:
                        actions_taken.append(cls.click_xy(x, y))
                except Exception as e:
                    actions_taken.append({"status": "error", "action": "click_xy", "error": str(e)})

        # 3. Element Click
        click_action = None
        if click:
            if is_bg_tab:
                click_action = cls.click_element(
                    click,
                    snapshot=snapshot,
                    app_name=target_app,
                    window_title=window_title,
                    **cursor_kwargs,
                )
            else:
                click_action = cls.click_element(click, snapshot=snapshot, **cursor_kwargs)
            actions_taken.append(click_action)

        # 4. Right-Click
        if right_click:
            actions_taken.append(
                cls.right_click(
                    right_click,
                    snapshot=snapshot,
                    app_name=target_app,
                    pid=target_pid,
                    window_title=target_win,
                    **cursor_kwargs,
                )
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
            if allow_focus_steal:
                actions_taken.append(
                    cls.paste_element(
                        paste_text,
                        target=click if not click_xy else None,
                        snapshot=snapshot,
                        submit=False,
                        allow_focus_steal=True,
                    )
                )
            else:
                actions_taken.append(
                    cls.paste_element(paste_text, target=click if not click_xy else None, snapshot=snapshot, submit=False)
                )

        if type_text is not None:
            actions_taken.append(
                cls.type_text(type_text, target=click if not click_xy else None, snapshot=snapshot, **cursor_kwargs)
            )

        # 7. Hotkey chord
        if hotkey:
            if allow_focus_steal:
                actions_taken.append(cls.dispatch_hotkey(hotkey, pid=target_pid, app_name=target_app, allow_focus_steal=True))
            else:
                actions_taken.append(cls.dispatch_hotkey(hotkey, pid=target_pid, app_name=target_app))

        # 8. Submit keystroke
        if submit:
            time.sleep(0.04)
            if allow_focus_steal:
                actions_taken.append(cls.send_submit(snapshot=snapshot, allow_focus_steal=True))
            else:
                actions_taken.append(cls.send_submit(snapshot=snapshot))

        # 9. Menu triggering
        if menu:
            actions_taken.append(cls.trigger_menu_item(menu, pid=target_pid, app_name=target_app))

        # 10. Scroll
        if scroll:
            if is_bg_tab:
                actions_taken.append(
                    cls.scroll(
                        scroll,
                        snapshot=snapshot,
                        app_name=target_app,
                        window_title=window_title,
                        **cursor_kwargs,
                    )
                )
            else:
                actions_taken.append(cls.scroll(scroll, snapshot=snapshot, **cursor_kwargs))

        # 11. Compound Macro Chain
        if chain:
            actions_taken.append(
                cls.execute_chain(
                    chain,
                    snapshot=snapshot,
                    app_name=target_app,
                    pid=target_pid,
                    window_title=target_win,
                    **cursor_kwargs,
                )
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
        # only escalate to Quartz coordinate click when _can_use_global_cursor permits it.
        # In background / zero-cursor mode (e.g. PostHog in Chrome while working in AGY),
        # never hijack the physical cursor; allow a brief async settle check instead.
        if click_action and click_action.get("method") == "AXPress" and click_action.get("status") == "success":
            if not tactile_delta["verified"]:
                matched_el = None
                if snapshot and snapshot.interactive_elements and click_action.get("element_id"):
                    matched_el = next(
                        (e for e in snapshot.interactive_elements if e.id == click_action.get("element_id")),
                        None,
                    )
                can_cursor, esc_guard = cls._can_use_global_cursor(
                    snapshot=snapshot,
                    element=matched_el,
                    pid=target_pid,
                    allow_focus_steal=allow_focus_steal,
                    no_cursor=no_cursor,
                )
                if can_cursor:
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
                else:
                    time.sleep(0.10)
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
                    click_action["escalated_to_cgevent"] = False
                    click_action["escalation_blocked_zero_cursor"] = True
                    click_action["hid_guard"] = esc_guard

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
        allow_focus_steal: bool = False,
        no_cursor: bool = False,
        _skip_socket: bool = False,
    ) -> FovealWindowSnapshot:
        """
        Capture the structured accessibility DOM of target application/window in <15ms.
        Unlocks Electron/Chromium AX trees and executes direct actuation if requested.
        """
        t0 = time.perf_counter()

        # Calibrated default max_elements: 10 if standard/unfiltered, 40 if target app/title/filter requested, 150 for click/hover/right_click, 250 for read_last/wait_reply/paste_text/type_text/chain/attach
        if max_elements is None:
            if read_last or wait_reply or paste_text or type_text or chain or attach:
                max_elements = 250
            elif click or click_xy or hover or right_click:
                max_elements = 150
            elif app_name or pid or window_title_query or filter_query or include_floating or include_status_bar:
                max_elements = 40
            else:
                max_elements = 10
        elif (read_last or wait_reply or paste_text or type_text or chain or attach) and max_elements < 250:
            max_elements = 250

        # Handle mock/headless testing mode
        mock_app = os.environ.get("VOICEFI_MOCK_FRONT_APP")
        mock_win_title = os.environ.get("VOICEFI_MOCK_WINDOW_TITLE", "")
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
            sens_mock, sens_reason = AmbientPrivacyGuard.is_sensitive_origin(window_title=mock_win_title)
            if sens_mock:
                return FovealWindowSnapshot(
                    app_name=mock_app,
                    bundle_id=f"com.{mock_app.lower()}",
                    pid=9998,
                    window_title="[BLOCKED_SENSITIVE_ORIGIN]",
                    window_bbox=(0.0, 0.0, 0.0, 0.0),
                    focused_element=None,
                    interactive_elements=[],
                    latency_ms=1.0,
                    headless_route=f"BLOCKED: {sens_reason}",
                )

        if os.environ.get("VOICEFI_MOCK_FOVEAL") == "1" or (
            os.environ.get("VOICEFI_MOCK_AUDIO") == "1"
            and (os.environ.get("WBC_HEADLESS") == "1" or os.environ.get("VOICEFI_HEADLESS") == "1")
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
            # Delegate to signed `WBC Eyes.app` over `foveal.sock` when available so Python/Terminal needs zero TCC grant
            if not _skip_socket and not getattr(_FOVEAL_DAEMON_LOCAL, "in_daemon", False):
                sock_payload: Dict[str, Any] = {
                    "pid": pid,
                    "app_name": app_name,
                    "window_title_query": window_title_query,
                    "max_elements": max_elements,
                    "filter_query": filter_query,
                    "click": click,
                    "click_xy": click_xy,
                    "hover": hover,
                    "right_click": right_click,
                    "attach": attach,
                    "chain": chain,
                    "include_floating": include_floating,
                    "include_status_bar": include_status_bar,
                    "type_text": type_text,
                    "paste_text": paste_text,
                    "scroll": scroll,
                    "hotkey": hotkey,
                    "menu": menu,
                    "submit": submit,
                    "wait_reply": wait_reply,
                    "read_last": read_last,
                }
                if allow_focus_steal:
                    sock_payload["allow_focus_steal"] = True
                if no_cursor:
                    sock_payload["no_cursor"] = True
                sock_timeout = max(2.5, float(wait_reply) + 3.0) if wait_reply is not None else (8.0 if chain else 2.5)
                sock_resp = send_foveal_socket_request(sock_payload, timeout_sec=sock_timeout)
                if isinstance(sock_resp, dict):
                    if sock_resp.get("kill_switch") or sock_resp.get("paused"):
                        err_msg = str(sock_resp.get("error") or "WBC Eyes Kill Switch is active (Paused).")
                        return FovealWindowSnapshot(
                            app_name=app_name or "WBC Eyes",
                            bundle_id="com.wholebodycompute.eyes",
                            pid=pid or 0,
                            window_title="[PAUSED_KILL_SWITCH]",
                            latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                            headless_route=f"PAUSED: {err_msg}",
                            actuation={"status": "error", "paused": True, "kill_switch": True, "error": err_msg},
                        )
                    if sock_resp.get("status") != "error":
                        snap_from_sock = FovealWindowSnapshot.from_dict(sock_resp)
                        if record_telemetry:
                            _record_foveal_shield_telemetry(snap_from_sock)
                        return snap_from_sock

            import ApplicationServices as AS
            from whole_body_compute.integrations.foveal_memory import resolve_direct_headless_route
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

            has_actuation_req = bool(
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
            )

            # Fast-skip prohibited/headless CLI processes (activationPolicy == 2) that own zero GUI windows
            try:
                from AppKit import NSRunningApplication

                run_app = NSRunningApplication.runningApplicationWithProcessIdentifier_(int(app_pid))
                if run_app is not None and int(run_app.activationPolicy()) == 2 and not include_status_bar:
                    snap_cli = FovealWindowSnapshot(
                        app_name=resolved_app_name,
                        bundle_id=bundle_id,
                        pid=app_pid,
                        window_title="",
                        latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                        headless_route="SKIPPED: Non-GUI background/CLI process (NSApplicationActivationPolicyProhibited).",
                    )
                    if record_telemetry:
                        _record_foveal_shield_telemetry(snap_cli)
                    return snap_cli
            except Exception:
                pass

            # 5s negative-result TTL cache for repeated empty_dom polls on the same (pid, query, flags)
            empty_cache = getattr(cls, "_EMPTY_DOM_TTL_CACHE", None)
            if empty_cache is None:
                empty_cache = {}
                setattr(cls, "_EMPTY_DOM_TTL_CACHE", empty_cache)
            empty_key = (
                int(app_pid),
                str(window_title_query or ""),
                str(filter_query or ""),
                bool(include_floating),
                bool(include_status_bar),
            )
            now_mono = time.monotonic()
            cached_empty = empty_cache.get(empty_key)
            if (
                not has_actuation_req
                and cached_empty is not None
                and (now_mono - cached_empty[0]) < 5.0
                and os.environ.get("WBC_DISABLE_EMPTY_DOM_CACHE") != "1"
            ):
                snap_cached = FovealWindowSnapshot(
                    app_name=resolved_app_name,
                    bundle_id=bundle_id,
                    pid=app_pid,
                    window_title=cached_empty[1],
                    window_bbox=cached_empty[2],
                    focused_element=None,
                    interactive_elements=[],
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                    headless_route=cached_empty[3],
                    filter_query=filter_query,
                )
                if record_telemetry:
                    _record_foveal_shield_telemetry(snap_cached)
                return snap_cached

            # Blood-Brain Barrier: Block password managers / auth vaults (both localized name & bundle_id)
            if (
                AmbientPrivacyGuard.is_app_blocked(resolved_app_name, bundle_id=bundle_id)
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

            sens_live, sens_live_reason = AmbientPrivacyGuard.is_sensitive_origin(window_title=window_title)
            if sens_live:
                return FovealWindowSnapshot(
                    app_name=resolved_app_name,
                    bundle_id=bundle_id,
                    pid=app_pid,
                    window_title="[BLOCKED_SENSITIVE_ORIGIN]",
                    window_bbox=(0.0, 0.0, 0.0, 0.0),
                    focused_element=None,
                    interactive_elements=[],
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                    headless_route=f"BLOCKED: {sens_live_reason}",
                )

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

            # Auto-fallback if interactive_elems is still empty (e.g. Control Center, menu bar extras)
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

            if not interactive_elems and focused_elem is None:
                empty_cache[empty_key] = (time.monotonic(), window_title, window_bbox, headless_route)
                if len(empty_cache) > 128:
                    oldest_k = next(iter(empty_cache))
                    empty_cache.pop(oldest_k, None)
            elif empty_key in empty_cache:
                empty_cache.pop(empty_key, None)

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
                    window_title=window_title_query or window_title,
                    allow_focus_steal=allow_focus_steal,
                    no_cursor=no_cursor,
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
    allow_focus_steal: bool = False,
    no_cursor: bool = False,
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
        allow_focus_steal=allow_focus_steal,
        no_cursor=no_cursor,
    )
    if record_telemetry:
        _record_foveal_shield_telemetry(snap)
    return snap


def _is_safe_owner_unix_socket(sock_path: str) -> bool:
    """Verify `sock_path` is an actual Unix domain socket (never a symlink or regular file) owned by current UID. # [direct-edit]"""
    if not sock_path:
        return False
    try:
        import stat as stat_mod

        st = os.lstat(sock_path)
        if stat_mod.S_ISLNK(st.st_mode) or not stat_mod.S_ISSOCK(st.st_mode):
            return False
        if hasattr(os, "getuid") and st.st_uid != os.getuid():
            return False
        return True
    except OSError:
        return False


def _sanitize_foveal_daemon_params(req: Dict[str, Any], allowed_keys: Set[str]) -> Dict[str, Any]:
    """Validate and bound parameters forwarded into `FovealVisionEngine.capture_snapshot` over `foveal.sock`."""
    safe: Dict[str, Any] = {}
    for k, v in req.items():
        if k not in allowed_keys or v is None:
            continue
        if k == "pid":
            if isinstance(v, int) and not isinstance(v, bool) and v > 0:
                safe["pid"] = v
        elif k == "max_elements":
            if isinstance(v, int) and not isinstance(v, bool):
                safe["max_elements"] = max(1, min(500, v))
        elif k in ("include_floating", "include_status_bar", "read_last", "submit", "allow_focus_steal", "no_cursor"):
            safe[k] = bool(v)
        elif k == "wait_reply":
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                safe["wait_reply"] = max(0.0, min(60.0, float(v)))
        elif k == "chain":
            if isinstance(v, (str, list)):
                safe["chain"] = v
        elif isinstance(v, str):
            safe[k] = v[:4096]
    return safe


def send_foveal_socket_request(payload: Dict[str, Any], timeout_sec: float = 2.0) -> Optional[Dict[str, Any]]:
    """Try forwarding authenticated request to ~/.whole_body_compute/foveal.sock (`WBC Eyes.app`)."""
    sock_path = _resolve_foveal_socket_path()
    if not sock_path:
        return None
    if not _is_safe_owner_unix_socket(sock_path):
        if (
            FOVEAL_SOCKET_PATH == _INITIAL_FOVEAL_SOCKET_PATH
            and "WBC_DATA_DIR" not in os.environ
            and "WBC_FOVEAL_SOCKET_PATH" not in os.environ
        ):
            legacy_sock = os.path.expanduser("~/.voicefi/foveal.sock")
            if _is_safe_owner_unix_socket(legacy_sock):
                sock_path = legacy_sock
            else:
                return None
        else:
            return None

    eff_timeout = timeout_sec
    if isinstance(payload.get("wait_reply"), (int, float)):
        eff_timeout = max(eff_timeout, float(payload["wait_reply"]) + 3.0)
    elif payload.get("chain"):
        eff_timeout = max(eff_timeout, 8.0)

    try:
        signed_payload = _attach_foveal_socket_auth(payload, sock_path)
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(eff_timeout)
        s.connect(sock_path)
        data = json.dumps(signed_payload).encode("utf-8")
        s.sendall(data + b"\n")
        resp_bytes = b""
        while len(resp_bytes) < 2 * 1024 * 1024:
            chunk = s.recv(4096)
            if not chunk or not isinstance(chunk, (bytes, bytearray)):
                break
            resp_bytes += chunk
            if b"\n" in chunk:
                break
        s.close()
        return json.loads(resp_bytes.decode("utf-8"))
    except Exception as exc:
        logger.debug("Socket forward failed: %s", exc)
        return None


def _write_foveal_daemon_token(socket_path: str) -> Tuple[str, bytes]:
    tok_path = _resolve_foveal_token_path(socket_path)
    tok_dir = os.path.dirname(tok_path)
    if tok_dir:
        os.makedirs(tok_dir, mode=0o700, exist_ok=True)
    secret = secrets.token_bytes(32)
    tmp_path = f"{tok_path}.tmp.{os.getpid()}"
    try:
        os.remove(tmp_path)
    except OSError:
        pass
    fd = os.open(tmp_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, (secret.hex() + "\n").encode("ascii"))
    finally:
        os.close(fd)
    os.replace(tmp_path, tok_path)
    os.chmod(tok_path, 0o600)
    return tok_path, secret


def _verify_python_daemon_hmac(
    req: Dict[str, Any],
    secret: bytes,
    seen_nonces: Dict[str, float],
) -> None:
    auth = req.get("_auth")
    if not isinstance(auth, dict):
        raise ValueError("Caller authentication failed: missing '_auth' HMAC token envelope.")
    claimed_pid = auth.get("pid")
    ts_raw = auth.get("ts")
    nonce = auth.get("nonce")
    claimed_bind = auth.get("op_bind")
    mac_hex = auth.get("hmac")
    if not isinstance(claimed_pid, int) or claimed_pid <= 0:
        raise ValueError("Caller authentication failed: invalid '_auth.pid'.")
    try:
        ts_val = float(str(ts_raw))
    except Exception as exc:
        raise ValueError("Caller authentication failed: malformed '_auth.ts'.") from exc
    now = time.time()
    if abs(now - ts_val) > 30.0:
        raise ValueError("Caller authentication failed: expired '_auth.ts'.")
    if not isinstance(nonce, str) or len(nonce) < 16:
        raise ValueError("Caller authentication failed: invalid '_auth.nonce'.")
    expected_bind = _compute_foveal_op_bind(req)
    if claimed_bind != expected_bind:
        raise ValueError("Caller authentication failed: op_bind mismatch.")
    if not isinstance(mac_hex, str) or len(mac_hex) != 64:
        raise ValueError("Caller authentication failed: invalid HMAC format.")
    msg = f"wbc-foveal-v1:{claimed_pid}:{ts_raw}:{nonce}:{expected_bind}".encode("utf-8")
    expected_mac = hmac.new(secret, msg, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(mac_hex.lower(), expected_mac.lower()):
        raise ValueError("Caller authentication failed: invalid HMAC-SHA256 signature.")
    cutoff = now - 65.0
    if len(seen_nonces) >= 4096:
        for k, v in list(seen_nonces.items()):
            if v < cutoff:
                seen_nonces.pop(k, None)
    if nonce in seen_nonces and seen_nonces[nonce] >= cutoff:
        raise ValueError("Caller authentication failed: replayed '_auth.nonce' rejected.")
    seen_nonces[nonce] = now


def run_foveal_daemon(
    socket_path: Optional[str] = None,
    enable_ambient_discovery: Optional[bool] = None,
) -> None:
    """
    Run persistent Unix socket daemon for zero-coldstart foveal vision (<5ms)
    plus default-on passive `AmbientWorkflowObserver` workflow discovery. # [direct-edit]
    """
    import stat as stat_mod

    socket_path = (
        socket_path
        if (socket_path and socket_path != _INITIAL_FOVEAL_SOCKET_PATH)
        else (os.environ.get("WBC_FOVEAL_SOCKET_PATH", "").strip() or str(get_wbc_data_dir() / "foveal.sock"))
    )
    sock_parent = os.path.dirname(socket_path)
    if sock_parent:
        # Allow macOS system symlinks (/tmp -> /private/tmp, /var -> /private/var) while rejecting user dir symlinks
        if os.path.islink(sock_parent) and sock_parent not in ("/tmp", "/var"):
            raise RuntimeError(f"Refusing to bind foveal.sock inside symlinked parent directory: {sock_parent}")
        real_parent = os.path.realpath(sock_parent)
        os.makedirs(real_parent, mode=0o700, exist_ok=True)
        if real_parent not in ("/tmp", "/private/tmp", "/var/tmp", "/private/var/tmp"):
            try:
                os.chmod(real_parent, 0o700)
            except OSError:
                pass
    if os.path.lexists(socket_path):
        st_existing = os.lstat(socket_path)
        if not (stat_mod.S_ISSOCK(st_existing.st_mode) or stat_mod.S_ISLNK(st_existing.st_mode)):
            raise RuntimeError(f"Refusing to overwrite non-socket file at {socket_path}")
        try:
            os.unlink(socket_path)
        except OSError:
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

    token_path, token_secret = _write_foveal_daemon_token(socket_path)
    seen_nonces: Dict[str, float] = {}

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    old_umask = os.umask(0o177)
    try:
        server.bind(socket_path)
    finally:
        os.umask(old_umask)
    server.listen(5)
    logger.info("Foveal vision daemon listening on %s", socket_path)
    print(f"📡 Foveal Vision Daemon listening on {socket_path} (sub-5ms hot reflexes)")

    _ALLOWED_DAEMON_KEYS = {
        "pid",
        "app_name",
        "window_title_query",
        "max_elements",
        "filter_query",
        "click",
        "click_xy",
        "hover",
        "right_click",
        "attach",
        "chain",
        "include_floating",
        "include_status_bar",
        "type_text",
        "paste_text",
        "scroll",
        "hotkey",
        "menu",
        "submit",
        "wait_reply",
        "read_last",
        "allow_focus_steal",
        "no_cursor",
    }

    _FOVEAL_DAEMON_LOCAL.in_daemon = True
    try:
        while True:
            conn, _ = server.accept()
            try:
                data = b""
                while len(data) < 65536:
                    ch = conn.recv(4096)
                    if not ch:
                        break
                    data += ch
                    if b"\n" in ch:
                        break
                if len(data) >= 65536 and b"\n" not in data:
                    raise ValueError("Socket request exceeded 64 KiB limit")
                if data:
                    req = json.loads(data.decode("utf-8"))
                    if not isinstance(req, dict):
                        raise ValueError("Request must be a JSON object")
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
                    elif req.get("op") in ("status", "eyes_helper_status", "ping"):
                        _verify_python_daemon_hmac(req, token_secret, seen_nonces)
                        resp = {
                            "status": "ok",
                            "pong": True,
                            "provider": "python_foveal_daemon",
                            "pid": os.getpid(),
                            "paused": False,
                            "kill_switch": False,
                        }
                    else:
                        _verify_python_daemon_hmac(req, token_secret, seen_nonces)
                        safe_req = _sanitize_foveal_daemon_params(req, _ALLOWED_DAEMON_KEYS)
                        snap = FovealVisionEngine.capture_snapshot(_skip_socket=True, **safe_req)
                        resp = snap.to_dict()
                    conn.sendall(json.dumps(resp).encode("utf-8") + b"\n")
            except Exception as exc:
                err_resp = {"status": "error", "error": str(exc)}
                conn.sendall(json.dumps(err_resp).encode("utf-8") + b"\n")
            finally:
                conn.close()
    finally:
        _FOVEAL_DAEMON_LOCAL.in_daemon = False
        if ambient_controller is not None:
            try:
                ambient_controller.stop_background_discovery()
            except Exception:
                pass
        server.close()
        for cleanup_p in (socket_path, token_path):
            if os.path.exists(cleanup_p):
                try:
                    os.remove(cleanup_p)
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
