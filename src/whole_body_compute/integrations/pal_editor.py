"""
Pal Live Edit Mode ("Select & Speak" Local Website/Code Editor) for VoiceFi.
Captures live text selections on webpages or apps via macOS Accessibility (AXUIElement)
and browser context, then patches local workspace source files in real time using:
  1. Sub-2ms deterministic fast-path rules (e.g., "change this to a colon", "delete this")
  2. On-device local models (Qwen 2.5 Coder / Gemma via Ollama or LiteRT) for rewrites
Zero cloud tokens, instant browser tab reload, and full multi-step Undo stack.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from AppKit import NSPasteboard, NSPasteboardTypeString, NSWorkspace
    from ApplicationServices import (
        AXUIElementCopyAttributeValue,
        AXUIElementCopyElementAtPosition,
        AXUIElementCopyParameterizedAttributeValue,
        AXUIElementCreateApplication,
        AXUIElementCreateSystemWide,
        AXUIElementPerformAction,
        AXUIElementSetAttributeValue,
    )
    import Quartz
except ImportError:
    NSPasteboard = None  # type: ignore[assignment]
    NSPasteboardTypeString = "public.utf8-plain-text"  # type: ignore[assignment]
    NSWorkspace = None  # type: ignore[assignment]
    AXUIElementCopyAttributeValue = None  # type: ignore[assignment]
    AXUIElementCopyElementAtPosition = None  # type: ignore[assignment]
    AXUIElementCopyParameterizedAttributeValue = None  # type: ignore[assignment]
    AXUIElementCreateApplication = None  # type: ignore[assignment]
    AXUIElementCreateSystemWide = None  # type: ignore[assignment]
    AXUIElementPerformAction = None  # type: ignore[assignment]
    AXUIElementSetAttributeValue = None  # type: ignore[assignment]
    Quartz = None  # type: ignore[assignment]

from whole_body_compute.integrations.pal_editor_engine import (
    LocalEditEngine,
    LocalSiteFileResolver,
)
from whole_body_compute.integrations.pal_editor_rules import (
    ACTIVE_EDITING_JOURNAL,
    ATTACHED_TRAILING_PUNCT,
    DASH_VARIANTS,
    DEFAULT_WORKSPACE_ROOTS,
    PAL_LOG_DIR,
    PAL_LOG_FILE,
    SPOKEN_PUNCTUATION_MAP,
    WORD_TO_NUM,
    PalEditRecord,
    SelectionSnapshot,
    _normalize_whitespace_and_entities,
    apply_color_change_to_line,
    apply_smart_replacement_in_line,
    extract_relational_line_actions,
    logger,
    pal_log,
    parse_fast_path_edit_instruction,
    parse_spoken_number,
    resolve_spoken_replacement,
)


class PalEditSession:
    """
    Live "Select & Speak" Edit Mode session orchestrator.
    Passively observes mouse selections across browsers/apps and pairs them with
    spoken edit instructions to instantaneously patch local files on disk.
    """

    BROWSER_APPS = ("Google Chrome", "Safari", "Brave Browser", "Arc")

    def __init__(
        self,
        session_name: str = "Live Website Edit",
        workspace_roots: Optional[List[Path]] = None,
        on_step: Optional[Callable[[PalEditRecord], None]] = None,
    ):
        self.session_name = (session_name or "Live Website Edit").strip()
        self.resolver = LocalSiteFileResolver(workspace_roots=workspace_roots)
        self.on_step = on_step
        self.records: List[PalEditRecord] = []
        self.active_step_number: int = 1
        self.last_selection: Optional[SelectionSnapshot] = None
        self.start_time: float = 0.0

        self._running = False
        self._tap = None
        self._loop = None
        self._loop_source = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._mouse_down_pos: Optional[Tuple[float, float]] = None
        self._ignore_rects: List[Tuple[float, float, float, float]] = []

    def set_ignore_rect(self, x: float, y: float, width: float, height: float) -> None:
        with self._lock:
            self._ignore_rects = [(float(x), float(y), float(width), float(height))]

    def _is_point_ignored(self, x: float, y: float) -> bool:
        for rx, ry, rw, rh in self._ignore_rects:
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                return True
        return False

    def start(self) -> None:
        if self._running:
            return
        self.start_time = time.time()
        self._running = True
        self._ensure_active_item()
        self._write_journal(active=True)
        self._thread = threading.Thread(
            target=self._run_tap, daemon=True, name="PalEditSessionTap"
        )
        self._thread.start()

    def _ensure_active_item(self) -> PalEditRecord:
        """Ensure there is an active recording item (#N) ready for selection and voice notes."""
        with self._lock:
            existing = next(
                (
                    r
                    for r in self.records
                    if r.step_number == self.active_step_number and not r.applied
                ),
                None,
            )
            if existing is not None:
                return existing

            init_sel = ""
            init_surr = ""
            init_app = ""
            init_title = ""
            init_url = ""
            if not self.records:
                try:
                    snap = self.capture_live_selection(allow_clipboard_fallback=False)
                    if snap and snap.selected_text.strip():
                        self.last_selection = snap
                        init_sel = snap.selected_text
                        init_surr = snap.surrounding_text
                        init_app = snap.app_name
                        init_title = snap.window_title
                        init_url = snap.page_url
                except Exception:
                    pass

            short_sel = init_sel.strip()
            if len(short_sel) > 34:
                short_sel = short_sel[:32] + "…"
            summary = f'Selected: "{short_sel}"' if short_sel else "Select text on page & speak..."

            step_num = len(self.records) + 1
            self.active_step_number = step_num
            rec = PalEditRecord(
                step_number=step_num,
                action="active_item",
                selected_text=init_sel,
                surrounding_text=init_surr,
                instruction="",
                file_path="",
                old_snippet=init_sel,
                new_snippet="",
                spoken_summary=summary,
                app_name=init_app,
                window_title=init_title,
                page_url=init_url,
                applied=False,
            )
            self.records.append(rec)
            self._write_journal(active=self._running)

        if self.on_step is not None:
            try:
                self.on_step(rec)
            except Exception:
                pass
        return rec

    def stop(self) -> List[PalEditRecord]:
        self._running = False
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
        self._clear_journal()
        return list(self.records)

    # ------------------------------------------------------------------
    # Selection Inspection (AXUIElement + Browser Context)
    # ------------------------------------------------------------------

    @classmethod
    def capture_live_selection(
        cls,
        x: Optional[float] = None,
        y: Optional[float] = None,
        allow_clipboard_fallback: bool = False,
    ) -> Optional[SelectionSnapshot]:
        """
        Query the currently highlighted text, surrounding sentence/element context,
        and active browser tab URL/title from macOS Accessibility and AppleScript.
        """
        app_name = ""
        window_title = ""
        pid = 0

        try:
            ws = NSWorkspace.sharedWorkspace()
            front = ws.frontmostApplication()
            if front:
                app_name = front.localizedName() or ""
                pid = front.processIdentifier()
        except Exception:
            pass

        selected_text = ""
        surrounding_text = ""
        page_url = ""

        candidate_pids: List[Tuple[int, str, str]] = []
        if pid:
            candidate_pids.append((pid, app_name, window_title))
        try:
            wins = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly
                | Quartz.kCGWindowListExcludeDesktopElements,
                Quartz.kCGNullWindowID,
            )
            for w in wins or []:
                if w.get("kCGWindowLayer", 0) == 0:
                    owner = str(w.get("kCGWindowOwnerName") or "").strip()
                    wpid = int(w.get("kCGWindowOwnerPID") or 0)
                    wtitle = str(w.get("kCGWindowName") or "").strip()
                    if (
                        owner in cls.BROWSER_APPS
                        and wpid
                        and all(wpid != p[0] for p in candidate_pids)
                    ):
                        candidate_pids.append((wpid, owner, wtitle))
                        break
        except Exception:
            pass

        for c_pid, c_app, c_wtitle in candidate_pids:
            try:
                app_ax = AXUIElementCreateApplication(c_pid)
                try:
                    AXUIElementSetAttributeValue(app_ax, "AXManualAccessibility", True)
                except Exception:
                    pass

                err, win = AXUIElementCopyAttributeValue(app_ax, "AXFocusedWindow", None)
                if err == 0 and win and not window_title:
                    err_t, title = AXUIElementCopyAttributeValue(win, "AXTitle", None)
                    if err_t == 0 and title:
                        window_title = str(title)
                    elif c_wtitle:
                        window_title = c_wtitle

                err_f, focused = AXUIElementCopyAttributeValue(app_ax, "AXFocusedUIElement", None)
                if err_f == 0 and focused:
                    sel_str, surr_str, url_str, title_str = cls._extract_ax_element_selection(
                        focused
                    )
                    if sel_str and not selected_text:
                        selected_text = sel_str
                        app_name = c_app
                    if surr_str and len(surr_str) > len(surrounding_text):
                        surrounding_text = surr_str
                    if url_str and not page_url:
                        page_url = url_str
                    if title_str and not window_title:
                        window_title = title_str

                if win and (not selected_text or not page_url):
                    web_sel, web_surr, web_url, web_title = cls._find_web_area_selection(win)
                    if web_sel and not selected_text:
                        selected_text = web_sel
                        app_name = c_app
                    if web_surr and len(web_surr) > len(surrounding_text):
                        surrounding_text = web_surr
                    if web_url and not page_url:
                        page_url = web_url
                    if web_title and not window_title:
                        window_title = web_title

                if selected_text:
                    break
            except Exception:
                pass

        if x is not None and y is not None:
            try:
                sys_elem = AXUIElementCreateSystemWide()
                err, elem = AXUIElementCopyElementAtPosition(
                    sys_elem, float(x), float(y), None
                )
                if err == 0 and elem:
                    if not selected_text:
                        _, sel2 = AXUIElementCopyAttributeValue(elem, "AXSelectedText", None)
                        if sel2:
                            selected_text = str(sel2)
                    for attr in ("AXValue", "AXTitle", "AXDescription"):
                        _, val = AXUIElementCopyAttributeValue(elem, attr, None)
                        if val and len(str(val).strip()) > len(surrounding_text):
                            surrounding_text = str(val).strip()[:500]
            except Exception:
                pass

        if not page_url:
            b_url, browser_title, js_sel, js_surr = cls._query_browser_context(app_name)
            if b_url:
                page_url = b_url
            if browser_title and not window_title:
                window_title = browser_title
            if js_sel and not selected_text:
                selected_text = js_sel
            if js_surr and len(js_surr) > len(surrounding_text):
                surrounding_text = js_surr

        if not selected_text and allow_clipboard_fallback:
            clip_sel = cls._read_selection_via_clipboard_preserve()
            if clip_sel:
                selected_text = clip_sel

        if not selected_text and not surrounding_text:
            return None

        return SelectionSnapshot(
            selected_text=selected_text,
            surrounding_text=surrounding_text,
            app_name=app_name,
            window_title=window_title,
            page_url=page_url,
            x=float(x or 0.0),
            y=float(y or 0.0),
        )

    @classmethod
    def _extract_ax_element_selection(cls, elem: Any) -> Tuple[str, str, str, str]:
        """Extract (selected_text, surrounding_text, page_url, title) from an AXUIElement or AXWebArea."""
        sel_text = ""
        surr_text = ""
        url_str = ""
        title_str = ""
        try:
            _, url_val = AXUIElementCopyAttributeValue(elem, "AXURL", None)
            if url_val is not None:
                url_str = str(getattr(url_val, "absoluteString", lambda: url_val)())
            _, t_val = AXUIElementCopyAttributeValue(elem, "AXTitle", None)
            if t_val:
                title_str = str(t_val)
            _, sel = AXUIElementCopyAttributeValue(elem, "AXSelectedText", None)
            if sel:
                sel_text = str(sel)
            _, val = AXUIElementCopyAttributeValue(elem, "AXValue", None)
            if val:
                surr_text = str(val)[:500]

            _, marker_rng = AXUIElementCopyAttributeValue(
                elem, "AXSelectedTextMarkerRange", None
            )
            if marker_rng is not None:
                err_m, m_str = AXUIElementCopyParameterizedAttributeValue(
                    elem, "AXStringForTextMarkerRange", marker_rng, None
                )
                if err_m == 0 and m_str and not sel_text:
                    sel_text = str(m_str)
                _, start_marker = AXUIElementCopyAttributeValue(
                    elem, "AXStartTextMarker", None
                )
                if not surr_text and sel_text:
                    err_u, leaf_elem = AXUIElementCopyParameterizedAttributeValue(
                        elem, "AXUIElementForTextMarker", start_marker, None
                    )
                    if err_u == 0 and leaf_elem is not None:
                        _, leaf_val = AXUIElementCopyAttributeValue(
                            leaf_elem, "AXValue", None
                        )
                        if leaf_val:
                            surr_text = str(leaf_val)[:500]
        except Exception:
            pass
        return sel_text, surr_text, url_str, title_str

    @classmethod
    def _find_web_area_selection(
        cls, root_elem: Any, depth: int = 0
    ) -> Tuple[str, str, str, str]:
        if depth > 10 or root_elem is None:
            return "", "", "", ""
        try:
            _, role = AXUIElementCopyAttributeValue(root_elem, "AXRole", None)
            if role == "AXWebArea":
                return cls._extract_ax_element_selection(root_elem)
            _, children = AXUIElementCopyAttributeValue(root_elem, "AXChildren", None)
            for child in children or []:
                res = cls._find_web_area_selection(child, depth + 1)
                if res[0] or res[2]:
                    return res
        except Exception:
            pass
        return "", "", "", ""

    @classmethod
    def _query_browser_context(cls, app_name: str) -> Tuple[str, str, str, str]:
        """Return (page_url, tab_title, js_selected_text, js_surrounding_text) from front browser."""
        if app_name not in cls.BROWSER_APPS:
            return "", "", "", ""

        url = ""
        title = ""
        js_sel = ""
        js_surr = ""

        try:
            if app_name == "Safari":
                script = (
                    'tell application "Safari"\n'
                    "  if (count of windows) > 0 then\n"
                    "    set u to URL of current tab of front window\n"
                    "    set t to name of current tab of front window\n"
                    '    return u & "|||" & t\n'
                    "  end if\n"
                    "end tell"
                )
            else:
                script = (
                    f'tell application "{app_name}"\n'
                    "  if (count of windows) > 0 then\n"
                    "    set u to URL of active tab of front window\n"
                    "    set t to title of active tab of front window\n"
                    '    return u & "|||" & t\n'
                    "  end if\n"
                    "end tell"
                )
            res = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=0.45,
                check=False,
            )
            if res.returncode == 0 and "|||" in (res.stdout or ""):
                parts = res.stdout.strip().split("|||", 1)
                url = parts[0].strip()
                title = parts[1].strip()
        except Exception:
            pass

        return url, title, js_sel, js_surr

    @classmethod
    def _read_selection_via_clipboard_preserve(cls) -> str:
        """
        Temporarily copy current selection via AXMenuBar (Edit -> Copy) or Cmd+C
        and immediately restore previous clipboard.
        """
        try:
            pb = NSPasteboard.generalPasteboard()
            old_clip = pb.stringForType_(NSPasteboardTypeString)

            target_pids: List[int] = []
            try:
                ws = NSWorkspace.sharedWorkspace()
                front = ws.frontmostApplication()
                if front and front.processIdentifier():
                    target_pids.append(int(front.processIdentifier()))
            except Exception:
                pass
            try:
                wins = Quartz.CGWindowListCopyWindowInfo(
                    Quartz.kCGWindowListOptionOnScreenOnly
                    | Quartz.kCGWindowListExcludeDesktopElements,
                    Quartz.kCGNullWindowID,
                )
                for w in wins or []:
                    if w.get("kCGWindowLayer", 0) == 0:
                        owner = str(w.get("kCGWindowOwnerName") or "").strip()
                        wpid = int(w.get("kCGWindowOwnerPID") or 0)
                        if owner in cls.BROWSER_APPS and wpid and wpid not in target_pids:
                            target_pids.append(wpid)
                            break
            except Exception:
                pass

            for tpid in target_pids:
                try:
                    app_ax = AXUIElementCreateApplication(tpid)
                    _, mbar = AXUIElementCopyAttributeValue(app_ax, "AXMenuBar", None)
                    _, items = AXUIElementCopyAttributeValue(mbar, "AXChildren", None)
                    for item in items or []:
                        _, title = AXUIElementCopyAttributeValue(item, "AXTitle", None)
                        if title == "Edit":
                            _, submenus = AXUIElementCopyAttributeValue(
                                item, "AXChildren", None
                            )
                            if submenus:
                                _, edit_items = AXUIElementCopyAttributeValue(
                                    submenus[0], "AXChildren", None
                                )
                                for ei in edit_items or []:
                                    _, etitle = AXUIElementCopyAttributeValue(
                                        ei, "AXTitle", None
                                    )
                                    _, enabled = AXUIElementCopyAttributeValue(
                                        ei, "AXEnabled", None
                                    )
                                    if etitle == "Copy" and enabled:
                                        pb.clearContents()
                                        AXUIElementPerformAction(ei, "AXPress")
                                        time.sleep(0.035)
                                        new_clip = pb.stringForType_(NSPasteboardTypeString)
                                        selected = str(new_clip) if new_clip else ""
                                        if old_clip is not None:
                                            pb.clearContents()
                                            pb.setString_forType_(
                                                str(old_clip), NSPasteboardTypeString
                                            )
                                        if selected:
                                            return selected
                except Exception:
                    pass

            return ""
        except Exception:
            return ""

    @classmethod
    def reload_active_browser_tab(cls, app_name: str = "") -> bool:
        """Reload the active browser tab in-place via AXMenuBar, CGEventPostToPid, or AppleScript."""
        target_app = app_name if app_name in cls.BROWSER_APPS else ""
        target_pid = 0

        try:
            wins = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly
                | Quartz.kCGWindowListExcludeDesktopElements,
                Quartz.kCGNullWindowID,
            )
            for w in wins or []:
                if w.get("kCGWindowLayer", 0) == 0:
                    owner = str(w.get("kCGWindowOwnerName") or "").strip()
                    name = str(w.get("kCGWindowName") or "").strip()
                    if owner in cls.BROWSER_APPS and (not target_app or owner == target_app):
                        if any(
                            k in name.lower()
                            for k in ("localhost", "127.0.0.1", "enterprise_corpus", "vifi")
                        ):
                            target_app = owner
                            target_pid = int(w.get("kCGWindowOwnerPID") or 0)
                            break
            if not target_pid:
                for w in wins or []:
                    if w.get("kCGWindowLayer", 0) == 0:
                        owner = str(w.get("kCGWindowOwnerName") or "").strip()
                        if owner in cls.BROWSER_APPS and (
                            not target_app or owner == target_app
                        ):
                            target_app = owner
                            target_pid = int(w.get("kCGWindowOwnerPID") or 0)
                            break
        except Exception:
            pass

        if not target_app:
            target_app = "Google Chrome"

        if target_pid:
            try:
                app_ax = AXUIElementCreateApplication(target_pid)
                _, mbar = AXUIElementCopyAttributeValue(app_ax, "AXMenuBar", None)
                _, items = AXUIElementCopyAttributeValue(mbar, "AXChildren", None)
                for item in items or []:
                    _, title = AXUIElementCopyAttributeValue(item, "AXTitle", None)
                    if title == "View":
                        _, submenus = AXUIElementCopyAttributeValue(item, "AXChildren", None)
                        if submenus:
                            _, view_items = AXUIElementCopyAttributeValue(
                                submenus[0], "AXChildren", None
                            )
                            for vi in view_items or []:
                                _, vtitle = AXUIElementCopyAttributeValue(vi, "AXTitle", None)
                                if str(vtitle or "") in ("Reload This Page", "Reload Page"):
                                    if AXUIElementPerformAction(vi, "AXPress") == 0:
                                        return True
            except Exception:
                pass

            try:
                ev_down = Quartz.CGEventCreateKeyboardEvent(None, 15, True)
                Quartz.CGEventSetFlags(ev_down, Quartz.kCGEventFlagMaskCommand)
                ev_up = Quartz.CGEventCreateKeyboardEvent(None, 15, False)
                Quartz.CGEventSetFlags(ev_up, Quartz.kCGEventFlagMaskCommand)
                Quartz.CGEventPostToPid(target_pid, ev_down)
                Quartz.CGEventPostToPid(target_pid, ev_up)
                return True
            except Exception:
                pass

        try:
            if target_app == "Safari":
                script = (
                    'tell application "Safari"\n'
                    "  if (count of windows) > 0 then\n"
                    "     set URL of current tab of front window to (URL of current tab of front window)\n"
                    "  end if\n"
                    "end tell"
                )
            else:
                script = (
                    f'tell application "{target_app}"\n'
                    "  if (count of windows) > 0 then\n"
                    "    reload active tab of front window\n"
                    "  end if\n"
                    "end tell"
                )
            res = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=0.6,
                check=False,
            )
            return res.returncode == 0
        except Exception:
            return False

    @classmethod
    def _write_and_sync_mirrors(cls, target_path: Path, new_content: str) -> List[Path]:
        """Write updated content to target_path and synchronize any corresponding build/dist/source mirror files."""
        written: List[Path] = [target_path]
        target_path.write_text(new_content, encoding="utf-8")

        try:
            resolved_target = target_path.resolve()
            parent_root: Optional[Path] = None
            for root in DEFAULT_WORKSPACE_ROOTS:
                try:
                    resolved_target.relative_to(root.resolve())
                    parent_root = root.resolve()
                    break
                except ValueError:
                    pass

            if parent_root is None:
                for p in resolved_target.parents:
                    if (p / ".git").is_dir() or (p / "package.json").is_file():
                        parent_root = p
                        break

            if parent_root is not None:
                rel = resolved_target.relative_to(parent_root)
                rel_parts = rel.parts
                if rel_parts:
                    top_dir = rel_parts[0]
                    sub_path = Path(*rel_parts[1:]) if len(rel_parts) > 1 else None
                    if sub_path:
                        if top_dir in ("site", "src", "public"):
                            mirror_dirs = ["dist_cloudflare", "dist", "build"]
                            for md in mirror_dirs:
                                cand = parent_root / md / sub_path
                                if cand.is_file() and cand.resolve() != resolved_target:
                                    cand.write_text(new_content, encoding="utf-8")
                                    written.append(cand)
                                    pal_log(f"🔄 Synced build mirror: {cand}")
                        elif top_dir in ("dist_cloudflare", "dist", "build"):
                            mirror_dirs = ["site", "src", "public"]
                            for md in mirror_dirs:
                                cand = parent_root / md / sub_path
                                if cand.is_file() and cand.resolve() != resolved_target:
                                    cand.write_text(new_content, encoding="utf-8")
                                    written.append(cand)
                                    pal_log(f"🔄 Synced source file from mirror: {cand}")
        except Exception as e:
            pal_log(f"Mirror sync notice: {e}", level="warning")

        return written

    # ------------------------------------------------------------------
    # Live Edit Execution & Undo
    # ------------------------------------------------------------------

    def apply_voice_edit(
        self,
        instruction: str,
        selection_override: Optional[SelectionSnapshot] = None,
        auto_reload_browser: bool = True,
    ) -> PalEditRecord:
        """Execute a spoken edit instruction against the active selection and patch the local file."""
        t0 = time.perf_counter()
        clean_inst = (instruction or "").strip()
        pal_log(f"🎙️ apply_voice_edit called with instruction='{clean_inst}'")

        snap = selection_override
        if snap is None:
            live_snap = self.capture_live_selection(allow_clipboard_fallback=True)
            if live_snap and (live_snap.selected_text or live_snap.surrounding_text):
                snap = live_snap
                self.last_selection = live_snap
            elif self.last_selection is not None:
                snap = self.last_selection

        parsed = parse_fast_path_edit_instruction(
            clean_inst, snap.selected_text if snap else ""
        )
        explicit_subtarget = parsed[1] if parsed else None

        sel_text = (snap.selected_text if snap else "") or (explicit_subtarget or "")
        surr_text = snap.surrounding_text if snap else ""
        page_url = snap.page_url if snap else ""
        win_title = snap.window_title if snap else ""
        app_name = snap.app_name if snap else ""

        if not sel_text and not surr_text:
            pal_log("❌ apply_voice_edit: No text selected on screen.", level="warning")
            rec = PalEditRecord(
                step_number=len(self.records) + 1,
                action="edit_failed",
                selected_text="",
                surrounding_text="",
                instruction=clean_inst,
                file_path="",
                old_snippet="",
                new_snippet="",
                spoken_summary="No text selected on screen. Highlight the text you want to change first.",
                app_name=app_name,
                window_title=win_title,
                page_url=page_url,
                applied=False,
            )
            self._record_and_notify(rec)
            return rec

        located = self.resolver.find_best_file_and_line(
            selected_text=sel_text,
            surrounding_text=surr_text,
            page_url=page_url,
            window_title=win_title,
            explicit_subtarget=explicit_subtarget,
        )
        if located is None:
            pal_log(
                f"❌ apply_voice_edit: No local file matched '{sel_text[:35]}'", level="warning"
            )
            rec = PalEditRecord(
                step_number=len(self.records) + 1,
                action="edit_failed",
                selected_text=sel_text,
                surrounding_text=surr_text,
                instruction=clean_inst,
                file_path="",
                old_snippet=sel_text,
                new_snippet="",
                spoken_summary=f"Could not find a local file matching '{sel_text[:28]}'.",
                app_name=app_name,
                window_title=win_title,
                page_url=page_url,
                applied=False,
            )
            self._record_and_notify(rec)
            return rec

        target_path, line_idx = located
        old_content = target_path.read_text(encoding="utf-8")
        lines = old_content.splitlines()
        old_line = lines[line_idx]

        new_chunk, start_idx, end_idx, engine_used = LocalEditEngine.apply_edit(
            lines=lines,
            line_idx=line_idx,
            selected_text=sel_text,
            instruction=clean_inst,
        )

        old_slice = lines[start_idx:end_idx] if 0 <= start_idx < len(lines) else [old_line]

        if new_chunk is None or (new_chunk == old_slice and engine_used == "none"):
            pal_log(
                f"❌ apply_voice_edit: Could not apply '{clean_inst}' to lines in {target_path.name}",
                level="warning",
            )
            rec = PalEditRecord(
                step_number=len(self.records) + 1,
                action="edit_failed",
                selected_text=sel_text,
                surrounding_text=surr_text,
                instruction=clean_inst,
                file_path=str(target_path),
                old_snippet="\n".join(old_slice).strip(),
                new_snippet="\n".join(old_slice).strip(),
                spoken_summary=f"Could not apply '{clean_inst[:32]}' to {target_path.name}.",
                app_name=app_name,
                window_title=win_title,
                page_url=page_url,
                applied=False,
            )
            self._record_and_notify(rec)
            return rec

        lines[start_idx:end_idx] = new_chunk
        new_content = "\n".join(lines) + ("\n" if old_content.endswith("\n") else "")
        self._write_and_sync_mirrors(target_path, new_content)

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        pal_log(
            f"✅ apply_voice_edit: Successfully applied in {elapsed_ms:.1f}ms to {target_path.name}:{start_idx+1}-{end_idx}"
        )

        if self.last_selection is not None and new_chunk:
            self.last_selection.surrounding_text = _normalize_whitespace_and_entities(
                " ".join(new_chunk)
            )

        if auto_reload_browser:
            self.reload_active_browser_tab(app_name)

        short_sel = sel_text.strip()
        if len(short_sel) > 22:
            short_sel = short_sel[:20] + "…"
        spoken_summary = (
            f"Updated '{short_sel}' in {target_path.name} ({elapsed_ms:.0f}ms local)"
        )

        rec = PalEditRecord(
            step_number=len(self.records) + 1,
            action="edit_applied",
            selected_text=sel_text,
            surrounding_text=surr_text,
            instruction=clean_inst,
            file_path=str(target_path),
            old_snippet="\n".join(old_slice).strip(),
            new_snippet="\n".join(new_chunk).strip(),
            spoken_summary=spoken_summary,
            app_name=app_name,
            window_title=win_title,
            page_url=page_url,
            engine_used=engine_used,
            elapsed_ms=elapsed_ms,
            applied=True,
            _old_file_content=old_content,
            _new_file_content=new_content,
        )
        try:
            from whole_body_compute.local.dot_memory import get_dot_store

            get_dot_store().record_lgtm_edit_diff(
                action_type="pal_voice_edit",
                target=target_path.name,
                original_draft=rec.old_snippet,
                final_content=rec.new_snippet,
            )
        except Exception:
            pass
        self._record_and_notify(rec)
        return rec

    def undo_last_edit(self, auto_reload_browser: bool = True) -> Optional[PalEditRecord]:
        """Revert the most recent applied edit on disk and reload the browser tab."""
        for rec in reversed(self.records):
            if (
                rec.applied
                and not rec.undone
                and rec.file_path
                and rec._old_file_content is not None
            ):
                fpath = Path(rec.file_path)
                if fpath.is_file():
                    self._write_and_sync_mirrors(fpath, rec._old_file_content)
                    rec.undone = True
                    rec.spoken_summary = f"Undid edit in {fpath.name}"
                    if auto_reload_browser:
                        self.reload_active_browser_tab(rec.app_name)
                    self._write_journal(active=self._running)
                    return rec
        return None

    def remove_step(
        self, step_number: int, auto_reload_browser: bool = True
    ) -> List[PalEditRecord]:
        """Called when the user clicks '×' on a step in the Granola panel."""
        with self._lock:
            target = next((r for r in self.records if r.step_number == step_number), None)
            if (
                target is not None
                and target.applied
                and not target.undone
                and target.file_path
                and target._old_file_content is not None
            ):
                fpath = Path(target.file_path)
                if fpath.is_file():
                    fpath.write_text(target._old_file_content, encoding="utf-8")
                    target.undone = True
                    if auto_reload_browser:
                        self.reload_active_browser_tab(target.app_name)

            self.records = [r for r in self.records if r.step_number != step_number]
            for idx, r in enumerate(self.records, start=1):
                r.step_number = idx
            self._write_journal(active=self._running)
            return list(self.records)

    def _record_and_notify(self, rec: PalEditRecord) -> None:
        with self._lock:
            if (
                rec.action in ("edit_applied", "edit_failed")
                and self.records
                and self.records[-1].action in ("select", "active_item")
                and not self.records[-1].applied
            ):
                self.records.pop()
            rec.step_number = len(self.records) + 1
            self.records.append(rec)
            self.active_step_number = rec.step_number
            self._write_journal(active=self._running)
        if self.on_step is not None:
            try:
                self.on_step(rec)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Multi-Change Item Management (Active Item + Voice Notes + Next/Enter)
    # ------------------------------------------------------------------

    def select_active_step(self, step_number: int) -> Optional[PalEditRecord]:
        """Set a specific item row as the currently active recording item."""
        target: Optional[PalEditRecord] = None
        with self._lock:
            for r in self.records:
                if r.step_number == step_number and not r.applied:
                    self.active_step_number = step_number
                    r.action = "active_item"
                    target = r
                elif r.action == "active_item" and not r.applied:
                    r.action = "select"
            self._write_journal(active=self._running)
        if target is not None and self.on_step is not None:
            try:
                for r in list(self.records):
                    self.on_step(r)
            except Exception:
                pass
        return target

    def record_voice_note(
        self, text: str, selection_override: Optional[SelectionSnapshot] = None
    ) -> Optional[PalEditRecord]:
        """Attach spoken voice instructions to the currently active change item (#N)."""
        clean = (text or "").strip()
        if not clean:
            return None

        auto_advance = False
        trailing_next_re = re.compile(
            r"(?:[\s,.;:!?\-]+|^)(?:next\s+change|new\s+change|next\s+item|new\s+item|apply\s+that|apply\s+change)[\s.!?,]*$",
            re.IGNORECASE,
        )
        m_next = trailing_next_re.search(clean)
        if m_next:
            auto_advance = True
            clean = clean[: m_next.start()].strip(" \t\n\r.,!?:;-")

        active_item = self._ensure_active_item()

        if selection_override is not None:
            self.last_selection = selection_override
            with self._lock:
                active_item.selected_text = selection_override.selected_text
                active_item.surrounding_text = selection_override.surrounding_text
                active_item.app_name = selection_override.app_name
                active_item.window_title = selection_override.window_title
                active_item.page_url = selection_override.page_url
                short_sel = selection_override.selected_text.strip()
                if len(short_sel) > 34:
                    short_sel = short_sel[:32] + "…"
                active_item.spoken_summary = f'Selected: "{short_sel}"'
        elif not active_item.selected_text.strip():
            snap = self.capture_live_selection(allow_clipboard_fallback=True)
            if snap and snap.selected_text.strip():
                self.last_selection = snap
                with self._lock:
                    active_item.selected_text = snap.selected_text
                    active_item.surrounding_text = snap.surrounding_text
                    active_item.app_name = snap.app_name
                    active_item.window_title = snap.window_title
                    active_item.page_url = snap.page_url
                    short_sel = snap.selected_text.strip()
                    if len(short_sel) > 34:
                        short_sel = short_sel[:32] + "…"
                    active_item.spoken_summary = f'Selected: "{short_sel}"'

        if clean:
            with self._lock:
                if active_item.instruction:
                    active_item.instruction = f"{active_item.instruction}. {clean}"
                else:
                    active_item.instruction = clean
                active_item.action = "active_item"
                if not active_item.selected_text.strip():
                    active_item.spoken_summary = "Waiting for text selection on page..."
                self._write_journal(active=self._running)

            if self.on_step is not None:
                try:
                    self.on_step(active_item)
                except Exception:
                    pass

        if auto_advance:
            return self.commit_and_advance_item(auto_reload_browser=True, async_apply=True)

        return active_item

    def commit_and_advance_item(
        self,
        auto_reload_browser: bool = True,
        async_apply: bool = True,
        open_next: bool = True,
    ) -> Optional[PalEditRecord]:
        """Lock in the current active item, hand it off to the local edit engine, and open the next item."""
        target_item: Optional[PalEditRecord] = None
        with self._lock:
            target_item = next(
                (
                    r
                    for r in self.records
                    if r.step_number == self.active_step_number and not r.applied
                ),
                None,
            )
            if target_item is None and self.records and not self.records[-1].applied:
                target_item = self.records[-1]

        if target_item is None:
            if open_next:
                return self._ensure_active_item()
            return None

        if not target_item.selected_text.strip():
            snap = self.capture_live_selection(allow_clipboard_fallback=True)
            if snap and snap.selected_text.strip():
                self.last_selection = snap
                with self._lock:
                    target_item.selected_text = snap.selected_text
                    target_item.surrounding_text = snap.surrounding_text
                    target_item.app_name = snap.app_name
                    target_item.window_title = snap.window_title
                    target_item.page_url = snap.page_url

        has_work = bool(target_item.selected_text.strip() or target_item.instruction.strip())
        if not has_work:
            return target_item

        short_sel = target_item.selected_text.strip() or "selection"
        if len(short_sel) > 26:
            short_sel = short_sel[:24] + "…"

        with self._lock:
            if target_item.instruction.strip():
                target_item.action = "applying"
                target_item.spoken_summary = f'Applying to "{short_sel}"...'
            else:
                target_item.action = "select"

            new_item: Optional[PalEditRecord] = None
            if open_next:
                next_num = len(self.records) + 1
                self.active_step_number = next_num
                new_item = PalEditRecord(
                    step_number=next_num,
                    action="active_item",
                    selected_text="",
                    surrounding_text="",
                    instruction="",
                    file_path="",
                    old_snippet="",
                    new_snippet="",
                    spoken_summary="Select text on page & speak...",
                    applied=False,
                )
                self.records.append(new_item)
            self._write_journal(active=self._running)

        if self.on_step is not None:
            try:
                self.on_step(target_item)
                if new_item is not None:
                    self.on_step(new_item)
            except Exception:
                pass

        if target_item.instruction.strip():
            if async_apply:
                threading.Thread(
                    target=self._apply_record_edit,
                    args=(target_item, auto_reload_browser),
                    daemon=True,
                    name=f"PalEditApply-{target_item.step_number}",
                ).start()
            else:
                self._apply_record_edit(target_item, auto_reload_browser)

        return target_item

    def _apply_record_edit(
        self, rec: PalEditRecord, auto_reload_browser: bool = True
    ) -> PalEditRecord:
        """Execute the local file edit for a specific `PalEditRecord` and switch its state to `edit_applied` (✅)."""
        t0 = time.perf_counter()
        clean_inst = (rec.instruction or "").strip()
        parsed = parse_fast_path_edit_instruction(clean_inst, rec.selected_text)
        explicit_subtarget = parsed[1] if parsed else None

        sel_text = (rec.selected_text or "").strip() or (explicit_subtarget or "")
        surr_text = rec.surrounding_text or ""
        page_url = rec.page_url or ""
        win_title = rec.window_title or ""
        app_name = rec.app_name or ""

        if not sel_text and not surr_text and self.last_selection is not None:
            sel_text = self.last_selection.selected_text
            surr_text = self.last_selection.surrounding_text
            page_url = page_url or self.last_selection.page_url
            win_title = win_title or self.last_selection.window_title
            app_name = app_name or self.last_selection.app_name

        pal_log(
            f"🛠️ [Step #{rec.step_number}] Starting apply: instruction='{clean_inst}', selected='{sel_text[:35]}', app='{app_name}', win='{win_title[:30]}'"
        )

        if not sel_text and not surr_text:
            pal_log(
                f"❌ [Step #{rec.step_number}] Failed: No text selected on screen.",
                level="warning",
            )
            with self._lock:
                rec.action = "edit_failed"
                rec.spoken_summary = "No text selected — highlight text first"
                rec.applied = False
                self._write_journal(active=self._running)
            if self.on_step is not None:
                try:
                    self.on_step(rec)
                except Exception:
                    pass
            return rec

        located = self.resolver.find_best_file_and_line(
            selected_text=sel_text,
            surrounding_text=surr_text,
            page_url=page_url,
            window_title=win_title,
            explicit_subtarget=explicit_subtarget,
        )
        if located is None:
            pal_log(
                f"❌ [Step #{rec.step_number}] Failed: No local file matched '{sel_text[:35]}'.",
                level="warning",
            )
            with self._lock:
                rec.action = "edit_failed"
                rec.spoken_summary = f"No local file matched '{sel_text[:24]}'"
                rec.applied = False
                self._write_journal(active=self._running)
            if self.on_step is not None:
                try:
                    self.on_step(rec)
                except Exception:
                    pass
            return rec

        target_path, line_idx = located
        old_content = target_path.read_text(encoding="utf-8")
        lines = old_content.splitlines()
        old_line = lines[line_idx]
        pal_log(f"📍 [Step #{rec.step_number}] Located match in {target_path}:{line_idx+1}")
        pal_log(f"  Old line: {old_line.strip()[:100]}")

        new_chunk, start_idx, end_idx, engine_used = LocalEditEngine.apply_edit(
            lines=lines,
            line_idx=line_idx,
            selected_text=sel_text,
            instruction=clean_inst,
        )

        old_slice = lines[start_idx:end_idx] if 0 <= start_idx < len(lines) else [old_line]

        if new_chunk is None or (new_chunk == old_slice and engine_used == "none"):
            pal_log(
                f"❌ [Step #{rec.step_number}] Failed: Could not apply '{clean_inst}' to lines in {target_path.name}",
                level="warning",
            )
            with self._lock:
                rec.action = "edit_failed"
                rec.file_path = str(target_path)
                rec.old_snippet = "\n".join(old_slice).strip()
                rec.new_snippet = "\n".join(old_slice).strip()
                rec.spoken_summary = f"Could not apply to {target_path.name}"
                rec.applied = False
                self._write_journal(active=self._running)
            if self.on_step is not None:
                try:
                    self.on_step(rec)
                except Exception:
                    pass
            return rec

        lines[start_idx:end_idx] = new_chunk
        new_content = "\n".join(lines) + ("\n" if old_content.endswith("\n") else "")
        self._write_and_sync_mirrors(target_path, new_content)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        pal_log(
            f"✅ [Step #{rec.step_number}] Successfully applied via {engine_used} in {elapsed_ms:.1f}ms!"
        )

        if auto_reload_browser:
            reloaded = self.reload_active_browser_tab(app_name)
            pal_log(
                f"🔄 Browser reload for app '{app_name}': {'success' if reloaded else 'fallback'}"
            )

        short_sel = sel_text.strip()
        if len(short_sel) > 24:
            short_sel = short_sel[:22] + "…"

        with self._lock:
            rec.action = "edit_applied"
            rec.file_path = str(target_path)
            rec.old_snippet = "\n".join(old_slice).strip()
            rec.new_snippet = "\n".join(new_chunk).strip()
            rec.engine_used = engine_used
            rec.elapsed_ms = elapsed_ms
            rec.applied = True
            rec._old_file_content = old_content
            rec._new_file_content = new_content
            rec.spoken_summary = f'"{short_sel}" → {target_path.name} ({elapsed_ms:.0f}ms)'
            self._write_journal(active=self._running)

        try:
            from whole_body_compute.audio.chimes import play_chime

            play_chime("done", block=False)
        except Exception:
            pass

        if self.on_step is not None:
            try:
                self.on_step(rec)
            except Exception:
                pass
        return rec

    # ------------------------------------------------------------------
    # Quartz Mouse Selection & Enter Key Event Tap
    # ------------------------------------------------------------------

    def _event_callback(self, proxy, event_type, event, refcon):
        try:
            if event_type == Quartz.kCGEventKeyDown:
                keycode = Quartz.CGEventGetIntegerValueField(
                    event, Quartz.kCGKeyboardEventKeycode
                )
                flags = Quartz.CGEventGetFlags(event)
                has_cmd_ctrl_opt = bool(
                    flags
                    & (
                        Quartz.kCGEventFlagMaskCommand
                        | Quartz.kCGEventFlagMaskControl
                        | Quartz.kCGEventFlagMaskAlternate
                    )
                )
                if keycode in (36, 76) and not has_cmd_ctrl_opt:
                    threading.Thread(
                        target=self.commit_and_advance_item,
                        kwargs={"auto_reload_browser": True, "async_apply": True},
                        daemon=True,
                    ).start()
                return event

            if event_type == Quartz.kCGEventLeftMouseDown:
                loc = Quartz.CGEventGetLocation(event)
                if self._is_point_ignored(loc.x, loc.y):
                    self._mouse_down_pos = None
                    return event
                self._mouse_down_pos = (loc.x, loc.y)

            elif event_type == Quartz.kCGEventLeftMouseUp:
                loc = Quartz.CGEventGetLocation(event)
                if self._is_point_ignored(loc.x, loc.y):
                    self._mouse_down_pos = None
                    return event
                click_state = Quartz.CGEventGetIntegerValueField(
                    event, Quartz.kCGMouseEventClickState
                )
                is_drag = False
                if self._mouse_down_pos is not None:
                    dist = math.hypot(
                        loc.x - self._mouse_down_pos[0], loc.y - self._mouse_down_pos[1]
                    )
                    is_drag = dist >= 4.0
                self._mouse_down_pos = None

                if is_drag or click_state >= 2:
                    threading.Timer(
                        0.06, lambda: self._on_mouse_selection_finished(loc.x, loc.y)
                    ).start()
        except Exception:
            pass
        return event

    def _on_mouse_selection_finished(self, x: float, y: float) -> None:
        snap = self.capture_live_selection(x=x, y=y, allow_clipboard_fallback=True)
        if snap and snap.selected_text.strip():
            self.last_selection = snap
            short_sel = snap.selected_text.strip()
            if len(short_sel) > 34:
                short_sel = short_sel[:32] + "…"
            active_item = self._ensure_active_item()
            with self._lock:
                active_item.selected_text = snap.selected_text
                active_item.surrounding_text = snap.surrounding_text
                active_item.old_snippet = snap.selected_text
                active_item.spoken_summary = f'Selected: "{short_sel}"'
                active_item.app_name = snap.app_name
                active_item.window_title = snap.window_title
                active_item.page_url = snap.page_url
                active_item.action = "active_item"
                self._write_journal(active=self._running)
            if self.on_step is not None:
                try:
                    self.on_step(active_item)
                except Exception:
                    pass

    def _run_tap(self) -> None:
        mask = (
            Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseDown)
            | Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseUp)
            | Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown)
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
            return
        self._loop_source = Quartz.CFMachPortCreateRunLoopSource(None, self._tap, 0)
        self._loop = Quartz.CFRunLoopGetCurrent()
        Quartz.CFRunLoopAddSource(self._loop, self._loop_source, Quartz.kCFRunLoopDefaultMode)
        Quartz.CGEventTapEnable(self._tap, True)
        while self._running:
            res = Quartz.CFRunLoopRunInMode(Quartz.kCFRunLoopDefaultMode, 0.5, False)
            if res not in (
                Quartz.kCFRunLoopRunTimedOut,
                Quartz.kCFRunLoopRunHandledSource,
            ):
                break

    # ------------------------------------------------------------------
    # Cross-Process Journal Persistence
    # ------------------------------------------------------------------

    def _write_journal(self, active: bool = True) -> None:
        try:
            ACTIVE_EDITING_JOURNAL.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "active": active,
                "pid": os.getpid(),
                "session_name": self.session_name,
                "start_time": self.start_time,
                "updated_at": time.time(),
                "records": [r.to_public_dict() for r in self.records],
            }
            ACTIVE_EDITING_JOURNAL.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    @classmethod
    def _clear_journal(cls) -> None:
        try:
            if ACTIVE_EDITING_JOURNAL.exists():
                ACTIVE_EDITING_JOURNAL.unlink()
        except Exception:
            pass

    @classmethod
    def load_journal(cls) -> Optional[Dict[str, Any]]:
        try:
            if not ACTIVE_EDITING_JOURNAL.exists():
                return None
            data = json.loads(ACTIVE_EDITING_JOURNAL.read_text(encoding="utf-8"))
            if not data.get("active"):
                return None
            if time.time() - float(data.get("updated_at", 0)) > 3600:
                cls._clear_journal()
                return None
            return data
        except Exception:
            return None
