"""
Organ 03 Sensory Extension: The Ears (`whole_body_compute.organs.ears`).

Community Free Forever ($0) AST-grounded local dictation & push-to-talk runtime:
  - Default Hotkey: Hold `Fn` (push-to-talk) | `Fn + Space` (hands-free lock toggle),
    matching Wispr Flow muscle memory with `--hotkey opt+space` override.
  - Native macOS Quartz `CGEventTap` global listener (listen-only, zero key-stealing),
    with automatic fallback to Enter-triggered CLI dictation if Accessibility / Input
    Monitoring permission is not granted yet.
  - Zero-Focus-Steal Injection: Pastes AST-grounded text directly into the focused
    IDE/terminal field while preserving the user's prior clipboard.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from whole_body_compute.stt.local_ears import (
    LocalEarsEngine,
    PTTSession,
    TranscriptionResult,
    check_macos_fn_key_conflict,
)

logger = logging.getLogger("whole_body_compute.organs.ears")

# macOS virtual keycodes & modifier masks
_KEYCODE_SPACE = 49
_FLAG_FN = 1 << 23          # kCGEventFlagMaskSecondaryFn (0x800000)
_FLAG_OPTION = 1 << 19      # kCGEventFlagMaskAlternate   (0x080000)


class HotkeyStateMachine:
    """
    Pure, platform-independent state machine for Push-to-Talk + Hands-Free Lock.

    Modes:
      - `hotkey="fn"` (default):
          * Press & hold `Fn` -> `on_start("ptt")`
          * While holding `Fn`, tap `Space` -> upgrades current recording to hands-free lock
          * Release `Fn` (in `ptt` mode) -> `on_stop()`
          * Tap `Fn` (or `Fn+Space`) while in `hands_free` mode -> `on_stop()`
      - `hotkey="opt+space"`:
          * Press `Option + Space` -> toggles recording start / stop
    """

    def __init__(
        self,
        *,
        hotkey: str = "fn",
        on_start: Callable[[str], None],
        on_stop: Callable[[], None],
    ):
        self.hotkey = (hotkey or "fn").strip().lower()
        self.on_start = on_start
        self.on_stop = on_stop
        self.fn_down: bool = False
        self.recording: bool = False
        self.hands_free: bool = False
        self._lock = threading.Lock()

    def handle_flags_changed(self, flags: int) -> Optional[str]:
        """Process a macOS `kCGEventFlagsChanged` modifier bitmask. Returns action taken."""
        if self.hotkey != "fn":
            return None
        is_fn = bool(flags & _FLAG_FN)
        with self._lock:
            if is_fn and not self.fn_down:
                self.fn_down = True
                if self.recording and self.hands_free:
                    # Second Fn tap stops hands-free recording
                    self.recording = False
                    self.hands_free = False
                    self.on_stop()
                    return "stop_hands_free"
                if not self.recording:
                    self.recording = True
                    self.hands_free = False
                    self.on_start("ptt")
                    return "start_ptt"
            elif not is_fn and self.fn_down:
                self.fn_down = False
                if self.recording and not self.hands_free:
                    self.recording = False
                    self.on_stop()
                    return "stop_ptt"
        return None

    def handle_key_down(self, keycode: int, flags: int) -> Optional[str]:
        """Process a macOS `kCGEventKeyDown` event. Returns action taken."""
        with self._lock:
            if self.hotkey == "fn" and keycode == _KEYCODE_SPACE and (self.fn_down or bool(flags & _FLAG_FN)):
                if self.recording and not self.hands_free:
                    self.hands_free = True
                    return "lock_hands_free"
                if self.recording and self.hands_free:
                    self.recording = False
                    self.hands_free = False
                    self.on_stop()
                    return "stop_hands_free"
                if not self.recording:
                    self.recording = True
                    self.hands_free = True
                    self.on_start("hands_free")
                    return "start_hands_free"
            elif self.hotkey in ("opt+space", "option+space", "alt+space"):
                if keycode == _KEYCODE_SPACE and bool(flags & _FLAG_OPTION):
                    if not self.recording:
                        self.recording = True
                        self.hands_free = True
                        self.on_start("opt+space")
                        return "start_opt_space"
                    self.recording = False
                    self.hands_free = False
                    self.on_stop()
                    return "stop_opt_space"
        return None


def is_voicefi_dictation_active() -> bool:
    """
    Detect if VoiceFi's own voice/dictation daemon is actively running on this Mac
    so `body ears` yields cleanly instead of double-capturing `Fn` / microphone.
    """
    import socket

    # 1. Check VoiceFi local daemon port (:5140)
    try:
        with socket.create_connection(("127.0.0.1", 5140), timeout=0.08):
            return True
    except OSError:
        pass
    # 2. Check VoiceFi dictation PID/lock file if present
    for lock_path in (
        Path.home() / ".voicefi" / "voicefi.pid",
        Path.home() / ".voicefi" / "dictation.lock",
    ):
        if lock_path.is_file():
            try:
                pid_txt = lock_path.read_text(encoding="utf-8").strip()
                if pid_txt.isdigit():
                    os.kill(int(pid_txt), 0)
                    return True
            except Exception:
                pass
    return False


def paste_into_focused_app(text: str) -> bool:
    """Inject transcribed text into the currently focused macOS app without stealing focus."""
    if not text:
        return False
    if platform.system() != "Darwin":
        return False
    try:
        prev = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=1.0).stdout
        subprocess.run(["pbcopy"], input=text, text=True, check=True, timeout=1.0)
        subprocess.run(
            [
                "osascript",
                "-e",
                'tell application "System Events" to keystroke "v" using command down',
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        time.sleep(0.08)
        subprocess.run(["pbcopy"], input=prev, text=True, timeout=1.0)
        return True
    except Exception as exc:
        logger.debug("paste_into_focused_app failed: %s", exc)
        return False


class OrganEars:
    """High-level interface for Community Free AST-grounded local dictation (`body ears`)."""

    def __init__(
        self,
        *,
        workspace: Optional[Path] = None,
        db_path: Optional[Path] = None,
        backend: str = "auto",
        hotkey: str = "fn",
    ):
        self.engine = LocalEarsEngine(workspace=workspace, db_path=db_path, backend=backend)
        self.hotkey = (hotkey or "fn").strip().lower()
        self._active_session: Optional[PTTSession] = None
        self._session_lock = threading.Lock()

    def status(self) -> Dict[str, Any]:
        info = self.engine.status()
        info["hotkey"] = self.hotkey
        info["hands_free_chord"] = "fn+space" if self.hotkey == "fn" else f"toggle {self.hotkey}"
        info["voicefi_active"] = is_voicefi_dictation_active()
        info["yielding_to_voicefi"] = bool(
            info["voicefi_active"] and os.environ.get("WBC_EARS_FORCE_OVER_VOICEFI") != "1"
        )
        return info

    def upgrade_model(self) -> Dict[str, Any]:
        return self.engine.upgrade_model()

    def transcribe(
        self,
        audio_path: Optional[str] = None,
        *,
        paste: bool = False,
        silence_sec: float = 1.4,
        max_sec: float = 30.0,
        active_file: Optional[str] = None,
    ) -> TranscriptionResult:
        if audio_path:
            res = self.engine.transcribe_file(audio_path, active_file=active_file)
        else:
            res = self.engine.transcribe_mic(
                silence_sec=silence_sec,
                max_sec=max_sec,
                active_file=active_file,
            )
        if paste and res.text:
            paste_into_focused_app(res.text)
        return res

    def _on_ptt_start(self, mode: str) -> None:
        if is_voicefi_dictation_active() and os.environ.get("WBC_EARS_FORCE_OVER_VOICEFI") != "1":
            logger.debug("Yielding PTT hotkey to active VoiceFi daemon (:5140).")
            return
        with self._session_lock:
            if self._active_session is not None:
                return
            session = self.engine.start_ptt_mic_session()
            if session is None:
                print("⚠️  Local ANE mic helper unavailable.", flush=True)
                return
            self._active_session = session
            sym_cnt = len(session.bundle.all_symbols)
            mode_label = "Hands-Free" if mode == "hands_free" else "Hold Fn"
            print(f"🔴 Listening ({mode_label} • {sym_cnt} AST symbols)...", flush=True)

    def _on_ptt_stop(self) -> None:
        with self._session_lock:
            session = self._active_session
            self._active_session = None
        if session is None:
            return

        def _finalize() -> None:
            res = session.finish()
            if res.error and not res.text:
                print(f"⚠️  {res.error}", flush=True)
            elif res.text:
                paste_into_focused_app(res.text)
                snaps = len(res.ast_snaps)
                badge = f" [{snaps} AST snap{'s' if snaps != 1 else ''}]" if snaps else ""
                print(f"✅ {res.text}{badge} ({res.latency_ms:.0f}ms)", flush=True)
            else:
                print("… (silence)", flush=True)

        threading.Thread(target=_finalize, daemon=True, name="EarsPTTFinalize").start()

    def _run_quartz_event_tap(self) -> bool:
        """
        Attach a listen-only macOS Quartz `CGEventTap` for `Fn` hold / `Fn+Space` / `opt+space`.
        Returns True if tap ran until KeyboardInterrupt, False if Quartz/Accessibility unavailable.
        """
        if platform.system() != "Darwin":
            return False
        try:
            import Quartz
        except Exception:
            return False

        sm = HotkeyStateMachine(
            hotkey=self.hotkey,
            on_start=self._on_ptt_start,
            on_stop=self._on_ptt_stop,
        )

        mask = (1 << Quartz.kCGEventFlagsChanged) | (1 << Quartz.kCGEventKeyDown)

        def _callback(proxy, event_type, event, refcon):
            try:
                if event_type == Quartz.kCGEventTapDisabledByTimeout:
                    Quartz.CGEventTapEnable(tap, True)
                    return event
                flags = Quartz.CGEventGetFlags(event)
                if event_type == Quartz.kCGEventFlagsChanged:
                    sm.handle_flags_changed(int(flags))
                elif event_type == Quartz.kCGEventKeyDown:
                    keycode = int(Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode))
                    action = sm.handle_key_down(keycode, int(flags))
                    if action == "lock_hands_free":
                        print("🔒 Hands-Free locked (tap Fn or Fn+Space to stop)...", flush=True)
            except Exception as exc:
                logger.debug("CGEventTap callback error: %s", exc)
            return event

        tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionListenOnly,
            mask,
            _callback,
            None,
        )
        if tap is None:
            return False

        run_loop_source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
        run_loop = Quartz.CFRunLoopGetCurrent()
        Quartz.CFRunLoopAddSource(run_loop, run_loop_source, Quartz.kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(tap, True)

        if self.hotkey == "fn":
            print("   Hold [Fn] to dictate • Tap [Fn+Space] for hands-free • Ctrl+C to stop", flush=True)
        else:
            print(f"   Press [{self.hotkey.upper()}] to toggle dictation • Ctrl+C to stop", flush=True)

        try:
            while True:
                Quartz.CFRunLoopRunInMode(Quartz.kCFRunLoopDefaultMode, 0.25, False)
        except KeyboardInterrupt:
            print("\n🛑 body ears stopped.")
        finally:
            try:
                Quartz.CGEventTapEnable(tap, False)
            except Exception:
                pass
        return True

    def run_push_to_talk_loop(self) -> int:
        """
        Interactive foreground push-to-talk listener.
        Uses native macOS Quartz CGEventTap for Hold `Fn` + `Fn+Space` hands-free (or `opt+space`);
        falls back gracefully to Enter-triggered CLI dictation if Accessibility tap is unavailable.
        """
        st = self.status()
        fn_check = st.get("fn_key_check") or {}
        print(
            f"🎙️  body ears active  •  backend={st['backend']}  •  "
            f"AST symbols={st['symbols_indexed']}  •  hotkey={self.hotkey.upper()}"
        )
        if self.hotkey == "fn" and fn_check.get("conflict") and fn_check.get("tip"):
            print(f"⚠️  macOS Fn key currently set to '{fn_check.get('label')}'.")
            print(f"   Tip: {fn_check['tip']}")

        if self._run_quartz_event_tap():
            return 0

        print(
            "ℹ️  Global hotkey tap unavailable (grant Input Monitoring / Accessibility to your terminal, "
            "or use Enter-triggered mode below)."
        )
        print("   Press [Enter] to dictate a turn (or Ctrl+C to stop)...")
        try:
            while True:
                sys.stdin.readline()
                print("🔴 Listening (RAM-only, zero disk audio)...", flush=True)
                res = self.transcribe(paste=True)
                if res.error:
                    print(f"⚠️  {res.error}", flush=True)
                elif res.text:
                    snaps = len(res.ast_snaps)
                    badge = f" [{snaps} AST snap{'s' if snaps != 1 else ''}]" if snaps else ""
                    print(f"✅ {res.text}{badge} ({res.latency_ms:.0f}ms)", flush=True)
                else:
                    print("… (silence)", flush=True)
        except KeyboardInterrupt:
            print("\n🛑 body ears stopped.")
            return 0
