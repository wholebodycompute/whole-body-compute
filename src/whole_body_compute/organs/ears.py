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
from typing import Any, Callable, Dict, Optional, Tuple

from whole_body_compute.stt.local_ears import (
    LocalEarsEngine,
    PTTSession,
    TranscriptionResult,
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


def _broker_ears_type_diff(backspaces: int, text: str) -> Optional[bool]:
    """Try brokering differential dictation typing through `Whole Body Compute.app` (`foveal.sock`)."""
    try:
        from whole_body_compute.integrations.foveal_vision import send_foveal_socket_request

        resp = send_foveal_socket_request(
            {"op": "ears_type_diff", "backspaces": int(backspaces), "text": str(text)},
            timeout_sec=1.5,
        )
        if isinstance(resp, dict):
            if resp.get("status") == "ok":
                return True
            if resp.get("security_blocked") or resp.get("kill_switch") or resp.get("ears_paused"):
                return False
    except Exception:
        pass
    return None


def _is_headless_env() -> bool:
    return bool(
        os.environ.get("WBC_HEADLESS") == "1"
        or os.environ.get("WBC_TESTING") == "1"
        or os.environ.get("VOICEFI_HEADLESS") == "1"
        or os.environ.get("VOICEFI_TESTING") == "1"
        or "PYTEST_CURRENT_TEST" in os.environ
    )


def broker_ears_highlight(
    action: str = "show",
    *,
    mode: str = "ptt",
    target: str = "auto",
    max_duration_s: float = 95.0,
) -> Optional[Dict[str, Any]]:
    """
    Show, update, or hide the click-through dictation highlight border around the
    focused text box (with automatic focused-window fallback) via `Whole Body Compute.app`
    (`foveal.sock`), falling back to `PalHighlightOverlay` when an in-process AppKit
    runloop is active.
    """
    if os.environ.get("WBC_EARS_NO_HIGHLIGHT") == "1":
        return None
    if platform.system() != "Darwin" and not os.environ.get("WBC_FOVEAL_SOCKET_PATH"):
        return None

    # 1. Primary path: Native Swift AppKit overlay inside `Whole Body Compute.app` (`foveal.sock`)
    try:
        from whole_body_compute.integrations.foveal_vision import send_foveal_socket_request

        resp = send_foveal_socket_request(
            {
                "op": "ears_highlight",
                "action": str(action),
                "mode": str(mode),
                "target": str(target),
                "max_duration_s": float(max_duration_s),
            },
            timeout_sec=0.35,
        )
        if isinstance(resp, dict):
            return resp
    except Exception:
        pass

    # 2. Fallback path: In-process PyObjC overlay (only when not headless and NSApp is running)
    if _is_headless_env():
        return None
    try:
        import AppKit

        app = AppKit.NSApplication.sharedApplication()
        if app is not None and app.isRunning():
            from whole_body_compute.ui.screen_highlight import PalHighlightOverlay

            overlay = PalHighlightOverlay.get_instance()
            if action == "hide":
                overlay.hide()
                return {"status": "ok", "visible": False, "provider": "pyobjc"}
            kind = overlay.show_for_ears_target(
                mode=mode,
                target=target,
                max_duration_s=max_duration_s,
            )
            return {"status": "ok", "visible": True, "target_kind": kind, "provider": "pyobjc"}
    except Exception:
        pass
    return None


def _send_backspaces(count: int) -> None:
    """Send `count` Delete/Backspace keystrokes with flags=0 (ignores held Fn key)."""
    if count <= 0 or platform.system() != "Darwin":
        return
    brokered = _broker_ears_type_diff(count, "")
    if brokered is not None:
        return
    try:
        import Quartz

        src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStatePrivate) or Quartz.CGEventSourceCreate(
            Quartz.kCGEventSourceStateHIDSystemState
        )
        for _ in range(count):
            down = Quartz.CGEventCreateKeyboardEvent(src, 51, True)   # kVK_Delete = 51
            up = Quartz.CGEventCreateKeyboardEvent(src, 51, False)
            Quartz.CGEventSetFlags(down, 0)
            Quartz.CGEventSetFlags(up, 0)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, down)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, up)
    except Exception as exc:
        logger.debug("_send_backspaces failed: %s", exc)


def type_unicode_direct(text: str) -> bool:
    """
    Type Unicode text directly into the focused macOS app via `CGEventKeyboardSetUnicodeString`
    with `flags = 0`, so it works cleanly even while `Fn` is physically held down (zero
    clipboard touching, zero `Fn+Cmd+V` collision).
    """
    if not text or platform.system() != "Darwin":
        return False
    brokered = _broker_ears_type_diff(0, text)
    if brokered is not None:
        return brokered
    try:
        import Quartz

        src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStatePrivate) or Quartz.CGEventSourceCreate(
            Quartz.kCGEventSourceStateHIDSystemState
        )
        # Chunk at 16 UTF-16 code units per event (macOS CGEventKeyboardSetUnicodeString limit is ~20)
        chunk_size = 16
        for idx in range(0, len(text), chunk_size):
            chunk = text[idx : idx + chunk_size]
            down = Quartz.CGEventCreateKeyboardEvent(src, 0, True)
            up = Quartz.CGEventCreateKeyboardEvent(src, 0, False)
            Quartz.CGEventSetFlags(down, 0)
            Quartz.CGEventSetFlags(up, 0)
            Quartz.CGEventKeyboardSetUnicodeString(down, len(chunk), chunk)
            Quartz.CGEventKeyboardSetUnicodeString(up, len(chunk), chunk)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, down)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, up)
        return True
    except Exception as exc:
        logger.debug("type_unicode_direct failed: %s", exc)
        return False


class LiveDifferentialTyper:
    """
    Types streaming partial transcripts into the focused app in real time,
    computing the minimal backspace + suffix diff whenever Apple Speech revises
    earlier words or Stage-2 ASTSymbolSnapper snaps an identifier on finalization.

    Includes built-in utterance-reset protection so if an on-device STT engine
    clears or resets its partial buffer after a brief pause while `Fn` is still held,
    previously spoken words are locked in rather than backspaced out of the box.
    """

    def __init__(
        self,
        *,
        type_fn: Optional[Callable[[str], bool]] = None,
        backspace_fn: Optional[Callable[[int], None]] = None,
    ):
        self._emitted: str = ""
        self._committed_segments: list[str] = []
        self._active_segment: str = ""
        self._last_update_ts: float = 0.0
        self._custom_io = type_fn is not None or backspace_fn is not None
        self._type_fn = type_fn or type_unicode_direct
        self._backspace_fn = backspace_fn or _send_backspaces
        self._lock = threading.Lock()

    @property
    def committed_prefix(self) -> str:
        return " ".join(self._committed_segments).strip()

    @staticmethod
    def _common_prefix_ci(a: str, b: str) -> int:
        al = a.lower()
        bl = b.lower()
        lim = min(len(al), len(bl))
        idx = 0
        while idx < lim and al[idx] == bl[idx]:
            idx += 1
        return idx

    def _merge_segments_locked(self, tail: str) -> str:
        clean = tail.strip()
        if not clean:
            parts = list(self._committed_segments)
            if self._active_segment:
                parts.append(self._active_segment)
            return " ".join(parts).strip()
        if not self._committed_segments:
            return clean
        first = self._committed_segments[0]
        if clean.lower().startswith(first[: min(8, len(first))].lower()):
            return clean
        last = self._committed_segments[-1]
        if self._common_prefix_ci(last, clean) >= min(4, len(clean), len(last)):
            return " ".join(self._committed_segments[:-1] + [clean]).strip()
        return " ".join(self._committed_segments + [clean]).strip()

    def merge_final_raw(self, raw_acoustic: str) -> str:
        """Merge a finalized acoustic string with any earlier segments committed across ANE resets."""
        with self._lock:
            return self._merge_segments_locked(raw_acoustic)

    def _apply_diff_locked(self, target_text: str) -> Tuple[int, str]:
        old = self._emitted
        if target_text == old:
            return 0, ""
        prefix_len = 0
        limit = min(len(old), len(target_text))
        while prefix_len < limit and old[prefix_len] == target_text[prefix_len]:
            prefix_len += 1
        backspaces = len(old) - prefix_len
        suffix = target_text[prefix_len:]

        # Atomic single-RPC diff when using default `foveal.sock` broker
        if not self._custom_io and (backspaces > 0 or suffix):
            brokered = _broker_ears_type_diff(backspaces, suffix)
            if brokered is not None:
                self._emitted = target_text
                return backspaces, suffix

        if backspaces > 0:
            self._backspace_fn(backspaces)
        if suffix:
            self._type_fn(suffix)
        self._emitted = target_text
        return backspaces, suffix

    def update(self, new_text: str) -> Tuple[int, str]:
        """Apply differential update for a live partial transcript. Returns `(backspaces, appended)`."""
        now = time.monotonic()
        with self._lock:
            incoming = new_text.strip()
            if not incoming:
                # Never backspace already-spoken words on an empty partial reset
                if self._active_segment:
                    self._committed_segments.append(self._active_segment)
                    self._active_segment = ""
                return 0, ""

            prefix_str = " ".join(self._committed_segments).strip()
            # If incoming already includes our committed prefix (e.g. Swift accumulator handled it), sync cleanly
            if prefix_str and incoming.lower().startswith(prefix_str.lower()):
                remainder = incoming[len(prefix_str) :].strip()
                self._active_segment = remainder
                self._last_update_ts = now
                full = f"{prefix_str} {remainder}".strip() if remainder else prefix_str
                return self._apply_diff_locked(full)

            # If SFSpeechRecognizer re-included the just-committed last segment after a transient empty partial
            if (
                not self._active_segment
                and self._committed_segments
                and incoming.lower().startswith(self._committed_segments[-1].lower())
            ):
                self._committed_segments.pop()
                prefix_str = " ".join(self._committed_segments).strip()
            elif self._active_segment:
                cp = self._common_prefix_ci(self._active_segment, incoming)
                collapsed = (
                    len(self._active_segment) >= 10
                    and cp < 3
                    and len(incoming) < int(len(self._active_segment) * 0.60)
                )
                pause_replaced = (
                    len(self._active_segment) >= 3
                    and self._last_update_ts > 0
                    and (now - self._last_update_ts) >= 0.65
                    and cp < 2
                    and len(incoming) <= len(self._active_segment) + 2
                )
                if collapsed or pause_replaced:
                    self._committed_segments.append(self._active_segment)
                    prefix_str = " ".join(self._committed_segments).strip()

            self._active_segment = incoming
            self._last_update_ts = now
            full = f"{prefix_str} {self._active_segment}".strip() if prefix_str else self._active_segment
            return self._apply_diff_locked(full)

    def finalize(self, final_text: str) -> Tuple[int, str]:
        """Apply final AST-snapped text with a trailing space so consecutive dictations don't glue together."""
        with self._lock:
            merged = self._merge_segments_locked(final_text)
            if not merged:
                if self._emitted:
                    if not self._emitted.endswith(" "):
                        return self._apply_diff_locked(self._emitted + " ")
                    return 0, ""
                return 0, ""
            return self._apply_diff_locked(merged + " ")


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
        highlight: bool = True,
        highlight_target: str = "auto",
    ):
        self.engine = LocalEarsEngine(workspace=workspace, db_path=db_path, backend=backend)
        self.hotkey = (hotkey or "fn").strip().lower()
        self.highlight = bool(highlight) and os.environ.get("WBC_EARS_NO_HIGHLIGHT") != "1"
        self.highlight_target = (highlight_target or "auto").strip().lower()
        if self.highlight_target not in ("auto", "text_box", "window"):
            self.highlight_target = "auto"
        self._active_session: Optional[PTTSession] = None
        self._active_typer: Optional[LiveDifferentialTyper] = None
        self._session_lock = threading.Lock()

    def _show_highlight(self, mode: str = "ptt", max_duration_s: float = 95.0) -> None:
        if not self.highlight:
            return
        broker_ears_highlight(
            "show",
            mode=mode,
            target=self.highlight_target,
            max_duration_s=max_duration_s,
        )

    def _hide_highlight(self) -> None:
        if not self.highlight:
            return
        broker_ears_highlight("hide", target=self.highlight_target)

    def status(self) -> Dict[str, Any]:
        info = self.engine.status()
        info["hotkey"] = self.hotkey
        info["hands_free_chord"] = "fn+space" if self.hotkey == "fn" else f"toggle {self.hotkey}"
        info["highlight_enabled"] = bool(self.highlight)
        info["highlight_target"] = self.highlight_target
        info["voicefi_active"] = is_voicefi_dictation_active()
        info["yielding_to_voicefi"] = bool(
            info["voicefi_active"] and os.environ.get("WBC_EARS_FORCE_OVER_VOICEFI") != "1"
        )
        return info

    def upgrade_model(self) -> Dict[str, Any]:
        return self.engine.upgrade_model()

    def _record_foveal_transcript(self, res: TranscriptionResult) -> None:
        """Persist finalized AST-grounded transcript into Foveal Memory (`foveal_memory.sqlite`) with secret redaction."""
        if not res.text or os.environ.get("WBC_EARS_DISABLE_MEMORY") == "1":
            return
        try:
            from whole_body_compute.integrations.foveal_memory import record_foveal_audio_transcription

            record_foveal_audio_transcription(
                transcription=res.text,
                device="body_ears",
                transcription_engine=res.backend or "ane_sfspeech",
            )
        except Exception as exc:
            logger.debug("_record_foveal_transcript skipped: %s", exc)

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
            if res.text:
                self._record_foveal_transcript(res)
            if paste and res.text:
                paste_into_focused_app(res.text)
            return res

        self._show_highlight(mode="ptt", max_duration_s=max(10.0, float(max_sec) + 5.0))
        try:
            res = self.engine.transcribe_mic(
                silence_sec=silence_sec,
                max_sec=max_sec,
                active_file=active_file,
            )
            if res.text:
                self._record_foveal_transcript(res)
            if paste and res.text:
                paste_into_focused_app(res.text)
            return res
        finally:
            self._hide_highlight()

    def _on_ptt_start(self, mode: str) -> None:
        if is_voicefi_dictation_active() and os.environ.get("WBC_EARS_FORCE_OVER_VOICEFI") != "1":
            logger.debug("Yielding PTT hotkey to active VoiceFi daemon (:5140).")
            return
        with self._session_lock:
            if self._active_session is not None:
                return
            self._show_highlight(
                mode=mode,
                max_duration_s=180.0 if mode in ("hands_free", "opt+space") else 95.0,
            )
            # Live-type in BOTH Hold-Fn and Hands-Free modes using `type_unicode_direct` (flags=0)
            typer = LiveDifferentialTyper()

            def _on_partial(partial_text: str) -> None:
                typer.update(partial_text)

            session = self.engine.start_ptt_mic_session(on_partial=_on_partial)
            if session is None:
                self._hide_highlight()
                print("⚠️  Local ANE mic helper unavailable.", flush=True)
                return
            self._active_session = session
            self._active_typer = typer
            sym_cnt = len(session.bundle.all_symbols)
            mode_label = "Hands-Free" if mode == "hands_free" else "Hold Fn"
            print(f"🔴 Listening ({mode_label} • {sym_cnt} AST symbols • live-type)...", flush=True)

    def _on_ptt_stop(self) -> None:
        with self._session_lock:
            session = self._active_session
            typer = self._active_typer
            self._active_session = None
            self._active_typer = None
        self._hide_highlight()
        if session is None:
            return

        def _finalize() -> None:
            res = session.finish()
            if typer is not None and typer.committed_prefix:
                merged_raw = typer.merge_final_raw(res.raw_acoustic or "")
                if merged_raw and merged_raw != (res.raw_acoustic or "").strip():
                    res = self.engine.apply_vocab_fusion(
                        merged_raw,
                        bundle=res.vocab_bundle,
                        backend=res.backend,
                        latency_ms=res.latency_ms,
                        error=res.error,
                    )
            if res.error and not res.text:
                print(f"⚠️  {res.error}", flush=True)
            elif res.text:
                if typer is not None:
                    typer.finalize(res.text)
                else:
                    paste_into_focused_app(res.text + " ")
                self._record_foveal_transcript(res)
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
                        self._show_highlight(mode="hands_free", max_duration_s=180.0)
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
            return self._run_brokered_hotkey_stream(sm)

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
            self._hide_highlight()
            try:
                Quartz.CGEventTapEnable(tap, False)
            except Exception:
                pass
        return True

    def _run_brokered_hotkey_stream(self, sm: HotkeyStateMachine) -> bool:
        """Stream modifier/Space events from notarized `Whole Body Compute.app --ears-hotkey-stream`."""
        try:
            import json
            from whole_body_compute.daemon import resolve_wbc_eyes_app_executable

            app_bin = resolve_wbc_eyes_app_executable(auto_build=False)
            if app_bin is None:
                return False
            proc = subprocess.Popen(
                [str(app_bin), "--ears-hotkey-stream"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception:
            return False

        try:
            assert proc.stdout is not None
            first = proc.stdout.readline().strip()
            if not first:
                proc.terminate()
                return False
            first_obj = json.loads(first)
            if first_obj.get("event") != "ready":
                proc.terminate()
                return False

            if self.hotkey == "fn":
                print("   Hold [Fn] to dictate • Tap [Fn+Space] for hands-free • Ctrl+C to stop", flush=True)
            else:
                print(f"   Press [{self.hotkey.upper()}] to toggle dictation • Ctrl+C to stop", flush=True)

            for raw_line in proc.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                kind = ev.get("event")
                flags = int(ev.get("flags", 0))
                if kind == "flags":
                    sm.handle_flags_changed(flags)
                elif kind == "keydown":
                    action = sm.handle_key_down(int(ev.get("keycode", 0)), flags)
                    if action == "lock_hands_free":
                        self._show_highlight(mode="hands_free", max_duration_s=180.0)
                        print("🔒 Hands-Free locked (tap Fn or Fn+Space to stop)...", flush=True)
        except KeyboardInterrupt:
            print("\n🛑 body ears stopped.")
        finally:
            self._hide_highlight()
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except Exception:
                proc.kill()
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
