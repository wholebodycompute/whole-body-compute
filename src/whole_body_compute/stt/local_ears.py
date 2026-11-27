"""
Sovereign Local Speech-to-Text Engine (`whole_body_compute.stt.local_ears`).

Community Free Forever ($0) on-device transcription with zero cloud egress and zero
audio files written to disk:
  - Tier 0 (Default, 0 MB download): macOS native Apple Neural Engine (`SFSpeechRecognizer`
    with `requiresOnDeviceRecognition = true` + `contextualStrings` hydrated from `symbols.db`).
  - Tier 1 (Opt-in `body ears --upgrade-model`): Apple Silicon Metal `mlx-whisper`
    (`mlx-community/whisper-large-v3-turbo`, 64MB Metal cache cap) or `faster-whisper`
    on Linux/Intel, guarded by `ThermalSupervisor` thermal pressure fallback.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import platform
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from whole_body_compute.stt.vocab_fusion import ASTSymbolSnapper, ASTVocabHydrator, SnapReceipt, VocabBundle

logger = logging.getLogger("whole_body_compute.stt.local_ears")

DEFAULT_MLX_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_FASTER_WHISPER_MODEL = "base.en"

# Embedded zero-download macOS Swift helper:
# Supports both `--file <wav>` (offline file) and `--mic` (one-shot mic until silence/EOF),
# running strictly on Apple Neural Engine (`requiresOnDeviceRecognition = true`) with
# `contextualStrings` injected from `symbols.db`. Zero audio files written to disk.
_SWIFT_ANE_DICTATE_SOURCE = r"""
import Foundation
import AVFoundation
import Speech

func emitJSON(_ obj: [String: Any]) {
    if let data = try? JSONSerialization.data(withJSONObject: obj, options: []),
       let line = String(data: data, encoding: .utf8) {
        FileHandle.standardOutput.write((line + "\n").data(using: .utf8)!)
    }
}

func parseArgs() -> (mode: String, filePath: String?, contextStrings: [String], silenceSec: Double, maxSec: Double) {
    var mode = "mic"
    var filePath: String? = nil
    var contextStrings: [String] = []
    var silenceSec: Double = 1.4
    var maxSec: Double = 30.0
    let args = CommandLine.arguments
    var i = 1
    while i < args.count {
        switch args[i] {
        case "--file":
            if i + 1 < args.count { mode = "file"; filePath = args[i + 1]; i += 1 }
        case "--mic":
            mode = "mic"
        case "--context-json":
            if i + 1 < args.count,
               let data = args[i + 1].data(using: .utf8),
               let arr = try? JSONSerialization.jsonObject(with: data) as? [String] {
                contextStrings = arr
                i += 1
            }
        case "--silence":
            if i + 1 < args.count, let v = Double(args[i + 1]) { silenceSec = v; i += 1 }
        case "--max-seconds":
            if i + 1 < args.count, let v = Double(args[i + 1]) { maxSec = v; i += 1 }
        default:
            break
        }
        i += 1
    }
    return (mode, filePath, contextStrings, silenceSec, maxSec)
}

let rawArgs = CommandLine.arguments
if rawArgs.count >= 3 && rawArgs[1] == "--launch-daemon" {
    let task = Process()
    task.executableURL = URL(fileURLWithPath: rawArgs[2])
    task.arguments = Array(rawArgs.dropFirst(3))
    task.standardInput = FileHandle.standardInput
    task.standardOutput = FileHandle.standardOutput
    task.standardError = FileHandle.standardError

    signal(SIGINT, SIG_IGN)
    signal(SIGTERM, SIG_IGN)
    let intSrc = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global(qos: .userInteractive))
    let termSrc = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .global(qos: .userInteractive))
    intSrc.setEventHandler {
        if task.isRunning { kill(task.processIdentifier, SIGINT) }
    }
    termSrc.setEventHandler {
        if task.isRunning { kill(task.processIdentifier, SIGTERM) }
    }
    intSrc.resume()
    termSrc.resume()

    do {
        try task.run()
        task.waitUntilExit()
        exit(task.terminationStatus)
    } catch {
        let msg = "WholeBodyComputeEars --launch-daemon failed: \(error.localizedDescription)\n"
        if let data = msg.data(using: .utf8) {
            FileHandle.standardError.write(data)
        }
        exit(1)
    }
}

let cfg = parseArgs()
guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US")),
      recognizer.supportsOnDeviceRecognition else {
    emitJSON(["error": "on_device_speech_unavailable"])
    exit(2)
}

let authSem = DispatchSemaphore(value: 0)
var authStatus = SFSpeechRecognizer.authorizationStatus()
if authStatus == .notDetermined {
    SFSpeechRecognizer.requestAuthorization { st in
        authStatus = st
        authSem.signal()
    }
    _ = authSem.wait(timeout: .now() + 10.0)
}
guard authStatus == .authorized else {
    emitJSON(["error": "speech_permission_denied"])
    exit(3)
}

let doneSem = DispatchSemaphore(value: 0)
var bestText = ""

if cfg.mode == "file", let path = cfg.filePath {
    let url = URL(fileURLWithPath: path)
    let req = SFSpeechURLRecognitionRequest(url: url)
    req.requiresOnDeviceRecognition = true
    req.shouldReportPartialResults = false
    if !cfg.contextStrings.isEmpty {
        req.contextualStrings = Array(cfg.contextStrings.prefix(100))
    }
    recognizer.recognitionTask(with: req) { res, err in
        if let res = res {
            bestText = res.bestTranscription.formattedString
            if res.isFinal {
                emitJSON(["text": bestText, "backend": "ane_sfspeech"])
                doneSem.signal()
            }
        } else if let err = err {
            emitJSON(["error": err.localizedDescription, "text": bestText, "backend": "ane_sfspeech"])
            doneSem.signal()
        }
    }
    _ = doneSem.wait(timeout: .now() + 60.0)
    exit(0)
}

// Live RAM-only microphone capture until silence timeout or SIGINT/SIGTERM
let engine = AVAudioEngine()
let req = SFSpeechAudioBufferRecognitionRequest()
req.requiresOnDeviceRecognition = true
req.shouldReportPartialResults = true
if !cfg.contextStrings.isEmpty {
    req.contextualStrings = Array(cfg.contextStrings.prefix(100))
}

let inputNode = engine.inputNode
let format = inputNode.outputFormat(forBus: 0)
var lastSpeechTs = Date().timeIntervalSince1970
var heardSpeech = false
let startTs = lastSpeechTs

inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { buf, _ in
    req.append(buf)
    if let ch = buf.floatChannelData?[0] {
        let frames = Int(buf.frameLength)
        var sumSq: Float = 0
        for idx in 0..<frames { sumSq += ch[idx] * ch[idx] }
        let rms = sqrt(sumSq / max(1, Float(frames)))
        if rms > 0.015 {
            heardSpeech = true
            lastSpeechTs = Date().timeIntervalSince1970
        }
    }
}

signal(SIGINT, SIG_IGN)
signal(SIGTERM, SIG_IGN)
let sigSrc = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global(qos: .userInteractive))
let termSrc = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .global(qos: .userInteractive))

var finished = false
var lastEmittedPartial = ""
func emitAndExit() {
    if finished { return }
    finished = true
    emitJSON(["type": "final", "text": bestText, "backend": "ane_sfspeech"])
    fflush(stdout)
    exit(0)
}

func finishMic() {
    engine.stop()
    inputNode.removeTap(onBus: 0)
    req.endAudio()
    // Give ANE recognizer up to 1.8s to deliver isFinal, then flush bestText
    DispatchQueue.global().asyncAfter(deadline: .now() + 1.8) {
        emitAndExit()
    }
}

sigSrc.setEventHandler { finishMic() }
termSrc.setEventHandler { finishMic() }
sigSrc.resume()
termSrc.resume()

do {
    engine.prepare()
    try engine.start()
} catch {
    emitJSON(["error": "audio_engine_start_failed: \(error.localizedDescription)"])
    exit(4)
}

var committedSegments: [String] = []
var currentSegment = ""
let _ = recognizer.recognitionTask(with: req) { res, err in
    if let res = res {
        let raw = res.bestTranscription.formattedString.trimmingCharacters(in: .whitespacesAndNewlines)
        if raw.isEmpty {
            if !currentSegment.isEmpty && !finished {
                committedSegments.append(currentSegment)
                currentSegment = ""
            }
        } else {
            if !currentSegment.isEmpty && currentSegment.count >= 10 && raw.count < Int(Double(currentSegment.count) * 0.60)
               && !raw.lowercased().hasPrefix(String(currentSegment.lowercased().prefix(3))) {
                committedSegments.append(currentSegment)
            }
            currentSegment = raw
            bestText = (committedSegments + [currentSegment]).joined(separator: " ")
        }
        if res.isFinal {
            emitAndExit()
        } else if !bestText.isEmpty && bestText != lastEmittedPartial {
            lastEmittedPartial = bestText
            emitJSON(["type": "partial", "text": bestText, "backend": "ane_sfspeech"])
            fflush(stdout)
        }
    } else if err != nil {
        emitAndExit()
    }
}

DispatchQueue.global().async {
    while !finished {
        Thread.sleep(forTimeInterval: 0.1)
        let now = Date().timeIntervalSince1970
        if (heardSpeech && (now - lastSpeechTs) >= cfg.silenceSec) || (now - startTs) >= cfg.maxSec {
            finishMic()
            break
        }
    }
}

RunLoop.main.run()
"""


@dataclass
class TranscriptionResult:
    """Structured receipt returned by `LocalEarsEngine.transcribe_*`."""

    text: str
    raw_acoustic: str
    backend: str
    latency_ms: float
    vocab_bundle: VocabBundle
    ast_snaps: List[SnapReceipt] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "raw_acoustic": self.raw_acoustic,
            "backend": self.backend,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
            "vocab_context": {
                "workspace": self.vocab_bundle.workspace,
                "focused_app": self.vocab_bundle.focused_app,
                "active_file": self.vocab_bundle.active_file,
                "git_branch": self.vocab_bundle.git_branch,
                "symbols_biased": len(self.vocab_bundle.all_symbols),
                "hydration_ms": self.vocab_bundle.hydration_ms,
                "ast_snaps": [
                    {"spoken": s.spoken, "snapped": s.snapped, "type": s.match_type}
                    for s in self.ast_snaps
                ],
            },
        }


def check_macos_fn_key_conflict() -> Dict[str, Any]:
    """
    Inspect `com.apple.HIToolbox AppleFnUsageType` on macOS to warn if pressing `Fn`
    currently triggers Emoji Picker / Change Input Source instead of Do Nothing (0).
    """
    if platform.system() != "Darwin":
        return {"platform": platform.system(), "conflict": False, "usage_type": None}
    try:
        proc = subprocess.run(
            ["defaults", "read", "com.apple.HIToolbox", "AppleFnUsageType"],
            capture_output=True,
            text=True,
            timeout=1.0,
        )
        raw = proc.stdout.strip()
        usage = int(raw) if raw.isdigit() else None
        # 0 = Do Nothing (ideal for Fn push-to-talk), 1 = Input Source, 2 = Emoji, 3 = Dictation
        conflict = usage in (1, 2, 3)
        labels = {0: "Do Nothing", 1: "Change Input Source", 2: "Show Emoji & Symbols", 3: "Start Apple Dictation"}
        return {
            "platform": "Darwin",
            "usage_type": usage,
            "label": labels.get(usage, "Default"),
            "conflict": conflict,
            "tip": (
                "macOS System Settings -> Keyboard -> 'Press 🌐 key to' -> 'Do Nothing' "
                "(or run `body ears --hotkey opt+space`)."
                if conflict
                else None
            ),
        }
    except Exception:
        return {"platform": "Darwin", "conflict": False, "usage_type": None}


def _thermal_pressure_is_high() -> bool:
    """Return True if Apple Silicon ThermalSupervisor reports SERIOUS or CRITICAL pressure."""
    try:
        from whole_body_compute.local.supervisor import ThermalSupervisor

        sup = ThermalSupervisor()
        state = str(sup.get_thermal_state()).upper()
        return state in ("SERIOUS", "CRITICAL")
    except Exception:
        return False


def _enforce_mlx_cache_cap() -> None:
    """Cap MLX Metal buffer cache to 64 MB to prevent Unified RAM bloat."""
    try:
        import mlx.core as mx

        if hasattr(mx, "set_cache_limit"):
            mx.set_cache_limit(64 * 1024 * 1024)
        elif hasattr(mx, "metal") and hasattr(mx.metal, "set_cache_limit"):
            mx.metal.set_cache_limit(64 * 1024 * 1024)
    except Exception:
        pass


def _clear_mlx_cache() -> None:
    try:
        import mlx.core as mx

        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass


def _build_app_icon_icns(resources_dir: Path) -> bool:
    """Build `AppIcon.icns` from WBC's bundled PNG avatar using macOS `sips` + `iconutil`."""
    icns_path = resources_dir / "AppIcon.icns"
    if icns_path.is_file() and icns_path.stat().st_size > 1024:
        return True
    # Locate repo / installed PNG logo
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root / "assets" / "wbc-github-avatar-light.png",
        repo_root / "assets" / "wbc-github-avatar-dark.png",
        Path(__file__).resolve().parent.parent / "bundled" / "wbc-icon.png",
    ]
    src_png = next((c for c in candidates if c.is_file()), None)
    if src_png is None:
        return False
    iconset = resources_dir / "AppIcon.iconset"
    try:
        iconset.mkdir(parents=True, exist_ok=True)
        for sz in (16, 32, 64, 128, 256, 512):
            subprocess.run(
                ["sips", "-z", str(sz), str(sz), str(src_png), "--out", str(iconset / f"icon_{sz}x{sz}.png")],
                capture_output=True,
                timeout=10.0,
            )
            sz2 = sz * 2
            subprocess.run(
                ["sips", "-z", str(sz2), str(sz2), str(src_png), "--out", str(iconset / f"icon_{sz}x{sz}@2x.png")],
                capture_output=True,
                timeout=10.0,
            )
        proc = subprocess.run(
            ["iconutil", "-c", "icns", str(iconset), "-o", str(icns_path)],
            capture_output=True,
            timeout=10.0,
        )
        return proc.returncode == 0 and icns_path.is_file()
    except Exception as exc:
        logger.debug("AppIcon.icns build skipped: %s", exc)
        return False
    finally:
        try:
            import shutil

            shutil.rmtree(iconset, ignore_errors=True)
        except Exception:
            pass


def _get_ane_helper_binary() -> Optional[Path]:
    """
    Resolve the notarized `Whole Body Compute.app` binary (preferred - zero duplicate `.app`
    bundles or ad-hoc wrappers), falling back to a standalone cached Swift ANE helper in
    `~/.whole_body_compute/bin/` only when `Whole Body Compute.app` is not installed.
    """
    if platform.system() != "Darwin":
        return None

    try:
        from whole_body_compute.daemon import resolve_wbc_eyes_app_executable

        unified_bin = resolve_wbc_eyes_app_executable(auto_build=False)
        if unified_bin is not None:
            return unified_bin
    except Exception:
        pass

    base_dir = Path.home() / ".whole_body_compute"
    bin_dir = base_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(_SWIFT_ANE_DICTATE_SOURCE.encode("utf-8")).hexdigest()[:12]
    bin_path = bin_dir / f"wbc-ane-dictate-{digest}"
    if bin_path.is_file() and os.access(str(bin_path), os.X_OK):
        return bin_path

    src_path = bin_dir / f"wbc-ane-dictate-{digest}.swift"
    try:
        src_path.write_text(_SWIFT_ANE_DICTATE_SOURCE, encoding="utf-8")
        proc = subprocess.run(
            ["swiftc", str(src_path), "-O", "-o", str(bin_path)],
            capture_output=True,
            text=True,
            timeout=30.0,
        )
        if proc.returncode == 0 and bin_path.is_file():
            return bin_path
        logger.debug("swiftc compile failed: %s", proc.stderr)
    except Exception as exc:
        logger.debug("ANE helper build skipped: %s", exc)
    return None


def ensure_ears_app_executable() -> Optional[Path]:
    """
    Ensure `~/.whole_body_compute/Whole Body Compute.app` (with `AppIcon.icns`, `Info.plist`,
    and ad-hoc codesign) is built and return the path to its `CFBundleExecutable`
    (`Contents/MacOS/WholeBodyComputeEars`). Used by `com.wholebodycompute.ears` LaunchAgent
    so macOS Login Items and TCC permission dialogs display "Whole Body Compute" + `AppIcon.icns`.
    """
    return _get_ane_helper_binary()


class LocalEarsEngine:
    """
    Unified on-device Speech-to-Text engine with AST Vocab-Fusion.

    Backend resolution (`auto`):
      1. If `mlx_whisper` is installed on Apple Silicon AND thermal pressure is normal -> `mlx_whisper`
      2. Else on macOS -> `ane_sfspeech` (0 MB download, Apple Neural Engine)
      3. Else if `faster_whisper` is installed -> `faster_whisper`
    """

    def __init__(
        self,
        *,
        workspace: Optional[Path] = None,
        db_path: Optional[Path] = None,
        backend: str = "auto",
        mlx_model: str = DEFAULT_MLX_MODEL,
        whisper_model: str = DEFAULT_FASTER_WHISPER_MODEL,
    ):
        self.workspace = Path(workspace).resolve() if workspace else Path.cwd().resolve()
        self.hydrator = ASTVocabHydrator(workspace=self.workspace, db_path=db_path)
        self.requested_backend = (backend or "auto").strip().lower()
        self.mlx_model = mlx_model
        self.whisper_model = whisper_model
        self._lock = threading.Lock()
        self._fw_model: Any = None

    def resolve_backend(self) -> str:
        """Select the optimal available local STT backend."""
        req = self.requested_backend
        is_mac_arm = platform.system() == "Darwin" and platform.machine() == "arm64"
        has_mlx = importlib.util.find_spec("mlx_whisper") is not None
        has_fw = importlib.util.find_spec("faster_whisper") is not None

        if req in ("ane", "ane_sfspeech", "apple"):
            return "ane_sfspeech"
        if req in ("mlx", "mlx_whisper") and has_mlx and not _thermal_pressure_is_high():
            return "mlx_whisper"
        if req in ("faster_whisper", "whisper") and has_fw:
            return "faster_whisper"

        # Auto selection: prefer opt-in MLX Metal when installed & cool, else 0-MB ANE on macOS
        if is_mac_arm and has_mlx and not _thermal_pressure_is_high():
            return "mlx_whisper"
        if platform.system() == "Darwin":
            return "ane_sfspeech"
        if has_fw:
            return "faster_whisper"
        return "unavailable"

    def status(self) -> Dict[str, Any]:
        """Return diagnostic status for `body ears --status`."""
        bundle = self.hydrator.hydrate(skip_eyes=True)
        return {
            "backend": self.resolve_backend(),
            "mlx_whisper_installed": importlib.util.find_spec("mlx_whisper") is not None,
            "faster_whisper_installed": importlib.util.find_spec("faster_whisper") is not None,
            "thermal_high": _thermal_pressure_is_high(),
            "workspace": str(self.workspace),
            "symbols_indexed": len(bundle.all_symbols),
            "hydration_ms": bundle.hydration_ms,
            "fn_key_check": check_macos_fn_key_conflict(),
        }

    def upgrade_model(self) -> Dict[str, Any]:
        """Opt-in 1-command local Whisper install (`body ears --upgrade-model`)."""
        is_mac_arm = platform.system() == "Darwin" and platform.machine() == "arm64"
        pkg = "mlx-whisper" if is_mac_arm else "faster-whisper"
        cmd = [sys.executable, "-m", "pip", "install", "-U", pkg]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 and "No module named pip" in (proc.stderr or ""):
            import shutil

            uv_bin = shutil.which("uv")
            if uv_bin:
                cmd = [uv_bin, "pip", "install", "--python", sys.executable, "-U", pkg]
                proc = subprocess.run(cmd, capture_output=True, text=True)
        return {
            "ok": proc.returncode == 0,
            "package": pkg,
            "model": self.mlx_model if is_mac_arm else self.whisper_model,
            "stdout": proc.stdout[-500:] if proc.stdout else "",
            "stderr": proc.stderr[-500:] if proc.stderr else "",
        }

    def _decode_ane_file(self, audio_path: Path, bundle: VocabBundle) -> Tuple[str, Optional[str]]:
        helper = _get_ane_helper_binary()
        if helper is None:
            return "", "ane_helper_unavailable"
        ctx_json = json.dumps(bundle.to_ane_contextual_strings(limit=100))
        proc = subprocess.run(
            [str(helper), "--file", str(audio_path), "--context-json", ctx_json],
            capture_output=True,
            text=True,
            timeout=65.0,
        )
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                return str(data.get("text", "")).strip(), data.get("error")
            except Exception:
                continue
        return "", proc.stderr.strip() or "ane_no_output"

    def _decode_ane_mic(self, bundle: VocabBundle, silence_sec: float = 1.4, max_sec: float = 30.0) -> Tuple[str, Optional[str]]:
        helper = _get_ane_helper_binary()
        if helper is None:
            return "", "ane_helper_unavailable"
        ctx_json = json.dumps(bundle.to_ane_contextual_strings(limit=100))
        proc = subprocess.run(
            [
                str(helper),
                "--mic",
                "--context-json",
                ctx_json,
                "--silence",
                str(silence_sec),
                "--max-seconds",
                str(max_sec),
            ],
            capture_output=True,
            text=True,
            timeout=max_sec + 10.0,
        )
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                return str(data.get("text", "")).strip(), data.get("error")
            except Exception:
                continue
        return "", proc.stderr.strip() or "ane_no_output"

    def start_ptt_mic_session(
        self,
        *,
        active_file: Optional[str] = None,
        max_sec: float = 90.0,
        on_partial: Optional[Any] = None,
    ) -> Optional["PTTSession"]:
        """
        Start a non-blocking RAM-only ANE mic session that records until `finish()`
        is called (e.g. on `Fn` key release or `Fn+Space` second press).
        If `on_partial(text: str)` is provided, streams live partial transcripts as you speak.
        """
        helper = _get_ane_helper_binary()
        if helper is None:
            return None
        t0 = time.perf_counter()
        bundle = self.hydrator.hydrate(active_file_override=active_file)
        ctx_json = json.dumps(bundle.to_ane_contextual_strings(limit=100))
        proc = subprocess.Popen(
            [
                str(helper),
                "--mic",
                "--context-json",
                ctx_json,
                "--silence",
                str(max_sec),  # Controlled explicitly by key release / SIGINT
                "--max-seconds",
                str(max_sec),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        return PTTSession(engine=self, proc=proc, bundle=bundle, start_ts=t0, on_partial=on_partial)

    def _decode_mlx(self, audio_input: Any, bundle: VocabBundle) -> Tuple[str, Optional[str]]:
        try:
            import mlx_whisper

            with self._lock:
                _enforce_mlx_cache_cap()
                try:
                    res = mlx_whisper.transcribe(
                        audio_input,
                        path_or_hf_repo=self.mlx_model,
                        language="en",
                        initial_prompt=bundle.to_whisper_prompt(max_terms=64),
                        temperature=0.0,
                        condition_on_previous_text=False,
                    )
                finally:
                    _clear_mlx_cache()
            text = res.get("text", "").strip() if isinstance(res, dict) else str(res).strip()
            return text, None
        except Exception as exc:
            return "", str(exc)

    def _decode_faster_whisper(self, audio_input: Any, bundle: VocabBundle) -> Tuple[str, Optional[str]]:
        try:
            from faster_whisper import WhisperModel

            with self._lock:
                if self._fw_model is None:
                    self._fw_model = WhisperModel(self.whisper_model, device="cpu", compute_type="int8")
                segments, _ = self._fw_model.transcribe(
                    audio_input,
                    language="en",
                    initial_prompt=bundle.to_whisper_prompt(max_terms=64),
                    beam_size=1,
                    vad_filter=True,
                )
                text = " ".join(s.text.strip() for s in segments if getattr(s, "text", "").strip()).strip()
            return text, None
        except Exception as exc:
            return "", str(exc)

    def apply_vocab_fusion(
        self,
        raw_acoustic: str,
        *,
        bundle: Optional[VocabBundle] = None,
        backend: str = "manual",
        latency_ms: float = 0.0,
        error: Optional[str] = None,
    ) -> TranscriptionResult:
        """Run Stage-2 AST Symbol Snapping + PhoneticNormalizer on a raw acoustic string."""
        b = bundle or self.hydrator.hydrate(skip_eyes=True)
        snapper = ASTSymbolSnapper(b)
        snapped_text, receipts = snapper.snap(raw_acoustic)
        return TranscriptionResult(
            text=snapped_text,
            raw_acoustic=raw_acoustic,
            backend=backend,
            latency_ms=latency_ms,
            vocab_bundle=b,
            ast_snaps=receipts,
            error=error,
        )

    def transcribe_file(
        self,
        audio_path: Union[str, Path],
        *,
        active_file: Optional[str] = None,
        skip_eyes: bool = True,
    ) -> TranscriptionResult:
        """Transcribe an offline audio file with live repo AST vocab grounding."""
        t0 = time.perf_counter()
        p = Path(audio_path).expanduser().resolve()
        bundle = self.hydrator.hydrate(active_file_override=active_file, skip_eyes=skip_eyes)
        if not p.is_file():
            return TranscriptionResult(
                text="",
                raw_acoustic="",
                backend="none",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                vocab_bundle=bundle,
                error=f"Audio file not found: {p}",
            )

        backend = self.resolve_backend()
        raw, err = "", None
        if backend == "mlx_whisper":
            raw, err = self._decode_mlx(str(p), bundle)
        elif backend == "ane_sfspeech":
            raw, err = self._decode_ane_file(p, bundle)
        elif backend == "faster_whisper":
            raw, err = self._decode_faster_whisper(str(p), bundle)
        else:
            err = "No local STT backend available. Run `body ears --upgrade-model` to install faster-whisper."

        return self.apply_vocab_fusion(
            raw,
            bundle=bundle,
            backend=backend,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            error=err,
        )

    def transcribe_mic(
        self,
        *,
        silence_sec: float = 1.4,
        max_sec: float = 30.0,
        active_file: Optional[str] = None,
    ) -> TranscriptionResult:
        """Record one-shot RAM-only mic utterance and return AST-grounded transcript."""
        t0 = time.perf_counter()
        bundle = self.hydrator.hydrate(active_file_override=active_file)
        backend = self.resolve_backend()

        # On macOS, ANE SFSpeechRecognizer handles zero-dependency RAM mic capture directly
        if platform.system() == "Darwin" and backend in ("ane_sfspeech", "mlx_whisper"):
            raw, err = self._decode_ane_mic(bundle, silence_sec=silence_sec, max_sec=max_sec)
            used_backend = "ane_sfspeech"
        else:
            raw, err = "", (
                "Live microphone capture requires macOS on-device Speech or `sounddevice` + `faster-whisper`. "
                "Pass an audio file path to `body transcribe <file.wav>` or run `body ears --upgrade-model`."
            )
            used_backend = backend

        return self.apply_vocab_fusion(
            raw,
            bundle=bundle,
            backend=used_backend,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            error=err,
        )


class PTTSession:
    """Handle for an in-flight push-to-talk RAM-only ANE recording session with live partial streaming."""

    def __init__(
        self,
        *,
        engine: LocalEarsEngine,
        proc: subprocess.Popen,
        bundle: VocabBundle,
        start_ts: float,
        on_partial: Optional[Any] = None,
    ):
        self.engine = engine
        self.proc = proc
        self.bundle = bundle
        self.start_ts = start_ts
        self.on_partial = on_partial
        self._finished = False
        self._final_text: str = ""
        self._error: Optional[str] = None
        self._done_evt = threading.Event()
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True, name="PTTSessionReader")
        self._reader_thread.start()

    def _read_loop(self) -> None:
        try:
            if self.proc.stdout:
                for line in self.proc.stdout:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue
                    msg_type = data.get("type", "final")
                    txt = str(data.get("text", "")).strip()
                    if data.get("error"):
                        self._error = str(data.get("error"))
                    if msg_type == "partial" and txt:
                        self._final_text = txt
                        if self.on_partial:
                            try:
                                self.on_partial(txt)
                            except Exception:
                                pass
                    elif msg_type == "final":
                        if txt:
                            self._final_text = txt
                        break
        finally:
            self._done_evt.set()

    def finish(self, timeout: float = 8.0) -> TranscriptionResult:
        """Send SIGINT to stop mic cleanly, wait for final ANE transcript, and run Stage-2 AST snap."""
        import signal

        if self._finished:
            return self.engine.apply_vocab_fusion("", bundle=self.bundle, backend="ane_sfspeech")
        self._finished = True
        try:
            if self.proc.poll() is None:
                self.proc.send_signal(signal.SIGINT)
            if not self._done_evt.wait(timeout=timeout):
                try:
                    self.proc.kill()
                except Exception:
                    pass
                if not self._final_text:
                    self._error = "ptt_timeout"
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass

        return self.engine.apply_vocab_fusion(
            self._final_text,
            bundle=self.bundle,
            backend="ane_sfspeech",
            latency_ms=(time.perf_counter() - self.start_ts) * 1000.0,
            error=self._error,
        )
