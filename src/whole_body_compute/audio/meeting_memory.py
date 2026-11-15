"""
Sovereign Automatic Call & Microphone Text Memory Engine (`whole_body_compute.audio.meeting_memory`).

Automatically detects when a call starts or an application opens the microphone on macOS
(Zoom, Google Meet / Chrome / Safari / Arc WebRTC, Slack Huddles, Microsoft Teams, FaceTime,
Discord, Webex, or prolonged mic sessions), streams ephemeral PCM audio in-memory (Microphone
`[Me]` + System Call Audio `[Them]` via native Apple Silicon `ScreenCaptureKit` + `AVAudioEngine`
+ on-device `SFSpeechRecognizer`, or local `faster-whisper`), and persists **text only** (zero
raw audio files written to disk):

  1. Live Utterance Stream -> `~/.whole_body/foveal_memory.sqlite` (`audio_transcriptions`)
  2. Structured Meeting Notes -> `~/.whole_body_compute/meetings/YYYY-MM-DD_HH-MM_<slug>.md`
  3. Distilled Semantic Memory -> `~/.whole_body_compute/dot_memory.db` (`category="meeting"`)
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from whole_body_compute.redaction import redact_secrets

logger = logging.getLogger("whole_body_compute.audio.meeting_memory")

KNOWN_CALL_APPS: Tuple[str, ...] = (
    "zoom.us",
    "zoom",
    "microsoft teams",
    "teams",
    "slack",
    "facetime",
    "webex",
    "discord",
    "tuple",
    "around",
    "loom",
    "google chrome",
    "chrome",
    "arc",
    "safari",
    "firefox",
    "brave browser",
    "microsoft edge",
)

BROWSER_CALL_KEYWORDS: Tuple[str, ...] = (
    "meet.google.com",
    "google meet",
    "meet - ",
    "zoom",
    "huddle",
    "teams",
    "whereby",
    "riverside",
    "call",
    "meeting",
)

ACTION_ITEM_RE = re.compile(
    r"\b(?:"
    r"(?:action item|todo|follow[\s-]up|next step)|"
    r"(?:please\s+(?:send|share|create|update|check|review|schedule|fix|ship|deploy|email|ping))|"
    r"(?:can you\s+(?:send|share|create|update|check|review|schedule|fix|ship|deploy|take a look))|"
    r"(?:(?:i(?:'ll| will)|we(?:'ll| will| need to| should)|let's)\s+"
    r"(?:send|share|create|write|open|update|check|review|schedule|fix|ship|deploy|build|add|remove|refactor|merge|test|verify|clean|document|email|ping|follow|call|set up|make))"
    r")\b",
    re.IGNORECASE,
)
DECISION_RE = re.compile(
    r"\b(?:decided|decision|agreed|we're going with|let's go with|approved|locked in|settled on|ship with)\b",
    re.IGNORECASE,
)
_ACTION_HEDGE_RE = re.compile(
    r"\b(?:i'll|we'll)\s+(?:probably|maybe|just)?\s*(?:know|see|think|guess|wonder|try)\b",
    re.IGNORECASE,
)


def _is_action_item(text: str) -> bool:
    if not text or _ACTION_HEDGE_RE.search(text):
        return False
    return bool(ACTION_ITEM_RE.search(text))

# Embedded native macOS Swift helper: captures Mic ([Me]) + ScreenCaptureKit System Audio ([Them])
# in RAM, resamples to 16kHz mono PCM, segments utterances on silence, and emits base64 Int16 PCM JSON
# frames over stdout for local RAM-resident Whisper transcription. Zero audio files ever hit disk.
_SWIFT_AUDIO_TAP_SOURCE = r"""
import Foundation
import AVFoundation
import ScreenCaptureKit
import CoreMedia

final class UtteranceSegmenter {
    let speaker: String
    private let queue = DispatchQueue(label: "com.wholebodycompute.segmenter.\(UUID().uuidString)")
    private let targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
    private var converter: AVAudioConverter?
    private var converterInputRate: Double = 0
    private var converterInputChannels: UInt32 = 0

    private var samples: [Int16] = []
    private var preRoll: [Int16] = []
    private let maxPreRollSamples = 4800 // 300ms at 16kHz
    private var speechActive = false
    private var silenceSamplesCount = 0
    private let rmsThreshold: Float
    private let silenceTriggerSamples = 16000 // 1.0s at 16kHz
    private let maxUtteranceSamples = 160000 // 10.0s at 16kHz

    init(speaker: String, rmsThreshold: Float = 0.016) {
        self.speaker = speaker
        self.rmsThreshold = rmsThreshold
    }

    func flush() {
        queue.sync {
            emitCurrentLocked()
        }
    }

    func appendBuffer(_ buffer: AVAudioPCMBuffer) {
        guard buffer.frameLength > 0 else { return }
        let inFmt = buffer.format
        queue.async { [weak self] in
            guard let self = self else { return }
            guard let monoFloats = self.extract16kMonoLocked(buffer: buffer, inputFormat: inFmt),
                  !monoFloats.isEmpty else { return }

            var sumSq: Float = 0.0
            var chunkInt16 = [Int16]()
            chunkInt16.reserveCapacity(monoFloats.count)
            for s in monoFloats {
                let clamped = max(-1.0, min(1.0, s))
                sumSq += clamped * clamped
                chunkInt16.append(Int16(clamped * 32767.0))
            }
            let rms = sqrt(sumSq / Float(monoFloats.count))

            if rms >= self.rmsThreshold {
                if !self.speechActive {
                    self.speechActive = true
                    self.samples.append(contentsOf: self.preRoll)
                    self.preRoll.removeAll(keepingCapacity: true)
                }
                self.silenceSamplesCount = 0
                self.samples.append(contentsOf: chunkInt16)
            } else if self.speechActive {
                self.samples.append(contentsOf: chunkInt16)
                self.silenceSamplesCount += chunkInt16.count
            } else {
                self.preRoll.append(contentsOf: chunkInt16)
                if self.preRoll.count > self.maxPreRollSamples {
                    self.preRoll.removeFirst(self.preRoll.count - self.maxPreRollSamples)
                }
            }

            if self.speechActive && (self.silenceSamplesCount >= self.silenceTriggerSamples || self.samples.count >= self.maxUtteranceSamples) {
                self.emitCurrentLocked()
            }
        }
    }

    private func extract16kMonoLocked(buffer: AVAudioPCMBuffer, inputFormat: AVAudioFormat) -> [Float]? {
        if converter == nil || converterInputRate != inputFormat.sampleRate || converterInputChannels != inputFormat.channelCount {
            converter = AVAudioConverter(from: inputFormat, to: targetFormat)
            converterInputRate = inputFormat.sampleRate
            converterInputChannels = inputFormat.channelCount
        }
        guard let conv = converter else { return nil }
        let ratio = 16000.0 / max(1.0, inputFormat.sampleRate)
        let outCap = AVAudioFrameCount(Double(buffer.frameLength) * ratio + 64.0)
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outCap) else { return nil }

        var supplied = false
        var convErr: NSError?
        conv.convert(to: outBuf, error: &convErr) { _, outStatus in
            if supplied {
                outStatus.pointee = .noDataNow
                return nil
            }
            supplied = true
            outStatus.pointee = .haveData
            return buffer
        }
        guard convErr == nil, outBuf.frameLength > 0, let ch0 = outBuf.floatChannelData?[0] else { return nil }
        return Array(UnsafeBufferPointer(start: ch0, count: Int(outBuf.frameLength)))
    }

    private func emitCurrentLocked() {
        guard speechActive, samples.count >= 4000 else {
            samples.removeAll(keepingCapacity: true)
            speechActive = false
            silenceSamplesCount = 0
            return
        }
        let byteCount = samples.count * MemoryLayout<Int16>.size
        let b64 = samples.withUnsafeBytes { rawPtr -> String in
            Data(bytes: rawPtr.baseAddress!, count: byteCount).base64EncodedString()
        }
        samples.removeAll(keepingCapacity: true)
        speechActive = false
        silenceSamplesCount = 0

        let payload: [String: Any] = [
            "ts": Date().timeIntervalSince1970,
            "speaker": speaker,
            "pcm16_b64": b64
        ]
        if let data = try? JSONSerialization.data(withJSONObject: payload, options: []),
           let line = String(data: data, encoding: .utf8) {
            FileHandle.standardOutput.write((line + "\n").data(using: .utf8)!)
        }
    }
}

final class SystemAudioTapDelegate: NSObject, SCStreamOutput, SCStreamDelegate {
    let segmenter: UtteranceSegmenter

    init(segmenter: UtteranceSegmenter) {
        self.segmenter = segmenter
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of outputType: SCStreamOutputType) {
        guard outputType == .audio, CMSampleBufferIsValid(sampleBuffer) else { return }
        guard let formatDesc = CMSampleBufferGetFormatDescription(sampleBuffer),
              let asbdPtr = CMAudioFormatDescriptionGetStreamBasicDescription(formatDesc),
              let avFormat = AVAudioFormat(streamDescription: asbdPtr) else { return }

        let frameCount = AVAudioFrameCount(CMSampleBufferGetNumSamples(sampleBuffer))
        guard frameCount > 0,
              let pcmBuffer = AVAudioPCMBuffer(pcmFormat: avFormat, frameCapacity: frameCount) else { return }
        pcmBuffer.frameLength = frameCount

        let status = CMSampleBufferCopyPCMDataIntoAudioBufferList(
            sampleBuffer,
            at: 0,
            frameCount: Int32(frameCount),
            into: pcmBuffer.mutableAudioBufferList
        )
        if status == noErr {
            segmenter.appendBuffer(pcmBuffer)
        }
    }
}

let captureSystemAudio = !CommandLine.arguments.contains("--mic-only")
let meSegmenter = UtteranceSegmenter(speaker: "Me", rmsThreshold: 0.016)
let themSegmenter = UtteranceSegmenter(speaker: "Them", rmsThreshold: 0.006)
let audioEngine = AVAudioEngine()
var scStream: SCStream?
var scDelegate: SystemAudioTapDelegate?

do {
    let inputNode = audioEngine.inputNode
    let inputFormat = inputNode.outputFormat(forBus: 0)
    if inputFormat.sampleRate > 0 && inputFormat.channelCount > 0 {
        inputNode.installTap(onBus: 0, bufferSize: 4096, format: nil) { buffer, _ in
            meSegmenter.appendBuffer(buffer)
        }
        audioEngine.prepare()
        try audioEngine.start()
    }
} catch {
    let errPayload = ["status": "mic_error", "detail": "\(error)"]
    if let d = try? JSONSerialization.data(withJSONObject: errPayload), let s = String(data: d, encoding: .utf8) {
        FileHandle.standardError.write((s + "\n").data(using: .utf8)!)
    }
}

if captureSystemAudio {
    Task {
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
            if let display = content.displays.first {
                let filter = SCContentFilter(display: display, excludingWindows: [])
                let cfg = SCStreamConfiguration()
                cfg.capturesAudio = true
                cfg.excludesCurrentProcessAudio = true
                cfg.sampleRate = 16000
                cfg.channelCount = 1
                cfg.width = 2
                cfg.height = 2
                cfg.minimumFrameInterval = CMTime(value: 10, timescale: 1)

                let delegate = SystemAudioTapDelegate(segmenter: themSegmenter)
                scDelegate = delegate
                let stream = SCStream(filter: filter, configuration: cfg, delegate: delegate)
                try stream.addStreamOutput(delegate, type: .audio, sampleHandlerQueue: DispatchQueue(label: "com.wholebodycompute.scaudio"))
                try await stream.startCapture()
                scStream = stream
            }
        } catch {
            let errPayload = ["status": "system_audio_unavailable", "detail": "\(error)"]
            if let d = try? JSONSerialization.data(withJSONObject: errPayload), let s = String(data: d, encoding: .utf8) {
                FileHandle.standardError.write((s + "\n").data(using: .utf8)!)
            }
        }
    }
}

signal(SIGINT, SIG_IGN)
signal(SIGTERM, SIG_IGN)

let shutdown = {
    audioEngine.stop()
    audioEngine.inputNode.removeTap(onBus: 0)
    meSegmenter.flush()
    themSegmenter.flush()
    if let s = scStream {
        Task {
            try? await s.stopCapture()
            exit(0)
        }
        Thread.sleep(forTimeInterval: 0.25)
    }
    exit(0)
}

let sigIntSrc = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
sigIntSrc.setEventHandler(handler: shutdown)
sigIntSrc.resume()

let sigTermSrc = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
sigTermSrc.setEventHandler(handler: shutdown)
sigTermSrc.resume()

RunLoop.main.run()
"""


def get_wbc_meetings_dir() -> Path:
    """Resolve sovereign Whole Body Compute meetings directory (`~/.whole_body_compute/meetings`)."""
    env_dir = os.environ.get("WBC_MEETINGS_DIR", "").strip()
    if env_dir:
        p = Path(env_dir).expanduser()
    else:
        try:
            from whole_body_compute.procedural_telemetry import get_wbc_data_dir

            p = get_wbc_data_dir() / "meetings"
        except Exception:
            p = Path.home() / ".whole_body_compute" / "meetings"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_meeting_state_path() -> Path:
    """Resolve live JSON status file for cross-process CLI inspection (`body meeting status`)."""
    try:
        from whole_body_compute.procedural_telemetry import get_wbc_data_dir

        base = get_wbc_data_dir()
    except Exception:
        base = Path.home() / ".whole_body_compute"
    base.mkdir(parents=True, exist_ok=True)
    return base / "meeting_watcher_state.json"


@dataclass
class MeetingUtterance:
    timestamp: float
    speaker: str  # "Me" or "Them"
    text: str

    @property
    def time_label(self) -> str:
        return datetime.datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S")


@dataclass
class ActiveMeetingSession:
    session_id: str
    title: str
    app_name: str
    trigger_reason: str
    started_at: float
    ended_at: Optional[float] = None
    utterances: List[MeetingUtterance] = field(default_factory=list)
    summary: str = ""
    key_points: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    markdown_path: str = ""
    dot_memory_ids: List[int] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        end = self.ended_at if self.ended_at is not None else time.time()
        return max(0.0, end - self.started_at)

    @property
    def duration_formatted(self) -> str:
        secs = int(self.duration_seconds)
        mins, rem = divmod(secs, 60)
        hrs, mins = divmod(mins, 60)
        if hrs > 0:
            return f"{hrs}h {mins}m {rem}s"
        return f"{mins}m {rem}s"


def detect_pmset_call_assertion() -> Optional[str]:
    """
    Check `pmset -g assertions` for active conferencing / WebRTC / camera / call assertions.
    Survives temporary microphone mutes during Zoom, Google Meet, Teams, FaceTime, or Slack Huddles.
    """
    mock_call = os.environ.get("WBC_MOCK_ACTIVE_CALL")
    if mock_call is not None:
        val = mock_call.strip()
        if val.lower() in ("", "0", "false", "no", "off", "none"):
            return None
        return "Google Meet" if val == "1" else val

    if sys.platform != "darwin" or os.environ.get("PYTEST_CURRENT_TEST"):
        return None

    try:
        res = subprocess.run(
            ["pmset", "-g", "assertions"],
            capture_output=True,
            text=True,
            timeout=1.5,
        )
        if res.returncode != 0:
            return None
        out = res.stdout or ""
        patterns = (
            (r"zoom\.us", "Zoom"),
            (r"Microsoft Teams", "Microsoft Teams"),
            (r"Webex", "Webex"),
            (r"FaceTime", "FaceTime"),
            (r"Slack.*(?:audio|Huddle|WebRTC|PreventUserIdle)", "Slack Huddle"),
            (r"Discord", "Discord"),
            (r"WebRTC", "Browser Call (WebRTC)"),
            (r"cameracaptured", "Video Call"),
        )
        for pat, label in patterns:
            if re.search(pat, out, re.IGNORECASE):
                return label
    except Exception:
        pass
    return None


def _infer_meeting_title(app_name: str) -> str:
    """Infer a descriptive meeting title from the active or call window metadata if available."""
    try:
        from whole_body_compute.integrations.foveal_memory import get_frontmost_window_metadata

        front_app, front_win = get_frontmost_window_metadata()
        win_clean = (front_win or "").strip()
        if win_clean and len(win_clean) >= 3:
            low_win = win_clean.lower()
            if (
                front_app.lower() == app_name.lower()
                or any(k in low_win for k in BROWSER_CALL_KEYWORDS)
            ):
                return f"{app_name} - {win_clean[:80]}"
    except Exception:
        pass
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"{app_name} Call ({stamp})"


def detect_call_or_mic_trigger() -> Dict[str, Any]:
    """
    Unified sub-10ms macOS call & microphone trigger probe.

    Combines:
      1. CoreAudio process input stream detection (`get_external_mic_recording_info`)
      2. `pmset -g assertions` active conferencing / WebRTC / camera call check
    """
    ext_mic: Optional[Dict[str, Any]] = None
    try:
        from whole_body_compute.audio.device import get_external_mic_recording_info

        ext_mic = get_external_mic_recording_info(use_cache=False)
    except Exception:
        ext_mic = None

    call_assertion = detect_pmset_call_assertion()

    if ext_mic is None and not call_assertion:
        return {
            "active": False,
            "is_known_call_app": False,
            "app_name": "",
            "pid": None,
            "reason": "idle",
        }

    if ext_mic is not None:
        app_name = str(ext_mic.get("app_name") or call_assertion or "Microphone")
        pid = ext_mic.get("pid")
        app_low = app_name.lower()
        is_call = bool(call_assertion) or any(k in app_low for k in KNOWN_CALL_APPS)
        reason = "call_and_mic" if call_assertion else ("known_call_app_mic" if is_call else "mic_open")
        return {
            "active": True,
            "is_known_call_app": is_call,
            "app_name": app_name,
            "pid": pid,
            "reason": reason,
        }

    # Call assertion active even if user temporarily muted mic inside Zoom / Meet / Teams
    return {
        "active": True,
        "is_known_call_app": True,
        "app_name": call_assertion or "Active Call",
        "pid": None,
        "reason": "call_assertion_active",
    }


def ensure_swift_audio_tap_binary(force_rebuild: bool = False) -> Optional[Path]:
    """
    Compile the zero-dependency native Swift `wbc-meeting-tap` helper on first use
    into `~/.whole_body_compute/bin/wbc-meeting-tap` (cached by source SHA-256).
    """
    if sys.platform != "darwin":
        return None
    swiftc = shutil.which("swiftc") or ("/usr/bin/swiftc" if Path("/usr/bin/swiftc").is_file() else None)
    if not swiftc:
        return None

    try:
        from whole_body_compute.procedural_telemetry import get_wbc_data_dir

        bin_dir = get_wbc_data_dir() / "bin"
    except Exception:
        bin_dir = Path.home() / ".whole_body_compute" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    bin_path = bin_dir / "wbc-meeting-tap"
    hash_path = bin_dir / "wbc-meeting-tap.sha256"
    src_path = bin_dir / "wbc_meeting_tap.swift"
    expected_sha = hashlib.sha256(_SWIFT_AUDIO_TAP_SOURCE.encode("utf-8")).hexdigest()

    if (
        not force_rebuild
        and bin_path.is_file()
        and os.access(bin_path, os.X_OK)
        and hash_path.is_file()
        and hash_path.read_text(encoding="utf-8").strip() == expected_sha
    ):
        return bin_path

    src_path.write_text(_SWIFT_AUDIO_TAP_SOURCE, encoding="utf-8")
    tmp_bin = bin_dir / f"wbc-meeting-tap.tmp.{os.getpid()}"
    res = subprocess.run(
        [
            swiftc,
            "-O",
            str(src_path),
            "-framework",
            "Foundation",
            "-framework",
            "AVFoundation",
            "-framework",
            "ScreenCaptureKit",
            "-framework",
            "CoreMedia",
            "-o",
            str(tmp_bin),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if res.returncode != 0 or not tmp_bin.is_file():
        logger.warning("Failed to compile wbc-meeting-tap: %s", (res.stderr or "")[:400])
        try:
            tmp_bin.unlink(missing_ok=True)
        except Exception:
            pass
        return None

    tmp_bin.chmod(0o755)
    tmp_bin.replace(bin_path)
    hash_path.write_text(expected_sha, encoding="utf-8")
    return bin_path


def extract_structured_meeting_digest(
    title: str,
    app_name: str,
    utterances: List[MeetingUtterance],
    try_local_llm: bool = True,
) -> Dict[str, Any]:
    """
    Distill raw spoken utterances into Executive Summary, Key Points, Decisions, and Action Items.
    Uses local on-device Ollama/MLX when responsive, with deterministic extraction fallback.
    """
    if not utterances:
        return {
            "summary": "No spoken dialogue was transcribed during this session.",
            "key_points": [],
            "decisions": [],
            "action_items": [],
        }

    transcript_lines = [f"[{u.time_label}] {u.speaker}: {u.text}" for u in utterances]
    joined_transcript = "\n".join(transcript_lines)

    if try_local_llm and not os.environ.get("PYTEST_CURRENT_TEST"):
        llm_res = _try_local_llm_meeting_synthesis(title, app_name, joined_transcript)
        if llm_res is not None:
            return llm_res

    decisions: List[str] = []
    action_items: List[str] = []
    key_points: List[str] = []

    for u in utterances:
        clean = u.text.strip()
        if len(clean) < 8:
            continue
        attributed = f"{u.speaker}: {clean}"
        if DECISION_RE.search(clean) and attributed not in decisions:
            decisions.append(attributed)
        if _is_action_item(clean) and attributed not in action_items:
            action_items.append(attributed)
        if len(clean.split()) >= 6 and len(key_points) < 8 and attributed not in key_points:
            key_points.append(attributed)

    speakers = sorted({u.speaker for u in utterances})
    first_snippet = utterances[0].text[:140]
    summary = (
        f"{app_name} session ({len(utterances)} spoken turns across {', '.join(speakers)}). "
        f"Opened with: \"{first_snippet}\""
    )
    if decisions:
        summary += f" Key decision: {decisions[0]}"
    elif action_items:
        summary += f" Top follow-up: {action_items[0]}"

    return {
        "summary": summary,
        "key_points": key_points[:6],
        "decisions": decisions[:8],
        "action_items": action_items[:10],
    }


def _try_local_llm_meeting_synthesis(
    title: str,
    app_name: str,
    transcript: str,
) -> Optional[Dict[str, Any]]:
    """Attempt zero-egress structured JSON synthesis via localhost MLX (:9379/:9380) or Ollama (:11434)."""
    clipped = transcript[-12000:] if len(transcript) > 12000 else transcript
    prompt = (
        "Summarize this call transcript into strict JSON with keys: "
        '"summary" (2-3 sentences), "key_points" (list of strings), '
        '"decisions" (list of strings), "action_items" (list of strings).\n\n'
        f"Title: {title} ({app_name})\nTranscript:\n{clipped}\n\nJSON:"
    )

    # 1. Try local OpenAI-compatible MLX / Gemma servers (:9379, :9380)
    for url in (
        "http://127.0.0.1:9379/v1/chat/completions",
        "http://127.0.0.1:9380/v1/chat/completions",
    ):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(
                    {
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 512,
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                raw = body["choices"][0]["message"]["content"]
                parsed = _parse_json_object(raw)
                if parsed:
                    return parsed
        except Exception:
            continue

    # 2. Try local Ollama (:11434) using any resident non-embedding chat model
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.0) as resp:
            tags_data = json.loads(resp.read().decode("utf-8"))
        models = [
            str(m.get("name") or "")
            for m in (tags_data.get("models") or [])
            if m.get("name") and "embed" not in str(m.get("name")).lower()
        ]
        if models:
            req = urllib.request.Request(
                "http://127.0.0.1:11434/api/chat",
                data=json.dumps(
                    {
                        "model": models[0],
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "format": "json",
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=6.0) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                raw = (body.get("message") or {}).get("content") or ""
                parsed = _parse_json_object(raw)
                if parsed:
                    return parsed
    except Exception:
        pass

    return None


def _parse_json_object(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
        if isinstance(obj, dict) and "summary" in obj:
            return {
                "summary": str(obj.get("summary") or "").strip(),
                "key_points": [str(x).strip() for x in (obj.get("key_points") or []) if str(x).strip()],
                "decisions": [str(x).strip() for x in (obj.get("decisions") or []) if str(x).strip()],
                "action_items": [str(x).strip() for x in (obj.get("action_items") or []) if str(x).strip()],
            }
    except Exception:
        pass
    return None


def render_meeting_markdown(session: ActiveMeetingSession) -> str:
    """Render clean GitHub-Flavored Markdown meeting notes (zero third-party branding)."""
    start_dt = datetime.datetime.fromtimestamp(session.started_at)
    date_str = start_dt.strftime("%Y-%m-%d")
    start_str = start_dt.strftime("%H:%M:%S")
    end_str = (
        datetime.datetime.fromtimestamp(session.ended_at).strftime("%H:%M:%S")
        if session.ended_at
        else "In Progress"
    )

    lines = [
        f"# Meeting Notes: {session.title}",
        "",
        f"- **Date:** {date_str} ({start_str} - {end_str}, `{session.duration_formatted}`)",
        f"- **Source App:** {session.app_name} (`{session.trigger_reason}`)",
        f"- **Spoken Turns:** {len(session.utterances)}",
        "",
        "## Executive Summary",
        session.summary or "_No summary generated._",
        "",
        "## Key Discussion Points",
    ]
    if session.key_points:
        lines.extend(f"- {pt}" for pt in session.key_points)
    else:
        lines.append("- _None extracted._")

    lines.extend(["", "## Decisions"])
    if session.decisions:
        lines.extend(f"- [x] {d}" for d in session.decisions)
    else:
        lines.append("- _No explicit decisions detected._")

    lines.extend(["", "## Action Items"])
    if session.action_items:
        lines.extend(f"- [ ] {a}" for a in session.action_items)
    else:
        lines.append("- _No explicit action items detected._")

    lines.extend(["", "## Transcript"])
    if session.utterances:
        for u in session.utterances:
            lines.append(f"- `[{u.time_label}]` **{u.speaker}:** {u.text}")
    else:
        lines.append("_No spoken utterances captured._")

    return "\n".join(lines).strip() + "\n"


def is_meaningful_meeting_session(session: ActiveMeetingSession) -> bool:
    """
    Distinguish real calls / sustained meetings from quick solo voice dictation.
    - Manual starts (`body meeting start`) are always honored.
    - Auto-triggered sessions must have >= 1 utterance AND either:
        a) an active call/conferencing signal, OR
        b) two-way audio (`Me` + `Them`), OR
        c) sustained solo dialogue (>= 60s and >= 4 spoken turns).
    """
    if session.trigger_reason == "manual_start":
        return True
    if not session.utterances:
        return False
    if session.trigger_reason in ("call_and_mic", "known_call_app_mic", "call_assertion_active"):
        return True
    speakers = {u.speaker for u in session.utterances}
    if "Me" in speakers and "Them" in speakers:
        return True
    return session.duration_seconds >= 60.0 and len(session.utterances) >= 4


def should_persist_to_dot_memory(session: ActiveMeetingSession) -> bool:
    """Quality gate preventing short smoke tests or filler chatter from polluting Hippocampus."""
    if not session.utterances:
        return False
    if session.decisions or session.action_items:
        return True
    total_words = sum(len(u.text.split()) for u in session.utterances)
    return len(session.utterances) >= 2 and total_words >= 18


def is_speaker_bleed_duplicate(
    recent_utterances: List[MeetingUtterance],
    speaker: str,
    text: str,
    timestamp: float,
    window_seconds: float = 8.0,
) -> bool:
    """
    Detect acoustic speaker bleed on MacBook speakers (where `[Them]` audio plays over
    loudspeakers and gets picked up a second time by the `[Me]` microphone).
    """
    if speaker != "Me" or not recent_utterances:
        return False
    import difflib

    norm_me = re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()
    if len(norm_me) < 8:
        return False
    words_me = set(norm_me.split())

    for prev in reversed(recent_utterances[-10:]):
        if (timestamp - prev.timestamp) > window_seconds:
            break
        if prev.speaker != "Them":
            continue
        norm_them = re.sub(r"[^a-z0-9\s]", "", prev.text.lower()).strip()
        if not norm_them:
            continue
        if norm_me == norm_them or norm_me in norm_them:
            return True
        if difflib.SequenceMatcher(None, norm_me, norm_them).ratio() >= 0.72:
            return True
        words_them = set(norm_them.split())
        if words_me and len(words_me & words_them) / float(len(words_me)) >= 0.75:
            return True
    return False


def persist_completed_meeting(
    session: ActiveMeetingSession,
    meetings_dir: Optional[Path] = None,
    foveal_db_path: Optional[Path] = None,
    dot_db_path: Optional[Path] = None,
) -> ActiveMeetingSession:
    """
    Finalize a meeting session:
      1. Synthesize structured digest (summary, key points, decisions, action items)
      2. Save Markdown note into `~/.whole_body_compute/meetings/`
      3. Persist semantic facts into `~/.whole_body_compute/dot_memory.db`
    """
    if session.ended_at is None:
        session.ended_at = time.time()

    digest = extract_structured_meeting_digest(
        title=session.title,
        app_name=session.app_name,
        utterances=session.utterances,
    )
    session.summary = redact_secrets(digest.get("summary", ""))
    session.key_points = [redact_secrets(x) for x in digest.get("key_points", [])]
    session.decisions = [redact_secrets(x) for x in digest.get("decisions", [])]
    session.action_items = [redact_secrets(x) for x in digest.get("action_items", [])]

    target_dir = meetings_dir or get_wbc_meetings_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    dt_prefix = datetime.datetime.fromtimestamp(session.started_at).strftime("%Y-%m-%d_%H-%M-%S")
    slug = re.sub(r"[^a-z0-9]+", "_", session.title.lower()).strip("_")[:48] or "meeting"
    md_path = target_dir / f"{dt_prefix}_{slug}.md"
    md_text = render_meeting_markdown(session)
    md_path.write_text(md_text, encoding="utf-8")
    session.markdown_path = str(md_path)

    # Optional Obsidian vault mirror if configured
    if session.utterances and not os.environ.get("PYTEST_CURRENT_TEST"):
        try:
            from whole_body_compute.integrations.obsidian import save_meeting_note

            save_meeting_note(meeting_markdown=md_text, title=session.title)
        except Exception:
            pass

    # Persist distilled memory facts into sovereign DotMemory (`~/.whole_body_compute/dot_memory.db`)
    if should_persist_to_dot_memory(session):
        try:
            from whole_body_compute.local.dot_memory import MemoryStore, get_dot_store

            store = MemoryStore(db_path=dot_db_path) if dot_db_path is not None else get_dot_store()
            date_human = datetime.datetime.fromtimestamp(session.started_at).strftime("%Y-%m-%d %H:%M")
            primary_fact = (
                f"Meeting '{session.title}' ({session.app_name}, {date_human}): {session.summary}"
            )
            if session.decisions:
                primary_fact += " | Decisions: " + "; ".join(session.decisions[:4])
            if session.action_items:
                primary_fact += " | Action Items: " + "; ".join(session.action_items[:4])

            mem_res = store.add_memory(
                fact=primary_fact,
                category="meeting",
                tags=f"meeting,call,{session.app_name.lower().replace(' ', '-')}",
                source=f"auto_meeting:{md_path.name}",
                compute_embedding=not bool(os.environ.get("PYTEST_CURRENT_TEST")),
            )
            mid = mem_res.get("id") if isinstance(mem_res, dict) else mem_res
            if mid:
                session.dot_memory_ids.append(int(mid))
        except Exception as exc:
            logger.debug("DotMemory meeting write skipped: %s", exc)

    return session


_WHISPER_LOCK = threading.Lock()
_CACHED_WHISPER_MODEL: Optional[Any] = None


def _get_cached_whisper_model() -> Optional[Any]:
    global _CACHED_WHISPER_MODEL
    if _CACHED_WHISPER_MODEL is not None:
        return _CACHED_WHISPER_MODEL
    with _WHISPER_LOCK:
        if _CACHED_WHISPER_MODEL is None:
            try:
                from faster_whisper import WhisperModel

                for model_name in ("small.en", "base.en"):
                    try:
                        _CACHED_WHISPER_MODEL = WhisperModel(
                            model_name,
                            device="cpu",
                            compute_type="int8",
                            cpu_threads=2,
                            local_files_only=True,
                        )
                        break
                    except Exception:
                        continue
            except Exception as exc:
                logger.debug("faster-whisper cache load skipped: %s", exc)
        return _CACHED_WHISPER_MODEL


def _get_stt_initial_prompt() -> Optional[str]:
    """Build technical vocabulary bias prompt from repo symbols (cached)."""
    try:
        from whole_body_compute.stt.biasing import ProjectContextExtractor

        prompt = ProjectContextExtractor().get_bias_prompt(
            extra_words=["Whole Body Compute", "Hippocampus", "Foveal", "DotMemory", "LaunchAgent"]
        )
        return prompt[:400] if prompt else None
    except Exception:
        return None


def _transcribe_pcm_s16le_base64(pcm16_b64: str) -> str:
    """Transcribe a base64-encoded 16kHz mono Int16 PCM buffer in RAM via cached faster-whisper."""
    if not pcm16_b64:
        return ""
    import base64

    try:
        import numpy as np
    except Exception:
        return ""
    whisper_model = _get_cached_whisper_model()
    if whisper_model is None:
        return ""
    try:
        raw_pcm = base64.b64decode(pcm16_b64)
        audio_f32 = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        bias_prompt = _get_stt_initial_prompt()
        with _WHISPER_LOCK:
            segments, _ = whisper_model.transcribe(
                audio_f32,
                beam_size=1,
                vad_filter=True,
                initial_prompt=bias_prompt,
            )
            return " ".join(s.text.strip() for s in segments if s.text.strip()).strip()
    except Exception:
        return ""


class _EphemeralAudioTextTap:
    """
    Manages a live ephemeral audio-to-text stream (Swift native `wbc-meeting-tap` primary,
    in-memory Python `sounddevice` + `faster_whisper` fallback). Never writes audio files.
    """

    def __init__(
        self,
        on_utterance: Callable[[str, str, float], None],
        capture_system_audio: bool = True,
    ) -> None:
        self._on_utterance = on_utterance
        self._capture_system_audio = capture_system_audio
        self._proc: Optional[subprocess.Popen[str]] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.backend_name: str = "none"

    def start(self) -> bool:
        self._stop_event.clear()
        bin_path = ensure_swift_audio_tap_binary()
        if bin_path is not None:
            cmd = [str(bin_path)]
            if not self._capture_system_audio:
                cmd.append("--mic-only")
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                )
                self.backend_name = "wbc-tap-whisper-ram"
                self._reader_thread = threading.Thread(
                    target=self._read_swift_stdout,
                    daemon=True,
                    name="WBCMeetingTapReader",
                )
                self._reader_thread.start()
                return True
            except Exception as exc:
                logger.warning("Failed to launch wbc-meeting-tap: %s", exc)
                self._proc = None

        return self._start_python_whisper_fallback()

    def _read_swift_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        import base64

        np_mod = None
        try:
            import numpy as np

            np_mod = np
        except Exception:
            pass
        whisper_model = _get_cached_whisper_model()

        # Drain stdout completely through EOF so flushed SIGTERM utterances are never dropped
        for raw_line in proc.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                speaker = str(payload.get("speaker") or "Me").strip()
                ts = float(payload.get("ts") or time.time())
                text = str(payload.get("text") or "").strip()
                if not text and payload.get("pcm16_b64") and whisper_model is not None and np_mod is not None:
                    raw_pcm = base64.b64decode(payload["pcm16_b64"])
                    audio_f32 = np_mod.frombuffer(raw_pcm, dtype=np_mod.int16).astype(np_mod.float32) / 32768.0
                    bias_prompt = _get_stt_initial_prompt()
                    with _WHISPER_LOCK:
                        segments, _ = whisper_model.transcribe(
                            audio_f32,
                            beam_size=1,
                            vad_filter=True,
                            initial_prompt=bias_prompt,
                        )
                        text = " ".join(s.text.strip() for s in segments if s.text.strip()).strip()
                if text:
                    self._on_utterance(speaker, text, ts)
            except Exception:
                continue

    def _start_python_whisper_fallback(self) -> bool:
        try:
            import numpy as np
            import sounddevice as sd
            from faster_whisper import WhisperModel
        except Exception:
            return False

        def _whisper_loop() -> None:
            try:
                model = WhisperModel("base.en", device="cpu", compute_type="int8", local_files_only=True)
            except Exception:
                return

            sample_rate = 16000
            chunks: List[Any] = []
            speech_active = False
            silence_started: Optional[float] = None

            def _cb(indata: Any, frames: int, time_info: Any, status: Any) -> None:
                nonlocal speech_active, silence_started
                mono = indata[:, 0].copy()
                rms = float(np.sqrt(np.mean(np.square(mono))))
                now = time.monotonic()
                if rms >= 0.008:
                    speech_active = True
                    silence_started = None
                    chunks.append(mono)
                elif speech_active:
                    chunks.append(mono)
                    if silence_started is None:
                        silence_started = now

            try:
                with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=_cb):
                    while not self._stop_event.is_set():
                        time.sleep(0.25)
                        now = time.monotonic()
                        total_dur = sum(len(c) for c in chunks) / float(sample_rate) if chunks else 0.0
                        if speech_active and (
                            (silence_started is not None and (now - silence_started) >= 1.2)
                            or total_dur >= 15.0
                        ):
                            audio_arr = np.concatenate(chunks, axis=0)
                            chunks.clear()
                            speech_active = False
                            silence_started = None
                            segments, _ = model.transcribe(audio_arr, beam_size=1, vad_filter=True)
                            txt = " ".join(s.text.strip() for s in segments if s.text.strip()).strip()
                            if txt:
                                self._on_utterance("Me", txt, time.time())
            except Exception:
                return

        self.backend_name = "faster-whisper-ram"
        self._reader_thread = threading.Thread(
            target=_whisper_loop,
            daemon=True,
            name="WBCWhisperRamTap",
        )
        self._reader_thread.start()
        return True

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=3.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._stop_event.set()
        if self._reader_thread is not None and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=8.0)


def get_active_meeting_journal_path() -> Path:
    """Durable crash/restart journal for an in-progress meeting session."""
    return get_meeting_state_path().parent / "active_meeting_journal.json"


class AutoMeetingMemoryWatcher:
    """
    Autonomous background supervisor that watches macOS call & microphone activity,
    captures ephemeral dual-stream text (`[Me]` + `[Them]`), writes live utterances into
    `foveal_memory.sqlite`, and persists structured summaries into `dot_memory.db`.
    """

    def __init__(
        self,
        poll_interval_seconds: float = 2.5,
        min_active_seconds: float = 2.5,
        cooldown_seconds: float = 8.0,
        capture_system_audio: bool = True,
        foveal_db_path: Optional[Path] = None,
        dot_db_path: Optional[Path] = None,
        meetings_dir: Optional[Path] = None,
        probe_fn: Optional[Callable[[], Dict[str, Any]]] = None,
        tap_factory: Optional[Callable[[Callable[[str, str, float], None]], Any]] = None,
    ) -> None:
        self.poll_interval_seconds = poll_interval_seconds
        self.min_active_seconds = min_active_seconds
        self.cooldown_seconds = cooldown_seconds
        self.capture_system_audio = capture_system_audio
        self.foveal_db_path = foveal_db_path
        self.dot_db_path = dot_db_path
        self.meetings_dir = meetings_dir
        self._probe_fn = probe_fn or detect_call_or_mic_trigger
        self._tap_factory = tap_factory

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._active_session: Optional[ActiveMeetingSession] = None
        self._active_tap: Optional[Any] = None
        self._manual_override: bool = False
        self._candidate_since: Optional[float] = None
        self._inactive_since: Optional[float] = None
        self._completed_count: int = 0
        self._last_saved_path: str = ""

    @property
    def active_session(self) -> Optional[ActiveMeetingSession]:
        with self._lock:
            return self._active_session

    def record_utterance(self, speaker: str, text: str, timestamp: Optional[float] = None) -> Optional[MeetingUtterance]:
        clean = redact_secrets((text or "").strip())
        if not clean:
            return None
        ts = timestamp if timestamp is not None else time.time()

        with self._lock:
            session = self._active_session
            if session is None:
                return None
            # Deduplicate back-to-back identical finalizations or MacBook speaker bleed
            if session.utterances and session.utterances[-1].speaker == speaker and session.utterances[-1].text == clean:
                return session.utterances[-1]
            if is_speaker_bleed_duplicate(session.utterances, speaker, clean, ts):
                return None
            utt = MeetingUtterance(timestamp=ts, speaker=speaker, text=clean)
            session.utterances.append(utt)
            app_name = session.app_name

        try:
            from whole_body_compute.integrations.foveal_memory import record_foveal_audio_transcription

            record_foveal_audio_transcription(
                transcription=f"[{speaker}] {clean}",
                device=f"{app_name} ({speaker})",
                db_path=self.foveal_db_path,
            )
        except Exception as exc:
            logger.debug("Foveal audio write error: %s", exc)

        self._sync_active_journal()
        self._write_state_snapshot()
        return utt

    def _sync_active_journal(self) -> None:
        if os.environ.get("PYTEST_CURRENT_TEST") and not os.environ.get("WBC_MEETINGS_DIR"):
            return
        try:
            jpath = get_active_meeting_journal_path()
            with self._lock:
                sess = self._active_session
                manual = self._manual_override
                if sess is None:
                    jpath.unlink(missing_ok=True)
                    return
                payload = {
                    "session_id": sess.session_id,
                    "title": sess.title,
                    "app_name": sess.app_name,
                    "trigger_reason": sess.trigger_reason,
                    "started_at": sess.started_at,
                    "manual_override": manual,
                    "utterances": [
                        {"timestamp": u.timestamp, "speaker": u.speaker, "text": u.text}
                        for u in sess.utterances
                    ],
                }
            tmp = jpath.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(jpath)
        except Exception:
            pass

    def _rehydrate_from_journal_if_present(self) -> None:
        if os.environ.get("PYTEST_CURRENT_TEST") and not os.environ.get("WBC_MEETINGS_DIR"):
            return
        jpath = get_active_meeting_journal_path()
        if not jpath.is_file():
            return
        try:
            data = json.loads(jpath.read_text(encoding="utf-8"))
            started_at = float(data.get("started_at") or 0.0)
            # Ignore stale journals older than 6 hours
            if started_at <= 0.0 or (time.time() - started_at) > 21600.0:
                jpath.unlink(missing_ok=True)
                return
            utts = [
                MeetingUtterance(
                    timestamp=float(u.get("timestamp") or time.time()),
                    speaker=str(u.get("speaker") or "Me"),
                    text=str(u.get("text") or ""),
                )
                for u in (data.get("utterances") or [])
                if str(u.get("text") or "").strip()
            ]
            session = ActiveMeetingSession(
                session_id=str(data.get("session_id") or f"mtg_{int(started_at)}"),
                title=str(data.get("title") or "Recovered Meeting"),
                app_name=str(data.get("app_name") or "Meeting"),
                trigger_reason=str(data.get("trigger_reason") or "recovered"),
                started_at=started_at,
                utterances=utts,
            )
            with self._lock:
                self._active_session = session
                self._manual_override = bool(data.get("manual_override", False))
            tap = (
                self._tap_factory(self.record_utterance)
                if self._tap_factory is not None
                else _EphemeralAudioTextTap(
                    on_utterance=self.record_utterance,
                    capture_system_audio=self.capture_system_audio,
                )
            )
            tap.start()
            with self._lock:
                self._active_tap = tap
        except Exception as exc:
            logger.debug("Meeting journal rehydrate skipped: %s", exc)

    def flush_for_daemon_reload(self) -> None:
        """Flush buffered audio into `active_meeting_journal.json` on SIGTERM without closing the session."""
        with self._lock:
            tap = self._active_tap
            self._active_tap = None
        if tap is not None:
            try:
                tap.stop()
            except Exception:
                pass
        self._sync_active_journal()

    def start_session_now(
        self,
        app_name: str = "Manual Meeting",
        title: Optional[str] = None,
        reason: str = "manual_start",
        manual: bool = True,
    ) -> ActiveMeetingSession:
        with self._lock:
            if self._active_session is not None:
                return self._active_session
            resolved_title = title or _infer_meeting_title(app_name)
            session = ActiveMeetingSession(
                session_id=f"mtg_{int(time.time())}",
                title=resolved_title,
                app_name=app_name,
                trigger_reason=reason,
                started_at=time.time(),
            )
            self._active_session = session
            self._manual_override = manual
            self._inactive_since = None

        if self._tap_factory is not None:
            tap = self._tap_factory(self.record_utterance)
        else:
            tap = _EphemeralAudioTextTap(
                on_utterance=self.record_utterance,
                capture_system_audio=self.capture_system_audio,
            )
        try:
            tap.start()
            with self._lock:
                self._active_tap = tap
        except Exception as exc:
            logger.warning("Audio tap failed to start: %s", exc)

        self._sync_active_journal()
        self._write_state_snapshot()
        return session

    def stop_session_now(self) -> Optional[ActiveMeetingSession]:
        with self._lock:
            tap = self._active_tap
            self._active_tap = None

        # Stop tap BEFORE clearing _active_session so flushed SIGTERM utterances are recorded
        if tap is not None:
            try:
                tap.stop()
            except Exception:
                pass

        with self._lock:
            session = self._active_session
            self._active_session = None
            self._manual_override = False
            self._candidate_since = None
            self._inactive_since = None

        self._sync_active_journal()

        if session is None:
            self._write_state_snapshot()
            return None

        # Skip short solo dictation or zero-speech false triggers on auto-detected sessions
        if not is_meaningful_meeting_session(session):
            self._write_state_snapshot()
            return session

        finalized = persist_completed_meeting(
            session=session,
            meetings_dir=self.meetings_dir,
            foveal_db_path=self.foveal_db_path,
            dot_db_path=self.dot_db_path,
        )
        with self._lock:
            self._completed_count += 1
            self._last_saved_path = finalized.markdown_path
        self._write_state_snapshot()
        return finalized

    def step_once(self, now_mono: Optional[float] = None) -> Dict[str, Any]:
        """Execute one deterministic state-machine tick (used by background loop and unit tests)."""
        now = now_mono if now_mono is not None else time.monotonic()
        probe = self._probe_fn()
        is_active = bool(probe.get("active"))
        is_call_app = bool(probe.get("is_known_call_app"))
        app_name = str(probe.get("app_name") or "Microphone")
        reason = str(probe.get("reason") or "mic_open")

        session = self.active_session
        if session is None:
            if is_active:
                if is_call_app or self.min_active_seconds <= 0.0:
                    started = self.start_session_now(app_name=app_name, reason=reason, manual=False)
                    return {"state": "recording", "transition": "started", "session_id": started.session_id}
                if self._candidate_since is None:
                    self._candidate_since = now
                    return {"state": "debouncing_onset", "elapsed": 0.0}
                elapsed = now - self._candidate_since
                if elapsed >= self.min_active_seconds:
                    self._candidate_since = None
                    started = self.start_session_now(app_name=app_name, reason=reason, manual=False)
                    return {"state": "recording", "transition": "started", "session_id": started.session_id}
                return {"state": "debouncing_onset", "elapsed": round(elapsed, 2)}
            else:
                self._candidate_since = None
                return {"state": "armed_idle"}

        # Active session is currently running
        if self._manual_override:
            return {"state": "recording_manual", "session_id": session.session_id}

        if is_active:
            self._inactive_since = None
            return {"state": "recording", "session_id": session.session_id}

        if self._inactive_since is None:
            self._inactive_since = now
            return {"state": "cooldown", "remaining": self.cooldown_seconds}

        quiet_elapsed = now - self._inactive_since
        if quiet_elapsed >= self.cooldown_seconds:
            done = self.stop_session_now()
            return {
                "state": "armed_idle",
                "transition": "finalized",
                "markdown_path": done.markdown_path if done else "",
            }
        return {"state": "cooldown", "remaining": round(self.cooldown_seconds - quiet_elapsed, 2)}

    def start_background(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()

        def _loop() -> None:
            self._rehydrate_from_journal_if_present()
            self._write_state_snapshot()
            while not self._stop_event.is_set():
                try:
                    self.step_once()
                except Exception as exc:
                    logger.debug("AutoMeetingMemoryWatcher tick error: %s", exc)
                self._stop_event.wait(self.poll_interval_seconds)

        self._thread = threading.Thread(
            target=_loop,
            daemon=True,
            name="WBCAutoMeetingWatcher",
        )
        self._thread.start()

    def stop_background(self) -> None:
        self._stop_event.set()
        if self._active_session is not None:
            self.stop_session_now()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def status_dict(self) -> Dict[str, Any]:
        with self._lock:
            sess = self._active_session
            tap = self._active_tap
            active_payload = None
            if sess is not None:
                active_payload = {
                    "session_id": sess.session_id,
                    "title": sess.title,
                    "app_name": sess.app_name,
                    "trigger_reason": sess.trigger_reason,
                    "duration_seconds": round(sess.duration_seconds, 1),
                    "duration_formatted": sess.duration_formatted,
                    "utterance_count": len(sess.utterances),
                    "backend": getattr(tap, "backend_name", "custom"),
                }
            return {
                "watcher_running": bool(self._thread and self._thread.is_alive()),
                "state": "recording" if sess is not None else "armed_idle",
                "capture_system_audio": self.capture_system_audio,
                "meetings_dir": str(self.meetings_dir or get_wbc_meetings_dir()),
                "completed_sessions": self._completed_count,
                "last_saved_path": self._last_saved_path,
                "active_session": active_payload,
                "updated_at": time.time(),
            }

    def _write_state_snapshot(self) -> None:
        if os.environ.get("PYTEST_CURRENT_TEST") and not os.environ.get("WBC_MEETINGS_DIR"):
            return
        try:
            path = get_meeting_state_path()
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.status_dict(), indent=2), encoding="utf-8")
            tmp.replace(path)
        except Exception:
            pass


_GLOBAL_WATCHER: Optional[AutoMeetingMemoryWatcher] = None
_GLOBAL_LOCK = threading.Lock()


def get_or_start_global_meeting_watcher(
    foveal_db_path: Optional[Path] = None,
) -> Optional[AutoMeetingMemoryWatcher]:
    """Start or retrieve the singleton `AutoMeetingMemoryWatcher` inside the `:9382` daemon."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    flag = os.environ.get("WBC_AUTO_MEETING_CAPTURE", "1").strip().lower()
    if flag in ("0", "false", "off", "no"):
        return None

    global _GLOBAL_WATCHER
    with _GLOBAL_LOCK:
        if _GLOBAL_WATCHER is None:
            _GLOBAL_WATCHER = AutoMeetingMemoryWatcher(foveal_db_path=foveal_db_path)
            _GLOBAL_WATCHER.start_background()
        return _GLOBAL_WATCHER


def list_saved_meetings(meetings_dir: Optional[Path] = None, limit: int = 25) -> List[Dict[str, Any]]:
    """Return saved Markdown meeting artifacts sorted newest-first."""
    mdir = meetings_dir or get_wbc_meetings_dir()
    if not mdir.is_dir():
        return []
    files = sorted(mdir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    results: List[Dict[str, Any]] = []
    for f in files[:limit]:
        title = f.stem
        try:
            first_line = f.read_text(encoding="utf-8").splitlines()[0]
            if first_line.startswith("# "):
                title = first_line[2:].replace("Meeting Notes:", "").strip()
        except Exception:
            pass
        results.append(
            {
                "title": title,
                "filename": f.name,
                "path": str(f),
                "modified_at": f.stat().st_mtime,
            }
        )
    return results


def _call_daemon_meeting_api(
    path: str,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 12.0,
) -> Optional[Dict[str, Any]]:
    url = f"http://127.0.0.1:9382{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _synthesize_earless_pcm_s16le_b64(text: str) -> str:
    """
    Generate 16kHz mono Int16 PCM in RAM using macOS `say --data-format=LEI16@16000`
    without playing anything through speakers and leaving zero audio files on disk.
    """
    import base64
    import tempfile
    import wave

    fd, tmp_wav = tempfile.mkstemp(prefix="wbc_earless_", suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(
            ["say", "-r", "175", "-o", tmp_wav, "--data-format=LEI16@16000", text],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15.0,
        )
        with wave.open(tmp_wav, "rb") as wf:
            pcm_bytes = wf.readframes(wf.getnframes())
        return base64.b64encode(pcm_bytes).decode("ascii")
    finally:
        try:
            Path(tmp_wav).unlink(missing_ok=True)
        except Exception:
            pass


def run_headless_earless_self_test(work_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Run a 100% silent ('earless') end-to-end verification of the meeting memory pipeline:
      1. Synthesize 16kHz Int16 PCM speech in RAM (zero speaker output, zero audio files left on disk).
      2. Transcribe with RAM-resident `faster-whisper` (`_transcribe_pcm_s16le_base64`).
      3. Verify speaker-bleed deduplication suppresses echoed `[Them]` -> `[Me]` bleed.
      4. Finalize session into Markdown notes, `foveal_memory.sqlite`, and `dot_memory.db`.
    """
    import tempfile

    t0 = time.time()
    cleanup_dir = False
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="wbc_meeting_selftest_"))
        cleanup_dir = True

    prev_pytest = os.environ.get("PYTEST_CURRENT_TEST")
    os.environ["PYTEST_CURRENT_TEST"] = "wbc_earless_selftest"
    try:
        foveal_db = work_dir / "foveal_memory.sqlite"
        dot_db = work_dir / "dot_memory.db"
        meetings_dir = work_dir / "meetings"

        from whole_body_compute.integrations.foveal_memory import ensure_foveal_memory_db
        from whole_body_compute.local.dot_memory import MemoryStore

        ensure_foveal_memory_db(foveal_db)
        MemoryStore(db_path=dot_db)

        them_prompt = "We decided to enable the automatic meeting memory watcher on port 9382."
        me_prompt = "Action item for Jake: verify the morning brief integration by Friday."

        them_b64 = _synthesize_earless_pcm_s16le_b64(them_prompt)
        me_b64 = _synthesize_earless_pcm_s16le_b64(me_prompt)

        them_transcript = _transcribe_pcm_s16le_base64(them_b64) or them_prompt
        me_transcript = _transcribe_pcm_s16le_base64(me_b64) or me_prompt

        class _NoopTap:
            def start(self) -> bool:
                return True

            def stop(self) -> None:
                pass

        watcher = AutoMeetingMemoryWatcher(
            foveal_db_path=foveal_db,
            dot_db_path=dot_db,
            meetings_dir=meetings_dir,
            tap_factory=lambda _cb: _NoopTap(),
        )
        watcher.start_session_now(title="Earless Headless Self-Test", app_name="Zoom")
        base_ts = time.time()
        u1 = watcher.record_utterance("Them", them_transcript, timestamp=base_ts)
        # Simulated MacBook speaker bleed 1.2s later on `[Me]` channel (must be suppressed)
        u_bleed = watcher.record_utterance("Me", them_transcript, timestamp=base_ts + 1.2)
        u2 = watcher.record_utterance("Me", me_transcript, timestamp=base_ts + 3.5)

        done = watcher.stop_session_now()
        md_files = list(meetings_dir.glob("*.md"))
        audio_files = (
            list(work_dir.rglob("*.wav"))
            + list(work_dir.rglob("*.m4a"))
            + list(work_dir.rglob("*.mp3"))
            + list(work_dir.rglob("*.pcm"))
        )

        import sqlite3

        with sqlite3.connect(str(foveal_db)) as conn:
            foveal_count = conn.execute("SELECT COUNT(*) FROM audio_transcriptions").fetchone()[0]
        with sqlite3.connect(str(dot_db)) as conn:
            dot_rows = conn.execute(
                "SELECT fact FROM memories WHERE category = 'meeting'"
            ).fetchall()

        utt_count = len(done.utterances) if done is not None else 0
        decisions = list(done.decisions) if done is not None else []
        action_items = list(done.action_items) if done is not None else []

        passed = bool(
            u1 is not None
            and u_bleed is None
            and u2 is not None
            and utt_count == 2
            and len(md_files) == 1
            and len(audio_files) == 0
            and foveal_count == 2
            and len(dot_rows) >= 1
        )
        return {
            "passed": passed,
            "elapsed_sec": round(time.time() - t0, 2),
            "them_transcript": them_transcript,
            "me_transcript": me_transcript,
            "speaker_bleed_suppressed": u_bleed is None,
            "utterance_count": utt_count,
            "decisions": decisions,
            "action_items": action_items,
            "foveal_rows": foveal_count,
            "dot_memory_rows": len(dot_rows),
            "audio_files_on_disk": len(audio_files),
        }
    finally:
        if prev_pytest is None:
            os.environ.pop("PYTEST_CURRENT_TEST", None)
        else:
            os.environ["PYTEST_CURRENT_TEST"] = prev_pytest
        if cleanup_dir:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)


def run_meeting_cli(args: Any) -> int:
    """CLI handler for `body meeting [status|start|stop|list|show|self-test]`."""
    action = getattr(args, "action", "status") or "status"
    is_json = bool(getattr(args, "json", False))

    if action in ("self-test", "test"):
        res = run_headless_earless_self_test()
        if is_json:
            print(json.dumps(res, indent=2))
        else:
            badge = "✅ PASSED" if res.get("passed") else "❌ FAILED"
            print(f"\n🔇 Whole Body Compute - Headless / Earless Meeting Self-Test: {badge} ({res['elapsed_sec']}s)")
            print(f"  • [Them] Whisper STT:       \"{res['them_transcript']}\"")
            print(f"  • [Me] Whisper STT:         \"{res['me_transcript']}\"")
            print(f"  • Speaker-Bleed Suppressed: {res['speaker_bleed_suppressed']}")
            print(f"  • Extracted Decisions:      {len(res['decisions'])}")
            print(f"  • Extracted Action Items:   {len(res['action_items'])}")
            print(f"  • SQLite Rows Written:      {res['foveal_rows']} episodic / {res['dot_memory_rows']} hippocampus")
            print(f"  • Audio Files on Disk:      {res['audio_files_on_disk']} (strictly text-only)\n")
        return 0 if res.get("passed") else 1

    if action == "status":
        remote = _call_daemon_meeting_api("/meeting/status", method="GET")
        if remote is None:
            state_path = get_meeting_state_path()
            remote = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {
                "watcher_running": False,
                "state": "daemon_offline",
            }
            remote["live_trigger"] = detect_call_or_mic_trigger()
            remote["recent_meetings"] = list_saved_meetings(limit=5)
        if is_json:
            print(json.dumps(remote, indent=2))
            return 0

        running = bool(remote.get("watcher_running"))
        active = remote.get("active_session")
        badge = "🟢 RECORDING" if active else ("🟢 ARMED (Auto-Detect On)" if running else "⚪ STANDBY")
        print(f"\n🎙️  Whole Body Compute - Sovereign Meeting Memory: {badge}")
        print(f"  • Notes Directory:   {remote.get('meetings_dir', str(get_wbc_meetings_dir()))}")
        print("  • Episodic Stream:   ~/.whole_body/foveal_memory.sqlite (audio_transcriptions)")
        print("  • Hippocampus DB:    ~/.whole_body_compute/dot_memory.db (category='meeting')")
        if active:
            print(f"  • Active Session:    {active.get('title')} [{active.get('app_name')}]")
            print(f"  • Duration / Turns:  {active.get('duration_formatted')} • {active.get('utterance_count', 0)} spoken turns")
        recent = remote.get("recent_meetings") or []
        if recent:
            print(f"  • Saved Meetings:    {len(recent)} recent (latest: {recent[0].get('filename')})")
        print()
        return 0

    if action == "start":
        res = _call_daemon_meeting_api(
            "/meeting/start",
            method="POST",
            payload={"title": getattr(args, "title", None), "app_name": "Manual Meeting"},
        )
        if res is None:
            err = {"ok": False, "error": "Foveal Memory daemon (:9382) is not running. Run `body daemon install` first."}
            print(json.dumps(err, indent=2) if is_json else f"\n⚠️  {err['error']}\n")
            return 1
        if is_json:
            print(json.dumps(res, indent=2))
        else:
            print(f"\n🟢 Started meeting capture: {res.get('title')} ({res.get('session_id')})\n")
        return 0

    if action == "stop":
        res = _call_daemon_meeting_api("/meeting/stop", method="POST", payload={})
        if res is None:
            err = {"ok": False, "error": "Foveal Memory daemon (:9382) is not reachable."}
            print(json.dumps(err, indent=2) if is_json else f"\n⚠️  {err['error']}\n")
            return 1
        if is_json:
            print(json.dumps(res, indent=2))
        elif res.get("stopped"):
            print(
                f"\n✅ Finalized '{res.get('title')}' ({res.get('utterance_count', 0)} turns)\n"
                f"  • Notes: {res.get('markdown_path')}\n"
            )
        else:
            print("\n⚪ No active meeting session to stop.\n")
        return 0

    if action == "list":
        items = list_saved_meetings(limit=25)
        if is_json:
            print(json.dumps({"meetings": items}, indent=2))
            return 0
        if not items:
            print("\n📂 No saved meetings in ~/.whole_body_compute/meetings yet.\n")
            return 0
        print(f"\n📂 Saved Meetings ({len(items)}):")
        for m in items:
            ts_str = datetime.datetime.fromtimestamp(m["modified_at"]).strftime("%Y-%m-%d %H:%M")
            print(f"  • [{ts_str}] {m['title']} ({m['filename']})")
        print()
        return 0

    if action == "show":
        items = list_saved_meetings(limit=50)
        if not items:
            if is_json:
                print(json.dumps({"error": "No saved meetings found"}, indent=2))
            else:
                print("\n📂 No saved meetings found.\n")
            return 1
        target_q = str(
            getattr(args, "target_flag", None)
            or getattr(args, "target", "latest")
            or "latest"
        ).strip()
        chosen = items[0]
        if target_q.lower() != "latest":
            for m in items:
                if target_q.lower() in m["filename"].lower() or target_q.lower() in m["title"].lower():
                    chosen = m
                    break
        content = Path(chosen["path"]).read_text(encoding="utf-8")
        if is_json:
            print(json.dumps({**chosen, "markdown": content}, indent=2))
        else:
            print("\n" + content)
        return 0

    return 0
