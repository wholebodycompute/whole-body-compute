"""
Dynamic End-of-Turn & Barge-In Detection Engine for VoiceFi.

Implements the Shortwire / Cue v1 hybrid architecture for local Apple Silicon:
1. Acoustic Prosody Scorer (`SmartTurnONNX`):
   - Runs Pipecat's `smart-turn-v3.2-cpu.onnx` (8.3 MB INT8 Whisper-Tiny encoder + classification head)
   - Evaluates the last 8.0 seconds of 16kHz audio (left-padded so pause onset sits at the right edge)
   - Captures pitch contour (F0 drop vs. suspended mid-thought pitch), final syllable lengthening,
     glottal holds, and micro-inhales in ~9 ms on CPU.
2. Semantic 1-Token Logit Scorer (`OllamaLogitScorer` + `LexicalTurnScorer`):
   - Queries a local GGUF model (`qwen2.5-coder:1.5b` via Ollama `/v1/chat/completions`) with
     `max_tokens=1`, `logprobs=True`, `reasoning_effort="none"`, and a static KV-cached system prefix.
   - Extracts exact softmax probability p(DONE) vs. p(WAIT) in ~30 ms.
   - Guarded by `ThermalSupervisor` and backed by a 0.05 ms deterministic `LexicalTurnScorer`
     for partial phone numbers, partial IP addresses, trailing conjunctions, and code tokens.
3. Combined Scoring & Dynamic Wait Table (`CueTurnCascade`):
   - Combined score: `score = p_end * sqrt(p_voice)`
   - Maps to dynamic silence wait durations:
     * DONE        -> 0.50s (Saves ~900ms of dead air vs. 1.40s baseline)
     * LIKELY_DONE -> 1.00s (Saves ~400ms of dead air)
     * MID_THOUGHT -> 1.75s (Holds mic open through "Christopher Walken" thinking pauses)
     * HOLDING     -> 2.25s (Holds mic open during unfinished numbers, lists, or clauses)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import os
import sys
from pathlib import Path
import threading
import re
import time
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

TurnState = Literal["DONE", "LIKELY_DONE", "MID_THOUGHT", "HOLDING"]
BargeInAction = Literal["STOP", "CONTINUE"]

# Shortwire / Cue v1 dynamic silence wait durations (seconds)
TURN_STATE_WAITS: Dict[TurnState, float] = {
    "DONE": 0.50,
    "LIKELY_DONE": 1.00,
    "MID_THOUGHT": 1.75,
    "HOLDING": 2.25,
}

# Number words for detecting partial spoken phone numbers / digit sequences
_SPOKEN_DIGITS = {
    "zero",
    "oh",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
}

# Trailing words that strongly indicate the speaker is mid-clause
_TRAILING_CONTINUATION_WORDS = {
    # Conjunctions
    "and",
    "or",
    "but",
    "nor",
    "so",
    "yet",
    "because",
    "although",
    "since",
    "unless",
    "while",
    "whereas",
    "whether",
    "if",
    "when",
    "whenever",
    # Prepositions
    "to",
    "for",
    "with",
    "without",
    "in",
    "into",
    "on",
    "onto",
    "at",
    "from",
    "by",
    "about",
    "between",
    "through",
    "during",
    "before",
    "after",
    "under",
    "over",
    "within",
    "via",
    "using",
    # Determiners / Pronouns / Auxiliaries
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "my",
    "your",
    "our",
    "their",
    "its",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "will",
    "would",
    "should",
    "could",
    "can",
    "may",
    "might",
    # Hesitations & fillers
    "um",
    "umm",
    "uh",
    "uhh",
    "er",
    "erm",
    "hmm",
    "hm",
    "like",
    # Technical / coding continuation tokens
    "function",
    "def",
    "class",
    "import",
    "return",
    "const",
    "let",
    "var",
    "port",
    "localhost",
    "equals",
    "called",
    "named",
}

# Trailing multi-word phrases indicating ongoing thought
_TRAILING_HOLD_PHRASES = (
    "wait",
    "hold on",
    "let me think",
    "let me see",
    "let's see",
    "what about",
    "how about",
    "such as",
    "for example",
    "in order to",
    "going to",
    "want to",
    "need to",
    "trying to",
    "want to do",
    "going to do",
    "need to do",
    "trying to do",
    "what i want",
    "so what i",
    "is to",
    "ip address is",
    "number is",
)

# Short complete standalone commands
_STANDALONE_COMPLETE_PHRASES = {
    "yes",
    "yep",
    "yeah",
    "no",
    "nope",
    "stop",
    "cancel",
    "proceed",
    "continue",
    "go ahead",
    "ship it",
    "run it",
    "do it",
    "looks good",
    "sounds good",
    "thank you",
    "thanks",
    "done",
    "next",
    "skip",
}

# Passive backchannel utterances during AI speech (CONTINUE speaking)
_BACKCHANNEL_CONTINUE_PHRASES = {
    "uh-huh",
    "uh huh",
    "mm-hmm",
    "mm hmm",
    "mhm",
    "hmm",
    "hm",
    "yeah",
    "yep",
    "yup",
    "okay",
    "ok",
    "right",
    "sure",
    "got it",
    "gotcha",
    "i see",
    "makes sense",
    "nice",
    "cool",
    "great",
    "awesome",
    "interesting",
    "oh",
    "ah",
}

# Active barge-in interruption triggers during AI speech (STOP speaking immediately)
_BARGE_IN_STOP_WORDS = {
    "wait",
    "stop",
    "hold",
    "pause",
    "no",
    "nope",
    "actually",
    "hang on",
    "hold on",
    "cancel",
    "nevermind",
    "never mind",
    "listen",
    "hey",
    "excuse me",
    "wrong",
    "instead",
    "skip",
}


@dataclass
class TurnDecision:
    """Result of an end-of-turn evaluation at pause onset."""

    state: TurnState
    wait_seconds: float
    score: float
    p_end: float
    p_voice: float
    latency_ms: float
    engine: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BargeInDecision:
    """Result of a barge-in vs. backchannel classification while agent is speaking."""

    action: BargeInAction
    p_stop: float
    latency_ms: float
    engine: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Explicit spoken invitations that release Hold Space / True Listening mode immediately
_LISTENING_INVITATION_PATTERNS = re.compile(
    r"(?:\bwhat\s+do\s+you\s+think\b|\bwhat\s+are\s+your\s+thoughts\b|\byour\s+thoughts\b|"
    r"\bhow\s+does\s+that\s+sound\b|\bdo\s+you\s+agree\b|\bcan\s+you\s+help\s+me\b|"
    r"\bgo\s+ahead\b|\bover\s+to\s+you\b|\btell\s+me\s+what\s+you\s+think\b|"
    r"\bwhat'?s\s+your\s+take\b|\brespond\s+to\s+that\b)\s*[.!?]*$",
    re.IGNORECASE,
)


def is_explicit_listening_invitation(text: Optional[str]) -> bool:
    """Return True if the speaker explicitly invited the listening agent to respond."""
    if not text or not text.strip():
        return False
    return bool(_LISTENING_INVITATION_PATTERNS.search(text.strip()))


def score_to_turn_state(
    score: float,
    baseline_silence_s: float = 1.4,
    is_dictation: bool = False,
    hold_space: bool = False,
    partial_text: Optional[str] = None,
) -> Tuple[TurnState, float]:
    """
    Map combined turn completion score [0.0, 1.0] to TurnState and wait seconds.
    Uses the Shortwire / Cue dynamic pause formula:
      - DONE        (score >= 0.65): sentence completed + falling pitch
      - LIKELY_DONE (0.40 <= score < 0.65): likely ending, moderate pause
      - MID_THOUGHT (0.20 <= score < 0.40): mid-sentence pause / filler words
      - HOLDING     (score < 0.20): trailing conjunction, digit sequence, or sustained pitch hold
    When `hold_space=True` (True Listening Mode), contemplative pauses are held for 6.50s
    unless the speaker explicitly invites the agent to respond.
    """
    s = max(0.0, min(1.0, float(score)))
    if s >= 0.65:
        state: TurnState = "DONE"
    elif s >= 0.40:
        state = "LIKELY_DONE"
    elif s >= 0.20:
        state = "MID_THOUGHT"
    else:
        state = "HOLDING"

    base = float(baseline_silence_s)
    if hold_space:
        if is_explicit_listening_invitation(partial_text):
            return "DONE", 0.60
        return "HOLDING", round(max(6.50, base), 2)

    if is_dictation:
        # In dictation mode, the user is live-autotyping directly into the focused field.
        # There is no benefit to cutting off speech prematurely on sentence boundaries or mid-thought pauses.
        # All states hold for a generous 8.0s safety window (or user's baseline if higher),
        # allowing the user to dictate naturally until they press Enter, say "Done", or fall silent for 8s.
        dict_hold = max(8.00, base)
        return state, round(dict_hold, 2)

    # Conversational agent mode:
    # If baseline is the standard 1.4s, return the exact Shortwire TURN_STATE_WAITS.
    # Otherwise, scale proportionally with baseline_silence_s.
    if abs(base - 1.4) < 1e-4:
        return state, TURN_STATE_WAITS[state]

    conv_multipliers = {
        "DONE": 0.50 / 1.40,        # ~0.357
        "LIKELY_DONE": 1.00 / 1.40, # ~0.714
        "MID_THOUGHT": 1.75 / 1.40, # 1.250
        "HOLDING": 2.25 / 1.40,     # ~1.607
    }
    scaled_wait = base * conv_multipliers[state]
    return state, round(max(0.40, scaled_wait), 2)


def find_smart_turn_model(auto_download: bool = False) -> Optional[Path]:
    """
    Locate Pipecat's `smart-turn-v3.2-cpu.onnx` (8.3 MB) model on disk.
    Checks:
      1. ~/.voicefi/models/smart-turn-v3.2-cpu.onnx
      2. ~/.voicefi/assets/smart-turn-v3.2-cpu.onnx
      3. Bundled package assets (src/voicefi/assets/ or PyInstaller frozen bundle)
      4. Local Hugging Face Hub cache (models--pipecat-ai--smart-turn-v3)
      5. Optional download via huggingface_hub if auto_download=True
    """
    candidates = [
        Path.home() / ".voicefi" / "models" / "smart-turn-v3.2-cpu.onnx",
        Path.home() / ".voicefi" / "assets" / "smart-turn-v3.2-cpu.onnx",
        Path(__file__).resolve().parent.parent / "assets" / "smart-turn-v3.2-cpu.onnx",
        Path(__file__).resolve().parent / "assets" / "smart-turn-v3.2-cpu.onnx",
    ]

    if getattr(sys, "frozen", False):
        base_meipass = getattr(sys, "_MEIPASS", None)
        if base_meipass:
            candidates.extend([
                Path(base_meipass) / "voicefi" / "assets" / "smart-turn-v3.2-cpu.onnx",
                Path(base_meipass) / "assets" / "smart-turn-v3.2-cpu.onnx",
                Path(base_meipass) / "smart-turn-v3.2-cpu.onnx",
            ])
        res_dir = Path(sys.executable).parent.parent / "Resources"
        candidates.extend([
            res_dir / "voicefi" / "assets" / "smart-turn-v3.2-cpu.onnx",
            res_dir / "smart-turn-v3.2-cpu.onnx",
        ])

    for c in candidates:
        if c.is_file() and c.stat().st_size > 1_000_000:
            return c

    # Search local Hugging Face cache without network calls
    hf_cache_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--pipecat-ai--smart-turn-v3"
        / "snapshots"
    )
    if hf_cache_dir.is_dir():
        for snap in sorted(hf_cache_dir.iterdir(), reverse=True):
            onnx_file = snap / "smart-turn-v3.2-cpu.onnx"
            if onnx_file.is_file() and onnx_file.stat().st_size > 1_000_000:
                return onnx_file

    if auto_download:
        try:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(
                repo_id="pipecat-ai/smart-turn-v3",
                filename="smart-turn-v3.2-cpu.onnx",
            )
            if downloaded and Path(downloaded).is_file():
                return Path(downloaded)
        except Exception:
            pass

    return None


class SmartTurnONNX:
    """
    Pipecat Smart Turn v3.2 (8.3 MB INT8 ONNX) acoustic turn completion classifier.
    Uses a Whisper-Tiny audio encoder backbone over the last 8.0s of 16kHz audio
    to detect prosodic turn endings (pitch drop, syllable lengthening) vs. mid-thought holds.
    """

    WINDOW_SECONDS = 8
    SAMPLE_RATE = 16000
    WINDOW_SAMPLES = WINDOW_SECONDS * SAMPLE_RATE  # 128,000 samples

    _SESSION_CACHE: Dict[str, Tuple[Any, str, Any]] = {}
    _SESSION_LOCK = threading.Lock()

    def __init__(self, model_path: Optional[Path] = None, auto_download: bool = False):
        self.model_path = model_path or find_smart_turn_model(auto_download=auto_download)
        self._session = None
        self._feature_extractor = None
        self._input_name: str = "input_features"
        self._init_session()

    def _init_session(self) -> None:
        if self.model_path is None or not self.model_path.exists():
            return
        cache_key = str(self.model_path)
        with SmartTurnONNX._SESSION_LOCK:
            cached = SmartTurnONNX._SESSION_CACHE.get(cache_key)
            if cached is not None:
                self._session, self._input_name, self._feature_extractor = cached
                return
            try:
                import onnxruntime as ort

                so = ort.SessionOptions()
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                so.inter_op_num_threads = 1
                so.intra_op_num_threads = 2
                so.enable_cpu_mem_arena = False
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                try:
                    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
                    so.add_session_config_entry("session.inter_op.allow_spinning", "0")
                except Exception:
                    pass

                feat_ext = None
                try:
                    from faster_whisper.feature_extractor import FeatureExtractor
                    feat_ext = FeatureExtractor(chunk_length=self.WINDOW_SECONDS)
                except Exception:
                    pass

                sess = ort.InferenceSession(
                    cache_key,
                    sess_options=so,
                    providers=["CPUExecutionProvider"],
                )
                inputs = sess.get_inputs()
                in_name = inputs[0].name if inputs else "input_features"
                SmartTurnONNX._SESSION_CACHE[cache_key] = (sess, in_name, feat_ext)
                self._session = sess
                self._input_name = in_name
                self._feature_extractor = feat_ext
            except Exception:
                self._session = None
                self._feature_extractor = None

    @property
    def is_available(self) -> bool:
        return self._session is not None and self._feature_extractor is not None

    @classmethod
    def prepare_audio_window(cls, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Format raw audio to 8.0 seconds (128,000 samples @ 16kHz), right-aligned:
        - Truncates to the last 8.0 seconds if longer.
        - Left-pads with zeros if shorter so the current pause sits at the right edge.
        - Normalizes to zero-mean, unit-variance as expected by Whisper FeatureExtractor.
        """
        arr = np.asarray(audio, dtype=np.float32).flatten()
        if arr.size == 0:
            return np.zeros(cls.WINDOW_SAMPLES, dtype=np.float32)

        if sample_rate != cls.SAMPLE_RATE and sample_rate > 0:
            # Fast linear interpolation resample to 16kHz if needed
            duration = arr.size / float(sample_rate)
            target_len = max(1, int(round(duration * cls.SAMPLE_RATE)))
            arr = np.interp(
                np.linspace(0.0, 1.0, target_len, endpoint=False),
                np.linspace(0.0, 1.0, arr.size, endpoint=False),
                arr,
            ).astype(np.float32)

        if arr.size > cls.WINDOW_SAMPLES:
            arr = arr[-cls.WINDOW_SAMPLES :]
        elif arr.size < cls.WINDOW_SAMPLES:
            padding = cls.WINDOW_SAMPLES - arr.size
            arr = np.pad(arr, (padding, 0), mode="constant", constant_values=0.0)

        # Zero-mean, unit-variance normalization
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if std > 1e-6:
            arr = (arr - mean) / std
        else:
            arr = arr - mean
        return arr.astype(np.float32)

    def predict_endpoint_prob(self, audio: np.ndarray, sample_rate: int = 16000) -> float:
        """
        Return acoustic probability p_voice in [0.0, 1.0] that the speaker has finished their turn.
        Returns 0.5 (neutral prior) if model is unavailable or audio is empty.
        """
        if not self.is_available or audio is None:
            return 0.5
        arr = np.asarray(audio, dtype=np.float32).flatten()
        if arr.size < int(sample_rate * 0.15):
            return 0.5

        try:
            aligned = self.prepare_audio_window(arr, sample_rate=sample_rate)
            # Pass padding=0 so FeatureExtractor produces exact (80, 800) instead of 801
            mel = self._feature_extractor(
                aligned,
                padding=0,
                chunk_length=self.WINDOW_SECONDS,
            )
            mel = np.asarray(mel, dtype=np.float32)
            if mel.ndim == 2:
                mel = np.expand_dims(mel, axis=0)
            # Ensure exact 800 frames
            if mel.shape[-1] > 800:
                mel = mel[:, :, :800]
            elif mel.shape[-1] < 800:
                mel = np.pad(mel, ((0, 0), (0, 0), (0, 800 - mel.shape[-1])), mode="constant")

            outputs = self._session.run(None, {self._input_name: mel})
            raw_val = float(np.asarray(outputs[0]).squeeze())
            # smart-turn-v3.2-cpu.onnx already applies sigmoid in its output node,
            # but clamp/handle logits defensively if outside [0.0, 1.0]
            if raw_val < 0.0 or raw_val > 1.0:
                prob = 1.0 / (1.0 + math.exp(-raw_val))
            else:
                prob = raw_val
            return max(0.001, min(0.999, prob))
        except Exception:
            return 0.5


class LexicalTurnScorer:
    """
    Sub-millisecond (0.05 ms) deterministic semantic scorer for end-of-turn and barge-in.
    Catches partial phone numbers, partial IPv4 addresses, trailing conjunctions,
    hesitation fillers, and explicit completion punctuation.
    """

    @staticmethod
    def score_text(text: Optional[str]) -> Tuple[float, str, bool]:
        """
        Evaluate partial transcript text.
        Returns: (p_end, reason, is_definitive)
        where `is_definitive=True` means the lexical pattern is unambiguous and can
        short-circuit the LLM call if desired.
        """
        if not text or not text.strip():
            return 0.50, "empty_transcript", False

        raw = text.strip()
        lower = raw.lower()
        # Strip trailing punctuation for token analysis while preserving raw punctuation check
        ends_with_question = raw.endswith("?")
        ends_with_exclaim = raw.endswith("!")
        ends_with_period = raw.endswith(".") and not raw.endswith("...")
        ends_with_ellipsis = raw.endswith("...") or raw.endswith("…") or raw.endswith("—") or raw.endswith("-")
        ends_with_comma = raw.endswith(",") or raw.endswith(":") or raw.endswith(";")

        words = re.findall(r"[a-z0-9']+", lower)
        if not words:
            return 0.50, "no_lexical_tokens", False

        clean_phrase = " ".join(words)
        last_word = words[-1]

        # 1. Explicit standalone complete short phrases ("proceed", "ship it", "yes", "stop")
        if clean_phrase in _STANDALONE_COMPLETE_PHRASES and not ends_with_ellipsis and not ends_with_comma:
            return 0.92, "standalone_command", True

        # 2. Trailing ellipsis, dash, comma, or colon -> speaker is mid-clause
        if ends_with_ellipsis:
            return 0.10, "trailing_ellipsis", True
        if ends_with_comma:
            return 0.18, "trailing_comma_or_colon", False

        # 3. Check for partial IPv4 address (e.g. "192.168.1" or "10.0.")
        if re.search(r"\b(?:\d{1,3}\.){1,2}\d{0,3}$", raw) or re.search(
            r"\b(?:\d{1,3}\.){3}$", raw
        ):
            return 0.08, "partial_ip_address", True

        # 4. Check for partial phone numbers / digit sequences (2 to 9 digits spoken or numeric)
        digit_chars = re.findall(r"\d", lower)
        spoken_digit_count = sum(1 for w in words if w in _SPOKEN_DIGITS)
        total_digits = len(digit_chars) + spoken_digit_count
        # If the tail of the utterance is a digit or spoken number word and total digits is 2..9
        if (last_word in _SPOKEN_DIGITS or last_word.isdigit()) and (2 <= total_digits <= 9):
            return 0.08, f"partial_digit_sequence_{total_digits}", True


        # 5. Check multi-word trailing hold phrases ("let me think", "port is", "going to")
        for phrase in _TRAILING_HOLD_PHRASES:
            if clean_phrase.endswith(phrase):
                return 0.12, f"trailing_phrase:{phrase.replace(' ', '_')}", True

        # 6. Check trailing continuation word ("and", "but", "the", "to", "um", "port")
        if last_word in _TRAILING_CONTINUATION_WORDS:
            return 0.10, f"trailing_continuation:{last_word}", True

        # 7. Questions and exclamations with >= 2 words and no trailing continuation
        if (ends_with_question or ends_with_exclaim) and len(words) >= 2:
            return 0.90, "terminal_question_or_exclamation", False

        # 8. Declarative sentence with period and >= 3 words
        if ends_with_period and len(words) >= 3:
            return 0.82, "declarative_sentence", False

        # 9. Unpunctuated phrase of >= 4 words not ending in a continuation word
        if len(words) >= 4:
            return 0.68, "unpunctuated_clause", False

        # Short 1-3 word fragment without terminal punctuation
        return 0.38, "short_fragment", False

    @staticmethod
    def score_barge_in(text: Optional[str], duration_s: float = 0.0) -> Tuple[float, str, bool]:
        """
        Classify whether user speech during AI playback is a true interruption (STOP)
        or passive backchanneling (CONTINUE).
        Returns: (p_stop, reason, is_definitive)
        """
        # Sustained speech over 1.1s is always a real interruption
        if duration_s >= 1.1:
            return 0.95, "sustained_speech_duration", True

        if not text or not text.strip():
            # Very short acoustic blip (<0.30s) with no words
            if duration_s < 0.30:
                return 0.25, "brief_untranscribed_sound", False
            return 0.65, "untranscribed_voice_activity", False

        clean = re.sub(r"[^\w\s'-]", "", text.strip().lower())
        clean = " ".join(clean.split())
        words = clean.split()

        if not words:
            return 0.30, "no_words", False

        # Exact match on passive backchannel ("uh-huh", "mm-hmm", "okay", "yeah", "got it")
        if clean in _BACKCHANNEL_CONTINUE_PHRASES and len(words) <= 2:
            return 0.08, f"backchannel:{clean.replace(' ', '_')}", True

        # Explicit interruption keyword ("wait", "stop", "hold on", "actually", "no")
        for w in words:
            if w in _BARGE_IN_STOP_WORDS:
                return 0.96, f"interrupt_keyword:{w}", True
        for stop_phrase in ("hold on", "hang on", "never mind", "excuse me"):
            if stop_phrase in clean:
                return 0.96, f"interrupt_phrase:{stop_phrase.replace(' ', '_')}", True

        # Multi-word substantive statement (>= 3 words) -> real barge-in
        if len(words) >= 3:
            return 0.90, "substantive_multiword_interruption", True

        return 0.55, "ambiguous_short_utterance", False


class OllamaLogitScorer:
    """
    1-Token Logit Turn & Barge-In Classifier using local Ollama GGUF models
    (`qwen2.5-coder:1.5b` default).
    Reads top-5 logprobs from `/v1/chat/completions` with `max_tokens=1`,
    `temperature=0.0`, and `reasoning_effort="none"` in ~30 ms.
    """

    TURN_SYSTEM_PROMPT = (
        "You are a real-time speech turn detector. Decide if the speaker's transcript is "
        "a complete turn (DONE) or if they are pausing mid-thought / mid-number / mid-sentence (WAIT).\n"
        "Examples:\n"
        'Transcript: "Can you open the config file?" -> DONE\n'
        'Transcript: "I want to refactor the" -> WAIT\n'
        'Transcript: "My number is four one five, eight six" -> WAIT\n'
        'Transcript: "Connect to 192.168.1" -> WAIT\n'
        'Transcript: "Run pytest on the audio module." -> DONE\n'
        'Transcript: "Wait, let me think about" -> WAIT\n'
        "Respond with exactly one word: DONE or WAIT."
    )

    BARGE_IN_SYSTEM_PROMPT = (
        "You are a full-duplex voice barge-in filter. While the AI assistant is speaking, "
        "the user made a sound. Output CONTINUE if it is a passive backchannel acknowledgement "
        '(like "uh-huh", "mm-hmm", "okay", "yeah", "got it", "right"). Output STOP if the user '
        'wants to interrupt, correct, or ask something (like "wait", "stop", "hold on", "no, check main.py").\n'
        "Respond with exactly one word: STOP or CONTINUE."
    )

    def __init__(
        self,
        model: str = "qwen2.5-coder:1.5b",
        base_url: Optional[str] = None,
        timeout_s: float = 0.18,
    ):
        self.model = model
        self.base_url = (
            base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        ).rstrip("/")
        self.timeout_s = timeout_s
        self._last_health_check: float = 0.0
        self._is_reachable: bool = False

    def _check_available(self) -> bool:
        """Fast cached check for thermal health and Ollama availability."""
        # Never make live network calls inside unit tests unless explicitly mocked/forced
        if os.environ.get("PYTEST_CURRENT_TEST") and not os.environ.get(
            "VOICEFI_ALLOW_OLLAMA_IN_TESTS"
        ):
            return False

        now = time.monotonic()
        if (now - self._last_health_check) < 15.0:
            return self._is_reachable

        self._last_health_check = now
        try:
            from whole_body_compute.local.supervisor import ThermalSupervisor

            health = ThermalSupervisor.check_health()
            if not health.get("healthy", True):
                self._is_reachable = False
                return False
        except Exception:
            pass

        try:
            import httpx

            resp = httpx.get(f"{self.base_url}/api/tags", timeout=0.12)
            if resp.status_code == 200:
                models = [m.get("name", "") for m in resp.json().get("models", [])]
                # Check if requested model (or base prefix) is present
                base_name = self.model.split(":")[0]
                self._is_reachable = any(
                    m == self.model or m.startswith(base_name) for m in models
                )
            else:
                self._is_reachable = False
        except Exception:
            self._is_reachable = False

        return self._is_reachable

    def _query_binary_logprobs(
        self,
        system_prompt: str,
        user_prompt: str,
        positive_Prefix: str,
        negative_Prefix: str,
    ) -> Optional[float]:
        """
        Send a 1-token chat completion request and compute exact softmax probability:
            p(positive) = exp(lp_pos) / (exp(lp_pos) + exp(lp_neg))
        """
        if not self._check_available():
            return None

        try:
            import httpx

            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": 5,
            }
            # Qwen3.5 / Tev1 models support reasoning_effort="none" to skip <think> tokens
            if "tev" in self.model.lower() or "qwen3" in self.model.lower():
                payload["reasoning_effort"] = "none"

            resp = httpx.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout_s,
            )
            if resp.status_code != 200:
                return None

            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content", "").strip().upper()
            lp_items = ((choice.get("logprobs") or {}).get("content") or [{}])[0].get(
                "top_logprobs", []
            )

            pos_lp: Optional[float] = None
            neg_lp: Optional[float] = None
            pos_key = positive_Prefix.upper()
            neg_key = negative_Prefix.upper()

            for item in lp_items:
                tok = str(item.get("token", "")).strip().upper()
                lp = float(item.get("logprob", -20.0))
                if tok.startswith(pos_key) and pos_lp is None:
                    pos_lp = lp
                elif tok.startswith(neg_key) and neg_lp is None:
                    neg_lp = lp

            if pos_lp is not None and neg_lp is not None:
                max_lp = max(pos_lp, neg_lp)
                p_pos = math.exp(pos_lp - max_lp)
                p_neg = math.exp(neg_lp - max_lp)
                return float(p_pos / (p_pos + p_neg))
            elif pos_lp is not None:
                return float(min(0.99, math.exp(pos_lp)))
            elif neg_lp is not None:
                return float(max(0.01, 1.0 - math.exp(neg_lp)))

            # Fallback if logprobs weren't populated but token text was returned
            if content.startswith(pos_key):
                return 0.85
            if content.startswith(neg_key):
                return 0.15
            return None
        except Exception:
            return None

    def score_turn_end(self, transcript: str) -> Optional[float]:
        """Return p(DONE) in [0.0, 1.0] from 1-token logits, or None if unavailable."""
        if not transcript or not transcript.strip():
            return None
        return self._query_binary_logprobs(
            system_prompt=self.TURN_SYSTEM_PROMPT,
            user_prompt=f'Transcript: "{transcript.strip()}"',
            positive_Prefix="DONE",
            negative_Prefix="WAIT",
        )

    def score_barge_in(self, transcript: str, ai_context: str = "") -> Optional[float]:
        """Return p(STOP) in [0.0, 1.0] from 1-token logits, or None if unavailable."""
        if not transcript or not transcript.strip():
            return None
        prompt = f'User said: "{transcript.strip()}"'
        if ai_context:
            prompt = f'AI was saying: "{ai_context.strip()}"\n{prompt}'
        return self._query_binary_logprobs(
            system_prompt=self.BARGE_IN_SYSTEM_PROMPT,
            user_prompt=prompt,
            positive_Prefix="STOP",
            negative_Prefix="CONTINUE",
        )


class CueTurnCascade:
    """
    Unified Shortwire / Cue Turn-Detection Cascade for VoiceFi.
    Combines:
      1. `SmartTurnONNX` (8.3 MB acoustic prosody & intonation scorer)
      2. `LexicalTurnScorer` (0.05 ms structural & digit-sequence rules)
      3. `OllamaLogitScorer` (`qwen2.5-coder:1.5b` 1-token logit scorer)
    using the Shortwire fusion formula:
      `score = p_end * sqrt(p_voice)`
    """

    def __init__(
        self,
        mode: Literal["auto", "cue", "lexical", "fixed"] = "auto",
        turn_model: str = "qwen2.5-coder:1.5b",
        smart_turn_enabled: bool = True,
        smart_turn_model_path: Optional[Path] = None,
    ):
        self.mode = mode
        self.smart_turn_enabled = smart_turn_enabled
        self.smart_turn = (
            SmartTurnONNX(model_path=smart_turn_model_path) if smart_turn_enabled else None
        )
        self.logit_scorer = (
            OllamaLogitScorer(model=turn_model) if mode in ("auto", "cue") else None
        )

    def evaluate_turn(
        self,
        audio: Optional[np.ndarray] = None,
        partial_text: Optional[str] = None,
        sample_rate: int = 16000,
        baseline_silence_s: float = 1.4,
        is_dictation: bool = False,
        hold_space: bool = False,
    ) -> TurnDecision:
        """
        Evaluate whether the user has finished their turn at the 350ms pause onset mark.
        Returns a `TurnDecision` with dynamic `wait_seconds` and diagnostic trace metadata.
        """
        t0 = time.perf_counter()

        if self.mode == "fixed":
            return TurnDecision(
                state="LIKELY_DONE",
                wait_seconds=float(baseline_silence_s),
                score=0.50,
                p_end=0.50,
                p_voice=0.50,
                latency_ms=0.0,
                engine="fixed",
                reason="fixed_silence_duration",
            )

        # Step 1: Acoustic Prosody Score (Smart Turn v3.2 ONNX + Direct Waveform Prosody, ~9-12 ms)
        has_smart_turn = bool(self.smart_turn and self.smart_turn.is_available and audio is not None)
        p_voice = (
            self.smart_turn.predict_endpoint_prob(audio, sample_rate=sample_rate)
            if has_smart_turn
            else 0.50
        )

        # Direct waveform prosody extraction (ΔF0 pitch slope, breath intake, vowel elongation)
        prosody_reason: Optional[str] = None
        has_direct_prosody = False
        if (
            self.smart_turn_enabled
            and audio is not None
            and hasattr(audio, "__len__")
            and len(audio) >= int(max(1, sample_rate) * 0.25)
        ):
            try:
                from voicefi.audio.decision_model import default_audio_decision_model

                prosody = default_audio_decision_model.extract_prosody_features(
                    audio, sample_rate=sample_rate
                )
                is_question_txt = bool(partial_text and str(partial_text).strip().endswith("?"))
                # 1. Pre-pause breath inhalation -> speaker is taking a breath to continue (HOLDING)
                if prosody.breath_intake_score >= 0.35:
                    p_voice = min(p_voice, 0.16)
                    prosody_reason = f"prosody_breath_intake({prosody.breath_intake_score:.2f})"
                    has_direct_prosody = True
                # 2. Rising terminal pitch + sustained vowel drawl (non-question) -> mid-thought hold
                elif (
                    prosody.f0_slope_hz_per_sec > 25.0
                    and prosody.vowel_elongation_ratio >= 0.45
                    and not is_question_txt
                ):
                    p_voice = min(p_voice, 0.18)
                    prosody_reason = (
                        f"prosody_rising_pitch_drawl(+{prosody.f0_slope_hz_per_sec:.0f}Hz/s)"
                    )
                    has_direct_prosody = True
                # 3. Falling declarative pitch + natural energy decay on short/unpadded clips
                elif (
                    prosody.f0_slope_hz_per_sec < -22.0
                    and prosody.terminal_energy_slope_db <= -2.0
                    and prosody.duration_ms >= 350.0
                ):
                    p_voice = max(p_voice, 0.78)
                    prosody_reason = (
                        f"prosody_falling_pitch_complete({prosody.f0_slope_hz_per_sec:.0f}Hz/s)"
                    )
                    has_direct_prosody = True
            except Exception:
                pass

        # Step 2: Semantic Text Score (Lexical + optional 1-token Ollama Logits)
        has_text = bool(partial_text and partial_text.strip())
        if has_text:
            lex_p_end, lex_reason, is_definitive = LexicalTurnScorer.score_text(partial_text)
            llm_p_end: Optional[float] = None

            # Query 1-token LLM logits when lexical rules aren't already an unambiguous hold
            if not is_definitive and self.logit_scorer is not None:
                llm_p_end = self.logit_scorer.score_turn_end(partial_text)

            if llm_p_end is not None:
                # Blend LLM 1-token probability (70%) with lexical structural prior (30%)
                p_end = 0.70 * llm_p_end + 0.30 * lex_p_end
                text_engine = "qwen_1tok"
                reason = f"{lex_reason}+llm({llm_p_end:.2f})"
            else:
                p_end = lex_p_end
                text_engine = "lexical"
                reason = lex_reason

            if has_smart_turn or has_direct_prosody:
                # Shortwire fusion formula: p_end * sqrt(p_voice)
                score = float(p_end * math.sqrt(max(0.01, p_voice)))
                engine = f"{text_engine}+smart_turn" if has_smart_turn else f"{text_engine}+prosody"
                if prosody_reason:
                    reason = f"{reason}+{prosody_reason}"
            else:
                score = float(p_end)
                engine = text_engine
        else:
            # No partial transcript available yet -> rely on Smart Turn v3.2 + Direct Waveform Prosody
            if has_smart_turn or has_direct_prosody:
                audio_dur_s = (
                    (len(audio) / max(1, sample_rate))
                    if audio is not None and hasattr(audio, "__len__")
                    else 0.0
                )
                # Guard: Smart Turn ONNX right-aligns into an 8.0s window; if total audio (including
                # pre-roll + silence) is <1.1s with zero transcribed words, the 8s window is ~90% padded
                # silence and ONNX can emit a high false-positive p_voice (~0.98). Clamp unless direct
                # waveform prosody confirmed a real falling declarative pitch contour.
                if 0.0 < audio_dur_s < 1.10 and not (
                    prosody_reason and "falling_pitch_complete" in prosody_reason
                ):
                    p_voice = min(p_voice, 0.30)
                p_end = p_voice
                score = p_voice
                engine = "smart_turn_only" if has_smart_turn else "prosody_only"
                if prosody_reason:
                    reason = prosody_reason
                elif p_voice >= 0.65:
                    reason = "acoustic_pitch_drop_complete"
                elif p_voice < 0.25:
                    reason = "acoustic_sustained_pitch_hold"
                else:
                    reason = "acoustic_neutral_prosody"
            else:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return TurnDecision(
                    state="LIKELY_DONE",
                    wait_seconds=float(baseline_silence_s),
                    score=0.50,
                    p_end=0.50,
                    p_voice=0.50,
                    latency_ms=round(latency_ms, 2),
                    engine="fallback_baseline",
                    reason="no_text_or_acoustic_model",
                )

        state, wait_s = score_to_turn_state(
            score,
            baseline_silence_s=baseline_silence_s,
            is_dictation=is_dictation,
            hold_space=hold_space,
            partial_text=partial_text,
        )

        # If user configured an intentionally long Fibonacci pause delay (e.g. >= 5.0s for brain dumps),
        # honor their minimum floor while still allowing Thought Hold extensions.
        if baseline_silence_s >= 5.0:
            wait_s = max(wait_s, float(baseline_silence_s))

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return TurnDecision(
            state=state,
            wait_seconds=wait_s,
            score=round(score, 4),
            p_end=round(p_end, 4),
            p_voice=round(p_voice, 4),
            latency_ms=round(latency_ms, 2),
            engine=engine,
            reason=reason,
        )

    def classify_barge_in(
        self,
        partial_text: Optional[str] = None,
        speech_duration_s: float = 0.0,
        ai_context: str = "",
    ) -> BargeInDecision:
        """
        Classify whether user vocal activity over AI speech is an active interruption (`STOP`)
        or a passive backchannel (`CONTINUE` like "mm-hmm", "okay", "yeah").
        """
        t0 = time.perf_counter()
        lex_p_stop, lex_reason, is_definitive = LexicalTurnScorer.score_barge_in(
            partial_text, duration_s=speech_duration_s
        )

        p_stop = lex_p_stop
        engine = "lexical"
        reason = lex_reason

        if (
            not is_definitive
            and partial_text
            and partial_text.strip()
            and self.logit_scorer is not None
        ):
            llm_p_stop = self.logit_scorer.score_barge_in(partial_text, ai_context=ai_context)
            if llm_p_stop is not None:
                p_stop = 0.75 * llm_p_stop + 0.25 * lex_p_stop
                engine = "qwen_1tok"
                reason = f"{lex_reason}+llm({llm_p_stop:.2f})"

        action: BargeInAction = "STOP" if p_stop >= 0.50 else "CONTINUE"
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return BargeInDecision(
            action=action,
            p_stop=round(p_stop, 4),
            latency_ms=round(latency_ms, 2),
            engine=engine,
            reason=reason,
        )

    @staticmethod
    def is_false_start(
        speech_duration_s: float,
        partial_text: Optional[str],
        p_voice: float,
    ) -> bool:
        """
        Detect whether an initial brief Turn-On latch (<=250ms) followed by silence
        was just a cough, sigh, chime tail, mic bump, or lip smack with zero spoken words,
        allowing `AudioRecorder` to reset back to idle waiting instead of firing an empty turn.
        """
        if speech_duration_s > 0.25:
            return False
        if partial_text and partial_text.strip():
            return False
        # Any <=250ms sound burst with zero transcribed words is a false start regardless of ONNX padding artifacts
        return True
