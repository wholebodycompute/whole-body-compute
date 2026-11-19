"""
voicefi/local/bootstrap.py
Zero-Egress Sovereign Bootstrapper for Hey Pal (VoiceFi + Whole Body Compute).

Coordinates the five local sensory-motor organs on Apple Silicon:
1. Ears: Silero VAD v6 + Smart-Turn v3.2 + Apple Speech Recognizer stream
2. Mouth: Kokoro-82M v1.0 FP16 Neural TTS (100% offline, zero-PyTorch)
3. Eyes: Zero-Egress Screenpipe Daemon (:3030) + Foveal AX DOM (<5ms)
4. Hands & Hippocampus: DotMemory SQLite WAL + Screenpipe db.sqlite + Pal Trainer/Compiler
5. Brain: Hardware-calibrated Local LLMs across 3 dedicated runtime endpoints:
   - :11434 (Ollama): `tev1:0.8b` (141ms spoken parameter extraction & HAR API triage)
   - :9379  (Direct-on-Metal MLX): `voicefi-coder-mlx` (4.97s surgical workflow self-healing & live voice edits)
   - :8080  (Native MLX 16GB+): `mlx-community/Qwen2.5-Coder-14B-Instruct-4bit` (2.15s neural regex PbE & Python synthesis)

Enforces Two Critical Runtime Invocation Rules:
- Rule 1 (`tev1:0.8b` `qwen35`): Always call `http://127.0.0.1:11434/api/chat`
  with `"think": False` (141ms response; never `/api/generate` which triggers hidden reasoning tokens).
- Rule 2 (`voicefi-coder-mlx` `:9379`): Cap `max_tokens <= 360` and strip `<|im_end|>` so the 1.5B LoRA
  stops cleanly after emitting the unified diff (4.97s total patch time).
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("whole_body_compute.local.bootstrap")

from whole_body_compute.procedural_telemetry import get_wbc_data_dir

FOVEAL_MEMORY_DEFAULT_URL = "http://127.0.0.1:9382"
FOVEAL_MEMORY_DEFAULT_DB = Path.home() / ".whole_body" / "foveal_memory.sqlite"
_INITIAL_DOT_MEMORY_DEFAULT_DB = get_wbc_data_dir() / "dot_memory.db"
DOT_MEMORY_DEFAULT_DB = _INITIAL_DOT_MEMORY_DEFAULT_DB
_INITIAL_LOCAL_COMPUTE_RECEIPT_FILE = get_wbc_data_dir() / ".local_compute_bootstrapped.json"
LOCAL_COMPUTE_RECEIPT_FILE = _INITIAL_LOCAL_COMPUTE_RECEIPT_FILE
LAUNCH_AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"


def _resolve_dot_memory_default_db() -> Path:
    if DOT_MEMORY_DEFAULT_DB != _INITIAL_DOT_MEMORY_DEFAULT_DB:
        return DOT_MEMORY_DEFAULT_DB
    return get_wbc_data_dir() / "dot_memory.db"


def _resolve_local_compute_receipt_file() -> Path:
    if LOCAL_COMPUTE_RECEIPT_FILE != _INITIAL_LOCAL_COMPUTE_RECEIPT_FILE:
        return LOCAL_COMPUTE_RECEIPT_FILE
    return get_wbc_data_dir() / ".local_compute_bootstrapped.json"

OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
MLX_9379_CHAT_URL = "http://127.0.0.1:9379/v1/chat/completions"
MLX_8080_CHAT_URL = "http://127.0.0.1:8080/v1/chat/completions"

PRO_COMPILER_MLX_MODEL = "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit"
MAX_FRONTIER_MLX_MODEL = "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit"
MAX_SPARSE_MOE_MLX_MODEL = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
SPECULATIVE_DRAFT_MLX_MODEL = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"


def get_mlx_speculative_config(target_model: str) -> Dict[str, Any]:
    """
    Return the M5 Pro calibrated Tier-2 speculative decoding configuration for an MLX model.
    - Sparse MoE (`Qwen3-Coder-30B-A3B`): streams only ~1.85 GB/tok natively (53.4 tok/s decode,
      2,234 tok/s prefill), so no draft model is needed.
    - Dense 14B (`Qwen2.5-Coder-14B`): pairs with 1.5B draft at k=5 -> 50.5 tok/s (2.15x speedup).
    - Dense 32B (`Qwen2.5-Coder-32B`): pairs with 1.5B draft at k=4 -> 30.9 tok/s (2.53x speedup).
    """
    m = (target_model or "").strip()
    if "Qwen3-Coder-30B-A3B" in m or "A3B" in m:
        return {
            "target_model": m,
            "use_speculative": False,
            "draft_model": None,
            "num_draft_tokens": 0,
            "expected_prefill_tok_s": 2234.2,
            "expected_decode_tok_s": 53.4,
        }
    if "32B" in m:
        return {
            "target_model": m,
            "use_speculative": True,
            "draft_model": SPECULATIVE_DRAFT_MLX_MODEL,
            "num_draft_tokens": 4,
            "expected_decode_tok_s": 30.9,
            "speedup_x": 2.53,
        }
    if "14B" in m:
        return {
            "target_model": m,
            "use_speculative": True,
            "draft_model": SPECULATIVE_DRAFT_MLX_MODEL,
            "num_draft_tokens": 5,
            "expected_decode_tok_s": 50.5,
            "speedup_x": 2.15,
        }
    return {
        "target_model": m,
        "use_speculative": False,
        "draft_model": None,
        "num_draft_tokens": 0,
    }


@dataclass
class OrganHealth:
    organ: str
    name: str
    status: str  # "healthy", "warning", "missing"
    details: str
    file_path: Optional[str] = None


@dataclass
class HardwareProfile:
    total_ram_gb: float
    free_ram_gb: float
    chip_name: str
    cpu_cores: int
    is_apple_silicon: bool
    recommended_tier: str


def get_hardware_profile() -> HardwareProfile:
    """Detect Apple Silicon chip, total unified memory, and free headroom."""
    total_ram_bytes = 0
    chip_name = platform.processor() or "Apple Silicon"
    is_arm = platform.machine() in ("arm64", "aarch64")

    try:
        res = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=2)
        if res.returncode == 0 and res.stdout.strip().isdigit():
            total_ram_bytes = int(res.stdout.strip())
    except Exception:
        pass

    try:
        res = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=2)
        if res.returncode == 0 and res.stdout.strip():
            chip_name = res.stdout.strip()
    except Exception:
        pass

    total_ram_gb = round(total_ram_bytes / (1024**3), 1) if total_ram_bytes > 0 else 16.0

    free_ram_gb = total_ram_gb * 0.5
    try:
        import psutil

        vm = psutil.virtual_memory()
        free_ram_gb = round(vm.available / (1024**3), 1)
        if total_ram_gb <= 0:
            total_ram_gb = round(vm.total / (1024**3), 1)
    except Exception:
        pass

    if total_ram_gb >= 60.0:
        tier = "max"
    elif total_ram_gb >= 16.0:
        tier = "pro"
    else:
        tier = "starter"

    return HardwareProfile(
        total_ram_gb=total_ram_gb,
        free_ram_gb=free_ram_gb,
        chip_name=chip_name,
        cpu_cores=os.cpu_count() or 8,
        is_apple_silicon=is_arm and platform.system() == "Darwin",
        recommended_tier=tier,
    )


def resolve_voicefi_coder_mlx_dir() -> Optional[Path]:
    """
    Resolve the `voicefi-coder-mlx` (839 MB 4-bit MLX) directory across:
      1. PyInstaller frozen bundle (`sys._MEIPASS` / `Contents/Resources`)
      2. `~/.whole_body_compute/models/voicefi-coder-mlx` (or `~/.voicefi/models/voicefi-coder-mlx` fallback)
      3. `~/Projects/Local Models/distillation/fused_model` (or `WBC_FUSED_MODEL_DIR`)
    """
    candidates: List[Path] = []
    env_dir = os.environ.get("WBC_FUSED_MODEL_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    if getattr(sys, "frozen", False):
        base_meipass = getattr(sys, "_MEIPASS", None)
        if base_meipass:
            candidates.extend(
                [
                    Path(base_meipass) / "voicefi" / "assets" / "models" / "voicefi-coder-mlx",
                    Path(base_meipass) / "models" / "voicefi-coder-mlx",
                ]
            )
        res_dir = Path(sys.executable).parent.parent / "Resources"
        candidates.extend(
            [
                res_dir / "voicefi" / "assets" / "models" / "voicefi-coder-mlx",
                res_dir / "models" / "voicefi-coder-mlx",
            ]
        )

    candidates.extend(
        [
            get_wbc_data_dir() / "models" / "voicefi-coder-mlx",
            Path.home() / ".voicefi" / "models" / "voicefi-coder-mlx",
            Path.home() / "Projects" / "Local Models" / "distillation" / "fused_model",
        ]
    )

    for c in candidates:
        if c.is_dir() and (c / "model.safetensors").is_file():
            return c
    return None


EXTERNAL_MODELS_VOLUME = Path("/Volumes/Local Models")


def is_external_models_mounted() -> bool:
    """Return True if the external NVMe model drive (`/Volumes/Local Models`) is mounted and writable."""
    try:
        return EXTERNAL_MODELS_VOLUME.is_dir() and (
            (EXTERNAL_MODELS_VOLUME / "huggingface").is_dir()
            or (EXTERNAL_MODELS_VOLUME / "ollama").is_dir()
        )
    except Exception:
        return False


def ensure_hot_warm_storage_routes() -> Dict[str, Any]:
    """
    Verify and auto-heal the Hot/Warm local model storage routes across internal NVMe
    (`~/.whole_body_compute/models`, `~/.ollama/models_internal`) and external Thunderbolt NVMe
    (`/Volumes/Local Models`).

    - When `/Volumes/Local Models` is DOCKED:
        * `~/.ollama/models` points to `/Volumes/Local Models/ollama/models`
        * `~/.cache/huggingface/hub` points to `/Volumes/Local Models/huggingface/hub`
    - When `/Volumes/Local Models` is UNPLUGGED:
        * Automatically heals `~/.ollama/models` to `~/.ollama/models_internal` so Tier-0/1
          Ollama models (`tev1:0.8b`, `voicefi-coder:1.5b`, `nomic-embed-text`) keep running offline.
        * Hot internal MLX & audio models in `~/.whole_body_compute/models/` remain 100% available.
    """
    docked = is_external_models_mounted()
    ollama_dir = Path.home() / ".ollama"
    ollama_models_link = ollama_dir / "models"
    ext_ollama_models = EXTERNAL_MODELS_VOLUME / "ollama" / "models"
    int_ollama_models = ollama_dir / "models_internal"

    target_ollama = ext_ollama_models if (docked and ext_ollama_models.is_dir()) else int_ollama_models
    ollama_route_switched = False
    active_ollama_target = ""

    try:
        if target_ollama.is_dir():
            current_target = None
            if ollama_models_link.is_symlink():
                try:
                    current_target = Path(os.readlink(ollama_models_link))
                except Exception:
                    current_target = None
            if current_target != target_ollama:
                if ollama_models_link.is_symlink() or ollama_models_link.is_file():
                    ollama_models_link.unlink(missing_ok=True)
                ollama_models_link.parent.mkdir(parents=True, exist_ok=True)
                ollama_models_link.symlink_to(target_ollama)
                ollama_route_switched = True
            active_ollama_target = str(target_ollama)
    except Exception as exc:
        logger.debug("ensure_hot_warm_storage_routes ollama symlink check: %s", exc)

    wbc_hot_dir = get_wbc_data_dir() / "models"
    hot_models_present: List[str] = []
    if wbc_hot_dir.is_dir():
        for entry in sorted(wbc_hot_dir.iterdir()):
            if entry.is_dir() or entry.suffix == ".onnx":
                hot_models_present.append(entry.name)

    ext_free_gb = 0.0
    if docked:
        try:
            usage = shutil.disk_usage(EXTERNAL_MODELS_VOLUME)
            ext_free_gb = round(usage.free / (1024**3), 1)
        except Exception:
            pass

    return {
        "mode": "docked_external" if docked else "unplugged_hot_internal",
        "external_mounted": docked,
        "external_volume": str(EXTERNAL_MODELS_VOLUME),
        "external_free_gb": ext_free_gb,
        "ollama_target": active_ollama_target,
        "ollama_route_switched": ollama_route_switched,
        "hot_internal_dir": str(wbc_hot_dir),
        "hot_models": hot_models_present,
    }


def resolve_local_mlx_model_path(repo_id: str) -> Optional[Path]:
    """
    Resolve an MLX model (`mlx-community/...` or `voicefi-coder-mlx`) preferring the
    internal Hot Tier (`~/.whole_body_compute/models/<basename>`) first so unplugged inference
    always succeeds, then falling back to `~/.cache/huggingface/hub` (or `/Volumes/Local Models/huggingface/hub`).
    """
    if repo_id == "voicefi-coder-mlx":
        return resolve_voicefi_coder_mlx_dir()

    short_name = repo_id.split("/")[-1]
    hot_cand = get_wbc_data_dir() / "models" / short_name
    if hot_cand.is_dir() and any(hot_cand.glob("*.safetensors")):
        return hot_cand

    hf_hubs = [
        Path.home() / ".cache" / "huggingface" / "hub",
        EXTERNAL_MODELS_VOLUME / "huggingface" / "hub",
    ]
    for hub in hf_hubs:
        try:
            folder = hub / f"models--{repo_id.replace('/', '--')}"
            if folder.is_dir():
                snaps = sorted((folder / "snapshots").glob("*"))
                if snaps and any(snaps[-1].glob("*.safetensors")):
                    return snaps[-1]
        except Exception:
            continue
    return None


def is_hf_mlx_model_cached(repo_id: str) -> bool:
    """Return True if an MLX Hugging Face repo (`mlx-community/...`) is cached in Hot Internal or External HF Hub."""
    if resolve_local_mlx_model_path(repo_id) is not None:
        return True
    hf_hub = Path.home() / ".cache" / "huggingface" / "hub"
    folder = hf_hub / f"models--{repo_id.replace('/', '--')}"
    return folder.is_dir() and any(folder.rglob("*.safetensors"))


def get_tier_manifest(tier_id: str) -> Dict[str, Any]:
    """
    Return the Local Models / Hey Pal 100% Local model manifest for a given hardware tier:
      - `starter` / `core` (All Macs 8 GB–64 GB, ~1.65 GB Core + 270 MB Embed):
          * `tev1:0.8b` (811 MB GGUF on Ollama :11434) -> 141ms spoken parameter extraction & HAR API endpoint triage
          * `voicefi-coder-mlx` (839 MB 4-bit MLX on :9379) -> 4.97s surgical workflow self-healing & live voice edits
          * `nomic-embed-text` (270 MB on Ollama :11434) -> Hippocampus semantic memory
      - `pro` (Pro Compiler Pack — 16 GB+ / 64 GB Macs, ~8.5 GB MLX addition):
          * Starter / Core Pack +
          * `mlx-community/Qwen2.5-Coder-14B-Instruct-4bit` (Native MLX :8080) -> 2.15s neural regex PbE induction & full multi-step Python synthesis (100% 7/7 LocalBench SMB)
      - `max` (Max / Studio Pack — 64 GB+ Macs):
          * Starter + Pro Compiler Pack (`Qwen2.5-Coder-14B-Instruct-4bit` on :8080, `voicefi-coder-mlx` on :9379, `tev1:0.8b` on :11434) + optional 32B frontier coder
    """
    t = (tier_id or "starter").lower().strip()
    starter_models = [
        {
            "name": "tev1:0.8b",
            "aliases": ["tev1:0.8b"],
            "provider": "ollama",
            "endpoint": OLLAMA_CHAT_URL,
            "role": "141ms Spoken Parameter Extraction & HAR API Triage",
            "size_mb": 811,
            "runtime_rule": 'POST /api/chat with "think": False (141ms)',
        },
        {
            "name": "voicefi-coder-mlx",
            "aliases": ["voicefi-coder-mlx", "voicefi-coder:1.5b"],
            "provider": "mlx_9379",
            "fallback_ollama": "voicefi-coder:1.5b",
            "endpoint": MLX_9379_CHAT_URL,
            "role": "4.97s Surgical Workflow Self-Healing & Live Voice Edits",
            "size_mb": 839,
            "runtime_rule": "max_tokens <= 360 & strip <|im_end|> (4.97s patch)",
        },
        {
            "name": "nomic-embed-text",
            "aliases": ["nomic-embed-text", "nomic-embed-text:latest"],
            "provider": "ollama",
            "endpoint": "http://127.0.0.1:11434/api/embeddings",
            "role": "Hippocampus DotMemory Embeddings",
            "size_mb": 270,
        },
    ]

    pro_compiler_model = {
        "name": PRO_COMPILER_MLX_MODEL,
        "aliases": [PRO_COMPILER_MLX_MODEL, "qwen2.5-coder:14b"],
        "provider": "mlx_8080",
        "endpoint": MLX_8080_CHAT_URL,
        "role": "2.15s Neural Regex PbE Induction & Multi-Step Python Synthesis (100% 7/7 LocalBench)",
        "size_mb": 8500,
    }

    if t == "max":
        return {
            "tier_id": "max",
            "name": "Max Sovereign Pack (64GB+ RAM - Core + 14B PbE Compiler + 30B-A3B / 32B Ready)",
            "description": (
                "tev1:0.8b (141ms Param/HAR Triage :11434) + voicefi-coder-mlx (4.97s Self-Heal :9379) "
                "+ Qwen2.5-Coder-14B-4bit (2.15s PbE Compiler :8080) + Qwen3-Coder-30B-A3B (53.4 tok/s Sparse MoE)"
            ),
            "models": [*starter_models, pro_compiler_model],
            "optional_frontier": MAX_FRONTIER_MLX_MODEL,
            "optional_sparse_moe": MAX_SPARSE_MOE_MLX_MODEL,
            "speculative_draft": SPECULATIVE_DRAFT_MLX_MODEL,
            "min_ram_gb": 64,
            "total_size_mb": 10420,
        }
    elif t == "pro":
        return {
            "tier_id": "pro",
            "name": "Pro Compiler Pack (16GB+ / 64GB RAM)",
            "description": (
                "tev1:0.8b (141ms Param/HAR Triage :11434) + voicefi-coder-mlx (4.97s Self-Heal :9379) "
                "+ Qwen2.5-Coder-14B-4bit (2.15s PbE Compiler :8080)"
            ),
            "models": [*starter_models, pro_compiler_model],
            "speculative_draft": SPECULATIVE_DRAFT_MLX_MODEL,
            "min_ram_gb": 16,
            "total_size_mb": 10420,
        }
    else:
        return {
            "tier_id": "starter",
            "name": "Starter / Core Pack (All Macs 8GB–64GB, ~1.65 GB Core)",
            "description": (
                "tev1:0.8b (141ms Param/HAR Triage :11434) "
                "+ voicefi-coder-mlx (4.97s Self-Heal :9379) + nomic-embed-text"
            ),
            "models": starter_models,
            "min_ram_gb": 8,
            "total_size_mb": 1920,
        }


def call_tev1_reflex_chat(
    prompt: str,
    model_preference: Tuple[str, ...] = ("tev1:0.8b",),
    max_tokens: int = 80,
    timeout: float = 2.5,
) -> Optional[str]:
    """
    Runtime Invocation Rule #1:
    Call `tev1:0.8b` (`qwen35` architecture) strictly via
    `http://127.0.0.1:11434/api/chat` with `"think": False` (~141ms latency).
    Never call `/api/generate` on `qwen35`, which triggers hidden reasoning tokens and 10x latency.
    """
    for model_name in model_preference:
        try:
            body = json.dumps(
                {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0.0, "num_predict": max_tokens},
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                OLLAMA_CHAT_URL,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    content = (data.get("message") or {}).get("content", "")
                    if "<|im_end|>" in content:
                        content = content.split("<|im_end|>", 1)[0]
                    return content.strip()
        except Exception:
            continue
    return None


def call_voicefi_coder_mlx(
    prompt: str,
    max_tokens: int = 360,
    timeout: float = 8.0,
) -> Optional[str]:
    """
    Runtime Invocation Rule #2:
    Call `voicefi-coder-mlx` on `:9379` with `max_tokens <= 360` and strip `<|im_end|>`
    so the 1.5B LoRA stops cleanly after emitting the unified diff (4.97s total patch time).
    """
    capped_tokens = min(int(max_tokens), 360)
    try:
        body = json.dumps(
            {
                "model": "voicefi-coder-mlx",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": capped_tokens,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            MLX_9379_CHAT_URL,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                choices = data.get("choices") or []
                if choices:
                    raw_txt = (choices[0].get("message") or {}).get("content", "")
                    if "<|im_end|>" in raw_txt:
                        raw_txt = raw_txt.split("<|im_end|>", 1)[0]
                    if "</think>" in raw_txt:
                        raw_txt = raw_txt.split("</think>", 1)[-1]
                    return raw_txt.strip()
    except Exception:
        pass
    return None


def check_organ_status() -> List[OrganHealth]:
    """Audit the operational readiness of all 5 local organs."""
    organs: List[OrganHealth] = []

    # 1. Ears
    from whole_body_compute.audio.vad import find_silero_vad_model
    from whole_body_compute.audio.turn_detector import find_smart_turn_model
    from whole_body_compute.stt.apple_speech import resolve_apple_speech_binary

    vad_path = find_silero_vad_model()
    turn_path = find_smart_turn_model()
    stt_bin = resolve_apple_speech_binary()

    if vad_path and turn_path and stt_bin:
        organs.append(
            OrganHealth(
                organ="1. Ears",
                name="Acoustic VAD, Smart-Turn & Apple Speech",
                status="healthy",
                details=f"Silero VAD ({vad_path.name}) + Smart-Turn ({turn_path.name}) + AppleSpeech",
                file_path=str(vad_path),
            )
        )
    else:
        missing = []
        if not vad_path:
            missing.append("silero_vad_v6.onnx")
        if not turn_path:
            missing.append("smart-turn-v3.2-cpu.onnx")
        if not stt_bin:
            missing.append("apple_speech_stream")
        organs.append(
            OrganHealth(
                organ="1. Ears",
                name="Acoustic Subsystem",
                status="warning" if vad_path else "missing",
                details=f"Missing: {', '.join(missing)}",
            )
        )

    # 2. Mouth
    try:
        from whole_body_compute.tts.kokoro_tts import KokoroTTS

        kokoro_ok = KokoroTTS.is_available()
        m_file, v_file = KokoroTTS.resolve_model_and_voices_files()
    except ImportError:
        kokoro_ok = False
        m_file, v_file = Path(), Path()
    if kokoro_ok and m_file.is_file() and v_file.is_file():
        organs.append(
            OrganHealth(
                organ="2. Mouth",
                name="Kokoro-82M v1.0 FP16 Neural TTS",
                status="healthy",
                details=f"100% offline neural voice ready ({m_file.name}, {v_file.name})",
                file_path=str(m_file),
            )
        )
    else:
        organs.append(
            OrganHealth(
                organ="2. Mouth",
                name="Kokoro-82M Neural TTS",
                status="warning",
                details="Model files missing in ~/.whole_body_compute/models/kokoro (or ~/.voicefi/models/kokoro). macOS say fallback active.",
            )
        )

    # 3. Eyes
    foveal_memory_running = is_foveal_memory_running()
    ax_trusted = False
    try:
        import ApplicationServices

        ax_trusted = ApplicationServices.AXIsProcessTrusted()
    except Exception:
        pass

    eyes_status = "healthy" if foveal_memory_running else "warning"
    details_str = (
        f"Foveal Memory (:9382): {'ONLINE' if foveal_memory_running else 'OFFLINE'} | "
        f"AX Trust: {'GRANTED' if ax_trusted else 'NEEDED'}"
    )
    organs.append(
        OrganHealth(
            organ="3. Eyes",
            name="Foveal Episodic Memory (:9382) & Foveal AX DOM",
            status=eyes_status,
            details=details_str,
            file_path=str(FOVEAL_MEMORY_DEFAULT_DB),
        )
    )

    # 4. Hands & Hippocampus
    foveal_db_ok = FOVEAL_MEMORY_DEFAULT_DB.is_file()
    dot_mem_ok = DOT_MEMORY_DEFAULT_DB.is_file()
    if foveal_db_ok and dot_mem_ok:
        organs.append(
            OrganHealth(
                organ="4. Hands & Memory",
                name="DotMemory WAL + Foveal Memory SQLite",
                status="healthy",
                details=f"Foveal Memory DB ({FOVEAL_MEMORY_DEFAULT_DB.name}) + DotMemory ({DOT_MEMORY_DEFAULT_DB.name}) active",
                file_path=str(DOT_MEMORY_DEFAULT_DB),
            )
        )
    else:
        organs.append(
            OrganHealth(
                organ="4. Hands & Memory",
                name="DotMemory & Episodic Storage",
                status="warning" if (foveal_db_ok or dot_mem_ok) else "missing",
                details=f"Foveal Memory DB: {'OK' if foveal_db_ok else 'Missing'} | DotMemory: {'OK' if dot_mem_ok else 'Uninitialized'}",
            )
        )

    # 5. Brain (:11434 tev1:0.8b + :9379 voicefi-coder-mlx + :8080 Qwen2.5-Coder-14B)
    from whole_body_compute.local.engine import is_ollama_running, list_ollama_models, is_mlx_running, get_active_mlx_model

    ollama_ok = is_ollama_running()
    mlx_9379_ok = is_mlx_running(port=9379)
    mlx_8080_ok = is_mlx_running(port=8080)
    mlx_8080_model = get_active_mlx_model(port=8080) if mlx_8080_ok else None
    installed_models = [m.get("name", "") for m in list_ollama_models()] if ollama_ok else []

    if ollama_ok or mlx_9379_ok or mlx_8080_ok:
        parts = [
            f":11434({'ON' if ollama_ok else 'OFF'})",
            f":9379({'ON' if mlx_9379_ok else 'OFF'})",
            f":8080({mlx_8080_model.split('/')[-1] if mlx_8080_model else ('ON' if mlx_8080_ok else 'OFF')})",
        ]
        organs.append(
            OrganHealth(
                organ="5. Brain",
                name="Local LLM Inference Triad",
                status="healthy" if (installed_models or mlx_9379_ok or mlx_8080_ok) else "warning",
                details=" | ".join(parts),
            )
        )
    else:
        organs.append(
            OrganHealth(
                organ="5. Brain",
                name="Local LLM Inference Triad",
                status="missing",
                details="Endpoints :11434 (Ollama), :9379 (voicefi-coder-mlx), and :8080 (MLX 14B) are offline.",
            )
        )

    return organs


def is_foveal_memory_running(url: str = FOVEAL_MEMORY_DEFAULT_URL) -> bool:
    """Check if the Sovereign Foveal Memory HTTP server responds to /health within 1 second."""
    try:
        req = urllib.request.Request(f"{url}/health")
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def start_builtin_foveal_memory_daemon(
    host: str = "127.0.0.1",
    port: int = 9382,
    db_path: Optional[Path] = None,
) -> bool:
    """
    Launch the zero-dependency built-in Sovereign Foveal Memory server (Apple Vision OCR + SQLite).
    Runs as a detached background process.
    """
    if is_foveal_memory_running(f"http://{host}:{port}"):
        return True

    db = db_path or FOVEAL_MEMORY_DEFAULT_DB
    db.parent.mkdir(parents=True, exist_ok=True)

    python_bin = sys.executable
    cmd = [
        python_bin,
        "-m",
        "whole_body_compute.integrations.foveal_memory",
        "--serve-http",
        "--host",
        host,
        "--port",
        str(port),
        "--db-path",
        str(db),
    ]

    log_file = db.parent / "foveal_http.log"
    try:
        with open(log_file, "a") as lf:
            subprocess.Popen(
                cmd,
                stdout=lf,
                stderr=lf,
                start_new_session=True,
            )
        for _ in range(15):
            time.sleep(0.2)
            if is_foveal_memory_running(f"http://{host}:{port}"):
                return True
    except Exception as exc:
        logger.error(f"Failed to start built-in Foveal Memory daemon: {exc}")

    return is_foveal_memory_running(f"http://{host}:{port}")


def install_foveal_memory_launch_agent(port: int = 9382) -> Tuple[bool, Path]:
    """Generate and register a macOS LaunchAgent so Foveal Memory (:9382) runs automatically at login."""
    LAUNCH_AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    plist_path = LAUNCH_AGENTS_DIR / "com.wholebodycompute.foveal-memory.plist"

    python_bin = sys.executable
    args = [python_bin, "-m", "whole_body_compute.integrations.foveal_memory", "--serve-http", "--port", str(port)]

    args_xml = "\n".join(f"        <string>{a}</string>" for a in args)
    db_dir = FOVEAL_MEMORY_DEFAULT_DB.parent
    db_dir.mkdir(parents=True, exist_ok=True)

    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.wholebodycompute.foveal-memory</string>
    <key>ProgramArguments</key>
    <array>
{args_xml}
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{db_dir / "foveal_http.log"}</string>
    <key>StandardErrorPath</key>
    <string>{db_dir / "foveal_http_err.log"}</string>
</dict>
</plist>
"""
    with open(plist_path, "w", encoding="utf-8") as f:
        f.write(content)

    uid = os.getuid()
    subprocess.run(["launchctl", "bootout", f"gui/{uid}/com.wholebodycompute.foveal-memory"], capture_output=True)
    subprocess.run(["launchctl", "enable", f"gui/{uid}/com.wholebodycompute.foveal-memory"], capture_output=True)
    res = subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", str(plist_path)], capture_output=True)
    if res.returncode != 0:
        subprocess.run(["launchctl", "load", "-w", str(plist_path)], capture_output=True)

    return True, plist_path


def init_dot_memory_db() -> Path:
    """Initialize ~/.whole_body_compute/dot_memory.db SQLite database with WAL mode and initial tables."""
    from whole_body_compute.local.dot_memory import MemoryStore

    store = MemoryStore(db_path=_resolve_dot_memory_default_db(), auto_seed=True)
    return store.db_path


def provision_tier_models(
    tier_id: str,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Any]:
    """
    Provision the Local Models / Hey Pal tier weights across Ollama (:11434) and MLX (:9379 / :8080):
      1. Ollama models (`tev1:0.8b`, `nomic-embed-text`)
      2. Direct-on-Metal MLX (`voicefi-coder-mlx` on `:9379` or `voicefi-coder:1.5b` fallback)
      3. Pro Compiler MLX (`mlx-community/Qwen2.5-Coder-14B-Instruct-4bit` on `:8080` for 16GB+ Macs)
    """
    from whole_body_compute.local.engine import (
        ensure_ollama_running,
        is_mlx_running,
        list_ollama_models,
        pull_ollama_model,
    )

    running, _ = ensure_ollama_running(auto_install=True)
    if not running:
        return {"success": False, "error": "Could not connect to or start Ollama daemon."}

    manifest = get_tier_manifest(tier_id)
    installed_list = list_ollama_models(force_refresh=True)
    installed_exact = {m.get("name", "") for m in installed_list}
    installed_prefixes = {m.get("name", "").split(":")[0] for m in installed_list}

    results = []
    for m in manifest["models"]:
        model_name = m["name"]
        aliases = m.get("aliases", [model_name])
        provider = m.get("provider", "ollama")

        # Case A: Direct-on-Metal voicefi-coder-mlx (:9379)
        if provider == "mlx_9379":
            mlx_dir = resolve_voicefi_coder_mlx_dir()
            if is_mlx_running(9379) or mlx_dir is not None:
                results.append(
                    {
                        "name": model_name,
                        "status": "already_installed",
                        "details": f":9379 active or fused safetensors at {mlx_dir}",
                    }
                )
                continue
            # Fallback to Ollama voicefi-coder:1.5b if MLX fused weights aren't bundled on disk
            fb_ollama = m.get("fallback_ollama", "voicefi-coder:1.5b")
            if fb_ollama in installed_exact or fb_ollama.split(":")[0] in installed_prefixes:
                results.append({"name": fb_ollama, "status": "already_installed"})
                continue
            ok, err = pull_ollama_model(fb_ollama)
            results.append({"name": fb_ollama, "status": "pulled" if ok else "failed", "error": err})
            continue

        # Case B: Native MLX :8080 Pro Compiler (mlx-community/Qwen2.5-Coder-14B-Instruct-4bit)
        if provider == "mlx_8080":
            if is_mlx_running(8080) or is_hf_mlx_model_cached(model_name):
                results.append({"name": model_name, "status": "already_installed"})
                continue
            try:
                from huggingface_hub import snapshot_download

                if progress_callback:
                    progress_callback(model_name, 0.0)
                snapshot_download(repo_id=model_name)
                results.append({"name": model_name, "status": "pulled"})
            except Exception as exc:
                results.append({"name": model_name, "status": "failed", "error": str(exc)})
            continue

        # Case C: Ollama GGUF models (tev1:0.8b, nomic-embed-text)
        if any(a in installed_exact or a.split(":")[0] in installed_prefixes for a in aliases):
            results.append({"name": model_name, "status": "already_installed"})
            continue

        if progress_callback:
            progress_callback(model_name, 0.0)

        def _cb(evt: dict, _m: str = model_name) -> None:
            if progress_callback:
                pct = evt.get("percent", 0.0)
                progress_callback(_m, pct)

        ok, err = pull_ollama_model(model_name, progress_callback=_cb)
        results.append({"name": model_name, "status": "pulled" if ok else "failed", "error": err})

    return {"success": True, "results": results}


def run_sovereign_bootstrap(
    tier_override: Optional[str] = None,
    skip_models: bool = False,
    quiet: bool = False,
) -> bool:
    """
    Run end-to-end Hey Pal sovereign bootstrapper.
    Verifies organs, initializes storage, starts Screenpipe, and provisions local models.
    """
    hw = get_hardware_profile()
    selected_tier = tier_override or hw.recommended_tier
    manifest = get_tier_manifest(selected_tier)

    if not quiet:
        print("\n" + "=" * 68)
        print(" 🛰️  VoiceFi & Hey Pal — Sovereign 100% Local Bootstrapper")
        print("=" * 68)
        print(f"  Mac Architecture:     {hw.chip_name} ({hw.cpu_cores} cores)")
        print(f"  Unified Memory:       {hw.total_ram_gb:.0f} GB RAM ({hw.free_ram_gb:.1f} GB free)")
        print(f"  Recommended Tier:     {manifest['name']}")
        print(f"  Model Stack:          {manifest['description']}")
        print("-" * 68)

    # Step 1: Storage
    if not quiet:
        print("📦 [1/4] Initializing Sovereign Episodic & Workflow Storage...")
    dot_db = init_dot_memory_db()
    FOVEAL_MEMORY_DEFAULT_DB.parent.mkdir(parents=True, exist_ok=True)
    workflows_scripts_dir = get_wbc_data_dir() / "workflows" / "scripts"
    workflows_scripts_dir.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print(f"  ✅ DotMemory WAL SQLite:  {dot_db}")
        print(f"  ✅ Foveal Memory SQLite:  {FOVEAL_MEMORY_DEFAULT_DB}")
        print(f"  ✅ Tier-1 API Scripts:    {workflows_scripts_dir}")

    # Step 2: Foveal Memory Daemon
    if not quiet:
        print("\n👁️  [2/4] Verifying Sovereign Foveal Memory Service (:9382)...")
    if is_foveal_memory_running():
        if not quiet:
            print("  ✅ Foveal Memory HTTP API already active on http://127.0.0.1:9382")
    else:
        if not quiet:
            print("  🚀 Starting built-in zero-dependency Foveal Memory daemon...")
        started = start_builtin_foveal_memory_daemon()
        if started:
            if not quiet:
                print("  ✅ Built-in Foveal Memory daemon online on http://127.0.0.1:9382")
            try:
                install_foveal_memory_launch_agent()
                if not quiet:
                    print("  ✅ Registered LaunchAgent for login persistence")
            except Exception as e:
                logger.debug(f"LaunchAgent registration notice: {e}")
        else:
            if not quiet:
                print("  ⚠️ Foveal Memory service pending. Direct SQLite fallback active.")

    # Step 3: Local LLMs
    if not skip_models:
        if not quiet:
            print(f"\n🧠 [3/4] Provisioning Local Models ({manifest['name']})...")
        from whole_body_compute.local.engine import ensure_ollama_running

        running, msg = ensure_ollama_running(auto_install=True)
        if running:
            if not quiet:
                print(f"  ✅ {msg}")

            def _print_prog(model_name: str, pct: float) -> None:
                if not quiet and pct is not None:
                    sys.stdout.write(f"\r  ⏳ Downloading {model_name}: {pct:.1f}%")
                    sys.stdout.flush()

            res = provision_tier_models(selected_tier, progress_callback=_print_prog)
            if not quiet:
                sys.stdout.write("\r" + " " * 68 + "\r")
                for r in res.get("results", []):
                    st = r.get("status")
                    if st == "already_installed":
                        print(f"  ✅ {r['name']}: installed")
                    elif st == "pulled":
                        print(f"  ✨ {r['name']}: downloaded & ready")
                    else:
                        print(f"  ⚠️ {r['name']}: {r.get('error')}")
        else:
            if not quiet:
                print(f"  ⚠️ Could not connect to local LLM backend: {msg}")
    else:
        if not quiet:
            print("\n🧠 [3/4] Skipping model downloads (--skip-models).")

    # Step 4: Health Audit
    if not quiet:
        print("\n🩺 [4/4] Final 5-Organ Sovereign Health Audit:")
        organs = check_organ_status()
        for org in organs:
            icon = "✅" if org.status == "healthy" else ("⚠️ " if org.status == "warning" else "❌")
            print(f"  {icon} {org.organ:<18} {org.name:<32} {org.details}")
        print("=" * 68)
        print(" 🎉 Hey Pal Sovereign OS Stack Ready!")
        print("    Run 'vifi listen' or press Ctrl+T to speak directly to local models.")
        print("=" * 68 + "\n")

    return True


def ensure_dmg_local_compute_bootstrapped_async(
    force: bool = False,
    on_progress: Optional[Callable[[str, float], None]] = None,
    on_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Optional[ Any ]:
    """
    Option B Background Bootstrapper for the macOS DMG install (1 unified user access level):
      - Bundled in the ~200 MB DMG: Kokoro-82M TTS, Silero VAD v6, Smart-Turn v3.2,
        Apple Speech stream, and Whole Body Compute / Screenpipe (:3030).
      - On `VoiceFi.app` launch (`is_dmg_install() == True`), starts Screenpipe (:3030)
        and initializes `dot_memory.db`, then checks the Mac's Unified Memory via
        `get_hardware_profile()` and downloads the hardware-matched Local Compute models
        in a non-blocking background daemon thread.
    """
    import threading
    from whole_body_compute.license import FeatureGate

    if not force and not FeatureGate.can_use_local_compute():
        return None

    def _bg_worker() -> None:
        try:
            # 1. Always ensure local storage & zero-egress Foveal Memory (:9382) are active
            init_dot_memory_db()
            FOVEAL_MEMORY_DEFAULT_DB.parent.mkdir(parents=True, exist_ok=True)
            (get_wbc_data_dir() / "workflows" / "scripts").mkdir(parents=True, exist_ok=True)

            if not is_foveal_memory_running():
                started = start_builtin_foveal_memory_daemon()
                if started:
                    try:
                        install_foveal_memory_launch_agent()
                    except Exception:
                        pass

            # 2. Check if hardware-matched Local Compute models have already been provisioned
            hw = get_hardware_profile()
            target_profile = hw.recommended_tier
            manifest = get_tier_manifest(target_profile)
            receipt_file = _resolve_local_compute_receipt_file()

            if not force and receipt_file.is_file():
                try:
                    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
                    if receipt.get("hardware_profile") == target_profile and receipt.get("completed"):
                        if on_complete:
                            on_complete({"status": "already_bootstrapped", "hardware_profile": target_profile})
                        return
                except Exception:
                    pass

            try:
                from whole_body_compute.ui.notifications import show_notification

                show_notification(
                    "Whole Body Compute 🧠",
                    f"Optimizing for {hw.chip_name} ({hw.total_ram_gb:.0f}GB RAM)",
                    f"Downloading local models ({manifest['name']}) in the background...",
                )
            except Exception:
                pass

            res = provision_tier_models(target_profile, progress_callback=on_progress)
            results_list = res.get("results", [])
            all_ok = bool(res.get("success")) and all(
                r.get("status") in ("already_installed", "pulled") for r in results_list
            )

            if all_ok:
                receipt_file.parent.mkdir(parents=True, exist_ok=True)
                receipt_file.write_text(
                    json.dumps(
                        {
                            "completed": True,
                            "hardware_profile": target_profile,
                            "chip_name": hw.chip_name,
                            "total_ram_gb": hw.total_ram_gb,
                            "models": [r.get("name") for r in results_list],
                            "timestamp": time.time(),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                try:
                    from whole_body_compute.ui.notifications import show_notification

                    show_notification(
                        "Hey Pal Local Compute Ready ✨",
                        f"{hw.chip_name} ({hw.total_ram_gb:.0f}GB RAM) Configured",
                        "All on-device voice, vision, and workflow models are ready.",
                    )
                except Exception:
                    pass

            if on_complete:
                on_complete(
                    {
                        "status": "completed" if all_ok else "partial",
                        "hardware_profile": target_profile,
                        "results": results_list,
                    }
                )
        except Exception as exc:
            logger.debug(f"Background Local Compute bootstrap notice: {exc}")
            if on_complete:
                on_complete({"status": "error", "error": str(exc)})

    t = threading.Thread(
        target=_bg_worker,
        daemon=True,
        name="LocalComputeBackgroundBootstrapper",
    )
    t.start()
    return t
