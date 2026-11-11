"""
VoiceFi Local Model Engine.
Manages the lifecycle of on-device LiteRT models and provides a high-level
interface to Google Antigravity SDK's LiteRTAgentConfig.
"""

import os
import sys
import json
import logging
import platform
import re
import threading
import asyncio
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("whole_body_compute.local.engine")

_ENGINE_CACHE: Dict[Tuple[str, str], Any] = {}
_ENGINE_LOCK = threading.Lock()


def is_litert_available() -> bool:
    """Check if LiteRT and/or Google Antigravity SDK are available in the Python environment."""
    try:
        import litert_lm  # noqa: F401

        return True
    except (ImportError, ModuleNotFoundError):
        pass

    try:
        from google.antigravity import LiteRTAgentConfig  # noqa: F401

        return True
    except (ImportError, ModuleNotFoundError):
        return False


def detect_hardware_backend() -> Tuple[str, str]:
    """
    Detect the optimal hardware backend for the current host.
    Returns: (backend_code, human_readable_description)
    """
    system = platform.system()
    machine = platform.machine()

    if system == "Darwin" and machine in ("arm64", "aarch64"):
        return ("gpu", "Apple Silicon Metal GPU (Metal 4)")
    elif system == "Linux":
        # Check for CUDA
        if os.path.exists("/proc/driver/nvidia/version") or os.environ.get("CUDA_VISIBLE_DEVICES"):
            return ("gpu", "NVIDIA CUDA GPU")
        return ("cpu", "CPU (Host x86_64/ARM)")
    elif system == "Windows":
        return ("gpu", "DirectX/CUDA GPU")
    return ("cpu", "CPU Fallback")


def get_default_models_dir() -> Path:
    """Return the default storage directory for LiteRT models."""
    return Path(os.path.expanduser("~/.litert-lm/models"))


def list_imported_models() -> List[Dict[str, Any]]:
    """List all models currently imported into ~/.litert-lm/models."""
    models_dir = get_default_models_dir()
    if not models_dir.exists():
        return []

    results = []
    try:
        for entry in models_dir.iterdir():
            if entry.is_dir():
                model_file = entry / "model.litertlm"
                if not model_file.exists():
                    # Check for any .litertlm file
                    litertlm_files = list(entry.glob("*.litertlm"))
                    model_file = litertlm_files[0] if litertlm_files else None

                if model_file and model_file.exists():
                    stat = model_file.stat()
                    results.append(
                        {
                            "name": entry.name,
                            "path": str(model_file.resolve()),
                            "size_bytes": stat.st_size,
                            "size_mb": round(stat.st_size / (1024 * 1024), 1),
                            "size_gb": round(stat.st_size / (1024 * 1024 * 1024), 2),
                            "modified_at": stat.st_mtime,
                        }
                    )
    except Exception as e:
        logger.warning(f"Error scanning LiteRT models directory: {e}")

    return sorted(results, key=lambda x: x["name"])


def is_ollama_running() -> bool:
    """Return True if the local Ollama HTTP daemon is reachable."""
    import urllib.request

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    try:
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=1.2) as resp:
            return resp.status == 200
    except Exception:
        return False


def is_mlx_running(port: int = 9379) -> bool:
    """Return True if the local Apple Silicon MLX HTTP worker is reachable on `port`."""
    import socket

    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=0.5):
            return True
    except OSError:
        return False


def find_ollama_binary() -> Optional[str]:
    """Locate the ollama CLI binary on macOS or PATH."""
    import shutil

    found = shutil.which("ollama")
    if found:
        return found
    candidates = [
        "/opt/homebrew/bin/ollama",
        "/usr/local/bin/ollama",
        "/Applications/Ollama.app/Contents/Resources/ollama",
        os.path.expanduser("~/.local/bin/ollama"),
    ]
    # Check ~/.ollama/v*/ollama versioned directories
    ollama_dir = Path.home() / ".ollama"
    if ollama_dir.is_dir():
        for versioned in sorted(ollama_dir.glob("v*/ollama"), reverse=True):
            candidates.append(str(versioned))

    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def ensure_ollama_running(auto_install: bool = False) -> Tuple[bool, str]:
    """
    Ensure the local Ollama daemon is installed and running.
    Returns (is_running, status_message).
    """
    import subprocess
    import time

    if is_ollama_running():
        return True, "Ollama daemon is active on 127.0.0.1:11434"

    ollama_bin = find_ollama_binary()
    if not ollama_bin and auto_install:
        import shutil

        brew_bin = shutil.which("brew") or (
            "/opt/homebrew/bin/brew" if os.path.isfile("/opt/homebrew/bin/brew") else None
        )
        if brew_bin:
            try:
                subprocess.run(
                    [brew_bin, "install", "ollama"],
                    check=True,
                    timeout=300,
                )
                ollama_bin = find_ollama_binary()
            except Exception as exc:
                return False, f"Homebrew install of ollama failed: {exc}"

    if not ollama_bin:
        return (
            False,
            "Ollama is not installed. Run `brew install ollama` or `vifi local setup` to install it.",
        )

    try:
        subprocess.Popen(
            [ollama_bin, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        for _ in range(15):
            time.sleep(0.3)
            if is_ollama_running():
                return True, f"Started Ollama daemon ({ollama_bin})"
    except Exception as exc:
        return False, f"Failed to start Ollama daemon: {exc}"

    return is_ollama_running(), "Waiting for Ollama daemon to bind to 127.0.0.1:11434"


_OLLAMA_MODELS_CACHE: Tuple[float, List[Dict[str, Any]]] = (0.0, [])


def list_ollama_models(force_refresh: bool = False) -> List[Dict[str, Any]]:
    """List all generative models installed in the local Ollama daemon (cached for 15s)."""
    global _OLLAMA_MODELS_CACHE
    import time
    import urllib.request

    now = time.monotonic()
    if not force_refresh and _OLLAMA_MODELS_CACHE[1] and (now - _OLLAMA_MODELS_CACHE[0]) < 15.0:
        return list(_OLLAMA_MODELS_CACHE[1])

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    results: List[Dict[str, Any]] = []
    try:
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for m in data.get("models", []):
                name = m.get("name", "")
                if not name or "embed" in name.lower():
                    continue
                size_bytes = int(m.get("size", 0) or 0)
                details = m.get("details", {}) or {}
                results.append(
                    {
                        "name": name,
                        "provider": "ollama",
                        "size_bytes": size_bytes,
                        "size_mb": round(size_bytes / (1024 * 1024), 1) if size_bytes else 0.0,
                        "size_gb": round(size_bytes / (1024 * 1024 * 1024), 2)
                        if size_bytes
                        else 0.0,
                        "family": details.get("family", ""),
                        "parameter_size": details.get("parameter_size", ""),
                        "quantization_level": details.get("quantization_level", ""),
                        "modified_at": m.get("modified_at", ""),
                    }
                )
        if results:
            _OLLAMA_MODELS_CACHE = (now, results)
    except Exception:
        return []
    return results


VOICEFI_CODER_MODELFILE = """FROM qwen2.5-coder:1.5b

PARAMETER temperature 0.15
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
PARAMETER num_keep -1

SYSTEM \"\"\"You are VoiceFi-Coder (voicefi-coder:1.5b), VoiceFi's custom-tuned local coding and voice model running on Apple Silicon Metal GPU.
- When asked what model you are, state clearly that you are VoiceFi-Coder (voicefi-coder:1.5b), VoiceFi's custom-tuned local coding model running on Apple Silicon Metal GPU.
- When asked about the user's code, repository, bugs, or UI behavior, ALWAYS answer directly and specifically in 1 to 2 concise sentences citing the exact file paths, function names, and variables from WORKSPACE ARCHITECTURE and LIVE CODE MATCHES.
- NEVER reply with vague meta-promises like "VoiceFi Coder can help you..." or generic capability lists—give the actual technical answer immediately.
- When asked to modify code or produce a diff, format your output strictly using SEARCH/REPLACE blocks:
<<<<<<< SEARCH
[exact original lines]
=======
[replacement lines]
>>>>>>> REPLACE
- When operating inside an autonomous tool loop, emit strictly valid JSON tool calls matching `{"tool": "<name>", "arguments": {...}}`.\"\"\"
"""


def build_voicefi_coder_model(tag: str = "voicefi-coder:1.5b") -> bool:
    """Create or update the VoiceFi-tuned `voicefi-coder:1.5b` model in Ollama."""
    import subprocess
    import tempfile

    ollama_bin = find_ollama_binary()
    if not ollama_bin:
        return False
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".Modelfile", delete=False, encoding="utf-8") as tf:
            tf.write(VOICEFI_CODER_MODELFILE)
            modelfile_path = tf.name
        res = subprocess.run(
            [ollama_bin, "create", tag, "-f", modelfile_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        try:
            os.unlink(modelfile_path)
        except Exception:
            pass
        return res.returncode == 0
    except Exception as exc:
        logger.debug(f"Failed to build {tag}: {exc}")
        return False


def get_recommended_starter_models(total_ram_gb: float = 16.0) -> List[Dict[str, Any]]:
    """Return curated local starter models calibrated to the Mac's Unified Memory."""
    models = [
        {
            "name": "voicefi-coder:1.5b",
            "label": "VoiceFi Coder 1.5B (VoiceFi Tuned • Default)",
            "size_gb": 0.92,
            "min_ram_gb": 8.0,
            "recommended": True,
            "voicefi_tuned": True,
            "description": "VoiceFi-tuned Apple Silicon model for surgical code diffs, AST recon, and fast voice chat (~920 MB).",
        },
        {
            "name": "qwen2.5-coder:1.5b",
            "label": "Qwen 2.5 Coder 1.5B (Base Starter)",
            "size_gb": 1.0,
            "min_ram_gb": 8.0,
            "recommended": False,
            "description": "Ultra-fast (<1.5s) base code & voice model (~980 MB).",
        },
        {
            "name": "gemma2:2b",
            "label": "Gemma 2 2B (Compact Voice & Chat)",
            "size_gb": 1.6,
            "min_ram_gb": 8.0,
            "recommended": False,
            "description": "Natural conversational voice assistant & quick brainstorming (~1.6 GB).",
        },
        {
            "name": "qwen2.5-coder:7b",
            "label": "Qwen 2.5 Coder 7B (Balanced Pro)",
            "size_gb": 4.7,
            "min_ram_gb": 16.0,
            "recommended": total_ram_gb >= 24.0,
            "description": "Deep multi-file code reasoning and test synthesis (~4.7 GB).",
        },
        {
            "name": "llama3.3:70b",
            "label": "Llama 3.3 70B (Apple Silicon Frontier)",
            "size_gb": 42.0,
            "min_ram_gb": 64.0,
            "recommended": total_ram_gb >= 64.0,
            "description": "Frontier 70B local reasoning for high-memory Apple Silicon Macs (~42 GB).",
        },
    ]
    return models


def list_all_local_models() -> List[Dict[str, Any]]:
    """Return a unified list of all installed VoiceFi-tuned, Ollama, MLX, and LiteRT local models."""
    combined: List[Dict[str, Any]] = []
    seen = set()

    # 1. Check Direct-on-Metal MLX fused VoiceFi-Coder weights
    mlx_fused = Path(
        os.environ.get(
            "WBC_FUSED_MODEL_WEIGHTS",
            str(Path.home() / "Projects" / "Local Models" / "distillation" / "fused_model" / "model.safetensors"),
        )
    )
    if mlx_fused.is_file():
        sz = mlx_fused.stat().st_size
        combined.append(
            {
                "name": "voicefi-coder-mlx",
                "provider": "mlx",
                "voicefi_tuned": True,
                "size_bytes": sz,
                "size_mb": round(sz / (1024 * 1024), 1),
                "size_gb": round(sz / (1024 * 1024 * 1024), 2),
                "family": "qwen2.5-mlx",
                "parameter_size": "1.5B-LoRA",
            }
        )
        seen.add("voicefi-coder-mlx")

    for om in list_ollama_models():
        entry = dict(om)
        if any(k in entry["name"].lower() for k in ("voicefi", "tev1", "nimble")):
            entry["voicefi_tuned"] = True
        combined.append(entry)
        seen.add(entry["name"])

    for lm in list_imported_models():
        if lm["name"] not in seen:
            entry = dict(lm)
            entry["provider"] = "litert"
            combined.append(entry)
            seen.add(lm["name"])

    # Prioritize VoiceFi-tuned models at the top of the list
    priority_order = {
        "voicefi-coder:1.5b": 0,
        "voicefi-coder-mlx": 1,
        "tev1:0.8b": 2,
        "tev1:latest": 3,
        "nimble:latest": 4,
        "qwen2.5-coder:1.5b": 5,
    }
    combined.sort(
        key=lambda m: (
            priority_order.get(m["name"], 10 if m.get("voicefi_tuned") else 50),
            m["name"],
        )
    )
    return combined


def pull_ollama_model(
    model_name: str,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Pull an Ollama model via the local HTTP daemon (`/api/pull`), streaming progress updates.
    If `voicefi-coder:1.5b` is requested, pulls base `qwen2.5-coder:1.5b` and builds the
    VoiceFi-tuned Modelfile automatically.
    """
    import urllib.request

    clean_model = (model_name or "voicefi-coder:1.5b").replace("ollama/", "").strip()
    is_voicefi_coder = clean_model in ("voicefi-coder:1.5b", "voicefi-coder")
    pull_target = "qwen2.5-coder:1.5b" if is_voicefi_coder else clean_model

    running, msg = ensure_ollama_running(auto_install=False)
    if not running:
        return {"success": False, "model": clean_model, "error": msg}

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    payload = json.dumps({"name": pull_target, "stream": True}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/pull",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    last_status = "starting"
    try:
        with urllib.request.urlopen(req, timeout=1800.0) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except Exception:
                    continue
                if "error" in evt:
                    return {
                        "success": False,
                        "model": clean_model,
                        "error": str(evt["error"]),
                    }
                last_status = evt.get("status", last_status)
                total = int(evt.get("total", 0) or 0)
                completed = int(evt.get("completed", 0) or 0)
                pct = round((completed / total) * 100.0, 1) if total > 0 else None
                if callable(progress_callback):
                    try:
                        progress_callback(
                            {
                                "model": clean_model,
                                "status": last_status,
                                "completed": completed,
                                "total": total,
                                "percent": pct,
                            }
                        )
                    except Exception:
                        pass

        if is_voicefi_coder:
            if callable(progress_callback):
                try:
                    progress_callback(
                        {
                            "model": "voicefi-coder:1.5b",
                            "status": "tuning VoiceFi-Coder Modelfile",
                            "completed": 100,
                            "total": 100,
                            "percent": 99.0,
                        }
                    )
                except Exception:
                    pass
            build_voicefi_coder_model("voicefi-coder:1.5b")

        return {"success": True, "model": clean_model, "status": "ready"}
    except Exception as exc:
        return {"success": False, "model": clean_model, "error": str(exc)}


def _resolve_ollama_model(requested: str = "gemma4-2b") -> Optional[str]:
    """Resolve a model name to an installed Ollama generative model if the Ollama daemon is reachable."""
    installed = [m["name"] for m in list_ollama_models()]
    if not installed:
        return None

    clean = (requested or "").replace("ollama/", "").strip()
    if clean in ("auto", "smart"):
        try:
            from whole_body_compute.local.router import default_router

            decision = default_router.select_model(task_type="coder", prefer_local=True)
            if decision.is_local and decision.selected_model in installed:
                return decision.selected_model
        except Exception:
            pass

    if clean in installed:
        return clean

    # Match prefix without tag (e.g., "qwen2.5-coder" -> "qwen2.5-coder:1.5b")
    for inst in installed:
        if inst.split(":")[0] == clean.split(":")[0] and clean not in ("gemma4-2b", "gemma4-26b", "auto", "smart"):
            return inst

    if any(k in clean for k in ("26b", "32b", "9b", "7b", "nimble", "70b")):
        prefs = [
            "llama3.3:70b",
            "qwen2.5-coder:32b",
            "qwen2.5-coder:7b",
            "nimble:latest",
            "tev1:latest",
            "voicefi-coder:1.5b",
            "qwen2.5-coder:1.5b",
            "gemma2:2b",
        ]
    else:
        prefs = [
            "voicefi-coder:1.5b",
            "qwen2.5-coder:1.5b",
            "tev1:0.8b",
            "gemma2:2b",
            "tev1:latest",
            "llama3.2:1b",
        ]

    for p in prefs:
        if p in installed:
            return p
    return installed[0]


GEMMA4_WALKIE_TALKIE_SYSTEM_PROMPT = (
    "You are a senior systems engineer on a voice channel. "
    "Never output Markdown formatting, bullet symbols, or raw code blocks in your spoken reply. "
    "Route code to tool calls and keep spoken answers under 3 sentences."
)

GEMMA4_RECOMMENDED_SAMPLING: Dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
}


def strip_gemma4_thought_block(text: str) -> Tuple[str, str]:
    """
    Split a Gemma 4 / reasoning model output into (internal_thought, spoken_verdict).
    Handles:
      - `<|channel>thought ... <|channel>response ...`
      - `<|channel>thought ... <|end_of_thought|> ...`
      - `<|channel>thought\n...\n\n[Final answer]`
      - Empty `<|channel>thought` delimiter tags when thinking is disabled
      - `<think> ... </think>` blocks
    """
    if not text:
        return "", ""

    raw = text
    thoughts: List[str] = []

    # 1. Explicit <think>...</think> blocks
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
    for m in think_pattern.finditer(raw):
        if m.group(1).strip():
            thoughts.append(m.group(1).strip())
    raw = think_pattern.sub("", raw)

    # 2. Gemma 4 <|channel>thought blocks
    if "<|channel>thought" in raw:
        after_tag = raw.split("<|channel>thought", 1)[1]
        # Check for explicit closing delimiter first
        closed = False
        for end_marker in ("<|channel>response", "<|channel>final", "<|end_of_thought|>", "<|thought_end|>"):
            if end_marker in after_tag:
                thought_part, answer_part = after_tag.split(end_marker, 1)
                if thought_part.strip():
                    thoughts.append(thought_part.strip())
                raw = answer_part
                closed = True
                break

        if not closed:
            # When thinking is disabled, Gemma 4 may emit an empty `<|channel>thought\n` before final response,
            # or separate the thought block from the final verdict via a double newline.
            stripped_leading = after_tag.lstrip("\r\n ")
            if "\n\n" in stripped_leading:
                parts = stripped_leading.rsplit("\n\n", 1)
                if len(parts) == 2 and parts[1].strip():
                    thoughts.append(parts[0].strip())
                    raw = parts[1]
                else:
                    raw = stripped_leading
            else:
                raw = stripped_leading

    # Clean any stray channel tags
    raw = re.sub(r"<\|channel>[a-zA-Z0-9_]*", "", raw).strip()
    return "\n".join(t for t in thoughts if t).strip(), raw


async def split_gemma4_thought_stream(
    token_iterator: AsyncIterator[str],
    on_thought_token: Optional[ Any] = None,
) -> AsyncIterator[str]:
    """
    Stream-split Gemma 4 26B / reasoning output so `<|channel>thought` or `<think>` tokens
    are piped live to `on_thought_token(chunk)` (for HUD display) while ONLY post-thought
    spoken verdict tokens are yielded to the caller / TTS engine.
    """
    buffer = ""
    mode = "detect"  # "detect" | "thought" | "response"
    end_markers = ("<|channel>response", "<|channel>final", "<|end_of_thought|>", "</think>")

    async for chunk in token_iterator:
        if not chunk:
            continue
        if mode == "response":
            clean_chunk = re.sub(r"<\|channel>[a-zA-Z0-9_]*", "", chunk)
            if clean_chunk:
                yield clean_chunk
            continue

        buffer += chunk

        if mode == "detect":
            if buffer.lstrip().startswith(("<|channel>thought", "<think>")):
                for start_tag in ("<|channel>thought", "<think>"):
                    if start_tag in buffer:
                        buffer = buffer.split(start_tag, 1)[1].lstrip("\r\n")
                        break
                # If immediately followed by an end marker (empty thought tag), switch straight to response
                switched = False
                for em in end_markers:
                    if em in buffer:
                        t_part, r_part = buffer.split(em, 1)
                        if t_part.strip() and on_thought_token:
                            on_thought_token(t_part)
                        mode = "response"
                        buffer = ""
                        if r_part:
                            yield r_part
                        switched = True
                        break
                if not switched:
                    mode = "thought"
            elif len(buffer) > 24 or ("\n" in buffer and "<" not in buffer):
                mode = "response"
                clean = re.sub(r"<\|channel>[a-zA-Z0-9_]*", "", buffer)
                buffer = ""
                if clean:
                    yield clean

        if mode == "thought":
            for em in end_markers:
                if em in buffer:
                    t_part, r_part = buffer.split(em, 1)
                    if t_part and on_thought_token:
                        on_thought_token(t_part)
                    mode = "response"
                    buffer = ""
                    if r_part.lstrip():
                        yield r_part.lstrip()
                    break
            else:
                # Double-newline transition for untagged end-of-thought
                if "\n\n" in buffer:
                    t_part, r_part = buffer.split("\n\n", 1)
                    if t_part.strip() and on_thought_token:
                        on_thought_token(t_part)
                    mode = "response"
                    buffer = ""
                    if r_part.lstrip():
                        yield r_part.lstrip()
                elif len(buffer) > 32:
                    # Flush safe prefix to HUD thought stream while keeping tail for delimiter detection
                    flush_part, buffer = buffer[:-20], buffer[-20:]
                    if flush_part and on_thought_token:
                        on_thought_token(flush_part)

    # Flush any remaining buffer at stream end
    if buffer.strip():
        thought_text, final_text = strip_gemma4_thought_block(buffer)
        if thought_text and on_thought_token:
            on_thought_token(thought_text)
        if final_text:
            yield final_text


def capture_active_window_image(output_path: Optional[str] = None) -> Optional[str]:
    """
    Silently capture the active macOS window (or main display fallback) in the background
    with zero shutter sound (`-x`) and zero focus steal for Gemma 4 26B Vision.
    """
    import subprocess
    import tempfile

    target = output_path or os.path.join(tempfile.gettempdir(), "vifi_vision_frame.png")
    window_id: Optional[int] = None

    if platform.system() == "Darwin":
        try:
            import Quartz

            windows = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
                Quartz.kCGNullWindowID,
            )
            for win in windows or []:
                layer = int(win.get("kCGWindowLayer", 99) or 99)
                owner = str(win.get("kCGWindowOwnerName", "") or "")
                bounds = win.get("kCGWindowBounds", {}) or {}
                width = float(bounds.get("Width", 0) or 0)
                height = float(bounds.get("Height", 0) or 0)
                if layer == 0 and width > 200 and height > 150 and "VoiceFi" not in owner:
                    window_id = int(win.get("kCGWindowNumber", 0) or 0)
                    if window_id > 0:
                        break
        except Exception:
            window_id = None

        try:
            cmd = ["screencapture", "-x", "-o"]
            if window_id:
                cmd.extend(["-l", str(window_id)])
            cmd.append(target)
            res = subprocess.run(cmd, capture_output=True, timeout=3.0)
            if res.returncode == 0 and os.path.isfile(target) and os.path.getsize(target) > 0:
                return target
        except Exception as exc:
            logger.debug(f"Silent window capture fallback notice: {exc}")

    return target if (os.path.isfile(target) and os.path.getsize(target) > 0) else None


class LocalModelEngine:
    """
    On-device execution engine for Gemma and open-weights models
    powered by Google AI Edge LiteRT, Ollama, and the Google Antigravity SDK.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "gemma4-2b",
        backend: Optional[str] = None,
        max_context_tokens: int = 65536,
        enable_speculative_decoding: bool = True,
        system_instructions: Optional[str] = None,
    ):
        if model_name in ("auto", "smart"):
            try:
                from whole_body_compute.local.router import default_router

                decision = default_router.select_model(task_type="coder", prefer_local=True)
                if decision.is_local and decision.selected_model:
                    model_name = decision.selected_model
            except Exception:
                pass

        if not model_path:
            target_path = os.path.expanduser(f"~/.litert-lm/models/{model_name}/model.litertlm")
            self.model_path = target_path
            self.model_name = model_name
            if not os.path.exists(target_path):
                imported = list_imported_models()
                matched = next((m for m in imported if m["name"] == model_name), None)
                if matched:
                    self.model_path = matched["path"]
                elif model_name == "gemma4-2b" and imported:
                    self.model_path = imported[0]["path"]
                    self.model_name = imported[0]["name"]
        else:
            self.model_path = os.path.expanduser(model_path)
            self.model_name = model_name
        detected_backend, _ = detect_hardware_backend()
        self.backend = backend or detected_backend
        self.max_context_tokens = max_context_tokens
        self.enable_speculative_decoding = enable_speculative_decoding
        self.system_instructions = system_instructions

    @property
    def is_installed(self) -> bool:
        return is_litert_available() or (_resolve_ollama_model(self.model_name) is not None)

    @property
    def model_exists(self) -> bool:
        return os.path.exists(self.model_path) or (_resolve_ollama_model(self.model_name) is not None)

    def get_status(self) -> Dict[str, Any]:
        """Return diagnostic status of the local model engine."""
        backend_code, backend_desc = detect_hardware_backend()
        model_file = Path(self.model_path)
        size_gb = 0.0
        if model_file.exists():
            size_gb = round(model_file.stat().st_size / (1024 * 1024 * 1024), 2)
        ollama_resolved = _resolve_ollama_model(self.model_name)
        ollama_models = list_ollama_models()
        imported_models = list_imported_models()

        return {
            "litert_available": is_litert_available(),
            "ollama_available": ollama_resolved is not None or is_ollama_running(),
            "ollama_running": is_ollama_running(),
            "ollama_model": ollama_resolved,
            "ollama_models": ollama_models,
            "backend": self.backend,
            "backend_desc": backend_desc,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "model_exists": self.model_exists,
            "model_size_gb": size_gb,
            "max_context_tokens": self.max_context_tokens,
            "speculative_decoding": self.enable_speculative_decoding,
            "imported_models": imported_models,
            "all_models": list_all_local_models(),
        }

    def build_agent_config(
        self,
        tools: Optional[List[Any]] = None,
        system_instructions: Optional[str] = None,
        capabilities: Optional[Any] = None,
    ) -> Any:
        """
        Build a LiteRTAgentConfig instance for use with google.antigravity.Agent.
        """
        si = system_instructions or self.system_instructions

        kwargs: Dict[str, Any] = {
            "model_path": self.model_path,
            "backend": self.backend,
            "enable_speculative_decoding": self.enable_speculative_decoding,
        }

        if si:
            kwargs["system_instructions"] = si
        if tools is not None:
            kwargs["tools"] = tools
        else:
            kwargs["tools"] = []
        if capabilities:
            kwargs["capabilities"] = capabilities

        try:
            from google.antigravity import LiteRTAgentConfig

            return LiteRTAgentConfig(**kwargs)
        except (ImportError, ModuleNotFoundError):
            if not self.is_installed:
                raise RuntimeError(
                    "LiteRT or Google Antigravity SDK is not installed. "
                    "Run `uv pip install google-antigravity litert-lm` or `pip install voicefi[local]`."
                )
            return kwargs

    async def chat_stream(
        self,
        prompt: str,
        system_instructions: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        on_thought_token: Optional[Any] = None,
        image_path: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Stream tokens directly from the local LiteRT or Ollama model on Metal GPU.
        Automatically splits `<|channel>thought` / `<think>` blocks to `on_thought_token`
        so only spoken verdict tokens are yielded to TTS.
        """
        async for token in split_gemma4_thought_stream(
            self._raw_chat_stream(
                prompt=prompt,
                system_instructions=system_instructions,
                tools=tools,
                image_path=image_path,
            ),
            on_thought_token=on_thought_token,
        ):
            yield token

    async def _raw_chat_stream(
        self,
        prompt: str,
        system_instructions: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        image_path: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Internal raw token stream from LiteRT, Direct-on-Metal MLX, or Ollama.
        """
        from whole_body_compute.local.supervisor import default_supervisor

        if hasattr(default_supervisor, "resolve_loadable_model"):
            approved, resolved_model, reason = default_supervisor.resolve_loadable_model(
                self.model_name
            )
            if not approved:
                default_supervisor.assert_can_load(self.model_name)
            elif resolved_model != self.model_name:
                logger.warning(
                    f"[LocalModelEngine] Auto-fallback '{self.model_name}' -> '{resolved_model}': {reason}"
                )
                self.model_name = resolved_model
        else:
            default_supervisor.assert_can_load(self.model_name)

        # 1. Try google.antigravity Agent only if tool execution is required
        if tools:
            try:
                from google.antigravity import Agent

                config = self.build_agent_config(
                    tools=tools,
                    system_instructions=system_instructions,
                )

                async with Agent(config=config) as agent:
                    async with asyncio.timeout(15.0):
                        response = await agent.chat(prompt)
                        async for token in response:
                            yield token
                return
            except Exception as e:
                logger.debug(f"Antigravity Agent skipped/fell back ({e}), proceeding to native engine.")

        # 2. If Direct-on-Metal MLX model requested, query warm :9379 Metal GPU bridge first
        if not image_path and ("mlx" in self.model_name.lower() or self.model_name == "voicefi-coder-mlx"):
            import urllib.request

            si = system_instructions or self.system_instructions
            msgs = []
            if si:
                msgs.append({"role": "system", "content": si.strip()})
            msgs.append({"role": "user", "content": prompt.strip()})
            mlx_payload = {
                "model": "voicefi-coder-mlx",
                "messages": msgs,
                "stream": False,
                "temperature": 0.1,
            }

            def _mlx_call() -> Optional[str]:
                try:
                    req = urllib.request.Request(
                        "http://127.0.0.1:9379/v1/chat/completions",
                        data=json.dumps(mlx_payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=20.0) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        choices = data.get("choices", [])
                        if choices and "message" in choices[0]:
                            return choices[0]["message"].get("content", "").strip()
                except Exception as e:
                    logger.debug(f"Direct-on-Metal MLX bridge unavailable ({e}), falling back to Ollama.")
                return None

            mlx_out = await asyncio.to_thread(_mlx_call)
            if mlx_out:
                for word in re.findall(r"\S+\s*", mlx_out):
                    yield word
                return

        # 3. If LiteRT model file is absent, stream or fetch from Ollama fallback
        if not os.path.exists(self.model_path):
            ollama_model = _resolve_ollama_model(self.model_name)
            if not ollama_model:
                raise FileNotFoundError(
                    f"Model file not found at {self.model_path} and no Ollama model is reachable."
                )
            import base64
            import urllib.request

            base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
            is_gemma4 = "gemma4" in self.model_name.lower() or "gemma4" in ollama_model.lower()
            si = system_instructions or self.system_instructions or (
                GEMMA4_WALKIE_TALKIE_SYSTEM_PROMPT if is_gemma4 else None
            )
            msgs = []
            if si:
                msgs.append({"role": "system", "content": si.strip()})
            user_msg: Dict[str, Any] = {"role": "user", "content": prompt.strip()}
            if image_path and os.path.isfile(image_path):
                try:
                    with open(image_path, "rb") as img_f:
                        user_msg["images"] = [base64.b64encode(img_f.read()).decode("ascii")]
                except Exception as exc:
                    logger.debug(f"Failed to encode image {image_path}: {exc}")
            msgs.append(user_msg)

            options: Dict[str, Any] = {"temperature": 0.15, "num_predict": 220, "num_ctx": 4096}
            if is_gemma4:
                options.update(GEMMA4_RECOMMENDED_SAMPLING)

            keep_alive = "30m"
            try:
                from whole_body_compute.local.supervisor import default_supervisor
                keep_alive = default_supervisor.get_adaptive_keep_alive(ollama_model).get("recommended_keep_alive", "30m")
            except Exception:
                pass

            payload = {
                "model": ollama_model,
                "messages": msgs,
                "stream": True,
                "keep_alive": keep_alive,
                "options": options,
            }

            ollama_q: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def _ollama_stream_worker() -> None:
                try:
                    req = urllib.request.Request(
                        f"{base_url}/api/chat",
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=30.0) as resp:
                        for raw_line in resp:
                            line = raw_line.decode("utf-8", errors="ignore").strip()
                            if not line:
                                continue
                            try:
                                data = json.loads(line)
                            except Exception:
                                continue
                            piece = data.get("message", {}).get("content", "")
                            if piece:
                                loop.call_soon_threadsafe(ollama_q.put_nowait, piece)
                            if "load_duration" in data:
                                try:
                                    from whole_body_compute.local.supervisor import default_supervisor
                                    default_supervisor.record_model_load_telemetry(
                                        model_name=ollama_model,
                                        load_duration_ms=data["load_duration"] / 1e6,
                                        total_duration_ms=data.get("total_duration", 0) / 1e6,
                                        runtime="ollama",
                                    )
                                except Exception:
                                    pass
                except Exception as exc:
                    loop.call_soon_threadsafe(ollama_q.put_nowait, exc)
                finally:
                    loop.call_soon_threadsafe(ollama_q.put_nowait, None)

            threading.Thread(
                target=_ollama_stream_worker,
                daemon=True,
                name=f"OllamaStream-{ollama_model}",
            ).start()

            while True:
                item = await ollama_q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
            return

        import litert_lm
        from litert_lm.interfaces import GPU, CPU

        cache_key = (self.model_path, self.backend)
        if cache_key not in _ENGINE_CACHE:
            with _ENGINE_LOCK:
                if cache_key not in _ENGINE_CACHE:
                    cache_dir = os.path.expanduser("~/.litert-lm/cache")
                    os.makedirs(cache_dir, exist_ok=True)
                    try:
                        backend_obj = GPU() if self.backend in ("gpu", "metal") else CPU()
                        _ENGINE_CACHE[cache_key] = litert_lm.Engine(
                            model_path=self.model_path,
                            backend=backend_obj,
                            cache_dir=cache_dir,
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to load LiteRT with GPU backend ({e}), falling back to CPU: {e}"
                        )
                        _ENGINE_CACHE[cache_key] = litert_lm.Engine(
                            model_path=self.model_path,
                            backend=CPU(),
                            cache_dir=cache_dir,
                        )

        engine = _ENGINE_CACHE[cache_key]
        full_prompt = prompt
        si = system_instructions or self.system_instructions
        if si:
            full_prompt = f"{si.strip()}\n\n{prompt.strip()}"

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _worker():
            conv = None
            try:
                conv = engine.create_conversation()
                for chunk in conv.send_message_async(full_prompt):
                    if isinstance(chunk, dict):
                        content = chunk.get("content", [])
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                loop.call_soon_threadsafe(queue.put_nowait, item["text"])
                            elif isinstance(item, str):
                                loop.call_soon_threadsafe(queue.put_nowait, item)
                    elif isinstance(chunk, str):
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as exc:
                logger.error(f"LiteRT token generation error: {exc}")
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                if conv:
                    try:
                        conv.close()
                    except Exception:
                        pass
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(
            target=_worker, daemon=True, name=f"LiteRTWorker-{self.model_name}"
        ).start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    async def chat_text(
        self,
        prompt: str,
        system_instructions: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        on_thought_token: Optional[Any] = None,
        image_path: Optional[str] = None,
    ) -> str:
        """
        Execute a prompt on the local model and return the full completed response text
        (with internal `<|channel>thought` blocks stripped and routed to `on_thought_token`).
        """
        tokens = []
        async for token in self.chat_stream(
            prompt=prompt,
            system_instructions=system_instructions,
            tools=tools,
            on_thought_token=on_thought_token,
            image_path=image_path,
        ):
            tokens.append(token)
        return "".join(tokens).strip()

    async def distill_turn(
        self,
        agent_output: str,
        max_words: int = 24,
        telegraphic: bool = False,
        measure: bool = True,
    ) -> Tuple[str, Optional[Any]]:
        """
        Distill agent turn output into a punchy spoken soundbite or telegraphic summary.
        Measures TTFB and throughput and appends to ~/.voicefi/benchmarks.json if measure=True.
        """
        import time
        from whole_body_compute.local.benchmark import record_inference_metrics

        if not agent_output or not agent_output.strip():
            return "", None

        # Truncate overly long context for sub-second turn distillation
        bounded = agent_output.strip()
        if len(bounded) > 3000:
            bounded = bounded[:1000] + "\n...\n" + bounded[-1500:]

        if telegraphic:
            system_prompt = (
                "You are VoiceFi's Telegraphic Speed-Talking Distiller. "
                f"Condense the agent's output into an ultra-dense, telegraphic spoken soundbite under {max_words} words. "
                "Cut all conversational filler, greetings, pleasantries, and markdown. "
                "Deliver pure signal, key metrics, and completion outcomes."
            )
            test_name = "Telegraphic Distillation"
        else:
            system_prompt = (
                "You are VoiceFi's spoken voice synthesizer for AI coding agents. "
                f"Condense the agent's output into a single punchy spoken sentence under {max_words} words. "
                "Do NOT include markdown, asterisks, backticks, emojis, bullet points, or raw code blocks. "
                "Speak directly to the developer (e.g. 'I refactored the auth middleware and all tests pass.')."
            )
            test_name = "Spoken Turn Distillation"

        prompt = f"Agent Output:\n{bounded}\n\nSpoken soundbite:"

        start_time = time.perf_counter()
        first_token_time = None
        tokens = []

        try:
            async for token in self.chat_stream(
                prompt=prompt,
                system_instructions=system_prompt,
                tools=[],
            ):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                tokens.append(token)
        except Exception as e:
            logger.warning(f"Local distill_turn execution error: {e}")
            return "", None

        total_time = max(0.001, time.perf_counter() - start_time)
        ttfb = (first_token_time - start_time) if first_token_time else total_time
        ttfb_ms = ttfb * 1000.0

        raw_result = "".join(tokens).strip()
        clean = re.sub(r"^[\"']|[\"']$", "", raw_result)
        clean = re.sub(r"[`*#_]", "", clean)
        clean = " ".join(clean.split()).strip()
        words = clean.split()
        if len(words) > max_words:
            clean = " ".join(words[:max_words]) + "."

        prompt_tokens = max(1, len(prompt) // 4)
        output_tokens = max(1, len(tokens))
        gen_time = max(0.001, total_time - ttfb)
        tok_per_sec = output_tokens / gen_time

        bench_res = None
        if measure:
            _, backend_desc = detect_hardware_backend()
            bench_res = record_inference_metrics(
                test_name=test_name,
                target_engine=f"Local ({self.model_name})",
                backend_desc=backend_desc,
                ttfb_ms=ttfb_ms,
                total_seconds=total_time,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                tok_per_sec=tok_per_sec,
                cost_usd=0.0,
                notes=f"{len(words)} words spoken soundbite",
            )

        return clean, bench_res

    async def classify_intent(
        self,
        prompt: str,
        measure: bool = True,
    ) -> Tuple[Dict[str, Any], Optional[Any]]:
        """
        Classify a spoken prompt into target destinations:
        - local_command (battery, git branch/status, time, pause, stop)
        - antigravity (IDE, multi-file code editing, architecture)
        - claude (terminal CLI, shell scripts)
        - codex (ChatGPT / Codex desktop prompt)
        - obsidian (notes, daily journal entry, bookmarks)
        - memo (private voice memo / brain dump)
        """
        import time
        import urllib.request
        import json
        import asyncio
        from whole_body_compute.local.benchmark import record_inference_metrics

        system_prompt = (
            "You are VoiceFi's sub-100ms on-device intent classifier for AI coding agents and macOS desktop actions. "
            "Analyze the developer's spoken prompt and select the correct routing target.\n"
            "Categories:\n"
            "- 'local_command': battery, time, git branch, volume, pause.\n"
            "- 'antigravity': general code editing, file changes, refactoring.\n"
            "- 'claude': CLI commands, terminal scripts, bash.\n"
            "- 'codex': ChatGPT / Codex desktop assistance.\n"
            "- 'obsidian': notes, daily journal, wiki.\n"
            "- 'memo': long voice brain dumps."
        )

        resolved_model = _resolve_ollama_model(self.model_name) or "voicefi-coder:1.5b"
        keep_alive = "30m"
        try:
            from whole_body_compute.local.supervisor import default_supervisor
            keep_alive = default_supervisor.get_adaptive_keep_alive(resolved_model).get("recommended_keep_alive", "30m")
        except Exception:
            pass

        payload = {
            "model": resolved_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Spoken prompt: {prompt}"},
            ],
            "format": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "enum": [
                            "local_command",
                            "antigravity",
                            "claude",
                            "codex",
                            "obsidian",
                            "memo",
                        ],
                        "description": "The routing destination.",
                    },
                    "action": {
                        "type": "string",
                        "description": "A concise summary or specific command to run based on the prompt.",
                    },
                },
                "required": ["target", "action"],
            },
            "stream": False,
            "keep_alive": keep_alive,
            "options": {"temperature": 0.0, "num_predict": 40},
        }

        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        start_time = time.perf_counter()

        def _fetch():
            try:
                with urllib.request.urlopen(req, timeout=10.0) as response:
                    return json.loads(response.read().decode())
            except Exception as e:
                logger.warning(f"Ollama intent API error: {e}")
                return None

        result = await asyncio.to_thread(_fetch)
        total_time = max(0.001, time.perf_counter() - start_time)

        if result and "load_duration" in result:
            try:
                from whole_body_compute.local.supervisor import default_supervisor
                default_supervisor.record_model_load_telemetry(
                    model_name=resolved_model,
                    load_duration_ms=result["load_duration"] / 1e6,
                    total_duration_ms=result.get("total_duration", 0) / 1e6,
                    runtime="ollama",
                )
            except Exception:
                pass

        parsed = {}
        if result and "message" in result and "content" in result["message"]:
            try:
                parsed = json.loads(result["message"]["content"])
            except Exception:
                pass

        if not parsed:
            # Fallback heuristic
            target = "antigravity"
            low = prompt.lower()
            if any(
                w in low for w in ("battery", "time", "branch", "volume", "pause", "stop audio")
            ):
                target = "local_command"
            elif "claude" in low:
                target = "claude"
            elif "codex" in low or "chatgpt" in low:
                target = "codex"
            elif "obsidian" in low or "note" in low or "journal" in low:
                target = "obsidian"
            elif "memo" in low:
                target = "memo"
            parsed = {"target": target, "action": prompt}

        bench_res = None
        if measure:
            _, backend_desc = detect_hardware_backend()
            bench_res = record_inference_metrics(
                test_name="Intent Routing (Ollama Structured)",
                target_engine=f"Local ({self.model_name})",
                backend_desc=backend_desc,
                ttfb_ms=total_time * 1000.0,
                total_seconds=total_time,
                prompt_tokens=100,
                output_tokens=result.get("eval_count", 15) if result else 15,
                tok_per_sec=(result.get("eval_count", 15) / total_time) if result else 0.0,
                cost_usd=0.0,
                notes=f"Target: {parsed.get('target', 'unknown')}",
            )

        return parsed, bench_res

    async def structure_memo(
        self,
        raw_transcript: str,
        measure: bool = True,
    ) -> Tuple[Dict[str, Any], Optional[Any]]:
        """
        Synthesize a raw spoken brain dump into structured architecture decisions and tasks.
        """
        import time
        from whole_body_compute.local.benchmark import record_inference_metrics

        system_prompt = (
            "You are an expert software architect analyzing a developer's voice memo brain dump. "
            "Extract the core technical vision into structured JSON with these exact keys: "
            "'title' (short punchy title, 3-6 words), "
            "'summary' (2-3 sentence overview), "
            "'decisions' (list of key architectural decisions), "
            "'action_items' (list of concrete implementation steps). "
            "Output ONLY valid JSON."
        )

        user_msg = f"Developer Voice Memo Transcript:\n{raw_transcript}\n\nJSON Output:"

        start_time = time.perf_counter()
        first_token_time = None
        tokens = []

        try:
            async for token in self.chat_stream(
                prompt=user_msg,
                system_instructions=system_prompt,
                tools=[],
            ):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                tokens.append(token)
        except Exception as e:
            logger.warning(f"Local structure_memo execution error: {e}")

        total_time = max(0.001, time.perf_counter() - start_time)
        ttfb = (first_token_time - start_time) if first_token_time else total_time
        ttfb_ms = ttfb * 1000.0

        raw_result = "".join(tokens).strip()
        parsed = {}
        try:
            clean_json = re.sub(r"^```json\s*", "", raw_result, flags=re.IGNORECASE)
            clean_json = re.sub(r"```$", "", clean_json.strip())
            parsed = json.loads(clean_json)
        except Exception:
            parsed = {
                "title": "Voice Memo",
                "summary": raw_transcript[:200],
                "decisions": [],
                "action_items": [],
            }

        prompt_tokens = max(1, len(user_msg) // 4)
        output_tokens = max(1, len(tokens))
        gen_time = max(0.001, total_time - ttfb)
        tok_per_sec = output_tokens / gen_time

        bench_res = None
        if measure:
            _, backend_desc = detect_hardware_backend()
            bench_res = record_inference_metrics(
                test_name="Air-Gapped Memo Scribing",
                target_engine=f"Local ({self.model_name})",
                backend_desc=backend_desc,
                ttfb_ms=ttfb_ms,
                total_seconds=total_time,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                tok_per_sec=tok_per_sec,
                cost_usd=0.0,
                notes=f"Title: {parsed.get('title', 'Voice Memo')}",
            )

        return parsed, bench_res


def prewarm_local_models() -> None:
    """
    Pre-warm VoiceFi-Coder (`voicefi-coder:1.5b`) and Gemma (`gemma4-2b` / `gemma2:2b`)
    in Metal GPU memory so the first spoken turn on the Mobile Companion is sub-second.
    Safe to invoke in a daemon background thread at server startup.
    """
    import urllib.request

    # 0. If local Metal GPU bridge (:9379 voicefi-coder-mlx) is already active,
    # skip duplicate Ollama GGUF warming to preserve ~4.35 GB of Unified RAM,
    # unless explicitly requested via VOICEFI_PREWARM_OLLAMA=1.
    if is_mlx_running(port=9379) and os.environ.get("VOICEFI_PREWARM_OLLAMA") != "1":
        logger.info("Local Metal GPU server on :9379 is online; skipping redundant Ollama prewarm.")
        return

    # 1. Warm Ollama models (voicefi-coder:1.5b and gemma2:2b) with keep_alive
    try:
        installed = {m["name"] for m in list_ollama_models()}
        if "qwen2.5-coder:1.5b" in installed and "voicefi-coder:1.5b" not in installed:
            build_voicefi_coder_model("voicefi-coder:1.5b")
            installed.add("voicefi-coder:1.5b")

        base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        for target in ("voicefi-coder:1.5b", "gemma2:2b"):
            if target in installed:
                payload = json.dumps(
                    {
                        "model": target,
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": False,
                        "keep_alive": "30m",
                        "options": {"num_predict": 1},
                    }
                ).encode("utf-8")
                req = urllib.request.Request(
                    f"{base_url}/api/chat",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=15.0) as resp:
                    resp.read()
    except Exception as exc:
        logger.debug(f"Ollama prewarm notice: {exc}")

    # 2. Warm LiteRT Gemma 4 2B in Metal GPU memory
    try:
        gemma_path = os.path.expanduser("~/.litert-lm/models/gemma4-2b/model.litertlm")
        if os.path.exists(gemma_path) and is_litert_available():
            import litert_lm
            from litert_lm.interfaces import GPU

            backend_code, _ = detect_hardware_backend()
            cache_key = (gemma_path, backend_code)
            if cache_key not in _ENGINE_CACHE:
                with _ENGINE_LOCK:
                    if cache_key not in _ENGINE_CACHE:
                        cache_dir = os.path.expanduser("~/.litert-lm/cache")
                        os.makedirs(cache_dir, exist_ok=True)
                        eng = litert_lm.Engine(
                            model_path=gemma_path,
                            backend=GPU(),
                            cache_dir=cache_dir,
                        )
                        _ENGINE_CACHE[cache_key] = eng
                        conv = eng.create_conversation()
                        for _ in conv.send_message_async("hi"):
                            break
                        conv.close()
    except Exception as exc:
        logger.debug(f"LiteRT Gemma 4 prewarm notice: {exc}")
