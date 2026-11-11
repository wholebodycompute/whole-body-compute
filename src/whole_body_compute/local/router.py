"""
voicefi/local/router.py
Dynamic Hardware-Aware Model Router for Hybrid Cloud + Apple Silicon Orchestration.

Automatically cascades across model tiers (70B -> 32B/26B -> 7B -> 2B/1.5B -> Cloud Frontier)
based on real-time Unified Memory headroom, active disk swap, CPU load, and thermal pressure
monitored by ThermalSupervisor, combined with installed local Ollama & LiteRT model availability.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from whole_body_compute.local.engine import list_imported_models
from whole_body_compute.local.supervisor import HardwareTelemetry, ThermalSupervisor, default_supervisor

logger = logging.getLogger("whole_body_compute.local.router")

WARM_FAST_REFLEX_MODELS: Tuple[str, ...] = (
    "qwen2.5-coder:1.5b",
    "voicefi-coder:1.5b",
    "gemma2:2b",
)

# Ordered cascade hierarchy from heaviest local frontier down to cloud fallback
CASCADE_HIERARCHY: List[str] = [
    "llama3.3:70b",
    "qwen2.5-coder:32b",
    "gemma4-26b",
    "nimble:latest",
    "qwen2.5-coder:7b",
    "dolphin-llama3:latest",
    "tev1:latest",
    "voicefi-coder-mlx",
    "voicefi-coder:1.5b",
    "qwen2.5-coder:1.5b",
    "gemma2:2b",
    "gemma4-2b",
    "tev1:0.8b",
    "embedding-gemma-2:740m",
    "embedding-gemma-2",
    "llama3.2:1b",
    "cloud:gemini-flash",
]

TASK_DEFAULT_PREFERENCES: Dict[str, List[str]] = {
    "tier0_reflex": [
        "tev1:0.8b",
        "voicefi-coder-mlx",
        "qwen2.5-coder:1.5b",
        "gemma4-2b",
    ],
    "embed": [
        "embedding-gemma-2:740m",
        "embedding-gemma-2",
    ],
    "frontier": [
        "gemma4-26b",
        "llama3.3:70b",
        "nimble:latest",
        "tev1:latest",
        "voicefi-coder-mlx",
        "voicefi-coder:1.5b",
        "qwen2.5-coder:1.5b",
        "gemma2:2b",
        "cloud:gemini-pro",
    ],
    "vision": [
        "gemma4-26b",
        "gemma2:2b",
        "cloud:gemini-flash",
    ],
    "coder": [
        "voicefi-coder-mlx",
        "qwen2.5-coder:1.5b",
        "voicefi-coder:1.5b",
        "tev1:latest",
        "nimble:latest",
        "gemma2:2b",
        "cloud:gemini-flash",
    ],
    "scout": [
        "voicefi-coder-mlx",
        "qwen2.5-coder:1.5b",
        "voicefi-coder:1.5b",
        "tev1:0.8b",
        "gemma2:2b",
        "tev1:latest",
        "llama3.2:1b",
        "cloud:gemini-flash-lite",
    ],
    "qa": [
        "voicefi-coder-mlx",
        "voicefi-coder:1.5b",
        "qwen2.5-coder:1.5b",
        "tev1:latest",
        "nimble:latest",
        "gemma2:2b",
        "cloud:gemini-flash",
    ],
    "creative": [
        "dolphin-llama3:latest",
        "nimble:latest",
        "tev1:latest",
        "gemma2:2b",
        "voicefi-coder-mlx",
        "voicefi-coder:1.5b",
        "cloud:gemini-flash",
    ],
}


def classify_model_tier(model_name: str) -> str:
    """Return a normalized tier label for a model identifier."""
    key = (model_name or "").lower().strip()
    if not key:
        return "unknown"
    if key.startswith("cloud:"):
        return "cloud_frontier"
    if "embed" in key:
        return "edge_tier0_embed"
    if "0.8b" in key or key == "tev1:0.8b":
        return "edge_tier0_reflex"
    if any(t in key for t in ("70b", "72b")):
        return "local_70b_frontier"
    if any(t in key for t in ("32b", "34b", "27b", "26b")):
        return "local_26b_32b_mid"
    if any(t in key for t in ("14b", "13b", "12b", "9b", "8b", "7b")) or "dolphin" in key or "nimble" in key:
        return "local_7b_9b_compact"
    if "tev1:latest" in key or "4b" in key:
        return "local_4b_reasoning"
    return "local_1b_2b_scout"


@dataclass
class RoutingDecision:
    task_type: str
    requested_model: str
    selected_model: str
    tier: str
    is_local: bool
    cascaded: bool
    cascaded_from: Optional[str]
    cascade_reason: Optional[str]
    thermal_state: str
    free_ram_gb: float
    total_ram_gb: float
    swap_used_gb: float
    load_avg_1m: float
    decision_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HybridRouter:
    """
    Hardware-gated model router that bridges Local Apple Silicon (Ollama + LiteRT)
    and Cloud Frontier (Gemini Pro / Flash) with automatic fallback cascading.
    """

    def __init__(
        self,
        supervisor: Optional[ThermalSupervisor] = None,
        ollama_base_url: Optional[str] = None,
        enable_async_prewarm: bool = False,
    ):
        self.supervisor = supervisor or default_supervisor
        raw_ollama = (
            ollama_base_url
            or os.environ.get("OLLAMA_BASE_URL")
            or os.environ.get("OLLAMA_HOST")
            or "http://127.0.0.1:11434"
        ).strip()
        if "://" not in raw_ollama:
            raw_ollama = f"http://{raw_ollama}"
        self.ollama_base_url = raw_ollama.rstrip("/")
        self.enable_async_prewarm = enable_async_prewarm
        self._installed_cache: Optional[List[str]] = None
        self._cache_ts: float = 0.0
        self._warm_ps_cache: Optional[List[str]] = None
        self._warm_ps_ts: float = 0.0
        self._mlx_warm_override: Optional[bool] = None
        self._litert_warm_override: Optional[bool] = None

    def get_warm_ollama_models(self, force_refresh: bool = False) -> List[str]:
        """Return currently loaded (warm) Ollama models from `/api/ps` (cached for 5s)."""
        now = time.time()
        if (
            not force_refresh
            and self._warm_ps_cache is not None
            and (now - self._warm_ps_ts) < 5.0
        ):
            return list(self._warm_ps_cache)

        warm: List[str] = []
        try:
            req = urllib.request.Request(f"{self.ollama_base_url}/api/ps")
            with urllib.request.urlopen(req, timeout=0.25) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for m in data.get("models", []):
                    name = m.get("name") or m.get("model") or ""
                    if name and name not in warm:
                        warm.append(name)
        except Exception as e:
            logger.debug(f"Ollama ps probe skipped: {e}")

        self._warm_ps_cache = warm
        self._warm_ps_ts = now
        return list(warm)

    def _is_specialist_endpoint_warm(self) -> Tuple[bool, bool]:
        """Check whether :9379 (MLX) or :9380 (SpecialistRouter / LiteRT) are warm/reachable."""
        if self._mlx_warm_override is not None and self._litert_warm_override is not None:
            return self._mlx_warm_override, self._litert_warm_override

        def _port_open(port: int) -> bool:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.05):
                    return True
            except Exception:
                return False

        mlx_warm = (
            self._mlx_warm_override
            if self._mlx_warm_override is not None
            else _port_open(9379)
        )
        litert_warm = (
            self._litert_warm_override
            if self._litert_warm_override is not None
            else _port_open(9380)
        )
        return mlx_warm, litert_warm

    def _maybe_trigger_async_prewarm(self, target_model: str) -> None:
        """Optionally trigger non-blocking background pre-warm for cold specialist tier."""
        if not (self.enable_async_prewarm or os.environ.get("WBC_ASYNC_PREWARM") == "1"):
            return

        def _bg_prewarm() -> None:
            try:
                from whole_body_compute.local.radix_cache import RadixKVCacheManager

                mgr = RadixKVCacheManager()
                if hasattr(mgr, "trigger_async_prewarm"):
                    mgr.trigger_async_prewarm(target_model)
            except Exception:
                pass

        threading.Thread(target=_bg_prewarm, daemon=True).start()

    def get_warm_fast_model(self, task_type: str = "code") -> Optional[str]:
        """
        Query the fastest currently-warm local model in <15ms for `ReconScout`,
        `ReconImplementer`, or `pal_workflow` callers.
        """
        mlx_warm, _ = self._is_specialist_endpoint_warm()
        t_key = (task_type or "code").lower().strip()
        if mlx_warm and t_key in ("code", "coder", "scout", "tier0_reflex", "qa", "fast"):
            approved, _, _ = self.supervisor.can_load_model("voicefi-coder-mlx", is_tier0=True)
            if approved:
                return "voicefi-coder-mlx"

        warm_ollama = self.get_warm_ollama_models()
        if not warm_ollama:
            return None

        tier0_flag = True if t_key in ("tier0_reflex", "embed", "scout", "code", "coder", "fast") else False
        for candidate in WARM_FAST_REFLEX_MODELS:
            matched = next(
                (
                    w
                    for w in warm_ollama
                    if w == candidate
                    or w == f"{candidate}:latest"
                    or (":" not in candidate and w.startswith(f"{candidate}:"))
                ),
                None,
            )
            if matched:
                approved, _, _ = self.supervisor.can_load_model(matched, is_tier0=tier0_flag)
                if approved:
                    return matched

        return None

    def get_installed_local_models(self, force_refresh: bool = False) -> List[str]:
        """Return list of locally available Ollama and LiteRT model names (cached for 15s)."""
        now = time.time()
        if not force_refresh and self._installed_cache is not None and (now - self._cache_ts) < 15.0:
            return list(self._installed_cache)

        models: List[str] = []
        # 1. Query Ollama /api/tags
        try:
            req = urllib.request.Request(f"{self.ollama_base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=1.2) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for m in data.get("models", []):
                    name = m.get("name", "")
                    if name and name not in models:
                        models.append(name)
        except Exception as e:
            logger.debug(f"Ollama tags probe skipped: {e}")

        # 2. Query LiteRT models directory
        try:
            for m in list_imported_models():
                name = m.get("name", "")
                if name and name not in models:
                    models.append(name)
        except Exception as e:
            logger.debug(f"LiteRT models probe skipped: {e}")

        # 3. Check Direct-on-Metal MLX LoRA fused_model (.safetensors)
        mlx_fused = os.environ.get(
            "WBC_FUSED_MODEL_WEIGHTS",
            os.path.expanduser("~/Projects/Local Models/distillation/fused_model/model.safetensors"),
        )
        if os.path.exists(mlx_fused) and "voicefi-coder-mlx" not in models:
            models.insert(0, "voicefi-coder-mlx")

        self._installed_cache = models
        self._cache_ts = now
        return list(models)

    def resolve_model(
        self,
        task_type: str = "scout",
        preferred_model: Optional[str] = None,
        require_installed: bool = True,
        prefer_warm_reflex: bool = True,
    ) -> RoutingDecision:
        """
        Select the optimal model for `task_type` given real-time Apple Silicon vital signs.
        Cascades down the hierarchy if the preferred model is blocked by ThermalSupervisor
        or not installed locally. When `prefer_warm_reflex=True` and :9379/:9380 are cold,
        prioritizes an already-warm Ollama reflex/coding model (<390ms TTFB).
        """
        t0 = time.perf_counter()
        telem: HardwareTelemetry = self.supervisor.get_telemetry()
        installed = self.get_installed_local_models() if require_installed else []

        t_lower = task_type.lower().strip()
        base_chain = list(
            TASK_DEFAULT_PREFERENCES.get(t_lower, TASK_DEFAULT_PREFERENCES["scout"])
        )
        if preferred_model:
            clean_pref = preferred_model.replace("ollama/", "").strip()
            if clean_pref in base_chain:
                base_chain.remove(clean_pref)
            base_chain.insert(0, clean_pref)
            # Also ensure recommended fallback from supervisor is in chain
            req_info = self.supervisor.get_model_requirement(clean_pref)
            fb = req_info.get("recommended_fallback")
            if fb and fb not in base_chain:
                base_chain.insert(1, fb)

        requested = base_chain[0]

        # Warm-First Fast-Path for tier0_reflex, scout, and single-turn coding/classification tasks
        if (
            prefer_warm_reflex
            and preferred_model is None
            and t_lower in ("tier0_reflex", "scout", "coder", "code", "qa", "classify", "fast")
        ):
            mlx_warm, litert_warm = self._is_specialist_endpoint_warm()
            if not mlx_warm and not litert_warm:
                warm_fast = self.get_warm_fast_model(task_type=t_lower)
                if warm_fast:
                    self._maybe_trigger_async_prewarm(requested)
                    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                    return RoutingDecision(
                        task_type=task_type,
                        requested_model=requested,
                        selected_model=warm_fast,
                        tier=classify_model_tier(warm_fast),
                        is_local=True,
                        cascaded=(warm_fast != requested),
                        cascaded_from=requested if warm_fast != requested else None,
                        cascade_reason=(
                            f"Warm-first fast-path: selected resident Ollama model `{warm_fast}` "
                            f"(<390ms TTFB) while `{requested}` endpoint (:9379/:9380/LiteRT) is cold."
                            if warm_fast != requested
                            else None
                        ),
                        thermal_state=telem.thermal_state,
                        free_ram_gb=telem.free_ram_gb,
                        total_ram_gb=telem.total_ram_gb,
                        swap_used_gb=telem.swap_used_gb,
                        load_avg_1m=telem.load_avg_1m,
                        decision_latency_ms=latency_ms,
                    )

        first_failure_reason: Optional[str] = None

        for idx, candidate in enumerate(base_chain):
            if candidate.startswith("cloud:"):
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                return RoutingDecision(
                    task_type=task_type,
                    requested_model=requested,
                    selected_model=candidate,
                    tier=classify_model_tier(candidate),
                    is_local=False,
                    cascaded=(idx > 0),
                    cascaded_from=requested if idx > 0 else None,
                    cascade_reason=first_failure_reason if idx > 0 else None,
                    thermal_state=telem.thermal_state,
                    free_ram_gb=telem.free_ram_gb,
                    total_ram_gb=telem.total_ram_gb,
                    swap_used_gb=telem.swap_used_gb,
                    load_avg_1m=telem.load_avg_1m,
                    decision_latency_ms=latency_ms,
                )

            # Check hardware headroom first (tier 0 thermal exemption only applies to reflex/embed tasks)
            tier0_task = task_type in ("tier0_reflex", "embed")
            approved, reason, _ = self.supervisor.can_load_model(
                candidate, is_tier0=None if tier0_task else False
            )
            if not approved:
                if first_failure_reason is None:
                    first_failure_reason = reason
                continue

            # Check local installation if required
            if require_installed and installed:
                is_present = any(
                    candidate == inst
                    or inst == f"{candidate}:latest"
                    or (":" not in candidate and inst.startswith(f"{candidate}:"))
                    for inst in installed
                )
                if not is_present:
                    if first_failure_reason is None:
                        first_failure_reason = (
                            f"Model `{candidate}` hardware-approved ({telem.free_ram_gb} GB free), "
                            f"but weights not currently loaded in local Ollama/LiteRT; cascaded to installed model."
                        )
                    continue

            # Resolved a valid local candidate
            selected = candidate
            if require_installed and installed and candidate not in installed:
                for inst in installed:
                    if inst == f"{candidate}:latest" or (
                        ":" not in candidate and inst.startswith(f"{candidate}:")
                    ):
                        selected = inst
                        break

            latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            return RoutingDecision(
                task_type=task_type,
                requested_model=requested,
                selected_model=selected,
                tier=classify_model_tier(selected),
                is_local=True,
                cascaded=(selected != requested),
                cascaded_from=requested if selected != requested else None,
                cascade_reason=first_failure_reason if selected != requested else None,
                thermal_state=telem.thermal_state,
                free_ram_gb=telem.free_ram_gb,
                total_ram_gb=telem.total_ram_gb,
                swap_used_gb=telem.swap_used_gb,
                load_avg_1m=telem.load_avg_1m,
                decision_latency_ms=latency_ms,
            )

        # Ultimate fallback to Cloud Flash
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return RoutingDecision(
            task_type=task_type,
            requested_model=requested,
            selected_model="cloud:gemini-flash",
            tier="cloud_frontier",
            is_local=False,
            cascaded=True,
            cascaded_from=requested,
            cascade_reason=first_failure_reason or "All local tiers unavailable",
            thermal_state=telem.thermal_state,
            free_ram_gb=telem.free_ram_gb,
            total_ram_gb=telem.total_ram_gb,
            swap_used_gb=telem.swap_used_gb,
            load_avg_1m=telem.load_avg_1m,
            decision_latency_ms=latency_ms,
        )

    def route(
        self,
        task_type: str = "scout",
        preferred_model: Optional[str] = None,
        require_installed: bool = True,
        prefer_warm_reflex: bool = True,
    ) -> RoutingDecision:
        """Alias for `resolve_model` supporting `prefer_warm_reflex`."""
        return self.resolve_model(
            task_type=task_type,
            preferred_model=preferred_model,
            require_installed=require_installed,
            prefer_warm_reflex=prefer_warm_reflex,
        )


default_router = HybridRouter()
