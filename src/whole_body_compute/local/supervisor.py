"""
voicefi/local/supervisor.py
Apple Silicon Thermal & Hardware Resource Supervisor.

Monitors macOS thermal pressure, Unified Memory utilization, and CPU load.
Provides sleep-safe circuit breakers, model-aware memory gates, and backoff
throttlers for on-device local LLM inference (Ollama, LiteRT, MLX) and audio synthesis.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shlex
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger("whole_body_compute.local.supervisor")

LAST_GATE_EVENT_PATH = Path("/tmp/wbc_last_gate_event.json")
VALID_GATE_STATUSES = ("APPROVED", "AUTO_SWEPT", "FALLBACK", "BLOCKED", "NOMINAL")

# Calibrated Unified Memory Headroom Requirements for Apple Silicon (Weights + KV Cache)
DEFAULT_MODEL_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    # Native MLX 64 GB Apple Silicon M5 Pro Tiers (:8080)
    "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit": {
        "min_free_ram_gb": 20.0,
        "max_load_avg": 12.0,
        "disallow_swap": True,
        "recommended_fallback": "voicefi-coder-mlx",
    },
    "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit": {
        "min_free_ram_gb": 20.0,
        "max_load_avg": 12.0,
        "disallow_swap": True,
        "recommended_fallback": "voicefi-coder-mlx",
    },
    "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit": {
        "min_free_ram_gb": 9.5,
        "max_load_avg": 12.0,
        "disallow_swap": False,
        "recommended_fallback": "voicefi-coder-mlx",
    },
    "mlx-community/gemma-2-9b-it-4bit": {
        "min_free_ram_gb": 6.0,
        "max_load_avg": 14.0,
        "disallow_swap": False,
        "recommended_fallback": "voicefi-coder-mlx",
    },
    # Frontier 70B Tier: 42.5 GB weights + 3.5 GB KV cache
    "llama3.3:70b": {
        "min_free_ram_gb": 46.0,
        "max_load_avg": 8.0,
        "disallow_swap": True,
        "recommended_fallback": "gemma4-26b",
    },
    "qwen2.5-coder:32b": {
        "min_free_ram_gb": 22.0,
        "max_load_avg": 14.0,
        "disallow_swap": True,
        "recommended_fallback": "qwen2.5-coder:7b",
    },
    # Sparse 40B MoE Coder (2.5B active per token, ~22.6 GB INT4 weights)
    "flint:latest": {
        "min_free_ram_gb": 23.0,
        "max_load_avg": 14.0,
        "disallow_swap": False,
        "recommended_fallback": "voicefi-coder-mlx",
    },
    # Mid-Tier Coders (26B - 32B)
    "gemma4-26b": {
        "min_free_ram_gb": 18.0,
        "max_load_avg": 14.0,
        "disallow_swap": True,
        "recommended_fallback": "gemma2:2b",
    },
    # Compact Coders (7B - 8B)
    "qwen2.5-coder:7b": {
        "min_free_ram_gb": 6.5,
        "max_load_avg": 12.0,
        "disallow_swap": False,
        "recommended_fallback": "qwen2.5-coder:1.5b",
    },
    "dolphin-llama3:latest": {
        "min_free_ram_gb": 6.5,
        "max_load_avg": 12.0,
        "disallow_swap": False,
        "recommended_fallback": "tev1:latest",
    },
    "nimble:latest": {
        "min_free_ram_gb": 7.5,
        "max_load_avg": 12.0,
        "disallow_swap": False,
        "recommended_fallback": "tev1:latest",
    },
    "tev1:latest": {
        "min_free_ram_gb": 4.0,
        "max_load_avg": 13.0,
        "disallow_swap": False,
        "recommended_fallback": "voicefi-coder:1.5b",
    },
    # Small Scouts & Intent Distillers (0.8B - 2.6B)
    "gemma2:2b": {
        "min_free_ram_gb": 2.5,
        "max_load_avg": 14.0,
        "disallow_swap": False,
        "recommended_fallback": None,
    },
    "gemma4-2b": {
        "min_free_ram_gb": 2.5,
        "max_load_avg": 14.0,
        "disallow_swap": False,
        "recommended_fallback": None,
    },
    "voicefi-coder-mlx": {
        "min_free_ram_gb": 1.5,
        "max_load_avg": 14.0,
        "disallow_swap": False,
        "recommended_fallback": "voicefi-coder:1.5b",
        "is_tier0": True,
    },
    "voicefi-coder:1.5b": {
        "min_free_ram_gb": 2.0,
        "max_load_avg": 14.0,
        "disallow_swap": False,
        "recommended_fallback": None,
        "is_tier0": True,
    },
    "qwen2.5-coder:1.5b": {
        "min_free_ram_gb": 2.0,
        "max_load_avg": 14.0,
        "disallow_swap": False,
        "recommended_fallback": None,
        "is_tier0": True,
    },
    "llama3.2:1b": {
        "min_free_ram_gb": 1.8,
        "max_load_avg": 14.0,
        "disallow_swap": False,
        "recommended_fallback": None,
    },
    "tev1:0.8b": {
        "min_free_ram_gb": 1.2,
        "max_load_avg": 14.0,
        "disallow_swap": False,
        "recommended_fallback": None,
        "is_tier0": True,
    },
    # Tier 0 Multimodal Vector Embeddings (EmbeddingGemma 2)
    "embedding-gemma-2:740m": {
        "min_free_ram_gb": 0.8,
        "max_load_avg": 14.0,
        "disallow_swap": False,
        "recommended_fallback": None,
        "is_tier0": True,
    },
    "embedding-gemma-2": {
        "min_free_ram_gb": 0.8,
        "max_load_avg": 14.0,
        "disallow_swap": False,
        "recommended_fallback": None,
        "is_tier0": True,
    },
}


@dataclass
class HardwareTelemetry:
    system: str
    thermal_state: str  # NORMAL, FAIR, SERIOUS, CRITICAL, UNKNOWN
    thermal_raw: str
    free_ram_gb: float
    total_ram_gb: float
    ram_percent: float
    load_avg_1m: float
    is_safe: bool
    swap_used_gb: float = 0.0
    throttle_reason: Optional[str] = None
    power_source: str = "AC Power"
    battery_pct: int = 100
    low_power_mode: bool = False
    cgroup_limit_gb: Optional[float] = None
    cgroup_used_gb: Optional[float] = None


class ResourceGateError(RuntimeError):
    """Raised when hardware or memory resources are insufficient to safely load a model."""

    def __init__(self, message: str, alert_data: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.alert_data = alert_data or {}
        self.model_name = self.alert_data.get("model", "unknown")


class ThermalSupervisor:
    """
    On-device hardware guardrail supervisor for Apple Silicon / local compute.
    Prevents thermal throttling, memory exhaustion, and macOS swap thrashing.
    """

    def __init__(
        self,
        min_free_ram_gb: float = 4.0,
        max_load_avg: float = 14.0,
        pause_on_serious: bool = True,
        gate_event_path: Optional[Path] = None,
        proc_root: Optional[Path] = None,
        sys_root: Optional[Path] = None,
        platform_system: Optional[str] = None,
    ):
        self.min_free_ram_gb = min_free_ram_gb
        self.max_load_avg = max_load_avg
        self.pause_on_serious = pause_on_serious
        self._platform_system = platform_system or platform.system()
        self.is_macos = self._platform_system == "Darwin"
        self.is_linux = self._platform_system == "Linux"
        self._proc_root = Path(proc_root) if proc_root else Path("/proc")
        self._sys_root = Path(sys_root) if sys_root else Path("/sys")
        self._model_last_active_ts: Dict[int, float] = {}
        self._cephalic_state: str = "STANDBY"
        self._cephalic_model: str = "qwen2.5-coder:1.5b"
        self._cephalic_stimulus: str = "none"
        self._cephalic_last_ts: float = 0.0
        self._gate_event_path: Path = gate_event_path or LAST_GATE_EVENT_PATH
        self._last_gate_event: Optional[Dict[str, Any]] = None
        self._runaway_alert_cooldown: Dict[int, float] = {}
        self._runaway_sigterm_ts: Dict[int, float] = {}
        self._autonomic_high_cpu_ts: Dict[int, float] = {}
        self._last_opportunistic_reap_ts: float = 0.0
        self._telemetry_cache: Optional[HardwareTelemetry] = None
        self._telemetry_cache_ts: float = 0.0

    def get_thermal_state_macos(self) -> Tuple[str, str]:
        """Query macOS thermal state using pmset -g therm and NSProcessInfo."""
        if not self.is_macos:
            return ("NORMAL", "Non-macOS platform")

        try:
            res = subprocess.run(
                ["pmset", "-g", "therm"],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            raw = res.stdout.strip()
            raw_lower = raw.lower()
            state = "NORMAL"
            if "critical" in raw_lower:
                state = "CRITICAL"
            elif "serious" in raw_lower or "heavy" in raw_lower:
                state = "SERIOUS"
            elif "fair" in raw_lower or "moderate" in raw_lower:
                state = "FAIR"
            elif "speed_limit" in raw_lower and "speed_limit = 100" not in raw_lower:
                state = "FAIR"

            if state == "NORMAL":
                try:
                    from Foundation import NSProcessInfo

                    ns_code = int(NSProcessInfo.processInfo().thermalState())
                    ns_map = {1: "FAIR", 2: "SERIOUS", 3: "CRITICAL"}
                    if ns_code in ns_map:
                        ns_state = ns_map[ns_code]
                        return (
                            ns_state,
                            f"{raw}\nNSProcessInfo.thermalState={ns_code} ({ns_state})",
                        )
                except Exception:
                    pass
            return (state, raw)
        except Exception as e:
            logger.debug(f"Failed to query pmset -g therm: {e}")
            return ("UNKNOWN", str(e))

    def get_swap_usage_macos(self) -> Tuple[float, float]:
        """Query macOS swap usage (used_gb, total_gb) via sysctl vm.swapusage."""
        if not self.is_macos:
            return (0.0, 0.0)
        try:
            res = subprocess.run(
                ["sysctl", "vm.swapusage"],
                capture_output=True,
                text=True,
                timeout=1.5,
            )
            raw = res.stdout.strip()
            m_total = re.search(r"total\s*=\s*([\d\.]+)M", raw)
            m_used = re.search(r"used\s*=\s*([\d\.]+)M", raw)
            used_gb = round(float(m_used.group(1)) / 1024.0, 2) if m_used else 0.0
            total_gb = round(float(m_total.group(1)) / 1024.0, 2) if m_total else 0.0
            return (used_gb, total_gb)
        except Exception as e:
            logger.debug(f"Failed to read vm.swapusage: {e}")
            return (0.0, 0.0)

    def get_power_state_macos(self) -> Tuple[str, int, bool]:
        """Query macOS power source ('AC Power' | 'Battery'), battery %, and Low Power Mode (powermode 1)."""
        if not self.is_macos:
            return ("AC Power", 100, False)
        power_source = "AC Power"
        battery_pct = 100
        low_power_mode = False
        try:
            batt_out = subprocess.run(
                ["pmset", "-g", "batt"],
                capture_output=True,
                text=True,
                timeout=1.5,
            ).stdout
            if "Battery Power" in batt_out or "discharging" in batt_out.lower():
                power_source = "Battery"
            m_pct = re.search(r"(\d+)%", batt_out)
            if m_pct:
                battery_pct = int(m_pct.group(1))

            pm_out = subprocess.run(
                ["pmset", "-g"],
                capture_output=True,
                text=True,
                timeout=1.5,
            ).stdout
            if re.search(r"\bpowermode\s+1\b", pm_out):
                low_power_mode = True
        except Exception as e:
            logger.debug(f"Failed to read pmset power state: {e}")
        return (power_source, battery_pct, low_power_mode)

    # -----------------------------------------------------------------------
    # Linux, WSL2 & Devcontainer Telemetry (/proc, cgroups v2 & /sys/class/thermal)
    # -----------------------------------------------------------------------

    def get_thermal_state_linux(self, sys_root: Optional[Path] = None) -> Tuple[str, str]:
        """
        Query Linux thermal sensors from `/sys/class/thermal/thermal_zone*` (with `/sys/class/hwmon`
        fallback). Gracefully returns `NORMAL` on WSL2, Docker containers, and cloud VMs where
        thermal zones are unexposed.
        """
        s_root = Path(sys_root) if sys_root else self._sys_root
        thermal_base = s_root / "class" / "thermal"
        severity_rank = {"NORMAL": 0, "FAIR": 1, "SERIOUS": 2, "CRITICAL": 3}
        worst_state = "NORMAL"
        max_temp_c: Optional[float] = None
        max_zone_label = "thermal"

        def _classify_zone(temp_c: float, crit_c: Optional[float], hot_c: Optional[float]) -> str:
            eff_crit = crit_c if (crit_c is not None and 85.0 <= crit_c <= 130.0) else 95.0
            eff_hot = hot_c if (hot_c is not None and 75.0 <= hot_c < eff_crit) else 85.0
            if temp_c >= eff_crit:
                return "CRITICAL"
            if temp_c >= eff_hot:
                return "SERIOUS"
            if temp_c >= 75.0:
                return "FAIR"
            return "NORMAL"

        try:
            if thermal_base.is_dir():
                for zdir in sorted(thermal_base.glob("thermal_zone*")):
                    temp_file = zdir / "temp"
                    if not temp_file.is_file():
                        continue
                    try:
                        raw_val = float(temp_file.read_text(encoding="utf-8").strip())
                    except Exception:
                        continue
                    temp_c = raw_val / 1000.0 if abs(raw_val) >= 200.0 else raw_val
                    if temp_c <= 0.0 or temp_c > 150.0:
                        continue

                    ztype = zdir.name
                    type_file = zdir / "type"
                    if type_file.is_file():
                        try:
                            ztype = type_file.read_text(encoding="utf-8").strip() or zdir.name
                        except Exception:
                            pass

                    crit_c: Optional[float] = None
                    hot_c: Optional[float] = None
                    for tp_type_file in zdir.glob("trip_point_*_type"):
                        try:
                            tp_kind = tp_type_file.read_text(encoding="utf-8").strip().lower()
                            tp_temp_file = zdir / tp_type_file.name.replace("_type", "_temp")
                            if tp_temp_file.is_file():
                                tp_raw = float(tp_temp_file.read_text(encoding="utf-8").strip())
                                tp_c = tp_raw / 1000.0 if abs(tp_raw) >= 200.0 else tp_raw
                                if tp_kind == "critical":
                                    crit_c = tp_c
                                elif tp_kind in ("hot", "passive"):
                                    hot_c = tp_c if hot_c is None else min(hot_c, tp_c)
                        except Exception:
                            continue

                    z_state = _classify_zone(temp_c, crit_c, hot_c)
                    if severity_rank[z_state] > severity_rank[worst_state]:
                        worst_state = z_state
                    if max_temp_c is None or temp_c > max_temp_c:
                        max_temp_c = temp_c
                        max_zone_label = ztype

            if max_temp_c is None:
                hwmon_base = s_root / "class" / "hwmon"
                if hwmon_base.is_dir():
                    for inp in sorted(hwmon_base.glob("hwmon*/temp*_input")):
                        try:
                            raw_val = float(inp.read_text(encoding="utf-8").strip())
                            temp_c = raw_val / 1000.0 if abs(raw_val) >= 200.0 else raw_val
                            if 0.0 < temp_c <= 150.0:
                                z_state = _classify_zone(temp_c, None, None)
                                if severity_rank[z_state] > severity_rank[worst_state]:
                                    worst_state = z_state
                                if max_temp_c is None or temp_c > max_temp_c:
                                    max_temp_c = temp_c
                                    max_zone_label = inp.parent.name
                        except Exception:
                            continue
        except Exception as exc:
            logger.debug(f"Linux thermal scan error: {exc}")

        if max_temp_c is None:
            return ("NORMAL", "Linux virtualized/container host (no thermal_zone sensors)")
        return (worst_state, f"Linux {max_zone_label}={max_temp_c:.1f}C ({worst_state})")

    def get_memory_and_swap_linux(self, proc_root: Optional[Path] = None) -> Dict[str, float]:
        """Parse `/proc/meminfo` for host RAM, available reclaimable cache, and swap usage."""
        p_root = Path(proc_root) if proc_root else self._proc_root
        meminfo_path = p_root / "meminfo"
        fields_kb: Dict[str, int] = {}
        try:
            if meminfo_path.is_file():
                for line in meminfo_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(":")
                        try:
                            fields_kb[key] = int(parts[1])
                        except ValueError:
                            continue
        except Exception as exc:
            logger.debug(f"Failed to parse {meminfo_path}: {exc}")

        total_kb = fields_kb.get("MemTotal", 0)
        if "MemAvailable" in fields_kb:
            avail_kb = fields_kb["MemAvailable"]
        else:
            avail_kb = max(
                0,
                fields_kb.get("MemFree", 0)
                + fields_kb.get("Buffers", 0)
                + fields_kb.get("Cached", 0)
                + fields_kb.get("SReclaimable", 0)
                - fields_kb.get("Shmem", 0),
            )
        swap_total_kb = fields_kb.get("SwapTotal", 0)
        swap_free_kb = fields_kb.get("SwapFree", 0)
        swap_used_kb = max(0, swap_total_kb - swap_free_kb)

        total_gb = round((total_kb * 1024) / (1024**3), 2) if total_kb > 0 else 0.0
        free_gb = round((avail_kb * 1024) / (1024**3), 2) if total_kb > 0 else 0.0
        ram_pct = (
            round(max(0.0, min(100.0, ((total_kb - avail_kb) / total_kb) * 100.0)), 1)
            if total_kb > 0
            else 0.0
        )
        swap_used_gb = round((swap_used_kb * 1024) / (1024**3), 2)
        swap_total_gb = round((swap_total_kb * 1024) / (1024**3), 2)
        return {
            "total_ram_gb": total_gb,
            "free_ram_gb": free_gb,
            "ram_percent": ram_pct,
            "swap_used_gb": swap_used_gb,
            "swap_total_gb": swap_total_gb,
        }

    def get_cgroup_limits_linux(
        self,
        sys_root: Optional[Path] = None,
        proc_root: Optional[Path] = None,
    ) -> Dict[str, Optional[float]]:
        """
        Inspect Linux cgroups v2 (`memory.max`, `memory.current`, `memory.stat` `inactive_file`,
        `memory.swap.current`, `cpu.max`) with cgroup v1 fallback so Docker/devcontainer/CI
        memory ceilings clamp host `/proc/meminfo` accurately and prevent exit-137 OOM kills.
        """
        s_root = Path(sys_root) if sys_root else self._sys_root
        p_root = Path(proc_root) if proc_root else self._proc_root
        cg_base = s_root / "fs" / "cgroup"
        UNLIMITED_THRESHOLD = 1 << 60

        result: Dict[str, Optional[float]] = {
            "limit_gb": None,
            "used_gb": None,
            "free_gb": None,
            "swap_used_gb": None,
            "cpu_quota_cores": None,
        }
        if not cg_base.is_dir():
            return result

        # Resolve specific cgroup v2 directory from `/proc/self/cgroup` when outside private ns
        v2_dirs: List[Path] = []
        self_cg = p_root / "self" / "cgroup"
        if self_cg.is_file():
            try:
                for line in self_cg.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.startswith("0::"):
                        rel = line.split("0::", 1)[1].strip().lstrip("/")
                        if rel:
                            cand_dir = cg_base / rel
                            if cand_dir.is_dir():
                                v2_dirs.append(cand_dir)
            except Exception:
                pass
        v2_dirs.append(cg_base)

        limit_bytes: Optional[int] = None
        current_bytes: Optional[int] = None
        inactive_file_bytes = 0

        for cg_dir in v2_dirs:
            mem_max_file = cg_dir / "memory.max"
            if limit_bytes is None and mem_max_file.is_file():
                try:
                    raw_max = mem_max_file.read_text(encoding="utf-8").strip()
                    if raw_max and raw_max != "max":
                        val = int(raw_max)
                        if 0 < val < UNLIMITED_THRESHOLD:
                            limit_bytes = val
                except Exception:
                    pass

            mem_curr_file = cg_dir / "memory.current"
            if current_bytes is None and mem_curr_file.is_file():
                try:
                    current_bytes = max(0, int(mem_curr_file.read_text(encoding="utf-8").strip()))
                except Exception:
                    pass

            mem_stat_file = cg_dir / "memory.stat"
            if inactive_file_bytes == 0 and mem_stat_file.is_file():
                try:
                    for sline in mem_stat_file.read_text(encoding="utf-8", errors="replace").splitlines():
                        sp = sline.split()
                        if len(sp) >= 2 and sp[0] == "inactive_file":
                            inactive_file_bytes = max(0, int(sp[1]))
                            break
                except Exception:
                    pass

            swap_curr_file = cg_dir / "memory.swap.current"
            if result["swap_used_gb"] is None and swap_curr_file.is_file():
                try:
                    sw_b = max(0, int(swap_curr_file.read_text(encoding="utf-8").strip()))
                    result["swap_used_gb"] = round(sw_b / (1024**3), 2)
                except Exception:
                    pass

            cpu_max_file = cg_dir / "cpu.max"
            if result["cpu_quota_cores"] is None and cpu_max_file.is_file():
                try:
                    cparts = cpu_max_file.read_text(encoding="utf-8").strip().split()
                    if len(cparts) >= 2 and cparts[0] != "max":
                        quota = float(cparts[0])
                        period = float(cparts[1])
                        if quota > 0 and period > 0:
                            result["cpu_quota_cores"] = round(quota / period, 2)
                except Exception:
                    pass

        # Fallback to cgroup v1 (`/sys/fs/cgroup/memory/memory.limit_in_bytes`)
        if limit_bytes is None:
            v1_limit_file = cg_base / "memory" / "memory.limit_in_bytes"
            if v1_limit_file.is_file():
                try:
                    val = int(v1_limit_file.read_text(encoding="utf-8").strip())
                    if 0 < val < UNLIMITED_THRESHOLD:
                        limit_bytes = val
                except Exception:
                    pass
            v1_usage_file = cg_base / "memory" / "memory.usage_in_bytes"
            if current_bytes is None and v1_usage_file.is_file():
                try:
                    current_bytes = max(0, int(v1_usage_file.read_text(encoding="utf-8").strip()))
                except Exception:
                    pass

        if limit_bytes is not None:
            result["limit_gb"] = round(limit_bytes / (1024**3), 2)
            if current_bytes is not None:
                working_set = max(0, current_bytes - inactive_file_bytes)
                result["used_gb"] = round(working_set / (1024**3), 2)
                result["free_gb"] = round(max(0, limit_bytes - working_set) / (1024**3), 2)

        return result

    def get_load_avg_linux(self, proc_root: Optional[Path] = None) -> Tuple[float, float, float]:
        """Read 1m/5m/15m load averages from `/proc/loadavg` with `os.getloadavg()` fallback."""
        p_root = Path(proc_root) if proc_root else self._proc_root
        load_file = p_root / "loadavg"
        try:
            if load_file.is_file():
                parts = load_file.read_text(encoding="utf-8").strip().split()
                if len(parts) >= 3:
                    return (
                        round(float(parts[0]), 2),
                        round(float(parts[1]), 2),
                        round(float(parts[2]), 2),
                    )
        except Exception:
            pass
        try:
            l1, l5, l15 = os.getloadavg()
            return (round(l1, 2), round(l5, 2), round(l15, 2))
        except Exception:
            return (0.0, 0.0, 0.0)

    def get_power_state_linux(self, sys_root: Optional[Path] = None) -> Tuple[str, int, bool]:
        """Read Linux power source, battery %, and ACPI low-power profile from `/sys`."""
        s_root = Path(sys_root) if sys_root else self._sys_root
        ps_base = s_root / "class" / "power_supply"
        power_source = "AC Power"
        battery_pct = 100
        low_power_mode = False
        try:
            if ps_base.is_dir():
                for bat_dir in sorted(ps_base.glob("BAT*")):
                    cap_file = bat_dir / "capacity"
                    if cap_file.is_file():
                        try:
                            battery_pct = max(0, min(100, int(cap_file.read_text(encoding="utf-8").strip())))
                        except Exception:
                            pass
                    st_file = bat_dir / "status"
                    if st_file.is_file():
                        st_txt = st_file.read_text(encoding="utf-8").strip().lower()
                        if st_txt == "discharging":
                            power_source = "Battery"
                    break

            prof_file = s_root / "firmware" / "acpi" / "platform_profile"
            if prof_file.is_file():
                if prof_file.read_text(encoding="utf-8").strip().lower() == "low-power":
                    low_power_mode = True
        except Exception as exc:
            logger.debug(f"Linux power supply scan notice: {exc}")
        return (power_source, battery_pct, low_power_mode)

    def get_telemetry(self, force_refresh: bool = False) -> HardwareTelemetry:
        """Collect current system telemetry across thermals, memory, swap, cgroups, and load (cached 3.0s)."""
        now = time.time()
        if (
            not force_refresh
            and self._telemetry_cache is not None
            and (now - self._telemetry_cache_ts) < 3.0
        ):
            return self._telemetry_cache

        cgroup_limit_gb: Optional[float] = None
        cgroup_used_gb: Optional[float] = None
        cgroup_cpu_cores: Optional[float] = None

        if self.is_linux:
            thermal_state, thermal_raw = self.get_thermal_state_linux()
            linux_mem = self.get_memory_and_swap_linux()
            total_ram_gb = linux_mem["total_ram_gb"]
            free_ram_gb = linux_mem["free_ram_gb"]
            ram_percent = linux_mem["ram_percent"]
            swap_used_gb = linux_mem["swap_used_gb"]

            if total_ram_gb == 0.0 and PSUTIL_AVAILABLE:
                try:
                    mem = psutil.virtual_memory()
                    free_ram_gb = round(mem.available / (1024**3), 2)
                    total_ram_gb = round(mem.total / (1024**3), 2)
                    ram_percent = mem.percent
                except Exception as e:
                    logger.debug(f"psutil memory fallback error: {e}")

            cg = self.get_cgroup_limits_linux()
            cgroup_limit_gb = cg.get("limit_gb")
            cgroup_used_gb = cg.get("used_gb")
            cgroup_cpu_cores = cg.get("cpu_quota_cores")

            if cgroup_limit_gb is not None and (total_ram_gb == 0.0 or cgroup_limit_gb < total_ram_gb):
                total_ram_gb = cgroup_limit_gb
                cg_free = cg.get("free_gb")
                if cg_free is not None:
                    free_ram_gb = round(min(free_ram_gb, cg_free) if free_ram_gb > 0 else cg_free, 2)
                else:
                    free_ram_gb = round(min(free_ram_gb, cgroup_limit_gb), 2)
                if total_ram_gb > 0:
                    ram_percent = round(
                        max(0.0, min(100.0, ((total_ram_gb - free_ram_gb) / total_ram_gb) * 100.0)),
                        1,
                    )
            cg_swap = cg.get("swap_used_gb")
            if cg_swap is not None and cg_swap > swap_used_gb:
                swap_used_gb = cg_swap

            load_1m = self.get_load_avg_linux()[0]
            power_source, battery_pct, low_power_mode = self.get_power_state_linux()
        else:
            thermal_state, thermal_raw = self.get_thermal_state_macos()

            free_ram_gb = 0.0
            total_ram_gb = 0.0
            ram_percent = 0.0

            if PSUTIL_AVAILABLE:
                try:
                    mem = psutil.virtual_memory()
                    free_ram_gb = round(mem.available / (1024**3), 2)
                    total_ram_gb = round(mem.total / (1024**3), 2)
                    ram_percent = mem.percent
                except Exception as e:
                    logger.debug(f"psutil memory read error: {e}")

            if total_ram_gb == 0.0 and self.is_macos:
                try:
                    memsize_out = subprocess.check_output(
                        ["sysctl", "-n", "hw.memsize"], text=True, timeout=1.5
                    ).strip()
                    total_bytes = int(memsize_out)
                    total_ram_gb = round(total_bytes / (1024**3), 2)

                    vm_out = subprocess.check_output(["vm_stat"], text=True, timeout=1.5)
                    page_size = 16384
                    m_ps = re.search(r"page size of (\d+) bytes", vm_out)
                    if m_ps:
                        page_size = int(m_ps.group(1))

                    avail_pages = 0
                    for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
                        m_pg = re.search(rf"{key}:\s+(\d+)", vm_out)
                        if m_pg:
                            avail_pages += int(m_pg.group(1))

                    avail_bytes = avail_pages * page_size
                    free_ram_gb = round(avail_bytes / (1024**3), 2)
                    if total_bytes > 0:
                        ram_percent = round(((total_bytes - avail_bytes) / total_bytes) * 100.0, 1)
                except Exception as e:
                    logger.debug(f"macOS vm_stat fallback error: {e}")

            swap_used_gb, _ = self.get_swap_usage_macos()
            power_source, battery_pct, low_power_mode = self.get_power_state_macos()

            load_1m = 0.0
            try:
                load_1m = round(os.getloadavg()[0], 2)
            except Exception:
                pass

        # Evaluate safety
        is_safe = True
        reason = None
        os_label = "Linux" if self.is_linux else "macOS"

        eff_supervisor_load_ceiling = self.max_load_avg
        if thermal_state == "NORMAL":
            host_cores = float(os.cpu_count() or 10)
            if cgroup_cpu_cores is not None and cgroup_cpu_cores > 0:
                eff_cores = max(1.0, min(host_cores, cgroup_cpu_cores))
            else:
                eff_cores = host_cores
            eff_supervisor_load_ceiling = max(self.max_load_avg, max(14.0, eff_cores * 1.15))

        if thermal_state in ("SERIOUS", "CRITICAL") and self.pause_on_serious:
            is_safe = False
            reason = f"{os_label} thermal state is {thermal_state} ({thermal_raw})"
        elif free_ram_gb > 0 and free_ram_gb < self.min_free_ram_gb:
            is_safe = False
            ram_kind = "cgroup RAM" if cgroup_limit_gb is not None else "unified RAM"
            reason = (
                f"Free {ram_kind} ({free_ram_gb} GB) below threshold ({self.min_free_ram_gb} GB)"
            )
        elif load_1m > eff_supervisor_load_ceiling:
            is_safe = False
            reason = f"Load average ({load_1m}) exceeds ceiling ({eff_supervisor_load_ceiling:.1f})"
        elif low_power_mode:
            reason = f"{os_label} Low Power Mode active on {power_source} ({battery_pct}%) — GPU/CPU clocks throttled"

        telem = HardwareTelemetry(
            system=self._platform_system,
            thermal_state=thermal_state,
            thermal_raw=thermal_raw,
            free_ram_gb=free_ram_gb,
            total_ram_gb=total_ram_gb,
            ram_percent=ram_percent,
            load_avg_1m=load_1m,
            is_safe=is_safe,
            swap_used_gb=swap_used_gb,
            throttle_reason=reason,
            power_source=power_source,
            battery_pct=battery_pct,
            low_power_mode=low_power_mode,
            cgroup_limit_gb=cgroup_limit_gb,
            cgroup_used_gb=cgroup_used_gb,
        )
        self._telemetry_cache = telem
        self._telemetry_cache_ts = time.time()
        self.maybe_opportunistic_reap()
        return telem

    def maybe_opportunistic_reap(self, interval_sec: float = 60.0) -> Optional[Dict[str, Any]]:
        """
        Opportunistically trigger runaway reaper sweep if at least interval_sec has elapsed.
        Uses in-memory timestamp and /tmp/wbc_last_reap.ts to avoid redundant sweeps across processes.
        Bypassed during pytest runs to avoid unexpected side effects in unit tests.
        """
        if "PYTEST_CURRENT_TEST" in os.environ:
            return None
        now = time.time()
        last_ts = getattr(self, "_last_opportunistic_reap_ts", 0.0)
        if (now - last_ts) < interval_sec:
            return None

        reap_stamp = Path("/tmp/wbc_last_reap.ts")
        try:
            if reap_stamp.exists():
                mtime = reap_stamp.stat().st_mtime
                if (now - mtime) < interval_sec:
                    self._last_opportunistic_reap_ts = now
                    return None
            reap_stamp.write_text(str(now), encoding="utf-8")
        except Exception:
            pass

        self._last_opportunistic_reap_ts = now
        try:
            return self.reap_and_clamp_runaway_processes(dry_run=False, recycle_tachycardic=True)
        except Exception as e:
            logger.debug(f"Opportunistic reap error: {e}")
            return None

    def is_safe_to_run(self) -> bool:
        """Instant boolean check if safe to proceed with compute."""
        return self.get_telemetry().is_safe

    def get_cognitive_telemetry(
        self,
        conv_id: Optional[str] = None,
        brain_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Inspect the active Antigravity conversation's cognitive saturation, CSI,
        and latency inflation ratio using CognitiveThermalSentry.
        """
        try:
            from whole_body_compute.organs.cognitive_sentry import CognitiveThermalSentry

            sentry = CognitiveThermalSentry(brain_dir=brain_dir)
            metrics = sentry.inspect_conversation(conv_id=conv_id)
            if metrics:
                return metrics.to_dict()
        except NotImplementedError:
            try:
                from whole_body_compute.finops_audit import inspect_conversation_lightweight

                lw = inspect_conversation_lightweight(conv_id=conv_id, brain_dir=brain_dir)
                if lw:
                    return lw
            except Exception:
                pass
        except Exception as e:
            logger.debug("Cognitive telemetry read notice: %s", e)

        return {
            "conv_id": "none",
            "short_id": "none",
            "steps": 0,
            "csi": 0.0,
            "health_tier": "GREEN",
            "compact_size_kb": 0.0,
            "full_size_kb": 0.0,
            "early_step_sec": 2.8,
            "recent_step_sec": 2.8,
            "latency_inflation_ratio": 1.0,
        }

    def get_model_requirement(self, model_name: str) -> Dict[str, Any]:
        """Resolve hardware headroom requirement for a specific model name."""
        key = (model_name or "").lower().strip()
        if key in DEFAULT_MODEL_REQUIREMENTS:
            return dict(DEFAULT_MODEL_REQUIREMENTS[key])

        # Heuristic resolution based on parameter tier in model string
        if any(token in key for token in ("70b", "72b")):
            return {
                "min_free_ram_gb": 46.0,
                "max_load_avg": 8.0,
                "disallow_swap": True,
                "recommended_fallback": "gemma4-26b",
            }
        elif any(token in key for token in ("32b", "34b", "27b", "26b")):
            return {
                "min_free_ram_gb": 18.0,
                "max_load_avg": 14.0,
                "disallow_swap": True,
                "recommended_fallback": "gemma2:2b",
            }
        elif any(token in key for token in ("14b", "13b", "12b")):
            return {
                "min_free_ram_gb": 11.0,
                "max_load_avg": 12.0,
                "disallow_swap": False,
                "recommended_fallback": "qwen2.5-coder:7b",
            }
        elif any(token in key for token in ("7b", "8b", "9b")):
            return {
                "min_free_ram_gb": 6.5,
                "max_load_avg": 12.0,
                "disallow_swap": False,
                "recommended_fallback": "qwen2.5-coder:1.5b",
            }
        elif any(token in key for token in ("embed", "embedding")):
            return {
                "min_free_ram_gb": 0.8,
                "max_load_avg": 14.0,
                "disallow_swap": False,
                "recommended_fallback": None,
                "is_tier0": True,
            }
        elif any(token in key for token in ("2b", "3b", "1.5b", "1b", "0.8b")):
            return {
                "min_free_ram_gb": 2.5,
                "max_load_avg": 14.0,
                "disallow_swap": False,
                "recommended_fallback": None,
            }

        return {
            "min_free_ram_gb": self.min_free_ram_gb,
            "max_load_avg": self.max_load_avg,
            "disallow_swap": False,
            "recommended_fallback": None,
        }

    def record_gate_event(
        self,
        requested_model: str,
        resolved_model: Optional[str],
        status: str,
        reason: str,
        free_ram_gb: float = 0.0,
        required_ram_gb: float = 0.0,
        reclaimed_gb: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Record the latest model ResourceGate decision in memory and persist it atomically
        to `/tmp/wbc_last_gate_event.json` so the HUD and telemetry server can display it.
        """
        norm_status = status.upper()
        if norm_status not in ("APPROVED", "AUTO_SWEPT", "FALLBACK", "BLOCKED"):
            norm_status = "BLOCKED"

        event: Dict[str, Any] = {
            "timestamp": round(time.time(), 3),
            "requested_model": requested_model,
            "resolved_model": resolved_model,
            "status": norm_status,
            "reason": reason,
            "free_ram_gb": round(float(free_ram_gb), 2),
            "required_ram_gb": round(float(required_ram_gb), 2),
            "reclaimed_gb": round(float(reclaimed_gb), 2),
        }
        self._last_gate_event = event
        try:
            self._gate_event_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._gate_event_path.with_suffix(f".{os.getpid()}.tmp")
            tmp_path.write_text(json.dumps(event), encoding="utf-8")
            os.replace(tmp_path, self._gate_event_path)
        except Exception as exc:
            logger.debug(f"Failed to persist gate event to {self._gate_event_path}: {exc}")
        return event

    def get_last_gate_event(self) -> Dict[str, Any]:
        """Return the latest ResourceGate event from memory or `/tmp/wbc_last_gate_event.json`."""
        try:
            if self._gate_event_path.is_file():
                raw = self._gate_event_path.read_text(encoding="utf-8")
                disk_ev = json.loads(raw)
                if isinstance(disk_ev, dict) and "status" in disk_ev:
                    mem_ts = float((self._last_gate_event or {}).get("timestamp", 0.0))
                    disk_ts = float(disk_ev.get("timestamp", 0.0))
                    if self._last_gate_event is None or disk_ts >= mem_ts:
                        self._last_gate_event = disk_ev
        except Exception as exc:
            logger.debug(f"Failed to read gate event from {self._gate_event_path}: {exc}")

        if self._last_gate_event is not None:
            return dict(self._last_gate_event)

        return {
            "timestamp": 0.0,
            "requested_model": None,
            "resolved_model": None,
            "status": "NOMINAL",
            "reason": "NOMINAL (No blocks)",
            "free_ram_gb": 0.0,
            "required_ram_gb": 0.0,
            "reclaimed_gb": 0.0,
        }

    def _notify_hud_gate_event(
        self,
        requested_model: str,
        resolved_model: Optional[str],
        status: str,
        reason: str,
    ) -> None:
        """Notify an already-instantiated UnifiedDynamicIslandHUD of a ResourceGate action."""
        try:
            from whole_body_compute.ui.unified_hud import UnifiedDynamicIslandHUD

            hud = getattr(UnifiedDynamicIslandHUD, "_instance", None)
            if hud is not None and hasattr(hud, "show_resource_gate_alert"):
                hud.show_resource_gate_alert(
                    requested_model=requested_model,
                    resolved_model=resolved_model,
                    status=status,
                    reason=reason,
                )
        except Exception as exc:
            logger.debug(f"HUD gate alert skipped: {exc}")

    @staticmethod
    def _is_endpoint_online(port: int, host: str = "127.0.0.1", timeout: float = 0.02) -> bool:
        """Fast <20ms TCP socket probe to check if a local model endpoint is running."""
        import socket

        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except Exception:
            return False

    def is_warm_endpoint_online(self, model_name: str) -> bool:
        """Check if local inference server hosting model_name is already warm and running."""
        m_lower = (model_name or "").lower()
        if "gemma4" in m_lower or "26b" in m_lower:
            return self._is_endpoint_online(9380)
        if "voicefi-coder" in m_lower or "2b" in m_lower or "mlx" in m_lower:
            return self._is_endpoint_online(9379)
        if "qwen" in m_lower or "llama" in m_lower or "ollama" in m_lower:
            return self._is_endpoint_online(11434)
        return False

    def get_effective_max_load(
        self,
        model_name: str,
        telem: Optional[HardwareTelemetry] = None,
    ) -> float:
        """
        Dynamically scale the CPU load ceiling for warm / mid-tier models on multi-core Macs
        when thermal state is NORMAL and RAM headroom is sufficient.
        """
        req = self.get_model_requirement(model_name)
        base_max_load = float(req.get("max_load_avg", self.max_load_avg))
        if telem is None:
            telem = self.get_telemetry()
        req_ram = float(req.get("min_free_ram_gb", self.min_free_ram_gb))

        # When Thermal State == NORMAL and unified memory headroom is sufficient:
        # Scale dynamically with os.cpu_count() or check if warm local model endpoint is online
        if getattr(telem, "thermal_state", "NORMAL") == "NORMAL" and getattr(telem, "free_ram_gb", 0.0) >= req_ram:
            is_warm_or_mid_tier = (
                any(
                    t in model_name.lower()
                    for t in (
                        "gemma4",
                        "gemma4-26b",
                        "voicefi-coder-mlx",
                        "qwen2.5-coder",
                        "26b",
                        "32b",
                    )
                )
                or self.is_warm_endpoint_online(model_name)
            )
            if is_warm_or_mid_tier:
                cpu_cores = os.cpu_count() or 10
                dynamic_ceiling = max(14.0, cpu_cores * 1.15)
                return max(base_max_load, dynamic_ceiling)

        return base_max_load

    def can_load_model(
        self,
        model_name: str,
        is_tier0: Optional[bool] = None,
        record_event: bool = True,
        telem: Optional[HardwareTelemetry] = None,
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Evaluate if system has sufficient hardware and unified RAM headroom to load a model.
        Returns: (can_load, reason_or_status, alert_options_dict)
        Tier 0 models (reflex routing, EmbeddingGemma 2, TEV 0.8B) receive a thermal reflex exemption
        under 'SERIOUS' thermal pressure to allow rapid dispatch and user notifications.
        """
        if telem is None:
            telem = self.get_telemetry()
        req = self.get_model_requirement(model_name)
        required_ram = req["min_free_ram_gb"]
        fallback = req.get("recommended_fallback") or "gemma2:2b"
        tier0_exempt = is_tier0 if is_tier0 is not None else req.get("is_tier0", False)

        # Check 1: Thermal Pressure Guard (Tier 0 allowed under SERIOUS, blocked only on CRITICAL)
        thermal_blocked = (
            telem.thermal_state == "CRITICAL"
            if tier0_exempt
            else (telem.thermal_state in ("SERIOUS", "CRITICAL") and self.pause_on_serious)
        )
        if thermal_blocked:
            reason = (
                f"Blocked {model_name} spin-up: macOS thermal state is {telem.thermal_state}. "
                "Pausing to allow hardware cooling."
            )
            alert = {
                "model": model_name,
                "cause": "thermal",
                "thermal_state": telem.thermal_state,
                "required_ram_gb": required_ram,
                "available_ram_gb": telem.free_ram_gb,
                "swap_used_gb": telem.swap_used_gb,
                "fallback_model": fallback,
                "options": [
                    {"id": "cloud", "label": "Route to Cloud Frontier (Antigravity/Gemini)"},
                    {"id": "wait", "label": "Wait for hardware to cool down"},
                ],
            }
            if record_event:
                self.record_gate_event(
                    requested_model=model_name,
                    resolved_model=None,
                    status="BLOCKED",
                    reason=reason,
                    free_ram_gb=telem.free_ram_gb,
                    required_ram_gb=required_ram,
                )
            return (False, reason, alert)

        # Check 2: Physical Unified Memory Headroom
        if telem.free_ram_gb < required_ram:
            reason = (
                f"Blocked {model_name} spin-up: Requires {required_ram:.1f} GB free unified RAM, "
                f"but only {telem.free_ram_gb:.1f} GB available (Swap in use: {telem.swap_used_gb:.1f} GB). "
                "Refusing load to prevent macOS swap thrashing."
            )
            alert = {
                "model": model_name,
                "cause": "insufficient_ram",
                "required_ram_gb": required_ram,
                "available_ram_gb": telem.free_ram_gb,
                "swap_used_gb": telem.swap_used_gb,
                "fallback_model": fallback,
                "options": [
                    {"id": "cloud", "label": "Route to Cloud Frontier (Antigravity/Gemini)"},
                    {"id": "fallback", "label": f"Use Mid-Tier Local Model ({fallback})"},
                    {"id": "free_ram", "label": "Quit background apps to free RAM and retry"},
                ],
            }
            if record_event:
                self.record_gate_event(
                    requested_model=model_name,
                    resolved_model=None,
                    status="BLOCKED",
                    reason=reason,
                    free_ram_gb=telem.free_ram_gb,
                    required_ram_gb=required_ram,
                )
            return (False, reason, alert)

        # Check 3: Active Swap Restriction for Heavy Frontier Models
        # Distinguish dormant swap from active swap pressure: only block when free physical RAM
        # lacks >= 8.0 GB extra headroom above required_ram.
        if (
            req.get("disallow_swap")
            and telem.swap_used_gb > 8.0
            and telem.free_ram_gb < (required_ram + 8.0)
        ):
            reason = (
                f"Blocked {model_name} spin-up: Active disk swap ({telem.swap_used_gb:.1f} GB) "
                "indicates high background memory pressure. Refusing frontier 70B load."
            )
            alert = {
                "model": model_name,
                "cause": "high_swap_pressure",
                "required_ram_gb": required_ram,
                "available_ram_gb": telem.free_ram_gb,
                "swap_used_gb": telem.swap_used_gb,
                "fallback_model": fallback,
                "options": [
                    {"id": "cloud", "label": "Route to Cloud Frontier (Antigravity/Gemini)"},
                    {"id": "fallback", "label": f"Use Mid-Tier Local Model ({fallback})"},
                    {"id": "free_ram", "label": "Purge memory/swap and retry"},
                ],
            }
            if record_event:
                self.record_gate_event(
                    requested_model=model_name,
                    resolved_model=None,
                    status="BLOCKED",
                    reason=reason,
                    free_ram_gb=telem.free_ram_gb,
                    required_ram_gb=required_ram,
                )
            return (False, reason, alert)

        # Check 4: Host Load Average
        eff_max_load = self.get_effective_max_load(model_name, telem)
        if telem.load_avg_1m > eff_max_load:
            reason = (
                f"Blocked {model_name} spin-up: 1m CPU load ({telem.load_avg_1m}) "
                f"exceeds ceiling ({eff_max_load:.1f})."
            )
            alert = {
                "model": model_name,
                "cause": "high_cpu_load",
                "load_avg_1m": telem.load_avg_1m,
                "required_ram_gb": required_ram,
                "available_ram_gb": telem.free_ram_gb,
                "swap_used_gb": telem.swap_used_gb,
                "fallback_model": fallback,
                "options": [
                    {"id": "cloud", "label": "Route to Cloud Frontier (Antigravity/Gemini)"},
                    {"id": "wait", "label": "Wait for CPU load to settle"},
                ],
            }
            if record_event:
                self.record_gate_event(
                    requested_model=model_name,
                    resolved_model=None,
                    status="BLOCKED",
                    reason=reason,
                    free_ram_gb=telem.free_ram_gb,
                    required_ram_gb=required_ram,
                )
            return (False, reason, alert)

        return (True, f"Nominal: {model_name} approved for on-device load.", None)

    def _auto_sweep_excess_models(
        self,
        running_models: List[Dict[str, Any]],
        target_model: str = "",
        needed_gb: float = 0.0,
        base_url: str = "http://127.0.0.1:11434",
    ) -> float:
        """
        Rank running idle Ollama models by eviction_protection_score ascending.
        Evict low-pain models first until needed_gb is reclaimed (or all non-tier0 models unloaded).
        Preserves high-pain and pinned models whenever possible.
        """
        import urllib.request

        target_clean = (target_model or "").replace("ollama/", "").strip().lower()
        candidates: List[Dict[str, Any]] = []

        for m in running_models:
            if not isinstance(m, dict):
                continue
            m_name = str(m.get("name") or m.get("model") or "").strip()
            if not m_name or m_name.lower() == target_clean:
                continue
            m_req = self.get_model_requirement(m_name)
            if m_req.get("is_tier0", False):
                continue

            policy = self.get_adaptive_keep_alive(m_name)
            protection_score = float(policy.get("eviction_protection_score", 0.0))
            is_pinned = bool(policy.get("should_pin_warm", False))
            candidates.append({
                "name": m_name,
                "dict": m,
                "score": protection_score,
                "is_pinned": is_pinned,
                "req": m_req,
            })

        # Sort ascending by eviction_protection_score, unpinned first
        candidates.sort(key=lambda c: (1 if c["is_pinned"] else 0, c["score"]))

        reclaimed_gb = 0.0
        for cand in candidates:
            m_name = cand["name"]
            m = cand["dict"]
            m_req = cand["req"]
            size_bytes = float(m.get("size") or m.get("size_vram") or 0.0)
            model_ram_gb = (size_bytes / (1024**3)) if size_bytes > 0 else (float(m_req.get("min_free_ram_gb", 2.0)) * 0.7)

            try:
                unload_payload = json.dumps({"model": m_name, "keep_alive": 0}).encode("utf-8")
                u_req = urllib.request.Request(
                    f"{base_url}/api/generate",
                    data=unload_payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(u_req, timeout=2.0) as u_resp:
                    u_resp.read()
                reclaimed_gb += model_ram_gb
            except Exception:
                try:
                    subprocess.run(
                        ["ollama", "stop", m_name],
                        capture_output=True,
                        timeout=2.0,
                    )
                    reclaimed_gb += model_ram_gb
                except Exception:
                    pass

            if needed_gb > 0.0 and reclaimed_gb >= needed_gb:
                break

        return reclaimed_gb

    def try_auto_sweep_for_model(
        self,
        model_name: str,
        required_gb: float,
        current_free_gb: float,
    ) -> float:
        """
        Attempt to reclaim Unified RAM for `model_name` by unloading non-Tier-0 idle models
        from Ollama (`/api/ps` -> `keep_alive: 0`) ranked by eviction_protection_score ascending
        and running a Selective Vascular Perfusion sweep.
        Returns the total reclaimed RAM in GB.
        """
        import urllib.request

        ollama_reclaimed_gb = 0.0
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

        # 1. Inspect and unload idle non-Tier-0 Ollama models via /api/ps ranked by eviction protection
        try:
            req = urllib.request.Request(f"{base_url}/api/ps", method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                ps_data = json.loads(resp.read().decode("utf-8"))
                running_models = ps_data.get("models", []) if isinstance(ps_data, dict) else []
            ollama_reclaimed_gb = self._auto_sweep_excess_models(
                running_models=running_models,
                target_model=model_name,
                needed_gb=max(0.0, required_gb - current_free_gb),
                base_url=base_url,
            )
        except Exception as exc:
            logger.debug(f"Auto-sweep Ollama /api/ps check skipped: {exc}")

        # 2. Run Selective Vascular Perfusion sweep for specialist MLX/llama-server processes
        vascular_reclaimed_gb = 0.0
        try:
            sweep_res = self.run_vascular_perfusion_sweep(dry_run=False, settle_sec=0.3)
            vascular_reclaimed_gb = float(sweep_res.get("reclaimed_ram_gb", 0.0))
        except Exception as exc:
            logger.debug(f"Auto-sweep vascular perfusion skipped: {exc}")

        # 3. Measure actual free RAM delta if available
        measured_delta_gb = 0.0
        try:
            self._telemetry_cache = None
            after_telem = self.get_telemetry()
            measured_delta_gb = max(0.0, float(after_telem.free_ram_gb) - float(current_free_gb))
        except Exception:
            pass

        reclaimed = max(ollama_reclaimed_gb + vascular_reclaimed_gb, measured_delta_gb)
        return round(reclaimed, 2)

    def resolve_loadable_model(
        self,
        model_name: str,
        auto_sweep: bool = True,
        allow_fallback: bool = True,
    ) -> Tuple[bool, str, str]:
        """
        3-Layer ResourceGate Resolution for `model_name`:
          1. Direct check via `can_load_model(model_name)`. If approved, returns `(True, model_name, msg)`.
          2. Auto-Sweep: If blocked due to insufficient RAM and `auto_sweep=True`, runs
             `try_auto_sweep_for_model(...)` and re-checks `can_load_model(model_name)`.
          3. Auto-Fallback Cascade: If still blocked and `allow_fallback=True`, walks the
             `recommended_fallback` chain in `DEFAULT_MODEL_REQUIREMENTS` with cycle protection.
          4. Otherwise records `BLOCKED` and returns `(False, model_name, reason)`.
        """
        req = self.get_model_requirement(model_name)
        req_ram = float(req.get("min_free_ram_gb", self.min_free_ram_gb))

        approved, msg, alert = self.can_load_model(model_name)
        avail_ram = (
            float(alert.get("available_ram_gb", 0.0))
            if isinstance(alert, dict) and "available_ram_gb" in alert
            else self._safe_free_ram_gb()
        )
        if isinstance(alert, dict) and "required_ram_gb" in alert:
            req_ram = float(alert["required_ram_gb"])

        if approved:
            self.record_gate_event(
                requested_model=model_name,
                resolved_model=model_name,
                status="APPROVED",
                reason=msg,
                free_ram_gb=avail_ram,
                required_ram_gb=req_ram,
                reclaimed_gb=0.0,
            )
            return (True, model_name, msg)

        reclaimed_gb = 0.0
        cause = alert.get("cause", "") if isinstance(alert, dict) else ""
        is_ram_blocked = (
            cause == "insufficient_ram"
            or "free unified ram" in msg.lower()
            or "insufficient" in msg.lower()
            or ("ram" in msg.lower() and "thermal" not in msg.lower() and "load" not in msg.lower())
        )

        # Layer 2: Auto-sweep idle specialist models & re-evaluate
        if is_ram_blocked and auto_sweep:
            reclaimed_gb = self.try_auto_sweep_for_model(model_name, req_ram, avail_ram)
            approved_after, msg_after, alert_after = self.can_load_model(model_name)
            if isinstance(alert_after, dict) and "available_ram_gb" in alert_after:
                avail_ram = float(alert_after["available_ram_gb"])
            elif reclaimed_gb > 0:
                avail_ram = round(avail_ram + reclaimed_gb, 2)

            if approved_after:
                swept_msg = (
                    f"Auto-swept {reclaimed_gb:.1f} GB idle model RAM to unlock {model_name}. {msg_after}"
                )
                self.record_gate_event(
                    requested_model=model_name,
                    resolved_model=model_name,
                    status="AUTO_SWEPT",
                    reason=swept_msg,
                    free_ram_gb=avail_ram,
                    required_ram_gb=req_ram,
                    reclaimed_gb=reclaimed_gb,
                )
                self._notify_hud_gate_event(model_name, model_name, "AUTO_SWEPT", swept_msg)
                return (True, model_name, swept_msg)
            msg = msg_after
            alert = alert_after

        # Layer 3: Walk the recommended_fallback cascade with visited-set cycle guard
        if allow_fallback:
            visited: set[str] = {model_name.lower().strip()}
            curr = model_name
            while True:
                curr_req = self.get_model_requirement(curr)
                fallback = curr_req.get("recommended_fallback")
                if not fallback:
                    break
                fb_key = fallback.lower().strip()
                if fb_key in visited:
                    break
                visited.add(fb_key)

                fb_ok, fb_msg, _ = self.can_load_model(fallback)
                if fb_ok:
                    fallback_msg = (
                        f"Auto-fallback: '{model_name}' blocked ({msg}); cascaded to '{fallback}' ({fb_msg})"
                    )
                    self.record_gate_event(
                        requested_model=model_name,
                        resolved_model=fallback,
                        status="FALLBACK",
                        reason=fallback_msg,
                        free_ram_gb=avail_ram,
                        required_ram_gb=req_ram,
                        reclaimed_gb=reclaimed_gb,
                    )
                    self._notify_hud_gate_event(model_name, fallback, "FALLBACK", fallback_msg)
                    return (True, fallback, fallback_msg)
                curr = fallback

        # All layers exhausted: record final BLOCKED state
        self.record_gate_event(
            requested_model=model_name,
            resolved_model=None,
            status="BLOCKED",
            reason=msg,
            free_ram_gb=avail_ram,
            required_ram_gb=req_ram,
            reclaimed_gb=reclaimed_gb,
        )
        self._notify_hud_gate_event(model_name, None, "BLOCKED", msg)
        return (False, model_name, msg)

    def _safe_free_ram_gb(self) -> float:
        try:
            return float(self.get_telemetry().free_ram_gb)
        except Exception:
            return 0.0

    def get_model_gate_matrix(self, reclaimable_gb: float = 0.0) -> Dict[str, Any]:
        """
        Compute live readiness matrix across Tier-0 (1.5B–2B), Mid-Tier (26B), and Frontier (70B)
        models for Organ 01 HUD display without mutating the persistent `last_gate_event`.
        """
        telem = self.get_telemetry()
        reclaim_val = max(0.0, float(reclaimable_gb or 0.0))

        tiers = [
            ("tier0", "gemma2:2b", "1.5B–2B"),
            ("mid", "gemma4-26b", "26B"),
            ("frontier", "llama3.3:70b", "70B"),
        ]
        matrix: Dict[str, Any] = {}

        for tier_key, model_id, label in tiers:
            req = self.get_model_requirement(model_id)
            req_ram = float(req.get("min_free_ram_gb", self.min_free_ram_gb))
            disallow_swap = bool(req.get("disallow_swap", False))
            fallback = req.get("recommended_fallback")
            tier0_exempt = bool(req.get("is_tier0", False)) or tier_key == "tier0"

            thermal_blocked = (
                telem.thermal_state == "CRITICAL"
                if tier0_exempt
                else (telem.thermal_state in ("SERIOUS", "CRITICAL") and self.pause_on_serious)
            )
            swap_blocked = (
                disallow_swap
                and telem.swap_used_gb > 8.0
                and telem.free_ram_gb < (req_ram + 8.0)
            )
            eff_max_load = self.get_effective_max_load(model_id, telem)
            load_blocked = telem.load_avg_1m > eff_max_load
            ram_ok = telem.free_ram_gb >= req_ram
            ram_sweepable = (not ram_ok) and ((telem.free_ram_gb + reclaim_val) >= req_ram)

            if thermal_blocked:
                status = "BLOCKED"
                reason = f"Thermal {telem.thermal_state} (cooling pause)"
            elif load_blocked:
                status = "BLOCKED"
                reason = f"1m load {telem.load_avg_1m} > {eff_max_load:.1f}"
            elif ram_ok and not swap_blocked:
                status = "READY"
                reason = f"Ready ({telem.free_ram_gb:.1f} GB free / {req_ram:.1f} GB req)"
            elif ram_sweepable and not swap_blocked:
                status = "SWEEPABLE"
                reason = (
                    f"Sweepable: {telem.free_ram_gb:.1f} GB free + {reclaim_val:.1f} GB reclaimable "
                    f">= {req_ram:.1f} GB req"
                )
            elif (ram_ok or ram_sweepable) and swap_blocked:
                status = "SWEEPABLE"
                reason = (
                    f"Swap pressure ({telem.swap_used_gb:.1f} GB > 8.0 GB cap); "
                    f"free RAM {telem.free_ram_gb:.1f} GB / {req_ram:.1f} GB req"
                )
            else:
                status = "BLOCKED"
                reason = (
                    f"Needs {req_ram:.1f} GB free RAM (only {telem.free_ram_gb:.1f} GB available"
                    + (f", swap {telem.swap_used_gb:.1f} GB" if swap_blocked else "")
                    + ")"
                )

            matrix[tier_key] = {
                "model": model_id,
                "label": label,
                "status": status,
                "required_gb": round(req_ram, 1),
                "free_gb": round(telem.free_ram_gb, 1),
                "swap_gb": round(telem.swap_used_gb, 1),
                "disallow_swap": disallow_swap,
                "reason": reason,
                "fallback": fallback,
            }

        matrix["reclaimable_gb"] = round(reclaim_val, 2)
        matrix["last_gate_event"] = self.get_last_gate_event()
        return matrix

    def assert_can_load(self, model_name: str) -> None:
        """
        Assert that a model can be safely loaded.
        Raises ResourceGateError if hardware headroom or memory is insufficient.
        """
        approved, reason, alert = self.can_load_model(model_name)
        if not approved:
            raise ResourceGateError(reason, alert)

    def format_resource_alert(self, alert_data: Dict[str, Any]) -> str:
        """Format a human-readable developer alert with actionable choices."""
        model = alert_data.get("model", "Model")
        req_ram = alert_data.get("required_ram_gb", 0.0)
        avail_ram = alert_data.get("available_ram_gb", 0.0)
        swap_gb = alert_data.get("swap_used_gb", 0.0)
        fallback = alert_data.get("fallback_model", "gemma4-26b")

        lines = [
            f"\n⚠️  [VoiceFi ResourceGate] Memory Protection Alert for `{model}`",
            "=" * 65,
            f"  • Required Headroom:  {req_ram:.1f} GB Free Unified RAM",
            f"  • Currently Available:{avail_ram:.1f} GB Free Unified RAM",
            f"  • Active Disk Swap:   {swap_gb:.1f} GB in use",
            "-" * 65,
            "  Action Options:",
            "  [1] ☁️  Route to Cloud Frontier (Antigravity/Gemini) - Zero local RAM",
            f"  [2] 💻 Use Mid-Tier Local Model ({fallback}) - Fits safely in RAM",
            "  [3] 🛑 Free RAM (quit background apps/VMs) and Retry",
            "=" * 65,
        ]
        return "\n".join(lines)

    def wait_if_throttled(
        self,
        poll_interval: float = 3.0,
        max_wait: float = 60.0,
        on_pause_cb: Optional[Callable[[HardwareTelemetry], None]] = None,
    ) -> bool:
        """
        Block and sleep while thermals or memory are unsafe.
        Returns True if recovered and safe, False if timed out waiting.
        """
        start_time = time.time()
        while time.time() - start_time < max_wait:
            telem = self.get_telemetry()
            if telem.is_safe:
                return True

            logger.warning(
                f"[ThermalSupervisor] Throttling compute: {telem.throttle_reason}. Pausing {poll_interval}s..."
            )
            if on_pause_cb:
                try:
                    on_pause_cb(telem)
                except Exception:
                    pass

            time.sleep(poll_interval)

        return False

    def guarded(self, min_free_ram_gb: Optional[float] = None) -> Callable:
        """Decorator to wrap any local inference function with automatic backoff."""

        def decorator(fn: Callable) -> Callable:
            @wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                saved_min = self.min_free_ram_gb
                if min_free_ram_gb is not None:
                    self.min_free_ram_gb = min_free_ram_gb
                try:
                    self.wait_if_throttled()
                    return fn(*args, **kwargs)
                finally:
                    self.min_free_ram_gb = saved_min

            return wrapper

        return decorator

    # -----------------------------------------------------------------------
    # Physiological RAM Hygiene: Selective Vascular Perfusion & Cephalic Pre-Warm
    # -----------------------------------------------------------------------

    def get_resident_model_processes(self) -> Dict[str, Any]:
        """
        Inspect resident local model processes in Unified Memory and classify them
        into physiological perfusion tiers:
          - tier0_autonomic: Always resident (voicefi tray, brainstem :9379 worker)
          - evictable_specialist: On-demand cortical/muscle servers (mlx_lm.server, llama-server)
        """
        autonomic: list[Dict[str, Any]] = []
        evictable: list[Dict[str, Any]] = []
        tachycardic: list[Dict[str, Any]] = []
        large_candidates: list[Dict[str, Any]] = []
        reclaimable_mb = 0.0
        autonomic_mb = 0.0

        ps_succeeded = False
        if not (self.is_linux and self._proc_root != Path("/proc")):
            try:
                res = subprocess.run(
                    ["ps", "-ax", "-o", "pid=,pcpu=,rss=,etime=,command="],
                    capture_output=True,
                    text=True,
                    timeout=1.5,
                )
                if not isinstance(res.returncode, int) or res.returncode == 0:
                    ps_succeeded = True
                    for line in res.stdout.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(None, 4)
                        if len(parts) < 5:
                            continue
                        try:
                            pid = int(parts[0])
                            pcpu = float(parts[1])
                            rss_mb = round(int(parts[2]) / 1024.0, 1)
                        except ValueError:
                            continue
                        etime = parts[3]
                        cmd = parts[4]

                        if "grep" in cmd or pid == os.getpid():
                            continue

                        if "mlx_lm.server" in cmd or "llama-server" in cmd or "vllm.entrypoints" in cmd:
                            label = (
                                "mlx_lm.server"
                                if "mlx_lm.server" in cmd
                                else ("vllm" if "vllm.entrypoints" in cmd else "llama-server")
                            )
                            evictable.append(
                                {
                                    "pid": pid,
                                    "name": label,
                                    "tier": "tier2_heavy_specialist" if "mlx_lm.server" in cmd or "vllm" in label else "tier1_working_cortex",
                                    "cpu_pct": pcpu,
                                    "rss_mb": rss_mb,
                                    "rss_gb": round(rss_mb / 1024.0, 2),
                                    "elapsed": etime,
                                    "command": cmd[:120],
                                }
                            )
                            reclaimable_mb += rss_mb
                        elif "voicefi tray" in cmd or "mlx_metal_worker.py" in cmd:
                            label = "voicefi tray" if "voicefi tray" in cmd else "mlx_metal_worker (:9379)"
                            elapsed_sec = self._parse_etime_seconds(etime)
                            is_tachycardic = (elapsed_sec >= 120 and pcpu >= 120.0) or ("voicefi tray" in cmd and rss_mb >= 4096.0)
                            entry = {
                                "pid": pid,
                                "name": label,
                                "tier": "tier0_autonomic",
                                "cpu_pct": pcpu,
                                "rss_mb": rss_mb,
                                "rss_gb": round(rss_mb / 1024.0, 2),
                                "elapsed": etime,
                                "is_tachycardic": is_tachycardic,
                                "command": cmd[:120],
                            }
                            autonomic.append(entry)
                            if is_tachycardic:
                                tachycardic.append(entry)
                            autonomic_mb += rss_mb

                        # Collect candidate for large_processes radar (>= 500 MB or >= 20% CPU)
                        is_excluded_radar = (
                            "Antigravity.app" in cmd
                            or ("whole_body_compute.cli" in cmd and "telemetry" not in cmd)
                        )
                        if not is_excluded_radar:
                            if rss_mb >= 500.0 or pcpu >= 20.0:
                                proc_label = self._extract_process_label(cmd)
                                large_candidates.append(
                                    {
                                        "pid": pid,
                                        "name": proc_label,
                                        "cpu_pct": pcpu,
                                        "rss_mb": rss_mb,
                                        "rss_gb": round(rss_mb / 1024.0, 2),
                                        "elapsed": etime,
                                        "command": cmd[:100],
                                    }
                                )
            except Exception as e:
                logger.debug(f"Failed to scan resident model processes via ps: {e}")

        # Linux /proc/[pid] fallback for slim containers without `procps` (`ps`)
        if not ps_succeeded and self.is_linux and self._proc_root.is_dir():
            try:
                for pdir in self._proc_root.iterdir():
                    if not pdir.name.isdigit():
                        continue
                    pid = int(pdir.name)
                    if pid == os.getpid():
                        continue
                    cmdline_file = pdir / "cmdline"
                    status_file = pdir / "status"
                    if not cmdline_file.is_file():
                        continue
                    raw_cmd = cmdline_file.read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
                    if not raw_cmd:
                        continue
                    rss_kb = 0
                    if status_file.is_file():
                        for sline in status_file.read_text(encoding="utf-8", errors="replace").splitlines():
                            if sline.startswith("VmRSS:"):
                                sp = sline.split()
                                if len(sp) >= 2 and sp[1].isdigit():
                                    rss_kb = int(sp[1])
                                break
                    rss_mb = round(rss_kb / 1024.0, 1)
                    if "llama-server" in raw_cmd or "vllm" in raw_cmd or "ollama runner" in raw_cmd:
                        label = "vllm" if "vllm" in raw_cmd else ("ollama-runner" if "ollama runner" in raw_cmd else "llama-server")
                        evictable.append(
                            {
                                "pid": pid,
                                "name": label,
                                "tier": "tier1_working_cortex",
                                "cpu_pct": 0.0,
                                "rss_mb": rss_mb,
                                "rss_gb": round(rss_mb / 1024.0, 2),
                                "elapsed": "00:00",
                                "command": raw_cmd[:120],
                            }
                        )
                        reclaimable_mb += rss_mb
                    if rss_mb >= 500.0:
                        large_candidates.append(
                            {
                                "pid": pid,
                                "name": self._extract_process_label(raw_cmd),
                                "cpu_pct": 0.0,
                                "rss_mb": rss_mb,
                                "rss_gb": round(rss_mb / 1024.0, 2),
                                "elapsed": "00:00",
                                "command": raw_cmd[:100],
                            }
                        )
            except Exception as exc:
                logger.debug(f"Linux /proc process scan notice: {exc}")

        large_processes = sorted(
            large_candidates,
            key=lambda x: (
                1 if x["cpu_pct"] >= 20.0 else 0,
                x["cpu_pct"] if x["cpu_pct"] >= 20.0 else x["rss_mb"],
                x["rss_mb"],
            ),
            reverse=True,
        )[:8]

        return {
            "autonomic_processes": autonomic,
            "evictable_processes": evictable,
            "autonomic_ram_gb": round(autonomic_mb / 1024.0, 2),
            "reclaimable_ram_gb": round(reclaimable_mb / 1024.0, 2),
            "tachycardic_processes": tachycardic,
            "large_processes": large_processes,
        }

    @staticmethod
    def _extract_process_label(cmd: str) -> str:
        """
        Extract a human-readable process label handling command paths with spaces.
        Special cases:
          - 'voicefi tray' -> 'voicefi tray'
          - 'mlx_metal_worker' -> 'mlx_metal_worker'
          - 'litert_openai_server.py' -> 'litert_openai_server (:9379/:9380)'
          - 'whole_body_compute.cli' and 'telemetry' -> 'wbc telemetry (:9381)'
          - Python scripts: 'python (<script_basename>)'
        """
        cmd_str = (cmd or "").strip()
        if not cmd_str:
            return "process"

        if "voicefi tray" in cmd_str:
            return "voicefi tray"
        if "mlx_metal_worker" in cmd_str:
            return "mlx_metal_worker"
        if "litert_openai_server.py" in cmd_str:
            return "litert_openai_server (:9379/:9380)"
        if "whole_body_compute.cli" in cmd_str and "telemetry" in cmd_str:
            return "wbc telemetry (:9381)"

        python_match = re.search(r'(?:^|\s|\/)(python[0-9.]*)\b', cmd_str)
        if python_match:
            after_python = cmd_str[python_match.end():].strip()
            tokens: list[str] = []
            try:
                tokens = shlex.split(after_python)
            except Exception:
                tokens = after_python.split()
            script_name = None
            for tok in tokens:
                if not tok.startswith("-"):
                    script_name = os.path.basename(tok)
                    break
            if script_name:
                return f"python ({script_name})"
            return "python"

        try:
            tokens = shlex.split(cmd_str)
            if tokens:
                return os.path.basename(tokens[0])
        except Exception:
            pass
        parts = cmd_str.split()
        return os.path.basename(parts[0]) if parts else "process"

    @staticmethod
    def _parse_etime_seconds(etime_str: str) -> int:
        """Parse ps elapsed time format [[dd-]hh:]mm:ss into total seconds."""
        s = (etime_str or "").strip()
        if not s:
            return 0
        days = 0
        if "-" in s:
            d_part, s = s.split("-", 1)
            try:
                days = int(d_part)
            except ValueError:
                days = 0
        parts = s.split(":")
        try:
            if len(parts) == 3:
                hrs, mins, secs = int(parts[0]), int(parts[1]), int(parts[2])
            elif len(parts) == 2:
                hrs, mins, secs = 0, int(parts[0]), int(parts[1])
            else:
                hrs, mins, secs = 0, 0, int(parts[0])
            return days * 86400 + hrs * 3600 + mins * 60 + secs
        except ValueError:
            return 0

    def reap_and_clamp_runaway_processes(
        self,
        dry_run: bool = False,
        pytest_max_age_sec: int = 600,
        cpu_clamp_threshold: float = 80.0,
        model_idle_ttl_sec: int = 90,
        evict_idle_models: bool = True,
        recycle_tachycardic: bool = True,
    ) -> Dict[str, Any]:
        """
        Active Thermal, CPU & Unified Memory Homeostasis:
          0. Autonomic Tachycardia Sentinel: Detects and flags Tier-0 autonomic processes
             (`voicefi tray`, `mlx_metal_worker`) spinning >= 120% CPU or leaking >= 750 MB RSS.
             When `recycle_tachycardic=True`, issues SIGTERM so launchd respawns them cleanly with low CPU/RAM.
          1. Orphaned Dev/Test Reaper: Terminates hung `pytest` workers spinning >= 50% CPU
             for longer than `pytest_max_age_sec` (default 10 mins).
          2. Apple Silicon E-Core QoS Clamping: Clamps runaway background helpers
             (`Spotify`, `Chrome Helper`, `StocksWidget`) burning >= `cpu_clamp_threshold`% CPU
             onto Efficiency (E) cores via `taskpolicy -b -p <pid>` and `renice +19`.
          3. Automatic Idle Specialist Model TTL Eviction: Evicts idle `mlx_lm.server` and
             `llama-server` processes (<2% CPU for >= `model_idle_ttl_sec` (default 90s), or >= 30s when a
             Steam/3D game is running) into the macOS Darwin Inactive File Cache, while
             never interrupting active inference or active `eval_localbench.py` runs.
        """
        reaped_tests: list[Dict[str, Any]] = []
        clamped_helpers: list[Dict[str, Any]] = []
        evicted_idle_models: list[Dict[str, Any]] = []
        tachycardic_autonomic: list[Dict[str, Any]] = []
        large_candidates: list[Dict[str, Any]] = []
        my_pid = os.getpid()
        my_ppid = os.getppid()

        if not self.is_macos:
            return {
                "reaped_tests": reaped_tests,
                "clamped_helpers": clamped_helpers,
                "evicted_idle_models": evicted_idle_models,
                "tachycardic_autonomic": tachycardic_autonomic,
                "large_processes": [],
            }

        try:
            res = subprocess.run(
                ["ps", "-ax", "-o", "pid=,pcpu=,rss=,etime=,stat=,command="],
                capture_output=True,
                text=True,
                timeout=1.8,
            )
            raw_lines = res.stdout.splitlines()
            now_ts = time.time()

            # Check if an active benchmark script or Steam game is currently running
            benchmark_active = any(
                ("eval_localbench.py" in ln or "eval_models.py" in ln or "qa_and_uat_harness" in ln)
                and "grep" not in ln
                for ln in raw_lines
            )
            game_active = any(
                "steamapps/common/" in ln and "grep" not in ln
                for ln in raw_lines
            )
            effective_idle_ttl = 30 if game_active else model_idle_ttl_sec
            live_model_pids: set[int] = set()

            for line in raw_lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 5)
                if len(parts) < 6:
                    continue
                try:
                    pid = int(parts[0])
                    pcpu = float(parts[1])
                    rss_mb = round(int(parts[2]) / 1024.0, 1)
                except ValueError:
                    continue
                etime = parts[3]
                stat = parts[4]
                cmd = parts[5]

                if pid in (my_pid, my_ppid):
                    continue

                # Collect candidate for large_processes radar (>= 500 MB or >= 20% CPU)
                is_excluded_radar = (
                    "Antigravity.app" in cmd
                    or "grep" in cmd
                    or ("whole_body_compute.cli" in cmd and "telemetry" not in cmd)
                )
                if not is_excluded_radar:
                    if rss_mb >= 500.0 or pcpu >= 20.0:
                        proc_label = self._extract_process_label(cmd)
                        large_candidates.append(
                            {
                                "pid": pid,
                                "name": proc_label,
                                "cpu_pct": pcpu,
                                "rss_mb": rss_mb,
                                "rss_gb": round(rss_mb / 1024.0, 2),
                                "elapsed": etime,
                                "command": cmd[:100],
                            }
                        )

                # Never touch primary IDE processes
                if any(tok in cmd for tok in ("Antigravity.app", "whole_body_compute.cli")):
                    continue

                elapsed_sec = self._parse_etime_seconds(etime)

                # Rule 0: Autonomic Tachycardia Sentinel & Broader Dev Runaway Process Sentinel
                is_autonomic = any(tok in cmd for tok in ("voicefi tray", "mlx_metal_worker"))
                is_other_dev = (
                    any(tok in cmd for tok in ("vifi", "voicefi", "python", "uv"))
                    and not any(tok in cmd for tok in ("mlx_lm.server", "llama-server", "ollama", "telemetry_server"))
                )

                is_runaway = False
                if is_autonomic:
                    if (elapsed_sec >= 120 and pcpu >= 120.0) or ("voicefi tray" in cmd and rss_mb >= 4096.0):
                        is_runaway = True
                elif is_other_dev:
                    is_tmp_script = "/tmp/" in cmd and any(tok in cmd for tok in ("python", "uv"))
                    if is_tmp_script and elapsed_sec >= 120 and pcpu >= 85.0:
                        is_runaway = True
                    elif elapsed_sec >= 180 and (pcpu >= 85.0 or rss_mb >= 3500.0):
                        is_runaway = True

                if is_runaway:
                    proc_name = self._extract_process_label(cmd)
                    now_kill = time.time()
                    sigterm_prev_ts = self._runaway_sigterm_ts.get(pid)
                    escalate_sigkill = (sigterm_prev_ts is not None and (now_kill - sigterm_prev_ts) >= 2.0)
                    action_tag = "autonomic_tachycardia_flagged"

                    if recycle_tachycardic and not dry_run:
                        if escalate_sigkill:
                            try:
                                os.kill(pid, 9)  # SIGKILL escalation
                                action_tag = "recycle_tachycardic_sigkill"
                                self._runaway_sigterm_ts.pop(pid, None)
                            except OSError:
                                pass
                        else:
                            try:
                                os.kill(pid, 15)  # SIGTERM for clean launchd respawn
                                self._runaway_sigterm_ts[pid] = now_kill
                                action_tag = "recycle_tachycardic_sigterm"
                                # For /tmp/ scratch scripts, check if still alive after 0.2s and escalate to SIGKILL
                                if "/tmp/" in cmd:
                                    time.sleep(0.2)
                                    try:
                                        os.kill(pid, 0)
                                        os.kill(pid, 9)
                                        action_tag = "recycle_tachycardic_sigkill"
                                        self._runaway_sigterm_ts.pop(pid, None)
                                    except OSError:
                                        self._runaway_sigterm_ts.pop(pid, None)
                            except OSError:
                                pass

                    entry = {
                        "pid": pid,
                        "name": proc_name,
                        "cpu_pct": pcpu,
                        "rss_mb": rss_mb,
                        "elapsed": etime,
                        "action": action_tag,
                        "command": cmd[:100],
                    }

                    # Notify HUD gate event with per-PID cooldown (30s)
                    now_alert = time.time()
                    last_alert_ts = self._runaway_alert_cooldown.get(pid, 0.0)
                    if (now_alert - last_alert_ts) >= 30.0:
                        self._runaway_alert_cooldown[pid] = now_alert
                        reason_msg = (
                            f"Recycled runaway {proc_name} ({rss_mb:.0f}MB, {pcpu:.0f}% CPU)"
                            if (recycle_tachycardic and not dry_run)
                            else f"Runaway flagged: {proc_name} ({rss_mb:.0f}MB, {pcpu:.0f}% CPU)"
                        )
                        self._notify_hud_gate_event(
                            requested_model=proc_name,
                            resolved_model=None,
                            status="AUTO_SWEPT",
                            reason=reason_msg,
                        )

                    tachycardic_autonomic.append(entry)
                    continue

                elapsed_sec = self._parse_etime_seconds(etime)

                # Rule 3: Automatic Idle Specialist Model TTL Eviction (mlx_lm.server / llama-server)
                if evict_idle_models and ("mlx_lm.server" in cmd or "llama-server" in cmd):
                    live_model_pids.add(pid)
                    if pcpu >= 2.0 or benchmark_active:
                        self._model_last_active_ts[pid] = now_ts
                    else:
                        last_active = self._model_last_active_ts.get(pid)
                        idle_sec = (now_ts - last_active) if last_active is not None else elapsed_sec
                        if elapsed_sec >= effective_idle_ttl and idle_sec >= effective_idle_ttl:
                            label = "mlx_lm.server" if "mlx_lm.server" in cmd else "llama-server"
                            entry = {
                                "pid": pid,
                                "name": label,
                                "cpu_pct": pcpu,
                                "rss_mb": rss_mb,
                                "rss_gb": round(rss_mb / 1024.0, 2),
                                "elapsed": etime,
                                "idle_sec": int(idle_sec),
                                "game_accelerated": game_active,
                                "action": "evict_idle_sigterm" if not dry_run else "would_evict_idle",
                                "command": cmd[:100],
                            }
                            if not dry_run:
                                try:
                                    os.kill(pid, 15)
                                    self._model_last_active_ts.pop(pid, None)
                                except OSError:
                                    pass
                            evicted_idle_models.append(entry)
                    continue

                # Rule 1: Hung / runaway pytest (>10m) or recursive grep crawler (>3m) spinning >=50% CPU
                is_hung_pytest = "pytest" in cmd and elapsed_sec >= pytest_max_age_sec and pcpu >= 50.0
                is_hung_grep = cmd.startswith("grep ") and ("-r" in cmd or "-R" in cmd) and elapsed_sec >= 180 and pcpu >= 50.0
                if is_hung_pytest or is_hung_grep:
                    entry = {
                        "pid": pid,
                        "name": "pytest (hung runner)" if is_hung_pytest else "grep -r (hung crawler)",
                        "cpu_pct": pcpu,
                        "rss_mb": rss_mb,
                        "elapsed": etime,
                        "action": "reap_sigkill" if not dry_run else "would_reap",
                        "command": cmd[:100],
                    }
                    if not dry_run:
                        try:
                            os.kill(pid, 9)
                        except OSError:
                            pass
                    reaped_tests.append(entry)
                    continue

                # Rule 1.5: Immediately reap unwanted StocksWidget / Stocks.app if macOS respawns it
                if "StocksWidget" in cmd or "/System/Applications/Stocks.app" in cmd:
                    entry = {
                        "pid": pid,
                        "name": "StocksWidget" if "StocksWidget" in cmd else "Stocks.app",
                        "cpu_pct": pcpu,
                        "rss_mb": rss_mb,
                        "elapsed": etime,
                        "action": "reap_stocks" if not dry_run else "would_reap_stocks",
                        "command": cmd[:100],
                    }
                    if not dry_run:
                        try:
                            subprocess.run(
                                ["pluginkit", "-e", "ignore", "-i", "com.apple.stocks.widget"],
                                capture_output=True,
                                timeout=1.0,
                            )
                        except Exception:
                            pass
                        try:
                            os.kill(pid, 9)
                        except OSError:
                            pass
                    reaped_tests.append(entry)
                    continue

                # Rule 2: Runaway background helper / renderer / widget burning CPU
                is_widget = "Widget.appex" in cmd
                is_clamp_target = is_widget or any(
                    tok in cmd
                    for tok in (
                        "Spotify",
                        "Google Chrome Helper",
                        "steam_osx",
                        "Steam Helper",
                    )
                )
                effective_clamp_threshold = 20.0 if is_widget else cpu_clamp_threshold
                if is_clamp_target and pcpu >= effective_clamp_threshold:
                    already_niced = "N" in stat
                    # If Spotify Renderer is still >110% CPU even after E-core nice clamping, recycle the helper
                    if "Spotify Helper (Renderer)" in cmd and pcpu >= 110.0 and already_niced:
                        action_label = "recycle_renderer" if not dry_run else "would_recycle_renderer"
                        if not dry_run:
                            try:
                                os.kill(pid, 15)
                            except OSError:
                                pass
                    else:
                        action_label = "ecore_clamp" if not dry_run else "would_ecore_clamp"
                        if not dry_run:
                            subprocess.run(
                                ["taskpolicy", "-b", "-p", str(pid)],
                                capture_output=True,
                                timeout=1.0,
                            )
                            subprocess.run(
                                ["renice", "+19", "-p", str(pid)],
                                capture_output=True,
                                timeout=1.0,
                            )
                    clamped_helpers.append(
                        {
                            "pid": pid,
                            "name": "Spotify Helper" if "Spotify" in cmd else ("Chrome Helper" if "Chrome" in cmd else "macOS Widget"),
                            "cpu_pct": pcpu,
                            "rss_mb": rss_mb,
                            "elapsed": etime,
                            "stat": stat,
                            "action": action_label,
                            "command": cmd[:100],
                        }
                    )

            # Prune exited PIDs from tracking dict
            for tracked_pid in list(self._model_last_active_ts.keys()):
                if tracked_pid not in live_model_pids:
                    self._model_last_active_ts.pop(tracked_pid, None)
        except Exception as e:
            logger.debug(f"Failed runaway process sweep: {e}")

        large_processes = sorted(
            large_candidates,
            key=lambda x: (
                1 if x["cpu_pct"] >= 20.0 else 0,
                x["cpu_pct"] if x["cpu_pct"] >= 20.0 else x["rss_mb"],
                x["rss_mb"],
            ),
            reverse=True,
        )[:8]

        return {
            "reaped_tests": reaped_tests,
            "clamped_helpers": clamped_helpers,
            "evicted_idle_models": evicted_idle_models,
            "tachycardic_autonomic": tachycardic_autonomic,
            "large_processes": large_processes,
        }

    def run_vascular_perfusion_sweep(
        self,
        dry_run: bool = False,
        settle_sec: float = 0.6,
        recycle_tachycardic: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute an Ultradian Glymphatic RAM Hygiene & Selective Vasoconstriction sweep.
        Evicts idle Tier-1/Tier-2 specialist model processes (`mlx_lm.server`, `llama-server`)
        back into the macOS Darwin Inactive File Cache (<1s warm-cache remapping), while
        strictly preserving Tier-0 Autonomic Brainstem processes (`voicefi tray`, `:9379`).
        Also reaps hung `pytest` runners and clamps runaway background helpers to E-cores.
        """
        before = self.get_telemetry()
        scan = self.get_resident_model_processes()
        thermal_hygiene = self.reap_and_clamp_runaway_processes(
            dry_run=dry_run, recycle_tachycardic=recycle_tachycardic
        )
        evicted = []

        # External runner (LM Studio, llama-server) perfusion management
        external_runners = self.discover_external_local_runners()
        external_eviction = self.evict_external_runner_models(dry_run=dry_run)

        if not dry_run:
            for proc in scan["evictable_processes"]:
                pid = proc["pid"]
                try:
                    os.kill(pid, 15)  # SIGTERM for clean mmap release to Darwin File Cache
                    evicted.append(proc)
                except OSError:
                    pass
            if (evicted or thermal_hygiene["reaped_tests"] or thermal_hygiene["clamped_helpers"]) and settle_sec > 0:
                time.sleep(settle_sec)
            self._telemetry_cache = None

        after = self.get_telemetry() if not dry_run else before
        reclaimed_gb = round(max(0.0, after.free_ram_gb - before.free_ram_gb), 2)
        cognitive_telem = self.get_cognitive_telemetry()

        return {
            "dry_run": dry_run,
            "mechanism": "Selective Vascular Vasoconstriction & Ultradian Glymphatic Flush",
            "before_free_ram_gb": before.free_ram_gb,
            "after_free_ram_gb": after.free_ram_gb,
            "reclaimed_ram_gb": reclaimed_gb,
            "estimated_reclaimable_gb": scan["reclaimable_ram_gb"],
            "cognitive_saturation": cognitive_telem,
            "evicted_processes": evicted if not dry_run else scan["evictable_processes"],
            "protected_autonomic_processes": scan["autonomic_processes"],
            "tachycardic_autonomic": thermal_hygiene.get("tachycardic_autonomic", []),
            "large_processes": thermal_hygiene.get("large_processes", scan.get("large_processes", [])),
            "reaped_tests": thermal_hygiene["reaped_tests"],
            "clamped_helpers": thermal_hygiene["clamped_helpers"],
            "external_runners": external_runners,
            "external_eviction": external_eviction,
            "darwin_file_cache_note": (
                "Unloaded model weights remain in macOS Inactive File Cache (or Linux Page Cache / "
                "Windows Standby List) for <1s zero-SSD remapping until physical pages are needed "
                "by heavier frontier models (e.g. gemma4-26b)."
            ),
            "kernel_page_cache": self.advise_kernel_page_cache(mode="dontneed"),
        }

    def audit_vascular_perfusion(self) -> Dict[str, Any]:
        """
        Audit current vascular perfusion across local runners (Ollama, MLX, LM Studio, llama-server).
        Returns a dry-run perfusion status report without altering resident processes.
        """
        return self.run_vascular_perfusion_sweep(dry_run=True)

    def discover_external_local_runners(self) -> Dict[str, Any]:
        """
        Discover active external local runners in Unified Memory:
          - LM Studio (default: http://127.0.0.1:1234, `lms` CLI)
          - llama-server (default: http://127.0.0.1:8080)
        Returns structured runner status with loaded models and process PIDs.
        """
        import shutil
        import urllib.request
        import urllib.error

        # 1. LM Studio discovery
        lm_base = os.environ.get("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234").rstrip("/")
        lm_online = False
        lm_loaded_models: list[str] = []
        lm_cli_path: Optional[str] = None
        lm_cli_available = False

        # Check lms CLI in PATH or default install directory
        which_lms = shutil.which("lms")
        if which_lms:
            lm_cli_path = which_lms
            lm_cli_available = True
        else:
            default_lms_paths = [
                os.path.expanduser("~/.lmstudio/bin/lms"),
                os.path.expanduser("~/.cache/lm-studio/bin/lms"),
            ]
            for p in default_lms_paths:
                if os.path.isfile(p) and os.access(p, os.X_OK):
                    lm_cli_path = p
                    lm_cli_available = True
                    break

        # Probe LM Studio endpoint
        try:
            req = urllib.request.Request(f"{lm_base}/v1/models", method="GET")
            with urllib.request.urlopen(req, timeout=0.3) as resp:
                if resp.status == 200:
                    lm_online = True
                    raw = resp.read().decode("utf-8")
                    try:
                        data = json.loads(raw)
                        if isinstance(data, dict):
                            entries = data.get("data", []) or data.get("models", [])
                            if isinstance(entries, list):
                                for item in entries:
                                    if isinstance(item, dict):
                                        m_id = item.get("id") or item.get("name")
                                        if m_id:
                                            lm_loaded_models.append(str(m_id))
                                    elif isinstance(item, str):
                                        lm_loaded_models.append(item)
                        elif isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    m_id = item.get("id") or item.get("name")
                                    if m_id:
                                        lm_loaded_models.append(str(m_id))
                                elif isinstance(item, str):
                                    lm_loaded_models.append(item)
                    except Exception as parse_err:
                        logger.debug(f"Failed to parse LM Studio /v1/models JSON: {parse_err}")
        except Exception as exc:
            logger.debug(f"LM Studio discovery probe skipped ({lm_base}): {exc}")

        # 2. llama-server (llama.cpp) discovery
        llama_base = os.environ.get("LLAMA_SERVER_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
        llama_online = False
        llama_loaded_models: list[str] = []
        llama_pids: list[int] = []

        # Find running llama-server PIDs
        try:
            ps_res = subprocess.run(
                ["pgrep", "-f", "llama-server"],
                capture_output=True,
                text=True,
                timeout=0.5,
            )
            if ps_res.returncode == 0:
                for line in ps_res.stdout.split():
                    line = line.strip()
                    if line.isdigit() and int(line) != os.getpid():
                        llama_pids.append(int(line))
        except Exception:
            pass

        # Probe llama-server health & models
        try:
            req_health = urllib.request.Request(f"{llama_base}/health", method="GET")
            with urllib.request.urlopen(req_health, timeout=0.3) as resp:
                if resp.status == 200:
                    llama_online = True
        except Exception as exc:
            logger.debug(f"llama-server /health probe skipped ({llama_base}): {exc}")

        try:
            req_models = urllib.request.Request(f"{llama_base}/v1/models", method="GET")
            with urllib.request.urlopen(req_models, timeout=0.3) as resp:
                if resp.status == 200:
                    llama_online = True
                    raw = resp.read().decode("utf-8")
                    try:
                        data = json.loads(raw)
                        if isinstance(data, dict):
                            entries = data.get("data", []) or data.get("models", [])
                            if isinstance(entries, list):
                                for item in entries:
                                    if isinstance(item, dict):
                                        m_id = item.get("id") or item.get("name")
                                        if m_id:
                                            llama_loaded_models.append(str(m_id))
                                    elif isinstance(item, str):
                                        llama_loaded_models.append(item)
                    except Exception as parse_err:
                        logger.debug(f"Failed to parse llama-server /v1/models JSON: {parse_err}")
        except Exception as exc:
            logger.debug(f"llama-server /v1/models probe skipped ({llama_base}): {exc}")

        if llama_pids and not llama_online:
            # PIDs are resident even if HTTP is binding or saturated
            llama_online = True

        return {
            "lm_studio": {
                "online": lm_online,
                "loaded_models": lm_loaded_models,
                "cli_available": lm_cli_available,
                "cli_path": lm_cli_path,
                "models_endpoint": f"{lm_base}/v1/models",
            },
            "llama_server": {
                "online": llama_online,
                "loaded_models": llama_loaded_models,
                "pids": llama_pids,
                "health_endpoint": f"{llama_base}/health",
                "models_endpoint": f"{llama_base}/v1/models",
            },
        }

    def evict_external_runner_models(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Selectively evict models hosted in external local runners (LM Studio, llama-server).
        Releases Unified RAM back to macOS Darwin Inactive File Cache without killing Tier-0.
          - LM Studio: uses `lms unload --all` (CLI) or POST `/api/v0/models/unload`
          - llama-server: calls `POST /slots/0?action=erase` (graceful KV-cache slot flush)
        """
        import urllib.request
        import urllib.error

        runners = self.discover_external_local_runners()
        lm_info = runners["lm_studio"]
        llama_info = runners["llama_server"]

        lm_base = os.environ.get("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234").rstrip("/")
        llama_base = os.environ.get("LLAMA_SERVER_BASE_URL", "http://127.0.0.1:8080").rstrip("/")

        # --- 1. LM Studio Eviction ---
        lm_evicted: list[str] = []
        lm_action = "none"
        lm_success = True

        if lm_info["online"] or lm_info["loaded_models"]:
            models_to_evict = list(lm_info["loaded_models"]) if lm_info["loaded_models"] else ["all_loaded_models"]
            if dry_run:
                lm_action = "would_unload_lms"
                lm_evicted = models_to_evict
            else:
                cli_path = lm_info["cli_path"] or "lms"
                unloaded_via_cli = False
                if lm_info["cli_available"]:
                    try:
                        res = subprocess.run(
                            [cli_path, "unload", "--all"],
                            capture_output=True,
                            text=True,
                            timeout=2.0,
                        )
                        if res.returncode == 0:
                            lm_action = "lms_cli_unload_all"
                            lm_evicted = models_to_evict
                            lm_success = True
                            unloaded_via_cli = True
                    except Exception as exc:
                        logger.debug(f"lms unload --all failed: {exc}")

                if not unloaded_via_cli:
                    # REST fallback: POST /api/v0/models/unload
                    try:
                        req = urllib.request.Request(
                            f"{lm_base}/api/v0/models/unload",
                            data=b"{}",
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        )
                        with urllib.request.urlopen(req, timeout=1.0) as resp:
                            lm_action = "lms_rest_unload"
                            lm_evicted = models_to_evict
                            lm_success = True
                    except Exception as exc:
                        lm_action = f"lms_unload_failed: {exc}"
                        lm_success = False

        # --- 2. llama-server Eviction (KV Slot Erase) ---
        llama_evicted: list[str] = []
        llama_action = "none"
        llama_success = True

        if llama_info["online"] or llama_info["pids"]:
            models_or_slots = list(llama_info["loaded_models"]) if llama_info["loaded_models"] else ["slot_0_kv_cache"]
            if dry_run:
                llama_action = "would_erase_kv_slots"
                llama_evicted = models_or_slots
            else:
                try:
                    req = urllib.request.Request(
                        f"{llama_base}/slots/0?action=erase",
                        data=b"{}",
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=1.0) as resp:
                        llama_action = "llama_slot_erase"
                        llama_evicted = models_or_slots
                        llama_success = True
                except Exception as exc:
                    llama_action = f"llama_slot_erase_advisory: {exc}"
                    llama_success = True
                    llama_evicted = models_or_slots

        return {
            "dry_run": dry_run,
            "lm_studio": {
                "evicted_models": lm_evicted,
                "action": lm_action,
                "success": lm_success,
            },
            "llama_server": {
                "evicted_models": llama_evicted,
                "action": llama_action,
                "success": llama_success,
                "pids": llama_info.get("pids", []),
            },
        }

    def advise_kernel_page_cache(
        self,
        weights_path: Optional[str | Path] = None,
        mode: str = "willneed",
        os_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Cross-Platform Two-Tier OS Kernel Page-Cache Handoff (`Tier 1 Active Resident` <-> `Tier 2 Warm Page Cache`):
          - macOS (`Darwin`): `Darwin Inactive File Cache` (`mmap` + `MADV_WILLNEED` / `MADV_DONTNEED` or clean `SIGTERM` mmap release)
          - Linux (`Linux`): `Linux Kernel Page Cache` (`posix_fadvise(POSIX_FADV_WILLNEED|POSIX_FADV_DONTNEED)` & `madvise(MADV_WILLNEED|MADV_DONTNEED)`)
          - Windows (`Windows` / `win32`): `Windows Standby Page List` (`PrefetchVirtualMemory` / `DiscardVirtualMemory` file-mapping handoff)
        """
        import mmap

        sys_name = str(os_override or platform.system() or "Darwin").strip()
        sys_lower = sys_name.lower()
        norm_mode = "dontneed" if "dont" in mode.lower() or "evict" in mode.lower() else "willneed"

        if "cuda" in sys_lower:
            os_family = "Linux/CUDA"
            cache_tier = "Host-Pinned DRAM Staging Buffer + Linux Page Cache"
            syscall_mech = (
                "cudaFreeHost / posix_fadvise(POSIX_FADV_DONTNEED) -> Host Page Cache"
                if norm_mode == "dontneed"
                else "cudaHostAllocMapped / mlock + posix_fadvise(POSIX_FADV_WILLNEED) -> PCIe/NVLink DMA"
            )
        elif "linux" in sys_lower:
            os_family = "Linux"
            cache_tier = "Linux Kernel Page Cache"
            syscall_mech = (
                "posix_fadvise(POSIX_FADV_DONTNEED) + madvise(MADV_DONTNEED)"
                if norm_mode == "dontneed"
                else "posix_fadvise(POSIX_FADV_WILLNEED) + madvise(MADV_WILLNEED)"
            )
        elif sys_lower.startswith("win") or "windows" in sys_lower:
            os_family = "Windows"
            cache_tier = "Windows Standby List"
            syscall_mech = (
                "DiscardVirtualMemory / UnmapViewOfFile -> Standby Page List"
                if norm_mode == "dontneed"
                else "PrefetchVirtualMemory(Standby List -> Active Working Set)"
            )
        else:
            os_family = "Darwin"
            cache_tier = "Darwin Inactive File Cache"
            syscall_mech = (
                "madvise(MADV_DONTNEED) / SIGTERM mmap release -> Darwin Inactive File Cache"
                if norm_mode == "dontneed"
                else "madvise(MADV_WILLNEED) -> Unified Memory Active Working Set"
            )

        advised_bytes = 0
        syscall_applied = False
        if weights_path:
            wp = Path(weights_path)
            if wp.is_file():
                try:
                    fsize = wp.stat().st_size
                    advised_bytes = fsize
                    if fsize > 0:
                        with wp.open("rb") as fh:
                            fd = fh.fileno()
                            if hasattr(os, "posix_fadvise"):
                                flag = (
                                    getattr(os, "POSIX_FADV_DONTNEED", 4)
                                    if norm_mode == "dontneed"
                                    else getattr(os, "POSIX_FADV_WILLNEED", 3)
                                )
                                try:
                                    os.posix_fadvise(fd, 0, fsize, flag)
                                    syscall_applied = True
                                except OSError:
                                    pass
                            map_len = min(fsize, 4 * 1024 * 1024)
                            with mmap.mmap(fd, map_len, access=mmap.ACCESS_READ) as mm:
                                if hasattr(mm, "madvise"):
                                    m_flag = (
                                        getattr(mmap, "MADV_DONTNEED", 4)
                                        if norm_mode == "dontneed"
                                        else getattr(mmap, "MADV_WILLNEED", 3)
                                    )
                                    try:
                                        mm.madvise(m_flag)
                                        syscall_applied = True
                                    except OSError:
                                        pass
                except Exception:
                    pass

        return {
            "os_family": os_family,
            "cache_tier": cache_tier,
            "mode": norm_mode,
            "syscall_mechanism": syscall_mech,
            "syscall_applied": syscall_applied,
            "advised_bytes": advised_bytes,
            "supported_tiers": [
                "Darwin Inactive File Cache",
                "Linux posix_fadvise/madvise Page Cache",
                "Windows Standby List",
                "Host-Pinned DRAM Staging Buffer (CUDA/PCIe DMA)",
            ],
        }

    @staticmethod
    def parse_keep_alive_seconds(keep_alive: Any) -> float:
        """Parse keep_alive strings like '5m', '15m', '300s', '1h', '0s', '0' into seconds."""
        if keep_alive is None:
            return 300.0
        if isinstance(keep_alive, (int, float)):
            return max(0.0, float(keep_alive))
        s = str(keep_alive).strip().lower()
        if not s or s == "auto":
            return 300.0
        if s.endswith("ms"):
            try:
                return max(0.0, float(s[:-2]) / 1000.0)
            except ValueError:
                return 300.0
        if s.endswith("s"):
            try:
                return max(0.0, float(s[:-1]))
            except ValueError:
                return 300.0
        if s.endswith("m"):
            try:
                return max(0.0, float(s[:-1]) * 60.0)
            except ValueError:
                return 300.0
        if s.endswith("h"):
            try:
                return max(0.0, float(s[:-1]) * 3600.0)
            except ValueError:
                return 300.0
        if s.endswith("d"):
            try:
                return max(0.0, float(s[:-1]) * 86400.0)
            except ValueError:
                return 300.0
        try:
            return max(0.0, float(s))
        except ValueError:
            return 300.0

    def record_model_load_telemetry(
        self,
        model_name: str,
        load_duration_ms: float,
        total_duration_ms: float = 0.0,
        runtime: Optional[str] = None,
        is_cold_start: Optional[bool] = None,
        configured_ttl_sec: float = 300.0,
        stimulus_source: str = "inference",
    ) -> Dict[str, Any]:
        """Record model load latency and cold-start classification into DotMemory ledger."""
        try:
            from whole_body_compute.local.dot_memory import get_dot_store
            telem = self.get_telemetry()
            free_ram = float(getattr(telem, "free_ram_gb", 16.0))
            thermal_st = str(getattr(telem, "thermal_state", "nominal"))
            store = get_dot_store()
            return store.record_model_invocation(
                model_name=model_name,
                runtime=runtime,
                load_duration_ms=load_duration_ms,
                total_duration_ms=total_duration_ms,
                is_cold_start=is_cold_start,
                configured_ttl_sec=configured_ttl_sec,
                free_ram_gb=free_ram,
                thermal_state=thermal_st,
                stimulus_source=stimulus_source,
            )
        except Exception as exc:
            logger.debug(f"Failed to record model load telemetry: {exc}")
            return {
                "status": "error",
                "error": str(exc),
                "model_name": model_name,
            }

    def get_adaptive_keep_alive(
        self,
        model_name: str,
        runtime: Optional[str] = None,
        default_keep_alive: str = "auto",
        current_hour: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate cold-start history and circadian windows to produce an adaptive keep_alive TTL
        clamped to current Unified RAM and thermal state.
        """
        try:
            from whole_body_compute.local.dot_memory import get_dot_store
            telem = self.get_telemetry()
            free_ram = float(getattr(telem, "free_ram_gb", 16.0))
            thermal_st = str(getattr(telem, "thermal_state", "nominal"))
            req = self.get_model_requirement(model_name)
            req_ram = float(req.get("min_free_ram_gb", 4.0))

            store = get_dot_store()
            policy = store.get_cold_start_policy(
                model_name=model_name,
                runtime=runtime,
                free_ram_gb=free_ram,
                required_ram_gb=req_ram,
                thermal_state=thermal_st,
                current_hour=current_hour,
            )
            # If no history exists, fall back to default_keep_alive with hardware clamps applied
            if policy.get("total_invocations", 0) == 0:
                is_tier0 = bool(req.get("is_tier0", False)) or any(
                    k in model_name.lower()
                    for k in ("qwen2.5-coder:1.5b", "voicefi-coder:1.5b", "tev1:0.8b", "reflex")
                )
                if default_keep_alive == "auto":
                    def_sec = 1800.0 if is_tier0 else 300.0
                    def_str = "30m" if is_tier0 else "5m"
                else:
                    def_sec = self.parse_keep_alive_seconds(default_keep_alive)
                    def_str = default_keep_alive
                    if is_tier0 and def_sec < 1800.0 and default_keep_alive in ("5m", "300s"):
                        def_sec = 1800.0
                        def_str = "30m"

                if thermal_st.lower() in ("critical", "serious") or free_ram < req_ram:
                    policy["recommended_ttl_sec"] = 0.0
                    policy["recommended_keep_alive"] = "0s"
                elif thermal_st.lower() == "fair" or free_ram < (req_ram * 1.3):
                    policy["recommended_ttl_sec"] = min(def_sec, 180.0)
                    policy["recommended_keep_alive"] = (
                        "3m" if policy["recommended_ttl_sec"] == 180.0 else f"{int(policy['recommended_ttl_sec'])}s"
                    )
                else:
                    policy["recommended_ttl_sec"] = def_sec
                    policy["recommended_keep_alive"] = def_str
            return policy
        except Exception as exc:
            logger.debug(f"Failed to calculate adaptive keep alive for {model_name}: {exc}")
            is_tier0 = any(
                k in model_name.lower()
                for k in ("qwen2.5-coder:1.5b", "voicefi-coder:1.5b", "tev1:0.8b", "reflex")
            )
            if default_keep_alive == "auto":
                def_sec = 1800.0 if is_tier0 else 300.0
                def_str = "30m" if is_tier0 else "5m"
            else:
                def_sec = self.parse_keep_alive_seconds(default_keep_alive)
                def_str = default_keep_alive
                if is_tier0 and def_sec < 1800.0 and default_keep_alive in ("5m", "300s"):
                    def_sec = 1800.0
                    def_str = "30m"
            return {
                "model_name": model_name,
                "runtime": runtime or "ollama",
                "recommended_ttl_sec": def_sec,
                "recommended_keep_alive": def_str,
                "eviction_protection_score": 0.0,
                "should_pin_warm": False,
                "error": str(exc),
            }

    def get_cold_start_report(self, window_hours: float = 168.0) -> Dict[str, Any]:
        """Aggregate cross-model cold-start report across the specified window."""
        try:
            from whole_body_compute.local.dot_memory import get_dot_store
            telem = self.get_telemetry()
            free_ram = float(getattr(telem, "free_ram_gb", 16.0))
            thermal_st = str(getattr(telem, "thermal_state", "nominal"))
            store = get_dot_store()
            return store.get_all_cold_start_summaries(
                window_hours=window_hours,
                free_ram_gb=free_ram,
                thermal_state=thermal_st,
            )
        except Exception as exc:
            logger.debug(f"Failed to generate cold start report: {exc}")
            return {
                "error": str(exc),
                "policies": [],
                "total_models": 0,
                "total_cold_starts": 0,
                "total_cold_penalty_ms": 0.0,
                "total_ttl_misses": 0,
                "pinned_models": [],
            }

    def trigger_cephalic_prewarm(
        self,
        model_name: str = "qwen2.5-coder:1.5b",
        keep_alive: str = "auto",
        ollama_base_url: str = "http://127.0.0.1:11434",
        background: bool = True,
        stimulus_source: str = "vad_speech_onset",
        keystroke_buffer_len: int = 0,
        debounce_sec: float = 0.0,
        weights_path: Optional[str | Path] = None,
        os_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Cephalic Phase Predictive Pre-Warming Reflex (Organ 05/06 -> Organ 03).
        Triggered BEFORE prompt submission upon:
          1. `keystroke_onset`: User begins typing in an IDE/chat prompt buffer before pressing Enter.
          2. `ide_focus_shift`: IDE window or editor tab focus shift (`NSWorkspaceDidActivateApplicationNotification`).
          3. `vad_speech_onset` / `wakeword`: Acoustic VAD speech onset or wakeword ('Hey Viv').
        Transitions model weights from the OS kernel warm page cache into Active Resident Memory
        with an adaptive idle TTL (`keep_alive`), masking cold-start latency to 0ms.
        """
        import json as _json
        import threading
        import urllib.request

        valid_stimuli = {
            "keystroke_onset",
            "ide_focus_shift",
            "vad_speech_onset",
            "wakeword",
            "cli_manual",
        }
        norm_stimulus = str(stimulus_source or "vad_speech_onset").strip().lower()
        if norm_stimulus not in valid_stimuli:
            norm_stimulus = "vad_speech_onset"

        # Adaptive TTL resolution: when keep_alive is 'auto' or '5m', resolve via cold-start history
        adaptive_pol = self.get_adaptive_keep_alive(model_name, default_keep_alive=keep_alive)
        if keep_alive in ("auto", "5m"):
            effective_keep_alive = str(adaptive_pol.get("recommended_keep_alive", keep_alive))
            configured_ttl_sec = float(adaptive_pol.get("recommended_ttl_sec", 300.0))
        else:
            effective_keep_alive = keep_alive
            configured_ttl_sec = self.parse_keep_alive_seconds(keep_alive)

        now_ts = time.time()
        if (
            debounce_sec > 0.0
            and (now_ts - self._cephalic_last_ts) < debounce_sec
            and self._cephalic_model == model_name
            and self._cephalic_state in ("WARMING", "WARMED", "WARMED_PAGE_CACHE")
        ):
            cached_pc = getattr(self, "_cephalic_last_page_cache", None) or self.advise_kernel_page_cache(
                weights_path=weights_path,
                mode="willneed",
                os_override=os_override,
            )
            return {
                "status": "debounced_active",
                "mechanism": "Cephalic Phase Predictive Pre-Warm",
                "stimulus_source": norm_stimulus,
                "keystroke_buffer_len": int(keystroke_buffer_len or 0),
                "model": model_name,
                "keep_alive_ttl": effective_keep_alive,
                "adaptive_policy": adaptive_pol,
                "kernel_page_cache": cached_pc,
            }

        approved, reason, _ = self.can_load_model(model_name)
        if not approved:
            return {
                "status": "blocked",
                "stimulus_source": norm_stimulus,
                "model": model_name,
                "reason": reason,
                "adaptive_policy": adaptive_pol,
            }

        page_cache_handoff = self.advise_kernel_page_cache(
            weights_path=weights_path,
            mode="willneed",
            os_override=os_override,
        )
        self._cephalic_last_page_cache = page_cache_handoff

        payload = _json.dumps(
            {
                "model": model_name.replace("ollama/", ""),
                "prompt": "",
                "keep_alive": effective_keep_alive,
            }
        ).encode("utf-8")

        self._cephalic_state = "WARMING"
        self._cephalic_model = model_name
        self._cephalic_stimulus = norm_stimulus
        self._cephalic_last_ts = now_ts

        def _do_warm() -> None:
            t0 = time.perf_counter()
            load_dur_ms = 0.0
            total_dur_ms = 0.0
            try:
                req = urllib.request.Request(
                    f"{ollama_base_url.rstrip('/')}/api/generate",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=8.0) as resp:
                    resp_bytes = resp.read()
                    elapsed_wall_ms = (time.perf_counter() - t0) * 1000.0
                    try:
                        raw_text = resp_bytes.decode("utf-8")
                        for line in raw_text.strip().split("\n"):
                            if not line.strip():
                                continue
                            p_line = _json.loads(line)
                            if isinstance(p_line, dict):
                                if "load_duration" in p_line:
                                    load_dur_ms = float(p_line["load_duration"]) / 1_000_000.0
                                if "total_duration" in p_line:
                                    total_dur_ms = float(p_line["total_duration"]) / 1_000_000.0
                    except Exception:
                        pass

                    if load_dur_ms <= 0.0:
                        load_dur_ms = elapsed_wall_ms
                    if total_dur_ms <= 0.0:
                        total_dur_ms = elapsed_wall_ms

                self._cephalic_state = "WARMED"
                self.record_model_load_telemetry(
                    model_name=model_name,
                    load_duration_ms=load_dur_ms,
                    total_duration_ms=total_dur_ms,
                    runtime="ollama",
                    configured_ttl_sec=configured_ttl_sec,
                    stimulus_source=norm_stimulus,
                )
            except Exception as exc:
                self._cephalic_state = "WARMED_PAGE_CACHE"
                logger.debug(f"Cephalic pre-warm probe skipped for {model_name}: {exc}")

        if background:
            t = threading.Thread(target=_do_warm, name=f"cephalic-prewarm-{model_name}", daemon=True)
            t.start()
            return {
                "status": "warming_async",
                "mechanism": "Cephalic Phase Predictive Pre-Warm",
                "stimulus_source": norm_stimulus,
                "keystroke_buffer_len": int(keystroke_buffer_len or 0),
                "model": model_name,
                "keep_alive_ttl": effective_keep_alive,
                "adaptive_policy": adaptive_pol,
                "kernel_page_cache": page_cache_handoff,
            }
        else:
            t0 = time.perf_counter()
            _do_warm()
            dur_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            return {
                "status": "warm",
                "mechanism": "Cephalic Phase Predictive Pre-Warm",
                "stimulus_source": norm_stimulus,
                "keystroke_buffer_len": int(keystroke_buffer_len or 0),
                "model": model_name,
                "keep_alive_ttl": effective_keep_alive,
                "adaptive_policy": adaptive_pol,
                "duration_ms": dur_ms,
                "kernel_page_cache": page_cache_handoff,
            }



# Singleton instance
default_supervisor = ThermalSupervisor()
PerfusionManager = ThermalSupervisor
default_perfusion_manager = default_supervisor


def record_gate_event(
    requested_model: str,
    resolved_model: Optional[str],
    status: str,
    reason: str,
    free_ram_gb: float = 0.0,
    required_ram_gb: float = 0.0,
    reclaimed_gb: float = 0.0,
) -> Dict[str, Any]:
    """Module-level helper to record a ResourceGate event via `default_supervisor`."""
    return default_supervisor.record_gate_event(
        requested_model=requested_model,
        resolved_model=resolved_model,
        status=status,
        reason=reason,
        free_ram_gb=free_ram_gb,
        required_ram_gb=required_ram_gb,
        reclaimed_gb=reclaimed_gb,
    )


def get_last_gate_event() -> Dict[str, Any]:
    """Module-level helper to retrieve the latest ResourceGate event via `default_supervisor`."""
    return default_supervisor.get_last_gate_event()


class SensoryCephalicReflex:
    """
    Pre-Submission Sensory Reflex Controller (Organ 01 -> Organ 02).
    Detects operator intent BEFORE prompt submission (keystroke-onset, IDE window focus shift,
    or acoustic VAD speech onset) and triggers asynchronous page-table remapping from the OS
    warm page cache into Active Resident Memory.
    """

    def __init__(self, supervisor: Optional[ThermalSupervisor] = None) -> None:
        self.supervisor = supervisor or default_supervisor

    def trigger_cephalic_prewarm(
        self,
        model_name: str = "qwen2.5-coder:1.5b",
        keep_alive: str = "auto",
        ollama_base_url: str = "http://127.0.0.1:11434",
        background: bool = True,
        stimulus_source: str = "vad_speech_onset",
        keystroke_buffer_len: int = 0,
        debounce_sec: float = 0.0,
        weights_path: Optional[str | Path] = None,
        os_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.supervisor.trigger_cephalic_prewarm(
            model_name=model_name,
            keep_alive=keep_alive,
            ollama_base_url=ollama_base_url,
            background=background,
            stimulus_source=stimulus_source,
            keystroke_buffer_len=keystroke_buffer_len,
            debounce_sec=debounce_sec,
            weights_path=weights_path,
            os_override=os_override,
        )

    def on_keystroke_onset(
        self,
        prompt_buffer: str = "",
        model_name: str = "qwen2.5-coder:1.5b",
        keep_alive: str = "auto",
        debounce_sec: float = 5.0,
        background: bool = True,
        weights_path: Optional[str | Path] = None,
        os_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Triggered when user begins typing in an IDE/prompt buffer prior to pressing Enter."""
        return self.trigger_cephalic_prewarm(
            model_name=model_name,
            keep_alive=keep_alive,
            background=background,
            stimulus_source="keystroke_onset",
            keystroke_buffer_len=len(prompt_buffer or ""),
            debounce_sec=debounce_sec,
            weights_path=weights_path,
            os_override=os_override,
        )

    def on_ide_focus_shift(
        self,
        app_bundle_id: str = "com.google.antigravity",
        model_name: str = "qwen2.5-coder:1.5b",
        keep_alive: str = "auto",
        debounce_sec: float = 10.0,
        background: bool = True,
        weights_path: Optional[str | Path] = None,
        os_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Triggered when IDE window or editor tab gains foreground focus."""
        res = self.trigger_cephalic_prewarm(
            model_name=model_name,
            keep_alive=keep_alive,
            background=background,
            stimulus_source="ide_focus_shift",
            debounce_sec=debounce_sec,
            weights_path=weights_path,
            os_override=os_override,
        )
        res["app_bundle_id"] = app_bundle_id
        return res

    def on_vad_speech_onset(
        self,
        model_name: str = "qwen2.5-coder:1.5b",
        keep_alive: str = "auto",
        background: bool = True,
        weights_path: Optional[str | Path] = None,
        os_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Triggered upon acoustic Voice Activity Detection (VAD) speech onset."""
        return self.trigger_cephalic_prewarm(
            model_name=model_name,
            keep_alive=keep_alive,
            background=background,
            stimulus_source="vad_speech_onset",
            weights_path=weights_path,
            os_override=os_override,
        )

    def advise_kernel_page_cache(
        self,
        weights_path: Optional[str | Path] = None,
        mode: str = "willneed",
        os_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute cross-platform OS kernel page-cache handoff (`Darwin`, `Linux`, `Windows`)."""
        return self.supervisor.advise_kernel_page_cache(
            weights_path=weights_path,
            mode=mode,
            os_override=os_override,
        )


default_sensory_reflex = SensoryCephalicReflex()


def trigger_cephalic_prewarm(
    model_name: str = "qwen2.5-coder:1.5b",
    keep_alive: str = "auto",
    ollama_base_url: str = "http://127.0.0.1:11434",
    background: bool = True,
    stimulus_source: str = "vad_speech_onset",
    keystroke_buffer_len: int = 0,
    debounce_sec: float = 0.0,
    weights_path: Optional[str | Path] = None,
    os_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Module-level convenience entrypoint for Cephalic Pre-Warming."""
    return default_sensory_reflex.trigger_cephalic_prewarm(
        model_name=model_name,
        keep_alive=keep_alive,
        ollama_base_url=ollama_base_url,
        background=background,
        stimulus_source=stimulus_source,
        keystroke_buffer_len=keystroke_buffer_len,
        debounce_sec=debounce_sec,
        weights_path=weights_path,
        os_override=os_override,
    )


class VagusBrowserAbortError(RuntimeError):
    """Raised when browser automation attempts an action while the Vagus Nerve Brake is tripped."""

    pass


class BrowserVagusBrake:
    """
    Browser Vagus Nerve Brake: Sub-50ms Spoken Abort for Playwright & CDP Loops.
    Maintains an in-memory and atomic state-file latch that instantly interrupts active
    browser automation sessions, aborts pending form actions/clicks, and signals browser workers.
    """

    DEFAULT_STATE_PATH = Path.home() / ".voicefi" / "vagus_browser_state.json"

    # Conversational negation patterns that cancel an apparent abort
    NEGATION_RE = re.compile(
        r"\b(?:don'?t\s+stop|can'?t\s+wait|never\s+stop|please\s+continue|keep\s+going|not\s+stopping)\b",
        re.IGNORECASE,
    )
    # Spoken interruption triggers
    INTERRUPTION_RE = re.compile(
        r"\b(?:stop|wait|hold\s+on|cancel|pause|abort|freeze|halt|no\s+no|shut\s+up|kill)\b",
        re.IGNORECASE,
    )

    def __init__(self, state_path: Optional[Path] = None) -> None:
        self.state_path = state_path or self.DEFAULT_STATE_PATH
        self._lock = threading.Lock()
        self._tripped: bool = False
        self._trip_reason: str = ""
        self._trip_timestamp: float = 0.0
        self._active_sessions: Dict[str, Dict[str, Any]] = {}

    def _resolve_state_path(self, state_path: Optional[Path] = None) -> Path:
        return state_path or self.state_path

    def _sync_to_disk(self, target_path: Path) -> None:
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "tripped": self._tripped,
                "reason": self._trip_reason,
                "timestamp": self._trip_timestamp,
                "sessions": self._active_sessions,
            }
            tmp_path = target_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, target_path)
        except Exception:
            pass

    def _read_disk_tripped(self, target_path: Path) -> bool:
        if not target_path.exists():
            return False
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return bool(data.get("tripped", False))
        except Exception:
            return False

    def register_browser_session(
        self,
        session_id: str,
        pid: Optional[int] = None,
        url: Optional[str] = None,
        action: Optional[str] = None,
        state_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Register an active browser automation session and its worker PID."""
        target = self._resolve_state_path(state_path)
        with self._lock:
            info = {
                "session_id": session_id,
                "pid": pid,
                "url": url,
                "action": action,
                "registered_at": time.time(),
            }
            self._active_sessions[session_id] = info
            self._sync_to_disk(target)
            return info

    def unregister_browser_session(
        self,
        session_id: str,
        state_path: Optional[Path] = None,
    ) -> bool:
        """Unregister a completed or aborted browser session."""
        target = self._resolve_state_path(state_path)
        with self._lock:
            if session_id in self._active_sessions:
                del self._active_sessions[session_id]
                self._sync_to_disk(target)
                return True
            return False

    def clear_brake_latch(self, state_path: Optional[Path] = None) -> None:
        """Clear tripped latch in-memory and in the atomic state file."""
        target = self._resolve_state_path(state_path)
        with self._lock:
            self._tripped = False
            self._trip_reason = ""
            self._trip_timestamp = 0.0
            self._sync_to_disk(target)

    def is_interruption_spoken(self, spoken_text: str) -> bool:
        """Evaluate if spoken text is an intentional abort, respecting conversational negations."""
        if not spoken_text or not spoken_text.strip():
            return False
        clean = spoken_text.strip().lower()

        # Negation guard: "don't stop", "can't wait", "never stop" MUST NOT trip the brake
        if self.NEGATION_RE.search(clean):
            return False

        # Interruption regex (stop, wait, hold on, cancel, pause, abort, freeze, halt, etc.)
        if bool(self.INTERRUPTION_RE.search(clean)):
            return True

        # Attempt acoustic cognitive safety classifier if available
        try:
            from whole_body_compute.audio.cognitive_safety import AcousticCognitiveSafetyClassifier

            if AcousticCognitiveSafetyClassifier.is_interruption_text(clean):
                return True
        except Exception:
            pass

        return False

    def trigger_browser_vagus_brake(
        self,
        spoken_text: str,
        mode: str = "abort",
        state_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Trip the Browser Vagus Nerve Brake upon acoustic barge-in.
        Executes in <50ms (typically <5ms):
        1. Evaluates conversational negations and interruption intent.
        2. Sets atomic state latch on disk and in memory.
        3. Dispatches signals to registered worker PIDs (SIGSTOP for freeze, SIGINT for abort).
        4. Generates CDP emergency stop commands for DevTools / Playwright.
        """
        t0 = time.perf_counter()
        target = self._resolve_state_path(state_path)

        if not self.is_interruption_spoken(spoken_text):
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            return {
                "triggered": False,
                "mode": mode,
                "spoken_text": spoken_text,
                "signaled_pids": [],
                "cdp_emergency_commands": [],
                "latency_ms": latency_ms,
            }

        signaled_pids: List[int] = []
        with self._lock:
            self._tripped = True
            self._trip_reason = spoken_text
            self._trip_timestamp = time.time()
            self._sync_to_disk(target)

            sig = signal.SIGSTOP if mode == "freeze" else signal.SIGINT
            current_pid = os.getpid()

            for sess in list(self._active_sessions.values()):
                pid = sess.get("pid")
                if pid and isinstance(pid, int) and pid > 1 and pid != current_pid:
                    try:
                        # Attempt process group signaling first, fallback to individual pid
                        try:
                            pgid = os.getpgid(pid)
                            if pgid != os.getpgrp():
                                os.killpg(pgid, sig)
                            else:
                                os.kill(pid, sig)
                        except (OSError, ProcessLookupError):
                            os.kill(pid, sig)
                        signaled_pids.append(pid)
                    except (ProcessLookupError, PermissionError):
                        pass

        cdp_emergency_commands = [
            {"method": "Page.stopLoading", "params": {}},
            {
                "method": "Runtime.evaluate",
                "params": {
                    "expression": "window.stop(); if (document.activeElement) document.activeElement.blur();",
                },
            },
        ]

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return {
            "triggered": True,
            "mode": mode,
            "spoken_text": spoken_text,
            "signaled_pids": signaled_pids,
            "cdp_emergency_commands": cdp_emergency_commands,
            "latency_ms": latency_ms,
        }

    def guard_browser_action(
        self,
        action_type: str,
        selector_or_url: str = "",
        state_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Pre-action gate called before Playwright / CDP steps (click, fill, submit, navigate).
        Checks in-memory latch and state file in <1ms.
        Raises VagusBrowserAbortError if tripped.
        """
        target = self._resolve_state_path(state_path)
        is_tripped = False
        reason = ""
        with self._lock:
            if self._tripped:
                is_tripped = True
                reason = self._trip_reason
            elif self._read_disk_tripped(target):
                is_tripped = True
                self._tripped = True

        if is_tripped:
            msg = f"Vagus Nerve Brake tripped before {action_type} on {selector_or_url}"
            if reason:
                msg += f" (spoken: '{reason}')"
            raise VagusBrowserAbortError(msg)

        return {
            "allowed": True,
            "action_type": action_type,
            "target": selector_or_url,
        }


default_browser_brake = BrowserVagusBrake()


def trigger_browser_vagus_brake(
    spoken_text: str,
    mode: str = "abort",
    state_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Module-level entrypoint for Browser Vagus Nerve Brake."""
    return default_browser_brake.trigger_browser_vagus_brake(
        spoken_text=spoken_text,
        mode=mode,
        state_path=state_path,
    )


def guard_browser_action(
    action_type: str,
    selector_or_url: str = "",
    state_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Pre-action gate called before Playwright / CDP steps."""
    return default_browser_brake.guard_browser_action(
        action_type=action_type,
        selector_or_url=selector_or_url,
        state_path=state_path,
    )


def register_browser_session(
    session_id: str,
    pid: Optional[int] = None,
    url: Optional[str] = None,
    action: Optional[str] = None,
    state_path: Optional[Path] = None,
) -> Dict[str, Any]:
    return default_browser_brake.register_browser_session(
        session_id=session_id,
        pid=pid,
        url=url,
        action=action,
        state_path=state_path,
    )


def unregister_browser_session(
    session_id: str,
    state_path: Optional[Path] = None,
) -> bool:
    return default_browser_brake.unregister_browser_session(
        session_id=session_id,
        state_path=state_path,
    )


def clear_browser_brake(state_path: Optional[Path] = None) -> None:
    default_browser_brake.clear_brake_latch(state_path=state_path)


if __name__ == "__main__":
    telem = default_supervisor.get_telemetry()
    print("=== Apple Silicon Hardware Telemetry ===")
    print(f"System:         {telem.system}")
    print(f"Thermal State:  {telem.thermal_state}")
    print(f"Thermal Raw:    {telem.thermal_raw}")
    print(
        f"Memory:         {telem.free_ram_gb} GB free / {telem.total_ram_gb} GB total ({telem.ram_percent}% used)"
    )
    print(f"Active Swap:    {telem.swap_used_gb} GB")
    print(f"1m Load Avg:    {telem.load_avg_1m}")
    print(f"Safe to Run:    {telem.is_safe}")
    if telem.throttle_reason:
        print(f"Throttle Cause: {telem.throttle_reason}")

    print("\n=== Model Resource Headroom Gate Check ===")
    for model in ("gemma2:2b", "gemma4-26b", "llama3.3:70b"):
        approved, reason, alert = default_supervisor.can_load_model(model)
        status = "✅ APPROVED" if approved else "⛔ BLOCKED"
        print(f"{model:<16} -> {status}: {reason}")
        if alert:
            print(default_supervisor.format_resource_alert(alert))
