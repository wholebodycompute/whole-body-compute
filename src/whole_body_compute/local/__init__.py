"""
VoiceFi Local Model & Recon Scout Engine.
Leverages Google AI Edge's LiteRT and Google Antigravity SDK to execute
on-device models (Gemma 4) on Apple Silicon Metal GPUs for 100% private,
zero-cost recon, code auditing, and benchmark analysis.
"""

from typing import Any
import importlib

from whole_body_compute.local.engine import LocalModelEngine, is_litert_available
from whole_body_compute.local.supervisor import (
    ThermalSupervisor,
    HardwareTelemetry,
    ResourceGateError,
    default_supervisor,
)

_LAZY_IMPORTS = {
    "ReconScout": ("whole_body_compute.local.scout", "ReconScout"),
    "ScoutResult": ("whole_body_compute.local.scout", "ScoutResult"),
    "ReconImplementer": ("whole_body_compute.local.implementer", "ReconImplementer"),
    "ImplementResult": ("whole_body_compute.local.implementer", "ImplementResult"),
    "LocalBenchmarkRunner": ("whole_body_compute.local.benchmark", "LocalBenchmarkRunner"),
    "BenchmarkResult": ("whole_body_compute.local.benchmark", "BenchmarkResult"),
    "ToTComparisonResult": ("whole_body_compute.local.benchmark", "ToTComparisonResult"),
    "TurnMetrics": ("whole_body_compute.local.benchmark", "TurnMetrics"),
    "record_inference_metrics": ("whole_body_compute.local.benchmark", "record_inference_metrics"),
    "measure_unified_ram_ingress": ("whole_body_compute.local.benchmark", "measure_unified_ram_ingress"),
    "measure_wan_ingress": ("whole_body_compute.local.benchmark", "measure_wan_ingress"),
    "LocalIntentRouter": ("whole_body_compute.local.intent", "LocalIntentRouter"),
    "parse_traceback": ("whole_body_compute.local.trace_parser", "parse_traceback"),
    "ParsedTrace": ("whole_body_compute.local.trace_parser", "ParsedTrace"),
    "LocalSymbolIndex": ("whole_body_compute.local.symbols", "LocalSymbolIndex"),
    "SymbolRecord": ("whole_body_compute.local.symbols", "SymbolRecord"),
    "LocalAutonomousLoop": ("whole_body_compute.local.agent_loop", "LocalAutonomousLoop"),
    "AutonomousLoopResult": ("whole_body_compute.local.agent_loop", "AutonomousLoopResult"),
    "LoopStep": ("whole_body_compute.local.agent_loop", "LoopStep"),
    "HybridRouter": ("whole_body_compute.local.router", "HybridRouter"),
    "RoutingDecision": ("whole_body_compute.local.router", "RoutingDecision"),
    "default_router": ("whole_body_compute.local.router", "default_router"),
    "HybridStepRecord": ("whole_body_compute.local.telemetry", "HybridStepRecord"),
    "HybridPipelineContext": ("whole_body_compute.local.telemetry", "HybridPipelineContext"),
    "record_hybrid_step": ("whole_body_compute.local.telemetry", "record_hybrid_step"),
    "update_workspace_scorecard": ("whole_body_compute.local.telemetry", "update_workspace_scorecard"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        mod_name, attr_name = _LAZY_IMPORTS[name]
        mod = importlib.import_module(mod_name)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LocalModelEngine",
    "is_litert_available",
    "ReconScout",
    "ScoutResult",
    "ReconImplementer",
    "ImplementResult",
    "LocalBenchmarkRunner",
    "BenchmarkResult",
    "ToTComparisonResult",
    "TurnMetrics",
    "record_inference_metrics",
    "measure_unified_ram_ingress",
    "measure_wan_ingress",
    "LocalIntentRouter",
    "parse_traceback",
    "ParsedTrace",
    "LocalSymbolIndex",
    "SymbolRecord",
    "LocalAutonomousLoop",
    "AutonomousLoopResult",
    "LoopStep",
    "ThermalSupervisor",
    "HardwareTelemetry",
    "ResourceGateError",
    "default_supervisor",
    "HybridRouter",
    "RoutingDecision",
    "default_router",
    "HybridStepRecord",
    "HybridPipelineContext",
    "record_hybrid_step",
    "update_workspace_scorecard",
]
