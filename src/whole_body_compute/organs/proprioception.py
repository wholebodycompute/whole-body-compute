"""
Subconscious Multi-Harness Proprioception & Visceral Interoception (`SensoryVector`).

Implements Phase 1 of `docs/AGENT_PROPRIOCEPTION_MASTER_PLAN.md`:
  - Sub-0.2ms local computation of `ProprioceptiveState` (host harness identity, physical line/byte
    tool ceilings, and read intent) and `InteroceptiveState` (thread step index, cognitive adenosine
    CSI estimate, Apple Silicon thermal state, truncation cascade state, and homeostatic regime).
  - Enforces the 4 Subconscious Autonomic Firewall Invariants:
      1. Sub-0.5ms Local Evaluation (`0` prompt tokens for raw telemetry).
      2. Silent In-Flight Auto-Healing (`0` retry turns when auto-slicing).
      3. Silent Homeostasis (`0` meta-commentary characters when nominal; single <=15-word
         autonomic reflex cue only when critical homeostatic thresholds are breached).
      4. Out-of-Band Human Cockpit (`immune_audit.jsonl` & telemetry HUD).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# Physical tool ceiling envelopes per host harness
HARNESS_ENVELOPES: Dict[str, Dict[str, int]] = {
    "antigravity": {
        "native_line_ceiling": 800,
        "native_byte_ceiling": 46080,
        "post_output_line_ceiling": 500,
        "post_output_byte_ceiling": 36000,
    },
    "claude_code": {
        "native_line_ceiling": 2000,
        "native_byte_ceiling": 200000,
        "post_output_line_ceiling": 1000,
        "post_output_byte_ceiling": 100000,
    },
    "cursor": {
        "native_line_ceiling": 1000,
        "native_byte_ceiling": 60000,
        "post_output_line_ceiling": 500,
        "post_output_byte_ceiling": 36000,
    },
    "windsurf": {
        "native_line_ceiling": 800,
        "native_byte_ceiling": 46080,
        "post_output_line_ceiling": 500,
        "post_output_byte_ceiling": 36000,
    },
    "goose": {
        "native_line_ceiling": 1000,
        "native_byte_ceiling": 100000,
        "post_output_line_ceiling": 500,
        "post_output_byte_ceiling": 36000,
    },
    "generic": {
        "native_line_ceiling": 800,
        "native_byte_ceiling": 46080,
        "post_output_line_ceiling": 500,
        "post_output_byte_ceiling": 36000,
    },
}

_ANTIGRAVITY_TOOLS = frozenset(
    {
        "read_file",
        "view_file",
        "edit_file",
        "write_file",
        "run_command",
        "grep_search",
        "find_by_name",
        "list_dir",
        "invoke_subagent",
        "send_message",
        "manage_subagents",
        "manage_task",
        "call_mcp_tool",
        "ask_question",
        "search_web",
        "search_marketplace",
        "list_resources",
        "read_resource",
    }
)

_CLAUDE_CODE_TOOLS = frozenset(
    {
        "Read",
        "NotebookRead",
        "Edit",
        "MultiEdit",
        "Write",
        "NotebookEdit",
        "Bash",
        "Grep",
        "Glob",
        "LS",
        "Task",
        "Agent",
        "WebFetch",
        "TodoWrite",
    }
)

_GOOSE_TOOLS = frozenset(
    {
        "developer__shell",
        "developer__text_editor",
        "computercontroller__computer_control",
        "screenpipe__inspect_foveal_window_dom",
    }
)

# Lightweight in-memory thermal cache to avoid subprocess overhead on the <0.2ms hook path
_THERMAL_CACHE_STATE: str = "NORMAL"
_THERMAL_CACHE_TS: float = 0.0
_THERMAL_CACHE_TTL_SEC: float = 5.0


@dataclass(slots=True)
class ProprioceptiveState:
    """
    External spatial & tool-envelope awareness for the active agent harness.
    """

    harness_name: str  # "antigravity" | "claude_code" | "cursor" | "windsurf" | "goose" | "generic"
    native_line_ceiling: int  # 800 (antigravity/windsurf/generic), 1000 (cursor/goose), 2000 (claude_code)
    native_byte_ceiling: int  # 46080 (antigravity/windsurf/generic), 60000 (cursor), 100000 (goose), 200000 (claude_code)
    read_intent: str  # "deliberate_bounded" | "predictive_query" | "blind_unbounded"
    post_output_line_ceiling: int = 500
    post_output_byte_ceiling: int = 36000
    has_bounds: bool = False
    requested_start_line: Optional[int] = None
    requested_end_line: Optional[int] = None
    requested_span_lines: int = 999999

    @property
    def tail_bridge_extra_lines(self) -> int:
        """
        Maximum extra lines bridged beyond `native_line_ceiling` in 1-turn Zero-Donut reads.
        Harnesses with >= 2,000L native ceilings (e.g. Claude Code) use 0 spillover so files
        > 2,000L route directly to Foveal AST Jump + Import Pinning (~4x lower token burn).
        """
        if self.native_line_ceiling >= 2000:
            return 0
        return min(350, max(200, int(round(self.native_line_ceiling * 0.35))))

    @property
    def tail_bridge_line_ceiling(self) -> int:
        """Upper total-file line ceiling eligible for 1-turn Lossless Tail-Bridge (`800->1080`, `1000->1350`, `2000->2000`)."""
        return self.native_line_ceiling + self.tail_bridge_extra_lines

    @property
    def tail_bridge_head_byte_ceiling(self) -> int:
        """Byte budget for the native head window before Lossless Tail-Bridge yields to Foveal AST Jump."""
        if self.native_byte_ceiling == 46080:
            return 42000
        return max(4096, int(self.native_byte_ceiling * 0.91))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "harness_name": self.harness_name,
            "native_line_ceiling": self.native_line_ceiling,
            "native_byte_ceiling": self.native_byte_ceiling,
            "tail_bridge_extra_lines": self.tail_bridge_extra_lines,
            "tail_bridge_line_ceiling": self.tail_bridge_line_ceiling,
            "tail_bridge_head_byte_ceiling": self.tail_bridge_head_byte_ceiling,
            "read_intent": self.read_intent,
            "post_output_line_ceiling": self.post_output_line_ceiling,
            "post_output_byte_ceiling": self.post_output_byte_ceiling,
            "has_bounds": self.has_bounds,
            "requested_start_line": self.requested_start_line,
            "requested_end_line": self.requested_end_line,
            "requested_span_lines": self.requested_span_lines,
        }


@dataclass(slots=True)
class InteroceptiveState:
    """
    Internal metabolic & hardware homeostasis state for the active conversation thread.
    """

    step_idx: int
    csi_est: float
    thermal_state: str  # "NORMAL" | "FAIR" | "SERIOUS" | "CRITICAL"
    truncation_cascade_active: bool
    homeostatic_regime: str  # "GREEN_ELASTIC" | "YELLOW_BALANCED" | "RED_TIGHT"
    transcript_kb: float = 0.0
    retry_churn_active: bool = False
    retry_churn_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_idx": self.step_idx,
            "csi_est": self.csi_est,
            "thermal_state": self.thermal_state,
            "truncation_cascade_active": self.truncation_cascade_active,
            "homeostatic_regime": self.homeostatic_regime,
            "transcript_kb": self.transcript_kb,
            "retry_churn_active": self.retry_churn_active,
            "retry_churn_count": self.retry_churn_count,
        }


class SensoryVector:
    """
    Unified Bicameral Sensory Vector combining `ProprioceptiveState` and `InteroceptiveState`.
    Computed in <0.2ms inside `PreToolUse` / `PostToolUse` at zero prompt token cost.
    """

    __slots__ = (
        "proprioception",
        "interoception",
        "compute_latency_ms",
        "effective_file_threshold",
        "effective_max_span",
        "auto_slice_window",
        "effective_post_line_ceiling",
        "effective_post_byte_ceiling",
        "predictive_ast_enabled",
        "autonomic_reflex_cue",
        "distraction_tokens",
    )

    def __init__(
        self,
        proprioception: Optional[ProprioceptiveState] = None,
        interoception: Optional[InteroceptiveState] = None,
        *,
        harness_name: str = "antigravity",
        native_line_ceiling: Optional[int] = None,
        native_byte_ceiling: Optional[int] = None,
        read_intent: str = "blind_unbounded",
        post_output_line_ceiling: Optional[int] = None,
        post_output_byte_ceiling: Optional[int] = None,
        has_bounds: bool = False,
        requested_start_line: Optional[int] = None,
        requested_end_line: Optional[int] = None,
        requested_span_lines: int = 999999,
        step_idx: int = 0,
        csi_est: Optional[float] = None,
        thermal_state: str = "NORMAL",
        truncation_cascade_active: bool = False,
        homeostatic_regime: Optional[str] = None,
        transcript_kb: float = 0.0,
        retry_churn_active: bool = False,
        retry_churn_count: int = 0,
        compute_latency_ms: float = 0.0,
        effective_file_threshold: Optional[int] = None,
        effective_max_span: Optional[int] = None,
        auto_slice_window: Optional[int] = None,
        effective_post_line_ceiling: Optional[int] = None,
        effective_post_byte_ceiling: Optional[int] = None,
        predictive_ast_enabled: Optional[bool] = None,
        autonomic_reflex_cue: Optional[str] = None,
    ) -> None:
        env_spec = HARNESS_ENVELOPES.get(harness_name, HARNESS_ENVELOPES["generic"])
        if proprioception is None:
            proprioception = ProprioceptiveState(
                harness_name=harness_name,
                native_line_ceiling=native_line_ceiling
                if native_line_ceiling is not None
                else env_spec["native_line_ceiling"],
                native_byte_ceiling=native_byte_ceiling
                if native_byte_ceiling is not None
                else env_spec["native_byte_ceiling"],
                read_intent=read_intent,
                post_output_line_ceiling=post_output_line_ceiling
                if post_output_line_ceiling is not None
                else env_spec["post_output_line_ceiling"],
                post_output_byte_ceiling=post_output_byte_ceiling
                if post_output_byte_ceiling is not None
                else env_spec["post_output_byte_ceiling"],
                has_bounds=has_bounds,
                requested_start_line=requested_start_line,
                requested_end_line=requested_end_line,
                requested_span_lines=requested_span_lines,
            )

        if interoception is None:
            computed_csi = (
                float(csi_est)
                if csi_est is not None
                else estimate_subconscious_csi(step_idx=step_idx, transcript_kb=transcript_kb)
            )
            computed_regime = homeostatic_regime or resolve_homeostatic_regime(
                step_idx=step_idx,
                csi_est=computed_csi,
                thermal_state=thermal_state,
                truncation_cascade_active=truncation_cascade_active,
            )
            interoception = InteroceptiveState(
                step_idx=step_idx,
                csi_est=computed_csi,
                thermal_state=thermal_state,
                truncation_cascade_active=truncation_cascade_active,
                homeostatic_regime=computed_regime,
                transcript_kb=transcript_kb,
                retry_churn_active=retry_churn_active,
                retry_churn_count=retry_churn_count,
            )

        self.proprioception: ProprioceptiveState = proprioception
        self.interoception: InteroceptiveState = interoception
        self.compute_latency_ms: float = compute_latency_ms

        regime = self.interoception.homeostatic_regime
        line_ceil = self.proprioception.native_line_ceiling
        post_line_base = self.proprioception.post_output_line_ceiling
        post_byte_base = self.proprioception.post_output_byte_ceiling

        if effective_file_threshold is not None:
            self.effective_file_threshold = effective_file_threshold
        elif self.interoception.truncation_cascade_active:
            self.effective_file_threshold = 80
        elif regime == "RED_TIGHT":
            self.effective_file_threshold = (
                80 if self.interoception.thermal_state == "CRITICAL" else min(line_ceil, 400)
            )
        else:
            self.effective_file_threshold = line_ceil

        if effective_max_span is not None:
            self.effective_max_span = effective_max_span
        elif self.interoception.truncation_cascade_active:
            if self.proprioception.read_intent == "deliberate_bounded":
                self.effective_max_span = min(line_ceil, 800)
            else:
                self.effective_max_span = 80
        elif regime == "RED_TIGHT":
            if self.proprioception.read_intent == "deliberate_bounded":
                self.effective_max_span = min(line_ceil, 800)
            else:
                self.effective_max_span = 80
        else:
            self.effective_max_span = line_ceil

        if auto_slice_window is not None:
            self.auto_slice_window = auto_slice_window
        elif regime == "RED_TIGHT" or self.interoception.truncation_cascade_active:
            self.auto_slice_window = 80
        else:
            self.auto_slice_window = 160

        if effective_post_line_ceiling is not None:
            self.effective_post_line_ceiling = effective_post_line_ceiling
        elif self.interoception.truncation_cascade_active:
            self.effective_post_line_ceiling = 80
        elif regime == "RED_TIGHT":
            self.effective_post_line_ceiling = max(80, int(post_line_base * 0.32))
        elif regime == "YELLOW_BALANCED":
            self.effective_post_line_ceiling = max(160, int(post_line_base * 0.70))
        else:
            self.effective_post_line_ceiling = post_line_base

        if effective_post_byte_ceiling is not None:
            self.effective_post_byte_ceiling = effective_post_byte_ceiling
        elif self.interoception.truncation_cascade_active:
            self.effective_post_byte_ceiling = 6000
        elif regime == "RED_TIGHT":
            self.effective_post_byte_ceiling = max(8000, post_byte_base // 3)
        elif regime == "YELLOW_BALANCED":
            self.effective_post_byte_ceiling = max(12000, (post_byte_base * 3) // 4)
        else:
            self.effective_post_byte_ceiling = post_byte_base

        if predictive_ast_enabled is not None:
            self.predictive_ast_enabled = bool(predictive_ast_enabled)
        else:
            self.predictive_ast_enabled = bool(
                regime == "RED_TIGHT"
                or self.proprioception.read_intent == "predictive_query"
                or self.interoception.retry_churn_active
            )

        if autonomic_reflex_cue is not None:
            self.autonomic_reflex_cue = autonomic_reflex_cue
        else:
            self.autonomic_reflex_cue = format_autonomic_reflex_cue(self.interoception)

        self.distraction_tokens: int = 0

    # Proprioception convenience properties
    @property
    def harness_name(self) -> str:
        return self.proprioception.harness_name

    @property
    def native_line_ceiling(self) -> int:
        return self.proprioception.native_line_ceiling

    @property
    def native_byte_ceiling(self) -> int:
        return self.proprioception.native_byte_ceiling

    @property
    def tail_bridge_extra_lines(self) -> int:
        return self.proprioception.tail_bridge_extra_lines

    @property
    def tail_bridge_line_ceiling(self) -> int:
        return self.proprioception.tail_bridge_line_ceiling

    @property
    def tail_bridge_head_byte_ceiling(self) -> int:
        return self.proprioception.tail_bridge_head_byte_ceiling

    @property
    def read_intent(self) -> str:
        return self.proprioception.read_intent

    @property
    def post_output_line_ceiling(self) -> int:
        return self.proprioception.post_output_line_ceiling

    @property
    def post_output_byte_ceiling(self) -> int:
        return self.proprioception.post_output_byte_ceiling

    @property
    def has_bounds(self) -> bool:
        return self.proprioception.has_bounds

    @property
    def requested_start_line(self) -> Optional[int]:
        return self.proprioception.requested_start_line

    @property
    def requested_end_line(self) -> Optional[int]:
        return self.proprioception.requested_end_line

    @property
    def requested_span_lines(self) -> int:
        return self.proprioception.requested_span_lines

    # Interoception convenience properties
    @property
    def step_idx(self) -> int:
        return self.interoception.step_idx

    @property
    def csi_est(self) -> float:
        return self.interoception.csi_est

    @property
    def thermal_state(self) -> str:
        return self.interoception.thermal_state

    @property
    def truncation_cascade_active(self) -> bool:
        return self.interoception.truncation_cascade_active

    @property
    def homeostatic_regime(self) -> str:
        return self.interoception.homeostatic_regime

    @property
    def retry_churn_active(self) -> bool:
        return self.interoception.retry_churn_active

    @property
    def retry_churn_count(self) -> int:
        return self.interoception.retry_churn_count

    @property
    def is_nominal(self) -> bool:
        """
        True when inside Silent Homeostasis (Subconscious Firewall Invariant 3):
        CSI < 0.70, Thermal in ('NORMAL', 'FAIR'), no active truncation cascade,
        and no active retry-churn loop.
        """
        return (
            self.interoception.csi_est < 0.70
            and self.interoception.thermal_state in ("NORMAL", "FAIR")
            and not self.interoception.truncation_cascade_active
            and not self.interoception.retry_churn_active
            and self.interoception.homeostatic_regime != "RED_TIGHT"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "harness_name": self.harness_name,
            "native_line_ceiling": self.native_line_ceiling,
            "native_byte_ceiling": self.native_byte_ceiling,
            "read_intent": self.read_intent,
            "step_idx": self.step_idx,
            "csi_est": self.csi_est,
            "thermal_state": self.thermal_state,
            "truncation_cascade_active": self.truncation_cascade_active,
            "retry_churn_active": self.retry_churn_active,
            "retry_churn_count": self.retry_churn_count,
            "homeostatic_regime": self.homeostatic_regime,
            "effective_file_threshold": self.effective_file_threshold,
            "effective_max_span": self.effective_max_span,
            "auto_slice_window": self.auto_slice_window,
            "effective_post_line_ceiling": self.effective_post_line_ceiling,
            "effective_post_byte_ceiling": self.effective_post_byte_ceiling,
            "predictive_ast_enabled": self.predictive_ast_enabled,
            "distraction_tokens": self.distraction_tokens,
            "is_nominal": self.is_nominal,
            "compute_latency_ms": self.compute_latency_ms,
            "proprioception": self.proprioception.to_dict(),
            "interoception": self.interoception.to_dict(),
        }

    def to_audit_dict(self) -> Dict[str, Any]:
        """Compact out-of-band telemetry representation for `immune_audit.jsonl`."""
        return {
            "harness_name": self.harness_name,
            "native_line_ceiling": self.native_line_ceiling,
            "native_byte_ceiling": self.native_byte_ceiling,
            "read_intent": self.read_intent,
            "step_idx": self.step_idx,
            "csi_est": self.csi_est,
            "thermal_state": self.thermal_state,
            "truncation_cascade_active": self.truncation_cascade_active,
            "retry_churn_active": self.retry_churn_active,
            "retry_churn_count": self.retry_churn_count,
            "homeostatic_regime": self.homeostatic_regime,
            "effective_file_threshold": self.effective_file_threshold,
            "effective_max_span": self.effective_max_span,
            "auto_slice_window": self.auto_slice_window,
            "effective_post_line_ceiling": self.effective_post_line_ceiling,
            "effective_post_byte_ceiling": self.effective_post_byte_ceiling,
            "predictive_ast_enabled": self.predictive_ast_enabled,
            "distraction_tokens": self.distraction_tokens,
            "sensory_latency_ms": self.compute_latency_ms,
            "sensory_vector": self.to_dict(),
        }


# Alias matching multi-harness Zero-Donut proprioceptive documentation & imports
SensoryProprioceptionVector = SensoryVector


def _normalize_harness_label(raw_label: str) -> Optional[str]:
    norm = raw_label.strip().lower().replace("-", "_").replace(" ", "_")
    if norm in ("antigravity", "agy", "gemini_antigravity"):
        return "antigravity"
    if norm in ("claude_code", "claudecode", "claude", "cc"):
        return "claude_code"
    if norm in ("cursor", "cursor_ide", "cursor_agent", "cursor_composer"):
        return "cursor"
    if norm in ("windsurf", "windsurf_cascade", "codeium", "cascade"):
        return "windsurf"
    if norm in ("goose", "goose_cli", "block_goose"):
        return "goose"
    if norm in ("generic", "default", "unknown", "custom"):
        return "generic"
    return None


def detect_host_harness(
    payload: Optional[Dict[str, Any]] = None,
    tool_name: str = "",
) -> str:
    """
    Detect the active host harness in <0.03ms from payload structure, tool naming
    conventions, or environment overrides.
    Returns one of: 'antigravity', 'claude_code', 'cursor', 'windsurf', 'goose', 'generic'.
    """
    p = payload if isinstance(payload, dict) else {}

    # 1. Explicit payload or environment override
    explicit = str(
        p.get("harness_name")
        or p.get("harness")
        or p.get("bot_name")
        or p.get("bot")
        or os.environ.get("WBC_HARNESS_NAME")
        or ""
    )
    if explicit:
        mapped = _normalize_harness_label(explicit)
        if mapped:
            return mapped

    t_name = str(
        tool_name
        or (p.get("toolCall") or {}).get("name")
        or p.get("tool_name")
        or p.get("tool")
        or ""
    ).strip()
    hook_evt = str(p.get("hook_event_name") or p.get("phase") or "").strip()
    client_ua = str(p.get("client") or p.get("user_agent") or p.get("ide") or "").lower()

    # 2. Goose CLI markers
    if (
        t_name in _GOOSE_TOOLS
        or t_name.startswith(("developer__", "computercontroller__", "screenpipe__"))
        or "goose_session" in p
        or "goose_db" in p
        or "goose" in client_ua
        or os.environ.get("GOOSE_PROVIDER")
    ):
        return "goose"

    # 3. Cursor IDE / Composer / Background Agent markers (checked before Claude Code
    #    because Cursor 1.7+ hooks can also include `hook_event_name: "PreToolUse"`)
    if (
        "cursor_version" in p
        or "composerId" in p
        or "cursor_session" in p
        or "workspace_roots" in p
        or hook_evt in ("beforeReadFile", "beforeShellExecution", "afterFileEdit", "pre-read", "pre_read", "pre-shell", "pre_shell")
        or t_name in ("beforeReadFile", "beforeShellExecution")
        or "cursor" in client_ua
        or (
            "PYTEST_CURRENT_TEST" not in os.environ
            and bool(os.environ.get("CURSOR_TRACE_ID") or os.environ.get("CURSOR_SESSION_ID"))
        )
    ):
        return "cursor"

    # 4. Windsurf Cascade markers
    if (
        "windsurf_session" in p
        or "cascadeId" in p
        or "cascade_id" in p
        or hook_evt in ("pre_read_code", "pre_write_code", "pre_run_command")
        or t_name in ("pre_read_code", "pre_write_code", "pre_run_command")
        or "windsurf" in client_ua
        or "cascade" in client_ua
        or (
            "PYTEST_CURRENT_TEST" not in os.environ
            and bool(os.environ.get("WINDSURF_SESSION_ID") or os.environ.get("CODEIUM_CASCADE_ID"))
        )
    ):
        return "windsurf"

    # 5. Claude Code markers
    if (
        t_name in _CLAUDE_CODE_TOOLS
        or "hook_event_name" in p
        or "transcript_path" in p
        or "tool_input" in p
    ):
        return "claude_code"

    # 6. Antigravity IDE / CLI markers
    if (
        t_name in _ANTIGRAVITY_TOOLS
        or "toolCall" in p
        or "conversationId" in p
        or "stepIdx" in p
        or "invocationNum" in p
        or "initialNumSteps" in p
        or "transcriptPath" in p
    ):
        return "antigravity"

    # 7. Generic fallback
    return "generic"


def resolve_harness_ceiling(harness_name: str) -> Tuple[int, int]:
    """
    Return `(native_line_ceiling, native_byte_ceiling)` for the given harness,
    honoring optional environment overrides.
    """
    line_ceil, byte_ceil, _, _ = _resolve_harness_envelope(harness_name)
    return line_ceil, byte_ceil


def _resolve_harness_envelope(harness_name: str) -> Tuple[int, int, int, int]:
    """
    Return `(native_line_ceiling, native_byte_ceiling, post_output_line_ceiling, post_output_byte_ceiling)`
    for the given harness, honoring optional environment overrides.
    """
    spec = HARNESS_ENVELOPES.get(harness_name, HARNESS_ENVELOPES["generic"])
    line_ceil = spec["native_line_ceiling"]
    byte_ceil = spec["native_byte_ceiling"]
    post_line = spec["post_output_line_ceiling"]
    post_byte = spec["post_output_byte_ceiling"]

    env_line = os.environ.get("WBC_NATIVE_LINE_CEILING")
    env_byte = os.environ.get("WBC_NATIVE_BYTE_CEILING")
    if env_line:
        try:
            line_ceil = int(env_line)
        except ValueError:
            pass
    if env_byte:
        try:
            byte_ceil = int(env_byte)
        except ValueError:
            pass

    if harness_name == "claude_code":
        env_cc = os.environ.get("WBC_CC_LINE_CEILING")
        if env_cc:
            try:
                line_ceil = int(env_cc)
            except ValueError:
                pass
    elif harness_name == "cursor":
        env_cur = os.environ.get("WBC_CURSOR_LINE_CEILING")
        if env_cur:
            try:
                line_ceil = int(env_cur)
            except ValueError:
                pass
    elif harness_name == "goose":
        env_goose = os.environ.get("WBC_GOOSE_LINE_CEILING")
        if env_goose:
            try:
                line_ceil = int(env_goose)
            except ValueError:
                pass
    elif harness_name == "windsurf":
        env_ws = os.environ.get("WBC_WINDSURF_LINE_CEILING")
        if env_ws:
            try:
                line_ceil = int(env_ws)
            except ValueError:
                pass
    elif harness_name == "antigravity":
        env_agy = os.environ.get("WBC_LIVE_FILE_THRESHOLD")
        if env_agy:
            try:
                line_ceil = int(env_agy)
            except ValueError:
                pass

    return line_ceil, byte_ceil, post_line, post_byte


def _classify_read_intent_details(
    args: Optional[Dict[str, Any]],
    payload: Optional[Dict[str, Any]] = None,
    native_line_ceiling: int = 800,
    tool_name: str = "",
) -> Tuple[str, bool, Optional[int], Optional[int], int]:
    """
    Classify a file/tool read into:
      - 'deliberate_bounded': explicit StartLine/EndLine, offset/limit, or view_range within native_line_ceiling
      - 'predictive_query': unbounded or wide read accompanied by a symbol/query/prompt hint
      - 'blind_unbounded': unbounded read without bounds or semantic query hint
    Returns `(read_intent, has_bounds, start_line, end_line, span_lines)`.
    """
    a = args if isinstance(args, dict) else {}
    p = payload if isinstance(payload, dict) else {}

    start_line = (
        a.get("StartLine")
        if a.get("StartLine") is not None
        else (
            a.get("start_line")
            if a.get("start_line") is not None
            else a.get("start_line_one_indexed")
        )
    )
    end_line = (
        a.get("EndLine")
        if a.get("EndLine") is not None
        else (
            a.get("end_line")
            if a.get("end_line") is not None
            else a.get("end_line_one_indexed_inclusive")
        )
    )
    view_range = a.get("view_range")
    limit_param = a.get("limit")
    offset_param = a.get("offset")

    has_bounds = False
    s_int: Optional[int] = None
    e_int: Optional[int] = None
    span_lines = 999999

    if start_line is not None and end_line is not None:
        try:
            s_int = int(start_line)
            e_int = int(end_line)
            if e_int >= s_int and s_int >= 1:
                has_bounds = True
                span_lines = e_int - s_int + 1
        except (ValueError, TypeError):
            has_bounds = False
    elif isinstance(view_range, (list, tuple)) and len(view_range) == 2:
        try:
            s_int = int(view_range[0])
            e_int = int(view_range[1])
            if e_int >= s_int and s_int >= 1:
                has_bounds = True
                span_lines = e_int - s_int + 1
        except (ValueError, TypeError):
            has_bounds = False
    elif limit_param is not None:
        try:
            lim_int = int(limit_param)
            off_int = int(offset_param) if offset_param is not None else 1
            if lim_int > 0 and off_int >= 1:
                has_bounds = True
                s_int = off_int
                e_int = off_int + lim_int - 1
                span_lines = lim_int
        except (ValueError, TypeError):
            has_bounds = False

    if has_bounds and span_lines <= native_line_ceiling:
        return "deliberate_bounded", True, s_int, e_int, span_lines

    if tool_name in ("grep_search", "Grep", "find_by_name", "Glob"):
        return "predictive_query", has_bounds, s_int, e_int, span_lines

    hint_candidates = (
        a.get("query"),
        a.get("Query"),
        a.get("symbol"),
        a.get("Symbol"),
        a.get("hint"),
        a.get("pattern"),
        p.get("query"),
        p.get("symbol"),
        p.get("prompt"),
        p.get("user_prompt"),
    )
    has_query_hint = any(
        isinstance(h, str) and len(h.strip()) >= 2 for h in hint_candidates
    )
    if has_query_hint:
        return "predictive_query", has_bounds, s_int, e_int, span_lines

    return "blind_unbounded", has_bounds, s_int, e_int, span_lines


def classify_read_intent(
    tool_or_args: Any,
    args: Optional[Dict[str, Any]] = None,
    payload: Optional[Dict[str, Any]] = None,
    native_line_ceiling: int = 800,
) -> str:
    """
    Return the read intent classification ('deliberate_bounded' | 'predictive_query' | 'blind_unbounded').
    Supports both `classify_read_intent(tool_name, args)` and `classify_read_intent(args, payload=...)`.
    """
    if isinstance(tool_or_args, str):
        intent, _, _, _, _ = _classify_read_intent_details(
            args=args,
            payload=payload,
            native_line_ceiling=native_line_ceiling,
            tool_name=tool_or_args,
        )
        return intent
    intent, _, _, _, _ = _classify_read_intent_details(
        args=tool_or_args if isinstance(tool_or_args, dict) else args,
        payload=payload,
        native_line_ceiling=native_line_ceiling,
    )
    return intent


def get_fast_thermal_state(payload: Optional[Dict[str, Any]] = None) -> str:
    """
    Read Apple Silicon thermal state in <0.02ms without blocking on subprocesses.
    Checks payload override -> env override -> `/tmp/wbc_thermal_state.json` (with 5s TTL cache).
    """
    global _THERMAL_CACHE_STATE, _THERMAL_CACHE_TS

    if isinstance(payload, dict):
        p_therm = payload.get("thermal_state") or payload.get("thermalState")
        if isinstance(p_therm, str) and p_therm.strip():
            return p_therm.strip().upper()

    env_therm = (
        os.environ.get("WBC_THERMAL_STATE")
        or os.environ.get("WBC_MOCK_THERMAL_STATE")
        or os.environ.get("VOICEFI_MOCK_THERMAL_STATE")
    )
    if env_therm:
        return env_therm.strip().upper()

    now = time.perf_counter()
    if (now - _THERMAL_CACHE_TS) < _THERMAL_CACHE_TTL_SEC:
        return _THERMAL_CACHE_STATE

    _THERMAL_CACHE_TS = now
    state_file = Path("/tmp/wbc_thermal_state.json")
    if state_file.is_file():
        try:
            raw = state_file.read_text(encoding="utf-8", errors="ignore")
            for candidate in ("CRITICAL", "SERIOUS", "FAIR", "NORMAL"):
                if candidate in raw:
                    _THERMAL_CACHE_STATE = candidate
                    return _THERMAL_CACHE_STATE
        except Exception:
            pass

    _THERMAL_CACHE_STATE = "NORMAL"
    return _THERMAL_CACHE_STATE


def estimate_subconscious_csi(
    step_idx: int,
    transcript_kb: float = 0.0,
    explicit_csi: Optional[float] = None,
) -> float:
    """
    Compute cognitive adenosine saturation index (`csi_est` in `[0.0, 1.5]`) in <0.01ms.
    Calibrated to the 3 Circadian Regimes in `docs/AGENT_PROPRIOCEPTION_MASTER_PLAN.md`:
      - Steps 0..10  -> CSI 0.00..0.28 (`GREEN_ELASTIC` < 0.30)
      - Steps 11..25 -> CSI 0.31..0.69 (`YELLOW_BALANCED` in [0.30, 0.70])
      - Steps > 25   -> CSI >= 0.72 (`RED_TIGHT` > 0.70)
    """
    if explicit_csi is not None:
        try:
            return round(max(0.0, float(explicit_csi)), 2)
        except (ValueError, TypeError):
            pass

    env_csi = os.environ.get("WBC_CSI_EST")
    if env_csi:
        try:
            return round(max(0.0, float(env_csi)), 2)
        except ValueError:
            pass

    step_component = max(0, int(step_idx)) / 36.0
    kb_component = max(0.0, float(transcript_kb)) / 900.0
    return round(max(step_component, kb_component), 2)


def resolve_homeostatic_regime(
    step_idx: int | float = 0,
    csi_est: float | str = 0.0,
    thermal_state: str | bool = "NORMAL",
    truncation_cascade_active: bool = False,
) -> str:
    """
    Map InteroceptiveState into one of the 3 Homeostatic Regimes:
      1. `GREEN_ELASTIC`: CSI < 0.30, Steps <= 10, Thermal NORMAL, no truncation cascade.
      2. `YELLOW_BALANCED`: CSI in [0.30, 0.70] or Steps 11-25 or Thermal FAIR.
      3. `RED_TIGHT`: CSI >= 0.70 or Steps > 25 or Thermal SERIOUS/CRITICAL or truncation cascade.
    Supports both `(step_idx, csi_est, thermal_state, truncation_cascade_active)` and
    shorthand `(csi_est, thermal_state, truncation_cascade_active)`.
    """
    if isinstance(csi_est, str):
        # Shorthand call: resolve_homeostatic_regime(csi_est, thermal_state, truncation_cascade_active)
        actual_csi = float(step_idx)
        actual_step = 0
        actual_thermal = csi_est
        actual_trunc = bool(thermal_state)
    else:
        actual_step = int(step_idx)
        actual_csi = float(csi_est)
        actual_thermal = str(thermal_state or "NORMAL")
        actual_trunc = bool(truncation_cascade_active)

    t_norm = actual_thermal.strip().upper()
    if (
        actual_trunc
        or actual_csi >= 0.70
        or actual_step > 25
        or t_norm in ("SERIOUS", "CRITICAL")
    ):
        return "RED_TIGHT"

    if actual_csi >= 0.30 or actual_step >= 11 or t_norm == "FAIR":
        return "YELLOW_BALANCED"

    return "GREEN_ELASTIC"


DEFAULT_FLUSH_KB_THRESHOLD = 1200.0


def _resolve_flush_kb_threshold() -> float:
    """Resolve user-configurable transcript KB threshold for `body flush --auto` reflex cue (default 1200.0 KB)."""
    env_val = os.environ.get("WBC_FLUSH_KB_THRESHOLD")
    if env_val:
        try:
            return max(50.0, float(env_val))
        except ValueError:
            pass
    return DEFAULT_FLUSH_KB_THRESHOLD


def _resolve_flush_step_threshold() -> int:
    """
    Resolve optional user-configurable step threshold for `body flush --auto` reflex cue.
    Defaults to 0 (disabled - Option B pure physiological saturation via transcript KB / CSI).
    """
    env_val = os.environ.get("WBC_FLUSH_STEP_THRESHOLD")
    if env_val:
        try:
            return max(0, int(env_val))
        except ValueError:
            pass
    return 0


def format_autonomic_reflex_cue(intero: InteroceptiveState) -> Optional[str]:
    """
    Subconscious Firewall Invariant 3:
      - Nominal State (CSI < 0.70, Thermal NORMAL/FAIR, no truncation cascade, no retry churn):
        emits `None` (0 prompt characters / 0 meta-tokens).
      - Critical Threshold Breached (CSI >= 0.70, SERIOUS/CRITICAL thermal pressure,
        active truncation cascade, or >=3x retry churn without AST progress):
        emits a single <=15-word surgical reflex cue.
      - Pure Saturation Gating (Option B - `transcript_kb < WBC_FLUSH_KB_THRESHOLD` under normal thermals):
        tightens auto-slices silently to <=80L + Predictive AST without interrupting coding flow with
        premature `body flush --auto` handoff nags; reserves `body flush --auto` strictly for genuine
        transcript saturation (`transcript_kb >= 1200.0` KB or optional `WBC_FLUSH_STEP_THRESHOLD`).
    """
    if intero.thermal_state in ("SERIOUS", "CRITICAL"):
        return (
            f"[⚠️ Thermal {intero.thermal_state} (CSI {intero.csi_est:.2f}): "
            "Offload locally or run 'body flush --auto'.]"
        )
    if intero.truncation_cascade_active:
        return (
            f"[⚠️ Truncation cascade (Step {intero.step_idx}): "
            "Slices clamped <=80L. Use bounded reads or 'body scout'.]"
        )
    if intero.retry_churn_active:
        cnt = max(3, int(intero.retry_churn_count or 3))
        return (
            f"[⚠️ Vagus Retry-Churn ({cnt}x): "
            "Zero AST progress. Inspect AST slice or run 'body scout'.]"
        )
    flush_kb_threshold = _resolve_flush_kb_threshold()
    flush_step_threshold = _resolve_flush_step_threshold()
    if intero.transcript_kb >= flush_kb_threshold or (
        flush_step_threshold > 0 and intero.step_idx >= flush_step_threshold
    ):
        return (
            f"[⚠️ Step {intero.step_idx} (CSI {intero.csi_est:.2f}): "
            "Context saturated. Run 'body flush --auto' to hand off.]"
        )
    if intero.csi_est >= 0.70 or intero.step_idx >= 26:
        return (
            f"[⚠️ Step {intero.step_idx} (CSI {intero.csi_est:.2f}): "
            "Tight <=80L slices active. Use bounded reads or 'body scout'.]"
        )
    return None


def compute_sensory_vector(
    payload: Optional[Dict[str, Any]] = None,
    *,
    tool_name: str = "",
    args: Optional[Dict[str, Any]] = None,
    step_idx: Optional[int] = None,
    truncation_offenders: Optional[Dict[str, Any]] = None,
    retry_churn_active: bool = False,
    retry_churn_count: int = 0,
) -> SensoryVector:
    """
    Compute the unified `<0.2ms` `SensoryVector` (`ProprioceptiveState` + `InteroceptiveState`)
    locally inside `PreToolUse` / `PostToolUse` at zero prompt token cost.
    """
    t0 = time.perf_counter()
    p = payload if isinstance(payload, dict) else {}
    tool_call = p.get("toolCall") if isinstance(p.get("toolCall"), dict) else {}
    resolved_tool = str(
        tool_name
        or tool_call.get("name")
        or p.get("tool_name")
        or p.get("tool")
        or ""
    ).strip()
    resolved_args = (
        args
        if isinstance(args, dict)
        else (
            tool_call.get("args")
            or tool_call.get("input")
            or p.get("tool_input")
            or p.get("args")
            or {}
        )
    )
    if not isinstance(resolved_args, dict):
        resolved_args = {}

    # 1. Proprioception
    harness_name = detect_host_harness(p, tool_name=resolved_tool)
    line_ceil, byte_ceil, post_line_ceil, post_byte_ceil = _resolve_harness_envelope(harness_name)
    read_intent, has_bounds, s_line, e_line, span_lines = _classify_read_intent_details(
        resolved_args, payload=p, native_line_ceiling=line_ceil, tool_name=resolved_tool
    )
    proprio = ProprioceptiveState(
        harness_name=harness_name,
        native_line_ceiling=line_ceil,
        native_byte_ceiling=byte_ceil,
        read_intent=read_intent,
        post_output_line_ceiling=post_line_ceil,
        post_output_byte_ceiling=post_byte_ceil,
        has_bounds=has_bounds,
        requested_start_line=s_line,
        requested_end_line=e_line,
        requested_span_lines=span_lines,
    )

    # 2. Interoception
    raw_step = (
        step_idx
        if step_idx is not None
        else (
            p.get("stepIdx")
            if p.get("stepIdx") is not None
            else (
                p.get("step_idx")
                if p.get("step_idx") is not None
                else p.get("initialNumSteps", os.environ.get("WBC_STEP_IDX", 0))
            )
        )
    )
    try:
        resolved_step_idx = max(0, int(raw_step or 0))
    except (ValueError, TypeError):
        resolved_step_idx = 0

    transcript_kb = 0.0
    if p.get("transcript_kb") is not None:
        try:
            transcript_kb = float(p["transcript_kb"])
        except (ValueError, TypeError):
            transcript_kb = 0.0
    else:
        t_path_str = str(p.get("transcriptPath") or p.get("transcript_path") or "").strip()
        if t_path_str:
            try:
                st = os.stat(t_path_str)
                transcript_kb = round(st.st_size / 1024.0, 1)
            except OSError:
                transcript_kb = 0.0
        else:
            conv_id_str = str(p.get("conversationId") or p.get("conversation_id") or "").strip()
            if len(conv_id_str) == 36 and "-" in conv_id_str:
                brain_root = (
                    os.environ.get("AGY_BRAIN_DIR")
                    or os.environ.get("ANTIGRAVITY_BRAIN_DIR")
                    or os.path.expanduser("~/.gemini/antigravity/brain")
                )
                logs_dir = os.path.join(brain_root, conv_id_str, ".system_generated", "logs")
                for cand in ("transcript_full.jsonl", "transcript.jsonl"):
                    try:
                        st = os.stat(os.path.join(logs_dir, cand))
                        transcript_kb = round(st.st_size / 1024.0, 1)
                        break
                    except OSError:
                        continue

    explicit_csi = p.get("csi_est") if p.get("csi_est") is not None else p.get("csi")
    csi_est = estimate_subconscious_csi(
        step_idx=resolved_step_idx,
        transcript_kb=transcript_kb,
        explicit_csi=explicit_csi,
    )
    thermal_state = get_fast_thermal_state(p)

    trunc_active = bool(p.get("truncation_cascade_active"))
    if not trunc_active and truncation_offenders:
        if resolved_tool in ("read_file", "view_file", "Read", "NotebookRead", "developer__text_editor"):
            trunc_active = bool(
                truncation_offenders.get("read_file")
                or truncation_offenders.get("view_file")
                or truncation_offenders.get("Read")
                or truncation_offenders.get("NotebookRead")
                or truncation_offenders.get(resolved_tool)
            )
        elif resolved_tool in ("run_command", "Bash", "execute_command", "developer__shell"):
            trunc_active = bool(
                truncation_offenders.get("run_command")
                or truncation_offenders.get("Bash")
                or truncation_offenders.get(resolved_tool)
            )
        elif resolved_tool:
            trunc_active = bool(truncation_offenders.get(resolved_tool))
        else:
            trunc_active = len(truncation_offenders) > 0

    resolved_churn_active = bool(retry_churn_active or p.get("retry_churn_active"))
    resolved_churn_count = max(
        int(retry_churn_count or 0),
        int(p.get("retry_churn_count", 0) or 0),
    )

    regime = resolve_homeostatic_regime(
        step_idx=resolved_step_idx,
        csi_est=csi_est,
        thermal_state=thermal_state,
        truncation_cascade_active=trunc_active,
    )
    intero = InteroceptiveState(
        step_idx=resolved_step_idx,
        csi_est=csi_est,
        thermal_state=thermal_state,
        truncation_cascade_active=trunc_active,
        homeostatic_regime=regime,
        transcript_kb=transcript_kb,
        retry_churn_active=resolved_churn_active,
        retry_churn_count=resolved_churn_count,
    )

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)
    return SensoryVector(
        proprioception=proprio,
        interoception=intero,
        compute_latency_ms=elapsed_ms,
    )
