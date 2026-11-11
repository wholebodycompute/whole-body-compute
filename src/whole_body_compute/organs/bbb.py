"""
Organ 05 - Blood Brain Barrier (BBB) Backwards-Compatible Re-Export Wrapper.

Canonical implementation lives in `whole_body_compute.organs.immune`.
"""

from __future__ import annotations

from whole_body_compute.organs.immune import (
    DEFAULT_IMMUNE_AUDIT_LOG,
    VASCULAR_SHELL_RULES,
    AntibodyRegistry,
    BloodBrainBarrier,
    OrganImmuneSystem,
    VascularTrafficMonitor,
    _fast_three_stage_ast_secgate,
    get_context_shield_summary,
    get_immune_pre_post_scorecard,
    inspect_shell_command_bloat,
    run_ide_pre_invocation_hook,
    run_ide_pre_tool_hook,
)

__all__ = [
    "BloodBrainBarrier",
    "DEFAULT_IMMUNE_AUDIT_LOG",
    "AntibodyRegistry",
    "OrganImmuneSystem",
    "run_ide_pre_tool_hook",
    "run_ide_pre_invocation_hook",
    "inspect_shell_command_bloat",
    "get_context_shield_summary",
    "get_immune_pre_post_scorecard",
    "VASCULAR_SHELL_RULES",
    "VascularTrafficMonitor",
    "_fast_three_stage_ast_secgate",
]

