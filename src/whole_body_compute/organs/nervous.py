"""
Autonomic Nervous System, Vagus Nerve Brake & Pre-Submission Sensory Cephalic Reflex
Backwards-Compatible Re-Export Wrapper.

Canonical implementation lives in `whole_body_compute.local.supervisor`.
"""

from __future__ import annotations

from whole_body_compute.doctor import check_organ_nervous_system
from whole_body_compute.local.supervisor import (
    BrowserVagusBrake,
    SensoryCephalicReflex,
    ThermalSupervisor,
    VagusBrowserAbortError,
    clear_browser_brake,
    default_browser_brake,
    default_sensory_reflex,
    default_supervisor,
    guard_browser_action,
    register_browser_session,
    trigger_browser_vagus_brake,
    trigger_cephalic_prewarm,
    unregister_browser_session,
)

__all__ = [
    "SensoryCephalicReflex",
    "default_sensory_reflex",
    "trigger_cephalic_prewarm",
    "check_organ_nervous_system",
    "VagusBrowserAbortError",
    "BrowserVagusBrake",
    "default_browser_brake",
    "trigger_browser_vagus_brake",
    "guard_browser_action",
    "register_browser_session",
    "unregister_browser_session",
    "clear_browser_brake",
    "ThermalSupervisor",
    "default_supervisor",
]

