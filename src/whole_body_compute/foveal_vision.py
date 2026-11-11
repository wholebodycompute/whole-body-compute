"""
Whole Body Compute: Foveal Vision Root Backwards-Compatible Re-Export Wrapper (Organ 03).
Re-exports all symbols from `whole_body_compute.organs.eyes`.
"""

from whole_body_compute.organs.eyes import (
    AXElementNode,
    CONTAINER_AX_ROLES,
    FovealUIElement,
    FovealVisionEngine,
    FovealWindowSnapshot,
    INTERACTIVE_AX_ROLES,
    OrganEyes,
    SpatialMotorCache,
    build_foveal_screen_context,
    capture_foveal_snapshot,
    check_organ_the_eyes,
    drive_app,
    inspect_app,
    run_foveal_daemon,
    send_foveal_socket_request,
)

__all__ = [
    "OrganEyes",
    "inspect_app",
    "drive_app",
    "check_organ_the_eyes",
    "AXElementNode",
    "CONTAINER_AX_ROLES",
    "FovealUIElement",
    "FovealVisionEngine",
    "FovealWindowSnapshot",
    "INTERACTIVE_AX_ROLES",
    "SpatialMotorCache",
    "build_foveal_screen_context",
    "capture_foveal_snapshot",
    "run_foveal_daemon",
    "send_foveal_socket_request",
]

