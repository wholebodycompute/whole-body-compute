"""
Anatomical Organ Registry for Whole Body Compute.
Exposes the 5 Core Organs (`check_5_organs`) and closed-loop reflex checks.
"""

from whole_body_compute.doctor import (
    check_5_organs,
    check_10_organs,
    check_organ_brainstem,
    check_organ_circadian,
    check_organ_heart,
    check_organ_hippocampus,
    check_organ_immune_system,
    check_organ_mouth_and_ears,
    check_organ_nervous_system,
    check_organ_prefrontal,
    check_organ_skeleton,
    check_organ_the_eyes,
    check_organ_the_hands,
    exercise_5_organ_reflex_loop,
    exercise_10_organ_reflex_loop,
)
from whole_body_compute.organs.immune import (
    AntibodyRecord,
    AntibodyRegistry,
    BloodBrainBarrier,
    EpithelialScreenFilter,
    OrganImmuneSystem,
    VascularTrafficMonitor,
    sanitize_untrusted_screen_text,
    slice_post_tool_output_buffer,
)
from whole_body_compute.local.supervisor import (
    SensoryCephalicReflex,
    default_sensory_reflex,
    trigger_cephalic_prewarm,
)
from whole_body_compute.organs.eyes import (
    OrganEyes,
    inspect_app,
    drive_app,
)

__all__ = [
    "check_5_organs",
    "check_10_organs",
    "check_organ_heart",
    "check_organ_skeleton",
    "check_organ_brainstem",
    "check_organ_prefrontal",
    "check_organ_nervous_system",
    "check_organ_mouth_and_ears",
    "check_organ_the_eyes",
    "check_organ_hippocampus",
    "check_organ_the_hands",
    "check_organ_immune_system",
    "check_organ_circadian",
    "exercise_5_organ_reflex_loop",
    "exercise_10_organ_reflex_loop",
    "AntibodyRecord",
    "AntibodyRegistry",
    "BloodBrainBarrier",
    "EpithelialScreenFilter",
    "OrganImmuneSystem",
    "VascularTrafficMonitor",
    "sanitize_untrusted_screen_text",
    "slice_post_tool_output_buffer",
    "SensoryCephalicReflex",
    "default_sensory_reflex",
    "trigger_cephalic_prewarm",
    "OrganEyes",
    "inspect_app",
    "drive_app",
]

