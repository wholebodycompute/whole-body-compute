"""
Organ 07 (Organ 2 of Sensory System): The Eyes - Foveal Vision & Screen Accessibility.

Provides sub-15ms (<420 tokens) zero-screenshot inspection and driving for Electron,
Chromium, and native macOS applications (Claude, Muse, Grok, Arc, Chrome) with 0 MB local model weights.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from whole_body_compute.integrations.foveal_vision import (
    AXElementNode,
    CONTAINER_AX_ROLES,
    FovealUIElement,
    FovealVisionEngine,
    FovealWindowSnapshot,
    INTERACTIVE_AX_ROLES,
    SpatialMotorCache,
    build_foveal_screen_context,
    capture_foveal_snapshot,
    run_foveal_daemon,
    send_foveal_socket_request,
)
from whole_body_compute.doctor import check_organ_the_eyes

logger = logging.getLogger("whole_body_compute.organs.eyes")


class OrganEyes:
    """High-level interface to Organ 07 (The Eyes / Foveal Vision)."""

    @classmethod
    def inspect(
        cls,
        app: Optional[str] = None,
        pid: Optional[int] = None,
        title: Optional[str] = None,
        max_elements: Optional[int] = None,
        filter_query: Optional[str] = None,
        include_floating: bool = False,
        include_status_bar: bool = False,
        read_last: bool = False,
    ) -> FovealWindowSnapshot:
        """Inspect active or background application/tab accessibility DOM in <15ms."""
        return FovealVisionEngine.capture_snapshot(
            pid=pid,
            app_name=app,
            window_title_query=title,
            max_elements=max_elements,
            filter_query=filter_query,
            include_floating=include_floating,
            include_status_bar=include_status_bar,
            read_last=read_last,
        )

    @classmethod
    def drive(
        cls,
        app: Optional[str] = None,
        pid: Optional[int] = None,
        title: Optional[str] = None,
        click: Optional[str] = None,
        click_xy: Optional[str] = None,
        hover: Optional[str] = None,
        right_click: Optional[str] = None,
        attach: Optional[str] = None,
        chain: Optional[Union[str, List[Union[str, Dict[str, Any]]]]] = None,
        type_text: Optional[str] = None,
        paste_text: Optional[str] = None,
        scroll: Optional[str] = None,
        hotkey: Optional[str] = None,
        menu: Optional[str] = None,
        submit: bool = False,
        wait_reply: Optional[float] = None,
        read_last: bool = False,
        max_elements: Optional[int] = None,
        filter_query: Optional[str] = None,
        include_floating: bool = False,
        include_status_bar: bool = False,
    ) -> FovealWindowSnapshot:
        """Inspect and execute tactile actuation on target application."""
        return FovealVisionEngine.capture_snapshot(
            pid=pid,
            app_name=app,
            window_title_query=title,
            max_elements=max_elements,
            filter_query=filter_query,
            include_floating=include_floating,
            include_status_bar=include_status_bar,
            click=click,
            click_xy=click_xy,
            hover=hover,
            right_click=right_click,
            attach=attach,
            chain=chain,
            type_text=type_text,
            paste_text=paste_text,
            scroll=scroll,
            hotkey=hotkey,
            menu=menu,
            submit=submit,
            wait_reply=wait_reply,
            read_last=read_last,
        )

    @classmethod
    def actuate(
        cls,
        click: Optional[str] = None,
        click_xy: Optional[str] = None,
        hover: Optional[str] = None,
        right_click: Optional[str] = None,
        attach: Optional[str] = None,
        chain: Optional[Union[str, List[Union[str, Dict[str, Any]]]]] = None,
        type_text: Optional[str] = None,
        paste_text: Optional[str] = None,
        scroll: Optional[str] = None,
        hotkey: Optional[str] = None,
        menu: Optional[str] = None,
        submit: bool = False,
        snapshot: Optional[FovealWindowSnapshot] = None,
        app_name: Optional[str] = None,
        pid: Optional[int] = None,
        window_title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform direct tactile actuation without re-inspecting."""
        return FovealVisionEngine.actuate(
            click=click,
            click_xy=click_xy,
            hover=hover,
            right_click=right_click,
            attach=attach,
            chain=chain,
            type_text=type_text,
            paste_text=paste_text,
            scroll=scroll,
            hotkey=hotkey,
            menu=menu,
            submit=submit,
            snapshot=snapshot,
            app_name=app_name,
            pid=pid,
            window_title=window_title,
        )

    @classmethod
    def wait_for_reply(
        cls,
        app_name: Optional[str] = None,
        pid: Optional[int] = None,
        window_title: Optional[str] = None,
        timeout_sec: float = 15.0,
        poll_interval_sec: float = 0.25,
        pre_snapshot: Optional[FovealWindowSnapshot] = None,
    ) -> Dict[str, Any]:
        """Wait for streaming reply in target app window to settle."""
        return FovealVisionEngine.wait_for_reply(
            app_name=app_name,
            pid=pid,
            window_title=window_title,
            timeout_sec=timeout_sec,
            poll_interval_sec=poll_interval_sec,
            pre_snapshot=pre_snapshot,
        )

    @classmethod
    def extract_text_blocks(cls, snapshot: Optional[FovealWindowSnapshot]) -> List[str]:
        """Extract meaningful text blocks from target snapshot."""
        return FovealVisionEngine.extract_text_blocks(snapshot)

    @classmethod
    def status(cls) -> Dict[str, Any]:
        """Check anatomical health status of the Eyes organ."""
        return check_organ_the_eyes()

    @classmethod
    def execute_recipe(
        cls,
        recipe_or_path: Any,
        raw_prompt: str = "",
        explicit_args: Optional[Dict[str, Any]] = None,
        require_signed: bool = True,
        trusted_keys: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        Verify SHA-256 + Ed25519 signature and Hoare contracts before executing a
        `PalWorkflow` recipe across `OrganEyes` and `Hands`.
        """
        from pathlib import Path
        import yaml

        from whole_body_compute.hub import verify_recipe_for_execution
        from whole_body_compute.integrations.pal_workflow import (
            PalWorkflowRecipe,
            PalWorkflowRunner,
        )

        if isinstance(recipe_or_path, (str, Path)):
            p = Path(recipe_or_path).expanduser()
            raw_data = yaml.safe_load(p.read_text(encoding="utf-8"))
            verify_recipe_for_execution(
                raw_data,
                require_signed=require_signed,
                trusted_keys=trusted_keys,
                check_compatibility=True,
            )
            recipe_obj = PalWorkflowRecipe.model_validate(raw_data)
        elif isinstance(recipe_or_path, dict):
            verify_recipe_for_execution(
                recipe_or_path,
                require_signed=require_signed,
                trusted_keys=trusted_keys,
                check_compatibility=True,
            )
            recipe_obj = PalWorkflowRecipe.model_validate(recipe_or_path)
        else:
            recipe_obj = recipe_or_path
            verify_recipe_for_execution(
                recipe_obj,
                require_signed=require_signed,
                trusted_keys=trusted_keys,
                check_compatibility=True,
            )

        return PalWorkflowRunner.execute_recipe(
            recipe_obj,
            raw_prompt=raw_prompt,
            explicit_args=explicit_args,
            require_signed=require_signed,
            trusted_keys=trusted_keys,
        )


def inspect_app(
    app: Optional[str] = None,
    pid: Optional[int] = None,
    title: Optional[str] = None,
    max_elements: Optional[int] = None,
    filter_query: Optional[str] = None,
    include_floating: bool = False,
    include_status_bar: bool = False,
    read_last: bool = False,
) -> FovealWindowSnapshot:
    """Functional convenience to inspect an application."""
    return OrganEyes.inspect(
        app=app,
        pid=pid,
        title=title,
        max_elements=max_elements,
        filter_query=filter_query,
        include_floating=include_floating,
        include_status_bar=include_status_bar,
        read_last=read_last,
    )


def drive_app(
    app: Optional[str] = None,
    pid: Optional[int] = None,
    title: Optional[str] = None,
    click: Optional[str] = None,
    click_xy: Optional[str] = None,
    hover: Optional[str] = None,
    right_click: Optional[str] = None,
    attach: Optional[str] = None,
    chain: Optional[Union[str, List[Union[str, Dict[str, Any]]]]] = None,
    type_text: Optional[str] = None,
    paste_text: Optional[str] = None,
    scroll: Optional[str] = None,
    hotkey: Optional[str] = None,
    menu: Optional[str] = None,
    submit: bool = False,
    wait_reply: Optional[float] = None,
    read_last: bool = False,
    max_elements: Optional[int] = None,
    filter_query: Optional[str] = None,
    include_floating: bool = False,
    include_status_bar: bool = False,
) -> FovealWindowSnapshot:
    """Functional convenience to inspect and actuate an application."""
    return OrganEyes.drive(
        app=app,
        pid=pid,
        title=title,
        click=click,
        click_xy=click_xy,
        hover=hover,
        right_click=right_click,
        attach=attach,
        chain=chain,
        type_text=type_text,
        paste_text=paste_text,
        scroll=scroll,
        hotkey=hotkey,
        menu=menu,
        submit=submit,
        wait_reply=wait_reply,
        read_last=read_last,
        max_elements=max_elements,
        filter_query=filter_query,
        include_floating=include_floating,
        include_status_bar=include_status_bar,
    )


__all__ = [
    "OrganEyes",
    "inspect_app",
    "drive_app",
    "check_organ_the_eyes",
    "AXElementNode",
    "FovealUIElement",
    "FovealVisionEngine",
    "FovealWindowSnapshot",
    "SpatialMotorCache",
    "capture_foveal_snapshot",
    "build_foveal_screen_context",
    "run_foveal_daemon",
    "send_foveal_socket_request",
    "INTERACTIVE_AX_ROLES",
    "CONTAINER_AX_ROLES",
]
