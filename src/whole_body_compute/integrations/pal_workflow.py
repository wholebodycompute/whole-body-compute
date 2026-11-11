"""
Declarative Workflow Recipe Schema & Deterministic Replay Engine for Pal macOS Computer-Use.
Provides state-verified execution (window readiness, element targeting, safety unwinding)
without fragile blind sleeps, and dynamic parameter extraction from voice prompts.
"""

import copy
import hashlib
import json
import os
import re
import shlex
import shutil
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from whole_body_compute.integrations.pal_harness import PalCommandResult, PalHarness


class ActionType(str, Enum):
    ACTIVATE_APP = "activate_app"
    WAIT_FOR_WINDOW = "wait_for_window"
    HOTKEY = "hotkey"
    KEYSTROKE = "keystroke"
    PRESS_KEY = "press_key"
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    DRAG = "drag"
    APPLESCRIPT = "applescript"
    RUN_SHELL = "run_shell"
    WAIT = "wait"
    SYSTEM_ACTION = "system_action"
    NOTIFY = "notify"
    OPEN_URL = "open_url"


StepType = ActionType


class ParameterDefinition(BaseModel):
    type: Literal["string", "int", "float", "boolean", "path", "enum"] = "string"
    description: str = ""
    default: Optional[Any] = None
    required: bool = False
    choices: Optional[List[str]] = None
    extract_patterns: List[str] = Field(
        default_factory=list,
        description="Regex patterns with (?P<value>...) to extract from voice prompts without an LLM",
    )


class PostConditionCheck(BaseModel):
    """
    OSWorld-inspired deterministic post-condition state getter.
    Verifies environment state after a step or recipe without cloud LLM calls.
    """

    getter_type: Literal[
        "file_exists",
        "file_nonempty",
        "window_title_contains",
        "clipboard_matches",
        "browser_url_contains",
        "dom_selector_exists",
        "shell_exit_zero",
        "app_running",
        "ax_element_exists",
        "cli_installed",
    ]
    target: str = ""
    expected: Optional[str] = None
    description: str = ""


class ClickTarget(BaseModel):
    target_type: Literal["menu_path", "ui_element", "dom_selector", "coordinates"] = "coordinates"
    menu_path: Optional[List[str]] = None
    role: Optional[str] = None  # e.g. "AXButton", "AXMenuItem", "AXTextField"
    title: Optional[str] = None
    # WorkArena / BrowserGym Tier-1.5 DOM & ARIA grounding
    dom_selector: Optional[str] = None  # e.g. "#submit-btn", "[data-bid='42']", "button[aria-label='Export']"
    aria_label: Optional[str] = None
    bid: Optional[str] = None
    # OSWorld / Set-of-Mark relative window-origin anchor for Tier-3 self-healing under window drift
    window_anchor_dx: Optional[float] = None
    window_anchor_dy: Optional[float] = None
    x: Optional[float] = None
    y: Optional[float] = None
    end_x: Optional[float] = None
    end_y: Optional[float] = None
    relative_to: Literal["window", "screen"] = "screen"


class WorkflowStep(BaseModel):
    name: str = "Step"
    action: ActionType
    description: Optional[str] = None
    timeout_ms: int = 4000
    on_failure: Literal["abort", "ignore", "retry"] = "abort"
    max_retries: int = 1

    # Action-specific arguments
    app_name: Optional[str] = None
    bundle_id: Optional[str] = None
    window_title: Optional[str] = None
    key: Optional[str] = None
    modifiers: Optional[List[str]] = None
    combo: Optional[str] = None  # e.g. "Cmd+Shift+E"
    text: Optional[str] = None
    press_enter: bool = False
    click_target: Optional[ClickTarget] = None
    script: Optional[str] = None
    command: Optional[str] = None
    url: Optional[str] = None
    is_authenticated: bool = False
    cwd: Optional[str] = None
    duration_ms: Optional[int] = None
    message: Optional[str] = None
    visual_anchor: Optional[str] = None
    grounding_prompt: Optional[str] = None
    postconditions: List[PostConditionCheck] = Field(default_factory=list)


class PalWorkflowRecipe(BaseModel):
    schema_version: Optional[str] = None
    namespace: Optional[str] = None
    id: str = Field(description="Unique recipe identifier slug")
    name: str
    version: str = "1.0.0"
    description: str = ""
    license: Optional[str] = None
    author: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    notes: List[str] = Field(
        default_factory=list,
        description="Spoken explanations recorded during demonstration",
    )
    triggers: List[str] = Field(
        default_factory=list,
        description="Spoken trigger phrases (e.g. 'morning setup', 'bounce stems')",
    )
    destructive: bool = False
    app_compatibility: Optional[Dict[str, Any]] = None
    parameters: Dict[str, ParameterDefinition] = Field(default_factory=dict)
    timeout_total_ms: int = 30000
    preconditions: List[PostConditionCheck] = Field(default_factory=list)
    steps: List[WorkflowStep] = Field(default_factory=list)
    postconditions: List[PostConditionCheck] = Field(default_factory=list)
    success_summary: str = "Workflow completed successfully"
    error_summary: str = "Workflow failed at step {step_name}"
    tier_summary: Optional[Dict[str, Any]] = None
    contract_summary: Optional[Dict[str, Any]] = None
    procedural_dna: Optional[Dict[str, Any]] = None
    signature: Optional[Dict[str, Any]] = None

    _raw_signed_manifest: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    _initial_runtime_digest: str = PrivateAttr(default="")

    @model_validator(mode="wrap")
    @classmethod
    def _capture_signed_manifest_and_digest(cls, data: Any, handler: Any) -> "PalWorkflowRecipe":
        raw_manifest: Optional[Dict[str, Any]] = None
        if isinstance(data, dict) and ("signature" in data or "namespace" in data):
            raw_manifest = copy.deepcopy(data)
        instance: PalWorkflowRecipe = handler(data)
        if raw_manifest is not None:
            instance._raw_signed_manifest = raw_manifest
        elif isinstance(data, cls) and data._raw_signed_manifest is not None:
            instance._raw_signed_manifest = copy.deepcopy(data._raw_signed_manifest)
        instance._initial_runtime_digest = instance._compute_runtime_digest()
        return instance

    @property
    def raw_signed_manifest(self) -> Optional[Dict[str, Any]]:
        return self._raw_signed_manifest

    def _compute_runtime_digest(self) -> str:
        core = {
            "namespace": self.namespace,
            "id": self.id,
            "version": self.version,
            "destructive": self.destructive,
            "preconditions": [c.model_dump() for c in self.preconditions],
            "steps": [s.model_dump() for s in self.steps],
            "postconditions": [c.model_dump() for c in self.postconditions],
        }
        return hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def is_runtime_mutated(self) -> bool:
        return bool(
            self._initial_runtime_digest
            and self._compute_runtime_digest() != self._initial_runtime_digest
        )


class PalStateGetters:
    """
    OSWorld-style deterministic environment state getters for zero-token post-condition verification.
    """

    @classmethod
    def evaluate(
        cls,
        check: PostConditionCheck,
        variables: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        vars_map = variables or {}
        target = PalWorkflowRunner._interpolate(check.target or "", vars_map)
        expected = (
            PalWorkflowRunner._interpolate(check.expected, vars_map)
            if check.expected is not None
            else None
        )
        gtype = check.getter_type

        if gtype == "file_exists":
            p = Path(target).expanduser()
            ok = p.exists()
            return ok, (f"File exists: {p}" if ok else f"Expected file missing: {p}")

        if gtype == "file_nonempty":
            p = Path(target).expanduser()
            if not p.is_file():
                return False, f"Expected file missing: {p}"
            size = p.stat().st_size
            ok = size > 0
            return ok, (f"File non-empty ({size} bytes): {p}" if ok else f"File is 0 bytes: {p}")

        if gtype == "window_title_contains":
            needle = (expected or target or "").strip().lower()
            script = """
            tell application "System Events"
                set frontApp to first application process whose frontmost is true
                if (count of windows of frontApp) > 0 then
                    return name of window 1 of frontApp
                end if
                return ""
            end tell
            """
            ok, out = PalHarness.run_applescript(script)
            win_title = (out or "").strip()
            matched = bool(ok and needle in win_title.lower())
            return matched, (
                f"Window title '{win_title}' contains '{needle}'"
                if matched
                else f"Window title '{win_title}' does not contain '{needle}'"
            )

        if gtype == "clipboard_matches":
            pattern = expected or target or ""
            clip_val = ""
            try:
                from AppKit import NSPasteboard, NSPasteboardTypeString

                pb = NSPasteboard.generalPasteboard()
                clip_val = str(pb.stringForType_(NSPasteboardTypeString) or "")
            except Exception:
                pass
            matched = bool(re.search(pattern, clip_val, re.I))
            return matched, (
                f"Clipboard matched /{pattern}/"
                if matched
                else f"Clipboard {clip_val!r} did not match /{pattern}/"
            )

        if gtype == "browser_url_contains":
            needle = (expected or target or "").strip().lower()
            script = (
                'tell application "Google Chrome" to if (count of windows) > 0 '
                "then return URL of active tab of front window"
            )
            ok, out = PalHarness.run_applescript(script)
            url_str = (out or "").strip()
            matched = bool(ok and needle in url_str.lower())
            return matched, (
                f"Browser URL '{url_str}' contains '{needle}'"
                if matched
                else f"Browser URL '{url_str}' does not contain '{needle}'"
            )

        if gtype == "dom_selector_exists":
            selector = (target or expected or "").strip()
            ok, msg = PalWorkflowRunner._check_browser_dom_selector(selector)
            return ok, msg

        if gtype == "shell_exit_zero":
            import subprocess

            try:
                res = subprocess.run(
                    target,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                )
                ok = res.returncode == 0
                if ok and expected:
                    ok = expected.lower() in (res.stdout or "").lower()
                return ok, (res.stdout.strip() or f"Exit code {res.returncode}")
            except Exception as e:
                return False, str(e)

        if gtype == "cli_installed":
            found = shutil.which(target.strip())
            ok = found is not None
            return ok, (f"CLI '{target}' found at {found}" if ok else f"Required CLI '{target}' not found on PATH")

        if gtype == "app_running":
            script = (
                'tell application "System Events" to return (name of processes) '
                f'contains "{target.strip()}"'
            )
            ok, out = PalHarness.run_applescript(script)
            running = bool(ok and "true" in (out or "").strip().lower())
            return running, (
                f"Application '{target}' is running"
                if running
                else f"Application '{target}' is not running"
            )

        if gtype == "ax_element_exists":
            try:
                from whole_body_compute.integrations.foveal_vision import FovealVisionEngine

                snap = FovealVisionEngine.capture_snapshot(
                    app_name=expected or None,
                    filter_query=target.strip(),
                    max_elements=20,
                )
                matched = len(snap.elements) > 0
                return matched, (
                    f"AX element matching '{target}' found ({len(snap.elements)} matches)"
                    if matched
                    else f"No AX element matching '{target}' found"
                )
            except Exception as e:
                return False, f"AX element check failed: {e}"

        return False, f"Unknown getter_type: {gtype}"


class PalWorkflowRunner:
    """Executes PalWorkflowRecipe definitions with deterministic safety checks."""

    # Per-turn Cloud Vision Computer-Use token weights (Retina screenshot tiles + coordinate grounding)
    SYS_PROMPT_TOKENS_PER_TURN = 2400
    ROLLING_HISTORY_TOKENS_PER_PRIOR_TURN = 4800
    RETINA_SCREENSHOT_MB = 1.8

    ACTION_VISION_TOKENS: Dict[ActionType, int] = {
        ActionType.CLICK: 14500,
        ActionType.DOUBLE_CLICK: 14500,
        ActionType.RIGHT_CLICK: 14500,
        ActionType.DRAG: 15500,
        ActionType.ACTIVATE_APP: 11000,
        ActionType.WAIT_FOR_WINDOW: 11000,
        ActionType.HOTKEY: 9500,
        ActionType.KEYSTROKE: 9500,
        ActionType.PRESS_KEY: 9500,
        ActionType.APPLESCRIPT: 12000,
        ActionType.RUN_SHELL: 18000,
        ActionType.SYSTEM_ACTION: 8500,
        ActionType.WAIT: 0,
        ActionType.NOTIFY: 0,
    }

    @classmethod
    def format_token_count(cls, tokens: int) -> str:
        """Format token count into compact human-readable string (e.g. '573K', '1.2M')."""
        if tokens >= 1_000_000:
            return f"{tokens / 1_000_000:.1f}M"
        if tokens >= 1_000:
            return f"{round(tokens / 1_000)}K"
        return str(tokens)

    @classmethod
    def estimate_cloud_savings(cls, recipe: PalWorkflowRecipe) -> Dict[str, Any]:
        """
        Calculate the cloud vision computer-use tokens and screenshot data avoided
        by executing this deterministic local recipe instead of a cloud screenshot loop.
        """
        active_turn = 0
        total_tokens = 0
        screenshot_turns = 0

        for step in recipe.steps:
            step_tokens = cls.ACTION_VISION_TOKENS.get(step.action, 0)
            if step_tokens <= 0:
                continue
            active_turn += 1
            screenshot_turns += 1
            turn_cost = (
                cls.SYS_PROMPT_TOKENS_PER_TURN
                + step_tokens
                + (active_turn - 1) * cls.ROLLING_HISTORY_TOKENS_PER_PRIOR_TURN
            )
            total_tokens += turn_cost

        screen_mb = round(screenshot_turns * cls.RETINA_SCREENSHOT_MB, 1)
        tok_label = cls.format_token_count(total_tokens)
        return {
            "active_steps": active_turn,
            "total_steps": len(recipe.steps),
            "tokens_saved": total_tokens,
            "tokens_saved_formatted": tok_label,
            "screen_mb_saved": screen_mb,
        }

    @classmethod
    def record_cloud_savings(
        cls,
        recipe_id: str,
        tokens_saved: int,
        screen_mb_saved: float,
        elapsed_s: float,
        telemetry_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Persist cumulative cloud token savings to ~/.voicefi/telemetry/pal_cloud_savings.json."""
        import json

        path = telemetry_path or (Path.home() / ".voicefi" / "telemetry" / "pal_cloud_savings.json")
        data: Dict[str, Any] = {
            "total_runs": 0,
            "total_tokens_saved": 0,
            "total_screen_mb_saved": 0.0,
            "workflows": {},
        }
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass

        data["total_runs"] = int(data.get("total_runs", 0)) + 1
        data["total_tokens_saved"] = int(data.get("total_tokens_saved", 0)) + int(tokens_saved)
        data["total_screen_mb_saved"] = round(
            float(data.get("total_screen_mb_saved", 0.0)) + float(screen_mb_saved), 2
        )

        wf_map = data.setdefault("workflows", {})
        wf_entry = wf_map.setdefault(
            recipe_id,
            {"runs": 0, "tokens_saved": 0, "screen_mb_saved": 0.0, "last_elapsed_s": 0.0},
        )
        wf_entry["runs"] = int(wf_entry.get("runs", 0)) + 1
        wf_entry["tokens_saved"] = int(wf_entry.get("tokens_saved", 0)) + int(tokens_saved)
        wf_entry["screen_mb_saved"] = round(
            float(wf_entry.get("screen_mb_saved", 0.0)) + float(screen_mb_saved), 2
        )
        wf_entry["last_elapsed_s"] = round(float(elapsed_s), 2)

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass
        return data

    @classmethod
    def execute_recipe(
        cls,
        recipe: PalWorkflowRecipe,
        raw_prompt: str = "",
        explicit_args: Optional[Dict[str, Any]] = None,
        require_signed: bool = False,
        trusted_keys: Optional[Dict[str, str]] = None,
    ) -> PalCommandResult:
        recipe_id = recipe.id
        steps = recipe.steps
        total_timeout_s = recipe.timeout_total_ms / 1000.0

        # 0. Pre-execution Cryptographic Signature & Tamper Gate (before OrganEyes / Hands run)
        has_hub_provenance = bool(
            recipe.signature or recipe.namespace or recipe.raw_signed_manifest or require_signed
        )
        if has_hub_provenance or os.environ.get("WBC_REQUIRE_SIGNED_RECIPES") == "1":
            if recipe.is_runtime_mutated():
                return PalCommandResult(
                    success=False,
                    command=f"{recipe_id}:signature",
                    action_type="signature_verification_failed",
                    details="Signed recipe steps or Hoare contracts were mutated in memory after loading",
                    spoken_summary="Blocked execution: signed recipe failed tamper check",
                )
            try:
                from whole_body_compute.hub import verify_recipe_for_execution

                verify_recipe_for_execution(
                    recipe,
                    require_signed=require_signed,
                    trusted_keys=trusted_keys,
                    check_compatibility=True,
                )
            except Exception as exc:
                return PalCommandResult(
                    success=False,
                    command=f"{recipe_id}:signature",
                    action_type="signature_verification_failed",
                    details=str(exc),
                    spoken_summary="Blocked execution: recipe failed cryptographic signature or safety check",
                )

        # 1. Resolve & bind parameters
        bound_vars = cls._bind_parameters(recipe.parameters, raw_prompt, explicit_args or {})
        start_time = time.monotonic()

        # 1b. Verify Hoare preconditions {P} before dispatching any OrganEyes / Hands step
        if recipe.preconditions:
            for pre in recipe.preconditions:
                pre_ok, pre_msg = PalStateGetters.evaluate(pre, bound_vars)
                if not pre_ok:
                    return PalCommandResult(
                        success=False,
                        command=f"{recipe_id}:precondition:{pre.getter_type}",
                        action_type="precondition_failed",
                        details=f"Recipe precondition ({pre.getter_type}) failed: {pre_msg}",
                        spoken_summary=recipe.error_summary.format(step_name="precondition"),
                    )

        for idx, step in enumerate(steps):
            if (time.monotonic() - start_time) > total_timeout_s:
                cls._unwind_failure()
                return PalCommandResult(
                    success=False,
                    command=recipe_id,
                    action_type="workflow_timeout",
                    details=f"Workflow exceeded total budget of {total_timeout_s}s",
                    spoken_summary=f"Workflow timed out during {step.name or f'step {idx + 1}'}",
                )

            max_attempts = max(1, int(step.max_retries or 1)) if step.on_failure == "retry" else 1
            success = False
            err = ""
            for attempt in range(max_attempts):
                success, err = cls._execute_step(step, bound_vars)
                if success and step.postconditions:
                    for pc in step.postconditions:
                        pc_ok, pc_msg = PalStateGetters.evaluate(pc, bound_vars)
                        if not pc_ok:
                            success = False
                            err = f"Postcondition ({pc.getter_type}) failed: {pc_msg}"
                            break
                if success:
                    break
                if attempt + 1 < max_attempts:
                    time.sleep(0.05)

            if not success:
                healed_ok, healed_err = cls._attempt_local_self_heal(step, bound_vars, err)
                if healed_ok:
                    success = True
                    err = ""
                elif healed_err:
                    err = healed_err

            if not success:
                if step.on_failure == "abort":
                    cls._unwind_failure()
                    err_summary = recipe.error_summary.format(
                        step_name=step.name or f"step {idx + 1}"
                    )
                    return PalCommandResult(
                        success=False,
                        command=f"{recipe_id}:{step.name}",
                        action_type=step.action.value,
                        details=err,
                        spoken_summary=err_summary,
                    )

        # Verify recipe-level OSWorld postconditions if present
        if recipe.postconditions:
            for pc in recipe.postconditions:
                pc_ok, pc_msg = PalStateGetters.evaluate(pc, bound_vars)
                if not pc_ok:
                    cls._unwind_failure()
                    return PalCommandResult(
                        success=False,
                        command=f"{recipe_id}:postcondition:{pc.getter_type}",
                        action_type="postcondition_failed",
                        details=f"Recipe postcondition ({pc.getter_type}) failed: {pc_msg}",
                        spoken_summary=recipe.error_summary.format(step_name="verification"),
                    )

        elapsed_s = round(time.monotonic() - start_time, 2)
        savings = cls.estimate_cloud_savings(recipe)
        tokens_saved = savings["tokens_saved"]
        screen_mb_saved = savings["screen_mb_saved"]
        tok_label = savings["tokens_saved_formatted"]
        badge = f"⚡ {elapsed_s:.1f}s · 0 tok (~{tok_label} saved)"

        if os.getenv("VOICEFI_HEADLESS") != "1":
            cls.record_cloud_savings(
                recipe_id=recipe_id,
                tokens_saved=tokens_saved,
                screen_mb_saved=screen_mb_saved,
                elapsed_s=elapsed_s,
            )

        success_msg = recipe.success_summary
        for k, v in bound_vars.items():
            success_msg = success_msg.replace(f"${{{k}}}", str(v))

        return PalCommandResult(
            success=True,
            command=recipe_id,
            action_type="workflow",
            details=f"Executed {len(steps)} steps successfully ({badge})",
            spoken_summary=success_msg,
            cloud_tokens_saved=tokens_saved,
            screen_mb_saved=screen_mb_saved,
            elapsed_s=elapsed_s,
            telemetry_badge=badge,
        )

    @classmethod
    def _execute_step(cls, step: WorkflowStep, variables: Dict[str, Any]) -> Tuple[bool, str]:
        action = step.action
        timeout_s = step.timeout_ms / 1000.0

        if action == ActionType.ACTIVATE_APP:
            app = cls._interpolate(step.app_name or "", variables)
            res = PalHarness.launch_app(app)
            return res.success, res.details

        elif action == ActionType.WAIT_FOR_WINDOW:
            app = cls._interpolate(step.app_name or "", variables)
            win_title = cls._interpolate(step.window_title or "", variables)
            return cls._wait_for_window_ready(app, win_title, timeout_s)

        elif action == ActionType.HOTKEY:
            if step.combo:
                # Parse combo e.g. "Cmd+Shift+N" or "Cmd+S"
                parts = [p.strip() for p in step.combo.split("+")]
                mods = parts[:-1]
                key = parts[-1].lower() if parts else ""
                return cls._send_hotkey(key, mods)
            key = step.key or ""
            mods = step.modifiers or []
            return cls._send_hotkey(key, mods)

        elif action in (ActionType.KEYSTROKE, "keystroke"):
            # Check if focused element is a secure text field using AmbientPrivacyGuard
            from whole_body_compute.integrations.ambient_discovery import AmbientPrivacyGuard

            is_auth = bool(
                getattr(step, "is_authenticated", False)
                or variables.get("__authenticated__", False)
                or variables.get("is_authenticated", False)
            )

            try:
                from whole_body_compute.integrations.foveal_vision import capture_foveal_snapshot

                snap = capture_foveal_snapshot(max_elements=5)
                foc = snap.focused_element.to_dict() if snap.focused_element else None
                if AmbientPrivacyGuard.is_element_secure(foc) and not is_auth:
                    return False, "Security violation: Aborted keystroke into secure text field (unauthenticated step)"
            except Exception:
                pass

            text = cls._interpolate(step.text or "", variables)
            return cls._send_keystroke(text, step.press_enter)

        elif action == ActionType.PRESS_KEY:
            key = step.key or ""
            return cls._send_special_key(key)

        elif action in (ActionType.CLICK, ActionType.DOUBLE_CLICK, ActionType.RIGHT_CLICK):
            is_double = action == ActionType.DOUBLE_CLICK
            is_right = action == ActionType.RIGHT_CLICK
            return cls._handle_click(
                step.click_target, variables, double_click=is_double, right_click=is_right
            )

        elif action == ActionType.DRAG:
            return cls._handle_drag(step.click_target, variables)

        elif action == ActionType.APPLESCRIPT:
            raw_script = step.script or ""
            interpolated = cls._interpolate(raw_script, variables, sanitize_as=True)
            return PalHarness.run_applescript(interpolated)

        elif action in (ActionType.RUN_SHELL, "run_shell"):
            raw_cmd = step.command or ""
            cmd = cls._interpolate(raw_cmd, variables, sanitize_shell=True)
            # Clean up trailing flags if an optional parameter bound to empty string ''
            cmd = re.sub(r"\s+--[a-zA-Z0-9_-]+\s+(?:''|\"\")(?=\s|$)", "", cmd).strip()

            from whole_body_compute.organs.immune import VascularTrafficMonitor

            v_check = VascularTrafficMonitor.inspect_shell_command(cmd)
            if not v_check.get("allowed", True):
                return False, f"Security violation: {v_check.get('reason', 'Blocked by VascularTrafficMonitor')}"

            try:
                import subprocess

                res = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    cwd=step.cwd,
                )
                return res.returncode == 0, res.stderr or res.stdout
            except Exception as e:
                return False, str(e)

        elif action in (ActionType.OPEN_URL, "open_url"):
            raw_url = getattr(step, "url", None) or step.command or ""
            url_str = cls._interpolate(raw_url, variables)
            if not url_str:
                return False, "Empty URL provided to open_url action"

            import urllib.parse
            from whole_body_compute.organs.immune import VascularTrafficMonitor

            parsed = urllib.parse.urlparse(url_str)
            host = (parsed.hostname or "").strip().lower()
            if not VascularTrafficMonitor.is_trusted_host(host):
                return False, f"Security violation: VascularTrafficMonitor blocked untrusted host '{host}' in open_url"

            try:
                import subprocess

                subprocess.run(["open", url_str], check=False)
                return True, f"Opened URL {url_str}"
            except Exception as e:
                return False, str(e)

        elif action == ActionType.WAIT:
            ms = step.duration_ms or 200
            time.sleep(min(ms, 5000) / 1000.0)
            return True, ""

        elif action == ActionType.NOTIFY:
            msg = cls._interpolate(step.message or "Notification", variables)
            try:
                import rumps

                rumps.notification("VoiceFi Pal", "", msg)
            except Exception:
                pass
            return True, ""

        return False, f"Unknown action: {action}"

    @classmethod
    def _wait_for_window_ready(
        cls, app_name: str, window_pattern: str, timeout_s: float
    ) -> Tuple[bool, str]:
        """Polls macOS System Events / AXUIElement until window is frontmost and present."""
        deadline = time.monotonic() + timeout_s
        clean_app = PalHarness.APP_MAP.get(app_name.lower(), app_name)
        script = f"""
        tell application "System Events"
            if not (exists process "{clean_app}") then return "not_running"
            set p to process "{clean_app}"
            if frontmost of p is true and (count of windows of p) > 0 then
                return "ready"
            end if
            return "not_ready"
        end tell
        """
        while time.monotonic() < deadline:
            ok, out = PalHarness.run_applescript(script)
            if ok and out == "ready":
                return True, "Window ready"
            time.sleep(0.04)  # 40ms low-overhead polling cadence
        return False, f"Timeout waiting for {clean_app} window"

    _SPECIAL_KEY_CODES = {
        "return": 36,
        "enter": 36,
        "tab": 48,
        "space": 49,
        "backspace": 51,
        "delete": 117,
        "forwarddelete": 117,
        "escape": 53,
        "esc": 53,
        "home": 115,
        "end": 119,
        "left": 123,
        "right": 124,
        "down": 125,
        "up": 126,
    }

    @classmethod
    def _send_hotkey(cls, key: str, modifiers: List[str]) -> Tuple[bool, str]:
        mod_items = []
        for m in modifiers:
            m_clean = (
                m.lower()
                .replace("cmd", "command")
                .replace("opt", "option")
                .replace("ctrl", "control")
            )
            mod_items.append(f"{m_clean} down")
        mod_clause = f" using {{{', '.join(mod_items)}}}" if mod_items else ""
        code = cls._SPECIAL_KEY_CODES.get((key or "").strip().lower())
        if code is not None:
            script = f'tell application "System Events" to key code {code}{mod_clause}'
        else:
            script = f'tell application "System Events" to keystroke "{key.lower()}"{mod_clause}'
        return PalHarness.run_applescript(script)

    @classmethod
    def _send_special_key(cls, key_name: str) -> Tuple[bool, str]:
        code = cls._SPECIAL_KEY_CODES.get((key_name or "").strip().lower())
        if code is not None:
            script = f'tell application "System Events" to key code {code}'
            return PalHarness.run_applescript(script)
        script = f'tell application "System Events" to keystroke "{key_name}"'
        return PalHarness.run_applescript(script)

    @classmethod
    def _send_keystroke(cls, text: str, press_enter: bool = False) -> Tuple[bool, str]:
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        script = f'tell application "System Events" to keystroke "{escaped}"'
        ok, out = PalHarness.run_applescript(script)
        if ok and press_enter:
            time.sleep(0.05)
            return PalHarness.run_applescript('tell application "System Events" to key code 36')
        return ok, out

    @classmethod
    def _build_dom_selector(cls, target: ClickTarget, variables: Dict[str, Any]) -> str:
        """Synthesize a CSS/ARIA/WorkArena bid selector from ClickTarget fields."""
        if target.dom_selector:
            return cls._interpolate(target.dom_selector, variables).strip()
        if target.bid:
            safe_bid = cls._interpolate(target.bid, variables).replace('"', '\\"').strip()
            return f'[data-bid="{safe_bid}"], [bid="{safe_bid}"]'
        if target.aria_label:
            safe_aria = cls._interpolate(target.aria_label, variables).replace('"', '\\"').strip()
            return f'[aria-label="{safe_aria}"]'
        return ""

    @classmethod
    def _check_browser_dom_selector(cls, selector: str) -> Tuple[bool, str]:
        """Check if a CSS/ARIA/bid selector exists in the active Chrome tab DOM."""
        if not selector:
            return False, "Empty DOM selector"
        js_safe = selector.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
        js_expr = f"(function(){{ return Boolean(document.querySelector('{js_safe}')); }})()"
        as_js = js_safe = js_expr.replace("\\", "\\\\").replace('"', '\\"')
        script = (
            'tell application "Google Chrome" to if (count of windows) > 0 '
            f'then execute active tab of front window javascript "{as_js}"'
        )
        ok, out = PalHarness.run_applescript(script)
        found = bool(ok and str(out).strip().lower() == "true")
        return found, (
            f"DOM selector '{selector}' found"
            if found
            else f"DOM selector '{selector}' not found in active tab"
        )

    @classmethod
    def _click_browser_dom_selector(cls, selector: str) -> Tuple[bool, str]:
        """
        Execute a WorkArena Tier-1.5 DOM click inside the active browser tab.
        Immune to window movement, scroll offset, and banner layout shifts.
        """
        if not selector:
            return False, "Empty DOM selector"
        js_safe = selector.replace("\\", "\\\\").replace("'", "\\'")
        js_expr = (
            "(function(){"
            f"const el = document.querySelector('{js_safe}');"
            "if (!el) return 'not_found';"
            "if (typeof el.scrollIntoView === 'function') el.scrollIntoView({block:'center', inline:'center'});"
            "el.click();"
            "return 'clicked_dom';"
            "})()"
        )
        as_js = js_expr.replace("\\", "\\\\").replace('"', '\\"')
        script = (
            'tell application "Google Chrome" to if (count of windows) > 0 '
            f'then execute active tab of front window javascript "{as_js}"'
        )
        ok, out = PalHarness.run_applescript(script)
        if ok and "clicked_dom" in str(out):
            return True, f"Clicked DOM selector '{selector}' (Tier 1.5)"
        return False, f"DOM selector '{selector}' not resolved ({out})"

    @classmethod
    def _get_frontmost_window_origin(cls) -> Optional[Tuple[float, float]]:
        """Query the frontmost application window's (x, y) origin from Quartz WindowServer (<0.2ms)."""
        try:
            import Quartz

            wins = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly
                | Quartz.kCGWindowListExcludeDesktopElements,
                Quartz.kCGNullWindowID,
            )
            if wins:
                for w in wins:
                    if w.get("kCGWindowLayer", 0) == 0:
                        owner = str(w.get("kCGWindowOwnerName") or "").strip()
                        if owner and owner not in (
                            "Window Server",
                            "SystemUIServer",
                            "Control Center",
                        ):
                            bounds = w.get("kCGWindowBounds") or {}
                            if "X" in bounds and "Y" in bounds:
                                return float(bounds["X"]), float(bounds["Y"])
                            break
        except Exception:
            pass
        return None

    @classmethod
    def _handle_click(
        cls,
        target: Optional[ClickTarget],
        variables: Dict[str, Any],
        double_click: bool = False,
        right_click: bool = False,
    ) -> Tuple[bool, str]:
        if not target:
            return False, "Missing click target"

        # 1. Menu path click (e.g. ["File", "Export", "All Tracks..."])
        if target.target_type == "menu_path" and target.menu_path:
            path = target.menu_path
            if len(path) >= 2:
                menu_bar_item = path[0]
                menu_item = path[1]
                sub_clause = ""
                if len(path) >= 3:
                    sub_item = path[2]
                    sub_clause = f' of menu item "{sub_item}" of menu 1'
                script = f"""
                tell application "System Events"
                    set frontApp to first application process whose frontmost is true
                    tell frontApp
                        click menu item "{menu_item}"{sub_clause} of menu 1 of menu bar item "{menu_bar_item}" of menu bar 1
                    end tell
                end tell
                """
                return PalHarness.run_applescript(script)

        # 1.25. Tier-1.5 WorkArena / BrowserGym DOM Selector click (immune to window & layout drift)
        selector = cls._build_dom_selector(target, variables)
        if selector and (target.target_type == "dom_selector" or target.dom_selector or target.bid):
            ok_dom, out_dom = cls._click_browser_dom_selector(selector)
            if ok_dom:
                return True, out_dom
            # If target_type is strictly dom_selector and no fallback coordinates/title exist, return error
            if target.x is None and not target.title:
                return False, out_dom

        # 1.5. Tier-2 Semantic UI element click (AXRole + AXTitle via System Events, immune to window movement)
        if target.target_type in ("ui_element", "dom_selector") and target.title:
            safe_title = target.title.replace("\\", "\\\\").replace('"', '\\"')
            role_hint = (target.role or "AXButton").strip()
            if role_hint == "AXMenuItem":
                element_expr = f'menu item "{safe_title}"'
            else:
                element_expr = f'UI element "{safe_title}"'
            script = f"""
            tell application "System Events"
                set frontApp to first application process whose frontmost is true
                tell frontApp
                    if (count of windows) > 0 then
                        click {element_expr} of window 1
                        return "clicked_semantic"
                    end if
                end tell
            end tell
            """
            ok, out = PalHarness.run_applescript(script)
            if ok and "clicked_semantic" in out:
                return True, f"Clicked semantic element '{target.title}' ({role_hint})"

        # 2. Tier-3 Coordinate click via Quartz (with OSWorld Window-Anchor Self-Healing)
        click_x = target.x
        click_y = target.y
        if (
            target.window_anchor_dx is not None
            and target.window_anchor_dy is not None
            and target.relative_to == "window"
        ):
            origin = cls._get_frontmost_window_origin()
            if origin is not None:
                click_x = origin[0] + float(target.window_anchor_dx)
                click_y = origin[1] + float(target.window_anchor_dy)

        if click_x is not None and click_y is not None:
            try:
                import Quartz
                from Quartz import CGPoint

                pt = CGPoint(float(click_x), float(click_y))
                if right_click:
                    down_event = Quartz.CGEventCreateMouseEvent(
                        None, Quartz.kCGEventRightMouseDown, pt, Quartz.kCGMouseButtonRight
                    )
                    up_event = Quartz.CGEventCreateMouseEvent(
                        None, Quartz.kCGEventRightMouseUp, pt, Quartz.kCGMouseButtonRight
                    )
                    Quartz.CGEventPost(Quartz.kCGHIDEventTap, down_event)
                    time.sleep(0.02)
                    Quartz.CGEventPost(Quartz.kCGHIDEventTap, up_event)
                    return True, f"Right-clicked ({click_x}, {click_y})"

                down_event = Quartz.CGEventCreateMouseEvent(
                    None, Quartz.kCGEventLeftMouseDown, pt, Quartz.kCGMouseButtonLeft
                )
                up_event = Quartz.CGEventCreateMouseEvent(
                    None, Quartz.kCGEventLeftMouseUp, pt, Quartz.kCGMouseButtonLeft
                )
                if double_click:
                    Quartz.CGEventSetIntegerValueField(
                        down_event, Quartz.kCGMouseEventClickState, 2
                    )
                    Quartz.CGEventSetIntegerValueField(up_event, Quartz.kCGMouseEventClickState, 2)

                Quartz.CGEventPost(Quartz.kCGHIDEventTap, down_event)
                time.sleep(0.02)
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, up_event)
                if double_click:
                    time.sleep(0.05)
                    Quartz.CGEventPost(Quartz.kCGHIDEventTap, down_event)
                    time.sleep(0.02)
                    Quartz.CGEventPost(Quartz.kCGHIDEventTap, up_event)
                return True, f"Clicked ({click_x}, {click_y})"
            except Exception as e:
                return False, f"Failed coordinate click: {e}"

        return False, f"Unsupported click target: {target}"

    @classmethod
    def _handle_drag(
        cls,
        target: Optional[ClickTarget],
        variables: Dict[str, Any],
    ) -> Tuple[bool, str]:
        if (
            not target
            or target.x is None
            or target.y is None
            or target.end_x is None
            or target.end_y is None
        ):
            return False, "Missing drag coordinates"
        try:
            import Quartz
            from Quartz import CGPoint

            sx, sy = float(target.x), float(target.y)
            ex, ey = float(target.end_x), float(target.end_y)
            start_pt = CGPoint(sx, sy)
            end_pt = CGPoint(ex, ey)

            down_event = Quartz.CGEventCreateMouseEvent(
                None, Quartz.kCGEventLeftMouseDown, start_pt, Quartz.kCGMouseButtonLeft
            )
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, down_event)
            time.sleep(0.04)

            steps = 10
            for i in range(1, steps + 1):
                ix = sx + (ex - sx) * (i / steps)
                iy = sy + (ey - sy) * (i / steps)
                drag_event = Quartz.CGEventCreateMouseEvent(
                    None,
                    Quartz.kCGEventLeftMouseDragged,
                    CGPoint(ix, iy),
                    Quartz.kCGMouseButtonLeft,
                )
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, drag_event)
                time.sleep(0.015)

            up_event = Quartz.CGEventCreateMouseEvent(
                None, Quartz.kCGEventLeftMouseUp, end_pt, Quartz.kCGMouseButtonLeft
            )
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, up_event)
            return True, f"Dragged ({sx}, {sy}) -> ({ex}, {ey})"
        except Exception as e:
            return False, f"Failed coordinate drag: {e}"

    @classmethod
    def _unwind_failure(cls) -> None:
        """Sends Escape key to safely dismiss any open modal sheets or menus."""
        script = 'tell application "System Events" to key code 53'
        PalHarness.run_applescript(script)

    @classmethod
    def _interpolate(
        cls,
        template: str,
        variables: Dict[str, Any],
        sanitize_shell: bool = False,
        sanitize_as: bool = False,
    ) -> str:
        result = template
        for k, v in variables.items():
            val = str(v)
            if sanitize_shell:
                val = shlex.quote(val)
            elif sanitize_as:
                val = val.replace("\\", "\\\\").replace('"', '\\"')
            result = result.replace(f"${{{k}}}", val)
        return result

    @classmethod
    def _bind_parameters(
        cls,
        param_defs: Dict[str, ParameterDefinition],
        prompt: str,
        explicit_args: Dict[str, Any],
    ) -> Dict[str, Any]:
        bound: Dict[str, Any] = {}
        unmatched_defs: Dict[str, str] = {}
        for name, pdef in param_defs.items():
            if name in explicit_args:
                bound[name] = explicit_args[name]
                continue
            matched = False
            for pat in pdef.extract_patterns:
                m = re.search(pat, prompt, re.IGNORECASE)
                if m and "value" in m.groupdict():
                    bound[name] = m.group("value")
                    matched = True
                    break
            if not matched:
                bound[name] = pdef.default if pdef.default is not None else ""
                if prompt and prompt.strip():
                    unmatched_defs[name] = pdef.description or name

        if unmatched_defs and prompt and prompt.strip() and not PalHarness._is_test_env():
            extracted = cls._extract_params_via_local_reflex(prompt, unmatched_defs)
            for k, v in extracted.items():
                if k in unmatched_defs and v not in (None, ""):
                    bound[k] = str(v)
        return bound

    @classmethod
    def _extract_params_via_local_reflex(
        cls, prompt: str, unmatched_defs: Dict[str, str]
    ) -> Dict[str, Any]:
        """Use sub-200ms local `tev1:0.8b` (:11434) to extract missing voice parameters."""
        import json
        import urllib.request

        schema_str = ", ".join(f'"{k}": "<{desc}>"' for k, desc in unmatched_defs.items())
        dot_ctx = ""
        try:
            from whole_body_compute.local.dot_memory import get_dot_store

            dot_ctx = get_dot_store().build_context_injection(
                prompt,
                top_k=2,
                categories=["workflow", "preference", "habit_pattern"],
            )
        except Exception:
            pass
        dot_prefix = f"{dot_ctx}\n" if dot_ctx else ""
        llm_prompt = (
            f"{dot_prefix}Extract parameters from the user voice command into JSON {{{schema_str}}}. "
            "Use empty string if not mentioned. Output ONLY JSON.\n"
            f"Command: {prompt!r}\nJSON:"
        )
        # 1. Resolve model via HybridRouter
        resolved_model = "tev1:0.8b"
        try:
            from whole_body_compute.local.router import default_router

            dec = default_router.resolve_model(task_type="tier0_reflex")
            if dec and dec.selected_model:
                resolved_model = dec.selected_model.replace("cloud:", "").strip()
        except Exception:
            pass

        # 2. Try :9380 SpecialistRouter & :9379 MLX endpoints first
        for endpoint, model_name in (
            ("http://127.0.0.1:9380/v1/chat/completions", "gemma4-26b"),
            ("http://127.0.0.1:9379/v1/chat/completions", "voicefi-coder-mlx"),
        ):
            try:
                body = json.dumps(
                    {
                        "model": model_name,
                        "messages": [{"role": "user", "content": llm_prompt}],
                        "temperature": 0.0,
                        "max_tokens": 80,
                    }
                ).encode("utf-8")
                req = urllib.request.Request(
                    endpoint,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=1.5) as resp:
                    if resp.status == 200:
                        raw = (
                            json.loads(resp.read().decode("utf-8"))
                            .get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )
                        m = re.search(r"\{.*?\}", raw, re.DOTALL)
                        if m:
                            return json.loads(m.group(0))
            except Exception:
                continue

        # 3. Fallback to Ollama (:11434) using resolved model
        for model_name in (resolved_model, "tev1:0.8b", "qwen2.5-coder:1.5b"):
            try:
                body = json.dumps(
                    {
                        "model": model_name,
                        "messages": [{"role": "user", "content": llm_prompt}],
                        "stream": False,
                        "think": False,
                        "options": {"temperature": 0.0, "num_predict": 80},
                    }
                ).encode("utf-8")
                req = urllib.request.Request(
                    "http://127.0.0.1:11434/api/chat",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=2.5) as resp:
                    if resp.status == 200:
                        raw = (
                            json.loads(resp.read().decode("utf-8"))
                            .get("message", {})
                            .get("content", "")
                        )
                        m = re.search(r"\{.*?\}", raw, re.DOTALL)
                        if m:
                            return json.loads(m.group(0))
            except Exception:
                continue
        return {}

    @classmethod
    def _attempt_local_self_heal(
        cls,
        step: WorkflowStep,
        variables: Dict[str, Any],
        error_msg: str,
    ) -> Tuple[bool, str]:
        """
        Attempt 100% on-device self-healing when a workflow step fails due to schema or UI drift:
        1. For RUN_SHELL steps invoking a local Python script (.py), invokes ReconImplementer
           (voicefi-coder-mlx :9379 -> Qwen2.5-Coder-14B :8080) guarded by the 3-Stage AST SecGate.
        2. For AXUIElement CLICK steps that drifted but retain fallback coordinates, falls back cleanly.
        """
        if os.getenv("VOICEFI_PAL_DISABLE_SELF_HEAL") == "1":
            return False, error_msg

        # Case 1: AXUIElement click drift -> coordinate fallback if present
        if step.action in (ActionType.CLICK, ActionType.DOUBLE_CLICK, ActionType.RIGHT_CLICK):
            ct = step.click_target
            if ct and ct.target_type == "ax_element" and ct.x is not None and ct.y is not None:
                fallback_ct = ClickTarget(
                    target_type="coordinates",
                    x=ct.x,
                    y=ct.y,
                    end_x=ct.end_x,
                    end_y=ct.end_y,
                )
                ok, fb_err = cls._handle_click(
                    fallback_ct,
                    variables,
                    double_click=(step.action == ActionType.DOUBLE_CLICK),
                    right_click=(step.action == ActionType.RIGHT_CLICK),
                )
                if ok:
                    return True, ""

        # Case 2: Tier-1 Python script self-healing via local MLX coder + 3-Stage AST SecGate
        if step.action == ActionType.RUN_SHELL and step.command:
            cmd_interp = cls._interpolate(step.command, variables, sanitize_shell=True)
            m_py = re.search(r"([^\s'\"]+\.py)\b", cmd_interp)
            if m_py:
                script_path = Path(m_py.group(1)).expanduser()
                if script_path.exists() and script_path.is_file():
                    try:
                        from whole_body_compute.integrations.pal_compiler import UniversalAPILoweringResolver
                        from whole_body_compute.local.implementer import ReconImplementer

                        original_src = script_path.read_text(encoding="utf-8")
                        task_desc = (
                            f"Fix the failing workflow script `{script_path.name}` which failed with error:\n"
                            f"{error_msg}\n"
                            f"Keep all existing `assert` precondition/postcondition statements intact."
                        )
                        for model_id in (
                            "voicefi-coder-mlx",
                            "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
                        ):
                            impl = ReconImplementer(
                                scout_model="mlx-community/gemma-2-9b-it-4bit",
                                coder_model=model_id,
                            )
                            res = impl.implement(
                                task=task_desc,
                                file_paths=[str(script_path)],
                                apply=True,
                            )
                            if res.get("applied"):
                                patched_src = script_path.read_text(encoding="utf-8")
                                sec_passed, _ = UniversalAPILoweringResolver.verify_workflow_script_secgate(
                                    patched_src
                                )
                                if sec_passed:
                                    retry_ok, retry_err = cls._execute_step(step, variables)
                                    if retry_ok:
                                        try:
                                            from whole_body_compute.local.dot_memory import get_dot_store

                                            get_dot_store().add_memory(
                                                fact=f"Self-healed workflow script `{script_path.name}` after error '{error_msg[:80]}' using {model_id}.",
                                                category="harness2_lesson",
                                                tags=f"self_heal,workflow,{script_path.name}",
                                            )
                                        except Exception:
                                            pass
                                        return True, ""
                                # Rollback if SecGate or retry failed
                                script_path.write_text(original_src, encoding="utf-8")
                    except Exception:
                        pass

        return False, error_msg


PalWorkflowEngine = PalWorkflowRunner


class PalWorkflowRegistry:
    """Discovers, indexes, and caches declarative Pal workflows across user and project roots."""

    _RECIPES: Dict[str, PalWorkflowRecipe] = {}
    _TRIGGER_INDEX: Dict[str, str] = {}  # trigger_phrase -> recipe_id
    _LAST_LOAD_TIME: float = 0.0

    @classmethod
    def get_search_directories(cls) -> List[Path]:
        dirs = []
        # 1. Project-local
        cwd_wf = Path.cwd() / ".voicefi" / "workflows"
        if cwd_wf.exists():
            dirs.append(cwd_wf)

        # 2. User-global
        home_wf = Path.home() / ".voicefi" / "workflows"
        if home_wf.exists():
            dirs.append(home_wf)

        # 3. Built-in bundled
        bundled = Path(__file__).parent / "pal_recipes"
        if bundled.exists():
            dirs.append(bundled)

        # 4. Signed Procedural DNA Hub recipes (~/.whole_body_compute/hub/recipes/<org>/)
        try:
            from whole_body_compute.hub import get_hub_install_dir

            hub_root = get_hub_install_dir()
            if hub_root.is_dir():
                for org_dir in sorted(hub_root.iterdir()):
                    if org_dir.is_dir():
                        dirs.append(org_dir)
        except Exception:
            pass

        return dirs

    @classmethod
    def reload_workflows(cls) -> None:
        """Scan directories and load all .yaml and .json workflow recipes."""
        recipes: Dict[str, PalWorkflowRecipe] = {}
        trigger_index: Dict[str, str] = {}

        for search_dir in cls.get_search_directories():
            for file_path in search_dir.glob("*.yaml"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                    if isinstance(data, dict) and "id" in data and "steps" in data:
                        if data.get("signature") or data.get("namespace"):
                            from whole_body_compute.hub import verify_recipe_for_execution

                            verify_recipe_for_execution(data, require_signed=True, check_compatibility=False)
                        recipe = PalWorkflowRecipe.model_validate(data)
                        recipes[recipe.id] = recipe
                        if recipe.namespace:
                            trigger_index[f"{recipe.namespace}/{recipe.id}".lower()] = recipe.id
                        # Index triggers
                        for trigger in recipe.triggers:
                            clean_t = trigger.lower().strip()
                            trigger_index[clean_t] = recipe.id
                        # Also index recipe name and id
                        trigger_index[recipe.name.lower().strip()] = recipe.id
                        trigger_index[recipe.id.lower().strip()] = recipe.id
                except Exception as e:
                    print(f"[PalWorkflowRegistry] Error loading {file_path}: {e}")

        # Merge loaded recipes with any existing in-memory recipes
        cls._RECIPES.update(recipes)
        cls._TRIGGER_INDEX.update(trigger_index)
        cls._LAST_LOAD_TIME = time.time()

    @classmethod
    def register_recipe(cls, recipe: PalWorkflowRecipe) -> None:
        """Register a synthesized recipe directly in memory."""
        cls._RECIPES[recipe.id] = recipe
        for trigger in recipe.triggers:
            cls._TRIGGER_INDEX[trigger.lower().strip()] = recipe.id
        cls._TRIGGER_INDEX[recipe.name.lower().strip()] = recipe.id
        cls._TRIGGER_INDEX[recipe.id.lower().strip()] = recipe.id
        cls._LAST_LOAD_TIME = time.time()

    @classmethod
    def match_recipe(cls, prompt: str) -> Optional[Tuple[PalWorkflowRecipe, str]]:
        """
        Match a voice prompt against loaded workflow triggers (<0.5ms).
        Returns (recipe, matched_trigger) if found.
        """
        p_clean = prompt.lower().strip()
        # Remove leading imperative verbs e.g. "run morning setup" -> "morning setup"
        p_stripped = re.sub(r"^(?:run|start|execute|do|trigger)\s+", "", p_clean).strip()

        if not cls._RECIPES or (time.time() - cls._LAST_LOAD_TIME > 60.0):
            cls.reload_workflows()

        recipe_id = cls._TRIGGER_INDEX.get(p_stripped) or cls._TRIGGER_INDEX.get(p_clean)
        if recipe_id and recipe_id in cls._RECIPES:
            return cls._RECIPES[recipe_id], p_stripped

        # Try prefix matching
        for trigger, r_id in cls._TRIGGER_INDEX.items():
            if p_clean == trigger or p_stripped == trigger or p_clean.startswith(trigger + " "):
                return cls._RECIPES[r_id], trigger

        return None

    @classmethod
    def promote_candidate(
        cls, candidate_id: str, share_learnings: bool = False
    ) -> Dict[str, Any]:
        """
        [direct-edit] Promote a staged workflow candidate into active production recipes,
        and optionally harvest/export its sanitized Procedural DNA learning card.
        """
        from whole_body_compute.local.rem_sleep import promote_staged_candidate

        return promote_staged_candidate(candidate_id, share_learnings=share_learnings)

    @classmethod
    def export_sanitized_learning_card(
        cls, recipe_id: str, persist_to_outbox: bool = True
    ) -> Dict[str, Any]:
        """
        [direct-edit] Export a zero-PII Procedural DNA learning card (Compiler IR +
        Generalized Recipe Skeleton) for an active or staged workflow recipe.
        """
        from whole_body_compute.procedural_telemetry import export_sanitized_learning_card

        return export_sanitized_learning_card(
            recipe_id, persist_to_outbox=persist_to_outbox
        )

    @classmethod
    def get_auto_learned_suggestions(
        cls, query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        [direct-edit] Surface active & staged ambient workflows, tool pruning recommendations,
        and faster execution routes from recipes/candidates/ and dot_memory.db.
        """
        suggestions: List[Dict[str, Any]] = []

        # 1. Staged & Active Ambient Workflow Candidates
        try:
            from whole_body_compute.local.rem_sleep import list_staged_candidates

            cands_report = list_staged_candidates()
            for cand in cands_report.get("candidates", []):
                tier = cand.get("dominant_tier", "TIER_0_LOCAL")
                rel = cand.get("reliability_score", 1.0)
                is_active = cand.get("promoted_to_active", False)
                cid = cand.get("id", "")
                cname = cand.get("name", cid)

                faster_summary = (
                    "Replaces multi-step GUI actions with 1 verified headless REST command (Tier 0 Local)."
                    if tier == "TIER_0_LOCAL"
                    else "Synthesized multi-app coordinate GUI flow ready for 4-Pass lowering."
                )

                cand_domain = cand.get("domain") or "desktop"
                suggestions.append(
                    {
                        "type": "ambient_workflow",
                        "id": cid,
                        "title": cname,
                        "domain": cand_domain,
                        "status": "active" if is_active else "staged",
                        "dominant_tier": tier,
                        "reliability_score": rel,
                        "contracts": cand.get("contracts", []),
                        "steps_count": cand.get("steps_count", 1),
                        "recommendation": faster_summary,
                        "trigger_example": f"body execute {cid}" if is_active else f"body candidates --promote {cid}",
                        "tags": ["ambient", "workflow", tier.lower(), cand_domain.lower()],
                    }
                )
        except Exception:
            pass

        # 2. Consolidated Tool Pruning & Speed Route Habits from DotMemory
        try:
            import sqlite3

            db_path = Path.home() / ".voicefi" / "dot_memory.db"
            if db_path.is_file():
                with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0) as conn:
                    conn.row_factory = sqlite3.Row
                    rows = conn.execute(
                        """
                        SELECT id, category, tags, fact, salience
                        FROM memories
                        WHERE category IN ('habit_pattern', 'harness2_lesson', 'rsi_superintelligence_rule')
                        ORDER BY salience DESC LIMIT 25
                        """
                    ).fetchall()

                    for r in rows:
                        fact_text = str(r["fact"] or "")
                        tags_text = str(r["tags"] or "")
                        cat = str(r["category"] or "")

                        if "prune toolset to" in fact_text:
                            # Extract pruned tools
                            tools_match = re.search(r"prune toolset to (\[.*?\])", fact_text)
                            pruned_tools = tools_match.group(1) if tools_match else "[]"
                            domain = "macos_desktop" if "macos_desktop" in tags_text else "shell_and_files"
                            suggestions.append(
                                {
                                    "type": "tool_pruning_habit",
                                    "id": f"habit_{domain}_{r['id']}",
                                    "title": f"Tool Pruning Optimization: {domain.upper().replace('_', ' ')}",
                                    "status": "learned",
                                    "domain": domain,
                                    "pruned_tools": pruned_tools,
                                    "recommendation": fact_text[:280],
                                    "salience": float(r["salience"] or 0.9),
                                    "tags": ["tool_pruning", domain, "habit"],
                                }
                            )
                        elif cat == "rsi_superintelligence_rule":
                            suggestions.append(
                                {
                                    "type": "rsi_rule",
                                    "id": f"rsi_{r['id']}",
                                    "title": (
                                        fact_text.split("]:", 1)[0].strip("[]")
                                        if "]:" in fact_text
                                        else f"RSI Rule #{r['id']}"
                                    ),
                                    "status": "active_rule",
                                    "recommendation": fact_text[:280],
                                    "salience": float(r["salience"] or 0.95),
                                    "tags": ["rsi", "guardrail"],
                                }
                            )
        except Exception:
            pass

        # 3. Filter by optional query string if provided
        if query and query.strip():
            q_low = query.strip().lower()
            filtered = []
            for s in suggestions:
                score = 0
                blob = f"{s.get('id', '')} {s.get('title', '')} {s.get('recommendation', '')} {' '.join(s.get('tags', []))}".lower()
                for term in q_low.split():
                    if term in blob:
                        score += 1
                if score > 0:
                    s["query_match_score"] = score
                    filtered.append(s)
            filtered.sort(key=lambda x: x.get("query_match_score", 0), reverse=True)
            return filtered

        return suggestions
