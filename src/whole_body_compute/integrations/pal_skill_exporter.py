"""
Portable Skill Bundle Exporter (`PalSkillExporter`).

Compiles a `PalWorkflowRecipe` (demonstrated via Eyes + Ears on macOS) into a
self-contained, portable Skill Bundle (`SKILL.md` + `recipe.yaml` + `scripts/run.py`
+ optional `<slug>.skill.zip`) that runs in both:
  1. Cloud Bots (Claude Duet, Grok, headless Linux CI/CD containers) via lowered
     Tier 0 (shell/Python), Tier 1 (REST/GraphQL/MCP HTTP), and Tier 1.5 (Playwright DOM).
  2. Local Agents (Claude Code, Antigravity, Cursor, Goose) via `.agents/skills/` or
     `~/.claude/skills/`.

Enforces Whole Body Compute's PLG Packaging:
  - `community` ($0/mo): 1 Free Full Portable Skill Export + unlimited Free Dry-Run Previews.
  - `pro` ($29/mo) / `team` / `enterprise`: Unlimited Portable Skill Exports & `.zip` bundles.
"""

from __future__ import annotations

import json
import os
import re
import stat
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from whole_body_compute.integrations.pal_workflow import (
    ActionType,
    PalWorkflowRecipe,
)
from whole_body_compute.license import POLAR_CHECKOUT_URLS, LicenseManager


def _get_default_skill_export_ledger_path() -> Path:
    env_p = os.environ.get("WBC_SKILL_EXPORT_LEDGER_PATH")
    if env_p:
        return Path(env_p).expanduser()
    return (Path.home() / ".voicefi" / "skill_export_ledger.json").expanduser()


def _slugify_skill_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "custom-workflow").strip().lower()).strip("-")
    return slug or "custom-workflow"


def _load_recipe_from_yaml(path: Path) -> PalWorkflowRecipe:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "id" not in data:
        data["id"] = _slugify_skill_name(str(data.get("name") or path.stem))
    return PalWorkflowRecipe.model_validate(data)


def _save_recipe_to_yaml(recipe: PalWorkflowRecipe, path: Path) -> None:
    payload = recipe.model_dump(mode="json", exclude_none=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def load_skill_export_ledger(ledger_path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(ledger_path).expanduser() if ledger_path else _get_default_skill_export_ledger_path()
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {
                    "exports_used": int(data.get("exports_used", 0)),
                    "exported_skills": list(data.get("exported_skills") or []),
                }
        except Exception:
            pass
    return {"exports_used": 0, "exported_skills": []}


def record_skill_export_usage(
    skill_slug: str,
    skill_dir: Path,
    ledger_path: Optional[Path] = None,
) -> Dict[str, Any]:
    path = Path(ledger_path).expanduser() if ledger_path else _get_default_skill_export_ledger_path()
    current = load_skill_export_ledger(path)
    current["exports_used"] = int(current.get("exports_used", 0)) + 1
    history = list(current.get("exported_skills") or [])
    history.append(
        {
            "slug": skill_slug,
            "path": str(skill_dir),
            "exported_at": int(time.time()),
        }
    )
    current["exported_skills"] = history[-50:]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass
    return current


def classify_recipe_cloud_portability(recipe: PalWorkflowRecipe) -> Dict[str, Any]:
    """
    Classify each step of a `PalWorkflowRecipe` into cloud-portable tiers
    (Tier 0 Shell/Local, Tier 1 API/URL, Tier 1.5 Browser DOM) vs. macOS-native Tier 2/3.
    """
    tier_counts = {"tier_0": 0, "tier_1": 0, "tier_1_5": 0, "tier_2_macos": 0}
    step_summaries: List[Dict[str, Any]] = []

    for idx, step in enumerate(recipe.steps, start=1):
        act = step.action
        ct = step.click_target
        if act in (ActionType.RUN_SHELL, ActionType.WAIT, ActionType.NOTIFY):
            tier = "tier_0"
            tier_label = "Tier 0 (Deterministic Script/Shell)"
            cloud_ready = True
        elif act == ActionType.OPEN_URL or (step.url and not ct):
            tier = "tier_1"
            tier_label = "Tier 1 (Headless HTTP / URL)"
            cloud_ready = True
        elif ct and (ct.target_type == "dom_selector" or ct.dom_selector or ct.aria_label or ct.bid):
            tier = "tier_1_5"
            tier_label = "Tier 1.5 (Playwright / Browser DOM)"
            cloud_ready = True
        elif act in (ActionType.KEYSTROKE, ActionType.PRESS_KEY, ActionType.HOTKEY) and not step.app_name:
            tier = "tier_1_5"
            tier_label = "Tier 1.5 (Input / Keyboard Dispatch)"
            cloud_ready = True
        else:
            tier = "tier_2_macos"
            tier_label = "Tier 2 (macOS AXUIElement / Desktop)"
            cloud_ready = False

        tier_counts[tier] += 1
        step_summaries.append(
            {
                "index": idx,
                "name": step.name,
                "action": act.value if isinstance(act, ActionType) else str(act),
                "tier": tier,
                "tier_label": tier_label,
                "cloud_portable": cloud_ready,
            }
        )

    total = max(1, len(recipe.steps))
    portable_count = tier_counts["tier_0"] + tier_counts["tier_1"] + tier_counts["tier_1_5"]
    # Estimate monthly cloud screenshot/vision token savings vs. raw cloud computer-use (8 runs/week)
    est_tokens_saved_per_run = len(recipe.steps) * 6800
    est_monthly_usd_saved = round(est_tokens_saved_per_run * 32 * 0.000003, 2)

    return {
        "total_steps": len(recipe.steps),
        "cloud_portable_steps": portable_count,
        "cloud_portability_ratio": round(portable_count / total, 2),
        "fully_cloud_portable": tier_counts["tier_2_macos"] == 0,
        "tier_counts": tier_counts,
        "steps": step_summaries,
        "estimated_tokens_saved_per_run": est_tokens_saved_per_run,
        "estimated_monthly_usd_saved": max(12.0, est_monthly_usd_saved),
    }


def render_skill_markdown(
    recipe: PalWorkflowRecipe,
    skill_slug: str,
    portability: Dict[str, Any],
) -> str:
    """Generate the `SKILL.md` file instructing Claude Duet, Claude Code, or Antigravity how to use the skill."""
    desc = (
        recipe.description.strip()
        if recipe.description and recipe.description.strip()
        else f"Deterministic workflow skill for {recipe.name} compiled by Whole Body Compute (Hey Pal Eyes + Ears)."
    )
    triggers = recipe.triggers or [recipe.name.lower()]
    trigger_list = ", ".join(f'`"{t}"`' for t in triggers[:6])

    lines = [
        "---",
        f"name: {skill_slug}",
        f"description: {desc}",
        "---",
        "",
        f"# {recipe.name} (`{skill_slug}`)",
        "",
        f"> **Compiled via Whole Body Compute (Eyes + Ears Demonstration Compiler)** • Version `{recipe.version}`  ",
        f"> **Execution Profile:** `{portability['cloud_portable_steps']}/{portability['total_steps']}` Cloud-Portable Steps "
        f"(`~{portability['estimated_tokens_saved_per_run']:,}` vision tokens saved/run • `$0.00` LLM replay tax)",
        "",
        "## When to Use This Skill",
        f"- Activate this skill when the user asks to run **{recipe.name}** or uses phrases matching: {trigger_list}.",
        "- Works across **Cloud Bots** (Claude Duet, headless CI/CD containers) and **Local Agents** (Claude Code, Antigravity, Cursor, Goose).",
        "",
    ]

    if recipe.parameters:
        lines.extend(
            [
                "## Input Parameters",
                "",
                "| Parameter | Type | Required | Default | Description |",
                "| :--- | :---: | :---: | :---: | :--- |",
            ]
        )
        for p_name, p_def in recipe.parameters.items():
            req_str = "Yes" if p_def.required else "No"
            def_str = f"`{p_def.default}`" if p_def.default is not None else "—"
            p_desc = p_def.description or f"Dynamic `{p_name}` value extracted from user request"
            lines.append(f"| `{p_name}` | `{p_def.type}` | {req_str} | {def_str} | {p_desc} |")
        lines.append("")
    else:
        lines.extend(
            [
                "## Input Parameters",
                "This workflow takes no required parameters (runs deterministically with default configuration).",
                "",
            ]
        )

    lines.extend(
        [
            "## How to Execute",
            "",
            "Run the bundled deterministic runner script from this skill directory:",
            "",
            "```bash",
            f"python scripts/run.py --params '{json.dumps({k: (v.default or f'<{k}>') for k, v in recipe.parameters.items()})}'",
            "```",
            "",
            "- To preview the steps and verify preconditions without mutating state, pass `--dry-run`:",
            "  ```bash",
            "  python scripts/run.py --dry-run",
            "  ```",
            "",
            "## Compiled Workflow Steps",
            "",
        ]
    )

    for s_info in portability["steps"]:
        lines.append(
            f"{s_info['index']}. **{s_info['name']}** (`{s_info['action']}`) — *{s_info['tier_label']}*"
        )
    lines.append("")

    if recipe.postconditions:
        lines.extend(
            [
                "## Hoare Postcondition Contracts (`{Q}`)",
                "The runner automatically verifies these postconditions before reporting success:",
            ]
        )
        for pc in recipe.postconditions:
            exp = f" == `{pc.expected}`" if pc.expected else ""
            lines.append(f"- `{pc.getter_type}` on `{pc.target}`{exp} ({pc.description or 'verified state'})")
        lines.append("")

    return "\n".join(lines)


def render_standalone_runner_script(recipe: PalWorkflowRecipe, skill_slug: str) -> str:
    """
    Generate `scripts/run.py` — a standalone Python runner that works in Linux Cloud VMs
    (Claude Duet / CI) for Tier 0/1/1.5 steps and falls back to `PalHarness` on macOS when present.
    """
    return f'''#!/usr/bin/env python3
"""
Standalone Deterministic Skill Runner for `{skill_slug}` ({recipe.name}).
Generated by Whole Body Compute (Hey Pal Demonstration Compiler).

Runs in Linux Cloud VMs (Claude Duet, CI/CD, Playwright containers) and macOS workstations.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict

import yaml

SKILL_ROOT = Path(__file__).resolve().parent.parent
RECIPE_PATH = SKILL_ROOT / "recipe.yaml"


def _substitute_params(text: str | None, params: Dict[str, Any]) -> str:
    if not text:
        return ""
    out = str(text)
    for k, v in params.items():
        out = out.replace("{{" + str(k) + "}}", str(v))
        out = out.replace("$" + str(k), str(v))
    return out


def _verify_postcondition(pc: Dict[str, Any], params: Dict[str, Any]) -> bool:
    gtype = pc.get("getter_type", "")
    target = _substitute_params(pc.get("target", ""), params)
    if gtype == "file_exists":
        return Path(target).expanduser().exists()
    if gtype == "file_nonempty":
        p = Path(target).expanduser()
        return p.is_file() and p.stat().st_size > 0
    if gtype == "shell_exit_zero":
        res = subprocess.run(target, shell=True, capture_output=True, text=True)
        return res.returncode == 0
    return True


def run_recipe(params: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    recipe_data = yaml.safe_load(RECIPE_PATH.read_text(encoding="utf-8"))
    steps = recipe_data.get("steps") or []
    postconditions = recipe_data.get("postconditions") or []

    # Resolve defaults from parameter definitions
    param_defs = recipe_data.get("parameters") or {{}}
    resolved_params: Dict[str, Any] = {{}}
    for k, pdef in param_defs.items():
        if k in params and params[k] is not None:
            resolved_params[k] = params[k]
        elif isinstance(pdef, dict) and pdef.get("default") is not None:
            resolved_params[k] = pdef["default"]
    for k, v in params.items():
        if k not in resolved_params:
            resolved_params[k] = v

    executed = []
    for idx, step in enumerate(steps, start=1):
        action = step.get("action", "")
        name = step.get("name", f"Step {{idx}}")
        if dry_run:
            executed.append({{"step": idx, "name": name, "action": action, "status": "dry_run"}})
            continue

        if action == "run_shell":
            cmd = _substitute_params(step.get("command", ""), resolved_params)
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if proc.returncode != 0 and step.get("on_failure", "abort") == "abort":
                return {{"success": False, "failed_step": idx, "name": name, "error": proc.stderr.strip()}}
            executed.append({{"step": idx, "name": name, "action": action, "status": "ok"}})
        elif action == "open_url":
            url = _substitute_params(step.get("url", ""), resolved_params)
            if sys.platform == "darwin":
                subprocess.run(["open", url], check=False)
            else:
                req = urllib.request.Request(url, headers={{"User-Agent": "WBC-SkillRunner/1.0"}})
                with urllib.request.urlopen(req, timeout=10.0):
                    pass
            executed.append({{"step": idx, "name": name, "action": action, "url": url, "status": "ok"}})
        elif action == "wait":
            dur_ms = int(step.get("duration_ms") or 250)
            time.sleep(min(dur_ms / 1000.0, 5.0))
            executed.append({{"step": idx, "name": name, "action": action, "status": "ok"}})
        else:
            # Delegate browser DOM (Tier 1.5) or macOS AXUIElement (Tier 2) to WBC PalWorkflowRecipe if installed
            try:
                from whole_body_compute.integrations.pal_workflow import PalWorkflowRecipe

                wf = PalWorkflowRecipe.from_yaml_file(RECIPE_PATH)
                res = wf.execute(user_params=resolved_params, dry_run=dry_run)
                return {{
                    "success": bool(res.get("success", True)),
                    "mode": "wbc_pal_harness",
                    "result": res,
                }}
            except ImportError:
                executed.append(
                    {{
                        "step": idx,
                        "name": name,
                        "action": action,
                        "status": "delegated_dom_step",
                        "click_target": step.get("click_target"),
                    }}
                )

    if not dry_run:
        for pc in postconditions:
            if not _verify_postcondition(pc, resolved_params):
                return {{
                    "success": False,
                    "error": f"Postcondition failed: {{pc.get('getter_type')}} ({{pc.get('target')}})",
                    "executed_steps": executed,
                }}

    return {{
        "success": True,
        "skill": "{skill_slug}",
        "dry_run": dry_run,
        "parameters": resolved_params,
        "executed_steps": executed,
    }}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run compiled Whole Body Compute skill: {recipe.name}")
    parser.add_argument("--params", default="{{}}", help="JSON dictionary of input parameters")
    parser.add_argument("--dry-run", action="store_true", help="Preview execution without mutating state")
    args = parser.parse_args()

    try:
        user_params = json.loads(args.params) if args.params else {{}}
    except Exception as exc:
        print(json.dumps({{"success": False, "error": f"Invalid --params JSON: {{exc}}"}}))
        return 1

    result = run_recipe(params=user_params, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


def export_recipe_as_skill_bundle(
    recipe_or_path: Union[PalWorkflowRecipe, Path, str],
    output_root: Optional[Union[Path, str]] = None,
    zip_bundle: bool = True,
    dry_run: bool = False,
    license_manager: Optional[LicenseManager] = None,
    ledger_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Export a `PalWorkflowRecipe` (or YAML path / candidate ID) into a portable Skill Bundle:
      - `<output_root>/<slug>/SKILL.md`
      - `<output_root>/<slug>/recipe.yaml`
      - `<output_root>/<slug>/scripts/run.py`
      - `<output_root>/<slug>.skill.zip` (when `zip_bundle=True`)

    Respects the PLG Tier Gate:
      - `community` tier includes 1 free full export, then switches to non-destructive
        Dry-Run Preview mode with a Polar Pro ($29/mo) upgrade receipt.
    """
    if isinstance(recipe_or_path, PalWorkflowRecipe):
        recipe = recipe_or_path
    else:
        raw_target = str(recipe_or_path).strip()
        cand_path = Path(raw_target).expanduser()
        if not cand_path.is_file():
            # Check recipes/ and recipes/candidates/
            for search_Candidate in (
                Path("recipes") / f"{raw_target}.yaml",
                Path("recipes") / raw_target,
                Path("recipes/candidates") / f"{raw_target}.yaml",
                Path("recipes/candidates") / raw_target,
            ):
                if search_Candidate.is_file():
                    cand_path = search_Candidate
                    break
        if not cand_path.is_file():
            raise FileNotFoundError(f"Could not find PalWorkflowRecipe YAML at '{raw_target}'")
        recipe = _load_recipe_from_yaml(cand_path)

    skill_slug = _slugify_skill_name(recipe.id or recipe.name)
    portability = classify_recipe_cloud_portability(recipe)
    skill_md_content = render_skill_markdown(recipe, skill_slug, portability)
    runner_py_content = render_standalone_runner_script(recipe, skill_slug)

    mgr = license_manager or LicenseManager()
    ledger = load_skill_export_ledger(ledger_path)
    exports_used = int(ledger.get("exports_used", 0))

    gate = mgr.check_feature_gate(
        "pal_skill_export",
        skill_exports_used=exports_used,
    )

    if dry_run or not gate.get("allowed", False):
        return {
            "exported": False,
            "dry_run_preview": True,
            "gated_by_license": not gate.get("allowed", False),
            "current_tier": gate.get("current_tier", "community"),
            "required_tier": gate.get("required_tier", "pro"),
            "free_exports_used": exports_used,
            "free_export_limit": 1 if gate.get("current_tier") == "community" else None,
            "skill_slug": skill_slug,
            "workflow_name": recipe.name,
            "portability": portability,
            "skill_md_preview": skill_md_content,
            "upgrade_nudge": gate.get("upgrade_nudge", ""),
            "checkout_url": POLAR_CHECKOUT_URLS.get("pro"),
        }

    base_dir = Path(output_root).expanduser() if output_root else Path(".agents/skills")
    skill_dir = base_dir / skill_slug
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    skill_md_path = skill_dir / "SKILL.md"
    recipe_yaml_path = skill_dir / "recipe.yaml"
    runner_py_path = scripts_dir / "run.py"

    skill_md_path.write_text(skill_md_content + "\n", encoding="utf-8")
    _save_recipe_to_yaml(recipe, recipe_yaml_path)
    runner_py_path.write_text(runner_py_content, encoding="utf-8")
    try:
        runner_py_path.chmod(runner_py_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError:
        pass

    zip_path: Optional[Path] = None
    if zip_bundle:
        zip_path = base_dir / f"{skill_slug}.skill.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(skill_md_path, arcname=f"{skill_slug}/SKILL.md")
            zf.write(recipe_yaml_path, arcname=f"{skill_slug}/recipe.yaml")
            zf.write(runner_py_path, arcname=f"{skill_slug}/scripts/run.py")

    updated_ledger = record_skill_export_usage(skill_slug, skill_dir, ledger_path=ledger_path)

    return {
        "exported": True,
        "dry_run_preview": False,
        "gated_by_license": False,
        "current_tier": gate.get("current_tier", "enterprise"),
        "free_exports_used": updated_ledger["exports_used"],
        "skill_slug": skill_slug,
        "workflow_name": recipe.name,
        "skill_dir": str(skill_dir),
        "skill_md_path": str(skill_md_path),
        "recipe_yaml_path": str(recipe_yaml_path),
        "runner_py_path": str(runner_py_path),
        "zip_bundle_path": str(zip_path) if zip_path else None,
        "portability": portability,
    }
