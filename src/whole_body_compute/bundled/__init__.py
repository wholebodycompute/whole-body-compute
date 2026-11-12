"""
Bundled runtime assets shipped inside the `whole-body-compute` wheel so Homebrew,
`uv tool`, PyPI, and Open-Core GitHub installations never depend on repo-external paths.
"""

from __future__ import annotations

from pathlib import Path

BUNDLED_DIR = Path(__file__).resolve().parent


def get_bundled_audit_script_path() -> Path:
    """Return the bundled stdlib-only 10-harness FinOps audit script path."""
    return BUNDLED_DIR / "standalone_audit.py"


def get_bundled_immune_hook_path() -> Path:
    """Return the bundled <2ms POSIX sh Blood-Brain Barrier PreToolUse hook script path."""
    return BUNDLED_DIR / "body_immune_hook.sh"
