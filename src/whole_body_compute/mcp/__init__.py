"""Whole Body Compute MCP App servers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from whole_body_compute.mcp.local_agents_server import (
        RESOURCE_MIME_TYPE,
        RESOURCE_URI,
        LocalAgentsMCPServer,
        main,
    )

__all__ = [
    "RESOURCE_MIME_TYPE",
    "RESOURCE_URI",
    "LocalAgentsMCPServer",
    "main",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve package exports so `python -m whole_body_compute.mcp.local_agents_server` emits no runpy warning."""
    if name in __all__:
        from whole_body_compute.mcp import local_agents_server

        return getattr(local_agents_server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
