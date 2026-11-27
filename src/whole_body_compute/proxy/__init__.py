"""
Whole Body Compute FinOps API Proxy (`body proxy`).

Drop-in local HTTP/SSE gateway on `127.0.0.1:9381` providing deterministic
Anthropic `cache_control` injection, pre-egress secret scrubbing, opt-in
prose hygiene, per-client attribution, and Community 250/day fail-open metering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from whole_body_compute.proxy.pricing import (
    MODEL_FULL_PRICING_PER_M,
    ProxyCostReceipt,
    compute_proxy_call_cost,
    is_local_model_slug,
    lookup_proxy_model_pricing,
)


@dataclass
class ProxyConfig:
    """Configuration for the local `body proxy` HTTP/SSE gateway."""

    host: str = "127.0.0.1"
    port: int = 9381
    anthropic_upstream: str = "https://api.anthropic.com"
    openai_upstream: str = "https://api.openai.com"
    anti_slop: bool = False
    chore_router: bool = False
    community_daily_cache_cap: int = 250


def run_proxy_server(*args: Any, **kwargs: Any) -> Any:
    """Lazy entrypoint to `whole_body_compute.proxy.server.run_proxy_server`."""
    from whole_body_compute.proxy.server import run_proxy_server as _run_proxy_server

    return _run_proxy_server(*args, **kwargs)


__all__ = [
    "MODEL_FULL_PRICING_PER_M",
    "ProxyConfig",
    "ProxyCostReceipt",
    "compute_proxy_call_cost",
    "is_local_model_slug",
    "lookup_proxy_model_pricing",
    "run_proxy_server",
]
