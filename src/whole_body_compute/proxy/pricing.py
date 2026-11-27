"""
FinOps API Proxy exact per-model USD pricing & savings calculator.

Computes baseline uncached cost vs. actual cache-adjusted cost
(`1.25x` ephemeral cache write, `0.10x` ephemeral cache read) across
Anthropic, OpenAI, and `$0.00` local Apple Silicon Metal models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

# (input_usd_per_m, cache_write_usd_per_m, cache_read_usd_per_m, output_usd_per_m)
MODEL_FULL_PRICING_PER_M: Dict[str, Tuple[float, float, float, float]] = {
    # Anthropic Claude 4 / 4.5 / 4.6
    "claude-opus-4-6": (5.00, 6.25, 0.50, 25.00),
    "claude-opus-4-5": (5.00, 6.25, 0.50, 25.00),
    "claude-opus-4-1": (15.00, 18.75, 1.50, 75.00),
    "claude-opus-4": (15.00, 18.75, 1.50, 75.00),
    "claude-3-opus": (15.00, 18.75, 1.50, 75.00),
    "claude-sonnet-4-5": (3.00, 3.75, 0.30, 15.00),
    "claude-sonnet-4": (3.00, 3.75, 0.30, 15.00),
    "claude-3-7-sonnet": (3.00, 3.75, 0.30, 15.00),
    "claude-3-5-sonnet": (3.00, 3.75, 0.30, 15.00),
    "claude-haiku-4-5": (1.00, 1.25, 0.10, 5.00),
    "claude-3-5-haiku": (0.80, 1.00, 0.08, 4.00),
    "claude-3-haiku": (0.25, 0.3125, 0.025, 1.25),
    # OpenAI GPT / o-series
    "gpt-5": (1.25, 1.25, 0.125, 10.00),
    "gpt-4.1": (2.00, 2.00, 0.50, 8.00),
    "gpt-4.1-mini": (0.40, 0.40, 0.10, 1.60),
    "gpt-4.1-nano": (0.10, 0.10, 0.025, 0.40),
    "gpt-4o": (2.50, 2.50, 1.25, 10.00),
    "gpt-4o-mini": (0.15, 0.15, 0.075, 0.60),
    "o3": (2.00, 2.00, 0.50, 8.00),
    "o3-mini": (1.10, 1.10, 0.55, 4.40),
    "o4-mini": (1.10, 1.10, 0.275, 4.40),
    # Local Apple Silicon ($0.00)
    "local-metal": (0.0, 0.0, 0.0, 0.0),
    "body-coder-mlx": (0.0, 0.0, 0.0, 0.0),
    "voicefi-coder-mlx": (0.0, 0.0, 0.0, 0.0),
}

_LOCAL_PREFIXES: Tuple[str, ...] = (
    "gemma",
    "qwen",
    "llama",
    "voicefi-",
    "body-coder",
    "mlx/",
    "ollama/",
    "local",
)


@dataclass(frozen=True)
class ProxyCostReceipt:
    """Exact USD and equivalent-token savings breakdown for a single proxied call."""

    model: str
    uncached_input_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    output_tokens: int
    baseline_uncached_usd: float
    actual_billed_usd: float
    net_usd_saved: float
    equivalent_input_tokens_saved: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def is_local_model_slug(model: str) -> bool:
    """Return True if `model` targets a local Apple Silicon / Ollama endpoint."""
    key = (model or "").strip().lower()
    if not key:
        return False
    return any(key.startswith(p) for p in _LOCAL_PREFIXES)


def lookup_proxy_model_pricing(model: str) -> Tuple[float, float, float, float]:
    """
    Resolve `(input_per_m, cache_write_per_m, cache_read_per_m, output_per_m)` for `model`.
    Falls back to Sonnet-class `$3/$3.75/$0.30/$15` for unknown cloud models.
    """
    key = (model or "").strip().lower()
    if not key:
        return (3.00, 3.75, 0.30, 15.00)
    if is_local_model_slug(key):
        return (0.0, 0.0, 0.0, 0.0)
    for prefix, rates in MODEL_FULL_PRICING_PER_M.items():
        if key == prefix or key.startswith(prefix):
            return rates
    if "opus" in key:
        if "4-5" in key or "4.5" in key or "4-6" in key or "4.6" in key:
            return (5.00, 6.25, 0.50, 25.00)
        return (15.00, 18.75, 1.50, 75.00)
    if "haiku" in key:
        if "4-5" in key or "4.5" in key:
            return (1.00, 1.25, 0.10, 5.00)
        return (0.80, 1.00, 0.08, 4.00)
    if "sonnet" in key:
        return (3.00, 3.75, 0.30, 15.00)
    return (3.00, 3.75, 0.30, 15.00)


def compute_proxy_call_cost(
    model: str,
    *,
    uncached_input_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
    output_tokens: int = 0,
    baseline_model: str | None = None,
) -> ProxyCostReceipt:
    """
    Compute baseline uncached USD vs. actual cache-adjusted (or chore-downshifted) USD.
    When `baseline_model` is provided (e.g. chore router downshifted Sonnet -> Haiku/$0 Metal),
    `baseline_uncached_usd` is priced at `baseline_model` while `actual_billed_usd` uses `model`.
    """
    u_in = max(0, int(uncached_input_tokens or 0))
    cw = max(0, int(cache_write_tokens or 0))
    cr = max(0, int(cache_read_tokens or 0))
    out = max(0, int(output_tokens or 0))

    b_in_p, _, _, b_out_p = lookup_proxy_model_pricing(baseline_model or model)
    a_in_p, a_cw_p, a_cr_p, a_out_p = lookup_proxy_model_pricing(model)

    total_in = u_in + cw + cr
    baseline_usd = ((total_in / 1_000_000.0) * b_in_p) + ((out / 1_000_000.0) * b_out_p)
    actual_usd = (
        ((u_in / 1_000_000.0) * a_in_p)
        + ((cw / 1_000_000.0) * a_cw_p)
        + ((cr / 1_000_000.0) * a_cr_p)
        + ((out / 1_000_000.0) * a_out_p)
    )
    net_saved = baseline_usd - actual_usd

    if b_in_p > 0.0 and baseline_model and baseline_model != model:
        eq_tok = int(round(max(0.0, net_saved) / (b_in_p / 1_000_000.0)))
    elif a_in_p > 0.0 and cr > 0:
        # 0.10x cache read saves 90% of read tokens minus 25% write premium
        eq_tok = max(0, int(round(cr * (1.0 - (a_cr_p / a_in_p)) - cw * ((a_cw_p / a_in_p) - 1.0))))
    else:
        eq_tok = 0

    return ProxyCostReceipt(
        model=model,
        uncached_input_tokens=u_in,
        cache_write_tokens=cw,
        cache_read_tokens=cr,
        output_tokens=out,
        baseline_uncached_usd=round(baseline_usd, 6),
        actual_billed_usd=round(actual_usd, 6),
        net_usd_saved=round(net_saved, 6),
        equivalent_input_tokens_saved=eq_tok,
    )


__all__ = [
    "MODEL_FULL_PRICING_PER_M",
    "ProxyCostReceipt",
    "compute_proxy_call_cost",
    "is_local_model_slug",
    "lookup_proxy_model_pricing",
]
