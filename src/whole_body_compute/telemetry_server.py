"""Whole Body Compute - Open-Core Pro Feature Stub.

This module is part of Whole Body Compute Pro ($29/mo) and Enterprise tiers.
It is excluded from the open-core community distribution and dynamically
bridges to `whole-body-compute-pro` when an active Polar license is verified.

To upgrade and activate this capability:
    body license --activate <POLAR_KEY>
or visit https://wholebodycompute.com/pricing
"""

from __future__ import annotations

from typing import Any, Dict, List

__wbc_pro_stub__: bool = True
__wbc_pro_module_key__: str = "telemetry_server"

UPGRADE_MESSAGE: str = (
    "Requires Whole Body Compute Pro ($29/mo). "
    "Run: body license --activate <POLAR_KEY> or visit https://wholebodycompute.com/pricing"
)

try:
    from whole_body_compute.pro_loader import ProUpgradeRequiredError, load_pro_module
except Exception:  # pragma: no cover - standalone spec_from_file_location fallback

    class ProUpgradeRequiredError(NotImplementedError):  # type: ignore[no-redef]
        """Raised when a Pro/Team capability is invoked without an active Pro+ license."""

    def load_pro_module(module_key: str, **kwargs: Any) -> Any:  # type: ignore[misc]
        return None


class _ProStubMeta(type):
    """Metaclass intercepting class-level calls, attribute lookups, and late Pro activation."""

    def __getattr__(cls, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        sym = cls.__dict__.get("_wbc_symbol_name")
        if sym:
            pro_mod = load_pro_module(__wbc_pro_module_key__, stub_file=globals().get("__file__"))
            if pro_mod is not None and hasattr(pro_mod, sym):
                return getattr(getattr(pro_mod, sym), name)
        return cls

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        sym = cls.__dict__.get("_wbc_symbol_name")
        if sym:
            pro_mod = load_pro_module(__wbc_pro_module_key__, stub_file=globals().get("__file__"))
            if pro_mod is not None and hasattr(pro_mod, sym):
                target = getattr(pro_mod, sym)
                if callable(target):
                    return target(*args, **kwargs)
        raise ProUpgradeRequiredError(UPGRADE_MESSAGE)

    def __bool__(cls) -> bool:
        return False


class ProFeatureStub(metaclass=_ProStubMeta):
    """Fallback stub class for Pro/Team capabilities in open-core."""

    _wbc_symbol_name: str = ""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ProUpgradeRequiredError(UPGRADE_MESSAGE)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise ProUpgradeRequiredError(UPGRADE_MESSAGE)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return self

    @classmethod
    def __class_getitem__(cls, item: Any) -> Any:
        return cls


_SYMBOL_PROXIES: Dict[str, type] = {}


def _get_symbol_proxy(name: str) -> Any:
    cached = _SYMBOL_PROXIES.get(name)
    if cached is not None:
        return cached
    proxy = _ProStubMeta(f"ProFeatureStub_{name}", (ProFeatureStub,), {"_wbc_symbol_name": name})
    _SYMBOL_PROXIES[name] = proxy
    return proxy


def __getattr__(name: str) -> Any:
    """Delegate dynamically to `whole-body-compute-pro` when licensed, else return bound ProFeatureStub."""
    if name.startswith("__") and name.endswith("__"):
        if name == "__all__":
            pro_mod = load_pro_module(__wbc_pro_module_key__, stub_file=globals().get("__file__"))
            return list(getattr(pro_mod, "__all__", [])) if pro_mod is not None else []
        raise AttributeError(name)

    pro_mod = load_pro_module(__wbc_pro_module_key__, stub_file=globals().get("__file__"))
    if pro_mod is not None:
        return getattr(pro_mod, name)
    return _get_symbol_proxy(name)


def __dir__() -> List[str]:
    pro_mod = load_pro_module(__wbc_pro_module_key__, stub_file=globals().get("__file__"))
    base = set(globals().keys())
    if pro_mod is not None:
        base.update(dir(pro_mod))
    return sorted(base)
