from importlib import import_module
from typing import Any

from .backend import RocmCapability, probe_rocm, require_rocm


def flash_attention_forward(*args, **kwargs):
    require_rocm()
    from .kernel import flash_attention_forward as triton_attention

    return triton_attention(*args, **kwargs)


def __getattr__(name: str) -> Any:
    if name in {
        "reference_attention",
        "reference_attention_backward",
        "reference_attention_delta",
        "reference_attention_forward",
    }:
        return getattr(import_module(".reference", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RocmCapability",
    "flash_attention_forward",
    "probe_rocm",
    "reference_attention",
    "reference_attention_backward",
    "reference_attention_delta",
    "reference_attention_forward",
    "require_rocm",
]
