from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class RocmCapability:
    available: bool
    reason: str | None
    torch_version: str | None
    hip_version: str | None
    triton_available: bool
    accelerator_available: bool
    gpu_name: str | None
    architecture: str | None
    bf16_supported: bool


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _device_details(torch: Any) -> tuple[str | None, str | None, bool]:
    if not torch.cuda.is_available():
        return None, None, False

    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    properties = torch.cuda.get_device_properties(device)
    architecture = getattr(properties, "gcnArchName", None)
    bf16_supported = bool(torch.cuda.is_bf16_supported())
    return gpu_name, architecture, bf16_supported


def probe_rocm(*, require_triton: bool = True) -> RocmCapability:
    """Inspect whether this process can launch the custom ROCm/Triton path."""
    triton_available = _module_available("triton")
    if not _module_available("torch"):
        return RocmCapability(
            available=False,
            reason="PyTorch is not installed",
            torch_version=None,
            hip_version=None,
            triton_available=triton_available,
            accelerator_available=False,
            gpu_name=None,
            architecture=None,
            bf16_supported=False,
        )

    torch = import_module("torch")
    torch_version = str(torch.__version__)
    hip_version = getattr(torch.version, "hip", None)
    accelerator_available = bool(torch.cuda.is_available())
    gpu_name, architecture, bf16_supported = _device_details(torch)

    reason = None
    if hip_version is None:
        reason = "the installed PyTorch build does not include ROCm/HIP"
    elif not accelerator_available:
        reason = "PyTorch ROCm cannot access an AMD accelerator"
    elif require_triton and not triton_available:
        reason = "Triton is not installed"

    return RocmCapability(
        available=reason is None,
        reason=reason,
        torch_version=torch_version,
        hip_version=str(hip_version) if hip_version is not None else None,
        triton_available=triton_available,
        accelerator_available=accelerator_available,
        gpu_name=gpu_name,
        architecture=architecture,
        bf16_supported=bf16_supported,
    )


def require_rocm(*, require_triton: bool = True) -> RocmCapability:
    """Return ROCm details or raise a single actionable environment error."""
    capability = probe_rocm(require_triton=require_triton)
    if not capability.available:
        raise RuntimeError(capability.reason or "ROCm is unavailable")
    return capability


def accelerator_device() -> Any:
    """Return the current PyTorch HIP device after validating the runtime."""
    require_rocm(require_triton=False)
    torch = import_module("torch")
    return torch.device("cuda", torch.cuda.current_device())


def synchronize() -> None:
    """Synchronize the active HIP stream through PyTorch's CUDA-compatible API."""
    require_rocm(require_triton=False)
    torch = import_module("torch")
    torch.cuda.synchronize()


def reset_peak_memory_stats() -> None:
    require_rocm(require_triton=False)
    torch = import_module("torch")
    torch.cuda.reset_peak_memory_stats()


def peak_memory_stats() -> tuple[int, int]:
    require_rocm(require_triton=False)
    torch = import_module("torch")
    return torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()


def timing_event() -> Any:
    require_rocm(require_triton=False)
    torch = import_module("torch")
    return torch.cuda.Event(enable_timing=True)


def empty_cache() -> None:
    require_rocm(require_triton=False)
    torch = import_module("torch")
    torch.cuda.empty_cache()