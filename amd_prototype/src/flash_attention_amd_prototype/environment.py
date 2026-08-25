import importlib.util
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .backend import probe_rocm


@dataclass(frozen=True)
class EnvironmentReport:
    platform: str
    python: str
    torch: str | None
    torch_cuda: str | None
    torch_hip: str | None
    runtime_backend: str
    accelerator_available: bool
    rocm_available: bool
    gpu_name: str | None
    gpu_architecture: str | None
    bf16_supported: bool
    triton: str | None
    custom_triton_available: bool
    custom_triton_reason: str | None
    flash_sdpa_available: bool | None
    ck_sdpa_available: bool | None
    preferred_rocm_fa_library: str | None
    flash_attention_2_available: bool
    hipcc: str | None


def _module_version(module_name: str) -> str | None:
    if importlib.util.find_spec(module_name) is None:
        return None
    try:
        return version(module_name)
    except PackageNotFoundError:
        return "installed"


def _command_version(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]
    return lines[0] if lines else None


def _optional_backend_value(backend: Any, name: str) -> Any:
    value = getattr(backend, name, None)
    if value is None:
        return None
    try:
        return value() if callable(value) else value
    except (RuntimeError, TypeError):
        return None


def _stringify_optional(value: Any) -> str | None:
    if value is None:
        return None
    return getattr(value, "name", None) or str(value)


def collect_environment() -> EnvironmentReport:
    """Collect ROCm, accelerator, and attention-backend capabilities."""
    capability = probe_rocm()
    base = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "triton": _module_version("triton"),
        "custom_triton_available": capability.available,
        "custom_triton_reason": capability.reason,
        "flash_attention_2_available": (
            importlib.util.find_spec("flash_attn") is not None
        ),
        "hipcc": _command_version(["hipcc", "--version"]),
    }
    if importlib.util.find_spec("torch") is None:
        return EnvironmentReport(
            **base,
            torch=capability.torch_version,
            torch_cuda=None,
            torch_hip=None,
            runtime_backend="cpu",
            accelerator_available=False,
            rocm_available=False,
            gpu_name=None,
            gpu_architecture=None,
            bf16_supported=capability.bf16_supported,
            flash_sdpa_available=None,
            ck_sdpa_available=None,
            preferred_rocm_fa_library=None,
        )

    torch = import_module("torch")
    torch_hip = getattr(torch.version, "hip", None)
    rocm_available = torch_hip is not None
    accelerator_available = capability.accelerator_available
    runtime_backend = "rocm" if rocm_available else (
        "cuda" if accelerator_available else "cpu"
    )
    gpu_name = capability.gpu_name
    gpu_architecture = capability.architecture
    bf16_supported = capability.bf16_supported
    if accelerator_available:
        device = torch.cuda.current_device()
        if gpu_architecture is None and not rocm_available:
            major, minor = torch.cuda.get_device_capability(device)
            gpu_architecture = f"sm_{major}{minor}"

    cuda_backend = getattr(getattr(torch, "backends", None), "cuda", None)
    flash_sdpa_available = _optional_backend_value(
        cuda_backend, "is_flash_attention_available"
    )
    ck_sdpa_available = _optional_backend_value(cuda_backend, "is_ck_sdpa_available")
    preferred_rocm_fa_library = _stringify_optional(
        _optional_backend_value(cuda_backend, "preferred_rocm_fa_library")
    )

    return EnvironmentReport(
        **base,
        torch=capability.torch_version,
        torch_cuda=getattr(torch.version, "cuda", None),
        torch_hip=torch_hip,
        runtime_backend=runtime_backend,
        accelerator_available=accelerator_available,
        rocm_available=rocm_available,
        gpu_name=gpu_name,
        gpu_architecture=gpu_architecture,
        bf16_supported=bf16_supported,
        flash_sdpa_available=flash_sdpa_available,
        ck_sdpa_available=ck_sdpa_available,
        preferred_rocm_fa_library=preferred_rocm_fa_library,
    )


def environment_dict() -> dict[str, Any]:
    return asdict(collect_environment())


def write_environment_json(
    path: str | Path,
    report: dict[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            environment_dict() if report is None else report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Report the AMD FlashAttention prototype environment"
    )
    parser.add_argument("--json", type=Path, help="also write the report as JSON")
    parser.add_argument(
        "--probe-triton",
        action="store_true",
        help="compile and execute a minimal Triton kernel on the active AMD GPU",
    )
    args = parser.parse_args()
    report = environment_dict()
    if args.probe_triton:
        try:
            from .triton_probe import run_triton_probe

            report["triton_probe"] = run_triton_probe()
        except (ImportError, RuntimeError, ValueError) as error:
            report["triton_probe"] = {
                "success": False,
                "error": f"{type(error).__name__}: {error}",
            }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json is not None:
        write_environment_json(args.json, report)


if __name__ == "__main__":
    main()