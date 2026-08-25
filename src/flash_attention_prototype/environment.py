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


@dataclass(frozen=True)
class EnvironmentReport:
    platform: str
    python: str
    torch: str | None
    torch_cuda: str | None
    cuda_available: bool
    gpu_name: str | None
    compute_capability: str | None
    driver_version: str | None
    bf16_supported: bool
    triton: str | None
    flash_attention_2_available: bool
    flex_attention_available: bool


def _module_version(module_name: str) -> str | None:
    if importlib.util.find_spec(module_name) is None:
        return None
    try:
        return version(module_name)
    except PackageNotFoundError:
        return "installed"


def _nvidia_driver_version() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    versions = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    return ", ".join(sorted(versions)) or None


def collect_environment() -> EnvironmentReport:
    """Collect software, GPU, and optional attention-backend capabilities."""
    if importlib.util.find_spec("torch") is None:
        return EnvironmentReport(
            platform=platform.platform(),
            python=sys.version.split()[0],
            torch=None,
            torch_cuda=None,
            cuda_available=False,
            gpu_name=None,
            compute_capability=None,
            driver_version=_nvidia_driver_version(),
            bf16_supported=False,
            triton=_module_version("triton"),
            flash_attention_2_available=(
                importlib.util.find_spec("flash_attn") is not None
            ),
            flex_attention_available=False,
        )

    torch = import_module("torch")
    cuda_available = torch.cuda.is_available()
    gpu_name = None
    compute_capability = None
    bf16_supported = False
    if cuda_available:
        device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device)
        major, minor = torch.cuda.get_device_capability(device)
        compute_capability = f"{major}.{minor}"
        bf16_supported = torch.cuda.is_bf16_supported()

    flex_attention_available = (
        importlib.util.find_spec("torch.nn.attention.flex_attention") is not None
    )

    return EnvironmentReport(
        platform=platform.platform(),
        python=sys.version.split()[0],
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        compute_capability=compute_capability,
        driver_version=_nvidia_driver_version(),
        bf16_supported=bf16_supported,
        triton=_module_version("triton"),
        flash_attention_2_available=(
            importlib.util.find_spec("flash_attn") is not None
        ),
        flex_attention_available=flex_attention_available,
    )


def environment_dict() -> dict[str, Any]:
    return asdict(collect_environment())


def write_environment_json(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(environment_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Report the FlashAttention prototype environment"
    )
    parser.add_argument("--json", type=Path, help="also write the report as JSON")
    args = parser.parse_args()
    report = environment_dict()
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json is not None:
        write_environment_json(args.json)


if __name__ == "__main__":
    main()