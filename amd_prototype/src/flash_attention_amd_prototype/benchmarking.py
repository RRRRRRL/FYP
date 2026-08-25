from __future__ import annotations

import csv
import importlib.util
import json
import math
import statistics
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

import torch
import torch.nn.functional as functional

from .backend import (
    accelerator_device,
    empty_cache,
    peak_memory_stats,
    probe_rocm,
    reset_peak_memory_stats,
    synchronize,
    timing_event,
)


PassName = Literal["forward", "backward"]


@dataclass(frozen=True)
class BenchmarkConfig:
    batch_size: int
    num_heads: int
    sequence_length: int
    head_dim: int
    dtype: torch.dtype
    causal: bool

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (
            self.batch_size,
            self.num_heads,
            self.sequence_length,
            self.head_dim,
        )


@dataclass(frozen=True)
class BaselineStatus:
    name: str
    available: bool
    reason: str | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    baseline: str
    pass_name: PassName
    status: str
    reason: str | None
    batch_size: int
    num_heads: int
    sequence_length: int
    head_dim: int
    dtype: str
    causal: bool
    runtime_backend: str
    gpu_architecture: str | None
    launch_config: str | None
    first_iteration_ms: float | None
    mean_ms: float | None
    median_ms: float | None
    p90_ms: float | None
    min_ms: float | None
    max_ms: float | None
    stdev_ms: float | None
    tflops: float | None
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class AttentionBaseline:
    name: str

    def status(self, config: BenchmarkConfig) -> BaselineStatus:
        raise NotImplementedError

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        raise NotImplementedError

    def launch_metadata(self, config: BenchmarkConfig) -> str | None:
        del config
        return None


def _rocm_status(
    name: str,
    *,
    require_triton: bool = False,
) -> BaselineStatus:
    capability = probe_rocm(require_triton=require_triton)
    return BaselineStatus(name, capability.available, capability.reason)


def _dtype_status(name: str, config: BenchmarkConfig) -> BaselineStatus | None:
    if config.dtype not in (torch.float16, torch.bfloat16):
        return BaselineStatus(name, False, "FP16 and BF16 are supported")
    capability = probe_rocm(require_triton=False)
    if config.dtype == torch.bfloat16 and not capability.bf16_supported:
        return BaselineStatus(name, False, "the active GPU does not support BF16")
    return None


def _backend_value(name: str) -> Any:
    cuda_backend = getattr(getattr(torch, "backends", None), "cuda", None)
    value = getattr(cuda_backend, name, None)
    if value is None:
        return None
    try:
        return value() if callable(value) else value
    except (RuntimeError, TypeError):
        return None


@contextmanager
def _preferred_rocm_flash_library(library: str) -> Iterator[None]:
    cuda_backend = getattr(getattr(torch, "backends", None), "cuda", None)
    selector = getattr(cuda_backend, "preferred_rocm_fa_library", None)
    if not callable(selector):
        raise RuntimeError(
            "this PyTorch build cannot select a ROCm FlashAttention library"
        )
    previous = selector()
    selector(library)
    try:
        yield
    finally:
        selector(previous)


def _forced_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool,
    backend_name: str,
) -> torch.Tensor:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    backend = getattr(SDPBackend, backend_name)
    with sdpa_kernel([backend]):
        return functional.scaled_dot_product_attention(
            query, key, value, is_causal=causal
        )


class CustomTritonBaseline(AttentionBaseline):
    name = "custom_triton_amd"

    def status(self, config: BenchmarkConfig) -> BaselineStatus:
        runtime_status = _rocm_status(self.name, require_triton=True)
        if not runtime_status.available:
            return runtime_status
        dtype_status = _dtype_status(self.name, config)
        if dtype_status is not None:
            return dtype_status
        if config.head_dim not in (16, 32, 64, 128):
            return BaselineStatus(self.name, False, "unsupported head dimension")
        return BaselineStatus(self.name, True)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        from .kernel import flash_attention_forward

        return flash_attention_forward(query, key, value, causal=causal)

    def launch_metadata(self, config: BenchmarkConfig) -> str | None:
        from .kernel import current_launch_config

        launch = current_launch_config(config.head_dim)
        return json.dumps(launch.to_dict(), sort_keys=True)


class PyTorchMathBaseline(AttentionBaseline):
    name = "pytorch_math"

    def status(self, config: BenchmarkConfig) -> BaselineStatus:
        del config
        return _rocm_status(self.name)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        return _forced_sdpa(query, key, value, causal, "MATH")


class RocmFlashBaseline(AttentionBaseline):
    def __init__(self, library: Literal["ck", "aotriton"]) -> None:
        self.library = library
        self.name = f"rocm_{library}"

    def status(self, config: BenchmarkConfig) -> BaselineStatus:
        runtime_status = _rocm_status(self.name)
        if not runtime_status.available:
            return runtime_status
        dtype_status = _dtype_status(self.name, config)
        if dtype_status is not None:
            return dtype_status
        cuda_backend = getattr(getattr(torch, "backends", None), "cuda", None)
        if not callable(getattr(cuda_backend, "preferred_rocm_fa_library", None)):
            return BaselineStatus(
                self.name,
                False,
                "PyTorch does not expose preferred_rocm_fa_library",
            )
        if self.library == "ck" and not bool(_backend_value("is_ck_sdpa_available")):
            return BaselineStatus(self.name, False, "CK SDPA is unavailable")
        if self.library == "aotriton" and not bool(
            _backend_value("is_flash_attention_available")
        ):
            return BaselineStatus(self.name, False, "flash SDPA is unavailable")
        return BaselineStatus(self.name, True)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        with _preferred_rocm_flash_library(self.library):
            return _forced_sdpa(
                query, key, value, causal, "FLASH_ATTENTION"
            )

    def launch_metadata(self, config: BenchmarkConfig) -> str | None:
        del config
        return json.dumps({"rocm_fa_library": self.library})


class FlashAttention2RocmBaseline(AttentionBaseline):
    name = "flash_attention_2_rocm"

    def status(self, config: BenchmarkConfig) -> BaselineStatus:
        runtime_status = _rocm_status(self.name)
        if not runtime_status.available:
            return runtime_status
        dtype_status = _dtype_status(self.name, config)
        if dtype_status is not None:
            return dtype_status
        if importlib.util.find_spec("flash_attn") is None:
            return BaselineStatus(
                self.name, False, "the flash-attn package is not installed"
            )
        return BaselineStatus(self.name, True)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        from flash_attn import flash_attn_func

        query_bshd = query.transpose(1, 2)
        key_bshd = key.transpose(1, 2)
        value_bshd = value.transpose(1, 2)
        return flash_attn_func(
            query_bshd,
            key_bshd,
            value_bshd,
            causal=causal,
        ).transpose(1, 2)


BASELINES: dict[str, AttentionBaseline] = {
    baseline.name: baseline
    for baseline in (
        CustomTritonBaseline(),
        PyTorchMathBaseline(),
        RocmFlashBaseline("ck"),
        RocmFlashBaseline("aotriton"),
        FlashAttention2RocmBaseline(),
    )
}


def attention_flops(config: BenchmarkConfig, pass_name: PassName) -> int:
    score_pairs = config.sequence_length * config.sequence_length
    if config.causal:
        score_pairs = config.sequence_length * (config.sequence_length + 1) // 2
    forward_flops = (
        4
        * config.batch_size
        * config.num_heads
        * score_pairs
        * config.head_dim
    )
    return forward_flops if pass_name == "forward" else 5 * forward_flops // 2


def operation_for(
    baseline: AttentionBaseline,
    config: BenchmarkConfig,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    pass_name: PassName,
) -> Callable[[], torch.Tensor | tuple[torch.Tensor, ...]]:
    if pass_name == "forward":

        def forward_operation() -> torch.Tensor:
            with torch.no_grad():
                return baseline.forward(query, key, value, config.causal)

        return forward_operation

    output = baseline.forward(query, key, value, config.causal)
    grad_output = torch.randn_like(output)

    def backward_operation() -> tuple[torch.Tensor, ...]:
        return torch.autograd.grad(
            output,
            (query, key, value),
            grad_outputs=grad_output,
            retain_graph=True,
        )

    return backward_operation


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _empty_result(
    baseline: str,
    config: BenchmarkConfig,
    pass_name: PassName,
    status: str,
    reason: str,
) -> BenchmarkResult:
    capability = probe_rocm(require_triton=False)
    return BenchmarkResult(
        baseline=baseline,
        pass_name=pass_name,
        status=status,
        reason=reason,
        batch_size=config.batch_size,
        num_heads=config.num_heads,
        sequence_length=config.sequence_length,
        head_dim=config.head_dim,
        dtype=str(config.dtype).removeprefix("torch."),
        causal=config.causal,
        runtime_backend="rocm" if capability.hip_version else "unavailable",
        gpu_architecture=capability.architecture,
        launch_config=None,
        first_iteration_ms=None,
        mean_ms=None,
        median_ms=None,
        p90_ms=None,
        min_ms=None,
        max_ms=None,
        stdev_ms=None,
        tflops=None,
        peak_allocated_bytes=None,
        peak_reserved_bytes=None,
    )


def benchmark_baseline(
    baseline: AttentionBaseline,
    config: BenchmarkConfig,
    pass_name: PassName,
    *,
    warmup: int,
    repetitions: int,
) -> BenchmarkResult:
    """Measure one ROCm baseline without permitting backend fallback."""
    if warmup < 0 or repetitions <= 0:
        raise ValueError("warmup must be non-negative and repetitions positive")
    status = baseline.status(config)
    if not status.available:
        return _empty_result(
            baseline.name,
            config,
            pass_name,
            "unavailable",
            status.reason or "baseline is unavailable",
        )

    try:
        requires_grad = pass_name == "backward"
        device = accelerator_device()
        query = torch.randn(
            config.shape,
            device=device,
            dtype=config.dtype,
            requires_grad=requires_grad,
        )
        key = torch.randn_like(query, requires_grad=requires_grad)
        value = torch.randn_like(query, requires_grad=requires_grad)
        operation = operation_for(
            baseline, config, query, key, value, pass_name
        )

        synchronize()
        first_start = time.perf_counter()
        operation()
        synchronize()
        first_iteration_ms = (time.perf_counter() - first_start) * 1000.0

        for _ in range(warmup):
            operation()
        synchronize()
        reset_peak_memory_stats()

        timings_ms: list[float] = []
        for _ in range(repetitions):
            start = timing_event()
            end = timing_event()
            synchronize()
            start.record()
            operation()
            end.record()
            synchronize()
            timings_ms.append(float(start.elapsed_time(end)))

        median_ms = statistics.median(timings_ms)
        allocated, reserved = peak_memory_stats()
        capability = probe_rocm(require_triton=False)
        return BenchmarkResult(
            baseline=baseline.name,
            pass_name=pass_name,
            status="ok",
            reason=None,
            batch_size=config.batch_size,
            num_heads=config.num_heads,
            sequence_length=config.sequence_length,
            head_dim=config.head_dim,
            dtype=str(config.dtype).removeprefix("torch."),
            causal=config.causal,
            runtime_backend="rocm",
            gpu_architecture=capability.architecture,
            launch_config=baseline.launch_metadata(config),
            first_iteration_ms=first_iteration_ms,
            mean_ms=statistics.fmean(timings_ms),
            median_ms=median_ms,
            p90_ms=_percentile(timings_ms, 0.90),
            min_ms=min(timings_ms),
            max_ms=max(timings_ms),
            stdev_ms=(
                statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0
            ),
            tflops=attention_flops(config, pass_name) / (median_ms * 1e9),
            peak_allocated_bytes=allocated,
            peak_reserved_bytes=reserved,
        )
    except (
        ImportError,
        RuntimeError,
        NotImplementedError,
        TypeError,
        ValueError,
    ) as error:
        message = str(error).splitlines()[0]
        if "out of memory" in message.lower():
            empty_cache()
            return _empty_result(
                baseline.name, config, pass_name, "oom", message
            )
        return _empty_result(
            baseline.name,
            config,
            pass_name,
            "error",
            f"{type(error).__name__}: {message}",
        )


def write_results(
    results: list[BenchmarkResult],
    output_directory: str | Path,
    metadata: dict[str, object],
) -> tuple[Path, Path]:
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    rows = [result.to_dict() for result in results]
    csv_path = destination / "results.csv"
    json_path = destination / "manifest.json"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {"metadata": metadata, "results": rows},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path