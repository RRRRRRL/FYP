import csv
import importlib.util
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

import torch
import torch.nn.functional as functional


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
    mean_ms: float | None
    median_ms: float | None
    min_ms: float | None
    max_ms: float | None
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


class CustomTritonBaseline(AttentionBaseline):
    name = "custom_triton"

    def status(self, config: BenchmarkConfig) -> BaselineStatus:
        if not torch.cuda.is_available():
            return BaselineStatus(self.name, False, "CUDA is unavailable")
        if importlib.util.find_spec("triton") is None:
            return BaselineStatus(self.name, False, "Triton is not installed")
        if config.dtype not in (torch.float16, torch.bfloat16):
            return BaselineStatus(
                self.name, False, "custom Triton supports FP16 and BF16 only"
            )
        if config.dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            return BaselineStatus(self.name, False, "GPU does not support BF16")
        if config.head_dim not in (16, 32, 64, 128):
            return BaselineStatus(
                self.name, False, "unsupported custom Triton head dimension"
            )
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


class PyTorchMathBaseline(AttentionBaseline):
    name = "pytorch_math"

    def status(self, config: BenchmarkConfig) -> BaselineStatus:
        if not torch.cuda.is_available():
            return BaselineStatus(self.name, False, "CUDA is unavailable")
        return BaselineStatus(self.name, True)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.MATH):
            return functional.scaled_dot_product_attention(
                query, key, value, is_causal=causal
            )


class FlashAttention2Baseline(AttentionBaseline):
    name = "flash_attention_2"

    def status(self, config: BenchmarkConfig) -> BaselineStatus:
        if not torch.cuda.is_available():
            return BaselineStatus(self.name, False, "CUDA is unavailable")
        if importlib.util.find_spec("flash_attn") is None:
            return BaselineStatus(
                self.name, False, "the flash-attn package is not installed"
            )
        if config.dtype not in (torch.float16, torch.bfloat16):
            return BaselineStatus(
                self.name, False, "FlashAttention-2 requires FP16 or BF16"
            )
        if config.dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            return BaselineStatus(self.name, False, "GPU does not support BF16")
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


class FlexAttentionBaseline(AttentionBaseline):
    name = "flex_attention"

    def __init__(self) -> None:
        self._block_masks: dict[tuple[int, int, int, str], object] = {}

    def status(self, config: BenchmarkConfig) -> BaselineStatus:
        if not torch.cuda.is_available():
            return BaselineStatus(self.name, False, "CUDA is unavailable")
        if importlib.util.find_spec("torch.nn.attention.flex_attention") is None:
            return BaselineStatus(
                self.name, False, "FlexAttention is unavailable in this PyTorch"
            )
        return BaselineStatus(self.name, True)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        from torch.nn.attention.flex_attention import (
            create_block_mask,
            flex_attention,
        )

        block_mask = None
        if causal:

            def causal_mask(
                batch: torch.Tensor,
                head: torch.Tensor,
                query_index: torch.Tensor,
                key_index: torch.Tensor,
            ) -> torch.Tensor:
                del batch, head
                return query_index >= key_index

            cache_key = (
                query.shape[0],
                query.shape[1],
                query.shape[2],
                str(query.device),
            )
            block_mask = self._block_masks.get(cache_key)
            if block_mask is None:
                block_mask = create_block_mask(
                    causal_mask,
                    B=query.shape[0],
                    H=query.shape[1],
                    Q_LEN=query.shape[2],
                    KV_LEN=key.shape[2],
                    device=query.device,
                )
                self._block_masks[cache_key] = block_mask
        return flex_attention(query, key, value, block_mask=block_mask)


BASELINES: dict[str, AttentionBaseline] = {
    baseline.name: baseline
    for baseline in (
        CustomTritonBaseline(),
        PyTorchMathBaseline(),
        FlashAttention2Baseline(),
        FlexAttentionBaseline(),
    )
}


def attention_flops(config: BenchmarkConfig, pass_name: PassName) -> int:
    """Return the conventional attention FLOP estimate used for comparisons.

    Forward counts QK^T and PV as four operations per score/head dimension.
    Backward is conventionally estimated as 2.5 times forward. Causal attention
    halves the score pairs while retaining the same convention.
    """
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


def _empty_result(
    baseline: str,
    config: BenchmarkConfig,
    pass_name: PassName,
    status: str,
    reason: str,
) -> BenchmarkResult:
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
        mean_ms=None,
        median_ms=None,
        min_ms=None,
        max_ms=None,
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
    """Measure one baseline with inputs allocated outside the timed region."""
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
        query = torch.randn(
            config.shape,
            device="cuda",
            dtype=config.dtype,
            requires_grad=requires_grad,
        )
        key = torch.randn_like(query, requires_grad=requires_grad)
        value = torch.randn_like(query, requires_grad=requires_grad)
        operation = operation_for(
            baseline, config, query, key, value, pass_name
        )

        for _ in range(warmup):
            operation()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        timings_ms: list[float] = []
        for _ in range(repetitions):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            operation()
            end.record()
            torch.cuda.synchronize()
            timings_ms.append(start.elapsed_time(end))

        median_ms = statistics.median(timings_ms)
        tflops = attention_flops(config, pass_name) / (median_ms * 1e9)
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
            mean_ms=statistics.fmean(timings_ms),
            median_ms=median_ms,
            min_ms=min(timings_ms),
            max_ms=max(timings_ms),
            tflops=tflops,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        return _empty_result(
            baseline.name, config, pass_name, "oom", str(error).splitlines()[0]
        )
    except (ImportError, RuntimeError, NotImplementedError) as error:
        return _empty_result(
            baseline.name,
            config,
            pass_name,
            "error",
            f"{type(error).__name__}: {str(error).splitlines()[0]}",
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