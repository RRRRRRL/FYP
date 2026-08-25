import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Iterator

import torch
import triton
import triton.language as tl

from .backend import require_rocm


@dataclass(frozen=True)
class AmdLaunchConfig:
    profile: str
    architecture: str
    block_m: int
    block_n: int
    num_warps: int
    num_stages: int

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


_LAUNCH_OVERRIDE: ContextVar[AmdLaunchConfig | None] = ContextVar(
    "amd_launch_override", default=None
)


def _architecture_family(architecture: str | None) -> str:
    if not architecture:
        return "unknown"
    return architecture.split(":", maxsplit=1)[0].lower()


def select_launch_config(
    head_dim: int,
    architecture: str | None = None,
) -> AmdLaunchConfig:
    """Select conservative pre-tuning defaults for an AMD architecture."""
    if head_dim not in (16, 32, 64, 128):
        raise ValueError("head_dim must be one of 16, 32, 64, or 128")
    family = _architecture_family(architecture)
    if family == "gfx90a":
        profile = "mi200-conservative"
    elif family in {"gfx940", "gfx941", "gfx942"}:
        profile = "mi300-conservative"
    else:
        profile = "generic-amd-conservative"

    block_size = 32 if head_dim == 128 else 64
    return AmdLaunchConfig(
        profile=profile,
        architecture=family,
        block_m=block_size,
        block_n=block_size,
        num_warps=4,
        num_stages=2,
    )


def candidate_launch_configs(
    head_dim: int,
    architecture: str | None,
) -> tuple[AmdLaunchConfig, ...]:
    """Return a bounded candidate set for hardware-time measurement."""
    default = select_launch_config(head_dim, architecture)
    family = _architecture_family(architecture)
    if head_dim == 128:
        shapes = ((32, 32), (32, 64), (64, 32))
    else:
        shapes = ((64, 64), (64, 32), (32, 64))
    warp_counts = (4, 8) if family in {"gfx940", "gfx941", "gfx942"} else (4,)
    stage_counts = (1, 2)
    candidates = [default]
    seen = {
        (
            default.block_m,
            default.block_n,
            default.num_warps,
            default.num_stages,
        )
    }
    for block_m, block_n in shapes:
        for num_warps in warp_counts:
            for num_stages in stage_counts:
                parameters = (block_m, block_n, num_warps, num_stages)
                if parameters in seen:
                    continue
                candidate = AmdLaunchConfig(
                    profile=(
                        f"tune-{family}-m{block_m}-n{block_n}-"
                        f"w{num_warps}-s{num_stages}"
                    ),
                    architecture=family,
                    block_m=block_m,
                    block_n=block_n,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
                candidates.append(candidate)
                seen.add(parameters)
    return tuple(candidates)


@contextmanager
def use_launch_config(config: AmdLaunchConfig) -> Iterator[None]:
    if config.block_m <= 0 or config.block_n <= 0:
        raise ValueError("launch block dimensions must be positive")
    if config.num_warps not in (1, 2, 4, 8):
        raise ValueError("num_warps must be one of 1, 2, 4, or 8")
    if config.num_stages <= 0:
        raise ValueError("num_stages must be positive")
    token = _LAUNCH_OVERRIDE.set(config)
    try:
        yield
    finally:
        _LAUNCH_OVERRIDE.reset(token)


def current_launch_config(head_dim: int) -> AmdLaunchConfig:
    override = _LAUNCH_OVERRIDE.get()
    if override is not None:
        return override
    capability = require_rocm()
    return select_launch_config(head_dim, capability.architecture)


@triton.jit
def _flash_attention_forward_kernel(
    query,
    key,
    value,
    output,
    logsumexp,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_on: tl.constexpr,
    num_heads: tl.constexpr,
    sequence_length: tl.constexpr,
    head_dim: tl.constexpr,
    softmax_scale: tl.constexpr,
    causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // num_heads
    head = batch_head % num_heads

    query_rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    key_rows = tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    query_offsets = (
        batch * stride_qb
        + head * stride_qh
        + query_rows[:, None] * stride_qn
        + dims[None, :]
    )
    query_tile = tl.load(
        query + query_offsets,
        mask=(query_rows[:, None] < sequence_length) & (dims[None, :] < head_dim),
        other=0.0,
    )

    row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    qk_scale = softmax_scale * 1.4426950408889634

    for key_start in range(0, sequence_length, BLOCK_N):
        current_keys = key_start + key_rows
        key_offsets = (
            batch * stride_kb
            + head * stride_kh
            + current_keys[:, None] * stride_kn
            + dims[None, :]
        )
        value_offsets = (
            batch * stride_vb
            + head * stride_vh
            + current_keys[:, None] * stride_vn
            + dims[None, :]
        )
        key_tile = tl.load(
            key + key_offsets,
            mask=(current_keys[:, None] < sequence_length)
            & (dims[None, :] < head_dim),
            other=0.0,
        )
        value_tile = tl.load(
            value + value_offsets,
            mask=(current_keys[:, None] < sequence_length)
            & (dims[None, :] < head_dim),
            other=0.0,
        )

        scores = tl.dot(query_tile, tl.trans(key_tile)) * qk_scale
        score_mask = current_keys[None, :] < sequence_length
        if causal:
            score_mask &= query_rows[:, None] >= current_keys[None, :]
        scores = tl.where(score_mask, scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        next_max = tl.maximum(row_max, block_max)
        correction = tl.exp2(row_max - next_max)
        probabilities = tl.exp2(scores - next_max[:, None])
        accumulator *= correction[:, None]
        accumulator += tl.dot(probabilities.to(value_tile.dtype), value_tile)
        row_sum = row_sum * correction + tl.sum(probabilities, axis=1)
        row_max = next_max

    output_tile = accumulator / row_sum[:, None]
    output_offsets = (
        batch * stride_ob
        + head * stride_oh
        + query_rows[:, None] * stride_on
        + dims[None, :]
    )
    tl.store(
        output + output_offsets,
        output_tile,
        mask=(query_rows[:, None] < sequence_length) & (dims[None, :] < head_dim),
    )
    tl.store(
        logsumexp + batch_head * sequence_length + query_rows,
        row_max + tl.log2(row_sum),
        mask=query_rows < sequence_length,
    )


@triton.jit
def _attention_delta_kernel(
    output,
    grad_output,
    delta,
    sequence_length: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, BLOCK_D)
    offsets = (
        batch_head * sequence_length * head_dim
        + rows[:, None] * head_dim
        + dims[None, :]
    )
    mask = (rows[:, None] < sequence_length) & (dims[None, :] < head_dim)
    output_tile = tl.load(output + offsets, mask=mask, other=0.0).to(tl.float32)
    grad_output_tile = tl.load(
        grad_output + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    tl.store(
        delta + batch_head * sequence_length + rows,
        tl.sum(output_tile * grad_output_tile, axis=1),
        mask=rows < sequence_length,
    )


@triton.jit
def _flash_attention_backward_query_kernel(
    query,
    key,
    value,
    grad_output,
    logsumexp,
    delta,
    grad_query,
    sequence_length: tl.constexpr,
    head_dim: tl.constexpr,
    softmax_scale: tl.constexpr,
    causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    query_rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    key_rows = tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    base = batch_head * sequence_length * head_dim
    query_offsets = base + query_rows[:, None] * head_dim + dims[None, :]
    query_mask = (query_rows[:, None] < sequence_length) & (
        dims[None, :] < head_dim
    )
    query_tile = tl.load(query + query_offsets, mask=query_mask, other=0.0)
    grad_output_tile = tl.load(
        grad_output + query_offsets, mask=query_mask, other=0.0
    )
    row_offsets = batch_head * sequence_length + query_rows
    row_lse = tl.load(
        logsumexp + row_offsets, mask=query_rows < sequence_length, other=0.0
    )
    row_delta = tl.load(
        delta + row_offsets, mask=query_rows < sequence_length, other=0.0
    )
    grad_query_accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    qk_scale = softmax_scale * 1.4426950408889634

    for key_start in range(0, sequence_length, BLOCK_N):
        current_keys = key_start + key_rows
        key_offsets = base + current_keys[:, None] * head_dim + dims[None, :]
        key_mask = (current_keys[:, None] < sequence_length) & (
            dims[None, :] < head_dim
        )
        key_tile = tl.load(key + key_offsets, mask=key_mask, other=0.0)
        value_tile = tl.load(value + key_offsets, mask=key_mask, other=0.0)
        scores = tl.dot(query_tile, tl.trans(key_tile)) * qk_scale
        score_mask = (query_rows[:, None] < sequence_length) & (
            current_keys[None, :] < sequence_length
        )
        if causal:
            score_mask &= query_rows[:, None] >= current_keys[None, :]
        probabilities = tl.where(
            score_mask, tl.exp2(scores - row_lse[:, None]), 0.0
        )
        grad_probabilities = tl.dot(grad_output_tile, tl.trans(value_tile))
        grad_scores = probabilities * (grad_probabilities - row_delta[:, None])
        grad_query_accumulator += tl.dot(
            grad_scores.to(key_tile.dtype), key_tile
        )

    tl.store(
        grad_query + query_offsets,
        grad_query_accumulator * softmax_scale,
        mask=query_mask,
    )


@triton.jit
def _flash_attention_backward_key_value_kernel(
    query,
    key,
    value,
    grad_output,
    logsumexp,
    delta,
    grad_key,
    grad_value,
    sequence_length: tl.constexpr,
    head_dim: tl.constexpr,
    softmax_scale: tl.constexpr,
    causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    key_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    query_rows = tl.arange(0, BLOCK_M)
    key_rows = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    base = batch_head * sequence_length * head_dim
    key_offsets = base + key_rows[:, None] * head_dim + dims[None, :]
    key_mask = (key_rows[:, None] < sequence_length) & (dims[None, :] < head_dim)
    key_tile = tl.load(key + key_offsets, mask=key_mask, other=0.0)
    value_tile = tl.load(value + key_offsets, mask=key_mask, other=0.0)
    grad_key_accumulator = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    grad_value_accumulator = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    qk_scale = softmax_scale * 1.4426950408889634

    for query_start in range(0, sequence_length, BLOCK_M):
        current_queries = query_start + query_rows
        query_offsets = base + current_queries[:, None] * head_dim + dims[None, :]
        query_mask = (current_queries[:, None] < sequence_length) & (
            dims[None, :] < head_dim
        )
        query_tile = tl.load(query + query_offsets, mask=query_mask, other=0.0)
        grad_output_tile = tl.load(
            grad_output + query_offsets, mask=query_mask, other=0.0
        )
        row_offsets = batch_head * sequence_length + current_queries
        row_lse = tl.load(
            logsumexp + row_offsets,
            mask=current_queries < sequence_length,
            other=0.0,
        )
        row_delta = tl.load(
            delta + row_offsets,
            mask=current_queries < sequence_length,
            other=0.0,
        )
        scores = tl.dot(query_tile, tl.trans(key_tile)) * qk_scale
        score_mask = (current_queries[:, None] < sequence_length) & (
            key_rows[None, :] < sequence_length
        )
        if causal:
            score_mask &= current_queries[:, None] >= key_rows[None, :]
        probabilities = tl.where(
            score_mask, tl.exp2(scores - row_lse[:, None]), 0.0
        )
        grad_probabilities = tl.dot(grad_output_tile, tl.trans(value_tile))
        grad_scores = probabilities * (grad_probabilities - row_delta[:, None])
        grad_value_accumulator += tl.dot(
            tl.trans(probabilities.to(grad_output_tile.dtype)), grad_output_tile
        )
        grad_key_accumulator += tl.dot(
            tl.trans(grad_scores.to(query_tile.dtype)), query_tile
        )

    tl.store(
        grad_key + key_offsets,
        grad_key_accumulator * softmax_scale,
        mask=key_mask,
    )
    tl.store(grad_value + key_offsets, grad_value_accumulator, mask=key_mask)


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    require_rocm()
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape [B, H, N, D]")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key, and value must have identical shapes")
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise ValueError("query, key, and value must be ROCm/HIP tensors")
    if query.device != key.device or query.device != value.device:
        raise ValueError("query, key, and value must be on the same HIP device")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("query, key, and value must have the same dtype")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("only torch.float16 and torch.bfloat16 are supported")
    if query.stride(-1) != 1 or key.stride(-1) != 1 or value.stride(-1) != 1:
        raise ValueError("the head dimension must be contiguous")


def _forward_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_heads, sequence_length, head_dim = query.shape
    output = torch.empty_like(query, memory_format=torch.contiguous_format)
    logsumexp = torch.empty(
        (batch_size, num_heads, sequence_length),
        device=query.device,
        dtype=torch.float32,
    )
    if output.numel() == 0:
        return output, logsumexp

    launch = current_launch_config(head_dim)
    grid = (
        triton.cdiv(sequence_length, launch.block_m),
        batch_size * num_heads,
    )
    _flash_attention_forward_kernel[grid](
        query,
        key,
        value,
        output,
        logsumexp,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        num_heads,
        sequence_length,
        head_dim,
        softmax_scale,
        causal,
        BLOCK_M=launch.block_m,
        BLOCK_N=launch.block_n,
        BLOCK_D=head_dim,
        num_warps=launch.num_warps,
        num_stages=launch.num_stages,
    )
    return output, logsumexp


def _backward_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    logsumexp: torch.Tensor,
    grad_output: torch.Tensor,
    causal: bool,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, num_heads, sequence_length, head_dim = query.shape
    grad_query = torch.empty_like(query, memory_format=torch.contiguous_format)
    grad_key = torch.empty_like(key, memory_format=torch.contiguous_format)
    grad_value = torch.empty_like(value, memory_format=torch.contiguous_format)
    if grad_query.numel() == 0:
        return grad_query, grad_key, grad_value

    contiguous_query = query.contiguous()
    contiguous_key = key.contiguous()
    contiguous_value = value.contiguous()
    contiguous_output = output.contiguous()
    grad_output = grad_output.contiguous()
    delta = torch.empty(
        (batch_size, num_heads, sequence_length),
        device=query.device,
        dtype=torch.float32,
    )
    launch = current_launch_config(head_dim)
    batch_heads = batch_size * num_heads
    query_grid = (triton.cdiv(sequence_length, launch.block_m), batch_heads)
    key_grid = (triton.cdiv(sequence_length, launch.block_n), batch_heads)

    _attention_delta_kernel[query_grid](
        contiguous_output,
        grad_output,
        delta,
        sequence_length,
        head_dim,
        BLOCK_M=launch.block_m,
        BLOCK_D=head_dim,
        num_warps=launch.num_warps,
    )
    _flash_attention_backward_query_kernel[query_grid](
        contiguous_query,
        contiguous_key,
        contiguous_value,
        grad_output,
        logsumexp,
        delta,
        grad_query,
        sequence_length,
        head_dim,
        softmax_scale,
        causal,
        BLOCK_M=launch.block_m,
        BLOCK_N=launch.block_n,
        BLOCK_D=head_dim,
        num_warps=launch.num_warps,
        num_stages=launch.num_stages,
    )
    _flash_attention_backward_key_value_kernel[key_grid](
        contiguous_query,
        contiguous_key,
        contiguous_value,
        grad_output,
        logsumexp,
        delta,
        grad_key,
        grad_value,
        sequence_length,
        head_dim,
        softmax_scale,
        causal,
        BLOCK_M=launch.block_m,
        BLOCK_N=launch.block_n,
        BLOCK_D=head_dim,
        num_warps=launch.num_warps,
        num_stages=launch.num_stages,
    )
    return grad_query, grad_key, grad_value


class _FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool,
        softmax_scale: float,
    ) -> torch.Tensor:
        output, logsumexp = _forward_impl(
            query, key, value, causal, softmax_scale
        )
        ctx.save_for_backward(query, key, value, output, logsumexp)
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        query, key, value, output, logsumexp = ctx.saved_tensors
        grad_query, grad_key, grad_value = _backward_impl(
            query,
            key,
            value,
            output,
            logsumexp,
            grad_output,
            ctx.causal,
            ctx.softmax_scale,
        )
        return grad_query, grad_key, grad_value, None, None


def flash_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Compute AMD ROCm attention with Triton forward and first-order backward."""
    _validate_inputs(query, key, value)
    head_dim = query.shape[-1]
    if head_dim not in (16, 32, 64, 128):
        raise ValueError("head_dim must be one of 16, 32, 64, or 128")
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
    if not math.isfinite(scale):
        raise ValueError("softmax_scale must be finite")
    return _FlashAttentionFunction.apply(query, key, value, causal, scale)