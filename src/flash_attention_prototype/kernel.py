import math

import torch
import triton
import triton.language as tl


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
            mask=(current_keys[:, None] < sequence_length) & (dims[None, :] < head_dim),
            other=0.0,
        )
        value_tile = tl.load(
            value + value_offsets,
            mask=(current_keys[:, None] < sequence_length) & (dims[None, :] < head_dim),
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
            score_mask = score_mask & (
                query_rows[:, None] >= current_keys[None, :]
            )
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
            score_mask = score_mask & (
                current_queries[:, None] >= key_rows[None, :]
            )
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
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape [B, H, N, D]")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key, and value must have identical shapes")
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise ValueError("query, key, and value must be CUDA tensors")
    if query.device != key.device or query.device != value.device:
        raise ValueError("query, key, and value must be on the same CUDA device")
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

    block_m = 32 if head_dim == 128 else 64
    block_n = 32 if head_dim == 128 else 64
    grid = (triton.cdiv(sequence_length, block_m), batch_size * num_heads)

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
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=head_dim,
        num_warps=4,
        num_stages=2,
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
    grad_output = grad_output.contiguous()
    delta = torch.empty(
        (batch_size, num_heads, sequence_length),
        device=query.device,
        dtype=torch.float32,
    )
    block_m = 32 if head_dim == 128 else 64
    block_n = 32 if head_dim == 128 else 64
    batch_heads = batch_size * num_heads
    query_grid = (triton.cdiv(sequence_length, block_m), batch_heads)
    key_grid = (triton.cdiv(sequence_length, block_n), batch_heads)

    _attention_delta_kernel[query_grid](
        output,
        grad_output,
        delta,
        sequence_length,
        head_dim,
        BLOCK_M=block_m,
        BLOCK_D=head_dim,
        num_warps=4,
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
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=head_dim,
        num_warps=4,
        num_stages=2,
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
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=head_dim,
        num_warps=4,
        num_stages=2,
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
    """Compute attention with Triton forward and first-order backward passes."""
    _validate_inputs(query, key, value)
    head_dim = query.shape[-1]
    if head_dim not in (16, 32, 64, 128):
        raise ValueError("head_dim must be one of 16, 32, 64, or 128")
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
    return _FlashAttentionFunction.apply(query, key, value, causal, scale)