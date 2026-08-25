import math

import torch


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape [B, H, N, D]")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key, and value must have identical shapes")
    if query.device != key.device or query.device != value.device:
        raise ValueError("query, key, and value must be on the same device")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("query, key, and value must have the same dtype")
    if not query.is_floating_point():
        raise ValueError("query, key, and value must use a floating-point dtype")


def _resolve_scale(head_dim: int, softmax_scale: float | None) -> float:
    if head_dim == 0:
        raise ValueError("head_dim must be greater than zero")
    return softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)


def _causal_mask(
    query_start: int,
    query_end: int,
    key_start: int,
    key_end: int,
    device: torch.device,
) -> torch.Tensor:
    query_indices = torch.arange(query_start, query_end, device=device)
    key_indices = torch.arange(key_start, key_end, device=device)
    return query_indices[:, None] >= key_indices[None, :]


def reference_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
    block_m: int = 64,
    block_n: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run tiled online-softmax attention and return output and row log-sum-exp."""
    _validate_inputs(query, key, value)
    if block_m <= 0 or block_n <= 0:
        raise ValueError("block_m and block_n must be greater than zero")

    batch_size, num_heads, sequence_length, head_dim = query.shape
    scale = _resolve_scale(head_dim, softmax_scale)
    output = torch.empty_like(query)
    logsumexp = torch.empty(
        (batch_size, num_heads, sequence_length),
        dtype=query.dtype,
        device=query.device,
    )
    if output.numel() == 0:
        return output, logsumexp

    for query_start in range(0, sequence_length, block_m):
        query_end = min(query_start + block_m, sequence_length)
        query_tile = query[:, :, query_start:query_end, :]
        row_shape = (*query_tile.shape[:-1], 1)
        row_max = torch.full(
            row_shape, -torch.inf, dtype=query.dtype, device=query.device
        )
        row_sum = torch.zeros(row_shape, dtype=query.dtype, device=query.device)
        accumulator = torch.zeros_like(query_tile)

        for key_start in range(0, sequence_length, block_n):
            key_end = min(key_start + block_n, sequence_length)
            key_tile = key[:, :, key_start:key_end, :]
            value_tile = value[:, :, key_start:key_end, :]
            scores = torch.matmul(query_tile, key_tile.transpose(-1, -2)) * scale
            if causal:
                mask = _causal_mask(
                    query_start,
                    query_end,
                    key_start,
                    key_end,
                    query.device,
                )
                scores = scores.masked_fill(~mask, -torch.inf)

            block_max = scores.amax(dim=-1, keepdim=True)
            next_max = torch.maximum(row_max, block_max)
            correction = torch.exp(row_max - next_max)
            probabilities = torch.exp(scores - next_max)
            accumulator = accumulator * correction + torch.matmul(
                probabilities, value_tile
            )
            row_sum = row_sum * correction + probabilities.sum(
                dim=-1, keepdim=True
            )
            row_max = next_max

        output[:, :, query_start:query_end, :] = accumulator / row_sum
        logsumexp[:, :, query_start:query_end] = (
            row_max + torch.log(row_sum)
        ).squeeze(-1)

    return output, logsumexp


def reference_attention_delta(
    output: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    """Compute the softmax-backward row scalar D = rowsum(dO * O)."""
    if output.shape != grad_output.shape:
        raise ValueError("output and grad_output must have identical shapes")
    return (output * grad_output).sum(dim=-1)


def reference_attention_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    logsumexp: torch.Tensor,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
    block_m: int = 64,
    block_n: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rematerialize probability tiles and compute analytical dQ, dK, and dV."""
    _validate_inputs(query, key, value)
    if output.shape != query.shape or grad_output.shape != query.shape:
        raise ValueError("output and grad_output must match the input shape")
    if logsumexp.shape != query.shape[:-1]:
        raise ValueError("logsumexp must have shape [B, H, N]")
    if block_m <= 0 or block_n <= 0:
        raise ValueError("block_m and block_n must be greater than zero")

    sequence_length = query.shape[-2]
    scale = _resolve_scale(query.shape[-1], softmax_scale)
    grad_query = torch.zeros_like(query)
    grad_key = torch.zeros_like(key)
    grad_value = torch.zeros_like(value)
    if query.numel() == 0:
        return grad_query, grad_key, grad_value

    delta = reference_attention_delta(output, grad_output)
    for query_start in range(0, sequence_length, block_m):
        query_end = min(query_start + block_m, sequence_length)
        query_tile = query[:, :, query_start:query_end, :]
        grad_output_tile = grad_output[:, :, query_start:query_end, :]
        row_lse = logsumexp[:, :, query_start:query_end, None]
        row_delta = delta[:, :, query_start:query_end, None]

        for key_start in range(0, sequence_length, block_n):
            key_end = min(key_start + block_n, sequence_length)
            key_tile = key[:, :, key_start:key_end, :]
            value_tile = value[:, :, key_start:key_end, :]
            scores = torch.matmul(query_tile, key_tile.transpose(-1, -2)) * scale
            if causal:
                mask = _causal_mask(
                    query_start,
                    query_end,
                    key_start,
                    key_end,
                    query.device,
                )
                scores = scores.masked_fill(~mask, -torch.inf)

            probabilities = torch.exp(scores - row_lse)
            grad_probabilities = torch.matmul(
                grad_output_tile, value_tile.transpose(-1, -2)
            )
            grad_scores = probabilities * (grad_probabilities - row_delta)
            grad_query[:, :, query_start:query_end, :] += (
                torch.matmul(grad_scores, key_tile) * scale
            )
            grad_key[:, :, key_start:key_end, :] += (
                torch.matmul(grad_scores.transpose(-1, -2), query_tile) * scale
            )
            grad_value[:, :, key_start:key_end, :] += torch.matmul(
                probabilities.transpose(-1, -2), grad_output_tile
            )

    return grad_query, grad_key, grad_value


class _ReferenceAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool,
        softmax_scale: float,
        block_m: int,
        block_n: int,
    ) -> torch.Tensor:
        output, logsumexp = reference_attention_forward(
            query,
            key,
            value,
            causal=causal,
            softmax_scale=softmax_scale,
            block_m=block_m,
            block_n=block_n,
        )
        ctx.save_for_backward(query, key, value, output, logsumexp)
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        ctx.block_m = block_m
        ctx.block_n = block_n
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        query, key, value, output, logsumexp = ctx.saved_tensors
        grad_query, grad_key, grad_value = reference_attention_backward(
            query,
            key,
            value,
            output,
            grad_output,
            logsumexp,
            causal=ctx.causal,
            softmax_scale=ctx.softmax_scale,
            block_m=ctx.block_m,
            block_n=ctx.block_n,
        )
        return grad_query, grad_key, grad_value, None, None, None, None


def reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
    block_m: int = 64,
    block_n: int = 64,
) -> torch.Tensor:
    """Compute differentiable tiled attention using ordinary PyTorch operations."""
    _validate_inputs(query, key, value)
    scale = _resolve_scale(query.shape[-1], softmax_scale)
    return _ReferenceAttentionFunction.apply(
        query, key, value, causal, scale, block_m, block_n
    )