import pytest
import torch
import torch.nn.functional as functional

from flash_attention_prototype import flash_attention_forward


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="an NVIDIA CUDA GPU is required"
    ),
]


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("sequence_length", [31, 128, 257])
@pytest.mark.parametrize("head_dim", [16, 32, 64, 128])
def test_matches_pytorch_sdpa(
    causal: bool, sequence_length: int, head_dim: int
) -> None:
    torch.manual_seed(0)
    shape = (2, 3, sequence_length, head_dim)
    query = torch.randn(shape, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    actual = flash_attention_forward(query, key, value, causal=causal)
    expected = functional.scaled_dot_product_attention(
        query, key, value, is_causal=causal
    )

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("sequence_length", [31, 129])
@pytest.mark.parametrize("head_dim", [16, 32, 64, 128])
@pytest.mark.parametrize("softmax_scale", [None, 0.37])
def test_backward_matches_pytorch_sdpa(
    causal: bool,
    sequence_length: int,
    head_dim: int,
    softmax_scale: float | None,
) -> None:
    torch.manual_seed(1)
    shape = (1, 2, sequence_length, head_dim)
    query = torch.randn(
        shape, device="cuda", dtype=torch.float16, requires_grad=True
    )
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)
    reference_query = query.detach().clone().requires_grad_(True)
    reference_key = key.detach().clone().requires_grad_(True)
    reference_value = value.detach().clone().requires_grad_(True)
    grad_output = torch.randn_like(query)

    actual = flash_attention_forward(
        query,
        key,
        value,
        causal=causal,
        softmax_scale=softmax_scale,
    )
    expected = functional.scaled_dot_product_attention(
        reference_query,
        reference_key,
        reference_value,
        is_causal=causal,
        scale=softmax_scale,
    )
    actual.backward(grad_output)
    expected.backward(grad_output)

    for actual_grad, expected_grad in (
        (query.grad, reference_query.grad),
        (key.grad, reference_key.grad),
        (value.grad, reference_value.grad),
    ):
        torch.testing.assert_close(
            actual_grad, expected_grad, atol=5e-2, rtol=5e-2
        )


def test_bfloat16_forward_and_backward_matches_sdpa() -> None:
    if not torch.cuda.is_bf16_supported():
        pytest.skip("the active GPU does not support bfloat16")
    torch.manual_seed(4)
    shape = (1, 2, 65, 64)
    query = torch.randn(
        shape, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)
    reference_inputs = tuple(
        tensor.detach().clone().requires_grad_(True)
        for tensor in (query, key, value)
    )
    grad_output = torch.randn_like(query)

    actual = flash_attention_forward(query, key, value, causal=True)
    expected = functional.scaled_dot_product_attention(
        *reference_inputs, is_causal=True
    )
    actual_gradients = torch.autograd.grad(
        actual, (query, key, value), grad_output
    )
    expected_gradients = torch.autograd.grad(
        expected, reference_inputs, grad_output
    )

    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)
    for actual_grad, expected_grad in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_grad, expected_grad, atol=8e-2, rtol=8e-2
        )


def test_noncontiguous_outer_layout_matches_sdpa() -> None:
    torch.manual_seed(5)
    base_shape = (1, 67, 2, 32)
    query = torch.randn(
        base_shape, device="cuda", dtype=torch.float16
    ).transpose(1, 2)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    assert not query.is_contiguous() and query.stride(-1) == 1

    actual = flash_attention_forward(query, key, value)
    expected = functional.scaled_dot_product_attention(query, key, value)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_rejects_cpu_tensors() -> None:
    tensor = torch.randn(1, 1, 16, 16, dtype=torch.float16)

    with pytest.raises(ValueError, match="CUDA tensors"):
        flash_attention_forward(tensor, tensor, tensor)
