import pytest
import torch
import torch.nn.functional as functional

from flash_attention_prototype import (
    reference_attention,
    reference_attention_backward,
    reference_attention_delta,
    reference_attention_forward,
)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("softmax_scale", [None, 0.37])
def test_reference_forward_matches_sdpa(
    causal: bool, softmax_scale: float | None
) -> None:
    torch.manual_seed(0)
    shape = (2, 2, 7, 5)
    query = torch.randn(shape, dtype=torch.float32)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    actual, _ = reference_attention_forward(
        query,
        key,
        value,
        causal=causal,
        softmax_scale=softmax_scale,
        block_m=3,
        block_n=4,
    )
    expected = functional.scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=causal,
        scale=softmax_scale,
    )

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_reference_delta_matches_definition() -> None:
    torch.manual_seed(1)
    output = torch.randn(2, 3, 7, 5, dtype=torch.float64)
    grad_output = torch.randn_like(output)

    actual = reference_attention_delta(output, grad_output)
    expected = torch.einsum("bhnd,bhnd->bhn", output, grad_output)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    ("gradient_index", "gradient_name"),
    [(0, "dQ"), (1, "dK"), (2, "dV")],
)
def test_reference_backward_gradient_matches_sdpa(
    causal: bool, gradient_index: int, gradient_name: str
) -> None:
    torch.manual_seed(2)
    shape = (1, 2, 7, 5)
    query = torch.randn(shape, dtype=torch.float64)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    grad_output = torch.randn_like(query)
    output, logsumexp = reference_attention_forward(
        query,
        key,
        value,
        causal=causal,
        block_m=3,
        block_n=4,
    )

    actual_gradients = reference_attention_backward(
        query,
        key,
        value,
        output,
        grad_output,
        logsumexp,
        causal=causal,
        block_m=3,
        block_n=4,
    )
    reference_inputs = tuple(
        tensor.detach().clone().requires_grad_(True)
        for tensor in (query, key, value)
    )
    expected_output = functional.scaled_dot_product_attention(
        *reference_inputs, is_causal=causal
    )
    expected_gradients = torch.autograd.grad(
        expected_output, reference_inputs, grad_output
    )

    torch.testing.assert_close(
        actual_gradients[gradient_index],
        expected_gradients[gradient_index],
        atol=1e-9,
        rtol=1e-7,
        msg=lambda message: f"{gradient_name} mismatch: {message}",
    )


@pytest.mark.parametrize("causal", [False, True])
def test_reference_autograd_gradcheck(causal: bool) -> None:
    torch.manual_seed(3)
    inputs = tuple(
        torch.randn(1, 1, 4, 3, dtype=torch.float64, requires_grad=True)
        for _ in range(3)
    )

    def attention(*tensors: torch.Tensor) -> torch.Tensor:
        return reference_attention(
            *tensors,
            causal=causal,
            block_m=2,
            block_n=3,
        )

    assert torch.autograd.gradcheck(
        attention,
        inputs,
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        fast_mode=True,
    )


def test_reference_is_stable_for_large_scores() -> None:
    query = torch.tensor(
        [[[[100.0, -100.0], [80.0, 90.0], [-95.0, 85.0]]]],
        dtype=torch.float64,
    )
    key = torch.tensor(
        [[[[90.0, -110.0], [-75.0, 95.0], [105.0, 70.0]]]],
        dtype=torch.float64,
    )
    value = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]],
        dtype=torch.float64,
    )

    actual = reference_attention(
        query, key, value, block_m=2, block_n=2
    )
    expected = functional.scaled_dot_product_attention(query, key, value)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)


def test_reference_rejects_invalid_block_size() -> None:
    tensor = torch.randn(1, 1, 4, 3)

    with pytest.raises(ValueError, match="greater than zero"):
        reference_attention_forward(tensor, tensor, tensor, block_m=0)