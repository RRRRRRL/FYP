import unittest

import torch
import torch.nn.functional as functional

from flash_attention_amd_prototype.reference import (
    reference_attention,
    reference_attention_backward,
    reference_attention_forward,
)


class ReferenceAttentionTests(unittest.TestCase):
    def test_forward_matches_sdpa(self) -> None:
        torch.manual_seed(0)
        for causal in (False, True):
            with self.subTest(causal=causal):
                query = torch.randn(1, 2, 7, 4, dtype=torch.float64)
                key = torch.randn_like(query)
                value = torch.randn_like(query)
                actual, _ = reference_attention_forward(
                    query,
                    key,
                    value,
                    causal=causal,
                    block_m=3,
                    block_n=4,
                )
                expected = functional.scaled_dot_product_attention(
                    query, key, value, is_causal=causal
                )
                torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    def test_analytical_backward_matches_autograd(self) -> None:
        torch.manual_seed(1)
        query = torch.randn(1, 1, 5, 4, dtype=torch.float64, requires_grad=True)
        key = torch.randn_like(query, requires_grad=True)
        value = torch.randn_like(query, requires_grad=True)
        output, logsumexp = reference_attention_forward(
            query, key, value, causal=True, block_m=3, block_n=2
        )
        grad_output = torch.randn_like(output)
        actual = reference_attention_backward(
            query,
            key,
            value,
            output,
            grad_output,
            logsumexp,
            causal=True,
            block_m=3,
            block_n=2,
        )
        expected_output = functional.scaled_dot_product_attention(
            query, key, value, is_causal=True
        )
        expected = torch.autograd.grad(
            expected_output, (query, key, value), grad_outputs=grad_output
        )
        for actual_gradient, expected_gradient in zip(actual, expected, strict=True):
            torch.testing.assert_close(
                actual_gradient, expected_gradient, rtol=1e-9, atol=1e-9
            )

    def test_custom_reference_autograd_is_finite(self) -> None:
        query = torch.full((1, 1, 4, 4), 100.0, dtype=torch.float64)
        query.requires_grad_(True)
        key = query.detach().clone().requires_grad_(True)
        value = torch.randn_like(query, requires_grad=True)
        output = reference_attention(query, key, value, causal=True, block_m=2)
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(output).all())
        for tensor in (query, key, value):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())


if __name__ == "__main__":
    unittest.main()