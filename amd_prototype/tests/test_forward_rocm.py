import unittest

import torch

from flash_attention_amd_prototype import flash_attention_forward
from flash_attention_amd_prototype.backend import probe_rocm


_CAPABILITY = probe_rocm()


@unittest.skipUnless(
    _CAPABILITY.available,
    _CAPABILITY.reason or "ROCm/Triton hardware is unavailable",
)
class RocmForwardBackwardTests(unittest.TestCase):
    def test_forward_and_gradients_match_math_sdpa(self) -> None:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        torch.manual_seed(7)
        for causal in (False, True):
            for head_dim in (16, 64, 128):
                with self.subTest(causal=causal, head_dim=head_dim):
                    custom_inputs = [
                        torch.randn(
                            1,
                            2,
                            37,
                            head_dim,
                            device="cuda",
                            dtype=torch.float16,
                            requires_grad=True,
                        )
                        for _ in range(3)
                    ]
                    reference_inputs = [
                        tensor.detach().clone().requires_grad_(True)
                        for tensor in custom_inputs
                    ]
                    actual = flash_attention_forward(
                        *custom_inputs, causal=causal
                    )
                    with sdpa_kernel([SDPBackend.MATH]):
                        expected = torch.nn.functional.scaled_dot_product_attention(
                            *reference_inputs, is_causal=causal
                        )
                    torch.testing.assert_close(
                        actual, expected, rtol=2e-2, atol=2e-2
                    )
                    grad_output = torch.randn_like(actual)
                    actual_gradients = torch.autograd.grad(
                        actual, custom_inputs, grad_output
                    )
                    expected_gradients = torch.autograd.grad(
                        expected, reference_inputs, grad_output
                    )
                    for actual_gradient, expected_gradient in zip(
                        actual_gradients, expected_gradients, strict=True
                    ):
                        torch.testing.assert_close(
                            actual_gradient,
                            expected_gradient,
                            rtol=3e-2,
                            atol=3e-2,
                        )


if __name__ == "__main__":
    unittest.main()