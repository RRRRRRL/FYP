import unittest

import torch

from flash_attention_amd_prototype.benchmarking import (
    BASELINES,
    BenchmarkConfig,
    attention_flops,
)


class BenchmarkingTests(unittest.TestCase):
    def test_registry_contains_explicit_rocm_backends(self) -> None:
        self.assertEqual(
            set(BASELINES),
            {
                "custom_triton_amd",
                "pytorch_math",
                "rocm_ck",
                "rocm_aotriton",
                "flash_attention_2_rocm",
            },
        )

    def test_attention_flops_uses_causal_triangle(self) -> None:
        dense = BenchmarkConfig(1, 2, 8, 16, torch.float16, False)
        causal = BenchmarkConfig(1, 2, 8, 16, torch.float16, True)
        self.assertEqual(attention_flops(dense, "forward"), 4 * 1 * 2 * 64 * 16)
        self.assertEqual(attention_flops(causal, "forward"), 4 * 1 * 2 * 36 * 16)
        self.assertEqual(
            attention_flops(dense, "backward"),
            5 * attention_flops(dense, "forward") // 2,
        )


if __name__ == "__main__":
    unittest.main()