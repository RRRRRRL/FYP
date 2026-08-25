import argparse
from pathlib import Path

import torch

from flash_attention_prototype.benchmarking import BASELINES, BenchmarkConfig
from flash_attention_prototype.environment import environment_dict
from flash_attention_prototype.training import (
    TrainingConfig,
    run_training_comparison,
    training_loss_decreased,
    write_training_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate GPT-style training with custom FlashAttention"
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--output", type=Path, default=Path("results/training"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for custom Triton training validation.")
    if args.quick:
        args.steps = 2
        args.sequence = 32
        args.batch = 1

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    config = TrainingConfig(
        sequence_length=args.sequence,
        batch_size=args.batch,
        steps=args.steps,
    )
    benchmark_config = BenchmarkConfig(
        batch_size=config.batch_size,
        num_heads=config.num_heads,
        sequence_length=config.sequence_length,
        head_dim=config.head_dim,
        dtype=dtype,
        causal=True,
    )
    custom_status = BASELINES["custom_triton"].status(benchmark_config)
    if not custom_status.available:
        raise SystemExit(custom_status.reason or "custom Triton is unavailable")

    history = run_training_comparison(
        config,
        BASELINES["custom_triton"],
        BASELINES["pytorch_math"],
        device=torch.device("cuda"),
        dtype=dtype,
    )
    csv_path, manifest_path = write_training_results(
        history,
        config,
        args.output,
        {
            "custom_backend": "custom_triton",
            "reference_backend": "pytorch_math",
            "dtype": args.dtype,
            "environment": environment_dict(),
        },
    )
    initial = history[0]
    final = history[-1]
    print(
        f"custom loss {initial.custom_loss:.4f} -> {final.custom_loss:.4f}; "
        f"reference loss {initial.reference_loss:.4f} -> "
        f"{final.reference_loss:.4f}"
    )
    print(f"Wrote {csv_path} and {manifest_path}")
    if not training_loss_decreased(history):
        raise SystemExit("training validation failed: loss did not decrease")


if __name__ == "__main__":
    main()