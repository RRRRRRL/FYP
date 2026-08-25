import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch

from flash_attention_amd_prototype.backend import probe_rocm
from flash_attention_amd_prototype.benchmarking import (
    BASELINES,
    BenchmarkConfig,
    benchmark_baseline,
    write_results,
)
from flash_attention_amd_prototype.environment import environment_dict


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark AMD ROCm attention implementations"
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument(
        "--sequences",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192, 16384],
    )
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument(
        "--baselines", nargs="+", choices=BASELINES, default=list(BASELINES)
    )
    parser.add_argument(
        "--passes",
        nargs="+",
        choices=["forward", "backward"],
        default=["forward", "backward"],
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="generate evaluation plots after the benchmark",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use a small workload and short timing loop",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capability = probe_rocm(require_triton=False)
    if not capability.available:
        raise SystemExit(
            f"ROCm is unavailable: {capability.reason}. "
            "Run flash-attention-amd-env for details."
        )
    if args.quick:
        args.batch = 1
        args.heads = 2
        args.sequences = [128]
        args.warmup = 2
        args.repetitions = 5

    output_directory = args.output or Path("results") / datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    results = []
    for sequence_length in args.sequences:
        config = BenchmarkConfig(
            batch_size=args.batch,
            num_heads=args.heads,
            sequence_length=sequence_length,
            head_dim=args.head_dim,
            dtype=DTYPES[args.dtype],
            causal=args.causal,
        )
        for baseline_name in args.baselines:
            for pass_name in args.passes:
                result = benchmark_baseline(
                    BASELINES[baseline_name],
                    config,
                    pass_name,
                    warmup=args.warmup,
                    repetitions=args.repetitions,
                )
                results.append(result)
                latency = (
                    f"{result.median_ms:.3f} ms" if result.median_ms else "-"
                )
                throughput = (
                    f"{result.tflops:.2f} TFLOP/s" if result.tflops else "-"
                )
                print(
                    f"N={sequence_length:<5} {baseline_name:<24} "
                    f"{pass_name:<8} {result.status:<11} {latency:<12} "
                    f"{throughput}"
                )

    arguments = vars(args).copy()
    arguments["output"] = str(output_directory)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "arguments": arguments,
        "environment": environment_dict(),
        "flop_convention": (
            "forward=4*B*H*score_pairs*D; backward=2.5*forward; "
            "causal score_pairs=N*(N+1)/2"
        ),
    }
    csv_path, manifest_path = write_results(
        results, output_directory, metadata
    )
    print(f"Wrote {csv_path} and {manifest_path}")
    if args.plot:
        from flash_attention_amd_prototype.evaluation import (
            plot_performance,
            write_evaluation_summary,
        )

        rows = [result.to_dict() for result in results]
        plot_performance(rows, output_directory)
        write_evaluation_summary(
            output_directory / "summary.md", benchmark_rows=rows
        )
        print(f"Wrote evaluation plots and {output_directory / 'summary.md'}")


if __name__ == "__main__":
    main()