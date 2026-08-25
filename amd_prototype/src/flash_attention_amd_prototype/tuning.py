import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch

from .backend import probe_rocm
from .benchmarking import (
    BASELINES,
    BenchmarkConfig,
    BenchmarkResult,
    PassName,
    benchmark_baseline,
    write_results,
)
from .environment import environment_dict


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def tune_launch_configs(
    config: BenchmarkConfig,
    pass_name: PassName,
    *,
    warmup: int,
    repetitions: int,
) -> list[BenchmarkResult]:
    capability = probe_rocm()
    if not capability.available:
        raise RuntimeError(capability.reason or "ROCm/Triton is unavailable")

    from .kernel import candidate_launch_configs, use_launch_config

    results = []
    baseline = BASELINES["custom_triton_amd"]
    for launch in candidate_launch_configs(
        config.head_dim, capability.architecture
    ):
        with use_launch_config(launch):
            result = benchmark_baseline(
                baseline,
                config,
                pass_name,
                warmup=warmup,
                repetitions=repetitions,
            )
        if result.launch_config is None:
            result = replace(
                result,
                launch_config=json.dumps(launch.to_dict(), sort_keys=True),
            )
        results.append(result)
    return results


def winning_results(
    results: Iterable[BenchmarkResult],
) -> dict[str, dict[str, object]]:
    winners: dict[str, BenchmarkResult] = {}
    for result in results:
        if result.status != "ok" or result.median_ms is None:
            continue
        current = winners.get(result.pass_name)
        if current is None or (
            current.median_ms is not None
            and result.median_ms < current.median_ms
        ):
            winners[result.pass_name] = result
    return {
        pass_name: {
            "median_ms": winner.median_ms,
            "p90_ms": winner.p90_ms,
            "launch_config": (
                json.loads(winner.launch_config)
                if winner.launch_config is not None
                else None
            ),
        }
        for pass_name, winner in winners.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep conservative AMD Triton launch configurations"
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--sequence", type=int, default=2048)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument(
        "--passes",
        nargs="+",
        choices=["forward", "backward"],
        default=["forward", "backward"],
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    capability = probe_rocm()
    if not capability.available:
        raise SystemExit(capability.reason or "ROCm/Triton is unavailable")
    config = BenchmarkConfig(
        batch_size=args.batch,
        num_heads=args.heads,
        sequence_length=args.sequence,
        head_dim=args.head_dim,
        dtype=DTYPES[args.dtype],
        causal=args.causal,
    )
    output = args.output or Path("results") / "tuning" / datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    results: list[BenchmarkResult] = []
    for pass_name in args.passes:
        pass_results = tune_launch_configs(
            config,
            pass_name,
            warmup=args.warmup,
            repetitions=args.repetitions,
        )
        results.extend(pass_results)
        for result in pass_results:
            launch = json.loads(result.launch_config or "{}")
            print(
                f"{pass_name:<8} {launch.get('profile', '-'):<38} "
                f"{result.status:<11} "
                f"{result.median_ms if result.median_ms is not None else '-'}"
            )

    arguments = vars(args).copy()
    arguments["output"] = str(output)
    write_results(
        results,
        output,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "arguments": arguments,
            "environment": environment_dict(),
            "purpose": "architecture-specific launch configuration sweep",
        },
    )
    winners = winning_results(results)
    winner_path = output / "winners.json"
    winner_path.write_text(
        json.dumps(winners, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote tuning results and {winner_path}")


if __name__ == "__main__":
    main()