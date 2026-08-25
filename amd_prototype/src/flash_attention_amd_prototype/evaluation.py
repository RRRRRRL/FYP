import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


BASELINE_LABELS = {
    "custom_triton_amd": "Custom AMD Triton",
    "pytorch_math": "PyTorch Math SDPA",
    "rocm_ck": "ROCm CK SDPA",
    "rocm_aotriton": "ROCm AOTriton SDPA",
    "flash_attention_2_rocm": "FlashAttention-2 ROCm",
}

BASELINE_COLORS = {
    "custom_triton_amd": "#007C91",
    "pytorch_math": "#C44E52",
    "rocm_ck": "#009E73",
    "rocm_aotriton": "#E69F00",
    "flash_attention_2_rocm": "#4C72B0",
}

NUMERIC_FIELDS = {
    "batch_size": int,
    "num_heads": int,
    "sequence_length": int,
    "head_dim": int,
    "first_iteration_ms": float,
    "mean_ms": float,
    "median_ms": float,
    "p90_ms": float,
    "min_ms": float,
    "max_ms": float,
    "stdev_ms": float,
    "tflops": float,
    "peak_allocated_bytes": int,
    "peak_reserved_bytes": int,
}


def load_benchmark_results(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"result file does not exist: {source}")
    with source.open(newline="", encoding="utf-8") as source_file:
        rows = list(csv.DictReader(source_file))
    for row in rows:
        for field, converter in NUMERIC_FIELDS.items():
            value = row.get(field)
            row[field] = converter(value) if value not in (None, "") else None
        row["causal"] = str(row.get("causal", "false")).lower() == "true"
    return rows


def successful_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("status") == "ok"]


def log_log_slope(
    rows: Iterable[dict[str, Any]],
    metric: str = "peak_allocated_bytes",
) -> float | None:
    points = [
        (float(row["sequence_length"]), float(row[metric]))
        for row in rows
        if row.get("sequence_length")
        and row.get(metric)
        and float(row[metric]) > 0
    ]
    if len(points) < 2:
        return None
    x_values = [math.log(point[0]) for point in points]
    y_values = [math.log(point[1]) for point in points]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator == 0:
        return None
    return sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values, strict=True)
    ) / denominator


def speedup_rows(
    rows: Iterable[dict[str, Any]],
    candidate: str = "custom_triton_amd",
    reference: str = "pytorch_math",
) -> list[dict[str, Any]]:
    successful = successful_rows(rows)
    lookup = {
        (
            row["baseline"],
            row["pass_name"],
            row["sequence_length"],
            row["batch_size"],
            row["num_heads"],
            row["head_dim"],
            row["dtype"],
            row["causal"],
        ): row
        for row in successful
    }
    comparisons = []
    for row in successful:
        if row["baseline"] != candidate or not row.get("median_ms"):
            continue
        key = (
            reference,
            row["pass_name"],
            row["sequence_length"],
            row["batch_size"],
            row["num_heads"],
            row["head_dim"],
            row["dtype"],
            row["causal"],
        )
        reference_row = lookup.get(key)
        if reference_row and reference_row.get("median_ms"):
            comparisons.append(
                {
                    "pass_name": row["pass_name"],
                    "sequence_length": row["sequence_length"],
                    "speedup": reference_row["median_ms"] / row["median_ms"],
                }
            )
    return sorted(
        comparisons,
        key=lambda row: (str(row["pass_name"]), int(row["sequence_length"])),
    )


def benchmark_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    all_rows = list(rows)
    successful = successful_rows(all_rows)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in successful:
        grouped[(str(row["baseline"]), str(row["pass_name"]))].append(row)
    return {
        "total_rows": len(all_rows),
        "successful_rows": len(successful),
        "statuses": {
            status: sum(row.get("status") == status for row in all_rows)
            for status in sorted({str(row.get("status")) for row in all_rows})
        },
        "memory_scaling_exponents": {
            f"{baseline}:{pass_name}": log_log_slope(group)
            for (baseline, pass_name), group in grouped.items()
        },
        "custom_vs_math_speedups": speedup_rows(successful),
    }


def _plotting_modules():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except ImportError as error:
        raise RuntimeError(
            "plotting requires: pip install -e '.[evaluation]'"
        ) from error
    return pyplot


def plot_performance(
    rows: Iterable[dict[str, Any]], output_directory: str | Path
) -> list[Path]:
    pyplot = _plotting_modules()
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    all_rows = successful_rows(rows)
    paths: list[Path] = []
    for metric, label, stem in (
        ("median_ms", "Median latency (ms)", "latency"),
        ("tflops", "Effective throughput (TFLOP/s)", "throughput"),
    ):
        figure, axes = pyplot.subplots(1, 2, figsize=(12, 4.6))
        for axis, pass_name in zip(axes, ("forward", "backward"), strict=True):
            for baseline in sorted({str(row["baseline"]) for row in all_rows}):
                points = sorted(
                    (
                        (int(row["sequence_length"]), float(row[metric]))
                        for row in all_rows
                        if row["baseline"] == baseline
                        and row["pass_name"] == pass_name
                        and row.get(metric) is not None
                    ),
                    key=lambda point: point[0],
                )
                if points:
                    axis.plot(
                        [point[0] for point in points],
                        [point[1] for point in points],
                        marker="o",
                        label=BASELINE_LABELS.get(baseline, baseline),
                        color=BASELINE_COLORS.get(baseline),
                    )
            axis.set_title(pass_name.title())
            axis.set_xlabel("Sequence length")
            axis.set_ylabel(label)
            axis.set_xscale("log", base=2)
            axis.grid(True, alpha=0.22)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncols=3, frameon=False)
        figure.tight_layout(rect=(0, 0.12, 1, 1))
        path = destination / f"{stem}.png"
        figure.savefig(path, dpi=220, bbox_inches="tight")
        pyplot.close(figure)
        paths.append(path)
    return paths


def write_evaluation_summary(
    path: str | Path,
    *,
    benchmark_rows: Iterable[dict[str, Any]],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary = benchmark_summary(benchmark_rows)
    lines = [
        "# AMD Attention Evaluation",
        "",
        "The custom implementation is FA3-inspired; it is not Hopper FA3.",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
    ]
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AMD benchmark results")
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, default=Path("evaluation"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    rows = load_benchmark_results(args.results)
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = write_evaluation_summary(
        args.output / "summary.md", benchmark_rows=rows
    )
    if args.plot:
        plot_performance(rows, args.output)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()