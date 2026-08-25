import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


BASELINE_LABELS = {
    "custom_triton": "Custom Triton",
    "pytorch_math": "PyTorch Math SDPA",
    "flash_attention_2": "FlashAttention-2",
    "flex_attention": "FlexAttention",
}

BASELINE_COLORS = {
    "custom_triton": "#007C91",
    "pytorch_math": "#C44E52",
    "flash_attention_2": "#E69F00",
    "flex_attention": "#4C72B0",
}

NUMERIC_BENCHMARK_FIELDS = {
    "batch_size": int,
    "num_heads": int,
    "sequence_length": int,
    "head_dim": int,
    "mean_ms": float,
    "median_ms": float,
    "min_ms": float,
    "max_ms": float,
    "tflops": float,
    "peak_allocated_bytes": int,
    "peak_reserved_bytes": int,
}

NUMERIC_TRAINING_FIELDS = {
    "step": int,
    "custom_loss": float,
    "reference_loss": float,
    "custom_grad_norm": float,
    "reference_grad_norm": float,
    "parameter_max_abs_difference": float,
}


def _parse_rows(
    path: str | Path,
    numeric_fields: dict[str, type[int] | type[float]],
) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"result file does not exist: {source}")
    with source.open(newline="", encoding="utf-8") as source_file:
        rows = list(csv.DictReader(source_file))
    for row in rows:
        for field, converter in numeric_fields.items():
            value = row.get(field)
            row[field] = converter(value) if value not in (None, "") else None
        if "causal" in row:
            row["causal"] = str(row["causal"]).lower() == "true"
    return rows


def load_benchmark_results(path: str | Path) -> list[dict[str, Any]]:
    return _parse_rows(path, NUMERIC_BENCHMARK_FIELDS)


def load_training_results(path: str | Path) -> list[dict[str, Any]]:
    return _parse_rows(path, NUMERIC_TRAINING_FIELDS)


def successful_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("status") == "ok"]


def log_log_slope(
    rows: Iterable[dict[str, Any]],
    metric: str = "peak_allocated_bytes",
) -> float | None:
    """Fit the exponent p in metric = c * sequence_length**p."""
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
    candidate: str = "custom_triton",
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
    by_workload: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in successful:
        key = (
            row["baseline"],
            row["pass_name"],
            row["batch_size"],
            row["num_heads"],
            row["head_dim"],
            row["dtype"],
            row["causal"],
        )
        by_workload[key].append(row)
    slopes = {
        (
            f"{key[0]}:{key[1]}:B{key[2]}:H{key[3]}:D{key[4]}:"
            f"{key[5]}:causal={key[6]}"
        ): log_log_slope(group)
        for key, group in by_workload.items()
    }
    speedups = speedup_rows(successful)
    return {
        "total_rows": len(all_rows),
        "successful_rows": len(successful),
        "statuses": {
            status: sum(row.get("status") == status for row in all_rows)
            for status in sorted({str(row.get("status")) for row in all_rows})
        },
        "memory_scaling_exponents": slopes,
        "custom_vs_math_speedups": speedups,
    }


def training_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    history = list(rows)
    if not history:
        return {"steps": 0}
    return {
        "steps": len(history),
        "custom_initial_loss": history[0]["custom_loss"],
        "custom_final_loss": history[-1]["custom_loss"],
        "reference_initial_loss": history[0]["reference_loss"],
        "reference_final_loss": history[-1]["reference_loss"],
        "final_parameter_max_abs_difference": history[-1][
            "parameter_max_abs_difference"
        ],
        "maximum_parameter_max_abs_difference": max(
            row["parameter_max_abs_difference"] for row in history
        ),
    }


def _plotting_modules():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
        from matplotlib.ticker import FuncFormatter
    except ImportError as error:
        raise RuntimeError(
            "plotting requires the evaluation extra: pip install -e '.[evaluation]'"
        ) from error
    return pyplot, FuncFormatter


def _style_axis(axis, title: str, x_label: str, y_label: str) -> None:
    axis.set_title(title, fontsize=11, fontweight="semibold")
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.22, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def _save_figure(figure, destination: Path, stem: str) -> list[Path]:
    paths = [destination / f"{stem}.png", destination / f"{stem}.pdf"]
    figure.savefig(paths[0], dpi=220, bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    return paths


def _group_metric(
    rows: Iterable[dict[str, Any]], metric: str
) -> dict[tuple[str, str], list[tuple[int, float]]]:
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in successful_rows(rows):
        value = row.get(metric)
        if value is not None:
            grouped[(row["baseline"], row["pass_name"])].append(
                (int(row["sequence_length"]), float(value))
            )
    for points in grouped.values():
        points.sort()
    return grouped


def plot_performance(
    rows: Iterable[dict[str, Any]], output_directory: str | Path
) -> list[Path]:
    pyplot, formatter = _plotting_modules()
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    all_rows = list(rows)
    paths = []

    for metric, y_label, filename in (
        ("median_ms", "Median latency (ms)", "latency.png"),
        ("tflops", "Effective throughput (TFLOP/s)", "throughput.png"),
    ):
        figure, axes = pyplot.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
        grouped = _group_metric(all_rows, metric)
        for axis, pass_name in zip(axes, ("forward", "backward"), strict=True):
            for (baseline, row_pass), points in grouped.items():
                if row_pass != pass_name:
                    continue
                axis.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    marker="o",
                    linewidth=2,
                    label=BASELINE_LABELS.get(baseline, baseline),
                    color=BASELINE_COLORS.get(baseline),
                )
            _style_axis(axis, pass_name.title(), "Sequence length", y_label)
            axis.set_xscale("log", base=2)
            axis.xaxis.set_major_formatter(formatter(lambda value, _: f"{int(value):,}"))
        handles, labels = axes[0].get_legend_handles_labels()
        if not handles:
            handles, labels = axes[1].get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside lower center", ncols=4, frameon=False)
        paths.extend(_save_figure(figure, destination, Path(filename).stem))
        pyplot.close(figure)

    figure, axes = pyplot.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    grouped = _group_metric(all_rows, "peak_allocated_bytes")
    for axis, pass_name in zip(axes, ("forward", "backward"), strict=True):
        for (baseline, row_pass), points in grouped.items():
            if row_pass != pass_name:
                continue
            axis.plot(
                [point[0] for point in points],
                [point[1] / (1024**3) for point in points],
                marker="o",
                linewidth=2,
                label=BASELINE_LABELS.get(baseline, baseline),
                color=BASELINE_COLORS.get(baseline),
            )
        _style_axis(axis, pass_name.title(), "Sequence length", "Peak allocated VRAM (GiB)")
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=2)
        axis.xaxis.set_major_formatter(formatter(lambda value, _: f"{int(value):,}"))
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=4, frameon=False)
    paths.extend(_save_figure(figure, destination, "memory-scaling"))
    pyplot.close(figure)

    comparisons = speedup_rows(all_rows)
    if comparisons:
        figure, axis = pyplot.subplots(figsize=(7.5, 4.6), constrained_layout=True)
        for pass_name, color in (("forward", "#007C91"), ("backward", "#C44E52")):
            points = [row for row in comparisons if row["pass_name"] == pass_name]
            axis.plot(
                [row["sequence_length"] for row in points],
                [row["speedup"] for row in points],
                marker="o",
                linewidth=2,
                label=pass_name.title(),
                color=color,
            )
        axis.axhline(1.0, color="#555555", linestyle="--", linewidth=1)
        _style_axis(
            axis,
            "Custom Triton Speedup over PyTorch Math SDPA",
            "Sequence length",
            "Speedup (x)",
        )
        axis.set_xscale("log", base=2)
        axis.xaxis.set_major_formatter(formatter(lambda value, _: f"{int(value):,}"))
        axis.legend(frameon=False)
        paths.extend(_save_figure(figure, destination, "speedup"))
        pyplot.close(figure)
    return paths


def plot_training(
    rows: Iterable[dict[str, Any]], output_directory: str | Path
) -> list[Path]:
    pyplot, _ = _plotting_modules()
    history = list(rows)
    if not history:
        raise ValueError("training results are empty")
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    steps = [row["step"] for row in history]
    figure, axes = pyplot.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].plot(steps, [row["custom_loss"] for row in history], label="Custom Triton", color="#007C91")
    axes[0].plot(steps, [row["reference_loss"] for row in history], label="Math SDPA", color="#C44E52")
    _style_axis(axes[0], "Training Loss", "Step", "Cross-entropy loss")
    axes[0].legend(frameon=False)
    axes[1].plot(steps, [row["custom_grad_norm"] for row in history], label="Custom Triton", color="#007C91")
    axes[1].plot(steps, [row["reference_grad_norm"] for row in history], label="Math SDPA", color="#C44E52")
    _style_axis(axes[1], "Gradient Norm", "Step", "Global L2 norm")
    axes[1].legend(frameon=False)
    axes[2].plot(steps, [row["parameter_max_abs_difference"] for row in history], color="#E69F00")
    _style_axis(axes[2], "Parameter Divergence", "Step", "Maximum absolute difference")
    axes[2].set_yscale("symlog", linthresh=1e-8)
    paths = _save_figure(figure, destination, "training-validation")
    pyplot.close(figure)
    return paths


def write_evaluation_summary(
    output_path: str | Path,
    benchmark_rows: Iterable[dict[str, Any]] | None = None,
    training_rows: Iterable[dict[str, Any]] | None = None,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Evaluation Summary", ""]
    if benchmark_rows is not None:
        summary = benchmark_summary(benchmark_rows)
        lines.extend(
            [
                "## Performance",
                "",
                f"- Successful measurements: {summary['successful_rows']} / {summary['total_rows']}",
                f"- Status counts: `{json.dumps(summary['statuses'], sort_keys=True)}`",
                "",
                "### Memory Scaling Exponents",
                "",
                "An exponent near 1 indicates linear scaling; near 2 indicates quadratic scaling.",
                "",
                "| Baseline and pass | Fitted exponent |",
                "| --- | ---: |",
            ]
        )
        for name, slope in sorted(summary["memory_scaling_exponents"].items()):
            formatted = f"{slope:.3f}" if slope is not None else "insufficient data"
            lines.append(f"| {name} | {formatted} |")
        lines.extend(["", "### Custom Triton Speedup", "", "| Pass | N | Speedup |", "| --- | ---: | ---: |"])
        for row in summary["custom_vs_math_speedups"]:
            lines.append(f"| {row['pass_name']} | {row['sequence_length']:,} | {row['speedup']:.2f}x |")
        lines.append("")
    if training_rows is not None:
        summary = training_summary(training_rows)
        lines.extend(["## Training Validation", ""])
        if summary["steps"]:
            lines.extend(
                [
                    f"- Steps: {summary['steps']}",
                    f"- Custom loss: {summary['custom_initial_loss']:.6f} -> {summary['custom_final_loss']:.6f}",
                    f"- Reference loss: {summary['reference_initial_loss']:.6f} -> {summary['reference_final_loss']:.6f}",
                    f"- Final maximum parameter difference: {summary['final_parameter_max_abs_difference']:.6g}",
                ]
            )
        else:
            lines.append("No training rows were provided.")
        lines.append("")
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Create plots and summaries from FlashAttention experiment artifacts"
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        help="benchmark results.csv produced by benchmark.py",
    )
    parser.add_argument(
        "--training",
        type=Path,
        help="training.csv produced by training_validation.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/evaluation"),
        help="directory for PNG plots and summary.md",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="write summary.md without requiring Matplotlib",
    )
    args = parser.parse_args()
    if args.benchmark is None and args.training is None:
        parser.error("provide --benchmark, --training, or both")
    args.output.mkdir(parents=True, exist_ok=True)
    benchmark_rows = (
        load_benchmark_results(args.benchmark)
        if args.benchmark is not None
        else None
    )
    training_rows = (
        load_training_results(args.training)
        if args.training is not None
        else None
    )
    generated = []
    if not args.summary_only:
        if benchmark_rows is not None:
            generated.extend(plot_performance(benchmark_rows, args.output))
        if training_rows is not None:
            generated.extend(plot_training(training_rows, args.output))
    generated.append(
        write_evaluation_summary(
            args.output / "summary.md", benchmark_rows, training_rows
        )
    )
    print("Generated:")
    for path in generated:
        print(f"  {path}")