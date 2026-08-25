import csv

import pytest

from flash_attention_prototype.evaluation import (
    benchmark_summary,
    load_benchmark_results,
    load_training_results,
    log_log_slope,
    speedup_rows,
    training_summary,
    write_evaluation_summary,
)


BENCHMARK_FIELDS = [
    "baseline",
    "pass_name",
    "status",
    "reason",
    "batch_size",
    "num_heads",
    "sequence_length",
    "head_dim",
    "dtype",
    "causal",
    "mean_ms",
    "median_ms",
    "min_ms",
    "max_ms",
    "tflops",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
]


def write_csv(path, fields, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def benchmark_row(baseline: str, sequence: int, latency: float, memory: int):
    return {
        "baseline": baseline,
        "pass_name": "forward",
        "status": "ok",
        "reason": "",
        "batch_size": 1,
        "num_heads": 8,
        "sequence_length": sequence,
        "head_dim": 64,
        "dtype": "float16",
        "causal": "True",
        "mean_ms": latency,
        "median_ms": latency,
        "min_ms": latency * 0.9,
        "max_ms": latency * 1.1,
        "tflops": 1.0,
        "peak_allocated_bytes": memory,
        "peak_reserved_bytes": memory,
    }


def test_benchmark_summary_computes_linear_scaling_and_speedup(tmp_path) -> None:
    path = tmp_path / "results.csv"
    rows = [
        benchmark_row("custom_triton", 1024, 2.0, 100),
        benchmark_row("custom_triton", 2048, 4.0, 200),
        benchmark_row("custom_triton", 4096, 8.0, 400),
        benchmark_row("pytorch_math", 1024, 4.0, 1000),
        benchmark_row("pytorch_math", 2048, 16.0, 4000),
        benchmark_row("pytorch_math", 4096, 64.0, 16000),
    ]
    write_csv(path, BENCHMARK_FIELDS, rows)

    loaded = load_benchmark_results(path)
    summary = benchmark_summary(loaded)

    assert log_log_slope(loaded[:3]) == pytest.approx(1.0)
    assert log_log_slope(loaded[3:]) == pytest.approx(2.0)
    assert [row["speedup"] for row in speedup_rows(loaded)] == [2.0, 4.0, 8.0]
    assert summary["successful_rows"] == 6


def test_training_summary_and_markdown(tmp_path) -> None:
    training_path = tmp_path / "training.csv"
    fields = [
        "step",
        "custom_loss",
        "reference_loss",
        "custom_grad_norm",
        "reference_grad_norm",
        "parameter_max_abs_difference",
    ]
    write_csv(
        training_path,
        fields,
        [
            {
                "step": 0,
                "custom_loss": 5.0,
                "reference_loss": 5.0,
                "custom_grad_norm": 2.0,
                "reference_grad_norm": 2.0,
                "parameter_max_abs_difference": 0.0,
            },
            {
                "step": 1,
                "custom_loss": 4.0,
                "reference_loss": 3.9,
                "custom_grad_norm": 1.5,
                "reference_grad_norm": 1.4,
                "parameter_max_abs_difference": 0.001,
            },
        ],
    )

    history = load_training_results(training_path)
    summary = training_summary(history)
    summary_path = write_evaluation_summary(
        tmp_path / "summary.md", training_rows=history
    )

    assert summary["custom_final_loss"] == 4.0
    text = summary_path.read_text(encoding="utf-8")
    assert "Training Validation" in text
    assert "5.000000 -> 4.000000" in text