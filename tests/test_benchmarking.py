import csv
import json

import torch

from flash_attention_prototype.benchmarking import (
    BASELINES,
    BenchmarkConfig,
    BenchmarkResult,
    attention_flops,
    write_results,
)


def make_config(*, causal: bool = False) -> BenchmarkConfig:
    return BenchmarkConfig(
        batch_size=2,
        num_heads=4,
        sequence_length=8,
        head_dim=16,
        dtype=torch.float16,
        causal=causal,
    )


def test_attention_flops_uses_documented_convention() -> None:
    config = make_config()
    expected_forward = 4 * 2 * 4 * 8 * 8 * 16

    assert attention_flops(config, "forward") == expected_forward
    assert attention_flops(config, "backward") == 5 * expected_forward // 2


def test_causal_flops_count_only_lower_triangle() -> None:
    config = make_config(causal=True)
    expected_forward = 4 * 2 * 4 * (8 * 9 // 2) * 16

    assert attention_flops(config, "forward") == expected_forward


def test_all_required_baselines_have_explicit_status() -> None:
    assert set(BASELINES) == {
        "custom_triton",
        "pytorch_math",
        "flash_attention_2",
        "flex_attention",
    }

    for baseline in BASELINES.values():
        status = baseline.status(make_config())
        assert status.name == baseline.name
        assert status.available or status.reason


def test_write_results_produces_csv_and_manifest(tmp_path) -> None:
    result = BenchmarkResult(
        baseline="custom_triton",
        pass_name="forward",
        status="ok",
        reason=None,
        batch_size=1,
        num_heads=2,
        sequence_length=128,
        head_dim=64,
        dtype="float16",
        causal=True,
        mean_ms=1.1,
        median_ms=1.0,
        min_ms=0.9,
        max_ms=1.2,
        tflops=4.2,
        peak_allocated_bytes=1024,
        peak_reserved_bytes=2048,
    )

    csv_path, manifest_path = write_results(
        [result], tmp_path, {"environment": {"gpu_name": "test"}}
    )

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert rows[0]["baseline"] == "custom_triton"
    assert rows[0]["pass_name"] == "forward"
    assert manifest["metadata"]["environment"]["gpu_name"] == "test"
    assert manifest["results"][0]["peak_allocated_bytes"] == 1024