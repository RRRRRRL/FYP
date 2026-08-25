import json

from flash_attention_prototype.environment import (
    EnvironmentReport,
    collect_environment,
    write_environment_json,
)


def test_collect_environment_has_reproducibility_fields() -> None:
    report = collect_environment()

    assert isinstance(report, EnvironmentReport)
    assert report.platform
    assert report.python
    assert isinstance(report.cuda_available, bool)
    assert isinstance(report.flash_attention_2_available, bool)
    assert isinstance(report.flex_attention_available, bool)


def test_write_environment_json(tmp_path) -> None:
    destination = write_environment_json(tmp_path / "environment.json")
    report = json.loads(destination.read_text(encoding="utf-8"))

    assert destination.exists()
    assert report["python"]
    assert "torch" in report
    assert "cuda_available" in report