import torch

from flash_attention_prototype.benchmarking import PyTorchMathBaseline
from flash_attention_prototype.training import (
    TinyGPT,
    TrainingConfig,
    run_training_comparison,
    training_loss_decreased,
)


def test_tiny_gpt_output_shape() -> None:
    config = TrainingConfig(
        vocab_size=32,
        sequence_length=8,
        embed_dim=16,
        num_heads=2,
        batch_size=2,
        steps=1,
    )
    model = TinyGPT(config, PyTorchMathBaseline())
    tokens = torch.randint(config.vocab_size, (2, 8))

    output = model(tokens)

    assert output.shape == (2, 8, config.vocab_size)


def test_matched_reference_training_stays_identical() -> None:
    config = TrainingConfig(
        vocab_size=32,
        sequence_length=8,
        embed_dim=16,
        num_heads=2,
        batch_size=1,
        steps=2,
    )
    baseline = PyTorchMathBaseline()

    history = run_training_comparison(
        config,
        baseline,
        baseline,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert len(history) == 2
    assert all(step.custom_loss == step.reference_loss for step in history)
    assert all(step.parameter_max_abs_difference == 0.0 for step in history)
    assert training_loss_decreased(history)