import copy
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .benchmarking import AttentionBaseline


@dataclass(frozen=True)
class TrainingConfig:
    vocab_size: int = 256
    sequence_length: int = 128
    embed_dim: int = 64
    num_heads: int = 4
    mlp_ratio: int = 4
    batch_size: int = 4
    steps: int = 100
    learning_rate: float = 1e-3
    seed: int = 0

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads


@dataclass(frozen=True)
class TrainingStep:
    step: int
    custom_loss: float
    reference_loss: float
    custom_grad_norm: float
    reference_grad_norm: float
    parameter_max_abs_difference: float


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        baseline: AttentionBaseline,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.baseline = baseline
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = inputs.shape
        query, key, value = self.qkv(inputs).view(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        ).unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attended = self.baseline.forward(query, key, value, causal=True)
        attended = attended.transpose(1, 2).reshape(
            batch_size, sequence_length, self.embed_dim
        )
        return self.output(attended)


class GPTBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int,
        baseline: AttentionBaseline,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(embed_dim)
        self.attention = CausalSelfAttention(embed_dim, num_heads, baseline)
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_ratio * embed_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * embed_dim, embed_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.attention(self.attention_norm(inputs))
        return inputs + self.mlp(self.mlp_norm(inputs))


class TinyGPT(nn.Module):
    def __init__(
        self,
        config: TrainingConfig,
        baseline: AttentionBaseline,
    ) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(
            config.sequence_length, config.embed_dim
        )
        self.block = GPTBlock(
            config.embed_dim,
            config.num_heads,
            config.mlp_ratio,
            baseline,
        )
        self.output_norm = nn.LayerNorm(config.embed_dim)
        self.language_model_head = nn.Linear(
            config.embed_dim, config.vocab_size, bias=False
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        sequence_length = token_ids.shape[1]
        positions = torch.arange(sequence_length, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        hidden = self.block(hidden)
        return self.language_model_head(self.output_norm(hidden))


def _gradient_norm(model: nn.Module) -> float:
    squared_norm = sum(
        parameter.grad.detach().float().square().sum().item()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    return math.sqrt(squared_norm)


def _parameter_difference(first: nn.Module, second: nn.Module) -> float:
    return max(
        (
            (first_parameter.detach().float() - second_parameter.detach().float())
            .abs()
            .max()
            .item()
        )
        for first_parameter, second_parameter in zip(
            first.parameters(), second.parameters(), strict=True
        )
    )


def run_training_comparison(
    config: TrainingConfig,
    custom_baseline: AttentionBaseline,
    reference_baseline: AttentionBaseline,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[TrainingStep]:
    """Train matched tiny GPT blocks on one deterministic next-token batch."""
    torch.manual_seed(config.seed)
    custom_model = TinyGPT(config, custom_baseline).to(device=device, dtype=dtype)
    reference_model = copy.deepcopy(custom_model)
    reference_model.block.attention.baseline = reference_baseline
    custom_optimizer = torch.optim.AdamW(
        custom_model.parameters(), lr=config.learning_rate
    )
    reference_optimizer = torch.optim.AdamW(
        reference_model.parameters(), lr=config.learning_rate
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 1)
    tokens = torch.randint(
        config.vocab_size,
        (config.batch_size, config.sequence_length + 1),
        generator=generator,
    ).to(device)
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    history: list[TrainingStep] = []

    for step in range(config.steps):
        losses: list[float] = []
        gradient_norms: list[float] = []
        for model, optimizer in (
            (custom_model, custom_optimizer),
            (reference_model, reference_optimizer),
        ):
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = functional.cross_entropy(
                logits.float().reshape(-1, config.vocab_size),
                targets.reshape(-1),
            )
            loss.backward()
            gradient_norm = _gradient_norm(model)
            if not math.isfinite(loss.item()) or not math.isfinite(gradient_norm):
                raise RuntimeError(f"non-finite training value at step {step}")
            optimizer.step()
            losses.append(loss.item())
            gradient_norms.append(gradient_norm)

        history.append(
            TrainingStep(
                step=step,
                custom_loss=losses[0],
                reference_loss=losses[1],
                custom_grad_norm=gradient_norms[0],
                reference_grad_norm=gradient_norms[1],
                parameter_max_abs_difference=_parameter_difference(
                    custom_model, reference_model
                ),
            )
        )

    return history


def training_loss_decreased(history: list[TrainingStep]) -> bool:
    if len(history) < 2:
        raise ValueError("at least two training steps are required")
    window = min(10, max(1, len(history) // 4))
    initial_custom = sum(step.custom_loss for step in history[:window]) / window
    final_custom = sum(step.custom_loss for step in history[-window:]) / window
    initial_reference = sum(
        step.reference_loss for step in history[:window]
    ) / window
    final_reference = sum(
        step.reference_loss for step in history[-window:]
    ) / window
    return final_custom < initial_custom and final_reference < initial_reference


def write_training_results(
    history: list[TrainingStep],
    config: TrainingConfig,
    output_directory: str | Path,
    metadata: dict[str, object],
) -> tuple[Path, Path]:
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    rows = [asdict(step) for step in history]
    csv_path = destination / "training.csv"
    json_path = destination / "training-manifest.json"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    json_path.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "metadata": metadata,
                "history": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path