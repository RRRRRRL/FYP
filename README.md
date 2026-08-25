# FlashAttention Forward and Backward Prototype

This project demonstrates the forward and backward passes of FlashAttention on
an NVIDIA GPU using custom Triton kernels. The kernels tile keys and values and
never materialize the full attention or softmax matrices.

It also includes a pure PyTorch tiled implementation used as a readable
mathematical oracle. The reference path runs on CPU, supports float64 gradcheck,
and exposes the saved row normalization and analytical backward stages.

This is an educational prototype, not a replacement for the production kernels
in PyTorch SDPA or the FlashAttention package. It currently supports:

- CUDA `float16` and `bfloat16` tensors
- `[batch, heads, sequence, head_dim]` input layout
- head dimensions 16, 32, 64, and 128
- causal and non-causal self-attention
- first-order PyTorch autograd for query, key, and value

## Environment

Use Linux or WSL2 with an NVIDIA driver, a CUDA-enabled PyTorch build, and a
Triton-compatible Python version. Install PyTorch using the command provided by
the PyTorch installation selector for your CUDA version, then install this
project:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Native Windows Triton support is not provided by the upstream Triton package;
WSL2 is the recommended Windows setup.

Record the active software and GPU capabilities before running experiments:

```bash
flash-attention-env --json results/environment.json
```

The command reports unavailable dependencies instead of failing, so it can also
be used to diagnose an incomplete environment.

## Run

Run the correctness suite against PyTorch scaled dot-product attention:

```bash
pytest -q
```

The CPU mathematical tests can be run independently of CUDA and Triton:

```bash
pytest -q tests/test_reference.py tests/test_environment.py
```

Use the differentiable reference implementation directly:

```python
from flash_attention_prototype import reference_attention

output = reference_attention(query, key, value, causal=True)
output.sum().backward()
```

Run a quick four-baseline forward/backward smoke benchmark:

```bash
python benchmark.py --quick --causal --dtype float16
```

Run the project-statement sequence sweep with the memory-isolation batch size:

```bash
python benchmark.py --batch 1 --heads 8 --head-dim 64 --causal \
	--sequences 1024 2048 4096 8192 16384 \
	--warmup 10 --repetitions 100 --output results/memory-scaling
```

The harness benchmarks custom Triton, explicitly forced PyTorch math SDPA,
official FlashAttention-2, and FlexAttention. Optional unavailable baselines are
recorded with a reason instead of being silently replaced. Forward and backward
latency, effective TFLOP/s, peak allocated/reserved VRAM, arguments, and the
environment report are written to CSV and JSON. Compilation and mask creation
occur before steady-state timing. Forward timing excludes backward, while
backward timing reuses a retained forward graph and measures gradient computation
only. Input allocation is outside both timed regions.

Run the deterministic GPT-style optimizer smoke test:

```bash
python training_validation.py --quick --dtype bfloat16
```

Run the 100-step validation and save its loss, gradient-norm, and parameter
divergence curves:

```bash
python training_validation.py --steps 100 --sequence 128 --batch 4 \
	--output results/training-100
```

The custom and forced-math reference models begin with identical parameters and
train on the same fixed next-token batch. Dropout is intentionally disabled;
deterministic Philox mask replay is outside the current project scope.

## Algorithm

For every query tile, the kernel streams over key/value tiles. It updates the
running row maximum $m$, softmax denominator $l$, and output accumulator $O$
with the numerically stable online-softmax recurrence:

$$
m' = \max(m, \operatorname{rowmax}(QK^T)),
$$

$$
l' = e^{m-m'}l + \operatorname{rowsum}(e^{QK^T-m'}),
$$

$$
O' = e^{m-m'}O + e^{QK^T-m'}V.
$$

The final tile output is $O/l$. Memory usage for attention scores is therefore
linear in sequence length rather than quadratic.

Backward saves only the output and one log-sum-exp value per query row. It
recomputes each probability tile and uses

$$
D_i = \sum_d O_{id}\,dO_{id},
\qquad
dS_{ij} = P_{ij}(dP_{ij} - D_i).
$$

Separate kernels accumulate $dQ$ by query tile and $dK,dV$ by key tile. This
gives each program exclusive ownership of its output and avoids atomic updates.
Second-order gradients are not supported.

The PyTorch reference follows the same equations with explicit Python tile
loops. Its float64 autograd wrapper is the finite-difference gradcheck target;
the FP16/BF16 Triton implementation is validated by direct output and gradient
comparison instead.

## Requirements Traceability

| Project criterion | Evidence command or artifact |
| --- | --- |
| Pure PyTorch forward/backward prototype | `pytest -q tests/test_reference.py` |
| Forward correctness in FP32 | `test_reference_forward_matches_sdpa` |
| Backward correctness and gradcheck | stagewise gradient tests and `test_reference_autograd_gradcheck` |
| Triton FP16/BF16 correctness | `pytest -q -m cuda tests/test_forward.py` |
| Linear forward/backward memory scaling | `benchmark.py` sweep CSV peak-memory columns |
| Four-baseline comparison | benchmark CSV rows for each available adapter |
| Forward/backward TFLOP/s | benchmark CSV `tflops` column and manifest convention |
| Training capability | `training_validation.py --steps 100` loss-curve artifacts |
| Reproducible target environment | `flash-attention-env --json ...` |

The 8K/16K sweeps and 100-step training run are explicit target-GPU acceptance
jobs, not ordinary unit tests. A criterion is not considered validated merely
because its implementation exists; retain the generated manifest as evidence.
