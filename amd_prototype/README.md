# FlashAttention AMD Prototype

This directory is a self-contained ROCm/Triton project. It neither imports from
nor modifies the NVIDIA implementation in the parent directory. Its Python
package is named `flash_attention_amd_prototype` to prevent import collisions.

The custom kernel is **FA3-inspired**, not a direct FlashAttention-3 port.
Online softmax, tiled IO, backward recomputation, and exclusive gradient
ownership are portable ideas. Hopper WGMMA, TMA, warpgroup scheduling, and its
FP8 layouts are not AMD features and are not claimed here.

## Status

The independent package, ROCm capability gate, tiled forward/backward kernels,
mathematical reference, baselines, benchmark harness, reporting, and training
validation are implemented. Kernel execution and performance remain
**unverified until run on the target hardware**.

| Target                   | Architecture               | Implementation               | Hardware validation      |
| ------------------------ | -------------------------- | ---------------------------- | ------------------------ |
| AMD Instinct MI200 class | `gfx90a`                   | Conservative launch profile  | Pending                  |
| AMD Instinct MI300 class | `gfx940`/`gfx941`/`gfx942` | Conservative launch profile  | Pending                  |
| Other AMD ROCm GPUs      | Reported dynamically       | Generic conservative profile | Unsupported until tested |

Exact PyTorch, ROCm, Triton, GPU, and driver versions belong in each benchmark
manifest. They will be pinned here only after a working target environment is
recorded.

## Scope

- Self-attention tensors in `[batch, heads, sequence, head_dim]` layout.
- FP16 and BF16 with head dimensions 16, 32, 64, or 128.
- Causal and non-causal forward passes.
- First-order custom autograd for dQ, dK, and dV.
- MI200 and MI300 as separate tuning targets.
- Forced PyTorch math, CK, and AOTriton SDPA comparisons without silent
  fallback.
- Optional external FlashAttention-2 ROCm comparison.

Cross-attention, arbitrary masks, dropout, variable-length attention, GQA,
FP8, and second-order gradients are outside this phase.

## Environment

Use a Linux host with a functioning ROCm installation and supported AMD GPU.
Install the PyTorch ROCm wheel selected for that ROCm release from the official
PyTorch installation matrix first. Then install this project from this
directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the matching PyTorch ROCm build before the editable package.
python -m pip install -e ".[test,evaluation]"
```

Do not reuse the parent NVIDIA virtual environment. Hardware execution is not
supported from native Windows; perform it on the intended Linux/ROCm instance.

Inspect capabilities without launching an attention kernel:

```bash
flash-attention-amd-env
```

The report includes `torch.version.hip`, the active `gcnArchName`, Triton,
BF16 support, CK/AOTriton availability, and the reason the custom path is
disabled when prerequisites are missing.

Before attention validation, compile and execute the minimal Triton probe:

```bash
flash-attention-amd-probe
# Equivalent:
python -m flash_attention_amd_prototype.environment --probe-triton
```

## Correctness model

The custom forward kernel applies tiled online softmax without materializing
the full attention matrix. It saves row log-sum-exp values in base 2. Backward
rematerializes probability tiles, computes `D = rowsum(dO * O)`, and uses
separate dQ and dK/dV kernels so each program owns its output tile.

The package also includes an ordinary PyTorch tiled implementation for a
float64 mathematical oracle. On target hardware, custom outputs and gradients
must be compared against forced PyTorch math SDPA before performance results
are accepted.

## Benchmarking

Run from this directory after the environment probe succeeds:

```bash
python benchmark.py --quick
python benchmark.py \
	--sequences 1024 2048 4096 8192 \
	--dtype bfloat16 \
	--baselines custom_triton_amd pytorch_math rocm_ck rocm_aotriton \
	--passes forward backward
```

Every unavailable baseline produces an explicit reason. CK and AOTriton are
selected through the ROCm SDPA preference API while flash attention is forced
as the only allowed SDPA implementation. The harness records first-iteration
cost separately from steady-state median, p90, standard deviation, throughput,
peak memory, architecture, and launch metadata.

Sweep the bounded launch candidates separately on each GPU generation:

```bash
flash-attention-amd-tune --sequence 2048 --head-dim 64 --dtype bfloat16
```

The sweep records successful and failed configurations and writes
`winners.json`. A winner from one architecture must not be promoted as the
default for another architecture without repeating the sweep there.

Generate a report from a result set:

```bash
python evaluate_results.py results/<run>/results.csv \
	--output results/<run>/evaluation --plot
```

## Training validation

After forward and gradient parity pass on the target GPU:

```bash
python training_validation.py --quick
python training_validation.py --steps 100 --dtype bfloat16
```

This trains matched tiny GPT blocks using the custom implementation and forced
math SDPA, rejects non-finite losses or gradient norms, and stores both traces.

## Hardware acceptance sequence

1. Save `flash-attention-amd-env --json results/environment.json`.
2. Run `flash-attention-amd-probe`.
3. Run CPU reference and mocked capability tests.
4. Run ROCm output and gradient parity for each supported dtype and head size.
5. Run the quick benchmark and training validation.
6. Run fixed benchmark manifests separately on MI200 and MI300.
7. Record winning launch configurations only after repeated measurements.

No performance claim should be made from the conservative launch profiles.
Vendor CK or AOTriton may remain faster; those results are first-class evidence,
not failures of the educational objective.
