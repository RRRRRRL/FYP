"""
====================================================================
FLASH ATTENTION IMPLEMENTATION - FINAL YEAR PROJECT
====================================================================
Compares Standard (naive) Attention vs Flash Attention forward pass
on NVIDIA RTX 5060.

Includes:
    1. Standard Attention implementation
    2. Flash Attention (Triton kernel) implementation
    3. Correctness verification
    4. Benchmarking (latency + memory)
    5. Visualization (plots for report)

Reference:
    Dao et al., "FlashAttention: Fast and Memory-Efficient Exact 
    Attention with IO-Awareness", 2022. https://arxiv.org/abs/2205.14135

Author: <Your Name>
Course: <Your Course / FYP Title>
====================================================================
"""

import math
import time
import argparse

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate


# ====================================================================
# SECTION 1: STANDARD ATTENTION
# ====================================================================

def standard_attention(Q, K, V, mask=None):
    """
    Standard scaled dot-product attention: Softmax(QK^T / sqrt(d)) V

    Materializes the full (N x N) attention matrix -> O(N^2) memory.

    Args:
        Q, K, V: (batch, heads, seq_len, head_dim)
        mask: optional attention mask

    Returns:
        output: (batch, heads, seq_len, head_dim)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output


def standard_attention_with_memory_tracking(Q, K, V, mask=None):
    """Runs standard attention and reports peak GPU memory used (MB)."""
    torch.cuda.reset_peak_memory_stats()
    output = standard_attention(Q, K, V, mask)
    torch.cuda.synchronize()
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return output, peak_mem


# ====================================================================
# SECTION 2: FLASH ATTENTION (Triton kernel)
# ====================================================================

@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, O,
    L,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    scale,
):
    """
    Flash Attention forward kernel using tiling + online softmax.
    Each program instance processes one BLOCK_M query block for one
    (batch, head) pair, streaming over K/V blocks without ever
    materializing the full N x N score matrix.
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    Q += off_z * stride_qb + off_h * stride_qh
    K += off_z * stride_kb + off_h * stride_kh
    V += off_z * stride_vb + off_h * stride_vh
    O += off_z * stride_ob + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        cur_offs_n = start_n + offs_n

        k_ptrs = K + cur_offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=cur_offs_n[:, None] < N_CTX, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * scale
        mask = cur_offs_n[None, :] < N_CTX
        qk = tl.where(mask, qk, -float("inf"))

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = V + cur_offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=cur_offs_n[:, None] < N_CTX, other=0.0)

        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]

    o_ptrs = O + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N_CTX)

    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, m_i + tl.log(l_i), mask=offs_m < N_CTX)


def flash_attention(Q, K, V, block_m=64, block_n=64):
    """
    Flash Attention forward pass wrapper.

    Args:
        Q, K, V: (batch, heads, seq_len, head_dim), contiguous,
                 fp16/bf16 recommended for best performance.

    Returns:
        output: (batch, heads, seq_len, head_dim)
    """
    B, H, N, D = Q.shape
    assert D in (16, 32, 64, 128), "head_dim should be a power of 2 (16/32/64/128)"

    O = torch.empty_like(Q)
    L = torch.empty((B * H, N), device=Q.device, dtype=torch.float32)

    scale = 1.0 / math.sqrt(D)
    grid = (triton.cdiv(N, block_m), B * H)

    _flash_attn_fwd_kernel[grid](
        Q, K, V, O, L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        B, H, N,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_DMODEL=D,
        scale=scale,
    )

    return O


def flash_attention_with_memory_tracking(Q, K, V, block_m=64, block_n=64):
    """Runs flash attention and reports peak GPU memory used (MB)."""
    torch.cuda.reset_peak_memory_stats()
    output = flash_attention(Q, K, V, block_m, block_n)
    torch.cuda.synchronize()
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return output, peak_mem


# ====================================================================
# SECTION 3: CORRECTNESS CHECK
# ====================================================================

def check_correctness(seq_len=1024, batch=2, heads=8, head_dim=64, device="cuda"):
    """
    Verify Flash Attention output numerically matches standard attention.
    Flash Attention is a mathematically EXACT algorithm (not an
    approximation), so outputs should match within fp16 tolerance.
    """
    torch.manual_seed(0)
    Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
    K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
    V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)

    with torch.no_grad():
        out_std = standard_attention(Q.float(), K.float(), V.float()).half()
        out_flash = flash_attention(Q, K, V)

    max_diff = (out_std - out_flash).abs().max().item()
    mean_diff = (out_std - out_flash).abs().mean().item()
    passed = max_diff < 1e-2

    print(f"[Correctness Check] seq_len={seq_len}")
    print(f"  Max abs diff : {max_diff:.6f}")
    print(f"  Mean abs diff: {mean_diff:.6f}")
    print(f"  Result: {'PASSED' if passed else 'FAILED'} (tolerance=1e-2 for fp16)\n")
    return passed


# ====================================================================
# SECTION 4: BENCHMARKING
# ====================================================================

def benchmark_fn(fn, *args, warmup=10, iters=50):
    """Time a GPU function accurately using CUDA events."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters  # ms per iteration


def run_benchmark(
    seq_lengths=(256, 512, 1024, 2048, 4096, 8192),
    batch=4,
    heads=8,
    head_dim=64,
    device="cuda",
    dtype=torch.float16,
):
    """
    Benchmarks Standard vs Flash Attention across sequence lengths.
    Returns a pandas DataFrame with latency, memory, and speedup stats.
    """
    assert torch.cuda.is_available(), "CUDA GPU required (RTX 5060)"
    print(f"Running on: {torch.cuda.get_device_name(0)}\n")

    results = []

    for seq_len in seq_lengths:
        print(f"Benchmarking seq_len={seq_len} ...")
        torch.manual_seed(42)
        Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
        K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
        V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

        # ---- Standard Attention ----
        std_oom = False
        try:
            std_time = benchmark_fn(standard_attention, Q, K, V)
            _, std_mem = standard_attention_with_memory_tracking(Q, K, V)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            std_time, std_mem = float('nan'), float('nan')
            std_oom = True
            print(f"  -> Standard attention OOM at seq_len={seq_len}")

        # ---- Flash Attention ----
        flash_time = benchmark_fn(flash_attention, Q, K, V)
        _, flash_mem = flash_attention_with_memory_tracking(Q, K, V)

        speedup = (std_time / flash_time) if not std_oom else float('inf')
        mem_reduction = (1 - flash_mem / std_mem) * 100 if not std_oom else 100.0

        results.append({
            "Seq Length": seq_len,
            "Standard Time (ms)": round(std_time, 3) if not std_oom else "OOM",
            "Flash Time (ms)": round(flash_time, 3),
            "Speedup (x)": round(speedup, 2) if not std_oom else "N/A",
            "Standard Mem (MB)": round(std_mem, 2) if not std_oom else "OOM",
            "Flash Mem (MB)": round(flash_mem, 2),
            "Mem Reduction (%)": round(mem_reduction, 2) if not std_oom else "N/A",
        })

        torch.cuda.empty_cache()

    return pd.DataFrame(results)


# ====================================================================
# SECTION 5: VISUALIZATION
# ====================================================================

def plot_latency(df, save_path="plot_latency.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    seq_lens = df["Seq Length"]
    std_times = pd.to_numeric(df["Standard Time (ms)"], errors="coerce")
    flash_times = pd.to_numeric(df["Flash Time (ms)"], errors="coerce")

    ax.plot(seq_lens, std_times, marker='o', label="Standard Attention",
             linewidth=2, color="#d62728")
    ax.plot(seq_lens, flash_times, marker='s', label="Flash Attention",
             linewidth=2, color="#2ca02c")

    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Forward Pass Latency: Standard vs Flash Attention\n(RTX 5060)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def plot_speedup(df, save_path="plot_speedup.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    seq_lens = df["Seq Length"]
    speedup = pd.to_numeric(df["Speedup (x)"], errors="coerce")

    bars = ax.bar(seq_lens.astype(str), speedup, color="#1f77b4")
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label="No speedup baseline")

    for bar, val in zip(bars, speedup):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f"{val:.2f}x", ha='center', fontweight='bold')

    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Speedup Factor (x)")
    ax.set_title("Flash Attention Speedup over Standard Attention\n(RTX 5060)")
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def plot_memory(df, save_path="plot_memory.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    seq_lens = df["Seq Length"]
    std_mem = pd.to_numeric(df["Standard Mem (MB)"], errors="coerce")
    flash_mem = pd.to_numeric(df["Flash Mem (MB)"], errors="coerce")

    x = np.arange(len(seq_lens))
    width = 0.35

    ax.bar(x - width / 2, std_mem, width, label="Standard Attention", color="#d62728")
    ax.bar(x + width / 2, flash_mem, width, label="Flash Attention", color="#2ca02c")

    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title("Peak GPU Memory Usage: Standard vs Flash Attention\n(RTX 5060)")
    ax.set_xticks(x)
    ax.set_xticklabels(seq_lens)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def plot_complexity_comparison(save_path="plot_complexity.png"):
    """Theoretical O(N^2) vs O(N) memory complexity illustration."""
    N = np.array([256, 512, 1024, 2048, 4096, 8192, 16384])
    quadratic = (N ** 2) / (N[0] ** 2)
    linear = N / N[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(N, quadratic, marker='o', label="Standard Attention O(N^2)", color="#d62728")
    ax.plot(N, linear, marker='s', label="Flash Attention O(N)", color="#2ca02c")
    ax.set_xlabel("Sequence Length (N)")
    ax.set_ylabel("Relative Memory Complexity")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title("Theoretical Memory Complexity Comparison")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def generate_all_plots(csv_path="benchmark_results.csv"):
    df = pd.read_csv(csv_path)
    plot_latency(df)
    plot_speedup(df)
    plot_memory(df)
    plot_complexity_comparison()


# ====================================================================
# SECTION 6: MAIN ENTRY POINT
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="Flash Attention FYP Benchmark")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--seq_lengths", type=int, nargs="+",
                         default=[256, 512, 1024, 2048, 4096, 8192])
    parser.add_argument("--csv_out", type=str, default="benchmark_results.csv")
    parser.add_argument("--skip_plots", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("FLASH ATTENTION vs STANDARD ATTENTION BENCHMARK")
    print("Hardware target: NVIDIA RTX 5060")
    print("=" * 60 + "\n")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected. This script requires an NVIDIA GPU.")

    # Step 1: Correctness check
    check_correctness(seq_len=1024, head_dim=args.head_dim)
    check_correctness(seq_len=2048, head_dim=args.head_dim)

    # Step 2: Performance benchmark
    df = run_benchmark(
        seq_lengths=args.seq_lengths,
        batch=args.batch,
        heads=args.heads,
        head_dim=args.head_dim,
    )

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))

    df.to_csv(args.csv_out, index=False)
    print(f"\nResults saved to {args.csv_out}")

    # Step 3: Generate plots
    if not args.skip_plots:
        print("\nGenerating plots...")
        generate_all_plots(args.csv_out)
        print("All plots generated for FYP report.")


if __name__ == "__main__":
    main()