import torch
import triton
import triton.language as tl

from .backend import require_rocm, synchronize


@triton.jit
def _vector_add_kernel(
    left,
    right,
    output,
    size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    tl.store(
        output + offsets,
        tl.load(left + offsets, mask=mask) + tl.load(right + offsets, mask=mask),
        mask=mask,
    )


def run_triton_probe(size: int = 256) -> dict[str, str | int | bool | None]:
    """Compile and execute a minimal Triton kernel on the active AMD device."""
    if size <= 0:
        raise ValueError("size must be greater than zero")
    capability = require_rocm()
    left = torch.arange(size, device="cuda", dtype=torch.float32)
    right = torch.ones_like(left)
    output = torch.empty_like(left)
    block_size = triton.next_power_of_2(size)
    _vector_add_kernel[(triton.cdiv(size, block_size),)](
        left,
        right,
        output,
        size,
        BLOCK_SIZE=block_size,
    )
    synchronize()
    expected = left + right
    correct = bool(torch.equal(output, expected))
    return {
        "success": correct,
        "size": size,
        "gpu_name": capability.gpu_name,
        "architecture": capability.architecture,
        "hip_version": capability.hip_version,
    }


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Compile and run a minimal Triton kernel on AMD ROCm"
    )
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(run_triton_probe(args.size), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()