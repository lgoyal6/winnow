"""Correctness + benchmark harness for the fused TurboQuant dequantize kernel.

STATUS: the CUDA half of this file has NEVER BEEN RUN. It was written on a
CPU-only host (no `nvidia-smi`, no `nvcc`), so this repo contains NO measured
numbers for `csrc/tq_dequant.cu`. Run this on a CUDA box to produce them; do not
quote a speedup that did not come out of this script.

Three modes:

  --check-math-only   CPU only, no CUDA, no nvcc. Verifies the algebraic claim
                      the kernel rests on -- that folding the per-vector norm
                      into the codebook gather gives bit-comparable results to
                      the existing `TurboQuantMSE.dequantize`. This is the part
                      that CAN be verified without a GPU, and it is.

  --check            Builds the extension and checks the kernel's output
                     against `TurboQuantMSE.dequantize` on real KV-shaped
                     tensors. Requires CUDA + nvcc.

  --bench            Times three implementations at KV-cache decode shapes:
                       torch-eager : the current dequantize (gather -> GEMM -> scale)
                       fused-cuda  : csrc/tq_dequant.cu gather+scale -> one GEMM
                       triton      : ONLY if a Triton implementation of THIS
                                     path exists; turboquant-poc has none
                                     (turboquant_poc.py says so in its module
                                     docstring: "no triton, no custom CUDA, no
                                     bit-packing"), so that row is reported as
                                     ABSENT rather than invented. turboquant_kv
                                     does have a Triton kernel, but it reads
                                     bit-packed cache buffers rather than the
                                     unpacked uint8 idx used here, so it is not
                                     a drop-in comparison for this harness.
                     Requires CUDA.

On a host that cannot run a mode, this prints what is missing and exits 2. It
never falls back to an estimated, extrapolated or simulated number, and it never
imports torch before it has said so - the same contract, and the same shape of
refusal, as `turboquant_kv/bench_cuda_kernel.py`.

Usage:
    python turboquant-poc/bench_tq_dequant.py --check-math-only
    python turboquant-poc/bench_tq_dequant.py --check --bench
"""

from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.join(HERE, "csrc", "tq_dequant.cu")

# (rows, head_dim) at decode: rows = batch * num_kv_heads * seq_len.
# Qwen2.5-14B: 8 KV heads, head_dim 128, 48 layers.
SHAPES = [
    (8 * 1024, 128),    # 1k context
    (8 * 8192, 128),    # 8k context
    (8 * 32768, 128),   # 32k context
]


# --------------------------------------------------------------------------- #
# Prerequisites: refuse with a message, do not crash and do not guess           #
# --------------------------------------------------------------------------- #
def _turboquant_mse():
    """Import the POC quantizer on demand.

    `turboquant_poc` pulls in torch, numpy, scipy and transformers, so importing
    it - or torch - at module scope would turn "this host cannot run the kernel"
    into a bare ModuleNotFoundError before argparse ever ran, which is the one
    thing the refusals below exist to prevent.
    """
    sys.path.insert(0, HERE)
    from turboquant_poc import TurboQuantMSE

    return TurboQuantMSE


def _require_torch() -> None:
    """--check-math-only needs no GPU, but it is torch algebra, so it does need
    torch. Say so and exit 2 rather than surfacing an import traceback."""
    try:
        import torch  # noqa: F401
    except ImportError:
        print("BLOCKED: --check-math-only cannot run here.")
        print("  - torch is not installed")
        print("\nPrerequisite: a CPU torch build is enough for this mode; no GPU\n"
              "and no nvcc are needed. turboquant_poc.py also imports numpy,\n"
              "scipy and transformers.\n"
              "\nNo estimated, extrapolated or simulated result is printed.")
        sys.exit(2)


def _require_cuda() -> None:
    """--check / --bench build and time csrc/tq_dequant.cu, which needs a CUDA
    device and nvcc. Both are checked, so a host with neither still gets the
    message rather than a traceback."""
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch is not installed")
    else:
        if not torch.cuda.is_available():
            missing.append("torch.cuda.is_available() is False")
    from shutil import which
    if which("nvcc") is None:
        missing.append("nvcc is not on PATH")
    if not missing:
        return
    print("BLOCKED: csrc/tq_dequant.cu cannot be built or timed here.")
    for m in missing:
        print(f"  - {m}")
    print("\nPrerequisite: an NVIDIA GPU with a matching CUDA toolkit (nvcc) on\n"
          "PATH and a CUDA-enabled torch build.\n"
          "Run --check-math-only for the part that works on CPU.\n"
          "\nNo estimated, extrapolated or simulated timing is printed. This\n"
          "kernel has never been run on any host, so unlike turboquant_kv there\n"
          "is not even a measured number elsewhere to misquote; nothing may be\n"
          "said about its speed until --check --bench has actually run.")
    sys.exit(2)


def _reference_fused(tq, idx, norms):
    """What the kernel computes, expressed in torch: fold the norm into the
    gather, then a single GEMM. Used to check the algebra without a GPU."""
    y_scaled = tq.centroids[idx.long()] * norms.reshape(-1, 1)
    return y_scaled @ tq.Pi


def check_math_only(bit_width: int = 4, head_dim: int = 128, rows: int = 4096) -> int:
    import torch

    TurboQuantMSE = _turboquant_mse()
    torch.manual_seed(0)
    tq = TurboQuantMSE(bit_width=bit_width, head_dim=head_dim, device="cpu")
    x = torch.randn(rows, head_dim)
    idx, norms = tq.quantize(x)

    baseline = tq.dequantize(idx, norms)          # gather -> GEMM -> scale
    fused = _reference_fused(tq, idx, norms)      # (gather * scale) -> GEMM

    abs_err = (baseline - fused).abs().max().item()
    scale = baseline.abs().max().item()
    rel_err = abs_err / scale if scale else 0.0
    print(f"rows={rows} head_dim={head_dim} bit_width={bit_width}")
    print(f"  max |baseline - fused| = {abs_err:.3e}   (relative {rel_err:.3e})")
    print(f"  float32 eps            = {torch.finfo(torch.float32).eps:.3e}")
    ok = rel_err < 1e-6
    print("  ALGEBRA VERIFIED" if ok else "  ALGEBRA MISMATCH")
    print("\nNOTE: this verifies the kernel's MATH only. csrc/tq_dequant.cu itself "
          "is UNRUN;\n      no GPU was available. Use --check --bench on a CUDA host.")
    return 0 if ok else 1


def _load_extension():
    from torch.utils.cpp_extension import load

    return load(name="tq_dequant_ext", sources=[CSRC], verbose=True)


def check(ext, bit_width: int, head_dim: int, rows: int) -> None:
    import torch

    TurboQuantMSE = _turboquant_mse()
    tq = TurboQuantMSE(bit_width=bit_width, head_dim=head_dim, device="cuda")
    x = torch.randn(rows, head_dim, device="cuda")
    idx, norms = tq.quantize(x)
    baseline = tq.dequantize(idx, norms)
    y = ext.tq_dequant_gather_scale(
        idx.reshape(-1, head_dim).contiguous(), tq.centroids,
        norms.reshape(-1).float().contiguous(), torch.float32)
    fused = y @ tq.Pi
    err = (baseline - fused).abs().max().item()
    print(f"  check rows={rows}: max abs err = {err:.3e}")
    assert err < 1e-4, f"kernel disagrees with TurboQuantMSE.dequantize: {err}"


def _time(fn, iters: int = 50, warmup: int = 10) -> float:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def bench(ext, bit_width: int) -> None:
    import torch

    TurboQuantMSE = _turboquant_mse()
    print(f"\n{'rows':>9} {'dim':>5} {'torch-eager ms':>15} {'fused-cuda ms':>14} "
          f"{'speedup':>8} {'triton ms':>10}")
    for rows, dim in SHAPES:
        tq = TurboQuantMSE(bit_width=bit_width, head_dim=dim, device="cuda")
        x = torch.randn(rows, dim, device="cuda")
        idx, norms = tq.quantize(x)
        flat_idx = idx.reshape(-1, dim).contiguous()
        flat_norms = norms.reshape(-1).float().contiguous()

        eager = _time(lambda: tq.dequantize(idx, norms))
        fused = _time(lambda: ext.tq_dequant_gather_scale(
            flat_idx, tq.centroids, flat_norms, torch.float32) @ tq.Pi)
        print(f"{rows:>9} {dim:>5} {eager:>15.3f} {fused:>14.3f} "
              f"{eager / fused:>7.2f}x {'ABSENT':>10}")
    print("\ntriton column is ABSENT because turboquant-poc has no Triton "
          "implementation of\nthis dequantize path (turboquant_poc.py: 'no triton, "
          "no custom CUDA'). The\nTriton kernel in turboquant_kv reads bit-packed "
          "cache buffers, not the unpacked\nuint8 idx this harness uses, so it is not "
          "a drop-in comparison.\nNothing is estimated in its place.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-math-only", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--bit-width", type=int, default=4)
    args = ap.parse_args()

    if args.check_math_only:
        _require_torch()  # exits 2 on a host with no torch
        return check_math_only(bit_width=args.bit_width)

    if not (args.check or args.bench):
        ap.error("pick --check-math-only, --check, and/or --bench")

    _require_cuda()  # exits 2 on a host with no CUDA device or no nvcc

    ext = _load_extension()
    if args.check:
        for rows, dim in SHAPES:
            check(ext, args.bit_width, dim, rows)
    if args.bench:
        bench(ext, args.bit_width)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
