"""Phase C follow-on: the native CUDA kernel against the Triton one.

`csrc/tq_dequant.cu` has been compiled and timed. It was built and measured on
an NVIDIA RTX A6000 (sm_86, driver 595.71.05) on 2026-09-05 with CUDA 12.9,
torch 2.13.0+cu129 and triton 3.7.1; `--check` printed CORRECT and the raw
sweeps are in `results/cuda/`. This harness still REFUSES to print a timing it
did not measure: on a host with no CUDA device it prints what is missing and
exits 2 rather than reusing or extrapolating from those numbers.

What it compares, and against which arm:

  torch  - the PyTorch dequantization, identical to `bench_kernel.py::torch_path`
  triton - `kernel.py::tq_dequant`, with `--tf32` off unless you pass it
  cuda   - `csrc/tq_dequant.cu`, an fp32 FMA rotation on the CUDA cores

The CUDA kernel does the rotation in plain fp32, so its honest peer is the
Triton kernel with TF32 OFF - the arm that LOST to PyTorch by 12.6x at the
largest shape (results/phaseC_kernel.json). It is NOT a peer of the TF32
tensor-core arm that produced the README's 6.6x and 34.5x rows, and this script
will not put those numbers in the same table.

Runnable here, with no GPU:

    python bench_cuda_kernel.py --check-bits

That checks the one part of the kernel that is pure arithmetic: the bit
extraction, transcribed from the .cu, against an independent reference packer.
A straddle or end-of-row bug there corrupts output silently, and it is exactly
the kind of thing a speedup table will not show you.
"""
from __future__ import annotations

import argparse
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CU = os.path.join(HERE, "csrc", "tq_dequant.cu")

SUPPORTED = (2, 3, 4, 5, 6, 8)


# --------------------------------------------------------------------------- #
# CPU-only: the kernel's index arithmetic, checked against a reference packer   #
# --------------------------------------------------------------------------- #
def _reference_pack(idx, bit_width):
    """Independent LSB-first bit packer, written from packing.py's stated
    layout rather than from its code, so a shared bug does not cancel out."""
    bits = []
    for v in idx:
        for b in range(bit_width):
            bits.append((v >> b) & 1)
    assert len(bits) % 8 == 0
    return [sum(bits[i + b] << b for b in range(8)) for i in range(0, len(bits), 8)]


def _kernel_extract(packed, bit_width, head_dim):
    """The extraction from csrc/tq_dequant.cu, transcribed line for line:

        const int bit_off = tx * BW;
        const int byte0   = bit_off >> 3;
        const int sh      = bit_off & 7;
        const int b0      = base[byte0];
        const int b1      = ((byte0 + 1) < NB) ? base[byte0 + 1] : 0;
        const int idx     = ((b0 >> sh) | (b1 << (8 - sh))) & ((1 << BW) - 1);

    Two different things can be wrong here, and only one of them changes a
    value. Because `head_dim * BW` is always a whole number of bytes, the last
    index of a row ends exactly on the final byte, so the high byte it would
    read is always masked off - dropping the `< NB` guard therefore produces the
    RIGHT number from an OUT-OF-BOUNDS read. In Python that surfaces as an
    IndexError, which this checker reports; in CUDA it is a silent stray read.
    So the guard is a memory-safety guard, not a correctness one, and the two
    are checked separately below.
    """
    nb = len(packed)
    out = []
    for tx in range(head_dim):
        bit_off = tx * bit_width
        byte0 = bit_off >> 3
        sh = bit_off & 7
        b0 = packed[byte0]
        b1 = packed[byte0 + 1] if (byte0 + 1) < nb else 0
        out.append(((b0 >> sh) | (b1 << (8 - sh))) & ((1 << bit_width) - 1))
    return out


def check_bits(head_dim: int = 128, trials: int = 200) -> int:
    rng = random.Random(0)
    print(f"checking csrc/tq_dequant.cu index extraction, head_dim={head_dim}, "
          f"{trials} random rows per width\n")
    bad = 0
    for bw in SUPPORTED:
        if (head_dim * bw) % 8:
            print(f"  {bw}-bit: SKIPPED (head_dim*bw not a whole number of bytes)")
            continue
        worst = None
        for t in range(trials):
            # First and last trial are the boundary cases: all-zero and all-max
            # indices, which is where an end-of-row read shows up.
            if t == 0:
                idx = [0] * head_dim
            elif t == 1:
                idx = [(1 << bw) - 1] * head_dim
            else:
                idx = [rng.randrange(1 << bw) for _ in range(head_dim)]
            try:
                got = _kernel_extract(_reference_pack(idx, bw), bw, head_dim)
            except IndexError:
                worst = "reads past the end of the packed row"
                break
            if got != idx:
                worst = f"wrong value at index " + str(
                    next(i for i, (a, b) in enumerate(zip(idx, got)) if a != b))
                break
        if worst is None:
            print(f"  {bw}-bit: OK  ({head_dim * bw // 8} B/vector, "
                  f"{'straddles byte boundaries' if 8 % bw else 'byte-aligned'})")
        else:
            print(f"  {bw}-bit: MISMATCH - {worst}")
            bad += 1
    print()
    if bad:
        print("BIT EXTRACTION FAILED")
        return 1
    print("BIT EXTRACTION VERIFIED")
    print("\nNOTE: this verifies the kernel's INDEX MATH only, and nothing here\n"
          "      says anything about its speed. The kernel's measured timings\n"
          "      were taken on an RTX A6000 and live in results/cuda/; to take\n"
          "      them again, use --check --bench on a CUDA host.")
    return 0


# --------------------------------------------------------------------------- #
# GPU path: refuses to guess                                                   #
# --------------------------------------------------------------------------- #
def _require_cuda() -> None:
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
          "PATH and a CUDA-enabled torch build. The README's numbers were taken\n"
          "on an RTX A6000 (sm_86); that is the card to reproduce on.\n"
          "Run --check-bits for the part that works without a GPU.\n"
          "\nNo estimated, extrapolated or simulated timing is printed. The\n"
          "kernel's measured numbers are in results/cuda/, taken on an RTX A6000;\n"
          "nothing may be quoted for THIS host until --check --bench has actually\n"
          "run on it.")
    sys.exit(2)


def _load_extension():
    from torch.utils.cpp_extension import load

    return load(name="tq_dequant_cuda_ext", sources=[CU], verbose=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-bits", action="store_true",
                    help="CPU-only: verify the kernel's bit extraction")
    ap.add_argument("--check", action="store_true",
                    help="GPU: build the kernel and compare it to the torch path")
    ap.add_argument("--bench", action="store_true",
                    help="GPU: time torch vs triton vs cuda over the shape sweep")
    ap.add_argument("--bw", type=int, default=6)
    ap.add_argument("--tf32", action="store_true",
                    help="let the TRITON arm use tensor cores (the CUDA arm is "
                         "fp32 either way, so this makes the table apples to "
                         "oranges; off by default on purpose)")
    a = ap.parse_args()

    if not (a.check_bits or a.check or a.bench):
        ap.print_help()
        return 1
    if a.check_bits and not (a.check or a.bench):
        return check_bits()

    _require_cuda()  # exits 2 on this host

    # --- everything below has NEVER been executed -------------------------- #
    import statistics

    import torch

    sys.path.insert(0, HERE)
    from cache import TurboQuantMSE
    from kernel import tq_dequant
    from packing import pack, packed_bytes, unpack

    ext = _load_extension()
    dev, D, bw = "cuda", 128, a.bw
    torch.manual_seed(0)
    tq = TurboQuantMSE(bw, D, dev)
    NB = packed_bytes(D, bw)

    @torch.no_grad()
    def torch_path(packed, norms, out):
        """Identical to bench_kernel.py::torch_path."""
        idx = unpack(packed, bw, D)
        y = tq.centroids.index_select(
            0, idx.reshape(-1, D).reshape(-1).to(torch.int32)).view(-1, D)
        x = y @ tq.Pi
        x *= norms.reshape(-1, 1).to(x.dtype)
        out.copy_(x.view(out.shape))
        return out

    def cuda_path(packed, norms, out, N, L, maxlen):
        return ext.tq_dequant_cuda(packed.contiguous(), norms, tq.centroids,
                                   tq.Pi, out, bw, D, N, L, maxlen)

    def timed(fn, inner, warmup=10, reps=5):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        out = []
        for _ in range(reps):
            s.record()
            for _ in range(inner):
                fn()
            e.record(); e.synchronize()
            out.append(s.elapsed_time(e) / 1e3 / inner)
        return min(out)

    H = 4
    shapes = [(B, H, L) for B in (1, 4, 16)
              for L in (1, 8, 64, 128, 512, 2048, 8192, 16384)]
    print(f"bit_width={bw} head_dim={D} packed={NB} B/vector | "
          f"triton tf32={a.tf32}, cuda arm is fp32\n")
    hdr = (f"{'B':>4}{'L':>7}{'N':>9}{'torch us':>10}{'triton us':>11}"
           f"{'cuda us':>10}{'cuda/torch':>11}{'cuda/triton':>12}{'max|err|':>11}")
    print(hdr); print("-" * len(hdr))

    worst = 0.0
    for (B, H_, L) in shapes:
        N = B * H_ * L
        x = torch.randn(B, H_, L, D, device=dev)
        idx, norms = tq.quantize(x)
        packed = pack(idx, bw)
        nrm16 = norms.to(torch.float16)
        ref = torch.empty(B, H_, L, D, dtype=torch.bfloat16, device=dev)
        got = torch.empty_like(ref)

        # Correctness gate BEFORE any timing: a speedup on a kernel that
        # computes something else is not a speedup.
        torch_path(packed, nrm16, ref)
        cuda_path(packed, nrm16, got, N, L, L)
        err = (got.float() - ref.float()).abs().max().item()
        worst = max(worst, err)
        if not a.bench:
            print(f"{B:>4}{L:>7}{N:>9}{'':>10}{'':>11}{'':>10}{'':>11}{'':>12}"
                  f"{err:>11.2e}")
            continue

        inner = 20 if N <= 262144 else 5
        t_t = timed(lambda: torch_path(packed, nrm16, ref), inner)
        t_k = timed(lambda: tq_dequant(packed, nrm16, tq.centroids, tq.Pi, bw, D,
                                       out=got, allow_tf32=a.tf32), inner)
        t_c = timed(lambda: cuda_path(packed, nrm16, got, N, L, L), inner)
        print(f"{B:>4}{L:>7}{N:>9}{t_t*1e6:>10.1f}{t_k*1e6:>11.1f}"
              f"{t_c*1e6:>10.1f}{t_t/t_c:>11.2f}{t_k/t_c:>12.2f}{err:>11.2e}")
        del x, idx, norms, packed, ref, got
        torch.cuda.empty_cache()

    print(f"\nworst absolute error vs the torch path: {worst:.3e}")
    print("CORRECT" if worst < 1e-2 else "ERROR TOO LARGE - do not quote timings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
