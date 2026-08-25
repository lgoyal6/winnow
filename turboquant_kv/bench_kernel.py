"""Phase C gate: correctness first, then the shape range where the kernel wins.

Order matters. A speedup on a kernel that computes something slightly different
is not a speedup, so every shape is checked against the PyTorch path before any
timing is reported, and the sweep prints the worst error it saw alongside the
speedups rather than in a separate section that is easy to skim past.

The losing region is the point of the sweep, not an embarrassment to be avoided:
step 1 measured a 7.5-12 us per-launch floor on this card, so any shape whose
useful work is under roughly 10 us cannot win no matter how good the kernel is.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import TurboQuantMSE          # noqa: E402
from kernel import tq_dequant            # noqa: E402
from packing import pack, packed_bytes, unpack   # noqa: E402


def timed(fn, inner=20, warmup=10, reps=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(reps):
        s.record()
        for _ in range(inner):
            fn()
        e.record(); e.synchronize()
        out.append(s.elapsed_time(e) / 1e3 / inner)
    return min(out), statistics.median(out)


@torch.no_grad()
def torch_path(tq, packed, norms, bw, D, out):
    """The current PyTorch dequantization, as implemented in cache.py."""
    idx = unpack(packed, bw, D)
    flat = idx.reshape(-1, D)
    y = tq.centroids.index_select(0, flat.reshape(-1).to(torch.int32)).view(-1, D)
    x = y @ tq.Pi
    x *= norms.reshape(-1, 1).to(x.dtype)
    out.copy_(x.view(out.shape))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bw", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--out", default="results/phaseC_kernel.json")
    ap.add_argument("--block-n", type=int, default=64)
    ap.add_argument("--tf32", action="store_true",
                    help="let the rotation use TF32 tensor cores")
    a = ap.parse_args()

    dev = "cuda"
    D, bw = a.head_dim, a.bw
    torch.manual_seed(0)
    tq = TurboQuantMSE(bw, D, dev)
    NB = packed_bytes(D, bw)

    # Real decode geometry for Qwen2.5-7B: 4 KV heads, head_dim 128.
    # N = batch * kv_heads * cache_len, swept from a single decode step at
    # batch 1 all the way to a 16k cache at batch 16.
    H = 4
    shapes = []
    for B in (1, 4, 16):
        for L in (1, 8, 64, 128, 512, 2048, 8192, 16384):
            shapes.append((B, H, L))

    rows = []
    print(f"bit_width={bw} head_dim={D} packed={NB} B/vector  "
          f"(fp16 would be {2*D} B/vector)\n")
    hdr = (f"{'B':>4}{'L':>7}{'N':>9}{'torch us':>10}{'triton us':>11}"
           f"{'speedup':>9}{'t GB/s':>9}{'k GB/s':>9}{'max|err|':>11}  who")
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

        torch_path(tq, packed, nrm16, bw, D, ref)
        tq_dequant(packed, nrm16, tq.centroids, tq.Pi, bw, D, out=got,
                   block_n=a.block_n, allow_tf32=a.tf32)
        err = (got.float() - ref.float()).abs().max().item()
        worst = max(worst, err)

        inner = 20 if N <= 262144 else 5
        t_t, _ = timed(lambda: torch_path(tq, packed, nrm16, bw, D, ref), inner=inner)
        t_k, _ = timed(lambda: tq_dequant(packed, nrm16, tq.centroids, tq.Pi,
                                          bw, D, out=got, block_n=a.block_n,
                                          allow_tf32=a.tf32),
                       inner=inner)
        # Compulsory traffic: read packed indices + norms, write bf16 output.
        moved = N * (NB + 2) + N * D * 2
        flops = 2.0 * N * D * D          # the (N,D)@(D,D) inverse rotation
        rows.append({
            "batch": B, "cache_len": L, "N": N,
            "torch_us": t_t * 1e6, "triton_us": t_k * 1e6,
            "speedup": t_t / t_k,
            "torch_gbs": moved / t_t / 1e9, "triton_gbs": moved / t_k / 1e9,
            "max_abs_err": err,
            "flops": flops,
            "arith_intensity": flops / moved,
            "torch_tflops": flops / t_t / 1e12,
            "triton_tflops": flops / t_k / 1e12,
        })
        who = "triton" if t_k < t_t else "TORCH WINS"
        print(f"{B:>4}{L:>7}{N:>9}{t_t*1e6:>10.1f}{t_k*1e6:>11.1f}"
              f"{t_t/t_k:>9.2f}{moved/t_t/1e9:>9.1f}{moved/t_k/1e9:>9.1f}"
              f"{err:>11.2e}  {who}")
        del x, idx, norms, packed, ref, got
        torch.cuda.empty_cache()

    # Which ceiling is actually binding? Step 1 measured this card at
    # 711 GB/s, 23.9 TFLOP/s fp32, 65.1 TFLOP/s tf32, 128.5 TFLOP/s bf16.
    big = max(rows, key=lambda r: r["N"])
    print(f"\nat the largest shape (N={big['N']}): "
          f"AI = {big['arith_intensity']:.0f} FLOP/byte")
    for name, peak in (("fp32", 23.9), ("tf32", 65.1), ("bf16", 128.5)):
        print(f"  ridge point in {name}: {peak*1e12/711e9:6.1f} FLOP/byte  "
              f"-> {'COMPUTE' if big['arith_intensity'] > peak*1e12/711e9 else 'memory'}"
              f"-bound in {name}")
    print(f"  torch achieves {big['torch_tflops']:.2f} TFLOP/s / "
          f"{big['torch_gbs']:.0f} GB/s")
    print(f"  triton achieves {big['triton_tflops']:.2f} TFLOP/s / "
          f"{big['triton_gbs']:.0f} GB/s")

    print(f"\nworst absolute error across all shapes: {worst:.3e}")
    losers = [r for r in rows if r["speedup"] < 1.0]
    if losers:
        big = max(r["N"] for r in losers)
        print(f"PyTorch wins at N <= {big} "
              f"({len(losers)}/{len(rows)} shapes); crossover is between "
              f"N={big} and N="
              f"{min((r['N'] for r in rows if r['N'] > big), default='-')}")
    else:
        print("no shape in the sweep where PyTorch wins")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"bit_width": bw, "head_dim": D, "rows": rows,
                   "worst_abs_err": worst}, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
