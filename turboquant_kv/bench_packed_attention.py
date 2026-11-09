"""Decode-attention microbenchmark: packed direct vs matched and built-in arms.

    python bench_packed_attention.py --out RUN_DIR [--batches 1 8 32] [--lens 512 4096]
    python bench_packed_attention.py --smoke --out RUN_DIR      # tiny, any device

Arms, all on identical paged inputs for one Qwen2.5-7B layer shape
(Hq=28, Hkv=4, D=128) unless overridden:

  packed_direct_s{1,2,4}  this project's kernel, 1/2/4 KV splits
  packed_reconstruct      existing fused dequantizer per sequence, then SDPA:
                          what the packed cache did before this change
  dense_paged             the same kernel schedule over unquantized bf16 pages
  sdpa_dense              torch SDPA over contiguous bf16 K/V (no paging)
  flashinfer_dense        FlashInfer paged decode, if installed (strong built-in)

Each (shape, repetition) runs every arm in a freshly shuffled order after its
own warmup, timed with CUDA events around `iters` back-to-back calls. Every
raw sample is written to samples.jsonl; summary.json is derived from it by
`report`. Bytes moved use a simple traffic model (KV payload read once) so
achieved bandwidth is comparable across arms; it is a model, not a counter.
Per-arm peak extra allocation is measured with the CUDA caching allocator.

Off CUDA only `--smoke` runs, through the reference implementations, to check
the CLI end to end. Its timings are meaningless and are labelled cpu-smoke.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import packed_attention as pa
from cache import _get_quantizer

try:
    import packed_attention_triton as pt
except Exception:                        # noqa: BLE001
    pt = None
try:
    import flashinfer                    # noqa: F401
except Exception:                        # noqa: BLE001
    flashinfer = None


def build(B, L, Hq, Hkv, D, S, bw, device, seed):
    g = torch.Generator().manual_seed(seed)
    lens = [L] * B
    k = torch.randn(B, Hkv, L, D, generator=g)
    v = torch.randn(B, Hkv, L, D, generator=g)
    q = torch.randn(B, Hq, 1, D, generator=g)
    kq, vq = _get_quantizer(bw, D, "cpu", 42), _get_quantizer(bw, D, "cpu", 42)
    kv = pa.pages_from_dense(k, v, lens, kq, vq, S, generator=g)
    dev = torch.device(device)
    kvd = pa.PackedKV(kv.k_pages.to(dev), kv.k_norms.to(dev), kv.v_pages.to(dev),
                      kv.v_norms.to(dev), kv.page_table.to(dev), kv.seq_lens.to(dev),
                      pa.Quantizer(kq.Pi.to(dev), kq.centroids.to(dev), bw),
                      pa.Quantizer(vq.Pi.to(dev), vq.centroids.to(dev), bw), D)
    # Dense pages hold the *reconstructed* values, so every arm attends to the
    # same numbers and outputs are comparable, not just shapes.
    dt = torch.bfloat16 if dev.type == "cuda" else torch.float32
    P = kv.k_pages.shape[0]
    kd = torch.zeros(P, Hkv, S, D, dtype=dt)
    vd = torch.zeros(P, Hkv, S, D, dtype=dt)
    for b in range(B):
        hist = []
        for pages, norms, quant in ((kv.k_pages, kv.k_norms, kv.k_quant),
                                    (kv.v_pages, kv.v_norms, kv.v_quant)):
            rows, nrm = pa._gather_history(pages, norms, kv, b)
            hist.append((pa._centroids(rows, quant, D) @ quant.Pi) * nrm.float()[..., None])
        for t in range((L + S - 1) // S):
            pid, n = int(kv.page_table[b, t]), min(S, L - t * S)
            kd[pid, :, :n] = hist[0][:, t * S:t * S + n].to(dt)
            vd[pid, :, :n] = hist[1][:, t * S:t * S + n].to(dt)
    dense = pa.DenseKV(kd.to(dev), vd.to(dev), kvd.page_table, kvd.seq_lens, D)
    contiguous = (torch.stack([kd[kv.page_table[b, :(L + S - 1) // S].long()]
                               .permute(1, 0, 2, 3).reshape(Hkv, -1, D)[:, :L] for b in range(B)]).to(dev),
                  torch.stack([vd[kv.page_table[b, :(L + S - 1) // S].long()]
                               .permute(1, 0, 2, 3).reshape(Hkv, -1, D)[:, :L] for b in range(B)]).to(dev))
    return q.to(dev, dt), kvd, dense, contiguous


def traffic_bytes(arm, B, L, Hkv, D, nb):
    """KV payload read once per decode step (model, not a hardware counter)."""
    if arm.startswith("packed"):
        per_tok = 2 * (nb + 2)
        if arm == "packed_reconstruct":       # read packed, write+read bf16 history
            per_tok += 2 * 2 * D * 2
        return B * Hkv * L * per_tok
    return B * Hkv * L * 2 * D * 2


def arms_for(q, kvd, dense, contiguous, scale, S):
    cuda = q.is_cuda
    arms = {}
    if cuda and pt is not None:
        for s in (1, 2, 4):
            arms[f"packed_direct_s{s}"] = lambda s=s: pt.packed_decode_attention(q, kvd, scale, num_splits=s)
        arms["dense_paged"] = lambda: pt.dense_decode_attention(q, dense, scale)
    else:
        arms["packed_direct_ref"] = lambda: pa.tiled_compressed_attention(q, kvd, scale)
    import packed_backend as pb
    arms["packed_reconstruct"] = lambda: pb._reconstruct_sdpa(q, kvd, scale)
    k, v = contiguous
    arms["sdpa_dense"] = lambda: F.scaled_dot_product_attention(
        q, k, v, scale=scale, enable_gqa=q.shape[1] != k.shape[1])
    if cuda and flashinfer is not None:
        arms["flashinfer_dense"] = _flashinfer_arm(q, dense, scale, S)
    return arms


def _flashinfer_arm(q, dense, scale, S):
    import flashinfer
    B, Hq, _, D = q.shape
    Hkv = dense.num_kv_heads
    lens = dense.seq_lens.long()
    npages = (lens + S - 1) // S
    indptr = torch.zeros(B + 1, dtype=torch.int32, device=q.device)
    indptr[1:] = torch.cumsum(npages, 0)
    indices = torch.cat([dense.page_table[b, :int(npages[b])] for b in range(B)]).int()
    last = (lens - (npages - 1) * S).int()
    ws = torch.empty(128 * 2**20, dtype=torch.uint8, device=q.device)
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "HND")
    w.plan(indptr, indices, last, Hq, Hkv, D, S, pos_encoding_mode="NONE",
           q_data_type=q.dtype, kv_data_type=dense.k_pages.dtype, sm_scale=scale)
    qq = q[:, :, 0].contiguous()
    kv = (dense.k_pages, dense.v_pages)
    return lambda: w.run(qq, kv)


def time_arm(fn, iters, cuda):
    if cuda:
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1e3 / iters, torch.cuda.max_memory_allocated() - base
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1e6 / iters, 0


def check_outputs(arms, q, kvd, scale):
    """Every arm must produce the same attention (within bf16), or it is not a
    comparison. Returns max abs error per arm against the FP32 reference."""
    ref = pa.tiled_compressed_attention(q.float(), kvd, scale)
    errs = {}
    for name, fn in arms.items():
        out = fn()
        out = out.view(ref.shape) if out.numel() == ref.numel() else out
        errs[name] = (out.float().reshape(ref.shape) - ref).abs().max().item()
    return errs


def run(args):
    cuda = torch.cuda.is_available()
    if not cuda and not args.smoke:
        raise SystemExit("BLOCKED: timing runs need CUDA; use --smoke for a CLI check")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    device = "cuda" if cuda else "cpu"
    env = {"torch": torch.__version__, "python": platform.python_version(),
           "device": torch.cuda.get_device_name() if cuda else "cpu-smoke",
           "cuda": torch.version.cuda, "triton": getattr(sys.modules.get("triton"), "__version__", None),
           "flashinfer": getattr(flashinfer, "__version__", None) if flashinfer else None,
           "argv": sys.argv}
    (out / "env.json").write_text(json.dumps(env, indent=2) + "\n")
    rng = random.Random(args.seed)
    nb = args.head_dim * args.bit_width // 8
    with (out / "samples.jsonl").open("w") as fh:
        for B in args.batches:
            for L in args.lens:
                q, kvd, dense, contig = build(B, L, args.hq, args.hkv, args.head_dim,
                                              args.page_size, args.bit_width, device, args.seed)
                scale = pa.default_scale(args.head_dim)
                arms = arms_for(q, kvd, dense, contig, scale, args.page_size)
                errs = check_outputs(arms, q, kvd, scale)
                for name, fn in arms.items():
                    for _ in range(args.warmup):
                        fn()
                for rep in range(args.reps):
                    order = list(arms)
                    rng.shuffle(order)
                    for name in order:
                        us, extra = time_arm(arms[name], args.iters, cuda)
                        fh.write(json.dumps({
                            "B": B, "L": L, "arm": name, "rep": rep, "position": order.index(name),
                            "us_per_call": us, "peak_extra_bytes": extra,
                            "model_bytes": traffic_bytes(name, B, L, args.hkv, args.head_dim, nb),
                            "max_abs_err_vs_fp32_ref": errs[name]}) + "\n")
                del q, kvd, dense, contig, arms
                if cuda:
                    torch.cuda.empty_cache()
    report(out)


def report(out):
    out = Path(out)
    rows = [json.loads(x) for x in (out / "samples.jsonl").read_text().splitlines()]
    groups = {}
    for r in rows:
        groups.setdefault((r["B"], r["L"], r["arm"]), []).append(r)
    summary = []
    for (B, L, arm), rs in sorted(groups.items()):
        us = sorted(r["us_per_call"] for r in rs)
        med = statistics.median(us)
        summary.append({"B": B, "L": L, "arm": arm, "reps": len(us), "median_us": med,
                        "min_us": us[0], "max_us": us[-1],
                        "achieved_GBps_model": rs[0]["model_bytes"] / med / 1e3,
                        "peak_extra_bytes": max(r["peak_extra_bytes"] for r in rs),
                        "max_abs_err_vs_fp32_ref": rs[0]["max_abs_err_vs_fp32_ref"]})
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["| B | L | arm | median us | GB/s (model) | peak extra MiB | err |", "|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append(f"| {s['B']} | {s['L']} | {s['arm']} | {s['median_us']:.1f} | "
                     f"{s['achieved_GBps_model']:.0f} | {s['peak_extra_bytes'] / 2**20:.1f} | "
                     f"{s['max_abs_err_vs_fp32_ref']:.2e} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Packed decode attention microbenchmark.")
    ap.add_argument("--out", required=True, help="new directory for raw samples and summary")
    ap.add_argument("--report-only", action="store_true", help="rebuild summary from samples.jsonl")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU-capable CLI check")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 32])
    ap.add_argument("--lens", type=int, nargs="+", default=[512, 2048, 8192])
    ap.add_argument("--hq", type=int, default=28)
    ap.add_argument("--hkv", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--bit-width", type=int, default=8)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.report_only:
        return report(args.out)
    if args.smoke:
        args.batches, args.lens, args.reps, args.iters, args.warmup = [2], [40], 2, 2, 1
    run(args)


if __name__ == "__main__":
    main()
