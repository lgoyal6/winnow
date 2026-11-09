"""Profile one decode-attention call per arm: kernels, rotations, allocations.

    python profile_decode.py --out DIR [--batch 16 --len 8192]

Writes, per arm, the profiler's CUDA-time table (so the query/output rotation
GEMMs, the Triton kernel and the split merge are separated), a Chrome trace,
and allocator peaks. Needs CUDA; it says BLOCKED and exits non-zero otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import packed_attention as pa
from bench_packed_attention import arms_for, build


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--len", type=int, default=8192)
    a = ap.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("BLOCKED: profiling needs CUDA")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    q, kvd, dense, contig = build(a.batch, a.len, 28, 4, 128, 16, 8, "cuda", 0)
    arms = arms_for(q, kvd, dense, contig, pa.default_scale(128), 16)
    summary = {}
    for name, fn in arms.items():
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(out / f"{name}.trace.json"))
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)
        (out / f"{name}.txt").write_text(table)
        kernels = [{"name": e.key, "cuda_us_per_call": e.device_time_total / 10, "count": e.count / 10}
                   for e in prof.key_averages() if e.device_time_total > 0]
        summary[name] = {"peak_extra_bytes": torch.cuda.max_memory_allocated() - base,
                         "kernels": sorted(kernels, key=lambda k: -k["cuda_us_per_call"])[:10]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: [(x["name"][:40], round(x["cuda_us_per_call"], 1)) for x in v["kernels"][:4]]
                      for k, v in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
