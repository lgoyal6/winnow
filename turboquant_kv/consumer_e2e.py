"""End-to-end real-consumer timing with and without the tuned registry.

    python consumer_e2e.py --manifest M.json --kv packed8 [--registry REG] --out RUN/e2e-packed-tuned

Runs the actual engine (chunked prefill through dequantize_history, batched
decode through the Triton attention kernel) on a fixed workload: the first
`--requests` final-split cohort questions, `--max-new` output tokens each
(EOS ignored so every arm does identical work), `--max-batch` concurrent.
Run each arm in a fresh process so the dispatcher, Triton's compile cache in
memory and the CUDA allocator start equal. Records wall time, first-request
time (includes JIT compiles), steady-state decode step times, and the
dispatcher's per-path call counts, which are the proof that promoted code (or
the fallback) actually ran.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--kv", default="packed8", choices=["packed8", "dense"])
    ap.add_argument("--registry")
    ap.add_argument("--requests", type=int, default=24)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--kv-budget-gib", type=float, default=12.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("BLOCKED: needs CUDA")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)

    import grpc_server as gs
    import packed_backend as pb
    import tuned_dispatch as td
    from engine import EngineConfig, Request
    from model_guard import guarded_from_pretrained
    from transformers import AutoModelForCausalLM

    disp = td.reset(a.registry)
    m = json.loads(Path(a.manifest).read_text())
    qs = [q for q in m["questions"] if q["split"] == "final"][:a.requests]
    pb.register()
    model = guarded_from_pretrained(AutoModelForCausalLM, m["model"], dtype=torch.bfloat16,
                                    attn_implementation="sdpa").to("cuda").eval()
    eng = gs.build_engine(model, kv_mode=a.kv, kv_budget_bytes=int(a.kv_budget_gib * 2**30),
                          page_size=16, decode_backend="auto",
                          cfg=EngineConfig(max_batch=a.max_batch, max_queue=len(qs), prefill_chunk=2048,
                                           max_prompt_tokens=m["cap_tokens"], max_new_tokens_cap=a.max_new))
    step_times = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    reqs = [eng.submit(Request(f"e2e-{i}", q["token_ids"], a.max_new, ignore_eos=True))
            for i, q in enumerate(qs)]
    first_done = None
    while not eng.idle():
        s0 = time.perf_counter()
        eng.step()
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - s0)
        if first_done is None and any(r.state == "done" for r in reqs):
            first_done = time.perf_counter() - t0
    wall = time.perf_counter() - t0
    rec = {"kv": a.kv, "registry": a.registry, "requests": len(qs), "max_new": a.max_new,
           "max_batch": a.max_batch, "wall_s": wall, "first_request_done_s": first_done,
           "steps": len(step_times), "step_s_median": statistics.median(step_times),
           "step_s_p95": sorted(step_times)[int(0.95 * (len(step_times) - 1))],
           "statuses": sorted({r.status for r in reqs}),
           "dispatch_counts": dict(disp.counts),
           "last_decisions": {k: v.__dict__ for k, v in disp.last.items()},
           "backend_counts": dict(pb.BACKEND_COUNTS),
           "output_tokens_sha": __import__("hashlib").sha256(
               json.dumps([r.generated for r in reqs]).encode()).hexdigest(),
           "gpu": torch.cuda.get_device_name(), "peak_allocated": torch.cuda.max_memory_allocated()}
    (out / "e2e.json").write_text(json.dumps(rec, indent=2) + "\n")
    (out / "step_times.json").write_text(json.dumps(step_times) + "\n")
    print(json.dumps({k: rec[k] for k in ("kv", "registry", "wall_s", "step_s_median", "dispatch_counts")}))


if __name__ == "__main__":
    main()
