"""C27, the half that was never measured: FULL DECODE.

Kernel microbenchmarks say the native CUDA kernel is 8.1x slower than the TF32
Triton one. This asks the only question a user cares about: what does that do to
tokens per second when the kernel is sitting inside a real model decoding real
tokens?

Same model, same prompt, same cache format, same seed. The ONLY thing that
changes between arms is which dequantization kernel TQPackedLayer._load calls.
"""
import sys, json, time, argparse, gc
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "turboquant_kv"))
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
ap.add_argument("--ctxs", type=int, nargs="+", default=[1024, 2048, 4096])
ap.add_argument("--gen", type=int, default=32)
ap.add_argument("--bw", type=int, default=6)
ap.add_argument("--arms", nargs="+",
                default=["triton_tf32", "triton_fp32", "native_cuda_fp32", "fp16_baseline"])
ap.add_argument("--out", default="decode.json")
a = ap.parse_args()

torch.cuda.set_per_process_memory_fraction(4.0 / 48.0, 0)   # announced envelope, enforced

import cache as C
from transformers import AutoModelForCausalLM
from torch.utils.cpp_extension import load
from model_guard import guarded_from_pretrained

CU = ROOT / "turboquant_kv" / "csrc" / "tq_dequant.cu"
ext = load(name="tq_dequant_cuda_ext", sources=[str(CU)], verbose=False)
def cuda_fused(packed, norms, centroids, Pi, bw, D, out=None, allow_tf32=False):
    """Same signature as the Triton entry point cache.py already calls, so the
    ONLY difference between arms is which kernel runs."""
    lead = packed.shape[:-1]
    NB = packed.shape[-1]
    N = 1
    for s in lead:
        N *= s
    L = lead[-1]
    maxlen = (packed.stride(-3) // packed.stride(-2)) if packed.dim() >= 3 else L
    if out is None:
        out = torch.empty(*lead, D, dtype=torch.bfloat16, device=packed.device)
    return ext.tq_dequant_cuda(packed, norms, centroids, Pi, out, bw, D, N, L, maxlen)


model = guarded_from_pretrained(
    AutoModelForCausalLM, a.model, torch_dtype=torch.bfloat16
).cuda().eval()
cfg = model.config
print("model=%s layers=%d kv_heads=%s head_dim=%s  weights=%.2f GB" % (
    a.model, cfg.num_hidden_layers, getattr(cfg, "num_key_value_heads", None),
    getattr(cfg, "head_dim", None),
    sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9))


@torch.no_grad()
def run(arm, ctx, gen):
    torch.manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size - 1, (1, ctx), device="cuda")
    if arm == "fp16_baseline":
        from transformers import DynamicCache
        pkv = DynamicCache()
    else:
        pkv = C.TQPackedCache(cfg, a.bw, max_cache_len=ctx + gen + 8, device="cuda",
                              use_kernel=True, allow_tf32=(arm == "triton_tf32"),
                              kernel_backend=arm,
                              native_kernel=(cuda_fused if arm == "native_cuda_fp32"
                                             else None))
    # prefill
    torch.cuda.synchronize(); t0 = time.perf_counter()
    # logits_to_keep=1: without it the prefill materialises
    # (1, ctx, 151936) bf16 logits - 2.5 GB at ctx=8192 - which blows the 4 GB
    # envelope for reasons that have nothing to do with the KV cache.
    o = model(ids, past_key_values=pkv, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize(); prefill = time.perf_counter() - t0
    nxt = o.logits[:, -1:].argmax(-1)
    # a few untimed steps so any lazy compile happens outside the window
    for _ in range(3):
        o = model(nxt, past_key_values=pkv, use_cache=True, logits_to_keep=1)
        nxt = o.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(gen):
        o = model(nxt, past_key_values=pkv, use_cache=True, logits_to_keep=1)
        nxt = o.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    del pkv, o, ids, nxt
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return dict(prefill_s=prefill, decode_s=dt, ms_per_token=dt / gen * 1e3,
                tok_per_s=gen / dt, peak_gib=peak)


res = {}
for ctx in a.ctxs:
    res[str(ctx)] = {}
    for arm in a.arms:
        try:
            r = run(arm, ctx, a.gen)
        except torch.OutOfMemoryError as e:
            print("%-6d %-18s OOM under the 4 GB cap" % (ctx, arm))
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            res[str(ctx)][arm] = dict(oom=True); continue
        res[str(ctx)][arm] = r
        print("ctx=%-6d %-18s %8.3f ms/tok  %7.2f tok/s  prefill %6.3f s  peak %.2f GiB"
              % (ctx, arm, r["ms_per_token"], r["tok_per_s"], r["prefill_s"], r["peak_gib"]))
    print()

json.dump(res, open(HERE / "out" / a.out, "w"), indent=2)
print("OK")
