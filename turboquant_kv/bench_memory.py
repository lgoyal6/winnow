"""Phase A: what TurboQuant costs and saves in bytes the process actually holds.

Three things get measured for every arm, because the existing report conflates
the first with the second:

  * **logical bits** (`mem_bits()`): the bits the scheme needs. This is the
    number behind the published "3.8-4.9x", and it is not a byte count.
  * **resident KV bytes**: every tensor reachable from the cache, deduplicated
    by storage pointer, so an implementation that keeps both quantized indices
    and a dequantized copy is charged for both. Implementation-agnostic: the
    same function measures the fp16 baseline and both TurboQuant variants.
  * **peak allocator bytes**: `torch.cuda.max_memory_allocated` across the whole
    decode, which catches the transient dequantized tensor and the `torch.cat`
    spike that resident accounting misses.

Resident bytes are split into *persistent* (held between steps, so it is what
limits how many sequences fit at once) and *transient* (allocated for one step
and freed). The distinction is the entire point: a scheme can hold 4x fewer
persistent bytes while still touching a full-precision tensor once per step.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "turboquant-poc"))

from cache import TQPackedCache                      # noqa: E402
from turboquant_poc import TQCache                   # noqa: E402  (existing impl)

MODEL = "Qwen/Qwen2.5-7B-Instruct"


# ---------------------------------------------------------------------------
def resident_bytes(cache) -> int:
    """Unique tensor storage bytes reachable from a cache's layers.

    Deduplicated by `data_ptr` so views are not double counted, and so an
    implementation holding a dequantized copy alongside its indices is charged
    for both. Works on any of the three arms without knowing their internals.
    """
    seen, total = set(), 0
    def visit(obj, depth=0):
        nonlocal total
        if depth > 3:
            return
        if torch.is_tensor(obj):
            if obj.numel() == 0:
                return
            p = obj.untyped_storage().data_ptr()
            if p not in seen:
                seen.add(p)
                total += obj.untyped_storage().nbytes()
            return
        if isinstance(obj, (list, tuple)):
            for o in obj:
                visit(o, depth + 1)
            return
        if hasattr(obj, "__dict__"):
            for o in vars(obj).values():
                visit(o, depth + 1)
    visit(getattr(cache, "layers", []))
    return total


def make_prompt(tok, n_tokens):
    """A prompt of roughly `n_tokens` tokens, deterministic."""
    base = ("The maintenance log records that on the third inspection cycle the "
            "coolant pressure held steady while the secondary loop was vented. ")
    ids = tok(base, return_tensors="pt").input_ids
    reps = max(1, n_tokens // ids.shape[1] + 1)
    text = base * reps
    ids = tok(text, return_tensors="pt").input_ids[:, :n_tokens]
    return ids


@torch.no_grad()
def run_arm(model, tok, arm, ctx, decode, batch, cfg):
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()

    ids = make_prompt(tok, ctx).to("cuda")
    if batch > 1:
        ids = ids.repeat(batch, 1)
    hd = (getattr(cfg, "head_dim", None)
          or cfg.hidden_size // cfg.num_attention_heads)

    if arm == "fp16":
        cache = DynamicCache()
    elif arm.endswith("-current"):
        bw, no, ob = ARMS[arm]
        cache = TQCache(head_dim=hd, bit_width=bw,
                        num_layers=cfg.num_hidden_layers, device="cuda",
                        num_outlier_channels=no, outlier_bits=ob)
    else:
        bw, no, ob = ARMS[arm]
        cache = TQPackedCache(cfg, bit_width=bw,
                              max_cache_len=ctx + decode + 8, device="cuda",
                              num_outlier_channels=no, outlier_bits=ob)

    # --- prefill (TTFT) ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    # logits_to_keep=1 keeps prefill from materializing a
    # (batch, ctx, 152064) logit tensor, which at ctx=2048 is 622 MB in bf16 and
    # would dominate the peak we are trying to attribute to the KV cache.
    out = model(input_ids=ids, past_key_values=cache, use_cache=True,
                logits_to_keep=1)
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0
    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    del out

    prefill_resident = resident_bytes(cache)
    peak_after_prefill = torch.cuda.max_memory_allocated() - base_alloc

    # --- decode ---
    gen = [nxt]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(decode - 1):
        o = model(input_ids=nxt, past_key_values=cache, use_cache=True,
                  logits_to_keep=1)
        nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
        gen.append(nxt)
        del o
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tok_s = (decode - 1) / dt if dt > 0 else float("nan")

    resident = resident_bytes(cache)
    peak = torch.cuda.max_memory_allocated() - base_alloc
    text = tok.decode(torch.cat(gen, dim=-1)[0], skip_special_tokens=True)

    # Persistent vs transient, where the implementation exposes the split.
    if hasattr(cache, "persistent_bytes_used"):
        persistent = cache.persistent_bytes_used()
        transient = cache.transient_bytes()
    elif arm == "fp16":
        persistent, transient = resident, 0
    else:
        # Existing TQ arm: everything reachable is persistent (it keeps both the
        # indices and the dequantized cache between steps).
        persistent, transient = resident, 0

    logical_bits = (cache.mem_bits() if hasattr(cache, "mem_bits") else None)
    eff = None
    if hasattr(cache, "eff_bits"):
        eff = cache.eff_bits()
    elif hasattr(cache, "effective_bits"):
        eff = cache.effective_bits()

    r = {
        "arm": arm, "ctx": ctx, "decode": decode, "batch": batch,
        "ttft_s": ttft, "decode_tok_s": tok_s,
        "resident_kv_mb": resident / 2**20,
        "persistent_kv_mb": persistent / 2**20,
        "transient_kv_mb": transient / 2**20,
        "peak_mb": peak / 2**20,
        "peak_after_prefill_mb": peak_after_prefill / 2**20,
        "logical_kv_mb": (logical_bits / 8 / 2**20) if logical_bits else None,
        "eff_bits": eff,
        "text_head": text[:60].replace("\n", " "),
    }
    del cache, gen
    gc.collect(); torch.cuda.empty_cache()
    return r


ARMS = {
    "fp16": (16, 0, 0),
    "tq4-current": (4, 0, 0), "tq4-packed": (4, 0, 0),
    "tq3-current": (3, 0, 0), "tq3-packed": (3, 0, 0),
    "tq3.5-current": (3, 32, 4), "tq3.5-packed": (3, 32, 4),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--ctxs", type=int, nargs="*", default=[2048, 8192, 16384])
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--batches", type=int, nargs="*", default=[1])
    ap.add_argument("--out", default="results/phaseA_memory.json")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda").eval()
    cfg = model.config
    print(f"{MODEL}: {cfg.num_hidden_layers} layers, "
          f"{cfg.num_key_value_heads} KV heads, head_dim "
          f"{getattr(cfg,'head_dim',None) or cfg.hidden_size//cfg.num_attention_heads}, "
          f"weights {torch.cuda.memory_allocated()/2**30:.2f} GiB\n")

    rows = []
    for b in a.batches:
        for ctx in a.ctxs:
            for arm in a.arms:
                try:
                    r = run_arm(model, tok, arm, ctx, a.decode, b, cfg)
                except torch.cuda.OutOfMemoryError:
                    print(f"  b{b} ctx{ctx:6d} {arm:<14} OOM")
                    gc.collect(); torch.cuda.empty_cache()
                    continue
                rows.append(r)
                print(f"  b{b} ctx{ctx:6d} {arm:<14} "
                      f"{r['decode_tok_s']:6.2f} tok/s  "
                      f"persist {r['persistent_kv_mb']:8.1f} MB  "
                      f"transient {r['transient_kv_mb']:7.1f} MB  "
                      f"peak {r['peak_mb']:8.1f} MB  "
                      f"logical {str(round(r['logical_kv_mb'],1)) if r['logical_kv_mb'] else '-':>8} MB")

    # --- did the rewrite change behaviour? --------------------------------
    # fp16 vs TQ is *expected* to differ (that is what quantization does, and
    # phase B scores it properly). What must not differ is -current vs -packed:
    # bit-packing is lossless and the only real change is fp16 norms, so the
    # generated text should match or come very close.
    print("\n=== existing implementation vs packed rewrite (same bit width) ===")
    for b in a.batches:
        for ctx in a.ctxs:
            for base in ("tq4", "tq3", "tq3.5"):
                cur = next((r for r in rows if r["arm"] == base + "-current"
                            and r["ctx"] == ctx and r["batch"] == b), None)
                pk = next((r for r in rows if r["arm"] == base + "-packed"
                           and r["ctx"] == ctx and r["batch"] == b), None)
                if not (cur and pk):
                    continue
                same = cur["text_head"] == pk["text_head"]
                print(f"  b{b} ctx{ctx:6d} {base:<6} identical_text={same}")
                if not same:
                    print(f"      current: {cur['text_head']!r}")
                    print(f"      packed : {pk['text_head']!r}")

    print("\n=== reported vs real: KV bytes ===")
    print(f"{'ctx':>7}{'batch':>6}{'arm':>16}{'logical MB':>12}"
          f"{'resident MB':>13}{'gap':>8}{'vs fp16':>9}")
    for b in a.batches:
        for ctx in a.ctxs:
            f16 = next((r for r in rows if r["arm"] == "fp16"
                        and r["ctx"] == ctx and r["batch"] == b), None)
            for r in rows:
                if r["ctx"] != ctx or r["batch"] != b:
                    continue
                lg = r["logical_kv_mb"]
                gap = (r["persistent_kv_mb"] / lg) if lg else float("nan")
                vs = (f16["persistent_kv_mb"] / r["persistent_kv_mb"]
                      if f16 and r["persistent_kv_mb"] else float("nan"))
                print(f"{ctx:>7}{b:>6}{r['arm']:>16}"
                      f"{(f'{lg:.1f}' if lg else '-'):>12}"
                      f"{r['persistent_kv_mb']:>13.1f}"
                      f"{(f'{gap:.2f}x' if lg else '-'):>8}"
                      f"{vs:>8.2f}x")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"model": MODEL, "rows": rows}, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
