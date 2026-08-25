"""Phase B: score the KV-quantization path on LongBench-E.

Gives two things phase C needs and the project does not currently have:

  * an fp16 baseline score per task and per context-length bucket, and
  * a scored gate, so "the kernel did not change the answers" is a number
    rather than whether one planted passphrase survived.

The existing TurboQuant correctness evidence is a single needle string
("violet-harbor-1987") checked over 12 runs. That is a smoke test: it is one bit
of signal per run, it cannot see a partial degradation, and a cache that
mangles everything except a verbatim rare token would pass it.

Decoding is a plain greedy loop rather than `model.generate` so that every arm
goes through identical code and the only difference is the cache object.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import TQPackedCache            # noqa: E402
from longbench import TASKS, load_task, score, truncate_middle  # noqa: E402

MODEL = "Qwen/Qwen2.5-7B-Instruct"

# bit_width, num_outlier_channels, outlier_bits
# The bit-width sweep (bitwidth_sweep.py) put the quality cliff between 5 and
# 6 bits on this model: 6-bit reproduces the fp16 continuation verbatim, 5-bit
# and below do not. The arms bracket that cliff rather than clustering below it.
ARMS = {
    "fp16": None,
    "tq8": (8, 0, 0),
    "tq6": (6, 0, 0),
    "tq5": (5, 0, 0),
    "tq4": (4, 0, 0),
}


def bucket_of(n: int) -> str:
    """LongBench-E's own length buckets."""
    if n < 4096:
        return "0-4k"
    if n < 8192:
        return "4-8k"
    return "8k+"


@torch.no_grad()
def generate(model, tok, cfg, arm, ids, max_new, eos_ids,
             use_kernel=True):
    """Greedy decode. Returns (text, prefill_s, decode_s, n_new)."""
    n_ctx = ids.shape[1]
    if ARMS[arm] is None:
        cache = DynamicCache()
    else:
        bw, no, ob = ARMS[arm]
        cache = TQPackedCache(cfg, bit_width=bw,
                              max_cache_len=n_ctx + max_new + 8,
                              device="cuda", num_outlier_channels=no,
                              outlier_bits=ob, use_kernel=use_kernel)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(input_ids=ids, past_key_values=cache, use_cache=True,
                logits_to_keep=1)
    torch.cuda.synchronize()
    prefill = time.perf_counter() - t0

    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    del out
    got = [nxt.item()]
    t0 = time.perf_counter()
    for _ in range(max_new - 1):
        if got[-1] in eos_ids:
            break
        o = model(input_ids=nxt, past_key_values=cache, use_cache=True,
                  logits_to_keep=1)
        nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
        del o
        got.append(nxt.item())
    torch.cuda.synchronize()
    dec = time.perf_counter() - t0

    del cache
    gc.collect()
    torch.cuda.empty_cache()
    text = tok.decode([g for g in got if g not in eos_ids],
                      skip_special_tokens=True)
    return text.strip(), prefill, dec, len(got)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--per-bucket", type=int, default=10,
                    help="samples per (task, length bucket)")
    ap.add_argument("--max-ctx", type=int, default=16384,
                    help="middle-truncate prompts to this many tokens")
    ap.add_argument("--no-kernel", action="store_true",
                    help="use the chunked torch dequant instead of the fused "
                         "Triton kernel (phase C's A/B)")
    ap.add_argument("--out", default="results/phaseB_longbench.json")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    eos_ids = {tok.eos_token_id}
    if tok.convert_tokens_to_ids("<|im_end|>") is not None:
        eos_ids.add(tok.convert_tokens_to_ids("<|im_end|>"))
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda").eval()
    cfg = model.config

    # --- assemble a fixed, bucket-balanced sample set, shared by every arm ---
    samples = []
    for task in a.tasks:
        ds = load_task(task)
        per = defaultdict(list)
        for ex in ds:
            per[bucket_of(ex["length"])].append(ex)
        for b in ("0-4k", "4-8k", "8k+"):
            for ex in per[b][:a.per_bucket]:
                samples.append((task, b, ex))
    print(f"{MODEL} | {len(samples)} samples "
          f"({a.per_bucket}/bucket x {len(a.tasks)} tasks x 3 buckets) "
          f"| max_ctx {a.max_ctx}\n")

    rows = []
    for arm in a.arms:
        agg = defaultdict(list)
        t_arm = time.perf_counter()
        for i, (task, b, ex) in enumerate(samples):
            template, max_gen, metric = TASKS[task]
            prompt = template.format(context=ex["context"], input=ex["input"])
            prompt = truncate_middle(tok, prompt, a.max_ctx)
            ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
            try:
                text, pf, dc, n = generate(model, tok, cfg, arm, ids,
                                           max_gen, eos_ids,
                                           use_kernel=not a.no_kernel)
            except torch.cuda.OutOfMemoryError:
                gc.collect(); torch.cuda.empty_cache()
                print(f"    OOM on {task}/{b} at {ids.shape[1]} tok, skipped")
                continue
            s = score(metric, text, ex["answers"])
            agg[(task, b)].append(s)
            rows.append({"arm": arm, "task": task, "bucket": b,
                         "n_ctx": int(ids.shape[1]), "score": s,
                         "metric": metric, "prefill_s": pf, "decode_s": dc,
                         "n_new": n, "pred": text[:200],
                         "gold": ex["answers"][:3]})
            if (i + 1) % 20 == 0:
                print(f"    {arm}: {i+1}/{len(samples)}")
        el = time.perf_counter() - t_arm
        overall = [s for v in agg.values() for s in v]
        print(f"  {arm:<7} overall {100*sum(overall)/len(overall):5.2f}  "
              f"({len(overall)} samples, {el/60:.1f} min)")
        for (task, b), v in sorted(agg.items()):
            print(f"      {task:<22}{b:>6}  {100*sum(v)/len(v):6.2f}  (n={len(v)})")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"model": MODEL, "max_ctx": a.max_ctx,
                   "per_bucket": a.per_bucket, "rows": rows}, f, indent=2)

    # --- final table: score by arm x bucket, and the delta from fp16 --------
    print("\n=== LongBench-E score by arm and context bucket ===")
    by = defaultdict(list)
    for r in rows:
        by[(r["arm"], r["bucket"])].append(r["score"])
        by[(r["arm"], "all")].append(r["score"])
    buckets = ["0-4k", "4-8k", "8k+", "all"]
    print(f"{'arm':<8}" + "".join(f"{b:>10}" for b in buckets)
          + "".join(f"{'d '+b:>10}" for b in buckets))
    base = {b: (100 * sum(by[("fp16", b)]) / len(by[("fp16", b)])
                if by.get(("fp16", b)) else float("nan")) for b in buckets}
    for arm in a.arms:
        line = f"{arm:<8}"
        vals = {}
        for b in buckets:
            v = by.get((arm, b))
            vals[b] = 100 * sum(v) / len(v) if v else float("nan")
            line += f"{vals[b]:>10.2f}"
        for b in buckets:
            line += f"{vals[b]-base[b]:>+10.2f}"
        print(line)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
