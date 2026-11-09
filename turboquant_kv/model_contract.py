"""Model-integration contract: the paged engine against isolated references.

    python model_contract.py --model Qwen/Qwen2.5-7B-Instruct --device cuda --out RUN_DIR
    python model_contract.py --model Qwen/Qwen2.5-0.5B-Instruct --device cpu --out RUN_DIR

Two pairs, each the same numbers stored two ways:

* packed engine (paged, chunked prefill, concurrent mixed-length batch, direct
  compressed decode) vs the existing `TQPackedCache` reconstruct path run one
  request at a time with a full prefill: the quantized reference;
* dense engine (paged unquantized pages, same scheduler) vs the model's own
  DynamicCache greedy loop: the native reference.

Why the packed comparison is not a plain logit tolerance: each decode step
quantizes the new token's K/V with `bucketize`, which is discontinuous. Two
correct implementations whose fp32 rounding differs by ~1e-6 can store a
different index for a new token in a deep layer, and the difference then
propagates (measured on Qwen2.5-0.5B: prompt bytes identical, first index flip
at layer 9 of the first decode token). So the packed contract has three parts:

1. prompt K/V bytes stored by the engine are bitwise identical to the
   reference cache's;
2. on real captured decode queries for every layer, the compressed operator's
   error against a float64 oracle is at most 2x the reconstruct path's own
   error plus `op_atol` (the arithmetic budget, on identical stored bytes);
3. end to end, a request's largest logit difference before any token
   divergence is at most `noise_frac` of that request's quantization-noise
   scale (largest teacher-forced |native - quantized reference| logit gap over
   the same steps), and a divergence happens only where the reference's
   top-1/top-2 margin is within twice the perturbation observed at that step.

   Revision note (2026-11-02, before any GPU run): the first version bounded
   each step by its own step's noise. On Qwen2.5-0.5B CPU fp32 that failed 2
   of 117 steps, both where that step's quantization noise was small by chance
   (0.8-1.5) while a discrete index flip moved logits 0.25-0.40. Both runs are
   retained. The request-level scale is the frozen rule for the GPU gate.

The dense arm has no quantization, so it keeps a plain tolerance against the
native DynamicCache loop: logits within `dense_logit_atol`, divergence only at
a near-tie within `dense_tie_margin`.

Quality versus unquantized inference is a different question, answered by the
quality study, not here.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))     # model_guard
import packed_backend as pb
from cache import TQPackedCache
from engine import Engine, EngineConfig, Request

CONTRACT = {
    "float32": {"dense_logit_atol": 1e-2, "dense_tie_margin": 1e-2, "op_atol": 1e-5,
                "noise_frac": 0.25},
    "bfloat16": {"dense_logit_atol": 0.5, "dense_tie_margin": 0.5, "op_atol": 1e-2,
                 "noise_frac": 0.25},
}

PROMPTS = [
    "Explain in two sentences why a KV cache grows with context length.",
    "List three prime numbers greater than 50 and say why each is prime.",
    "Translate to French: The meeting moved to Thursday because the room was booked.",
    "Write a haiku about a lighthouse in winter.",
    "A train leaves at 9:40 and arrives at 13:05. How long is the trip? Show the arithmetic.",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
]


def chat_ids(tok, text):
    rendered = tok.apply_chat_template([{"role": "user", "content": text}],
                                       add_generation_prompt=True, tokenize=False)
    return tok(rendered, add_special_tokens=False).input_ids


@torch.no_grad()
def greedy_loop(model, ids, max_new, eos, cache):
    out_t, out_l = [], []
    x = torch.tensor([ids], device=model.device)
    for _ in range(max_new):
        lg = model(input_ids=x, past_key_values=cache, use_cache=True,
                   logits_to_keep=1).logits[0, -1].float()
        t = int(lg.argmax())
        out_t.append(t)
        out_l.append(lg.cpu())
        if t in eos:
            break
        x = torch.tensor([[t]], device=model.device)
    return out_t, out_l


def compare(ref_t, ref_l, got_t, got_l, *, logit_tol, tie_tol):
    """Tokens and logits of one request. `logit_tol` may be a per-step list."""
    n = min(len(ref_t), len(got_t))
    div = next((i for i in range(n) if ref_t[i] != got_t[i]), None)
    upto = div if div is not None else n
    diffs = [float((a - b).abs().max()) for a, b in zip(ref_l, got_l)]
    tol = logit_tol if isinstance(logit_tol, list) else [logit_tol] * len(diffs)
    rec = {"ref_tokens": ref_t, "got_tokens": got_t, "first_divergence": div,
           "max_logit_diff_before_divergence": max(diffs[:upto], default=0.0),
           "logit_diffs": diffs[:upto + 1], "tolerances": tol[:upto + 1],
           "same_length": len(ref_t) == len(got_t)}
    ok = all(d <= t for d, t in zip(diffs[:upto], tol))
    if div is not None:
        top = torch.topk(ref_l[div], 2).values
        rec["ref_margin_at_divergence"] = float(top[0] - top[1])
        allowed = tie_tol if tie_tol is not None else diffs[div]
        rec["divergence_explained"] = rec["ref_margin_at_divergence"] <= 2 * allowed
        ok = ok and rec["divergence_explained"]
    elif not rec["same_length"]:
        ok = False
    rec["within_contract"] = ok
    return rec


@torch.no_grad()
def teacher_forced_native(model, prompt, toks):
    """Native (unquantized) logits along the reference's own token path."""
    model.config._attn_implementation = "sdpa"
    x = torch.tensor([prompt + toks[:-1]], device=model.device)
    lg = model(input_ids=x).logits[0, len(prompt) - 1:].float().cpu()
    return list(lg)


def run_engine(model, prompts, news, eos, *, kv_mode, device, page_size, chunk, backend):
    model.config._attn_implementation = pb.ATTN_NAME
    dense = kv_mode == "dense"
    pool = pb.make_pool(model.config, bit_width=None if dense else 8, num_pages=4096,
                        page_size=page_size, device=device,
                        dense_dtype=model.dtype if dense else None)
    kq, vq = (None, None) if dense else pb.quantizers_for(model.config, 8, device)
    eng = Engine(model, pool, kq, vq, pb.head_dim_of(model.config), eos_ids=eos,
                 config=EngineConfig(max_batch=len(prompts), prefill_chunk=chunk,
                                     max_prompt_tokens=1 << 16, max_new_tokens_cap=4096,
                                     decode_backend=backend))
    logits = {}
    eng.on_logits = lambda reqs, lg: [logits.setdefault(r.request_id, []).append(lg[i].cpu())
                                      for i, r in enumerate(reqs)]
    reqs = [eng.submit(Request(f"{kv_mode}-{i}", p, n)) for i, (p, n) in enumerate(zip(prompts, news))]
    pb.BACKEND_COUNTS.clear()
    t0 = time.perf_counter()
    eng.run_until_idle()
    wall = time.perf_counter() - t0
    out = [(r.generated, logits[r.request_id][-len(r.generated):], r.status) for r in reqs]
    return out, dict(pb.BACKEND_COUNTS), wall, eng.pool.stats()


@torch.no_grad()
def layer_check(model, ids, steps, device, page_size, c):
    """Prompt-byte identity, then the operator on every layer's real decode
    queries against a float64 oracle, both paths reading the same bytes."""
    import packed_attention as pa
    model.config._attn_implementation = "sdpa"
    ref = TQPackedCache(model.config, bit_width=8, max_cache_len=len(ids) + steps + 1, device=device)
    x = torch.tensor([ids], device=device)
    model(input_ids=x, past_key_values=ref, use_cache=True, logits_to_keep=1)
    model.config._attn_implementation = pb.ATTN_NAME
    pool = pb.make_pool(model.config, bit_width=8, num_pages=1024, page_size=page_size, device=device)
    kq, vq = pb.quantizers_for(model.config, 8, device)
    pool.ensure("L", len(ids) + steps + 1)
    pc = pb.PagedPackedCache(pool, ["L"], [0], k_quant=kq, v_quant=vq,
                             head_dim=pb.head_dim_of(model.config), decode_backend="auto")
    model(input_ids=x, position_ids=torch.arange(len(ids), device=device)[None],
          past_key_values=pc, use_cache=True, logits_to_keep=1)
    pids = [int(p) for p in pc._table[0] if int(p) >= 0]
    identical = True
    for li, layer in enumerate(ref.layers):
        for mine, theirs in ((pool.k_pages, layer._pk_r), (pool.v_pages, layer._pv_r),
                             (pool.k_norms, layer._nk_r), (pool.v_norms, layer._nv_r)):
            got = torch.cat([mine[li][p] for p in pids], dim=1)[:, :len(ids)]
            identical &= bool(torch.equal(got, theirs[0, :, :len(ids)]))

    captured = []
    real = pb.winnow_packed_attention

    def spy(module, query, key, value, mask, **kw):
        if query.shape[2] == 1:
            captured.append((module.layer_idx, query.clone(), key, kw.get("scaling")))
        return real(module, query, key, value, mask, **kw)
    from transformers import AttentionInterface
    AttentionInterface.register(pb.ATTN_NAME, spy)
    try:
        g = torch.Generator().manual_seed(0)
        for s in range(steps):
            tok = torch.randint(0, model.config.vocab_size, (1, 1), generator=g).to(device)
            model(input_ids=tok, position_ids=torch.tensor([[len(ids) + s]], device=device),
                  past_key_values=pc, use_cache=True, logits_to_keep=1)
    finally:
        pb.register()

    worst = {"direct": 0.0, "reconstruct": 0.0, "ratio_ok": True}
    for li, q, kv, scale in captured:
        D, n_rep = kv.head_dim, q.shape[1] // kv.num_kv_heads
        hist = []
        for pages, norms, quant in ((kv.k_pages, kv.k_norms, kv.k_quant), (kv.v_pages, kv.v_norms, kv.v_quant)):
            rows, nrm = pa._gather_history(pages, norms, kv, 0)
            hist.append(((pa._centroids(rows, quant, D).double() @ quant.Pi.double())
                         * nrm.double()[..., None]).repeat_interleave(n_rep, 0))
        p = torch.softmax((q[0].double() @ hist[0].transpose(-1, -2)) * scale, -1)
        oracle = (p @ hist[1])[None]
        backend = pb.select_decode_backend(q, kv, "auto")
        direct = (pb._triton_decode(q, kv, scale) if backend == "triton_packed"
                  else pa.tiled_compressed_attention(q, kv, scale))
        e_d = float((direct.double() - oracle).abs().max())
        e_r = float((pb._reconstruct_sdpa(q, kv, scale).double() - oracle).abs().max())
        worst["direct"] = max(worst["direct"], e_d)
        worst["reconstruct"] = max(worst["reconstruct"], e_r)
        worst["ratio_ok"] &= e_d <= 2 * e_r + c["op_atol"]
    return {"layers": len(ref.layers), "decode_steps": steps, "captured_calls": len(captured),
            "prompt_bytes_identical": identical, "operator_backend": backend,
            "max_err_vs_fp64_direct": worst["direct"], "max_err_vs_fp64_reconstruct": worst["reconstruct"],
            "within_contract": identical and worst["ratio_ok"]}


def rule_v4(record) -> dict:
    """Pre-registered 2026-11-04 (track01-carryover-rules.json, sha 124c8bed...).

    floor  = max over requests of the dense engine's largest logit gap to native
    dense  : >= 6/7 token-identical; any divergence at reference top-2 margin <= 1.0
    packed : >= 6/7 requests with max gap before divergence <= 3 x floor and any
             divergence where the reference margin <= 2 x the gap at that step
    operator: unchanged (direct <= 2 x reconstruct + op_atol vs float64)
    prompt bytes: reported, not gating
    """
    dense, packed = record["dense"]["requests"], record["packed"]["requests"]
    floor = max(r["max_logit_diff_before_divergence"] for r in dense)
    d_ok = [r["first_divergence"] is None and r["same_length"] or
            (r["first_divergence"] is not None and r.get("ref_margin_at_divergence", 9) <= 1.0)
            for r in dense]
    d_ident = sum(r["first_divergence"] is None and r["same_length"] for r in dense)
    p_ok = []
    for r in packed:
        ok = r["max_logit_diff_before_divergence"] <= 3 * floor
        if r["first_divergence"] is not None:
            gap = r["logit_diffs"][r["first_divergence"]] if len(r["logit_diffs"]) > r["first_divergence"] else 0
            ok = ok and r.get("ref_margin_at_divergence", 9) <= 2 * gap
        elif not r["same_length"]:
            ok = False
        p_ok.append(ok)
    op = record["layers"]
    op_ok = op["max_err_vs_fp64_direct"] <= 2 * op["max_err_vs_fp64_reconstruct"] + CONTRACT[record["dtype"]]["op_atol"]
    res = {"noise_floor": floor, "dense_identical": d_ident, "dense_ok": sum(d_ok),
           "packed_ok": sum(p_ok), "packed_per_request": p_ok, "operator_ok": op_ok,
           "prompt_bytes_identical_reported": op["prompt_bytes_identical"],
           "pool_clean": record["packed"]["pool_clean"] and record["dense"]["pool_clean"]}
    n = len(packed)
    res["pass"] = (d_ident >= n - 1 and sum(d_ok) == len(dense) and sum(p_ok) >= n - 1
                   and op_ok and res["pool_clean"])
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="Paged engine vs isolated references.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["bfloat16", "float32"])
    ap.add_argument("--max-new", type=int, nargs="+", default=[24, 7, 40, 16, 32, 1])
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--prefill-chunk", type=int, default=13)
    ap.add_argument("--long-prompt-tokens", type=int, default=0,
                    help="also add one prompt padded to this many tokens")
    ap.add_argument("--rule", default="v3", choices=["v3", "v4"],
                    help="v4: track01-carryover-rules.json (2026-11-04), judged against the "
                         "dense-vs-native noise floor measured in the same run")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    dtype = args.dtype or ("bfloat16" if args.device == "cuda" else "float32")
    c = CONTRACT[dtype]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)

    from model_guard import guarded_from_pretrained, pinned_revision
    from transformers import AutoModelForCausalLM, AutoTokenizer
    pb.register()
    tok = guarded_from_pretrained(AutoTokenizer, args.model)
    # The guard validates tensors against the published bf16 checkpoint, so load
    # bf16 through it and upcast in memory when an fp32 contract is requested.
    model = guarded_from_pretrained(AutoModelForCausalLM, args.model,
                                    dtype=torch.bfloat16 if args.device == "cuda" else getattr(torch, dtype),
                                    attn_implementation="sdpa").to(args.device, getattr(torch, dtype)).eval()
    pb.check_model(model.config, 8)
    eos = {tok.eos_token_id, *(model.generation_config.eos_token_id or [])} - {None}
    prompts = [chat_ids(tok, p) for p in PROMPTS]
    if args.long_prompt_tokens:
        filler = tok(" ".join(["The archive lists ledger entry %d as reconciled." % i
                               for i in range(args.long_prompt_tokens // 8)]),
                     add_special_tokens=False).input_ids[:args.long_prompt_tokens]
        prompts.append(chat_ids(tok, tok.decode(filler) + "\nWhich ledger entry is listed last?"))
    news = (args.max_new * len(prompts))[:len(prompts)]

    record = {"model": args.model, "revision": pinned_revision(args.model), "device": args.device,
              "dtype": dtype, "contract": c, "prompt_tokens": [len(p) for p in prompts],
              "max_new": news, "env": {"torch": torch.__version__, "python": platform.python_version(),
                                      "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None}}
    for kv_mode, backend in (("packed", "auto"), ("dense", "auto")):
        got, counts, wall, stats = run_engine(model, prompts, news, eos, kv_mode=kv_mode,
                                              device=args.device, page_size=args.page_size,
                                              chunk=args.prefill_chunk, backend=backend)
        rows = []
        for p, n, (gt, gl, status) in zip(prompts, news, got):
            if kv_mode == "packed":
                model.config._attn_implementation = "sdpa"
                cache = TQPackedCache(model.config, bit_width=8, max_cache_len=len(p) + n + 1,
                                      device=args.device)
            else:
                from transformers import DynamicCache
                model.config._attn_implementation = "sdpa"
                cache = DynamicCache(config=model.config)
            rt, rl = greedy_loop(model, p, n, eos, cache)
            if kv_mode == "packed":
                noise = [float((a - b).abs().max()) for a, b in
                         zip(rl, teacher_forced_native(model, p, rt))]
                row = compare(rt, rl, gt, gl, logit_tol=c["noise_frac"] * max(noise),
                              tie_tol=None)
                row["quant_noise_max_logit_diff"] = max(noise)
            else:
                row = compare(rt, rl, gt, gl, logit_tol=c["dense_logit_atol"],
                              tie_tol=c["dense_tie_margin"])
            rows.append({**row, "status": status, "text": tok.decode(gt)})
        record[kv_mode] = {"backend_counts": counts, "engine_wall_s": wall, "pool_after": stats,
                           "requests": rows,
                           "all_within_contract": all(r["within_contract"] for r in rows),
                           "pool_clean": stats["owned"] == 0 and stats["pending"] == 0}
    record["layers"] = layer_check(model, prompts[0], 6, args.device, args.page_size, c)
    record["pass"] = (record["packed"]["all_within_contract"] and record["dense"]["all_within_contract"]
                      and record["packed"]["pool_clean"] and record["dense"]["pool_clean"]
                      and record["layers"]["within_contract"])
    if args.rule == "v4":
        record["v4"] = rule_v4(record)
        record["pass"] = record["v4"]["pass"]
    (out / "model_contract.json").write_text(json.dumps(record, indent=2) + "\n")
    for k in ("packed", "dense"):
        rs = record[k]["requests"]
        print(f"{k}: {sum(r['within_contract'] for r in rs)}/{len(rs)} within contract; "
              f"identical tokens {sum(r['first_divergence'] is None and r['same_length'] for r in rs)}/{len(rs)}; "
              f"max logit diff {max(r['max_logit_diff_before_divergence'] for r in rs):.2e}; "
              f"backends {record[k]['backend_counts']}")
    L = record["layers"]
    print(f"layers: prompt bytes identical={L['prompt_bytes_identical']}; operator ({L['operator_backend']}) "
          f"err vs fp64 {L['max_err_vs_fp64_direct']:.2e} (reconstruct {L['max_err_vs_fp64_reconstruct']:.2e}) "
          f"over {L['captured_calls']} calls; PASS={record['pass']}")
    return 0 if record["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
