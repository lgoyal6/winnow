"""Where does TurboQuant stop destroying the model?

4-bit reconstructs real K/V to 9.7% relative L2 error and Qwen2.5-7B produces
unusable text from 8 tokens of context upward. That number matches the
theoretical Lloyd-Max bound for 16 levels on a Gaussian, so the implementation
is not at fault and the only free variable left is bit width.

This sweeps it, and reports three things per width so the failure has a cause
attached rather than just a verdict:

  * relative L2 error on the model's own K/V, against the same error measured on
    ideal Gaussian input. If those two agree, the quantizer is achieving its
    theoretical distortion and the loss is inherent to the bit width, not to the
    rotation failing on real data.
  * bytes per KV vector, so a quality verdict can be traded against the actual
    storage cost rather than a nominal bit count.
  * generated text against the fp16 continuation, greedy, so the comparison is
    deterministic.
"""
from __future__ import annotations

import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cache import TQPackedCache, TurboQuantMSE   # noqa: E402
from model_guard import guarded_from_pretrained  # noqa: E402
from packing import packed_bytes                 # noqa: E402

MODEL = "Qwen/Qwen2.5-7B-Instruct"
WIDTHS = (3, 4, 5, 6, 8)


@torch.no_grad()
def main():
    tok = guarded_from_pretrained(AutoTokenizer, MODEL)
    model = guarded_from_pretrained(
        AutoModelForCausalLM, MODEL, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="cuda").eval()
    cfg = model.config
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads

    text = ("The maintenance log records that on the third inspection cycle "
            "the coolant pressure held steady while the secondary loop was "
            "vented. ") * 24
    ids = tok(text, return_tensors="pt").input_ids[:, :512].cuda()

    # Real K/V to measure reconstruction against.
    cache = DynamicCache()
    model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    real = torch.cat([cache.layers[i].keys.float() for i in (0, 7, 14, 21, 27)],
                     dim=2)
    ideal = torch.randn_like(real)
    del cache

    print(f"{MODEL}, head_dim={hd}\n")
    print(f"{'bits':>5}{'B/vector':>10}{'vs fp16':>9}{'rel_L2 real':>13}"
          f"{'rel_L2 gaussian':>17}{'ratio':>8}")
    print("-" * 62)
    rows = []
    for bw in WIDTHS:
        tq = TurboQuantMSE(bw, hd, "cuda")
        errs = {}
        for name, x in (("real", real), ("gauss", ideal)):
            idx, nm = tq.quantize(x)
            rec = torch.empty_like(x)
            tq.dequantize_into(idx, nm, rec)
            errs[name] = ((rec - x).norm(dim=-1)
                          / x.norm(dim=-1).clamp(min=1e-10)).mean().item()
        nb = packed_bytes(hd, bw) + 2          # + fp16 norm
        rows.append({"bits": bw, "bytes_per_vector": nb,
                     "compression_vs_fp16": (hd * 2) / nb,
                     "rel_l2_real": errs["real"],
                     "rel_l2_gaussian": errs["gauss"]})
        print(f"{bw:>5}{nb:>10}{(hd*2)/nb:>8.2f}x{errs['real']:>13.4f}"
              f"{errs['gauss']:>17.4f}{errs['real']/errs['gauss']:>8.2f}")

    print("\n(ratio ~1.0 means the rotation is doing its job: real K/V is "
          "quantized as well as ideal Gaussian input, so any loss is inherent "
          "to the bit width)\n")

    # --- does the text survive? ------------------------------------------
    print("=== greedy continuation, ctx=512 ===")
    def gen(cache_fn, n=18):
        c = cache_fn()
        o = model(input_ids=ids, past_key_values=c, use_cache=True,
                  logits_to_keep=1)
        nx = o.logits[:, -1, :].argmax(-1, keepdim=True)
        got = [nx.item()]
        for _ in range(n - 1):
            o = model(input_ids=nx, past_key_values=c, use_cache=True,
                      logits_to_keep=1)
            nx = o.logits[:, -1, :].argmax(-1, keepdim=True)
            got.append(nx.item())
        del c
        torch.cuda.empty_cache()
        return got

    ref = gen(lambda: DynamicCache())
    ref_txt = tok.decode(ref, skip_special_tokens=True)
    print(f"  {'fp16':>6}  {ref_txt!r}")
    for bw, row in zip(WIDTHS, rows):
        got = gen(lambda bw=bw: TQPackedCache(cfg, bit_width=bw,
                                              max_cache_len=ids.shape[1] + 32,
                                              device="cuda"))
        match = sum(a == b for a, b in zip(ref, got)) / len(ref)
        row["greedy_token_match_vs_fp16"] = match
        row["text"] = tok.decode(got, skip_special_tokens=True)
        print(f"  {bw:>4}b  {row['text']!r}")
        print(f"         token match vs fp16: {100*match:.0f}%")

    os.makedirs("results", exist_ok=True)
    with open("results/bitwidth_sweep.json", "w") as f:
        json.dump({"model": MODEL, "rows": rows}, f, indent=2)
    print("\nwrote results/bitwidth_sweep.json")


if __name__ == "__main__":
    main()
