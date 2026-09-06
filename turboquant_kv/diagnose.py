"""Why does TurboQuant produce gibberish on Qwen2.5-7B?

`test_cache.py` validates the quantizer on `torch.randn` input, which is exactly
the distribution its Lloyd-Max codebook was fitted for, so it cannot fail. Real
keys and values are not that distribution: attention states are known to carry a
few very high-magnitude channels, which is the reason the paper has an outlier
mechanism at all.

This measures the reconstruction error TurboQuant actually incurs on the model's
own K/V tensors, layer by layer, and separates the two candidate explanations:

  * per-vector relative L2 error -- how much of each KV vector survives, and
  * the kurtosis / max-to-RMS ratio of the rotated coordinates -- whether the
    Gaussian assumption behind the codebook holds after the rotation.

It also runs the shortest possible context, because error that only appears at
length is accumulation, while error present at 8 tokens is the quantizer.
"""
from __future__ import annotations

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cache import TurboQuantMSE   # noqa: E402
from model_guard import guarded_from_pretrained   # noqa: E402

MODEL = "Qwen/Qwen2.5-7B-Instruct"


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

    cache = DynamicCache()
    model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)

    tq = TurboQuantMSE(4, hd, "cuda")
    print(f"real K/V reconstruction error, {MODEL}, 4-bit, {ids.shape[1]} tokens\n")
    print(f"{'layer':>6}{'tensor':>8}{'rel_L2':>9}{'max|x|/rms':>12}"
          f"{'rot_kurtosis':>14}{'clip_frac':>11}")
    print("-" * 60)

    stats = []
    for li in (0, 1, 7, 14, 21, 27):
        for name in ("keys", "values"):
            x = getattr(cache.layers[li], name).float()      # (B,H,L,D)
            flat = x.reshape(-1, hd)
            norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-10)
            unit = flat / norms
            y = unit @ tq.Pi.T                                # rotated coords

            # Does the rotated coordinate distribution match the N(0, 1/sqrt(d))
            # the codebook was fitted to?
            kurt = ((y - y.mean()) ** 4).mean() / (y.var() ** 2)
            # Fraction of coordinates outside the outermost codebook cell, i.e.
            # values the codebook cannot represent except by clipping.
            edge = tq.centroids.abs().max()
            clip = (y.abs() > edge).float().mean()

            idx, nm = tq.quantize(x)
            rec = torch.empty_like(x)
            tq.dequantize_into(idx, nm, rec)
            rel = ((rec - x).norm(dim=-1) / x.norm(dim=-1).clamp(min=1e-10))
            m2r = (x.abs().max(dim=-1).values
                   / x.pow(2).mean(dim=-1).sqrt()).mean()

            print(f"{li:>6}{name:>8}{rel.mean().item():>9.3f}{m2r.item():>12.2f}"
                  f"{kurt.item():>14.2f}{clip.item()*100:>10.2f}%")
            stats.append(rel.mean().item())

    print(f"\nmean relative L2 error across sampled layers: "
          f"{sum(stats)/len(stats):.3f}")
    print("(a Gaussian would have kurtosis 3.0; higher means heavier tails than "
          "the codebook assumes)")

    # --- does short context survive? -------------------------------------
    print("\n=== generated text vs context length (4-bit TQ) ===")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cache import TQPackedCache
    for n_ctx in (8, 32, 128, 512):
        sub = ids[:, :n_ctx]
        for arm in ("fp16", "tq4"):
            c = (DynamicCache() if arm == "fp16" else
                 TQPackedCache(cfg, bit_width=4, max_cache_len=n_ctx + 24,
                               device="cuda"))
            o = model(input_ids=sub, past_key_values=c, use_cache=True,
                      logits_to_keep=1)
            nx = o.logits[:, -1, :].argmax(-1, keepdim=True)
            got = [nx.item()]
            for _ in range(15):
                o = model(input_ids=nx, past_key_values=c, use_cache=True,
                          logits_to_keep=1)
                nx = o.logits[:, -1, :].argmax(-1, keepdim=True)
                got.append(nx.item())
            print(f"  ctx {n_ctx:>4} {arm:<6} "
                  f"{tok.decode(got, skip_special_tokens=True)!r}")
            del c


if __name__ == "__main__":
    main()
