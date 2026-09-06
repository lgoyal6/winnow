"""
TurboQuant POC on Modal (A100-80GB).

Goal: send an input prompt -> Mistral-7B-Instruct-v0.3 generates on an A100
using a TurboQuant-compressed KV cache -> prints the output, alongside a
baseline FP16 run for quality comparison and the realized KV-cache compression.

TurboQuant = KV-cache compression from Google Research (ICLR 2026,
arXiv:2504.19874). A fixed random orthogonal rotation maps each KV vector to a
near-Gaussian distribution where per-coordinate Lloyd-Max scalar quantization
is provably near-optimal -- no training, no calibration. Implemented here as a
drop-in `transformers` DynamicCache subclass (TQCache) used as past_key_values.

Usage:
    modal run modal_app.py                              # default prompt, 4-bit TQ
    modal run modal_app.py --prompt "Explain RAG."      # custom prompt
    modal run modal_app.py --bit-width 3                # 3-bit
    modal run modal_app.py --bit-width 3 --outlier-channels 32 --outlier-bits 4  # 3.5-bit
"""

import modal

# Default model: Qwen2.5-14B-Instruct (48 layers, GQA, head_dim=128, ungated,
# Apache-2.0). Validated on a single A100-80GB: ~28GB fp16 weights leave ample
# headroom for the KV cache, so one 80GB A100 is sufficient (no multi-GPU).
# Override with --model-id for other models, e.g. mistralai/Mistral-7B-Instruct-v0.3.
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
ALT_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"  # smaller/faster alternative, ungated

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "sentencepiece",
        "protobuf",
        "scipy",
        "numpy",
    )
    .env({"HF_HOME": "/cache/hf"})
    # Ship the artifact guard into the image (pins, safetensors, no remote code).
    .add_local_python_source("model_guard")
)

app = modal.App("turboquant-poc")
hf_cache = modal.Volume.from_name("turboquant-hf-cache", create_if_missing=True)


# ===========================================================================
# TurboQuant core (inlined for a self-contained Modal image). Faithful to the
# repo's benchmarks/gpu.py inline implementation, which is the version verified
# to work on GPU with the current transformers cache API.
# ===========================================================================
def _build_tq_classes():
    import math
    import numpy as np
    import torch
    from scipy.stats import norm
    from transformers.cache_utils import DynamicCache, DynamicLayer

    def _lloyd_max_gaussian(num_levels, sigma=1.0, max_iter=200):
        k = num_levels
        centroids = np.array([sigma * norm.ppf((2 * i + 1) / (2 * k)) for i in range(k)])
        for _ in range(max_iter):
            boundaries = np.empty(k + 1)
            boundaries[0], boundaries[k] = -np.inf, np.inf
            for i in range(1, k):
                boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0
            new_c = np.empty(k)
            for i in range(k):
                lo, hi = boundaries[i], boundaries[i + 1]
                lo_c, hi_c = max(lo, -6 * sigma), min(hi, 6 * sigma)
                num = norm.expect(lambda x: x, loc=0, scale=sigma, lb=lo_c, ub=hi_c)
                den = norm.cdf(hi, scale=sigma) - norm.cdf(lo, scale=sigma)
                new_c[i] = num / den if den > 1e-15 else (lo_c + hi_c) / 2.0
            if np.allclose(centroids, new_c, atol=1e-12):
                break
            centroids = new_c
        boundaries = np.empty(k + 1)
        boundaries[0], boundaries[k] = -np.inf, np.inf
        for i in range(1, k):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0
        return centroids, boundaries

    class TurboQuantMSE:
        def __init__(self, bit_width, head_dim, device, rotation_seed=42):
            d = head_dim
            gen = torch.Generator(device="cpu").manual_seed(rotation_seed)
            G = torch.randn(d, d, generator=gen, dtype=torch.float32)
            Q, R = torch.linalg.qr(G)
            ds = torch.sign(torch.diag(R)); ds[ds == 0] = 1.0
            self.Pi = (Q * ds.unsqueeze(0)).to(device).contiguous()
            sigma = 1.0 / math.sqrt(d)
            c_np, b_np = _lloyd_max_gaussian(2 ** bit_width, sigma=sigma)
            self.centroids = torch.tensor(c_np, dtype=torch.float32, device=device).contiguous()
            self.boundaries = torch.tensor(b_np[1:-1], dtype=torch.float32, device=device).contiguous()
            self.head_dim = head_dim

        @torch.no_grad()
        def quantize(self, x):
            flat = x.float().reshape(-1, self.head_dim)
            norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-10)
            y = (flat / norms) @ self.Pi.T
            idx = torch.bucketize(y, self.boundaries).to(torch.uint8)
            return idx.view(x.shape), norms.squeeze(-1).view(x.shape[:-1])

        @torch.no_grad()
        def dequantize(self, idx, norms):
            flat_idx = idx.reshape(-1, self.head_dim)
            y_hat = self.centroids[flat_idx.long()]
            x_hat = y_hat @ self.Pi
            x_hat = x_hat * norms.reshape(-1, 1)
            return x_hat.view(idx.shape)

    # Every layer with the same (bit_width, dim, seed) builds an IDENTICAL
    # rotation matrix + Lloyd-Max codebook. Building the codebook is CPU-bound
    # scipy integration (~seconds each); doing it per-layer means 48x redundant
    # work for a 14B model, which can take minutes on a CPU-starved container.
    # Memoize so it's built once and shared across all layers.
    _QUANTIZER_CACHE = {}

    def _get_quantizer(bw, dim, dev, seed=42):
        key = (bw, dim, str(dev), seed)
        q = _QUANTIZER_CACHE.get(key)
        if q is None:
            q = TurboQuantMSE(bw, dim, dev, rotation_seed=seed)
            _QUANTIZER_CACHE[key] = q
        return q

    class TQLayer(DynamicLayer):
        def __init__(self, hd, bw, dev, num_outlier_ch=0, outlier_bw=0):
            super().__init__()
            self._bw = bw
            self._hd = hd
            self._outlier_ch = num_outlier_ch
            self._outlier_bw = outlier_bw
            use_out = num_outlier_ch > 0 and outlier_bw > bw
            self._regular_dim = hd - num_outlier_ch if use_out else hd
            self._outlier_dim = num_outlier_ch if use_out else 0
            self._tq = _get_quantizer(bw, self._regular_dim, dev)
            self._tq_out = _get_quantizer(outlier_bw, num_outlier_ch, dev, seed=43) if self._outlier_dim > 0 else None
            self._key_data, self._val_data = [], []
            self._ck = self._cv = None
            self._channel_mask = None

        def lazy_initialization(self, ks, vs):
            self.dtype, self.device, self.is_initialized = ks.dtype, ks.device, True
            if self._tq_out is not None and self._channel_mask is None:
                rms = ks.float().pow(2).mean(dim=(0, 1, 2)).sqrt()
                _, top = rms.topk(min(self._outlier_ch, rms.shape[0]))
                self._channel_mask = torch.zeros(rms.shape[0], dtype=torch.bool, device=ks.device)
                self._channel_mask[top] = True

        def _quant(self, x):
            shape = x.shape
            if self._tq_out is not None and self._channel_mask is not None:
                reg_mask = ~self._channel_mask
                xf = x.float()
                r = xf[..., reg_mask].reshape(-1, self._regular_dim)
                ri, rn = self._tq.quantize(r)
                o = xf[..., self._channel_mask].reshape(-1, self._outlier_dim)
                oi, on = self._tq_out.quantize(o)
                return {"ri": ri, "rn": rn, "oi": oi, "on": on, "s": shape}
            idx, norms = self._tq.quantize(x.float().reshape(-1, self._hd))
            return {"idx": idx, "norms": norms, "s": shape}

        def _dequant_one(self, d):
            shape = d["s"]
            if "ri" in d:
                r_hat = self._tq.dequantize(d["ri"], d["rn"]).reshape(shape[0], shape[1], shape[2], self._regular_dim)
                o_hat = self._tq_out.dequantize(d["oi"], d["on"]).reshape(shape[0], shape[1], shape[2], self._outlier_dim)
                out = torch.zeros(shape, dtype=torch.float32, device=self.device)
                out[..., ~self._channel_mask] = r_hat
                out[..., self._channel_mask] = o_hat
                return out.to(self.dtype)
            return self._tq.dequantize(d["idx"], d["norms"]).reshape(shape).to(self.dtype)

        def update(self, ks, vs, cache_kwargs=None):
            if not self.is_initialized:
                self.lazy_initialization(ks, vs)
            kd = self._quant(ks); self._key_data.append(kd)
            vd = self._quant(vs); self._val_data.append(vd)
            nk = self._dequant_one(kd)
            nv = self._dequant_one(vd)
            if self._ck is None:
                self._ck, self._cv = nk, nv
            else:
                self._ck = torch.cat([self._ck, nk], dim=-2)
                self._cv = torch.cat([self._cv, nv], dim=-2)
            return self._ck, self._cv

        def get_seq_length(self, *a, **k):
            return sum(d["s"][-2] for d in self._key_data) if self._key_data else 0

        def get_max_cache_shape(self, *a, **k):
            return -1

        def mem_bits(self):
            t = 0
            for d in self._key_data + self._val_data:
                if "ri" in d:
                    t += d["ri"].numel() * self._bw + d["oi"].numel() * self._outlier_bw
                    t += (d["rn"].numel() + d["on"].numel()) * 32
                else:
                    t += d["idx"].numel() * self._bw + d["norms"].numel() * 32
            return t

        def eff_bits(self):
            if self._outlier_dim > 0:
                return (self._regular_dim * self._bw + self._outlier_dim * self._outlier_bw) / self._hd
            return float(self._bw)

        @property
        def keys(self):
            return self._ck if self._ck is not None else torch.tensor([])

        @keys.setter
        def keys(self, v):
            pass

        @property
        def values(self):
            return self._cv if self._cv is not None else torch.tensor([])

        @values.setter
        def values(self, v):
            pass

    class TQCache(DynamicCache):
        def __init__(self, hd, bw, nl, dev, num_outlier_ch=0, outlier_bw=0):
            super().__init__()
            self.layers = [TQLayer(hd, bw, dev, num_outlier_ch, outlier_bw) for _ in range(nl)]

        def mem_bits(self):
            return sum(l.mem_bits() for l in self.layers)

        def eff_bits(self):
            return self.layers[0].eff_bits() if self.layers else 0.0

    return TQCache


def _baseline_kv_bytes(cache):
    """FP16 KV-cache size in bytes from a standard DynamicCache."""
    t = 0
    for l in cache.layers:
        for attr in ("keys", "values"):
            x = getattr(l, attr, None)
            if x is not None and hasattr(x, "numel") and x.numel() > 0:
                t += x.numel() * x.element_size()
    return t


@app.function(
    image=image,
    gpu="A100-80GB",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/cache": hf_cache},
    timeout=3600,
)
def run(prompt: str, bit_width: int = 4, max_new_tokens: int = 200,
        outlier_channels: int = 0, outlier_bits: int = 0, model_id: str = MODEL_ID):
    import os, time, gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    device = torch.device("cuda")
    print(f"=== TurboQuant POC ===", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda} | torch {torch.__version__}", flush=True)
    print(f"Model: {model_id}", flush=True)
    print(f"TurboQuant bit_width={bit_width} outlier_channels={outlier_channels} outlier_bits={outlier_bits}", flush=True)

    print("Loading model + tokenizer...", flush=True)
    from model_guard import guarded_from_pretrained

    tokenizer = guarded_from_pretrained(AutoTokenizer, model_id, token=hf_token)
    model = guarded_from_pretrained(
        AutoModelForCausalLM, model_id,
        dtype=torch.float16, device_map="cuda", low_cpu_mem_usage=True, token=hf_token
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    hf_cache.commit()  # persist downloaded weights to the volume

    hd = model.config.hidden_size // model.config.num_attention_heads
    nl = model.config.num_hidden_layers
    nh = model.config.num_key_value_heads
    print(f"Loaded: {nl} layers, {nh} KV heads, head_dim={hd}", flush=True)

    TQCache = _build_tq_classes()

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    n_in = inputs["input_ids"].shape[1]

    def generate(cache):
        torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.no_grad():
            kw = dict(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                      max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                      return_dict_in_generate=True)
            if cache is not None:
                kw["past_key_values"] = cache
            out = model.generate(**kw)
        torch.cuda.synchronize()
        dt = time.time() - t0
        gen_ids = out.sequences[0][n_in:]
        if cache is None:
            kv_bytes = _baseline_kv_bytes(out.past_key_values)
            eff = 16.0
        else:
            kv_bytes = out.past_key_values.mem_bits() // 8
            eff = out.past_key_values.eff_bits()
        return {
            "text": tokenizer.decode(gen_ids, skip_special_tokens=True),
            "tokens": int(gen_ids.shape[0]),
            "time_s": round(dt, 2),
            "tps": round(int(gen_ids.shape[0]) / dt, 1) if dt > 0 else 0,
            "kv_bytes": int(kv_bytes),
            "eff_bits": round(float(eff), 2),
            "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
        }

    print("\n--- Baseline FP16 ---", flush=True)
    baseline = generate(None)
    print(f"  {baseline['tokens']} tok, {baseline['tps']} tok/s, KV={baseline['kv_bytes']} B", flush=True)

    print(f"\n--- TurboQuant {bit_width}-bit ---", flush=True)
    tq_cache = TQCache(hd, bit_width, nl, device, outlier_channels, outlier_bits)
    tq = generate(tq_cache)
    print(f"  {tq['tokens']} tok, {tq['tps']} tok/s, KV={tq['kv_bytes']} B, eff_bits={tq['eff_bits']}", flush=True)

    ratio = baseline["kv_bytes"] / tq["kv_bytes"] if tq["kv_bytes"] else float("inf")
    result = {
        "model": model_id, "prompt": prompt, "n_input_tokens": n_in,
        "config": {"layers": nl, "kv_heads": nh, "head_dim": hd},
        "baseline": baseline, "turboquant": tq,
        "kv_compression_ratio": round(ratio, 2),
    }

    print("\n" + "=" * 78, flush=True)
    print(f"PROMPT: {prompt}", flush=True)
    print("-" * 78, flush=True)
    print(f"[BASELINE FP16]\n{baseline['text']}", flush=True)
    print("-" * 78, flush=True)
    print(f"[TURBOQUANT {tq['eff_bits']}-bit]\n{tq['text']}", flush=True)
    print("=" * 78, flush=True)
    print(f"KV-cache compression: {ratio:.2f}x  ({baseline['kv_bytes']} B -> {tq['kv_bytes']} B)", flush=True)
    print("=" * 78, flush=True)
    return result


@app.local_entrypoint()
def main(prompt: str = "Explain what KV-cache quantization is and why it matters, in 3 sentences.",
         bit_width: int = 4, max_new_tokens: int = 200,
         outlier_channels: int = 0, outlier_bits: int = 0,
         model_id: str = MODEL_ID):
    # model_id lets us point the SAME TurboQuant cache at any standard
    # MHA/GQA causal LM (Mistral-7B, Qwen2.5-14B, Llama, Phi, ...). The cache
    # adapts to the model's head_dim / layer count / KV-head count via config.
    r = run.remote(prompt, bit_width, max_new_tokens, outlier_channels, outlier_bits, model_id)
    print("\n========== RESULT (local) ==========")
    print(f"Model: {r['model']}")
    print(f"Arch: {r['config']['layers']} layers, {r['config']['kv_heads']} KV heads, head_dim={r['config']['head_dim']}")
    print(f"KV compression: {r['kv_compression_ratio']}x | TQ eff_bits: {r['turboquant']['eff_bits']}")
    print(f"Baseline: {r['baseline']['tps']} tok/s | TurboQuant: {r['turboquant']['tps']} tok/s")
    print(f"\n[TurboQuant output]\n{r['turboquant']['text']}")
