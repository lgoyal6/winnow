"""
Winnow bench LLM worker (Qwen2.5-14B-Instruct on A100-80GB).

Serves the model-space arms (vanilla_llm, llm_tq) AND acts as the fixed fp16
reader for the token-space compressed arms (lingua, union, intersection).

Copied verbatim from turboquant_modal.py: the image, MODEL_ID, CACHE_DIR,
hf_cache_vol, and _build_tq_classes. The difference is an `answer()` method that
supports BOTH a vanilla fp16 KV path (past_key_values=None) and the TurboQuant
path (past_key_values=TQCache).

    modal deploy experiments/bench/bench_llm_modal.py
"""

import modal

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

CACHE_DIR = "/cache"
hf_cache_vol = modal.Volume.from_name("turboquant-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers", "accelerate", "sentencepiece", "protobuf",
        "scipy", "numpy",
    )
    .env({"HF_HOME": CACHE_DIR})
    # Ship the artifact guard into the image (pins, safetensors, no remote code).
    .add_local_python_source("model_guard")
)

app = modal.App("winnow-bench-llm", image=image)


# --- TurboQuant KV-cache classes (VERBATIM from turboquant_modal.py) ----------
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


@app.cls(
    gpu="A100-80GB",
    volumes={CACHE_DIR: hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=600,
    timeout=3600,
)
class BenchLLM:
    @modal.enter()
    def load(self):
        """Runs once per container start. Mirrors TurboQuantModel.load."""
        import os
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from model_guard import guarded_from_pretrained

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self.torch = torch
        self.tokenizer = guarded_from_pretrained(AutoTokenizer, MODEL_ID, token=token)
        self.model = guarded_from_pretrained(
            AutoModelForCausalLM, MODEL_ID, dtype=torch.float16, device_map="cuda",
            low_cpu_mem_usage=True, token=token,
        )
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        hf_cache_vol.commit()

        cfg = self.model.config
        self.hd = cfg.hidden_size // cfg.num_attention_heads
        self.nl = cfg.num_hidden_layers
        self.nh = cfg.num_key_value_heads
        self.device = torch.device("cuda")

        self.TQCache = _build_tq_classes()
        self.TQCache(self.hd, 4, self.nl, self.device)  # pre-warm 4-bit codebook
        print(
            f"BenchLLM ready: {MODEL_ID} | {self.nl} layers, "
            f"{self.nh} KV heads, head_dim={self.hd}", flush=True,
        )

    @modal.method()
    def answer(self, prompt: str, mode: str = "vanilla", bit_width: int = 4,
               max_new_tokens: int = 48):
        """Greedy answer. mode=='vanilla' -> fp16 KV; mode=='tq' -> TurboQuant KV."""
        torch = self.torch

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        n_in = int(inputs["input_ids"].shape[1])

        if mode == "tq":
            cache = self.TQCache(self.hd, bit_width, self.nl, self.device, 0, 0)
        else:
            cache = None

        with torch.no_grad():
            out = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                past_key_values=cache, return_dict_in_generate=True,
            )

        gen = out.sequences[0][n_in:]
        n_out = int(gen.shape[0])
        seq = n_in + n_out
        fp16_kv = 2 * self.nl * self.nh * seq * self.hd * 2  # K+V, 2 bytes/elem

        if mode == "tq":
            kv_bytes = int(out.past_key_values.mem_bits() // 8)
            eff = float(out.past_key_values.eff_bits())
            kv_compression_x = round(fp16_kv / kv_bytes, 2) if kv_bytes else None
        else:
            kv_bytes = int(fp16_kv)
            eff = 16.0
            kv_compression_x = 1.0

        return {
            "model": MODEL_ID,
            "text": self.tokenizer.decode(gen, skip_special_tokens=True),
            "input_tokens": n_in,
            "output_tokens": n_out,
            "kv_bytes": kv_bytes,
            "fp16_kv_bytes": int(fp16_kv),
            "kv_compression_x": kv_compression_x,
            "eff_bits": round(eff, 2),
        }


@app.local_entrypoint()
def main(prompt: str = "What is the capital of France?", mode: str = "vanilla"):
    out = BenchLLM().answer.remote(prompt, mode=mode)
    print(out["text"])
    print(out)
