"""TurboQuant KV cache that actually holds fewer bytes.

What the existing `TQLayer` does per decode step:

    kd = self._quant(ks); self._key_data.append(kd)   # keep uint8 indices
    nk = self._dequant_one(kd)                        # dequantize the new token
    self._ck = torch.cat([self._ck, nk], dim=-2)      # grow a full-precision cache
    return self._ck, self._cv

So the quantized indices and a full-precision copy of the whole cache are held
at the same time, and the `torch.cat` allocates a second full-precision copy on
every step before freeing the first. That is why the project's own timing report
records peak GPU going *up* under TurboQuant. The 3.8-4.9x figure it reports is
`mem_bits()`, a logical bit count, not bytes the process holds.

Four changes here:

1. **Bit-packing.** Indices go into exactly `bit_width` bits each instead of a
   `uint8` apiece (see `packing.py`). At 4 bits that alone is 2x.
2. **No persistent full-precision cache.** The dequantized tensor handed to
   attention is transient: allocated for the step, freed after it. Persistent
   per-sequence bytes become the packed indices plus one fp16 norm per vector.
   This is the change that matters for concurrency: N sequences hold N packed
   caches and share one transient, rather than N full-precision caches.
3. **Preallocated ring buffer, no `torch.cat`.** Writes land in place at the
   current position. (Note the fp16 baseline `DynamicLayer.update` cats too, so
   this beats the baseline's allocation behaviour, not just TurboQuant's.)
4. **No int64 index blowup.** `centroids[idx.long()]` turned uint8 indices into
   int64, an 8x temporary on the hot path. Dequantization here runs in chunks
   with `index_select` on int32, so the index temporary is 4 bytes rather than 8
   and is bounded by `chunk` rather than by cache length.

The honest cost: dropping the full-precision cache means re-dequantizing the
history on every step, O(length) work per step instead of O(1). Decode gets
slower. That regression is measured in `bench_memory.py` and is the thing the
fused Triton kernel (phase C) exists to remove.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import norm as _norm
from transformers.cache_utils import DynamicCache, DynamicLayer

from packing import pack, packed_bytes, unpack
from dispatch import (
    Hardware, NumericalContract, Shape, default_dispatcher,
)
from native_kernel import existing_native_kernel

try:
    from kernel import tq_dequant as _fused_dequant
except Exception:                       # triton missing / unsupported GPU
    _fused_dequant = None


# ---------------------------------------------------------------------------
# Lloyd-Max codebook (unchanged from the existing implementation)
# ---------------------------------------------------------------------------
def _lloyd_max_gaussian(levels: int, sigma: float = 1.0, iters: int = 200,
                        tol: float = 1e-12):
    """Lloyd-Max centroids/boundaries for N(0, sigma^2)."""
    c = np.linspace(-3 * sigma, 3 * sigma, levels)
    for _ in range(iters):
        b = np.concatenate(([-np.inf], (c[:-1] + c[1:]) / 2.0, [np.inf]))
        new_c = np.empty_like(c)
        for i in range(levels):
            lo, hi = b[i] / sigma, b[i + 1] / sigma
            p = _norm.cdf(hi) - _norm.cdf(lo)
            if p < 1e-300:
                new_c[i] = c[i]
                continue
            # E[x | lo<x<hi] for a zero-mean Gaussian
            new_c[i] = sigma * (_norm.pdf(lo) - _norm.pdf(hi)) / p
        if np.max(np.abs(new_c - c)) < tol:
            c = new_c
            break
        c = new_c
    b = np.concatenate(([-np.inf], (c[:-1] + c[1:]) / 2.0, [np.inf]))
    return c, b


class TurboQuantMSE:
    """Fixed random rotation + per-coordinate Lloyd-Max scalar quantizer."""

    def __init__(self, bit_width: int, head_dim: int, device="cpu",
                 rotation_seed: int = 42):
        d = head_dim
        device = torch.device(device)
        gen = torch.Generator(device="cpu").manual_seed(rotation_seed)
        G = torch.randn(d, d, generator=gen, dtype=torch.float32)
        Q, R = torch.linalg.qr(G)
        ds = torch.sign(torch.diag(R))
        ds[ds == 0] = 1.0
        self.Pi = (Q * ds.unsqueeze(0)).to(device).contiguous()
        sigma = 1.0 / math.sqrt(d)
        c_np, b_np = _lloyd_max_gaussian(2 ** bit_width, sigma=sigma)
        self.centroids = torch.tensor(c_np, dtype=torch.float32,
                                      device=device).contiguous()
        self.boundaries = torch.tensor(b_np[1:-1], dtype=torch.float32,
                                       device=device).contiguous()
        self.bit_width = bit_width
        self.head_dim = head_dim
        self.device = device

    @torch.no_grad()
    def quantize(self, x: torch.Tensor):
        """(..., D) -> (uint8 idx (..., D), fp32 norms (...,))"""
        flat = x.float().reshape(-1, self.head_dim)
        norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        y = (flat / norms) @ self.Pi.T
        idx = torch.bucketize(y, self.boundaries).to(torch.uint8)
        return idx.view(x.shape), norms.squeeze(-1).view(x.shape[:-1])

    @torch.no_grad()
    def dequantize_into(self, idx: torch.Tensor, norms: torch.Tensor,
                        out: torch.Tensor) -> None:
        """Dequantize `idx` and write into `out` (any dtype), no int64 temp.

        `index_select` takes int32, so the gather index is 4 bytes per element
        instead of the 8 that `centroids[idx.long()]` produced.
        """
        D = self.head_dim
        flat = idx.reshape(-1, D)
        y_hat = self.centroids.index_select(
            0, flat.reshape(-1).to(torch.int32)).view(-1, D)
        x_hat = y_hat @ self.Pi
        x_hat *= norms.reshape(-1, 1).to(x_hat.dtype)
        out.copy_(x_hat.view(out.shape))


_QUANTIZER_CACHE: dict = {}


def _get_quantizer(bw, dim, dev, seed=42):
    """Codebooks are identical for equal (bit_width, dim, seed); build once.

    Building one costs seconds of scipy integration, and a 48-layer model would
    otherwise build 48 identical copies.
    """
    key = (bw, dim, str(dev), seed)
    q = _QUANTIZER_CACHE.get(key)
    if q is None:
        q = TurboQuantMSE(bw, dim, dev, rotation_seed=seed)
        _QUANTIZER_CACHE[key] = q
    return q


# ---------------------------------------------------------------------------
# Packed layer
# ---------------------------------------------------------------------------
class TQPackedLayer(DynamicLayer):
    """Per-layer packed KV storage. Shapes are (batch, kv_heads, seq, head_dim)."""

    def __init__(self, head_dim: int, bit_width: int, device,
                 max_cache_len: int, num_outlier_channels: int = 0,
                 outlier_bits: int = 0, chunk: int = 1024,
                 use_kernel: bool = True, allow_tf32: bool = True,
                 kernel_backend: str = "auto", native_kernel=None,
                 dispatcher=None):
        super().__init__()
        # The fused kernel covers the plain (no outlier channel) path; the
        # outlier variant splits the head dim between two codebooks and still
        # goes through the chunked torch path.
        self.use_kernel = use_kernel
        self.allow_tf32 = allow_tf32
        self.kernel_backend = kernel_backend if use_kernel else "torch_fp32"
        if self.kernel_backend not in {
            "auto", "torch_fp32", "triton_fp32", "triton_tf32",
            "native_cuda_fp32",
        }:
            raise ValueError(f"unknown kernel_backend={self.kernel_backend!r}")
        self._native_dequant = native_kernel or existing_native_kernel()
        self._dispatcher = dispatcher or default_dispatcher()
        self.last_dispatch_decision = None
        self._hardware = None
        self._bw = bit_width
        self._hd = head_dim
        self._max = max_cache_len
        self._chunk = chunk
        use_out = num_outlier_channels > 0 and outlier_bits > bit_width
        self._reg_dim = head_dim - num_outlier_channels if use_out else head_dim
        self._out_dim = num_outlier_channels if use_out else 0
        self._out_bw = outlier_bits
        self._tq = _get_quantizer(bit_width, self._reg_dim, device)
        self._tq_out = (_get_quantizer(outlier_bits, self._out_dim, device,
                                        seed=43) if self._out_dim else None)
        self._nb_reg = packed_bytes(self._reg_dim, bit_width)
        self._nb_out = packed_bytes(self._out_dim, outlier_bits) if self._out_dim else 0
        self._len = 0
        self._mask = None
        self.is_initialized = False

    # -- allocation ------------------------------------------------------
    def lazy_initialization(self, ks: torch.Tensor, vs: torch.Tensor) -> None:
        B, H, _, D = ks.shape
        self.dtype, self.device = ks.dtype, ks.device
        self._hardware = Hardware.from_torch(torch, self.device)
        dev = ks.device
        def buf(nb):
            return torch.empty(B, H, self._max, nb, dtype=torch.uint8, device=dev)
        self._pk_r, self._pv_r = buf(self._nb_reg), buf(self._nb_reg)
        if self._out_dim:
            self._pk_o, self._pv_o = buf(self._nb_out), buf(self._nb_out)
        # fp16 norms: the norm is a scale, and fp16's ~1e-3 relative error is two
        # orders below 4-bit quantization noise. Verified in test_cache.py.
        self._nk_r = torch.empty(B, H, self._max, dtype=torch.float16, device=dev)
        self._nv_r = torch.empty(B, H, self._max, dtype=torch.float16, device=dev)
        if self._out_dim:
            self._nk_o = torch.empty(B, H, self._max, dtype=torch.float16, device=dev)
            self._nv_o = torch.empty(B, H, self._max, dtype=torch.float16, device=dev)
            rms = ks.float().pow(2).mean(dim=(0, 1, 2)).sqrt()
            _, top = rms.topk(min(self._out_dim, rms.shape[0]))
            self._mask = torch.zeros(D, dtype=torch.bool, device=dev)
            self._mask[top] = True
        self._shape = (B, H, D)
        self.is_initialized = True

    # -- write -----------------------------------------------------------
    def _store(self, x, pos, n, p_r, n_r, p_o, n_o):
        """Quantize x (B,H,n,D) and write packed indices at [pos, pos+n)."""
        if self._out_dim:
            xf = x.float()
            r = xf[..., ~self._mask]
            i, nm = self._tq.quantize(r)
            p_r[:, :, pos:pos + n] = pack(i, self._bw)
            n_r[:, :, pos:pos + n] = nm.to(torch.float16)
            o = xf[..., self._mask]
            i, nm = self._tq_out.quantize(o)
            p_o[:, :, pos:pos + n] = pack(i, self._out_bw)
            n_o[:, :, pos:pos + n] = nm.to(torch.float16)
        else:
            i, nm = self._tq.quantize(x)
            p_r[:, :, pos:pos + n] = pack(i, self._bw)
            n_r[:, :, pos:pos + n] = nm.to(torch.float16)

    # -- read ------------------------------------------------------------
    def _load(self, length, p_r, n_r, p_o, n_o):
        """Dequantize [0, length) into a fresh transient tensor."""
        B, H, D = self._shape
        out = torch.empty(B, H, length, D, dtype=self.dtype, device=self.device)

        backend = "torch_fp32"
        if self.use_kernel and not self._out_dim:
            if self.kernel_backend == "auto":
                available = {"torch_fp32"}
                if _fused_dequant is not None:
                    available.update({"triton_fp32", "triton_tf32"})
                if self._native_dequant is not None:
                    available.add("native_cuda_fp32")
                contract = (
                    NumericalContract.quantization_aware()
                    if self.allow_tf32 else NumericalContract.reference_exact()
                )
                decision = self._dispatcher.select(
                    self._hardware,
                    Shape(B, H, length, D, self._bw), contract, available,
                )
                backend = decision.backend
                self.last_dispatch_decision = decision
            else:
                backend = self.kernel_backend

        if backend in {"triton_fp32", "triton_tf32"}:
            if _fused_dequant is None:
                raise RuntimeError(f"requested {backend}, but Triton is unavailable")
            # One launch for the whole history: unpack, gather, rotate and
            # rescale in registers, writing the output dtype directly.
            _fused_dequant(p_r[:, :, :length], n_r[:, :, :length],
                           self._tq.centroids, self._tq.Pi, self._bw, D,
                           out=out, allow_tf32=(backend == "triton_tf32"))
            return out

        if backend == "native_cuda_fp32":
            if self._native_dequant is None:
                raise RuntimeError(
                    "requested native_cuda_fp32, but no built extension was provided"
                )
            self._native_dequant(
                p_r[:, :, :length], n_r[:, :, :length],
                self._tq.centroids, self._tq.Pi, self._bw, D, out=out,
            )
            return out

        for s in range(0, length, self._chunk):
            e = min(s + self._chunk, length)
            if self._out_dim:
                # Two quantizers write disjoint channel sets of the same slice.
                sl = out[:, :, s:e]
                tmp_r = torch.empty(B, H, e - s, self._reg_dim,
                                    dtype=self.dtype, device=self.device)
                self._tq.dequantize_into(
                    unpack(p_r[:, :, s:e], self._bw, self._reg_dim),
                    n_r[:, :, s:e], tmp_r)
                tmp_o = torch.empty(B, H, e - s, self._out_dim,
                                    dtype=self.dtype, device=self.device)
                self._tq_out.dequantize_into(
                    unpack(p_o[:, :, s:e], self._out_bw, self._out_dim),
                    n_o[:, :, s:e], tmp_o)
                sl[..., ~self._mask] = tmp_r
                sl[..., self._mask] = tmp_o
            else:
                self._tq.dequantize_into(
                    unpack(p_r[:, :, s:e], self._bw, self._hd),
                    n_r[:, :, s:e], out[:, :, s:e])
        return out

    # -- cache protocol --------------------------------------------------
    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        n = key_states.shape[-2]
        if self._len + n > self._max:
            raise RuntimeError(
                f"TQPackedLayer overflow: {self._len}+{n} > max_cache_len="
                f"{self._max}. Raise max_cache_len.")
        po = getattr(self, "_pk_o", None)
        no = getattr(self, "_nk_o", None)
        self._store(key_states, self._len, n, self._pk_r, self._nk_r, po, no)
        po = getattr(self, "_pv_o", None)
        no = getattr(self, "_nv_o", None)
        self._store(value_states, self._len, n, self._pv_r, self._nv_r, po, no)
        self._len += n
        k = self._load(self._len, self._pk_r, self._nk_r,
                       getattr(self, "_pk_o", None), getattr(self, "_nk_o", None))
        v = self._load(self._len, self._pv_r, self._nv_r,
                       getattr(self, "_pv_o", None), getattr(self, "_nv_o", None))
        return k, v

    def get_seq_length(self, *a, **k) -> int:
        return self._len

    def get_mask_sizes(self, query_length: int, *a, **k):
        return self._len, 0

    def get_max_cache_shape(self, *a, **k):
        return self._max

    def reset(self):
        self._len = 0

    # -- accounting ------------------------------------------------------
    def persistent_bytes(self) -> int:
        """Bytes this layer holds between steps (what limits concurrency)."""
        if not self.is_initialized:
            return 0
        tot = 0
        for name in ("_pk_r", "_pv_r", "_pk_o", "_pv_o",
                     "_nk_r", "_nv_r", "_nk_o", "_nv_o"):
            t = getattr(self, name, None)
            if t is not None:
                tot += t.numel() * t.element_size()
        return tot

    def persistent_bytes_used(self) -> int:
        """Same, counting only the `_len` positions actually written."""
        if not self.is_initialized or self._max == 0:
            return 0
        return int(self.persistent_bytes() * self._len / self._max)

    def transient_bytes(self) -> int:
        """Bytes of the dequantized K and V handed to attention this step."""
        if not self.is_initialized:
            return 0
        B, H, D = self._shape
        return 2 * B * H * self._len * D * torch.tensor(
            [], dtype=self.dtype).element_size()


class TQPackedCache(DynamicCache):
    """Drop-in `past_key_values` built from `TQPackedLayer`s."""

    def __init__(self, config, bit_width: int, max_cache_len: int,
                 device="cuda", num_outlier_channels: int = 0,
                 outlier_bits: int = 0, chunk: int = 1024,
                 use_kernel: bool = True, allow_tf32: bool = True,
                 kernel_backend: str = "auto", native_kernel=None,
                 dispatcher=None):
        head_dim = (getattr(config, "head_dim", None)
                    or config.hidden_size // config.num_attention_heads)
        n_layers = config.num_hidden_layers
        super().__init__()
        self.layers = [
            TQPackedLayer(head_dim, bit_width, device, max_cache_len,
                          num_outlier_channels, outlier_bits, chunk,
                          use_kernel, allow_tf32, kernel_backend,
                          native_kernel, dispatcher)
            for _ in range(n_layers)
        ]
        self.bit_width = bit_width
        self.head_dim = head_dim
        self.append_new_layers = lambda *a, **k: None

    def persistent_bytes(self) -> int:
        return sum(l.persistent_bytes() for l in self.layers)

    def persistent_bytes_used(self) -> int:
        return sum(l.persistent_bytes_used() for l in self.layers)

    def transient_bytes(self) -> int:
        return max((l.transient_bytes() for l in self.layers), default=0)

    def effective_bits(self) -> float:
        """Stored bits per KV element, counting the norm. This is the honest
        version of `eff_bits()`: it includes the norm and assumes packing."""
        l = self.layers[0]
        if not l.is_initialized:
            return float("nan")
        per_vec_bits = 8 * (l._nb_reg + l._nb_out) + 16 * (2 if l._out_dim else 1)
        return per_vec_bits / self.head_dim

    def get_seq_length(self, layer_idx: int = 0, *a, **k) -> int:
        return self.layers[layer_idx].get_seq_length()
