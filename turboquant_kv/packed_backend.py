"""Transformers integration: attention that reads the paged packed cache.

`PagedPackedCache.update` quantizes the new K/V into pool pages and returns a
`PackedKV` handle in place of reconstructed tensors. The attention function
registered as ``winnow_packed`` receives that handle as `key`/`value`:

* decode (one query token per sequence) runs the compressed-domain operator,
  so no (B, Hkv, L, D) history is ever built;
* a prefill chunk (several query tokens, one sequence) dequantizes that one
  sequence's history and runs SDPA with an offset causal mask. This is the
  documented prefill fallback; it attends to the same quantized K/V the decode
  path reads, so prefill and decode stay one consistent quantized model.

RoPE is already applied to K before `update` and to Q before attention, as the
model does it; the operator's query rotation happens after RoPE.

Supported model contract (checked by `check_model`): a Qwen2-architecture
decoder with full attention on every layer, no sliding window, head_dim with
head_dim*bit_width % 8 == 0, and outlier channels disabled. Anything else is
rejected before load, not discovered mid-generation.
"""
from __future__ import annotations

from collections import Counter

import torch
import torch.nn.functional as F

import packed_attention as pa
import tuned_dispatch
from cache import _get_quantizer
from packing import pack, packed_bytes, unpack
from page_pool import PagePool, StalePage

ATTN_NAME = "winnow_packed"
SUPPORTED_ARCH = ("Qwen2ForCausalLM",)

# How many attention calls each backend served. The model-integration tests and
# the study read this to prove which path actually ran.
BACKEND_COUNTS: Counter = Counter()

try:                                     # Linux + CUDA + Triton only
    from packed_attention_triton import dense_decode_attention as _triton_dense
    from packed_attention_triton import packed_decode_attention as _triton_decode
except Exception:                        # noqa: BLE001 - absence is the fallback
    _triton_decode = _triton_dense = None

try:
    from kernel import tq_dequant as _fused_dequant
except Exception:                        # noqa: BLE001
    _fused_dequant = None


class UnsupportedModel(ValueError):
    pass


def check_model(config, bit_width: int) -> None:
    arch = (getattr(config, "architectures", None) or ["?"])[0]
    if arch not in SUPPORTED_ARCH:
        raise UnsupportedModel(f"{arch} is not a supported architecture {SUPPORTED_ARCH}")
    if getattr(config, "use_sliding_window", False):
        raise UnsupportedModel("sliding-window attention is not supported")
    types = set(getattr(config, "layer_types", None) or ["full_attention"])
    if types != {"full_attention"}:
        raise UnsupportedModel(f"layer types {sorted(types)} are not all full attention")
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    packed_bytes(head_dim, bit_width)     # raises on an unpackable width


def head_dim_of(config) -> int:
    return getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads


def make_pool(config, *, bit_width: int | None, num_pages: int, page_size: int, device,
              max_pages_per_request=None, fence_factory=None,
              dense_dtype: torch.dtype | None = None) -> PagePool:
    """Packed pool at `bit_width`, or with `dense_dtype` an unquantized pool."""
    D = head_dim_of(config)
    nb = D if dense_dtype is not None else packed_bytes(D, bit_width)
    return PagePool(num_layers=config.num_hidden_layers, num_pages=num_pages,
                    num_kv_heads=config.num_key_value_heads, page_size=page_size,
                    nb_k=nb, nb_v=nb, device=device,
                    max_pages_per_request=max_pages_per_request,
                    fence_factory=fence_factory, dense_dtype=dense_dtype)


class PagedPackedCache:
    """Duck-typed cache for one forward over a batch of pool-backed sequences.

    `owners[b]` is the request id whose pages hold sequence b and `lengths[b]`
    the tokens already stored. The forward appends the same number of tokens n
    to every sequence (n = 1 in batched decode, n = chunk in single-sequence
    prefill). Lengths advance after the last layer has written.
    """

    def __init__(self, pool: PagePool, owners, lengths, *, k_quant, v_quant,
                 head_dim: int, decode_backend: str = "auto"):
        self.pool = pool
        self.owners = list(owners)
        self.lengths = list(lengths)
        self.kq, self.vq = k_quant, v_quant
        self.head_dim = head_dim
        self.decode_backend = decode_backend
        self._table = None
        self._seq = None
        self._refs = None

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if len(set(self.lengths)) != 1:
            raise RuntimeError("mixed lengths: pass explicit position_ids")
        return self.lengths[0]

    def _prepare(self, n: int, device):
        """Once per forward: check ownership and reservation, build the table.

        Pages are reserved by the scheduler at admission. This never allocates:
        a forward racing a release must fail, not quietly re-acquire pages for a
        request that has already been finished and would never free them again.
        """
        refs = [self.pool.held(o) for o in self.owners]
        for o, r, L in zip(self.owners, refs, self.lengths):
            if self.pool.pages_for(L + n) > len(r):
                raise StalePage(f"{o} holds {len(r)} pages, needs {self.pool.pages_for(L + n)}")
            self.pool.validate(o, r)
        width = max(len(r) for r in refs)
        table = torch.full((len(refs), width), -1, dtype=torch.int32)
        for b, r in enumerate(refs):
            table[b, :len(r)] = torch.tensor([x.page for x in r], dtype=torch.int32)
        self._refs = refs
        self._table = table.to(device, non_blocking=True)
        self._seq = torch.tensor([L + n for L in self.lengths], dtype=torch.int32).to(
            device, non_blocking=True)

    @torch.no_grad()
    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        B, Hkv, n, D = key_states.shape
        if B != len(self.owners):
            raise RuntimeError(f"batch {B} != {len(self.owners)} owners")
        if layer_idx == 0:
            self._prepare(n, key_states.device)
        S = self.pool.page_size
        pos = torch.tensor([[L + i for i in range(n)] for L in self.lengths])   # (B, n)
        pids = torch.gather(self._table.cpu().long(), 1, pos // S).to(key_states.device)
        slots = (pos % S).to(key_states.device)
        if self.pool.dense:
            for src, pages in ((key_states, self.pool.k_pages), (value_states, self.pool.v_pages)):
                pages[layer_idx][pids, :, slots] = src.permute(0, 2, 1, 3).to(pages.dtype)
            handle = pa.DenseKV(self.pool.k_pages[layer_idx], self.pool.v_pages[layer_idx],
                                self._table, self._seq, D)
            handle.decode_backend = self.decode_backend
            if layer_idx == self.pool.num_layers - 1:
                self.lengths = [L + n for L in self.lengths]
            return handle, handle
        for src, tq, pages, norms in (
                (key_states, self.kq, self.pool.k_pages, self.pool.k_norms),
                (value_states, self.vq, self.pool.v_pages, self.pool.v_norms)):
            idx, nrm = tq.quantize(src)                                  # (B,Hkv,n,D),(B,Hkv,n)
            packed = pack(idx, tq.bit_width).permute(0, 2, 1, 3)         # (B,n,Hkv,NB)
            pages[layer_idx][pids, :, slots] = packed
            norms[layer_idx][pids, :, slots] = nrm.permute(0, 2, 1).to(torch.float16)
        handle = pa.PackedKV(
            self.pool.k_pages[layer_idx], self.pool.k_norms[layer_idx],
            self.pool.v_pages[layer_idx], self.pool.v_norms[layer_idx],
            self._table, self._seq, pa.Quantizer.from_tq(self.kq),
            pa.Quantizer.from_tq(self.vq), D)
        handle.decode_backend = self.decode_backend
        if layer_idx == self.pool.num_layers - 1:
            self.lengths = [L + n for L in self.lengths]
        return handle, handle


def select_decode_backend(query: torch.Tensor, kv, requested: str) -> str:
    """Which operator serves a decode step. Unsupported requests fail loudly."""
    if isinstance(kv, pa.DenseKV):
        if requested in ("auto", "triton_dense"):
            return "triton_dense" if query.is_cuda and _triton_dense is not None else "dense_sdpa"
        if requested != "dense_sdpa":
            raise ValueError(f"backend {requested!r} does not read dense pages")
        return requested
    if requested == "auto":
        if query.is_cuda and _triton_decode is not None:
            return "triton_packed"
        return "tiled_reference"
    if requested == "triton_packed" and (_triton_decode is None or not query.is_cuda):
        raise RuntimeError("triton_packed requested but Triton/CUDA is unavailable")
    if requested not in {"triton_packed", "tiled_reference", "reconstruct_sdpa"}:
        raise ValueError(f"unknown decode backend {requested!r}")
    return requested


def dequantize_history(kv: pa.PackedKV, pages, norms, quant, b: int, dtype) -> torch.Tensor:
    """(1, Hkv, L, D) reconstruction of sequence b. Prefill/fallback only."""
    L = int(kv.seq_lens[b])
    rows, nrm = pa._gather_history(pages, norms, kv, b)                  # (Hkv,L,NB),(Hkv,L)
    out = torch.empty(1, rows.shape[0], L, kv.head_dim, dtype=dtype, device=rows.device)
    if rows.is_cuda and dtype == torch.bfloat16:
        # Promoted tsc-generated kernel, else the handwritten one (tuned_dispatch).
        tuned_dispatch.DISPATCH.dequant(rows.contiguous(), nrm.contiguous(), quant.centroids,
                                        quant.Pi, quant.bit_width, kv.head_dim, out[0])
    elif rows.is_cuda and _fused_dequant is not None:
        _fused_dequant(rows.contiguous(), nrm.contiguous(), quant.centroids, quant.Pi,
                       quant.bit_width, kv.head_dim, out=out[0])
    else:
        idx = unpack(rows, quant.bit_width, kv.head_dim)
        y = quant.centroids.index_select(0, idx.reshape(-1).to(torch.int32)).view(idx.shape)
        out[0] = ((y @ quant.Pi) * nrm.float()[..., None]).to(dtype)
    return out


def _reconstruct_sdpa(query, kv: pa.PackedKV, scaling):
    """Per sequence: dequantize, then SDPA with an offset causal mask."""
    B, Hq, Q, D = query.shape
    out = torch.zeros(B, Hq, Q, D, dtype=query.dtype, device=query.device)
    for b in range(B):
        L = int(kv.seq_lens[b])
        if L == 0:
            continue
        k = dequantize_history(kv, kv.k_pages, kv.k_norms, kv.k_quant, b, query.dtype)
        v = dequantize_history(kv, kv.v_pages, kv.v_norms, kv.v_quant, b, query.dtype)
        mask = torch.arange(L, device=query.device)[None, :] <= (
            L - Q + torch.arange(Q, device=query.device))[:, None]
        out[b] = F.scaled_dot_product_attention(
            query[b:b + 1], k, v, attn_mask=mask, scale=scaling,
            enable_gqa=Hq != k.shape[1])[0]
    return out


def _dense_sdpa(query, kv: pa.DenseKV, scaling):
    """Per sequence: gather unquantized pages, SDPA with an offset causal mask."""
    B, Hq, Q, D = query.shape
    out = torch.zeros(B, Hq, Q, D, dtype=query.dtype, device=query.device)
    S = kv.page_size
    for b in range(B):
        L = int(kv.seq_lens[b])
        if L == 0:
            continue
        ids = kv.page_table[b, :(L + S - 1) // S].long()
        hist = lambda p: p[ids].permute(1, 0, 2, 3).reshape(p.shape[1], -1, D)[None, :, :L]
        mask = torch.arange(L, device=query.device)[None, :] <= (
            L - Q + torch.arange(Q, device=query.device))[:, None]
        out[b] = F.scaled_dot_product_attention(
            query[b:b + 1], hist(kv.k_pages).to(query.dtype), hist(kv.v_pages).to(query.dtype),
            attn_mask=mask, scale=scaling, enable_gqa=Hq != kv.num_kv_heads)[0]
    return out


def _tuned_schedule(query, kv) -> dict:
    """Launch schedule for this decode step: promoted record or the default."""
    from tsc.schedule import AttnProblem
    dense = isinstance(kv, pa.DenseKV)
    prob = AttnProblem(query.shape[0], int(kv.page_table.shape[1]) * kv.page_size, query.shape[1],
                       kv.num_kv_heads, kv.head_dim, kv.page_size,
                       0 if dense else kv.k_quant.bit_width, dense)
    d = tuned_dispatch.DISPATCH.attention_schedule(prob.family(), prob.batch, prob.seq_len,
                                                   query.device)
    return d.schedule or {}


def winnow_packed_attention(module, query, key, value, attention_mask,
                            dropout: float = 0.0, scaling=None, **kwargs):
    """`AttentionInterface` entry point. Returns (B, Q, Hq, D), None."""
    scaling = scaling if scaling is not None else query.shape[-1] ** -0.5
    if not isinstance(key, (pa.PackedKV, pa.DenseKV)):
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        BACKEND_COUNTS["native_sdpa"] += 1
        return sdpa_attention_forward(module, query, key, value, attention_mask,
                                      dropout=dropout, scaling=scaling, **kwargs)
    Q = query.shape[2]
    if isinstance(key, pa.DenseKV):
        backend = "dense_sdpa" if Q > 1 else select_decode_backend(
            query, key, getattr(key, "decode_backend", "auto"))
        BACKEND_COUNTS[("prefill_" if Q > 1 else "") + backend] += 1
        out = (_triton_dense(query, key, scaling, **_tuned_schedule(query, key))
               if backend == "triton_dense"
               else _dense_sdpa(query, key, scaling))
        return out.transpose(1, 2).contiguous(), None
    if Q > 1:
        BACKEND_COUNTS["prefill_reconstruct_sdpa"] += 1
        out = _reconstruct_sdpa(query, key, scaling)
    else:
        backend = select_decode_backend(query, key, getattr(key, "decode_backend", "auto"))
        BACKEND_COUNTS[backend] += 1
        if backend == "triton_packed":
            out = _triton_decode(query, key, scaling, **_tuned_schedule(query, key))
        elif backend == "tiled_reference":
            out = pa.tiled_compressed_attention(query, key, scaling).to(query.dtype)
        else:
            out = _reconstruct_sdpa(query, key, scaling)
    return out.transpose(1, 2).contiguous(), None


def register() -> None:
    from transformers import AttentionInterface
    AttentionInterface.register(ATTN_NAME, winnow_packed_attention)


def quantizers_for(config, bit_width: int, device, k_seed: int = 42, v_seed: int = 42):
    """Default: K and V share seed 42, the same quantizer `TQPackedCache` uses,
    so the paged path and the existing reconstruct path store identical bytes."""
    D = head_dim_of(config)
    return (_get_quantizer(bit_width, D, device, seed=k_seed),
            _get_quantizer(bit_width, D, device, seed=v_seed))
