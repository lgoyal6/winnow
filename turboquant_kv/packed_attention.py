"""Attention computed directly on Winnow's compressed KV representation.

The packed cache stores, per head vector x (a row vector of length D),

    n = ||x||,   y = (x / n) Pi^T,   c = codebook[bucketize(y)]

and reconstructs x_hat = n c Pi. Ordinary attention therefore needs the whole
history reconstructed before it can run, which is what `TQPackedLayer._load`
does every step. This module never builds x_hat for the history. Because Pi is
orthogonal and the norm is a scalar per key:

    q . k_hat_j = n_K,j  (q Pi_K^T) . c_K,j

so the query is rotated into the key quantizer's basis once, and every key is
read as packed indices plus one fp16 norm. On the value side

    sum_j p_j v_hat_j = (sum_j p_j n_V,j c_V,j) Pi_V

so the softmax-weighted sum is accumulated in the value quantizer's basis and
rotated back once per output row. K and V may use different rotations and
codebooks; the tests give them different ones so a K/V mix-up cannot cancel.

The history is held in fixed-size pages, as a serving allocator holds it:

    k_pages, v_pages   (P, Hkv, S, NB) uint8   packed indices, NB = D*bw/8
    k_norms, v_norms   (P, Hkv, S)     fp16
    page_table         (B, max_pages)  int32   page ids, in sequence order
    seq_lens           (B,)            int32   valid tokens per sequence

Grouped-query attention maps query head h to kv head h // (Hq // Hkv), which
is the head `repeat_kv` would have copied into position h.

Three implementations live here, all FP32 inside:

* `reconstruct_attention`: dequantize, then ordinary attention. The quantized
  reference the others must match.
* `explicit_compressed_attention`: the algebra above with a single softmax over
  every key. Small, obviously correct, quadratic in memory.
* `tiled_compressed_attention`: one page at a time with an online softmax,
  which is the schedule the GPU kernel follows. It allocates O(B*Hq*S) per tile
  and never a (B, H, L, D) history tensor.

None of these is the fast path. They are what the fast path is tested against.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from packing import packed_bytes, unpack


class UnsupportedLayout(ValueError):
    """Raised before any launch when inputs violate the operator contract."""


@dataclass(frozen=True)
class Quantizer:
    """The pieces of a `TurboQuantMSE` the attention operator reads."""

    Pi: torch.Tensor          # (D, D) fp32, orthogonal
    centroids: torch.Tensor   # (2**bw,) fp32
    bit_width: int

    @classmethod
    def from_tq(cls, tq) -> "Quantizer":
        return cls(tq.Pi, tq.centroids, tq.bit_width)


@dataclass
class PackedKV:
    """Paged packed K/V plus the per-sequence view of it."""

    k_pages: torch.Tensor
    k_norms: torch.Tensor
    v_pages: torch.Tensor
    v_norms: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    k_quant: Quantizer
    v_quant: Quantizer
    head_dim: int

    @property
    def page_size(self) -> int:
        return self.k_pages.shape[2]

    @property
    def num_kv_heads(self) -> int:
        return self.k_pages.shape[1]


@dataclass
class DenseKV:
    """Unquantized paged K/V: pages (P, Hkv, S, D) in the model dtype."""

    k_pages: torch.Tensor
    v_pages: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    head_dim: int

    @property
    def page_size(self) -> int:
        return self.k_pages.shape[2]

    @property
    def num_kv_heads(self) -> int:
        return self.k_pages.shape[1]


def check_contract(q: torch.Tensor, kv: PackedKV) -> None:
    """Reject anything the operators do not support, with the reason.

    q is (B, Hq, Q, D). Every check here is cheap metadata, except the page id
    range check, which reads the page table once.
    """
    if q.dim() != 4:
        raise UnsupportedLayout(f"q must be (B, Hq, Q, D), got {tuple(q.shape)}")
    B, Hq, _, D = q.shape
    if D != kv.head_dim:
        raise UnsupportedLayout(f"q head_dim {D} != cache head_dim {kv.head_dim}")
    for name, quant in (("k", kv.k_quant), ("v", kv.v_quant)):
        if quant.Pi.shape != (D, D) or quant.Pi.dtype != torch.float32:
            raise UnsupportedLayout(f"{name} rotation must be fp32 ({D}, {D})")
        if quant.centroids.dtype != torch.float32 or quant.centroids.numel() != 2 ** quant.bit_width:
            raise UnsupportedLayout(f"{name} codebook must be fp32 with 2**bw entries")
    nb_k = packed_bytes(D, kv.k_quant.bit_width)
    nb_v = packed_bytes(D, kv.v_quant.bit_width)
    P, Hkv, S = kv.k_pages.shape[:3]
    if kv.k_pages.dtype != torch.uint8 or kv.v_pages.dtype != torch.uint8:
        raise UnsupportedLayout("packed pages must be uint8")
    if kv.k_pages.shape != (P, Hkv, S, nb_k) or kv.v_pages.shape[:3] != (P, Hkv, S) \
            or kv.v_pages.shape[3] != nb_v:
        raise UnsupportedLayout(
            f"page shapes {tuple(kv.k_pages.shape)}/{tuple(kv.v_pages.shape)} do not "
            f"match (P, Hkv, S, NB) with NB={nb_k}/{nb_v}")
    if kv.k_norms.shape != (P, Hkv, S) or kv.v_norms.shape != (P, Hkv, S):
        raise UnsupportedLayout("norms must be (P, Hkv, S)")
    if Hq % Hkv:
        raise UnsupportedLayout(f"Hq={Hq} is not a multiple of Hkv={Hkv}")
    if kv.page_table.dim() != 2 or kv.page_table.shape[0] != B:
        raise UnsupportedLayout("page_table must be (B, max_pages)")
    if kv.seq_lens.shape != (B,):
        raise UnsupportedLayout("seq_lens must be (B,)")
    if kv.page_table.dtype != torch.int32 or kv.seq_lens.dtype != torch.int32:
        raise UnsupportedLayout("page_table and seq_lens must be int32")
    devices = {t.device for t in (q, kv.k_pages, kv.v_pages, kv.k_norms, kv.v_norms,
                                  kv.page_table, kv.seq_lens, kv.k_quant.Pi,
                                  kv.v_quant.Pi)}
    if len(devices) != 1:
        raise UnsupportedLayout(f"all operands must share a device, got {devices}")
    lens = kv.seq_lens.to("cpu", torch.int64)
    if (lens < 0).any() or (lens > kv.page_table.shape[1] * S).any():
        raise UnsupportedLayout("seq_lens out of range for the page table")
    used = (lens + S - 1) // S
    table = kv.page_table.to("cpu", torch.int64)
    for b in range(B):
        ids = table[b, :used[b]]
        if ((ids < 0) | (ids >= P)).any():
            raise UnsupportedLayout(f"sequence {b} references a page outside [0, {P})")


# -- the three algebraic pieces, kept separate so the tests can break each one --

def kv_head_for(q_head: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Query head -> kv head under grouped-query attention (repeat_kv order)."""
    return q_head // n_rep


def rotate_query(q: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
    """Row vector q into the quantizer basis: q Pi^T."""
    return q @ Pi.T


def rotate_output(a_rot: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
    """Row vector in the quantizer basis back to the model basis: a Pi."""
    return a_rot @ Pi


def _centroids(pages, quant: Quantizer, head_dim: int) -> torch.Tensor:
    idx = unpack(pages, quant.bit_width, head_dim)
    return quant.centroids.index_select(0, idx.reshape(-1).to(torch.int32)).view(idx.shape)


def _gather_history(pages, norms, kv: PackedKV, b: int):
    """(Hkv, L, NB) packed rows and (Hkv, L) norms for sequence b, in order."""
    L = int(kv.seq_lens[b])
    S = kv.page_size
    ids = kv.page_table[b, :(L + S - 1) // S].long()
    p = pages[ids].permute(1, 0, 2, 3).reshape(pages.shape[1], -1, pages.shape[3])
    n = norms[ids].permute(1, 0, 2).reshape(norms.shape[1], -1)
    return p[:, :L], n[:, :L]


def _causal_limit(kv: PackedKV, Q: int) -> torch.Tensor:
    """(B, Q) index of the last key each query may see, inclusive.

    The cache already holds the Q new tokens, so query i of sequence b sits at
    absolute position seq_lens[b] - Q + i.
    """
    pos = torch.arange(Q, device=kv.seq_lens.device)
    return kv.seq_lens.long()[:, None] - Q + pos[None, :]


def reconstruct_attention(q: torch.Tensor, kv: PackedKV, scale: float) -> torch.Tensor:
    """Dequantize every sequence's history, then plain causal attention."""
    check_contract(q, kv)
    B, Hq, Q, D = q.shape
    n_rep = Hq // kv.num_kv_heads
    out = torch.zeros(B, Hq, Q, D, dtype=torch.float32, device=q.device)
    limit = _causal_limit(kv, Q)
    heads = kv_head_for(torch.arange(Hq, device=q.device), n_rep)
    for b in range(B):
        L = int(kv.seq_lens[b])
        if L == 0:
            continue
        pk, nk = _gather_history(kv.k_pages, kv.k_norms, kv, b)
        pv, nv = _gather_history(kv.v_pages, kv.v_norms, kv, b)
        k = (_centroids(pk, kv.k_quant, D) @ kv.k_quant.Pi) * nk.float()[..., None]
        v = (_centroids(pv, kv.v_quant, D) @ kv.v_quant.Pi) * nv.float()[..., None]
        k, v = k[heads], v[heads]                                    # (Hq, L, D)
        s = (q[b].float() @ k.transpose(-1, -2)) * scale             # (Hq, Q, L)
        allowed = torch.arange(L, device=q.device)[None, :] <= limit[b][:, None]
        s = s.masked_fill(~allowed[None], float("-inf"))
        any_ok = allowed.any(-1)                                     # (Q,)
        p = torch.softmax(s, dim=-1).nan_to_num(0.0)
        out[b] = (p @ v) * any_ok[None, :, None]
    return out


def explicit_compressed_attention(q: torch.Tensor, kv: PackedKV, scale: float) -> torch.Tensor:
    """Compressed-domain algebra, one full softmax per row. No x_hat history."""
    check_contract(q, kv)
    B, Hq, Q, D = q.shape
    n_rep = Hq // kv.num_kv_heads
    heads = kv_head_for(torch.arange(Hq, device=q.device), n_rep)
    out = torch.zeros(B, Hq, Q, D, dtype=torch.float32, device=q.device)
    limit = _causal_limit(kv, Q)
    for b in range(B):
        L = int(kv.seq_lens[b])
        if L == 0:
            continue
        pk, nk = _gather_history(kv.k_pages, kv.k_norms, kv, b)
        pv, nv = _gather_history(kv.v_pages, kv.v_norms, kv, b)
        ck = _centroids(pk, kv.k_quant, D)[heads]                    # (Hq, L, D)
        cv = _centroids(pv, kv.v_quant, D)[heads]
        q_rot = rotate_query(q[b].float(), kv.k_quant.Pi)            # (Hq, Q, D)
        s = (q_rot @ ck.transpose(-1, -2)) * nk.float()[heads][:, None, :] * scale
        allowed = torch.arange(L, device=q.device)[None, :] <= limit[b][:, None]
        s = s.masked_fill(~allowed[None], float("-inf"))
        p = torch.softmax(s, dim=-1).nan_to_num(0.0)
        a_rot = (p * nv.float()[heads][:, None, :]) @ cv              # (Hq, Q, D)
        out[b] = rotate_output(a_rot, kv.v_quant.Pi) * allowed.any(-1)[None, :, None]
    return out


def tiled_compressed_attention(q: torch.Tensor, kv: PackedKV, scale: float,
                               split_pages: int | None = None) -> torch.Tensor:
    """Page-at-a-time online softmax in the quantizer bases.

    Per row the running state is (m, l, a_rot) and each page updates it as

        m' = max(m, max_j s_j);  alpha = exp(m - m')
        l' = alpha l + sum_j exp(s_j - m')
        a' = alpha a + sum_j exp(s_j - m') n_V,j c_V,j

    with exp(-inf - -inf) defined as 0, so an all-masked tile or an empty
    sequence leaves l = 0 and the row returns zeros instead of NaN. Output is
    (a / l) Pi_V, normalized once at the end.

    `split_pages` partitions each sequence's pages into chunks processed as if
    by separate thread blocks, then merges the partial (m, l, a) states the way
    a split-KV kernel must. It exists to test that merge on CPU.
    """
    check_contract(q, kv)
    B, Hq, Q, D = q.shape
    S = kv.page_size
    n_rep = Hq // kv.num_kv_heads
    heads = kv_head_for(torch.arange(Hq, device=q.device), n_rep)
    q_rot = rotate_query(q.float(), kv.k_quant.Pi) * scale            # (B, Hq, Q, D)
    limit = _causal_limit(kv, Q)                                       # (B, Q)
    max_pages = kv.page_table.shape[1]
    chunks = [(0, max_pages)] if not split_pages else [
        (s, min(s + split_pages, max_pages)) for s in range(0, max_pages, split_pages)]

    partials = []
    for lo, hi in chunks:
        m = torch.full((B, Hq, Q), float("-inf"), device=q.device)
        l = torch.zeros(B, Hq, Q, device=q.device)
        a = torch.zeros(B, Hq, Q, D, device=q.device)
        for t in range(lo, hi):
            ids = kv.page_table[:, t].long().clamp(0, kv.k_pages.shape[0] - 1)
            ck = _centroids(kv.k_pages[ids], kv.k_quant, D)[:, heads]  # (B, Hq, S, D)
            cv = _centroids(kv.v_pages[ids], kv.v_quant, D)[:, heads]
            nk = kv.k_norms[ids].float()[:, heads]                     # (B, Hq, S)
            nv = kv.v_norms[ids].float()[:, heads]
            s = (q_rot @ ck.transpose(-1, -2)) * nk[:, :, None, :]     # (B, Hq, Q, S)
            key_pos = t * S + torch.arange(S, device=q.device)
            ok = key_pos[None, None, :] <= limit[:, :, None]           # (B, Q, S)
            s = s.masked_fill(~ok[:, None], float("-inf"))
            m, l, a = _online_update(m, l, a, s, nv, cv)
        partials.append((m, l, a))

    m, l, a = partials[0]
    for part in partials[1:]:
        m, l, a = merge_partials((m, l, a), part)
    out = torch.where(l[..., None] > 0, a / l.clamp(min=1e-30)[..., None], 0.0)
    return rotate_output(out, kv.v_quant.Pi)


def _online_update(m, l, a, s, nv, cv):
    m_new = torch.maximum(m, s.amax(-1))
    alpha = _safe_exp(m, m_new)
    p = _safe_exp(s, m_new[..., None])                                 # (B, Hq, Q, S)
    l = alpha * l + p.sum(-1)
    a = alpha[..., None] * a + (p * nv[:, :, None, :]) @ cv
    return m_new, l, a


def _safe_exp(x, m):
    """exp(x - m), with -inf rows (nothing seen yet) giving 0 rather than NaN."""
    return torch.where(torch.isneginf(m) | torch.isneginf(x), 0.0, torch.exp(x - m))


def merge_partials(left, right):
    """Combine two online-softmax states covering disjoint key ranges."""
    m1, l1, a1 = left
    m2, l2, a2 = right
    m = torch.maximum(m1, m2)
    w1, w2 = _safe_exp(m1, m), _safe_exp(m2, m)
    return m, w1 * l1 + w2 * l2, w1[..., None] * a1 + w2[..., None] * a2


def default_scale(head_dim: int) -> float:
    return 1.0 / math.sqrt(head_dim)


@torch.no_grad()
def pages_from_dense(k: torch.Tensor, v: torch.Tensor, seq_lens, k_tq, v_tq,
                     page_size: int, num_pages: int | None = None,
                     generator: torch.Generator | None = None) -> PackedKV:
    """Quantize dense (B, Hkv, Lmax, D) K/V into scattered pages.

    Pages are assigned in a random order so a kernel that assumes page ids are
    contiguous or monotone fails. Unused page-table slots hold -1.
    """
    from packing import pack

    B, Hkv, _, D = k.shape
    lens = [int(x) for x in seq_lens]
    need = [(L + page_size - 1) // page_size for L in lens]
    total = num_pages or max(sum(need), 1)
    order = torch.randperm(total, generator=generator).tolist()
    nb_k, nb_v = packed_bytes(D, k_tq.bit_width), packed_bytes(D, v_tq.bit_width)
    dev = k.device
    kp = torch.zeros(total, Hkv, page_size, nb_k, dtype=torch.uint8, device=dev)
    vp = torch.zeros(total, Hkv, page_size, nb_v, dtype=torch.uint8, device=dev)
    kn = torch.zeros(total, Hkv, page_size, dtype=torch.float16, device=dev)
    vn = torch.zeros(total, Hkv, page_size, dtype=torch.float16, device=dev)
    table = torch.full((B, max(max(need), 1)), -1, dtype=torch.int32)
    nxt = 0
    for b in range(B):
        for t in range(need[b]):
            pid = order[nxt]
            nxt += 1
            table[b, t] = pid
            lo, hi = t * page_size, min((t + 1) * page_size, lens[b])
            for src, tq, pages, norms in ((k, k_tq, kp, kn), (v, v_tq, vp, vn)):
                idx, nrm = tq.quantize(src[b, :, lo:hi])
                pages[pid, :, :hi - lo] = pack(idx, tq.bit_width)
                norms[pid, :, :hi - lo] = nrm.to(torch.float16)
    return PackedKV(kp, kn, vp, vn, table.to(dev),
                    torch.tensor(lens, dtype=torch.int32, device=dev),
                    Quantizer.from_tq(k_tq), Quantizer.from_tq(v_tq), D)
