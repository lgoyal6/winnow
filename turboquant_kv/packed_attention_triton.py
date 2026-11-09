"""Triton decode attention over paged packed KV (one query token per sequence).

Same algebra as `packed_attention.tiled_compressed_attention`, which is the
reference these kernels are tested against:

    q_rot = scale * q Pi_K^T                         (one batched GEMM, torch)
    per page: s = n_K * (q_rot . c_K);  online softmax;  a += p n_V c_V
    out = (a / l) Pi_V                               (one batched GEMM, torch)

Grid is (batch, kv_head, split). A program owns one kv head, so it decodes each
K/V page once and scores it against all `n_rep` query heads of its GQA group,
padded to BLOCK_H rows for `tl.dot`. Codebook gathers read a 2**bw-entry fp32
table that stays in L1. No program reads a page outside its sequence's page
table, and no (L, D) history is written anywhere.

With `num_splits > 1` each program covers a contiguous range of page slots and
writes compact partial (m, l, a) rows; a merge kernel combines them with the same
stable rescaling as the reference. That helps when B * Hkv programs cannot fill
the GPU; whether it does is a measured question (bench_packed_attention.py).

Contract: CUDA tensors; page_size a power of two >= 16; head_dim a power of two
>= 16; uint8 pages; fp16 norms; int32 page_table/seq_lens. Violations raise
before launch.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

import packed_attention as pa


@triton.jit
def _unpack_ptrs(row_base, offs_d, row_ok, NB: tl.constexpr, BW: tl.constexpr):
    """(R, D) codebook indices, LSB-first, possibly straddling bytes. Each row
    has its own base pointer: an iteration gathers rows from several pages."""
    if BW == 8:
        return tl.load(row_base[:, None] + offs_d[None, :], mask=row_ok[:, None], other=0).to(tl.int32)
    bit = offs_d[None, :] * BW
    byte0 = bit // 8
    sh = bit % 8
    ptr = row_base[:, None] + byte0
    b0 = tl.load(ptr, mask=row_ok[:, None], other=0).to(tl.int32)
    b1 = tl.load(ptr + 1, mask=row_ok[:, None] & ((byte0 + 1) < NB), other=0).to(tl.int32)
    return ((b0 >> sh) | (b1 << (8 - sh))) & ((1 << BW) - 1)


@triton.jit
def _packed_decode_kernel(
    qrot_ptr, kp_ptr, kn_ptr, vp_ptr, vn_ptr, ck_ptr, cv_ptr, table_ptr, lens_ptr,
    acc_ptr, m_ptr, l_ptr,
    stride_qb, stride_qh,
    stride_pp, stride_ph,            # pages: (P, Hkv, S, NB) strides, bytes
    stride_np, stride_nh,            # norms: (P, Hkv, S) strides
    stride_tb, pages_per_split,
    N_REP: tl.constexpr, D: tl.constexpr, S: tl.constexpr,
    NBK: tl.constexpr, NBV: tl.constexpr, BWK: tl.constexpr, BWV: tl.constexpr,
    BLOCK_H: tl.constexpr, ALLOW_TF32: tl.constexpr, DENSE: tl.constexpr,
    PPI: tl.constexpr,
):
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    split = tl.program_id(2)
    n_splits = tl.num_programs(2)
    Hkv = tl.num_programs(1)

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, D)
    T: tl.constexpr = PPI * S                     # tokens per loop iteration
    offs_t = tl.arange(0, T)
    page_in_iter = offs_t // S
    slot = offs_t % S
    h_ok = offs_h < N_REP
    qh = kvh * N_REP + offs_h

    q = tl.load(qrot_ptr + b * stride_qb + qh[:, None] * stride_qh + offs_d[None, :],
                mask=h_ok[:, None], other=0.0)                         # (BLOCK_H, D) fp32

    L = tl.load(lens_ptr + b)
    n_pages = (L + S - 1) // S
    lo = split * pages_per_split
    hi = tl.minimum(lo + pages_per_split, n_pages)

    m = tl.full((BLOCK_H,), float("-inf"), tl.float32)
    l = tl.zeros((BLOCK_H,), tl.float32)
    acc = tl.zeros((BLOCK_H, D), tl.float32)

    # PPI pages per iteration: more independent loads in flight per program.
    # pages_per_split is a multiple of PPI, so groups never straddle splits.
    for t0 in range(lo, hi, PPI):
        page = t0 + page_in_iter                                       # (T,)
        page_ok = page < hi
        pid = tl.load(table_ptr + b * stride_tb + page, mask=page_ok, other=0).to(tl.int64)
        pos = page * S + slot
        ok = page_ok & (pos < L)
        kbase = kp_ptr + pid * stride_pp + kvh * stride_ph
        vbase = vp_ptr + pid * stride_pp + kvh * stride_ph
        if DENSE:
            # Matched unquantized baseline: same schedule, rows read as-is.
            ck = tl.load(kbase[:, None] + slot[:, None] * D + offs_d[None, :],
                         mask=ok[:, None], other=0.0).to(tl.float32)
            nk = tl.where(ok, 1.0, 0.0)
            nv = nk
        else:
            nk = tl.load(kn_ptr + pid * stride_np + kvh * stride_nh + slot,
                         mask=ok, other=0.0).to(tl.float32)
            nv = tl.load(vn_ptr + pid * stride_np + kvh * stride_nh + slot,
                         mask=ok, other=0.0).to(tl.float32)
            ck = tl.load(ck_ptr + _unpack_ptrs(kbase + slot * NBK, offs_d, ok, NBK, BWK))  # (T, D)
        s = tl.dot(q, tl.trans(ck), allow_tf32=ALLOW_TF32) * nk[None, :]           # (BLOCK_H, T)
        s = tl.where(ok[None, :], s, float("-inf"))

        m_new = tl.maximum(m, tl.max(s, 1))
        # exp(-inf - -inf) must be 0, not NaN: a row with no visible key yet.
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.where(m == float("-inf"), 0.0, tl.exp(m - m_safe))
        p = tl.where(ok[None, :], tl.exp(s - m_safe[:, None]), 0.0)
        l = alpha * l + tl.sum(p, 1)
        if DENSE:
            cv = tl.load(vbase[:, None] + slot[:, None] * D + offs_d[None, :],
                         mask=ok[:, None], other=0.0).to(tl.float32)
        else:
            cv = tl.load(cv_ptr + _unpack_ptrs(vbase + slot * NBV, offs_d, ok, NBV, BWV))  # (T, D)
        acc = acc * alpha[:, None] + tl.dot(p * nv[None, :], cv, allow_tf32=ALLOW_TF32)
        m = m_new

    row = (b * Hkv + kvh) * n_splits + split
    out_rows = row * N_REP + offs_h
    tl.store(acc_ptr + out_rows[:, None] * D + offs_d[None, :], acc, mask=h_ok[:, None])
    tl.store(m_ptr + out_rows, m, mask=h_ok)
    tl.store(l_ptr + out_rows, l, mask=h_ok)


@triton.jit
def _merge_decode_partials(acc_ptr, m_ptr, l_ptr, out_ptr,
                           HQ: tl.constexpr, N_REP: tl.constexpr,
                           NSPLITS: tl.constexpr, BLOCK_S: tl.constexpr,
                           D: tl.constexpr):
    """Reduce split states directly into compact normalized query-head rows."""
    row = tl.program_id(0)
    b, h = row // HQ, row % HQ
    splits = tl.arange(0, BLOCK_S)
    dims = tl.arange(0, D)
    partial_rows = ((b * (HQ // N_REP) + h // N_REP) * NSPLITS + splits) * N_REP + h % N_REP
    valid = splits < NSPLITS
    m = tl.load(m_ptr + partial_rows, mask=valid, other=float("-inf"))
    l = tl.load(l_ptr + partial_rows, mask=valid, other=0.0)
    acc = tl.load(acc_ptr + partial_rows[:, None] * D + dims[None, :],
                  mask=valid[:, None], other=0.0)
    maximum = tl.max(m, 0)
    safe_maximum = tl.where(maximum == float("-inf"), 0.0, maximum)
    weights = tl.where(valid & (m != float("-inf")), tl.exp(m - safe_maximum), 0.0)
    denominator = tl.sum(l * weights, 0)
    numerator = tl.sum(acc * weights[:, None], 0)
    result = tl.where(denominator > 0, numerator / tl.maximum(denominator, 1e-30), 0.0)
    tl.store(out_ptr + row * D + dims, result)


def _pow2(x: int) -> bool:
    return x >= 16 and (x & (x - 1)) == 0


def check_kernel_contract(q: torch.Tensor, kv: pa.PackedKV) -> None:
    pa.check_contract(q, kv)
    if q.shape[2] != 1:
        raise pa.UnsupportedLayout("the Triton kernel serves decode (one query token) only")
    if not q.is_cuda and os.environ.get("TRITON_INTERPRET") != "1":
        raise pa.UnsupportedLayout("the Triton kernel needs CUDA tensors")
    if not _pow2(kv.page_size) or not _pow2(kv.head_dim):
        raise pa.UnsupportedLayout("page_size and head_dim must be powers of two >= 16")
    if kv.k_pages.stride(-1) != 1 or kv.k_pages.stride(-2) != kv.k_pages.shape[-1] \
            or kv.v_pages.stride() != kv.k_pages.stride() or kv.k_norms.stride(-1) != 1 \
            or kv.v_norms.stride() != kv.k_norms.stride():
        raise pa.UnsupportedLayout("pages need contiguous (S, NB) rows and matching K/V strides")


def packed_decode_attention(q: torch.Tensor, kv: pa.PackedKV, scale: float, *,
                            num_splits: int = 1, allow_tf32: bool = False,
                            num_warps: int = 4, num_stages: int = 2,
                            pages_per_iter: int = 1) -> torch.Tensor:
    """q (B, Hq, 1, D) -> (B, Hq, 1, D) in q.dtype."""
    check_kernel_contract(q, kv)
    return _launch(q, kv, scale, dense=False, num_splits=num_splits, allow_tf32=allow_tf32,
                   num_warps=num_warps, num_stages=num_stages, pages_per_iter=pages_per_iter)


def dense_decode_attention(q: torch.Tensor, kv: "pa.DenseKV", scale: float, *,
                           num_splits: int = 1, allow_tf32: bool = False,
                           num_warps: int = 4, num_stages: int = 2,
                           pages_per_iter: int = 1) -> torch.Tensor:
    """Unquantized paged decode with the identical schedule (the matched baseline)."""
    if q.shape[2] != 1 or (not q.is_cuda and os.environ.get("TRITON_INTERPRET") != "1"):
        raise pa.UnsupportedLayout("dense kernel: CUDA decode only")
    if not _pow2(kv.page_size) or not _pow2(kv.head_dim) or kv.k_pages.stride(-1) != 1 \
            or kv.v_pages.stride() != kv.k_pages.stride():
        raise pa.UnsupportedLayout("dense kernel: unsupported page layout")
    return _launch(q, kv, scale, dense=True, num_splits=num_splits, allow_tf32=allow_tf32,
                   num_warps=num_warps, num_stages=num_stages, pages_per_iter=pages_per_iter)


def _launch(q, kv, scale, *, dense, num_splits, allow_tf32, num_warps, num_stages,
            pages_per_iter=1):
    B, Hq, _, D = q.shape
    Hkv = kv.num_kv_heads
    n_rep = Hq // Hkv
    block_h = max(16, triton.next_power_of_2(n_rep))
    max_pages = kv.page_table.shape[1]
    num_splits = max(1, min(num_splits, max_pages))
    ppi = pages_per_iter
    if ppi < 1 or ppi & (ppi - 1):
        raise pa.UnsupportedLayout("pages_per_iter must be a power of two >= 1")
    # Round each split up to whole page groups so a group never crosses splits.
    per_split = triton.cdiv(triton.cdiv(max_pages, num_splits), ppi) * ppi

    if dense:
        q_rot = (q[:, :, 0].float() * scale).contiguous()
        norms = kv.k_pages                      # never read when DENSE
        k_cent = v_cent = q_rot
        nbk = nbv = D
        bwk = bwv = 8
        kn = vn = norms
        n_strides = (0, 0)
    else:
        q_rot = ((q[:, :, 0].float() @ kv.k_quant.Pi.T) * scale).contiguous()   # (B, Hq, D)
        k_cent, v_cent = kv.k_quant.centroids, kv.v_quant.centroids
        nbk, nbv = kv.k_pages.shape[-1], kv.v_pages.shape[-1]
        bwk, bwv = kv.k_quant.bit_width, kv.v_quant.bit_width
        kn, vn = kv.k_norms, kv.v_norms
        n_strides = (kv.k_norms.stride(0), kv.k_norms.stride(1))
    rows = B * Hq * num_splits
    acc = torch.empty(rows, D, dtype=torch.float32, device=q.device)
    m = torch.empty(rows, dtype=torch.float32, device=q.device)
    l = torch.empty(rows, dtype=torch.float32, device=q.device)

    _packed_decode_kernel[(B, Hkv, num_splits)](
        q_rot, kv.k_pages, kn, kv.v_pages, vn,
        k_cent, v_cent, kv.page_table, kv.seq_lens,
        acc, m, l,
        q_rot.stride(0), q_rot.stride(1),
        kv.k_pages.stride(0), kv.k_pages.stride(1),
        n_strides[0], n_strides[1],
        kv.page_table.stride(0), per_split,
        N_REP=n_rep, D=D, S=kv.page_size,
        NBK=nbk, NBV=nbv, BWK=bwk, BWV=bwv,
        BLOCK_H=block_h, ALLOW_TF32=allow_tf32, DENSE=dense, PPI=ppi,
        num_warps=num_warps, num_stages=num_stages,
    )
    # Query rotation is no longer needed. Release it before allocating the
    # merged rows, and release split scratch before the inverse rotation.
    del q_rot, k_cent, v_cent
    a = torch.empty(B, Hq, D, dtype=torch.float32, device=q.device)
    _merge_decode_partials[(B * Hq,)](
        acc, m, l, a, HQ=Hq, N_REP=n_rep, NSPLITS=num_splits,
        BLOCK_S=triton.next_power_of_2(num_splits), D=D,
    )
    del acc, m, l
    if not dense:
        a = a @ kv.v_quant.Pi
    return a.to(q.dtype)[:, :, None, :]
