"""Phase C: one Triton kernel for the whole TurboQuant dequantization.

The PyTorch path is four passes over the data, each writing a full intermediate
to HBM before the next one reads it back:

    idx   = unpack(packed)                       # (N, D) uint8
    y_hat = centroids.index_select(0, idx.int()) # (N, D) fp32   <- 4 bytes/elem
    x_hat = y_hat @ Pi                           # (N, D) fp32   <- 4 bytes/elem
    out   = (x_hat * norms).to(bf16)             # (N, D) bf16

For N = B*H*L vectors at D=128, the two fp32 intermediates alone are 8 bytes per
output element, against 64 bytes of packed input per *vector* (0.5 bytes per
element at 4 bits). Step 1 of this project measured this card at 711 GB/s with a
ridge point of 180.7 FLOP/byte; this operation is nowhere near that ridge, so it
is bandwidth-bound and the intermediates are the whole cost.

This kernel does unpack, codebook gather, the 128x128 inverse rotation, and the
norm rescale in registers, and writes bf16 once. Traffic per vector goes from
"read 64 B packed, write and re-read 512 B fp32 twice, write 256 B bf16" to
"read 64 B packed, write 256 B bf16".

Bit extraction is general over `BW`: an index may straddle a byte boundary (it
does at 3 bits), so each one is assembled from two bytes with a masked load for
the final index, whose second byte is past the end of the row.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _tq_dequant_kernel(
    packed_ptr, norms_ptr, cent_ptr, pi_ptr, out_ptr,
    N, L, MAXLEN,
    D: tl.constexpr, NB: tl.constexpr, BW: tl.constexpr,
    BLOCK_N: tl.constexpr, ALLOW_TF32: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    offs_d = tl.arange(0, D)
    # The cache hands us a (B, H, MAXLEN, NB) buffer sliced to the first L
    # positions, which is not contiguous. Indexing through the real layout
    # avoids a full `.contiguous()` copy of the packed cache on every step --
    # the exact extra pass over memory this kernel exists to remove.
    bh = rows // L
    pos = rows % L
    src_row = bh * MAXLEN + pos

    # --- unpack BW-bit indices, LSB-first, possibly straddling a byte -------
    bit_off = offs_d[None, :] * BW
    byte0 = bit_off // 8
    sh = bit_off % 8
    base = packed_ptr + src_row[:, None].to(tl.int64) * NB
    b0 = tl.load(base + byte0, mask=rmask[:, None], other=0).to(tl.int32)
    # The high byte only exists when the index crosses into it; the last index
    # of a row would otherwise read one byte past the row.
    hi_ok = rmask[:, None] & ((byte0 + 1) < NB)
    b1 = tl.load(base + byte0 + 1, mask=hi_ok, other=0).to(tl.int32)
    idx = ((b0 >> sh) | (b1 << (8 - sh))) & ((1 << BW) - 1)

    # --- codebook gather (table is 2^BW entries, so it stays in L1) ---------
    y = tl.load(cent_ptr + idx)                              # (BLOCK_N, D) fp32

    # --- inverse rotation: (BLOCK_N, D) @ (D, D) ---------------------------
    pi = tl.load(pi_ptr + offs_d[:, None] * D + offs_d[None, :])
    x = tl.dot(y, pi, allow_tf32=ALLOW_TF32)

    # --- rescale and store once, in the output dtype -----------------------
    nrm = tl.load(norms_ptr + src_row, mask=rmask, other=0.0).to(tl.float32)
    x = x * nrm[:, None]
    tl.store(out_ptr + rows[:, None].to(tl.int64) * D + offs_d[None, :],
             x.to(out_ptr.dtype.element_ty), mask=rmask[:, None])


def tq_dequant(packed: torch.Tensor, norms: torch.Tensor,
               centroids: torch.Tensor, Pi: torch.Tensor,
               bit_width: int, head_dim: int,
               out: torch.Tensor | None = None,
               block_n: int = 64, allow_tf32: bool = False,
               num_warps: int = 4, num_stages: int = 2) -> torch.Tensor:
    """Fused dequantization. `packed` is (..., NB) uint8, `norms` is (...,).

    Returns (..., head_dim) in `out`'s dtype (bf16 by default). `allow_tf32` is
    False so the rotation matches the fp32 PyTorch reference bit-for-bit-ish;
    turning it on is faster and measurably less accurate, which is quantified in
    bench_kernel.py rather than assumed either way.
    """
    lead = packed.shape[:-1]
    NB = packed.shape[-1]
    N = 1
    for s in lead:
        N *= s
    L = lead[-1]
    # Distance in rows between consecutive (batch, head) planes. For a plain
    # contiguous tensor this is just L; for a slice of a preallocated cache
    # buffer it is the buffer's max_cache_len.
    if packed.dim() >= 3 and packed.stride(-2) > 0:
        maxlen = packed.stride(-3) // packed.stride(-2)
    else:
        maxlen = L
    if out is None:
        out = torch.empty(*lead, head_dim, dtype=torch.bfloat16,
                          device=packed.device)
    assert centroids.dtype == torch.float32 and Pi.dtype == torch.float32
    grid = (triton.cdiv(N, block_n),)
    _tq_dequant_kernel[grid](
        packed, norms, centroids, Pi, out,
        N, L, maxlen, D=head_dim, NB=NB, BW=bit_width,
        BLOCK_N=block_n, ALLOW_TF32=allow_tf32,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out
