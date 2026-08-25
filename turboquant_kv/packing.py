"""Exact bit-packing for TurboQuant codebook indices.

The existing TurboQuant path stores a `bit_width`-bit index in a `torch.uint8`,
so a 4-bit quantizer holds 4 bits in 8 and a 3-bit quantizer holds 3 in 8. The
reported compression ratio (`mem_bits()` / `eff_bits()`) is therefore a
*logical* figure: it counts the bits the scheme needs, not the bytes the process
holds. At 4 bits that is a factor of two between the number in the report and
the number in `nvidia-smi`.

These two functions close that gap. Packing is exact, not word-aligned: an index
is allowed to straddle a byte boundary, so 128 three-bit indices occupy exactly
48 bytes rather than the 64 a nibble-per-byte or 10-per-int32 scheme would need.

Layout: LSB-first within an index, indices in order along the last axis. For
`bit_width == 4` that makes byte *i* equal `idx[2i] | (idx[2i+1] << 4)`, which
is the obvious nibble layout, so the fast path and the general path agree
bit-for-bit. `test_packing.py` asserts that on random data for every supported
width.
"""
from __future__ import annotations

import torch

SUPPORTED = (2, 3, 4, 5, 6, 8)


def packed_bytes(head_dim: int, bit_width: int) -> int:
    """Bytes needed for one `head_dim`-long vector of `bit_width`-bit indices."""
    total_bits = head_dim * bit_width
    if total_bits % 8 != 0:
        raise ValueError(
            f"head_dim*bit_width must be a multiple of 8, got "
            f"{head_dim}*{bit_width}={total_bits}")
    return total_bits // 8


@torch.no_grad()
def pack(idx: torch.Tensor, bit_width: int) -> torch.Tensor:
    """(..., D) uint8 indices -> (..., D*bit_width/8) uint8.

    Values must fit in `bit_width` bits; this is asserted in debug builds only
    because the caller is `bucketize` output, which cannot exceed the codebook.
    """
    if bit_width == 8:
        return idx.contiguous()
    D = idx.shape[-1]
    nbytes = packed_bytes(D, bit_width)

    if bit_width == 4:
        # Fast path: one byte per index pair. Identical layout to the general
        # path below, which test_packing.py verifies.
        flat = idx.reshape(-1, D)
        out = (flat[:, 0::2] | (flat[:, 1::2] << 4))
        return out.reshape(*idx.shape[:-1], nbytes).contiguous()

    lead = idx.shape[:-1]
    # Explode to bits, LSB-first per index.
    sh = torch.arange(bit_width, device=idx.device, dtype=torch.uint8)
    bits = (idx.reshape(-1, D).unsqueeze(-1) >> sh) & 1        # (N, D, bw)
    bits = bits.reshape(-1, D * bit_width, 1).reshape(-1, nbytes, 8)
    # int16 accumulation: 8 terms up to 128 each would overflow uint8 mid-sum.
    w = (1 << torch.arange(8, device=idx.device, dtype=torch.int16))
    out = (bits.to(torch.int16) * w).sum(-1).to(torch.uint8)
    return out.reshape(*lead, nbytes).contiguous()


@torch.no_grad()
def unpack(packed: torch.Tensor, bit_width: int, head_dim: int) -> torch.Tensor:
    """(..., D*bit_width/8) uint8 -> (..., D) uint8 indices."""
    if bit_width == 8:
        return packed
    lead = packed.shape[:-1]
    nbytes = packed.shape[-1]

    if bit_width == 4:
        flat = packed.reshape(-1, nbytes)
        out = torch.empty(flat.shape[0], head_dim, dtype=torch.uint8,
                          device=packed.device)
        out[:, 0::2] = flat & 0x0F
        out[:, 1::2] = flat >> 4
        return out.reshape(*lead, head_dim)

    sh = torch.arange(8, device=packed.device, dtype=torch.uint8)
    bits = (packed.reshape(-1, nbytes).unsqueeze(-1) >> sh) & 1   # (N, nbytes, 8)
    bits = bits.reshape(-1, nbytes * 8)[:, :head_dim * bit_width]
    bits = bits.reshape(-1, head_dim, bit_width)
    w = (1 << torch.arange(bit_width, device=packed.device, dtype=torch.int16))
    out = (bits.to(torch.int16) * w).sum(-1).to(torch.uint8)
    return out.reshape(*lead, head_dim)
