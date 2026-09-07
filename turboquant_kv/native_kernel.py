"""Optional adapter for the already-built native CUDA extension.

Importing this module never invokes a compiler. A caller may provide the same
callable explicitly, as the C27 harness does, or make an existing
``tq_dequant_cuda_ext`` module importable before constructing the cache.
"""
from __future__ import annotations

import importlib


def existing_native_kernel():
    try:
        extension = importlib.import_module("tq_dequant_cuda_ext")
    except (ImportError, OSError):
        return None

    def dequant(packed, norms, centroids, pi, bit_width, head_dim,
                out=None, allow_tf32=False):
        del allow_tf32
        import torch

        lead = packed.shape[:-1]
        rows = 1
        for size in lead:
            rows *= size
        cache_len = lead[-1]
        max_cache_len = (
            packed.stride(-3) // packed.stride(-2)
            if packed.dim() >= 3 else cache_len
        )
        if out is None:
            out = torch.empty(*lead, head_dim, dtype=torch.bfloat16,
                              device=packed.device)
        return extension.tq_dequant_cuda(
            packed, norms, centroids, pi, out, bit_width, head_dim,
            rows, cache_len, max_cache_len,
        )

    return dequant
