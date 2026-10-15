"""Opt-in Transformers generation adapter for the measured packed KV cache.

CPU and unsupported hardware retain the model's normal cache path. A caller
must explicitly request packing; no model loading or benchmark runs at import.
"""
from __future__ import annotations


def generate(model, inputs, *, bit_width: int, max_cache_len: int,
             num_outlier_channels: int = 0, outlier_bits: int = 0,
             packed_device: str | None = None, **generation_kwargs):
    """Generate with TQPackedCache on CUDA, otherwise use the native cache.

    Dispatch inside ``TQPackedCache`` still requires a matching measured GPU,
    shape and error profile; the cache's reference kernel is the fallback.
    """
    device = packed_device or str(getattr(inputs, "device", "cpu"))
    if not device.startswith("cuda"):
        return model.generate(inputs, **generation_kwargs)
    if isinstance(bit_width, bool) or not isinstance(bit_width, int) or not 1 <= bit_width <= 8:
        raise ValueError("bit_width must be an integer from 1 through 8")
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("model.config is required for a packed cache")
    from cache import TQPackedCache
    cache = TQPackedCache(
        config, bit_width=bit_width, max_cache_len=max_cache_len,
        device=device, num_outlier_channels=num_outlier_channels,
        outlier_bits=outlier_bits,
    )
    return model.generate(inputs, past_key_values=cache, **generation_kwargs)
