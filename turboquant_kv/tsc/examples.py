"""Canonical DSL source for the TurboQuant dequantization pipeline.

`example_source(bit_width, head_dim)` is the programmatic form used by the
GPU evidence run for arbitrary bit widths. The checked-in fixtures under
`examples/` are the golden-test inputs; `test_tsc_parser.py` asserts the
generator and the fixtures agree, so they cannot drift apart.
"""
from __future__ import annotations

from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"


def example_source(bit_width: int, head_dim: int = 128) -> str:
    nb = head_dim * bit_width // 8
    levels = 2 ** bit_width
    return f"""\
# TurboQuant KV dequantization: unpack -> gather -> rotate -> rescale -> cast -> store
pipeline tq_dequant(bit_width={bit_width}, head_dim={head_dim}) {{
  input  packed:    u8[N, {nb}]
  input  norms:     f16[N]
  input  centroids: f32[{levels}]
  input  pi:        f32[head_dim, head_dim]
  output out:       bf16[N, head_dim]

  codes = load(packed)
  cb    = load(centroids)
  rotm  = load(pi)
  nrm   = load(norms)

  idx   = unpack(codes, bits=bit_width)
  y     = gather(cb, idx)
  x     = rot(y, rotm)
  nf    = cast(nrm, f32)
  z     = rescale(x, nf)
  o     = cast(z, bf16)
  store(out, o)
}}
"""


def load_example(name: str) -> str:
    return (EXAMPLES_DIR / name).read_text()
