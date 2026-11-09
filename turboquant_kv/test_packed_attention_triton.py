"""GPU differential tests: Triton packed decode vs the tiled FP32 reference.

Runs on CUDA. With TRITON_INTERPRET=1 it also runs on CPU through Triton's
interpreter, which checks kernel logic only: it is not CUDA execution and its
results are never GPU evidence. Without either, every test here skips, and a
skip means this gate is blocked, not passed.

The kernel and reference share inputs, so the only error is arithmetic: the
FP32 (non-TF32) tolerance is frozen at ATOL/RTOL below.
"""
import os

import pytest
import torch

import packed_attention as pa
from cache import _get_quantizer

INTERP = os.environ.get("TRITON_INTERPRET") == "1"
DEV = "cuda" if torch.cuda.is_available() else ("cpu" if INTERP else None)
pytestmark = pytest.mark.skipif(DEV is None, reason="BLOCKED: needs CUDA (or TRITON_INTERPRET=1 for logic only)")

if DEV is not None:
    import packed_attention_triton as pt

ATOL = 2e-3
RTOL = 2e-3


def case(seed, B, Hq, Hkv, D, lens, S, bw, q_scale=1.0, distinct=True):
    g = torch.Generator().manual_seed(seed)
    Lmax = max(max(lens), 1)
    k = torch.randn(B, Hkv, Lmax, D, generator=g)
    v = torch.randn(B, Hkv, Lmax, D, generator=g)
    q = torch.randn(B, Hq, 1, D, generator=g) * q_scale
    kq = _get_quantizer(bw, D, "cpu", 42)
    vq = _get_quantizer(bw, D, "cpu", 43 if distinct else 42)
    kv = pa.pages_from_dense(k, v, lens, kq, vq, S, generator=g)
    return q, kv


def to(kv, dev):
    return pa.PackedKV(kv.k_pages.to(dev), kv.k_norms.to(dev), kv.v_pages.to(dev),
                       kv.v_norms.to(dev), kv.page_table.to(dev), kv.seq_lens.to(dev),
                       pa.Quantizer(kv.k_quant.Pi.to(dev), kv.k_quant.centroids.to(dev), kv.k_quant.bit_width),
                       pa.Quantizer(kv.v_quant.Pi.to(dev), kv.v_quant.centroids.to(dev), kv.v_quant.bit_width),
                       kv.head_dim)


# Frozen differential matrix: (name, B, Hq, Hkv, D, lens, S, bw, q_scale)
CASES = [
    ("qwen7b_gqa_short", 3, 28, 4, 128, (1, 17, 64), 16, 8, 1.0),
    ("qwen7b_gqa_mixed", 4, 28, 4, 128, (0, 300, 31, 129), 32, 8, 1.0),
    ("mha_like", 2, 8, 8, 64, (40, 7), 16, 8, 1.0),
    ("gqa8_2", 3, 8, 2, 64, (16, 33, 1), 16, 8, 1.0),
    ("huge_scores", 2, 28, 4, 128, (90, 47), 16, 8, 50.0),
    ("four_bit", 2, 28, 4, 128, (77, 160), 16, 4, 1.0),
    ("three_bit", 2, 8, 2, 128, (50, 19), 16, 3, 1.0),
]
if not INTERP:
    CASES += [("long_4k", 2, 28, 4, 128, (4096, 2500), 32, 8, 1.0),
              ("batch32", 32, 28, 4, 128, tuple(range(5, 5 + 32 * 61, 61)), 16, 8, 1.0)]

# Python's process-randomized hash(name) did not define reproducible inputs.
CASE_SEEDS = {"qwen7b_gqa_short": 101, "qwen7b_gqa_mixed": 102,
              "mha_like": 103, "gqa8_2": 104, "huge_scores": 105,
              "four_bit": 106, "three_bit": 107, "long_4k": 108,
              "batch32": 109}


@pytest.mark.parametrize("splits", [1, 3])
@pytest.mark.parametrize("name,B,Hq,Hkv,D,lens,S,bw,qs", CASES, ids=[c[0] for c in CASES])
def test_kernel_matches_reference(name, B, Hq, Hkv, D, lens, S, bw, qs, splits):
    q, kv = case(CASE_SEEDS[name], B, Hq, Hkv, D, lens, S, bw, qs)
    scale = pa.default_scale(D)
    ref = pa.tiled_compressed_attention(q, kv, scale)
    got = pt.packed_decode_attention(q.to(DEV), to(kv, DEV), scale, num_splits=splits)
    got = got.float().cpu()
    assert torch.isfinite(got).all()
    assert torch.allclose(got, ref, atol=ATOL, rtol=RTOL), (got - ref).abs().max()
    for b, L in enumerate(lens):
        if L == 0:
            assert (got[b] == 0).all()


def test_bf16_query_output_dtype():
    q, kv = case(3, 2, 28, 4, 128, (60, 9), 16, 8)
    scale = pa.default_scale(128)
    got = pt.packed_decode_attention(q.to(DEV, torch.bfloat16), to(kv, DEV), scale)
    assert got.dtype == torch.bfloat16
    ref = pa.tiled_compressed_attention(q.bfloat16().float(), kv, scale)
    assert (got.float().cpu() - ref).abs().max() < 2e-2     # bf16 output rounding


def test_strided_pool_layer_view():
    """Pages handed over as one layer of a (layers, P, ...) pool tensor."""
    q, kv = case(4, 2, 8, 2, 64, (45, 20), 16, 8)
    stack = lambda t: torch.stack([torch.zeros_like(t), t, torch.zeros_like(t)])[1]
    kv2 = pa.PackedKV(stack(kv.k_pages), stack(kv.k_norms), stack(kv.v_pages),
                      stack(kv.v_norms), kv.page_table, kv.seq_lens, kv.k_quant,
                      kv.v_quant, kv.head_dim)
    assert kv2.k_pages.storage_offset() > 0
    scale = pa.default_scale(64)
    got = pt.packed_decode_attention(q.to(DEV), to(kv2, DEV), scale).float().cpu()
    assert torch.allclose(got, pa.tiled_compressed_attention(q, kv, scale), atol=ATOL, rtol=RTOL)


def test_rejects_before_launch():
    q, kv = case(5, 2, 8, 2, 64, (20, 3), 16, 8)
    with pytest.raises(pa.UnsupportedLayout):
        pt.packed_decode_attention(q.repeat(1, 1, 2, 1).to(DEV), to(kv, DEV), 1.0)   # Q=2
    q2, kv2 = case(6, 2, 8, 2, 64, (20, 3), 8, 8)                                     # S=8
    with pytest.raises(pa.UnsupportedLayout):
        pt.packed_decode_attention(q2.to(DEV), to(kv2, DEV), 1.0)


def test_split_workspace_only_stores_real_query_heads(monkeypatch):
    """Tensor-core padding is local to a program, not global split scratch."""
    q, kv = case(11, 2, 28, 4, 128, (65, 9), 16, 8)
    qd, kvd = q.to(DEV), to(kv, DEV)
    allocated = []
    original_empty = torch.empty

    def record_empty(*shape, **kwargs):
        allocated.append(shape)
        return original_empty(*shape, **kwargs)

    monkeypatch.setattr(pt.torch, "empty", record_empty)
    got = pt.packed_decode_attention(qd, kvd, 0.1, num_splits=3)
    assert (2 * 28 * 3, 128) in allocated, allocated
    ref = pa.tiled_compressed_attention(q, kv, 0.1)
    assert torch.allclose(got.float().cpu(), ref, atol=ATOL, rtol=RTOL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BLOCKED: allocator trace needs CUDA")
def test_no_full_history_materialization():
    """Peak allocation during the call stays far below one (B, Hkv, L, D) bf16
    history, the tensor the reconstruct path allocates per layer per step."""
    B, Hq, Hkv, D, L = 8, 28, 4, 128, 4096
    q, kv = case(7, B, Hq, Hkv, D, (L,) * B, 32, 8)
    kvd, qd = to(kv, "cuda"), q.to("cuda", torch.bfloat16)
    pt.packed_decode_attention(qd, kvd, 0.1)                      # compile/warm
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    pt.packed_decode_attention(qd, kvd, 0.1, num_splits=4)
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - base
    history = B * Hkv * L * D * 2
    assert extra < history / 50, (extra, history)


@pytest.mark.parametrize("splits,ppi", [(1, 1), (2, 1), (1, 4), (3, 2)])
def test_dense_kernel_matches_sdpa(splits, ppi):
    """The matched unquantized arm must be exact attention, not a straw man."""
    import packed_backend as pb
    g = torch.Generator().manual_seed(8)
    B, Hq, Hkv, D, S, lens = 3, 28, 4, 128, 16, (33, 0, 70)
    P = sum((L + S - 1) // S for L in lens) + 2
    kp = torch.randn(P, Hkv, S, D, generator=g)
    vp = torch.randn(P, Hkv, S, D, generator=g)
    order = torch.randperm(P, generator=g).tolist()
    table = torch.full((B, 5), -1, dtype=torch.int32)
    for b, L in enumerate(lens):
        for t in range((L + S - 1) // S):
            table[b, t] = order.pop()
    kv = pa.DenseKV(kp, vp, table, torch.tensor(lens, dtype=torch.int32), D)
    q = torch.randn(B, Hq, 1, D, generator=g)
    ref = pb._dense_sdpa(q, kv, 0.09)
    kvd = pa.DenseKV(kp.to(DEV), vp.to(DEV), table.to(DEV), kv.seq_lens.to(DEV), D)
    got = pt.dense_decode_attention(q.to(DEV), kvd, 0.09, num_splits=splits,
                                    pages_per_iter=ppi).float().cpu()
    assert torch.allclose(got, ref, atol=ATOL, rtol=RTOL), (got - ref).abs().max()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BLOCKED: allocator trace needs CUDA")
def test_extra_memory_does_not_grow_with_history():
    """The invariant behind 'no full-history materialization': the kernel's
    temporary allocation (rotated query, split workspace, merge) is set by
    batch, heads and splits, not by sequence length. Added after the fixed
    1/50-of-history threshold above failed on an A100 at L=4096 (2.24 MB vs
    33.5 MB history); that failure is kept, this checks the actual property."""
    B, Hq, Hkv, D = 8, 28, 4, 128
    extras = {}
    for L in (1024, 16384):
        q, kv = case(9, B, Hq, Hkv, D, (L,) * B, 32, 8)
        kvd, qd = to(kv, "cuda"), q.to("cuda", torch.bfloat16)
        pt.packed_decode_attention(qd, kvd, 0.1, num_splits=4)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        pt.packed_decode_attention(qd, kvd, 0.1, num_splits=4)
        torch.cuda.synchronize()
        extras[L] = torch.cuda.max_memory_allocated() - base
        del q, kv, kvd, qd
    history_16k = B * Hkv * 16384 * D * 2
    assert extras[16384] <= 1.05 * extras[1024], extras
    assert extras[16384] < history_16k / 50, (extras, history_16k)


PPI_CASES = [c for c in CASES if c[0] in ("qwen7b_gqa_mixed", "four_bit", "three_bit", "gqa8_2")]


@pytest.mark.parametrize("ppi,splits", [(2, 1), (4, 1), (2, 3), (8, 2)])
@pytest.mark.parametrize("name,B,Hq,Hkv,D,lens,S,bw,qs", PPI_CASES, ids=[c[0] for c in PPI_CASES])
def test_pages_per_iter_matches_reference(name, B, Hq, Hkv, D, lens, S, bw, qs, ppi, splits):
    """Several pages per loop iteration, including groups that run past a
    sequence's last page and splits whose size is rounded up to whole groups.

    Wasteful-but-correct shapes (groups past the last page, empty splits) still
    run here: the tuner avoids them, but the kernel must handle them. Only
    schedules exceeding sm80 hardware resources are skipped; on 2026-11-04 an
    A100 refused ppi=8 with 32-token pages (286,720 B shared memory), which
    tsc.schedule.attn_legal rejects before launch."""
    from tsc import schedule as sch
    prob = sch.AttnProblem(B, max(lens), Hq, Hkv, D, S, bw)
    ok, why = sch.attn_legal(sch.AttnSchedule(splits, ppi, 4, 2), prob, sch.TARGETS["sm80"])
    if not ok and "shared memory" in why:
        pytest.skip(f"exceeds sm80 resources, rejected before launch: {why}")
    q, kv = case(CASE_SEEDS[name], B, Hq, Hkv, D, lens, S, bw, qs)
    scale = pa.default_scale(D)
    ref = pa.tiled_compressed_attention(q, kv, scale)
    got = pt.packed_decode_attention(q.to(DEV), to(kv, DEV), scale, num_splits=splits,
                                     pages_per_iter=ppi).float().cpu()
    assert torch.allclose(got, ref, atol=ATOL, rtol=RTOL), (got - ref).abs().max()


def test_pages_per_iter_rejects_non_power_of_two():
    q, kv = case(5, 2, 8, 2, 64, (20, 3), 16, 8)
    with pytest.raises(pa.UnsupportedLayout):
        pt.packed_decode_attention(q.to(DEV), to(kv, DEV), 1.0, pages_per_iter=3)
