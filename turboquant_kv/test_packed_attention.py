"""Compressed-domain attention must equal reconstruct-then-attend.

Two error budgets are kept apart. Kernel arithmetic error is the difference
between the compressed-domain operators and attention over the *same* quantized
K/V reconstructed in fp32; it is frozen at ATOL/RTOL below. Quantization error,
the difference from unquantized attention, is a property of the storage format
and is only sanity-bounded here.

Each seeded mutant breaks exactly one algebraic piece of the real
implementation (rotation direction, head mapping, scale, K/V basis, online
rescale, split merge). A test that passes on a mutant would be measuring
nothing, so every mutant must miss the tolerance by a wide margin.
"""
import math

import pytest
import torch

import packed_attention as pa
from cache import TurboQuantMSE, _get_quantizer

ATOL = 2e-4
RTOL = 2e-4
MUTANT_MIN_ERR = 100 * ATOL


def close(a, b):
    return torch.allclose(a, b, atol=ATOL, rtol=RTOL)


def max_err(a, b):
    return (a - b).abs().max().item()


def dense_attention(q, k, v, lens, scale):
    """Unquantized reference over dense (B, Hkv, Lmax, D) K/V with GQA."""
    B, Hq, Q, D = q.shape
    n_rep = Hq // k.shape[1]
    out = torch.zeros(B, Hq, Q, D)
    for b in range(B):
        L = int(lens[b])
        if L == 0:
            continue
        kk = k[b, :, :L].repeat_interleave(n_rep, 0)
        vv = v[b, :, :L].repeat_interleave(n_rep, 0)
        s = q[b] @ kk.transpose(-1, -2) * scale
        lim = L - Q + torch.arange(Q)
        s = s.masked_fill(torch.arange(L)[None, :] > lim[:, None], float("-inf"))
        out[b] = torch.softmax(s, -1).nan_to_num(0) @ vv
    return out


def make_case(seed=0, B=3, Hq=8, Hkv=2, D=64, lens=(37, 1, 64), Q=1, S=16, bw=8,
              q_scale=1.0, distinct_vq=True):
    g = torch.Generator().manual_seed(seed)
    Lmax = max(max(lens), 1)
    k = torch.randn(B, Hkv, Lmax, D, generator=g)
    v = torch.randn(B, Hkv, Lmax, D, generator=g)
    q = torch.randn(B, Hq, Q, D, generator=g) * q_scale
    k_tq = _get_quantizer(bw, D, "cpu", seed=42)
    v_tq = _get_quantizer(bw, D, "cpu", seed=43 if distinct_vq else 42)
    kv = pa.pages_from_dense(k, v, lens, k_tq, v_tq, S, generator=g)
    return q, k, v, kv


OPS = {
    "explicit": pa.explicit_compressed_attention,
    "tiled": pa.tiled_compressed_attention,
    "tiled_split": lambda q, kv, s: pa.tiled_compressed_attention(q, kv, s, split_pages=2),
}


@pytest.mark.parametrize("op", sorted(OPS))
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("bw", [8, 4, 3])
def test_matches_reconstruct(op, seed, bw):
    q, _, _, kv = make_case(seed=seed, bw=bw)
    scale = pa.default_scale(q.shape[-1])
    ref = pa.reconstruct_attention(q, kv, scale)
    got = OPS[op](q, kv, scale)
    assert close(got, ref), max_err(got, ref)


@pytest.mark.parametrize("op", sorted(OPS))
def test_prefill_chunk_causal(op):
    # Q > 1: the new tokens are already in the cache and see only their past.
    q, _, _, kv = make_case(seed=5, lens=(40, 17, 5), Q=5)
    scale = pa.default_scale(q.shape[-1])
    assert close(OPS[op](q, kv, scale), pa.reconstruct_attention(q, kv, scale))


@pytest.mark.parametrize("op", sorted(OPS))
def test_empty_and_partial_pages(op):
    # length 0 (no pages), exactly one page, a one-token tail.
    q, _, _, kv = make_case(seed=7, lens=(0, 16, 17), S=16)
    scale = pa.default_scale(q.shape[-1])
    got = OPS[op](q, kv, scale)
    assert torch.isfinite(got).all()
    assert (got[0] == 0).all()
    assert close(got, pa.reconstruct_attention(q, kv, scale))


@pytest.mark.parametrize("op", sorted(OPS))
def test_all_masked_rows_are_zero_not_nan(op):
    # Q=3 over a length-2 cache: the first query row has no visible key.
    q, _, _, kv = make_case(seed=8, lens=(2, 9, 30), Q=3)
    scale = pa.default_scale(q.shape[-1])
    got = OPS[op](q, kv, scale)
    assert torch.isfinite(got).all()
    assert (got[0, :, 0] == 0).all()
    assert close(got, pa.reconstruct_attention(q, kv, scale))


@pytest.mark.parametrize("op", sorted(OPS))
def test_huge_score_range(op):
    # Scores in the thousands: a softmax without max subtraction overflows.
    q, _, _, kv = make_case(seed=9, q_scale=3e3, lens=(50, 33, 64))
    scale = pa.default_scale(q.shape[-1])
    ref = pa.reconstruct_attention(q, kv, scale)
    got = OPS[op](q, kv, scale)
    assert torch.isfinite(got).all()
    assert close(got, ref), max_err(got, ref)


@pytest.mark.parametrize("op", sorted(OPS))
def test_zero_vectors(op):
    g = torch.Generator().manual_seed(10)
    B, H, L, D = 2, 2, 20, 64
    k, v = torch.randn(B, H, L, D, generator=g), torch.randn(B, H, L, D, generator=g)
    k[0, :, 3] = 0
    v[1, :, 5:9] = 0
    tq = _get_quantizer(8, D, "cpu", 42)
    kv = pa.pages_from_dense(k, v, (L, L), tq, _get_quantizer(8, D, "cpu", 43), 8, generator=g)
    q = torch.randn(B, 4, 1, D, generator=g)
    scale = pa.default_scale(D)
    got = OPS[op](q, kv, scale)
    assert torch.isfinite(got).all()
    assert close(got, pa.reconstruct_attention(q, kv, scale))


def test_gqa_head_groups_are_distinct():
    # Every kv head carries different data; each query head must read its own.
    q, _, _, kv = make_case(seed=11, Hq=12, Hkv=3)
    scale = pa.default_scale(q.shape[-1])
    assert close(pa.tiled_compressed_attention(q, kv, scale),
                 pa.reconstruct_attention(q, kv, scale))


def test_quantization_error_is_a_separate_small_budget():
    q, k, v, kv = make_case(seed=12, bw=8)
    scale = pa.default_scale(q.shape[-1])
    exact = dense_attention(q, k, v, kv.seq_lens, scale)
    quant = pa.reconstruct_attention(q, kv, scale)
    kernel = pa.tiled_compressed_attention(q, kv, scale)
    q_err, k_err = max_err(quant, exact), max_err(kernel, quant)
    assert q_err < 0.1                  # 8-bit storage noise, sanity only
    assert k_err < ATOL < q_err         # arithmetic error sits far inside it


def test_rotation_direction_with_nonsymmetric_matrix():
    """A hand-built orthogonal Pi with Pi != Pi^T, small enough to inspect.

    If the code rotated queries by Pi instead of Pi^T, or values back by Pi^T
    instead of Pi, this matrix makes the answer wrong rather than coincidentally
    right, which a symmetric or identity rotation would not.
    """
    D = 8
    c, s = math.cos(0.7), math.sin(0.7)
    R = torch.eye(D)
    R[0, 0], R[0, 1], R[1, 0], R[1, 1] = c, -s, s, c
    perm = torch.eye(D)[[2, 0, 1, 5, 3, 4, 7, 6]]
    Pi = (perm @ R).contiguous()
    assert torch.allclose(Pi @ Pi.T, torch.eye(D), atol=1e-6)
    assert not torch.allclose(Pi, Pi.T)
    k_tq, v_tq = TurboQuantMSE(8, D), TurboQuantMSE(8, D, rotation_seed=7)
    k_tq.Pi = Pi
    g = torch.Generator().manual_seed(13)
    k, v = torch.randn(1, 1, 6, D, generator=g), torch.randn(1, 1, 6, D, generator=g)
    kv = pa.pages_from_dense(k, v, (6,), k_tq, v_tq, 4, generator=g)
    q = torch.randn(1, 2, 1, D, generator=g)
    scale = pa.default_scale(D)
    ref = pa.reconstruct_attention(q, kv, scale)
    assert close(pa.tiled_compressed_attention(q, kv, scale), ref)
    wrong = pa.reconstruct_attention(q, pa.PackedKV(**{
        **kv.__dict__, "k_quant": pa.Quantizer(Pi.T.contiguous(), k_tq.centroids, 8)}), scale)
    assert max_err(wrong, ref) > MUTANT_MIN_ERR


def test_contract_rejections():
    q, _, _, kv = make_case(seed=14)
    with pytest.raises(pa.UnsupportedLayout):
        pa.tiled_compressed_attention(q[..., :32], kv, 1.0)
    with pytest.raises(pa.UnsupportedLayout):
        pa.tiled_compressed_attention(q[:, :7], kv, 1.0)          # Hq % Hkv
    bad = pa.PackedKV(**{**kv.__dict__, "page_table": kv.page_table.long()})
    with pytest.raises(pa.UnsupportedLayout):
        pa.tiled_compressed_attention(q, bad, 1.0)
    bad = pa.PackedKV(**{**kv.__dict__, "page_table": kv.page_table.clone()})
    bad.page_table[0, 0] = kv.k_pages.shape[0]
    with pytest.raises(pa.UnsupportedLayout):
        pa.tiled_compressed_attention(q, bad, 1.0)
    bad = pa.PackedKV(**{**kv.__dict__, "seq_lens": kv.seq_lens + 1000})
    with pytest.raises(pa.UnsupportedLayout):
        pa.tiled_compressed_attention(q, bad, 1.0)


# -- seeded mutants ---------------------------------------------------------

def _mutant_err(monkeypatch, attr, fn, op="tiled", **case):
    q, _, _, kv = make_case(seed=21, **case)
    scale = pa.default_scale(q.shape[-1])
    ref = pa.reconstruct_attention(q, kv, scale)
    assert close(OPS[op](q, kv, scale), ref)
    monkeypatch.setattr(pa, attr, fn)
    return max_err(OPS[op](q, kv, scale), ref)


@pytest.mark.parametrize("op", ["explicit", "tiled"])
def test_mutant_query_rotation_transposed(monkeypatch, op):
    assert _mutant_err(monkeypatch, "rotate_query", lambda q, Pi: q @ Pi, op) > MUTANT_MIN_ERR


@pytest.mark.parametrize("op", ["explicit", "tiled"])
def test_mutant_output_rotation_transposed(monkeypatch, op):
    assert _mutant_err(monkeypatch, "rotate_output", lambda a, Pi: a @ Pi.T, op) > MUTANT_MIN_ERR


@pytest.mark.parametrize("op", ["explicit", "tiled"])
def test_mutant_gqa_head_interleaved(monkeypatch, op):
    # h % Hkv is the classic wrong GQA mapping; Hq=8, Hkv=2.
    assert _mutant_err(monkeypatch, "kv_head_for", lambda h, n: h % 2, op) > MUTANT_MIN_ERR


def test_mutant_missing_scale():
    q, _, _, kv = make_case(seed=22)
    ref = pa.reconstruct_attention(q, kv, pa.default_scale(q.shape[-1]))
    assert max_err(pa.tiled_compressed_attention(q, kv, 1.0), ref) > MUTANT_MIN_ERR


def test_mutant_value_read_in_key_basis():
    q, _, _, kv = make_case(seed=23)
    scale = pa.default_scale(q.shape[-1])
    ref = pa.reconstruct_attention(q, kv, scale)
    swapped = pa.PackedKV(**{**kv.__dict__, "v_quant": kv.k_quant})
    assert max_err(pa.tiled_compressed_attention(q, swapped, scale), ref) > MUTANT_MIN_ERR


def test_mutant_online_softmax_without_rescale(monkeypatch):
    def no_alpha(m, l, a, s, nv, cv):
        m_new = torch.maximum(m, s.amax(-1))
        p = pa._safe_exp(s, m_new[..., None])
        return m_new, l + p.sum(-1), a + (p * nv[:, :, None, :]) @ cv
    # Several pages with a large score range so the running max moves.
    assert _mutant_err(monkeypatch, "_online_update", no_alpha,
                       lens=(64, 48, 64), q_scale=4.0) > MUTANT_MIN_ERR


def test_mutant_split_merge_without_rescale(monkeypatch):
    def naive_merge(left, right):
        m1, l1, a1 = left
        m2, l2, a2 = right
        return torch.maximum(m1, m2), l1 + l2, a1 + a2
    assert _mutant_err(monkeypatch, "merge_partials", naive_merge, op="tiled_split",
                       lens=(64, 48, 64), q_scale=4.0) > MUTANT_MIN_ERR


def test_tiled_never_allocates_history(monkeypatch):
    """No tensor shaped like a (.., L, D) history appears in the tiled path."""
    q, _, _, kv = make_case(seed=24, lens=(64, 64, 64), S=16)
    seen = []
    real = pa._centroids

    def spy(pages, quant, d):
        out = real(pages, quant, d)
        seen.append(tuple(out.shape))
        return out
    monkeypatch.setattr(pa, "_centroids", spy)
    pa.tiled_compressed_attention(q, kv, 1.0)
    # Each decode tile is one page per sequence: (B, Hkv, S, D), never L=64.
    assert seen and all(s[-2] == 16 for s in seen)
