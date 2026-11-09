"""Model integration and request lifecycle, on a tiny seeded Qwen2 (CPU).

The quantized reference is the existing `TQPackedCache` reconstruct path under
SDPA, run one request at a time with a full (unchunked) prefill. With K and V
quantized by the same seed-42 quantizer, both paths store identical bytes, so
any divergence is the new attention path or the paged bookkeeping. These run in
fp32 on CPU: they pin the integration contract, they are not GPU evidence.
"""
import threading
import time

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2 import modeling_qwen2

import packed_backend as pb
from cache import TQPackedCache
from engine import Engine, EngineConfig, Refused, Request
from page_pool import PagePool, StalePage

BW = 8
LOGIT_ATOL = 1e-3


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=96, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=512, initializer_range=0.25, architectures=["Qwen2ForCausalLM"],
                      attn_implementation="sdpa")
    m = Qwen2ForCausalLM(cfg).eval()
    pb.register()
    return m


def prompts(n, seed=1, lo=3, hi=60):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 96, (int(torch.randint(lo, hi, (1,), generator=g)),),
                          generator=g).tolist() for _ in range(n)]


@torch.no_grad()
def reference(model, prompt, max_new, eos=()):
    """Isolated quantized reference: existing packed cache, SDPA, greedy."""
    model.config._attn_implementation = "sdpa"
    cache = TQPackedCache(model.config, bit_width=BW, max_cache_len=len(prompt) + max_new,
                          device="cpu")
    ids = torch.tensor([prompt])
    toks, logits = [], []
    for _ in range(max_new):
        out = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        lg = out.logits[0, -1].float()
        t = int(lg.argmax())
        toks.append(t)
        logits.append(lg)
        if t in eos:
            break
        ids = torch.tensor([[t]])
    return toks, logits


def make_engine(model, pages=64, page_size=8, cap=None, fence=None, **cfg):
    model.config._attn_implementation = pb.ATTN_NAME
    pool = pb.make_pool(model.config, bit_width=BW, num_pages=pages, page_size=page_size,
                        device="cpu", max_pages_per_request=cap, fence_factory=fence)
    kq, vq = pb.quantizers_for(model.config, BW, "cpu")
    return Engine(model, pool, kq, vq, pb.head_dim_of(model.config),
                  config=EngineConfig(**cfg))


def run(engine, reqs):
    for r in reqs:
        engine.submit(r)
    engine.run_until_idle()
    return reqs


# -- generation / logits contract ---------------------------------------------

@pytest.mark.parametrize("backend", ["tiled_reference", "reconstruct_sdpa"])
def test_single_request_logits_match_reference(model, backend):
    prompt = prompts(1, seed=3, lo=20, hi=21)[0]
    ref_toks, ref_logits = reference(model, prompt, 12)
    eng = make_engine(model, decode_backend=backend, prefill_chunk=7)
    got = []
    eng.on_logits = lambda reqs, lg: got.append(lg[0].clone())
    pb.BACKEND_COUNTS.clear()
    (r,) = run(eng, [Request("a", prompt, 12)])
    assert r.generated == ref_toks
    # Prefill chunks before the last produce logits the reference never sees.
    decode_logits = got[-len(ref_logits):]
    for a, b in zip(decode_logits, ref_logits):
        assert (a - b).abs().max() < LOGIT_ATOL
    assert pb.BACKEND_COUNTS[backend] > 0
    assert pb.BACKEND_COUNTS["prefill_reconstruct_sdpa"] > 0
    assert pb.BACKEND_COUNTS["native_sdpa"] == 0


def test_concurrent_mixed_lengths_equal_isolated(model):
    ps = prompts(6, seed=4)
    news = [5, 17, 1, 9, 13, 3]
    refs = [reference(model, p, n)[0] for p, n in zip(ps, news)]
    eng = make_engine(model, max_batch=4, prefill_chunk=11)
    reqs = run(eng, [Request(f"r{i}", p, n) for i, (p, n) in enumerate(zip(ps, news))])
    for r, ref in zip(reqs, refs):
        assert r.status == "length" and r.generated == ref
    assert eng.pool.stats()["owned"] == 0


def test_eos_stops_and_frees(model):
    prompt = prompts(1, seed=5)[0]
    ref, _ = reference(model, prompt, 10)
    eos = ref[3]
    first = ref.index(eos)
    eng = make_engine(model)
    eng.eos = {eos}
    (r,) = run(eng, [Request("e", prompt, 10)])
    assert r.status == "eos" and r.generated == ref[:first + 1]
    assert eng.pool.stats()["owned"] == 0


def test_direct_path_never_builds_history(model, monkeypatch):
    """Decode must not call the dequantizer that builds (1, Hkv, L, D)."""
    calls = []
    real = pb.dequantize_history
    monkeypatch.setattr(pb, "dequantize_history",
                        lambda kv, *a, **k: calls.append(int(kv.seq_lens.numel())) or real(kv, *a, **k))
    eng = make_engine(model, decode_backend="tiled_reference", prefill_chunk=64)
    phase = []
    eng.on_logits = lambda reqs, lg: phase.append(len(calls))
    run(eng, [Request("h", prompts(1, seed=6, lo=30, hi=31)[0], 8)])
    # All dequantize calls happened during the single prefill forward.
    assert calls and phase[0] == len(calls) and phase[-1] == phase[0]


def test_rope_order_mutant_is_detected(model, monkeypatch):
    """Quantizing K before RoPE must change logits; this proves the contract
    test is sensitive to RoPE/quantization ordering."""
    prompt = prompts(1, seed=7, lo=25, hi=26)[0]
    _, ref_logits = reference(model, prompt, 4)
    real = modeling_qwen2.apply_rotary_pos_emb
    monkeypatch.setattr(modeling_qwen2, "apply_rotary_pos_emb",
                        lambda q, k, cos, sin, **kw: (real(q, k, cos, sin)[0], k))
    eng = make_engine(model)
    got = []
    eng.on_logits = lambda reqs, lg: got.append(lg[0].clone())
    run(eng, [Request("m", prompt, 4)])
    assert max((a - b).abs().max().item() for a, b in zip(got[-4:], ref_logits)) > 100 * LOGIT_ATOL


def test_unsupported_models_rejected(model):
    cfg = model.config.to_dict()
    with pytest.raises(pb.UnsupportedModel):
        pb.check_model(Qwen2Config(**{**cfg, "architectures": ["LlamaForCausalLM"]}), BW)
    with pytest.raises(pb.UnsupportedModel):
        pb.check_model(Qwen2Config(**{**cfg, "use_sliding_window": True,
                                      "sliding_window": 16, "max_window_layers": 0}), BW)
    with pytest.raises(ValueError):                 # head_dim 15 * 3 bits = 45 bits
        pb.check_model(Qwen2Config(**{**cfg, "hidden_size": 60, "head_dim": 15}), 3)
    pb.check_model(model.config, BW)


# -- lifecycle ---------------------------------------------------------------

def test_admission_refusals(model):
    eng = make_engine(model, pages=8, page_size=8, cap=4, max_queue=1,
                      max_prompt_tokens=40, max_new_tokens_cap=16)
    with pytest.raises(Refused, match="prompt_too_long"):
        eng.submit(Request("a", [1] * 41, 4))
    with pytest.raises(Refused, match="exceeds_request_cap"):
        eng.submit(Request("b", [1] * 30, 16))      # 46 tokens = 6 pages > 4
    with pytest.raises(Refused, match="bad_max_new_tokens"):
        eng.submit(Request("c", [1] * 3, 0))
    eng.submit(Request("d", [1] * 3, 2))
    with pytest.raises(Refused, match="queue_full"):
        eng.submit(Request("e", [1] * 3, 2))
    with pytest.raises(Refused, match="duplicate_request_id"):
        eng.submit(Request("d", [1] * 3, 2))


def test_pool_exhaustion_queues_then_admits(model):
    # 6 pages of 8: each request needs 4 pages worst case, so only one fits.
    eng = make_engine(model, pages=6, page_size=8, max_batch=4)
    ps = prompts(3, seed=8, lo=20, hi=21)
    reqs = [Request(f"q{i}", p, 8) for i, p in enumerate(ps)]
    for r in reqs:
        eng.submit(r)
    eng.step()
    assert [r.state for r in reqs] == ["decode", "queued", "queued"]
    eng.run_until_idle()
    assert all(r.status == "length" for r in reqs)
    for r, p in zip(reqs, ps):
        assert r.generated == reference(model, p, 8)[0]


def test_cancel_mid_decode_reclaims(model):
    eng = make_engine(model, max_batch=2)
    a, b = Request("a", prompts(1, 9)[0], 30), Request("b", prompts(1, 10)[0], 6)
    eng.submit(a), eng.submit(b)
    while len(a.generated) < 3:
        eng.step()
    assert eng.cancel("a")
    eng.step()
    assert a.status == "cancelled" and a.done.is_set()
    assert len(eng.pool.held("a")) == 0
    eng.run_until_idle()
    assert b.generated == reference(model, b.prompt_ids, 6)[0]
    assert eng.pool.stats() == {"free": 64, "pending": 0, "owned": 0, "requests": 0, "total": 64}


def test_deadline_expires(model):
    t = [0.0]
    eng = make_engine(model)
    eng.clock = lambda: t[0]
    r = eng.submit(Request("d", prompts(1, 11)[0], 50, deadline=5.0))
    for _ in range(4):
        eng.step()
        t[0] += 1.0
    t[0] = 6.0
    eng.step()
    assert r.status == "deadline" and 0 < len(r.generated) < 50
    assert eng.pool.stats()["owned"] == 0


def test_shutdown_cancels_everything(model):
    eng = make_engine(model)
    rs = [eng.submit(Request(f"s{i}", p, 20)) for i, p in enumerate(prompts(3, 12))]
    eng.step()
    eng.shutdown()
    assert all(r.status == "cancelled" for r in rs)
    with pytest.raises(Refused, match="shutting_down"):
        eng.submit(Request("late", [1, 2], 2))
    assert eng.pool.stats()["owned"] == 0


def test_churn_returns_to_steady_state(model):
    eng = make_engine(model, pages=32, page_size=8, max_batch=4)
    base = eng.pool.stats()
    for round_ in range(5):
        rs = [eng.submit(Request(f"c{round_}-{i}", p, 4 + i))
              for i, p in enumerate(prompts(4, 20 + round_))]
        eng.step()
        eng.cancel(rs[0].request_id)
        eng.run_until_idle()
        assert eng.pool.stats() == base


def test_threaded_submit_and_cancel(model):
    eng = make_engine(model, max_batch=4)
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            if not eng.step():
                time.sleep(0.001)
    th = threading.Thread(target=loop)
    th.start()
    try:
        rs = [eng.submit(Request(f"t{i}", p, 10)) for i, p in enumerate(prompts(5, 30))]
        eng.cancel("t2")
        for r in rs:
            assert r.done.wait(30)
    finally:
        stop.set()
        th.join()
    for i, r in enumerate(rs):
        if r.status == "length":
            assert r.generated == reference(model, r.prompt_ids, 10)[0]
        else:
            assert i == 2 and r.status == "cancelled"
    assert eng.pool.stats()["owned"] == 0


# -- ownership mutants --------------------------------------------------------

class Pending:
    """A fence that stays incomplete until fired, like a slow CUDA kernel."""

    def __init__(self):
        self.fired = False

    def query(self):
        return self.fired

    def synchronize(self):
        self.fired = True


def small_pool(fence=None, pages=2):
    return PagePool(num_layers=1, num_pages=pages, num_kv_heads=1, page_size=4,
                    nb_k=4, nb_v=4, fence_factory=fence)


def test_released_page_waits_for_fence():
    pool, ev = small_pool(), Pending()
    pool.ensure("a", 8)
    pool.release("a", ev)
    with pytest.raises(Exception, match="awaiting fences"):
        pool.ensure("b", 4)
    ev.fired = True
    assert pool.ensure("b", 4)


def test_mutant_reclaim_ignoring_fence_is_caught(monkeypatch):
    """With the fence check removed, the page is handed out while in flight."""
    monkeypatch.setattr(PagePool, "_event_done", lambda self, f: True)
    pool, ev = small_pool(), Pending()
    pool.ensure("a", 8)
    pool.release("a", ev)
    reused = pool.ensure("b", 4)          # succeeds only because of the mutant
    assert reused and not ev.query()      # the condition the real pool forbids


def _reuse_scenario():
    """Client reuses request id "a" after it finished; the new "a" receives
    the same physical page. A launch still holding the old refs must fail."""
    pool = small_pool(pages=1)
    old = pool.ensure("a", 4)
    pool.release("a")
    assert pool.ensure("a", 4)[0].page == old[0].page
    return pool, old


def test_stale_refs_rejected_after_id_reuse():
    pool, old = _reuse_scenario()
    with pytest.raises(StalePage):
        pool.validate("a", old)


def test_mutant_without_generation_bump_is_caught(monkeypatch):
    real_release = PagePool.release

    def no_bump(self, owner, fence=None):
        refs = list(self._held.get(owner, []))
        n = real_release(self, owner, fence)
        for r in refs:
            self._gen[r.page] -= 1        # undo the bump: the seeded bug
        return n
    monkeypatch.setattr(PagePool, "release", no_bump)
    pool, old = _reuse_scenario()
    pool.validate("a", old)               # the mutant lets the stale launch through


# -- matched unquantized arm --------------------------------------------------

@torch.no_grad()
def native_reference(model, prompt, max_new):
    model.config._attn_implementation = "sdpa"
    # Explicit all-ones mask: prompts may contain the pad id, which generate
    # would otherwise infer as padding and mask out.
    ids = torch.tensor([prompt])
    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
                         do_sample=False, min_new_tokens=max_new, pad_token_id=0)
    return out[0, len(prompt):].tolist()


def test_dense_arm_matches_native_generate(model):
    ps, news = prompts(4, seed=40), [6, 11, 3, 8]
    model.config._attn_implementation = pb.ATTN_NAME
    pool = pb.make_pool(model.config, bit_width=None, num_pages=64, page_size=8,
                        device="cpu", dense_dtype=torch.float32)
    eng = Engine(model, pool, None, None, pb.head_dim_of(model.config),
                 config=EngineConfig(max_batch=3, prefill_chunk=9))
    pb.BACKEND_COUNTS.clear()
    reqs = run(eng, [Request(f"d{i}", p, n) for i, (p, n) in enumerate(zip(ps, news))])
    for r, p, n in zip(reqs, ps, news):
        assert r.generated == native_reference(model, p, n)
    assert pb.BACKEND_COUNTS["dense_sdpa"] > 0 and pb.BACKEND_COUNTS["native_sdpa"] == 0


def test_forward_after_release_fails_without_reallocating(model):
    """Regression: a forward racing shutdown re-acquired a finished request's
    pages through ensure() and leaked them. Now it must raise and hold nothing."""
    eng = make_engine(model)
    r = eng.submit(Request("x", prompts(1, 70)[0], 5))
    eng.step()
    eng.pool.release("x")
    with pytest.raises(StalePage):
        eng._forward([r], torch.tensor([[1]]), torch.tensor([[r.stored]]))
    assert eng.pool.held("x") == [] and eng.pool.stats()["owned"] == 0
