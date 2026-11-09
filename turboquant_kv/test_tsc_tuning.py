"""Track 02 CPU contracts: legality, bounded search, cost model, promotion, dispatch.

The search tests use a synthetic measurer with a known cost landscape, known
compile/launch failures and a fast-but-wrong candidate, so selection, budget
accounting and the correctness screen can be checked exactly. Real timings
come only from tune_kernels.py on a GPU.
"""
import json
import math
import multiprocessing as mp
import random

import pytest

from tsc import schedule as sch
from tsc import search
from tsc.promotion import PromotionError, Registry, identity_key, seal

A100 = sch.TARGETS["sm80"]
A6000 = sch.TARGETS["sm86"]
QWEN7B = dict(hq=28, hkv=4, head_dim=128, page_size=16, bit_width=8)


# -- legality ----------------------------------------------------------------

def test_dequant_legality_is_deterministic_and_explained():
    p = sch.DequantProblem(rows=4096, head_dim=128, bit_width=8)
    ok, why = sch.dequant_legal(sch.DequantSchedule(256, 4), p, A6000)
    assert not ok and "shared memory" in why            # (256*128 + 128*128)*4 > 99 KB
    ok, why = sch.dequant_legal(sch.DequantSchedule(128, 1), p, A100)
    assert not ok and "registers" in why                 # 128*128/32 = 512 regs/thread
    ok, why = sch.dequant_legal(sch.DequantSchedule(8, 4), p, A100)
    assert not ok and "power of two >= 16" in why
    assert sch.dequant_legal(sch.HEURISTIC_DEQUANT, p, A100) == (True, "legal")
    legal, bad = sch.enumerate_legal(sch.DEQUANT_SPACE, p, A100, sch.dequant_legal)
    legal2, _ = sch.enumerate_legal(sch.DEQUANT_SPACE, p, A100, sch.dequant_legal)
    assert legal == legal2 and len(legal) + len(bad) == len(sch.DEQUANT_SPACE)
    assert all(isinstance(r, str) and r for _, r in bad)


def test_attention_legality():
    short = sch.AttnProblem(batch=4, seq_len=40, **QWEN7B)          # 3 pages
    ok, why = sch.attn_legal(sch.AttnSchedule(1, 4, 4, 2), short, A100)
    assert not ok and "exceeds" in why
    ok, why = sch.attn_legal(sch.AttnSchedule(4, 1, 4, 2), short, A100)
    assert not ok and "empty splits" in why
    big = sch.AttnProblem(batch=4096, seq_len=8192, **QWEN7B)
    ok, why = sch.attn_legal(sch.AttnSchedule(16, 1, 4, 1), big, A100)
    assert not ok and "workspace" in why
    ok, why = sch.attn_legal(sch.AttnSchedule(1, 8, 4, 3), sch.AttnProblem(4, 8192, **QWEN7B), A6000)
    assert not ok and "shared memory" in why
    assert sch.attn_legal(sch.HEURISTIC_ATTN, sch.AttnProblem(16, 8192, **QWEN7B), A100)[0]


def test_family_excludes_sizes():
    a = sch.AttnProblem(1, 512, **QWEN7B).family()
    b = sch.AttnProblem(32, 8192, **QWEN7B).family()
    assert a == b and "batch" not in a


# -- search with a synthetic measurer -------------------------------------------

class Synthetic(search.Measurer):
    """Cost landscape over the attention space with a known optimum, plus faults."""

    def __init__(self, seed=0, broken=(), crash=(), wrong=(), compile_s=2.0, trial_s=0.05):
        self.rng = random.Random(seed)
        self.broken, self.crash, self.wrong = set(broken), set(crash), set(wrong)
        self.compile_s, self.trial_s = compile_s, trial_s
        self.compiled = []

    def true_us(self, s):
        return (100 * (1 + abs(math.log2(s.num_splits) - 2)) * (1 + 0.5 * abs(math.log2(s.pages_per_iter) - 2))
                * (1 + 0.2 * abs(math.log2(s.num_warps) - 2)) * (1 + 0.1 * abs(s.num_stages - 2)))

    def compile(self, s):
        self.compiled.append(s.key())
        if s.key() in self.broken:
            e = search.CompileError("synthetic compile failure")
            e.seconds = self.compile_s
            raise e
        return self.compile_s

    def time(self, s, trials):
        if s.key() in self.crash:
            raise search.LaunchError("synthetic launch failure")
        base = self.true_us(s) * (0.5 if s.key() in self.wrong else 1.0)
        return [base * (1 + 0.02 * self.rng.gauss(0, 1)) for _ in range(trials)], trials * self.trial_s

    def validate(self, s):
        ok = s.key() not in self.wrong
        return {"passed": ok, "reason": "passed" if ok else "beyond envelope"}, 1.0

    def features(self, s):
        return [math.log2(s.num_splits), math.log2(s.pages_per_iter), math.log2(s.num_warps),
                s.num_stages, math.log2(s.num_splits) ** 2, math.log2(s.pages_per_iter) ** 2,
                math.log2(s.num_warps) ** 2, s.num_stages ** 2]


PROB = sch.AttnProblem(16, 8192, **QWEN7B)
LEGAL, ILLEGAL = sch.enumerate_legal(sch.ATTN_SPACE, PROB, A100, sch.attn_legal)
OPT = min(LEGAL, key=Synthetic().true_us)


def run_all(m_factory, seconds=200.0, compiles=40, seed=0):
    out = {}
    for name, fn in (("random", search.random_search), ("sh", search.successive_halving),
                     ("model", search.model_guided)):
        b = search.Budget(seconds, compiles)
        out[name] = fn(m_factory(), LEGAL, b, seed=seed, illegal=ILLEGAL)
    out["heuristic"] = search.heuristic(m_factory(), sch.HEURISTIC_ATTN, search.Budget(seconds, compiles))
    return out


def test_every_strategy_respects_the_compile_budget_and_validates():
    res = run_all(lambda: Synthetic(), compiles=25)
    for name, r in res.items():
        assert r.budget.compiles_used <= 25, name
        sel = [e for e in r.ledger if e.status == "selected"]
        assert len(sel) == (1 if r.selected else 0)
        assert all(e.validation is not None for e in sel)


def test_fast_wrong_candidate_is_never_selected():
    wrong = OPT.key()
    for seed in range(3):
        res = run_all(lambda: Synthetic(seed, wrong=[wrong]), seed=seed, compiles=60)
        for name, r in res.items():
            assert r.selected is None or r.selected.key() != wrong, name
            e = next((e for e in r.ledger if e.key == wrong), None)
            if e is not None and e.validation is not None:
                assert e.status == "rejected_validation"


def test_compile_and_launch_failures_are_ledger_outcomes():
    keys = [s.key() for s in LEGAL]
    broken = keys[::7]
    crash = [k for k in keys[3::11] if k not in broken]       # compile failure wins
    res = run_all(lambda: Synthetic(broken=broken, crash=crash), compiles=len(LEGAL))
    r = res["random"]
    statuses = {e.key: e.status for e in r.ledger}
    assert all(statuses.get(k) == "compile_error" for k in broken if k in statuses)
    assert all(statuses.get(k) == "launch_error" for k in crash if k in statuses)
    assert r.selected is not None and r.selected.key() not in set(broken) | set(crash)


def test_no_valid_candidate_keeps_the_safe_path():
    everything = [s.key() for s in LEGAL] + [sch.HEURISTIC_ATTN.key()]
    res = run_all(lambda: Synthetic(wrong=everything), compiles=20)
    for r in res.values():
        assert r.selected is None and "safe path" in r.reason


def test_guided_search_regret_under_equal_budget():
    """On a smooth landscape the model-guided and halving searches should land
    near the optimum with a small compile budget; reported, not tuned to win."""
    best = Synthetic().true_us(OPT)
    regrets = {k: [] for k in ("random", "sh", "model")}
    for seed in range(5):
        res = run_all(lambda: Synthetic(seed), seconds=1e9, compiles=24, seed=seed)
        for k in regrets:
            sel = res[k].selected
            regrets[k].append(Synthetic().true_us(sel) / best - 1)
    med = {k: sorted(v)[2] for k, v in regrets.items()}
    assert med["model"] <= 0.25 and med["sh"] <= 0.6, med


def test_cost_model_recovers_a_linear_law_and_is_less_sure_off_data():
    rng = random.Random(0)
    X = [[rng.uniform(0, 1), rng.uniform(0, 1)] for _ in range(60)]
    y = [2 * a - 3 * b + 1 + rng.gauss(0, 0.01) for a, b in X]
    cm = search.CostModel(seed=1).fit(X, y)
    mean, std = cm.predict([[0.5, 0.5], [5.0, -5.0]])
    assert abs(mean[0] - 0.5) < 0.05
    assert std[1] > 5 * std[0]


# -- promotion registry ----------------------------------------------------------

IDENT = {"op": "packed_decode_attention", "family": PROB.family(), "contract": "fp32",
         "kernel_source_sha": "a" * 64, "toolchain": {"torch": "2.14.0", "triton": "3.7", "cuda": "13.0"},
         "target": {"cc": [8, 0], "name": "NVIDIA A100-SXM4-80GB"}}
GOOD_VAL = {"passed": True, "reason": "passed"}
BENCH = {"baseline_median_us": 2700.0, "candidate_median_us": 900.0, "samples_path": "x"}
COV = {"batch": {"min": 1, "max": 32}, "seq_len": {"min": 512, "max": 8192}}


def promote(reg, ident=IDENT, **kw):
    return reg.promote(ident, {"buckets": []}, kw.get("val", GOOD_VAL), kw.get("bench", BENCH),
                       kw.get("cov", COV))


def test_promotion_refuses_unvalidated_or_unhelpful(tmp_path):
    reg = Registry(tmp_path)
    with pytest.raises(PromotionError):
        promote(reg, val={"passed": False, "reason": "mutant"})
    with pytest.raises(PromotionError):
        promote(reg, bench={"baseline_median_us": 100.0, "candidate_median_us": 120.0})
    assert reg.load(IDENT)[0] is None


def test_load_checks_identity_checksum_and_coverage(tmp_path):
    reg = Registry(tmp_path)
    path = promote(reg)
    rec, why = reg.load(IDENT, size={"batch": 16, "seq_len": 2048})
    assert rec is not None and why == "promoted"
    assert reg.load(IDENT, size={"batch": 64, "seq_len": 2048}) == (None, "size {'batch': 64, 'seq_len': 2048} outside validated coverage")
    for field, value in (("target", {"cc": [8, 6], "name": "NVIDIA RTX A6000"}),
                         ("toolchain", {**IDENT["toolchain"], "triton": "3.8"}),
                         ("kernel_source_sha", "b" * 64)):
        other = {**IDENT, field: value}
        rec, why = reg.load(other)
        assert rec is None and "no promotion" in why        # different identity, different key
    # Same key, tampered content: checksum must catch it.
    data = json.loads(path.read_text())
    data["schedule"] = {"buckets": [{"batch_max": 99, "len_max": 99, "schedule": {}}]}
    path.write_text(json.dumps(data))
    rec, why = reg.load(IDENT)
    assert rec is None and "checksum" in why
    path.write_bytes(b"\x00not json")
    assert "unreadable" in reg.load(IDENT)[1]


def test_identity_collision_is_detected(tmp_path):
    """A record whose stored identity differs from the requested one (e.g. a
    renamed file) is refused even though its checksum is valid."""
    reg = Registry(tmp_path)
    path = promote(reg)
    other = {**IDENT, "contract": "tf32"}
    (tmp_path / path.name).rename(reg.path(other))
    rec, why = reg.load(other)
    assert rec is None and "identity mismatch" in why and "contract" in why


def _writer(root, i):
    bench = {**BENCH, "candidate_median_us": 900.0 + i}
    Registry(root).promote(IDENT, {"buckets": [], "writer": i}, GOOD_VAL, bench, COV)


def test_concurrent_promotions_leave_one_intact_record(tmp_path):
    ps = [mp.get_context("spawn").Process(target=_writer, args=(str(tmp_path), i)) for i in range(8)]
    [p.start() for p in ps]
    [p.join(60) for p in ps]
    assert all(p.exitcode == 0 for p in ps)
    rec, why = Registry(tmp_path).load(IDENT)
    assert rec is not None, why
    assert not list(tmp_path.glob("*.tmp.*"))


def test_seal_is_order_independent():
    a = seal({"x": 1, "y": {"b": 2, "a": 1}})
    b = seal({"y": {"a": 1, "b": 2}, "x": 1})
    assert a["checksum"] == b["checksum"] and identity_key(IDENT) == identity_key(dict(reversed(IDENT.items())))


# -- dispatch decisions -------------------------------------------------------------

def test_dispatch_falls_back_with_reasons(tmp_path, monkeypatch):
    import tuned_dispatch as td
    d = td.TunedDispatch(None)
    assert d.dequant_decision(4096, 8, 128, "cpu").path == "torch"
    assert d.attention_schedule(PROB.family(), 16, 8192, "cpu").reason == "not a CUDA device"
    monkeypatch.setattr(td, "target_of", lambda dev: {"cc": [8, 0], "name": "NVIDIA A100-SXM4-80GB"})
    monkeypatch.setattr(td, "handwritten_available", lambda: True)
    assert d.dequant_decision(4096, 8, 128, "cuda").reason == "no tuned registry configured"
    d = td.TunedDispatch(tmp_path)
    dec = d.dequant_decision(4096, 8, 128, "cuda")
    assert dec.path == "handwritten" and "cache miss" in dec.reason


def test_dispatch_uses_promoted_record_and_rejects_stale_source(tmp_path, monkeypatch):
    import hashlib

    import tuned_dispatch as td
    from tsc.compiler import compile_source
    from tsc.examples import example_source
    tgt = {"cc": [8, 0], "name": "NVIDIA A100-SXM4-80GB"}
    monkeypatch.setattr(td, "target_of", lambda dev: tgt)
    monkeypatch.setattr(td, "handwritten_available", lambda: True)
    real_sha = hashlib.sha256(compile_source(example_source(8, 128)).generated.encode()).hexdigest()
    reg = Registry(tmp_path)
    ident = td.dequant_identity(8, 128, "fp32", tgt)
    for sha, expect in (("0" * 64, "handwritten"), (real_sha, "generated")):
        reg.promote(ident, {"block_n": 128, "num_warps": 8}, GOOD_VAL,
                    {"baseline_median_us": 50.0, "candidate_median_us": 40.0},
                    {"rows": {"min": 1, "max": 1 << 24}}, extra={"generated_source_sha": sha})
        d = td.TunedDispatch(tmp_path)
        monkeypatch.setattr(d, "_generated", lambda bw, hd, want, d=d: (
            (object() if want == real_sha else None), real_sha))
        dec = d.dequant_decision(4096, 8, 128, "cuda")
        assert dec.path == expect, dec
        if expect == "handwritten":
            assert "stale record" in dec.reason
        else:
            assert dec.schedule == {"block_n": 128, "num_warps": 8}


def test_attention_bucket_selection(tmp_path, monkeypatch):
    import tuned_dispatch as td
    tgt = {"cc": [8, 0], "name": "NVIDIA A100-SXM4-80GB"}
    monkeypatch.setattr(td, "target_of", lambda dev: tgt)
    fam = PROB.family()
    buckets = [{"batch_max": 4, "len_max": 8192, "schedule": {"num_splits": 8, "pages_per_iter": 4,
                                                               "num_warps": 4, "num_stages": 2}},
               {"batch_max": 32, "len_max": 8192, "schedule": {"num_splits": 2, "pages_per_iter": 4,
                                                                "num_warps": 4, "num_stages": 2}}]
    Registry(tmp_path).promote(td.attention_identity(fam, tgt), {"buckets": buckets}, GOOD_VAL,
                               BENCH, COV)
    d = td.TunedDispatch(tmp_path)
    assert d.attention_schedule(fam, 2, 4096, "cuda").schedule["num_splits"] == 8
    assert d.attention_schedule(fam, 16, 4096, "cuda").schedule["num_splits"] == 2
    out = d.attention_schedule(fam, 64, 4096, "cuda")
    assert out.path == "default" and "coverage" in out.reason
