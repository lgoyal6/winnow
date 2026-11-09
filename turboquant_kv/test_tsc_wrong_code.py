"""The correctness screen must pass real generated code and reject wrong code.

Runs where generated Triton can execute: CUDA, or TRITON_INTERPRET=1 (kernel
logic on CPU, not GPU evidence). Elsewhere every test skips, which means this
gate is blocked, not passed.
"""
import os

import pytest
import torch

INTERP = os.environ.get("TRITON_INTERPRET") == "1"
DEV = "cuda" if torch.cuda.is_available() else ("cpu" if INTERP else None)
pytestmark = pytest.mark.skipif(DEV is None, reason="BLOCKED: needs CUDA (or TRITON_INTERPRET=1)")

if DEV is not None:
    from tsc.compiler import CompileResult, compile_source
    from tsc.examples import example_source
    from tsc.runtime import load_generated
    from tsc.schedule import AttnSchedule, DequantSchedule
    from tsc.validation import MUTANTS, validate_attention, validate_dequant

BWS = (8, 4) if INTERP else (8, 4, 3)
SCHEDULES = ((64, 4), (128, 8), (16, 1)) if not INTERP else ((64, 4), (32, 2))


def module_for(bw, mutate=None):
    res = compile_source(example_source(bw, 128))
    src = res.generated if mutate is None else mutate(res.generated)
    return load_generated(CompileResult(res.key + (mutate.__name__ if mutate else ""), src,
                                        res.ir_before, res.ir_after, res.stats, False, 0.0))


@pytest.mark.parametrize("bw", BWS)
@pytest.mark.parametrize("bn,w", SCHEDULES)
def test_generated_candidate_passes(bw, bn, w):
    r = validate_dequant(module_for(bw).run, DequantSchedule(bn, w), bit_width=bw, head_dim=128,
                         device=DEV)
    assert r.passed, r.reason
    assert r.nonfinite == 0 and r.canary_violations == 0


@pytest.mark.parametrize("name", sorted(MUTANTS) if DEV else [])
@pytest.mark.parametrize("bw", BWS)
def test_mutant_is_rejected(name, bw):
    mod = module_for(bw, MUTANTS[name])
    r = validate_dequant(mod.run, DequantSchedule(64, 4), bit_width=bw, head_dim=128, device=DEV)
    assert not r.passed, f"{name} slipped through"
    if name == "dropped_store_mask":
        assert r.canary_violations > 0 or r.beyond_envelope > 0


@pytest.mark.parametrize("sched", [(1, 1, 4, 2), (4, 2, 4, 2), (2, 4, 8, 1)])
def test_attention_schedules_pass(sched):
    r = validate_attention(AttnSchedule(*sched), hq=28, hkv=4, head_dim=128, page_size=16,
                           bit_width=8, device=DEV)
    assert r.passed, r.reason


def test_attention_wrong_scale_is_rejected(monkeypatch):
    import packed_attention_triton as pt
    real = pt._launch
    monkeypatch.setattr(pt, "_launch", lambda q, kv, scale, **k: real(q, kv, scale * 1.05, **k))
    r = validate_attention(AttnSchedule(2, 2, 4, 2), hq=28, hkv=4, head_dim=128, page_size=16,
                           bit_width=8, device=DEV)
    assert not r.passed
