"""Regression tests for the GPU differential's numerical acceptance model.

The 2026-09-08 A6000 run proved the old fixed max-abs gates (fp32 1e-5,
tf32 3.125e-2) mis-specified: they flagged 16/48 rows whose worst mismatch
was 1 bf16 output ULP, benign rounding from differing fp32/tf32 accumulation
order. The replacement admits exactly the proven envelope, element-wise:
bf16 ULP distance <= 1, or the near-zero floor (fp32: |err| <= 2^-21 with
|ref| <= 2^-14; tf32: |err| <= 2^-8 with |ref| <= 0.5). These tests pin the
envelope with bf16-exact values so it can neither silently widen nor
re-reject proven-benign rounding. All values are constructed from the bf16
ULP rule ulp(x) = 2^(e-7) for |x| in [2^e, 2^(e+1)).
"""
from __future__ import annotations

import importlib.util
import os

import pytest

torch = pytest.importorskip("torch")

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "run_gpu_differential",
    os.path.join(os.path.dirname(HERE), "tools", "run_gpu_differential.py"))
rgd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rgd)


def _bf16(*values):
    return torch.tensor(list(values), dtype=torch.bfloat16)


def test_one_ulp_rounding_passes_envelope_but_failed_legacy_gate():
    # The mismatch class behind all 16 rows the old gate failed: 1 bf16 ULP
    # at magnitude (here 2.0 -> 2.015625). A harness-gate artifact, not a
    # kernel failure: the envelope admits it, the legacy fixed gate did not.
    got, ref = _bf16(2.015625), _bf16(2.0)
    for mode in ("fp32", "tf32"):
        diag = rgd.check_envelope(got, ref, mode)
        assert diag["beyond_envelope"] == 0
        assert diag["max_ulp"] == 1
    assert diag["max_abs_err"] > rgd.LEGACY_FIXED_GATES["fp32"]


def test_two_ulp_at_magnitude_is_rejected_in_both_modes():
    got, ref = _bf16(2.03125), _bf16(2.0)   # 2 ULP at |ref| in [2, 4)
    for mode in ("fp32", "tf32"):
        assert rgd.check_envelope(got, ref, mode)["beyond_envelope"] == 1


def test_fp32_near_zero_floor_is_exactly_the_proven_bound():
    ref = _bf16(2.0 ** -15)
    within = _bf16(2.0 ** -15 + 2.0 ** -21)   # 2 ULP, err == 2^-21: proven
    beyond = _bf16(2.0 ** -15 + 2.0 ** -20)   # 4 ULP, err == 2^-20: not
    assert rgd.check_envelope(within, ref, "fp32")["beyond_envelope"] == 0
    assert rgd.check_envelope(beyond, ref, "fp32")["beyond_envelope"] == 1


def test_tf32_near_zero_floor_is_exactly_the_proven_bound():
    # Multi-ULP error within the tf32 floor implies |ref| < 0.5 (bf16 ULP
    # at [0.5, 1) is already 2^-8), so the |ref| clause is documentation of
    # the proven envelope; the active boundary is the 2^-8 error floor.
    ref = _bf16(0.25)
    within = _bf16(0.25 + 2.0 ** -8)   # 2 ULP, err == 2^-8: tf32 truncation
    beyond = _bf16(0.25 + 2.0 ** -7)   # 4 ULP, err == 2^-7: not admitted
    assert rgd.check_envelope(within, ref, "tf32")["beyond_envelope"] == 0
    assert rgd.check_envelope(beyond, ref, "tf32")["beyond_envelope"] == 1
    # the same 2-ULP error is never admitted under fp32 accumulation
    assert rgd.check_envelope(within, ref, "fp32")["beyond_envelope"] == 1


def test_planted_fault_magnitude_stays_caught():
    # The 2026-09-08 injected off-by-one produced errors >= 3.92; the
    # envelope must keep rejecting that in both modes, as must sign flips.
    for got in (_bf16(5.92), _bf16(-2.0)):
        for mode in ("fp32", "tf32"):
            diag = rgd.check_envelope(got, _bf16(2.0), mode)
            assert diag["beyond_envelope"] == 1
            assert diag["max_ulp"] > 100   # bf16 binades are 128 ULP wide


def test_ulp_distance_orders_negative_values():
    assert int(rgd.bf16_ulp_distance(_bf16(-2.015625), _bf16(-2.0))) == 1
    assert int(rgd.bf16_ulp_distance(_bf16(-0.0), _bf16(0.0))) == 0


def _run_all():
    import traceback
    names = [k for k in sorted(globals()) if k.startswith("test_")]
    failed = 0
    for name in names:
        try:
            globals()[name]()
            print(f"  ok  {name}")
        except Exception:
            failed += 1
            traceback.print_exc()
    print(f"\n{len(names) - failed}/{len(names)} tsc envelope tests passed.")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    _run_all()
