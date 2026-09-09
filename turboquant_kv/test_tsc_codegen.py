"""Codegen tests: the backend is IR-driven, not a template matcher."""
from __future__ import annotations

import pytest

from tsc.codegen import generate_triton
from tsc.errors import TSCError
from tsc.examples import example_source
from tsc.ir import lower
from tsc.optimize import optimize
from tsc.parser import parse
from tsc.validate import validate


def _module(bw: int = 4, src: str | None = None, optimize_ir: bool = True):
    m = lower(validate(parse(src if src is not None else example_source(bw))))
    if optimize_ir:
        m, _ = optimize(m)
    return m


def test_codegen_is_standalone_executable_python():
    source = generate_triton(_module())
    compile(source, "generated_tq_dequant.py", "exec")
    assert "@triton.jit" in source
    assert "tl.dot" in source
    assert "allow_tf32=ALLOW_TF32" in source
    assert "MAXLEN" in source                      # strided cache-slice support


def test_kernel_meta_records_the_schedule():
    source = generate_triton(_module())
    meta_line = next(l for l in source.splitlines()
                     if l.startswith("KERNEL_META"))
    ns: dict = {}
    exec(meta_line, ns)                            # just the metadata constant
    assert ns["KERNEL_META"] == {
        "pipeline": "tq_dequant", "head_dim": 128, "nb": 64, "bits": 4,
        "indexing_delta": 0}


def test_unoptimized_ir_also_compiles():
    # The backend emits any valid IR, including the normalized form with
    # redundant casts and a separate unpack; that is what makes disabling
    # the optimizer a real negative control rather than a crash.
    source = generate_triton(_module(optimize_ir=False))
    compile(source, "generated_noopt.py", "exec")
    assert "unpack" in source and "gather" in source
    assert source.count(".to(tl.float32)") >= 4    # normalization casts emitted


def test_optimized_source_is_smaller_than_unoptimized():
    opt = generate_triton(_module())
    noopt = generate_triton(_module(optimize_ir=False))
    assert len(opt.splitlines()) < len(noopt.splitlines())
    assert "decode" in opt and "unpack %codes" not in opt


def test_mutated_but_valid_operation_order_still_compiles():
    # Reordering the norm cast ahead of the rotation is legal SSA; a real
    # backend generates it rather than rejecting an unexpected op sequence.
    raw = example_source(4).replace(
        "  x     = rot(y, rotm)\n  nf    = cast(nrm, f32)",
        "  nf    = cast(nrm, f32)\n  x     = rot(y, rotm)",
    )
    source = generate_triton(_module(src=raw))
    compile(source, "generated_reordered.py", "exec")
    assert "tl.dot" in source


def test_bw8_generates_direct_byte_load():
    source = generate_triton(_module(bw=8))
    assert "b1" not in source                      # no straddle machinery
    assert "decode" in source
    assert "KERNEL_META" in source and "'bits': 8" in source


def test_bw3_generates_straddling_extraction():
    src = example_source(3)
    source = generate_triton(_module(src=src))
    assert "(b1 << (8 - sh))" in source
    assert "& 7" in source                         # (1 << 3) - 1


def test_fault_injection_changes_generated_indexing():
    clean = generate_triton(_module())
    injected = generate_triton(_module(), indexing_delta=1)
    assert clean != injected
    assert "byte0 + 1" in clean                    # the straddle byte
    assert "byte0 + 1 + 1" in injected             # off-by-one on the high byte
    assert "base + byte0 + 1," in injected         # and on the low byte
    compile(injected, "generated_injected.py", "exec")


def test_double_output_program_is_rejected_by_backend():
    src = example_source(4).replace(
        "output out:       bf16[N, head_dim]",
        "output out:       bf16[N, head_dim]\n"
        "  output out2:      bf16[N, head_dim]").replace(
        "store(out, o)", "store(out, o)\n  store(out2, o)")
    with pytest.raises(TSCError, match="exactly one output"):
        generate_triton(_module(src=src))


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
    print(f"\n{len(names) - failed}/{len(names)} tsc codegen tests passed.")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    _run_all()
