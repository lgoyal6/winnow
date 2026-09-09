"""AST-to-IR golden tests: lowering into normalized, explicitly-typed IR."""
from __future__ import annotations

from tsc.examples import EXAMPLES_DIR, example_source
from tsc.ir import dump_ir, lower
from tsc.optimize import optimize
from tsc.parser import parse
from tsc.validate import validate

GOLDENS = EXAMPLES_DIR.parent / "goldens"


def _lowered(bw: int = 4):
    return lower(validate(parse(example_source(bw))))


def test_lowering_golden_bw4():
    assert dump_ir(_lowered(4)) == (GOLDENS / "tq_dequant_bw4_ir.txt").read_text()


def test_lowering_golden_bw8():
    assert dump_ir(_lowered(8)) == (GOLDENS / "tq_dequant_bw8_ir.txt").read_text()


def test_optimized_golden_bw4():
    opt, _ = optimize(_lowered(4))
    assert dump_ir(opt) == (GOLDENS / "tq_dequant_bw4_ir_opt.txt").read_text()


def test_optimized_golden_bw8():
    opt, _ = optimize(_lowered(8))
    assert dump_ir(opt) == (GOLDENS / "tq_dequant_bw8_ir_opt.txt").read_text()


def test_ast_lowers_to_explicit_ir():
    text = dump_ir(_lowered(4))
    assert "%idx: u8[N, 128] = unpack %codes {bits=4}" in text
    assert "%y: f32[N, 128] = gather %cb, %idx" in text
    assert text.endswith("  store %out, %o\n")


def test_lowering_normalizes_compute_operands():
    # Lowering inserts explicit f32 casts on rot and rescale operands; on a
    # valid program those are no-ops that the cleanup pass later removes.
    text = dump_ir(_lowered(4))
    assert "%y__f32: f32[N, 128] = cast %y, f32" in text
    assert "%rotm__f32: f32[128, 128] = cast %rotm, f32" in text
    assert "rot %y__f32, %rotm__f32" in text
    assert "rescale %x__f32, %nf__f32" in text


def test_every_instruction_is_typed():
    module = _lowered(4)
    for ins in module.instrs:
        if ins.op != "store":
            assert ins.result is not None and ins.result.type is not None, ins
    types = module.value_types()
    assert str(types["idx"]) == "u8[N, 128]"
    assert str(types["z"]) == "f32[N, 128]"
    assert str(types["out"]) == "bf16[N, 128]"


def test_unpack_bits_attr_is_resolved_to_int():
    module = _lowered(4)
    unpack = next(i for i in module.instrs if i.op == "unpack")
    assert dict(unpack.attrs) == {"bits": 4}


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
    print(f"\n{len(names) - failed}/{len(names)} tsc IR tests passed.")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    _run_all()
