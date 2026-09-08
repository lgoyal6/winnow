"""Parser tests for the tensor-schedule DSL: goldens plus malformed syntax.

Golden files live in tsc/goldens/. They are committed source-of-truth: if a
grammar change moves the dump, the diff must be reviewed, not regenerated
blindly.
"""
from __future__ import annotations

import pytest

from tsc.errors import TSCSyntaxError
from tsc.examples import EXAMPLES_DIR, example_source, load_example
from tsc.parser import parse
from tsc.syntax import Assign, ExprStmt, dump_ast

GOLDENS = EXAMPLES_DIR.parent / "goldens"


# -- goldens -----------------------------------------------------------------
def test_parser_golden_bw4():
    got = dump_ast(parse(load_example("tq_dequant_bw4.tsc")))
    assert got == (GOLDENS / "tq_dequant_bw4_ast.txt").read_text()


def test_parser_golden_bw8():
    got = dump_ast(parse(load_example("tq_dequant_bw8.tsc")))
    assert got == (GOLDENS / "tq_dequant_bw8_ast.txt").read_text()


def test_example_generator_matches_fixtures():
    # The programmatic generator and the checked-in fixtures must not drift.
    assert example_source(4) == load_example("tq_dequant_bw4.tsc")
    assert example_source(8) == load_example("tq_dequant_bw8.tsc")


def test_parsed_structure_bw4():
    prog = parse(example_source(4))
    assert prog.name == "tq_dequant"
    assert {p.name: p.value for p in prog.params} == {
        "bit_width": 4, "head_dim": 128}
    assert [d.kind for d in prog.decls] == ["input"] * 4 + ["output"]
    packed = prog.decls[0]
    assert packed.type.dtype == "u8" and packed.type.shape == ("N", 64)
    assert isinstance(prog.stmts[-1], ExprStmt)
    assert prog.stmts[-1].call.op == "store"
    assigns = [s for s in prog.stmts if isinstance(s, Assign)]
    assert [a.call.op for a in assigns] == [
        "load", "load", "load", "load", "unpack", "gather", "rot", "cast",
        "rescale", "cast"]
    unpack = next(a for a in assigns if a.call.op == "unpack")
    attrs = dict(unpack.call.attrs)
    assert str(attrs["bits"]) == "bit_width"


# -- malformed syntax ---------------------------------------------------------
def _rejects(src: str, needle: str):
    with pytest.raises(TSCSyntaxError) as err:
        parse(src)
    assert needle in str(err.value), str(err.value)


def test_missing_closing_brace():
    _rejects("pipeline p() { input a: u8[4]\n a2 = load(a)",
             "expected a statement")


def test_unknown_op():
    _rejects("pipeline p() { input a: u8[4]\n b = scatter(a) }",
             "unknown op 'scatter'")


def test_unknown_dtype():
    _rejects("pipeline p() { input a: i64[4] }", "unknown dtype 'i64'")


def test_wrong_arity():
    _rejects("pipeline p() { input a: u8[4]\n input b: u8[4]\n"
             " c = load(a, b) }", "'load' takes 1 argument(s), got 2")


def test_store_cannot_be_assigned():
    _rejects("pipeline p() { input a: u8[4]\n output o: u8[4]\n"
             " b = load(a)\n c = store(o, b) }", "cannot be assigned")


def test_non_store_result_must_be_named():
    _rejects("pipeline p() { input a: u8[4]\n load(a) }",
             "must be assigned to a name")


def test_unpack_requires_bits():
    _rejects("pipeline p() { input a: u8[4]\n b = load(a)\n c = unpack(b) }",
             "requires attribute 'bits'")


def test_unpack_rejects_unknown_attr():
    _rejects("pipeline p() { input a: u8[4]\n b = load(a)\n"
             " c = unpack(b, width=4) }", "takes no attribute 'width'")


def test_reserved_word_as_tensor_name():
    _rejects("pipeline p() { input load: u8[4] }", "reserved")


def test_stray_character():
    _rejects("pipeline p() { input a: u8[4] $ }", "unexpected character '$'")


def test_trailing_garbage_after_program():
    _rejects("pipeline p() { input a: u8[4] } extra", "end of input")


def test_error_carries_line_number():
    src = "pipeline p() {\n  input a: u8[4]\n  b = scatter(a)\n}"
    with pytest.raises(TSCSyntaxError) as err:
        parse(src)
    assert err.value.line == 3


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
    print(f"\n{len(names) - failed}/{len(names)} tsc parser tests passed.")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    _run_all()
