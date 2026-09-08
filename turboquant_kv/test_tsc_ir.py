from tsc.examples import example_source
from tsc.ir import dump_ir, lower
from tsc.parser import parse
from tsc.validate import validate


def test_ast_lowers_to_explicit_ir():
    ir = lower(validate(parse(example_source(4))))
    text = dump_ir(ir)
    assert "%idx: u8[N, 128] = unpack %codes {bits=4}" in text
    assert "%y: f32[N, 128] = gather %cb, %idx" in text
    assert text.endswith("  store %out, %o\n")
