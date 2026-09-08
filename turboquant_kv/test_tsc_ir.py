from tsc.examples import example_source
from tsc.ir import dump_ir, fuse_unpack_gather, lower
from tsc.parser import parse
from tsc.validate import validate


def test_ast_lowers_to_explicit_ir():
    ir = lower(validate(parse(example_source(4))))
    text = dump_ir(ir)
    assert "%idx: u8[N, 128] = unpack %codes {bits=4}" in text
    assert "%y: f32[N, 128] = gather %cb, %idx" in text
    assert text.endswith("  store %out, %o\n")


def test_unpack_gather_fusion_changes_canonical_ir():
    before = lower(validate(parse(example_source(4))))
    after = fuse_unpack_gather(before)
    assert len(after.instrs) == len(before.instrs) - 1
    text = dump_ir(after)
    assert "unpack" not in text
    assert "%y: f32[N, 128] = decode %cb, %codes {bits=4}" in text


def test_fusion_is_not_applied_when_unpacked_indices_have_two_users():
    source = example_source(4).replace(
        "  y     = gather(cb, idx)",
        "  y2    = gather(cb, idx)\n  y     = gather(cb, idx)",
    )
    before = lower(validate(parse(source)))
    after = fuse_unpack_gather(before)
    assert dump_ir(after) == dump_ir(before)
