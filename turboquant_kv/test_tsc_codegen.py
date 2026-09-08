import pytest

from tsc.codegen import generate_triton
from tsc.examples import example_source
from tsc.ir import fuse_unpack_gather, lower
from tsc.parser import parse
from tsc.validate import validate


def _module():
    return fuse_unpack_gather(lower(validate(parse(example_source(4)))))


def test_codegen_is_standalone_executable_python():
    source = generate_triton(_module())
    compile(source, "generated_tq_dequant.py", "exec")
    assert "@triton.jit" in source
    assert "tl.dot" in source
    assert "NB=64, BW=4" in source


def test_codegen_rejects_mutated_operation_order():
    raw = example_source(4).replace(
        "  x     = rot(y, rotm)\n  nf    = cast(nrm, f32)",
        "  nf    = cast(nrm, f32)\n  x     = rot(y, rotm)",
    )
    module = fuse_unpack_gather(lower(validate(parse(raw))))
    with pytest.raises(ValueError, match="unsupported IR order"):
        generate_triton(module)


def test_fault_injection_changes_generated_indexing():
    source = generate_triton(_module(), indexing_delta=1)
    assert "base + byte + 1" in source
    assert source != generate_triton(_module())
