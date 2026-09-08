"""Validation tests: shape, dtype, rank, bounds, and operation-order rules.

Each rejected fixture is the canonical bw4 pipeline with exactly one line
mutated, so a failure names the rule that caught it rather than a cascade.
"""
from __future__ import annotations

import pytest

from tsc.errors import TSCValidationError
from tsc.examples import example_source
from tsc.parser import parse
from tsc.syntax import TensorType
from tsc.validate import validate


def _validated(bw: int = 4):
    return validate(parse(example_source(bw)))


def _rejects(src: str, needle: str):
    with pytest.raises(TSCValidationError) as err:
        validate(parse(src))
    assert needle in str(err.value), str(err.value)


def _mutate(old: str, new: str, bw: int = 4) -> str:
    src = example_source(bw)
    assert old in src, f"fixture drifted: {old!r} not found"
    return src.replace(old, new)


# -- the canonical program validates and is fully typed ------------------------
def test_canonical_program_validates():
    v = _validated()
    assert v.inputs["packed"] == TensorType("u8", ("N", 64))
    assert v.outputs["out"] == TensorType("bf16", ("N", 128))
    assert v.types["idx"] == TensorType("u8", ("N", 128))
    assert v.types["y"] == TensorType("f32", ("N", 128))
    assert v.types["nf"] == TensorType("f32", ("N",))
    assert v.types["o"] == TensorType("bf16", ("N", 128))
    assert v.unpack_bits["idx"] == 4
    assert v.loaded_from == {"codes": "packed", "cb": "centroids",
                             "rotm": "pi", "nrm": "norms"}


def test_bw8_program_validates():
    v = _validated(8)
    assert v.inputs["packed"] == TensorType("u8", ("N", 128))
    assert v.unpack_bits["idx"] == 8


# -- operation order -----------------------------------------------------------
def test_gather_before_unpack_rejected():
    # Swap the gather and unpack lines: gather now uses `idx` before it exists.
    src = _mutate(
        "  idx   = unpack(codes, bits=bit_width)\n  y     = gather(cb, idx)",
        "  y     = gather(cb, idx)\n  idx   = unpack(codes, bits=bit_width)")
    _rejects(src, "'idx' is not defined at this point")


def test_rescale_of_unloaded_tensor_rejected():
    # Use the raw `norms` input directly instead of the loaded+cast value.
    src = _mutate("z     = rescale(x, nf)", "z     = rescale(x, norms)")
    _rejects(src, "input 'norms' must be loaded before use")


def test_unpack_of_unloaded_input_rejected():
    src = _mutate("idx   = unpack(codes, bits=bit_width)",
                  "idx   = unpack(packed, bits=bit_width)")
    _rejects(src, "input 'packed' must be loaded before use")


def test_unpack_of_non_load_result_rejected():
    src = _mutate("y     = gather(cb, idx)",
                  "idx2  = unpack(idx, bits=bit_width)\n  y     = gather(cb, idx2)")
    _rejects(src, "not the result of a `load`")


def test_single_assignment_enforced():
    src = _mutate("nf    = cast(nrm, f32)",
                  "nf    = cast(nrm, f32)\n  nf    = cast(nrm, f32)")
    _rejects(src, "already defined")


def test_store_must_target_output():
    src = _mutate("store(out, o)", "store(packed, o)")
    _rejects(src, "not a declared output")


def test_output_never_stored_rejected():
    src = _mutate("output out:       bf16[N, head_dim]",
                  "output out:       bf16[N, head_dim]\n"
                  "  output out2:      bf16[N, head_dim]")
    _rejects(src, "output 'out2' is never stored")


def test_double_store_rejected():
    src = _mutate("store(out, o)", "store(out, o)\n  store(out, o)")
    _rejects(src, "stored twice")


def test_load_of_intermediate_rejected():
    src = _mutate("idx   = unpack(codes, bits=bit_width)",
                  "idx   = unpack(codes, bits=bit_width)\n  bad   = load(idx)")
    _rejects(src, "'idx' is an intermediate")


def test_undeclared_input_rejected():
    src = _mutate("codes = load(packed)", "codes = load(mystery)")
    _rejects(src, "'mystery' is not a declared input")


# -- dtype rules ----------------------------------------------------------------
def test_unpack_needs_u8():
    src = _mutate("input  packed:    u8[N, 64]", "input  packed:    f16[N, 64]")
    _rejects(src, "`unpack` needs u8 packed codes")


def test_gather_table_must_be_f32():
    src = _mutate("input  centroids: f32[16]", "input  centroids: f16[16]")
    _rejects(src, "rank-1 f32 codebook")


def test_rot_operands_must_be_f32():
    src = _mutate("input  pi:        f32[head_dim, head_dim]",
                  "input  pi:        bf16[head_dim, head_dim]")
    _rejects(src, "`rot` operands must be f32")


def test_rescale_scale_must_be_f32():
    src = _mutate("nf    = cast(nrm, f32)", "nf    = cast(nrm, f16)")
    _rejects(src, "`rescale` operands must be f32")


def test_store_dtype_mismatch_rejected():
    src = _mutate("o     = cast(z, bf16)", "o     = cast(z, f16)")
    _rejects(src, "value is f16 but output 'out' is bf16")


# -- shape and rank rules --------------------------------------------------------
def test_gather_table_rank_checked():
    src = _mutate("input  centroids: f32[16]", "input  centroids: f32[16, 2]")
    _rejects(src, "rank-1 f32 codebook")


def test_rot_matrix_must_be_square():
    src = _mutate("input  pi:        f32[head_dim, head_dim]",
                  "input  pi:        f32[head_dim, 64]")
    _rejects(src, "square rank-2")


def test_rot_inner_dim_mismatch():
    src = _mutate("input  pi:        f32[head_dim, head_dim]",
                  "input  pi:        f32[64, 64]")
    _rejects(src, "inner dims disagree")


def test_rescale_leading_dims_must_match():
    # A scale tied to a different symbolic dim than the value's rows.
    src = _mutate("input  norms:     f16[N]", "input  norms:     f16[M]")
    _rejects(src, "leading dims")


def test_store_shape_mismatch_rejected():
    src = _mutate("output out:       bf16[N, head_dim]",
                  "output out:       bf16[N, 64]")
    _rejects(src, "`store` shape mismatch")


# -- bounds -----------------------------------------------------------------------
def test_unsupported_bit_width_rejected():
    src = example_source(4).replace("bit_width=4", "bit_width=7") \
                           .replace("u8[N, 64]", "u8[N, 112]") \
                           .replace("f32[16]", "f32[128]")
    _rejects(src, "bits=7 unsupported")


def test_partial_index_bytes_rejected():
    # 64 bytes hold 512 bits, and 512 is not divisible by 5.
    src = example_source(4).replace("bit_width=4", "bit_width=5") \
                           .replace("f32[16]", "f32[32]")
    _rejects(src, "whole number of 5-bit indices")


def test_codebook_size_must_match_bits():
    src = _mutate("input  centroids: f32[16]", "input  centroids: f32[17]")
    _rejects(src, "codebook has 17 entries but the indices are 4-bit")


def test_zero_dimension_rejected():
    src = _mutate("input  centroids: f32[16]", "input  centroids: f32[0]")
    _rejects(src, "dimension must be positive")


def test_unknown_param_in_attr_rejected():
    src = _mutate("bits=bit_width", "bits=bitwidth")
    _rejects(src, "unknown parameter 'bitwidth'")


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
    print(f"\n{len(names) - failed}/{len(names)} tsc validation tests passed.")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    _run_all()
