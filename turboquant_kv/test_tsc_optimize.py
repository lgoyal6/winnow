"""Red/green tests for each optimization pass: the IR must actually change.

Every test asserts op counts or instruction text before AND after, so a pass
that silently becomes a no-op fails here rather than in a GPU run.
"""
from __future__ import annotations

from tsc.examples import example_source
from tsc.ir import dump_ir, lower
from tsc.optimize import (
    eliminate_redundant_casts, fuse_cast_into_store, fuse_unpack_gather,
    optimize, remove_dead_intermediates,
)
from tsc.parser import parse
from tsc.validate import validate


def _lowered(src: str = None, bw: int = 4):
    return lower(validate(parse(src if src is not None else example_source(bw))))


def _ops(module):
    return [i.op for i in module.instrs]


# -- the full pipeline on the canonical program ---------------------------------
def test_optimize_shrinks_canonical_ir_15_to_9():
    before = _lowered()
    after, stats = optimize(before)
    assert len(before.instrs) == 15
    assert len(after.instrs) == 9
    assert stats == {
        "instrs_before": 15,
        "eliminate_redundant_casts": 4,
        "fuse_unpack_gather": 1,
        "fuse_cast_into_store": 1,
        "remove_dead_intermediates": 0,
        "instrs_after": 9,
    }
    assert _ops(after) == ["load", "load", "load", "load", "decode", "rot",
                           "cast", "rescale", "store"]


def test_optimized_ir_matches_handwritten_kernel_structure():
    after, _ = optimize(_lowered())
    text = dump_ir(after)
    # one decode (unpack+gather in registers), one real cast (f16 norms to
    # f32), and the output conversion folded into the store
    assert "decode %cb, %codes {bits=4}" in text
    assert text.count(" cast ") == 1 and "cast %nrm, f32" in text
    assert "store %out, %z {cast=bf16}" in text


# -- eliminate_redundant_casts ---------------------------------------------------
def test_redundant_casts_are_removed_and_real_casts_kept():
    before = _lowered()
    after, removed = eliminate_redundant_casts(before)
    assert removed == 4
    casts_before = [i for i in before.instrs if i.op == "cast"]
    casts_after = [i for i in after.instrs if i.op == "cast"]
    assert len(casts_before) == 6 and len(casts_after) == 2
    kept = {(i.args[0], i.args[1]) for i in casts_after}
    assert kept == {("nrm", "f32"), ("z", "bf16")}   # f16->f32 and f32->bf16


def test_cast_users_are_rewired_to_the_original_value():
    after, _ = eliminate_redundant_casts(_lowered())
    rot = next(i for i in after.instrs if i.op == "rot")
    assert rot.args == ("y", "rotm")


# -- fuse_unpack_gather ----------------------------------------------------------
def test_unpack_gather_fusion_changes_canonical_ir():
    before = _lowered()
    after, fused = fuse_unpack_gather(before)
    assert fused == 1
    assert len(after.instrs) == len(before.instrs) - 1
    text = dump_ir(after)
    assert "unpack" not in text
    assert "%y: f32[N, 128] = decode %cb, %codes {bits=4}" in text


def test_fusion_is_not_applied_when_unpacked_indices_have_two_users():
    source = example_source(4).replace(
        "  y     = gather(cb, idx)",
        "  y2    = gather(cb, idx)\n  y     = gather(cb, idx)",
    )
    before = _lowered(source)
    after, fused = fuse_unpack_gather(before)
    assert fused == 0
    assert dump_ir(after) == dump_ir(before)


# -- fuse_cast_into_store --------------------------------------------------------
def test_final_cast_folds_into_store():
    before = _lowered()
    after, fused = fuse_cast_into_store(before)
    assert fused == 1
    store = next(i for i in after.instrs if i.op == "store")
    assert store.args == ("out", "z")
    assert dict(store.attrs) == {"cast": "bf16"}
    assert "o" not in {i.result.name for i in after.instrs if i.result}


def test_cast_with_second_user_is_not_folded_into_store():
    # `o` feeds both the store and another op, so the cast must survive.
    source = example_source(4).replace(
        "  store(out, o)",
        "  o2    = cast(o, f32)\n  store(out, o)",
    )
    before = _lowered(source)
    after, fused = fuse_cast_into_store(before)
    assert fused == 0


# -- remove_dead_intermediates ----------------------------------------------------
def test_dead_chain_is_removed_transitively():
    # dead2 depends on dead1; neither reaches the store. Both must go, and
    # remove_dead_intermediates alone must find them.
    source = example_source(4).replace(
        "  store(out, o)",
        "  dead1 = cast(nrm, f32)\n  dead2 = cast(dead1, bf16)\n  store(out, o)",
    )
    before = _lowered(source)
    after, removed = remove_dead_intermediates(before)
    assert removed == 2
    names = {i.result.name for i in after.instrs if i.result}
    assert "dead1" not in names and "dead2" not in names


def test_live_values_survive_dead_code_removal():
    after, removed = remove_dead_intermediates(_lowered())
    assert removed == 0
    assert len(after.instrs) == 15


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
    print(f"\n{len(names) - failed}/{len(names)} tsc optimization tests passed.")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    _run_all()
