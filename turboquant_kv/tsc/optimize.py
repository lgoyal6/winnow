"""IR optimization passes, each individually testable and disableable.

    eliminate_redundant_casts   cast whose target dtype equals its operand's
                                dtype becomes an alias and disappears
    fuse_unpack_gather          single-use unpack feeding gather fuses into
                                one `decode` op (bit extraction + codebook
                                lookup in registers, as the handwritten
                                kernel does)
    fuse_cast_into_store        single-use cast feeding a store whose target
                                dtype is the output's dtype folds into the
                                store (the `.to(out.dtype)` in the store)
    remove_dead_intermediates   results never consumed and never stored are
                                dropped, transitively

`optimize()` runs them in that order and reports what each pass did, so the
red/green tests can assert the IR actually changed rather than trusting a
flag.
"""
from __future__ import annotations

from dataclasses import replace

from tsc.ir import Instr, Module, substitute


def eliminate_redundant_casts(module: Module) -> tuple[Module, int]:
    """Remove `cast` instructions whose operand already has the target dtype."""
    types = module.value_types()
    aliases: dict[str, str] = {}
    kept: list[Instr] = []
    for ins in module.instrs:
        if (ins.op == "cast" and ins.result is not None
                and types.get(ins.args[0]) is not None
                and types[ins.args[0]].dtype == ins.args[1]):
            aliases[ins.result.name] = ins.args[0]
        else:
            kept.append(ins)
    if not aliases:
        return module, 0
    return substitute(replace(module, instrs=tuple(kept)), aliases), len(aliases)


def _use_counts(module: Module) -> dict[str, int]:
    uses: dict[str, int] = {}
    for ins in module.instrs:
        for arg in ins.args:
            uses[arg] = uses.get(arg, 0) + 1
    return uses


def fuse_unpack_gather(module: Module) -> tuple[Module, int]:
    """Fuse a single-use unpack feeding gather into one decode instruction."""
    uses = _use_counts(module)
    unpack_by_result = {
        ins.result.name: ins
        for ins in module.instrs
        if ins.op == "unpack" and ins.result is not None
        and uses.get(ins.result.name) == 1
    }
    removed: set[str] = set()
    fused = 0
    optimized: list[Instr] = []
    for ins in module.instrs:
        if (ins.op == "gather" and len(ins.args) == 2
                and ins.args[1] in unpack_by_result):
            unpack = unpack_by_result[ins.args[1]]
            optimized.append(replace(ins, op="decode",
                                     args=(ins.args[0], unpack.args[0]),
                                     attrs=unpack.attrs))
            removed.add(unpack.result.name)
            fused += 1
        else:
            optimized.append(ins)
    optimized = [ins for ins in optimized
                 if not (ins.op == "unpack" and ins.result
                         and ins.result.name in removed)]
    return replace(module, instrs=tuple(optimized)), fused


def fuse_cast_into_store(module: Module) -> tuple[Module, int]:
    """Fold a store's producing cast into the store itself.

    `o = cast(z, bf16); store(out, o)` becomes `store out, z {cast=bf16}`,
    which the Triton backend emits as `.to(out.dtype.element_ty)` inside the
    `tl.store`, exactly as the handwritten kernel writes its output.
    """
    uses = _use_counts(module)
    outputs = {v.name for v in module.outputs}
    cast_by_result = {
        ins.result.name: ins
        for ins in module.instrs
        if ins.op == "cast" and ins.result is not None
        and uses.get(ins.result.name) == 1
    }
    removed: set[str] = set()
    fused = 0
    optimized: list[Instr] = []
    for ins in module.instrs:
        if (ins.op == "store" and len(ins.args) == 2
                and ins.args[0] in outputs
                and ins.args[1] in cast_by_result):
            cast = cast_by_result[ins.args[1]]
            optimized.append(replace(
                ins, args=(ins.args[0], cast.args[0]),
                attrs=ins.attrs + (("cast", cast.args[1]),)))
            removed.add(cast.result.name)
            fused += 1
        else:
            optimized.append(ins)
    optimized = [ins for ins in optimized
                 if not (ins.op == "cast" and ins.result
                         and ins.result.name in removed)]
    return replace(module, instrs=tuple(optimized)), fused


def remove_dead_intermediates(module: Module) -> tuple[Module, int]:
    """Drop instructions whose results are never consumed, transitively."""
    removed_total = 0
    while True:
        uses = _use_counts(module)
        kept = [ins for ins in module.instrs
                if ins.result is None or uses.get(ins.result.name, 0) > 0]
        removed = len(module.instrs) - len(kept)
        if removed == 0:
            return module, removed_total
        removed_total += removed
        module = replace(module, instrs=tuple(kept))


PASSES = (
    eliminate_redundant_casts,
    fuse_unpack_gather,
    fuse_cast_into_store,
    remove_dead_intermediates,
)


def optimize(module: Module) -> tuple[Module, dict]:
    """Run every pass once, in order. Returns the module and per-pass stats."""
    stats = {"instrs_before": len(module.instrs)}
    for tick in PASSES:
        module, count = tick(module)
        stats[tick.__name__] = count
    stats["instrs_after"] = len(module.instrs)
    return module, stats
