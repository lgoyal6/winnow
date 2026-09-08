"""Semantic validation: types, shapes, ranks, bounds, and operation order.

Rules enforced here, each with a test in test_tsc_validate.py:

order / dataflow
  - every name is defined before use (so `gather` before `unpack` is rejected)
  - inputs may only be consumed by `load` (so `rescale` of an unloaded tensor
    is rejected)
  - `load` takes an input, `store` writes an output, single assignment,
    every output stored exactly once
types / shapes / ranks
  - per-op dtype signatures (unpack wants u8 codes, rot wants f32 operands, ...)
  - shape agreement, with symbolic dims (N) unified by name
  - store value must match the declared output type exactly
bounds
  - bits in the packing module's supported set, packed bytes holding a whole
    number of indices, gather table sized exactly 2**bits

Validation resolves param references and returns a ValidatedProgram carrying
the resolved type of every tensor plus the dataflow facts lowering needs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from tsc.errors import TSCValidationError
from tsc.syntax import (
    Assign, DTYPES, DtypeLit, ExprStmt, IntLit, OpCall, Program, Ref,
    TensorType,
)

# Keep this in lockstep with packing.SUPPORTED without importing torch here.
SUPPORTED_BITS = (2, 3, 4, 5, 6, 8)


def _err(msg: str, node) -> TSCValidationError:
    return TSCValidationError(msg, getattr(node, "line", None))


@dataclass
class ValidatedProgram:
    prog: Program
    params: dict[str, int]
    inputs: dict[str, TensorType]
    outputs: dict[str, TensorType]
    types: dict[str, TensorType]        # intermediates
    loaded_from: dict[str, str] = field(default_factory=dict)
    unpack_bits: dict[str, int] = field(default_factory=dict)
    call_bits: dict[int, int] = field(default_factory=dict)  # id(call) -> bits


class _Ctx(ValidatedProgram):
    def __init__(self, prog: Program):
        super().__init__(prog, {p.name: p.value for p in prog.params},
                         {}, {}, {})
        self.produced_by: dict[str, OpCall] = {}
        self.stored: set[str] = set()

    # -- resolution helpers -------------------------------------------------
    def resolve_dim(self, dim, node):
        if isinstance(dim, int):
            if dim <= 0:
                raise _err(f"dimension must be positive, got {dim}", node)
            return dim
        if dim in self.params:
            return self.params[dim]
        return dim                     # symbolic (e.g. N)

    def resolve_type(self, t: TensorType, node) -> TensorType:
        return TensorType(t.dtype, tuple(self.resolve_dim(d, node)
                                         for d in t.shape))

    def resolve_int(self, v, node) -> int:
        if isinstance(v, IntLit):
            return v.value
        if isinstance(v, Ref):
            if v.name not in self.params:
                raise _err(f"unknown parameter {v.name!r}", node)
            return self.params[v.name]
        raise _err("expected an integer or parameter reference", node)

    def type_of(self, name: str, node) -> TensorType:
        if name in self.types:
            return self.types[name]
        if name in self.inputs:
            raise _err(
                f"input {name!r} must be loaded before use; only `load` may "
                f"read an input directly", node)
        if name in self.outputs:
            raise _err(f"output {name!r} can only be written by `store`", node)
        raise _err(f"{name!r} is not defined at this point", node)

    def ref(self, arg, node) -> str:
        if not isinstance(arg, Ref):
            raise _err(f"expected a tensor name, got {arg}", node)
        return arg.name


def validate(prog: Program) -> ValidatedProgram:
    """Check the program; raises TSCValidationError on the first violation."""
    ctx = _Ctx(prog)

    for p in prog.params:
        if p.value <= 0:
            raise _err(f"parameter {p.name!r} must be positive", p)

    for d in prog.decls:
        if d.name in ctx.inputs or d.name in ctx.outputs:
            raise _err(f"tensor {d.name!r} declared twice", d)
        table = ctx.inputs if d.kind == "input" else ctx.outputs
        table[d.name] = ctx.resolve_type(d.type, d)

    if not ctx.outputs:
        raise _err("program declares no output",
                   prog.decls[-1] if prog.decls else prog)

    for stmt in prog.stmts:
        if isinstance(stmt, Assign):
            if (stmt.name in ctx.types or stmt.name in ctx.inputs
                    or stmt.name in ctx.outputs):
                raise _err(f"{stmt.name!r} is already defined; the language "
                           f"is single-assignment", stmt)
            result = _check_call(ctx, stmt.call, stmt.name)
            ctx.types[stmt.name] = result
            ctx.produced_by[stmt.name] = stmt.call
        elif isinstance(stmt, ExprStmt):
            _check_call(ctx, stmt.call, None)
        else:                                       # pragma: no cover
            raise _err(f"unknown statement {stmt!r}", stmt)

    missing = sorted(set(ctx.outputs) - ctx.stored)
    if missing:
        raise _err(f"output {missing[0]!r} is never stored",
                   prog.stmts[-1] if prog.stmts else prog)
    return ctx


def _check_call(ctx: _Ctx, call: OpCall, target: str | None):
    return _CHECKERS[call.op](ctx, call, target)


# -- per-op rules --------------------------------------------------------------
def _op_load(ctx: _Ctx, call: OpCall, target) -> TensorType:
    name = ctx.ref(call.args[0], call)
    if name in ctx.types:
        raise _err(f"`load` reads inputs, but {name!r} is an intermediate",
                   call)
    if name in ctx.outputs:
        raise _err(f"`load` reads inputs, but {name!r} is an output", call)
    if name not in ctx.inputs:
        raise _err(f"{name!r} is not a declared input", call)
    if target is not None:
        ctx.loaded_from[target] = name
    return ctx.inputs[name]


def _op_unpack(ctx: _Ctx, call: OpCall, target) -> TensorType:
    src = ctx.ref(call.args[0], call)
    t = ctx.type_of(src, call)
    if t.dtype != "u8":
        raise _err(f"`unpack` needs u8 packed codes, got {t}", call)
    if len(t.shape) < 1 or not isinstance(t.shape[-1], int):
        raise _err("`unpack` needs a known last (byte) dimension", call)
    producer = ctx.produced_by.get(src)
    if producer is None or producer.op != "load":
        raise _err(f"`unpack` consumes loaded codes; {src!r} is not the "
                   f"result of a `load`", call)
    bits = ctx.resolve_int(dict(call.attrs)["bits"], call)
    if bits not in SUPPORTED_BITS:
        raise _err(f"bits={bits} unsupported; packing supports "
                   f"{SUPPORTED_BITS}", call)
    nb = t.shape[-1]
    if (nb * 8) % bits != 0:
        raise _err(f"{nb} packed bytes do not hold a whole number of "
                   f"{bits}-bit indices", call)
    ctx.call_bits[id(call)] = bits
    if target is not None:
        ctx.unpack_bits[target] = bits
    return TensorType("u8", t.shape[:-1] + (nb * 8 // bits,))


def _op_gather(ctx: _Ctx, call: OpCall, target) -> TensorType:
    table_name = ctx.ref(call.args[0], call)
    idx_name = ctx.ref(call.args[1], call)
    table = ctx.type_of(table_name, call)
    idx = ctx.type_of(idx_name, call)
    if table.dtype != "f32" or len(table.shape) != 1:
        raise _err(f"`gather` table must be a rank-1 f32 codebook, got "
                   f"{table}", call)
    if idx.dtype != "u8":
        raise _err(f"`gather` indices must be u8, got {idx}", call)
    bits = ctx.unpack_bits.get(idx_name)
    if bits is not None and isinstance(table.shape[0], int):
        if table.shape[0] != 2 ** bits:
            raise _err(
                f"codebook has {table.shape[0]} entries but the indices are "
                f"{bits}-bit; expected {2 ** bits}", call)
    return TensorType("f32", idx.shape)


def _op_rot(ctx: _Ctx, call: OpCall, target) -> TensorType:
    x = ctx.type_of(ctx.ref(call.args[0], call), call)
    m = ctx.type_of(ctx.ref(call.args[1], call), call)
    if x.dtype != "f32" or m.dtype != "f32":
        raise _err(f"`rot` operands must be f32, got {x} and {m}", call)
    if len(m.shape) != 2 or m.shape[0] != m.shape[1]:
        raise _err(f"`rot` matrix must be square rank-2, got {m}", call)
    if len(x.shape) < 1 or x.shape[-1] != m.shape[0]:
        raise _err(f"`rot` inner dims disagree: {x} vs {m}", call)
    return x


def _op_rescale(ctx: _Ctx, call: OpCall, target) -> TensorType:
    x = ctx.type_of(ctx.ref(call.args[0], call), call)
    s = ctx.type_of(ctx.ref(call.args[1], call), call)
    if x.dtype != "f32" or s.dtype != "f32":
        raise _err(f"`rescale` operands must be f32, got {x} and {s}", call)
    if len(s.shape) != len(x.shape) - 1 or s.shape != x.shape[:-1]:
        raise _err(f"`rescale` scale must match the value's leading dims: "
                   f"{x} vs {s}", call)
    return x


def _op_cast(ctx: _Ctx, call: OpCall, target) -> TensorType:
    x = ctx.type_of(ctx.ref(call.args[0], call), call)
    dt = call.args[1]
    if not isinstance(dt, DtypeLit) or dt.name not in DTYPES:
        raise _err(f"`cast` target must be one of {', '.join(DTYPES)}", call)
    return TensorType(dt.name, x.shape)


def _op_store(ctx: _Ctx, call: OpCall, target) -> None:
    dst = ctx.ref(call.args[0], call)
    src = ctx.ref(call.args[1], call)
    if dst not in ctx.outputs:
        raise _err(f"`store` writes outputs, but {dst!r} is not a declared "
                   f"output", call)
    if dst in ctx.stored:
        raise _err(f"output {dst!r} is stored twice", call)
    val = ctx.type_of(src, call)
    want = ctx.outputs[dst]
    if val.dtype != want.dtype:
        raise _err(f"`store` value is {val.dtype} but output {dst!r} is "
                   f"{want.dtype}; cast explicitly", call)
    if val.shape != want.shape:
        raise _err(f"`store` shape mismatch: value {val} vs output {want}",
                   call)
    ctx.stored.add(dst)
    return None


_CHECKERS = {
    "load": _op_load,
    "unpack": _op_unpack,
    "gather": _op_gather,
    "rot": _op_rot,
    "rescale": _op_rescale,
    "cast": _op_cast,
    "store": _op_store,
}
