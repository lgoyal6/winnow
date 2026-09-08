"""Typed AST for the tensor-schedule DSL.

The DSL is a straight-line, single-assignment description of the TurboQuant
KV-cache dequantization pipeline. A program declares typed inputs and outputs,
then a sequence of ops: load, unpack, gather, rot, rescale, cast, store.

Dims are either literal ints, param references (resolved during validation),
or symbolic names such as N that stay symbolic through the whole pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field


DTYPES = ("u8", "f16", "bf16", "f32")

# Every op the grammar admits, with its arity (positional args).
OPS = {
    "load": 1,       # load(input_tensor)
    "unpack": 1,     # unpack(codes, bits=K)
    "gather": 2,     # gather(table, idx)
    "rot": 2,        # rot(x, matrix)
    "rescale": 2,    # rescale(x, scale)
    "cast": 2,       # cast(x, dtype)
    "store": 2,      # store(output_tensor, value)
}

OP_ATTRS = {"unpack": ("bits",)}   # keyword attrs each op accepts


# Dim = int | str (symbolic or param name); resolved dims are int | str(symbol)
@dataclass(frozen=True)
class TensorType:
    dtype: str
    shape: tuple[object, ...]      # elements are int or str

    def __str__(self) -> str:
        dims = ", ".join(str(d) for d in self.shape)
        return f"{self.dtype}[{dims}]"


@dataclass(frozen=True)
class Ref:
    name: str
    line: int = 0

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class IntLit:
    value: int
    line: int = 0

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class DtypeLit:
    name: str
    line: int = 0

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Param:
    name: str
    value: int
    line: int = 0


@dataclass(frozen=True)
class TensorDecl:
    kind: str                      # "input" | "output"
    name: str
    type: TensorType
    line: int = 0


@dataclass(frozen=True)
class OpCall:
    op: str
    args: tuple[object, ...]       # Ref | IntLit | DtypeLit
    attrs: tuple[tuple[str, object], ...] = ()   # ((name, Ref|IntLit), ...)
    line: int = 0

    def __str__(self) -> str:
        parts = [str(a) for a in self.args]
        parts += [f"{k}={v}" for k, v in self.attrs]
        return f"{self.op}({', '.join(parts)})"


@dataclass(frozen=True)
class Assign:
    name: str
    call: OpCall
    line: int = 0


@dataclass(frozen=True)
class ExprStmt:
    call: OpCall
    line: int = 0


@dataclass(frozen=True)
class Program:
    name: str
    params: tuple[Param, ...]
    decls: tuple[TensorDecl, ...]
    stmts: tuple[object, ...]      # Assign | ExprStmt
    types: dict = field(default_factory=dict, compare=False)
    # `types` is filled by validate(): name -> resolved TensorType for every
    # input, output, and intermediate. Empty on a freshly parsed program.


def dump_ast(prog: Program) -> str:
    """Stable text form of the AST, used by the parser golden tests."""
    lines = [f"Program {prog.name}"]
    if prog.params:
        lines.append("  params: " + ", ".join(
            f"{p.name}={p.value}" for p in prog.params))
    for d in prog.decls:
        lines.append(f"  {d.kind} {d.name}: {d.type}")
    lines.append("  stmts:")
    for s in prog.stmts:
        if isinstance(s, Assign):
            lines.append(f"    {s.name} = {s.call}")
        else:
            lines.append(f"    {s.call}")
    return "\n".join(lines) + "\n"
