"""Explicit linear IR: typed instructions over named single-assignment values.

Lowering normalizes the program into an explicit-conversion form: every
operand of a compute op (`rot`, `rescale`) is passed through an explicit
`cast` to f32, whether or not the source dtype already is f32. That mirrors
how production lowerings insert conversions unconditionally and leave cleanup
to a pass; on a valid program most of these casts are no-ops, and
`optimize.eliminate_redundant_casts` removes exactly those. The unoptimized
IR is still fully executable (interpreter and Triton backend both accept it),
so disabling the passes is a legitimate negative control, not a broken mode.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from tsc.syntax import (
    Assign, DTYPES, DtypeLit, ExprStmt, IntLit, Ref, TensorType,
)
from tsc.validate import ValidatedProgram


@dataclass(frozen=True)
class Value:
    name: str
    type: TensorType | None


@dataclass(frozen=True)
class Instr:
    op: str
    result: Value | None
    args: tuple[str, ...]
    attrs: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class Module:
    name: str
    inputs: tuple[Value, ...]
    outputs: tuple[Value, ...]
    instrs: tuple[Instr, ...]

    def value_types(self) -> dict[str, TensorType]:
        types = {v.name: v.type for v in self.inputs + self.outputs}
        for ins in self.instrs:
            if ins.result is not None:
                types[ins.result.name] = ins.result.type
        return types


def _arg(value: object) -> str:
    if isinstance(value, Ref):
        return value.name
    if isinstance(value, DtypeLit):
        return value.name
    if isinstance(value, IntLit):
        return str(value.value)
    raise TypeError(value)


# Ops whose operands are normalized to explicit f32 casts during lowering.
_NORMALIZED = {"rot", "rescale"}


def lower(valid: ValidatedProgram) -> Module:
    """Lower a validated AST into normalized, explicitly-typed instructions."""
    instrs: list[Instr] = []
    taken = set(valid.inputs) | set(valid.outputs) | set(valid.types)

    def fresh(base: str) -> str:
        name = f"{base}__f32"
        while name in taken:
            name += "_"
        taken.add(name)
        return name

    def normalized(name: str, t: TensorType) -> str:
        """Emit an explicit cast-to-f32 of `name` and return the new value."""
        out = fresh(name)
        instrs.append(Instr("cast", Value(out, TensorType("f32", t.shape)),
                            (name, "f32")))
        return out

    types = dict(valid.types)
    for stmt in valid.prog.stmts:
        call = stmt.call
        result = None
        if isinstance(stmt, Assign):
            result = Value(stmt.name, valid.types[stmt.name])
        elif not isinstance(stmt, ExprStmt):
            raise TypeError(stmt)
        args = tuple(_arg(a) for a in call.args)
        if call.op in _NORMALIZED:
            all_types = {**valid.inputs, **valid.outputs, **types}
            args = tuple(normalized(a, all_types[a]) for a in args)
        attrs = tuple((key, valid.resolve_int(value, call))
                      for key, value in call.attrs)
        instrs.append(Instr(call.op, result, args, attrs))
    return Module(
        valid.prog.name,
        tuple(Value(name, typ) for name, typ in valid.inputs.items()),
        tuple(Value(name, typ) for name, typ in valid.outputs.items()),
        tuple(instrs),
    )


def substitute(module: Module, mapping: dict[str, str]) -> Module:
    """Rewrite every argument through `mapping` (used by alias-removal passes)."""
    def resolve(name: str) -> str:
        seen = set()
        while name in mapping:
            if name in seen:                       # pragma: no cover
                raise ValueError(f"alias cycle at {name!r}")
            seen.add(name)
            name = mapping[name]
        return name

    instrs = tuple(
        replace(ins, args=tuple(resolve(a) for a in ins.args))
        for ins in module.instrs
    )
    return replace(module, instrs=instrs)


def dump_ir(module: Module) -> str:
    lines = [f"module {module.name}"]
    for value in module.inputs:
        lines.append(f"  input %{value.name}: {value.type}")
    for value in module.outputs:
        lines.append(f"  output %{value.name}: {value.type}")
    for ins in module.instrs:
        lhs = f"%{ins.result.name}: {ins.result.type} = " if ins.result else ""
        attrs = "" if not ins.attrs else " {" + ", ".join(
            f"{k}={v}" for k, v in ins.attrs) + "}"
        lines.append(f"  {lhs}{ins.op} " + ", ".join(
            a if a in DTYPES else f"%{a}" for a in ins.args) + attrs)
    return "\n".join(lines) + "\n"
