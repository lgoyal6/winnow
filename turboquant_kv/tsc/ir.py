"""Explicit linear IR and the unpack+gather fusion pass."""
from __future__ import annotations

from dataclasses import dataclass

from tsc.syntax import Assign, DtypeLit, ExprStmt, IntLit, Ref, TensorType
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


def _arg(value: object) -> str:
    if isinstance(value, Ref):
        return value.name
    if isinstance(value, DtypeLit):
        return value.name
    if isinstance(value, IntLit):
        return str(value.value)
    raise TypeError(value)


def lower(valid: ValidatedProgram) -> Module:
    """Lower a validated AST without retaining parser-only node structure."""
    instrs = []
    for stmt in valid.prog.stmts:
        call = stmt.call
        result = None
        if isinstance(stmt, Assign):
            result = Value(stmt.name, valid.types[stmt.name])
        elif not isinstance(stmt, ExprStmt):
            raise TypeError(stmt)
        attrs = []
        for key, value in call.attrs:
            attrs.append((key, valid.resolve_int(value, call)))
        instrs.append(Instr(call.op, result, tuple(_arg(a) for a in call.args), tuple(attrs)))
    return Module(
        valid.prog.name,
        tuple(Value(name, typ) for name, typ in valid.inputs.items()),
        tuple(Value(name, typ) for name, typ in valid.outputs.items()),
        tuple(instrs),
    )


def dump_ir(module: Module) -> str:
    lines = [f"module {module.name}"]
    for value in module.inputs:
        lines.append(f"  input %{value.name}: {value.type}")
    for value in module.outputs:
        lines.append(f"  output %{value.name}: {value.type}")
    for ins in module.instrs:
        lhs = f"%{ins.result.name}: {ins.result.type} = " if ins.result else ""
        attrs = "" if not ins.attrs else " {" + ", ".join(f"{k}={v}" for k, v in ins.attrs) + "}"
        lines.append(f"  {lhs}{ins.op} " + ", ".join(f"%{a}" for a in ins.args) + attrs)
    return "\n".join(lines) + "\n"
