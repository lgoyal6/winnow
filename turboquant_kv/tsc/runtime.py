"""Execution dispatch for compiled schedules, with a hard fallback rule.

The generated Triton kernel runs only when a CUDA device and a working triton
import are both present. Anything else (this includes Apple Silicon, CPU-only
hosts, and CUDA machines without triton) falls back to the pure-PyTorch
reference interpreter and says so in the report. The generated kernel is NOT
wired into TQPackedLayer or the dispatcher: the production path is untouched
and this module is only reachable from tests and the differential harness.
"""
from __future__ import annotations

import types
from dataclasses import dataclass

import torch

from tsc.compiler import CompileResult, compile_source
from tsc.ir import Module, lower
from tsc.optimize import optimize
from tsc.parser import parse
from tsc.reference import run_reference
from tsc.validate import validate


@dataclass(frozen=True)
class ExecutionReport:
    backend: str          # "generated_triton" | "reference_torch"
    reason: str


def _triton_available() -> bool:
    try:
        import triton                                        # noqa: F401
        return True
    except Exception:
        return False


def load_generated(result: CompileResult) -> types.ModuleType:
    """exec() the generated source into a fresh module object."""
    mod = types.ModuleType(f"tsc_generated_{result.key[:12]}")
    exec(compile(result.generated, f"<tsc:{result.key[:12]}>", "exec"),
         mod.__dict__)
    return mod


def _module_for(source: str, optimize_ir: bool) -> Module:
    module = lower(validate(parse(source)))
    if optimize_ir:
        module, _ = optimize(module)
    return module


def execute(source: str, inputs: dict[str, torch.Tensor], *,
            optimize_ir: bool = True, allow_tf32: bool = False,
            cache_dir=None) -> tuple[torch.Tensor, ExecutionReport]:
    """Compile and run `source` on `inputs`, generated kernel if possible.

    Returns the output tensor and a report naming the backend that actually
    executed, so a caller (or a test) can assert the fallback fired.
    """
    result = compile_source(source, optimize_ir=optimize_ir,
                            cache_dir=cache_dir)
    module = _module_for(source, optimize_ir)

    on_cuda = any(t.is_cuda for t in inputs.values())
    if torch.cuda.is_available() and on_cuda and _triton_available():
        gen = load_generated(result)
        ordered = [inputs[v.name] for v in module.inputs]
        out = gen.run(*ordered, allow_tf32=allow_tf32)
        return out, ExecutionReport(
            "generated_triton",
            f"CUDA device and triton present; ran {module.name}_kernel")

    if not torch.cuda.is_available():
        reason = "no CUDA device; ran the PyTorch reference interpreter"
    elif not on_cuda:
        reason = "inputs are not CUDA tensors; ran the PyTorch reference interpreter"
    else:
        reason = "triton unavailable; ran the PyTorch reference interpreter"
    out = run_reference(module, inputs)
    return out, ExecutionReport("reference_torch", reason)
