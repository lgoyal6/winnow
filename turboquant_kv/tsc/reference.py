"""Pure-PyTorch interpreter for the IR: the reference the kernel is judged by.

Executes any valid module, optimized or not, one instruction at a time with
plain torch ops (`packing.unpack` for bit extraction). This is both the
numerical reference for the GPU differential harness and the execution
fallback on hosts with no supported GPU. It runs everything in fp32 and casts
exactly where the IR says to, so a generated fp32 kernel is compared against
semantics, not against another kernel.
"""
from __future__ import annotations

import torch

from packing import unpack
from tsc.errors import TSCError
from tsc.ir import Module

_TORCH_DTYPE = {"u8": torch.uint8, "f16": torch.float16,
                "bf16": torch.bfloat16, "f32": torch.float32}


def run_reference(module: Module, inputs: dict[str, torch.Tensor],
                  out: torch.Tensor | None = None) -> torch.Tensor:
    """Execute `module` on `inputs` (keyed by declared input names)."""
    missing = [v.name for v in module.inputs if v.name not in inputs]
    if missing:
        raise TSCError(f"missing inputs: {missing}")
    if len(module.outputs) != 1:
        raise TSCError("reference interpreter supports exactly one output")
    out_val = module.outputs[0]

    env: dict[str, torch.Tensor] = {}
    result: torch.Tensor | None = None

    for ins in module.instrs:
        if ins.op == "load":
            env[ins.result.name] = inputs[ins.args[0]]
        elif ins.op == "unpack":
            bits = dict(ins.attrs)["bits"]
            d = ins.result.type.shape[-1]
            env[ins.result.name] = unpack(env[ins.args[0]].contiguous(),
                                          bits, d)
        elif ins.op == "gather":
            table, idx = env[ins.args[0]], env[ins.args[1]]
            flat = idx.reshape(-1).to(torch.int32)
            env[ins.result.name] = table.index_select(0, flat).view(idx.shape)
        elif ins.op == "decode":
            bits = dict(ins.attrs)["bits"]
            d = ins.result.type.shape[-1]
            table, codes = env[ins.args[0]], env[ins.args[1]]
            idx = unpack(codes.contiguous(), bits, d)
            flat = idx.reshape(-1).to(torch.int32)
            env[ins.result.name] = table.index_select(0, flat).view(idx.shape)
        elif ins.op == "rot":
            env[ins.result.name] = env[ins.args[0]] @ env[ins.args[1]]
        elif ins.op == "rescale":
            x, s = env[ins.args[0]], env[ins.args[1]]
            env[ins.result.name] = x * s.unsqueeze(-1)
        elif ins.op == "cast":
            env[ins.result.name] = env[ins.args[0]].to(
                _TORCH_DTYPE[ins.args[1]])
        elif ins.op == "store":
            value = env[ins.args[1]]
            want = _TORCH_DTYPE[out_val.type.dtype]
            value = value.to(want)      # folded cast, or a no-op if unfused
            if out is None:
                result = value.clone()
            else:
                out.copy_(value)
                result = out
        else:
            raise TSCError(f"reference interpreter has no rule for {ins.op!r}")

    if result is None:
        raise TSCError("module never stored its output")
    return result
