"""GPU completion harness for generated versus reference and handwritten kernels."""
from __future__ import annotations

import argparse
import json
import time
import types

import torch
import triton

from kernel import tq_dequant
from packing import pack
from tsc.codegen import generate_triton
from tsc.examples import example_source
from tsc.ir import fuse_unpack_gather, lower
from tsc.parser import parse
from tsc.validate import validate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--negative-index", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("BLOCKED: CUDA GPU required for execution differential")
    torch.manual_seed(0)
    rows, dim, bits = 64, 128, 4
    idx = torch.randint(0, 2**bits, (rows, dim), device="cuda", dtype=torch.uint8)
    centroids = torch.linspace(-1, 1, 2**bits, device="cuda", dtype=torch.float32)
    pi = torch.eye(dim, device="cuda", dtype=torch.float32)
    norms = torch.rand(rows, device="cuda", dtype=torch.float16)
    packed = pack(idx, bits)
    reference = (centroids[idx.long()] @ pi * norms.float()[:, None]).to(torch.bfloat16)
    module = fuse_unpack_gather(lower(validate(parse(example_source(bits, dim)))))
    source = generate_triton(module, indexing_delta=1 if args.negative_index else 0)
    generated = types.ModuleType("generated_tq_dequant")
    exec(compile(source, "generated_tq_dequant.py", "exec"), generated.__dict__)
    torch.cuda.synchronize()
    started = time.perf_counter()
    got = generated.run(packed, norms, centroids, pi)
    torch.cuda.synchronize()
    compile_and_run_ms = (time.perf_counter() - started) * 1000
    handwritten = tq_dequant(packed, norms, centroids, pi, bits, dim)
    gen_ok = torch.allclose(got.float(), reference.float(), atol=2e-2, rtol=2e-2)
    hand_ok = torch.allclose(handwritten.float(), reference.float(), atol=2e-2, rtol=2e-2)
    error = (got.float() - reference.float()).abs()
    report = {
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "shape": list(reference.shape),
        "input_dtype": str(packed.dtype),
        "output_dtype": str(got.dtype),
        "atol": 2e-2,
        "rtol": 2e-2,
        "generated_matches_reference": gen_ok,
        "handwritten_matches_reference": hand_ok,
        "generated_matches_handwritten": torch.equal(got, handwritten),
        "generated_max_abs_error": float(error.max()),
        "generated_mean_abs_error": float(error.mean()),
        "compile_and_first_run_ms": compile_and_run_ms,
        "negative_index": args.negative_index,
    }
    print(json.dumps(report, sort_keys=True))
    expected = not args.negative_index
    if gen_ok != expected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
