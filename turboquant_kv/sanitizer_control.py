"""Require memcheck to detect an intentional error in a separate CUDA process.

This is a detection control, never part of the production kernel or a test that
suppresses its diagnostics. The raw failing child's output is retained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


def corrupt():
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def invalid_store(pointer):
        # Far beyond PyTorch's small-allocation slab, not just its tensor view.
        tl.store(pointer + 16777216, 1.0)

    value = torch.zeros(1, device="cuda")
    invalid_store[(1,)](value)
    torch.cuda.synchronize()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    if args.child:
        corrupt()
        return 0
    if not args.out:
        parser.error("--out is required")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    command = ["compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "86",
               sys.executable, str(Path(__file__).resolve()), "--child"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=300)
    (out / "intentional-oob.log").write_text(result.stdout)
    detected = bool(re.search(r"Invalid __global__ (?:write|read)", result.stdout))
    record = {"command": command, "exit": result.returncode,
              "intentional_invalid_access_detected": detected,
              "pass": detected and result.returncode != 0}
    (out / "control.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))
    return 0 if record["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
