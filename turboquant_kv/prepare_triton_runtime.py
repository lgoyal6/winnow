"""Backport upstream Triton #12128 to its pinned 3.8.0 runtime source.

Run once after installing requirements-packed-gpu.txt, before launching a GPU
process. The exact source hashes prevent patching an unexpected installation.
No CUDA API or sanitizer error is suppressed. Upstream change:
https://github.com/triton-lang/triton/commit/abfa53d26a7311e527c2d18942dfe91ac85c1647
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

BASE_SHA = "c32586d1814fc6770f966087b88fce0611db5e10e96d95579ba51631576d533c"
FIXED_SHA = "e1670c93abde5c4745378e6c1f084fb4440398cdbc0ca452e2e5a04bb01a9936"
OLD = """  if (shared > 49152 && shared_optin > 49152) {
    // XXX: remove attribute enum defs once latest cuda.h is in use
    // try to use oversized shared memory via new API
    // L1$ size is set to 8KB, CGA scheduling must be in SPREAD mode.
    if (CUDA_SUCCESS !=
        cuFuncSetAttribute(
            fun, CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE,
            CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY)) {
      // use legacy api if cuda doesn't support oversized shared memory
"""
NEW = """  if (shared > 49152 && shared_optin > 49152) {
    int major, minor;
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuDeviceGetAttribute(
        &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuDeviceGetAttribute(
        &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    // XXX: remove attribute enum defs once latest cuda.h is in use
    // try to use oversized shared memory via new API
    // L1$ size is set to 8KB, CGA scheduling must be in SPREAD mode.
    // Use the same CC 10.7 guard as the oversized device-attribute query.
    if (major == 10 && minor == 7) {
      CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuFuncSetAttribute(
          fun, CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE,
          CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY));
    } else {
      // Use legacy opt-in on other devices.
"""


def prepare_driver(path: Path) -> dict:
    source = path.read_bytes()
    before = hashlib.sha256(source).hexdigest()
    if before == FIXED_SHA:
        return {"driver": str(path), "before_sha256": before,
                "after_sha256": before, "changed": False}
    if before != BASE_SHA:
        raise RuntimeError(f"Unexpected Triton driver source {before}; refusing to modify it")
    text = source.decode()
    if text.count(OLD) != 1:
        raise RuntimeError("Upstream patch context is not unique")
    fixed = text.replace(OLD, NEW, 1).encode()
    after = hashlib.sha256(fixed).hexdigest()
    if after != FIXED_SHA:
        raise RuntimeError("Runtime patch does not match the upstream diff")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(fixed)
        temporary.chmod(path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"driver": str(path), "before_sha256": before,
            "after_sha256": after, "changed": True}


if __name__ == "__main__":
    distribution = importlib.metadata.distribution("triton")
    path = Path(distribution.locate_file("triton/backends/nvidia/driver.c"))
    print(json.dumps({"triton": distribution.version, **prepare_driver(path)}))
