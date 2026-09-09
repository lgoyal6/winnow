"""Regression tests for runtime.load_generated.

triton 3.7.1's @jit decorator inspects the kernel's source at definition
time, which fails for code exec'd under a pseudo-filename ("@jit functions
should be defined in a Python file"). That crashed the first GPU
differential attempt on 2026-09-08. These tests run without triton or a
GPU: a stub triton whose jit decorator performs the same source inspection
reproduces the failure mode, so the loader must produce a real, inspectable
module file to pass.
"""
from __future__ import annotations

import contextlib
import inspect
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from tsc.compiler import compile_source  # noqa: E402
from tsc.examples import example_source  # noqa: E402
from tsc.runtime import load_generated  # noqa: E402


@contextlib.contextmanager
def _stub_triton():
    """A triton stand-in whose @jit inspects source like triton 3.7.1."""
    def jit(fn):
        inspect.getsourcelines(fn)   # raises OSError for pseudo-filenames
        return fn

    triton = types.ModuleType("triton")
    triton.jit = jit
    tl = types.ModuleType("triton.language")
    tl.constexpr = object()
    triton.language = tl
    saved = {k: sys.modules.get(k) for k in ("triton", "triton.language")}
    sys.modules["triton"] = triton
    sys.modules["triton.language"] = tl
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_load_generated_survives_jit_source_inspection():
    result = compile_source(example_source(4))
    with _stub_triton():
        mod = load_generated(result)
    # the decorated kernel's source must remain retrievable after loading
    assert "tq_dequant_kernel" in inspect.getsource(mod.tq_dequant_kernel)


def test_load_generated_module_file_is_real_and_matches_source():
    result = compile_source(example_source(4))
    with _stub_triton():
        mod = load_generated(result)
    assert os.path.isfile(mod.__file__)
    with open(mod.__file__) as fh:
        assert fh.read() == result.generated
    assert mod.KERNEL_META["bits"] == 4


def test_pseudo_filename_exec_is_rejected_by_jit_inspection():
    # The pre-fix loader failure mode, kept as proof the stub is faithful.
    result = compile_source(example_source(4))
    mod = types.ModuleType("tsc_pseudo")
    with _stub_triton(), pytest.raises(OSError):
        exec(compile(result.generated, "<tsc:pseudo>", "exec"), mod.__dict__)


def _run_all():
    import traceback
    names = [k for k in sorted(globals()) if k.startswith("test_")]
    failed = 0
    for name in names:
        try:
            globals()[name]()
            print(f"  ok  {name}")
        except Exception:
            failed += 1
            traceback.print_exc()
    print(f"\n{len(names) - failed}/{len(names)} tsc loader tests passed.")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    _run_all()
