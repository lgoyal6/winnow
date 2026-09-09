"""Unknown-hardware fallback: compile and validate everywhere, execute the
PyTorch reference when there is no supported GPU.

These tests need torch but no GPU and no triton; they are the local half of
the execution story. The GPU half lives in tools/run_gpu_differential.py and
runs on a CUDA host.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from packing import pack  # noqa: E402
from tsc.examples import example_source  # noqa: E402
from tsc.ir import lower  # noqa: E402
from tsc.optimize import optimize  # noqa: E402
from tsc.parser import parse  # noqa: E402
from tsc.reference import run_reference  # noqa: E402
from tsc.runtime import execute  # noqa: E402
from tsc.validate import validate  # noqa: E402

HAS_CUDA = torch.cuda.is_available()


def _random_problem(bw: int, rows: int = 64, d: int = 128, seed: int = 0):
    """Random codebook, random orthogonal rotation, random packed codes."""
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, 2 ** bw, (rows, d), generator=gen,
                        dtype=torch.uint8)
    centroids = torch.randn(2 ** bw, generator=gen).sort().values
    q, r = torch.linalg.qr(torch.randn(d, d, generator=gen))
    pi = (q * torch.sign(torch.diag(r))).contiguous()
    norms = torch.rand(rows, generator=gen, dtype=torch.float16) + 0.5
    return {
        "packed": pack(idx, bw),
        "norms": norms,
        "centroids": centroids,
        "pi": pi,
    }, idx


def _direct_formula(inputs, idx):
    """The dequantization written out independently of the compiler."""
    y = inputs["centroids"][idx.long()]
    x = y @ inputs["pi"]
    return (x * inputs["norms"].float().unsqueeze(-1)).to(torch.bfloat16)


@pytest.mark.parametrize("bw", [3, 4, 8])
def test_reference_interpreter_matches_direct_formula(bw):
    inputs, idx = _random_problem(bw)
    module, _ = optimize(lower(validate(parse(example_source(bw)))))
    got = run_reference(module, inputs)
    want = _direct_formula(inputs, idx)
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, want)


def test_unoptimized_ir_executes_identically():
    inputs, idx = _random_problem(4)
    module = lower(validate(parse(example_source(4))))
    opt, _ = optimize(module)
    assert torch.equal(run_reference(module, inputs),
                       run_reference(opt, inputs))


@pytest.mark.skipif(HAS_CUDA, reason="this asserts the no-GPU fallback")
def test_execute_falls_back_to_reference_without_gpu():
    inputs, idx = _random_problem(4)
    out, report = execute(example_source(4), inputs)
    assert report.backend == "reference_torch"
    assert "no CUDA device" in report.reason
    assert torch.equal(out, _direct_formula(inputs, idx))


@pytest.mark.skipif(HAS_CUDA, reason="this asserts the no-GPU fallback")
def test_fallback_still_validates_and_reports_bad_source():
    from tsc.errors import TSCValidationError
    bad = example_source(4).replace("z     = rescale(x, nf)",
                                    "z     = rescale(x, norms)")
    with pytest.raises(TSCValidationError):
        execute(bad, _random_problem(4)[0])


def test_reference_handles_strided_cache_slice():
    # A (B, H, MAXLEN, NB) buffer sliced to :L, like TQPackedLayer hands out.
    bw, d, B, H, L, MAXLEN = 4, 128, 2, 4, 48, 64
    gen = torch.Generator().manual_seed(1)
    idx = torch.randint(0, 2 ** bw, (B, H, MAXLEN, d), generator=gen,
                        dtype=torch.uint8)
    buf = pack(idx, bw)
    norms_buf = torch.rand(B, H, MAXLEN, generator=gen,
                           dtype=torch.float16) + 0.5
    base, _ = _random_problem(bw, rows=1, d=d)
    inputs = {"packed": buf[:, :, :L], "norms": norms_buf[:, :, :L],
              "centroids": base["centroids"], "pi": base["pi"]}
    module, _ = optimize(lower(validate(parse(example_source(bw)))))
    got = run_reference(module, inputs)
    assert got.shape == (B, H, L, d)
    want = _direct_formula(inputs, idx[:, :, :L])
    assert torch.equal(got, want)


def _run_all():
    import traceback
    names = [k for k in sorted(globals()) if k.startswith("test_")]
    failed = 0
    for name in names:
        fn = globals()[name]
        marks = getattr(fn, "pytestmark", [])
        try:
            if any(m.name == "parametrize" for m in marks):
                for bw in (3, 4, 8):
                    fn(bw)
            elif any(m.name == "skipif" and m.args[0] for m in marks):
                print(f"  skip {name}")
                continue
            else:
                fn()
            print(f"  ok  {name}")
        except Exception:
            failed += 1
            traceback.print_exc()
    print(f"\n{len(names) - failed}/{len(names)} tsc fallback tests passed.")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    _run_all()
