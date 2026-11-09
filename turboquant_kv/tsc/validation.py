"""Correctness screen for tuned candidates, and the wrong-code mutants it must reject.

`validate_dequant` runs a candidate dequantizer (generated source + schedule)
against the tsc reference interpreter on a frozen adversarial set:

  index extremes (all zero, all max code), norm extremes (0, fp16 subnormal,
  1, 6e4), row tails (N not a multiple of BLOCK_N, N = 1), a strided cache
  slice (MAXLEN > L), and three rotations (identity, a fixed permutation, two
  seeded random orthogonal matrices).

Elements are judged with the repository's proven bf16 envelope
(tools/run_gpu_differential.check_envelope: at most 1 bf16 ULP, or under the
measured near-zero floor). The output buffer is a view into a larger buffer
whose tail is a canary, so a kernel that writes past N fails even though the
returned tensor looks right.

`validate_attention` runs a schedule of the packed (or dense) decode kernel
against the tiled FP32 reference on its own frozen case set.

`MUTANTS` are deterministic rewrites of generated dequantizer source: wrong
codebook index, dropped store mask, dropped norm rescale, transposed rotation.
They exist so tests can prove the screen rejects wrong code. The result is an
immutable record: case seeds, reference version, max/aggregate error,
nonfinite count, pass/fail and reason.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

REFERENCE_VERSION = "tsc.reference+packed_attention.tiled@v1"
VALIDATION_SEEDS = (1101, 1102, 1103)


def _envelope():
    spec = importlib.util.spec_from_file_location("run_gpu_differential",
                                                  ROOT / "tools" / "run_gpu_differential.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("run_gpu_differential", mod)
    spec.loader.exec_module(mod)
    return mod.check_envelope


@dataclass(frozen=True)
class ValidationResult:
    op: str
    schedule: str
    reference_version: str
    cases: tuple
    seeds: tuple
    max_abs_err: float
    max_ulp: int
    beyond_envelope: int
    nonfinite: int
    canary_violations: int
    passed: bool
    reason: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sha256"] = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
        return d


# -- dequantization ---------------------------------------------------------

def _rotations(D):
    eye = torch.eye(D)
    perm = eye[torch.roll(torch.arange(D), 3)]
    out = [("identity", eye), ("permutation", perm)]
    for seed in VALIDATION_SEEDS[:2]:
        g = torch.Generator().manual_seed(seed)
        Q, R = torch.linalg.qr(torch.randn(D, D, generator=g))
        out.append((f"orthogonal_{seed}", (Q * torch.sign(torch.diag(R))).contiguous()))
    return out


def dequant_cases(bit_width: int, head_dim: int):
    """Frozen adversarial inputs. Yields (name, packed, norms, centroids, pi)."""
    from packing import pack, packed_bytes
    levels = 2 ** bit_width
    nb = packed_bytes(head_dim, bit_width)
    g = torch.Generator().manual_seed(VALIDATION_SEEDS[2])
    cent = torch.sort(torch.randn(levels, generator=g)).values * 0.1
    rots = _rotations(head_dim)

    def rows(n, mode="random"):
        if mode == "zeros":
            idx = torch.zeros(n, head_dim, dtype=torch.uint8)
        elif mode == "max":
            idx = torch.full((n, head_dim), levels - 1, dtype=torch.uint8)
        else:
            idx = torch.randint(0, levels, (n, head_dim), generator=g, dtype=torch.uint8)
        return pack(idx, bit_width)

    norms_kinds = {"unit": lambda n: torch.ones(n), "zero": lambda n: torch.zeros(n),
                   "subnormal": lambda n: torch.full((n,), 6e-8),
                   "large": lambda n: torch.full((n,), 6.0e4),
                   "mixed": lambda n: torch.rand(n, generator=g) * 300}
    for rot_name, pi in rots:
        for n, mode, nk in ((1, "random", "mixed"), (17, "zeros", "unit"), (517, "max", "large"),
                            (1000, "random", "subnormal"), (256, "random", "zero"),
                            (4096, "random", "mixed")):
            yield (f"{rot_name}/n{n}/{mode}/{nk}", rows(n, mode),
                   norms_kinds[nk](n).half(), cent, pi)
    # Strided cache slice: (B, H, MAXLEN, NB) buffer read through its first L positions.
    B, H, MAXLEN, L = 2, 3, 40, 27
    buf = rows(B * H * MAXLEN).view(B, H, MAXLEN, nb)
    nbuf = (torch.rand(B, H, MAXLEN, generator=g) * 50).half()
    yield ("strided_slice", buf[:, :, :L], nbuf[:, :, :L], cent, rots[-1][1])


def _reference_dequant(packed, norms, cent, pi, bit_width, head_dim):
    from packing import unpack
    idx = unpack(packed.reshape(-1, packed.shape[-1]).cpu(), bit_width, head_dim)
    y = cent.cpu().index_select(0, idx.reshape(-1).long()).view(-1, head_dim)
    x = (y @ pi.cpu()) * norms.reshape(-1).cpu().float()[:, None]
    return x.to(torch.bfloat16).view(*packed.shape[:-1], head_dim)


def validate_dequant(run_fn, schedule, *, bit_width: int, head_dim: int, device,
                     contract: str = "fp32") -> ValidationResult:
    """`run_fn(packed, norms, centroids, pi, out=..., block_n=, num_warps=)`."""
    check = _envelope()
    worst_abs, worst_ulp, beyond, nonfinite, canary = 0.0, 0, 0, 0, 0
    names, fail = [], ""
    sentinel = -12352.0          # exactly representable in bf16, so untouched guard compares equal
    for name, packed, norms, cent, pi in dequant_cases(bit_width, head_dim):
        names.append(name)
        lead = packed.shape[:-1]
        n = int(torch.tensor(lead).prod())
        guard = torch.full((n + 256, head_dim), sentinel, dtype=torch.bfloat16, device=device)
        out = guard[:n].view(*lead, head_dim)
        try:
            run_fn(packed.to(device), norms.to(device), cent.to(device), pi.to(device),
                   out=out, block_n=schedule.block_n, num_warps=schedule.num_warps,
                   allow_tf32=contract == "tf32")
        except Exception as exc:                                    # noqa: BLE001
            fail = fail or f"{name}: raised {type(exc).__name__}: {exc}"[:300]
            continue
        ref = _reference_dequant(packed, norms, cent, pi, bit_width, head_dim)
        got = out.detach().cpu()
        nonfinite += int((~torch.isfinite(got.float())).sum())
        d = check(got.reshape(-1, head_dim), ref.reshape(-1, head_dim), contract)
        worst_abs = max(worst_abs, d["max_abs_err"])
        worst_ulp = max(worst_ulp, d["max_ulp"])
        beyond += d["beyond_envelope"]
        tail_bad = int((guard[n:].float().cpu() != sentinel).sum())
        canary += tail_bad
        if not fail and (d["beyond_envelope"] or tail_bad):
            fail = f"{name}: {d['beyond_envelope']} elements beyond envelope, {tail_bad} canary writes"
    passed = not fail and beyond == 0 and nonfinite == 0 and canary == 0
    return ValidationResult("tq_dequant", schedule.key(), REFERENCE_VERSION, tuple(names),
                            VALIDATION_SEEDS, worst_abs, worst_ulp, beyond, nonfinite, canary,
                            passed, "passed" if passed else (fail or "nonfinite outputs"))


def _sub(pattern, repl, src):
    out, n = re.subn(pattern, repl, src)
    if n == 0:
        raise ValueError(f"mutant anchor not found: {pattern}")
    return out


def _index_mutant(src):
    levels = int(re.search(r"KERNEL_META = .*'bits': (\d+)", src).group(1))
    return _sub(r"tl\.load\(centroids_ptr \+ (\w+)\)",
                rf"tl.load(centroids_ptr + ((\1 + 1) % {2 ** levels}))", src)


MUTANTS = {
    "wrong_index": _index_mutant,
    "dropped_store_mask": lambda s: _sub(r"(z\.to\(out_ptr\.dtype\.element_ty\)), mask=rmask\[:, None\]\)",
                                         r"\1)", s),
    "dropped_rescale": lambda s: _sub(r"= x \* nf\[:, None\]", "= x", s),
    "transposed_rotation": lambda s: _sub(r"tl\.dot\((\w+), (rotm_\d+),", r"tl.dot(\1, tl.trans(\2),", s),
}


# -- packed / dense decode attention -----------------------------------------

def attention_cases(hq, hkv, head_dim, page_size, bit_width):
    """Frozen cases: empty, one-token, page-boundary, tail, long, huge scores."""
    import packed_attention as pa
    from cache import _get_quantizer
    specs = [((0, 1, page_size, page_size + 1), 1.0), ((3 * page_size - 1, 70, 5), 1.0),
             ((1024, 517), 1.0), ((200, 33, 64, 9), 40.0)]
    for i, (lens, qs) in enumerate(specs):
        g = torch.Generator().manual_seed(VALIDATION_SEEDS[0] + i)
        B, Lmax = len(lens), max(max(lens), 1)
        k = torch.randn(B, hkv, Lmax, head_dim, generator=g)
        v = torch.randn(B, hkv, Lmax, head_dim, generator=g)
        q = torch.randn(B, hq, 1, head_dim, generator=g) * qs
        kv = pa.pages_from_dense(k, v, lens, _get_quantizer(bit_width, head_dim, "cpu", 42),
                                 _get_quantizer(bit_width, head_dim, "cpu", 43), page_size,
                                 generator=g)
        yield f"lens{lens}_q{qs}", q, kv


def validate_attention(schedule, *, hq, hkv, head_dim, page_size, bit_width, device,
                       atol=2e-3, rtol=2e-3) -> ValidationResult:
    import packed_attention as pa
    import packed_attention_triton as pt
    worst, nonfinite, names, fail = 0.0, 0, [], ""
    for name, q, kv in attention_cases(hq, hkv, head_dim, page_size, bit_width):
        names.append(name)
        scale = pa.default_scale(head_dim)
        ref = pa.tiled_compressed_attention(q, kv, scale)
        dev_kv = pa.PackedKV(*(t.to(device) for t in (kv.k_pages, kv.k_norms, kv.v_pages,
                                                      kv.v_norms, kv.page_table, kv.seq_lens)),
                             pa.Quantizer(kv.k_quant.Pi.to(device), kv.k_quant.centroids.to(device),
                                          kv.k_quant.bit_width),
                             pa.Quantizer(kv.v_quant.Pi.to(device), kv.v_quant.centroids.to(device),
                                          kv.v_quant.bit_width), kv.head_dim)
        try:
            got = pt.packed_decode_attention(q.to(device), dev_kv, scale, **schedule.kwargs())
        except Exception as exc:                                    # noqa: BLE001
            fail = fail or f"{name}: raised {type(exc).__name__}: {exc}"[:300]
            continue
        got = got.float().cpu()
        nonfinite += int((~torch.isfinite(got)).sum())
        err = (got - ref).abs()
        worst = max(worst, float(err.max()))
        if not fail and not torch.allclose(got, ref, atol=atol, rtol=rtol):
            fail = f"{name}: max abs err {float(err.max()):.3e} beyond atol {atol}"
    passed = not fail and nonfinite == 0
    return ValidationResult("packed_decode_attention", schedule.key(), REFERENCE_VERSION,
                            tuple(names), VALIDATION_SEEDS, worst, 0, 0, nonfinite, 0, passed,
                            "passed" if passed else (fail or "nonfinite outputs"))
