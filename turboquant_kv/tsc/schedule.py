"""Finite schedule spaces and static legality for the two tunable kernels.

Two operators are tuned, each with only the knobs its backend really has:

* the tsc-generated dequantizer (`tq_dequant`): BLOCK_N rows per program and
  num_warps. The kernel body has no loop (one tl.dot against the resident
  rotation), so Triton's num_stages software pipelining has nothing to
  pipeline; it is fixed rather than offered as a knob that changes nothing.
  allow_tf32 changes the numerical contract, so it is part of the problem
  (which contract is being tuned), never a free schedule choice.
* the packed decode attention kernel (packed_attention_triton): KV splits,
  pages gathered per loop iteration (PPI), num_warps, num_stages. All four
  change the generated program; PPI and splits change its parallelism and
  loads in flight, which is what the track 01 profile said was missing.

Legality is decided before any compile or launch, from tensor geometry,
target resources and compiler capabilities. Resource checks are conservative
static estimates (operand tiles staged in shared memory, accumulators in
registers); they can only reject. A candidate that passes them can still fail
to compile or launch, and the search records that as a ledger outcome rather
than trusting the estimate.
"""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

# -- targets ---------------------------------------------------------------

@dataclass(frozen=True)
class Target:
    name: str
    compute_capability: tuple[int, int]
    smem_per_block: int            # opt-in dynamic shared memory per block, bytes
    regs_per_thread: int = 255
    max_threads: int = 1024
    sms: int = 108


TARGETS = {
    "sm80": Target("sm80", (8, 0), 166912, sms=108),     # A100
    "sm86": Target("sm86", (8, 6), 101376, sms=84),      # A6000 / RTX 30xx
    "sm89": Target("sm89", (8, 9), 101376, sms=128),     # L4 / L40 / RTX 40xx
    "sm90": Target("sm90", (9, 0), 232448, sms=132),     # H100
}


def target_for(compute_capability: tuple[int, int]) -> Target | None:
    key = f"sm{compute_capability[0]}{compute_capability[1]}"
    return TARGETS.get(key)


# Registers per thread the estimate leaves for addressing, masks and loop state.
_REG_RESERVE = 48


def _pow2(x: int, lo: int = 1) -> bool:
    return x >= lo and (x & (x - 1)) == 0


# -- dequantization --------------------------------------------------------

@dataclass(frozen=True)
class DequantProblem:
    rows: int                  # N = B * H * L vectors
    head_dim: int
    bit_width: int
    contract: str = "fp32"     # "fp32" (exact rotation) or "tf32"

    def family(self) -> dict:
        """What a promoted artifact must match exactly (rows is a size, not family)."""
        return {"op": "tq_dequant", "head_dim": self.head_dim,
                "bit_width": self.bit_width, "contract": self.contract,
                "out_dtype": "bf16"}


@dataclass(frozen=True)
class DequantSchedule:
    block_n: int
    num_warps: int

    def key(self) -> str:
        return f"bn{self.block_n}_w{self.num_warps}"


HEURISTIC_DEQUANT = DequantSchedule(block_n=64, num_warps=4)    # the generated default
DEQUANT_SPACE = tuple(DequantSchedule(bn, w) for bn, w in
                      itertools.product((16, 32, 64, 128, 256), (1, 2, 4, 8)))


def dequant_legal(s: DequantSchedule, p: DequantProblem, t: Target) -> tuple[bool, str]:
    if not _pow2(s.block_n, 16):
        return False, "block_n must be a power of two >= 16 (tl.dot M dimension)"
    if not _pow2(s.num_warps) or s.num_warps * 32 > t.max_threads:
        return False, "num_warps must be a power of two within the thread limit"
    if not _pow2(p.head_dim, 16):
        return False, "head_dim must be a power of two >= 16 for this backend"
    # tl.dot(y, pi): both operands staged in shared memory, fp32.
    smem = (s.block_n * p.head_dim + p.head_dim * p.head_dim) * 4
    if smem > t.smem_per_block:
        return False, f"dot operands need {smem} B shared memory > {t.smem_per_block}"
    # fp32 accumulator tile spread over the program's threads.
    regs = s.block_n * p.head_dim // (s.num_warps * 32)
    if regs + _REG_RESERVE > t.regs_per_thread:
        return False, f"~{regs} accumulator registers/thread exceeds {t.regs_per_thread - _REG_RESERVE}"
    return True, "legal"


# -- packed decode attention ----------------------------------------------

@dataclass(frozen=True)
class AttnProblem:
    batch: int
    seq_len: int
    hq: int
    hkv: int
    head_dim: int
    page_size: int
    bit_width: int
    dense: bool = False

    def family(self) -> dict:
        """Exact-match fields; batch and seq_len are sizes covered by validation."""
        return {"op": "dense_decode_attention" if self.dense else "packed_decode_attention",
                "hq": self.hq, "hkv": self.hkv, "head_dim": self.head_dim,
                "page_size": self.page_size,
                "bit_width": None if self.dense else self.bit_width}

    @property
    def max_pages(self) -> int:
        return -(-self.seq_len // self.page_size)


@dataclass(frozen=True)
class AttnSchedule:
    num_splits: int
    pages_per_iter: int
    num_warps: int
    num_stages: int

    def key(self) -> str:
        return f"s{self.num_splits}_p{self.pages_per_iter}_w{self.num_warps}_st{self.num_stages}"

    def kwargs(self) -> dict:
        return asdict(self)


HEURISTIC_ATTN = AttnSchedule(num_splits=1, pages_per_iter=1, num_warps=4, num_stages=2)
ATTN_SPACE = tuple(AttnSchedule(sp, ppi, w, st) for sp, ppi, w, st in
                   itertools.product((1, 2, 4, 8, 16), (1, 2, 4, 8), (2, 4, 8), (1, 2, 3)))

_WORKSPACE_CAP = 256 << 20      # split partials (m, l, acc) per launch


def attn_legal(s: AttnSchedule, p: AttnProblem, t: Target) -> tuple[bool, str]:
    n_rep = p.hq // p.hkv
    block_h = max(16, 1 << (n_rep - 1).bit_length())
    tokens = s.pages_per_iter * p.page_size
    if p.hq % p.hkv:
        return False, "hq must be a multiple of hkv"
    if not _pow2(p.page_size, 16) or not _pow2(p.head_dim, 16):
        return False, "page_size and head_dim must be powers of two >= 16"
    if not _pow2(s.pages_per_iter) or not _pow2(s.num_warps) or s.num_warps * 32 > t.max_threads:
        return False, "pages_per_iter and num_warps must be powers of two within limits"
    if s.num_stages < 1:
        return False, "num_stages must be >= 1"
    if s.num_splits < 1:
        return False, "num_splits must be >= 1"
    if s.pages_per_iter > p.max_pages:
        return False, f"pages_per_iter {s.pages_per_iter} exceeds the {p.max_pages} pages of the sequence"
    groups = -(-p.max_pages // s.pages_per_iter)
    if s.num_splits > groups:
        return False, f"{s.num_splits} splits > {groups} page groups: empty splits"
    elem = 2 if p.dense else 4          # dense pages load bf16 tiles; packed decode to fp32
    # Per stage: q (BLOCK_H x D) and K tile (T x D) for the score dot; P (BLOCK_H x T)
    # and V tile (T x D) for the value dot.
    per_stage = (block_h * p.head_dim + tokens * p.head_dim) * 4 + \
                (block_h * tokens + tokens * p.head_dim) * elem
    smem = per_stage * s.num_stages
    if smem > t.smem_per_block:
        return False, f"~{smem} B staged shared memory > {t.smem_per_block}"
    regs = (block_h * p.head_dim + block_h * tokens) // (s.num_warps * 32)
    if regs + _REG_RESERVE > t.regs_per_thread:
        return False, f"~{regs} accumulator registers/thread exceeds {t.regs_per_thread - _REG_RESERVE}"
    workspace = p.batch * p.hq * s.num_splits * (p.head_dim + 2) * 4
    if workspace > _WORKSPACE_CAP:
        return False, f"split workspace {workspace} B > cap {_WORKSPACE_CAP}"
    return True, "legal"


def enumerate_legal(space, problem, target, legal_fn):
    """Partition a space into legal candidates and (candidate, reason) rejections."""
    ok, bad = [], []
    for s in space:
        good, why = legal_fn(s, problem, target)
        (ok if good else bad).append(s if good else (s, why))
    return ok, bad
