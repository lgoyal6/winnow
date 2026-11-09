"""Runtime selection of promoted tuned kernels, with an explained fallback.

Two production call sites consult this module:

* `packed_backend.dequantize_history`, the materializing path every prefill
  chunk takes (track 01's direct decode never dequantizes history; prefill
  still does, ~13.7k calls in one 7B contract run). It runs the
  tsc-generated dequantizer at its promoted schedule, or the handwritten
  `kernel.tq_dequant` (the previous behaviour), or the torch path off CUDA.
* `packed_backend`'s Triton decode call, which takes its launch schedule
  (splits, pages per iteration, warps, stages) from a promoted record for
  the exact model family, target and kernel source, or the built-in default.

Nothing tunes during a request. A registry is opt-in (WINNOW_TUNED_REGISTRY
or an explicit path), records come from `tune_kernels.py` offline, and any
miss, mismatch, corruption or coverage gap selects the known-safe path. Every
decision is kept with its reason; `counts` says how often each path ran.
"""
from __future__ import annotations

import hashlib
import os
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def toolchain() -> dict:
    try:
        import triton
        tv = triton.__version__
        driver = Path(triton.__file__).parent / "backends/nvidia/driver.c"
        driver_sha = hashlib.sha256(driver.read_bytes()).hexdigest() if driver.is_file() else None
    except Exception:                                       # noqa: BLE001
        tv = None
        driver_sha = None
    return {"torch": torch.__version__, "triton": tv, "cuda": torch.version.cuda,
            "triton_driver_sha256": driver_sha}


def target_of(device) -> dict | None:
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    cc = torch.cuda.get_device_capability(device)
    return {"cc": [cc[0], cc[1]], "name": torch.cuda.get_device_name(device)}


def file_sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dequant_identity(bit_width, head_dim, contract, target, tc=None) -> dict:
    from tsc.compiler import COMPILER_VERSION
    from tsc.examples import example_source
    return {"op": "tq_dequant",
            "family": {"bit_width": bit_width, "head_dim": head_dim, "out_dtype": "bf16"},
            "contract": contract,
            "dsl_source_sha": hashlib.sha256(example_source(bit_width, head_dim).encode()).hexdigest(),
            "compiler_version": COMPILER_VERSION,
            "codegen": {"optimize_ir": True, "indexing_delta": 0},
            "toolchain": tc or toolchain(), "target": target}


def attention_identity(family: dict, target, tc=None) -> dict:
    return {"op": family["op"], "family": family, "contract": "fp32",
            "kernel_source_sha": file_sha(HERE / "packed_attention_triton.py"),
            "toolchain": tc or toolchain(), "target": target}


@dataclass(frozen=True)
class Decision:
    path: str
    reason: str
    schedule: dict | None = None
    source_sha: str | None = None


def handwritten_available() -> bool:
    try:
        from kernel import tq_dequant                       # noqa: F401 - needs triton
        return True
    except Exception:                                       # noqa: BLE001
        return False


class TunedDispatch:
    def __init__(self, registry_root=None):
        root = registry_root or os.environ.get("WINNOW_TUNED_REGISTRY")
        if root:
            from tsc.promotion import Registry
            self.registry = Registry(root)
        else:
            self.registry = None
        self.counts: Counter = Counter()
        self.last: dict[str, Decision] = {}
        self._modules: dict = {}
        self._lock = threading.Lock()

    def _note(self, op, d: Decision) -> Decision:
        self.counts[f"{op}:{d.path}"] += 1
        self.last[op] = d
        return d

    # -- dequantization ---------------------------------------------------
    def _generated(self, bit_width, head_dim, want_sha):
        key = (bit_width, head_dim, want_sha)
        with self._lock:
            if key not in self._modules:
                from tsc.compiler import compile_source
                from tsc.examples import example_source
                from tsc.runtime import load_generated
                res = compile_source(example_source(bit_width, head_dim))
                got = hashlib.sha256(res.generated.encode()).hexdigest()
                self._modules[key] = (load_generated(res) if got == want_sha else None, got)
            return self._modules[key]

    def dequant_decision(self, rows: int, bit_width, head_dim, device, contract="fp32") -> Decision:
        target = target_of(device)
        if target is None:
            return Decision("torch", "not a CUDA device")
        safe = Decision("handwritten", "", None) if handwritten_available() else Decision("torch", "")
        if self.registry is None:
            return Decision(safe.path, "no tuned registry configured")
        ident = dequant_identity(bit_width, head_dim, contract, target)
        rec, why = self.registry.load(ident, size={"rows": rows})
        if rec is None:
            return Decision(safe.path, why)
        mod, got = self._generated(bit_width, head_dim, rec["extra"]["generated_source_sha"])
        if mod is None:
            return Decision(safe.path, f"regenerated source sha {got[:12]} != promoted "
                                       f"{rec['extra']['generated_source_sha'][:12]} (stale record)")
        return Decision("generated", "promoted record matched", rec["schedule"], got)

    def dequant(self, packed, norms, centroids, pi, bit_width, head_dim, out, contract="fp32"):
        rows = out.numel() // head_dim
        d = self._note("dequant", self.dequant_decision(rows, bit_width, head_dim, packed.device,
                                                         contract))
        if d.path == "generated":
            mod, _ = self._generated(bit_width, head_dim, d.source_sha)
            mod.run(packed, norms, centroids, pi, out=out, allow_tf32=contract == "tf32",
                    **d.schedule)
        elif d.path == "handwritten":
            from kernel import tq_dequant
            tq_dequant(packed, norms, centroids, pi, bit_width, head_dim, out=out,
                       allow_tf32=contract == "tf32")
        else:
            from packing import unpack
            idx = unpack(packed, bit_width, head_dim)
            y = centroids.index_select(0, idx.reshape(-1).to(torch.int32)).view(idx.shape)
            out.copy_(((y @ pi) * norms.float()[..., None]).to(out.dtype))
        return d

    # -- attention schedule -------------------------------------------------
    def attention_schedule(self, family: dict, batch: int, seq_len: int, device) -> Decision:
        target = target_of(device)
        if target is None:
            return self._note("attention", Decision("default", "not a CUDA device"))
        if self.registry is None:
            return self._note("attention", Decision("default", "no tuned registry configured"))
        rec, why = self.registry.load(attention_identity(family, target),
                                      size={"batch": batch, "seq_len": seq_len})
        if rec is None:
            return self._note("attention", Decision("default", why))
        for bucket in rec["schedule"]["buckets"]:
            if batch <= bucket["batch_max"] and seq_len <= bucket["len_max"]:
                return self._note("attention", Decision("tuned", f"bucket b<={bucket['batch_max']} "
                                                                 f"L<={bucket['len_max']}",
                                                        bucket["schedule"]))
        return self._note("attention", Decision("default", "no bucket covers this size"))


DISPATCH = TunedDispatch()


def reset(registry_root=None) -> TunedDispatch:
    """Point the process-wide dispatcher at another registry (tests, tools)."""
    global DISPATCH
    DISPATCH = TunedDispatch(registry_root)
    return DISPATCH
