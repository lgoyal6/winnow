"""Profile-guided kernel selection for TurboQuant dequantization.

The planner is deliberately conservative. It will use a retained profile only
when the current GPU, numerical budget, tensor geometry, and available runtime
backends are all covered by measured evidence. Every other case returns the
PyTorch reference path.

No benchmark runs at import time. The checked-in A6000 results are the profile;
``dispatch_report.py`` evaluates them with held-out shapes and measures the CPU
cost of making a decision.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Hardware:
    name: str
    compute_capability: tuple[int, int] | None

    @classmethod
    def from_torch(cls, torch_module, device) -> "Hardware":
        try:
            index = torch_module.device(device).index
            if index is None:
                index = torch_module.cuda.current_device()
            return cls(
                name=torch_module.cuda.get_device_name(index),
                compute_capability=tuple(torch_module.cuda.get_device_capability(index)),
            )
        except Exception:
            return cls(name="unknown", compute_capability=None)


@dataclass(frozen=True)
class Shape:
    batch: int
    kv_heads: int
    cache_len: int
    head_dim: int
    bit_width: int

    @property
    def rows(self) -> int:
        return self.batch * self.kv_heads * self.cache_len


@dataclass(frozen=True)
class NumericalContract:
    """Largest measured deviation a selected backend may have from PyTorch."""

    max_abs_error: float

    @classmethod
    def reference_exact(cls) -> "NumericalContract":
        return cls(max_abs_error=0.0)

    @classmethod
    def quantization_aware(cls) -> "NumericalContract":
        # Retained A6000 TF32 sweeps observed at most one bf16 quantum here.
        return cls(max_abs_error=0.03125)


@dataclass(frozen=True)
class Comparison:
    shape: Shape
    backend: str
    backend_us: float
    torch_us: float
    max_abs_error: float
    source: str

    @property
    def speedup(self) -> float:
        return self.torch_us / self.backend_us


@dataclass(frozen=True)
class Decision:
    backend: str
    reason: str
    estimated_speedup: float
    evidence: tuple[str, ...]


class RetainedProfile:
    """A hardware-bound set of matched backend/PyTorch comparisons."""

    def __init__(self, comparisons: Iterable[Comparison]):
        self.comparisons = tuple(comparisons)

    @classmethod
    def a6000(cls, root: Path = ROOT) -> "RetainedProfile":
        points: list[Comparison] = []
        phase_c = root / "turboquant_kv" / "results"
        for filename, backend in (
            ("phaseC_kernel.json", "triton_fp32"),
            ("phaseC_kernel_tf32.json", "triton_tf32"),
            ("phaseC_kernel_bw6.json", "triton_tf32"),
        ):
            path = phase_c / filename
            result = json.loads(path.read_text())
            for row in result["rows"]:
                points.append(Comparison(
                    shape=Shape(
                        batch=int(row["batch"]), kv_heads=4,
                        cache_len=int(row["cache_len"]),
                        head_dim=int(result["head_dim"]),
                        bit_width=int(result["bit_width"]),
                    ),
                    backend=backend,
                    backend_us=float(row["triton_us"]),
                    torch_us=float(row["torch_us"]),
                    max_abs_error=float(row["max_abs_err"]),
                    source=str(path.relative_to(root)),
                ))

        # This retained C27 shape is the only native-CUDA record that also
        # timed the PyTorch reference. The other matched records remain useful
        # negative controls, but cannot prove a safe switch away from PyTorch.
        path = root / "experiments" / "c27" / "out" / "matched_N262144.json"
        result = json.loads(path.read_text())
        torch_us = float(result["arms"]["torch_fp32"]["us"])
        for backend in ("triton_fp32", "triton_tf32", "native_cuda_fp32"):
            arm = result["arms"][backend]
            points.append(Comparison(
                shape=Shape(
                    batch=int(result["B"]), kv_heads=int(result["H"]),
                    cache_len=int(result["L"]), head_dim=int(result["D"]),
                    bit_width=int(result["BW"]),
                ),
                backend=backend,
                backend_us=float(arm["us"]), torch_us=torch_us,
                max_abs_error=float(arm["max_abs_err"]),
                source=str(path.relative_to(root)),
            ))
        return cls(points)


class Dispatcher:
    """Select the lowest-risk measured backend for one dequantization call."""

    PROFILE_HARDWARE = Hardware("NVIDIA RTX A6000", (8, 6))

    def __init__(self, profile: RetainedProfile | None = None,
                 min_speedup: float = 1.05, neighbours: int = 3):
        self.profile = profile or RetainedProfile.a6000()
        self.min_speedup = min_speedup
        self.neighbours = neighbours
        self._cache: dict[tuple, Decision] = {}

    @staticmethod
    def _hardware_matches(hardware: Hardware) -> bool:
        return (
            hardware.compute_capability == Dispatcher.PROFILE_HARDWARE.compute_capability
            and "A6000" in hardware.name.upper()
        )

    @staticmethod
    def _distance(left: Shape, right: Shape) -> float:
        return (
            abs(math.log2(left.rows) - math.log2(right.rows))
            + 0.25 * abs(math.log2(left.batch) - math.log2(right.batch))
            + 0.25 * abs(math.log2(left.cache_len) - math.log2(right.cache_len))
        )

    def select(self, hardware: Hardware, shape: Shape,
               contract: NumericalContract,
               available: Iterable[str]) -> Decision:
        available_set = frozenset(available) | {"torch_fp32"}
        key = (hardware, shape, contract, available_set)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if not self._hardware_matches(hardware):
            return self._remember(key, Decision(
                "torch_fp32", "no retained profile matches this GPU", 1.0, ()))

        matching = [
            point for point in self.profile.comparisons
            if point.shape.bit_width == shape.bit_width
            and point.shape.head_dim == shape.head_dim
            and point.shape.kv_heads == shape.kv_heads
            and point.backend in available_set
            and point.max_abs_error <= contract.max_abs_error
        ]
        choices: list[tuple[float, str, tuple[Comparison, ...]]] = []
        for backend in sorted({point.backend for point in matching}):
            points = [point for point in matching if point.backend == backend]
            rows = [point.shape.rows for point in points]
            if not rows or not (min(rows) <= shape.rows <= max(rows)):
                continue
            ranked = tuple(sorted(points, key=lambda p: self._distance(shape, p.shape))[
                :self.neighbours
            ])
            exact_geometry = any(point.shape == shape for point in ranked)
            # One measured point supports only that exact geometry. Interpolate
            # only when several independent shapes agree on the winner.
            if len(points) < self.neighbours and not exact_geometry:
                continue
            if any(point.speedup < self.min_speedup for point in ranked):
                continue
            conservative = min(point.speedup for point in ranked)
            choices.append((conservative, backend, ranked))

        if not choices:
            decision = Decision(
                "torch_fp32",
                "no measured backend satisfies the shape and error contract",
                1.0, (),
            )
        else:
            speedup, backend, evidence = max(choices, key=lambda item: item[0])
            decision = Decision(
                backend,
                f"{len(evidence)} nearby measured shapes all beat PyTorch by at least "
                f"{speedup:.2f}x within max_abs_error={contract.max_abs_error:g}",
                speedup,
                tuple(sorted({point.source for point in evidence})),
            )
        return self._remember(key, decision)

    def _remember(self, key: tuple, decision: Decision) -> Decision:
        self._cache[key] = decision
        return decision


_DEFAULT_DISPATCHER: Dispatcher | None = None


def default_dispatcher() -> Dispatcher:
    global _DEFAULT_DISPATCHER
    if _DEFAULT_DISPATCHER is None:
        _DEFAULT_DISPATCHER = Dispatcher()
    return _DEFAULT_DISPATCHER
