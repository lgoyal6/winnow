"""Credential-free verification of the retained A6000 dispatch profile."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from dispatch import (
    Comparison, Dispatcher, Hardware, NumericalContract, RetainedProfile, Shape,
)


HERE = Path(__file__).resolve().parent
A6000 = Hardware("NVIDIA RTX A6000", (8, 6))


def held_out_evaluation() -> dict:
    full = RetainedProfile.a6000()
    tf32 = [
        point for point in full.comparisons
        if point.backend == "triton_tf32"
    ]
    # Cache lengths were chosen before evaluating: keep the end points in the
    # calibration set and hold out three interior lengths for every batch.
    held_lengths = {8, 128, 2048}
    held = [point for point in tf32 if point.shape.cache_len in held_lengths]
    held_ids = {(point.backend, point.shape) for point in held}
    calibration = [
        point for point in full.comparisons
        if (point.backend, point.shape) not in held_ids
    ]
    dispatcher = Dispatcher(RetainedProfile(calibration))
    regrets = []
    choices = {}
    for point in held:
        decision = dispatcher.select(
            A6000, point.shape, NumericalContract.quantization_aware(),
            {"torch_fp32", "triton_tf32"},
        )
        selected_us = point.backend_us if decision.backend == point.backend else point.torch_us
        best_us = min(point.backend_us, point.torch_us)
        regrets.append(selected_us / best_us)
        choices[decision.backend] = choices.get(decision.backend, 0) + 1
    return {
        "calibration_shapes": len(tf32) - len(held),
        "held_out_shapes": len(held),
        "choices": choices,
        "max_regret": max(regrets),
        "mean_regret": statistics.mean(regrets),
    }


def selection_overhead(iterations: int = 20_000) -> dict:
    dispatcher = Dispatcher()
    shape = Shape(4, 4, 2048, 128, 4)
    contract = NumericalContract.quantization_aware()
    available = {"torch_fp32", "triton_tf32", "triton_fp32"}
    dispatcher.select(A6000, shape, contract, available)
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        dispatcher.select(A6000, shape, contract, available)
        samples.append(time.perf_counter_ns() - started)
    samples.sort()
    return {
        "iterations": iterations,
        "median_us": statistics.median(samples) / 1_000,
        "p95_us": samples[int(0.95 * len(samples))] / 1_000,
    }


def negative_controls() -> dict:
    dispatcher = Dispatcher()
    shape = Shape(4, 4, 2048, 128, 4)
    contract = NumericalContract.quantization_aware()
    unknown = dispatcher.select(
        Hardware("NVIDIA H100", (9, 0)), shape, contract,
        {"torch_fp32", "triton_tf32"},
    )
    too_strict = dispatcher.select(
        A6000, shape, NumericalContract.reference_exact(),
        {"torch_fp32", "triton_tf32"},
    )
    unavailable = dispatcher.select(
        A6000, shape, contract, {"torch_fp32"},
    )
    assert unknown.backend == "torch_fp32"
    assert too_strict.backend == "torch_fp32"
    assert unavailable.backend == "torch_fp32"
    return {
        "unknown_hardware": unknown.backend,
        "zero_error_budget": too_strict.backend,
        "optimized_backend_unavailable": unavailable.backend,
    }


def retained_findings() -> dict:
    root = HERE.parent
    matched = json.loads((
        root / "experiments" / "c27" / "out" / "matched_N1048576.json"
    ).read_text())
    native_us = matched["arms"]["native_cuda_fp32"]["us"]
    triton_us = matched["arms"]["triton_tf32"]["us"]
    decode = json.loads((
        root / "experiments" / "c27" / "out" / "decode_0.6b.json"
    ).read_text())
    return {
        "matched_shape": {
            "rows": matched["N"],
            "native_cuda_us": native_us,
            "triton_tf32_us": triton_us,
            "native_cuda_slowdown": native_us / triton_us,
            "dispatcher_implication": (
                "The approximation-tolerant policy does not choose native CUDA "
                "at this shape."
            ),
        },
        "full_decode_limitation": {
            "fp16_beats_quantized_arms_at_all_contexts": all(
                min(
                    arm["ms_per_token"] for name, arm in arms.items()
                    if name != "fp16_baseline"
                ) > arms["fp16_baseline"]["ms_per_token"]
                for arms in decode.values()
            ),
            "contexts": sorted(int(context) for context in decode),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max-regret", type=float, default=1.05)
    parser.add_argument("--max-p95-us", type=float, default=25.0)
    args = parser.parse_args()
    report = {
        "profile": "RTX A6000 sm_86 retained measurements",
        "held_out": held_out_evaluation(),
        "selection_overhead": selection_overhead(),
        "negative_controls": negative_controls(),
        "retained_findings": retained_findings(),
        "limitation": (
            "The profile covers A6000 sm_86, head_dim=128, and measured bit widths; "
            "unknown hardware or unsupported shapes use PyTorch."
        ),
    }
    assert report["held_out"]["max_regret"] <= args.max_regret
    assert report["selection_overhead"]["p95_us"] <= args.max_p95_us
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("PASS: profile-guided dispatch verified")
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
