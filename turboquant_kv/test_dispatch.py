"""CPU-only tests for the profile-guided kernel dispatcher."""
from __future__ import annotations

from dataclasses import replace

from dispatch import (
    Comparison, Dispatcher, Hardware, NumericalContract, RetainedProfile, Shape,
)
from dispatch_report import held_out_evaluation, negative_controls


A6000 = Hardware("NVIDIA RTX A6000", (8, 6))


def run() -> None:
    dispatcher = Dispatcher()
    approximate = dispatcher.select(
        A6000, Shape(4, 4, 2048, 128, 4),
        NumericalContract.quantization_aware(),
        {"torch_fp32", "triton_fp32", "triton_tf32"},
    )
    assert approximate.backend == "triton_tf32", approximate
    assert approximate.estimated_speedup >= 5.0, approximate

    exact_native = dispatcher.select(
        A6000, Shape(4, 4, 16384, 128, 6),
        NumericalContract.reference_exact(),
        {"torch_fp32", "triton_fp32", "triton_tf32", "native_cuda_fp32"},
    )
    assert exact_native.backend == "native_cuda_fp32", exact_native

    large_approximate = dispatcher.select(
        A6000, Shape(16, 4, 16384, 128, 6),
        NumericalContract.quantization_aware(),
        {"torch_fp32", "triton_tf32", "native_cuda_fp32"},
    )
    assert large_approximate.backend == "triton_tf32", large_approximate
    assert large_approximate.estimated_speedup >= 30.0, large_approximate

    assert negative_controls() == {
        "unknown_hardware": "torch_fp32",
        "zero_error_budget": "torch_fp32",
        "optimized_backend_unavailable": "torch_fp32",
    }

    held = held_out_evaluation()
    assert held["held_out_shapes"] == 18, held
    assert held["choices"] == {"triton_tf32": 18}, held
    assert held["max_regret"] == 1.0, held

    # Negative control: poisoning every TF32 accuracy record must force the
    # reference path rather than silently relaxing the requested error budget.
    poisoned = [
        replace(point, max_abs_error=1.0)
        if point.backend == "triton_tf32" else point
        for point in RetainedProfile.a6000().comparisons
    ]
    rejected = Dispatcher(RetainedProfile(poisoned)).select(
        A6000, Shape(4, 4, 2048, 128, 4),
        NumericalContract.quantization_aware(),
        {"torch_fp32", "triton_tf32"},
    )
    assert rejected.backend == "torch_fp32", rejected
    print("6/6 dispatcher tests passed")


if __name__ == "__main__":
    run()
