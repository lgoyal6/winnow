"""Two-process gloo tests for the DDP harness.

These are the real gate on the harness's central claim: ranks that
synchronize gradients end with bit-identical parameters, and the planted
gradient-sync fault breaks that in a way the checksum catches.

CPU gloo, two processes, one host. Not a multi-GPU result, and nothing here
is timed as a performance claim.
"""
from __future__ import annotations

from dataclasses import replace

from ddp.loop import RunSpec
from ddp.selftest import gloo_pair

# Small enough to run in seconds, with bucket_cap_mb low so the model spans
# several DDP buckets: "the fault hits exactly one bucket" is only a real
# statement when there is more than one bucket.
SPEC = RunSpec(
    preset="tiny-2l-128d",
    global_batch_size=8,
    seq_len=32,
    warmup_steps=1,
    measured_steps=4,
    bucket_cap_mb=0.5,
    label="pytest-gloo",
)
FAULTED = replace(SPEC, inject_grad_sync_fault=True, label="pytest-gloo-fault")


def test_gloo_pair_ends_with_identical_parameters():
    record = gloo_pair(SPEC)
    assert record["initial_checksums_match_across_ranks"], record["verdict"]
    assert record["final_checksums_match_across_ranks"], record["verdict"]
    assert record["max_cross_rank_parameter_divergence"] == 0.0, record
    assert record["passed"], record["verdict"]


def test_gloo_pair_records_both_ranks_and_the_comm_hook_fired():
    record = gloo_pair(SPEC)
    assert len(record["per_rank_metrics"]) == 2, record["per_rank_metrics"]
    assert record["bucket_calls_last_step"] >= 2, record
    assert record["comm_timing_source"] == "time.perf_counter"
    assert record["precision"] == "fp32"
    assert record["backend"] == "gloo"
    assert len(record["step_latency_seconds"]) == SPEC.measured_steps
    assert record["comm_seconds_total"] > 0.0, record


def test_injected_grad_sync_fault_is_caught():
    record = gloo_pair(FAULTED)
    # The fault must actually have fired: a control that did not fire proves
    # nothing about the gate.
    assert record["fault_actually_applied"], record["fault_injection_per_rank"]
    assert record["fault_applied_bucket_calls_total"] > 0, record
    # And it must be caught.
    assert record["initial_checksums_match_across_ranks"], record
    assert not record["final_checksums_match_across_ranks"], record
    assert record["max_cross_rank_parameter_divergence"] > 0.0, record
    assert not record["passed"], record["verdict"]
    assert "diverged" in record["verdict"]


def test_only_the_injected_rank_faults():
    record = gloo_pair(FAULTED)
    counts = [entry["faulted_bucket_calls"]
              for entry in record["fault_injection_per_rank"]]
    assert sum(1 for count in counts if count > 0) == 1, counts
