"""A DDP communication hook that times each bucket's all-reduce, and the
test-only gradient-sync fault that proves the checksum gate can fail.

Why a hook rather than a wall-clock difference: DDP overlaps each bucket's
all-reduce with the rest of the backward pass on purpose, so "step time minus
compute time" is not communication time, it is exposed communication time
plus measurement error. Registering a hook is the only place where the
collective itself can be bracketed.

Read the reported communication percentage as an UPPER BOUND on exposed
communication, not as a stall: the per-bucket intervals summed here overlap
one another and overlap backward compute by design. A run whose comm
percentage is 40% is not necessarily 40% idle.

Timing source, chosen per device and recorded in every artifact:

  * CUDA: `torch.cuda.Event` pairs, so the number is device time. A
    `perf_counter` around an async collective on a GPU measures enqueue cost
    and would look absurdly cheap.
  * CPU (gloo): `time.perf_counter`, which is the real thing there.

The fault injection, and one honest deviation from the naive design
=================================================================

`--inject-grad-sync-fault` makes rank 1 apply its LOCAL, unreduced gradient
for exactly one bucket. The naive implementation is to return early from the
hook on that rank without calling `all_reduce`. That deadlocks: the other
ranks are already inside a matched collective and would block forever, so the
run would hang rather than fail, and a hang is not a negative control.

So the injected rank still JOINS the collective (nothing can hang), and then
discards the reduced result and returns its own local gradient copy for that
one bucket. The observable defect is exactly the specified one: on rank 1, one
bucket's gradient is never averaged, the optimizer step differs, and the
parameters diverge. This is test-only and off unless asked for.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist


@dataclass
class CommTimingState:
    """Per-step accumulator for the hook. One instance per process."""

    world_size: int
    rank: int
    use_cuda_events: bool
    # Fault injection: off unless inject is True. Test-only.
    inject: bool = False
    fault_rank: int = 1
    fault_bucket: int = 0

    _cpu_seconds: float = 0.0
    _events: list[tuple[Any, Any]] = field(default_factory=list)
    bucket_calls: int = 0
    faulted_bucket_calls: int = 0

    def reset_step(self) -> None:
        self._cpu_seconds = 0.0
        self._events.clear()
        self.bucket_calls = 0

    def step_comm_seconds(self) -> float:
        """Call after the step has been synchronized."""
        if not self.use_cuda_events:
            return self._cpu_seconds
        total_ms = 0.0
        for start, end in self._events:
            total_ms += start.elapsed_time(end)
        return total_ms / 1000.0

    def should_fault(self, bucket_index: int) -> bool:
        return (self.inject
                and self.rank == self.fault_rank
                and bucket_index == self.fault_bucket)


def _bucket_index(bucket) -> int:
    # GradBucket.index() is the documented accessor; fall back rather than
    # crash if a torch build renames it, since the timing is not worth a
    # hard failure.
    getter = getattr(bucket, "index", None)
    return int(getter()) if callable(getter) else -1


# `bucket` is deliberately left unannotated. DDP's `_check_comm_hook` rejects
# any annotation that is not literally the `dist.GradBucket` class object, and
# this module uses `from __future__ import annotations`, which turns every
# annotation into a string. An empty annotation is the only spelling torch
# accepts here; the parameter is a `dist.GradBucket`.
def timing_allreduce_hook(state: CommTimingState, bucket):
    """Average gradients across ranks, timing each bucket's collective.

    Semantically identical to torch's built-in `allreduce_hook` (all-reduce
    then divide by world size), plus timing, plus the test-only fault.
    """
    tensor = bucket.buffer()
    index = _bucket_index(bucket)
    faulting = state.should_fault(index)

    local_copy = tensor.detach().clone() if faulting else None

    if state.use_cuda_events:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_event = end_event = None
        cpu_start = time.perf_counter()

    work = dist.all_reduce(tensor, async_op=True)
    state.bucket_calls += 1

    def finish(fut):
        reduced = fut.value()[0] if isinstance(fut.value(), list) else fut.value()
        if state.use_cuda_events:
            end_event.record()
            state._events.append((start_event, end_event))
        else:
            state._cpu_seconds += time.perf_counter() - cpu_start
        if faulting:
            # Test-only: throw away the average, keep this rank's own gradient.
            state.faulted_bucket_calls += 1
            return reduced.copy_(local_copy)
        return reduced.div_(state.world_size)

    return work.get_future().then(finish)


def describe_fault(state: CommTimingState) -> dict:
    return {
        "enabled": state.inject,
        "fault_rank": state.fault_rank,
        "fault_bucket": state.fault_bucket,
        "faulted_bucket_calls": state.faulted_bucket_calls,
        "mechanism": ("on the injected rank only, one DDP bucket's reduced "
                      "gradient is discarded and the rank's local gradient is "
                      "applied instead; the collective is still joined so no "
                      "rank can hang"),
    }
