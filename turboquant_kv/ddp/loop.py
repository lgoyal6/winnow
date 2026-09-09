"""The measured DistributedDataParallel training loop.

Design decisions worth stating, because each one is the difference between a
number that means something and a number that does not.

**DDP, not FSDP.** The proxy model is about 35.8M parameters, roughly 143 MB
of fp32 gradients. It fits many times over on any card that can run this
harness, so there is no memory pressure for FSDP's parameter sharding to
relieve, and sharding would add collectives to the forward pass that have
nothing to do with the question being asked. DDP is the right tool for a
replicated-model, sharded-data scaling measurement at this size. FSDP would be
the right tool at a size where the model does not fit, which this is not.

**Global work is held fixed.** The per-rank batch is global_batch_size /
world_size. Both arms therefore train on exactly the same tokens per step, so
`throughput_2 / (2 * throughput_1)` is a scaling efficiency. Holding the
PER-RANK batch fixed instead would double the work at world size 2 and turn
the same ratio into a weak-scaling number that flatters the harness.

**Every rank starts from identical parameters, and this is verified rather
than assumed.** The seed is set before construction so all ranks build the
same tensors, and then a cross-rank checksum is compared BEFORE the first
step. A run whose ranks started differently would report a plausible
throughput and a meaningless result, so it is refused instead.

**Every rank must end with identical parameters.** The same checksum is
compared after the measured steps, plus an all-gathered sampled slice giving
the magnitude of any divergence. This is the gate the planted
gradient-sync fault is designed to trip.

**5 warmup steps precede any timing.** The first steps pay for CUDA context,
allocator growth, autotuning and the first bucket rebuild. Timing them would
measure startup.
"""
from __future__ import annotations

import os
import platform
import resource
import statistics
import sys
import time
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from ddp.checksum import parameter_checksum, sampled_parameter_vector
from ddp.comm import CommTimingState, describe_fault, timing_allreduce_hook
from ddp.data import SyntheticStream
from ddp.model import PRESETS, build_model, describe


@dataclass(frozen=True)
class RunSpec:
    """Everything the frozen manifest pins, in one immutable object."""

    preset: str = "proxy-6l-512d"
    global_batch_size: int = 32
    seq_len: int = 512
    warmup_steps: int = 5
    measured_steps: int = 30
    seed: int = 1234
    lr: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    bucket_cap_mb: float = 25.0
    divergence_sample_per_tensor: int = 8
    inject_grad_sync_fault: bool = False
    fault_rank: int = 1
    fault_bucket: int = 0
    label: str = "unlabelled"

    def as_dict(self) -> dict:
        return asdict(self)


def peak_host_rss_bytes() -> int:
    """ru_maxrss, normalized. Darwin reports bytes, Linux reports kilobytes."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw) if sys.platform == "darwin" else int(raw) * 1024


def _gather_objects(obj, world_size: int) -> list:
    if world_size == 1:
        return [obj]
    bucket = [None] * world_size
    dist.all_gather_object(bucket, obj)
    return bucket


def _max_cross_rank_divergence(model: nn.Module, spec: RunSpec,
                               world_size: int, device: torch.device) -> float:
    """Largest absolute parameter difference between any rank and rank 0."""
    local = sampled_parameter_vector(
        model, spec.divergence_sample_per_tensor).to(device)
    if world_size == 1:
        return 0.0
    gathered = [torch.zeros_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    worst = 0.0
    for other in gathered[1:]:
        worst = max(worst, float((other - gathered[0]).abs().max().item()))
    return worst


def train(spec: RunSpec, rank: int, world_size: int,
          device: torch.device) -> dict:
    """Run the harness in an already-initialized process group.

    Returns the per-run record. Never calls sys.exit, so the same function
    serves `train_ddp.py` (which turns the record into an exit code) and the
    pytest gloo self-test (which asserts on it).
    """
    on_cuda = device.type == "cuda"
    cfg = PRESETS[spec.preset]

    if on_cuda:
        precision = ("bf16-autocast" if torch.cuda.is_bf16_supported()
                     else "fp32-no-autocast")
    else:
        precision = "fp32"
    autocast_dtype = torch.bfloat16 if precision == "bf16-autocast" else None

    model = build_model(cfg, spec.seed).to(device)
    initial = parameter_checksum(model)
    initial_all = _gather_objects(initial.as_dict(), world_size)
    initial_match = all(entry == initial_all[0] for entry in initial_all)

    ddp_kwargs = {"bucket_cap_mb": spec.bucket_cap_mb}
    if on_cuda:
        ddp_kwargs["device_ids"] = [device.index]
    ddp = DistributedDataParallel(model, **ddp_kwargs) if world_size > 1 else model

    comm_state = CommTimingState(
        world_size=world_size, rank=rank, use_cuda_events=on_cuda,
        inject=spec.inject_grad_sync_fault, fault_rank=spec.fault_rank,
        fault_bucket=spec.fault_bucket)
    if world_size > 1:
        ddp.register_comm_hook(comm_state, timing_allreduce_hook)

    optimizer = torch.optim.AdamW(
        ddp.parameters(), lr=spec.lr, betas=(spec.beta1, spec.beta2),
        eps=spec.eps, weight_decay=spec.weight_decay)
    stream = SyntheticStream(
        seed=spec.seed, vocab_size=cfg.vocab_size,
        global_batch_size=spec.global_batch_size, seq_len=spec.seq_len)
    per_rank_batch = stream.per_rank_batch_size(world_size)

    def one_step(step: int) -> tuple[float, float]:
        """Returns (loss, comm_seconds). Timing is done by the caller."""
        tokens, targets = stream.rank_batch(step, rank, world_size)
        tokens, targets = tokens.to(device), targets.to(device)
        comm_state.reset_step()
        optimizer.zero_grad(set_to_none=True)
        if autocast_dtype is not None:
            with torch.autocast("cuda", dtype=autocast_dtype):
                loss = ddp_loss(ddp, tokens, targets)
        else:
            loss = ddp_loss(ddp, tokens, targets)
        loss.backward()
        if spec.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ddp.parameters(), spec.grad_clip)
        optimizer.step()
        return float(loss.detach().item()), comm_state.step_comm_seconds()

    for step in range(spec.warmup_steps):
        one_step(step)
    if on_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    if world_size > 1:
        dist.barrier()

    step_seconds: list[float] = []
    comm_seconds: list[float] = []
    losses: list[float] = []
    wall_start = time.perf_counter()
    for offset in range(spec.measured_steps):
        step = spec.warmup_steps + offset
        if on_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            loss, _ = one_step(step)
            end_event.record()
            torch.cuda.synchronize(device)
            step_seconds.append(start_event.elapsed_time(end_event) / 1000.0)
            comm_seconds.append(comm_state.step_comm_seconds())
        else:
            t0 = time.perf_counter()
            loss, comm = one_step(step)
            step_seconds.append(time.perf_counter() - t0)
            comm_seconds.append(comm)
        losses.append(loss)
    wall_seconds = time.perf_counter() - wall_start

    final = parameter_checksum(ddp)
    final_all = _gather_objects(final.as_dict(), world_size)
    final_match = all(entry == final_all[0] for entry in final_all)
    divergence = _max_cross_rank_divergence(ddp, spec, world_size, device)
    # Gathered across ranks because the fault fires on ONE rank and rank 0 is
    # the one that writes the artifact. A negative control whose fault never
    # actually fired must not be reportable as "the fault was caught".
    fault_all = _gather_objects(describe_fault(comm_state), world_size)
    faulted_calls_total = sum(int(entry["faulted_bucket_calls"])
                              for entry in fault_all)

    tokens_per_step = stream.global_tokens_per_step()
    tokens_per_second = [tokens_per_step / s for s in step_seconds]
    total_comm = sum(comm_seconds)
    total_step = sum(step_seconds)

    # Peak memory is a per-rank quantity, and on a real two-GPU host the two
    # ranks sit on different cards. Gathering it here means rank 0's single
    # artifact carries every rank's figure, instead of the aggregation having
    # to stitch together N files that may not all have been written.
    local_metrics = {
        "rank": rank,
        "device": str(device),
        "device_name": (torch.cuda.get_device_name(device) if on_cuda
                        else platform.processor() or platform.machine()),
        "peak_memory_bytes": (torch.cuda.max_memory_allocated(device)
                              if on_cuda else peak_host_rss_bytes()),
        "step_latency_median_seconds": statistics.median(step_seconds),
        "comm_seconds_total": total_comm,
        "comm_percent_of_step_time": (
            100.0 * total_comm / total_step if total_step > 0 else 0.0),
    }
    per_rank_metrics = sorted(
        _gather_objects(local_metrics, world_size), key=lambda m: m["rank"])

    record = {
        "label": spec.label,
        "spec": spec.as_dict(),
        "model": describe(cfg),
        "data": stream.describe(),
        "world_size": world_size,
        "rank": rank,
        "device": str(device),
        "device_name": (torch.cuda.get_device_name(device) if on_cuda
                        else platform.processor() or platform.machine()),
        "backend": dist.get_backend() if dist.is_initialized() else "none",
        "precision": precision,
        "per_rank_batch_size": per_rank_batch,
        "global_tokens_per_step": tokens_per_step,
        "measured_steps": spec.measured_steps,
        "warmup_steps": spec.warmup_steps,
        "step_latency_seconds": step_seconds,
        "step_latency_median_seconds": statistics.median(step_seconds),
        "step_latency_p95_seconds": _p95(step_seconds),
        "step_latency_mean_seconds": statistics.fmean(step_seconds),
        "tokens_per_second_median": statistics.median(tokens_per_second),
        "tokens_per_second_mean": statistics.fmean(tokens_per_second),
        "throughput_tokens_per_second": (
            tokens_per_step * spec.measured_steps / total_step),
        "comm_seconds_total": total_comm,
        "comm_percent_of_step_time": (
            100.0 * total_comm / total_step if total_step > 0 else 0.0),
        "comm_percent_note": ("upper bound on exposed communication: "
                              "per-bucket all-reduce intervals overlap each "
                              "other and overlap backward compute by design"),
        "comm_timing_source": ("torch.cuda.Event" if on_cuda
                               else "time.perf_counter"),
        "bucket_calls_last_step": comm_state.bucket_calls,
        "loss_first_measured": losses[0],
        "loss_last_measured": losses[-1],
        "loss_note": ("synthetic noise tokens: loss is recorded only so that "
                      "a run which stopped optimizing is visible, and is not "
                      "a model-quality claim"),
        "peak_memory_bytes": (torch.cuda.max_memory_allocated(device)
                              if on_cuda else peak_host_rss_bytes()),
        "peak_memory_source": (
            "torch.cuda.max_memory_allocated over the measured steps"
            if on_cuda else
            "resource.getrusage ru_maxrss: HOST RSS since process start, "
            "not a GPU allocator figure and not resettable"),
        "wall_seconds": wall_seconds,
        "per_rank_metrics": per_rank_metrics,
        "initial_checksum": initial.as_dict(),
        "initial_checksums_match_across_ranks": initial_match,
        "final_checksum": final.as_dict(),
        "final_checksums_match_across_ranks": final_match,
        "final_checksums_per_rank": final_all,
        "max_cross_rank_parameter_divergence": divergence,
        "fault_injection": describe_fault(comm_state),
        "fault_injection_per_rank": fault_all,
        "fault_applied_bucket_calls_total": faulted_calls_total,
        "fault_actually_applied": faulted_calls_total > 0,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "host_platform": platform.platform(),
        "pid": os.getpid(),
    }
    record["passed"] = bool(initial_match and final_match
                            and divergence == 0.0)
    record["verdict"] = _verdict(record)
    return record


def ddp_loss(module: nn.Module, tokens: torch.Tensor,
             targets: torch.Tensor) -> torch.Tensor:
    """Compute the loss through the DDP wrapper so hooks and buckets fire.

    Calling `model.module.loss(...)` would bypass DDP's forward and therefore
    bypass gradient synchronization entirely, which is exactly the bug this
    harness exists to detect. So the loss is computed from the wrapper's own
    forward output.
    """
    logits = module(tokens)
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1))


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    # Nearest-rank p95 on a 30-sample window; no interpolation, so the number
    # is an observed step time rather than a synthesized one.
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return ordered[index]


def _verdict(record: dict) -> str:
    if not record["initial_checksums_match_across_ranks"]:
        return ("REFUSED: ranks did not start from identical parameters, so no "
                "measurement here would mean anything")
    if not record["final_checksums_match_across_ranks"]:
        return ("FAILED: cross-rank parameter checksums diverged after the "
                "measured steps, so gradient synchronization did not hold")
    if record["max_cross_rank_parameter_divergence"] != 0.0:
        return ("FAILED: sampled parameters differ across ranks by "
                f"{record['max_cross_rank_parameter_divergence']:.6e}")
    return "PASSED: all ranks hold bit-identical parameters after the run"
