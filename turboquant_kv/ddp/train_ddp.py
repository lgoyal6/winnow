"""torchrun entrypoint for the data-parallel harness.

Usage (world size N, one process per device):

    torchrun --nproc_per_node 2 turboquant_kv/ddp/train_ddp.py \
        --out turboquant_kv/results/runs/2gpu-repeat1.json

Exit codes, which the wrapper scripts depend on:

    0  every rank ended with bit-identical parameters
    1  the cross-rank checksum or the divergence probe FAILED, which is the
       expected outcome of --inject-grad-sync-fault
    2  the harness refused to run (bad launch environment)

The 1-process configuration deliberately does NOT wrap the model in DDP.
That is what a single-device training run actually is, and it is the honest
baseline for a scaling comparison: charging the baseline for a self-all-reduce
that a real single-GPU job would never issue would flatter the 2-device arm.

Nothing in this file downloads anything, and it does not require a GPU. On a
host without CUDA it runs on CPU with the gloo backend, which is a
CORRECTNESS SELF-TEST OF THIS HARNESS AND NOTHING ELSE. Two processes on one
CPU are not two GPUs, are not multi-GPU, and any timing they produce is not a
scaling or throughput claim.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch                                              # noqa: E402
import torch.distributed as dist                          # noqa: E402

from ddp.loop import RunSpec, train                       # noqa: E402
from ddp.model import PRESETS                             # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = RunSpec()
    parser = argparse.ArgumentParser(
        description="DDP scaling harness for a synthetic decoder-only proxy")
    parser.add_argument("--preset", default=defaults.preset,
                        choices=sorted(PRESETS))
    parser.add_argument("--global-batch-size", type=int,
                        default=defaults.global_batch_size)
    parser.add_argument("--seq-len", type=int, default=defaults.seq_len)
    parser.add_argument("--warmup-steps", type=int,
                        default=defaults.warmup_steps)
    parser.add_argument("--measured-steps", type=int,
                        default=defaults.measured_steps)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--bucket-cap-mb", type=float,
                        default=defaults.bucket_cap_mb)
    parser.add_argument("--label", default=defaults.label)
    parser.add_argument("--out", default=None,
                        help="rank 0 writes the run record here as JSON")
    parser.add_argument(
        "--inject-grad-sync-fault", action="store_true",
        help=("TEST ONLY negative control: the injected rank applies its "
              "local, unreduced gradient for exactly one bucket. The run MUST "
              "then fail the cross-rank checksum and exit 1."))
    parser.add_argument("--fault-rank", type=int, default=defaults.fault_rank)
    parser.add_argument("--fault-bucket", type=int,
                        default=defaults.fault_bucket)
    return parser.parse_args(argv)


def resolve_device(local_rank: int) -> tuple[torch.device, str]:
    """Pick the device and the matching collective backend."""
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank), "nccl"
    return torch.device("cpu"), "gloo"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for required in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        if required not in os.environ:
            print(f"REFUSING: {required} is not set. Launch this with "
                  f"`torchrun --nproc_per_node N`, not directly.",
                  file=sys.stderr)
            return 2

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if args.global_batch_size % world_size != 0:
        print(f"REFUSING: global batch size {args.global_batch_size} is not "
              f"divisible by world size {world_size}; the fixed-global-work "
              f"comparison would not hold.", file=sys.stderr)
        return 2
    if args.inject_grad_sync_fault and world_size < 2:
        print("REFUSING: --inject-grad-sync-fault needs world size >= 2; with "
              "one rank there is no gradient synchronization to break.",
              file=sys.stderr)
        return 2

    device, backend = resolve_device(local_rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    try:
        spec = RunSpec(
            preset=args.preset,
            global_batch_size=args.global_batch_size,
            seq_len=args.seq_len,
            warmup_steps=args.warmup_steps,
            measured_steps=args.measured_steps,
            seed=args.seed,
            bucket_cap_mb=args.bucket_cap_mb,
            inject_grad_sync_fault=args.inject_grad_sync_fault,
            fault_rank=args.fault_rank,
            fault_bucket=args.fault_bucket,
            label=args.label,
        )
        record = train(spec, rank, world_size, device)
        record["mode"] = (
            "gpu-physical-devices" if device.type == "cuda"
            else "cpu-gloo-single-host-selftest")
        record["boundary"] = (
            "measured on physical CUDA devices, one process per device"
            if device.type == "cuda" else
            "CPU gloo self-test: two or more processes on ONE host with no "
            "GPU. This verifies harness correctness only. It is not a "
            "multi-GPU result and its timings are not a scaling or "
            "throughput claim.")

        if rank == 0:
            print(f"[rank 0] {record['verdict']}")
            print(f"[rank 0] world_size={world_size} device={device} "
                  f"precision={record['precision']} "
                  f"backend={record['backend']}")
            print(f"[rank 0] throughput="
                  f"{record['throughput_tokens_per_second']:.1f} tok/s "
                  f"median_step={record['step_latency_median_seconds']*1e3:.3f} ms "
                  f"comm={record['comm_percent_of_step_time']:.2f}%")
            print(f"[rank 0] mode={record['mode']}")
            if args.out:
                os.makedirs(os.path.dirname(os.path.abspath(args.out)),
                            exist_ok=True)
                with open(args.out, "w") as handle:
                    json.dump(record, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                print(f"[rank 0] wrote {args.out}")
        if not record["passed"]:
            print(f"[rank {rank}] {record['verdict']}", file=sys.stderr)
        return 0 if record["passed"] else 1
    finally:
        # No barrier here on purpose: if one rank raised, a barrier would turn
        # a visible traceback into a hang, and a hang is not a test result.
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
