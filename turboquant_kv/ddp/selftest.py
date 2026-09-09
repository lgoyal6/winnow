"""Run the harness as two gloo processes in-process, for the test suite.

`train_ddp.py` is launched by torchrun, which is right for a benchmark and
wrong for a unit test: a test that shells out to torchrun cannot assert on the
run record, only on an exit code. This module spawns the same loop with
`torch.multiprocessing.spawn` on a free port and hands rank 0's record back,
so the tests can assert on the checksums themselves.

Same boundary as everywhere else: gloo on CPU, two processes, one host. This
verifies harness correctness. It is not a multi-GPU result and produces no
performance claim.
"""
from __future__ import annotations

import socket
from dataclasses import asdict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ddp.loop import RunSpec, train


def free_port() -> int:
    """An ephemeral port the OS just told us is free.

    There is an unavoidable race between closing this socket and the
    rendezvous binding it. Preferable to a fixed port, which would collide
    with any other distributed job on the machine, including another copy of
    this test suite.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _worker(rank: int, world_size: int, port: int, spec_fields: dict,
            queue) -> None:
    dist.init_process_group(
        backend="gloo", init_method=f"tcp://127.0.0.1:{port}",
        rank=rank, world_size=world_size)
    try:
        record = train(RunSpec(**spec_fields), rank, world_size,
                       torch.device("cpu"))
        if rank == 0:
            queue.put(record)
    finally:
        dist.destroy_process_group()


def gloo_pair(spec: RunSpec, world_size: int = 2) -> dict:
    """Run `spec` across `world_size` gloo processes; return rank 0's record."""
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    mp.spawn(_worker,
             args=(world_size, free_port(), asdict(spec), queue),
             nprocs=world_size, join=True)
    return queue.get()
