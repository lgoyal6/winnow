"""Data-parallel training harness for a decoder-only proxy workload.

This package exists to answer one question with evidence rather than
assertion: does adding a second physical GPU to a fixed amount of training
work make that work finish faster, and by how much, once gradient
synchronization is paid for.

Scope and honesty boundary, stated once here and repeated in every artifact
this package writes:

  * The model is a compact causal LM defined entirely in this repository. It
    is a SYNTHETIC-WORKLOAD PROXY for the decoder-only models Winnow serves,
    not one of those models. Nothing is downloaded.
  * The token stream is deterministic synthetic noise. Throughput measured on
    it describes the training loop and the gradient all-reduce; it says
    nothing about model quality, and no loss value here is a quality claim.
  * `run-ddp-cpu-selftest.sh` runs two processes on ONE host with the gloo
    backend on CPU. That is a correctness self-test of this harness. Two
    processes on one CPU are never a multi-GPU result and never a scaling or
    throughput claim.
  * `run-multigpu-gate.sh` is the only path that produces a scaling number,
    and it refuses to run unless `nvidia-smi` reports at least two physical
    GPUs with distinct UUIDs and distinct PCI bus ids.
"""
from __future__ import annotations

__all__ = [
    "checksum",
    "data",
    "inventory",
    "loop",
    "model",
]
