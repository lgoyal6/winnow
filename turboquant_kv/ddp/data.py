"""Deterministic synthetic token stream.

The point of a synthetic stream in a scaling measurement is that it removes
the dataloader from the result. There is no disk, no shuffle buffer, no
tokenizer, and no host-to-device staging queue whose depth could differ
between the 1-GPU and 2-GPU arms; a batch is a pure function of
(seed, step, global_batch_size, seq_len). Whatever difference the second GPU
makes is therefore attributable to compute and gradient synchronization.

What that costs in interpretation, stated plainly: these tokens are noise
drawn from a fixed vocabulary. Throughput measured on them describes the
training loop and the all-reduce. It says NOTHING about model quality, and
the loss values the harness records are not a quality signal. They are
recorded only so that a run which silently stopped learning (a dead
optimizer, an all-reduce that zeroed gradients) is visible.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SyntheticStream:
    """Fixed-shape batches, reproducible from the seed and the step index.

    The GLOBAL batch for a step is generated identically on every rank and
    then sliced by rank. Two consequences that the harness depends on:

      1. Global work per step is held fixed across world sizes, which is what
         makes `throughput_2 / (2 * throughput_1)` a scaling efficiency rather
         than a comparison of two different amounts of work.
      2. Each rank sees a DIFFERENT slice, so ranks compute different local
         gradients. Without that, a missing all-reduce would be undetectable:
         averaging identical gradients is a no-op, and the planted
         gradient-sync fault would produce no divergence to catch.
    """

    seed: int
    vocab_size: int
    global_batch_size: int
    seq_len: int

    def __post_init__(self) -> None:
        if self.global_batch_size <= 0 or self.seq_len < 2:
            raise ValueError("need global_batch_size > 0 and seq_len >= 2")

    def per_rank_batch_size(self, world_size: int) -> int:
        if self.global_batch_size % world_size != 0:
            raise ValueError(
                f"global_batch_size {self.global_batch_size} is not divisible "
                f"by world_size {world_size}; the fixed-global-work comparison "
                f"would not hold")
        return self.global_batch_size // world_size

    def global_tokens_per_step(self) -> int:
        # seq_len - 1 supervised positions: the last token has no target.
        return self.global_batch_size * (self.seq_len - 1)

    def _step_generator(self, step: int) -> torch.Generator:
        # A per-step generator rather than one advanced in place: the batch for
        # step k must not depend on how many steps preceded it, so a 1-GPU run
        # and a 2-GPU run see the identical global stream.
        gen = torch.Generator(device="cpu")
        gen.manual_seed((self.seed * 1_000_003 + step) % (2 ** 63 - 1))
        return gen

    def global_batch(self, step: int) -> torch.Tensor:
        return torch.randint(
            0, self.vocab_size,
            (self.global_batch_size, self.seq_len),
            generator=self._step_generator(step), dtype=torch.long)

    def rank_batch(self, step: int, rank: int, world_size: int
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (inputs, targets) for this rank: next-token prediction."""
        per_rank = self.per_rank_batch_size(world_size)
        rows = self.global_batch(step)[rank * per_rank:(rank + 1) * per_rank]
        return rows[:, :-1].contiguous(), rows[:, 1:].contiguous()

    def describe(self) -> dict:
        return {
            "generator": "torch.randint over a per-step seeded CPU Generator",
            "seed": self.seed,
            "vocab_size": self.vocab_size,
            "global_batch_size": self.global_batch_size,
            "seq_len": self.seq_len,
            "global_tokens_per_step": self.global_tokens_per_step(),
            "step_seed_rule": "(seed * 1000003 + step) mod (2**63 - 1)",
            "sharding": ("the global batch is generated identically on every "
                         "rank and sliced by rank, holding global work fixed "
                         "across world sizes"),
            "caveat": ("synthetic noise tokens: throughput here measures the "
                       "training loop and the gradient all-reduce, not model "
                       "quality; recorded loss values are not a quality claim"),
        }
