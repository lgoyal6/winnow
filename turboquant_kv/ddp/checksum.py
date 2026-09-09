"""Cross-rank parameter checksums, and a sampled divergence probe.

A data-parallel run that has been synchronizing correctly ends with every
rank holding BIT-IDENTICAL parameters. That is a strong, cheap, falsifiable
property, and it is the only thing standing between "the throughput number is
real" and "the throughput number is what you get when one rank stops
all-reducing". A faster run that has silently stopped averaging gradients is
not a faster run, it is a different and wrong computation.

Two checksums are computed rather than one, because they fail differently:

  * `float64_sum` casts every parameter to float64 and sums. It catches a
    numeric change of any magnitude that survives float64 accumulation, and
    its value is human-comparable across ranks in a log line.
  * `bytes_sha256` hashes the raw parameter bytes in name-sorted order. It is
    exact: any single flipped bit changes it, including a change too small for
    the float64 sum to resolve, and including a NaN that would make arithmetic
    comparison useless.

Equality of both is required. The pair is deliberately redundant.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ParameterChecksum:
    float64_sum: float
    bytes_sha256: str
    parameter_count: int
    tensor_count: int
    dtypes: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "float64_sum": self.float64_sum,
            "bytes_sha256": self.bytes_sha256,
            "parameter_count": self.parameter_count,
            "tensor_count": self.tensor_count,
            "dtypes": list(self.dtypes),
        }

    def matches(self, other: "ParameterChecksum") -> bool:
        return (self.bytes_sha256 == other.bytes_sha256
                and self.float64_sum == other.float64_sum
                and self.parameter_count == other.parameter_count
                and self.tensor_count == other.tensor_count)


def _unwrap(module: nn.Module) -> nn.Module:
    """Reach the wrapped module so DDP's `module.` name prefix is not hashed."""
    inner = getattr(module, "module", None)
    return inner if isinstance(inner, nn.Module) else module


def parameter_checksum(module: nn.Module) -> ParameterChecksum:
    total = 0.0
    digest = hashlib.sha256()
    count = 0
    tensors = 0
    dtypes: set[str] = set()
    target = _unwrap(module)
    # Sorted by name so that the hash does not depend on registration order,
    # and so a DDP-wrapped module and a bare one hash identically.
    for name, param in sorted(target.named_parameters(), key=lambda kv: kv[0]):
        flat = param.detach().to("cpu").contiguous()
        total += float(flat.to(torch.float64).sum().item())
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(flat.shape)).encode("utf-8"))
        digest.update(str(flat.dtype).encode("utf-8"))
        # view(uint8) is an exact reinterpretation of the stored bytes and
        # works for bf16, which numpy cannot represent at all.
        digest.update(flat.view(torch.uint8).numpy().tobytes())
        count += flat.numel()
        tensors += 1
        dtypes.add(str(flat.dtype))
    return ParameterChecksum(
        float64_sum=total,
        bytes_sha256=digest.hexdigest(),
        parameter_count=count,
        tensor_count=tensors,
        dtypes=tuple(sorted(dtypes)),
    )


def sampled_parameter_vector(module: nn.Module, per_tensor: int = 8
                             ) -> torch.Tensor:
    """A small fixed slice of the parameters, for an all-gather divergence check.

    The checksum answers "identical or not". This answers "by how much", which
    is what tells a real desynchronization apart from a hypothetical
    last-bit-of-accumulation artifact: a missing all-reduce moves parameters by
    optimizer-step magnitudes, not by one ULP.
    """
    target = _unwrap(module)
    chunks = []
    for _, param in sorted(target.named_parameters(), key=lambda kv: kv[0]):
        flat = param.detach().reshape(-1)
        chunks.append(flat[:per_tensor].to(torch.float32))
    return torch.cat(chunks)


def perturb_one_element(module: nn.Module, delta: float = 1e-3) -> None:
    """Move exactly one parameter element. Test support for the checksums.

    Used only by the tests, to prove the checksum is actually sensitive rather
    than merely present.
    """
    target = _unwrap(module)
    _, first_param = sorted(
        target.named_parameters(), key=lambda kv: kv[0])[0]
    with torch.no_grad():
        first_param.reshape(-1)[0] += delta
