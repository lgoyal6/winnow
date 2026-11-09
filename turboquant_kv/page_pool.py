"""Request-scoped page pool for packed KV, with fenced reclamation.

One pool backs every layer: page id p addresses `k_pages[layer, p]` for all
layers, so a request allocates pages once per token span, not once per layer.

Ownership is tracked per page by (owner, generation). A request holds
`PageRef(page, generation)` values. Releasing a request bumps the generation of
each page immediately, so any later attempt to launch with the old refs fails
`validate` instead of reading another request's tokens.

Bumping the generation does not make the page reusable. Kernels already queued
on the GPU may still be reading it, so a released page waits in `pending` with
the fence event recorded after the last launch that referenced it, and only
returns to the free list when that event has completed. On CPU every fence is
already complete; on CUDA it is a `torch.cuda.Event`. Tests inject an event
that stays incomplete to prove the pool waits.

Admission is decided here too: `allocate` refuses (raises `PoolExhausted`)
rather than evicting, and a single request is capped at `max_pages_per_request`.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

import torch


class PoolExhausted(RuntimeError):
    """Not enough free pages; the caller should refuse or queue the request."""


class StalePage(RuntimeError):
    """A launch referenced a page its caller no longer owns."""


@dataclass(frozen=True)
class PageRef:
    page: int
    generation: int


class _Done:
    """Fence for work that has already finished (CPU execution)."""

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        pass


def default_fence(device: torch.device):
    if device.type == "cuda":
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream(device))
        return ev
    return _Done()


class PagePool:
    def __init__(self, *, num_layers: int, num_pages: int, num_kv_heads: int,
                 page_size: int, nb_k: int, nb_v: int, device="cpu",
                 max_pages_per_request: int | None = None, fence_factory=None,
                 dense_dtype: torch.dtype | None = None):
        """Packed pages are uint8 rows of nb bytes plus fp16 norms. With
        `dense_dtype`, pages instead hold unquantized rows of nb elements of that
        dtype and there are no norms: the matched unquantized baseline."""
        self.device = torch.device(device)
        self.num_layers, self.num_pages, self.page_size = num_layers, num_pages, page_size
        self.dense = dense_dtype is not None
        shape = (num_layers, num_pages, num_kv_heads, page_size)
        dt = dense_dtype or torch.uint8
        self.k_pages = torch.zeros(*shape, nb_k, dtype=dt, device=self.device)
        self.v_pages = torch.zeros(*shape, nb_v, dtype=dt, device=self.device)
        self.k_norms = None if self.dense else torch.zeros(*shape, dtype=torch.float16, device=self.device)
        self.v_norms = None if self.dense else torch.zeros(*shape, dtype=torch.float16, device=self.device)
        self.max_pages_per_request = max_pages_per_request or num_pages
        self._fence = fence_factory or (lambda: default_fence(self.device))
        self._gen = [0] * num_pages
        self._owner: list[str | None] = [None] * num_pages
        self._free = deque(range(num_pages))
        self._pending: list[tuple[int, object]] = []
        self._held: dict[str, list[PageRef]] = {}
        self._lock = threading.Lock()

    # -- accounting -------------------------------------------------------
    def bytes(self) -> int:
        return sum(t.numel() * t.element_size()
                   for t in (self.k_pages, self.v_pages, self.k_norms, self.v_norms)
                   if t is not None)

    def bytes_per_token(self) -> int:
        """Resident KV bytes one token costs across all layers."""
        return self.bytes() // (self.num_pages * self.page_size)

    def stats(self) -> dict:
        with self._lock:
            return {"free": len(self._free), "pending": len(self._pending),
                    "owned": sum(len(v) for v in self._held.values()),
                    "requests": len(self._held), "total": self.num_pages}

    def held(self, owner: str) -> list[PageRef]:
        with self._lock:
            return list(self._held.get(owner, ()))

    # -- lifecycle --------------------------------------------------------
    def pages_for(self, tokens: int) -> int:
        return (tokens + self.page_size - 1) // self.page_size

    def ensure(self, owner: str, tokens: int) -> list[PageRef]:
        """Grow `owner`'s allocation to hold `tokens` tokens; all-or-nothing."""
        with self._lock:
            refs = self._held.setdefault(owner, [])
            need = self.pages_for(tokens) - len(refs)
            if need <= 0:
                return list(refs)
            if len(refs) + need > self.max_pages_per_request:
                raise PoolExhausted(
                    f"{owner} needs {len(refs) + need} pages > per-request cap "
                    f"{self.max_pages_per_request}")
            self._reclaim_locked()
            if need > len(self._free):
                raise PoolExhausted(
                    f"{owner} needs {need} more pages, {len(self._free)} free, "
                    f"{len(self._pending)} awaiting fences")
            for _ in range(need):
                p = self._free.popleft()
                self._owner[p] = owner
                refs.append(PageRef(p, self._gen[p]))
            return list(refs)

    def release(self, owner: str, fence=None) -> int:
        """Revoke `owner`'s pages now; recycle them once `fence` completes.

        `fence` must be recorded after the last launch that read these pages.
        When omitted, one is recorded on the current stream here.
        """
        fence = fence if fence is not None else self._fence()
        with self._lock:
            refs = self._held.pop(owner, [])
            for r in refs:
                self._gen[r.page] += 1
                self._owner[r.page] = None
                self._pending.append((r.page, fence))
            self._reclaim_locked()
            return len(refs)

    def reclaim(self) -> int:
        with self._lock:
            return self._reclaim_locked()

    def _event_done(self, fence) -> bool:
        return fence.query()

    def _reclaim_locked(self) -> int:
        keep, done = [], 0
        for page, fence in self._pending:
            if self._event_done(fence):
                self._free.append(page)
                done += 1
            else:
                keep.append((page, fence))
        self._pending = keep
        return done

    def validate(self, owner: str, refs) -> None:
        """Raise unless every ref is currently owned by `owner` at its generation."""
        with self._lock:
            for r in refs:
                if self._owner[r.page] != owner or self._gen[r.page] != r.generation:
                    raise StalePage(
                        f"{owner} referenced page {r.page} gen {r.generation}; "
                        f"current owner {self._owner[r.page]!r} gen {self._gen[r.page]}")
