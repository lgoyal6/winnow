"""Continuous-batching decode engine over the paged packed cache.

One scheduler thread calls `step()`; any thread may `submit` or `cancel`.
Each step:

1. finishes requests that were cancelled or passed their deadline, releasing
   their pages behind a fence recorded after the last forward that read them;
2. admits queued requests while the batch has room and the pool can reserve a
   request's worst case (prompt + max_new_tokens) up front, so a running
   request can never hit an allocation failure mid-decode;
3. runs one prefill chunk for one request (B = 1, documented fallback path);
4. runs one batched decode forward for every request past prefill, each at its
   own position, through the compressed-domain operator.

Requests that can never fit (prompt or worst case beyond the per-request page
cap) and requests arriving at a full queue are refused at `submit`, with a
reason, rather than queued forever. Decoding is greedy, so a request's tokens
are a deterministic function of its prompt; concurrent output must equal the
same request run alone, which the tests check.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import torch

from packed_backend import PagedPackedCache
from page_pool import PoolExhausted


class Refused(RuntimeError):
    """Admission refused at submit; carries a machine-readable reason."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


@dataclass
class Request:
    request_id: str
    prompt_ids: list[int]
    max_new_tokens: int
    deadline: float | None = None            # time.monotonic() seconds
    on_event: object = None                  # callable(request, event: dict)
    ignore_eos: bool = False                 # load studies: fixed output length
    # -- engine-owned state --
    state: str = "queued"                    # queued, prefill, decode, done
    status: str = ""                         # ok, eos, length, cancelled, deadline, error
    detail: str = ""
    generated: list[int] = field(default_factory=list)
    stored: int = 0                          # tokens written to the cache
    submitted_at: float = 0.0
    first_token_at: float | None = None
    token_times: list[float] = field(default_factory=list)
    finished_at: float | None = None
    cancel_requested: bool = False
    done: threading.Event = field(default_factory=threading.Event)


@dataclass
class EngineConfig:
    max_batch: int = 8
    max_queue: int = 64
    prefill_chunk: int = 512
    max_prompt_tokens: int = 8192
    max_new_tokens_cap: int = 512
    decode_backend: str = "auto"


class Engine:
    def __init__(self, model, pool, k_quant, v_quant, head_dim: int,
                 eos_ids=(), config: EngineConfig | None = None, clock=time.monotonic):
        self.model, self.pool = model, pool
        self.kq, self.vq, self.head_dim = k_quant, v_quant, head_dim
        self.eos = set(int(e) for e in eos_ids)
        self.cfg = config or EngineConfig()
        self.clock = clock
        self.device = next(model.parameters()).device
        self._lock = threading.Lock()
        self._step_lock = threading.Lock()     # step() is single-threaded by contract
        self._queue: deque[Request] = deque()
        self._active: list[Request] = []
        self._by_id: dict[str, Request] = {}
        self.closed = False
        self.steps = 0
        self.on_logits = None                # test/diagnostic hook: (reqs, logits)

    # -- client side ------------------------------------------------------
    def submit(self, req: Request) -> Request:
        cfg = self.cfg
        if self.closed:
            raise Refused("shutting_down")
        if not req.prompt_ids:
            raise Refused("empty_prompt")
        if len(req.prompt_ids) > cfg.max_prompt_tokens:
            raise Refused("prompt_too_long", f"{len(req.prompt_ids)} > {cfg.max_prompt_tokens}")
        if not 1 <= req.max_new_tokens <= cfg.max_new_tokens_cap:
            raise Refused("bad_max_new_tokens", f"must be 1..{cfg.max_new_tokens_cap}")
        worst = self.pool.pages_for(len(req.prompt_ids) + req.max_new_tokens)
        if worst > self.pool.max_pages_per_request:
            raise Refused("exceeds_request_cap", f"{worst} pages > {self.pool.max_pages_per_request}")
        with self._lock:
            if req.request_id in self._by_id:
                raise Refused("duplicate_request_id")
            if len(self._queue) >= cfg.max_queue:
                raise Refused("queue_full", f"{cfg.max_queue} queued")
            req.submitted_at = self.clock()
            self._queue.append(req)
            self._by_id[req.request_id] = req
        return req

    def cancel(self, request_id: str) -> bool:
        with self._lock:
            req = self._by_id.get(request_id)
            if req is None or req.state == "done":
                return False
            req.cancel_requested = True
            return True

    def idle(self) -> bool:
        with self._lock:
            return not self._queue and not self._active

    def shutdown(self) -> None:
        """Refuse new work, finish everything outstanding as cancelled."""
        with self._lock:
            self.closed = True
            for r in list(self._queue) + list(self._active):
                r.cancel_requested = True
                r.detail = "server shutting down"
        self.step()

    # -- scheduler side ---------------------------------------------------
    def _emit(self, req, **event):
        if req.on_event is not None:
            req.on_event(req, event)

    def _finish(self, req: Request, status: str, detail: str = "", fence=None):
        req.state, req.status = "done", status
        req.detail = req.detail or detail
        req.finished_at = self.clock()
        self.pool.release(req.request_id, fence)
        with self._lock:
            self._by_id.pop(req.request_id, None)
        self._emit(req, kind="final", status=status, detail=req.detail)
        req.done.set()

    def _reap(self):
        now = self.clock()
        with self._lock:
            queued = list(self._queue)
        for req in queued + list(self._active):
            reason = ("cancelled" if req.cancel_requested else
                      "deadline" if req.deadline is not None and now > req.deadline else None)
            if reason is None:
                continue
            with self._lock:
                if req in self._queue:
                    self._queue.remove(req)
            if req in self._active:
                self._active.remove(req)
            self._finish(req, reason)

    def _admit(self):
        while len(self._active) < self.cfg.max_batch:
            with self._lock:
                if not self._queue:
                    return
                req = self._queue[0]
            try:
                self.pool.ensure(req.request_id, len(req.prompt_ids) + req.max_new_tokens)
            except PoolExhausted:
                return                       # stays queued; FIFO, no overtaking
            with self._lock:
                self._queue.popleft()
            req.state = "prefill"
            self._active.append(req)

    def _forward(self, reqs, input_ids, positions):
        cache = PagedPackedCache(self.pool, [r.request_id for r in reqs],
                                 [r.stored for r in reqs], k_quant=self.kq,
                                 v_quant=self.vq, head_dim=self.head_dim,
                                 decode_backend=self.cfg.decode_backend)
        out = self.model(input_ids=input_ids, position_ids=positions,
                         past_key_values=cache, use_cache=True, logits_to_keep=1)
        for r, L in zip(reqs, cache.lengths):
            r.stored = L
        logits = out.logits[:, -1].float()
        if self.on_logits is not None:
            self.on_logits(reqs, logits)
        return logits

    def _accept(self, req, token: int, now: float):
        req.generated.append(token)
        req.token_times.append(now)
        if req.first_token_at is None:
            req.first_token_at = now
        self._emit(req, kind="token", index=len(req.generated) - 1, token_id=token, t=now)
        if token in self.eos and not req.ignore_eos:
            return "eos"
        if len(req.generated) >= req.max_new_tokens:
            return "length"
        return None

    def _fence(self):
        from page_pool import default_fence
        return default_fence(self.device)

    def step(self) -> bool:
        """One scheduling iteration. Returns False when there was nothing to do."""
        with self._step_lock:
            return self._step()

    @torch.no_grad()
    def _step(self) -> bool:
        self.steps += 1
        self._reap()
        self._admit()
        did = False
        finished = []

        pre = next((r for r in self._active if r.state == "prefill"), None)
        if pre is not None:
            did = True
            lo = pre.stored
            hi = min(lo + self.cfg.prefill_chunk, len(pre.prompt_ids))
            ids = torch.tensor([pre.prompt_ids[lo:hi]], device=self.device)
            pos = torch.arange(lo, hi, device=self.device)[None]
            try:
                logits = self._forward([pre], ids, pos)
            except Exception as exc:          # noqa: BLE001 - surfaced per request
                self._active.remove(pre)
                self._finish(pre, "error", f"{type(exc).__name__}: {exc}", self._fence())
                pre = None
            if pre is not None and hi == len(pre.prompt_ids):
                pre.state = "decode"
                tok = int(logits[0].argmax())
                end = self._accept(pre, tok, self.clock())
                if end:
                    finished.append((pre, end))

        dec = [r for r in self._active if r.state == "decode"
               and not any(r is f for f, _ in finished)]
        if dec:
            did = True
            ids = torch.tensor([[r.generated[-1]] for r in dec], device=self.device)
            pos = torch.tensor([[r.stored] for r in dec], device=self.device)
            try:
                logits = self._forward(dec, ids, pos)
                toks = logits.argmax(-1).tolist()
                now = self.clock()
                for r, t in zip(dec, toks):
                    end = self._accept(r, int(t), now)
                    if end:
                        finished.append((r, end))
            except Exception as exc:          # noqa: BLE001
                fence = self._fence()
                for r in dec:
                    self._active.remove(r)
                    self._finish(r, "error", f"{type(exc).__name__}: {exc}", fence)

        if finished:
            fence = self._fence()
            for r, end in finished:
                self._active.remove(r)
                self._finish(r, end, fence=fence)
        return did

    def run_until_idle(self, max_steps: int = 100000) -> None:
        for _ in range(max_steps):
            if self.idle():
                return
            self.step()
        raise RuntimeError("engine did not drain")
