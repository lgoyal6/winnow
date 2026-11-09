"""gRPC front end for the paged packed (or dense) decode engine.

    python grpc_server.py --model Qwen/Qwen2.5-7B-Instruct --kv packed8 \
        --kv-budget-gib 4 --address 127.0.0.1:50071

One scheduler thread owns the engine and calls `step()`; RPC threads only
submit, cancel and read events. A stream maps to one engine request:

* the stream's end (normal, client disconnect, or RPC deadline) cancels the
  request through `context.add_callback`, so an abandoned stream stops
  consuming decode slots and its pages are reclaimed behind a fence;
* `Cancel` reaches the same path by request id;
* admission refusals become RESOURCE_EXHAUSTED or INVALID_ARGUMENT with the
  engine's reason, before any GPU work.

Bounds: the model revision is pinned at startup and requests naming another
are refused; prompt length, max_new_tokens, batch, queue and KV pages are all
capped. Binding anything but loopback requires a bearer token
(WINNOW_SERVE_TOKEN), checked in constant time.
"""
from __future__ import annotations

import argparse
import hmac
import os
import queue
import sys
import threading
import time
import uuid
from concurrent import futures
from pathlib import Path

import grpc
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))     # model_guard
import packed_backend as pb
import winnow_serving_pb2 as pb2
import winnow_serving_pb2_grpc as pb2_grpc
from engine import Engine, EngineConfig, Refused, Request

_EXHAUSTED = {"queue_full", "shutting_down", "exceeds_request_cap"}


class SchedulerThread:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.wake = threading.Event()
        self.stop = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._loop, name="winnow-scheduler", daemon=True)

    def start(self):
        self.thread.start()

    def _loop(self):
        try:
            while not self.stop.is_set():
                if not self.engine.step():
                    self.wake.wait(0.005)
                    self.wake.clear()
        except BaseException as exc:          # noqa: BLE001 - surfaced by Health
            self.error = exc
            raise

    def close(self, timeout=10.0):
        """Stop the loop first, then reap on this thread: never two steppers."""
        self.stop.set()
        self.wake.set()
        self.thread.join(timeout)
        self.engine.shutdown()


class GenerationService(pb2_grpc.GenerationServicer):
    def __init__(self, engine: Engine, scheduler: SchedulerThread, *, model_revision: str,
                 kv_mode: str, tokenizer=None, token: str | None = None):
        self.engine, self.sched = engine, scheduler
        self.revision, self.kv_mode = model_revision, kv_mode
        self.tokenizer = tokenizer
        self.token = token

    # -- helpers ----------------------------------------------------------
    def _auth(self, context):
        if self.token is None:
            return
        md = dict(context.invocation_metadata())
        got = md.get("authorization", "")
        if not hmac.compare_digest(got.encode(), f"Bearer {self.token}".encode()):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "missing or bad bearer token")

    def _prompt_ids(self, request, context) -> list[int]:
        given = [bool(request.token_ids), bool(request.prompt), bool(request.messages)]
        if sum(given) != 1:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                          "give exactly one of token_ids, prompt, messages")
        if request.token_ids:
            vocab = self.engine.model.config.vocab_size
            ids = list(request.token_ids)
            if min(ids) < 0 or max(ids) >= vocab:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "token id outside vocabulary")
            return ids
        if self.tokenizer is None:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "server has no tokenizer; send token_ids")
        if request.prompt:
            return self.tokenizer(request.prompt, add_special_tokens=False).input_ids
        msgs = [{"role": m.role, "content": m.content} for m in request.messages]
        rendered = self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return self.tokenizer(rendered, add_special_tokens=False).input_ids

    def _text(self, tok: int) -> str:
        return self.tokenizer.decode([tok]) if self.tokenizer is not None else ""

    # -- RPCs -------------------------------------------------------------
    def Generate(self, request, context):
        self._auth(context)
        t0 = time.monotonic()
        if request.model_revision != self.revision:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                          f"server serves revision {self.revision}")
        ids = self._prompt_ids(request, context)
        # Finish 50 ms inside the transport deadline, so a client normally sees
        # a clean Final(status="deadline") rather than a bare DEADLINE_EXCEEDED.
        rpc_left = context.time_remaining()
        limits = [d for d in (request.deadline_s or None,
                              max(rpc_left - 0.05, 0.001) if rpc_left else None) if d]
        deadline = t0 + min(limits) if limits else None
        events: queue.Queue = queue.Queue()
        req = Request(request.request_id or uuid.uuid4().hex, ids, int(request.max_new_tokens),
                      deadline=deadline, on_event=lambda r, e: events.put(e),
                      ignore_eos=request.ignore_eos)
        try:
            self.engine.submit(req)
        except Refused as exc:
            code = (grpc.StatusCode.RESOURCE_EXHAUSTED if exc.reason in _EXHAUSTED
                    else grpc.StatusCode.INVALID_ARGUMENT)
            context.abort(code, str(exc))
        context.add_callback(lambda: (self.engine.cancel(req.request_id), self.sched.wake.set()))
        self.sched.wake.set()
        while True:
            ev = events.get()
            if ev["kind"] == "token":
                yield pb2.TokenEvent(token=pb2.Token(
                    sequence=ev["index"], token_id=ev["token_id"], text=self._text(ev["token_id"]),
                    server_time_s=ev["t"] - t0))
                continue
            first = req.first_token_at
            yield pb2.TokenEvent(final=pb2.Final(
                status=ev["status"], detail=ev["detail"], prompt_tokens=len(ids),
                output_tokens=len(req.generated),
                ttft_s=(first - t0) if first else 0.0,
                total_s=(req.finished_at or time.monotonic()) - t0,
                queue_s=max(0.0, req.submitted_at - t0)))
            return

    def Cancel(self, request, context):
        self._auth(context)
        found = self.engine.cancel(request.request_id)
        self.sched.wake.set()
        return pb2.CancelReply(found=found)

    def Health(self, request, context):
        self._auth(context)
        if self.sched.error is not None:
            context.abort(grpc.StatusCode.INTERNAL, f"scheduler died: {self.sched.error!r}")
        st = self.engine.pool.stats()
        with self.engine._lock:
            queued, active = len(self.engine._queue), len(self.engine._active)
        cuda = self.engine.device.type == "cuda"
        return pb2.HealthReply(
            model_revision=self.revision, kv_mode=self.kv_mode,
            decode_backend=self.engine.cfg.decode_backend,
            free_pages=st["free"], pending_pages=st["pending"], owned_pages=st["owned"],
            total_pages=st["total"], queued=queued, active=active,
            pool_bytes=self.engine.pool.bytes(),
            kv_bytes_per_token=self.engine.pool.bytes_per_token(),
            device_allocated_bytes=torch.cuda.memory_allocated() if cuda else 0,
            device_reserved_bytes=torch.cuda.memory_reserved() if cuda else 0)


class Running:
    """A started server plus everything needed to stop it cleanly."""

    def __init__(self, server, sched, port):
        self.server, self.sched, self.port = server, sched, port

    def stop(self, grace=5.0):
        self.sched.close()             # cancels outstanding work, emits finals
        self.server.stop(grace).wait()


def start(engine: Engine, *, model_revision: str, kv_mode: str, address="127.0.0.1:0",
          tokenizer=None, token: str | None = None, max_rpcs: int | None = None) -> Running:
    host = address.rsplit(":", 1)[0].strip("[]")
    if host not in ("127.0.0.1", "localhost", "::1") and not token:
        raise SystemExit("refusing a non-loopback bind without WINNOW_SERVE_TOKEN")
    sched = SchedulerThread(engine)
    limit = max_rpcs or engine.cfg.max_queue + engine.cfg.max_batch + 8
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=limit + 4),
                         maximum_concurrent_rpcs=limit)
    pb2_grpc.add_GenerationServicer_to_server(
        GenerationService(engine, sched, model_revision=model_revision, kv_mode=kv_mode,
                          tokenizer=tokenizer, token=token), server)
    port = server.add_insecure_port(address)
    sched.start()
    server.start()
    return Running(server, sched, port)


def build_engine(model, *, kv_mode: str, kv_budget_bytes: int, page_size: int,
                 decode_backend: str, cfg: EngineConfig, eos_ids=()) -> Engine:
    """Size the page pool from a KV byte budget, so arms compare at equal memory."""
    model.config._attn_implementation = pb.ATTN_NAME
    dev = next(model.parameters()).device
    if kv_mode == "dense":                       # unquantized pages in the model dtype
        probe = pb.make_pool(model.config, bit_width=None, num_pages=1, page_size=page_size,
                             device="meta", dense_dtype=model.dtype)
        kq = vq = None
    elif kv_mode.startswith("packed"):
        bw = int(kv_mode.removeprefix("packed"))
        pb.check_model(model.config, bw)
        probe = pb.make_pool(model.config, bit_width=bw, num_pages=1, page_size=page_size,
                             device="meta")
        kq, vq = pb.quantizers_for(model.config, bw, dev)
    else:
        raise ValueError(f"unknown kv mode {kv_mode!r}")
    pages = max(1, kv_budget_bytes // probe.bytes())
    pool = pb.make_pool(model.config, bit_width=None if kq is None else kq.bit_width,
                        num_pages=pages, page_size=page_size, device=dev,
                        dense_dtype=model.dtype if kq is None else None)
    return Engine(model, pool, kq, vq, pb.head_dim_of(model.config), eos_ids=eos_ids,
                  config=cfg)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--kv", default="packed8", choices=["packed8", "packed4", "dense"])
    ap.add_argument("--decode-backend", default="auto")
    ap.add_argument("--kv-budget-gib", type=float, default=4.0)
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--max-queue", type=int, default=64)
    ap.add_argument("--prefill-chunk", type=int, default=1024)
    ap.add_argument("--max-prompt-tokens", type=int, default=8192)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--address", default="127.0.0.1:50071")
    args = ap.parse_args(argv)

    from model_guard import guarded_from_pretrained, pinned_revision
    from transformers import AutoModelForCausalLM, AutoTokenizer
    pb.register()
    revision = pinned_revision(args.model)
    tok = guarded_from_pretrained(AutoTokenizer, args.model)
    model = guarded_from_pretrained(AutoModelForCausalLM, args.model, dtype=torch.bfloat16,
                                    attn_implementation="sdpa").to("cuda").eval()
    eos = {tok.eos_token_id, *(getattr(model.generation_config, "eos_token_id", None) or [])}
    eos.discard(None)
    cfg = EngineConfig(max_batch=args.max_batch, max_queue=args.max_queue,
                       prefill_chunk=args.prefill_chunk, max_prompt_tokens=args.max_prompt_tokens,
                       max_new_tokens_cap=args.max_new_tokens, decode_backend=args.decode_backend)
    engine = build_engine(model, kv_mode=args.kv, kv_budget_bytes=int(args.kv_budget_gib * 2**30),
                          page_size=args.page_size, decode_backend=args.decode_backend, cfg=cfg,
                          eos_ids=eos)
    run = start(engine, model_revision=revision, kv_mode=args.kv, address=args.address,
                tokenizer=tok, token=os.environ.get("WINNOW_SERVE_TOKEN"))
    print(f"serving {args.model}@{revision} kv={args.kv} pages={engine.pool.num_pages} "
          f"on port {run.port}", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        run.stop()


if __name__ == "__main__":
    main()
