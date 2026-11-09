"""Replay a frozen workload against a serving endpoint and keep every row.

    python loadgen.py --workload W.json --target grpc://127.0.0.1:50071 \
        --revision REV --arm packed8-direct --rate 4 --out RUN_DIR
    python loadgen.py --workload W.json --target openai://127.0.0.1:8000 \
        --model Qwen/Qwen2.5-7B-Instruct --arm vllm-bf16 --concurrency 16 --out RUN_DIR

Workload: {"requests": [{"id", "token_ids", "max_new_tokens", ...}, ...]}.
Arrivals are open-loop Poisson at `--rate` (seeded, so every arm sees the same
offered schedule) or closed-loop at `--concurrency`. A refusal or error is a
row with its status, never a dropped request: SLO denominators count offered
requests. Token timestamps are taken on arrival at this client, so they
include network and serialization time for both target kinds.

gRPC targets are also polled for Health every 0.5 s (pool pages, device
allocated/reserved bytes) into health.jsonl. For OpenAI-compatible targets
(vLLM) device memory must be sampled out of band (nvidia-smi), recorded by the
run script.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _grpc_call(stub, pb2, req, revision, metadata, timeout):
    t0 = time.monotonic()
    row = {"t_submit": t0, "token_times": [], "token_ids": []}
    try:
        stream = stub.Generate(pb2.GenerateRequest(
            model_revision=revision, token_ids=req["token_ids"],
            max_new_tokens=req["max_new_tokens"], request_id=req["uid"],
            ignore_eos=req.get("ignore_eos", False)), metadata=metadata, timeout=timeout)
        for ev in stream:
            now = time.monotonic()
            if ev.HasField("token"):
                row["token_times"].append(now - t0)
                row["token_ids"].append(ev.token.token_id)
            else:
                f = ev.final
                row.update(status=f.status, detail=f.detail, server_ttft_s=f.ttft_s,
                           server_total_s=f.total_s)
    except Exception as exc:                                    # noqa: BLE001
        code = getattr(exc, "code", lambda: None)()
        row.update(status="refused" if code is not None and code.name in
                   ("RESOURCE_EXHAUSTED",) else "error",
                   detail=f"{getattr(code, 'name', type(exc).__name__)}: "
                          f"{getattr(exc, 'details', lambda: str(exc))()}")
    row["t_end"] = time.monotonic() - t0
    return row


def _openai_call(client, base, model, req, timeout):
    t0 = time.monotonic()
    row = {"t_submit": t0, "token_times": [], "token_ids": []}
    body = {"model": model, "prompt": req["token_ids"], "max_tokens": req["max_new_tokens"],
            "temperature": 0, "stream": True, "ignore_eos": req.get("ignore_eos", False),
            "stream_options": {"include_usage": True}, "return_tokens_as_token_ids": True}
    try:
        with client.stream("POST", f"{base}/v1/completions", json=body, timeout=timeout) as r:
            if r.status_code != 200:
                row.update(status="refused" if r.status_code in (429, 503) else "error",
                           detail=f"HTTP {r.status_code}: {r.read()[:200]!r}")
                row["t_end"] = time.monotonic() - t0
                return row
            for line in r.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                msg = json.loads(line[6:])
                for ch in msg.get("choices", []):
                    if ch.get("text"):
                        row["token_times"].append(time.monotonic() - t0)
                        row["token_ids"].append(ch.get("text"))
                    if ch.get("finish_reason"):
                        row["status"] = {"stop": "eos", "length": "length"}.get(
                            ch["finish_reason"], ch["finish_reason"])
                if msg.get("usage"):
                    row["usage"] = msg["usage"]
    except Exception as exc:                                    # noqa: BLE001
        row.update(status="error", detail=f"{type(exc).__name__}: {exc}")
    row["t_end"] = time.monotonic() - t0
    return row


def schedule(n, rate, seed):
    rng = random.Random(seed)
    t, out = 0.0, []
    for _ in range(n):
        out.append(t)
        t += rng.expovariate(rate)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Replay a frozen workload; keep every row.")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--target", required=True, help="grpc://host:port or openai://host:port")
    ap.add_argument("--arm", required=True, help="label recorded on every row")
    ap.add_argument("--out", required=True, help="new directory")
    ap.add_argument("--revision", default="", help="model revision (gRPC)")
    ap.add_argument("--model", default="", help="model name (OpenAI-compatible)")
    ap.add_argument("--rate", type=float, help="open-loop Poisson arrivals per second")
    ap.add_argument("--concurrency", type=int, help="closed-loop concurrent requests")
    ap.add_argument("--limit", type=int, help="first N workload requests only")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ignore-eos", action="store_true", help="fixed-length outputs (load studies)")
    args = ap.parse_args(argv)
    if (args.rate is None) == (args.concurrency is None):
        ap.error("give exactly one of --rate or --concurrency")

    reqs = json.loads(Path(args.workload).read_text())["requests"][:args.limit]
    for i, r in enumerate(reqs):
        r["uid"] = f"{args.arm}-{i}-{r['id']}"
        r["ignore_eos"] = args.ignore_eos
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    kind, addr = args.target.split("://", 1)
    stop = threading.Event()

    if kind == "grpc":
        import os

        import grpc
        import winnow_serving_pb2 as pb2
        import winnow_serving_pb2_grpc as pb2_grpc
        ch = grpc.insecure_channel(addr, options=[("grpc.max_receive_message_length", 1 << 24)])
        stub = pb2_grpc.GenerationStub(ch)
        tok = os.environ.get("WINNOW_SERVE_TOKEN")
        md = [("authorization", f"Bearer {tok}")] if tok else None
        call = lambda r: _grpc_call(stub, pb2, r, args.revision, md, args.timeout)

        def poll():
            with (out / "health.jsonl").open("w") as fh:
                while not stop.is_set():
                    try:
                        h = stub.Health(pb2.HealthRequest(), metadata=md, timeout=2)
                        fh.write(json.dumps({"t": time.monotonic(), "free": h.free_pages,
                                             "pending": h.pending_pages, "owned": h.owned_pages,
                                             "queued": h.queued, "active": h.active,
                                             "allocated": h.device_allocated_bytes,
                                             "reserved": h.device_reserved_bytes,
                                             "pool_bytes": h.pool_bytes}) + "\n")
                    except Exception as exc:                    # noqa: BLE001
                        fh.write(json.dumps({"t": time.monotonic(), "error": str(exc)}) + "\n")
                    stop.wait(0.5)
        poller = threading.Thread(target=poll, daemon=True)
        poller.start()
    elif kind == "openai":
        import httpx
        client = httpx.Client(timeout=args.timeout, limits=httpx.Limits(max_connections=512))
        call = lambda r: _openai_call(client, f"http://{addr}", args.model, r, args.timeout)
        poller = None
    else:
        ap.error(f"unknown target kind {kind!r}")

    meta = {"argv": sys.argv, "n": len(reqs), "start_wall": time.time()}
    rows = [None] * len(reqs)
    t_start = time.monotonic()
    if args.rate is not None:
        offsets = schedule(len(reqs), args.rate, args.seed)
        with ThreadPoolExecutor(max_workers=min(len(reqs), 1024)) as ex:
            futs = []
            for i, (r, off) in enumerate(zip(reqs, offsets)):
                delay = t_start + off - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                futs.append((i, off, ex.submit(call, r)))
            for i, off, f in futs:
                rows[i] = {**f.result(), "offered_at": off}
    else:
        nxt = iter(range(len(reqs)))
        lock = threading.Lock()

        def worker():
            while True:
                with lock:
                    i = next(nxt, None)
                if i is None:
                    return
                off = time.monotonic() - t_start
                rows[i] = {**call(reqs[i]), "offered_at": off}
        ths = [threading.Thread(target=worker) for _ in range(args.concurrency)]
        [t.start() for t in ths]
        [t.join() for t in ths]
    meta["duration_s"] = time.monotonic() - t_start
    stop.set()
    if poller:
        poller.join(3)
    with (out / "requests.jsonl").open("w") as fh:
        for r, row in zip(reqs, rows):
            row["t_submit"] -= t_start
            fh.write(json.dumps({"arm": args.arm, "id": r["id"], "uid": r["uid"],
                                 "prompt_tokens": len(r["token_ids"]),
                                 "max_new_tokens": r["max_new_tokens"],
                                 **{k: r[k] for k in ("cohort", "context_tokens") if k in r},
                                 **row}) + "\n")
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    ok = sum(1 for r in rows if r.get("status") in ("eos", "length"))
    print(f"{args.arm}: {ok}/{len(rows)} completed in {meta['duration_s']:.1f}s -> {out}")


if __name__ == "__main__":
    main()
