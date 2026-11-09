"""Bounded real-model gRPC isolation, cancellation, and reclamation check.

This runner is deliberately separate from the preregistered model contract:
the logit comparison is diagnostic, while token identity and resource lifecycle
checks are strict. It never launches or provisions a remote machine.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

import grpc
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent))

import grpc_server as gs
import packed_backend as pb
import winnow_serving_pb2 as pb2
import winnow_serving_pb2_grpc as pb2_grpc
from engine import EngineConfig
from model_contract import PROMPTS, chat_ids
from model_guard import guarded_from_pretrained, pinned_revision


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_hashes() -> dict[str, str]:
    paths = [HERE / n for n in ("gpu_serving_isolation.py", "grpc_server.py", "engine.py",
                                 "page_pool.py", "packed_backend.py", "model_contract.py",
                                 "proto/winnow_serving.proto", "winnow_serving_pb2.py",
                                 "winnow_serving_pb2_grpc.py")]
    return {p.name: sha(p.read_bytes()) for p in paths if p.is_file()}


def health(stub) -> dict:
    h = stub.Health(pb2.HealthRequest(), timeout=5)
    return {k: int(getattr(h, k)) if k not in ("model_revision", "kv_mode", "decode_backend")
            else getattr(h, k) for k in ("model_revision", "kv_mode", "decode_backend",
                "free_pages", "pending_pages", "owned_pages", "total_pages", "queued",
                "active", "pool_bytes", "kv_bytes_per_token", "device_allocated_bytes",
                "device_reserved_bytes")}


def wait_clean(stub, timeout: float) -> tuple[bool, dict]:
    end = time.monotonic() + timeout
    last = health(stub)
    while time.monotonic() < end:
        if (last["owned_pages"] == 0 and last["pending_pages"] == 0 and last["active"] == 0
                and last["queued"] == 0 and last["free_pages"] == last["total_pages"]):
            return True, last
        time.sleep(0.05)
        last = health(stub)
    return False, last


def call(stub, revision, ids, count, rid, *, timeout=None, deadline_s=0.0):
    req = pb2.GenerateRequest(model_revision=revision, token_ids=ids,
        max_new_tokens=count, request_id=rid, deadline_s=deadline_s, ignore_eos=True)
    toks, final = [], None
    for ev in stub.Generate(req, timeout=timeout):
        if ev.HasField("token"):
            toks.append(int(ev.token.token_id))
        else:
            final = {"status": ev.final.status, "detail": ev.final.detail,
                     "output_tokens": int(ev.final.output_tokens), "total_s": ev.final.total_s}
    return toks, final


def logit_capture(engine):
    captured, lock = {}, threading.Lock()
    def on_logits(reqs, logits):
        with lock:
            for i, req in enumerate(reqs):
                captured.setdefault(req.request_id, []).append(logits[i].detach().float().cpu().clone())
    engine.on_logits = on_logits
    return captured, lock


def compare_logits(ref_tokens, got_tokens, ref_logits, got_logits):
    n = min(len(ref_tokens), len(got_tokens))
    div = next((i for i in range(n) if ref_tokens[i] != got_tokens[i]), None)
    upto = div if div is not None else n
    ln = min(len(ref_logits), len(got_logits))
    diffs = [(a - b).abs().max().item() for a, b in zip(ref_logits[:ln], got_logits[:ln])]
    return {"first_token_mismatch": div, "token_mismatches": sum(a != b for a, b in zip(ref_tokens, got_tokens)),
            "length_mismatch": len(ref_tokens) != len(got_tokens), "logit_steps": ln,
            "logit_trace_length_mismatch": len(ref_logits) != len(got_logits),
            "logit_steps_not_bitwise_identical": sum(not torch.equal(a, b) for a, b in zip(ref_logits[:ln], got_logits[:ln])),
            "max_logit_abs_diff_before_token_mismatch": max(diffs[:upto], default=0.0),
            "logit_mismatches_over_1e-3": sum(x > 1e-3 for x in diffs),
            "max_logit_abs_diff_all_common_steps": max(diffs, default=0.0)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--kv-budget-gib", type=float, default=4)
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--deadline-seconds", type=float, default=0.15)
    ap.add_argument("--stage-timeout", type=float, default=300)
    ap.add_argument("--churn-rounds", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=3)
    args = ap.parse_args(argv)
    if args.stage_timeout <= 0 or args.stage_timeout > 1800:
        ap.error("--stage-timeout must be in (0, 1800]")
    if args.churn_rounds not in range(1, 21) or args.concurrency not in range(2, 9):
        ap.error("churn rounds must be 1..20 and concurrency 2..8")
    if args.concurrency > len(PROMPTS) or not 1 <= args.max_new_tokens <= 64:
        ap.error("concurrency must fit available prompts and token count must be 1..64")
    if args.output.exists():
        ap.error("output already exists; preserve prior evidence and choose a new path")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; no work was performed")

    revision = pinned_revision(args.model)
    pb.register()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = guarded_from_pretrained(AutoTokenizer, args.model)
    model = guarded_from_pretrained(AutoModelForCausalLM, args.model,
        dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    eos = {tok.eos_token_id}
    extra = getattr(model.generation_config, "eos_token_id", None)
    eos.update(extra if isinstance(extra, (list, tuple)) else ([extra] if extra is not None else []))
    eos.discard(None)
    cfg = EngineConfig(max_batch=args.concurrency, max_queue=args.concurrency * 3,
        prefill_chunk=512, max_prompt_tokens=8192, max_new_tokens_cap=max(64, args.max_new_tokens),
        decode_backend="triton_packed")
    engine = gs.build_engine(model, kv_mode="packed8", kv_budget_bytes=int(args.kv_budget_gib * 2**30),
        page_size=args.page_size, decode_backend=cfg.decode_backend, cfg=cfg, eos_ids=eos)
    logits, logits_lock = logit_capture(engine)
    run = gs.start(engine, model_revision=revision, kv_mode="packed8", tokenizer=tok,
                   max_rpcs=args.concurrency + 8)
    ch = grpc.insecure_channel(f"127.0.0.1:{run.port}")
    grpc.channel_ready_future(ch).result(timeout=10)
    stub = pb2_grpc.GenerationStub(ch)
    records = {"schema": "winnow.gpu-serving-isolation.v1", "started_unix": time.time(),
        "model": args.model, "model_revision": revision, "dtype": str(model.dtype),
        "device": torch.cuda.get_device_name(), "kv_mode": "packed8", "decode_backend": cfg.decode_backend,
        "source_sha256": source_hashes(), "limits": vars(args) | {"output": str(args.output)},
        "prompt_sha256": [sha(json.dumps(chat_ids(tok, p), separators=(",", ":")).encode()) for p in PROMPTS[:args.concurrency]],
        "stages": {}, "acceptance": "UNDECIDED"}
    failed = []
    try:
        ids = [chat_ids(tok, p) for p in PROMPTS[:args.concurrency]]
        counts = [max(1, args.max_new_tokens - i * 3) for i in range(len(ids))]
        # Isolated baselines and then identical prompts through real concurrent RPCs.
        solo = []
        for i, (p, n) in enumerate(zip(ids, counts)):
            rid = f"solo-{i}-{uuid.uuid4().hex}"
            toks, fin = call(stub, revision, p, n, rid, timeout=args.stage_timeout)
            solo.append((toks, fin, rid))
        before = health(stub)
        if before["kv_mode"] != "packed8" or before["decode_backend"] != "triton_packed":
            failed.append("configured-backend-identity-mismatch")
        jobs = [lambda i=i: call(stub, revision, ids[i], counts[i], f"parallel-{i}-{uuid.uuid4().hex}", timeout=args.stage_timeout)
                for i in range(len(ids))]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as ex:
            futures = [ex.submit(f) for f in jobs]
            parallel = [f.result(timeout=args.stage_timeout) for f in futures]
        conc = []
        for i, ((rt, rf, rid), (gt, gf)) in enumerate(zip(solo, parallel)):
            with logits_lock:
                lr, lg = logits.get(rid, []), []
                # concurrent IDs are encoded in each request trace; map by insertion order below
                prefix = f"parallel-{i}-"
                candidates = [(k, v) for k, v in logits.items() if k.startswith(prefix)]
                lg = candidates[-1][1] if candidates else []
            n = min(len(lr), len(lg)); diffs = [(a-b).abs().max().item() for a,b in zip(lr[:n], lg[:n])]
            cmp = compare_logits(rt, gt, lr, lg)
            cmp.update({"logit_trace_steps_isolated": len(lr), "logit_trace_steps_concurrent": len(lg),
                        "max_trace_logit_abs_diff": max(diffs, default=None),
                        "prompt_sha256": records["prompt_sha256"][i],
                        "isolated_tokens_sha256": sha(json.dumps(rt).encode()),
                        "concurrent_tokens_sha256": sha(json.dumps(gt).encode()),
                        "isolated_final": rf, "concurrent_final": gf})
            conc.append(cmp)
            if rt != gt or rf is None or gf is None or rf["status"] != gf["status"]:
                failed.append(f"request-{i}-token-or-final-mismatch")
            if len(lr) != len(rt) or len(lg) != len(gt):
                failed.append(f"request-{i}-incomplete-logit-trace")
        clean, after = wait_clean(stub, args.stage_timeout)
        records["stages"]["isolation"] = {"health_before": before, "health_after": after,
            "pool_health_fingerprint_before": sha(json.dumps(before, sort_keys=True).encode()),
            "pool_health_fingerprint_after": sha(json.dumps(after, sort_keys=True).encode()),
            "requests": conc, "pool_reclaimed": clean}
        if not clean: failed.append("isolation-pool-not-reclaimed")

        # Cancellation by request id.
        rid = "cancel-" + uuid.uuid4().hex
        stream = stub.Generate(pb2.GenerateRequest(model_revision=revision, token_ids=ids[0],
            max_new_tokens=cfg.max_new_tokens_cap, request_id=rid, ignore_eos=True), timeout=args.stage_timeout)
        next(stream)
        cancel_found = stub.Cancel(pb2.CancelRequest(request_id=rid), timeout=5).found
        cancel_tokens, cancel_final = [], None
        for ev in stream:
            if ev.HasField("token"): cancel_tokens.append(ev.token.token_id)
            else: cancel_final = ev.final.status
        cancel_clean, cancel_health = wait_clean(stub, args.stage_timeout)
        records["stages"]["cancel_rpc"] = {"cancel_found": cancel_found, "final": cancel_final,
            "tokens_received": len(cancel_tokens), "health_after": cancel_health, "pool_reclaimed": cancel_clean}
        if not cancel_found or cancel_final != "cancelled" or not cancel_clean: failed.append("cancel-rpc-lifecycle")

        # Client disconnect cancellation.
        rid = "disconnect-" + uuid.uuid4().hex
        stream = stub.Generate(pb2.GenerateRequest(model_revision=revision, token_ids=ids[0],
            max_new_tokens=cfg.max_new_tokens_cap, request_id=rid, ignore_eos=True), timeout=args.stage_timeout)
        next(stream); stream.cancel()
        disc_clean, disc_health = wait_clean(stub, args.stage_timeout)
        records["stages"]["disconnect"] = {"health_after": disc_health, "pool_reclaimed": disc_clean}
        if not disc_clean: failed.append("disconnect-pool-not-reclaimed")

        # Engine deadline and transport deadline are distinct cancellation paths.
        rid = "engine-deadline-" + uuid.uuid4().hex
        try:
            _, final = call(stub, revision, ids[0], cfg.max_new_tokens_cap, rid,
                            timeout=args.stage_timeout, deadline_s=args.deadline_seconds)
            status = final["status"] if final else "missing-final"
        except grpc.RpcError as exc: status = exc.code().name
        dclean, dh = wait_clean(stub, args.stage_timeout)
        records["stages"]["engine_deadline"] = {"status": status, "health_after": dh, "pool_reclaimed": dclean}
        if status != "deadline" or not dclean: failed.append("engine-deadline-lifecycle")
        rid = "transport-deadline-" + uuid.uuid4().hex
        try:
            _, final = call(stub, revision, ids[0], cfg.max_new_tokens_cap, rid,
                            timeout=args.deadline_seconds)
            status = final["status"] if final else "missing-final"
        except grpc.RpcError as exc: status = exc.code().name
        tclean, th = wait_clean(stub, args.stage_timeout)
        records["stages"]["transport_deadline"] = {"status": status, "health_after": th, "pool_reclaimed": tclean}
        if status not in ("deadline", "DEADLINE_EXCEEDED") or not tclean: failed.append("transport-deadline-lifecycle")

        # Empty prompt is refused before GPU admission.
        try:
            list(stub.Generate(pb2.GenerateRequest(model_revision=revision, max_new_tokens=1), timeout=5))
            refusal = "accepted"
        except grpc.RpcError as exc: refusal = exc.code().name
        records["stages"]["refusal"] = {"empty_prompt_status": refusal, "health": health(stub)}
        if refusal != "INVALID_ARGUMENT": failed.append("empty-prompt-not-refused")

        # Repeated mixed workloads then check the fixed pool returns to its baseline.
        base = health(stub); rounds = []
        for r in range(args.churn_rounds):
            jobs = [lambda i=i, r=r: call(stub, revision, ids[i % len(ids)], min(4, args.max_new_tokens),
                f"churn-{r}-{i}-{uuid.uuid4().hex}", timeout=args.stage_timeout) for i in range(args.concurrency)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as ex:
                results = [f.result(timeout=args.stage_timeout) for f in [ex.submit(j) for j in jobs]]
            clean, h = wait_clean(stub, args.stage_timeout)
            rounds.append({"round": r, "statuses": [x[1]["status"] if x[1] else "missing" for x in results],
                           "health": h, "pool_reclaimed": clean})
            if not clean or any(x[1] is None for x in results): failed.append(f"churn-round-{r}")
        records["stages"]["churn"] = {"baseline": base, "rounds": rounds, "final": health(stub)}
        records["acceptance"] = "FAILED_ACCEPTANCE" if failed else "PASSED_LIFECYCLE_WITH_NUMERICAL_DIAGNOSTICS"
        records["failures"] = failed
    except BaseException as exc:
        records["acceptance"] = "FAILED_ACCEPTANCE"
        records["failures"] = failed + [f"runner-exception:{type(exc).__name__}:{exc}"]
        raise
    finally:
        records["finished_unix"] = time.time()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        tmp.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, args.output)
        ch.close()
        run.stop()
    print(json.dumps({"acceptance": records["acceptance"], "failures": records.get("failures", []),
                      "output": str(args.output)}, indent=2))
    return 1 if records["acceptance"] == "FAILED_ACCEPTANCE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
