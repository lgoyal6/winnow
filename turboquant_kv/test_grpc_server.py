"""Real gRPC over a localhost port, tiny Qwen2 on CPU.

These pin the service lifecycle (isolation under concurrency, cancel by RPC and
by disconnect, deadlines, refusals, auth, shutdown, churn). Running the same
checks against the real model on a GPU is a separate gate.
"""
import threading
import time

import grpc
import pytest
import torch

import grpc_server as gs
import packed_backend as pb
import winnow_serving_pb2 as pb2
import winnow_serving_pb2_grpc as pb2_grpc
from engine import EngineConfig
from test_packed_model import model, prompts, reference  # noqa: F401 - fixture

REV = "tiny-rev"


@pytest.fixture
def served(model):
    running = []

    def _start(token=None, **cfg):
        model.config._attn_implementation = pb.ATTN_NAME
        pool = pb.make_pool(model.config, bit_width=8, num_pages=cfg.pop("pages", 64),
                            page_size=8, device="cpu")
        kq, vq = pb.quantizers_for(model.config, 8, "cpu")
        from engine import Engine
        eng = Engine(model, pool, kq, vq, pb.head_dim_of(model.config),
                     config=EngineConfig(**cfg))
        run = gs.start(eng, model_revision=REV, kv_mode="packed8", token=token)
        running.append(run)
        ch = grpc.insecure_channel(f"127.0.0.1:{run.port}")
        return eng, run, pb2_grpc.GenerationStub(ch)
    yield _start
    for r in running:
        r.stop(1)


def gen(stub, ids, n, rid, metadata=None, timeout=None, deadline_s=0.0):
    return stub.Generate(pb2.GenerateRequest(model_revision=REV, token_ids=ids, max_new_tokens=n,
                                             request_id=rid, deadline_s=deadline_s),
                         metadata=metadata, timeout=timeout)


def collect(stream):
    toks, final = [], None
    for ev in stream:
        if ev.HasField("token"):
            toks.append(ev.token.token_id)
        else:
            final = ev.final
    return toks, final


def wait_for(pred, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_concurrent_streams_match_isolated(served):
    eng, _, stub = served(max_batch=4)
    ps, news = prompts(6, seed=50), [7, 3, 12, 5, 9, 1]
    refs = [reference(stub_model := eng.model, p, n)[0] for p, n in zip(ps, news)]
    eng.model.config._attn_implementation = pb.ATTN_NAME
    out = [None] * 6

    def worker(i):
        out[i] = collect(gen(stub, ps[i], news[i], f"c{i}"))
    ths = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    [t.start() for t in ths]
    [t.join(60) for t in ths]
    for (toks, final), ref in zip(out, refs):
        assert final.status == "length" and toks == ref
        assert final.output_tokens == len(ref) and final.ttft_s > 0
    assert wait_for(lambda: eng.pool.stats()["owned"] == 0)


def test_cancel_rpc_reclaims(served):
    eng, _, stub = served()
    stream = gen(stub, prompts(1, 51)[0], 400, "victim")
    first = next(stream)
    assert first.HasField("token")
    assert stub.Cancel(pb2.CancelRequest(request_id="victim")).found
    toks, final = collect(stream)
    assert final.status == "cancelled" and len(toks) < 399
    assert wait_for(lambda: eng.pool.stats()["owned"] == 0)
    assert not stub.Cancel(pb2.CancelRequest(request_id="victim")).found


def test_client_disconnect_cancels(served):
    eng, _, stub = served()
    stream = gen(stub, prompts(1, 52)[0], 400, "gone")
    next(stream)
    stream.cancel()                                    # client walks away
    assert wait_for(lambda: eng.pool.stats()["owned"] == 0)
    assert wait_for(lambda: eng.idle())


def test_deadline_status(served):
    eng, _, stub = served()
    toks, final = collect(gen(stub, prompts(1, 53)[0], 400, "late", deadline_s=0.3))
    assert final.status == "deadline" and len(toks) < 400
    assert wait_for(lambda: eng.pool.stats()["owned"] == 0)


def test_grpc_deadline_ends_stream_and_frees(served):
    """Either the server's deadline (inside the transport one) or the transport
    deadline may end the stream first; both must free the pages."""
    eng, _, stub = served()
    try:
        toks, final = collect(gen(stub, prompts(1, 54)[0], 400, "grpc-late", timeout=0.3))
        assert final.status == "deadline" and len(toks) < 400
    except grpc.RpcError as err:
        assert err.code() == grpc.StatusCode.DEADLINE_EXCEEDED
    assert wait_for(lambda: eng.pool.stats()["owned"] == 0)


def test_refusals(served):
    eng, _, stub = served(max_queue=1, max_batch=1, max_prompt_tokens=50)
    with pytest.raises(grpc.RpcError) as e:
        collect(stub.Generate(pb2.GenerateRequest(model_revision="other", token_ids=[1], max_new_tokens=1)))
    assert e.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    with pytest.raises(grpc.RpcError) as e:
        collect(gen(stub, [1] * 51, 2, "long"))
    assert e.value.code() == grpc.StatusCode.INVALID_ARGUMENT and "prompt_too_long" in e.value.details()
    with pytest.raises(grpc.RpcError) as e:
        collect(gen(stub, [10_000], 2, "vocab"))
    assert e.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    with pytest.raises(grpc.RpcError) as e:
        collect(stub.Generate(pb2.GenerateRequest(model_revision=REV, prompt="hi", max_new_tokens=1)))
    assert e.value.code() == grpc.StatusCode.INVALID_ARGUMENT        # no tokenizer
    # Fill the single batch slot and the single queue slot, then overflow.
    a = gen(stub, [1, 2, 3], 300, "a")
    next(a)
    b = gen(stub, [1, 2, 3], 300, "b")
    assert wait_for(lambda: len(eng._queue) == 1)
    with pytest.raises(grpc.RpcError) as e:
        collect(gen(stub, [1, 2, 3], 2, "c"))
    assert e.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED and "queue_full" in e.value.details()
    a.cancel(), b.cancel()
    assert wait_for(lambda: eng.pool.stats()["owned"] == 0)


def test_auth(served):
    _, _, stub = served(token="s3cret")
    with pytest.raises(grpc.RpcError) as e:
        stub.Health(pb2.HealthRequest())
    assert e.value.code() == grpc.StatusCode.UNAUTHENTICATED
    h = stub.Health(pb2.HealthRequest(), metadata=[("authorization", "Bearer s3cret")])
    assert h.model_revision == REV and h.kv_mode == "packed8"


def test_nonloopback_bind_requires_token(model):
    with pytest.raises(SystemExit):
        gs.start(None, model_revision=REV, kv_mode="packed8", address="0.0.0.0:0")


def test_shutdown_mid_stream(served):
    eng, run, stub = served()
    stream = gen(stub, prompts(1, 55)[0], 400, "s")
    next(stream)
    run.sched.close()
    toks, final = collect(stream)
    assert final.status == "cancelled"
    assert eng.pool.stats()["owned"] == 0


def test_churn_steady_state(served):
    eng, _, stub = served(max_batch=4, pages=48)
    base = stub.Health(pb2.HealthRequest())
    for rnd in range(4):
        streams = [gen(stub, p, 6, f"r{rnd}-{i}") for i, p in enumerate(prompts(5, 60 + rnd))]
        next(streams[0])
        streams[0].cancel()
        for s in streams[1:]:
            collect(s)
        assert wait_for(lambda: eng.pool.stats()["owned"] == 0)
        h = stub.Health(pb2.HealthRequest())
        assert (h.free_pages, h.pending_pages, h.owned_pages) == (base.free_pages, 0, 0)
