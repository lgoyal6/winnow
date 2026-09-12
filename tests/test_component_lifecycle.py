"""What happens to this service when one of its four workers does not come up.

Every worker here is a fake: three callables and a name, built in this file.
Nothing imports `modal`, nothing opens a socket, nothing reads a credential and
nothing downloads a model. That is the point of the lifecycle taking callables
rather than Modal handles -- the interesting paths are the failure ones, and
none of them can be exercised against real GPU infrastructure without paying for
an outage to look at.

The split that most of this file is about: LLMLingua-2 and the default
TurboQuant route are what the service *is*, so losing either is not a degraded
service, it is no service. AttentionRAG and LCLM are each one feature, and
taking the process down over one feature costs all of them.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402
from lifecycle import (  # noqa: E402
    ComponentSpec,
    ComponentState,
    ComponentUnavailable,
    Lifecycle,
    StartupAborted,
)


class FakeWorker:
    """A worker that records what was done to it, and can be told to fail.

    Deliberately has no `remote` attribute unless a test gives it one. A fake
    that answered `.remote.aio(...)` for any method name would let a missing
    call site pass silently, and a missing call site is exactly the bug where an
    endpoint reaches a worker nobody started.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.started = 0
        self.warmed = 0
        self.cleaned = 0
        self.fail_start = False
        self.fail_warmup = False
        self.gate: asyncio.Event | None = None
        self.on_warm = None
        self.events: list[str] = []


class Recorder:
    """The order things happened in, across every component."""

    def __init__(self) -> None:
        self.log: list[str] = []

    def note(self, what: str) -> None:
        self.log.append(what)

    def only(self, suffix: str) -> list[str]:
        return [entry for entry in self.log if entry.endswith(suffix)]


def spec_for(worker: FakeWorker, recorder: Recorder, *, critical: bool) -> ComponentSpec:
    async def start():
        if worker.fail_start:
            raise RuntimeError(f"{worker.name} could not reach its runtime")
        worker.started += 1
        recorder.note(f"{worker.name}:start")
        return worker

    async def warmup(handle):
        if worker.on_warm is not None:
            worker.on_warm()
        if worker.gate is not None:
            await worker.gate.wait()
        if worker.fail_warmup:
            raise RuntimeError(f"{worker.name} failed to load its model")
        handle.warmed += 1
        recorder.note(f"{worker.name}:warm")

    async def cleanup(handle):
        if handle is not None:
            handle.cleaned += 1
        recorder.note(f"{worker.name}:clean")

    return ComponentSpec(
        name=worker.name,
        capability=f"{worker.name} capability",
        critical=critical,
        start=start,
        warmup=warmup,
        cleanup=cleanup,
    )


@pytest.fixture
def pool():
    """The four workers, classified exactly as the service classifies them."""
    recorder = Recorder()
    workers = {
        server.COMPRESSOR: FakeWorker(server.COMPRESSOR),
        server.ATTENTIONRAG: FakeWorker(server.ATTENTIONRAG),
        server.TURBOQUANT: FakeWorker(server.TURBOQUANT),
        server.LCLM: FakeWorker(server.LCLM),
    }
    critical = {server.COMPRESSOR: True, server.ATTENTIONRAG: False,
                server.TURBOQUANT: True, server.LCLM: False}

    def build():
        return [spec_for(workers[name], recorder, critical=critical[name])
                for name in (server.COMPRESSOR, server.ATTENTIONRAG,
                             server.TURBOQUANT, server.LCLM)]

    return workers, recorder, build


# ------------------------------------------------------------- the happy path


def test_every_component_starts_warms_and_reports_ready(pool):
    workers, _, build = pool
    life = Lifecycle(build())
    asyncio.run(life.start())

    assert life.ready
    assert not life.degraded
    for worker in workers.values():
        assert worker.started == 1
        assert worker.warmed == 1
    assert all(life.status(name).state is ComponentState.READY for name in workers)


def test_warmups_run_concurrently_rather_than_one_after_another(pool):
    """Four models loading in series is four cold starts end to end. The old
    lifespan gathered them, and that is worth keeping.

    Every warmup blocks on a gate that is only opened once all four have
    entered. Serial warmup cannot reach that point -- the first one would wait
    on a gate nothing will ever open -- so this fails by timing out rather than
    by passing vacuously.
    """
    workers, _, build = pool
    for worker in workers.values():
        worker.gate = asyncio.Event()

    async def scenario():
        inflight = 0
        all_entered = asyncio.Event()

        def entered():
            nonlocal inflight
            inflight += 1
            if inflight == len(workers):
                all_entered.set()

        for worker in workers.values():
            worker.on_warm = entered

        life = Lifecycle(build())
        task = asyncio.create_task(life.start())
        await asyncio.wait_for(all_entered.wait(), timeout=2)
        for worker in workers.values():
            worker.gate.set()
        await asyncio.wait_for(task, timeout=2)
        return life

    life = asyncio.run(scenario())
    assert life.ready


# ------------------------------------------ an optional worker that does not


def test_an_attentionrag_warmup_failure_leaves_both_critical_workers_ready(pool):
    """NEGATIVE CONTROL 1. One feature is gone; the service is not."""
    workers, _, build = pool
    workers[server.ATTENTIONRAG].fail_warmup = True

    life = Lifecycle(build())
    asyncio.run(life.start())

    assert life.status(server.COMPRESSOR).state is ComponentState.READY
    assert life.status(server.TURBOQUANT).state is ComponentState.READY
    assert life.status(server.ATTENTIONRAG).state is ComponentState.FAILED
    assert life.ready, "a failed optional worker took the service down"
    assert life.degraded, "a failed optional worker was not reported"


def test_an_lclm_start_failure_leaves_the_default_generation_route_ready(pool):
    workers, _, build = pool
    workers[server.LCLM].fail_start = True

    life = Lifecycle(build())
    asyncio.run(life.start())

    assert life.status(server.TURBOQUANT).state is ComponentState.READY
    assert life.status(server.LCLM).state is ComponentState.FAILED
    assert life.ready
    assert life.degraded


def test_an_optional_failure_is_cleaned_up_rather_than_left_holding_its_context(pool):
    workers, recorder, build = pool
    workers[server.ATTENTIONRAG].fail_warmup = True

    life = Lifecycle(build())
    asyncio.run(life.start())

    assert workers[server.ATTENTIONRAG].cleaned == 1
    assert workers[server.COMPRESSOR].cleaned == 0, "a working worker was closed"


def test_an_optional_failure_says_which_capability_and_why(pool):
    workers, _, build = pool
    workers[server.LCLM].fail_warmup = True

    life = Lifecycle(build())
    asyncio.run(life.start())

    status = life.status(server.LCLM)
    assert status.state is ComponentState.FAILED
    assert "could not warm up" in status.detail
    assert "RuntimeError" in status.detail
    assert "failed to load its model" in status.detail


# ------------------------------------------- a critical worker that does not


def test_a_compressor_warmup_failure_aborts_startup(pool):
    workers, _, build = pool
    workers[server.COMPRESSOR].fail_warmup = True

    life = Lifecycle(build())
    with pytest.raises(StartupAborted):
        asyncio.run(life.start())

    assert not life.ready


def test_a_turboquant_start_failure_aborts_startup(pool):
    workers, _, build = pool
    workers[server.TURBOQUANT].fail_start = True

    life = Lifecycle(build())
    with pytest.raises(StartupAborted):
        asyncio.run(life.start())

    assert not life.ready


def test_a_critical_failure_closes_started_components_in_reverse_order_once(pool):
    """NEGATIVE CONTROL 2. A half-started process that answers requests is
    worse than one that never came up, and a container nobody closed outlives
    the process that leaked it."""
    workers, recorder, build = pool
    workers[server.LCLM].fail_warmup = False
    workers[server.COMPRESSOR].fail_warmup = True

    life = Lifecycle(build())
    with pytest.raises(StartupAborted):
        asyncio.run(life.start())

    closed = [entry.split(":")[0] for entry in recorder.only(":clean")]
    assert closed == [server.LCLM, server.TURBOQUANT, server.ATTENTIONRAG, server.COMPRESSOR], \
        f"closed in {closed}, want the reverse of the order they started in"
    for worker in workers.values():
        assert worker.cleaned == 1, f"{worker.name} was closed {worker.cleaned} times"


def test_a_start_failure_does_not_start_anything_after_it(pool):
    """There is nothing to be gained from bringing up three more workers for a
    service that is already not going to serve."""
    workers, _, build = pool
    workers[server.COMPRESSOR].fail_start = True

    life = Lifecycle(build())
    with pytest.raises(StartupAborted):
        asyncio.run(life.start())

    assert workers[server.ATTENTIONRAG].started == 0
    assert workers[server.TURBOQUANT].started == 0


def test_a_critical_failure_waits_for_the_other_warmups_first(pool):
    """Raising the moment one warmup fails would leave the rest loading models
    into a process that is about to exit."""
    workers, _, build = pool
    workers[server.COMPRESSOR].fail_warmup = True

    life = Lifecycle(build())
    with pytest.raises(StartupAborted):
        asyncio.run(life.start())

    assert workers[server.TURBOQUANT].warmed == 1
    assert workers[server.LCLM].warmed == 1


# --------------------------------------------------------------- shutdown


def test_a_normal_shutdown_closes_every_started_component_exactly_once(pool):
    workers, recorder, build = pool

    async def scenario():
        life = Lifecycle(build())
        await life.start()
        await life.stop()
        return life

    life = asyncio.run(scenario())

    for worker in workers.values():
        assert worker.cleaned == 1, f"{worker.name} closed {worker.cleaned} times"
    closed = [entry.split(":")[0] for entry in recorder.only(":clean")]
    assert closed == [server.LCLM, server.TURBOQUANT, server.ATTENTIONRAG, server.COMPRESSOR]
    assert all(life.status(name).state is ComponentState.STOPPED for name in workers)


def test_stopping_twice_does_not_close_anything_twice(pool):
    workers, _, build = pool

    async def scenario():
        life = Lifecycle(build())
        await life.start()
        await life.stop()
        await life.stop()

    asyncio.run(scenario())
    for worker in workers.values():
        assert worker.cleaned == 1


def test_a_cleanup_that_raises_does_not_strand_the_components_below_it(pool):
    workers, recorder, build = pool

    def build_with_bad_cleanup():
        specs = build()
        out = []
        for spec in specs:
            if spec.name != server.TURBOQUANT:
                out.append(spec)
                continue

            async def cleanup(_handle, _spec=spec):
                recorder.note(f"{_spec.name}:clean")
                raise RuntimeError("the runtime did not answer on the way out")

            out.append(ComponentSpec(
                name=spec.name, capability=spec.capability, critical=spec.critical,
                start=spec.start, warmup=spec.warmup, cleanup=cleanup,
            ))
        return out

    async def scenario():
        life = Lifecycle(build_with_bad_cleanup())
        await life.start()
        await life.stop()
        return life

    life = asyncio.run(scenario())
    assert workers[server.COMPRESSOR].cleaned == 1, "a failed cleanup stranded the one below it"
    assert "cleanup failed" in life.status(server.TURBOQUANT).detail


# --------------------------------------------------------------- readiness


def test_readiness_cannot_report_ready_before_warmup_completes(pool):
    """NEGATIVE CONTROL 3. The old health check reported ready the instant a
    handle object existed, which is true long before a model is loaded -- so the
    one question it was asked was the one it could not answer."""
    workers, _, build = pool
    gate = asyncio.Event()
    workers[server.COMPRESSOR].gate = gate

    async def scenario():
        life = Lifecycle(build())
        task = asyncio.create_task(life.start())
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        blocked = life.readiness()
        assert not blocked["ready"], "ready while a critical warmup was still blocked"
        assert life.status(server.COMPRESSOR).state is ComponentState.STARTING

        gate.set()
        await task
        return life.readiness()

    released = asyncio.run(scenario())
    assert released["ready"]


def test_readiness_lists_every_component_and_the_top_level_state(pool):
    workers, _, build = pool
    workers[server.LCLM].fail_warmup = True

    life = Lifecycle(build())
    asyncio.run(life.start())
    states = life.readiness()

    assert [c["name"] for c in states["components"]] == [
        server.COMPRESSOR, server.ATTENTIONRAG, server.TURBOQUANT, server.LCLM
    ]
    assert states["ready"] is True
    assert states["degraded"] is True
    lclm = next(c for c in states["components"] if c["name"] == server.LCLM)
    assert lclm["ready"] is False
    assert lclm["state"] == "failed"
    assert lclm["critical"] is False


def test_failure_detail_does_not_carry_anything_that_looks_like_a_credential(pool):
    """Readiness is served over HTTP, which makes it the one place a token
    leaves without anybody thinking of it as an export."""
    workers, recorder, build = pool
    # Invented, and deliberately assembled from two pieces so that no
    # credential-shaped literal exists in the source for a secret scanner to
    # find or a reader to worry about.
    token = "sk-ant-api03-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6"

    def build_with_leaky_error():
        specs = build()
        out = []
        for spec in specs:
            if spec.name != server.LCLM:
                out.append(spec)
                continue

            async def warmup(_handle):
                raise RuntimeError(f"authentication rejected for token {token}")

            out.append(ComponentSpec(
                name=spec.name, capability=spec.capability, critical=spec.critical,
                start=spec.start, warmup=warmup, cleanup=spec.cleanup,
            ))
        return out

    life = Lifecycle(build_with_leaky_error())
    asyncio.run(life.start())

    detail = life.status(server.LCLM).detail
    assert token not in detail
    assert "[redacted]" in detail


def test_asking_for_a_component_that_is_not_ready_names_the_capability(pool):
    workers, _, build = pool
    workers[server.LCLM].fail_warmup = True

    life = Lifecycle(build())
    asyncio.run(life.start())

    with pytest.raises(ComponentUnavailable) as raised:
        life.handle(server.LCLM)
    assert server.LCLM in str(raised.value)
    assert "unavailable" in str(raised.value)


# ------------------------------------------------------------- the endpoints


def serving(build):
    """A TestClient over the real app, with fake workers behind it."""
    server.component_factory = build
    return TestClient(server.app)


@pytest.fixture(autouse=True)
def restore_factory():
    original = server.component_factory
    yield
    server.component_factory = original


def test_an_endpoint_backed_by_a_failed_optional_component_returns_503(pool):
    """NEGATIVE CONTROL 4. The alternative is answering an LCLM request from a
    different model and saying nothing about it, which gives the caller a
    plausible answer to a question they did not ask."""
    workers, _, build = pool
    workers[server.LCLM].fail_warmup = True

    with serving(build) as client:
        response = client.post("/generate", json={"lclm": True, "prompt": "hello"})

    assert response.status_code == 503, response.text
    assert server.LCLM in response.json()["detail"]
    assert workers[server.TURBOQUANT].warmed == 1, "the default worker was disturbed"


def test_the_failed_optional_route_does_not_quietly_use_the_other_model(pool):
    workers, _, build = pool
    workers[server.LCLM].fail_warmup = True

    with serving(build) as client:
        response = client.post("/generate", json={"lclm": True, "prompt": "hello"})

    assert response.status_code == 503
    # The TurboQuant fake has no `.generate`, so any silent substitution would
    # have raised rather than answered. The assertion that matters is that the
    # caller was told no.
    assert "text" not in response.json()


def test_a_question_aware_compress_refuses_when_attentionrag_is_unavailable(pool):
    workers, _, build = pool
    workers[server.ATTENTIONRAG].fail_warmup = True

    with serving(build) as client:
        response = client.post("/compress", json={"text": "hello world", "question": "what?"})

    assert response.status_code == 503, response.text
    assert server.ATTENTIONRAG in response.json()["detail"]


def test_an_endpoint_backed_only_by_ready_critical_components_still_works(pool):
    """NEGATIVE CONTROL 1, from the caller's side."""
    workers, _, build = pool
    workers[server.ATTENTIONRAG].fail_warmup = True
    workers[server.LCLM].fail_warmup = True

    async def compress(text, rate=0.5, return_labels=False):
        return {
            "compressed_prompt": "hello", "origin_tokens": 2,
            "compressed_tokens": 1, "rate": "50.0%", "ratio": "2.00x",
        }

    class Remote:
        def __init__(self, fn):
            self.aio = fn

    class Callable_:
        def __init__(self, fn):
            self.remote = Remote(fn)

    def build_with_working_compressor():
        specs = build()
        out = []
        for spec in specs:
            if spec.name != server.COMPRESSOR:
                out.append(spec)
                continue

            async def start():
                worker = workers[server.COMPRESSOR]
                worker.started += 1
                worker.compress = Callable_(compress)
                return worker

            out.append(ComponentSpec(
                name=spec.name, capability=spec.capability, critical=spec.critical,
                start=start, warmup=spec.warmup, cleanup=spec.cleanup,
            ))
        return out

    with serving(build_with_working_compressor) as client:
        response = client.post("/compress", json={"text": "hello world"})

    assert response.status_code == 200, response.text
    assert response.json()["compressed_prompt"] == "hello"


def test_health_reports_a_worker_ready_only_once_it_has_warmed(pool):
    workers, _, build = pool
    workers[server.LCLM].fail_warmup = True

    with serving(build) as client:
        body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["compressor_ready"] is True
    assert body["turboquant_ready"] is True
    assert body["attentionrag_ready"] is True
    assert body["lclm_ready"] is False


def test_ready_answers_200_while_degraded_and_names_what_is_missing(pool):
    workers, _, build = pool
    workers[server.ATTENTIONRAG].fail_warmup = True

    with serving(build) as client:
        response = client.get("/ready")

    assert response.status_code == 200, "a degraded service was taken out of rotation"
    body = response.json()
    assert body["ready"] is True
    assert body["degraded"] is True
    failed = next(c for c in body["components"] if c["name"] == server.ATTENTIONRAG)
    assert failed["state"] == "failed"
    assert "could not warm up" in failed["detail"]


def test_a_critical_failure_stops_the_app_from_starting_at_all(pool):
    workers, _, build = pool
    workers[server.COMPRESSOR].fail_warmup = True

    with pytest.raises(StartupAborted):
        with serving(build) as client:
            client.get("/health")

    for worker in workers.values():
        assert worker.cleaned == 1, f"{worker.name} was left holding its context"


# --------------------------------------------------- the boundary itself


def test_nothing_in_this_module_imports_modal():
    """The fakes cannot contact Modal because nothing here can.

    `server` imports the worker modules inside `modal_components`, not at module
    scope, so importing the app costs nothing and reaches nothing. If that ever
    moves back to the top of the file this fails, which is the point.
    """
    assert "modal" not in sys.modules


def test_no_fake_captured_a_modal_call_or_a_credential(pool):
    workers, recorder, build = pool
    life = Lifecycle(build())
    asyncio.run(life.start())
    asyncio.run(life.stop())

    for worker in workers.values():
        assert not hasattr(worker, "remote"), "a fake grew a Modal-shaped call surface"
    for entry in recorder.log:
        assert ":" in entry and entry.split(":")[1] in ("start", "warm", "clean")
