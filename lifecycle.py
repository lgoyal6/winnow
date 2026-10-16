"""Which workers this service has, which ones it cannot serve without, and what
it says when one of them is missing.

The lifespan used to start four Modal apps and warm all four together, which
meant any one of them failing took the whole service down -- including the three
that were fine. Two of the four are not worth an outage for: AttentionRAG makes
`/compress` question-aware and LCLM adds a second generation route, and the
service still does its main job without either.

So a component declares whether it is critical, and that is the only thing that
differs when one fails:

* **critical** -- startup aborts, and every component that already started is
  closed, in reverse order, before the exception leaves. A half-started process
  that answers requests is worse than one that never came up.
* **optional** -- the failure is recorded, that one component is cleaned up, and
  everything else carries on. Endpoints that needed it answer 503 and name the
  capability; endpoints that did not are untouched.

Nothing in this file knows what Modal is. A component is a name and three
callables, which is what lets every path above be driven in a test with no GPU,
no account, no credential and no network.

There is no flag anywhere here. A component is ready because its warmup returned,
and degraded because something observably failed -- never because a setting said
so. A readiness endpoint that can be told what to say is not a readiness
endpoint.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ComponentState(str, Enum):
    """Where a component is. Only ever set from something that happened."""

    STARTING = "starting"
    """Its context is being acquired, or it is warming. Not usable yet, and
    readiness must not round this up."""

    READY = "ready"
    """Warmup returned. This is the only state an endpoint may be served from."""

    FAILED = "failed"
    """Start or warmup raised. Cleaned up already if it had got far enough to
    need it."""

    STOPPED = "stopped"
    """Never started, or closed on the way out."""


@dataclass(frozen=True)
class ComponentSpec:
    """One worker, as three callables.

    `start` acquires whatever the worker needs and returns the handle endpoints
    will use. `warmup` is given that handle and returns when the worker can
    actually answer -- this is the step that must not be skipped, because a
    handle exists long before a model is loaded. `cleanup` releases what `start`
    took, and is the reason a failed startup does not leak containers.
    """

    name: str
    capability: str
    """What this component lets the service do, phrased for the person reading a
    503. "lclm" is a variable name; "LCLM long-context generation" tells them
    which request to stop sending."""

    critical: bool
    start: Callable[[], Awaitable[Any]]
    warmup: Callable[[Any], Awaitable[Any]]
    cleanup: Callable[[Any], Awaitable[Any]] | None = None


@dataclass
class ComponentStatus:
    """What is currently known about one component."""

    name: str
    capability: str
    critical: bool
    state: ComponentState = ComponentState.STOPPED
    detail: str = ""

    @property
    def ready(self) -> bool:
        return self.state is ComponentState.READY

    def as_dict(self) -> dict:
        out = {
            "name": self.name,
            "capability": self.capability,
            "critical": self.critical,
            "state": self.state.value,
            "ready": self.ready,
        }
        if self.detail:
            out["detail"] = self.detail
        return out


class ComponentUnavailable(Exception):
    """Raised when an endpoint asks for a component that is not ready.

    Carries the capability rather than only the name, so the handler can turn it
    into a 503 a caller can act on without the handler needing to know anything
    about which worker that was.
    """

    def __init__(self, status: ComponentStatus) -> None:
        self.status = status
        message = f"{status.capability} is unavailable ({status.state.value})"
        if status.detail:
            message = f"{message}: {status.detail}"
        super().__init__(message)


class StartupAborted(RuntimeError):
    """A critical component failed. Everything that started has been closed."""


_SECRET_RUN = re.compile(r"[A-Za-z0-9_\-]{24,}")
"""What an API token looks like in an exception message.

Failure detail is served from a readiness endpoint, which is the one place a
credential leaves without anybody thinking of it as an export. A provider
library that raises "invalid token sk-ant-..." would otherwise publish the
token. Twenty-four characters is long enough that ordinary words, module paths
and class names survive intact.
"""


def _redact(text: str) -> str:
    return _SECRET_RUN.sub("[redacted]", text)


def _detail(what: str, exc: BaseException, limit: int = 200) -> str:
    """One line, no traceback, no secrets.

    The full exception belongs in the process log, where it is already going.
    What a readiness probe needs is enough to tell two failures apart.
    """
    lines = str(exc).strip().splitlines()
    message = lines[0] if lines else ""
    return f"{what}: {type(exc).__name__}: {_redact(message)}"[:limit]


@dataclass
class Lifecycle:
    """Starts a set of components, and knows which of them are usable.

    Start order is declaration order, because a context that is cheap to acquire
    should not wait behind one that is not. Warmups run concurrently once every
    context is up, which is what the hard-coded lifespan did and is worth
    keeping: four models loading in series is four cold starts end to end.
    """

    specs: Sequence[ComponentSpec]
    _status: dict[str, ComponentStatus] = field(init=False)
    _handles: dict[str, Any] = field(default_factory=dict, init=False)
    _started: list[ComponentSpec] = field(default_factory=list, init=False)
    _closed: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self.specs = tuple(self.specs)
        self._status = {
            spec.name: ComponentStatus(
                name=spec.name, capability=spec.capability, critical=spec.critical
            )
            for spec in self.specs
        }

    # ------------------------------------------------------------- startup

    async def start(self) -> None:
        """Bring everything up, or bring nothing up.

        "Nothing" is the important half. If a critical component fails, this
        closes every component that already started before the exception leaves,
        so the process does not survive holding containers it will never use.
        """
        for spec in self.specs:
            self._status[spec.name].state = ComponentState.STARTING
            try:
                self._handles[spec.name] = await spec.start()
            except Exception as exc:  # noqa: BLE001 - any failure is a failure
                self._fail(spec, "could not start", exc)
                if spec.critical:
                    await self._close_all()
                    raise StartupAborted(
                        f"{spec.capability} could not start, so the service did not"
                    ) from exc
                continue
            self._started.append(spec)

        warming = list(self._started)
        results = await asyncio.gather(
            *(spec.warmup(self._handles[spec.name]) for spec in warming),
            return_exceptions=True,
        )

        # Every warmup is waited for before anything is decided, even when one
        # has already failed. Raising early would leave the others loading
        # models into a process that is about to exit.
        failed_critical: tuple[ComponentSpec, BaseException] | None = None
        for spec, result in zip(warming, results):
            if isinstance(result, BaseException):
                self._fail(spec, "could not warm up", result)
                if spec.critical:
                    failed_critical = failed_critical or (spec, result)
                else:
                    await self._close(spec)
                continue
            self._status[spec.name].state = ComponentState.READY

        if failed_critical is not None:
            spec, exc = failed_critical
            await self._close_all()
            raise StartupAborted(
                f"{spec.capability} could not warm up, so the service did not"
            ) from exc

    # ------------------------------------------------------------ shutdown

    async def stop(self) -> None:
        """Close everything that started, once each, and wait for it."""
        await self._close_all()

    async def _close_all(self) -> None:
        for spec in reversed(self._started):
            await self._close(spec)

    async def _close(self, spec: ComponentSpec) -> None:
        if spec.name in self._closed:
            return
        self._closed.add(spec.name)
        handle = self._handles.pop(spec.name, None)
        if spec.cleanup is not None:
            try:
                await spec.cleanup(handle)
            except Exception as exc:  # noqa: BLE001
                # Best effort, and it must not stop the next one closing: a
                # cleanup that raises would otherwise strand every component
                # below it in the stack.
                status = self._status[spec.name]
                status.detail = _detail("cleanup failed", exc)
        status = self._status[spec.name]
        if status.state is not ComponentState.FAILED:
            status.state = ComponentState.STOPPED

    def _fail(self, spec: ComponentSpec, what: str, exc: BaseException) -> None:
        status = self._status[spec.name]
        status.state = ComponentState.FAILED
        status.detail = _detail(what, exc)

    # --------------------------------------------------------------- using

    def handle(self, name: str) -> Any:
        """The worker behind this component, or a refusal naming what is missing.

        Ready or nothing. Returning a handle for a component that is merely
        started would hand an endpoint a worker with no model in it, and the
        caller would see a timeout instead of an answer.
        """
        status = self._status[name]
        if not status.ready:
            raise ComponentUnavailable(status)
        return self._handles[name]

    def status(self, name: str) -> ComponentStatus:
        return self._status[name]

    # ----------------------------------------------------------- reporting

    @property
    def ready(self) -> bool:
        """Whether every component the service cannot work without is usable."""
        return all(s.ready for s in self._status.values() if s.critical)

    @property
    def degraded(self) -> bool:
        """Whether anything at all is not usable.

        Separate from `ready` on purpose: a service missing only an optional
        worker is serving, and saying so while also saying something is wrong is
        the honest pair of facts. Collapsing them would either page somebody for
        a working service or hide a dead worker behind a green light.
        """
        return not all(s.ready for s in self._status.values())

    def readiness(self) -> dict:
        return {
            "ready": self.ready,
            "degraded": self.degraded,
            "components": [self._status[spec.name].as_dict() for spec in self.specs],
        }
