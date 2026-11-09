"""Bounded schedule search with a full candidate ledger.

Four strategies share one `Measurer`, one budget and one correctness screen,
so a strategy cannot win by spending more compiles or skipping validation:

* `heuristic`        - the existing default schedule, measured once.
* `random_search`    - seeded uniform order over legal candidates.
* `successive_halving` - screen many candidates on short trials, keep the best
  1/eta, re-time them with eta x more trials, repeat; validate finalists.
* `model_guided`     - seeded random warm start, then a ridge cost model on log
  latency with a bootstrap ensemble picks the next candidate by lower
  confidence bound (with epsilon exploration); finalists are re-timed and
  validated exactly as in successive halving. Model fitting time is charged to
  the budget.

Budget is accounted in the measurer's own seconds (compile + timing +
validation wall time as reported by the measurer) plus a compile-count cap.
Compile/launch errors and timeouts are ledger outcomes, not crashes. If no
candidate passes validation the result selects nothing and the caller keeps
its known-safe path; that is a legitimate outcome, recorded with its reason.
"""
from __future__ import annotations

import math
import random
import statistics
import time
from dataclasses import dataclass, field

import numpy as np


class CompileError(RuntimeError):
    pass


class LaunchError(RuntimeError):
    pass


@dataclass
class Budget:
    seconds: float
    compiles: int
    spent_s: float = 0.0
    compiles_used: int = 0

    def left(self) -> bool:
        return self.spent_s < self.seconds and self.compiles_used < self.compiles


@dataclass
class Entry:
    key: str
    schedule: dict
    status: str = "pending"        # timed, compile_error, launch_error, rejected_validation, selected
    detail: str = ""
    compile_s: float = 0.0
    samples_us: list = field(default_factory=list)
    rounds: int = 0
    validation: dict | None = None
    predicted_log_us: float | None = None
    predicted_std: float | None = None
    obj: object = field(default=None, repr=False, compare=False)

    @property
    def median_us(self) -> float:
        return statistics.median(self.samples_us) if self.samples_us else math.inf


@dataclass
class SearchResult:
    strategy: str
    selected: object | None
    reason: str
    ledger: list
    budget: Budget
    model_overhead_s: float = 0.0
    illegal: list = field(default_factory=list)

    def summary(self) -> dict:
        timed = [e for e in self.ledger if e.samples_us]
        return {"strategy": self.strategy, "selected": getattr(self.selected, "key", lambda: None)()
                if self.selected else None, "reason": self.reason,
                "selected_median_us": next((e.median_us for e in self.ledger
                                            if e.status == "selected"), None),
                "candidates_compiled": self.budget.compiles_used,
                "candidates_timed": len(timed),
                "compile_errors": sum(e.status == "compile_error" for e in self.ledger),
                "launch_errors": sum(e.status == "launch_error" for e in self.ledger),
                "rejected_validation": sum(e.status == "rejected_validation" for e in self.ledger),
                "illegal_static": len(self.illegal),
                "spent_s": self.budget.spent_s, "model_overhead_s": self.model_overhead_s}


class Measurer:
    """Interface. Implementations report the seconds each call consumed."""

    def compile(self, schedule) -> float: ...
    def time(self, schedule, trials: int) -> tuple[list[float], float]: ...
    def validate(self, schedule) -> tuple[dict, float]: ...
    def features(self, schedule) -> list[float]: ...


# -- shared mechanics --------------------------------------------------------

def _entry(ledger: dict, s) -> Entry:
    k = s.key()
    if k not in ledger:
        ledger[k] = Entry(k, dict(s.__dict__), obj=s)
    return ledger[k]


def _measure(m: Measurer, s, trials: int, ledger: dict, budget: Budget) -> Entry:
    e = _entry(ledger, s)
    if e.status in ("compile_error", "launch_error"):
        return e
    if e.rounds == 0:
        if budget.compiles_used >= budget.compiles:
            e.detail = "compile budget exhausted"
            return e
        budget.compiles_used += 1
        try:
            dt = m.compile(s)
            e.compile_s = dt
            budget.spent_s += dt
        except CompileError as exc:
            e.status, e.detail = "compile_error", str(exc)[:200]
            budget.spent_s += getattr(exc, "seconds", 0.0)
            return e
    try:
        samples, dt = m.time(s, trials)
        budget.spent_s += dt
        e.samples_us.extend(samples)
        e.rounds += 1
        e.status = "timed"
    except LaunchError as exc:
        e.status, e.detail = "launch_error", str(exc)[:200]
        budget.spent_s += getattr(exc, "seconds", 0.0)
    return e


def _finalize(strategy, m: Measurer, finalists, ledger, budget, illegal, final_trials,
              model_s=0.0) -> SearchResult:
    """Re-time finalists, validate in speed order, select the fastest valid one.

    Final timing and validation are charged to the same budget, but they may
    run past it: a strategy is never allowed to skip the correctness screen.
    """
    timed = [e for e in finalists if e.samples_us]
    for e in timed:
        _measure(m, e.obj, final_trials, ledger, budget)
    for e in sorted(timed, key=lambda e: e.median_us):
        result, dt = m.validate(e.obj)
        budget.spent_s += dt
        e.validation = result
        if result.get("passed"):
            e.status = "selected"
            return SearchResult(strategy, e.obj,
                                f"fastest validated finalist ({e.median_us:.1f} us)",
                                list(ledger.values()), budget, model_s, illegal)
        e.status, e.detail = "rejected_validation", result.get("reason", "")
    return SearchResult(strategy, None, "no candidate passed validation; keep the safe path",
                        list(ledger.values()), budget, model_s, illegal)


# -- strategies ------------------------------------------------------------

def heuristic(m: Measurer, default, budget: Budget, illegal=(), trials=5, final_trials=15):
    ledger = {}
    e = _measure(m, default, trials, ledger, budget)
    return _finalize("heuristic", m, [e], ledger, budget, list(illegal), final_trials)


def random_search(m: Measurer, legal, budget: Budget, seed=0, illegal=(), trials=5,
                  final_trials=15, finalists=3):
    order = list(legal)
    random.Random(seed).shuffle(order)
    ledger = {}
    for s in order:
        if not budget.left():
            break
        _measure(m, s, trials, ledger, budget)
    best = sorted((e for e in ledger.values() if e.samples_us), key=lambda e: e.median_us)
    return _finalize("random", m, best[:finalists], ledger, budget, list(illegal), final_trials)


def successive_halving(m: Measurer, legal, budget: Budget, seed=0, illegal=(), eta=3,
                       r0=2, n0=None, final_trials=15, finalists=3):
    pool = list(legal)
    random.Random(seed).shuffle(pool)
    if n0 is not None:
        pool = pool[:n0]
    ledger = {}
    trials = r0
    while pool and budget.left():
        alive = []
        for s in pool:
            if not budget.left():
                break
            e = _measure(m, s, trials, ledger, budget)
            if e.samples_us:
                alive.append((e.median_us, s))
        alive.sort(key=lambda x: x[0])
        if len(alive) <= finalists:
            pool = [s for _, s in alive]
            break
        pool = [s for _, s in alive[:max(finalists, len(alive) // eta)]]
        trials *= eta
    fin = [ledger[s.key()] for s in pool]
    return _finalize("successive_halving", m, fin, ledger, budget, list(illegal), final_trials)


class CostModel:
    """Ridge regression on log latency, bootstrap ensemble for uncertainty."""

    def __init__(self, n_models=16, alpha=1.0, seed=0):
        self.n, self.alpha, self.rng = n_models, alpha, np.random.default_rng(seed)
        self.models = []

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))])
        self.models = []
        for _ in range(self.n):
            idx = self.rng.integers(0, len(X), len(X))
            A, b = Z[idx], y[idx]
            reg = self.alpha * np.eye(Z.shape[1])
            reg[-1, -1] = 0.0
            self.models.append(np.linalg.solve(A.T @ A + reg, A.T @ b))
        return self

    def predict(self, X):
        Z = np.hstack([(np.asarray(X, float) - self.mu) / self.sd, np.ones((len(X), 1))])
        P = np.stack([Z @ w for w in self.models])
        return P.mean(0), P.std(0)


def model_guided(m: Measurer, legal, budget: Budget, seed=0, illegal=(), warm=8, trials=5,
                 kappa=1.0, epsilon=0.1, final_trials=15, finalists=3):
    rng = random.Random(seed)
    pool = list(legal)
    rng.shuffle(pool)
    ledger = {}
    model_s = 0.0
    for s in pool[:warm]:                                   # deterministic cold start
        if not budget.left():
            break
        _measure(m, s, trials, ledger, budget)
    remaining = [s for s in pool if s.key() not in ledger]
    while remaining and budget.left():
        seen = [e for e in ledger.values() if e.samples_us]
        if len(seen) >= 3 and rng.random() >= epsilon:
            t0 = time.perf_counter()
            cm = CostModel(seed=seed).fit([m.features(e.obj) for e in seen],
                                          [math.log(e.median_us) for e in seen])
            mean, std = cm.predict([m.features(s) for s in remaining])
            i = int(np.argmin(mean - kappa * std))
            dt = time.perf_counter() - t0
            model_s += dt
            budget.spent_s += dt
            s = remaining.pop(i)
            e = _entry(ledger, s)
            e.predicted_log_us, e.predicted_std = float(mean[i]), float(std[i])
        else:
            s = remaining.pop(rng.randrange(len(remaining)))
        _measure(m, s, trials, ledger, budget)
    best = sorted((e for e in ledger.values() if e.samples_us), key=lambda e: e.median_us)
    return _finalize("model_guided", m, best[:finalists], ledger, budget, list(illegal),
                     final_trials, model_s)


def regret(result: SearchResult, best_us: float) -> float | None:
    """Relative slowdown of the selection vs the best measured legal candidate."""
    sel = next((e for e in result.ledger if e.status == "selected"), None)
    return None if sel is None else sel.median_us / best_us - 1.0
