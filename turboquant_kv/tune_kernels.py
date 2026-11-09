"""Offline tuning, promotion and held-out evaluation for the two tuned kernels.

    python tune_kernels.py plan --target sm80 --out RUN/plan.json          # CPU: legal space
    python tune_kernels.py tune --protocol P.json --registry REG --out RUN/tune     # GPU
    python tune_kernels.py heldout --protocol P.json --registry REG --out RUN/heldout  # GPU

protocol.json fixes, before any measurement: tuning shapes, held-out shapes,
the per-strategy budget (compiles and seconds), timing trials, seeds and the
promotion margin. `tune` runs heuristic / random / successive halving /
model-guided search with the same budget and correctness screen per tuning
shape, plus an exhaustive sweep of the (small) legal space as the oracle used
to compute regret; it promotes the model-guided selection only when it beats
the heuristic by the margin and validates. `heldout` then measures, on shapes
never used for tuning, the default schedule, whatever the dispatcher selects
from the registry, and the exhaustive best, and reports per-shape regret and
speedup (no geometric mean without the per-shape table).

Timing: CUDA events around `iters` back-to-back calls after warmup; the first
launch of a new kernel variant is timed separately as its compile cost.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
from tsc import schedule as sch
from tsc import search


# -- inputs (random but well-formed; correctness is the validator's job) ------

def attn_inputs(p: sch.AttnProblem, device, seed=0):
    import packed_attention as pa
    from cache import _get_quantizer
    g = torch.Generator(device="cpu").manual_seed(seed)
    S, D = p.page_size, p.head_dim
    pages = p.batch * p.max_pages
    perm = torch.randperm(pages, generator=g).to(torch.int32)
    table = perm.view(p.batch, p.max_pages).to(device)
    lens = torch.full((p.batch,), p.seq_len, dtype=torch.int32, device=device)
    q = torch.randn(p.batch, p.hq, 1, D, generator=g).to(device, torch.bfloat16)
    if p.dense:
        kp = torch.randn(pages, p.hkv, S, D, generator=g).to(device, torch.bfloat16)
        vp = torch.randn(pages, p.hkv, S, D, generator=g).to(device, torch.bfloat16)
        return q, pa.DenseKV(kp, vp, table, lens, D)
    nb = D * p.bit_width // 8
    kp = torch.randint(0, 256, (pages, p.hkv, S, nb), generator=g, dtype=torch.uint8).to(device)
    vp = torch.randint(0, 256, (pages, p.hkv, S, nb), generator=g, dtype=torch.uint8).to(device)
    kn = (torch.rand(pages, p.hkv, S, generator=g) * 20).half().to(device)
    vn = (torch.rand(pages, p.hkv, S, generator=g) * 20).half().to(device)
    tq = _get_quantizer(p.bit_width, D, device, 42)
    quant = pa.Quantizer(tq.Pi, tq.centroids, p.bit_width)
    return q, pa.PackedKV(kp, kn, vp, vn, table, lens, quant, quant, D)


def cuda_time(fn, trials, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    t0 = time.perf_counter()
    for _ in range(trials):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) * 1e3 / iters)
    return out, time.perf_counter() - t0


def first_launch_s(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def variant(s) -> str:
    """Which Triton binary a schedule needs: constexprs + launch meta, not splits."""
    d = dict(s.__dict__)
    d.pop("num_splits", None)
    return json.dumps(d, sort_keys=True)


class AttnMeasurer(search.Measurer):
    """`cold_compile_s` maps variant -> seconds measured cold (empty Triton
    cache) during the exhaustive sweep. Strategies are charged that recorded
    cost per new variant, because a later strategy in the same process would
    otherwise get in-memory cache hits and look cheaper than it is."""

    def __init__(self, problem, device="cuda", iters=20, cold_compile_s=None):
        import packed_attention as pa
        import packed_attention_triton as pt
        self.p, self.dev, self.iters = problem, device, iters
        self.q, self.kv = attn_inputs(problem, device)
        self.scale = pa.default_scale(problem.head_dim)
        self.fn = pt.dense_decode_attention if problem.dense else pt.packed_decode_attention
        self.cold = cold_compile_s

    def call(self, s):
        return lambda: self.fn(self.q, self.kv, self.scale, **s.kwargs())

    def compile(self, s):
        try:
            dt = first_launch_s(self.call(s))
        except Exception as exc:                                    # noqa: BLE001
            err = search.CompileError(f"{type(exc).__name__}: {exc}")
            err.seconds = (self.cold or {}).get(variant(s), 0.0)
            raise err from exc
        return self.cold.get(variant(s), dt) if self.cold else dt

    def time(self, s, trials):
        try:
            return cuda_time(self.call(s), trials, self.iters)
        except Exception as exc:                                    # noqa: BLE001
            raise search.LaunchError(f"{type(exc).__name__}: {exc}") from exc

    def validate(self, s):
        from tsc.validation import validate_attention
        t0 = time.perf_counter()
        if self.p.dense:
            return {"passed": True, "reason": "dense arm: validated by test_dense_kernel_matches_sdpa"}, 0.0
        r = validate_attention(s, hq=self.p.hq, hkv=self.p.hkv, head_dim=self.p.head_dim,
                               page_size=self.p.page_size, bit_width=self.p.bit_width,
                               device=self.dev)
        return r.to_dict(), time.perf_counter() - t0

    def features(self, s):
        sms = torch.cuda.get_device_properties(self.dev).multi_processor_count
        progs = self.p.batch * self.p.hkv * s.num_splits
        iters = math.ceil(self.p.max_pages / s.pages_per_iter / s.num_splits)
        return [math.log2(s.num_splits), math.log2(s.pages_per_iter), math.log2(s.num_warps),
                s.num_stages, math.log2(progs / sms), math.log2(max(iters, 1)),
                math.log2(progs / sms) ** 2, math.log2(max(iters, 1)) * math.log2(s.pages_per_iter)]


class DequantMeasurer(search.Measurer):
    def __init__(self, problem: sch.DequantProblem, device="cuda", iters=20, cold_compile_s=None):
        from packing import packed_bytes
        from tsc.compiler import compile_source
        from tsc.examples import example_source
        from tsc.runtime import load_generated
        from cache import _get_quantizer
        self.p, self.dev, self.iters, self.cold = problem, device, iters, cold_compile_s
        res = compile_source(example_source(problem.bit_width, problem.head_dim))
        self.generated_sha = hashlib.sha256(res.generated.encode()).hexdigest()
        self.mod = load_generated(res)
        g = torch.Generator().manual_seed(0)
        nb = packed_bytes(problem.head_dim, problem.bit_width)
        self.packed = torch.randint(0, 256, (problem.rows, nb), generator=g, dtype=torch.uint8).to(device)
        self.norms = (torch.rand(problem.rows, generator=g) * 20).half().to(device)
        tq = _get_quantizer(problem.bit_width, problem.head_dim, device, 42)
        self.cent, self.pi = tq.centroids, tq.Pi
        self.out = torch.empty(problem.rows, problem.head_dim, dtype=torch.bfloat16, device=device)

    def call(self, s):
        return lambda: self.mod.run(self.packed, self.norms, self.cent, self.pi, out=self.out,
                                    block_n=s.block_n, num_warps=s.num_warps,
                                    allow_tf32=self.p.contract == "tf32")

    def handwritten(self):
        from kernel import tq_dequant
        return lambda: tq_dequant(self.packed, self.norms, self.cent, self.pi, self.p.bit_width,
                                  self.p.head_dim, out=self.out, allow_tf32=self.p.contract == "tf32")

    compile = AttnMeasurer.compile
    time = AttnMeasurer.time

    def validate(self, s):
        from tsc.validation import validate_dequant
        t0 = time.perf_counter()
        r = validate_dequant(self.mod.run, s, bit_width=self.p.bit_width, head_dim=self.p.head_dim,
                             device=self.dev, contract=self.p.contract)
        return r.to_dict(), time.perf_counter() - t0

    def features(self, s):
        sms = torch.cuda.get_device_properties(self.dev).multi_processor_count
        progs = math.ceil(self.p.rows / s.block_n)
        return [math.log2(s.block_n), math.log2(s.num_warps), math.log2(progs / sms),
                s.block_n * self.p.head_dim / (s.num_warps * 32), math.log2(s.block_n) ** 2]


# -- commands -------------------------------------------------------------

def env_record():
    return {"torch": torch.__version__, "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "cc": list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
            "argv": sys.argv}


def cmd_plan(a):
    t = sch.TARGETS[a.target]
    proto = json.loads(Path(a.protocol).read_text()) if a.protocol else None
    shapes = proto["attention"]["tune_shapes"] if proto else [[1, 1024], [16, 8192]]
    out = {"target": a.target, "attention": [], "dequant": []}
    for b, L in shapes:
        p = sch.AttnProblem(b, L, **(proto or {}).get("family", {"hq": 28, "hkv": 4, "head_dim": 128,
                                                                    "page_size": 16, "bit_width": 8}))
        legal, bad = sch.enumerate_legal(sch.ATTN_SPACE, p, t, sch.attn_legal)
        out["attention"].append({"batch": b, "seq_len": L, "legal": [s.key() for s in legal],
                                 "rejected": [[s.key(), r] for s, r in bad]})
    for rows in (proto["dequant"]["tune_rows"] if proto else [4096, 65536]):
        p = sch.DequantProblem(rows, 128, 8)
        legal, bad = sch.enumerate_legal(sch.DEQUANT_SPACE, p, t, sch.dequant_legal)
        out["dequant"].append({"rows": rows, "legal": [s.key() for s in legal],
                               "rejected": [[s.key(), r] for s, r in bad]})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    for k in ("attention", "dequant"):
        for row in out[k]:
            print(k, {x: row[x] for x in row if x in ("batch", "seq_len", "rows")},
                  f"legal {len(row['legal'])} rejected {len(row['rejected'])}")


def _ledger(res):
    return [{**e.__dict__, "obj": None, "median_us": e.median_us if e.samples_us else None}
            for e in res.ledger]


def cold_table(ex) -> dict:
    return {r["variant"]: r["compile_s"] for r in ex if r.get("cold") and "compile_s" in r}


def _search_all(meas_factory, legal, illegal, heuristic, budget_cfg, seeds):
    out = {}
    for name, fn in (("random", search.random_search), ("successive_halving", search.successive_halving),
                     ("model_guided", search.model_guided)):
        runs = []
        for seed in seeds:
            b = search.Budget(budget_cfg["seconds"], budget_cfg["compiles"])
            r = fn(meas_factory(), legal, b, seed=seed, illegal=illegal)
            runs.append({"seed": seed, **r.summary(), "ledger": _ledger(r),
                         "selected_schedule": r.selected.__dict__ if r.selected else None})
        out[name] = runs
    b = search.Budget(budget_cfg["seconds"], budget_cfg["compiles"])
    r = search.heuristic(meas_factory(), heuristic, b)
    out["heuristic"] = [{"seed": None, **r.summary(), "ledger": _ledger(r),
                         "selected_schedule": r.selected.__dict__ if r.selected else None}]
    return out


def _exhaustive(m, legal, trials):
    """Run first, in a fresh process with an empty TRITON_CACHE_DIR, so the
    first launch of each variant is a cold compile."""
    rows, seen = [], set()
    for s in legal:
        try:
            c = m.compile(s)
            cold = variant(s) not in seen
            seen.add(variant(s))
            samples, _ = m.time(s, trials)
            rows.append({"schedule": s.__dict__, "key": s.key(), "compile_s": c, "cold": cold,
                         "variant": variant(s),
                         "median_us": statistics.median(samples), "samples_us": samples})
        except (search.CompileError, search.LaunchError) as exc:
            rows.append({"schedule": s.__dict__, "key": s.key(), "error": str(exc)[:200]})
    return rows


def cmd_tune(a):
    import tuned_dispatch as td
    from tsc.promotion import PromotionError, Registry
    proto = json.loads(Path(a.protocol).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "env.json").write_text(json.dumps(env_record(), indent=2) + "\n")
    tgt = td.target_of("cuda")
    target = sch.target_for(tuple(tgt["cc"]))
    reg = Registry(a.registry)
    fam = proto["family"]
    seeds = proto["seeds"]
    budget = proto["budget"]
    summary = {"attention": [], "dequant": []}

    # Attention: per tuning shape, all strategies + exhaustive oracle; promote buckets.
    for dense in (False, True):
        buckets, base_meds, cand_meds = [], [], []
        for b, L in proto["attention"]["tune_shapes"]:
            p = sch.AttnProblem(b, L, dense=dense, **fam)
            legal, bad = sch.enumerate_legal(sch.ATTN_SPACE, p, target, sch.attn_legal)
            ex = _exhaustive(AttnMeasurer(p), legal, proto["exhaustive_trials"])
            cold = cold_table(ex)
            # Strategy comparison on the packed kernel; the dense baseline only needs
            # its exhaustive best (the track 01 carryover's split-swept dense arm).
            res = (_search_all(lambda: AttnMeasurer(p, cold_compile_s=cold), legal, bad,
                               sch.HEURISTIC_ATTN, budget, seeds) if not dense else
                   {"model_guided": [{"selected_schedule": min(
                       (r for r in ex if "median_us" in r), key=lambda r: r["median_us"])["schedule"]}]})
            best = min((r for r in ex if "median_us" in r), key=lambda r: r["median_us"])
            heur = next(r for r in ex if r["key"] == sch.HEURISTIC_ATTN.key())
            chosen = res["model_guided"][0]["selected_schedule"]
            tag = f"{'dense' if dense else 'packed'}_b{b}_L{L}"
            (out / f"attention_{tag}.json").write_text(json.dumps(
                {"problem": p.__dict__, "illegal": [[s.key(), r] for s, r in bad],
                 "search": res, "exhaustive": ex}, indent=1) + "\n")
            cmed = next((r["median_us"] for r in ex if chosen and r["key"] ==
                         sch.AttnSchedule(**chosen).key()), None)
            row = {"tag": tag, "legal": len(legal), "illegal": len(bad), "cold_compile_s": cold,
                   "heuristic_us": heur["median_us"], "exhaustive_best_us": best["median_us"],
                   "exhaustive_best": best["key"], "model_guided_selected": chosen,
                   "model_guided_us": cmed,
                   "strategies": {k: [{x: v[x] for x in ("seed", "selected", "selected_median_us",
                                                          "candidates_compiled", "compile_errors",
                                                          "launch_errors", "rejected_validation",
                                                          "spent_s", "model_overhead_s")} for v in vs]
                                  for k, vs in res.items()} if not dense else {"dense": "exhaustive best"}}
            summary["attention"].append(row)
            if chosen and cmed and cmed < heur["median_us"] * (1 - proto["promotion_margin"]):
                buckets.append({"batch_max": b, "len_max": L, "schedule": chosen})
                base_meds.append(heur["median_us"])
                cand_meds.append(cmed)
        buckets.sort(key=lambda x: (x["batch_max"], x["len_max"]))
        if buckets:
            # Every promoted schedule must also pass validation (dense: kernel tests).
            vals = [AttnMeasurer(sch.AttnProblem(1, 1024, dense=dense, **fam)).validate(
                sch.AttnSchedule(**bk["schedule"]))[0] for bk in buckets]
            ok = all(v.get("passed") for v in vals)
            p0 = sch.AttnProblem(1, 1024, dense=dense, **fam)
            ident = td.attention_identity(p0.family(), tgt)
            shapes = proto["attention"]["tune_shapes"]
            try:
                path = reg.promote(
                    ident, {"buckets": buckets},
                    {"passed": ok, "reason": "all bucket schedules validated" if ok else "a bucket failed",
                     "per_bucket": vals},
                    {"baseline_median_us": statistics.median(base_meds),
                     "candidate_median_us": statistics.median(cand_meds),
                     "samples_path": str(out)},
                    {"batch": {"min": 1, "max": max(s[0] for s in shapes)},
                     "seq_len": {"min": 1, "max": max(s[1] for s in shapes)}})
                summary.setdefault("promoted", []).append(str(path))
            except PromotionError as exc:
                summary.setdefault("not_promoted", []).append(f"{'dense' if dense else 'packed'}: {exc}")

    # Dequantizer: generated kernel vs handwritten at the default schedule.
    for rows in proto["dequant"]["tune_rows"]:
        p = sch.DequantProblem(rows, fam["head_dim"], fam["bit_width"])
        legal, bad = sch.enumerate_legal(sch.DEQUANT_SPACE, p, target, sch.dequant_legal)
        m = DequantMeasurer(p)
        ex = _exhaustive(m, legal, proto["exhaustive_trials"])
        cold = cold_table(ex)
        res = _search_all(lambda: DequantMeasurer(p, cold_compile_s=cold), legal, bad,
                          sch.HEURISTIC_DEQUANT, budget, seeds)
        hand, _ = cuda_time(m.handwritten(), proto["exhaustive_trials"], m.iters)
        best = min((r for r in ex if "median_us" in r), key=lambda r: r["median_us"])
        chosen = res["model_guided"][0]["selected_schedule"]
        cmed = next((r["median_us"] for r in ex if chosen and r["key"] ==
                     sch.DequantSchedule(**chosen).key()), None)
        (out / f"dequant_rows{rows}.json").write_text(json.dumps(
            {"problem": p.__dict__, "search": res, "exhaustive": ex, "handwritten_samples_us": hand},
            indent=1) + "\n")
        row = {"rows": rows, "handwritten_us": statistics.median(hand),
               "generated_default_us": next(r["median_us"] for r in ex if r["key"] == sch.HEURISTIC_DEQUANT.key()),
               "exhaustive_best_us": best["median_us"], "exhaustive_best": best["key"],
               "model_guided_selected": chosen, "model_guided_us": cmed}
        summary["dequant"].append(row)
        if chosen and cmed and cmed < row["handwritten_us"] * (1 - proto["promotion_margin"]):
            val, _ = m.validate(sch.DequantSchedule(**chosen))
            try:
                path = reg.promote(td.dequant_identity(p.bit_width, p.head_dim, p.contract, tgt), chosen, val,
                                   {"baseline_median_us": row["handwritten_us"], "candidate_median_us": cmed,
                                    "samples_path": str(out)},
                                   {"rows": {"min": 1, "max": max(proto["dequant"]["tune_rows"])}},
                                   extra={"generated_source_sha": m.generated_sha})
                summary.setdefault("promoted", []).append(str(path))
            except PromotionError as exc:
                summary.setdefault("not_promoted", []).append(f"dequant rows {rows}: {exc}")
        else:
            summary.setdefault("not_promoted", []).append(
                f"dequant rows {rows}: no candidate beat the handwritten kernel by the margin")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k in ("promoted", "not_promoted")}, indent=1))


def cmd_heldout(a):
    import tuned_dispatch as td
    proto = json.loads(Path(a.protocol).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    tgt = td.target_of("cuda")
    target = sch.target_for(tuple(tgt["cc"]))
    disp = td.TunedDispatch(a.registry)
    rows = []
    for dense in (False, True):
        for b, L in proto["attention"]["heldout_shapes"]:
            p = sch.AttnProblem(b, L, dense=dense, **proto["family"])
            legal, _ = sch.enumerate_legal(sch.ATTN_SPACE, p, target, sch.attn_legal)
            m = AttnMeasurer(p)
            ex = _exhaustive(m, legal, proto["exhaustive_trials"])
            meds = {r["key"]: r["median_us"] for r in ex if "median_us" in r}
            dec = disp.attention_schedule(p.family(), b, L, "cuda")
            sel = sch.AttnSchedule(**dec.schedule) if dec.schedule else sch.HEURISTIC_ATTN
            best_key = min(meds, key=meds.get)
            rows.append({"arm": "dense" if dense else "packed", "batch": b, "seq_len": L,
                         "decision": dec.path, "reason": dec.reason, "selected": sel.key(),
                         "selected_us": meds.get(sel.key()), "default_us": meds.get(sch.HEURISTIC_ATTN.key()),
                         "best_us": meds[best_key], "best": best_key,
                         "speedup_vs_default": meds.get(sch.HEURISTIC_ATTN.key()) / meds.get(sel.key())
                         if meds.get(sel.key()) else None,
                         "regret_vs_best": meds.get(sel.key()) / meds[best_key] - 1 if meds.get(sel.key()) else None})
            (out / f"heldout_{rows[-1]['arm']}_b{b}_L{L}.json").write_text(json.dumps(ex) + "\n")
    (out / "heldout.json").write_text(json.dumps({"env": env_record(), "rows": rows}, indent=2) + "\n")
    for r in rows:
        print(r["arm"], r["batch"], r["seq_len"], r["decision"], r["selected"],
              f"{r['selected_us']:.0f}us" if r["selected_us"] else "-", f"default {r['default_us']:.0f}us",
              f"best {r['best_us']:.0f}us")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--target", default="sm80", choices=sorted(sch.TARGETS))
    p.add_argument("--protocol")
    p.add_argument("--out", required=True)
    for name in ("tune", "heldout"):
        q = sub.add_parser(name)
        q.add_argument("--protocol", required=True)
        q.add_argument("--registry", required=True)
        q.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd != "plan" and not torch.cuda.is_available():
        raise SystemExit("BLOCKED: tuning and held-out timing need CUDA")
    {"plan": cmd_plan, "tune": cmd_tune, "heldout": cmd_heldout}[a.cmd](a)


if __name__ == "__main__":
    main()
