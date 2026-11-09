"""Rebuild the serving study tables from raw loadgen rows and a frozen protocol.

    python report_packed_serving.py --protocol protocol.json --runs RUN_DIR... --out REPORT_DIR

protocol.json (frozen before final measurement, hashed in the report):
    {"ttft_slo_s": float, "itl_p95_slo_s": float, "baseline_arm": str,
     "bootstrap_draws": int, "seed": int, ...}

Per request: completed = status in {eos, length}; TTFT = first token arrival;
ITL p95 = 95th percentile of that request's inter-token gaps; SLO met =
completed and TTFT <= ttft_slo_s and ITL p95 <= itl_p95_slo_s. Every offered
request, refused or failed included, stays in the denominator.

Goodput = SLO-met requests (and their output tokens) per second of the load
window, first offer to last finish. Arms are compared at the same load label
on the same workload ids. The interval is a paired bootstrap over request ids
with the window held fixed: it covers which requests were drawn, not run-to-run
timing noise, which only repeated runs measure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def per_request(row, proto):
    ok = row.get("status") in ("eos", "length")
    times = row.get("token_times") or []
    gaps = [b - a for a, b in zip(times, times[1:])]
    ttft = times[0] if times else None
    itl95 = pct(gaps, 0.95) if gaps else 0.0
    met = bool(ok and ttft is not None and ttft <= proto["ttft_slo_s"]
               and itl95 <= proto["itl_p95_slo_s"])
    return {"id": row["id"], "ok": ok, "status": row.get("status", "missing"), "ttft": ttft,
            "gaps": gaps, "itl95": itl95, "met": met, "out_tokens": len(times),
            "end": row.get("offered_at", 0.0) + row.get("t_end", 0.0),
            "offered_at": row.get("offered_at", 0.0)}


def load_run(d: Path, proto):
    rows = [json.loads(x) for x in (d / "requests.jsonl").read_text().splitlines()]
    meta = json.loads((d / "meta.json").read_text())
    reqs = [per_request(r, proto) for r in rows]
    health = []
    if (d / "health.jsonl").exists():
        health = [json.loads(x) for x in (d / "health.jsonl").read_text().splitlines()]
    smi = d / "nvidia_smi.csv"
    smi_peak = None
    if smi.exists():
        vals = [float(line.split(",")[1]) for line in smi.read_text().splitlines()[1:] if "," in line]
        smi_peak = max(vals) * 2**20 if vals else None
    label = json.loads((d / "label.json").read_text()) if (d / "label.json").exists() else {}
    return {"dir": str(d), "arm": rows[0]["arm"], "load": label.get("load", meta.get("argv", ["?"])[-1]),
            "reqs": reqs, "meta": meta, "health": health, "smi_peak_bytes": smi_peak}


def summarize(run):
    rs = run["reqs"]
    window = max(r["end"] for r in rs) - min(r["offered_at"] for r in rs)
    met = [r for r in rs if r["met"]]
    allgaps = [g for r in rs if r["ok"] for g in r["gaps"]]
    ttfts = [r["ttft"] for r in rs if r["ok"] and r["ttft"] is not None]
    statuses = {}
    for r in rs:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    h = [x for x in run["health"] if "allocated" in x]
    return {
        "arm": run["arm"], "load": run["load"], "offered": len(rs),
        "completed": sum(r["ok"] for r in rs), "slo_met": len(met), "statuses": statuses,
        "window_s": window,
        "goodput_req_s": len(met) / window if window else 0.0,
        "goodput_tok_s": sum(r["out_tokens"] for r in met) / window if window else 0.0,
        "ttft_s": {q: pct(ttfts, p) for q, p in (("p50", .5), ("p95", .95), ("p99", .99))},
        "itl_s": {q: pct(allgaps, p) for q, p in (("p50", .5), ("p95", .95), ("p99", .99))},
        "peak_device_allocated_bytes": max((x["allocated"] for x in h), default=None),
        "peak_device_reserved_bytes": max((x["reserved"] for x in h), default=None),
        "kv_pool_bytes": h[0]["pool_bytes"] if h else None,
        "peak_smi_used_bytes": run["smi_peak_bytes"],
        "dir": run["dir"],
    }


def paired_bootstrap(a, b, draws, seed):
    """Relative goodput change a vs b over shared request ids, windows fixed."""
    ia = {r["id"]: r for r in a["reqs"]}
    ib = {r["id"]: r for r in b["reqs"]}
    ids = sorted(set(ia) & set(ib))
    if not ids:
        return None
    wa = max(r["end"] for r in a["reqs"]) - min(r["offered_at"] for r in a["reqs"])
    wb = max(r["end"] for r in b["reqs"]) - min(r["offered_at"] for r in b["reqs"])
    rng = random.Random(seed)
    out = []
    for _ in range(draws):
        s = [rng.choice(ids) for _ in ids]
        ga = sum(ia[i]["met"] for i in s) / wa
        gb = sum(ib[i]["met"] for i in s) / wb
        out.append((ga - gb) / gb if gb else float("inf") if ga else 0.0)
    out.sort()
    point = ((sum(ia[i]["met"] for i in ids) / wa) - (sum(ib[i]["met"] for i in ids) / wb))
    base = sum(ib[i]["met"] for i in ids) / wb
    return {"pairs": len(ids), "relative_change": point / base if base else None,
            "ci95": [out[int(0.025 * draws)], out[int(0.975 * draws) - 1]]}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Serving study report from raw rows.")
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    proto_bytes = Path(args.protocol).read_bytes()
    proto = json.loads(proto_bytes)
    runs = [load_run(Path(d), proto) for d in args.runs]
    summ = [summarize(r) for r in runs]
    comps = []
    for r, s in zip(runs, summ):
        if r["arm"] == proto["baseline_arm"]:
            continue
        base = next((x for x in runs if x["arm"] == proto["baseline_arm"] and x["load"] == r["load"]), None)
        if base is not None:
            comps.append({"arm": r["arm"], "vs": base["arm"], "load": r["load"],
                          **(paired_bootstrap(r, base, proto.get("bootstrap_draws", 2000),
                                              proto.get("seed", 0)) or {})})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    report = {"protocol_sha256": hashlib.sha256(proto_bytes).hexdigest(), "protocol": proto,
              "runs": summ, "comparisons": comps}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    f = lambda x, n=3: "-" if x is None else f"{x:.{n}f}"
    lines = [f"Protocol sha256 `{report['protocol_sha256']}`; TTFT SLO {proto['ttft_slo_s']} s, "
             f"ITL p95 SLO {proto['itl_p95_slo_s']} s.", "",
             "| arm | load | offered | completed | SLO met | goodput req/s | goodput tok/s | "
             "TTFT p50/p95/p99 s | ITL p50/p95/p99 s | peak alloc GiB | KV pool GiB |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in sorted(summ, key=lambda s: (str(s["load"]), s["arm"])):
        g = lambda b: "-" if b is None else f"{b / 2**30:.2f}"
        lines.append(f"| {s['arm']} | {s['load']} | {s['offered']} | {s['completed']} | {s['slo_met']} | "
                     f"{f(s['goodput_req_s'])} | {f(s['goodput_tok_s'], 1)} | "
                     f"{f(s['ttft_s']['p50'])}/{f(s['ttft_s']['p95'])}/{f(s['ttft_s']['p99'])} | "
                     f"{f(s['itl_s']['p50'], 4)}/{f(s['itl_s']['p95'], 4)}/{f(s['itl_s']['p99'], 4)} | "
                     f"{g(s['peak_device_allocated_bytes'] or s['peak_smi_used_bytes'])} | {g(s['kv_pool_bytes'])} |")
    if comps:
        lines += ["", "| arm | vs | load | pairs | goodput change | paired bootstrap 95% |", "|---|---|---|---|---|---|"]
        for c in comps:
            rc = c.get("relative_change")
            ci = c.get("ci95") or [None, None]
            lines.append(f"| {c['arm']} | {c['vs']} | {c['load']} | {c.get('pairs')} | "
                         f"{'-' if rc is None else f'{rc:+.1%}'} | "
                         f"{'-' if ci[0] is None else f'[{ci[0]:+.1%}, {ci[1]:+.1%}]'} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
