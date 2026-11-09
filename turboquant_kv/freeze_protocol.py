"""Freeze the serving protocol: pre-registered rules + SLOs from calibration.

    python freeze_protocol.py --rules RULES.json --calibration CAL_RUN --out protocol.json

The rules file is written before any GPU measurement. It fixes how SLOs are
derived (multipliers on the dense arm's concurrency-1 calibration medians), the
arms, rates, budgets, repetitions, quality margin and win criterion. This step
only applies that rule mechanically to the calibration rows and refuses to
overwrite an existing protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from report_packed_serving import pct


def main(argv=None):
    ap = argparse.ArgumentParser(description="Freeze protocol.json from rules and calibration.")
    ap.add_argument("--rules", required=True)
    ap.add_argument("--calibration", required=True, help="loadgen output dir of the calibration run")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; a frozen protocol is never rewritten")
    rules_bytes = Path(a.rules).read_bytes()
    rules = json.loads(rules_bytes)
    rows_path = Path(a.calibration) / "requests.jsonl"
    rows = [json.loads(x) for x in rows_path.read_text().splitlines()]
    ok = [r for r in rows if r.get("status") in ("eos", "length") and r.get("token_times")]
    if len(ok) < max(3, len(rows) // 2):
        raise SystemExit(f"calibration too thin: {len(ok)}/{len(rows)} completed")
    ttft = statistics.median(r["token_times"][0] for r in ok)
    itl95 = statistics.median(pct([b - a for a, b in zip(r["token_times"], r["token_times"][1:])], 0.95)
                              for r in ok)
    proto = {**rules,
             "ttft_slo_s": round(rules["ttft_multiplier"] * ttft, 4),
             "itl_p95_slo_s": round(rules["itl_multiplier"] * itl95, 5),
             "derived_from": {"calibration_rows_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
                              "completed": len(ok), "offered": len(rows),
                              "median_ttft_s": ttft, "median_request_itl_p95_s": itl95},
             "rules_sha256": hashlib.sha256(rules_bytes).hexdigest(),
             "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out.write_text(json.dumps(proto, indent=2) + "\n")
    print(json.dumps({k: proto[k] for k in ("ttft_slo_s", "itl_p95_slo_s")}))


if __name__ == "__main__":
    main()
