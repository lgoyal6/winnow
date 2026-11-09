"""Rebuild Track 02 tables from saved GPU evidence, without running kernels.

python report_tuning.py --evidence RUN/evidence02 --protocol protocol02.json \
    --model-contract RUN/evidence01/model-contract-v4/model_contract.json --out REPORT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def build(evidence: Path, protocol: Path, contract: Path) -> dict:
    inputs = {}

    def read(path):
        raw = path.read_bytes()
        inputs[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    proto, model = read(protocol), read(contract)
    layers = model["layers"]["layers"]
    if not isinstance(layers, int) or layers <= 0:
        raise ValueError("invalid model layer count")
    stage_path = evidence / "stages.jsonl"
    raw = stage_path.read_bytes()
    inputs[str(stage_path)] = hashlib.sha256(raw).hexdigest()
    stages = [json.loads(line) for line in raw.splitlines()]
    tune_stages = [s for s in stages if s.get("stage") == "tune"]
    if not tune_stages or tune_stages[-1].get("exit") != 0:
        raise ValueError("successful tuning stage required")
    tune_seconds = sum(s["seconds"] for s in tune_stages)
    if not math.isfinite(tune_seconds) or tune_seconds < 0:
        raise ValueError("invalid measured tuning cost")
    tuning = read(evidence / "tune/summary.json")
    heldout = read(evidence / "heldout/heldout.json")["rows"]
    expected = {(arm, b, length) for arm in ("packed", "dense")
                for b, length in proto["attention"]["heldout_shapes"]}
    observed = [(r["arm"], r["batch"], r["seq_len"]) for r in heldout]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("missing or duplicate held-out shape")
    consumers, amortized = [], []
    for arm in ("packed8", "dense"):
        pairs = []
        for rep in (1, 2):
            a, b = [read(evidence / f"e2e/{arm}-{mode}-rep{rep}/e2e.json")
                    for mode in ("default", "tuned")]
            for record in (a, b):
                if any(record[k] != v for k, v in
                       (("requests", 24), ("max_new", 64), ("max_batch", 8))):
                    raise ValueError("consumer workload differs from frozen protocol")
                if record["statuses"] != ["length"]:
                    raise ValueError("consumer contains incomplete/refused requests")
                if not math.isfinite(record["wall_s"]) or record["wall_s"] <= 0:
                    raise ValueError("invalid consumer duration")
            backend = "triton_packed" if arm == "packed8" else "triton_dense"
            forwards = a["backend_counts"][backend] / layers
            if forwards <= 0 or a["backend_counts"][backend] != b["backend_counts"][backend]:
                raise ValueError("consumer arms did different decode work")
            pairs.append((a, b, forwards))
            consumers.append({"arm": arm, "rep": rep, "default": a, "tuned": b,
                              "output_hash_equal": a["output_tokens_sha"] == b["output_tokens_sha"],
                              "raw_improvement": 1 - b["wall_s"] / a["wall_s"]})
        default = statistics.mean(a["wall_s"] for a, _, _ in pairs)
        tuned = statistics.mean(b["wall_s"] for _, b, _ in pairs)
        forwards = statistics.mean(n for _, _, n in pairs)
        charged = tuned + tune_seconds * forwards / 10_000
        amortized.append({"arm": arm, "default_mean_s": default, "tuned_mean_s": tuned,
                          "decode_forwards_per_workload": forwards,
                          "reuse_decode_forwards": 10_000, "tuning_seconds": tune_seconds,
                          "amortized_workload_s": charged,
                          "amortized_improvement": 1 - charged / default,
                          "meets_10_percent_timing_target": charged <= .9 * default,
                          "break_even_decode_forwards": tune_seconds * forwards / (default - tuned)
                          if default > tuned else None})
    latest = {s["stage"]: s for s in stages}
    required = ["sanitizer_control", "pytest_kernels", "sanitizer_ppi", "pytest_all", "tune", "heldout"]
    required += [f"e2e-{arm}-{mode}-{rep}" for arm in ("packed8", "dense")
                 for mode in ("default", "tuned") for rep in (1, 2)]
    return {"scope": "Recorded A100 study; two consumer repetitions, no confidence interval over devices.",
            "cost_scope": "Entire measured tuning stage, including exhaustive oracle/comparison work; no isolated deployment-only setup cost measured.",
            "quality_scope": "Output-hash differences are retained; timing results alone do not establish token equality or model quality.",
            "win_criterion": proto["win_criterion"], "tuning": tuning, "heldout": heldout,
            "consumers": consumers, "amortization": amortized,
            "generated_consumer_observed": all(r["tuned"]["dispatch_counts"].get("dequant:generated", 0) > 0
                                               for r in consumers if r["arm"] == "packed8"),
            "nonzero_or_missing_stage_exit": [latest.get(s, {"stage": s, "exit": None, "missing": True})
                                              for s in required if latest.get(s, {}).get("exit") != 0],
            "input_sha256": inputs}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("evidence", "protocol", "model-contract", "out"):
        ap.add_argument(f"--{key}", type=Path, required=True)
    args = ap.parse_args(argv)
    report = build(args.evidence, args.protocol, args.model_contract)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = [report["scope"], "", report["cost_scope"], "", report["quality_scope"], "",
             "| Consumer | Default s | Tuned s | Tuned plus setup s | Amortized change |",
             "|---|---:|---:|---:|---:|"]
    for r in report["amortization"]:
        lines.append(f'| {r["arm"]} | {r["default_mean_s"]:.3f} | {r["tuned_mean_s"]:.3f} | '
                     f'{r["amortized_workload_s"]:.3f} | {r["amortized_improvement"]:+.1%} |')
    lines += ["", "Positive change means less runtime. The full JSON retains every repetition and output-hash comparison.", "",
              "| Held-out arm | Batch | Length | Decision | Default us | Selected us | Best us |",
              "|---|---:|---:|---|---:|---:|---:|"]
    for r in report["heldout"]:
        lines.append(f'| {r["arm"]} | {r["batch"]} | {r["seq_len"]} | {r["decision"]} | '
                     f'{r["default_us"]} | {r["selected_us"]} | {r["best_us"]} |')
    (args.out / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
