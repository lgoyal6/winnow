"""Export only retained benchmark measurements. Never imports torch or runs inference."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "turboquant_kv/results/phaseA_memory.json",
    "turboquant_kv/results/phaseB_longbench.json",
    "turboquant_kv/results/bitwidth_sweep.json",
    "experiments/c27/out/decode_0.6b.json",
    "experiments/c27/out/matched_N1048576.json",
    "turboquant_kv/bench_memory.py", "turboquant_kv/README.md",
    "experiments/c27/README.md",
]


def build():
    def load(path):
        return json.loads((ROOT / path).read_text())
    memory, quality, sweep, decode, kernel = [load(path) for path in FILES[:5]]
    groups = defaultdict(list)
    for row in quality["rows"]:
        for task in ("all", row["task"]):
            for bucket in ("all", row["bucket"]):
                groups[(row["arm"], task, bucket)].append(row["score"])
    scores = [
        {"arm": arm, "task": task, "bucket": bucket, "n": len(values),
         "score": sum(values) / len(values) * 100}
        for (arm, task, bucket), values in sorted(groups.items())
    ]
    keys = ["arm", "ctx", "decode", "batch", "decode_tok_s", "resident_kv_mb",
            "persistent_kv_mb", "transient_kv_mb", "peak_mb", "logical_kv_mb"]
    source_commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", *FILES], cwd=ROOT, text=True).strip()
    return {
        "schema_version": 1, "evidence_class": "recorded_gpu_benchmarks",
        "source_commit": source_commit,
        "sources": [{"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}
                    for path in FILES],
        "memory_model": memory["model"],
        "memory": [{key: row[key] for key in keys} for row in memory["rows"]],
        "quality_model": quality["model"], "quality": scores,
        "quality_tasks": sorted({row["task"] for row in quality["rows"]}),
        "bitwidth": [{key: row[key] for key in ["bits", "bytes_per_vector",
                     "compression_vs_fp16", "rel_l2_real", "greedy_token_match_vs_fp16"]}
                     for row in sweep["rows"]],
        "decode": decode, "kernel": kernel,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = Path(__file__).parent / "site/data.json"
    result = json.dumps(build(), indent=2, allow_nan=False) + "\n"
    if args.check:
        if not path.is_file() or path.read_text() != result:
            raise SystemExit("Recorded demo data is stale; run python3 public-demo/export.py")
        print("All retained source hashes and exported measurements match.")
    else:
        path.parent.mkdir(exist_ok=True)
        path.write_text(result)
        print(path)
