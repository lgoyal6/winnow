"""Quality cohort for the packed serving study: prepare on CPU, run on GPU.

    python quality_study.py prepare --out PRIVATE_DIR            # CPU, no GPU
    python quality_study.py workload --manifest M.json --split final --max-new 128 --out W.json
    python quality_study.py run --manifest M.json --split final --arms dense packed8 --out RUN_DIR
    python quality_study.py report --runs RUN_DIR --baseline dense --margin 0.03 --out REPORT_DIR

Cohort: four LongBench-E tasks with deterministic metrics (hotpotqa, 2wikimqa,
multifieldqa_en, passage_retrieval_en) x three publisher length buckets x
eight questions chosen by a fixed SHA-256 ordering = 96 questions, each
middle-truncated to `cap` tokens (8192 by default) under the Qwen2.5 tokenizer.
One question per (task, bucket) is the calibration split, used to tune and to
freeze SLOs; the other 84 are the final split, untouched until the protocol is
frozen. Publisher answers are used without new adjudication; training overlap
with the model is unknown. Raw data stays in the private directory.

`run` drives the real engine in-process (greedy, EOS honoured, official
max_gen per task) for each arm on the same questions, and keeps every
generated token, status and score. `report` gives per-arm mean score with every
offered question in the denominator (a failure scores 0), the paired
difference against the baseline with a question-level bootstrap, per-task and
per-bucket breakdowns, and whether the lower bound clears the frozen
non-inferiority margin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
from longbench import TASKS, score

MODEL = "Qwen/Qwen2.5-7B-Instruct"
SALT = "winnow-packed-attention-v1"


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare(out: Path, cap: int, per_bucket: int):
    from huggingface_hub import HfApi, hf_hub_download
    from model_guard import guarded_from_pretrained, pinned_revision
    from transformers import AutoTokenizer
    out.mkdir(parents=True, exist_ok=False)
    revision = HfApi().dataset_info("THUDM/LongBench").sha
    archive = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset", revision=revision)
    tok = guarded_from_pretrained(AutoTokenizer, MODEL)
    qs = []
    with zipfile.ZipFile(archive) as z:
        for task in sorted(TASKS):
            name = next(n for n in z.namelist() if n.endswith(f"/{task}_e.jsonl"))
            raw = z.read(name)
            (out / f"{task}_e.jsonl").write_bytes(raw)
            buckets = {"0-4k": [], "4-8k": [], "8k+": []}
            for i, line in enumerate(raw.decode().splitlines()):
                if line.strip():
                    row = json.loads(line)
                    b = "0-4k" if row["length"] < 4000 else "4-8k" if row["length"] < 8000 else "8k+"
                    buckets[b].append((i, row))
            for b, rows in buckets.items():
                rows.sort(key=lambda r: hashlib.sha256(f"{SALT}:{task}:{r[0]}".encode()).hexdigest())
                if len(rows) < per_bucket:
                    raise ValueError(f"{task}/{b}: only {len(rows)} rows")
                for rank, (i, row) in enumerate(rows[:per_bucket]):
                    prompt = TASKS[task][0].format(context=row["context"], input=row["input"])
                    ids = tok(prompt, add_special_tokens=False).input_ids
                    if len(ids) > cap:
                        ids = ids[:cap // 2] + ids[-(cap - cap // 2):]
                    qs.append({"id": f"{task}/{i}", "task": task, "bucket": b,
                               "split": "calibration" if rank == 0 else "final",
                               "original_tokens": len(tok(prompt, add_special_tokens=False).input_ids),
                               "token_ids": ids, "input_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                               "answers": row["answers"], "metric": TASKS[task][2],
                               "max_gen": TASKS[task][1]})
    manifest = {"model": MODEL, "model_revision": pinned_revision(MODEL), "dataset": "THUDM/LongBench",
                "dataset_revision": revision, "archive_sha256": sha_file(archive), "cap_tokens": cap,
                "salt": SALT, "per_bucket": per_bucket, "questions": qs,
                "counts": {s: sum(q["split"] == s for q in qs) for s in ("calibration", "final")},
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "manifest.json").write_text(json.dumps(manifest) + "\n")
    digest = sha_file(out / "manifest.json")
    (out / "manifest.sha256").write_text(digest + "\n")
    print(f"{len(qs)} questions ({manifest['counts']}); manifest sha256 {digest}")


def workload(manifest, split, max_new, out):
    m = json.loads(Path(manifest).read_text())
    reqs = [{"id": q["id"], "token_ids": q["token_ids"], "max_new_tokens": max_new,
             "cohort": f"{q['task']}/{q['bucket']}", "context_tokens": len(q["token_ids"])}
            for q in m["questions"] if q["split"] == split]
    random.Random(f"{SALT}:order").shuffle(reqs)
    Path(out).write_text(json.dumps({"manifest_sha256": sha_file(manifest), "split": split,
                                     "max_new_tokens": max_new, "requests": reqs}) + "\n")
    print(f"{len(reqs)} requests -> {out}")


def run(manifest, split, arms, out, max_batch, kv_budget_gib, page_size):
    import torch
    import grpc_server as gs
    import packed_backend as pb
    from engine import EngineConfig, Request
    from model_guard import guarded_from_pretrained
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = json.loads(Path(manifest).read_text())
    qs = [q for q in m["questions"] if q["split"] == split]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    pb.register()
    tok = guarded_from_pretrained(AutoTokenizer, m["model"])
    model = guarded_from_pretrained(AutoModelForCausalLM, m["model"], dtype=torch.bfloat16,
                                    attn_implementation="sdpa").to("cuda").eval()
    eos = {tok.eos_token_id, *(model.generation_config.eos_token_id or [])} - {None}
    with (out / "rows.jsonl").open("w") as fh:
        for arm in arms:
            eng = gs.build_engine(model, kv_mode=arm, kv_budget_bytes=int(kv_budget_gib * 2**30),
                                  page_size=page_size, decode_backend="auto",
                                  cfg=EngineConfig(max_batch=max_batch, max_queue=len(qs),
                                                   prefill_chunk=2048, max_prompt_tokens=m["cap_tokens"],
                                                   max_new_tokens_cap=128), eos_ids=eos)
            pb.BACKEND_COUNTS.clear()
            reqs = [eng.submit(Request(f"{arm}:{q['id']}", q["token_ids"], q["max_gen"])) for q in qs]
            t0 = time.perf_counter()
            eng.run_until_idle()
            wall = time.perf_counter() - t0
            for q, r in zip(qs, reqs):
                text = tok.decode(r.generated, skip_special_tokens=True)
                ok = r.status in ("eos", "length")
                fh.write(json.dumps({"arm": arm, "id": q["id"], "task": q["task"], "bucket": q["bucket"],
                                     "status": r.status, "detail": r.detail, "tokens": r.generated,
                                     "text": text,
                                     "score": score(q["metric"], text, q["answers"]) if ok else 0.0}) + "\n")
            (out / f"engine-{arm}.json").write_text(json.dumps(
                {"wall_s": wall, "backend_counts": dict(pb.BACKEND_COUNTS),
                 "pool_pages": eng.pool.num_pages, "pool_bytes": eng.pool.bytes()}, indent=2) + "\n")
            del eng
            torch.cuda.empty_cache()
    (out / "meta.json").write_text(json.dumps({"manifest_sha256": sha_file(manifest), "split": split,
                                               "arms": arms, "argv": sys.argv,
                                               "gpu": torch.cuda.get_device_name()}, indent=2) + "\n")


def report(runs, baseline, margin, draws, out):
    rows = [json.loads(x) for d in runs for x in (Path(d) / "rows.jsonl").read_text().splitlines()]
    by = {}
    for r in rows:
        by.setdefault(r["arm"], {})[r["id"]] = r
    base = by[baseline]
    res = {"baseline": baseline, "non_inferiority_margin": margin, "arms": {}}
    rng = random.Random(0)
    for arm, rs in by.items():
        ids = sorted(set(rs) & set(base))
        mean = sum(rs[i]["score"] for i in ids) / len(ids)
        entry = {"questions": len(ids), "mean_score": mean,
                 "failures": sum(rs[i]["status"] not in ("eos", "length") for i in ids),
                 "per_task": {}, "per_bucket": {}}
        for key in ("task", "bucket"):
            groups = {}
            for i in ids:
                groups.setdefault(rs[i][key], []).append(i)
            entry[f"per_{key}"] = {g: {"n": len(v), "mean": sum(rs[i]["score"] for i in v) / len(v),
                                       "diff_vs_baseline": sum(rs[i]["score"] - base[i]["score"] for i in v) / len(v)}
                                   for g, v in sorted(groups.items())}
        if arm != baseline:
            d = [rs[i]["score"] - base[i]["score"] for i in ids]
            boot = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(draws))
            lo, hi = boot[int(0.025 * draws)], boot[int(0.975 * draws) - 1]
            entry.update(paired_diff=sum(d) / len(d), ci95=[lo, hi],
                         identical_outputs=sum(rs[i]["tokens"] == base[i]["tokens"] for i in ids),
                         non_inferior=lo > -margin)
        res["arms"][arm] = entry
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "quality.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({a: {k: v for k, v in e.items() if not k.startswith("per_")} for a, e in res["arms"].items()}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", required=True)
    p.add_argument("--cap", type=int, default=8192)
    p.add_argument("--per-bucket", type=int, default=8)
    w = sub.add_parser("workload")
    w.add_argument("--manifest", required=True)
    w.add_argument("--split", default="final", choices=["calibration", "final"])
    w.add_argument("--max-new", type=int, default=128)
    w.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--manifest", required=True)
    r.add_argument("--split", default="final", choices=["calibration", "final"])
    r.add_argument("--arms", nargs="+", default=["dense", "packed8"])
    r.add_argument("--max-batch", type=int, default=16)
    r.add_argument("--kv-budget-gib", type=float, default=12.0)
    r.add_argument("--page-size", type=int, default=16)
    r.add_argument("--out", required=True)
    q = sub.add_parser("report")
    q.add_argument("--runs", nargs="+", required=True)
    q.add_argument("--baseline", default="dense")
    q.add_argument("--margin", type=float, required=True, help="frozen non-inferiority margin")
    q.add_argument("--draws", type=int, default=2000)
    q.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        prepare(Path(a.out), a.cap, a.per_bucket)
    elif a.cmd == "workload":
        workload(a.manifest, a.split, a.max_new, a.out)
    elif a.cmd == "run":
        run(a.manifest, a.split, a.arms, a.out, a.max_batch, a.kv_budget_gib, a.page_size)
    else:
        report(a.runs, a.baseline, a.margin, a.draws, a.out)


if __name__ == "__main__":
    main()
