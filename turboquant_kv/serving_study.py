"""Frozen, paired CUDA diagnostic through the actual Transformers adapter.

Prepare inputs on CPU before renting compute. Raw rows, prompts and GPU receipts
belong in an external output directory. This is an in-process serving consumer,
not an HTTP load test or a full LongBench leaderboard submission.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from longbench import TASKS, score
from model_guard import guarded_from_pretrained, pinned_revision
from serving_adapter import generate

MODEL = "Qwen/Qwen2.5-7B-Instruct"
ARMS = ("native", "packed8_auto")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(output):
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    revision = HfApi().dataset_info("THUDM/LongBench").sha
    archive = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset", revision=revision)
    tokenizer = guarded_from_pretrained(AutoTokenizer, MODEL)
    samples = []
    with zipfile.ZipFile(archive) as zipped:
        for task in ("hotpotqa", "passage_retrieval_en"):
            name = next(n for n in zipped.namelist() if n.endswith(f"/{task}_e.jsonl") or n == f"{task}_e.jsonl")
            source = zipped.read(name)
            (output / f"{task}_e.jsonl").write_bytes(source)
            buckets = {"0-4k": [], "4-8k": [], "8k+": []}
            for index, line in enumerate(source.decode().splitlines()):
                if not line.strip():
                    continue
                row = json.loads(line)
                bucket = "0-4k" if row["length"] < 4096 else "4-8k" if row["length"] < 8192 else "8k+"
                buckets[bucket].append((index, row))
            for bucket, rows in buckets.items():
                rows.sort(key=lambda pair: hashlib.sha256(f"winnow-serving-v1:{task}:{pair[0]}".encode()).hexdigest())
                if len(rows) < 2:
                    raise ValueError("insufficient source cases in a frozen stratum")
                for index, row in rows[:2]:
                    prompt = TASKS[task][0].format(context=row["context"], input=row["input"])
                    tokens = tokenizer(prompt, add_special_tokens=False).input_ids
                    for cap in (2048, 4096):
                        ids = tokens if len(tokens) <= cap else tokens[:cap // 2] + tokens[-cap // 2:]
                        samples.append({"id": f"{task}/{index}/{cap}", "question_id": f"{task}/{index}",
                                        "task": task, "publisher_bucket": bucket, "context_cap": cap,
                                        "original_tokens": len(tokens), "input_ids": ids,
                                        "input_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                                        "answers": row["answers"], "metric": TASKS[task][2]})
    manifest = {"model": MODEL, "model_revision": pinned_revision(MODEL),
                "dataset": "THUDM/LongBench", "dataset_revision": revision,
                "archive_sha256": sha(archive), "max_new_tokens": 32, "repeats": 2,
                "arms": list(ARMS), "ttft_slo_s": 2.0, "completion_slo_s": 10.0,
                "selection": "Two hash-selected questions in each of three publisher length buckets for each of two tasks; each at 2048 and 4096 token caps. Raw official task prompts, no chat wrapper.",
                "samples": samples}
    save(output / "manifest.json", manifest)
    print(json.dumps({"questions": len({r['question_id'] for r in samples}), "inputs": len(samples), "manifest_sha256": sha(output / "manifest.json")}))


def resident_bytes(cache, torch):
    """Count unique reachable CUDA tensor storage, including shared codebooks."""
    objects, storages = set(), {}
    def visit(obj):
        if id(obj) in objects:
            return
        objects.add(id(obj))
        if torch.is_tensor(obj):
            if obj.numel() and obj.device.type == "cuda":
                storage = obj.untyped_storage()
                storages[(str(obj.device), storage.data_ptr())] = storage.nbytes()
        elif isinstance(obj, dict):
            for value in obj.values():
                visit(value)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                visit(value)
        elif hasattr(obj, "__dict__"):
            for value in vars(obj).values():
                visit(value)
    visit(cache.layers)
    return sum(storages.values())


class TokenClock:
    def __init__(self):
        self.prompt_seen = False
        self.times = []
    def put(self, value):
        # Transformers sends the prompt first, then each decoded token on CPU.
        if not self.prompt_seen:
            self.prompt_seen = True
        else:
            self.times.append(time.perf_counter())
    def end(self):
        pass


def request(model, tokenizer, sample, arm, max_new, torch, **extra):
    ids = torch.tensor([sample["input_ids"]], device="cuda")
    clock = TokenClock()
    kwargs = dict(attention_mask=torch.ones_like(ids), use_cache=True, max_new_tokens=max_new,
                  do_sample=False, return_dict_in_generate=True, streamer=clock,
                  pad_token_id=tokenizer.eos_token_id, **extra)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        if arm == "native":
            result = model.generate(ids, **kwargs)
        else:
            result = generate(model, ids, bit_width=8, max_cache_len=ids.shape[1] + max_new + 8, **kwargs)
    torch.cuda.synchronize()
    finished = time.perf_counter()
    if not clock.times:
        raise RuntimeError("no decoded-token callback")
    generated = result.sequences[0, ids.shape[1]:].tolist()
    cache = result.past_key_values
    decisions = [getattr(layer, "last_dispatch_decision", None) for layer in cache.layers]
    row = {"status": "ok", "output_ids": generated, "text": tokenizer.decode(generated, skip_special_tokens=True).strip(),
           "new_tokens": len(generated), "ttft_s": clock.times[0] - start,
           "completion_s": finished - start,
           "inter_token_s": [b - a for a, b in zip(clock.times, clock.times[1:])],
           "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
           "peak_incremental_bytes": torch.cuda.max_memory_allocated() - base,
           "resident_cache_bytes": resident_bytes(cache, torch),
           "dispatch_backends": sorted({d.backend for d in decisions if d is not None})}
    del cache, result, ids
    return row


def controls(model, tokenizer, sample, torch):
    from transformers import StoppingCriteria, StoppingCriteriaList
    from cache import TQPackedCache

    class StopAfterTwo(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return input_ids.shape[1] >= len(sample["input_ids"]) + 2

    ids = torch.tensor([sample["input_ids"]], device="cuda")
    with torch.inference_mode():
        kwargs = dict(attention_mask=torch.ones_like(ids), max_new_tokens=4,
                      do_sample=False, pad_token_id=tokenizer.eos_token_id)
        expected = model.generate(ids, **kwargs)
        fallback = generate(model, ids, bit_width=8, max_cache_len=len(sample["input_ids"]) + 12,
                            packed_device="cpu", **kwargs)
    assert torch.equal(expected, fallback)
    del ids, expected, fallback
    stopped = request(model, tokenizer, sample, "packed8_auto", 8, torch,
                      stopping_criteria=StoppingCriteriaList([StopAfterTwo()]))
    assert stopped["new_tokens"] <= 2
    bounded = TQPackedCache(model.config, bit_width=8, max_cache_len=1, device="cuda")
    hd = getattr(model.config, "head_dim", None) or model.config.hidden_size // model.config.num_attention_heads
    keys = torch.zeros(1, model.config.num_key_value_heads, 2, hd, device="cuda", dtype=torch.bfloat16)
    try:
        bounded.update(keys, keys, 0)
    except RuntimeError as error:
        assert "overflow" in str(error)
    else:
        raise AssertionError("cache overflow accepted")
    del bounded, keys
    gc.collect(); torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    try:
        allocation = torch.empty(free + (1 << 30), dtype=torch.uint8, device="cuda")
    except torch.cuda.OutOfMemoryError:
        pass
    else:
        del allocation
        raise AssertionError("expected controlled over-capacity allocation to fail")
    gc.collect(); torch.cuda.empty_cache()
    resumed = request(model, tokenizer, sample, "packed8_auto", 4, torch)
    assert resumed["new_tokens"] > 0
    return {"native_fallback_exact_token_parity": True,
            "stopped_after_tokens": stopped["new_tokens"], "overflow_rejected": True,
            "owned_gpu_over_capacity_allocation_rejected": True, "fresh_request_after_oom": True,
            "device_total_bytes": total}


def run(manifest_path, output):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    manifest = json.loads(Path(manifest_path).read_text())
    if manifest["model_revision"] != pinned_revision(manifest["model"]) or manifest["arms"] != list(ARMS):
        raise ValueError("manifest does not match the frozen model/arms")
    for sample in manifest["samples"]:
        if hashlib.sha256(json.dumps(sample["input_ids"]).encode()).hexdigest() != sample["input_sha256"]:
            raise ValueError("input hash mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("this study requires an actual CUDA device")
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    save(output / "manifest.json", manifest)
    save(output / "environment.json", {"manifest_sha256": sha(manifest_path), "torch": torch.__version__,
         "transformers": transformers.__version__, "cuda": torch.version.cuda, "python": sys.version,
         "host": platform.platform(), "gpu": torch.cuda.get_device_name(),
         "capability": torch.cuda.get_device_capability(),
         "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
         "source_sha256": {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")}})
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = guarded_from_pretrained(AutoTokenizer, manifest["model"])
    model = guarded_from_pretrained(AutoModelForCausalLM, manifest["model"], dtype=torch.bfloat16,
                                   attn_implementation="sdpa", device_map="cuda").eval()
    warm = {"input_ids": manifest["samples"][0]["input_ids"][:128]}
    for arm in ARMS:
        request(model, tokenizer, warm, arm, 4, torch)
    save(output / "controls.json", controls(model, tokenizer, warm, torch))
    # Separate diagnostic trace; profiled requests never enter the timing sample.
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, profile_memory=True) as profile:
            request(model, tokenizer, warm, "packed8_auto", 4, torch)
        profile.export_chrome_trace(str(output / "packed-profile.json"))
        (output / "packed-profile.txt").write_text(profile.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
    except Exception as error:
        save(output / "profile-error.json", {"type": type(error).__name__, "message": str(error)[:500]})
    rows = []
    with (output / "rows.jsonl").open("x") as raw:
        for repeat in range(manifest["repeats"]):
            for index, sample in enumerate(manifest["samples"]):
                arms = ARMS if (repeat + index) % 2 == 0 else ARMS[::-1]
                for arm in arms:
                    try:
                        row = request(model, tokenizer, sample, arm, manifest["max_new_tokens"], torch)
                    except (RuntimeError, ValueError) as error:
                        gc.collect(); torch.cuda.empty_cache()
                        row = {"status": "error", "error_type": type(error).__name__, "error": str(error)[:500], "text": ""}
                    row.update(id=sample["id"], question_id=sample["question_id"], task=sample["task"],
                               context_cap=sample["context_cap"], input_tokens=len(sample["input_ids"]),
                               input_sha256=sample["input_sha256"], arm=arm, repeat=repeat)
                    row["score"] = score(sample["metric"], row["text"], sample["answers"]) if row["status"] == "ok" else 0.0
                    row["slo_qualified"] = row["status"] == "ok" and row["ttft_s"] <= manifest["ttft_slo_s"] and row["completion_s"] <= manifest["completion_slo_s"]
                    raw.write(json.dumps(row, allow_nan=False) + "\n"); raw.flush(); rows.append(row)
                    print(f"{len(rows)} {sample['id']} {arm}: {row['status']}", flush=True)
    save(output / "summary.json", summarize(rows))


def summarize(rows, arms=ARMS):
    if len(arms) != 2 or len(set(arms)) != 2:
        raise ValueError("exactly two distinct comparison arms are required")
    baseline, candidate = arms
    result = {}
    for arm in arms:
        offered = [r for r in rows if r["arm"] == arm]
        successful = [r for r in offered if r["status"] == "ok"]
        result[arm] = {"offered": len(offered), "completed": len(successful),
                       "slo_qualified": sum(r["slo_qualified"] for r in offered),
                       "mean_score_all_offered": statistics.mean(r["score"] for r in offered)}
        for key in ("ttft_s", "completion_s", "peak_allocated_bytes", "peak_incremental_bytes", "resident_cache_bytes"):
            values = sorted(r[key] for r in successful)
            result[arm][key] = {"p50": statistics.median(values), "p95": values[math.ceil(.95 * len(values)) - 1]} if values else None
        intervals = sorted(value for row in successful for value in row["inter_token_s"])
        result[arm]["pooled_inter_token_s"] = ({"p50": statistics.median(intervals),
            "p95": intervals[math.ceil(.95 * len(intervals)) - 1]} if intervals else None)
    paired = {}
    for row in rows:
        pair = paired.setdefault((row["id"], row["repeat"]), {})
        if row["arm"] in pair:
            raise ValueError("duplicate request result")
        pair[row["arm"]] = row
    if any(set(pair) != set(arms) for pair in paired.values()):
        raise ValueError("incomplete paired results")
    clusters = {}
    for pair in paired.values():
        for key in ("question_id", "input_sha256", "task", "context_cap"):
            if pair[baseline][key] != pair[candidate][key]:
                raise ValueError("paired results disagree on input")
        delta = pair[candidate]["score"] - pair[baseline]["score"]
        clusters.setdefault(pair[baseline]["question_id"], []).append(delta)
    means = [statistics.mean(group) for group in clusters.values()]
    rng = random.Random(20261101)
    draws = sorted(statistics.mean(rng.choices(means, k=len(means))) for _ in range(2000))
    delta_key = "packed_minus_native" if tuple(arms) == ARMS else "candidate_minus_baseline"
    result["paired_quality"] = {delta_key: statistics.mean(means), "questions": len(means),
                                "question_bootstrap_95_percentile": [draws[49], draws[1949]], "draws": 2000}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare"); prep.add_argument("--out", required=True)
    study = sub.add_parser("run"); study.add_argument("--manifest", required=True); study.add_argument("--out", required=True)
    args = parser.parse_args()
    prepare(args.out) if args.command == "prepare" else run(args.manifest, args.out)
