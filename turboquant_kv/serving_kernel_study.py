"""Exploratory, explicit fused-FP32 comparison; never changes automatic dispatch."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import serving_study as study

ARMS = ("packed8_auto", "packed8_triton_fp32")


def fused_generate(model, inputs, *, bit_width, max_cache_len, **kwargs):
    from cache import TQPackedCache
    cache = TQPackedCache(model.config, bit_width=bit_width, max_cache_len=max_cache_len,
                          device=inputs.device, kernel_backend="triton_fp32", allow_tf32=False)
    return model.generate(inputs, past_key_values=cache, **kwargs)


def kernel_controls(torch):
    from cache import TQPackedLayer
    results = []
    # Include actual prefill lengths, decode tails, and noncontiguous cache views.
    for seed in (17, 29):
        torch.manual_seed(seed)
        for length in (128, 2048, 2079, 4096, 4127):
            keys = torch.randn(1, 4, length, 128, device="cuda", dtype=torch.bfloat16) * .1
            values = torch.randn_like(keys)
            options = dict(head_dim=128, bit_width=8, device="cuda", max_cache_len=length + 40,
                           allow_tf32=False)
            reference = TQPackedLayer(**options, kernel_backend="torch_fp32")
            fused = TQPackedLayer(**options, kernel_backend="triton_fp32")
            expected = reference.update(keys, values)
            actual = fused.update(keys, values)
            error = max((a.float() - e.float()).abs().max().item() for a, e in zip(actual, expected))
            assert all(torch.isfinite(a).all().item() for a in actual)
            assert error <= .03125, f"fused kernel exceeds retained numerical budget: {error}"
            results.append(dict(seed=seed, length=length, max_abs_error=error, budget=.03125))
            del keys, values, reference, fused, expected, actual
    gc.collect(); torch.cuda.empty_cache()
    return results


def run(manifest_path, output):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model_guard import guarded_from_pretrained, pinned_revision

    manifest = json.loads(Path(manifest_path).read_text())
    assert manifest["model_revision"] == pinned_revision(manifest["model"])
    for sample in manifest["samples"]:
        assert hashlib.sha256(json.dumps(sample["input_ids"]).encode()).hexdigest() == sample["input_sha256"]
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    study.save(output / "protocol.json", dict(manifest_sha256=study.sha(manifest_path),
        arms=ARMS, numerical_budget=.03125, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        transformers=transformers.__version__, cuda=torch.version.cuda,
        source_sha256={p.name: study.sha(p) for p in Path(__file__).parent.glob("*.py")},
        scope="Exploratory follow-up selected after inspecting baseline. Same questions, no independent holdout. Automatic dispatcher unchanged."))
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    study.save(output / "kernel-controls.json", kernel_controls(torch))
    torch.manual_seed(0)
    tokenizer = guarded_from_pretrained(AutoTokenizer, manifest["model"])
    model = guarded_from_pretrained(AutoModelForCausalLM, manifest["model"], dtype=torch.bfloat16,
                                   attn_implementation="sdpa", device_map="cuda").eval()
    original = study.generate

    def request(sample, arm):
        # Replace only the explicit experimental consumer, never the dispatcher.
        with patch.object(study, "generate", fused_generate if arm == ARMS[1] else original):
            return study.request(model, tokenizer, sample, "packed8_auto", manifest["max_new_tokens"], torch)

    # Compile and warm both actual context shapes before measuring either arm.
    for cap in (2048, 4096):
        sample = next(s for s in manifest["samples"] if s["context_cap"] == cap)
        for arm in ARMS: request(sample, arm)
    rows = []
    with (output / "rows.jsonl").open("x") as raw:
        for repeat in range(manifest["repeats"]):
            for index, sample in enumerate(manifest["samples"]):
                for arm in (ARMS if (repeat + index) % 2 == 0 else ARMS[::-1]):
                    try:
                        row = request(sample, arm)
                    except (RuntimeError, ValueError) as error:
                        gc.collect(); torch.cuda.empty_cache()
                        row = dict(status="error", error_type=type(error).__name__, error=str(error)[:500], text="")
                    row.update(id=sample["id"], question_id=sample["question_id"], task=sample["task"],
                               context_cap=sample["context_cap"], input_tokens=len(sample["input_ids"]),
                               input_sha256=sample["input_sha256"], arm=arm, repeat=repeat,
                               backend_override="triton_fp32" if arm == ARMS[1] else None)
                    row["score"] = study.score(sample["metric"], row["text"], sample["answers"]) if row["status"] == "ok" else 0.
                    row["slo_qualified"] = row["status"] == "ok" and row["ttft_s"] <= manifest["ttft_slo_s"] and row["completion_s"] <= manifest["completion_slo_s"]
                    raw.write(json.dumps(row, allow_nan=False) + "\n"); raw.flush(); rows.append(row)
                    print(len(rows), sample["id"], arm, row["status"], flush=True)
    study.save(output / "summary.json", study.summarize(rows, arms=ARMS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args(); run(args.manifest, args.out)
