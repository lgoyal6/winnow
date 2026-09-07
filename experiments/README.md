# experiments/ — archived eval & research (not part of the product)

This folder holds the investigation into **semantic sentence-dedup (MMR)** on top
of the core `winnow` compressor. It is kept for reference and is **not** required
to deploy or run the service (see the repo root).

Two self-contained evidence bundles are also retained here:

- [`asr/`](asr/) recomputes the 630-row ASR error-amplification analysis from
  compact transcripts and compressor outputs, with no audio or model download.
- [`c27/`](c27/) contains the native CUDA versus Triton harness, matched-shape
  controls, full-decode results, compact static artifacts, and raw Nsight
  Systems traces from the RTX A6000 run.

## What we tested
`(LLMLingua-2 + BGE reranker)` with vs. without an MMR sentence-dedup pre-pass,
across two regimes: a sparse spoken **ramble** and a medium-density **multi-doc**
set, scored deterministically by having Claude answer comprehension questions
(SQuAD exact-match + token-F1, temperature 0).

## Conclusion
- **MMR sentence-dedup added nothing** at sane thresholds on realistic text
  (identical tokens and F1 with/without). It only fired — and *over-pruned* — on
  an adversarially redundant synthetic doc. Not shipped in the core pipeline.
- At 50–75% retention, LLMLingua-2 + reranker preserves facts well; the ramble
  scored a perfect 7/7.
- In multi-doc, the real bottleneck was the **reranker** dropping a relevant
  passage while keeping a distractor — not compression and not dedup.

## LCLM + TurboQuant: timing & memory benchmark

A second, independent experiment lives alongside the eval above. It measures the
**latency and memory** trade-off of running an **LCLM** (Latent Context Language
Model — encoder→adapter→decoder soft-token context compressor, arXiv:2606.09659)
decoder with **vanilla fp16 KV cache** vs. with **TurboQuant** KV-cache
compression (`TQCache`, our training-free random-rotation + Lloyd-Max quantizer).

The two arms run in **two separate warm Modal containers** (`LCLMVanilla` and
`LCLMTurboQuant`, both A100-80GB) so peak-memory and timing are isolated. We
sweep realistically long input contexts (filler docs with a planted "needle"
fact) × TQ bit-widths × decode lengths, and report TTFT, decode tok/s, peak GPU
MB, KV bytes, and KV compression ×.

**Honest finding:** TurboQuant here is pure-PyTorch dequant (no custom CUDA
kernel), so per-token decode is *slower* in wall-clock than fp16, while KV
**memory** is much smaller. The win is memory, not speed (matches the paper,
which needs custom kernels for speed). Both arms retrieve the needle correctly.
See `lclm_tq_timing_report.md` for the measured numbers.

### Re-running (from this folder)
```bash
cd experiments
modal deploy lclm_tq_timing.py        # warm LCLMVanilla + LCLMTurboQuant workers
python run_lclm_timing.py             # sweep -> lclm_tq_timing_results.json
```
The weights (`latent-context/0.6b-4b-LCLM-16x`) are read from the
`turboquant-hf-cache` Modal volume (already populated; download-if-missing).

### LCLM-timing files
- `lclm_tq_timing.py` — Modal app: `LCLMVanilla` + `LCLMTurboQuant` warm workers, inlined `TQCache`, filler-context generator.
- `run_lclm_timing.py` — driver: sweeps context × TQ config × decode length → `lclm_tq_timing_results.json`.
- `lclm_tq_timing_report.md` — the measured timing/memory comparison report.

## Files (MMR eval)
- `eval_modal.py` — Modal app exposing `compress_eval` (encoder-only) + the MMR pre-pass.
- `eval_sets.py` — the two task fixtures (ramble + multi-doc) with gold QA.
- `run_eval.py` — driver: runs both tasks × {no-dedup, +dedup} × {0.75, 0.5} retention.
- `score_eval.py` — Claude reader + deterministic scoring → `eval_report.md`.
- `run_arms.py` — older single-doc driver (superseded by `run_eval.py`).
- `test_data.py` — the original synthetic stress-test doc.
- `eval_report.md` — the scored comparison report.
- `scratchpad.ipynb` — exploratory notebook.

## Re-running (from this folder)
```bash
cd experiments
../../.venv/bin/modal deploy eval_modal.py     # uses two_stage_compressor.py from repo root
../../.venv/bin/python run_eval.py             # writes eval_results.json
../../.venv/bin/python score_eval.py           # reads ../.env, writes eval_report.md
```
