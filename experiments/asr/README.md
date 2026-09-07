# ASR error amplification experiment

This experiment asks whether Winnow's LLMLingua-2 compression amplifies content
loss already introduced by ASR, and how that effect changes as word error rate
rises. It uses 14 held-out speakers from LibriSpeech `test-clean`, three local
whisper.cpp models, clean audio plus babble at four SNRs, and three compression
rates. The 630-cell result includes speaker-disjoint checks, latency in two quiet
windows, repeatability, short and long input checks, and ablations for surface
form, compression rate, streaming granularity, error type, and the original
amplification metric's ceiling.

The result is negative for the inherited headline. The ratio `Lc / L0` appears
largest when baseline ASR loss `L0` is small because the ratio has a shrinking
denominator. The bounded excess-loss metric moves in the opposite direction.
See `out/ANALYSIS.txt`, `out/ceiling.txt`, and `out/surface_form.txt`.

## Clean-clone verification

The compact committed inputs are recognizer transcripts and compressor outputs,
not audio or weights. They are sufficient to recompute every metric table:

```bash
cd experiments/asr
python3 -m venv .venv
.venv/bin/pip install -r requirements-analysis.txt
.venv/bin/python test_metrics.py
.venv/bin/python reproduce_results.py
```

`reproduce_results.py` recomputes all 630 scored rows and the complete analysis,
then compares both byte for byte with the committed result. It does not contact
an API or download a model.

## Full experiment

The large artifacts are intentionally excluded. Put the following under this
directory before running the full pipeline:

- LibriSpeech `test-clean` under `data/LibriSpeech/test-clean`
- `ggml-tiny.en.bin`, `ggml-base.en.bin`, and `ggml-small.en.bin` under `models/`
- `ffmpeg`, `ffprobe`, and `whisper-cli` on `PATH`. Override them with `FFMPEG`,
  `FFPROBE`, and `WHISPER_CLI` if needed.
- An environment with `llmlingua==0.2.2`, the root package's `hf` dependencies,
  and `jiwer==4.0.0`

Expected source artifact digests:

```text
39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23  test-clean.tar.gz
921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f  ggml-tiny.en.bin
a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002  ggml-base.en.bin
c6138d6d58ecc8322097e0f987c32f1be8bb0a18532a3f88f734d1bbf9c41e5d  ggml-small.en.bin
```

Run in order:

```bash
.venv/bin/python prep_corpus.py
.venv/bin/python make_audio.py
.venv/bin/python verify_corpus.py
.venv/bin/python negctl_corpus.py
.venv/bin/python run_asr.py
.venv/bin/python run_asr_segments.py
.venv/bin/python run_compress.py
.venv/bin/python run_granularity.py
.venv/bin/python test_stability.py
.venv/bin/python measure_latency.py w1
# Repeat measure_latency.py as w2 in a separate quiet window.
.venv/bin/python reproduce_results.py
```

The latency harness refuses to start above a one-minute load average of 8.0 and
records load at both ends of each window. The reported M3 Pro and MPS timings do
not transfer to the deployed GPU worker.

LibriSpeech `test-clean` comes from OpenSLR SLR12 and is distributed under
CC BY 4.0. whisper.cpp and its model conversion are separate upstream tools.
Neither source audio nor model weights are redistributed here.

## Scope limits

LibriSpeech is read audiobook speech, so this experiment does not establish
behavior on conversational disfluencies, restarts, or filled pauses. It tests
ASR noise and utterance chunking. Deepgram integration remains an API integration
question and is not counted as model research.
