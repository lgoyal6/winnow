"""Transcribe the whole (passage x condition x model) grid with local whisper.cpp.

No hosted ASR is contacted. whisper-cli is the Homebrew whisper.cpp build using
the Metal backend on this M3 Pro; the ggml models are local files.

Decoding is fixed (greedy-with-beam, fixed threads, English forced) so the only
things that vary across the grid are the audio condition and the model size.
Wall times are recorded here for bookkeeping, but this bulk pass runs under
whatever machine load exists; the reported latency numbers come from the
dedicated quiet-window runs in measure_latency.py, never from this file.
"""
from __future__ import annotations
import json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WHISPER = os.environ.get("WHISPER_CLI") or shutil.which("whisper-cli") or "whisper-cli"
MODELS = {"tiny": ROOT/"models/ggml-tiny.en.bin",
          "base": ROOT/"models/ggml-base.en.bin",
          "small": ROOT/"models/ggml-small.en.bin"}
THREADS = "4"

# Exact upstream byte sizes, read from the HF `x-linked-size` header for
# ggerganov/whisper.cpp. A half-downloaded ggml file does not announce itself: it
# fails with the SAME "failed to initialize whisper context" that GPU contention
# produces, so a retry loop written for contention will happily burn every
# attempt on a file that can never load. ggml-small.en.bin arrived truncated at
# 251,584,512 of 487,614,201 bytes and cost a whole pass before that was noticed.
MODEL_BYTES = {"tiny": 77704715, "base": 147964211, "small": 487614201}


def check_models():
    for name, path in MODELS.items():
        got = path.stat().st_size if path.exists() else -1
        want = MODEL_BYTES[name]
        if got != want:
            raise SystemExit(f"{path.name}: {got} bytes on disk, upstream is {want}. "
                             "Truncated or wrong file; re-download before running.")
        print(f"  model ok  {path.name:22s} {got} bytes")
OUT = ROOT / "out" / "asr.json"


def transcribe(model_path: Path, wav: Path, threads: str = THREADS, retries: int = 12):
    """One whisper-cli call, with retry on Metal context-init failure.

    Observed on this box: when another process is hammering the GPU, whisper-cli
    intermittently dies with "failed to initialize whisper context" plus a
    ggml-metal residency-set assert. It is contention, not a bad input - the same
    command succeeds unchanged moments later. The retry loop backs off, and the
    last attempt drops to CPU so a busy GPU can never silently truncate the grid.
    Which attempt succeeded, and on which backend, is recorded per row.
    """
    last = ""
    for attempt in range(retries):
        # Alternate Metal / CPU rather than saving CPU for last. Metal init is what
        # fails under contention; the CPU backend has never failed here, so the
        # cheapest recovery is to just take the slower backend on the next try.
        cpu = attempt % 2 == 1
        cmd = [WHISPER, "-m", str(model_path), "-f", str(wav), "-l", "en",
               "-nt", "-np", "-t", threads, "-bs", "5", "-bo", "5"]
        if cpu:
            cmd.append("-ng")
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.perf_counter() - t0
        if p.returncode == 0 and p.stdout.strip():
            return " ".join(p.stdout.split()), wall, attempt, ("cpu" if cpu else "metal")
        last = p.stderr[-800:]
        time.sleep(min(15.0, 2.0 * (attempt + 1)))
    raise RuntimeError(f"whisper failed on {wav} after {retries} attempts:\n{last}")


def main():
    check_models()
    man = json.loads((ROOT / "out" / "audio_manifest.json").read_text())
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    rows = dict(prev)
    todo = [(m, r) for m in MODELS for r in man]
    print(f"{len(todo)} transcriptions ({len(MODELS)} models x {len(man)} renders)")
    t_start = time.perf_counter()
    failed = []
    for i, (mname, r) in enumerate(todo, 1):
        key = f"{mname}|{r['passage_id']}|{r['cond']}"
        if key in rows:
            continue
        try:
            text, wall, attempt, backend = transcribe(MODELS[mname], Path(r["wav"]))
        except RuntimeError as e:
            # One item losing a GPU-contention race must not abandon the other
            # 400. Record it, keep going, and let a later pass pick it up.
            failed.append(key)
            print(f"  SKIP {key}: {str(e).splitlines()[0]}")
            continue
        rows[key] = {"model": mname, "passage_id": r["passage_id"],
                     "split": r["split"], "cond": r["cond"],
                     "snr_target": r["snr_target"], "dur_s": r["dur_s"],
                     "wall_s": round(wall, 3), "retries": attempt,
                     "backend": backend, "hyp": text}
        if i % 10 == 0 or i == len(todo):
            OUT.write_text(json.dumps(rows, indent=1))
            el = time.perf_counter() - t_start
            print(f"  {i}/{len(todo)}  {el/60:.1f} min elapsed  last={mname}/{r['cond']}"
                  f" rtf={wall/r['dur_s']:.3f}")
    OUT.write_text(json.dumps(rows, indent=1))
    print(f"wrote {OUT}: {len(rows)} transcripts; {len(failed)} failed this pass")
    return 0 if len(rows) == len(todo) else 2


if __name__ == "__main__":
    sys.exit(main())
