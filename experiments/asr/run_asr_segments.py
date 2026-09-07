"""Segment-level transcription for the STREAMING-GRANULARITY ablation.

Winnow does not compress a finished transcript. Per its own README, the browser
streams audio to the recogniser and "each `speech_final` utterance fires a
request to `/api/compress`" - so the real production unit of compression is ONE
UTTERANCE, a few seconds of speech ending at a pause, not a 200-word passage.

whisper.cpp's own segments are the local equivalent of `speech_final`: the
decoder closes a segment at a pause. `-oj` gives those segments with their start
and end times, which are also what the end-to-end streaming latency is measured
against later.

Scope: EVAL split only, one model (base.en), all five audio conditions. That is
enough to answer "does compressing per utterance behave differently from
compressing the whole passage" without re-running the entire 420-cell grid.
"""
from __future__ import annotations
import json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WHISPER = os.environ.get("WHISPER_CLI") or shutil.which("whisper-cli") or "whisper-cli"
MODEL = ROOT / "models" / "ggml-base.en.bin"
OUT = ROOT / "out" / "asr_segments.json"


def transcribe_segments(wav: Path, retries: int = 12):
    last = ""
    for attempt in range(retries):
        with tempfile.TemporaryDirectory() as td:
            stem = Path(td) / "o"
            cmd = [WHISPER, "-m", str(MODEL), "-f", str(wav), "-l", "en", "-np",
                   "-t", "4", "-bs", "5", "-bo", "5", "-oj", "-of", str(stem)]
            if attempt % 2 == 1:
                cmd.append("-ng")
            t0 = time.perf_counter()
            p = subprocess.run(cmd, capture_output=True, text=True)
            wall = time.perf_counter() - t0
            j = stem.with_suffix(".json")
            if p.returncode == 0 and j.exists():
                d = json.loads(j.read_text())
                segs = [{"t0": s["offsets"]["from"] / 1000.0,
                         "t1": s["offsets"]["to"] / 1000.0,
                         "text": " ".join(s["text"].split())}
                        for s in d.get("transcription", [])]
                return segs, wall
            last = p.stderr[-600:]
        time.sleep(min(15.0, 2.0 * (attempt + 1)))
    raise RuntimeError(f"whisper -oj failed on {wav}: {last}")


def main():
    man = json.loads((ROOT / "out" / "audio_manifest.json").read_text())
    todo = [r for r in man if r["split"] == "eval"]
    rows = json.loads(OUT.read_text()) if OUT.exists() else {}
    failed = []
    print(f"{len(todo)} segment transcriptions (eval x base.en x 5 conditions)")
    for i, r in enumerate(todo, 1):
        key = f"{r['passage_id']}|{r['cond']}"
        if key in rows:
            continue
        try:
            segs, wall = transcribe_segments(Path(r["wav"]))
        except RuntimeError as e:
            failed.append(key); print(f"  SKIP {key}"); continue
        rows[key] = {"passage_id": r["passage_id"], "cond": r["cond"],
                     "snr_target": r["snr_target"], "dur_s": r["dur_s"],
                     "wall_s": round(wall, 3), "n_segments": len(segs),
                     "segments": segs}
        if i % 10 == 0 or i == len(todo):
            OUT.write_text(json.dumps(rows, indent=1))
            print(f"  {i}/{len(todo)}  segs={len(segs)}")
    OUT.write_text(json.dumps(rows, indent=1))
    print(f"wrote {OUT}: {len(rows)} rows, {len(failed)} failed")
    return 0 if len(rows) == len(todo) else 2


if __name__ == "__main__":
    sys.exit(main())
