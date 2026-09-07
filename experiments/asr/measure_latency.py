"""Streaming latency, split into MODEL cost and PLUMBING cost.

Winnow is a real-time system, so the number that matters is not "how long does a
200-word transcript take" but "when an utterance ends, how long until its
compressed text is ready" - and whether that is faster than the audio arriving
behind it.

MODEL vs PLUMBING. These are separated because they have completely different
fixes and only one of them is a research result:

  MODEL     whisper encode+decode for the utterance; the LLMLingua-2 forward
            pass over the utterance's tokens. Irreducible without changing the
            model.
  PLUMBING  per-call whisper-cli process spawn and per-call ggml model load,
            plus the compressor's one-off weight load. A streaming integration
            holding a warm recogniser pays these ONCE, not per utterance. Any
            latency chart that folds these into the per-chunk number is
            measuring a subprocess, not a model.

whisper-cli reports its own load / encode / decode breakdown, so the split is
read from the recogniser's instrumentation rather than guessed.

MACHINE LOAD. Other agents share this box and a busy box already invalidated one
published figure in this repo. Every measurement here refuses to start above
LOAD_MAX, records the 1-minute load average before and after, and discards any
sample whose window went busy mid-run.
"""
from __future__ import annotations
import json, os, re, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WHISPER = os.environ.get("WHISPER_CLI") or shutil.which("whisper-cli") or "whisper-cli"
FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"
MODEL = ROOT / "models" / "ggml-base.en.bin"
LOAD_MAX = 8.0

_T = {k: re.compile(rf"{p}\s*=\s*([\d.]+)\s*ms")
      for k, p in (("load", "load time"), ("encode", "encode time"),
                   ("decode", "decode time"), ("total", "total time"))}


def load1() -> float:
    return os.getloadavg()[0]


def require_quiet(what: str):
    l = load1()
    if l > LOAD_MAX:
        raise SystemExit(f"REFUSING to measure {what}: 1-min load {l:.2f} > {LOAD_MAX}. "
                         "Wait for a quiet window.")
    return l


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    i = min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1))))
    return xs[i]


def whisper_once(wav: Path):
    """Returns (wall_s, whisper's own {load,encode,decode,total} in ms)."""
    t0 = time.perf_counter()
    p = subprocess.run([WHISPER, "-m", str(MODEL), "-f", str(wav), "-l", "en",
                        "-nt", "-t", "4", "-bs", "5", "-bo", "5"],
                       capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        return None, None
    parts = {}
    for k, rx in _T.items():
        m = rx.search(p.stderr)
        parts[k] = float(m.group(1)) if m else None
    return wall, parts


def measure_asr(utt_wavs, repeats=5):
    l_before = require_quiet("ASR latency")
    rows = []
    for wav, dur in utt_wavs:
        for _ in range(repeats):
            wall, parts = whisper_once(wav)
            if wall is None or parts["total"] is None:
                continue
            model_ms = (parts["encode"] or 0) + (parts["decode"] or 0)
            rows.append({"wav": wav.name, "dur_s": dur, "wall_ms": wall * 1000,
                         "whisper_load_ms": parts["load"],
                         "whisper_total_ms": parts["total"],
                         "model_ms": model_ms,
                         "spawn_ms": wall * 1000 - parts["total"]})
    return rows, l_before, load1()


def measure_compressor(texts, repeats=7, rate=0.33):
    import compress_lib as C
    l_before = require_quiet("compressor latency")
    t0 = time.perf_counter()
    comp, dev = C.load_compressor()
    load_s = time.perf_counter() - t0
    # Warm up: the first forward pass on MPS pays kernel compilation, which is a
    # one-off cost and would otherwise be charged to the first utterance.
    for _ in range(3):
        C.compress(comp, texts[0][1], rate=rate)
    rows = []
    for name, txt in texts:
        for _ in range(repeats):
            t0 = time.perf_counter()
            out, _r = C.compress(comp, txt, rate=rate)
            rows.append({"name": name, "n_words": len(txt.split()),
                         "ms": (time.perf_counter() - t0) * 1000,
                         "n_out": len(out.split())})
    return rows, load_s, dev, l_before, load1()


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "w1"
    segs = json.loads((ROOT / "out" / "asr_segments.json").read_text())
    man = {r["passage_id"]: r for r in
           json.loads((ROOT / "out" / "audio_manifest.json").read_text())
           if r["cond"] == "clean"}

    # Cut real utterances out of real audio: the streaming unit, not a fixed window.
    utt_dir = ROOT / "utt"; utt_dir.mkdir(exist_ok=True)
    utts = []
    for key in sorted(segs):
        if not key.endswith("|clean"):
            continue
        row = segs[key]
        src = Path(man[row["passage_id"]]["wav"])
        for i, s in enumerate(row["segments"][:3]):
            out = utt_dir / f"{row['passage_id']}_u{i}.wav"
            if not out.exists():
                subprocess.run([FFMPEG, "-y", "-hide_banner",
                                "-loglevel", "error", "-i", str(src), "-ss", f"{s['t0']:.3f}",
                                "-to", f"{s['t1']:.3f}", "-ac", "1", "-ar", "16000", str(out)],
                               check=True)
            utts.append((out, s["t1"] - s["t0"], s["text"]))
        if len(utts) >= 18:
            break

    print(f"[{tag}] {len(utts)} real utterances, "
          f"median {sorted(u[1] for u in utts)[len(utts)//2]:.1f}s of audio each")

    a_rows, a_l0, a_l1 = measure_asr([(w, d) for w, d, _ in utts])
    c_rows, c_load, dev, c_l0, c_l1 = measure_compressor(
        [(w.name, t) for w, _, t in utts])

    res = {"tag": tag, "device": dev,
           "load_asr": [a_l0, a_l1], "load_comp": [c_l0, c_l1],
           "compressor_load_s": c_load, "asr": a_rows, "comp": c_rows}
    p = ROOT / "out" / f"latency_{tag}.json"
    p.write_text(json.dumps(res, indent=1))

    print(f"\n[{tag}] ASR  load {a_l0:.2f} -> {a_l1:.2f}   (n={len(a_rows)})")
    for k, lbl in (("model_ms", "MODEL  encode+decode"),
                   ("whisper_load_ms", "PLUMB  ggml model load"),
                   ("spawn_ms", "PLUMB  process spawn/exit"),
                   ("wall_ms", "TOTAL  wall per utterance")):
        v = [r[k] for r in a_rows if r[k] is not None]
        print(f"    {lbl:28s} median {pct(v,50):7.1f} ms   p95 {pct(v,95):7.1f} ms")
    rtf = [r["model_ms"] / 1000 / r["dur_s"] for r in a_rows]
    print(f"    {'MODEL  real-time factor':28s} median {pct(rtf,50):7.3f}      p95 {pct(rtf,95):7.3f}")

    print(f"\n[{tag}] COMPRESSOR on {dev}  load {c_l0:.2f} -> {c_l1:.2f}   (n={len(c_rows)})")
    print(f"    {'PLUMB  one-off weight load':28s} {c_load*1000:9.1f} ms (paid once)")
    v = [r["ms"] for r in c_rows]
    print(f"    {'MODEL  forward per utterance':28s} median {pct(v,50):7.1f} ms   p95 {pct(v,95):7.1f} ms")

    e2e = pct([r["model_ms"] for r in a_rows], 50) + pct(v, 50)
    med_dur = sorted(u[1] for u in utts)[len(utts)//2] * 1000
    print(f"\n[{tag}] END-TO-END per utterance (model only, warm services)")
    print(f"    median {e2e:.1f} ms for a median {med_dur:.0f} ms utterance "
          f"-> {e2e/med_dur:.3f} x real time")
    print(f"    wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
