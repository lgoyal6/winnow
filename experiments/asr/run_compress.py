"""Compress every reference and every ASR hypothesis, at several rates.

Produces the raw material for the amplification analysis. No scoring here; this
file only records what the compressor produced, so scoring can be re-run and
re-argued without paying for the model again.

Surface forms:
  flat - uppercase, punctuation stripped. This is the COMMON form: the
         LibriSpeech reference has no case or punctuation, so comparing a
         punctuated hypothesis against it would confound "what the compressor
         did" with "what the surface form was". The main result uses flat only.
  raw  - whisper's own output, with its punctuation and casing. Used ONLY for the
         punctuation ablation, and always scored against the same flat reference,
         so the only thing that changes between the two arms is what the
         COMPRESSOR was shown.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import compress_lib as C

ROOT = Path(__file__).resolve().parent
RATES = [0.5, 0.33, 0.2]
RAW_ABLATION_RATE = 0.33
OUT = ROOT / "out" / "compressed.json"


def main():
    corpus = json.loads((ROOT / "out" / "corpus.json").read_text())
    asr = json.loads((ROOT / "out" / "asr.json").read_text())
    refs = {p["passage_id"]: " ".join(p["ref_words"])
            for s in ("tune", "eval") for p in corpus["splits"][s]}

    rows = json.loads(OUT.read_text()) if OUT.exists() else {}
    comp, dev = C.load_compressor()
    print(f"compressor on {dev}; {len(refs)} refs, {len(asr)} hypotheses")

    jobs = []
    for pid, text in refs.items():
        for r in RATES:
            jobs.append((f"ref|{pid}|flat|{r}", text, r))
    for key, row in asr.items():
        flat = C.flatten(row["hyp"])
        for r in RATES:
            jobs.append((f"hyp|{key}|flat|{r}", flat, r))
        jobs.append((f"hyp|{key}|raw|{RAW_ABLATION_RATE}", row["hyp"], RAW_ABLATION_RATE))

    todo = [j for j in jobs if j[0] not in rows]
    print(f"{len(jobs)} compressions, {len(todo)} to do")
    t0 = time.perf_counter()
    for i, (key, text, rate) in enumerate(todo, 1):
        out, _ = C.compress(comp, text, rate=rate)
        rows[key] = {"rate": rate, "n_in": len(text.split()),
                     "n_out": len(out.split()), "text": out}
        if i % 200 == 0 or i == len(todo):
            OUT.write_text(json.dumps(rows, indent=1))
            print(f"  {i}/{len(todo)}  {(time.perf_counter()-t0)/60:.1f} min")
    OUT.write_text(json.dumps(rows, indent=1))
    print(f"wrote {OUT}: {len(rows)} compressions")


if __name__ == "__main__":
    sys.exit(main())
