"""ABLATION: batch compression vs Winnow's real per-utterance streaming.

Winnow's README describes the production path as: the browser streams audio, and
"each `speech_final` utterance fires a request to /api/compress". So in the live
product the compressor never sees the transcript. It sees ONE UTTERANCE at a
time, and the transcript the LLM finally receives is the concatenation of many
independent small compressions.

Two arms, both self-consistent (reference and hypothesis compressed the same way,
so neither arm is scored against the other's granularity):

  batch   compress(whole passage)
  stream  concat(compress(each utterance))   <- what Winnow actually ships

Reference utterance boundaries are LibriSpeech's own segmentation; hypothesis
utterance boundaries are whisper's segments, which is what a live recogniser
would emit. Running the CLEAN condition through both arms also answers a
question that has nothing to do with ASR: does per-utterance compression change
the output even when the transcript is perfect?
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import compress_lib as C

ROOT = Path(__file__).resolve().parent
LS = ROOT / "data" / "LibriSpeech" / "test-clean"
RATE = 0.33
OUT = ROOT / "out" / "granularity.json"


def ref_utterances(p):
    chdir = LS / p["speaker"] / p["chapter"]
    trans = {l.split(" ", 1)[0]: l.split(" ", 1)[1].strip()
             for l in (chdir / f"{p['speaker']}-{p['chapter']}.trans.txt").read_text().splitlines()
             if l.strip()}
    return [trans[u] for u in p["utt_ids"]]


def main():
    corpus = json.loads((ROOT / "out" / "corpus.json").read_text())
    segs = json.loads((ROOT / "out" / "asr_segments.json").read_text())
    rows = json.loads(OUT.read_text()) if OUT.exists() else {}
    comp, dev = C.load_compressor()
    print(f"granularity ablation on {dev}, rate={RATE}")

    evals = {p["passage_id"]: p for p in corpus["splits"]["eval"]}
    t0 = time.perf_counter()
    for n, (pid, p) in enumerate(sorted(evals.items()), 1):
        # ---- reference, both granularities ----
        if f"ref|{pid}|batch" not in rows:
            whole = " ".join(p["ref_words"])
            rows[f"ref|{pid}|batch"] = C.compress(comp, whole, rate=RATE)[0]
            utts = ref_utterances(p)
            rows[f"ref|{pid}|stream"] = " ".join(
                C.compress(comp, u, rate=RATE)[0] for u in utts)
            rows[f"ref|{pid}|n_utts"] = len(utts)
        # ---- hypotheses, both granularities ----
        for key in [k for k in segs if k.startswith(pid + "|")]:
            if f"hyp|{key}|stream" in rows:
                continue
            sg = segs[key]["segments"]
            flat_segs = [C.flatten(s["text"]) for s in sg]
            rows[f"hyp|{key}|batch"] = C.compress(
                comp, " ".join(flat_segs), rate=RATE)[0]
            rows[f"hyp|{key}|stream"] = " ".join(
                C.compress(comp, s, rate=RATE)[0] for s in flat_segs if s.strip())
            rows[f"hyp|{key}|n_segs"] = len(sg)
        OUT.write_text(json.dumps(rows, indent=1))
        print(f"  {n}/{len(evals)} {pid}  ({(time.perf_counter()-t0)/60:.1f} min)")
    print(f"wrote {OUT}: {len(rows)} entries")


if __name__ == "__main__":
    sys.exit(main())
