"""NEGATIVE CONTROL for the amplification metric A = Lc / L0.

The claim A makes is "A == 1 means the compressor is transparent to ASR error,
A > 1 means it amplifies it". A metric that cannot be made to move is not
measuring anything, so the metric is run against stand-in compressors whose
behaviour is known in advance, on the SAME real ASR hypotheses:

  identity   keeps every word. C(R)=R and C(H)=H, so Lc is L0 by construction
             and A must come out at exactly 1.000. If it does not, the metric is
             broken, not the compressor.
  random33   keeps a random 33% of words, drawn INDEPENDENTLY for reference and
             hypothesis. Even on a perfect transcript the two keep-sets disagree,
             so A must come out far above 1.
  prefix33   keeps the first 33% of words. This one came out BELOW 1 (it absorbs
             error), which was not the guess going in. The reason was then
             measured rather than assumed: over these passages whisper's WER
             rises from 0.335 in the first third to 0.529 in the last, so a
             compressor that only ever reads the opening is reading the
             healthiest part of the transcript. It is kept in the control set
             because a metric that can only move upward is only half tested.

Only the compressor is substituted. The audio, the recogniser, the references and
the scoring code are the real ones.
"""
from __future__ import annotations
import json, random, statistics as st, sys, zlib
from pathlib import Path

from metrics import amplification, build_rare_vocab, score_pair

ROOT = Path(__file__).resolve().parent
RATE = 0.33


def identity(words, rng):
    return words


def random33(words, rng):
    k = max(1, round(RATE * len(words)))
    return [w for w in words if rng.random() < RATE] or words[:k]


def prefix33(words, rng):
    return words[: max(1, round(RATE * len(words)))]


def main():
    corpus = json.loads((ROOT / "out" / "corpus.json").read_text())
    asr = json.loads((ROOT / "out" / "asr.json").read_text())
    refs = {p["passage_id"]: " ".join(p["ref_words"]) for p in corpus["splits"]["eval"]}
    rare, _ = build_rare_vocab(list(refs.values()))
    cells = [a for a in asr.values() if a["split"] == "eval" and a["model"] == "base"]
    print(f"{len(cells)} real eval cells (base.en, all conditions)\n")

    results = {}
    for name, fn in (("identity", identity), ("prefix33", prefix33), ("random33", random33)):
        As, Lcs = [], []
        for a in cells:
            ref = refs[a["passage_id"]]
            L0 = 1 - score_pair(ref, a["hyp"], rare)["word_recall"]
            # crc32, not hash(): Python randomises str hashing per process, which
            # would make this control give a different number on every run.
            rng_r = random.Random(zlib.crc32(f"ref|{a['passage_id']}".encode()))
            rng_h = random.Random(zlib.crc32(f"hyp|{a['passage_id']}|{a['cond']}".encode()))
            cr = " ".join(fn(ref.split(), rng_r))
            ch = " ".join(fn(a["hyp"].split(), rng_h))
            Lc = 1 - score_pair(cr, ch, rare)["word_recall"]
            A = amplification(L0, Lc)
            Lcs.append(Lc)
            if A is not None:
                As.append(A)
        results[name] = (st.mean(As), st.stdev(As), len(As), st.mean(Lcs))
        print(f"  {name:9s} mean A = {st.mean(As):6.3f} +- {st.stdev(As):5.3f}  "
              f"(n={len(As)})   mean Lc = {st.mean(Lcs):.3f}")

    ok = True
    a_id = results["identity"][0]
    a_rnd = results["random33"][0]
    print()
    c1 = abs(a_id - 1.0) < 1e-9
    print(f"  [{'CONTROL OK' if c1 else 'CONTROL BROKEN'}] identity compressor gives "
          f"A = {a_id:.6f}, must be exactly 1")
    ok &= c1
    c2 = a_rnd > 1.5
    print(f"  [{'CONTROL OK' if c2 else 'CONTROL BROKEN'}] independent-random compressor "
          f"gives A = {a_rnd:.3f}, must be well above 1")
    ok &= c2
    a_pre = results["prefix33"][0]
    c3 = a_pre < 1.0
    print(f"  [{'CONTROL OK' if c3 else 'CONTROL BROKEN'}] prefix compressor "
          f"A = {a_pre:.3f}, below 1: the metric can register ABSORPTION too")
    ok &= c3
    c4 = a_pre < a_id < a_rnd
    print(f"  [{'CONTROL OK' if c4 else 'CONTROL BROKEN'}] the three stand-ins span the "
          f"scale: {a_pre:.3f} < {a_id:.3f} < {a_rnd:.3f}")
    ok &= c4
    print(f"\nNEGATIVE CONTROL {'VALID' if ok else 'BROKEN'}: A is exactly 1 for a no-op "
          f"compressor and moves in BOTH directions on real ASR output.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
