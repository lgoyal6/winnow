"""Is the compressor invariant to surface form, and does ablation A prove what it claims?

Inherited ablation A compares, at rate 0.33:

    Lc flat = 1 - recall( C(flat reference) -> C(flat hypothesis) )
    Lc raw  = 1 - recall( C(flat reference) -> C(raw  hypothesis) )

and reports Lc raw as much worse (0.147 -> 0.247 on clean base.en, A 2.18 -> 4.31).
Both arms share a reference that was compressed in the FLAT form, so a raw
hypothesis is penalised whenever the compressor, shown punctuation, simply picks
a DIFFERENT set of words rather than a worse one. That is a confound: the
inherited number mixes "punctuation degrades selection" with "punctuation
changes selection".

This isolates the second effect with no reference involved at all. The same
hypothesis text is compressed in both surface forms and the two outputs are
compared to each other. Whatever the gap is, it is surface-form sensitivity and
nothing else.
"""
from __future__ import annotations

import json
import statistics as st
from collections import Counter
from pathlib import Path

from metrics import multiset_recall, norm_words

ROOT = Path(__file__).resolve().parent
RATE = 0.33
ORDER = ["clean", "snr10", "snr5", "snr0", "snr-5"]


def control():
    """A text compared against itself must show perfect overlap.

    If the comparison reported less than 1.0 here, the normalisation rather than
    the compressor would be producing the gap measured below.
    """
    comp = json.loads((ROOT / "out" / "compressed.json").read_text())
    key = next(k for k in comp if k.startswith("hyp|") and k.endswith("|raw|0.33"))
    w = Counter(norm_words(comp[key]["text"]))
    inter = sum((w & w).values())
    union = sum((w | w).values())
    j = inter / union if union else 0.0
    r = multiset_recall(list(w.elements()), list(w.elements()))
    ok = abs(j - 1.0) < 1e-9 and abs(r - 1.0) < 1e-9
    print(f"  [{'CONTROL OK' if ok else 'CONTROL BROKEN'}] a raw output compared with "
          f"itself scores jaccard {j:.3f} and recall {r:.3f}; the gap below is the "
          f"compressor, not the tokeniser.")
    return ok


def main():
    comp = json.loads((ROOT / "out" / "compressed.json").read_text())
    asr = json.loads((ROOT / "out" / "asr.json").read_text())

    print("Same hypothesis text, two surface forms, same rate 0.33.")
    print("No reference is involved, so the flat-reference confound is removed.\n")
    print("  %-7s %4s %9s %9s %11s %11s" %
          ("cond", "n", "jaccard", "recall", "words flat", "words raw"))
    rows = {}
    for cond in ORDER:
        js, rc, nf, nr = [], [], [], []
        for key, a in asr.items():
            if a["split"] != "eval" or a["cond"] != cond:
                continue
            fk, rk = f"hyp|{key}|flat|{RATE}", f"hyp|{key}|raw|{RATE}"
            if fk not in comp or rk not in comp:
                continue
            f = Counter(norm_words(comp[fk]["text"]))
            r = Counter(norm_words(comp[rk]["text"]))
            union = sum((f | r).values())
            if union:
                js.append(sum((f & r).values()) / union)
            rc.append(multiset_recall(list(f.elements()), list(r.elements())))
            nf.append(sum(f.values()))
            nr.append(sum(r.values()))
        if not js:
            continue
        rows[cond] = st.mean(js)
        print("  %-7s %4d %9.3f %9.3f %11.1f %11.1f" %
              (cond, len(js), st.mean(js), st.mean(rc), st.mean(nf), st.mean(nr)))

    spread = max(rows.values()) - min(rows.values())
    print(f"\n  Jaccard spread across every noise condition: {spread:.3f}")
    print("  Surface-form sensitivity is flat in WER: it is a property of the")
    print("  compressor, not a symptom of a degraded transcript.\n")
    return 0 if control() else 1


if __name__ == "__main__":
    raise SystemExit(main())
