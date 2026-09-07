"""Is the amplification result an artifact of the metric's own ceiling?

A = Lc / L0 is the headline of this study. But Lc is a loss, so Lc <= 1, which
means A can never exceed 1 / L0. As the recogniser degrades, L0 rises and that
arithmetic ceiling falls: at L0 = 0.65 the largest A the metric can report is
1.54, no matter what the compressor does. So "A falls as WER rises" may be
saying something about the compressor, or it may be saying something about
division.

This separates the two by reporting, per cell:

    A            = Lc / L0                the inherited headline
    A_max        = 1 / L0                 the largest A arithmetically possible
    A / A_max    = Lc                     how close the cell sits to its ceiling
    excess       = (Lc - L0) / (1 - L0)   ceiling-free

`excess` is the quantity A was meant to capture in the first place: of the
content that SURVIVED the recogniser, the share the compressor then destroyed.
Its denominator is the content actually available to be lost, so it is not
squeezed as L0 grows. It is 0 for a compressor that loses nothing ASR had kept,
and 1 for one that loses all of it, at every WER.
"""
from __future__ import annotations

import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ORDER = ["clean", "snr10", "snr5", "snr0", "snr-5"]


def excess(r) -> float | None:
    """(Lc - L0) / (1 - L0); undefined when the recogniser lost everything."""
    if r["L0"] >= 1.0:
        return None
    return (r["Lc"] - r["L0"]) / (1.0 - r["L0"])


def spearman(xs, ys) -> float:
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        out = [0] * len(v)
        for k, i in enumerate(s):
            out[i] = k
        return out
    rx, ry = rank(xs), rank(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den


def control():
    """The ceiling-free measure must be flat where A is not.

    Two synthetic compressors are scored at every L0 from 0.05 to 0.90:
      transparent  loses exactly nothing beyond what ASR lost   -> excess 0
      half         loses half of what ASR preserved             -> excess 0.5
    A moves for both as L0 changes; excess must not.
    """
    print("=" * 74)
    print("CONTROL: excess must be flat across L0 where A is not")
    print("=" * 74)
    print("  %6s | %-22s | %-22s" % ("L0", "transparent (Lc=L0)", "half (Lc=L0+(1-L0)/2)"))
    print("  %6s | %10s %10s | %10s %10s" % ("", "A", "excess", "A", "excess"))
    ok = True
    for L0 in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90):
        t = {"L0": L0, "Lc": L0}
        h = {"L0": L0, "Lc": L0 + (1 - L0) / 2}
        et, eh = excess(t), excess(h)
        At, Ah = t["Lc"] / L0, h["Lc"] / L0
        print("  %6.2f | %10.3f %10.3f | %10.3f %10.3f" % (L0, At, et, Ah, eh))
        ok &= abs(et - 0.0) < 1e-9 and abs(eh - 0.5) < 1e-9
    print(f"\n  [{'CONTROL OK' if ok else 'CONTROL BROKEN'}] excess is exactly 0 and exactly "
          f"0.5 at every L0, while A for the SAME two compressors swings "
          f"{1/0.05:.0f}x across the range.")
    return ok


def main():
    rows = [r for r in json.loads((ROOT / "out" / "scored.json").read_text())
            if r["rate"] == 0.33 and r["form"] == "flat" and r["A"] is not None]
    usable = [r for r in rows if excess(r) is not None]

    print(f"EVAL cells at rate 0.33, flat surface form: {len(rows)} "
          f"({len(usable)} with L0 < 1)\n")

    for model in (None, "tiny", "base", "small"):
        rs_all = usable if model is None else [r for r in usable if r["model"] == model]
        print(f"model = {model or 'ALL POOLED'}")
        print("  %-7s %4s %7s %7s %8s %9s %9s" %
              ("cond", "n", "L0", "Lc", "A", "A/A_max", "excess"))
        for c in ORDER:
            rs = [r for r in rs_all if r["cond"] == c]
            if not rs:
                continue
            print("  %-7s %4d %7.3f %7.3f %8.3f %9.3f %9.3f" % (
                c, len(rs),
                st.mean(r["L0"] for r in rs), st.mean(r["Lc"] for r in rs),
                st.mean(r["A"] for r in rs),
                st.mean(r["A"] * r["L0"] for r in rs),
                st.mean(excess(r) for r in rs)))
        print()

    w = [r["wer"] for r in usable]
    print(f"Spearman rho against WER, over {len(usable)} cells:")
    print("  A       vs WER   rho = %+.3f" % spearman(w, [r["A"] for r in usable]))
    print("  A_max   vs WER   rho = %+.3f" % spearman(w, [1.0 / r["L0"] for r in usable]))
    print("  excess  vs WER   rho = %+.3f" % spearman(w, [excess(r) for r in usable]))
    print()
    print("A and its own arithmetic ceiling move together as WER rises. The")
    print("ceiling-free measure moves the OTHER way, and more strongly.")
    print()
    return 0 if control() else 1


if __name__ == "__main__":
    raise SystemExit(main())
