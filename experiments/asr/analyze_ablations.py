"""Ablations + failure-mode hunt. EVAL speakers only."""
from __future__ import annotations
import json, statistics as st, sys
from collections import defaultdict
from pathlib import Path

from metrics import amplification, build_rare_vocab, score_pair, wer

ROOT = Path(__file__).resolve().parent


def load():
    corpus = json.loads((ROOT / "out" / "corpus.json").read_text())
    asr = json.loads((ROOT / "out" / "asr.json").read_text())
    comp = json.loads((ROOT / "out" / "compressed.json").read_text())
    refs = {p["passage_id"]: " ".join(p["ref_words"])
            for s in ("tune", "eval") for p in corpus["splits"][s]}
    eval_pids = [p["passage_id"] for p in corpus["splits"]["eval"]]
    rare, _ = build_rare_vocab([refs[p] for p in eval_pids])
    return corpus, asr, comp, refs, rare, eval_pids


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "      --        "
    m = st.mean(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:8.3f}+-{s:<6.3f}"


def ablation_punctuation(asr, comp, refs, rare):
    print("\n" + "=" * 78)
    print("ABLATION A: does giving the compressor PUNCTUATION change what survives?")
    print("  Both arms are scored against the SAME flat reference compression, so the")
    print("  only difference is the surface form the COMPRESSOR was shown.")
    print("=" * 78)
    g = defaultdict(lambda: defaultdict(list))
    for key, a in asr.items():
        if a["split"] != "eval":
            continue
        pid = a["passage_id"]
        ck = f"ref|{pid}|flat|0.33"
        if ck not in comp:
            continue
        s0 = score_pair(refs[pid], a["hyp"], rare)
        L0 = 1 - s0["word_recall"]
        for form in ("flat", "raw"):
            hk = f"hyp|{key}|{form}|0.33"
            if hk not in comp:
                continue
            sc = score_pair(comp[ck]["text"], comp[hk]["text"], rare)
            g[(a["model"], a["cond"])][form].append(
                (1 - sc["word_recall"], amplification(L0, 1 - sc["word_recall"]),
                 comp[hk]["n_out"] / max(1, comp[hk]["n_in"])))
    print(f"  {'model':>6} {'cond':>7} | {'Lc flat':>16} {'Lc raw':>16} | "
          f"{'A flat':>16} {'A raw':>16}")
    for k in sorted(g):
        r = g[k]
        print(f"  {k[0]:>6} {k[1]:>7} | {ms([x[0] for x in r['flat']])} "
              f"{ms([x[0] for x in r['raw']])} | {ms([x[1] for x in r['flat']])} "
              f"{ms([x[1] for x in r['raw']])}")


def ablation_granularity(corpus, asr, refs, rare):
    p = ROOT / "out" / "granularity.json"
    if not p.exists():
        print("\n(granularity data not present yet)")
        return
    gr = json.loads(p.read_text())
    print("\n" + "=" * 78)
    print("ABLATION B: BATCH compression vs Winnow's real PER-UTTERANCE streaming")
    print("  Each arm compresses reference and hypothesis at the same granularity.")
    print("=" * 78)
    hyp = {f"{a['passage_id']}|{a['cond']}": a for a in asr.values()
           if a["model"] == "base" and a["split"] == "eval"}
    g = defaultdict(lambda: defaultdict(list))
    for key, a in hyp.items():
        pid = a["passage_id"]
        if f"hyp|{key}|batch" not in gr:
            continue
        s0 = score_pair(refs[pid], a["hyp"], rare)
        L0 = 1 - s0["word_recall"]
        for arm in ("batch", "stream"):
            sc = score_pair(gr[f"ref|{pid}|{arm}"], gr[f"hyp|{key}|{arm}"], rare)
            n_in = len(gr[f"hyp|{key}|{arm}"].split())
            g[a["cond"]][arm].append((1 - sc["word_recall"], amplification(L0, 1 - sc["word_recall"]),
                                      1 - sc["rare_recall"], n_in))
    print(f"  {'cond':>7} | {'Lc batch':>16} {'Lc stream':>16} | "
          f"{'A batch':>16} {'A stream':>16}")
    for k in sorted(g, key=lambda c: {"clean": 0, "snr10": 1, "snr5": 2, "snr0": 3, "snr-5": 4}[c]):
        r = g[k]
        print(f"  {k:>7} | {ms([x[0] for x in r['batch']])} {ms([x[0] for x in r['stream']])} | "
              f"{ms([x[1] for x in r['batch']])} {ms([x[1] for x in r['stream']])}")
    # ASR-free question: do the two granularities even agree on a PERFECT transcript?
    same = [(gr[f"ref|{p}|batch"], gr[f"ref|{p}|stream"])
            for p in {a["passage_id"] for a in hyp.values()} if f"ref|{p}|batch" in gr]
    ident = sum(1 for a, b in same if a.split() == b.split())
    ov = [score_pair(a, b, rare)["word_recall"] for a, b in same]
    nb = [len(a.split()) for a, b in same]; ns = [len(b.split()) for a, b in same]
    print(f"\n  On a PERFECT transcript (no ASR at all), batch vs stream compression of")
    print(f"  the SAME reference agreed exactly on {ident}/{len(same)} passages.")
    print(f"  word-recall of batch output inside stream output: {ms(ov)}")
    print(f"  kept words: batch {ms([float(x) for x in nb])} stream {ms([float(x) for x in ns])}")


def ablation_error_type(asr, comp, refs, rare):
    print("\n" + "=" * 78)
    print("ABLATION C: which ERROR TYPE drives amplification, and does the CAUSE of")
    print("  the WER matter (noise-induced vs small-model-induced)?")
    print("=" * 78)
    rows = []
    for key, a in asr.items():
        if a["split"] != "eval":
            continue
        pid = a["passage_id"]
        ck, hk = f"ref|{pid}|flat|0.33", f"hyp|{key}|flat|0.33"
        if ck not in comp or hk not in comp:
            continue
        w = wer(refs[pid], a["hyp"])
        s0 = score_pair(refs[pid], a["hyp"], rare)
        sc = score_pair(comp[ck]["text"], comp[hk]["text"], rare)
        A = amplification(1 - s0["word_recall"], 1 - sc["word_recall"])
        tot = max(1, w["sub"] + w["dele"] + w["ins"])
        rows.append({**a, "wer": w["wer"], "A": A,
                     "f_sub": w["sub"] / tot, "f_del": w["dele"] / tot, "f_ins": w["ins"] / tot,
                     "Lc": 1 - sc["word_recall"], "L0": 1 - s0["word_recall"]})
    # matched-WER comparison: bucket by WER, compare noise-caused vs model-caused
    print("  Matched-WER buckets. 'noise' = base.en on degraded audio;")
    print("  'capacity' = a smaller model on CLEAN audio.")
    print(f"  {'WER band':>12} {'source':>9} | {'n':>3} {'mean WER':>9} {'A':>16} {'Lc':>16}")
    bands = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 0.60), (0.60, 2.0)]
    for lo, hi in bands:
        for src, sel in (("noise", lambda r: r["model"] == "base" and r["cond"] != "clean"),
                         ("capacity", lambda r: r["cond"] == "clean")):
            rs = [r for r in rows if lo <= r["wer"] < hi and sel(r)]
            if not rs:
                continue
            print(f"  {f'{lo:.2f}-{hi:.2f}':>12} {src:>9} | {len(rs):>3} "
                  f"{st.mean([r['wer'] for r in rs]):>9.3f} {ms([r['A'] for r in rs])} "
                  f"{ms([r['Lc'] for r in rs])}")
    # correlation of A with each error fraction
    import math
    def corr(xs, ys):
        xs, ys = list(xs), list(ys)
        if len(xs) < 3: return float("nan")
        mx, my = st.mean(xs), st.mean(ys)
        num = sum((a-mx)*(b-my) for a, b in zip(xs, ys))
        den = math.sqrt(sum((a-mx)**2 for a in xs) * sum((b-my)**2 for b in ys))
        return num/den if den else float("nan")
    va = [r for r in rows if r["A"] is not None]
    print(f"\n  Pearson r of amplification A against the error mix (n={len(va)}):")
    for f, lbl in (("f_sub", "substitution share"), ("f_del", "deletion share"),
                   ("f_ins", "insertion share"), ("wer", "WER itself")):
        print(f"    A vs {lbl:22s} r = {corr([r[f] for r in va], [r['A'] for r in va]):+.3f}")
    return rows


def failure_modes(rows, asr, comp, refs, rare):
    print("\n" + "=" * 78)
    print("FAILURE MODES: the cells where compression does the most damage")
    print("=" * 78)
    va = sorted((r for r in rows if r["A"] is not None), key=lambda r: -r["A"])
    print(f"  worst 6 of {len(va)} scored cells by amplification A:")
    for r in va[:6]:
        print(f"    A={r['A']:5.2f}  {r['model']:>5}/{r['cond']:>6} {r['passage_id']:<20} "
              f"WER={r['wer']:.2f}  L0={r['L0']:.2f} -> Lc={r['Lc']:.2f}")
    worst = va[0]
    key = f"{worst['model']}|{worst['passage_id']}|{worst['cond']}"
    print(f"\n  worst cell in full: {key}")
    print(f"    REF (first 150 chars)  : {refs[worst['passage_id']][:150]}")
    print(f"    HYP (first 150 chars)  : {asr[key]['hyp'][:150]}")
    cref = comp["ref|%s|flat|0.33" % worst["passage_id"]]["text"]
    chyp = comp["hyp|%s|flat|0.33" % key]["text"]
    print(f"    C(REF) (first 150)     : {cref[:150]}")
    print(f"    C(HYP) (first 150)     : {chyp[:150]}")


def length_collapse(asr, comp, refs):
    """How much of the reference's LENGTH is still there after ASR, and after
    compression on top of it.

    A compressor set to 'keep 33%' keeps 33% OF WHAT IT WAS GIVEN. When the
    recogniser has already dropped half the words, the two losses multiply: the
    nominal rate is not the end-to-end retention, and nothing in the pipeline
    reports the difference.
    """
    print("\n" + "=" * 78)
    print("LENGTH COLLAPSE: the nominal compression rate is not the retention rate")
    print("=" * 78)
    g = defaultdict(list)
    for key, a in asr.items():
        if a["split"] != "eval":
            continue
        hk = f"hyp|{key}|flat|0.33"
        if hk not in comp:
            continue
        n_ref = len(refs[a["passage_id"]].split())
        n_hyp = comp[hk]["n_in"]
        n_out = comp[hk]["n_out"]
        g[(a["model"], a["cond"])].append((n_hyp / n_ref, n_out / max(1, n_hyp),
                                           n_out / n_ref))
    print(f"  {'model':>6} {'cond':>7} | {'ASR keeps':>16} {'compressor keeps':>16} "
          f"{'END-TO-END kept':>16}")
    order = {"clean": 0, "snr10": 1, "snr5": 2, "snr0": 3, "snr-5": 4}
    for k in sorted(g, key=lambda t: (t[0], order[t[1]])):
        r = g[k]
        print(f"  {k[0]:>6} {k[1]:>7} | {ms([x[0] for x in r])} {ms([x[1] for x in r])} "
              f"{ms([x[2] for x in r])}")
    print("\n  'compressor keeps' is measured against the transcript it was handed;")
    print("  'END-TO-END kept' is measured against the words actually spoken.")


def main():
    corpus, asr, comp, refs, rare, eval_pids = load()
    length_collapse(asr, comp, refs)
    ablation_punctuation(asr, comp, refs, rare)
    ablation_granularity(corpus, asr, refs, rare)
    rows = ablation_error_type(asr, comp, refs, rare)
    failure_modes(rows, asr, comp, refs, rare)
    return 0


if __name__ == "__main__":
    sys.exit(main())
