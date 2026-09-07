"""Score the compression grid and answer the question.

Reported numbers come from the EVAL speakers only. TUNE speakers exist so that
any choice made while building this harness (rate grid, surface form, rare-word
threshold) was made while looking at different voices than the ones reported on.
"""
from __future__ import annotations
import json, statistics as st, sys
from collections import defaultdict
from pathlib import Path

from metrics import amplification, build_rare_vocab, score_pair, wer

ROOT = Path(__file__).resolve().parent
SPLIT = "eval"


def load():
    corpus = json.loads((ROOT / "out" / "corpus.json").read_text())
    asr = json.loads((ROOT / "out" / "asr.json").read_text())
    comp = json.loads((ROOT / "out" / "compressed.json").read_text())
    refs = {p["passage_id"]: " ".join(p["ref_words"])
            for s in ("tune", "eval") for p in corpus["splits"][s]}
    return corpus, asr, comp, refs


def build_rows(asr, comp, refs, split=SPLIT, rates=(0.5, 0.33, 0.2), form="flat"):
    # Rare vocabulary is built from the references of THIS split only.
    split_pids = {k.split("|")[1] for k, v in asr.items() if v["split"] == split}
    rare, _ = build_rare_vocab([refs[p] for p in sorted(split_pids)])
    rows = []
    for key, a in asr.items():
        if a["split"] != split:
            continue
        pid, ref = a["passage_id"], refs[a["passage_id"]]
        w = wer(ref, a["hyp"])
        s0 = score_pair(ref, a["hyp"], rare)          # uncompressed damage
        for r in rates:
            ck, hk = f"ref|{pid}|flat|{r}", f"hyp|{key}|{form}|{r}"
            if ck not in comp or hk not in comp:
                continue
            cr, ch = comp[ck], comp[hk]
            sc = score_pair(cr["text"], ch["text"], rare)
            rows.append({
                "model": a["model"], "cond": a["cond"], "snr": a["snr_target"],
                "passage_id": pid, "rate": r, "form": form,
                "wer": w["wer"], "sub": w["sub"], "dele": w["dele"], "ins": w["ins"],
                "L0": 1 - s0["word_recall"], "Lc": 1 - sc["word_recall"],
                "L0_rare": 1 - s0["rare_recall"], "Lc_rare": 1 - sc["rare_recall"],
                "A": amplification(1 - s0["word_recall"], 1 - sc["word_recall"]),
                "A_rare": amplification(1 - s0["rare_recall"], 1 - sc["rare_recall"]),
                "achieved_rate_ref": cr["n_out"] / max(1, cr["n_in"]),
                "achieved_rate_hyp": ch["n_out"] / max(1, ch["n_in"]),
                "n_ref_words": s0["n_ref"],
            })
    return rows, rare


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0), len(vals)


def table(rows, group_keys, cols, title):
    print(f"\n{title}")
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in group_keys)].append(r)
    head = "  " + "  ".join(f"{k:>8}" for k in group_keys)
    head += "  " + "  ".join(f"{c:>16}" for c in cols) + "      n"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for k in sorted(g, key=lambda t: tuple(str(x) for x in t)):
        rs = g[k]
        line = "  " + "  ".join(f"{str(x):>8}" for x in k)
        for c in cols:
            m, sd, n = agg([r[c] for r in rs])
            line += f"  {m:>8.3f}+-{sd:<5.3f}" if m is not None else f"  {'--':>16}"
        print(line + f"  {len(rs):>5}")


def main():
    corpus, asr, comp, refs = load()
    rows, rare = build_rows(asr, comp, refs)
    if not rows:
        print("no scored rows yet"); return 1
    print(f"EVAL split: {len({r['passage_id'] for r in rows})} passages, "
          f"{len({r['model'] for r in rows})} models, "
          f"{len({r['cond'] for r in rows})} conditions, {len(rows)} scored cells")
    print(f"rare vocabulary: {len(rare)} word types (freq <= 2 in eval references)")

    table(rows, ["model", "cond"], ["wer", "L0", "Lc", "A"],
          "WER and amplification by model x condition (rate 0.33 + 0.5 + 0.2 pooled)")
    r33 = [r for r in rows if r["rate"] == 0.33]
    table(r33, ["model", "cond"], ["wer", "L0", "Lc", "A", "A_rare"],
          "RATE 0.33 only - headline grid")
    table(rows, ["rate"], ["wer", "L0", "Lc", "A", "A_rare"],
          "ABLATION: compression rate")
    json.dump(rows, open(ROOT / "out" / "scored.json", "w"), indent=1)
    print(f"\nwrote out/scored.json ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
