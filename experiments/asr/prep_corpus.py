"""Build a speaker-disjoint passage corpus from LibriSpeech test-clean.

A "passage" is a run of CONSECUTIVE utterances from ONE (speaker, chapter). Because
LibriSpeech segments a single continuous audiobook reading, consecutive utterances
from one chapter reconstruct the original continuous prose. Nothing is fabricated:
the passage text is the corpus's own reference transcript, concatenated in order.

Speakers are partitioned into THREE disjoint sets by a seeded shuffle:
  TUNE   - used to pick compressor settings / sanity-check the harness
  EVAL   - used for every reported number
  BABBLE - never a passage; only ever mixed in as background-noise voices,
           so a voice heard as noise is never a voice under test
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LS = ROOT / "data" / "LibriSpeech" / "test-clean"
SEED = 20260905

# Passage sizing. ~200 reference words is roughly 75-110s of LibriSpeech audio,
# long enough that a compressor has real redundancy to exploit and short enough
# that a 9-condition ASR grid finishes on a laptop.
MIN_WORDS = 190
MAX_UTTS = 24


def read_chapter(chdir: Path):
    """[(utt_id, ref_words, flac_path), ...] in utterance order."""
    trans = list(chdir.glob("*.trans.txt"))
    if len(trans) != 1:
        raise RuntimeError(f"expected exactly one .trans.txt in {chdir}, got {trans}")
    out = []
    for line in trans[0].read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        utt_id, _, text = line.partition(" ")
        flac = chdir / f"{utt_id}.flac"
        if not flac.exists():
            raise RuntimeError(f"missing audio for {utt_id}")
        out.append((utt_id, text.split(), flac))
    out.sort(key=lambda r: r[0])
    return out


def build_passages(speaker: str):
    """Greedy consecutive-utterance packing, one pass per chapter."""
    passages = []
    for chdir in sorted((LS / speaker).iterdir()):
        if not chdir.is_dir():
            continue
        utts = read_chapter(chdir)
        i = 0
        while i < len(utts):
            words, chunk = 0, []
            while i < len(utts) and words < MIN_WORDS and len(chunk) < MAX_UTTS:
                chunk.append(utts[i])
                words += len(utts[i][1])
                i += 1
            if words < MIN_WORDS:
                break  # trailing remainder: drop, never pad
            passages.append(
                {
                    "passage_id": f"{speaker}-{chdir.name}-{chunk[0][0].split('-')[-1]}",
                    "speaker": speaker,
                    "chapter": chdir.name,
                    "utt_ids": [u[0] for u in chunk],
                    "n_utts": len(chunk),
                    "ref_words": [w for u in chunk for w in u[1]],
                    "flacs": [str(u[2]) for u in chunk],
                }
            )
    return passages


def main():
    speakers = sorted((d.name for d in LS.iterdir() if d.is_dir()), key=int)
    rng = random.Random(SEED)
    shuffled = speakers[:]
    rng.shuffle(shuffled)

    # 14 / 14 / 12. TUNE and EVAL carry passages; BABBLE is noise voices only.
    tune = sorted(shuffled[:14], key=int)
    evl = sorted(shuffled[14:28], key=int)
    babble = sorted(shuffled[28:], key=int)

    # ---- disjointness proof, computed not asserted-by-comment ----
    proof = {
        "seed": SEED,
        "all_speakers": speakers,
        "n_all": len(speakers),
        "tune": tune,
        "eval": evl,
        "babble": babble,
        "tune_n_eval": sorted(set(tune) & set(evl), key=int),
        "tune_n_babble": sorted(set(tune) & set(babble), key=int),
        "eval_n_babble": sorted(set(evl) & set(babble), key=int),
        "union_equals_all": sorted(set(tune) | set(evl) | set(babble), key=int) == speakers,
    }
    for k in ("tune_n_eval", "tune_n_babble", "eval_n_babble"):
        assert proof[k] == [], f"speaker split is NOT disjoint: {k} = {proof[k]}"
    assert proof["union_equals_all"], "split does not cover all speakers"

    corpus = {"proof": proof, "splits": {}}
    for name, spk in (("tune", tune), ("eval", evl)):
        rows = []
        for s in spk:
            ps = build_passages(s)
            if ps:
                rows.append(ps[0])  # one passage per speaker: no speaker dominates
        # every passage's speaker must live in exactly this split
        for p in rows:
            assert p["speaker"] in spk, f"{p['passage_id']} leaked into {name}"
            for other in (evl if name == "tune" else tune) + babble:
                assert p["speaker"] != other
        corpus["splits"][name] = rows

    corpus["babble_flacs"] = [
        str(p) for s in babble for p in sorted((LS / s).rglob("*.flac"))[:4]
    ]

    out = ROOT / "out" / "corpus.json"
    out.write_text(json.dumps(corpus, indent=1))
    print(f"seed={SEED}")
    print(f"speakers total   : {len(speakers)}")
    print(f"TUNE   speakers  : {len(tune)}  {tune}")
    print(f"EVAL   speakers  : {len(evl)}  {evl}")
    print(f"BABBLE speakers  : {len(babble)}  {babble}")
    print(f"TUNE  n EVAL     : {proof['tune_n_eval']}   (must be [])")
    print(f"TUNE  n BABBLE   : {proof['tune_n_babble']}   (must be [])")
    print(f"EVAL  n BABBLE   : {proof['eval_n_babble']}   (must be [])")
    print(f"union == all     : {proof['union_equals_all']}")
    for name in ("tune", "eval"):
        rows = corpus["splits"][name]
        wc = [len(r["ref_words"]) for r in rows]
        print(
            f"{name:5s}: {len(rows)} passages, "
            f"{sum(wc)} ref words, {min(wc)}-{max(wc)} per passage, "
            f"speakers={sorted({r['speaker'] for r in rows}, key=int)}"
        )
    print(f"babble clips     : {len(corpus['babble_flacs'])}")
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
