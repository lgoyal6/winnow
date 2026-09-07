"""Independent verification of the inherited corpus. Recomputes everything from
the LibriSpeech source rather than trusting corpus.json's own 'proof' block."""
import json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LS = ROOT / "data" / "LibriSpeech" / "test-clean"
C = json.loads((ROOT / "out" / "corpus.json").read_text())
fails = []

def chk(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok: fails.append(name)

# 1. Speaker sets recomputed from the filesystem, not from the proof block.
disk_speakers = sorted((d.name for d in LS.iterdir() if d.is_dir()), key=int)
tune_s = sorted({p["speaker"] for p in C["splits"]["tune"]}, key=int)
eval_s = sorted({p["speaker"] for p in C["splits"]["eval"]}, key=int)
babble_s = sorted({Path(f).parts[-3] for f in C["babble_flacs"]}, key=int)
print(f"disk speakers: {len(disk_speakers)}")
print(f"tune={len(tune_s)} eval={len(eval_s)} babble={len(babble_s)}")
chk("TUNE n EVAL empty", set(tune_s) & set(eval_s) == set(), f"{sorted(set(tune_s)&set(eval_s))}")
chk("TUNE n BABBLE empty", set(tune_s) & set(babble_s) == set(), f"{sorted(set(tune_s)&set(babble_s))}")
chk("EVAL n BABBLE empty", set(eval_s) & set(babble_s) == set(), f"{sorted(set(eval_s)&set(babble_s))}")
chk("splits cover disk speakers", set(tune_s)|set(eval_s)|set(babble_s) == set(disk_speakers))

# 2. Every eval passage's audio files belong to that passage's speaker dir.
bad = [(p["passage_id"], f) for p in C["splits"]["eval"] for f in p["flacs"]
       if Path(f).parts[-3] != p["speaker"]]
chk("eval audio paths match declared speaker", not bad, str(bad[:3]))

# 3. Passage text really is the corpus reference transcript, verbatim & in order.
mismatch = []
for split in ("tune", "eval"):
    for p in C["splits"][split]:
        chdir = LS / p["speaker"] / p["chapter"]
        trans = {l.split(" ",1)[0]: l.split(" ",1)[1].strip()
                 for l in (chdir / f"{p['speaker']}-{p['chapter']}.trans.txt").read_text().splitlines() if l.strip()}
        want = [w for u in p["utt_ids"] for w in trans[u].split()]
        if want != p["ref_words"]:
            mismatch.append(p["passage_id"])
chk("passage text == source .trans.txt verbatim", not mismatch, str(mismatch[:3]))

# 4. Utterance ids are consecutive within the chapter (continuous prose).
noncons = []
for split in ("tune", "eval"):
    for p in C["splits"][split]:
        idx = [int(u.split("-")[-1]) for u in p["utt_ids"]]
        if idx != list(range(idx[0], idx[0]+len(idx))):
            noncons.append(p["passage_id"])
chk("utterances consecutive within chapter", not noncons, str(noncons[:3]))

# 5. No eval SPEAKER appears in babble audio (voice under test never used as noise).
chk("no eval voice used as babble", not (set(eval_s) & set(babble_s)))

print(f"\n{'ALL CORPUS CHECKS PASS' if not fails else 'FAILURES: ' + str(fails)}")
sys.exit(1 if fails else 0)
