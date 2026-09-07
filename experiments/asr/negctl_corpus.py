"""NEGATIVE CONTROL for verify_corpus.py, the speaker-disjointness proof.

verify_corpus.py printing ALL CORPUS CHECKS PASS is only evidence if it is
capable of printing something else. Each control below breaks exactly one
property of the corpus on disk, prints the file's sha256 before and after so the
mutation is shown to have landed in the file the checker actually reads, runs
the checker as a SUBPROCESS (so it re-reads the mutated file rather than a
module cached in this process), and asserts the specific check went red.

corpus.json is restored after every control, and the run ends by re-running the
unmutated checker.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "out" / "corpus.json"
PY = sys.executable


def sha() -> str:
    return hashlib.sha256(CORPUS.read_bytes()).hexdigest()[:12]


def run_checker():
    p = subprocess.run([PY, str(ROOT / "verify_corpus.py")],
                       capture_output=True, text=True, cwd=ROOT)
    return p.returncode, p.stdout


def control(name: str, mutate, expect_fail: str):
    before = sha()
    data = json.loads(CORPUS.read_text())
    mutate(data)
    CORPUS.write_text(json.dumps(data, indent=1))
    after = sha()
    print("=" * 70)
    print(f"CONTROL: {name}")
    if before == after:
        print("  !! corpus.json did not change; control is meaningless")
        return False
    print(f"  corpus.json sha256 {before} -> {after} (file the checker reads was mutated)")
    rc, out = run_checker()
    failed = [l.strip() for l in out.splitlines() if l.strip().startswith("FAIL")]
    print(f"  checker exit={rc}")
    for l in failed:
        print(f"    {l}")
    ok = rc != 0 and any(expect_fail in l for l in failed)
    print(f"  [{'CONTROL OK' if ok else 'CONTROL BROKEN'}] expected '{expect_fail}' to go red")
    shutil.copy(BAK, CORPUS)
    assert sha() == before, "restore failed"
    return ok


def leak_eval_speaker_into_tune(d):
    """The exact failure the split exists to prevent: a voice in both folds."""
    victim = d["splits"]["eval"][0]
    d["splits"]["tune"].append(json.loads(json.dumps(victim)))


def use_eval_voice_as_babble(d):
    """A voice under test also used as background noise."""
    spk = d["splits"]["eval"][0]["speaker"]
    ch = d["splits"]["eval"][0]["chapter"]
    d["babble_flacs"].append(
        str(ROOT / "data" / "LibriSpeech" / "test-clean" / spk / ch
            / f"{d['splits']['eval'][0]['utt_ids'][0]}.flac"))


def corrupt_reference_text(d):
    """Reference words no longer match the source transcript."""
    d["splits"]["eval"][0]["ref_words"][3] = "ZEBRAFISH"


def mislabel_speaker(d):
    """A passage's audio no longer belongs to its declared speaker."""
    other = d["splits"]["eval"][1]["speaker"]
    d["splits"]["eval"][0]["speaker"] = other


def shuffle_utterance_order(d):
    """Utterances no longer consecutive: the passage is not continuous prose."""
    u = d["splits"]["eval"][0]["utt_ids"]
    if len(u) >= 2:
        u[0], u[-1] = u[-1], u[0]


if __name__ == "__main__":
    tmp = tempfile.mkdtemp()
    BAK = Path(tmp) / "corpus.json"
    shutil.copy(CORPUS, BAK)
    baseline = sha()

    print("### BASELINE (unmutated corpus, checker must pass)")
    rc, out = run_checker()
    print(f"  exit={rc}  {out.strip().splitlines()[-1]}")
    assert rc == 0, "baseline corpus does not pass; controls would be meaningless"

    results = [
        control("an EVAL speaker is also placed in TUNE",
                leak_eval_speaker_into_tune, "TUNE n EVAL empty"),
        control("an EVAL voice is also used as babble noise",
                use_eval_voice_as_babble, "EVAL n BABBLE empty"),
        control("a passage's reference text is altered",
                corrupt_reference_text, "passage text == source"),
        control("a passage is attributed to the wrong speaker",
                mislabel_speaker, "eval audio paths match declared speaker"),
        control("a passage's utterances are no longer consecutive",
                shuffle_utterance_order, "utterances consecutive within chapter"),
    ]

    print("=" * 70)
    print("### RESTORED (must pass again)")
    assert sha() == baseline, "corpus.json was not restored"
    rc, out = run_checker()
    print(f"  corpus.json sha256 {sha()} (identical to baseline)")
    print(f"  exit={rc}  {out.strip().splitlines()[-1]}")

    good = all(results) and rc == 0
    print(f"\nNEGATIVE CONTROL {'VALID' if good else 'BROKEN'}: the disjointness proof "
          f"fails on {sum(results)}/{len(results)} seeded corpus defects and passes clean.")
    sys.exit(0 if good else 1)
