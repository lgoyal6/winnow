"""How short does an utterance have to be before the compressor erases it?

This matters because of the architecture, not as a curiosity: Winnow compresses
ONE UTTERANCE per request, and conversational speech is full of very short ones.
"""
import json, statistics as st
from pathlib import Path
import compress_lib as C

ROOT = Path(__file__).resolve().parent
comp, dev = C.load_compressor()
corpus = json.loads((ROOT / "out" / "corpus.json").read_text())
words = " ".join(corpus["splits"]["eval"][0]["ref_words"]).split()

print(f"device={dev}\n\nPrefixes of a real reference passage, rate 0.33:")
print(f"  {'in':>4} {'out':>4}  {'kept':>6}  first words of output")
first_nonempty = None
for n in list(range(1, 21)) + [25, 30, 40, 60, 100]:
    txt = " ".join(words[:n])
    out, _ = C.compress(comp, txt, rate=0.33)
    k = len(out.split())
    if k > 0 and first_nonempty is None:
        first_nonempty = n
    print(f"  {n:>4} {k:>4}  {k/n:>6.2f}  {out[:56]}".rstrip())
print(f"\nFirst input length that survives at all: {first_nonempty} words")

# What fraction of REAL whisper utterances are at or below that length?
segs = json.loads((ROOT / "out" / "asr_segments.json").read_text())
lens = [len(s["text"].split()) for r in segs.values() for s in r["segments"]]
lens.sort()
below = sum(1 for l in lens if l < first_nonempty)
print(f"\nReal whisper segments in the eval set: {len(lens)}")
print(f"  median length {st.median(lens):.0f} words, "
      f"10th pct {lens[len(lens)//10]} words, min {lens[0]}")
print(f"  segments shorter than {first_nonempty} words: {below} "
      f"({100*below/len(lens):.1f}%) - these would be erased entirely")

# LibriSpeech is read audiobook prose. Conversational speech is far choppier,
# so state the caveat with the measured number rather than extrapolating.
print("\nNOTE: LibriSpeech is read audiobook prose with long sentences. Spontaneous")
print("conversation has many more short utterances, so this percentage is a")
print("LOWER bound on how often a live session would hit the erasure.")
