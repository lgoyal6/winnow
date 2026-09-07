"""Scoring for the ASR-noise-vs-compression study. Pure functions, no models.

THE QUESTION THIS SCORES
------------------------
Winnow compresses a transcript before handing it to an LLM. On clean text the
compressor is known to keep the informative words. On REAL ASR output the input
already has words missing, words wrong and words invented. The question is
whether compression is TRANSPARENT to that damage (it loses only what ASR
already lost) or whether it AMPLIFIES it.

So every condition is scored twice, against the same ground-truth reference:

  L0 = 1 - recall(ref            -> hyp            )   damage from ASR alone
  Lc = 1 - recall(compress(ref)  -> compress(hyp)  )   damage after compression

and the headline quantity is the amplification factor

  A = Lc / L0

  A = 1  compression is transparent: it discards content in exactly the
         proportion ASR already destroyed it.
  A > 1  compression AMPLIFIES ASR error: content that survived the recogniser
         is then thrown away (or displaced by misrecognised words) by the
         compressor.
  A < 1  compression ABSORBS ASR error: the words ASR got wrong were words the
         compressor was going to drop anyway.

"Content" is deliberately NOT defined by a curated stopword list, which would be
an arbitrary knob. Two definitions are reported instead:
  * word_recall - every word, no filtering at all.
  * rare_recall - only words that are rare in this corpus (frequency <= RARE_MAX
    over the union of all reference passages). Proper nouns, numbers and topic
    words live here; these are the words whose loss actually changes meaning,
    and they are exactly where ASR errors concentrate.

Recall is MULTISET recall, so a word the reference uses three times must appear
three times to score three times. Bag-of-words set recall would hide repetition
errors entirely.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence

RARE_MAX = 2  # a word occurring <= this many times across all references is "rare"

_NONWORD = re.compile(r"[^a-z0-9' ]+")
_APOS = re.compile(r"(^'+|'+$)")


def norm_words(text: str) -> List[str]:
    """Surface-form-independent tokenisation.

    Whisper emits case and punctuation; LibriSpeech references have neither. All
    scoring runs through here so a comparison never rewards or punishes a
    surface-form difference, only a difference in the words themselves.
    """
    t = text.lower().replace("-", " ").replace("’", "'")
    t = _NONWORD.sub(" ", t)
    return [w for w in (_APOS.sub("", x) for x in t.split()) if w]


def multiset_recall(ref: Sequence[str], hyp: Sequence[str]) -> float:
    """|ref n hyp| / |ref| over multisets. Empty ref -> 1.0 (nothing to lose)."""
    if not ref:
        return 1.0
    return sum((Counter(ref) & Counter(hyp)).values()) / len(ref)


def build_rare_vocab(reference_texts: Iterable[str], rare_max: int = RARE_MAX):
    """Words occurring at most `rare_max` times across ALL references.

    Computed from references only. A hypothesis never influences which words
    count as rare, so a hallucinating recogniser cannot move its own goalposts.
    """
    c = Counter(w for t in reference_texts for w in norm_words(t))
    return frozenset(w for w, n in c.items() if n <= rare_max), c


def score_pair(ref_text: str, hyp_text: str, rare: frozenset) -> Dict[str, float]:
    r, h = norm_words(ref_text), norm_words(hyp_text)
    rr = [w for w in r if w in rare]
    hr = [w for w in h if w in rare]
    return {
        "word_recall": multiset_recall(r, h),
        "rare_recall": multiset_recall(rr, hr),
        "n_ref": len(r), "n_hyp": len(h), "n_ref_rare": len(rr),
    }


def amplification(loss_uncompressed: float, loss_compressed: float,
                  min_l0: float = 0.02):
    """Lc / L0, or None when L0 is too small to divide by.

    Below `min_l0` the transcript is essentially clean, the ratio is dominated by
    noise in a near-zero denominator, and reporting it would invent structure.
    """
    if loss_uncompressed < min_l0:
        return None
    return loss_compressed / loss_uncompressed


def wer(ref_text: str, hyp_text: str) -> Dict[str, float]:
    """Standard WER plus its S/D/I breakdown, via jiwer on normalised words."""
    import jiwer
    r, h = " ".join(norm_words(ref_text)), " ".join(norm_words(hyp_text))
    if not r:
        return {"wer": 0.0, "sub": 0, "dele": 0, "ins": 0, "hits": 0}
    o = jiwer.process_words([r], [h])
    return {"wer": o.wer, "sub": o.substitutions, "dele": o.deletions,
            "ins": o.insertions, "hits": o.hits}
