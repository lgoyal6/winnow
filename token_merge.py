"""
Token-level merge of two compressors (LLMLingua-2 + AttentionRAG) over the
ORIGINAL text, preserving chronology.

The merge walks the original word-by-word and keeps a word iff a boolean rule
over the two methods' keep-decisions holds:

  * intersection: keep iff BOTH kept the word
  * union:        keep iff EITHER kept the word

Crucially this is an ALIGNMENT, not a set/string-membership test. The two
methods are aligned to ONE canonical token sequence -- LLMLingua-2's per-word
label list -- which is already an in-order segmentation of the original:

  * LLMLingua keep-mask  = its labels directly (1=keep, 0=drop).
  * AttentionRAG mask    = char-span overlap: each canonical word's char-span
    (reconstructed by an in-order forward scan of the original) is tested
    against AttentionRAG's kept sentence char-spans.

Empty-AttentionRAG fallback: if AttentionRAG kept nothing (all chunks gated
'none'), we fall back to LLMLingua-only so we never wipe the whole text.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from typing import List, Optional, Sequence, Tuple

_TRUE = ("1", "1.0", "true", "True")


def _coerce_label(l) -> int:
    s = str(l).strip()
    if s in _TRUE:
        return 1
    try:
        return 1 if int(round(float(s))) >= 1 else 0
    except ValueError:
        return 0


# --------------------------------------------------------------------------- #
# LLMLingua label normalization                                               #
# --------------------------------------------------------------------------- #
def normalize_labels(word_labels) -> List[Tuple[str, int]]:
    """Coerce LLMLingua's word-label output into [(word, 0|1), ...].

    Handles the real LLMLingua-2 `fn_labeled_original_prompt` STRING format:
        'Daniel 1\\t\\t|\\t\\ttravelled 1\\t\\t|\\t\\tto 0 ...'
    (each entry is "<word> <label>", entries separated by '|' with surrounding
    whitespace), as well as a list of [word, label] pairs and a 'word,label'
    fallback. Labels are coerced to 0/1.
    """
    if not word_labels:
        return []
    pairs: List[Tuple[str, int]] = []

    if isinstance(word_labels, str):
        # Real format is '|'-delimited "word label" entries; fall back to
        # whitespace-split for a 'word,label ...' style string.
        if "|" in word_labels:
            parts = re.split(r"\s*\|\s*", word_labels)
        else:
            parts = word_labels.split()
        for part in parts:
            part = part.strip()
            if not part:
                continue
            toks = part.split()
            if len(toks) >= 2 and toks[-1] in ("0", "1"):
                w, l = " ".join(toks[:-1]), toks[-1]
            elif "," in part:
                w, l = part.rsplit(",", 1)
            else:
                w, l = part, "1"
            pairs.append((w, _coerce_label(l)))
        return pairs

    for item in word_labels:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            w, l = item[0], item[1]
        else:  # bare word -> assume kept
            w, l = item, 1
        pairs.append((str(w), _coerce_label(l)))
    return pairs


# --------------------------------------------------------------------------- #
# Reconstruct char-spans for the canonical words                              #
# --------------------------------------------------------------------------- #
def reconstruct_word_spans(
    original: str, labeled_words: Sequence[Tuple[str, int]]
) -> List[Tuple[str, int, int, int]]:
    """Locate each labeled word in `original`, in order, returning
    (word, label, start, end). Uses a moving cursor so repeated words map to
    successive occurrences (no duplicate ambiguity)."""
    spans: List[Tuple[str, int, int, int]] = []
    cursor = 0
    for word, label in labeled_words:
        if word == "":
            spans.append((word, label, cursor, cursor))
            continue
        idx = original.find(word, cursor)
        if idx == -1:  # tolerant fallback: search from start
            idx = original.find(word)
        if idx == -1:  # unmatched -> zero-width at cursor (rare)
            spans.append((word, label, cursor, cursor))
            continue
        end = idx + len(word)
        spans.append((word, label, idx, end))
        cursor = end
    return spans


# --------------------------------------------------------------------------- #
# AttentionRAG mask via char-span overlap                                     #
# --------------------------------------------------------------------------- #
def _coalesce(spans: Sequence[Tuple[int, int]]) -> Tuple[List[int], List[int]]:
    """Sort and coalesce kept spans into disjoint, non-touching intervals.

    Returns parallel (starts, ends) lists. Because the survivors never touch,
    BOTH lists are strictly increasing, which is what lets `attnrag_mask` find
    a word's candidate interval with one bisect instead of a scan.

    Coalescing touching spans (``s <= cur_end``) does not change any overlap
    answer: a word can only sit in the seam between two touching spans if it is
    zero-width at the shared boundary, and that word already matched the right
    span under the per-span rule.
    """
    starts: List[int] = []
    ends: List[int] = []
    for s, e in sorted((int(a), int(b)) for a, b in spans):
        if ends and s <= ends[-1]:
            if e > ends[-1]:
                ends[-1] = e
        else:
            starts.append(s)
            ends.append(e)
    return starts, ends


def attnrag_mask(
    word_spans: Sequence[Tuple[str, int, int, int]],
    kept_spans: Sequence[Tuple[int, int]],
) -> List[bool]:
    """Keep-mask over the canonical words: True iff the word's char-span meets a
    kept AttentionRAG span.

    The rule is unchanged: half-open overlap (``s < ke and e > ks``), plus a
    zero-width word counting as inside when ``ks <= s < ke``. What changed is
    the lookup. This used to test every word against every kept span, which is
    O(words x spans); on a 36.5k-word input with 1739 kept spans that quadratic
    term dominated the whole merge (see .agent-work/scale_merge.py). Coalescing
    the spans once and bisecting makes it O(words log spans).
    """
    starts, ends = _coalesce(kept_spans)
    if not starts:
        return [False] * len(word_spans)

    mask: List[bool] = []
    n = len(starts)
    for _w, _l, s, e in word_spans:
        i = bisect_right(ends, s)  # first interval whose end is past s
        if i >= n:
            mask.append(False)
            continue
        ks = starts[i]
        mask.append(ks <= s if s == e else ks < e)
    return mask


# --------------------------------------------------------------------------- #
# Reconstruct compressed text preserving original spacing for kept runs       #
# --------------------------------------------------------------------------- #
def splice_kept(
    original: str,
    word_spans: Sequence[Tuple[str, int, int, int]],
    keep: Sequence[bool],
) -> str:
    """Reconstruct the compressed text from the kept canonical words.

    Each kept word contributes ONLY its own token - the exact original substring
    when its located span is valid & forward, else the label text - joined by a
    single space. We deliberately do NOT re-slice arbitrary ``original[prev_end:s]``
    gaps: when ``reconstruct_word_spans`` yields non-monotonic spans (repeated
    tokens make the moving ``find()`` reset backward), that gap-fill re-inserts
    large overlapping spans and blows the output up many-fold - worst in union,
    which keeps long unbroken runs of words. Joining kept tokens keeps the output
    bounded by the kept content (no duplication possible).
    """
    n = len(original)
    out: List[str] = []
    for (word, _l, s, e), k in zip(word_spans, keep):
        if not k:
            continue
        out.append(original[s:e] if 0 <= s < e <= n else word)
    return " ".join(out)


# --------------------------------------------------------------------------- #
# Main merge                                                                   #
# --------------------------------------------------------------------------- #
def merge_compress(
    original: str,
    word_labels,
    kept_spans: Sequence[Tuple[int, int]],
    mode: str = "intersection",
    attnrag_empty: bool = False,
) -> dict:
    """Combine LLMLingua labels with AttentionRAG kept-spans over `original`.

    Returns compressed_prompt, merged per-word labels (for the strike-through
    UI), and the two source masks for diagnostics.
    """
    if mode not in ("intersection", "union"):
        raise ValueError(f"mode must be 'intersection' or 'union', got {mode!r}")

    pairs = normalize_labels(word_labels)
    word_spans = reconstruct_word_spans(original, pairs)
    mask_L = [bool(l) for _w, l, _s, _e in word_spans]

    # Empty-AttentionRAG fallback -> LLMLingua only (never wipe the text).
    empty = attnrag_empty or not kept_spans
    if empty:
        mask_A = [False] * len(word_spans)
        merged = list(mask_L)
        used_fallback = True
    else:
        mask_A = attnrag_mask(word_spans, kept_spans)
        if mode == "intersection":
            merged = [a and l for a, l in zip(mask_A, mask_L)]
        else:
            merged = [a or l for a, l in zip(mask_A, mask_L)]
        used_fallback = False

    compressed = splice_kept(original, word_spans, merged)
    merged_labels = [[w, 1 if k else 0] for (w, _l, _s, _e), k in zip(word_spans, merged)]

    return {
        "compressed_prompt": compressed,
        "word_labels": merged_labels,
        "mode": mode,
        "used_llmlingua_fallback": used_fallback,
        "n_words": len(word_spans),
        "n_kept": sum(1 for k in merged if k),
        "mask_llmlingua": [1 if x else 0 for x in mask_L],
        "mask_attentionrag": [1 if x else 0 for x in mask_A],
    }
