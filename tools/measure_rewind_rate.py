"""Measure how often `reconstruct_word_spans` hits its zero-width sentinel
fallback ("cursor rewind would have triggered") on realistic inputs.

Why this exists
---------------
The previous global-find fallback could silently rewind the cursor and produce
a cascade of overlapping spans (the union token explosion). The fix replaces
that fallback with a zero-width sentinel. We claimed the path is "rare" —
this script makes that claim falsifiable.

What it measures
----------------
Loads the real LongBench contexts already shipped in
`experiments/bench/data.json`, then generates per-word labels for each context
in two ways:

  baseline    : split on whitespace -> labels are EXACT verbatim substrings of
                the original. Zero unlocated entries expected. Lower bound.
  proxy_soft  : whitespace-split + NFC normalization only. NFC is a no-op for
                pure ASCII, so the rate isolates the impact of unicode
                normalization drift on tokens with combining marks.
  proxy_hard  : proxy_soft + ASCII-folding of smart quotes, en/em-dashes,
                ellipsis, and NBSP. This is an aggressive stress test — once
                a short token like '-' is hunted for in text that only
                contains '–', `find()` either misses (sentinel fires) OR
                matches an unrelated later hyphen and advances the cursor
                past valid future words (a forward-overshoot cascade). Both
                show up in n_unlocated_words. Numbers here are an UPPER bound.

These are NOT real LLMLingua output. Real LLMLingua-2 preserves most surface
forms; its actual drift rate sits somewhere between proxy_soft and proxy_hard
depending on how much the input deviates from ASCII.

To verify against REAL LLMLingua labels, run the bench
(`python experiments/bench/run_compress.py`) and read `n_unlocated_words` off
each `merge_compress` result.

Usage
-----
    python3 tools/measure_rewind_rate.py
"""

from __future__ import annotations

import json
import os
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from token_merge import merge_compress  # noqa: E402


# --------------------------------------------------------------------------- #
# Label generators                                                            #
# --------------------------------------------------------------------------- #
def _verbatim_labels(text: str) -> list:
    """Whitespace-split; every word is a verbatim substring of `text`."""
    return [[w, 1] for w in text.split()]


# Map of common normalizations LLMLingua-2 / sentencepiece tokenizers may apply.
_NORMALIZE_TABLE = str.maketrans({
    "‘": "'",   # left single quote
    "’": "'",   # right single quote
    "“": '"',   # left double quote
    "”": '"',   # right double quote
    "–": "-",   # en dash
    "—": "-",   # em dash
    "…": "...", # ellipsis
    " ": " ",   # non-breaking space
})


def _soft_normalized_labels(text: str) -> list:
    """NFC only. No-op on pure ASCII; isolates combining-mark drift."""
    return [[unicodedata.normalize("NFC", w), 1] for w in text.split()]


def _hard_normalized_labels(text: str) -> list:
    """NFC + ASCII-fold smart quotes/dashes/NBSP. Aggressive stress test."""
    pairs = []
    for raw in text.split():
        w = unicodedata.normalize("NFC", raw).translate(_NORMALIZE_TABLE)
        if w:
            pairs.append([w, 1])
    return pairs


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #
def _load_contexts() -> list:
    path = REPO_ROOT / "experiments" / "bench" / "data.json"
    with open(path) as f:
        d = json.load(f)
    return [ex["context"] for ex in d["examples"] if ex.get("context")]


def _measure(contexts: list, label_fn) -> dict:
    total_words = 0
    total_unlocated = 0
    per_ex = []
    for ctx in contexts:
        labels = label_fn(ctx)
        # Mode/spans don't matter for the counter — pick something cheap.
        out = merge_compress(ctx, labels, [(0, len(ctx))], mode="union")
        total_words += out["n_words"]
        total_unlocated += out["n_unlocated_words"]
        per_ex.append((out["n_words"], out["n_unlocated_words"]))
    return {
        "n_examples": len(contexts),
        "n_words": total_words,
        "n_unlocated": total_unlocated,
        "rate": (total_unlocated / total_words) if total_words else 0.0,
        "per_example_unlocated": [u for _w, u in per_ex],
    }


def main() -> int:
    contexts = _load_contexts()
    if not contexts:
        print("no contexts found", file=sys.stderr)
        return 1

    baseline = _measure(contexts, _verbatim_labels)
    soft = _measure(contexts, _soft_normalized_labels)
    hard = _measure(contexts, _hard_normalized_labels)

    print(f"Real LongBench contexts loaded: {baseline['n_examples']}")
    print(f"Total canonical words: {baseline['n_words']}")
    print()
    print(f"{'arm':<14} {'unlocated':>10} {'rate':>10}")
    print(f"{'-'*36}")
    print(f"{'baseline':<14} {baseline['n_unlocated']:>10} {baseline['rate']*100:>9.4f}%")
    print(f"{'proxy_soft':<14} {soft['n_unlocated']:>10} {soft['rate']*100:>9.4f}%")
    print(f"{'proxy_hard':<14} {hard['n_unlocated']:>10} {hard['rate']*100:>9.4f}%")
    print()
    if hard["n_unlocated"]:
        worst = max(range(len(contexts)), key=lambda i: hard["per_example_unlocated"][i])
        n_words = len(contexts[worst].split())
        print(f"proxy_hard worst example: index {worst}, "
              f"{hard['per_example_unlocated'][worst]}/{n_words} unlocated "
              f"({hard['per_example_unlocated'][worst]/max(1, n_words)*100:.2f}%)")
        print("  (large numbers usually reflect cursor forward-overshoot:")
        print("   one short normalized token like '-' falsely matches an")
        print("   unrelated character far ahead, pushing the cursor past")
        print("   valid future words. The counter captures both sentinel")
        print("   misses and post-overshoot misses — that's correct, both")
        print("   modes were silent before.)")
    print()
    print("baseline  = whitespace split (verbatim) -> 0 unlocated EXPECTED.")
    print("proxy_soft= NFC only. ASCII text -> 0; combining-mark drift only.")
    print("proxy_hard= NFC + smart-quote/dash/NBSP ASCII fold (aggressive).")
    print()
    print("These are PROXIES, not real LLMLingua output. For real rates, run")
    print("experiments/bench/run_compress.py and read n_unlocated_words off")
    print("each merge_compress result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
