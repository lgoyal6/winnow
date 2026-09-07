"""External-consumer fixture for the install and upgrade check."""

from __future__ import annotations

import json

from attentionrag.core import select_sentences
from token_merge import merge_compress


def main() -> None:
    selected, indices = select_sentences(
        "Alpha is noise. Beta has the answer.",
        [(0, 5), (6, 8), (16, 20), (21, 24), (25, 28), (29, 35)],
        [0.1, 0.1, 0.1, 0.9, 0.8, 0.7],
        top_k=2,
    )
    merged = merge_compress(
        "alpha beta gamma",
        [["alpha", 1], ["beta", 0], ["gamma", 1]],
        [(6, 10), (11, 16)],
        mode="union",
    )
    print(json.dumps({
        "attentionrag": {"selected": selected, "indices": indices},
        "token_merge": {
            "compressed_prompt": merged["compressed_prompt"],
            "n_kept": merged["n_kept"],
            "fallback": merged["used_llmlingua_fallback"],
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
