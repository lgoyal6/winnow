"""
Token-space compression arms for the Winnow bench.

Produces compressed.json (SCHEMA.md shape) with 3 arms per example:
  - lingua        : LLMLingua-2 only
  - union         : LLMLingua-2 ∪ AttentionRAG  (merge_compress mode="union")
  - intersection  : LLMLingua-2 ∩ AttentionRAG  (merge_compress mode="intersection")

Replicates server.py /compress merge wiring exactly (single_chunk + attn_empty
fallback). Token counts for ALL THREE arms via one consistent tokenizer
(xlm-roberta from the LLMLingua-2 model), falling back to whitespace word counts
if the tokenizer can't be loaded.
"""

import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

# Repo root on path for token_merge.
REPO_ROOT = "/Users/swastikagrawal/Documents/winnow"
BENCH_DIR = os.path.join(REPO_ROOT, "experiments", "bench")
sys.path.insert(0, REPO_ROOT)

import modal  # noqa: E402

from token_merge import merge_compress  # noqa: E402

DATA_PATH = os.path.join(BENCH_DIR, "data.json")
OUT_PATH = os.path.join(BENCH_DIR, "compressed.json")

RATE = 0.5
CHUNK_SIZE = 300
TOP_K = 12
MAX_CONCURRENCY = 4

# --------------------------------------------------------------------------- #
# Tokenizer (consistent token counts across all 3 arms)                       #
# --------------------------------------------------------------------------- #
_TOK = None
_TOK_KIND = None  # "xlm-roberta" or "whitespace"


def load_tokenizer():
    global _TOK, _TOK_KIND
    try:
        from transformers import AutoTokenizer

        from model_guard import guarded_from_pretrained

        _TOK = guarded_from_pretrained(
            AutoTokenizer, "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
        )
        _TOK_KIND = "xlm-roberta"
        print("[tok] loaded xlm-roberta tokenizer", flush=True)
    except Exception as exc:
        print(f"[tok] FALLBACK to whitespace word counts ({exc})", flush=True)
        _TOK, _TOK_KIND = None, "whitespace"


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _TOK is not None:
        return len(_TOK.encode(text, add_special_tokens=False))
    return len(text.split())


# --------------------------------------------------------------------------- #
# data.json polling                                                           #
# --------------------------------------------------------------------------- #
def wait_for_data(timeout_s=300, interval_s=10):
    waited = 0
    while not os.path.exists(DATA_PATH):
        if waited >= timeout_s:
            return False
        print(f"[data] data.json not found; waited {waited}s, sleeping {interval_s}s...", flush=True)
        time.sleep(interval_s)
        waited += interval_s
    return True


# --------------------------------------------------------------------------- #
# Per-example compression (replicates server.py /compress merge exactly)      #
# --------------------------------------------------------------------------- #
def process_example(ex, compressor, attn_service):
    ex_id = ex["id"]
    context = ex["context"]
    question = ex.get("question", "")

    # (a) LLMLingua-2.
    llm_out = compressor.compress.remote(context, rate=RATE, return_labels=True)
    lingua_prompt = llm_out["compressed_prompt"]
    word_labels = llm_out.get("fn_labeled_original_prompt") or llm_out.get("word_labels")
    if not word_labels:
        raise RuntimeError("LLMLingua returned no labels")

    # (b) AttentionRAG (fall back to lingua-only on failure).
    try:
        attn_out = attn_service.compress_spans.remote(
            context, question, chunk_size=CHUNK_SIZE, top_k=TOP_K, use_openai_hint=False
        )
        kept_spans = attn_out.get("kept_spans", [])
        # server.py wiring: single_chunk overrides an is_empty_prefix gate.
        single_chunk = attn_out.get("n_chunks", 0) == 1
        attn_empty = attn_out.get("is_empty_prefix", False) and not single_chunk
    except Exception as exc:
        print(f"[{ex_id}] AttentionRAG failed -> lingua-only fallback: {exc}", flush=True)
        kept_spans, attn_empty = [], True

    # (c) merge: union + intersection (called EXACTLY as server.py /compress does).
    union = merge_compress(context, word_labels, kept_spans, mode="union", attnrag_empty=attn_empty)
    inter = merge_compress(context, word_labels, kept_spans, mode="intersection", attnrag_empty=attn_empty)

    # splice_kept can occasionally duplicate large slices of the original when a
    # canonical word fails its in-order substring match (a pre-existing quirk in
    # token_merge.splice_kept), producing a prompt far LONGER than the original.
    # The keep-DECISIONS in merged["word_labels"] are still correct, so when the
    # spliced text overshoots the original length we rebuild the prompt from the
    # kept words (loses only exact original spacing). Normal case is untouched.
    def merged_prompt(m):
        p = m["compressed_prompt"]
        if len(p) > len(context):
            return " ".join(w for w, k in m["word_labels"] if k)
        return p

    union_prompt = merged_prompt(union)
    inter_prompt = merged_prompt(inter)

    # (3) consistent token counts. origin = full context tokenized once.
    origin_tokens = count_tokens(context)

    def arm(prompt):
        c = count_tokens(prompt)
        return {
            "prompt": prompt,
            "origin_tokens": origin_tokens,
            "compressed_tokens": c,
            "retention": (c / origin_tokens) if origin_tokens else 0.0,
        }

    return {
        "id": ex_id,
        "arms": {
            "lingua": arm(lingua_prompt),
            "union": arm(union_prompt),
            "intersection": arm(inter_prompt),
        },
    }


def main():
    if not wait_for_data():
        print("ERROR: data.json never appeared after ~5 min. Exiting.", flush=True)
        sys.exit(2)

    with open(DATA_PATH) as f:
        data = json.load(f)
    examples = data.get("examples", [])
    print(f"[data] loaded {len(examples)} examples", flush=True)

    load_tokenizer()

    print("[modal] connecting to deployed classes...", flush=True)
    Compressor = modal.Cls.from_name("llmlingua2-xlm", "Compressor")
    AttnService = modal.Cls.from_name("attentionrag", "AttentionRAGService")
    compressor = Compressor()
    attn_service = AttnService()

    results = []
    failed = []
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
        futs = {
            pool.submit(process_example, ex, compressor, attn_service): ex["id"]
            for ex in examples
        }
        for fut in as_completed(futs):
            ex_id = futs[fut]
            try:
                results.append(fut.result())
                print(f"[ok] {ex_id}", flush=True)
            except Exception:
                failed.append(ex_id)
                print(f"[FAIL] {ex_id}\n{traceback.format_exc()}", flush=True)

    # Preserve input order.
    order = {ex["id"]: i for i, ex in enumerate(examples)}
    results.sort(key=lambda r: order.get(r["id"], 1 << 30))

    with open(OUT_PATH, "w") as f:
        json.dump({"examples": results}, f, indent=2)

    # Means.
    def mean_ret(arm_name):
        vals = [r["arms"][arm_name]["retention"] for r in results]
        return sum(vals) / len(vals) if vals else 0.0

    print("\n===== SUMMARY =====", flush=True)
    print(f"compressed.json: {OUT_PATH}", flush=True)
    print(f"tokenizer: {_TOK_KIND}", flush=True)
    print(f"examples compressed: {len(results)} / {len(examples)}", flush=True)
    if failed:
        print(f"failed ids: {failed}", flush=True)
    for a in ("lingua", "union", "intersection"):
        print(f"mean retention [{a}]: {mean_ret(a):.4f}", flush=True)


if __name__ == "__main__":
    main()
