"""
One-off: compress DEVPOST.md with LLMLingua-2 UNION AttentionRAG.

Replicates server.py /compress merge wiring exactly (single_chunk + attn_empty
fallback, splice overshoot guard). Writes DEVPOST_compressed.md and prints stats.

Run: python3 experiments/bench/compress_devpost.py
"""

import os
import sys

REPO_ROOT = "/Users/swastikagrawal/Documents/winnow"
sys.path.insert(0, REPO_ROOT)

import modal  # noqa: E402

from token_merge import merge_compress  # noqa: E402

SRC = os.path.join(REPO_ROOT, "DEVPOST.md")
OUT = os.path.join(REPO_ROOT, "DEVPOST_compressed.md")

RATE = 0.5
CHUNK_SIZE = 300
TOP_K = 12
# Broad, whole-doc question so AttentionRAG retains the substantive passages.
QUESTION = (
    "What is Winnow, how does its two-axis (token-space and model-space) "
    "compression pipeline work, and how are the LLMLingua and AttentionRAG "
    "compressors merged?"
)


def count_tokens(text, tok):
    if not text:
        return 0
    if tok is not None:
        return len(tok.encode(text, add_special_tokens=False))
    return len(text.split())


def main():
    with open(SRC) as f:
        context = f.read()

    print(f"[src] DEVPOST.md: {len(context)} chars, {len(context.split())} words", flush=True)

    print("[modal] connecting to deployed classes...", flush=True)
    Compressor = modal.Cls.from_name("llmlingua2-xlm", "Compressor")
    AttnService = modal.Cls.from_name("attentionrag", "AttentionRAGService")
    compressor = Compressor()
    attn_service = AttnService()

    # (a) LLMLingua-2 (canonical spine).
    print("[llmlingua] compressing...", flush=True)
    llm_out = compressor.compress.remote(context, rate=RATE, return_labels=True)
    lingua_prompt = llm_out["compressed_prompt"]
    word_labels = llm_out.get("fn_labeled_original_prompt") or llm_out.get("word_labels")
    if not word_labels:
        raise RuntimeError("LLMLingua returned no labels")

    # (b) AttentionRAG (fall back to lingua-only on failure).
    print("[attentionrag] compressing spans...", flush=True)
    try:
        attn_out = attn_service.compress_spans.remote(
            context, QUESTION, chunk_size=CHUNK_SIZE, top_k=TOP_K, use_openai_hint=False
        )
        kept_spans = attn_out.get("kept_spans", [])
        single_chunk = attn_out.get("n_chunks", 0) == 1
        attn_empty = attn_out.get("is_empty_prefix", False) and not single_chunk
        print(
            f"[attentionrag] kept {attn_out.get('n_kept_chunks', 0)}/{attn_out.get('n_chunks', 0)} chunks, "
            f"{len(kept_spans)} spans, is_empty_prefix={attn_out.get('is_empty_prefix')}, "
            f"single_chunk={single_chunk}, hint={attn_out.get('hint_prefix')!r}",
            flush=True,
        )
    except Exception as exc:
        print(f"[attentionrag] FAILED -> lingua-only fallback: {exc}", flush=True)
        kept_spans, attn_empty = [], True

    # (c) UNION merge (exactly as server.py /compress does).
    union = merge_compress(context, word_labels, kept_spans, mode="union", attnrag_empty=attn_empty)

    # splice overshoot guard (same as run_compress.py).
    def merged_prompt(m):
        p = m["compressed_prompt"]
        if len(p) > len(context):
            print(f"[warn] splice overshoot ({len(p)} > {len(context)}); rebuilding from labels", flush=True)
            return " ".join(w for w, k in m["word_labels"] if k)
        return p

    union_prompt = merged_prompt(union)

    # token counts
    try:
        from transformers import AutoTokenizer

        from model_guard import guarded_from_pretrained

        tok = guarded_from_pretrained(
            AutoTokenizer, "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
        )
    except Exception as exc:
        print(f"[tok] whitespace fallback ({exc})", flush=True)
        tok = None

    o = count_tokens(context, tok)
    cu = count_tokens(union_prompt, tok)
    cl = count_tokens(lingua_prompt, tok)

    with open(OUT, "w") as f:
        f.write(union_prompt)
        if not union_prompt.endswith("\n"):
            f.write("\n")

    print("\n===== DEVPOST union compression =====", flush=True)
    print(f"question: {QUESTION}", flush=True)
    print(f"used_llmlingua_fallback: {union['used_llmlingua_fallback']}", flush=True)
    print(f"words total/kept: {union['n_words']}/{union['n_kept']}", flush=True)
    print(f"origin tokens:        {o}", flush=True)
    print(f"lingua-only tokens:   {cl}  ({cl / o:.1%})", flush=True)
    print(f"UNION tokens:         {cu}  ({cu / o:.1%})", flush=True)
    print(f"wrote: {OUT}", flush=True)


if __name__ == "__main__":
    main()
