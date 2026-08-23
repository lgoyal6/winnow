# Winnow benchmark - shared contract (all subagents read this)

Goal: measure **accuracy vs compression** for 7 methods on ONE small shared
benchmark, scored ONE deterministic way. Time-boxed (~30 min of GPU). Keep
samples SMALL. Work inside `experiments/bench/`.

## The 7 arms
Token-space (compress text, then a fixed Qwen2.5-14B reader answers, fp16 KV):
- `lingua`         - LLMLingua-2 only
- `union`          - LLMLingua-2 ∪ AttentionRAG (merge_compress mode="union")
- `intersection`   - LLMLingua-2 ∩ AttentionRAG (merge_compress mode="intersection")
Model-space (full context fed straight to the model):
- `vanilla_llm`    - Qwen2.5-14B-Instruct, fp16 KV
- `llm_tq`         - Qwen2.5-14B-Instruct + TurboQuant 4-bit KV
- `vanilla_lclm`   - LCLM 16x (context→latents), fp16 decoder KV
- `lclm_tq`        - LCLM 16x + TurboQuant 4-bit decoder KV

## Shared QA prompt (use VERBATIM so arms are comparable)
INSTRUCTION = "Answer the question using ONLY the provided context. Answer in as few words as possible. If the answer is not present, respond with exactly: UNKNOWN."
- LLM (Qwen) full / compressed: prompt = f"{INSTRUCTION}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
- LCLM: context goes inside <|memory_start|>...<|memory_end|>, question verbatim after. The worker's _build_lclm_prompt already does this - pass context=<ctx>, prompt=f"{INSTRUCTION}\n\nQuestion: {question}\nAnswer:".
max_new_tokens = 48, do_sample=False (greedy), temperature 0.

## data.json  (produced by data subagent; everyone reads this)
{
  "config": {"source":"LongBench","tasks":[...],"n_per_task":N,"max_context_words":M},
  "examples": [
    {"id":"hotpotqa-0","task":"hotpotqa","context":"<truncated context>","question":"...","answers":["gold1","gold2",...]}
  ]
}

## *_answers.json  (one per arm-group)
{"arm_group":"llm|lclm|...","meta":{...},
 "answers":[{"id":"hotpotqa-0","arm":"vanilla_llm","answer":"<model text>","extra":{"kv_compression_x":..,"eff_bits":..,"input_tokens":..,"latent_tokens":..}}]}

## compressed.json  (token-space subagent)
{"examples":[{"id":"hotpotqa-0",
  "arms":{
    "lingua":      {"prompt":"...","origin_tokens":O,"compressed_tokens":C,"retention":C/O},
    "union":       {"prompt":"...","origin_tokens":O,"compressed_tokens":C,"retention":C/O},
    "intersection":{"prompt":"...","origin_tokens":O,"compressed_tokens":C,"retention":C/O}
  }}]}
Token counts: use the LLMLingua xlm-roberta tokenizer for BOTH origin and compressed
(consistent). Origin = the example's full context.

## Deployed apps to REUSE (already warm; do NOT redeploy these from scratch)
- LLMLingua-2:  modal.Cls.from_name("llmlingua2-compressor" or actual name, "Compressor") - confirm exact app name via `modal app list`. Method: .compress(text, rate=0.5, return_labels=True) -> dict with fn_labeled_original_prompt/word_labels, origin_tokens, compressed_tokens, rate, ratio.
- AttentionRAG: modal.Cls.from_name("attentionrag","AttentionRAGService").compress_spans(text, question, chunk_size=300, top_k=12, use_openai_hint=False) -> {kept_spans, n_chunks, n_kept_chunks, is_empty_prefix, hint_prefix}.
- Merge:        from token_merge import merge_compress (repo root). See server.py /compress for exact wiring (handles single_chunk + attn_empty fallback).
- LCLM:         app "lclm-tq-timing", classes LCLMVanilla / LCLMTurboQuant (A100). Need a QA method added (see LCLM subagent task).

rate for LLMLingua = 0.5 (keep ~50% tokens). chunk_size=300, top_k=12 for AttentionRAG.
