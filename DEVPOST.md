# Winnow - multi-path prompt compression for LLM pipelines

## Inspiration

Tokens are the meter on every LLM bill, and almost all of them are waste.
Spoken language, retrieved RAG context, and long documents are massively
redundant - but the redundancy lives in different places. Some is *low-information
tokens* ("um, so basically the idea is…"), some is *irrelevant passages* that a
query never touches, and some is the sheer *bit-width of the KV cache* the model
carries while it generates. No single paper kills all three.

So instead of picking one compression technique and hoping, we built Winnow: a
pipeline that runs several state-of-the-art compressors **in parallel**, merges
their decisions, and then - optionally - hands the result to an LLM whose KV
cache is itself compressed. Every stage is independently justified by a recent
paper; the contribution is making them *compose*.

## What it does

We tried our best to include every strong, **parallelizable and patchable**
compression method we could find - and where no public repo existed, we
re-implemented the paper ourselves from scratch (realistically, only LLMLingua
shipped usable code). The papers Winnow builds on:

- **LLMLingua-2** - Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression - [arXiv:2403.12968](https://arxiv.org/abs/2403.12968) · [repo](https://github.com/microsoft/LLMLingua)
- **LongLLMLingua** - Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression - [arXiv:2310.06839](https://arxiv.org/abs/2310.06839) · [repo](https://github.com/microsoft/LLMLingua)
- **AttentionRAG** - Attention-Guided Context Pruning in Retrieval-Augmented Generation - [arXiv:2503.10720](https://arxiv.org/abs/2503.10720) *(no public repo - self-implemented)*
- **LCLM** - End-to-End Context Compression at Scale - [arXiv:2606.09659](https://arxiv.org/abs/2606.09659) · [repo](https://github.com/LeonLixyz/LCLM)
- **TurboQuant** - Online Vector Quantization with Near-Optimal Distortion Rate (KV-cache compression) - [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) · [reference impl](https://github.com/OmarHory/turboquant)
- **CompactPrompt** - A Unified Pipeline for Prompt & Data Compression in LLM Workflows - [arXiv:2510.18043](https://arxiv.org/abs/2510.18043) *(explored; no public dataset to reproduce its style metric)*

Winnow compresses a prompt along two orthogonal axes and lets the user choose how
far to push each one.

**1. Token-space compression (which words survive).**
Given a piece of text, Winnow fans it out to two independent compressor families
at once:

- **LLMLingua-2 + LongLLMLingua.** LLMLingua-2 is an extractive token classifier
  (a fine-tuned XLM-RoBERTa that labels each token keep/drop). LongLLMLingua adds
  the query-aware arm: a BGE cross-encoder reranker does coarse, question-aware
  document selection and reordering (to fight "lost-in-the-middle"), then the
  causal perplexity path compresses the survivors conditioned on the question.
- **AttentionRAG.** A faithful implementation of *Attention-Guided Context Pruning
  in RAG* (arXiv:2503.10720). It reformulates the query into an incomplete-answer
  template with a single "focal" blank, runs next-token prediction over each
  context chunk, and uses the focal token's summed-over-all-layers attention to
  rank tokens - keeping the sentences that hold the top-k. Chunks the model
  answers `none` on are dropped entirely.

Both arms produce per-token keep decisions. Winnow then **merges them token-by-token
over the original text**, preserving chronology, under a boolean rule the user
picks: **intersection** (keep a word only if *both* methods kept it - maximally
aggressive) or **union** (keep if *either* did - safer, recall-oriented). The
merge is a true alignment to one canonical token sequence (LLMLingua-2's word
labels), not a fragile string-membership test, with a fallback to LLMLingua-only
if AttentionRAG gates everything out so we never wipe the text.

**2. Model-space compression (how the answer is generated).**
Once Winnow has the compressed string, the user chooses where it goes:

- **Black box.** Send the compressed prompt straight to a hosted API - Anthropic
  Claude or OpenAI GPT, picked per request. The savings are immediate and
  provider-agnostic, because the compressed prompt is just text.
- **Self-hosted + TurboQuant.** Route to our own Qwen model on Modal, generating
  with a **TurboQuant**-compressed KV cache (Google Research, ICLR 2026,
  arXiv:2504.19874) - a data-free random-rotation quantizer that drops the KV
  cache to ~4 bits (3–4× smaller) as a `DynamicCache` subclass, validated on
  Mistral-7B and Qwen2.5-14B.
- **Self-hosted + LCLM + TurboQuant.** Route to **LCLM** (*End-to-End Context
  Compression at Scale*, latent-context's released encoder→adapter→decoder
  checkpoints, ~2 weeks old at build time). LCLM compresses the long context into
  a handful of latent **soft tokens** that the decoder consumes as input
  embeddings - shrinking the KV cache's *sequence length*. TurboQuant then
  compresses the *bits per entry* of that same decoder cache. The two are
  orthogonal, so **the savings multiply**: fewer KV entries × fewer bits each.

The net effect: token-level pruning before the model, sequence-length and
bit-width compression inside it - three independent papers stacked end to end.

## How we built it

- **Compression workers on Modal GPUs.** LLMLingua-2/LongLLMLingua, AttentionRAG,
  TurboQuant, and LCLM+TurboQuant each run as a warm Modal A100 worker, tied to a
  FastAPI server's lifecycle so models load once at startup and real requests pay
  no cold-start cost. `/compress` fans LLMLingua and AttentionRAG out concurrently
  and merges; `/generate` routes between the Qwen-TurboQuant and LCLM-TurboQuant
  workers on an `lclm` flag.
- **The merge engine** (`token_merge.py`) is pure Python: it normalizes
  LLMLingua's labels, reconstructs each word's char-span in the original by an
  in-order forward scan, tests AttentionRAG's kept sentence-spans by char overlap,
  applies the union/intersection rule, and splices survivors back together
  preserving original spacing.
- **TurboQuant** is a drop-in `DynamicCache` subclass - Lloyd-Max Gaussian
  quantization with an optional outlier-channel path - so it slots straight in as
  any HF model's `past_key_values`, including the LCLM decoder.
- **Frontend** is a Next.js app that shows the raw vs. compressed diff live, token
  counts, % and $ saved, and an A/B Q&A box that asks the same question against
  both prompts to prove meaning survived.

## Challenges we ran into

- **Pairing TurboQuant with LCLM.** Getting TurboQuant's low-bit (down to ~3.5-bit
  outlier) quantization to drop cleanly into LCLM's decoder was the hardest
  integration. LCLM feeds the decoder `inputs_embeds` (the soft tokens), not token
  ids, and we had to confirm our `TQCache` survives that path and the encoder's
  single prefill without corrupting the latent memory block.
- **Inventing the merge algorithm.** Unionizing/intersecting two *different*
  compressors is not a set operation on strings - the two methods segment text
  differently. We had to align both to one canonical token sequence and operate on
  char-spans so the merge is order-preserving and unambiguous even with repeated
  words, plus a fallback so an empty AttentionRAG result never erases everything.
- **Tuning LongLLMLingua.** Choosing the right lead anchors / causal backbone and
  reranker for the question-aware arm took real iteration - what to condition on,
  how to reorder context against position bias, and where the causal path actually
  beats the cheaper extractive classifier.
- **CompactPrompt had no public data.** We tried to implement CompactPrompt's
  style-metric approach, but the paper shipped no public datasets to fit/evaluate
  the metric against, so we couldn't reproduce it faithfully and left it out rather
  than ship something unvalidated.

## Accomplishments we're proud of

- A working **two-axis** compressor: token-space (LLMLingua ∪/∩ AttentionRAG) and
  model-space (LCLM × TurboQuant) stacked in one pipeline.
- A genuinely novel **merge between two heterogeneous compression algorithms** that
  stays order-preserving and never destroys the text.
- TurboQuant validated end-to-end (3–4× KV compression, output matching FP16) and
  composing with LCLM so the KV savings multiply.

## What we learned

- The big wins come from compressing along *different* axes and stacking them  - 
  token count, sequence length, and bit-width are independent levers.
- Heterogeneous methods need a common canonical representation before you can
  combine them; alignment, not set math, is the real work.
- Faithfully reproducing a paper is gated by its artifacts: no public dataset
  (CompactPrompt) effectively means no faithful reimplementation.

## What's next

- Learned (rather than user-picked) routing between intersection/union and between
  black-box vs. self-hosted, per prompt and budget.
- Custom CUDA kernels for TurboQuant to turn the memory win into a latency win.
- Revisiting CompactPrompt if/when evaluation data becomes available.

## Built with

Modal · FastAPI · Next.js · LLMLingua-2 · LongLLMLingua · AttentionRAG
(arXiv:2503.10720) · LCLM (End-to-End Context Compression at Scale) · TurboQuant
(arXiv:2504.19874) · BGE rerankers · Qwen · Anthropic Claude · OpenAI GPT ·
Deepgram
