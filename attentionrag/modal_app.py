"""
AttentionRAG (arXiv:2503.10720) on Modal.

Runs the faithful pipeline end-to-end on a GPU with a real causal LM:
  hint prefix (B.1) -> chunk -> per-chunk anchor token + all-layer attention
  (Eq. 1/2) -> top-k token -> sentence selection (Eq. 3) -> answer.

Conventions (matching the rest of this repo + memory):
  * models are read from the shared `llmlingua2-hf-cache` volume, downloaded once
    if missing, then cached for every later run (never re-downloaded);
  * test with `modal run --detach` so the job survives the laptop sleeping.

Usage:
    pip install modal && python -m modal setup        # one-time
    modal run --detach attentionrag/modal_app.py                  # demo
    modal run --detach attentionrag/modal_app.py --question "..." --context-file ...
"""

import os

import modal

# Paper uses Qwen-2.5-7B-Instruct / Llama-3.1-8B-Instruct as the compression +
# generation model. Qwen is ungated (no HF token needed) -> default here.
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


# Faithful hint-prefix authoring uses GPT-4o Mini -> ship OPENAI_API_KEY into the
# container from the Modal-stored `openai-secret` (set up via `modal secret`).
openai_secret = modal.Secret.from_name("openai-secret")

CACHE_DIR = "/cache"
# Shared HF cache volume (per memory convention): models download here once if
# missing, then persist across runs and across projects. Qwen-2.5-7B is fetched
# on the first run and read from the volume on every run after.
hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.44",
        "accelerate",
        "huggingface_hub",
        "hf_transfer",
        "sentencepiece",
        "protobuf",
        "openai",
    )
    .env({"HF_HOME": CACHE_DIR, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # Ship the AttentionRAG package and the artifact guard into the image.
    .add_local_python_source("attentionrag", "model_artifacts", "model_guard")
)

app = modal.App("attentionrag", image=image)


@app.cls(
    gpu="A100",  # 40GB: more compute than A10G for the per-chunk eager-attention
    volumes={CACHE_DIR: hf_cache_vol},
    secrets=[openai_secret],  # OPENAI_API_KEY for GPT-4o-mini hint authoring
    scaledown_window=600,
    timeout=1800,
)
class AttentionRAGService:
    @modal.enter()
    def load(self):
        from model_guard import verified_snapshot_download

        # Populate-once-then-read: only downloads if the volume lacks the model,
        # and always at the pinned revision rather than whatever `main` is now.
        verified_snapshot_download(MODEL_NAME)
        hf_cache_vol.commit()

        from attentionrag.core import AttentionRAG
        from attentionrag.hf_backend import HFBackend

        self.backend = HFBackend(model_name=MODEL_NAME, device="cuda")
        self._AttentionRAG = AttentionRAG
        hf_cache_vol.commit()

    @modal.method()
    def run(
        self,
        context: str,
        question: str,
        chunk_size: int = 300,
        top_k: int = 12,
        hint_prefix: str = None,
        use_fixed_prefix: bool = False,
        use_openai_hint: bool = False,
        with_baseline: bool = True,
    ):
        import time

        # toggle the hint author (GPT-4o-mini vs local model) per call
        self.backend.use_openai_hint = use_openai_hint

        rag = self._AttentionRAG(
            self.backend,
            chunk_size=chunk_size,
            top_k=top_k,
            use_fixed_prefix=use_fixed_prefix,
        )

        # Warm every path once so no measured call pays one-time init:
        #   - focus_attention -> the 2-pass attention forward (CUDA kernels)
        #   - generate_answer -> the generate() path (used by the local hint too)
        #   - OpenAI client    -> first-call client init + TLS/connection setup
        # All excluded from the reported timings below.
        if not getattr(self, "_warmed", False):
            try:
                self.backend.focus_attention(
                    "Warm up sentence one. Warm up sentence two.", "warm?", "It is"
                )
            except Exception:
                pass
            try:
                self.backend.generate_answer("Warm up.", "warm?")
            except Exception:
                pass
            if os.environ.get("OPENAI_API_KEY"):
                try:
                    prev = self.backend.use_openai_hint
                    self.backend.use_openai_hint = True
                    self.backend.generate_hint_prefix("Where is Daniel?")
                    self.backend.use_openai_hint = prev
                except Exception:
                    pass
            self._warmed = True

        # --- timed: compression only (no model load, no warmup) -------------
        # (1) hint formulation (the GPT-4o-mini call when use_openai_hint=True)
        t_hint = 0.0
        hint = hint_prefix
        hint_source = "fixed" if use_fixed_prefix else "provided"
        if not use_fixed_prefix and hint_prefix is None:
            hint_source = "gpt-4o-mini" if use_openai_hint else "local-model"
            t0 = time.perf_counter()
            hint = self.backend.generate_hint_prefix(question)
            t_hint = time.perf_counter() - t0

        # (2) attention-guided pruning (chunk -> anchor+attention -> select)
        t1 = time.perf_counter()
        comp = rag.compress(context, question, hint_prefix=hint)
        t_attn = time.perf_counter() - t1

        compression_time = t_hint + t_attn
        # ---------------------------------------------------------------------

        answer = rag.answer(comp.compressed_context, question)

        orig_tokens = self.backend.count_tokens(context)
        comp_tokens = self.backend.count_tokens(comp.compressed_context)

        result = {
            "question": question,
            "hint_prefix": comp.hint_prefix,
            "hint_source": hint_source,
            "is_empty_prefix": comp.is_empty_prefix,
            "compressed_context": comp.compressed_context,
            "answer": answer,
            "n_chunks": comp.n_chunks,
            "n_kept_chunks": comp.n_kept_chunks,
            "origin_tokens": orig_tokens,
            "compressed_tokens": comp_tokens,
            "compression_ratio": round(orig_tokens / max(comp_tokens, 1), 2),
            "hint_time_s": round(t_hint, 3),
            "attention_time_s": round(t_attn, 3),
            "compression_time_s": round(compression_time, 3),
            "chunk_anchors": [
                {"i": c.index, "anchor": c.anchor, "skipped": c.skipped}
                for c in comp.chunks
            ],
        }
        if with_baseline:
            result["baseline_answer"] = rag.answer(context, question)
        return result

    @modal.method()
    def compress_spans(
        self,
        text: str,
        question: str,
        chunk_size: int = 300,
        top_k: int = 12,
        use_fixed_prefix: bool = False,
        use_openai_hint: bool = False,
    ):
        """Return kept ORIGINAL char-spans (for token-mask merging with another
        compressor). See HFBackend.compress_spans."""
        self.backend.use_openai_hint = use_openai_hint
        return self.backend.compress_spans(
            text,
            question,
            chunk_size=chunk_size,
            top_k=top_k,
            use_fixed_prefix=use_fixed_prefix,
        )


# --- demo data --------------------------------------------------------------
# A bAbI-style "needle in distractors" example (like BABILong) -- the answer
# sentence is buried among irrelevant ones.
BABI_CONTEXT = (
    "Mary moved to the bathroom. John went to the hallway. "
    "Daniel went back to the kitchen. Sandra journeyed to the garden. "
    "Daniel travelled to the park. John picked up the football there. "
    "Mary went back to the bedroom. Sandra grabbed the apple in the garden. "
    "John dropped the football. Daniel got the milk in the park. "
    "The weather was sunny and warm throughout the afternoon. "
    "A train passed by the station at noon carrying many passengers."
)
BABI_QUESTION = "Where is Daniel?"

# A multi-hop (HotpotQA-style) example.
HOTPOT_CONTEXT = (
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in "
    "Paris, France. It was designed by the engineer Gustave Eiffel. "
    "Gustave Eiffel was a French civil engineer born in Dijon in 1832. "
    "He also designed the internal frame of the Statue of Liberty. "
    "Photosynthesis converts sunlight into chemical energy in plants. "
    "The Seine is a river that flows through Paris. "
    "Dijon is the capital city of the Cote-d'Or department in eastern France, "
    "famous for its mustard. The Louvre is the world's most-visited museum."
)
HOTPOT_QUESTION = "In which city was the designer of the Eiffel Tower born?"


@app.local_entrypoint()
def main(
    question: str = "",
    context: str = "",
    context_file: str = "",
    chunk_size: int = 0,
    top_k: int = 0,
):
    svc = AttentionRAGService()

    if question and (context or context_file):
        if context_file:
            with open(context_file) as f:
                context = f.read()
        demos = [(question, context, chunk_size or 300, top_k or 12)]
    else:
        # Per the paper: small chunk / small k for short sparse (BABILong-style)
        # contexts; larger chunk so a multi-hop passage stays whole (HotpotQA).
        demos = [
            (BABI_QUESTION, BABI_CONTEXT, chunk_size or 300, top_k or 5),
            (HOTPOT_QUESTION, HOTPOT_CONTEXT, chunk_size or 300, top_k or 8),
        ]

    for q, ctx, cs, tk in demos:
        out = svc.run.remote(ctx, q, chunk_size=cs, top_k=tk, use_openai_hint=True)
        print("\n" + "=" * 70)
        print("QUESTION:        ", out["question"])
        print("PARAMS:          ", f"chunk_size={cs}, top_k={tk}")
        print("HINT PREFIX:     ", repr(out["hint_prefix"]),
              f"[via {out['hint_source']}]",
              "(empty)" if out["is_empty_prefix"] else "")
        print("CHUNKS:          ", f"{out['n_kept_chunks']}/{out['n_chunks']} kept")
        print("ANCHORS:         ",
              ", ".join(f"{a['anchor']}{'*' if a['skipped'] else ''}"
                        for a in out["chunk_anchors"]))
        print("TOKENS:          ",
              f"{out['origin_tokens']} -> {out['compressed_tokens']} "
              f"({out['compression_ratio']}x compression)")
        print("TIMING:          ",
              f"compression={out['compression_time_s']}s "
              f"(hint={out['hint_time_s']}s + attention={out['attention_time_s']}s)")
        print("-" * 70)
        print("COMPRESSED CONTEXT:")
        print(out["compressed_context"])
        print("-" * 70)
        print("ANSWER (compressed): ", out["answer"])
        if "baseline_answer" in out:
            print("ANSWER (full ctx):   ", out["baseline_answer"])
    print("\n" + "=" * 70)


@app.local_entrypoint()
def compare():
    """Time the hint formulation with the LOCAL Qwen model vs GPT-4o-mini on the
    same warmed container, and break down hint vs attention-pruning time.

    Each demo is run twice (local hint, then gpt hint); the attention/selection
    stage is identical (same Qwen model), so the difference is purely the hint
    author -- on-GPU generate() vs a GPT-4o-mini network round-trip.
    """
    svc = AttentionRAGService()
    demos = [
        (BABI_QUESTION, BABI_CONTEXT, 300, 5),
        (HOTPOT_QUESTION, HOTPOT_CONTEXT, 300, 8),
    ]

    # One throwaway call to trigger the in-container warmup (CUDA + OpenAI client)
    # so neither of the two measured runs pays one-time init.
    svc.run.remote(BABI_CONTEXT, BABI_QUESTION, chunk_size=300, top_k=5,
                   use_openai_hint=False, with_baseline=False)

    rows = []
    for q, ctx, cs, tk in demos:
        local = svc.run.remote(ctx, q, chunk_size=cs, top_k=tk,
                               use_openai_hint=False, with_baseline=False)
        gpt = svc.run.remote(ctx, q, chunk_size=cs, top_k=tk,
                             use_openai_hint=True, with_baseline=False)
        rows.append((q, local, gpt))

        print("\n" + "=" * 74)
        print("QUESTION:", q)
        print(f"  context tokens: {local['origin_tokens']} -> "
              f"{local['compressed_tokens']} ({local['compression_ratio']}x)")
        print(f"  {'source':<14}{'hint':>10}{'attention':>12}{'total comp':>13}"
              f"   hint prefix")
        for label, r in (("local Qwen", local), ("gpt-4o-mini", gpt)):
            print(f"  {label:<14}{r['hint_time_s']:>9.3f}s"
                  f"{r['attention_time_s']:>11.3f}s"
                  f"{r['compression_time_s']:>12.3f}s"
                  f"   {r['hint_prefix']!r} [{r['hint_source']}]")
        d_hint = gpt["hint_time_s"] - local["hint_time_s"]
        d_total = gpt["compression_time_s"] - local["compression_time_s"]
        print(f"  delta (gpt - local): hint {d_hint:+.3f}s   total {d_total:+.3f}s")
        print(f"  answer (local hint): {local['answer']!r}")
        print(f"  answer (gpt hint):   {gpt['answer']!r}")

    # aggregate
    print("\n" + "=" * 74)
    print("SUMMARY (averages across demos)")
    n = len(rows)
    avg = lambda key, idx: sum(r[idx][key] for r in rows) / n
    print(f"  local Qwen hint:  {avg('hint_time_s', 1):.3f}s")
    print(f"  gpt-4o-mini hint: {avg('hint_time_s', 2):.3f}s")
    print(f"  attention prune:  {avg('attention_time_s', 1):.3f}s (same model both)")
    print("=" * 74)


@app.local_entrypoint()
def scaling():
    """Measure how compression time scales with the number of INPUT tokens.

    Hint formulation is a fixed one-shot cost (independent of context length);
    the attention-guided pruning runs once per chunk, so it grows with input
    size. We sweep increasing contexts (fixed chunk_size=300) and report
    attention time per 1k input tokens. Local Qwen hint is used to avoid network
    jitter. The filler keeps every chunk on-topic so both attention passes run
    (a chunk whose anchor is 'none' is cheaper -- pass 1 only).
    """
    svc = AttentionRAGService()
    locs = ["kitchen", "garden", "park", "office", "bedroom", "hallway",
            "bathroom", "cinema", "market", "library", "stadium", "cafe"]
    question = "Where is Daniel?"

    def make_context(n_sentences):
        return " ".join(
            f"Daniel walked into the {locs[i % len(locs)]} during the afternoon."
            for i in range(n_sentences)
        )

    # warmup container (CUDA + client); not measured
    svc.run.remote(make_context(20), question, chunk_size=300, top_k=12,
                   use_openai_hint=False, with_baseline=False)

    print("\n" + "=" * 84)
    print(f"{'in_tokens':>10}{'chunks':>8}{'kept':>6}{'hint_s':>9}"
          f"{'attn_s':>9}{'total_s':>9}{'attn ms/tok':>13}{'ms/chunk':>10}")
    print("-" * 84)
    for ns in [10, 30, 70, 150, 300, 600]:
        ctx = make_context(ns)
        out = svc.run.remote(ctx, question, chunk_size=300, top_k=12,
                             use_openai_hint=False, with_baseline=False)
        tok = out["origin_tokens"]
        kept = out["n_kept_chunks"]
        attn = out["attention_time_s"]
        ms_per_tok = attn * 1000 / max(tok, 1)
        ms_per_chunk = attn * 1000 / max(kept, 1)
        print(f"{tok:>10}{out['n_chunks']:>8}{kept:>6}"
              f"{out['hint_time_s']:>9.3f}{attn:>9.3f}"
              f"{out['compression_time_s']:>9.3f}{ms_per_tok:>13.3f}{ms_per_chunk:>10.1f}")
    print("=" * 84)
    print("hint time is ~constant; attention time scales ~linearly with input "
          "tokens (one forward pass per chunk).")
