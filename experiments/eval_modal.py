"""
Eval harness (compression half) for: does MMR dedup-before-compression retain
more meaning at an EQUAL token budget than the baseline?

Runs two arms on the same document at the same target_token:
  A (baseline): chunk -> BGE reranker (intent-aware reorder) -> LLMLingua-2 tokens
  B (+MMR):     MMR sentence dedup -> [same as A]

Writes both compressed contexts + stats to winnow/eval_outputs.json. The Claude
answering + deterministic scoring lives in score_eval.py (pure-local, decoupled
so we don't re-pay GPU compression on every scoring tweak).

Run:
    modal run eval_modal.py                         # defaults
    modal run eval_modal.py --target-token 120 --redundancy-threshold 0.80
"""

import os
import sys

import modal

# This eval lives in experiments/ but imports two_stage_compressor.py from the
# repo root - put the root on the path so it resolves from either location.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_NAME = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"
EMBEDDER_NAME = "BAAI/bge-small-en-v1.5"  # MMR bi-encoder (small, English)
# Causal backbone for REAL LongLLMLingua (use_llmlingua2=False). Nous mirror is
# ungated (no HF license/token needed). ~13GB; needs a 24GB GPU alongside the
# encoders, hence A10G below.
LONGLLMLINGUA_MODEL = "NousResearch/Llama-2-7b-hf"

CACHE_DIR = "/cache"
hf_cache_vol = modal.Volume.from_name("llmlingua2-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "llmlingua==0.2.2", "torch", "transformers", "huggingface_hub",
        "hf_transfer", "numpy", "accelerate", "sentencepiece", "protobuf",
    )
    .env({"HF_HOME": CACHE_DIR, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("two_stage_compressor", "model_guard")
)

app = modal.App("winnow-eval", image=image)


def chunk_by_sentences(text: str, k: int = 3):
    """Group sentences into k-sentence chunks (identical logic for both arms)."""
    from two_stage_compressor import split_sentences

    sents = split_sentences(text)
    return [" ".join(sents[i : i + k]) for i in range(0, len(sents), k)] or [" "]


@app.cls(
    gpu="T4",  # encoder-only: LLMLingua-2 + BGE reranker + MMR embedder
    volumes={CACHE_DIR: hf_cache_vol},
    scaledown_window=600,
)
class Compressor:
    @modal.enter()
    def load(self):
        from model_guard import (assert_no_pickled_weights,
                                 llmlingua_model_config,
                                 pinned_snapshot_download)

        for name in (MODEL_NAME, RERANKER_NAME, EMBEDDER_NAME):
            assert_no_pickled_weights(pinned_snapshot_download(name))
        hf_cache_vol.commit()

        from llmlingua import PromptCompressor
        from two_stage_compressor import CrossEncoderReranker, SmallEmbedder

        # Encoder token classifier (LLMLingua-2). No causal/SLM backbone - the
        # LongLLMLingua path was cancelled (wrong regime for short single docs).
        self.compressor = PromptCompressor(
            model_name=MODEL_NAME, use_llmlingua2=True, device_map="cuda",
            # llmlingua 0.2.2 defaults trust_remote_code to True. See model_guard.
            model_config=llmlingua_model_config(MODEL_NAME),
        )
        self.reranker = CrossEncoderReranker(RERANKER_NAME, device="cuda", use_fp16=True)
        self.embedder = SmallEmbedder(EMBEDDER_NAME, device="cuda", use_fp16=True)

    @modal.method()
    def sentence_sims(self, text: str, intent: str, top: int = 18):
        """Diagnostic: top sentence pairs by cosine (to calibrate the threshold)."""
        import numpy as np
        from two_stage_compressor import split_sentences

        sents = split_sentences(text)
        emb = self.embedder.encode(sents)
        sims = emb @ emb.T
        pairs = []
        for i in range(len(sents)):
            for j in range(i + 1, len(sents)):
                pairs.append((float(sims[i, j]), i, j))
        pairs.sort(reverse=True)
        return {"sentences": sents,
                "top_pairs": [(round(s, 3), i, j) for s, i, j in pairs[:top]]}

    @modal.method()
    def compress_eval(
        self,
        text: str = "",
        intent: str = "",
        instruction: str = "",
        rate: float = 0.75,          # fraction of tokens KEPT (~retention)
        use_mmr: bool = False,
        redundancy_threshold: float = 0.80,
        chunk_size: int = 3,
        documents: list = None,      # multi-doc input (real passage units)
        top_k: int = None,           # keep top-k passages (reranker selection)
    ):
        from two_stage_compressor import (
            FORCE_TOKENS, mmr_dedupe_sentences, select_and_reorder,
        )

        # ---- assemble reranker units + optional Stage 0 MMR dedup ----
        mmr_dbg = None
        if documents is not None:
            # Multi-doc: dedup each passage in place (keeps passage boundaries
            # so the reranker still sees real documents).
            units = list(documents)
            if use_mmr:
                deduped, all_dropped, total_sents = [], [], 0
                for d in units:
                    dd, dbg = mmr_dedupe_sentences(
                        self.embedder, d, intent, redundancy_threshold=redundancy_threshold
                    )
                    deduped.append(dd)
                    all_dropped += dbg["dropped_sentences"]
                    total_sents += dbg["sentences"]
                units = deduped
                mmr_dbg = {"sentences": total_sents, "dropped": [None] * len(all_dropped),
                           "dropped_sentences": all_dropped}
            origin_text = "\n\n".join(documents)
        else:
            # Single doc: dedup whole text, then chunk into sentence windows.
            work_text = text
            if use_mmr:
                work_text, mmr_dbg = mmr_dedupe_sentences(
                    self.embedder, text, intent, redundancy_threshold=redundancy_threshold
                )
            units = chunk_by_sentences(work_text, k=chunk_size)
            origin_text = text

        # ---- coarse: reranker selection + reorder ----
        selected, coarse = select_and_reorder(self.reranker, intent, units, top_k=top_k)
        if not selected:
            selected = [" "]

        # ---- fine: LLMLingua-2 token compression ----
        fine = self.compressor.compress_prompt(
            selected, rate=rate,
            use_context_level_filter=False, use_token_level_filter=True,
            force_tokens=FORCE_TOKENS, force_reserve_digit=True, drop_consecutive=True,
        )
        compressed_context = fine["compressed_prompt"]

        # ---- uniform token accounting (xlm-r tokenizer) ----
        tok = self.compressor.tokenizer
        origin_tokens = len(tok.encode(origin_text, add_special_tokens=False))
        comp_tokens = len(tok.encode(compressed_context, add_special_tokens=False))

        return {
            "use_mmr": use_mmr,
            "compressed_context": compressed_context,
            "context_origin_tokens": origin_tokens,
            "context_compressed_tokens": comp_tokens,
            "retention": round(comp_tokens / max(origin_tokens, 1), 3),
            "mmr": mmr_dbg,
            "total_units": len(units),
            "kept_indices": coarse["kept_indices"],   # which passages survived
            "reranker_scores": coarse["scores"],
        }


# Driving / scoring is done from run_eval.py (deploy + client), which avoids the
# flaky `modal run` log-stream heartbeat. Deploy with: modal deploy eval_modal.py
