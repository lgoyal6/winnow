"""
LLMLingua-2 (XLM-RoBERTa-large) on Modal.

First step toward a custom compression model: get the released checkpoint
running on a GPU, with weights baked into the image so cold starts are fast.

Usage:
    pip install modal
    python -m modal setup           # one-time auth
    modal run llmlingua2_modal.py                       # runs the demo
    modal run llmlingua2_modal.py --rate 0.33           # keep ~33% of tokens
    modal run llmlingua2_modal.py --text "your text..." --rate 0.5
"""

import modal

# Token-level extractive compressor (encoder). Multilingual XLM-RoBERTa.
MODEL_NAME = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
# Coarse-stage, question-aware reranker (cross-encoder, NOT a causal LM).
# bge-reranker-v2-m3: current lightweight multilingual BGE reranker - pairs well
# with the multilingual compressor above. See two_stage_compressor.py for why.
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"

# Persistent HF cache. Weights are downloaded into this volume once and read
# from it on every subsequent run - never re-downloaded, even across image
# rebuilds. create_if_missing=True means the first run provisions it automatically.
CACHE_DIR = "/cache"
hf_cache_vol = modal.Volume.from_name("llmlingua2-hf-cache", create_if_missing=True)


image = (
    modal.Image.debian_slim(python_version="3.11")
    # Pin llmlingua: argument names (rate vs ratio, the context-filter flags) have
    # shifted across releases; 0.2.2 is the version this code was verified against.
    .pip_install(
        "llmlingua==0.2.2", "torch", "transformers", "huggingface_hub", "hf_transfer"
    )
    # Point Hugging Face at the mounted volume so snapshot_download and the models
    # both read/write the standard HF cache layout there.
    .env({"HF_HOME": CACHE_DIR, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # Ship our local two-stage logic and the artifact guard into the image.
    .add_local_python_source("two_stage_compressor", "model_artifacts", "model_guard")
)

app = modal.App("llmlingua2-xlm", image=image)


@app.cls(
    gpu="A100",  # A100 for faster compression (only affects cold-start cost)
    volumes={CACHE_DIR: hf_cache_vol},
    # Keep a warmed container alive 30 min after the last request so it stays hot
    # through a demo (between questions) without re-warming. Auto-scales to zero
    # afterward - no lingering cost.
    scaledown_window=1800,
    # Memory snapshots: capture the fully-loaded model (incl. GPU memory) so that
    # future cold starts RESTORE that state instead of re-loading the model.
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},  # alpha: snapshot GPU memory too
)
class Compressor:
    @modal.enter(snap=True)
    def load(self):
        # Runs only when CREATING the snapshot - i.e. the very first cold start,
        # or after a code/image change invalidates the existing snapshot. The
        # loaded model and its GPU memory are captured here; every later cold
        # start restores this state directly and skips all of this work.
        #
        # Populate-once-then-read: downloads only if the volume is empty,
        # otherwise this resolves straight from the cached volume.
        from model_guard import (
            activate_loaded_model,
            llmlingua_model_config,
            verified_snapshot_download,
        )

        # Pinned revisions: without one this resolves `main`, so the same model
        # name can mean different bytes on a volume that is never re-downloaded.
        snapshots = {
            name: verified_snapshot_download(name)
            for name in (MODEL_NAME, RERANKER_NAME)
        }
        hf_cache_vol.commit()  # persist any newly downloaded files to the volume

        from llmlingua import PromptCompressor

        candidate = PromptCompressor(
            model_name=snapshots[MODEL_NAME],
            use_llmlingua2=True,
            device_map="cuda",
            # llmlingua 0.2.2 defaults trust_remote_code to True and forwards it
            # into AutoConfig/AutoTokenizer/AutoModel. See model_guard.
            model_config=llmlingua_model_config(MODEL_NAME, local_files_only=True),
        )
        self.compressor = activate_loaded_model(
            snapshots[MODEL_NAME],
            candidate,
            tensor_owner=candidate.model,
            activation_key=(MODEL_NAME, "llmlingua"),
        )

        # Coarse-stage reranker (raw transformers cross-encoder; see module docs
        # for why we don't use FlagEmbedding). fp16 on the GPU for speed.
        from two_stage_compressor import CrossEncoderReranker

        self.reranker = CrossEncoderReranker(RERANKER_NAME, device="cuda", use_fp16=True)
        hf_cache_vol.commit()

    @modal.method()
    def compress(self, text: str, rate: float = 0.5, return_labels: bool = False):
        # Single-text, token-level only path (no coarse stage).
        return self.compressor.compress_prompt(
            text,
            rate=rate,
            force_tokens=["\n", ".", "!", "?", ","],
            force_reserve_digit=True,      # don't drop digits (numbers stay intact)
            drop_consecutive=True,
            return_word_label=return_labels,  # per-word keep/discard labels
        )

    @modal.method()
    def compress_rag(
        self,
        instruction: str,
        question: str,
        documents: list[str],
        rate: float = 0.5,
        target_token: int = -1,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ):
        # Two-stage: reranker coarse selection + reorder, then LLMLingua-2 tokens.
        from two_stage_compressor import two_stage_compress

        return two_stage_compress(
            self.reranker,
            self.compressor,
            instruction,
            question,
            documents,
            rate=rate,
            target_token=target_token,
            top_k=top_k,
            score_threshold=score_threshold,
        )


SAMPLE = (
    "The committee convened on Tuesday to review the proposed budget allocations "
    "for the upcoming fiscal year. After a lengthy discussion that touched on "
    "several departmental requests, the members agreed to defer the final vote "
    "until additional documentation could be gathered from the finance office, "
    "which had not yet submitted its quarterly projections."
)


@app.local_entrypoint()
def main(text: str = SAMPLE, rate: float = 0.5, labels: bool = False):
    out = Compressor().compress.remote(text, rate=rate, return_labels=labels)
    print("\n=== compressed ===")
    print(out["compressed_prompt"])
    print("\n=== stats ===")
    print(f"origin tokens:     {out['origin_tokens']}")
    print(f"compressed tokens: {out['compressed_tokens']}")
    print(f"ratio:             {out['ratio']}")
    print(f"rate (kept):       {out['rate']}")


# --- Runnable two-stage example with dummy retrieved documents ---------------
# A RAG-style mix: some docs are relevant to the question, some are distractors.
# The reranker should keep the Paris docs and drop the biology ones.
DEMO_DOCS = [
    "The Eiffel Tower, completed in 1889 for the World's Fair, stands 330 metres "
    "tall on the Champ de Mars in Paris and draws about 7 million visitors a year.",
    "Photosynthesis is the process by which green plants convert sunlight, water, "
    "and carbon dioxide into glucose and oxygen inside their chloroplasts.",
    "Paris, the capital of France, is home to the Louvre, the Notre-Dame cathedral, "
    "and the Musee d'Orsay, all within walking distance of the Seine.",
    "The mitochondrion is often called the powerhouse of the cell because it "
    "generates most of the cell's supply of ATP through aerobic respiration.",
    "France's high-speed TGV trains depart from Gare de Lyon in Paris and reach "
    "Lyon in about two hours, making day trips from the capital practical.",
]


@app.local_entrypoint()
def demo_rag(
    question: str = "What are the main things to see in Paris, France?",
    instruction: str = "Answer the question using only the provided context.",
    rate: float = 0.5,
    top_k: int = 3,
):
    out = Compressor().compress_rag.remote(
        instruction, question, DEMO_DOCS, rate=rate, top_k=top_k
    )
    print("\n=== compressed prompt ===")
    print(out["compressed_prompt"])
    print("\n=== stats ===")
    print(f"documents kept:  {out['kept_documents']}/{out['total_documents']}")
    print(f"reranker scores: {[round(s, 3) for s in out['reranker_scores']]}")
    print(
        f"context tokens:  {out['context_origin_tokens']} -> "
        f"{out['context_compressed_tokens']}"
    )
    print(
        f"total tokens:    {out['origin_tokens']} -> {out['compressed_tokens']} "
        f"(kept rate {out['rate']:.2f})"
    )
