"""
Two-stage, question-aware prompt compressor for RAG.

WHY THIS SHAPE (the key finding)
--------------------------------
The brief asked whether LLMLingua-2's PromptCompressor can do reranker-based,
question-aware COARSE document selection inside a single `compress_prompt` call
(`use_llmlingua2=True`). Reading the llmlingua 0.2.2 source settles it: it cannot.

`compress_prompt(...)` short-circuits at the top:

    if self.use_llmlingua2:
        return self.compress_prompt_llmlingua2(
            context, rate, target_token, use_context_level_filter,
            target_context, context_level_rate, context_level_target_token, ...)

It forwards ONLY those args - `question`, `instruction`, `rank_method`, and
`reorder_context` are silently dropped. And `compress_prompt_llmlingua2(...)` has
NO `question`/`rank_method`/`reorder_context` parameters at all. So with
`use_llmlingua2=True`:

  * the only document-level knob (`use_context_level_filter` + `target_context`/
    `context_level_rate`/`context_level_target_token`) ranks documents by the
    encoder's own predicted compression score - it is QUESTION-BLIND;
  * `rank_method` (including `bge_reranker`) only takes effect on the causal
    LongLLMLingua path (`use_llmlingua2=False`), which loads a causal backbone we
    are explicitly avoiding.

ARCHITECTURE CHOSEN: explicit two-step pipeline (no causal model anywhere)
-------------------------------------------------------------------------
  1. COARSE (question-aware): a BGE cross-encoder reranker scores each document
     against the question; keep the top-k by budget; reorder survivors
     most-relevant-first to fight position bias (LongLLMLingua §4.2 idea).
  2. FINE (token-level): the surviving documents (as a LIST, one element per
     chunk) go to LLMLingua-2's extractive token classifier for token-level
     compression at the target rate / token budget. The instruction and question
     are preserved VERBATIM (short, sensitive, task-defining) and reattached
     around the compressed context.

VERSIONS / MODELS (verify against the installed signature, not just the docs)
----------------------------------------------------------------------------
  * llmlingua == 0.2.2
  * compressor: microsoft/llmlingua-2-xlm-roberta-large-meetingbank (encoder)
  * reranker:   BAAI/bge-reranker-v2-m3 via FlagEmbedding.FlagReranker
        chosen over the library's hardcoded bge-reranker-large: v2-m3 is the
        current (2025) lightweight BGE reranker, multilingual (matches the
        multilingual XLM-RoBERTa compressor), fast, and supersedes -large.
  * rank_method: NOT USED. The library's `rank_method` only works on the causal
    path; we do the reranking ourselves so the token stage stays pure LLMLingua-2.

This module is framework-agnostic: it takes ALREADY-LOADED model objects, so the
heavy models load once (e.g. on a warm GPU) and these functions stay cheap.
"""

import re
from typing import List, Optional

from model_guard import guarded_from_pretrained

# Token-level structural tokens LLMLingua-2 should never drop.
FORCE_TOKENS = ["\n", ".", "!", "?", ","]


# --------------------------------------------------------------------------
# Stage 0 (optional): MMR-style intent-aware semantic redundancy removal.
#
# Spoken transcripts restate the same MEANING many ways ("the idea is..." /
# "basically the concept is..."). LLMLingua-2 prunes low-info TOKENS but does
# not collapse repeated meaning. This stage drops near-duplicate SENTENCES
# before token compression, so the token budget is spent on distinct content.
#
# It is intent-aware in the MMR spirit (Carbonell & Goldstein 1998): sentences
# are visited most-relevant-to-the-intent first, and a sentence is kept only if
# it is not too similar (cosine >= redundancy_threshold) to anything already
# kept. Survivors are returned in original order for readability. Training-free.
# --------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter (no nltk download needed for this stage)."""
    sents: List[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        for s in _SENT_SPLIT_RE.split(para):
            s = s.strip()
            if s:
                sents.append(s)
    return sents


class SmallEmbedder:
    """Tiny BGE bi-encoder via raw transformers (CLS pooling + L2 normalize).

    Deliberately avoids sentence-transformers to keep the Modal image on the
    exact transformers/torch stack llmlingua already pins (mirrors how
    CrossEncoderReranker avoids FlagEmbedding). Returns a numpy array.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5",
                 device: str = "cuda", max_length: int = 256, use_fp16: bool = True):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.device = device
        self.max_length = max_length
        # Pinned revision + safetensors + no remote code; see model_guard.
        self.tokenizer = guarded_from_pretrained(AutoTokenizer, model_name)
        model = guarded_from_pretrained(AutoModel, model_name)
        if use_fp16 and device.startswith("cuda"):
            model = model.half()
        self.model = model.eval().to(device)

    def encode(self, texts: List[str], batch_size: int = 64):
        import numpy as np

        torch = self._torch
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True,
                return_tensors="pt", max_length=self.max_length,
            ).to(self.device)
            with torch.no_grad():
                hidden = self.model(**inputs, return_dict=True).last_hidden_state
            cls = hidden[:, 0].float()  # BGE uses the [CLS] token
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            out.append(cls.cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 1))


def mmr_dedupe_sentences(
    embedder,
    text: str,
    intent: str,
    redundancy_threshold: float = 0.82,
):
    """Stage 0: remove semantically redundant sentences, intent-aware.

    Args:
        embedder: a SmallEmbedder (or anything with .encode(list)->np.ndarray).
        text: the raw (rambly) transcript / narration.
        intent: the user goal; sentences are visited most-relevant-first.
        redundancy_threshold: cosine >= this vs. an already-kept sentence => drop
            as a near-duplicate. Lower = more aggressive dedup.

    Returns:
        (deduped_text, debug) where deduped_text keeps survivors in original
        order, and debug records kept/dropped indices, the dropped sentence
        texts, and the similarity that triggered each drop.
    """
    import numpy as np

    sents = split_sentences(text)
    if len(sents) <= 1:
        return text, {"sentences": len(sents), "kept": list(range(len(sents))),
                      "dropped": [], "dropped_sentences": [], "sentences_list": sents}

    emb = embedder.encode(sents)               # (n, d), L2-normalized
    q = embedder.encode([intent])[0]           # (d,)
    relevance = emb @ q                         # cosine to intent (normalized)
    order = list(np.argsort(-relevance))        # most relevant first

    kept_idx: List[int] = []
    kept_emb: List = []
    dropped = []  # (idx, max_sim_to_kept)
    for i in order:
        if kept_emb:
            sim = float(max(float(emb[i] @ ke) for ke in kept_emb))
        else:
            sim = 0.0
        if sim >= redundancy_threshold:
            dropped.append((int(i), sim))
            continue
        kept_idx.append(int(i))
        kept_emb.append(emb[i])

    kept_sorted = sorted(kept_idx)              # restore reading order
    deduped_text = " ".join(sents[i] for i in kept_sorted)
    debug = {
        "sentences": len(sents),
        "kept": kept_sorted,
        "dropped": dropped,
        "dropped_sentences": [sents[i] for i, _ in dropped],
        "sentences_list": sents,
    }
    return deduped_text, debug


class CrossEncoderReranker:
    """Minimal BGE cross-encoder reranker via raw transformers.

    We deliberately avoid FlagEmbedding here: its encoder-reranker calls
    `tokenizer.prepare_for_model(...)`, which was removed in transformers 5.x
    (llmlingua pulls in transformers 5.x), so FlagReranker crashes. This mirrors
    exactly what LLMLingua itself does internally for `rank_method="bge_reranker"`:
    score [query, passage] pairs with AutoModelForSequenceClassification logits.

    Exposes `compute_score(pairs, normalize=True)` so it is a drop-in for the rest
    of this module (and matches FlagReranker's signature).
    """

    def __init__(self, model_name: str, device: str = "cuda", max_length: int = 512,
                 use_fp16: bool = True):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = device
        self.max_length = max_length
        self.tokenizer = guarded_from_pretrained(AutoTokenizer, model_name)
        model = guarded_from_pretrained(AutoModelForSequenceClassification, model_name)
        if use_fp16 and device.startswith("cuda"):
            model = model.half()
        self.model = model.eval().to(device)

    def compute_score(self, pairs, normalize: bool = True, batch_size: int = 64):
        torch = self._torch
        queries = [p[0] for p in pairs]
        passages = [p[1] for p in pairs]
        scores: List[float] = []
        for i in range(0, len(pairs), batch_size):
            inputs = self.tokenizer(
                queries[i : i + batch_size],
                passages[i : i + batch_size],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs, return_dict=True).logits.view(-1).float()
            if normalize:
                logits = torch.sigmoid(logits)
            scores.extend(logits.cpu().tolist())
        return scores


def select_and_reorder(
    reranker,
    question: str,
    documents: List[str],
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
):
    """Coarse stage: question-aware ranking -> budget selection -> reorder.

    Args:
        reranker: a loaded FlagEmbedding.FlagReranker (BGE cross-encoder).
        question: the user query that drives relevance.
        documents: retrieved chunks, one per list element.
        top_k: keep at most this many documents (None = keep all that survive).
        score_threshold: optional minimum reranker score in [0, 1] to keep a doc.

    Returns:
        (selected_documents, debug) where selected_documents is ordered
        most-relevant-first, and debug carries the raw scores + kept indices.
    """
    if not documents:
        return [], {"scores": [], "kept_indices": []}

    # BGE cross-encoder scores [query, passage] pairs; normalize=True -> sigmoid 0..1.
    pairs = [[question, doc] for doc in documents]
    scores = reranker.compute_score(pairs, normalize=True)
    if not isinstance(scores, list):  # a single pair returns a bare float
        scores = [scores]
    scores = [float(s) for s in scores]  # plain floats -> JSON serializable

    # Rank document indices by relevance, most relevant first.
    ranked = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)

    # Budget selection: optional relevance floor, then optional top-k cap.
    if score_threshold is not None:
        ranked = [i for i in ranked if scores[i] >= score_threshold]
    if top_k is not None:
        ranked = ranked[:top_k]

    # Reorder survivors most-relevant-first (mitigates lost-in-the-middle bias).
    selected = [documents[i] for i in ranked]
    return selected, {"scores": scores, "kept_indices": ranked}


def longllmlingua_fine(compressor_long, documents, question, rate=0.75, target_token=-1):
    """Fine stage using REAL LongLLMLingua (causal/perplexity path).

    This is the query-aware compressor your repo originally skipped: it runs on a
    causal backbone (use_llmlingua2=False) with question-conditioned perplexity,
    document reordering, and dynamic per-context ratios. Returns just the
    compressed CONTEXT string (the trailing question is stripped so it matches the
    LLMLingua-2 arm for fair scoring).
    """
    kwargs = dict(
        instruction="",
        question=question,
        condition_in_question="after_condition",   # question-aware conditioning
        reorder_context="sort",                     # fight lost-in-the-middle
        dynamic_context_compression_ratio=0.3,      # per-doc dynamic ratio
        condition_compare=True,
        context_budget="+100",
        rank_method="longllmlingua",                # the LongLLMLingua ranker
    )
    if target_token and target_token > 0:
        kwargs["target_token"] = target_token
    else:
        kwargs["rate"] = rate

    res = compressor_long.compress_prompt(documents, **kwargs)
    cc = res.get("compressed_prompt", "")
    q = (question or "").strip()
    if q and cc.rstrip().endswith(q):  # strip the appended question
        cc = cc.rstrip()[: -len(q)].strip()
    return cc


def two_stage_compress(
    reranker,
    compressor,
    instruction: str,
    question: str,
    documents: List[str],
    rate: float = 0.5,
    target_token: int = -1,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
):
    """End-to-end two-stage compression (THE deliverable function).

    Args:
        reranker:    loaded FlagEmbedding.FlagReranker (coarse stage).
        compressor:  loaded llmlingua.PromptCompressor (use_llmlingua2=True).
        instruction: system/task instruction; preserved verbatim.
        question:    user query; drives coarse ranking; preserved verbatim.
        documents:   list of retrieved chunks (one element per chunk).
        rate:        fine-stage keep rate (fraction of tokens). Used unless
                     target_token > 0.
        target_token: fine-stage hard token budget for the context (-1 = use rate).
        top_k:       max documents to keep in the coarse stage (None = all ranked).
        score_threshold: optional minimum reranker score in [0, 1] to keep a doc.

    Returns:
        dict with the assembled compressed prompt and token counts.
    """
    # ---- Stage 1: question-aware coarse selection + reordering ----
    selected, coarse = select_and_reorder(
        reranker, question, documents, top_k=top_k, score_threshold=score_threshold
    )
    if not selected:
        selected = [" "]  # keep llmlingua happy if everything was filtered out

    # ---- Stage 2: LLMLingua-2 token-level compression of the survivors ----
    # use_context_level_filter=False: the reranker already did coarse selection,
    # so the encoder does ONLY token-level compression here (no doc dropping).
    fine = compressor.compress_prompt(
        selected,
        rate=rate,
        target_token=target_token,
        use_context_level_filter=False,
        use_token_level_filter=True,
        force_tokens=FORCE_TOKENS,
        force_reserve_digit=True,   # keep digits intact (numbers matter in RAG)
        drop_consecutive=True,
    )
    compressed_context = fine["compressed_prompt"]

    # ---- Assemble final prompt: instruction + compressed context + question ----
    # Instruction and question are NOT compressed: they are short, sensitive, and
    # define the task. Only the retrieved context is compressed.
    parts = [
        p for p in (instruction.strip(), compressed_context.strip(), question.strip()) if p
    ]
    final_prompt = "\n\n".join(parts)

    # Token counts via the compressor's own tokenizer for a consistent measure.
    tok = compressor.tokenizer
    full_original = "\n\n".join(
        p
        for p in (instruction.strip(), "\n\n".join(documents).strip(), question.strip())
        if p
    )
    origin_total = len(tok.encode(full_original, add_special_tokens=False))
    compressed_total = len(tok.encode(final_prompt, add_special_tokens=False))

    return {
        "compressed_prompt": final_prompt,
        # the compressed CONTEXT only (no instruction/question) - what a reader
        # should be given alongside its own downstream questions:
        "compressed_context": compressed_context,
        # context-only numbers straight from LLMLingua-2 (the token stage):
        "context_origin_tokens": fine["origin_tokens"],
        "context_compressed_tokens": fine["compressed_tokens"],
        # whole-prompt numbers (include the preserved instruction + question):
        "origin_tokens": origin_total,
        "compressed_tokens": compressed_total,
        "rate": compressed_total / max(origin_total, 1),
        # coarse-stage diagnostics:
        "kept_documents": len(coarse["kept_indices"]),
        "total_documents": len(documents),
        "reranker_scores": coarse["scores"],
    }
