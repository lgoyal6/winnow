"""
HuggingFace/torch backend for AttentionRAG (Eq. 1/2 + generation).

Implements the `Backend` protocol from core.py using a causal LM. This is the
faithful, model-touching half: it tokenizes the anchor prompt so that the exact
token positions of the context chunk are known, predicts the single anchor token
a_j, and -- when a_j is not 'none' -- reads that token's attention over the
context tokens, SUMMED OVER ALL LAYERS (Eq. 2), averaged over heads.

Requires `attn_implementation="eager"` so attention weights are returned.

Paper defaults: compression/generation model = Llama-3.1-8B-Instruct or
Qwen-2.5-7B-Instruct; hint prefix authored by GPT-4o Mini. Here the same local
model authors the hint prefix by default (set OPENAI_API_KEY + use_openai_hint
to match the paper exactly).
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_guard import guarded_from_pretrained  # noqa: E402

from .core import (
    FocusResultData,
    _is_none_anchor,
    select_sentences,
    split_sentence_spans,
)
from .prompts import (
    ANSWER_PROMPT,
    FIXED_HINT_PREFIX,
    HINT_PREFIX_PROMPT,
    build_anchor_segments,
)


class HFBackend:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
        use_openai_hint: bool = False,
        openai_model: str = "gpt-4o-mini",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.model_name = model_name
        self.use_openai_hint = use_openai_hint
        self.openai_model = openai_model

        self.tokenizer = guarded_from_pretrained(AutoTokenizer, model_name)
        # eager attention is REQUIRED to get attention weights back from forward.
        self.model = guarded_from_pretrained(
            AutoModelForCausalLM, model_name,
            torch_dtype=getattr(torch, dtype),
            attn_implementation="eager",
            device_map=device,
        ).eval()
        self.n_layers = self.model.config.num_hidden_layers

    # ------------------------------------------------------------------ #
    # Step 1: answer hint prefix (B.1)                                    #
    # ------------------------------------------------------------------ #
    def generate_hint_prefix(self, question: str) -> str:
        prompt = HINT_PREFIX_PROMPT.format(question=question)
        if self.use_openai_hint:
            return self._openai_hint(prompt)
        text = self._chat_generate(prompt, max_new_tokens=32)
        # keep only the first line; the model is told to return only the format
        return text.strip().splitlines()[0].strip() if text.strip() else "None"

    def _openai_hint(self, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=32,
        )
        out = (resp.choices[0].message.content or "").strip()
        return out.splitlines()[0].strip() if out else "None"

    # ------------------------------------------------------------------ #
    # Step 2: chunking (uniform, m tokens per chunk; n = ceil(|C|/m))     #
    # ------------------------------------------------------------------ #
    def chunk_context(self, context: str, chunk_size: int) -> List[str]:
        ids = self.tokenizer(context, add_special_tokens=False).input_ids
        if not ids:
            return []
        chunks = []
        for i in range(0, len(ids), chunk_size):
            piece = ids[i : i + chunk_size]
            chunks.append(self.tokenizer.decode(piece))
        return chunks

    # ------------------------------------------------------------------ #
    # Eq. 1/2: anchor token + attention over context (summed over layers)#
    # ------------------------------------------------------------------ #
    def focus_attention(
        self, chunk_text: str, question: str, prefix: str
    ) -> FocusResultData:
        torch = self.torch
        pre, ctx, post = build_anchor_segments(chunk_text, question, prefix)

        # Tokenize each segment separately so we know the EXACT positions of the
        # context tokens and have char offsets aligned to them.
        ids_pre = self.tokenizer(pre, add_special_tokens=True).input_ids
        enc_ctx = self.tokenizer(
            ctx, add_special_tokens=False, return_offsets_mapping=True
        )
        ids_ctx = enc_ctx["input_ids"]
        offsets = [tuple(o) for o in enc_ctx["offset_mapping"]]
        ids_post = self.tokenizer(post, add_special_tokens=False).input_ids

        if not ids_ctx:
            return FocusResultData(anchor="none", attention=[], token_offsets=[])

        ctx_start = len(ids_pre)
        ctx_end = ctx_start + len(ids_ctx)
        input_ids = torch.tensor(
            [ids_pre + ids_ctx + ids_post], device=self.device
        )

        with torch.no_grad():
            # Pass 1: predict the anchor token a_j (greedy). Keep the cache so the
            # second pass only has to process the single anchor token.
            out1 = self.model(input_ids, use_cache=True)
            anchor_id = int(out1.logits[0, -1].argmax())
            anchor_text = self.tokenizer.decode([anchor_id]).strip()

            if self._is_none(anchor_text):
                return FocusResultData(
                    anchor=anchor_text or "none",
                    attention=[],
                    token_offsets=offsets,
                )

            # Pass 2: append a_j and read ITS attention over the context tokens.
            anchor_tok = torch.tensor([[anchor_id]], device=self.device)
            out2 = self.model(
                anchor_tok,
                past_key_values=out1.past_key_values,
                use_cache=True,
                output_attentions=True,
            )
            # attentions: tuple(L) of [batch, heads, q_len=1, kv_len=full]
            # Eq. 2: A_j = sum_l Attention_l(c_j, a_j), averaged over heads.
            A = None
            for layer_attn in out2.attentions:
                # focal (anchor) row is the only query position -> index 0
                head_mean = layer_attn[0, :, 0, ctx_start:ctx_end].mean(dim=0)
                A = head_mean if A is None else A + head_mean
            attention = A.float().cpu().tolist()

        return FocusResultData(
            anchor=anchor_text, attention=attention, token_offsets=offsets
        )

    @staticmethod
    def _is_none(anchor: str) -> bool:
        a = anchor.strip().strip(".,;:!?\"'").lower()
        return a == "" or a.startswith("none")

    # ------------------------------------------------------------------ #
    # Step 5: final answer from compressed context                       #
    # ------------------------------------------------------------------ #
    def generate_answer(self, compressed_context: str, question: str) -> str:
        prompt = ANSWER_PROMPT.format(context=compressed_context, question=question)
        return self._chat_generate(prompt, max_new_tokens=128).strip()

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #
    def _chat_generate(self, user_content: str, max_new_tokens: int) -> str:
        torch = self.torch
        messages = [{"role": "user", "content": user_content}]
        # return_dict=True -> BatchEncoding(input_ids, attention_mask); robust
        # across transformers versions (a bare tensor return was deprecated).
        enc = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)
        prompt_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0, prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)

    # ------------------------------------------------------------------ #
    # Offset-exact compression: returns kept ORIGINAL char-spans          #
    # ------------------------------------------------------------------ #
    def compress_spans(
        self,
        text: str,
        question: str,
        chunk_size: int = 300,
        top_k: int = 12,
        hint_prefix: Optional[str] = None,
        use_fixed_prefix: bool = False,
    ) -> dict:
        """Run AttentionRAG but return the kept sentences as exact char-spans in
        the ORIGINAL `text` (not re-decoded text). This lets a caller build a
        keep-mask over the original by char-position containment -- the robust
        way to intersect/union with another compressor.

        Chunks are formed from token ranges of the original (via offset mapping),
        so each chunk is an exact substring `text[char_start:char_end]` and every
        kept sentence maps back to an exact original char-span.
        """
        # Step 1: hint prefix
        if use_fixed_prefix:
            raw_prefix = FIXED_HINT_PREFIX
        elif hint_prefix is not None:
            raw_prefix = hint_prefix
        else:
            raw_prefix = self.generate_hint_prefix(question)
        is_empty = raw_prefix.strip().lower() in ("none", "")
        eff_prefix = "" if is_empty else raw_prefix.strip()

        # Step 2: chunk by token ranges of the ORIGINAL, keeping char offsets
        enc = self.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
        ids = enc["input_ids"]
        offs = enc["offset_mapping"]

        kept_spans = []  # (orig_start, orig_end)
        anchors = []
        n_chunks = 0
        n_kept_chunks = 0
        # When the whole input fits in a single chunk, a "none" anchor would
        # otherwise drop the entire text from AttentionRAG (there is no second
        # chunk to fall back on, and a "none" anchor carries no attention signal
        # to rank sentences). In that single-chunk case keep the chunk wholesale
        # instead of dropping it, so the downstream merge still has spans to work
        # with. Multi-chunk inputs keep the normal per-chunk "none" drop - that
        # coarse relevance gate is the point of AttentionRAG on long contexts.
        single_chunk = len(ids) <= chunk_size
        for i in range(0, len(ids), chunk_size):
            j = min(i + chunk_size, len(ids))
            char_start = offs[i][0]
            char_end = offs[j - 1][1]
            chunk_text = text[char_start:char_end]  # exact substring of original
            n_chunks += 1

            focus = self.focus_attention(chunk_text, question, eff_prefix)
            anchors.append(focus.anchor)
            if _is_none_anchor(focus.anchor):
                if single_chunk:
                    # Keep every sentence of the only chunk rather than dropping it.
                    for s, e, _t in split_sentence_spans(chunk_text):
                        kept_spans.append((char_start + s, char_start + e))
                    n_kept_chunks += 1
                continue

            _kept_text, kept_idx = select_sentences(
                chunk_text, focus.token_offsets, focus.attention, top_k
            )
            spans = split_sentence_spans(chunk_text)
            for k in kept_idx:
                s, e, _t = spans[k]
                kept_spans.append((char_start + s, char_start + e))
            n_kept_chunks += 1

        return {
            "hint_prefix": raw_prefix.strip(),
            "is_empty_prefix": is_empty,
            "kept_spans": kept_spans,  # exact char-spans in the ORIGINAL text
            "anchors": anchors,
            "n_chunks": n_chunks,
            "n_kept_chunks": n_kept_chunks,
        }
