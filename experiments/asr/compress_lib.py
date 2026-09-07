"""Winnow's real compressor, loaded locally instead of on Modal.

The compression call is copied verbatim from the production path in
llmlingua2_modal.py (`Compressor.compress`): same model, same force_tokens, same
force_reserve_digit / drop_consecutive flags. Only the device changes, because
this box has no CUDA. Nothing about the compressor's behaviour is reimplemented
here - that would make the whole study measure a replica rather than the product.

Loading goes through the repo's own model_guard, so the pinned revision, the
safetensors-only rule and the trust_remote_code refusal all still apply.
"""
from __future__ import annotations

import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf"))
sys.path.insert(0, str(REPO))

MODEL_NAME = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
FORCE_TOKENS = ["\n", ".", "!", "?", ","]


def pick_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_compressor(device: str | None = None):
    from llmlingua import PromptCompressor
    from model_guard import (
        activate_loaded_model,
        llmlingua_model_config,
        verified_snapshot_download,
    )
    device = device or pick_device()
    snapshot = verified_snapshot_download(MODEL_NAME)
    candidate = PromptCompressor(
        model_name=snapshot,
        use_llmlingua2=True,
        device_map=device,
        model_config=llmlingua_model_config(MODEL_NAME, local_files_only=True),
    )
    compressor = activate_loaded_model(
        snapshot,
        candidate,
        tensor_owner=candidate.model,
        activation_key=(MODEL_NAME, "llmlingua"),
    )
    return compressor, device


def compress(compressor, text: str, rate: float = 0.5, force_tokens=None):
    """The production token-level call. Returns (compressed_text, raw_result)."""
    if not text.strip():
        return "", {"compressed_prompt": "", "empty_input": True}
    ft = FORCE_TOKENS if force_tokens is None else force_tokens
    out = compressor.compress_prompt(
        text,
        rate=rate,
        force_tokens=ft,
        force_reserve_digit=True,
        drop_consecutive=True,
    )
    return out["compressed_prompt"], out


def flatten(text: str) -> str:
    """Strip punctuation and case: what a streaming recogniser without a
    punctuation model actually hands you, and the surface form of the
    LibriSpeech reference."""
    import re
    t = re.sub(r"[^A-Za-z0-9' ]+", " ", text.replace("-", " "))
    return " ".join(t.upper().split())
