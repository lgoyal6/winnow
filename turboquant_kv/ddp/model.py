"""A compact causal transformer LM, defined in-repo, with a hashable config.

Why a model definition lives here at all: a data-parallel scaling measurement
is only meaningful if the thing being trained cannot change between the
1-GPU and 2-GPU arms. Naming a checkpoint would not give that guarantee, so
the "model revision" recorded in the frozen manifest is the sha256 of this
config's canonical JSON. Change any field and the hash changes, which makes a
mid-experiment substitution visible instead of silent.

This is a SYNTHETIC-WORKLOAD PROXY for the decoder-only models Winnow serves
(pre-norm blocks, causal self-attention, GELU MLP, tied embeddings). It is
not one of those models, it is not trained to convergence, and no loss value
produced with it is a quality claim. Nothing here downloads anything.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LMConfig:
    """Everything that defines the parameter tensor set and the forward pass.

    Frozen and hashed rather than merely documented: `config_sha256()` is the
    model revision the manifest pins, so the 1-GPU and 2-GPU arms can be
    proven to have trained the same architecture.
    """

    name: str
    n_layers: int
    d_model: int
    n_heads: int
    d_ff: int
    vocab_size: int
    max_seq_len: int
    tie_embeddings: bool = True
    norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model {self.d_model} not divisible by n_heads {self.n_heads}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def canonical_json(self) -> str:
        """Byte-stable serialization: sorted keys, no incidental whitespace."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def config_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def parameter_count(self) -> int:
        """Closed-form count, so the manifest can state it without building."""
        d, ff, v, l = self.d_model, self.d_ff, self.vocab_size, self.n_layers
        embed = v * d + self.max_seq_len * d
        if not self.tie_embeddings:
            embed += v * d
        per_layer = (
            2 * d                    # attention pre-norm weight + bias
            + 3 * d * d + 3 * d      # fused qkv projection
            + d * d + d              # attention output projection
            + 2 * d                  # mlp pre-norm weight + bias
            + d * ff + ff            # mlp up
            + ff * d + d             # mlp down
        )
        return embed + l * per_layer + 2 * d  # final norm weight + bias


# Two presets, and only two, so that no run can quietly invent a third.
#
# PROXY is the configuration the frozen manifest pins for the GPU gate: large
# enough that the gradient all-reduce is a real cost (about 143 MB of fp32
# gradients, so several DDP buckets), small enough to fit any modern card.
#
# TINY is for the local CPU gloo self-test only. It exists so the harness's
# correctness path can be exercised in seconds on a laptop. A number measured
# with TINY is a self-test result, never a scaling result.
PROXY = LMConfig(
    name="proxy-6l-512d",
    n_layers=6,
    d_model=512,
    n_heads=8,
    d_ff=2048,
    vocab_size=32000,
    max_seq_len=1024,
)
TINY = LMConfig(
    name="tiny-2l-128d",
    n_layers=2,
    d_model=128,
    n_heads=4,
    d_ff=256,
    vocab_size=1024,
    max_seq_len=64,
)
PRESETS = {PROXY.name: PROXY, TINY.name: TINY}


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: LMConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        # is_causal handles the mask, so no buffer to keep in sync across ranks.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(b, t, self.n_heads * self.head_dim)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: LMConfig) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.norm_mlp = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x))
        return x + self.down(F.gelu(self.up(self.norm_mlp(x))))


class CausalLM(nn.Module):
    """Pre-norm decoder-only LM with learned positions and tied embeddings."""

    def __init__(self, cfg: LMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_out = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_embed.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # Deterministic given a seeded global RNG, which is how every rank is
        # made to start from bit-identical parameters without a broadcast.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, t = tokens.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds max_seq_len "
                             f"{self.cfg.max_seq_len}")
        pos = torch.arange(t, device=tokens.device)
        x = self.tok_embed(tokens) + self.pos_embed(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm_out(x))

    def loss(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self.forward(tokens)
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1))


def build_model(cfg: LMConfig, seed: int) -> CausalLM:
    """Seed then build, so every rank produces the same initial parameters.

    The harness does not trust this: it computes a cross-rank checksum before
    the first step and refuses to measure anything if the ranks disagree.
    """
    torch.manual_seed(seed)
    return CausalLM(cfg)


def flops_per_token(cfg: LMConfig) -> int:
    """Rough forward+backward FLOPs per token, for context only.

    6 * N is the standard estimate (2 for the forward multiply-accumulate,
    4 for the backward), attention's quadratic term excluded. Reported as a
    diagnostic; the harness's throughput claims are measured, not modelled.
    """
    return 6 * cfg.parameter_count()


def describe(cfg: LMConfig) -> dict:
    return {
        "name": cfg.name,
        "config": json.loads(cfg.canonical_json()),
        "config_sha256": cfg.config_sha256(),
        "parameter_count": cfg.parameter_count(),
        "head_dim": cfg.head_dim,
        "flops_per_token_estimate": flops_per_token(cfg),
        "role": ("synthetic-workload proxy for the decoder-only models Winnow "
                 "serves; defined in-repo, nothing downloaded, not a "
                 "quality-bearing model"),
    }
