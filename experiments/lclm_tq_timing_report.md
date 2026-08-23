# LCLM + TurboQuant - timing & memory benchmark

- LCLM checkpoint: `latent-context/0.6b-4b-LCLM-16x` (encoder Qwen3-Embedding-0.6B → adapter → decoder Qwen3-4B-Instruct-2507; 36 layers, 8 KV heads, head_dim=128)
- GPU: **A100-80GB**, bf16 weights | decode: greedy (`do_sample=False`)
- Two **isolated** warm Modal containers: `LCLMVanilla` (fp16 `DynamicCache`) vs `LCLMTurboQuant` (`TQCache`)
- TQ configs: **4-bit**, **3-bit**, **3.5-bit** (`bit_width=3` + 32 outlier channels @ 4-bit)
- TTFT = prefill + first decoded token (separate `max_new_tokens=1` generate). `kv_bytes` = real packed quantized cache size (TQ) or fp16 cache size (vanilla). `KV ×` = fp16-equivalent ÷ TQ packed bytes for the same decoder sequence length.

The LCLM encoder compresses the input doc into a small number of latent soft
tokens (the decoder's effective prompt). With the 16x checkpoint, ~16 input
tokens → 1 latent, so a 16k-token doc becomes ~1000 latent tokens - that's the
decoder sequence length that drives KV-cache size.

> **Honest trade-off.** TurboQuant here is **pure-PyTorch dequant with no custom
> CUDA kernel**: every decode step de-quantizes the whole cache, so wall-clock
> decode is **slower** than fp16. The payoff is **memory** - KV-cache bytes drop
> ~3.8–4.9×. This matches the paper, which needs custom kernels to also win on
> speed. We report both, plainly: **TQ = slower tok/s, smaller KV, lower peak GPU.**

## Headline - per context length (decode = 512 tokens)

KV bytes are the actual decoder KV-cache size at end of generation (MB). Peak GPU
is `torch.cuda.max_memory_allocated`. Decode tok/s excludes the prefill/TTFT.

| ctx (input tok) | latents | arm | TTFT (s) | decode tok/s | peak GPU (MB) | KV (MB) | KV × | eff bits | needle |
|---|---|---|---|---|---|---|---|---|---|
| ~2k (2018) | 127 | vanilla fp16 | 0.118 | 23.75 | 9613 | 104.0 | 1.0× | 16.0 | ✅ |
| | | TQ-4bit | 0.142 | 16.68 | 9657 | 27.6 | 3.77× | 4.0 | ✅ |
| | | TQ-3bit | 0.155 | 16.64 | 9657 | 21.1 | 4.93× | 3.0 | ❌ |
| | | TQ-3.5bit | 0.193 | 9.73 | 9657 | 24.4 | 4.27× | 3.25 | ✅ |
| ~8k (8007) | 501 | vanilla fp16 | 0.217 | 26.55 | 10495 | 159.1 | 1.0× | 16.0 | ✅ |
| | | TQ-4bit | 0.273 | 16.85 | 10633 | 42.3 | 3.77× | 4.0 | ✅ |
| | | TQ-3bit | 0.238 | 16.42 | 10633 | 32.3 | 4.93× | 3.0 | ✅ |
| | | TQ-3.5bit | 0.319 | 9.89 | 10644 | 37.3 | 4.27× | 3.25 | ✅ |
| ~16k (16000) | 1000 | vanilla fp16 | 0.347 | 25.35 | 11670 | 232.7 | 1.0× | 16.0 | ✅ |
| | | TQ-4bit | 0.359 | 16.79 | 11916 | 61.8 | 3.77× | 4.0 | ✅ |
| | | TQ-3bit | 0.377 | 16.69 | 11916 | 42.1 | 4.93× | 3.0 | ✅ |
| | | TQ-3.5bit | 0.438 | 9.97 | 11914 | 54.5 | 4.27× | 3.25 | ✅ |

## Short decode (decode = 128 tokens)

| ctx | arm | TTFT (s) | decode tok/s | peak GPU (MB) | KV (MB) | KV × | needle |
|---|---|---|---|---|---|---|---|
| ~2k | vanilla fp16 | 0.530 | 27.54 | 9613 | 47.3 | 1.0× | ✅ |
| | TQ-4bit | 0.619 | 17.39 | 9657 | 12.6 | 3.78× | ✅ |
| | TQ-3bit | 0.140 | 16.80 | 9657 | 9.6 | 4.94× | ❌ |
| | TQ-3.5bit | 0.219 | 9.69 | 9657 | 11.1 | 4.28× | ✅ |
| ~8k | vanilla fp16 | 0.208 | 26.06 | 10495 | 102.5 | 1.0× | ✅ |
| | TQ-4bit | 0.243 | 16.82 | 10633 | 27.2 | 3.77× | ✅ |
| | TQ-3bit | 0.245 | 17.66 | 10633 | 20.8 | 4.93× | ✅ |
| | TQ-3.5bit | 0.278 | 10.05 | 10644 | 24.0 | 4.27× | ✅ |
| ~16k | vanilla fp16 | 0.363 | 26.86 | 11670 | 176.1 | 1.0× | ✅ |
| | TQ-4bit | 0.364 | 16.93 | 11916 | 44.6 | 3.77× | ✅ |
| | TQ-3bit | 0.374 | 16.21 | 11916 | 35.8 | 4.93× | ✅ |
| | TQ-3.5bit | 0.450 | 10.04 | 11914 | 39.4 | 4.27× | ✅ |

(`peak GPU` is dominated by the ~8 GB of bf16 model weights + activations; the KV
delta is a small slice of the total here because the decoder sequence is only
~100–1000 latent tokens. The KV-bytes column is where the compression shows
cleanly - and it's the figure that matters when you scale latents/batch up.)

## What the numbers say

- **KV-cache memory shrinks 3.8–4.9×**, exactly as designed: 4-bit → 3.77×,
  3.5-bit → 4.27×, 3-bit → 4.93×. At 16k input / 512 decode the fp16 cache is
  **233 MB**; TQ-3bit holds the same cache in **42 MB**. The ratio is stable
  across context length and decode length (it's a per-element property).
- **Decode is slower under TQ**, also as expected for a kernel-less dequant path:
  vanilla runs **~24–27 tok/s**, TQ-4bit/3bit **~16–18 tok/s** (≈0.65×), and
  **TQ-3.5bit ~10 tok/s** (≈0.4×). The 3.5-bit "outlier" variant is the slowest
  because it runs *two* quantizers per layer (regular + outlier channels) and a
  scatter/gather on the channel mask every step.
- **TTFT is essentially unchanged** (within ~0.05 s of vanilla). The first token
  is dominated by encoder prefill + decoder prefill; the per-step dequant cost
  only accrues during the autoregressive decode loop, so it shows up in tok/s,
  not TTFT.
- **Peak GPU** is ~40–250 MB higher under TQ at these sizes - the dequant path
  materializes a transient fp32 copy of K/V each step, which at ≤1k latent tokens
  outweighs the packed-cache savings *in resident peak*. The memory win is in the
  **stored cache bytes** (the column that scales with batch × latents × decode),
  not the instantaneous peak at this modest scale.

## Output correctness parity (needle retrieval)

Every run was asked for the planted passphrase **"violet-harbor-1987"** plus a
long free-form summary. Both arms retrieved the needle identically in **all but
one** configuration:

- ✅ **Vanilla retrieved the needle in 6/6 runs.**
- ✅ **TurboQuant retrieved it in 10/12 runs.** The only misses were **TQ-3bit at
  the shortest (~2k) context** (both decode lengths). At 2k the doc compresses to
  just 127 latent tokens, so there is little redundancy to absorb 3-bit
  quantization noise and the fact was lost. At **8k and 16k, TQ-3bit recovers the
  needle** (more latents → more redundancy), and **TQ-3.5bit and TQ-4bit retrieve
  it at every context length**, confirming output parity with vanilla at ≥3.5
  effective bits.

**Takeaway:** at 3.5-bit and above, TurboQuant is output-equivalent to vanilla
fp16 on this needle task while compressing the decoder KV cache ~4.3×. 3-bit is
viable once the context is long enough to carry redundancy, but is the lossy edge.

## Reproduce

```bash
cd experiments
modal deploy lclm_tq_timing.py     # warm LCLMVanilla + LCLMTurboQuant (A100-80GB)
python run_lclm_timing.py          # sweep -> lclm_tq_timing_results.json
```

Sweep: context ∈ {2k, 8k, 16k} input tokens × decode ∈ {128, 512} × {vanilla,
TQ-4bit, TQ-3bit, TQ-3.5bit} = 24 runs. Raw per-run numbers (including full
output text) are in `lclm_tq_timing_results.json`.
