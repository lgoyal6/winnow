# turboquant_kv

Winnow's TurboQuant path reports 3.77-4.93x KV-cache compression. Measured in
bytes the process actually holds, it saves **nothing**: at 16k context the
quantized cache is 867.8 MB against fp16's 866.1 MB. The reported figure comes
from `mem_bits()`, which counts the bits the scheme needs, while the code keeps
the uint8 indices and a full-precision dequantized copy at the same time and
grows the latter with `torch.cat`.

turboquant_kv makes the ratio real, then asks the question the ratio was hiding:
at what bit width does the model still work? The answer is **8-bit**, which is
1.97x, not 3.8-4.9x.

> **Thesis.** A compression ratio is a claim about resident bytes and an
> accuracy claim is a claim about a score. This project had a logical bit count
> validated by a single planted passphrase, and both halves failed the moment
> they were measured properly. Nothing here is a coding error: the quantizer hits
> its theoretical Lloyd-Max distortion to three decimals. The operating point
> was wrong, and a smoke test could not see it.

The fused kernel is the same story from the other direction. Built to save
memory traffic it was **12x slower than PyTorch**, because the operation is
compute-bound in fp32 and I had optimized the wrong ceiling. Moving the rotation
to TF32 tensor cores makes the identical kernel 6.6x to 34.5x faster.

A correction is recorded in place rather than edited away: an earlier conclusion
that 6-bit was usable came from a greedy-token-match proxy on a repeated-sentence
prompt, and the scored benchmark overturned it by 25 points. That is the same
class of error as the needle test it was meant to replace.

Measured on an RTX A6000 (sm_86, 48 GB), `Qwen/Qwen2.5-7B-Instruct` in bf16,
torch 2.13.0+cu129, triton 3.7.1, transformers 5.15.1, seed 0.

The card's own ceilings, measured rather than assumed (see the `decode-anatomy`
roofline study): **711 GB/s** achieved HBM bandwidth, **23.9 TFLOP/s** fp32,
**65.1 TFLOP/s** tf32, **128.5 TFLOP/s** bf16.

---

## Phase A: what the cache actually costs

### The problem

`TQLayer.update` in `turboquant_modal.py` keeps the quantized indices *and* a
dequantized full-precision cache at the same time, and grows the latter with
`torch.cat`:

```python
kd = self._quant(ks); self._key_data.append(kd)   # uint8 indices, kept
nk = self._dequant_one(kd)                        # dequantize the new token
self._ck = torch.cat([self._ck, nk], dim=-2)      # full-precision cache, grown
return self._ck, self._cv
```

Three costs follow. The indices sit in `uint8`, so a 4-bit index wastes half a
byte. `_ck`/`_cv` are a full fp16 cache that the quantization was supposed to
replace. And `torch.cat` allocates a second full-precision copy every step
before freeing the first, which is both an O(n^2) total copy and a 2x peak
spike.

`mem_bits()` reports none of this: it counts the bits the *scheme* needs.

### The fix

`cache.py` bit-packs indices exactly (`packing.py`, indices may straddle byte
boundaries so 128 three-bit indices take 48 bytes rather than 64), preallocates
a ring buffer so there is no `torch.cat`, drops the persistent full-precision
cache, and replaces `centroids[idx.long()]` (an 8x int64 temporary on the hot
path) with a chunked `index_select` on int32.

### Result

Resident bytes are measured by walking every tensor reachable from the cache and
deduplicating by storage pointer, so the same function measures all three arms
and an implementation that keeps two copies is charged for two.

| ctx | arm | reported (`mem_bits`) | **resident** | gap | vs fp16 |
|---|---|---|---|---|---|
| 2048 | fp16 | - | 115.2 MB | - | 1.00x |
| 2048 | tq4-current | 30.6 MB | **116.9 MB** | 3.82x | 0.99x |
| 2048 | tq4-packed | - | **29.7 MB** | - | **3.88x** |
| 8192 | fp16 | - | 437.2 MB | - | 1.00x |
| 8192 | tq4-current | 116.1 MB | **438.9 MB** | 3.78x | 1.00x |
| 8192 | tq4-packed | - | **112.7 MB** | - | **3.88x** |
| 16384 | fp16 | - | 866.1 MB | - | 1.00x |
| 16384 | tq4-current | 230.1 MB | **867.8 MB** | 3.77x | 1.00x |
| 16384 | tq4-packed | - | **223.3 MB** | - | **3.88x** |

The existing arm holds 0.99-1.00x of fp16: it saves nothing, and the "gap"
column is the factor between what it reports and what it holds.

Packed, by bit width, at ctx 16384: 4-bit **3.88x**, 3-bit **5.12x**, 3.5-bit
**4.57x**. Peak allocator bytes drop 21-31% as well, because the transient
dequantized tensor exists for one layer at a time rather than all 28 at once.

### The cost, stated plainly

| ctx | fp16 | tq4-current | tq4-packed (torch dequant) |
|---|---|---|---|
| 2048 | 34.90 tok/s | 21.20 | 10.93 |
| 8192 | 35.04 tok/s | 21.43 | 5.29 |
| 16384 | ~35 tok/s | 21.43 | 3.03 |

Dropping the full-precision cache means re-dequantizing the history every step:
O(length) instead of O(1). That regression is what phase C removes, and it is
why phase C exists.

The trade is between *persistent* and *transient* bytes. Both schemes touch a
full-precision tensor once per step; only one of them keeps it. For a single
sequence the peak is similar, but N concurrent sequences hold N packed caches
and share one transient, versus N full-precision caches.

---

## Phase B: a scored benchmark

The existing correctness evidence for TurboQuant is one planted passphrase
("violet-harbor-1987") checked over 12 runs. That is a smoke test: one bit of
signal per run, blind to partial degradation, and a cache that mangles
everything except a rare verbatim token would pass it.

**This phase also caught an error in this repo's own earlier conclusion, which
is the best argument for it existing.** `bitwidth_sweep.py` reported that 6-bit
reproduces the fp16 continuation token-for-token, and that was written up here as
"the usable operating point is 6-bit at 2.61x". It is not. That sweep continued a
*repeated sentence* for 18 greedy tokens, which is close to the easiest
prediction task available, and 6-bit passes it comfortably. Scored on real
retrieval and QA, 6-bit loses 25 points. A cheap proxy metric produced exactly
the same false confidence as the needle passphrase it was meant to replace.

`run_longbench.py` scores LongBench-E with the official prompt templates and
metrics (pure string functions, no model-as-judge), bucketed 0-4k / 4-8k / 8k+.
`datasets.load_dataset("THUDM/LongBench", ...)` no longer works (the repo ships
a loader script and current `datasets` refuses to run those), so `longbench.py`
fetches and reads `data.zip` directly.

LongBench-E, 120 samples per arm (10 per task per bucket, 4 tasks, 3 buckets),
`Qwen2.5-7B-Instruct`, greedy, middle-truncated to 16384 tokens.

| arm | 0-4k | 4-8k | 8k+ | all | delta vs fp16 |
|---|---|---|---|---|---|
| fp16 | 36.89 | 36.11 | 38.96 | **37.32** | - |
| tq8 | 36.13 | 36.27 | 37.77 | **36.72** | **-0.60** |
| tq6 | 22.60 | 7.53 | 6.78 | 12.30 | **-25.02** |
| tq5 | 2.78 | 0.29 | 0.07 | 1.05 | -36.27 |
| tq4 | 2.68 | 2.58 | 0.00 | 1.76 | -35.57 |

Two things to read off this beyond the headline.

**Degradation grows with context.** 6-bit loses 14.29 points in the 0-4k bucket
and 32.18 in the 8k+ bucket. More quantized entries compete in each attention
softmax, so per-entry error that is tolerable over 2k positions is not tolerable
over 8k. Any KV-quantization result reported at one context length is reporting
the easy case.

**8-bit's -0.60 is within noise at this sample size; 6-bit's -25.02 is not.**
With n=120 per arm (n=40 per bucket) this run separates "fine" from "broken"
confidently and cannot resolve a one-point regression. That is the right
resolution for choosing a bit width and the wrong resolution for defending a
sub-point claim.

---

## Phase C: the fused kernel, and why the first version lost

### What the kernel does

`kernel.py` does unpack, codebook gather, the 128x128 inverse rotation and the
norm rescale in registers, writing bf16 once. Bit extraction is general over
width, assembling each index from two bytes with a masked load for the final one
whose second byte lies past the end of the row. It is stride-aware, so it reads
the preallocated cache buffer in place rather than forcing a `.contiguous()`
copy of the whole packed cache every step.

### The first version lost by 12x, and that is the finding

Fused in fp32, the kernel was **12x slower** than the PyTorch path it replaced
(3.1 GB/s against 39 GB/s). The premise was that the two fp32 intermediates
dominate. Measuring arithmetic intensity says otherwise:

| | AI | ridge point | verdict |
|---|---|---|---|
| fp32 | 93-102 FLOP/byte | 33.6 | **compute-bound** |
| tf32 | 93-102 | 91.6 | compute-bound |
| bf16 | 93-102 | 180.7 | memory-bound |

The operation is compute-bound in fp32, so fusing to cut memory traffic attacks
the wrong ceiling, and a hand-rolled fp32 GEMM loses to cuBLAS. PyTorch reached
4.02 TFLOP/s of the card's 23.9 fp32; the fused kernel reached 0.32.

Moving the rotation to TF32 tensor cores changes the binding constraint and the
same kernel wins everywhere:

| bit width | PyTorch | fused (TF32) | speedup | PyTorch | fused |
|---|---|---|---|---|---|
| 4-bit, N=1048576 | 8.54 ms | **1.30 ms** | **6.6x** | 40 GB/s | **259 GB/s** |
| 6-bit, N=1048576 | 44.77 ms | **1.30 ms** | **34.5x** | 8 GB/s | **286 GB/s** |

6-bit gains more because PyTorch's general bit-unpacking (a shift per bit plus a
masked sum) is far worse than its 4-bit nibble path, while the kernel extracts
bits in registers at any width.

### The losing region

In fp32 the kernel loses at every shape above N=2048. In TF32 there is no shape
in the sweep where PyTorch wins, including a single decode step at batch 1
(N=4), because the PyTorch path is four kernel launches against one and the
per-launch floor on this card is 7.5-12 us. The sweep covers N = 4 to 1,048,576
(batch 1-16, cache length 1-16384, 4 KV heads, head_dim 128).

### Accuracy cost of TF32

TF32 has a 10-bit mantissa, and the fused kernel's worst absolute deviation from
the fp32 PyTorch path is **3.1e-2**. That is not free, and it is only acceptable
because it sits inside a quantization error two orders larger: 4-bit TurboQuant
already has 9.7% relative L2 reconstruction error, and 6-bit has 2.6%. The fp32
kernel (`--tf32` off) is available and exact; it is simply slower than PyTorch.

---

## The operating point this all points at

> **Correction.** An earlier version of this file concluded 6-bit at 2.61x,
> based on the greedy-continuation table below. The scored benchmark in phase B
> overturned that: 6-bit loses 25 points on LongBench-E. The table below is kept
> because it is still the correct diagnosis of *why* low bit widths fail, but its
> `greedy match` column is a bad quality gate and should not be read as one.

`bitwidth_sweep.py`, on real K/V from the model:

| bits | B/vector | vs fp16 | rel L2 (real) | rel L2 (Gaussian) | ratio | greedy match vs fp16 |
|---|---|---|---|---|---|---|
| 3 | 50 | 5.12x | 0.1851 | 0.1838 | 1.01 | 17% |
| 4 | 66 | 3.88x | 0.0964 | 0.0963 | 1.00 | 17% |
| 5 | 82 | 3.12x | 0.0497 | 0.0496 | 1.00 | 11% |
| **6** | **98** | **2.61x** | **0.0262** | 0.0262 | 1.00 | **100%** |
| 8 | 130 | 1.97x | 0.0070 | 0.0070 | 1.00 | 100% |

The real-vs-Gaussian ratio is 1.00 at every width. The rotation is doing exactly
what it is supposed to: real K/V quantizes as well as ideal Gaussian input,
rotated-coordinate kurtosis is 3.00, and under 1% of coordinates clip. Nothing
is broken. 4 bits is simply not enough for this model's KV, and the measured
9.7% error matches the theoretical Lloyd-Max distortion for 16 levels on a
Gaussian (0.0975) to three decimals.

At ctx 512 the 6-bit continuation is token-for-token identical to fp16 while
4-bit emits `" the the the the the the..."`.

**So: 8-bit, 1.97x real compression, with the fused kernel making dequantization
6.5-34x cheaper than the PyTorch path.** That is a much smaller compression
number than the project currently claims, and it is the one that survives a
scored benchmark rather than a smoke test.

At 8 bits the indices are already byte-aligned, so bit-packing contributes
nothing there and the entire memory win comes from dropping the redundant
full-precision cache: at ctx 16384 that is roughly 440 MB against the current
implementation's 867.8 MB and fp16's 866.1 MB.

The uncomfortable summary: the method reconstructs exactly as well as its theory
says it should, the implementation is faithful, and it still does not buy what
the project claims it buys on this model. Nothing here is a coding error; the
gap is entirely between a logical bit count validated by a needle test and
resident bytes validated by a score.

### Scope

These numbers are for `Qwen2.5-7B-Instruct` with raw text context on an A6000.
The project's published TurboQuant results use a different setup (a Qwen3-4B
LCLM decoder over 127-1000 encoder-compressed latent tokens on an A100), which
was not reproduced here, so this does not contradict those numbers directly. It
does mean the needle-passphrase test would pass on a cache whose prose output is
unusable, which is the reason phase B exists.

---

## Files

```
packing.py         exact bit-packing, any width; 4-bit fast path
cache.py           TQPackedLayer / TQPackedCache, Lloyd-Max quantizer
kernel.py          fused Triton dequantization, stride-aware
longbench.py       LongBench-E task configs, official metrics, data loading
test_packing.py    pack/unpack round-trip, fast path vs general path
test_cache.py      packed path vs the original dequantization
bench_memory.py    phase A: resident vs reported bytes, tok/s
bitwidth_sweep.py  reconstruction error and output parity vs bit width
diagnose.py        why 4-bit fails: error, kurtosis, clipping, ctx sensitivity
bench_kernel.py    phase C: correctness gate, shape sweep, losing region
run_longbench.py   phase B: scored eval across arms
```

## Commands

```bash
python test_packing.py
python test_cache.py
python bench_memory.py --ctxs 2048 8192 16384 --decode 128
python bitwidth_sweep.py
python diagnose.py
python bench_kernel.py --bw 6 --tf32
python bench_kernel.py --bw 6            # fp32, exact and slower
python run_longbench.py --per-bucket 10
python run_longbench.py --per-bucket 10 --no-kernel   # phase C gate
```

Every GPU job on the shared box goes through a `flock` wrapper. Two benchmarks
sharing the card corrupt each other: an overlapping profiler run inflated wall
time 8% and moved apparent non-kernel time from 30% to 59% before this was
enforced.
