# A bounded real-consumer GPU diagnostic

This evaluates the actual `serving_adapter.generate` entry point against
`model.generate` on identical inputs. Both use Qwen/Qwen2.5-7B-Instruct, bf16,
SDPA and greedy decoding. Model assets use the existing reviewed revision and
artifact guard. No model API or model-as-judge is used.

Prepare inputs on CPU before renting compute:

```sh
python turboquant_kv/serving_study.py prepare --out "$PRIVATE_INPUTS"
```

The input directory must be new and outside the repository. Preparation pins
the current LongBench dataset revision, saves source task files and hashes the
archive. It selects two questions by a fixed SHA-256 ordering from each of three
publisher length buckets for HotpotQA and English passage retrieval. Each of
the twelve questions is evaluated with middle-truncated token caps of 2,048 and
4,096. The resulting 24 inputs and manifest hash are frozen before inference.
Publisher labels are used without new independent adjudication. Dataset/model
training overlap is unknown. Data and model assets are not redistributed here.

On the owned CUDA instance, install `requirements-serving-study.txt` under
Python 3.12, preserve the exact environment and run:

```sh
python turboquant_kv/serving_study.py run --manifest "$FROZEN_MANIFEST" --out "$PRIVATE_RESULTS"
```

The result directory must be new. The study uses two repetitions with rotated
arm order, for 96 offered requests. Native caching is compared with eight-bit
packed caching under the existing conservative dispatcher. An unfamiliar GPU
must retain its fallback behavior. No A6000 profile is transplanted to an A100.
Codebook/model warmup and a separate profiler trace precede measured requests.

Raw rows retain actual generated token IDs, publisher-metric scores, failures,
per-token emission times, time to first token, completion time, allocator peaks,
unique reachable cache storage and selected dequantization backends. Streamer
callbacks add the same timing instrumentation to both arms. These are in-process
consumer timings, not an HTTP endpoint or concurrent multi-tenant service test.
The fixed diagnostic latency targets are TTFT <= 2 seconds and completion <= 10
seconds. Every offered request stays in the completion/SLO/quality denominators;
a failed request receives score zero. Report useful completed work alongside
latency rather than substituting quick rejection for throughput.

Paired quality uncertainty resamples questions, retaining both context caps and
repetitions together, with 2,000 draws. This twelve-question pilot cannot establish
quality equivalence, broad LongBench performance or deployment SLO guarantees.
Context truncation may remove answer-bearing evidence. Keep task scores, output
tokens, failures and losing axes visible. Model weights dominate total memory;
cache-byte reduction does not imply the same reduction in whole-process peak.

The CUDA controls check exact native-fallback token parity, explicit stopping,
cache-capacity overflow refusal and a fresh request after a deliberate allocation
beyond free GPU memory. The last control must run only on the owned test instance.
It is allocator recovery evidence, not provider interruption or live cancellation
over HTTP. A profiler failure is saved explicitly and does not become a bottleneck
claim. Do not change numerical behavior merely to produce a positive speedup.

For the first JarvisLabs pilot, the approved envelope is one A100 40 GB in IN2,
100 GB storage and a two-hour provider-side auto-pause limit, approximately $1.81
for two hours at the inspected rates. Retrieve and verify artifacts before cleanup.
Storage continues billing after pause. Retain actual GPU/driver, source, model,
input, dependency and billing receipts privately. No new resume claim follows
automatically from a completed run.

## Explicit kernel follow-up

After inspecting the baseline, `serving_kernel_study.py` compares the automatic
packed cache with the existing explicitly requested fused FP32 dequantizer:

```sh
python turboquant_kv/serving_kernel_study.py --manifest "$FROZEN_MANIFEST" --out "$NEW_PRIVATE_RESULTS"
```

This is an exploratory serial experiment, selected after baseline inspection.
It replaces the adapter callable only within each experimental request and
restores it immediately; automatic hardware dispatch is unchanged. Ten numerical
controls cover two seeds, actual context lengths, decode tails and strided cache
views. The predeclared maximum absolute error budget is 0.03125, inherited from
the existing quantization-aware dispatch contract. This does not mean exact
numerical parity. Both actual context shapes are warmed before the rotated
96-request comparison. Raw task scores and failed requests remain in paired
question-clustered evaluation. Reusing the same questions gives no independent
heldout validation of a selected optimization.

## November 1, 2026 A100 pilot

One JarvisLabs NVIDIA A100-PCIE-40GB ran the frozen Qwen2.5-7B input manifest
`327babf4e22d3f855bd20263f49efda9991494c7776bf495c09a3c6b8e803aaf`.
All 96 baseline requests completed, as did all 96 exploratory kernel requests.
Each arm met the fixed 2-second TTFT and 10-second completion targets in this
small serial diagnostic. No HTTP or concurrent-service result is implied.

| Baseline metric | Native | Packed automatic |
|---|---:|---:|
| TTFT p50 / p95, seconds | 0.280 / 0.368 | 0.290 / 0.387 |
| Completion p50 / p95, seconds | 1.092 / 1.254 | 2.581 / 3.369 |
| Resident cache p50, MiB | 169.285 | 86.488 |
| Whole-process allocator peak p50, GiB | 14.767 | 14.713 |
| Mean publisher task score | 0.248371 | 0.248867 |

Cache storage fell approximately 1.96x while completion became slower. This
is not a 1.96x reduction in whole-process GPU memory. The paired score delta
was +0.000496, question-clustered bootstrap 95% interval [0, 0.001186], on only
twelve independent questions. Exact output tokens matched in 26 of 48 pairs;
similar aggregate scores do not imply numerical or quality equivalence.
The native-fallback parity, two-token stop, overflow refusal and controlled
OOM recovery checks all passed. Every automatic packed request selected
`torch_fp32`, as required for the unprofiled A100.

A separate 4096-cap profile exposed 6,944 small reference inverse-rotation
matrix operations and thousands of copies. This justified an exploratory test
of the existing fused FP32 kernel. Its ten numerical controls passed with
maximum absolute difference 0.015625 against a 0.03125 budget. It then lost the
paired end-to-end comparison: completion p50/p95 was 2.327/3.616 seconds versus
1.945/2.623 for automatic packing in the same follow-up run. Mean task score
was 0.242457 versus 0.248867; paired delta -0.006410, bootstrap interval
[-0.019231, 0]. Memory was the same. Automatic dispatch remains unchanged.

Absolute automatic-path latency changed across the two serial runs. Compare
candidate arms within their own run; this single-device pilot does not provide
independent run-level timing uncertainty. The candidate reused inspected
questions and is exploratory. A future optimization needs a new matched run,
separate evaluation inputs, and a demonstrated end-to-end win before dispatch
promotion. More GPU time alone does not justify promoting this losing kernel.
Raw per-request outputs, profiles, environment/source hashes, dependency lock,
model/input receipts and archive verification remain in the private run record.
