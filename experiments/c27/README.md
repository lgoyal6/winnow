# C27 native CUDA and Nsight experiment

This directory makes the retained native-CUDA comparison reviewable from the
owner repository. The main run used one RTX A6000 48 GB (sm_86), torch
2.13.0+cu129, Triton 3.7.1, CUDA toolkit 12.9, and gcc 14.3.0 on 2026-09-05.
Each GPU script enforces a 4 GiB process-memory fraction.

The result is less flattering than the kernel-only number:

- At `B=16, H=4, L=16384, D=128, BW=6`, Triton TF32 took 1,298.8 us and
  native CUDA fp32 took 10,548.3 us. Native CUDA was 8.12x slower, but exact;
  TF32 had max absolute error 0.03125 and relative L2 error 0.00202.
- In Qwen3-0.6B batch-1 eager decode, native CUDA was slightly faster than
  Triton TF32 at contexts 2,048 and 8,192. It was 88.95 versus 66.26 ms/token
  at context 16,384, a 1.34x loss rather than 8.12x.
- The unquantized bf16 cache was fastest at every tested context, 36.71
  ms/token at context 16,384. For this model and batch size, quantization saved
  memory and reduced decode throughput.
- Nsight Systems recorded 672 dequant launches in each 16,384-token trace.
  Dequant consumed 141.6 ms with Triton TF32 and 913.5 ms with native CUDA.

`out/` retains the JSON results, exported Nsight kernel-summary CSVs, the two
raw `.nsys-rep` traces, and the cubin/PTX files used for occupancy and static
instruction analysis. `SHA256SUMS` covers every retained artifact. No model
weights are included.

## Review without a GPU

```bash
python3 experiments/c27/verify_results.py --negative-control
```

This recomputes shape, byte, and MAC invariants; checks numerical constraints,
the 4 GiB envelope, decode ordering, occupancy, the two Nsight summaries, and
all retained digests. Its negative control corrupts the native-CUDA accuracy
field in a temporary copy and proves the verifier rejects it.

## Reproduce on an NVIDIA GPU

The source under `turboquant_kv/` and these scripts expect a CUDA-enabled torch
build, Triton, a matching CUDA toolkit with `nvcc`, and a supported C++
compiler. The recorded environment was:

```text
GPU: RTX A6000 48 GB, sm_86, 84 SMs
driver: 595.71.05
runtime/toolkit: CUDA 13.2 / 12.9
Python: 3.12
torch: 2.13.0+cu129
Triton: 3.7.1
gcc: 14.3.0
```

Run the matched headline shape and the smaller arm that also fits the PyTorch
reference under the cap:

```bash
python experiments/c27/matched.py \
  --B 16 --L 16384 --ref-chunk 1 --reps 5 --inner 5 \
  --tag matched_N1048576
python experiments/c27/matched.py \
  --B 4 --L 16384 --ref-chunk 1 --time-torch --reps 5 --inner 5 \
  --tag matched_N262144
```

The decode model is pinned in `model_guard.py` at commit
`c1899de289a04d12100db370d81485cdf75e47ca`; its exact consumed-file digests
are in `model_artifacts.py`. The guarded loader rejects drift before model
activation.

```bash
python experiments/c27/decode.py \
  --model Qwen/Qwen3-0.6B --ctxs 2048 8192 16384 --gen 32 \
  --out decode_0.6b.json
```

Capture and export the context-16,384 traces one arm at a time:

```bash
nsys profile --trace=cuda --output=experiments/c27/out/decode_triton_tf32 \
  python experiments/c27/decode.py --ctxs 16384 --gen 8 \
  --arms triton_tf32 --out nsys_triton_tf32.json
nsys profile --trace=cuda --output=experiments/c27/out/decode_native_cuda_fp32 \
  python experiments/c27/decode.py --ctxs 16384 --gen 8 \
  --arms native_cuda_fp32 --out nsys_native_cuda_fp32.json
nsys stats --report cuda_gpu_kern_sum --format csv \
  experiments/c27/out/decode_triton_tf32.nsys-rep
nsys stats --report cuda_gpu_kern_sum --format csv \
  experiments/c27/out/decode_native_cuda_fp32.nsys-rep
```

`triton_static.py`, `occupancy.py`, and `sassmix.py` reproduce the static
register, spill, occupancy, and instruction-mix analysis. `controls.cu`
contains the measured-bandwidth and deliberately strided-access controls.

## Profiler limit

Nsight Compute 2025.2.1 was installed and invoked, but the shared host has
`RmProfilingAdminOnly: 1`. Both `ncu` and Nsight Systems hardware-counter mode
returned `ERR_NVGPUCTRPERM`. The retained `.nsys-rep` files use CUDA activity
tracing, which worked. Occupancy comes from the CUDA driver API, and instruction
counts come from `cuobjdump`; neither is a substitute for achieved occupancy,
DRAM traffic, tensor-pipe utilization, or stall counters. Those four dynamic
counter classes remain unmeasured.

## Fresh owner-port check

On 2026-09-06, the bounded harness was rerun on the same A6000 at
`B=1, H=4, L=2048, D=128, BW=6`. The GPU returned to 0 percent utilization and
28 MiB after the run. `out/fresh_owner_port_check.json` records:

| arm | time (us) | max abs error | relative L2 error |
|---|---:|---:|---:|
| Triton fp32 | 882.1 | 0 | 0 |
| Triton TF32 | 40.0 | 0.03125 | 0.002012 |
| native CUDA fp32 | 92.7 | 0 | 0 |

This fresh small-shape run verifies compilation, GPU execution, and the
correctness controls. It does not replace the retained headline-shape or full
decode measurements.

The remote and owner files have different leading comments, but their compiled
bodies, from the first CUDA include onward, share SHA-256
`df0cacf56f639dce2087834f62515b70cac5c3671d7b6fd0c93b103334623b3e`.
