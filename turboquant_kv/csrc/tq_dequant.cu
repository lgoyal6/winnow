// STATUS: compiled, run and profiled. On 2026-09-05 this kernel was built and
// measured on an NVIDIA RTX A6000 (sm_86, 84 SMs, driver 595.71.05) with CUDA
// 12.9, torch 2.13.0+cu129 and triton 3.7.1, under a 4 GB per-process memory
// cap. `bench_cuda_kernel.py --check` printed CORRECT. Raw harness output for
// the 24-shape correctness and speed sweeps is in `results/cuda/`. Where a
// number appears below it was measured on that card; where something is still
// intent rather than measurement, it says so.
//
// WHAT THIS IS
// ------------
// A native CUDA transcription of `turboquant_kv/kernel.py`, so the Triton
// kernel finally has something other than PyTorch to be compared against. It
// computes exactly the same thing:
//
//     idx   = unpack(packed)                          // (N, D), BW bits each
//     y     = centroids[idx]                          // (N, D) fp32
//     x     = y @ Pi                                  // (N, D) fp32, D x D
//     out   = (x * norms).to(bf16)                    // (N, D) bf16
//
// with the same layout rules as the Triton version: LSB-first bit packing that
// may straddle a byte boundary, a masked high-byte load so the last index of a
// row never reads past it, and stride-aware addressing so a slice of the
// preallocated (B, H, MAXLEN, NB) cache buffer is read in place rather than
// forcing a `.contiguous()` copy.
//
// THE RESULT, STATED WITHOUT FLATTERY
// -----------------------------------
// At the matched headline shape (B=16, H=4, L=16384, N=1048576, D=128,
// bit_width=6) this kernel takes 10548.3 us. The TF32 Triton arm takes 1298.8
// us, so this kernel is 8.12x SLOWER than that arm. It is 10.15x FASTER than
// the fp32 Triton arm, which is its actual numeric peer at 107047.2 us.
//
// It is exact where the TF32 arm is not. Against the fp32 PyTorch reference at
// that shape this kernel measures 0.000e+00 max absolute error and 0.000e+00
// relative L2; TF32 Triton measures 3.125e-02 and 2.020e-03. Across the whole
// 24-shape sweep in `results/cuda/` this kernel's worst absolute error is
// 7.812e-03, and it is exactly 0 at every shape from N=2048 upward.
//
// THE 8.12x DOES NOT SURVIVE A DECODE STEP
// ----------------------------------------
// Swapping only which kernel `TQPackedLayer._load` calls, on Qwen3-0.6B bf16,
// batch 1, greedy, 32 generated tokens, 3 untimed warmups:
//
//     context      TF32 Triton      this kernel
//       2048       65.651 ms/tok    64.095 ms/tok    <- this kernel is faster
//       8192       66.241 ms/tok    64.483 ms/tok    <- this kernel is faster
//      16384       66.262 ms/tok    88.950 ms/tok    <- 1.34x, not 8.12x
//
// So the honest reading is that an 8.12x microbenchmark gap is worth between
// 0.97x and 1.34x to a user depending on context length, and nothing at all
// below 16k. A batch-1 eager decode loop is launch-bound: nsys puts the extra
// GPU work at 64.3 ms per forward pass against 22.7 ms of extra wall time per
// token, so about 35% of the extra kernel time surfaces and the rest disappears
// into idle the fast kernel was already leaving. For completeness, an
// unquantized fp16 cache beats every quantized arm at every context length here
// (36.708 ms/tok at ctx 16384), so this kernel is a real speedup over the
// PyTorch dequantization path and still a net decode slowdown against simply
// not quantizing.
//
// WHY IT LOSES TO TF32, FROM COUNTS RATHER THAN NARRATIVE
// -------------------------------------------------------
// Nsight Compute could not be used: the driver has RmProfilingAdminOnly=1 and
// there is no root on that box, so `ncu` returns ERR_NVGPUCTRPERM. The numbers
// below come from `ptxas -v`, `cuobjdump -sass`, the CUDA driver occupancy API
// and `nsys profile --trace=cuda`, none of which need the counter permission.
// They are static counts and analytic figures, not hardware counters.
//
//  * NOT occupancy. This kernel: 30 registers/thread, no spills, 4096 B shared
//    per block, 66.67% theoretical occupancy. Both Triton arms: 8.33%. The
//    kernel that loses by 8.12x has eight times the occupancy of the one that
//    beats it, so occupancy cannot be the explanation.
//  * NOT coalescing. `pi[j*D + tx]` is contiguous across a warp; deliberately
//    breaking it into a strided variant costs a further 10.9x, which you cannot
//    lose by breaking a property you did not have.
//  * NOT DRAM bandwidth. 35.2 GB/s achieved against 683.4 GB/s measured on this
//    card, i.e. 5.15% of peak. TF32 Triton reaches 41.8%.
//  * IT IS instruction issue. SASS opcode counts: this kernel issues 0 MMA of
//    any kind and 8 FFMA in a 16-iteration loop; the TF32 arm issues 256 HMMA
//    and 0 FFMA. Per block that is 32 MACs per issued instruction here against
//    1024 there, a 32x issue-efficiency gap, which comfortably contains an
//    8.12x wall-clock gap. Compounding it, Pi is re-read once per FMA, so this
//    kernel issues one global load per 32 MACs against one per 1310 for the
//    TF32 arm, about 41x more load instructions per unit of arithmetic.
//
// This also answers the question the fp32 comparison was posed to settle: the
// fp32 Triton arm spills its accumulator (40 registers, 8192 spill bytes per
// thread, 8259 LDL + 1976 STL in 20680 SASS instructions), so its 12.6x loss to
// PyTorch was Triton's fp32 dot spilling, not the fusion idea being wrong.
//
// WHICH TRITON ARM THIS IS COMPARABLE TO
// --------------------------------------
// The rotation here is a plain fp32 FMA loop on the CUDA cores. That is the
// same numeric regime as `bench_kernel.py --bw N` WITHOUT `--tf32`, and it is
// NOT the same regime as the TF32 tensor-core numbers in the README (the 6.6x
// and 34.5x rows). The 8.12x figure above is quoted against the TF32 arm
// anyway because that is the arm a user would actually be served by, but it is
// a comparison across numeric regimes and the error columns above are the
// reason it is not a like-for-like win for either side.
//
// DESIGN NOTES, AND WHAT THE PROFILE SAID ABOUT THEM
// --------------------------------------------------
//  * One block covers ROWS_PER_BLOCK vectors; blockDim.x == D, so thread tx
//    owns output column tx and every global access along x is coalesced.
//    Confirmed: see the coalescing control above.
//  * TQ_ROWS_PER_BLOCK 8 x 128 = 1024 threads means only one block fits in the
//    1536-thread SM, which is what caps occupancy at 66.67%. Registers are not
//    the constraint: forcing -maxrregcount=16 moved them 30 -> 24 and left
//    occupancy unchanged, while asking the same cubin for a 512-thread block
//    returned 100%. That is a real inefficiency worth roughly 1.5x, not 8x.
//  * The gathered codebook values are staged in shared memory because each of
//    them is read D times by the rotation, once per output column.
//  * Pi (D x D fp32 = 64 KB at D=128) is deliberately NOT staged in shared
//    memory: it exceeds the 48 KB default per-block limit on sm_86 and it is
//    read identically by every block, so it should live in L2. It does - the
//    kernel is nowhere near the DRAM ceiling - but the price is the one global
//    load per FMA counted above, and that is the second half of why this loses.
//  * The codebook is at most 256 fp32 entries, so it is left in global memory
//    and expected to sit in L1 after the first touch. Not separately measured.

#include <cuda_bf16.h>
#include <torch/extension.h>

#define TQ_ROWS_PER_BLOCK 8

template <int D, int BW>
__global__ void tq_dequant_kernel(
    const uint8_t* __restrict__ packed,   // (N_planes, MAXLEN, NB)
    const __half* __restrict__ norms,     // (N,) fp16, as bench_kernel.py passes
    const float* __restrict__ centroids,  // (1 << BW,)
    const float* __restrict__ pi,         // (D, D)
    __nv_bfloat16* __restrict__ out,      // (N, D)
    int N, int L, int MAXLEN, int NB) {
  __shared__ float y_sh[TQ_ROWS_PER_BLOCK][D];

  const int tx = threadIdx.x;                 // output column, 0 .. D-1
  const int ty = threadIdx.y;                 // row within the block
  const int row = blockIdx.x * TQ_ROWS_PER_BLOCK + ty;

  // Stride-aware source row: rows are laid out (plane, position) and the
  // buffer's position stride is MAXLEN, not L, whenever we are looking at a
  // slice of the preallocated cache.
  const int plane = row / L;
  const int pos = row % L;
  const long src_row = (long)plane * MAXLEN + pos;

  float y = 0.0f;
  if (row < N) {
    // Unpack one BW-bit index, LSB-first, possibly straddling a byte.
    const int bit_off = tx * BW;
    const int byte0 = bit_off >> 3;
    const int sh = bit_off & 7;
    const uint8_t* base = packed + src_row * (long)NB;
    const int b0 = (int)base[byte0];
    // The high byte only exists when the index crosses into it; without this
    // guard the last index of a row reads one byte past the row.
    const int b1 = ((byte0 + 1) < NB) ? (int)base[byte0 + 1] : 0;
    const int idx = ((b0 >> sh) | (b1 << (8 - sh))) & ((1 << BW) - 1);
    y = centroids[idx];
  }
  y_sh[ty][tx] = y;
  __syncthreads();

  if (row >= N) return;

  // Inverse rotation, one output column per thread: acc = y_sh[ty] . Pi[:, tx]
  float acc = 0.0f;
#pragma unroll 8
  for (int j = 0; j < D; ++j) {
    acc = fmaf(y_sh[ty][j], pi[j * D + tx], acc);
  }
  acc *= __half2float(norms[src_row]);
  out[(long)row * D + tx] = __float2bfloat16(acc);
}

// Dispatch over the bit widths packing.py supports. BW is a template parameter
// so the shift arithmetic and the mask fold into constants.
#define TQ_LAUNCH(BWV)                                                     \
  tq_dequant_kernel<128, BWV><<<grid, block>>>(                            \
      packed.data_ptr<uint8_t>(),                                          \
      reinterpret_cast<const __half*>(norms.data_ptr<at::Half>()),         \
      centroids.data_ptr<float>(), pi.data_ptr<float>(),                   \
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),      \
      N, L, MAXLEN, NB);                                                   \
  break;

torch::Tensor tq_dequant_cuda(torch::Tensor packed, torch::Tensor norms,
                              torch::Tensor centroids, torch::Tensor pi,
                              torch::Tensor out, int64_t bit_width,
                              int64_t head_dim, int64_t N, int64_t L,
                              int64_t MAXLEN) {
  TORCH_CHECK(head_dim == 128, "this kernel is specialised for head_dim=128");
  TORCH_CHECK(centroids.scalar_type() == torch::kFloat32, "centroids must be fp32");
  TORCH_CHECK(pi.scalar_type() == torch::kFloat32, "Pi must be fp32");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16, "out must be bf16");
  TORCH_CHECK(norms.scalar_type() == torch::kHalf, "norms must be fp16");

  const int NB = (int)packed.size(-1);
  dim3 block(128, TQ_ROWS_PER_BLOCK);
  dim3 grid((N + TQ_ROWS_PER_BLOCK - 1) / TQ_ROWS_PER_BLOCK);

  switch (bit_width) {
    case 2: TQ_LAUNCH(2)
    case 3: TQ_LAUNCH(3)
    case 4: TQ_LAUNCH(4)
    case 5: TQ_LAUNCH(5)
    case 6: TQ_LAUNCH(6)
    case 8: TQ_LAUNCH(8)
    default: TORCH_CHECK(false, "unsupported bit_width ", bit_width);
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tq_dequant_cuda", &tq_dequant_cuda, "fused TurboQuant dequant (CUDA)");
}
