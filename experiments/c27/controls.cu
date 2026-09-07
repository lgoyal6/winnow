// Negative controls for the C27 profiling claims.
//  bwtest   : a pure streaming copy, to measure this card DRAM peak with the
//             same clock and the same timing method used for the kernels. If
//             the bandwidth arithmetic in matched.py were wrong, this would not
//             land near the vendor number.
//  coalesced/strided: the SAME arithmetic and the SAME byte count, with only
//             the Pi access pattern changed. coalesced reads pi[j*D+tx]
//             (consecutive across a warp); strided reads pi[tx*D+j] (512 B
//             apart across a warp). This is the control for "the native kernel
//             global accesses are coalesced": if they were not already
//             coalesced, breaking them could not make it slower.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void copyk(const float4* __restrict__ a, float4* __restrict__ b, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) b[i] = a[i];
}

#define D 128
#define RPB 8
template <bool STRIDED>
__global__ void rot(const float* __restrict__ pi, const float* __restrict__ yin,
                    float* __restrict__ out, int N) {
  __shared__ float y_sh[RPB][D];
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.x * RPB + ty;
  y_sh[ty][tx] = (row < N) ? yin[(long)row * D + tx] : 0.f;
  __syncthreads();
  if (row >= N) return;
  float acc = 0.f;
#pragma unroll 8
  for (int j = 0; j < D; ++j)
    acc = fmaf(y_sh[ty][j], STRIDED ? pi[tx * D + j] : pi[j * D + tx], acc);
  out[(long)row * D + tx] = acc;
}

static float bench(void (*launch)(void*), void* arg, int reps) { return 0; }

int main() {
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  // ---- streaming copy: measured DRAM peak -------------------------------
  size_t bytes = 512ull << 20;              // 512 MB each way, 1 GB touched
  size_t n4 = bytes / sizeof(float4);
  float4 *a, *b; cudaMalloc(&a, bytes); cudaMalloc(&b, bytes); cudaMemset(a, 1, bytes);
  for (int i = 0; i < 3; ++i) copyk<<<(n4 + 255) / 256, 256>>>(a, b, n4);
  cudaDeviceSynchronize();
  cudaEventRecord(s);
  for (int i = 0; i < 20; ++i) copyk<<<(n4 + 255) / 256, 256>>>(a, b, n4);
  cudaEventRecord(e); cudaEventSynchronize(e);
  float ms; cudaEventElapsedTime(&ms, s, e);
  double gbs = 2.0 * bytes * 20 / (ms / 1e3) / 1e9;
  printf("MEASURED_DRAM_PEAK_COPY_GBs %.1f  (%.3f ms for 20 x %zu MB r+w)\n", gbs, ms / 20, bytes >> 20);
  cudaFree(a); cudaFree(b);

  // ---- coalescing control ------------------------------------------------
  int N = 1 << 20;
  float *pi, *yin, *out;
  cudaMalloc(&pi, D * D * 4); cudaMalloc(&yin, (size_t)N * D * 4); cudaMalloc(&out, (size_t)N * D * 4);
  cudaMemset(pi, 0, D * D * 4); cudaMemset(yin, 0, (size_t)N * D * 4);
  dim3 blk(D, RPB), grd((N + RPB - 1) / RPB);
  for (int mode = 0; mode < 2; ++mode) {
    for (int i = 0; i < 5; ++i) {
      if (mode) rot<true><<<grd, blk>>>(pi, yin, out, N); else rot<false><<<grd, blk>>>(pi, yin, out, N);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    for (int i = 0; i < 10; ++i) {
      if (mode) rot<true><<<grd, blk>>>(pi, yin, out, N); else rot<false><<<grd, blk>>>(pi, yin, out, N);
    }
    cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms, s, e);
    printf("ROT_%s_us %.1f\n", mode ? "STRIDED_pi_tx_D_plus_j" : "COALESCED_pi_j_D_plus_tx", ms / 10 * 1e3);
  }
  printf("err=%s\n", cudaGetErrorString(cudaGetLastError()));
  cudaFree(pi); cudaFree(yin); cudaFree(out);
  return 0;
}
