// TurboQuant dequantize: fused codebook gather + per-vector norm scaling.
//
// STATUS: UNRUN. This file has never been compiled or executed. The host this
// was written on has no GPU -- `nvidia-smi` and `nvcc` are both absent -- so
// there are NO performance numbers for it anywhere in this repo, and none may
// be quoted until it has actually been built and benchmarked on a CUDA device.
// `bench_tq_dequant.py` is the harness that produces those numbers; it refuses
// to run without CUDA rather than estimating.
//
// WHAT IT REPLACES
// ----------------
// `TurboQuantMSE.dequantize` (turboquant_poc.py) currently does:
//
//     y_hat = self.centroids[flat_idx.long()]      # (1) uint8 -> int64 copy,
//                                                  #     then a gather
//     x_hat = y_hat @ self.Pi                      # (2) GEMM
//     x_hat = x_hat * norms.reshape(-1, 1)         # (3) row scale
//
// Step (1) materialises an int64 index tensor 8x the size of the uint8 codes
// and then a second fp32 tensor for y_hat; step (3) is another full pass over
// the output. On the decode path this runs for every layer on every token, and
// the repo's own note (turboquant-poc/README.md) is that the pure-PyTorch
// dequantize path is why token rate sits below fp16.
//
// THE FUSION
// ----------
// Row scaling COMMUTES with the rotation, because the rotation is a row-wise
// linear map:
//
//     (y_hat @ Pi) * norms[:, None]  ==  (y_hat * norms[:, None]) @ Pi
//
// So the scale can be folded into the gather, leaving exactly one GEMM and no
// int64 index copy:
//
//     y_scaled[r][c] = centroids[idx[r][c]] * norms[r]      <- this kernel
//     x_hat          = y_scaled @ Pi                        <- cuBLAS, unchanged
//
// The codebook is at most 2**bit_width entries (256 at 8 bits, 16 at 4 bits),
// so it is staged in shared memory once per block and every lookup after that
// is a shared-memory read instead of a global gather.
//
// The equivalence above is verified numerically against the existing
// implementation on CPU by `bench_tq_dequant.py --check-math-only`, which needs
// no GPU. The KERNEL is still unrun.

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kMaxCodebook = 256;  // 8-bit codes are the widest TurboQuant uses

// One thread per output element. `rows` is the flattened vector count
// (batch * heads * seq_len) and `dim` is head_dim.
template <typename scalar_t>
__global__ void tq_dequant_gather_scale_kernel(
    const uint8_t* __restrict__ idx,        // (rows, dim)
    const float* __restrict__ centroids,    // (n_levels,)
    const float* __restrict__ norms,        // (rows,)
    scalar_t* __restrict__ out,             // (rows, dim)
    const long rows,
    const int dim,
    const int n_levels) {
  extern __shared__ float s_centroids[];

  for (int i = threadIdx.x; i < n_levels; i += blockDim.x) {
    s_centroids[i] = centroids[i];
  }
  __syncthreads();

  const long total = rows * static_cast<long>(dim);
  const long stride = static_cast<long>(blockDim.x) * gridDim.x;
  for (long t = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
       t < total; t += stride) {
    const long row = t / dim;
    // uint8 load: no int64 index tensor is ever materialised.
    const float centroid = s_centroids[static_cast<int>(idx[t])];
    out[t] = static_cast<scalar_t>(centroid * norms[row]);
  }
}

}  // namespace

// Returns y_scaled = centroids[idx] * norms[:, None], ready for a single
// `y_scaled @ Pi` GEMM. Caller keeps the GEMM in cuBLAS.
torch::Tensor tq_dequant_gather_scale(
    torch::Tensor idx,        // (rows, dim), uint8, contiguous, CUDA
    torch::Tensor centroids,  // (n_levels,), float32, CUDA
    torch::Tensor norms,      // (rows,), float32, CUDA
    c10::ScalarType out_dtype) {
  TORCH_CHECK(idx.is_cuda() && centroids.is_cuda() && norms.is_cuda(),
              "tq_dequant_gather_scale: all inputs must be CUDA tensors");
  TORCH_CHECK(idx.scalar_type() == torch::kUInt8, "idx must be uint8");
  TORCH_CHECK(centroids.scalar_type() == torch::kFloat32, "centroids must be float32");
  TORCH_CHECK(norms.scalar_type() == torch::kFloat32, "norms must be float32");
  TORCH_CHECK(idx.dim() == 2, "idx must be 2-D (rows, dim)");
  TORCH_CHECK(norms.numel() == idx.size(0), "norms must have one entry per row");
  TORCH_CHECK(centroids.numel() <= kMaxCodebook,
              "codebook larger than ", kMaxCodebook, " entries");

  idx = idx.contiguous();
  centroids = centroids.contiguous();
  norms = norms.contiguous();

  const long rows = idx.size(0);
  const int dim = static_cast<int>(idx.size(1));
  const int n_levels = static_cast<int>(centroids.numel());

  auto out = torch::empty({rows, dim}, idx.options().dtype(out_dtype));

  const int threads = 256;
  const long total = rows * static_cast<long>(dim);
  const int blocks = static_cast<int>(std::min<long>((total + threads - 1) / threads, 65535L));
  const size_t shmem = static_cast<size_t>(n_levels) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, out_dtype,
      "tq_dequant_gather_scale", [&] {
        tq_dequant_gather_scale_kernel<scalar_t>
            <<<blocks, threads, shmem, stream>>>(
                idx.data_ptr<uint8_t>(),
                centroids.data_ptr<float>(),
                norms.data_ptr<float>(),
                out.data_ptr<scalar_t>(),
                rows, dim, n_levels);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tq_dequant_gather_scale", &tq_dequant_gather_scale,
        "TurboQuant fused codebook gather + per-vector norm scale (CUDA)");
}
