"""The packed path must reconstruct what the original path reconstructs.

Two claims to check before any memory number is worth reporting:

1. Bit-packing is lossless, so packed dequantization differs from the original
   `centroids[idx.long()] @ Pi * norms` only through the fp16 norm store.
2. That fp16 norm error is far below the quantization error it sits inside, so
   storing norms in fp16 rather than fp32 is not a silent accuracy regression.
"""
import torch

from cache import TurboQuantMSE
from packing import pack, unpack


@torch.no_grad()
def original_dequantize(tq, idx, norms_fp32):
    """The dequantization path as it exists today, verbatim in behaviour."""
    flat_idx = idx.reshape(-1, tq.head_dim)
    y_hat = tq.centroids[flat_idx.long()]
    x_hat = y_hat @ tq.Pi
    x_hat = x_hat * norms_fp32.reshape(-1, 1)
    return x_hat.view(idx.shape)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    D = 128
    print(f"{'bw':>3} {'pack_exact':>11} {'vs_original':>12} {'quant_err':>10} "
          f"{'norm_err/quant_err':>19}")
    all_ok = True
    for bw in (3, 4):
        tq = TurboQuantMSE(bw, D, dev)
        x = torch.randn(2, 4, 64, D, device=dev)

        idx, norms = tq.quantize(x)

        # 1. packing is lossless
        pack_exact = torch.equal(unpack(pack(idx, bw), bw, D), idx)

        # 2. reference reconstruction, fp32 norms, original gather
        ref = original_dequantize(tq, idx, norms)

        # 3. packed path: unpack + int32 index_select + fp16 norms
        out = torch.empty_like(ref)
        tq.dequantize_into(unpack(pack(idx, bw), bw, D),
                           norms.to(torch.float16), out)

        delta = (out - ref).abs().max().item()
        # The error the reconstruction already carries, for scale.
        quant_err = (ref - x).abs().max().item()
        ratio = delta / quant_err if quant_err else float("inf")
        ok = pack_exact and ratio < 0.01
        all_ok = all_ok and ok
        print(f"{bw:>3} {str(pack_exact):>11} {delta:>12.3e} {quant_err:>10.3e} "
              f"{ratio:>19.2e}")

    print(f"\npacking lossless and fp16 norms negligible: {all_ok}")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
