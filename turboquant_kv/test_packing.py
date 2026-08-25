"""pack/unpack must be an exact round-trip, and the 4-bit fast path must agree
with the general bit path bit-for-bit."""
import torch

from packing import SUPPORTED, pack, packed_bytes, unpack


def _general_pack(idx, bw):
    """The general path, with the bit_width==4 shortcut disabled."""
    D = idx.shape[-1]
    nbytes = packed_bytes(D, bw)
    sh = torch.arange(bw, device=idx.device, dtype=torch.uint8)
    bits = (idx.reshape(-1, D).unsqueeze(-1) >> sh) & 1
    bits = bits.reshape(-1, nbytes, 8)
    w = (1 << torch.arange(8, device=idx.device, dtype=torch.int16))
    return (bits.to(torch.int16) * w).sum(-1).to(torch.uint8).reshape(
        *idx.shape[:-1], nbytes)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    D = 128
    ok = True
    for bw in SUPPORTED:
        nb = packed_bytes(D, bw)
        idx = torch.randint(0, 2 ** bw, (7, 4, 33, D), dtype=torch.uint8,
                            device=dev)
        p = pack(idx, bw)
        assert p.shape == (7, 4, 33, nb), (bw, p.shape)
        back = unpack(p, bw, D)
        exact = torch.equal(back, idx)
        # bytes per vector, against the unpacked uint8 the old path stored
        ratio = D / nb
        agree = True
        if bw != 8:
            agree = torch.equal(pack(idx, bw), _general_pack(idx, bw))
        print(f"  bw={bw}: {D} idx -> {nb} B ({ratio:.2f}x smaller than uint8) "
              f"roundtrip_exact={exact} fastpath_agrees={agree}")
        ok = ok and exact and agree
    # 3-bit must be exactly 48 bytes, not 64: straddling byte boundaries is the
    # whole point of packing bit-exactly rather than per-nibble.
    assert packed_bytes(128, 3) == 48, packed_bytes(128, 3)
    assert packed_bytes(128, 4) == 64
    print(f"\nall exact: {ok}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
