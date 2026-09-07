"""Matched-shape kernel comparison with achieved bandwidth, plus the negative
controls. One shape: the README headline shape B=16 H=4 L=16384 N=1048576,
D=128, BW=6. Every arm computes the same thing and its error is reported next
to its time."""
import sys, json, argparse, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "turboquant_kv"))
import torch, triton
from cache import TurboQuantMSE
from packing import pack, packed_bytes, unpack
from kernel import tq_dequant
from torch.utils.cpp_extension import load

OUT = HERE / "out"
CU = ROOT / "turboquant_kv" / "csrc" / "tq_dequant.cu"
ap = argparse.ArgumentParser()
ap.add_argument("--B", type=int, default=16); ap.add_argument("--L", type=int, default=16384)
ap.add_argument("--reps", type=int, default=5); ap.add_argument("--inner", type=int, default=5)
ap.add_argument("--nsys", action="store_true", help="one pass of each, for tracing")
ap.add_argument("--tag", default="matched")
ap.add_argument("--ref-chunk", type=int, default=None,
                help="batch slice for the accuracy reference, to stay under 4 GB")
ap.add_argument("--time-torch", action="store_true",
                help="also time the unchunked PyTorch arm (needs >4 GB at N=1048576)")
a = ap.parse_args()

dev, D, BW, H = "cuda", 128, 6, 4
# HARD-ENFORCE the announced 4 GB envelope: torch raises OOM rather than
# silently exceeding it.
torch.cuda.set_per_process_memory_fraction(4.0 / 48.0, 0)
torch.manual_seed(0)
tq = TurboQuantMSE(BW, D, dev)
NB = packed_bytes(D, BW)
ext = load(name="tq_dequant_cuda_ext", sources=[str(CU)], verbose=False)

B, L = a.B, a.L
N = B * H * L
# Build the packed cache one batch slice at a time. Materialising the whole
# (B,H,L,D) fp32 tensor and its packing intermediates at once needs >4 GB, and
# the 4 GB cap above turns that into an OOM rather than a quiet overrun.
packed = torch.empty(B, H, L, NB, dtype=torch.uint8, device=dev)
nrm16 = torch.empty(B, H, L, dtype=torch.float16, device=dev)
for b in range(B):
    xb = torch.randn(1, H, L, D, device=dev)
    ib, nb_ = tq.quantize(xb)
    packed[b:b+1].copy_(pack(ib, BW))
    nrm16[b:b+1].copy_(nb_.to(torch.float16))
    del xb, ib, nb_
torch.cuda.empty_cache()
ref = torch.empty(B, H, L, D, dtype=torch.bfloat16, device=dev)
got = torch.empty_like(ref)

@torch.no_grad()
def torch_path(out, chunk=None):
    """The PyTorch reference. `chunk` slices the batch so the fp32
    intermediates fit the announced 4 GB envelope; chunk=None is the original
    single-shot path and is the only form whose timing is comparable to the
    README table."""
    step = chunk or B
    for b0 in range(0, B, step):
        pk = packed[b0:b0+step]
        i = unpack(pk, BW, D)
        y = tq.centroids.index_select(0, i.reshape(-1, D).reshape(-1).to(torch.int32)).view(-1, D)
        z = y @ tq.Pi
        z *= nrm16[b0:b0+step].reshape(-1, 1).to(z.dtype)
        out[b0:b0+step].copy_(z.view(out[b0:b0+step].shape))
        del i, y, z
    return out

def cuda_path(out):
    return ext.tq_dequant_cuda(packed, nrm16, tq.centroids, tq.Pi, out, BW, D, N, L, L)
def triton_path(out, tf32):
    return tq_dequant(packed, nrm16, tq.centroids, tq.Pi, BW, D, out=out, allow_tf32=tf32)

def timed(fn, inner, reps, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True); best = []
    for _ in range(reps):
        s.record()
        for _ in range(inner): fn()
        e.record(); e.synchronize(); best.append(s.elapsed_time(e) / 1e3 / inner)
    return min(best)

# --- exact DRAM-visible bytes this operation must move -----------------------
bytes_in  = N * NB + N * 2 + (1 << BW) * 4 + D * D * 4     # packed, norms, codebook, Pi
bytes_out = N * D * 2                                       # bf16 output
BYTES = bytes_in + bytes_out
MACS  = N * D * D                                           # (N,D) @ (D,D)

torch_path(ref, chunk=a.ref_chunk); torch.cuda.empty_cache()
ARMS = [("triton_fp32", lambda o: triton_path(o, False), False),
        ("triton_tf32", lambda o: triton_path(o, True), True),
        ("native_cuda_fp32", lambda o: cuda_path(o), None)]
if a.time_torch:
    ARMS.insert(0, ("torch_fp32", lambda o: torch_path(o), None))
arms = {}
for name, fn, tf32 in ARMS:
    got.zero_(); fn(got)
    err = (got.float() - ref.float()).abs().max().item()
    rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
    if a.nsys:
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(got); torch.cuda.synchronize()
        t = time.perf_counter() - t0
    else:
        t = timed(lambda: fn(got), a.inner, a.reps)
    arms[name] = dict(us=t * 1e6, max_abs_err=err, rel_l2_err=rel,
                      achieved_GBs=BYTES / t / 1e9, eff_TFLOPs=2 * MACS / t / 1e12)
    print("%-18s %10.1f us   max|err|=%9.3e  relL2=%9.3e  %7.1f GB/s  %6.2f TFLOP/s"
          % (name, t * 1e6, err, rel, BYTES / t / 1e9, 2 * MACS / t / 1e12))

res = dict(B=B, H=H, L=L, N=N, D=D, BW=BW, NB=NB, bytes_moved=BYTES, macs=MACS, arms=arms)
if not a.nsys:
    json.dump(res, open(OUT / (a.tag + ".json"), "w"), indent=2)
print("\nbytes moved per call: %.1f MB   MACs: %.3e" % (BYTES / 1e6, MACS))
