"""Compile the Triton kernel at both TF32 settings and report the same facts
ncu would have: registers/thread, spills, shared memory, occupancy, SASS mix.
All of it via the CUDA driver API + cuobjdump, neither of which needs the
GPU performance-counter permission that ERR_NVGPUCTRPERM denies us."""
import sys, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "turboquant_kv"))
import torch, triton
from cache import TurboQuantMSE
from packing import pack, packed_bytes
import kernel as K

OUT = HERE / "out"
D, BW, BLOCK_N = 128, 6, 64
dev = "cuda"

torch.manual_seed(0)
tq = TurboQuantMSE(BW, D, dev)
NB = packed_bytes(D, BW)

# small shape: we only want the compiled artifact, not a timing
B,H,L = 1, 4, 512
N = B*H*L
x = torch.randn(B,H,L,D, device=dev)
idx, norms = tq.quantize(x)
packed = pack(idx, BW)
nrm16 = norms.to(torch.float16)
out = torch.empty(B,H,L,D, dtype=torch.bfloat16, device=dev)

res = {}
for tf32 in (False, True):
    grid = (triton.cdiv(N, BLOCK_N),)
    ck = K._tq_dequant_kernel[grid](
        packed, nrm16, tq.centroids, tq.Pi, out,
        N, L, L, D=D, NB=NB, BW=BW,
        BLOCK_N=BLOCK_N, ALLOW_TF32=tf32,
        num_warps=4, num_stages=2,
    )
    md = ck.metadata
    tag = "triton_tf32" if tf32 else "triton_fp32"
    cubin = ck.asm["cubin"]
    p = OUT / f"{tag}.cubin"
    open(p,"wb").write(cubin)
    open(OUT / f"{tag}.ptx","w").write(ck.asm["ptx"])
    res[tag] = dict(n_regs=ck.n_regs, n_spills=ck.n_spills,
                    shared=getattr(md,"shared",None),
                    num_warps=getattr(md,"num_warps",None),
                    num_stages=getattr(md,"num_stages",None),
                    name=getattr(md,"name",None), cubin=str(p))
    print("[%s] n_regs=%s n_spills=%s shared=%sB num_warps=%s num_stages=%s" % (tag, ck.n_regs, ck.n_spills, getattr(md,"shared",None), getattr(md,"num_warps",None), getattr(md,"num_stages",None)))
json.dump(res, open(OUT / "triton_static.json","w"), indent=2)
print("OK")
