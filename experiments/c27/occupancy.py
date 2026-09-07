"""Registers, spill (local) bytes, shared bytes and OCCUPANCY for all three
kernels, straight from the CUDA driver API.

Why this and not ncu: cuFuncGetAttribute and cuOccupancyMaxActiveBlocksPer-
Multiprocessor are ordinary driver entry points, not hardware performance
counters, so RmProfilingAdminOnly=1 does not block them. They return the same
launch__registers_per_thread / launch__occupancy_limit_* numbers Nsight Compute
would print, because Nsight reads them from the same place.
"""
import ctypes, json, subprocess
from pathlib import Path

LIBCUDA = "/run/opengl-driver/lib/libcuda.so"
cu = ctypes.CDLL(LIBCUDA)
OUT = Path(__file__).resolve().parent / "out"

def chk(r, what):
    if r != 0:
        p = ctypes.c_char_p()
        cu.cuGetErrorString(r, ctypes.byref(p))
        raise RuntimeError("%s failed: %s (%d)" % (what, p.value, r))

chk(cu.cuInit(0), "cuInit")
devt = ctypes.c_int()
chk(cu.cuDeviceGet(ctypes.byref(devt), 0), "cuDeviceGet")
ctx = ctypes.c_void_p()
chk(cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, devt), "cuCtxCreate")

def dattr(n):
    v = ctypes.c_int()
    cu.cuDeviceGetAttribute(ctypes.byref(v), n, devt)
    return v.value

# CUdevice_attribute enum values
A = dict(MAX_THREADS_PER_SM=39, MAX_SHARED_PER_SM=(81), REGS_PER_SM=(82),
         WARP_SIZE=10, SM_COUNT=16, MAX_BLOCKS_PER_SM=106,
         MAX_SHARED_PER_BLOCK_OPTIN=97, MAX_SHARED_PER_BLOCK=8, REGS_PER_BLOCK=12)
dev_info = {k: dattr(v) for k, v in A.items()}

# CUfunction_attribute enum
F = dict(MAX_THREADS_PER_BLOCK=0, SHARED_SIZE_BYTES=1, CONST_SIZE_BYTES=2,
         LOCAL_SIZE_BYTES=3, NUM_REGS=4, PTX_VERSION=5, BINARY_VERSION=6)

def analyse(label, cubin_path, want_sub, block_threads, dynamic_smem=0):
    data = open(cubin_path, "rb").read()
    mod = ctypes.c_void_p()
    chk(cu.cuModuleLoadData(ctypes.byref(mod), data), "cuModuleLoadData " + label)
    # find the mangled name
    import re
    o = subprocess.run(["cuobjdump", "-sass", cubin_path], capture_output=True, text=True)
    names = re.findall(r"Function : (\S+)", o.stdout)
    cand = [n for n in names if want_sub in n] or names
    name = cand[0]
    fn = ctypes.c_void_p()
    chk(cu.cuModuleGetFunction(ctypes.byref(fn), mod, name.encode()), "cuModuleGetFunction " + name)
    at = {}
    for k, v in F.items():
        x = ctypes.c_int()
        cu.cuFuncGetAttribute(ctypes.byref(x), v, fn)
        at[k] = x.value
    # opt in to the full 100 KB dynamic shared window (what Triton does)
    if dynamic_smem > 48 * 1024:
        cu.cuFuncSetAttribute(fn, 8, ctypes.c_int(dev_info["MAX_SHARED_PER_BLOCK_OPTIN"]))
    nb = ctypes.c_int()
    chk(cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        ctypes.byref(nb), fn, ctypes.c_int(block_threads),
        ctypes.c_size_t(dynamic_smem)), "cuOccupancy " + label)
    warps_per_block = block_threads // 32
    max_warps = dev_info["MAX_THREADS_PER_SM"] // 32
    active_warps = nb.value * warps_per_block
    occ = 100.0 * active_warps / max_warps
    r = dict(label=label, mangled=name, regs_per_thread=at["NUM_REGS"],
             spill_local_bytes_per_thread=at["LOCAL_SIZE_BYTES"],
             static_smem_bytes=at["SHARED_SIZE_BYTES"], dynamic_smem_bytes=dynamic_smem,
             block_threads=block_threads, warps_per_block=warps_per_block,
             max_active_blocks_per_sm=nb.value, active_warps_per_sm=active_warps,
             max_warps_per_sm=max_warps, theoretical_occupancy_pct=round(occ, 2))
    print(json.dumps(r, indent=2))
    return r

print("DEVICE:", json.dumps(dev_info))
res = {"device": dev_info, "kernels": []}
res["kernels"].append(analyse("native_cuda_fp32", str(OUT / "tq_dequant.cubin"),
                              "Li128ELi6E", 128 * 8, 0))
ts = json.load(open(OUT / "triton_static.json"))
for tag in ("triton_fp32", "triton_tf32"):
    res["kernels"].append(analyse(tag, str(OUT / (tag + ".cubin")), "tq_dequant",
                                  ts[tag]["num_warps"] * 32, ts[tag]["shared"]))
json.dump(res, open(OUT / "occupancy.json", "w"), indent=2)
print("OK")
