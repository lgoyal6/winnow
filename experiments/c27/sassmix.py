"""Static SASS instruction mix. Counts opcodes in a cubin/so via cuobjdump."""
import re, sys, subprocess, collections, os, json

def sass_mix(path, filter_sub=None):
    out = subprocess.run(["cuobjdump","-sass",path],capture_output=True,text=True)
    if out.returncode != 0:
        return None, out.stderr[:400]
    cur=None; mixes=collections.defaultdict(collections.Counter)
    for line in out.stdout.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m: cur = m.group(1); continue
        # SASS body lines look like:  /*0010*/  IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
        m = re.match(r"\s*/\*[0-9a-f]+\*/\s+(@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)", line)
        if m and cur: mixes[cur][m.group(2).split(".")[0]] += 1
    return mixes, None

if __name__ == "__main__":
    path = sys.argv[1]
    want = sys.argv[2] if len(sys.argv)>2 else None
    mixes, err = sass_mix(path)
    if err: print("ERR", err); sys.exit(1)
    for fn, c in mixes.items():
        if want and want not in fn: continue
        tot = sum(c.values())
        print(f"--- {fn}  ({tot} SASS instructions) ---")
        # tensor-core ops on sm_86
        tc = sum(v for k,v in c.items() if k in ("HMMA","IMMA","BMMA","OMMA","DMMA"))
        fma = c.get("FFMA",0); ffm=c.get("FMUL",0)+c.get("FADD",0)
        print(f"  TENSOR-CORE MMA ops: {tc}   FFMA: {fma}   FMUL+FADD: {ffm}")
        for k,v in c.most_common(18): print(f"    {k:<12} {v}")
        print()
