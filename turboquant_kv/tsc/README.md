# tsc: a tensor-schedule compiler for the TurboQuant dequantization

`tsc` compiles a small declarative description of the KV-cache dequantization
pipeline (unpack, codebook gather, inverse rotation, rescale, cast, load,
store) into a fused Triton kernel, through a real pipeline: parser, typed AST,
semantic validation, explicit IR, optimization passes, code generation, and a
content-addressed codegen cache.

**The honest boundary, first.** The generated kernel is NOT the production
path. `kernel.py` (handwritten) remains what `TQPackedLayer` and the
profile-guided dispatcher use; nothing in `cache.py` or `dispatch.py` imports
`tsc`. The generated kernel is reachable only from tests and from
`tools/run_gpu_differential.py`, and it would earn a production role only by
passing every numerical gate and winning timing on the measured shapes, on a
recorded run. Until such a report exists, the claim here is a working
compiler pipeline, not a faster kernel.

## The language

```
# TurboQuant KV dequantization: unpack -> gather -> rotate -> rescale -> cast -> store
pipeline tq_dequant(bit_width=4, head_dim=128) {
  input  packed:    u8[N, 64]
  input  norms:     f16[N]
  input  centroids: f32[16]
  input  pi:        f32[head_dim, head_dim]
  output out:       bf16[N, head_dim]

  codes = load(packed)
  cb    = load(centroids)
  rotm  = load(pi)
  nrm   = load(norms)

  idx   = unpack(codes, bits=bit_width)
  y     = gather(cb, idx)
  x     = rot(y, rotm)
  nf    = cast(nrm, f32)
  z     = rescale(x, nf)
  o     = cast(z, bf16)
  store(out, o)
}
```

Grammar (EBNF; `#` comments run to end of line):

```
program   := "pipeline" IDENT "(" [params] ")" "{" {decl} {stmt} "}"
params    := IDENT "=" INT {"," IDENT "=" INT}
decl      := ("input" | "output") IDENT ":" DTYPE "[" dim {"," dim} "]"
dim       := INT | IDENT              # literal, param ref, or symbol like N
stmt      := IDENT "=" call | call    # the bare-call form is store only
call      := OP "(" arg {"," arg} ")"
arg       := IDENT "=" value | value
value     := IDENT | INT | DTYPE
DTYPE     := "u8" | "f16" | "bf16" | "f32"
OP        := "load" | "unpack" | "gather" | "rot" | "rescale" | "cast" | "store"
```

A dim that names a parameter resolves to its value; other identifiers (`N`)
stay symbolic through the whole pipeline and are unified by name. The
language is single-assignment and straight-line.

## Pipeline stages

1. **Parse** (`parser.py`) into a typed AST (`syntax.py`), locations on every
   node. Golden dumps in `goldens/*_ast.txt`.
2. **Validate** (`validate.py`): per-op dtype/rank/shape signatures, bounds
   (bits in the packing module's supported set, packed rows holding a whole
   number of indices, codebook sized exactly 2^bits), and operation order:
   use-before-def is rejected (`gather` before `unpack`), inputs are readable
   only through `load` (`rescale` of an unloaded tensor is rejected), outputs
   only writable by `store`, every output stored exactly once.
3. **Lower** (`ir.py`) into explicit linear IR, normalized so every
   rot/rescale operand passes through an explicit f32 cast. Goldens in
   `goldens/*_ir.txt`.
4. **Optimize** (`optimize.py`), four passes with red/green tests:
   redundant-cast elimination, unpack+gather fusion into a `decode` op,
   cast-into-store folding, dead-intermediate removal. The canonical bw4
   program shrinks from 15 to 9 instructions and lands exactly on the
   handwritten kernel's structure. Goldens in `goldens/*_ir_opt.txt`.
5. **Generate** (`codegen.py`): a standalone Python module with one
   `@triton.jit` kernel and a wrapper mirroring `kernel.tq_dequant`,
   including the MAXLEN stride handling for non-contiguous cache slices and
   the `allow_tf32` switch. The backend walks the IR and emits a block per
   op; it accepts unoptimized IR too, which is what makes disabling the
   optimizer a real control.
6. **Cache** (`compiler.py`): sha256 over source, options, and compiler
   version; a second compile of the same source is a hit, any change misses.
7. **Execute** (`runtime.py`): the generated kernel runs only with a CUDA
   device and triton present; otherwise execution falls back to the pure
   PyTorch IR interpreter (`reference.py`) and the report says so. The
   interpreter is proven bit-identical to an independently written
   dequantization formula in `test_tsc_fallback.py`.

## Running the tests

CPU-only, no GPU or triton needed (torch needed only for the fallback file):

```bash
cd turboquant_kv
python -m pytest test_tsc_parser.py test_tsc_validate.py test_tsc_ir.py \
    test_tsc_optimize.py test_tsc_codegen.py test_tsc_compile_cache.py \
    test_tsc_fallback.py
# or, without pytest, each file also runs standalone:
python test_tsc_parser.py
```

GPU execution evidence (an authorized CUDA host with torch and triton):

```bash
python tools/run_gpu_differential.py --out results/tsc_gpu_differential.json
python tools/run_gpu_differential.py --inject-index-bug   # negative control
```

The harness compares the generated kernel against the PyTorch reference
(fp32 gate 1e-5, tf32 gate 3.125e-2) and against the handwritten kernel on
the existing 24-shape matrix, times all arms, checks a strided cache slice,
and exits 2 rather than inventing numbers on a host with no CUDA device.
`--inject-index-bug` proves the differential actually catches a broken
kernel: it passes only when the off-by-one it injects makes the tests fail.

## Files

```
tsc/syntax.py     typed AST dataclasses, AST dump
tsc/parser.py     lexer + recursive-descent parser
tsc/validate.py   type/shape/rank/bounds/order rules -> ValidatedProgram
tsc/ir.py         explicit IR, normalized lowering, IR dump
tsc/optimize.py   the four passes + optimize() with per-pass stats
tsc/codegen.py    IR-driven Triton emission (+ indexing_delta fault hook)
tsc/compiler.py   compile_source() driver + content-addressed cache
tsc/reference.py  pure-PyTorch IR interpreter (the numerical reference)
tsc/runtime.py    execute() with the unknown-hardware fallback
tsc/examples.py   canonical sources; fixtures in tsc/examples/
tsc/goldens/      AST and IR dump goldens
```
