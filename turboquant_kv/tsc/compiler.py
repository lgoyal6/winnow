"""Front-to-back compile driver with a content-addressed codegen cache.

`compile_source` runs parse -> validate -> lower -> optimize -> codegen and
returns everything the tests and the runtime need: both IR dumps, the pass
stats, and the generated Triton source. The cache key is a hash of the DSL
source and every option that changes the output, plus the compiler version,
so a hit can skip the whole pipeline and a changed source can never collide.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from tsc.codegen import generate_triton
from tsc.ir import dump_ir, lower
from tsc.optimize import optimize
from tsc.parser import parse
from tsc.validate import validate

# Bump when any stage changes its output for the same source.
COMPILER_VERSION = "1"


@dataclass(frozen=True)
class CompileResult:
    key: str
    generated: str
    ir_before: str
    ir_after: str
    stats: dict
    cache_hit: bool
    elapsed_s: float


def cache_key(source: str, *, optimize_ir: bool, indexing_delta: int) -> str:
    payload = json.dumps({
        "source": source,
        "optimize": optimize_ir,
        "indexing_delta": indexing_delta,
        "version": COMPILER_VERSION,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def compile_source(source: str, *, optimize_ir: bool = True,
                   cache_dir: str | Path | None = None,
                   indexing_delta: int = 0) -> CompileResult:
    started = time.perf_counter()
    key = cache_key(source, optimize_ir=optimize_ir,
                    indexing_delta=indexing_delta)

    path = None
    if cache_dir is not None:
        path = Path(cache_dir) / f"{key}.json"
        if path.exists():
            payload = json.loads(path.read_text())
            return CompileResult(
                key=key, generated=payload["generated"],
                ir_before=payload["ir_before"], ir_after=payload["ir_after"],
                stats=payload["stats"], cache_hit=True,
                elapsed_s=time.perf_counter() - started)

    module = lower(validate(parse(source)))
    ir_before = dump_ir(module)
    if optimize_ir:
        module, stats = optimize(module)
    else:
        stats = {"instrs_before": len(module.instrs), "skipped": True,
                 "instrs_after": len(module.instrs)}
    ir_after = dump_ir(module)
    generated = generate_triton(module, indexing_delta=indexing_delta)

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "generated": generated, "ir_before": ir_before,
            "ir_after": ir_after, "stats": stats,
        }))
        tmp.replace(path)

    return CompileResult(key=key, generated=generated, ir_before=ir_before,
                         ir_after=ir_after, stats=stats, cache_hit=False,
                         elapsed_s=time.perf_counter() - started)
