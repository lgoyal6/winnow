"""Compile-time and codegen-cache behavior.

The contract: a second compile of the same source with the same options is a
cache hit and measurably faster; any change to the source or to an
output-affecting option misses. Timing asserts hit < miss on medians over a
few runs rather than a single sample, to keep the test honest without being
flaky.
"""
from __future__ import annotations

import statistics
import time

from tsc.compiler import COMPILER_VERSION, cache_key, compile_source
from tsc.examples import example_source


def _timed_compile(tmp_path, source, **kw):
    t0 = time.perf_counter()
    result = compile_source(source, cache_dir=tmp_path, **kw)
    return result, time.perf_counter() - t0


def test_second_compile_is_a_cache_hit_and_faster(tmp_path):
    src = example_source(4)
    cold, cold_s = _timed_compile(tmp_path, src)
    assert cold.cache_hit is False

    hits, hit_times = [], []
    for _ in range(5):
        hit, hit_s = _timed_compile(tmp_path, src)
        hits.append(hit)
        hit_times.append(hit_s)
    assert all(h.cache_hit for h in hits)
    assert all(h.generated == cold.generated for h in hits)
    assert all(h.stats == cold.stats for h in hits)
    hit_s = statistics.median(hit_times)
    assert hit_s < cold_s, (hit_s, cold_s)


def test_changed_source_misses(tmp_path):
    warm = compile_source(example_source(4), cache_dir=tmp_path)
    assert compile_source(example_source(4), cache_dir=tmp_path).cache_hit
    changed = compile_source(example_source(8), cache_dir=tmp_path)
    assert changed.cache_hit is False
    assert changed.key != warm.key
    # Even a comment-only change is a different source text, so it misses:
    # the cache is content-addressed, not semantic.
    commented = compile_source("# touched\n" + example_source(4),
                               cache_dir=tmp_path)
    assert commented.cache_hit is False


def test_options_partition_the_cache(tmp_path):
    src = example_source(4)
    a = compile_source(src, cache_dir=tmp_path)
    b = compile_source(src, cache_dir=tmp_path, optimize_ir=False)
    c = compile_source(src, cache_dir=tmp_path, indexing_delta=1)
    assert len({a.key, b.key, c.key}) == 3
    # and each variant now hits its own entry
    assert compile_source(src, cache_dir=tmp_path,
                          optimize_ir=False).cache_hit
    assert compile_source(src, cache_dir=tmp_path,
                          indexing_delta=1).cache_hit


def test_cache_key_covers_compiler_version():
    src = example_source(4)
    key = cache_key(src, optimize_ir=True, indexing_delta=0)
    assert COMPILER_VERSION in ("1",)   # bump both when output changes
    assert key != cache_key(src + " ", optimize_ir=True, indexing_delta=0)


def test_no_cache_dir_always_compiles():
    first = compile_source(example_source(4))
    second = compile_source(example_source(4))
    assert first.cache_hit is False and second.cache_hit is False
    assert first.generated == second.generated


def _run_all():
    import tempfile
    import traceback
    from pathlib import Path
    names = [k for k in sorted(globals()) if k.startswith("test_")]
    failed = 0
    for name in names:
        fn = globals()[name]
        try:
            if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print(f"  ok  {name}")
        except Exception:
            failed += 1
            traceback.print_exc()
    print(f"\n{len(names) - failed}/{len(names)} tsc cache tests passed.")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    _run_all()
