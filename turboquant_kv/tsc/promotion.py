"""Measured promotion records: what may replace the safe path, and where.

The codegen cache (compiler.py) answers "what source does this DSL produce";
this registry answers "which validated, measured schedule may production use
on this exact device, toolchain, source and contract". They are separate on
purpose: a cache hit is never a promotion.

A record binds:
  op and shape family (exact-match fields), numerical contract,
  DSL source sha + compiler version + codegen options + generated source sha
  (dequantizer) or kernel source sha (attention),
  backend/toolchain (torch, triton, CUDA) and target (compute capability, device name),
  schedule, the immutable validation result, the benchmark summary with its
  raw-sample path, coverage (sizes the schedule was validated and measured on),
  and a sha256 checksum over everything else.

Loading checks the checksum, every identity field and the coverage; any
mismatch, parse failure or missing file returns no record plus the reason,
and the caller keeps its safe path. Generated code is never read from the
registry: the consumer regenerates it with the local compiler and runs it
only if its hash equals the promoted one.

Writes are atomic (temp file, fsync, rename) under an exclusive flock, so
concurrent tuners cannot interleave a record; the later valid write wins.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

PROMOTION_VERSION = 1


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def identity_key(identity: dict) -> str:
    return sha256_bytes(canonical(identity))[:32]


def seal(record: dict) -> dict:
    body = {k: v for k, v in record.items() if k != "checksum"}
    return {**body, "checksum": sha256_bytes(canonical(body))}


class PromotionError(ValueError):
    pass


class Registry:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, identity: dict) -> Path:
        return self.root / f"{identity['op']}-{identity_key(identity)}.json"

    def promote(self, identity: dict, schedule: dict, validation: dict, benchmark: dict,
                coverage: dict, extra: dict | None = None) -> Path:
        """Refuses anything not validated or not measurably faster than its baseline."""
        if not validation.get("passed"):
            raise PromotionError(f"validation did not pass: {validation.get('reason')}")
        base, cand = benchmark.get("baseline_median_us"), benchmark.get("candidate_median_us")
        if base is None or cand is None or not cand < base:
            raise PromotionError(f"no measured benefit (baseline {base}, candidate {cand})")
        record = seal({"promotion_version": PROMOTION_VERSION, "identity": identity,
                       "schedule": schedule, "validation": validation, "benchmark": benchmark,
                       "coverage": coverage, "extra": extra or {},
                       "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        path = self.path(identity)
        lock = self.root / ".lock"
        with open(lock, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            tmp = path.with_suffix(f".tmp.{os.getpid()}")
            with open(tmp, "wb") as fh:
                fh.write(canonical(record))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
            fcntl.flock(lf, fcntl.LOCK_UN)
        return path

    def load(self, identity: dict, size: dict | None = None) -> tuple[dict | None, str]:
        path = self.path(identity)
        if not path.exists():
            return None, "no promotion for this identity (cache miss)"
        try:
            record = json.loads(path.read_bytes())
        except (OSError, ValueError) as exc:
            return None, f"unreadable promotion record: {type(exc).__name__}"
        if not isinstance(record, dict) or "checksum" not in record:
            return None, "malformed promotion record"
        if seal(record)["checksum"] != record["checksum"]:
            return None, "checksum mismatch (corrupted or edited record)"
        if record.get("promotion_version") != PROMOTION_VERSION:
            return None, "promotion version mismatch"
        if record.get("identity") != identity:
            diff = sorted(k for k in set(identity) | set(record.get("identity", {}))
                          if identity.get(k) != record.get("identity", {}).get(k))
            return None, f"identity mismatch on {diff}"
        if not record.get("validation", {}).get("passed"):
            return None, "record carries a failed validation"
        if size is not None and not covered(record.get("coverage", {}), size):
            return None, f"size {size} outside validated coverage"
        return record, "promoted"


def covered(coverage: dict, size: dict) -> bool:
    """Sizes must fall inside every validated [min, max] range (or an exact list)."""
    for k, v in size.items():
        c = coverage.get(k)
        if c is None:
            return False
        if isinstance(c, list) and v not in c:
            return False
        if isinstance(c, dict) and not (c["min"] <= v <= c["max"]):
            return False
    return True
