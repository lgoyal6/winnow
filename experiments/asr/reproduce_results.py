"""Recompute and byte-compare the committed ASR analysis."""

from __future__ import annotations

import difflib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(script: str) -> str:
    result = subprocess.run(
        [sys.executable, str(ROOT / script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    lines = "\n".join(line.rstrip() for line in result.stdout.splitlines())
    return lines + ("\n" if result.stdout.endswith("\n") else "")


def main() -> int:
    expected_scored = (ROOT / "out" / "scored.json").read_bytes()
    analysis = (
        "### analyze.py ###\n"
        + run("analyze.py")
        + "\n### analyze_ablations.py ###\n"
        + run("analyze_ablations.py")
    )
    expected_analysis = (ROOT / "out" / "ANALYSIS.txt").read_text()
    expected_ceiling = (ROOT / "out" / "ceiling.txt").read_text()
    expected_surface = (ROOT / "out" / "surface_form.txt").read_text()
    actual_scored = (ROOT / "out" / "scored.json").read_bytes()
    ceiling = run("analyze_ceiling.py")
    surface = run("analyze_surface_form.py")

    failures = []
    if actual_scored != expected_scored:
        failures.append("out/scored.json changed after recomputation")
    if analysis != expected_analysis:
        diff = "".join(difflib.unified_diff(
            expected_analysis.splitlines(keepends=True),
            analysis.splitlines(keepends=True),
            fromfile="committed/ANALYSIS.txt",
            tofile="recomputed/ANALYSIS.txt",
        ))
        failures.append("analysis text changed:\n" + diff[:4000])
    if ceiling != expected_ceiling:
        failures.append("ceiling analysis changed")
    if surface != expected_surface:
        failures.append("surface-form analysis changed")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("PASS: 630 rows and all deterministic analyses reproduce byte for byte")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
