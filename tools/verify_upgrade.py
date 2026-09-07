"""Install the first packaged core, upgrade it in place, and run a consumer."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIRST_PACKAGED_REF = "8b426fc"


def command(args, **kwargs):
    result = subprocess.run(args, text=True, capture_output=True, **kwargs)
    if result.returncode:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    return result.stdout.strip()


def install(source: Path, target: Path, *, upgrade: bool = False) -> None:
    args = [
        sys.executable, "-m", "pip", "install",
        "--no-index", "--no-build-isolation", "--no-deps",
        "--target", str(target),
    ]
    if upgrade:
        args.extend(["--upgrade", "--force-reinstall"])
    args.append(str(source))
    command(args)


def run_consumer(target: Path) -> str:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(target)
    return command([sys.executable, str(ROOT / "compat" / "core_consumer.py")], env=env)


def verify_current_guard_package(target: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(target)
    command([
        sys.executable,
        "-c",
        "from model_guard import artifact_manifest; "
        "artifact_manifest('BAAI/bge-small-en-v1.5')",
    ], env=env)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-ref", default=FIRST_PACKAGED_REF)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="winnow-upgrade-") as raw:
        work = Path(raw)
        archive = work / "old.tar"
        old_source = work / "old-source"
        target = work / "site-packages"
        old_source.mkdir()
        target.mkdir()

        with archive.open("wb") as fh:
            result = subprocess.run(
                ["git", "-C", str(ROOT), "archive", args.old_ref], stdout=fh
            )
        if result.returncode:
            raise SystemExit(result.returncode)
        command(["tar", "-xf", str(archive), "-C", str(old_source)])

        install(old_source, target)
        before = run_consumer(target)
        install(ROOT, target, upgrade=True)
        after = run_consumer(target)
        verify_current_guard_package(target)
        if before != after:
            print(f"consumer changed across upgrade:\nold: {before}\nnew: {after}", file=sys.stderr)
            return 1
        print(f"PASS: consumer output survived pip upgrade from {args.old_ref} to working tree")
        print(after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
