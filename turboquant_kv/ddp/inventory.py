"""What counts as "two GPUs", and the parser that refuses everything else.

"Multi-GPU" is the easiest claim in this whole area to make by accident. Two
processes on one card, two MIG slices of one card, two CUDA_VISIBLE_DEVICES
entries pointing at the same device, or two processes on a CPU all produce a
world size of 2 and a plausible-looking scaling table. None of them is a
second physical GPU, and a scaling number derived from any of them is false.

So the gate does not trust world size. It requires `nvidia-smi` to report at
least two entries with:

  * distinct GPU UUIDs, which separates two devices from one device counted
    twice, and
  * distinct PCI bus ids, which is what separates two physical cards from two
    MIG instances carved out of one card. MIG instances get their own
    `MIG-...` UUIDs but inherit the parent card's bus id, so the UUID check
    alone would pass them.

Anything less is a refusal with exit code 2, and no benchmark number is
produced. Refusing to measure is a valid outcome; inventing a number is not.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass

QUERY_FIELDS = ("index", "name", "uuid", "pci.bus_id", "driver_version")
NVIDIA_SMI_ARGS = (
    f"--query-gpu={','.join(QUERY_FIELDS)}", "--format=csv,noheader")


class InventoryRefusal(Exception):
    """Raised when the visible hardware cannot support a multi-GPU claim."""


@dataclass(frozen=True)
class GpuRecord:
    index: int
    name: str
    uuid: str
    pci_bus_id: str
    driver_version: str


def parse_nvidia_smi_csv(text: str) -> list[GpuRecord]:
    """Parse `nvidia-smi --query-gpu=... --format=csv,noheader` output.

    Pure text in, records out: the tests feed it fixture strings so that the
    refusal logic is provable on a host with no NVIDIA hardware at all.
    """
    records: list[GpuRecord] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = [piece.strip() for piece in line.split(",")]
        if len(parts) != len(QUERY_FIELDS):
            raise InventoryRefusal(
                f"REFUSING: nvidia-smi line {line_number} has {len(parts)} "
                f"fields, expected {len(QUERY_FIELDS)} "
                f"({','.join(QUERY_FIELDS)}): {line!r}")
        try:
            index = int(parts[0])
        except ValueError as exc:
            raise InventoryRefusal(
                f"REFUSING: nvidia-smi line {line_number} has a non-integer "
                f"GPU index {parts[0]!r}") from exc
        records.append(GpuRecord(index=index, name=parts[1], uuid=parts[2],
                                 pci_bus_id=parts[3], driver_version=parts[4]))
    return records


def require_two_physical_gpus(records: list[GpuRecord]) -> None:
    """Raise InventoryRefusal unless there are 2+ distinct physical GPUs."""
    if len(records) < 2:
        raise InventoryRefusal(
            f"REFUSING: fewer than two physical GPUs are visible "
            f"(nvidia-smi reported {len(records)}). This harness will not "
            f"produce a multi-GPU scaling number on this host.")

    uuids = [record.uuid for record in records]
    duplicate_uuid = _first_duplicate(uuids)
    if duplicate_uuid is not None:
        raise InventoryRefusal(
            f"REFUSING: fewer than two physical GPUs are visible: GPU UUID "
            f"{duplicate_uuid} appears {uuids.count(duplicate_uuid)} times, so "
            f"these entries are one device counted more than once.")

    bus_ids = [record.pci_bus_id for record in records]
    duplicate_bus = _first_duplicate(bus_ids)
    if duplicate_bus is not None:
        raise InventoryRefusal(
            f"REFUSING: fewer than two physical GPUs are visible: PCI bus id "
            f"{duplicate_bus} appears {bus_ids.count(duplicate_bus)} times, so "
            f"these entries share one physical card (MIG instances of a single "
            f"GPU look like this).")


def _first_duplicate(values: list[str]) -> str | None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


def build_inventory(csv_text: str, cuda_version: str | None = None,
                    torch_version: str | None = None) -> dict:
    """Assemble the inventory record. Does NOT enforce the two-GPU rule.

    Separated on purpose: the inventory is written whatever the hardware turns
    out to be, and the refusal is a separate, explicit decision on top of it.
    """
    records = parse_nvidia_smi_csv(csv_text)
    drivers = sorted({record.driver_version for record in records})
    try:
        require_two_physical_gpus(records)
        refusal = None
    except InventoryRefusal as exc:
        refusal = str(exc)
    return {
        "gpu_count": len(records),
        "gpus": [asdict(record) for record in records],
        "gpu_names": [record.name for record in records],
        "gpu_uuids": [record.uuid for record in records],
        "pci_bus_ids": [record.pci_bus_id for record in records],
        "driver_versions": drivers,
        "cuda_version": cuda_version,
        "torch_version": torch_version,
        "two_physical_gpus": refusal is None,
        "refusal": refusal,
        "requirement": ("at least two nvidia-smi entries with distinct GPU "
                        "UUIDs and distinct PCI bus ids"),
    }


def _nvidia_smi_cuda_version() -> str | None:
    """CUDA version as the driver reports it, or None if unavailable."""
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None
    try:
        out = subprocess.run([binary, "--version"], capture_output=True,
                             text=True, timeout=30, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        if "CUDA Version" in line:
            return line.split(":", 1)[-1].strip()
    return None


def _torch_version() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    return torch.__version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inventory the visible NVIDIA GPUs and gate on two "
                    "distinct physical devices")
    parser.add_argument("--out", default=None, help="write the inventory JSON here")
    parser.add_argument("--csv-file", default=None,
                        help="read nvidia-smi CSV from a file instead of "
                             "invoking nvidia-smi (for testing the gate)")
    args = parser.parse_args(argv)

    if args.csv_file:
        with open(args.csv_file) as handle:
            csv_text = handle.read()
    else:
        binary = shutil.which("nvidia-smi")
        if binary is None:
            print("REFUSING: fewer than two physical GPUs are visible: "
                  "nvidia-smi is not on this host, so there is no NVIDIA GPU "
                  "to measure. No inventory and no benchmark numbers written.",
                  file=sys.stderr)
            return 2
        completed = subprocess.run([binary, *NVIDIA_SMI_ARGS],
                                   capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            print(f"REFUSING: nvidia-smi exited {completed.returncode}: "
                  f"{completed.stderr.strip()}", file=sys.stderr)
            return 2
        csv_text = completed.stdout

    try:
        inventory = build_inventory(
            csv_text, cuda_version=_nvidia_smi_cuda_version(),
            torch_version=_torch_version())
    except InventoryRefusal as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(inventory, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"wrote {args.out}")

    print(f"visible GPUs: {inventory['gpu_count']}")
    for gpu in inventory["gpus"]:
        print(f"  [{gpu['index']}] {gpu['name']} uuid={gpu['uuid']} "
              f"bus={gpu['pci_bus_id']}")
    if inventory["refusal"]:
        print(inventory["refusal"], file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
