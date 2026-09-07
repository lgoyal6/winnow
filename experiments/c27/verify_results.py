"""Validate the retained C27 CUDA, decode, and Nsight result bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


class ResultRejected(AssertionError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ResultRejected(message)


def load_json(out: Path, name: str):
    return json.loads((out / name).read_text())


def verify_digests(out: Path) -> None:
    for line in (HERE / "SHA256SUMS").read_text().splitlines():
        digest, rel = line.split("  ", 1)
        payload = (HERE / rel).read_bytes()
        require(hashlib.sha256(payload).hexdigest() == digest, f"digest mismatch: {rel}")


def verify_matched(result: dict, *, headline: bool) -> None:
    require(result["N"] == result["B"] * result["H"] * result["L"], "N is inconsistent")
    expected_bytes = (
        result["N"] * result["NB"]
        + result["N"] * 2
        + (1 << result["BW"]) * 4
        + result["D"] * result["D"] * 4
        + result["N"] * result["D"] * 2
    )
    require(result["bytes_moved"] == expected_bytes, "byte accounting is inconsistent")
    require(result["macs"] == result["N"] * result["D"] ** 2, "MAC count is inconsistent")
    arms = result["arms"]
    for name in ("triton_fp32", "triton_tf32", "native_cuda_fp32"):
        require(name in arms and arms[name]["us"] > 0, f"missing timing for {name}")
    require(arms["triton_fp32"]["max_abs_err"] == 0, "Triton fp32 lost exactness")
    require(arms["native_cuda_fp32"]["max_abs_err"] == 0, "native CUDA lost exactness")
    require(arms["triton_tf32"]["max_abs_err"] > 0, "TF32 control no longer distinguishes precision")
    require(arms["triton_tf32"]["rel_l2_err"] > 0, "TF32 relative error is unexpectedly zero")
    if headline:
        ratio = arms["native_cuda_fp32"]["us"] / arms["triton_tf32"]["us"]
        require(7.0 < ratio < 9.0, f"headline native/TF32 ratio moved outside 7x to 9x: {ratio}")


def csv_kernel_row(path: Path, fragment: str) -> dict:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    matches = [row for row in rows if fragment in row["Name"]]
    require(len(matches) == 1, f"expected one {fragment} row in {path.name}")
    return matches[0]


def verify(out: Path, *, digests: bool = True) -> None:
    if digests:
        verify_digests(out)

    verify_matched(load_json(out, "matched_N1048576.json"), headline=True)
    verify_matched(load_json(out, "matched_N262144.json"), headline=False)
    verify_matched(load_json(out, "fresh_owner_port_check.json"), headline=False)

    decode = load_json(out, "decode_0.6b.json")
    require(set(decode) == {"2048", "8192", "16384"}, "decode contexts changed")
    for ctx, arms in decode.items():
        require(set(arms) == {
            "triton_tf32", "triton_fp32", "native_cuda_fp32", "fp16_baseline"
        }, f"decode arms changed at context {ctx}")
        require(max(arm["peak_gib"] for arm in arms.values()) < 3.95,
                f"4 GiB envelope exceeded at context {ctx}")
        require(arms["fp16_baseline"]["ms_per_token"] < arms["triton_tf32"]["ms_per_token"],
                f"retained negative result disappeared at context {ctx}")
    require(decode["2048"]["native_cuda_fp32"]["ms_per_token"]
            < decode["2048"]["triton_tf32"]["ms_per_token"],
            "native CUDA no longer wins the 2048 decode control")
    require(decode["16384"]["native_cuda_fp32"]["ms_per_token"]
            > decode["16384"]["triton_tf32"]["ms_per_token"],
            "native CUDA no longer loses the 16384 decode control")

    occupancy = load_json(out, "occupancy.json")
    kernels = {item["label"]: item for item in occupancy["kernels"]}
    require(occupancy["device"]["SM_COUNT"] == 84, "retained profile is not the A6000 run")
    require(kernels["native_cuda_fp32"]["theoretical_occupancy_pct"]
            > kernels["triton_tf32"]["theoretical_occupancy_pct"],
            "occupancy control no longer contradicts the timing order")
    require(kernels["triton_fp32"]["spill_local_bytes_per_thread"] == 8192,
            "retained fp32 Triton spill evidence changed")

    triton = csv_kernel_row(
        out / "decode_triton_tf32_kernels_cuda_gpu_kern_sum.csv", "_tq_dequant_kernel"
    )
    native = csv_kernel_row(
        out / "decode_native_cuda_fp32_kernels_cuda_gpu_kern_sum.csv", "tq_dequant_kernel"
    )
    require(int(triton["Instances"]) == int(native["Instances"]) == 672,
            "Nsight traces do not contain the matched 672 dequant launches")
    require(int(native["Total Time (ns)"]) > int(triton["Total Time (ns)"]),
            "Nsight dequant timing order changed")


def negative_control() -> None:
    with tempfile.TemporaryDirectory(prefix="winnow-c27-negctl-") as raw:
        out = Path(raw) / "out"
        shutil.copytree(HERE / "out", out)
        path = out / "matched_N1048576.json"
        result = json.loads(path.read_text())
        result["arms"]["native_cuda_fp32"]["max_abs_err"] = 1.0
        path.write_text(json.dumps(result))
        try:
            verify(out, digests=False)
        except ResultRejected as exc:
            require("native CUDA lost exactness" in str(exc), "wrong negative-control failure")
            print("NEGATIVE CONTROL PASS: corrupted native-CUDA accuracy was rejected")
            return
        raise ResultRejected("negative control was not rejected")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative-control", action="store_true")
    args = parser.parse_args()
    verify(HERE / "out")
    print("PASS: retained C27 kernel, decode, occupancy, and Nsight results are consistent")
    if args.negative_control:
        negative_control()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
