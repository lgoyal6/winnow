"""Aggregation and reporting: turn per-run records into the committed evidence.

Kept in Python rather than in the shell scripts so that the scripts stay a
readable sequence of individual commands, and so the arithmetic behind
"scaling efficiency" is in one reviewable place.

The one rule this module enforces above all others: it reports what the runs
measured. There is no path here that upgrades a slowdown into a speedup, and
`scaling_verdict` says "slower" out loud when two devices lose to one on
fixed global work. A negative scaling result is a real finding about a
workload, an interconnect, or a batch size. Suppressing it would be the only
actual failure available at this stage.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

CPU_SELFTEST_MODE = "cpu-gloo-single-host-selftest"
CPU_SELFTEST_SENTENCE = (
    "This is NOT a multi-GPU result. It is two processes on one host using "
    "the gloo backend on CPU, with no GPU involved. It demonstrates that this "
    "harness synchronizes gradients correctly and that its cross-rank "
    "checksum gate detects a planted gradient-sync fault. No timing in this "
    "file is a scaling, speedup, or throughput claim.")


def load(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def dump(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)
        handle.write("\n")


def summarize_configuration(runs: list[dict]) -> dict:
    """Collapse the repeats of one configuration into one summary."""
    throughputs = [run["throughput_tokens_per_second"] for run in runs]
    medians = [run["step_latency_median_seconds"] for run in runs]
    p95s = [run["step_latency_p95_seconds"] for run in runs]
    comms = [run["comm_percent_of_step_time"] for run in runs]
    peaks: dict[str, int] = {}
    for run in runs:
        for metric in run["per_rank_metrics"]:
            key = f"rank{metric['rank']}:{metric['device']}"
            peaks[key] = max(peaks.get(key, 0), int(metric["peak_memory_bytes"]))
    return {
        "world_size": runs[0]["world_size"],
        "repeats": len(runs),
        "labels": [run["label"] for run in runs],
        "per_rank_batch_size": runs[0]["per_rank_batch_size"],
        "global_tokens_per_step": runs[0]["global_tokens_per_step"],
        "precision": runs[0]["precision"],
        "backend": runs[0]["backend"],
        "device_names": sorted({metric["device_name"]
                                for run in runs
                                for metric in run["per_rank_metrics"]}),
        "throughput_tokens_per_second_per_repeat": throughputs,
        "throughput_tokens_per_second_median": statistics.median(throughputs),
        "step_latency_median_seconds": statistics.median(medians),
        "step_latency_p95_seconds": max(p95s),
        "comm_percent_of_step_time_median": statistics.median(comms),
        "peak_memory_bytes_max_per_rank": peaks,
        "final_checksums": [run["final_checksum"]["bytes_sha256"]
                            for run in runs],
        "all_repeats_passed": all(run["passed"] for run in runs),
    }


def scaling(one: dict | None, two: dict | None) -> dict:
    if one is None or two is None:
        return {
            "computable": False,
            "reason": "need both a 1-device and a 2-device configuration",
        }
    t1 = one["throughput_tokens_per_second_median"]
    t2 = two["throughput_tokens_per_second_median"]
    efficiency = t2 / (2.0 * t1) if t1 > 0 else 0.0
    speedup = t2 / t1 if t1 > 0 else 0.0
    if t2 > t1:
        verdict = (f"two devices beat one on fixed global work: {speedup:.3f}x "
                   f"throughput, scaling efficiency {efficiency:.3f}")
    else:
        verdict = (f"two devices did NOT beat one on fixed global work: "
                   f"{speedup:.3f}x throughput (a SLOWDOWN), scaling "
                   f"efficiency {efficiency:.3f}. Reported as measured.")
    return {
        "computable": True,
        "throughput_1_device": t1,
        "throughput_2_devices": t2,
        "speedup": speedup,
        "scaling_efficiency": efficiency,
        "definition": "scaling_efficiency = throughput_2 / (2 * throughput_1) "
                      "at identical global batch size and sequence length",
        "verdict": verdict,
    }


def build_benchmark(run_paths: list[str], inventory_path: str | None) -> dict:
    runs = [load(path) for path in run_paths]
    by_world: dict[int, list[dict]] = {}
    for run in runs:
        by_world.setdefault(run["world_size"], []).append(run)
    configurations = {f"{world}-device": summarize_configuration(group)
                      for world, group in sorted(by_world.items())}
    inventory = load(inventory_path) if inventory_path else None
    modes = sorted({run.get("mode", "unknown") for run in runs})
    return {
        "mode": modes[0] if len(modes) == 1 else "mixed",
        "modes_seen": modes,
        "evidence_class": ("measured on physical NVIDIA GPUs, one process per "
                           "device, single host"
                           if modes == ["gpu-physical-devices"] else
                           "NOT a physical multi-GPU measurement"),
        "single_host": True,
        "inventory": inventory,
        "configurations": configurations,
        "scaling": scaling(configurations.get("1-device"),
                           configurations.get("2-device")),
        "model_config_sha256": runs[0]["model"]["config_sha256"],
        "all_runs_passed": all(run["passed"] for run in runs),
        "run_files": run_paths,
    }


def build_negative_control(run: dict, exit_code: int) -> dict:
    """Was the planted gradient-sync fault actually caught?

    All four conditions are required. In particular a fault that never fired
    (`fault_actually_applied` false) is a broken control, not a passing one:
    it would prove nothing about the checksum gate.
    """
    applied = bool(run.get("fault_actually_applied"))
    diverged = not run["final_checksums_match_across_ranks"]
    magnitude = float(run["max_cross_rank_parameter_divergence"])
    caught = bool(applied and diverged and magnitude > 0.0 and exit_code != 0)
    return {
        "mode": run.get("mode", "unknown"),
        "fault_caught": caught,
        "fault_actually_applied": applied,
        "fault_applied_bucket_calls_total":
            run.get("fault_applied_bucket_calls_total", 0),
        "fault_injection_per_rank": run.get("fault_injection_per_rank", []),
        "exit_code": exit_code,
        "final_checksums_match_across_ranks":
            run["final_checksums_match_across_ranks"],
        "final_checksums_per_rank": run["final_checksums_per_rank"],
        "max_cross_rank_parameter_divergence": magnitude,
        "verdict": run["verdict"],
        "mechanism": run["fault_injection"]["mechanism"],
        "requirement": ("the fault must fire, the cross-rank checksum must "
                        "differ, the sampled divergence must be non-zero, and "
                        "the process must exit non-zero"),
        "run": run,
    }


def build_cpu_selftest(positive: dict, positive_exit: int, negative: dict,
                       negative_exit: int, rerun: dict, rerun_exit: int) -> dict:
    control = build_negative_control(negative, negative_exit)
    passed = bool(positive["passed"] and positive_exit == 0
                  and control["fault_caught"]
                  and rerun["passed"] and rerun_exit == 0)
    return {
        "mode": CPU_SELFTEST_MODE,
        "not_a_multi_gpu_result": CPU_SELFTEST_SENTENCE,
        "single_host": True,
        "world_size": positive["world_size"],
        "backend": positive["backend"],
        "precision": positive["precision"],
        "device": positive["device"],
        "model": positive["model"],
        "spec": positive["spec"],
        "passed": passed,
        "positive": {
            "exit_code": positive_exit,
            "passed": positive["passed"],
            "verdict": positive["verdict"],
            "final_checksum_sha256": positive["final_checksum"]["bytes_sha256"],
            "final_checksums_match_across_ranks":
                positive["final_checksums_match_across_ranks"],
            "initial_checksums_match_across_ranks":
                positive["initial_checksums_match_across_ranks"],
            "max_cross_rank_parameter_divergence":
                positive["max_cross_rank_parameter_divergence"],
            "step_latency_median_seconds":
                positive["step_latency_median_seconds"],
            "step_latency_p95_seconds": positive["step_latency_p95_seconds"],
            "throughput_tokens_per_second":
                positive["throughput_tokens_per_second"],
            "comm_percent_of_step_time":
                positive["comm_percent_of_step_time"],
            "comm_timing_source": positive["comm_timing_source"],
            "per_rank_metrics": positive["per_rank_metrics"],
        },
        "negative_control": control,
        "positive_rerun_after_control": {
            "exit_code": rerun_exit,
            "passed": rerun["passed"],
            "verdict": rerun["verdict"],
            "final_checksum_sha256": rerun["final_checksum"]["bytes_sha256"],
            "reproduces_positive_checksum": (
                rerun["final_checksum"]["bytes_sha256"]
                == positive["final_checksum"]["bytes_sha256"]),
        },
        "timing_caveat": ("step latency and communication percentage here are "
                          "gloo-on-CPU numbers from two processes sharing one "
                          "machine's cores. They exercise the measurement "
                          "path; they are not a performance result."),
    }


def _bytes_mib(value: int) -> str:
    return f"{value / (1024 * 1024):.1f} MiB"


def render_markdown(benchmark: dict, control: dict | None,
                    rerun: dict | None) -> str:
    lines: list[str] = []
    lines.append("# Multi-GPU DDP scaling gate")
    lines.append("")
    lines.append(f"- evidence class: {benchmark['evidence_class']}")
    lines.append(f"- mode: {benchmark['mode']}")
    lines.append("- single host, one process per device")
    lines.append(f"- model revision (config sha256): "
                 f"`{benchmark['model_config_sha256']}`")
    lines.append("")

    inventory = benchmark.get("inventory")
    if inventory:
        lines.append("## Hardware inventory")
        lines.append("")
        lines.append(f"- visible GPUs: {inventory['gpu_count']}")
        lines.append(f"- driver: {', '.join(inventory['driver_versions'])}")
        lines.append(f"- CUDA (driver-reported): {inventory['cuda_version']}")
        lines.append(f"- torch: {inventory['torch_version']}")
        lines.append("")
        lines.append("| index | name | uuid | pci bus id |")
        lines.append("|---|---|---|---|")
        for gpu in inventory["gpus"]:
            lines.append(f"| {gpu['index']} | {gpu['name']} | "
                         f"`{gpu['uuid']}` | `{gpu['pci_bus_id']}` |")
        lines.append("")

    lines.append("## Configurations")
    lines.append("")
    lines.append("| config | repeats | per-rank batch | global tokens/step | "
                 "throughput tok/s (median) | median step | p95 step | comm % |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, config in benchmark["configurations"].items():
        lines.append(
            f"| {name} | {config['repeats']} | "
            f"{config['per_rank_batch_size']} | "
            f"{config['global_tokens_per_step']} | "
            f"{config['throughput_tokens_per_second_median']:.1f} | "
            f"{config['step_latency_median_seconds'] * 1e3:.3f} ms | "
            f"{config['step_latency_p95_seconds'] * 1e3:.3f} ms | "
            f"{config['comm_percent_of_step_time_median']:.2f}% |")
    lines.append("")

    lines.append("## Peak memory per rank")
    lines.append("")
    lines.append("| config | rank | peak |")
    lines.append("|---|---|---|")
    for name, config in benchmark["configurations"].items():
        for key, value in sorted(config["peak_memory_bytes_max_per_rank"].items()):
            lines.append(f"| {name} | {key} | {_bytes_mib(int(value))} |")
    lines.append("")

    scale = benchmark["scaling"]
    lines.append("## Scaling")
    lines.append("")
    if scale["computable"]:
        lines.append(f"- {scale['definition']}")
        lines.append(f"- 1 device: {scale['throughput_1_device']:.1f} tok/s")
        lines.append(f"- 2 devices: {scale['throughput_2_devices']:.1f} tok/s")
        lines.append(f"- speedup: {scale['speedup']:.3f}x")
        lines.append(f"- scaling efficiency: {scale['scaling_efficiency']:.3f}")
        lines.append(f"- **{scale['verdict']}**")
    else:
        lines.append(f"- not computable: {scale['reason']}")
    lines.append("")

    lines.append("## Negative control")
    lines.append("")
    if control is None:
        lines.append("- not run")
    else:
        lines.append(f"- fault caught: **{control['fault_caught']}**")
        lines.append(f"- mechanism: {control['mechanism']}")
        lines.append(f"- faulted bucket calls across ranks: "
                     f"{control['fault_applied_bucket_calls_total']}")
        lines.append(f"- exit code: {control['exit_code']}")
        lines.append(f"- max cross-rank parameter divergence: "
                     f"{control['max_cross_rank_parameter_divergence']:.6e}")
        lines.append(f"- verdict: {control['verdict']}")
    lines.append("")

    lines.append("## Positive rerun after the control")
    lines.append("")
    if rerun is None:
        lines.append("- not run")
    else:
        lines.append(f"- passed: {rerun['passed']}")
        lines.append(f"- verdict: {rerun['verdict']}")
        lines.append(f"- final checksum: "
                     f"`{rerun['final_checksum']['bytes_sha256']}`")
    lines.append("")

    lines.append("## Boundary")
    lines.append("")
    lines.append("- Every number above was measured on the host named in the "
                 "inventory, with one process per physical GPU, on a single "
                 "host. There is no multi-node result here.")
    lines.append("- The workload is a synthetic-noise token stream over an "
                 "in-repo proxy model. Throughput describes the training loop "
                 "and the gradient all-reduce, not model quality.")
    lines.append("- Communication percentage is an upper bound on exposed "
                 "communication: per-bucket all-reduce intervals overlap each "
                 "other and overlap backward compute by design.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DDP harness reporting")
    sub = parser.add_subparsers(dest="command", required=True)

    bench = sub.add_parser("benchmark", help="aggregate per-run records")
    bench.add_argument("--run", action="append", default=[], required=True)
    bench.add_argument("--inventory", default=None)
    bench.add_argument("--out", required=True)

    control = sub.add_parser("negative-control",
                             help="score one fault-injected run")
    control.add_argument("--run", required=True)
    control.add_argument("--exit-code", type=int, required=True)
    control.add_argument("--out", required=True)

    markdown = sub.add_parser("markdown", help="render the report")
    markdown.add_argument("--benchmark", required=True)
    markdown.add_argument("--negative-control", default=None)
    markdown.add_argument("--positive-rerun", default=None)
    markdown.add_argument("--out", required=True)

    selftest = sub.add_parser("cpu-selftest",
                              help="score the local CPU gloo self-test")
    selftest.add_argument("--positive", required=True)
    selftest.add_argument("--positive-exit", type=int, required=True)
    selftest.add_argument("--negative", required=True)
    selftest.add_argument("--negative-exit", type=int, required=True)
    selftest.add_argument("--positive-rerun", required=True)
    selftest.add_argument("--positive-rerun-exit", type=int, required=True)
    selftest.add_argument("--out", required=True)

    args = parser.parse_args(argv)

    if args.command == "benchmark":
        result = build_benchmark(args.run, args.inventory)
        dump(result, args.out)
        print(f"wrote {args.out}")
        print(result["scaling"].get("verdict", result["scaling"].get("reason")))
        return 0 if result["all_runs_passed"] else 1

    if args.command == "negative-control":
        result = build_negative_control(load(args.run), args.exit_code)
        dump(result, args.out)
        print(f"wrote {args.out}")
        print(f"fault_caught={result['fault_caught']}")
        return 0 if result["fault_caught"] else 1

    if args.command == "markdown":
        control_obj = (load(args.negative_control)
                       if args.negative_control else None)
        rerun_obj = load(args.positive_rerun) if args.positive_rerun else None
        text = render_markdown(load(args.benchmark), control_obj, rerun_obj)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            handle.write(text)
        print(f"wrote {args.out}")
        return 0

    result = build_cpu_selftest(
        load(args.positive), args.positive_exit,
        load(args.negative), args.negative_exit,
        load(args.positive_rerun), args.positive_rerun_exit)
    dump(result, args.out)
    print(f"wrote {args.out}")
    print(f"passed={result['passed']} "
          f"fault_caught={result['negative_control']['fault_caught']}")
    if not result["passed"]:
        print("CPU gloo self-test did not satisfy every condition",
              file=sys.stderr)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
