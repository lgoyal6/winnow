"""CPU-only tests for the DDP harness's non-distributed pieces.

Four things are worth testing here because each one is load-bearing for a
claim the harness makes:

  * the model config hash, because it is the "model revision" the frozen
    manifest pins, so a silent architecture change must move it;
  * the synthetic stream's determinism, because both arms of the scaling
    comparison must see the same tokens;
  * the checksum's sensitivity, because a checksum that cannot see a
    one-element change is decoration, not a gate;
  * the GPU inventory parser, because "two GPUs" is the easiest claim in this
    area to make by accident. Fixture text only; nvidia-smi is never invoked.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from ddp.checksum import (
    parameter_checksum, perturb_one_element, sampled_parameter_vector,
)
from ddp.data import SyntheticStream
from ddp.inventory import (
    InventoryRefusal, build_inventory, parse_nvidia_smi_csv,
    require_two_physical_gpus,
)
from ddp.model import PROXY, PRESETS, TINY, build_model, describe
from ddp.report import (
    build_negative_control, render_markdown, scaling, summarize_configuration,
)

MANIFEST = pathlib.Path(__file__).parent / "ddp" / "manifest.json"

TWO_DISTINCT_GPUS = """
0, NVIDIA RTX A6000, GPU-1111aaaa, 00000000:1A:00.0, 595.71.05
1, NVIDIA RTX A6000, GPU-2222bbbb, 00000000:1B:00.0, 595.71.05
"""
ONE_GPU = "0, NVIDIA RTX A6000, GPU-1111aaaa, 00000000:1A:00.0, 595.71.05\n"
SAME_UUID_TWICE = """
0, NVIDIA RTX A6000, GPU-1111aaaa, 00000000:1A:00.0, 595.71.05
1, NVIDIA RTX A6000, GPU-1111aaaa, 00000000:1B:00.0, 595.71.05
"""
TWO_MIG_SLICES_ONE_CARD = """
0, NVIDIA A100-SXM4-40GB, MIG-1111aaaa, 00000000:1A:00.0, 595.71.05
1, NVIDIA A100-SXM4-40GB, MIG-2222bbbb, 00000000:1A:00.0, 595.71.05
"""


# --- model revision ---------------------------------------------------------

def test_model_config_hash_is_stable():
    # Pinned literals on purpose. If a refactor changes the config's
    # serialization these fail, which is the point: the manifest's model
    # revision must not move silently.
    assert PROXY.config_sha256() == (
        "d368998ea8e9db4cfcda9e931a069b5ff9b48b570943decb4a54d8f769102102")
    assert TINY.config_sha256() == (
        "da122e86e9b3f782376bf0f3a2cf0a1d51112f6da22b00c7a2552ae5a5998677")


def test_model_config_hash_is_order_independent_but_field_sensitive():
    from dataclasses import replace
    assert PROXY.config_sha256() == PROXY.config_sha256()
    assert replace(PROXY, n_layers=7).config_sha256() != PROXY.config_sha256()
    assert replace(PROXY, norm_eps=1e-6).config_sha256() != PROXY.config_sha256()


def test_closed_form_parameter_count_matches_the_built_model():
    for cfg in PRESETS.values():
        model = build_model(cfg, seed=0)
        built = sum(param.numel() for param in model.parameters())
        assert built == cfg.parameter_count(), cfg.name


def test_same_seed_builds_bit_identical_parameters():
    first = parameter_checksum(build_model(TINY, seed=99))
    second = parameter_checksum(build_model(TINY, seed=99))
    assert first.matches(second)
    other = parameter_checksum(build_model(TINY, seed=100))
    assert not first.matches(other)


def test_describe_states_the_proxy_boundary():
    described = describe(PROXY)
    assert "synthetic-workload proxy" in described["role"]
    assert "nothing downloaded" in described["role"]


def test_model_refuses_head_dim_that_does_not_divide():
    from dataclasses import replace
    with pytest.raises(ValueError):
        replace(PROXY, n_heads=7)


# --- synthetic data ---------------------------------------------------------

def test_synthetic_stream_is_deterministic():
    import torch
    stream = SyntheticStream(seed=1234, vocab_size=1024,
                             global_batch_size=8, seq_len=32)
    assert torch.equal(stream.global_batch(3), stream.global_batch(3))
    assert not torch.equal(stream.global_batch(3), stream.global_batch(4))
    other_seed = SyntheticStream(seed=1235, vocab_size=1024,
                                 global_batch_size=8, seq_len=32)
    assert not torch.equal(stream.global_batch(3), other_seed.global_batch(3))


def test_rank_shards_are_disjoint_and_cover_the_global_batch():
    import torch
    stream = SyntheticStream(seed=1234, vocab_size=1024,
                             global_batch_size=8, seq_len=32)
    whole = stream.global_batch(5)
    left, _ = stream.rank_batch(5, 0, 2)
    right, _ = stream.rank_batch(5, 1, 2)
    assert left.shape == right.shape == (4, 31)
    assert not torch.equal(left, right)
    assert torch.equal(torch.cat([left, right]), whole[:, :-1])


def test_targets_are_the_inputs_shifted_by_one():
    import torch
    stream = SyntheticStream(seed=7, vocab_size=64, global_batch_size=2,
                             seq_len=16)
    tokens, targets = stream.rank_batch(0, 0, 1)
    assert torch.equal(tokens[:, 1:], targets[:, :-1])


def test_global_work_is_fixed_across_world_sizes():
    stream = SyntheticStream(seed=1234, vocab_size=1024,
                             global_batch_size=32, seq_len=512)
    assert stream.per_rank_batch_size(1) == 32
    assert stream.per_rank_batch_size(2) == 16
    assert stream.global_tokens_per_step() == 32 * 511


def test_indivisible_global_batch_is_refused():
    stream = SyntheticStream(seed=1, vocab_size=64, global_batch_size=3,
                             seq_len=8)
    with pytest.raises(ValueError):
        stream.per_rank_batch_size(2)


# --- checksums --------------------------------------------------------------

def test_checksum_detects_a_one_element_perturbation():
    baseline = build_model(TINY, seed=5)
    perturbed = build_model(TINY, seed=5)
    before = parameter_checksum(baseline)
    perturb_one_element(perturbed, delta=1e-3)
    after = parameter_checksum(perturbed)
    assert not before.matches(after)
    assert before.bytes_sha256 != after.bytes_sha256
    assert before.float64_sum != after.float64_sum
    assert before.parameter_count == after.parameter_count


def test_byte_hash_catches_a_perturbation_too_small_for_the_float_sum():
    # One float32 ULP at a parameter's own magnitude is far below the
    # resolution of a float64 sum over 404480 values. The byte hash is what
    # makes the gate exact, which is why both are computed.
    import torch
    baseline = build_model(TINY, seed=11)
    perturbed = build_model(TINY, seed=11)
    with torch.no_grad():
        flat = sorted(perturbed.named_parameters())[0][1].reshape(-1)
        flat[0] = torch.nextafter(flat[0], torch.tensor(float("inf")))
    before = parameter_checksum(baseline)
    after = parameter_checksum(perturbed)
    assert before.bytes_sha256 != after.bytes_sha256
    assert not before.matches(after)


def test_sampled_vector_is_small_and_deterministic():
    import torch
    model = build_model(TINY, seed=3)
    first = sampled_parameter_vector(model, per_tensor=8)
    second = sampled_parameter_vector(model, per_tensor=8)
    assert torch.equal(first, second)
    assert first.numel() < 1000


# --- GPU inventory ----------------------------------------------------------

def test_two_distinct_gpus_are_accepted():
    records = parse_nvidia_smi_csv(TWO_DISTINCT_GPUS)
    assert len(records) == 2
    require_two_physical_gpus(records)
    inventory = build_inventory(TWO_DISTINCT_GPUS)
    assert inventory["two_physical_gpus"] is True
    assert inventory["refusal"] is None
    assert inventory["gpu_count"] == 2
    assert inventory["driver_versions"] == ["595.71.05"]


def test_one_gpu_is_refused():
    with pytest.raises(InventoryRefusal) as caught:
        require_two_physical_gpus(parse_nvidia_smi_csv(ONE_GPU))
    assert "fewer than two physical GPUs" in str(caught.value)
    inventory = build_inventory(ONE_GPU)
    assert inventory["two_physical_gpus"] is False
    assert "fewer than two physical GPUs" in inventory["refusal"]


def test_the_same_uuid_twice_is_refused():
    with pytest.raises(InventoryRefusal) as caught:
        require_two_physical_gpus(parse_nvidia_smi_csv(SAME_UUID_TWICE))
    message = str(caught.value)
    assert "fewer than two physical GPUs" in message
    assert "GPU-1111aaaa" in message
    assert build_inventory(SAME_UUID_TWICE)["two_physical_gpus"] is False


def test_two_mig_slices_of_one_card_are_refused():
    # Distinct UUIDs, one physical card. The bus-id check is the only thing
    # standing between this and a fabricated multi-GPU result.
    records = parse_nvidia_smi_csv(TWO_MIG_SLICES_ONE_CARD)
    assert len({record.uuid for record in records}) == 2
    with pytest.raises(InventoryRefusal) as caught:
        require_two_physical_gpus(records)
    assert "PCI bus id" in str(caught.value)


def test_malformed_nvidia_smi_output_is_refused_not_guessed():
    with pytest.raises(InventoryRefusal):
        parse_nvidia_smi_csv("0, NVIDIA RTX A6000, GPU-1111aaaa\n")
    with pytest.raises(InventoryRefusal):
        parse_nvidia_smi_csv(
            "first, NVIDIA RTX A6000, GPU-a, 00000000:1A:00.0, 595.71.05\n")


# --- reporting --------------------------------------------------------------

def _fake_run(world_size: int, throughput: float, step: float,
              peak: int) -> dict:
    return {
        "label": f"{world_size}x",
        "world_size": world_size,
        "per_rank_batch_size": 32 // world_size,
        "global_tokens_per_step": 32 * 511,
        "precision": "bf16-autocast",
        "backend": "nccl",
        "throughput_tokens_per_second": throughput,
        "step_latency_median_seconds": step,
        "step_latency_p95_seconds": step * 1.1,
        "comm_percent_of_step_time": 12.5,
        "passed": True,
        "final_checksum": {"bytes_sha256": "abc"},
        "per_rank_metrics": [
            {"rank": rank, "device": f"cuda:{rank}",
             "device_name": "NVIDIA RTX A6000", "peak_memory_bytes": peak}
            for rank in range(world_size)],
    }


def test_scaling_reports_a_speedup_as_a_speedup():
    one = summarize_configuration([_fake_run(1, 1000.0, 0.5, 1 << 30)])
    two = summarize_configuration([_fake_run(2, 1800.0, 0.28, 1 << 29)])
    result = scaling(one, two)
    assert result["speedup"] == pytest.approx(1.8)
    assert result["scaling_efficiency"] == pytest.approx(0.9)
    assert "beat one" in result["verdict"]


def test_scaling_reports_a_slowdown_as_a_slowdown():
    one = summarize_configuration([_fake_run(1, 1000.0, 0.5, 1 << 30)])
    two = summarize_configuration([_fake_run(2, 800.0, 0.62, 1 << 29)])
    result = scaling(one, two)
    assert result["speedup"] < 1.0
    assert "did NOT beat one" in result["verdict"]
    assert "SLOWDOWN" in result["verdict"]


def test_scaling_is_not_computable_with_one_configuration():
    one = summarize_configuration([_fake_run(1, 1000.0, 0.5, 1 << 30)])
    assert scaling(one, None)["computable"] is False


def test_negative_control_requires_the_fault_to_have_fired():
    caught_run = {
        "mode": "gpu-physical-devices",
        "fault_actually_applied": True,
        "fault_applied_bucket_calls_total": 35,
        "fault_injection_per_rank": [],
        "fault_injection": {"mechanism": "m"},
        "final_checksums_match_across_ranks": False,
        "final_checksums_per_rank": [],
        "max_cross_rank_parameter_divergence": 0.003,
        "verdict": "FAILED: diverged",
    }
    assert build_negative_control(caught_run, exit_code=1)["fault_caught"]

    never_fired = dict(caught_run, fault_actually_applied=False,
                       fault_applied_bucket_calls_total=0)
    assert not build_negative_control(never_fired, exit_code=1)["fault_caught"]

    exited_zero = dict(caught_run)
    assert not build_negative_control(exited_zero, exit_code=0)["fault_caught"]

    no_divergence = dict(caught_run,
                         final_checksums_match_across_ranks=True,
                         max_cross_rank_parameter_divergence=0.0)
    assert not build_negative_control(no_divergence, exit_code=1)["fault_caught"]


def test_markdown_states_the_boundary_and_the_verdict():
    benchmark = {
        "evidence_class": "measured on physical NVIDIA GPUs",
        "mode": "gpu-physical-devices",
        "model_config_sha256": PROXY.config_sha256(),
        "inventory": None,
        "configurations": {
            "1-device": summarize_configuration([_fake_run(1, 1000.0, 0.5, 1 << 30)]),
            "2-device": summarize_configuration([_fake_run(2, 800.0, 0.62, 1 << 29)]),
        },
        "scaling": scaling(
            summarize_configuration([_fake_run(1, 1000.0, 0.5, 1 << 30)]),
            summarize_configuration([_fake_run(2, 800.0, 0.62, 1 << 29)])),
    }
    text = render_markdown(benchmark, None, None)
    assert "SLOWDOWN" in text
    assert "not model quality" in text
    assert "upper bound on exposed" in text
    assert "single host" in text


# --- the frozen manifest ----------------------------------------------------

def test_manifest_pins_the_code_that_actually_runs():
    """The freeze is only real if the manifest and the code agree.

    This is the test that would fail if someone changed the model, the batch
    size, or the step schedule after the manifest was committed.
    """
    manifest = json.loads(MANIFEST.read_text())
    revision = manifest["model_revision"]
    assert revision["benchmark_preset"]["config_sha256"] == PROXY.config_sha256()
    assert revision["cpu_selftest_preset"]["config_sha256"] == TINY.config_sha256()
    assert revision["benchmark_preset"]["parameter_count"] == PROXY.parameter_count()

    workload = manifest["workload"]
    stream = SyntheticStream(
        seed=manifest["seed"], vocab_size=PROXY.vocab_size,
        global_batch_size=workload["global_batch_size"],
        seq_len=workload["sequence_length"])
    assert workload["global_tokens_per_step"] == stream.global_tokens_per_step()
    assert workload["sequence_length"] <= PROXY.max_seq_len

    schedule = manifest["schedule"]
    assert schedule["warmup_steps"] == 5
    assert schedule["measured_steps"] >= 30
    assert schedule["repeats_per_configuration"] == 3
    assert [config["world_size"] for config in manifest["configurations"]] == [1, 2]

    selftest = manifest["local_cpu_selftest"]
    assert selftest["mode"] == "cpu-gloo-single-host-selftest"
    assert selftest["preset"] == TINY.name
    assert "not two GPUs" in selftest["what_it_does_not_prove"]


def test_manifest_matches_what_the_gate_script_launches():
    """The manifest is worthless if the shell script passes other numbers."""
    manifest = json.loads(MANIFEST.read_text())
    gate = (pathlib.Path(__file__).parents[1] / "scripts"
            / "run-multigpu-gate.sh").read_text()
    assert f"--global-batch-size {manifest['workload']['global_batch_size']}" in gate
    assert f"--seq-len {manifest['workload']['sequence_length']}" in gate
    assert f"--warmup-steps {manifest['schedule']['warmup_steps']}" in gate
    assert f"--measured-steps {manifest['schedule']['measured_steps']}" in gate
    assert f"--preset {manifest['model_revision']['benchmark_preset']['name']}" in gate

    selftest = manifest["local_cpu_selftest"]
    script = (pathlib.Path(__file__).parents[1] / "scripts"
              / "run-ddp-cpu-selftest.sh").read_text()
    assert f"--preset {selftest['preset']}" in script
    assert f"--global-batch-size {selftest['global_batch_size']}" in script
    assert f"--seq-len {selftest['sequence_length']}" in script
    assert f"--warmup-steps {selftest['warmup_steps']}" in script
    assert f"--measured-steps {selftest['measured_steps']}" in script
    assert f"--nproc_per_node {selftest['world_size']}" in script
