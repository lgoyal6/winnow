"""The measurement chain end to end on CPU: loadgen -> gRPC -> report.

Checks plumbing and accounting (every offered request in the denominator,
refusals counted, tables rebuilt from raw rows), not performance.
"""
import json

import loadgen
import report_packed_serving as rep
from test_grpc_server import REV, served  # noqa: F401 - fixture
from test_packed_model import model, prompts  # noqa: F401 - fixture


def test_loadgen_and_report(served, tmp_path):
    eng, run, _ = served(max_batch=2, max_queue=2, pages=64)
    ps = prompts(10, seed=80)
    wl = tmp_path / "w.json"
    wl.write_text(json.dumps({"requests": [{"id": f"q{i}", "token_ids": p, "max_new_tokens": 6}
                                           for i, p in enumerate(ps)]}))
    outs = []
    for arm, rate in (("base", 2.0), ("burst", 500.0)):
        d = tmp_path / arm
        loadgen.main(["--workload", str(wl), "--target", f"grpc://127.0.0.1:{run.port}",
                      "--revision", REV, "--arm", arm, "--rate", str(rate), "--out", str(d),
                      "--ignore-eos"])
        (d / "label.json").write_text(json.dumps({"load": "L1"}))
        outs.append(d)
    rows = [json.loads(x) for x in (outs[1] / "requests.jsonl").read_text().splitlines()]
    assert len(rows) == 10
    # A 500 req/s burst against max_batch 2 + queue 2 must refuse some requests,
    # and those rows must be present, not dropped.
    assert any(r["status"] == "refused" for r in rows)
    assert all(r["status"] in ("length", "refused") for r in rows)
    assert all(len(r["token_times"]) == 6 for r in rows if r["status"] == "length")
    proto = tmp_path / "protocol.json"
    proto.write_text(json.dumps({"ttft_slo_s": 30.0, "itl_p95_slo_s": 30.0, "baseline_arm": "base",
                                 "bootstrap_draws": 200, "seed": 0}))
    rep.main(["--protocol", str(proto), "--runs", *map(str, outs), "--out", str(tmp_path / "rep")])
    report = json.loads((tmp_path / "rep" / "report.json").read_text())
    burst = next(r for r in report["runs"] if r["arm"] == "burst")
    assert burst["offered"] == 10
    assert burst["slo_met"] == burst["statuses"].get("length", 0) < 10
    assert report["comparisons"][0]["pairs"] == 10
    assert (tmp_path / "rep" / "report.md").read_text().count("|") > 20
    h = [json.loads(x) for x in (outs[0] / "health.jsonl").read_text().splitlines()]
    assert h and "free" in h[0]


def test_freeze_protocol_applies_rule_and_never_overwrites(served, tmp_path):
    import pytest

    import freeze_protocol
    eng, run, _ = served(max_batch=1)
    wl = tmp_path / "w.json"
    wl.write_text(json.dumps({"requests": [{"id": f"c{i}", "token_ids": p, "max_new_tokens": 5}
                                           for i, p in enumerate(prompts(4, seed=90))]}))
    cal = tmp_path / "cal"
    loadgen.main(["--workload", str(wl), "--target", f"grpc://127.0.0.1:{run.port}", "--revision", REV,
                  "--arm", "dense-cal", "--concurrency", "1", "--ignore-eos", "--out", str(cal)])
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({"ttft_multiplier": 3.0, "itl_multiplier": 3.0, "baseline_arm": "dense"}))
    out = tmp_path / "protocol.json"
    freeze_protocol.main(["--rules", str(rules), "--calibration", str(cal), "--out", str(out)])
    p = json.loads(out.read_text())
    d = p["derived_from"]
    assert p["ttft_slo_s"] == round(3.0 * d["median_ttft_s"], 4) and d["completed"] == 4
    with pytest.raises(SystemExit):
        freeze_protocol.main(["--rules", str(rules), "--calibration", str(cal), "--out", str(out)])
