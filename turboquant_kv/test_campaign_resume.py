"""Linux campaign bookkeeping without renting or initializing a GPU."""
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import pytest

HERE = Path(__file__).resolve().parent
pytestmark = pytest.mark.skipif(not shutil.which("timeout") or not shutil.which("flock"),
                                reason="requires Linux timeout and flock")


@pytest.fixture(params=["gpu_campaign.sh", "gpu_campaign_02.sh"])
def campaign(request, tmp_path):
    path = HERE / request.param
    text = path.read_text()
    start = text.index("stage() {")
    function = text[start:text.index("\n}\n", start) + 3]
    out = tmp_path / "evidence"
    (out / "done").mkdir(parents=True)
    env = {**os.environ, "OUT": str(out), "DEADLINE_EPOCH": "9999999999",
           "COHORT": str(tmp_path), "PROTOCOL": str(tmp_path / "protocol.json"),
           "PROTOCOL_RULES": str(tmp_path / "rules.json")}
    return path, function, out, env


def extract_function(path, name):
    text = path.read_text()
    start = text.index(f"{name}() {{")
    return text[start:text.index("\n}\n", start) + 3]


def test_interrupted_output_and_log_survive_resume(campaign, tmp_path):
    _, function, out, env = campaign
    destination = out / "load/run"
    destination.mkdir(parents=True)
    (destination / "partial.txt").write_text("partial evidence")
    (out / "sample.log").write_text("interrupted process")
    writer = tmp_path / "writer.py"
    writer.write_text("from pathlib import Path\nimport sys\n"
                      "p=Path(sys.argv[2]);p.mkdir();(p/'complete.txt').write_text('complete')\n")
    command = "stage sample 10 " + shlex.join([sys.executable, str(writer), "--out", str(destination)])
    subprocess.run(["bash", "-c", function + "\n" + command + "\n" + command],
                   env=env, check=True, capture_output=True, text=True)
    assert (destination / "complete.txt").read_text() == "complete"
    assert [p.read_text() for p in (out / "interrupted").glob("*/partial.txt")] == ["partial evidence"]
    assert [p.read_text() for p in out.glob("sample-interrupted-*.log")] == ["interrupted process"]
    assert (out / "done/sample").exists()
    records = [json.loads(x) for x in (out / "stages.jsonl").read_text().splitlines()]
    assert len(records) == 1 and records[0]["exit"] == 0


def test_completed_failure_is_recorded_and_not_replaced(campaign):
    _, function, out, env = campaign
    command = "stage failed 10 " + shlex.join([sys.executable, "-c", "raise SystemExit(7)"])
    subprocess.run(["bash", "-c", function + "\n" + command + "\n" + command],
                   env=env, check=True, capture_output=True, text=True)
    records = [json.loads(x) for x in (out / "stages.jsonl").read_text().splitlines()]
    assert len(records) == 1 and records[0]["exit"] == 7


def test_deadline_skip_is_not_a_completion(campaign):
    _, function, out, env = campaign
    result = subprocess.run(["bash", "-c", function + "\nstage late 10 false"],
                            env={**env, "DEADLINE_EPOCH": "0"}, capture_output=True, text=True)
    assert result.returncode == 0
    assert not (out / "done/late").exists()
    assert json.loads((out / "stages.jsonl").read_text())["skipped"] == "deadline"


def test_full_stage_timeout_is_a_completed_failure(campaign):
    _, function, out, env = campaign
    command = "stage bounded 2 " + shlex.join([sys.executable, "-c", "import time; time.sleep(60)"])
    before = time.monotonic()
    subprocess.run(["bash", "-c", function + "\n" + command],
                   env={**env, "DEADLINE_EPOCH": str(int(time.time()) + 10)},
                   check=True, capture_output=True, text=True, timeout=8)
    assert 1.5 <= time.monotonic() - before < 8
    record = json.loads((out / "stages.jsonl").read_text())
    assert record["exit"] == 124
    assert "allocation_deadline" not in record
    assert (out / "done/bounded").exists()


def test_allocation_truncated_stage_preserves_partial_output_and_resumes(campaign, tmp_path):
    _, function, out, env = campaign
    destination = out / "load/run"
    writer = tmp_path / "writer.py"
    writer.write_text(
        "from pathlib import Path\nimport signal,sys,time\n"
        "p=Path(sys.argv[3]); p.mkdir(parents=True,exist_ok=True)\n"
        "if sys.argv[1]=='partial':\n"
        " (p/'partial.txt').write_text('partial evidence')\n"
        " signal.signal(signal.SIGINT, lambda *_: sys.exit(130))\n"
        " time.sleep(30)\n"
        "else:\n (p/'complete.txt').write_text('complete')\n"
    )
    interrupted = "stage sample 30 " + shlex.join([sys.executable, str(writer), "partial", "--out", str(destination)])
    resumed = "stage sample 30 " + shlex.join([sys.executable, str(writer), "complete", "--out", str(destination)])
    shell = (function + "\n"
             f"DEADLINE_EPOCH={int(time.time()) + 2}\n{interrupted}\n"
             "[ ! -f \"$OUT/done/sample\" ] || exit 91\n"
             f"DEADLINE_EPOCH={int(time.time()) + 20}\n{resumed}\n")
    subprocess.run(["bash", "-c", shell], env=env, check=True, capture_output=True, text=True, timeout=8)
    assert (destination / "complete.txt").read_text() == "complete"
    assert [p.read_text() for p in (out / "interrupted").glob("*/partial.txt")] == ["partial evidence"]
    assert [p.read_text() for p in out.glob("sample-interrupted-*.log")]
    assert (out / "done/sample").exists()
    records = [json.loads(x) for x in (out / "stages.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["exit"] == 124 and records[0]["allocation_deadline"] is True
    assert records[0]["resumable"] is True and records[1]["exit"] == 0


def test_group_marker_requires_every_rate_marker(campaign):
    path, _, out, env = campaign
    helper = extract_function(path, "mark_group_complete")
    shell = (helper + "\n"
             "touch \"$OUT/done/serve\" \"$OUT/done/load-1\"\n"
             "if mark_group_complete serve load-1 load-2; then exit 91; fi\n"
             "[ -f \"$OUT/done/serve\" ] || exit 92\n"
             "touch \"$OUT/done/load-2\"\n"
             "mark_group_complete serve load-1 load-2\n")
    subprocess.run(["bash", "-c", shell], env=env, check=True, capture_output=True, text=True)
    assert (out / "done/serve").exists()


def test_second_campaign_is_refused_before_runtime_setup(campaign):
    path, _, out, env = campaign
    with (out / ".campaign.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(["bash", str(path)], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "already running" in result.stdout
    assert not list(out.glob("triton-runtime-*.json"))
