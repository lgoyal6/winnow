#!/usr/bin/env bash
#
# LOCAL PROXY for the multi-GPU gate. This is not the gate.
#
# Runs the DDP harness with `torchrun --nproc_per_node 2` on CPU with the gloo
# backend: two processes, ONE host, NO GPU. What it proves:
#
#   1. positive run: two ranks that synchronize gradients end with
#      bit-identical parameters, and the harness says so and exits 0;
#   2. negative control: with --inject-grad-sync-fault, rank 1 applies its
#      local unreduced gradient for one bucket, the cross-rank checksum
#      MISMATCHES, and the run exits non-zero;
#   3. positive rerun: with the fault off again, the run passes and reproduces
#      the positive run's final checksum, so the failure was the fault and not
#      the harness drifting.
#
# What it does NOT prove, stated here because the number is easy to misquote:
# nothing about multi-GPU performance. Two processes on one CPU are not two
# GPUs. The step latency and communication percentage this writes are a
# measurement-path exercise, not a scaling, speedup, or throughput claim.
# Only ./scripts/run-multigpu-gate.sh produces a scaling number, and it
# refuses to run without two distinct physical GPUs.
#
# Usage:
#   ./scripts/run-ddp-cpu-selftest.sh
#   PYTHON=/path/to/venv/bin/python ./scripts/run-ddp-cpu-selftest.sh
#   RUN_DIR=/tmp/selftest ./scripts/run-ddp-cpu-selftest.sh   # keep raw logs
#
# Exit codes: 0 all three conditions held; 1 a condition failed; 2 no usable
# interpreter (no torch).
set -uo pipefail

cd "$(dirname "$0")/.."
REPO="$PWD"
RESULTS="$REPO/turboquant_kv/results"
OUT_JSON="$RESULTS/ddp-cpu-gloo-selftest.json"

# Interpreter resolution, in the same spirit as run.sh: an explicit PYTHON
# wins, then a plain python3, then a venv beside or above the repo.
PY="${PYTHON:-python3}"
if ! "$PY" -c "import torch" > /dev/null 2>&1; then
  for CANDIDATE in "$REPO/.venv/bin/python" "$REPO/../.venv/bin/python"; do
    if [[ -x "$CANDIDATE" ]] && "$CANDIDATE" -c "import torch" > /dev/null 2>&1; then
      PY="$CANDIDATE"
      break
    fi
  done
fi
if ! "$PY" -c "import torch" > /dev/null 2>&1; then
  echo "REFUSING: no interpreter with torch was found." >&2
  echo "  tried: PYTHON=${PYTHON:-unset}, python3, ./.venv/bin/python, ../.venv/bin/python" >&2
  echo "  set PYTHON to an interpreter that has torch installed and rerun." >&2
  exit 2
fi

RUN_DIR="${RUN_DIR:-$(mktemp -d)}"
mkdir -p "$RUN_DIR"

echo "==> interpreter: $PY"
"$PY" -c "import torch; print('    torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"
echo "==> raw run records and logs: $RUN_DIR"
echo "==> two processes, ONE host, gloo on CPU. Not a multi-GPU result."
echo

# The workload is deliberately tiny and short: this script is a correctness
# gate that has to be runnable on a laptop, not a benchmark. bucket-cap-mb is
# small so the model spans several DDP buckets, which is what makes "the fault
# hits exactly one bucket" a real statement.
COMMON_ARGS=(
  --preset tiny-2l-128d
  --global-batch-size 8
  --seq-len 32
  --warmup-steps 5
  --measured-steps 10
  --bucket-cap-mb 0.5
)
LAUNCH=(
  "$PY" -m torch.distributed.run
  --nproc_per_node 2
  --rdzv-backend=c10d
  --rdzv-endpoint=localhost:0
  turboquant_kv/ddp/train_ddp.py
)

echo "==> [1/4] positive run (gradient sync intact), expecting exit 0"
"${LAUNCH[@]}" "${COMMON_ARGS[@]}" \
  --label cpu-gloo-positive \
  --out "$RUN_DIR/positive.json" 2>&1 | tee "$RUN_DIR/positive.log"
POSITIVE_EXIT=${PIPESTATUS[0]}
echo "    exit=$POSITIVE_EXIT"
echo

echo "==> [2/4] NEGATIVE CONTROL (--inject-grad-sync-fault), expecting non-zero"
"${LAUNCH[@]}" "${COMMON_ARGS[@]}" \
  --inject-grad-sync-fault \
  --label cpu-gloo-negative-control \
  --out "$RUN_DIR/negative.json" 2>&1 | tee "$RUN_DIR/negative.log"
NEGATIVE_EXIT=${PIPESTATUS[0]}
echo "    exit=$NEGATIVE_EXIT"
echo

echo "==> [3/4] positive rerun after the control, expecting exit 0"
"${LAUNCH[@]}" "${COMMON_ARGS[@]}" \
  --label cpu-gloo-positive-rerun \
  --out "$RUN_DIR/positive-rerun.json" 2>&1 | tee "$RUN_DIR/positive-rerun.log"
RERUN_EXIT=${PIPESTATUS[0]}
echo "    exit=$RERUN_EXIT"
echo

if [[ "$POSITIVE_EXIT" -ne 0 ]]; then
  echo "FAILED: the positive run did not pass (exit $POSITIVE_EXIT)." >&2
  exit 1
fi
if [[ "$NEGATIVE_EXIT" -eq 0 ]]; then
  echo "FAILED: the planted gradient-sync fault was NOT caught: the injected" >&2
  echo "run exited 0. The checksum gate is not doing its job." >&2
  exit 1
fi
if [[ "$RERUN_EXIT" -ne 0 ]]; then
  echo "FAILED: the positive rerun after the control did not pass (exit $RERUN_EXIT)." >&2
  exit 1
fi

echo "==> [4/4] scoring the self-test into $OUT_JSON"
"$PY" turboquant_kv/ddp/report.py cpu-selftest \
  --positive "$RUN_DIR/positive.json" --positive-exit "$POSITIVE_EXIT" \
  --negative "$RUN_DIR/negative.json" --negative-exit "$NEGATIVE_EXIT" \
  --positive-rerun "$RUN_DIR/positive-rerun.json" \
  --positive-rerun-exit "$RERUN_EXIT" \
  --out "$OUT_JSON"
SCORE_EXIT=$?
if [[ "$SCORE_EXIT" -ne 0 ]]; then
  echo "FAILED: the scored self-test did not satisfy every condition." >&2
  exit 1
fi

echo
echo "CPU gloo self-test PASSED: positive run matched checksums, the planted"
echo "gradient-sync fault was caught, and the positive rerun passed again."
echo "This is a single-host CPU gloo self-test. It is NOT a multi-GPU result."
exit 0
