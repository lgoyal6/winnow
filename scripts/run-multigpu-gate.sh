#!/usr/bin/env bash
#
# THE multi-GPU gate. This is the only script in this repository that produces
# a data-parallel scaling number, and it refuses to run anywhere that cannot
# support one.
#
# Preconditions, checked and not assumed:
#   * nvidia-smi is present, and
#   * it reports at least two entries with distinct GPU UUIDs and distinct PCI
#     bus ids. Two MIG slices of one card share a bus id and are refused; one
#     card listed twice shares a UUID and is refused.
# Anything less exits 2 and writes no benchmark numbers.
#
# What it runs, at identical global batch size and sequence length so that the
# comparison holds global work fixed:
#   3 repeats of the 1-device configuration  (CUDA_VISIBLE_DEVICES=0)
#   3 repeats of the 2-device configuration  (CUDA_VISIBLE_DEVICES=0,1)
#   1 negative control: 2 devices with --inject-grad-sync-fault, which MUST
#     fail the cross-rank parameter checksum and exit non-zero
#   1 positive 2-device rerun, to show the failure was the planted fault
#
# Outputs, all under turboquant_kv/results/:
#   multigpu-inventory.json        the hardware this ran on
#   multigpu-benchmark.json        throughput, latency, memory, comm, scaling
#   multigpu-negative-control.json the planted fault and whether it was caught
#   multigpu-report.md             the readable summary
#
# A slowdown is a valid result. If two GPUs do not beat one on fixed global
# work, the report says so; nothing here converts that into a speedup.
#
# Usage:
#   ./scripts/run-multigpu-gate.sh
#   PYTHON=/path/to/venv/bin/python ./scripts/run-multigpu-gate.sh
#
# Exit codes: 0 gate passed; 1 a run or the control failed; 2 refused (no
# usable interpreter, or fewer than two physical GPUs).
set -uo pipefail

cd "$(dirname "$0")/.."
REPO="$PWD"
RESULTS="$REPO/turboquant_kv/results"

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
  exit 2
fi

echo "==> interpreter: $PY"
echo "==> [1/6] hardware inventory and the two-physical-GPU gate"
"$PY" turboquant_kv/ddp/inventory.py --out "$RESULTS/multigpu-inventory.json"
INVENTORY_EXIT=$?
if [[ "$INVENTORY_EXIT" -ne 0 ]]; then
  echo >&2
  echo "REFUSED: this host does not satisfy the two-physical-GPU requirement." >&2
  echo "No benchmark numbers were written. Run this on a host where" >&2
  echo "nvidia-smi lists at least two GPUs with distinct UUIDs and distinct" >&2
  echo "PCI bus ids." >&2
  exit 2
fi
echo

RUN_DIR="${RUN_DIR:-$RESULTS/runs}"
mkdir -p "$RUN_DIR"

# Frozen by turboquant_kv/ddp/manifest.json: proxy preset, global batch 32,
# sequence length 512, 5 warmup steps, 30 measured steps, 3 repeats per
# configuration. Global batch is held fixed across world sizes, so the
# per-rank batch is 32 at world size 1 and 16 at world size 2.
COMMON_ARGS=(
  --preset proxy-6l-512d
  --global-batch-size 32
  --seq-len 512
  --warmup-steps 5
  --measured-steps 30
)

echo "==> [2/6] 1-device configuration, 3 repeats"
CUDA_VISIBLE_DEVICES=0 "$PY" -m torch.distributed.run --nproc_per_node 1 \
  --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
  turboquant_kv/ddp/train_ddp.py "${COMMON_ARGS[@]}" \
  --label 1gpu-repeat1 --out "$RUN_DIR/1gpu-repeat1.json"
R1=$?
CUDA_VISIBLE_DEVICES=0 "$PY" -m torch.distributed.run --nproc_per_node 1 \
  --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
  turboquant_kv/ddp/train_ddp.py "${COMMON_ARGS[@]}" \
  --label 1gpu-repeat2 --out "$RUN_DIR/1gpu-repeat2.json"
R2=$?
CUDA_VISIBLE_DEVICES=0 "$PY" -m torch.distributed.run --nproc_per_node 1 \
  --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
  turboquant_kv/ddp/train_ddp.py "${COMMON_ARGS[@]}" \
  --label 1gpu-repeat3 --out "$RUN_DIR/1gpu-repeat3.json"
R3=$?
echo "    exits: $R1 $R2 $R3"
echo

echo "==> [3/6] 2-device configuration, 3 repeats"
CUDA_VISIBLE_DEVICES=0,1 "$PY" -m torch.distributed.run --nproc_per_node 2 \
  --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
  turboquant_kv/ddp/train_ddp.py "${COMMON_ARGS[@]}" \
  --label 2gpu-repeat1 --out "$RUN_DIR/2gpu-repeat1.json"
R4=$?
CUDA_VISIBLE_DEVICES=0,1 "$PY" -m torch.distributed.run --nproc_per_node 2 \
  --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
  turboquant_kv/ddp/train_ddp.py "${COMMON_ARGS[@]}" \
  --label 2gpu-repeat2 --out "$RUN_DIR/2gpu-repeat2.json"
R5=$?
CUDA_VISIBLE_DEVICES=0,1 "$PY" -m torch.distributed.run --nproc_per_node 2 \
  --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
  turboquant_kv/ddp/train_ddp.py "${COMMON_ARGS[@]}" \
  --label 2gpu-repeat3 --out "$RUN_DIR/2gpu-repeat3.json"
R6=$?
echo "    exits: $R4 $R5 $R6"
echo

if [[ "$R1$R2$R3$R4$R5$R6" != "000000" ]]; then
  echo "FAILED: at least one positive run did not end with matching" >&2
  echo "cross-rank checksums. Not aggregating a scaling number from runs" >&2
  echo "whose gradient synchronization is in question." >&2
  exit 1
fi

echo "==> [4/6] NEGATIVE CONTROL: 2 devices with --inject-grad-sync-fault"
CUDA_VISIBLE_DEVICES=0,1 "$PY" -m torch.distributed.run --nproc_per_node 2 \
  --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
  turboquant_kv/ddp/train_ddp.py "${COMMON_ARGS[@]}" \
  --inject-grad-sync-fault \
  --label 2gpu-negative-control --out "$RUN_DIR/negative-control.json"
CONTROL_EXIT=$?
echo "    exit=$CONTROL_EXIT (non-zero is required here)"
"$PY" turboquant_kv/ddp/report.py negative-control \
  --run "$RUN_DIR/negative-control.json" --exit-code "$CONTROL_EXIT" \
  --out "$RESULTS/multigpu-negative-control.json"
CONTROL_SCORE=$?
if [[ "$CONTROL_SCORE" -ne 0 ]]; then
  echo "FAILED: the planted gradient-sync fault was not caught." >&2
  exit 1
fi
echo

echo "==> [5/6] positive 2-device rerun after the control"
CUDA_VISIBLE_DEVICES=0,1 "$PY" -m torch.distributed.run --nproc_per_node 2 \
  --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
  turboquant_kv/ddp/train_ddp.py "${COMMON_ARGS[@]}" \
  --label 2gpu-positive-rerun --out "$RUN_DIR/positive-rerun.json"
RERUN_EXIT=$?
echo "    exit=$RERUN_EXIT"
if [[ "$RERUN_EXIT" -ne 0 ]]; then
  echo "FAILED: the positive rerun after the control did not pass." >&2
  exit 1
fi
echo

echo "==> [6/6] aggregating"
"$PY" turboquant_kv/ddp/report.py benchmark \
  --run "$RUN_DIR/1gpu-repeat1.json" \
  --run "$RUN_DIR/1gpu-repeat2.json" \
  --run "$RUN_DIR/1gpu-repeat3.json" \
  --run "$RUN_DIR/2gpu-repeat1.json" \
  --run "$RUN_DIR/2gpu-repeat2.json" \
  --run "$RUN_DIR/2gpu-repeat3.json" \
  --inventory "$RESULTS/multigpu-inventory.json" \
  --out "$RESULTS/multigpu-benchmark.json"
BENCH_EXIT=$?
if [[ "$BENCH_EXIT" -ne 0 ]]; then
  echo "FAILED: aggregation refused the runs." >&2
  exit 1
fi
"$PY" turboquant_kv/ddp/report.py markdown \
  --benchmark "$RESULTS/multigpu-benchmark.json" \
  --negative-control "$RESULTS/multigpu-negative-control.json" \
  --positive-rerun "$RUN_DIR/positive-rerun.json" \
  --out "$RESULTS/multigpu-report.md"

echo
echo "Gate PASSED. Measured on the physical GPUs listed in"
echo "$RESULTS/multigpu-inventory.json, one process per device, single host."
echo "Read $RESULTS/multigpu-report.md; the scaling verdict there is what the"
echo "runs measured, including if it is a slowdown."
exit 0
