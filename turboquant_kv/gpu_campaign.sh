#!/usr/bin/env bash
# Track 01 GPU campaign. Runs ON a single-GPU Linux host, from turboquant_kv/.
#
#   OUT=/home/evidence COHORT=/home/cohort PROTOCOL_RULES=/home/protocol-rules.json \
#   DEADLINE_EPOCH=<unix seconds> bash gpu_campaign.sh
#
# Every stage runs under `timeout`, logs to $OUT/<stage>.log and records its
# exit code in $OUT/stages.jsonl; a failing stage does not hide later stages'
# results, and nothing here turns a failure or skip into a pass. The SLO values
# are derived from the calibration split by the rule already written in
# PROTOCOL_RULES, then frozen into $OUT/protocol.json (hashed) before any final
# run starts. The script stops starting new stages after DEADLINE_EPOCH.
set -u
: "${OUT:?}" "${COHORT:?}" "${PROTOCOL_RULES:?}" "${DEADLINE_EPOCH:?}"
mkdir -p "$OUT/done"
exec 9>"$OUT/.campaign.lock"
flock -n 9 || { echo "campaign already running for $OUT"; exit 2; }
PY=${PY:-python}
$PY prepare_triton_runtime.py > "$OUT/triton-runtime-$(date -u +%Y%m%dT%H%M%SZ).json" || exit 3
REV=$($PY -c "import sys; sys.path.insert(0, '..'); from model_guard import pinned_revision; print(pinned_revision('Qwen/Qwen2.5-7B-Instruct'))")
rule() { $PY -c "import json,sys; print(json.load(open('$PROTOCOL_RULES'))$1)"; }

stage() {  # stage NAME TIMEOUT_S CMD...   (names starting report- ignore the deadline)
  local name=$1 limit=$2; shift 2
  [ -f "$OUT/done/$name" ] && return 0
  local remaining=$(( DEADLINE_EPOCH - $(date +%s) ))
  if [ "$remaining" -le 0 ] && [ "${name#report-}" = "$name" ]; then
    echo "{\"stage\": \"$name\", \"exit\": null, \"skipped\": \"deadline\"}" >> "$OUT/stages.jsonl"; return 0
  fi
  local allocation_capped=0
  if [ "${name#report-}" = "$name" ] && [ "$limit" -gt "$remaining" ]; then
    limit=$remaining
    allocation_capped=1
  fi
  # A pause may leave an output directory without its completion marker.
  # Preserve it outside load/ so a retry cannot overwrite or report it.
  local previous="" arg
  for arg in "$@"; do
    if [ "$previous" = --out ] && [ -e "$arg" ]; then
      mkdir -p "$OUT/interrupted"
      mv -- "$arg" "$OUT/interrupted/$name-$(date +%s)"
    fi
    previous=$arg
  done
  [ ! -f "$OUT/$name.log" ] || mv "$OUT/$name.log" "$OUT/$name-interrupted-$(date +%s).log"
  local t0=$(date +%s%N)
  timeout --signal=INT --kill-after=30 "$limit" "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  local elapsed_ms=$(( ($(date +%s%N) - t0) / 1000000 ))
  if [ "$allocation_capped" -eq 1 ] && [ "$elapsed_ms" -ge $((limit * 1000 - 100)) ] && { [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; }; then
    echo "{\"stage\": \"$name\", \"exit\": $rc, \"seconds\": $(( elapsed_ms / 1000 )), \"allocation_deadline\": true, \"resumable\": true}" >> "$OUT/stages.jsonl"
    return 0
  fi
  echo "{\"stage\": \"$name\", \"exit\": $rc, \"seconds\": $(( elapsed_ms / 1000 ))}" >> "$OUT/stages.jsonl"
  touch "$OUT/done/$name"
  return 0
}

mark_group_complete() {
  local group=$1; shift
  local marker
  for marker in "$@"; do
    [ -f "$OUT/done/$marker" ] || return 1
  done
  touch "$OUT/done/$group"
}

server_up() {  # server_up NAME KV BACKEND BUDGET_GIB PORT
  [ "$(date +%s)" -lt "$DEADLINE_EPOCH" ] || return 1
  $PY grpc_server.py --kv "$2" --decode-backend "$3" --kv-budget-gib "$4" --address "127.0.0.1:$5" \
      --max-batch "$(rule "['max_batch']")" --max-queue 256 --prefill-chunk 2048 \
      --max-prompt-tokens 8192 --max-new-tokens 128 > "$OUT/server-$1.log" 2>&1 &
  echo $! > "$OUT/server-$1.pid"
  for _ in $(seq 1 240); do
    [ "$(date +%s)" -lt "$DEADLINE_EPOCH" ] || return 1
    grep -q "^serving" "$OUT/server-$1.log" && return 0
    kill -0 "$(cat "$OUT/server-$1.pid")" 2>/dev/null || return 1
    sleep 2
  done
  return 1
}
server_down() { kill -INT "$(cat "$OUT/server-$1.pid")" 2>/dev/null; sleep 5; kill -9 "$(cat "$OUT/server-$1.pid")" 2>/dev/null; true; }

vllm_up() {  # vllm_up NAME KV_DTYPE BUDGET_GIB PORT
  [ "$(date +%s)" -lt "$DEADLINE_EPOCH" ] || return 1
  "$VLLM_PY" -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --revision "$REV" \
      --dtype bfloat16 --kv-cache-dtype "$2" --kv-cache-memory-bytes "$(( ${3%.*} * 1073741824 ))" \
      --max-num-seqs "$(rule "['max_batch']")" --max-model-len 8448 --port "$4" --host 127.0.0.1 \
      --no-enable-prefix-caching \
      --disable-log-requests > "$OUT/server-$1.log" 2>&1 &
  echo $! > "$OUT/server-$1.pid"
  for _ in $(seq 1 300); do
    [ "$(date +%s)" -lt "$DEADLINE_EPOCH" ] || return 1
    curl -sf "127.0.0.1:$4/health" >/dev/null && return 0
    kill -0 "$(cat "$OUT/server-$1.pid")" 2>/dev/null || return 1
    sleep 2
  done
  return 1
}

smi_log() { nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -lms 500 > "$1" & echo $!; }

# -- provenance ---------------------------------------------------------------
[ -f "$OUT/done/provenance" ] || {
nvidia-smi > "$OUT/nvidia-smi.txt" 2>&1
nvidia-smi topo -m > "$OUT/nvidia-topo.txt" 2>&1
$PY -c 'import importlib.metadata as m; print("\n".join(sorted(d.metadata["Name"]+"=="+d.version for d in m.distributions())))' > "$OUT/pip-freeze.txt"
$PY -c "import torch,triton;print(torch.__version__,torch.version.cuda,triton.__version__,torch.cuda.get_device_name())" > "$OUT/versions.txt" 2>&1
git -C .. rev-parse HEAD > "$OUT/source-head.txt" 2>/dev/null || true
sha256sum *.py proto/*.proto > "$OUT/source.sha256"
sha256sum "$COHORT/manifest.json" "$PROTOCOL_RULES" > "$OUT/inputs.sha256"
touch "$OUT/done/provenance"; }
nvidia-smi -q > "$OUT/launch-device-$(date -u +%Y%m%dT%H%M%SZ).txt" 2>&1

# -- correctness gates ----------------------------------------------------------
stage sanitizer_control 360 $PY sanitizer_control.py --out "$OUT/sanitizer-control"
stage pytest_gpu 1800 $PY -m pytest -q -rs test_packed_attention_triton.py
stage pytest_all 2400 $PY -m pytest -q -rs --ignore=test_ddp_gloo.py
stage sanitizer 1800 compute-sanitizer --tool memcheck --error-exitcode 1 \
      $PY -m pytest -q -x test_packed_attention_triton.py
stage model_contract 2400 $PY model_contract.py --model Qwen/Qwen2.5-7B-Instruct --device cuda \
      --rule "${MODEL_CONTRACT_RULE:-v3}" --long-prompt-tokens 6000 --out "$OUT/model-contract"

# Approved follow-up runs stop before expensive work on a failed correctness gate.
if [ "${STOP_ON_GATE_FAILURE:-0}" = 1 ]; then
  $PY -c 'import json,sys; rows=[json.loads(x) for x in open(sys.argv[1])]; latest={r.get("stage"): r for r in rows}; bad=[s for s in sys.argv[2:] if latest.get(s,{}).get("exit") != 0]; print("failed or missing correctness gates:", bad); sys.exit(bool(bad))' "$OUT/stages.jsonl" sanitizer_control pytest_gpu pytest_all sanitizer model_contract || exit 4
fi

# -- performance diagnosis --------------------------------------------------------
stage microbench 2400 $PY bench_packed_attention.py --out "$OUT/microbench" \
      --batches 1 4 16 32 --lens 512 2048 8192 --reps 7 --iters 50
stage profile 900 $PY profile_decode.py --out "$OUT/profile"

# -- calibration: derive and freeze SLOs -------------------------------------------
BUDGET=$(rule "['primary_kv_budget_gib']")
server_up cal-dense dense auto "$BUDGET" 50070 || { server_down cal-dense; exit 4; }
stage calibration 1200 $PY loadgen.py --workload "$COHORT/workload-calibration-128.json" \
      --target grpc://127.0.0.1:50070 --revision "$REV" --arm dense-cal \
      --concurrency 1 --ignore-eos --out "$OUT/calibration"
server_down cal-dense
stage freeze_protocol 120 $PY freeze_protocol.py --rules "$PROTOCOL_RULES" \
      --calibration "$OUT/calibration" --out "$OUT/protocol.json"
[ -f "$OUT/protocol.json" ] || { echo "protocol not frozen; stopping before final runs" >> "$OUT/stages.jsonl"; exit 1; }
sha256sum "$OUT/protocol.json" > "$OUT/protocol.sha256"

# -- quality ---------------------------------------------------------------------------
stage quality_run 3600 $PY quality_study.py run --manifest "$COHORT/manifest.json" --split final \
      --arms dense packed8 --kv-budget-gib 12 --out "$OUT/quality-run"
stage report-quality 300 $PY quality_study.py report --runs "$OUT/quality-run" --baseline dense \
      --margin "$(rule "['quality_non_inferiority_margin']")" --out "$OUT/quality"

# -- final load study (arm order rotated per repetition) -----------------------------
RATES=$(rule "['rates']" | tr -d '[],')
REPS=$(rule "['repetitions']")
for BUDGET_KEY in primary_kv_budget_gib secondary_kv_budget_gib; do
  BUDGET=$(rule "['$BUDGET_KEY']")
  ARMS="dense:dense:auto packed8-direct:packed8:auto packed8-reconstruct:packed8:reconstruct_sdpa"
  for rep in $(seq 1 "$REPS"); do
    [ $((rep % 2)) -eq 0 ] && ARMS=$(echo "$ARMS" | tr ' ' '\n' | tac | tr '\n' ' ')
    for spec in $ARMS; do
      IFS=: read -r name kv backend <<< "$spec"
      load_markers=()
      for rate in $RATES; do load_markers+=("load-$name-$BUDGET_KEY-$rate-$rep"); done
      mark_group_complete "serve-$name-$BUDGET_KEY-$rep" "${load_markers[@]}" && continue
      server_up "$name-$BUDGET_KEY-$rep" "$kv" "$backend" "$BUDGET" 50071 || { server_down "$name-$BUDGET_KEY-$rep"; continue; }
      for rate in $RATES; do
        d="$OUT/load/$BUDGET_KEY/$name/rate-$rate/rep-$rep"
        stage "load-$name-$BUDGET_KEY-$rate-$rep" 1500 $PY loadgen.py \
            --workload "$COHORT/workload-final-128.json" --target grpc://127.0.0.1:50071 \
            --revision "$REV" --arm "$name" --rate "$rate" --seed "$rep" --ignore-eos --out "$d"
        echo "{\"load\": \"$BUDGET_KEY@$rate/rep$rep\"}" > "$d/label.json" 2>/dev/null || true
      done
      server_down "$name-$BUDGET_KEY-$rep"
      mark_group_complete "serve-$name-$BUDGET_KEY-$rep" "${load_markers[@]}" || true
    done
    if [ -n "${VLLM_PY:-}" ]; then
      for spec in vllm-bf16:auto vllm-fp8:fp8; do
        IFS=: read -r name kvd <<< "$spec"
        for rate in $RATES; do
          d="$OUT/load/$BUDGET_KEY/$name/rate-$rate/rep-$rep"
          [ -f "$OUT/done/load-$name-$BUDGET_KEY-$rate-$rep" ] && continue
          vllm_up "$name-$BUDGET_KEY-$rate-$rep" "$kvd" "$BUDGET" 8000 || { server_down "$name-$BUDGET_KEY-$rate-$rep"; continue; }
          if [ -d "$d" ]; then
            mkdir -p "$OUT/interrupted"
            mv "$d" "$OUT/interrupted/load-$name-$BUDGET_KEY-$rate-$rep-$(date +%s)"
          fi
          SMI=$(mkdir -p "$d" && smi_log "$d/nvidia_smi.csv")
          stage "load-$name-$BUDGET_KEY-$rate-$rep" 1500 $PY loadgen.py \
              --workload "$COHORT/workload-final-128.json" --target openai://127.0.0.1:8000 \
              --model Qwen/Qwen2.5-7B-Instruct --arm "$name" --rate "$rate" --seed "$rep" \
              --ignore-eos --out "$d/run"
          kill "$SMI" 2>/dev/null
          mv "$d/run/"* "$d/" 2>/dev/null; rmdir "$d/run" 2>/dev/null
          echo "{\"load\": \"$BUDGET_KEY@$rate/rep$rep\"}" > "$d/label.json"
          server_down "$name-$BUDGET_KEY-$rate-$rep"
        done
      done
    fi
  done
done

# -- reports ------------------------------------------------------------------------------
for BUDGET_KEY in primary_kv_budget_gib secondary_kv_budget_gib; do
  stage "report-$BUDGET_KEY" 300 $PY report_packed_serving.py --protocol "$OUT/protocol.json" \
      --runs $(find "$OUT/load/$BUDGET_KEY" -name requests.jsonl -exec dirname {} \; | sort) \
      --out "$OUT/report-$BUDGET_KEY"
done
find "$OUT" -type f ! -name evidence.sha256 -exec sha256sum {} \; > "$OUT/evidence.sha256"
echo done > "$OUT/CAMPAIGN_DONE"
