#!/usr/bin/env bash
# Shared GPU session: track 02 gates + track 01 carryover. Resumable.
#
#   OUT=/home/evidence02 COHORT=/home/cohort PROTOCOL=/home/protocol02.json \
#   CARRY=/home/track01-carryover-rules.json RULES01=/home/protocol-rules.json \
#   PROTOCOL01=/home/protocol01.json DEADLINE_EPOCH=<unix s> PY=... VLLM_PY=... bash gpu_campaign_02.sh
#
# Every stage writes $OUT/done/<stage> on completion and is skipped when that
# marker exists, so a provider pause loses at most the stage in flight; rerun
# the same command after resume. Exit codes go to $OUT/stages.jsonl.
set -u
: "${OUT:?}" "${COHORT:?}" "${PROTOCOL:?}" "${DEADLINE_EPOCH:?}"
mkdir -p "$OUT/done"
exec 9>"$OUT/.campaign.lock"
flock -n 9 || { echo "campaign already running for $OUT"; exit 2; }
PY=${PY:-python}
$PY prepare_triton_runtime.py > "$OUT/triton-runtime-$(date -u +%Y%m%dT%H%M%SZ).json" || exit 3
export HF_HOME=${HF_HOME:-/home/hf}
REV=$($PY -c "import sys; sys.path.insert(0, '..'); from model_guard import pinned_revision; print(pinned_revision('Qwen/Qwen2.5-7B-Instruct'))")
REG=$OUT/registry

stage() {  # stage NAME TIMEOUT_S CMD...   (report-* stages ignore the deadline)
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
  local previous="" arg
  for arg in "$@"; do
    if [ "$previous" = --out ] && [ -e "$arg" ]; then
      mkdir -p "$OUT/interrupted"
      mv -- "$arg" "$OUT/interrupted/$name-$(date +%s)"
    fi
    previous=$arg
  done
  [ ! -f "$OUT/$name.log" ] || mv "$OUT/$name.log" "$OUT/$name-interrupted-$(date +%s).log"
  local t0=$(date +%s)
  local t0_ns=$(date +%s%N)
  timeout --signal=INT --kill-after=30 "$limit" "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  local elapsed_ms=$(( ($(date +%s%N) - t0_ns) / 1000000 ))
  if [ "$allocation_capped" -eq 1 ] && [ "$elapsed_ms" -ge $((limit * 1000 - 100)) ] && { [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; }; then
    echo "{\"stage\": \"$name\", \"exit\": $rc, \"seconds\": $(( elapsed_ms / 1000 )), \"allocation_deadline\": true, \"resumable\": true}" >> "$OUT/stages.jsonl"
    return 0
  fi
  echo "{\"stage\": \"$name\", \"exit\": $rc, \"seconds\": $(( $(date +%s) - t0 )), \"at\": \"$(date -u +%H:%M:%S)\"}" >> "$OUT/stages.jsonl"
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

server_up() {  # NAME KV BUDGET_GIB PORT [registry]
  [ "$(date +%s)" -lt "$DEADLINE_EPOCH" ] || return 1
  WINNOW_TUNED_REGISTRY=${5:-} $PY grpc_server.py --kv "$2" --kv-budget-gib "$3" --address "127.0.0.1:$4" \
      --max-batch 16 --max-queue 256 --prefill-chunk 2048 --max-prompt-tokens 8192 \
      --max-new-tokens 128 > "$OUT/server-$1.log" 2>&1 &
  echo $! > "$OUT/server-$1.pid"
  for _ in $(seq 1 240); do
    [ "$(date +%s)" -lt "$DEADLINE_EPOCH" ] || return 1
    grep -q "^serving" "$OUT/server-$1.log" && return 0
    kill -0 "$(cat "$OUT/server-$1.pid")" 2>/dev/null || return 1
    sleep 2
  done
  return 1
}
vllm_up() {  # NAME KV_DTYPE BUDGET_GIB PORT
  [ "$(date +%s)" -lt "$DEADLINE_EPOCH" ] || return 1
  "$VLLM_PY" -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --revision "$REV" \
      --dtype bfloat16 --kv-cache-dtype "$2" --kv-cache-memory-bytes "$(( ${3%.*} * 1073741824 ))" \
      --no-enable-prefix-caching --max-num-seqs 16 --max-model-len 8448 --port "$4" --host 127.0.0.1 \
      > "$OUT/server-$1.log" 2>&1 &
  echo $! > "$OUT/server-$1.pid"
  for _ in $(seq 1 300); do
    [ "$(date +%s)" -lt "$DEADLINE_EPOCH" ] || return 1
    curl -sf "127.0.0.1:$4/health" >/dev/null && return 0
    kill -0 "$(cat "$OUT/server-$1.pid")" 2>/dev/null || return 1
    sleep 2
  done
  return 1
}
down() { kill -INT "$(cat "$OUT/server-$1.pid")" 2>/dev/null; sleep 5; kill -9 "$(cat "$OUT/server-$1.pid")" 2>/dev/null; true; }

# -- provenance --------------------------------------------------------------
[ -f "$OUT/done/provenance" ] || {
  nvidia-smi > "$OUT/nvidia-smi.txt" 2>&1
  $PY -c 'import importlib.metadata as m; print("\n".join(sorted(d.metadata["Name"]+"=="+d.version for d in m.distributions())))' > "$OUT/pip-freeze.txt"
  sha256sum *.py tsc/*.py proto/*.proto "$PROTOCOL" "$COHORT/manifest.json" ${CARRY:-} > "$OUT/source.sha256"
  touch "$OUT/done/provenance"; }
nvidia-smi -q > "$OUT/launch-device-$(date -u +%Y%m%dT%H%M%SZ).txt" 2>&1

# -- track 02 CUDA gates -------------------------------------------------------
stage sanitizer_control 360 $PY sanitizer_control.py --out "$OUT/sanitizer-control"
stage pytest_kernels 2400 $PY -m pytest -q -rs test_packed_attention_triton.py test_tsc_wrong_code.py
stage sanitizer_ppi 1800 compute-sanitizer --tool memcheck --error-exitcode 1 $PY -m pytest -q -x \
      test_packed_attention_triton.py -k "pages_per_iter and qwen7b"
stage pytest_all 2400 $PY -m pytest -q -rs --ignore=test_ddp_gloo.py

# Approved follow-up runs stop before expensive work on a failed correctness gate.
if [ "${STOP_ON_GATE_FAILURE:-0}" = 1 ]; then
  $PY -c 'import json,sys; rows=[json.loads(x) for x in open(sys.argv[1])]; latest={r.get("stage"): r for r in rows}; bad=[s for s in sys.argv[2:] if latest.get(s,{}).get("exit") != 0]; print("failed or missing correctness gates:", bad); sys.exit(bool(bad))' "$OUT/stages.jsonl" sanitizer_control pytest_kernels sanitizer_ppi pytest_all || exit 4
fi

# -- tuning (fresh Triton cache so first launches are cold compiles) ------------
stage tune 7200 env TRITON_CACHE_DIR="$OUT/triton-cache-tune" $PY tune_kernels.py tune \
      --protocol "$PROTOCOL" --registry "$REG" --out "$OUT/tune"
stage heldout 3600 $PY tune_kernels.py heldout --protocol "$PROTOCOL" --registry "$REG" --out "$OUT/heldout"

# -- real consumer end to end, fresh process per arm, order alternated ---------
for rep in 1 2; do
  ARMS="packed8:default packed8:tuned dense:default dense:tuned"
  [ "$rep" = 2 ] && ARMS="dense:tuned dense:default packed8:tuned packed8:default"
  for spec in $ARMS; do
    IFS=: read -r kv mode <<< "$spec"
    regarg=(); [ "$mode" = tuned ] && regarg=(--registry "$REG")
    stage "e2e-$kv-$mode-$rep" 1800 $PY consumer_e2e.py --manifest "$COHORT/manifest.json" --kv "$kv" \
          "${regarg[@]}" --out "$OUT/e2e/$kv-$mode-rep$rep"
  done
done

if [ "${TRACK01_CARRYOVER:-1}" = 1 ]; then
# -- track 01 carryover: v4 model contract with the production (tuned) path -----
stage contract_v4 2400 env WINNOW_TUNED_REGISTRY="$REG" $PY model_contract.py --model Qwen/Qwen2.5-7B-Instruct \
      --device cuda --long-prompt-tokens 6000 --rule v4 --out "$OUT/model-contract-v4"

# -- track 01 carryover: serving rerun at the primary 3 GiB budget ---------------
for rep in 1 2; do
  for spec in dense:dense packed8-direct:packed8; do
    IFS=: read -r name kv <<< "$spec"
    load_markers=()
    for rate in 0.5 1 2 4; do load_markers+=("load-$name-$rate-$rep"); done
    mark_group_complete "serve-$name-$rep" "${load_markers[@]}" && continue
    server_up "$name-$rep" "$kv" 3 50071 "$REG" || { down "$name-$rep"; continue; }
    for rate in 0.5 1 2 4; do
      d="$OUT/load/$name/rate-$rate/rep-$rep"
      stage "load-$name-$rate-$rep" 1500 $PY loadgen.py --workload "$COHORT/workload-final-128.json" \
          --target grpc://127.0.0.1:50071 --revision "$REV" --arm "$name" --rate "$rate" --seed "$rep" \
          --ignore-eos --out "$d"
      echo "{\"load\": \"primary@$rate/rep$rep\"}" > "$d/label.json" 2>/dev/null
    done
    down "$name-$rep"
    mark_group_complete "serve-$name-$rep" "${load_markers[@]}" || true
  done
done
if [ -n "${VLLM_PY:-}" ]; then
  for spec in vllm-bf16:auto vllm-fp8:fp8; do
    IFS=: read -r name kvd <<< "$spec"
    for rate in 0.5 1 2 4; do                  # fresh server per rate; prefix caching off
      d="$OUT/load/$name/rate-$rate/rep-1"
      [ -f "$OUT/done/load-$name-$rate-1" ] && continue
      vllm_up "$name-$rate" "$kvd" 3 8000 || { down "$name-$rate"; continue; }
      if [ -d "$d" ]; then
        mkdir -p "$OUT/interrupted"
        mv "$d" "$OUT/interrupted/load-$name-$rate-1-$(date +%s)"
      fi
      mkdir -p "$d"
      nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 500 > "$d/nvidia_smi.csv" & SMI=$!
      stage "load-$name-$rate-1" 1500 $PY loadgen.py --workload "$COHORT/workload-final-128.json" \
          --target openai://127.0.0.1:8000 --model Qwen/Qwen2.5-7B-Instruct --arm "$name" --rate "$rate" \
          --seed 1 --ignore-eos --out "$d/run"
      kill $SMI 2>/dev/null; mv "$d/run/"* "$d/" 2>/dev/null; rmdir "$d/run" 2>/dev/null
      echo "{\"load\": \"primary@$rate/rep1\"}" > "$d/label.json"
      down "$name-$rate"
    done
  done
fi

fi  # TRACK01_CARRYOVER

# -- reports (run even past the deadline) -----------------------------------------
[ -n "${PROTOCOL01:-}" ] && stage report-serving 300 $PY report_packed_serving.py --protocol "$PROTOCOL01" \
      --runs $(find "$OUT/load" -name requests.jsonl -exec dirname {} \; | sort) --out "$OUT/report-serving"
find "$OUT" -type f ! -name evidence.sha256 ! -path "*/triton-cache-*" -exec sha256sum {} \; > "$OUT/evidence.sha256"
echo done > "$OUT/CAMPAIGN_DONE"
