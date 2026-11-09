#!/usr/bin/env bash
# Local driver for the track 01 GPU campaign on one JarvisLabs A100.
#
#   AUTHORIZED=yes MAX_USD=12 MAX_HOURS=5 JL=/path/to/jl bash jarvis_campaign.sh
#
# Refuses to start without AUTHORIZED=yes and explicit MAX_USD/MAX_HOURS. Reads
# balance and A100 price first and refuses if price * MAX_HOURS (+ storage) can
# exceed MAX_USD or the balance. A local watchdog destroys the instance at
# MAX_HOURS whatever happens; the normal path downloads evidence, destroys the
# instance and checks the instance list is empty. The API key comes from the
# jl CLI's own login (jl setup) and is never echoed.
#
# Not yet executed: written while jl had no credentials on this machine, so the
# JSON field names parsed below are unverified and the script fails closed on
# anything it cannot parse.
set -euo pipefail
[ "${AUTHORIZED:-}" = "yes" ] || { echo "refusing: set AUTHORIZED=yes after a user-approved budget"; exit 2; }
: "${MAX_USD:?}" "${MAX_HOURS:?}"
JL=${JL:-jl}
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=$HOME/.codex/work/project-extensions/01-winnow
RUN=$BASE/runs/gpu-$(date -u +%Y%m%dT%H%M%SZ)
COHORT=${COHORT:?path to the prepared cohort dir}
RULES=${RULES:?path to protocol-rules.json}
mkdir -p "$RUN"

"$JL" status --json > "$RUN/jl-status.json"
"$JL" gpus --json > "$RUN/jl-gpus.json"
python3 - "$RUN" "$MAX_USD" "$MAX_HOURS" <<'PY'
import json, sys
run, max_usd, hours = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
status = json.load(open(f"{run}/jl-status.json"))
gpus = json.load(open(f"{run}/jl-gpus.json"))
bal = next((v for k, v in (status.items() if isinstance(status, dict) else []) if "balance" in k.lower()), None)
rows = gpus if isinstance(gpus, list) else gpus.get("gpus", [])
prices = [r.get("price_per_hour") or r.get("price") for r in rows if "A100" in json.dumps(r)]
prices = [float(p) for p in prices if p is not None]
if bal is None or not prices:
    sys.exit(f"cannot parse balance/price (balance={bal}, prices={prices}); refusing")
worst = max(prices) * hours + 0.05 * hours          # + storage allowance
if worst > max_usd or float(bal) < max_usd:
    sys.exit(f"refusing: worst-case ${worst:.2f} vs cap ${max_usd}, balance ${bal}")
json.dump({"balance": bal, "a100_price_per_hour": max(prices), "worst_case_usd": worst},
          open(f"{run}/budget-check.json", "w"), indent=2)
PY

ID=$("$JL" create --gpu A100 --storage 100 --name "winnow-01-$(date +%s)" --json | python3 -c \
     "import json,sys; d=json.load(sys.stdin); print(d.get('machine_id') or d.get('id'))")
echo "$ID" > "$RUN/instance-id.txt"
( sleep "$(python3 -c "print(int($MAX_HOURS*3600))")"; "$JL" destroy -y "$ID" > "$RUN/watchdog-destroy.log" 2>&1 ) &
WATCHDOG=$!
cleanup() {
  "$JL" destroy -y "$ID" > "$RUN/destroy.log" 2>&1 || true
  kill "$WATCHDOG" 2>/dev/null || true
  "$JL" list --json > "$RUN/jl-list-after.json" 2>&1 || true
}
trap cleanup EXIT

# Source bundle: tracked tree plus this track's uncommitted files, hashed.
( cd "$HERE/.." && git ls-files -co --exclude-standard | grep -v '^\.agent-work/' \
    | tar czf "$RUN/source.tgz" -T - )
shasum -a 256 "$RUN/source.tgz" > "$RUN/source.tgz.sha256"
"$JL" upload "$ID" "$RUN/source.tgz" /home/source.tgz
tar czf "$RUN/cohort.tgz" -C "$(dirname "$COHORT")" "$(basename "$COHORT")"
"$JL" upload "$ID" "$RUN/cohort.tgz" /home/cohort.tgz
"$JL" upload "$ID" "$RULES" /home/protocol-rules.json

DEADLINE=$(( $(date +%s) + $(python3 -c "print(int(($MAX_HOURS - 0.5) * 3600))") ))
"$JL" exec "$ID" "set -e; cd /home && mkdir -p src && tar xzf source.tgz -C src && tar xzf cohort.tgz && \
  python3 -m venv /home/venv && /home/venv/bin/pip install -q -r src/turboquant_kv/requirements-packed-gpu.txt && \
  python3 -m venv /home/vllm && (/home/vllm/bin/pip install -q vllm || true) && \
  cd src/turboquant_kv && OUT=/home/evidence COHORT=/home/$(basename "$COHORT") \
  PROTOCOL_RULES=/home/protocol-rules.json DEADLINE_EPOCH=$DEADLINE PY=/home/venv/bin/python \
  VLLM_PY=\$( [ -x /home/vllm/bin/python ] && /home/vllm/bin/python -c 'import vllm' 2>/dev/null && echo /home/vllm/bin/python ) \
  bash gpu_campaign.sh" > "$RUN/campaign-exec.log" 2>&1 || true
"$JL" download -r "$ID" /home/evidence "$RUN/evidence"
echo "evidence in $RUN/evidence"
