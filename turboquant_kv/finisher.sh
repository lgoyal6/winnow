#!/usr/bin/env bash
# Local evidence-first finisher for a Jarvis GPU campaign; runs without the agent.
#
#   JL=/path/to/jl RUN=<local run dir> REMOTE_OUT=/home/evidence02 HARD_STOP_EPOCH=<unix s> bash finisher.sh
#
# RUN/instance-id.txt holds the current machine id (rewrite it after a resume).
# Every 5 minutes:
#   * instance paused by the provider  -> leave it paused (disk kept); the agent resumes it
#   * CAMPAIGN_DONE present remotely   -> download REMOTE_OUT, verify its evidence.sha256,
#                                         then destroy and confirm the instance is gone
#   * HARD_STOP_EPOCH reached          -> try the same download; destroy only if it verified,
#                                         otherwise pause so the evidence survives on disk
# Written after the 2026-11-04 incident where a watchdog destroyed the instance, and the
# finished evidence with it, while the agent was unavailable.
set -u
: "${JL:?}" "${RUN:?}" "${REMOTE_OUT:?}" "${HARD_STOP_EPOCH:?}"
LOG=$RUN/finisher.log
say() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$LOG"; }

fetch() {  # -> 0 when the local copy verifies against the remote manifest
  local id=$1 dest=$RUN/evidence-$(date -u +%H%M%S)
  timeout 1800 "$JL" download -r "$id" "$REMOTE_OUT" "$dest" --json > "$dest.download.json" 2>&1 || {
    say "download failed: $(tail -c 200 "$dest.download.json")"; return 1; }
  [ -f "$dest/evidence.sha256" ] || { say "no evidence.sha256 in $dest"; return 1; }
  ( cd "$dest" && sed "s|$REMOTE_OUT/|./|" evidence.sha256 | shasum -a 256 -c --quiet ) > "$dest.verify.txt" 2>&1 \
    || { say "hash verification failed, see $dest.verify.txt"; return 1; }
  say "evidence verified in $dest ($(wc -l < "$dest/evidence.sha256") files)"
  echo "$dest" > "$RUN/EVIDENCE_DIR"
  return 0
}

destroy() {
  local id=$1
  for i in 1 2 3 4 5; do "$JL" destroy "$id" --yes --json >> "$RUN/destroy.json" 2>&1 && break; sleep 20; done
  sleep 10
  if "$JL" list --json 2>/dev/null | python3 -c "import json,sys; sys.exit(any(x['machine_id']==$id for x in json.load(sys.stdin)))"; then
    say "destroyed $id; instance list no longer contains it"; else say "WARNING: $id still listed after destroy"; fi
}

while true; do
  id=$(cat "$RUN/instance-id.txt")
  status=$("$JL" list --json 2>/dev/null | python3 -c \
    "import json,sys; print(next((x['status'] for x in json.load(sys.stdin) if x['machine_id']==$id), 'absent'))") \
    || { sleep 60; continue; }
  if [ "$status" = "absent" ]; then say "instance $id absent; finisher exits"; exit 0; fi
  if [ "$(date +%s)" -ge "$HARD_STOP_EPOCH" ]; then
    say "hard stop reached for $id ($status)"
    if [ "$status" = "Running" ] && fetch "$id"; then destroy "$id"; exit 0; fi
    [ "$status" = "Running" ] && "$JL" pause "$id" --yes --json >> "$RUN/pause.json" 2>&1
    say "evidence not secured: left $id paused (disk billed) for manual retrieval"; exit 2
  fi
  if [ "$status" = "Running" ]; then
    done_flag=$("$JL" exec "$id" --json -- sh -lc "test -f $REMOTE_OUT/CAMPAIGN_DONE && echo yes" 2>/dev/null \
      | python3 -c "import json,sys; print(json.load(sys.stdin).get('stdout','').strip())" 2>/dev/null)
    if [ "$done_flag" = "yes" ]; then
      say "campaign done on $id; retrieving"
      if fetch "$id"; then destroy "$id"; exit 0; fi
    fi
  fi
  sleep 300
done
