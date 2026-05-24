#!/usr/bin/env bash
# Runs ON the RunPod pod. Lifecycle:
#   auth gcloud (service account) -> clone repo -> pip install -> pull prior results
#   from GCS (resume) -> run the command (periodic push in background) -> on exit,
#   final push of results + worker.log to GCS -> self-terminate the pod.
#
# Always pushes state + log to GCS before terminating, so even a crash leaves the
# partial results (resumable) and the log (debuggable) in the bucket. Env vars are
# injected by launch.py:
#   GCS_DEST          gs://<bucket>/<exp-name>
#   REPO_URL REPO_REF WORKDIR RUN_CMD
#   GCP_SA_KEY_B64    base64 of the GCP service-account JSON key
#   PUSH_INTERVAL     seconds between background pushes
#   AUTO_TERMINATE    "1" to remove the pod on exit
#   RUNPOD_API_KEY    for self-termination     (RUNPOD_POD_ID is set by RunPod)
set -uo pipefail

WORK=/workspace
mkdir -p "$WORK"
LOG="$WORK/worker.log"
# Mirror everything to a log file we can ship to GCS for post-mortem debugging.
exec > >(tee -a "$LOG") 2>&1

RESULTS_LOCAL=""        # set once we know the repo workdir
PUSH_PID=""

log() { echo "[worker $(date -u +%H:%M:%S)] $*"; }

sync_up() {
  [ -n "$RESULTS_LOCAL" ] && [ -d "$RESULTS_LOCAL" ] || return 0
  gsutil -m rsync -r "$RESULTS_LOCAL" "$GCS_DEST/results" || true
}
sync_down() {
  [ -n "$RESULTS_LOCAL" ] || return 0
  mkdir -p "$RESULTS_LOCAL"
  # only pull if a prior run actually exists in the bucket (quiet on first run)
  gsutil ls "$GCS_DEST/results/" >/dev/null 2>&1 || { log "no prior results; fresh run"; return 0; }
  gsutil -m rsync -r "$GCS_DEST/results" "$RESULTS_LOCAL" || true
}

terminate_self() {
  [ "${AUTO_TERMINATE:-0}" = "1" ] || { log "auto_terminate off; leaving pod up"; return; }
  [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ] || { log "no pod id/key; cannot self-terminate"; return; }
  log "terminating pod $RUNPOD_POD_ID"
  curl -s --request POST --header "Content-Type: application/json" \
    --url "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
    --data "{\"query\":\"mutation { podTerminate(input: {podId: \\\"${RUNPOD_POD_ID}\\\"}) }\"}" || true
}

cleanup() {
  local code=$?
  log "exit code $code; final sync + log upload"
  [ -n "$PUSH_PID" ] && kill "$PUSH_PID" 2>/dev/null || true
  sync_up
  gsutil -m cp "$LOG" "$GCS_DEST/worker.log" || true
  terminate_self
}
trap cleanup EXIT

log "authenticating service account"
echo "$GCP_SA_KEY_B64" | base64 -d > "$WORK/sa.json"
gcloud auth activate-service-account --key-file="$WORK/sa.json" --quiet

log "installing git + pip"
apt-get update -qq && apt-get install -y -qq git python3-pip curl >/dev/null

log "cloning $REPO_URL @ $REPO_REF"
git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$WORK/repo"
cd "$WORK/repo/$WORKDIR"
RESULTS_LOCAL="$(pwd)/results"

log "installing package"
# The base image's Python is externally-managed (PEP 668); the pod is ephemeral so
# installing into the system env is fine. Fail fast if it doesn't install, so we
# never run the experiment with missing deps.
python3 -m pip install --break-system-packages -q -e . || { log "FATAL: pip install failed"; exit 1; }

log "pulling prior results from $GCS_DEST/results (resume)"
sync_down

log "starting periodic push every ${PUSH_INTERVAL}s"
( while true; do sleep "$PUSH_INTERVAL"; sync_up; done ) &
PUSH_PID=$!

log "RUN: $RUN_CMD"
bash -lc "$RUN_CMD"
log "command finished"
# trap cleanup() runs next: final sync + terminate
