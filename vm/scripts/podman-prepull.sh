#!/usr/bin/env bash
set -euo pipefail

STATUS_FILE=/run/podman-prepull.status
STATE_DIR=/var/lib/podman-prepull
STATE_FILE=$STATE_DIR/state
LOG_TAG="podman-prepull"

images=(
    "ghcr.io/mlflow/mlflow:v3.11.1-full"
    "docker.io/chromadb/chroma:1.5.8"
    "docker.io/ollama/ollama:0.24.0"
)

total=${#images[@]}
i=0
current_image=""

timestamp() {
    date --iso-8601=seconds
}

write_status() {
    local state="$1"
    local message="$2"

    mkdir -p "$STATE_DIR"
    printf '%s\n' "$state" >"$STATE_FILE"
    cat >"$STATUS_FILE" <<EOF
FCOS initial container pre-pull
State: $state
Time: $(timestamp)
Message: $message
EOF
}

log() {
    local message="$1"
    printf '[%s] %s\n' "$(timestamp)" "$message"
    /usr/bin/logger -t "$LOG_TAG" "$message"
}

on_error() {
    local exit_code=$?
    local failed_image="${current_image:-unknown image}"

    write_status "FAILED" "Pull failed while processing: $failed_image"
    log "FAILED while processing: $failed_image"
    exit "$exit_code"
}

trap on_error ERR

if [ -f "$STATE_FILE" ] && grep -q '^DONE$' "$STATE_FILE"; then
    write_status "DONE" "Initial image pre-pull already completed in a previous run"
    log "State is DONE already; skipping image pre-pull"
    exit 0
fi

write_status "RUNNING" "Starting initial image pre-pull (0/$total complete)"
log "Starting initial image pre-pull ($total images)"

for img in "${images[@]}"; do
    i=$((i + 1))
    current_image="$img"

    write_status "RUNNING" "[$i/$total] Currently pulling $img"
    log "[$i/$total] pulling $img"

    /usr/bin/podman pull "$img"

    write_status "RUNNING" "[$i/$total] Finished pulling $img"
    log "[$i/$total] finished $img"
done

write_status "DONE" "Initial image pre-pull completed successfully ($total/$total complete)"
log "Initial image pre-pull completed successfully"
