#!/usr/bin/env bash
#
# stop_qwen38.sh — cleanly stop the Qwen SGLang container on the DGX Spark,
# and optionally force-reclaim leaked GPU unified memory (no reboot needed).
#
set -euo pipefail

readonly CONTAINER="vllm-qwen38-sglang"
readonly STOP_TIMEOUT=60

usage() {
  cat <<'EOF'
stop_qwen38.sh - cleanly stop the Qwen SGLang container on the DGX Spark

Usage: stop_qwen38.sh [OPTIONS]

Options:
  -h, --help    Show this help and exit.
  --reclaim     After stopping the container, if no other GPU container is
                running, force-reclaim leaked unified memory without a reboot:
                  sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm
                WARNING: requires sudo, and ALL GPU processes must be stopped
                first. If any other GPU container is still running, the
                reclamation is skipped.

Behavior:
  - Stops vllm-qwen38-sglang with: docker stop -t 60
    The long timeout lets SGLang release GPU/UVM memory cleanly, avoiding
    the GB10 memory leak.
  - Skips the stop gracefully if the container is not running.
  - Prints "free -g" before and after the operation.
EOF
}

# --- argument parsing -------------------------------------------------------

reclaim=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --reclaim)
      reclaim=1
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

# --- helpers ----------------------------------------------------------------

log() { printf '[stop_qwen38] %s\n' "$*"; }

require_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    log "error: docker not found on PATH." >&2
    exit 1
  fi
  if ! docker info >/dev/null 2>&1; then
    log "error: cannot talk to the docker daemon." >&2
    exit 1
  fi
}

container_state() {
  local status
  status="$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || true)"
  if [[ -n "$status" ]]; then
    printf '%s\n' "$status"
  else
    printf 'missing\n'
  fi
}

print_memory() {
  log "$1:"
  if command -v free >/dev/null 2>&1; then
    free -g
  else
    log "(free -g not available on this host)"
  fi
}

# Returns 0 if at least one running container looks like it uses the GPU
# (nvidia runtime, or nvidia/cuda/gpu/sglang/vllm in image or name).
has_running_gpu_container() {
  local line id image name runtime
  local found=0
  while IFS= read -r line; do
    IFS=$'\t' read -r id image name <<< "$line"
    runtime="$(docker inspect -f '{{.HostConfig.Runtime}}' "$id" 2>/dev/null || true)"
    case "$runtime" in
      nvidia*) found=1 ;;
    esac
    case "${image}${name}" in
      *nvidia*|*cuda*|*gpu*|*sglang*|*vllm*) found=1 ;;
    esac
  done < <(docker ps --format '{{.ID}}\t{{.Image}}\t{{.Names}}' 2>/dev/null || true)
  [[ $found -eq 1 ]]
}

# --- main -------------------------------------------------------------------

main() {
  require_docker

  print_memory "Memory before"

  # Stop the SGLang container (gracefully skip if not running).
  local state
  state="$(container_state)"
  if [[ "$state" == "running" ]]; then
    log "Stopping ${CONTAINER} (docker stop -t ${STOP_TIMEOUT}) ..."
    docker stop -t "$STOP_TIMEOUT" "$CONTAINER"
    log "Stopped ${CONTAINER}."
  else
    log "Container ${CONTAINER} is not running (state: ${state}) — skipping stop."
  fi

  # Optionally reclaim leaked unified memory by reloading nvidia_uvm.
  if (( reclaim )); then
    if has_running_gpu_container; then
      log "WARNING: another GPU container is still running — skipping nvidia_uvm reload."
      docker ps --format '  running: {{.Names}} ({{.Image}})' || true
    else
      log "WARNING: reclamation requires sudo and ALL GPU processes stopped first."
      log "Reclaiming leaked unified memory: sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm ..."
      if sudo rmmod nvidia_uvm; then
        sudo modprobe nvidia_uvm
        log "nvidia_uvm reloaded — leaked unified memory reclaimed."
      else
        log "nvidia_uvm was not loaded — nothing to reclaim."
      fi
    fi
  fi

  print_memory "Memory after"
}

main
