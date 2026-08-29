#!/usr/bin/env bash
# Dedicated GPU container — never `compose run` (that fights container_name).
# Image already has the venv. This script starts the named container and execs.
#
#   ./run.sh              start if needed, then interactive bash
#   ./run.sh up           start (no rebuild); survives reboot via unless-stopped
#   ./run.sh up --build   rebuild image then start
#   ./run.sh exec <cmd>   exec in the running container
#   ./run.sh <cmd...>     same as exec (after ensuring the container is up)
#   ./run.sh stop         docker compose stop  (container remains; restart: unless-stopped)
#   ./run.sh start        docker compose start (after stop)
#
# Do not `compose down` unless you intend to remove the container.
# Bind-mounted /workspace is the source of truth for code and results.
set -euo pipefail
cd "$(dirname "$0")"

export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"

mkdir -p home

if [[ -n "${DISPLAY:-}" ]]; then
  xhost +SI:localuser:"$(id -un)" >/dev/null 2>&1 || xhost +local: >/dev/null 2>&1 || true
fi

wait_running() {
  local i
  for i in $(seq 1 60); do
    if docker compose ps --status running --services 2>/dev/null | grep -qx lewm; then
      return 0
    fi
    sleep 1
  done
  echo "[lewm-docker] container did not start:" >&2
  docker compose logs --tail 80 lewm >&2
  return 1
}

ensure_up() {
  local build_flag="${1:-}"
  if docker compose ps --status running --services 2>/dev/null | grep -qx lewm; then
    return 0
  fi
  if [[ "$build_flag" == "--build" ]]; then
    docker compose up -d --build
  else
    docker compose up -d
  fi
  wait_running
}

if [[ "${1:-}" == "stop" ]]; then
  exec docker compose stop
fi

if [[ "${1:-}" == "start" ]]; then
  docker compose start
  wait_running
  echo "container weltmodelle-lewm is running.  ./run.sh exec bash"
  exit 0
fi

if [[ "${1:-}" == "up" ]]; then
  shift
  ensure_up "${1:-}"
  echo "container weltmodelle-lewm is up (restart: unless-stopped).  ./run.sh exec bash"
  exit 0
fi

if [[ "${1:-}" == "exec" ]]; then
  shift
  ensure_up
  if [[ $# -eq 0 ]]; then
    set -- bash
  fi
  exec docker compose exec lewm "$@"
fi

ensure_up
if [[ $# -eq 0 ]]; then
  exec docker compose exec lewm bash
fi
exec docker compose exec lewm "$@"
