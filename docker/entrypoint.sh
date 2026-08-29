#!/usr/bin/env bash
# Thin wrapper: PATH + MuJoCo/X11 env. Packages live in the image at /opt/venv.
set -euo pipefail

export VIRTUAL_ENV="${VIRTUAL_ENV:-/opt/venv}"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export STABLEWM_HOME="${STABLEWM_HOME:-/workspace/ws_le-wm/stablewm}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export HOME="${HOME:-/workspace/docker/home}"
# Headless only when no X11. Dummy SDL breaks the MuJoCo GLFW viewer.
if [[ -z "${DISPLAY:-}" ]]; then
  export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
fi

mkdir -p "$HOME" "$STABLEWM_HOME" 2>/dev/null || true
cd /workspace/ws_le-wm/le-wm 2>/dev/null || cd /workspace
exec "$@"
