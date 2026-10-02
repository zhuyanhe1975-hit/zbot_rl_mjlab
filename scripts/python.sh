#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MJLAB_ROOT="${MJLAB_ROOT:-/home/yhzhu/AI/mjlab}"
if [[ -n "${ZBOT_PYTHON:-}" ]]; then
    PYTHON_BIN="$ZBOT_PYTHON"
elif [[ -x "$MJLAB_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$MJLAB_ROOT/.venv/bin/python"
else
    PYTHON_BIN=python3
fi
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/zbot-mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/zbot-cache}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
cd "$PROJECT_DIR"
exec "$PYTHON_BIN" "$@"
