#!/usr/bin/env bash
# Verify the official mjlab environment; this project does not create a venv.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
./scripts/python.sh -c 'import importlib.metadata as m; import zbot_rl_mjlab; print(zbot_rl_mjlab.TASK_ID); print({p: m.version(p) for p in ("mjlab", "mujoco", "rsl-rl-lib", "torch")})'
