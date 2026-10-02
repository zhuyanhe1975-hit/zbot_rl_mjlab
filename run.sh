#!/usr/bin/env bash
set -euo pipefail

# 默认加载 32 个机器人；支持 ./run.sh 64 或 ./run.sh --num-envs 64。
NUM_ENVS=32
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    NUM_ENVS="$1"
    shift
fi
PROJECT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
LOG_ROOT="$PROJECT_DIR/logs/rsl_rl/zbot_g1_velocity"
for arg in "$@"; do
    if [[ "$arg" == "--log-root" || "$arg" == --log-root=* ]]; then
        LOG_ROOT=""
        break
    fi
done
if [[ -n "$LOG_ROOT" ]]; then
    set -- --log-root "$LOG_ROOT" "$@"
fi
exec "$PROJECT_DIR/scripts/python.sh" -m zbot_rl_mjlab.cli play --num-envs "$NUM_ENVS" "$@"
