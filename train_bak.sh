#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_ID="Mjlab-Zbot-6dof-Periodic-Stepping"
LOG_ROOT="${ZBOT_LOG_ROOT:-$PROJECT_DIR/logs/rsl_rl}"
if [[ "${1:-}" == "--task" ]]; then
  case "${2:-}" in
    stepping) TASK_ID="Mjlab-Zbot-6dof-Periodic-Stepping" ;;
    walking) TASK_ID="Mjlab-Zbot-6dof-Walking" ;;
    "Mjlab-Zbot-6dof-Periodic-Stepping"|"Mjlab-Zbot-6dof-Walking") TASK_ID="$2" ;;
    *) echo "Unknown task '$2' (choose stepping or walking)" >&2; exit 2 ;;
  esac
  shift 2
fi
exec "$PROJECT_DIR/scripts/python.sh" -c 'import zbot_rl_mjlab; from mjlab.scripts.train import main; main()' "$TASK_ID" --log-root "$LOG_ROOT" "$@"
