#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname -- "$0")/scripts/python.sh" -m zbot_rl_mjlab.cli smoke "$@"
