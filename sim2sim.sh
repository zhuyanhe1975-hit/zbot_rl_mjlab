#!/usr/bin/env bash
set -euo pipefail
source /home/yhzhu/AI/mjlab/.venv/bin/activate
# python viewer_compare.py \
#     --task Mjlab-Zbot-6dof-Walking \
#     --frequency 1.0 \

python viewer_compare.py \
    --task Mjlab-Zbot-6dof-Disturbed-Walking \
    --frequency 1.0 \
    # --force-scale 0 \
    # --torque-scale 0