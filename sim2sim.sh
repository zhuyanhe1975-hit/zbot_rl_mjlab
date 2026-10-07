#!/usr/bin/env bash
set -euo pipefail
source /home/yhzhu/AI/mjlab/.venv/bin/activate
# python viewer_compare.py \
#     --task Mjlab-Zbot-6dof-Walking \
#     --frequency 0.5 \

python viewer_compare.py \
    --task Mjlab-Zbot-6dof-Disturbed-Walking \
    --frequency 0.75 \
    --force-scale 1 \
    --torque-scale 1