#!/usr/bin/env bash
# 바탕화면 시연용 실행 파일. Safety 없이 Nav2 무한 순찰만 실행한다.
set -euo pipefail

exec /bin/bash /home/user/sw/robot/scripts/run_demo_patrol_safety.sh \
    --map /home/user/sw/robot/maps/demo_presentation.yaml "$@"
