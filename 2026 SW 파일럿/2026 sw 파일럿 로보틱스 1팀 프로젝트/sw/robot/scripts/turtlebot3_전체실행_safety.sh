#!/usr/bin/env bash
# 기존 전체 실행과 완전히 분리된 Safety Stop 포함 실행 파일.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT_DIR/scripts/run_navigation_gesture_joystick_safety.sh" \
    --patrol --without-sound "$@"
