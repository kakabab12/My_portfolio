#!/usr/bin/env bash
# Linux에서 .exe처럼 쓰는 전체 실행 파일.
# 소음 탐지는 별도 실행기로 분리했다. 필요하면 호출할 때 --with-sound를 넘긴다.
# OpenCR 포트가 하나면 자동 선택하고, 필요하면 --usb-port /dev/ttyACM1처럼 넘긴다.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT_DIR/scripts/run_navigation_gesture_joystick.sh" --patrol --without-sound "$@"
