#!/usr/bin/env bash
# 기본값은 안전한 드라이런이다. 실제 모터는 --enable-arm을 명시할 때만 열린다.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/../.venv/bin/python" "$SCRIPT_DIR/teleop_camera_arm.py" "$@"
