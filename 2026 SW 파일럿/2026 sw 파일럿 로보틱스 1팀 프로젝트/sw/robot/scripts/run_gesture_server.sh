#!/usr/bin/env bash
# 젯슨/리눅스에서 gesture_engine(Flask)을 실행한다.
set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
python3 -m src.server.app
