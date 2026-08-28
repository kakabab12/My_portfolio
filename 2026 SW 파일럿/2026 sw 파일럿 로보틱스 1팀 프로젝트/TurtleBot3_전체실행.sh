#!/usr/bin/env bash
# 바탕화면용 TurtleBot3 전체 실행기.
# 실제 설정과 제스처 코드는 프로젝트의 최신 파일을 항상 사용한다.
set -euo pipefail

export LDS_MODEL="${LDS_MODEL:-LDS-03}"
# 전체실행에서는 제스처·Nav2·LiDAR와 함께 현재 sound_anomaly 패키지도
# 기본으로 실행한다. 필요하면 호출 끝에 --without-sound를 넘겨 일시적으로 끌 수 있다.
exec /bin/bash /home/user/sw/robot/scripts/turtlebot3_전체실행.sh --with-sound "$@"
