#!/usr/bin/env bash
# 젯슨 오린 나노 설치 스크립트 — 설치순서.md의 [설치] 파트를 자동화한다.
#
# 실행(젯슨에서):
#   bash scripts/install_jetson.sh
#
# 여러 번 실행해도 안전하다(idempotent) — 이미 설치·설정된 항목은 건너뛴다.
# 한 줄씩 손으로 옮겨 치다 줄이 꼬이는 실수(붙여넣기 중 bashrc 줄 깨짐 등)를
# 없애는 게 목적이다 — 각 단계가 실패해도 어느 단계인지 바로 보이도록
# set -e + 단계별 안내를 쓴다.
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

step() { echo; echo "==> $1"; }
ok()   { echo "    OK: $1"; }
warn() { echo "    [경고] $1"; }

step "1/6 손 인식 라이브러리 설치 (requirements-gesture.txt)"
if ! pip3 install -r requirements-gesture.txt; then
    warn "고정 버전 설치 실패 — mediapipe/opencv-contrib-python을 버전 없이 재시도"
    pip3 install mediapipe opencv-contrib-python
fi

step "2/6 손 인식 모델 다운로드"
python3 scripts/download_weights.py

step "3/6 ROS2 브리지용 라이브러리 설치 (requirements-ros2.txt)"
pip3 install -r requirements-ros2.txt

step "4/6 TURTLEBOT3_MODEL 환경변수 설정 (버거)"
if grep -q "TURTLEBOT3_MODEL" ~/.bashrc 2>/dev/null; then
    ok "이미 설정돼 있음 — 건너뜀"
else
    echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc
    ok "~/.bashrc에 추가함"
fi

step "5/6 OpenCR 연결 확인"
if ls /dev/ttyACM* >/dev/null 2>&1; then
    ok "찾음: $(ls /dev/ttyACM* | tr '\n' ' ')"
else
    warn "/dev/ttyACM*가 안 보입니다 — 로봇(OpenCR) 배터리가 켜져 있는지 확인하세요"
fi

step "6/6 ROS2 Humble + 터틀봇3 패키지"
if command -v ros2 >/dev/null 2>&1; then
    ok "ROS2가 이미 설치돼 있습니다 — 이 단계를 건너뜁니다"
else
    echo "    ROS2가 안 보입니다. 설치하면 몇 분 걸립니다."
    read -r -p "    지금 ROS2 Humble + 터틀봇3 패키지를 설치할까요? [y/N] " REPLY
    if [[ "$REPLY" =~ ^[Yy]$ ]]; then
        echo "    -- ROS2 Humble 설치 --"
        sudo apt update && sudo apt install -y locales
        sudo locale-gen en_US en_US.UTF-8
        sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
        sudo apt install -y software-properties-common
        sudo add-apt-repository -y universe
        sudo apt update && sudo apt install -y curl

        sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
            | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

        if [ ! -f /etc/apt/sources.list.d/ros2.list ]; then
            echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" \
                | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
            ok "/etc/apt/sources.list.d/ros2.list 생성"
        else
            ok "ros2.list 이미 있음 — 건너뜀"
        fi

        sudo apt update && sudo apt upgrade -y
        sudo apt install -y ros-humble-ros-base python3-colcon-common-extensions python3-rosdep

        sudo rosdep init 2>/dev/null || true   # 이미 init됐으면 에러 -> 무시하고 계속
        rosdep update

        if grep -q "source /opt/ros/humble/setup.bash" ~/.bashrc 2>/dev/null; then
            ok "~/.bashrc에 이미 등록됨 — 건너뜀"
        else
            echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
            ok "~/.bashrc에 추가함"
        fi
        # shellcheck disable=SC1091
        source /opt/ros/humble/setup.bash

        echo "    -- 터틀봇3 ROS2 패키지 설치 --"
        if ! sudo apt install -y ros-humble-turtlebot3-msgs ros-humble-turtlebot3 \
            ros-humble-dynamixel-sdk ros-humble-hls-lfcd-lds-driver; then
            warn "apt 설치 실패 — README.md의 '소스로 빌드' 절차를 참고하세요"
        fi
    else
        echo "    건너뜀 — README.md '3. ROS2 + 터틀봇3 패키지 설치'를 나중에 직접 진행하세요."
    fi
fi

echo
echo "=================================================================="
echo " 설치 스크립트 끝."
echo " 남은 할 일: 젯슨에서 ls /dev/video*로 카메라 장치 번호 확인"
echo '   camera:'
echo '     devices: [0, 1, 2]'
echo " 그다음 검증: python3 -m unittest discover tests -v"
echo " 실행 순서는 설치순서.md의 [매번 실행] 참고."
echo "=================================================================="
