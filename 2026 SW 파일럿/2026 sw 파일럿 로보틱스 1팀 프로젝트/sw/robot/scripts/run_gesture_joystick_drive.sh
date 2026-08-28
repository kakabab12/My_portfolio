#!/usr/bin/env bash
# SLAM/Nav2 없이 제스처와 조이스틱만 TurtleBot3를 조종한다.
# mux가 둘 중 하나의 속도만 /cmd_vel_muxed로 전달한다.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USB_PORT=""
ROS_DISTRO_NAME="${ROS_DISTRO:-humble}"
GESTURE_PYTHON="${GESTURE_PYTHON:-${ROOT_DIR}/.venv/bin/python}"

usage() {
    cat <<'EOF'
Usage: bash scripts/run_gesture_joystick_drive.sh [--usb-port /dev/ttyACM0]

Runs TurtleBot3 bringup with cmd_vel mux, gesture recognition/controller,
and the joystick. It intentionally does not start SLAM or Nav2.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --usb-port)
            [[ $# -ge 2 ]] || { echo "--usb-port에는 장치 경로가 필요합니다." >&2; exit 2; }
            USB_PORT="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "알 수 없는 옵션: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -f "/opt/ros/${ROS_DISTRO_NAME}/setup.bash" ]]; then
    echo "ROS 2 ${ROS_DISTRO_NAME} 환경을 찾지 못했습니다." >&2
    exit 1
fi
if [[ ! -x "$GESTURE_PYTHON" ]]; then
    echo "제스처 Python을 찾지 못했습니다: $GESTURE_PYTHON" >&2
    echo "필요하면 GESTURE_PYTHON=/경로/python 을 지정하세요." >&2
    exit 1
fi
if [[ ! -e /dev/input/js0 ]]; then
    echo "조이스틱 장치를 찾지 못했습니다: /dev/input/js0" >&2
    exit 1
fi
if ! command -v curl >/dev/null; then
    echo "curl을 찾지 못했습니다. 제스처 서버 준비 상태를 확인할 수 없습니다." >&2
    exit 1
fi

if [[ -z "$USB_PORT" ]]; then
    shopt -s nullglob
    acm_ports=(/dev/ttyACM*)
    shopt -u nullglob
    if [[ ${#acm_ports[@]} -ne 1 ]]; then
        echo "OpenCR 포트를 자동 선택할 수 없습니다. --usb-port /dev/ttyACM0 처럼 지정하세요." >&2
        exit 1
    fi
    USB_PORT="${acm_ports[0]}"
fi
if [[ ! -e "$USB_PORT" ]]; then
    echo "OpenCR 포트가 없습니다: $USB_PORT" >&2
    exit 1
fi

# ROS Humble setup은 선언되지 않은 환경변수를 참조할 수 있어 source하는 동안만
# nounset(set -u)을 해제한다.
set +u
source "/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
set -u
export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-burger}"
# 바탕화면(.desktop) 실행은 ~/.bashrc의 환경변수를 상속하지 않을 수 있다.
# 이 로봇에 장착된 COIN-D4(M1CT_TOF)는 LDS-03이므로 안전한 기본값을 명시한다.
export LDS_MODEL="${LDS_MODEL:-LDS-03}"

if ros2 node list 2>/dev/null | grep -qx '/turtlebot3_node'; then
    echo "이미 TurtleBot3 bringup이 실행 중입니다. 기존 bringup/브리지/mux를 종료한 뒤 다시 실행하세요." >&2
    exit 1
fi

pids=()
cleanup() {
    local pid
    trap - EXIT INT TERM
    echo
    echo "정지 중..."
    for pid in "${pids[@]:-}"; do
        kill -INT "$pid" 2>/dev/null || true
    done
    for pid in "${pids[@]:-}"; do
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

cd "$ROOT_DIR"
echo "[1/5] TurtleBot3 bringup + mux 전용 하드웨어 입력 시작 (OpenCR: $USB_PORT)"
ros2 launch "$ROOT_DIR/ros2_bridge/robot_with_mux.launch.py" "usb_port:=${USB_PORT}" &
pids+=("$!")

echo "TurtleBot3 컨트롤러 준비 대기..."
for _ in {1..50}; do
    if ros2 node list 2>/dev/null | grep -qx '/turtlebot3_node'; then
        break
    fi
    sleep 0.2
done
if ! ros2 node list 2>/dev/null | grep -qx '/turtlebot3_node'; then
    echo "TurtleBot3 컨트롤러가 10초 안에 시작되지 않았습니다." >&2
    exit 1
fi

echo "[2/5] 제스처 서버 시작"
GESTURE_INITIAL_CONTROL_MODE=gesture "$GESTURE_PYTHON" -m src.server.app &
pids+=("$!")

echo "제스처 서버 준비 대기..."
for _ in {1..50}; do
    if curl --fail --silent --output /dev/null http://127.0.0.1:5000/health; then
        break
    fi
    sleep 0.2
done
if ! curl --fail --silent --output /dev/null http://127.0.0.1:5000/health; then
    echo "제스처 서버가 10초 안에 준비되지 않았습니다." >&2
    exit 1
fi

echo "[3/5] 제스처·조이스틱 속도 mux 시작"
/usr/bin/python3 ros2_bridge/cmd_vel_mux.py &
pids+=("$!")

echo "[4/5] 조이스틱 입력 시작 (/dev/input/js0)"
ros2 launch "$ROOT_DIR/ros2_bridge/joystick_with_mux.launch.py" &
pids+=("$!")

echo "[5/5] 제스처 컨트롤러 시작 (/cmd_vel_gesture)"
/usr/bin/python3 ros2_bridge/cmd_vel_bridge.py \
    --base-url http://127.0.0.1:5000 \
    --disable-navigation \
    --start-gesture-enabled &
pids+=("$!")

echo
echo "준비 완료: Nav2/SLAM은 실행하지 않았습니다. 제스처 모드는 시작부터 ON이며, mux가 제스처 또는 조이스틱 하나만 선택합니다."
echo "제스처 ON 상태에서 따봉 1.5초 유지=모드 OFF, 모드 OFF 상태의 짧은 따봉=제스처 ON, OK 사인=조이스틱 ON."
echo "조이스틱 ON 뒤에는 OK 사인을 한 번 풀고 다시 1.5초 유지하면 컨트롤러·제스처 OFF 후 AUTO/Nav2로 복귀합니다."
echo "종료는 Ctrl+C입니다."
wait "${pids[0]}"
