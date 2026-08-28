#!/usr/bin/env bash
# 카메라·조이스틱 없이 자이로 장갑만으로 TurtleBot3를 수동 조종한다.
# 장갑은 mux에만 연결되므로 Nav2/다른 수동 입력과 하드웨어에 동시에 쓰지 않는다.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USB_PORT=""
GLOVE_PORT=""
GLOVE_BAUD="115200"
INVERT_PITCH=false
INVERT_ROLL=false
ROS_DISTRO_NAME="${ROS_DISTRO:-humble}"

usage() {
    cat <<'EOF'
Usage: bash scripts/run_glove_gyro_drive.sh --glove-port /dev/ttyUSB1 [options]

Options:
  --glove-port /dev/ttyUSB1  자이로 장갑 직렬/Bluetooth 포트 (필수)
  --glove-baud 115200        장갑 펌웨어 baud rate (기본: 115200)
  --usb-port /dev/ttyACM0    TurtleBot3 OpenCR 포트 (생략 시 하나일 때 자동 선택)
  --invert-pitch             앞/뒤 기울임 방향 반전
  --invert-roll              좌/우 기울임 방향 반전

장갑 포트와 TurtleBot3 OpenCR 포트는 반드시 달라야 한다.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --glove-port)
            [[ $# -ge 2 ]] || { echo "--glove-port에는 장치 경로가 필요합니다." >&2; exit 2; }
            GLOVE_PORT="$2"
            shift 2
            ;;
        --glove-baud)
            [[ $# -ge 2 ]] || { echo "--glove-baud에는 숫자가 필요합니다." >&2; exit 2; }
            GLOVE_BAUD="$2"
            shift 2
            ;;
        --usb-port)
            [[ $# -ge 2 ]] || { echo "--usb-port에는 장치 경로가 필요합니다." >&2; exit 2; }
            USB_PORT="$2"
            shift 2
            ;;
        --invert-pitch)
            INVERT_PITCH=true
            shift
            ;;
        --invert-roll)
            INVERT_ROLL=true
            shift
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

[[ -n "$GLOVE_PORT" ]] || { echo "안전을 위해 --glove-port을 반드시 지정하세요." >&2; usage >&2; exit 2; }
[[ -f "/opt/ros/${ROS_DISTRO_NAME}/setup.bash" ]] || {
    echo "ROS 2 ${ROS_DISTRO_NAME} 환경을 찾지 못했습니다." >&2; exit 1; }
[[ -e "$GLOVE_PORT" ]] || { echo "장갑 포트가 없습니다: $GLOVE_PORT" >&2; exit 1; }

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
[[ -e "$USB_PORT" ]] || { echo "OpenCR 포트가 없습니다: $USB_PORT" >&2; exit 1; }
[[ "$GLOVE_PORT" != "$USB_PORT" ]] || {
    echo "장갑 포트와 OpenCR 포트가 같습니다. OpenCR에 장갑 노드를 연결하지 않습니다." >&2; exit 2; }

set +u
source "/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
set -u
export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-burger}"
export LDS_MODEL="${LDS_MODEL:-LDS-03}"

if ! /usr/bin/python3 -c 'import serial' 2>/dev/null; then
    echo "pyserial이 없습니다. /usr/bin/python3 -m pip install pyserial 을 먼저 실행하세요." >&2
    exit 1
fi
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
echo "[1/3] TurtleBot3 bringup + mux 전용 하드웨어 입력 시작 (OpenCR: $USB_PORT)"
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

echo "[2/3] 속도 mux 시작"
/usr/bin/python3 ros2_bridge/cmd_vel_mux.py &
pids+=("$!")

echo "[3/3] 자이로 장갑 입력 시작 ($GLOVE_PORT, ${GLOVE_BAUD} baud)"
glove_args=(--port "$GLOVE_PORT" --baud "$GLOVE_BAUD")
[[ "$INVERT_PITCH" == true ]] && glove_args+=(--invert-pitch)
[[ "$INVERT_ROLL" == true ]] && glove_args+=(--invert-roll)
/usr/bin/python3 ros2_bridge/glove_gyro_teleop.py "${glove_args[@]}" &
pids+=("$!")

echo
echo "준비 완료: 처음 2초간 장갑을 편한 중립 자세로 유지하세요. 이후 앞/뒤 기울임=전진/후진, 오른쪽/왼쪽 기울임=우/좌회전입니다."
echo "입력이 0.35초 끊기면 즉시 정지합니다. 처음에는 바퀴를 들어 안전하게 확인하고, 종료는 Ctrl+C입니다."
wait "${pids[0]}"
