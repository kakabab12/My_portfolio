#!/usr/bin/env bash
# ESP32 Wi-Fi 자이로 장갑만으로 TurtleBot3를 수동 조종한다.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USB_PORT=""
UDP_PORT="5005"
INVERT_PITCH=false
INVERT_ROLL=false
ROS_DISTRO_NAME="${ROS_DISTRO:-humble}"

usage() {
    cat <<'EOF'
Usage: bash scripts/run_glove_wifi_drive.sh [options]

Options:
  --usb-port /dev/ttyACM0  TurtleBot3 OpenCR 포트
  --udp-port 5005          ESP32 장갑 UDP 포트 (기본: 5005)
  --invert-pitch           앞/뒤 기울임 방향 반전
  --invert-roll            좌/우 기울임 방향 반전

ESP32와 TurtleBot 컴퓨터는 반드시 같은 Wi-Fi에 연결되어야 한다.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --usb-port)
            [[ $# -ge 2 ]] || { echo "--usb-port에는 장치 경로가 필요합니다." >&2; exit 2; }
            USB_PORT="$2"; shift 2 ;;
        --udp-port)
            [[ $# -ge 2 ]] || { echo "--udp-port에는 포트 번호가 필요합니다." >&2; exit 2; }
            UDP_PORT="$2"; shift 2 ;;
        --invert-pitch) INVERT_PITCH=true; shift ;;
        --invert-roll) INVERT_ROLL=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "알 수 없는 옵션: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -f "/opt/ros/${ROS_DISTRO_NAME}/setup.bash" ]] || {
    echo "ROS 2 ${ROS_DISTRO_NAME} 환경을 찾지 못했습니다." >&2; exit 1; }
if [[ -z "$USB_PORT" ]]; then
    shopt -s nullglob
    acm_ports=(/dev/ttyACM*)
    shopt -u nullglob
    [[ ${#acm_ports[@]} -eq 1 ]] || {
        echo "OpenCR 포트를 자동 선택할 수 없습니다. --usb-port을 지정하세요." >&2; exit 1; }
    USB_PORT="${acm_ports[0]}"
fi
[[ -e "$USB_PORT" ]] || { echo "OpenCR 포트가 없습니다: $USB_PORT" >&2; exit 1; }

set +u
source "/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
set -u
export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-burger}"
export LDS_MODEL="${LDS_MODEL:-LDS-03}"

if ros2 node list 2>/dev/null | grep -qx '/turtlebot3_node'; then
    echo "이미 TurtleBot3 bringup이 실행 중입니다. 기존 bringup/브리지/mux를 종료하세요." >&2
    exit 1
fi

pids=()
cleanup() {
    local pid
    trap - EXIT INT TERM
    echo
    echo "정지 중..."
    for pid in "${pids[@]:-}"; do kill -INT "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]:-}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

cd "$ROOT_DIR"
echo "[1/3] TurtleBot3 bringup 시작 (OpenCR: $USB_PORT)"
ros2 launch "$ROOT_DIR/ros2_bridge/robot_with_mux.launch.py" "usb_port:=${USB_PORT}" &
pids+=("$!")

echo "TurtleBot3 컨트롤러 준비 대기..."
for _ in {1..50}; do
    ros2 node list 2>/dev/null | grep -qx '/turtlebot3_node' && break
    sleep 0.2
done
ros2 node list 2>/dev/null | grep -qx '/turtlebot3_node' || {
    echo "TurtleBot3 컨트롤러가 10초 안에 시작되지 않았습니다." >&2; exit 1; }

echo "[2/3] 속도 mux 시작"
/usr/bin/python3 ros2_bridge/cmd_vel_mux.py &
pids+=("$!")

echo "[3/3] Wi-Fi 장갑 UDP 수신 시작 (포트: $UDP_PORT)"
glove_args=(--port "$UDP_PORT")
[[ "$INVERT_PITCH" == true ]] && glove_args+=(--invert-pitch)
[[ "$INVERT_ROLL" == true ]] && glove_args+=(--invert-roll)
/usr/bin/python3 ros2_bridge/wifi_glove_teleop.py "${glove_args[@]}" &
pids+=("$!")

echo
echo "준비 완료: 장갑과 TurtleBot 컴퓨터가 같은 Wi-Fi여야 합니다. 처음 2초간 장갑을 중립 자세로 유지하세요."
echo "UDP 입력이 0.35초 끊기면 즉시 정지합니다. 처음에는 바퀴를 들어 확인하고 종료는 Ctrl+C입니다."
wait "${pids[0]}"
