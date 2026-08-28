#!/usr/bin/env bash
# 시연 전용: TurtleBot3 + Nav2/AMCL/RViz + A-B-C-D 무한 자율순찰.
# Safety supervisor·Safety mux·Safety 미션은 실행하지 않는다.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USB_PORT=""
MAP_PATH="${NAV_MAP:-/home/user/sw/robot/maps/factory_map_final.yaml}"
ROS_DISTRO_NAME="${ROS_DISTRO:-humble}"

usage() {
    cat <<'EOF'
사용법: bash scripts/run_demo_patrol_safety.sh [옵션]

옵션:
  --usb-port /dev/ttyACM0   OpenCR 시리얼 포트
  --map /절대경로/map.yaml  Nav2 지도 YAML (기본: factory_map_final.yaml)
  -h, --help                도움말

로봇을 A 위치/방향에 놓고 실행하세요. A->B->C->D->A를 Nav2로 무한 반복합니다.
Safety supervisor·Safety mux·Safety 복구 동작은 실행하지 않습니다.
종료는 Ctrl+C입니다.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --usb-port)
            [[ $# -ge 2 ]] || {
                echo "--usb-port에는 장치 경로가 필요합니다." >&2
                exit 2
            }
            USB_PORT="$2"
            shift 2
            ;;
        --map)
            [[ $# -ge 2 ]] || {
                echo "--map에는 지도 YAML 경로가 필요합니다." >&2
                exit 2
            }
            MAP_PATH="$2"
            shift 2
            ;;
        -h|--help)
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
if [[ ! -f /home/user/turtlebot3_ws/install/setup.bash ]]; then
    echo "TurtleBot3 workspace 설치 환경을 찾지 못했습니다." >&2
    exit 1
fi
if [[ ! -f "$MAP_PATH" ]]; then
    echo "지도 YAML을 찾지 못했습니다: $MAP_PATH" >&2
    exit 1
fi
for required_file in \
    "$ROOT_DIR/ros2_bridge/robot_with_mux.launch.py" \
    "$ROOT_DIR/ros2_bridge/navigation_with_mux.launch.py" \
    "$ROOT_DIR/ros2_bridge/cmd_vel_mux.py" \
    "$ROOT_DIR/ros2_bridge/waypoint_patrol_demo.py"; do
    if [[ ! -f "$required_file" ]]; then
        echo "시연용 구성 파일을 찾지 못했습니다: $required_file" >&2
        exit 1
    fi
done

if [[ -z "$USB_PORT" ]]; then
    shopt -s nullglob
    acm_ports=(/dev/ttyACM*)
    shopt -u nullglob
    if [[ ${#acm_ports[@]} -ne 1 ]]; then
        echo "OpenCR 포트를 자동 선택할 수 없습니다. --usb-port /dev/ttyACM0처럼 지정하세요." >&2
        exit 1
    fi
    USB_PORT="${acm_ports[0]}"
fi
if [[ ! -e "$USB_PORT" ]]; then
    echo "OpenCR 포트가 없습니다: $USB_PORT" >&2
    exit 1
fi

set +u
source "/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
source /home/user/turtlebot3_ws/install/setup.bash
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-30}"
export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-burger}"
export LDS_MODEL="${LDS_MODEL:-LDS-03}"

if pgrep -f "turtlebot3_ros.*-i[[:space:]]+${USB_PORT}" >/dev/null 2>&1; then
    echo "이미 TurtleBot3 관련 실행이 동작 중입니다. 기존 실행을 종료한 뒤 다시 시작하세요." >&2
    exit 1
fi

pids=()
terminate_process_tree() {
    local parent_pid="$1"
    local child_pid
    while IFS= read -r child_pid; do
        [[ -n "$child_pid" ]] && terminate_process_tree "$child_pid"
    done < <(pgrep -P "$parent_pid" 2>/dev/null || true)
    kill -TERM "$parent_pid" 2>/dev/null || true
}

wait_for_ros_node() {
    local node_name="$1"
    local timeout_seconds="$2"
    local deadline=$((SECONDS + timeout_seconds))
    while (( SECONDS < deadline )); do
        if timeout 2s ros2 node list 2>/dev/null | grep -qx "$node_name"; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

wait_for_ros_topic_message() {
    local topic_name="$1"
    local timeout_seconds="$2"
    local qos_reliability="${3:-}"
    local deadline=$((SECONDS + timeout_seconds))
    local remaining_seconds
    local attempt_timeout
    local -a echo_args=(ros2 topic echo --once "$topic_name")
    if [[ -n "$qos_reliability" ]]; then
        echo_args+=(--qos-reliability "$qos_reliability")
    fi
    while (( SECONDS < deadline )); do
        remaining_seconds=$((deadline - SECONDS))
        attempt_timeout=$((remaining_seconds < 3 ? remaining_seconds : 3))
        if timeout "${attempt_timeout}s" "${echo_args[@]}" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.2
    done
    return 1
}

cleanup() {
    local pid
    trap - EXIT INT TERM
    echo
    echo "시연용 순찰 정지 중..."
    for pid in "${pids[@]:-}"; do
        terminate_process_tree "$pid"
    done
    for pid in "${pids[@]:-}"; do
        wait "$pid" 2>/dev/null || true
    done
    echo "시연용 관련 프로세스를 종료했습니다."
}
trap cleanup EXIT INT TERM

cd "$ROOT_DIR"
echo "[1/4] TurtleBot3 하드웨어 bringup 시작 (OpenCR: $USB_PORT, LiDAR: $LDS_MODEL)"
ros2 launch "$ROOT_DIR/ros2_bridge/robot_with_mux.launch.py" \
    "usb_port:=${USB_PORT}" &
pids+=("$!")

echo "TurtleBot3 컨트롤러 준비 대기..."
if ! wait_for_ros_node /turtlebot3_node 15; then
    echo "TurtleBot3 컨트롤러가 15초 안에 시작되지 않았습니다." >&2
    exit 1
fi
echo "OpenCR /battery_state 수신 대기..."
if ! wait_for_ros_topic_message /battery_state 30; then
    echo "OpenCR에서 /battery_state를 받지 못했습니다. USB·전원·케이블을 확인하세요." >&2
    exit 1
fi
echo "LDS-03 /scan 수신 대기..."
if ! wait_for_ros_node /lidar_node 30 || \
   ! wait_for_ros_topic_message /scan 10 best_effort; then
    echo "LDS-03 /scan을 받지 못했습니다. LiDAR 연결·전원·LDS_MODEL을 확인하세요." >&2
    exit 1
fi

echo "[2/4] Nav2 전용 속도 mux 시작"
/usr/bin/python3 ros2_bridge/cmd_vel_mux.py &
pids+=("$!")

echo "[3/4] 저장 지도 Nav2/AMCL + RViz 시작 ($MAP_PATH)"
ros2 launch "$ROOT_DIR/ros2_bridge/navigation_with_mux.launch.py" \
    "map:=${MAP_PATH}" &
pids+=("$!")

echo "[4/4] A->B->C->D->A 무한 Nav2 순찰 시작"
/usr/bin/python3 ros2_bridge/waypoint_patrol_demo.py &
pids+=("$!")

echo
echo "시연용 실행 완료: Nav2/AMCL, RViz, 기본 속도 mux, 무한 waypoint 순찰만 동작합니다."
echo "Safety supervisor·Safety mux·Safety 복구/후진 로직은 실행하지 않습니다."
echo "로봇은 A->B->C->D->A 순서로 계속 반복합니다. 종료는 Ctrl+C입니다."

set +e
wait -n "${pids[@]}"
exit_status=$?
set -e
echo "핵심 프로세스 하나가 종료되어 시연용 전체 실행을 정리합니다." >&2
exit "$exit_status"
