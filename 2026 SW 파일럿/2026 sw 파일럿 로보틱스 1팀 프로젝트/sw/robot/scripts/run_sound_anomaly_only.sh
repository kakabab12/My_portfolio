#!/usr/bin/env bash
# C270 웹캠 마이크 소음 탐지와 OpenCR 상태 LED만 실행한다.
# Nav2, LiDAR, 제스처, 조이스틱, Wi-Fi 장갑, 순찰은 시작하지 않는다.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_DISTRO_NAME="${ROS_DISTRO:-humble}"
TURTLEBOT_WORKSPACE="${TURTLEBOT_WORKSPACE:-/home/user/turtlebot3_ws}"
SOUND_WORKSPACE="${SOUND_WORKSPACE:-/home/user/ros2_ws}"
USB_PORT=""
# C270 두 대 중 첫 번째 웹캠 마이크만 쓴다. P5U USB 마이크는 사용하지 않는다.
# PortAudio 장치 0은 현재 C270 첫 번째 입력이며, 필요하면 --audio-device로 바꿀 수 있다.
AUDIO_DEVICE="${SOUND_AUDIO_DEVICE:-0}"

usage() {
    cat <<'EOF'
Usage: bash scripts/run_sound_anomaly_only.sh [options]

Options:
  --usb-port /dev/ttyACM0  OpenCR serial port (default: auto-select)
  --audio-device 0         PortAudio input device (default: first C270 microphone)

Starts only the TurtleBot3 OpenCR connection required for the LEDs and the
first C270 webcam-microphone sound anomaly node.  It does not start Nav2, LiDAR,
gesture control, joysticks, Wi-Fi glove, or waypoint patrol.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --usb-port)
            [[ $# -ge 2 ]] || { echo "--usb-port에는 장치 경로가 필요합니다." >&2; exit 2; }
            USB_PORT="$2"
            shift 2
            ;;
        --audio-device)
            [[ $# -ge 2 ]] || { echo "--audio-device에는 PortAudio 장치 번호가 필요합니다." >&2; exit 2; }
            AUDIO_DEVICE="$2"
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
if [[ ! -f "${TURTLEBOT_WORKSPACE}/install/setup.bash" ]]; then
    echo "TurtleBot3 workspace를 찾지 못했습니다: ${TURTLEBOT_WORKSPACE}" >&2
    exit 1
fi
if [[ ! -f "${SOUND_WORKSPACE}/install/setup.bash" ]]; then
    echo "소리 이상감지 ROS workspace를 찾지 못했습니다: ${SOUND_WORKSPACE}" >&2
    exit 1
fi
if [[ ! -f "${SOUND_WORKSPACE}/src/dyeun_robotics/sound_anomaly/config/sound_anomaly.yaml" ]]; then
    echo "sound_anomaly 설정 파일을 찾지 못했습니다." >&2
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

# ROS setup은 선언되지 않은 변수를 참조할 수 있으므로 source할 때만 nounset을 끈다.
set +u
source "/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
source "${TURTLEBOT_WORKSPACE}/install/setup.bash"
source "${SOUND_WORKSPACE}/install/setup.bash"
set -u
export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-burger}"

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
        if ros2 node list 2>/dev/null | grep -qx "$node_name"; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

wait_for_ros_topic_message() {
    local topic_name="$1"
    local timeout_seconds="$2"
    local deadline=$((SECONDS + timeout_seconds))

    while (( SECONDS < deadline )); do
        if timeout 3s ros2 topic echo --once "$topic_name" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.2
    done
    return 1
}

has_turtlebot_process_for_port() {
    local port="$1"
    # DDS discovery가 늦거나 일시적으로 실패해도 같은 OpenCR 포트를 두 번 열면
    # 안 된다. 실제 turtlebot3_ros 프로세스가 포트를 사용 중인지 먼저 확인한다.
    pgrep -f "turtlebot3_ros.*-i[[:space:]]+${port}" >/dev/null 2>&1
}

has_full_launcher_process() {
    # 바탕화면 전체실행은 이 보조 스크립트를 exec한다. 사용자가 두 아이콘을
    # 연달아 눌러도 OpenCR를 이중으로 열지 않도록 전체실행의 하드웨어 준비를 기다린다.
    pgrep -f "scripts/run_navigation_gesture_joystick\.sh" >/dev/null 2>&1
}

hardware_pid=""
sound_pid=""
started_hardware=false
cleanup() {
    trap - EXIT INT TERM
    echo
    echo "소음 탐지 정지 중..."
    # sound_anomaly가 LED 끄기 명령을 먼저 OpenCR로 전달할 수 있게 한다.
    if [[ -n "$sound_pid" ]]; then
        # ros2 launch 부모가 먼저 끝나도 sound_anomaly_node가 고아로 남지 않도록
        # launch 전용 세션 전체에 TERM을 보낸다. 노드는 SIGTERM에서 LED를 끈다.
        kill -TERM -- "-$sound_pid" 2>/dev/null || terminate_process_tree "$sound_pid"
        wait "$sound_pid" 2>/dev/null || true
    fi
    if [[ "$started_hardware" == true && -n "$hardware_pid" ]]; then
        terminate_process_tree "$hardware_pid"
        wait "$hardware_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if ros2 node list 2>/dev/null | grep -qx '/sound_anomaly_node'; then
    echo "이미 소리 이상감지 노드가 실행 중입니다. 기존 노드를 종료한 뒤 다시 실행하세요." >&2
    exit 1
fi

use_existing_bringup=false
if has_turtlebot_process_for_port "$USB_PORT" || \
   ros2 node list 2>/dev/null | grep -qx '/turtlebot3_node'; then
    use_existing_bringup=true
elif has_full_launcher_process; then
    echo "전체실행의 OpenCR 준비를 기다립니다..."
    for _ in {1..60}; do
        if has_turtlebot_process_for_port "$USB_PORT" || \
           ros2 node list 2>/dev/null | grep -qx '/turtlebot3_node'; then
            use_existing_bringup=true
            break
        fi
        # 전체실행이 중간에 끝났다면 단독 실행이 자체 OpenCR를 시작할 수 있다.
        has_full_launcher_process || break
        sleep 0.5
    done
    if [[ "$use_existing_bringup" != true ]] && has_full_launcher_process; then
        echo "전체실행이 30초 안에 OpenCR를 준비하지 못했습니다. 전체실행 로그를 확인하세요." >&2
        exit 1
    fi
fi

if [[ "$use_existing_bringup" == true ]]; then
    echo "기존 TurtleBot3 bringup을 사용합니다. 소리 탐지만 추가합니다."
else
    echo "[1/2] OpenCR 연결 시작 (LED 제어 전용, 주행 입력 없음: ${USB_PORT})"
    ros2 launch "${ROOT_DIR}/ros2_bridge/robot_with_mux.launch.py" "usb_port:=${USB_PORT}" &
    hardware_pid="$!"
    started_hardware=true

    echo "TurtleBot3 컨트롤러 준비 대기..."
    if ! wait_for_ros_node /turtlebot3_node 15; then
        echo "TurtleBot3 컨트롤러가 15초 안에 시작되지 않았습니다." >&2
        exit 1
    fi
    echo "OpenCR /battery_state 수신 대기..."
    if ! wait_for_ros_topic_message /battery_state 30; then
        echo "OpenCR에서 /battery_state를 30초 안에 받지 못했습니다." >&2
        exit 1
    fi
fi

echo "[2/2] C270 웹캠 마이크 소리 이상감지 시작 (PortAudio 장치: ${AUDIO_DEVICE})"
# 별도 세션으로 실행해 종료 시 launch와 노드를 한 그룹으로 정리한다.
SOUND_AUDIO_DEVICE="$AUDIO_DEVICE" setsid ros2 launch "${ROOT_DIR}/ros2_bridge/sound_anomaly_with_led.launch.py" &
sound_pid="$!"

echo "소리 이상감지 노드 준비 대기..."
if ! wait_for_ros_node /sound_anomaly_node 45; then
    echo "소리 이상감지 노드가 45초 안에 시작되지 않았습니다. C270 마이크 연결과 장치 번호를 확인하세요." >&2
    exit 1
fi

# state와 LED heartbeat는 노드가 이미 정상 동작해도 ROS CLI discovery 시점에 따라
# 놓칠 수 있다. 노드 목록도 DDS discovery 지연으로 재확인 때 사라진 것처럼 보일 수
# 있다. 이 메시지/재조회 수신을 준비 조건으로 삼으면 실제로 판정 중인 C270 노드를
# 종료하는 false negative가 생긴다. 위의 첫 노드 등록 확인을 통과한 뒤에는 스트림에
# 짧은 초기화 시간을 주고 launch가 계속 감시하도록 둔다.
echo "C270 오디오 스트림 안정화 대기..."
sleep 3

echo
echo "준비 완료: C270 마이크 소음 탐지와 OpenCR LED만 실행 중입니다. 종료는 Ctrl+C입니다."
echo "상태: ros2 topic echo /sound_anomaly_node/state"
echo "확률: ros2 topic echo /sound_anomaly_node/anomaly_probability"

# OpenCR 또는 소리 노드 중 하나가 멈추면 모두 정리해 LED와 상태가 어긋나지 않게 한다.
if [[ "$started_hardware" == true ]]; then
    wait -n "$hardware_pid" "$sound_pid"
else
    wait "$sound_pid"
fi
