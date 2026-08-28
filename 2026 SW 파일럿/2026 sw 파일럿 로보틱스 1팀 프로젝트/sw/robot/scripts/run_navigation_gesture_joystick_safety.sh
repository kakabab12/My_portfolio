#!/usr/bin/env bash
# 기존 전체 실행과 분리된 Safety Stop 포함 A-B-C-D-A 실행기.
# A->B/D->A의 Nav2 구간에서만 갑작스러운 장애물 자동복구를 허용한다.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USB_PORT=""
MAP_PATH="${NAV_MAP:-/home/user/sw/robot/maps/factory_map_final.yaml}"
ROS_DISTRO_NAME="${ROS_DISTRO:-humble}"
GESTURE_PYTHON="${GESTURE_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
SOUND_WORKSPACE="${SOUND_WORKSPACE:-/home/user/ros2_ws}"
START_PATROL=false
ENABLE_SOUND=true
JOYSTICK_AVAILABLE=true

usage() {
    cat <<'EOF'
Usage: bash scripts/run_navigation_gesture_joystick_safety.sh [options]

Options:
  --usb-port /dev/ttyACM0   OpenCR serial port
  --map /absolute/map.yaml  Nav2 map yaml (default: factory_map_final.yaml)
  --patrol                  Start A->B(Nav2)->C(gesture)->D(joystick)->A(Nav2)
  --with-sound              Start C270 webcam-microphone sound detection (default)
  --without-sound           Skip sound detection and OpenCR sound-status LEDs

Runs hardware, C270 webcam-microphone sound anomaly detection, Nav2/AMCL,
gesture recognition, joystick, Wi-Fi glove, Safety supervisor, and the
Safety-priority velocity mux. P5U is not used by this launcher.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --usb-port)
            [[ $# -ge 2 ]] || { echo "--usb-port에는 장치 경로가 필요합니다." >&2; exit 2; }
            USB_PORT="$2"
            shift 2
            ;;
        --map)
            [[ $# -ge 2 ]] || { echo "--map에는 지도 yaml 경로가 필요합니다." >&2; exit 2; }
            MAP_PATH="$2"
            shift 2
            ;;
        --patrol)
            START_PATROL=true
            shift
            ;;
        --with-sound)
            ENABLE_SOUND=true
            shift
            ;;
        --without-sound)
            ENABLE_SOUND=false
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

if [[ ! -f "/opt/ros/${ROS_DISTRO_NAME}/setup.bash" ]]; then
    echo "ROS 2 ${ROS_DISTRO_NAME} 환경을 찾지 못했습니다." >&2
    exit 1
fi
if [[ "$ENABLE_SOUND" == true && ! -f "$SOUND_WORKSPACE/install/setup.bash" ]]; then
    echo "소리 이상감지 ROS workspace를 찾지 못했습니다: $SOUND_WORKSPACE" >&2
    exit 1
fi
if [[ "$ENABLE_SOUND" == true && ! -f "$SOUND_WORKSPACE/src/dyeun_robotics/sound_anomaly/config/sound_anomaly.yaml" ]]; then
    echo "sound_anomaly 설정 파일을 찾지 못했습니다." >&2
    exit 1
fi
if [[ ! -x "$GESTURE_PYTHON" ]]; then
    echo "제스처 Python을 찾지 못했습니다: $GESTURE_PYTHON" >&2
    exit 1
fi
if [[ ! -f "$MAP_PATH" ]]; then
    echo "지도 yaml을 찾지 못했습니다: $MAP_PATH" >&2
    exit 1
fi
if [[ ! -e /dev/input/js0 ]]; then
    # 조이스틱이 없더라도 Nav2·제스처·Wi-Fi 장갑·이상음 감지는 독립적으로
    # 시험할 수 있다. joy_node만 생략하고, 연결 후 다시 실행하면 조이스틱도
    # 같은 mux 입력으로 자동 포함된다.
    JOYSTICK_AVAILABLE=false
    echo "조이스틱 장치(/dev/input/js0)가 없어 조이스틱 입력만 생략합니다." >&2
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
source /home/user/turtlebot3_ws/install/setup.bash
if [[ "$ENABLE_SOUND" == true ]]; then
    source "$SOUND_WORKSPACE/install/setup.bash"
fi
set -u
export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-burger}"
# 바탕화면(.desktop) 실행은 ~/.bashrc의 환경변수를 상속하지 않을 수 있다.
# 이 로봇에 장착된 COIN-D4(M1CT_TOF)는 LDS-03이므로 안전한 기본값을 명시한다.
export LDS_MODEL="${LDS_MODEL:-LDS-03}"

if pgrep -f "turtlebot3_ros.*-i[[:space:]]+${USB_PORT}" >/dev/null 2>&1 || \
   ros2 node list 2>/dev/null | grep -qx '/turtlebot3_node'; then
    echo "이미 TurtleBot3 bringup이 실행 중입니다. 기존 bringup/브리지/mux/Nav2를 종료한 뒤 다시 실행하세요." >&2
    exit 1
fi

pids=()
sound_pid=""
terminate_process_tree() {
    # ros2 launch는 lidar/RViz/Nav2처럼 자식 프로세스를 만든다. launch 부모만
    # 끝내면 자식이 고아로 남아 다음 실행의 /turtlebot3_node 중복을 일으킨다.
    # 자식부터 TERM을 보내 모두 정상 종료한 뒤 부모를 종료한다.
    local parent_pid="$1"
    local child_pid
    while IFS= read -r child_pid; do
        [[ -n "$child_pid" ]] && terminate_process_tree "$child_pid"
    done < <(pgrep -P "$parent_pid" 2>/dev/null || true)
    kill -TERM "$parent_pid" 2>/dev/null || true
}

wait_for_ros_node() {
    # 이 실행 환경에서는 ROS CLI의 topic discovery가 실제 DDS 노드보다 늦게
    # 도착할 수 있다. 노드가 올라오면 SoundAnomalyNode는 이미 웹캠 InputStream을
    # 열었고 LiDAR 노드도 포트를 연 상태이므로, 노드 발견을 준비 기준으로 쓴다.
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
    # 노드 등록만으로는 장치가 실제로 데이터를 내보낸다는 보장이 없다. 예를 들어
    # USB 허브가 실행 중 분리되면 lidar_node는 살아 있어도 /scan은 더 이상
    # 발행되지 않는다. Nav2는 실제 센서 메시지를 받은 뒤에만 시작한다.
    #
    # `ros2 topic echo --once`는 publisher가 아직 없으면 기다리지 않고 즉시
    # 실패한다. 따라서 timeout 하나로 감싸면 TurtleBot3의 자이로 초기화처럼
    # publisher가 조금 늦게 만들어지는 정상 상황도 실패로 오판한다. 제한시간
    # 안에서 짧게 재시도해 실제 메시지를 기다린다.
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
    echo "정지 중..."
    # 소음 노드를 먼저 끝내 OpenCR LED 종료 명령이 전달되도록 한다.
    if [[ -n "$sound_pid" ]]; then
        terminate_process_tree "$sound_pid"
        wait "$sound_pid" 2>/dev/null || true
    fi
    for pid in "${pids[@]:-}"; do
        terminate_process_tree "$pid"
    done
    for pid in "${pids[@]:-}"; do
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

cd "$ROOT_DIR"
echo "[1/9] TurtleBot3 bringup + mux 전용 하드웨어 입력 시작 (OpenCR: $USB_PORT)"
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

echo "OpenCR /battery_state 수신 대기 (최대 30초; 자이로 캘리브레이션·DDS 초기 발견 포함)..."
# TurtleBot3 노드는 먼저 ROS graph에 등록된 뒤 자이로 캘리브레이션(약 5초)을
# 마치고 battery_state publisher를 만든다. 재부팅 직후에는 DDS CLI discovery도
# 추가로 걸릴 수 있으므로, 기존 10초 제한으로 정상 OpenCR을 오판하지 않게 둔다.
if ! wait_for_ros_topic_message /battery_state 30; then
    echo "OpenCR에서 /battery_state를 30초 안에 받지 못했습니다. OpenCR USB·전원·케이블을 확인하세요." >&2
    exit 1
fi

echo "LDS-03 /scan 수신 대기..."
# LDS-03 드라이버는 포트를 연 뒤 `/lidar_node`로 등록된다.
if ! wait_for_ros_node /lidar_node 30; then
    echo "LDS-03 드라이버가 30초 안에 기동하지 않았습니다. /dev/ttyUSB0 연결·전원과 LDS_MODEL=LDS-03 설정을 확인하세요." >&2
    exit 1
fi

echo "LDS-03 /scan 실제 메시지 수신 대기..."
if ! wait_for_ros_topic_message /scan 10 best_effort; then
    echo "LDS-03에서 /scan을 10초 안에 받지 못했습니다. LiDAR USB·전원·케이블을 확인하세요. Nav2는 시작하지 않습니다." >&2
    exit 1
fi

echo "Safety 직선 후진 거리 계산용 /odom 실제 메시지 수신 대기..."
if ! wait_for_ros_topic_message /odom 10; then
    echo "OpenCR에서 /odom을 10초 안에 받지 못했습니다. Safety 실행을 시작하지 않습니다." >&2
    exit 1
fi

echo "[2/9] 제스처 서버 시작"
GESTURE_INITIAL_CONTROL_MODE=auto "$GESTURE_PYTHON" -m src.server.app &
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

echo "[3/9] Safety 우선순위 포함 속도 mux 시작"
/usr/bin/python3 ros2_bridge/cmd_vel_mux_safety.py &
pids+=("$!")

echo "[4/9] 갑작스러운 전방 장애물 Safety supervisor 시작"
ros2 launch turtlebot3_waypoint_patrol safety_patrol.launch.py &
pids+=("$!")

echo "Safety supervisor 및 heartbeat 준비 대기..."
if ! wait_for_ros_node /turtlebot3_safety_mission_manager 10; then
    echo "Safety supervisor가 10초 안에 시작되지 않았습니다. Safety 실행을 중단합니다." >&2
    exit 1
fi
if ! wait_for_ros_topic_message /safety/heartbeat 15; then
    echo "Safety heartbeat를 15초 안에 받지 못했습니다. 속도 mux가 정지 상태이므로 실행을 중단합니다." >&2
    exit 1
fi
echo "Safety heartbeat 확인 완료 — 센서 유실 시 mux가 자동으로 속도를 차단합니다."

echo "[5/9] Wi-Fi 장갑 입력 시작 (컨트롤러 모드에서만, 앞뒤 반전 적용)"
/usr/bin/python3 ros2_bridge/wifi_glove_teleop.py \
    --invert-pitch \
    --controller-mode-only &
pids+=("$!")

if [[ "$JOYSTICK_AVAILABLE" == true ]]; then
    echo "[6/9] 조이스틱 입력 시작 (/dev/input/js0)"
    ros2 launch "$ROOT_DIR/ros2_bridge/joystick_with_mux.launch.py" &
    pids+=("$!")
else
    echo "[6/9] 조이스틱 입력 생략 (/dev/input/js0 미연결)"
fi

echo "[7/9] 저장 지도 기반 Nav2/AMCL 시작 ($MAP_PATH)"
ros2 launch "$ROOT_DIR/ros2_bridge/navigation_with_mux.launch.py" "map:=${MAP_PATH}" &
pids+=("$!")

echo "Nav2 action 서버 준비 대기..."
nav_ready=false
for _ in {1..100}; do
    if ros2 action list 2>/dev/null | grep -qx '/navigate_to_pose'; then
        nav_ready=true
        break
    fi
    sleep 0.2
done
if [[ "$nav_ready" != true ]]; then
    echo "Nav2 navigate_to_pose 서버가 20초 안에 준비되지 않았습니다. 위의 map/AMCL 오류를 확인하세요." >&2
    exit 1
fi

echo "[8/9] 제스처 컨트롤러 시작 (처음 제어권=AUTO/Nav2)"
/usr/bin/python3 ros2_bridge/cmd_vel_bridge.py \
    --base-url http://127.0.0.1:5000 \
    --disable-navigation &
pids+=("$!")

if [[ "$ENABLE_SOUND" == true ]]; then
    echo "[9/9] C270 웹캠 마이크 소리 이상감지 + OpenCR LED 시작"
    ros2 launch "$ROOT_DIR/ros2_bridge/sound_anomaly_with_led.launch.py" &
    sound_pid="$!"

    echo "웹캠 마이크 소리 이상감지 노드 준비 대기..."
    if ! wait_for_ros_node /sound_anomaly_node 45; then
        echo "웹캠 마이크 소리 이상감지 노드가 45초 안에 시작되지 않았습니다. C270 연결과 장치명을 확인하세요." >&2
        exit 1
    fi
    if ! wait_for_ros_topic_message /sound_anomaly_node/state 15; then
        echo "웹캠 마이크 소리 이상감지 상태 토픽을 받지 못했습니다. 소음 탐지를 시작하지 않습니다." >&2
        exit 1
    fi
    # 오디오 입력을 연 직후의 허브/전원 문제는 LiDAR 스트림이 먼저 끊긴다. 순찰 전에 다시
    # 센서 메시지를 확인해, 이상 징후가 있으면 모터를 움직이지 않은 채 종료한다.
    echo "웹캠 마이크 시작 후 LDS-03 /scan 안정성 재확인..."
    if ! wait_for_ros_topic_message /scan 15 best_effort; then
        echo "웹캠 마이크 시작 뒤 /scan이 끊겼습니다. USB 허브·전원 문제로 판단하여 순찰을 시작하지 않습니다." >&2
        exit 1
    fi
else
    echo "[9/9] 웹캠 마이크 소음 탐지 생략 (--with-sound으로 활성화)"
fi

if [[ "$START_PATROL" == true ]]; then
    echo "[추가] 구간별 waypoint 미션 시작 (A->B Nav2, B->C 제스처, C->D 수동, D->A Nav2)"
    /usr/bin/python3 ros2_bridge/waypoint_handoff_mission_safety.py &
    pids+=("$!")
fi

echo
echo "Safety 준비 완료: 초기 제어권은 AUTO/Nav2이며 Safety heartbeat가 감시됩니다."
if [[ "$ENABLE_SOUND" == true ]]; then
    echo "소리 감지: 첫 번째 C270 웹캠 마이크만 사용합니다. P5U는 사용하지 않습니다. IDLE=양쪽 LED, NORMAL=초록 점멸, ABNORMAL=빨강 점멸입니다."
fi
echo "짧은 따봉=제스처 ON, 제스처 ON에서 따봉 1.5초=Nav2 복귀, OK 사인=조이스틱 ON."
echo "제스처 모드에서 손 입력이 15초 동안 없으면 정지 후 컨트롤러 모드로 자동 전환됩니다."
echo "컨트롤러 모드에서만 Wi-Fi 장갑이 활성화됩니다. 스틱을 움직이면 조이스틱 우선, 스틱이 중립이면 장갑을 사용할 수 있습니다."
echo "컨트롤러 ON 중 OK 사인 1.5초=컨트롤러·제스처 OFF 후 AUTO/Nav2 복귀. 종료는 Ctrl+C입니다."
if [[ "$START_PATROL" == true ]]; then
    echo "A->B 도착 완료까지는 Safety 감시 AUTO/Nav2로 주행합니다. B 도착 후 짧은 따봉으로 제스처 모드를 켜 C로 이동하세요."
    echo "C 도착 후 OK 사인으로 조이스틱을 켜 D로 이동하세요. D 도착 후 OK 사인을 1.5초 유지하면 AUTO/Nav2로 A에 복귀합니다."
fi
wait "${pids[0]}"
