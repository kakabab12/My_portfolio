#!/usr/bin/env bash
# Start, inspect, or stop the sound-anomaly and OpenCR LED services.
set -eo pipefail

readonly ROS_SETUP="/opt/ros/humble/setup.bash"
readonly SOUND_SETUP="/home/user/ros2_ws/install/setup.bash"
readonly TURTLEBOT_SETUP="/home/user/turtlebot3_ws/install/setup.bash"
readonly TURTLEBOT_NODE="/home/user/turtlebot3_ws/install/turtlebot3_node/lib/turtlebot3_node/turtlebot3_ros"
readonly TURTLEBOT_PARAMS="/home/user/turtlebot3_ws/install/turtlebot3_node/share/turtlebot3_node/param/burger.yaml"
# OpenCR is exposed as /dev/ttyACM0.  /dev/ttyUSB0 is the CP2102 LiDAR adapter.
# Override for a one-off run with OPENCR_PORT=/dev/ttyACM1 ./sound_anomaly_led.sh start
# if the OpenCR device name changes.
readonly OPENCR_PORT="${OPENCR_PORT:-/dev/ttyACM0}"
readonly RUNTIME_DIR="/home/user/.ros/sound_anomaly_led"
readonly SOUND_PID_FILE="${RUNTIME_DIR}/sound_anomaly.pid"
readonly LED_PID_FILE="${RUNTIME_DIR}/turtlebot3_node.pid"
readonly SOUND_LOG="${RUNTIME_DIR}/sound_anomaly.log"
readonly LED_LOG="${RUNTIME_DIR}/turtlebot3_node.log"

source "${ROS_SETUP}"
source "${TURTLEBOT_SETUP}"
source "${SOUND_SETUP}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-30}"
set -u

mkdir -p "${RUNTIME_DIR}"

usage() {
  echo "Usage: $(basename "$0") {start|status|stop}"
}

running_pid() {
  local pid_file="$1"
  local pid

  [[ -s "${pid_file}" ]] || return 1
  read -r pid < "${pid_file}"
  kill -0 "${pid}" 2>/dev/null
}

stop_pid() {
  local label="$1"
  local pid_file="$2"
  local pid

  if ! running_pid "${pid_file}"; then
    rm -f "${pid_file}"
    return 0
  fi

  read -r pid < "${pid_file}"
  echo "Stopping ${label} (PID ${pid})..."
  kill -INT "${pid}"
  for _ in {1..50}; do
    kill -0 "${pid}" 2>/dev/null || break
    sleep 0.1
  done
  if kill -0 "${pid}" 2>/dev/null; then
    echo "${label} did not exit after SIGINT; leaving it running." >&2
    return 1
  fi
  rm -f "${pid_file}"
}

start() {
  if running_pid "${SOUND_PID_FILE}" || pgrep -f 'ros2 launch sound_anomaly sound_anomaly.launch.py' >/dev/null; then
    echo "sound_anomaly is already running. Run '$0 status' instead." >&2
    return 1
  fi
  if running_pid "${LED_PID_FILE}" || pgrep -f "turtlebot3_node/turtlebot3_ros.*${OPENCR_PORT}" >/dev/null; then
    echo "turtlebot3_node is already running. Run '$0 status' instead." >&2
    return 1
  fi
  if [[ ! -e "${OPENCR_PORT}" ]]; then
    echo "OpenCR was not found at ${OPENCR_PORT}. Check USB and power." >&2
    return 1
  fi

  echo "Starting OpenCR LED controller..."
  nohup "${TURTLEBOT_NODE}" -i "${OPENCR_PORT}" --ros-args \
    --params-file "${TURTLEBOT_PARAMS}" >"${LED_LOG}" 2>&1 < /dev/null &
  echo $! > "${LED_PID_FILE}"
  sleep 7
  if ! running_pid "${LED_PID_FILE}"; then
    echo "turtlebot3_node failed to start. See ${LED_LOG}" >&2
    rm -f "${LED_PID_FILE}"
    return 1
  fi

  echo "Starting sound anomaly detection..."
  nohup ros2 launch sound_anomaly sound_anomaly.launch.py \
    >"${SOUND_LOG}" 2>&1 < /dev/null &
  echo $! > "${SOUND_PID_FILE}"
  sleep 3
  if ! running_pid "${SOUND_PID_FILE}"; then
    echo "sound_anomaly failed to start. See ${SOUND_LOG}" >&2
    rm -f "${SOUND_PID_FILE}"
    stop_pid "turtlebot3_node" "${LED_PID_FILE}" || true
    return 1
  fi

  echo "Started. Use '$0 status' to inspect state and '$0 stop' to stop safely."
}

status() {
  if running_pid "${SOUND_PID_FILE}"; then
    echo "sound_anomaly: running (PID $(<"${SOUND_PID_FILE}"))"
  else
    echo "sound_anomaly: stopped"
  fi
  if running_pid "${LED_PID_FILE}"; then
    echo "turtlebot3_node: running (PID $(<"${LED_PID_FILE}"))"
  else
    echo "turtlebot3_node: stopped"
  fi

  ros2 node list 2>/dev/null || true
  ros2 topic info /opencr_led_status 2>/dev/null || true
}

stop() {
  # Stop detection first: its shutdown hook publishes LED mode 0.
  stop_pid "sound_anomaly" "${SOUND_PID_FILE}" || true

  # Send a second explicit off command while the OpenCR subscriber is still alive.
  if running_pid "${LED_PID_FILE}"; then
    ros2 topic pub --once /opencr_led_status std_msgs/msg/UInt8 "{data: 0}" || true
  fi
  stop_pid "turtlebot3_node" "${LED_PID_FILE}" || true
  echo "Stopped. OpenCR LED mode 0 was requested before the controller exited."
}

case "${1:-}" in
  start) start ;;
  status) status ;;
  stop) stop ;;
  *) usage; exit 2 ;;
esac
