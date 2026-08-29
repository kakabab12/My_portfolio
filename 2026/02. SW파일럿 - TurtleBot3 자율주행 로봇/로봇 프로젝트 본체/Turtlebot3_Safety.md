# TurtleBot3 Safety 현재 상태 — 2026-08-24

이 문서는 오늘 최종 구현과 시연 결과만 정리합니다. 이전 계획, 폐기된 회전 복구 방식, 중간 실패 분석과 임시 시험 지침은 삭제했습니다.

## 1. 현재 결론

- 전체 A→B→C→D→A 시연을 완료했습니다.
- Safety 복구는 **즉시 정지 → Nav2 목표 취소 → 직선 후진 → 장애물 제거 확인 → 기존 목표 재개**만 수행합니다.
- Safety 자체 좌우 회전 로직은 없습니다. 주행 중 보이는 회전은 Nav2의 경로 추종 또는 자체 recovery다.
- Safety는 A→B `auto_to_b`, D→A `auto_to_a` 자율주행 구간에서만 활성화됩니다.
- 제스처·컨트롤러·장갑·조이스틱 등 수동 구간에서는 Safety가 전혀 개입하지 않습니다.
- 제스처 모드에서 손 입력이 15초 동안 없으면 정지 후 컨트롤러 모드로 자동 전환됩니다.
- 바탕화면 `TurtleBot3 전체 실행 (Safety)` 아이콘에 모든 수정이 반영돼 있습니다.
- 현재 전체 프로세스는 종료 상태입니다.

## 2. 전체 미션 흐름

| 구간 | 제어 방식 | Safety |
|---|---|---|
| A→B | Nav2 자율주행 | 활성 |
| B→C | 제스처 수동 | 비활성 |
| C→D | 컨트롤러 수동 | 비활성 |
| D→A | Nav2 자율주행 | 활성 |

waypoint 원본: `/home/user/sw/robot/ros2_bridge/waypoint_handoff_mission.py`

| 지점 | map 좌표 `(x, y, yaw)` | 역할 |
|---|---|---|
| A | `(0.062, -0.072, 0.003)` | 출발 / 최종 Nav2 복귀 |
| B | `(1.358, -0.051, -1.517)` | A→B Nav2 목표 |
| C | `(1.232, -1.355, -2.081)` | 제스처 구간 도착점 |
| D | `(-0.067, -1.253, 1.627)` | 컨트롤러 구간 도착점 |

- Nav2 XY goal tolerance는 0.10m다.
- Nav2 성공 후 상위 미션은 최신 `map→base_link` TF로 실제 도착을 다시 검증합니다.
- 검증 계산에는 수치 오차용 0.02m 여유가 있어 최대 0.12m까지 허용합니다.
- yaw tolerance는 `2π`이므로 도착 방향은 강제하지 않습니다.
- C와 D 수동 도착 반경은 0.10m이며 1초간 안정적으로 머물러야 확정됩니다.

## 3. Safety 복구 사양

### 3.1 상태기계

```text
MONITORING
  → CANDIDATE
  → WAIT_NAV_CANCEL
  → REVERSE
  → CLEAR_WAIT
  → WAIT_NAV_RESUME
  → COOLDOWN
  → MONITORING

센서·미션 heartbeat 유실, Nav2 취소/재개 실패, 후방 장애물,
후진 제한시간 초과, costmap 정리 실패 → HALT + 0 속도 유지
```

### 3.2 갑작스러운 장애물 판정

- LiDAR 정면 ±15도, 0.35m 이내만 자동 후진 후보로 봅니다.
- 최소 8 points가 2 frame 연속 확인돼야 합니다.
- 최근 전진 의도 0.06m/s 이상이며 각속도 0.30rad/s 이하인 자율주행 직진 상황에서만 복구를 시작합니다.
- `LaserScan.header.stamp` 시각의 `map→base_scan` TF로 scan endpoint를 지도에 투영합니다.
- 저장 지도 주변 0.15m에 원래 존재하는 벽·구조물은 갑툭튀 후보에서 제외합니다.
- 정확한 scan 시각 TF는 최대 0.08초 기다립니다.
- TF가 없으면 raw scan으로 우회하지 않고 해당 frame의 자동 후진을 금지합니다.
- 자율주행 중 0.15m 절대 보호정지는 별도 정면 ±30도·최소 3 points 조건으로 유지합니다.
- 수동 구간에서는 자동 후진뿐 아니라 0.15m 보호정지도 사용하지 않습니다.

### 3.3 정지와 직선 후진

1. Safety가 속도 우선권을 가져와 즉시 0 속도를 보냅니다.
2. 진행 중인 Nav2 목표를 취소합니다.
3. 감지 순간 odom 위치와 진행축을 후진 기준으로 저장합니다.
4. Nav2의 최종 취소 확인 후 0.10m/s로 직선 후진합니다.
5. 감지 위치 기준 순후퇴 거리가 0.25m가 되면 정지합니다.
6. 후방 ±35도, 0.25m 이내에 물체가 있으면 후진하지 않고 `HALT`합니다.

Safety 후진 과정에서 좌우 회전, 회전량 보정, 회전 후 odom 좌표 보정은 사용하지 않습니다.

### 3.4 장애물 제거와 경로 복귀

- 후진 완료 후 `CLEAR_WAIT`에서 0 속도를 유지합니다.
- 사람이 장애물을 치울 때까지 시간 제한 없이 기다립니다.
- 장애물 확정 순간의 동적 군집을 map 좌표로 저장합니다.
- 현재 scan에서 원래 장애물 군집과 0.08m 이내로 일치하는 points가 3개 미만인 상태가 1초 지속되면 제거 완료로 봅니다.
- 주변의 다른 벽이나 구조물이 보여도 원래 감지 군집과 일치하지 않으면 재개를 막지 않습니다.
- 최신 `map→base_link` TF를 확인하고 global/local costmap을 모두 정리합니다.
- 중단했던 B 또는 A 목표를 다시 전송합니다.
- 새 global path의 시작점과 끝점이 유효한지 확인한 뒤 Safety 제어권을 해제합니다.
- 재개 과정에서는 `/initialpose` 재발행이나 과거 AMCL 위치 보정을 하지 않습니다.

## 4. 주요 Safety 설정값

설정 원본: `/home/user/turtlebot3_ws/src/turtlebot3_waypoint_patrol/config/safety_mission.yaml`

| 항목 | 현재 값 | 의미 |
|---|---:|---|
| `front_angle_deg` | ±15도 | 자동 후진 정면 범위 |
| `sudden_trigger_distance` | 0.35m | 갑툭튀 후보 거리 |
| `min_surprise_points` | 8 | 자동 후진 최소 군집 |
| `consecutive_scan_frames` | 2 | 연속 확인 frame |
| `static_map_margin` | 0.15m | 저장 지도 주변 제외 범위 |
| `scan_tf_wait_sec` | 0.08s | scan 시각 TF 대기 상한 |
| `minimum_forward_speed` | 0.06m/s | 전진 의도 기준 |
| `maximum_angular_speed` | 0.30rad/s | 직진 판별 기준 |
| `protective_angle_deg` | ±30도 | 자율구간 절대 보호정지 범위 |
| `hard_stop_distance` | 0.15m | 절대 보호정지 거리 |
| `hard_release_distance` | 0.20m | 절대 보호정지 해제 거리 |
| `escape_speed` | 0.10m/s | 후진 속도 |
| `escape_distance` | 0.25m | 감지 위치 기준 후진 거리 |
| `escape_timeout_sec` | 6.0s | 후진 제한시간 |
| `rear_stop_distance` | 0.25m | 후방 장애물 정지 거리 |
| `resume_clear_distance` | 0.75m | 저장 군집 재탐색 거리 |
| `clear_match_distance` | 0.08m | 원래 장애물 군집 일치 반경 |
| `clear_stable_sec` | 1.0s | 제거 안정 확인시간 |
| `clear_wait_timeout_sec` | 0.0s | 장애물 제거까지 무기한 대기 |
| `resume_timeout_sec` | 10.0s | Nav2 재개 확인 제한시간 |

## 5. 제스처 무입력 15초 자동 전환

- B 도착으로 제스처 모드가 켜지는 순간 15초 타이머를 시작합니다.
- 카메라가 손을 정상 검출하면 입력 시각을 갱신합니다.
- 손 입력이 15초 연속 없으면 0 속도를 먼저 보내고 제스처 모드를 끕니다.
- 이어서 컨트롤러 모드를 켭니다.
- 컨트롤러 모드에서는 조이스틱 입력이 활성일 때 조이스틱이 우선합니다.
- 조이스틱이 중립이면 Wi-Fi 장갑을 사용할 수 있습니다.
- 1m 이동·90도 회전·180도 wave 같은 원샷 명령이 실행 중이면 중간에 끊지 않고 명령 완료 후 전환합니다.
- C 도착 전에 자동 전환돼도 컨트롤러로 C까지 계속 이동할 수 있습니다.
- 컨트롤러 상태로 C 도착이 확인되면 별도 재전환 없이 바로 D 구간으로 이어집니다.
- 이 기능은 `/home/user/sw/robot/ros2_bridge/cmd_vel_bridge.py`의 기본값 `15.0s`로 실행됩니다.

## 6. 2026-08-24 최종 시연 결과

최종 실행은 바탕화면 Safety 아이콘과 같은 경로로 진행했습니다.

```text
A→B Nav2 시작
B 실제 도착: TF 거리 0.092m
B→C 제스처 이동
C 도착 반경 진입: 0.04m, 1초 안정화 후 완료
제스처 손 입력 없음 15초
정지 후 컨트롤러 모드 자동 전환
C→D 컨트롤러 이동
D 도착 반경 진입: 0.06m, 1초 안정화 후 완료
D→A Nav2 복귀 시작
정면 신규 군집: 0.34m / 17 points
Nav2 A 목표 취소
감지 위치 기준 0.26m 직선 후진
장애물 제거 확인 및 costmap 정리
A 목표 재전송 및 새 경로 확인
A 실제 도착: TF 거리 0.094m
A→B→C→D→A 미션 완료
```

확인된 핵심 결과:

- 무장애 A→B에서 B 주변 벽을 Safety 장애물로 오인하지 않았습니다.
- B 실제 도착 검증이 한 번에 통과했습니다.
- 제스처 원샷 이동·회전·주먹 정지가 실제 동작했습니다.
- 15초 무입력 자동 컨트롤러 전환이 실제 로그와 로봇 동작으로 확인됐습니다.
- C와 D 수동 도착 판정이 정상 작동했습니다.
- D→A 자율주행에서 Safety 정지·후진·장애물 제거·A 목표 재개가 성공했습니다.
- A 실제 도착 검증 후 전체 미션이 완료됐습니다.
- 사용자 요청으로 `Ctrl+C` 종료했고 관련 프로세스가 남지 않은 것을 확인했습니다.

실기 로그:

- 미션: `/home/user/.ros/log/python3_82969_1787563888631.log`
- 제스처: `/home/user/.ros/log/python3_82968_1787563888659.log`
- Safety: `/home/user/.ros/log/python3_82663_1787563873283.log`

## 7. 테스트와 검증

```bash
cd /home/user/sw/robot
python3 -m unittest discover -s tests -v
```

- 전체 회귀 테스트: `203 tests, OK`
- 15초 경계 전환 테스트 통과
- 전환 시 제스처 OFF·컨트롤러 ON·정지 발행 테스트 통과
- 원샷 명령 실행 중 타임아웃 보류 테스트 통과
- 컨트롤러 전환 후 C 도착과 D 구간 연결 테스트 통과
- 기존 Safety·mux·waypoint·제스처 테스트 전부 통과
- Python 문법 검사 통과
- 통합 실행 스크립트 Bash 문법 검사 통과
- 바탕화면 Safety `.desktop` 실행 파일 검증 통과
- `turtlebot3_waypoint_patrol` 마지막 빌드 성공 로그: `/home/user/turtlebot3_ws/log/build_2026-08-24_18-05-08/`

Python 브리지와 미션 파일은 통합 실행기에서 원본을 직접 실행하므로 이번 15초 전환 수정에 별도 `colcon build`는 필요하지 않습니다.

## 8. 실행 파일과 주요 코드

### 바탕화면과 실행기

- `/home/user/Desktop/TurtleBot3_전체실행_Safety.desktop`
- `/home/user/sw/robot/scripts/turtlebot3_전체실행_safety.sh`
- `/home/user/sw/robot/scripts/run_navigation_gesture_joystick_safety.sh`

바탕화면 아이콘의 실제 실행 경로:

```text
TurtleBot3_전체실행_Safety.desktop
  → turtlebot3_전체실행_safety.sh
  → run_navigation_gesture_joystick_safety.sh
  → cmd_vel_bridge.py + waypoint_handoff_mission_safety.py
```

### 미션·제스처·mux

- `/home/user/sw/robot/ros2_bridge/cmd_vel_bridge.py`
- `/home/user/sw/robot/ros2_bridge/cmd_vel_mux_safety.py`
- `/home/user/sw/robot/ros2_bridge/navigation_with_mux.launch.py`
- `/home/user/sw/robot/ros2_bridge/waypoint_handoff_mission.py`
- `/home/user/sw/robot/ros2_bridge/waypoint_handoff_mission_safety.py`

### TurtleBot3 Safety package

- `/home/user/turtlebot3_ws/src/turtlebot3_waypoint_patrol/turtlebot3_waypoint_patrol/safety_mission_manager.py`
- `/home/user/turtlebot3_ws/src/turtlebot3_waypoint_patrol/config/safety_mission.yaml`
- `/home/user/turtlebot3_ws/src/turtlebot3_waypoint_patrol/launch/safety_patrol.launch.py`
- runtime mirror: `/home/user/sw/robot/runtime_sources/turtlebot3_ws/src/turtlebot3_waypoint_patrol/`

### 테스트

- `/home/user/sw/robot/tests/test_cmd_vel_bridge.py`
- `/home/user/sw/robot/tests/test_waypoint_tolerances.py`
- `/home/user/sw/robot/tests/test_safety_components.py`

## 9. 다음 실행 방법

실행 전:

1. 관련 ROS/TurtleBot3 프로세스가 종료 상태인지 확인합니다.
2. 로봇을 A 좌표와 진행 방향에 맞게 배치합니다.
3. 후진 시험을 할 때 로봇 뒤쪽 0.25m 이상을 비웁니다.
4. 갑툭튀 장애물은 로봇을 감싸지 말고 진행 방향 정면만 가립니다.

바탕화면에서 `TurtleBot3 전체 실행 (Safety)` 아이콘을 실행하거나 다음 명령을 사용합니다.

```bash
env LDS_MODEL=LDS-03 /bin/bash /home/user/sw/robot/scripts/turtlebot3_전체실행_safety.sh
```

같은 네트워크의 태블릿 웹 화면:

```text
http://192.168.137.66:5000
```

IP 주소는 네트워크 재연결 후 바뀔 수 있습니다. 웹 서버는 전체 프로세스가 실행 중일 때만 열립니다.

장애물 시험:

1. A→B 또는 D→A 자율주행 중 로봇 정면 근거리에 장애물을 갑자기 놓습니다.
2. 정지와 약 0.25m 직선 후진이 끝날 때까지 장애물을 유지합니다.
3. 로봇이 완전히 정지한 뒤 장애물을 치웁니다.
4. 기존 B 또는 A 목표로 복귀하는지 확인합니다.

제스처 자동 전환 시험:

1. B 도착 후 제스처 모드 진입을 확인합니다.
2. 두 카메라 모두에서 손이 보이지 않게 합니다.
3. 15초 뒤 로봇이 정지 상태를 유지하며 컨트롤러 모드로 바뀌는지 확인합니다.
4. 조이스틱 또는 Wi-Fi 장갑으로 C/D 이동을 이어갑니다.

종료는 실행 터미널에서 `Ctrl+C` 한 번입니다. 종료 후 관련 프로세스가 남지 않았는지 확인합니다.

## 10. 주의사항

- Safety 회전 코드는 없지만 Nav2는 경로 탐색 과정에서 회전할 수 있습니다.
- 작은 44×43 지도 경계에서 Nav2 `worldToMap failed` 경고가 보일 수 있습니다. 최종 시연은 이 경고가 있어도 전체 미션을 완료했습니다.
- 실제 경로가 지도와 명확히 어긋나거나 충돌 위험이 있으면 즉시 `Ctrl+C`로 종료합니다.
- Safety는 수동 모드에 개입하지 않으므로 제스처·조이스틱·장갑 조작 중 장애물 회피 책임은 조작자에게 있습니다.
- 장애물 제거 전에는 로봇 뒤쪽 안전 공간과 실물 정지를 직접 확인합니다.
