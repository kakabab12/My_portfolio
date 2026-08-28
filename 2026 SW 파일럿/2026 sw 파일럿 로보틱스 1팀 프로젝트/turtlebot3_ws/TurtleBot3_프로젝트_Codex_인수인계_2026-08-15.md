# TurtleBot3 공장 순찰·제스처 수동복귀 프로젝트 인수인계

- 작성일: 2026-08-15
- 인수 대상: Jetson Orin Nano에서 실행되는 Codex
- 프로젝트 역할: TurtleBot3 Burger 자율주행 파트
- 문서 목적: 지금까지의 대화, 첨부 코드 분석 결과, 확정된 시나리오와 다음 작업 순서를 한 번에 전달한다.

> 이 문서를 읽는 Codex는 새 구조를 임의로 제안하기 전에 실제 Jetson의 작업공간과 설치 상태를 먼저 확인해야 한다. 특히 이미 완성도가 있는 제스처 시스템의 패키지와 파일 구조를 유지하고, 외부 제스처 패키지나 불필요한 새 노드를 추가하지 않는다.

---

## 1. 프로젝트 환경과 전제

### 하드웨어

- 이동 로봇: TurtleBot3 Burger
- 메인 컴퓨터: Jetson Orin Nano
- 모터 제어: OpenCR + DYNAMIXEL
- 거리 센서: TurtleBot3 탑재 2D LiDAR, ROS 2 토픽 `/scan`
- 제스처 입력: 다른 노트북의 일반 RGB 웹캠
- 공장 모형: 외곽 규격 2m × 2m 정사각형

### 소프트웨어

- Ubuntu 22.04 LTS
- ROS 2 Humble
- TurtleBot3 bringup
- SLAM Toolbox 사용 예정
- Nav2 + AMCL 사용 예정
- 기존 제스처 인식 시스템은 MediaPipe, OpenCV, Flask 기반
- 수동·자율 속도 명령 중재는 `twist_mux` 사용 예정

### 중요한 과거 결정

- 공식 TurtleBot3 e-Manual의 기본 SLAM 예제는 Cartographer지만, 이 프로젝트에서는 SLAM Toolbox를 사용하기로 결정했다.
- 기계학습이나 강화학습은 현재 고정 Waypoint 시나리오에 필요하지 않아 제외한다.
- 로봇팔 계획은 취소되었다.
- 1차 목표에서는 RealSense와 라이다 갈림길 선택 기능을 필수 기능으로 취급하지 않는다.
- 사용자가 완성한 제스처 코드와 그 안에서 사용한 패키지를 유지한다.

---

## 2. 현재 확정된 1차 시연 시나리오

공장 모형 안에서 다음 세 지점을 사용한다.

| 지점 | 대략적인 물리적 위치 | 역할 |
| --- | --- | --- |
| A | 왼쪽 아래 | 시작점, 홈, 수동복귀 목표 |
| B | 왼쪽 위 | 첫 번째 Nav2 목적지 |
| C | 오른쪽 위 | 두 번째 Nav2 목적지, 자율주행 종료점 |

최종 동작 순서는 다음과 같다.

```text
로봇을 A에 배치
→ AMCL 초기 위치 설정
→ 사용자가 제스처로 autonomous 모드 전환
→ Nav2로 B까지 자율주행
→ B 도착 확인
→ Nav2로 C까지 자율주행
→ C 도착 후 자율 미션 완료 및 정지
→ 사용자가 기존 한 손가락 제스처로 manual 모드 전환
→ C에서 A까지 기존 D-pad 제스처만으로 수동주행
→ 사용자가 A 도착을 눈으로 확인하고 정지
```

### A의 정확한 의미

A도 `map` 좌표계 안에 저장되는 고정 Pose다. 하지만 자동주행 경로에는 B와 C만 들어간다.

```text
고정 Pose: A, B, C
자동 경로: [B, C]
수동 복귀 목표: A
```

A·B·C가 좌표계 자체인 것은 아니다.

- 고정좌표계: `map`
- A·B·C: `map` 좌표계에 속한 고정 Pose `(x, y, yaw)`
- 같은 저장 지도를 계속 사용하면 A·B·C는 사실상 절대 위치처럼 취급할 수 있다.
- 지도를 새로 SLAM해서 저장하면 `origin`이 달라질 수 있으므로 A·B·C를 다시 측정해야 한다.

---

## 3. 지금까지 분석한 기존 작업공간

분석 대상 압축본은 `20260811_ws.zip`이었다.

### 중요한 구조

```text
20260811_ws/
├── camera_server/
│   └── camera_server.py
├── configs/
│   ├── config.yaml
│   ├── twist_mux_config.yaml
│   └── waypoints.yaml
├── models/weights/
│   └── hand_landmarker.task
├── ros2_bridge/
│   ├── cmd_vel_bridge.py
│   └── nav_mission_node.py
├── src/
│   ├── capture/
│   ├── control/
│   ├── inference/
│   ├── pipeline/
│   ├── postprocess/
│   ├── server/
│   └── utils/
├── tests/
├── requirements-camera-server.txt
├── requirements-gesture.txt
├── requirements-realsense.txt
├── requirements-ros2.txt
└── README.md
```

### 현재는 정식 ROS 2 Python 패키지가 아님

압축본에는 다음 파일이 없었다.

```text
package.xml
setup.py
setup.cfg
launch/
```

따라서 현재 구조는 `ament_python` 패키지가 아니라 Python 스크립트를 직접 실행하는 작업공간이다.

```bash
python3 ros2_bridge/cmd_vel_bridge.py
python3 ros2_bridge/nav_mission_node.py
```

현재 기준 수치는 다음과 같다.

- 커스텀 ROS 2 패키지: 0개
- `setup.py` 등록 실행 노드: 0개
- 직접 작성된 커스텀 ROS 노드 파일: 2개
- 새로 만들 필요가 있는 커스텀 노드: 현재 시나리오 기준 0개

나중에 모든 기능이 개별 검증된 뒤 필요하면 하나의 `ament_python` 패키지로 감싸고 두 실행 파일을 등록할 수 있다. 지금은 패키징보다 기능 검증이 우선이다.

---

## 4. 전체 실행 구조

```text
[원격 노트북]
웹캠
→ camera_server/camera_server.py
→ MJPEG 영상 송출

[Jetson Orin Nano]
src.server.app
→ MediaPipe 손 인식
→ 손모양 및 D-pad 판정
→ Flask GET /cmd

ros2_bridge/cmd_vel_bridge.py
→ Flask /cmd 폴링
→ geometry_msgs/msg/Twist 변환
→ /cmd_vel_manual

ros2_bridge/nav_mission_node.py
→ Flask의 nav_mode 확인
→ Nav2에 B, C 목적지 요청

Nav2 controller_server
→ /cmd_vel_nav

/cmd_vel_manual ─┐
                 ├→ twist_mux → /cmd_vel → turtlebot3_node → OpenCR → 모터
/cmd_vel_nav ────┘
```

### 직접 관리하는 주요 프로세스와 노드

| 이름 | 종류 | 실행 위치 | 역할 |
| --- | --- | --- | --- |
| `camera_server.py` | 비ROS 프로세스 | 원격 노트북 | 웹캠 영상만 MJPEG로 송출 |
| `src.server.app` | 비ROS 프로세스 | Jetson | 손 인식, D-pad, 모드 상태, Flask API |
| `gesture_cmd_vel_bridge` | 커스텀 ROS 노드 | Jetson | 제스처 명령을 `/cmd_vel_manual`로 변환 |
| `nav_mission_node` | 커스텀 ROS 노드 | Jetson | Nav2 목표 순서와 미션 상태 관리 |
| `twist_mux` | 기존 ROS 노드 | Jetson | 수동·자율 속도 중재 |

TurtleBot3 bringup, Map Server, AMCL, Nav2의 Planner·Controller·BT Navigator 등은 설치된 패키지에서 실행되는 기존 노드다. 전체 `ros2 node list`에는 약 17~20개가 보일 수 있지만, 우리가 직접 작성한 ROS 노드 파일은 2개다.

---

## 5. 기존 제스처 시스템에서 유지해야 할 패키지

### 노트북 카메라 송출

`requirements-camera-server.txt`

- `opencv-contrib-python==4.10.0.84`
- `flask==3.0.3`

노트북은 손 인식을 수행하지 않고 영상만 보낸다.

### Jetson 제스처 엔진

`requirements-gesture.txt`

- `mediapipe==0.10.14`
- `opencv-contrib-python==4.10.0.84`
- `numpy==1.26.4`
- `pyyaml==6.0.3`
- `flask==3.0.3`

### ROS 2 브리지와 자율주행

- `requests==2.32.3`
- `pyyaml==6.0.3`
- `rclpy`
- `geometry_msgs`
- `sensor_msgs`
- `nav2_simple_commander`
- `twist_mux`
- `turtlebot3_bringup`
- `turtlebot3_navigation2`
- `slam_toolbox`

### 이번 1차 목표에서 추가하지 말 것

- 새로운 외부 제스처 ROS 패키지
- YOLO 기반 손 검출
- `GestureBot` 같은 별도 저장소
- 별도 `gesture_teleop.py`
- 별도 `waypoint_node.py`
- 불필요한 `collision_monitor` 추가
- RealSense가 없는데 `pyrealsense2` 강제 설치

현재 제스처 처리와 수동 속도 생성은 이미 기존 시스템에 있으므로 교체하지 않는다.

---

## 6. `cmd_vel_bridge.py` 분석 결과

노드 이름은 `gesture_cmd_vel_bridge`다.

주요 역할:

1. Flask `/cmd`를 약 20Hz로 폴링한다.
2. `linear_x`, `angular_z`, `nav_mode`, `age_sec`를 읽는다.
3. 수동 모드에서만 `/cmd_vel_manual`을 발행한다.
4. 자율 모드에서는 0을 계속 발행하지 않고 수동 채널을 침묵시킨다.
5. 응답 실패나 오래된 명령이면 0 `Twist`를 발행한다.
6. TurtleBot3 Burger 속도 한계로 값을 다시 제한한다.

중요한 토픽 구조:

```text
제스처 수동조종 → /cmd_vel_manual
Nav2 자율주행   → /cmd_vel_nav
twist_mux 출력  → /cmd_vel
```

Nav2가 기본 `/cmd_vel`을 직접 발행하면 `twist_mux`를 우회하므로, 실기에서 Nav2 Controller 출력을 `/cmd_vel_nav`로 리매핑하는 검증이 반드시 필요하다.

---

## 7. `nav_mission_node.py` 전체 분석 결과

분석한 파일은 304줄, 15,606바이트였고 Python 문법 검사를 통과했다. 별도로 업로드된 파일과 압축본 내부 파일의 SHA-256 해시가 동일했다.

### 현재 구현된 기능

- Flask `/cmd`에서 `nav_mode` 조회
- `nav_mode == autonomous`일 때만 미션 수행
- `waypoints.yaml` 읽기
- `BasicNavigator.goToPose()`로 목표 Pose 전달
- Nav2 완료 여부와 `TaskResult` 확인
- 순차 ID의 다음 Waypoint 전달
- 수동 전환 감지 시 `cancelTask()` 호출
- `/scan` 구독
- 등록된 갈림길의 라이다 조기 감지
- 갈림길에서 Flask에 방향 선택 요청

### 현재 계획 부합도

현재 1차 시나리오에 대한 부합도는 약 **68%**로 평가했다.

| 계획 항목 | 배점 | 현재 점수 | 판단 |
| --- | ---: | ---: | --- |
| A에서 시작 | 10 | 5 | A를 시작 위치가 아니라 첫 Nav2 목표로 다시 지정 |
| A→B Nav2 주행 | 20 | 15 | 가능하지만 첫 목표 처리 수정 필요 |
| B→C Nav2 주행 | 20 | 18 | 순차 ID이면 가능 |
| C 도착 후 완전 정지 | 15 | 4 | 마지막 지점에서 C 목표를 다시 호출하는 문제 |
| C에서 수동 전환 | 15 | 8 | 외부 전환은 감지하지만 스스로 전환하지 않음 |
| C→A 제스처 수동주행 | 15 | 13 | Nav2 취소와 수동 인계 가능, A 도착 판단 없음 |
| 안전 처리 | 5 | 5 | 통신 실패 시 수동 간주 및 목표 취소 |

### 현재 코드의 실제 순서

A=1, B=2, C=3으로 등록했다고 가정하면 다음처럼 동작한다.

```text
manual 상태에서 대기
→ 사용자가 autonomous 전환
→ 가장 작은 ID인 A를 첫 Nav2 목표로 전송
→ A 성공 후 B 전송
→ B 성공 후 C 전송
→ C 성공 후 다음 ID 없음
→ 상태를 IDLE로 변경
→ nav_mode가 아직 autonomous이므로 다음 tick에서 C를 다시 전송
→ 수동 전환 전까지 C 목표 반복
```

### 필수 수정사항

#### 1. 자동 경로를 `[B, C]`로 명시

현재 코드는 가장 작은 ID를 첫 목표로 사용한다.

```python
self._current_id = min(self._waypoints)
```

원하는 구조는 다음과 같다.

```yaml
waypoints:
  A: {x: ..., y: ..., yaw: ...}
  B: {x: ..., y: ..., yaw: ...}
  C: {x: ..., y: ..., yaw: ...}

autonomous_route: [B, C]
```

#### 2. `STATE_COMPLETED` 추가

C 도착 후 다음 목표를 다시 보내지 않아야 한다.

```text
C 도착
→ STATE_COMPLETED
→ Nav2 목표 추가 발행 금지
→ 기존 제스처의 manual 전환 대기
```

#### 3. 미션 재시작 초기화

수동에서 자율로 새로 전환할 때 현재 목표를 다시 B로 초기화해야 한다. 현재 코드는 `_current_id`가 C에 남을 수 있다.

#### 4. 실패 재시도 제한

현재는 B나 C 실패 시 같은 목표를 무한 재시도한다. 예를 들어 같은 지점을 최대 3회 시도한 뒤 `MISSION_FAILED`로 정지하도록 바꾸는 것이 좋다.

#### 5. `yaw` 지원

현재 모든 Pose는 `orientation.w = 1.0`, 즉 `yaw=0`으로 고정된다. C 도착 시 수동복귀가 편하도록 `waypoints.yaml`의 `yaw`를 읽어 Quaternion으로 변환하는 기능이 권장된다.

### 현재 시나리오에서 불필요한 부분

- 라이다 갈림길 조기 감지
- `/fork/request`, `/fork/status`
- 방향 선택을 통한 분기

삭제할 필요까지는 없지만 B와 C의 설정에 `fork:`를 넣지 않아야 한다. 1차 시연에서는 이 기능이 실행되지 않게 하는 편이 안전하다.

필수 수정 후 계획 부합도는 약 92~95%로 예상한다. 나머지는 실제 좌표, Nav2 리매핑, 실기 검증 영역이다.

---

## 8. SLAM, Nav2, Waypoint와 커스텀 노드의 관계

### SLAM을 한다고 새 Python 노드 파일이 생기지 않음

SLAM Toolbox는 설치된 ROS 패키지의 노드를 실행한다. 우리가 `slam_node.py`를 작성하는 작업이 아니다.

지도 저장 결과:

```text
factory_map.pgm
factory_map.yaml
```

### Nav2를 한다고 새 Python 노드 파일이 생기지 않음

저장된 지도로 Navigation을 실행하면 다음과 같은 기존 노드가 실행된다.

- Map Server
- AMCL
- Planner Server
- Controller Server
- BT Navigator
- Behavior Server
- Costmaps
- Lifecycle Managers

### Waypoint 좌표를 지정한다고 새 노드 파일이 생기지 않음

좌표는 `waypoints.yaml`이라는 데이터 파일에 저장한다. 자동 B→C 실행은 기존 `nav_mission_node.py`가 담당한다.

따라서 새 `waypoint_node.py`를 만들지 않는다. 목적지 순서와 미션 상태는 `nav_mission_node.py` 하나에서 관리한다.

### SLAM·Nav2 소스와 `nav_mission_node.py`를 합치지 않음

- SLAM: 지도 제작 단계
- AMCL/Nav2: 저장된 지도에서 위치 추정 및 경로주행
- `nav_mission_node.py`: Nav2에 언제 어디로 갈지를 전달하는 상위 미션 관리자

소스코드를 합치는 것이 아니라 나중에 하나의 Launch 파일에서 함께 실행한다.

---

## 9. RViz와 좌표 측정에 관한 결론

SLAM 중 RViz에서는 로봇이 지도 위에서 실시간으로 움직이는 모습을 볼 수 있다. 하지만 숫자 `x`, `y`, `yaw`가 기본 화면에 자동 표시되는 것은 아니다.

### 로봇의 실시간 `map` 좌표 확인

```bash
ros2 run tf2_ros tf2_echo map base_footprint
```

프레임이 다르면 다음도 확인한다.

```bash
ros2 run tf2_ros tf2_echo map base_link
```

Waypoint에는 `/odom` 좌표가 아니라 `map` 기준 좌표를 사용한다.

### RViz에서 클릭한 지점 좌표 확인

1. RViz `Fixed Frame`을 `map`으로 설정한다.
2. `Publish Point`를 누른다.
3. 원하는 위치를 클릭한다.
4. 다음 토픽으로 확인한다.

```bash
ros2 topic echo /clicked_point
```

이 방법은 클릭한 점의 `x`, `y`를 얻으며 방향 `yaw`는 포함하지 않는다.

### SLAM 중 좌표를 최종값으로 확정하지 않는 이유

SLAM 중 Loop Closure와 지도 최적화가 발생하면 `map` 기준 로봇 위치가 조금 보정될 수 있다. 따라서:

- SLAM 중에는 A·B·C의 물리적 위치를 결정한다.
- 바닥에 테이프로 표시한다.
- 좌표는 후보값으로 기록한다.
- 최종 지도 저장 후 AMCL로 다시 위치를 잡는다.
- 같은 물리적 A·B·C에서 최종 좌표를 다시 측정한다.

---

## 10. 빈 정사각형 지도와 나중 장애물 배치에 관한 결론

사용자가 제안한 방식은 1차 MVP에서 성립한다.

```text
외곽 2m × 2m 벽만 정적 지도에 저장
+ map 좌표계 안의 고정 A·B·C
+ 나중에 배치한 실시간 장애물
+ Nav2 Costmap 회피
```

같은 `factory_map.yaml`을 계속 사용하는 한 A·B·C 좌표는 그대로 유지된다. 나중에 놓은 장애물은 라이다와 Nav2의 Costmap이 실시간으로 감지할 수 있다.

### 가능한 조건

- 장애물이 A·B·C 자체를 덮지 않아야 한다.
- 장애물이 경로를 완전히 차단하지 않아야 한다.
- 로봇이 통과할 수 있는 충분한 폭이 있어야 한다.
- 장애물이 LiDAR 스캔 평면보다 너무 낮지 않아야 한다.
- 같은 외벽과 같은 저장 지도를 유지해야 한다.

### 2m × 2m에서 주의할 점

TurtleBot3 Burger의 크기는 약 138mm × 178mm지만, Nav2는 로봇 크기만 보는 것이 아니라 Costmap Inflation 안전영역도 본다. 안정적인 초기 실험을 위해:

- A·B·C 중심은 벽이나 장애물에서 약 30cm 이상 떨어뜨린다.
- 통로는 초기 실험 기준 약 40~50cm 이상 확보한다.
- C에서는 정지 후 수동으로 회전할 공간까지 남긴다.

위 수치는 물리적 절대 최소폭이 아니라, 작은 2m × 2m 모형에서 안정적으로 주행·회전시키기 위한 초기 설계 권장치다. 실제 Nav2 footprint와 inflation 설정을 확인한 뒤 조정한다.

### 완전한 빈 정사각형의 문제

완전히 대칭적인 정사각형은 네 모서리가 비슷해 AMCL이 방향이나 위치를 헷갈릴 수 있다. 전체 설비를 먼저 놓을 필요는 없지만, 다음 중 하나 정도를 최초 지도에 포함하면 좋다.

- 한쪽 벽의 작은 돌출 구조
- 한쪽 모서리만 다른 형상
- LiDAR가 인식할 수 있는 고정 기둥 하나
- 작은 L자 벽 구조

즉 다음 판단이다.

```text
완전히 빈 대칭 정사각형        가능하지만 위치추정에 불리
외벽 + 비대칭 고정 특징 하나    권장
전체 설비를 미리 배치            필수 아님
```

---

## 11. 권장 개발 순서

현재 `nav_mission_node.py`는 지도와 기본 Nav2 검증 전에 만들어진 프로토타입이다. 폐기할 필요는 없지만 지금 당장 먼저 고치는 것보다 기반을 순차적으로 검증해야 한다.

### 1단계: 기존 제스처 수동주행 확인

확인 흐름:

```text
제스처
→ /cmd_vel_manual
→ twist_mux
→ /cmd_vel
→ TurtleBot3
```

확인 사항:

- 전진·후진·좌회전·우회전
- 손을 놓으면 즉시 정지
- 제스처 서버 통신이 끊기면 정지
- 수동 속도 상한 적용

### 2단계: 공장 외곽 구성 확정 및 SLAM

- 2m × 2m 외벽 설치
- 가능하면 비대칭 고정 특징 하나 추가
- TurtleBot3 bringup 실행
- SLAM Toolbox 실행
- 컨트롤러 또는 검증된 수동 제스처로 전체 공간 주행
- 지도 품질 확인
- 최종 지도 저장

예상 지도 파일:

```text
factory_map.pgm
factory_map.yaml
```

### 3단계: 기본 Nav2 단일 목표 검증

아직 `nav_mission_node.py`를 실행하지 않는다.

1. 저장된 `factory_map.yaml`을 Nav2로 불러온다.
2. RViz `2D Pose Estimate`로 A에서 AMCL 초기 위치를 설정한다.
3. RViz `Navigation2 Goal`로 B를 한 번 클릭한다.
4. B까지 잘 가는지 확인한다.
5. RViz에서 C를 클릭한다.
6. C까지 잘 가는지 확인한다.
7. 장애물을 놓고 회피 여부를 확인한다.

B와 C에 단일 목표로도 못 가면 미션 노드를 수정해도 자동 순찰은 성공하지 않는다.

### 4단계: A·B·C 최종 좌표 확정

- SLAM 중 정한 A·B·C 위치를 바닥 테이프로 표시한다.
- 저장 지도를 Nav2로 다시 불러온다.
- AMCL을 설정한다.
- 각 테이프 위치에서 `map → base_footprint` 좌표를 측정한다.
- 최종 `(x, y, yaw)`를 `waypoints.yaml`에 기록한다.

### 5단계: `nav_mission_node.py` 최소 수정

- `autonomous_route: [B, C]`
- `STATE_COMPLETED`
- C 목표 반복 방지
- 수동→자율 재시작 시 B로 초기화
- 목표 실패 재시도 제한
- Waypoint `yaw` 지원
- B와 C의 `fork:` 제거

### 6단계: 제스처 수동복귀 결합

최종 통합 검증:

```text
A에서 autonomous 전환
→ B 자율주행
→ C 자율주행
→ C에서 COMPLETED 및 정지
→ 기존 한 손가락 제스처로 manual 전환
→ 기존 손바닥 D-pad로 A까지 수동복귀
```

### 7단계: 마지막에 Launch 통합 및 패키징

각 기능이 개별적으로 검증된 뒤에만 단일 Launch 파일을 만든다.

```text
factory_patrol.launch.py
├── TurtleBot3 bringup
├── Navigation2 + AMCL + Map Server
├── gesture_cmd_vel_bridge
├── twist_mux
└── nav_mission_node
```

필요하면 이 단계에서 `package.xml`, `setup.py`, `setup.cfg`를 추가해 `ament_python` 패키지로 구성한다.

---

## 12. 사용자가 말한 진행 방식으로 정리하면

이 부분은 현재 사용자의 최신 의도를 그대로 실행 순서로 정리한 것이다.

1. 2m × 2m 외벽을 설치한다.
2. 가능하면 한쪽에 비대칭 고정 특징 하나를 둔다.
3. 컨트롤러로 TurtleBot3를 조작하면서 SLAM을 진행한다.
4. RViz에서 지도를 실시간으로 확인한다.
5. 주행하면서 “여기가 A로 좋다, 여기가 B로 좋다, 여기가 C로 좋다”를 눈으로 결정한다.
6. A·B·C의 물리적 위치를 바닥 테이프로 표시한다.
7. SLAM 중 좌표는 후보값으로 기록한다.
8. 지도가 충분히 완성되면 `factory_map.pgm`, `factory_map.yaml`로 저장한다.
9. 저장한 지도를 Nav2로 다시 불러온다.
10. A에서 RViz `2D Pose Estimate`로 AMCL 초기 위치를 설정한다.
11. A·B·C 테이프 위치에서 최종 `map` 좌표와 방향을 다시 측정한다.
12. 최종 A·B·C를 `waypoints.yaml`에 저장한다.
13. 이후 장애물을 배치한다.
14. RViz에서 먼저 B와 C를 단일 Navigation2 Goal로 각각 시험한다.
15. 라이다와 Costmap이 나중에 배치한 장애물을 감지하고 우회하는지 확인한다.
16. 경로가 완전히 막히지 않았는지, A·B·C가 장애물 Inflation 안에 들어가지 않았는지 확인한다.
17. 기본 Nav2 시험이 성공한 뒤 `nav_mission_node.py`를 `[B, C]` 시나리오에 맞게 수정한다.
18. A→B→C 자율주행을 시험한다.
19. C에서 미션 완료 및 정지를 확인한다.
20. 기존 제스처로 manual 모드로 바꾼다.
21. C에서 A까지 제스처로 수동복귀한다.
22. 모든 기능이 따로 성공한 뒤 마지막에 Launch 파일로 통합한다.

---

## 13. Jetson의 Codex가 바로 해야 할 일

### 우선 확인

1. 실제 작업공간 위치를 확인한다.
2. `20260811_ws`와 현재 Jetson 파일이 같은 버전인지 확인한다.
3. `git status`가 있으면 사용자 변경사항을 보존한다.
4. ROS 2 Humble과 TurtleBot3 환경을 source한다.
5. 설치된 패키지와 실제 토픽 이름을 확인한다.

확인 예시:

```bash
source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 pkg list | grep -E 'turtlebot3|slam_toolbox|nav2|twist_mux'
ros2 topic list
ros2 node list
```

### 현재 단계에서 하지 말 것

- A·B·C 실좌표가 없는데 임의 좌표로 `nav_mission_node.py` 완성 처리
- 새로운 제스처 패키지로 기존 시스템 교체
- 별도 Waypoint 노드 추가
- SLAM과 Nav2 내부 알고리즘을 커스텀 노드에 복사
- 모든 기능을 한 Launch에 먼저 합친 뒤 한꺼번에 디버깅
- 현재 사용하지 않는 갈림길·RealSense 기능 확장

### 다음 실제 작업

현재 우선순위는 다음과 같다.

```text
1. 2m × 2m 물리 환경 확정
2. bringup 및 컨트롤러 수동주행 확인
3. SLAM Toolbox로 최종 지도 제작
4. 저장 지도에서 AMCL + Nav2 단일 목표 검증
5. A·B·C 좌표 확정
6. nav_mission_node.py 수정
7. 제스처 수동복귀 통합
```

---

## 14. 실기 전 검증 체크리스트

### TurtleBot3 기반

- [ ] `/scan`이 정상 발행된다.
- [ ] `/odom`이 정상 발행된다.
- [ ] `map → odom → base_footprint` TF가 연결된다.
- [ ] `/cmd_vel`로 기본 주행이 된다.

### SLAM

- [ ] 2m × 2m 외벽이 끊기지 않고 그려진다.
- [ ] 벽 두께와 모서리가 과도하게 겹치지 않는다.
- [ ] 출발지로 돌아왔을 때 지도 정합이 맞는다.
- [ ] 지도 `.pgm`과 `.yaml`이 저장된다.

### AMCL·Nav2

- [ ] A에서 `2D Pose Estimate`가 잘 맞는다.
- [ ] 라이다 스캔이 저장 지도 벽과 겹친다.
- [ ] RViz 단일 목표 B 주행이 성공한다.
- [ ] RViz 단일 목표 C 주행이 성공한다.
- [ ] 나중에 놓은 장애물을 Costmap이 표시한다.
- [ ] 장애물을 피해 대체 경로를 만든다.

### 속도 명령 중재

- [ ] 제스처는 `/cmd_vel_manual`로 발행된다.
- [ ] Nav2는 `/cmd_vel_nav`로 발행된다.
- [ ] `twist_mux`만 최종 `/cmd_vel`을 발행한다.
- [ ] 수동 입력이 자율 입력보다 높은 우선순위를 가진다.
- [ ] 수동모드 전환 시 Nav2 목표가 취소된다.

### 미션

- [ ] 자동 경로는 B→C만 포함한다.
- [ ] A를 첫 Nav2 목표로 다시 보내지 않는다.
- [ ] B 성공 후 C를 한 번만 보낸다.
- [ ] C에서 `STATE_COMPLETED`로 유지된다.
- [ ] C 목표가 반복되지 않는다.
- [ ] 실패 시 무한 재시도하지 않는다.
- [ ] C에서 manual 전환 후 제스처 명령이 적용된다.
- [ ] C→A는 Nav2가 아니라 제스처로만 주행한다.

---

## 15. 최종 한 문장

이 프로젝트는 **SLAM Toolbox로 2m × 2m 공장 지도를 만들고, 같은 `map` 좌표계에서 A·B·C를 고정한 뒤, A에서 B와 C까지는 Nav2로 자율주행하고 C에서는 기존 제스처 시스템으로 수동모드에 전환하여 A까지 복귀하는 TurtleBot3 Burger 통합 프로젝트**다.

