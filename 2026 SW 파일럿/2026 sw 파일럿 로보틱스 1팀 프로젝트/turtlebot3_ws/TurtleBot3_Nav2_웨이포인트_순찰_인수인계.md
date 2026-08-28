# TurtleBot3 Nav2 웨이포인트 순찰 인수인계

작성일: 2026-08-17

## 다음 작업을 시작할 때

Codex에게 다음과 같이 요청한다.

> `~/turtlebot3_ws/TurtleBot3_Nav2_웨이포인트_순찰_인수인계.md`를 읽고 TurtleBot3 Nav2 웨이포인트 순찰 작업을 이어서 진행해 줘.

## 시스템 구성

- ROS 2: Humble
- 로봇: TurtleBot3 Burger
- 라이다: LDS-03
- 조이스틱: TGZ-850M (Linux에서 Xbox 360 Controller로 인식)
- 워크스페이스: `~/turtlebot3_ws`
- 사용 지도: `~/turtlebot3_ws/maps/square_2m.yaml`
- 지도 이미지: `~/turtlebot3_ws/maps/square_2m.pgm`

새 터미널마다 다음 환경을 불러와야 한다.

```bash
conda deactivate 2>/dev/null || true
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
```

새 패키지가 없다고 나오면 해당 터미널에서 반드시 `install/setup.bash`를 다시 source한다.

## 지금까지 완료한 작업

1. TurtleBot3 본체, OpenCR, LDS-03 bringup 동작 확인
2. TGZ-850M 조이스틱 런치 생성 및 동작 확인
3. SLAM Toolbox와 사전 설정된 RViz2 런치 생성
4. SLAM으로 `square_2m` 지도 생성 및 저장
5. 저장한 지도를 Nav2와 RViz2에서 불러와 수동 Nav2 Goal 주행 성공
6. 실제 로봇을 조이스틱으로 이동하며 A/B/C/D의 `map` 좌표 측정
7. A를 초기 위치로 자동 설정하고 `A → B → C → D → A`를 한 바퀴 순찰하는 패키지 생성
8. `turtlebot3_waypoint_patrol` 패키지 빌드 및 ROS 2 패키지 인식 검사 완료

## 주요 패키지와 파일

### 조이스틱

- 런치: `src/turtlebot3_joystick/launch/teleop.launch.py`
- 설정: `src/turtlebot3_joystick/config/tgz_850m.yaml`

조이스틱 설정:

- 왼쪽 스틱 위/아래: 전진/후진
- 왼쪽 스틱 좌/우: 회전
- enable 버튼 불필요
- 선속도 스케일: `0.08 m/s`
- 각속도 스케일: `0.6 rad/s`

### SLAM Toolbox

- 런치: `src/turtlebot3_slam_toolbox/launch/slam.launch.py`
- 파라미터: `src/turtlebot3_slam_toolbox/config/mapper_params.yaml`
- RViz 설정: `src/turtlebot3_slam_toolbox/rviz/mapping.rviz`

SLAM 런치는 SLAM Toolbox와 설정된 RViz2를 함께 실행한다. RViz Fixed Frame은 `map`, 지도 토픽은 `/map`, 라이다 토픽은 `/scan`이다.

### Nav2 웨이포인트 순찰

- 패키지: `src/turtlebot3_waypoint_patrol`
- 순찰 노드: `src/turtlebot3_waypoint_patrol/turtlebot3_waypoint_patrol/patrol_node.py`
- 런치: `src/turtlebot3_waypoint_patrol/launch/patrol.launch.py`
- 설치 위치: `install/turtlebot3_waypoint_patrol`

이 패키지는 Nav2 공식 `nav2_simple_commander.BasicNavigator`와 `FollowWaypoints`를 사용한다.

동작:

1. A를 AMCL 초기 위치로 자동 발행
2. Nav2와 AMCL 활성화 대기
3. `A → B → C → D → A` 순서로 한 바퀴 주행
4. A 복귀 후 순찰 노드 종료

## 웨이포인트 좌표

모든 좌표는 `square_2m` 지도의 `map` 프레임 기준이며, yaw 단위는 radian이다.

```yaml
A:
  x: 0.088
  y: -0.071
  yaw: -0.210

B:
  x: 1.237
  y: -0.431
  yaw: -0.665

C:
  x: 0.810
  y: -1.489
  yaw: 3.140

D:
  x: -0.018
  y: -1.072
  yaw: 2.125
```

순찰 경로는 다음과 같다.

```text
A → B → C → D → A → 종료
```

## 자율 순찰 실행 순서

### 실행 전 물리적 조건

- 로봇을 실제 A 표시 위치에 놓는다.
- 좌표를 측정했을 때와 동일한 방향(`yaw=-0.210`)을 바라보게 한다.
- 조이스틱 노드는 실행하지 않는다.
- 주행 경로에 사람이나 위험한 장애물이 없는지 확인한다.

### 터미널 1: TurtleBot3 본체와 라이다

```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger
export LDS_MODEL=LDS-03

ros2 launch turtlebot3_bringup robot.launch.py
```

### 터미널 2: Nav2와 RViz2

```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger

ros2 launch turtlebot3_navigation2 navigation2.launch.py \
  map:=$HOME/turtlebot3_ws/maps/square_2m.yaml
```

RViz에 지도와 라이다가 나타날 때까지 기다린다. 자동 순찰 시에는 RViz의 `2D Pose Estimate`와 `Nav2 Goal`을 누르지 않는다.

### 터미널 4: 한 바퀴 자동 순찰

```bash
conda deactivate 2>/dev/null || true
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash

ros2 launch turtlebot3_waypoint_patrol patrol.launch.py
```

## 위치 좌표 확인 방법

Nav2/AMCL로 위치 추정이 된 상태에서 다음 명령을 사용한다.

```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
ros2 run tf2_ros tf2_echo map base_footprint
```

`Translation`의 첫 번째와 두 번째 값이 x/y이고, `Rotation: in RPY`의 세 번째 값이 yaw이다.

AMCL 메시지를 직접 확인하려면 다음 명령을 사용한다.

```bash
ros2 topic echo /amcl_pose
```

## 조이스틱으로 좌표를 다시 측정할 때

Nav2는 실행하되 Goal을 보내지 않은 상태에서 조이스틱을 실행할 수 있다.

```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
ros2 launch turtlebot3_joystick teleop.launch.py
```

각 지점으로 이동해 완전히 정지하고 2~3초 기다린 다음 `tf2_echo map base_footprint` 값을 기록한다. 자율 순찰을 시작하기 전에는 조이스틱 런치를 `Ctrl+C`로 종료해야 한다. `/cmd_vel` 충돌을 피하기 위해 조이스틱과 자동 순찰을 동시에 실행하지 않는다.

## 지도 저장 명령

SLAM 실행 중 새 지도를 저장할 때:

```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
ros2 run nav2_map_server map_saver_cli \
  -f ~/turtlebot3_ws/maps/새_지도_이름
```

YAML과 PGM 파일 두 개가 생성된다.

## 순찰 패키지 재빌드

순찰 노드 또는 런치 파일을 수정한 경우:

```bash
cd ~/turtlebot3_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select turtlebot3_waypoint_patrol
source ~/turtlebot3_ws/install/setup.bash
```

패키지 인식 확인:

```bash
ros2 pkg prefix turtlebot3_waypoint_patrol
```

정상 출력:

```text
/home/user/turtlebot3_ws/install/turtlebot3_waypoint_patrol
```

## 알려진 문제와 대처

### Package not found

오래 열어둔 터미널은 새 빌드를 모른다. 그 터미널에서 다음을 다시 실행한다.

```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
```

### 프롬프트에 `(base)`가 표시됨

Conda 환경 때문에 ROS Python 패키지 충돌이 발생할 수 있으므로 다음을 실행한다.

```bash
conda deactivate
```

그 다음 ROS 환경을 다시 source한다.

### RViz의 라이다 점과 지도 벽이 크게 어긋남

자동 순찰을 시작하지 말고 다음을 확인한다.

- 실제 로봇이 A 표시 위치에 정확히 놓였는가
- 로봇 방향이 A를 측정했을 때와 같은가
- 올바른 `square_2m.yaml` 지도를 불러왔는가

필요하면 수동 `2D Pose Estimate`로 위치를 맞춘 뒤 좌표 또는 실제 A 표시를 다시 교정한다.

### 모터가 켜지지 않음

```bash
source /opt/ros/humble/setup.bash
ros2 service call /motor_power std_srvs/srv/SetBool "{data: true}"
```

## 다음 고도화 후보

- 한 바퀴가 아닌 무한 반복 순찰
- 순찰 횟수 파라미터화
- 각 웨이포인트에서 대기 시간 설정
- 좌표를 Python 코드가 아닌 YAML 파일로 분리
- 장애물 때문에 특정 지점 실패 시 재시도 정책
- 조이스틱과 Nav2 사이에 `twist_mux`를 넣어 안전한 수동 개입 지원
- 긴급 정지와 순찰 재개 기능
- RViz Marker로 A/B/C/D 이름과 위치 표시
