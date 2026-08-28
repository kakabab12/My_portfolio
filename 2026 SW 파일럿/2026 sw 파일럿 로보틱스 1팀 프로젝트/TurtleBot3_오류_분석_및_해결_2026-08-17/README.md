# TurtleBot3 Jetson 이전 오류 분석 및 정상화 기록

- 작성일: 2026-08-17
- 대상: Jetson Orin Nano, Ubuntu 22.04, ROS 2 Humble
- 로봇: TurtleBot3 Burger
- 라이다: LDS-03
- 워크스페이스: `~/turtlebot3_ws`

## 1. 최종 상태

현재 다음 항목을 정상 확인했다.

- 통합 환경 스크립트 `~/turtlebot3_ws/install/setup.bash`가 오류 없이 로드됨
- TurtleBot3 OpenCR이 `/dev/ttyACM0`으로 연결됨
- TurtleBot3 본체 노드, 모터, 센서 및 오도메트리가 정상 시작됨
- LDS-03이 `/dev/ttyUSB0`으로 연결됨
- LDS-03에서 실제 `/scan` 거리 데이터가 수신됨
- TGZ-850M 조이스틱이 Xbox 360 Controller로 인식됨
- 조이스틱의 `/cmd_vel` 출력과 실제 수동 주행이 정상 동작함
- RViz2, SLAM Toolbox, Nav2 및 TurtleBot3 Navigation2가 설치되어 있음

최종 bringup 로그는 다음 파일에 보관했다.

```text
raw_logs/ros_runtime/final_bringup/launch.log
```

## 2. 장애 요약

한 가지 오류가 아니라 아래 문제가 순서대로 겹쳐 있었다.

1. 조이스틱 패키지의 잘못된 rosdep 의존성 선언
2. Conda Python과 시스템 ROS Python 환경 충돌
3. 과거 C++ 빌드가 강제 종료되어 남은 불완전한 `install` 폴더
4. 조이스틱 시험 당시 모터 토크 비활성화 및 여러 `/cmd_vel` 발행자 충돌 가능성
5. 실제 라이다는 LDS-03인데 환경이 LDS-02로 설정됨
6. LDS-03 기본 포트 별칭 `/dev/tb3_lidar`가 tty가 아닌 USB 장치를 가리킴

## 3. `ament_python` rosdep 오류

### 증상

```text
turtlebot3_joystick: Cannot locate rosdep definition for [ament_python]
```

### 조사 위치

```text
src/turtlebot3_joystick/package.xml
```

문제가 된 선언은 다음과 같았다.

```xml
<buildtool_depend>ament_python</buildtool_depend>
```

`ament_python`은 이 패키지에서 `<build_type>`으로 사용되지만 현재 Humble rosdep
데이터베이스에서는 해당 이름을 시스템 의존성 키로 해석하지 못했다.

### 수정

위 `buildtool_depend` 한 줄을 제거하고 빌드 형식 선언은 유지했다.

```xml
<export>
  <build_type>ament_python</build_type>
</export>
```

수정본은 다음 파일에 보관했다.

```text
config_snapshots/turtlebot3_joystick_package_FIXED.xml
```

수정 후 결과:

```text
#All required rosdeps installed successfully
Summary: 1 package finished
```

## 4. 조이스틱은 명령을 만들지만 로봇이 움직이지 않던 문제

조이스틱 드라이버는 처음부터 정상적으로 장치를 열었다.

```text
Opened joystick: Xbox 360 Controller
Linear axis x on 1 at scale -0.080000
Angular axis yaw on 0 at scale 0.600000
```

또한 `/cmd_vel`에서 실제 속도 명령을 확인했다.

```yaml
linear:
  x: -0.08
angular:
  z: -0.6
```

따라서 조이스틱 설정이나 OpenCR 펌웨어가 직접 원인은 아니었다. OpenCR의
`PUSH SW1` 자체 시험과 공식 키보드 teleop이 모두 성공해 펌웨어와 모터도
정상임을 확인했다.

시험 과정에서 모터 전원을 다음 서비스로 활성화했다.

```bash
ros2 service call /motor_power std_srvs/srv/SetBool "{data: true}"
```

응답:

```text
SetBool_Response(success=True, message='Succeeded to write data')
```

또한 키보드 teleop과 조이스틱 teleop을 동시에 실행하면 두 노드가 같은
`/cmd_vel`을 발행한다. 조이스틱 노드는 스틱을 놓은 동안에도 20 Hz로 0 속도를
발행하므로 다른 주행 명령을 빠르게 덮어쓸 수 있다. 최종적으로 bringup과 한 개의
teleop 노드만 실행해 정상 주행을 확인했다.

## 5. 통합 `install/setup.bash`가 깨진 원인

### 증상

```text
not found: ".../install/dynamixel_sdk_examples/share/dynamixel_sdk_examples/local_setup.bash"
not found: ".../install/turtlebot3_node/share/turtlebot3_node/local_setup.bash"
```

### 원본 로그 조사 결과

2026-08-08 빌드에서 먼저 Python 환경 오류가 발생했다.

```text
ModuleNotFoundError: No module named 'catkin_pkg'
```

원본:

```text
raw_logs/build_failures/catkin_pkg/stderr.log
```

이후 다음 패키지들의 C++ 컴파일러 프로세스가 강제로 종료됐다.

```text
c++: fatal error: Killed signal terminated program cc1plus
```

관련 원본 로그:

```text
raw_logs/build_failures/oom_dynamixel_sdk_examples/stderr.log
raw_logs/build_failures/oom_turtlebot3_node/stderr.log
raw_logs/build_failures/oom_coin_d4_driver/stderr.log
```

`cc1plus`의 `Killed`는 외부 SIGKILL을 뜻한다. Jetson의 제한된 메모리에서 여러
빌드가 반복 실패한 정황상 메모리 부족이 가장 유력하다. 당시 커널 OOM 기록은
남아 있지 않아 커널 로그로 확정한 것은 아니지만, 단일 작업 재빌드에서는 같은
소스가 모두 성공했다.

실패한 빌드는 환경 스크립트 일부만 `install`에 만든 뒤 실제 컴파일과 설치를
완료하지 못했다. 그 결과 통합 `setup.bash`가 존재하지 않는 `local_setup.bash`를
참조하는 부분 설치 상태가 됐다.

### 복구 방법

Conda 환경을 사용하지 않고 컴파일 병렬도를 1로 제한해 실패 패키지를 하나씩
재빌드했다.

```bash
source /opt/ros/humble/setup.bash
export CMAKE_BUILD_PARALLEL_LEVEL=1
export MAKEFLAGS=-j1
colcon build --packages-select dynamixel_sdk_examples \
  --symlink-install --executor sequential --parallel-workers 1
```

```bash
source /opt/ros/humble/setup.bash
export CMAKE_BUILD_PARALLEL_LEVEL=1
export MAKEFLAGS=-j1
colcon build --packages-select turtlebot3_node \
  --symlink-install --executor sequential --parallel-workers 1
```

`coin_d4_driver`도 같은 방식으로 다시 빌드했다.

복구 결과:

```text
dynamixel_sdk_examples  colcon_build.rc = 0
turtlebot3_node         colcon_build.rc = 0
coin_d4_driver          colcon_build.rc = 0
```

성공 원본 로그:

```text
raw_logs/build_success/dynamixel_sdk_examples/
raw_logs/build_success/turtlebot3_node/
raw_logs/build_success/coin_d4_driver/
```

## 6. 라이다가 `/scan`을 발행하지 않던 원인

### 최초 증상

환경 설정이 다음과 같았다.

```bash
export LDS_MODEL=LDS-02
```

bringup은 LDS-02 드라이버인 `ld08_driver`를 실행했고 다음 로그를 냈다.

```text
Can't find LDS-02
```

실제 장치는 LDS-03이므로 드라이버 선택 자체가 잘못돼 있었다.

### 장치 및 드라이버 확인

USB 조사 결과:

```text
/dev/ttyACM0  OpenCR Virtual ComPort
/dev/ttyUSB0  CP2102 USB to UART Bridge Controller
```

LDS-03용 `coin_d4_driver`는 이미 설치되어 있었다. 이 드라이버를
`/dev/ttyUSB0`으로 직접 실행하자 실제 LaserScan 프레임을 수신했다.

```text
version M1CT_TOF
Lidar port: /dev/ttyUSB0
Activated lidar grab thread for port /dev/ttyUSB0
TOF version lidar start for /dev/ttyUSB0
```

실제 수신 예:

```yaml
frame_id: base_scan
angle_min: 0.0
angle_max: 6.2831854820251465
scan_time: 0.09929700195789337
range_min: 0.10000000149011612
range_max: 100.0
ranges:
  - 9.767000198364258
  - 9.809000015258789
  - 9.859999656677246
```

관련 런타임 로그:

```text
raw_logs/ros_runtime/lds03_direct/launch.log
raw_logs/ros_runtime/verification_transcript.txt
```

### 포트 별칭 추가 문제

LDS-03 기본 설정은 다음 경로를 사용했다.

```yaml
port: "/dev/tb3_lidar"
```

그러나 당시 시스템에서 이 별칭은 tty 포트가 아니라 USB 장치 노드를 가리켰다.

```text
/dev/tb3_lidar -> bus/usb/001/021
```

드라이버 오류:

```text
Failed to open lidar port
Lidar port is wrong
```

실제 정상 통신을 확인한 고정 포트로 워크스페이스 설정을 수정했다.

```yaml
port: "/dev/ttyUSB0"
```

수정 파일:

```text
src/coin_d4_driver/params/single_lidar_node.yaml
config_snapshots/coin_d4_driver_single_lidar_node_FIXED.yaml
```

마지막으로 `.bashrc`의 모델 설정을 영구 수정했다.

```bash
export TURTLEBOT3_MODEL=burger
export LDS_MODEL=LDS-03
```

## 7. 최종 정상 실행 방법

### TurtleBot3 본체 및 LDS-03

```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
ros2 launch turtlebot3_bringup robot.launch.py
```

### 조이스틱

다른 터미널에서 실행한다.

```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
ros2 launch turtlebot3_joystick teleop.launch.py
```

키보드 teleop, 조이스틱 teleop, Nav2처럼 `/cmd_vel`을 발행하는 노드는 동시에
사용하지 않는다. 필요하면 `twist_mux` 같은 명령 다중화 계층을 별도로 구성한다.

## 8. 향후 전체 빌드 권장 명령

Jetson에서 다시 메모리 부족이 발생하지 않도록 다음처럼 단일 작업으로 빌드한다.

```bash
conda deactivate
source /opt/ros/humble/setup.bash
cd ~/turtlebot3_ws
CMAKE_BUILD_PARALLEL_LEVEL=1 MAKEFLAGS=-j1 \
colcon build --symlink-install --executor sequential --parallel-workers 1
```

빌드 후 다음 검사를 권장한다.

```bash
source ~/turtlebot3_ws/install/setup.bash
ros2 pkg prefix turtlebot3_node
ros2 pkg prefix coin_d4_driver
ros2 pkg prefix turtlebot3_joystick
```

## 9. 보관 파일 구조

```text
project_diagnostics_2026-08-17/
├── README.md
├── config_snapshots/
│   ├── coin_d4_driver_single_lidar_node_FIXED.yaml
│   ├── teleop.launch.py
│   ├── tgz_850m.yaml
│   └── turtlebot3_joystick_package_FIXED.xml
└── raw_logs/
    ├── build_failures/
    │   ├── catkin_pkg/
    │   ├── oom_coin_d4_driver/
    │   ├── oom_dynamixel_sdk_examples/
    │   └── oom_turtlebot3_node/
    ├── build_success/
    │   ├── coin_d4_driver/
    │   ├── dynamixel_sdk_examples/
    │   └── turtlebot3_node/
    └── ros_runtime/
        ├── final_bringup/
        ├── lds03_direct/
        └── verification_transcript.txt
```
