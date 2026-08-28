# TurtleBot3 TGZ-850M 조이스틱 조종 방법

## 준비

1. OpenCR 전원을 켠다.
2. Jetson에 OpenCR, LDS-03, TGZ-850M 조이스틱을 연결한다.
3. 처음 시험할 때는 안전을 위해 바퀴를 바닥에서 띄운다.

현재 설정:

- 로봇 모델: TurtleBot3 Burger
- 라이다: LDS-03
- 조이스틱 인식 이름: Xbox 360 Controller
- 왼쪽 스틱 위·아래: 전진·후진
- 왼쪽 스틱 좌·우: 좌회전·우회전
- LB 버튼: 누를 필요 없음

## 터미널 1 — TurtleBot3 본체 실행

```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
ros2 launch turtlebot3_bringup robot.launch.py
```

다음과 비슷한 로그가 나오면 OpenCR과 LDS-03이 정상이다.

```text
Succeeded to open the port(/dev/ttyACM0)!
Lidar port: /dev/ttyUSB0
TOF version lidar start for /dev/ttyUSB0
Run!
```

터미널 1은 종료하지 않고 그대로 둔다.

## 터미널 2 — 조이스틱 실행

```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
ros2 launch turtlebot3_joystick teleop.launch.py
```

다음 로그가 나오면 조이스틱 인식이 정상이다.

```text
Opened joystick: Xbox 360 Controller
Linear axis x on 1 at scale -0.080000
Angular axis yaw on 0 at scale 0.600000
```

이제 왼쪽 스틱으로 조종한다.

## 스틱 입력은 되지만 바퀴가 움직이지 않을 때

터미널 1과 2를 실행한 상태에서 터미널 3을 열고 다음 명령을 실행한다.

```bash
source /opt/ros/humble/setup.bash
ros2 service call /motor_power std_srvs/srv/SetBool "{data: true}"
```

정상 응답:

```text
success=True
message='Succeeded to write data'
```

## 종료

터미널 2에서 `Ctrl+C`를 눌러 조이스틱을 먼저 종료하고, 터미널 1에서도 `Ctrl+C`를 누른다.

## 주의사항

- 키보드 teleop과 조이스틱 teleop을 동시에 실행하지 않는다.
- Nav2 실행 중에는 조이스틱을 함께 실행하지 않는다.
- 여러 노드가 `/cmd_vel`을 동시에 발행하면 주행 명령이 서로 덮어써질 수 있다.
- 전체 워크스페이스 환경은 `install/setup.bash`로 불러온다. 예전처럼 조이스틱의 긴 `package.bash` 경로를 직접 입력할 필요가 없다.
- 프롬프트 앞에 `(base)`가 보이고 ROS Python 오류가 발생하면 `conda deactivate` 후 다시 실행한다.
