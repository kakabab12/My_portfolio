# TurtleBot3 기본 실행 명령

현재 영구 설정은 `burger`, `LDS-03`이다. 새 터미널마다 아래 두 줄로 ROS 환경을 불러온다.

```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
```

## 본체 실행 — 터미널 1

```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
ros2 launch turtlebot3_bringup robot.launch.py
```

## 조이스틱 실행 — 터미널 2

```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
ros2 launch turtlebot3_joystick teleop.launch.py
```

## 모터가 움직이지 않을 때 — 터미널 3

```bash
source /opt/ros/humble/setup.bash
ros2 service call /motor_power std_srvs/srv/SetBool "{data: true}"
```

## 종료

실행 중인 각 터미널에서 `Ctrl+C`를 누른다.

> 키보드 조종, 조이스틱 조종, Nav2를 동시에 실행하지 않는다. 여러 노드가 `/cmd_vel`을 서로 덮어쓸 수 있다.

