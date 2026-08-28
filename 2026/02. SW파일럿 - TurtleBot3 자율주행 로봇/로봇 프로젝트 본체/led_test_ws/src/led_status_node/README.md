# Jetson Orin Nano ROS 2 LED test

물리 헤더 핀(BOARD) 31번은 초록색, 33번은 빨간색으로 사용합니다. 각 LED에는 적절한 직렬 저항을 사용하고 Jetson과 모듈의 GND를 공통으로 연결하세요. GPIO에는 5 V를 인가하지 마세요.

## 빌드

```bash
cd ~/led_test_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## 배선 없이 먼저 확인

```bash
ros2 run led_status_node led_status --ros-args -p mock:=true
```

## 실제 LED 실행

```bash
ros2 launch led_status_node led_test.launch.py
```

다른 터미널에서 상태를 한 번씩 발행합니다.

```bash
source /opt/ros/humble/setup.bash
source ~/led_test_ws/install/setup.bash
ros2 topic pub --once /led_status std_msgs/msg/String "{data: idle}"
ros2 topic pub --once /led_status std_msgs/msg/String "{data: normal}"
ros2 topic pub --once /led_status std_msgs/msg/String "{data: anomaly}"
ros2 topic pub --once /led_status std_msgs/msg/String "{data: off}"
```

- `idle`: 초록색과 빨간색 모두 켜짐
- `normal`: 초록색만 점멸
- `anomaly`: 빨간색만 점멸
- `off`: 모두 꺼짐

LED 모듈이 LOW 입력에서 켜지는 active-low 방식이면 실행 인자에 `-p active_high:=false`를 사용하세요.
