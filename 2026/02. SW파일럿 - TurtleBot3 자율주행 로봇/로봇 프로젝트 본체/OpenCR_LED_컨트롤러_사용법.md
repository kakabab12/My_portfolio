# OpenCR LED 컨트롤러 사용 방법

## 1. 구성 개요

Jetson Orin Nano의 ROS 2 노드가 `/opencr_led_status` 토픽을 구독하는
`turtlebot3_node`로 명령을 전달하고, OpenCR가 GPIO 50과 GPIO 51에 연결된
LED를 제어합니다.

데이터 흐름:

```text
다른 ROS 2 노드 또는 터미널
    → /opencr_led_status (std_msgs/msg/UInt8)
    → turtlebot3_node
    → USB / Dynamixel 제어 테이블 주소 51
    → OpenCR GPIO 50, 51
```

## 2. LED 연결

OpenCR GPIO 확장 커넥터를 기준으로 연결합니다.

| OpenCR 연결 | 용도 |
|---|---|
| 물리 핀 3 / GPIO 50 | 첫 번째 LED 제어 |
| 물리 핀 4 / GPIO 51 | 두 번째 LED 제어 |
| GND | 두 LED의 공통 접지 |

- 각 LED에 직렬 저항을 사용합니다.
- 배선 변경 전에는 Jetson과 OpenCR 전원을 끕니다.
- LED의 `+`를 GPIO에, `-`를 GND에 연결합니다.

## 3. LED 모드

| 값 | 동작 |
|---:|---|
| 0 | 두 LED 모두 끄기 |
| 1 | 두 LED 모두 켜기 |
| 2 | GPIO 50 LED만 점멸 |
| 3 | GPIO 51 LED만 점멸 |
| 4 | GPIO 50 LED만 켜기 |
| 5 | GPIO 51 LED만 켜기 |
| 6 | 두 LED 모두 점멸 |

점멸 주기는 약 0.5초입니다. 범위를 벗어난 값은 Orin 노드에서 무시됩니다.

## 4. TurtleBot3 노드 실행

새 터미널에서는 ROS 2와 로컬 TurtleBot3 workspace가 `~/.bashrc`를 통해
자동으로 적용됩니다. 수동으로 적용해야 할 때는 다음 명령을 사용합니다.

```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
```

일반적인 TurtleBot3 bringup을 실행합니다.

```bash
ros2 launch turtlebot3_bringup robot.launch.py
```

LiDAR 없이 TurtleBot3 노드만 시험할 때는 다음과 같이 실행할 수 있습니다.

```bash
ros2 run turtlebot3_node turtlebot3_ros -i /dev/ttyACM0 \
  --ros-args \
  --params-file ~/turtlebot3_ws/install/turtlebot3_node/share/turtlebot3_node/param/burger.yaml
```

OpenCR 연결 확인:

```bash
ls -l /dev/ttyACM*
```

## 5. 터미널에서 제어

두 LED 모두 켜기:

```bash
ros2 topic pub --once /opencr_led_status std_msgs/msg/UInt8 "{data: 1}"
```

GPIO 50 LED만 점멸:

```bash
ros2 topic pub --once /opencr_led_status std_msgs/msg/UInt8 "{data: 2}"
```

GPIO 51 LED만 켜기:

```bash
ros2 topic pub --once /opencr_led_status std_msgs/msg/UInt8 "{data: 5}"
```

두 LED 모두 끄기:

```bash
ros2 topic pub --once /opencr_led_status std_msgs/msg/UInt8 "{data: 0}"
```

토픽 확인:

```bash
ros2 topic info /opencr_led_status
```

## 6. 다른 Python ROS 2 노드와 연동

이벤트가 발생할 때 셸 명령을 실행하는 것보다, 노드 안에서 publisher를
만들어 `UInt8` 메시지를 직접 발행하는 방식을 권장합니다.

```python
from std_msgs.msg import UInt8

# __init__ 안에서 한 번 생성
self.opencr_led_pub = self.create_publisher(
    UInt8,
    '/opencr_led_status',
    10,
)

def set_opencr_led(self, mode: int) -> None:
    if not 0 <= mode <= 6:
        self.get_logger().warning(f'Invalid OpenCR LED mode: {mode}')
        return

    msg = UInt8()
    msg.data = mode
    self.opencr_led_pub.publish(msg)

# 이벤트 처리 예시
def on_warning_detected(self) -> None:
    self.set_opencr_led(3)  # GPIO 51 LED 점멸

def on_normal_state(self) -> None:
    self.set_opencr_led(4)  # GPIO 50 LED 점등

def on_system_stopped(self) -> None:
    self.set_opencr_led(0)  # 모두 소등
```

publisher는 노드가 시작될 때 한 번 생성하고, 이벤트마다 `publish()`만 호출합니다.

## 7. 재빌드

Orin의 `turtlebot3_node` 소스를 수정한 경우:

```bash
cd ~/turtlebot3_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select turtlebot3_node --symlink-install
source install/setup.bash
```

빌드 후 실행 중인 기존 TurtleBot3 노드를 종료하고 다시 시작해야 변경 사항이
적용됩니다.

## 8. 문제 해결

### `/opencr_led_status` 토픽에 subscriber가 없을 때

```bash
ros2 topic info /opencr_led_status
ros2 pkg prefix turtlebot3_node
```

패키지 경로는 다음과 같이 로컬 workspace를 가리켜야 합니다.

```text
/home/user/turtlebot3_ws/install/turtlebot3_node
```

### `/dev/ttyACM0`가 없을 때

1. OpenCR USB 연결과 전원을 확인합니다.
2. OpenCR의 `RESET` 버튼을 짧게 한 번 누릅니다.
3. 다시 확인합니다.

```bash
ls -l /dev/ttyACM*
```

### 명령은 성공하지만 LED가 바뀌지 않을 때

- 수정된 `turtlebot3_node`가 실행 중인지 확인합니다.
- 터미널의 `ROS_DOMAIN_ID`가 다른 노드와 같은지 확인합니다. 현재 기본값은 `30`입니다.
- OpenCR GPIO 물리 핀 3, 4와 GND 배선을 확인합니다.
- 노드 로그에 아래와 같은 메시지가 출력되는지 확인합니다.

```text
OpenCR LED mode: 1, sdk_msg: Succeeded to write data
```

## 9. 현재 설치 정보

- OpenCR 펌웨어: `burger_led_control V260818R3`
- OpenCR 제어 테이블 주소: `51` (`uint8` LED 모드)
- ROS 2 토픽: `/opencr_led_status`
- 메시지 형식: `std_msgs/msg/UInt8`
- Orin workspace: `~/turtlebot3_ws`
- TurtleBot3 모델: `burger`
- ROS 2 배포판: `Humble`
