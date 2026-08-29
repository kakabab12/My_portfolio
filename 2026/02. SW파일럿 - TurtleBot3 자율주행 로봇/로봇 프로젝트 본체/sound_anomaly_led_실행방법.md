# 소리 이상감지 및 OpenCR LED 실행 방법

이 문서는 기어박스 소리 이상감지와 OpenCR 외부 LED(GPIO 50/51)를 함께 실행하는 방법을 정리합니다. 시작과 종료는 같은 폴더의 `sound_anomaly_led.sh`를 사용합니다.

## 동작 흐름

```text
USB Microphone
  -> sound_anomaly_node
  -> /opencr_led_status (std_msgs/msg/UInt8)
  -> turtlebot3_node
  -> OpenCR GPIO 50/51 LED
```

| 판정 | LED 모드 | 표시 |
| --- | ---: | --- |
| `IDLE` | 1 | 두 LED 켜짐 |
| `NORMAL` | 2 | GPIO 50 녹색 LED 점멸 |
| `ABNORMAL` | 3 | GPIO 51 빨간 LED 점멸 |
| 종료 | 0 | 두 LED 끔 |

## 시작 전 확인

OpenCR를 USB와 전원에 연결하고, USB 마이크를 연결합니다. 다음 두 장치가 보여야 합니다.

```bash
ls -l /dev/ttyACM0
cd /home/user/ros2_ws/src/dyeun_robotics/sound_anomaly
PYTHONPATH=/home/user/ros2_ws/python_deps python3 scripts/list_audio_devices.py
```

현재 설정은 마이크의 변하는 ALSA 번호 대신 `USB Microphone: Audio` 이름으로 선택합니다. OpenCR는 `/dev/ttyACM0`을 사용합니다.

처음 설치하거나 소스를 변경한 뒤에는 빌드합니다.

```bash
cd /home/user/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select sound_anomaly --symlink-install
```

추론용 파이썬 라이브러리가 없다면 다음을 한 번 실행합니다.

```bash
python3 -m pip install --target /home/user/ros2_ws/python_deps \
  -r /home/user/ros2_ws/src/dyeun_robotics/sound_anomaly/requirements-inference.txt
touch /home/user/ros2_ws/python_deps/COLCON_IGNORE
```

## 실행

터미널에서 다음을 실행합니다.

```bash
cd /home/user/Desktop
./sound_anomaly_led.sh start
```

스크립트는 먼저 `turtlebot3_node`를 시작해 OpenCR LED 구독자를 준비하고, 그다음 `sound_anomaly` launch를 시작합니다. `ROS_DOMAIN_ID`가 설정돼 있지 않으면 두 노드 모두 기본값 `30`을 사용합니다.

실행 상태와 토픽 연결은 다음으로 확인합니다.

```bash
./sound_anomaly_led.sh status
```

정상 연결의 핵심은 `/opencr_led_status`에 publisher 1개(`sound_anomaly_node`)와 subscriber 1개(`turtlebot3_node`)가 표시되는 것입니다.

로그 파일은 아래에 저장됩니다.

```text
/home/user/.ros/sound_anomaly_led/sound_anomaly.log
/home/user/.ros/sound_anomaly_led/turtlebot3_node.log
```

## 안전 종료

다음 명령으로 종료합니다.

```bash
cd /home/user/Desktop
./sound_anomaly_led.sh stop
```

종료 스크립트는 이상감지 노드를 먼저 내리고, OpenCR가 살아 있는 동안 LED 소등 모드 `0`을 한 번 더 발행한 후 LED 제어 노드를 종료합니다.

## 문제 해결

- `OpenCR was not found`: `/dev/ttyACM0`가 없으면 OpenCR USB 연결·전원·RESET 버튼을 확인합니다.
- LED가 반응하지 않음: `./sound_anomaly_led.sh status`에서 subscriber가 1개인지, OpenCR GPIO 50/51 및 GND 배선이 맞는지 확인합니다.
- 이상감지 노드가 시작되지 않음: `sound_anomaly.log`에서 마이크 이름을 확인하고, `list_audio_devices.py` 결과에 `USB Microphone: Audio`가 있는지 확인합니다.
- `input overflow` 경고가 반복됨: Jetson의 GUI·원격 데스크톱 등 CPU 부하를 줄이고 실제 현장 음성으로 판정 결과를 재검증합니다.
