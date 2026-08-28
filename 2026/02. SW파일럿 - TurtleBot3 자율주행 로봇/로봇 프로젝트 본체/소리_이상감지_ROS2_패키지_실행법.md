# 소리 이상감지 ROS 2 패키지 실행 방법

## 1. 구성 개요

기어박스 소리를 USB 마이크로 계속 수집하여 정상, 이상 또는 무감지 상태로
판정하고 `/opencr_led_status` 토픽을 통해 OpenCR LED를 제어한다.

| 감지 상태 | OpenCR 모드 | LED 동작 |
|---|---:|---|
| `NORMAL` | 2 | GPIO 50 초록 LED만 점멸 |
| `ABNORMAL` | 3 | GPIO 51 빨간 LED만 점멸 |
| `IDLE` | 1 | 초록·빨간 LED 모두 계속 켜짐 |

`IDLE`은 입력 소리가 무음 기준보다 작거나, 아직 판정할 오디오가 충분하지
않거나, 추론 오류가 발생한 경우에 사용한다.

## 2. 현재 설치 정보

- ROS 2 배포판: Humble
- 이상감지 패키지: `sound_anomaly`
- ROS 2 workspace: `~/ros2_ws`
- 패키지 소스: `~/ros2_ws/src/dyeun_robotics/sound_anomaly`
- TurtleBot3 workspace: `~/turtlebot3_ws`
- USB 마이크: `USB Microphone: Audio (hw:2,0)`
- 마이크 샘플레이트: 44,100 Hz
- 입력 채널: 모노 1채널
- OpenCR LED 토픽: `/opencr_led_status`

## 3. TurtleBot3 및 OpenCR 실행

LED 명령을 OpenCR로 전달하려면 `turtlebot3_node`가 실행 중이어야 한다.
첫 번째 터미널에서 다음 명령을 실행한다.

```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
ros2 launch turtlebot3_bringup robot.launch.py
```

LiDAR 없이 TurtleBot3 노드만 시험할 때는 다음 명령을 사용할 수 있다.

```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
ros2 run turtlebot3_node turtlebot3_ros -i /dev/ttyACM0 \
  --ros-args \
  --params-file ~/turtlebot3_ws/install/turtlebot3_node/share/turtlebot3_node/param/burger.yaml
```

## 4. 소리 이상감지 패키지 실행

두 번째 터미널에서 다음 명령을 실행한다.

```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch sound_anomaly sound_anomaly.launch.py
```

launch 파일은 이상감지 노드가 예기치 않게 종료되면 2초 뒤 다시 실행한다.
터미널을 닫거나 `Ctrl+C`를 누르면 launch도 종료된다.

## 5. 감지 결과 확인

새 터미널에서 ROS 2 환경을 적용한다.

```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
source ~/ros2_ws/install/setup.bash
```

현재 감지 상태 확인:

```bash
ros2 topic echo /sound_anomaly_node/state
```

이상 확률 확인:

```bash
ros2 topic echo /sound_anomaly_node/anomaly_probability
```

OpenCR로 전달되는 LED 모드 확인:

```bash
ros2 topic echo /opencr_led_status
```

노드 실행 여부 확인:

```bash
ros2 node list | grep sound_anomaly
```

## 6. 설정 변경

설정 파일 위치:

```text
~/ros2_ws/src/dyeun_robotics/sound_anomaly/config/sound_anomaly.yaml
```

주요 설정:

| 설정 | 현재 값 | 설명 |
|---|---|---|
| `audio_device` | `USB Microphone: Audio (hw:2,0)` | 사용할 USB 입력 마이크 |
| `capture_sample_rate` | `44100` | 마이크 입력 샘플레이트 |
| `prediction_interval` | `1.0` | 판정 간격(초) |
| `anomaly_threshold` | `-1.0` | 모델에 저장된 임계값 사용 |
| `silence_rms_threshold` | `0.005` | 이 값보다 작은 입력은 `IDLE` 처리 |
| `green_blink_mode` | `2` | GPIO 50 초록 LED 점멸 |
| `red_blink_mode` | `3` | GPIO 51 빨간 LED 점멸 |
| `idle_led_mode` | `1` | 두 LED 모두 상시 점등 |
| `shutdown_led_mode` | `0` | 이상감지 노드 종료 시 두 LED 모두 끄기 |

설정을 변경한 후에는 실행 중인 launch를 `Ctrl+C`로 종료하고 다시 실행한다.
`--symlink-install`로 빌드되어 있으므로 YAML 값만 바꾼 경우 일반적으로 재빌드할
필요가 없다.

전체 시스템을 종료할 때는 이상감지 노드를 먼저 종료해야 모드 `0`이 실행 중인
TurtleBot3 노드를 거쳐 OpenCR에 전달된다. 그 다음 Nav2, 순찰 노드와 TurtleBot3
bringup을 종료한다.

Python 코드, launch 구성 또는 패키지 메타데이터를 변경했다면 다시 빌드한다.

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select sound_anomaly --symlink-install
source install/setup.bash
```

## 7. 마이크 확인

현재 연결된 입력 장치 목록 확인:

```bash
PYTHONPATH=~/ros2_ws/python_deps \
python3 ~/ros2_ws/src/dyeun_robotics/sound_anomaly/scripts/list_audio_devices.py
```

ALSA에서 USB 마이크 확인:

```bash
arecord -l
```

현재 확인된 USB 마이크는 ALSA 기준 `card 2, device 0`이다. USB 장치를 다시
연결하면 카드 번호는 바뀔 수 있지만 ROS 설정은 장치 이름을 사용하므로
PortAudio 장치 번호 변경의 영향을 받지 않는다.

## 8. LED가 동작하지 않을 때

OpenCR 장치 확인:

```bash
ls -l /dev/ttyACM*
```

LED 토픽 subscriber 확인:

```bash
ros2 topic info /opencr_led_status
```

`Subscription count`가 0이면 수정된 로컬 `turtlebot3_node`가 실행 중인지
확인한다.

```bash
ros2 pkg prefix turtlebot3_node
```

다음 로컬 경로가 출력되어야 한다.

```text
/home/user/turtlebot3_ws/install/turtlebot3_node
```

두 노드의 `ROS_DOMAIN_ID`도 같아야 한다. 현재 기본값은 `30`이다.

```bash
echo $ROS_DOMAIN_ID
```

## 9. 주의 사항

- 현재 모델은 3초 길이의 소리 구간을 사용하여 판정한다.
- 공개 MIMII 기어박스 데이터로 학습된 모델이므로 실제 설비 및 현장 소음에서
  정상·이상 음원으로 반드시 재검증해야 한다.
- 이 모델의 판정만을 설비 안전 정지의 단독 근거로 사용하면 안 된다.
- 부팅 후 자동 백그라운드 실행은 마이크와 LED 현장 시험을 완료한 뒤 systemd
  서비스로 등록한다.
