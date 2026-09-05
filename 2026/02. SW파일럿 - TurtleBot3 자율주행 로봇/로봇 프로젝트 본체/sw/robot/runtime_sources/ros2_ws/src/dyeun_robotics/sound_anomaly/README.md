# 기어박스 소리 이상 감지

Jetson Orin Nano에서 ROS 2 노드로 패키징할 수 있도록 준비한 기어박스 정상/비정상 감지 모델과 추론 코드입니다. 원음 WAV와 학습용 ZIP은 저장소에 넣지 않습니다.

## 포함 내용

- `models/gearbox_svm_source.joblib` — Jetson에서 불러 쓸 학습 모델 (약 2 MB)
- `models/*report.json` — 보류 평가 결과
- `scripts/live_predict.py` — 마이크로 3초마다 실시간 판정
- `scripts/predict_file.py` — WAV 파일 판정
- `scripts/audio_utils.py` — 특징 추출 공통 코드
- `scripts/train_machine.py` — 동일 형식의 원음이 있을 때 재학습하는 도구

## 현재 검증 성능

MIMII 기어박스 음원의 `source_test` 762개로 학습하고, 사용하지 않은 `target_test` 645개로 평가했습니다.

| 측정 기준 | 결과 |
| --- | ---: |
| 파일 전체(3개 3초 구간 평균) 정확도 | 78.1% |
| 3초 실시간 구간 정확도 | 74.8% |
| 3초 실시간 구간 비정상 F1 | 75.6% |

이 값은 공개 MIMII 음원 기준입니다. Jetson에 연결한 실제 마이크와 현장 소리에서는 반드시 정상·비정상 녹음으로 재검증해야 하며, 안전 제어의 단독 근거로 사용하면 안 됩니다.

## Jetson에서 파일 판정

```bash
python3 -m pip install -r requirements-inference.txt
python3 scripts/predict_file.py /path/to/gearbox.wav --model models/gearbox_svm_source.joblib
```

## Jetson에서 마이크 실시간 판정

```bash
python3 -m pip install -r requirements-inference.txt
python3 scripts/list_audio_devices.py
python3 scripts/live_predict.py --model models/gearbox_svm_source.joblib --device <마이크_번호>
```

`requirements-inference.txt`는 모델을 만든 환경의 핵심 라이브러리 버전을 고정합니다. `live_predict.py`는 모델에 저장된 16 kHz, 3초 구간, 임계값 0.50을 자동 적용합니다. ROS 2 패키지화할 때는 이 추론 로직을 노드로 옮겨 이상 확률, 정상/비정상 결과, 시각을 토픽으로 발행하면 됩니다.

## ROS 2 실시간 감지 및 OpenCR LED

이 폴더는 ROS 2 Humble `ament_python` 패키지입니다. 감지 상태에 따라 LED가 이렇게 움직입니다.

| 감지 상태 | OpenCR 모드 | LED 동작 |
| --- | ---: | --- |
| `ABNORMAL` | 3 | GPIO 51 빨간 LED만 점멸 |
| `NORMAL` | 2 | GPIO 50 초록 LED만 점멸 |
| `IDLE` | 1 | 빨강과 초록 LED 모두 계속 켜짐 |

노드를 종료하면 종료 전용 OpenCR 모드 `0`을 마지막으로 발행하여 두 LED를 모두
끈다. 이를 위해 전체 시스템을 끌 때는 이상감지 노드를 TurtleBot3 노드보다 먼저
종료해야 한다.

`IDLE`은 시작 중, 입력 음량이 `silence_rms_threshold`보다 작은 경우 또는 추론
오류가 발생한 경우입니다. LED 명령은 `/opencr_led_status`에
`std_msgs/msg/UInt8`로 발행됩니다. 현재 상태와 확률은 각각
`/sound_anomaly_node/state`, `/sound_anomaly_node/anomaly_probability`에서 확인합니다.

의존성은 Jetson의 시스템 Python과 충돌하지 않도록 별도 폴더에 설치합니다.

```bash
python3 -m pip install --target ~/ros2_ws/python_deps \
  -r ~/ros2_ws/src/dyeun_robotics/sound_anomaly/requirements-inference.txt
touch ~/ros2_ws/python_deps/COLCON_IGNORE
```

빌드 및 실행:

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select sound_anomaly --symlink-install
source install/setup.bash
ros2 launch sound_anomaly sound_anomaly.launch.py
```

다른 터미널에서 결과 확인:

```bash
ros2 topic echo /sound_anomaly_node/state
ros2 topic echo /sound_anomaly_node/anomaly_probability
ros2 topic echo /opencr_led_status
```

마이크, 무음 기준 및 LED 배선 설정은 `config/sound_anomaly.yaml`에서 변경합니다.
실제 마이크의 장치 번호는 다음 명령으로 확인합니다.

```bash
PYTHONPATH=~/ros2_ws/python_deps python3 scripts/list_audio_devices.py
```

`launch` 파일은 노드가 예기치 않게 종료되면 2초 뒤 다시 실행합니다. Jetson 부팅
후 자동 백그라운드 실행은 마이크와 LED 현장 시험이 끝난 뒤 systemd 서비스로
등록하는 것을 권장합니다.
