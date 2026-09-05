# 2026 SW 파일럿 로보틱스 1팀 프로젝트

TurtleBot3 Burger와 Jetson Orin Nano로 만든 로봇 프로젝트입니다. 손 제스처·조이스틱·자이로
장갑으로 직접 몰기, SLAM/Nav2로 웨이포인트 순찰하기, 기어박스 소리로 이상 감지해 OpenCR
LED로 알리기, SO-101 로봇팔 원격 조종까지 들어 있습니다.

아래 설명은 폴더를 하나씩 열어 파일을 확인하면서 정리한 것입니다.

> **실행 환경** — Ubuntu 22.04 · Jetson Orin Nano · ROS 2 Humble에서 만들고 돌렸습니다.

## 폴더 구조

### `sw/robot/` — 핵심 소스 코드 (이 프로젝트의 본체)

젯슨에서 카메라로 손을 읽어 TurtleBot3를 조종하고, 자율주행·소리 이상감지·LED·로봇팔까지 한데 묶은 실행 코드입니다. 사용법은 `sw/robot/README.md`, 설치 순서는 `sw/robot/설치순서.md`에 있습니다.

- `src/` — 제스처 인식 파이프라인 본체
  - `capture/` — 카메라 캡처(`camera_stream.py`) 및 다중 카메라 중 활성 카메라 선택(`dual_camera.py`)
  - `inference/` — MediaPipe HandLandmarker 래퍼(`hand_tracker.py`)
  - `postprocess/` — 손 모양 판별(`hand_shape.py`, 21개 손 랜드마크 좌표 기반 기하 규칙), One Euro 필터(`point_filter.py`), 손 주변 디지털 줌(`roi_zoom.py`), 여러 사람 중 조작자 손을 고정 추적(`hand_lock.py`), 카메라 흔들림 보정(`camera_motion.py`)
  - `control/` — 손끝 위치를 상/하/좌/우 D-pad 방향으로 매핑(`dpad.py`), 손가락 개수 기반 명령(`finger_commands.py`)
  - `pipeline/` — 위 구성요소를 조립해 도는 백그라운드 루프(`gesture_loop.py`)
  - `server/` — Flask 서버(`app.py`) — `/cmd`(속도 명령 JSON), `/video_feed`(디버그 오버레이 영상), `/health` 제공
- `ros2_bridge/` — ROS 2 노드 모음(`rclpy`)
  - `cmd_vel_bridge.py` — Flask 서버의 `/cmd`를 읽어 `/cmd_vel`로 발행, 응답 지연 시 워치독으로 자동 정지
  - `cmd_vel_mux*.py` — 제스처·조이스틱·Nav2 등 여러 속도 입력 중 하나만 로봇에 전달하는 다중화 노드
  - `glove_gyro*.py`, `wifi_glove_teleop.py` — ESP32 자이로 장갑(Wi-Fi/시리얼) 입력을 `/cmd_vel_glove`로 변환
  - `waypoint_*.py` — A→B→C→D→A 순찰 미션, 좌표 샘플링, 자율주행↔수동조종 제어권 핸드오프
  - `sound_anomaly_with_led.launch.py`, `*_mux.launch.py`, `*_with_mux.launch.py` — 통합 실행용 launch 파일
- `camera_server/camera_server.py` — 노트북 등 별도 PC에서 카메라 영상만 네트워크로 송출하는 경량 서버(MediaPipe 불필요, 젯슨-노트북 분산 구성용)
- `so101_mediapipe_arm/` — SO-101 로봇팔 예제. 웹캠으로 사람 팔 관절(어깨·팔꿈치·손목·손가락 벌림)을 읽어 로봇팔에 상대 위치로 넘깁니다. TurtleBot3 코드와는 따로 돕니다
- `opencr_update/` — OpenCR 펌웨어를 컴퓨터에서 로봇으로 업로드하는 공식 도구(Windows `update.bat` / Linux `update.sh`)와 바이너리 생성 도구
- `firmware/esp32_mpu6050_glove/` — 자이로 장갑에 들어가는 ESP32 펌웨어(Arduino `.ino`, MPU6050 센서, Wi-Fi 설정)
- `configs/` — 실행 설정값: `config.yaml`(카메라·모델 임계값·D-pad 감도·안전 파라미터), `tgz_850m_mux.yaml`(조이스틱 매핑), `turtlebot3_navigation.rviz`(RViz 화면 구성)
- `maps/`, `models/` — 저장된 SLAM 지도(공장 지도, 시연용 지도)와 학습된 판별 모델(기어박스 소리 이상감지 SVM `.joblib`)
- `scripts/` — 실행 스크립트 모음. 제스처 단독 주행, 제스처+조이스틱, 자율주행 중 제스처/조이스틱 전환, 자이로 장갑 주행, 소리 이상감지 단독 실행, 오늘 지도 기준 A→B→C→D→A 전체 순찰 실행(`turtlebot3_전체실행.sh`) 등
- `tests/` — 카메라·모델·로봇 없이 도는 단위 테스트(손 모양 판별, D-pad 매핑, mux 로직, waypoint 도착 판정 등)
- `runtime_sources/` — 실제 로봇에서 돌아가는 ROS 2 워크스페이스(`~/ros2_ws`, `~/turtlebot3_ws`)의 소스 코드 원본 백업. 소리 이상감지 ROS 2 패키지(`ros2_ws/src`)와 OpenCR LED·LiDAR·Dynamixel 관련 소스(`turtlebot3_ws/src`)를 보관하며, 빌드 산출물(`build/`, `install/`, 가상환경)은 기기별로 재생성되므로 제외했습니다

### `turtlebot3_ws/` — 표준 ROS 2 워크스페이스

로봇 구동에 필요한 ROS 2 패키지 소스. ROBOTIS 공식 패키지(`turtlebot3`, `turtlebot3_msgs`, `DynamixelSDK`, `ld08_driver`, `coin_d4_driver`, `turtlebot3_slam_toolbox`, `turtlebot3_joystick`)와, 이 팀이 직접 작성한 커스텀 패키지 `turtlebot3_waypoint_patrol`(`patrol_node.py` — 순찰 노드, `safety_mission_manager.py` — 자율주행 중 장애물 감지 시 정지→후진→재개 안전 로직)이 함께 들어 있습니다. `maps/`에는 이 워크스페이스 기준 지도 파일이 있습니다.

### `20260810/` — ROS 2 설치본(bringup) 백업

2026-08-10 시점에 실제 로봇을 구동시킨 `/opt/ros/humble/share/` 아래의 TurtleBot3 bringup 관련 설치 파일(launch, URDF, LDS-02 드라이버, OpenCR 통신 노드 등)을 그대로 복사해 보관한 스냅샷. 어떤 launch 파일이 로봇의 어떤 하드웨어(라이다, OpenCR, 상태 발행)를 담당하는지 안내하는 `20260810_터틀봇3_파일위치_안내.md`가 포함되어 있습니다.

### `TurtleBot3_오류_분석_및_해결_2026-08-17/` — Jetson 이전 트러블슈팅 기록

로봇 제어 컴퓨터를 Jetson Orin Nano로 옮기는 과정에서 겹쳐 발생한 6가지 문제(조이스틱 패키지의 잘못된 rosdep 선언, Conda/시스템 Python 충돌, 불완전한 빌드 잔재, 모터 토크 비활성화, 라이다 모델 설정 오류, 라이다 포트 별칭 오류)를 원인·조사 과정·수정 방법 순으로 정리한 기록. 최종적으로 OpenCR·LDS-03 라이다·조이스틱·SLAM/Nav2가 모두 정상 동작한 상태로 마무리됐습니다. 당시 원본 로그(`raw_logs/`)와 수정된 설정 파일(`config_snapshots/`)도 함께 보관되어 있습니다.

### `led_test_ws/` — OpenCR LED 상태 표시 노드 (독립 테스트용)

젯슨 GPIO로 초록/빨강 LED를 켜서 로봇 상태(대기/정상/이상)를 표시하는 ROS 2 패키지 `led_status_node`. 하드웨어 없이도 상태 전이 로직만 검증할 수 있도록 GPIO 접근부와 상태 머신(`controller.py`)을 분리했고, 단위 테스트(`test_controller.py`)가 포함되어 있습니다.

## 최상위 문서·스크립트 파일

작업 기록과 실행 가이드가 시간 순으로 쌓여 있습니다.

| 파일 | 내용 |
|---|---|
| `turtlebot3_setup_progress_2026-08-08.md` | Jetson + TurtleBot3 Burger 최초 환경 구축 기록 |
| `20260815.md` | 제스처 제어 시스템 인수인계 문서 (포트·IP·상태 정리) |
| `모든 제스처 설명서.md` | 최종 손 제스처 전체 목록과 기능(모드 전환, 조이스틱 토글 등) |
| `OpenCR_LED_컨트롤러_사용법.md` | `/opencr_led_status` 토픽으로 OpenCR LED를 제어하는 방법 |
| `소리_이상감지_ROS2_패키지_실행법.md` | USB 마이크 기반 기어박스 소리 이상감지 패키지 실행법 |
| `sound_anomaly_led_실행방법.md` | 소리 이상감지 + LED 연동 통합 실행 스크립트 사용법 |
| `내일할것들.md` | 2026-08-22 시점 TurtleBot3 + 소리 이상감지 통합 시험 계획 |
| `Turtlebot3_Safety.md` | 자율주행 안전 정지 로직(장애물 감지 시 정지→후진→재개) 최종 구현 정리 |
| `TurtleBot3_전체실행.sh` / `.desktop`, `TurtleBot3_전체실행_Safety.desktop` | 오늘 지도 기준 A→B→C→D→A 전체 순찰을 한 번에 실행하는 스크립트/바로가기 |
| `sound_anomaly_led.sh`, `소음_이상감지_단독실행.desktop` | 소리 이상감지 + LED만 단독 실행 |
| `시연용.sh`, `시연용 복사본.sh` | 시연용 실행 스크립트 |
| `Codex_ESP32_IMU_WiFi_Automation_Guide.pdf`, `esp32_test_code_bundle.pdf` | 자이로 장갑 ESP32(IMU·Wi-Fi) 개발에 참고한 자동화 가이드와 테스트 코드 |
| `TURTLEBOT3_QS_BURGER_VER2544(online).pdf` | TurtleBot3 Burger 공식 퀵스타트 매뉴얼 |
| `factory_map_20260824_presentation.png` | 시연용 공장 지도 이미지 |
| `nomachine_9.8.2_1_arm64.deb` | 젯슨 원격 접속(NoMachine) 설치 패키지 |
| `20260810.zip`, `TurtleBot3_오류_분석_및_해결_2026-08-17.zip` | 위 두 폴더(`20260810/`, `TurtleBot3_오류_분석_및_해결_2026-08-17/`)의 압축 백업본 |
