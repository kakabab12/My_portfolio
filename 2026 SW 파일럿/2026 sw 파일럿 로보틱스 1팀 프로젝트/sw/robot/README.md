# 터틀봇3 제스처 수동조종 (젯슨 오린 나노 + Flask + ROS2)

터틀봇3 버거를 손짓으로 수동조종하기 위한 서브시스템. 소음 탐지 로봇 프로젝트의
첫 단계이며, 이 저장소는 **수동조종(원격조종)만** 다룬다.

카메라로 조작자의 손을 인식해 **리모컨 D-pad 방식**(방향 존을 가리키는
동안 고정 속도로 계속 이동 — RC카 리모컨처럼)으로 로봇 속도를 계산하고,
같은 컴퓨터 안에서 ROS2 노드가 이를 읽어 `/cmd_vel`로 발행한다.
손 검출·모양 판별은 [`D:\작업\리모콘 ui\gesture_kiosk`](../../../../작업/리모콘%20ui/gesture_kiosk)
(제스처 키오스크 프로젝트)의 MediaPipe 기반 모듈을 이식·재사용했다. **YOLO 등
별도 딥러닝 검출 모델은 쓰지 않는다** — 손 검출은 MediaPipe HandLandmarker
하나뿐이고, 주먹/한손가락/손바닥 판별은 21개 랜드마크 좌표로 직접 계산하는
기하 규칙(`src/postprocess/hand_shape.py`)이다.

## 아키텍처 (젯슨 직접 연결 카메라 3대)

**카메라 3대·손 인식·판정·ROS2를 전부 젯슨 오린 나노에서 처리**한다.
노트북 카메라 서버나 영상 네트워크 통신은 필요 없다.

```
[카메라 0, 1, 2] ── USB/V4L2 ──→              [젯슨 오린 나노]
                                              src.server.app (gesture_engine, Flask)
                                                - 카메라 3대의 프레임을 직접 수신
                                                - 손 추적(카메라별 HandTracker) → 손모양 판별
                                                - 손 품질이 가장 좋은 활성 카메라 선택
                                                - 손 주변 디지털 줌(ROI crop)으로 원거리 인식 유지
                                                  - D-pad 방향 존(상/하/좌/우/중앙) 매핑
                                                  → GET /cmd, /video_feed, /health
                                                        │
                                                        │ HTTP (localhost, ~20Hz)
                                                        ▼
                                              ros2_bridge/cmd_vel_bridge.py (rclpy 노드)
                                                  - 워치독: 응답 지연/실패 시 무조건 0 Twist
                                                  - 버거 물리 한계로 최종 클램프 후 /cmd_vel 발행
                                                        │ (기존 turtlebot3_bringup)
                                                        ▼
                                                     OpenCR → 모터
```

장치 번호 확인과 실행 방법은 아래 **"2-B"** 섹션을 참고한다.

**안전 설계(삼중 정지)**: 주행 잠금(아래 "조작 방법") 상태거나, 손이 안
보이거나, 손모양이 `open`(손바닥)이 아니면 gesture_engine이 그 프레임부터
유예 없이 (0, 0)을 낸다. 여기에 더해
`cmd_vel_bridge.py`가 자체 워치독으로 응답이 오래됐거나(`age_sec` 초과) HTTP
요청 자체가 실패해도 무조건 0을 발행한다 — gesture_engine 프로세스가
멈추거나 로봇이 조작자에게서 멀어져 인식이 끊겨도 반드시 정지한다.

**추론 장치**: 현재 설치된 MediaPipe 빌드는 GPU delegate를 포함하지 않으므로
`configs/config.yaml`의 `hand_tracker.delegate: cpu`를 기본값으로 둔다. 손 모델,
카메라 수신·색상 변환·제스처 판정·ROS 제어가 모두 CPU에서 실행된다. 실행 뒤
`http://127.0.0.1:5000/health`에서
`inference.trackers[*].active_delegate: "cpu"`인지 확인할 수 있다.

## 디렉터리 구조

```
configs/config.yaml   # 카메라·모델 임계값·조이스틱 매핑·안전·서버 설정 — 튜닝은 여기서만
models/weights/        # hand_landmarker.task (scripts/download_weights.py로 받음)
src/
├─ capture/            # 카메라 캡처(camera_stream) + 다중 카메라 활성 선택(dual_camera)
├─ inference/          # MediaPipe HandLandmarker 래퍼(hand_tracker)
├─ postprocess/        # 손모양 판별(hand_shape) · One Euro 필터(point_filter) · ROI 줌(roi_zoom)
│                       #   · 배경 인물 무시(hand_lock)
├─ control/            # D-pad 방향 존 매핑(dpad)
├─ pipeline/           # 위 전부를 조립하는 백그라운드 루프(gesture_loop)
└─ server/             # Flask 앱(app) — /cmd, /video_feed, /health
ros2_bridge/cmd_vel_bridge.py       # ROS2 노드 — 젯슨의 ROS2 환경에서 별도 실행
camera_server/camera_server.py      # 카메라만 네트워크로 송출 — 노트북 등에서 별도 실행
                                     #   (mediapipe 불필요, requirements-camera-server.txt)
tests/                 # 카메라·모델 없이 도는 단위 테스트
```

## 설치

> ★ **바로 실전(로봇) 구성으로 가려면** 아래로 내려서 **"노트북 카메라 +
> 젯슨 중앙집중"** 섹션으로 바로 가도 된다(체크리스트는
> [설치순서.md](설치순서.md)). 아래 1~4번은 순서대로 읽는 참고 설명이고,
> 2-B(카메라를 젯슨에 직접 연결)는 노트북 없이 젯슨 단독으로 갈 때만
> 필요한 **대안**이다.

### 1. 손 모델 다운로드 (공통)

```bash
python scripts/download_weights.py
```

### 2-A. 윈도우에서 개발용 웹캠 테스트

```bash
pip install -r requirements-gesture.txt
python -m src.server.app
```

(또는 `scripts\run_gesture_server.bat` 더블클릭.) 브라우저로 `http://127.0.0.1:5000/` 접속 — 카메라 화면(디버그 오버레이 포함)과
`/cmd` JSON이 같이 보인다. 웹캠이 1대뿐이면 `configs/config.yaml`의
`camera.devices`가 `[0, 1, 2]`라도 없는 카메라는 "열기 실패" 경고만 남기고
자동으로 건너뛴다 — 나머지 한 대만으로 정상 동작한다.

### 2-B. (대안) 젯슨 오린 나노 배포 — 카메라를 젯슨에 직접 연결

코드는 이식 없이 그대로 옮기면 된다(리눅스 V4L2 카메라 경로가 이미 있고,
윈도우 전용 코드는 없다) — 다만 **"실행 환경"** 3가지는 젯슨에서 별도로
맞춰야 한다: ①파이썬 라이브러리 설치 ②카메라 장치 번호 ③(있다면) 모델
파일. 아래 순서대로.

**0) 이 폴더를 젯슨으로 옮기기** — 방법은 무엇이든 상관없다(코드는 그대로).
   - USB 드라이브로 복사, 또는
   - 같은 네트워크에서 scp(윈도우 Git Bash 기준):
     ```bash
     scp -r "SW파일럿" <젯슨계정>@<젯슨IP주소>:~/gesture_robot
     ```
   - 이미 `models/weights/hand_landmarker.task`(8MB)를 이 윈도우 PC에서
     받아둔 상태라면, 폴더째로 옮기면 젯슨에서 인터넷 연결 없이도 모델
     다운로드 단계(1번)를 건너뛸 수 있다.

**1) 손 모델 확인/다운로드** — 젯슨에서:
```bash
cd ~/gesture_robot
python3 scripts/download_weights.py   # 이미 파일이 있으면 그대로 두고 스킵
```

**2) 파이썬 라이브러리 설치** — 젯슨에서:
```bash
pip3 install -r requirements-gesture.txt
```
- `mediapipe`/`opencv-contrib-python`은 aarch64용 wheel 가용 여부가 젯팩·
  파이썬 버전마다 달라 `requirements-gesture.txt`에 적힌 정확한 버전이 안
  깔릴 수 있다. 실패하면:
  ```bash
  pip3 install mediapipe opencv-contrib-python   # 버전 안 박고 되는 걸로
  ```
  로 다시 시도하고, 성공한 버전으로 `requirements-gesture.txt`를 갱신해 둘 것.
  코드 자체는 mediapipe Tasks API를 쓰고 버전에 따라 바뀌지 않으므로 이
  두 패키지만 젯슨에서 되는 버전으로 바꿔도 손댈 코드는 없다.

**3) 카메라 장치 번호 확인 후 config 수정** — 젯슨에서:
```bash
ls /dev/video*              # 예: /dev/video0 /dev/video2 /dev/video4 ...
# v4l2-utils가 있으면 어떤 장치인지 더 자세히: sudo apt install v4l-utils
v4l2-ctl --list-devices
```
카메라 1대당 보통 장치 번호가 2개씩(video0/video1 등) 잡히는 경우가 많다 —
실제로 영상이 나오는 번호(보통 짝수)를 골라 `configs/config.yaml`의
`camera.devices: [0, 1, 2]`를 실제 영상 장치 번호로 바꾼다. 카메라 한 대가
여러 `/dev/video*` 노드를 만들면 실제 영상이 나오는 번호가 `[0, 2, 4]`처럼
될 수 있다.

**4) 실행**:
```bash
python3 -m src.server.app
# 또는
bash scripts/run_gesture_server.sh
```
같은 네트워크의 다른 PC/폰 브라우저에서 `http://<젯슨IP주소>:5000/`으로
접속하면 카메라 화면·조작 상태가 보인다(젯슨에 모니터가 없어도 확인 가능).

### 3. ROS2 + 터틀봇3 패키지 설치 (젯슨에 ROS2가 아직 없다면)

**이 저장소 코드와는 무관한, 로봇 쪽(ROBOTIS 공식) 설치 절차다.** 이미
ROS2·turtlebot3_bringup이 OpenCR과 통신하고 있으면 이 단계는 건너뛰고
바로 4번으로. 아래는 젯슨 오린 나노(젯팩 6.x = Ubuntu 22.04) 기준 ROS2
**Humble**(Ubuntu 22.04 LTS 짝) 설치다 — 공식 문서가 최종 근거이니 버전별
차이가 있으면 그쪽을 따를 것:
- ROS2 공식 설치 문서: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html
- 터틀봇3 공식 퀵스타트(ROBOTIS e-Manual): https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/

**3-1) ROS2 Humble 설치**
```bash
locale  # UTF-8 확인, 아니면 아래로 설정
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

sudo apt install -y software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install -y curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update && sudo apt upgrade -y
sudo apt install -y ros-humble-ros-base python3-colcon-common-extensions python3-rosdep
# GUI(RViz 등) 도구까지 필요하면 ros-humble-ros-base 대신 ros-humble-desktop

sudo rosdep init
rosdep update

echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

**3-2) 터틀봇3 ROS2 패키지 설치** — apt로 먼저 시도:
```bash
sudo apt install -y ros-humble-turtlebot3-msgs ros-humble-turtlebot3 \
  ros-humble-dynamixel-sdk ros-humble-hls-lfcd-lds-driver
```
apt에 없거나 버전이 안 맞으면 소스로 빌드(공식 절차):
```bash
mkdir -p ~/turtlebot3_ws/src && cd ~/turtlebot3_ws/src
git clone -b humble-devel https://github.com/ROBOTIS-GIT/turtlebot3.git
git clone -b humble-devel https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git
git clone -b humble-devel https://github.com/ROBOTIS-GIT/DynamixelSDK.git
cd ~/turtlebot3_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
echo "source ~/turtlebot3_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

**3-3) 모델·환경변수 지정** — 조립한 로봇이 버거(Burger)이므로:
```bash
echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc
source ~/.bashrc
```

**3-4) OpenCR 연결 확인** — USB로 연결 후:
```bash
ls /dev/ttyACM*        # 보통 /dev/ttyACM0 — 안 보이면 케이블·연결 확인
```
조립 과정에서 이미 터틀봇3 공식 절차대로 OpenCR 펌웨어를 구웠다면 이 단계는
그걸로 끝이다(재설치 불필요) — 처음 조립해 펌웨어를 구운 적이 없다면
공식 퀵스타트의 "OpenCR 설정" 절만 별도로 따라야 한다(Arduino IDE로 진행,
이 저장소와는 무관한 1회성 작업).

**3-5) bringup 실행 + 단독 확인** — 우리 코드를 붙이기 **전에** 먼저
표준 키보드 텔레옵으로 OpenCR·모터까지 실제로 도는지 확인해 둘 것
(문제가 생겼을 때 "우리 코드 문제"와 "로봇/ROS2 설정 문제"를 구분하기 위함):
```bash
ros2 launch turtlebot3_bringup robot.launch.py
```
다른 터미널에서:
```bash
ros2 topic list                 # /cmd_vel이 보여야 한다
ros2 run turtlebot3_teleop teleop_keyboard   # w/a/s/d/x로 실제 바퀴가 도는지 확인
```
여기까지 되면 로봇·ROS2 쪽은 검증 끝 — 이제 4번의 `cmd_vel_bridge.py`가
`teleop_keyboard` 대신 손짓으로 같은 `/cmd_vel`에 명령을 넣는 것뿐이다.

### 4. ROS2 브리지 실행 (이 저장소 코드)

위 3번(bringup)이 별도 터미널에서 이미 돌고 있는 상태로, 또 다른 터미널에서:

```bash
source /opt/ros/<distro>/setup.bash    # 예: humble
pip install -r requirements-ros2.txt
python3 ros2_bridge/cmd_vel_bridge.py --base-url http://127.0.0.1:5000
```
(`--base-url`은 항상 gesture_engine이 **떠 있는** 컴퓨터를 가리킨다 —
gesture_engine은 이 저장소의 권장 구성에서 항상 젯슨 자신에서 돈다
(카메라만 노트북 걸 쓰더라도)이므로 `127.0.0.1`을 그대로 쓴다. gesture_engine
자체를 노트북에서 돌리는 대안 구성으로 갈 때만 노트북 IP로 바꾼다.)

지금은 colcon 패키지가 아니라 단일 스크립트다 — ROS2 환경만 source되어 있으면
바로 실행된다. 나중에 launch 파일·다른 노드와의 의존 관계가 필요해지면
`ament_python` 패키지(`package.xml` + `setup.py`)로 승격하면 된다.

### 4-A. SLAM/Nav2 없이 제스처 주행만 실물 확인

SLAM/Nav2·mux·조이스틱을 모두 빼고, 제스처 서버와 컨트롤러만으로 바퀴가
움직이는지 확인할 때는 아래 한 명령을 쓴다. 이 스크립트는 TurtleBot3 bringup,
제스처 서버, 그리고 `/cmd_vel` 직접 브리지만 시작한다. 기존 bringup·Nav2·mux·
제스처 브리지는 먼저 모두 종료해야 한다.

```bash
cd /home/user/sw/robot
bash scripts/run_gesture_drive_test.sh --usb-port /dev/ttyACM0
```

`/dev/ttyACM0`은 실제 OpenCR 포트로 바꾼다. 하나만 연결되어 있으면
`--usb-port`은 생략할 수 있다. 현재 기본 설정은 손가락 명령 모드이므로
1손가락=전진 1m, 2손가락=후진 1m, 3손가락=우회전 90°, 4손가락=좌회전
90°다. 움직이는 중에는 주먹을 약 0.25초 유지하면 정지하며, 전체 종료는
`Ctrl+C`다. 처음에는 바퀴가 바닥에 닿지 않게 들거나 충분히 넓은 공간에서
시험한다.

### 4-B. SLAM/Nav2 없이 제스처 + 조이스틱으로만 주행

제스처와 조이스틱 둘만 사용할 때는 아래 스크립트를 쓴다. 하드웨어는
`/cmd_vel_muxed`만 구독하고, mux가 제스처 또는 조이스틱 중 하나의 속도만
전달하므로 두 입력이 동시에 바퀴를 제어하지 않는다. Nav2/SLAM은 시작하지
않는다.

```bash
cd /home/user/sw/robot
bash scripts/run_gesture_joystick_drive.sh --usb-port /dev/ttyACM0
```

제스처 제어는 시작부터 켜진 상태다. 제스처 ON 상태에서 **따봉을 1.5초 유지**하면
제스처 모드가 꺼지고 AUTO/Nav2로 복귀한다. 모드 OFF 상태에서는 짧게 보인
**따봉**으로 제스처 모드를 다시 켤 수 있다. **OK 사인**을 하면 조이스틱 모드가
켜진다. 켤 때 사용한 OK 사인을 한 번 풀고 다시 **1.5초 유지**하면 조이스틱과
제스처가 모두 꺼지고 AUTO/Nav2로 복귀한다. 제스처 모드에서 손 입력이 15초 동안 없으면
로봇을 정지시킨 뒤 컨트롤러 모드로 자동 전환한다. 원샷 거리·회전 명령 중에는
동작을 중간에 끊지 않고 완료 후 전환한다.
기존 bringup·직접 제스처 브리지·Nav2·mux는 이 스크립트를 실행하기 전에
종료해야 한다.

웹 화면(`http://<젯슨IP>:5000/`)에는 `/dev/video0`, `/dev/video2`의 영상이
동시에 보이며, 상단에 현재 **제스처 모드**, **컨트롤러 모드**, 또는
**자율주행 모드**가 표시된다.

### 4-C. 자율주행 중 제스처·조이스틱으로 제어권 전환

저장 지도 기반 Nav2/AMCL을 실행하면서 자율주행과 수동 조작을 전환할 때는
아래를 사용한다. 시작 제어권은 자율주행이며, Nav2의 위치 추정과 목표 상태는
제스처·조이스틱 조작 중에도 계속 실행된다. 웹 화면도 서버가 시작되는 즉시
**자율주행 모드**를 표시한다.

```bash
cd /home/user/sw/robot
bash scripts/run_navigation_gesture_joystick.sh \
  --usb-port /dev/ttyACM0 \
  --map /home/user/turtlebot3_ws/maps/factory_map_20260821.yaml
```

이 통합 실행은 첫 번째 C270 웹캠의 내장 마이크로 소음 이상을 탐지하고 OpenCR LED를
연동한다. 외장 P5U(MUSIC-BOOST) 마이크는 USB 허브 안정성 문제로 사용하지 않는다.
필요하면 `--without-sound`로 소음 탐지와 LED 연동을 생략할 수 있다.

RViz에서 먼저 AMCL 초기 위치와 2D Goal Pose를 설정한다. **짧은 따봉**은
제스처 제어권을 가져오고, 제스처 ON 상태에서 **따봉을 1.5초 유지**하면 Nav2로
돌아간다. **짧은 OK 사인**은 조이스틱 제어권을 가져오며, 손을 풀었다가 다시
**OK 사인을 1.5초 유지**하면 조이스틱·제스처를 모두 해제해 Nav2로 돌아간다.

### 4-D. 한 번에 전체 실행 (오늘의 A → B → C → D → A 순찰)

오늘 측정된 `factory_map_20260821` 지도와 `turtlebot3_waypoint_patrol`의 A → B
→ C → D → A 순찰까지 한 번에 시작하려면 아래 실행 파일을 쓴다.

```bash
cd /home/user/sw/robot
bash scripts/turtlebot3_전체실행.sh
```

파일 관리자에서는 `TurtleBot3_전체실행.desktop`을 더블클릭해도 된다. 최초
한 번은 파일 속성에서 “실행 허용”을 선택할 수 있다. 이 실행은 AMCL 초기 위치를
오늘의 A 좌표로 자동 설정한다. A·B는 Nav2, C·D는 수동 주행으로 구간을
완료하며 모든 waypoint는 목표 좌표 반경 0.3m 이내면 도착으로 처리한다.
AMCL이 준비될 때까지 초기 위치 A를 최대 15초 동안 재전송한 뒤 A→B를 시작한다.

### 4-E. 자이로 장갑으로 컨트롤러 대체

자이로 장갑은 기존 게임 컨트롤러와 별도의 `/cmd_vel_glove` 입력을 사용한다.
`TurtleBot3 전체 실행`에도 Wi-Fi 장갑 수신기가 포함되며, **컨트롤러 모드에서만**
장갑 제어권이 활성화된다. 이때 스틱을 실제로 움직이는 동안 조이스틱이 우선이고,
스틱이 중립이면 장갑을 사용할 수 있다. 따라서 TurtleBot3 하드웨어에는 언제나 한
경로(`/cmd_vel_muxed`)만 전달된다. 장갑 데이터가 0.35초 이상 끊기면 mux도 0 속도를
유지한다.

장갑 펌웨어는 개행마다 아래 중 하나를 전송해야 한다(각도 단위는 도).

```text
{"pitch": 12.5, "roll": -8.0}
pitch:12.5, roll:-8.0
12.5,-8.0
```

`/dev/ttyACM0`은 TurtleBot3 OpenCR일 수 있으므로 장갑 포트로 절대 지정하지 않는다.
장갑을 꽂은 뒤 `ls -l /dev/serial/by-id/ /dev/ttyUSB* /dev/ttyACM*`로 장갑의 포트를
확인하고, 처음에는 바퀴를 든 상태에서 실행한다.

```bash
cd /home/user/sw/robot
/usr/bin/python3 -m pip install -r requirements-ros2.txt
bash scripts/run_glove_gyro_drive.sh \
  --usb-port /dev/ttyACM0 \
  --glove-port /dev/ttyUSB1 \
  --glove-baud 115200
```

시작 후 처음 2초 동안 장갑을 편한 중립 자세로 유지한다. 기본값은 앞으로
기울이면 전진, 뒤로 기울이면 후진, 오른쪽/왼쪽으로 기울이면 우/좌회전이다.
방향이 반대면 실행 중인 장갑 노드에 `--invert-pitch` 또는 `--invert-roll`을
추가한다. CSV가 `roll,pitch` 순서라면 `--csv-order roll-pitch`를 추가한다.

## 노트북 카메라 + 젯슨 중앙집중 (핫스팟 연동, 권장 구성)

**손 인식·판정·ROS2까지 전부 젯슨에서 처리**하고, 노트북은 카메라 영상만
네트워크로 내보낸다(`camera_server/camera_server.py`) — mediapipe 같은
인식 관련 무거운 라이브러리는 노트북에 전혀 필요 없다, 순수 영상 전달만
한다. `src/capture/camera_stream.py`가 `camera.devices`에 URL 문자열이
오면 로컬 장치 대신 그 네트워크 스트림을 그대로 읽어들인다(`cv2.VideoCapture(url)`).

> 대안: 반대로 노트북에서 인식까지 끝내고 젯슨은 브리지만 돌리는 구성도
> 가능하다(`python -m src.server.app`을 노트북에서, `cmd_vel_bridge.py`
> `--base-url`에 노트북 IP를 넣어 젯슨에서) — 이 경우 젯슨에
> mediapipe/opencv가 필요 없어지는 대신, 반응은 지금 구성이 더 빠르다
> (이 구성은 네트워크로 작은 숫자만 오가고, 저 구성은 영상 자체가 오간다).
> 아래는 **이 저장소가 실제로 검증한 권장 구성**만 다룬다.

**역할 분담**

| 기기 | 실행하는 것 |
|---|---|
| 노트북 | `python camera_server/camera_server.py --devices 0 --port 8090` — 카메라 영상만 송출 |
| 젯슨 | `python3 -m src.server.app` — 노트북 스트림을 받아 인식·판정 |
| 젯슨 | `ros2 launch turtlebot3_bringup robot.launch.py` |
| 젯슨 | `ros2_bridge/cmd_vel_bridge.py --base-url http://127.0.0.1:5000` (**젯슨 자기 자신**이므로 localhost 그대로) |

★ `localhost`는 "지금 그 명령을 치는 기기 자신"을 가리킨다. 이 구성에서
`http://127.0.0.1:...`을 쓰는 건 젯슨→젯슨(브리지→gesture_engine)뿐이고,
젯슨→노트북(카메라 스트림)은 반드시 **노트북의 실제 IP 주소**를 써야 한다.

**1) 노트북용 라이브러리 설치** (mediapipe 불필요)
```bash
pip install -r requirements-camera-server.txt
```

**2) 같은 네트워크로 묶기 (핫스팟)** — 학교·회사 와이파이는 기기 간 통신을
막는 "AP 격리"가 걸려있는 경우가 많아, 핫스팟으로 따로 묶는 게 안전하다.

- **A. 휴대폰 핫스팟(추천, 간단)** — 폰에서 개인 핫스팟(테더링)을 켜고,
  노트북과 젯슨 둘 다 그 와이파이에 연결한다. 시스템 설정을 안 건드려도 된다.
- **B. 노트북을 핫스팟으로** — 윈도우 **설정 → 네트워크 및 인터넷 → 모바일
  핫스팟**에서 켠다. 거기 표시되는 SSID·암호로 젯슨을 연결한다. (노트북에
  무선 어댑터가 하나뿐이면 켜는 동안 노트북 자체의 기존 와이파이 인터넷
  연결은 끊길 수 있다 — 인터넷은 안 써도 되므로 문제없다.)

**3) 노트북의 IP 확인** — 새로 연결된 네트워크(핫스팟) 기준으로:
```powershell
ipconfig
```
방금 붙은 어댑터의 IPv4 주소를 적어둔다 (A안은 보통 `192.168.43.x`대,
B안은 보통 `192.168.137.1`로 고정) — 아래에서 `<노트북IP>` 자리에 쓴다.

**4) 노트북에서 camera_server 실행**
```bash
python camera_server/camera_server.py --devices 0 --port 8090
```
카메라가 2대면 `--devices 0,1`. `http://<노트북IP>:8090/`을 아무 브라우저로
열어 스트림 링크가 보이면 정상.

**5) 윈도우 방화벽 확인** — 새 네트워크를 윈도우가 "공용 네트워크"로 잡으면
인바운드 연결을 막을 수 있다. 젯슨에서 접속이 안 되면 대부분 이 문제다 —
방화벽 알림이 뜨면 "허용", 또는 설정에서 그 네트워크를 "개인 네트워크"로 바꾼다.

**6) 젯슨의 config.yaml에 노트북 스트림 주소 넣기**
```yaml
camera:
  devices: ["http://<노트북IP>:8090/cam/0/stream"]
```

**7) 젯슨에서 연결 확인 후 gesture_engine·bringup·브리지 순서로 실행**
```bash
curl http://<노트북IP>:8090/health          # 먼저 연결 자체 확인
python3 -m src.server.app                   # 터미널 1
ros2 launch turtlebot3_bringup robot.launch.py   # 터미널 2
python3 ros2_bridge/cmd_vel_bridge.py --base-url http://127.0.0.1:5000   # 터미널 3
```

연결이 안 되면(타임아웃·연결거부) 순서대로 의심할 것: ①같은 네트워크에
붙어있는지(`ipconfig`/`ip a`로 서로 확인) ②5번 방화벽 ③AP 격리(그래도
안 되면 A안 휴대폰 핫스팟으로 교체) — 코드·설정 문제가 아니라 대부분
네트워크 연결 문제다.

노트북이 절전모드로 들어가거나 와이파이 연결이 끊기면 당연히 통신도
끊긴다 — 실기 테스트 중에는 노트북 절전 설정을 꺼 둘 것.

## 조작 방법

리모컨 D-pad 방식이다 — gesture_kiosk의 `main_dpad.py`와 같은 캘리브레이션·커서
방식을 그대로 따른다: **손이 처음 나타난 자리가 곧바로 기준(0점)**이 된다
(화면 중앙에 서 있을 필요 없음). 그 기준에서 손끝이 상/하/좌/우 중 어느
방향에 있는지만 보고, **그 방향을 가리키는 동안 고정 속도로 계속 움직인다**
(RC카 리모컨 버튼처럼) — 손끝을 얼마나 멀리 뻗었는지는 속도 크기에 영향을
주지 않는다. 손끝 오프셋은 `dpad.cursor_sensitivity`(기본 3배, main_dpad.py와
동일)만큼 증폭해서 판정하므로 팔을 크게 안 휘둘러도 존에 닿는다. `/video_feed`에
중앙 원 + 상하좌우 화살표 + 증폭된 커서 점으로 지금 어느 존을 가리키고
있는지 밝게 표시된다.

- **엔진을 막 시작하면 항상 잠금(LOCKED) 상태**다 — 안전 기본값. 중앙
  (center 존)에서 **손바닥을 편 채로(open)** 약 1.2초 가만히 있으면 잠금이
  풀린다(ARMED). 같은 동작을 다시 하면 도로 잠긴다 — 토글이다.
- **잠금 해제(ARMED) 상태에서만**, 손바닥을 편 채로 손끝을 중앙 반경 밖
  위/아래/왼쪽/오른쪽 존으로 옮기면 그 방향으로 계속 이동한다 —
  위=전진, 아래=후진, 오른쪽=우회전, 왼쪽=좌회전. 존을 벗어나거나(중앙으로
  복귀 포함) 대각선처럼 방향이 애매하면 그 프레임부터 즉시 멈춘다.
- **주먹을 쥐거나 손을 감추거나 잠금 상태면** 항상 정지(0, 0) — 명확한 정지 동작.
- 손이 한 번 사라졌다 다시 나타나면(예: 손을 완전히 내렸다가 다시 든 경우)
  그 자리가 또 새 기준으로 재캘리브레이션된다. 손을 떼지 않은 채로 자리만
  살짝 틀어졌으면, 잠깐(기본 2.5초) 가만히 있는 것만으로도 그 자리가 새
  중심이 된다(재중심 — main_dpad.py의 "허공에 가만히"와 동일).
- 방향이 반대로 느껴지면 `configs/config.yaml`의 `dpad.invert_x`/`invert_y`를
  뒤집는다. 커서가 너무 예민하거나 둔감하면 `dpad.cursor_sensitivity`를
  조정한다(1.0=증폭 없음). 속도는 `dpad.linear_mps`/`angular_radps`(존마다 고정 속도라 이
  값 자체가 곧 주행 속도), 중앙 존 크기는 `dpad.center_radius_ratio`,
  잠금 시작 상태·유지 시간은 `dpad.lock_toggle`에서 조정한다.

## 검증 순서 (실제 로봇 구동 전 반드시 순서대로)

1. **단위 테스트** — 카메라·모델 불필요:
   ```bash
   python -m unittest discover tests -v
   ```
2. **웹캠 스모크 테스트**(위 2-A) — `/cmd`의 `armed`가 시작 시 `false`인지,
   중앙에서 손바닥을 펴고 잠시 기다리면 `true`로 바뀌는지, 그 상태에서 손을
   움직이면 `linear_x`/`angular_z`가 기대 방향대로 바뀌는지, **손을 감추거나
   카메라를 가리거나 다시 잠그면 즉시 0,0이 되는지** 눈으로 확인.
3. **bringup 단독 확인**(3-5번) — 이 저장소 코드 없이 `teleop_keyboard`로
   먼저 실제 바퀴가 도는지 확인. 여기서 문제가 나면 로봇/ROS2/OpenCR 쪽
   문제이지 이 저장소 코드 문제가 아니다.
4. **ROS2 드라이런**(로봇 bringup은 켜두고, 아직 실주행은 안 함) —
   `cmd_vel_bridge.py`(4번) 실행 후 `ros2 topic echo /cmd_vel`로 값이
   손짓과 일치하는지, gesture_engine을 죽이거나 손을 감췄을 때 정확히
   0으로 떨어지는지 확인.
5. **실주행 전 최종 점검** — 바퀴가 지면에 안 닿게 로봇을 들거나 열린 공간에서
   먼저 테스트. 손을 카메라 밖으로 뺐을 때/카메라를 가렸을 때 즉시 멈추는지
   반드시 확인한 뒤에만 실제 주행 테스트로 넘어간다.

## 알려진 한계 / 다음 단계

- 조작자 1인을 가정한다. 붐비는 배경 대응으로 `src/postprocess/hand_lock.py`
  (`PrimaryHandTracker`)가 ①너무 작은(먼) 손은 후보에서 제외하고 ②이미
  추적 중인 손은 신뢰도가 더 높은 다른 손이 나타나도 공간 연속성으로 계속
  우선한다 — 매 프레임 새로 뽑지 않고 한 번 잡은 손에 "고정"된다
  (`configs/config.yaml`의 `primary_hand` 섹션에서 조정). 그래도 부족하면
  (예: 배경 사람이 조작자보다 카메라에 더 가까이 다가오는 경우) gesture_kiosk의
  포즈 기반 머리 앵커 게이트(`hand_select.py`)를 가져와 확장할 것 — 별도 포즈
  모델이 필요해 지금은 의도적으로 빼뒀다.
- 카메라 2대를 매 프레임 모두 추론한다 — 젯슨에서 부담이 크면
  `dual_camera.py`에 "비활성 카메라는 주기적으로만 확인" 옵션을 추가해 완화할 것.
- ROS2 Humble(젯팩 6 / Ubuntu 22.04) 가정 — 다른 배포판이면 rclpy 기본 API가
  같아 큰 변경 없이 동작할 것으로 예상되나 미검증.
