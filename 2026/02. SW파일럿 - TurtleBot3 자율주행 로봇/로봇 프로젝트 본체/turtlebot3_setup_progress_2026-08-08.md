# TurtleBot3 Burger + Jetson Orin Nano 설정 기록

- 작성일: 2026-08-08
- 환경: NVIDIA Jetson Orin Nano, Ubuntu 22.04.5 LTS, ARM64 (`aarch64`)
- ROS 배포판: ROS 2 Humble
- 로봇: TurtleBot3 Burger
- LiDAR: LDS-02 (현재 테스트에서는 분리)
- 조이스틱: Waveshare 판매 TGZ-850M 2.4 GHz 무선 게임패드

## 1. 시스템 역할 정리

TurtleBot3 매뉴얼에서 사용하는 용어는 다음과 같다.

- Remote PC: 로봇 외부의 노트북이나 데스크톱. RViz, SLAM, Navigation, Teleop 등을 실행한다.
- SBC (Single Board Computer): 로봇에 탑재된 메인 컴퓨터. OpenCR 및 센서와 직접 통신한다.
- OpenCR: 모터와 저수준 하드웨어를 제어하는 보드다.

현재 구성에서는 Jetson Orin Nano가 TurtleBot3의 SBC 역할을 한다. ROS 2 Desktop 및 Navigation 관련 패키지도 설치했으므로 일부 Remote PC 역할도 함께 수행할 수 있다.

## 2. 초기 rosdep 상태

다음 명령은 이미 실행된 상태였다.

```bash
sudo rosdep init
rosdep update
```

`sudo rosdep init` 실행 시 다음 메시지가 나왔다.

```text
ERROR: default sources list file already exists:
    /etc/ros/rosdep/sources.list.d/20-default.list
```

이는 오류로 재설정해야 하는 상태가 아니라 rosdep이 이미 초기화됐다는 의미다. `rosdep update`는 정상적으로 완료됐다.

## 3. 기존 ROS 상태 확인

확인된 초기 상태:

- Ubuntu 22.04.5 LTS
- `/opt/ros/humble` 존재
- `ros2` 실행 파일 존재
- `ros-humble-ros-base` 설치됨
- `ros-humble-desktop`은 처음에는 미설치
- `demo_nodes_cpp`는 처음에는 미설치

ROS 환경 적용 명령:

```bash
source /opt/ros/humble/setup.bash
```

## 4. PC Setup 관련 패키지 설치

다음 패키지들의 설치를 완료하고 설치 상태를 확인했다.

```bash
sudo apt-get update
sudo apt-get install -y \
  ros-humble-desktop \
  ros-humble-cartographer \
  ros-humble-cartographer-ros \
  ros-humble-navigation2 \
  ros-humble-nav2-bringup \
  ros-humble-slam-toolbox \
  python3-colcon-common-extensions
```

설치 완료 항목:

- ROS 2 Humble Desktop
- Cartographer
- Cartographer ROS
- Navigation2
- Nav2 Bringup
- SLAM Toolbox
- colcon 공통 확장

### Gazebo

Gazebo는 현재 단계에서 사용하지 않기로 결정했다. Jetson ARM64 환경에는 TurtleBot3 Humble 매뉴얼에서 사용하는 Gazebo Classic 패키지 구성이 일반 PC와 동일하게 제공되지 않았으며, 실제 로봇 구동에는 Gazebo가 필요하지 않다.

## 5. TurtleBot3 패키지 설치

다음 공식 저장소를 `humble` 브랜치로 내려받았다.

```text
~/turtlebot3_ws/src/DynamixelSDK
~/turtlebot3_ws/src/turtlebot3_msgs
~/turtlebot3_ws/src/turtlebot3
```

사용한 명령 형태:

```bash
mkdir -p ~/turtlebot3_ws/src
cd ~/turtlebot3_ws/src
git clone -b humble https://github.com/ROBOTIS-GIT/DynamixelSDK.git
git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git
git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3.git
```

Conda `(base)` 환경 때문에 CMake가 `/home/user/miniconda3/bin/python3`를 선택해 다음 오류가 발생했다.

```text
ModuleNotFoundError: No module named 'catkin_pkg'
```

이를 피하기 위해 빌드 시 시스템 Python `/usr/bin/python3`을 지정했다.

메모리 문제로 여러 번 `cc1plus`가 종료됐으며, 패키지를 하나씩 순차 빌드했다. 소스 빌드가 완료된 주요 패키지는 다음과 같다.

- `dynamixel_sdk`
- `dynamixel_sdk_custom_interfaces`
- `turtlebot3_msgs`
- `turtlebot3_cartographer`
- `turtlebot3_navigation2`
- `turtlebot3_description`
- `turtlebot3_teleop`
- `turtlebot3_example`

`turtlebot3_node` 소스 빌드는 반복적으로 강제 종료됐다. 최종적으로 공식 바이너리 패키지를 설치해 해결했다.

```bash
sudo apt install -y ros-humble-turtlebot3
```

다음 패키지의 prefix가 `/opt/ros/humble`로 정상 출력되는 것을 확인했다.

```bash
ros2 pkg prefix turtlebot3_node
ros2 pkg prefix turtlebot3_bringup
```

## 6. SBC Setup

SBC용 필수 패키지가 설치된 것을 확인했다.

- `python3-argcomplete`
- `python3-colcon-common-extensions`
- `libboost-system-dev`
- `build-essential`
- `ros-humble-hls-lfcd-lds-driver`
- `ros-humble-turtlebot3-msgs`
- `ros-humble-dynamixel-sdk`
- `ros-humble-xacro`
- `libudev-dev`

LDS-02용 공식 소스도 내려받았다.

```text
~/turtlebot3_ws/src/ld08_driver
~/turtlebot3_ws/src/coin_d4_driver
```

소스 빌드 대신 ARM64 공식 바이너리 패키지를 설치했다.

```bash
sudo apt install -y \
  ros-humble-ld08-driver \
  ros-humble-coin-d4-driver
```

OpenCR udev 규칙도 설치했다.

```bash
source /opt/ros/humble/setup.bash
sudo cp \
  "$(ros2 pkg prefix turtlebot3_bringup)/share/turtlebot3_bringup/script/99-turtlebot3-cdc.rules" \
  /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

확인된 규칙 파일:

```text
/etc/udev/rules.d/99-turtlebot3-cdc.rules
```

`.bashrc`에 다음 환경 변수를 추가했다.

```bash
export TURTLEBOT3_MODEL=burger
export LDS_MODEL=LDS-02
export ROS_DOMAIN_ID=30
```

적용 명령:

```bash
source ~/.bashrc
```

주의: 파일 이름은 `~/.bashrc`다. `~/.bashrcrc`는 오타다.

## 7. USB 장치 확인

연결됐을 때 확인된 장치:

```text
/dev/ttyACM0  OpenCR
/dev/ttyUSB0  LDS-02 USB2LDS 어댑터
/dev/input/js0  TGZ-850M 게임패드
```

사용자는 `dialout` 그룹에 포함되어 있다. OpenCR과 LDS 직렬 장치는 `root:dialout` 소유로 나타났다.

확인 명령:

```bash
ls -l /dev/ttyACM0 /dev/ttyUSB0 /dev/input/js0
```

현재 LDS-02는 의도적으로 분리해 놓았기 때문에 `/dev/ttyUSB0`가 없는 것이 정상이다.

## 8. OpenCR Burger 펌웨어 설치

OpenCR은 `/dev/ttyACM0`으로 정상 인식됐다.

공식 ARM 업로더가 32비트 ARM 실행 파일이어서 다음 의존성을 설치했다.

```bash
sudo dpkg --add-architecture armhf
sudo apt-get update
sudo apt-get install -y libc6:armhf
```

공식 펌웨어를 다음 위치에 준비했다.

```text
~/opencr_firmware/opencr_update/burger.opencr
```

업로드 명령:

```bash
cd ~/opencr_firmware/opencr_update
./update.sh /dev/ttyACM0 burger.opencr
```

업로드 결과:

```text
OpenCR R1.0
Firmware: burger V230127R1
[OK] flash_erase
[OK] flash_write
[OK] CRC Check
[OK] Download
[OK] jump_to_fw
```

업로드 후 OpenCR이 재부팅됐고 `/dev/ttyACM0`가 다시 정상 인식됐다.

## 9. OpenCR 전원 및 모터 테스트

처음에는 OpenCR에서 반복적인 경고음이 나고 SW1/SW2를 눌러도 모터가 움직이지 않았다.

원인은 배터리 전원이 없거나 방전되어 OpenCR이 USB 5V만 공급받던 상태였다. USB 전원만으로 MCU는 켜질 수 있지만 DYNAMIXEL 모터 구동 전원은 부족하다.

주의 사항:

- 220V AC를 OpenCR에 직접 연결하면 안 된다.
- 공식 권장 전원은 DC 12V 5A 정전압 SMPS다.
- 방전된 배터리와 SMPS를 동시에 연결하지 않는다.
- 배터리는 OpenCR에서 분리한 뒤 전용 충전기로 충전한다.

DC 12V 5A SMPS를 정상 연결한 뒤 다음 테스트가 성공했다.

- OpenCR SW1 길게 누르기: 전진 동작
- OpenCR SW2 길게 누르기: 회전 동작

따라서 OpenCR 펌웨어, 모터 배선 및 기본 DYNAMIXEL 설정이 정상임을 확인했다.

## 10. TGZ-850M 게임패드

TGZ-850M은 Bluetooth가 아니라 전용 USB 수신기를 사용하는 2.4 GHz 게임패드다.

페어링 방법:

1. USB 수신기를 Jetson에 연결한다.
2. 컨트롤러 배터리를 확인한다.
3. 컨트롤러 스위치를 `ON`으로 켠다.
4. 자동 연결을 기다린다.
5. 연결되지 않으면 `HOME` 버튼을 빠르게 두 번 누른다.
6. 빨간 LED가 고정 점등되면 연결 성공이다.

일부 유통 버전은 페어링 성공 시 짧은 확인음을 낼 수 있다. 공식 설명서에는 빨간 LED 고정 점등이 성공 표시로 명시되어 있다.

Linux에서 다음 장치로 정상 인식됐다.

```text
crw-rw-r--+ 1 root input ... /dev/input/js0
```

필요한 ROS 패키지도 설치되어 있다.

- `ros-humble-joy`
- `ros-humble-teleop-twist-joy`

입력 확인 명령:

첫 번째 터미널:

```bash
source /opt/ros/humble/setup.bash
ros2 run joy joy_node --ros-args -p device_id:=0
```

두 번째 터미널:

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /joy
```

스틱과 버튼을 움직일 때 `/joy`의 `axes`와 `buttons` 값이 정상적으로 변하는 것을 확인했다.

## 11. LiDAR 없이 TurtleBot3 Bringup

기본 `robot.launch.py`는 LDS 드라이버를 항상 실행한다. LDS-02를 분리한 상태에서 실행하면 `ld08_driver`가 남아 있는 OpenCR 포트를 잘못 시도하면서 다음 오류가 발생했다.

```text
/dev/ttyACM0 OpenCR Virtual ComPort in FS Mode
Can't find LDS-02
```

LiDAR 없이 OpenCR, 모터 노드 및 Robot State Publisher만 실행하기 위해 전용 launch 파일을 만들었다.

```text
~/turtlebot3_no_lidar.launch.py
```

실행 명령:

```bash
source ~/.bashrc
ros2 launch ~/turtlebot3_no_lidar.launch.py
```

검증 결과:

```text
Succeeded to open the port(/dev/ttyACM0)!
Succeeded to change the baudrate!
Start Calibration of Gyro
Calibration End
Add Motors
Add Wheels
Succeeded to create battery state publisher
Succeeded to create imu publisher
Succeeded to create sensor state publisher
Succeeded to create joint state publisher
Run!
diff_drive_controller: Run!
```

따라서 LiDAR 없이 OpenCR 및 모터 제어 노드를 실행할 수 있는 상태다.

## 12. 게임패드 Teleop 실행 방법

안전을 위해 처음에는 바퀴를 공중에 띄우고 낮은 속도로 테스트한다.

### 터미널 1: LiDAR 없는 TurtleBot3 Bringup

```bash
source ~/.bashrc
ros2 launch ~/turtlebot3_no_lidar.launch.py
```

### 터미널 2: 게임패드 입력 노드

```bash
source ~/.bashrc
ros2 run joy joy_node --ros-args -p device_id:=0
```

### 터미널 3: Joy 입력을 `/cmd_vel`로 변환

```bash
source ~/.bashrc
ros2 run teleop_twist_joy teleop_node --ros-args \
  -p axis_linear.x:=1 \
  -p scale_linear.x:=0.08 \
  -p axis_angular.yaw:=3 \
  -p scale_angular.yaw:=0.6 \
  -p enable_button:=0 \
  -p require_enable_button:=true
```

예상 조작:

- A 버튼을 누른 상태에서 왼쪽 스틱 위/아래: 전진/후진
- A 버튼을 누른 상태에서 오른쪽 스틱 좌/우: 회전
- A 버튼을 놓으면 정지

축 번호가 실제 컨트롤러 매핑과 다르면 `/joy` 출력에서 변하는 `axes` 인덱스를 확인해 `axis_linear.x`와 `axis_angular.yaw`를 수정한다.

전송 확인:

```bash
ros2 topic echo /cmd_vel
```

## 13. 메모리 문제 및 악성코드 발견

빌드와 Bringup 과정에서 다음 증상이 반복됐다.

```text
c++: fatal error: Killed signal terminated program cc1plus
Killed
```

초기에는 일반적인 메모리 부족으로 판단했으나 프로세스를 상세 점검한 결과, 다음 숨김 실행 파일이 `systemd`, `bash`, `nginx: worker process` 등으로 위장해 실행되고 있었다.

```text
/home/user/.GNFqecmXVp
```

파일 정보:

```text
ELF 64-bit ARM aarch64
statically linked
UPX 형태의 메모리 매핑
SHA-256: dbb7ebb960dc0d5a480f97ddde3a227a2d83fcaca7d37ae672e6a0a6785631e9
```

해당 SHA-256은 공개 악성코드 분석에서 ARM64 암호화폐 채굴 악성코드로 판정된 샘플과 정확히 일치했다.

악성 프로세스는 약 2.4 GB의 메모리를 점유했으며 외부 IP와 연결하고 있었다.

확인된 crontab 지속성:

```text
@reboot /home/user/.GNFqecmXVp
```

수행한 조치:

- 악성 프로세스 종료
- 외부 연결 종료
- 사용자 crontab의 악성 자동실행 제거
- 재생성된 악성 파일을 실행 불가능하게 설정
- 파일을 삭제하지 않고 `~/quarantine`으로 이동

격리 파일:

```text
~/quarantine/GNFqecmXVp.coinminer.sha256-dbb7ebb960dc0d5a
~/quarantine/GNFqecmXVp.coinminer.respawned-20260808-1446
```

조치 후 사용 가능 메모리는 약 5.3 GB까지 회복됐고 LiDAR 없는 TurtleBot3 Bringup이 정상적으로 완료됐다.

### 필수 보안 후속 조치

이 장치는 이미 침해된 것으로 간주해야 한다. 현재 확인된 프로세스와 자동 시작 항목은 제거했지만 다른 백도어 또는 변경 사항이 없다고 보장할 수 없다.

권장 순서:

1. 중요한 ROS 소스와 설정 파일만 백업한다.
2. 신뢰할 수 있는 공식 Jetson 이미지로 운영체제를 재설치한다.
3. 이 장치에서 사용했던 로그인 비밀번호를 다른 안전한 장치에서 변경한다.
4. SSH 키를 폐기하고 새로 발급한다.
5. 외부에 공개된 SSH 포트와 공유기 포트포워딩을 확인한다.
6. 재설치 후 Ubuntu 및 JetPack 보안 업데이트를 적용한다.
7. 격리 파일을 다른 시스템에서 실행하거나 압축 해제하지 않는다.

## 14. 현재 최종 상태

- ROS 2 Humble: 정상
- ROS 2 Desktop: 설치 완료
- Cartographer: 설치 완료
- Navigation2: 설치 완료
- SLAM Toolbox: 설치 완료
- TurtleBot3 바이너리 패키지: 설치 완료
- OpenCR Burger 펌웨어: 업로드 완료 및 검증 성공
- OpenCR USB: `/dev/ttyACM0` 정상
- 모터: OpenCR SW1/SW2 하드웨어 테스트 성공
- TGZ-850M: `/dev/input/js0` 인식 및 `/joy` 데이터 확인 성공
- LDS-02: 현재 의도적으로 분리
- LiDAR 없는 Bringup: 정상 검증 완료
- 게임패드 Teleop: 실행 준비 완료
- Gazebo: 의도적으로 설치하지 않음
- 악성코드: 활성 프로세스 및 crontab 제거, 파일 격리 완료

## 15. 바로 이어서 할 작업

1. 바퀴를 공중에 띄운다.
2. `turtlebot3_no_lidar.launch.py`를 실행한다.
3. `joy_node`를 실행한다.
4. `teleop_twist_joy`를 낮은 속도로 실행한다.
5. `/cmd_vel`과 바퀴 반응을 확인한다.
6. 조이스틱 축 또는 버튼 매핑이 다르면 `/joy` 데이터를 기준으로 수정한다.
7. 기능 확인 후 운영체제 재설치 및 계정 보안 조치를 우선 수행한다.

