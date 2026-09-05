# TurtleBot3 실행 소스 묶음

전체 실행에 필요한 원본 소스와 설정을 이 폴더에 복사해 뒀습니다.

- `ros2_ws/src/` — P5U 소리 이상감지 ROS 2 패키지, 추론 모델, 설정
- `turtlebot3_ws/src/` — TurtleBot3/OpenCR LED, LiDAR, Dynamixel 관련 ROS 2 소스
- `desktop/` — 전체 실행 아이콘과 소리·LED 사용법 문서

실제 실행 스크립트와 연동 launch는 이 저장소 루트의 `scripts/`와 `ros2_bridge/`에
이미 있습니다. 지금 실행 환경은 기존 `~/ros2_ws`와 `~/turtlebot3_ws`의 빌드
설치본을 씁니다.

`build/`, `install/`, `log/`, Python 가상환경과 `python_deps/`는 기기마다 다시
만들어지는 것이라 복사하지 않았습니다. 소리감지에 필요한 Python 패키지 목록은
`ros2_ws/src/dyeun_robotics/sound_anomaly/requirements-inference.txt`에 있다.

지금 P5U 설정은 `ros2_ws/src/dyeun_robotics/sound_anomaly/config/sound_anomaly.yaml`의
`audio_device: "USB Microphone: Audio (hw:1,0)"` 입니다. OpenCR 포트는 연결 순서에
따라 바뀔 수 있는데, 지금은 `/dev/ttyACM1` 입니다.
