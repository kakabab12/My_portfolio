"""C270 웹캠 마이크 이상감지를 TurtleBot3 OpenCR LED와 함께 실행한다."""
from __future__ import annotations

import os
from pathlib import Path

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    # P5U는 열지 않는다. PortAudio 장치 0은 연결된 C270 웹캠 마이크 중 첫 번째
    # 입력이며, USB 재연결 뒤에도 동일한 웹캠 경로를 우선 선택한다. 필요하면
    # SOUND_AUDIO_DEVICE 환경 변수로 명시적인 장치를 시험할 수 있다.
    audio_device = os.environ.get("SOUND_AUDIO_DEVICE", "0")
    sound_workspace = Path.home() / "ros2_ws"
    source_config = (
        sound_workspace / "src" / "dyeun_robotics" / "sound_anomaly"
        / "config" / "sound_anomaly.yaml"
    )
    dependency_path = sound_workspace / "python_deps"
    python_path = str(dependency_path) + os.pathsep + os.environ.get("PYTHONPATH", "")
    return LaunchDescription([
        Node(
            package="sound_anomaly",
            executable="sound_anomaly_node",
            name="sound_anomaly_node",
            output="screen",
            parameters=[
                str(source_config),
                {
                    # model_path를 별도로 덮어쓰지 않는다. 따라서 바탕화면의
                    # sound_anomaly_led.sh와 동일하게 sound_anomaly 패키지에서
                    # 현재 활성으로 설정한 모델을 사용한다.
                    "audio_device": audio_device,
                    # 웹캠 마이크의 native rate를 사용하고 모델 입력 16 kHz로
                    # 노드 안에서 변환한다. Nav2·제스처·RViz와 함께 실행해도
                    # PortAudio callback 여유가 생기도록 4096(약 85 ms) 대신
                    # 48000(약 1초) 샘플 단위로 받는다. 판정 창은 3초이고
                    # 1초마다 판정하므로 이 증가는 사용자 체감 지연을 만들지 않는다.
                    "capture_sample_rate": 0,
                    "audio_blocksize": 48000,
                    # 최초 librosa/Numba 준비와 일시적인 시스템 부하 동안에도
                    # 입력을 보존한다. 48 kHz mono에서 약 16초 분량이며 3 MiB 미만이다.
                    "audio_queue_max_blocks": 16,
                    "audio_process_period_sec": 0.25,
                },
            ],
            additional_env={
                "PYTHONPATH": python_path,
                # 모델/오디오 라이브러리 초기화 실패가 launch 화면에 즉시 보이게 한다.
                "PYTHONUNBUFFERED": "1",
            },
            # 장치명/USB 오류에서 2초마다 다시 열기를 반복하면 허브와 CPU 부하가
            # 누적된다. 오류는 통합 런처가 감지해 전체 구성을 안전하게 종료한다.
            respawn=False,
        ),
    ])
