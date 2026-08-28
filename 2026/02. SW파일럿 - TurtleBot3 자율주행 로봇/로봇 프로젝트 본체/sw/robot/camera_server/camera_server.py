"""카메라 서버 — 로컬 웹캠을 MJPEG로 네트워크에 노출한다 (노트북 등에서 실행).

젯슨에서 모든 처리(손 인식·D-pad·잠금 토글·ROS2 브리지)를 하고 카메라만
이 컴퓨터(예: 노트북) 것을 쓰고 싶을 때 쓴다. **mediapipe 등 인식 관련
의존성이 전혀 없다** — opencv-contrib-python + flask만 있으면 된다
(requirements-camera-server.txt). 손 인식·판정 로직은 이 파일에 없고
전부 젯슨의 src.server.app 쪽에서 처리한다 — 여기는 순수 영상 전달만 한다.

src.capture.camera_stream의 CameraStream/init_camera를 그대로 재사용한다
(로컬 장치 캡처·자동 복구 로직 중복 구현 방지) — 이 파일에서 필요한 것은
config dict 형태뿐이라 YAML 없이 커맨드라인 인자로 즉석에서 만든다.

사용법:
    python camera_server.py --devices 0 --port 8090
    python camera_server.py --devices 0,1 --port 8090   (카메라 2대)

젯슨의 configs/config.yaml에서 camera.devices에 이 서버가 실제로 여는
장치 번호와 같은 이름의 스트림 URL을 적으면 된다 — 예:
    devices: ["http://<노트북IP>:8090/cam/0/stream"]
"""
import argparse
import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import cv2
from flask import Flask, Response, jsonify

from src.capture.camera_stream import CameraStream
from src.utils.logger import get_logger, init_logging

logger = get_logger("camera_server")

app = Flask(__name__)
_streams = {}   # device_id(int) -> CameraStream
_jpeg_quality = 80


def _build_config(args):
    """CameraStream이 요구하는 config dict를 인자로 즉석에서 구성한다."""
    return {
        "camera": {
            "width_px": args.width,
            "height_px": args.height,
            "fourcc": args.fourcc,
            "recovery_timeout_sec": 3.0,
            "recovery_retry_sec": 2.0,
        },
        "logging": {"level": "INFO", "save_dir": os.path.join(ROOT_DIR, "logs")},
    }


def _mjpeg_generator(stream):
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), _jpeg_quality]
    boundary = b"--frame"
    last_seq = 0
    while True:
        try:
            frame, last_seq = stream.capture_new_frame(last_seq)
        except RuntimeError:
            time.sleep(0.1)
            continue
        ok, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not ok:
            continue
        yield (boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n"
              + buffer.tobytes() + b"\r\n")


@app.route("/")
def index():
    links = "".join(f'<li><a href="/cam/{d}/stream">/cam/{d}/stream</a></li>'
                    for d in _streams)
    return f"<h3>camera_server</h3><ul>{links}</ul>"


@app.route("/health")
def health():
    return jsonify({"status": "ok", "devices": list(_streams.keys())})


@app.route("/cam/<int:device_id>/stream")
def cam_stream(device_id):
    stream = _streams.get(device_id)
    if stream is None:
        return f"device {device_id} not available", 404
    return Response(_mjpeg_generator(stream),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


def main():
    global _jpeg_quality
    parser = argparse.ArgumentParser(description="로컬 웹캠을 MJPEG로 네트워크에 노출")
    parser.add_argument("--devices", default="0",
                        help="쉼표로 구분한 카메라 장치 번호 (예: 0,1)")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fourcc", default="mjpg")
    parser.add_argument("--jpeg-quality", type=int, default=80)
    args = parser.parse_args()
    _jpeg_quality = args.jpeg_quality

    config = _build_config(args)
    init_logging(config)

    device_ids = [int(d.strip()) for d in args.devices.split(",") if d.strip()]
    started_count = 0
    for device_id in device_ids:
        try:
            _streams[device_id] = CameraStream(config, device_id=device_id).start()
            started_count += 1
        except RuntimeError as error:
            logger.warning("카메라 시작 실패(device_id=%s) — 건너뜀: %s", device_id, error)
    if started_count == 0:
        raise RuntimeError("카메라를 하나도 열지 못했습니다 — 장치 연결/번호를 확인하세요")

    logger.info("camera_server 시작 %s:%d (장치: %s)", args.host, args.port,
               list(_streams.keys()))
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
