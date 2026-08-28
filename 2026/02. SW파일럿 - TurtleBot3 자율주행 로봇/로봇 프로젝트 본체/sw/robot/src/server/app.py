"""Flask 앱 — 제스처 엔진의 HTTP 접점 (ros2_bridge/cmd_vel_bridge.py가 폴링).

GET /cmd        -> 최신 속도 명령 JSON
GET /video_feed -> MJPEG 디버그 스트림 (모니터 없는 젯슨을 브라우저로 확인)
GET /health     -> 생존 확인
GET /           -> 위 내용을 한 페이지에서 보는 간단한 상태 페이지

주의: 파이프라인은 실제 카메라·모델을 여는 무거운 작업이라 프로세스당 정확히
한 번만 시작해야 한다 — `app.run(debug=True)`의 자동 리로더는 프로세스를
다시 실행해 카메라를 중복으로 열려 하므로 절대 켜지 말 것.
"""
import logging
import os
import threading
import time

import cv2
from flask import Flask, Response, jsonify, request

from src.pipeline.gesture_loop import run_pipeline
from src.utils.config_loader import load_config
from src.utils.logger import get_logger, init_logging

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")
# 대안 config로 띄우고 싶을 때(예: 원격 카메라 스트림 테스트) — 평소엔 안 씀
CONFIG_PATH = os.environ.get("GESTURE_CONFIG_PATH", DEFAULT_CONFIG_PATH)

config = load_config(CONFIG_PATH)
init_logging(config)
logger = get_logger("server")

app = Flask(__name__)
app.config["PIPELINE_STATE"] = None
_state_lock = threading.Lock()
_control_mode_lock = threading.Lock()
_initial_control_mode = os.environ.get("GESTURE_INITIAL_CONTROL_MODE", "unknown")
_control_mode = (
    _initial_control_mode
    if _initial_control_mode in {"gesture", "joystick", "auto", "unknown"}
    else "unknown")
_control_mode_updated_sec = time.monotonic()
_mission_status_lock = threading.Lock()
_mission_status = {
    "phase": "waiting",
    "label": "순찰 시작 대기",
    "target": "-",
    "checkpoints": {"A": "대기", "B": "대기", "C": "대기", "D": "대기"},
    "position": None,
}
_mission_status_updated_sec = time.monotonic()
_start_sec = time.monotonic()

CONTROL_MODE_LABELS = {
    "gesture": "제스처 모드",
    "joystick": "컨트롤러 모드",
    "auto": "자율주행 모드",
    "unknown": "모드 확인 중",
}


def _control_mode_snapshot():
    with _control_mode_lock:
        return {
            "mode": _control_mode,
            "label": CONTROL_MODE_LABELS[_control_mode],
            "age_sec": time.monotonic() - _control_mode_updated_sec,
        }


def _mission_status_snapshot():
    with _mission_status_lock:
        return {
            **_mission_status,
            "checkpoints": dict(_mission_status["checkpoints"]),
            "position": (dict(_mission_status["position"])
                         if _mission_status["position"] is not None else None),
            "age_sec": time.monotonic() - _mission_status_updated_sec,
        }

def _status_page():
    camera_cards = "".join(
        f'''<section class="camera-card">
              <h3>Camera {index} <span>/dev/video{device_id}</span></h3>
              <img src="/camera_feed/{index}" alt="Camera {index}">
            </section>'''
        for index, device_id in enumerate(config["camera"]["devices"])
    )
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>gesture_engine</title>
<style>
body {{ font-family: sans-serif; margin: 20px; background:#111; color:#eee; }}
.camera-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }}
.camera-card {{ padding:10px; background:#1b1b1b; border:1px solid #444; border-radius:8px; }}
.camera-card h3 {{ margin:0 0 8px; font-size:16px; }}
.camera-card span {{ color:#aaa; font-weight:normal; }}
.camera-card img {{ display:block; width:100%; aspect-ratio:16/9; object-fit:contain; background:#000; }}
#cmd {{ padding:12px; background:#1b1b1b; border-radius:8px; }}
#control-mode {{ padding:12px; background:#1b1b1b; border:1px solid #5c8; border-radius:8px;
                 font-size:20px; font-weight:bold; }}
#mission {{ padding:12px; background:#1b1b1b; border:1px solid #58c; border-radius:8px;
            white-space:pre-line; }}
@media (max-width:900px) {{ .camera-grid {{ grid-template-columns:1fr; }} }}
</style>
</head><body>
<h2>Jetson gesture_engine — cameras</h2>
<div class="camera-grid">{camera_cards}</div>
<h3>Control mode</h3>
<div id="control-mode">loading...</div>
<h3>Mission / checkpoints</h3>
<div id="mission">loading...</div>
<h3>Gesture command</h3>
<pre id="cmd">loading...</pre>
<script>
async function poll() {{
  try {{
    const [cmdRes, modeRes, missionRes] = await Promise.all([
      fetch('/cmd'), fetch('/control_mode'), fetch('/mission_status')]);
    document.getElementById('cmd').textContent = JSON.stringify(await cmdRes.json(), null, 2);
    const mode = await modeRes.json();
    document.getElementById('control-mode').textContent = mode.label;
    const mission = await missionRes.json();
    const checkpoints = Object.entries(mission.checkpoints)
      .map(([name, state]) => `${{name}}: ${{state}}`).join(' / ');
    const position = mission.position
      ? `현재 좌표: (${{mission.position.x.toFixed(3)}}, ${{mission.position.y.toFixed(3)}})`
      : '현재 좌표: 수신 대기';
    document.getElementById('mission').textContent =
      `${{mission.label}}\n목표: ${{mission.target}}\n${{checkpoints}}\n${{position}}`;
  }} catch (e) {{}}
}}
setInterval(poll, 200);
poll();
</script>
</body></html>"""


def _state():
    """파이프라인 상태를 지연 초기화(첫 요청 시 1회)한다 — 모듈 임포트만으로
    카메라·모델이 열리지 않게 해 테스트·정적 분석 시 부작용이 없다."""
    state = app.config["PIPELINE_STATE"]
    if state is not None:
        return state
    with _state_lock:
        state = app.config["PIPELINE_STATE"]
        if state is None:
            state = run_pipeline(config)
            app.config["PIPELINE_STATE"] = state
        return state


@app.route("/")
def index():
    return _status_page()


@app.route("/cmd")
def cmd():
    return jsonify(_state().snapshot())


@app.route("/health")
def health():
    state = _state()
    return jsonify({
        "status": "ok",
        "uptime_sec": time.monotonic() - _start_sec,
        "cameras": state.camera_status(config["camera"]["devices"]),
        "inference": state.inference_status(),
    })


@app.route("/control_mode", methods=["GET", "POST"])
def control_mode():
    """현재 mux 제어권. GET은 웹 화면용, POST는 ROS 브리지의 상태 보고용."""
    global _control_mode, _control_mode_updated_sec
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        mode = payload.get("mode")
        if mode not in CONTROL_MODE_LABELS:
            return jsonify({"error": "mode must be gesture, joystick, auto, or unknown"}), 400
        with _control_mode_lock:
            _control_mode = mode
            _control_mode_updated_sec = time.monotonic()
    return jsonify(_control_mode_snapshot())


@app.route("/mission_status", methods=["GET", "POST"])
def mission_status():
    """웨이포인트 전환 노드가 올리는 도착 상태와 현재 AMCL 좌표."""
    global _mission_status, _mission_status_updated_sec
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        phase = payload.get("phase")
        label = payload.get("label")
        target = payload.get("target")
        checkpoints = payload.get("checkpoints")
        position = payload.get("position")
        if not all(isinstance(value, str) for value in (phase, label, target)):
            return jsonify({"error": "phase, label, and target must be strings"}), 400
        if (not isinstance(checkpoints, dict)
                or set(checkpoints) != {"A", "B", "C", "D"}
                or not all(isinstance(value, str) for value in checkpoints.values())):
            return jsonify({"error": "checkpoints must contain A, B, C, and D strings"}), 400
        if position is not None and (
                not isinstance(position, dict)
                or not all(isinstance(position.get(axis), (int, float))
                           for axis in ("x", "y"))):
            return jsonify({"error": "position must contain numeric x and y"}), 400
        with _mission_status_lock:
            _mission_status = {
                "phase": phase,
                "label": label,
                "target": target,
                "checkpoints": dict(checkpoints),
                "position": (dict(position) if position is not None else None),
            }
            _mission_status_updated_sec = time.monotonic()
    return jsonify(_mission_status_snapshot())


def _mjpeg_generator(camera_index=None):
    """시청 중에만 add_viewer()로 등록 — gesture_loop가 그동안만 오버레이를
    그린다(모듈독스트링 참고). 연결이 끊기면(브라우저 닫기 등) Flask가 이
    제너레이터를 close()하면서 finally가 실행돼 반드시 등록이 풀린다."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), config["server"].get("jpeg_quality", 70)]
    boundary = b"--frame"
    state = _state()
    state.add_viewer()
    try:
        while True:
            frame = (state.get_frame() if camera_index is None
                     else state.get_camera_frame(camera_index))
            if frame is None:
                time.sleep(0.05)
                continue
            ok, buffer = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                continue
            yield (boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n"
                  + buffer.tobytes() + b"\r\n")
            time.sleep(0.03)
    finally:
        state.remove_viewer()


@app.route("/video_feed")
def video_feed():
    if not config["server"].get("video_feed_enabled", True):
        return "video_feed disabled in config", 404
    return Response(_mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/camera_feed/<int:camera_index>")
def camera_feed(camera_index):
    """설정 순서 기준 카메라별 MJPEG 스트림을 HTML에 동시에 표시한다."""
    if not config["server"].get("video_feed_enabled", True):
        return "video_feed disabled in config", 404
    if camera_index < 0 or camera_index >= len(config["camera"]["devices"]):
        return "unknown camera index", 404
    return Response(_mjpeg_generator(camera_index),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


def main():
    _state()   # 서버가 요청을 받기 전에 카메라·모델을 미리 띄워 초기 지연·오류를 앞당긴다
    # ROS 브리지가 /cmd를 30Hz로 읽으므로 요청마다 로그를 쓰면 CPU·디스크 I/O가
    # 인식 루프를 방해한다. 경고·오류만 남겨 제스처 전달 지연을 줄인다.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    host = config["server"].get("host", "0.0.0.0")
    port = config["server"].get("port", 5000)
    logger.info("gesture_engine Flask 서버 시작 %s:%d", host, port)
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
