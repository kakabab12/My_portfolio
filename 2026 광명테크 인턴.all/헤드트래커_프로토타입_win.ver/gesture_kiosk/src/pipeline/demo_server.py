"""예시 UI 서버 — 회사 키오스크 프로그램이 들어올 자리의 임시 대체물.

TODO(기획서 9장 №7·№8): 회사 프로그램(UI) 파일을 받으면 이 서버와 demo_ui/는
제거하고, event_sender.py 규격으로 이벤트만 전달한다.

회사 프로그램 연동 계약(이 서버가 시연하는 것):
- 이벤트(엔진→UI): /data 폴링(events) 또는 event_output(udp) — select·go_back
  (config classes 목록). 커서 좌표(cursor_x_ratio/cursor_y_ratio)는 /data status에만
  실리는 데모 UI 전용 필드다 — UDP 계약은 이산 이벤트만 유지한다(연속 좌표를 매 프레임
  이벤트 로그에 넣지 않는다). 회사 UI가 실시간 커서가 필요하면 별도 협의 필요(TODO).
- 음성 안내(UI→엔진): POST /announce {"text": "발급하기 버튼"} — 포커스 항목
  설명처럼 화면 구조를 아는 쪽(UI)이 문구를 만들어 엔진 TTS로 읽힌다
- 커서 입력 모드 전환(UI→엔진, 데모 전용): POST /set_cursor_mode {"mode": "head"|"eye"} —
  cursor_x_ratio/y_ratio와 같은 성격의 데모 UI 전용 채널이라 회사 연동 계약에는
  포함하지 않는다(회사 UI가 모드 전환이 필요하면 별도 협의)
(2026-07-16: 주민등록증 OCR 기능 제거 — 제스처 집중, /ocr/* 엔드포인트 삭제)
(2026-07-18: 팔 쓸기 → 헤드트래커 전환 — classes가 move_left/right·go_home 제외,
 select·go_back 2종으로 축소, 커서 필드 신설)
(2026-07-22: 아이트래킹(시선) 커서 모드 추가 — 데모 UI 버튼으로 head_tracker의 코/시선
 입력을 전환. POST /set_cursor_mode 신설, /data status에 cursor_mode 필드 추가)
"""
import asyncio
import os
from typing import Literal

import cv2
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from src.postprocess.head_tracker import CURSOR_MODE_EYE, CURSOR_MODE_HEAD

try:
    import psutil
except ImportError:  # psutil 미설치 환경(개발 PC)에서도 서버는 떠야 한다
    psutil = None

DEMO_UI_HTML = "demo_ui/index.html"
RECENT_EVENT_COUNT = 20


class AnnounceBody(BaseModel):
    text: str


class SetCursorModeBody(BaseModel):
    mode: Literal[CURSOR_MODE_HEAD, CURSOR_MODE_EYE]


def create_app(state, config):
    app = FastAPI(title="Gesture Kiosk Demo UI (예시 — 회사 프로그램 대체 예정)")
    index_path = os.path.join(config["root_dir"], DEMO_UI_HTML)
    stream_interval_sec = 1.0 / config["demo_ui"]["stream_fps"]
    jpeg_quality = config["demo_ui"]["jpeg_quality"]

    @app.get("/")
    async def serve_index():
        return FileResponse(index_path)

    async def _stream():
        while True:
            frame = state.get_frame()
            if frame is None:
                await asyncio.sleep(0.05)
                continue
            ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if not ret:
                await asyncio.sleep(0.05)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            await asyncio.sleep(stream_interval_sec)

    @app.get("/video_feed")
    async def video_feed():
        return StreamingResponse(
            _stream(), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.get("/data")
    async def get_data():
        events = []
        for e in state.event_log[-RECENT_EVENT_COUNT:]:
            item = {"class_name": e.class_name, "conf": round(e.conf, 2), "ts_sec": e.ts_sec}
            if e.data is not None:
                item["data"] = e.data
            events.append(item)
        return {
            "stats": {
                "cpu": psutil.cpu_percent(interval=None) if psutil else 0.0,
                "memory": psutil.virtual_memory().percent if psutil else 0.0,
                "capture_fps": round(state.capture_fps, 1),
                "infer_fps": round(state.infer_fps, 1),
            },
            "status": {
                "is_user_locked": state.is_user_locked,
                "cursor_x_ratio": state.cursor_x_ratio,
                "cursor_y_ratio": state.cursor_y_ratio,
                "cursor_mode": state.cursor_mode,
            },
            "debug": state.debug,   # 판정 계기판 — 실기 튜닝용 (연동 계약 아님)
            "classes": config["classes"],
            "events": events,
        }

    # ----- 데모 UI 전용 (회사 프로그램 인수 시 함께 제거 — cursor_x/y_ratio와 같은 성격) -----

    @app.post("/set_cursor_mode")
    async def set_cursor_mode(body: SetCursorModeBody):
        """헤드트래커(코)/아이트래킹(시선) 커서 입력 모드 전환 — 데모 UI 버튼이 호출한다."""
        state.head_tracker.set_cursor_mode(body.mode)
        return {"ok": True}

    # ----- 회사 프로그램 연동 계약 엔드포인트 -----

    @app.post("/announce")
    async def announce(body: AnnounceBody):
        """UI가 화면 맥락(포커스 항목 등)을 음성 안내로 요청한다."""
        state.announcer.announce(body.text)
        return {"ok": True}


    return app
