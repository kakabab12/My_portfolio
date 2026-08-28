"""이벤트 전송 — 확정된 제스처 이벤트를 회사 키오스크 프로그램으로 넘기는 접점.

회사 GUI = **델파이7(Delphi 7) 네이티브 프로그램**. 연동 = **파이프(stdio)** 확정
(2026-07-23 — 회사 요청: 이벤트를 print만 하면 된다): 엔진이 stdout에 텍스트
한 줄을 찍으면, 델파이가 엔진을 자식 프로세스로 띄워 익명 파이프로 줄 단위
수신한다 (docs/델파이7_연동가이드.md — CreateProcess + ReadFile 샘플).
네트워크 연동(UDP·웹소켓)은 2026-07-23 전면 철회 — 포트·방화벽 이슈 소멸.

주의: 로그는 stderr·파일로만 나간다(logger.py의 StreamHandler 기본 = stderr) —
stdout은 이벤트 전용 채널이라 다른 출력이 섞이면 델파이 파서가 오염된다.
"""
import sys

from src.utils.logger import get_logger

logger = get_logger("pipeline")


def build_text_payload(gesture_event):
    """델파이7용 텍스트 규격 — `GESTURE|이벤트|손|신뢰도|시각` 한 줄(ASCII) 바이트.

    구분자 '|'는 값에 절대 등장하지 않는다(이벤트명·손은 영문 고정, 숫자 2종).
    delphi Pos/Copy 또는 4번의 구분자 분리로 파싱 가능. CRLF(#13#10)로 끝나
    수신기가 줄 단위로 읽어도 된다.

    2026-07-30 헤드트래커 병합: cursor 이벤트만 **뒤에 x·y 두 칸을 더 붙인다** —
    `GESTURE|cursor||1.00|123.456|0.512|0.487` (화면 폭·높이 대비 0~1 비율).
    앞 5칸의 의미·순서는 그대로라 기존 7종 이벤트만 읽는 수신부는 고칠 필요가 없다
    (구분자로 자른 뒤 앞에서부터 쓰면 남는 칸은 무시된다) — 커서를 그리려는
    수신부만 6·7번째 칸을 추가로 읽으면 된다.
    """
    payload = "GESTURE|{}|{}|{:.2f}|{:.3f}".format(
        gesture_event.class_name,
        gesture_event.hand_side or "",
        gesture_event.conf,
        gesture_event.ts_sec,
    )
    data = gesture_event.data or {}
    if "cursor_x_ratio" in data:
        payload += "|{:.3f}|{:.3f}".format(data["cursor_x_ratio"], data["cursor_y_ratio"])
    return (payload + "\r\n").encode("ascii")


def build_name_payload(gesture_event):
    """간소 규격(2026-07-31 사용자 요청) — 이벤트명 한 줄만 (예: b"left\\r\\n").

    cmd 창 확인·간단한 수신기에 맞춘 최소 출력 — 손·신뢰도·시각이 필요하면
    config event_output.format을 text(GESTURE| 규격)로 되돌린다.

    2026-08-04 예외(사용자 요청): cursor 이벤트는 이름("cursor")만 찍으면
    위치 정보가 없어 수신부가 커서를 그릴 수 없다 — x·y 비율 두 칸만
    "0.512|0.487" 형식으로 대신 찍는다(text 규격과 동일 소수점 3자리·구분자).
    """
    data = gesture_event.data or {}
    if gesture_event.class_name == CURSOR_CLASS_NAME and "cursor_x_ratio" in data:
        return "{:.3f}|{:.3f}\r\n".format(
            data["cursor_x_ratio"], data["cursor_y_ratio"]
        ).encode("ascii")
    return (gesture_event.class_name + "\r\n").encode("ascii")


PAYLOAD_BUILDERS = {"text": build_text_payload, "name": build_name_payload}

CURSOR_CLASS_NAME = "cursor"


def _is_loggable(gesture_event):
    """cursor는 매 프레임 나가는 연속 신호라 로그에서 뺀다 (2026-07-30 헤드트래커 병합).

    초당 수십 줄이 쌓이면 로그 파일이 커서 좌표로 덮여 정작 확정 이벤트(select·home 등)를
    실기 로그에서 찾을 수 없게 된다 — 커서는 디버그 창 계기판으로 본다.
    """
    return gesture_event.class_name != CURSOR_CLASS_NAME


class ConsoleEventSender:
    """개발용 — 이벤트를 로그로만 기록한다 (stdout 미사용)."""

    def send(self, gesture_event):
        if not _is_loggable(gesture_event):
            return
        logger.info(
            "event_output(console): %s (conf=%.2f)",
            gesture_event.class_name,
            gesture_event.conf,
        )


class StdioEventSender:
    """파이프(stdio) 연동 — 이벤트 한 줄을 stdout에 쓰고 즉시 flush한다.

    flush가 핵심: stdout이 파이프에 물리면 블록 버퍼링으로 바뀌어, flush 없이는
    버퍼(수 KB)가 찰 때까지 델파이에 아무것도 도착하지 않는다 (이벤트는 드문드문
    발생하므로 사실상 무한 지연). 바이트로 직접 써서 인코딩 변수도 제거한다.
    """

    def __init__(self, build_payload=build_text_payload):
        self._build_payload = build_payload

    def send(self, gesture_event):
        sys.stdout.buffer.write(self._build_payload(gesture_event))
        sys.stdout.buffer.flush()
        if not _is_loggable(gesture_event):
            return
        # INFO 기록은 파일 로그 몫 — 콘솔은 WARNING부터라 이벤트 한 줄만 보인다
        # (2026-07-31 사용자 요청: cmd 창 중복 출력 제거, logger.py console_level)
        logger.info("event_output(stdio): %s (conf=%.2f)",
                    gesture_event.class_name, gesture_event.conf)


def create_event_sender(config):
    """config의 event_output(mode·format)에 맞는 Sender를 만든다."""
    output_cfg = config["event_output"]
    mode = output_cfg["mode"]
    format_name = output_cfg.get("format", "text")   # 키 없으면 종전(GESTURE| 규격)
    if format_name not in PAYLOAD_BUILDERS:
        raise ValueError(f"지원하지 않는 event_output.format: {format_name}")
    if mode == "stdio":
        return StdioEventSender(PAYLOAD_BUILDERS[format_name])
    if mode == "console":
        return ConsoleEventSender()
    raise ValueError(f"지원하지 않는 event_output.mode: {mode}")
