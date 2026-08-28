"""이벤트 전송 규격 테스트 — 델파이7 텍스트 규격 + 파이프(stdio) 출력 검증.

2026-07-23: UDP·웹소켓 제거(회사 결정 — 네트워크 철회), stdio 전환.
stdout은 이벤트 전용 채널이므로 바이트 단위까지 규격을 고정한다.
"""
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.event_sender import (
    ConsoleEventSender, StdioEventSender, build_name_payload, build_text_payload,
    create_event_sender,
)
from src.postprocess.gesture_filter import GestureEvent


def _event(class_name="confirm", hand_side="left"):
    return GestureEvent(class_name=class_name, conf=1.0, ts_sec=12345.6789, hand_side=hand_side)


class PayloadFormatTest(unittest.TestCase):
    def test_text_payload_is_delphi_line(self):
        # GESTURE|이벤트|손|신뢰도|시각 + CRLF — 델파이7 Pos/Copy 파싱 규격
        self.assertEqual(
            build_text_payload(_event()), b"GESTURE|confirm|left|1.00|12345.679\r\n"
        )

    def test_text_payload_without_hand_side(self):
        # 손 미상 — 3번째 필드는 빈 문자열 (구분자 수는 항상 4개로 고정)
        line = build_text_payload(_event(hand_side=None)).decode("ascii")
        self.assertEqual(line.count("|"), 4)
        self.assertEqual(line.split("|")[2], "")

    def test_text_payload_is_ascii_single_line(self):
        line = build_text_payload(_event("home", "right"))
        self.assertTrue(line.endswith(b"\r\n"))
        self.assertEqual(line.count(b"\n"), 1)      # 이벤트 1건 = 한 줄

    def test_name_payload_is_event_name_only(self):
        # 간소 규격(2026-07-31 사용자 요청) — cmd 창에 이벤트명 한 줄만
        self.assertEqual(build_name_payload(_event("left")), b"left\r\n")


class _FakeStdout:
    """sys.stdout 대역 — buffer에 쓰인 바이트를 그대로 보관한다."""

    def __init__(self):
        self.buffer = io.BytesIO()


class StdioSenderTest(unittest.TestCase):
    def setUp(self):
        self._real_stdout = sys.stdout
        sys.stdout = _FakeStdout()

    def tearDown(self):
        sys.stdout = self._real_stdout

    def test_stdio_writes_exact_line_to_stdout(self):
        StdioEventSender().send(_event("left", "right"))
        self.assertEqual(sys.stdout.buffer.getvalue(),
                         b"GESTURE|left|right|1.00|12345.679\r\n")

    def test_consecutive_events_are_separate_lines(self):
        sender = StdioEventSender()
        sender.send(_event("right", "right"))
        sender.send(_event("confirm", "left"))
        lines = sys.stdout.buffer.getvalue().split(b"\r\n")
        self.assertEqual(lines[0], b"GESTURE|right|right|1.00|12345.679")
        self.assertEqual(lines[1], b"GESTURE|confirm|left|1.00|12345.679")

    def test_name_format_sender_writes_event_name_only(self):
        # config format: name — stdout에 이벤트명 한 줄만 (현행 배포 설정)
        sender = create_event_sender({"event_output": {"mode": "stdio", "format": "name"}})
        sender.send(_event("confirm", "left"))
        self.assertEqual(sys.stdout.buffer.getvalue(), b"confirm\r\n")


class BrokenPipeTest(unittest.TestCase):
    """수신부가 파이프를 닫았을 때 — 조용히 죽지 않는다 (2026-08-04 실기 사고).

    델파이(또는 명령줄 파이프)가 stdout을 닫으면 write가 BrokenPipeError를 던진다.
    종전엔 그대로 위로 튀어 **추론 스레드가 죽고** 창만 멈춘 좀비가 됐다 —
    이벤트도 로그도 안 나와 원인 파악이 어려웠다. 이제 "보낼 곳 없음"을
    돌려주고, 호출부(realtime_loop)가 엔진을 접는다.
    """

    def setUp(self):
        self._real_stdout = sys.stdout
        sys.stdout = _FakeStdout()

    def tearDown(self):
        sys.stdout = self._real_stdout

    def _break_pipe(self, error):
        def raise_error(_payload):
            raise error
        sys.stdout.buffer.write = raise_error

    def test_broken_pipe_reports_closed(self):
        sender = StdioEventSender()
        self._break_pipe(BrokenPipeError())
        self.assertTrue(sender.send(_event("left", "right")))

    def test_closed_buffer_reports_closed(self):
        # 닫힌 버퍼에 쓰면 ValueError — 파이프 소멸과 같은 상황으로 다룬다
        sender = StdioEventSender()
        self._break_pipe(ValueError("I/O operation on closed file"))
        self.assertTrue(sender.send(_event("left", "right")))

    def test_normal_send_reports_open(self):
        self.assertFalse(StdioEventSender().send(_event("left", "right")))

    def test_does_not_retry_after_close(self):
        # 한 번 끊기면 다시 쓰지 않는다 — 매 이벤트마다 예외를 내지 않게
        sender = StdioEventSender()
        self._break_pipe(BrokenPipeError())
        sender.send(_event("left", "right"))
        sys.stdout.buffer.write = lambda _payload: self.fail("닫힌 뒤에 또 썼다")
        self.assertTrue(sender.send(_event("right", "right")))

    def test_console_sender_reports_open(self):
        # 계약 통일 — 콘솔 sender도 같은 형식으로 답한다
        self.assertFalse(ConsoleEventSender().send(_event("left", "right")))


class CreateSenderTest(unittest.TestCase):
    def test_stdio_mode(self):
        sender = create_event_sender({"event_output": {"mode": "stdio"}})
        self.assertIsInstance(sender, StdioEventSender)

    def test_console_mode(self):
        sender = create_event_sender({"event_output": {"mode": "console"}})
        self.assertIsInstance(sender, ConsoleEventSender)

    def test_unknown_mode_rejected_at_startup(self):
        # 오타는 이벤트 발생 시점이 아니라 시작 시점에 죽어야 한다
        with self.assertRaises(ValueError):
            create_event_sender({"event_output": {"mode": "udp"}})


if __name__ == "__main__":
    unittest.main()
