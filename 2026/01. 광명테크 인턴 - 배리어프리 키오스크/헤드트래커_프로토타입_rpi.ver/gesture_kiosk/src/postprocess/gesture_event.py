"""postprocess 모듈 — 확정된 동작 이벤트의 공통 데이터 구조 (기획서 4.6).

여러 판정 모듈(head_tracker 등)과 전달 모듈(event_sender·demo_server)이 공유하는
단일 정의 — "한 개념 한 이름" 원칙에 따라 이 파일 하나에만 둔다.
"""
from dataclasses import dataclass


@dataclass
class GestureEvent:
    """확정된 동작 이벤트 1건 — 회사 프로그램(키오스크 UI)으로 전달되는 단위."""

    class_name: str
    conf: float
    ts_sec: float
    data: dict = None       # 부가 정보 확장용 (예: head_tracker의 trigger 종류)
