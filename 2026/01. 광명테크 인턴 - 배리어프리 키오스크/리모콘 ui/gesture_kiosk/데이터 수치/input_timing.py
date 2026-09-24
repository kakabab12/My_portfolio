# -*- coding: utf-8 -*-
"""가상 사용자의 입 동작 속도가 판정 결론을 바꾸나 (2026-09-11).

왜 측정하나
-----------
논문 Ⅵ-1은 "만들어 넣은 입력의 시간 특성이 비현실적이면 시험 결과가 판정 코드가 아니라
입력 설정에서 나온다"고 주장한다. 그런데 이를 받치는 숫자가 저장소에 없다
(tests/virtual_user.py에 "계단처럼 주면 판정이 실제보다 쉬워진다"는 서술만 있다).
그래서 판정 코드는 그대로 두고 **가상 사용자의 입 동작 시간만 바꿔** 클릭 판정이 어떻게
달라지는지 측정한다.

무엇을 측정하나
---------------
전체 파이프라인(가상 카메라 → HeadTracker → 입 판정)을 tests/test_virtual_user_click.py의
실행 도구 그대로 쓴다. 고개 8도로 겨눈 채 입을 크게 벌렸다(0.45) 0.3초 머물고 다문다(0.05).

  A. 다무는 데 걸리는 시간 — 1프레임(계단), 0.15초(시험 기본값), 0.3, 0.5, 0.7, 1.0초.
     벌리는 시간은 0.12초로 둔다.
  B. 벌리는 데 걸리는 시간 — 1프레임(계단), 0.12초(시험 기본값), 0.3, 0.5초.
     다무는 시간은 0.15초로 둔다.

판정 결과 이름표가 [PRESS, CLICK]이면 클릭, HOLD_START가 끼면 드래그로 본다.
시험 기본값(0.12/0.15초)은 시험 파일이 정한 값이지 사람을 측정한 값이 아니다 —
사람의 실제 입 동작 시간은 이 저장소에 측정 자료가 없다.

    py -3 "데이터 수치/input_timing.py"
"""
import csv
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from src.postprocess.mouth_gesture import CLICK, HOLD_START, PRESS  # noqa: E402
from tests.test_virtual_user_click import JAW_SHUT, JAW_WIDE, WARMUP, _use  # noqa: E402
from tests.virtual_user import FPS, aim, mouth_open, rest  # noqa: E402

YAW = 8.0
PLATEAU = 0.3
ONE_FRAME = 1.0 / FPS
CLOSE_TIMES = (ONE_FRAME, 0.15, 0.3, 0.5, 0.7, 1.0)
OPEN_TIMES = (ONE_FRAME, 0.12, 0.3, 0.5)


def run(open_sec, close_sec):
    actions = (WARMUP + [aim(1.0, yaw_deg=YAW)]
               + [mouth_open(open_sec, jaw_open=JAW_WIDE, yaw_deg=YAW, label="벌림"),
                  mouth_open(PLATEAU, jaw_open=JAW_WIDE, yaw_deg=YAW, label="벌린 채"),
                  mouth_open(close_sec, jaw_open=JAW_SHUT, yaw_deg=YAW, label="다묾")]
               + [rest(1.5, yaw_deg=YAW, label="대기")])
    events = _use(actions)["events"]
    if events == [PRESS, CLICK]:
        verdict = "클릭"
    elif HOLD_START in events:
        verdict = "드래그"
    else:
        verdict = "기타"
    return verdict, "+".join(events) or "(없음)"


def label_of(sec):
    return "1프레임(계단)" if abs(sec - ONE_FRAME) < 1e-9 else "%.2f초" % sec


def main():
    rows = []
    for close_sec in CLOSE_TIMES:
        verdict, names = run(0.12, close_sec)
        rows.append(["A 다무는 시간", label_of(close_sec), round(close_sec, 4), 0.12,
                     round(close_sec, 4), verdict, names])
    for open_sec in OPEN_TIMES:
        verdict, names = run(open_sec, 0.15)
        rows.append(["B 벌리는 시간", label_of(open_sec), round(open_sec, 4),
                     round(open_sec, 4), 0.15, verdict, names])

    path = os.path.join(HERE, "입력속도_클릭판정.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["바꾼 것", "값", "바꾼 값(초)", "벌리는 시간(초)", "다무는 시간(초)",
                         "판정", "이름표"])
        writer.writerows(rows)

    print("입 동작 시간만 바꿨을 때의 판정 (벌린 채 %.1f초, 판정 코드는 그대로)" % PLATEAU)
    for row in rows:
        print("  %-10s %-14s → %-4s  %s" % (row[0], row[1], row[5], row[6]))
    print("저장:", path)


if __name__ == "__main__":
    main()
