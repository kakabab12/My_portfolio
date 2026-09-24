# -*- coding: utf-8 -*-
"""그림 3의 원자료 — 클릭 뒤 입에 남은 벌림에 따른 클릭 성공 여부 (2026-09-11).

예전 그림 3은 판정 규칙을 손으로 옮긴 도식이었다. x값(0.05·0.07·0.09…)을 직접
적었고, 테스트가 확인하지 않은 0.20까지 "고친 뒤 성공"으로 그렸다. 이 스크립트는
**실제 판정 코드**를 그대로 돌려 값을 낸다.

두 판정
  · 고친 뒤  — 지금의 MouthGesture (얼마나 벌렸었는지와 견준다)
  · 고치기 전 — 다묾 선을 "기준선 + 여유" 하나로만 보던 판정. MouthGesture를
    상속해 _close_level·_rest_level만 예전 식으로 되돌리고, 예전에 없던 갇힘
    안전장치(stuck_open_sec)는 끈다.

입력은 tests/test_mouth_gesture.py와 같다(값·헬퍼를 그대로 가져온다) —
0.5초 다묾 → 0.3초 크게 벌림(0.45) → 2초 동안 잔여 벌림 유지.
성공 = 이름표가 정확히 [PRESS, CLICK].

범위는 벌림 임계값(기준선 0.05 + 0.12 = 0.17) **아래까지만** 측정한다. 그 위는
다문 것인지 벌린 것인지 판단할 근거가 없는 다른 영역이라(같은 테스트 파일의
test_resting_above_the_open_threshold_settles_down 참고) 이 그림의 질문이 아니다.

    py -3 "데이터 수치/mouth_cliff.py"
"""
import csv
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.postprocess.mouth_gesture import CLICK, PRESS, MouthGesture  # noqa: E402
from tests.test_mouth_gesture import (  # noqa: E402
    BASE, CLOSE, CONFIRM, HOLD, OPEN, REL_CONFIRM, REL_MARGIN, _gesture, _play)

OPEN_LINE = round(BASE + OPEN, 3)          # 0.17 — 이 아래까지만 측정한다
# 테스트가 "고친 뒤 깨끗한 클릭"을 확인하는 값 — 스윕이 이것과 어긋나면 멈춘다
TESTED_AFTER = (0.06, 0.08, 0.10, 0.11, 0.12, 0.15, 0.16)


class BeforeFix(MouthGesture):
    """2026-09-09 수정 전의 다묾 판정."""

    def _rest_level(self, jaw_base):
        return jaw_base                       # 다문 값을 따라가지 않았다

    def _close_level(self, jaw_base):
        margin = (self._hold_release_margin if self.is_holding
                  else self._close_margin)
        return jaw_base + margin              # 얼마나 벌렸었는지를 안 봤다


def _before():
    return BeforeFix(OPEN, CLOSE, HOLD, CONFIRM, REL_MARGIN, REL_CONFIRM,
                     close_fraction=0.55, hold_release_fraction=0.75,
                     rest_window_sec=6.0, stuck_open_sec=1e9)


def _outcome(gesture, residual):
    seen = _play(gesture, [(0.5, BASE), (0.3, 0.45), (2.0, residual)])
    names = [name for _, name in seen]
    return names == [PRESS, CLICK], "+".join(names) or "(없음)"


def main():
    residuals = [round(0.050 + 0.005 * i, 3) for i in range(24)]
    residuals = [r for r in residuals if r < OPEN_LINE]
    rows = []
    for r in residuals:
        ok_before, names_before = _outcome(_before(), r)
        ok_after, names_after = _outcome(_gesture(), r)
        rows.append([r, int(ok_before), names_before, int(ok_after), names_after])

    by_residual = {row[0]: row for row in rows}
    for r in TESTED_AFTER:
        if r in by_residual and not by_residual[r][3]:
            raise SystemExit("고친 뒤 판정이 테스트(%.2f)와 어긋난다 — 멈춤" % r)

    path = os.path.join(HERE, "입다묾_절벽.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["잔여 벌림", "고치기 전 성공", "고치기 전 이름표",
                         "고친 뒤 성공", "고친 뒤 이름표"])
        writer.writerows(rows)

    print("잔여 벌림에 따른 클릭 (기준선 %.2f, 벌림 임계 %.2f 아래까지)"
          % (BASE, OPEN_LINE))
    for r, ok_b, names_b, ok_a, names_a in rows:
        print("  %.3f   고치기 전 %s %-20s  고친 뒤 %s %s" % (
            r, "성공" if ok_b else "실패", names_b,
            "성공" if ok_a else "실패", names_a))
    last_ok = max((row[0] for row in rows if row[1]), default=None)
    first_bad = min((row[0] for row in rows if not row[1]), default=None)
    print()
    print("고치기 전: 마지막 성공 %s, 첫 실패 %s" % (last_ok, first_bad))
    print("고친 뒤: %d/%d 성공" % (sum(row[3] for row in rows), len(rows)))
    print("저장:", path)


if __name__ == "__main__":
    main()
