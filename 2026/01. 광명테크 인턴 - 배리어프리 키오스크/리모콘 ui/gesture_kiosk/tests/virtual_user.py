"""가상 사용자 — 실제로 써 보는 시늉을 낸다 (2026-09-09 신설).

왜 만들었나
-----------
2026-09-09에 실기 보고가 두 번 들어왔다.

    "입 벌리면 커서 움직이고 아이콘에 커서 가져다 댈려고 하면 뭔가 흔들려.
     커서가 파란색이 아닌데 드래그가 되는 현상이 있네."
    "입 벌릴 때 클릭 노란색으로 안 뜨더라. 한 번 클릭 하려고 하는데
     바로 파란색으로 변하고 드래그 되더라."

두 번 다 코드를 읽어서 원인을 짐작하고 고쳤는데, 두 번째는 **첫 번째 수정이
만든 문제**였다. 짐작으로 고치는 것을 그만두고, 앞으로는 여기서 **먼저 써
본다.**

무엇을 흉내 내나
----------------
사람이 키오스크 앞에서 하는 일을 시간축 위에 늘어놓는다.

- 아이콘을 겨눈다 — 고개를 돌려 커서를 옮기고, 머문다(사람 손발은 완전히
  멈추지 않으므로 미세한 떨림이 남는다).
- 입을 벌렸다 다문다 — 클릭. 벌린 시간과 **다문 뒤 남는 턱 벌림**을 정할 수
  있다. 사람은 입을 완전히 원래대로 안 닫는 일이 흔해서, 이 값이 중요하다.
- 오래 벌리고 고개를 돌린다 — 드래그.

그 시간축을 **진짜 파이프라인**에 통과시킨다. 가상 카메라가 만든 얼굴
랜드마크 → HeadTracker → 커서 좌표·jaw_open/jaw_base → MouthGesture.
중간을 흉내 내지 않으므로, 여기서 나오는 결과는 실제로 그렇게 돈다는 뜻이다.

무엇을 못 보나 (정직하게)
-------------------------
- **입을 벌릴 때 얼굴 랜드마크가 밀리는 것**은 못 흉내 낸다. 가상 카메라에
  얼굴 변형 모형이 없다(`데이터 수치/00_README.md` "못 뽑은 것" 참고).
  그래서 "입 벌릴 때 커서가 흔들린다"의 **크기**는 여기서 안 나온다.
  대신 벌리는 동안 커서를 붙잡는지 아닌지는 확인할 수 있다.
- MediaPipe가 실제로 뱉는 jawOpen 값의 분포도 모른다. 여기서는 사람이
  입을 벌리고 다무는 모양을 시간 함수로 준다.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess.mouth_gesture import (           # noqa: E402
    CLICK, HOLD_END, HOLD_START, PRESS, MouthGesture)

FPS = 30.0


class Action:
    """가상 사용자가 하는 동작 하나."""

    def __init__(self, seconds, yaw_deg=None, pitch_deg=None,
                 jaw_open=None, label=""):
        self.seconds = seconds
        self.yaw_deg = yaw_deg
        self.pitch_deg = pitch_deg
        self.jaw_open = jaw_open
        self.label = label


def rest(seconds, yaw_deg=0.0, pitch_deg=0.0, jaw_open=0.05, label="가만히"):
    """가만히 있는다. jaw_open 기본값은 다물었을 때의 흔한 값."""
    return Action(seconds, yaw_deg, pitch_deg, jaw_open, label)


def aim(seconds, yaw_deg, pitch_deg=0.0, jaw_open=0.05, label="겨냥"):
    """고개를 돌려 겨눈다(끝 자세를 준다 — 사이는 부드럽게 이어진다)."""
    return Action(seconds, yaw_deg, pitch_deg, jaw_open, label)


def mouth_open(seconds, jaw_open=0.45, yaw_deg=None, pitch_deg=None,
               label="입 벌림"):
    return Action(seconds, yaw_deg, pitch_deg, jaw_open, label)


class VirtualUser:
    """동작 목록을 프레임 단위 (시각, 고개각, 턱벌림)으로 펼친다.

    자세와 턱 벌림은 동작 사이를 **부드럽게 잇는다**(코사인 보간). 사람의
    고개도 턱도 한 프레임에 순간이동하지 않고, 계단처럼 주면 판정이 실제보다
    쉬워진다 — 임계값을 단번에 넘으니까.

    tremor_deg
        겨누고 멈춰 있을 때 남는 미세한 떨림. 0으로 두면 사람이 아니다.
    """

    def __init__(self, actions, tremor_deg=0.12, seed=20260909):
        self.actions = actions
        self.tremor_deg = tremor_deg
        self._rng = np.random.default_rng(seed)

    def frames(self):
        """(시각, yaw, pitch, jaw_open, 지금 동작 이름)을 차례로 내놓는다."""
        yaw = self.actions[0].yaw_deg or 0.0
        pitch = self.actions[0].pitch_deg or 0.0
        jaw = self.actions[0].jaw_open if self.actions[0].jaw_open is not None else 0.05
        now = 0.0
        for action in self.actions:
            target_yaw = action.yaw_deg if action.yaw_deg is not None else yaw
            target_pitch = action.pitch_deg if action.pitch_deg is not None else pitch
            target_jaw = action.jaw_open if action.jaw_open is not None else jaw
            start_yaw, start_pitch, start_jaw = yaw, pitch, jaw
            steps = max(1, int(round(action.seconds * FPS)))
            for step in range(steps):
                # 코사인 보간 — 시작과 끝에서 속도가 0이라 사람 움직임에 가깝다
                ratio = 0.5 - 0.5 * math.cos(math.pi * (step + 1) / steps)
                yaw = start_yaw + (target_yaw - start_yaw) * ratio
                pitch = start_pitch + (target_pitch - start_pitch) * ratio
                jaw = start_jaw + (target_jaw - start_jaw) * ratio
                now += 1.0 / FPS
                shake = self._rng.normal(0.0, self.tremor_deg, 2)
                yield (now, yaw + shake[0], pitch + shake[1], jaw, action.label)


class Session:
    """가상 사용자가 한 번 쓰는 동안 무슨 일이 있었는지 남긴다.

    **사용자가 보는 것**(커서 색, 커서 위치)과 **운영체제가 받는 것**(버튼
    눌림, 드래그 거리)을 따로 남긴다. 2026-09-09의 버그가 정확히 이 둘이
    어긋나서 생겼기 때문이다.
    """

    def __init__(self):
        self.events = []            # (시각, 이름표)
        self.button_down_sec = None
        self.presses = []           # (누른 시각, 뗀 시각, 그 사이 커서 이동 px)
        self.saw_click_flash = False
        self.color_timeline = []    # (시각, 색 이름)

    def color_at(self, sec):
        seen = "초록"
        for when, color in self.color_timeline:
            if when <= sec:
                seen = color
        return seen

    @property
    def click_count(self):
        return sum(1 for _, name in self.events if name == CLICK)

    @property
    def drag_count(self):
        return sum(1 for _, name in self.events if name == HOLD_START)

    def drag_px_while_green(self):
        """초록색(=안 눌린 것처럼 보이는) 동안 커서가 끌려간 거리의 합."""
        total = 0.0
        for down_sec, up_sec, moved_px, green_px in self.presses:
            total += green_px
        return total
