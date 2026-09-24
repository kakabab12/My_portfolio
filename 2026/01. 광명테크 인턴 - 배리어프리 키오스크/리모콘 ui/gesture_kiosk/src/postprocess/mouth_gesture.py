"""입 벌리기 클릭·드래그 판정 (2026-09-09 신설).

왜 따로 꺼냈나
--------------
이 판정은 head.py · eyebrow.py · forehead.py 세 곳에 **똑같이 복사돼** 있었다.
세 파일 모두 main() 안쪽 클로저라 밖에서 부를 수가 없어서, 실제 얼굴 없이는
"짧게 벌리면 클릭, 길게 벌리면 드래그"가 정말 그렇게 갈리는지 확인할 방법이
없었다. 테스트에서 두 번 어긋난 뒤에야(2026-09-09) 그게 문제라는 걸 알았다.

여기로 옮겨 놓으면 가상 얼굴로 **직접 써 볼 수 있다.** 세 파일이 같은 코드를
쓰게 되는 것은 덤이다(tests/virtual_user.py 참고).

무엇을 하고 무엇을 안 하나
--------------------------
이 클래스는 **판정만** 한다. 마우스를 누르거나 색을 바꾸는 일은 안 하고,
무슨 일이 일어났는지를 이름표(PRESS/CLICK/HOLD_START/HOLD_END)로 돌려준다.
부르는 쪽이 그 이름표를 자기 방식대로 처리한다.

부작용을 섞지 않은 이유는 시험 때문만이 아니다. 부작용이 안에 있으면 "언제
누르고 언제 떼는가"를 읽으려고 마우스·커서·콘솔 코드까지 함께 봐야 한다.

판정 방식 — 왜 이렇게 갈리나
----------------------------
입을 벌리는 **순간** 버튼을 누른다(PRESS). 다물면 클릭(CLICK), 계속 벌리고
있으면 드래그(HOLD_START)다. 2026-08-31에 이렇게 바꿨다 — 그전에는 다물 때
클릭을 한 번에 보냈고, 벌리고-다무는 시간이 통째로 지연으로 느껴졌다.

여닫는 임계값이 다르다(히스테리시스). 벌림은 기준선+open_margin을 넘어야
하고, 다묾은 기준선+close_margin 아래로 내려와야 한다. 같은 값을 쓰면
턱이 임계값 근처에서 떨 때 열림/닫힘이 연달아 발화한다.

드래그 중에는 더 깊이·더 오래 다물어야 놓아준다(hold_release_*). 드래그
도중에 턱이 잠깐 올라갔다고 놓아 버리면 끌던 것이 중간에 떨어진다.
"""

# 무슨 일이 일어났는지 — update()가 돌려주는 이름표
PRESS = "press"            # 입이 벌어졌다. 버튼을 누르고 커서를 붙잡는다
CLICK = "click"            # 짧게 벌렸다 다물었다 = 한 번 클릭
HOLD_START = "hold_start"  # 계속 벌리고 있다 = 여기서부터 드래그
HOLD_END = "hold_end"      # 드래그 중에 다물었다 = 드래그 끝


class MouthGesture:
    """입 벌림 하나로 클릭과 드래그를 가른다.

    open_margin / close_margin
        기준선(다물었을 때의 값)에서 얼마나 벌어져야 벌림/다묾으로 보는지.
        **open_margin > close_margin 이어야 한다** — 그래야 임계값 근처에서
        떨어도 열림/닫힘이 연달아 발화하지 않는다.
    hold_sec
        이만큼 계속 벌리고 있으면 드래그로 넘어간다.
    close_confirm_sec
        이만큼 계속 다물고 있어야 진짜 다문 것으로 본다. 턱이 잠깐 내려간
        것을 다묾으로 세면 드래그가 매번 중간에 끊긴다.
    hold_release_margin / hold_release_confirm_sec
        드래그 중일 때 쓰는 더 빡빡한 다묾 조건.
    """

    def __init__(self, open_margin, close_margin, hold_sec,
                 close_confirm_sec, hold_release_margin,
                 hold_release_confirm_sec,
                 close_fraction=0.55, hold_release_fraction=0.75,
                 rest_window_sec=6.0, stuck_open_sec=8.0):
        if not open_margin > close_margin:
            # 이걸 어기면 열림/닫힘이 연달아 발화한다 — 조용히 이상하게
            # 도는 것보다 시작할 때 죽는 편이 낫다
            raise ValueError(
                "open_margin(%r)은 close_margin(%r)보다 커야 한다 — "
                "같거나 작으면 턱 떨림에 열림/닫힘이 연달아 발화한다"
                % (open_margin, close_margin))
        self._open_margin = open_margin
        self._close_margin = close_margin
        self._hold_sec = hold_sec
        self._close_confirm_sec = close_confirm_sec
        self._hold_release_margin = hold_release_margin
        self._hold_release_confirm_sec = hold_release_confirm_sec
        self._close_fraction = close_fraction
        self._hold_release_fraction = hold_release_fraction
        self._rest_window_sec = rest_window_sec
        self._stuck_open_sec = stuck_open_sec
        self.reset()

    def reset(self):
        """추적을 잃었을 때 등 — 처음 상태로 되돌린다."""
        self.is_open = False
        self.is_holding = False
        self.open_since_sec = None
        self._close_since_sec = None
        self._peak = None
        self._rest_samples = []   # 다물고 있을 때의 최근 값 — _rest_level 참고

    def _rest_level(self, jaw_base):
        """지금 이 사람이 **실제로 다물었을 때** 나오는 값.

        ★2026-09-09 — jaw_base는 캘리브레이션 때 한 번 정해지면 **영영
        안 바뀐다**(head_tracker._MedianCalibrator: "한 번 확정되면 이후
        표본은 무시한다"). 그런데 사람의 다문 턱 위치는 그 뒤에도 바뀐다.
        의자에 기대거나, 말을 하거나, 캘리브레이션을 살짝 벌린 채 했거나.

        그 값이 jaw_base + open_margin을 넘어서 버리면 **입을 다물고 있는데도
        계속 벌린 것으로 판정**된다. 그러면 눌림이 풀리지 않아 마우스 버튼이
        눌린 채로 남는다. 가상 사용자로 재현했다(잔여 0.20).

        그래서 다물고 있는 동안의 최근 값 중 **가장 낮은 것**을 함께 본다.
        정의상 그게 "이 사람이 지금 가장 다문 상태"다. 캘리브레이션 값보다
        낮게는 안 내린다 — 잡음으로 문턱이 낮아져 헛클릭이 나면 더 나쁘다.
        """
        if not self._rest_samples:
            return jaw_base
        return max(jaw_base, min(value for _, value in self._rest_samples))

    def _close_level(self, jaw_base):
        """이 값 이하로 내려오면 다문 것으로 본다.

        ★2026-09-09 — 기준선에서 측정한 절대 임계값만 쓰다가 **버튼이 눌린 채
        영영 안 풀리는** 버그가 났다(가상 사용자로 재현, tests/test_mouth_gesture.py).

        기준선 0.05, 벌림 임계 +0.12, 다묾 임계 +0.05일 때 다묾 선은 0.10이다.
        그런데 사람은 클릭한 뒤 입을 **완전히 원래대로 안 다문다.** 턱이
        0.11에 머무르면 0.10 아래로 안 내려오므로
          · 다묾이 확정되지 않고 → 클릭이 안 되고(노란 깜빡임도 없고)
          · 0.7초 뒤 드래그로 넘어가고 → 파란색이 되고
          · 드래그 해제 조건은 더 빡빡해서(+0.03) 역시 안 걸린다.
        결국 **마우스 버튼이 눌린 채로 남는다.** 사용자 보고 그대로였다.
        0.10과 0.11 사이에 절벽이 있었던 것이다.

        그래서 **얼마나 벌렸었는지에 견주어** 판단한다. 크게 벌렸다가 절반
        넘게 내려왔으면, 절대값이 기준선보다 조금 높아도 다문 것이다.

            다묾 선 = max(기준선 + 절대 여유, 최대치 - 비율 x (최대치 - 기준선))

        max를 쓰는 이유 — 살짝만 벌린 경우(최대치가 임계값 언저리)에는 상대
        조건이 절대 조건보다 느슨해져서 열자마자 닫히기 때문이다. 둘 중
        **덜 관대한** 쪽을 쓴다.
        """
        margin = (self._hold_release_margin if self.is_holding
                  else self._close_margin)
        level = jaw_base + margin
        if self._peak is None:
            return level
        fraction = (self._hold_release_fraction if self.is_holding
                    else self._close_fraction)
        rise = self._peak - jaw_base
        if rise <= 0.0:
            return level
        return max(level, self._peak - fraction * rise)

    @property
    def held_sec(self):
        """지금 벌리고 있는 시간. 안 벌리고 있으면 None."""
        return None if self.open_since_sec is None else self._held_sec

    def update(self, jaw_open, jaw_base, now_sec):
        """한 프레임. 일어난 일들을 순서대로 돌려준다(없으면 빈 튜플).

        jaw_base가 None이면(아직 기준선을 못 잡음) 아무 판정도 하지 않는다.
        """
        self._held_sec = (0.0 if self.open_since_sec is None
                          else now_sec - self.open_since_sec)
        if jaw_base is None or jaw_open is None:
            return ()

        if not self.is_open:
            # 다물고 있는 동안만 모은다 — 벌린 값이 섞이면 "다문 값"이 아니다
            self._rest_samples.append((now_sec, jaw_open))
            while (self._rest_samples
                   and now_sec - self._rest_samples[0][0] > self._rest_window_sec):
                self._rest_samples.pop(0)
        jaw_base = self._rest_level(jaw_base)

        if not self.is_open:
            if jaw_open >= jaw_base + self._open_margin:
                self.is_open = True
                self.open_since_sec = now_sec
                self._close_since_sec = None
                self._peak = jaw_open
                self._held_sec = 0.0
                return (PRESS,)
            return ()

        # 얼마나 벌렸었는지 — 다묾 판단에 쓴다(_close_level 참고)
        self._peak = jaw_open if self._peak is None else max(self._peak, jaw_open)
        close_confirm = (self._hold_release_confirm_sec if self.is_holding
                         else self._close_confirm_sec)

        if jaw_open <= self._close_level(jaw_base):
            if self._close_since_sec is None:
                self._close_since_sec = now_sec
                return ()
            if now_sec - self._close_since_sec < close_confirm:
                return ()
            was_holding = self.is_holding
            self.is_open = False
            self.is_holding = False
            self.open_since_sec = None
            self._close_since_sec = None
            self._peak = None
            self._held_sec = 0.0
            return (HOLD_END,) if was_holding else (CLICK,)

        # 아직 벌리고 있다 — 다묾 후보는 취소하고 벌린 시간은 계속 쌓는다
        self._close_since_sec = None

        # ★안전장치 — 이렇게 오래 벌린 채로 있을 리가 없다. 여기까지 왔다면
        # 판정이 갇힌 것이다(다문 값이 통째로 올라갔는데 _rest_samples가 아직
        # 못 따라잡은 경우 등). 버튼을 눌린 채 두는 것보다 놓는 편이 낫다 —
        # 눌린 채로 두면 사용자가 할 수 있는 일이 아무것도 없다.
        # 놓으면서 지금 값을 "다문 값"으로 넣어 다시 갇히지 않게 한다.
        if self._held_sec >= self._stuck_open_sec:
            was_holding = self.is_holding
            self.is_open = False
            self.is_holding = False
            self.open_since_sec = None
            self._close_since_sec = None
            self._peak = None
            self._held_sec = 0.0
            self._rest_samples = [(now_sec, jaw_open)]
            return (HOLD_END,) if was_holding else (CLICK,)

        if not self.is_holding and self._held_sec >= self._hold_sec:
            self.is_holding = True
            return (HOLD_START,)
        return ()


class ClickFreeze:
    """클릭하는 동안 커서를 붙잡아 둔다 (2026-09-09 신설).

    왜 필요한가
    -----------
    입을 벌리면 턱이 내려가면서 얼굴 랜드마크가 밀린다. 그 밀림이 그대로
    커서로 들어가면 **누르는 순간 커서가 흘러** 겨눴던 곳이 아닌 데를 누른다.
    사용자 보고 그대로다("입 벌리면 커서 움직이고").

    그리고 벌리는 순간 버튼이 눌리므로(MouthGesture 독스트링 참고), 그 흘러간
    거리는 그냥 어긋남이 아니라 **의도하지 않은 드래그**다.

    어떻게
    ------
    벌어졌다고 판정된 순간부터 드래그로 넘어갈 때까지 커서를 한 점에 붙잡는다.
    붙잡는 지점은 **지금 위치가 아니라 lookback_sec 이전 위치**다 — 벌어졌다고
    판정될 무렵엔 이미 턱이 내려가 랜드마크가 밀린 뒤라, 지금 위치를 붙잡으면
    밀려난 자리를 붙잡게 된다.

    드래그로 넘어갈 때 그냥 놓으면 커서가 튄다(붙잡고 있는 동안 고개가
    움직였을 수 있다). unfreeze_sec에 걸쳐 현재 위치로 이어 준다.
    """

    def __init__(self, lookback_sec=0.15, unfreeze_sec=0.25,
                 history_sec=0.6, enabled=True):
        self._lookback_sec = lookback_sec
        self._unfreeze_sec = unfreeze_sec
        self._history_sec = history_sec
        self._enabled = enabled
        self._history = []          # [(시각, x, y)]
        self.frozen_xy = None
        self._unfreeze_since_sec = None
        self._unfreeze_from_xy = None

    @property
    def is_frozen(self):
        return self.frozen_xy is not None

    def _lookback_xy(self, now_sec):
        target_sec = now_sec - self._lookback_sec
        chosen = None
        for sample in self._history:
            if sample[0] <= target_sec:
                chosen = sample
            else:
                break
        if chosen is None:
            chosen = self._history[0] if self._history else None
        return None if chosen is None else (chosen[1], chosen[2])

    def begin(self, now_sec):
        """입이 벌어졌다 — 겨눴던 지점에 붙잡는다."""
        self.frozen_xy = self._lookback_xy(now_sec)
        self._unfreeze_since_sec = None
        self._unfreeze_from_xy = None

    def end(self, now_sec, blend):
        """붙잡기 해제. blend=True(드래그 시작)면 현재 위치로 이어 준다."""
        frozen = self.frozen_xy
        self.frozen_xy = None
        if blend and frozen is not None:
            self._unfreeze_from_xy = frozen
            self._unfreeze_since_sec = now_sec
        else:
            self._unfreeze_from_xy = None
            self._unfreeze_since_sec = None

    def apply(self, now_sec, x_ratio, y_ratio):
        """커서 좌표를 내보내기 직전에 통과시킨다 — 기록도 여기서 쌓는다."""
        self._history.append((now_sec, x_ratio, y_ratio))
        while (self._history
               and now_sec - self._history[0][0] > self._history_sec):
            self._history.pop(0)
        if not self._enabled:
            return x_ratio, y_ratio
        if self.frozen_xy is not None:
            return self.frozen_xy
        since = self._unfreeze_since_sec
        origin = self._unfreeze_from_xy
        if since is not None and origin is not None:
            ratio = (now_sec - since) / self._unfreeze_sec
            if ratio < 1.0:
                return (origin[0] + (x_ratio - origin[0]) * ratio,
                        origin[1] + (y_ratio - origin[1]) * ratio)
            self._unfreeze_since_sec = None
            self._unfreeze_from_xy = None
        return x_ratio, y_ratio

    def reset(self):
        self._history.clear()
        self.frozen_xy = None
        self._unfreeze_since_sec = None
        self._unfreeze_from_xy = None
