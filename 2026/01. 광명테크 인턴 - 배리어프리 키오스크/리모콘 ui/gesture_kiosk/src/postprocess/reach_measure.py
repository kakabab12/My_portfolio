"""그 사람이 실제로 고개를 얼마나 돌리는지 잰다 (2026-09-05 신설).

무엇이 문제인가
---------------
회전 매핑은 "화면 절반 폭에 닿는 각도"를 정해 두고 커서를 만든다. 목을 그만큼
못 돌리는 사람은 화면 가장자리에 아예 못 닿는다:

    좌우로 돌릴 수 있는 최대   화면 폭에서 닿는 범위
          5.0도                     32.7%
          7.0도                     45.8%
         10.0도                     65.8%
         15.0도                    100.0%

마비가 있는 사용자가 정확히 이 경우다. 가장자리 버튼을 못 누르므로 정확도가
아니라 "쓸 수 있냐 없냐"의 문제다. 고치려면 도달 배율(orientation_reach_gain)을
올리면 되는데, **그 값을 얼마로 할지 아무도 모른다.**

왜 자동으로 안 하나 — 재 보고 접었다
------------------------------------
처음에는 관측만으로 자동 판정하려 했다. "물리적 한계에 부딪히는 사람은 크게
돌릴 때마다 거의 같은 값에서 멈추고, 그냥 가운데만 쓴 사람은 최대치가
흩어진다"는 발상이었다. 왕복 최대치의 상위 75%/95% 비로 갈라 봤다.

    7도에서 막히는 사람 (편차 0.6도)      0.93
    4~11도만 쓴 사람 (더 돌릴 수 있음)     0.87
    5~10도만 쓴 사람                      0.90

**겹친다.** 문턱을 어디에 둬도 한쪽을 틀린다. 틀리는 방향이 나쁘다 — 목이
멀쩡한 사람의 감도를 올리면 아무 이득 없이 떨림만 배율만큼 커진다(커서
위치가 회전탄젠트/tan(반폭)이라 배율이 잡음에 그대로 곱해진다).

그래서 **자동 판정은 넣지 않는다.** 대신 이 파일은 재기만 하고, 값은 사람이
보고 정한다(scripts/measure_reach.py). 이 프로젝트가 계속 지켜 온 방식이다 —
못 가리는 것을 가리는 척하지 않고, 재는 도구를 준다.

  · **Rousseeuw, P. J. (1984).** "Least Median of Squares Regression."
    *JASA* 79(388), 871-880 — 최대치를 평균이나 최댓값으로 보면 한 번의 큰
    움직임이나 검출 오류에 통째로 끌려간다. 분위수로 본다.

떨림 예산 — 권장값의 상한
-------------------------
도달 배율을 g로 올리면 정지 시 커서 떨림도 정확히 g배가 된다. 그래서 지금
이 사람의 떨림을 함께 재서, 올린 뒤의 떨림이 예산을 넘지 않는 선까지만
권한다. 어두운 방이나 먼 거리라 이미 떨림이 큰 사람에게는 덜 권하게 된다 —
닿는 범위를 얻자고 커서를 못 쓸 만큼 떨게 만들지 않는다.
"""
import math

# --- 왕복(excursion) 판정 ---------------------------------------------------
# 반폭 대비 이만큼 넘어가면 "크게 움직였다"로 보고 최대치를 재기 시작한다.
# 너무 낮으면 잡음이 왕복으로 세어지고, 너무 높으면 좁은 사람의 움직임이
# 통째로 안 세어진다 — 7도 사용자도 반폭 15도의 0.25배(3.75도)는 넘는다
ENTER_RATIO = 0.25
# 이만큼 아래로 돌아오면 한 번의 왕복이 끝난 것으로 본다 (되돌림 히스테리시스)
EXIT_RATIO = 0.10

MIN_PEAKS = 12          # 이만큼은 왕복해야 분위수를 말할 수 있다
MAX_PEAKS = 200         # 최근 것만 본다

# --- 떨림 재기 --------------------------------------------------------------
STILL_WINDOW = 15            # 이 개수가
STILL_SPAN_RATIO = 0.06      # 반폭 대비 이 폭 안에 있으면 "가만히 있다"
MIN_STILL_SAMPLES = 8

# 권장 배율을 올린 뒤의 정지 시 커서 떨림 상한 (화면 비율, 1 sigma).
# 0.006 = 화면 폭의 0.6%. 1920px에서 약 11px
JITTER_BUDGET = 0.006
MAX_GAIN = 3.0               # 설정에서 받아 주는 상한과 같다


def _quantile(sorted_vals, q):
    """정렬된 목록의 분위수."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


class _Axis:
    """한 축(가로 또는 세로)의 가동범위와 떨림을 모은다."""

    def __init__(self, tan_span):
        self.tan_span = max(1e-6, float(tan_span))
        self.peaks = []
        self.still_devs = []
        self._in_excursion = False
        self._peak = 0.0
        self._recent = []

    def add(self, tan_value):
        if not math.isfinite(tan_value):
            return
        self._feed_peaks(abs(tan_value))
        self._feed_still(tan_value)

    def _feed_peaks(self, mag):
        enter = self.tan_span * ENTER_RATIO
        exit_at = self.tan_span * EXIT_RATIO
        if self._in_excursion:
            if mag > self._peak:
                self._peak = mag
            if mag < exit_at:
                self.peaks.append(self._peak)
                if len(self.peaks) > MAX_PEAKS:
                    del self.peaks[0]
                self._in_excursion = False
                self._peak = 0.0
        elif mag > enter:
            self._in_excursion = True
            self._peak = mag

    def _feed_still(self, value):
        """가만히 있는 구간에서만 떨림을 잰다 — 움직임을 떨림으로 세지 않게."""
        self._recent.append(value)
        if len(self._recent) > STILL_WINDOW:
            del self._recent[0]
        if len(self._recent) < STILL_WINDOW:
            return
        if max(self._recent) - min(self._recent) > self.tan_span * STILL_SPAN_RATIO:
            return                      # 움직이는 중이다
        mean = sum(self._recent) / len(self._recent)
        var = sum((v - mean) ** 2 for v in self._recent) / (len(self._recent) - 1)
        self.still_devs.append(math.sqrt(var))
        if len(self.still_devs) > MAX_PEAKS:
            del self.still_devs[0]

    # -- 보고 -----------------------------------------------------------
    def reach_deg(self):
        """이 사람이 실제로 돌린 각도 (상위 95%). 표본이 모자라면 None."""
        if len(self.peaks) < MIN_PEAKS:
            return None
        hi = _quantile(sorted(self.peaks), 0.95)
        return math.degrees(math.atan(hi)) if hi > 0 else None

    def span_deg(self):
        return math.degrees(math.atan(self.tan_span))

    def reach_ratio(self):
        """화면 폭(또는 높이)의 몇 %까지 닿는가. 못 재면 None."""
        deg = self.reach_deg()
        if deg is None:
            return None
        return min(1.0, math.tan(math.radians(deg)) / self.tan_span)

    def jitter_ratio(self):
        """지금 정지 시 커서 떨림 (화면 비율, 1 sigma). 못 재면 None."""
        if len(self.still_devs) < MIN_STILL_SAMPLES:
            return None
        dev = _quantile(sorted(self.still_devs), 0.5)     # 중앙값 (Rousseeuw)
        return dev / self.tan_span * 0.5

    def ceiling_tightness(self):
        """최대치가 한 점에 몰려 있는 정도 (상위 75% / 상위 95%).

        1에 가까울수록 "늘 같은 데서 멈춘다"는 뜻이지만, **이것만으로는
        못 가른다** — 위 독스트링의 측정 참고. 참고용으로만 보고한다.
        """
        if len(self.peaks) < MIN_PEAKS:
            return None
        peaks = sorted(self.peaks)
        hi = _quantile(peaks, 0.95)
        if hi <= 0.0:
            return None
        return _quantile(peaks, 0.75) / hi

    def recommended_gain(self):
        """화면 끝에 닿게 하려면 배율이 얼마여야 하나 (떨림 예산으로 제한).

        (권장값, 왜 그 값인지) 를 돌려준다. 못 정하면 (None, 이유).
        """
        ratio = self.reach_ratio()
        if ratio is None:
            return None, "왕복이 %d번뿐이라 못 정한다 (최소 %d번)" % (
                len(self.peaks), MIN_PEAKS)
        if ratio >= 0.95:
            return 1.0, "이미 화면 끝까지 닿는다 — 손대지 않는다"
        want = 1.0 / ratio
        jitter = self.jitter_ratio()
        if jitter is None:
            return None, "떨림을 못 쟀다 — 가만히 있는 구간이 더 필요하다"
        if jitter <= 0.0:
            budget = MAX_GAIN
        else:
            budget = JITTER_BUDGET / jitter
        if budget < want:
            gain = max(1.0, min(MAX_GAIN, budget))
            return gain, ("떨림 예산에 걸렸다 — 끝까지 닿으려면 %.2f가 필요한데 "
                          "지금 떨림(%.2f%%)에서는 %.2f가 한계다"
                          % (want, jitter * 100, gain))
        gain = max(1.0, min(MAX_GAIN, want))
        return gain, "화면 %.0f%%까지 닿는 사람 -> 끝까지 닿게 한다" % (ratio * 100)


class ReachMeasure:
    """가로·세로 각각의 가동범위와 떨림을 모은다.

        m = ReachMeasure(tan_span_x, tan_span_y)
        m.add(raw_tan_x, raw_tan_y)     # 매 프레임
        print(m.report())
    """

    def __init__(self, tan_span_x, tan_span_y):
        self.x = _Axis(tan_span_x)
        self.y = _Axis(tan_span_y)

    def add(self, tan_x, tan_y):
        self.x.add(tan_x)
        self.y.add(tan_y)

    def report(self):
        out = {}
        for name, axis in (("x", self.x), ("y", self.y)):
            gain, why = axis.recommended_gain()
            out[name] = {
                "peaks": len(axis.peaks),
                "span_deg": axis.span_deg(),
                "reach_deg": axis.reach_deg(),
                "reach_ratio": axis.reach_ratio(),
                "jitter_ratio": axis.jitter_ratio(),
                "ceiling_tightness": axis.ceiling_tightness(),
                "recommended_gain": gain,
                "reason": why,
            }
        return out
