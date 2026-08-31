"""잔여 곡률 자동 소거 — 쓰는 동안 스스로 재서 빼는 온라인 보정 (2026-08-31 신설).

무엇을 푸는가
-------------
"카메라 위치·각도 때문에 커서가 포물선으로 움직인다. 측정 없이, 어떤 배치에서든
알아서 맞아야 한다."

상대 회전 매핑(head_orientation.py)이 곡률의 근본 원인(투영 왜곡)을 없앴고,
회전 재료를 변환행렬로 바꿔 더 줄였지만, **잔여 곡률이 0이라고 보장할 수는
없다**. MediaPipe의 정합 오차, 조명, 사람마다 다른 얼굴 모양 등 우리가 모델링
못 한 요인이 남는다. 예전 처방(ARC_COMPENSATION)은 그 잔여를 사람이 재서
상수로 넣는 것이었고, 그래서 카메라를 옮길 때마다 다시 재야 했다.

이 모듈은 그 측정을 **커서가 움직이는 동안 자동으로** 한다. 좌우 오프셋 x와
세로 오프셋 y를 계속 모아 2차 회귀(y = a + b·x + c·x²)를 돌리고, 얻은 2차
계수 c만큼을 세로에서 빼 준다. 이것은 제어공학에서 말하는 온라인 시스템
식별(system identification)의 가장 단순한 형태다 — 계통 오차의 모형(2차식)을
정해 두고, 그 계수를 관측으로 갱신한다.

왜 이게 안전한가 — 의도한 움직임을 지우지 않는 근거
---------------------------------------------------
빼는 것은 **2차항뿐**이다. 회귀에는 1차항(b)도 넣지만 빼지 않는다.

  · 대각선 이동(의도) — 1차 상관이라 b에 흡수되고, c에는 안 실린다.
  · 원 그리기(의도) — 같은 x에서 y가 위아래로 대칭이라 2차 적합이 0으로
    상쇄된다.
  · 세로 이동(의도) — x가 안 변하므로 갱신 자체가 안 일어난다(안전장치 ②).
  · 포물선(계통 오차) — 좌우로 훑을 때마다 같은 방향으로 휘므로 c에
    일관되게 쌓인다. 시간이 지나면 이것만 남는다.

안전장치
--------
  ① 표본이 충분히 모여야(MIN_SAMPLES) 갱신한다.
  ② 가로로 충분히 움직였을 때만(MIN_X_SPAN) 갱신한다 — 좁은 구간의 2차
     적합은 잡음이 계수를 지배한다.
  ③ 계수는 한 번에 확 바꾸지 않고 느리게 따라간다(COEF_ALPHA) — 커서가
     갑자기 다르게 움직이면 사용자가 원인을 알 수 없다.
  ④ 계수에 상한(MAX_COEF)을 둔다 — 병적인 표본이 들어와도 보정이 화면
     높이의 일정 비율을 넘지 못한다.
  ⑤ 적합도가 낮으면(R² < MIN_R2) 그 창은 버린다 — 애초에 포물선이 아닌
     데이터로 계수를 만들지 않는다. 8/28에 낮은 R²를 무시하고 진행했다가
     데인 것과 반대 방향의 같은 교훈이다.
"""
import math

# 갱신에 필요한 최소 표본 수. 30fps 기준 약 8초 — 그 사이 좌우 왕복이
# 두어 번은 들어온다
MIN_SAMPLES = 240

# 버퍼가 찬 뒤로는 이 간격(표본 수)마다 다시 적합한다. 30fps 기준 4초 —
# "실행하면 바로 적용"이 목표라, 처음 한 번만 신중히 기다리고(MIN_SAMPLES)
# 그 뒤로는 절반 간격으로 따라간다
FIT_INTERVAL = 120

# 이만큼 가로로 움직인 창에서만 적합한다 (탄젠트 단위 — half_span 15도면
# 화면 절반이 tan(15°)=0.27이므로, 0.15는 화면의 절반 이상을 훑었다는 뜻)
MIN_X_SPAN = 0.15

# 2차 적합이 이보다 못 맞으면 그 창은 버린다 — 포물선이 아닌 데이터
MIN_R2 = 0.15

# 계수를 새 적합값으로 얼마나 따라갈지 (한 창마다). 0.3이면 첫 적합 후
# 약 20초에 참값의 90%에 수렴하면서, 한 창이 오염돼도 커서가 확 틀어지지
# 않는다 (처음 0.2로 했더니 수렴이 느려 단위 테스트가 잡았다 — 6창이 지나도
# 참값의 67%였다)
COEF_ALPHA = 0.3

# 계수 상한 — |c|·(x최대)² 이 화면 세로의 절반을 넘지 못하게.
# x최대(클램프 후)가 대략 0.27이므로 2.0이면 보정 최대 0.146 (화면의 15%)
MAX_COEF = 2.0

# 링 버퍼 길이 — 오래된 표본은 밀려난다. 사용자가 자세를 바꾸면
# 옛 곡률 표본이 이만큼 지나 자연히 사라진다 (30fps 기준 약 16초)
WINDOW = 480


class OnlineArcCompensator:
    """세로 = 세로 - c·(가로)² 의 c를 관측으로 계속 추정한다.

    쓰는 법 (매 프레임):
        y_corrected = comp.update(x, y_raw)

    update가 갱신 판단까지 알아서 한다 — 호출부는 조건을 몰라도 된다.
    """

    def __init__(self):
        self.coef = 0.0
        self._xs = []
        self._ys = []
        self._since_fit = 0

    def reset(self):
        self.coef = 0.0
        self._xs = []
        self._ys = []
        self._since_fit = 0

    def update(self, offset_x, offset_y):
        """관측 1건 반영 -> 보정된 세로 오프셋.

        보정에는 **지금까지 확정된 계수**를 쓰고, 그 다음에 표본을 넣는다 —
        방금 들어온 표본이 곧바로 자기 보정에 쓰이는 순환을 막는다.
        """
        corrected = offset_y - self.coef * offset_x * offset_x

        # 표본은 원본(y_raw)으로 모은다 — 보정된 값으로 모으면 계수가
        # 자기 자신을 빼는 만큼 계속 자라거나 줄어드는 되먹임이 생긴다
        self._xs.append(offset_x)
        self._ys.append(offset_y)
        if len(self._xs) > WINDOW:
            self._xs.pop(0)
            self._ys.pop(0)
        self._since_fit += 1

        if self._since_fit >= FIT_INTERVAL and len(self._xs) >= MIN_SAMPLES:
            self._since_fit = 0
            self._maybe_refit()
        return corrected

    def _maybe_refit(self):
        xs, ys = self._xs, self._ys
        if (max(xs) - min(xs)) < MIN_X_SPAN:
            return                       # 안전장치 ② — 가로로 충분히 안 움직였다
        fit = _fit_quadratic(xs, ys)
        if fit is None:
            return
        c, r2 = fit
        if r2 < MIN_R2:
            return                       # 안전장치 ⑤ — 포물선이 아니다
        c = max(-MAX_COEF, min(MAX_COEF, c))
        # 안전장치 ③ — 느리게 따라간다
        self.coef += COEF_ALPHA * (c - self.coef)


def _fit_quadratic(xs, ys):
    """y = a + b·x + c·x² 최소제곱 -> (c, R²). measure_arc.py와 같은 방식.

    1차항 b를 함께 적합하는 이유는 모듈 독스트링 참고 — 의도한 대각선
    움직임이 b로 빠져나가야 c가 순수한 곡률만 담는다.
    """
    n = len(xs)
    if n < 6:
        return None
    s = [sum(x ** k for x in xs) for k in range(5)]
    t = [sum(y * (x ** k) for x, y in zip(xs, ys)) for k in range(3)]
    mat = [[s[0], s[1], s[2], t[0]],
           [s[1], s[2], s[3], t[1]],
           [s[2], s[3], s[4], t[2]]]
    for col in range(3):
        pivot_row = max(range(col, 3), key=lambda r: abs(mat[r][col]))
        if abs(mat[pivot_row][col]) < 1e-15:
            return None
        mat[col], mat[pivot_row] = mat[pivot_row], mat[col]
        pivot = mat[col][col]
        for j in range(col, 4):
            mat[col][j] /= pivot
        for r in range(3):
            if r == col:
                continue
            factor = mat[r][col]
            for j in range(col, 4):
                mat[r][j] -= factor * mat[col][j]
    a, b, c = mat[0][3], mat[1][3], mat[2][3]
    my = sum(ys) / n
    ss_res = sum((y - (a + b * x + c * x * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    if ss_tot < 1e-15:
        return None
    return c, 1.0 - ss_res / ss_tot
