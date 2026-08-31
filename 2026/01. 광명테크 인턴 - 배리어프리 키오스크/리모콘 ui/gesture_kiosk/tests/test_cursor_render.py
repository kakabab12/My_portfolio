"""커서 그리기 검증 (2026-08-31 신설).

가장 중요한 것은 test_everything_drawn_fits_inside_reach 다. 각 트래커는
커서 주변 사각형만 지우고 다시 그린다(더티 사각형). 그 사각형을 정하는 값이
cursor_reach_px()인데, 실제로 칠하는 범위가 그보다 넓으면 **지워지지 않은
자국이 화면에 남는다**. 커서 모양을 바꿀 때마다 사람이 눈으로 확인하는 대신
여기서 자동으로 걸리게 한다.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.cursor_render import (   # noqa: E402
    CURSOR_MARKER_SIZE_PX, CURSOR_RADIUS_PX, CURSOR_THICKNESS_PX,
    cursor_reach_px, draw_cursor,
)

BACKGROUND = (255, 0, 255)   # 트래커가 쓰는 크로마키 색과 같은 취지
CURSOR_COLOR = (0, 220, 0)


def _canvas(w=400, h=400):
    return np.full((h, w, 3), BACKGROUND, dtype=np.uint8)


def _painted_bounds(canvas):
    """배경색이 아닌 픽셀의 경계 (x0, y0, x1, y1). 없으면 None."""
    mask = np.any(canvas != np.array(BACKGROUND, dtype=np.uint8), axis=2)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


@pytest.mark.parametrize("kwargs", [
    {},
    {"filled": True},
    {"recenter_progress_ratio": 0.5},
    {"recenter_progress_ratio": 1.0},
    {"filled": True, "recenter_progress_ratio": 0.75},
])
def test_everything_drawn_fits_inside_reach(kwargs):
    """★칠한 픽셀이 전부 cursor_reach_px() 안에 들어와야 한다.

    벗어나면 더티 사각형이 못 지워 잔상이 남는다.
    """
    canvas = _canvas()
    h, w = canvas.shape[:2]
    draw_cursor(canvas, 0.5, 0.5, color=CURSOR_COLOR, **kwargs)

    bounds = _painted_bounds(canvas)
    assert bounds is not None
    cx, cy = w * 0.5, h * 0.5
    reach = cursor_reach_px()
    assert bounds[0] >= cx - reach, "왼쪽이 더티 사각형을 벗어났습니다"
    assert bounds[1] >= cy - reach, "위쪽이 더티 사각형을 벗어났습니다"
    assert bounds[2] <= cx + reach, "오른쪽이 더티 사각형을 벗어났습니다"
    assert bounds[3] <= cy + reach, "아래쪽이 더티 사각형을 벗어났습니다"


def test_reach_is_not_wastefully_large():
    """반대로 너무 넉넉해도 안 된다 — 지우는 넓이가 그만큼 늘어난다.

    실제로 칠하는 범위의 1.4배 안이면 충분하다.
    """
    canvas = _canvas()
    h, w = canvas.shape[:2]
    draw_cursor(canvas, 0.5, 0.5, color=CURSOR_COLOR, recenter_progress_ratio=1.0)
    bounds = _painted_bounds(canvas)
    actual = max(w * 0.5 - bounds[0], bounds[2] - w * 0.5,
                 h * 0.5 - bounds[1], bounds[3] - h * 0.5)
    assert cursor_reach_px() <= actual * 1.4


def test_none_position_draws_nothing():
    canvas = _canvas()
    draw_cursor(canvas, None, 0.5)
    draw_cursor(canvas, 0.5, None)
    assert _painted_bounds(canvas) is None


def test_cursor_is_centered_on_the_requested_point():
    canvas = _canvas()
    h, w = canvas.shape[:2]
    draw_cursor(canvas, 0.25, 0.75, color=CURSOR_COLOR)
    x0, y0, x1, y1 = _painted_bounds(canvas)
    assert (x0 + x1) / 2 == pytest.approx(w * 0.25, abs=2.0)
    assert (y0 + y1) / 2 == pytest.approx(h * 0.75, abs=2.0)


def test_subpixel_movement_changes_the_image():
    """★부분 픽셀 — 1픽셀보다 작게 움직여도 그림이 달라져야 한다.

    예전에는 int()로 잘라서 1픽셀 미만 이동이 통째로 사라졌고, 커서를 천천히
    움직이면 툭툭 건너뛰었다. 이 테스트가 그 회귀를 막는다.
    """
    a = _canvas()
    b = _canvas()
    draw_cursor(a, 0.5, 0.5, color=CURSOR_COLOR)
    draw_cursor(b, 0.5 + 0.3 / a.shape[1], 0.5, color=CURSOR_COLOR)   # 0.3px 이동
    assert not np.array_equal(a, b)


def test_dark_halo_makes_cursor_visible_on_matching_background():
    """★대비 테두리 — 커서와 같은 색 배경에서도 형태가 보여야 한다.

    초록 커서를 초록 배경에 그리면 테두리가 없을 때 사라진다. 배리어프리
    키오스크는 어떤 화면 위에 올라갈지 모르므로 이게 실제 문제다.
    """
    green_bg = np.full((300, 300, 3), CURSOR_COLOR, dtype=np.uint8)
    draw_cursor(green_bg, 0.5, 0.5, color=CURSOR_COLOR)
    # 배경과 다른 픽셀(= 어두운 테두리)이 충분히 있어야 한다
    differing = np.any(green_bg != np.array(CURSOR_COLOR, dtype=np.uint8), axis=2)
    assert differing.sum() > 200


def test_filled_and_outline_differ():
    """드래그 중(속 채움)은 형태로 구분돼야 한다 — 색맹 사용자 대비."""
    a, b = _canvas(), _canvas()
    draw_cursor(a, 0.5, 0.5, color=CURSOR_COLOR, filled=False)
    draw_cursor(b, 0.5, 0.5, color=CURSOR_COLOR, filled=True)
    painted_a = np.any(a != np.array(BACKGROUND, dtype=np.uint8), axis=2).sum()
    painted_b = np.any(b != np.array(BACKGROUND, dtype=np.uint8), axis=2).sum()
    assert painted_b > painted_a


def test_progress_ring_grows_with_ratio():
    """진행 링이 비율만큼 그려져야 한다 — 남은 시간을 눈으로 알리는 표시다."""
    counts = []
    for ratio in (0.25, 0.5, 1.0):
        canvas = _canvas()
        draw_cursor(canvas, 0.5, 0.5, color=CURSOR_COLOR, recenter_progress_ratio=ratio)
        counts.append(np.any(canvas != np.array(BACKGROUND, dtype=np.uint8), axis=2).sum())
    assert counts[0] < counts[1] < counts[2]


def test_draw_does_not_crash_at_canvas_edges():
    """화면 가장자리에서도 죽지 않아야 한다 (OpenCV가 알아서 잘라 준다)."""
    for x, y in ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)):
        canvas = _canvas()
        draw_cursor(canvas, x, y, color=CURSOR_COLOR, recenter_progress_ratio=0.5)


def test_geometry_constants_are_consistent():
    """십자가 원 밖으로 나가면 모양이 깨진다."""
    assert CURSOR_MARKER_SIZE_PX / 2 < CURSOR_RADIUS_PX
    assert CURSOR_THICKNESS_PX >= 1
    assert cursor_reach_px(progress_ring=False) < cursor_reach_px(progress_ring=True)


# ------------------------------- 투명색(마젠타) 오염 — 실기 보고 "커서 뒤 분홍색"

KEY_COLOR = (255, 0, 255)   # 오버레이 투명색 (TRANSPARENT_KEY_COLOR와 같은 값)


def _blended_with_key(canvas):
    """투명색과 섞인 중간색 픽셀 — 화면에 분홍 테두리로 보이는 것들.

    LWA_COLORKEY는 지정색과 **정확히 일치**하는 픽셀만 투명하게 만든다.
    안티에일리어싱이 만든 중간색은 그 조건을 못 맞춰 그대로 보인다.
    "파랑과 빨강이 둘 다 높은데 정확히 투명색은 아닌" 픽셀을 찾는다.
    """
    b = canvas[:, :, 0].astype(int)
    g = canvas[:, :, 1].astype(int)
    r = canvas[:, :, 2].astype(int)
    exact_key = (b == 255) & (g == 0) & (r == 255)
    pinkish = (b > 100) & (r > 100) & (b - g > 60) & (r - g > 60)
    return int((pinkish & ~exact_key).sum())


@pytest.mark.parametrize("kwargs", [
    {},
    {"filled": True},
    {"recenter_progress_ratio": 0.6},
    {"filled": True, "recenter_progress_ratio": 1.0},
])
def test_no_pixels_blended_with_the_transparency_key(kwargs):
    """★커서 둘레에 분홍 테두리가 남으면 안 된다 (2026-08-31 실기 보고).

    오버레이는 마젠타를 투명색으로 쓴다. 안티에일리어싱이 커서 색과 마젠타를
    섞으면 그 픽셀은 투명 처리가 안 돼 분홍으로 보인다. 바깥을 향한 요소는
    안티에일리어싱을 끄는 것으로 고쳤고, 이 시험이 그 회귀를 막는다.
    """
    canvas = np.full((300, 300, 3), KEY_COLOR, dtype=np.uint8)
    draw_cursor(canvas, 0.5, 0.5, color=CURSOR_COLOR, **kwargs)
    assert _blended_with_key(canvas) == 0


def test_no_blending_at_subpixel_positions():
    """부분 픽셀 위치에서도 투명색이 오염되면 안 된다 — 커서는 계속 움직인다."""
    for offset in (0.0, 0.13, 0.37, 0.5, 0.71, 0.94):
        canvas = np.full((300, 300, 3), KEY_COLOR, dtype=np.uint8)
        draw_cursor(canvas, 0.5 + offset / 300.0, 0.5 + offset / 300.0,
                    color=CURSOR_COLOR)
        assert _blended_with_key(canvas) == 0, offset


def test_inner_edges_are_still_antialiased():
    """바깥만 딱 떨어지게 하고, 안쪽 곡선은 여전히 매끄러워야 한다.

    본체 원은 검은 테두리 위에 그려지므로 안티에일리어싱을 그대로 쓴다 —
    초록과 검정 사이의 중간색이 있어야 정상이다.
    """
    canvas = np.full((300, 300, 3), KEY_COLOR, dtype=np.uint8)
    draw_cursor(canvas, 0.5, 0.5, color=CURSOR_COLOR)
    g = canvas[:, :, 1].astype(int)
    # 순수 초록(220)도 순수 검정(0)도 아닌 중간 밝기 픽셀 = 안티에일리어싱 흔적
    midtones = ((g > 30) & (g < 190)).sum()
    assert midtones > 50
