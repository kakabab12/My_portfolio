"""오버레이 한글 문구가 남김없이 지워지는지 (2026-08-31 신설).

사용자 실기 보고: "처음에 커서 뜨면 한글이 안 지워지던데."

원인은 두 곳에서 글자 범위를 **따로 계산**한 것이었다. put_korean_text는
글자 크기의 절반을 여백으로 두르고 아래로는 그 두 배를 쓰는데, 지울 목록을
만드는 쪽은 그것과 다른 식으로 계산해 두 줄 문구 기준 아래 36px·좌우 11px가
지울 범위 밖이었다. 이제 칠한 쪽이 범위를 돌려주고 부르는 쪽은 그걸 쓴다.

여기서는 그 계약을 지킨다 — put_korean_text가 돌려준 사각형만 배경색으로
되돌리면 캔버스가 **완전히** 처음 상태로 돌아와야 한다.
"""
import importlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BACKGROUND = (255, 0, 255)
TEXT_COLOR = (0, 220, 0)
TRACKERS = ("eyebrow", "forehead", "head")


def _canvas(w=900, h=500):
    return np.full((h, w, 3), BACKGROUND, dtype=np.uint8)


@pytest.mark.parametrize("module_name", TRACKERS)
@pytest.mark.parametrize("text,font_px", [
    ("자리 잡는 중입니다", 30),
    ("고개를 움직이면 다시 시작합니다", 22),
    ("카메라 신호가 없습니다 - 연결을 확인해주세요", 26),
    ("한", 40),
])
def test_returned_rect_covers_every_painted_pixel(module_name, text, font_px):
    """★돌려준 범위만 지우면 자국이 하나도 안 남아야 한다."""
    module = importlib.import_module(module_name)
    canvas = _canvas()
    rect = module.put_korean_text(canvas, text, (120, 200), font_px, TEXT_COLOR)
    assert rect is not None

    painted = np.any(canvas != np.array(BACKGROUND, dtype=np.uint8), axis=2)
    assert painted.sum() > 0, "글자가 아예 안 그려졌습니다"

    x0, y0, x1, y1 = rect
    canvas[y0:y1, x0:x1] = BACKGROUND          # 트래커의 _blank_rect와 같은 동작
    leftover = np.any(canvas != np.array(BACKGROUND, dtype=np.uint8), axis=2)
    assert leftover.sum() == 0, "지우고도 %d픽셀이 남았습니다" % leftover.sum()


@pytest.mark.parametrize("module_name", TRACKERS)
def test_two_line_message_erases_completely(module_name):
    """실제로 문제가 됐던 형태 — 두 줄 문구를 겹쳐 그리고 한 번에 지운다."""
    module = importlib.import_module(module_name)
    canvas = _canvas()
    line1_y, font1, font2 = 180, 30, 22
    line2_y = line1_y + font1 + 10
    rects = [
        module.put_korean_text(canvas, "자리 잡는 중입니다", (300, line1_y), font1, TEXT_COLOR),
        module.put_korean_text(canvas, "고개를 움직이면 다시 시작합니다", (250, line2_y), font2, TEXT_COLOR),
    ]
    assert all(r is not None for r in rects)
    for x0, y0, x1, y1 in rects:
        canvas[y0:y1, x0:x1] = BACKGROUND
    leftover = np.any(canvas != np.array(BACKGROUND, dtype=np.uint8), axis=2)
    assert leftover.sum() == 0, "두 줄 문구가 %d픽셀 남았습니다" % leftover.sum()


@pytest.mark.parametrize("module_name", TRACKERS)
def test_offscreen_text_returns_none(module_name):
    """캔버스 밖이면 None — 부르는 쪽이 지울 목록에 넣지 않도록."""
    module = importlib.import_module(module_name)
    canvas = _canvas(200, 200)
    assert module.put_korean_text(canvas, "밖", (5000, 5000), 20, TEXT_COLOR) is None


@pytest.mark.parametrize("module_name", TRACKERS)
def test_rect_is_clipped_to_the_canvas(module_name):
    """가장자리에 걸쳐도 돌려준 범위가 캔버스 안이어야 한다 —
    음수 좌표를 넘기면 numpy 슬라이스가 반대편을 지운다."""
    module = importlib.import_module(module_name)
    canvas = _canvas(300, 300)
    rect = module.put_korean_text(canvas, "가장자리", (-10, -10), 24, TEXT_COLOR)
    if rect is not None:
        x0, y0, x1, y1 = rect
        assert 0 <= x0 < x1 <= 300
        assert 0 <= y0 < y1 <= 300


# ----------------- 패널 배경 — 실기 보고 "글씨 주변에 핑크색 테두리" (2026-08-31 밤)

def _pinkish(canvas):
    """투명색과 섞인 분홍 픽셀 — 오버레이에서 글자 테두리로 보이는 것들."""
    b = canvas[:, :, 0].astype(int)
    g = canvas[:, :, 1].astype(int)
    r = canvas[:, :, 2].astype(int)
    exact_key = (b == 255) & (g == 0) & (r == 255)
    return int((((b > 100) & (r > 100) & (b - g > 60) & (r - g > 60)) & ~exact_key).sum())


@pytest.mark.parametrize("module_name", TRACKERS)
def test_panel_prevents_pink_fringe(module_name):
    """★패널을 깔면 투명색과 섞인 분홍 가장자리가 생기지 않아야 한다.

    글자의 안티에일리어싱이 마젠타 대신 패널색과 섞이게 하는 것이 수정의
    핵심이다. 패널 없이 그리면 분홍이 실제로 생기는 것도 함께 확인해
    이 테스트가 헛돌지 않음을 보장한다.
    """
    module = importlib.import_module(module_name)
    canvas = _canvas()
    module.put_korean_text(canvas, "자리 잡는 중입니다", (200, 120), 30, TEXT_COLOR)
    assert _pinkish(canvas) > 0, "패널 없이는 분홍이 생겨야 검증이 의미 있다"

    canvas2 = _canvas()
    module.put_korean_text(canvas2, "자리 잡는 중입니다", (200, 120), 30, TEXT_COLOR,
                           panel_color=(28, 28, 28))
    assert _pinkish(canvas2) == 0


@pytest.mark.parametrize("module_name", TRACKERS)
def test_panel_text_still_erases_completely(module_name):
    """패널을 깔아도 돌려준 범위만 지우면 남김없이 지워져야 한다 —
    패널이 반환 범위(ROI)를 정확히 채우므로 성립한다."""
    module = importlib.import_module(module_name)
    canvas = _canvas()
    rect = module.put_korean_text(canvas, "고개를 움직이면 다시 시작합니다",
                                  (150, 200), 22, TEXT_COLOR, panel_color=(28, 28, 28))
    assert rect is not None
    x0, y0, x1, y1 = rect
    canvas[y0:y1, x0:x1] = BACKGROUND
    leftover = np.any(canvas != np.array(BACKGROUND, dtype=np.uint8), axis=2)
    assert leftover.sum() == 0
