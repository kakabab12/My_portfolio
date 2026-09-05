"""화면 물리 크기 자동 인식 검증 (2026-09-05 신설).

커서가 "얼굴이 향한 곳"에 있으려면 화면이 실제로 얼마나 큰지 알아야 한다.
**픽셀 해상도로는 알 수 없다** — 1920x1080이 14인치 노트북일 수도 55인치
키오스크일 수도 있고 물리 크기는 4배 차이다. 그래서 운영체제가 모니터에서
읽어 온 값(EDID)을 쓴다. 이게 있어야 데스크탑과 키오스크가 한 빌드로 돈다.

여기서 보증하는 것은 전부 **"엉뚱한 값을 안 쓴다"** 쪽이다. 화면 크기를
틀리면 커서 감도가 통째로 틀어지므로, 의심스러우면 쓰지 않고 설정값이나
예전 방식으로 넘어가는 것이 맞다.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import display_size as DS                        # noqa: E402


# ── 말이 되는 값만 받아들인다 ──────────────────────────────────────────────

@pytest.mark.parametrize("w,h", [
    (310.0, 174.0),      # 14인치 노트북
    (531.0, 299.0),      # 24인치 모니터
    (700.0, 393.0),      # 32인치 키오스크
    (1210.0, 680.0),     # 55인치 사이니지
    (304.0, 228.0),      # 15인치 4:3 (옛 산업용 패널)
])
def test_real_screen_sizes_pass(w, h):
    assert DS._sane(w, h)


@pytest.mark.parametrize("w,h,why", [
    (0.0, 0.0, "드라이버가 0을 준다"),
    (None, None, "아예 없다"),
    (100.0, 56.0, "너무 작다 — 화면이 아니다"),
    (3000.0, 1700.0, "너무 크다"),
    (531.0, 20.0, "가로세로비가 말이 안 된다"),
    (200.0, 300.0, "세로가 더 길다 — 회전 화면이거나 잘못 읽었다"),
])
def test_nonsense_sizes_are_rejected(w, h, why):
    assert not DS._sane(w, h), why


# ── 못 알아내면 지어내지 않는다 ────────────────────────────────────────────

def test_returns_none_when_nothing_works(monkeypatch):
    DS.detect_screen_size_mm.cache_clear()
    monkeypatch.setattr(DS, "_from_edid", lambda: None)
    monkeypatch.setattr(DS, "_from_device_caps", lambda: None)
    try:
        assert DS.detect_screen_size_mm() is None
    finally:
        DS.detect_screen_size_mm.cache_clear()


def test_insane_values_are_thrown_away(monkeypatch):
    """드라이버가 96 DPI로 지어낸 값을 그대로 쓰면 커서가 통째로 틀어진다."""
    DS.detect_screen_size_mm.cache_clear()
    monkeypatch.setattr(DS, "_from_edid", lambda: (0.0, 0.0))
    monkeypatch.setattr(DS, "_from_device_caps", lambda: (5.0, 3.0))
    try:
        assert DS.detect_screen_size_mm() is None
    finally:
        DS.detect_screen_size_mm.cache_clear()


def test_edid_wins_over_driver(monkeypatch):
    """모니터가 말하는 값이 드라이버 계산값보다 믿을 만하다."""
    DS.detect_screen_size_mm.cache_clear()
    monkeypatch.setattr(DS, "_from_edid", lambda: (531.0, 299.0))
    monkeypatch.setattr(DS, "_from_device_caps", lambda: (344.0, 194.0))
    try:
        assert DS.detect_screen_size_mm() == (531.0, 299.0)
    finally:
        DS.detect_screen_size_mm.cache_clear()


def test_falls_back_to_driver_when_edid_missing(monkeypatch):
    DS.detect_screen_size_mm.cache_clear()
    monkeypatch.setattr(DS, "_from_edid", lambda: None)
    monkeypatch.setattr(DS, "_from_device_caps", lambda: (344.0, 194.0))
    try:
        assert DS.detect_screen_size_mm() == (344.0, 194.0)
    finally:
        DS.detect_screen_size_mm.cache_clear()


def test_exceptions_do_not_escape(monkeypatch):
    """인식에 실패했다고 프로그램이 죽으면 안 된다 — 커서 없이도 돌아야 한다."""
    def boom():
        raise OSError("WMI가 없다")
    DS.detect_screen_size_mm.cache_clear()
    monkeypatch.setattr(DS, "_from_edid", boom)
    monkeypatch.setattr(DS, "_from_device_caps", boom)
    try:
        assert DS.detect_screen_size_mm() is None
    finally:
        DS.detect_screen_size_mm.cache_clear()


def test_result_is_cached(monkeypatch):
    """EDID를 읽는 데 시간이 걸린다 — 트래커를 만들 때마다 부르면 안 된다."""
    calls = []

    def once():
        calls.append(1)
        return (531.0, 299.0)

    DS.detect_screen_size_mm.cache_clear()
    monkeypatch.setattr(DS, "_from_edid", once)
    try:
        DS.detect_screen_size_mm()
        DS.detect_screen_size_mm()
        DS.detect_screen_size_mm()
        assert len(calls) == 1
    finally:
        DS.detect_screen_size_mm.cache_clear()


# ── 실제 이 기계에서 ──────────────────────────────────────────────────────

def test_on_this_machine_it_either_works_or_says_it_cannot():
    """값이 나오면 말이 되는 값이어야 하고, 안 나오면 None이어야 한다.

    특정 기계의 화면 크기를 기대하지 않는다 — 이 시험은 어느 기계에서
    돌려도 같은 뜻이어야 한다.
    """
    DS.detect_screen_size_mm.cache_clear()
    try:
        size = DS.detect_screen_size_mm()
    finally:
        DS.detect_screen_size_mm.cache_clear()
    assert size is None or (len(size) == 2 and DS._sane(size[0], size[1]))
