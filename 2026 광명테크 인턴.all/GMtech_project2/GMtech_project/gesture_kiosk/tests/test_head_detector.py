"""head_detector 단위 테스트 — 카메라·mediapipe 없이 도는 부분만.

HeadDetector.__init__은 mediapipe를 임포트하고 모델을 로딩하므로 여기서는
__new__로 인스턴스만 만들고 필요한 속성을 직접 채운다 — 검증 대상인
_segmentation_mask_for는 result 객체만 보는 순수한 추출 로직이라 그것으로 충분하다.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.head_detector import HeadDetector


class FakeMaskImage:
    """mediapipe Image 대역 — numpy_view()가 **내부 버퍼의 뷰**를 주는 것까지 흉내낸다."""

    def __init__(self, array):
        self.array = array

    def numpy_view(self):
        return self.array


class FakeResult:
    def __init__(self, masks):
        self.segmentation_masks = masks


class SegmentationMaskExtractionTest(unittest.TestCase):
    def _detector(self, want_segmentation=True):
        detector = HeadDetector.__new__(HeadDetector)
        detector._want_segmentation = want_segmentation
        detector._segmentation_warn_logged = False
        return detector

    def test_returns_none_when_disabled(self):
        detector = self._detector(want_segmentation=False)
        result = FakeResult([FakeMaskImage(np.ones((4, 4), dtype=np.float32))])
        self.assertIsNone(detector._segmentation_mask_for(result, 0))

    def test_returns_none_without_masks(self):
        detector = self._detector()
        self.assertIsNone(detector._segmentation_mask_for(FakeResult([]), 0))

    def test_returns_none_when_index_out_of_range(self):
        # 사람이 2명 관측됐는데 마스크는 1개만 오는 경우 — 인덱스로 터지면 안 된다
        detector = self._detector()
        result = FakeResult([FakeMaskImage(np.ones((4, 4), dtype=np.float32))])
        self.assertIsNone(detector._segmentation_mask_for(result, 1))

    def test_mask_is_copied_not_a_view(self):
        # ★2026-08-07 코드 리뷰 지적의 회귀 방지. numpy_view()는 MediaPipe 내부
        # 버퍼의 뷰라, 그대로 넘기면 ①result가 GC될 때 버퍼가 사라지거나
        # ②MediaPipe가 버퍼를 재사용해 **저장해 둔 마스크가 조용히 다음 프레임
        # 내용으로 바뀐다**. 여기서는 ②를 원본 배열을 덮어써 재현한다 —
        # 복사본이면 안 바뀌어야 한다(스레드 간 저장·지연 사용이 전제라 필수)
        detector = self._detector()
        source = np.ones((4, 4), dtype=np.float32)
        extracted = detector._segmentation_mask_for(FakeResult([FakeMaskImage(source)]), 0)
        source[:] = 0.0                       # MediaPipe가 버퍼를 재사용한 상황
        np.testing.assert_array_equal(extracted, np.ones((4, 4), dtype=np.float32))

    def test_extraction_failure_returns_none_without_raising(self):
        # 마스크 추출 실패가 포즈 추론 스레드 전체를 죽이면 안 된다
        detector = self._detector()

        class Exploding:
            @property
            def segmentation_masks(self):
                raise RuntimeError("버전별 반환 형태 차이 등")

        self.assertIsNone(detector._segmentation_mask_for(Exploding(), 0))
        self.assertTrue(detector._segmentation_warn_logged)   # 로그는 1회만


if __name__ == "__main__":
    unittest.main()
