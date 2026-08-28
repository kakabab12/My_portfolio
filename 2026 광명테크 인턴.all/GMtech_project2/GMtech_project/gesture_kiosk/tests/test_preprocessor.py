"""preprocessor 단위 테스트 — 인식 대상 외 블러(2026-08-07 신설, 실험적).

카메라·모델 없이 순수 함수(anchor_keep_sharp_box·blur_outside_region·
blur_outside_mask)만 검증한다 — 나머지(Preprocessor.preprocess_frame)는
기존에 테스트가 없던 자명한 거울 반전 래퍼라 범위 밖.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.preprocessor import (
    anchor_keep_sharp_box, arm_reach_mask, blur_outside_mask, blur_outside_region,
    combine_reach_masks, hand_box_reach_mask, keep_component_at,
)

FRAME_WIDTH_PX = 1280
FRAME_HEIGHT_PX = 720


class AnchorKeepSharpBoxTest(unittest.TestCase):
    def test_none_anchor_returns_none(self):
        self.assertIsNone(
            anchor_keep_sharp_box(None, 1.6, FRAME_WIDTH_PX, FRAME_HEIGHT_PX))

    def test_zero_width_returns_none(self):
        self.assertIsNone(
            anchor_keep_sharp_box((640.0, 360.0, 0.0), 1.6,
                                  FRAME_WIDTH_PX, FRAME_HEIGHT_PX))

    def test_centered_box_matches_ratio(self):
        # 어깨너비 200px × 1.6배 = 320px 한 변 -> 중심에서 ±160px
        box = anchor_keep_sharp_box((640.0, 360.0, 200.0), 1.6,
                                    FRAME_WIDTH_PX, FRAME_HEIGHT_PX)
        self.assertEqual(box, (480, 200, 800, 520))

    def test_box_clamped_to_frame_bounds(self):
        # 화면 구석 근처 앵커 — 박스가 프레임 밖으로 안 나가고 잘린다
        box = anchor_keep_sharp_box((20.0, 20.0, 200.0), 1.6,
                                    FRAME_WIDTH_PX, FRAME_HEIGHT_PX)
        x1, y1, x2, y2 = box
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, FRAME_WIDTH_PX)
        self.assertLessEqual(y2, FRAME_HEIGHT_PX)


class BlurOutsideRegionTest(unittest.TestCase):
    def _checkerboard_frame(self):
        # 흐림 여부를 눈으로도 코드로도 확인하기 쉬운 고대비 체크무늬
        frame = np.zeros((FRAME_HEIGHT_PX, FRAME_WIDTH_PX, 3), dtype=np.uint8)
        frame[::2, ::2] = 255
        frame[1::2, 1::2] = 255
        return frame

    def test_none_region_returns_frame_unchanged(self):
        frame = self._checkerboard_frame()
        result = blur_outside_region(frame, None)
        np.testing.assert_array_equal(result, frame)

    def test_inside_region_stays_sharp(self):
        frame = self._checkerboard_frame()
        region = (480, 200, 800, 520)
        result = blur_outside_region(frame, region)
        x1, y1, x2, y2 = region
        np.testing.assert_array_equal(result[y1:y2, x1:x2], frame[y1:y2, x1:x2])

    def test_outside_region_is_blurred(self):
        frame = self._checkerboard_frame()
        region = (480, 200, 800, 520)
        result = blur_outside_region(frame, region)
        # 체크무늬가 블러되면 더 이상 원본과 같지 않아야 한다(고주파 성분 손실)
        self.assertFalse(np.array_equal(result[0:100, 0:100], frame[0:100, 0:100]))
        # 원본 미변형 — 블러는 사본에만 적용된다
        np.testing.assert_array_equal(frame, self._checkerboard_frame())


class BlurOutsideMaskTest(unittest.TestCase):
    def _checkerboard_frame(self):
        frame = np.zeros((FRAME_HEIGHT_PX, FRAME_WIDTH_PX, 3), dtype=np.uint8)
        frame[::2, ::2] = 255
        frame[1::2, 1::2] = 255
        return frame

    def _box_mask(self, region):
        x1, y1, x2, y2 = region
        mask = np.zeros((FRAME_HEIGHT_PX, FRAME_WIDTH_PX), dtype=np.float32)
        mask[y1:y2, x1:x2] = 1.0
        return mask

    def test_none_mask_returns_frame_unchanged(self):
        frame = self._checkerboard_frame()
        result = blur_outside_mask(frame, None, 0.5)
        np.testing.assert_array_equal(result, frame)

    def test_mismatched_shape_returns_frame_unchanged(self):
        # 방어적 처리 — 마스크와 프레임 크기가 안 맞으면(해상도 불일치 등)
        # 블러 없이 원본 반환. 호출부(realtime_loop)가 이럴 때 사각형 폴백으로 넘어간다
        frame = self._checkerboard_frame()
        wrong_shape_mask = np.ones((10, 10), dtype=np.float32)
        result = blur_outside_mask(frame, wrong_shape_mask, 0.5)
        np.testing.assert_array_equal(result, frame)

    def test_inside_mask_stays_sharp(self):
        frame = self._checkerboard_frame()
        region = (480, 200, 800, 520)
        result = blur_outside_mask(frame, self._box_mask(region), 0.5)
        x1, y1, x2, y2 = region
        np.testing.assert_array_equal(result[y1:y2, x1:x2], frame[y1:y2, x1:x2])

    def test_outside_mask_is_blurred(self):
        frame = self._checkerboard_frame()
        region = (480, 200, 800, 520)
        result = blur_outside_mask(frame, self._box_mask(region), 0.5)
        self.assertFalse(np.array_equal(result[0:100, 0:100], frame[0:100, 0:100]))
        # 원본 미변형 — 블러는 사본에만 적용된다
        np.testing.assert_array_equal(frame, self._checkerboard_frame())

    def test_two_disjoint_regions_can_both_stay_sharp(self):
        # 사각형 폴백(blur_outside_region)과 달리 마스크는 연결 안 된 두 영역도
        # 동시에 선명하게 남길 수 있다 — 옆에 붙어 앉은 사람을 갈라내는 핵심 능력
        frame = self._checkerboard_frame()
        mask = np.zeros((FRAME_HEIGHT_PX, FRAME_WIDTH_PX), dtype=np.float32)
        mask[100:150, 100:150] = 1.0
        mask[500:550, 900:950] = 1.0
        result = blur_outside_mask(frame, mask, 0.5)
        np.testing.assert_array_equal(result[100:150, 100:150], frame[100:150, 100:150])
        np.testing.assert_array_equal(result[500:550, 900:950], frame[500:550, 900:950])
        # 두 선명 영역 사이(마스크 0)는 흐려진다 — 사각형이면 이 사이도 다 남았을 것
        self.assertFalse(
            np.array_equal(result[300:350, 500:550], frame[300:350, 500:550]))

    def test_bound_box_limits_mask_to_one_person(self):
        # ★1인 제한(2026-08-07 사용자 지적 — "블러 사람 한명만"): 마스크가 두
        # 사람을 덮어도(실측상 상대편으로 ~7% 유출이 있다) 선명 영역은 상자를
        # 넘지 못한다. 왼쪽=앵커, 오른쪽=옆 사람으로 두고 오른쪽이 흐려지는지 본다
        frame = self._checkerboard_frame()
        mask = np.zeros((FRAME_HEIGHT_PX, FRAME_WIDTH_PX), dtype=np.float32)
        mask[300:400, 100:200] = 1.0     # 앵커 사람
        mask[300:400, 900:1000] = 1.0    # 마스크가 새어 나간 옆 사람
        result = blur_outside_mask(frame, mask, 0.5, dilate_px=0,
                                   bound_box=(0, 0, 640, FRAME_HEIGHT_PX))
        np.testing.assert_array_equal(result[300:400, 100:200], frame[300:400, 100:200])
        self.assertFalse(
            np.array_equal(result[300:400, 900:1000], frame[300:400, 900:1000]))

    def test_exclude_mask_blurs_the_other_person(self):
        # ★"앵커 안 잡힌 사람은 반드시 흐리게"(2026-08-07 사용자 요청):
        # 앵커 마스크가 상대편으로 새더라도(가까이 붙을수록 심해진다)
        # 상대 실루엣을 빼면 그 부분은 흐려진다
        frame = self._checkerboard_frame()
        anchor = np.zeros((FRAME_HEIGHT_PX, FRAME_WIDTH_PX), dtype=np.float32)
        anchor[300:400, 100:300] = 1.0            # 앵커 + 상대편으로 샌 영역 포함
        other = np.zeros((FRAME_HEIGHT_PX, FRAME_WIDTH_PX), dtype=np.float32)
        other[300:400, 200:300] = 1.0             # 그중 오른쪽 절반이 상대 몸
        result = blur_outside_mask(frame, anchor, 0.5, exclude_mask=other)
        # 앵커만의 영역 — 선명
        np.testing.assert_array_equal(result[300:400, 100:200], frame[300:400, 100:200])
        # 상대 실루엣과 겹친 영역 — 흐리는 쪽으로 판정
        self.assertFalse(
            np.array_equal(result[300:400, 200:300], frame[300:400, 200:300]))

    def test_exclude_applies_after_dilation(self):
        # 순서 보증 — 팽창이 상대 몸으로 번져도 그 뒤에 빼므로 남지 않는다.
        # 먼저 빼면 팽창이 도로 덮어써 뺀 의미가 없어진다
        frame = self._checkerboard_frame()
        anchor = np.zeros((FRAME_HEIGHT_PX, FRAME_WIDTH_PX), dtype=np.float32)
        anchor[300:400, 100:200] = 1.0
        other = np.zeros((FRAME_HEIGHT_PX, FRAME_WIDTH_PX), dtype=np.float32)
        other[300:400, 200:300] = 1.0             # 앵커 바로 오른쪽에 붙어 있다
        result = blur_outside_mask(frame, anchor, 0.5, dilate_px=40, exclude_mask=other)
        probe = np.s_[340:360, 205:215]           # 팽창이 번졌을 상대 몸 위
        self.assertFalse(np.array_equal(result[probe], frame[probe]))

    def test_exclude_mask_none_keeps_behaviour(self):
        frame = self._checkerboard_frame()
        mask = self._box_mask((100, 300, 300, 400))
        with_none = blur_outside_mask(frame, mask, 0.5, exclude_mask=None)
        plain = blur_outside_mask(frame, mask, 0.5)
        np.testing.assert_array_equal(with_none, plain)

    def test_bound_box_none_keeps_full_mask(self):
        # 상자를 안 주면 종전 동작(마스크 전체가 선명) — 회귀 방지
        frame = self._checkerboard_frame()
        mask = self._box_mask((900, 300, 1000, 400))
        result = blur_outside_mask(frame, mask, 0.5)
        np.testing.assert_array_equal(result[300:400, 900:1000], frame[300:400, 900:1000])

    def test_threshold_boundary(self):
        frame = self._checkerboard_frame()
        mask = np.full((FRAME_HEIGHT_PX, FRAME_WIDTH_PX), 0.4, dtype=np.float32)
        below = blur_outside_mask(frame, mask, 0.5)   # 0.4 < 0.5 -> 전부 블러
        self.assertFalse(np.array_equal(below, frame))
        above = blur_outside_mask(frame, mask, 0.3)   # 0.4 >= 0.3 -> 전부 선명
        np.testing.assert_array_equal(above, frame)

    def test_dilate_zero_matches_omitted_default(self):
        # dilate_px 기본값(0)이 실수로 바뀌면 여기서 잡힌다 — 종전 동작 보증
        frame = self._checkerboard_frame()
        mask = self._box_mask((480, 200, 800, 520))
        omitted = blur_outside_mask(frame, mask, 0.5)
        explicit_zero = blur_outside_mask(frame, mask, 0.5, dilate_px=0)
        np.testing.assert_array_equal(omitted, explicit_zero)

    def test_dilate_grows_sharp_region_beyond_raw_mask(self):
        # ★2026-08-07 3차(사용자 실기 보고 — "손가락 손바닥 쪽에서 블러를 잘
        # 못 잡는다"): 손끝처럼 작은/얇은 영역은 raw 마스크 경계 바로 밖이
        # 블러되기 쉽다 — dilate_px를 주면 그 경계 밖 몇 픽셀까지 선명하게 살아야 한다
        frame = self._checkerboard_frame()
        mask = self._box_mask((600, 300, 620, 320))   # 20x20 작은 영역(손끝 크기)
        probe = np.s_[309:312, 592:595]   # 원본 박스(x>=600) 바로 왼쪽 밖 — 7px 여유
        no_dilate = blur_outside_mask(frame, mask, 0.5, dilate_px=0)
        self.assertFalse(np.array_equal(no_dilate[probe], frame[probe]))
        dilated = blur_outside_mask(frame, mask, 0.5, dilate_px=20)
        np.testing.assert_array_equal(dilated[probe], frame[probe])


class ArmReachMaskTest(unittest.TestCase):
    """옆 사람 차단 — 앵커의 팔 주변만 선명하게 (2026-08-07 신설).

    옆 사람은 거리가 같아 깊이 관문이 안 듣고, 사각형은 내 도달을 담으려면
    커져서 옆 사람까지 덮는다. 판정 영역이 팔을 따라다니게 해서 가른다.
    """

    def _checkerboard_frame(self):
        frame = np.zeros((FRAME_HEIGHT_PX, FRAME_WIDTH_PX, 3), dtype=np.uint8)
        frame[::2, ::2] = 255
        frame[1::2, 1::2] = 255
        return frame

    def test_none_when_no_points(self):
        # 손목을 못 믿으면 제약을 걸지 않는다 — 내 손을 자르느니 방어를 거른다
        self.assertIsNone(arm_reach_mask((720, 1280), [], 100))

    def test_none_when_radius_zero(self):
        self.assertIsNone(arm_reach_mask((720, 1280), [(100, 100)], 0))

    def test_covers_near_joint_only(self):
        reach = arm_reach_mask((FRAME_HEIGHT_PX, FRAME_WIDTH_PX), [(300, 400)], 120)
        self.assertTrue(reach[400, 300])       # 관절 위
        self.assertTrue(reach[400, 400])       # 반경 안(100px)
        self.assertFalse(reach[400, 900])      # 옆 사람 자리

    def test_neighbour_hand_is_blurred_but_mine_stays(self):
        # ★핵심 시나리오: 앵커 마스크가 옆 사람 손까지 덮어버려도, 그 손은
        # 내 팔 관절에서 멀어 흐려진다. 내 손목 옆의 내 손은 살아남는다
        frame = self._checkerboard_frame()
        mask = np.zeros((FRAME_HEIGHT_PX, FRAME_WIDTH_PX), dtype=np.float32)
        mask[350:450, 250:350] = 1.0     # 내 손(내 손목 옆)
        mask[350:450, 850:950] = 1.0     # 옆 사람 손 — 마스크가 잘못 덮었다
        reach = arm_reach_mask((FRAME_HEIGHT_PX, FRAME_WIDTH_PX), [(300, 400)], 150)
        result = blur_outside_mask(frame, mask, 0.5, reach_mask=reach)
        np.testing.assert_array_equal(result[360:440, 260:340], frame[360:440, 260:340])
        self.assertFalse(
            np.array_equal(result[350:450, 850:950], frame[350:450, 850:950]))

    def test_no_reach_mask_keeps_previous_behaviour(self):
        frame = self._checkerboard_frame()
        mask = np.zeros((FRAME_HEIGHT_PX, FRAME_WIDTH_PX), dtype=np.float32)
        mask[350:450, 850:950] = 1.0
        result = blur_outside_mask(frame, mask, 0.5, reach_mask=None)
        np.testing.assert_array_equal(result[350:450, 850:950], frame[350:450, 850:950])


class HandBoxReachMaskTest(unittest.TestCase):
    """관절 반경의 사각지대 보강 — 손의 최근 위치 자체를 반경 기준으로 (2026-08-07 4차).

    사용자 보고: "제스처를 해도 블러 때문에 초록색 관절값이 사라진다". 실측
    화면 캡처로 확인하니, 손을 들거나 펴는 정상 동작만으로도 손끝이 몸통
    관절 기준 반경 경계에 바로 걸렸다. 손이 실제로 있던 자리(추적 박스)
    자체를 반경 기준으로 삼아 그 사각지대를 메운다.
    """

    def test_none_when_no_box(self):
        self.assertIsNone(hand_box_reach_mask((720, 1280), None, 1.4))

    def test_none_when_ratio_zero(self):
        self.assertIsNone(hand_box_reach_mask((720, 1280), (100, 100, 200, 200), 0))

    def test_covers_the_box_with_margin(self):
        # ratio=1.0이면 원이 박스 모서리에 딱 닿는 최소값 — 그보다 커야 진짜 여유다
        box = (300, 300, 400, 400)  # 100x100 정사각 박스, 중심 (350,350)
        reach = hand_box_reach_mask((720, 1280), box, 1.4)
        self.assertTrue(reach[350, 350])         # 박스 중심
        self.assertTrue(reach[300, 300])         # 박스 모서리 — 반경 안에 들어야 함
        self.assertFalse(reach[350, 900])        # 멀리 떨어진 곳(옆 사람 자리)

    def test_gesture_that_extends_past_joint_radius_still_kept(self):
        # ★핵심 회귀 재현: 손끝이 관절 반경(joint) 밖으로 나가도, 추적 박스
        # 반경(hand)이 있으면 살아남아야 한다 — 사용자가 실제로 겪은 증상
        joint_reach = arm_reach_mask((720, 1280), [(300, 400)], 80)   # 좁은 관절 반경
        self.assertFalse(joint_reach[100, 700])   # 손끝(관절에서 멀리 뻗은 지점) — 관절 반경 밖
        hand_box = (650, 50, 750, 150)             # 손끝이 실제로 있는 자리(추적 박스)
        hand_reach = hand_box_reach_mask((720, 1280), hand_box, 1.4)
        combined = combine_reach_masks(joint_reach, hand_reach)
        self.assertTrue(combined[100, 700])       # 합치면 손끝이 살아난다


class CombineReachMasksTest(unittest.TestCase):
    def test_all_none_returns_none(self):
        self.assertIsNone(combine_reach_masks(None, None))

    def test_single_mask_passthrough(self):
        m = arm_reach_mask((720, 1280), [(100, 100)], 50)
        result = combine_reach_masks(None, m, None)
        np.testing.assert_array_equal(result, m)

    def test_union_not_intersection(self):
        # 서로 안 겹치는 두 반경을 합치면 **둘 다** 남아야 한다(교집합이면 전부 사라짐)
        m1 = arm_reach_mask((720, 1280), [(100, 100)], 30)
        m2 = arm_reach_mask((720, 1280), [(900, 600)], 30)
        combined = combine_reach_masks(m1, m2)
        self.assertTrue(combined[100, 100])
        self.assertTrue(combined[600, 900])


class KeepComponentAtTest(unittest.TestCase):
    """연결 성분 1인 보장 — 앵커 몸과 이어지지 않은 덩어리는 통째로 뺀다."""

    def _two_blobs(self):
        binary = np.zeros((FRAME_HEIGHT_PX, FRAME_WIDTH_PX), dtype=bool)
        binary[300:400, 100:200] = True     # 앵커
        binary[300:400, 900:1000] = True    # 떨어져 선 다른 사람
        return binary

    def test_keeps_only_the_seeded_blob(self):
        kept = keep_component_at(self._two_blobs(), (150, 350))
        self.assertTrue(kept[350, 150])       # 앵커 덩어리 유지
        self.assertFalse(kept[350, 950])      # 떨어진 덩어리 제거

    def test_none_seed_returns_input(self):
        binary = self._two_blobs()
        np.testing.assert_array_equal(keep_component_at(binary, None), binary)

    def test_seed_outside_mask_returns_input(self):
        # 앵커 중심이 실루엣 밖(관측이 어긋난 상태) — 엉뚱한 덩어리를 고르느니
        # 손대지 않는다
        binary = self._two_blobs()
        np.testing.assert_array_equal(keep_component_at(binary, (640, 100)), binary)

    def test_seed_out_of_bounds_returns_input(self):
        binary = self._two_blobs()
        np.testing.assert_array_equal(keep_component_at(binary, (-5, 350)), binary)

    def test_hand_with_small_gap_is_not_dropped(self):
        # ★사용자 보고 회귀 방지(2026-08-07) — "뒷사람이 있으면 손 관절값도
        # 안 보인다". 손목에서 마스크가 몇 화소 끊기면 손이 별개 덩어리가 되어
        # 통째로 버려졌다. 작은 틈은 이어서 판정해야 한다
        binary = np.zeros((FRAME_HEIGHT_PX, FRAME_WIDTH_PX), dtype=bool)
        binary[300:400, 100:200] = True     # 몸
        binary[330:370, 206:260] = True     # 손 — 6px 틈을 두고 떨어져 있다
        kept = keep_component_at(binary, (150, 350))
        self.assertTrue(kept[350, 230])     # 손이 살아남아야 한다

    def test_far_blob_still_dropped_despite_bridging(self):
        # 틈 메우기가 너무 관대해져 멀리 있는 사람까지 붙이면 안 된다
        kept = keep_component_at(self._two_blobs(), (150, 350))
        self.assertFalse(kept[350, 950])

    def test_single_blob_is_untouched(self):
        binary = np.zeros((FRAME_HEIGHT_PX, FRAME_WIDTH_PX), dtype=bool)
        binary[300:400, 100:200] = True
        np.testing.assert_array_equal(keep_component_at(binary, (150, 350)), binary)

    def test_blur_drops_disconnected_person(self):
        # ★통합 확인: 마스크가 뒷사람까지 덮어도, 앵커 몸과 안 이어지면
        # 블러 결과에서 그 사람은 흐려진다
        frame = self._checkerboard_frame()
        mask = np.zeros((FRAME_HEIGHT_PX, FRAME_WIDTH_PX), dtype=np.float32)
        mask[300:400, 100:200] = 1.0
        mask[300:400, 900:1000] = 1.0
        result = blur_outside_mask(frame, mask, 0.5, seed_point=(150, 350))
        np.testing.assert_array_equal(result[300:400, 100:200], frame[300:400, 100:200])
        self.assertFalse(
            np.array_equal(result[300:400, 900:1000], frame[300:400, 900:1000]))

    def _checkerboard_frame(self):
        frame = np.zeros((FRAME_HEIGHT_PX, FRAME_WIDTH_PX, 3), dtype=np.uint8)
        frame[::2, ::2] = 255
        frame[1::2, 1::2] = 255
        return frame


if __name__ == "__main__":
    unittest.main()
