"""face_anchor 단위 테스트 — "뒷사람" 방어(FaceAnchor) 검증.

카메라·모델 없이 FakeFace(양쪽 눈 바깥쪽 끝 랜드마크만 있는 대역)와 FakeClock만
으로 검증한다 — tests/test_hand_select.py와 동일 스타일(FakeClock, make_config,
feed 헬퍼).

기하 축약: 안구간거리(interocular_px)를 "가까움"의 자로 쓴다 — 값이 클수록
카메라에 가깝다. 기본 config(continuity_ratio=1.5, continuity_depth_ratio=0.75,
jump_reject_ratio=0.30, move_reject_ratio=0.6, median_count=3,
anchor_grace_sec=1.0)로 검증한다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.face_estimator import LMK_LEFT_EYE_OUTER, LMK_RIGHT_EYE_OUTER
from src.postprocess.face_anchor import FaceAnchor

FRAME_DT_SEC = 1.0 / 20.0   # 추론 스레드 대략 20Hz 가정


class FakeClock:
    def __init__(self):
        self.now_sec = 1000.0

    def __call__(self):
        return self.now_sec

    def tick(self, dt_sec):
        self.now_sec += dt_sec


class FakeFace:
    """양쪽 눈 바깥쪽 끝 랜드마크만 있는 대역 — FaceAnchor가 쓰는 두 점만 있으면 된다."""

    def __init__(self, center_xy, interocular_px, tag=None):
        cx, cy = center_xy
        half_px = interocular_px / 2.0
        self._points = {
            LMK_LEFT_EYE_OUTER: (cx - half_px, cy),
            LMK_RIGHT_EYE_OUTER: (cx + half_px, cy),
        }
        self.tag = tag   # 테스트에서 "어느 얼굴이 선택됐는지" 식별용(FaceAnchor는 안 씀)

    def landmark_px(self, index):
        return self._points[index]


def make_config(**overrides):
    face_anchor_cfg = {
        "continuity_ratio": 1.5,
        "continuity_depth_ratio": 0.75,
        "jump_reject_ratio": 0.30,
        "move_reject_ratio": 0.6,
        "median_count": 3,
        "anchor_grace_sec": 1.0,
    }
    face_anchor_cfg.update(overrides)
    return {"face_anchor": face_anchor_cfg}


def make_anchor(config=None):
    clock = FakeClock()
    anchor = FaceAnchor(config or make_config(), clock=clock)
    return anchor, clock


def feed(anchor, clock, frames):
    """프레임 목록(각 프레임 = FakeFace 목록) 공급 -> 각 프레임의 선택 결과 목록."""
    results = []
    for faces in frames:
        results.append(anchor.update(faces))
        clock.tick(FRAME_DT_SEC)
    return results


class AcquisitionTest(unittest.TestCase):
    def test_first_pick_is_nearest_face(self):
        # 안구간거리가 큰(=가까운) 얼굴이 사용자로 뽑힌다
        near = FakeFace((500, 400), 80, tag="near")
        far = FakeFace((900, 400), 50, tag="far")
        anchor, clock = make_anchor()
        chosen = feed(anchor, clock, [[near, far]])[0]
        self.assertEqual(chosen.tag, "near")

    def test_no_faces_returns_none(self):
        anchor, clock = make_anchor()
        self.assertIsNone(feed(anchor, clock, [[]])[0])


class StickyContinuityTest(unittest.TestCase):
    """핵심 방어 — 한 번 정한 사용자를, 다른(더 크게 보이는) 얼굴이 나타나도 계속 붙잡는다."""

    def test_bigger_distant_face_does_not_steal_anchor(self):
        user = FakeFace((500, 400), 80, tag="user")
        anchor, clock = make_anchor()
        feed(anchor, clock, [[user]])   # 사용자 먼저 획득

        # 다음 프레임: 사용자는 그대로, 훨씬 멀리서(연속 반경 밖) 뒷사람이 더
        # 크게(=더 가깝게) 등장 — 대기줄이 아니라 완전히 다른 위치라 이음
        # 후보조차 아니다
        intruder = FakeFace((900, 400), 150, tag="intruder")
        chosen = feed(anchor, clock, [[user, intruder]])[0]
        self.assertEqual(chosen.tag, "user")

    def test_anchor_follows_small_natural_movement(self):
        anchor, clock = make_anchor()
        frames = [[FakeFace((500 + 3 * i, 400), 80, tag="user")] for i in range(10)]
        results = feed(anchor, clock, frames)
        self.assertTrue(all(face is not None and face.tag == "user" for face in results))

    def test_intruder_near_anchor_position_but_farther_is_rejected(self):
        # 대기줄 상황 재현 — 뒷사람이 위치상으로는 사용자와 가깝지만(연속 반경 안),
        # 안구간거리가 뚜렷이 작다(더 멀다) -> _drop_farther_candidates가 걸러낸다
        user = FakeFace((500, 400), 80, tag="user")
        anchor, clock = make_anchor()
        feed(anchor, clock, [[user]])

        # continuity_depth_ratio=0.75 -> 80*0.75=60px 미만이면 제외된다. 55px로
        # 확실히 그 밑 — 위치는 사용자 바로 옆(연속 반경 1.5*80=120px 안)
        queue_person = FakeFace((560, 400), 55, tag="queue_person")
        chosen = feed(anchor, clock, [[user, queue_person]])[0]
        self.assertEqual(chosen.tag, "user")


class GraceAndReacquireTest(unittest.TestCase):
    def test_short_dropout_within_grace_reanchors_same_identity(self):
        user = FakeFace((500, 400), 80, tag="user")
        anchor, clock = make_anchor()
        feed(anchor, clock, [[user]])

        # 0.3초 소실(anchor_grace_sec=1.0보다 짧다) — 그동안 다른 사람이 화면에
        # 등장해도 앵커 자체는 살아있다
        clock.tick(0.3)
        someone_else = FakeFace((900, 400), 200, tag="someone_else")
        result_during_gap = anchor.update([someone_else])
        # 등장한 사람이 이음 반경 밖이라 이번 프레임은 후보가 없다 -> None
        self.assertIsNone(result_during_gap)

        # 사용자가 비슷한 자리로 돌아온다 — 같은 사람으로 다시 이어 붙어야 한다
        user_returns = FakeFace((505, 405), 80, tag="user")
        chosen = anchor.update([user_returns, someone_else])
        self.assertEqual(chosen.tag, "user")

    def test_dropout_beyond_grace_allows_fresh_acquisition(self):
        user = FakeFace((500, 400), 80, tag="user")
        anchor, clock = make_anchor()
        feed(anchor, clock, [[user]])

        # anchor_grace_sec(1.0s)를 넘겨서 완전히 소실
        clock.tick(1.5)
        anchor.update([])   # 유예 만료 처리(내부 _drop_expired_anchor)

        # 이제 새로 나타난, 더 가까운 사람이 처음 선정처럼 새로 잡힌다
        new_person = FakeFace((900, 400), 150, tag="new_person")
        chosen = anchor.update([new_person])
        self.assertEqual(chosen.tag, "new_person")


class ImplausibleJumpTest(unittest.TestCase):
    def test_isolated_face_survives_despite_flagged_jump(self):
        # 후보가 하나뿐이면(연속 반경 안에 그 얼굴 하나) "신뢰 못 할 관측"으로
        # 판정되어도 이번 프레임 출력 자체는 그 얼굴을 돌려준다(완전히 얼어붙지
        # 않는다) — 다만 앵커의 내부 기준 위치는 천천히만 반영된다(부분 반영)
        user = FakeFace((500, 400), 80, tag="user")
        anchor, clock = make_anchor()
        feed(anchor, clock, [[user]])

        # move_reject_ratio=0.6 -> 80*0.6=48px 초과 이동이면 "이상치"로 잡힌다.
        # 60px 이동, 그런데 연속 반경(1.5*80=120px) 안이라 후보로는 남는다
        jumped = FakeFace((560, 400), 80, tag="user")
        chosen = anchor.update([jumped])
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.tag, "user")


if __name__ == "__main__":
    unittest.main()
