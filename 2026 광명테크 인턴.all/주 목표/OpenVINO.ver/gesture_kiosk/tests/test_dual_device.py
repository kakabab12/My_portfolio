"""CPU + 보조 디바이스(Intel iGPU) 동시 추론(2026-07-27 신설) 단위 테스트.

실제 openvino·rtmlib 없이 DualDevicePoseEstimator의 스케줄링 로직(2단 파이프라인 —
이번 프레임은 배정만 하고 직전 프레임 결과를 돌려주는 방식)과 build_pose_estimator의
폴백 조건(설정 꺼짐·backend 불일치·보조 디바이스 컴파일 실패)만 검증한다.
개발 PC는 dGPU가 OpenVINO GPU 슬롯을 차지해 실제 iGPU 검증이 불가하므로(설정 파일 주석
참고), 여기서는 스케줄러 자체의 정확성(순서 보존·동시 실행)만 확인하고 실측은 실기에서 한다.
"""
import os
import sys
import threading
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.pose_estimator import DualDevicePoseEstimator, build_pose_estimator


class _SequentialDevice:
    """호출마다 (이름, frame)을 그대로 돌려주는 대역 — 순서 검증용."""

    def __init__(self, name):
        self.name = name
        self.calls = []

    def infer(self, frame):
        self.calls.append(frame)
        return (self.name, frame)


class _BarrierDevice:
    """두 디바이스가 진짜 동시에(barrier에서 만나야) 통과하는 대역 — 동시성 증명용.

    스케줄러가 실수로 완전 순차 실행(한 번에 디바이스 하나만 사용)이 되면 이 barrier가
    파트너를 못 만나 타임아웃(BrokenBarrierError)으로 테스트가 명확히 실패한다.
    """

    def __init__(self, name, barrier):
        self.name = name
        self._barrier = barrier

    def infer(self, frame):
        self._barrier.wait(timeout=2)
        return (self.name, frame)


class DualDevicePoseEstimatorTest(unittest.TestCase):
    def test_cold_start_returns_empty_then_prior_frame_in_order(self):
        primary = _SequentialDevice("primary")
        secondary = _SequentialDevice("secondary")
        estimator = DualDevicePoseEstimator(primary, secondary)

        self.assertEqual(estimator.infer("f0"), [])                    # 콜드 스타트
        self.assertEqual(estimator.infer("f1"), ("primary", "f0"))      # 1프레임 지연 반환
        self.assertEqual(estimator.infer("f2"), ("secondary", "f1"))
        self.assertEqual(estimator.infer("f3"), ("primary", "f2"))
        self.assertEqual(estimator.infer("f4"), ("secondary", "f3"))

        # 짝수 순번=primary, 홀수 순번=secondary로 번갈아 배정됐는지 확인
        self.assertEqual(primary.calls, ["f0", "f2", "f4"])
        self.assertEqual(secondary.calls, ["f1", "f3"])

    def test_devices_actually_overlap_in_time(self):
        # 스케줄러가 매 호출마다 방금 배정한 디바이스를 그대로 기다려버리면(버그) barrier가
        # 파트너를 못 만나 타임아웃한다 — 통과 자체가 "두 디바이스 동시 실행"의 증거다.
        barrier = threading.Barrier(2)
        primary = _BarrierDevice("primary", barrier)
        secondary = _BarrierDevice("secondary", barrier)
        estimator = DualDevicePoseEstimator(primary, secondary)

        estimator.infer(0)               # primary(0) 배정만(콜드 스타트, barrier 대기 시작)
        result = estimator.infer(1)      # secondary(1) 배정 -> primary(0)와 동시에 barrier 통과
        self.assertEqual(result, ("primary", 0))


def _make_dual_config(dual_enabled, backend="openvino"):
    return {
        "model": {
            "device": "cpu",
            "pose_engine": "body",
            "pose_mode": "lightweight",
            "backend": backend,
            "dual_device": {"enabled": dual_enabled, "secondary_device": "GPU"},
        },
        "person_lock": {"kpt_conf_threshold": 0.3},
    }


class _FakeToolForBuild:
    def __init__(self):
        self.backend = "openvino"
        self.onnx_model = "dummy.onnx"


class _FakeSolutionForBuild:
    instances = []

    def __init__(self, mode=None, backend=None, device=None, to_openpose=False):
        self.det_model = _FakeToolForBuild()
        self.pose_model = _FakeToolForBuild()
        _FakeSolutionForBuild.instances.append(self)


class BuildPoseEstimatorFallbackTest(unittest.TestCase):
    """dual_device 폴백 조건 — 꺼짐/backend 불일치/컴파일 실패 시 CPU 단독으로 돌아간다."""

    def setUp(self):
        _FakeSolutionForBuild.instances = []
        fake_rtmlib = types.ModuleType("rtmlib")
        fake_rtmlib.Body = _FakeSolutionForBuild
        fake_rtmlib.Wholebody = _FakeSolutionForBuild
        self._saved_rtmlib = sys.modules.get("rtmlib")
        sys.modules["rtmlib"] = fake_rtmlib

        from src.inference.pose_estimator import PoseEstimator

        self.PoseEstimator = PoseEstimator

    def tearDown(self):
        if self._saved_rtmlib is None:
            sys.modules.pop("rtmlib", None)
        else:
            sys.modules["rtmlib"] = self._saved_rtmlib

    def test_disabled_returns_plain_estimator(self):
        estimator = build_pose_estimator(_make_dual_config(dual_enabled=False))
        self.assertIsInstance(estimator, self.PoseEstimator)

    def test_non_openvino_backend_falls_back(self):
        estimator = build_pose_estimator(
            _make_dual_config(dual_enabled=True, backend="onnxruntime")
        )
        self.assertIsInstance(estimator, self.PoseEstimator)

    def test_recompile_failure_falls_back(self):
        with patch(
            "src.inference.pose_estimator._recompile_openvino_device", return_value=False
        ):
            estimator = build_pose_estimator(_make_dual_config(dual_enabled=True))
        self.assertIsInstance(estimator, self.PoseEstimator)

    def test_recompile_success_returns_dual_estimator(self):
        with patch(
            "src.inference.pose_estimator._recompile_openvino_device", return_value=True
        ):
            estimator = build_pose_estimator(_make_dual_config(dual_enabled=True))
        self.assertIsInstance(estimator, DualDevicePoseEstimator)


if __name__ == "__main__":
    unittest.main()
