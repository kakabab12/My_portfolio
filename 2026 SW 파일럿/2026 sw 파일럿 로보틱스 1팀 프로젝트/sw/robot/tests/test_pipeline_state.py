"""다중 카메라 웹 진단 상태의 단위 테스트."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.gesture_loop import PipelineState


class PipelineCameraFramesTest(unittest.TestCase):
    def test_stores_camera_frames_independently(self):
        state = PipelineState()
        first = np.zeros((2, 3, 3), dtype=np.uint8)
        third = np.full((2, 3, 3), 7, dtype=np.uint8)

        state.update_camera_frame(0, first, 11)
        state.update_camera_frame(2, third, 23)

        np.testing.assert_array_equal(state.get_camera_frame(0), first)
        self.assertIsNone(state.get_camera_frame(1))
        np.testing.assert_array_equal(state.get_camera_frame(2), third)
        self.assertEqual(
            state.camera_status([0, 1, 2]),
            [
                {"index": 0, "device_id": 0, "frame_received": True, "sequence": 11},
                {"index": 1, "device_id": 1, "frame_received": False, "sequence": 0},
                {"index": 2, "device_id": 2, "frame_received": True, "sequence": 23},
            ],
        )

    def test_get_returns_a_copy(self):
        state = PipelineState()
        frame = np.zeros((1, 1, 3), dtype=np.uint8)
        state.update_camera_frame(0, frame, 1)
        returned = state.get_camera_frame(0)
        returned[:] = 255
        np.testing.assert_array_equal(state.get_camera_frame(0), frame)

    def test_inference_status_reports_gpu_only_when_all_trackers_use_it(self):
        state = PipelineState()

        class Tracker:
            def __init__(self, active_delegate):
                self._active_delegate = active_delegate

            def inference_status(self):
                return {
                    "requested_delegate": "gpu",
                    "active_delegate": self._active_delegate,
                    "gpu_accelerated": self._active_delegate == "gpu",
                    "fallback_reason": None,
                }

        state.set_inference_status([Tracker("gpu"), Tracker("gpu")])
        self.assertTrue(state.inference_status()["gpu_accelerated"])

        state.set_inference_status([Tracker("gpu"), Tracker("cpu")])
        self.assertFalse(state.inference_status()["gpu_accelerated"])


if __name__ == "__main__":
    unittest.main()
