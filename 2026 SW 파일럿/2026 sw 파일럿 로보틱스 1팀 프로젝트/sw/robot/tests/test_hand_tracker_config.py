"""HandTracker의 CPU/GPU delegate 설정 검증 — MediaPipe·카메라 없이 실행한다."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.hand_tracker import normalize_delegate


class HandTrackerDelegateConfigTest(unittest.TestCase):
    def test_accepts_gpu_case_insensitively(self):
        self.assertEqual(normalize_delegate(" GPU "), "gpu")

    def test_accepts_cpu(self):
        self.assertEqual(normalize_delegate("cpu"), "cpu")

    def test_rejects_unknown_delegate(self):
        with self.assertRaises(ValueError):
            normalize_delegate("tensorrt")


if __name__ == "__main__":
    unittest.main()
