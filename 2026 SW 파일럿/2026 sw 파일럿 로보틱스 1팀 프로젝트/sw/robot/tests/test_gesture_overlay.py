import unittest

from src.pipeline.gesture_loop import _clamp_cursor_to_frame


class ClampCursorToFrameTest(unittest.TestCase):
    def test_keeps_cursor_circle_inside_all_edges(self):
        shape = (480, 640, 3)
        center = (320, 240)
        self.assertEqual(_clamp_cursor_to_frame(center, (-1000, -1000), shape), (9, 9))
        self.assertEqual(_clamp_cursor_to_frame(center, (1000, 1000), shape), (630, 470))

    def test_preserves_cursor_when_already_inside(self):
        self.assertEqual(
            _clamp_cursor_to_frame((320, 240), (20.5, -30.5), (480, 640, 3)),
            (340, 209),
        )


if __name__ == "__main__":
    unittest.main()
