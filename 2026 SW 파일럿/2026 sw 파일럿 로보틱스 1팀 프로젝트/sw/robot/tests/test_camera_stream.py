"""camera_stream 단위 테스트 — device_id가 URL(문자열)일 때 네트워크 스트림으로
취급해 로컬 전용 설정(백엔드 지정·해상도·포맷 요청)을 건너뛰는지 검증한다.
실제 카메라·네트워크 없이 cv2.VideoCapture를 모킹해서 확인한다.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.capture.camera_stream import init_camera


def _fake_cap(is_opened=True):
    cap = MagicMock()
    cap.isOpened.return_value = is_opened
    cap.get.return_value = 0
    cap.getBackendName.return_value = "FFMPEG"
    return cap


class InitCameraUrlTest(unittest.TestCase):
    def test_string_device_id_opens_as_plain_url(self):
        fake_cap = _fake_cap()
        url = "http://192.168.0.5:8090/cam/0/stream"
        with patch("src.capture.camera_stream.cv2.VideoCapture", return_value=fake_cap) as ctor:
            cap = init_camera({"camera": {}}, url)   # width_px 등 없어도 되어야 한다
        ctor.assert_called_once_with(url)   # 백엔드 인자 없이 URL 하나만
        self.assertIs(cap, fake_cap)

    def test_string_device_id_skips_local_only_settings(self):
        fake_cap = _fake_cap()
        with patch("src.capture.camera_stream.cv2.VideoCapture", return_value=fake_cap):
            init_camera({"camera": {}}, "http://192.168.0.5:8090/cam/0/stream")
        # fourcc·해상도는 로컬 장치 하드웨어 설정이라 네트워크 스트림엔 의미가 없다
        fake_cap.set.assert_not_called()

    def test_string_device_id_raises_if_not_opened(self):
        fake_cap = _fake_cap(is_opened=False)
        with patch("src.capture.camera_stream.cv2.VideoCapture", return_value=fake_cap):
            with self.assertRaises(RuntimeError):
                init_camera({"camera": {}}, "http://bad-host/stream")

    def test_int_device_id_still_configures_local_capture(self):
        fake_cap = _fake_cap()
        config = {"camera": {"width_px": 1280, "height_px": 720, "fourcc": "mjpg"}}
        with patch("src.capture.camera_stream.cv2.VideoCapture", return_value=fake_cap):
            init_camera(config, 0)
        # 로컬 장치는 fourcc·폭·높이를 요청해야 한다(최소 3회 set)
        self.assertGreaterEqual(fake_cap.set.call_count, 3)


if __name__ == "__main__":
    unittest.main()
