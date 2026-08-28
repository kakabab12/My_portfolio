"""카메라 자동 협상 단위 테스트 — 2026-08-27 신설.

실제 카메라·자식 프로세스 없이 로직만 검증한다:
  · init_camera가 기본 설정으로 열리면 협상을 아예 시도하지 않는가(비용 0)
  · 실패하면 config_path가 있을 때만 협상하고, 없으면 예전처럼 바로 예외인가
  · 협상이 찾은 조합으로 config가 실제로 갱신되는가
  · camera_negotiate의 자식 프로세스 결과 파싱이 정확한가

★가장 중요한 검사는 "config_path를 안 넘기면 예전과 완전히 동일한가"다 —
main.py 등 이 값을 아직 안 넘기는 기존 호출부가 조용히 바뀌면 안 된다.

실행 (프로젝트 루트에서):
    python -m unittest discover tests -v
"""
import os
import subprocess
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.capture.camera_negotiate as negotiate_module   # noqa: E402
import src.capture.camera_stream as camera_stream_module  # noqa: E402


class _FakeCap:
    def __init__(self, opened):
        self._opened = opened
        self.released = False

    def isOpened(self):
        return self._opened

    def get(self, _prop):
        return 0

    def set(self, _prop, _value):
        pass

    def getBackendName(self):
        return "FAKE"

    def release(self):
        self.released = True


class InitCameraFallbackTest(unittest.TestCase):
    """init_camera의 협상 진입 조건 — camera_stream.py에 넣은 로직."""

    def _config(self):
        return {"camera": {"device_id": 0, "windows_backend": "msmf",
                           "fourcc": "mjpg", "width_px": 1280, "height_px": 720}}

    def test_success_path_never_touches_negotiate_module(self):
        """★가장 중요 — 기본 설정으로 잘 열리면 협상 모듈을 아예 안 건드린다.

        절대다수인 이 경우에 비용이 조금이라도 늘면 안 된다. import까지도
        일어나면 안 되므로, camera_negotiate에 아무 패치도 안 걸어 둔 채
        find_working_combo가 불렸다면 바로 AttributeError로 드러나게 한다.
        """
        with mock.patch.object(camera_stream_module, "_open_raw",
                               return_value=_FakeCap(opened=True)):
            cap = camera_stream_module.init_camera(
                self._config(), config_path="아무_경로.yaml")
        self.assertTrue(cap.isOpened())

    def test_no_config_path_raises_immediately_like_before(self):
        """config_path를 안 넘기면(기존 호출부) 예전과 동일하게 즉시 예외."""
        with mock.patch.object(camera_stream_module, "_open_raw",
                               return_value=_FakeCap(opened=False)):
            with self.assertRaises(RuntimeError):
                camera_stream_module.init_camera(self._config(), config_path=None)

    def test_config_path_given_triggers_negotiation_on_failure(self):
        """열기 실패 + config_path 있음 -> 협상 호출, 찾은 값으로 config 갱신."""
        config = self._config()
        winner = {"windows_backend": "dshow", "fourcc": "auto",
                  "width_px": 640, "height_px": 480}
        opens = [_FakeCap(opened=False), _FakeCap(opened=True)]   # 1차 실패, 2차(협상 후) 성공

        with mock.patch.object(camera_stream_module, "_open_raw",
                               side_effect=opens) as mock_open, \
             mock.patch.object(negotiate_module, "find_working_combo",
                               return_value=winner) as mock_find:
            cap = camera_stream_module.init_camera(
                config, config_path="가상_config.yaml")

        mock_find.assert_called_once_with("가상_config.yaml", 0)
        self.assertEqual(mock_open.call_count, 2)   # 1차 시도 + 협상 후 재시도, 딱 두 번
        self.assertTrue(cap.isOpened())
        # config가 협상된 값으로 실제로 바뀌었는가
        self.assertEqual(config["camera"]["windows_backend"], "dshow")
        self.assertEqual(config["camera"]["width_px"], 640)

    def test_negotiation_exhausted_raises_clear_error(self):
        """모든 조합이 실패하면(find_working_combo가 None) 명확히 예외를 낸다."""
        with mock.patch.object(camera_stream_module, "_open_raw",
                               return_value=_FakeCap(opened=False)), \
             mock.patch.object(negotiate_module, "find_working_combo",
                               return_value=None):
            with self.assertRaises(RuntimeError):
                camera_stream_module.init_camera(
                    self._config(), config_path="가상_config.yaml")


class FindWorkingComboTest(unittest.TestCase):
    """camera_negotiate.find_working_combo — 순서대로 시도해 첫 성공을 쓰는가."""

    def test_returns_first_successful_combo_and_stops(self):
        """세 번째 조합에서 성공하면 그 뒤 조합은 아예 시도하지 않는다."""
        call_log = []

        def fake_try(config_path, device_id, backend, fourcc, width, height):
            call_log.append(backend)
            # COMBOS의 세 번째(msmf/auto/640x480)에서만 성공
            if (backend, fourcc, width, height) == negotiate_module.COMBOS[2]:
                return {"opened": True, "frames": 5}
            return {"opened": False, "frames": 0}

        with mock.patch.object(negotiate_module, "_try_combo_isolated",
                               side_effect=fake_try):
            result = negotiate_module.find_working_combo("cfg.yaml", 0)

        backend, fourcc, width, height = negotiate_module.COMBOS[2]
        self.assertEqual(result, {"windows_backend": backend, "fourcc": fourcc,
                                  "width_px": width, "height_px": height})
        self.assertEqual(len(call_log), 3, "세 번째에서 성공했는데 더 시도했다")

    def test_all_combos_fail_returns_none(self):
        with mock.patch.object(negotiate_module, "_try_combo_isolated",
                               return_value={"opened": False, "frames": 0}):
            self.assertIsNone(negotiate_module.find_working_combo("cfg.yaml", 0))

    def test_opened_but_no_frames_does_not_count_as_success(self):
        """열리기만 하고 화면이 안 들어오면(실기에서 흔한 케이스) 실패로 본다."""
        with mock.patch.object(negotiate_module, "_try_combo_isolated",
                               return_value={"opened": True, "frames": 0}):
            self.assertIsNone(negotiate_module.find_working_combo("cfg.yaml", 0))


class TryComboIsolatedParsingTest(unittest.TestCase):
    """_try_combo_isolated — 자식 프로세스 출력 파싱만 검증(진짜 프로세스는 안 띄운다)."""

    def _run_with_stdout(self, stdout_text):
        fake_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=stdout_text, stderr="")
        with mock.patch("os.path.exists", return_value=True), \
             mock.patch("subprocess.run", return_value=fake_result):
            return negotiate_module._try_combo_isolated(
                "cfg.yaml", 0, "msmf", "mjpg", 1280, 720)

    def test_parses_result_line_among_other_output(self):
        """다른 로그 줄에 섞여 있어도 __RESULT__ 줄만 골라 파싱한다."""
        stdout = ("[INFO] 뭔가 로딩 중\n"
                 '__RESULT__{"opened": true, "frames": 12, "width": 1280}\n'
                 "[INFO] 종료\n")
        result = self._run_with_stdout(stdout)
        self.assertEqual(result, {"opened": True, "frames": 12, "width": 1280})

    def test_no_result_line_returns_none(self):
        result = self._run_with_stdout("아무 결과도 안 찍혔다\n")
        self.assertIsNone(result)

    def test_malformed_json_returns_none_not_raises(self):
        result = self._run_with_stdout("__RESULT__{이상한json\n")
        self.assertIsNone(result)

    def test_timeout_returns_none(self):
        with mock.patch("os.path.exists", return_value=True), \
             mock.patch("subprocess.run",
                        side_effect=subprocess.TimeoutExpired(cmd="x", timeout=1)):
            result = negotiate_module._try_combo_isolated(
                "cfg.yaml", 0, "msmf", "mjpg", 1280, 720)
        self.assertIsNone(result)

    def test_missing_camera_check_script_returns_none_without_subprocess(self):
        """camera_check.py 자체가 없으면(빌드 누락 등) subprocess조차 안 띄운다."""
        with mock.patch("os.path.exists", return_value=False), \
             mock.patch("subprocess.run") as mock_run:
            result = negotiate_module._try_combo_isolated(
                "cfg.yaml", 0, "msmf", "mjpg", 1280, 720)
        self.assertIsNone(result)
        mock_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
