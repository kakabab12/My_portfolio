# -*- coding: utf-8 -*-
"""실행 스크립트마다 --help가 파이프·cp949에서도 죽지 않는가 (2026-09-25 신설).

[왜 이 테스트가 있나]
cp949 줄표(—) 버그는 2026-08-25부터 네 번 되풀이됐고, 8/28에 공용 함수와
"print 안의 글자" 검사(test_console_encoding.py)로 막았다. 그런데 **도움말은
print가 아니라 argparse가 찍는다.** 9/25 exe를 다시 빌드해 `dpad.exe --help`를
돌려 보니 main_dpad.py가 UnicodeEncodeError로 죽었고, 찾아보니 넷이 같았다 —
main_dpad.py, scripts/camera_check.py, dpad_ui_demo.py, head_check.py.
넷 다 보호 장치가 없거나 parse_args보다 **뒤에** 있었다.

콘솔 창에서 직접 치면 안 죽는다(파이썬이 콘솔에는 유니코드로 쓴다). 출력이
**파이프**로 나갈 때 죽는다 — 델파이가 엔진을 부르는 방식이 바로 파이프다.

그래서 여기서는 글자를 세지 않고 **직접 돌려 본다.** 옵션을 받는 스크립트를
전부 찾아 `--help`를 파이프로, 출력 인코딩을 cp949로 못박아 실행하고 종료
코드 0과 usage 줄을 확인한다. 도움말에서 argparse가 바로 끝내므로 카메라·창은
열리지 않는다.
"""
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMEOUT_SEC = 180


def _entry_scripts():
    """옵션을 받는 실행 스크립트 — 루트와 scripts/의 .py 중 ArgumentParser와 parse_args가 둘 다 있는 것."""
    found = []
    for folder in (ROOT_DIR, os.path.join(ROOT_DIR, "scripts")):
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".py"):
                continue
            path = os.path.join(folder, name)
            src = open(path, encoding="utf-8").read()
            if "ArgumentParser(" in src and "parse_args(" in src:
                found.append(path)
    return found


def _run_help(path):
    """cp949·파이프로 --help — (종료 코드, 표준출력, 표준에러)."""
    env = dict(os.environ, PYTHONIOENCODING="cp949")    # 한국어 윈도우 기본과 같게, 오류 처리 없이
    env.pop("PYTHONUTF8", None)
    proc = subprocess.run([sys.executable, path, "--help"], cwd=ROOT_DIR, env=env,
                          stdin=subprocess.DEVNULL, capture_output=True, timeout=TIMEOUT_SEC)
    return proc.returncode, proc.stdout, proc.stderr


class EntryHelpTest(unittest.TestCase):
    def test_every_entry_prints_help_through_a_cp949_pipe(self):
        scripts = _entry_scripts()
        self.assertGreaterEqual(len(scripts), 10, "검사할 스크립트를 못 찾았다 — 테스트가 헛돌고 있다")
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(_run_help, scripts))
        broken = []
        for path, (rc, out, err) in zip(scripts, results):
            if rc != 0 or b"usage:" not in out:
                last = err.decode("utf-8", "replace").strip().splitlines()[-1:] or ["(출력 없음)"]
                broken.append("%s (종료 %s: %s)" % (os.path.relpath(path, ROOT_DIR), rc, last[0][:100]))
        self.assertEqual(broken, [],
                         "--help가 파이프·cp949에서 죽는다. main() 맨 앞, ArgumentParser보다 먼저 "
                         "console.enable_utf8_output()을 부를 것 -> " + "; ".join(broken))

    def test_checker_catches_the_bug(self):
        """검사가 헛돌지 않는지 — 보호 없이 줄표 도움말을 찍는 스크립트는 걸려야 한다."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad_entry.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write("import argparse\n"
                        "p = argparse.ArgumentParser(description='줄표 — 가 든 도움말')\n"
                        "p.parse_args()\n")
            rc, out, _err = _run_help(path)
        self.assertNotEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
