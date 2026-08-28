"""head.exe 런처 원본 — PyInstaller로 이 스크립트만 빌드해 얇은 실행기를 만든다
(2026-08-13 신설, scripts/dpad_overlay_launcher.py와 동일 패턴 — 그 파일
독스트링 참고).

이 exe 자체는 mediapipe·opencv 등 무거운 의존을 담지 않는다(가벼운 런처만
빌드 — venv 없이 시스템 파이썬 공유 방향 유지). 시스템 파이썬을 찾아 실제
헤드트래커 엔진(head.py, 프로젝트 루트)을 자식으로 그대로 실행하고
표준입출력을 그대로 물려준다.

빌드 방법 (make_head_exe.bat 참고 — 임시 venv에서 pyinstaller 설치 후 빌드,
시스템 파이썬은 건드리지 않는다):
    make_head_exe.bat
"""
import ctypes
import os
import subprocess
import sys

PYTHON_CANDIDATES = (("py", "-3.11"), ("python",))
REQUIRED_VERSION_TAG = "3.11"


def _set_console_utf8():
    """콘솔 코드페이지를 UTF-8로 — dpad_overlay_launcher.py와 동일 이유(한글 로그 가독성)."""
    try:
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except (AttributeError, OSError):
        pass   # 콘솔이 없는 환경(파이프 전용 등) — 무시해도 무방


def _get_root_dir():
    """head.exe 자신이 있는 폴더 — 프로젝트 루트(head.py와 같은 위치를 전제)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_python_cmd():
    for candidate in PYTHON_CANDIDATES:
        try:
            result = subprocess.run(
                [*candidate, "--version"], capture_output=True, text=True, timeout=5,
            )
        except (FileNotFoundError, OSError):
            continue
        version_text = result.stdout + result.stderr
        if result.returncode == 0 and REQUIRED_VERSION_TAG in version_text:
            return list(candidate)
    return None


def main():
    _set_console_utf8()
    root_dir = _get_root_dir()
    python_cmd = _get_python_cmd()
    if python_cmd is None:
        print("[FAIL] Python 3.11 not found - install.bat을 먼저 실행하세요", file=sys.stderr)
        return 1

    entry_py_path = os.path.join(root_dir, "head.py")
    result = subprocess.run([*python_cmd, entry_py_path, *sys.argv[1:]], cwd=root_dir)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
