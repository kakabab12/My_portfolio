"""run.exe 런처 원본 — PyInstaller로 이 스크립트만 빌드해 얇은 실행기를 만든다 (2026-08-07 신설).

배경: 델파이7이 CreateProcess로 엔진을 자식 프로세스로 띄워 stdout을 파이프로
읽는다(docs/델파이7_연동가이드.md). .bat 파일은 cmd.exe가 중간에 껴 인코딩·
goto 버그(install.bat 헤더 주석 참고)에 노출되고, CreateProcess로 .bat을
직접 실행하는 것도 까다롭다 — 델파이 쪽에서 바로 띄울 진짜 실행 파일이 필요하다.

이 exe 자체는 mediapipe·numpy 등 무거운 의존을 담지 않는다(가벼운 런처만 빌드
— venv 없이 시스템 파이썬 공유 방향 유지, install.bat 참고: "가상환경 제거,
시스템 파이썬에 직접 설치"). 시스템 파이썬을 찾아 실제 엔진(main.py — 프로젝트
루트의 진입점)을 자식으로 그대로 실행하고 표준입출력을 그대로 물려준다 —
그래서 델파이까지 stdout이 손실 없이 전달된다(event_sender.py의 "stdout은
이벤트 전용" 규칙 유지 — 이 파일도 stdout에는 아무것도 찍지 않는다, 진단
메시지는 stderr로만).

빌드 방법 (make_exe.bat 참고 — 임시 venv에서 pyinstaller 설치 후 빌드,
시스템 파이썬은 건드리지 않는다):
    make_exe.bat
"""
import ctypes
import os
import subprocess
import sys

PYTHON_CANDIDATES = (("py", "-3.11"), ("python",))
REQUIRED_VERSION_TAG = "3.11"


def _set_console_utf8():
    """콘솔 코드페이지를 UTF-8로 — .bat의 chcp 65001과 동일 효과.

    run.exe는 .bat을 거치지 않아 콘솔 코드페이지가 시스템 기본(cp949)로 남는다
    — 자식 파이썬의 한글 로그(stderr)가 깨져 보인다. 델파이 파서가 읽는 stdout
    프로토콜은 ASCII 고정이라 무관하지만, 진단용 한글 로그 가독성을 위해 맞춰준다.
    """
    try:
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except (AttributeError, OSError):
        pass   # 콘솔이 없는 환경(파이프 전용 등) — 무시해도 무방


def _get_root_dir():
    """run.exe 자신이 있는 폴더 — 프로젝트 루트(main.py와 같은 위치를 전제)."""
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

    main_py_path = os.path.join(root_dir, "main.py")
    # stdout·stderr를 그대로 물려준다(리다이렉트하지 않음) — 델파이가 이 exe의
    # stdout을 파이프로 읽는 구조라(docs/델파이7_연동가이드.md) 가로채면 안 된다
    result = subprocess.run([*python_cmd, main_py_path, *sys.argv[1:]], cwd=root_dir)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
