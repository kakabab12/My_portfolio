"""화면비 고정 exe 6개가 공유하는 런처 본체 (2026-09-09 신설).

사용자 요청 "각각 exe 파일 16:9랑 9:16 총 6개". 기준점 3가지(코끝·눈썹·이마)
× 화면비 2가지(16:9 데스크탑 · 9:16 키오스크)입니다.

기존 head.exe / eyebrow.exe / forehead.exe와 하는 일은 같습니다 — 시스템
파이썬을 찾아 프로젝트 루트의 엔진(head.py 등)을 자식으로 실행하고 표준
입출력을 그대로 물려줍니다(scripts/forehead_launcher.py 독스트링 참고).
다른 점은 `--aspect`를 붙여 준다는 것 하나뿐입니다. 그래서 예를 들어
head_9-16_키오스크.exe 는 `head.py --aspect kiosk`와 같습니다.

여섯 개가 같은 코드를 쓰도록 로직은 여기 한 벌만 둡니다 — 개별 진입 파일
(scripts/exe_entries/*.py)은 트래커 이름과 화면비만 정해서 run()을 부릅니다.
엔진 파일 자체는 늘리지 않았습니다(그 이유는 launchers/README.md 참고).

빌드: make_aspect_exes.bat
"""
import ctypes
import os
import subprocess
import sys

PYTHON_CANDIDATES = (("py", "-3.11"), ("python",))
REQUIRED_VERSION_TAG = "3.11"


def _set_console_utf8():
    """콘솔 코드페이지를 UTF-8로 — 한글 로그가 깨지지 않게."""
    try:
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except (AttributeError, OSError):
        pass   # 콘솔이 없는 환경(파이프 전용 등) — 무시해도 무방


def _get_root_dir():
    """exe 자신이 있는 폴더 — 엔진(head.py 등)과 같은 위치를 전제한다."""
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


def run(tracker, aspect):
    """tracker: "head"/"eyebrow"/"forehead", aspect: "desktop"/"kiosk"."""
    _set_console_utf8()
    root_dir = _get_root_dir()
    python_cmd = _get_python_cmd()
    if python_cmd is None:
        print("[FAIL] Python 3.11 not found - install.bat을 먼저 실행하세요",
              file=sys.stderr)
        return 1

    entry_py_path = os.path.join(root_dir, tracker + ".py")
    if not os.path.isfile(entry_py_path):
        # exe만 따로 옮겨 놓은 경우 — 엔진이 없으면 무엇이 없는지 알려 준다
        print(f"[FAIL] {tracker}.py를 찾을 수 없습니다 - exe를 프로젝트 폴더에 두세요"
              f" (찾아본 곳: {root_dir})", file=sys.stderr)
        return 1

    # 사용자가 직접 --aspect를 주면 그쪽을 존중한다(뒤에 온 값이 이긴다).
    # 이 exe의 화면비는 기본값일 뿐 잠금이 아니다.
    cmd = [*python_cmd, entry_py_path, "--aspect", aspect, *sys.argv[1:]]
    result = subprocess.run(cmd, cwd=root_dir)
    return result.returncode
