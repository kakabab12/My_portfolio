"""forehead.exe 빌드 로직 — make_forehead_exe.bat이 이 스크립트 하나만 호출한다
(2026-08-27 신설, scripts/build_eyebrow_exe_helper.py와 동일 패턴 — 그 파일
독스트링 참고).

PyInstaller는 시스템 파이썬에 깔지 않는다 — 빌드 전용 임시 venv에만 설치하고
끝나면 지운다. 결과물(forehead.exe)만 프로젝트 루트에 남는다. 다른 exe 빌드와
별도 임시 venv 디렉터리를 쓴다 — 여러 빌드가 겹쳐 돌아도 서로 밟지 않게.
"""
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import shutil
import subprocess
import sys
from src.utils.console import enable_utf8_output


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_VENV_DIR = os.path.join(ROOT_DIR, ".build_venv_tmp_forehead")


def _create_build_venv():
    print("[INFO] 빌드용 임시 venv 생성...")
    subprocess.run([sys.executable, "-m", "venv", BUILD_VENV_DIR], cwd=ROOT_DIR, check=True)
    return os.path.join(BUILD_VENV_DIR, "Scripts", "python.exe")


def _build_exe(build_python):
    print("[INFO] PyInstaller 설치(빌드 전용 venv)...")
    subprocess.run([build_python, "-m", "pip", "install", "--quiet", "pyinstaller"],
                    cwd=ROOT_DIR, check=True)

    print("[INFO] forehead.exe 빌드 중...")
    cmd = [
        build_python, "-m", "PyInstaller", "--onefile", "--console", "--name", "forehead",
        "--distpath", ROOT_DIR,
        "--workpath", os.path.join(BUILD_VENV_DIR, "work"),
        "--specpath", BUILD_VENV_DIR,
        os.path.join("scripts", "forehead_launcher.py"),
    ]
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def main():
    enable_utf8_output()   # cp949 콘솔에서 줄표(—) 등으로 죽는 것 방지
    build_python = _create_build_venv()
    try:
        _build_exe(build_python)
    except subprocess.CalledProcessError as exc:
        print(f"[FAIL] 빌드 실패 — {exc}")
        return 1
    finally:
        print("[INFO] 임시 venv 정리...")
        shutil.rmtree(BUILD_VENV_DIR, ignore_errors=True)

    print("[DONE] forehead.exe 생성 완료 - 프로젝트 루트에서 확인하세요")
    return 0


if __name__ == "__main__":
    sys.exit(main())
