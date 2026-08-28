"""run.exe 빌드 로직 — make_exe.bat이 이 스크립트 하나만 호출한다 (2026-08-03 신설).

install_helper.py와 같은 이유: 이 PC의 cmd.exe가 한글이 섞인 배치 파일을
처리하다 위치 계산이 틀어지는 버그가 있어(길이·한글 밀도가 높을수록 잘 재현 —
goto/label 없이도 재현됨, install_helper.py 독스트링의 사례보다 넓은 증상)
실질 로직은 전부 파이썬으로 옮긴다.

PyInstaller는 시스템 파이썬에 깔지 않는다(경량화 방향 유지) — 빌드 전용 임시
venv에만 설치하고 끝나면 지운다. 결과물(run.exe)만 프로젝트 루트에 남는다.
"""
import os
import shutil
import subprocess
import sys



ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_VENV_DIR = os.path.join(ROOT_DIR, ".build_venv_tmp")


def _create_build_venv():
    print("[INFO] 빌드용 임시 venv 생성...")
    subprocess.run([sys.executable, "-m", "venv", BUILD_VENV_DIR], cwd=ROOT_DIR, check=True)
    return os.path.join(BUILD_VENV_DIR, "Scripts", "python.exe")


def _build_exe(build_python):
    print("[INFO] PyInstaller 설치(빌드 전용 venv)...")
    subprocess.run([build_python, "-m", "pip", "install", "--quiet", "pyinstaller"],
                    cwd=ROOT_DIR, check=True)

    print("[INFO] run.exe 빌드 중...")
    cmd = [
        build_python, "-m", "PyInstaller", "--onefile", "--console", "--name", "run",
        "--distpath", ROOT_DIR,
        "--workpath", os.path.join(BUILD_VENV_DIR, "work"),
        "--specpath", BUILD_VENV_DIR,
        os.path.join("scripts", "run_launcher.py"),
    ]
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def main():
    build_python = _create_build_venv()
    try:
        _build_exe(build_python)
    except subprocess.CalledProcessError as exc:
        print(f"[FAIL] 빌드 실패 — {exc}")
        return 1
    finally:
        print("[INFO] 임시 venv 정리...")
        shutil.rmtree(BUILD_VENV_DIR, ignore_errors=True)

    print("[DONE] run.exe 생성 완료 — 프로젝트 루트에서 확인하세요")
    return 0


if __name__ == "__main__":
    sys.exit(main())
