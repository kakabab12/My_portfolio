"""화면비 고정 exe 6개를 한 번에 빌드한다 — make_aspect_exes.bat이 이것만 부른다.

2026-09-09 신설(사용자 요청 "각각 exe 파일 16:9랑 9:16 총 6개").

기존 build_forehead_exe_helper.py 등과 같은 방침입니다 — PyInstaller는 시스템
파이썬에 깔지 않고 빌드 전용 임시 venv에만 설치했다가 끝나면 지웁니다. 다른
점은 **여섯 개를 임시 venv 하나에서** 만든다는 것뿐입니다. 트래커별로 venv를
새로 만들면 PyInstaller를 여섯 번 내려받게 되는데, 여기서 만드는 것은 전부
같은 의존(표준 라이브러리뿐)을 쓰는 얇은 런처라 나눌 이유가 없습니다.

진입 스크립트(scripts/exe_entries/*.py)도 이 파일이 만들어 씁니다 — 여섯 개를
손으로 두는 대신, 표에서 찍어 냅니다.
"""
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.console import enable_utf8_output


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_VENV_DIR = os.path.join(ROOT_DIR, ".build_venv_tmp_aspect")
ENTRY_DIR = os.path.join(ROOT_DIR, "scripts", "exe_entries")

TRACKERS = (("head", "코끝"), ("eyebrow", "눈썹"), ("forehead", "이마"))
ASPECTS = (("desktop", "16-9_데스크탑", "16:9 가로 모니터"),
           ("kiosk", "9-16_키오스크", "9:16 세로 키오스크"))

ENTRY_TEMPLATE = '''"""{tracker_ko} 기준점 · {aspect_ko} — {exe_name}.exe 진입점.

build_aspect_exes_helper.py가 만들어 낸 파일입니다. 직접 고치지 마세요 —
빌드할 때마다 다시 만들어집니다. 로직은 scripts/aspect_launcher_common.py에
한 벌만 있습니다.
"""
import sys

from aspect_launcher_common import run

if __name__ == "__main__":
    sys.exit(run("{tracker}", "{aspect}"))
'''


def _write_entries():
    """진입 스크립트 6개를 새로 찍어 낸다. 반환값: [(exe 이름, 진입 파일 경로)]."""
    os.makedirs(ENTRY_DIR, exist_ok=True)
    entries = []
    for tracker, tracker_ko in TRACKERS:
        for aspect, tag, aspect_ko in ASPECTS:
            exe_name = f"{tracker}_{tag}"
            path = os.path.join(ENTRY_DIR, f"{exe_name}.py")
            with open(path, "w", encoding="utf-8", newline="\n") as fp:
                fp.write(ENTRY_TEMPLATE.format(
                    tracker=tracker, tracker_ko=tracker_ko,
                    aspect=aspect, aspect_ko=aspect_ko, exe_name=exe_name))
            entries.append((exe_name, path))
    print(f"[INFO] 진입 스크립트 {len(entries)}개 생성 - {ENTRY_DIR}")
    return entries


def _create_build_venv():
    print("[INFO] 빌드용 임시 venv 생성...")
    subprocess.run([sys.executable, "-m", "venv", BUILD_VENV_DIR],
                   cwd=ROOT_DIR, check=True)
    return os.path.join(BUILD_VENV_DIR, "Scripts", "python.exe")


def _build_all(build_python, entries):
    print("[INFO] PyInstaller 설치(빌드 전용 venv)...")
    subprocess.run([build_python, "-m", "pip", "install", "--quiet", "pyinstaller"],
                   cwd=ROOT_DIR, check=True)

    for index, (exe_name, entry_path) in enumerate(entries, start=1):
        print(f"[INFO] ({index}/{len(entries)}) {exe_name}.exe 빌드 중...")
        cmd = [
            build_python, "-m", "PyInstaller", "--onefile", "--console",
            "--name", exe_name,
            "--distpath", ROOT_DIR,
            "--workpath", os.path.join(BUILD_VENV_DIR, "work", exe_name),
            "--specpath", BUILD_VENV_DIR,
            # aspect_launcher_common을 찾게 해 준다 — 진입 스크립트가 그것만 부른다
            "--paths", os.path.join(ROOT_DIR, "scripts"),
            entry_path,
        ]
        subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def main():
    enable_utf8_output()   # cp949 콘솔에서 줄표(—) 등으로 죽는 것 방지
    entries = _write_entries()
    build_python = _create_build_venv()
    try:
        _build_all(build_python, entries)
    except subprocess.CalledProcessError as exc:
        print(f"[FAIL] 빌드 실패 - {exc}")
        return 1
    finally:
        print("[INFO] 임시 venv 정리...")
        shutil.rmtree(BUILD_VENV_DIR, ignore_errors=True)

    print(f"[DONE] exe {len(entries)}개 생성 완료 - 프로젝트 루트에서 확인하세요")
    for exe_name, _ in entries:
        print(f"       {exe_name}.exe")
    return 0


if __name__ == "__main__":
    sys.exit(main())
