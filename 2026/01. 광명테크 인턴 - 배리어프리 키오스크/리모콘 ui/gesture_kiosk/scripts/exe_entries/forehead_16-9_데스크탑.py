"""이마 기준점 · 16:9 가로 모니터 — forehead_16-9_데스크탑.exe 진입점.

build_aspect_exes_helper.py가 만들어 낸 파일입니다. 직접 고치지 마세요 —
빌드할 때마다 다시 만들어집니다. 로직은 scripts/aspect_launcher_common.py에
한 벌만 있습니다.
"""
import sys

from aspect_launcher_common import run

if __name__ == "__main__":
    sys.exit(run("forehead", "desktop"))
