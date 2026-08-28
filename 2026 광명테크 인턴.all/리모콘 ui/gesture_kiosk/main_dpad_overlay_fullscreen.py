"""main_dpad.py의 "--overlay --fullscreen" 프리셋 실행기 (2026-08-05 신설,
사용자 요청 — 매번 두 플래그를 직접 타이핑하지 않고 바로 그 조합으로 뜨게).

실제 판정·렌더링 로직은 전부 main_dpad.py에 있다 — 이 파일은 main_dpad를
그대로 임포트해 sys.argv 앞쪽에 두 플래그만 끼워 넣고 main_dpad.main()을
호출할 뿐, 코드를 복제하지 않는다(두 파일이 따로 놀며 어긋나는 것 방지).
추가 인자를 붙이면(예: --device 1, --overlay-alpha 180) 그대로 함께
전달된다 — --blank-bg처럼 --overlay와 상호 배타인 인자를 더하면 main_dpad.py
자체의 argparse 검증이 그대로 걸러낸다(정상 동작).

사용법 (프로젝트 루트에서):
    py main_dpad_overlay_fullscreen.py [추가 인자...]
종료: q 또는 ESC (그 밖의 모든 동작은 main_dpad.py --overlay --fullscreen과 완전히 동일)
"""
import sys

import main_dpad

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--overlay", "--fullscreen", *sys.argv[1:]]
    sys.exit(main_dpad.main())
