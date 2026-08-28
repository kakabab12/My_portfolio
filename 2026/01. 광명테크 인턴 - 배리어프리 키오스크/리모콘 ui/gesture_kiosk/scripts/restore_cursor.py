"""마우스 화살표를 강제로 되돌린다 — 비상용 한 줄 도구.

head.py / eyebrow.py는 실행 중 네이티브 화살표를 숨기고, 꺼질 때 되돌린다.
어떤 경로로 꺼지든 되돌아오도록 안전장치가 걸려 있지만(그 파일들의
_SystemCursorHider.install_exit_guards 참고), `taskkill /F`처럼 OS가 프로세스를
즉시 없애는 경우만은 어떤 프로그램도 정리 코드를 돌릴 수 없다.

그럴 때 이 파일을 실행하면 곧바로 화살표가 돌아온다(트래커를 다시 실행해도
같은 복구가 자동으로 일어난다):

    py -3.11 scripts/restore_cursor.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ctypes
from src.utils.console import enable_utf8_output

SPI_SETCURSORS = 0x0057

if __name__ == "__main__":
    enable_utf8_output()   # cp949 콘솔에서 줄표(—) 등으로 죽는 것 방지
    # 레지스트리에 등록된 기본 커서들을 다시 불러온다 — SetSystemCursor로
    # 바꿔치기한 것 전부가 한 번에 원복되는 윈도우 공식 동작
    ok = ctypes.windll.user32.SystemParametersInfoW(SPI_SETCURSORS, 0, None, 0)
    print("마우스 화살표를 되돌렸습니다." if ok else "복구 실패 — 로그아웃 후 다시 로그인하면 확실히 돌아옵니다.")
