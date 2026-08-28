"""이 빌드가 무엇인지 알려준다 (2026-08-25 신설).

왜 필요한가
-----------
현장에서 "커서가 이상하다"는 연락을 받았을 때, 지금은 **그 기계에 어느 버전이
깔려 있는지 알 방법이 없다**. 로그에도 안 남고 화면에도 안 나온다. 그래서
이미 고친 문제를 다시 조사하거나, 반대로 옛 버전에만 있는 문제를 최신 코드에서
찾느라 시간을 버리게 된다.

시작할 때 한 줄만 남겨두면 그럴 일이 없다. exe로 묶으면 소스 파일이 사라지므로
버전 문자열을 코드 안에 박아 두고, 소스로 돌릴 때는 실제 파일이 마지막으로
고쳐진 날짜를 함께 보여준다(빌드 날짜 대용 — 깃 저장소가 아니라서 커밋
해시를 쓸 수 없다).

버전을 올리는 규칙
------------------
현장에 새로 배포할 때마다 VERSION의 날짜를 그날로 바꾼다. 같은 날 두 번
배포하면 뒤에 -2, -3을 붙인다. 거창한 체계는 필요 없다 — "그 기계에 깔린 게
언제 것인가"만 알면 되는 용도다.
"""
import os
import sys

VERSION = "2026-08-25"
PRODUCT = "제스처 키오스크 헤드트래커"


def _source_stamp():
    """소스로 돌릴 때 — 진입점 파일이 마지막으로 고쳐진 날짜."""
    try:
        main_file = getattr(sys.modules.get("__main__"), "__file__", None)
        if not main_file:
            return None
        import datetime
        mtime = os.path.getmtime(main_file)
        return datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    except Exception:   # noqa: 방어적 — 버전 표시가 실행을 막으면 안 된다
        return None


def _is_frozen():
    """PyInstaller로 묶인 exe로 돌고 있는가."""
    return getattr(sys, "frozen", False)


def describe(entry_name=""):
    """로그·화면에 한 줄로 쓸 설명."""
    parts = ["%s %s" % (PRODUCT, VERSION)]
    if entry_name:
        parts.append("(%s)" % entry_name)
    parts.append("exe 빌드" if _is_frozen() else "소스 실행")
    stamp = _source_stamp()
    if stamp and not _is_frozen():
        parts.append("- 소스 수정 %s" % stamp)
    return " ".join(parts)


def environment():
    """같이 남겨두면 현장 문제를 가릴 때 도움이 되는 것들."""
    rows = [("파이썬", sys.version.split()[0])]
    for label, module_name in (("OpenCV", "cv2"), ("MediaPipe", "mediapipe"),
                               ("NumPy", "numpy")):
        try:
            module = __import__(module_name)
            rows.append((label, getattr(module, "__version__", "?")))
        except Exception:   # noqa: 방어적 — 없으면 그냥 건너뛴다
            pass
    return ", ".join("%s %s" % (label, value) for label, value in rows)
