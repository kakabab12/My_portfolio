"""콘솔 인코딩 방어 테스트 — 2026-08-28 신설.

[왜 이 테스트가 있나]
한국어 윈도우 기본 인코딩(cp949)에는 줄표(—, U+2014)를 담을 자리가 없다.
그래서 그 글자가 든 print 한 줄이 프로그램을 통째로 죽인다. 이 프로젝트는
**같은 버그에 네 번 걸렸다.**

    2026-08-25  트래커 본체에서 발견 -> 그 파일만 고침
    2026-08-27  빌드 스크립트 4개에서 재발 -> 그 파일들만 고침
    2026-08-27  변환 스크립트에서 또 재발
    2026-08-28  새로 만든 measure_head_pose.py 에서 또 재발

매번 그 파일에만 인라인으로 고쳤기 때문에 새 파일을 만들 때마다 되살아났다.
공용 함수(console.enable_utf8_output)로 옮기고, 그 함수가 제 일을 하는지와
**새 스크립트가 그걸 부르고 있는지**를 여기서 검사한다.

실행 (프로젝트 루트에서):
    python -m unittest discover tests -v
"""
import io
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.console import enable_utf8_output   # noqa: E402

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 문제를 일으키는 글자들 — 이 프로젝트 문서·안내문에 실제로 자주 쓴다
CP949_UNSAFE = "—★⚠€"


class _FakeStream:
    """reconfigure를 흉내 내는 가짜 스트림."""

    def __init__(self, isatty):
        self._isatty = isatty
        self.calls = []

    def isatty(self):
        return self._isatty

    def reconfigure(self, **kwargs):
        self.calls.append(kwargs)


class _NoReconfigureStream:
    """reconfigure가 없는 스트림 (오래된 파이썬·특수 환경)."""

    def isatty(self):
        return True


class EnableUtf8OutputTest(unittest.TestCase):
    def setUp(self):
        self._saved = (sys.stdout, sys.stderr)

    def tearDown(self):
        sys.stdout, sys.stderr = self._saved

    def test_tty_only_sets_errors_replace(self):
        """진짜 콘솔이면 인코딩은 건드리지 않고 대체 문자만 켠다."""
        out, err = _FakeStream(isatty=True), _FakeStream(isatty=True)
        sys.stdout, sys.stderr = out, err
        enable_utf8_output()
        self.assertEqual(out.calls, [{"errors": "replace"}])
        self.assertEqual(err.calls, [{"errors": "replace"}])

    def test_pipe_forces_utf8(self):
        """파이프·파일로 나갈 때는 UTF-8로 고정한다."""
        out = _FakeStream(isatty=False)
        sys.stdout, sys.stderr = out, _FakeStream(isatty=False)
        enable_utf8_output()
        self.assertEqual(out.calls, [{"encoding": "utf-8", "errors": "replace"}])

    def test_missing_reconfigure_does_not_raise(self):
        """reconfigure가 없는 환경에서도 조용히 넘어가야 한다."""
        sys.stdout, sys.stderr = _NoReconfigureStream(), _NoReconfigureStream()
        enable_utf8_output()   # 예외가 안 나면 통과

    def test_none_stream_does_not_raise(self):
        """pythonw 처럼 표준 출력이 아예 없어도 죽으면 안 된다."""
        sys.stdout, sys.stderr = None, None
        enable_utf8_output()

    def test_actually_survives_cp949_unsafe_chars(self):
        """★핵심 — 대체 문자 설정이 실제로 cp949 인코딩 오류를 막는가.

        errors='replace' 없이 cp949로 쓰면 UnicodeEncodeError가 나고,
        붙이면 안 난다는 것을 직접 보인다.
        """
        raw = io.BytesIO()
        strict = io.TextIOWrapper(raw, encoding="cp949")
        with self.assertRaises(UnicodeEncodeError):
            strict.write(CP949_UNSAFE)
            strict.flush()

        raw2 = io.BytesIO()
        safe = io.TextIOWrapper(raw2, encoding="cp949", errors="replace")
        safe.write(CP949_UNSAFE)      # 예외 없이 통과해야 한다
        safe.flush()


def _cp949_unsafe_chars(text):
    """text 안에서 cp949로 인코딩할 수 없는 글자들을 모아 돌려준다.

    "어떤 글자가 위험한가"를 목록으로 적어 두지 않고 **직접 시도해서** 가른다.
    실측해 보면 한글·★··(가운뎃점)·→·²·× 는 cp949에 있어서 안전하고,
    —(줄표)·⚠·✔ 가 없어서 죽는다. 목록을 손으로 관리하면 언젠가 어긋나므로
    파이썬에게 직접 물어보는 편이 정확하다.
    """
    bad = set()
    for ch in text:
        try:
            ch.encode("cp949")
        except UnicodeEncodeError:
            bad.add(ch)
    return bad


class ScriptsCallTheHelperTest(unittest.TestCase):
    """★재발 방지 — cp949에 없는 글자를 print 하는 scripts/*.py 가 보호 장치를 부르는가.

    이 프로젝트가 네 번이나 같은 버그에 걸린 이유는 "새 파일을 만들 때
    빠뜨려서"였다. 사람이 기억하는 대신 테스트가 기억하게 한다.
    """

    EXEMPT = {
        "camera_check.py",        # main() 안에서 같은 처리를 직접 한다(자식 프로세스 규약)
    }

    def _risky_scripts(self):
        """print 안에 cp949 불가 글자가 든 스크립트만 골라낸다."""
        scripts_dir = os.path.join(ROOT_DIR, "scripts")
        for name in sorted(os.listdir(scripts_dir)):
            if not name.endswith(".py") or name in self.EXEMPT:
                continue
            if name.endswith("_launcher.py"):
                continue   # 런처는 SetConsoleOutputCP로 따로 처리한다
            src = open(os.path.join(scripts_dir, name), encoding="utf-8").read()
            prints = re.findall(r"print\((.{0,600}?)\)", src, re.S)
            unsafe = set()
            for p in prints:
                unsafe |= _cp949_unsafe_chars(p)
            if unsafe:
                yield name, src, unsafe

    def test_unsafe_printing_scripts_enable_utf8(self):
        missing = []
        checked = 0
        for name, src, unsafe in self._risky_scripts():
            checked += 1
            if "enable_utf8_output" not in src and "reconfigure" not in src:
                missing.append(f"{name}({''.join(sorted(unsafe))})")
        self.assertGreater(checked, 0, "검사 대상 스크립트를 못 찾았다 — 테스트가 헛돌고 있다")
        self.assertEqual(
            missing, [],
            "cp949에 없는 글자를 print 하는데 인코딩 보호가 없다. main() 맨 앞에서 "
            "console.enable_utf8_output() 를 부를 것 -> " + ", ".join(missing))

    def test_detector_itself_works(self):
        """검사기가 헛돌지 않는지 — 위험한 글자는 잡고 안전한 글자는 안 잡아야 한다."""
        self.assertEqual(_cp949_unsafe_chars("정상적인 한글과 ★ · → ² ×"), set())
        self.assertIn("—", _cp949_unsafe_chars("줄표 — 가 들었다"))


if __name__ == "__main__":
    unittest.main()
