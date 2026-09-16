# -*- coding: utf-8 -*-
"""스모크 테스트 (읽기 전용).

검사 내용은 system_cleaner/selftest.py 에 있다. 빌드한 exe 에서도
`SystemCleaner.exe --selftest` 로 같은 검사를 돌릴 수 있게 패키지 안에 뒀다.

    python tests/smoke_test.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from system_cleaner.selftest import run  # noqa: E402

if __name__ == "__main__":
    sys.exit(run(report_path=None, echo=True))
