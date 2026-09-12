# -*- coding: utf-8 -*-
"""SYSTEM CLEANER 실행 진입점.

  python run.py              일반 실행 (권한 없으면 UAC 승격 시도)
  python run.py --no-elevate 승격 없이 제한 모드로 실행
"""

import sys

from system_cleaner.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
