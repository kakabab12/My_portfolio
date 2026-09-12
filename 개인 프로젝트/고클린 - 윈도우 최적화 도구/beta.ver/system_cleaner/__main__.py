# -*- coding: utf-8 -*-
"""진입점:  python -m system_cleaner"""

from __future__ import annotations

import os
import sys


def main() -> int:
    if os.name != "nt":
        print("This tool only runs on Windows.")
        return 1

    from .core import Config, is_admin, relaunch_as_admin

    cfg = Config()
    admin = is_admin()

    # 권한이 없으면 한 번만 승격을 시도하고, 실패하거나 사용자가 UAC 를 취소하면
    # '제한 모드'로 그대로 실행한다. (조용히 종료하면 사용자는 앱이 안 켜졌다고 느낀다)
    if not admin and "--no-elevate" not in sys.argv:
        if relaunch_as_admin():
            return 0
        sys.argv.append("--no-elevate")

    from .app import SystemCleanerApp

    app = SystemCleanerApp(cfg, admin)
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
