# -*- coding: utf-8 -*-
"""진입점:  python -m system_cleaner

  (옵션 없음)            권한이 없으면 UAC 승격을 한 번 시도
  --no-elevate           승격 없이 제한 모드로 실행
  --selftest [보고서]     읽기 전용 자가 검사 후 종료 (빌드한 exe 검증용)
"""

from __future__ import annotations

import os
import sys
import threading


def _install_error_hooks():
    """메인/스레드에서 잡히지 않은 예외를 파일로 남긴다 (exe 에는 콘솔이 없다)."""
    from .core import log_error

    previous = sys.excepthook

    def on_main(exc_type, exc, tb):
        log_error("main", exc.with_traceback(tb) if exc is not None else None)
        if previous is not None:
            try:
                previous(exc_type, exc, tb)
            except Exception:
                pass

    def on_thread(args):
        if args.exc_type is SystemExit:
            return
        log_error(f"thread:{getattr(args.thread, 'name', '?')}",
                  args.exc_value.with_traceback(args.exc_traceback)
                  if args.exc_value is not None else None)

    sys.excepthook = on_main
    threading.excepthook = on_thread


def _selftest_report_path() -> str | None:
    if "--selftest" not in sys.argv:
        return None
    i = sys.argv.index("--selftest")
    if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
        return sys.argv[i + 1]
    import tempfile
    return os.path.join(tempfile.gettempdir(), "SystemCleaner_selftest.txt")


def main() -> int:
    if os.name != "nt":
        print("This tool only runs on Windows.")
        return 1

    _install_error_hooks()

    report = _selftest_report_path()
    if report is not None:
        from .selftest import run
        return run(report_path=report, echo=sys.stdout is not None)

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
