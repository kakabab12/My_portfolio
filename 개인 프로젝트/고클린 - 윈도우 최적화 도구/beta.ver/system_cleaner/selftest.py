# -*- coding: utf-8 -*-
"""자가 검사 (읽기 전용).

    SystemCleaner.exe --selftest [보고서 경로]
    python run.py --selftest

소스에서 잘 돌아도 exe 로 묶으면 깨지는 것들이 있다. 빠진 모듈, 번들에 안 들어간
테마 파일, 콘솔이 없어서 사라지는 예외, 창 모드 exe 의 subprocess 핸들 문제 등.
그래서 테스트를 tests/ 밖에 두지 않고 패키지 안에 넣어, 빌드한 exe 로 같은 검사를
돌릴 수 있게 했다.

파일 삭제 · 프로세스 종료 · 설정 변경은 절대 호출하지 않는다.
(언어 전환은 설정 파일에 쓰지만 두 번 바꿔 원래대로 돌려놓는다)

tkinter 의 after() 는 메인 스레드가 mainloop() 안에 있을 때만 동작한다.
update() 루프로 돌리면 워커 스레드의 UI 콜백이 전부 조용히 버려지므로,
진짜 이벤트 루프 위에서 제너레이터로 단계를 진행한다.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path


class _Result:
    def __init__(self, echo: bool):
        self.echo = echo
        self.lines: list[str] = []
        self.passed: list[str] = []
        self.failed: list[str] = []

    def say(self, text: str):
        self.lines.append(text)
        if self.echo:
            try:
                print(text, flush=True)
            except Exception:
                pass

    def check(self, cond, msg: str):
        if cond:
            self.passed.append(msg)
            self.say(f"  [OK]   {msg}")
        else:
            self.failed.append(msg)
            self.say(f"  [FAIL] {msg}")


class Wait:
    def __init__(self, pred, timeout: float = 15.0, label: str = ""):
        self.pred, self.deadline, self.label = pred, time.time() + timeout, label

    def done(self) -> bool:
        try:
            return bool(self.pred())
        except Exception:
            return False

    def expired(self) -> bool:
        return time.time() > self.deadline


class _Driver:
    """제너레이터가 yield 하는 값(초 / Wait)에 맞춰 after() 로 단계를 진행한다."""

    def __init__(self, app, gen, result: _Result):
        self.app, self.gen, self.r = app, gen, result
        self.waiting: Wait | None = None
        self.resume_at = 0.0

    def start(self):
        self.app.after(300, self._tick)

    def _tick(self):
        try:
            if self.waiting is not None:
                if self.waiting.done():
                    self.waiting = None
                elif self.waiting.expired():
                    self.r.check(False, f"시간 초과: {self.waiting.label}")
                    self.waiting = None
                else:
                    self.app.after(100, self._tick)
                    return
            if time.time() < self.resume_at:
                self.app.after(50, self._tick)
                return
            value = next(self.gen)
            if isinstance(value, Wait):
                self.waiting = value
            elif isinstance(value, (int, float)):
                self.resume_at = time.time() + value
        except StopIteration:
            self.app.quit()
            return
        except Exception as e:  # noqa: BLE001
            import traceback
            self.r.check(False, f"검사 진행 중 예외: {e}")
            self.r.say(traceback.format_exc())
            self.app.quit()
            return
        self.app.after(50, self._tick)


# =====================================================================
# 창 없이 확인하는 것들
# =====================================================================
def _static_checks(r: _Result):
    from . import APP_VERSION, core, i18n

    r.say("== 실행 환경 ==")
    r.say(f"        version   {APP_VERSION}")
    r.say(f"        frozen    {core.is_frozen()}  ({sys.executable})")
    r.say(f"        python    {platform.python_version()}  {platform.architecture()[0]}")
    r.say(f"        windows   {platform.version()}")
    r.say(f"        admin     {core.is_admin()}")

    r.say("== 번들 구성 ==")
    try:
        import customtkinter as ctk
        theme = ctk.ThemeManager.theme
        r.check(bool(theme) and "CTkButton" in theme,
                f"customtkinter {ctk.__version__} 테마 파일 로드")
    except Exception as e:  # noqa: BLE001
        r.check(False, f"customtkinter 로드 실패: {e}")

    try:
        import tkinter
        r.check(tkinter.TkVersion >= 8.6, f"Tcl/Tk {tkinter.TkVersion}")
    except Exception as e:  # noqa: BLE001
        r.check(False, f"tkinter 로드 실패: {e}")

    try:
        import psutil
        r.check(psutil.virtual_memory().total > 0, f"psutil {psutil.__version__}")
    except Exception as e:  # noqa: BLE001
        r.check(False, f"psutil 로드 실패: {e}")

    r.check(core.ICON_PATH.exists(), f"아이콘 리소스 포함 ({core.ICON_PATH.name})")

    # 창 모드 exe 는 표준 입출력 핸들이 없어서 subprocess 가 'handle is invalid' 로
    # 실패하는 경우가 있다. PowerShell 과 콘솔 명령 둘 다 실제로 돌려본다.
    runner = core.Runner()
    p = runner.run(["cmd", "/c", "echo", "selftest"], timeout=20)
    r.check(p.ok and "selftest" in p.out, f"외부 명령 실행 (rc={p.rc})")
    p = runner.run(["cmd", "/c", "exit", "3"], timeout=20)
    r.check(not p.ok and p.rc == 3, "실패한 명령을 실패로 판정")
    p = runner.powershell("Write-Output '한글확인'", timeout=60)
    r.check("한글확인" in p.out, f"PowerShell 실행 + 한글 인코딩 ({p.out!r})")

    r.say("== 승격 명령 ==")
    target, params = core.elevation_command()
    if core.is_frozen():
        r.check(os.path.normcase(target) == os.path.normcase(sys.executable)
                and target.lower().endswith(".exe"),
                "exe 로 실행 시 자기 자신을 다시 실행")
        r.check(".py" not in params.lower(),
                "exe 로 실행 시 스크립트 경로를 인자로 끼워 넣지 않음")
    else:
        r.check(".py" in params.lower(), "소스 실행 시 스크립트 경로를 인자로 전달")
    r.say(f"        {target}  {params}")

    r.say("== 기본값 / 보호 장치 ==")
    missing = [k for k, v in i18n.STR.items() if not v[0] or not v[1]]
    r.check(not missing, f"번역 누락 없음 ({len(i18n.STR)}개)")
    defaults = core.DEFAULT_CONFIG
    for key in ("opt_recycle_bin", "opt_winsock_reset", "opt_prefetch", "opt_kill_office"):
        r.check(defaults[key] is False, f"위험 옵션 기본 OFF: {key}")
    r.check(core.is_protected(Path("C:/Windows")), "C:\\Windows 삭제 거부")
    r.check(core.is_protected(Path("C:/")), "드라이브 루트 삭제 거부")
    r.check(not core.is_protected(Path("C:/Windows/Temp")), "Windows\\Temp 는 정리 허용")


# =====================================================================
# 창을 띄워서 확인하는 것들
# =====================================================================
def _scenario(app, r: _Result):
    from . import core, tasks
    from .app import PAGE_CLASSES

    def idle():
        return not (app.worker and app.worker.is_alive())

    def rows(key):
        return len(app.pages[key].table.all_payloads())

    yield 0.5

    r.say("== 오류 기록 장치 ==")
    # 일부러 콜백에서 예외를 내서, exe 에서도 파일로 남는지 본다.
    # 실제 로그를 더럽히지 않도록 임시 폴더로 잠깐 돌린다.
    real_log_dir = core.LOG_DIR
    probe_dir = Path(tempfile.mkdtemp(prefix="sc_selftest_"))
    core.LOG_DIR = probe_dir
    before = len(core.ERROR_RECORDS)

    def boom():
        raise RuntimeError("selftest: intentional callback error")

    app.after(0, boom)
    yield 0.6
    core.LOG_DIR = real_log_dir
    logged = list(probe_dir.glob("error_*.log"))
    r.check(len(core.ERROR_RECORDS) == before + 1 and logged
            and "intentional callback error" in logged[0].read_text(encoding="utf-8"),
            "버튼 콜백 예외가 오류 로그에 기록됨")
    del core.ERROR_RECORDS[before:]
    for f in logged:
        f.unlink(missing_ok=True)
    try:
        probe_dir.rmdir()
    except OSError:
        pass
    baseline_errors = len(core.ERROR_RECORDS)

    r.say("== 화면 생성 ==")
    # 대시보드는 처음 열릴 때 자동 검사를 돌린다. 끝날 때까지 기다린다.
    yield Wait(idle, 180, "대시보드 자동 검사")
    for key in PAGE_CLASSES:
        app.show_page(key)
        yield 0.25
        page = app.pages.get(key)
        r.check(page is not None and page.winfo_exists(), f"{key} 화면")

    r.say("== 목록 불러오기 ==")
    for key, label in (("startup", "시작 프로그램"), ("service", "서비스"),
                       ("programs", "설치 프로그램"), ("process", "프로세스")):
        app.show_page(key)
        app.pages[key].reload()
        yield 0.1
        # 이미 표에 있던 행으로 통과하지 않도록 새로고침 표시("...")가 끝날 때까지 기다린다
        yield Wait(lambda k=key: rows(k) > 0
                   and app.pages[k].status.cget("text") not in ("", "..."), 60, label)
        r.check(rows(key) > 0,
                f"{label} {rows(key)}개  ·  {app.pages[key].status.cget('text')}")

    r.say("== 표 동작 ==")
    tbl = app.pages["programs"].table
    tbl.sort_by("size")
    yield 0.2
    asc = [p.size_bytes for p in tbl.all_payloads()[:5]]
    tbl.sort_by("size")
    yield 0.2
    desc = [p.size_bytes for p in tbl.all_payloads()[:5]]
    r.check(asc == sorted(asc) and desc == sorted(desc, reverse=True),
            "크기 정렬 (표시 문자열이 아니라 실제 바이트 기준)")

    t2 = app.pages["startup"].table
    t2.set_all(True)
    yield 0.2
    r.check(t2.checked_count() == len(t2.all_payloads()) > 0, "전체 체크")
    t2.set_all(False)
    yield 0.2
    r.check(t2.checked_count() == 0, "전체 해제")

    r.say("== 언어 전환 ==")
    lang0 = app.lang
    app.toggle_language()
    yield 1.5
    nav = app.nav_buttons["startup"].cget("text")
    r.check(app.lang != lang0 and ("Startup Apps" in nav or "시작 프로그램" in nav),
            f"{lang0} -> {app.lang}  ({nav.strip()})")
    app.toggle_language()
    yield 1.5
    r.check(app.lang == lang0, f"원래 언어로 복귀 ({app.lang})")

    r.say("== 작업 실행 (읽기 전용) ==")
    app.show_page("info")
    yield Wait(idle, 60, "대기")
    app.run_task(tasks.task_info, page=app.pages["info"])
    yield 0.5
    yield Wait(idle, 180, "내 PC 정보")
    out = app.pages["info"].terminal.text()
    r.check("CPU" in out and "GPU" in out and "DISK" in out, "내 PC 정보 리포트")
    for line in out.splitlines():
        if line.startswith("║   ") and any(k in line for k in ("CPU ", "GPU ", "RAM ", "OS ")):
            r.say("        " + line.strip()[:110])

    app.show_page("dashboard")
    yield Wait(idle, 60, "대기")
    app.pages["dashboard"].deep_scan()
    yield 0.5
    yield Wait(idle, 240, "대시보드 정밀 검사")
    h = app.pages["dashboard"]._health
    r.check(h is not None, "대시보드 정밀 검사"
            + (f"  ·  점수 {h.score}, 정리 가능 {core.human_bytes(h.junk_bytes)}" if h else ""))

    app.show_page("clean")
    yield Wait(idle, 60, "대기")
    app.pages["clean"].scan_only()
    yield 0.5
    yield Wait(idle, 240, "청소 대상 검사")
    r.check("═" in app.pages["clean"].terminal.text(), "청소 대상 검사 (삭제 없음)")

    app.show_page("disk")
    yield Wait(lambda: rows("disk") > 0 and idle(), 120, "디스크")
    r.check(rows("disk") > 0, f"디스크 볼륨 {rows('disk')}개")

    app.show_page("tweaks")
    yield 1.2
    r.check(len(app.pages["tweaks"]._switches) > 0,
            f"윈도우 설정 스위치 {len(app.pages['tweaks']._switches)}개")

    r.say("== 취소 ==")

    def slow(ctx):
        ctx.begin(1)
        ctx.runner.run(["ping", "-n", "40", "127.0.0.1"], timeout=90)
        ctx.check()

    app.show_page("clean")
    yield Wait(idle, 60, "대기")
    app.run_task(slow, page=app.pages["clean"])
    yield 1.5
    t0 = time.time()
    app.request_cancel()
    yield Wait(idle, 20, "취소")
    r.check(idle() and time.time() - t0 < 12,
            f"취소 시 자식 프로세스까지 종료 ({time.time() - t0:.1f}초)")

    r.say("== 숨은 오류 ==")
    new_errors = core.ERROR_RECORDS[baseline_errors:]
    r.check(not new_errors, "검사 중 기록된 예외 없음"
            + ("" if not new_errors else f": {new_errors[:3]}"))


# =====================================================================
# 실행
# =====================================================================
def run(report_path: str | None = None, echo: bool = True) -> int:
    started = time.time()
    r = _Result(echo)
    r.say(f"SYSTEM CLEANER self-test  ·  {datetime.now():%Y-%m-%d %H:%M:%S}")

    try:
        _static_checks(r)
    except Exception as e:  # noqa: BLE001
        import traceback
        r.check(False, f"정적 검사 예외: {e}")
        r.say(traceback.format_exc())

    app = None
    try:
        from .app import SystemCleanerApp
        from .core import Config, is_admin

        app = SystemCleanerApp(Config(), admin=is_admin())
        r.check(True, "메인 창 생성")
        _Driver(app, _scenario(app, r), r).start()
        # 검사가 어딘가에서 멈추면 영원히 떠 있지 않도록 안전 타이머
        app.after(15 * 60 * 1000, app.quit)
        app.mainloop()
    except Exception as e:  # noqa: BLE001
        import traceback
        r.check(False, f"창 검사 예외: {e}")
        r.say(traceback.format_exc())
    finally:
        if app is not None:
            try:
                app.destroy()
            except Exception:
                pass

    elapsed = time.time() - started
    r.say("")
    if r.failed:
        r.say(f"RESULT: FAILED  {len(r.failed)} / {len(r.passed) + len(r.failed)}  "
              f"({elapsed:.0f}s)")
        for item in r.failed:
            r.say(f"  - {item}")
    else:
        r.say(f"RESULT: PASSED  {len(r.passed)} checks  ({elapsed:.0f}s)")

    if report_path:
        try:
            path = Path(report_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(r.lines) + "\n", encoding="utf-8")
            path.with_suffix(".json").write_text(json.dumps({
                "passed": r.passed, "failed": r.failed, "elapsed": round(elapsed, 1),
                "frozen": bool(getattr(sys, "frozen", False)),
                "executable": sys.executable,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    return 1 if r.failed else 0
