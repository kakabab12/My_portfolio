# -*- coding: utf-8 -*-
"""베타 패키지 스모크 테스트 (읽기 전용).

tkinter 의 after() 는 메인 스레드가 mainloop() 안에 있을 때만 동작하므로,
update() 루프가 아니라 진짜 이벤트 루프 위에서 제너레이터로 단계를 진행한다.
"""
import sys, time, pathlib
sys.argv = [sys.argv[0], "--no-elevate"]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

fails = []
def ok(m):  print(f"  [OK]   {m}")
def bad(m): print(f"  [FAIL] {m}"); fails.append(m)
def check(cond, msg): ok(msg) if cond else bad(msg)

from system_cleaner import core, i18n, tasks
from system_cleaner.app import SystemCleanerApp, PAGE_CLASSES


class Wait:
    def __init__(self, pred, timeout=15.0, label=""):
        self.pred, self.deadline, self.label = pred, time.time() + timeout, label
    def done(self):
        try:
            return bool(self.pred())
        except Exception:
            return False
    def expired(self):
        return time.time() > self.deadline


class Driver:
    """제너레이터가 yield 하는 값(초/Wait/None)에 따라 after 로 단계를 진행."""
    def __init__(self, app, gen):
        self.app, self.gen, self.waiting, self.resume_at = app, gen, None, 0.0
    def start(self):
        self.app.after(200, self._tick)
    def _tick(self):
        try:
            if self.waiting is not None:
                if self.waiting.done() or self.waiting.expired():
                    if self.waiting.expired() and not self.waiting.done():
                        bad(f"시간 초과: {self.waiting.label}")
                    self.waiting = None
                else:
                    return self.app.after(100, self._tick)
            if time.time() < self.resume_at:
                return self.app.after(50, self._tick)
            value = next(self.gen)
            if isinstance(value, Wait):
                self.waiting = value
            elif isinstance(value, (int, float)):
                self.resume_at = time.time() + value
        except StopIteration:
            return self.app.quit()
        except Exception:
            import traceback; traceback.print_exc()
            bad("테스트 진행 중 예외")
            return self.app.quit()
        self.app.after(50, self._tick)


# ---------------------------------------------------------------- 준비
missing = [k for k, v in i18n.STR.items() if not v[0] or not v[1]]
check(not missing, f"번역 누락 없음 ({len(i18n.STR)}개 키)")

cfg = core.Config()
for k in ("opt_recycle_bin", "opt_winsock_reset", "opt_prefetch", "opt_kill_office"):
    check(cfg[k] is False, f"위험 옵션 기본 OFF: {k}")
check(cfg["opt_restore_point"] is True, "복원 지점 기본 ON")
check(core.is_protected(core.Path("C:/Windows")), "C:\\Windows 보호")
check(core.is_protected(core.Path("C:/")), "드라이브 루트 보호")
check(not core.is_protected(core.Path("C:/Windows/Temp")), "Temp 는 삭제 허용")

app = SystemCleanerApp(cfg, admin=core.is_admin())
ok(f"메인 창 생성 (admin={core.is_admin()})")

idle = lambda: not (app.worker and app.worker.is_alive())
rows = lambda k: len(app.pages[k].table.all_payloads())


def scenario():
    yield 0.5

    print("== 페이지 전환 ==")
    for key in PAGE_CLASSES:
        app.show_page(key)
        yield 0.25
        page = app.pages.get(key)
        check(page is not None and page.winfo_exists(), f"{key} 페이지 생성")

    print("== 목록 데이터 ==")
    for key, label in (("startup", "시작 프로그램"), ("service", "서비스"),
                       ("programs", "설치 프로그램"), ("process", "프로세스")):
        app.show_page(key)
        app.pages[key].reload()
        yield Wait(lambda k=key: rows(k) > 0, 25, label)
        check(rows(key) > 0, f"{label} {rows(key)}개 로드")
    print(f"        상태줄: {app.pages['startup'].status.cget('text')}")
    print(f"        상태줄: {app.pages['service'].status.cget('text')}")
    print(f"        상태줄: {app.pages['programs'].status.cget('text')}")
    print(f"        상태줄: {app.pages['process'].status.cget('text')}")

    print("== 표 동작 ==")
    tbl = app.pages["programs"].table
    tbl.sort_by("size"); yield 0.2
    asc = [p.size_bytes for p in tbl.all_payloads()[:5]]
    tbl.sort_by("size"); yield 0.2
    desc = [p.size_bytes for p in tbl.all_payloads()[:5]]
    check(asc == sorted(asc) and desc == sorted(desc, reverse=True),
          f"크기 정렬 (표시 문자열이 아니라 실제 바이트로) {desc[:2]}")

    t2 = app.pages["startup"].table
    t2.set_all(True); yield 0.2
    check(t2.checked_count() == len(t2.all_payloads()) > 0, "전체 체크")
    t2.set_all(False); yield 0.2
    check(t2.checked_count() == 0, "전체 해제")

    print("== 언어 전환 ==")
    lang0 = app.lang
    app.toggle_language()
    yield 1.0
    check(app.lang != lang0, f"언어 전환 {lang0} -> {app.lang}")
    nav_text = app.nav_buttons["startup"].cget("text")
    check("Startup Apps" in nav_text or "시작 프로그램" in nav_text,
          f"네비 텍스트 갱신: {nav_text.strip()}")
    app.toggle_language()
    yield 1.0

    print("== 내 PC 정보 ==")
    app.show_page("info"); yield 0.3
    app.run_task(tasks.task_info, page=app.pages["info"])
    yield Wait(idle, 120, "task_info")
    out = app.pages["info"].terminal.text()
    check("CPU" in out and "GPU" in out and "DISK" in out, "PC 정보 리포트 생성")
    for line in out.splitlines():
        if line.startswith("║   ") and any(
                k in line for k in ("CPU ", "GPU ", "RAM ", "OS ", "P-IP", "DISK")):
            print("        " + line.strip()[:104])

    print("== 대시보드 정밀 검사 ==")
    app.show_page("dashboard"); yield 0.3
    app.pages["dashboard"].deep_scan()
    yield Wait(idle, 180, "deep_scan")
    h = app.pages["dashboard"]._health
    check(h is not None, "정밀 검사 완료")
    if h:
        print(f"        점수 {h.score} · 정리가능 {core.human_bytes(h.junk_bytes)} · "
              f"시작 {h.startup_count}개 · 휴지통 {core.human_bytes(h.bin_bytes)}")
        for kind, text in h.advice:
            print(f"        - [{kind}] {text}")

    print("== 청소 검사(삭제 없음) ==")
    app.show_page("clean"); yield 0.3
    app.pages["clean"].scan_only()
    yield Wait(idle, 180, "scan_only")
    out = app.pages["clean"].terminal.text()
    check("═" in out, "청소 대상 미리보기 출력")
    for line in out.splitlines()[:18]:
        if line.strip():
            print("        " + line.strip()[:104])

    print("== 윈도우 설정 ==")
    app.show_page("tweaks"); yield 1.2
    check(len(app.pages["tweaks"]._switches) > 0,
          f"윈도우 설정 {len(app.pages['tweaks']._switches)}개 스위치 생성")

    print("== 디스크 상태 ==")
    app.show_page("disk"); yield Wait(lambda: rows("disk") > 0, 60, "disk")
    check(rows("disk") > 0, f"볼륨 {rows('disk')}개 표시")
    print(f"        {app.pages['disk'].status.cget('text') or '(경고 없음)'}")

    print("== 취소 ==")
    def slow(ctx):
        ctx.begin(1)
        ctx.runner.run(["ping", "-n", "40", "127.0.0.1"], timeout=90)
        ctx.check()
    app.run_task(slow, page=app.pages["clean"])
    yield 1.2
    t0 = time.time()
    app.request_cancel()
    yield Wait(idle, 20, "cancel")
    check(idle() and time.time() - t0 < 12,
          f"취소가 자식 프로세스까지 종료 ({time.time() - t0:.1f}초)")

    print("== 종료 ==")


Driver(app, scenario()).start()
app.mainloop()
try:
    app.destroy()
except Exception:
    pass

print()
if fails:
    print(f"FAILED: {len(fails)}")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL SMOKE TESTS PASSED")
