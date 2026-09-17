# -*- coding: utf-8 -*-
"""오래 걸리는 작업들. 모두 워커 스레드에서 실행되고 TaskContext 로만 UI 와 대화한다."""

from __future__ import annotations

import os
import re
import shutil
import threading
import time
import webbrowser
from dataclasses import dataclass, field

from . import procs, scan, sysinfo
from .core import (Runner, create_restore_point, empty_recycle_bin, human_bytes,
                   is_admin, recycle_bin_size)
from .i18n import t

try:
    import psutil
except ImportError:
    psutil = None


class Cancelled(Exception):
    """사용자가 취소 버튼을 눌렀을 때."""


class QuietContext:
    """상태 표시줄·터미널을 건드리지 않는 조용한 컨텍스트.

    대시보드 점검처럼 읽기만 하는 작업용이다. 이런 작업이 공용 작업 칸(run_task)을
    차지하면, 앱을 켜자마자 다른 기능을 눌렀을 때 '진행 중'으로 거절된다.
    """

    def __init__(self, app, cancel: threading.Event | None = None):
        self.app = app
        self.cancel = cancel or threading.Event()
        self.runner = Runner()
        self.errors: list[str] = []

    lang = property(lambda self: self.app.lang)
    cfg = property(lambda self: self.app.cfg)

    def t(self, key, **kw) -> str:
        return t(key, self.lang, **kw)

    def begin(self, total_steps: int):
        pass

    def step(self, label: str = ""):
        self.check()

    def progress(self, fraction: float):
        pass

    def check(self):
        if self.cancel.is_set():
            raise Cancelled()

    def log(self, msg: str, tag: str = "info"):
        pass

    def summary(self, title, rows):
        pass

    def fail(self, item: str, detail: str = ""):
        self.errors.append(item if not detail else f"{item}: {detail}")


class TaskContext:
    """워커 스레드 <-> UI 사이의 유일한 통로.

    tkinter 는 스레드 세이프하지 않으므로 UI 갱신은 전부 app.ui() 를 거친다.
    """

    def __init__(self, app, cancel: threading.Event):
        self.app = app
        self.cancel = cancel
        self.runner = Runner()
        self.errors: list[str] = []
        self._steps = 1
        self._done = 0

    # --- 언어/설정 ---
    @property
    def lang(self) -> str:
        return self.app.lang

    @property
    def cfg(self):
        return self.app.cfg

    def t(self, key, **kw) -> str:
        return t(key, self.lang, **kw)

    # --- 진행률 ---
    def begin(self, total_steps: int):
        self._steps = max(1, total_steps)
        self._done = 0
        self.app.ui(self.app.set_progress, 0.0)

    def step(self, label: str = ""):
        self.check()
        self._done += 1
        self.app.ui(self.app.set_progress, min(1.0, self._done / self._steps))
        if label:
            self.log(f">> {label}")

    def progress(self, fraction: float):
        self.app.ui(self.app.set_progress, max(0.0, min(1.0, fraction)))

    def check(self):
        if self.cancel.is_set():
            raise Cancelled()

    # --- 출력 ---
    def log(self, msg: str, tag: str = "info"):
        self.app.ui(self.app.log, msg, tag)

    def summary(self, title: str, rows: list[tuple[str, str]]):
        self.app.ui(self.app.log_summary, title, rows)

    def fail(self, item: str, detail: str = ""):
        self.errors.append(item if not detail else f"{item}: {detail}")

    # --- 확인 창 (UI 스레드에서 띄우고 응답까지 워커를 세운다) ---
    def ask(self, factory):
        box: dict = {}
        done = threading.Event()

        def on_ui():
            try:
                box["value"] = factory()
            except Exception as e:  # noqa: BLE001
                box["error"] = e
            finally:
                done.set()

        self.app.ui(on_ui)
        done.wait()
        if "error" in box:
            raise box["error"]
        self.check()
        return box.get("value")

    # --- 복원 지점 ---
    def ensure_restore_point(self, reason: str) -> bool:
        """위험한 작업 전에 복원 지점을 만든다. 실패하면 계속할지 물어본다."""
        if not self.cfg["opt_restore_point"] or not is_admin():
            return True
        self.log(self.t("rp_making"), "dim")
        ok, detail = create_restore_point(self.runner, f"SystemCleaner - {reason}")
        if ok:
            self.log(self.t("rp_ok", d=detail), "ok")
            return True
        self.log(self.t("rp_fail", d=detail), "warn")
        return bool(self.ask(lambda: self.app.confirm(
            self.t("rp_fail", d=detail), self.t("rp_off"))))


# =====================================================================
# 1. 청소
# =====================================================================
@dataclass
class CleanPlan:
    targets: list = field(default_factory=list)
    total: int = 0


def scan_clean(ctx: TaskContext, share: float = 1.0) -> CleanPlan:
    """삭제하지 않고 대상 크기만 잰다. share: 전체 진행률 중 검사가 차지하는 비율."""
    ctx.log(ctx.t("clean_scan"), "dim")
    targets = scan.build_clean_targets(ctx.cfg, ctx.lang)
    if not targets:
        return CleanPlan([], 0)

    def on_progress(frac, tgt):
        ctx.progress(frac * share)

    total = scan.measure_targets(targets, ctx.cancel, on_progress)
    ctx.check()

    rows = [(tg.label, f"{human_bytes(tg.size)}  ({tg.files:,})")
            for tg in targets if tg.size > 0]
    rows.append((ctx.t("clean_freed"), human_bytes(total)))
    ctx.summary(ctx.t("rpt_preview"), rows)
    ctx.log(ctx.t("clean_found", s=human_bytes(total), n=len(targets)), "ok")
    return CleanPlan(targets, total)


def task_clean(ctx: TaskContext):
    """검사 -> 확인 -> 삭제 -> 부가 최적화."""
    ctx.begin(7)
    # 검사가 막대를 100% 까지 채웠다가 첫 단계에서 14% 로 되돌아가던 것을 막는다
    plan = scan_clean(ctx, share=1 / 7)
    if not plan.targets:
        ctx.log(ctx.t("v_none"), "warn")
        return

    warns = []
    if ctx.cfg["opt_recycle_bin"]:
        warns.append(ctx.t("warn_bin"))
    if ctx.cfg["opt_prefetch"]:
        warns.append(ctx.t("warn_pref"))
    if ctx.cfg["opt_winsock_reset"]:
        warns.append(ctx.t("warn_sock"))
    if ctx.cfg["opt_browser_cache"]:
        warns.append(ctx.t("browser_warn"))

    ok = ctx.ask(lambda: ctx.app.confirm(
        ctx.t("clean_confirm", s=human_bytes(plan.total)),
        ctx.t("browser_safe"),
        warn="\n".join(warns) if warns else None))
    if not ok:
        raise Cancelled()

    if ctx.cfg["opt_winsock_reset"] or ctx.cfg["opt_dism"]:
        if not ctx.ensure_restore_point("Clean & Boost"):
            raise Cancelled()

    rows: list[tuple[str, str]] = []

    # --- DNS / ARP ---
    ctx.step("DNS / ARP")
    r = ctx.runner.run(["ipconfig", "/flushdns"], timeout=30)
    ctx.runner.run(["arp", "-d", "*"], timeout=30)
    rows.append((ctx.t("it_dns"), ctx.t("v_ok") if r.ok else ctx.t("v_failed")))
    if not r.ok:
        ctx.fail("flushdns", r.err[:120])

    # --- 파일 삭제 (파이썬으로 직접 지워서 용량을 정확히 집계) ---
    ctx.step("Deleting files")
    freed = deleted = skipped = 0
    browser_freed = 0
    for tgt in plan.targets:
        ctx.check()
        if tgt.special == "recycle_bin":
            if empty_recycle_bin():
                freed += tgt.size
                deleted += tgt.files
            continue
        if tgt.path is None:
            continue
        f, d, s = scan.purge_tree(tgt.path, ctx.cancel)
        freed += f
        deleted += d
        skipped += s
        if tgt.key.startswith("br_"):
            browser_freed += f
        if f > 0:
            ctx.log(f"   - {tgt.label[:44]:<46} {human_bytes(f)}", "dim")

    rows.append((ctx.t("it_temp"),
                 f"{human_bytes(freed - browser_freed)}  ({deleted:,} del / "
                 f"{skipped:,} {ctx.t('clean_inuse')})"))
    if ctx.cfg["opt_browser_cache"]:
        rows.append((ctx.t("it_browser"), human_bytes(browser_freed)))

    # --- 휴지통 ---
    ctx.step("Recycle Bin")
    if ctx.cfg["opt_recycle_bin"]:
        size, items = recycle_bin_size()
        rows.append((ctx.t("it_recycle"),
                     ctx.t("v_ok") if items == 0 else f"{human_bytes(size)} left"))
    else:
        rows.append((ctx.t("it_recycle"), ctx.t("v_skipped")))

    # --- DISM ---
    ctx.step("DISM component cleanup")
    if ctx.cfg["opt_dism"] and is_admin():
        r = ctx.runner.run(["dism.exe", "/online", "/Cleanup-Image",
                            "/StartComponentCleanup", "/Quiet"], timeout=1800)
        if r.timed_out:
            rows.append((ctx.t("it_dism"), ctx.t("v_timeout")))
        else:
            rows.append((ctx.t("it_dism"), ctx.t("v_ok") if r.ok else f"rc={r.rc}"))
            if not r.ok:
                ctx.fail("DISM", f"rc={r.rc}")
    else:
        rows.append((ctx.t("it_dism"), ctx.t("v_skipped")))

    # --- winsock ---
    ctx.step("Winsock")
    if ctx.cfg["opt_winsock_reset"] and is_admin():
        r = ctx.runner.run(["netsh", "winsock", "reset"], timeout=60)
        rows.append((ctx.t("it_socket"), ctx.t("v_reboot") if r.ok else ctx.t("v_failed")))
    else:
        rows.append((ctx.t("it_socket"), ctx.t("v_skipped")))

    # --- 디스크 최적화 ---
    ctx.step("Disk optimize")
    if ctx.cfg["opt_defrag"] and is_admin():
        rows.append((ctx.t("it_defrag"), _optimize_disks(ctx)))
    else:
        rows.append((ctx.t("it_defrag"), ctx.t("v_skipped")))

    ctx.step("Report")
    rows.append((ctx.t("clean_freed"), human_bytes(freed)))
    ctx.summary(ctx.t("rpt_clean"), rows)


def _optimize_disks(ctx: TaskContext) -> str:
    """SSD 는 TRIM(/L), HDD 만 조각 모음(/D). 전 볼륨 defrag 는 SSD 에 불필요하다."""
    plan = sysinfo.optimize_plan(ctx.runner)
    if not plan:
        return ctx.t("v_failed")
    done = []
    for drive, media, flag in plan:
        ctx.check()
        r = ctx.runner.run(["defrag.exe", f"{drive}:", flag], timeout=1800)
        done.append(f"{drive}:{media[:3]}{'OK' if r.ok else 'X'}")
    return "  ".join(done) if done else ctx.t("v_none")


# =====================================================================
# 2. 메모리 정리
# =====================================================================
def task_memory(ctx: TaskContext):
    if psutil is None:
        ctx.log(ctx.t("no_psutil"), "err")
        ctx.fail("psutil")
        return

    ctx.begin(3)
    ctx.step("Reading memory state")
    before, total, pct = procs.memory_state()

    ctx.step("Trimming working sets")
    ok, tried = procs.trim_working_sets(ctx.cancel)

    ctx.step("Measuring")
    time.sleep(1.2)
    after, _, pct2 = procs.memory_state()

    ctx.summary(ctx.t("rpt_mem"), [
        (ctx.t("mem_before"), f"{human_bytes(before)}  ({pct:.0f}% used)"),
        (ctx.t("mem_after"), f"{human_bytes(after)}  ({pct2:.0f}% used)"),
        (ctx.t("mem_gain"), human_bytes(max(0, after - before))),
        (ctx.t("mem_trimmed"), f"{ok} / {tried}"),
    ])
    ctx.log("\n" + ctx.t("mem_desc"), "dim")


# =====================================================================
# 3. 게임 모드
# =====================================================================
def task_gamemode(ctx: TaskContext):
    if psutil is None:
        ctx.log(ctx.t("no_psutil"), "err")
        ctx.fail("psutil")
        return

    ctx.begin(4)
    ctx.step("Scanning processes")
    found = procs.find_game_targets(ctx.cfg["opt_kill_office"])
    if not found:
        ctx.summary(ctx.t("rpt_game"), [(ctx.t("it_target"), ctx.t("v_none"))])
        return

    items = []
    for name, plist in sorted(found.items()):
        rss = sum(p.memory for p in plist)
        suffix = f"({len(plist)}개 · {human_bytes(rss)})" if ctx.lang == "KO" \
            else f"({len(plist)} proc · {human_bytes(rss)})"
        items.append((name, f"{name}   {suffix}", True))

    picked = ctx.ask(lambda: ctx.app.confirm(
        ctx.t("game_pick"), ctx.t("game_hint"),
        items=items, warn=ctx.t("game_save")))
    if not picked:
        raise Cancelled()
    chosen = [n for n, on in picked.items() if on]
    if not chosen:
        raise Cancelled()

    before, _, _ = procs.memory_state()

    ctx.step("Closing apps")
    victims = [p for n in chosen for p in found.get(n, [])]
    closed, failed = procs.close_processes(victims)

    ctx.step("Power plan")
    power = ctx.t("v_skipped")
    if ctx.cfg["opt_game_power"]:
        if procs.set_power_plan(ctx.runner, procs.HIGH_PERF_GUID):
            power = "High performance" if ctx.lang == "EN" else "고성능"
        else:
            power = ctx.t("v_failed") if is_admin() else ctx.t("need_admin")

    ctx.step("Measuring")
    time.sleep(1.2)
    after, _, _ = procs.memory_state()

    rows = [
        (ctx.t("it_target"), f"{len(victims)}"),
        (ctx.t("it_closed"), f"{closed}" + (f"  ({len(failed)} failed)" if failed else "")),
        (ctx.t("mem_before"), human_bytes(before)),
        (ctx.t("mem_after"), human_bytes(after)),
        (ctx.t("it_ram_gain"), human_bytes(max(0, after - before))),
        (ctx.t("it_power"), power),
    ]
    ctx.summary(ctx.t("rpt_game"), rows)
    if failed:
        ctx.fail("종료하지 못한 프로세스", ", ".join(sorted(set(failed))[:5]))


def task_restore_power(ctx: TaskContext):
    ctx.begin(1)
    ctx.step("Power plan")
    ok = procs.set_power_plan(ctx.runner, procs.BALANCED_GUID)
    ctx.summary(ctx.t("rpt_game"), [
        (ctx.t("it_power"), ("Balanced" if ctx.lang == "EN" else "균형 조정")
         if ok else ctx.t("v_failed"))])


# =====================================================================
# 4. 업데이트
# =====================================================================
_WINGET_ROW = re.compile(r"^(?P<name>.+?)\s{2,}(?P<id>[\w\.\-\+]+)\s{2,}"
                         r"(?P<cur>[\w\.\-\+]+)\s{2,}(?P<new>[\w\.\-\+]+)\s{2,}")


def _parse_winget_list(out: str) -> list[tuple[str, str, str]]:
    """winget upgrade 출력에서 (이름, 현재 버전, 새 버전) 뽑기."""
    rows = []
    for line in out.splitlines():
        if not line.strip() or set(line.strip()) <= {"-", "─"}:
            continue
        m = _WINGET_ROW.match(line)
        if m and m.group("cur") != m.group("new"):
            rows.append((m.group("name").strip(), m.group("cur"), m.group("new")))
    return rows


def task_update(ctx: TaskContext):
    ctx.begin(5)
    rows: list[tuple[str, str]] = []

    # --- winget ---
    ctx.step("winget")
    if shutil.which("winget") is None:
        rows.append((ctx.t("it_sw"), ctx.t("v_notfound")))
    else:
        r = ctx.runner.run(["winget", "upgrade", "--include-unknown",
                            "--accept-source-agreements", "--disable-interactivity"],
                           timeout=240)
        pending = _parse_winget_list(r.out)
        if not pending:
            rows.append((ctx.t("it_sw"), ctx.t("upd_none")))
        else:
            ctx.summary(ctx.t("upd_list"),
                        [(name[:38], f"{cur}  ->  {new}") for name, cur, new in pending])
            go = ctx.ask(lambda: ctx.app.confirm(
                ctx.t("upd_all_q", n=len(pending)),
                "\n".join(f"· {n}" for n, _, _ in pending[:12])))
            if not go:
                rows.append((ctx.t("it_sw"), ctx.t("v_skipped")))
            else:
                r2 = ctx.runner.run(
                    ["winget", "upgrade", "--all", "--include-unknown", "--silent",
                     "--accept-package-agreements", "--accept-source-agreements",
                     "--disable-interactivity"], timeout=1800)
                if r2.timed_out:
                    rows.append((ctx.t("it_sw"), ctx.t("v_timeout")))
                    ctx.fail("winget", "30분 초과")
                else:
                    rows.append((ctx.t("it_sw"),
                                 f"{ctx.t('v_ok')} ({len(pending)} pkg)" if r2.ok
                                 else f"{ctx.t('v_failed')} (rc={r2.rc})"))

    # --- Windows Update / Store ---
    ctx.step("Windows Update")
    rows.append((ctx.t("it_os"),
                 ctx.t("v_opened") if _open_uri("ms-settings:windowsupdate-action")
                 else ctx.t("v_failed")))
    ctx.step("Microsoft Store")
    rows.append((ctx.t("it_store"),
                 ctx.t("v_opened") if _open_uri("ms-windows-store://downloadsandupdates")
                 else ctx.t("v_failed")))

    # --- GPU 드라이버 ---
    ctx.step("GPU driver")
    hw = sysinfo.hardware_info(ctx.runner)
    name, url = sysinfo.gpu_driver_site(hw.get("gpu", ""))
    webbrowser.open(url)
    rows.append((ctx.t("it_gpu"), f"{name} {ctx.t('v_opened')}"))

    ctx.step("Report")
    ctx.summary(ctx.t("rpt_update"), rows)
    ctx.log("\n[!] " + ctx.t("upd_manual"), "warn")


def _open_uri(uri: str) -> bool:
    try:
        os.startfile(uri)   # 'start' 셸 명령보다 빠르고 안전하다
        return True
    except OSError:
        return False


# =====================================================================
# 5. 네트워크
# =====================================================================
def task_network(ctx: TaskContext):
    ctx.begin(len(sysinfo.DNS_SERVERS) + 2)

    ctx.step("Adapters")
    adapters = sysinfo.active_adapters(ctx.runner)
    if adapters:
        ctx.summary("ADAPTER" if ctx.lang == "EN" else "네트워크 어댑터", [
            (a.name, f"{a.ipv4}  gw {a.gateway}  dns {', '.join(a.dns) or '-'}")
            for a in adapters])

    results = []
    for i, (label, primary, secondary) in enumerate(sysinfo.DNS_SERVERS):
        ctx.step(f"ping {label}")
        res = sysinfo.ping(ctx.runner, primary, label)
        res.secondary = secondary
        results.append(res)

    ctx.step("Report")
    rows = []
    best = None
    for res in results:
        if res.ok:
            value = (f"{res.avg:.0f}ms ({res.grade(ctx.lang)})  "
                     f"min {res.lo} / max {res.hi} / jitter {res.jitter:.1f} / loss {res.loss}%")
            if best is None or res.avg < best.avg:
                best = res
        else:
            value = ctx.t("v_failed")
        rows.append((f"{res.label} ({res.host})", value))
    ctx.summary(ctx.t("rpt_net"), rows)

    if best is not None:
        msg = (f"가장 빠른 DNS: {best.label} ({best.host}) {best.avg:.0f}ms"
               if ctx.lang == "KO"
               else f"Fastest DNS: {best.label} ({best.host}) {best.avg:.0f}ms")
        ctx.log("\n" + msg, "ok")
        ctx.app.ui(ctx.app.set_best_dns, best, adapters)


# =====================================================================
# 6. 내 PC 정보
# =====================================================================
def task_info(ctx: TaskContext):
    ctx.begin(4)

    ctx.step("Hardware")
    hw = sysinfo.hardware_info(ctx.runner)
    if not hw:
        ctx.fail("hardware query")

    ctx.step("Storage")
    vols = sysinfo.volume_info(ctx.runner)
    disks = sysinfo.disk_info(ctx.runner)

    ctx.step("Network")
    lip = sysinfo.local_ip(ctx.runner)
    pip = sysinfo.public_ip()

    ctx.step("Report")
    ram = f"{hw.get('ram_gb', '-')} GB"
    if hw.get("ram_type"):
        ram += f"  ({hw['ram_type']})"
    if psutil:
        vm = psutil.virtual_memory()
        ram += f"  · {human_bytes(vm.available)} free"

    up = sysinfo.uptime_seconds()
    up_txt = f"{int(up // 86400)}d {int(up % 86400 // 3600)}h {int(up % 3600 // 60)}m"

    rows = [
        ("CPU", f"{hw.get('cpu', '-')}  ({hw.get('cores', '-')}C/{hw.get('threads', '-')}T "
                f"{hw.get('clock', '-')}MHz)"),
        ("M/B", hw.get("board", "-")),
        ("BIOS", hw.get("bios", "-")),
        ("GPU", hw.get("gpu", "-")),
        ("RAM", ram),
        ("OS", f"{hw.get('os', '-')}  build {hw.get('build', '-')}"),
        ("BOOT", f"{hw.get('boot', '-')}  (+{up_txt})"),
        ("HOST", f"{hw.get('host', '-')} / {hw.get('user', '-')}"),
    ]
    for d in disks:
        extra = f"  {d.health}"
        if d.temperature is not None:
            extra += f" {d.temperature}°C"
        rows.append(("DISK", f"{d.name}  {d.media}  {human_bytes(d.size)}{extra}"))
    for v in vols:
        rows.append((f"  {v.letter}:", f"{human_bytes(v.free)} free / {human_bytes(v.size)}"
                                       f"  ({v.percent_free}%)  {v.label}"))
    rows.append(("L-IP", lip or "-"))
    rows.append(("P-IP", pip or "-"))

    ctx.summary(ctx.t("rpt_spec"), rows)


# =====================================================================
# 7. 대시보드 점검
# =====================================================================
@dataclass
class Health:
    score: int = 100
    cpu: float = 0.0
    ram_percent: float = 0.0
    ram_available: int = 0
    ram_total: int = 0
    volumes: list = field(default_factory=list)
    uptime: float = 0.0
    junk_bytes: int = 0
    bin_bytes: int = 0
    bin_items: int = 0
    startup_count: int = 0
    startup_broken: int = 0
    process_count: int = 0
    disks: list = field(default_factory=list)
    advice: list = field(default_factory=list)


def collect_health(ctx: TaskContext, deep: bool = True) -> Health:
    """대시보드용 종합 점검. deep=False 면 파일 스캔을 건너뛴다."""
    from . import startup as startup_mod

    h = Health()
    steps = 6 if deep else 3
    ctx.begin(steps)

    ctx.step("System")
    if psutil:
        h.cpu = psutil.cpu_percent(interval=0.4)
        vm = psutil.virtual_memory()
        h.ram_percent, h.ram_available, h.ram_total = vm.percent, vm.available, vm.total
        h.process_count = len(psutil.pids())
    h.uptime = sysinfo.uptime_seconds()
    h.volumes = sysinfo.volume_info_fast()

    ctx.step("Startup")
    try:
        entries = startup_mod.list_startup(ctx.runner)
        h.startup_count = sum(1 for e in entries if e.enabled)
        h.startup_broken = sum(1 for e in entries if not e.target_exists)
    except Exception:
        pass

    ctx.step("Recycle Bin")
    h.bin_bytes, h.bin_items = recycle_bin_size()

    if deep:
        ctx.step("Junk files")
        targets = scan.build_clean_targets(ctx.cfg, ctx.lang)
        h.junk_bytes = scan.measure_targets(targets, ctx.cancel)

        ctx.step("Disks")
        h.disks = sysinfo.disk_info(ctx.runner)

        ctx.step("Scoring")

    # --- 점수/조언 ---
    # 조언은 번역된 문장이 아니라 (분류, 문자열 키, 값) 으로 남긴다.
    # 문장으로 저장하면 검사 뒤 언어를 바꿔도 검사할 때 언어로 남는다.
    score = 100
    for v in h.volumes:
        if v.size and v.percent_free < 15:
            score -= 20 if v.percent_free < 8 else 10
            h.advice.append(("disk", "adv_disk_low", dict(d=v.letter, p=v.percent_free)))
    if h.junk_bytes > 2 * 1024 ** 3:
        score -= 10
        h.advice.append(("clean", "adv_junk", dict(s=human_bytes(h.junk_bytes))))
    elif h.junk_bytes > 512 * 1024 ** 2:
        score -= 5
        h.advice.append(("clean", "adv_junk", dict(s=human_bytes(h.junk_bytes))))
    if h.startup_count > 10:
        score -= 10
        h.advice.append(("startup", "adv_startup", dict(n=h.startup_count)))
    if h.startup_broken:
        score -= 3
        h.advice.append(("startup", "adv_broken_startup", dict(n=h.startup_broken)))
    if h.ram_percent >= 85:
        score -= 10
        h.advice.append(("memory", "adv_ram", dict(p=int(h.ram_percent))))
    days = int(h.uptime // 86400)
    if days >= 7:
        score -= 5
        h.advice.append(("system", "adv_uptime", dict(d=days)))
    if h.bin_bytes > 1024 ** 3:
        score -= 3
        h.advice.append(("clean", "adv_bin", dict(s=human_bytes(h.bin_bytes))))
    for d in h.disks:
        if not d.healthy:
            score -= 25
            h.advice.append(("disk", "adv_disk_bad", dict(n=d.name, h=d.health)))

    h.score = max(0, min(100, score))
    return h
