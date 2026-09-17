# -*- coding: utf-8 -*-
"""작업형 화면: 대시보드, 청소, 메모리, 게임 모드, 업데이트, 네트워크, PC 정보."""

from __future__ import annotations

import customtkinter as ctk

from . import procs, sysinfo, tasks
from .core import create_restore_point, human_bytes, is_admin
from .core import LOG_DIR
from .ui import (ACCENT, BORDER, CARD, DANGER, OK, TXT, WARN,
                 Page, StatCard, Terminal, toolbar_button)


# =====================================================================
# 공통: 터미널이 붙은 작업 화면
# =====================================================================
class TaskPage(Page):
    task_fn = None
    run_label_key = "btn_run"
    run_kind = "accent"

    def __init__(self, app, **kw):
        super().__init__(app, **kw)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.header.grid(row=0, column=0, sticky="ew", pady=(0, 12))

        self.toolbar = ctk.CTkFrame(self, fg_color="transparent")
        self.toolbar.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.btn_run = toolbar_button(self.toolbar, self.t(self.run_label_key),
                                      self.run, self.run_kind, width=130)
        self.btn_run.pack(side="left")
        self.build_toolbar(self.toolbar)

        self.terminal = Terminal(self)
        self.terminal.grid(row=2, column=0, sticky="nsew")

    def build_toolbar(self, bar):
        """하위 화면에서 버튼 추가."""

    def run(self):
        if self.task_fn is None:
            return
        self.app.run_task(self.task_fn, page=self)

    def retranslate(self):
        super().retranslate()
        self.btn_run.configure(text=self.t(self.run_label_key))


# =====================================================================
# 대시보드
# =====================================================================
class DashboardPage(Page):
    key = "dashboard"
    title_key = "dash_title"
    desc_key = ""

    def __init__(self, app, **kw):
        super().__init__(app, **kw)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)
        self._timer = None
        self._health = None
        self._auto_scanned = False
        self._scanning = False

        self.header.grid(row=0, column=0, sticky="ew", pady=(0, 14))

        # --- 상단 카드 ---
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.grid(row=1, column=0, sticky="ew")
        for i in range(4):
            top.grid_columnconfigure(i, weight=1, uniform="cards")

        self.card_score = StatCard(top, title=self.t("dash_score"), value="--",
                                   sub="", accent=ACCENT, bar=True, height=118)
        self.card_cpu = StatCard(top, title=self.t("dash_cpu"), value="--%",
                                 accent=OK, bar=True, height=118)
        self.card_ram = StatCard(top, title=self.t("dash_ram"), value="--%",
                                 accent=OK, bar=True, height=118)
        self.card_disk = StatCard(top, title=self.t("dash_disk"), value="--%",
                                  accent=OK, bar=True, height=118)
        for i, card in enumerate((self.card_score, self.card_cpu,
                                  self.card_ram, self.card_disk)):
            card.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 10, 0))

        # --- 하단 카드 ---
        mid = ctk.CTkFrame(self, fg_color="transparent")
        mid.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        for i in range(4):
            mid.grid_columnconfigure(i, weight=1, uniform="cards")

        self.card_junk = StatCard(mid, title=self.t("dash_junk"), value="?",
                                  sub=self.t("btn_scan"), accent=WARN, height=96)
        self.card_startup = StatCard(mid, title=self.t("dash_startup"), value="--",
                                     accent=TXT, height=96)
        self.card_procs = StatCard(mid, title=self.t("dash_procs"), value="--",
                                   accent=TXT, height=96)
        self.card_bin = StatCard(mid, title=self.t("dash_bin"), value="--",
                                 accent=TXT, height=96)
        for i, card in enumerate((self.card_junk, self.card_startup,
                                  self.card_procs, self.card_bin)):
            card.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 10, 0))

        # --- 빠른 실행 ---
        quick = ctk.CTkFrame(self, fg_color="transparent")
        quick.grid(row=3, column=0, sticky="ew", pady=(16, 8))
        self.btn_scan = toolbar_button(quick, self.t("btn_scan"), self.deep_scan,
                                       "accent", width=140)
        self.btn_scan.pack(side="left")
        self.quick_buttons = []
        for key, target in (("nav_clean", "clean"), ("nav_memory", "memory"),
                            ("nav_game", "game"), ("nav_startup", "startup")):
            btn = toolbar_button(quick, self.t(key),
                                 lambda tgt=target: self.app.show_page(tgt), "ghost", width=130)
            btn.pack(side="left", padx=(8, 0))
            self.quick_buttons.append((btn, key))
        self.btn_rp = toolbar_button(quick, self.t("btn_restore"), self.make_restore_point,
                                     "warn", width=130)
        self.btn_rp.pack(side="left", padx=(8, 0))

        # --- 점검 결과 ---
        self.advice_box = ctk.CTkScrollableFrame(
            self, fg_color=CARD, corner_radius=10, border_width=1, border_color=BORDER,
            label_text=self.t("dash_advice"), label_font=ctk.CTkFont(size=12, weight="bold"),
            label_text_color=ACCENT, label_fg_color="#1f1f1f")
        self.advice_box.grid(row=4, column=0, sticky="nsew", pady=(6, 0))
        self._advice_rows = []
        self._advice_labels = []
        self._advice_source = []
        self._set_advice([(None, "dash_scanning", {})])

    # --- 주기적 갱신 ---
    def on_show(self):
        self._tick()
        # 점수가 '--' 인 채로 놓아두면 대시보드가 하는 말이 없다. 처음 한 번은 자동 검사.
        if self._health is None and not self._auto_scanned:
            self._auto_scanned = True
            self.after(600, self.deep_scan)

    def on_hide(self):
        if self._timer:
            self.after_cancel(self._timer)
            self._timer = None

    def _tick(self):
        try:
            self._refresh_live()
        except Exception:
            pass
        self._timer = self.after(2000, self._tick)

    def _refresh_live(self):
        if procs.psutil is None:
            return
        cpu = procs.psutil.cpu_percent(interval=None)
        vm = procs.psutil.virtual_memory()
        self.card_cpu.update_values(value=f"{cpu:.0f}%", fraction=cpu / 100,
                                    accent=_level_color(cpu))
        self.card_ram.update_values(
            value=f"{vm.percent:.0f}%",
            sub=f"{human_bytes(vm.available)} free / {human_bytes(vm.total)}",
            fraction=vm.percent / 100, accent=_level_color(vm.percent))
        self.card_procs.update_values(value=str(len(procs.psutil.pids())))

        vols = sysinfo.volume_info_fast()
        sys_vol = next((v for v in vols if v.letter.upper().startswith("C")), None)
        if sys_vol and sys_vol.size:
            used_pct = 100 - sys_vol.percent_free
            self.card_disk.update_values(
                value=f"{used_pct}%",
                sub=f"{human_bytes(sys_vol.free)} free / {human_bytes(sys_vol.size)}",
                fraction=used_pct / 100, accent=_level_color(used_pct, 80, 92))

        up = sysinfo.uptime_seconds()
        days, rem = int(up // 86400), int(up % 86400)
        up_txt = (self.t("dash_days", d=days, h=rem // 3600) if days
                  else self.t("dash_hours", h=rem // 3600, m=rem % 3600 // 60))
        self.card_score.update_values(sub=f"{self.t('dash_uptime')}  {up_txt}")

    # --- 정밀 검사 ---
    def deep_scan(self):
        """읽기만 하는 점검이라 공용 작업 칸(run_task)을 쓰지 않는다.

        예전에는 앱을 켜자마자 이 점검이 작업 칸을 차지해서, 그 사이 '내 PC 정보' 같은
        다른 기능을 누르면 조용히 거절되고 빈 화면만 남았다.
        """
        if self._scanning:
            return
        self._scanning = True
        self.btn_scan.configure(state="disabled", text=self.t("working"))
        if self._health is None:
            self._set_advice([(None, "dash_scanning", {})])

        def work():
            return tasks.collect_health(tasks.QuietContext(self.app), deep=True)

        self.app.run_bg(work, self._scan_done)

    def is_scanning(self) -> bool:
        return self._scanning

    def _scan_done(self, result):
        self._scanning = False
        self.btn_scan.configure(state="normal", text=self.t("btn_scan"))
        if isinstance(result, Exception):
            self._set_advice([("error", "err_logged", {"path": str(LOG_DIR)})])
            return
        self._apply_health(result)

    def _apply_health(self, health: tasks.Health):
        self._health = health
        color = OK if health.score >= 80 else (WARN if health.score >= 55 else DANGER)
        self.card_score.update_values(value=f"{health.score}", fraction=health.score / 100,
                                      accent=color)
        self.card_junk.update_values(value=human_bytes(health.junk_bytes),
                                     sub=self.t("nav_clean"))
        self.card_startup.update_values(
            value=str(health.startup_count),
            sub=(f"{health.startup_broken} {self.t('st_missing')}"
                 if health.startup_broken else ""))
        self.card_bin.update_values(value=human_bytes(health.bin_bytes),
                                    sub=f"{health.bin_items:,} items")
        if health.advice:
            self._set_advice(health.advice)
        else:
            self._set_advice([(None, "dash_allgood", {})])

    def _set_advice(self, rows):
        """rows: [(분류, 문자열 키, 값)] - 그릴 때 현재 언어로 번역한다."""
        same_shape = (len(rows) == len(self._advice_labels)
                      and [(k, key) for k, key, _ in rows]
                      == [(k, key) for k, key, _ in self._advice_source])
        self._advice_source = rows
        if same_shape and rows:
            # 줄 구성이 같으면 글자만 바꾼다. CTk 위젯은 지우고 새로 만드는 데 줄당
            # 0.1초 가까이 걸려서, 언어를 바꿀 때 대시보드가 눈에 띄게 멈췄다.
            for label, (kind, key, values) in zip(self._advice_labels, rows):
                bullet = "•" if kind is None else "!"
                label.configure(text=f" {bullet}  {self.t(key, **values)}")
            return
        self._advice_labels = []
        for widget in self._advice_rows:
            widget.destroy()
        self._advice_rows = []
        nav = {"clean": "clean", "startup": "startup", "memory": "memory",
               "disk": "disk", "system": None}
        for kind, key, values in rows:
            text = self.t(key, **values)
            row = ctk.CTkFrame(self.advice_box, fg_color="transparent")
            row.pack(fill="x", pady=3, padx=4)
            bullet = "•" if kind is None else "!"
            label = ctk.CTkLabel(row, text=f" {bullet}  {text}", font=ctk.CTkFont(size=12),
                                 text_color=TXT if kind is None else WARN,
                                 anchor="w", justify="left", wraplength=680)
            label.pack(side="left", fill="x", expand=True)
            self._advice_labels.append(label)
            target = nav.get(kind)
            if target:
                toolbar_button(row, "→", lambda tgt=target: self.app.show_page(tgt),
                               "ghost", width=40).pack(side="right", padx=6)
            self._advice_rows.append(row)

    def make_restore_point(self):
        if not self.app.require_admin():
            return
        self.app.run_task(self._rp_task, page=self, status_key="busy")

    def _rp_task(self, ctx: tasks.TaskContext):
        ctx.begin(1)
        ctx.step("Restore point")
        ok, detail = create_restore_point(ctx.runner, "SystemCleaner manual")
        ctx.app.ui(self.app.toast,
                   ctx.t("rp_ok", d=detail) if ok else ctx.t("rp_fail", d=detail),
                   "ok" if ok else "warn")
        if not ok:
            ctx.fail("restore point", detail)

    def retranslate(self):
        super().retranslate()
        self.card_score.set_title(self.t("dash_score"))
        self.card_cpu.set_title(self.t("dash_cpu"))
        self.card_ram.set_title(self.t("dash_ram"))
        self.card_disk.set_title(self.t("dash_disk"))
        self.card_junk.set_title(self.t("dash_junk"))
        self.card_startup.set_title(self.t("dash_startup"))
        self.card_procs.set_title(self.t("dash_procs"))
        self.card_bin.set_title(self.t("dash_bin"))
        self.btn_scan.configure(text=self.t("working" if self._scanning else "btn_scan"))
        self.btn_rp.configure(text=self.t("btn_restore"))
        for btn, key in self.quick_buttons:
            btn.configure(text=self.t(key))
        self.advice_box.configure(label_text=self.t("dash_advice"))
        if self._health:
            self._apply_health(self._health)
        else:
            self._set_advice(self._advice_source)
        self._refresh_live()


def _level_color(value: float, warn: float = 70, danger: float = 88) -> str:
    if value >= danger:
        return DANGER
    if value >= warn:
        return WARN
    return OK


# =====================================================================
# 청소
# =====================================================================
class CleanPage(TaskPage):
    key = "clean"
    title_key = "clean_title"
    desc_key = "clean_desc"
    task_fn = staticmethod(tasks.task_clean)

    def build_toolbar(self, bar):
        self.btn_only_scan = toolbar_button(bar, self.t("btn_scan"), self.scan_only,
                                            "ghost", width=120)
        self.btn_only_scan.pack(side="left", padx=(8, 0))

    def scan_only(self):
        self.app.run_task(self._scan_task, page=self, status_key="busy")

    def _scan_task(self, ctx):
        ctx.begin(1)
        tasks.scan_clean(ctx)

    def retranslate(self):
        super().retranslate()
        self.btn_only_scan.configure(text=self.t("btn_scan"))


# =====================================================================
# 메모리
# =====================================================================
class MemoryPage(TaskPage):
    key = "memory"
    title_key = "mem_title"
    desc_key = "mem_desc"
    task_fn = staticmethod(tasks.task_memory)


# =====================================================================
# 게임 모드
# =====================================================================
class GamePage(TaskPage):
    key = "game"
    title_key = "game_title"
    desc_key = "game_desc"
    task_fn = staticmethod(tasks.task_gamemode)

    def build_toolbar(self, bar):
        self.btn_power = toolbar_button(bar, self.t("game_restore"),
                                        self.restore_power, "ghost", width=180)
        self.btn_power.pack(side="left", padx=(8, 0))

    def restore_power(self):
        # 권한 없이 실행하면 '실패'만 찍혀서 이유를 알 수 없었다
        if not self.app.require_admin():
            return
        self.app.run_task(tasks.task_restore_power, page=self, status_key="busy")

    def retranslate(self):
        super().retranslate()
        self.btn_power.configure(text=self.t("game_restore"))


# =====================================================================
# 업데이트
# =====================================================================
class UpdatePage(TaskPage):
    key = "update"
    title_key = "upd_title"
    desc_key = "upd_desc"
    task_fn = staticmethod(tasks.task_update)


# =====================================================================
# 네트워크
# =====================================================================
class NetworkPage(TaskPage):
    key = "network"
    title_key = "net_title"
    desc_key = "net_desc"
    task_fn = staticmethod(tasks.task_network)
    run_label_key = "net_bench"

    def build_toolbar(self, bar):
        self._best = None
        self._adapters = []
        self.btn_apply = toolbar_button(bar, self.t("net_apply"), self.apply_dns,
                                        "warn", width=180)
        self.btn_apply.pack(side="left", padx=(8, 0))
        self.btn_apply.configure(state="disabled")
        self.btn_auto = toolbar_button(bar, self.t("net_auto"), self.reset_dns,
                                       "ghost", width=180)
        self.btn_auto.pack(side="left", padx=(8, 0))

    def set_best(self, best, adapters):
        self._best, self._adapters = best, adapters
        if best and adapters and is_admin():
            self.btn_apply.configure(state="normal")

    def _adapter(self):
        return self._adapters[0] if self._adapters else None

    def apply_dns(self):
        adapter = self._adapter()
        if not (self._best and adapter) or not self.app.require_admin():
            return
        ok = self.app.confirm(self.t("net_dns_q", a=adapter.name,
                                     d=f"{self._best.label} ({self._best.host})"), "")
        if not ok:
            return
        self.app.run_task(self._apply_task, page=self, status_key="busy")

    def _apply_task(self, ctx):
        ctx.begin(1)
        ctx.step("DNS")
        adapter = self._adapter()
        ok, detail = sysinfo.set_dns(ctx.runner, adapter.name,
                                     self._best.host, self._best.secondary)
        if ok:
            ctx.log("\n" + ctx.t("net_dns_ok", d=self._best.host), "ok")
        else:
            ctx.fail("DNS", detail)

    def reset_dns(self):
        # 권한 확인보다 어댑터 조회(PowerShell)를 먼저 해서, 권한이 없을 때도
        # 화면이 1~2초 멈춘 뒤에야 안내가 나왔다
        if not self.app.require_admin():
            return
        self.app.run_task(self._reset_task, page=self, status_key="busy")

    def _reset_task(self, ctx):
        ctx.begin(2)
        ctx.step("Adapter")
        adapter = self._adapter()
        if adapter is None:
            adapters = sysinfo.active_adapters(ctx.runner)
            adapter = adapters[0] if adapters else None
        if adapter is None:
            ctx.fail("DNS", ctx.t("v_notfound"))
            return
        ctx.step("DNS")
        ok, detail = sysinfo.reset_dns(ctx.runner, adapter.name)
        if ok:
            ctx.log("\n" + ctx.t("net_dns_rev"), "ok")
        else:
            ctx.fail("DNS", detail)

    def retranslate(self):
        super().retranslate()
        self.btn_apply.configure(text=self.t("net_apply"))
        self.btn_auto.configure(text=self.t("net_auto"))


# =====================================================================
# 내 PC 정보
# =====================================================================
class InfoPage(TaskPage):
    key = "info"
    title_key = "info_title"
    desc_key = ""
    task_fn = staticmethod(tasks.task_info)
    run_label_key = "btn_scan"

    def on_show(self):
        if not self.terminal.text().strip():
            self._run_when_free(attempts=150)

    def _run_when_free(self, attempts: int):
        if self.app.current_key != self.key or self.terminal.text().strip():
            return
        if self.app.worker and self.app.worker.is_alive():
            if attempts > 0:
                self.after(400, lambda: self._run_when_free(attempts - 1))
            return
        self.run()
