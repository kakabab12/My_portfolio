# -*- coding: utf-8 -*-
"""목록형 화면: 시작 프로그램, 프로세스, 서비스, 프로그램 삭제, 파일 검사, 디스크."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

from . import procs, programs, scan, services, startup, sysinfo
from .core import (BACKUP_DIR, Runner, human_bytes, is_admin, open_in_explorer,
                   send_to_recycle_bin)
from .ui import (ACCENT, DANGER, OK, TXT_DIM, WARN, DataTable, Page, StatCard,
                 toolbar_button)


class TablePage(Page):
    """헤더 + 툴바 + 표 + 상태줄."""

    columns: list = []
    checkable = True

    def __init__(self, app, **kw):
        super().__init__(app, **kw)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self.header.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        self.toolbar = ctk.CTkFrame(self, fg_color="transparent")
        self.toolbar.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self._buttons: list[tuple[ctk.CTkButton, str]] = []
        self.build_toolbar(self.toolbar)

        self.table = DataTable(self, self.make_columns(), checkable=self.checkable,
                               on_double=self.on_double)
        self.table.grid(row=2, column=0, sticky="nsew")

        self.status = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=12),
                                   text_color=TXT_DIM, anchor="w")
        self.status.grid(row=3, column=0, sticky="ew", pady=(8, 0))

    # --- 하위 구현 ---
    def make_columns(self):
        return self.columns

    def build_toolbar(self, bar):
        pass

    def on_double(self, payload):
        pass

    def reload(self):
        pass

    def rerender(self):
        """언어만 바뀌었을 때. 기본은 다시 불러오기, 데이터를 들고 있는 화면은 다시 그리기만."""
        self.reload()

    # --- 도구 ---
    def add_button(self, bar, key, command, kind="ghost", width=110, pad=(0, 0)):
        btn = toolbar_button(bar, self.t(key), command, kind, width)
        btn.pack(side="left", padx=pad)
        self._buttons.append((btn, key))
        return btn

    def set_status(self, text: str, color: str = TXT_DIM):
        self.status.configure(text=text, text_color=color)

    def checked_or_selected(self):
        items = self.table.checked_payloads()
        return items or self.table.selected_payloads()

    def target_or_warn(self):
        item = self.table.target_payload()
        if item is None:
            self.set_status(self.t("nothing_sel"), WARN)
        return item

    def run_actions(self, work, done):
        """시스템을 바꾸는 동작은 백그라운드에서 한다.

        화면 스레드에서 PowerShell 을 기다리면 서비스 하나당 수십 초씩
        '응답 없음' 상태가 된다.
        """
        self.set_status(self.t("working"), TXT_DIM)
        for btn, _key in self._buttons:
            btn.configure(state="disabled")

        def finished(result):
            for btn, _key in self._buttons:
                btn.configure(state="normal")
            if isinstance(result, Exception):
                self.set_status(str(result), DANGER)
                return
            done(result)

        self.app.run_bg(work, finished)

    def reload_then(self, message: str, color: str):
        """새로고침이 끝난 뒤에 결과 메시지를 보여준다.

        먼저 보여주면 새로고침이 끝나는 순간 '총 N개' 문구에 곧바로 덮인다.
        """
        self._pending_status = (message, color)
        self.reload()

    def show_pending_or(self, text: str, color: str = TXT_DIM):
        pending = getattr(self, "_pending_status", None)
        if pending:
            self._pending_status = None
            self.set_status(*pending)
        else:
            self.set_status(text, color)

    def retranslate(self):
        super().retranslate()
        for btn, key in self._buttons:
            btn.configure(text=self.t(key))
        self.table.set_headings(self.make_columns())
        self.set_status("")          # 이전 언어로 남아 있는 안내 문구 지우기
        self.rerender()


# =====================================================================
# 시작 프로그램
# =====================================================================
class StartupPage(TablePage):
    key = "startup"
    title_key = "startup_title"
    desc_key = "startup_desc"

    def __init__(self, app, **kw):
        self._entries: list = []
        self._loading = False
        super().__init__(app, **kw)

    def make_columns(self):
        return [("name", self.t("col_name"), 200, "w"),
                ("status", self.t("col_status"), 110, "center"),
                ("loc", self.t("col_loc"), 140, "center"),
                ("cmd", self.t("col_cmd"), 420, "w")]

    def build_toolbar(self, bar):
        self.add_button(bar, "btn_refresh", self.reload, "accent")
        self.add_button(bar, "btn_enable", lambda: self.apply(True), "ghost", pad=(8, 0))
        self.add_button(bar, "btn_disable", lambda: self.apply(False), "warn")
        self.add_button(bar, "btn_delete", self.delete, "danger", pad=(8, 0))
        self.add_button(bar, "btn_open", self.open_location, "ghost", width=130, pad=(8, 0))

    def on_show(self):
        # 처음 열었을 때 표가 비어 있던 문제: 이 화면만 on_show 에서 불러오지 않았다
        if not self._entries and not self._loading:
            self.reload()

    def reload(self):
        # 작업 스케줄러 조회에 PowerShell 이 필요해 UI 스레드에서 하면 몇 초 멈춘다
        self._loading = True
        self.set_status("...")
        self.app.run_bg(lambda: startup.list_startup(Runner()), self._apply)

    def rerender(self):
        if self._entries:
            self._apply(self._entries)
        else:
            self.reload()

    def _apply(self, entries):
        self._loading = False
        if isinstance(entries, Exception):
            self.set_status(str(entries), DANGER)
            return
        self._entries = entries
        rows, tags = [], {}
        for i, e in enumerate(entries):
            status = self.t("st_enabled") if e.enabled else self.t("st_disabled")
            if not e.target_exists:
                status = self.t("st_missing")
                tags[i] = "danger"
            elif not e.enabled:
                tags[i] = "dim"
            rows.append((e, (e.name, status, e.location, e.command)))
        self.table.set_rows(rows, tags)
        enabled = sum(1 for e in entries if e.enabled)
        self.show_pending_or(self.t("startup_count", n=len(entries), e=enabled))

    def apply(self, enable: bool):
        items = self.checked_or_selected()
        if not items:
            self.set_status(self.t("nothing_sel"), WARN)
            return
        if any(e.needs_admin for e in items) and not self.app.require_admin():
            return

        def work():
            runner = Runner()
            return [(e, *startup.set_enabled(e, enable, runner)) for e in items]

        def done(results):
            fails = [f"{e.name}: {d}" for e, good, d in results if not good]
            ok = len(results) - len(fails)
            msg = f"{ok} {self.t('v_ok')}" + (
                f" / {len(fails)} {self.t('v_failed')} - {fails[0]}" if fails else "")
            self.reload_then(msg, WARN if fails else OK)

        self.run_actions(work, done)

    def delete(self):
        # 예전엔 작업 스케줄러 항목을 여기서 걸러내서, 그런 항목만 선택하고 삭제를 누르면
        # 실제로는 선택했는데도 '선택한 항목이 없습니다'가 떴다 (예: KMS_Activation)
        items = self.checked_or_selected()
        if not items:
            self.set_status(self.t("nothing_sel"), WARN)
            return
        if any(e.needs_admin for e in items) and not self.app.require_admin():
            return
        warn = self.t("backup_note", p=BACKUP_DIR)
        if any(e.kind == "task" for e in items):
            warn += "\n" + self.t("startup_task_backup")
        ok = self.app.confirm(
            self.t("startup_del_q", n=len(items)),
            "\n".join(f"· {e.name}  [{e.location}]" for e in items[:12]),
            warn=warn, danger=True)
        if not ok:
            return
        startup.backup_entries(items)        # 취소했을 때 백업 파일만 쌓이지 않도록 확인 뒤에

        def work():
            runner = Runner()
            return [(e, *startup.delete_entry(e, runner)) for e in items]

        def done(results):
            fails = [f"{e.name}: {detail}" for e, good, detail in results if not good]
            ok_n = len(results) - len(fails)
            self.reload_then(f"{ok_n} / {len(results)} {self.t('v_ok')}"
                             + (f" - {fails[0]}" if fails else ""),
                             WARN if fails else OK)

        self.run_actions(work, done)

    def open_location(self):
        entry = self.target_or_warn()
        if entry is None:
            return
        target = startup.extract_target(entry.command) if entry.kind != "folder" else entry.source
        if target:
            open_in_explorer(os.path.expandvars(target))

    def on_double(self, payload):
        if payload is not None:
            self.apply(not payload.enabled)


# =====================================================================
# 프로세스
# =====================================================================
class ProcessPage(TablePage):
    key = "process"
    title_key = "proc_title"
    desc_key = "proc_desc"

    def __init__(self, app, **kw):
        self._items: list = []
        super().__init__(app, **kw)
        self._timer = None

    def make_columns(self):
        return [("name", self.t("col_name"), 200, "w"),
                ("pid", self.t("col_pid"), 70, "center"),
                ("mem", self.t("col_mem"), 110, "e"),
                ("cpu", self.t("col_cpu"), 70, "e"),
                ("user", self.t("col_user"), 110, "center"),
                ("path", self.t("col_path"), 380, "w")]

    def build_toolbar(self, bar):
        self.add_button(bar, "btn_refresh", self.reload, "accent")
        self.add_button(bar, "btn_kill", self.kill, "danger", width=130, pad=(8, 0))
        self.add_button(bar, "btn_open", self.open_location, "ghost", width=130, pad=(8, 0))
        self.auto_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(bar, text="AUTO 3s", variable=self.auto_var, width=90,
                        font=ctk.CTkFont(size=11), text_color=TXT_DIM,
                        fg_color=ACCENT, hover_color=ACCENT, checkmark_color="black",
                        command=self._toggle_auto).pack(side="left", padx=(14, 0))

    def on_show(self):
        self.reload()

    def on_hide(self):
        self._stop_auto()

    def _toggle_auto(self):
        self._stop_auto()
        if self.auto_var.get():
            self._timer = self.after(3000, self._auto_tick)

    def _auto_tick(self):
        self.reload()
        if self.auto_var.get():
            self._timer = self.after(3000, self._auto_tick)

    def _stop_auto(self):
        if self._timer:
            self.after_cancel(self._timer)
            self._timer = None

    def reload(self):
        self.app.run_bg(lambda: procs.list_processes(sample_cpu=True), self._apply)

    def rerender(self):
        if self._items:
            self._apply(self._items)
        else:
            self.reload()

    def _apply(self, items):
        if isinstance(items, Exception):
            self.set_status(str(items), DANGER)
            return
        self._items = items
        rows, tags, sorts = [], {}, []
        for i, p in enumerate(items):
            rows.append((p, (p.name, p.pid, human_bytes(p.memory), f"{p.cpu:.1f}%",
                             p.username, p.exe)))
            sorts.append((p.name.lower(), p.pid, p.memory, p.cpu, p.username, p.exe))
            if p.protected:
                tags[i] = "dim"
            elif p.warn:
                tags[i] = "warn"
        self.table.set_rows(rows, tags, sort_values=sorts)
        total = sum(p.memory for p in items)
        self.show_pending_or(self.t("proc_count", n=len(items), m=human_bytes(total)))

    def kill(self):
        items = self.checked_or_selected()
        if not items:
            self.set_status(self.t("nothing_sel"), WARN)
            return
        blocked = [p for p in items if p.protected]
        if blocked:
            self.app.toast(self.t("proc_protected", n=blocked[0].name), "warn")
        targets = [p for p in items if not p.protected]
        if not targets:
            return
        ok = self.app.confirm(
            self.t("proc_kill_q", n=len(targets)),
            "\n".join(f"· {p.name}  (PID {p.pid}, {human_bytes(p.memory)})"
                      for p in targets[:12]),
            danger=True)
        if not ok:
            return

        def done(result):
            closed, failed = result
            self.reload_then(f"{closed} / {len(targets)} {self.t('v_ok')}"
                             + (f" · {', '.join(sorted(set(failed))[:3])}" if failed else ""),
                             WARN if failed else OK)

        # 정상 종료를 기다리는 데 최대 10초 - 화면 스레드에서 기다리면 '응답 없음'
        self.run_actions(lambda: procs.close_processes(targets), done)

    def open_location(self):
        item = self.target_or_warn()
        if item is not None and item.exe:
            open_in_explorer(item.exe)


# =====================================================================
# 서비스
# =====================================================================
class ServicePage(TablePage):
    key = "service"
    title_key = "svc_title"
    desc_key = "svc_desc"

    def __init__(self, app, **kw):
        super().__init__(app, **kw)
        self._all: list = []

    def make_columns(self):
        return [("display", self.t("col_name"), 230, "w"),
                ("name", "ID", 150, "w"),
                ("status", self.t("col_status"), 100, "center"),
                ("start", self.t("col_start"), 100, "center"),
                ("risk", self.t("col_risk"), 90, "center"),
                ("desc", self.t("col_desc"), 360, "w")]

    def build_toolbar(self, bar):
        self.add_button(bar, "btn_refresh", self.reload, "accent")
        self.add_button(bar, "btn_start", lambda: self.control("start"), "ghost", pad=(8, 0))
        self.add_button(bar, "btn_stop", lambda: self.control("stop"), "warn")
        self.add_button(bar, "svc_auto", lambda: self.set_type("auto"), "ghost",
                        width=90, pad=(14, 0))
        self.add_button(bar, "svc_manual", lambda: self.set_type("manual"), "ghost", width=90)
        self.add_button(bar, "svc_dis", lambda: self.set_type("disabled"), "danger", width=110)
        self.only_var = ctk.BooleanVar(value=True)
        self.only_box = ctk.CTkCheckBox(bar, text="", variable=self.only_var, width=150,
                                        font=ctk.CTkFont(size=11), text_color=TXT_DIM,
                                        fg_color=ACCENT, hover_color=ACCENT,
                                        checkmark_color="black", command=self._render)
        self.only_box.pack(side="left", padx=(14, 0))
        self._sync_filter_label()

    def _sync_filter_label(self):
        self.only_box.configure(text=self.t("svc_filter"))

    def on_show(self):
        if not self._all:
            self.reload()

    def reload(self):
        self.set_status("...")
        self.app.run_bg(services.list_services, self._apply)

    def rerender(self):
        if self._all:
            self._render()
        else:
            self.reload()

    def _apply(self, items):
        if isinstance(items, Exception):
            self.set_status(str(items), DANGER)
            return
        self._all = items
        self._render()

    def _render(self):
        show_all = not self.only_var.get()
        items = [s for s in self._all if show_all or s.risk in (services.RISK_SAFE,
                                                               services.RISK_CARE)]
        risk_label = {services.RISK_SAFE: self.t("svc_risk_safe"),
                      services.RISK_CARE: self.t("svc_risk_care"),
                      services.RISK_CORE: self.t("svc_risk_core"), "": ""}
        start_label = {"auto": self.t("svc_auto"), "manual": self.t("svc_manual"),
                       "disabled": self.t("svc_dis")}
        rows, tags = [], {}
        for i, s in enumerate(items):
            note = s.note_ko if self.lang == "KO" else s.note_en
            rows.append((s, (s.display, s.name,
                             self.t("st_running") if s.running else self.t("st_stopped"),
                             start_label.get(s.start_type, s.start_type),
                             risk_label.get(s.risk, ""), note)))
            if s.protected:
                tags[i] = "dim"
            elif s.risk == services.RISK_SAFE:
                tags[i] = "good"
            elif s.risk == services.RISK_CARE:
                tags[i] = "warn"
        self.table.set_rows(rows, tags)
        running = sum(1 for s in self._all if s.running)
        self.show_pending_or(self.t("svc_count", n=len(self._all), r=running))

    def control(self, action: str):
        items = self.checked_or_selected()
        if not items:
            self.set_status(self.t("nothing_sel"), WARN)
            return
        if not self.app.require_admin():
            return

        def work():
            runner = Runner()
            return [services.control(runner, s, action) for s in items]

        self.run_actions(work, self._report)

    def _report(self, results):
        fails = [d for good, d in results if not good]
        ok = len(results) - len(fails)
        self.reload_then(f"{ok} {self.t('v_ok')}"
                         + (f" / {len(fails)} {self.t('v_failed')} - {fails[-1]}"
                            if fails else ""),
                         WARN if fails else OK)

    def set_type(self, start: str):
        items = self.checked_or_selected()
        if not items:
            self.set_status(self.t("nothing_sel"), WARN)
            return
        if not self.app.require_admin():
            return
        label = {"auto": self.t("svc_auto"), "manual": self.t("svc_manual"),
                 "disabled": self.t("svc_dis")}[start]
        ok = self.app.confirm(
            self.t("svc_set_q", n=len(items), m=label),
            "\n".join(f"· {s.display}" for s in items[:12]),
            warn=self.t("backup_note", p=BACKUP_DIR), danger=(start == "disabled"))
        if not ok:
            return
        services.backup_services(self._all)   # 취소했을 때 백업 파일만 쌓이지 않도록 확인 뒤에

        def work():
            runner = Runner()
            return [services.set_start_type(runner, s, start) for s in items]

        self.run_actions(work, self._report)

    def retranslate(self):
        self._sync_filter_label()
        super().retranslate()


# =====================================================================
# 설치된 프로그램
# =====================================================================
class ProgramsPage(TablePage):
    key = "programs"
    title_key = "prog_title"
    desc_key = "prog_desc"
    checkable = False

    def __init__(self, app, **kw):
        super().__init__(app, **kw)
        self._all: list = []

    def make_columns(self):
        return [("name", self.t("col_name"), 300, "w"),
                ("size", self.t("col_size"), 110, "e"),
                ("ver", self.t("col_ver"), 130, "center"),
                ("pub", self.t("col_pub"), 200, "w"),
                ("date", self.t("col_date"), 110, "center")]

    def build_toolbar(self, bar):
        self.add_button(bar, "btn_refresh", self.reload, "accent")
        self.add_button(bar, "btn_uninstall", self.uninstall, "danger", width=120, pad=(8, 0))
        self.add_button(bar, "btn_open", self.open_location, "ghost", width=130, pad=(8, 0))
        self.search = ctk.CTkEntry(bar, width=200, height=32,
                                   placeholder_text=self.t("prog_search"),
                                   fg_color="#1a1a1a", border_color="#3d3d3d")
        self.search.pack(side="left", padx=(14, 0))
        self.search.bind("<KeyRelease>", lambda _e: self._render())

    def on_show(self):
        if not self._all:
            self.reload()

    def reload(self):
        self.set_status("...")
        self.app.run_bg(programs.list_programs, self._apply)

    def rerender(self):
        self.search.configure(placeholder_text=self.t("prog_search"))
        if self._all:
            self._render()
        else:
            self.reload()

    def _apply(self, items):
        if isinstance(items, Exception):
            self.set_status(str(items), DANGER)
            return
        self._all = items
        self._render()

    def _render(self):
        query = (self.search.get() or "").strip().lower()
        items = [p for p in self._all
                 if not query or query in p.name.lower() or query in p.publisher.lower()]
        rows, sorts = [], []
        for p in items:
            rows.append((p, (p.name, human_bytes(p.size_bytes) if p.size_bytes else "-",
                             p.version, p.publisher, p.install_date)))
            sorts.append((p.name.lower(), p.size_bytes, p.version,
                          p.publisher.lower(), p.install_date))
        self.table.set_rows(rows, sort_values=sorts)
        self.set_status(self.t("prog_count", n=len(items),
                               s=human_bytes(programs.total_size(items))))

    def uninstall(self):
        prog = self.table.focused_payload()
        if prog is None:
            self.set_status(self.t("nothing_sel"), WARN)
            return
        if not prog.removable:
            self.set_status(self.t("prog_no_cmd"), WARN)
            return
        ok = self.app.confirm(self.t("prog_run_q", n=prog.name),
                              f"{prog.publisher}  {prog.version}\n"
                              f"{human_bytes(prog.size_bytes)}", danger=True)
        if not ok:
            return
        good, detail = programs.uninstall(prog)
        if good:
            self.set_status(self.t("prog_started", n=prog.name), OK)
        else:
            self.set_status(detail, DANGER)

    def open_location(self):
        prog = self.target_or_warn()
        if prog is not None and not programs.open_location(prog):
            self.set_status(f"{prog.name}: {self.t('v_notfound')}", WARN)

    def on_double(self, payload):
        self.uninstall()


# =====================================================================
# 파일 검사 공통 (대용량 / 중복)
# =====================================================================
class _FileScanPage(TablePage):
    def build_scan_controls(self, bar, with_size=True):
        self.add_button(bar, "btn_scan", self.start_scan, "accent", width=110)
        self.add_button(bar, "btn_delete", self.delete_checked, "danger",
                        width=140, pad=(8, 0))
        self.add_button(bar, "btn_open", self.open_location, "ghost", width=120, pad=(8, 0))

        self.folder_var = ctk.StringVar(
            value=self.app.cfg["scan_root"] or str(Path.home()))
        self.folder_entry = ctk.CTkEntry(bar, textvariable=self.folder_var, width=250,
                                         height=32, fg_color="#1a1a1a",
                                         border_color="#3d3d3d")
        self.folder_entry.pack(side="left", padx=(14, 0))
        self.btn_browse = toolbar_button(bar, self.t("big_browse"), self.browse, "ghost",
                                         width=90)
        self.btn_browse.pack(side="left", padx=(6, 0))
        self._buttons.append((self.btn_browse, "big_browse"))
        if with_size:
            self.size_var = ctk.StringVar(value=str(self.app.cfg["large_file_min_mb"]))
            ctk.CTkEntry(bar, textvariable=self.size_var, width=70, height=32,
                         fg_color="#1a1a1a", border_color="#3d3d3d"
                         ).pack(side="left", padx=(10, 0))
            ctk.CTkLabel(bar, text="MB", font=ctk.CTkFont(size=11),
                         text_color=TXT_DIM).pack(side="left", padx=(4, 0))

    def browse(self):
        chosen = filedialog.askdirectory(initialdir=self.folder_var.get() or str(Path.home()))
        if chosen:
            self.folder_var.set(chosen)
            self.app.cfg["scan_root"] = chosen

    def root_path(self) -> Path | None:
        raw = (self.folder_var.get() or "").strip()
        if not raw or not os.path.isdir(raw):
            self.set_status(self.t("v_notfound"), WARN)
            return None
        self.app.cfg["scan_root"] = raw
        return Path(raw)

    def start_scan(self):
        raise NotImplementedError

    def rerender(self):
        pass                     # 검사 결과 목록은 언어와 무관

    def delete_checked(self):
        # 체크만 보던 탓에 줄을 클릭하고 삭제를 누르면 '선택한 항목이 없습니다'가 나왔다
        items = self.checked_or_selected()
        paths = [self._path_of(i) for i in items]
        paths = [p for p in paths if p]
        if not paths:
            self.set_status(self.t("nothing_sel"), WARN)
            return
        total = sum(self._size_of(i) for i in items)
        ok = self.app.confirm(self.t("del_confirm", n=len(paths), s=human_bytes(total)),
                              "\n".join(f"· {p}" for p in paths[:12]), danger=True)
        if not ok:
            return
        moved, failed = send_to_recycle_bin(paths)
        self.set_status(self.t("del_done", n=moved)
                        + (f" / {len(failed)} {self.t('v_failed')}" if failed else ""),
                        WARN if failed else OK)
        self.after(200, self.rescan_after_delete)

    def rescan_after_delete(self):
        pass

    def open_location(self):
        item = self.target_or_warn()
        path = self._path_of(item) if item is not None else None
        if path:
            open_in_explorer(path)

    def _path_of(self, item):
        return getattr(item, "path", None)

    def _size_of(self, item):
        return getattr(item, "size", 0)


# =====================================================================
# 대용량 파일
# =====================================================================
class BigFilesPage(_FileScanPage):
    key = "bigfiles"
    title_key = "big_title"
    desc_key = "big_desc"

    def make_columns(self):
        return [("path", self.t("col_path"), 620, "w"),
                ("size", self.t("col_size"), 110, "e"),
                ("mtime", self.t("col_modified"), 140, "center")]

    def build_toolbar(self, bar):
        self.build_scan_controls(bar, with_size=True)

    def start_scan(self):
        root = self.root_path()
        if root is None:
            return
        try:
            min_mb = max(1, int(float(self.size_var.get())))
        except ValueError:
            min_mb = 100
        self.app.cfg["large_file_min_mb"] = min_mb
        self._scan_dir, self._min_bytes = root, min_mb * 1024 * 1024
        self.app.run_task(self._scan_task, page=self, status_key="busy")

    def _scan_task(self, ctx):
        ctx.begin(1)
        ctx.log(f">> {self._scan_dir}  (>= {human_bytes(self._min_bytes)})")

        def progress(cur, found):
            ctx.check()

        hits = scan.find_large_files(self._scan_dir, self._min_bytes, ctx.cancel,
                                     limit=800, progress=progress)
        ctx.progress(1.0)
        ctx.app.ui(self._apply, hits)

    def _apply(self, hits):
        rows, sorts = [], []
        for h in hits:
            rows.append((h, (h.path, human_bytes(h.size),
                             datetime.fromtimestamp(h.mtime).strftime("%Y-%m-%d"))))
            sorts.append((h.path.lower(), h.size, h.mtime))
        self.table.set_rows(rows, sort_values=sorts)
        self.set_status(self.t("big_found", n=len(hits),
                               s=human_bytes(sum(h.size for h in hits))))

    def rescan_after_delete(self):
        self.start_scan()


# =====================================================================
# 중복 파일
# =====================================================================
class DuplicatesPage(_FileScanPage):
    key = "dupes"
    title_key = "dup_title"
    desc_key = "dup_desc"

    def make_columns(self):
        return [("group", "#", 60, "center"),
                ("path", self.t("col_path"), 580, "w"),
                ("size", self.t("col_size"), 110, "e")]

    def build_toolbar(self, bar):
        self.build_scan_controls(bar, with_size=True)

    def start_scan(self):
        root = self.root_path()
        if root is None:
            return
        try:
            min_mb = max(1, int(float(self.size_var.get())))
        except ValueError:
            min_mb = 1
        self._scan_dir, self._min_bytes = root, min_mb * 1024 * 1024
        self.app.run_task(self._scan_task, page=self, status_key="busy")

    def _scan_task(self, ctx):
        ctx.begin(1)
        ctx.log(f">> {self._scan_dir}  (>= {human_bytes(self._min_bytes)})")

        def progress(stage, done, total):
            ctx.check()
            if total:
                ctx.progress(min(0.95, done / total))

        groups = scan.find_duplicates(self._scan_dir, self._min_bytes, ctx.cancel, progress)
        ctx.progress(1.0)
        ctx.app.ui(self._apply, groups)

    def _apply(self, groups):
        rows, tags = [], {}
        i = 0
        extra_files = extra_bytes = 0
        for gi, group in enumerate(groups, start=1):
            for pi, path in enumerate(group.paths):
                hit = scan.FileHit(path, group.size, 0.0)
                rows.append((hit, (gi, path, human_bytes(group.size))))
                if pi == 0:
                    tags[i] = "good"     # 그룹의 첫 파일은 남겨둔다
                i += 1
            extra_files += len(group.paths) - 1
            extra_bytes += group.wasted
        self.table.set_rows(rows, tags)
        # 각 그룹의 첫 파일만 빼고 자동 체크
        self._auto_check(groups)
        self.set_status(self.t("dup_found", g=len(groups), n=extra_files,
                               s=human_bytes(extra_bytes)))

    def _auto_check(self, groups):
        keep = {g.paths[0] for g in groups}
        self.table.check_where(lambda payload: payload is not None
                               and payload.path not in keep)

    def rescan_after_delete(self):
        self.start_scan()


# =====================================================================
# 디스크 상태
# =====================================================================
class DiskPage(TablePage):
    key = "disk"
    title_key = "disk_title"
    desc_key = "disk_desc"
    checkable = False

    def __init__(self, app, **kw):
        super().__init__(app, **kw)
        # 카드(물리 디스크) 를 표(볼륨) 위로 올린다
        self.grid_rowconfigure(2, weight=0)
        self.grid_rowconfigure(3, weight=1)
        self._cards_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._cards_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self.table.grid_configure(row=3)
        self.status.grid_configure(row=4)
        self._cards = []
        self._last = None

    def make_columns(self):
        return [("letter", self.t("col_name"), 80, "center"),
                ("label", "Label", 170, "w"),
                ("fs", self.t("col_type"), 80, "center"),
                ("total", self.t("col_total"), 130, "e"),
                ("used", self.t("col_used"), 130, "e"),
                ("free", self.t("col_free"), 130, "e"),
                ("pct", self.t("col_free") + " %", 90, "center")]

    def build_toolbar(self, bar):
        self.add_button(bar, "btn_refresh", self.reload, "accent")

    def on_show(self):
        if not self._cards:
            self.reload()

    def reload(self):
        self.app.run_task(self._task, page=self, status_key="busy")

    def rerender(self):
        # 언어만 바꿨는데 디스크를 다시 조회하면, 다른 작업 중일 때 '진행 중' 경고가 뜬다
        if self._last is not None:
            self._apply(*self._last)

    def _task(self, ctx):
        ctx.begin(2)
        ctx.step("Disks")
        disks = sysinfo.disk_info(ctx.runner)
        ctx.step("Volumes")
        vols = sysinfo.volume_info(ctx.runner)
        ctx.app.ui(self._apply, disks, vols)

    def _apply(self, disks, vols):
        self._last = (disks, vols)
        for card in self._cards:
            card.destroy()
        self._cards = []
        for i, d in enumerate(disks):
            self._cards_frame.grid_columnconfigure(i, weight=1)
            bits = [d.media or "?", human_bytes(d.size)]
            if d.temperature is not None:
                bits.append(f"{d.temperature}°C")
            if d.wear is not None:
                bits.append(f"{self.t('disk_wear')} {d.wear}%")
            if d.power_on_hours is not None:
                bits.append(f"{d.power_on_hours:,}h")
            card = StatCard(self._cards_frame, title=d.name[:34],
                            value=d.health or "?", sub="  ·  ".join(bits),
                            accent=OK if d.healthy else DANGER, height=104, width=240)
            card.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 10, 0))
            self._cards.append(card)

        rows, tags, sorts = [], {}, []
        for i, v in enumerate(vols):
            rows.append((v, (f"{v.letter}:", v.label, v.fs, human_bytes(v.size),
                             human_bytes(v.used), human_bytes(v.free),
                             f"{v.percent_free}%")))
            sorts.append((v.letter, v.label.lower(), v.fs, v.size, v.used, v.free,
                          v.percent_free))
            if v.percent_free < 10:
                tags[i] = "danger"
            elif v.percent_free < 20:
                tags[i] = "warn"
        self.table.set_rows(rows, tags, sort_values=sorts)

        note = ""
        if not is_admin() and disks and all(d.temperature is None for d in disks):
            note = self.t("disk_smart_admin")
        self.set_status(note, WARN if note else TXT_DIM)
