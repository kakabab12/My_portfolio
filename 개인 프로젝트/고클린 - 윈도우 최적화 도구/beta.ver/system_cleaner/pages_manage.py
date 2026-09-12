# -*- coding: utf-8 -*-
"""목록형 화면: 시작 프로그램, 프로세스, 서비스, 프로그램 삭제, 파일 검사, 디스크."""

from __future__ import annotations

import os
import threading
from datetime import datetime
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

from . import procs, programs, scan, services, startup, sysinfo
from .core import (Runner, human_bytes, is_admin, open_in_explorer,
                   send_to_recycle_bin)
from .ui import (ACCENT, BORDER, CARD, DANGER, OK, TXT, TXT_DIM, WARN,
                 DataTable, Page, StatCard, toolbar_button)


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

    def retranslate(self):
        super().retranslate()
        for btn, key in self._buttons:
            btn.configure(text=self.t(key))
        self.reload()


# =====================================================================
# 시작 프로그램
# =====================================================================
class StartupPage(TablePage):
    key = "startup"
    title_key = "startup_title"
    desc_key = "startup_desc"

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

    def reload(self):
        # 작업 스케줄러 조회에 PowerShell 이 필요해 UI 스레드에서 하면 몇 초 멈춘다
        self.set_status("...")
        self.app.run_bg(lambda: startup.list_startup(Runner()), self._apply)

    def _apply(self, entries):
        if isinstance(entries, Exception):
            self.set_status(str(entries), DANGER)
            return
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
        self.set_status(self.t("startup_count", n=len(entries), e=enabled))

    def apply(self, enable: bool):
        items = self.checked_or_selected()
        if not items:
            self.set_status(self.t("nothing_sel"), WARN)
            return
        if any(e.needs_admin for e in items) and not self.app.require_admin():
            return
        ok = fail = 0
        problems = []
        for e in items:
            good, detail = startup.set_enabled(e, enable, self.app.shared_runner)
            if good:
                ok += 1
            else:
                fail += 1
                problems.append(f"{e.name}: {detail}")
        self.reload()
        msg = f"{ok} {self.t('v_ok')}" + (f" / {fail} {self.t('v_failed')}" if fail else "")
        self.set_status(msg, WARN if fail else OK)
        if problems:
            self.app.toast(problems[0], "warn")

    def delete(self):
        items = [e for e in self.checked_or_selected() if e.kind != "task"]
        if not items:
            self.set_status(self.t("nothing_sel"), WARN)
            return
        if any(e.needs_admin for e in items) and not self.app.require_admin():
            return
        backup = startup.backup_entries(items)
        ok = self.app.confirm(
            self.t("startup_del_q", n=len(items)),
            "\n".join(f"· {e.name}  [{e.location}]" for e in items[:12]),
            warn=self.t("startup_backup", p=backup) if backup else None,
            danger=True)
        if not ok:
            return
        done = 0
        for e in items:
            good, _ = startup.delete_entry(e)
            done += 1 if good else 0
        self.reload()
        self.set_status(f"{done} / {len(items)} {self.t('v_ok')}", OK)

    def open_location(self):
        entry = self.table.focused_payload()
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

    def _apply(self, items):
        if isinstance(items, Exception):
            self.set_status(str(items), DANGER)
            return
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
        self.set_status(self.t("proc_count", n=len(items), m=human_bytes(total)))

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
        closed, failed = procs.close_processes(targets)
        self.reload()
        self.set_status(f"{closed} / {len(targets)} {self.t('v_ok')}"
                        + (f" · {', '.join(sorted(set(failed))[:3])}" if failed else ""),
                        WARN if failed else OK)

    def open_location(self):
        item = self.table.focused_payload()
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
        self.only_box.configure(text="권장 항목만" if self.lang == "KO" else "Optional only")

    def on_show(self):
        if not self._all:
            self.reload()

    def reload(self):
        self.set_status("...")
        self.app.run_bg(services.list_services, self._apply)

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
        self.set_status(self.t("svc_count", n=len(self._all), r=running))

    def control(self, action: str):
        items = self.checked_or_selected()
        if not items:
            self.set_status(self.t("nothing_sel"), WARN)
            return
        if not self.app.require_admin():
            return
        ok = fail = 0
        last = ""
        for s in items:
            good, detail = services.control(self.app.shared_runner, s, action)
            if good:
                ok += 1
            else:
                fail += 1
                last = detail
        self.reload()
        self.set_status(f"{ok} {self.t('v_ok')}"
                        + (f" / {fail} {self.t('v_failed')} - {last}" if fail else ""),
                        WARN if fail else OK)

    def set_type(self, start: str):
        items = self.checked_or_selected()
        if not items:
            self.set_status(self.t("nothing_sel"), WARN)
            return
        if not self.app.require_admin():
            return
        label = {"auto": self.t("svc_auto"), "manual": self.t("svc_manual"),
                 "disabled": self.t("svc_dis")}[start]
        services.backup_services(self._all)
        ok = self.app.confirm(
            self.t("svc_set_q", n=len(items), m=label),
            "\n".join(f"· {s.display}" for s in items[:12]),
            danger=(start == "disabled"))
        if not ok:
            return
        good_n = fail_n = 0
        last = ""
        for s in items:
            good, detail = services.set_start_type(self.app.shared_runner, s, start)
            if good:
                good_n += 1
            else:
                fail_n += 1
                last = detail
        self.reload()
        self.set_status(f"{good_n} {self.t('v_ok')}"
                        + (f" / {fail_n} {self.t('v_failed')} - {last}" if fail_n else ""),
                        WARN if fail_n else OK)

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
        self.search = ctk.CTkEntry(bar, width=200, height=32, placeholder_text="search...",
                                   fg_color="#1a1a1a", border_color="#3d3d3d")
        self.search.pack(side="left", padx=(14, 0))
        self.search.bind("<KeyRelease>", lambda _e: self._render())

    def on_show(self):
        if not self._all:
            self.reload()

    def reload(self):
        self.set_status("...")
        self.app.run_bg(programs.list_programs, self._apply)

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
            self.app.toast("제거 명령이 등록되어 있지 않습니다", "warn")
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
        prog = self.table.focused_payload()
        if prog is not None and not programs.open_location(prog):
            self.app.toast(self.t("v_notfound"), "warn")

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
        toolbar_button(bar, self.t("big_browse"), self.browse, "ghost",
                       width=90).pack(side="left", padx=(6, 0))
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

    def delete_checked(self):
        items = self.table.checked_payloads()
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
        item = self.table.focused_payload()
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
        self._root, self._min_bytes = root, min_mb * 1024 * 1024
        self.app.run_task(self._scan_task, page=self, status_key="busy")

    def _scan_task(self, ctx):
        ctx.begin(1)
        ctx.log(f">> {self._root}  (>= {human_bytes(self._min_bytes)})")

        def progress(cur, found):
            ctx.check()

        hits = scan.find_large_files(self._root, self._min_bytes, ctx.cancel,
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
        self._root, self._min_bytes = root, min_mb * 1024 * 1024
        self.app.run_task(self._scan_task, page=self, status_key="busy")

    def _scan_task(self, ctx):
        ctx.begin(1)
        ctx.log(f">> {self._root}  (>= {human_bytes(self._min_bytes)})")

        def progress(stage, done, total):
            ctx.check()
            if total:
                ctx.progress(min(0.95, done / total))

        groups = scan.find_duplicates(self._root, self._min_bytes, ctx.cancel, progress)
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

    def _task(self, ctx):
        ctx.begin(2)
        ctx.step("Disks")
        disks = sysinfo.disk_info(ctx.runner)
        ctx.step("Volumes")
        vols = sysinfo.volume_info(ctx.runner)
        ctx.app.ui(self._apply, disks, vols)

    def _apply(self, disks, vols):
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
            note = ("S.M.A.R.T. 상세 값(온도·마모도)은 관리자 권한이 필요합니다."
                    if self.lang == "KO"
                    else "Detailed S.M.A.R.T. values need administrator rights.")
        self.set_status(note, WARN if note else TXT_DIM)
