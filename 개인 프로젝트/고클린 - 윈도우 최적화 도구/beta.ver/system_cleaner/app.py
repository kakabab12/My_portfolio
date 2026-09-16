# -*- coding: utf-8 -*-
"""메인 창: 네비게이션, 작업 실행기, 상태 표시줄, 설정/정보."""

from __future__ import annotations

import sys
import threading
import time
import traceback
import webbrowser
from datetime import datetime
from pathlib import Path

import customtkinter as ctk

from . import (APP_NAME, APP_VERSION, pages_action, pages_manage, pages_tweaks,
               tasks)
from .core import (CONFIG_PATH, ICON_PATH, LOG_DIR, Config, Runner, is_admin, log_error)
from .i18n import t
from .ui import (ACCENT, ACCENT_DIM, BG, BORDER, DANGER, OK, SIDEBAR, TXT, TXT_DIM, WARN,
                 ConfirmDialog, apply_table_style)

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# (그룹 키, [(페이지 키, 문자열 키, 아이콘)])
NAV = [
    ("nav_group_status",   [("dashboard", "nav_dashboard", "▦")]),
    ("nav_group_clean",    [("clean", "nav_clean", "✦"),
                            ("memory", "nav_memory", "▤"),
                            ("game", "nav_game", "▶")]),
    ("nav_group_manage",   [("startup", "nav_startup", "◉"),
                            ("process", "nav_process", "☰"),
                            ("service", "nav_service", "⚙"),
                            ("programs", "nav_programs", "▣")]),
    ("nav_group_scan",     [("bigfiles", "nav_bigfiles", "◼"),
                            ("dupes", "nav_dupes", "⧉"),
                            ("disk", "nav_disk", "◱")]),
    ("nav_group_etc",      [("tweaks", "nav_tweaks", "✎"),
                            ("network", "nav_network", "⇅"),
                            ("update", "nav_update", "⟳"),
                            ("info", "nav_info", "ⓘ")]),
]

PAGE_CLASSES = {
    "dashboard": pages_action.DashboardPage,
    "clean":     pages_action.CleanPage,
    "memory":    pages_action.MemoryPage,
    "game":      pages_action.GamePage,
    "network":   pages_action.NetworkPage,
    "update":    pages_action.UpdatePage,
    "info":      pages_action.InfoPage,
    "startup":   pages_manage.StartupPage,
    "process":   pages_manage.ProcessPage,
    "service":   pages_manage.ServicePage,
    "programs":  pages_manage.ProgramsPage,
    "bigfiles":  pages_manage.BigFilesPage,
    "dupes":     pages_manage.DuplicatesPage,
    "disk":      pages_manage.DiskPage,
    "tweaks":      pages_tweaks.TweaksPage,
}

# 관리자 권한이 없으면 막는 화면
ADMIN_PAGES = {"clean", "service"}


class SystemCleanerApp(ctk.CTk):
    def __init__(self, cfg: Config, admin: bool):
        super().__init__()
        self.cfg = cfg
        self.admin = admin
        self.lang = cfg["lang"]
        self.shared_runner = Runner()

        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.active_ctx: tasks.TaskContext | None = None
        self.active_page = None
        self.current_key = ""
        self._started_at = 0.0
        self._timer_job = None
        self._toast_job = None

        self.title(f"{APP_NAME} [{APP_VERSION}]")
        self.geometry("1240x800")
        self.minsize(1080, 680)
        self.configure(fg_color=BG)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        if ICON_PATH.exists():
            try:
                self.iconbitmap(str(ICON_PATH))   # 호출해야 CTk 기본 아이콘으로 덮이지 않는다
            except Exception:
                pass
        apply_table_style()
        self._build_sidebar()
        self._build_content()

        self.pages: dict[str, object] = {}
        self.show_page("dashboard")

        self.bind("<Escape>", lambda _e: self.request_cancel())
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        if not self.admin:
            self.after(400, lambda: self.toast(t("limited_warn", self.lang), "warn", 8000))

    # =================================================================
    # UI 구성
    # =================================================================
    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=238, corner_radius=0, fg_color=SIDEBAR)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_rowconfigure(1, weight=1)

        head = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", pady=(16, 4))
        # 두 줄 로고는 세로 공간을 많이 먹어 네비 항목이 그만큼 가려진다
        ctk.CTkLabel(head, text="SYSTEM CLEANER",
                     font=ctk.CTkFont(family="Arial", size=17, weight="bold"),
                     text_color=ACCENT).pack(padx=14)
        self.ver_label = ctk.CTkLabel(head, text="", font=ctk.CTkFont(size=10),
                                      text_color=TXT_DIM)
        self.ver_label.pack(padx=14, pady=(1, 0))

        self.nav_frame = ctk.CTkScrollableFrame(self.sidebar, fg_color="transparent",
                                                scrollbar_button_color="#2a2a2a")
        self.nav_frame.grid(row=1, column=0, sticky="nsew", padx=4)

        self.nav_buttons: dict[str, ctk.CTkButton] = {}
        self.nav_labels: list[tuple[ctk.CTkLabel, str]] = []
        self.nav_keys: dict[str, str] = {}
        for group_key, entries in NAV:
            label = ctk.CTkLabel(self.nav_frame, text=t(group_key, self.lang),
                                 font=ctk.CTkFont(size=10, weight="bold"),
                                 text_color="#5f5f5f", anchor="w")
            label.pack(fill="x", padx=14, pady=(10, 1))
            self.nav_labels.append((label, group_key))
            for page_key, text_key, icon in entries:
                btn = ctk.CTkButton(
                    self.nav_frame, text=f"  {icon}   {t(text_key, self.lang)}",
                    command=lambda k=page_key: self.show_page(k),
                    font=ctk.CTkFont(size=13), height=33, corner_radius=6,
                    fg_color="transparent", text_color="#C8C8C8",
                    hover_color="#242424", anchor="w")
                btn.pack(fill="x", padx=8, pady=1)
                self.nav_buttons[page_key] = btn
                self.nav_keys[page_key] = (text_key, icon)

        bottom = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        bottom.grid(row=2, column=0, sticky="ew", padx=16, pady=12)
        self.btn_about = ctk.CTkButton(
            bottom, text="?", width=32, height=32, corner_radius=16,
            fg_color="transparent", border_width=1, border_color="#4a4a4a",
            text_color=TXT_DIM, hover_color="#2a2a2a",
            font=ctk.CTkFont(size=14, weight="bold"), command=self.show_about)
        self.btn_about.pack(side="left", padx=(0, 8))
        self.btn_gear = ctk.CTkButton(
            bottom, text="⚙", width=32, height=32, corner_radius=16,
            fg_color="transparent", border_width=1, border_color="#4a4a4a",
            text_color=TXT_DIM, hover_color="#2a2a2a",
            font=ctk.CTkFont(size=15), command=self.show_settings)
        self.btn_gear.pack(side="left", padx=(0, 8))
        self.btn_lang = ctk.CTkButton(
            bottom, text=t("btn_lang", self.lang), width=52, height=32, corner_radius=6,
            fg_color="#242424", text_color=TXT, hover_color="#383838",
            font=ctk.CTkFont(size=12, weight="bold"), command=self.toggle_language)
        self.btn_lang.pack(side="left")

        self._sync_badge()

    def _build_content(self):
        wrapper = ctk.CTkFrame(self, fg_color="transparent")
        wrapper.grid(row=0, column=1, sticky="nsew", padx=22, pady=20)
        wrapper.grid_columnconfigure(0, weight=1)
        wrapper.grid_rowconfigure(0, weight=1)

        self.content = ctk.CTkFrame(wrapper, fg_color="transparent")
        self.content.grid(row=0, column=0, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)

        # --- 상태 표시줄 ---
        bar = ctk.CTkFrame(wrapper, fg_color="#151515", corner_radius=8,
                           border_width=1, border_color=BORDER, height=54)
        bar.grid(row=1, column=0, sticky="ew", pady=(14, 0))
        bar.grid_propagate(False)
        bar.grid_columnconfigure(1, weight=1)

        self.status_label = ctk.CTkLabel(bar, text=t("ready", self.lang),
                                         font=ctk.CTkFont(size=13, weight="bold"),
                                         text_color=TXT, anchor="w", width=300)
        self.status_label.grid(row=0, column=0, sticky="w", padx=(16, 10), pady=16)

        self.progressbar = ctk.CTkProgressBar(bar, height=6, corner_radius=3,
                                              progress_color=ACCENT, fg_color="#2a2a2a")
        self.progressbar.grid(row=0, column=1, sticky="ew", padx=10)
        self.progressbar.set(0)

        self.pct_label = ctk.CTkLabel(bar, text="0%", width=44,
                                      font=ctk.CTkFont(size=11), text_color=TXT_DIM)
        self.pct_label.grid(row=0, column=2, padx=(6, 4))

        self.elapsed_label = ctk.CTkLabel(bar, text="", width=110,
                                          font=ctk.CTkFont(size=11), text_color=TXT_DIM)
        self.elapsed_label.grid(row=0, column=3, padx=(0, 6))

        self.btn_cancel = ctk.CTkButton(
            bar, text=t("btn_cancel", self.lang), command=self.request_cancel,
            width=130, height=30, corner_radius=6,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#3a1c1c", text_color="#FF8A80",
            border_width=1, border_color="#7a2b2b", hover_color="#4d2424")
        self.btn_cancel.grid(row=0, column=4, padx=(0, 14))
        self.btn_cancel.grid_remove()

    def _sync_badge(self):
        badge = t("admin_ok", self.lang) if self.admin else t("admin_no", self.lang)
        self.ver_label.configure(text=f"{APP_VERSION}  ·  {badge}",
                                 text_color=TXT_DIM if self.admin else WARN)

    # =================================================================
    # 네비게이션
    # =================================================================
    def show_page(self, key: str):
        if key == self.current_key:
            return
        if key not in PAGE_CLASSES:
            return

        old = self.pages.get(self.current_key)
        if old is not None:
            old.on_hide()
            old.grid_forget()

        page = self.pages.get(key)
        if page is None:
            page = PAGE_CLASSES[key](self)
            self.pages[key] = page
        page.grid(row=0, column=0, sticky="nsew")
        self.current_key = key
        self._highlight_nav(key)
        page.on_show()

        if key in ADMIN_PAGES and not self.admin:
            self.toast(t("need_admin", self.lang), "warn")

    def _highlight_nav(self, key: str):
        for page_key, btn in self.nav_buttons.items():
            if page_key == key:
                btn.configure(fg_color="#00323b", text_color=ACCENT)
            else:
                btn.configure(fg_color="transparent", text_color="#C8C8C8")

    # =================================================================
    # 스레드 -> UI
    # =================================================================
    def ui(self, fn, *args):
        """워커 스레드가 UI 를 만지는 유일한 통로 (tkinter 는 스레드 세이프하지 않다)."""
        try:
            self.after(0, lambda: fn(*args))
        except RuntimeError:
            pass   # 창이 이미 닫힌 경우

    def run_bg(self, fn, on_done=None):
        """상태 표시줄을 건드리지 않는 가벼운 백그라운드 작업 (목록 새로고침 등)."""
        def worker():
            try:
                result = fn()
            except Exception as e:  # noqa: BLE001
                log_error(f"bg:{getattr(fn, '__name__', '?')}", e)
                result = e
            if on_done is not None:
                self.ui(on_done, result)
        threading.Thread(target=worker, daemon=True).start()

    # =================================================================
    # 로그 (활성 페이지의 터미널로 보낸다)
    # =================================================================
    def _terminal(self):
        page = self.active_page or self.pages.get(self.current_key)
        return getattr(page, "terminal", None)

    def log(self, message, tag="info"):
        term = self._terminal()
        if term is not None:
            term.log(message, tag)

    def log_summary(self, title, rows):
        term = self._terminal()
        if term is not None:
            term.log_summary(title, rows)

    def save_log(self):
        term = self._terminal()
        if term is None:
            return
        try:
            folder = Path.home() / "Documents" / "SystemCleaner"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"log_{datetime.now():%Y%m%d_%H%M%S}.txt"
            path.write_text(term.text(), encoding="utf-8")
            self.toast(t("log_saved", self.lang, path=str(path)), "ok", 6000)
        except OSError as e:
            self.toast(str(e), "err")

    # =================================================================
    # 상태 표시줄
    # =================================================================
    def set_progress(self, value: float):
        self.progressbar.set(value)
        self.pct_label.configure(text=f"{int(value * 100)}%")

    def set_status(self, text: str, color: str = TXT):
        self.status_label.configure(text=text, text_color=color)

    def toast(self, message: str, kind: str = "info", duration: int = 4500):
        colors = {"ok": OK, "warn": WARN, "err": DANGER, "info": TXT}
        self.set_status(str(message)[:120], colors.get(kind, TXT))
        if self._toast_job:
            self.after_cancel(self._toast_job)
        self._toast_job = self.after(duration, self._clear_toast)

    def _clear_toast(self):
        self._toast_job = None
        if not (self.worker and self.worker.is_alive()):
            self.set_status(t("ready", self.lang), TXT)

    def _tick(self):
        if self.worker and self.worker.is_alive():
            self.elapsed_label.configure(
                text=t("elapsed", self.lang, sec=time.time() - self._started_at))
            self._timer_job = self.after(200, self._tick)

    # =================================================================
    # 작업 실행
    # =================================================================
    def run_task(self, fn, page=None, status_key: str = "busy"):
        if self.worker and self.worker.is_alive():
            self.toast(t("busy", self.lang), "warn")
            return

        self.active_page = page or self.pages.get(self.current_key)
        term = self._terminal()
        if term is not None:
            term.clear()

        self.cancel_event = threading.Event()
        ctx = tasks.TaskContext(self, self.cancel_event)
        self.active_ctx = ctx

        self.set_status(t(status_key, self.lang), WARN)
        self.set_progress(0.0)
        self.btn_cancel.grid()
        self._started_at = time.time()
        self._tick()

        def wrapper():
            try:
                fn(ctx)
                self.ui(self._finish, "failed" if ctx.errors else "done", ctx)
            except tasks.Cancelled:
                self.ui(self._finish, "cancelled", ctx)
            except Exception:  # noqa: BLE001
                tb = traceback.format_exc()
                log_error(f"task:{getattr(fn, '__name__', '?')}", tb_text=tb)
                self.ui(self.log, f"\n[FATAL]\n{tb}", "err")
                self.ui(self._finish, "failed", ctx)

        self.worker = threading.Thread(target=wrapper, daemon=True)
        self.worker.start()

    def _finish(self, key: str, ctx: tasks.TaskContext):
        colors = {"done": OK, "failed": WARN, "cancelled": WARN}
        self.set_status(t(key, self.lang), colors.get(key, TXT))
        if key != "cancelled":
            self.set_progress(1.0)
        self.btn_cancel.grid_remove()
        if self._timer_job:
            self.after_cancel(self._timer_job)
            self._timer_job = None
        self.elapsed_label.configure(
            text=t("elapsed", self.lang, sec=time.time() - self._started_at))
        if ctx.errors:
            self.log("", "info")
            for err in ctx.errors:
                self.log(f"[!] {err}", "warn")
        self.active_ctx = None

    def request_cancel(self):
        if self.worker and self.worker.is_alive():
            self.cancel_event.set()
            if self.active_ctx:
                self.active_ctx.runner.cancel_current()
            self.log("\n>> CANCEL REQUESTED...", "warn")
            self.set_status(t("cancelled", self.lang), WARN)

    def report_callback_exception(self, exc, val, tb):
        """버튼·타이머 콜백에서 난 예외. 기본 동작은 stderr 출력인데 exe 에서는 stderr 가
        없어서 흔적 없이 사라진다. 파일에 남기고 사용자에게도 알린다."""
        text = "".join(traceback.format_exception(exc, val, tb))
        path = log_error("tk-callback", tb_text=text)
        try:
            self.toast(t("err_logged", self.lang, path=str(path or LOG_DIR)), "err", 9000)
        except Exception:
            pass

    def _on_close(self):
        self.cancel_event.set()
        if self.active_ctx:
            self.active_ctx.runner.cancel_current()
        self.destroy()

    # =================================================================
    # 공통 다이얼로그
    # =================================================================
    def confirm(self, title, message="", items=None, warn=None, danger=False,
                ok_text=None):
        return ConfirmDialog(self, self.lang, title, message, items=items, warn=warn,
                             danger=danger, ok_text=ok_text).result

    def require_admin(self) -> bool:
        if self.admin:
            return True
        self.toast(t("need_admin", self.lang), "warn")
        return False

    def set_best_dns(self, best, adapters):
        page = self.pages.get("network")
        if page is not None:
            page.set_best(best, adapters)

    # =================================================================
    # 언어
    # =================================================================
    def toggle_language(self):
        self.lang = "KO" if self.lang == "EN" else "EN"
        self.cfg["lang"] = self.lang
        self.retranslate()

    def retranslate(self):
        self.btn_lang.configure(text=t("btn_lang", self.lang))
        self.btn_cancel.configure(text=t("btn_cancel", self.lang))
        self._sync_badge()
        for label, key in self.nav_labels:
            label.configure(text=t(key, self.lang))
        for page_key, btn in self.nav_buttons.items():
            text_key, icon = self.nav_keys[page_key]
            btn.configure(text=f"  {icon}   {t(text_key, self.lang)}")
        for page in self.pages.values():
            try:
                page.retranslate()
            except Exception:
                pass
        if not (self.worker and self.worker.is_alive()):
            self.set_status(t("ready", self.lang), TXT)

    # =================================================================
    # 설정 / 정보
    # =================================================================
    def show_settings(self):
        ko = self.lang == "KO"
        opts = [
            ("opt_restore_point",
             "위험한 작업 전 복원 지점 만들기" if ko else "Create a restore point first",
             self.cfg["opt_restore_point"]),
            ("opt_browser_cache",
             "브라우저 캐시도 청소 (비밀번호·쿠키는 건드리지 않음)" if ko
             else "Also clean browser cache (passwords/cookies untouched)",
             self.cfg["opt_browser_cache"]),
            ("opt_wu_cache",
             "Windows 업데이트 캐시 정리" if ko else "Clean Windows Update cache",
             self.cfg["opt_wu_cache"]),
            ("opt_dism",
             "DISM 시스템 구성 정리 (수 분 소요)" if ko
             else "DISM component cleanup (takes minutes)", self.cfg["opt_dism"]),
            ("opt_defrag",
             "디스크 최적화 (SSD=TRIM / HDD=조각 모음)" if ko
             else "Disk optimize (SSD=TRIM / HDD=defrag)", self.cfg["opt_defrag"]),
            ("opt_prefetch",
             "Prefetch 삭제 (다음 부팅이 잠깐 느려짐)" if ko
             else "Clear Prefetch (slower next boot)", self.cfg["opt_prefetch"]),
            ("opt_recycle_bin",
             "휴지통 비우기 (영구 삭제!)" if ko else "Empty Recycle Bin (permanent!)",
             self.cfg["opt_recycle_bin"]),
            ("opt_winsock_reset",
             "Winsock 초기화 (재부팅 필요!)" if ko else "Winsock reset (needs reboot!)",
             self.cfg["opt_winsock_reset"]),
            ("opt_kill_office",
             "게임 모드에서 오피스/한글도 종료 (문서 손실 주의)" if ko
             else "Close Office/Hwp in Game Mode (data loss risk)",
             self.cfg["opt_kill_office"]),
            ("opt_game_power",
             "게임 모드에서 고성능 전원 관리로 전환" if ko
             else "Switch to High performance in Game Mode", self.cfg["opt_game_power"]),
        ]
        result = self.confirm(
            t("btn_settings", self.lang),
            ("체크한 항목만 실행합니다." if ko else "Only checked items will run.")
            + f"\n{CONFIG_PATH}",
            items=opts)
        if result:
            for key, value in result.items():
                self.cfg[key] = value
            self.toast("설정을 저장했습니다." if ko else "Settings saved.", "ok")

    def show_about(self):
        ko = self.lang == "KO"
        body = (
            f"{APP_NAME}  {APP_VERSION}\n\n"
            f"DEVELOPER : 이지용 (LEE_JI_YONG)\n"
            f"CONTACT   : yeez0612@naver.com\n"
            f"PYTHON    : {sys.version.split()[0]}\n"
            f"MODE      : {'ADMIN' if self.admin else 'LIMITED'}\n"
            f"CONFIG    : {CONFIG_PATH}\n"
            f"LOGS      : {LOG_DIR}\n"
        )
        bmc = (self.cfg["donation_bmc"] or "").strip()
        account = (self.cfg["donation_account"] or "").strip()
        hint = ("" if (bmc or account) else
                ("config.json 의 donation_bmc / donation_account 를 채우면 "
                 "후원 버튼이 표시됩니다." if ko else
                 "Fill donation_bmc / donation_account in config.json to show "
                 "the support button."))

        if not (bmc or account):
            self.confirm("ABOUT", body, warn=hint or None, ok_text=t("btn_close", self.lang))
            return

        items = []
        if bmc:
            items.append(("bmc", "☕  Buy Me a Coffee", False))
        if account:
            items.append(("account", f"🟡  계좌번호 복사  ({account})", False))
        result = self.confirm("ABOUT", body, items=items,
                              ok_text=t("btn_close", self.lang))
        if not result:
            return
        if result.get("bmc") and bmc:
            webbrowser.open(bmc)
        if result.get("account") and account:
            self.clipboard_clear()
            self.clipboard_append(account)
            self.toast("계좌번호를 복사했습니다." if ko else "Account number copied.", "ok")


def launch(cfg: Config, admin: bool) -> SystemCleanerApp:
    app = SystemCleanerApp(cfg, admin)
    return app
