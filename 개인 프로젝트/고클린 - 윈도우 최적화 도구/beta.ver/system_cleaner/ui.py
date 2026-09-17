# -*- coding: utf-8 -*-
"""재사용 위젯: 터미널, 확인 창, 데이터 테이블, 통계 카드."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import customtkinter as ctk

from .core import display_width
from .i18n import t

# 색상
BG = "#0f0f0f"
SIDEBAR = "#161616"
CARD = "#1a1a1a"
CARD_HI = "#202020"
ACCENT = "#00E5FF"
ACCENT_DIM = "#00B8D4"
TERM_BG = "#000000"
TERM_TXT = "#00FF41"
TXT = "#E6E6E6"
TXT_DIM = "#8A8A8A"
BORDER = "#2E2E2E"
OK = "#4CAF50"
WARN = "#FFB300"
DANGER = "#FF5252"


def apply_table_style():
    """ttk.Treeview 를 다크 테마로. 수백~수천 행을 다뤄야 해서 Treeview 를 쓴다."""
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("SC.Treeview",
                    background="#121212", fieldbackground="#121212", foreground=TXT,
                    borderwidth=0, relief="flat", rowheight=26,
                    font=("Segoe UI", 10))
    style.configure("SC.Treeview.Heading",
                    background="#1f1f1f", foreground=ACCENT, relief="flat",
                    borderwidth=0, font=("Segoe UI", 9, "bold"), padding=(6, 6))
    style.map("SC.Treeview.Heading", background=[("active", "#2b2b2b")])
    style.map("SC.Treeview",
              background=[("selected", "#004A57")],
              foreground=[("selected", "#FFFFFF")])
    style.layout("SC.Treeview", [("SC.Treeview.treearea", {"sticky": "nswe"})])


# =====================================================================
# 터미널
# =====================================================================
class Terminal(ctk.CTkFrame):
    def __init__(self, master, **kw):
        super().__init__(master, fg_color=TERM_BG, corner_radius=8,
                         border_width=1, border_color=BORDER, **kw)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.box = ctk.CTkTextbox(self, font=("Consolas", 13), text_color=TERM_TXT,
                                  fg_color="transparent", wrap="none")
        self.box.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        for tag, color in (("info", TERM_TXT), ("ok", ACCENT), ("warn", WARN),
                           ("err", DANGER), ("dim", "#6f8f6f")):
            try:
                self.box.tag_config(tag, foreground=color)
            except Exception:
                pass
        self.box.configure(state="disabled")   # 사용자가 로그 창에 타이핑하지 못하게

    def log(self, message: str, tag: str = "info"):
        self.box.configure(state="normal")
        start = self.box.index("end-1c")
        self.box.insert("end", str(message) + "\n")
        try:
            self.box.tag_add(tag, start, "end-1c")
        except Exception:
            pass
        self.box.see("end")
        self.box.configure(state="disabled")

    def log_summary(self, title: str, rows: list[tuple[str, str]]):
        if not rows:
            return
        label_w = max(display_width(a) for a, _ in rows)
        lines = [f"{a}{' ' * max(0, label_w - display_width(a))} : {b}" for a, b in rows]
        inner = max([display_width(x) for x in lines] + [display_width(title) + 8])
        top = f"╔══ [ {title} ] " + "═" * max(2, inner - display_width(title) - 6)
        self.log("")
        self.log(top, "ok")
        self.log("║", "ok")
        for line in lines:
            self.log(f"║   {line}", "ok")
        self.log("║", "ok")
        self.log("╚" + "═" * (display_width(top) - 1), "ok")

    def clear(self):
        self.box.configure(state="normal")
        self.box.delete("0.0", "end")
        self.box.configure(state="disabled")

    def text(self) -> str:
        return self.box.get("0.0", "end")


# =====================================================================
# 확인 창
# =====================================================================
class ConfirmDialog(ctk.CTkToplevel):
    """모달 확인 창. items 를 주면 체크박스 목록이 된다."""

    def __init__(self, master, lang, title, message, items=None, warn=None,
                 ok_text=None, ok_color=ACCENT, danger=False):
        super().__init__(master)
        self.result = None
        self.vars: dict[str, ctk.BooleanVar] = {}

        self.title(title)
        self.configure(fg_color="#141414")
        self.resizable(False, False)
        self.transient(master)

        ctk.CTkLabel(self, text=title, font=ctk.CTkFont(size=16, weight="bold"),
                     text_color=TXT, wraplength=460, justify="left"
                     ).pack(padx=24, pady=(22, 6), anchor="w")
        if message:
            ctk.CTkLabel(self, text=message, font=ctk.CTkFont(size=12),
                         text_color=TXT_DIM, wraplength=460, justify="left"
                         ).pack(padx=24, pady=(0, 10), anchor="w")
        if warn:
            ctk.CTkLabel(self, text="⚠  " + warn,
                         font=ctk.CTkFont(size=12, weight="bold"),
                         text_color=DANGER if danger else WARN,
                         wraplength=460, justify="left"
                         ).pack(padx=24, pady=(0, 12), anchor="w")

        if items:
            box = ctk.CTkScrollableFrame(self, fg_color="#0d0d0d",
                                         height=min(280, 34 * len(items) + 10))
            box.pack(fill="x", padx=24, pady=(0, 12))
            for key, label, default in items:
                var = ctk.BooleanVar(value=default)
                self.vars[key] = var
                ctk.CTkCheckBox(box, text=label, variable=var, font=ctk.CTkFont(size=12),
                                text_color=TXT, fg_color=ACCENT, hover_color=ACCENT_DIM,
                                checkmark_color="black", border_color="#555555"
                                ).pack(anchor="w", pady=3, padx=4)

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=24, pady=(0, 20))
        ctk.CTkButton(row, text=t("dlg_no", lang), width=110, fg_color="#2b2b2b",
                      hover_color="#3a3a3a", command=self._cancel).pack(side="right", padx=(8, 0))
        ctk.CTkButton(row, text=ok_text or t("dlg_yes", lang), width=140,
                      fg_color=DANGER if danger else ok_color,
                      text_color="white" if danger else "black",
                      hover_color="#C62828" if danger else ACCENT_DIM,
                      command=self._ok).pack(side="right")

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.bind("<Escape>", lambda _e: self._cancel())
        self.bind("<Return>", lambda _e: self._ok())

        self.update_idletasks()
        self._center(master)
        self.after(60, self._grab)      # 창이 실제로 보인 뒤에 grab
        master.wait_window(self)

    def _grab(self):
        try:
            self.grab_set()
            self.focus_force()
        except Exception:
            pass

    def _center(self, master):
        try:
            w, h = self.winfo_width(), self.winfo_height()
            x = master.winfo_rootx() + (master.winfo_width() - w) // 2
            y = master.winfo_rooty() + (master.winfo_height() - h) // 3
            self.geometry(f"+{max(0, x)}+{max(0, y)}")
        except Exception:
            pass

    def _ok(self):
        self.result = {k: v.get() for k, v in self.vars.items()} if self.vars else True
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()


# =====================================================================
# 데이터 테이블
# =====================================================================
CHECK_ON = "☑"
CHECK_OFF = "☐"


class DataTable(ctk.CTkFrame):
    """체크박스 + 정렬 + 더블클릭을 지원하는 표.

    columns: [(key, 제목, 폭, anchor)]
    """

    def __init__(self, master, columns, checkable=True, on_double=None,
                 on_select=None, **kw):
        super().__init__(master, fg_color="#121212", corner_radius=8,
                         border_width=1, border_color=BORDER, **kw)
        self.columns = columns
        self.checkable = checkable
        self.on_double = on_double
        self.on_select = on_select
        self._payloads: dict[str, object] = {}
        self._checked: set[str] = set()
        self._sort_key: str | None = None
        self._sort_desc = False
        self._rows: list[tuple[object, tuple]] = []
        self._tags: dict[str, str] = {}

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        col_ids = (["__chk__"] if checkable else []) + [c[0] for c in columns]
        self.tree = ttk.Treeview(self, columns=col_ids, show="headings",
                                 style="SC.Treeview", selectmode="extended")
        if checkable:
            self.tree.heading("__chk__", text=CHECK_OFF,
                              command=self._toggle_all_from_header)
            self.tree.column("__chk__", width=38, minwidth=38, stretch=False, anchor="center")
        for key, title, width, anchor in columns:
            self.tree.heading(key, text=title, command=lambda k=key: self.sort_by(k))
            self.tree.column(key, width=width, minwidth=40, anchor=anchor,
                             stretch=(anchor == "w"))
        self.tree.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=6)

        self.scroll = ctk.CTkScrollbar(self, command=self.tree.yview,
                                       button_color="#3a3a3a", button_hover_color="#4a4a4a")
        self.scroll.grid(row=0, column=1, sticky="ns", padx=(2, 6), pady=6)
        self.tree.configure(yscrollcommand=self.scroll.set)

        self.tree.tag_configure("odd", background="#161616")
        self.tree.tag_configure("dim", foreground=TXT_DIM)
        self.tree.tag_configure("warn", foreground=WARN)
        self.tree.tag_configure("danger", foreground=DANGER)
        self.tree.tag_configure("good", foreground=OK)

        self.tree.bind("<Button-1>", self._on_click, add="+")
        self.tree.bind("<space>", lambda _e: self._toggle_selected())
        self.tree.bind("<Double-1>", self._on_double, add="+")
        if on_select:
            self.tree.bind("<<TreeviewSelect>>", lambda _e: on_select(self.focused_payload()))

    # --- 데이터 ---
    def set_rows(self, rows: list[tuple[object, tuple]], tags: dict[int, str] | None = None,
                 sort_values: list[tuple] | None = None):
        """rows: [(payload, (열값들...))] / tags: {행 인덱스: 태그}

        sort_values 를 주면 정렬은 그 값으로 한다. '1.5 GB' 같은 표시용 문자열을
        그대로 정렬하면 사전순이 되어 크기 정렬이 망가지기 때문이다.
        """
        self._rows = [
            (payload, values, (tags or {}).get(i, ""),
             sort_values[i] if sort_values and i < len(sort_values) else values)
            for i, (payload, values) in enumerate(rows)
        ]
        self._sort_key = None
        self._render()

    def _render(self):
        self.tree.delete(*self.tree.get_children())
        self._payloads.clear()
        self._checked.clear()
        for i, (payload, values, tag, _sv) in enumerate(self._rows):
            iid = f"r{i}"
            tag_list = ["odd"] if i % 2 else []
            if tag:
                tag_list.append(tag)
            shown = ((CHECK_OFF,) if self.checkable else ()) + tuple(
                "" if v is None else str(v) for v in values)
            self.tree.insert("", "end", iid=iid, values=shown, tags=tuple(tag_list))
            self._payloads[iid] = payload
        self._update_header_check()

    def clear(self):
        self.set_rows([])

    # --- 체크 ---
    def _set_check(self, iid: str, on: bool):
        if not self.checkable:
            return
        vals = list(self.tree.item(iid, "values"))
        if not vals:
            return
        vals[0] = CHECK_ON if on else CHECK_OFF
        self.tree.item(iid, values=vals)
        if on:
            self._checked.add(iid)
        else:
            self._checked.discard(iid)

    def _toggle(self, iid: str):
        self._set_check(iid, iid not in self._checked)
        self._update_header_check()

    def _toggle_selected(self):
        for iid in self.tree.selection():
            self._toggle(iid)

    def _toggle_all_from_header(self):
        everything = self.tree.get_children()
        turn_on = len(self._checked) < len(everything)
        self.set_all(turn_on)

    def set_all(self, on: bool):
        for iid in self.tree.get_children():
            self._set_check(iid, on)
        self._update_header_check()

    def check_where(self, predicate):
        for iid in self.tree.get_children():
            self._set_check(iid, bool(predicate(self._payloads.get(iid))))
        self._update_header_check()

    def _update_header_check(self):
        if not self.checkable:
            return
        total = len(self.tree.get_children())
        mark = CHECK_ON if (total and len(self._checked) == total) else CHECK_OFF
        self.tree.heading("__chk__", text=mark)

    def set_headings(self, columns):
        """언어를 바꿀 때 머리글도 바꾼다. (처음 만들 때만 정하면 영문 모드에 한글이 남는다)"""
        self.columns = columns
        for key, title, _width, _anchor in columns:
            self.tree.heading(key, text=title)

    def checked_payloads(self) -> list:
        return [self._payloads[i] for i in self.tree.get_children() if i in self._checked]

    def checked_count(self) -> int:
        return len(self._checked)

    def focused_payload(self):
        sel = self.tree.selection()
        return self._payloads.get(sel[0]) if sel else None

    def target_payload(self):
        """'위치 열기'처럼 한 항목만 쓰는 동작의 대상. 선택한 줄이 없으면 체크한 첫 줄."""
        focused = self.focused_payload()
        if focused is not None:
            return focused
        checked = self.checked_payloads()
        return checked[0] if checked else None

    def selected_payloads(self) -> list:
        return [self._payloads[i] for i in self.tree.selection() if i in self._payloads]

    def all_payloads(self) -> list:
        return [self._payloads[i] for i in self.tree.get_children()]

    # --- 이벤트 ---
    def _on_click(self, event):
        if not self.checkable:
            return
        if self.tree.identify_region(event.x, event.y) != "cell":
            return
        if self.tree.identify_column(event.x) != "#1":
            return
        iid = self.tree.identify_row(event.y)
        if iid:
            self._toggle(iid)
            return "break"

    def _on_double(self, event):
        iid = self.tree.identify_row(event.y)
        if iid and self.on_double:
            self.on_double(self._payloads.get(iid))

    # --- 정렬 ---
    def sort_by(self, key: str):
        idx = [c[0] for c in self.columns].index(key)
        self._sort_desc = not self._sort_desc if self._sort_key == key else False
        self._sort_key = key

        def sort_value(item):
            row_sort = item[3]
            value = row_sort[idx] if idx < len(row_sort) else ""
            if isinstance(value, (int, float)):
                return (0, value, "")
            return (1, 0, str(value).lower())

        self._rows = sorted(self._rows, key=sort_value, reverse=self._sort_desc)
        self._render()


# =====================================================================
# 통계 카드
# =====================================================================
class StatCard(ctk.CTkFrame):
    def __init__(self, master, title="", value="-", sub="", accent=ACCENT,
                 bar=False, width=170, **kw):
        super().__init__(master, fg_color=CARD, corner_radius=10,
                         border_width=1, border_color=BORDER, width=width, **kw)
        self.grid_propagate(False)
        self.accent = accent
        self.title_label = ctk.CTkLabel(self, text=title, font=ctk.CTkFont(size=11),
                                        text_color=TXT_DIM, anchor="w")
        self.title_label.pack(fill="x", padx=14, pady=(12, 0))
        self.value_label = ctk.CTkLabel(self, text=value,
                                        font=ctk.CTkFont(size=24, weight="bold"),
                                        text_color=accent, anchor="w")
        self.value_label.pack(fill="x", padx=14, pady=(2, 0))
        self.sub_label = ctk.CTkLabel(self, text=sub, font=ctk.CTkFont(size=11),
                                      text_color=TXT_DIM, anchor="w")
        self.sub_label.pack(fill="x", padx=14, pady=(0, 8))
        self.bar = None
        if bar:
            self.bar = ctk.CTkProgressBar(self, height=4, corner_radius=2,
                                          progress_color=accent, fg_color="#2a2a2a")
            self.bar.set(0)
            self.bar.pack(fill="x", padx=14, pady=(0, 12))

    def update_values(self, value=None, sub=None, fraction=None, accent=None):
        if value is not None:
            self.value_label.configure(text=value)
        if sub is not None:
            self.sub_label.configure(text=sub)
        if accent is not None:
            self.value_label.configure(text_color=accent)
            if self.bar is not None:
                self.bar.configure(progress_color=accent)
        if fraction is not None and self.bar is not None:
            self.bar.set(max(0.0, min(1.0, fraction)))

    def set_title(self, title: str):
        self.title_label.configure(text=title)


# =====================================================================
# 페이지 공통 헤더
# =====================================================================
class PageHeader(ctk.CTkFrame):
    def __init__(self, master, title="", desc="", **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self.title_label = ctk.CTkLabel(self, text=title,
                                        font=ctk.CTkFont(size=21, weight="bold"),
                                        text_color=TXT, anchor="w")
        self.title_label.pack(fill="x")
        self.desc_label = ctk.CTkLabel(self, text=desc, font=ctk.CTkFont(size=12),
                                       text_color=TXT_DIM, anchor="w", justify="left",
                                       wraplength=900)
        self.desc_label.pack(fill="x", pady=(4, 0))
        # 창 크기가 바뀌면 줄바꿈 폭도 따라가야 설명문이 잘리지 않는다
        self.bind("<Configure>", self._fit_wrap)

    def _fit_wrap(self, event):
        width = max(360, event.width - 12)
        if abs(self.desc_label.cget("wraplength") - width) > 24:
            self.desc_label.configure(wraplength=width)

    def set(self, title=None, desc=None):
        if title is not None:
            self.title_label.configure(text=title)
        if desc is not None:
            self.desc_label.configure(text=desc)


def toolbar_button(master, text, command, kind="ghost", width=110):
    styles = {
        "ghost":  dict(fg_color="transparent", border_width=1, border_color="#3d3d3d",
                       text_color=TXT, hover_color="#242424"),
        "accent": dict(fg_color=ACCENT, text_color="black", hover_color=ACCENT_DIM,
                       border_width=0),
        "danger": dict(fg_color="#3a1c1c", text_color="#FF8A80", border_width=1,
                       border_color="#7a2b2b", hover_color="#4d2424"),
        "warn":   dict(fg_color="#3a2f10", text_color=WARN, border_width=1,
                       border_color="#6b551c", hover_color="#4a3c16"),
    }
    return ctk.CTkButton(master, text=text, command=command, height=32, width=width,
                         corner_radius=6, font=ctk.CTkFont(size=12, weight="bold"),
                         **styles.get(kind, styles["ghost"]))


# =====================================================================
# 페이지 베이스
# =====================================================================
class Page(ctk.CTkFrame):
    """네비게이션에 붙는 화면 한 장."""

    key = ""
    title_key = ""
    desc_key = ""

    def __init__(self, app, **kw):
        super().__init__(app.content, fg_color="transparent", **kw)
        self.app = app
        self.header = PageHeader(self, title=t(self.title_key, app.lang),
                                 desc=t(self.desc_key, app.lang))

    @property
    def lang(self) -> str:
        return self.app.lang

    def t(self, key, **kw) -> str:
        return t(key, self.app.lang, **kw)

    def on_show(self):
        """화면에 나타날 때."""

    def on_hide(self):
        """화면에서 벗어날 때 (타이머 정리 등)."""

    def retranslate(self):
        self.header.set(title=self.t(self.title_key), desc=self.t(self.desc_key))
