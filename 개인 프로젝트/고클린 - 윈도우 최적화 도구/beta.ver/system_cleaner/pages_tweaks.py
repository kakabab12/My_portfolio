# -*- coding: utf-8 -*-
"""윈도우 설정 화면. 표 대신 스위치 목록이 읽기 편하다."""

from __future__ import annotations

import winreg
from dataclasses import dataclass

import customtkinter as ctk

from . import tweaks
from .core import is_admin
from .ui import (ACCENT, BORDER, CARD, DANGER, OK, TXT, TXT_DIM, WARN,
                 Page, toolbar_button)


@dataclass
class _Row:
    tweak: tweaks.Tweak
    title: ctk.CTkLabel
    desc: ctk.CTkLabel
    var: ctk.BooleanVar
    switch: ctk.CTkSwitch
    locked: bool


class TweaksPage(Page):
    key = "tweaks"
    title_key = "twk_title"
    desc_key = "twk_desc"

    def __init__(self, app, **kw):
        super().__init__(app, **kw)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self.header.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.btn_refresh = toolbar_button(bar, self.t("btn_refresh"), self.refresh_states,
                                          "accent", width=120)
        self.btn_refresh.pack(side="left")
        self.btn_explorer = toolbar_button(bar, self.t("twk_explorer_btn"),
                                           self.restart_explorer, "ghost", width=160)
        self.btn_explorer.pack(side="left", padx=(8, 0))

        self.list = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.list.grid(row=2, column=0, sticky="nsew")
        self.list.grid_columnconfigure(0, weight=1)

        self.status = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=12),
                                   text_color=TXT_DIM, anchor="w")
        self.status.grid(row=3, column=0, sticky="ew", pady=(8, 0))

        # 카드는 한 번만 만든다. 예전엔 화면을 열 때와 언어를 바꿀 때마다 위젯 80여 개를
        # 지우고 다시 만들어서 그때마다 화면이 1~2초 멈췄다.
        self._heads: list[tuple[ctk.CTkLabel, str]] = []
        self._rows: dict[str, _Row] = {}
        self._build()
        self.refresh_states()

    # 기존 테스트/호출과의 호환
    @property
    def _switches(self) -> dict:
        return {k: r.switch for k, r in self._rows.items()}

    def reload(self):
        self.refresh_states()

    def on_show(self):
        self.refresh_states()

    # --- 목록 ---
    def _build(self):
        current = None
        for tweak in tweaks.TWEAKS:
            if tweak.category != current:
                current = tweak.category
                head = ctk.CTkLabel(self.list, text="", font=ctk.CTkFont(size=11, weight="bold"),
                                    text_color=ACCENT, anchor="w")
                head.grid(sticky="ew", padx=6, pady=(14, 4))
                self._heads.append((head, current))
            self._rows[tweak.key] = self._make_row(tweak)
        self._apply_texts()

    def _make_row(self, tweak: tweaks.Tweak) -> _Row:
        locked = tweak.needs_admin and not is_admin()

        card = ctk.CTkFrame(self.list, fg_color=CARD, corner_radius=8,
                            border_width=1, border_color=BORDER)
        card.grid(sticky="ew", padx=4, pady=3)
        card.grid_columnconfigure(0, weight=1)

        left = ctk.CTkFrame(card, fg_color="transparent")
        left.grid(row=0, column=0, sticky="ew", padx=(14, 8), pady=10)

        title = ctk.CTkLabel(left, text="", font=ctk.CTkFont(size=13, weight="bold"),
                             text_color=TXT_DIM if locked else TXT, anchor="w")
        title.pack(fill="x")
        desc = ctk.CTkLabel(left, text="", font=ctk.CTkFont(size=11), text_color=TXT_DIM,
                            anchor="w", justify="left", wraplength=760)
        desc.pack(fill="x", pady=(2, 0))
        hive_name = "HKCU" if tweak.hive == winreg.HKEY_CURRENT_USER else "HKLM"
        ctk.CTkLabel(left, text=f"{hive_name}\\{tweak.path}\\{tweak.name}",
                     font=ctk.CTkFont(size=10), text_color="#5a5a5a",
                     anchor="w").pack(fill="x", pady=(3, 0))

        var = ctk.BooleanVar(value=False)
        switch = ctk.CTkSwitch(card, text="", variable=var, width=48,
                               progress_color=ACCENT, button_color="#DDDDDD",
                               button_hover_color="#FFFFFF",
                               command=lambda tw=tweak: self._toggle(tw))
        switch.grid(row=0, column=1, padx=(0, 16))
        if locked:
            switch.configure(state="disabled")
        return _Row(tweak, title, desc, var, switch, locked)

    def refresh_states(self):
        """레지스트리에서 현재 값을 다시 읽어 스위치와 표시만 갱신한다."""
        for row in self._rows.values():
            row.var.set(tweaks.state_of(row.tweak) == "applied")
            self._update_title(row)
        note = "" if is_admin() else self.t("limited_warn")
        self.status.configure(text=note, text_color=WARN if note else TXT_DIM)

    def _update_title(self, row: _Row):
        if row.locked:
            badge = self.t("twk_admin")
        else:
            badge = self.t("twk_applied") if row.var.get() else self.t("twk_default")
        row.title.configure(text=f"{row.tweak.title(self.lang)}   ·  {badge}")

    def _apply_texts(self):
        for head, category in self._heads:
            label = tweaks.CATEGORY_LABEL.get(category, (category, category))
            head.configure(text=label[1] if self.lang == "KO" else label[0])
        for row in self._rows.values():
            row.desc.configure(text=row.tweak.desc(self.lang))
            self._update_title(row)

    # --- 조작 ---
    def _toggle(self, tweak: tweaks.Tweak):
        row = self._rows[tweak.key]
        enable = row.var.get()
        ok, detail = tweaks.apply(tweak, enable, self.app.shared_runner)
        if not ok:
            row.var.set(not enable)
            self.status.configure(text=self.t("twk_failed", d=detail), text_color=DANGER)
            return
        # 예전엔 스위치만 움직이고 제목 옆 '기본값/적용됨' 표시는 그대로 남았다
        self._update_title(row)
        state = self.t("twk_applied") if enable else self.t("twk_default")
        if tweak.restart_explorer:
            self.status.configure(
                text=f"{tweak.title(self.lang)}  ·  {state}  ·  {self.t('twk_restart_hint')}",
                text_color=WARN)
        else:
            self.status.configure(text=f"{tweak.title(self.lang)}  ·  {state}", text_color=OK)

    def restart_explorer(self):
        # 열린 폴더 창이 전부 닫히는 동작인데 확인 없이 바로 실행됐다
        if not self.app.confirm(self.t("twk_explorer_q"), self.t("twk_explorer_d")):
            return
        self.btn_explorer.configure(state="disabled")

        def done(result):
            self.btn_explorer.configure(state="normal")
            if isinstance(result, Exception) or not result:
                self.status.configure(text=self.t("v_failed"), text_color=DANGER)
            else:
                self.status.configure(text=self.t("twk_restart"), text_color=OK)

        # taskkill 대기(최대 20초)를 화면 스레드에서 하지 않는다
        self.app.run_bg(lambda: tweaks.restart_explorer(self.app.shared_runner), done)

    def retranslate(self):
        super().retranslate()
        self.btn_refresh.configure(text=self.t("btn_refresh"))
        self.btn_explorer.configure(text=self.t("twk_explorer_btn"))
        self._apply_texts()
        self.refresh_states()
