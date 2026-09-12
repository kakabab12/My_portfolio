# -*- coding: utf-8 -*-
"""윈도우 설정 화면. 표 대신 스위치 목록이 읽기 편하다."""

from __future__ import annotations

import winreg

import customtkinter as ctk

from . import tweaks
from .core import is_admin
from .i18n import t
from .ui import (ACCENT, ACCENT_DIM, BORDER, CARD, DANGER, OK, TXT, TXT_DIM, WARN,
                 Page, toolbar_button)


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
        self.btn_refresh = toolbar_button(bar, self.t("btn_refresh"), self.reload,
                                          "accent", width=120)
        self.btn_refresh.pack(side="left")
        self.btn_explorer = toolbar_button(bar, "탐색기 다시 시작", self.restart_explorer,
                                           "ghost", width=160)
        self.btn_explorer.pack(side="left", padx=(8, 0))

        self.list = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.list.grid(row=2, column=0, sticky="nsew")
        self.list.grid_columnconfigure(0, weight=1)

        self.status = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=12),
                                   text_color=TXT_DIM, anchor="w")
        self.status.grid(row=3, column=0, sticky="ew", pady=(8, 0))

        self._rows: list = []
        self._switches: dict[str, ctk.CTkSwitch] = {}
        self._needs_explorer = False
        self.reload()

    def on_show(self):
        self.reload()

    # --- 목록 ---
    def reload(self):
        for widget in self._rows:
            widget.destroy()
        self._rows = []
        self._switches = {}

        current_category = None
        for tweak in tweaks.TWEAKS:
            if tweak.category != current_category:
                current_category = tweak.category
                label = tweaks.CATEGORY_LABEL.get(current_category, (current_category,) * 2)
                head = ctk.CTkLabel(
                    self.list, text=label[1] if self.lang == "KO" else label[0],
                    font=ctk.CTkFont(size=11, weight="bold"), text_color=ACCENT,
                    anchor="w")
                head.grid(sticky="ew", padx=6, pady=(14, 4))
                self._rows.append(head)

            self._rows.append(self._make_row(tweak))

        admin_note = "" if is_admin() else self.t("limited_warn")
        self.status.configure(text=admin_note, text_color=WARN if admin_note else TXT_DIM)

    def _make_row(self, tweak: tweaks.Tweak):
        state = tweaks.state_of(tweak)
        locked = tweak.needs_admin and not is_admin()

        card = ctk.CTkFrame(self.list, fg_color=CARD, corner_radius=8,
                            border_width=1, border_color=BORDER)
        card.grid(sticky="ew", padx=4, pady=3)
        card.grid_columnconfigure(0, weight=1)

        left = ctk.CTkFrame(card, fg_color="transparent")
        left.grid(row=0, column=0, sticky="ew", padx=(14, 8), pady=10)

        title = tweak.title(self.lang)
        badge = (f"   ·  {self.t('twk_admin')}" if locked else
                 f"   ·  {self.t('twk_applied') if state == 'applied' else self.t('twk_default')}")
        ctk.CTkLabel(left, text=title + badge, font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=TXT_DIM if locked else TXT, anchor="w").pack(fill="x")
        ctk.CTkLabel(left, text=tweak.desc(self.lang), font=ctk.CTkFont(size=11),
                     text_color=TXT_DIM, anchor="w", justify="left",
                     wraplength=760).pack(fill="x", pady=(2, 0))
        hive_name = "HKCU" if tweak.hive == winreg.HKEY_CURRENT_USER else "HKLM"
        ctk.CTkLabel(left, text=f"{hive_name}\\{tweak.path}\\{tweak.name}",
                     font=ctk.CTkFont(size=10), text_color="#5a5a5a",
                     anchor="w").pack(fill="x", pady=(3, 0))

        var = ctk.BooleanVar(value=(state == "applied"))
        switch = ctk.CTkSwitch(card, text="", variable=var, width=48,
                               progress_color=ACCENT, button_color="#DDDDDD",
                               button_hover_color="#FFFFFF",
                               command=lambda tw=tweak, v=var: self._toggle(tw, v))
        switch.grid(row=0, column=1, padx=(0, 16))
        if locked:
            switch.configure(state="disabled")
        self._switches[tweak.key] = switch
        return card

    def _toggle(self, tweak: tweaks.Tweak, var: ctk.BooleanVar):
        enable = var.get()
        ok, detail = tweaks.apply(tweak, enable, self.app.shared_runner)
        if not ok:
            var.set(not enable)
            self.status.configure(text=self.t("twk_failed", d=detail), text_color=DANGER)
            return
        if tweak.restart_explorer:
            self._needs_explorer = True
            self.status.configure(
                text=tweak.title(self.lang) + "  ·  "
                     + ("탐색기를 다시 시작하면 적용됩니다."
                        if self.lang == "KO" else "Restart Explorer to apply."),
                text_color=WARN)
        else:
            self.status.configure(text=f"{tweak.title(self.lang)}  ·  "
                                       f"{self.t('twk_applied') if enable else self.t('twk_default')}",
                                  text_color=OK)

    def restart_explorer(self):
        tweaks.restart_explorer(self.app.shared_runner)
        self._needs_explorer = False
        self.status.configure(text=self.t("twk_restart"), text_color=OK)

    def retranslate(self):
        super().retranslate()
        self.btn_refresh.configure(text=self.t("btn_refresh"))
        self.btn_explorer.configure(
            text="탐색기 다시 시작" if self.lang == "KO" else "Restart Explorer")
        self.reload()
