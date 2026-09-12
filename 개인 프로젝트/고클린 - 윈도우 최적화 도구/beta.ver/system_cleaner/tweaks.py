# -*- coding: utf-8 -*-
"""윈도우 설정 트윅.

모두 레지스트리 값 하나를 켜고 끄는 수준이고, 되돌리는 것도 같은 방법이다.
설명 없이 '최적화' 라고 뭉뚱그리지 않고 무엇이 바뀌는지 항목마다 적었다.
"""

from __future__ import annotations

import os
import winreg
from dataclasses import dataclass

from .core import Runner, audit, is_admin

HKCU = winreg.HKEY_CURRENT_USER
HKLM = winreg.HKEY_LOCAL_MACHINE

_EXPLORER_ADV = r"Software\Microsoft\Windows\CurrentVersion\Explorer\Advanced"
_CDM = r"Software\Microsoft\Windows\CurrentVersion\ContentDeliveryManager"


@dataclass
class Tweak:
    key: str
    title_ko: str
    title_en: str
    desc_ko: str
    desc_en: str
    hive: int
    path: str
    name: str
    on_value: int          # '적용' 상태의 값
    off_value: int         # '기본' 상태의 값
    reg_type: int = winreg.REG_DWORD
    needs_admin: bool = False
    restart_explorer: bool = False
    category: str = "explorer"

    def title(self, lang):
        return self.title_ko if lang == "KO" else self.title_en

    def desc(self, lang):
        return self.desc_ko if lang == "KO" else self.desc_en


TWEAKS: list[Tweak] = [
    # --- 탐색기 ---
    Tweak("show_ext", "파일 확장명 표시", "Show file extensions",
          "song.mp3.exe 같은 위장 파일을 알아볼 수 있습니다. 보안상 켜두는 걸 권합니다.",
          "Lets you spot disguised files like song.mp3.exe. Recommended.",
          HKCU, _EXPLORER_ADV, "HideFileExt", 0, 1, restart_explorer=True),
    Tweak("show_hidden", "숨김 파일 표시", "Show hidden files",
          "숨김 속성이 걸린 파일과 폴더를 보이게 합니다.",
          "Shows files and folders marked hidden.",
          HKCU, _EXPLORER_ADV, "Hidden", 1, 2, restart_explorer=True),
    Tweak("launch_to_pc", "탐색기를 '내 PC'로 열기", "Open Explorer to This PC",
          "기본값인 '빠른 액세스' 대신 드라이브 목록이 먼저 보입니다.",
          "Starts at the drive list instead of Quick access.",
          HKCU, _EXPLORER_ADV, "LaunchTo", 1, 2, restart_explorer=True),
    Tweak("taskbar_search", "작업 표시줄 검색 상자 숨기기", "Hide taskbar search box",
          "작업 표시줄 공간을 넓힙니다. 검색은 시작 버튼으로 계속 됩니다.",
          "Frees taskbar space. Search still works from Start.",
          HKCU, r"Software\Microsoft\Windows\CurrentVersion\Search",
          "SearchboxTaskbarMode", 0, 1, restart_explorer=True, category="taskbar"),

    # --- 광고 / 추천 ---
    Tweak("ad_id", "광고 ID 끄기", "Disable advertising ID",
          "앱이 사용자별 광고 식별자로 관심사를 추적하지 못하게 합니다.",
          "Stops apps from tracking you with a per-user ad ID.",
          HKCU, r"Software\Microsoft\Windows\CurrentVersion\AdvertisingInfo",
          "Enabled", 0, 1, category="privacy"),
    Tweak("start_suggest", "시작 메뉴 추천 앱 끄기", "Disable Start menu suggestions",
          "시작 메뉴에 뜨는 '추천' 이라는 이름의 앱 광고를 없앱니다.",
          "Removes the app ads shown as 'suggestions' in Start.",
          HKCU, _CDM, "SystemPaneSuggestionsEnabled", 0, 1, category="privacy"),
    Tweak("silent_apps", "자동 설치되는 추천 앱 끄기", "Stop auto-installed suggested apps",
          "윈도우가 임의로 설치하는 홍보용 앱(캔디크러시 등)을 막습니다.",
          "Blocks promo apps Windows installs on its own.",
          HKCU, _CDM, "SilentInstalledAppsEnabled", 0, 1, category="privacy"),
    Tweak("tips", "윈도우 팁/제안 알림 끄기", "Disable Windows tips",
          "'Windows를 최대한 활용하세요' 류의 알림을 끕니다.",
          "Turns off 'Get the most out of Windows' notifications.",
          HKCU, _CDM, "SubscribedContent-338389Enabled", 0, 1, category="privacy"),
    Tweak("lockscreen_ad", "잠금 화면 광고 끄기", "Disable lock screen ads",
          "Windows 스포트라이트 잠금 화면에 끼워 넣는 홍보를 끕니다.",
          "Turns off promos on the Spotlight lock screen.",
          HKCU, _CDM, "RotatingLockScreenOverlayEnabled", 0, 1, category="privacy"),
    Tweak("settings_ad", "설정 앱 추천 콘텐츠 끄기", "Disable Settings suggestions",
          "설정 화면 곳곳에 뜨는 추천 배너를 끕니다.",
          "Removes suggestion banners inside Settings.",
          HKCU, _CDM, "SubscribedContent-338393Enabled", 0, 1, category="privacy"),

    # --- 성능 / 게임 ---
    Tweak("game_dvr", "게임 DVR(백그라운드 녹화) 끄기", "Disable Game DVR",
          "Xbox Game Bar 의 상시 녹화 기능을 꺼서 게임 중 프레임 손실을 줄입니다.",
          "Stops Game Bar background recording, which costs FPS.",
          HKCU, r"System\GameConfigStore", "GameDVR_Enabled", 0, 1, category="game"),
    Tweak("mouse_accel", "마우스 가속 끄기", "Disable mouse acceleration",
          "'포인터 정확도 향상' 을 꺼서 같은 거리 이동 = 같은 커서 이동이 되게 합니다. "
          "FPS 게임에서 선호합니다.",
          "Turns off pointer precision so movement maps 1:1. Preferred for FPS games.",
          HKCU, r"Control Panel\Mouse", "MouseSpeed", 0, 1,
          reg_type=winreg.REG_SZ, category="game"),
    Tweak("verbose_status", "부팅 시 상세 상태 표시", "Verbose boot status",
          "부팅/종료가 어느 단계에서 오래 걸리는지 글자로 보여줍니다. 문제 진단에 유용합니다.",
          "Shows what Windows is actually doing during boot. Useful for diagnosis.",
          HKLM, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System",
          "VerboseStatus", 1, 0, needs_admin=True, category="boot"),

    # --- 개인정보 / 네트워크 ---
    Tweak("telemetry", "진단 데이터 최소화", "Minimize telemetry",
          "선택적 진단 데이터 전송을 끕니다. (Home 에디션은 완전히 꺼지지 않을 수 있습니다)",
          "Turns off optional diagnostic data. (Home edition may keep a baseline)",
          HKLM, r"SOFTWARE\Policies\Microsoft\Windows\DataCollection",
          "AllowTelemetry", 0, 1, needs_admin=True, category="privacy"),
    Tweak("do_p2p", "배달 최적화 P2P 업로드 끄기", "Disable Delivery Optimization P2P",
          "내 PC 가 다른 사람에게 윈도우 업데이트를 업로드하지 않게 합니다. 업로드 대역폭이 절약됩니다.",
          "Stops your PC from uploading Windows updates to strangers.",
          HKLM, r"SOFTWARE\Policies\Microsoft\Windows\DeliveryOptimization",
          "DODownloadMode", 0, 1, needs_admin=True, category="network"),
]


def _read(tweak: Tweak):
    try:
        with winreg.OpenKey(tweak.hive, tweak.path, 0, winreg.KEY_READ) as k:
            return winreg.QueryValueEx(k, tweak.name)[0]
    except OSError:
        return None


def state_of(tweak: Tweak) -> str:
    """applied | default | unset"""
    value = _read(tweak)
    if value is None:
        return "unset"
    if tweak.reg_type == winreg.REG_SZ:
        return "applied" if str(value) == str(tweak.on_value) else "default"
    try:
        value = int(value)
    except (TypeError, ValueError):
        return "default"
    return "applied" if value == tweak.on_value else "default"


def apply(tweak: Tweak, enable: bool, runner: Runner | None = None) -> tuple[bool, str]:
    if tweak.needs_admin and not is_admin():
        return False, "관리자 권한 필요"
    value = tweak.on_value if enable else tweak.off_value
    try:
        with winreg.CreateKeyEx(tweak.hive, tweak.path, 0, winreg.KEY_SET_VALUE) as k:
            if tweak.reg_type == winreg.REG_SZ:
                winreg.SetValueEx(k, tweak.name, 0, winreg.REG_SZ, str(value))
            else:
                winreg.SetValueEx(k, tweak.name, 0, winreg.REG_DWORD, int(value))
    except OSError as e:
        return False, str(e)

    # 마우스 가속은 레지스트리만 바꿔서는 즉시 반영되지 않는다
    if tweak.key == "mouse_accel":
        _apply_mouse_speed(enable)

    audit("tweak", f"{tweak.key} -> {'on' if enable else 'off'}")
    return True, ""


def _apply_mouse_speed(disable_accel: bool):
    """SystemParametersInfo 로 즉시 반영 (재로그인 없이)."""
    import ctypes
    SPI_SETMOUSE = 0x0004
    SPIF_SENDCHANGE = 0x02
    params = (ctypes.c_int * 3)(0, 0, 0) if disable_accel else (ctypes.c_int * 3)(6, 10, 1)
    try:
        ctypes.windll.user32.SystemParametersInfoW(SPI_SETMOUSE, 0, params, SPIF_SENDCHANGE)
    except Exception:
        pass


def restart_explorer_needed(tweaks: list[Tweak]) -> bool:
    return any(t.restart_explorer for t in tweaks)


def restart_explorer(runner: Runner) -> bool:
    runner.run(["taskkill", "/f", "/im", "explorer.exe"], timeout=20)
    try:
        os.startfile("explorer.exe")
        return True
    except OSError:
        return False


def categories() -> list[str]:
    seen = []
    for t in TWEAKS:
        if t.category not in seen:
            seen.append(t.category)
    return seen


CATEGORY_LABEL = {
    "explorer": ("Explorer", "탐색기"),
    "taskbar":  ("Taskbar", "작업 표시줄"),
    "privacy":  ("Privacy & ads", "개인정보 · 광고"),
    "game":     ("Gaming", "게임"),
    "boot":     ("Boot", "부팅"),
    "network":  ("Network", "네트워크"),
}
