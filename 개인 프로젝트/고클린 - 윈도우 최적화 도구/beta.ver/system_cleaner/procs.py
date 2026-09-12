# -*- coding: utf-8 -*-
"""프로세스 관리, 메모리 정리, 게임 모드 대상/전원 관리."""

from __future__ import annotations

import ctypes
import os
import time
from ctypes import wintypes
from dataclasses import dataclass, field

from .core import Runner, audit, is_admin

try:
    import psutil
except ImportError:
    psutil = None


# 종료하면 시스템이 죽거나 재부팅되는 프로세스 - 목록에는 보이되 종료를 막는다
PROTECTED_PROCS = {
    "system", "system idle process", "registry", "memory compression",
    "smss.exe", "csrss.exe", "wininit.exe", "winlogon.exe", "services.exe",
    "lsass.exe", "lsaiso.exe", "fontdrvhost.exe", "dwm.exe", "svchost.exe",
    "sihost.exe", "ctfmon.exe", "audiodg.exe", "spoolsv.exe",
    "securityhealthservice.exe", "msmpeng.exe", "wudfhost.exe",
    "system32", "idle",
}

# 종료해도 되지만 데스크톱이 잠깐 사라지므로 경고가 필요한 것
WARN_PROCS = {"explorer.exe"}


@dataclass
class ProcItem:
    pid: int
    name: str
    username: str = ""
    memory: int = 0
    cpu: float = 0.0
    exe: str = ""
    obj: object = field(default=None, repr=False)

    @property
    def protected(self) -> bool:
        return self.name.lower() in PROTECTED_PROCS

    @property
    def warn(self) -> bool:
        return self.name.lower() in WARN_PROCS


def list_processes(sample_cpu: bool = False) -> list[ProcItem]:
    if psutil is None:
        return []
    procs = list(psutil.process_iter(["pid", "name", "username", "memory_info", "exe"]))
    if sample_cpu:
        for p in procs:
            try:
                p.cpu_percent(None)          # 1차 호출은 기준점만 잡는다
            except Exception:
                pass
        time.sleep(0.35)

    n_cpu = os.cpu_count() or 1
    out: list[ProcItem] = []
    for p in procs:
        try:
            info = p.info
            mem = info.get("memory_info")
            cpu = 0.0
            if sample_cpu:
                try:
                    cpu = p.cpu_percent(None) / n_cpu
                except Exception:
                    cpu = 0.0
            user = info.get("username") or ""
            if "\\" in user:
                user = user.split("\\")[-1]
            out.append(ProcItem(
                pid=info.get("pid") or 0,
                name=info.get("name") or "?",
                username=user,
                memory=getattr(mem, "rss", 0) if mem else 0,
                cpu=cpu,
                exe=info.get("exe") or "",
                obj=p,
            ))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    out.sort(key=lambda x: x.memory, reverse=True)
    return out


def close_processes(items: list[ProcItem], grace: float = 6.0,
                    force_wait: float = 4.0) -> tuple[int, list[str]]:
    """정상 종료 요청 -> 대기 -> 남은 것만 강제 종료. (종료 수, 실패 이름들)."""
    if psutil is None:
        return 0, []
    victims = []
    for it in items:
        if it.protected:
            continue
        p = it.obj
        if p is None:
            try:
                p = psutil.Process(it.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        victims.append(p)

    for p in victims:
        try:
            p.terminate()          # 저장 프롬프트가 뜰 기회를 준다
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    gone, alive = psutil.wait_procs(victims, timeout=grace)

    for p in alive:
        try:
            p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    gone2, still = psutil.wait_procs(alive, timeout=force_wait)

    failed = []
    for p in still:
        try:
            failed.append(p.name())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    closed = len(gone) + len(gone2)
    if closed:
        audit("process.close", f"{closed} closed")
    return closed, failed


def restart_explorer(runner: Runner) -> bool:
    runner.run(["taskkill", "/f", "/im", "explorer.exe"], timeout=20)
    try:
        os.startfile("explorer.exe")
        return True
    except OSError:
        return False


# =====================================================================
# 메모리 정리 (작업 집합 트림)
# =====================================================================
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_QUERY_INFO = 0x0400
_PROCESS_QUERY_LIMITED = 0x1000


def trim_working_sets(cancel=None) -> tuple[int, int]:
    """각 프로세스의 작업 집합을 비운다. (성공 수, 시도 수).

    EmptyWorkingSet 은 램을 '해제'하는 게 아니라 물리 메모리에서 페이지 파일로
    내리는 것이다. 앱이 다시 쓰면 되돌아오므로 효과는 일시적이다.
    (숫자만 예쁘게 만드는 메모리 클리너들이 실제로 하는 일도 이것이다.)
    """
    if psutil is None:
        return 0, 0
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]

    ok = tried = 0
    for p in psutil.process_iter(["pid", "name"]):
        if cancel is not None and cancel.is_set():
            break
        pid = p.info.get("pid") or 0
        if pid <= 4:
            continue
        tried += 1
        handle = kernel32.OpenProcess(
            _PROCESS_SET_QUOTA | _PROCESS_QUERY_LIMITED, False, pid)
        if not handle:
            continue
        try:
            if psapi.EmptyWorkingSet(handle):
                ok += 1
        except Exception:
            pass
        finally:
            kernel32.CloseHandle(handle)
    audit("memory.trim", f"{ok}/{tried}")
    return ok, tried


def memory_state() -> tuple[int, int, float]:
    """(available, total, percent_used)."""
    if psutil is None:
        return 0, 0, 0.0
    vm = psutil.virtual_memory()
    return vm.available, vm.total, vm.percent


# =====================================================================
# 게임 모드
# =====================================================================
GAME_TARGETS = [
    "chrome.exe", "msedge.exe", "whale.exe", "firefox.exe", "opera.exe", "brave.exe",
    "KakaoTalk.exe", "Discord.exe", "Slack.exe", "Telegram.exe", "LINE.exe",
    "Skype.exe", "Teams.exe", "ms-teams.exe", "Zoom.exe",
    "EpicGamesLauncher.exe", "Battle.net.exe", "RiotClientServices.exe",
    "GalaxyClient.exe", "Origin.exe", "EADesktop.exe", "UplayWebCore.exe",
    "OneDrive.exe", "Dropbox.exe", "GoogleDriveFS.exe",
    "Spotify.exe", "iTunes.exe", "Music.UI.exe",
    "Adobe Desktop Service.exe", "CCXProcess.exe", "AdobeIPCBroker.exe",
]

# 문서가 날아갈 수 있어 기본 목록에서 뺀 것들 (설정에서 켤 수 있음)
OFFICE_TARGETS = [
    "EXCEL.EXE", "WINWORD.EXE", "POWERPNT.EXE", "OUTLOOK.EXE",
    "Hwp.exe", "Hword.exe", "notepad++.exe", "Code.exe", "pycharm64.exe",
]

HIGH_PERF_GUID = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
BALANCED_GUID = "381b4222-f694-41f0-9685-ff5bb260df2e"


def find_game_targets(include_office: bool = False) -> dict[str, list[ProcItem]]:
    """지금 실제로 떠 있는 대상 프로세스만 이름별로 묶어서 돌려준다."""
    wanted = {n.lower() for n in GAME_TARGETS}
    if include_office:
        wanted |= {n.lower() for n in OFFICE_TARGETS}

    found: dict[str, list[ProcItem]] = {}
    for item in list_processes():
        if item.name.lower() in wanted:
            found.setdefault(item.name, []).append(item)
    return found


def set_power_plan(runner: Runner, guid: str) -> bool:
    if not is_admin():
        return False
    r = runner.run(["powercfg", "/setactive", guid], timeout=20)
    if r.ok:
        audit("power.plan", guid)
    return r.ok


def active_power_plan(runner: Runner) -> str:
    r = runner.run(["powercfg", "/getactivescheme"], timeout=20)
    if not r.ok:
        return ""
    return r.out.strip()
