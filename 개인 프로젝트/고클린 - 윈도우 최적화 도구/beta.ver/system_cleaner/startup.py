# -*- coding: utf-8 -*-
"""시작 프로그램 관리.

'사용 안 함'은 항목을 지우지 않고 Windows 가 실제로 쓰는
Explorer\\StartupApproved 키에 비활성 플래그를 써서 처리한다.
작업 관리자가 쓰는 방식과 같고, 언제든 되돌릴 수 있다.
"""

from __future__ import annotations

import os
import re
import shlex
import winreg
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .core import BACKUP_DIR, Runner, audit, is_admin, save_backup, send_to_recycle_bin

# (표시용 위치, 루트 키, 서브키, StartupApproved 서브키, 32비트 뷰 여부)
_RUN_KEYS = [
    ("HKCU:Run", winreg.HKEY_CURRENT_USER,
     r"Software\Microsoft\Windows\CurrentVersion\Run",
     r"Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run", False),
    ("HKLM:Run", winreg.HKEY_LOCAL_MACHINE,
     r"Software\Microsoft\Windows\CurrentVersion\Run",
     r"Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run", False),
    ("HKLM:Run32", winreg.HKEY_LOCAL_MACHINE,
     r"Software\Wow6432Node\Microsoft\Windows\CurrentVersion\Run",
     r"Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run32", False),
    ("HKCU:RunOnce", winreg.HKEY_CURRENT_USER,
     r"Software\Microsoft\Windows\CurrentVersion\RunOnce", "", False),
    ("HKLM:RunOnce", winreg.HKEY_LOCAL_MACHINE,
     r"Software\Microsoft\Windows\CurrentVersion\RunOnce", "", False),
]

_APPROVED_FOLDER = r"Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\StartupFolder"


@dataclass
class StartupEntry:
    name: str
    command: str
    location: str            # 'HKCU:Run', 'StartupFolder', 'Task' ...
    enabled: bool = True
    target_exists: bool = True
    source: str = ""         # 레지스트리 경로 또는 .lnk 파일 경로
    kind: str = "registry"   # registry | folder | task
    hive: int = winreg.HKEY_CURRENT_USER
    subkey: str = ""
    approved_key: str = ""
    task_path: str = ""

    @property
    def needs_admin(self) -> bool:
        return self.hive == winreg.HKEY_LOCAL_MACHINE or self.kind == "task" \
            or self.location == "CommonStartupFolder"


# =====================================================================
# 명령줄에서 실행 파일 경로 뽑기
# =====================================================================
def extract_target(command: str) -> str:
    cmd = (command or "").strip()
    if not cmd:
        return ""
    if cmd.startswith('"'):
        end = cmd.find('"', 1)
        if end > 0:
            return cmd[1:end]
    # 따옴표가 없으면 .exe 까지를 경로로 본다
    low = cmd.lower()
    idx = low.find(".exe")
    if idx > 0:
        return cmd[: idx + 4]
    try:
        parts = shlex.split(cmd, posix=False)
        return parts[0].strip('"') if parts else cmd
    except ValueError:
        return cmd.split(" ")[0]


def _target_exists(command: str) -> bool:
    target = extract_target(command)
    if not target:
        return False
    target = os.path.expandvars(target)
    if os.path.exists(target):
        return True
    # rundll32 등 System32 에 있는 것들
    from shutil import which
    return which(target) is not None


# =====================================================================
# StartupApproved 플래그
# =====================================================================
def _approved_state(hive: int, approved_key: str, value_name: str) -> bool | None:
    """True=사용, False=사용 안 함, None=플래그 없음(=사용)."""
    if not approved_key:
        return None
    try:
        with winreg.OpenKey(hive, approved_key, 0, winreg.KEY_READ) as k:
            data, _ = winreg.QueryValueEx(k, value_name)
        if isinstance(data, bytes) and data:
            # 첫 바이트의 1번 비트가 서 있으면 '사용 안 함' (0x03, 0x07)
            return not bool(data[0] & 1)
    except FileNotFoundError:
        return None
    except OSError:
        return None
    return None


def _filetime_now() -> bytes:
    delta = datetime.now(timezone.utc) - datetime(1601, 1, 1, tzinfo=timezone.utc)
    ticks = int(delta.total_seconds() * 10_000_000)
    return ticks.to_bytes(8, "little")


def _set_approved(hive: int, approved_key: str, value_name: str, enabled: bool) -> bool:
    if not approved_key:
        return False
    blob = (b"\x02\x00\x00\x00" + b"\x00" * 8) if enabled \
        else (b"\x03\x00\x00\x00" + _filetime_now())
    try:
        with winreg.CreateKeyEx(hive, approved_key, 0, winreg.KEY_SET_VALUE) as k:
            winreg.SetValueEx(k, value_name, 0, winreg.REG_BINARY, blob)
        return True
    except OSError:
        return False


# =====================================================================
# 목록 수집
# =====================================================================
def _read_run_key(label, hive, subkey, approved, _wow) -> list[StartupEntry]:
    out: list[StartupEntry] = []
    try:
        with winreg.OpenKey(hive, subkey, 0, winreg.KEY_READ) as k:
            count = winreg.QueryInfoKey(k)[1]
            for i in range(count):
                try:
                    name, data, _ = winreg.EnumValue(k, i)
                except OSError:
                    continue
                if not isinstance(data, str):
                    data = str(data)
                state = _approved_state(hive, approved, name)
                out.append(StartupEntry(
                    name=name, command=data, location=label,
                    enabled=True if state is None else state,
                    target_exists=_target_exists(data),
                    source=f"{'HKCU' if hive == winreg.HKEY_CURRENT_USER else 'HKLM'}\\{subkey}",
                    kind="registry", hive=hive, subkey=subkey, approved_key=approved))
    except FileNotFoundError:
        pass
    except OSError:
        pass
    return out


def _startup_folders() -> list[tuple[str, Path, int]]:
    out = []
    user = os.environ.get("APPDATA")
    if user:
        p = Path(user) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
        out.append(("StartupFolder", p, winreg.HKEY_CURRENT_USER))
    common = os.environ.get("PROGRAMDATA")
    if common:
        p = Path(common) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
        out.append(("CommonStartupFolder", p, winreg.HKEY_LOCAL_MACHINE))
    return out


def _read_folders() -> list[StartupEntry]:
    out: list[StartupEntry] = []
    for label, folder, hive in _startup_folders():
        if not folder.exists():
            continue
        try:
            for entry in os.scandir(folder):
                if not entry.is_file():
                    continue
                if entry.name.lower() in ("desktop.ini",):
                    continue
                state = _approved_state(hive, _APPROVED_FOLDER, entry.name)
                out.append(StartupEntry(
                    name=Path(entry.name).stem, command=entry.path, location=label,
                    enabled=True if state is None else state,
                    target_exists=True, source=entry.path, kind="folder",
                    hive=hive, approved_key=_APPROVED_FOLDER))
        except OSError:
            continue
    return out


def _read_tasks(runner: Runner) -> list[StartupEntry]:
    """로그온 시 실행되는 작업 스케줄러 항목 (Adobe, 각종 업데이터 등)."""
    script = (
        "$ErrorActionPreference='SilentlyContinue';"
        "Get-ScheduledTask | Where-Object { $_.Triggers -ne $null -and "
        "($_.Triggers | ForEach-Object { $_.CimClass.CimClassName }) -match 'LogonTrigger|BootTrigger' } | "
        "ForEach-Object { [pscustomobject]@{ "
        "name=$_.TaskName; path=$_.TaskPath; state=$_.State.ToString(); "
        "cmd=(($_.Actions | ForEach-Object { $_.Execute }) -join ' ') } } | "
        "ConvertTo-Json -Compress"
    )
    rows = runner.ps_json(script, timeout=90)
    out: list[StartupEntry] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("name") or ""
        if not name:
            continue
        # \Microsoft\Windows\ 아래는 OS 내장 작업이라 건드릴 일이 없다. 목록만 지저분해진다.
        if (row.get("path") or "").startswith("\\Microsoft\\Windows\\"):
            continue
        cmd = row.get("cmd") or ""
        out.append(StartupEntry(
            name=name, command=cmd, location="Task",
            enabled=(row.get("state") or "").lower() != "disabled",
            target_exists=_target_exists(cmd) if cmd else True,
            source=(row.get("path") or "") + name, kind="task",
            task_path=(row.get("path") or "\\")))
    return out


def list_startup(runner: Runner | None = None, include_tasks: bool = True) -> list[StartupEntry]:
    out: list[StartupEntry] = []
    for spec in _RUN_KEYS:
        out.extend(_read_run_key(*spec))
    out.extend(_read_folders())
    if include_tasks and runner is not None:
        try:
            out.extend(_read_tasks(runner))
        except Exception:
            pass
    out.sort(key=lambda e: (not e.enabled, e.location, e.name.lower()))
    return out


# =====================================================================
# 변경
# =====================================================================
def set_enabled(entry: StartupEntry, enabled: bool, runner: Runner | None = None) -> tuple[bool, str]:
    if entry.needs_admin and not is_admin():
        return False, "관리자 권한 필요"

    if entry.kind == "task":
        runner = runner or Runner()
        verb = "Enable-ScheduledTask" if enabled else "Disable-ScheduledTask"
        r = runner.powershell(
            f"{verb} {_task_args(entry)} -ErrorAction Stop | Out-Null", timeout=45)
        if r.ok:
            audit("startup.task", f"{entry.name} -> {'enabled' if enabled else 'disabled'}")
            return True, ""
        return False, _first_line(r)

    value_name = entry.name if entry.kind == "registry" else Path(entry.source).name
    if not entry.approved_key:
        return False, "이 위치는 사용 안 함을 지원하지 않습니다 (RunOnce)"
    if _set_approved(entry.hive, entry.approved_key, value_name, enabled):
        audit("startup.flag", f"{entry.location}\\{entry.name} -> "
                              f"{'enabled' if enabled else 'disabled'}")
        return True, ""
    return False, "레지스트리 쓰기 실패"


def _ps_quote(text: str) -> str:
    """PowerShell 작은따옴표 문자열. 큰따옴표는 $ 가 변수로 풀려 이름에 따라 깨진다."""
    return "'" + (text or "").replace("'", "''") + "'"


def _task_args(entry: StartupEntry) -> str:
    return f"-TaskName {_ps_quote(entry.name)} -TaskPath {_ps_quote(entry.task_path or chr(92))}"


def _first_line(r) -> str:
    text = (r.err or r.out or "").strip()
    return text.splitlines()[0][:200] if text else str(r)


def delete_task(entry: StartupEntry, runner: Runner | None = None) -> tuple[bool, str]:
    """작업 스케줄러 항목 삭제. 먼저 XML 로 내보내 백업하고, 백업에 실패하면 지우지 않는다.

    되돌리기:  Register-ScheduledTask -Xml (Get-Content '<백업.xml>' -Raw) -TaskName '<이름>'
    """
    runner = runner or Runner()
    exported = runner.powershell(f"Export-ScheduledTask {_task_args(entry)} -ErrorAction Stop",
                                 timeout=60)
    if not exported.ok or "<Task" not in exported.out:
        return False, "작업 정의를 백업하지 못해 삭제하지 않았습니다: " + _first_line(exported)
    try:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r'[\\/:*?"<>|]+', "_", entry.name)[:60]
        xml_path = BACKUP_DIR / f"task_{safe}_{datetime.now():%Y%m%d_%H%M%S}.xml"
        xml_path.write_text(exported.out, encoding="utf-16")   # Export-ScheduledTask 형식과 동일
    except OSError as e:
        return False, f"작업 정의를 백업하지 못해 삭제하지 않았습니다: {e}"

    removed = runner.powershell(
        f"Unregister-ScheduledTask {_task_args(entry)} -Confirm:$false -ErrorAction Stop",
        timeout=60)
    if removed.ok:
        audit("startup.task.delete", f"{entry.task_path}{entry.name} :: backup={xml_path}")
        return True, str(xml_path)
    return False, _first_line(removed)


def delete_entry(entry: StartupEntry, runner: Runner | None = None) -> tuple[bool, str]:
    if entry.needs_admin and not is_admin():
        return False, "관리자 권한 필요"

    if entry.kind == "folder":
        moved, failed = send_to_recycle_bin([entry.source])
        if moved:
            audit("startup.delete", entry.source)
            return True, ""
        return False, "휴지통으로 옮기지 못했습니다"

    if entry.kind == "task":
        # 예전엔 작업 스케줄러 항목을 삭제 대상에서 조용히 빼서,
        # 선택하고 삭제를 누르면 '선택한 항목이 없습니다'라고 떴다
        return delete_task(entry, runner)

    try:
        with winreg.OpenKey(entry.hive, entry.subkey, 0, winreg.KEY_SET_VALUE) as k:
            winreg.DeleteValue(k, entry.name)
        audit("startup.delete", f"{entry.location}\\{entry.name} = {entry.command}")
        return True, ""
    except FileNotFoundError:
        return True, ""
    except OSError as e:
        return False, str(e)


def backup_entries(entries: list[StartupEntry]):
    payload = [{
        "name": e.name, "command": e.command, "location": e.location,
        "enabled": e.enabled, "source": e.source, "kind": e.kind,
    } for e in entries]
    return save_backup("startup", payload)
