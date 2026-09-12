# -*- coding: utf-8 -*-
"""설치된 프로그램 목록 / 제거.

레지스트리의 Uninstall 키를 직접 읽는다. '프로그램 추가/제거' 와 같은 소스라
제어판과 목록이 일치하고, 크기 순 정렬이 가능해 용량 확보에 쓰기 좋다.
"""

from __future__ import annotations

import os
import subprocess
import winreg
from dataclasses import dataclass
from datetime import datetime

from .core import CREATE_NO_WINDOW, audit

_UNINSTALL_KEYS = [
    (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall", "64"),
    (winreg.HKEY_LOCAL_MACHINE,
     r"SOFTWARE\Wow6432Node\Microsoft\Windows\CurrentVersion\Uninstall", "32"),
    (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall", "user"),
]


@dataclass
class Program:
    name: str
    version: str = ""
    publisher: str = ""
    install_date: str = ""
    size_bytes: int = 0
    uninstall_cmd: str = ""
    quiet_cmd: str = ""
    install_location: str = ""
    arch: str = ""
    key_name: str = ""

    @property
    def removable(self) -> bool:
        return bool(self.uninstall_cmd or self.quiet_cmd)


def _get(key, name, default=""):
    try:
        value, _ = winreg.QueryValueEx(key, name)
        return value
    except OSError:
        return default


def _fmt_date(raw) -> str:
    s = str(raw or "").strip()
    if len(s) == 8 and s.isdigit():
        try:
            return datetime.strptime(s, "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            return s
    return s


def list_programs() -> list[Program]:
    seen: dict[str, Program] = {}
    for hive, path, arch in _UNINSTALL_KEYS:
        try:
            root = winreg.OpenKey(hive, path, 0, winreg.KEY_READ)
        except OSError:
            continue
        with root:
            n_sub = winreg.QueryInfoKey(root)[0]
            for i in range(n_sub):
                try:
                    sub_name = winreg.EnumKey(root, i)
                except OSError:
                    continue
                try:
                    with winreg.OpenKey(root, sub_name, 0, winreg.KEY_READ) as k:
                        name = str(_get(k, "DisplayName") or "").strip()
                        if not name:
                            continue
                        # 업데이트 패치/시스템 구성요소는 목록을 어지럽히므로 제외
                        if _get(k, "SystemComponent", 0) == 1:
                            continue
                        if _get(k, "ParentKeyName") or _get(k, "ReleaseType") in (
                                "Security Update", "Update Rollup", "Hotfix"):
                            continue
                        est = _get(k, "EstimatedSize", 0)
                        try:
                            size = int(est) * 1024
                        except (TypeError, ValueError):
                            size = 0
                        prog = Program(
                            name=name,
                            version=str(_get(k, "DisplayVersion") or ""),
                            publisher=str(_get(k, "Publisher") or ""),
                            install_date=_fmt_date(_get(k, "InstallDate")),
                            size_bytes=size,
                            uninstall_cmd=str(_get(k, "UninstallString") or ""),
                            quiet_cmd=str(_get(k, "QuietUninstallString") or ""),
                            install_location=str(_get(k, "InstallLocation") or ""),
                            arch=arch,
                            key_name=sub_name,
                        )
                        dedup = f"{prog.name}|{prog.version}"
                        if dedup not in seen or seen[dedup].size_bytes < prog.size_bytes:
                            seen[dedup] = prog
                except OSError:
                    continue

    out = list(seen.values())
    out.sort(key=lambda p: (-p.size_bytes, p.name.lower()))
    return out


def uninstall(prog: Program, quiet: bool = False) -> tuple[bool, str]:
    """제거 프로그램을 띄운다. 이후 진행은 사용자가 마법사에서 직접 한다."""
    cmd = (prog.quiet_cmd if (quiet and prog.quiet_cmd) else prog.uninstall_cmd).strip()
    if not cmd:
        return False, "제거 명령이 등록되어 있지 않습니다"

    # MSI 는 /I 를 /X 로 바꿔야 제거가 된다
    if "msiexec" in cmd.lower():
        cmd = cmd.replace("/I{", "/X{").replace("/i{", "/x{").replace("/I ", "/X ")

    try:
        # 제거 명령은 인용/인자 형식이 제각각이라 셸에 그대로 넘기는 편이 안전하다.
        # 레지스트리에서 읽은 값이고 사용자가 확인 후 실행한다.
        subprocess.Popen(cmd, shell=True, creationflags=CREATE_NO_WINDOW)
        audit("program.uninstall", f"{prog.name} :: {cmd}")
        return True, ""
    except OSError as e:
        return False, str(e)


def open_location(prog: Program) -> bool:
    loc = (prog.install_location or "").strip().strip('"')
    if loc and os.path.isdir(loc):
        try:
            os.startfile(loc)
            return True
        except OSError:
            return False
    return False


def total_size(progs: list[Program]) -> int:
    return sum(p.size_bytes for p in progs)
