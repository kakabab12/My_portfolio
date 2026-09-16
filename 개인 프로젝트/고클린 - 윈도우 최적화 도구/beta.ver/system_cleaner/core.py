# -*- coding: utf-8 -*-
"""공통 기반: 권한, 설정, 프로세스 실행, 단위 변환, 휴지통, 복원 지점, 감사 로그."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
import threading
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

CREATE_NO_WINDOW = 0x08000000


# =====================================================================
# 관리자 권한
# =====================================================================
def is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def is_frozen() -> bool:
    """PyInstaller 로 빌드된 exe 로 실행 중인가."""
    return bool(getattr(sys, "frozen", False))


def elevation_command() -> tuple[str, str]:
    """(실행 파일, 인자 문자열). 테스트할 수 있게 relaunch_as_admin 에서 분리했다.

    ' '.join(sys.argv) 로 인자를 만들면 경로에 공백이 있을 때 쪼개진다.
    exe 로 빌드하면 sys.executable 이 exe 자신이라, 스크립트 경로를 끼워 넣으면
    exe 가 자기 경로를 인자로 받는 이상한 실행이 된다.
    """
    if is_frozen():
        return sys.executable, subprocess.list2cmdline(sys.argv[1:])
    return sys.executable, subprocess.list2cmdline(
        [os.path.abspath(sys.argv[0])] + sys.argv[1:])


def relaunch_as_admin() -> bool:
    """UAC 승격 재실행. 성공하면 True (호출부에서 현재 프로세스를 끝내야 한다)."""
    try:
        target, params = elevation_command()
        rc = ctypes.windll.shell32.ShellExecuteW(None, "runas", target, params, os.getcwd(), 1)
        return int(rc) > 32   # 사용자가 UAC 를 취소하면 5(SE_ERR_ACCESSDENIED)
    except Exception:
        return False


# 아이콘 등 패키지 안의 리소스. PyInstaller 도 모듈의 __file__ 을 _MEIPASS 기준으로
# 채워주므로 소스 실행과 exe 실행에서 같은 코드로 찾을 수 있다.
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
ICON_PATH = ASSETS_DIR / "icon.ico"


# =====================================================================
# 경로 / 설정
# =====================================================================
DATA_DIR = Path(os.environ.get("APPDATA", Path.home())) / "SystemCleaner"
CONFIG_PATH = DATA_DIR / "config.json"
BACKUP_DIR = DATA_DIR / "backup"
LOG_DIR = DATA_DIR / "logs"

DEFAULT_CONFIG = {
    "lang": "EN",
    "theme": "Dark",

    # 청소 옵션
    "opt_prefetch": False,        # 지우면 다음 부팅이 잠깐 느려짐
    "opt_recycle_bin": False,     # 영구 삭제라 기본 OFF
    "opt_browser_cache": True,
    "opt_dism": True,
    "opt_defrag": True,
    "opt_winsock_reset": False,   # 재부팅 필요 + VPN/프록시 설정이 깨질 수 있음
    "opt_wu_cache": True,

    # 게임 모드
    "opt_kill_office": False,     # 오피스/한글 강제 종료 = 문서 손실 위험
    "opt_game_power": True,       # 고성능 전원 관리 옵션으로 전환

    # 안전 장치
    "opt_restore_point": True,    # 위험 작업 전 복원 지점 생성
    "opt_confirm_all": True,      # 파괴적 작업 전 항상 확인
    "opt_backup_before_change": True,

    # 파일 검사
    "large_file_min_mb": 100,
    "scan_root": "",

    "donation_bmc": "",
    "donation_account": "",
}


class Config:
    def __init__(self):
        self.data = dict(DEFAULT_CONFIG)
        self.load()

    def load(self):
        try:
            if CONFIG_PATH.exists():
                saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                if isinstance(saved, dict):
                    self.data.update({k: v for k, v in saved.items() if k in DEFAULT_CONFIG})
        except Exception:
            pass  # 설정이 깨져도 기본값으로 계속 동작

    def save(self):
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            CONFIG_PATH.write_text(json.dumps(self.data, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        except Exception:
            pass

    def __getitem__(self, key):
        return self.data.get(key, DEFAULT_CONFIG.get(key))

    def __setitem__(self, key, value):
        self.data[key] = value
        self.save()


# =====================================================================
# 감사 로그 - 무엇을 바꿨는지 파일로 남긴다
# =====================================================================
_log_lock = threading.Lock()


def audit(action: str, detail: str = ""):
    """되돌릴 필요가 생겼을 때 볼 수 있도록 변경 이력을 남긴다."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S}\t{action}\t{detail}\n"
        with _log_lock:
            with open(LOG_DIR / f"audit_{datetime.now():%Y%m}.log", "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        pass


# 이번 실행에서 기록된 오류 (자가 검사가 '조용히 삼켜진 오류'까지 잡아내는 데 쓴다)
ERROR_RECORDS: list[str] = []


def log_error(where: str, exc: BaseException | None = None, tb_text: str = "") -> Path | None:
    """예외를 파일로 남긴다.

    console=False 로 빌드한 exe 는 sys.stderr 가 None 이라, 버튼 콜백에서 난 예외가
    화면에도 콘솔에도 안 남고 그냥 사라진다. 사용자가 '가끔 버튼이 안 먹는다'고만
    느끼게 되므로 반드시 파일로 남긴다.
    """
    import traceback

    if not tb_text and exc is not None:
        tb_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    ERROR_RECORDS.append(f"{where}: {tb_text.strip().splitlines()[-1] if tb_text else ''}")
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = LOG_DIR / f"error_{datetime.now():%Y%m}.log"
        with _log_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"===== {datetime.now():%Y-%m-%d %H:%M:%S}  [{where}]  "
                        f"frozen={is_frozen()}  admin={is_admin()}\n{tb_text}\n")
        return path
    except Exception:
        return None


def save_backup(name: str, payload) -> Path | None:
    """변경 전 상태를 JSON 으로 백업."""
    try:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        path = BACKUP_DIR / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
        return path
    except Exception:
        return None


# =====================================================================
# 콘솔 인코딩 / 프로세스 실행
# =====================================================================
def _console_encodings() -> list[str]:
    encs = []
    try:
        encs.append(f"cp{ctypes.windll.kernel32.GetOEMCP()}")
    except Exception:
        pass
    encs += ["cp949", "utf-8", "mbcs"]
    seen, out = set(), []
    for e in encs:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


_ENCODINGS = _console_encodings()


def decode_console(data: bytes) -> str:
    for enc in _ENCODINGS:
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode(_ENCODINGS[0], errors="replace")


@dataclass
class Proc:
    rc: int = -1
    out: str = ""
    err: str = ""
    timed_out: bool = False
    not_found: bool = False

    @property
    def ok(self) -> bool:
        return self.rc == 0 and not self.timed_out and not self.not_found

    def __str__(self) -> str:
        if self.not_found:
            return "not found"
        if self.timed_out:
            return "timeout"
        return f"rc={self.rc}"


class Runner:
    """실행 중인 자식 프로세스를 들고 있다가 취소 시 죽인다.

    subprocess.run() 은 종료 코드가 0 이 아니어도 예외를 던지지 않는다.
    성공/실패를 실제 rc 로 판정하기 위해 얇게 감쌌다.
    """

    def __init__(self):
        self._current: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def run(self, args, timeout: float = 120.0, shell: bool = False) -> Proc:
        if not shell and isinstance(args, (list, tuple)) and args:
            if shutil.which(str(args[0])) is None:
                return Proc(rc=-1, not_found=True, err=f"'{args[0]}' not found in PATH")
        try:
            p = subprocess.Popen(
                args, shell=shell,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                creationflags=CREATE_NO_WINDOW,
            )
        except FileNotFoundError as e:
            return Proc(rc=-1, not_found=True, err=str(e))
        except OSError as e:
            return Proc(rc=-1, err=str(e))

        with self._lock:
            self._current = p
        try:
            out, err = p.communicate(timeout=timeout)
            return Proc(p.returncode, decode_console(out).strip(), decode_console(err).strip())
        except subprocess.TimeoutExpired:
            self._kill(p)
            return Proc(rc=-1, timed_out=True, err="timeout")
        finally:
            with self._lock:
                self._current = None

    @staticmethod
    def _kill(p: subprocess.Popen):
        try:
            p.kill()
            p.communicate(timeout=5)
        except Exception:
            pass

    def cancel_current(self):
        with self._lock:
            p = self._current
        if p is not None:
            self._kill(p)

    def powershell(self, script: str, timeout: float = 60.0) -> Proc:
        return self.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-Command", script],
            timeout=timeout)

    def ps_json(self, script: str, timeout: float = 60.0, default=None):
        """PowerShell 결과를 JSON 으로 받는다. 단일 객체도 항상 list 로 정규화."""
        r = self.powershell(script, timeout=timeout)
        if not r.ok or not r.out.strip():
            return default if default is not None else []
        try:
            data = json.loads(r.out)
        except json.JSONDecodeError:
            return default if default is not None else []
        if data is None:
            return default if default is not None else []
        return data if isinstance(data, list) else [data]


# 공용 러너 (UI 스레드에서 가벼운 조회를 할 때)
shared_runner = Runner()


# =====================================================================
# 단위 / 폭 계산
# =====================================================================
def human_bytes(n) -> str:
    if n is None:
        return "-"
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "-"
    neg = n < 0
    n = abs(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0:
            s = f"{n:,.0f} B" if unit == "B" else f"{n:,.1f} {unit}"
            return ("-" + s) if neg else s
        n /= 1024.0
    s = f"{n:,.1f} PB"
    return ("-" + s) if neg else s


def _is_wide(ch: str) -> bool:
    o = ord(ch)
    return (0x1100 <= o <= 0x115F or 0x2E80 <= o <= 0xA4CF or 0xAC00 <= o <= 0xD7A3
            or 0xF900 <= o <= 0xFAFF or 0xFE30 <= o <= 0xFE6F or 0xFF00 <= o <= 0xFF60
            or 0xFFE0 <= o <= 0xFFE6)


def display_width(s: str) -> int:
    """한글 등 전각 문자를 2칸으로 계산 (터미널 리포트 정렬용)."""
    return sum(2 if _is_wide(ch) else 1 for ch in s)


def ellipsis(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 1)] + "…"


# =====================================================================
# 보호 경로
# =====================================================================
_PROTECTED_RAW = [
    "C:/", "C:/Windows", "C:/Windows/System32", "C:/Windows/SysWOW64",
    "C:/Users", "C:/Program Files", "C:/Program Files (x86)", "C:/ProgramData",
]


def _protected_set() -> set[Path]:
    out = set()
    for p in _PROTECTED_RAW:
        try:
            out.add(Path(p).resolve())
        except OSError:
            pass
    try:
        out.add(Path.home().resolve())
    except OSError:
        pass
    return out


def is_protected(path: Path) -> bool:
    """삭제하면 안 되는 경로인지. 해석 실패 시 안전하게 True."""
    try:
        resolved = Path(path).resolve()
    except OSError:
        return True
    if resolved.parent == resolved:   # 드라이브 루트
        return True
    return resolved in _protected_set()


# =====================================================================
# 휴지통으로 보내기 (영구 삭제 대신)
# =====================================================================
class _SHFILEOPSTRUCTW(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("wFunc", wintypes.UINT),
        ("pFrom", wintypes.LPCWSTR),
        ("pTo", wintypes.LPCWSTR),
        ("fFlags", ctypes.c_uint16),
        ("fAnyOperationsAborted", wintypes.BOOL),
        ("hNameMappings", ctypes.c_void_p),
        ("lpszProgressTitle", wintypes.LPCWSTR),
    ]


_FO_DELETE = 3
_FOF_ALLOWUNDO = 0x0040
_FOF_NOCONFIRMATION = 0x0010
_FOF_SILENT = 0x0004
_FOF_NOERRORUI = 0x0400


def send_to_recycle_bin(paths: list[str | Path]) -> tuple[int, list[str]]:
    """휴지통으로 이동. (성공 개수, 실패 경로들).

    영구 삭제가 아니라 되돌릴 수 있게 휴지통을 쓴다.
    """
    valid = []
    for p in paths:
        try:
            path = Path(p)
            if path.exists() and not is_protected(path):
                valid.append(str(path.resolve()))
        except OSError:
            continue
    if not valid:
        return 0, [str(p) for p in paths]

    op = _SHFILEOPSTRUCTW()
    op.hwnd = None
    op.wFunc = _FO_DELETE
    op.pFrom = "\0".join(valid) + "\0\0"
    op.pTo = None
    op.fFlags = _FOF_ALLOWUNDO | _FOF_NOCONFIRMATION | _FOF_SILENT | _FOF_NOERRORUI
    op.fAnyOperationsAborted = False
    op.hNameMappings = None
    op.lpszProgressTitle = None

    try:
        rc = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    except Exception:
        return 0, valid
    if rc != 0 or op.fAnyOperationsAborted:
        return 0, valid
    audit("recycle_bin.move", f"{len(valid)} items")
    return len(valid), []


def empty_recycle_bin() -> bool:
    """휴지통 비우기 (영구 삭제). 호출 전에 반드시 사용자 확인을 받을 것."""
    SHERB_NOCONFIRMATION, SHERB_NOPROGRESSUI, SHERB_NOSOUND = 0x1, 0x2, 0x4
    try:
        rc = ctypes.windll.shell32.SHEmptyRecycleBinW(
            None, None, SHERB_NOCONFIRMATION | SHERB_NOPROGRESSUI | SHERB_NOSOUND)
    except Exception:
        return False
    ok = (rc & 0xFFFFFFFF) in (0, 0x8000010A)   # S_OK 또는 '이미 비어 있음'
    if ok:
        audit("recycle_bin.empty")
    return ok


def recycle_bin_size() -> tuple[int, int]:
    """(바이트, 항목 수)."""
    class SHQUERYRBINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("i64Size", ctypes.c_int64),
                    ("i64NumItems", ctypes.c_int64)]
    info = SHQUERYRBINFO()
    info.cbSize = ctypes.sizeof(SHQUERYRBINFO)
    try:
        rc = ctypes.windll.shell32.SHQueryRecycleBinW(None, ctypes.byref(info))
        if (rc & 0xFFFFFFFF) == 0:
            return int(info.i64Size), int(info.i64NumItems)
    except Exception:
        pass
    return 0, 0


# =====================================================================
# 시스템 복원 지점
# =====================================================================
def create_restore_point(runner: Runner, description: str = "SystemCleaner") -> tuple[bool, str]:
    """복원 지점 생성. (성공, 메시지).

    윈도우는 기본적으로 24시간에 1개만 만들게 제한한다. 최적화 도구 입장에서는
    '만들었다고 생각했는데 안 만들어진' 상태가 제일 위험하므로, 제한을 0으로
    낮춘 뒤 생성하고 그 사실을 로그에 남긴다.
    """
    if not is_admin():
        return False, "관리자 권한 필요"

    runner.run(["reg", "add",
                r"HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\SystemRestore",
                "/v", "SystemRestorePointCreationFrequency", "/t", "REG_DWORD",
                "/d", "0", "/f"], timeout=20)

    stamp = f"{description} {datetime.now():%Y-%m-%d %H:%M}"
    r = runner.powershell(
        f'Checkpoint-Computer -Description "{stamp}" -RestorePointType "MODIFY_SETTINGS"',
        timeout=180)
    if r.ok:
        audit("restore_point.create", stamp)
        return True, stamp
    msg = (r.err or r.out or str(r)).strip().splitlines()
    detail = msg[0] if msg else str(r)
    if "사용" in detail or "disabled" in detail.lower() or "0x8004" in detail:
        detail += "  (시스템 보호가 꺼져 있을 수 있습니다)"
    return False, detail


def open_restore_ui():
    try:
        os.startfile("rstrui.exe")
        return True
    except OSError:
        return False


def open_in_explorer(path: str | Path) -> bool:
    """탐색기에서 해당 항목을 선택한 상태로 연다."""
    try:
        p = Path(path)
        if p.is_dir():
            os.startfile(str(p))
        else:
            subprocess.Popen(["explorer", "/select,", str(p)], creationflags=CREATE_NO_WINDOW)
        return True
    except Exception:
        return False
