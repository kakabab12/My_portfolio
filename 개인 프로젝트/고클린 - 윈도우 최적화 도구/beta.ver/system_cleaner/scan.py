# -*- coding: utf-8 -*-
"""파일 검사/삭제: 임시 파일, 브라우저 캐시, 대용량 파일, 중복 파일."""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .core import human_bytes, is_admin, is_protected, recycle_bin_size


# =====================================================================
# 공통 트리 순회
# =====================================================================
def scan_tree(path: Path, cancel: threading.Event | None = None) -> tuple[int, int]:
    """(바이트, 파일 수). 삭제하지 않고 크기만 센다."""
    total = count = 0
    stack = [Path(path)]
    while stack:
        if cancel is not None and cancel.is_set():
            break
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                            count += 1
                    except OSError:
                        continue
        except OSError:
            continue
    return total, count


def purge_tree(path: Path, cancel: threading.Event | None = None) -> tuple[int, int, int]:
    """(확보 바이트, 삭제 파일 수, 건너뛴 파일 수). 대상 폴더 자체는 남긴다."""
    path = Path(path)
    if is_protected(path) or not path.exists():
        return 0, 0, 0

    freed = deleted = skipped = 0
    dirs_seen: list[Path] = []
    stack = [path]
    while stack:
        if cancel is not None and cancel.is_set():
            break
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for entry in it:
                    if cancel is not None and cancel.is_set():
                        break
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                            dirs_seen.append(Path(entry.path))
                        else:
                            size = entry.stat(follow_symlinks=False).st_size
                            try:
                                os.chmod(entry.path, 0o666)
                            except OSError:
                                pass
                            os.remove(entry.path)
                            freed += size
                            deleted += 1
                    except OSError:
                        skipped += 1   # 사용 중인 파일은 조용히 넘어간다
        except OSError:
            continue

    for d in sorted(dirs_seen, key=lambda p: len(p.parts), reverse=True):
        try:
            d.rmdir()
        except OSError:
            pass
    return freed, deleted, skipped


def _env_path(var: str, *parts) -> Path | None:
    base = os.environ.get(var)
    if not base:
        return None
    return Path(base).joinpath(*parts)


# =====================================================================
# 청소 대상
# =====================================================================
@dataclass
class CleanTarget:
    key: str
    label: str
    path: Path | None = None
    admin_only: bool = False
    optional_key: str = ""        # 설정에서 켜야 포함되는 항목
    special: str = ""             # 'recycle_bin' 처럼 경로가 없는 항목
    size: int = 0
    files: int = 0
    selected: bool = True

    @property
    def where(self) -> str:
        return str(self.path) if self.path else "-"


_TEMP_SPECS = [
    # (key, 표시 이름, 경로, 관리자 전용, 옵션 키)
    ("temp_user",   "사용자 임시 폴더",        lambda: _env_path("TEMP"), False, ""),
    ("temp_win",    "Windows 임시 폴더",       lambda: Path("C:/Windows/Temp"), True, ""),
    ("crashdumps",  "크래시 덤프",             lambda: _env_path("LOCALAPPDATA", "CrashDumps"), False, ""),
    ("wer_queue",   "오류 보고 대기",          lambda: _env_path("PROGRAMDATA", "Microsoft", "Windows", "WER", "ReportQueue"), True, ""),
    ("wer_archive", "오류 보고 보관",          lambda: _env_path("PROGRAMDATA", "Microsoft", "Windows", "WER", "ReportArchive"), True, ""),
    ("inetcache",   "인터넷 임시 파일",        lambda: _env_path("LOCALAPPDATA", "Microsoft", "Windows", "INetCache"), False, ""),
    ("thumbcache",  "썸네일 캐시",             lambda: _env_path("LOCALAPPDATA", "Microsoft", "Windows", "Explorer"), False, ""),
    ("wu_cache",    "Windows 업데이트 캐시",   lambda: Path("C:/Windows/SoftwareDistribution/Download"), True, "opt_wu_cache"),
    ("deliv_opt",   "배달 최적화 캐시",        lambda: Path("C:/Windows/SoftwareDistribution/DeliveryOptimization"), True, "opt_wu_cache"),
    ("prefetch",    "Prefetch",                lambda: Path("C:/Windows/Prefetch"), True, "opt_prefetch"),
]

_TEMP_LABELS_EN = {
    "temp_user": "User Temp", "temp_win": "Windows Temp", "crashdumps": "Crash Dumps",
    "wer_queue": "Error Report Queue", "wer_archive": "Error Report Archive",
    "inetcache": "Internet Cache", "thumbcache": "Thumbnail Cache",
    "wu_cache": "Windows Update Cache", "deliv_opt": "Delivery Optimization",
    "prefetch": "Prefetch",
}


# --- 브라우저 캐시 ----------------------------------------------------
# 비밀번호(Login Data) / 쿠키 / 방문 기록 / 즐겨찾기는 절대 포함하지 않는다.
_CHROMIUM_BROWSERS = [
    ("chrome",  "Google Chrome", ("LOCALAPPDATA", "Google", "Chrome", "User Data")),
    ("edge",    "Microsoft Edge", ("LOCALAPPDATA", "Microsoft", "Edge", "User Data")),
    ("whale",   "Naver Whale",   ("LOCALAPPDATA", "Naver", "Naver Whale", "User Data")),
    ("brave",   "Brave",         ("LOCALAPPDATA", "BraveSoftware", "Brave-Browser", "User Data")),
    ("opera",   "Opera",         ("APPDATA", "Opera Software", "Opera Stable")),
]
_CHROMIUM_CACHE_DIRS = [
    "Cache", "Code Cache", "GPUCache", "ShaderCache", "GrShaderCache",
    os.path.join("Service Worker", "CacheStorage"),
    os.path.join("Service Worker", "ScriptCache"),
]


def browser_cache_targets() -> list[CleanTarget]:
    out: list[CleanTarget] = []
    for key, label, (var, *parts) in _CHROMIUM_BROWSERS:
        root = _env_path(var, *parts)
        if root is None or not root.exists():
            continue
        # 프로필 폴더: Default, Profile 1, ...
        profiles = [root]
        try:
            for entry in os.scandir(root):
                if entry.is_dir() and (entry.name == "Default"
                                       or entry.name.startswith("Profile ")):
                    profiles.append(Path(entry.path))
        except OSError:
            pass
        for prof in profiles:
            for sub in _CHROMIUM_CACHE_DIRS:
                p = prof / sub
                if p.exists():
                    out.append(CleanTarget(f"br_{key}_{prof.name}_{sub}",
                                           f"{label} · {prof.name} · {sub}", p))

    # Firefox
    ff = _env_path("LOCALAPPDATA", "Mozilla", "Firefox", "Profiles")
    if ff and ff.exists():
        try:
            for entry in os.scandir(ff):
                cache = Path(entry.path) / "cache2"
                if cache.exists():
                    out.append(CleanTarget(f"br_firefox_{entry.name}",
                                           f"Mozilla Firefox · {entry.name}", cache))
        except OSError:
            pass
    return out


def build_clean_targets(cfg, lang: str = "KO", include_browser: bool = True) -> list[CleanTarget]:
    """설정과 권한을 반영한 청소 대상 목록 (아직 크기는 안 잼)."""
    out: list[CleanTarget] = []
    admin = is_admin()
    for key, label_ko, getter, admin_only, opt_key in _TEMP_SPECS:
        if admin_only and not admin:
            continue
        if opt_key and not cfg[opt_key]:
            continue
        try:
            path = getter()
        except Exception:
            path = None
        if path is None or not path.exists() or is_protected(path):
            continue
        label = label_ko if lang == "KO" else _TEMP_LABELS_EN.get(key, key)
        out.append(CleanTarget(key, label, path, admin_only, opt_key))

    if include_browser and cfg["opt_browser_cache"]:
        out.extend(browser_cache_targets())

    size, items = recycle_bin_size()
    if items > 0 and cfg["opt_recycle_bin"]:
        rb = CleanTarget("recycle_bin",
                         "휴지통" if lang == "KO" else "Recycle Bin",
                         None, False, "opt_recycle_bin", special="recycle_bin")
        rb.size, rb.files = size, items
        out.append(rb)
    return out


def measure_targets(targets: list[CleanTarget], cancel: threading.Event | None = None,
                    progress=None) -> int:
    """각 대상의 크기를 재고 총합을 돌려준다."""
    total = 0
    for i, tgt in enumerate(targets):
        if cancel is not None and cancel.is_set():
            break
        if tgt.special == "recycle_bin":
            total += tgt.size
        elif tgt.path is not None:
            tgt.size, tgt.files = scan_tree(tgt.path, cancel)
            total += tgt.size
        if progress is not None:
            progress((i + 1) / max(1, len(targets)), tgt)
    return total


# =====================================================================
# 대용량 파일
# =====================================================================
@dataclass
class FileHit:
    path: str
    size: int
    mtime: float


_SKIP_DIRS = {
    "windows", "$recycle.bin", "system volume information", "programdata",
    "winsxs", "$windows.~ws", "$windows.~bt", "recovery",
}


def find_large_files(root: Path, min_bytes: int, cancel: threading.Event | None = None,
                     limit: int = 500, progress=None) -> list[FileHit]:
    """min_bytes 이상인 파일을 찾는다. 시스템 폴더는 건너뛴다."""
    hits: list[FileHit] = []
    stack = [Path(root)]
    seen_dirs = 0
    while stack:
        if cancel is not None and cancel.is_set():
            break
        cur = stack.pop()
        seen_dirs += 1
        if progress is not None and seen_dirs % 200 == 0:
            progress(cur, len(hits))
        try:
            with os.scandir(cur) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name.lower() in _SKIP_DIRS:
                                continue
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            st = entry.stat(follow_symlinks=False)
                            if st.st_size >= min_bytes:
                                hits.append(FileHit(entry.path, st.st_size, st.st_mtime))
                    except OSError:
                        continue
        except OSError:
            continue
    hits.sort(key=lambda h: h.size, reverse=True)
    return hits[:limit]


# =====================================================================
# 중복 파일
# =====================================================================
@dataclass
class DupeGroup:
    size: int
    paths: list[str] = field(default_factory=list)

    @property
    def wasted(self) -> int:
        return self.size * max(0, len(self.paths) - 1)


def _hash_file(path: str, chunk_limit: int | None = None) -> str | None:
    h = hashlib.blake2b(digest_size=16)
    read = 0
    try:
        with open(path, "rb") as f:
            while True:
                block = f.read(1 << 20)
                if not block:
                    break
                h.update(block)
                read += len(block)
                if chunk_limit is not None and read >= chunk_limit:
                    break
    except OSError:
        return None
    return h.hexdigest()


def find_duplicates(root: Path, min_bytes: int = 1 << 20,
                    cancel: threading.Event | None = None,
                    progress=None, max_groups: int = 300) -> list[DupeGroup]:
    """크기 -> 앞부분 해시 -> 전체 해시 3단계로 좁힌다."""
    by_size: dict[int, list[str]] = {}
    stack = [Path(root)]
    scanned = 0
    while stack:
        if cancel is not None and cancel.is_set():
            return []
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name.lower() in _SKIP_DIRS:
                                continue
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            st = entry.stat(follow_symlinks=False)
                            if st.st_size >= min_bytes:
                                by_size.setdefault(st.st_size, []).append(entry.path)
                                scanned += 1
                                if progress is not None and scanned % 500 == 0:
                                    progress("scan", scanned, 0)
                    except OSError:
                        continue
        except OSError:
            continue

    candidates = {s: ps for s, ps in by_size.items() if len(ps) > 1}

    # 2단계: 앞 64KB 해시
    stage2: dict[tuple[int, str], list[str]] = {}
    done = 0
    for size, paths in candidates.items():
        if cancel is not None and cancel.is_set():
            return []
        for p in paths:
            h = _hash_file(p, chunk_limit=64 * 1024)
            if h:
                stage2.setdefault((size, h), []).append(p)
            done += 1
            if progress is not None and done % 100 == 0:
                progress("hash1", done, sum(len(v) for v in candidates.values()))

    # 3단계: 전체 해시
    groups: list[DupeGroup] = []
    stage3: dict[tuple[int, str], list[str]] = {}
    for (size, _), paths in stage2.items():
        if len(paths) < 2:
            continue
        if cancel is not None and cancel.is_set():
            return []
        for p in paths:
            h = _hash_file(p)
            if h:
                stage3.setdefault((size, h), []).append(p)

    for (size, _), paths in stage3.items():
        if len(paths) > 1:
            groups.append(DupeGroup(size, sorted(paths)))

    groups.sort(key=lambda g: g.wasted, reverse=True)
    return groups[:max_groups]


def summarize(total_bytes: int, n_files: int) -> str:
    return f"{human_bytes(total_bytes)} / {n_files:,}"
