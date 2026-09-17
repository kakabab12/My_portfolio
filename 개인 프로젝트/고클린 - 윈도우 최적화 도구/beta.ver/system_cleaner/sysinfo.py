# -*- coding: utf-8 -*-
"""하드웨어 정보, 디스크 상태(S.M.A.R.T.), 네트워크 / DNS.

wmic 은 Windows 11 24H2 부터 OS 에서 기본 제거되었다.
모든 조회는 PowerShell CIM 으로 한다.
"""

from __future__ import annotations

import re
import socket
import statistics
import urllib.request
from dataclasses import dataclass, field

from .core import Runner, audit, is_admin

try:
    import psutil
except ImportError:
    psutil = None


# =====================================================================
# 하드웨어 정보
# =====================================================================
_HW_SCRIPT = r"""
$ErrorActionPreference='SilentlyContinue'
$cs = Get-CimInstance Win32_ComputerSystem
$os = Get-CimInstance Win32_OperatingSystem
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$bb = Get-CimInstance Win32_BaseBoard
$bios = Get-CimInstance Win32_BIOS
$ramSticks = Get-CimInstance Win32_PhysicalMemory
[ordered]@{
  cpu      = $cpu.Name
  cores    = $cpu.NumberOfCores
  threads  = $cpu.NumberOfLogicalProcessors
  clock    = $cpu.MaxClockSpeed
  board    = ($bb.Manufacturer + ' ' + $bb.Product)
  bios     = ($bios.Manufacturer + ' ' + $bios.SMBIOSBIOSVersion)
  gpu      = ((Get-CimInstance Win32_VideoController).Name -join ', ')
  ram_gb   = [math]::Round($cs.TotalPhysicalMemory/1GB, 1)
  ram_type = (($ramSticks | ForEach-Object { "$($_.Capacity/1GB)GB@$($_.Speed)MHz" }) -join ' + ')
  os       = ($os.Caption + ' ' + $os.OSArchitecture)
  build    = $os.Version
  boot     = $os.LastBootUpTime.ToString('yyyy-MM-dd HH:mm')
  host     = $cs.Name
  user     = $cs.UserName
} | ConvertTo-Json -Compress
"""


def hardware_info(runner: Runner) -> dict:
    rows = runner.ps_json(_HW_SCRIPT, timeout=90)
    return rows[0] if rows else {}


# =====================================================================
# 디스크 상태
# =====================================================================
@dataclass
class DiskInfo:
    name: str = ""
    media: str = ""
    size: int = 0
    health: str = ""
    bus: str = ""
    temperature: int | None = None
    wear: int | None = None
    power_on_hours: int | None = None
    read_errors: int | None = None
    write_errors: int | None = None

    @property
    def healthy(self) -> bool:
        return (self.health or "").lower() in ("healthy", "정상", "")


@dataclass
class VolumeInfo:
    letter: str = ""
    label: str = ""
    fs: str = ""
    size: int = 0
    free: int = 0

    @property
    def used(self) -> int:
        return max(0, self.size - self.free)

    @property
    def percent_free(self) -> int:
        return int(self.free / self.size * 100) if self.size else 0


_DISK_SCRIPT = r"""
$ErrorActionPreference='SilentlyContinue'
Get-PhysicalDisk | ForEach-Object {
  $rc = $_ | Get-StorageReliabilityCounter
  [ordered]@{
    name  = $_.FriendlyName
    media = "$($_.MediaType)"
    bus   = "$($_.BusType)"
    size  = [int64]$_.Size
    health= "$($_.HealthStatus)"
    temp  = $rc.Temperature
    wear  = $rc.Wear
    hours = $rc.PowerOnHours
    rerr  = $rc.ReadErrorsTotal
    werr  = $rc.WriteErrorsTotal
  }
} | ConvertTo-Json -Compress
"""

_VOLUME_SCRIPT = r"""
$ErrorActionPreference='SilentlyContinue'
Get-Volume | Where-Object { $_.DriveLetter -and $_.DriveType -eq 'Fixed' } | ForEach-Object {
  [ordered]@{
    letter = "$($_.DriveLetter)"
    label  = "$($_.FileSystemLabel)"
    fs     = "$($_.FileSystem)"
    size   = [int64]$_.Size
    free   = [int64]$_.SizeRemaining
  }
} | ConvertTo-Json -Compress
"""


def _as_int(v):
    try:
        if v is None or v == "":
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def disk_info(runner: Runner) -> list[DiskInfo]:
    out = []
    for row in runner.ps_json(_DISK_SCRIPT, timeout=90):
        if not isinstance(row, dict):
            continue
        out.append(DiskInfo(
            name=row.get("name") or "?",
            media=row.get("media") or "",
            bus=row.get("bus") or "",
            size=_as_int(row.get("size")) or 0,
            health=row.get("health") or "",
            temperature=_as_int(row.get("temp")),
            wear=_as_int(row.get("wear")),
            power_on_hours=_as_int(row.get("hours")),
            read_errors=_as_int(row.get("rerr")),
            write_errors=_as_int(row.get("werr")),
        ))
    return out


def volume_info(runner: Runner) -> list[VolumeInfo]:
    out = []
    for row in runner.ps_json(_VOLUME_SCRIPT, timeout=60):
        if not isinstance(row, dict):
            continue
        out.append(VolumeInfo(
            letter=row.get("letter") or "",
            label=row.get("label") or "",
            fs=row.get("fs") or "",
            size=_as_int(row.get("size")) or 0,
            free=_as_int(row.get("free")) or 0,
        ))
    out.sort(key=lambda v: v.letter)
    return out


def volume_info_fast() -> list[VolumeInfo]:
    """psutil 로 빠르게 (대시보드 갱신용). PowerShell 을 띄우지 않는다."""
    if psutil is None:
        return []
    out = []
    for part in psutil.disk_partitions(all=False):
        if "cdrom" in part.opts or not part.fstype:
            continue
        try:
            u = psutil.disk_usage(part.mountpoint)
        except OSError:
            continue
        out.append(VolumeInfo(letter=part.device.rstrip(":\\"), label="",
                              fs=part.fstype, size=u.total, free=u.free))
    return out


# =====================================================================
# 디스크 최적화 대상 (SSD 는 TRIM, HDD 만 조각 모음)
# =====================================================================
_OPT_SCRIPT = r"""
$ErrorActionPreference='SilentlyContinue'
Get-Volume | Where-Object { $_.DriveLetter -and $_.DriveType -eq 'Fixed' } | ForEach-Object {
  $d = $_.DriveLetter
  $m = 'Unspecified'
  $p = Get-Partition -DriveLetter $d
  if ($p) {
    $disk = Get-PhysicalDisk | Where-Object { $_.DeviceId -eq $p.DiskNumber }
    if ($disk) { $m = "$($disk.MediaType)" }
  }
  [ordered]@{ drive = "$d"; media = $m }
} | ConvertTo-Json -Compress
"""


def optimize_plan(runner: Runner) -> list[tuple[str, str, str]]:
    """[(드라이브, 미디어, defrag 플래그)]. SSD=/L(retrim), HDD=/D(조각 모음)."""
    plan = []
    for row in runner.ps_json(_OPT_SCRIPT, timeout=90):
        if not isinstance(row, dict):
            continue
        drive = (row.get("drive") or "").strip()
        media = (row.get("media") or "Unspecified").strip()
        if not drive:
            continue
        flag = "/L" if media.upper() == "SSD" else "/D"
        plan.append((drive, media, flag))
    return plan


# =====================================================================
# 네트워크
# =====================================================================
DNS_SERVERS = [
    ("Google", "8.8.8.8", "8.8.4.4"),
    ("Cloudflare", "1.1.1.1", "1.0.0.1"),
    ("KT", "168.126.63.1", "168.126.63.2"),
    ("SK Broadband", "219.250.36.130", "210.220.163.82"),
    ("LG U+", "164.124.101.2", "203.248.252.2"),
    ("Quad9", "9.9.9.9", "149.112.112.112"),
]


@dataclass
class PingResult:
    label: str
    host: str
    avg: float | None = None
    lo: int | None = None
    hi: int | None = None
    jitter: float = 0.0
    loss: int = 100
    secondary: str = ""

    @property
    def ok(self) -> bool:
        return self.avg is not None

    def grade(self, lang: str = "EN") -> str:
        if self.avg is None:
            return "FAIL" if lang == "EN" else "실패"
        if self.avg <= 10:
            return "EXCELLENT" if lang == "EN" else "매우 좋음"
        if self.avg <= 30:
            return "GOOD" if lang == "EN" else "좋음"
        if self.avg <= 60:
            return "NORMAL" if lang == "EN" else "보통"
        return "SLOW" if lang == "EN" else "느림"


def ping(runner: Runner, host: str, label: str = "", count: int = 5) -> PingResult:
    """언어 설정과 무관하게 동작하도록 TTL 이 있는 응답 줄만 파싱한다.

    영문 'time=21ms', 한글 '시간=21ms', 'time<1ms' 를 모두 커버한다.
    요약 줄(최소/최대/평균)에 의존하지 않는다.
    """
    res = PingResult(label or host, host)
    r = runner.run(["ping", "-n", str(count), "-w", "1500", host], timeout=15 + count * 2)
    times: list[int] = []
    for line in r.out.splitlines():
        if "TTL" not in line.upper():
            continue
        m = re.search(r"[=<]\s*(\d+)\s*ms", line)
        if m:
            times.append(int(m.group(1)))
    if not times:
        return res
    res.avg = statistics.fmean(times)
    res.lo, res.hi = min(times), max(times)
    res.loss = round((count - len(times)) / count * 100)
    if len(times) > 1:
        res.jitter = statistics.fmean([abs(b - a) for a, b in zip(times, times[1:])])
    return res


def dns_benchmark(runner: Runner, cancel=None, progress=None) -> list[PingResult]:
    out = []
    for i, (label, primary, secondary) in enumerate(DNS_SERVERS):
        if cancel is not None and cancel.is_set():
            break
        res = ping(runner, primary, label, count=5)
        res.secondary = secondary
        out.append(res)
        if progress is not None:
            progress((i + 1) / len(DNS_SERVERS), res)
    return out


@dataclass
class Adapter:
    name: str = ""
    description: str = ""
    ipv4: str = ""
    gateway: str = ""
    dns: list[str] = field(default_factory=list)


_ADAPTER_SCRIPT = r"""
$ErrorActionPreference='SilentlyContinue'
Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway } | ForEach-Object {
  [ordered]@{
    name = "$($_.InterfaceAlias)"
    desc = "$($_.InterfaceDescription)"
    ip   = "$($_.IPv4Address.IPAddress)"
    gw   = "$($_.IPv4DefaultGateway.NextHop)"
    dns  = @($_.DNSServer | Where-Object { $_.AddressFamily -eq 2 } |
             ForEach-Object { $_.ServerAddresses } )
  }
} | ConvertTo-Json -Compress -Depth 4
"""


def active_adapters(runner: Runner) -> list[Adapter]:
    out = []
    for row in runner.ps_json(_ADAPTER_SCRIPT, timeout=60):
        if not isinstance(row, dict):
            continue
        dns = row.get("dns") or []
        if isinstance(dns, str):
            dns = [dns]
        out.append(Adapter(
            name=row.get("name") or "",
            description=row.get("desc") or "",
            ipv4=row.get("ip") or "",
            gateway=row.get("gw") or "",
            dns=[d for d in dns if d],
        ))
    return out


def set_dns(runner: Runner, adapter: str, primary: str, secondary: str = "") -> tuple[bool, str]:
    if not is_admin():
        return False, "관리자 권한 필요"
    servers = f'"{primary}"' + (f',"{secondary}"' if secondary else "")
    r = runner.powershell(
        f'Set-DnsClientServerAddress -InterfaceAlias "{adapter}" '
        f'-ServerAddresses {servers} -ErrorAction Stop', timeout=45)
    if r.ok:
        audit("network.dns", f"{adapter} -> {primary},{secondary}")
        runner.run(["ipconfig", "/flushdns"], timeout=20)
        return True, ""
    return False, (r.err or r.out or str(r)).splitlines()[0][:200] if (r.err or r.out) else str(r)


def reset_dns(runner: Runner, adapter: str) -> tuple[bool, str]:
    if not is_admin():
        return False, "관리자 권한 필요"
    r = runner.powershell(
        f'Set-DnsClientServerAddress -InterfaceAlias "{adapter}" '
        f'-ResetServerAddresses -ErrorAction Stop', timeout=45)
    if r.ok:
        audit("network.dns", f"{adapter} -> DHCP")
        runner.run(["ipconfig", "/flushdns"], timeout=20)
        return True, ""
    return False, (r.err or r.out or str(r)).splitlines()[0][:200] if (r.err or r.out) else str(r)


def local_ip(runner: Runner | None = None) -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(2)
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        pass
    if runner is not None:
        adapters = active_adapters(runner)
        if adapters:
            return adapters[0].ipv4
    return ""


def public_ip(timeout: float = 5.0) -> str:
    """타임아웃 없는 'curl ifconfig.me' 는 앱을 영원히 붙잡을 수 있다."""
    for url in ("https://api.ipify.org", "https://ifconfig.me/ip"):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                text = resp.read().decode("utf-8", "replace").strip()
                if text and len(text) < 64:
                    return text
        except Exception:
            continue
    return ""


# =====================================================================
# GPU 제조사 -> 드라이버 페이지
# =====================================================================
GPU_SITES = [
    (("nvidia", "geforce", "rtx", "gtx"), "NVIDIA",
     "https://www.nvidia.co.kr/Download/index.aspx?lang=kr"),
    (("amd", "radeon"), "AMD", "https://www.amd.com/ko/support"),
    (("intel", "arc", "uhd graphics", "iris"), "Intel",
     "https://www.intel.co.kr/content/www/kr/ko/download-center/home.html"),
]


def gpu_driver_site(gpu_names: str) -> tuple[str, str]:
    low = (gpu_names or "").lower()
    for keys, name, url in GPU_SITES:
        if any(k in low for k in keys):
            return name, url
    return "Search", "https://www.google.com/search?q=graphics+driver+download"


def uptime_seconds() -> float:
    if psutil is None:
        return 0.0
    import time as _t
    return max(0.0, _t.time() - psutil.boot_time())
