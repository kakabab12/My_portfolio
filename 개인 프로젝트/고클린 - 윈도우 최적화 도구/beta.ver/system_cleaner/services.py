# -*- coding: utf-8 -*-
"""윈도우 서비스 관리.

시작 유형 변경은 시스템 전체에 영향을 주므로
 - 핵심 서비스는 아예 변경을 막고
 - '꺼도 됨'으로 널리 알려진 것만 권장 표시를 하며
 - 변경 전 상태를 JSON 으로 백업한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from .core import Runner, audit, is_admin, save_backup

try:
    import psutil
except ImportError:
    psutil = None


RISK_SAFE = "safe"    # 일반적인 데스크톱에서 꺼도 문제 없음
RISK_CARE = "care"    # 쓰는 사람은 불편해짐
RISK_CORE = "core"    # 끄면 안 됨 (변경 차단)

# 끄면 윈도우가 정상 동작하지 않는 서비스 - 변경 자체를 막는다
PROTECTED = {
    "rpcss", "dcomlaunch", "rpcendpointmapper", "lsm", "power", "plugplay",
    "winmgmt", "eventlog", "eventsystem", "schedule", "cryptsvc", "dhcp",
    "dnscache", "nsi", "brokerinfrastructure", "systemeventsbroker",
    "usersvc", "profsvc", "themes", "audiosrv", "audioendpointbuilder",
    "gpsvc", "samss", "netlogon", "wlansvc", "bfe", "mpssvc", "trustedinstaller",
    "wudfsvc", "camsvc", "coremessagingregistrar", "staterepository",
    "tokenbroker", "usosvc", "storsvc", "dusmsvc",
}

# (서비스명 소문자, 위험도, 한글 설명, 영문 설명)
KNOWN: dict[str, tuple[str, str, str]] = {
    "diagtrack":       (RISK_SAFE, "사용자 경험 원격 분석(텔레메트리) 전송", "Telemetry collection"),
    "dmwappushservice": (RISK_SAFE, "WAP 푸시 메시지 라우팅", "WAP push routing"),
    "mapsbroker":      (RISK_SAFE, "오프라인 지도 다운로드 관리", "Downloaded maps manager"),
    "remoteregistry":  (RISK_SAFE, "원격에서 레지스트리 접근 허용 (보안상 끄는 게 낫습니다)",
                        "Remote registry access (safer off)"),
    "fax":             (RISK_SAFE, "팩스 송수신", "Fax"),
    "retaildemo":      (RISK_SAFE, "매장 전시용 데모 모드", "Retail demo mode"),
    "walletservice":   (RISK_SAFE, "결제 카드 지갑", "Wallet service"),
    "lfsvc":           (RISK_SAFE, "위치(GPS) 서비스", "Geolocation"),
    "wersvc":          (RISK_SAFE, "오류 보고서 MS 전송", "Windows Error Reporting"),
    "ajrouter":        (RISK_SAFE, "AllJoyn IoT 라우터", "AllJoyn router"),
    "wpcmonsvc":       (RISK_SAFE, "자녀 보호 기능", "Parental controls"),
    "scardsvr":        (RISK_SAFE, "스마트카드 리더", "Smart card"),
    "semgrsvc":        (RISK_SAFE, "NFC / 모바일 결제", "Payments and NFC"),
    "sharedaccess":    (RISK_SAFE, "인터넷 연결 공유(ICS)", "Internet Connection Sharing"),
    "remoteaccess":    (RISK_SAFE, "라우팅 및 원격 액세스", "Routing and Remote Access"),
    "wmpnetworksvc":   (RISK_SAFE, "Windows Media Player 네트워크 공유", "WMP network sharing"),
    "xblauthmanager":  (RISK_SAFE, "Xbox Live 인증 (Xbox 앱 안 쓰면 불필요)", "Xbox Live auth"),
    "xblgamesave":     (RISK_SAFE, "Xbox 게임 저장", "Xbox game save"),
    "xboxnetapisvc":   (RISK_SAFE, "Xbox 네트워킹", "Xbox networking"),
    "xboxgipsvc":      (RISK_SAFE, "Xbox 주변기기 관리", "Xbox accessory management"),
    "diagnosticshub.standardcollector.service":
                       (RISK_SAFE, "개발자 진단 수집기", "Diagnostics hub collector"),
    "printnotify":     (RISK_SAFE, "프린터 알림 (프린터 안 쓰면 불필요)", "Printer notifications"),

    "wsearch":         (RISK_CARE, "Windows 검색 색인 - 끄면 시작 메뉴 검색이 느려집니다",
                        "Search indexing - Start menu search gets slower"),
    "sysmain":         (RISK_CARE, "SuperFetch - SSD 에서는 효과가 적지만 HDD 에서는 도움이 됩니다",
                        "SuperFetch - helps on HDD, little effect on SSD"),
    "spooler":         (RISK_CARE, "인쇄 스풀러 - 프린터를 쓰면 켜두세요", "Print spooler"),
    "tabletinputservice": (RISK_CARE, "터치 키보드/필기 - 터치 화면이 있으면 켜두세요",
                           "Touch keyboard and handwriting"),
    "wbiosrvc":        (RISK_CARE, "생체 인식 - 지문/얼굴 로그인을 쓰면 켜두세요", "Biometrics"),
    "pcasvc":          (RISK_CARE, "프로그램 호환성 도우미", "Program Compatibility Assistant"),
    "dps":             (RISK_CARE, "진단 정책 서비스 - 네트워크 문제 해결사가 동작하지 않습니다",
                        "Diagnostic Policy Service"),
    "sessionenv":      (RISK_CARE, "원격 데스크톱 구성", "Remote Desktop configuration"),
    "termservice":     (RISK_CARE, "원격 데스크톱 서비스", "Remote Desktop Services"),
    "wpnservice":      (RISK_CARE, "푸시 알림 - 알림이 오지 않습니다", "Push notifications"),

    "wuauserv":        (RISK_CORE, "Windows 업데이트", "Windows Update"),
    "bits":            (RISK_CORE, "백그라운드 전송 - 업데이트에 필요", "BITS - needed by Windows Update"),
    "wscsvc":          (RISK_CORE, "보안 센터", "Security Center"),
    "windefend":       (RISK_CORE, "Microsoft Defender", "Microsoft Defender"),
}

START_TYPES = {
    "auto": ("Automatic", "자동"),
    "manual": ("Manual", "수동"),
    "disabled": ("Disabled", "사용 안 함"),
}


@dataclass
class ServiceItem:
    name: str
    display: str
    status: str          # running / stopped ...
    start_type: str      # automatic / manual / disabled ...
    risk: str = ""
    note_ko: str = ""
    note_en: str = ""
    pid: int = 0

    @property
    def protected(self) -> bool:
        return self.name.lower() in PROTECTED or self.risk == RISK_CORE

    @property
    def running(self) -> bool:
        return self.status.lower() == "running"


def _first_line(r) -> str:
    text = (r.err or r.out or "").strip()
    if not text:
        return str(r)
    return text.splitlines()[0][:200]


def _norm_start(raw: str) -> str:
    r = (raw or "").lower()
    if "disab" in r:
        return "disabled"
    if "auto" in r:
        return "auto"
    if "manual" in r or "demand" in r:
        return "manual"
    return r or "-"


def list_services() -> list[ServiceItem]:
    if psutil is None:
        return []
    out: list[ServiceItem] = []
    for svc in psutil.win_service_iter():
        try:
            info = svc.as_dict()
        except Exception:
            continue
        name = info.get("name") or ""
        known = KNOWN.get(name.lower())
        out.append(ServiceItem(
            name=name,
            display=info.get("display_name") or name,
            status=(info.get("status") or "").lower(),
            start_type=_norm_start(info.get("start_type") or ""),
            risk=known[0] if known else "",
            note_ko=known[1] if known else "",
            note_en=known[2] if known else "",
            pid=int(info.get("pid") or 0),
        ))
    # 권장 항목 -> 실행 중 -> 이름 순
    order = {RISK_SAFE: 0, RISK_CARE: 1, "": 2, RISK_CORE: 3}
    out.sort(key=lambda s: (order.get(s.risk, 2), not s.running, s.display.lower()))
    return out


def set_start_type(runner: Runner, item: ServiceItem, start: str) -> tuple[bool, str]:
    """start: auto | manual | disabled"""
    if not is_admin():
        return False, "관리자 권한 필요"
    if item.protected and start == "disabled":
        return False, "핵심 서비스라 사용 안 함으로 바꿀 수 없습니다"
    ps_value = {"auto": "Automatic", "manual": "Manual", "disabled": "Disabled"}.get(start)
    if ps_value is None:
        return False, f"알 수 없는 시작 유형: {start}"

    r = runner.powershell(
        f'Set-Service -Name "{item.name}" -StartupType {ps_value} -ErrorAction Stop', timeout=45)
    if not r.ok:
        # sc.exe 는 'start=' 와 값을 반드시 공백으로 띄워야 한다 (인자를 나눠서 전달)
        sc_value = {"auto": "auto", "manual": "demand", "disabled": "disabled"}[start]
        r = runner.run(["sc.exe", "config", item.name, "start=", sc_value], timeout=30)

    if r.ok:
        audit("service.start_type", f"{item.name}: {item.start_type} -> {start}")
        item.start_type = start
        return True, ""
    return False, _first_line(r)


def control(runner: Runner, item: ServiceItem, action: str) -> tuple[bool, str]:
    """action: start | stop"""
    if not is_admin():
        return False, "관리자 권한 필요"
    if item.protected and action == "stop":
        return False, "핵심 서비스라 중지할 수 없습니다"
    verb = "Start-Service" if action == "start" else "Stop-Service"
    extra = "" if action == "start" else " -Force"
    r = runner.powershell(f'{verb} -Name "{item.name}"{extra} -ErrorAction Stop', timeout=60)
    if r.ok:
        audit("service.control", f"{item.name} {action}")
        item.status = "running" if action == "start" else "stopped"
        return True, ""
    return False, _first_line(r)


def backup_services(items: list[ServiceItem]):
    return save_backup("services", [
        {"name": i.name, "display": i.display, "start_type": i.start_type, "status": i.status}
        for i in items])
