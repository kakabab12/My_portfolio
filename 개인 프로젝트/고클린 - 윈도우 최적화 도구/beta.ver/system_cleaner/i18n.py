# -*- coding: utf-8 -*-
"""다국어 문자열. 값은 (EN, KO) 쌍."""

from __future__ import annotations

STR: dict[str, tuple[str, str]] = {
    # ---------------- 네비게이션 ----------------
    "nav_group_status":  ("STATUS", "상태"),
    "nav_group_clean":   ("OPTIMIZE", "최적화"),
    "nav_group_manage":  ("MANAGE", "관리"),
    "nav_group_scan":    ("INSPECT", "검사"),
    "nav_group_etc":     ("SYSTEM", "시스템"),

    "nav_dashboard": ("Dashboard", "대시보드"),
    "nav_clean":     ("Clean & Boost", "최적화 및 부스트"),
    "nav_memory":    ("Memory Boost", "메모리 정리"),
    "nav_game":      ("Game Mode", "게임 모드"),
    "nav_startup":   ("Startup Apps", "시작 프로그램"),
    "nav_process":   ("Processes", "프로세스"),
    "nav_service":   ("Services", "서비스"),
    "nav_programs":  ("Programs", "프로그램 삭제"),
    "nav_bigfiles":  ("Large Files", "대용량 파일"),
    "nav_dupes":     ("Duplicates", "중복 파일"),
    "nav_disk":      ("Disk Health", "디스크 상태"),
    "nav_network":   ("Network", "네트워크"),
    "nav_update":    ("Update", "업데이트"),
    "nav_info":      ("My PC Info", "내 PC 정보"),

    # ---------------- 공통 ----------------
    "ready":      ("SYSTEM READY", "시스템 준비 완료"),
    "done":       ("[OK] FINISHED", "[완료] 작업 끝"),
    "failed":     ("[!] FINISHED WITH ERRORS", "[!] 일부 작업 실패"),
    "cancelled":  ("[--] CANCELLED", "[--] 사용자가 취소함"),
    "busy":       ("[BUSY] WORKING...", "[작업 중] 처리하는 중..."),
    "elapsed":    ("Elapsed {sec:.1f}s", "소요 {sec:.1f}초"),

    "btn_run":       ("RUN", "실행"),
    "btn_scan":      ("SCAN", "검사"),
    "btn_refresh":   ("REFRESH", "새로고침"),
    "btn_cancel":    ("CANCEL (ESC)", "취소 (ESC)"),
    "btn_apply":     ("APPLY", "적용"),
    "btn_close":     ("CLOSE", "닫기"),
    "btn_savelog":   ("SAVE LOG", "로그 저장"),
    "btn_clearlog":  ("CLEAR", "지우기"),
    "btn_selectall": ("SELECT ALL", "전체 선택"),
    "btn_selnone":   ("DESELECT", "선택 해제"),
    "btn_open":      ("OPEN LOCATION", "위치 열기"),
    "btn_delete":    ("DELETE", "삭제"),
    "btn_enable":    ("ENABLE", "사용"),
    "btn_disable":   ("DISABLE", "사용 안 함"),
    "btn_kill":      ("END TASK", "작업 끝내기"),
    "btn_start":     ("START", "시작"),
    "btn_stop":      ("STOP", "중지"),
    "btn_uninstall": ("UNINSTALL", "제거"),
    "btn_lang":      ("한글", "ENG"),
    "btn_settings":  ("Settings", "설정"),
    "btn_restore":   ("RESTORE POINT", "복원 지점"),

    "dlg_yes": ("PROCEED", "진행"),
    "dlg_no":  ("CANCEL", "취소"),

    "admin_ok":     ("ADMIN", "관리자"),
    "admin_no":     ("LIMITED", "권한 제한"),
    "need_admin":   ("This feature needs administrator rights.",
                     "이 기능은 관리자 권한이 필요합니다."),
    "limited_warn": ("Running without admin rights - some features are disabled.",
                     "관리자 권한이 없어 일부 기능이 비활성화됩니다."),
    "nothing_sel":  ("Nothing selected.", "선택한 항목이 없습니다."),
    "err_logged":   ("An error occurred and was logged: {path}",
                     "오류가 발생해 로그에 기록했습니다: {path}"),
    "no_psutil":    ("psutil is required:  pip install psutil",
                     "psutil 이 필요합니다:  pip install psutil"),

    "col_name":    ("Name", "이름"),
    "col_status":  ("Status", "상태"),
    "col_size":    ("Size", "크기"),
    "col_path":    ("Path", "경로"),
    "col_type":    ("Type", "종류"),
    "col_pub":     ("Publisher", "제조사"),
    "col_ver":     ("Version", "버전"),
    "col_date":    ("Date", "날짜"),
    "col_pid":     ("PID", "PID"),
    "col_cpu":     ("CPU", "CPU"),
    "col_mem":     ("Memory", "메모리"),
    "col_user":    ("User", "사용자"),
    "col_start":   ("Startup", "시작 유형"),
    "col_desc":    ("Description", "설명"),
    "col_risk":    ("Risk", "위험도"),
    "col_count":   ("Count", "개수"),
    "col_loc":     ("Location", "위치"),
    "col_cmd":     ("Command", "명령"),
    "col_modified":("Modified", "수정일"),
    "col_used":    ("Used", "사용 중"),
    "col_free":    ("Free", "여유"),
    "col_total":   ("Total", "전체"),

    "st_enabled":  ("Enabled", "사용"),
    "st_disabled": ("Disabled", "사용 안 함"),
    "st_missing":  ("File missing", "파일 없음"),
    "st_running":  ("Running", "실행 중"),
    "st_stopped":  ("Stopped", "중지됨"),

    # ---------------- 대시보드 ----------------
    "dash_title":   ("SYSTEM STATUS", "시스템 상태"),
    "dash_score":   ("HEALTH SCORE", "종합 점수"),
    "dash_cpu":     ("CPU", "CPU"),
    "dash_ram":     ("MEMORY", "메모리"),
    "dash_disk":    ("DISK C:", "디스크 C:"),
    "dash_uptime":  ("UPTIME", "연속 가동"),
    "dash_junk":    ("JUNK FILES", "정리 가능"),
    "dash_startup": ("STARTUP APPS", "시작 프로그램"),
    "dash_procs":   ("PROCESSES", "프로세스"),
    "dash_bin":     ("RECYCLE BIN", "휴지통"),
    "dash_quick":   ("QUICK ACTIONS", "빠른 실행"),
    "dash_advice":  ("SUGGESTIONS", "점검 결과"),
    "dash_scanning":("Checking system...", "시스템 점검 중..."),
    "dash_allgood": ("No problems found.", "특별한 문제가 없습니다."),
    "dash_days":    ("{d}d {h}h", "{d}일 {h}시간"),
    "dash_hours":   ("{h}h {m}m", "{h}시간 {m}분"),

    "adv_disk_low":  ("Drive {d}: has only {p}% free. Clean up or move files.",
                      "{d}: 드라이브 여유 공간이 {p}% 뿐입니다. 정리가 필요합니다."),
    "adv_junk":      ("{s} of temporary files can be removed.",
                      "임시 파일 {s} 를 정리할 수 있습니다."),
    "adv_startup":   ("{n} startup apps are slowing down boot.",
                      "시작 프로그램이 {n}개라 부팅이 느려질 수 있습니다."),
    "adv_ram":       ("Memory usage is {p}%. Close unused apps.",
                      "메모리 사용률이 {p}% 입니다. 안 쓰는 앱을 닫아보세요."),
    "adv_uptime":    ("Running for {d} days without a reboot.",
                      "{d}일째 재부팅하지 않았습니다."),
    "adv_bin":       ("Recycle Bin holds {s}.",
                      "휴지통에 {s} 가 들어 있습니다."),
    "adv_disk_bad":  ("Drive '{n}' reports health: {h}. Back up important data.",
                      "'{n}' 디스크 상태가 '{h}' 입니다. 중요 데이터를 백업하세요."),
    "adv_broken_startup": ("{n} startup entries point to missing files.",
                           "시작 프로그램 {n}개가 없는 파일을 가리킵니다."),

    # ---------------- 청소 ----------------
    "clean_title":   ("CLEAN & BOOST", "최적화 및 부스트"),
    "clean_desc":    ("Scan first, then choose what to remove. Nothing is deleted until you confirm.",
                      "먼저 검사한 뒤 지울 항목을 고릅니다. 확인 전에는 아무것도 삭제하지 않습니다."),
    "clean_scan":    ("Scanning targets...", "삭제 대상 검사 중..."),
    "clean_found":   ("Found {s} in {n} locations.", "{n}곳에서 {s} 를 찾았습니다."),
    "clean_confirm": ("Delete {s}?", "{s} 를 삭제할까요?"),
    "clean_freed":   ("TOTAL FREED", "총 확보 용량"),
    "clean_inuse":   ("in use", "사용 중"),
    "rpt_clean":     ("CLEAN REPORT", "최적화 결과"),
    "rpt_preview":   ("SCAN RESULT", "검사 결과"),
    "it_dns":        ("DNS / ARP", "DNS / ARP"),
    "it_temp":       ("TEMP FILES", "임시 파일"),
    "it_browser":    ("BROWSER CACHE", "브라우저 캐시"),
    "it_recycle":    ("RECYCLE BIN", "휴지통"),
    "it_dism":       ("SYSTEM COMP", "시스템 구성"),
    "it_socket":     ("SOCKET RESET", "소켓 초기화"),
    "it_defrag":     ("DEFRAG / TRIM", "디스크 정리"),
    "browser_warn":  ("Close your browsers first - locked cache files will be skipped.",
                      "브라우저를 먼저 닫아주세요. 잠긴 캐시 파일은 건너뜁니다."),
    "browser_safe":  ("Passwords, cookies and bookmarks are never touched.",
                      "비밀번호·쿠키·즐겨찾기는 건드리지 않습니다."),

    # ---------------- 메모리 ----------------
    "mem_title":  ("MEMORY BOOST", "메모리 정리"),
    "mem_desc":   ("Trims the working set of running processes. Windows pages memory back in "
                   "when an app needs it again, so treat this as a temporary cleanup.",
                   "실행 중인 프로세스의 작업 집합을 정리합니다. 앱이 다시 필요로 하면 윈도우가 "
                   "메모리를 되돌려 주므로 일시적인 효과입니다."),
    "mem_before": ("AVAILABLE BEFORE", "정리 전 여유"),
    "mem_after":  ("AVAILABLE AFTER", "정리 후 여유"),
    "mem_gain":   ("FREED", "확보량"),
    "mem_trimmed":("PROCESSES TRIMMED", "정리한 프로세스"),
    "rpt_mem":    ("MEMORY REPORT", "메모리 정리 결과"),

    # ---------------- 게임 모드 ----------------
    "game_title":  ("GAME MODE", "게임 모드"),
    "game_desc":   ("Closes background apps and switches to the High performance power plan.",
                    "백그라운드 앱을 정리하고 고성능 전원 관리 옵션으로 전환합니다."),
    "game_pick":   ("Select apps to close", "종료할 앱 선택"),
    "game_hint":   ("Only apps running right now are listed. Unchecked apps stay open.",
                    "지금 실행 중인 앱만 표시됩니다. 체크 해제한 앱은 그대로 둡니다."),
    "game_save":   ("Save your work first - apps are asked to close, then force-killed.",
                    "먼저 작업을 저장하세요. 정상 종료를 요청한 뒤 응답이 없으면 강제 종료합니다."),
    "rpt_game":    ("GAME MODE REPORT", "게임 모드 결과"),
    "it_target":   ("CANDIDATES", "대상 앱"),
    "it_closed":   ("CLOSED", "종료됨"),
    "it_power":    ("POWER PLAN", "전원 관리"),
    "it_ram_gain": ("RAM FREED", "확보한 메모리"),
    "game_restore":("RESTORE POWER PLAN", "전원 설정 되돌리기"),

    # ---------------- 시작 프로그램 ----------------
    "startup_title": ("STARTUP APPS", "시작 프로그램"),
    "startup_desc":  ("Disabling is reversible - Windows keeps the entry and just skips it. "
                      "Delete removes it for good (a backup is saved first).",
                      "'사용 안 함'은 되돌릴 수 있습니다. 항목은 남기고 실행만 건너뜁니다. "
                      "'삭제'는 완전히 지웁니다(삭제 전 백업을 남깁니다)."),
    "startup_count": ("{n} entries, {e} enabled", "총 {n}개, 사용 중 {e}개"),
    "startup_del_q": ("Delete {n} startup entries permanently?",
                      "시작 프로그램 {n}개를 완전히 삭제할까요?"),
    "startup_backup":("A backup was saved to: {p}", "백업을 저장했습니다: {p}"),

    # ---------------- 프로세스 ----------------
    "proc_title": ("PROCESSES", "실행 중인 프로세스"),
    "proc_desc":  ("Sorted by memory. Critical Windows processes are protected.",
                   "메모리 사용량 순입니다. 핵심 윈도우 프로세스는 종료할 수 없습니다."),
    "proc_count": ("{n} processes · {m} used", "프로세스 {n}개 · {m} 사용 중"),
    "proc_kill_q":("End {n} processes? Unsaved work will be lost.",
                   "프로세스 {n}개를 종료할까요? 저장하지 않은 작업은 사라집니다."),
    "proc_protected": ("'{n}' is a critical system process and cannot be ended here.",
                       "'{n}' 은 핵심 시스템 프로세스라 여기서 종료할 수 없습니다."),

    # ---------------- 서비스 ----------------
    "svc_title": ("WINDOWS SERVICES", "윈도우 서비스"),
    "svc_desc":  ("Changing service startup type affects the whole system. "
                  "Entries marked SAFE are widely considered optional.",
                  "서비스 시작 유형 변경은 시스템 전체에 영향을 줍니다. "
                  "'권장'으로 표시된 항목은 일반적으로 꺼도 되는 것들입니다."),
    "svc_count": ("{n} services · {r} running", "서비스 {n}개 · 실행 중 {r}개"),
    "svc_auto":   ("Automatic", "자동"),
    "svc_manual": ("Manual", "수동"),
    "svc_dis":    ("Disabled", "사용 안 함"),
    "svc_set_q":  ("Set {n} services to '{m}'?", "서비스 {n}개를 '{m}'(으)로 바꿀까요?"),
    "svc_risk_safe": ("Optional", "꺼도 됨"),
    "svc_risk_care": ("Careful", "주의"),
    "svc_risk_core": ("Core", "핵심"),

    # ---------------- 프로그램 삭제 ----------------
    "prog_title": ("INSTALLED PROGRAMS", "설치된 프로그램"),
    "prog_desc":  ("Sorted by size. The program's own uninstaller opens - follow its steps.",
                   "크기 순입니다. 각 프로그램의 제거 마법사가 실행되니 안내를 따라 진행하세요."),
    "prog_count": ("{n} programs · {s} total", "프로그램 {n}개 · 총 {s}"),
    "prog_run_q": ("Run the uninstaller for '{n}'?", "'{n}' 제거 프로그램을 실행할까요?"),
    "prog_started":("Uninstaller started for '{n}'.", "'{n}' 제거를 시작했습니다."),

    # ---------------- 대용량 / 중복 파일 ----------------
    "big_title": ("LARGE FILES", "대용량 파일 찾기"),
    "big_desc":  ("Finds files above the size threshold. Deleted files go to the Recycle Bin.",
                  "지정한 크기 이상인 파일을 찾습니다. 삭제한 파일은 휴지통으로 갑니다."),
    "big_min":   ("Minimum size (MB)", "최소 크기 (MB)"),
    "big_folder":("Folder", "검사 폴더"),
    "big_browse":("BROWSE", "폴더 선택"),
    "big_found": ("{n} files · {s}", "파일 {n}개 · {s}"),

    "dup_title": ("DUPLICATE FILES", "중복 파일 찾기"),
    "dup_desc":  ("Compares by size, then by content hash. One copy per group stays checked off.",
                  "크기로 1차 비교 후 내용 해시로 확인합니다. 각 그룹에서 한 개는 남겨둡니다."),
    "dup_found": ("{g} groups · {n} extra files · {s} recoverable",
                  "중복 {g}그룹 · 여분 {n}개 · {s} 회수 가능"),
    "dup_keep":  ("KEEP FIRST IN EACH GROUP", "그룹마다 첫 파일 유지"),
    "del_confirm": ("Move {n} files ({s}) to the Recycle Bin?",
                    "파일 {n}개 ({s}) 를 휴지통으로 옮길까요?"),
    "del_done":    ("Moved {n} files to the Recycle Bin.", "{n}개를 휴지통으로 옮겼습니다."),

    # ---------------- 디스크 ----------------
    "disk_title": ("DISK HEALTH", "디스크 상태"),
    "disk_desc":  ("S.M.A.R.T. data reported by Windows. Not every drive exposes every field.",
                   "윈도우가 읽어오는 S.M.A.R.T. 정보입니다. 드라이브에 따라 빈 항목이 있을 수 있습니다."),
    "disk_health":("Health", "상태"),
    "disk_temp":  ("Temp", "온도"),
    "disk_wear":  ("Wear", "마모도"),
    "disk_hours": ("Power-on", "사용 시간"),
    "disk_errors":("Errors", "오류"),
    "disk_vol":   ("VOLUMES", "볼륨"),
    "disk_free":  ("free of", "여유 /"),

    # ---------------- 네트워크 ----------------
    "net_title":  ("NETWORK", "네트워크"),
    "net_desc":   ("Measures latency to public DNS servers and lets you switch to the fastest one.",
                   "공용 DNS 서버의 응답 속도를 측정하고 가장 빠른 곳으로 바꿀 수 있습니다."),
    "net_bench":  ("DNS BENCHMARK", "DNS 응답 속도"),
    "net_apply":  ("APPLY FASTEST DNS", "가장 빠른 DNS 적용"),
    "net_auto":   ("BACK TO AUTOMATIC", "자동(DHCP)으로 되돌리기"),
    "net_dns_q":  ("Change DNS of '{a}' to {d}?", "'{a}' 어댑터의 DNS 를 {d} 로 바꿀까요?"),
    "net_dns_ok": ("DNS changed to {d}.", "DNS 를 {d} 로 변경했습니다."),
    "net_dns_rev":("DNS is back to automatic.", "DNS 를 자동 할당으로 되돌렸습니다."),
    "rpt_net":    ("NETWORK REPORT", "네트워크 결과"),

    # ---------------- 업데이트 ----------------
    "upd_title": ("SYSTEM UPDATE", "시스템 업데이트"),
    "upd_desc":  ("Checks winget packages, opens Windows Update and Microsoft Store, "
                  "and finds your GPU driver page.",
                  "winget 패키지를 확인하고 Windows 업데이트·MS 스토어를 열며, "
                  "그래픽 드라이버 페이지를 찾아줍니다."),
    "rpt_update":("UPDATE REPORT", "업데이트 결과"),
    "it_sw":     ("SOFTWARE", "소프트웨어"),
    "it_os":     ("WINDOWS OS", "윈도우 OS"),
    "it_store":  ("MS STORE", "MS 스토어"),
    "it_gpu":    ("GPU DRIVER", "그래픽 드라이버"),
    "upd_list":  ("UPGRADABLE", "업그레이드 가능"),
    "upd_all_q": ("Upgrade all {n} packages? Some apps may restart.",
                  "{n}개 패키지를 모두 업그레이드할까요? 일부 앱이 재시작될 수 있습니다."),
    "upd_none":  ("Everything is up to date.", "모두 최신 상태입니다."),
    "upd_manual":("Finish the updates in the windows that just opened.",
                  "열린 창에서 업데이트를 마저 진행해주세요."),

    # ---------------- 내 PC 정보 ----------------
    "info_title": ("MY PC INFO", "내 PC 정보"),
    "rpt_spec":   ("SYSTEM SPECS", "시스템 사양"),

    # ---------------- 상태 값 ----------------
    "v_ok":       ("OK", "성공"),
    "v_failed":   ("FAILED", "실패"),
    "v_timeout":  ("TIMEOUT", "시간 초과"),
    "v_notfound": ("Not found", "없음"),
    "v_opened":   ("Opened", "열림"),
    "v_skipped":  ("Skipped", "건너뜀"),
    "v_none":     ("Nothing to do", "대상 없음"),
    "v_reboot":   ("OK (reboot required)", "성공 (재부팅 필요)"),
    "v_na":       ("N/A", "해당 없음"),

    # ---------------- 안전 장치 ----------------
    "rp_making":  ("Creating a restore point first...", "먼저 복원 지점을 만드는 중..."),
    "rp_ok":      ("Restore point created: {d}", "복원 지점을 만들었습니다: {d}"),
    "rp_fail":    ("Could not create a restore point: {d}", "복원 지점 생성 실패: {d}"),
    "rp_off":     ("Continue anyway?", "그래도 계속할까요?"),
    "warn_sock":  ("Winsock reset requires a REBOOT and may reset VPN / proxy settings.",
                   "winsock 초기화는 재부팅이 필요하고 VPN·프록시 설정이 초기화될 수 있습니다."),
    "warn_bin":   ("The Recycle Bin will be emptied PERMANENTLY.",
                   "휴지통이 영구적으로 비워집니다. 복구할 수 없습니다."),
    "warn_pref":  ("Clearing Prefetch makes the next boot slightly slower.",
                   "Prefetch 를 지우면 다음 부팅이 조금 느려집니다."),


    # ---------------- 윈도우 설정 ----------------
    "nav_tweaks":  ("Windows Tweaks", "윈도우 설정"),
    "twk_title": ("WINDOWS TWEAKS", "윈도우 설정"),
    "twk_desc":  ("Each switch flips exactly one registry value, and flipping it back undoes "
                  "it. What changes is written out for every item - no vague 'optimization'.",
                  "각 항목은 레지스트리 값 하나를 바꾸고, 다시 끄면 그대로 돌아옵니다. "
                  "무엇이 바뀌는지 항목마다 적어두었습니다."),
    "twk_applied": ("applied", "적용됨"),
    "twk_default": ("default", "기본값"),
    "twk_admin":   ("needs admin", "관리자 권한 필요"),
    "twk_restart": ("Explorer restarted to apply the change.",
                    "변경 사항 적용을 위해 탐색기를 다시 시작했습니다."),
    "twk_failed":  ("Could not apply: {d}", "적용 실패: {d}"),

    # ---------------- 로그 ----------------
    "log_saved":  ("Log saved: {path}", "로그를 저장했습니다: {path}"),
    "log_lang":   (">> LANGUAGE: ENGLISH", ">> 언어 설정: 한국어"),
    "log_ready":  (">> WAITING FOR INPUT...", ">> 명령 대기 중..."),
}


def t(key: str, lang: str = "EN", **kw) -> str:
    pair = STR.get(key)
    if pair is None:
        return key
    text = pair[0] if lang == "EN" else pair[1]
    if kw:
        try:
            return text.format(**kw)
        except (KeyError, IndexError, ValueError):
            return text
    return text
