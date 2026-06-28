import customtkinter as ctk
import subprocess
import threading
import sys
import ctypes
import re
import webbrowser
import os

# === 1. 관리자 권한 자동 획득 ===
def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

if not is_admin():
    ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
    sys.exit()

# === 2. 테마 설정 ===
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# 색상 정의
COLOR_BG = "#0f0f0f"
COLOR_SIDEBAR = "#181818"
COLOR_ACCENT = "#00E5FF"
COLOR_BTN_HOVER = "#00B8D4"
COLOR_TERMINAL_BG = "#000000"
COLOR_TERMINAL_TXT = "#00FF41"

class SystemMasterApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- 언어 설정 데이터 ---
        self.current_lang = "EN"
        self.ui_locales = {
            "update": {"EN": "  >  SYSTEM UPDATE", "KO": "  >  시스템 업데이트"},
            "clean":  {"EN": "  >  CLEAN & BOOST", "KO": "  >  최적화 및 부스트"},
            "game":   {"EN": "  >  GAME MODE",     "KO": "  >  게임 모드"},
            "ping":   {"EN": "  >  PING TEST",     "KO": "  >  핑 테스트"},
            "info":   {"EN": "  >  MY PC INFO",    "KO": "  >  내 PC 정보"},
            "ready":  {"EN": "SYSTEM READY",       "KO": "시스템 준비 완료"},
            "done":   {"EN": "[OK] JOB FINISHED",  "KO": "[완료] 작업 끝"},
            "lang_btn": {"EN": "한글", "KO": "ENG"}
        }

        # --- CMD 출력 메시지용 데이터 ---
        self.cmd_locales = {
            "rpt_update": {"EN": "UPDATE REPORT", "KO": "업데이트 결과"},
            "sw":         {"EN": "SOFTWARE    ", "KO": "소프트웨어  "},
            "pip":        {"EN": "PYTHON PIP  ", "KO": "파이썬 PIP  "},
            "os":         {"EN": "WINDOWS OS  ", "KO": "윈도우 OS   "},
            "store":      {"EN": "MS STORE    ", "KO": "MS 스토어   "},
            "gpu":        {"EN": "GPU DRIVER  ", "KO": "그래픽드라이버"},
            
            "rpt_clean":  {"EN": "CLEAN REPORT", "KO": "최적화 결과"},
            "dns":        {"EN": "DNS / ARP   ", "KO": "DNS / ARP   "},
            "temp":       {"EN": "TEMP FILES  ", "KO": "임시 파일   "},
            "syscomp":    {"EN": "SYSTEM COMP ", "KO": "시스템 구성 "},
            "socket":     {"EN": "SOCKET RESET", "KO": "소켓 초기화 "},
            "defrag":     {"EN": "DEFRAG/TRIM ", "KO": "디스크 정리 "},

            "rpt_game":   {"EN": "GAME MODE REPORT", "KO": "게임 모드 결과"},
            "target":     {"EN": "TARGET APPS ", "KO": "대상 앱     "},
            "kill":       {"EN": "TERMINATED  ", "KO": "종료됨      "},
            "exp":        {"EN": "EXPLORER    ", "KO": "탐색기      "},
            "mem":        {"EN": "MEMORY      ", "KO": "메모리      "},
            "mem_opt":    {"EN": "OPTIMIZED",    "KO": "최적화됨"},

            "rpt_net":    {"EN": "NETWORK REPORT", "KO": "네트워크 결과"},
            "g_dns":      {"EN": "GOOGLE DNS  ",   "KO": "구글 DNS    "},
            "cf_dns":     {"EN": "CLOUDFLARE  ",   "KO": "클라우드플레어"},

            "rpt_spec":   {"EN": "SYSTEM SPECS", "KO": "시스템 사양"},
        }

        # 창 설정
        self.title("SYSTEM CLEANER [ALPHA ver.]")
        self.geometry("950x650")
        self.resizable(False, False)
        self.configure(fg_color=COLOR_BG)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 사이드바
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0, fg_color=COLOR_SIDEBAR)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(8, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar, text="SYSTEM\nCLEANER", font=ctk.CTkFont(family="Arial", size=26, weight="bold"), text_color=COLOR_ACCENT)
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 5))

        self.ver_label = ctk.CTkLabel(self.sidebar, text="ALPHA.ver", font=ctk.CTkFont(size=12), text_color="gray")
        self.ver_label.grid(row=1, column=0, padx=20, pady=(0, 30))

        btn_config = {
            "font": ctk.CTkFont(size=13, weight="bold"),
            "height": 45, "corner_radius": 6, "fg_color": "transparent",
            "border_width": 1, "border_color": "#444444", "text_color": "#CCCCCC",
            "hover_color": "#2d2d2d", "anchor": "w"
        }

        self.btn_update = ctk.CTkButton(self.sidebar, text=self.ui_locales["update"]["EN"], command=self.start_update, **btn_config)
        self.btn_update.grid(row=2, column=0, padx=15, pady=8, sticky="ew")

        self.btn_clean = ctk.CTkButton(self.sidebar, text=self.ui_locales["clean"]["EN"], command=self.start_clean, **btn_config)
        self.btn_clean.grid(row=3, column=0, padx=15, pady=8, sticky="ew")

        self.btn_game = ctk.CTkButton(self.sidebar, text=self.ui_locales["game"]["EN"], command=self.start_gamemode, **btn_config)
        self.btn_game.grid(row=4, column=0, padx=15, pady=8, sticky="ew")

        self.btn_ping = ctk.CTkButton(self.sidebar, text=self.ui_locales["ping"]["EN"], command=self.start_ping, **btn_config)
        self.btn_ping.grid(row=5, column=0, padx=15, pady=8, sticky="ew")

        self.btn_info = ctk.CTkButton(self.sidebar, text=self.ui_locales["info"]["EN"], command=self.start_info, 
                                      font=ctk.CTkFont(size=13, weight="bold"), height=45, corner_radius=6, 
                                      fg_color=COLOR_ACCENT, text_color="black", hover_color=COLOR_BTN_HOVER, anchor="w")
        self.btn_info.grid(row=6, column=0, padx=15, pady=20, sticky="ew")

        # 하단 버튼 그룹
        self.bottom_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.bottom_frame.grid(row=9, column=0, padx=20, pady=20, sticky="w")

        self.btn_about = ctk.CTkButton(
            self.bottom_frame, text="?", width=30, height=30, corner_radius=15,
            fg_color="transparent", border_width=1, border_color="#555555",
            text_color="gray", hover_color="#333333",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.start_about
        )
        self.btn_about.pack(side="left", padx=(0, 10))

        self.btn_lang = ctk.CTkButton(
            self.bottom_frame, text="한글", width=40, height=30, corner_radius=6,
            fg_color="#222222", border_width=0,
            text_color="white", hover_color="#444444",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self.toggle_language
        )
        self.btn_lang.pack(side="left")

        # 메인 영역
        self.main_area = ctk.CTkFrame(self, fg_color="transparent")
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=25, pady=25)
        self.main_area.grid_rowconfigure(1, weight=1)
        self.main_area.grid_columnconfigure(0, weight=1)

        self.status_frame = ctk.CTkFrame(self.main_area, fg_color="transparent")
        self.status_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.status_title = ctk.CTkLabel(self.status_frame, text=self.ui_locales["ready"]["EN"], font=ctk.CTkFont(size=20, weight="bold"), text_color="white", anchor="w")
        self.status_title.pack(side="left")

        self.progressbar = ctk.CTkProgressBar(self.main_area, height=4, corner_radius=2, progress_color=COLOR_ACCENT)
        self.progressbar.grid(row=2, column=0, sticky="ew", pady=(15, 0))
        self.progressbar.set(0)

        self.terminal_frame = ctk.CTkFrame(self.main_area, fg_color=COLOR_TERMINAL_BG, corner_radius=8, border_width=1, border_color="#333333")
        self.terminal_frame.grid(row=1, column=0, sticky="nsew")
        self.terminal_frame.grid_columnconfigure(0, weight=1)
        self.terminal_frame.grid_rowconfigure(0, weight=1)

        self.textbox = ctk.CTkTextbox(self.terminal_frame, font=("Consolas", 13), text_color=COLOR_TERMINAL_TXT, fg_color="transparent", wrap="none")
        self.textbox.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.log(">> INITIALIZING KERNEL...")
        self.log(">> SYSTEM LANGUAGE: ENGLISH")
        self.log(">> WAITING FOR INPUT...\n")

        # [NEW] 후원 버튼
        self.btn_link = None

    def toggle_language(self):
        self.current_lang = "KO" if self.current_lang == "EN" else "EN"
        lang = self.current_lang
        
        self.btn_update.configure(text=self.ui_locales["update"][lang])
        self.btn_clean.configure(text=self.ui_locales["clean"][lang])
        self.btn_game.configure(text=self.ui_locales["game"][lang])
        self.btn_ping.configure(text=self.ui_locales["ping"][lang])
        self.btn_info.configure(text=self.ui_locales["info"][lang])
        self.btn_lang.configure(text=self.ui_locales["lang_btn"][lang])

        if "BUSY" not in self.status_title.cget("text"):
             self.status_title.configure(text=self.ui_locales["ready"][lang])

        msg = f">> LANGUAGE SWITCHED TO: {lang}" if lang == "EN" else ">> 언어 설정이 [한국어]로 변경되었습니다."
        self.log(msg)

    def log_summary(self, title, items):
        box_width = 40
        top_border = f"╔══ [ {title} ] " + ("═" * (box_width - len(title)))
        bottom_border = "╚" + ("═" * (len(top_border) - 1))
        txt = f"\n{top_border}\n║\n"
        for item in items:
            txt += f"║   {item}\n"
        txt += f"║\n{bottom_border}\n"
        self.log(txt)

    def safe_decode(self, byte_data):
        try: return byte_data.decode('cp949')
        except: pass
        try: return byte_data.decode('utf-8')
        except: pass
        return byte_data.decode('cp949', errors='replace')

    def get_cmd_output(self, cmd):
        try:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            output = subprocess.check_output(cmd, shell=True, startupinfo=startupinfo)
            return self.safe_decode(output).strip()
        except: return ""

    def log(self, message):
        self.textbox.insert("end", message + "\n")
        self.textbox.see("end")

    def set_working_state(self, is_working, status_text_en="PROCESSING"):
        lang = self.current_lang
        
        if self.btn_link is not None:
            self.btn_link.destroy()
            self.btn_link = None

        if is_working:
            display_text = f"[BUSY] {status_text_en}"
            if lang == "KO":
                if "UPDATING" in status_text_en: display_text = "[작업 중] 업데이트 확인 중..."
                elif "CLEANING" in status_text_en: display_text = "[작업 중] 시스템 청소 중..."
                elif "GAME" in status_text_en: display_text = "[작업 중] 게임 최적화 중..."
                elif "PING" in status_text_en: display_text = "[작업 중] 핑 측정 중..."
                elif "SCANNING" in status_text_en: display_text = "[작업 중] 하드웨어 스캔 중..."
            
            self.status_title.configure(text=display_text, text_color="#FFD700")
            self.progressbar.configure(mode="indeterminate")
            self.progressbar.start()
            for btn in [self.btn_update, self.btn_clean, self.btn_game, self.btn_ping, self.btn_info]:
                btn.configure(state="disabled")
        else:
            self.status_title.configure(text=self.ui_locales["done"][lang], text_color=COLOR_ACCENT)
            self.progressbar.configure(mode="determinate")
            self.progressbar.stop()
            self.progressbar.set(1)
            for btn in [self.btn_update, self.btn_clean, self.btn_game, self.btn_ping, self.btn_info]:
                btn.configure(state="normal")

    # === 개발자 정보 & 후원 (팝업 선택 방식) ===
    def start_about(self):
        self.textbox.delete("0.0", "end")
        self.log(">> ACCESSING SECURE DATA BLOCK...")
        
        credit_box = """
╔══════════════════════════════════════════════════════
║
║   [ SYSTEM ARCHITECT ]
║     DEVELOPER  : 이지용(LEE_JI_YONG)
║     CONTACT    : yeez0612@naver.com
║
║   [ AI CO-PILOT ]
║     CORE LOGIC : GOOGLE GEMINI
║     STATUS     : ONLINE
║     BUILD      : ALPHA.ver
║
╚══════════════════════════════════════════════════════
"""
        self.log(credit_box)
        self.log("\n[!] PRESS THE BUTTON BELOW TO SUPPORT DEV.")
        self.log(">> END OF TRANSMISSION.")

        # --- 후원 선택 팝업 함수 ---
        def open_donation_popup():
            # 1. 팝업창 생성
            top = ctk.CTkToplevel(self)
            top.title("Donation Method")
            top.geometry("300x180")
            top.resizable(False, False)
            top.attributes("-topmost", True) # 항상 위에 표시
            
            # 안내 멘트
            label = ctk.CTkLabel(top, text="후원 방식을 선택해주세요\n(Select Donation Method)", font=ctk.CTkFont(size=14, weight="bold"))
            label.pack(pady=(20, 15))

            # 2. 기능: Buy Me a Coffee (링크 열기)
            def go_bmc():
                # [수정] 본인의 BMC 주소 입력
                webbrowser.open("https://buymeacoffee.com/kakabab12") 
                top.destroy()

            # 3. 기능: 카카오뱅크 (계좌번호 복사)
            def copy_kakao():
                # [수정] 본인의 카카오뱅크 계좌번호
                my_account = "3333-xx-xxxxxx" 
                
                self.clipboard_clear()
                self.clipboard_append(my_account)
                self.log(f"\n[!] ACCOUNT COPIED: {my_account}")
                self.log("[!] 카카오뱅크 계좌번호가 복사되었습니다.")
                
                # 버튼 텍스트를 잠시 '복사됨!'으로 변경
                btn_kakao.configure(text="복사 완료! (Copied!)", fg_color="green")
                top.after(1000, top.destroy) # 1초 뒤 창 닫기

            # 버튼 배치
            btn_bmc = ctk.CTkButton(top, text="☕ Buy Me a Coffee", fg_color="#FFDD00", text_color="black", hover_color="#E6C700", command=go_bmc)
            btn_bmc.pack(pady=5, padx=20, fill="x")

            btn_kakao = ctk.CTkButton(top, text="🟡 KakaoBank (Copy Number)", fg_color="#FEE500", text_color="black", hover_color="#E6CE00", command=copy_kakao)
            btn_kakao.pack(pady=5, padx=20, fill="x")

        # 기존 버튼 초기화
        if self.btn_link is not None: self.btn_link.destroy()
        
        # 메인 화면의 버튼 (이걸 누르면 위 팝업이 뜸)
        self.btn_link = ctk.CTkButton(
            self.terminal_frame, 
            text="☕  DONATION & SUPPORT", 
            fg_color="#FFD700", text_color="black", hover_color="#DAA520",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=open_donation_popup # 함수 연결
        )
        self.btn_link.place(relx=0.5, rely=0.8, anchor="center")

    # === 1. 업데이트 (제조사 감지 후 사이트 연결) ===
    def start_update(self):
        self.textbox.delete("0.0", "end")
        self.set_working_state(True, "UPDATING SYSTEM...")
        
        msg = ">> CHECKING UPDATES..." if self.current_lang == "EN" else ">> 업데이트 확인 중..."
        self.log(msg)

        def task():
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            # 1. 소프트웨어 (Winget)
            try:
                subprocess.run("winget upgrade --all --include-unknown --accept-package-agreements --accept-source-agreements", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, startupinfo=startupinfo)
                s1 = "Up to date"
            except: s1 = "Check Required"

            # 2. 파이썬 (PIP)
            try:
                subprocess.run("python -m pip install --upgrade pip", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, startupinfo=startupinfo)
                s2 = "Updated"
            except: s2 = "Not Found"

            # 3. 윈도우 OS 업데이트 (설정창)
            try:
                subprocess.run("start ms-settings:windowsupdate-action", shell=True, startupinfo=startupinfo)
                s3 = "Settings Opened" if self.current_lang == "EN" else "설정창 열림"
            except: s3 = "Failed"

            # 4. 마이크로소프트 스토어
            try:
                subprocess.run("start ms-windows-store:updates", shell=True, startupinfo=startupinfo)
                s4 = "Store Opened" if self.current_lang == "EN" else "스토어 열림"
            except: s4 = "Failed"

            # 5. GPU 드라이버 (제조사 자동 감지 -> 사이트 오픈)
            try:
                # WMIC 명령어로 그래픽카드 이름 확인 (nvidia, amd, intel 등 포함 여부 확인)
                gpu_info = self.get_cmd_output("wmic path win32_VideoController get name").lower()
                
                url = ""
                site_name = ""

                if "nvidia" in gpu_info:
                    url = "https://www.nvidia.co.kr/Download/index.aspx?lang=kr"
                    site_name = "NVIDIA Site"
                elif "amd" in gpu_info or "radeon" in gpu_info:
                    url = "https://www.amd.com/ko/support"
                    site_name = "AMD Site"
                elif "intel" in gpu_info:
                    url = "https://www.intel.co.kr/content/www/kr/ko/download-center/home.html"
                    site_name = "Intel Site"
                else:
                    # 감지 실패 시 구글 검색
                    url = "https://www.google.com/search?q=그래픽카드+드라이버+다운로드"
                    site_name = "Manual Search"
                
                webbrowser.open(url)
                gpu_status = f"{site_name} Opened"

            except:
                webbrowser.open("https://www.google.com/search?q=그래픽카드+드라이버+다운로드")
                gpu_status = "Web Fallback"

            # --- 결과 리포트 ---
            L = self.current_lang
            C = self.cmd_locales
            
            self.log_summary(C["rpt_update"][L], [
                f"{C['sw'][L]} : {s1}", 
                f"{C['pip'][L]} : {s2}", 
                f"{C['os'][L]} : {s3}", 
                f"{C['store'][L]} : {s4}",
                f"{C['gpu'][L]} : {gpu_status}"
            ])
            
            note = "[!] Please check the opened windows/websites for manual updates." if L == "EN" else "[!] 열린 창과 웹사이트에서 업데이트를 수동으로 진행해주세요."
            self.log(f"\n{note}")
            
            self.set_working_state(False)
        threading.Thread(target=task, daemon=True).start()

    # === 2. 청소 ===
    def start_clean(self):
        self.textbox.delete("0.0", "end")
        self.set_working_state(True, "CLEANING SYSTEM...")
        msg = ">> STARTING OPTIMIZATION..." if self.current_lang == "EN" else ">> 최적화 시작..."
        self.log(msg)

        def task():
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            try:
                subprocess.run("ipconfig /flushdns & arp -d *", shell=True, stdout=subprocess.DEVNULL, startupinfo=startupinfo)
                s1 = "Flushed"
            except: s1 = "Failed"

            try:
                subprocess.run('del /s /q /f "%TEMP%\\*.*"', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, startupinfo=startupinfo)
                subprocess.run('del /s /q /f "C:\\Windows\\Temp\\*.*"', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, startupinfo=startupinfo)
                s2 = "Deleted"
            except: s2 = "Partial"

            try:
                subprocess.run("dism /online /Cleanup-Image /StartComponentCleanup", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, startupinfo=startupinfo)
                s3 = "Optimized"
            except: s3 = "Pending"

            try:
                subprocess.run("netsh winsock reset", shell=True, stdout=subprocess.DEVNULL, startupinfo=startupinfo)
                s4 = "Reset"
            except: s4 = "Failed"
            
            try:
                subprocess.Popen("defrag /C /O", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, startupinfo=startupinfo)
                s5 = "Running..."
            except: s5 = "Failed"

            L = self.current_lang
            C = self.cmd_locales

            self.log_summary(C["rpt_clean"][L], [
                f"{C['dns'][L]} : {s1}", 
                f"{C['temp'][L]} : {s2}", 
                f"{C['syscomp'][L]} : {s3}", 
                f"{C['socket'][L]} : {s4}", 
                f"{C['defrag'][L]} : {s5}"
            ])
            subprocess.run("taskkill /f /im explorer.exe", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, startupinfo=startupinfo)
            subprocess.Popen("explorer.exe", shell=True, startupinfo=startupinfo)
            self.set_working_state(False)
        threading.Thread(target=task, daemon=True).start()

    # === 3. 게임 모드 ===
    def start_gamemode(self):
        self.textbox.delete("0.0", "end")
        self.set_working_state(True, "GAME MODE ACTIVE...")
        msg = ">> KILLING PROCESSES..." if self.current_lang == "EN" else ">> 백그라운드 프로세스 정리 중..."
        self.log(msg)

        def task():
            targets = ["chrome.exe", "msedge.exe", "whale.exe", "firefox.exe", "KakaoTalk.exe", "Discord.exe", "Skype.exe", "Teams.exe", "Zoom.exe", "steam.exe", "EpicGamesLauncher.exe", "Battle.net.exe", "RiotClientServices.exe", "nxsteam.exe", "notepad++.exe", "Hwp.exe", "EXCEL.EXE", "WINWORD.EXE", "POWERPNT.EXE"]
            killed = 0
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            for exe in targets:
                try:
                    subprocess.check_output(f"taskkill /F /IM {exe} /T", shell=True, stderr=subprocess.DEVNULL, startupinfo=startupinfo)
                    killed += 1
                except: pass

            try:
                subprocess.run("taskkill /f /im explorer.exe", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, startupinfo=startupinfo)
                subprocess.Popen("explorer.exe", shell=True, startupinfo=startupinfo)
                explorer_status = "Refreshed"
            except: explorer_status = "Failed"

            L = self.current_lang
            C = self.cmd_locales

            self.log_summary(C["rpt_game"][L], [
                f"{C['target'][L]} : {len(targets)}", 
                f"{C['kill'][L]} : {killed}", 
                f"{C['exp'][L]} : {explorer_status}", 
                f"{C['mem'][L]} : {C['mem_opt'][L]}"
            ])
            self.set_working_state(False)
        threading.Thread(target=task, daemon=True).start()

    # === 4. 핑 테스트 ===
    def start_ping(self):
        self.textbox.delete("0.0", "end")
        self.set_working_state(True, "PING TEST...")
        msg = ">> ANALYZING LATENCY..." if self.current_lang == "EN" else ">> 지연 시간 분석 중..."
        self.log(msg)

        def task():
            def get_ms(ip):
                try:
                    out = self.get_cmd_output(f"ping {ip} -n 3")
                    m = re.search(r'(평균|Average)\s*=\s*(\d+)ms', out)
                    return int(m.group(2)) if m else None
                except: return None
            def get_stat(ms):
                if ms is None: return "FAIL"
                if ms <= 10: return "EXCELLENT" if self.current_lang == "EN" else "매우 좋음"
                if ms <= 30: return "GOOD" if self.current_lang == "EN" else "좋음"
                val = "NORMAL" if ms <= 60 else "SLOW"
                if self.current_lang == "KO":
                    val = "보통" if ms <= 60 else "느림"
                return val

            gms = get_ms("8.8.8.8")
            cms = get_ms("1.1.1.1")
            
            L = self.current_lang
            C = self.cmd_locales

            self.log_summary(C["rpt_net"][L], [
                f"{C['g_dns'][L]} : {gms}ms ({get_stat(gms)})",
                f"{C['cf_dns'][L]} : {cms}ms ({get_stat(cms)})"
            ])
            self.set_working_state(False)
        threading.Thread(target=task, daemon=True).start()

    # === 5. 내 정보 ===
    def start_info(self):
        self.textbox.delete("0.0", "end")
        self.set_working_state(True, "SCANNING SPECS...")
        msg = ">> GATHERING INFO..." if self.current_lang == "EN" else ">> 정보 수집 중..."
        self.log(msg)

        def task():
            try:
                cpu = self.get_cmd_output("wmic cpu get name").replace("Name", "").strip()
                mb = self.get_cmd_output("wmic baseboard get product").replace("Product", "").strip()
                gpu = self.get_cmd_output("wmic path win32_VideoController get name").replace("Name", "").strip()
                ram = self.get_cmd_output('powershell -Command "Write-Host ([math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2).ToString() + \' GB\')"')
                pub_ip = self.get_cmd_output("curl -s ifconfig.me")
                loc_ip = self.get_cmd_output('ipconfig | findstr "IPv4"')
                if loc_ip: loc_ip = loc_ip.split(":")[-1].strip()

                L = self.current_lang
                C = self.cmd_locales

                self.log_summary(C["rpt_spec"][L], [
                    f"CPU   : {cpu}", 
                    f"M/B   : {mb}", 
                    f"GPU   : {gpu}", 
                    f"RAM   : {ram}", 
                    f"L-IP  : {loc_ip}", 
                    f"P-IP  : {pub_ip}"
                ])
            except: self.log("[ERROR] SCAN FAILED")
            finally: self.set_working_state(False)
        threading.Thread(target=task, daemon=True).start()

if __name__ == "__main__":
    app = SystemMasterApp()
    app.mainloop()