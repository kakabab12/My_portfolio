@echo off
title SYSTEM MASTER TOOL (COMPACT)
:: 유니코드(UTF-8) 설정
chcp 65001 > nul
:: 0b = 검은 배경에 하늘색 글씨
color 0b

:: ============================================================
:: [자동 관리자 권한 획득]
:: ============================================================
fltmc >nul 2>&1 || (
    echo [!] 관리자 권한을 요청합니다...
    powershell Start-Process -FilePath '%0' -Verb RunAs
    exit /b
)

:: 창 크기 설정 (화면에 딱 맞게 축소: 가로 100, 세로 50)
mode con: cols=110 lines=40
cls

:MENU
cls
echo.
echo.
:: --- 메인 아스키 아트 ---
echo.
echo                  -.            -.           .,.            ,,            ,,            -.            
echo                  ~$=~          ~$*,          !==           ~==           ~==.          ;$:           
echo                   ~#~       .. .!$,          .#@            !@            !@,          -#!           
echo           ,~. -#@ ~#~   !@#@@@: ;$,           #@      ,-    !@         :. !@,     ,.   ,$!           
echo           ,$*,.#@ -#~    ..-$$. ;#-. ,;:,-!#; #@    .:$#=.  !@    ;##=#@; !@,    -*#;. ,#!,.         
echo           ,*$**@@ ~#:,    .!#-  ;#=$~ ~@#!;~. #@    !#~~$=. !@        #$  !@,   ~#*!#$ ,#=*=.        
echo           .**, #@ -@==~   ;$~   ;$~~.  @=     #@-~,.==. -#~ !@       :@~  !@,  .!=, !@ ,$=~-         
echo           .*=::@@ -#!;-  ;*-    ~;     @=     #@*=:.=#,.!$, !@      -#;   !@,  .;$~,*@ ,#!           
echo           .**~-$: -$-  .;~,-,  ~=;     @=  .~!@@~-. ~#$=#;  !@     ,=!    !@,   -$$=@: ,#=!!         
echo            ,.. ,  -=,   ,  :$~ .$=     @@$$=:,#@     ,~~,.  !@    .!;.    !@,    ,-~,  ,$$*;         
echo             .!#@  ,;       ~#@@#@$     ;$!-.  #@.           !@   .;~      !@,          ,#!.          
echo               =@           ~#:  =$            #@            !@   .        !@,          ,#!           
echo               *@,,,:!,     ~#!~-$=            ##            ;#            !$.          ,$;           
echo               ,******-     -*;~~!;            =!            :;            ;:           ,*~           
echo                             ,   .             -             ,.            ,.            ,
echo.
echo.
echo                            =========================================
echo                                   [ SYSTEM MASTER CONTROL ]
echo                            =========================================
echo.
echo                             1. 시스템 업데이트 (Update All)
echo.
echo                             2. 시스템 청소 (Clean + Boost)
echo.
echo                             3. 프로세스 초기화 (Game Mode)
echo.
echo                             4. 인터넷 핑 테스트 (Ping Check)
echo.
echo                             5. 내 IP 및 PC 정보 확인 (My Info)
echo.
echo                             6. 종료 (Exit)
echo.
echo                            =========================================
echo.
set /p choice="                             원하는 기능의 번호 입력 (1~6): "

if "%choice%"=="1" goto UPDATE
if "%choice%"=="2" goto CLEAN
if "%choice%"=="3" goto KILLPROCESS
if "%choice%"=="4" goto PINGTEST
if "%choice%"=="5" goto MYINFO
if "%choice%"=="6" goto EXIT
goto MENU

:UPDATE
cls
echo.
echo.
echo                            =========================================
echo                                [!] 시스템 업데이트를 시작합니다
echo                            =========================================
echo.
timeout /t 1 > nul

echo [1/6] 프로그램 업데이트 (Winget)
winget upgrade --all --include-unknown --accept-package-agreements --accept-source-agreements

echo.
echo [2/6] 파이썬 PIP 업데이트
python -m pip install --upgrade pip

echo.
echo [3/6] 리눅스 서브시스템(WSL) 업데이트
wsl --update

echo.
echo [4/6] 윈도우 + 드라이버 검색
UsoClient StartScan
UsoClient StartDownload
UsoClient StartInstall

echo.
echo [5/6] MS 스토어 업데이트
start ms-windows-store://downloadsandupdates

echo.
echo [6/6] 그래픽카드 드라이버 (GeForce)
winget upgrade Nvidia.GeForceExperience

echo.
echo                            =========================================
echo                                  [완료] 업데이트가 끝났습니다.
echo                            =========================================
echo.
echo 아무 키나 누르면 메뉴로 돌아갑니다.
pause > nul
goto MENU

:CLEAN
cls
echo.
echo.
echo                            =========================================
echo                              [!] 시스템 최적화를 시작합니다
echo                            =========================================
echo.
timeout /t 1 > nul

echo [1/8] DNS 및 ARP 캐시 초기화
ipconfig /flushdns
arp -d *

echo.
echo [2/8] 임시 파일 및 시스템 찌꺼기 제거
del /s /q /f "%TEMP%\*.*" > nul 2>&1
del /s /q /f "C:\Windows\Temp\*.*" > nul 2>&1

echo.
echo [3/8] 윈도우 업데이트 파일 정리
dism /online /Cleanup-Image /StartComponentCleanup

echo.
echo [4/8] 윈도우 이미지 원본 복구
dism /online /Cleanup-Image /RestoreHealth

echo.
echo [5/8] 시스템 파일 무결성 검사
sfc /scannow

echo.
echo [6/8] 네트워크 소켓 초기화
netsh winsock reset

echo.
echo [7/8] 모든 디스크 최적화
defrag /C /O

echo.
echo [8/8] 탐색기 리프레시
taskkill /f /im explorer.exe & start explorer.exe

echo.
echo                            =========================================
echo                                [완료] 최적화가 끝났습니다.
echo                            =========================================
echo.
echo 아무 키나 누르면 메뉴로 돌아갑니다.
pause > nul
goto MENU

:KILLPROCESS
cls
echo.
echo.
echo                            =========================================
echo                              [!] 프로세스 초기화 (게임 모드)
echo                            =========================================
echo.
echo    [주의] 현재 작업 중인 인터넷 창, 카톡, 게임, 문서가 강제로 종료됩니다.
echo    정말 진행하시겠습니까? (메모리 확보 및 렉 제거)
echo.
echo                            =========================================
echo.
set /p confirm="                             진행하려면 'Y', 취소하려면 'N' 입력: "

if /i "%confirm%" neq "Y" goto MENU

echo.
echo [1/3] 주요 메모리 점유 프로그램 강제 종료 중...

:: 브라우저
taskkill /F /IM chrome.exe /T >nul 2>&1
taskkill /F /IM msedge.exe /T >nul 2>&1
taskkill /F /IM whale.exe /T >nul 2>&1
taskkill /F /IM firefox.exe /T >nul 2>&1

:: 메신저
taskkill /F /IM KakaoTalk.exe /T >nul 2>&1
taskkill /F /IM Discord.exe /T >nul 2>&1
taskkill /F /IM Skype.exe /T >nul 2>&1
taskkill /F /IM Teams.exe /T >nul 2>&1
taskkill /F /IM Zoom.exe /T >nul 2>&1

:: 게임 런처
taskkill /F /IM steam.exe /T >nul 2>&1
taskkill /F /IM EpicGamesLauncher.exe /T >nul 2>&1
taskkill /F /IM Battle.net.exe /T >nul 2>&1
taskkill /F /IM RiotClientServices.exe /T >nul 2>&1
taskkill /F /IM nxsteam.exe /T >nul 2>&1

:: 기타 앱
taskkill /F /IM notepad++.exe /T >nul 2>&1
taskkill /F /IM Hwp.exe /T >nul 2>&1
taskkill /F /IM EXCEL.EXE /T >nul 2>&1
taskkill /F /IM WINWORD.EXE /T >nul 2>&1
taskkill /F /IM POWERPNT.EXE /T >nul 2>&1

echo.
echo [2/3] 윈도우 탐색기(Explorer) 재시작
taskkill /f /im explorer.exe >nul 2>&1
start explorer.exe

echo.
echo [3/3] 대기 메모리 정리 완료.
echo.
echo                            =========================================
echo                              [완료] 시스템이 초기화되었습니다.
echo                            =========================================
echo.
echo 아무 키나 누르면 메뉴로 돌아갑니다.
pause > nul
goto MENU

:PINGTEST
cls
echo.
echo.
echo                            =========================================
echo                                [!] 인터넷 연결 정밀 점검
echo                            =========================================
echo.
echo.
echo [1/2] 구글 서버 (8.8.8.8) 응답 속도 측정 중...
echo ------------------------------------------------------------------
ping 8.8.8.8 -n 5
echo ------------------------------------------------------------------
echo.
echo [2/2] 클라우드플레어 (1.1.1.1) 응답 속도 측정 중...
echo ------------------------------------------------------------------
ping 1.1.1.1 -n 5
echo ------------------------------------------------------------------
echo.
echo                            =========================================
echo                              [결과] 손실율 0%% 면 정상입니다.
echo                              [참고] 시간=10ms 이하면 아주 빠름.
echo                            =========================================
echo.
echo 아무 키나 누르면 메뉴로 돌아갑니다.
pause > nul
goto MENU

:MYINFO
cls
echo.
echo.
echo                            =========================================
echo                              [!] 내 PC 및 네트워크 정보 확인
echo                            =========================================
echo.
echo.
echo [1] 하드웨어 정보
echo ------------------------------------------------------------------
echo CPU 모델:
wmic cpu get name | findstr /v "Name"
echo.
echo 그래픽카드(GPU):
wmic path win32_VideoController get name | findstr /v "Name"
echo.
echo 전체 메모리(RAM):
powershell -Command "Write-Host ([math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2).ToString() + ' GB')"
echo ------------------------------------------------------------------
echo.
echo.
echo [2] 네트워크 정보 (IP 주소)
echo ------------------------------------------------------------------
echo 내부 IP (Local): 
ipconfig | findstr "IPv4"
echo.
echo 외부 IP (Public): 
curl -s ifconfig.me
echo.
echo ------------------------------------------------------------------
echo.
echo                            =========================================
echo                                [정보] 확인이 완료되었습니다.
echo                            =========================================
echo.
echo 아무 키나 누르면 메뉴로 돌아갑니다.
pause > nul
goto MENU

:EXIT
exit