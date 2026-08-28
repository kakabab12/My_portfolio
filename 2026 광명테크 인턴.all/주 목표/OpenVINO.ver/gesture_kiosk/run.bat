@echo off
chcp 65001 >nul
cd /d %~dp0

if not exist venv_win\Scripts\activate.bat (
    echo [FAIL] 설치가 안 되어 있습니다 — install.bat 을 먼저 실행하세요
    exit /b 1
)
call venv_win\Scripts\activate.bat

echo [INFO] 제스처 엔진 시작 — 이벤트가 이 콘솔(stdout)에 GESTURE^| 한 줄씩 찍힙니다
echo        델파이 연동: 델파이가 run.bat 을 자식 프로세스로 실행 (docs\델파이7_연동가이드.md)
echo        종료: 이 창에서 Ctrl+C 또는 디버그 창(카메라·계기판)에서 q/ESC
REM 2026-07-27: 사용자 요청 — run.bat 기본 실행에도 디버그 창(카메라 화면) 표시.
REM   %*로 추가 인자를 더 받을 수 있게 유지 (예: --config 등)
python scripts\run_demo.py --debug %*
