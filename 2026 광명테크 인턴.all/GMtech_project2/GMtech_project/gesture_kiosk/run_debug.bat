@echo off
chcp 65001 >nul
cd /d %~dp0

REM 2026-08-07 신설 — run.exe를 더블클릭하면 카메라 창이 안 뜬다(기본값 — main.py
REM --debug 옵션이 있어야 켜짐, 또는 콘솔에 cam on 입력). 매번 명령줄로 옵션을
REM 안 붙여도 되게 이 파일이 --debug를 붙여서 대신 실행한다

run.exe --debug
