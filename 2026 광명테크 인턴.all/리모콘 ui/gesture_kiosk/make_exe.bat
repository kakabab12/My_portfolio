@echo off
chcp 65001 >nul
cd /d %~dp0

REM 2026-08-03: 빌드 로직을 build_exe_helper.py로 옮겼다 — 이 PC의 cmd.exe가
REM 한글이 섞인 배치를 처리하다 위치 계산이 틀어지는 버그가 있어(scripts\
REM run_launcher.py 독스트링 참고) 이 파일은 최소 런처로만 남긴다

set PY_CMD=
py -3.11 --version >nul 2>&1 && set PY_CMD=py -3.11
if not defined PY_CMD python --version 2>nul | findstr /C:"3.11" >nul && set PY_CMD=python
if not defined PY_CMD echo [FAIL] Python 3.11 not found - install.bat first

%PY_CMD% scripts\build_exe_helper.py
pause
