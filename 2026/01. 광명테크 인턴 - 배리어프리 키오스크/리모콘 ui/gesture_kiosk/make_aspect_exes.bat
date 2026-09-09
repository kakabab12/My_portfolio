@echo off
chcp 65001 >nul
cd /d %~dp0

rem 화면비 고정 exe 6개 빌드 (2026-09-09 신설)
rem   head / eyebrow / forehead  x  16:9 데스크탑 / 9:16 키오스크
rem 빌드 로직은 scripts\build_aspect_exes_helper.py 한 곳에만 있다.
rem PyInstaller는 임시 venv에만 설치하고 끝나면 지운다(시스템 파이썬 안 건드림).

set PY_CMD=
py -3.11 --version >nul 2>&1 && set PY_CMD=py -3.11
if not defined PY_CMD python --version 2>nul | findstr /C:"3.11" >nul && set PY_CMD=python
if not defined PY_CMD echo [FAIL] Python 3.11 not found - install.bat first

%PY_CMD% scripts\build_aspect_exes_helper.py
pause
