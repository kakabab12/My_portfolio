@echo off
chcp 65001 >nul
cd /d %~dp0

set PY_CMD=
py -3.11 --version >nul 2>&1 && set PY_CMD=py -3.11
if not defined PY_CMD python --version 2>nul | findstr /C:"3.11" >nul && set PY_CMD=python
if not defined PY_CMD echo [FAIL] Python 3.11 not found - install.bat first

%PY_CMD% scripts\build_dpad_overlay_exe_helper.py
pause
