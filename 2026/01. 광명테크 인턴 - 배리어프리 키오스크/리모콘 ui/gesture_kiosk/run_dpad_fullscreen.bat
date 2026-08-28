@echo off
cd /d %~dp0

echo ============================================================
echo  gesture_kiosk D-pad UI demo (fullscreen)
echo ============================================================
echo Quit: q or ESC
echo.

py main_dpad.py --fullscreen

echo.
echo [Done] Press any key to close this window
pause >nul
