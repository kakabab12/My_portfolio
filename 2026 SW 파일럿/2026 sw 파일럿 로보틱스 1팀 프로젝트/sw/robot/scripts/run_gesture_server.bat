@echo off
REM 윈도우 개발 PC에서 gesture_engine(Flask)을 실행한다.
cd /d "%~dp0.."
py -m src.server.app
