@echo off
cd /d "%~dp0"
start "" msedge "http://127.0.0.1:8000"
py -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
pause
