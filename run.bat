@echo off
title PaperMind AI - Research Intelligence Engine
echo ====================================================
echo  PaperMind AI - Research Intelligence Engine
echo ====================================================
echo Initializing server and opening browser...

rem Free up port 8501 if an old process is stuck
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8501 ^| findstr LISTENING') do taskkill /f /pid %%a >nul 2>&1

rem Open browser directly to localhost:8501
start http://localhost:8501

rem Start Streamlit
where streamlit >nul 2>nul
if %ERRORLEVEL% equ 0 (
    streamlit run app.py --server.port 8501 --server.headless false
) else (
    "C:\Users\YASH\anaconda3\python.exe" -m streamlit run app.py --server.port 8501 --server.headless false
)
pause
