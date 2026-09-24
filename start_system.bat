@echo off
REM Startup script for Quality AI Diagnostic System (Windows)
REM This script starts both the FastAPI backend and Streamlit frontend

echo Starting Quality AI Diagnostic System...
echo ==========================================

REM Check if .env file exists
if not exist .env (
    echo Error: .env file not found. Please create it first.
    exit /b 1
)

REM Start FastAPI backend in background
echo Starting FastAPI backend...
start "FastAPI Backend" python api/main.py

REM Wait a moment for the API to start
timeout /t 3 /nobreak > nul

REM Start Streamlit frontend
echo Starting Streamlit frontend...
start "Streamlit Frontend" streamlit run ui/app.py

echo ==========================================
echo System started successfully!
echo API: http://localhost:8000
echo UI: http://localhost:8501
echo ==========================================
echo Close this window to stop both services

REM Keep the script running
pause