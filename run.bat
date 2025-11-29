@echo off
echo ========================================
echo  BASI BOT - LAUNCH SCRIPT
echo  Multi-Agent Discord LLM Chatbot System
echo ========================================
echo.

if not exist venv (
    echo ERROR: Virtual environment not found!
    echo Please run build.bat first to set up the environment.
    echo.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

echo Starting BASI Bot...
echo Gradio UI will open in your browser automatically.
echo.
python main.py

if errorlevel 1 (
    echo.
    echo ERROR: Application crashed or failed to start.
    echo Check the error messages above.
    echo.
    pause
)
