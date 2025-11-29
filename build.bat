@echo off
echo ========================================
echo  BASI BOT - BUILD SCRIPT
echo  Multi-Agent Discord LLM Chatbot System
echo ========================================
echo.

echo [1/5] Creating Python virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    echo Make sure Python 3.8+ is installed and in PATH
    pause
    exit /b 1
)
echo Virtual environment created successfully.
echo.

echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo.

echo [3/5] Upgrading pip...
python -m pip install --upgrade pip
echo.

echo [4/5] Installing required packages...
echo Installing: gradio discord.py openai cryptography requests chromadb python-chess english-words
pip install gradio discord.py openai cryptography requests chromadb python-chess english-words
if errorlevel 1 (
    echo ERROR: Failed to install packages
    pause
    exit /b 1
)
echo.

echo [5/5] Build complete!
echo.
echo ========================================
echo  BUILD SUCCESSFUL
echo ========================================
echo.
echo To start the bot, run: run.bat
echo.
pause
