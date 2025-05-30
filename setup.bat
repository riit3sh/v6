@echo off
REM Setup script for Cloby project on Windows

REM Check Python version
python --version
if errorlevel 1 (
    echo Python is not installed or not in PATH. Please install Python 3.8 or higher.
    exit /b 1
)

REM Create virtual environment
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install dependencies
pip install -r requirements.txt

echo Setup complete. To run Cloby:
echo 1. Make sure Ollama is installed and running with the Mistral model.
echo 2. Activate the virtual environment: call venv\Scripts\activate.bat
echo 3. Run the app: python cloby.py
echo 4. Open the URL provided by Gradio in your browser.

pause
