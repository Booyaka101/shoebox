@echo off
rem shoebox launcher: creates the venv, installs the CUDA torch build plus
rem dependencies, then serves the UI on http://127.0.0.1:8545
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo Python 3.11 is required. Install it from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" during setup.
  pause
  exit /b 1
)

if not exist .venv\Scripts\python.exe (
  echo Creating virtual environment...
  python -m venv .venv
)
set "PY=%~dp0.venv\Scripts\python.exe"

"%PY%" -m pip install --upgrade pip --quiet
"%PY%" -c "import torch" >nul 2>nul
if errorlevel 1 (
  echo Installing PyTorch with CUDA support - this is a large download, once only...
  "%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
)
"%PY%" -c "import fastapi, cv2, requests" >nul 2>nul
if errorlevel 1 (
  echo Installing dependencies...
  "%PY%" -m pip install -r requirements.txt --quiet
)

echo Starting shoebox at http://127.0.0.1:8545  (Ctrl+C to stop)
"%PY%" -m app.main
pause
