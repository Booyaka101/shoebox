@echo off
rem shoebox launcher: creates the venv, installs the CUDA torch build (or the
rem CPU build when no NVIDIA GPU is present) plus dependencies, then serves
rem the UI on http://127.0.0.1:8545
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
  where nvidia-smi >nul 2>nul
  if errorlevel 1 (
    echo No NVIDIA GPU detected - installing the CPU build of PyTorch...
    echo Restoring will work but will take minutes per photo.
    "%PY%" -m pip install torch torchvision
  ) else (
    echo Installing PyTorch with CUDA support - a large download, once only...
    "%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
  )
  if errorlevel 1 (
    echo.
    echo PyTorch installation failed. Check your internet connection and run
    echo run.bat again. If it keeps failing, install manually:
    echo   .venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    pause
    exit /b 1
  )
)
"%PY%" -c "import fastapi, cv2, requests" >nul 2>nul
if errorlevel 1 (
  echo Installing dependencies...
  "%PY%" -m pip install -r requirements.txt --quiet
  if errorlevel 1 (
    echo Dependency installation failed - check your internet connection and run run.bat again.
    pause
    exit /b 1
  )
)

echo Starting shoebox at http://127.0.0.1:8545  (Ctrl+C to stop)
"%PY%" -m app.main
pause
