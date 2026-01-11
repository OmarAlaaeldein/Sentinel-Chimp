@echo off
setlocal
set ROOT=%~dp0
cd /d "%ROOT%"

rem Use the specified conda env
set CONDA_ENV=%USERPROFILE%\miniconda3\envs\venv
set CONDA_ACTIVATE=%USERPROFILE%\miniconda3\Scripts\activate.bat
if not exist "%CONDA_ENV%\python.exe" (
    echo [ERROR] Conda env not found at %CONDA_ENV%
    pause
    exit /b 1
)
if not exist "%CONDA_ACTIVATE%" (
    echo [ERROR] Conda activate script not found at %CONDA_ACTIVATE%
    pause
    exit /b 1
)
call "%CONDA_ACTIVATE%" "%CONDA_ENV%" || (
    echo [ERROR] Failed to activate conda env at %CONDA_ENV%
    pause
    exit /b 1
)
echo Using Python: %PYTHON%

rem Ensure PyInstaller is available in the venv
python -m pyinstaller --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing PyInstaller into venv...
    python -m pip install -q pyinstaller || (
        echo [ERROR] Failed to install PyInstaller.
        pause
        exit /b 1
    )
)

rem Build one-file GUI exe with bundled models (requires caches present)
pyinstaller --onefile --noconsole --name Sentinel "%ROOT%Sentinel.py" ^
    --add-data "%ROOT%my_finbert_model;my_finbert_model" ^
    --add-data "%ROOT%my_distilbert_model;my_distilbert_model"

endlocal
