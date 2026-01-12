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

rem FIX: Proper syntax for activation error handling 
call "%CONDA_ACTIVATE%" "%CONDA_ENV%"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to activate conda env at %CONDA_ENV% 
    pause
    exit /b 1
)

echo Using Python: %CONDA_ENV%\python.exe

rem Build a slim version of Sentinel by explicitly excluding AI libraries
pyinstaller --onefile --noconsole ^
    --name "Sentinel" ^
    --exclude-module "torch" ^
    --exclude-module "transformers" ^
    --exclude-module "tensorflow" ^
    --exclude-module "tensorboard" ^
    --exclude-module "nvidia" ^
    --collect-submodules "matplotlib" ^
    "%ROOT%Sentinel.py" 

echo Build Complete.
pause
endlocal