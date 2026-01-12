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

call "%CONDA_ACTIVATE%" "%CONDA_ENV%"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to activate conda env.
    pause
    exit /b 1
)

echo Using Python from: %CONDA_ENV%

rem --- BUILD COMMAND ---
rem Added --clean to clear cache (prevents old builds from messing up new ones)
rem Added exclude-module for 'PIL' (Pillow) if you aren't using images, saves 5-10MB.
rem Added exclude-module for 'scipy' if you aren't using it (it's huge).

pyinstaller --onefile --noconsole --clean ^
    --name "Sentinel" ^
    --icon="logo.ico" ^
    --splash "loading.png" ^
    --exclude-module "torch" ^
    --exclude-module "transformers" ^
    --exclude-module "tensorflow" ^
    --exclude-module "tensorboard" ^
    --exclude-module "nvidia" ^
    --exclude-module "tkinter.test" ^
    --exclude-module "notebook" ^
    --exclude-module "scipy" ^
    --collect-submodules "matplotlib" ^
    "%ROOT%Sentinel.py" 

echo.
echo Build Complete. Check the 'dist' folder.
pause
endlocal