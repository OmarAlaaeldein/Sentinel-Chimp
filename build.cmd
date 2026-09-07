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
    --exclude-module "accelerate" ^
    --exclude-module "pandas.tests" ^
    --exclude-module "numpy.tests" ^
    --exclude-module "matplotlib.tests" ^
    --exclude-module "IPython" ^
    --exclude-module "ipykernel" ^
    --exclude-module "matplotlib.backends.backend_qt5agg" ^
    --exclude-module "matplotlib.backends.backend_qt6agg" ^
    --exclude-module "matplotlib.backends.backend_qtagg" ^
    --exclude-module "matplotlib.backends.backend_wx" ^
    --exclude-module "matplotlib.backends.backend_wxagg" ^
    --exclude-module "matplotlib.backends.backend_gtk3agg" ^
    --exclude-module "matplotlib.backends.backend_gtk3cairo" ^
    --exclude-module "matplotlib.backends.backend_gtk4agg" ^
    --exclude-module "matplotlib.backends.backend_gtk4cairo" ^
    --exclude-module "matplotlib.backends.backend_nbagg" ^
    --exclude-module "matplotlib.backends.backend_cairo" ^
    --collect-submodules "matplotlib" ^
    "%ROOT%sentinel.py"

echo.
echo Build Complete. Check the 'dist' folder.
pause
endlocal
