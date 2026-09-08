# Lite Mode Windows release builder (CI / local).
# Usage: powershell -ExecutionPolicy Bypass -File scripts/build_release.ps1
# Output: dist/Sentinel.exe (PyInstaller onefile, no zip)
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$AppScript = Join-Path $Root "sentinel.py"
if (-not (Test-Path $AppScript)) {
  throw "Missing entrypoint: $AppScript"
}

$Python = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
Write-Host "[INFO] Python: $Python"
Write-Host "[INFO] Entry: $AppScript"

# In conda environments on Windows, tcl/tk shared libraries live in Library\bin.
# Ensure Library\bin is on PATH so PyInstaller's splash feature can resolve them.
$ResolvedPython = if (Test-Path $Python) { (Resolve-Path $Python).Path } else {
  $cmd = Get-Command $Python -ErrorAction SilentlyContinue
  if ($cmd) {
    if ($cmd.Source) { $cmd.Source } elseif ($cmd.Path) { $cmd.Path } else { $null }
  } else { $null }
}

if ($ResolvedPython) {
  $PythonDir = Split-Path -Parent $ResolvedPython
  if ($PythonDir -and (Test-Path $PythonDir)) {
    $CondaLibBin = Join-Path $PythonDir "Library\bin"
    if (Test-Path $CondaLibBin) {
      Write-Host "[INFO] Added Conda Library\bin to PATH: $CondaLibBin"
      $env:PATH = "$CondaLibBin;" + $env:PATH
    }
  }
}

$Exclude = @(
  "--exclude-module", "torch",
  "--exclude-module", "transformers",
  "--exclude-module", "tensorflow",
  "--exclude-module", "tensorboard",
  "--exclude-module", "nvidia",
  "--exclude-module", "scipy",
  "--exclude-module", "accelerate",
  "--exclude-module", "tkinter.test",
  "--exclude-module", "notebook",
  # --- v2.2.2 size trims (no feature impact) ---
  "--exclude-module", "pandas.tests",
  "--exclude-module", "numpy.tests",
  "--exclude-module", "matplotlib.tests",
  # NOTE: Do NOT exclude unittest/pydoc — pyparsing (matplotlib dep) unconditionally
  # imports pyparsing.testing which requires unittest; pandas/pyarrow imports pydoc.
  "--exclude-module", "IPython",
  "--exclude-module", "ipykernel",
  "--exclude-module", "matplotlib.backends.backend_qt5agg",
  "--exclude-module", "matplotlib.backends.backend_qt6agg",
  "--exclude-module", "matplotlib.backends.backend_qtagg",
  "--exclude-module", "matplotlib.backends.backend_wx",
  "--exclude-module", "matplotlib.backends.backend_wxagg",
  "--exclude-module", "matplotlib.backends.backend_gtk3agg",
  "--exclude-module", "matplotlib.backends.backend_gtk3cairo",
  "--exclude-module", "matplotlib.backends.backend_gtk4agg",
  "--exclude-module", "matplotlib.backends.backend_gtk4cairo",
  "--exclude-module", "matplotlib.backends.backend_nbagg",
  "--exclude-module", "matplotlib.backends.backend_cairo"
  # NOTE: plotly is intentionally KEPT — the 3D HTML export needs it.
)

$PyiArgs = @(
  "--noconfirm",
  "--clean",
  "--onefile",
  "--noconsole",
  "--name", "Sentinel",
  "--collect-submodules", "matplotlib",
  "--collect-submodules", "core",
  "--collect-submodules", "main"
) + $Exclude

# UPX binary compression when `upx` is on PATH (effective on Windows).
# MSVC runtimes are excluded (stability). Skipped silently when absent.
$Upx = Get-Command upx -ErrorAction SilentlyContinue
if ($Upx) {
  $UpxPath = if ($Upx.Source) { $Upx.Source } elseif ($Upx.Path) { $Upx.Path } else { $null }
  $UpxDir = if ($UpxPath) { Split-Path -Parent $UpxPath } else { $null }
  if ($UpxDir -and (Test-Path $UpxDir)) {
    Write-Host "[INFO] UPX found at $UpxDir - enabling binary compression"
    $PyiArgs += @(
      "--upx-dir", $UpxDir,
      "--upx-exclude", "vcruntime*.dll",
      "--upx-exclude", "msvcp*.dll",
      "--upx-exclude", "msvcr*.dll"
    )
  } else {
    Write-Host "[INFO] UPX found on PATH - enabling binary compression"
    $PyiArgs += @(
      "--upx-exclude", "vcruntime*.dll",
      "--upx-exclude", "msvcp*.dll",
      "--upx-exclude", "msvcr*.dll"
    )
  }
} else {
  Write-Host "[INFO] UPX not found - skipping binary compression (choco install upx to enable)"
}

$Icon = Join-Path $Root "logo.ico"
if (Test-Path $Icon) {
  $PyiArgs += @("--icon", $Icon)
}

$Splash = Join-Path $Root "loading.png"
if (Test-Path $Splash) {
  $PyiArgs += @("--splash", $Splash)
}

New-Item -ItemType Directory -Force -Path (Join-Path $Root "dist") | Out-Null

# Clean up running Sentinel processes and stale artifacts
Stop-Process -Name Sentinel* -Force -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 500
Remove-Item (Join-Path $Root "dist\Sentinel.exe") -Force -ErrorAction SilentlyContinue
Get-ChildItem (Join-Path $Root "dist\*.zip") -ErrorAction SilentlyContinue | Remove-Item -Force

Write-Host "[INFO] Running PyInstaller (windows onefile lite)"
& $Python -m PyInstaller @PyiArgs $AppScript
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit $LASTEXITCODE" }

$Exe = Join-Path $Root "dist\Sentinel.exe"
if (-not (Test-Path $Exe)) {
  throw "Expected dist\Sentinel.exe missing"
}

Write-Host "[OK] Wrote $Exe"
Get-Item $Exe | Format-Table Name, Length
