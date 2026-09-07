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
  "--exclude-module", "unittest",
  "--exclude-module", "doctest",
  "--exclude-module", "pydoc",
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
  "--collect-submodules", "matplotlib"
) + $Exclude

# UPX binary compression when `upx` is on PATH (effective on Windows).
# MSVC runtimes are excluded (stability). Skipped silently when absent.
$Upx = Get-Command upx -ErrorAction SilentlyContinue
if ($Upx) {
  $UpxDir = Split-Path -Parent $Upx.Source
  Write-Host "[INFO] UPX found at $UpxDir — enabling binary compression"
  $PyiArgs += @(
    "--upx-dir", $UpxDir,
    "--upx-exclude", "vcruntime*.dll",
    "--upx-exclude", "msvcp*.dll",
    "--upx-exclude", "msvcr*.dll"
  )
} else {
  Write-Host "[INFO] UPX not found — skipping binary compression (choco install upx to enable)"
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

# Remove stale zip artifacts from prior packaging
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
