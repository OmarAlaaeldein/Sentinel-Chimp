# Lite Mode Windows release builder (CI / local).
# Usage: powershell -ExecutionPolicy Bypass -File scripts/build_release.ps1
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
  "--exclude-module", "notebook"
)

$PyiArgs = @(
  "--noconfirm",
  "--clean",
  "--onefile",
  "--noconsole",
  "--name", "Sentinel",
  "--collect-submodules", "matplotlib"
) + $Exclude

$Icon = Join-Path $Root "logo.ico"
if (Test-Path $Icon) {
  $PyiArgs += @("--icon", $Icon)
}

$Splash = Join-Path $Root "loading.png"
if (Test-Path $Splash) {
  $PyiArgs += @("--splash", $Splash)
}

New-Item -ItemType Directory -Force -Path (Join-Path $Root "dist") | Out-Null

Write-Host "[INFO] Running PyInstaller (windows onefile lite)"
& $Python -m PyInstaller @PyiArgs $AppScript
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit $LASTEXITCODE" }

$Exe = Join-Path $Root "dist\Sentinel.exe"
if (-not (Test-Path $Exe)) {
  throw "Expected dist\Sentinel.exe missing"
}

$Stage = Join-Path $Root "dist\staging-windows"
if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
New-Item -ItemType Directory -Force -Path $Stage | Out-Null
Copy-Item $Exe (Join-Path $Stage "Sentinel.exe")

$Readme = @"
Sentinel Chimp — Windows x64 (Lite Mode)
========================================

1. Extract this zip.
2. Run Sentinel.exe

Notes:
- Lite Mode: FinBERT / PyTorch AI sentiment is NOT bundled.
  For AI features, clone the repo and run: python sentinel.py
- No Python installation required.
- Windows Defender may scan the unsigned exe on first run.
"@
Set-Content -Path (Join-Path $Stage "README-RUN.txt") -Value $Readme -Encoding UTF8

$OutZip = Join-Path $Root "dist\Sentinel-Windows-x64.zip"
if (Test-Path $OutZip) { Remove-Item -Force $OutZip }

Compress-Archive -Path (Join-Path $Stage "*") -DestinationPath $OutZip -Force
Remove-Item -Recurse -Force $Stage

Write-Host "[OK] Wrote $OutZip"
Get-ChildItem (Join-Path $Root "dist\*.zip") | Format-Table Name, Length
