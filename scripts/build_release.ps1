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
