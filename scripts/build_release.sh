#!/usr/bin/env bash
# Cross-platform Lite Mode release builder for Linux and macOS CI/local.
# Usage: ./scripts/build_release.sh [--platform linux|macos]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PLATFORM=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform)
      PLATFORM="${2:-}"
      shift 2
      ;;
    --platform=*)
      PLATFORM="${1#*=}"
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--platform linux|macos]"
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$PLATFORM" ]]; then
  case "$(uname -s)" in
    Linux*)  PLATFORM="linux" ;;
    Darwin*) PLATFORM="macos" ;;
    *)
      echo "[ERROR] Unsupported OS; pass --platform linux|macos" >&2
      exit 1
      ;;
  esac
fi

PLATFORM="$(echo "$PLATFORM" | tr '[:upper:]' '[:lower:]')"
case "$PLATFORM" in
  linux|macos) ;;
  *)
    echo "[ERROR] --platform must be linux or macos (got: $PLATFORM)" >&2
    exit 1
    ;;
esac

APP_SCRIPT="$ROOT/sentinel.py"
if [[ ! -f "$APP_SCRIPT" ]]; then
  echo "[ERROR] Missing entrypoint: $APP_SCRIPT" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

echo "[INFO] Platform: $PLATFORM"
echo "[INFO] Python: $PYTHON_BIN"
echo "[INFO] Entry: $APP_SCRIPT"

EXCLUDE_ARGS=(
  --exclude-module torch
  --exclude-module transformers
  --exclude-module tensorflow
  --exclude-module tensorboard
  --exclude-module nvidia
  --exclude-module scipy
  --exclude-module accelerate
  --exclude-module tkinter.test
  --exclude-module notebook
)

mkdir -p "$ROOT/dist"
rm -rf "$ROOT/build" "$ROOT/dist/Sentinel" "$ROOT/dist/Sentinel.app" "$ROOT/dist/Sentinel.exe" 2>/dev/null || true

if [[ "$PLATFORM" == "linux" ]]; then
  PYI_ARGS=(
    --noconfirm
    --clean
    --onefile
    --noconsole
    --name Sentinel
    --collect-submodules matplotlib
    "${EXCLUDE_ARGS[@]}"
  )
  if [[ -f "$ROOT/logo.ico" ]]; then
    PYI_ARGS+=(--icon "$ROOT/logo.ico")
  fi
  if [[ -f "$ROOT/loading.png" ]]; then
    PYI_ARGS+=(--splash "$ROOT/loading.png")
  fi

  echo "[INFO] Running PyInstaller (linux onefile lite)"
  "$PYTHON_BIN" -m PyInstaller "${PYI_ARGS[@]}" "$APP_SCRIPT"

  STAGE="$ROOT/dist/staging-linux"
  rm -rf "$STAGE"
  mkdir -p "$STAGE"
  if [[ -f "$ROOT/dist/Sentinel" ]]; then
    cp "$ROOT/dist/Sentinel" "$STAGE/Sentinel"
    chmod +x "$STAGE/Sentinel"
  else
    echo "[ERROR] Expected dist/Sentinel binary missing" >&2
    exit 1
  fi

  cat > "$STAGE/README-RUN.txt" << 'README'
Sentinel Chimp — Linux x64 (Lite Mode)
======================================

1. Extract this zip.
2. Make the binary executable if needed:
     chmod +x Sentinel
3. Run:
     ./Sentinel

Notes:
- Lite Mode: FinBERT / PyTorch AI sentiment is NOT bundled.
  For AI features, clone the repo and run: python sentinel.py
- Requires a desktop environment with Tk (most desktop Linux distros).
- If the binary fails to start, install tk system packages, e.g.:
     sudo apt install python3-tk   # Debian/Ubuntu (runtime usually bundled)
README

  OUT="$ROOT/dist/Sentinel-Linux-x64.zip"
  rm -f "$OUT"
  (cd "$STAGE" && zip -r "$OUT" Sentinel README-RUN.txt)
  rm -rf "$STAGE"
  echo "[OK] Wrote $OUT"

elif [[ "$PLATFORM" == "macos" ]]; then
  PYI_ARGS=(
    --noconfirm
    --clean
    --windowed
    --onedir
    --name Sentinel
    --collect-submodules matplotlib
    "${EXCLUDE_ARGS[@]}"
  )
  if [[ -f "$ROOT/logo.icns" ]]; then
    PYI_ARGS+=(--icon "$ROOT/logo.icns")
  else
    echo "[INFO] logo.icns not found; using default app icon"
  fi
  # Splash is not supported on macOS PyInstaller builds — skip intentionally.

  echo "[INFO] Running PyInstaller (macos onedir windowed lite)"
  "$PYTHON_BIN" -m PyInstaller "${PYI_ARGS[@]}" "$APP_SCRIPT"

  if [[ ! -d "$ROOT/dist/Sentinel.app" ]]; then
    echo "[ERROR] Expected dist/Sentinel.app missing" >&2
    ls -la "$ROOT/dist" || true
    exit 1
  fi

  STAGE="$ROOT/dist/staging-macos"
  rm -rf "$STAGE"
  mkdir -p "$STAGE"
  cp -R "$ROOT/dist/Sentinel.app" "$STAGE/Sentinel.app"

  cat > "$STAGE/README-RUN.txt" << 'README'
Sentinel Chimp — macOS (Lite Mode, UNSIGNED)
============================================

This build is NOT code-signed and NOT notarized.

Gatekeeper / first launch:
1. Extract this zip.
2. Right-click (or Control-click) Sentinel.app → Open
3. Confirm Open in the dialog.

Or remove quarantine after download:
  xattr -dr com.apple.quarantine Sentinel.app
  open Sentinel.app

Notes:
- Lite Mode: FinBERT / PyTorch AI sentiment is NOT bundled.
  For AI features, clone the repo and run: python sentinel.py
README

  OUT="$ROOT/dist/Sentinel-macOS-unsigned.zip"
  rm -f "$OUT"
  (cd "$STAGE" && zip -r "$OUT" Sentinel.app README-RUN.txt)
  rm -rf "$STAGE"
  echo "[OK] Wrote $OUT"
fi

ls -lh "$ROOT/dist"/*.zip 2>/dev/null || true
