#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

APP_SCRIPT="$ROOT/sentinel.py"
PYTHON_BIN="${PYTHON_BIN:-}"
MODE_REQUEST="auto"
MODE=""
BUNDLE_MODE="onedir"
INSTALL_DEPS=0

usage() {
  echo "Usage: ./build_macos.sh [--auto|--lite|--full] [--onedir|--onefile] [--install-deps]"
  echo
  echo "  --auto     Detect mode from self.use_sentiment in sentinel.py (default)"
  echo "  --lite     Exclude heavy AI dependencies"
  echo "  --full     Include AI dependencies from requirements.txt"
  echo "  --onedir   Build Sentinel.app as a directory bundle (default)"
  echo "  --onefile  Build a single-file executable bundle"
  echo "  --install-deps  Install/upgrade dependencies into current Python env"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --auto)
      MODE_REQUEST="auto"
      ;;
    --lite)
      MODE_REQUEST="lite"
      ;;
    --full)
      MODE_REQUEST="full"
      ;;
    --onedir)
      BUNDLE_MODE="onedir"
      ;;
    --onefile)
      BUNDLE_MODE="onefile"
      ;;
    --install-deps)
      INSTALL_DEPS=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1"
      usage
      exit 1
      ;;
  esac
  shift
done

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "[WARN] This script is intended for macOS. Current OS: $(uname -s)"
fi

if [[ ! -f "$APP_SCRIPT" ]]; then
  echo "[ERROR] Could not find sentinel.py at: $APP_SCRIPT"
  exit 1
fi

detect_sentiment_flag() {
  local assignment
  assignment="$(grep -Eo 'self\.use_sentiment\s*=\s*(True|False)' "$APP_SCRIPT" | head -n1 || true)"
  if [[ -z "$assignment" ]]; then
    return 1
  fi

  assignment="${assignment##*=}"
  assignment="${assignment//[[:space:]]/}"

  case "$assignment" in
    True|False)
      echo "$assignment"
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

if [[ "$MODE_REQUEST" == "auto" ]]; then
  if SENTIMENT_FLAG="$(detect_sentiment_flag)"; then
    if [[ "$SENTIMENT_FLAG" == "True" ]]; then
      MODE="full"
    else
      MODE="lite"
    fi
    echo "[INFO] Detected self.use_sentiment = $SENTIMENT_FLAG -> $MODE build mode"
  else
    MODE="lite"
    echo "[WARN] Could not detect self.use_sentiment in sentinel.py; defaulting to lite mode"
  fi
else
  MODE="$MODE_REQUEST"
  echo "[INFO] Build mode forced via flag: $MODE"
fi

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
  elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  elif command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.11)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  fi
fi

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python interpreter not found."
  echo "Activate your environment first, or set PYTHON_BIN=/path/to/python"
  exit 1
fi

echo "[INFO] Using Python: $PYTHON_BIN"
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  echo "[INFO] Active venv: ${VIRTUAL_ENV}"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
  echo "[INFO] Active conda env: ${CONDA_PREFIX}"
else
  echo "[INFO] No active virtual environment detected; using current Python as-is"
fi

if [[ "$INSTALL_DEPS" -eq 1 ]]; then
  "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
  "$PYTHON_BIN" -m pip install --upgrade pyinstaller

  if [[ "$MODE" == "full" ]]; then
    echo "[INFO] Installing full dependency set from requirements.txt"
    "$PYTHON_BIN" -m pip install -r "$ROOT/requirements.txt"
  else
    echo "[INFO] Installing lite dependency set (excluding AI/heavy packages)"
    TMP_REQ="$(mktemp)"
    trap 'rm -f "$TMP_REQ"' EXIT
    grep -Evi '^(torch|transformers|tensorflow|tensorboard|nvidia|accelerate|scipy)([<>=!~].*)?$' "$ROOT/requirements.txt" > "$TMP_REQ"
    "$PYTHON_BIN" -m pip install -r "$TMP_REQ"
  fi
else
  echo "[INFO] Skipping dependency installation (use --install-deps to enable)"
fi

REQUIRED_MODULES=(PyInstaller tkinter yfinance pandas numpy requests urllib3 matplotlib)
if [[ "$MODE" == "full" ]]; then
  REQUIRED_MODULES+=(torch transformers accelerate)
fi

if ! "$PYTHON_BIN" - "${REQUIRED_MODULES[@]}" <<'PY'
import importlib.util
import sys

mods = sys.argv[1:]
missing = [m for m in mods if importlib.util.find_spec(m) is None]

if missing:
    print("MISSING_MODULES=" + ",".join(missing))
    raise SystemExit(1)

print("[INFO] Required modules found")
PY
then
  echo "[ERROR] Required modules are missing in: $PYTHON_BIN"
  echo "[HINT] Activate the environment that already has your libs, or run with --install-deps"
  echo "[HINT] If tkinter is missing: brew install python-tk@3.11"
  exit 1
fi

PYI_ARGS=(
  --noconfirm
  --clean
  --windowed
  --name Sentinel
  --exclude-module tkinter.test
  --exclude-module notebook
  --collect-submodules matplotlib
)

if [[ "$MODE" == "lite" ]]; then
  PYI_ARGS+=(
    --exclude-module torch
    --exclude-module transformers
    --exclude-module tensorflow
    --exclude-module tensorboard
    --exclude-module nvidia
    --exclude-module scipy
    --exclude-module accelerate
  )
else
  echo "[INFO] AI mode enabled: including sentiment dependencies in build"
fi

if [[ "$BUNDLE_MODE" == "onefile" ]]; then
  PYI_ARGS+=(--onefile)
else
  PYI_ARGS+=(--onedir)
fi

if [[ -f "$ROOT/logo.icns" ]]; then
  PYI_ARGS+=(--icon "$ROOT/logo.icns")
else
  echo "[INFO] logo.icns not found; using default app icon"
fi

# Splash screens are intentionally omitted because PyInstaller splash is not macOS-compatible.
echo "[INFO] Running PyInstaller (${BUNDLE_MODE}, ${MODE})"
"$PYTHON_BIN" -m PyInstaller "${PYI_ARGS[@]}" "$APP_SCRIPT"

echo
if [[ -d "$ROOT/dist/Sentinel.app" ]]; then
  echo "[OK] Build complete: $ROOT/dist/Sentinel.app"
  echo "Run with: open \"$ROOT/dist/Sentinel.app\""
elif [[ -f "$ROOT/dist/Sentinel" ]]; then
  echo "[OK] Build complete: $ROOT/dist/Sentinel"
  echo "Run with: \"$ROOT/dist/Sentinel\""
else
  echo "[WARN] Build finished, but expected output was not found in dist/."
fi
