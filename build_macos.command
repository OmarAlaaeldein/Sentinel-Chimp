#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "[INFO] Launching macOS build from: $ROOT"
echo

./build_macos.sh "$@"
STATUS=$?

echo
if [[ $STATUS -eq 0 ]]; then
  echo "[OK] Build completed successfully."
else
  echo "[ERROR] Build failed with exit code: $STATUS"
fi

echo
read -r -p "Press Enter to close..." _
exit $STATUS
