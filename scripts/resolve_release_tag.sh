#!/usr/bin/env bash
set -euo pipefail
if [[ "${GITHUB_EVENT_NAME:-}" == "workflow_dispatch" ]]; then
  tag="${DISPATCH_VERSION:-}"
else
  tag="${GITHUB_REF_NAME:-}"
fi
if [[ ! "$tag" =~ ^v?[0-9]+\.[0-9]+(\.[0-9]+)?(-[0-9A-Za-z][0-9A-Za-z.-]*)?(\+[0-9A-Za-z][0-9A-Za-z.-]*)?$ ]]; then
  echo "Invalid release tag: expected a numeric version, optionally prefixed with v." >&2
  exit 1
fi
printf 'tag=%s\n' "$tag" >> "$GITHUB_OUTPUT"
printf 'Using tag: %s\n' "$tag"
