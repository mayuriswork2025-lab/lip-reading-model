#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-../LipRead-Studio-v1.zip}"
cd "$ROOT"

if ! command -v zip >/dev/null 2>&1; then
  echo "zip utility is required. Install it (apt install zip / brew install zip)."
  exit 1
fi

zip -r "$OUT" . -x "evaluation/models/*.h5" "frontend/node_modules/*" ".venv/*" "*.zip"

echo "Created $OUT"
