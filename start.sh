#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Create and activate virtualenv
if [ ! -d ".venv" ]; then
  python -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate

pip install --upgrade pip
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
fi

# Start backend
nohup .venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 > backend.log 2>&1 &
sleep 2

# Start frontend (Vite)
if [ -d frontend ]; then
  cd frontend
  if [ ! -d node_modules ]; then
    if command -v npm >/dev/null 2>&1; then
      npm ci
    else
      echo "npm not found; please install Node.js and npm"
      exit 1
    fi
  fi
  nohup npm run dev -- --host 0.0.0.0 > ../frontend.log 2>&1 &
  cd "$ROOT"
fi

# Open browser if available
if command -v xdg-open >/dev/null 2>&1; then
  xdg-open http://127.0.0.1:5173 || true
elif command -v open >/dev/null 2>&1; then
  open http://127.0.0.1:5173 || true
fi

echo "Launched backend and frontend. Backend health: http://127.0.0.1:8000/health"
