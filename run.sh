#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

pip install -r requirements.txt

if [ ! -f models/best_model.pkl ] || [ ! -f models/std_scaler.pkl ] || [ ! -f models/model_features.pkl ]; then
  python model_training.py
fi

PORT=5000
if command -v lsof >/dev/null 2>&1 && lsof -iTCP:"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
  PORT=5001
fi

echo "Starting app on http://localhost:${PORT}"
exec env PORT="$PORT" python app.py