#!/usr/bin/env bash
set -euo pipefail

python -m pytest -q
python inference.py > /tmp/tradedesk_inference.json
python - <<'PY'
from app.env import TradeDeskOpenEnv

env = TradeDeskOpenEnv()
obs = env.reset()
assert obs.task_id == "easy_signal_detection"
print("reset ok")
PY

echo "Local validation passed."
