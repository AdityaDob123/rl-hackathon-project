#!/usr/bin/env bash
set -e
pip install -r requirements.txt
python inference.py
uvicorn app.api:app --host 0.0.0.0 --port 7860
