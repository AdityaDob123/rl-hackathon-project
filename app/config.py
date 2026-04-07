from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (no-op if file is absent)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)


# ── Server ──────────────────────────────────────
APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT: int = int(os.getenv("APP_PORT", "7860"))
APP_TITLE: str = os.getenv("APP_TITLE", "TradeDesk OpenEnv")
APP_VERSION: str = os.getenv("APP_VERSION", "1.2.0")

# ── Environment ─────────────────────────────────
DEFAULT_TASK: str = os.getenv("DEFAULT_TASK", "easy_signal_detection")
OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "outputs")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")

# ── LLM Inference (optional) ────────────────────
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: str | None = os.getenv("HF_TOKEN")
