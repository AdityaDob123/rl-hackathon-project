---
title: tradedesk-openenv
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# TradeDesk OpenEnv

A deterministic trading-desk environment built for the **Scaler x Meta PyTorch OpenEnv Hackathon**.

An AI agent receives market data snapshots and makes trading decisions (buy, sell, hold, reduce, rebalance). Each decision is scored by a grading function. No real money, no live data — everything is deterministic and reproducible.

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│                   inference.py                    │
│   LLMReasoningAgent → fallback → TradingAgent     │
└────────────┬─────────────────────┬───────────────┘
             │  Action             │  Observation
      ┌──────▼──────┐      ┌──────▼──────┐
      │   env.py    │◄────►│  graders.py │
      │ TradeDeskEnv│      │ easy/med/hrd│
      └──────┬──────┘      └─────────────┘
             │
      ┌──────▼──────┐
      │  tasks.py   │
      │ 3 scenarios │
      └─────────────┘
```

**Environment** (`app/env.py`): Manages the episode loop — `reset()` loads a task, `step()` applies an action, scores it, and returns the next observation.

**Agent** (`agent/llm_agent.py`): Contains `StrategyEngine` (rule-based, deterministic) and `LLMReasoningAgent` (LLM wrapper with deterministic fallback).

**Grader** (`app/graders.py`): Compares each action against gold-standard answers using difficulty-specific rubrics.

---

## Core Idea: Global Market Phase Simulation

The environment simulates three global trading sessions rotating across steps:

```
Step 0 → ASIAN
Step 1 → LONDON
Step 2 → NEW_YORK
Step 3 → ASIAN  (cycles)
```

The `TradingAgent` (fallback agent) uses a `GlobalSignalEngine` that tracks price returns across Asian and London sessions, then makes decisions during New York hours:

- **Bullish continuation**: Asian up + London up → buy
- **Pullback reversal**: Asian up + London down → buy on dip
- **Bearish**: Both sessions down → sell

This simulates how real institutional desks use overnight data to inform their trading day.

---

## Risk Management

The system implements multiple layers of risk control:

| Layer | Logic |
|---|---|
| **Drawdown monitoring** | `current_drawdown_pct` from `PortfolioState` triggers forced reductions above 15% |
| **Loss streak detection** | `RiskManager` tracks consecutive losses and escalates from reduce → switch → force_reduce |
| **Concentration limits** | Hard task enforces `max_single_name_weight` (50%) and `min_cash_reserve` (10%) |
| **Order sizing** | `StrategyEngine` sizes orders based on signal strength, not fixed fractions |
| **Confidence gating** | Low-confidence signals below 0.3 default to hold |

The agent uses real indicator fields from the environment:

- `rsi`, `macd`, `macd_signal` — momentum signals
- `ema_20_gap_pct` — trend distance
- `trend_label`, `volatility_label` — categorical context
- `current_drawdown_pct`, `exposure_pct` — portfolio risk metrics

---

## The 3 Tasks

| Task | Difficulty | Steps | What happens |
|---|---|---|---|
| **Signal Detection** | Easy | 1 | Read AAPL indicators, pick buy/sell/hold |
| **Position Management** | Medium | 2 | MSFT is losing money — cut the position, then wait for stabilization |
| **Portfolio Allocation** | Hard | 2 | Allocate $100k across 3 stocks, then rebalance after a volatility spike |

---

## How to Run

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run the tests

```powershell
pytest -v
```

All 25 tests should pass.

### 4. Run the inference pipeline

```powershell
python inference.py
```

Runs the agent on all 3 tasks, prints step-by-step scores, and saves results to `outputs/`.

### 5. Start the API server

```powershell
uvicorn app.api:app --host 0.0.0.0 --port 7860
```

---

## Example Output

```
[START] task=easy_signal_detection
[STEP] step=0 action=buy reward=0.7755
[END] task=easy_signal_detection score=0.9000 steps=1

[START] task=medium_position_management
[STEP] step=0 action=sell reward=0.8755
[STEP] step=1 action=hold reward=0.6530
[END] task=medium_position_management score=0.7000 steps=2

[START] task=hard_portfolio_allocation
[STEP] step=0 action=rebalance reward=0.7805
[STEP] step=1 action=rebalance reward=0.7955
[END] task=hard_portfolio_allocation score=0.9800 steps=2
```

---

## API Endpoints

| Method | Endpoint | What it does |
|---|---|---|
| `GET` | `/` | App info |
| `GET` | `/health` | Health check |
| `GET` | `/tasks` | List available tasks |
| `POST` | `/reset` | Start a task (`{}` for default) |
| `POST` | `/step` | Submit a trading action |
| `GET` | `/state` | Current environment state |

---

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `DEFAULT_TASK` | `easy_signal_detection` | Default task on reset |
| `OUTPUT_DIR` | `outputs` | Where results are saved |
| `LOG_LEVEL` | `info` | Logging verbosity |
| `API_BASE_URL` | `https://api.openai.com/v1` | LLM endpoint (optional) |
| `MODEL_NAME` | `gpt-4o-mini` | LLM model (optional) |
| `HF_TOKEN` | — | Auth token for LLM (optional) |

---

## Docker

```powershell
docker build -t tradedesk-openenv .
docker run -p 7860:7860 tradedesk-openenv
```

---

## Project Structure

```
├── app/
│   ├── config.py           # Settings from .env
│   ├── api.py              # FastAPI server
│   ├── env.py              # Environment (reset/step/state)
│   ├── models.py           # Pydantic schemas
│   ├── tasks.py            # 3 task scenarios with gold answers
│   ├── graders.py          # Difficulty-specific scoring
│   ├── reward.py           # Dense reward computation
│   ├── agent.py            # TradingAgent (fallback with GlobalSignalEngine)
│   ├── risk_manager.py     # Loss streak and drawdown tracking
│   ├── state_manager.py    # Episode state tracking
│   └── logger.py           # Structured logging
├── agent/
│   ├── llm_agent.py        # StrategyEngine + LLMReasoningAgent
│   └── baseline.py         # Random baseline agent
├── inference.py            # Main pipeline (runs all tasks)
├── visualization/
│   └── plots.py            # Chart generation
├── tests/
│   └── test_smoke.py       # 25 test cases
├── Dockerfile
├── openenv.yaml
├── requirements.txt
└── pytest.ini
```
