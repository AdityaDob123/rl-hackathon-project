# TradeDesk OpenEnv

A trading decision environment built for the **Scaler x Meta PyTorch OpenEnv Hackathon (Round 1)**.

An AI agent is given market data and must make trading decisions (buy, sell, hold, rebalance). Each decision is scored by a grader. No real money, no live data — everything is deterministic and runs locally.

---

## How to Run

Open **PowerShell** and navigate to the project folder (the directory that contains `requirements.txt`):

```powershell
cd path\to\tradedesk-openenv-complete
```

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
pytest -q
```

You should see `25 passed`.

### 4. Run the baseline agent

```powershell
python inference.py
```

This runs the deterministic agent on all 3 tasks, prints scores, and saves results to the `outputs\` folder.

### 5. Start the API server

```powershell
uvicorn app.api:app --host 0.0.0.0 --port 7860
```

Open [http://localhost:7860](http://localhost:7860) in your browser to verify it's running.

---

## Configuration (`.env` file)

All settings live in the `.env` file at the project root. The project works out of the box without changing anything.

| Variable | Default | What it does |
|---|---|---|
| `APP_HOST` | `0.0.0.0` | Server bind address |
| `APP_PORT` | `7860` | Server port |
| `DEFAULT_TASK` | `easy_signal_detection` | Which task loads by default |
| `OUTPUT_DIR` | `outputs` | Where results and charts are saved |
| `LOG_LEVEL` | `info` | Logging verbosity |

### Optional: LLM mode

If you want the agent to use an AI model instead of the hardcoded baseline, fill these in `.env`:

```env
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
API_KEY=sk-your-key-here
```

Then run `python inference.py` — it will automatically use the LLM.

---

## The 3 Tasks

| Task | Difficulty | What happens |
|---|---|---|
| **Signal Detection** | Easy | Look at AAPL's indicators, pick buy/sell/hold. (1 step) |
| **Position Management** | Medium | MSFT is losing money — cut the position, then decide what to do when it stabilizes. (2 steps) |
| **Portfolio Allocation** | Hard | Allocate $100k across 3 stocks, then rebalance when volatility spikes. (2 steps) |

---

## API Endpoints

Once the server is running, you can interact with it from a **second PowerShell window**:

| Method | Endpoint | What it does |
|---|---|---|
| `GET` | `/` | Shows app info |
| `GET` | `/health` | Health check |
| `GET` | `/tasks` | Lists available tasks |
| `POST` | `/reset` | Starts a task (send `{}` for default) |
| `POST` | `/step` | Submits a trading action |
| `GET` | `/state` | Shows current environment state |

### Example: Run a task via PowerShell

```powershell
# Reset to the easy task
Invoke-RestMethod -Method POST -Uri "http://localhost:7860/reset" `
  -ContentType "application/json" -Body '{}'

# Submit a buy action
Invoke-RestMethod -Method POST -Uri "http://localhost:7860/step" `
  -ContentType "application/json" -Body '{
    "action_type": "buy",
    "ticker": "AAPL",
    "order_fraction": 0.25,
    "confidence": 0.8,
    "rationale_tags": ["trend_up", "macd_bullish"]
  }'
```

---

## Docker

```powershell
docker build -t tradedesk-openenv .
docker run -p 7860:7860 tradedesk-openenv
```

---

## Output Files

After running `python inference.py`, you'll find these in the `outputs\` folder:

- `baseline_summary.json` — scores and step details for all tasks
- `task_scores.png` — bar chart of final scores
- `reward_breakdown.png` — reward component breakdown
- `confidence_vs_score.png` — confidence vs score scatter plot
- `reward_trajectory.png` — reward over steps

---

## Project Structure

```
├── .env                    # Configuration file
├── app\
│   ├── config.py           # Loads settings from .env
│   ├── api.py              # FastAPI server
│   ├── env.py              # Main environment (reset/step/state)
│   ├── models.py           # Data schemas (Observation, Action, Reward)
│   ├── tasks.py            # The 3 task scenarios
│   ├── graders.py          # Scoring logic
│   └── reward.py           # Reward computation
├── agent\
│   └── baseline.py         # Random baseline agent
├── inference.py            # Runs the agent on all tasks
├── visualization\
│   └── plots.py            # Chart generation
├── tests\
│   └── test_smoke.py       # Test suite
├── pytest.ini              # Pytest: only collects tests/ at repo root
├── Dockerfile              # Container build
├── openenv.yaml            # Environment metadata
└── requirements.txt        # Dependencies
```
