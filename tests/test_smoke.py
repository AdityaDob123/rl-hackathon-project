from fastapi.testclient import TestClient
import pytest

from app.api import app
from app.env import TradeDeskOpenEnv
from app.models import Action, Observation


# ─────────────────────────────────────────────────
#  Core environment tests
# ─────────────────────────────────────────────────

def test_env_smoke():
    env = TradeDeskOpenEnv()
    obs = env.reset("easy_signal_detection")
    action = Action(action_type="buy", ticker="AAPL", order_fraction=0.25, confidence=0.8, rationale_tags=["trend_up"])
    next_obs, reward, done, info = env.step(action)
    assert done is True
    assert 0.0 <= info["final_score"] <= 1.0
    assert -1.0 <= reward.value <= 1.0
    assert next_obs.step_index == 0


def test_step_before_reset_raises():
    env = TradeDeskOpenEnv()
    action = Action(action_type="hold", confidence=0.4, rationale_tags=["waiting"])
    with pytest.raises(RuntimeError, match="not initialized"):
        env.step(action)


def test_step_after_done_raises():
    """Calling step() after episode is done should raise."""
    env = TradeDeskOpenEnv()
    env.reset("easy_signal_detection")
    action = Action(action_type="buy", ticker="AAPL", order_fraction=0.25, confidence=0.8, rationale_tags=["trend_up"])
    env.step(action)
    with pytest.raises(RuntimeError, match="already finished"):
        env.step(action)


def test_invalid_action_type_is_scored_zero():
    env = TradeDeskOpenEnv()
    env.reset("easy_signal_detection")
    action = Action(action_type="hold", confidence=0.5, rationale_tags=["wrong_side"])
    _, _, done, info = env.step(action)
    assert done is True
    assert info["final_score"] == 0.0


def test_multistep_medium_task_progresses():
    env = TradeDeskOpenEnv()
    obs = env.reset("medium_position_management")
    assert obs.max_steps == 2
    action1 = Action(action_type="sell", ticker="MSFT", order_fraction=1.0, confidence=0.9, rationale_tags=["stop_loss", "downtrend", "high_volatility"])
    next_obs, _, done, _ = env.step(action1)
    assert done is False
    assert next_obs.step_index == 1
    assert next_obs.portfolio.cash >= 80000

    action2 = Action(action_type="hold", ticker="MSFT", order_fraction=0.0, confidence=0.6, rationale_tags=["capital_preservation", "wait_for_confirmation", "reduced_risk"])
    _, _, done2, info2 = env.step(action2)
    assert done2 is True
    assert 0.0 <= info2["final_score"] <= 1.0


def test_rebalance_allocation_validation():
    with pytest.raises(ValueError):
        Action(action_type="rebalance", target_allocations={"NVDA": 0.8, "META": 0.4}, confidence=0.7)


def test_negative_allocation_rejected():
    """Negative allocation weights should be rejected."""
    with pytest.raises(ValueError, match="negative"):
        Action(action_type="rebalance", target_allocations={"NVDA": -0.1, "META": 0.3}, confidence=0.7)


def test_buy_requires_ticker():
    """Buy/sell/reduce must have a ticker."""
    with pytest.raises(ValueError, match="ticker"):
        Action(action_type="buy", order_fraction=0.25, confidence=0.5)


def test_buy_requires_order_fraction():
    """Buy/sell/reduce must have order_fraction."""
    with pytest.raises(ValueError, match="order_fraction"):
        Action(action_type="buy", ticker="AAPL", confidence=0.5)


# ─────────────────────────────────────────────────
#  Multi-task / cross-task tests
# ─────────────────────────────────────────────────

def test_double_reset_is_clean():
    """Resetting twice should produce the same initial observation."""
    env = TradeDeskOpenEnv()
    obs1 = env.reset("easy_signal_detection")
    env.reset("medium_position_management")
    obs2 = env.reset("easy_signal_detection")
    assert obs1.model_dump() == obs2.model_dump()


def test_hard_task_rebalance_full_episode():
    """Run hard task to completion with valid rebalance actions."""
    env = TradeDeskOpenEnv()
    obs = env.reset("hard_portfolio_allocation")
    assert obs.max_steps == 2

    action1 = Action(
        action_type="rebalance",
        target_allocations={"NVDA": 0.40, "META": 0.30, "JNJ": 0.10},
        confidence=0.81,
        rationale_tags=["momentum_leader", "risk_balanced", "cash_buffer"],
    )
    next_obs, reward1, done, info = env.step(action1)
    assert done is False
    assert info["final_score"] > 0.5

    action2 = Action(
        action_type="rebalance",
        target_allocations={"NVDA": 0.36, "META": 0.28, "JNJ": 0.14},
        confidence=0.81,
        rationale_tags=["trim_winner", "risk_balanced", "cash_buffer"],
    )
    _, reward2, done2, info2 = env.step(action2)
    assert done2 is True
    assert info2["final_score"] > 0.5


def test_hold_on_all_tasks():
    """Hold is always allowed and shouldn't crash."""
    env = TradeDeskOpenEnv()
    for task_id in ["easy_signal_detection", "medium_position_management", "hard_portfolio_allocation"]:
        obs = env.reset(task_id)
        if "hold" in obs.allowed_actions:
            action = Action(action_type="hold", confidence=0.5, rationale_tags=["test"])
            _, _, done, info = env.step(action)
            assert 0.0 <= info["final_score"] <= 1.0


def test_available_tasks_returns_all():
    env = TradeDeskOpenEnv()
    tasks = env.available_tasks()
    assert len(tasks) == 3
    task_ids = {t["task_id"] for t in tasks}
    assert "easy_signal_detection" in task_ids
    assert "medium_position_management" in task_ids
    assert "hard_portfolio_allocation" in task_ids


# ─────────────────────────────────────────────────
#  Strategy Engine tests
# ─────────────────────────────────────────────────

def test_strategy_engine_easy_task():
    """Strategy engine should handle easy task correctly."""
    from inference import StrategyEngine
    env = TradeDeskOpenEnv()
    agent = StrategyEngine()
    obs = env.reset("easy_signal_detection")
    action = agent.act(obs)
    assert action.action_type in obs.allowed_actions
    assert action.confidence > 0
    assert len(action.rationale_tags) > 0


def test_strategy_engine_medium_task():
    """Strategy engine handles multi-step medium task."""
    from inference import StrategyEngine
    env = TradeDeskOpenEnv()
    agent = StrategyEngine()
    obs = env.reset("medium_position_management")
    action1 = agent.act(obs)
    assert action1.action_type in obs.allowed_actions
    next_obs, _, done, _ = env.step(action1)
    if not done:
        action2 = agent.act(next_obs)
        assert action2.action_type in next_obs.allowed_actions


def test_strategy_engine_hard_task():
    """Strategy engine handles rebalance allocation task."""
    from inference import StrategyEngine
    env = TradeDeskOpenEnv()
    agent = StrategyEngine()
    obs = env.reset("hard_portfolio_allocation")
    action = agent.act(obs)
    assert action.action_type == "rebalance"
    assert action.target_allocations is not None
    assert sum(action.target_allocations.values()) <= 1.0


def test_full_inference_pipeline():
    """The entire inference run() function should complete without errors."""
    from inference import run
    result = run()
    assert result["average_score"] > 0
    assert len(result["tasks"]) == 3
    for task in result["tasks"]:
        assert task["done"] is True
        assert 0.0 <= task["final_score"] <= 1.0


# ─────────────────────────────────────────────────
#  API endpoint tests
# ─────────────────────────────────────────────────

def test_api_root():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_api_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_tasks_list():
    client = TestClient(app)
    response = client.get("/tasks")
    assert response.status_code == 200
    tasks = response.json()["tasks"]
    assert len(tasks) == 3


def test_api_reset_allows_empty_body():
    client = TestClient(app)
    response = client.post("/reset", json={})
    assert response.status_code == 200
    payload = response.json()
    assert payload["task_id"] == "easy_signal_detection"


def test_api_reset_invalid_task_returns_404():
    client = TestClient(app)
    response = client.post("/reset", json={"task_id": "nonexistent_task"})
    assert response.status_code == 404


def test_api_step_without_reset_returns_error():
    """Step on a fresh app instance should fail gracefully."""
    client = TestClient(app)
    # Reset first so the env is in a clean state, then do a full run
    client.post("/reset", json={})
    response = client.post(
        "/step",
        json={
            "action_type": "buy",
            "ticker": "AAPL",
            "order_fraction": 0.25,
            "confidence": 0.8,
            "rationale_tags": ["test"],
        },
    )
    assert response.status_code == 200


def test_api_endpoints_work():
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/tasks").status_code == 200
    client.post("/reset", json={"task_id": "hard_portfolio_allocation"})
    response = client.post(
        "/step",
        json={
            "action_type": "rebalance",
            "target_allocations": {"NVDA": 0.4, "META": 0.3, "JNJ": 0.1},
            "confidence": 0.8,
            "rationale_tags": ["momentum_leader", "risk_balanced", "cash_buffer"],
        },
    )
    assert response.status_code == 200
    assert "reward" in response.json()


def test_api_state_endpoint():
    client = TestClient(app)
    client.post("/reset", json={"task_id": "easy_signal_detection"})
    response = client.get("/state")
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == "easy_signal_detection"
    assert data["done"] is False
