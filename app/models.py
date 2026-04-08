from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


Difficulty = Literal["easy", "medium", "hard"]
ActionType = Literal["buy", "sell", "hold", "reduce", "rebalance"]


class StockFeatureSnapshot(BaseModel):
    ticker: str
    close: float = Field(gt=0)
    ema_20_gap_pct: float
    macd: float
    macd_signal: float
    rsi: float = Field(ge=0, le=100)
    stochastic_k: float = Field(ge=0, le=100)
    stochastic_d: float = Field(ge=0, le=100)
    bb_pos: float = Field(ge=0, le=1, description="Position within Bollinger band range")
    volume_zscore: float
    obv_slope: float
    trend_label: Literal["up", "down", "sideways"]
    volatility_label: Literal["low", "medium", "high"]


class PositionState(BaseModel):
    shares_held: int = Field(ge=0)
    entry_price: float = Field(ge=0)
    market_value: float = Field(ge=0)
    unrealized_pnl_pct: float


class PortfolioState(BaseModel):
    cash: float = Field(ge=0)
    total_value: float = Field(gt=0)
    max_drawdown_pct: float = Field(ge=0, le=100)
    current_drawdown_pct: float = Field(ge=0, le=100)
    exposure_pct: float = Field(ge=0, le=100)
    cooldown_remaining: int = Field(ge=0)

class Observation(BaseModel):
    task_id: str
    difficulty: Difficulty
    step_index: int = Field(ge=0)
    max_steps: int = Field(gt=0)
    market: List[StockFeatureSnapshot]
    portfolio: PortfolioState
    positions: Dict[str, PositionState]
    allowed_actions: List[ActionType]
    notes: Optional[str] = None
    market_phase: Optional[str] = None

class Action(BaseModel):
    action_type: ActionType
    ticker: Optional[str] = None
    order_fraction: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    target_allocations: Optional[Dict[str, float]] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale_tags: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_shape(self) -> "Action":
        if self.action_type in {"buy", "sell", "reduce"} and not self.ticker:
            raise ValueError("ticker is required for buy/sell/reduce")
        if self.action_type in {"buy", "sell", "reduce"} and self.order_fraction is None:
            raise ValueError("order_fraction is required for buy/sell/reduce")
        if self.action_type == "rebalance":
            if not self.target_allocations:
                raise ValueError("target_allocations are required for rebalance")
            total = sum(self.target_allocations.values())
            if total > 1.000001:
                raise ValueError("target_allocations must sum to <= 1.0")
            if any(v < 0 for v in self.target_allocations.values()):
                raise ValueError("target_allocations cannot be negative")
        return self


class Reward(BaseModel):
    value: float = Field(ge=-1.0, le=1.0)
    components: Dict[str, float]
    message: str


class StateView(BaseModel):
    task_id: Optional[str] = None
    difficulty: Optional[Difficulty] = None
    done: bool = False
    step_index: int = 0
    max_steps: int = 0
    last_action: Optional[Action] = None
    cumulative_reward: float = 0.0
    final_score: Optional[float] = None
    observation: Optional[Observation] = None
