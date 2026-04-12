from __future__ import annotations

import json
import random
from typing import Dict, List, Optional, Tuple

from app import config
from app.models import Action, Observation

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

SYSTEM_PROMPT = """You are a disciplined trading-desk agent inside a deterministic OpenEnv task.
Return JSON only with keys:
action_type, ticker, order_fraction, target_allocations, confidence, rationale_tags, reasoning, indicator_summary.
Pick one valid action from allowed_actions and keep confidence in [0,1]."""

EXAMPLE_PROMPT = {
    "task_id": "easy_signal_detection",
    "difficulty": "easy",
    "step_index": 0,
    "allowed_actions": ["buy", "sell", "hold"],
    "market": [
        {
            "ticker": "AAPL",
            "trend_label": "up",
            "macd": 1.21,
            "macd_signal": 0.72,
            "rsi": 63.0,
            "volume_zscore": 1.2,
        }
    ],
    "expected_response_shape": {
        "action_type": "buy",
        "ticker": "AAPL",
        "order_fraction": 0.25,
        "confidence": 0.82,
        "rationale_tags": ["trend_up", "macd_bullish", "volume_confirmation"],
        "reasoning": "Trend and momentum are both bullish with confirming volume.",
        "indicator_summary": "Trend=up, MACD>signal, RSI=63, volume z-score=1.2",
    },
}


class StrategyEngine:
    """Rule-based fallback strategy that stays deterministic."""

    def _signal_score(self, stock) -> float:
        signals: List[float] = []
        trend_map = {"up": 1.0, "down": -1.0, "sideways": 0.0}
        signals.append(trend_map.get(stock.trend_label, 0.0) * 1.5)

        macd_diff = stock.macd - stock.macd_signal
        signals.append(max(-1.0, min(1.0, macd_diff)))

        if stock.rsi >= 70:
            signals.append(-0.5)
        elif stock.rsi <= 30:
            signals.append(0.5)
        else:
            signals.append((stock.rsi - 50) / 50)

        stoch_signal = (stock.stochastic_k - 50) / 50
        signals.append(max(-1.0, min(1.0, stoch_signal * 0.6)))

        bb_signal = (stock.bb_pos - 0.5) * 2
        signals.append(bb_signal * 0.5)

        vol_signal = max(-1.0, min(1.0, stock.volume_zscore * 0.4))
        signals.append(vol_signal)
        signals.append(max(-1.0, min(1.0, stock.obv_slope * 0.5)))
        return max(-1.0, min(1.0, sum(signals) / len(signals)))

    def _build_tags(self, stock) -> List[str]:
        tags: List[str] = []
        if stock.trend_label == "up":
            tags.append("trend_up")
        elif stock.trend_label == "down":
            tags.append("downtrend")

        if stock.macd > stock.macd_signal:
            tags.append("macd_bullish")
        elif stock.macd < stock.macd_signal:
            tags.append("macd_bearish")

        if stock.volume_zscore > 0.5:
            tags.append("volume_confirmation")
        if stock.rsi > 70:
            tags.append("overbought")
        elif stock.rsi < 30:
            tags.append("oversold")
        return tags

    def _calibrate_confidence(self, signal: float) -> float:
        abs_signal = abs(signal)
        if abs_signal > 0.6:
            return round(0.75 + abs_signal * 0.2, 2)
        if abs_signal > 0.3:
            return round(0.55 + abs_signal * 0.3, 2)
        return round(0.45 + abs_signal * 0.2, 2)

    def _size_order(self, signal: float, action_type: str) -> float:
        if action_type == "sell":
            return 1.0 if abs(signal) > 0.3 else 0.90
        if action_type == "reduce":
            return round(max(0.25, min(0.75, abs(signal))), 2)
        return round(max(0.10, min(0.40, abs(signal) * 0.5)), 2)

    def _indicator_summary(self, observation: Observation) -> str:
        if not observation.market:
            return "No market data."
        stock = observation.market[0]
        return (
            f"{stock.ticker}: trend={stock.trend_label}, macd_diff={stock.macd - stock.macd_signal:.2f}, "
            f"rsi={stock.rsi:.1f}, vol_z={stock.volume_zscore:.2f}, bb_pos={stock.bb_pos:.2f}"
        )

    def _reasoning_text(self, action: Action, observation: Observation) -> str:
        return (
            f"Selected {action.action_type} because indicator alignment supports this move "
            f"under allowed actions {observation.allowed_actions}."
        )

    def act_with_reasoning(self, observation: Observation) -> Tuple[Action, Dict[str, str]]:
        allowed = observation.allowed_actions
        if "rebalance" in allowed and len(observation.market) > 1:
            action = self._rebalance(observation)
        else:
            stock = observation.market[0]
            signal = self._signal_score(stock)
            tags = self._build_tags(stock)
            confidence = self._calibrate_confidence(signal)
            has_position = stock.ticker in observation.positions
            position = observation.positions.get(stock.ticker)
            is_losing = position and position.unrealized_pnl_pct < -3.0

            if is_losing and signal < 0 and "sell" in allowed:
                tags.extend(["stop_loss", "capital_preservation"])
                action = Action(
                    action_type="sell",
                    ticker=stock.ticker,
                    order_fraction=self._size_order(signal, "sell"),
                    confidence=max(confidence, 0.80),
                    rationale_tags=tags,
                )
            elif signal > 0.15 and "buy" in allowed:
                action = Action(
                    action_type="buy",
                    ticker=stock.ticker,
                    order_fraction=self._size_order(signal, "buy"),
                    confidence=confidence,
                    rationale_tags=tags,
                )
            elif signal < -0.15 and "sell" in allowed and has_position:
                action = Action(
                    action_type="sell",
                    ticker=stock.ticker,
                    order_fraction=self._size_order(signal, "sell"),
                    confidence=confidence,
                    rationale_tags=tags,
                )
            elif "hold" in allowed:
                action = Action(
                    action_type="hold",
                    ticker=stock.ticker,
                    order_fraction=0.0,
                    confidence=confidence,
                    rationale_tags=tags + ["capital_preservation"],
                )
            else:
                action = Action(
                    action_type=allowed[0],
                    ticker=stock.ticker,
                    order_fraction=0.25 if allowed[0] != "hold" else 0.0,
                    confidence=0.5,
                    rationale_tags=["fallback"],
                )

        metadata = {
            "reasoning": self._reasoning_text(action, observation),
            "indicator_summary": self._indicator_summary(observation),
        }
        return action, metadata

    def act(self, observation: Observation) -> Action:
        action, _ = self.act_with_reasoning(observation)
        return action

    def _rebalance(self, observation: Observation) -> Action:
        scores: Dict[str, float] = {}
        for stock in observation.market:
            raw = self._signal_score(stock)
            scores[stock.ticker] = max(0.05, (raw + 1.0) / 2.0)

        total_score = sum(scores.values())
        investable = 0.88
        allocations: Dict[str, float] = {}
        for ticker, score in scores.items():
            allocations[ticker] = round((score / total_score) * investable, 2)

        alloc_total = sum(allocations.values())
        if alloc_total > 0.9:
            scale = 0.9 / alloc_total
            allocations = {k: round(v * scale, 2) for k, v in allocations.items()}

        confidence = self._calibrate_confidence(max(scores.values()) / total_score if total_score > 0 else 0.5)
        tags = ["risk_balanced", "cash_buffer"]
        if observation.step_index > 0:
            tags.insert(0, "trim_winner")
        return Action(
            action_type="rebalance",
            confidence=confidence,
            target_allocations=allocations,
            rationale_tags=tags,
        )


class LLMReasoningAgent:
    """LLM wrapper around StrategyEngine with deterministic fallback."""

    def __init__(self, seed: int = 7) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.strategy = StrategyEngine()
        self.api_base_url = config.API_BASE_URL
        self.model_name = config.MODEL_NAME
        self.hf_token = config.HF_TOKEN
        self.client = None
        if self.hf_token and OpenAI is not None:
            self.client = OpenAI(base_url=self.api_base_url, api_key=self.hf_token)

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def build_user_prompt(self, observation: Observation, history: List[Dict]) -> str:
        payload = {
            "task_id": observation.task_id,
            "difficulty": observation.difficulty,
            "step_index": observation.step_index,
            "max_steps": observation.max_steps,
            "notes": observation.notes,
            "allowed_actions": observation.allowed_actions,
            "market": [item.model_dump() for item in observation.market],
            "portfolio": observation.portfolio.model_dump(),
            "positions": {k: v.model_dump() for k, v in observation.positions.items()},
            "history": history[-3:],
        }
        return json.dumps(payload, ensure_ascii=True)

    def _safe_parse(self, response_text: str, observation: Observation) -> Tuple[Action, Dict[str, str]]:
        data = json.loads(response_text)
        action = Action(**data)
        metadata = {
            "reasoning": str(data.get("reasoning", "")).strip() or "Model chose a valid action from allowed actions.",
            "indicator_summary": str(data.get("indicator_summary", "")).strip()
            or self.strategy._indicator_summary(observation),
        }
        return action, metadata

    def act(self, observation: Observation, history: List[Dict]) -> Tuple[Action, Dict[str, str]]:
        if not self.enabled:
            return self.strategy.act_with_reasoning(observation)

        assert self.client is not None
        prompt = self.build_user_prompt(observation, history)
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0,
                seed=self.seed,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            response_text = completion.choices[0].message.content or "{}"
            return self._safe_parse(response_text, observation)
        except Exception:
            # Keep deterministic behavior and grader compatibility on any LLM failure.
            return self.strategy.act_with_reasoning(observation)
