from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

from app.graders import grade_action
from app.logger import get_logger
from app.models import Action, Observation, PortfolioState, PositionState, Reward, StockFeatureSnapshot
from app.reward import build_reward
from app.state_manager import StateManager
from app.tasks import get_default_task_id, get_task, list_tasks

log = get_logger("env")


def get_market_phase(step: int) -> str:
    return ["ASIAN", "LONDON", "NEW_YORK"][step % 3]


class TradeDeskOpenEnv:
    def __init__(self) -> None:
        self.state_manager = StateManager()
        self.current_task: Optional[Dict] = None
        self.current_step_index: int = 0

    def available_tasks(self) -> List[Dict]:
        return [
            {
                "task_id": t["task_id"],
                "difficulty": t["difficulty"],
                "notes": t["notes"],
                "max_steps": t["max_steps"],
            }
            for t in list_tasks()
        ]

    def _observation_from_step(self, task: Dict, step_index: int) -> Observation:
        step = task["scenario_steps"][step_index]

        return Observation(
            task_id=task["task_id"],
            difficulty=task["difficulty"],
            step_index=step_index,
            max_steps=task["max_steps"],
            market=[StockFeatureSnapshot(**m) for m in step["market"]],
            portfolio=PortfolioState(**step["portfolio"]),
            positions={k: PositionState(**v) for k, v in step.get("positions", {}).items()},
            allowed_actions=step["allowed_actions"],
            notes=step.get("notes") or task.get("notes"),
            market_phase=get_market_phase(step_index),
        )

    def _merge_action_effects(self, base_step: Dict, action: Action) -> Dict:
        step = deepcopy(base_step)
        prices = {item["ticker"]: item["close"] for item in step["market"]}
        portfolio = step["portfolio"]
        positions = step.setdefault("positions", {})

        def update_portfolio_totals() -> None:
            positions_value = sum(pos["market_value"] for pos in positions.values())
            portfolio["total_value"] = round(portfolio["cash"] + positions_value, 2)
            if portfolio["total_value"] > 0:
                portfolio["exposure_pct"] = round((positions_value / portfolio["total_value"]) * 100, 2)

        if action.action_type in {"buy", "sell", "reduce"} and action.ticker and action.order_fraction is not None:
            if action.ticker not in prices:
                log.warning("Ticker %s not found in snapshot.", action.ticker)
            else:
                price = prices[action.ticker]
                existing = positions.get(
                    action.ticker,
                    {
                        "shares_held": 0,
                        "entry_price": price,
                        "market_value": 0.0,
                        "unrealized_pnl_pct": 0.0,
                    },
                )

                if action.action_type == "buy":
                    deploy_cash = portfolio["cash"] * action.order_fraction
                    shares = int(deploy_cash / price)
                    if shares > 0:
                        existing_cost = existing["shares_held"] * existing["entry_price"]
                        new_cost = shares * price
                        existing["shares_held"] += shares
                        existing["entry_price"] = round((existing_cost + new_cost) / existing["shares_held"], 2)
                        portfolio["cash"] = round(max(0.0, portfolio["cash"] - shares * price), 2)

                elif action.action_type in {"sell", "reduce"}:
                    owned = existing["shares_held"]
                    shares_to_sell = owned if action.action_type == "sell" else int(owned * action.order_fraction)
                    shares_to_sell = min(owned, shares_to_sell)
                    if shares_to_sell > 0:
                        portfolio["cash"] += shares_to_sell * price
                        existing["shares_held"] -= shares_to_sell

                if existing["shares_held"] <= 0:
                    positions.pop(action.ticker, None)
                else:
                    existing["market_value"] = round(existing["shares_held"] * price, 2)
                    existing["unrealized_pnl_pct"] = round(
                        ((price - existing["entry_price"]) / existing["entry_price"]) * 100, 2
                    )
                    positions[action.ticker] = existing

                update_portfolio_totals()

        if action.action_type == "rebalance" and action.target_allocations:
            total_value = portfolio["total_value"]
            new_positions = {}

            for ticker, weight in action.target_allocations.items():
                if ticker not in prices:
                    continue

                price = prices[ticker]
                dollars = total_value * weight
                shares = int(dollars / price)

                if shares <= 0:
                    continue

                new_positions[ticker] = {
                    "shares_held": shares,
                    "entry_price": price,
                    "market_value": round(shares * price, 2),
                    "unrealized_pnl_pct": 0.0,
                }

            positions.clear()
            positions.update(new_positions)

            deployed = sum(pos["market_value"] for pos in positions.values())
            portfolio["cash"] = round(max(0.0, total_value - deployed), 2)

            update_portfolio_totals()

        step["portfolio"] = portfolio
        step["positions"] = positions

        return step

    def reset(self, task_id: Optional[str] = None) -> Observation:
        chosen_task = task_id or get_default_task_id()
        task = deepcopy(get_task(chosen_task))

        self.current_task = task
        self.current_step_index = 0

        observation = self._observation_from_step(task, step_index=0)
        self.state_manager.load_task(observation)

        log.info("RESET -> task=%s difficulty=%s", task["task_id"], task["difficulty"])

        return observation

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self.current_task is None:
            raise RuntimeError("Environment not initialized. Call reset(task_id) first.")
        if self.state_manager.done:
            raise RuntimeError("Episode already finished.")

        current_step = self.current_task["scenario_steps"][self.current_step_index]
        valid = action.action_type in current_step["allowed_actions"]

        final_score = grade_action(action, self.current_task, self.current_step_index) if valid else 0.0
        reward = build_reward(action, self.current_task, final_score, valid=valid)

        next_index = min(self.current_step_index + 1, self.current_task["max_steps"] - 1)

        next_step = deepcopy(self.current_task["scenario_steps"][next_index])
        next_step = self._merge_action_effects(next_step, action)

        self.current_task["scenario_steps"][next_index] = next_step

        next_obs = self._observation_from_step(self.current_task, step_index=next_index)

        done = self.current_step_index >= self.current_task["max_steps"] - 1
        self.current_step_index = next_index

        self.state_manager.apply_step(action, reward.value, final_score, next_obs, done=done)

        info = {
            "task_id": self.current_task["task_id"],
            "difficulty": self.current_task["difficulty"],
            "final_score": final_score,
            "market_phase": get_market_phase(self.current_step_index),
        }

        return next_obs, reward, done, info

    def state(self):
        return self.state_manager.state_view()