from __future__ import annotations

import random

from app.models import Action, Observation


class RandomAgent:
    def __init__(self, seed: int = 7) -> None:
        self.rng = random.Random(seed)

    def act(self, observation: Observation) -> Action:
        action_type = self.rng.choice(observation.allowed_actions)
        ticker = observation.market[0].ticker if observation.market else None

        if action_type == "rebalance":
            tickers = [item.ticker for item in observation.market]
            allocations = {}
            remaining = 0.85
            for symbol in tickers[:-1]:
                weight = round(self.rng.uniform(0.1, min(0.4, remaining)), 2)
                allocations[symbol] = weight
                remaining -= weight
            if tickers:
                allocations[tickers[-1]] = round(max(0.05, remaining), 2)
            if sum(allocations.values()) > 0.9:
                scale = 0.9 / sum(allocations.values())
                allocations = {k: round(v * scale, 2) for k, v in allocations.items()}
            return Action(
                action_type="rebalance",
                target_allocations=allocations,
                confidence=0.55,
                rationale_tags=["random_baseline"],
            )

        if action_type in {"buy", "sell", "reduce"}:
            return Action(
                action_type=action_type,
                ticker=ticker,
                order_fraction=0.25 if action_type != "sell" else 1.0,
                confidence=0.5,
                rationale_tags=["random_baseline"],
            )

        return Action(action_type="hold", confidence=0.45, rationale_tags=["random_baseline"])
