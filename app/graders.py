from __future__ import annotations

import math
from typing import Dict, List

from .models import Action


def _safe_score(score: float) -> float:
    eps = 1e-6
    fallback = 0.5

    if score is None:
        return fallback

    try:
        safe_value = float(score)
    except (TypeError, ValueError):
        return fallback

    if math.isnan(safe_value):
        return fallback

    safe_value = max(eps, min(1 - eps, safe_value))
    return round(safe_value, 6)


def _tag_overlap(submitted: List[str], gold: List[str]) -> float:
    if not gold:
        return 0.0
    submitted_tags = set(submitted)
    gold_tags = set(gold)
    common = submitted_tags & gold_tags
    return len(common) / len(gold_tags)


def grade_easy(action: Action, step: Dict) -> float:
    gold = step["gold"]
    score = 0.0

    if action.action_type == gold["action_type"]:
        score += 0.40

    if action.ticker == gold["ticker"]:
        score += 0.10

    low, high = gold["confidence_band"]
    if low <= action.confidence <= high:
        score += 0.20

    score += 0.30 * _tag_overlap(action.rationale_tags, gold["rationale_tags"])

    score = round(score, 4)
    return _safe_score(score)


def grade_medium(action: Action, step: Dict) -> float:
    gold = step["gold"]
    score = 0.0

    if action.action_type == gold["action_type"]:
        score += 0.35
    elif action.action_type == "reduce" and gold["action_type"] == "sell":
        score += 0.15

    if action.ticker == gold["ticker"]:
        score += 0.10

    if action.order_fraction is not None:
        low, high = gold["order_fraction_band"]
        if low <= action.order_fraction <= high:
            score += 0.25
        elif action.order_fraction >= 0.50:
            score += 0.10
        elif gold["action_type"] == "hold" and action.order_fraction <= high:
            score += 0.15

    conf_low, conf_high = gold["confidence_band"]
    if conf_low <= action.confidence <= conf_high:
        score += 0.10

    score += 0.20 * _tag_overlap(action.rationale_tags, gold["rationale_tags"])

    score = round(score, 4)
    return _safe_score(score)


def grade_hard(action: Action, task: Dict, step: Dict) -> float:
    gold = step["gold"]
    constraints = task["constraints"]
    score = 0.0

    if action.action_type == "rebalance":
        score += 0.10

    allocations = action.target_allocations or {}

    if allocations:
        top_pick = max(allocations.items(), key=lambda x: x[1])[0]

        if top_pick == gold["top_pick"]:
            score += 0.20

        max_weight = max(allocations.values())
        if max_weight <= constraints["max_single_name_weight"]:
            score += 0.15

        for ticker, band in gold["allocation_bands"].items():
            value = allocations.get(ticker, 0.0)
            low, high = band
            if low <= value <= high:
                score += 0.15
            elif max(0.0, low - 0.08) <= value <= high + 0.08:
                score += 0.08

        cash_reserve = max(0.0, 1.0 - sum(allocations.values()))
        cash_low, cash_high = gold["cash_reserve_band"]
        if cash_low <= cash_reserve <= cash_high:
            score += 0.10

        if abs(sum(allocations.values()) - 1.0) <= 0.25:
            score += 0.10

        score += 0.05 * _tag_overlap(
            action.rationale_tags,
            gold.get("rationale_tags", [])
        )

    score = round(score, 4)
    return _safe_score(score)


def grade_action(action: Action, task: Dict, step_index: int) -> float:
    difficulty = task["difficulty"]
    step = task["scenario_steps"][step_index]

    if difficulty == "easy":
        return grade_easy(action, step)
    elif difficulty == "medium":
        return grade_medium(action, step)
    elif difficulty == "hard":
        return grade_hard(action, task, step)
    else:
        raise ValueError(f"Unsupported difficulty: {difficulty}")