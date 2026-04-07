from __future__ import annotations

from typing import Dict, List

from app.models import Action


def _tag_overlap(submitted: List[str], gold: List[str]) -> float:
    if not gold:
        return 0.0
    submitted_set = set(submitted)
    gold_set = set(gold)
    return len(submitted_set & gold_set) / len(gold_set)


def grade_easy(action: Action, step: Dict) -> float:
    gold = step["gold"]
    score = 0.0
    if action.action_type == gold["action_type"]:
        score += 0.40
    if action.ticker == gold["ticker"]:
        score += 0.10

    lo, hi = gold["confidence_band"]
    if lo <= action.confidence <= hi:
        score += 0.20

    score += 0.30 * _tag_overlap(action.rationale_tags, gold["rationale_tags"])
    return max(0.0, min(1.0, round(score, 4)))



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
        lo, hi = gold["order_fraction_band"]
        if lo <= action.order_fraction <= hi:
            score += 0.25
        elif action.order_fraction >= 0.50:
            score += 0.10
        elif gold["action_type"] == "hold" and action.order_fraction <= hi:
            score += 0.15

    clo, chi = gold["confidence_band"]
    if clo <= action.confidence <= chi:
        score += 0.10

    score += 0.20 * _tag_overlap(action.rationale_tags, gold["rationale_tags"])
    return max(0.0, min(1.0, round(score, 4)))



def grade_hard(action: Action, task: Dict, step: Dict) -> float:
    score = 0.0
    gold = step["gold"]
    constraints = task["constraints"]

    if action.action_type == "rebalance":
        score += 0.10

    allocations = action.target_allocations or {}
    if allocations:
        top_pick = max(allocations.items(), key=lambda kv: kv[1])[0]
        if top_pick == gold["top_pick"]:
            score += 0.20

        max_weight = max(allocations.values()) if allocations else 0.0
        if max_weight <= constraints["max_single_name_weight"]:
            score += 0.15

        for ticker, band in gold["allocation_bands"].items():
            val = allocations.get(ticker, 0.0)
            lo, hi = band
            if lo <= val <= hi:
                score += 0.15
            elif max(0.0, lo - 0.08) <= val <= hi + 0.08:
                score += 0.08

        cash_reserve = max(0.0, 1.0 - sum(allocations.values()))
        clo, chi = gold["cash_reserve_band"]
        if clo <= cash_reserve <= chi:
            score += 0.10

        if abs(sum(allocations.values()) - 1.0) <= 0.25:
            score += 0.10

        score += 0.05 * _tag_overlap(action.rationale_tags, gold.get("rationale_tags", []))

    return max(0.0, min(1.0, round(score, 4)))



def grade_action(action: Action, task: Dict, step_index: int) -> float:
    difficulty = task["difficulty"]
    step = task["scenario_steps"][step_index]
    if difficulty == "easy":
        return grade_easy(action, step)
    if difficulty == "medium":
        return grade_medium(action, step)
    if difficulty == "hard":
        return grade_hard(action, task, step)
    raise ValueError(f"Unsupported difficulty: {difficulty}")
