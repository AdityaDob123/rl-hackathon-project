from __future__ import annotations

from app.models import Action, Reward


def build_reward(action: Action, task: dict, final_score: float, valid: bool = True) -> Reward:
    components = {
        "score_progress": round(final_score * 0.75, 4),
        "confidence_calibration": 0.05 if 0.4 <= action.confidence <= 0.95 else -0.05,
        "action_validity": 0.05 if valid else -0.25,
        "task_completion_bonus": 0.10 if final_score >= 0.85 else (0.03 if final_score >= 0.50 else 0.0),
    }

    if task["difficulty"] == "medium" and action.action_type in {"sell", "reduce", "hold"}:
        components["risk_awareness"] = 0.05
    elif task["difficulty"] == "hard" and action.action_type == "rebalance":
        components["risk_awareness"] = 0.05
    else:
        components["risk_awareness"] = 0.0

    raw_value = sum(components.values())
    clipped_value = max(-1.0, min(1.0, round(raw_value, 4)))
    message = (
        f"Task {task['task_id']} evaluated with final score {final_score:.2f}; "
        f"dense reward clipped to {clipped_value:.2f}."
    )
    return Reward(value=clipped_value, components=components, message=message)
