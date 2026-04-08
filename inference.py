from __future__ import annotations

import json
import random
from typing import Dict, List

from app import config
from app.env import TradeDeskOpenEnv
from visualization.plots import ensure_output_dir, save_all_plots
from app.agent import TradingAgent
from app.models import Action

DETERMINISTIC_SEED = 7
random.seed(DETERMINISTIC_SEED)

__all__ = ["run"]


def run() -> Dict:
    env = TradeDeskOpenEnv()
    agent = TradingAgent()

    summary: Dict = {
        "tasks": [],
        "average_score": 0.0,
        "agent_mode": "custom_trading_agent",
        "seed": DETERMINISTIC_SEED,
    }

    total_score = 0.0

    for task in env.available_tasks():
        task_id = task["task_id"]

        print(f"[START] task={task_id}", flush=True)

        observation = env.reset(task_id)
        history: List[Dict] = []

        task_result = {
            "task_id": task_id,
            "difficulty": task["difficulty"],
            "steps": [],
            "done": False,
            "final_score": 0.0,
        }

        while True:
            action_type, reasoning_text = agent.act(observation)

            ticker = observation.market[0].ticker if observation.market else None

            action = Action(
                action_type=action_type,
                ticker=ticker,
                confidence=0.5,
                rationale_tags=["trend", "risk"],
            )

            next_obs, reward, done, info = env.step(action)

            print(
                f"[STEP] step={observation.step_index} action={action.action_type} reward={reward.value:.4f}",
                flush=True,
            )

            step_payload = {
                "step_index": observation.step_index,
                "action": action.model_dump(),
                "reward": reward.model_dump(),
                "reasoning": reasoning_text,
                "final_score": info["final_score"],
            }

            history.append(step_payload)
            task_result["steps"].append(step_payload)
            task_result["done"] = done
            task_result["final_score"] = info["final_score"]

            observation = next_obs

            if done:
                break

        print(
            f"[END] task={task_id} score={task_result['final_score']:.4f} steps={len(task_result['steps'])}",
            flush=True,
        )

        summary["tasks"].append(task_result)
        total_score += task_result["final_score"]

    summary["average_score"] = round(total_score / len(summary["tasks"]), 4)

    return summary


if __name__ == "__main__":
    result = run()

    out_dir = ensure_output_dir(config.OUTPUT_DIR)
    summary_path = out_dir / "baseline_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    save_all_plots(result, output_dir=str(out_dir))