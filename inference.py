from __future__ import annotations

import json
import random
from typing import Dict, List

from agent.llm_agent import EXAMPLE_PROMPT, LLMReasoningAgent, StrategyEngine
from app import config
from app.env import TradeDeskOpenEnv
from visualization.plots import ensure_output_dir, save_all_plots

DETERMINISTIC_SEED = 7
random.seed(DETERMINISTIC_SEED)
__all__ = ["run", "StrategyEngine"]


def run() -> Dict:
    env = TradeDeskOpenEnv()
    agent = LLMReasoningAgent(seed=DETERMINISTIC_SEED)
    summary: Dict = {
        "tasks": [],
        "average_score": 0.0,
        "agent_mode": "llm" if agent.enabled else "strategy_engine",
        "seed": DETERMINISTIC_SEED,
        "example_prompt": EXAMPLE_PROMPT,
    }

    total_score = 0.0
    for task in env.available_tasks():
        task_id = task["task_id"]
        print(f"START task={task_id}")
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
            action, reasoning = agent.act(observation, history)
            next_obs, reward, done, info = env.step(action)
            print(f"STEP action={action.action_type} reward={reward.value:.4f}")

            step_payload = {
                "step_index": observation.step_index,
                "action": action.model_dump(),
                "reward": reward.model_dump(),
                "reasoning": reasoning["reasoning"],
                "indicator_summary": reasoning["indicator_summary"],
                "final_score": info["final_score"],
            }
            history.append(step_payload)
            task_result["steps"].append(step_payload)
            task_result["done"] = done
            task_result["final_score"] = info["final_score"]
            observation = next_obs
            if done:
                break

        print(f"END score={task_result['final_score']:.4f}")
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
