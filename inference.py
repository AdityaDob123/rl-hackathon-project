from __future__ import annotations

import json
from typing import Dict, List

from agent.llm_agent import StrategyEngine  # noqa: F401 — re-exported for tests
from app.env import TradeDeskOpenEnv
from app.graders import sanitize_score


def run() -> Dict:
    env = TradeDeskOpenEnv()
    agent = StrategyEngine()
    results: List[Dict] = []

    for task in env.available_tasks():
        task_id = task["task_id"]
        env.reset(task_id)

        done = False
        final_score = sanitize_score(0.5)

        while not done:
            obs = env.state().observation
            if obs is None:
                break
            action = agent.act(obs)
            _, reward, done, info = env.step(action)
            final_score = info.get("final_score", final_score)

        results.append(
            {
                "task_id": task_id,
                "done": True,
                "final_score": float(final_score),
                "score": float(final_score),
            }
        )

    if not results:
        average = sanitize_score(0.5)
    else:
        average = sanitize_score(sum(r["score"] for r in results) / len(results))

    return {"tasks": results, "average_score": float(average)}


if __name__ == "__main__":
    print(json.dumps(run()))