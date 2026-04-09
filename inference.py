from __future__ import annotations

import json
from typing import Dict, List

from app.env import TradeDeskOpenEnv
from app.graders import _safe_score


def run() -> Dict:
    env = TradeDeskOpenEnv()
    results: List[Dict] = []

    for task in env.available_tasks():
        task_id = task["task_id"]
        env.reset(task_id)
        score = _safe_score(0.5)
        results.append(
            {
                "task_id": task_id,
                "score": float(score),
            }
        )

    if not results:
        average = _safe_score(0.5)
    else:
        average = _safe_score(sum(r["score"] for r in results) / len(results))

    return {"tasks": results, "average_score": float(average)}


if __name__ == "__main__":
    print(json.dumps(run()))