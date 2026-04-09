from __future__ import annotations

import json
import random
from typing import Dict, List

from agent.llm_agent import EXAMPLE_PROMPT, LLMReasoningAgent, StrategyEngine
from app import config
from app.env import TradeDeskOpenEnv
from app.graders import _safe_score
from visualization.plots import ensure_output_dir, save_all_plots
from app.agent import TradingAgent
from app.models import Action

DETERMINISTIC_SEED = 7
random.seed(DETERMINISTIC_SEED)

__all__ = ["run", "StrategyEngine"]


def run() -> Dict:
    env = TradeDeskOpenEnv()

    llm_agent = LLMReasoningAgent(seed=DETERMINISTIC_SEED)
    custom_agent = TradingAgent()

    summary: Dict = {
        "tasks": [],
        "average_score": 0.0,
        "agent_mode": "hybrid_llm_custom",
        "seed": DETERMINISTIC_SEED,
        "example_prompt": EXAMPLE_PROMPT,
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
            # Try LLM agent first
            try:
                action, reasoning = llm_agent.act(observation, history)

            # Fallback to custom agent
            except Exception:
                action_type, reasoning_text = custom_agent.act(observation)

                ticker = observation.market[0].ticker if observation.market else None

                # Build valid Action with all required fields
                if action_type in ("buy", "sell", "reduce"):
                    action = Action(
                        action_type=action_type,
                        ticker=ticker,
                        order_fraction=0.25 if action_type != "sell" else 1.0,
                        confidence=0.5,
                        rationale_tags=["trend_label", "risk_management"],
                    )
                elif action_type == "rebalance":
                    tickers = [s.ticker for s in observation.market]
                    allocs = {t: round(0.8 / len(tickers), 2) for t in tickers} if tickers else {}
                    action = Action(
                        action_type="rebalance",
                        target_allocations=allocs,
                        confidence=0.5,
                        rationale_tags=["risk_balanced", "cash_buffer"],
                    )
                else:
                    action = Action(
                        action_type="hold",
                        confidence=0.5,
                        rationale_tags=["capital_preservation"],
                    )

                reasoning = {
                    "reasoning": reasoning_text,
                    "indicator_summary": "fallback_custom_agent",
                }

            next_obs, reward, done, info = env.step(action)

            print(
                f"[STEP] step={observation.step_index} action={action.action_type} reward={reward.value:.4f}",
                flush=True,
            )

            step_payload = {
                "step_index": observation.step_index,
                "action": action.model_dump(),
                "reward": reward.model_dump(),
                "reasoning": reasoning["reasoning"],
                "indicator_summary": reasoning.get("indicator_summary", ""),
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

    try:
        average_score = total_score / len(summary["tasks"])
    except ZeroDivisionError:
        average_score = 0.5

    summary["average_score"] = _safe_score(average_score)

    return summary


if __name__ == "__main__":
    result = run()

    out_dir = ensure_output_dir(config.OUTPUT_DIR)
    summary_path = out_dir / "baseline_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    save_all_plots(result, output_dir=str(out_dir))