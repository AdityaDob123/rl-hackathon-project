from __future__ import annotations

from typing import Optional

from app.models import Action, Observation, StateView


class StateManager:
    def __init__(self) -> None:
        self.reset_all()

    def reset_all(self) -> None:
        self.task_id: Optional[str] = None
        self.difficulty = None
        self.done = False
        self.step_index = 0
        self.max_steps = 0
        self.last_action: Optional[Action] = None
        self.cumulative_reward: float = 0.0
        self.final_score: Optional[float] = None
        self.observation: Optional[Observation] = None

    def load_task(self, observation: Observation) -> None:
        self.task_id = observation.task_id
        self.difficulty = observation.difficulty
        self.done = False
        self.step_index = observation.step_index
        self.max_steps = observation.max_steps
        self.last_action = None
        self.cumulative_reward = 0.0
        self.final_score = None
        self.observation = observation

    def apply_step(
        self,
        action: Action,
        reward_value: float,
        final_score: float,
        next_observation: Observation,
        done: bool,
    ) -> None:
        self.step_index = next_observation.step_index
        self.max_steps = next_observation.max_steps
        self.last_action = action
        self.cumulative_reward += reward_value
        self.final_score = final_score
        self.observation = next_observation
        self.done = done

    def state_view(self) -> StateView:
        return StateView(
            task_id=self.task_id,
            difficulty=self.difficulty,
            done=self.done,
            step_index=self.step_index,
            max_steps=self.max_steps,
            last_action=self.last_action,
            cumulative_reward=round(self.cumulative_reward, 4),
            final_score=self.final_score,
            observation=self.observation,
        )
