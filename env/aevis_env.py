"""
AEVIS OpenEnv — Main environment class.
Implements reset(), step(), state() as required by OpenEnv spec.
"""

import random
from typing import Optional
from env.models import Action, Observation, Reward, TaskLevel
from env.patient_loader import PATIENT_CASES
from env.reward import compute_reward


class AEVISEnv:
    """
    AEVIS retinal screening environment.

    Tasks:
      easy   — Binary screening: normal vs refer
      medium — DR severity grading (7 classes)
      hard   — Full patient workflow: classify + urgency + recommendation + report
    """

    MAX_STEPS = 1  # Each case is evaluated in a single step

    def __init__(self, task_level: str = "easy", seed: Optional[int] = None):
        if task_level not in ("easy", "medium", "hard"):
            raise ValueError(f"task_level must be 'easy', 'medium', or 'hard'. Got: {task_level}")
        self.task_level = task_level
        self.seed = seed
        self._rng = random.Random(seed)

        self._current_case: Optional[dict] = None
        self._step_count: int = 0
        self._episode_done: bool = False
        self._history: list = []

    # ── OpenEnv API ───────────────────────────────────────────────

    def reset(self) -> dict:
        """
        Reset the environment and load a new patient case.
        Returns the initial Observation as a dict.
        """
        self._current_case = self._rng.choice(PATIENT_CASES)
        self._step_count = 0
        self._episode_done = False
        self._history = []

        obs = self._build_observation()
        return obs.to_dict()

    def step(self, action: dict) -> dict:
        """
        Take one step in the environment.
        action: dict with keys matching the Action dataclass fields.
        Returns: {"observation": ..., "reward": ..., "done": bool, "info": ...}
        """
        if self._current_case is None:
            raise RuntimeError("Call reset() before step().")
        if self._episode_done:
            raise RuntimeError("Episode is done. Call reset() to start a new case.")

        self._step_count += 1

        # Parse action
        act = Action(
            screening_result=action.get("screening_result"),
            disease_label=action.get("disease_label"),
            confidence=action.get("confidence"),
            urgency=action.get("urgency"),
            recommendation=action.get("recommendation"),
            report_summary=action.get("report_summary"),
        )

        # Compute reward
        reward = compute_reward(
            task_level=self.task_level,
            action=act,
            ground_truth=self._current_case,
            step_number=self._step_count,
            max_steps=self.MAX_STEPS,
        )

        self._episode_done = reward.is_terminal
        self._history.append({
            "step": self._step_count,
            "action": act.to_dict(),
            "reward": reward.to_dict(),
        })

        obs = self._build_observation()

        return {
            "observation": obs.to_dict(),
            "reward": reward.to_dict(),
            "done": self._episode_done,
            "info": {
                "patient_id": self._current_case["patient_id"],
                "task_level": self.task_level,
                "step": self._step_count,
            },
        }

    def state(self) -> dict:
        """
        Return a full snapshot of current environment state.
        """
        return {
            "task_level": self.task_level,
            "episode_done": self._episode_done,
            "step_count": self._step_count,
            "current_patient": self._current_case["patient_id"] if self._current_case else None,
            "history": self._history,
        }

    # ── Helpers ───────────────────────────────────────────────────

    def _build_observation(self) -> Observation:
        case = self._current_case
        return Observation(
            task_level=self.task_level,
            patient_id=case["patient_id"],
            patient_age=case["patient_age"],
            diabetes_years=case["diabetes_years"],
            image_description=case["image_description"],
            previous_actions=[h["action"] for h in self._history],
            step_number=self._step_count,
            task_complete=self._episode_done,
        )

    def _get_task_prompt(self) -> str:
        """Returns the task instruction for the agent."""
        if self.task_level == "easy":
            return (
                "TASK (Easy): Review the retinal scan description and decide whether "
                "this patient should be referred for further examination or is normal. "
                "Respond with screening_result: 'normal' or 'refer'."
            )
        elif self.task_level == "medium":
            return (
                "TASK (Medium): Classify the severity of retinal disease shown. "
                "Choose one: normal, mild_dr, moderate_dr, severe_dr, "
                "proliferative_dr, glaucoma_suspect, amd. "
                "Also provide a confidence score 0.0–1.0."
            )
        else:
            return (
                "TASK (Hard): Perform a full clinical workflow. Provide: "
                "(1) disease_label classification, "
                "(2) urgency level: monitor/refer/urgent, "
                "(3) recommendation (what should happen next), "
                "(4) report_summary (brief patient report). "
                "All four fields required."
            )
