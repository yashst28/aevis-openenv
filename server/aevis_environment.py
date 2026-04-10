"""
AEVIS OpenEnv — Server-side environment extending openenv-core Environment.
"""
import random
from typing import Optional
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
except ImportError:
    class Environment:
        pass

try:
    from ..models import AEVISAction, AEVISObservation, AEVISState
except ImportError:
    from models import AEVISAction, AEVISObservation, AEVISState

from patient_loader import PATIENT_CASES
from graders import grade_task1, grade_task2, grade_task3


def clamp_score(score: float) -> float:
    return max(0.001, min(0.999, score))


def compute_reward(task_level: str, action: AEVISAction, ground_truth: dict, step_number: int) -> dict:
    if task_level == "easy":
        grader_score, feedback = grade_task1(action, ground_truth)
    elif task_level == "medium":
        grader_score, feedback = grade_task2(action, ground_truth)
    else:
        grader_score, feedback = grade_task3(action, ground_truth)

    efficiency_bonus = 0.05 if grader_score >= 0.8 and step_number == 1 else 0.0
    safety_penalty = 0.0
    gt_urgency = ground_truth.get("ground_truth_urgency", "monitor")

    if gt_urgency == "urgent":
        if task_level == "easy" and action.screening_result == "normal":
            safety_penalty = 0.2
        elif task_level in ("medium", "hard") and action.disease_label == "normal":
            safety_penalty = 0.2
        elif task_level == "hard" and action.urgency == "monitor":
            safety_penalty = 0.15

    completeness_bonus = 0.0
    if task_level == "hard":
        fields = sum([
            action.disease_label is not None,
            action.urgency is not None,
            action.recommendation is not None and len(action.recommendation) > 20,
            action.report_summary is not None and len(action.report_summary) > 30,
        ])
        if fields == 4:
            completeness_bonus = 0.05
        elif fields <= 1:
            completeness_bonus = -0.05

    raw = grader_score + efficiency_bonus - safety_penalty + completeness_bonus
    final = clamp_score(round(raw, 3))

    return {"score": final, "feedback": feedback, "is_terminal": True}


class AEVISEnvironment(Environment):
    """
    AEVIS retinal screening environment server-side implementation.
    Extends openenv-core Environment base class.
    """

    def __init__(self):
        self._rng = random.Random(42)
        self._current_case: Optional[dict] = None
        self._step_count: int = 0
        self._episode_done: bool = False
        self._history: list = []
        self._task_level: str = "easy"
        try:
            super().__init__()
        except Exception:
            pass

    def reset(self, task_level: str = "easy", seed: Optional[int] = None) -> AEVISObservation:
        if seed is not None:
            self._rng = random.Random(seed)
        self._task_level = task_level
        self._current_case = self._rng.choice(PATIENT_CASES)
        self._step_count = 0
        self._episode_done = False
        self._history = []
        return self._build_observation()

    def step(self, action: AEVISAction) -> AEVISObservation:
        if self._current_case is None:
            raise RuntimeError("Call reset() before step().")
        if self._episode_done:
            raise RuntimeError("Episode done. Call reset().")

        self._step_count += 1
        reward = compute_reward(self._task_level, action, self._current_case, self._step_count)
        self._episode_done = reward["is_terminal"]
        self._history.append({"step": self._step_count, "action": action.model_dump(), "reward": reward})
        return self._build_observation()

    @property
    def state(self) -> AEVISState:
        return AEVISState(
            task_level=self._task_level,
            episode_done=self._episode_done,
            step_count=self._step_count,
            current_patient=self._current_case["patient_id"] if self._current_case else None,
        )

    def _build_observation(self) -> AEVISObservation:
        case = self._current_case
        return AEVISObservation(
            task_level=self._task_level,
            patient_id=case["patient_id"],
            patient_age=case["patient_age"],
            diabetes_years=case["diabetes_years"],
            image_description=case["image_description"],
            previous_actions=[h["action"] for h in self._history],
            step_number=self._step_count,
            task_complete=self._episode_done,
        )
