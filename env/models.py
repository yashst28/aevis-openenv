"""
AEVIS OpenEnv — Typed models for Observation, Action, Reward
"""
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class DiseaseLabel(str, Enum):
    NORMAL = "normal"
    MILD_DR = "mild_dr"
    MODERATE_DR = "moderate_dr"
    SEVERE_DR = "severe_dr"
    PROLIFERATIVE_DR = "proliferative_dr"
    GLAUCOMA_SUSPECT = "glaucoma_suspect"
    AMD = "amd"


class UrgencyLevel(str, Enum):
    MONITOR = "monitor"
    REFER = "refer"
    URGENT = "urgent"


class TaskLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class Observation:
    """What the AI agent sees at each step."""
    task_level: str
    patient_id: str
    patient_age: int
    diabetes_years: Optional[int]       # None if no diabetes
    image_description: str              # Text description of retinal features
    previous_actions: List[dict] = field(default_factory=list)
    step_number: int = 0
    task_complete: bool = False

    def to_dict(self) -> dict:
        return {
            "task_level": self.task_level,
            "patient_id": self.patient_id,
            "patient_age": self.patient_age,
            "diabetes_years": self.diabetes_years,
            "image_description": self.image_description,
            "previous_actions": self.previous_actions,
            "step_number": self.step_number,
            "task_complete": self.task_complete,
        }


@dataclass
class Action:
    """What the AI agent submits as a decision."""
    # Task 1 (Easy)
    screening_result: Optional[str] = None       # "normal" or "refer"

    # Task 2 (Medium)
    disease_label: Optional[str] = None          # DiseaseLabel value
    confidence: Optional[float] = None           # 0.0 - 1.0

    # Task 3 (Hard)
    urgency: Optional[str] = None                # UrgencyLevel value
    recommendation: Optional[str] = None         # Free text recommendation
    report_summary: Optional[str] = None         # Brief report

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class Reward:
    """Step-by-step reward with detailed feedback."""
    score: float                        # 0.0 → 1.0
    breakdown: dict = field(default_factory=dict)
    feedback: str = ""
    is_terminal: bool = False

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "breakdown": self.breakdown,
            "feedback": self.feedback,
            "is_terminal": self.is_terminal,
        }
