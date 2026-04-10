"""
AEVIS OpenEnv — Typed models using openenv-core base classes.
"""
from typing import Optional, List
from pydantic import Field

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from pydantic import BaseModel as Action
    from pydantic import BaseModel as Observation
    from pydantic import BaseModel as State


class AEVISAction(Action):
    """Action for AEVIS retinal screening tasks."""
    # Task 1 (Easy)
    screening_result: Optional[str] = Field(None, description="'normal' or 'refer'")
    # Task 2 (Medium)
    disease_label: Optional[str] = Field(None, description="Disease classification label")
    confidence: Optional[float] = Field(None, description="Confidence score 0.0-1.0")
    # Task 3 (Hard)
    urgency: Optional[str] = Field(None, description="'monitor', 'refer', or 'urgent'")
    recommendation: Optional[str] = Field(None, description="Clinical recommendation")
    report_summary: Optional[str] = Field(None, description="Patient report summary")


class AEVISObservation(Observation):
    """Observation returned to the agent."""
    task_level: str = Field(..., description="easy / medium / hard")
    patient_id: str = Field(..., description="Unique patient identifier")
    patient_age: int = Field(..., description="Patient age in years")
    diabetes_years: Optional[int] = Field(None, description="Years since diabetes diagnosis")
    image_description: str = Field(..., description="Retinal fundus findings text")
    previous_actions: List[dict] = Field(default_factory=list)
    step_number: int = Field(0)
    task_complete: bool = Field(False)


class AEVISState(State):
    """Full environment state snapshot."""
    task_level: str = "easy"
    episode_done: bool = False
    step_count: int = 0
    current_patient: Optional[str] = None
