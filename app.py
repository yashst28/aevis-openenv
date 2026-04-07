"""
AEVIS OpenEnv — FastAPI app for Hugging Face Space deployment.
Exposes the environment via HTTP API.
Endpoints: POST /reset, POST /step, GET /state, GET /health
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os

from env.aevis_env import AEVISEnv

app = FastAPI(
    title="AEVIS OpenEnv",
    description="AI-powered retinal screening environment for OpenEnv evaluation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One environment instance per task level
_envs: dict[str, AEVISEnv] = {}


def get_env(task_level: str) -> AEVISEnv:
    if task_level not in _envs:
        _envs[task_level] = AEVISEnv(task_level=task_level, seed=42)
    return _envs[task_level]


# ── Request / Response models ─────────────────────────────────────

class ResetRequest(BaseModel):
    task_level: str = "easy"
    seed: Optional[int] = 42


class StepRequest(BaseModel):
    task_level: str = "easy"
    action: dict


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/health", status_code=200)
def health():
    """Health check — returns 200 if service is running."""
    return {"status": "ok", "service": "AEVIS OpenEnv"}


@app.post("/reset", status_code=200)
def reset(request: ResetRequest):
    """
    Reset the environment for a given task level.
    Returns the first observation.
    """
    if request.task_level not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail="task_level must be 'easy', 'medium', or 'hard'")

    env = AEVISEnv(task_level=request.task_level, seed=request.seed)
    _envs[request.task_level] = env
    obs = env.reset()

    return {
        "status": "ok",
        "observation": obs,
        "task_instructions": env._get_task_prompt(),
    }


@app.post("/step", status_code=200)
def step(request: StepRequest):
    """
    Submit an action and receive reward + next observation.
    """
    if request.task_level not in _envs:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialised. Call /reset first."
        )

    env = get_env(request.task_level)

    try:
        result = env.step(request.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"status": "ok", **result}


@app.get("/state", status_code=200)
def state(task_level: str = "easy"):
    """Return the current environment state."""
    if task_level not in _envs:
        return {"status": "not_initialised", "task_level": task_level}
    env = get_env(task_level)
    return {"status": "ok", "state": env.state()}


@app.get("/tasks", status_code=200)
def list_tasks():
    """List all available tasks with descriptions."""
    return {
        "tasks": [
            {
                "id": "easy",
                "name": "Binary screening",
                "description": "Classify retinal scan as normal or refer",
                "action_fields": ["screening_result"],
            },
            {
                "id": "medium",
                "name": "DR severity grading",
                "description": "Classify disease type and severity across 7 classes",
                "action_fields": ["disease_label", "confidence"],
            },
            {
                "id": "hard",
                "name": "Full patient workflow",
                "description": "Complete clinical workflow: classify + urgency + recommendation + report",
                "action_fields": ["disease_label", "urgency", "recommendation", "report_summary"],
            },
        ]
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
