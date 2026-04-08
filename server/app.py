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
    task_level: Optional[str] = "easy"
    seed: Optional[int] = 42

    model_config = {"extra": "allow"}


class StepRequest(BaseModel):
    task_level: Optional[str] = "easy"
    action: Optional[dict] = {}

    model_config = {"extra": "allow"}


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/health", status_code=200)
def health():
    return {"status": "ok", "service": "AEVIS OpenEnv"}


@app.post("/reset", status_code=200)
async def reset(request: Optional[ResetRequest] = None):
    """
    Reset the environment. Accepts empty body or JSON with task_level and seed.
    """
    if request is None:
        request = ResetRequest()

    task_level = request.task_level or "easy"
    seed = request.seed if request.seed is not None else 42

    if task_level not in ("easy", "medium", "hard"):
        task_level = "easy"

    env = AEVISEnv(task_level=task_level, seed=seed)
    _envs[task_level] = env
    obs = env.reset()

    return {
        "status": "ok",
        "observation": obs,
        "task_instructions": env._get_task_prompt(),
    }


@app.post("/step", status_code=200)
async def step(request: StepRequest):
    """
    Submit an action and receive reward + next observation.
    """
    task_level = request.task_level or "easy"

    if task_level not in _envs:
        # Auto-reset if not initialised
        env = AEVISEnv(task_level=task_level, seed=42)
        _envs[task_level] = env
        env.reset()

    env = get_env(task_level)

    try:
        result = env.step(request.action or {})
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"status": "ok", **result}


@app.get("/state", status_code=200)
def state(task_level: str = "easy"):
    if task_level not in _envs:
        return {"status": "not_initialised", "task_level": task_level}
    env = get_env(task_level)
    return {"status": "ok", "state": env.state()}


@app.get("/tasks", status_code=200)
def list_tasks():
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
