"""
AEVIS OpenEnv — Server app using openenv-core create_app pattern.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openenv.core.env_server import create_app
    HAS_OPENENV = True
except ImportError:
    HAS_OPENENV = False

try:
    from ..models import AEVISAction, AEVISObservation
except ImportError:
    from models import AEVISAction, AEVISObservation

from aevis_environment import AEVISEnvironment
import uvicorn

if HAS_OPENENV:
    app = create_app(AEVISEnvironment, AEVISAction, AEVISObservation, env_name="aevis-openenv")
else:
    # Fallback to plain FastAPI if openenv-core not available
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from typing import Optional
    from pydantic import BaseModel

    app = FastAPI(title="AEVIS OpenEnv", version="1.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    _env = AEVISEnvironment()

    class ResetRequest(BaseModel):
        task_level: Optional[str] = "easy"
        seed: Optional[int] = 42
        model_config = {"extra": "allow"}

    class StepRequest(BaseModel):
        task_level: Optional[str] = "easy"
        action: Optional[dict] = {}
        model_config = {"extra": "allow"}

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "AEVIS OpenEnv"}

    @app.post("/reset", status_code=200)
    async def reset(request: Optional[ResetRequest] = None):
        if request is None:
            request = ResetRequest()
        obs = _env.reset(task_level=request.task_level or "easy", seed=request.seed or 42)
        return {"status": "ok", "observation": obs.model_dump()}

    @app.post("/step", status_code=200)
    async def step(request: StepRequest):
        action = AEVISAction(**(request.action or {}))
        obs = _env.step(action)
        reward = _env._history[-1]["reward"] if _env._history else {"score": 0.5}
        return {"status": "ok", "observation": obs.model_dump(), "reward": reward, "done": _env._episode_done}

    @app.get("/state", status_code=200)
    def state():
        return {"status": "ok", "state": _env.state.model_dump()}


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
