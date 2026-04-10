"""
AEVIS OpenEnv — Client for connecting to the environment server.
"""
try:
    from openenv.core.env_client import EnvClient
except ImportError:
    EnvClient = object

try:
    from .models import AEVISAction, AEVISObservation, AEVISState
except ImportError:
    from models import AEVISAction, AEVISObservation, AEVISState


class AEVISEnv(EnvClient):
    """
    Client for the AEVIS retinal screening environment.

    Usage:
        async with AEVISEnv(base_url="https://yash0028-aevis-openenv.hf.space") as env:
            obs = await env.reset()
            result = await env.step(AEVISAction(screening_result="refer"))
    """
    action_type = AEVISAction
    observation_type = AEVISObservation
    state_type = AEVISState
