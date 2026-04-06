from openenv.core.env_client import EnvClient, StepResult
from envs.free_guy.models import FreeGuyAction, FreeGuyObservation, FreeGuyState

class FreeGuyEnv(EnvClient[FreeGuyAction, FreeGuyObservation, FreeGuyState]):
    
    def _step_payload(self, action: FreeGuyAction) -> dict:
        return {"action_id": action.action_id}

    def _parse_result(self, payload: dict) -> StepResult:


        inner_data = payload.get("observation", payload)
        
        obs = FreeGuyObservation(**inner_data)
        
        return StepResult(
            observation=obs,
            reward=obs.reward,
            done=obs.done
        )

    def _parse_state(self, payload: dict) -> FreeGuyState:
        return FreeGuyState(**payload)