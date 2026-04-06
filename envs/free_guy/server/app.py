from fastapi import FastAPI
from envs.free_guy.server.environment import FreeGuyEnvironment
from envs.free_guy.models import FreeGuyAction, FreeGuyObservation
from openenv.core.env_server import create_fastapi_app
from fastapi.middleware.cors import CORSMiddleware


app = create_fastapi_app(FreeGuyEnvironment, FreeGuyAction, FreeGuyObservation)

# Keep our security overrides so the WebSocket connects
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/reset")
def reset_environment():
    """Endpoint to start a new episode."""
    obs = env.reset()

    return {
        "observation": obs.dict(), 
        "reward": 0.0, 
        "done": False
    }

@app.post("/step")
def step_environment(action: FreeGuyAction):
    """Endpoint to take an action."""
    obs, reward, terminated, truncated = env.step(action)
    
    # OpenEnv client expects a combined "done" flag
    done = terminated or truncated
    
    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done
    }

@app.get("/state")
def get_state():
    """Endpoint for human debugging."""
    return env.state.dict()