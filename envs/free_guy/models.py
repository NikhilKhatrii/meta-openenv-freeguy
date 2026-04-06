from pydantic import BaseModel, Field

class FreeGuyAction(BaseModel):
    action_id: int = Field(..., ge=0, le=3)

class FreeGuyObservation(BaseModel):
    time_of_day: int = Field(...)
    day: int = Field(...)
    mood: float = Field(...)
    money: float = Field(...)
    energy: float = Field(...)
    sleep: float = Field(...)
    sickness: int = Field(...)
    # --- ADD THESE NEW METRICS ---
    reward: float = Field(default=0.0)
    done: bool = Field(default=False)

class FreeGuyState(BaseModel):
    is_workday: bool
    total_steps_taken: int
    current_status: str