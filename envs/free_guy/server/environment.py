import random
from openenv.core.env_server import Environment  # <--- Import Meta's base class
from envs.free_guy.models import FreeGuyAction, FreeGuyObservation, FreeGuyState

class FreeGuyEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.max_days = 30
        self.reset()

    def reset(self) -> FreeGuyObservation:
        self.time_of_day = 0
        self.day = 1
        self.mood = 0.5
        self.money = 5000.0
        self.energy = 0.7
        self.sleep = 0.7
        self.sickness = 0
        self.total_steps = 0
        
        return self._get_obs(reward=0.0, done=False)

    def _get_obs(self, reward: float, done: bool) -> FreeGuyObservation:
        return FreeGuyObservation(
            time_of_day=self.time_of_day,
            day=self.day,
            mood=self.mood,
            money=self.money,
            energy=self.energy,
            sleep=self.sleep,
            sickness=self.sickness,
            reward=reward,
            done=done
        )

    def _is_workday(self) -> bool:
        return ((self.day - 1) % 7) < 5

    def step(self, action: FreeGuyAction) -> FreeGuyObservation:
        reward = 0.0
        terminated = False
        truncated = False
        self.total_steps += 1
        
        act_id = action.action_id

        if act_id == 0:  
            self.energy = min(1.0, self.energy + 0.3)
            self.sleep = min(1.0, self.sleep + 0.3)
            self.mood = max(0.0, self.mood - 0.05)
        elif act_id == 1: 
            if self._is_workday():
                self.money += 150.0
                self.energy -= 0.2
                self.sleep -= 0.1
                self.mood -= 0.1
            else:
                reward -= 5.0
        elif act_id == 2:  
            self.mood = min(1.0, self.mood + 0.1)
            self.money -= 50.0
            self.energy -= 0.05
        elif act_id == 3:  
            if not self._is_workday():
                self.mood = min(1.0, self.mood + 0.4)
                self.money -= 200.0
                self.energy -= 0.3
            else:
                reward -= 5.0
                
        self.time_of_day += 1
        if self.time_of_day > 23:
            self.time_of_day = 0
            self.day += 1
            
        if random.random() < 0.02:
            self.sickness = 1
            self.energy -= 0.2
            self.mood -= 0.2
            reward -= 2.0
        else:
            self.sickness = 0
            
        if self.energy <= 0 or self.sleep <= 0 or self.mood <= 0 or self.money <= 0:
            terminated = True
            reward -= 100.0
            
        if self.day > self.max_days:
            truncated = True
            if self.mood >= 0.7 and self.energy >= 0.7 and self.sleep >= 0.7 and self.money >= 10000:
                reward += 200.0  
            else:
                reward += (self.mood + self.energy + self.sleep) * 10.0
                
        if not terminated and not truncated:

            reward += (self.mood + self.energy + self.sleep) * 0.1
            
            if self.money < 1000:
                reward -= 0.5
                
        done = terminated or truncated
        
        return self._get_obs(reward=reward, done=done)

    @property
    def state(self) -> FreeGuyState:
        return FreeGuyState(
            is_workday=self._is_workday(),
            total_steps_taken=self.total_steps,
            current_status="Alive" if self.energy > 0 else "Failed"
        )