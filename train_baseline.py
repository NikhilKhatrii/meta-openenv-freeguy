import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from envs.free_guy.client import FreeGuyEnv
from envs.free_guy.models import FreeGuyAction
from stable_baselines3.common.callbacks import CheckpointCallback

class GymWrapper(gym.Env):
    def __init__(self, openenv_client):
        super().__init__()
        self.client = openenv_client
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(7,), 
            dtype=np.float32
        )

    def _flatten_obs(self, obs):
        """Squashes all variables into a 0.0 to 1.0 scale for the Neural Network"""
        norm_time = obs.time_of_day / 23.0
        norm_day = obs.day / 30.0
        norm_money = min(1.0, obs.money / 10000.0)
        
        return np.array([
            norm_time, norm_day, obs.mood, 
            norm_money, obs.energy, obs.sleep, float(obs.sickness)
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        result = self.client.reset()
        return self._flatten_obs(result.observation), {}

    def step(self, action):
        pydantic_action = FreeGuyAction(action_id=int(action))
        result = self.client.step(pydantic_action)
        
        obs = result.observation
        
        terminated = bool(obs.energy <= 0 or obs.sleep <= 0 or obs.mood <= 0 or obs.money <= 0)
        truncated = bool(obs.day > 30)

        return self._flatten_obs(obs), result.reward, terminated, truncated, {}

# --- 2. THE TRAINING LOOP ---
print("🔌 Connecting to Free Guy API...")
with FreeGuyEnv(base_url="http://localhost:8000").sync() as openenv_client:
    save_frequency = 10000
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_frequency, 
        save_path="./_saved_models/",
        name_prefix="FreeGuy"
    )
    # Wrap our client
    env = GymWrapper(openenv_client)
    
    # Initialize the PPO Neural Network
    print("Initializing PPO Agent...")
    model = PPO("MlpPolicy", env, verbose=1,device="cpu")
    
    print("Starting Training Phase...")
    model.learn(total_timesteps=5_00_000, callback=checkpoint_callback)
    model.save("freeguy")
    
    for episode in range(1, 6):
        obs, info = env.reset()
        total_reward = 0.0
        
        print(f"\n--- Starting Life {episode} ---")
        
        for hour in range(720):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            raw_time = int(obs[0] * 23.0)
            raw_day = int(obs[1] * 30.0)
            raw_money = obs[3] * 10000.0
            
            if hour > 0 and hour % 24 == 0:
                print(f"🌅 Finished Day {raw_day - 1} -> Money: ${raw_money:.2f} | Energy: {obs[4]:.2f} | Mood: {obs[2]:.2f}")
                
            if terminated or truncated:
                reason = "SURVIVED 30 DAYS 🎉" if truncated else "DIED 💀"
                print(f"{reason} at Day {raw_day}, Hour {raw_time}")
                break