import os
from openai import OpenAI
from envs.free_guy.client import FreeGuyEnv
from envs.free_guy.models import FreeGuyAction

# ==========================================
# 1. THE AI CONNECTION (Strictly HTTP)
# ==========================================
LLM_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=HF_TOKEN if HF_TOKEN else "dummy_key_for_local_testing"
)

def get_llm_action(observation) -> int:
    prompt = f"""
    You are managing a life simulator. 
    Current Stats: Time: {observation.time_of_day}, Day: {observation.day}, Mood: {observation.mood}, Money: ${observation.money}, Energy: {observation.energy}, Sleep: {observation.sleep}.
    Actions: 0 (Sleep), 1 (Work), 2 (Small Leisure), 3 (Weekend Adventure).
    Reply with ONLY a single integer (0, 1, 2, or 3).
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.3
        )
        text_reply = response.choices[0].message.content.strip()
        return int(''.join(filter(str.isdigit, text_reply))[0])
    except Exception:
        return 0

# ==========================================
# 2. THE GAME CONNECTION (Strictly WebSockets)
# ==========================================
def run_inference():
    print("START")
    
    # We strictly isolate the environment URL. 
    # If the grader doesn't provide it, we default to the port we defined in openenv.yaml (7860)
    target_url = os.getenv("OPENENV_BASE_URL", "http://localhost:7860")
    
    print(f"Connecting to environment at: {target_url}")
    
    try:
        with FreeGuyEnv(base_url=target_url).sync() as env:
            result = env.reset()
            
            for _ in range(720):
                action_id = get_llm_action(result.observation)
                action = FreeGuyAction(action_id=action_id)
                result = env.step(action)
                print(f"STEP")
                
                if result.done:
                    break
                    
    except Exception as e:
        print(f"Connection Error Details: {e}")
        raise e
        
    print("END")

if __name__ == "__main__":
    run_inference()
