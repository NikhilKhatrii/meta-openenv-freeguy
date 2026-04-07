import os
from openai import OpenAI
from envs.free_guy.client import FreeGuyEnv
from envs.free_guy.models import FreeGuyAction

# LLM connection details
LLM_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=HF_TOKEN if HF_TOKEN else "dummy_key_for_local_testing"
)

def get_llm_action(observation) -> int:
    # ... (Keep your existing get_llm_action code exactly the same) ...
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

def run_inference():
    print("START")
    
    # --- THE CRITICAL FIX ---
    # Meta passes the target environment URL via OPENENV_BASE_URL or API_BASE_URL.
    # We check OPENENV_BASE_URL first, fallback to API_BASE_URL, then fallback to localhost.
    target_url = os.getenv("OPENENV_BASE_URL", os.getenv("API_BASE_URL", "http://localhost:8000"))
    
    # Ensure it doesn't accidentally use the OpenAI API URL for the game environment
    if "api.openai.com" in target_url:
        target_url = "http://localhost:8000"

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
        # Phase 2 requires fail-fast, but we want to see the error in the logs
        raise e
        
    print("END")

if __name__ == "__main__":
    run_inference()
