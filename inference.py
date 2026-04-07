import os
from openai import OpenAI
from envs.free_guy.client import FreeGuyEnv
from envs.free_guy.models import FreeGuyAction

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

def run_inference():
    # Define a task name for the grader
    task_name = "freeguy_survival_eval"
    
    # 1. EXACT FORMAT: [START] task=NAME with flush=True
    print(f"[START] task={task_name}", flush=True)
    
    target_url = os.getenv("OPENENV_BASE_URL", "http://localhost:7860")
    
    # Initialize variables so they exist even if it crashes
    step_count = 0
    total_reward = 0.0
    
    try:
        with FreeGuyEnv(base_url=target_url).sync() as env:
            result = env.reset()
            
            for step in range(1, 721): # Max 30 days
                step_count = step
                action_id = get_llm_action(result.observation)
                action = FreeGuyAction(action_id=action_id)
                
                result = env.step(action)
                
                # Grab reward (default to 0.0 if not found)
                current_reward = getattr(result.observation, 'reward', 0.0)
                total_reward += current_reward
                
                # 2. EXACT FORMAT: [STEP] step=N reward=R with flush=True
                print(f"[STEP] step={step_count} reward={current_reward}", flush=True)
                
                if result.done:
                    break
                    
            # Create a simple score (1.0 if they survived all 720 hours)
            final_score = step_count / 720.0
            
            # 3. EXACT FORMAT: [END] task=NAME score=S steps=N with flush=True
            print(f"[END] task={task_name} score={final_score:.2f} steps={step_count}", flush=True)
            
    except Exception as e:
        # If it crashes mid-game, we MUST still print [END] so the grader doesn't freeze
        print(f"[END] task={task_name} score=0.0 steps={step_count}", flush=True)
        raise e

if __name__ == "__main__":
    run_inference()
