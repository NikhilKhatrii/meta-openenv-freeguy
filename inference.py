import os
from openai import OpenAI
from envs.free_guy.client import FreeGuyEnv
from envs.free_guy.models import FreeGuyAction

# Environment Variables with specific defaults
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN") # No default allowed per checklist!

#All LLM calls use the OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN if HF_TOKEN else "dummy_key_for_local_testing"
)

def get_llm_action(observation) -> int:
    """Asks the LLM what to do based on the current observation."""
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
        # Extract the number from the LLM's response
        text_reply = response.choices[0].message.content.strip()
        return int(''.join(filter(str.isdigit, text_reply))[0])
    except Exception:
        return 0 # Fallback to sleep if the LLM crashes or gives bad output

def run_inference():
    # Stdout logs must follow START/STEP/END exactly
    print("START")
    
    # Connect to the local server spun up by Meta's grader
    with FreeGuyEnv(base_url="http://localhost:8000").sync() as env:
        result = env.reset()
        
        # Run one episode (Max 30 days / 720 hours)
        for _ in range(720):
            # Get action from LLM
            action_id = get_llm_action(result.observation)
            action = FreeGuyAction(action_id=action_id)
            
            # Take step
            result = env.step(action)
            
            # Print the exact STEP format required
            print(f"STEP")
            
            if result.done:
                break
                
    print("END")

if __name__ == "__main__":
    run_inference()