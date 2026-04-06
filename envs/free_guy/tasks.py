def survival_grader(observation):
    # Reward is based purely on how many days the agent lived out of 30
    return observation.day / 30.0

def wealth_grader(observation):
    # Reward is 1.0 if they hit the $10k cap, 0.0 otherwise
    return 1.0 if observation.money >= 10000 else 0.0

def wellness_grader(observation):
    # High score only if they survived AND stayed happy
    if observation.day > 15 and observation.mood > 0.7:
        return 1.0
    return 0.0