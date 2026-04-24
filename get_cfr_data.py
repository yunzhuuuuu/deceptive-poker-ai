import rlcard
from rlcard.agents import CFRAgent
import numpy as np
import pickle

# Config
ENV_NAME = 'limit-holdem'
TRAIN_ITERATIONS = 100
NUM_EPISODES = 1000
DATA_PATH = './cfr_dataset.pkl'

# Env
env = rlcard.make(
    ENV_NAME,
    config={'allow_step_back': True, 'seed': 42}
)

# Train CFR 
agent = CFRAgent(env)

print("Training CFR...")
for i in range(TRAIN_ITERATIONS):
    agent.train()
    print(f"Iteration {i}")

print("Training done.")

# Self-play 
env.set_agents([agent, agent])

dataset = []

for episode in range(NUM_EPISODES):
    trajectories, _ = env.run(is_training=False)

    for player_id in range(len(trajectories)):

        for transition in trajectories[player_id]:

            if not isinstance(transition, dict):
                continue

            # STATE 
            raw_obs = transition['raw_obs'] # human understandable observations
            obs = transition['obs']  # numpy array
            state_key = obs.tobytes()

            legal_actions = list(transition['legal_actions'].keys())

            action_record = transition['action_record']
            if len(action_record) == 0:
                continue

            action_taken = action_record[-1][1]

            # TARGET LABELS
            action_probs = agent.action_probs(
                state_key,
                legal_actions,
                agent.policy
            )

            # sanity
            if np.sum(action_probs) == 0:
                continue

            dataset.append({
                "features": obs,  # ML input
                "state_key": state_key,  # CFR lookup key
                "action_probs": action_probs,  # supervision target
                "action_taken": action_taken
            })

    if episode % 100 == 0:
        print(f"Episode {episode}, dataset size = {len(dataset)}")

# Save
with open(DATA_PATH, 'wb') as f:
    pickle.dump(dataset, f)

print(f"Saved {len(dataset)} samples to {DATA_PATH}")

print(dataset[0]['features'].shape)
print(dataset[0]['action_probs'])
print(sum(dataset[0]['action_probs']))
