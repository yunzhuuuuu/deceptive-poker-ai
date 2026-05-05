import rlcard
from rlcard.agents import CFRAgent
import numpy as np
import pickle
import os
from collections import Counter

# Config
ENV_NAME = 'limit-holdem'
TRAIN_ITERATIONS = 500
NUM_EPISODES = 5000
DATA_PATH = './data/cfr_dataset_5000eps_balanced.pkl'
MODEL_PATH = './data/cfr_model_500itr'

# Env
env = rlcard.make(
    ENV_NAME,
    config={'allow_step_back': True, 'seed': 42}
)

# Train CFR 
agent = CFRAgent(env, model_path=MODEL_PATH)

if os.path.exists(MODEL_PATH):
    print("Loading saved CFR model...")
    agent.load()
else:
    print("Training CFR...")
    for i in range(TRAIN_ITERATIONS):
        agent.train()
        print(f"Iteration {i}")
    agent.save()
    print("Training done, model saved.")

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
                agent.average_policy  
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


# Check action distribution
actions = [d['action_taken'] for d in dataset]
action_counts = Counter(actions)
print(f"\nTotal samples: {len(dataset)}")
print(f"\nAction distribution:")
for action, count in sorted(action_counts.items()):
    print(f"  {action:10s}: {count:6d} ({count/len(dataset)*100:.1f}%)")


# Filter for 50-50 uniform/non-uniform split
print("FILTERING FOR 50-50 UNIFORM/NON-UNIFORM SPLIT")

# Separate uniform and non-uniform examples
uniform_examples = []
nonuniform_examples = []

for d in dataset:
    probs = d['action_probs']
    probs_nonzero = probs[probs > 0]  # Only look at legal actions
    if len(probs_nonzero) <= 1:
        # Only one legal action OR no legal actions - skip this sample
        continue
    elif np.std(probs_nonzero) < 0.01:
        # All legal actions have equal probability - uniform
        uniform_examples.append(d)
    else:
        # Mixed strategy OR deterministic - non-uniform
        nonuniform_examples.append(d)

print(f"\nOriginal dataset: {len(dataset)} samples")
print(f"  Uniform: {len(uniform_examples)} ({len(uniform_examples)/len(dataset)*100:.1f}%)")
print(f"  Non-uniform: {len(nonuniform_examples)} ({len(nonuniform_examples)/len(dataset)*100:.1f}%)")

# Take 50-50 split
target_size = min(len(uniform_examples), len(nonuniform_examples))

# Randomly sample to balance
np.random.seed(42)
uniform_sampled = np.random.choice(len(uniform_examples), target_size, replace=False)
nonuniform_sampled = np.random.choice(len(nonuniform_examples), target_size, replace=False)

balanced_dataset = []
for idx in uniform_sampled:
    balanced_dataset.append(uniform_examples[idx])
for idx in nonuniform_sampled:
    balanced_dataset.append(nonuniform_examples[idx])

# Shuffle
np.random.shuffle(balanced_dataset)

print(f"\nBalanced dataset: {len(balanced_dataset)} samples")

# Check new action distribution
balanced_actions = [d['action_taken'] for d in balanced_dataset]
balanced_counts = Counter(balanced_actions)
print(f"\nBalanced action distribution:")
for action, count in sorted(balanced_counts.items()):
    print(f"  {action:10s}: {count:6d} ({count/len(balanced_dataset)*100:.1f}%)")

# Replace original dataset with balanced one
dataset = balanced_dataset


# Save
with open(DATA_PATH, 'wb') as f:
    pickle.dump(dataset, f)

print(f"Saved {len(dataset)} samples to {DATA_PATH}")
