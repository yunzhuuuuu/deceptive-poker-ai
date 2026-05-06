"""
Generate CFR training data for poker agents

1. Trains a CFR agent on Limit Hold'em
2. Collects state-action pairs through self-play
3. Balances dataset 50-50 uniform/non-uniform strategies
4. Saves to pickle file for supervised learning
"""

import rlcard
from rlcard.agents import CFRAgent
import numpy as np
import pickle
import os
from collections import Counter

# Configuration
ENV_NAME = 'limit-holdem'
TRAIN_ITERATIONS = 500
NUM_EPISODES = 5000
DATA_PATH = './data/cfr_dataset_5000eps_balanced.pkl'
MODEL_PATH = './data/cfr_model_500itr'

print("CFR DATA COLLECTION")

# Initialize environment
env = rlcard.make(ENV_NAME, config={'allow_step_back': True, 'seed': 42})

# Train or load CFR agent
agent = CFRAgent(env, model_path=MODEL_PATH)

if os.path.exists(MODEL_PATH):
    print(f"Loading saved CFR model from {MODEL_PATH}...")
    agent.load()
else:
    print(f"Training CFR for {TRAIN_ITERATIONS} iterations...")
    for i in range(TRAIN_ITERATIONS):
        agent.train()
        print(f"  Iteration {i + 1}/{TRAIN_ITERATIONS}")
    agent.save()
    print("CFR training complete, model saved.")

# Collect data through self-play
print(f"\nCollecting data from {NUM_EPISODES} games...")
env.set_agents([agent, agent])
dataset = []

for episode in range(NUM_EPISODES):
    trajectories, _ = env.run(is_training=False)

    for player_id in range(len(trajectories)):
        for transition in trajectories[player_id]:
            if not isinstance(transition, dict):
                continue

            # Extract state information
            obs = transition['obs']
            state_key = obs.tobytes()
            legal_actions = list(transition['legal_actions'].keys())
            
            # Get action taken
            action_record = transition['action_record']
            if len(action_record) == 0:
                continue
            action_taken = action_record[-1][1]

            # Get CFR policy for this state
            action_probs = agent.action_probs(
                state_key,
                legal_actions,
                agent.average_policy
            )

            # Skip invalid states
            if np.sum(action_probs) == 0:
                continue

            dataset.append({
                "features": obs,
                "state_key": state_key,
                "action_probs": action_probs,
                "action_taken": action_taken
            })

    if (episode + 1) % 500 == 0:
        print(f"  Collected {episode + 1}/{NUM_EPISODES} games ({len(dataset)} samples)")

print(f"Collection complete: {len(dataset)} total samples")

# Analyze original distribution
print("\n" + "=" * 70)
print("ORIGINAL DATASET")

actions = [d['action_taken'] for d in dataset]
action_counts = Counter(actions)
print(f"\nAction distribution:")
for action, count in sorted(action_counts.items()):
    print(f"  {action:10s}: {count:6d} ({count/len(dataset)*100:.1f}%)")

# Balance dataset 50-50 uniform/non-uniform
print("\n" + "=" * 70)
print("BALANCING DATASET (50-50 UNIFORM/NON-UNIFORM)")
print("=" * 70)

uniform_examples = []
nonuniform_examples = []

for d in dataset:
    probs = d['action_probs']
    probs_nonzero = probs[probs > 0]
    
    if len(probs_nonzero) <= 1:
        continue  # Skip trivial cases
    elif np.std(probs_nonzero) < 0.01:
        uniform_examples.append(d)  # Uniform strategy
    else:
        nonuniform_examples.append(d)  # Mixed/deterministic strategy

print(f"\nBefore balancing:")
print(f"  Uniform:     {len(uniform_examples):6d} ({len(uniform_examples)/len(dataset)*100:.1f}%)")
print(f"  Non-uniform: {len(nonuniform_examples):6d} ({len(nonuniform_examples)/len(dataset)*100:.1f}%)")

# Sample equal amounts
target_size = min(len(uniform_examples), len(nonuniform_examples))
print(f"\nTarget size per category: {target_size}")

np.random.seed(42)
uniform_sampled = np.random.choice(len(uniform_examples), target_size, replace=False)
nonuniform_sampled = np.random.choice(len(nonuniform_examples), target_size, replace=False)

balanced_dataset = []
for idx in uniform_sampled:
    balanced_dataset.append(uniform_examples[idx])
for idx in nonuniform_sampled:
    balanced_dataset.append(nonuniform_examples[idx])

np.random.shuffle(balanced_dataset)

print(f"\nAfter balancing: {len(balanced_dataset)} samples (50% uniform, 50% non-uniform)")

# Check balanced action distribution
balanced_actions = [d['action_taken'] for d in balanced_dataset]
balanced_counts = Counter(balanced_actions)
print(f"\nBalanced action distribution:")
for action, count in sorted(balanced_counts.items()):
    print(f"  {action:10s}: {count:6d} ({count/len(balanced_dataset)*100:.1f}%)")

# Save dataset
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
with open(DATA_PATH, 'wb') as f:
    pickle.dump(balanced_dataset, f)

print(f"Saved {len(balanced_dataset)} samples to {DATA_PATH}")
