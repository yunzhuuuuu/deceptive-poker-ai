import pickle
from collections import Counter

path = './data/cfr_dataset_5000eps.pkl'

with open(path, "rb") as f:
    dataset = pickle.load(f)

print("size:", len(dataset))

# Count actions
actions = [d['action_taken'] for d in dataset]
print(Counter(actions))

print("\nfirst sample:")
print(dataset[0].keys())

print("features shape:", dataset[0]["features"].shape)
print("action_probs:", dataset[0]["action_probs"])
print("sum probs:", sum(dataset[0]["action_probs"]))
print("action_taken:", dataset[0]["action_taken"])

print("\naction probs:")
for i in range(50):
    print(i, dataset[i]["action_probs"])


