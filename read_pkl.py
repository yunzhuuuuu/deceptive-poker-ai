import pickle

path = './cfr_dataset.pkl'

with open(path, "rb") as f:
    dataset = pickle.load(f)

print("size:", len(dataset))

print("\nfirst sample:")
print(dataset[0].keys())

print("features shape:", dataset[0]["features"].shape)
print("action_probs:", dataset[0]["action_probs"])
print("sum probs:", sum(dataset[0]["action_probs"]))
print("action_taken:", dataset[0]["action_taken"])

print("\naction probs:")
for i in range(10):
    print(i, dataset[i]["action_probs"])
