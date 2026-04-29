# Learning Paradigms in Machine Learning
# Author: M V Kiran
# github.com/kiranmv002

# I am going to show simple examples of all 3 paradigms

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# load the iris dataset
# this is a simple flower dataset with 3 types of flowers
data = load_iris()
X = data.data
y = data.target

print("Dataset loaded")
print("Total samples:", len(X))
print("Features:", data.feature_names)
print("Classes:", data.target_names)


# ── 1. Supervised Learning example ──────────
# I am giving the model both X (features) and y (labels)
# It learns from this and predicts new data

print("\n--- Supervised Learning ---")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("I trained a Decision Tree with labels")
print("Accuracy:", round(accuracy * 100, 2), "%")


# ── 2. Unsupervised Learning example ────────
# I am giving only X, no labels
# The model groups the data on its own

print("\n--- Unsupervised Learning ---")

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

print("I trained KMeans without any labels")
print("Groups found:", np.unique(kmeans.labels_))
print("First 10 group assignments:", kmeans.labels_[:10])


# ── 3. Reinforcement Learning ────────────────
# I cannot show a full RL example easily
# but here is the basic idea in simple code

print("\n--- Reinforcement Learning (simple idea) ---")

# imagine an agent trying to move right on a number line
# it gets +1 reward for moving right, -1 for moving left

position = 0
total_reward = 0

actions = ["right", "right", "left", "right", "right"]

for action in actions:
    if action == "right":
        position += 1
        total_reward += 1
        print(f"Moved right → position: {position}, reward: +1")
    else:
        position -= 1
        total_reward -= 1
        print(f"Moved left  → position: {position}, reward: -1")

print("Final position:", position)
print("Total reward:", total_reward)
print("Agent learns to always move right to get more reward!")


# ── Simple comparison plot ───────────────────
labels = ["Supervised\n(with labels)", "Unsupervised\n(no labels)", "Reinforcement\n(trial & error)"]
values = [3, 2, 1]
colors = ["steelblue", "seagreen", "tomato"]

plt.figure(figsize=(7, 4))
plt.bar(labels, values, color=colors)
plt.title("3 Learning Paradigms")
plt.ylabel("Complexity Level")
plt.tight_layout()
plt.savefig("learning_paradigms.png")
plt.show()

print("\nDone! learning_paradigms.png saved")
