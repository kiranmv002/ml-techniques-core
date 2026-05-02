# Learning Paradigms in Machine Learning
# Author: M V Kiran
# github.com/kiranmv002

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# load dataset
df = pd.read_csv("data/learning_paradigms.csv")

print("Dataset loaded!")
print("\nFirst 5 rows:")
print(df.head())
print("\nShape:", df.shape)
print("\nMissing values:", df.isnull().sum().sum())


# features and target
X = df[["study_hours", "attendance", "previous_score"]].values
y = df["passed"].values

print("\nFeatures shape:", X.shape)
print("Target values:", np.unique(y), "→ 0 = failed, 1 = passed")


# ── 1. Supervised Learning ───────────────────
# giving model both X and y (with labels)
# like studying with an answer key

print("\n--- Supervised Learning ---")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Trained Decision Tree with labels")
print("Accuracy:", round(accuracy * 100, 2), "%")

# sample prediction
sample = [[6, 70, 60]]
result = model.predict(sample)
print("Student with 6 study hours, 70% attendance, 60 previous score:")
print("Prediction:", "Passed ✅" if result[0] == 1 else "Failed ❌")


# ── 2. Unsupervised Learning ─────────────────
# giving only X, no labels
# model groups students on its own

print("\n--- Unsupervised Learning ---")

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X)

print("Trained KMeans without labels")
print("Groups found:", np.unique(kmeans.labels_))
print("First 10 assignments:", kmeans.labels_[:10])


# ── 3. Reinforcement Learning ────────────────
# simple trial and error idea
# student tries different study hours to pass

print("\n--- Reinforcement Learning (simple idea) ---")

total_reward = 0
study_plan = [2, 4, 6, 8, 10]

for hours in study_plan:
    if hours >= 6:
        total_reward += 1
        print(f"Studied {hours} hrs → Passed ✅ reward: +1")
    else:
        total_reward -= 1
        print(f"Studied {hours} hrs → Failed ❌ reward: -1")

print("Total reward:", total_reward)
print("Student learns: study at least 6 hours to pass!")


# ── Comparison Plot ───────────────────────────
plt.figure(figsize=(8, 4))

# supervised - scatter plot
plt.subplot(1, 2, 1)
colors = ["red" if p == 0 else "green" for p in y]
plt.scatter(df["study_hours"], df["attendance"], c=colors, alpha=0.6)
plt.xlabel("Study Hours")
plt.ylabel("Attendance")
plt.title("Supervised Learning\n(Red=Failed, Green=Passed)")

# unsupervised - clusters
plt.subplot(1, 2, 2)
plt.scatter(df["study_hours"], df["attendance"],
            c=kmeans.labels_, cmap="Set1", alpha=0.6)
plt.xlabel("Study Hours")
plt.ylabel("Attendance")
plt.title("Unsupervised Learning\n(Groups found by model)")

plt.tight_layout()
plt.savefig("learning_paradigms.png")
plt.show()
print("\nPlot saved!")


print("""
==============================
KEY TAKEAWAYS
==============================
- Supervised   : Needs labels, learns from examples
- Unsupervised : No labels, finds patterns itself
- Reinforcement: Trial and error, reward based
- Dataset      : Student study data, 50 samples
==============================
""")
