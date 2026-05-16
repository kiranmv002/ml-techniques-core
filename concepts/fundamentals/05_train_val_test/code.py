# Train Validation Test Split
# Author: M V Kiran
# github.com/kiranmv002

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# load dataset
df = pd.read_csv("data/exam_scores.csv")

print("Dataset loaded!")
print("\nFirst 5 rows:")
print(df.head())
print("\nShape:", df.shape)


# features and target
X = df[["study_hours", "sleep_hours", "attendance",
        "previous_score", "stress_level"]].values
y = df["final_score"].values

print("\nTotal samples:", len(X))


# ── Step 1: Basic Train Test Split ───────────
# splitting into 80% train and 20% test

print("\n--- Basic Train Test Split (80/20) ---")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Test samples    :", len(X_test))


# ── Step 2: Train Val Test Split ─────────────
# splitting into 70% train, 15% val, 15% test

print("\n--- Train Val Test Split (70/15/15) ---")

# first split off test set
X_temp, X_test2, y_temp, y_test2 = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# then split remaining into train and val
X_train2, X_val, y_train2, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42
)

print("Training samples  :", len(X_train2))
print("Validation samples:", len(X_val))
print("Test samples      :", len(X_test2))


# ── Step 3: Train and Evaluate ───────────────
print("\n--- Training and Evaluation ---")

model = LinearRegression()
model.fit(X_train2, y_train2)

# validation performance
val_preds = model.predict(X_val)
val_mae = mean_absolute_error(y_val, val_preds)
val_r2 = r2_score(y_val, val_preds)

print("Validation MAE:", round(val_mae, 2))
print("Validation R2 :", round(val_r2, 2))

# test performance (only at the very end)
test_preds = model.predict(X_test2)
test_mae = mean_absolute_error(y_test2, test_preds)
test_r2 = r2_score(y_test2, test_preds)

print("\nTest MAE:", round(test_mae, 2))
print("Test R2 :", round(test_r2, 2))


# ── Step 4: Cross Validation ─────────────────
# splits data multiple ways and tests each time
# gives more reliable performance estimate

print("\n--- Cross Validation (5 Fold) ---")

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(LinearRegression(), X, y,
                             cv=kfold, scoring="r2")

print("R2 scores for each fold:", np.round(cv_scores, 2))
print("Average R2            :", round(cv_scores.mean(), 2))
print("Standard deviation    :", round(cv_scores.std(), 2))


# ── Visualization ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# data split visualization
sizes = [len(X_train2), len(X_val), len(X_test2)]
labels = ["Train\n70%", "Validation\n15%", "Test\n15%"]
colors = ["steelblue", "seagreen", "tomato"]
axes[0].pie(sizes, labels=labels, colors=colors,
            autopct="%1.0f%%", startangle=90)
axes[0].set_title("Data Split")

# actual vs predicted on test set
axes[1].scatter(y_test2, test_preds, color="steelblue", alpha=0.7)
axes[1].plot([y_test2.min(), y_test2.max()],
             [y_test2.min(), y_test2.max()],
             color="red", linestyle="--")
axes[1].set_xlabel("Actual Score")
axes[1].set_ylabel("Predicted Score")
axes[1].set_title("Actual vs Predicted\n(Test Set)")

# cross validation scores
axes[2].bar([f"Fold {i+1}" for i in range(5)],
            cv_scores, color="purple", alpha=0.7)
axes[2].axhline(y=cv_scores.mean(), color="red",
                linestyle="--", label="Average")
axes[2].set_xlabel("Fold")
axes[2].set_ylabel("R2 Score")
axes[2].set_title("Cross Validation Scores")
axes[2].legend()

plt.tight_layout()
plt.savefig("train_val_test.png")
plt.show()
print("Plot saved!")


print("""
==============================
KEY TAKEAWAYS
==============================
- Train set     : Model learns from this (70%)
- Validation    : Tune model during training (15%)
- Test set      : Final check only at the end (15%)
- Cross val     : More reliable than single split
- Never use test set during training!
==============================
""")
