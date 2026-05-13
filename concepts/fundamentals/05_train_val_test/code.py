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
