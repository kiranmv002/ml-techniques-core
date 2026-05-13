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
