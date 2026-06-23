# Logistic Regression
# Author: M V Kiran
# github.com/kiranmv002

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score,
                              confusion_matrix, classification_report)
from sklearn.preprocessing import MinMaxScaler

# load dataset
df = pd.read_csv("data/diabetes.csv")

print("Dataset loaded!")
print("\nFirst 5 rows:")
print(df.head())
print("\nShape:", df.shape)
print("\nClass distribution:")
print(df["diabetic"].value_counts())
print("0 = Not Diabetic, 1 = Diabetic")


# ── Step 1: Prepare Data ─────────────────────
feature_names = ["glucose", "bmi", "age", "blood_pressure",
                 "insulin", "skin_thickness", "pregnancies"]

X = df[feature_names].values
y = df["diabetic"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining samples:", len(X_train))
print("Test samples    :", len(X_test))


# ── Step 2: Train Model ──────────────────────
print("\n--- Training Logistic Regression ---")

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

print("Model trained!")
