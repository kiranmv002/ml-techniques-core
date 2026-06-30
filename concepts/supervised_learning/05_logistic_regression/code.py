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


# ── Step 3: Predictions ──────────────────────
print("\n--- Predictions ---")

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("First 10 predictions:")
for i in range(10):
    print(f"  Patient {i+1}: probability={round(y_prob[i], 2)}"
          f"  predicted={'Diabetic' if y_pred[i]==1 else 'Not Diabetic'}"
          f"  actual={'Diabetic' if y_test[i]==1 else 'Not Diabetic'}")


# ── Step 4: Evaluate Model ───────────────────
print("\n--- Model Evaluation ---")

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)

print(f"Accuracy : {round(accuracy  * 100, 2)}%")
print(f"Precision: {round(precision * 100, 2)}%")
print(f"Recall   : {round(recall    * 100, 2)}%")
print(f"F1 Score : {round(f1        * 100, 2)}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Not Diabetic", "Diabetic"]))


