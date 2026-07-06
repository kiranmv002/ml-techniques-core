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


# ── Step 5: Feature Importance ───────────────
print("\n--- Feature Importance ---")
for name, coef in zip(feature_names, model.coef_[0]):
    print(f"  {name:20}: {round(coef, 3)}")


# ── Step 6: Sample Prediction ────────────────
print("\n--- Sample Prediction ---")

new_patient = pd.DataFrame({
    "glucose"        : [148],
    "bmi"            : [33.6],
    "age"            : [50],
    "blood_pressure" : [72],
    "insulin"        : [0],
    "skin_thickness" : [35],
    "pregnancies"    : [6]
})

new_scaled = scaler.transform(new_patient)
prob = model.predict_proba(new_scaled)[0][1]
pred = model.predict(new_scaled)[0]

print("Patient: glucose=148, bmi=33.6, age=50")
print(f"Probability of diabetes: {round(prob * 100, 2)}%")
print(f"Prediction: {'Diabetic ⚠️' if pred == 1 else 'Not Diabetic ✅'}")


# ── Visualization ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
im = axes[0].imshow(cm, cmap="Blues")
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(["Not Diabetic", "Diabetic"])
axes[0].set_yticklabels(["Not Diabetic", "Diabetic"])
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
axes[0].set_title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, cm[i, j], ha="center",
                     va="center", fontsize=14, color="black")

# probability distribution
diabetic_probs = y_prob[y_test == 1]
not_diabetic_probs = y_prob[y_test == 0]
axes[1].hist(not_diabetic_probs, bins=8, alpha=0.6,
             color="steelblue", label="Not Diabetic")
axes[1].hist(diabetic_probs, bins=8, alpha=0.6,
             color="tomato", label="Diabetic")
axes[1].axvline(x=0.5, color="black",
                linestyle="--", label="Threshold 0.5")
axes[1].set_xlabel("Predicted Probability")
axes[1].set_ylabel("Count")
axes[1].set_title("Probability Distribution")
axes[1].legend()

# feature importance
coefs = model.coef_[0]
colors = ["seagreen" if c > 0 else "tomato" for c in coefs]
axes[2].barh(feature_names, coefs, color=colors)
axes[2].axvline(x=0, color="black", linestyle="--")
axes[2].set_xlabel("Coefficient")
axes[2].set_title("Feature Importance\n(Green=Increases Risk)")

plt.tight_layout()
plt.savefig("logistic_regression.png")
plt.show()
print("\nPlot saved!")


print("""
==============================
KEY TAKEAWAYS
==============================
- Logistic regression predicts probability
- Output is always between 0 and 1
- Default threshold is 0.5
- Accuracy  : overall correctness
- Precision : when we say yes how often right
- Recall    : how many actual yes did we catch
- F1 Score  : balance of precision and recall
- Glucose and BMI are top risk factors
==============================
""")
