# Decision Tree
# Author: M V Kiran
# github.com/kiranmv002

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix)
from sklearn.preprocessing import MinMaxScaler

# load dataset
df = pd.read_csv("data/loan_approval.csv")

print("Dataset loaded!")
print("\nFirst 5 rows:")
print(df.head())
print("\nShape:", df.shape)
print("\nClass distribution:")
print(df["approved"].value_counts())
print("0 = Rejected, 1 = Approved")


# ── Step 1: Prepare Data ─────────────────────
feature_names = ["age", "income", "credit_score",
                 "employment_years", "existing_loans", "loan_amount"]

X = df[feature_names].values
y = df["approved"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", len(X_train))
print("Test samples    :", len(X_test))


# ── Step 2: Train Simple Tree (depth 3) ──────
print("\n--- Decision Tree (depth 3) ---")

simple_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
simple_tree.fit(X_train, y_train)

simple_preds = simple_tree.predict(X_test)
simple_acc = accuracy_score(y_test, simple_preds)

print("Accuracy:", round(simple_acc * 100, 2), "%")
print("\nTree structure:")
print(export_text(simple_tree, feature_names=feature_names))


# ── Step 3: Train Deep Tree (no limit) ───────
print("\n--- Decision Tree (no depth limit) ---")

deep_tree = DecisionTreeClassifier(max_depth=None, random_state=42)
deep_tree.fit(X_train, y_train)

train_acc = accuracy_score(y_train, deep_tree.predict(X_train))
test_acc = accuracy_score(y_test, deep_tree.predict(X_test))

print("Train accuracy:", round(train_acc * 100, 2), "%")
print("Test  accuracy:", round(test_acc  * 100, 2), "%")
print("Gap shows overfitting if train >> test")


# ── Step 4: Find Best Depth ───────────────────
print("\n--- Finding Best Tree Depth ---")

depths = range(1, 12)
train_accs = []
test_accs = []

for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    train_accs.append(accuracy_score(y_train, dt.predict(X_train)))
    test_accs.append(accuracy_score(y_test, dt.predict(X_test)))

best_depth = depths[np.argmax(test_accs)]
print("Best depth:", best_depth)
print("Best test accuracy:", round(max(test_accs) * 100, 2), "%")


# ── Step 5: Best Model Evaluation ────────────
print("\n--- Best Model Evaluation ---")

best_tree = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
best_tree.fit(X_train, y_train)
best_preds = best_tree.predict(X_test)

print(classification_report(y_test, best_preds,
      target_names=["Rejected", "Approved"]))


# ── Step 6: Feature Importance ───────────────
print("\n--- Feature Importance ---")
importances = best_tree.feature_importances_
for name, imp in zip(feature_names, importances):
    print(f"  {name:20}: {round(imp, 3)}")


# ── Step 7: Sample Prediction ────────────────
print("\n--- Sample Prediction ---")

new_applicant = pd.DataFrame({
    "age"              : [35],
    "income"           : [65000],
    "credit_score"     : [740],
    "employment_years" : [8],
    "existing_loans"   : [0],
    "loan_amount"      : [250000]
})

pred = best_tree.predict(new_applicant)[0]
prob = best_tree.predict_proba(new_applicant)[0]

print("Applicant: age=35, income=65000, credit=740")
print(f"Approval probability: {round(prob[1] * 100, 2)}%")
print(f"Decision: {'Approved ✅' if pred == 1 else 'Rejected ❌'}")


# ── Visualization ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
