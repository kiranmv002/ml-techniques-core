# Model Interpretability
# Author: M V Kiran
# github.com/kiranmv002

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

# load dataset
df = pd.read_csv("data/employee.csv")

print("Dataset loaded!")
print("\nFirst 5 rows:")
print(df.head())
print("\nShape:", df.shape)
print("\nClass distribution:")
print(df["left"].value_counts())
print("0 = Stayed, 1 = Left company")


# ── Step 1: Prepare Data ─────────────────────
feature_names = ["age", "salary", "years_at_company",
                 "work_hours", "promotions",
                 "satisfaction", "projects"]

X = df[feature_names].values
y = df["left"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining samples:", len(X_train))
print("Test samples    :", len(X_test))


# ── Step 2: Train 3 Models ───────────────────
print("\n--- Training 3 Models ---")

# logistic regression - white box
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test_scaled))
print("Logistic Regression accuracy:", round(lr_acc * 100, 2), "%")

# decision tree - white box
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt.predict(X_test))
print("Decision Tree accuracy      :", round(dt_acc * 100, 2), "%")

# random forest - less interpretable
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
print("Random Forest accuracy      :", round(rf_acc * 100, 2), "%")


# ── Step 3: Logistic Regression Coefficients ─
print("\n--- Logistic Regression Coefficients ---")
print("Positive = increases chance of leaving")
print("Negative = decreases chance of leaving")
print()
for name, coef in zip(feature_names, lr.coef_[0]):
    direction = "↑ increases risk" if coef > 0 else "↓ decreases risk"
    print(f"  {name:20}: {round(coef, 3):7} {direction}")


# ── Step 4: Decision Tree Rules ──────────────
print("\n--- Decision Tree Rules ---")
rules = export_text(dt, feature_names=feature_names)
print(rules)


# ── Step 5: Feature Importance (Tree) ────────
print("\n--- Feature Importance (Decision Tree) ---")
dt_importance = dt.feature_importances_
for name, imp in zip(feature_names, dt_importance):
    bar = "█" * int(imp * 50)
    print(f"  {name:20}: {round(imp, 3)} {bar}")


# ── Step 6: Feature Importance (Random Forest)
print("\n--- Feature Importance (Random Forest) ---")
rf_importance = rf.feature_importances_
for name, imp in zip(feature_names, rf_importance):
    bar = "█" * int(imp * 50)
    print(f"  {name:20}: {round(imp, 3)} {bar}")


# ── Step 7: Explain One Prediction ───────────
print("\n--- Explaining One Prediction ---")

# pick one employee from test set
sample_idx = 0
sample = X_test[sample_idx:sample_idx+1]
sample_scaled = X_test_scaled[sample_idx:sample_idx+1]

actual = y_test[sample_idx]
dt_pred = dt.predict(sample)[0]
lr_pred = lr.predict(sample_scaled)[0]

print("Employee details:")
for name, val in zip(feature_names, sample[0]):
    print(f"  {name:20}: {val}")

print(f"\nActual outcome : {'Left ❌' if actual == 1 else 'Stayed ✅'}")
print(f"DT prediction  : {'Left ❌' if dt_pred == 1 else 'Stayed ✅'}")
print(f"LR prediction  : {'Left ❌' if lr_pred == 1 else 'Stayed ✅'}")

# top reasons based on feature importance
top_features = np.argsort(rf_importance)[::-1][:3]
print("\nTop 3 reasons for prediction:")
for rank, idx in enumerate(top_features, 1):
    print(f"  {rank}. {feature_names[idx]} = {sample[0][idx]}")
