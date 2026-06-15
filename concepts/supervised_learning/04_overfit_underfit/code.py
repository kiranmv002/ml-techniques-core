# Overfitting and Underfitting
# Author: M V Kiran
# github.com/kiranmv002

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# load dataset
df = pd.read_csv("data/student_performance.csv")

print("Dataset loaded!")
print("\nFirst 5 rows:")
print(df.head())
print("\nShape:", df.shape)

# features and target
feature_names = ["study_hours", "sleep_hours", "attendance",
                 "assignments", "participation", "stress", "prev_score"]
X = df[feature_names].values
y = df["final_score"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ── Case 1: Underfitting ──────────────────────
# using only one feature and simple model
# model is too simple to learn patterns

print("\n--- Case 1: Underfitting ---")

X_under_train = X_train[:, 0:1]  # only study hours
X_under_test = X_test[:, 0:1]

underfit_model = LinearRegression()
underfit_model.fit(X_under_train, y_train)

train_preds = underfit_model.predict(X_under_train)
test_preds = underfit_model.predict(X_under_test)

train_r2 = r2_score(y_train, train_preds)
test_r2 = r2_score(y_test, test_preds)

print("Train R2:", round(train_r2, 2))
print("Test  R2:", round(test_r2, 2))
print("Gap     :", round(abs(train_r2 - test_r2), 2))
print("Status  : Model too simple → Underfitting")


# ── Case 2: Overfitting ───────────────────────
# using very deep decision tree
# model memorizes training data

print("\n--- Case 2: Overfitting ---")

overfit_model = DecisionTreeRegressor(max_depth=None, random_state=42)
overfit_model.fit(X_train, y_train)

train_preds2 = overfit_model.predict(X_train)
test_preds2 = overfit_model.predict(X_test)

train_r2_2 = r2_score(y_train, train_preds2)
test_r2_2 = r2_score(y_test, test_preds2)

print("Train R2:", round(train_r2_2, 2))
print("Test  R2:", round(test_r2_2, 2))
print("Gap     :", round(abs(train_r2_2 - test_r2_2), 2))
print("Status  : Big gap → Overfitting")


# ── Case 3: Good Fit ──────────────────────────
# using all features with linear regression
# model learns well without memorizing

print("\n--- Case 3: Good Fit ---")

good_model = LinearRegression()
good_model.fit(X_train, y_train)

train_preds3 = good_model.predict(X_train)
test_preds3 = good_model.predict(X_test)

train_r2_3 = r2_score(y_train, train_preds3)
test_r2_3 = r2_score(y_test, test_preds3)

print("Train R2:", round(train_r2_3, 2))
print("Test  R2:", round(test_r2_3, 2))
print("Gap     :", round(abs(train_r2_3 - test_r2_3), 2))
print("Status  : Small gap → Good fit")


# ── Case 4: Regularization Fix ───────────────
# ridge and lasso add penalty to reduce overfitting

print("\n--- Case 4: Regularization (Ridge) ---")

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

train_preds4 = ridge_model.predict(X_train)
test_preds4 = ridge_model.predict(X_test)

train_r2_4 = r2_score(y_train, train_preds4)
test_r2_4 = r2_score(y_test, test_preds4)

print("Train R2:", round(train_r2_4, 2))
print("Test  R2:", round(test_r2_4, 2))
print("Gap     :", round(abs(train_r2_4 - test_r2_4), 2))
print("Status  : Regularization helps balance fit")


# ── Decision Tree Depth Comparison ───────────
print("\n--- Decision Tree Depth vs R2 ---")

depths = range(1, 15)
train_scores = []
test_scores = []

for depth in depths:
    dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    train_scores.append(r2_score(y_train, dt.predict(X_train)))
    test_scores.append(r2_score(y_test, dt.predict(X_test)))

best_depth = depths[np.argmax(test_scores)]
print("Best depth for test data:", best_depth)


# ── Visualization ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# underfit overfit goodfit comparison
models = ["Underfit", "Overfit", "Good Fit", "Ridge"]
train_r2s = [train_r2, train_r2_2, train_r2_3, train_r2_4]
test_r2s = [test_r2, test_r2_2, test_r2_3, test_r2_4]

x = np.arange(len(models))
width = 0.35
axes[0].bar(x - width/2, train_r2s, width,
            label="Train R2", color="steelblue")
axes[0].bar(x + width/2, test_r2s, width,
            label="Test R2", color="tomato")
axes[0].set_xticks(x)
axes[0].set_xticklabels(models)
axes[0].set_ylabel("R2 Score")
axes[0].set_title("Train vs Test R2\nAll Cases")
axes[0].legend()

# depth vs score curve
axes[1].plot(depths, train_scores, label="Train R2",
             color="steelblue", marker="o")
axes[1].plot(depths, test_scores, label="Test R2",
             color="tomato", marker="s")
axes[1].axvline(x=best_depth, color="green",
                linestyle="--", label=f"Best depth={best_depth}")
axes[1].set_xlabel("Tree Depth")
axes[1].set_ylabel("R2 Score")
axes[1].set_title("Depth vs R2 Score\n(Finding Sweet Spot)")
axes[1].legend()

# actual vs predicted for good fit
axes[2].scatter(y_test, test_preds3,
                color="seagreen", alpha=0.7)
axes[2].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             color="red", linestyle="--")
axes[2].set_xlabel("Actual Score")
axes[2].set_ylabel("Predicted Score")
axes[2].set_title("Good Fit Model\nActual vs Predicted")

plt.tight_layout()
plt.savefig("overfit_underfit.png")
plt.show()
print("\nPlot saved!")


print("""
==============================
KEY TAKEAWAYS
==============================
- Underfit  : Low train AND test score
- Overfit   : High train, low test score
- Good fit  : High train AND test score
- Big gap between train and test = overfitting
- Fix overfit  : Simpler model or regularization
- Fix underfit : More features or complex model
- Always check both train and test performance!
==============================
""")
