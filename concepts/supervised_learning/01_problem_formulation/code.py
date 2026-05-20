# Problem Formulation in Machine Learning
# Author: M V Kiran
# github.com/kiranmv002

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_diabetes
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# load problems dataset
df = pd.read_csv("data/problems.csv")

print("Problems dataset loaded!")
print("\nAll ML problems:")
print(df[["description", "problem_type", "metric"]].to_string())


# ── Problem Types Summary ────────────────────
print("\n--- Problem Types Count ---")
print(df["problem_type"].value_counts())


# ── Example 1: Regression Problem ───────────
# predicting a number (blood sugar level)
# input: age, bmi, blood pressure etc
# target: disease progression score (a number)

print("\n--- Example 1: Regression Problem ---")
print("Problem: Predict disease progression score")
print("Input  : Age, BMI, Blood pressure etc")
print("Target : A number (progression score)")
print("Metric : MAE and R2")

diabetes = load_diabetes()
X_reg = diabetes.data
y_reg = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
reg_preds = reg_model.predict(X_test)

print("\nRegression Results:")
print("MAE:", round(mean_absolute_error(y_test, reg_preds), 2))
print("R2 :", round(r2_score(y_test, reg_preds), 2))


# ── Example 2: Classification Problem ────────
# predicting a category (flower type)
# input: sepal and petal measurements
# target: setosa versicolor or virginica

print("\n--- Example 2: Classification Problem ---")
print("Problem: Predict flower species")
print("Input  : Sepal length, width, petal length, width")
print("Target : A category (flower type)")
print("Metric : Accuracy")

iris = load_iris()
X_clf = iris.data
y_clf = iris.target

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

clf_model = LogisticRegression(max_iter=200)
clf_model.fit(X_train2, y_train2)
clf_preds = clf_model.predict(X_test2)

print("\nClassification Results:")
print("Accuracy:", round(accuracy_score(y_test2, clf_preds) * 100, 2), "%")


# ── Problem Formulation Template ─────────────
print("\n--- Problem Formulation Template ---")

template = {
    "What to predict"  : "Disease progression score",
    "Input features"   : "Age, BMI, Blood pressure",
    "Problem type"     : "Regression",
    "Success metric"   : "MAE < 50, R2 > 0.5",
    "Data available"   : "442 patient records",
    "Constraints"      : "Model must be explainable"
}

for key, value in template.items():
    print(f"{key:20}: {value}")


# ── Visualization ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# problem types distribution
problem_counts = df["problem_type"].value_counts()
axes[0].bar(problem_counts.index, problem_counts.values,
            color=["steelblue", "seagreen", "tomato"])
axes[0].set_title("Problem Types")
axes[0].set_xlabel("Type")
axes[0].set_ylabel("Count")

# regression actual vs predicted
