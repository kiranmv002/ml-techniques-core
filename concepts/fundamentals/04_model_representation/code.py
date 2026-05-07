# Model Representation in Machine Learning
# Author: M V Kiran
# github.com/kiranmv002

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_text
from sklearn.preprocessing import LabelEncoder

# load dataset
df = pd.read_csv("data/house_prices.csv")

print("Dataset loaded!")
print("\nFirst 5 rows:")
print(df.head())
print("\nShape:", df.shape)


# encode location column (city=0, suburb=1)
le = LabelEncoder()
df["location_encoded"] = le.fit_transform(df["location"])
print("\nLocation encoding:", dict(zip(le.classes_, le.transform(le.classes_))))


# features and target
X = df[["area", "rooms", "age", "location_encoded"]].values
y = df["price"].values

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)


# ── 1. Linear Model Representation ──────────
# model learns as an equation
# price = w1*area + w2*rooms + w3*age + w4*location + bias

print("\n--- Linear Model ---")

linear_model = LinearRegression()
linear_model.fit(X, y)

print("Model learned these weights (parameters):")
feature_names = ["area", "rooms", "age", "location"]
for name, weight in zip(feature_names, linear_model.coef_):
    print(f"  {name}: {round(weight, 2)}")
print("Bias (intercept):", round(linear_model.intercept_, 2))

# what the equation looks like
print("\nModel equation:")
print(f"Price = {round(linear_model.coef_[0],2)} x area"
      f" + {round(linear_model.coef_[1],2)} x rooms"
      f" + {round(linear_model.coef_[2],2)} x age"
      f" + {round(linear_model.coef_[3],2)} x location"
      f" + {round(linear_model.intercept_,2)}")

# sample prediction
sample = np.array([[1000, 3, 5, 0]])  # city house
predicted = linear_model.predict(sample)
print("\nPrediction for 1000sqft, 3 rooms, 5yr old, city house:")
print("Predicted price:", round(predicted[0], 2))


# ── 2. Decision Tree Representation ─────────
# model learns as a series of yes/no questions
# stores as a tree structure

print("\n--- Decision Tree Model ---")

tree_model = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_model.fit(X, y)

print("Decision Tree structure (first 3 levels):")
tree_rules = export_text(tree_model, feature_names=feature_names)
print(tree_rules)

# sample prediction
predicted_tree = tree_model.predict(sample)
print("Prediction for same house using Decision Tree:")
print("Predicted price:", round(predicted_tree[0], 2))


# ── Visualization ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# linear model - actual vs predicted
linear_preds = linear_model.predict(X)
axes[0].scatter(y, linear_preds, color="steelblue", alpha=0.6)
axes[0].plot([y.min(), y.max()], [y.min(), y.max()],
             color="red", linestyle="--")
axes[0].set_xlabel("Actual Price")
axes[0].set_ylabel("Predicted Price")
axes[0].set_title("Linear Model\nActual vs Predicted")

# feature importance from tree
importances = tree_model.feature_importances_
axes[1].bar(feature_names, importances, color="seagreen")
axes[1].set_xlabel("Features")
axes[1].set_ylabel("Importance")
axes[1].set_title("Decision Tree\nFeature Importance")

plt.tight_layout()
plt.savefig("model_representation.png")
plt.show()
print("Plot saved!")


print("""
==============================
KEY TAKEAWAYS
==============================
- Linear model  : Stores learning as equation
- Decision tree : Stores learning as yes/no questions
- Parameters    : Values the model learns (weights)
- Bias          : Base value before any feature
- Area has the highest importance for house price
==============================
""")
