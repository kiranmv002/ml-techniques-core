# Linear Regression
# Author: M V Kiran
# github.com/kiranmv002

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# load dataset
df = pd.read_csv("data/house_sales.csv")

print("Dataset loaded!")
print("\nFirst 5 rows:")
print(df.head())
print("\nShape:", df.shape)


# ── Step 1: Simple Linear Regression ────────
# using only one feature (size) to predict price
# this is the simplest form

print("\n--- Simple Linear Regression ---")
print("Using only house size to predict price")

X_simple = df[["size_sqft"]].values
y = df["price"].values

X_train, X_test, y_train, y_test = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)

simple_model = LinearRegression()
simple_model.fit(X_train, y_train)

print("\nModel learned:")
print("Slope (m)    :", round(simple_model.coef_[0], 2))
print("Intercept (c):", round(simple_model.intercept_, 2))
print("\nEquation: Price =",
      round(simple_model.coef_[0], 2), "x size +",
      round(simple_model.intercept_, 2))

simple_preds = simple_model.predict(X_test)
print("\nSimple Model Results:")
print("MAE :", round(mean_absolute_error(y_test, simple_preds), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, simple_preds)), 2))
print("R2  :", round(r2_score(y_test, simple_preds), 2))


# ── Step 2: Multiple Linear Regression ──────
# using all features to predict price
# usually gives better results

print("\n--- Multiple Linear Regression ---")
print("Using all features to predict price")

X_multi = df[["size_sqft", "bedrooms", "bathrooms",
              "age_years", "distance_km"]].values

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_multi, y, test_size=0.2, random_state=42
)

multi_model = LinearRegression()
multi_model.fit(X_train2, y_train2)

feature_names = ["size_sqft", "bedrooms", "bathrooms",
                 "age_years", "distance_km"]
print("\nModel learned weights:")
for name, coef in zip(feature_names, multi_model.coef_):
    print(f"  {name:15}: {round(coef, 2)}")
print(f"  {'intercept':15}: {round(multi_model.intercept_, 2)}")

multi_preds = multi_model.predict(X_test2)
print("\nMultiple Model Results:")
print("MAE :", round(mean_absolute_error(y_test2, multi_preds), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test2, multi_preds)), 2))
print("R2  :", round(r2_score(y_test2, multi_preds), 2))


# ── Step 3: Make a Prediction ────────────────
print("\n--- Sample Predictions ---")

# predict price for a new house
new_house = pd.DataFrame({
    "size_sqft"  : [1000],
    "bedrooms"   : [3],
    "bathrooms"  : [2],
    "age_years"  : [5],
    "distance_km": [10]
})

predicted_price = multi_model.predict(new_house)
print("House: 1000 sqft, 3 bed, 2 bath, 5yr old, 10km from city")
print("Predicted price: Rs", round(predicted_price[0], 2))


# ── Visualization ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# simple regression line
axes[0].scatter(df["size_sqft"], df["price"],
                color="steelblue", alpha=0.6, label="Actual")
x_line = np.linspace(df["size_sqft"].min(),
                     df["size_sqft"].max(), 100).reshape(-1, 1)
y_line = simple_model.predict(x_line)
axes[0].plot(x_line, y_line, color="red",
             linewidth=2, label="Regression line")
axes[0].set_xlabel("Size (sqft)")
axes[0].set_ylabel("Price")
axes[0].set_title("Simple Linear Regression")
axes[0].legend()

# actual vs predicted
axes[1].scatter(y_test2, multi_preds,
                color="seagreen", alpha=0.6)
axes[1].plot([y_test2.min(), y_test2.max()],
             [y_test2.min(), y_test2.max()],
             color="red", linestyle="--")
axes[1].set_xlabel("Actual Price")
axes[1].set_ylabel("Predicted Price")
axes[1].set_title("Multiple Linear Regression\nActual vs Predicted")

# feature weights
axes[2].barh(feature_names, multi_model.coef_,
             color="purple", alpha=0.7)
axes[2].set_xlabel("Weight")
axes[2].set_title("Feature Weights")
axes[2].axvline(x=0, color="red", linestyle="--")

plt.tight_layout()
plt.savefig("linear_regression.png")
plt.show()
print("\nPlot saved!")


print("""
==============================
KEY TAKEAWAYS
==============================
- Linear regression finds the best fit line
- Simple    : one feature → one target
- Multiple  : many features → one target
- Slope     : how much target changes per unit
- R2 = 1.0  : perfect fit
- R2 = 0.0  : model learned nothing
- Always check MAE and R2 together
==============================
""")
