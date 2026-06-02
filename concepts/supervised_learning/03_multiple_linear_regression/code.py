# Multiple Linear Regression
# Author: M V Kiran
# github.com/kiranmv002

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# load dataset
df = pd.read_csv("data/car_prices.csv")

print("Dataset loaded!")
print("\nFirst 5 rows:")
print(df.head())
print("\nShape:", df.shape)


# ── Step 1: Explore Features ─────────────────
print("\n--- Feature Overview ---")
print(df.describe())

print("\nCorrelation with price:")
correlations = df.corr()["price"].sort_values(ascending=False)
print(correlations)


# ── Step 2: Prepare Data ─────────────────────
feature_names = ["mileage_km", "age_years", "engine_cc",
                 "owners", "brand_encoded", "service_history"]

X = df[feature_names].values
y = df["price"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", len(X_train))
print("Test samples    :", len(X_test))


# ── Step 3: Train Without Scaling ────────────
print("\n--- Without Scaling ---")

model_no_scale = LinearRegression()
model_no_scale.fit(X_train, y_train)
preds_no_scale = model_no_scale.predict(X_test)

mae_no_scale = mean_absolute_error(y_test, preds_no_scale)
r2_no_scale = r2_score(y_test, preds_no_scale)

print("MAE:", round(mae_no_scale, 2))
print("R2 :", round(r2_no_scale, 2))


# ── Step 4: Train With Scaling ────────────────
# scaling makes weights more comparable
# and helps model learn better

print("\n--- With Scaling ---")

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = LinearRegression()
model_scaled.fit(X_train_scaled, y_train)
preds_scaled = model_scaled.predict(X_test_scaled)

mae_scaled = mean_absolute_error(y_test, preds_scaled)
r2_scaled = r2_score(y_test, preds_scaled)

print("MAE:", round(mae_scaled, 2))
print("R2 :", round(r2_scaled, 2))


# ── Step 5: Feature Weights ───────────────────
print("\n--- Feature Weights (after scaling) ---")
print("Higher weight = more important feature")
for name, weight in zip(feature_names, model_scaled.coef_):
    print(f"  {name:20}: {round(weight, 2)}")
print(f"  {'intercept':20}: {round(model_scaled.intercept_, 2)}")


