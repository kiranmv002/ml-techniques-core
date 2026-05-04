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

