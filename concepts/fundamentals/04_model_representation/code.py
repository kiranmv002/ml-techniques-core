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
