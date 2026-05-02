# Data Representation in Machine Learning
# Author: M V Kiran
# github.com/kiranmv002

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

# load dataset
df = pd.read_csv("data/students.csv")

print("Dataset loaded!")
print("\nFirst 5 rows:")
print(df.head())
print("\nShape:", df.shape)
print("\nColumn types:")
print(df.dtypes)


# ── Step 1: Identify Data Types ──────────────
print("\n--- Numerical Columns ---")
numerical = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
print(numerical)

print("\n--- Categorical Columns ---")
categorical = df.select_dtypes(include=["object"]).columns.tolist()
print(categorical)


# ── Step 2: Label Encoding ───────────────────
# converting gender and grade to numbers
# simple when category has only 2-3 values

print("\n--- Label Encoding ---")

le = LabelEncoder()

df["gender_encoded"] = le.fit_transform(df["gender"])
print("Gender encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

df["grade_encoded"] = le.fit_transform(df["grade"])
print("Grade encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

print("\nAfter label encoding:")
print(df[["gender", "gender_encoded", "grade", "grade_encoded"]].head(6))


# ── Step 3: One Hot Encoding ─────────────────
# converting city to 0s and 1s
# better when category has many values with no order

print("\n--- One Hot Encoding ---")

city_dummies = pd.get_dummies(df["city"], prefix="city")
print("City columns created:", city_dummies.columns.tolist())
print(city_dummies.head(6))

# add to dataframe
df = pd.concat([df, city_dummies], axis=1)
