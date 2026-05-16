# Data Preprocessing in Machine Learning
# Author: M V Kiran
# github.com/kiranmv002

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

# load raw dataset
df = pd.read_csv("data/raw_patients.csv")

print("Raw dataset loaded!")
print("\nFirst 5 rows:")
print(df.head())
print("\nShape:", df.shape)


# ── Step 1: Check Data Problems ──────────────
print("\n--- Checking Data Problems ---")

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nBasic stats (check for outliers):")
print(df.describe())


# ── Step 2: Remove Duplicates ────────────────
print("\n--- Removing Duplicates ---")

before = len(df)
df = df.drop_duplicates()
after = len(df)

print(f"Removed {before - after} duplicate rows")
print("Rows remaining:", len(df))


# ── Step 3: Handle Outliers ──────────────────
# age 999 and age 0 are clearly wrong
# replacing with median age

print("\n--- Handling Outliers ---")

print("Age values before:", sorted(df["age"].dropna().unique()))

median_age = df[(df["age"] > 0) & (df["age"] < 120)]["age"].median()
df["age"] = df["age"].apply(
    lambda x: median_age if (x > 120 or x == 0) else x
)

print("Outliers replaced with median age:", median_age)
