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


# ── Step 4: Handle Missing Values ────────────
print("\n--- Handling Missing Values ---")

# fill missing age with median
df["age"] = df["age"].fillna(df["age"].median())
print("Missing age filled with median:", df["age"].median())

# fill missing gender with most common value
most_common_gender = df["gender"].mode()[0]
df["gender"] = df["gender"].fillna(most_common_gender)
print("Missing gender filled with:", most_common_gender)

print("\nMissing values after fixing:")
print(df.isnull().sum())


# ── Step 5: Encode Categorical Data ──────────
print("\n--- Encoding Categorical Data ---")

le = LabelEncoder()
df["gender_encoded"] = le.fit_transform(df["gender"])
print("Gender encoding:", dict(zip(le.classes_, le.transform(le.classes_))))


# ── Step 6: Scale Features ───────────────────
print("\n--- Scaling Features ---")

# min max scaling - brings to 0 to 1 range
minmax = MinMaxScaler()
df["age_minmax"] = minmax.fit_transform(df[["age"]])
df["bp_minmax"] = minmax.fit_transform(df[["blood_pressure"]])

# standard scaling - mean 0 std 1
standard = StandardScaler()
df["age_standard"] = standard.fit_transform(df[["age"]])
df["bp_standard"] = standard.fit_transform(df[["blood_pressure"]])

print("Age min-max range:",
      round(df["age_minmax"].min(), 2), "to",
      round(df["age_minmax"].max(), 2))
print("Age standard mean:",
      round(df["age_standard"].mean(), 2),
      "std:", round(df["age_standard"].std(), 2))


