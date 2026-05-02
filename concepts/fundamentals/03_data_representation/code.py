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


# ── Step 4: Scaling ──────────────────────────
# bringing all numbers to same range (0 to 1)
# so one feature doesnt dominate others

print("\n--- Scaling (MinMax) ---")

scaler = MinMaxScaler()
df["age_scaled"] = scaler.fit_transform(df[["age"]])
df["study_hours_scaled"] = scaler.fit_transform(df[["study_hours"]])

print("Age before scaling - min:", df["age"].min(), "max:", df["age"].max())
print("Age after scaling  - min:", round(df["age_scaled"].min(), 2),
      "max:", round(df["age_scaled"].max(), 2))

print("\nStudy hours before - min:", df["study_hours"].min(),
      "max:", df["study_hours"].max())
print("Study hours after  - min:", round(df["study_hours_scaled"].min(), 2),
      "max:", round(df["study_hours_scaled"].max(), 2))


# ── Step 5: Visualize ────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# gender distribution
df["gender"].value_counts().plot(kind="bar", ax=axes[0],
                                  color=["steelblue", "salmon"])
axes[0].set_title("Gender Distribution")
axes[0].set_xlabel("Gender")
axes[0].set_ylabel("Count")

# city distribution
df["city"].value_counts().plot(kind="bar", ax=axes[1], color="seagreen")
axes[1].set_title("City Distribution")
axes[1].set_xlabel("City")

# study hours distribution
axes[2].hist(df["study_hours"], bins=5, color="purple", alpha=0.7)
axes[2].set_title("Study Hours Distribution")
axes[2].set_xlabel("Study Hours")
axes[2].set_ylabel("Count")

plt.tight_layout()
plt.savefig("data_representation.png")
plt.show()
print("Plot saved!")


print("""
==============================
KEY TAKEAWAYS
==============================
- Numerical data  : Ready to use directly
- Categorical data : Must convert to numbers
- Label Encoding  : Good for 2-3 categories
- One Hot Encoding: Good for many categories
- Scaling         : Brings all values to 0-1 range
- Dataset         : 30 student records
==============================
""")
