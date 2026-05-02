# ============================================
# 📊 Exploring Data in Machine Learning
# Topic   : 01 - Exploring Data
# Folder  : fundamentals
# Author  : M V Kiran
# GitHub  : github.com/kiranmv002
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load dataset from local csv file
df = pd.read_csv("data/iris.csv")

print("Dataset loaded!")
print("\nFirst 5 rows:")
print(df.head())


# basic info
print("\nShape (rows, columns):", df.shape)
print("\nColumn Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())


# statistics
print("\nBasic Statistics:")
print(df.describe())


# visualize distributions
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, col in enumerate(features):
    sns.histplot(df[col], kde=True, ax=axes[i], color="steelblue")
    axes[i].set_title(col)
plt.suptitle("Feature Distributions", fontsize=14)
plt.tight_layout()
plt.savefig("distributions.png")
plt.show()
print("Distribution plot saved!")


# correlation heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation.png")
plt.show()
print("Heatmap saved!")


# class distribution
plt.figure(figsize=(5, 3))
sns.countplot(x="species", data=df, palette="Set2")
plt.title("Class Distribution")
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.show()
print("Class distribution saved!")


print("""
==============================
KEY TAKEAWAYS
==============================
- Dataset  : 150 rows, 4 features, 3 classes
- Missing  : No missing values found
- Insight  : Petal length and width are highly correlated
- Balance  : Classes are equal (50 each)
- Rule     : Always explore data before building any model!
==============================
""")
