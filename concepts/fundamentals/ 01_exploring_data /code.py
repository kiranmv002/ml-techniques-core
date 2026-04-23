# ============================================
# 📊 Exploring Data in Machine Learning
# Topic   : 01 - Exploring Data
# Folder  : fundamentals
# Author  : M V Kiran
# GitHub  : github.com/kiranmv002
# ============================================


# ── Step 1: Import Libraries ────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("✅ Libraries loaded!")


# ── Step 2: Load Dataset ─────────────────────
# Real life: Like loading patient health records
from sklearn.datasets import load_iris

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df['species'] = df['target'].map({
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
})

print("\n📋 First 5 rows:")
print(df.head())


# ── Step 3: Basic Info ───────────────────────
print("\n📐 Shape (rows, columns):", df.shape)
print("\n🔤 Column Types:")
print(df.dtypes)
print("\n❓ Missing Values:")
print(df.isnull().sum())
print("\n🔁 Duplicate Rows:", df.duplicated().sum())


# ── Step 4: Statistics ───────────────────────
print("\n📊 Basic Statistics:")
print(df.describe())


# ── Step 5: Visualize Distributions ─────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, col in enumerate(data.feature_names):
    sns.histplot(df[col], kde=True, ax=axes[i], color='steelblue')
    axes[i].set_title(col.split()[0])
plt.suptitle("Feature Distributions", fontsize=14)
plt.tight_layout()
plt.savefig("distributions.png")
plt.show()
print("✅ Distribution plot saved!")


# ── Step 6: Correlation Heatmap ─────────────
plt.figure(figsize=(6, 4))
sns.heatmap(
    df.iloc[:, :4].corr(),
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation.png")
plt.show()
print("✅ Heatmap saved!")


# ── Step 7: Class Distribution ───────────────
plt.figure(figsize=(5, 3))
sns.countplot(x='species', data=df, palette='Set2')
plt.title("Class Distribution")
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.show()
print("✅ Class distribution saved!")


# ── Step 8: Key Takeaways ────────────────────
print("""
==============================
✅ KEY TAKEAWAYS
==============================
- Dataset  : 150 rows, 4 features, 3 classes
- Missing  : No missing values found
- Insight  : Petal length & width are highly correlated
- Balance  : Classes are equal (50 each)
- Rule     : Always explore data before any model!
==============================
""")
