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


