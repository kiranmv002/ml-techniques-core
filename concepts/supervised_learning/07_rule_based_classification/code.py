# Rule Based Classification
# Author: M V Kiran
# github.com/kiranmv002

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# load dataset
df = pd.read_csv("data/weather.csv")

print("Dataset loaded!")
print("\nFirst 5 rows:")
print(df.head())
print("\nShape:", df.shape)
print("\nClass distribution:")
print(df["play"].value_counts())
print("0 = Dont Play, 1 = Play")


# ── Step 1: Manual Rule Based Classification ─
# writing rules by hand based on common sense

print("\n--- Manual Rules ---")

def manual_rules(row):
    # Rule 1: sunny and low humidity → play
    if row["weather"] == "sunny" and row["humidity"] <= 75:
        return 1
    # Rule 2: rainy and high wind → dont play
    elif row["weather"] == "rainy" and row["wind_speed"] > 20:
        return 0
    # Rule 3: cloudy → usually play
    elif row["weather"] == "cloudy":
        return 1
    # Rule 4: rainy but calm wind → play
    elif row["weather"] == "rainy" and row["wind_speed"] <= 20:
        return 1
    # Rule 5: default → dont play
    else:
        return 0

df["manual_pred"] = df.apply(manual_rules, axis=1)
manual_acc = accuracy_score(df["play"], df["manual_pred"])
print("Manual rules accuracy:", round(manual_acc * 100, 2), "%")

print("\nManual Rules used:")
print("Rule 1: IF weather=sunny AND humidity<=75 THEN play")
print("Rule 2: IF weather=rainy AND wind>20 THEN dont play")
print("Rule 3: IF weather=cloudy THEN play")
print("Rule 4: IF weather=rainy AND wind<=20 THEN play")
print("Rule 5: ELSE dont play")


# ── Step 2: Encode for ML Model ──────────────
le_weather = LabelEncoder()
le_temp = LabelEncoder()

df["weather_enc"] = le_weather.fit_transform(df["weather"])
df["temp_enc"] = le_temp.fit_transform(df["temperature"])

feature_names = ["weather_enc", "temp_enc",
                 "humidity", "wind_speed"]

X = df[feature_names].values
y = df["play"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ── Step 3: Learn Rules from Decision Tree ───
print("\n--- Learned Rules from Decision Tree ---")

tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)

tree_preds = tree.predict(X_test)
tree_acc = accuracy_score(y_test, tree_preds)
print("Decision tree accuracy:", round(tree_acc * 100, 2), "%")

print("\nExtracted rules from tree:")
rules = export_text(tree, feature_names=feature_names)
print(rules)


# ── Step 4: Extract Rules as If Then Statements
print("\n--- Rules in Simple English ---")

# manually reading key paths from the tree
# and writing them as human readable rules
print("Rule A: IF humidity <= 82 AND wind_speed <= 21")
print("        THEN Play")
print()
print("Rule B: IF humidity > 82 AND weather = rainy")
print("        THEN Dont Play")
print()
print("Rule C: IF wind_speed > 21")
print("        THEN Dont Play")
print()
print("Rule D: IF weather = cloudy AND humidity <= 88")
print("        THEN Play")
