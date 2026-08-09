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


# ── Step 5: Compare Manual vs Learned Rules ──
print("\n--- Comparison ---")
print(f"Manual Rules Accuracy : {round(manual_acc * 100, 2)}%")
print(f"Decision Tree Accuracy: {round(tree_acc  * 100, 2)}%")


# ── Step 6: Sample Prediction ────────────────
print("\n--- Sample Prediction ---")

# using manual rules
test_day = {"weather": "sunny", "temperature": "mild",
            "humidity": 68, "wind_speed": 12, "play": 1}
test_series = pd.Series(test_day)
manual_result = manual_rules(test_series)

print("Day: sunny, mild, humidity=68, wind=12")
print("Manual rule says:", "Play ✅" if manual_result == 1 else "Dont Play ❌")

# using tree
test_encoded = np.array([[
    le_weather.transform(["sunny"])[0],
    le_temp.transform(["mild"])[0],
    68,
    12
]])
tree_result = tree.predict(test_encoded)[0]
print("Decision tree says:", "Play ✅" if tree_result == 1 else "Dont Play ❌")


# ── Visualization ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# weather vs play
weather_play = df.groupby(["weather", "play"]).size().unstack(fill_value=0)
weather_play.plot(kind="bar", ax=axes[0],
                  color=["tomato", "seagreen"], alpha=0.8)
axes[0].set_title("Weather vs Play Decision")
axes[0].set_xlabel("Weather")
axes[0].set_ylabel("Count")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
axes[0].legend(["Dont Play", "Play"])

# humidity vs play
axes[1].scatter(df[df["play"] == 1]["humidity"],
                df[df["play"] == 1]["wind_speed"],
                color="seagreen", label="Play", alpha=0.6)
axes[1].scatter(df[df["play"] == 0]["humidity"],
                df[df["play"] == 0]["wind_speed"],
                color="tomato", label="Dont Play", alpha=0.6)
axes[1].axvline(x=82, color="black",
                linestyle="--", label="humidity=82")
axes[1].axhline(y=21, color="blue",
                linestyle="--", label="wind=21")
axes[1].set_xlabel("Humidity")
axes[1].set_ylabel("Wind Speed")
axes[1].set_title("Rules Visualized\n(Decision Boundaries)")
axes[1].legend(fontsize=7)

# accuracy comparison
models = ["Manual Rules", "Decision Tree"]
accs = [manual_acc * 100, tree_acc * 100]
axes[2].bar(models, accs,
            color=["steelblue", "seagreen"], alpha=0.8)
axes[2].set_ylabel("Accuracy %")
axes[2].set_title("Manual vs Learned Rules")
axes[2].set_ylim([0, 110])
for i, v in enumerate(accs):
    axes[2].text(i, v + 1, str(round(v, 1)) + "%",
                 ha="center", fontsize=11)

plt.tight_layout()
plt.savefig("rule_based_classification.png")
plt.show()
print("\nPlot saved!")


print("""
==============================
KEY TAKEAWAYS
==============================
- Rules are simple if then else statements
- Manual rules need domain expert knowledge
- Learned rules come from decision tree paths
- Rules are very easy to explain to anyone
- Good for medical legal and finance decisions
- Does not scale well to very complex problems
- Always compare manual vs learned rule accuracy
==============================
""")
