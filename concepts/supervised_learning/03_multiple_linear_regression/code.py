# Multiple Linear Regression
# Author: M V Kiran
# github.com/kiranmv002

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# load dataset
df = pd.read_csv("data/car_prices.csv")

print("Dataset loaded!")
print("\nFirst 5 rows:")
print(df.head())
print("\nShape:", df.shape)


# ── Step 1: Explore Features ─────────────────
print("\n--- Feature Overview ---")
print(df.describe())

print("\nCorrelation with price:")
correlations = df.corr()["price"].sort_values(ascending=False)
print(correlations)


# ── Step 2: Prepare Data ─────────────────────
feature_names = ["mileage_km", "age_years", "engine_cc",
                 "owners", "brand_encoded", "service_history"]

X = df[feature_names].values
y = df["price"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", len(X_train))
print("Test samples    :", len(X_test))


# ── Step 3: Train Without Scaling ────────────
print("\n--- Without Scaling ---")

model_no_scale = LinearRegression()
model_no_scale.fit(X_train, y_train)
preds_no_scale = model_no_scale.predict(X_test)

mae_no_scale = mean_absolute_error(y_test, preds_no_scale)
r2_no_scale = r2_score(y_test, preds_no_scale)

print("MAE:", round(mae_no_scale, 2))
print("R2 :", round(r2_no_scale, 2))


# ── Step 4: Train With Scaling ────────────────
# scaling makes weights more comparable
# and helps model learn better

print("\n--- With Scaling ---")

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = LinearRegression()
model_scaled.fit(X_train_scaled, y_train)
preds_scaled = model_scaled.predict(X_test_scaled)

mae_scaled = mean_absolute_error(y_test, preds_scaled)
r2_scaled = r2_score(y_test, preds_scaled)

print("MAE:", round(mae_scaled, 2))
print("R2 :", round(r2_scaled, 2))


# ── Step 5: Feature Weights ───────────────────
print("\n--- Feature Weights (after scaling) ---")
print("Higher weight = more important feature")
for name, weight in zip(feature_names, model_scaled.coef_):
    print(f"  {name:20}: {round(weight, 2)}")
print(f"  {'intercept':20}: {round(model_scaled.intercept_, 2)}")


# ── Step 6: Sample Prediction ─────────────────
print("\n--- Sample Prediction ---")

new_car = pd.DataFrame({
    "mileage_km"     : [40000],
    "age_years"      : [5],
    "engine_cc"      : [1600],
    "owners"         : [1],
    "brand_encoded"  : [1],
    "service_history": [1]
})

new_car_scaled = scaler.transform(new_car)
predicted = model_scaled.predict(new_car_scaled)
print("Car: 40000km, 5yr old, 1600cc, 1 owner, good brand, service done")
print("Predicted price: Rs", round(predicted[0], 2))


# ── Visualization ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# correlation heatmap
corr_matrix = df[feature_names + ["price"]].corr()
im = axes[0].imshow(corr_matrix, cmap="coolwarm", aspect="auto")
axes[0].set_xticks(range(len(corr_matrix.columns)))
axes[0].set_yticks(range(len(corr_matrix.columns)))
axes[0].set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=7)
axes[0].set_yticklabels(corr_matrix.columns, fontsize=7)
axes[0].set_title("Feature Correlation")
plt.colorbar(im, ax=axes[0])

# actual vs predicted
axes[1].scatter(y_test, preds_scaled, color="steelblue", alpha=0.6)
axes[1].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             color="red", linestyle="--")
axes[1].set_xlabel("Actual Price")
axes[1].set_ylabel("Predicted Price")
axes[1].set_title("Actual vs Predicted")

# feature weights bar chart
axes[2].barh(feature_names, model_scaled.coef_,
             color=["seagreen" if w > 0 else "tomato"
                    for w in model_scaled.coef_])
axes[2].axvline(x=0, color="black", linestyle="--")
axes[2].set_xlabel("Weight")
axes[2].set_title("Feature Weights\n(Green=Positive, Red=Negative)")

plt.tight_layout()
plt.savefig("multiple_linear_regression.png")
plt.show()
print("\nPlot saved!")


print("""
==============================
KEY TAKEAWAYS
==============================
- Multiple regression uses many features
- Scaling makes weights more comparable
- Negative weight = feature reduces price
- Positive weight = feature increases price
- Mileage and age reduce car price
- Engine size and service history increase it
- Always scale before comparing weights
==============================
""")
