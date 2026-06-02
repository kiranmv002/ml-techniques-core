# 📈 Multiple Linear Regression

## 🤔 What is Multiple Linear Regression?
Multiple linear regression is just like linear regression
but instead of one input feature you use many features
at the same time to predict the target.

---

## 🌍 Real Life Example

> 🚗 **Car Price Scenario:**
> You want to predict the price of a used car.
> Just using mileage alone is not enough.
> You also need:
> - Age of the car
> - Brand
> - Engine size
> - Number of owners
>
> Multiple linear regression combines all these
> features together into one equation to give
> a better price prediction.

---

## 🔑 How it Works

### The Equation
y = w1x1 + w2x2 + w3x3 + ... + c

- y  = target (car price)
- x1 = feature 1 (mileage)
- x2 = feature 2 (age)
- x3 = feature 3 (engine size)
- w1 w2 w3 = weights (importance of each feature)
- c  = intercept (base value)

### What the Model Learns
For each feature the model learns a weight.
Higher weight means that feature affects
the prediction more.

---

## 📌 Key Terms

| Term | Meaning | Example |
|------|---------|---------|
| Multiple features | Many input columns | Mileage age engine size |
| Weight | Importance of a feature | Engine size weight = 5000 |
| Multicollinearity | Two features are too similar | Height and weight |
| Adjusted R2 | R2 adjusted for number of features | Better than R2 for multiple features |
| Feature selection | Choosing best features | Remove useless columns |
