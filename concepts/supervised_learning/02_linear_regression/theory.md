# 📈 Linear Regression

## 🤔 What is Linear Regression?
Linear regression is the simplest ML algorithm.
It finds a straight line that best fits your data
and uses that line to predict new values.

---

## 🌍 Real Life Example

> 🏠 **House Sales Scenario:**
> You notice that bigger houses cost more.
> If you plot house size vs price on a graph
> you get points that roughly follow a straight line.
>
> Linear regression finds that exact best fit line.
> Then when someone asks the price of a 1200 sqft house
> you just look at where 1200 falls on that line.

---

## 🔑 How it Works

### The Equation
y = mx + c

- y = what you are predicting (price)
- x = input feature (house size)
- m = slope (how much price increases per sqft)
- c = intercept (base price when size is 0)

### What the Model Learns
The model finds the best values of m and c
that minimize the error between predicted
and actual values.

### How Error is Measured
Mean Squared Error (MSE) = average of (actual - predicted)²
The model tries to make this as small as possible.

---

## 📌 Key Terms

| Term | Meaning | Example |
|------|---------|---------|
| Slope | How steep the line is | Price increases 50 per sqft |
| Intercept | Where line crosses y axis | Base price = 10000 |
| MSE | Average squared error | How wrong predictions are |
| R2 Score | How well line fits data | 0.9 means 90% fit |
| Residual | Difference between actual and predicted | Actual 50000 - Predicted 48000 = 2000 |

---

## 💡 Why It Matters
- Simplest and most interpretable model
- Good starting point for any regression problem
- Fast to train even on large datasets
- Easy to explain to non technical people

---

## ✅ Check Yourself Before Code
- [ ] What does the slope tell you in linear regression?
- [ ] What is MSE and why do we minimize it?
- [ ] What does R2 score of 0.9 mean?
- [ ] When would linear regression fail?
