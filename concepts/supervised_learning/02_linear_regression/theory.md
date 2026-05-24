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
