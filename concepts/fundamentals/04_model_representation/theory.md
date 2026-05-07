# 🏗️ Model Representation in Machine Learning

## 🤔 What is Model Representation?
Model representation means **how a machine learning model
stores what it has learned.** Every ML algorithm represents
its learning in a different way — some use equations,
some use trees, some use boundaries.

---

## 🌍 Real Life Example

> 🏠 **House Price Scenario:**
> You want to predict the price of a house.
> You notice that:
> - Bigger house = Higher price
> - More rooms = Higher price
> - Older house = Lower price
>
> A ML model learns these relationships and
> stores them as a mathematical equation.
> That equation IS the model representation.

---

## 🔑 Types of Model Representation

### 1. Linear Model (Equation)
- Stores learning as a straight line equation
- y = mx + c
- Example: Price = 500 x Area + 10000
- Simple and easy to understand

### 2. Decision Tree (Tree Structure)
- Stores learning as a series of yes/no questions
- Like a flowchart
- Example:
  - Is area > 1000 sqft?
    - Yes → Is rooms > 3? → Yes → High Price
    - No → Low Price

### 3. Boundary (Classification)
- Stores learning as a line or curve that separates classes
- Example: Everything above this line = Spam
- Everything below = Not Spam

---

## 📌 Key Terms

| Term | Meaning | Example |
|------|---------|---------|
| Model | What the algorithm learned | Equation, Tree |
| Parameters | Values the model learned | Slope, intercept |
| Weight | How important a feature is | Area has high weight |
| Bias | Base value before features | Starting price |
| Prediction | Output from the model | House price = 50000 |

---

## 💡 Why It Matters
- Understanding model representation helps you
  choose the right algorithm
- It helps you explain predictions to others
- It helps you debug when model gives wrong results

---

## ✅ Check Yourself Before Code
- [ ] What is a parameter in a model?
- [ ] How does a linear model represent learning?
- [ ] How does a decision tree represent learning?
- [ ] What is the difference between weight and bias?
