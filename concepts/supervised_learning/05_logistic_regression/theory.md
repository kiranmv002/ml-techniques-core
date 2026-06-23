# 🔵 Logistic Regression

## 🤔 What is Logistic Regression?
Even though it has regression in the name,
logistic regression is actually a classification algorithm.
It predicts the probability of something being
yes or no, true or false, 0 or 1.

---

## 🌍 Real Life Example

> 🏥 **Diabetes Detection Scenario:**
> A doctor wants to know if a patient has diabetes or not.
> They have data like glucose level, BMI, age etc.
>
> Logistic regression looks at all these values and
> gives a probability between 0 and 1.
> If probability > 0.5 → Diabetic
> If probability < 0.5 → Not Diabetic
>
> It does not predict a number like linear regression.
> It predicts which category something belongs to.

---

## 🔑 How it Works

### Step 1
Calculate a linear equation like linear regression
z = w1x1 + w2x2 + ... + c

### Step 2
Pass z through sigmoid function
probability = 1 / (1 + e^-z)

### Step 3
Sigmoid converts any number to 0-1 range
This becomes the probability of being class 1

### Step 4
If probability >= 0.5 → predict class 1
If probability < 0.5  → predict class 0

---
