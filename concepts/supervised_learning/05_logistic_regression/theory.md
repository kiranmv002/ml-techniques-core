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

## 📌 Key Terms

| Term | Meaning | Example |
|------|---------|---------|
| Sigmoid | Function that converts to 0-1 | 0.8 means 80% chance |
| Threshold | Decision boundary | Default is 0.5 |
| Precision | Of all predicted yes how many are correct | 90% |
| Recall | Of all actual yes how many did we catch | 85% |
| F1 Score | Balance of precision and recall | 87% |
| Confusion Matrix | Table of correct and wrong predictions | TP FP TN FN |

---

## 💡 Why It Matters
- Most common classification algorithm
- Gives probability not just class label
- Easy to interpret and explain
- Works well when classes are linearly separable
- Fast to train even on large datasets

---

## ✅ Check Yourself Before Code
- [ ] Why is logistic regression a classification algorithm?
- [ ] What does the sigmoid function do?
- [ ] What is the difference between precision and recall?
- [ ] When would you lower the threshold below 0.5?
