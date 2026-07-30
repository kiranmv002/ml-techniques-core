# 🔍 Model Interpretability

## 🤔 What is Model Interpretability?
Model interpretability means understanding
why a model made a particular prediction.
It is not enough to just get a prediction.
You need to know the reason behind it.

---

## 🌍 Real Life Example

> 👔 **Employee Attrition Scenario:**
> A company uses ML to predict if an employee
> will leave the company or not.
>
> The model says: "This employee will leave"
>
> But HR asks: WHY will they leave?
> - Is it because of low salary?
> - Too many work hours?
> - No promotion in years?
>
> Model interpretability answers that WHY question.
> Without it the prediction is useless to HR.

---

## 🔑 Key Ways to Interpret Models

### 1. Feature Importance
- Which features affected the prediction most
- Available directly from tree based models
- Higher value = more important feature

### 2. Coefficients (Linear Models)
- Each feature has a weight
- Positive weight = increases prediction
- Negative weight = decreases prediction

### 3. Partial Dependence
- How does prediction change as one feature changes
- All other features stay fixed
- Shows the relationship clearly

### 4. LIME
- Local Interpretable Model Agnostic Explanations
- Explains one prediction at a time
- Works for any ML model

---
