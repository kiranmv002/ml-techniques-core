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

## 📌 Key Terms

| Term | Meaning | Example |
|------|---------|---------|
| Interpretability | Understanding why model predicted | Why did loan get rejected |
| Feature importance | How much each feature contributed | Salary importance = 0.35 |
| Global explanation | Overall model behavior | Top features for all predictions |
| Local explanation | One specific prediction | Why this employee will leave |
| Black box | Model we cannot explain | Deep neural network |
| White box | Model we can explain | Linear regression decision tree |

---

## 💡 Why It Matters
- Builds trust in the model
- Required in finance health and legal domains
- Helps debug wrong predictions
- Helps improve the model by removing useless features
- Regulators often require explainability

---

## ✅ Check Yourself Before Code
- [ ] What is the difference between global and local explanation?
- [ ] Which models are naturally interpretable?
- [ ] Why is interpretability important in medical ML?
- [ ] What does feature importance score of 0 mean?
