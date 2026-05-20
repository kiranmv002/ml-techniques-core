# 🎯 Problem Formulation in Machine Learning

## 🤔 What is Problem Formulation?
Problem formulation means **clearly defining what you want
the ML model to do** before writing any code. It is the
most important step. If you define the problem wrong,
even the best model will give useless results.

---

## 🌍 Real Life Example

> 🏦 **Bank Scenario:**
> A bank wants to use ML. But what exactly do they want?
>
> Wrong way: "We want to use AI for our bank"
> Right way: "We want to predict if a loan applicant
> will default or not, using their income, credit score
> and employment history"
>
> The right way clearly says:
> - What to predict (default or not)
> - What inputs to use (income, credit score, employment)
> - What type of problem it is (classification)

---

## 🔑 Key Questions to Ask

### 1. What are you predicting?
- A number? → Regression problem
- A category? → Classification problem
- Groups in data? → Clustering problem

### 2. What data do you have?
- What features are available?
- How many samples do you have?
- Is the data labeled or not?

### 3. How will you measure success?
- Accuracy, precision, recall for classification
- MAE, RMSE, R2 for regression

### 4. What are the constraints?
- How fast does prediction need to be?
- How much data can you store?
- Does the model need to be explainable?

---

## 📌 Types of ML Problems

| Problem Type | Predict | Example |
|-------------|---------|---------|
| Binary Classification | Yes or No | Spam or Not Spam |
| Multi Classification | One of many | Dog, Cat or Bird |
| Regression | A number | House price |
| Clustering | Groups | Customer segments |
| Ranking | Order | Search results |

---

## 💡 Why It Matters
- Wrong problem definition wastes weeks of work
- It guides which algorithm to choose
- It defines what data you need to collect
- It sets clear success criteria

---

## ✅ Check Yourself Before Code
- [ ] What is the difference between regression and classification?
- [ ] How do you decide which ML problem type to use?
- [ ] What metrics would you use for a classification problem?
- [ ] What metrics would you use for a regression problem?
