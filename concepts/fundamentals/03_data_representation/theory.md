# 📦 Data Representation in Machine Learning

## 🤔 What is Data Representation?
Data representation means **how we store and present data**
to a machine learning model. Machines cannot understand
text or categories directly. We need to convert everything
into numbers.

---

## 🌍 Real Life Example

> 🛒 **Online Shopping Scenario:**
> You have customer data like:
> - Name: "John"
> - Gender: "Male"
> - City: "Chennai"
> - Purchase Amount: 5000
>
> The machine cannot understand "John" or "Male" or "Chennai"
> We need to convert them into numbers before feeding to ML model.

---

## 🔑 Key Types of Data

### 1. Numerical Data
- Already in number format
- Ready to use directly
- Example: Age, Height, Weight, Score

### 2. Categorical Data
- Text or label format
- Must be converted to numbers
- Example: Gender (Male/Female), City, Color

### 3. How to Convert Categorical to Numbers

#### Label Encoding
- Assigns a number to each category
- Male = 0, Female = 1
- Chennai = 0, Mumbai = 1, Delhi = 2

#### One Hot Encoding
- Creates a new column for each category
- Better when categories have no order
- Example:
  - Chennai → [1, 0, 0]
  - Mumbai  → [0, 1, 0]
  - Delhi   → [0, 0, 1]

---

## 📌 Key Terms

| Term | Meaning | Example |
|------|---------|---------|
| Feature | Input column | Age, City |
| Numerical | Number data | 25, 5000 |
| Categorical | Text/label data | Male, Chennai |
| Label Encoding | Convert text to number | Male=0, Female=1 |
| One Hot Encoding | Convert to 0s and 1s | Chennai=[1,0,0] |
| Scaling | Bring numbers to same range | 0 to 1 |

---

## 💡 Why It Matters
- ML models only understand numbers
- Wrong encoding can confuse the model
- Scaling helps model learn faster and better

---

## ✅ Check Yourself Before Code
- [ ] What is the difference between numerical and categorical data?
- [ ] When do you use label encoding vs one hot encoding?
- [ ] Why do we need to scale data?
- [ ] Can a ML model understand text directly?
