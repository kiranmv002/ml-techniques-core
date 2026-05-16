# 🧹 Data Preprocessing in Machine Learning

## 🤔 What is Data Preprocessing?
Data preprocessing means **cleaning and preparing raw data**
before feeding it to a ML model. Real world data is messy.
It has missing values, wrong formats, outliers and noise.
We need to fix all of this before training any model.

---

## 🌍 Real Life Example

> 🏥 **Hospital Patient Data Scenario:**
> A hospital collects patient data but it has problems:
> - Some patients did not fill age → missing value
> - One patient entered age as 999 → outlier
> - Blood pressure is in different units → needs scaling
> - Gender is stored as M/F → needs encoding
>
> Before using this data to predict diseases,
> we need to clean and fix all these problems.
> That is data preprocessing.

---

## 🔑 Key Steps in Preprocessing

### 1. Handle Missing Values
- Remove rows with missing values (if few rows affected)
- Fill with mean or median (for numbers)
- Fill with most common value (for categories)

### 2. Remove Outliers
- Values that are way too high or too low
- Example: Age = 999, Height = 0
- Can be removed or replaced with median

### 3. Encode Categorical Data
- Convert text to numbers
- Male = 0, Female = 1
- Already covered in data representation topic

### 4. Scale / Normalize Features
- Bring all features to same range
- So one feature does not dominate others
- Min-Max scaling → 0 to 1
- Standard scaling → mean 0, std 1

### 5. Remove Duplicates
- Same row appearing multiple times
- Wastes training time and skews results

---

## 📌 Key Terms

| Term | Meaning | Example |
|------|---------|---------|
| Missing value | Empty cell in data | NaN, null |
| Outlier | Extreme unusual value | Age = 999 |
| Scaling | Bring to same range | 0 to 1 |
| Normalization | Scale to mean 0 std 1 | Standard scaler |
| Duplicate | Same row repeated | Two identical patients |
| Imputation | Filling missing values | Fill with mean |

---

## 💡 Why It Matters
- Dirty data leads to wrong predictions
- Missing values crash many ML algorithms
- Outliers pull model in wrong direction
- Unscaled features confuse distance based models

---

## ✅ Check Yourself Before Code
- [ ] What are the 5 main preprocessing steps?
- [ ] When do you fill missing values with mean vs median?
- [ ] What is the difference between scaling and normalization?
- [ ] Why do outliers affect ML models?
