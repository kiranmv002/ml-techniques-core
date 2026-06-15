# ⚖️ Overfitting and Underfitting

## 🤔 What is Overfitting and Underfitting?
These are the two most common problems in ML.
They happen when a model learns too much or too little
from the training data.

---

## 🌍 Real Life Example

> 📚 **Exam Preparation Scenario:**
>
> Underfitting = Student barely studied.
> Does badly on both practice tests and real exam.
> Did not learn enough.
>
> Overfitting = Student memorized every practice question
> word by word. Does great on practice tests but fails
> real exam because real questions are slightly different.
>
> Good Fit = Student understood the concepts.
> Does well on both practice tests and real exam.

---

## 🔑 The 3 Cases

### 1. Underfitting
- Model is too simple
- Does badly on training data
- Does badly on test data
- High bias low variance
- Fix: Use more complex model or more features

### 2. Overfitting
- Model is too complex
- Does great on training data
- Does badly on test data
- Low bias high variance
- Fix: Use simpler model, more data, or regularization

### 3. Good Fit
- Model learned the right patterns
- Does well on both training and test data
- Low bias low variance
- This is what we always aim for

---

## 📌 Key Terms

| Term | Meaning | Example |
|------|---------|---------|
| Bias | Error from wrong assumptions | Too simple model |
| Variance | Sensitivity to training data | Too complex model |
| Regularization | Penalty to reduce overfitting | L1 L2 |
| Cross validation | Test on multiple splits | K fold |
| Learning curve | Train vs val error over data size | Diagnosis tool |

---

## 💡 Why It Matters
- Overfitting is the most common ML mistake
- A model that only works on training data is useless
- Always compare training and test performance
- Big gap between train and test = overfitting

---

## ✅ Check Yourself Before Code
- [ ] What is the difference between overfitting and underfitting?
- [ ] How do you detect overfitting?
- [ ] What is regularization in simple words?
- [ ] What does high bias mean?
