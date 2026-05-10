# 🔀 Train, Validation and Test Split

## 🤔 What is Train Val Test Split?
When we build a ML model we cannot use all data for training.
We need to keep some data aside to check if the model
actually learned well or just memorized the training data.

So we split data into 3 parts.

---

## 🌍 Real Life Example

> 📚 **Exam Preparation Scenario:**
> Imagine you are preparing for an exam.
>
> - Train set = Textbook chapters you study from
> - Validation set = Practice tests you take while studying
>   to check how well you are learning
> - Test set = The actual final exam
>   (you see these questions for the first time)
>
> If you only study the practice test answers without
> understanding, you will fail the real exam.
> That is exactly what overfitting is in ML.

---

## 🔑 The 3 Parts

### 1. Training Set (60-70%)
- Model learns from this data
- Like studying from textbook
- Largest portion of data

### 2. Validation Set (10-20%)
- Used to tune the model during training
- Like practice tests
- Helps avoid overfitting

### 3. Test Set (10-20%)
- Used only at the very end
- Never seen by model during training
- Like the final exam
- Gives real performance score

---

## 📌 Key Terms

| Term | Meaning | Example |
|------|---------|---------|
| Overfitting | Model memorized training data | 100% train, 50% test |
| Underfitting | Model did not learn enough | 60% train, 58% test |
| Good fit | Model learned well | 90% train, 88% test |
| Cross validation | Split data multiple ways to test | K-Fold |

---

## 💡 Why It Matters
- Without test set you dont know real performance
- Without validation set model may overfit
- Always split data before any preprocessing
- Never use test set during training

---

## ✅ Check Yourself Before Code
- [ ] What is the difference between validation and test set?
- [ ] What is overfitting in simple words?
- [ ] Why should test set never be seen during training?
- [ ] What is a good train val test split ratio?
