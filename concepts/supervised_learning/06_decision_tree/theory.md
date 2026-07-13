# 🌳 Decision Tree

## 🤔 What is a Decision Tree?
A decision tree is a ML algorithm that makes predictions
by asking a series of yes or no questions.
It looks exactly like a tree upside down.
It starts from one question at the top and
branches down until it reaches a final answer.

---

## 🌍 Real Life Example

> 🏦 **Loan Approval Scenario:**
> A bank decides whether to approve a loan or not.
> They ask questions one by one:
>
> Is income > 50000?
> → Yes: Is credit score > 700?
>        → Yes: Approve loan ✅
>        → No: Reject loan ❌
> → No: Reject loan ❌
>
> That is exactly how a decision tree works.
> Each question splits the data into two groups.
> At the end you get a final decision.

---

## 🔑 How it Works

### Step 1 - Find Best Question
The tree finds the question that best separates
the data into groups. It uses Gini impurity or
entropy to measure how pure each split is.

### Step 2 - Split Data
Data is split into two groups based on the answer.
Yes goes left. No goes right.

### Step 3 - Repeat
For each group the tree asks another best question.
This continues until a stopping condition is met.

### Step 4 - Make Prediction
New data travels down the tree answering questions
until it reaches a leaf node which gives the prediction.

---

## 📌 Key Terms

| Term | Meaning | Example |
|------|---------|---------|
| Root node | First question at top | Is income > 50000 |
| Branch | Path after answering yes or no | Yes path |
| Leaf node | Final answer at bottom | Approve or Reject |
| Depth | How many levels the tree has | Depth 3 |
| Gini impurity | How mixed a group is | 0 = pure 1 = mixed |
| Pruning | Cutting tree to avoid overfit | Remove deep branches |

---

## 💡 Why It Matters
- Easy to understand and visualize
- No need to scale features
- Works for both classification and regression
- Can handle missing values
- But deep trees overfit easily

---

## ✅ Check Yourself Before Code
- [ ] What is a root node and leaf node?
- [ ] What is Gini impurity?
- [ ] Why does a very deep tree overfit?
- [ ] What is pruning and why do we do it?
