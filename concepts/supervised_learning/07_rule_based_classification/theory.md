# 📋 Rule Based Classification

## 🤔 What is Rule Based Classification?
Rule based classification means making predictions
using a set of if then else rules.
Instead of learning patterns from math equations,
the model uses simple human readable rules
to decide which class something belongs to.

---

## 🌍 Real Life Example

> ⛅ **Weather Decision Scenario:**
> You want to decide if you should play cricket outside.
>
> Rule 1: IF weather = sunny AND humidity < 70 THEN play
> Rule 2: IF weather = rainy THEN dont play
> Rule 3: IF weather = cloudy AND wind < 20 THEN play
> Rule 4: IF none of above THEN dont play
>
> These are hand written rules based on common sense.
> Rule based classification works the same way.
> Either rules are written by experts or
> learned automatically from data.

---

## 🔑 How it Works

### Manual Rules
- Domain experts write the rules
- Like a doctor writing diagnosis rules
- Easy to understand and explain
- But hard to scale with many features

### Learned Rules (from Decision Tree)
- Train a decision tree
- Extract rules from each path root to leaf
- Each path becomes one rule
- More scalable than manual rules

### Rule Format
IF condition1 AND condition2 THEN class
Example:
IF income > 50000 AND credit_score > 700
THEN approve loan

