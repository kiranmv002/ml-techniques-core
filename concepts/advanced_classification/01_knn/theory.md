# 🔵 K-Nearest Neighbors (KNN)

## 🤔 What is KNN?
KNN is one of the simplest ML algorithms.
It makes predictions by looking at the
K closest data points to the new data point
and takes a majority vote.

No training happens in KNN.
It just memorizes all the data and
uses it at prediction time.

---

## 🌍 Real Life Example

> 🏥 **Patient Diagnosis Scenario:**
> A new patient comes to hospital with certain symptoms.
> The doctor looks at past patients who had similar
> symptoms and checks what disease they had.
>
> If 5 out of 7 similar patients had diabetes
> the doctor concludes this patient likely has diabetes too.
>
> That is exactly KNN.
> K = 7 (number of neighbors to look at)
> Majority vote = 5 out of 7 had diabetes

---

## 🔑 How it Works

### Step 1
Calculate distance from new point to all training points.
Usually uses Euclidean distance.
distance = sqrt((x2-x1)² + (y2-y1)²)

### Step 2
Sort all training points by distance.
Pick the K closest ones.

### Step 3
For classification: majority vote wins.
For regression: take average of K neighbors.

### Step 4
Return the winning class as prediction.

---

## 📌 Key Terms

| Term | Meaning | Example |
|------|---------|---------|
| K | Number of neighbors to look at | K=5 |
| Distance | How far two points are | Euclidean Manhattan |
| Majority vote | Most common class wins | 4 out of 5 = diabetes |
| Lazy learner | No training phase | KNN memorizes all data |
| Curse of dimensionality | KNN fails with too many features | 100+ features |

---

## 💡 Why It Matters
- Very simple and easy to understand
- No training time needed
- Good for small datasets
- Works for both classification and regression
- But slow at prediction time for large datasets
- Sensitive to irrelevant features and scaling

