# Predictive Analytics — Classification Model

**Notebook:** `Predictive_Analytics_Classification_Model.ipynb`  
**Dataset:** `Lab_Exam_binary_classification_dataset.csv`  
**Task:** Binary Classification — predict `Target` (Yes / No) from `Feature1` and `Feature2`  
**Model:** Logistic Regression (with `class_weight='balanced'`)

---

## 📁 Files Required

| File | Description |
|------|-------------|
| `Predictive_Analytics_Classification_Model.ipynb` | Main notebook |
| `Lab_Exam_binary_classification_dataset.csv` | Input dataset |

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Raw samples | 1,020 |
| Features | `Feature1` (float), `Feature2` (int) |
| Target | Binary — `Yes` / `No` |
| Missing Target rows | 20 → **dropped** |
| Outlier | `Feature1 = 10,000` → **removed** |
| Clean samples | **999** |
| Class balance | No = 784 (78.5%) · Yes = 215 (21.5%) |

---

## 📓 Notebook Structure

### 1. Load Libraries
Imports: `numpy`, `pandas`, `seaborn`, `matplotlib`, `sklearn`

---

### 2. Load & Clean Data
```python
df = pd.read_csv("Lab_Exam_binary_classification_dataset.csv")
df = df.dropna()               # drop 20 rows with missing Target
df = df[df["Feature1"] < 100]  # remove outlier (Feature1 = 10,000)
df = df.reset_index(drop=True)
```

---

### Task 1 — Exploratory Data Analysis (EDA)

A 2×2 figure with four panels:

| Panel | Plot |
|-------|------|
| (A) | Bar chart — class distribution |
| (B) | Histogram — Feature1 by class |
| (C) | Histogram — Feature2 by class |
| (D) | Scatter plot — Feature1 vs Feature2, coloured by class |

**Key finding:** The scatter plot reveals that `Yes` samples form a **central island** — the boundary between classes is non-linear.

---

### Task 2 — Build Classification Model

**Steps:**

1. **Encode target** — `Yes → 1`, `No → 0`
2. **Split** — 80% train / 20% test (stratified)
3. **Scale** — `StandardScaler` (Feature2 is ~100× larger than Feature1)
4. **Train** — `LogisticRegression(C=1.0, class_weight='balanced')` to handle class imbalance
5. **Cross-validate** — 5-fold `StratifiedKFold` inside a `Pipeline` (prevents data leakage)

```
5-Fold CV Accuracy : ~0.78
5-Fold CV ROC-AUC  : ~0.53
```

> **Note:** Logistic Regression captures only a linear boundary. The dataset's non-linear structure limits its performance. A kernel-based model (e.g. SVM-RBF) would improve results significantly.

---

### Task 3 — Decision Boundary

Plots the model's decision boundary over the scaled feature space using a mesh grid and `predict_proba`. The dashed black line marks the **0.5 probability threshold**.

> ⚠️ The decision boundary cell references `model` but the training cell uses `log_reg`. Fix by replacing `model` with `log_reg` in that cell, or add this line before it:
> ```python
> model = log_reg
> ```

---

### Task 4 — Evaluate Model Performance

**Metrics computed:**

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| Precision | Of predicted Yes, how many are actually Yes |
| Recall | Of actual Yes, how many were correctly predicted |
| F1-Score | Harmonic mean of Precision and Recall |
| ROC-AUC | Area under the ROC curve |
| 5-Fold CV | Cross-validated Accuracy and ROC-AUC |

**Plots saved:**
- `confusion_matrix.png` — True/False Positives and Negatives
- `roc_curve.png` — ROC curve with AUC score

---

## ⚠️ Known Issue in Notebook

The decision boundary cell (Task 3) uses the variable name `model`:
```python
Z = model.predict_proba(...)
```
But the trained model is stored as `log_reg`. Fix by replacing `model` with `log_reg` in that cell, or add this line before it:
```python
model = log_reg
```

---

## 📈 Results Summary

| Metric | Value |
|--------|-------|
| Test Accuracy | ~78% |
| Test ROC-AUC | ~0.53 |
| 5-Fold CV Accuracy | ~0.78 ± 0.00 |
| 5-Fold CV ROC-AUC | ~0.53 ± 0.03 |

> Logistic Regression gives ~78% accuracy by predicting the majority class (`No`). The low AUC reflects poor discrimination of the minority class (`Yes`). Consider SVM (RBF), Random Forest, or Gradient Boosting for better results on this non-linearly separable dataset.

---

*Predictive Analytics Lab Exam-2 | 2026*
