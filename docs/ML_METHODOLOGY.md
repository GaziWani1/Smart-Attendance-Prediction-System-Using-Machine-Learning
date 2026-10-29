# Machine Learning Methodology

This document details the machine learning design, feature engineering, data leakage prevention, algorithm selection, evaluation criteria, and statistical interpretation for the **Smart Attendance Prediction System**.

---

## 1. Problem Formulation and Target Definition

### 1.1 Objective
The problem is formulated as a supervised binary classification task:
**Will a student finish the current academic semester with a final attendance percentage below the mandatory institutional threshold of 75.0%?**

$$\text{attendance\_risk} = \begin{cases} 1 & \text{if } \text{final\_attendance\_pct} < 75.0 \\ 0 & \text{if } \text{final\_attendance\_pct} \ge 75.0 \end{cases}$$

### 1.2 The Critical Distinction: Prediction vs. Calculation
A common mistake in beginner attendance projects is to merely calculate the current percentage:

$$\text{current\_pct} = \frac{\text{classes\_attended}}{\text{classes\_conducted}} \times 100$$

While informative, this is a backward-looking arithmetic calculation. It does not predict future behavioral outcome. For example:
- A student currently at 82% who has missed 4 recent consecutive lectures and whose trajectory is steeply declining may easily fall below 75% by semester end.
- A student currently at 72% who has attended the last 10 consecutive lectures and whose engagement is improving may easily recover above 75%.

The ML model's task is to estimate the **probability of the final outcome** based on behavioral indicators observed up to the prediction checkpoint.

---

## 2. Preventing Data Leakage (Synthetic Simulation Design)

Target leakage occurs when information about the future outcome is inadvertently exposed to the model during training.

### 2.1 Two-Stage Simulation Process
To guarantee zero data leakage during synthetic dataset generation (`ml/generate_dataset.py`):

1. **Stage 1: Past Attendance Simulation (Up to Checkpoint)**
   - Total planned semester classes ($N_{\text{total}} \in [60, 90]$) are divided into conducted so far ($N_{\text{conducted}}$) and remaining ($N_{\text{remaining}}$).
   - Each student is assigned a latent propensity. Classes conducted are simulated step-by-step with momentum drift.
   - We record:
     - `classes_conducted`
     - `classes_attended`
     - `current_attendance_pct`
     - `previous_attendance_trend` (`declining`, `stable`, `improving`)
     - `consecutive_absences_recent`
     - `participation_score`

2. **Stage 2: Independent Future Outcome Simulation**
   - Future attendance across the $N_{\text{remaining}}$ classes is simulated as a separate stochastic trial, influenced by latent propensity, recent trend drift, and absence inertia.
   - We compute:
     $$\text{final\_classes\_attended} = \text{classes\_attended} + \text{future\_classes\_attended}$$
     $$\text{final\_attendance\_pct} = \frac{\text{final\_classes\_attended}}{N_{\text{conducted}} + N_{\text{remaining}}} \times 100$$
     $$\text{attendance\_risk} = 1 \text{ if } \text{final\_attendance\_pct} < 75.0 \text{ else } 0$$

3. **Feature Isolation**:
   At model training and inference time, `future_classes_attended` and `final_attendance_pct` are strictly excluded from the feature matrix $X$. The pipeline receives **only** features that would be physically available on the day an advisor queries the system.

---

## 3. Feature Engineering & Preprocessing

The model uses 7 features:

| Feature Name | Type | Physical Meaning | Range / Values |
|--------------|------|------------------|----------------|
| `classes_conducted` | Numerical (int) | Lectures held up to the prediction date | $5 \le N \le 150$ |
| `classes_attended` | Numerical (int) | Lectures attended by the student | $0 \le A \le \text{conducted}$ |
| `classes_remaining` | Numerical (int) | Lectures remaining in semester | $1 \le R \le 100$ |
| `current_attendance_pct`| Numerical (float) | Current historical attendance rate | $0.0\% \le P \le 100.0\%$ |
| `consecutive_absences_recent` | Numerical (int) | Missed sessions immediately preceding checkpoint | $0 \le C \le 10$ |
| `participation_score` | Numerical (float) | Quiz / assignment engagement rating | $1.0 \le S \le 10.0$ |
| `previous_attendance_trend` | Categorical (str) | Trajectory over recent 2-3 weeks | `declining`, `stable`, `improving` |

### Preprocessing via `ColumnTransformer`
To prevent leakage across validation folds and test splits, preprocessing is encapsulated in a scikit-learn `ColumnTransformer`:
- **Numerical Features**: Scaled using `StandardScaler` (zero mean, unit variance).
- **Categorical Feature (`previous_attendance_trend`)**: Encoded using `OneHotEncoder(categories=[["declining", "stable", "improving"]], handle_unknown="ignore", sparse_output=False)`.

---

## 4. Model Selection & Cross-Validation

### 4.1 Algorithms Evaluated
1. **Baseline Model: Logistic Regression**
   - Linear classification with L2 regularization (`solver="lbfgs"`, `max_iter=1000`).
   - `class_weight="balanced"` to account for the minority at-risk class (~23%).
   - Strengths: Ultra-lightweight (~3.8 KB), sub-millisecond inference, directly interpretable odds ratios, calibrated probabilities.
2. **Challenger Model: Decision Tree Classifier (max_depth=4)**
   - Single shallow decision tree (`max_depth=4`, `min_samples_leaf=5`, `class_weight="balanced"`).
   - No heavy ensembles (no Random Forest).
   - Strengths: Compact (~6.1 KB), human-interpretable rule paths, captures non-linear attendance inflection points.

### 4.2 Validation Strategy
- **Stratified Split**: 80% train (2,000 students), 20% test (500 students), preserving the 23% at-risk class balance in both partitions.
- **Cross-Validation**: 5-Fold Stratified K-Fold performed strictly on the training set.

---

## 5. Evaluation Metrics & Trade-offs

In academic early-warning systems, raw accuracy is a misleading metric. Because ~77% of students are safe, a naive model that always predicts "Safe" would achieve 77% accuracy while failing 100% of at-risk students.

### Why Recall on the Positive Class (At Risk) is Paramount
- **False Negative (FN)**: An at-risk student is predicted "Safe". No intervention is triggered. The student is debarred from final exams or loses course credit. **Cost: Very High.**
- **False Positive (FP)**: A safe student is predicted "At Risk". An advisor sends a check-in inquiry or study tip. **Cost: Low.**

Therefore, we optimize for **Recall on Class 1**, while maintaining sufficient **Precision** to avoid advisor notification fatigue:

$$\text{Recall} = \frac{TP}{TP + FN}, \quad \text{Precision} = \frac{TP}{TP + FP}, \quad F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Held-Out Test Set Performance (500 Students)

| Metric | Logistic Regression (Linear) | Decision Tree (Compact d=4) |
|--------|------------------------------|-----------------------------|
| **Accuracy** | 95.80% | **96.20%** |
| **Recall (At Risk)** | 93.91% | **95.65%** |
| **Precision (At Risk)** | 88.52% | **88.71%** |
| **F1-Score** | 0.9114 | **0.9205** |
| **ROC-AUC** | **0.9942** | 0.9816 |
| **Artifact Size** | **3.8 KB** | **6.1 KB** |

The shallow Decision Tree achieves **95.65% recall** on at-risk students while remaining tiny (6.1 KB), with zero heavy ensemble overhead.

---

## 6. Model Serialization & Inference

The fitted `ColumnTransformer` and the selected lightweight classifier are bundled as a single unified pipeline:

```python
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", classifier)
])
joblib.dump(pipeline, "artifacts/attendance_pipeline.joblib")
```

At prediction time:
1. Raw feature JSON is validated by Pydantic.
2. The pipeline handles scaling and one-hot encoding internally.
3. Probability is obtained via `pipeline.predict_proba(df)[:, 1]`.

---

## 7. Limitations & Ethical Considerations

1. **Synthetic Data Disclaimer**: Data was generated via simulation. Real-world validation on actual university attendance requires institutional review, IRB approval, and FERPA/data protection compliance.
2. **Association vs. Causality**: High feature importance (e.g. recent consecutive absences) describes a strong statistical association, not an intrinsic personality trait. Absences may be caused by illness, caregiving responsibilities, or financial hurdles.
3. **Intervention Use Only**: This software is designed strictly as a supportive advising tool to help students recover, never as an automated disciplinary or punitive filter.

