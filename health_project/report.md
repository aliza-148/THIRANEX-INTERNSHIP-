# Patient Health Records: Diabetes Risk Analysis
### End-to-End Data Science Project — Healthcare Domain

---

## 1. Objective

Analyze a patient records dataset to understand the demographic, clinical,
and lifestyle factors associated with diabetes diagnosis, and build a
predictive model that flags patients at elevated risk.

## 2. Dataset

- **Source:** Synthetically generated patient records (2,000 patients), built
  with clinically plausible distributions and relationships (age, BMI, blood
  pressure, cholesterol, and glucose scaled and correlated the way they are
  in real epidemiological data). Synthetic data was used to avoid any
  real-world patient privacy concerns while preserving realistic patterns
  for the pipeline.
- **Features:** age, gender, BMI, smoking status, exercise level, family
  history of diabetes, systolic blood pressure, cholesterol, glucose level.
- **Target:** `diabetes_diagnosis` (1 = diagnosed, 0 = not diagnosed)
- **Class balance:** 34.9% positive (diabetes), 65.1% negative — a realistic,
  moderately imbalanced clinical prevalence.
- **Data quality:** ~2% missing values injected into cholesterol and blood
  pressure columns (mirroring real EHR extracts) and imputed with the column
  median before modeling.

## 3. Exploratory Data Analysis — Key Observations

| # | Visual | Finding |
|---|--------|---------|
| 1 | `01_target_balance.png` | Diabetes prevalence in the cohort is ~35%, giving a workable but imbalanced classification problem. |
| 2 | `02_age_distribution.png` | Diabetes diagnoses skew toward older patients; the diagnosed group's age distribution is shifted right relative to the non-diagnosed group. |
| 3 | `03_bmi_vs_glucose.png` | Patients with both **higher BMI and higher glucose** cluster heavily in the diabetes-positive group — the two features jointly separate classes better than either alone. |
| 4 | `04_correlation_heatmap.png` | Glucose has the strongest correlation with diagnosis, followed by BMI and age. Blood pressure and cholesterol are only weakly correlated with the outcome directly, but still contribute via interactions. |
| 5 | `05_lifestyle_risk_factors.png` | Diabetes rate is visibly higher among smokers, patients with **low exercise levels**, and those with a family history — confirming known clinical risk factors. |
| 6 | `06_bmi_by_gender.png` | BMI distributions are broadly similar across genders, with a slightly wider spread in one group; gender alone is not a strong differentiator. |

**Takeaway:** Glucose level and BMI are the dominant signals, with age,
family history, and low physical activity acting as reinforcing risk
factors — consistent with established diabetes epidemiology.

## 4. Predictive Modeling

Two classifiers were trained on a 75/25 train-test split (stratified) to
predict diabetes diagnosis from the 9 engineered features:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.72 | 0.64 | 0.47 | 0.54 | **0.78** |
| Random Forest | 0.70 | 0.62 | 0.37 | 0.46 | 0.76 |

- **Logistic Regression slightly outperforms Random Forest** on this dataset,
  suggesting the underlying risk relationship is close to linear/additive in
  the log-odds — matching how the data was generated and how real clinical
  risk scores (e.g. Framingham-style scores) are often structured.
- **ROC-AUC of 0.76–0.78** indicates good — though not perfect —
  discriminative ability. This is realistic: diabetes risk in real
  populations also depends on factors not captured here (diet detail,
  medication history, genetics beyond family history, HbA1c trends over
  time).
- **Recall is the weaker metric (37–47%)**, meaning a meaningful share of
  true diabetes cases are missed. In a clinical screening context, this
  matters more than raw accuracy — missing a diabetic patient is costlier
  than a false alarm. This suggests the classification threshold should be
  lowered (favor recall over precision) if this were deployed as a
  screening tool, and/or that class-imbalance handling (e.g. class weights,
  SMOTE) would help in a follow-up iteration.
- **Feature importance (Random Forest):** glucose and BMI dominate, followed
  by age, cholesterol, and systolic blood pressure. Lifestyle flags
  (smoking, exercise, family history) contribute smaller but non-trivial
  importance — see `09_feature_importance.png`.

## 5. Conclusions

1. **Glucose and BMI are the primary drivers** of diabetes risk in this
   cohort, consistent with clinical literature — any screening tool should
   prioritize collecting these reliably.
2. **Lifestyle factors compound clinical risk**: low exercise, smoking, and
   family history all independently raise diabetes prevalence, reinforcing
   the value of lifestyle-focused preventive care alongside clinical
   monitoring.
3. **A simple logistic regression model performs competitively** with a more
   complex Random Forest (AUC 0.78 vs 0.76), which is good news for
   deployability — simpler models are easier to explain to clinicians and
   patients (a meaningful factor in healthcare ML adoption).
4. **Recall should be prioritized over precision** in any real screening
   deployment, since false negatives (missed diabetics) carry a higher
   clinical cost than false positives (an extra confirmatory test).
5. **Next steps** for a production version: use real, de-identified EHR
   data; add HbA1c and longitudinal glucose trends; apply class-weighting
   or resampling to improve recall; and validate on an external patient
   population before any clinical use.

## 6. Ethical Note

This project uses **fully synthetic data** — no real patient records were
used or are included. Any production health-risk model would require
IRB/ethics approval, real de-identified data under appropriate governance,
and clinical validation before informing patient care decisions.
