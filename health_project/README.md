# Real-World Data Project: Health (Patient Records)

An end-to-end applied data science project on the **healthcare** domain:
diabetes risk analysis and prediction using patient records.

## Project Structure

```
health_project/
├── data/
│   └── patient_records.csv        # 2,000-patient synthetic dataset
├── notebook/
│   ├── 01_generate_data.py        # Step 1: generate the dataset
│   ├── 02_eda_visualizations.py   # Step 2: exploratory data analysis
│   └── 03_predictive_model.py     # Step 3: train & evaluate models
├── visuals/                       # 9 charts produced by the pipeline
│   ├── 01_target_balance.png
│   ├── 02_age_distribution.png
│   ├── 03_bmi_vs_glucose.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_lifestyle_risk_factors.png
│   ├── 06_bmi_by_gender.png
│   ├── 07_roc_curve.png
│   ├── 08_confusion_matrix.png
│   └── 09_feature_importance.png
├── outputs/
│   ├── model_metrics.csv          # accuracy/precision/recall/F1/AUC comparison
│   └── classification_report_rf.txt
├── report.md                      # Full write-up: findings & conclusions
└── README.md                      # This file
```

## How to Reproduce

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
cd notebook
python3 01_generate_data.py          # creates data/patient_records.csv
python3 02_eda_visualizations.py     # creates visuals/01-06
python3 03_predictive_model.py       # creates visuals/07-09 + outputs/
```

## Summary

- **Dataset:** 2,000 synthetic patient records (age, BMI, blood pressure,
  cholesterol, glucose, smoking, exercise, family history) with a diabetes
  diagnosis target (~35% prevalence).
- **Analysis:** Full EDA covering class balance, demographic patterns,
  clinical correlations, and lifestyle risk factors.
- **Modeling:** Logistic Regression vs Random Forest classifiers, evaluated
  on accuracy, precision, recall, F1, and ROC-AUC. Logistic Regression
  reached the best ROC-AUC (0.78).
- **Findings & conclusions:** see [`report.md`](report.md) for the full
  write-up, including clinical interpretation and recommended next steps.

## Note on Data

Real patient data was **not** used, for privacy and ethical reasons. The
dataset is synthetically generated using clinically plausible statistical
relationships (see `notebook/01_generate_data.py`), so the analysis
techniques and pipeline transfer directly to a real EHR extract.
