# Predictive Modeling Report — Customer Churn Prediction

**Raw dataset shape:** 1000 rows x 10 columns

## 1. Data Preparation

- Filled 90 missing values (median for numeric, 'No' for TechSupport).
- Cleaned dataset saved to `data/customer_churn_cleaned.csv`.
- Label-encoded categorical columns: Contract, InternetService, TechSupport, PaymentMethod.
- Train/test split: 750 training rows, 250 test rows (75/25 split).

## 2. Models Trained

### Logistic Regression
- Accuracy: **0.708**  |  Precision: 0.667  |  Recall: 0.623  |  F1: 0.644  |  ROC-AUC: 0.767

### Decision Tree
- Accuracy: **0.668**  |  Precision: 0.635  |  Recall: 0.509  |  F1: 0.565  |  ROC-AUC: 0.679

### Random Forest
- Accuracy: **0.684**  |  Precision: 0.645  |  Recall: 0.566  |  F1: 0.603  |  ROC-AUC: 0.723

## 3. Visualizations

- Saved 4 charts to the `visuals/` folder:
  1. Accuracy Comparison (bar)
  2. Confusion Matrices - all 3 models
  3. ROC Curves - all 3 models overlaid
  4. Feature Importance (Random Forest)

## 4. Key Findings

- **Logistic Regression** had the highest accuracy (0.708).
- **Logistic Regression** had the best ROC-AUC score (0.767), meaning it separates churn vs non-churn customers best overall.
- **Tenure_Months** was the most important predictor of churn according to the Random Forest model.
- Class balance in the data: 574 No-Churn vs 426 Churn customers (~42.6% churn rate).