# Predictive Modeling Report — Customer Churn Prediction

**Dataset shape:** 1000 rows × 10 columns

**Target variable:** `Churn` (1 = customer left, 0 = customer stayed)
**Class balance:** {1: 0.503, 0: 0.497}

## 1. Preprocessing

- Label-encoded categorical columns: ['Contract', 'InternetService', 'TechSupport', 'PaymentMethod']
- Train/test split: 800 train rows, 200 test rows (80/20, stratified)
- Scaled numeric features with StandardScaler (needed for Logistic Regression)

## 2. Models Trained

### Logistic Regression
- Accuracy: 0.710
- Precision: 0.722
- Recall: 0.693
- F1 Score: 0.707
- ROC-AUC: 0.835

### Decision Tree
- Accuracy: 0.690
- Precision: 0.679
- Recall: 0.733
- F1 Score: 0.705
- ROC-AUC: 0.746

### Random Forest
- Accuracy: 0.760
- Precision: 0.773
- Recall: 0.743
- F1 Score: 0.758
- ROC-AUC: 0.838

## 3. Model Comparison

```
                     Accuracy  Precision  Recall     F1    AUC
Logistic Regression      0.71      0.722   0.693  0.707  0.835
Decision Tree            0.69      0.679   0.733  0.705  0.746
Random Forest            0.76      0.773   0.743  0.758  0.838
```

**Best performing model (by F1 score): Random Forest**

## 4. Visualizations

- Saved 4 charts to the `visuals/` folder:
  1. Confusion matrices (all 3 models)
  2. ROC curves (all 3 models compared)
  3. Model performance comparison (bar chart)
  4. Feature importance (Random Forest)

## 5. Key Findings

- **Random Forest** performed best overall (F1 = 0.758, AUC = 0.838).
- **Tenure_Months** is the most important predictor of churn according to Random Forest.
- All models scored an AUC above 0.75, meaning they're all meaningfully better than random guessing at separating churners from non-churners.