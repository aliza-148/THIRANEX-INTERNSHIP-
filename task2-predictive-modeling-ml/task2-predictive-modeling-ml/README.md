# 🤖 Predictive Modeling Using Machine Learning — Customer Churn Prediction

A supervised learning project that predicts whether a customer will churn
(leave the service), comparing three classic ML algorithms and evaluating
them properly.

## 🎯 Objective
1. Prepare and encode raw customer data for machine learning
2. Train multiple classification models
3. Evaluate and compare them using proper metrics (not just accuracy)
4. Visualize performance (confusion matrices, ROC curves) and interpret results

## 🗂️ Project Structure
```
predictive-modeling-project/
├── data/
│   ├── customer_churn_raw.csv          # original dataset
│   └── customer_churn_cleaned.csv      # cleaned, encoded-ready dataset
├── notebooks/
│   └── predictive_modeling_churn.ipynb # full walkthrough notebook
├── visuals/
│   ├── 01_accuracy_comparison.png
│   ├── 02_confusion_matrices.png
│   ├── 03_roc_curves.png
│   └── 04_feature_importance.png
├── reports/
│   └── model_evaluation_report.md      # auto-generated evaluation summary
├── train_and_evaluate.py               # standalone script version
├── requirements.txt
└── README.md
```

## 🧠 Models Trained
- **Logistic Regression** — simple, interpretable linear baseline
- **Decision Tree** — captures non-linear patterns, easy to interpret
- **Random Forest** — ensemble of trees, usually the strongest of the three

## 📏 Evaluation Metrics
Each model is evaluated with:
- **Accuracy** — overall % correct
- **Precision / Recall / F1** — because in churn prediction, missing an
  actual churner (recall) is usually costlier than a false alarm
- **Confusion Matrix** — visual breakdown of correct vs incorrect predictions
- **ROC Curve & AUC** — how well the model separates the two classes across
  all thresholds, not just the default 0.5 cutoff

## 🔑 Key Findings
- Logistic Regression achieved the strongest accuracy and ROC-AUC in this run
- `Tenure_Months` was the most important churn predictor (Random Forest)
- The dataset has a ~43% churn rate — reasonably balanced

## ⚙️ How to Run
```bash
pip install -r requirements.txt
python train_and_evaluate.py
```
Or open `notebooks/predictive_modeling_churn.ipynb` in Jupyter / VS Code /
Google Colab for the full step-by-step walkthrough with inline charts.

## 🛠️ Tools Used
Python · Pandas · NumPy · scikit-learn · Matplotlib · Seaborn

---
*Note: this uses a synthetically generated customer churn dataset (created
for this project) with a realistic, learnable relationship between features
and churn, so the models produce meaningful (not random) results.*
