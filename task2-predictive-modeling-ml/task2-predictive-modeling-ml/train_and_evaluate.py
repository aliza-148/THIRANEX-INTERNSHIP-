"""
Predictive Modeling Using Machine Learning — Customer Churn Prediction
========================================================================
Build, train, and evaluate classification models to predict whether a
customer will churn (leave the service).

Steps:
1. Load & clean the raw dataset
2. Encode categorical features
3. Split into train/test sets
4. Train 3 models: Logistic Regression, Decision Tree, Random Forest
5. Evaluate with accuracy, precision/recall/F1, confusion matrix, ROC-AUC
6. Visualize performance and feature importance
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

RAW_PATH = "data/customer_churn_raw.csv"
CLEAN_PATH = "data/customer_churn_cleaned.csv"
VISUALS_DIR = "visuals"
REPORT_PATH = "reports/model_evaluation_report.md"

report_lines = []


def log(line=""):
    print(line)
    report_lines.append(line)


# ---------------------------------------------------------------
# 1. LOAD & CLEAN
# ---------------------------------------------------------------
df = pd.read_csv(RAW_PATH)

log("# Predictive Modeling Report — Customer Churn Prediction\n")
log(f"**Raw dataset shape:** {df.shape[0]} rows x {df.shape[1]} columns\n")

log("## 1. Data Preparation\n")

missing_before = df.isna().sum().sum()
df["Age"] = df["Age"].fillna(df["Age"].median())
df["MonthlyCharges"] = df["MonthlyCharges"].fillna(df["MonthlyCharges"].median())
df["TechSupport"] = df["TechSupport"].fillna("No")
log(f"- Filled {missing_before} missing values (median for numeric, 'No' for TechSupport).")

df = df.drop_duplicates()
df.to_csv(CLEAN_PATH, index=False)
log(f"- Cleaned dataset saved to `{CLEAN_PATH}`.")

cat_cols = ["Contract", "InternetService", "TechSupport", "PaymentMethod"]
df_model = df.drop(columns=["CustomerID"]).copy()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    encoders[col] = le
log(f"- Label-encoded categorical columns: {', '.join(cat_cols)}.")

X = df_model.drop(columns=["Churn"])
y = df_model["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
log(f"- Train/test split: {len(X_train)} training rows, {len(X_test)} test rows (75/25 split).")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 2. TRAIN MODELS
# ---------------------------------------------------------------
log("\n## 2. Models Trained\n")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
}

results = {}
for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    results[name] = {
        "model": model, "y_pred": y_pred, "y_proba": y_proba,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "fpr": fpr, "tpr": tpr, "auc": roc_auc
    }

    log(f"### {name}")
    log(f"- Accuracy: **{acc:.3f}**  |  Precision: {prec:.3f}  |  Recall: {rec:.3f}  |  F1: {f1:.3f}  |  ROC-AUC: {roc_auc:.3f}\n")

# ---------------------------------------------------------------
# 3. VISUALIZE
# ---------------------------------------------------------------
log("## 3. Visualizations\n")

plt.figure(figsize=(7, 5))
model_names = list(results.keys())
accs = [results[m]["accuracy"] for m in model_names]
sns.barplot(x=model_names, y=accs, hue=model_names, palette="viridis", legend=False)
plt.ylim(0, 1)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
for i, v in enumerate(accs):
    plt.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/01_accuracy_comparison.png")
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/02_confusion_matrices.png")
plt.close()

plt.figure(figsize=(7, 6))
for name, res in results.items():
    plt.plot(res["fpr"], res["tpr"], label=f"{name} (AUC = {res['auc']:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Model Comparison")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/03_roc_curves.png")
plt.close()

rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8, 5))
sns.barplot(x=importances.values, y=importances.index, hue=importances.index,
            palette="mako", legend=False)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/04_feature_importance.png")
plt.close()

log("- Saved 4 charts to the `visuals/` folder:")
log("  1. Accuracy Comparison (bar)")
log("  2. Confusion Matrices - all 3 models")
log("  3. ROC Curves - all 3 models overlaid")
log("  4. Feature Importance (Random Forest)")

# ---------------------------------------------------------------
# 4. FINDINGS
# ---------------------------------------------------------------
log("\n## 4. Key Findings\n")
best_model = max(results, key=lambda m: results[m]["accuracy"])
best_auc_model = max(results, key=lambda m: results[m]["auc"])
top_feature = importances.index[0]

log(f"- **{best_model}** had the highest accuracy ({results[best_model]['accuracy']:.3f}).")
log(f"- **{best_auc_model}** had the best ROC-AUC score ({results[best_auc_model]['auc']:.3f}), "
    f"meaning it separates churn vs non-churn customers best overall.")
log(f"- **{top_feature}** was the most important predictor of churn according to the Random Forest model.")
log(f"- Class balance in the data: {(y==0).sum()} No-Churn vs {(y==1).sum()} Churn customers "
    f"(~{y.mean()*100:.1f}% churn rate).")

os.makedirs("reports", exist_ok=True)
with open(REPORT_PATH, "w") as f:
    f.write("\n".join(report_lines))

print(f"\nDone. Report written to {REPORT_PATH}")
