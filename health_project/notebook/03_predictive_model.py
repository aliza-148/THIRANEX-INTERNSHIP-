"""
03_predictive_model.py
------------------------
Builds and evaluates classification models to predict diabetes diagnosis
from patient clinical & lifestyle data. Compares Logistic Regression vs
Random Forest, and saves evaluation visuals + a metrics summary.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

sns.set_theme(style="whitegrid")
VIS = "/home/claude/health_project/visuals"
OUT = "/home/claude/health_project/outputs"

df = pd.read_csv("/home/claude/health_project/data/patient_records.csv")

# ---- Cleaning ----
for col in ["cholesterol", "systolic_bp"]:
    df[col] = df[col].fillna(df[col].median())

# ---- Feature engineering ----
df["smoker_flag"] = (df["smoker"] == "Yes").astype(int)
df["family_history_flag"] = (df["family_history_diabetes"] == "Yes").astype(int)
df["gender_flag"] = (df["gender"] == "Male").astype(int)
exercise_map = {"Low": 0, "Moderate": 1, "High": 2}
df["exercise_score"] = df["exercise_level"].map(exercise_map)

features = [
    "age", "bmi", "systolic_bp", "cholesterol", "glucose",
    "smoker_flag", "family_history_flag", "gender_flag", "exercise_score",
]
X = df[features]
y = df["diabetes_diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- Model 1: Logistic Regression ----
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
y_prob_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

# ---- Model 2: Random Forest ----
rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

def get_metrics(y_true, y_pred, y_prob, name):
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }

results = pd.DataFrame([
    get_metrics(y_test, y_pred_lr, y_prob_lr, "Logistic Regression"),
    get_metrics(y_test, y_pred_rf, y_prob_rf, "Random Forest"),
])
results.to_csv(f"{OUT}/model_metrics.csv", index=False)
print(results.round(3))

# ---- Visualization: ROC curves ----
plt.figure(figsize=(6, 5.5))
for y_prob, name, color in [
    (y_prob_lr, "Logistic Regression", "#3d5a80"),
    (y_prob_rf, "Random Forest", "#e07a5f"),
]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})", color=color, linewidth=2)
plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{VIS}/07_roc_curve.png")
plt.close()

# ---- Visualization: Confusion matrix (best model = Random Forest) ----
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Diabetes", "Diabetes"],
            yticklabels=["No Diabetes", "Diabetes"])
plt.title("Confusion Matrix — Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{VIS}/08_confusion_matrix.png")
plt.close()

# ---- Visualization: Feature importance (Random Forest) ----
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)
plt.figure(figsize=(7, 5))
importances.plot(kind="barh", color="#81b29a")
plt.title("Feature Importance — Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{VIS}/09_feature_importance.png")
plt.close()

# save classification report text
with open(f"{OUT}/classification_report_rf.txt", "w") as f:
    f.write("Random Forest Classification Report\n")
    f.write("=" * 40 + "\n")
    f.write(classification_report(y_test, y_pred_rf, target_names=["No Diabetes", "Diabetes"]))

print("\nModeling complete. Outputs saved to:", OUT)
print("Visuals saved to:", VIS)
