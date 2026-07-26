"""
02_eda_visualizations.py
-------------------------
Exploratory data analysis on the patient records dataset.
Generates visualizations saved to ../visuals/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 110

df = pd.read_csv("/home/claude/health_project/data/patient_records.csv")

# Basic cleaning: fill missing numeric values with column median
for col in ["cholesterol", "systolic_bp"]:
    df[col] = df[col].fillna(df[col].median())

VIS = "/home/claude/health_project/visuals"

# 1. Target class balance
plt.figure(figsize=(5, 4))
ax = sns.countplot(x="diabetes_diagnosis", data=df)
ax.set_xticklabels(["No Diabetes", "Diabetes"])
plt.title("Diabetes Diagnosis Distribution")
plt.xlabel("")
plt.ylabel("Number of Patients")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width()/2, p.get_height()),
                ha="center", va="bottom")
plt.tight_layout()
plt.savefig(f"{VIS}/01_target_balance.png")
plt.close()

# 2. Age distribution by diagnosis
plt.figure(figsize=(7, 4.5))
sns.histplot(data=df, x="age", hue="diabetes_diagnosis", multiple="stack", bins=20,
             palette=["#8fd3c8", "#e07a5f"])
plt.title("Age Distribution by Diabetes Diagnosis")
plt.xlabel("Age")
plt.ylabel("Count")
plt.legend(title="Diagnosis", labels=["Diabetes", "No Diabetes"])
plt.tight_layout()
plt.savefig(f"{VIS}/02_age_distribution.png")
plt.close()

# 3. BMI vs Glucose scatter colored by diagnosis
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="bmi", y="glucose", hue="diabetes_diagnosis",
                 palette=["#8fd3c8", "#e07a5f"], alpha=0.6)
plt.title("BMI vs Glucose Level by Diagnosis")
plt.xlabel("BMI")
plt.ylabel("Glucose (mg/dL)")
plt.legend(title="Diagnosis", labels=["No Diabetes", "Diabetes"])
plt.tight_layout()
plt.savefig(f"{VIS}/03_bmi_vs_glucose.png")
plt.close()

# 4. Correlation heatmap of numeric features
numeric_cols = ["age", "bmi", "systolic_bp", "cholesterol", "glucose", "diabetes_diagnosis"]
plt.figure(figsize=(7, 6))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix of Clinical Features")
plt.tight_layout()
plt.savefig(f"{VIS}/04_correlation_heatmap.png")
plt.close()

# 5. Diabetes prevalence by lifestyle factors
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, col, title in zip(
    axes,
    ["smoker", "exercise_level", "family_history_diabetes"],
    ["Smoking Status", "Exercise Level", "Family History"],
):
    rate = df.groupby(col)["diabetes_diagnosis"].mean() * 100
    rate.plot(kind="bar", ax=ax, color="#e07a5f")
    ax.set_title(f"Diabetes Rate by {title}")
    ax.set_ylabel("Diabetes Rate (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig(f"{VIS}/05_lifestyle_risk_factors.png")
plt.close()

# 6. BMI distribution by gender
plt.figure(figsize=(6, 4.5))
sns.boxplot(data=df, x="gender", y="bmi", palette="Set2")
plt.title("BMI Distribution by Gender")
plt.tight_layout()
plt.savefig(f"{VIS}/06_bmi_by_gender.png")
plt.close()

print("EDA visualizations saved to", VIS)
print("\nSummary statistics:\n", df[numeric_cols].describe().round(1))
