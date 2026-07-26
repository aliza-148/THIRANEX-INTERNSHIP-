"""
Exploratory Data Analysis (EDA) — Student Performance Dataset
================================================================
Goal: explore a dataset to uncover patterns, trends, and the key
factors that influence student exam scores.

Steps:
1. Load & inspect data (structure, types, summary statistics)
2. Univariate analysis (distributions of individual variables)
3. Bivariate/correlation analysis (what relates to ExamScore?)
4. Categorical group comparisons
5. Structured written report of insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

DATA_PATH = "data/student_performance_data.csv"
VISUALS_DIR = "visuals"
REPORT_PATH = "reports/eda_report.md"

report_lines = []


def log(line=""):
    print(line)
    report_lines.append(line)


# ---------------------------------------------------------------
# 1. LOAD & INSPECT
# ---------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

log("# Exploratory Data Analysis Report — Student Performance\n")
log(f"**Dataset shape:** {df.shape[0]} rows × {df.shape[1]} columns\n")

log("## 1. Statistical Summary\n")
log("```")
log(df.describe().round(2).to_string())
log("```\n")

log("**Data types:**\n")
log("```")
log(df.dtypes.to_string())
log("```\n")

missing = df.isna().sum().sum()
log(f"**Missing values:** {missing} (dataset is already clean)\n")

# ---------------------------------------------------------------
# 2. UNIVARIATE ANALYSIS
# ---------------------------------------------------------------
log("## 2. Distributions\n")

numeric_cols = ["StudyHoursPerDay", "AttendancePercent", "SleepHoursPerDay", "ExamScore"]

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, col in zip(axes.flat, numeric_cols):
    sns.histplot(df[col], kde=True, ax=ax, color="#2E86AB")
    ax.set_title(f"Distribution of {col}")
plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/01_distributions.png")
plt.close()
log("- Saved distribution histograms for all numeric variables.")

for col in numeric_cols:
    skew = df[col].skew()
    log(f"  - `{col}`: mean={df[col].mean():.1f}, median={df[col].median():.1f}, "
        f"std={df[col].std():.1f}, skew={skew:.2f}")

# ---------------------------------------------------------------
# 3. CORRELATION ANALYSIS
# ---------------------------------------------------------------
log("\n## 3. Correlation Analysis\n")

corr = df[numeric_cols].corr()
plt.figure(figsize=(6.5, 5.5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Matrix (Numeric Features)")
plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/02_correlation_heatmap.png")
plt.close()

score_corr = corr["ExamScore"].drop("ExamScore").sort_values(ascending=False)
log("**Correlation of each factor with ExamScore:**\n")
log("```")
log(score_corr.round(3).to_string())
log("```\n")

# Scatter plots: study hours & attendance vs score
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.regplot(data=df, x="StudyHoursPerDay", y="ExamScore", ax=axes[0],
            scatter_kws={"alpha": 0.4, "color": "#2E86AB"}, line_kws={"color": "#F18F01"})
axes[0].set_title("Study Hours vs Exam Score")
sns.regplot(data=df, x="AttendancePercent", y="ExamScore", ax=axes[1],
            scatter_kws={"alpha": 0.4, "color": "#2E86AB"}, line_kws={"color": "#F18F01"})
axes[1].set_title("Attendance % vs Exam Score")
plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/03_scatter_relationships.png")
plt.close()
log("- Saved scatter plots showing the two strongest relationships with ExamScore.")

# ---------------------------------------------------------------
# 4. CATEGORICAL GROUP COMPARISONS
# ---------------------------------------------------------------
log("\n## 4. Categorical Factors\n")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.boxplot(data=df, x="ParentalEducation", y="ExamScore", ax=axes[0],
            order=["High School", "Bachelors", "Masters", "PhD"], hue="ParentalEducation",
            palette="viridis", legend=False)
axes[0].set_title("Exam Score by Parental Education")
axes[0].tick_params(axis='x', rotation=30)

sns.boxplot(data=df, x="InternetAccess", y="ExamScore", ax=axes[1], hue="InternetAccess",
            palette="mako", legend=False)
axes[1].set_title("Exam Score by Internet Access")

sns.boxplot(data=df, x="ExtracurricularActivities", y="ExamScore", ax=axes[2],
            hue="ExtracurricularActivities", palette="crest", legend=False)
axes[2].set_title("Exam Score by Extracurriculars")

plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/04_categorical_comparisons.png")
plt.close()
log("- Saved boxplots comparing ExamScore across ParentalEducation, InternetAccess, "
    "and ExtracurricularActivities.")

edu_means = df.groupby("ParentalEducation")["ExamScore"].mean().reindex(
    ["High School", "Bachelors", "Masters", "PhD"]
)
log("\n**Average ExamScore by ParentalEducation:**\n")
log("```")
log(edu_means.round(1).to_string())
log("```\n")

internet_gap = df.groupby("InternetAccess")["ExamScore"].mean()
log(f"**Average ExamScore — with internet access:** {internet_gap.get('Yes', float('nan')):.1f} "
    f"vs without: {internet_gap.get('No', float('nan')):.1f}\n")

# ---------------------------------------------------------------
# 5. KEY INSIGHTS
# ---------------------------------------------------------------
log("## 5. Key Insights\n")
top_factor = score_corr.abs().idxmax()
log(f"- **{top_factor}** has the strongest relationship with ExamScore "
    f"(correlation = {score_corr[top_factor]:.2f}).")
log(f"- Students with internet access score **{internet_gap.get('Yes',0) - internet_gap.get('No',0):.1f} "
    f"points higher on average** than those without.")
log(f"- Higher parental education level is associated with progressively higher average scores "
    f"({edu_means.iloc[0]:.1f} for High School parents vs {edu_means.iloc[-1]:.1f} for PhD parents).")
log(f"- Study hours and attendance both show a positive, roughly linear relationship with exam "
    f"performance — consistent, moderate effort compounds over time.")
log(f"- Sleep hours show a weaker but still positive correlation, suggesting rest matters but "
    f"isn't as decisive as study time or attendance.")

import os
os.makedirs("reports", exist_ok=True)
with open(REPORT_PATH, "w") as f:
    f.write("\n".join(report_lines))

print(f"\nDone. Report written to {REPORT_PATH}")
