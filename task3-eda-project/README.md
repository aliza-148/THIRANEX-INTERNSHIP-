# 🔍 Exploratory Data Analysis (EDA) — Student Performance

Analyzing a student performance dataset to uncover patterns, trends, and
the key factors that influence exam scores — using statistical summaries,
correlation analysis, and visualizations.

## 🎯 Objective
1. Summarize the dataset statistically (mean, median, spread, skew)
2. Explore distributions of individual variables
3. Identify which factors correlate most strongly with exam performance
4. Compare performance across categorical groups
5. Present findings in a structured, readable report

## 🗂️ Project Structure
```
task3-eda-project/
├── data/
│   └── student_performance_data.csv   # dataset (600 students)
├── notebooks/
│   └── eda_analysis.ipynb             # full walkthrough notebook
├── visuals/
│   ├── 01_distributions.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_scatter_relationships.png
│   └── 04_categorical_comparisons.png
├── reports/
│   └── eda_report.md                  # auto-generated structured insights report
├── analyze_eda.py                     # standalone script version
├── requirements.txt
└── README.md
```

## 📊 Dataset
600 students with: study hours/day, attendance %, sleep hours/day, parental
education level, extracurricular participation, internet access, gender,
and exam score (target of interest).

## 🔑 Key Insights
- **Study hours** has the strongest relationship with exam score (correlation ≈ 0.69) — by far the biggest lever a student controls directly.
- Students with **internet access** score ~5 points higher on average.
- Higher **parental education** level is associated with progressively higher average scores (High School → PhD).
- **Attendance** and **sleep** both show positive but weaker correlations — supportive, not decisive, factors.
- Overall: consistent study habits matter more than any single background factor in this data.

## ⚙️ How to Run
```bash
pip install -r requirements.txt
python analyze_eda.py
```
Or open `notebooks/eda_analysis.ipynb` in Jupyter / Google Colab / VS Code for
the full step-by-step walkthrough with inline charts.

## 🛠️ Tools Used
Python · Pandas · NumPy · Matplotlib · Seaborn

---
*Note: this uses a synthetically generated student dataset (created for this
project) with realistic performance drivers (study time, attendance, sleep,
background factors) built into it, so the correlations reflect genuine,
interpretable patterns.*
