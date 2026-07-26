# 📊 Retail Sales — Data Cleaning & Visualization Project

An end-to-end data preprocessing and exploratory data analysis (EDA) project.
A raw, messy retail sales dataset is cleaned, processed, and turned into
visual insights using Python.

## 🎯 Objective
Learn and demonstrate the core data analyst workflow:
1. Identify data quality issues (missing values, duplicates, outliers, inconsistent formatting)
2. Clean and process the data
3. Visualize key trends and patterns
4. Summarize findings ("storytelling with data")

## 🗂️ Project Structure
```
data-cleaning-project/
├── data/
│   ├── raw_sales_data.csv          # original, messy dataset
│   └── cleaned_sales_data.csv      # cleaned, analysis-ready dataset
├── notebooks/
│   └── data_cleaning_and_visualization.ipynb   # full walkthrough notebook
├── visuals/
│   ├── 01_revenue_by_category.png
│   ├── 02_monthly_revenue_trend.png
│   ├── 03_revenue_by_region.png
│   ├── 04_payment_method_share.png
│   ├── 05_customer_age_distribution.png
│   └── 06_correlation_heatmap.png
├── reports/
│   └── data_cleaning_report.md     # auto-generated cleaning summary
├── clean_and_analyze.py            # standalone script version (run it directly)
├── requirements.txt
└── README.md
```

## 🧹 Data Cleaning Steps
- **Duplicates:** removed exact duplicate rows
- **Missing values:** categorical columns (`Region`, `PaymentMethod`) filled with
  `"Unknown"`; numeric columns (`UnitsSold`, `UnitPrice`, `CustomerAge`) filled
  with the column median
- **Outliers:** detected and capped using the IQR (Interquartile Range) method
  on `UnitsSold` and `CustomerAge`
- **Inconsistent formatting:** standardized text casing (e.g. `north` / `SOUTH` → `North`)
- **Feature engineering:** derived `Month` and `Quarter` columns from `OrderDate`

## 📈 Visual Insights
The project generates 6 charts covering revenue by category/region, the
monthly revenue trend, payment method distribution, customer age
distribution, and a correlation heatmap of the numeric features.

## 🔑 Key Findings
- **Grocery** is the top revenue-generating category
- **June** was the strongest sales month of 2024
- **North** region leads in total revenue
- **Credit Card** is the most common payment method (~33% of orders)

## ⚙️ How to Run
```bash
git clone https://github.com/<your-username>/data-cleaning-project.git
cd data-cleaning-project
pip install -r requirements.txt
python clean_and_analyze.py
```
Or open `notebooks/data_cleaning_and_visualization.ipynb` in Jupyter / VS Code
to see the step-by-step walkthrough with inline charts.

## 🛠️ Tools Used
Python · Pandas · NumPy · Matplotlib · Seaborn · Jupyter Notebook

---
*Note: this uses a synthetically generated retail sales dataset (created for
this project) so the cleaning steps could showcase realistic messy-data
scenarios — missing values, duplicates, outliers, and inconsistent text.*
