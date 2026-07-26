"""
Retail Sales Data — Cleaning, Processing & Visualization
==========================================================
A beginner-friendly, end-to-end data cleaning + EDA project.

Steps:
1. Load raw data
2. Inspect for missing values, duplicates, outliers
3. Clean the data
4. Engineer a couple of useful columns
5. Generate visual insights (saved as PNGs)
6. Export a cleaned dataset + a text summary report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

RAW_PATH = "data/raw_sales_data.csv"
CLEAN_PATH = "data/cleaned_sales_data.csv"
VISUALS_DIR = "visuals"
REPORT_PATH = "reports/data_cleaning_report.md"

report_lines = []


def log(line=""):
    """Print to console and also collect for the markdown report."""
    print(line)
    report_lines.append(line)


# ---------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------
df = pd.read_csv(RAW_PATH, parse_dates=["OrderDate"])

log("# Data Cleaning & Visualization Report\n")
log(f"**Raw dataset shape:** {df.shape[0]} rows × {df.shape[1]} columns\n")

# ---------------------------------------------------------------
# 2. INSPECT
# ---------------------------------------------------------------
log("## 1. Initial Data Quality Check\n")

missing = df.isna().sum()
missing = missing[missing > 0]
log("**Missing values per column:**\n")
log("```")
log(missing.to_string())
log("```\n")

dupe_count = df.duplicated().sum()
log(f"**Exact duplicate rows found:** {dupe_count}\n")

# ---------------------------------------------------------------
# 3. CLEAN
# ---------------------------------------------------------------
log("## 2. Cleaning Steps Applied\n")

# 3a. Remove exact duplicates
before = len(df)
df = df.drop_duplicates()
log(f"- Removed **{before - len(df)}** duplicate rows.")

# 3b. Standardize text columns (Region had mixed casing e.g. 'north' / 'SOUTH')
df["Region"] = df["Region"].str.strip().str.title()
df["Category"] = df["Category"].str.strip().str.title()
log("- Standardized text casing in `Region` and `Category` (e.g. 'north' -> 'North').")

# 3c. Handle missing values
# Region & PaymentMethod: categorical -> fill with 'Unknown' (don't invent a real answer)
df["Region"] = df["Region"].fillna("Unknown")
df["PaymentMethod"] = df["PaymentMethod"].fillna("Unknown")
log("- Filled missing `Region` / `PaymentMethod` with 'Unknown' (categorical, can't be guessed safely).")

# UnitsSold, UnitPrice, CustomerAge: numeric -> fill with column median (robust to outliers)
for col in ["UnitsSold", "UnitPrice", "CustomerAge"]:
    median_val = df[col].median()
    n_missing = df[col].isna().sum()
    df[col] = df[col].fillna(median_val)
    log(f"- Filled {n_missing} missing `{col}` values with median ({median_val:.1f}).")

# 3d. Fix outliers using the IQR method
def cap_outliers_iqr(series, factor=1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    return series.clip(lower=lower, upper=upper), lower, upper

for col in ["UnitsSold", "CustomerAge"]:
    capped, lo, hi = cap_outliers_iqr(df[col])
    n_affected = ((df[col] < lo) | (df[col] > hi)).sum()
    df[col] = capped
    log(f"- Capped {n_affected} outliers in `{col}` to the IQR range [{lo:.1f}, {hi:.1f}].")

# 3e. Recompute Revenue now that UnitsSold/UnitPrice are clean (also fixes the NaNs that were there)
df["Revenue"] = (df["UnitsSold"] * df["UnitPrice"]).round(2)
log("- Recomputed `Revenue` = UnitsSold × UnitPrice so it's consistent after cleaning.")

# 3f. Feature engineering
df["Month"] = df["OrderDate"].dt.month_name()
df["Quarter"] = df["OrderDate"].dt.to_period("Q").astype(str)
log("- Added `Month` and `Quarter` columns for time-based analysis.")

# Save cleaned data
df.to_csv(CLEAN_PATH, index=False)
log(f"\n**Cleaned dataset shape:** {df.shape[0]} rows × {df.shape[1]} columns")
log(f"**Saved to:** `{CLEAN_PATH}`\n")

# ---------------------------------------------------------------
# 4. VISUALIZE
# ---------------------------------------------------------------
log("## 3. Key Visual Insights\n")

# Chart 1: Revenue by Category
plt.figure(figsize=(8, 5))
rev_by_cat = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
sns.barplot(x=rev_by_cat.values, y=rev_by_cat.index, hue=rev_by_cat.index,
            palette="viridis", legend=False)
plt.title("Total Revenue by Category")
plt.xlabel("Revenue")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/01_revenue_by_category.png")
plt.close()

# Chart 2: Monthly revenue trend
plt.figure(figsize=(10, 5))
month_order = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]
rev_by_month = df.groupby("Month")["Revenue"].sum().reindex(month_order)
sns.lineplot(x=rev_by_month.index, y=rev_by_month.values, marker="o", color="#2E86AB")
plt.title("Monthly Revenue Trend (2024)")
plt.xticks(rotation=45)
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/02_monthly_revenue_trend.png")
plt.close()

# Chart 3: Revenue by Region
plt.figure(figsize=(7, 5))
rev_by_region = df.groupby("Region")["Revenue"].sum().sort_values(ascending=False)
sns.barplot(x=rev_by_region.index, y=rev_by_region.values, hue=rev_by_region.index,
            palette="mako", legend=False)
plt.title("Total Revenue by Region")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/03_revenue_by_region.png")
plt.close()

# Chart 4: Payment method distribution
plt.figure(figsize=(6, 6))
payment_counts = df["PaymentMethod"].value_counts()
plt.pie(payment_counts.values, labels=payment_counts.index, autopct="%1.1f%%",
        colors=sns.color_palette("pastel"))
plt.title("Orders by Payment Method")
plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/04_payment_method_share.png")
plt.close()

# Chart 5: Customer age distribution (post-cleaning, no more 150/200 outliers)
plt.figure(figsize=(8, 5))
sns.histplot(df["CustomerAge"], bins=20, kde=True, color="#F18F01")
plt.title("Customer Age Distribution (Cleaned)")
plt.xlabel("Age")
plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/05_customer_age_distribution.png")
plt.close()

# Chart 6: Correlation heatmap
plt.figure(figsize=(6, 5))
numeric_cols = df[["UnitsSold", "UnitPrice", "CustomerAge", "Revenue"]]
sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Numeric Features")
plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/06_correlation_heatmap.png")
plt.close()

log(f"- Saved 6 charts to the `{VISUALS_DIR}/` folder:")
log("  1. Revenue by Category (bar)")
log("  2. Monthly Revenue Trend (line)")
log("  3. Revenue by Region (bar)")
log("  4. Payment Method Share (pie)")
log("  5. Customer Age Distribution (histogram)")
log("  6. Correlation Heatmap")

# ---------------------------------------------------------------
# 5. KEY FINDINGS (storytelling)
# ---------------------------------------------------------------
log("\n## 4. Key Findings\n")
top_cat = rev_by_cat.idxmax()
top_month = rev_by_month.idxmax()
top_region = rev_by_region.idxmax()
top_payment = payment_counts.idxmax()

log(f"- **{top_cat}** is the top revenue-generating category (Rs. {rev_by_cat.max():,.0f}).")
log(f"- **{top_month}** was the strongest sales month (Rs. {rev_by_month.max():,.0f}).")
log(f"- **{top_region}** region leads in total revenue.")
log(f"- **{top_payment}** is the most common payment method "
    f"({payment_counts.max()} of {payment_counts.sum()} orders, "
    f"{payment_counts.max()/payment_counts.sum()*100:.1f}%).")
log(f"- Average order revenue: Rs. {df['Revenue'].mean():,.2f}")

import os
os.makedirs("reports", exist_ok=True)
with open(REPORT_PATH, "w") as f:
    f.write("\n".join(report_lines))

print(f"\nDone. Report written to {REPORT_PATH}")
