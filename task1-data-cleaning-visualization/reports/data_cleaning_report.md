# Data Cleaning & Visualization Report

**Raw dataset shape:** 515 rows × 9 columns

## 1. Initial Data Quality Check

**Missing values per column:**

```
Region           57
UnitsSold        31
UnitPrice        31
PaymentMethod    31
CustomerAge      32
Revenue          30
```

**Exact duplicate rows found:** 15

## 2. Cleaning Steps Applied

- Removed **15** duplicate rows.
- Standardized text casing in `Region` and `Category` (e.g. 'north' -> 'North').
- Filled missing `Region` / `PaymentMethod` with 'Unknown' (categorical, can't be guessed safely).
- Filled 30 missing `UnitsSold` values with median (26.0).
- Filled 30 missing `UnitPrice` values with median (263.1).
- Filled 30 missing `CustomerAge` values with median (45.0).
- Capped 8 outliers in `UnitsSold` to the IQR range [-18.0, 70.0].
- Capped 6 outliers in `CustomerAge` to the IQR range [-2.6, 94.4].
- Recomputed `Revenue` = UnitsSold × UnitPrice so it's consistent after cleaning.
- Added `Month` and `Quarter` columns for time-based analysis.

**Cleaned dataset shape:** 500 rows × 11 columns
**Saved to:** `data/cleaned_sales_data.csv`

## 3. Key Visual Insights

- Saved 6 charts to the `visuals/` folder:
  1. Revenue by Category (bar)
  2. Monthly Revenue Trend (line)
  3. Revenue by Region (bar)
  4. Payment Method Share (pie)
  5. Customer Age Distribution (histogram)
  6. Correlation Heatmap

## 4. Key Findings

- **Grocery** is the top revenue-generating category (Rs. 737,735).
- **June** was the strongest sales month (Rs. 331,428).
- **North** region leads in total revenue.
- **Credit Card** is the most common payment method (163 of 500 orders, 32.6%).
- Average order revenue: Rs. 6,626.89