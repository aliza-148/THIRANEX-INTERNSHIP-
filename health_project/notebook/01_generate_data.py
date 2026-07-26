"""
01_generate_data.py
--------------------
Generates a realistic SYNTHETIC patient records dataset for a diabetes-risk
analysis project. Real patient data is never used here (privacy/ethics) -
instead we simulate patients using clinically plausible distributions and
relationships, so the downstream analysis and modeling pipeline mirrors
what you'd do with a real EHR extract (e.g. from Kaggle's Pima Indians
Diabetes dataset or a hospital data warehouse).

Output: ../data/patient_records.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 2000  # number of patients

# ---- Demographics ----
age = np.clip(np.random.normal(48, 16, N), 18, 90).round(0).astype(int)
gender = np.random.choice(["Male", "Female"], size=N, p=[0.48, 0.52])

# ---- Lifestyle factors ----
smoker = np.random.choice(["Yes", "No"], size=N, p=[0.21, 0.79])
exercise_level = np.random.choice(
    ["Low", "Moderate", "High"], size=N, p=[0.35, 0.45, 0.20]
)
family_history = np.random.choice(["Yes", "No"], size=N, p=[0.30, 0.70])

# ---- Vitals & labs (correlated with age / lifestyle for realism) ----
bmi = np.clip(
    np.random.normal(27, 5, N)
    + (exercise_level == "Low") * 2.0
    - (exercise_level == "High") * 1.5,
    15, 55
).round(1)

systolic_bp = np.clip(
    np.random.normal(120, 15, N) + (age - 48) * 0.3 + (bmi - 27) * 0.5, 90, 200
).round(0).astype(int)

cholesterol = np.clip(
    np.random.normal(200, 35, N) + (age - 48) * 0.4 + (bmi - 27) * 1.2, 120, 350
).round(0).astype(int)

glucose = np.clip(
    np.random.normal(100, 20, N)
    + (bmi - 27) * 1.5
    + (age - 48) * 0.3
    + (family_history == "Yes") * 10
    + (smoker == "Yes") * 5,
    70, 300
).round(0).astype(int)

# ---- Target: Diabetes diagnosis (simulated via logistic risk model) ----
risk_score = (
    -11.5
    + 0.04 * age
    + 0.09 * bmi
    + 0.03 * glucose
    + 0.01 * systolic_bp
    + 0.008 * cholesterol
    + 0.9 * (family_history == "Yes")
    + 0.5 * (smoker == "Yes")
    - 0.4 * (exercise_level == "High")
)
prob = 1 / (1 + np.exp(-risk_score))
diabetes = (np.random.rand(N) < prob).astype(int)

df = pd.DataFrame({
    "patient_id": [f"P{1000+i}" for i in range(N)],
    "age": age,
    "gender": gender,
    "bmi": bmi,
    "smoker": smoker,
    "exercise_level": exercise_level,
    "family_history_diabetes": family_history,
    "systolic_bp": systolic_bp,
    "cholesterol": cholesterol,
    "glucose": glucose,
    "diabetes_diagnosis": diabetes,
})

# introduce a few realistic missing values (as in real EHR data)
for col in ["cholesterol", "systolic_bp"]:
    idx = np.random.choice(df.index, size=int(0.02 * N), replace=False)
    df.loc[idx, col] = np.nan

out_path = "/home/claude/health_project/data/patient_records.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} records to {out_path}")
print(df.head())
print("\nDiabetes prevalence: {:.1f}%".format(df["diabetes_diagnosis"].mean() * 100))
