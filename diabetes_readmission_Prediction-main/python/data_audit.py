"""
data_audit.py — Python-side data quality audit for diabetic_data.csv.
Complements R/01_data_audit.R with dtype inspection, outlier detection,
and cardinality summary. Run from project root.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

RAW_PATH = "data/raw/diabetic_data.csv"
OUT_RESULTS = "outputs/results"
OUT_FIGURES = "outputs/figures"

os.makedirs(OUT_RESULTS, exist_ok=True)
os.makedirs(OUT_FIGURES, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(RAW_PATH, na_values=["?", "", "NA"])
print(f"Raw shape: {df.shape[0]:,} rows x {df.shape[1]} cols")

# ── 1. Data types ─────────────────────────────────────────────────────────────
print("\n=== DTYPES ===")
print(df.dtypes.value_counts())

# ── 2. Missing values ─────────────────────────────────────────────────────────
missing = (
    df.isnull()
    .sum()
    .rename("n_missing")
    .to_frame()
    .assign(pct_missing=lambda x: (x["n_missing"] / len(df) * 100).round(1))
    .query("n_missing > 0")
    .sort_values("pct_missing", ascending=False)
)
print("\n=== MISSING VALUES ===")
print(missing.to_string())
missing.to_csv(f"{OUT_RESULTS}/py_missing_audit.csv")

# ── 3. Target variable ────────────────────────────────────────────────────────
print("\n=== TARGET: readmitted ===")
target_counts = df["readmitted"].value_counts(dropna=False)
target_pct = (target_counts / len(df) * 100).round(1)
print(pd.concat([target_counts, target_pct.rename("pct")], axis=1).to_string())

# ── 4. Duplicate patients ─────────────────────────────────────────────────────
n_unique = df["patient_nbr"].nunique()
n_total = len(df)
n_duped = (df["patient_nbr"].value_counts() > 1).sum()
print(f"\n=== PATIENTS ===")
print(f"Total encounters : {n_total:,}")
print(f"Unique patients  : {n_unique:,}")
print(f"Multi-encounter  : {n_duped:,}")

# ── 5. Cardinality of object columns ─────────────────────────────────────────
obj_cols = df.select_dtypes("object").columns.tolist()
cardinality = (
    df[obj_cols]
    .nunique()
    .rename("unique_vals")
    .sort_values(ascending=False)
    .to_frame()
)
print("\n=== CARDINALITY (object columns) ===")
print(cardinality.to_string())
cardinality.to_csv(f"{OUT_RESULTS}/py_cardinality.csv")

# ── 6. Numeric summary ────────────────────────────────────────────────────────
num_cols = df.select_dtypes(np.number).columns.tolist()
num_summary = df[num_cols].describe().T.round(2)
print("\n=== NUMERIC SUMMARY ===")
print(num_summary.to_string())
num_summary.to_csv(f"{OUT_RESULTS}/py_numeric_summary.csv")

# ── 7. Outlier detection (IQR method) ────────────────────────────────────────
key_numeric = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]
key_numeric = [c for c in key_numeric if c in df.columns]

outlier_rows = []
for col in key_numeric:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[col] < lo) | (df[col] > hi)).sum()
    outlier_rows.append({
        "feature": col, "q1": q1, "q3": q3,
        "iqr_lo": round(lo, 2), "iqr_hi": round(hi, 2),
        "n_outliers": int(n_out),
        "pct_outliers": round(n_out / len(df) * 100, 1),
    })

outliers_df = pd.DataFrame(outlier_rows).sort_values("n_outliers", ascending=False)
print("\n=== OUTLIERS (IQR method) ===")
print(outliers_df.to_string(index=False))
outliers_df.to_csv(f"{OUT_RESULTS}/py_outliers.csv", index=False)

# ── 8. Deceased / hospice flag ────────────────────────────────────────────────
dead_ids = {11, 13, 14, 19, 20, 21}
n_dead = df["discharge_disposition_id"].isin(dead_ids).sum()
print(f"\nDeceased/hospice encounters (removed in preprocessing): "
      f"{n_dead:,} ({n_dead / len(df) * 100:.1f}%)")

# ── 9. Figure: missing value bar chart ────────────────────────────────────────
if not missing.empty:
    fig, ax = plt.subplots(figsize=(8, 4))
    top = missing.head(10)
    ax.barh(top.index[::-1], top["pct_missing"][::-1], color="#D85A30", alpha=0.85)
    for i, (_, row) in enumerate(top[::-1].iterrows()):
        ax.text(row["pct_missing"] + 0.5, i,
                f'{row["pct_missing"]}%', va="center", fontsize=9)
    ax.set_xlabel("% missing")
    ax.set_title("missing value rate — top columns (raw data)")
    ax.set_xlim(0, 110)
    plt.tight_layout()
    plt.savefig(f"{OUT_FIGURES}/00_missing_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nsaved 00_missing_bar.png")

# ── 10. Figure: numeric feature box plots ────────────────────────────────────
plot_cols = [c for c in key_numeric if c in df.columns]
n_plots = len(plot_cols)
cols_per_row = 4
n_rows = (n_plots + cols_per_row - 1) // cols_per_row

fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(14, n_rows * 3))
axes = axes.flatten()
for i, col in enumerate(plot_cols):
    axes[i].hist(df[col].dropna(), bins=30, color="#7F77DD", alpha=0.8, edgecolor="white")
    axes[i].set_title(col, fontsize=9)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("numeric feature distributions (raw data)", y=1.01, fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT_FIGURES}/00b_numeric_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved 00b_numeric_distributions.png")

# ── Summary ───────────────────────────────────────────────────────────────────
summary = pd.Series({
    "total_rows":                int(len(df)),
    "total_cols":                int(df.shape[1]),
    "unique_patients":           int(n_unique),
    "multi_encounter_patients":  int(n_duped),
    "deceased_hospice":          int(n_dead),
    "cols_with_missing":         int(len(missing)),
    "pct_readmit_lt30":          round(float((df["readmitted"] == "<30").mean() * 100), 1),
    "pct_readmit_gt30":          round(float((df["readmitted"] == ">30").mean() * 100), 1),
    "pct_not_readmit":           round(float((df["readmitted"] == "NO").mean()  * 100), 1),
}, name="value")

summary.to_frame().to_csv(f"{OUT_RESULTS}/py_data_summary.csv")

print("\n=== SUMMARY ===")
for k, v in summary.items():
    print(f"  {k:<35} {v}")

print("\nAudit complete. CSVs -> outputs/results/  |  Figures -> outputs/figures/")
